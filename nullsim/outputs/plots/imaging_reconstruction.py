"""Optional aperture-synthesis image reconstruction output."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
from collections.abc import Iterable

import matplotlib  # noqa: F401  # backend choice managed by styles module
import numpy as np

from nullsim.outputs.styles import _ensure_headless_backend  # noqa: F401

from matplotlib import pyplot as plt
from matplotlib.colors import AsinhNorm, Normalize, TwoSlopeNorm
from numpy.typing import NDArray

from nullsim.config.schema import ImagingReconstructionOutputConfig
from nullsim.outputs import RunCollection, RunRecord
from nullsim.outputs.plots._chip_field_inputs import (
    dark_indices as _dark_indices,
    design_wavelength_um as _design_wavelength_um,
    find_component as _find_component,
    maybe_find_component as _maybe_find_component,
    mode_diameters_m as _mode_diameters_m,
    mode_positions_enu_m as _mode_positions_enu_m,
    stellar_amplitudes as _stellar_amplitudes,
    subapertures_from_positions,
    unitary_matrices_for_imaging as _unitary_matrices_for_imaging,
    wavelength_grid_um as _wavelength_grid_um,
)
from nullsim.physics.imaging import (
    MAUNAKEA_LAT_RAD,
    MAS_PER_RAD,
    SkyScene,
    SubAperture,
    ReconstructedImage,
    build_response_matrix,
    create_scene,
    reconstruct_nnls,
    reconstruct_snr_map,
    simulate_observation,
    simulate_phase_chopped_observation,
)
from nullsim.physics.reflected_light import reflected_light_contrast
from nullsim.pipeline import SimulationState, register_plot


@dataclass(frozen=True)
class _PlanetView:
    separation_mas: float
    position_angle_deg: float
    contrast: float
    label: str
    flux_model: str = "fixed_contrast"
    radius_earth: float | None = None
    semi_major_axis_au: float | None = None
    geometric_albedo: float | None = None
    phase_angle_deg: float | None = None
    contrast_source: str = "fixed_contrast"


@dataclass(frozen=True)
class _ExozodiGlowView:
    image: NDArray[np.float64]
    total_contrast: float
    target_rate_hz_per_port: float
    unit_rate_hz_per_port: float
    effective_level_zodi: float
    inclination_deg: float
    position_angle_deg: float
    inner_radius_au: float
    outer_radius_au: float
    radial_power: float


@dataclass(frozen=True)
class _ImagingInputs:
    subapertures: tuple[SubAperture, ...]
    amplitudes: NDArray[np.complex128]
    unitary: NDArray[np.complex128]
    unitary_wavelength_pairs: tuple[tuple[NDArray[np.complex128], float], ...]
    dark_port_indices: NDArray[np.intp]
    wavelength_m: float
    stellar_photon_rate_hz: float
    stellar_rate_hz_by_wavelength: NDArray[np.float64]
    null_depth: float
    dark_count_rate_hz_per_port: float
    dark_count_rate_hz_per_port_by_wavelength: NDArray[np.float64]
    sky_rate_hz_per_port: float
    sky_rate_hz_per_port_by_wavelength: NDArray[np.float64]
    integration_time_s: float
    eps_systematic: float
    with_fringe_tracking: bool
    servo_bw_hz: float
    mkid_max_count_rate_hz: float | None
    planets: tuple[_PlanetView, ...]
    exozodi_payload: dict[str, Any] | None
    star_distance_pc: float | None
    latitude_rad: float
    declination_rad: float
    hour_angles_rad: NDArray[np.float64]
    roll_angles_rad: NDArray[np.float64] | None
    rotation_mode: str


@register_plot
def imaging_reconstruction(
    collection: RunCollection,
    output_dir: Path,
    formats: Iterable[str],
) -> tuple[Path, ...]:
    """Write optional truth/SNR/beam imaging artifacts."""

    if not collection.records:
        return ()

    config = (
        collection.config.output.imaging_reconstruction
        or ImagingReconstructionOutputConfig()
    )
    records = collection.records[: config.max_records]
    plot_dir = output_dir / "plots"
    table_dir = output_dir / "tables"
    array_dir = output_dir / "imaging"
    plot_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    if config.save_arrays:
        array_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    summaries: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        inputs = _build_imaging_inputs(record, config)
        fov_mas = _resolve_fov_mas(config, inputs.planets)
        point_source_scene = create_scene(
            n_pix=int(config.n_pix),
            fov_mas=fov_mas,
            planets=tuple(
                (p.separation_mas, p.position_angle_deg, p.contrast)
                for p in inputs.planets
            ),
        )
        scene, exozodi_glow = _scene_with_exozodi_glow(
            scene=point_source_scene,
            inputs=inputs,
            config=config,
        )
        integration_time_per_step_s = (
            inputs.integration_time_s / max(len(inputs.hour_angles_rad), 1)
        )
        observation_fn = _observation_simulator(config)
        observation = observation_fn(
            scene=scene,
            subapertures=inputs.subapertures,
            amplitudes=inputs.amplitudes,
            unitary=inputs.unitary,
            dark_port_indices=inputs.dark_port_indices,
            hour_angles_rad=inputs.hour_angles_rad,
            declination_rad=inputs.declination_rad,
            wavelength_m=inputs.wavelength_m,
            stellar_photon_rate_hz=inputs.stellar_photon_rate_hz,
            null_depth=inputs.null_depth,
            integration_time_per_step_s=integration_time_per_step_s,
            dark_count_rate_hz_per_port=inputs.dark_count_rate_hz_per_port,
            sky_rate_hz_per_port=inputs.sky_rate_hz_per_port,
            seed=_record_seed(config, record, index),
            latitude_rad=inputs.latitude_rad,
            null_instability=inputs.eps_systematic,
            ft_servo_bandwidth_hz=inputs.servo_bw_hz,
            mkid_max_count_rate_hz=inputs.mkid_max_count_rate_hz,
            unitaries_by_wavelength=inputs.unitary_wavelength_pairs,
            roll_angles_rad=inputs.roll_angles_rad,
            poisson=bool(config.poisson),
            keep_response_matrix=config.chop_mode == "chip_phase",
            stellar_photon_rate_hz_by_wavelength=inputs.stellar_rate_hz_by_wavelength,
        )
        reconstruction = reconstruct_snr_map(
            data=observation,
            subapertures=inputs.subapertures,
            amplitudes=inputs.amplitudes,
            unitary=inputs.unitary,
            declination_rad=inputs.declination_rad,
            wavelength_m=inputs.wavelength_m,
            l_coords=scene.l_coords,
            m_coords=scene.m_coords,
            latitude_rad=inputs.latitude_rad,
            null_instability=inputs.eps_systematic,
            ft_servo_bandwidth_hz=inputs.servo_bw_hz,
            integration_time_per_step_s=integration_time_per_step_s,
            unitaries_by_wavelength=inputs.unitary_wavelength_pairs,
            roll_angles_rad=inputs.roll_angles_rad,
        )
        nnls = None
        nnls_image: NDArray[np.float64] | None = None
        nnls_snr_image: NDArray[np.float64] | None = None
        nnls_observation = observation
        if config.compute_nnls:
            if (
                config.nnls_wavelength_mode == "resolved"
                and len(inputs.unitary_wavelength_pairs) > 1
            ):
                nnls_observation = observation_fn(
                    scene=scene,
                    subapertures=inputs.subapertures,
                    amplitudes=inputs.amplitudes,
                    unitary=inputs.unitary,
                    dark_port_indices=inputs.dark_port_indices,
                    hour_angles_rad=inputs.hour_angles_rad,
                    declination_rad=inputs.declination_rad,
                    wavelength_m=inputs.wavelength_m,
                    stellar_photon_rate_hz=inputs.stellar_photon_rate_hz,
                    null_depth=inputs.null_depth,
                    integration_time_per_step_s=integration_time_per_step_s,
                    dark_count_rate_hz_per_port=inputs.dark_count_rate_hz_per_port,
                    sky_rate_hz_per_port=inputs.sky_rate_hz_per_port,
                    seed=_record_seed(config, record, index),
                    latitude_rad=inputs.latitude_rad,
                    null_instability=inputs.eps_systematic,
                    ft_servo_bandwidth_hz=inputs.servo_bw_hz,
                    mkid_max_count_rate_hz=inputs.mkid_max_count_rate_hz,
                    unitaries_by_wavelength=inputs.unitary_wavelength_pairs,
                    roll_angles_rad=inputs.roll_angles_rad,
                    poisson=bool(config.poisson),
                    keep_response_matrix=True,
                    wavelength_resolved=True,
                    stellar_photon_rate_hz_by_wavelength=(
                        inputs.stellar_rate_hz_by_wavelength
                    ),
                    dark_count_rate_hz_per_port_by_wavelength=(
                        inputs.dark_count_rate_hz_per_port_by_wavelength
                    ),
                    sky_rate_hz_per_port_by_wavelength=(
                        inputs.sky_rate_hz_per_port_by_wavelength
                    ),
                )
            nnls = reconstruct_nnls(
                data=nnls_observation,
                subapertures=inputs.subapertures,
                amplitudes=inputs.amplitudes,
                unitary=inputs.unitary,
                declination_rad=inputs.declination_rad,
                wavelength_m=inputs.wavelength_m,
                l_coords=scene.l_coords,
                m_coords=scene.m_coords,
                latitude_rad=inputs.latitude_rad,
                stellar_photon_rate_hz=inputs.stellar_photon_rate_hz,
                integration_time_per_step_s=integration_time_per_step_s,
                unitaries_by_wavelength=inputs.unitary_wavelength_pairs,
                roll_angles_rad=inputs.roll_angles_rad,
                candidate_indices=_nnls_candidate_indices(
                    snr_map=reconstruction.image,
                    planets=inputs.planets,
                    fov_mas=fov_mas,
                    n_pix=int(config.n_pix),
                    max_candidates=(
                        int(config.nnls_candidate_pixels)
                        if config.chop_mode == "chip_phase"
                        else None
                    ),
                ),
                regularization_fraction=(
                    float(config.nnls_regularization_fraction)
                    if config.chop_mode == "chip_phase"
                    else 0.0
                ),
            )
            nnls_image = nnls.image
            nnls_snr_image = _nnls_component_snr_image(
                nnls_image=nnls_image,
                observation=nnls_observation,
                inputs=inputs,
                scene=scene,
            )

        stem = (
            "imaging_reconstruction"
            if len(records) == 1
            else f"imaging_reconstruction_{index:03d}"
        )
        fig = _plot_truth_snr_beam(
            truth_image=scene.image,
            snr_map=reconstruction.image,
            matched_filter_beam=reconstruction.synthesized_beam,
            fov_mas=fov_mas,
            planets=inputs.planets,
            title=_plot_title(record),
        )
        for fmt in formats:
            path = plot_dir / f"{stem}.{fmt}"
            fig.savefig(path)
            written.append(path)
        plt.close(fig)

        if nnls_image is not None:
            nnls_fig = _plot_truth_nnls(
                truth_image=scene.image,
                nnls_image=nnls_image,
                snr_map=nnls_snr_image,
                fov_mas=fov_mas,
                planets=inputs.planets,
                title=f"{_plot_title(record)} NNLS",
            )
            for fmt in formats:
                path = plot_dir / f"{stem}_nnls.{fmt}"
                nnls_fig.savefig(path)
                written.append(path)
            plt.close(nnls_fig)

        if config.save_arrays:
            arrays: dict[str, NDArray[Any]] = {
                "truth_image": scene.image,
                "point_source_truth_image": point_source_scene.image,
                "snr_map": reconstruction.image,
                "matched_filter_beam": reconstruction.synthesized_beam,
                "synthesized_beam": reconstruction.synthesized_beam,
                "measurements": observation.measurements,
                "hour_angles_rad": inputs.hour_angles_rad,
                "dark_port_indices": inputs.dark_port_indices,
            }
            if exozodi_glow is not None:
                arrays["exozodi_image"] = exozodi_glow.image
            if inputs.roll_angles_rad is not None:
                arrays["roll_angles_rad"] = inputs.roll_angles_rad
            if config.chop_mode != "none":
                arrays["chop_mode"] = np.asarray(config.chop_mode)
            if nnls_image is not None:
                arrays["nnls_image"] = nnls_image
                if nnls_snr_image is not None:
                    arrays["nnls_snr_map"] = nnls_snr_image
                arrays["nnls_measurements"] = nnls_observation.measurements
                if nnls_observation.wavelengths_m is not None:
                    arrays["nnls_wavelengths_m"] = nnls_observation.wavelengths_m
            array_path = array_dir / f"{stem}.npz"
            np.savez_compressed(array_path, **arrays)
            written.append(array_path)

        summaries.append(
            _summary_payload(
                stem=stem,
                record=record,
                inputs=inputs,
                config=config,
                fov_mas=fov_mas,
                snr_map=(
                    nnls_snr_image if nnls_snr_image is not None else reconstruction.image
                ),
                nnls_image=nnls_image,
                nnls=nnls,
                nnls_observation=nnls_observation if nnls_image is not None else None,
                planets=inputs.planets,
                exozodi_glow=exozodi_glow,
                integration_time_per_step_s=integration_time_per_step_s,
                skipped_records=max(len(collection.records) - len(records), 0),
            )
        )

    summary_path = table_dir / "imaging_reconstruction.json"
    summary_path.write_text(
        json.dumps({"records": summaries}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    written.append(summary_path)
    return tuple(written)


def _observation_simulator(config: ImagingReconstructionOutputConfig):
    if config.chop_mode == "chip_phase":
        return simulate_phase_chopped_observation
    return simulate_observation


def _build_imaging_inputs(
    record: RunRecord,
    config: ImagingReconstructionOutputConfig,
) -> _ImagingInputs:
    state = record.result.state
    chip_payload = _find_component(state, "chip.kernel_mzi_mesh")
    u_stack = np.asarray(chip_payload.get("u_stack"), dtype=np.complex128)
    if u_stack.ndim != 3 or u_stack.shape[0] != u_stack.shape[1]:
        raise ValueError(
            "imaging_reconstruction requires chip.kernel_mzi_mesh to publish "
            f"u_stack with shape (M, M, n_wavelength); got {u_stack.shape}."
        )
    n_modes = int(u_stack.shape[0])
    wavelengths_um = _wavelength_grid_um(state, chip_payload, u_stack.shape[2])
    wavelength_indices = _selected_wavelength_indices(
        n_wavelengths=len(wavelengths_um),
        max_wavelengths=config.max_wavelengths,
    )
    design_index = int(
        np.argmin(np.abs(np.asarray(wavelengths_um) - _design_wavelength_um(chip_payload)))
    )
    sensitivity = _maybe_find_component(state, "sensitivity.detection_curve")
    base_dark_indices = _dark_indices(chip_payload, sensitivity, n_modes)
    unitary, unitary_wavelength_pairs, dark_indices = _unitary_matrices_for_imaging(
        state=state,
        chip_payload=chip_payload,
        u_stack=u_stack,
        base_dark_indices=base_dark_indices,
        wavelength_indices=wavelength_indices,
        design_index=design_index,
        wavelengths_um=wavelengths_um,
    )
    positions = _mode_positions_enu_m(state, n_modes)
    diameters = _mode_diameters_m(state, n_modes, sensitivity)
    subapertures = subapertures_from_positions(positions, diameters)
    amplitudes = _stellar_amplitudes(state, n_modes, sensitivity)
    performance = _maybe_find_component(state, "performance.standard")
    detector = _find_detector_component(state)

    integration_time_s = _positive_float(
        performance.get("integration_time_s")
        if performance is not None
        else detector.get("integration_time_s"),
        "integration_time_s",
    )
    stellar_rate_hz = _positive_float(
        _physical_stellar_rate_hz(performance, detector),
        "stellar_photon_rate_hz",
    )
    null_depth = _nonnegative_float(
        _first_present(
            performance,
            ("null_depth_effective", "null_depth_residual"),
            fallback=None,
        )
        if performance is not None
        else detector.get("null_depth_raw"),
        "null_depth",
    )

    latitude_deg = (
        float(config.latitude_deg)
        if config.latitude_deg is not None
        else _site_latitude_deg(record)
    )
    hour_angles_rad, roll_angles_rad = _sampling_angles_rad(
        center_hour_angle_h=float(record.config.observation.hour_angle_h),
        integration_time_s=integration_time_s,
        config=config,
    )
    dark_rate_hz_per_port = _nonnegative_float(
        _first_present(performance, ("dark_rate_hz_per_port",), 0.0),
        "dark_rate_hz_per_port",
    )
    sky_rate_hz_per_port = _nonnegative_float(
        _first_present(performance, ("sky_rate_hz_per_port",), 0.0),
        "sky_rate_hz_per_port",
    )
    star_distance_pc = (
        float(record.config.scene.star.distance_pc)
        if state.scene.star_distance_pc is None
        else float(state.scene.star_distance_pc)
    )
    exozodi_payload = None
    if config.include_exozodi_glow and star_distance_pc is not None:
        exozodi_payload = _maybe_find_component(state, "background.space_zodi")
    sky_rate_hz_per_port_for_imaging = _sky_rate_without_exozodi(
        sky_rate_hz_per_port,
        exozodi_payload,
    )
    with_fringe_tracking = bool(
        _first_present(performance, ("with_fringe_tracking",), True)
    )

    return _ImagingInputs(
        subapertures=subapertures,
        amplitudes=amplitudes,
        unitary=unitary,
        unitary_wavelength_pairs=unitary_wavelength_pairs,
        dark_port_indices=dark_indices,
        wavelength_m=float(wavelengths_um[design_index]) * 1.0e-6,
        stellar_photon_rate_hz=stellar_rate_hz,
        stellar_rate_hz_by_wavelength=_stellar_rates_by_wavelength(
            state=state,
            wavelength_indices=wavelength_indices,
            total_rate_hz=stellar_rate_hz,
        ),
        null_depth=null_depth,
        dark_count_rate_hz_per_port=dark_rate_hz_per_port,
        dark_count_rate_hz_per_port_by_wavelength=_uniform_rates_by_wavelength(
            total_rate_hz=dark_rate_hz_per_port,
            n_wavelengths=len(wavelength_indices),
        ),
        sky_rate_hz_per_port=sky_rate_hz_per_port_for_imaging,
        sky_rate_hz_per_port_by_wavelength=_background_rates_by_wavelength(
            state=state,
            wavelength_indices=wavelength_indices,
            total_rate_hz=sky_rate_hz_per_port_for_imaging,
            exclude_exozodi=exozodi_payload is not None,
        ),
        integration_time_s=integration_time_s,
        # The fractional rms of the leakage that performance.standard charged
        # as its systematic term (null instability and residual speckle
        # combined), not the bare sensitivity null_instability.
        eps_systematic=_nonnegative_float(
            _first_present(performance, ("eps_systematic_used",), 0.0),
            "eps_systematic_used",
        ),
        with_fringe_tracking=with_fringe_tracking,
        # Without fringe tracking the systematic is not averaged over servo
        # cycles: performance.standard charges (eps * L * t)^2, so imaging
        # draws the leakage scale with rms eps and no 1/sqrt(2 f t) factor.
        servo_bw_hz=(
            _nonnegative_float(
                _first_present(performance, ("servo_bw_hz",), 0.0),
                "servo_bw_hz",
            )
            if with_fringe_tracking
            else 0.0
        ),
        mkid_max_count_rate_hz=_detector_cap_hz(detector),
        planets=_planet_views(record, config),
        exozodi_payload=exozodi_payload,
        star_distance_pc=star_distance_pc,
        latitude_rad=np.radians(latitude_deg),
        declination_rad=np.radians(float(config.declination_deg)),
        hour_angles_rad=hour_angles_rad,
        roll_angles_rad=roll_angles_rad,
        rotation_mode=str(config.rotation_mode),
    )


def _physical_stellar_rate_hz(
    performance: dict[str, Any] | None,
    detector: dict[str, Any],
) -> float:
    """Stellar band total before detector saturation.

    A planet at contrast ``c`` delivers ``c`` times the star's physical rate,
    whatever the bright-port pixels can count, and the realized null is a
    fraction of that same physical rate. The capped readout total understates
    both by the saturation ratio (review Issue 9), so the uncapped total wins
    whenever it is published.
    """

    if performance is not None:
        return _first_present(
            performance,
            ("stellar_rate_hz_total_uncapped", "stellar_rate_hz_total"),
            None,
        )
    rates = detector.get("uncapped_port_rates", detector["port_rates"])
    return float(np.sum(np.asarray(rates, dtype=np.float64)))


def _scene_with_exozodi_glow(
    *,
    scene: SkyScene,
    inputs: _ImagingInputs,
    config: ImagingReconstructionOutputConfig,
) -> tuple[SkyScene, _ExozodiGlowView | None]:
    payload = inputs.exozodi_payload
    if payload is None:
        return scene, None
    if inputs.star_distance_pc is None or inputs.star_distance_pc <= 0.0:
        return scene, None
    target_rate_hz_per_port = _positive_exozodi_rate_hz_per_port(payload)
    if target_rate_hz_per_port <= 0.0:
        return scene, None

    inclination_deg = _payload_float(payload, "exozodi_inclination_deg", 0.0)
    template = _exozodi_template_image(
        scene=scene,
        distance_pc=float(inputs.star_distance_pc),
        inclination_deg=inclination_deg,
        position_angle_deg=float(config.exozodi_position_angle_deg),
        inner_radius_au=float(config.exozodi_inner_radius_au),
        outer_radius_au=float(config.exozodi_outer_radius_au),
        radial_power=float(config.exozodi_radial_power),
    )
    if float(np.sum(template)) <= 0.0:
        return scene, None

    unit_rate_hz_per_port = _exozodi_unit_rate_hz_per_port(
        template=template,
        scene=scene,
        inputs=inputs,
    )
    if unit_rate_hz_per_port <= 0.0:
        return scene, None
    total_contrast = target_rate_hz_per_port / unit_rate_hz_per_port
    exozodi_image = template * total_contrast
    glow = _ExozodiGlowView(
        image=exozodi_image,
        total_contrast=float(np.sum(exozodi_image)),
        target_rate_hz_per_port=float(target_rate_hz_per_port),
        unit_rate_hz_per_port=float(unit_rate_hz_per_port),
        effective_level_zodi=_payload_float(
            payload,
            "exozodi_effective_level_zodi",
            _payload_float(payload, "exozodi_level_zodi", 0.0),
        ),
        inclination_deg=float(inclination_deg),
        position_angle_deg=float(config.exozodi_position_angle_deg),
        inner_radius_au=float(config.exozodi_inner_radius_au),
        outer_radius_au=float(config.exozodi_outer_radius_au),
        radial_power=float(config.exozodi_radial_power),
    )
    return SkyScene(image=scene.image + exozodi_image, fov_rad=scene.fov_rad), glow


def _exozodi_template_image(
    *,
    scene: SkyScene,
    distance_pc: float,
    inclination_deg: float,
    position_angle_deg: float,
    inner_radius_au: float,
    outer_radius_au: float,
    radial_power: float,
) -> NDArray[np.float64]:
    coords_mas = scene.l_coords * MAS_PER_RAD
    east_mas, north_mas = np.meshgrid(coords_mas, coords_mas, indexing="xy")
    east_au = east_mas * 1.0e-3 * float(distance_pc)
    north_au = north_mas * 1.0e-3 * float(distance_pc)

    pa_rad = np.radians(float(position_angle_deg))
    major_au = east_au * np.sin(pa_rad) + north_au * np.cos(pa_rad)
    minor_sky_au = east_au * np.cos(pa_rad) - north_au * np.sin(pa_rad)
    cos_i = max(float(np.cos(np.radians(float(inclination_deg)))), 1.0e-3)
    minor_disk_au = minor_sky_au / cos_i
    radius_au = np.hypot(major_au, minor_disk_au)

    safe_radius_au = np.maximum(radius_au, float(inner_radius_au))
    radial = np.power(safe_radius_au / float(inner_radius_au), -float(radial_power))
    inner_taper = 1.0 - np.exp(-np.power(radius_au / float(inner_radius_au), 4.0))
    disk = np.where(radius_au <= float(outer_radius_au), radial * inner_taper, 0.0)
    disk = np.asarray(np.maximum(disk, 0.0), dtype=np.float64)
    total = float(np.sum(disk))
    if total <= 0.0 or not np.isfinite(total):
        return np.zeros_like(scene.image, dtype=np.float64)
    return disk / total


def _exozodi_unit_rate_hz_per_port(
    *,
    template: NDArray[np.float64],
    scene: SkyScene,
    inputs: _ImagingInputs,
) -> float:
    response = build_response_matrix(
        subapertures=inputs.subapertures,
        amplitudes=inputs.amplitudes,
        unitary=inputs.unitary,
        dark_port_indices=inputs.dark_port_indices,
        hour_angles_rad=inputs.hour_angles_rad,
        declination_rad=inputs.declination_rad,
        wavelength_m=inputs.wavelength_m,
        l_coords=scene.l_coords,
        m_coords=scene.m_coords,
        latitude_rad=inputs.latitude_rad,
        unitaries_by_wavelength=inputs.unitary_wavelength_pairs,
        roll_angles_rad=inputs.roll_angles_rad,
        wavelength_weights=(
            inputs.stellar_rate_hz_by_wavelength
            / np.sum(inputs.stellar_rate_hz_by_wavelength)
        ),
    )
    total_area = float(np.sum(np.abs(inputs.amplitudes) ** 2))
    if total_area <= 0.0:
        return 0.0
    rates = (
        float(inputs.stellar_photon_rate_hz)
        / total_area
        * (response @ np.asarray(template, dtype=np.float64).ravel())
    )
    finite = np.asarray(rates[np.isfinite(rates)], dtype=np.float64)
    if finite.size == 0:
        return 0.0
    return max(float(np.mean(finite)), 0.0)


def _positive_exozodi_rate_hz_per_port(payload: dict[str, Any]) -> float:
    return _payload_float(payload, "exozodi_rate_hz_per_dark_port", 0.0)


def _payload_float(payload: dict[str, Any], key: str, fallback: float) -> float:
    raw = payload.get(key, fallback)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(fallback)
    if not np.isfinite(value):
        return float(fallback)
    return value


def _sky_rate_without_exozodi(
    sky_rate_hz_per_port: float,
    exozodi_payload: dict[str, Any] | None,
) -> float:
    if exozodi_payload is None:
        return float(sky_rate_hz_per_port)
    exozodi_rate = _positive_exozodi_rate_hz_per_port(exozodi_payload)
    if exozodi_rate <= 0.0:
        return float(sky_rate_hz_per_port)
    return max(float(sky_rate_hz_per_port) - exozodi_rate, 0.0)


def _plot_truth_snr_beam(
    *,
    truth_image: NDArray[np.float64],
    snr_map: NDArray[np.float64],
    matched_filter_beam: NDArray[np.float64],
    fov_mas: float,
    planets: tuple[_PlanetView, ...],
    title: str,
) -> plt.Figure:
    extent = [-fov_mas, fov_mas, -fov_mas, fov_mas]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), constrained_layout=True)
    ax_truth, ax_snr, ax_beam = axes

    im_truth = ax_truth.imshow(
        truth_image,
        origin="lower",
        extent=extent,
        cmap="magma",
        norm=_truth_display_norm(truth_image),
    )
    fig.colorbar(im_truth, ax=ax_truth, shrink=0.85).set_label("Contrast")
    ax_truth.set_title("Input Scene (Truth)")

    snr_peak = float(np.nanmax(snr_map)) if snr_map.size else 0.0
    im_snr = ax_snr.imshow(
        snr_map,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        norm=_snr_display_norm(snr_map),
    )
    fig.colorbar(im_snr, ax=ax_snr, shrink=0.85).set_label(
        r"Matched-filter SNR ($\sigma$)"
    )
    levels = _snr_contour_levels(snr_peak)
    if levels:
        coords = np.linspace(-fov_mas, fov_mas, snr_map.shape[0])
        ax_snr.contour(
            coords,
            coords,
            snr_map,
            levels=levels,
            colors=["white"] * len(levels),
            linewidths=[0.6] * len(levels),
            linestyles=["--"] + ["-"] * (len(levels) - 1),
            alpha=0.7,
        )
    ax_snr.set_title(
        rf"Common-Mode Projected SNR (peak = {snr_peak:.1f}$\sigma$)"
    )

    im_beam = ax_beam.imshow(
        matched_filter_beam,
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=0.0,
        vmax=max(float(np.max(matched_filter_beam)), 1.0),
    )
    fig.colorbar(im_beam, ax=ax_beam, shrink=0.85).set_label(
        "Matched-filter beam RMS (normalized)"
    )
    ax_beam.set_title("Matched-Filter Beam")

    for ax in axes:
        ax.set_xlabel(r"$\Delta\alpha$ (mas)")
        ax.set_ylabel(r"$\Delta\delta$ (mas)")
        ax.axhline(0.0, color="white", linestyle=":", linewidth=0.5, alpha=0.4)
        ax.axvline(0.0, color="white", linestyle=":", linewidth=0.5, alpha=0.4)

    _annotate_planets(axes, snr_map, fov_mas, planets)
    if title:
        fig.suptitle(title, y=1.02)
    return fig


def _plot_truth_nnls(
    *,
    truth_image: NDArray[np.float64],
    nnls_image: NDArray[np.float64],
    snr_map: NDArray[np.float64],
    fov_mas: float,
    planets: tuple[_PlanetView, ...],
    title: str,
) -> plt.Figure:
    extent = [-fov_mas, fov_mas, -fov_mas, fov_mas]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), constrained_layout=True)
    ax_truth, ax_snr, ax_nnls = axes

    im_truth = ax_truth.imshow(
        truth_image,
        origin="lower",
        extent=extent,
        cmap="magma",
        norm=_truth_display_norm(truth_image),
    )
    fig.colorbar(im_truth, ax=ax_truth, shrink=0.85).set_label("Contrast")
    ax_truth.set_title("Input Scene (Truth)")

    snr_peak = float(np.nanmax(snr_map)) if snr_map.size else 0.0
    im_snr = ax_snr.imshow(
        snr_map,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        norm=_snr_display_norm(snr_map),
    )
    fig.colorbar(im_snr, ax=ax_snr, shrink=0.85).set_label(
        r"NNLS component SNR ($\sigma$)"
    )
    levels = _snr_contour_levels(snr_peak)
    if levels:
        coords = np.linspace(-fov_mas, fov_mas, snr_map.shape[0])
        ax_snr.contour(
            coords,
            coords,
            snr_map,
            levels=levels,
            colors=["red"] * len(levels),
            linewidths=[1.0] + [1.3] * (len(levels) - 1),
            linestyles=["--"] + ["-"] * (len(levels) - 1),
        )
    ax_snr.set_title(rf"NNLS Component SNR (peak = {snr_peak:.1f}$\sigma$)")

    nnls_max = float(np.max(nnls_image)) if np.max(nnls_image) > 0.0 else 1.0
    im_nnls = ax_nnls.imshow(
        nnls_image,
        origin="lower",
        extent=extent,
        cmap="magma",
        norm=AsinhNorm(
            linear_width=max(nnls_max * 0.01, 1.0e-12),
            vmin=0.0,
            vmax=nnls_max,
        ),
    )
    fig.colorbar(im_nnls, ax=ax_nnls, shrink=0.85).set_label("Recovered contrast")
    ax_nnls.set_title("NNLS Reconstruction")

    for ax in axes:
        ax.set_xlabel(r"$\Delta\alpha$ (mas)")
        ax.set_ylabel(r"$\Delta\delta$ (mas)")
        ax.axhline(0.0, color="white", linestyle=":", linewidth=0.5, alpha=0.4)
        ax.axvline(0.0, color="white", linestyle=":", linewidth=0.5, alpha=0.4)

    _annotate_planets(axes, snr_map, fov_mas, planets)
    if title:
        fig.suptitle(title, y=1.02)
    return fig


def _snr_display_norm(snr_map: NDArray[np.float64]) -> Normalize:
    finite = np.asarray(snr_map[np.isfinite(snr_map)], dtype=np.float64)
    if finite.size == 0:
        return Normalize(vmin=0.0, vmax=5.0)
    snr_min = float(np.min(finite))
    snr_max = float(np.max(finite))
    if snr_min < 0.0:
        limit = max(abs(snr_min), abs(snr_max), 5.0)
        return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    return Normalize(vmin=0.0, vmax=max(snr_max, 5.0))


def _truth_display_norm(truth_image: NDArray[np.float64]) -> AsinhNorm:
    truth_max = float(np.max(truth_image)) if np.max(truth_image) > 0 else 1.0
    return AsinhNorm(
        linear_width=max(truth_max * 1.0e-4, 1.0e-15),
        vmin=0.0,
        vmax=truth_max,
    )


def _snr_contour_levels(snr_peak: float) -> list[float]:
    if snr_peak <= 5.0:
        return []
    raw_levels = [5.0]
    raw_levels.extend(
        level for level in (0.25 * snr_peak, 0.5 * snr_peak, 0.75 * snr_peak)
        if level > 6.0
    )
    levels: list[float] = []
    for level in raw_levels:
        if not levels or level > levels[-1] * 1.05:
            levels.append(float(level))
    return levels


def _nnls_component_snr_image(
    *,
    nnls_image: NDArray[np.float64],
    observation: Any,
    inputs: _ImagingInputs,
    scene: Any,
) -> NDArray[np.float64]:
    response = _nnls_response_matrix(
        observation=observation,
        inputs=inputs,
        scene=scene,
    )
    n_rows = int(observation.measurements.size)
    response_scale = (
        np.asarray(observation.response_scale_per_row, dtype=np.float64).ravel()
        if observation.response_scale_per_row is not None
        and observation.response_scale_per_row.size == n_rows
        else np.ones(n_rows, dtype=np.float64)
    )
    variance_counts = _observation_variance_counts(observation, n_rows)
    sqrt_inv_variance = np.sqrt(
        np.divide(
            1.0,
            variance_counts,
            out=np.zeros_like(variance_counts, dtype=np.float64),
            where=variance_counts > 0.0,
        )
    )
    row_weight = response_scale * sqrt_inv_variance
    weighted_response = response * row_weight[:, None]
    norm_sq = np.einsum("ij,ij->j", weighted_response, weighted_response)
    common = np.mean(weighted_response, axis=1)
    common_norm_sq = float(common @ common)
    if common_norm_sq > 0.0:
        common_dot = common @ weighted_response
        norm_sq = norm_sq - common_dot * common_dot / common_norm_sq
    component_snr = nnls_image.ravel() * np.sqrt(np.maximum(norm_sq, 0.0))
    return component_snr.reshape(nnls_image.shape)


def _observation_variance_counts(observation: Any, n_rows: int) -> NDArray[np.float64]:
    variance = getattr(observation, "variance_counts", None)
    if variance is not None and variance.size == n_rows:
        return np.asarray(variance, dtype=np.float64).ravel()
    baseline = getattr(observation, "baseline_counts", None)
    if baseline is not None and baseline.size == n_rows:
        return np.asarray(baseline, dtype=np.float64).ravel()
    return np.ones(n_rows, dtype=np.float64)


def _nnls_candidate_indices(
    *,
    snr_map: NDArray[np.float64],
    planets: tuple[_PlanetView, ...],
    fov_mas: float,
    n_pix: int,
    max_candidates: int | None,
) -> NDArray[np.intp] | None:
    if max_candidates is None:
        return None
    flat_abs_snr = np.abs(np.asarray(snr_map, dtype=np.float64).ravel())
    n_total = int(flat_abs_snr.size)
    n_keep = min(max(int(max_candidates), 1), n_total)
    if n_keep == n_total:
        candidates = np.arange(n_total, dtype=np.intp)
    else:
        candidates = np.argpartition(flat_abs_snr, -n_keep)[-n_keep:].astype(np.intp)
    planet_candidates = _planet_candidate_indices(
        planets=planets,
        fov_mas=fov_mas,
        n_pix=n_pix,
    )
    return np.unique(np.concatenate((candidates, planet_candidates))).astype(np.intp)


def _planet_candidate_indices(
    *,
    planets: tuple[_PlanetView, ...],
    fov_mas: float,
    n_pix: int,
) -> NDArray[np.intp]:
    coords = np.linspace(-float(fov_mas), float(fov_mas), int(n_pix))
    indices: list[int] = []
    for planet in planets:
        pa_rad = np.radians(planet.position_angle_deg)
        l_mas = planet.separation_mas * np.sin(pa_rad)
        m_mas = planet.separation_mas * np.cos(pa_rad)
        col = int(np.argmin(np.abs(coords - l_mas)))
        row = int(np.argmin(np.abs(coords - m_mas)))
        indices.append(row * int(n_pix) + col)
    return np.asarray(indices, dtype=np.intp)


def _nnls_response_matrix(
    *,
    observation: Any,
    inputs: _ImagingInputs,
    scene: Any,
) -> NDArray[np.float64]:
    n_rows = int(observation.measurements.size)
    n_pix_sq = int(scene.n_pix * scene.n_pix)
    if observation.response_matrix is not None and observation.response_matrix.shape == (
        n_rows,
        n_pix_sq,
    ):
        return np.asarray(observation.response_matrix, dtype=np.float64)
    if np.asarray(observation.measurements).ndim != 2:
        raise ValueError(
            "imaging_reconstruction: wavelength-resolved NNLS component SNR "
            "requires observation.response_matrix."
        )
    return build_response_matrix(
        subapertures=inputs.subapertures,
        amplitudes=inputs.amplitudes,
        unitary=inputs.unitary,
        dark_port_indices=inputs.dark_port_indices,
        hour_angles_rad=inputs.hour_angles_rad,
        declination_rad=inputs.declination_rad,
        wavelength_m=inputs.wavelength_m,
        l_coords=scene.l_coords,
        m_coords=scene.m_coords,
        latitude_rad=inputs.latitude_rad,
        unitaries_by_wavelength=inputs.unitary_wavelength_pairs,
        roll_angles_rad=inputs.roll_angles_rad,
        wavelength_weights=observation.wavelength_weights,
    )


def _annotate_planets(
    axes: NDArray[Any] | tuple[plt.Axes, ...],
    snr_map: NDArray[np.float64],
    fov_mas: float,
    planets: tuple[_PlanetView, ...],
) -> None:
    coords = np.linspace(-fov_mas, fov_mas, snr_map.shape[0])
    ax_snr = axes[1]
    for planet in planets:
        pa_rad = np.radians(planet.position_angle_deg)
        l_mas = planet.separation_mas * np.sin(pa_rad)
        m_mas = planet.separation_mas * np.cos(pa_rad)
        col = int(np.argmin(np.abs(coords - l_mas)))
        row = int(np.argmin(np.abs(coords - m_mas)))
        l_snap = float(coords[col])
        m_snap = float(coords[row])
        for ax in axes:
            ax.plot(l_snap, m_snap, "o", mfc="none", mec="cyan", ms=14, mew=1.4)
            ax.annotate(
                planet.label,
                (l_snap, m_snap),
                textcoords="offset points",
                xytext=(9, 9),
                color="cyan",
                fontsize=10,
                fontweight="bold",
            )
        ax_snr.annotate(
            rf"{float(snr_map[row, col]):.1f}$\sigma$",
            (l_snap, m_snap),
            textcoords="offset points",
            xytext=(9, -15),
            color="white",
            fontsize=9,
            fontweight="bold",
        )


def _summary_payload(
    *,
    stem: str,
    record: RunRecord,
    inputs: _ImagingInputs,
    config: ImagingReconstructionOutputConfig,
    fov_mas: float,
    snr_map: NDArray[np.float64],
    nnls_image: NDArray[np.float64] | None,
    nnls: ReconstructedImage | None,
    nnls_observation: Any | None,
    planets: tuple[_PlanetView, ...],
    exozodi_glow: _ExozodiGlowView | None,
    integration_time_per_step_s: float,
    skipped_records: int,
) -> dict[str, Any]:
    coords = np.linspace(-fov_mas, fov_mas, snr_map.shape[0])
    planet_rows: list[dict[str, Any]] = []
    for planet in planets:
        pa_rad = np.radians(planet.position_angle_deg)
        col = int(np.argmin(np.abs(coords - planet.separation_mas * np.sin(pa_rad))))
        row = int(np.argmin(np.abs(coords - planet.separation_mas * np.cos(pa_rad))))
        planet_rows.append(
            {
                "label": planet.label,
                "separation_mas": planet.separation_mas,
                "position_angle_deg": planet.position_angle_deg,
                "contrast": planet.contrast,
                "flux_model": planet.flux_model,
                "radius_earth": planet.radius_earth,
                "semi_major_axis_au": planet.semi_major_axis_au,
                "geometric_albedo": planet.geometric_albedo,
                "phase_angle_deg": planet.phase_angle_deg,
                "contrast_source": planet.contrast_source,
                "local_snr": float(snr_map[row, col]),
                "nnls_recovered_contrast": (
                    None if nnls_image is None else float(nnls_image[row, col])
                ),
                "pixel_row": row,
                "pixel_col": col,
            }
        )

    return {
        "stem": stem,
        "run_name": record.config.run.name,
        "coord": dict(record.coord_tuple),
        "peak_snr": float(np.nanmax(snr_map)) if snr_map.size else 0.0,
        "nnls_peak_contrast": (
            None if nnls_image is None else float(np.nanmax(nnls_image))
        ),
        "nnls_total_recovered_contrast": (
            None if nnls_image is None else float(np.sum(nnls_image))
        ),
        "n_pix": int(config.n_pix),
        "fov_mas": float(fov_mas),
        "n_time": int(len(inputs.hour_angles_rad)),
        "rotation_mode": inputs.rotation_mode,
        "integration_time_s": float(inputs.integration_time_s),
        "integration_time_per_step_s": float(integration_time_per_step_s),
        "stellar_photon_rate_hz": float(inputs.stellar_photon_rate_hz),
        "null_depth": float(inputs.null_depth),
        "dark_count_rate_hz_per_port": float(inputs.dark_count_rate_hz_per_port),
        "sky_rate_hz_per_port": float(inputs.sky_rate_hz_per_port),
        "include_exozodi_glow": bool(config.include_exozodi_glow),
        "exozodi_glow_total_contrast": (
            None if exozodi_glow is None else float(exozodi_glow.total_contrast)
        ),
        "exozodi_glow_target_rate_hz_per_port": (
            None
            if exozodi_glow is None
            else float(exozodi_glow.target_rate_hz_per_port)
        ),
        "exozodi_glow_unit_rate_hz_per_port": (
            None if exozodi_glow is None else float(exozodi_glow.unit_rate_hz_per_port)
        ),
        "exozodi_effective_level_zodi": (
            None if exozodi_glow is None else float(exozodi_glow.effective_level_zodi)
        ),
        "exozodi_inclination_deg": (
            None if exozodi_glow is None else float(exozodi_glow.inclination_deg)
        ),
        "exozodi_position_angle_deg": (
            None if exozodi_glow is None else float(exozodi_glow.position_angle_deg)
        ),
        "exozodi_inner_radius_au": (
            None if exozodi_glow is None else float(exozodi_glow.inner_radius_au)
        ),
        "exozodi_outer_radius_au": (
            None if exozodi_glow is None else float(exozodi_glow.outer_radius_au)
        ),
        "exozodi_radial_power": (
            None if exozodi_glow is None else float(exozodi_glow.radial_power)
        ),
        "eps_systematic_used": float(inputs.eps_systematic),
        "with_fringe_tracking": bool(inputs.with_fringe_tracking),
        "servo_bw_hz": float(inputs.servo_bw_hz),
        "n_dark_ports": int(inputs.dark_port_indices.size),
        "n_wavelengths": int(len(inputs.unitary_wavelength_pairs)),
        "wavelengths_um": [
            float(wavelength_m * 1.0e6)
            for _unitary, wavelength_m in inputs.unitary_wavelength_pairs
        ],
        "declination_deg": float(np.degrees(inputs.declination_rad)),
        "latitude_deg": float(np.degrees(inputs.latitude_rad)),
        "hour_angles_h": [
            float(np.degrees(value) / 15.0) for value in inputs.hour_angles_rad
        ],
        "roll_angles_deg": (
            None
            if inputs.roll_angles_rad is None
            else [float(np.degrees(value)) for value in inputs.roll_angles_rad]
        ),
        "poisson": bool(config.poisson),
        "chop_mode": str(config.chop_mode),
        "compute_nnls": bool(config.compute_nnls),
        "nnls_wavelength_mode": (
            None if nnls_image is None else str(config.nnls_wavelength_mode)
        ),
        "nnls_candidate_pixels": (
            None if nnls_image is None else int(config.nnls_candidate_pixels)
        ),
        "nnls_regularization_fraction": (
            None if nnls_image is None else float(config.nnls_regularization_fraction)
        ),
        "nnls_measurement_shape": (
            None if nnls_observation is None else list(nnls_observation.measurements.shape)
        ),
        "nnls_rows": (
            None if nnls_observation is None else int(nnls_observation.measurements.size)
        ),
        "nnls_converged": None if nnls is None else bool(nnls.converged),
        "nnls_solver_status": None if nnls is None else nnls.solver_status,
        "nnls_solver_message": None if nnls is None else nnls.solver_message,
        "nnls_n_iterations": None if nnls is None else nnls.n_iterations,
        "nnls_residual_norm": None if nnls is None else nnls.residual_norm,
        "planets": planet_rows,
        "skipped_records": int(skipped_records),
    }


def _find_detector_component(state: SimulationState) -> dict[str, Any]:
    for name, value in reversed(state.results.components.items()):
        if not name.startswith("detector.") or not isinstance(value, dict):
            continue
        if {"port_rates", "port_labels", "integration_time_s"}.issubset(value):
            return value
    raise ValueError(
        "imaging_reconstruction requires an upstream detector component with "
        "port_rates, port_labels, and integration_time_s."
    )


def _selected_wavelength_indices(
    *,
    n_wavelengths: int,
    max_wavelengths: int | None,
) -> tuple[int, ...]:
    if max_wavelengths is None or max_wavelengths >= n_wavelengths:
        return tuple(range(n_wavelengths))
    return tuple(
        int(idx)
        for idx in np.linspace(0, n_wavelengths - 1, int(max_wavelengths)).round()
    )


def _stellar_rates_by_wavelength(
    *,
    state: SimulationState,
    wavelength_indices: tuple[int, ...],
    total_rate_hz: float,
) -> NDArray[np.float64]:
    rates = state.photon_rates.rates_hz
    if rates is None:
        return _uniform_rates_by_wavelength(
            total_rate_hz=total_rate_hz,
            n_wavelengths=len(wavelength_indices),
        )
    arr = np.asarray(rates, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] <= max(wavelength_indices, default=-1):
        return _uniform_rates_by_wavelength(
            total_rate_hz=total_rate_hz,
            n_wavelengths=len(wavelength_indices),
        )
    selected = np.sum(arr[:, wavelength_indices], axis=0)
    return _rescaled_rates_by_wavelength(
        selected,
        total_rate_hz=total_rate_hz,
        name="state.photon_rates.rates_hz",
    )


def _background_rates_by_wavelength(
    *,
    state: SimulationState,
    wavelength_indices: tuple[int, ...],
    total_rate_hz: float,
    exclude_exozodi: bool = False,
) -> NDArray[np.float64]:
    if exclude_exozodi:
        zodi = _maybe_find_component(state, "background.space_zodi")
        if zodi is not None and "local_zodi_rate_hz_per_bin" in zodi:
            values = np.asarray(zodi["local_zodi_rate_hz_per_bin"], dtype=np.float64)
            if values.ndim == 1 and values.size > max(wavelength_indices, default=-1):
                return _rescaled_rates_by_wavelength(
                    values[list(wavelength_indices)],
                    total_rate_hz=total_rate_hz,
                    name="local_zodi_rate_hz_per_bin",
                )
    for _name, payload in state.results.components.items():
        if not isinstance(payload, dict):
            continue
        for key in (
            "background_rate_hz_per_bin",
            "zodi_rate_hz_per_bin",
            "sky_rate_hz_per_bin",
        ):
            if key not in payload:
                continue
            values = np.asarray(payload[key], dtype=np.float64)
            if values.ndim == 1 and values.size > max(wavelength_indices, default=-1):
                return _rescaled_rates_by_wavelength(
                    values[list(wavelength_indices)],
                    total_rate_hz=total_rate_hz,
                    name=key,
                )
    return _uniform_rates_by_wavelength(
        total_rate_hz=total_rate_hz,
        n_wavelengths=len(wavelength_indices),
    )


def _uniform_rates_by_wavelength(
    *,
    total_rate_hz: float,
    n_wavelengths: int,
) -> NDArray[np.float64]:
    if n_wavelengths <= 0:
        raise ValueError("n_wavelengths must be positive.")
    return np.full(
        int(n_wavelengths),
        float(total_rate_hz) / float(n_wavelengths),
        dtype=np.float64,
    )


def _rescaled_rates_by_wavelength(
    rates_hz: NDArray[np.float64],
    *,
    total_rate_hz: float,
    name: str,
) -> NDArray[np.float64]:
    values = np.asarray(rates_hz, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if np.any(values < 0.0) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} entries must be finite and non-negative.")
    total = float(np.sum(values))
    if total <= 0.0:
        return _uniform_rates_by_wavelength(
            total_rate_hz=total_rate_hz,
            n_wavelengths=values.size,
        )
    return values * (float(total_rate_hz) / total)


def _planet_views(
    record: RunRecord,
    config: ImagingReconstructionOutputConfig,
) -> tuple[_PlanetView, ...]:
    planets: list[_PlanetView] = []
    for index, planet in enumerate(config.planets):
        contrast, semi_major_axis_au, contrast_source = _planet_contrast(planet)
        planets.append(
            _PlanetView(
                separation_mas=float(planet.separation_mas),
                position_angle_deg=float(planet.position_angle_deg),
                contrast=contrast,
                label=planet.label or _planet_label(index),
                flux_model=str(planet.flux_model),
                radius_earth=(
                    None if planet.radius_earth is None else float(planet.radius_earth)
                ),
                semi_major_axis_au=semi_major_axis_au,
                geometric_albedo=float(planet.geometric_albedo),
                phase_angle_deg=float(planet.phase_angle_deg),
                contrast_source=contrast_source,
            )
        )
    if planets:
        return tuple(planets)

    scene = record.result.state.scene
    if (
        scene.planet_separation_mas is None
        or scene.planet_position_angle_deg is None
        or scene.planet_contrast is None
    ):
        raise ValueError(
            "imaging_reconstruction requires scene.planet fields or "
            "output.imaging_reconstruction.planets."
        )
    return (
        _PlanetView(
            separation_mas=float(scene.planet_separation_mas),
            position_angle_deg=float(scene.planet_position_angle_deg),
            contrast=float(scene.planet_contrast),
            label="b",
        ),
    )


def _planet_contrast(planet) -> tuple[float, float | None, str]:
    if planet.flux_model == "fixed_contrast":
        if planet.contrast is None:
            raise ValueError(
                "imaging_reconstruction: fixed_contrast planet missing contrast."
            )
        return float(planet.contrast), (
            None
            if planet.semi_major_axis_au is None
            else float(planet.semi_major_axis_au)
        ), "fixed_contrast"

    semi_major_axis_au = _planet_semi_major_axis_au(planet)
    contrast = reflected_light_contrast(
        radius_earth=float(planet.radius_earth),
        semi_major_axis_au=semi_major_axis_au,
        geometric_albedo=float(planet.geometric_albedo),
        phase_angle_rad=np.radians(float(planet.phase_angle_deg)),
    )
    return contrast, semi_major_axis_au, "reflected_light"


def _planet_semi_major_axis_au(planet) -> float:
    if planet.semi_major_axis_au is not None:
        return float(planet.semi_major_axis_au)
    raise ValueError(
        "imaging_reconstruction: reflected_light planets require "
        "semi_major_axis_au; projected separation is not a physical orbital "
        "semi-major axis."
    )


def _resolve_fov_mas(
    config: ImagingReconstructionOutputConfig,
    planets: tuple[_PlanetView, ...],
) -> float:
    if config.fov_mas is not None:
        return float(config.fov_mas)
    max_sep = max((planet.separation_mas for planet in planets), default=10.0)
    return max(45.0, 2.25 * max_sep)


def _sampling_angles_rad(
    *,
    center_hour_angle_h: float,
    integration_time_s: float,
    config: ImagingReconstructionOutputConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    if config.rotation_mode == "spacecraft_roll":
        roll_angles = _roll_angles_rad(
            roll_span_deg=float(config.roll_span_deg),
            n_roll_steps=int(config.n_roll_steps),
        )
        hour_angles = np.zeros_like(roll_angles, dtype=np.float64)
        return hour_angles, roll_angles

    n_time = int(config.n_time)
    if config.rotation_mode == "fixed":
        return np.full(n_time, np.radians(center_hour_angle_h * 15.0)), None

    span = (
        float(config.hour_angle_span_h)
        if config.hour_angle_span_h is not None
        else float(integration_time_s) / 3600.0
    )
    if n_time == 1:
        hours = np.asarray([center_hour_angle_h], dtype=np.float64)
    else:
        hours = np.linspace(
            center_hour_angle_h - 0.5 * span,
            center_hour_angle_h + 0.5 * span,
            n_time,
        )
    return np.radians(hours * 15.0), None


def _roll_angles_rad(
    *,
    roll_span_deg: float,
    n_roll_steps: int,
) -> NDArray[np.float64]:
    if n_roll_steps == 1:
        return np.asarray([0.0], dtype=np.float64)
    return np.radians(
        np.linspace(
            -0.5 * float(roll_span_deg),
            0.5 * float(roll_span_deg),
            int(n_roll_steps),
            dtype=np.float64,
        )
    )


def _site_latitude_deg(record: RunRecord) -> float:
    site = record.config.site.model_dump(mode="python")
    value = site.get("latitude_deg")
    if value is None:
        return float(np.degrees(MAUNAKEA_LAT_RAD))
    return float(value)


def _detector_cap_hz(detector: dict[str, Any]) -> float | None:
    for key in ("mkid_max_count_rate_hz", "snspd_max_count_rate_hz"):
        if key in detector:
            return float(detector[key])
    return None


def _first_present(
    payload: dict[str, Any] | None,
    keys: tuple[str, ...],
    fallback: Any,
) -> Any:
    if payload is None:
        return fallback
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return fallback


def _positive_float(value: Any, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return out


def _nonnegative_float(value: Any, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be non-negative and finite.")
    return out


def _record_seed(
    config: ImagingReconstructionOutputConfig,
    record: RunRecord,
    index: int,
) -> int | None:
    if config.seed is not None:
        return int(config.seed) + index
    return int(record.config.run.seed) + 7919 * index


def _plot_title(record: RunRecord) -> str:
    if not record.coord_tuple:
        return record.config.run.name
    coord = ", ".join(f"{key}={value}" for key, value in record.coord_tuple)
    return f"{record.config.run.name}: {coord}"


def _planet_label(index: int) -> str:
    labels = ("b", "c", "d", "e", "f", "g", "h")
    return labels[index] if index < len(labels) else str(index + 1)


__all__ = ["imaging_reconstruction"]
