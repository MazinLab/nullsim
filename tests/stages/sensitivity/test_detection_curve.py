"""Tests for the sensitivity.detection_curve stage."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nullsim.pipeline import (
    Field,
    Geometry,
    RunContext,
    Scene,
    SimulationState,
    Throughput,
    Wavefront,
)
from nullsim.pipeline.state import OPDBudget, OPDComponent, Results
from nullsim.physics.realized_null import RealizedNullResult
from nullsim.physics.ruilier_cassaing import coupling_efficiency
from nullsim.stages.sensitivity import DetectionCurve
from nullsim.stages.sensitivity.detection_curve import (
    _angular_jitter_convolve_profile,
    _combine_realized_null_results,
    _resolve_amp_servo_photon_rate_hz,
    _per_azimuth_stellar_fields,
    _planet_input_transport_scales,
)


def _context(tmp_path: Path, *, source_path: Path | None = None) -> RunContext:
    return RunContext(
        rng=np.random.default_rng(0),
        config_hash="sensitivity-detection-curve",
        sweep_coord_tuple=(),
        instance_id="sensitivity",
        run_dir=tmp_path,
        source_path=source_path,
    )


def _identity_u_stack(n_modes: int, n_wl: int) -> np.ndarray:
    stack = np.zeros((n_modes, n_modes, n_wl), dtype=np.complex128)
    eye = np.eye(n_modes, dtype=np.complex128)
    for k in range(n_wl):
        stack[:, :, k] = eye
    return stack


def test_combine_realized_null_results_uses_paired_arm_samples() -> None:
    samples_a = np.array([[0.0, 2.0], [0.0, 2.0]], dtype=np.float64)
    samples_b = np.array([[2.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    for samples in (samples_a, samples_b):
        samples.setflags(write=False)
    result_a = RealizedNullResult(
        mean_null_depth=1.0,
        median_null_depth=1.0,
        std_null_depth=1.0,
        null_instability=1.0,
        ideal_null_depth=0.1,
        ideal_null_depth_kilo=0.2,
        mean_null_depth_kilo=10.0,
        n_samples=4,
        n_azimuths=1,
        n_realizations_per_azimuth=2,
        samples_by_wavelength=samples_a,
        samples_by_wavelength_kilo=10.0 * samples_a,
    )
    result_b = RealizedNullResult(
        mean_null_depth=1.0,
        median_null_depth=1.0,
        std_null_depth=1.0,
        null_instability=1.0,
        ideal_null_depth=0.3,
        ideal_null_depth_kilo=0.4,
        mean_null_depth_kilo=10.0,
        n_samples=4,
        n_azimuths=1,
        n_realizations_per_azimuth=2,
        samples_by_wavelength=samples_b,
        samples_by_wavelength_kilo=10.0 * samples_b,
    )

    combined = _combine_realized_null_results(
        (result_a, result_b),
        np.array([0.5, 0.5], dtype=np.float64),
        np.array([0.5, 0.5], dtype=np.float64),
    )

    assert combined.mean_null_depth == pytest.approx(1.0)
    assert combined.median_null_depth == pytest.approx(1.0)
    assert combined.std_null_depth == pytest.approx(0.0)
    assert combined.null_instability == pytest.approx(0.0)
    assert combined.null_fluctuation_rms == pytest.approx(0.0)
    assert combined.mean_null_depth_kilo == pytest.approx(10.0)
    assert combined.ideal_null_depth == pytest.approx(0.2)


def test_angular_jitter_convolution_smooths_radial_profiles() -> None:
    separations = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    profile = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    smoothed = _angular_jitter_convolve_profile(
        profile,
        separations_mas=separations,
        sigma_mas=1.0,
    )
    unchanged = _angular_jitter_convolve_profile(
        profile,
        separations_mas=separations,
        sigma_mas=0.0,
    )

    np.testing.assert_allclose(unchanged, profile)
    assert smoothed[1] < profile[1]
    assert smoothed[0] > 0.0
    assert smoothed[2] > 0.0
    assert smoothed[0] == pytest.approx(smoothed[2])


def test_angular_jitter_kernel_is_unbiased_on_nonuniform_grid() -> None:
    """Row 30: a log-spaced grid must not weight the dense side.

    A linear profile f(r) = r is reproduced exactly by a symmetric kernel
    away from the grid ends; row normalization on the bare nodes pulls it
    toward the densely sampled small-r side.
    """

    separations = np.geomspace(0.2, 300.0, 70)
    smoothed = _angular_jitter_convolve_profile(
        separations, separations_mas=separations, sigma_mas=0.5,
    )
    interior = (separations > 3.0) & (separations < 290.0)
    # Old row normalization biased this by 2.7%; the residual 3e-6 is the
    # trapezoid step error of the sigma/8 fine grid (it halves with the step).
    np.testing.assert_allclose(smoothed[interior], separations[interior], rtol=1e-5)


def test_amp_servo_rate_uses_physical_pickoff_mode_count() -> None:
    wavelengths_um = (1.20, 1.25, 1.30)
    base_params = SimpleNamespace(
        auto_amp_servo_photon_rate_per_sub_hz=None,
        amp_wfs_pickoff_fraction=0.30,
        simultaneous_science_bands=(),
        simultaneous_science_band_transmissions=(),
        n_bright_ports=12,
        n_dark_ports=11,
        n_spectro_ports=1,
    )
    state = SimulationState.empty({"wavelength_grid_um": wavelengths_um})
    state = replace(
        state,
        scene=Scene(star_jmag=4.0, star_spectral_type="M"),
        geometry=Geometry(area_m2=100.0),
        field=Field(
            amplitudes=np.ones((24, len(wavelengths_um), 1), dtype=np.complex128),
            wavelength_grid_um=wavelengths_um,
        ),
        throughput=Throughput(
            wavelength_grid_um=wavelengths_um,
            transmission_fraction=tuple(1.0 for _ in wavelengths_um),
            components={
                "transport.photometric_pickoff": {
                    "tap_fraction": 0.30,
                    "science_fraction": 0.70,
                    "n_input_modes": 12,
                }
            },
        ),
        results=Results(
            components={
                "chip.kernel_mzi_mesh.test": {
                    "n_modes": 24,
                }
            },
        ),
    )
    fallback_state = replace(
        state,
        throughput=Throughput(
            wavelength_grid_um=wavelengths_um,
            transmission_fraction=tuple(1.0 for _ in wavelengths_um),
        ),
    )

    physical_rate = _resolve_amp_servo_photon_rate_hz(state, base_params)
    fallback_rate = _resolve_amp_servo_photon_rate_hz(fallback_state, base_params)

    assert physical_rate == pytest.approx(2.0 * fallback_rate)


def test_amp_servo_rate_excludes_detector_qe_and_includes_pre_pickoff_field_loss() -> None:
    wavelengths_um = (1.25,)
    params = SimpleNamespace(
        auto_amp_servo_photon_rate_per_sub_hz=None,
        amp_wfs_pickoff_fraction=0.30,
        simultaneous_science_bands=(),
        simultaneous_science_band_transmissions=(),
        n_bright_ports=12,
        n_dark_ports=11,
        n_spectro_ports=1,
    )
    base = SimulationState.empty({"wavelength_grid_um": wavelengths_um})
    base = replace(
        base,
        scene=Scene(star_jmag=4.0, star_spectral_type="M"),
        geometry=Geometry(area_m2=100.0),
        field=Field(
            amplitudes=np.ones((24, 1, 1), dtype=np.complex128),
            wavelength_grid_um=wavelengths_um,
        ),
        results=Results(
            components={
                "chip.kernel_mzi_mesh.test": {
                    "n_modes": 24,
                }
            },
        ),
    )
    no_qe = replace(
        base,
        throughput=Throughput(
            wavelength_grid_um=wavelengths_um,
            transmission_fraction=(1.0,),
            components={
                "transport.photometric_pickoff": {
                    "tap_fraction": 0.30,
                    "science_fraction": 0.70,
                    "n_input_modes": 12,
                },
            },
        ),
    )
    with_qe = replace(
        base,
        throughput=Throughput(
            wavelength_grid_um=wavelengths_um,
            transmission_fraction=(0.7,),
            components={
                "transport.photometric_pickoff": {
                    "tap_fraction": 0.30,
                    "science_fraction": 0.70,
                    "n_input_modes": 12,
                },
                "detector.mkid_qe": {
                    "transmission_fraction": (0.7,),
                    "qe": 0.7,
                },
            },
        ),
    )
    with_delay = replace(
        base,
        throughput=Throughput(
            wavelength_grid_um=wavelengths_um,
            transmission_fraction=(0.7,),
            components={
                "delay_line_loss": {
                    "transmission_fraction": (0.5,),
                    "field_applied_loss": True,
                },
                "transport.photometric_pickoff": {
                    "tap_fraction": 0.30,
                    "science_fraction": 0.70,
                    "n_input_modes": 12,
                },
                "detector.mkid_qe": {
                    "transmission_fraction": (0.7,),
                    "qe": 0.7,
                },
            },
        ),
    )

    no_qe_rate = _resolve_amp_servo_photon_rate_hz(no_qe, params)
    qe_rate = _resolve_amp_servo_photon_rate_hz(with_qe, params)
    delay_rate = _resolve_amp_servo_photon_rate_hz(with_delay, params)

    assert qe_rate == pytest.approx(no_qe_rate)
    assert delay_rate == pytest.approx(0.5 * no_qe_rate)


def _keck_like_state(
    *,
    n_modes: int = 4,
    wavelengths_um: tuple[float, ...] = (1.20, 1.25, 1.30),
    star_diameter_mas: float = 1.0,
    n_apertures: int = 2,
    u_stack: np.ndarray | None = None,
    positions: np.ndarray | None = None,
    strehl: float = 0.93,
    piston_opd_nm_total: float = 1.35,  # corresponds to ~6.79 mrad at J band
) -> SimulationState:
    if u_stack is None:
        u_stack = _identity_u_stack(n_modes, len(wavelengths_um))
    if positions is None:
        # Two telescopes, two sub-apertures each, baseline 86 m east.
        positions = np.array(
            [
                [43.0, 0.0],
                [44.5, 0.0],
                [-43.0, 0.0],
                [-44.5, 0.0],
            ]
        )[:n_modes]

    state = SimulationState.empty({"wavelength_grid_um": wavelengths_um})
    state = replace(
        state,
        scene=Scene(
            star_jmag=4.0,
            star_angular_diameter_mas=star_diameter_mas,
        ),
        geometry=Geometry(
            diameter_m=10.0,
            area_m2=math.pi * 5.0 ** 2,
            n_apertures=n_apertures,
            aperture_positions_pupil_m=positions,
        ),
        field=Field(
            amplitudes=np.ones((n_modes, len(wavelengths_um), 1), dtype=np.complex128),
            wavelength_grid_um=wavelengths_um,
        ),
        wavefront=Wavefront(
            wavelength_grid_um=wavelengths_um,
            strehl_ratio=tuple(strehl for _ in wavelengths_um),
            amplitude_jitter_fraction=tuple(0.0 for _ in wavelengths_um),
            components={
                "ao_residual": {
                    "piston_opd_nm": {"total": float(piston_opd_nm_total)},
                    "piston_subaperture_diameter_m": 1.5,
                },
            },
        ),
    )
    # Inject a chip.kernel_mzi_mesh component carrying the unitary stack.
    components = dict(state.results.components)
    components["chip.kernel_mzi_mesh.test"] = {
        "u_stack": u_stack,
        "wavelength_grid_um": wavelengths_um,
        "wl_design_um": 1.25,
        "n_modes": n_modes,
    }
    state = replace(state, results=Results(components=components))
    return state


def test_detection_curve_requires_chip_unitary(tmp_path: Path) -> None:
    state = SimulationState.empty({"wavelength_grid_um": (1.25,)})
    state = replace(state, scene=Scene(star_jmag=4.0))
    stage = DetectionCurve(
        params={"n_bright_ports": 1, "n_dark_ports": 1, "separations_mas": (1.0,)},
        context=_context(tmp_path),
    )
    with pytest.raises(ValueError, match="chip.kernel_mzi_mesh"):
        stage.apply(state)


def test_detection_curve_identity_unitary_dark_throughput_one(tmp_path: Path) -> None:
    """U=I, dark=all -> per-separation throughput == 1 at every sep (no fringe rolloff)."""

    n_modes = 4
    state = _keck_like_state(
        n_modes=n_modes,
        u_stack=_identity_u_stack(n_modes, 1),
        wavelengths_um=(1.25,),
        # Place all sub-apertures at origin so OPD = 0 at every separation.
        positions=np.zeros((n_modes, 2)),
        strehl=1.0,
        piston_opd_nm_total=0.0,
    )
    # Identity unitary with 1 bright + (n_modes-1) dark ports. With sub-aperture
    # positions at origin, OPD = 0 so every input mode carries equal phase;
    # for U=I dark-port total power equals sum_{k>=1} |E_k|^2 = (M-1)/M of the total.
    stage = DetectionCurve(
        params={
            "n_bright_ports": 1,
            "n_dark_ports": n_modes - 1,
            "separations_mas": (0.5, 1.0, 5.0),
            "include_angular_coupling": False,
            "n_mc_realizations": 0,
        },
        context=_context(tmp_path),
    )
    out = stage.apply(state)
    payload = out.results.components["sensitivity.detection_curve"]
    expected_throughput = (n_modes - 1) / n_modes
    np.testing.assert_allclose(
        payload["planet_throughputs"], expected_throughput, rtol=1e-12,
    )
    # T(r) is flat across the disk, so the resolved disk adds nothing beyond
    # the on-axis value already in the realized null (Issue 17).
    assert payload["disk_null_floor"] == pytest.approx(0.0, abs=1e-14)
    assert payload["effective_null_with_disk"] == pytest.approx(
        expected_throughput, rel=1e-12,
    )
    # U=I is not a real null: dark-port fraction of stellar power equals (M-1)/M.
    assert payload["realized_null_depth"] == pytest.approx(expected_throughput, rel=1e-12)
    assert payload["null_instability"] == 0.0  # MC disabled => no fluctuation


def test_detection_curve_accepts_rectangular_chip_transfer(tmp_path: Path) -> None:
    n_input_modes = 2
    tracker_rows = np.eye(n_input_modes, dtype=np.complex128)
    dark_row = np.array([[1.0, -1.0]], dtype=np.complex128) / math.sqrt(2.0)
    spectro_row = np.array([[1.0, 1.0]], dtype=np.complex128) / math.sqrt(2.0)
    rectangular = np.vstack([tracker_rows, dark_row, spectro_row])[:, :, np.newaxis]
    state = _keck_like_state(
        n_modes=n_input_modes,
        u_stack=rectangular,
        wavelengths_um=(1.25,),
        positions=np.zeros((n_input_modes, 2)),
        strehl=1.0,
        piston_opd_nm_total=0.0,
    )
    components = dict(state.results.components)
    components["chip.kernel_mzi_mesh.test"] = {
        **components["chip.kernel_mzi_mesh.test"],
        "n_input_modes": n_input_modes,
        "n_output_ports": 4,
        "n_bright_ports": 2,
        "n_dark_ports": 1,
        "n_spectro_ports": 1,
    }
    state = replace(state, results=Results(components=components))
    stage = DetectionCurve(
        params={
            "n_bright_ports": 2,
            "n_dark_ports": 1,
            "n_spectro_ports": 1,
            "separations_mas": (0.5, 1.0),
            "include_angular_coupling": False,
            "n_mc_realizations": 0,
        },
        context=_context(tmp_path),
    )

    out = stage.apply(state)
    payload = out.results.components["sensitivity.detection_curve"]

    assert payload["chip_n_input_modes"] == n_input_modes
    assert payload["chip_n_output_ports"] == 4
    np.testing.assert_allclose(payload["planet_throughputs"], 0.0, atol=1.0e-14)
    assert payload["realized_null_depth"] == pytest.approx(0.0, abs=1.0e-14)


def test_detection_curve_disk_floor_is_independent_of_separation_grid(
    tmp_path: Path,
) -> None:
    """Row 38: the disk integral uses its own nodes inside the stellar radius.

    A one-point grid that does not reach the 0.5 mas stellar radius and the
    default 70-point grid give the same disk floor.
    """

    n_modes = 4
    rng = np.random.default_rng(5)
    q, _ = np.linalg.qr(
        rng.normal(size=(n_modes, n_modes)) + 1j * rng.normal(size=(n_modes, n_modes))
    )
    state = _keck_like_state(
        n_modes=n_modes,
        u_stack=q[:, :, np.newaxis],
        wavelengths_um=(1.25,),
        star_diameter_mas=1.0,
        strehl=1.0,
        piston_opd_nm_total=0.0,
    )

    def disk_floor(separations: dict[str, tuple[float, ...]]) -> float:
        stage = DetectionCurve(
            params={
                "n_bright_ports": 1,
                "n_dark_ports": n_modes - 1,
                "include_angular_coupling": False,
                "n_mc_realizations": 0,
                **separations,
            },
            context=_context(tmp_path),
        )
        payload = stage.apply(state).results.components["sensitivity.detection_curve"]
        return float(payload["disk_null_floor"])

    short = disk_floor({"separations_mas": (0.1,)})
    assert abs(short) > 1e-6  # this chip leaks less off axis: a negative excess
    assert short == pytest.approx(disk_floor({}), rel=1e-12)


def test_per_azimuth_fields_replay_chromatic_delay_line_phase() -> None:
    wavelengths_um = (1.50, 1.55, 1.60)
    aperture_positions_enu_m = np.array(
        [[0.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
        dtype=np.float64,
    )
    delay_diag = {
        "fiber_lengths_m": (1.0, 1.0),
        "n_eff": 1.0003,
        "phase_index_ref": 1.0003,
        "group_index_ref": 1.0004,
        "dispersion_ps_nm_km": 1.0,
        "attenuation_db_per_m": 0.0,
        "altitude_deg": 0.0,
        "fixed_delay_line_loss_db_per_telescope": None,
        "chromatic_control": "multi_wavelength_group_delay",
        "reference_wavelength_um": 1.55,
    }
    state = SimulationState.empty({"wavelength_grid_um": wavelengths_um})
    state = replace(
        state,
        scene=Scene(star_jmag=4.0, altitude_deg=0.0, azimuth_deg=0.0),
        geometry=Geometry(
            n_apertures=2,
            aperture_positions_enu_m=aperture_positions_enu_m,
        ),
        throughput=Throughput(
            wavelength_grid_um=wavelengths_um,
            transmission_fraction=(1.0, 1.0, 1.0),
            components={
                "delay_line_loss": {
                    "wavelength_grid_um": wavelengths_um,
                    "transmission_fraction": (1.0, 1.0, 1.0),
                    "per_aperture": delay_diag,
                },
            },
        ),
    )

    fields = _per_azimuth_stellar_fields(
        state=state,
        base_field=np.ones(2, dtype=np.complex128),
        n_azimuths=1,
        n_apertures=2,
    )

    assert fields.shape == (1, 3, 2)
    np.testing.assert_allclose(fields[0, :, 0], 1.0 + 0.0j)
    assert fields[0, 1, 1] == pytest.approx(1.0 + 0.0j)
    assert abs(fields[0, 0, 1] - 1.0) > 1.0e-6
    assert abs(fields[0, 2, 1] - 1.0) > 1.0e-6


def test_planet_input_transport_scales_expand_delay_loss() -> None:
    wavelengths_um = (1.25, 1.30)
    aperture_positions_enu_m = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    delay_diag = {
        "fiber_lengths_m": (1.0, 1.0),
        "n_eff": 1.0,
        "phase_index_ref": 1.0,
        "group_index_ref": 1.0,
        "dispersion_ps_nm_km": 0.0,
        "attenuation_db_per_m": 0.0,
        "altitude_deg": 0.0,
        "fixed_delay_line_loss_db_per_telescope": (0.0, 6.0),
        "chromatic_control": "none",
        "reference_wavelength_um": 1.25,
    }
    state = SimulationState.empty({"wavelength_grid_um": wavelengths_um})
    state = replace(
        state,
        scene=Scene(star_jmag=4.0, altitude_deg=0.0, azimuth_deg=90.0),
        geometry=Geometry(
            n_apertures=2,
            aperture_positions_enu_m=aperture_positions_enu_m,
        ),
        throughput=Throughput(
            wavelength_grid_um=wavelengths_um,
            transmission_fraction=(1.0, 1.0),
            components={
                "delay_line_loss": {
                    "wavelength_grid_um": wavelengths_um,
                    "transmission_fraction": (1.0, 1.0),
                    "per_aperture": delay_diag,
                },
            },
        ),
    )

    scales = _planet_input_transport_scales(
        state=state,
        n_azimuths=1,
        n_modes=2,
        n_wl=2,
        n_apertures=2,
    )

    assert scales is not None
    assert scales.shape == (1, 2, 2)
    np.testing.assert_allclose(scales[0, :, 0], 1.0)
    np.testing.assert_allclose(scales[0, :, 1], np.sqrt(10.0 ** (-0.6)))


def test_planet_input_transport_scales_keep_attenuator_chromatic_phase() -> None:
    wavelengths_um = (1.50, 1.55, 1.60)
    aperture_positions_enu_m = np.array(
        [[0.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
        dtype=np.float64,
    )
    delay_diag = {
        "fiber_lengths_m": (1.0, 1.0),
        "n_eff": 1.0003,
        "phase_index_ref": 1.0003,
        "group_index_ref": 1.0004,
        "dispersion_ps_nm_km": 1.0,
        "attenuation_db_per_m": 0.0,
        "altitude_deg": 0.0,
        "fixed_delay_line_loss_db_per_telescope": (2.0, 2.0),
        "fixed_delay_line_loss_mode": "attenuator",
        "chromatic_control": "multi_wavelength_group_delay",
        "reference_wavelength_um": 1.55,
    }
    state = SimulationState.empty({"wavelength_grid_um": wavelengths_um})
    state = replace(
        state,
        scene=Scene(star_jmag=4.0, altitude_deg=0.0, azimuth_deg=0.0),
        geometry=Geometry(
            n_apertures=2,
            aperture_positions_enu_m=aperture_positions_enu_m,
        ),
        throughput=Throughput(
            wavelength_grid_um=wavelengths_um,
            transmission_fraction=(1.0, 1.0, 1.0),
            components={
                "delay_line_loss": {
                    "wavelength_grid_um": wavelengths_um,
                    "transmission_fraction": (1.0, 1.0, 1.0),
                    "per_aperture": delay_diag,
                },
            },
        ),
    )

    scales = _planet_input_transport_scales(
        state=state,
        n_azimuths=2,
        n_modes=2,
        n_wl=3,
        n_apertures=2,
    )

    assert scales is not None
    expected_amp = np.sqrt(10.0 ** (-0.2))
    np.testing.assert_allclose(np.abs(scales), expected_amp)
    assert not np.allclose(scales[0], scales[1])
    assert abs(scales[0, 0, 1] - expected_amp) > 1.0e-6


def test_detection_curve_applies_delay_transport_to_planet_throughput(
    tmp_path: Path,
) -> None:
    n_modes = 2
    wavelengths_um = (1.25,)
    state = _keck_like_state(
        n_modes=n_modes,
        wavelengths_um=wavelengths_um,
        star_diameter_mas=0.0,
        n_apertures=2,
        u_stack=_identity_u_stack(n_modes, 1),
        positions=np.zeros((n_modes, 2), dtype=np.float64),
        strehl=1.0,
        piston_opd_nm_total=0.0,
    )
    geometry = replace(
        state.geometry,
        aperture_positions_enu_m=np.array(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
    )
    throughput = Throughput(
        wavelength_grid_um=wavelengths_um,
        transmission_fraction=(1.0,),
        components={
            "delay_line_loss": {
                "wavelength_grid_um": wavelengths_um,
                "transmission_fraction": (1.0,),
                "per_aperture": {
                    "fiber_lengths_m": (1.0, 1.0),
                    "n_eff": 1.0,
                    "phase_index_ref": 1.0,
                    "group_index_ref": 1.0,
                    "dispersion_ps_nm_km": 0.0,
                    "attenuation_db_per_m": 0.0,
                    "altitude_deg": 0.0,
                    "fixed_delay_line_loss_db_per_telescope": (0.0, 6.0),
                    "chromatic_control": "none",
                    "reference_wavelength_um": 1.25,
                },
            },
        },
    )
    state = replace(state, geometry=geometry, throughput=throughput)
    stage = DetectionCurve(
        params={
            "n_bright_ports": 1,
            "n_dark_ports": 1,
            "separations_mas": (0.0,),
            "include_angular_coupling": False,
            "n_mc_realizations": 0,
            "n_azimuths": 1,
        },
        context=_context(tmp_path),
    )

    out = stage.apply(state)
    payload = out.results.components["sensitivity.detection_curve"]

    assert payload["planet_delay_transport_applied"] is True
    assert payload["planet_throughputs"][0] == pytest.approx(
        10.0 ** (-0.6) / (1.0 + 10.0 ** (-0.6)),
        rel=1.0e-12,
    )


def test_detection_curve_uses_per_pol_arm_deployed_unitaries(tmp_path: Path) -> None:
    n_modes = 2
    wavelengths_um = (1.25,)
    shared = np.eye(n_modes, dtype=np.complex128)[:, :, np.newaxis]
    hadamard = (
        np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
        / np.sqrt(2.0)
    )
    per_arm = np.stack([hadamard, hadamard], axis=0)[:, :, :, np.newaxis]
    state = _keck_like_state(
        n_modes=n_modes,
        wavelengths_um=wavelengths_um,
        star_diameter_mas=0.0,
        n_apertures=1,
        u_stack=shared,
        positions=np.zeros((n_modes, 2), dtype=np.float64),
        strehl=1.0,
        piston_opd_nm_total=0.0,
    )
    components = dict(state.results.components)
    components["chip.kernel_mzi_mesh.test"] = {
        **components["chip.kernel_mzi_mesh.test"],
        "deployment_mode": "per_pol_arm",
        "chip_channel_labels": ("A", "B"),
        "u_stack_per_arm": per_arm,
    }
    state = replace(
        state,
        field=Field(
            amplitudes=np.ones((n_modes, len(wavelengths_um), 2), dtype=np.complex128),
            wavelength_grid_um=wavelengths_um,
        ),
        results=Results(components=components),
    )
    stage = DetectionCurve(
        params={
            "n_bright_ports": 1,
            "n_dark_ports": 1,
            "separations_mas": (1.0,),
            "include_angular_coupling": False,
            "n_mc_realizations": 0,
        },
        context=_context(tmp_path),
    )

    out = stage.apply(state)
    payload = out.results.components["sensitivity.detection_curve"]

    assert payload["chip_deployment_mode"] == "per_pol_arm"
    assert tuple(payload["chip_channel_labels"]) == ("A", "B")
    np.testing.assert_allclose(payload["unitary_arm_weights"], (0.5, 0.5))
    np.testing.assert_allclose(payload["planet_throughputs"], 0.0, atol=1e-12)
    assert payload["realized_null_depth"] == pytest.approx(0.0, abs=1e-12)
    assert tuple(payload["realized_null_depth_per_arm"]) == pytest.approx(
        (0.0, 0.0),
        abs=1e-12,
    )


def test_detection_curve_per_pol_arm_uses_geometry_phase_frame(
    tmp_path: Path,
) -> None:
    n_modes = 2
    wavelengths_um = (1.25,)
    shared = np.eye(n_modes, dtype=np.complex128)[:, :, np.newaxis]
    hadamard = (
        np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
        / np.sqrt(2.0)
    )
    correction = np.array(
        [
            [1.0, np.exp(-1j * np.pi / 3.0)],
            [np.exp(-1j * np.pi / 2.0), np.exp(1j * np.pi / 4.0)],
        ],
        dtype=np.complex128,
    )
    per_arm = (
        hadamard[np.newaxis, :, :, np.newaxis]
        * correction.T[:, np.newaxis, :, np.newaxis]
    )
    state = _keck_like_state(
        n_modes=n_modes,
        wavelengths_um=wavelengths_um,
        star_diameter_mas=0.0,
        n_apertures=1,
        u_stack=shared,
        positions=np.zeros((n_modes, 2), dtype=np.float64),
        strehl=1.0,
        piston_opd_nm_total=0.0,
    )
    components = dict(state.results.components)
    components["chip.kernel_mzi_mesh.test"] = {
        **components["chip.kernel_mzi_mesh.test"],
        "deployment_mode": "per_pol_arm",
        "chip_channel_labels": ("A", "B"),
        "input_phase_correction": correction,
        "u_stack_per_arm": per_arm,
    }
    state = replace(
        state,
        field=Field(
            amplitudes=np.ones((n_modes, len(wavelengths_um), 2), dtype=np.complex128),
            wavelength_grid_um=wavelengths_um,
        ),
        results=Results(components=components),
    )
    stage = DetectionCurve(
        params={
            "n_bright_ports": 1,
            "n_dark_ports": 1,
            "separations_mas": (1.0,),
            "include_angular_coupling": False,
            "n_mc_realizations": 0,
        },
        context=_context(tmp_path),
    )

    out = stage.apply(state)
    payload = out.results.components["sensitivity.detection_curve"]

    assert payload["input_phase_correction_applied"] is True
    np.testing.assert_allclose(payload["planet_throughputs"], 0.0, atol=1e-12)
    assert payload["realized_null_depth"] == pytest.approx(0.0, abs=1e-12)


def test_detection_curve_uses_upstream_injection_coupling(
    tmp_path: Path,
) -> None:
    state = _keck_like_state(
        n_modes=4,
        wavelengths_um=(1.25,),
        positions=np.zeros((4, 2), dtype=np.float64),
        strehl=1.0,
        piston_opd_nm_total=0.0,
    )
    state = replace(
        state,
        throughput=Throughput(
            wavelength_grid_um=(1.25,),
            transmission_fraction=(1.0,),
            components={
                "injection_ideal": {
                    "transmission_fraction": (0.25,),
                },
            },
        ),
    )
    stage = DetectionCurve(
        params={
            "n_bright_ports": 1,
            "n_dark_ports": 3,
            "separations_mas": (1.0,),
            "include_angular_coupling": False,
            "n_mc_realizations": 0,
        },
        context=_context(tmp_path),
    )

    out = stage.apply(state)
    payload = out.results.components["sensitivity.detection_curve"]

    np.testing.assert_allclose(payload["coupling_per_sub"], 0.25)
    assert (
        payload["coupling_source"]
        == "throughput:injection_ideal.transmission_fraction"
    )


def _state_with_detector_component(
    *,
    bright_rate_hz: float = 1.0e8,
    dark_rate_hz_per_port: float = 2.0e3,
    spectro_rate_hz: float = 5.0e2,
) -> SimulationState:
    """Keck-like state with a synthetic detector component for rate tests."""

    state = _keck_like_state()
    components = dict(state.results.components)
    port_rates = np.array(
        [bright_rate_hz, bright_rate_hz]
        + [dark_rate_hz_per_port, dark_rate_hz_per_port]
        + [spectro_rate_hz],
        dtype=np.float64,
    )
    port_labels = ("bright_0", "bright_1", "dark_0", "dark_1", "spectro_0")
    components["detector.mkid"] = {
        "port_rates": port_rates,
        "port_labels": port_labels,
        "integration_time_s": 3600.0,
    }
    return replace(state, results=Results(components=components))


def test_detection_curve_publishes_effective_leakage_rates(tmp_path: Path) -> None:
    """Effective leakage = effective_null * total stellar rate; static = dark total."""

    bright_rate_hz = 1.0e8
    dark_rate_hz_per_port = 2.0e3
    spectro_rate_hz = 5.0e2
    state = _state_with_detector_component(
        bright_rate_hz=bright_rate_hz,
        dark_rate_hz_per_port=dark_rate_hz_per_port,
        spectro_rate_hz=spectro_rate_hz,
    )
    stage = DetectionCurve(
        params={
            "n_bright_ports": 2,
            "n_dark_ports": 2,
            "separations_mas": (1.0, 10.0, 100.0),
            "n_mc_realizations": 0,
        },
        context=_context(tmp_path),
    )
    out = stage.apply(state)
    payload = out.results.components["sensitivity.detection_curve"]

    effective_null = float(payload["effective_null_with_disk"])
    # total_stellar_rate_hz mirrors performance.standard: bright + dark + spectro.
    total_stellar_rate_hz = (
        2.0 * bright_rate_hz + 2.0 * dark_rate_hz_per_port + spectro_rate_hz
    )
    static_chip_leakage_hz = 2.0 * dark_rate_hz_per_port

    assert payload["leakage_rate_hz_effective"] == pytest.approx(
        effective_null * total_stellar_rate_hz
    )
    assert payload["leakage_rate_hz_static_chip"] == pytest.approx(
        static_chip_leakage_hz
    )
    assert payload["effective_to_static_leakage_ratio"] == pytest.approx(
        (effective_null * total_stellar_rate_hz) / static_chip_leakage_hz
    )


def test_detection_curve_off_zenith_pointing_foreshortens_fringe(
    tmp_path: Path,
) -> None:
    """Off-zenith pointing along a baseline shifts planet throughput vs sep."""

    n_modes = 2
    # Two apertures separated along North (y); pointing North at low altitude
    # foreshortens this baseline, widening the on-sky fringe.
    positions = np.array([[0.0, 40.0], [0.0, -40.0]], dtype=np.float64)
    # 2-mode 50/50 nuller: row 0 bright sum, row 1 differential dark port.
    u_stack = (
        np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
        / math.sqrt(2.0)
    )[:, :, np.newaxis]

    def _throughputs(altitude_deg: float | None) -> np.ndarray:
        state = _keck_like_state(
            n_modes=n_modes,
            u_stack=u_stack,
            wavelengths_um=(1.25,),
            positions=positions,
            strehl=1.0,
            piston_opd_nm_total=0.0,
            star_diameter_mas=0.0,
        )
        state = replace(
            state,
            scene=replace(
                state.scene,
                altitude_deg=altitude_deg,
                azimuth_deg=None if altitude_deg is None else 0.0,
            ),
        )
        stage = DetectionCurve(
            params={
                "n_bright_ports": 1,
                "n_dark_ports": 1,
                "separations_mas": (5.0, 10.0, 20.0),
                "include_angular_coupling": False,
                "n_mc_realizations": 0,
            },
            context=_context(tmp_path),
        )
        out = stage.apply(state)
        return np.asarray(
            out.results.components["sensitivity.detection_curve"][
                "planet_throughputs"
            ],
            dtype=np.float64,
        )

    zenith = _throughputs(None)
    low_alt = _throughputs(30.0)
    # The N-S baseline foreshortens by sin(30deg) = 0.5, so the fringe pattern
    # changes: the throughput curve must differ from the zenith case.
    assert not np.allclose(zenith, low_alt, atol=1e-6)


def test_detection_curve_leakage_rates_none_without_detector(tmp_path: Path) -> None:
    """Without an upstream detector the effective leakage rates are absent."""

    state = _keck_like_state()
    stage = DetectionCurve(
        params={
            "n_bright_ports": 2,
            "n_dark_ports": 2,
            "separations_mas": (1.0, 10.0, 100.0),
            "n_mc_realizations": 0,
        },
        context=_context(tmp_path),
    )
    out = stage.apply(state)
    payload = out.results.components["sensitivity.detection_curve"]

    assert payload["leakage_rate_hz_effective"] is None
    assert payload["leakage_rate_hz_static_chip"] is None
    assert payload["effective_to_static_leakage_ratio"] is None


def test_detection_curve_uses_pupil_modes_mode_diameter(tmp_path: Path) -> None:
    mode_diameter_m = 1.75
    geometry = _keck_like_state(n_modes=4, n_apertures=1).geometry.add_component(
        "pupil_modes",
        {"mode_diameter_m": mode_diameter_m},
    )
    state = replace(_keck_like_state(n_modes=4, n_apertures=1), geometry=geometry)
    stage = DetectionCurve(
        params={
            "n_bright_ports": 1,
            "n_dark_ports": 3,
            "separations_mas": (10.0,),
            "n_mc_realizations": 0,
        },
        context=_context(tmp_path),
    )

    out = stage.apply(state)

    payload = out.results.components["sensitivity.detection_curve"]
    np.testing.assert_allclose(
        payload["sub_diameters_m"],
        np.full(4, mode_diameter_m),
    )
    np.testing.assert_allclose(
        payload["sub_areas_m2"],
        np.full(4, math.pi * (mode_diameter_m / 2.0) ** 2),
    )


def test_detection_curve_uses_pupil_modes_mode_area_when_present(
    tmp_path: Path,
) -> None:
    mode_diameter_m = 1.75
    mode_area_m2 = 12.5
    geometry = _keck_like_state(n_modes=4, n_apertures=1).geometry.add_component(
        "pupil_modes",
        {"mode_diameter_m": mode_diameter_m, "mode_area_m2": mode_area_m2},
    )
    state = replace(_keck_like_state(n_modes=4, n_apertures=1), geometry=geometry)
    stage = DetectionCurve(
        params={
            "n_bright_ports": 1,
            "n_dark_ports": 3,
            "separations_mas": (10.0,),
            "n_mc_realizations": 0,
        },
        context=_context(tmp_path),
    )

    out = stage.apply(state)

    payload = out.results.components["sensitivity.detection_curve"]
    np.testing.assert_allclose(
        payload["sub_diameters_m"],
        np.full(4, mode_diameter_m),
    )
    np.testing.assert_allclose(
        payload["sub_areas_m2"],
        np.full(4, mode_area_m2),
    )


def test_detection_curve_excludes_spectro_port_from_dark_indices(
    tmp_path: Path,
) -> None:
    n_modes = 4
    state = _keck_like_state(
        n_modes=n_modes,
        u_stack=_identity_u_stack(n_modes, 1),
        wavelengths_um=(1.25,),
        positions=np.zeros((n_modes, 2)),
        strehl=1.0,
        piston_opd_nm_total=0.0,
    )
    stage = DetectionCurve(
        params={
            "n_bright_ports": 1,
            "n_dark_ports": 2,
            "n_spectro_ports": 1,
            "separations_mas": (1.0,),
            "include_angular_coupling": False,
            "n_mc_realizations": 0,
        },
        context=_context(tmp_path),
    )

    out = stage.apply(state)
    payload = out.results.components["sensitivity.detection_curve"]

    np.testing.assert_array_equal(payload["dark_indices"], np.array([1, 2]))
    assert payload["n_spectro_ports"] == 1
    assert payload["planet_throughputs"][0] == pytest.approx(2.0 / n_modes)


def test_detection_curve_runs_mc_realized_null(tmp_path: Path) -> None:
    """With finite piston OPD and MC realizations, null depth is finite."""

    n_modes = 4
    # Build a unitary that routes in-phase inputs to bright port 0; this
    # gives a non-trivial null structure where piston jitter actually
    # contributes to leakage.
    rng = np.random.default_rng(0)
    A = rng.normal(size=(n_modes, n_modes)) + 1j * rng.normal(size=(n_modes, n_modes))
    q, _ = np.linalg.qr(A)
    # Put the (1,1,1,1)/2 vector in the first column so it's routed to port 0.
    bright_col = np.ones(n_modes, dtype=np.complex128) / math.sqrt(n_modes)
    cols = q.copy()
    cols[:, 0] = bright_col
    cols, _ = np.linalg.qr(cols)
    u = cols.conj().T  # rows are outputs
    u_stack = u[:, :, np.newaxis]

    state = _keck_like_state(
        n_modes=n_modes,
        u_stack=u_stack,
        wavelengths_um=(1.25,),
        positions=np.zeros((n_modes, 2)),
        strehl=0.93,
        piston_opd_nm_total=1.35,
    )
    stage = DetectionCurve(
        params={
            "n_bright_ports": 1,
            "n_dark_ports": n_modes - 1,
            "separations_mas": (1.0, 10.0),
            "n_mc_realizations": 500,
            "mc_seed": 0,
        },
        context=_context(tmp_path),
    )
    out = stage.apply(state)
    payload = out.results.components["sensitivity.detection_curve"]
    # MC realized null should be small (~ sigma_phi^2 ~ 7e-5 here) but nonzero.
    assert 1e-7 < payload["realized_null_depth"] < 1e-2
    assert payload["null_instability"] > 0.0
    # Phase 1: 500 back-compat realizations distributed across 6 default
    # azimuths gives floor(500/6) * 6 = 498 samples.
    assert payload["n_mc_samples"] == 498


def _ft_state_with_common_mode(
    *,
    n_modes: int,
    u_stack: np.ndarray,
    iid_per_mode_nm: float,
    common_differential_nm: float,
    opd_budget_scalar_nm: float,
) -> SimulationState:
    """Keck-like state carrying an injected fringe-tracking residual split.

    ``opd_budget`` holds the per-sub scalar piston (used by the scalar MC path)
    and ``results`` holds the IID / common differential split that the
    realized-null covariance handoff reads.
    """

    state = _keck_like_state(
        n_modes=n_modes,
        u_stack=u_stack,
        wavelengths_um=(1.25,),
        positions=np.zeros((n_modes, 2)),
        strehl=0.93,
        piston_opd_nm_total=1.35,
    )
    opd_budget = OPDBudget(
        components={
            "fringe_tracking.closed_loop_residual": OPDComponent(
                rms_opd_nm=float(opd_budget_scalar_nm),
                reduction_stage="fringe_tracking.closed_loop",
            )
        },
        total_rms_opd_nm=float(opd_budget_scalar_nm),
    )
    components = dict(state.results.components)
    components["fringe_tracking.closed_loop"] = {
        "residual_rms_opd_nm": float(opd_budget_scalar_nm),
        "residual_iid_per_mode_nm": float(iid_per_mode_nm),
        "residual_common_differential_nm": float(common_differential_nm),
    }
    return replace(
        state,
        opd_budget=opd_budget,
        results=Results(components=components),
    )


def _two_telescope_nulling_unitary() -> np.ndarray:
    """4-mode, 2-telescope Hadamard nuller: bright sum + telescope-diff dark.

    Row 0 is the bright sum port; row 1 is the telescope-differential dark port
    that is maximally sensitive to inter-telescope (common-mode) piston; rows 2
    and 3 are within-telescope difference dark ports.
    """

    u = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
        ],
        dtype=np.complex128,
    )
    for row_index in range(u.shape[0]):
        u[row_index] /= np.linalg.norm(u[row_index])
    return u[:, :, np.newaxis]


def test_detection_curve_common_mode_exceeds_equal_rms_iid(tmp_path: Path) -> None:
    """A common differential FT residual must not average down across modes.

    For equal *per-sub* piston RMS, the rank-1 inter-telescope common mode
    leaves a deeper realized null floor than a purely independent per-mode
    piston (which partially averages away through the dark-port combination).
    This is the entire point of routing the common term through the covariance
    MC. The per-sub RMS of the rank-1 covariance diagonal is
    ``sqrt(iid^2 + common^2 / 4)`` (each telescope carries +/- common / 2), so
    the matched IID scalar uses the same per-sub value.
    """

    n_modes = 4
    u_stack = _two_telescope_nulling_unitary()

    common_nm = 30.0
    iid_nm = 5.0
    equal_per_sub_nm = math.sqrt(iid_nm**2 + common_nm**2 / 4.0)

    common_state = _ft_state_with_common_mode(
        n_modes=n_modes,
        u_stack=u_stack,
        iid_per_mode_nm=iid_nm,
        common_differential_nm=common_nm,
        opd_budget_scalar_nm=math.sqrt(iid_nm**2 + common_nm**2),
    )
    iid_state = _ft_state_with_common_mode(
        n_modes=n_modes,
        u_stack=u_stack,
        iid_per_mode_nm=equal_per_sub_nm,
        common_differential_nm=0.0,
        opd_budget_scalar_nm=equal_per_sub_nm,
    )

    def _run(state: SimulationState) -> dict[str, object]:
        stage = DetectionCurve(
            params={
                "n_bright_ports": 1,
                "n_dark_ports": n_modes - 1,
                "separations_mas": (1.0, 10.0),
                "n_mc_realizations": 40000,
                "mc_seed": 0,
            },
            context=_context(tmp_path),
        )
        return stage.apply(state).results.components["sensitivity.detection_curve"]

    common_payload = _run(common_state)
    iid_payload = _run(iid_state)

    assert common_payload["realized_null_depth"] > iid_payload["realized_null_depth"]
    # The common-mode case routes through the realized-null covariance MC; the
    # equal-RMS IID case stays on the scalar per-sub path.
    assert common_payload["effective_systematic_bandwidth_diagnostics"][
        "piston_covariance_used_in_realized_null_mc"
    ]
    assert not iid_payload["effective_systematic_bandwidth_diagnostics"][
        "piston_covariance_used_in_realized_null_mc"
    ]


def test_detection_curve_override_preserves_mc_null_instability(tmp_path: Path) -> None:
    """Regression test for Tier 1.3 fix.

    ``effective_null_override`` must replace the null *depth* only and
    must NOT zero ``null_instability``. The MC's CV of dark-port leakage
    is real physics; downstream f_eff handling decides whether it lights
    up the speckle term, but the value itself is always emitted.

    Pre-fix nullsim zeroed ``null_instability_out`` whenever the override
    was set without an accompanying ``effective_systematic_bandwidth_hz``
    override, silently dropping the dominant noise term in Keck-only.
    """

    n_modes = 4
    rng = np.random.default_rng(0)
    A = rng.normal(size=(n_modes, n_modes)) + 1j * rng.normal(size=(n_modes, n_modes))
    q, _ = np.linalg.qr(A)
    bright_col = np.ones(n_modes, dtype=np.complex128) / math.sqrt(n_modes)
    cols = q.copy()
    cols[:, 0] = bright_col
    cols, _ = np.linalg.qr(cols)
    u_stack = cols.conj().T[:, :, np.newaxis]

    state = _keck_like_state(
        n_modes=n_modes,
        u_stack=u_stack,
        wavelengths_um=(1.25,),
        positions=np.zeros((n_modes, 2)),
        strehl=0.93,
        piston_opd_nm_total=1.35,
    )

    override_null = 1.95e-5
    stage = DetectionCurve(
        params={
            "n_bright_ports": 1,
            "n_dark_ports": n_modes - 1,
            "separations_mas": (1.0, 10.0),
            "n_mc_realizations": 500,
            "mc_seed": 0,
            "effective_null_override": override_null,
        },
        context=_context(tmp_path),
    )
    out = stage.apply(state)
    payload = out.results.components["sensitivity.detection_curve"]

    # The override REPLACES effective_null_with_disk with the floor.
    assert payload["effective_null_with_disk"] == pytest.approx(override_null)
    # The MC fluctuation MUST survive the override. With no trim bias the
    # RMS is the MC's, and null_instability is it over the effective null
    # in force, so effective null x null_instability is the RMS.
    assert payload["null_fluctuation_rms"] > 0.0
    assert payload["null_fluctuation_rms"] == pytest.approx(
        payload["null_fluctuation_rms_mc"], rel=1e-12,
    )
    assert payload["null_instability"] == pytest.approx(
        payload["null_fluctuation_rms"] / override_null, rel=1e-12,
    )
    assert payload["null_instability_mc"] == pytest.approx(
        payload["null_fluctuation_rms_mc"] / payload["realized_null_depth"], rel=1e-12,
    )
    # When the bandwidth override is unset, the payload field is None so
    # the performance stage falls back to its own default f_eff.
    assert payload["effective_systematic_bandwidth_hz"] is None


def test_detection_curve_applies_residual_starlight_after_null_override(
    tmp_path: Path,
) -> None:
    n_modes = 2
    nuller = np.array(
        [
            [1.0, 1.0],
            [1.0, -1.0],
        ],
        dtype=np.complex128,
    ) / math.sqrt(2.0)
    state = _keck_like_state(
        n_modes=n_modes,
        n_apertures=1,
        u_stack=nuller[:, :, np.newaxis],
        wavelengths_um=(1.25,),
        star_diameter_mas=0.0,
        positions=np.zeros((n_modes, 2)),
        strehl=1.0,
        piston_opd_nm_total=0.0,
    )
    state = replace(
        state,
        scene=replace(state.scene, planet_separation_mas=10.0),
    )
    stage = DetectionCurve(
        params={
            "n_bright_ports": 1,
            "n_dark_ports": 1,
            "separations_mas": (1.0, 10.0, 100.0),
            "n_realizations_per_azimuth": 0,
            "include_angular_coupling": False,
            "effective_null_override": 1.0e-8,
            "residual_starlight_separations_mas": (1.0, 10.0, 100.0),
            "residual_starlight_contrast": (1.0e-9, 1.0e-7, 1.0e-9),
            "residual_starlight_combine_mode": "max",
        },
        context=_context(tmp_path),
    )

    payload = stage.apply(state).results.components["sensitivity.detection_curve"]

    assert payload["effective_null_before_residual_starlight"] == pytest.approx(
        1.0e-8
    )
    np.testing.assert_allclose(
        payload["effective_null_with_disk_by_separation"],
        [1.0e-8, 1.0e-7, 1.0e-8],
    )
    assert payload["effective_null_with_disk"] == pytest.approx(1.0e-7)


def test_setpoint_trim_terms_add_to_effective_null(tmp_path: Path) -> None:
    n_modes = 2
    nuller = np.array(
        [
            [1.0, 1.0],
            [1.0, -1.0],
        ],
        dtype=np.complex128,
    ) / math.sqrt(2.0)
    state = _keck_like_state(
        n_modes=n_modes,
        n_apertures=1,
        u_stack=nuller[:, :, np.newaxis],
        wavelengths_um=(1.25,),
        star_diameter_mas=0.0,
        positions=np.zeros((n_modes, 2)),
        strehl=1.0,
        piston_opd_nm_total=0.0,
    )
    stage = DetectionCurve(
        params={
            "n_bright_ports": 1,
            "n_dark_ports": 1,
            "separations_mas": (1.0, 10.0, 100.0),
            "n_realizations_per_azimuth": 0,
            "include_angular_coupling": False,
            "null_setpoint_residual_nm": 1.0,
            "setpoint_dither_rms_nm": 2.5,
        },
        context=_context(tmp_path),
    )

    payload = stage.apply(state).results.components["sensitivity.detection_curve"]
    null_setpoint_leakage = (2.0 * math.pi * 1.0e-9 / 1.25e-6) ** 2 / 4.0
    setpoint_dither_leakage = (2.0 * math.pi * 2.5e-9 / 1.25e-6) ** 2 / 4.0

    assert payload["null_setpoint_leakage"] == pytest.approx(
        null_setpoint_leakage,
        rel=1.0e-2,
    )
    assert payload["setpoint_dither_leakage"] == pytest.approx(
        setpoint_dither_leakage,
        rel=1.0e-2,
    )
    assert payload["effective_null_with_disk"] == pytest.approx(
        payload["realized_null_depth"]
        + payload["disk_null_floor"]
        + null_setpoint_leakage
        + setpoint_dither_leakage,
        rel=1.0e-6,
    )
    np.testing.assert_allclose(
        payload["effective_null_with_disk_by_separation"],
        payload["effective_null_with_disk"],
        rtol=1.0e-12,
    )


def test_effective_null_override_preserves_setpoint_trim_terms(
    tmp_path: Path,
) -> None:
    n_modes = 2
    nuller = np.array(
        [
            [1.0, 1.0],
            [1.0, -1.0],
        ],
        dtype=np.complex128,
    ) / math.sqrt(2.0)
    state = _keck_like_state(
        n_modes=n_modes,
        n_apertures=1,
        u_stack=nuller[:, :, np.newaxis],
        wavelengths_um=(1.25,),
        star_diameter_mas=0.0,
        positions=np.zeros((n_modes, 2)),
        strehl=1.0,
        piston_opd_nm_total=0.0,
    )
    override_null = 1.0e-8
    stage = DetectionCurve(
        params={
            "n_bright_ports": 1,
            "n_dark_ports": 1,
            "separations_mas": (1.0, 10.0, 100.0),
            "n_realizations_per_azimuth": 0,
            "include_angular_coupling": False,
            "effective_null_override": override_null,
            "null_setpoint_residual_nm": 1.0,
            "setpoint_dither_rms_nm": 2.5,
        },
        context=_context(tmp_path),
    )

    payload = stage.apply(state).results.components["sensitivity.detection_curve"]
    null_setpoint_leakage = (2.0 * math.pi * 1.0e-9 / 1.25e-6) ** 2 / 4.0
    setpoint_dither_leakage = (2.0 * math.pi * 2.5e-9 / 1.25e-6) ** 2 / 4.0
    expected_null = (
        override_null
        + null_setpoint_leakage
        + setpoint_dither_leakage
    )

    assert payload["effective_null_scalar_source"] == "effective_null_override"
    assert payload["null_setpoint_leakage"] == pytest.approx(null_setpoint_leakage)
    assert payload["setpoint_dither_leakage"] == pytest.approx(setpoint_dither_leakage)
    assert payload["effective_null_before_residual_starlight"] == pytest.approx(
        expected_null,
        rel=1.0e-12,
    )
    assert payload["effective_null_with_disk"] == pytest.approx(
        expected_null,
        rel=1.0e-12,
    )
    np.testing.assert_allclose(
        payload["effective_null_with_disk_by_separation"],
        expected_null,
        rtol=1.0e-12,
    )


def test_detection_curve_applies_flat_residual_starlight_floor(
    tmp_path: Path,
) -> None:
    n_modes = 2
    nuller = np.array(
        [
            [1.0, 1.0],
            [1.0, -1.0],
        ],
        dtype=np.complex128,
    ) / math.sqrt(2.0)
    state = _keck_like_state(
        n_modes=n_modes,
        n_apertures=1,
        u_stack=nuller[:, :, np.newaxis],
        wavelengths_um=(1.25,),
        star_diameter_mas=0.0,
        positions=np.zeros((n_modes, 2)),
        strehl=1.0,
        piston_opd_nm_total=0.0,
    )
    state = replace(
        state,
        scene=replace(state.scene, planet_separation_mas=10.0),
    )
    stage = DetectionCurve(
        params={
            "n_bright_ports": 1,
            "n_dark_ports": 1,
            "separations_mas": (1.0, 10.0, 100.0),
            "n_realizations_per_azimuth": 0,
            "include_angular_coupling": False,
            "residual_starlight_contrast": (1.0e-10,),
        },
        context=_context(tmp_path),
    )

    payload = stage.apply(state).results.components["sensitivity.detection_curve"]

    assert payload["effective_null_before_residual_starlight"] < 1.0e-20
    assert payload["effective_null_with_disk"] == pytest.approx(1.0e-10)
    np.testing.assert_allclose(payload["residual_starlight_contrast"], 1.0e-10)
    np.testing.assert_allclose(
        payload["effective_null_with_disk_by_separation"],
        1.0e-10,
    )
    assert payload["residual_starlight_applied"] is True
    assert payload["residual_starlight_combine_mode"] == "max"
    assert (
        payload["effective_null_scalar_source"]
        == "interpolated at scene.planet_separation_mas"
    )


def test_detection_curve_interpolates_residual_starlight_csv_and_declares_dependency(
    tmp_path: Path,
) -> None:
    table = tmp_path / "residual_starlight.csv"
    table.write_text(
        "separation_mas,contrast\n"
        "100,1e-11\n"
        "1,1e-9\n"
        "10,1e-10\n",
        encoding="utf-8",
    )
    n_modes = 2
    nuller = np.array(
        [
            [1.0, 1.0],
            [1.0, -1.0],
        ],
        dtype=np.complex128,
    ) / math.sqrt(2.0)
    state = _keck_like_state(
        n_modes=n_modes,
        n_apertures=1,
        u_stack=nuller[:, :, np.newaxis],
        wavelengths_um=(1.25,),
        star_diameter_mas=0.0,
        positions=np.zeros((n_modes, 2)),
        strehl=1.0,
        piston_opd_nm_total=0.0,
    )
    state = replace(
        state,
        scene=replace(state.scene, planet_separation_mas=10.0),
    )
    stage = DetectionCurve(
        params={
            "n_bright_ports": 1,
            "n_dark_ports": 1,
            "separations_mas": (1.0, 10.0, 100.0),
            "n_realizations_per_azimuth": 0,
            "include_angular_coupling": False,
            "residual_starlight_table_path": str(table),
            "residual_starlight_combine_mode": "replace",
        },
        context=_context(tmp_path),
    )

    dependencies = stage.external_dependencies()
    payload = stage.apply(state).results.components["sensitivity.detection_curve"]

    assert len(dependencies) == 1
    assert dependencies[0].source == "sensitivity.residual_starlight_table"
    assert dependencies[0].path == table
    assert dependencies[0].content_hash is not None
    np.testing.assert_allclose(
        payload["residual_starlight_contrast"],
        [1.0e-9, 1.0e-10, 1.0e-11],
    )
    np.testing.assert_allclose(
        payload["effective_null_with_disk_by_separation"],
        [1.0e-9, 1.0e-10, 1.0e-11],
    )
    assert payload["effective_null_with_disk"] == pytest.approx(1.0e-10)
    assert payload["residual_starlight_table_path"] == str(table)


def test_detection_curve_resolves_residual_table_relative_to_config(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    table_dir = config_dir / "tables"
    table_dir.mkdir(parents=True)
    table = table_dir / "residual_starlight.csv"
    table.write_text(
        "separation_mas,contrast\n"
        "1,1e-9\n"
        "10,1e-10\n",
        encoding="utf-8",
    )
    n_modes = 2
    nuller = np.array(
        [
            [1.0, 1.0],
            [1.0, -1.0],
        ],
        dtype=np.complex128,
    ) / math.sqrt(2.0)
    state = _keck_like_state(
        n_modes=n_modes,
        n_apertures=1,
        u_stack=nuller[:, :, np.newaxis],
        wavelengths_um=(1.25,),
        star_diameter_mas=0.0,
        positions=np.zeros((n_modes, 2)),
        strehl=1.0,
        piston_opd_nm_total=0.0,
    )
    state = replace(
        state,
        scene=replace(state.scene, planet_separation_mas=10.0),
    )
    stage = DetectionCurve(
        params={
            "n_bright_ports": 1,
            "n_dark_ports": 1,
            "separations_mas": (1.0, 10.0),
            "n_realizations_per_azimuth": 0,
            "include_angular_coupling": False,
            "residual_starlight_table_path": "tables/residual_starlight.csv",
        },
        context=_context(tmp_path, source_path=config_dir / "config.toml"),
    )

    dependencies = stage.external_dependencies()
    payload = stage.apply(state).results.components["sensitivity.detection_curve"]

    assert dependencies[0].path == table.resolve()
    assert payload["residual_starlight_table_path"] == str(table.resolve())
    assert payload["effective_null_with_disk"] == pytest.approx(1.0e-10)


def test_detection_curve_interpolates_wavelength_resolved_residual_starlight_csv(
    tmp_path: Path,
) -> None:
    table = tmp_path / "residual_starlight_surface.csv"
    table.write_text(
        "separation_mas,wavelength_um,contrast\n"
        "1,0.7,1e-9\n"
        "1,0.9,3e-9\n"
        "10,0.7,1e-10\n"
        "10,0.9,3e-10\n",
        encoding="utf-8",
    )
    n_modes = 2
    nuller = np.array(
        [
            [1.0, 1.0],
            [1.0, -1.0],
        ],
        dtype=np.complex128,
    ) / math.sqrt(2.0)
    u_stack = np.repeat(nuller[:, :, np.newaxis], 3, axis=2)
    state = _keck_like_state(
        n_modes=n_modes,
        n_apertures=1,
        u_stack=u_stack,
        wavelengths_um=(0.75, 0.80, 0.85),
        star_diameter_mas=0.0,
        positions=np.zeros((n_modes, 2)),
        strehl=1.0,
        piston_opd_nm_total=0.0,
    )
    state = replace(
        state,
        scene=replace(state.scene, planet_separation_mas=1.0),
    )
    stage = DetectionCurve(
        params={
            "n_bright_ports": 1,
            "n_dark_ports": 1,
            "separations_mas": (1.0, 10.0),
            "n_realizations_per_azimuth": 0,
            "include_angular_coupling": False,
            "residual_starlight_table_path": str(table),
            "residual_starlight_combine_mode": "replace",
        },
        context=_context(tmp_path),
    )

    payload = stage.apply(state).results.components["sensitivity.detection_curve"]

    expected_surface = np.array(
        [
            [1.5e-9, 2.0e-9, 2.5e-9],
            [1.5e-10, 2.0e-10, 2.5e-10],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        payload["residual_starlight_contrast_by_wavelength"],
        expected_surface,
    )
    np.testing.assert_allclose(payload["residual_starlight_contrast"], [2e-9, 2e-10])
    np.testing.assert_allclose(payload["residual_starlight_wavelengths_um"], [0.75, 0.8, 0.85])
    np.testing.assert_allclose(payload["residual_starlight_wavelength_weights"], [1 / 3] * 3)
    assert payload["residual_starlight_interpolation"].startswith("bilinear")
    assert payload["effective_null_with_disk"] == pytest.approx(2.0e-9)


def test_detection_curve_pointing_jitter_smooths_residual_starlight_profile(
    tmp_path: Path,
) -> None:
    n_modes = 2
    nuller = np.array(
        [
            [1.0, 1.0],
            [1.0, -1.0],
        ],
        dtype=np.complex128,
    ) / math.sqrt(2.0)
    state = _keck_like_state(
        n_modes=n_modes,
        n_apertures=1,
        u_stack=nuller[:, :, np.newaxis],
        wavelengths_um=(1.25,),
        star_diameter_mas=0.0,
        positions=np.zeros((n_modes, 2)),
        strehl=1.0,
        piston_opd_nm_total=0.0,
    )
    state = replace(
        state,
        scene=replace(state.scene, planet_separation_mas=1.0),
        wavefront=state.wavefront.add_component(
            "wavefront.space_residual",
            {"pointing_jitter_rms_mas": 1.0},
        ),
    )
    stage = DetectionCurve(
        params={
            "n_bright_ports": 1,
            "n_dark_ports": 1,
            "separations_mas": (0.0, 1.0, 2.0),
            "n_realizations_per_azimuth": 0,
            "include_angular_coupling": False,
            "residual_starlight_separations_mas": (0.0, 1.0, 2.0),
            "residual_starlight_contrast": (0.0, 1.0e-10, 0.0),
            "residual_starlight_combine_mode": "replace",
        },
        context=_context(tmp_path),
    )

    payload = stage.apply(state).results.components["sensitivity.detection_curve"]

    assert payload["pointing_jitter_applied"] is True
    assert payload["pointing_jitter_rms_mas"] == pytest.approx(1.0)
    assert (
        payload["pointing_jitter_source"]
        == "wavefront:wavefront.space_residual.pointing_jitter_rms_mas"
    )
    np.testing.assert_allclose(
        payload["residual_starlight_contrast_before_pointing_jitter"],
        [0.0, 1.0e-10, 0.0],
    )
    smoothed = np.asarray(payload["residual_starlight_contrast"], dtype=np.float64)
    assert smoothed[1] < 1.0e-10
    assert smoothed[0] > 0.0
    assert smoothed[2] > 0.0
    np.testing.assert_allclose(
        payload["effective_null_with_disk_by_separation"],
        smoothed,
    )



def test_detection_curve_auto_amplitude_servo_derives_sigma_i_and_f_eff(
    tmp_path: Path,
) -> None:
    """Regression test for Tier 2.3b/2.5b wiring.

    With ``auto_amplitude_servo_residual=True``, σ_I is derived from
    physics (``amplitude_servo_residual_sigma_i``) and the effective
    f_eff is the variance-weighted blend
    (``effective_systematic_bandwidth_hz``) — both should appear in
    the payload.
    """
    from nullsim.physics.scintillation import amplitude_servo_residual_sigma_i
    from nullsim.physics.systematic_bandwidth import (
        effective_systematic_bandwidth_hz,
    )

    n_modes = 4
    rng = np.random.default_rng(0)
    A = rng.normal(size=(n_modes, n_modes)) + 1j * rng.normal(size=(n_modes, n_modes))
    q, _ = np.linalg.qr(A)
    bright_col = np.ones(n_modes, dtype=np.complex128) / math.sqrt(n_modes)
    cols = q.copy()
    cols[:, 0] = bright_col
    cols, _ = np.linalg.qr(cols)
    u_stack = cols.conj().T[:, :, np.newaxis]

    state = _keck_like_state(
        n_modes=n_modes,
        u_stack=u_stack,
        wavelengths_um=(1.25,),
        positions=np.zeros((n_modes, 2)),
        strehl=0.93,
        piston_opd_nm_total=1.35,
    )
    state = replace(state, scene=replace(state.scene, altitude_deg=60.0))

    photon_rate_per_sub_hz = 1.0e6
    amp_bw_hz = 5.0e3
    anchor = 0.041
    f_piston_hz = 100.0

    stage = DetectionCurve(
        params={
            "n_bright_ports": 1,
            "n_dark_ports": n_modes - 1,
            "separations_mas": (1.0, 10.0),
            "n_mc_realizations": 200,
            "mc_seed": 0,
            "auto_amplitude_servo_residual": True,
            "auto_amp_servo_bandwidth_hz": amp_bw_hz,
            "auto_amp_servo_photon_rate_per_sub_hz": photon_rate_per_sub_hz,
            "auto_scintillation_anchor_h5m_zenith": anchor,
            "auto_f_piston_hz": f_piston_hz,
        },
        context=_context(tmp_path),
    )
    out = stage.apply(state)
    payload = out.results.components["sensitivity.detection_curve"]

    # σ_I in payload must match a direct call to the physics function.
    expected_residual = amplitude_servo_residual_sigma_i(
        wavelength_m=1.25e-6,
        sub_diameter_m=float(payload["sub_diameters_m"][0]),
        servo_bandwidth_hz=amp_bw_hz,
        photon_rate_per_sub_hz=photon_rate_per_sub_hz,
        scintillation_index_h5m_zenith=anchor,
        airmass=1.0 / math.sin(math.radians(60.0)),
    )
    assert payload["scintillation_index_per_sub"] == pytest.approx(
        expected_residual.sigma_I_residual, rel=1.0e-12
    )
    assert payload["amplitude_servo_residual"] is not None
    assert payload["amplitude_servo_residual"]["sigma_I_open"] == pytest.approx(
        expected_residual.sigma_I_open, rel=1.0e-12
    )
    assert payload["amplitude_servo_residual"]["airmass"] == pytest.approx(
        1.0 / math.sin(math.radians(60.0)),
        rel=1.0e-12,
    )

    # f_eff weights the legs by their band leakage variance (row 27):
    # rerun the MC independently with one perturbation each.
    from nullsim.physics.realized_null import realized_null_depth_mc

    field = np.sqrt(payload["coupling_per_sub"] * payload["sub_areas_m2"]).astype(
        np.complex128
    )
    leg_common = dict(
        u_stack=u_stack,
        stellar_field=np.broadcast_to(field, (payload["n_azimuths"], n_modes)),
        dark_indices=payload["dark_indices"],
        wavelengths_m=np.array([1.25e-6]),
        n_realizations_per_azimuth=payload["n_realizations_per_azimuth"],
        seed=0,
    )
    v_phi = realized_null_depth_mc(
        ao_piston_opd_rms_per_sub_m=payload["ao_piston_opd_rms_per_sub_m"],
        **leg_common,
    ).null_fluctuation_rms ** 2
    v_I = realized_null_depth_mc(
        ao_piston_opd_rms_per_sub_m=0.0,
        scintillation_index_per_sub=expected_residual.sigma_I_residual,
        **leg_common,
    ).null_fluctuation_rms ** 2
    diagnostics = payload["effective_systematic_bandwidth_diagnostics"]
    assert diagnostics["piston_leakage_variance"] == pytest.approx(v_phi, rel=1e-10)
    assert diagnostics["amplitude_leakage_variance"] == pytest.approx(v_I, rel=1e-10)
    expected_f_eff = effective_systematic_bandwidth_hz(
        piston_leakage_variance=v_phi,
        amplitude_leakage_variance=v_I,
        f_piston_hz=f_piston_hz,
        f_amp_hz=amp_bw_hz,
    )
    assert payload["effective_systematic_bandwidth_hz"] == pytest.approx(
        expected_f_eff, rel=1.0e-10
    )
    assert (
        payload["effective_systematic_bandwidth_diagnostics"]["piston_rms_model"]
        == "scalar_rms"
    )


def test_detection_curve_injects_wavefront_amplitude_jitter(
    tmp_path: Path,
) -> None:
    n_modes = 4
    state = _keck_like_state(
        n_modes=n_modes,
        wavelengths_um=(1.20, 1.30),
        positions=np.zeros((n_modes, 2)),
        piston_opd_nm_total=0.0,
    )
    state = replace(
        state,
        wavefront=replace(
            state.wavefront,
            amplitude_jitter_fraction=(0.03, 0.04),
        ),
    )
    manual_sigma_i = 0.02

    stage = DetectionCurve(
        params={
            "n_bright_ports": 1,
            "n_dark_ports": n_modes - 1,
            "separations_mas": (1.0, 10.0),
            "n_realizations_per_azimuth": 0,
            "scintillation_index_per_sub": manual_sigma_i,
        },
        context=_context(tmp_path),
    )

    payload = stage.apply(state).results.components["sensitivity.detection_curve"]
    wavefront_rms = math.sqrt((0.03**2 + 0.04**2) / 2.0)
    expected_sigma_i = math.hypot(manual_sigma_i, wavefront_rms)

    assert payload["scintillation_index_per_sub"] == pytest.approx(expected_sigma_i)
    diagnostics = payload["amplitude_servo_residual"]
    assert diagnostics["mode"] == "manual"
    assert diagnostics["sigma_I_manual"] == pytest.approx(manual_sigma_i)
    assert diagnostics["wavefront_amplitude_jitter_fraction_rms"] == pytest.approx(
        wavefront_rms
    )
    assert diagnostics["sigma_I_injected"] == pytest.approx(expected_sigma_i)


def test_detection_curve_can_use_fringe_observability_piston_residual(
    tmp_path: Path,
) -> None:
    n_modes = 4
    state = _keck_like_state(
        n_modes=n_modes,
        wavelengths_um=(1.25,),
        positions=np.zeros((n_modes, 2)),
        piston_opd_nm_total=10.0,
    )
    components = dict(state.results.components)
    components["fringe_observability.pre_sensitivity"] = {
        "n_differential_piston_modes": n_modes - 1,
        "monte_carlo": {"median_residual_rms_nm": 1.85},
        "crlb_truncated_piston_rms_nm": 2.5,
    }
    state = replace(state, results=Results(components=components))

    stage = DetectionCurve(
        params={
            "n_bright_ports": 1,
            "n_dark_ports": n_modes - 1,
            "separations_mas": (1.0, 10.0),
            "n_realizations_per_azimuth": 0,
            "ao_piston_residual_source": "fringe_observability",
            "fringe_observability_piston_metric": "linear_monte_carlo_median",
            "fringe_observability_predictive_gain_variance": 4.0,
            "fringe_observability_piston_floor_nm": 0.5,
        },
        context=_context(tmp_path),
    )

    payload = stage.apply(state).results.components["sensitivity.detection_curve"]

    assert payload["ao_piston_opd_rms_per_sub_m"] == pytest.approx(0.925e-9)
    diagnostics = payload["ao_piston_residual"]
    assert diagnostics["source"] == "fringe_observability"
    assert diagnostics["fringe_observability_component"] == (
        "fringe_observability.pre_sensitivity"
    )
    assert diagnostics["fringe_observability_metric"] == (
        "monte_carlo.median_residual_rms_nm"
    )
    assert diagnostics["fringe_observability_raw_rms_nm"] == pytest.approx(1.85)
    assert diagnostics["fringe_observability_filtered_rms_nm"] == pytest.approx(0.925)
    assert diagnostics["fringe_observability_resolved_rms_nm"] == pytest.approx(0.925)


def test_detection_curve_can_use_fringe_observability_piston_covariance(
    tmp_path: Path,
) -> None:
    n_modes = 4
    state = _keck_like_state(
        n_modes=n_modes,
        wavelengths_um=(1.25,),
        positions=np.zeros((n_modes, 2)),
        piston_opd_nm_total=10.0,
    )
    raw_covariance = np.eye(n_modes, dtype=np.float64) * (2.0e-9) ** 2
    components = dict(state.results.components)
    components["fringe_observability.pre_sensitivity"] = {
        "n_differential_piston_modes": n_modes - 1,
        "monte_carlo": {"median_residual_rms_nm": 8.0},
        "piston_crlb_covariance_m2": raw_covariance,
        "piston_crlb_covariance_finite": True,
    }
    state = replace(state, results=Results(components=components))

    stage = DetectionCurve(
        params={
            "n_bright_ports": 1,
            "n_dark_ports": n_modes - 1,
            "separations_mas": (1.0, 10.0),
            "n_realizations_per_azimuth": 0,
            "ao_piston_residual_source": "fringe_observability",
            "fringe_observability_piston_metric": "linear_monte_carlo_median",
            "fringe_observability_piston_covariance": "crlb",
            "fringe_observability_predictive_gain_variance": 4.0,
            "fringe_observability_piston_floor_nm": 0.5,
        },
        context=_context(tmp_path),
    )

    payload = stage.apply(state).results.components["sensitivity.detection_curve"]

    assert payload["ao_piston_opd_rms_per_sub_m"] == pytest.approx(1.0e-9)
    np.testing.assert_allclose(
        payload["ao_piston_opd_covariance_m2"],
        np.eye(n_modes, dtype=np.float64) * 1.0e-18,
    )
    diagnostics = payload["ao_piston_residual"]
    assert diagnostics["fringe_observability_filtered_rms_nm"] == pytest.approx(4.0)
    assert diagnostics["fringe_observability_piston_covariance"] == "crlb"
    assert diagnostics["fringe_observability_covariance_raw_rms_nm"] == pytest.approx(
        2.0
    )
    assert diagnostics["fringe_observability_covariance_resolved_rms_nm"] == (
        pytest.approx(1.0)
    )
    assert (
        diagnostics["fringe_observability_covariance_magnitude_model"].startswith(
            "full-rank bright-port CRLB covariance"
        )
    )
    f_eff = payload["effective_systematic_bandwidth_diagnostics"]
    assert f_eff["piston_rms_model"] == "covariance_trace_rms_scalarized"
    assert f_eff["piston_covariance_used_in_realized_null_mc"] is True
    assert f_eff["piston_covariance_directionality_used_in_f_eff"] is False


def test_detection_curve_can_use_temporal_kalman_piston_covariance(
    tmp_path: Path,
) -> None:
    n_modes = 4
    state = _keck_like_state(
        n_modes=n_modes,
        wavelengths_um=(1.25,),
        piston_opd_nm_total=10.0,
    )
    raw_covariance = np.eye(n_modes, dtype=np.float64) * (2.0e-9) ** 2
    raw_covariance[0, :] = 0.0
    raw_covariance[:, 0] = 0.0
    components = dict(state.results.components)
    components["fringe_observability.pre_sensitivity"] = {
        "n_differential_piston_modes": n_modes - 1,
        "monte_carlo": {"median_residual_rms_nm": 8.0},
        "piston_crlb_covariance_m2": raw_covariance,
        "piston_crlb_covariance_finite": True,
        "positions_pupil_m": state.geometry.aperture_positions_pupil_m,
        "frame_time_s": 0.04,
    }
    state = replace(state, results=Results(components=components))

    stage = DetectionCurve(
        params={
            "n_bright_ports": 1,
            "n_dark_ports": n_modes - 1,
            "separations_mas": (1.0, 10.0),
            "n_realizations_per_azimuth": 0,
            "ao_piston_residual_source": "fringe_observability",
            "fringe_observability_piston_metric": "linear_monte_carlo_median",
            "fringe_observability_piston_covariance": "crlb",
            "fringe_observability_predictive_model": "temporal_kalman",
            "fringe_observability_temporal_residual_kind": "posterior",
            "fringe_observability_temporal_wind_speed_m_s": 10.0,
            "fringe_observability_piston_floor_nm": 0.0,
        },
        context=_context(tmp_path),
    )

    payload = stage.apply(state).results.components["sensitivity.detection_curve"]
    covariance = payload["ao_piston_opd_covariance_m2"]
    diagnostics = payload["ao_piston_residual"]

    assert covariance.shape == (n_modes, n_modes)
    assert diagnostics["fringe_observability_predictive_model"] == "temporal_kalman"
    assert diagnostics["fringe_observability_temporal_prior_rms_nm"] == pytest.approx(
        10.0
    )
    assert diagnostics["fringe_observability_temporal_prior_source"] == (
        "wavefront.ao_residual.piston_opd_nm.total"
    )
    assert diagnostics["fringe_observability_temporal_sample_time_s"] == pytest.approx(
        0.04
    )
    assert diagnostics[
        "fringe_observability_temporal_subaperture_diameter_m"
    ] == pytest.approx(1.5)
    assert diagnostics[
        "fringe_observability_temporal_subaperture_diameter_source"
    ] == "wavefront.ao_residual.piston_subaperture_diameter_m"
    assert diagnostics["fringe_observability_covariance_resolved_rms_nm"] < 2.0
    assert "Conan" in diagnostics["fringe_observability_temporal_literature"]


def test_detection_curve_requires_fringe_observability_when_selected(
    tmp_path: Path,
) -> None:
    n_modes = 4
    state = _keck_like_state(
        n_modes=n_modes,
        wavelengths_um=(1.25,),
        positions=np.zeros((n_modes, 2)),
    )
    stage = DetectionCurve(
        params={
            "n_bright_ports": 1,
            "n_dark_ports": n_modes - 1,
            "separations_mas": (1.0, 10.0),
            "n_realizations_per_azimuth": 0,
            "ao_piston_residual_source": "fringe_observability",
        },
        context=_context(tmp_path),
    )

    with pytest.raises(ValueError, match="requires an upstream"):
        stage.apply(state)


def test_detection_curve_auto_amplitude_servo_can_inject_open_loop_scintillation(
    tmp_path: Path,
) -> None:
    """Open-loop mode injects uncontrolled scintillation, not servo residuals."""

    from nullsim.physics.scintillation import amplitude_servo_residual_sigma_i
    from nullsim.physics.systematic_bandwidth import (
        effective_systematic_bandwidth_hz,
    )

    n_modes = 4
    u_stack = np.eye(n_modes, dtype=np.complex128)[:, :, np.newaxis]
    state = _keck_like_state(
        n_modes=n_modes,
        u_stack=u_stack,
        wavelengths_um=(1.25,),
        positions=np.zeros((n_modes, 2)),
        strehl=0.93,
        piston_opd_nm_total=1.35,
    )

    photon_rate_per_sub_hz = 1.0e6
    amp_bw_hz = 5.0e3
    f_piston_hz = 100.0
    stage = DetectionCurve(
        params={
            "n_bright_ports": 1,
            "n_dark_ports": n_modes - 1,
            "separations_mas": (1.0, 10.0),
            "n_realizations_per_azimuth": 0,
            "auto_amplitude_servo_residual": True,
            "auto_amplitude_servo_mode": "open_loop",
            "auto_amp_servo_bandwidth_hz": amp_bw_hz,
            "auto_amp_servo_photon_rate_per_sub_hz": photon_rate_per_sub_hz,
            "auto_f_piston_hz": f_piston_hz,
        },
        context=_context(tmp_path),
    )

    payload = stage.apply(state).results.components["sensitivity.detection_curve"]
    expected_residual = amplitude_servo_residual_sigma_i(
        wavelength_m=1.25e-6,
        sub_diameter_m=float(payload["sub_diameters_m"][0]),
        servo_bandwidth_hz=amp_bw_hz,
        photon_rate_per_sub_hz=photon_rate_per_sub_hz,
    )
    expected_amp_bandwidth_hz = 0.5 / expected_residual.tau_scint_s

    assert payload["scintillation_index_per_sub"] == pytest.approx(
        expected_residual.sigma_I_open, rel=1.0e-12
    )
    diagnostics = payload["amplitude_servo_residual"]
    assert diagnostics["mode"] == "open_loop"
    assert diagnostics["sigma_I_injected_source"] == "open_loop_scintillation"
    assert diagnostics["sigma_I_injected"] == pytest.approx(
        expected_residual.sigma_I_open, rel=1.0e-12
    )
    assert diagnostics["injected_amplitude_bandwidth_hz"] == pytest.approx(
        expected_amp_bandwidth_hz, rel=1.0e-12
    )

    # The MC is off, so neither leg has leakage variance and the blend takes
    # its widest-bandwidth fallback.
    expected_f_eff = effective_systematic_bandwidth_hz(
        piston_leakage_variance=0.0,
        amplitude_leakage_variance=0.0,
        f_piston_hz=f_piston_hz,
        f_amp_hz=expected_amp_bandwidth_hz,
    )
    assert payload["effective_systematic_bandwidth_hz"] == pytest.approx(
        expected_f_eff, rel=1.0e-12
    )


def test_detection_curve_auto_amp_servo_requires_photon_rate(
    tmp_path: Path,
) -> None:
    """Phase 3: auto-amp-servo without an explicit photon rate now derives
    from chip-input rate / n_modes (KILO natural_design.py L897). The
    error fires only when scene.star_jmag is also missing."""

    n_modes = 4
    rng = np.random.default_rng(0)
    A = rng.normal(size=(n_modes, n_modes)) + 1j * rng.normal(size=(n_modes, n_modes))
    q, _ = np.linalg.qr(A)
    bright_col = np.ones(n_modes, dtype=np.complex128) / math.sqrt(n_modes)
    cols = q.copy()
    cols[:, 0] = bright_col
    cols, _ = np.linalg.qr(cols)
    u_stack = cols.conj().T[:, :, np.newaxis]

    state = _keck_like_state(
        n_modes=n_modes,
        u_stack=u_stack,
        wavelengths_um=(1.25,),
        positions=np.zeros((n_modes, 2)),
        strehl=0.93,
        piston_opd_nm_total=1.35,
    )
    # Strip scene.star_jmag so the chip-input fallback also fails.
    state = replace(state, scene=Scene())

    stage = DetectionCurve(
        params={
            "n_bright_ports": 1,
            "n_dark_ports": n_modes - 1,
            "separations_mas": (1.0, 10.0),
            "auto_amplitude_servo_residual": True,
            # auto_amp_servo_photon_rate_per_sub_hz intentionally unset
        },
        context=_context(tmp_path),
    )
    with pytest.raises(ValueError, match="scene.star_jmag"):
        stage.apply(state)


def test_detection_curve_amp_servo_uses_chip_input_rate(
    tmp_path: Path,
) -> None:
    """Phase 3: when auto_amp_servo_photon_rate_per_sub_hz is unset, the
    amplitude-servo photon rate is derived from chip-input rate /
    n_modes (KILO natural_design.py L897), not the AO WFS payload."""

    n_modes = 4
    wavelengths_um = (1.20, 1.25, 1.30)
    # u_stack with 3 wavelengths: identity matrix at each.
    u_stack = np.zeros((n_modes, n_modes, len(wavelengths_um)), dtype=np.complex128)
    for k in range(len(wavelengths_um)):
        u_stack[:, :, k] = np.eye(n_modes)

    state = _keck_like_state(
        n_modes=n_modes,
        u_stack=u_stack,
        wavelengths_um=wavelengths_um,
        positions=np.zeros((n_modes, 2)),
        strehl=0.93,
        piston_opd_nm_total=1.35,
    )

    stage = DetectionCurve(
        params={
            "n_bright_ports": 1,
            "n_dark_ports": n_modes - 1,
            "separations_mas": (1.0, 10.0),
            "auto_amplitude_servo_residual": True,
            # auto_amp_servo_photon_rate_per_sub_hz intentionally unset:
            # chip-input fallback should take over.
            "n_mc_realizations": 0,
        },
        context=_context(tmp_path),
    )
    result = stage.apply(state)

    payload = result.results.components["sensitivity.detection_curve"]
    diag = payload.get("amp_servo_diagnostics") or payload.get(
        "amplitude_servo_residual"
    )
    assert diag is not None
    # Chip-input rate for J=4, area = pi * 25 m^2, 3 narrow wavelength bins
    # is ~10^9 ph/s. Per-sub rate (n_modes=4) ~ 2.5e8 ph/s.
    photon_rate_per_sub_hz = diag.get("photon_rate_per_sub_hz")
    assert photon_rate_per_sub_hz is not None
    assert 1e7 < photon_rate_per_sub_hz < 1e10, (
        f"chip-input amp-servo rate {photon_rate_per_sub_hz} not in "
        "expected J=4 Keck range."
    )


def test_detection_curve_amp_pickoff_scales_photon_rate(
    tmp_path: Path,
) -> None:
    """Phase 3: amp_wfs_pickoff_fraction linearly scales the per-sub
    rate driving the amplitude servo (KILO KILO_AMP_WFS_PICKOFF_FRAC)."""

    n_modes = 4
    wavelengths_um = (1.20, 1.25, 1.30)
    u_stack = np.zeros((n_modes, n_modes, len(wavelengths_um)), dtype=np.complex128)
    for k in range(len(wavelengths_um)):
        u_stack[:, :, k] = np.eye(n_modes)

    def _run(pickoff: float) -> float:
        state = _keck_like_state(
            n_modes=n_modes,
            u_stack=u_stack,
            wavelengths_um=wavelengths_um,
            positions=np.zeros((n_modes, 2)),
            strehl=0.93,
            piston_opd_nm_total=1.35,
        )
        stage = DetectionCurve(
            params={
                "n_bright_ports": 1,
                "n_dark_ports": n_modes - 1,
                "separations_mas": (1.0, 10.0),
                "auto_amplitude_servo_residual": True,
                "amp_wfs_pickoff_fraction": pickoff,
                "n_mc_realizations": 0,
            },
            context=_context(tmp_path),
        )
        result = stage.apply(state)
        payload = result.results.components["sensitivity.detection_curve"]
        diag = payload.get("amp_servo_diagnostics") or payload.get(
            "amplitude_servo_residual"
        )
        return float(diag["photon_rate_per_sub_hz"])

    full = _run(1.0)
    half = _run(0.5)
    assert half / full == pytest.approx(0.5, rel=1.0e-6)


def test_detection_curve_amp_servo_uses_simultaneous_science_bands(
    tmp_path: Path,
) -> None:
    """The amplitude servo can use summed J/H/K science-band photons."""

    n_modes = 4
    wavelengths_um = (1.20, 1.25, 1.30)
    u_stack = np.zeros((n_modes, n_modes, len(wavelengths_um)), dtype=np.complex128)
    for k in range(len(wavelengths_um)):
        u_stack[:, :, k] = np.eye(n_modes)

    def _run(bands: tuple[str, ...] = ()) -> float:
        state = _keck_like_state(
            n_modes=n_modes,
            u_stack=u_stack,
            wavelengths_um=wavelengths_um,
            positions=np.zeros((n_modes, 2)),
            strehl=0.93,
            piston_opd_nm_total=1.35,
        )
        stage = DetectionCurve(
            params={
                "n_bright_ports": 1,
                "n_dark_ports": n_modes - 1,
                "separations_mas": (1.0, 10.0),
                "auto_amplitude_servo_residual": True,
                "simultaneous_science_bands": bands,
                "simultaneous_science_band_transmissions": (
                    (0.985, 0.97, 0.92) if bands else ()
                ),
                "n_mc_realizations": 0,
            },
            context=_context(tmp_path),
        )
        result = stage.apply(state)
        payload = result.results.components["sensitivity.detection_curve"]
        diag = payload.get("amplitude_servo_residual")
        return float(diag["photon_rate_per_sub_hz"])

    single = _run()
    simultaneous = _run(("J", "H", "K"))

    assert simultaneous > single


def test_detection_curve_planet_coupling_picks_up_obstruction(
    tmp_path: Path,
) -> None:
    """Phase 4: planet coupling reads central_obstruction_ratio from the
    upstream telescope diagnostic, matching the stellar-injection
    convention. The pre-Phase-4 code hardcoded 0.0."""

    n_modes = 4
    wavelengths_um = (1.25,)
    u_stack = np.eye(n_modes, dtype=np.complex128)[:, :, np.newaxis]

    def _run_with_obstruction(epsilon: float) -> float:
        state = _keck_like_state(
            n_modes=n_modes,
            u_stack=u_stack,
            wavelengths_um=wavelengths_um,
            positions=np.zeros((n_modes, 2)),
            strehl=0.93,
            piston_opd_nm_total=1.35,
        )
        geometry = state.geometry.add_component(
            "fixed_aperture",
            {"central_obstruction_ratio": float(epsilon)},
        )
        state = replace(state, geometry=geometry)

        stage = DetectionCurve(
            params={
                "n_bright_ports": 1,
                "n_dark_ports": n_modes - 1,
                "separations_mas": (1.0, 10.0),
                "apply_strehl_coupling": True,
                "apply_fresnel_injection_loss": False,
                "n_mc_realizations": 0,
            },
            context=_context(tmp_path),
        )
        result = stage.apply(state)
        payload = result.results.components["sensitivity.detection_curve"]
        coupling = float(payload["coupling_per_sub"][0])
        return coupling

    eta_unobstructed = _run_with_obstruction(0.0)
    eta_keck = _run_with_obstruction(0.24)
    # A 24% linear obstruction reduces the Ruilier-Cassaing geometric
    # coupling. At Strehl=0.93 the ratio eta(eps=0.24) / eta(eps=0)
    # comes out to ~0.864 (verified against the physics primitive).
    assert eta_keck < eta_unobstructed
    expected = coupling_efficiency(0.93, 0.24) / coupling_efficiency(0.93, 0.0)
    assert eta_keck / eta_unobstructed == pytest.approx(expected, rel=1e-6)


def test_auto_setpoint_trim_derives_photon_limited_budget(tmp_path: Path) -> None:
    """auto_setpoint_trim sizes dither and held setpoint from the photon budget.

    The stage must reproduce physics.fringe_tracking.photon_limited_setpoint_trim
    evaluated at its own base null (realized + disk) and the detector stellar
    rate, with the equal dither/bias partition of the optimum, and publish the
    derived equivalent OPD terms.
    """
    from nullsim.physics.fringe_tracking import photon_limited_setpoint_trim

    bright_rate_hz = 1.0e8
    state = _state_with_detector_component(bright_rate_hz=bright_rate_hz)
    stage = DetectionCurve(
        params={
            "n_bright_ports": 2,
            "n_dark_ports": 2,
            "separations_mas": (1.0, 10.0, 100.0),
            "n_mc_realizations": 0,
            "auto_setpoint_trim": True,
            "trim_loop_bandwidth_hz": 1.0,
        },
        context=_context(tmp_path),
    )
    payload = stage.apply(state).results.components["sensitivity.detection_curve"]

    total_stellar_rate_hz = 2.0 * bright_rate_hz + 2.0 * 2.0e3 + 5.0e2
    base_null = payload["realized_null_depth"] + payload["disk_null_floor"]
    # The stage iterates the trim to a fixed point: its own added leakage
    # feeds back into the 1f lock-in shot-noise floor (review 2026-06-10
    # Issue 27e). Replicate the fixed point here (no staged background in
    # this fixture, so background_null_equiv = 0).
    expected = photon_limited_setpoint_trim(
        stellar_rate_hz=total_stellar_rate_hz,
        dark_floor_null=base_null,
        trim_loop_bandwidth_hz=1.0,
        n_trim_modes=2,  # default: n_dark_ports + n_spectro_ports
        wavelength_m=1.25e-6,
    )
    for _ in range(8):
        next_budget = photon_limited_setpoint_trim(
            stellar_rate_hz=total_stellar_rate_hz,
            dark_floor_null=base_null + expected.total_leakage,
            trim_loop_bandwidth_hz=1.0,
            n_trim_modes=2,
            wavelength_m=1.25e-6,
        )
        delta = abs(next_budget.total_leakage - expected.total_leakage)
        expected = next_budget
        if delta <= 1.0e-3 * max(expected.total_leakage, 1e-30):
            break
    assert payload["auto_setpoint_trim"] is True
    assert payload["n_trim_modes"] == 2
    assert payload["setpoint_dither_leakage"] == pytest.approx(
        expected.dither_leakage, rel=1.0e-9
    )
    assert payload["null_setpoint_leakage"] == pytest.approx(
        expected.bias_leakage, rel=1.0e-9
    )
    assert payload["setpoint_dither_leakage"] == pytest.approx(
        payload["null_setpoint_leakage"], rel=1.0e-9
    )
    assert payload["setpoint_dither_rms_nm"] == pytest.approx(
        expected.dither_rms_nm, rel=1.0e-9
    )
    assert payload["null_setpoint_residual_nm"] == pytest.approx(
        math.sqrt(2.0) * expected.setpoint_residual_nm_per_mode, rel=1.0e-9
    )
    assert payload["effective_null_with_disk"] == pytest.approx(
        base_null + expected.total_leakage, rel=1.0e-6
    )


def test_auto_setpoint_trim_budget_grows_for_fainter_star(tmp_path: Path) -> None:
    """The derived trim budget scales as 1/sqrt(flux): fainter -> bigger leakage."""
    payloads = {}
    for label, rate in (("bright", 1.0e8), ("faint", 1.0e8 / 40.0)):
        state = _state_with_detector_component(bright_rate_hz=rate)
        stage = DetectionCurve(
            params={
                "n_bright_ports": 2,
                "n_dark_ports": 2,
                "separations_mas": (1.0, 10.0, 100.0),
                "n_mc_realizations": 0,
                "auto_setpoint_trim": True,
            },
            context=_context(tmp_path),
        )
        payloads[label] = stage.apply(state).results.components[
            "sensitivity.detection_curve"
        ]
    bright_total = (
        payloads["bright"]["setpoint_dither_leakage"]
        + payloads["bright"]["null_setpoint_leakage"]
    )
    faint_total = (
        payloads["faint"]["setpoint_dither_leakage"]
        + payloads["faint"]["null_setpoint_leakage"]
    )
    assert faint_total > 4.0 * bright_total


def test_auto_setpoint_trim_requires_detector_component(tmp_path: Path) -> None:
    state = _keck_like_state()
    stage = DetectionCurve(
        params={
            "n_bright_ports": 2,
            "n_dark_ports": 2,
            "separations_mas": (1.0, 10.0, 100.0),
            "n_mc_realizations": 0,
            "auto_setpoint_trim": True,
        },
        context=_context(tmp_path),
    )
    with pytest.raises(ValueError, match="auto_setpoint_trim"):
        stage.apply(state)
