"""SPEC §15.7 HWO-class space-nuller sanity tests.

The bar here is intentionally low and explicit: the pipeline must
*accept* and *run* a config that exercises a stage shape no prior
nullsim TOML has — no atmosphere, no AO, no fringe tracking, no delay
lines — and the headline outputs must still be produced. That alone
demonstrates nullsim is a general-purpose tool rather than a re-
implementation of KILO/NAYRA.

Tests come in three buckets:

1. **Hard sanity** (must always pass): config loads, pipeline runs
   end-to-end without exception, the telescope resolves to the requested
   8 m unobstructed aperture, the detector is not saturation limited,
   the dual-pol PBS/chip path is active, and the headline
   ``detection_space`` plot writes a PNG.
2. **Surrogate-dependent sanity**: the current 36-mode ideal kernel seed
   must be clean at the design wavelength, while the deployed physical
   Clements mesh is allowed to show finite-bandwidth chromatic leakage.
3. **Debug-output sanity**: the 36-mode exploratory run must emit
   separation-dependent contrast curves and the full diagnostic plot set.

New gaps discovered while writing this config (NOT KILO/NAYRA gaps):

* No ``metrology.formation_flying`` (or analogous) stage exists yet.
  KILO carries inter-telescope piston via ``fringe_tracking.closed_loop``
  and NAYRA via the AO loop on a monolithic pupil. This config now has a
  scalar ``wavefront.space_residual`` WFE/pointing placeholder, but a real
  formation-flying HWO nuller would need a proper residual-OPD model.

* No "space-grade detector noise model". The MKID stage exposes
  ``mkid_max_count_rate_hz``, ``qe``, and ``integration_time_s``, but
  no explicit dark-count or radiation-damage model. The HWO config now
  includes local zodiacal and exozodiacal backgrounds through
  ``background.space_zodi``; detector-intrinsic terms remain zero.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from nullsim.config import expand_sweeps, load_config
from nullsim.outputs import RunCollection, RunRecord
from nullsim.outputs.plots.chip_diagnostics import chip_diagnostics
from nullsim.outputs.plots.detection_space import detection_space
from nullsim.pipeline import run

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "examples" / "hwo_space.toml"


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def fiducial_state():
    """Run the pipeline at the fiducial cell (planet contrast = 1e-10)."""

    cfg = load_config(str(CONFIG_PATH))
    cells = expand_sweeps(cfg)
    for cell in cells:
        coord_map = dict(cell.coord_tuple)
        if math.isclose(coord_map.get("scene.planet.contrast", -1.0), 1.0e-10):
            return run(cell.config).state
    raise AssertionError(
        "No sweep cell with scene.planet.contrast=1e-10 in hwo_space.toml."
    )


@pytest.fixture(scope="module")
def sweep_cells():
    """Pre-expanded sweep cells (cached for the contrast-sweep test)."""

    return expand_sweeps(load_config(str(CONFIG_PATH)))


# ----------------------------------------------------------------------
# Hard sanity (must always pass)
# ----------------------------------------------------------------------


def test_hwo_space_config_loads() -> None:
    """The HWO TOML must parse and resolve to a usable ResolvedConfig."""

    cfg = load_config(str(CONFIG_PATH))
    assert cfg.run.name == "hwo_space"
    assert cfg.run.seed == 1
    catalog = cfg.raw_config["catalog"]["exoplanets"]
    snapshot_path = str(catalog["snapshot_path"])
    assert snapshot_path == (
        "../nullsim/data/exoplanet_catalogs/"
        "nasa_exoplanet_archive_paperv3.csv"
    )
    assert "tests/catalogs/fixtures" not in snapshot_path
    resolved_snapshot = (CONFIG_PATH.parent / snapshot_path).resolve()
    assert resolved_snapshot.is_file()
    with resolved_snapshot.open(encoding="utf-8") as handle:
        assert sum(1 for _line in handle) > 100


def test_hwo_telescope_resolves_to_8m_unobstructed_primary() -> None:
    """The HWO config must materialize the requested primary assumptions."""

    cfg = load_config(str(CONFIG_PATH))
    stages = {
        str(stage.id): stage.model_dump(mode="python")
        for stage in cfg.pipeline.stages
    }

    aperture = cfg.telescope.model_dump()["aperture"]
    assert aperture["diameter_m"] == pytest.approx(8.0)
    assert aperture["obscuration_fraction"] == 0.0
    assert cfg.scene.star.jmag == pytest.approx(2.13)
    assert cfg.scene.star.spectral_type == "G2V"
    assert cfg.scene.star.angular_diameter_mas == pytest.approx(1.86)
    assert cfg.scene.star.distance_pc == pytest.approx(5.0)
    assert cfg.scene.planet.separation_mas == pytest.approx(200.0)
    assert cfg.scene.exozodi.level_zodi == pytest.approx(1.0)

    scene = stages["scene"]
    assert scene["star_distance_pc"] == pytest.approx(5.0)
    assert scene["exozodi_level_zodi"] == pytest.approx(1.0)

    telescope = stages["telescope"]
    assert telescope["diameter_m"] == pytest.approx(8.0)
    assert telescope["n_apertures"] == 1
    assert telescope["central_obstruction_ratio"] == 0.0
    assert telescope["mirror_transmission"] == pytest.approx(0.95)

    injection_seed = stages["injection_seed"]
    assert injection_seed["central_obscuration_ratio"] == 0.0
    assert injection_seed["apply_strehl_coupling"] is False
    injection = stages["injection"]
    assert injection["central_obscuration_ratio"] == 0.0

    pupil_modes = stages["pupil_modes"]
    assert pupil_modes["n_modes"] == 36
    assert pupil_modes["layout"] == "nonredundant_circular"
    assert pupil_modes["span_fraction"] == pytest.approx(1.0)
    assert pupil_modes["normalize_max_baseline_to_span"] is False
    assert pupil_modes["preserve_collecting_area"] is False

    chip_opt = stages["chip_opt"]
    assert chip_opt["type"] == "chip_optimization.ideal_kernel"
    assert chip_opt["n_modes"] == 36
    assert chip_opt["n_bright_ports"] == 10
    assert chip_opt["n_dark_ports"] == 26
    assert chip_opt["wl_design_um"] == pytest.approx(0.8)
    assert chip_opt["kernel_order"] == 4
    assert chip_opt["target_null_depth"] == pytest.approx(1.0e-10)
    assert chip_opt["deployment"] == "physical_real_clements"

    pbs = stages["pbs"]
    assert pbs["type"] == "polarization.pbs_split"
    assert pbs["input_interpretation"] == "unpolarized"

    transport = stages["pm780_transport"]
    assert transport["type"] == "transport.pm780"
    assert transport["length_m"] == pytest.approx(2.0)
    assert transport["attenuation_db_per_km"] == pytest.approx(4.0)
    assert transport["phi_rms_rad"] == pytest.approx(0.0)
    assert transport["length_matching_error_m"] == pytest.approx(0.0)

    chip = stages["chip"]
    assert chip["deployment_mode"] == "per_pol_arm"
    assert chip["chip_chromaticity_model"] == "parametric"
    assert chip["phase_chromaticity_model"] == "fixed_opd_neff"

    for stage_id in ("chip", "detector", "sensitivity"):
        stage = stages[stage_id]
        assert stage["n_bright_ports"] == 10
        assert stage["n_dark_ports"] == 26

    detector = stages["detector"]
    assert detector["n_modes"] == 36
    assert detector["band"] == "I"
    assert detector["derive_band_mag_from_scene_jmag"] is True
    assert detector["resolving_power_ref"] == pytest.approx(30.0)
    assert detector["lambda_ref_um"] == pytest.approx(1.0)

    zodi = stages["zodi"]
    assert zodi["type"] == "background.space_zodi"
    assert zodi["band"] == "I"
    assert zodi["local_zodi_v_mag_arcsec2"] == pytest.approx(23.3)
    assert zodi["derive_exozodi_mag_from_scene_jmag"] is True

    assert "calibration" not in stages  # photon-noise baseline: no imposed floor

    performance = stages["performance"]
    assert performance["contrast_curve_reference_time_s"] == pytest.approx(
        cfg.observation.integration_time_s
    )


def test_hwo_space_pipeline_runs_end_to_end(fiducial_state) -> None:
    """Pipeline must execute every stage without exception.

    This is the headline sanity check for §15.7. The act of producing a
    ``fiducial_state`` already proves the pipeline ran; the assertions
    here just pin the components we expect to see populated.
    """

    component_names = set(fiducial_state.results.components.keys())
    assert "background.space_zodi" in component_names
    assert "calibration.floor_lookup" not in component_names
    assert "sensitivity.detection_curve" in component_names
    assert "polarization.pm_fiber" not in component_names
    assert any(name.startswith("detector.mkid") for name in component_names)

    field_components = fiducial_state.field.components
    assert field_components["polarization.pbs_split"]["output_n_pol"] == 2
    assert "polarization.pm_fiber" not in field_components

    throughput_components = fiducial_state.throughput.components
    injection_seed = throughput_components["injection_ideal"]
    assert injection_seed["transmission_fraction"][0] == pytest.approx(1.0)
    injection = throughput_components["single_mode_fiber"]
    assert injection["transmission_fraction"][0] == pytest.approx(
        0.8145287551781475
    )
    transport = throughput_components["transport.pm780_loss"]
    assert transport["transmission_fraction"][0] == pytest.approx(
        10.0 ** (-(4.0 * 2.0 / 1000.0) / 10.0)
    )
    assert "polarization.pm_fiber_loss" not in throughput_components

    chip = next(
        payload
        for name, payload in fiducial_state.results.components.items()
        if name.startswith("chip.kernel_mzi_mesh")
    )
    assert chip["deployment_mode"] == "per_pol_arm"
    assert chip["active_chip_count"] == 2


def test_hwo_pupil_surrogate_uses_physical_holes_across_8m_pupil(
    fiducial_state,
) -> None:
    """The 36-mode HWO surrogate uses non-overlapping physical holes."""

    full_area = math.pi * 4.0**2
    assert fiducial_state.geometry.effective_collecting_area_m2 < full_area
    assert fiducial_state.geometry.baselines_m is not None

    payload = fiducial_state.geometry.components["pupil_modes"]
    assert payload["preserve_collecting_area"] is False
    assert payload["normalize_max_baseline_to_span"] is False
    assert payload["span_fraction"] == pytest.approx(1.0)
    assert payload["span_m"] == pytest.approx(8.0)
    assert payload["effective_area_source"] == "packed_mode_circles"
    assert fiducial_state.geometry.effective_collecting_area_m2 == pytest.approx(
        payload["geometric_mode_area_m2"],
    )
    positions = payload["aperture_positions_pupil_m"]
    mode_radius = 0.5 * payload["mode_diameter_m"]
    assert payload["mode_diameter_m"] > 1.0
    assert (
        payload["geometric_mode_area_m2"] / full_area
    ) > 0.70
    radii = np.sqrt(np.sum(positions * positions, axis=1))
    assert float(np.max(radii + mode_radius)) <= 4.0 + 1.0e-5
    diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diffs * diffs, axis=2))
    distances = distances[np.triu_indices(positions.shape[0], k=1)]
    assert float(np.min(distances)) + 5.0e-5 >= payload["mode_diameter_m"]


def test_hwo_performance_has_no_calibration_floor(
    fiducial_state,
) -> None:
    """No calibration stage: performance must run floor-free.

    The photon-noise baseline drops calibration.floor_lookup entirely;
    performance.standard records the absent floor source instead of an
    imposed 10^-10 systematic.
    """

    perf = next(
        payload
        for name, payload in fiducial_state.results.components.items()
        if name.startswith("performance.standard")
    )
    assert perf.get("calibration_floor_source") in (None, "none", "")


def test_hwo_residual_starlight_profile_sets_design_contrast(fiducial_state) -> None:
    """The HWO baseline should carry an explicit residual-starlight profile."""

    sensitivity = fiducial_state.results.components["sensitivity.detection_curve"]
    space_wavefront = fiducial_state.wavefront.components["wavefront.space_residual"]
    performance = next(
        payload
        for name, payload in fiducial_state.results.components.items()
        if name.startswith("performance.standard")
    )
    assert space_wavefront["total_wfe_rms_pm"] == pytest.approx(math.hypot(10.0, 5.0))
    assert min(space_wavefront["strehl_ratio"]) > 0.99999999
    assert space_wavefront["pointing_jitter_rms_mas"] == pytest.approx(0.5)
    assert sensitivity["residual_starlight_applied"] is True
    assert sensitivity["residual_starlight_combine_mode"] == "max"
    assert sensitivity["residual_starlight_table_path"].endswith(
        "data/hwo_reference/residual_contrast_i_band.csv"
    )
    assert sensitivity["pointing_jitter_applied"] is True
    assert sensitivity["pointing_jitter_rms_mas"] == pytest.approx(0.5)
    assert (
        sensitivity["pointing_jitter_source"]
        == "wavefront:wavefront.space_residual.pointing_jitter_rms_mas"
    )
    assert sensitivity["effective_null_before_residual_starlight"] < 1.0e-10
    residual = np.asarray(sensitivity["residual_starlight_contrast"], dtype=float)
    residual_by_wavelength = np.asarray(
        sensitivity["residual_starlight_contrast_by_wavelength"],
        dtype=float,
    )
    separations = np.asarray(sensitivity["separations_mas"], dtype=float)
    assert residual_by_wavelength.shape == (
        separations.size,
        len(sensitivity["residual_starlight_wavelengths_um"]),
    )
    assert residual[0] > 1.0e-7
    assert residual[np.argmin(np.abs(separations - 200.0))] < 1.2e-10
    assert np.max(residual) > 1000.0 * np.min(residual)
    assert 8.0e-11 < sensitivity["effective_null_with_disk"] < 1.3e-10
    np.testing.assert_allclose(
        sensitivity["effective_null_with_disk_by_separation"],
        np.maximum(
            sensitivity["effective_null_before_residual_starlight"],
            residual,
        ),
    )
    assert performance["residual_speckle_stability_factor"] == pytest.approx(0.01)
    assert performance["eps_systematic_used"] == pytest.approx(0.01)


def test_hwo_detector_not_saturation_limited(fiducial_state) -> None:
    """Zodi-background baseline must not be capped by detector saturation."""

    detector = next(
        payload
        for name, payload in fiducial_state.results.components.items()
        if name.startswith("detector.mkid")
    )
    assert detector["photometric_band"] == "I"
    assert detector["derive_band_mag_from_scene_jmag"] is True
    assert detector["resolving_power_ref"] == pytest.approx(30.0)
    assert detector["lambda_ref_um"] == pytest.approx(1.0)
    assert len(detector["resolving_power_per_bin"]) == 11
    assert detector["star_magnitude"] == pytest.approx(2.58)
    assert not any(bool(flag) for flag in detector["saturated_per_port"])


def test_hwo_space_zodi_background_reaches_performance(fiducial_state) -> None:
    """Local zodi and scene exozodi must enter the performance noise budget."""

    zodi = fiducial_state.results.components["background.space_zodi"]
    assert zodi["band"] == "I"
    assert zodi["local_zodi_level"] == pytest.approx(1.0)
    assert zodi["exozodi_level_distribution"] == "hosts2020_lognormal"
    assert zodi["exozodi_level_source"] == "hosts2020_lognormal"
    assert zodi["exozodi_level_zodi"] > 0.0
    assert zodi["exozodi_morphology"] == "smooth_inclined_disk"
    assert 0.0 <= zodi["exozodi_inclination_deg"] <= 80.0
    assert zodi["exozodi_surface_brightness_multiplier"] >= 1.0
    assert zodi["exozodi_effective_level_zodi"] == pytest.approx(
        zodi["exozodi_level_zodi"] * zodi["exozodi_surface_brightness_multiplier"]
    )
    assert zodi["host_star_mag_for_exozodi"] == pytest.approx(2.58)
    assert zodi["exozodi_mag_arcsec2"] == pytest.approx(2.58 + 22.0 - 4.83)
    assert zodi["local_zodi_v_mag_arcsec2"] == pytest.approx(23.3)
    expected_spaxel_arcsec2 = (0.8e-6 / 8.0 * 206_264.80624709636) ** 2
    assert zodi["spaxel_arcsec2"] == pytest.approx(expected_spaxel_arcsec2)
    assert zodi["local_zodi_rate_hz_per_dark_port"] > 0.0
    assert zodi["exozodi_rate_hz_per_dark_port"] > 0.0
    assert zodi["background_rate_hz_per_dark_port"] == pytest.approx(
        zodi["local_zodi_rate_hz_per_dark_port"] + zodi["exozodi_rate_hz_per_dark_port"]
    )
    sensitivity = fiducial_state.results.components["sensitivity.detection_curve"]
    np.testing.assert_allclose(
        zodi["background_separations_mas"],
        sensitivity["separations_mas"],
    )
    np.testing.assert_allclose(
        zodi["background_routing_throughputs"],
        sensitivity["planet_throughputs"],
    )

    perf = next(
        payload
        for name, payload in fiducial_state.results.components.items()
        if name.startswith("performance.standard")
    )
    assert perf["sky_rate_source"] == "background.space_zodi"
    assert perf["sky_rate_hz_per_port"] == pytest.approx(
        zodi["background_rate_hz_per_dark_port"]
    )
    np.testing.assert_allclose(
        perf["contrast_curve_sky_rate_hz_per_port"],
        zodi["background_rate_hz_per_dark_port_per_sep"],
    )
    assert perf["Sky"] > 0.0
    assert perf["D"] == pytest.approx(0.0)


def test_hwo_no_atmosphere_no_ao_no_fringe_tracking(fiducial_state) -> None:
    """No atmosphere / AO / FT components may appear in the OPD budget."""

    opd_components = set(fiducial_state.opd_budget.components.keys())
    assert any(name.startswith("wavefront.space_residual") for name in opd_components)
    assert "atmos_piston" not in opd_components
    assert not any(name.startswith("ao.") for name in opd_components)
    assert not any(
        name.startswith("fringe_tracking.") for name in opd_components
    )


def test_hwo_contrast_sweep_has_five_cells(sweep_cells) -> None:
    """The contrast sweep declared in TOML must expand to 5 cells."""

    contrast_values = sorted(
        dict(cell.coord_tuple).get("scene.planet.contrast", math.nan)
        for cell in sweep_cells
    )
    assert len(contrast_values) == 5
    assert contrast_values[0] == pytest.approx(1.0e-12)
    assert contrast_values[-1] == pytest.approx(1.0e-8)


def test_hwo_sensitivity_grid_tracks_8m_resolution() -> None:
    """HWO contrast curves should be sampled in the 8 m λ/D regime."""

    cfg = load_config(str(CONFIG_PATH))
    stages = {
        str(stage.id): stage.model_dump(mode="python")
        for stage in cfg.pipeline.stages
    }
    separations = np.asarray(stages["sensitivity"]["separations_mas"], dtype=float)
    sensitivity = stages["sensitivity"]
    aperture = cfg.telescope.model_dump(mode="python")["aperture"]
    assert cfg.grid.wavelength_center_um == pytest.approx(0.8)
    assert cfg.grid.wavelength_bandwidth_um == pytest.approx(0.08)
    assert stages["space_wavefront"]["static_wfe_rms_pm"] == pytest.approx(10.0)
    assert stages["space_wavefront"]["dynamic_wfe_rms_pm"] == pytest.approx(5.0)
    assert stages["space_wavefront"]["pointing_jitter_rms_mas"] == pytest.approx(0.5)
    assert (
        sensitivity["residual_starlight_table_path"]
        == "data/hwo_reference/residual_contrast_i_band.csv"
    )
    assert sensitivity["residual_starlight_combine_mode"] == "max"
    assert "pointing_jitter_rms_mas" not in sensitivity
    assert stages["performance"]["residual_speckle_stability_factor"] == pytest.approx(
        0.01
    )
    lambda_over_d_mas = (
        cfg.grid.wavelength_center_um * 1.0e-6 / float(aperture["diameter_m"])
    ) * (180.0 / math.pi) * 3.6e6

    assert lambda_over_d_mas == pytest.approx(20.63, rel=1.0e-3)
    assert separations.min() / lambda_over_d_mas == pytest.approx(0.0024, rel=0.03)
    assert separations.max() / lambda_over_d_mas == pytest.approx(24.2, rel=0.03)
    assert np.min(np.abs(separations - 1.0)) < 1.0e-12
    assert np.min(np.abs(separations - 200.0)) < 10.0


def test_hwo_output_requests_all_diagnostic_plots() -> None:
    """The debug config should emit every registered plot family."""

    cfg = load_config(str(CONFIG_PATH))
    assert set(cfg.output.plots) == {
        "detection_space",
        "contrast_curve",
        "null_transmission_map",
        "subaperture_layout",
        "uv_coverage_grid",
        "chip_diagnostics",
        "loss_budget",
        "contrast_curves_multiband",
        "calibration_time",
        "debug_report",
        "imaging_reconstruction",
    }
    null_map = cfg.output.null_transmission_map
    assert null_map is not None
    assert null_map.fov_mas == pytest.approx(500.0)
    assert null_map.max_records == 1
    imaging = cfg.output.imaging_reconstruction
    assert imaging is not None
    assert imaging.rotation_mode == "spacecraft_roll"
    assert imaging.roll_span_deg == pytest.approx(120.0)
    assert imaging.n_roll_steps == 25
    assert imaging.n_pix == 64
    assert imaging.fov_mas == pytest.approx(350.0)
    assert imaging.poisson is False
    assert imaging.compute_nnls is True
    assert imaging.nnls_wavelength_mode == "resolved"
    assert imaging.nnls_candidate_pixels == 512
    assert imaging.nnls_regularization_fraction == pytest.approx(0.01)
    assert imaging.chop_mode == "chip_phase"
    assert imaging.include_exozodi_glow is True
    assert imaging.exozodi_inner_radius_au == pytest.approx(0.2)
    assert imaging.exozodi_outer_radius_au == pytest.approx(5.0)
    assert imaging.exozodi_radial_power == pytest.approx(1.0)
    assert imaging.exozodi_position_angle_deg == pytest.approx(60.0)
    assert len(imaging.planets) == 3
    assert [planet.flux_model for planet in imaging.planets] == [
        "reflected_light",
        "reflected_light",
        "reflected_light",
    ]
    assert [planet.label for planet in imaging.planets] == ["Venus", "Earth", "Mars"]
    assert [planet.radius_earth for planet in imaging.planets] == pytest.approx(
        [0.9499, 1.0, 0.532],
    )
    assert [planet.semi_major_axis_au for planet in imaging.planets] == pytest.approx(
        [0.723, 1.0, 1.524],
    )


def test_hwo_contrast_curve_uses_separation_dependent_sensitivity(
    fiducial_state,
) -> None:
    """Detection-space curves must not use the flat performance fallback."""

    curve = fiducial_state.results.contrast_curve
    assert curve is not None
    assert curve.size > 10
    contrasts = np.asarray(curve["contrast"], dtype=np.float64)
    finite_positive = contrasts[np.isfinite(contrasts) & (contrasts > 0.0)]
    assert finite_positive.size > 10
    assert np.ptp(np.log10(finite_positive)) > 0.25

    perf = next(
        payload
        for name, payload in fiducial_state.results.components.items()
        if name.startswith("performance.standard")
    )
    assert perf["upstream_sensitivity_component"] == "sensitivity.detection_curve"


def test_hwo_detection_space_plot_writes_png(
    fiducial_state, tmp_path: Path
) -> None:
    """The headline detection_space plot must produce a non-empty PNG.

    Smoke test analogous to ``test_detection_space.py``. We synthesize a
    minimal ``RunCollection`` from the fiducial run so the plot has at
    least one record to draw, then verify a real file on disk.
    """

    cfg = load_config(str(CONFIG_PATH))
    # Build a single-record RunCollection from a re-run of the fiducial
    # cell (the fixture only exposes the SimulationState; we need the
    # RunResult wrapper for RunRecord). Re-running is cheap (the cache
    # handles the heavy lifting).
    fiducial_cell = next(
        cell
        for cell in expand_sweeps(cfg)
        if math.isclose(
            dict(cell.coord_tuple).get("scene.planet.contrast", -1.0),
            1.0e-10,
        )
    )
    result = run(fiducial_cell.config)
    record = RunRecord(
        config=fiducial_cell.config,
        result=result,
        run_dir=tmp_path,
        coord_tuple=fiducial_cell.coord_tuple,
    )
    collection = RunCollection(
        config=fiducial_cell.config,
        records=(record,),
        output_dir=tmp_path,
    )

    paths = detection_space(collection, tmp_path, ("png",))
    assert len(paths) == 1
    png_path = paths[0]
    assert png_path.exists()
    assert png_path.suffix == ".png"
    # Non-zero PNG; the plot may degrade to a placeholder figure if the
    # contrast curve is empty (NaN null depth), so we keep the size bar
    # loose. ~3 kB is enough to confirm the figure rendered with axes.
    assert png_path.stat().st_size > 3_000


def test_hwo_chip_diagnostics_plot_writes_png(tmp_path: Path) -> None:
    """The fixed-OPD ideal-kernel deployment must still emit chip diagnostics."""

    cfg = load_config(str(CONFIG_PATH))
    fiducial_cell = next(
        cell
        for cell in expand_sweeps(cfg)
        if math.isclose(
            dict(cell.coord_tuple).get("scene.planet.contrast", -1.0),
            1.0e-10,
        )
    )
    result = run(fiducial_cell.config)
    record = RunRecord(
        config=fiducial_cell.config,
        result=result,
        run_dir=tmp_path,
        coord_tuple=fiducial_cell.coord_tuple,
    )
    collection = RunCollection(
        config=fiducial_cell.config,
        records=(record,),
        output_dir=tmp_path,
    )

    paths = chip_diagnostics(collection, tmp_path, ("png",))
    assert len(paths) == 1
    png_path = paths[0]
    assert png_path.exists()
    assert png_path.suffix == ".png"
    assert png_path.stat().st_size > 3_000


# ----------------------------------------------------------------------
# Surrogate-dependent sanity (see §15.5 gap notes)
# ----------------------------------------------------------------------


def test_hwo_null_depth_calibrated_non_negative(fiducial_state) -> None:
    """Calibrated null depth must be finite and non-negative.

    The 36-mode kernel chip is still a surrogate, but it must yield a
    finite raw null depth so downstream performance metrics stay usable.
    """

    perf = next(
        payload
        for name, payload in fiducial_state.results.components.items()
        if name.startswith("performance.standard")
    )
    null_depth = float(perf["null_depth_calibrated"])
    assert math.isfinite(null_depth), (
        f"null_depth_calibrated is {null_depth!r} (NaN means the chip "
        "surrogate could not reach a null; see §15.5)."
    )
    assert null_depth >= 0.0


def test_hwo_performance_snr_finite(fiducial_state) -> None:
    """performance.standard must record a finite SNR.

    The HWO config has no atmosphere, AO, or fringe tracker, so the
    performance stage exercises its optional-OPD path.
    """

    perf_components = [
        name
        for name in fiducial_state.results.components
        if name.startswith("performance.standard")
    ]
    assert perf_components
    perf = fiducial_state.results.components[perf_components[0]]
    snr = float(perf["snr"])
    assert math.isfinite(snr), f"performance.standard.snr is {snr!r}."


def test_hwo_signed_real_fixed_opd_chip_stays_in_clean_regime(
    fiducial_state,
) -> None:
    """The signed-real physical mesh should avoid internal fixed-OPD leakage."""

    perf = next(
        payload
        for name, payload in fiducial_state.results.components.items()
        if name.startswith("performance.standard")
    )
    null_depth = float(perf["null_depth_calibrated"])
    assert math.isfinite(null_depth), (
        f"null_depth_calibrated is {null_depth!r}; cannot evaluate "
        "design-floor regime under the §15.5 chip surrogate."
    )
    assert null_depth >= 0.0

    chip_opt = fiducial_state.results.components["chip_optimization"]
    normalized_null = float(chip_opt["optimization_normalized_null_final"])
    assert math.isfinite(normalized_null), (
        f"optimization_normalized_null_final is {normalized_null!r}."
    )
    assert 0.0 <= normalized_null < 1.0e-20
    assert chip_opt["kernel_rescales_per_wavelength"] is False

    chip = next(
        payload
        for name, payload in fiducial_state.results.components.items()
        if name.startswith("chip.kernel_mzi_mesh")
    )
    assert chip["chip_chromaticity_model"] == "parametric"
    assert chip["phase_chromaticity_model"] == "fixed_opd_neff"

    sens = next(
        payload
        for name, payload in fiducial_state.results.components.items()
        if name.startswith("sensitivity.detection_curve")
    )
    realized_null = float(sens["realized_null_depth"])
    disk_floor = float(sens["disk_null_floor"])
    effective_null = float(sens["effective_null_with_disk"])
    assert math.isfinite(realized_null)
    assert math.isfinite(disk_floor)
    assert math.isfinite(effective_null)
    assert 0.0 <= realized_null < 1.0e-20
    assert 0.0 <= disk_floor < 1.0e-8
    assert 0.0 <= effective_null < 1.0e-8

    detector = next(
        payload
        for name, payload in fiducial_state.results.components.items()
        if name.startswith("detector.mkid")
    )
    detector_null = float(detector["null_depth_raw"])
    assert 0.0 <= detector_null < 1.0e-20

    perf = next(
        payload
        for name, payload in fiducial_state.results.components.items()
        if name.startswith("performance.standard")
    )
    assert perf["null_depth_source"] == "sensitivity:sensitivity.detection_curve"
    assert perf["null_depth_effective"] == pytest.approx(effective_null)
    throughput_at_planet = float(
        np.interp(
            fiducial_state.scene.planet_separation_mas,
            np.asarray(sens["separations_mas"], dtype=np.float64),
            np.asarray(sens["planet_throughputs"], dtype=np.float64),
        )
    )
    assert perf["planet_dark_throughput_used"] == pytest.approx(throughput_at_planet)
    assert throughput_at_planet != pytest.approx(0.5)


def test_hwo_contrast_sweep_snr_monotonic(sweep_cells) -> None:
    """SNR must monotonically decrease as planet contrast deepens.

    Deeper planet contrast (smaller value) → fewer planet photons →
    lower SNR. Requires performance.standard to be in the pipeline; see
    surrogate-gap note.
    """

    snr_by_contrast: dict[float, float] = {}
    for cell in sweep_cells:
        coord_map = dict(cell.coord_tuple)
        contrast = float(coord_map.get("scene.planet.contrast", math.nan))
        if math.isnan(contrast):
            continue
        state = run(cell.config).state
        perf_components = [
            name
            for name in state.results.components
            if name.startswith("performance.standard")
        ]
        assert perf_components, (
            "performance.standard not in pipeline; see surrogate-gap note."
        )
        snr_by_contrast[contrast] = float(
            state.results.components[perf_components[0]]["snr"]
        )

    # Order from deepest (1e-12) to shallowest (1e-8); SNR must
    # monotonically increase in that direction (so non-decreasing when
    # we walk shallow→deep, equivalently non-increasing when we walk
    # deep→shallow on contrast magnitude).
    ordered_contrasts = sorted(snr_by_contrast.keys())
    snrs_in_order = [snr_by_contrast[c] for c in ordered_contrasts]
    for previous, current in zip(snrs_in_order, snrs_in_order[1:]):
        assert current >= previous, (
            "SNR must be non-decreasing as planet contrast shallows; got "
            f"{snrs_in_order} for contrasts {ordered_contrasts}."
        )
