"""Tests for the KILO-provenance PyramidWFS AO stage."""

from __future__ import annotations

import importlib
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from nullsim.physics.pupil_geometry import EELT_DIAMETER_M
from nullsim.pipeline import (
    OPDComponent,
    RunResult,
    build_stage_instances,
    run,
    validate_dependencies,
)

REFERENCE_CONFIG: dict[str, float] = {
    "wavelength_um": 1.65,
    "diameter_m": 10.0,
    "central_obstruction": 0.24,
    "r0_500nm_m": 0.15,
    "tau0_500nm_s": 2.83e-3,
    "theta0_500nm_arcsec": 1.19,
    "v_wind_m_s": 10.0,
    "guide_star_R_mag": 8.0,
    "airmass": 1.0,
    "piston_subaperture_diameter_m": 1.8,
}

KILO_REFERENCE: dict[str, float] = {
    "wfs_photons_per_subap_per_frame": 73.60971154412574,
    "wfs_noise_phase_variance_at_wfs_rad2": 0.02726260624964504,
    "residual_fitting_rad2": 0.036530458245298755,
    "residual_temporal_rad2": 0.00833128845804355,
    "residual_wfs_noise_rad2": 0.004906768434279547,
    "residual_total_rad2": 0.049768515137621854,
    "strehl": 0.9514491641576695,
    "piston_temporal_rad2": 0.003837321342996607,
    "piston_fitting_rad2": 4.3048451547883666e-06,
    "piston_wfs_noise_rad2": 8.075372145221129e-05,
    "piston_total_rad2": 0.003922379909603607,
    "ao_temporal_opd_nm": 16.270224327845456,
    "ao_fitting_opd_nm": 0.5448573867633121,
    "ao_wfs_noise_opd_nm": 2.3598551618844352,
    "ao_piston_opd_rms_nm": 16.449761459191763,
}


def test_pyramid_wfs_pipeline_runs_after_kolmogorov_atmosphere(tmp_path: Path) -> None:
    result = _run_pyramid_pipeline(tmp_path)

    assert "atmosphere" in result.state.wavefront.components
    assert "ao_residual" in result.state.wavefront.components
    assert result.state.wavefront.wavelength_grid_um == (REFERENCE_CONFIG["wavelength_um"],)


def test_pyramid_wfs_improves_strehl_over_open_loop_atmosphere(tmp_path: Path) -> None:
    result = _run_pyramid_pipeline(tmp_path)

    atmosphere_component = result.state.wavefront.components["atmosphere"]
    ao_component = result.state.wavefront.components["ao_residual"]
    pre_ao_strehl = atmosphere_component["strehl_ratio"][0]
    post_ao_strehl = ao_component["strehl_ratio"][0]

    assert post_ao_strehl > pre_ao_strehl
    assert result.state.wavefront.strehl_ratio == pytest.approx(
        (KILO_REFERENCE["strehl"],),
        rel=0.2,  # 20% absorbs the strict-square (k**2=3025) pitch shift from KILO ref at k=floor(sqrt(3000))=54
    )


def test_pyramid_wfs_records_named_opd_components(tmp_path: Path) -> None:
    result = _run_pyramid_pipeline(tmp_path)

    expected_components = {"atmos_piston", "ao_temporal", "ao_fitting", "ao_wfs_noise"}
    assert expected_components.issubset(result.state.opd_budget.components)

    temporal = result.state.opd_budget.components["ao_temporal"]
    fitting = result.state.opd_budget.components["ao_fitting"]
    wfs_noise = result.state.opd_budget.components["ao_wfs_noise"]

    assert isinstance(temporal, OPDComponent)
    assert isinstance(fitting, OPDComponent)
    assert isinstance(wfs_noise, OPDComponent)
    assert temporal.reduction_stage == "pyramid_wfs"
    assert fitting.reduction_stage == "pyramid_wfs"
    assert wfs_noise.reduction_stage == "pyramid_wfs"


def test_pyramid_wfs_matches_kilo_reference_strehl_and_residuals(
    tmp_path: Path,
) -> None:
    """Reference values use v_eff derived from tau0 = 0.314 r0 / v_bar."""

    result = _run_pyramid_pipeline(tmp_path)

    ao_component = result.state.wavefront.components["ao_residual"]
    residuals = ao_component["residual_phase_variance_rad2"]
    assert residuals["fitting"] == pytest.approx(
        KILO_REFERENCE["residual_fitting_rad2"],
        rel=0.2,  # 20% absorbs the strict-square (k**2=3025) pitch shift from KILO ref at k=floor(sqrt(3000))=54
    )
    assert residuals["temporal"] == pytest.approx(
        KILO_REFERENCE["residual_temporal_rad2"],
        rel=0.2,  # 20% absorbs the strict-square (k**2=3025) pitch shift from KILO ref at k=floor(sqrt(3000))=54
    )
    assert residuals["wfs_noise"] == pytest.approx(
        KILO_REFERENCE["residual_wfs_noise_rad2"],
        rel=0.2,  # 20% absorbs the strict-square (k**2=3025) pitch shift from KILO ref at k=floor(sqrt(3000))=54
    )
    assert residuals["total"] == pytest.approx(
        KILO_REFERENCE["residual_total_rad2"],
        rel=0.2,  # 20% absorbs the strict-square (k**2=3025) pitch shift from KILO ref at k=floor(sqrt(3000))=54
    )
    assert ao_component["strehl_ratio"] == pytest.approx(
        (KILO_REFERENCE["strehl"],),
        rel=0.2,  # 20% absorbs the strict-square (k**2=3025) pitch shift from KILO ref at k=floor(sqrt(3000))=54
    )
    assert ao_component["wfs_photons_per_subap_per_frame"] == pytest.approx(
        KILO_REFERENCE["wfs_photons_per_subap_per_frame"],
        rel=0.2,  # 20% absorbs the strict-square (k**2=3025) pitch shift from KILO ref at k=floor(sqrt(3000))=54
    )
    assert ao_component["wfs_noise_phase_variance_at_wfs_rad2"] == pytest.approx(
        KILO_REFERENCE["wfs_noise_phase_variance_at_wfs_rad2"],
        rel=0.2,  # 20% absorbs the strict-square (k**2=3025) pitch shift from KILO ref at k=floor(sqrt(3000))=54
    )


def test_pyramid_wfs_matches_kilo_reference_piston_opd_budget(
    tmp_path: Path,
) -> None:
    """Reference values use v_eff derived from tau0 = 0.314 r0 / v_bar."""

    result = _run_pyramid_pipeline(tmp_path)

    ao_component = result.state.wavefront.components["ao_residual"]
    piston = ao_component["piston_phase_variance_rad2"]
    assert piston["temporal"] == pytest.approx(
        KILO_REFERENCE["piston_temporal_rad2"],
        rel=0.2,  # 20% absorbs the strict-square (k**2=3025) pitch shift from KILO ref at k=floor(sqrt(3000))=54
    )
    assert piston["fitting"] == pytest.approx(
        KILO_REFERENCE["piston_fitting_rad2"],
        rel=0.2,  # 20% absorbs the strict-square (k**2=3025) pitch shift from KILO ref at k=floor(sqrt(3000))=54
    )
    assert piston["wfs_noise"] == pytest.approx(
        KILO_REFERENCE["piston_wfs_noise_rad2"],
        rel=0.2,  # 20% absorbs the strict-square (k**2=3025) pitch shift from KILO ref at k=floor(sqrt(3000))=54
    )
    assert piston["total"] == pytest.approx(
        KILO_REFERENCE["piston_total_rad2"],
        rel=0.2,  # 20% absorbs the strict-square (k**2=3025) pitch shift from KILO ref at k=floor(sqrt(3000))=54
    )

    assert result.state.opd_budget.components["ao_temporal"].rms_opd_nm == pytest.approx(
        KILO_REFERENCE["ao_temporal_opd_nm"],
        rel=0.2,  # 20% absorbs the strict-square (k**2=3025) pitch shift from KILO ref at k=floor(sqrt(3000))=54
    )
    assert result.state.opd_budget.components["ao_fitting"].rms_opd_nm == pytest.approx(
        KILO_REFERENCE["ao_fitting_opd_nm"],
        rel=0.2,  # 20% absorbs the strict-square (k**2=3025) pitch shift from KILO ref at k=floor(sqrt(3000))=54
    )
    assert result.state.opd_budget.components["ao_wfs_noise"].rms_opd_nm == pytest.approx(
        KILO_REFERENCE["ao_wfs_noise_opd_nm"],
        rel=0.2,  # 20% absorbs the strict-square (k**2=3025) pitch shift from KILO ref at k=floor(sqrt(3000))=54
    )
    assert result.state.opd_budget.total_rms_opd_nm == pytest.approx(
        KILO_REFERENCE["ao_piston_opd_rms_nm"],
        rel=0.2,  # 20% absorbs the strict-square (k**2=3025) pitch shift from KILO ref at k=floor(sqrt(3000))=54
    )


def test_pyramid_wfs_derives_effective_wind_from_tau0(tmp_path: Path) -> None:
    derived = _run_pyramid_pipeline(tmp_path)
    legacy = _run_pyramid_pipeline(
        tmp_path,
        ao_overrides={"derive_wind_from_tau0": False},
    )

    derived_payload = derived.state.wavefront.components["ao_residual"]
    legacy_payload = legacy.state.wavefront.components["ao_residual"]
    derived_temporal = derived_payload["residual_phase_variance_rad2"]["temporal"]
    legacy_temporal = legacy_payload["residual_phase_variance_rad2"]["temporal"]
    expected_wind_m_s = 0.314 * REFERENCE_CONFIG["r0_500nm_m"] / REFERENCE_CONFIG[
        "tau0_500nm_s"
    ]

    assert derived_payload["effective_wind_speed_m_s"] == pytest.approx(
        expected_wind_m_s,
        rel=1.0e-12,
    )
    assert legacy_payload["effective_wind_speed_m_s"] == pytest.approx(
        REFERENCE_CONFIG["v_wind_m_s"],
    )
    assert derived_temporal / legacy_temporal == pytest.approx(
        (expected_wind_m_s / REFERENCE_CONFIG["v_wind_m_s"]) ** (5.0 / 3.0),
        rel=1.0e-12,
    )


def test_pyramid_wfs_derives_airmass_from_scene_altitude(tmp_path: Path) -> None:
    result = _run_pyramid_pipeline(
        tmp_path,
        ao_overrides={"airmass": None},
        scene_overrides={"altitude_deg": 60.0},
    )

    ao_component = result.state.wavefront.components["ao_residual"]
    assert ao_component["airmass_source"] == "scene.altitude_deg"
    assert ao_component["airmass"] == pytest.approx(
        1.0 / math.sin(math.radians(60.0))
    )


def test_pyramid_wfs_explicit_airmass_overrides_scene_altitude(
    tmp_path: Path,
) -> None:
    result = _run_pyramid_pipeline(
        tmp_path,
        ao_overrides={"airmass": 1.1},
        scene_overrides={"altitude_deg": 60.0},
    )

    ao_component = result.state.wavefront.components["ao_residual"]
    assert ao_component["airmass_source"] == "params.airmass"
    assert ao_component["airmass"] == pytest.approx(1.1)


def test_few_actuators_fitting_error_dominates_and_strehl_collapses(
    tmp_path: Path,
) -> None:
    # n_actuators=4 (k=2, the smallest valid square per the strict-k**2
    # validator from Code Review Issue 9): 2×2 grid → pitch = D/2 ~ 5 m,
    # huge fitting error vs r0=0.15 m. Strehl collapses to zero.
    result = _run_pyramid_pipeline(tmp_path, ao_overrides={"n_actuators": 4})

    residuals = result.state.wavefront.components["ao_residual"]["residual_phase_variance_rad2"]
    assert residuals["fitting"] > residuals["temporal"] + residuals["wfs_noise"]
    # Strehl < 1e-3 at k=2 (was < 1e-6 at the original n_actuators=1 which
    # is now rejected by the strict-square validator — k=2 is the smallest
    # valid square).
    assert result.state.wavefront.strehl_ratio[0] < 1.0e-3


def test_very_large_actuator_count_drives_fitting_error_to_zero(
    tmp_path: Path,
) -> None:
    result = _run_pyramid_pipeline(
        tmp_path,
        ao_overrides={
            "n_actuators": 10**12,
            "piston_subaperture_diameter_m": None,
        },
    )

    residuals = result.state.wavefront.components["ao_residual"]["residual_phase_variance_rad2"]
    assert residuals["fitting"] < 1.0e-7


def test_faint_guide_star_wfs_noise_dominates(tmp_path: Path) -> None:
    result = _run_pyramid_pipeline(tmp_path, ao_overrides={"guide_star_R_mag": 20.0})

    residuals = result.state.wavefront.components["ao_residual"]["residual_phase_variance_rad2"]
    assert residuals["wfs_noise"] > residuals["fitting"] + residuals["temporal"]
    assert result.state.wavefront.strehl_ratio[0] < 1.0e-6


def test_bright_guide_star_wfs_noise_approaches_zero(tmp_path: Path) -> None:
    result = _run_pyramid_pipeline(tmp_path, ao_overrides={"guide_star_R_mag": 0.0})

    residuals = result.state.wavefront.components["ao_residual"]["residual_phase_variance_rad2"]
    assert residuals["wfs_noise"] < 1.0e-5


def test_derived_guide_star_r_mag_uses_scene_spectral_type(tmp_path: Path) -> None:
    result = _run_pyramid_pipeline(
        tmp_path,
        scene_overrides={"star_jmag": 5.0, "star_spectral_type": "M3V"},
        ao_overrides={"derive_r_mag_from_scene_jmag": True},
    )

    ao_component = result.state.wavefront.components["ao_residual"]
    assert ao_component["guide_star_R_mag"] == pytest.approx(7.0)
    assert ao_component["guide_star_R_mag_source"] == (
        "scene.star.jmag + spectral-type R-J color"
    )


def test_derived_guide_star_r_mag_defaults_to_f_star(tmp_path: Path) -> None:
    result = _run_pyramid_pipeline(
        tmp_path,
        scene_overrides={"star_jmag": 5.0},
        ao_overrides={"derive_r_mag_from_scene_jmag": True},
    )

    ao_component = result.state.wavefront.components["ao_residual"]
    assert ao_component["guide_star_R_mag"] == pytest.approx(5.4)


def test_nan_r0_propagates_through_ao_strehl_without_raising(tmp_path: Path) -> None:
    result = _run_pyramid_pipeline(
        tmp_path,
        atmosphere_overrides={"r0_500nm_m": math.nan},
    )

    ao_component = result.state.wavefront.components["ao_residual"]
    assert math.isnan(result.state.wavefront.strehl_ratio[0])
    assert math.isnan(ao_component["residual_phase_variance_rad2"]["total"])
    assert math.isnan(result.state.opd_budget.total_rms_opd_nm)


def test_zero_actuators_rejected_by_params() -> None:
    from nullsim.stages.ao import PyramidWFS

    with pytest.raises(ValidationError, match="n_actuators"):
        PyramidWFS.Params(n_actuators=0)


def test_ao_opd_components_sum_in_quadrature(tmp_path: Path) -> None:
    result = _run_pyramid_pipeline(tmp_path)

    component_names = ("ao_temporal", "ao_fitting", "ao_wfs_noise")
    component_rms_nm = [
        result.state.opd_budget.components[name].rms_opd_nm
        for name in component_names
    ]
    quadrature_sum_nm = math.sqrt(sum(value_nm**2 for value_nm in component_rms_nm))
    assert quadrature_sum_nm == pytest.approx(result.state.opd_budget.total_rms_opd_nm)


def test_full_scene_atmosphere_ao_injection_pipeline_validates_and_runs(
    tmp_path: Path,
) -> None:
    _register_stage_modules()
    config = _pyramid_config(tmp_path, include_injection=True)
    instances = build_stage_instances(config)

    validate_dependencies(instances)
    result = run(config)

    assert result.state.field.amplitudes is not None
    assert result.state.field.amplitudes.shape == (1, 1, 1)
    assert "ao_temporal" in result.state.opd_budget.components


def _run_pyramid_pipeline(
    tmp_path: Path,
    *,
    ao_overrides: dict[str, object] | None = None,
    atmosphere_overrides: dict[str, object] | None = None,
    scene_overrides: dict[str, object] | None = None,
    telescope_overrides: dict[str, object] | None = None,
) -> RunResult:
    """Run the future AO stage after the existing scene/telescope/atmosphere path."""

    _register_stage_modules()
    return run(
        _pyramid_config(
            tmp_path,
            ao_overrides=ao_overrides,
            atmosphere_overrides=atmosphere_overrides,
            scene_overrides=scene_overrides,
            telescope_overrides=telescope_overrides,
        )
    )


def _register_stage_modules() -> None:
    """Import stage modules whose registration is needed by this test."""

    importlib.import_module("nullsim.stages.atmosphere.kolmogorov_screens")
    importlib.import_module("nullsim.stages.ao.pyramid_wfs")
    importlib.import_module("nullsim.stages.injection.ideal")
    importlib.import_module("nullsim.stages.scene.point_source")
    importlib.import_module("nullsim.stages.telescope.fixed_aperture")


def _pyramid_config(
    tmp_path: Path,
    *,
    ao_overrides: dict[str, object] | None = None,
    atmosphere_overrides: dict[str, object] | None = None,
    scene_overrides: dict[str, object] | None = None,
    telescope_overrides: dict[str, object] | None = None,
    include_injection: bool = False,
) -> dict[str, object]:
    atmosphere_stage: dict[str, object] = {
        "id": "atmosphere",
        "type": "atmosphere.kolmogorov",
        "r0_500nm_m": REFERENCE_CONFIG["r0_500nm_m"],
    }
    if atmosphere_overrides:
        atmosphere_stage.update(atmosphere_overrides)

    ao_stage: dict[str, object] = {
        "id": "ao",
        "type": "ao.pyramid_wfs",
        "n_actuators": 3025,
        "loop_frequency_hz": 2000.0,
        "guide_star_R_mag": REFERENCE_CONFIG["guide_star_R_mag"],
        "central_obstruction": REFERENCE_CONFIG["central_obstruction"],
        "tau0_500nm_s": REFERENCE_CONFIG["tau0_500nm_s"],
        "theta0_500nm_arcsec": REFERENCE_CONFIG["theta0_500nm_arcsec"],
        "v_wind_m_s": REFERENCE_CONFIG["v_wind_m_s"],
        "airmass": REFERENCE_CONFIG["airmass"],
        "wfs_wavelength_um": 0.7,
        "wfs_bandwidth_um": 0.3,
        "n_subap_across": 50,
        "modulation_lambda_d": 0.0,
        "dichroic_throughput": 0.95,
        "atmospheric_throughput": 0.93,
        "mirror_throughput": 0.86,
        "quantum_efficiency": 0.85,
        "excess_noise_sq": 2.0,
        "readout_noise_e_rms": 0.5,
        "piston_subaperture_diameter_m": REFERENCE_CONFIG[
            "piston_subaperture_diameter_m"
        ],
    }
    if ao_overrides:
        ao_stage.update(ao_overrides)

    telescope_stage: dict[str, object] = {
        "id": "array",
        "type": "telescope.fixed_aperture",
        "diameter_m": REFERENCE_CONFIG["diameter_m"],
    }
    if telescope_overrides:
        telescope_stage.update(telescope_overrides)

    scene_stage: dict[str, object] = {
        "id": "scene",
        "type": "scene.point_source",
        "star_jmag": REFERENCE_CONFIG["guide_star_R_mag"],
    }
    if scene_overrides:
        scene_stage.update(scene_overrides)

    stages: list[dict[str, object]] = [
        scene_stage,
        telescope_stage,
        atmosphere_stage,
        ao_stage,
    ]
    if include_injection:
        stages.append({"id": "injection", "type": "injection.ideal"})

    return {
        "config_hash": "pyramid-wfs-kilo-reference",
        "grid": {"wavelength_grid_um": (REFERENCE_CONFIG["wavelength_um"],)},
        "run": {"seed": 1},
        "cache": {"enabled": False, "dir": tmp_path / "cache", "version": 1},
        "output": {"dir": tmp_path / "run"},
        "pipeline": {"stages": stages},
    }


# ---------------------------------------------------------------------------
# Geometry-derived piston_subaperture_diameter_m (replaces hardcoded TOML).
# ---------------------------------------------------------------------------


def test_ao_derives_keck_n12_d_sub_from_pupil_geometry(tmp_path: Path) -> None:
    """When ``piston_subaperture_diameter_m`` is None and
    ``n_subapertures_per_aperture`` + ``pupil_template="keck"`` are set,
    the stage runs ``nullsim.physics.pupil_geometry`` to derive d_sub
    from the actual Keck hexagonal pupil. Expected ~1.929 m, matching
    KILO's ``optimize_subaperture_positions``.
    """

    result = _run_pyramid_pipeline(
        tmp_path,
        ao_overrides={
            "piston_subaperture_diameter_m": None,
            "n_subapertures_per_aperture": 12,
            "pupil_template": "keck",
        },
    )
    ao_component = result.state.wavefront.components["ao_residual"]
    derived = ao_component["piston_subaperture_diameter_m"]
    packing = result.state.geometry.components["ao_subaperture_packing"]
    assert derived is not None
    assert derived == pytest.approx(1.929, abs=0.001)
    assert ao_component["n_subapertures_per_aperture"] == 12
    assert ao_component["pupil_template"] == "keck"
    assert packing["effective_area_source"] == "packed_subaperture_circles"
    assert packing["piston_subaperture_diameter_m"] == pytest.approx(derived)
    assert packing["n_modes"] == 12
    assert len(packing["mode_areas_m2"]) == packing["n_modes"]
    assert len(packing["mode_diameters_m"]) == packing["n_modes"]
    assert packing["mode_diameters_m"] == pytest.approx([derived] * 12)
    expected_mode_area = math.pi * (derived / 2.0) ** 2
    assert packing["mode_areas_m2"] == pytest.approx([expected_mode_area] * 12)
    assert 0.0 < packing["effective_to_full_collecting_area_fraction"] < 1.0


def test_ao_explicit_d_sub_overrides_derivation(tmp_path: Path) -> None:
    """When both ``piston_subaperture_diameter_m`` AND
    ``n_subapertures_per_aperture`` are set, the explicit value wins
    (so users can still pin a custom value while declaring the geometry).
    """

    result = _run_pyramid_pipeline(
        tmp_path,
        ao_overrides={
            "piston_subaperture_diameter_m": 2.5,
            "n_subapertures_per_aperture": 12,
            "pupil_template": "keck",
        },
    )
    ao_component = result.state.wavefront.components["ao_residual"]
    assert ao_component["piston_subaperture_diameter_m"] == pytest.approx(2.5)


def test_ao_derived_d_sub_changes_piston_filter_fractions(
    tmp_path: Path,
) -> None:
    """Different ``n_subapertures_per_aperture`` values produce different
    derived d_sub values, which propagate to different piston filter
    fractions. Pins the cross-stage flow.
    """

    result_n4 = _run_pyramid_pipeline(
        tmp_path,
        ao_overrides={
            "piston_subaperture_diameter_m": None,
            "n_subapertures_per_aperture": 4,
            "pupil_template": "keck",
        },
    )
    result_n12 = _run_pyramid_pipeline(
        tmp_path,
        ao_overrides={
            "piston_subaperture_diameter_m": None,
            "n_subapertures_per_aperture": 12,
            "pupil_template": "keck",
        },
    )
    d_n4 = result_n4.state.wavefront.components["ao_residual"][
        "piston_subaperture_diameter_m"
    ]
    d_n12 = result_n12.state.wavefront.components["ao_residual"][
        "piston_subaperture_diameter_m"
    ]
    # Fewer subs → each fits a bigger circle inside the pupil.
    assert d_n4 > d_n12


def test_ao_full_pupil_template_uses_aperture_diameter(tmp_path: Path) -> None:
    """``pupil_template="full_pupil"`` derives d_sub directly from
    state.geometry.diameter_m without running the optimizer — used by
    NAYRA E-ELT and KILO 4-telescope Maunakea where the SMF samples the
    entire per-telescope pupil.
    """

    result = _run_pyramid_pipeline(
        tmp_path,
        ao_overrides={
            "piston_subaperture_diameter_m": None,
            "pupil_template": "full_pupil",
        },
    )
    ao_component = result.state.wavefront.components["ao_residual"]
    assert ao_component["piston_subaperture_diameter_m"] == pytest.approx(
        REFERENCE_CONFIG["diameter_m"]
    )
    assert ao_component["pupil_template"] == "full_pupil"


def test_ao_full_pupil_template_does_not_require_n_subapertures(tmp_path: Path) -> None:
    """``pupil_template="full_pupil"`` is a sentinel that means "SMF sees
    the whole pupil", so n_subapertures_per_aperture is not required —
    the resolution short-circuits to aperture_diameter_m.
    """

    # Set neither piston_subaperture_diameter_m nor n_subapertures_per_aperture;
    # full_pupil alone must be enough.
    result = _run_pyramid_pipeline(
        tmp_path,
        ao_overrides={
            "piston_subaperture_diameter_m": None,
            "n_subapertures_per_aperture": None,
            "pupil_template": "full_pupil",
        },
    )
    ao_component = result.state.wavefront.components["ao_residual"]
    assert ao_component["piston_subaperture_diameter_m"] == pytest.approx(
        REFERENCE_CONFIG["diameter_m"]
    )


def test_ao_populates_single_eelt_subaperture_positions(tmp_path: Path) -> None:
    result = _run_pyramid_pipeline(
        tmp_path,
        telescope_overrides={"diameter_m": EELT_DIAMETER_M},
        ao_overrides={
            "piston_subaperture_diameter_m": None,
            "n_subapertures_per_aperture": 18,
            "pupil_template": "eelt",
        },
    )

    positions = result.state.geometry.aperture_positions_pupil_m
    assert positions is not None
    assert positions.shape == (18, 2)
    assert result.state.geometry.effective_collecting_area_m2 < (
        result.state.geometry.collecting_area_m2
    )
