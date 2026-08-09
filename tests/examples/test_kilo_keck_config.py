"""Cheap checks for the KILO Keck example configuration."""

from __future__ import annotations

import math
import tomllib
from pathlib import Path

import numpy as np

from nullsim.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "examples" / "kilo_keck.toml"
KILO_KECK_6_CONFIG_PATH = PROJECT_ROOT / "examples" / "kilo_keck_6.toml"


def _resolved_stage_dicts(config_path: Path) -> dict[str, dict[str, object]]:
    resolved_config = load_config(str(config_path))
    return {
        str(stage.id): stage.model_dump(mode="python")
        for stage in resolved_config.pipeline.stages
    }


def test_kilo_keck_uses_kilo_geodetic_enu_orientation() -> None:
    """The chip optimizer depends on the actual Keck I/II ENU orientation."""

    config = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    stages = config["pipeline"]["stages"]
    array_stage = next(stage for stage in stages if stage["id"] == "array")

    positions = np.asarray(
        array_stage["aperture_positions_enu_m_explicit"],
        dtype=np.float64,
    )
    expected = np.asarray(
        [
            [-26.8258640, -34.0022240, 0.0],
            [26.8258640, 34.0022240, 0.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(positions, expected, atol=1.0e-7, rtol=0.0)
    baseline = positions[1] - positions[0]
    assert math.isclose(float(np.linalg.norm(baseline)), 86.6205106, rel_tol=1e-7)
    assert abs(float(baseline[0])) > 1.0
    assert abs(float(baseline[1])) > 1.0


def test_kilo_keck_wires_phase_b_science_stage_params() -> None:
    """Sensitivity and science metrics should use the Phase B mitigation knobs."""

    config = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    stages = {stage["id"]: stage for stage in config["pipeline"]["stages"]}

    sensitivity = stages["sensitivity"]
    # 2026-06-10: fixed placeholder terms replaced by the photon-limited
    # auto trim (dither and held setpoint derived from the run's photon
    # budget; see physics.fringe_tracking.photon_limited_setpoint_trim).
    assert sensitivity["auto_setpoint_trim"] is True
    assert sensitivity["trim_loop_bandwidth_hz"] == 1.0
    assert "null_setpoint_residual_nm" not in sensitivity
    assert "setpoint_dither_rms_nm" not in sensitivity

    for stage_id in ("performance", "characterization", "characterization_curve"):
        assert stages[stage_id]["nsc_systematic_reduction"] == 10.0


def test_kilo_keck_6_drops_pre_sensitivity_observability_stage() -> None:
    """The N=6 config should rely on the OPD-budget sensitivity path."""

    raw_config = tomllib.loads(KILO_KECK_6_CONFIG_PATH.read_text(encoding="utf-8"))
    raw_stage_ids = [stage["id"] for stage in raw_config["pipeline"]["stages"]]

    # 6 inherited-stage overrides plus the dual-pol chain (hwp, pbs,
    # pm_fiber_jones), the chip deployment_mode override, and the
    # background n_polarizations override.
    assert len(raw_stage_ids) == 11
    assert "fringe_observability_pre_sensitivity" not in raw_stage_ids

    resolved_config = load_config(str(KILO_KECK_6_CONFIG_PATH))
    stage_ids = [stage.id for stage in resolved_config.pipeline.stages]
    stages = {stage.id: stage for stage in resolved_config.pipeline.stages}

    assert len(stage_ids) == 22
    assert "fringe_observability_pre_sensitivity" not in stage_ids
    assert stage_ids.index("sensitivity") < stage_ids.index("fringe_observability")
    assert stages["sensitivity"].ao_piston_residual_source == "opd_budget"


def test_kilo_keck_port_topology_comes_from_instrument_section() -> None:
    """Root KILO configs should not duplicate the shared port topology."""

    expected_parent = {
        "n_modes": 48,
        "n_input_modes": 24,
        "n_bright_ports": 24,
        "n_dark_ports": 23,
        "n_spectro_ports": 1,
    }
    expected_child = {
        "n_modes": 24,
        "n_input_modes": 12,
        "n_bright_ports": 12,
        "n_dark_ports": 11,
        "n_spectro_ports": 1,
    }

    for config_path, expected in (
        (CONFIG_PATH, expected_parent),
        (KILO_KECK_6_CONFIG_PATH, expected_child),
    ):
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        assert config["instrument"] == expected
        for stage in config["pipeline"]["stages"]:
            repeated_keys = set(expected).intersection(stage)
            assert not repeated_keys, (
                f"{config_path.name} stage {stage['id']} repeats instrument "
                f"keys {sorted(repeated_keys)}"
            )


def test_kilo_keck_includes_characterization_stage() -> None:
    """The characterization.spectro stage must be wired into the
    pipeline with a port split consistent with the chip optimizer.
    Catches accidental removal or port-count drift."""

    config = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    stages = _resolved_stage_dicts(CONFIG_PATH)
    char_stage = stages["characterization"]
    chip_opt_stage = stages["chip_opt"]

    assert char_stage["type"] == "characterization.spectro"
    assert chip_opt_stage["type"] == "chip_optimization.torch"
    for key in ("n_modes", "n_bright_ports", "n_dark_ports", "n_spectro_ports"):
        assert char_stage[key] == chip_opt_stage[key], (
            f"characterization.{key}={char_stage[key]} must match "
            f"chip_opt.{key}={chip_opt_stage[key]}."
        )
    assert (
        char_stage["n_bright_ports"]
        + char_stage["n_dark_ports"]
        + char_stage["n_spectro_ports"]
        == char_stage["n_modes"]
    )
    assert char_stage["dac_bits"] == chip_opt_stage["dac_bits"], (
        "characterization dac_bits must match the deployed chip's."
    )
    assert char_stage["wl_design_um"] == chip_opt_stage["wl_design_um"]
    assert char_stage["spectrograph"] == "mkid"
    assert char_stage["disk_n_rings"] == 0
    assert char_stage["disk_n_azimuths"] == 0
    assert char_stage["disk_integral_n_radial"] > 0
    assert char_stage["disk_integral_n_pa"] > 0

    tables = config["output"]["tables"]
    assert "characterization_summary" in tables, (
        "output.tables must list characterization_summary so the "
        "characterization payload is dumped."
    )


def test_kilo_keck_uses_staged_maunakea_sky_background() -> None:
    """Sky background should come from the generic Maunakea background stage."""

    config = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    stages = config["pipeline"]["stages"]
    background = next(stage for stage in stages if stage["id"] == "background")
    detector = next(stage for stage in stages if stage["id"] == "detector")
    performance = next(stage for stage in stages if stage["id"] == "performance")

    stage_ids = [stage["id"] for stage in stages]
    assert stage_ids.index("detector") < stage_ids.index("background")
    assert stage_ids.index("background") < stage_ids.index("fringe_tracking")
    assert background["type"] == "background.sky_surface_brightness"
    assert background["site_profile"] == "maunakea"
    assert background["band"] == "J"
    assert detector["band"] == "J"
    assert detector["resolving_power_ref"] == 30.0
    assert detector["lambda_ref_um"] == 1.0
    assert performance["n_sky_background_hz"] == 0.0


def test_kilo_keck_runtime_stages_preserve_spectro_split() -> None:
    """Chip, detector, and sensitivity must keep spectro separate from dark."""

    stages = _resolved_stage_dicts(CONFIG_PATH)
    chip_opt_stage = stages["chip_opt"]
    port_keys = ("n_bright_ports", "n_dark_ports", "n_spectro_ports")
    expected_ports = {key: chip_opt_stage[key] for key in port_keys}
    expected_n_modes = chip_opt_stage["n_modes"]

    for stage_id in ("chip", "detector", "sensitivity"):
        stage = stages[stage_id]
        assert {key: stage[key] for key in port_keys} == expected_ports
        if "n_modes" in stage:
            assert stage["n_modes"] == expected_n_modes
        else:
            assert sum(stage[key] for key in port_keys) == expected_n_modes


def test_kilo_keck_requests_applicable_baseline_plots() -> None:
    """The Keck baseline should emit both UV and image-plane null diagnostics."""

    config = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    assert {
        "detection_space",
        "contrast_curve",
        "null_transmission_map",
        "subaperture_layout",
        "uv_coverage_grid",
        "chip_diagnostics",
        "chip_diagnostics_characterization",
        "fringe_observability",
        "loss_budget",
        "contrast_curves_multiband",
        "debug_report",
    }.issubset(set(config["output"]["plots"]))
    assert config["output"]["null_transmission_map"]["max_records"] == 1
    assert (
        config["output"]["null_transmission_map"]["include_angular_coupling"] is True
    )
