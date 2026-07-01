"""Cheap structural checks for the KILO Maunakea example config."""

from __future__ import annotations

from pathlib import Path

from nullsim.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "examples" / "test" / "kilo_maunakea.toml"


def test_kilo_maunakea_fringe_tracking_uses_full_pupil_crossing_scale() -> None:
    config = load_config(str(CONFIG_PATH))
    stages = {
        str(stage.id): stage.model_dump(mode="python")
        for stage in config.pipeline.stages
    }

    array_stage = stages["array"]
    ao_stage = stages["ao"]
    fringe_tracking = stages["fringe_tracking"]
    background = stages["background"]
    sensitivity = stages["sensitivity"]
    performance = stages["performance"]

    assert array_stage["diameter_m"] == 10.0
    assert ao_stage["pupil_template"] == "full_pupil"
    assert "effective_baseline_m" not in fringe_tracking
    assert background["type"] == "background.sky_surface_brightness"
    assert background["site_profile"] == "maunakea"
    assert sensitivity["type"] == "sensitivity.detection_curve"
    assert sensitivity["n_bright_ports"] == 3
    assert sensitivity["n_dark_ports"] == 1
    assert sensitivity["include_angular_coupling"] is True
    assert sensitivity["auto_amplitude_servo_residual"] is True
    assert performance["compute_contrast_curve"] is True
    assert performance["with_fringe_tracking"] is True
    assert {
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
    }.issubset(set(config.output.plots))
    assert "null_transmission_map" in config.output.plots
    assert config.output.null_transmission_map is not None
    assert config.output.null_transmission_map.max_records == 1
