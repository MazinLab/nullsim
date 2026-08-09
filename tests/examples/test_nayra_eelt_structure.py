"""Cheap structural checks for the NAYRA E-ELT example config."""

from __future__ import annotations

from pathlib import Path

from nullsim.config import expand_sweeps, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "examples" / "nayra_eelt.toml"


def test_nayra_config_wires_dual_pol_frontend_before_chip() -> None:
    config = load_config(str(CONFIG_PATH))
    stages = tuple(stage.model_dump(mode="python") for stage in config.pipeline.stages)
    stage_types_by_id = {str(stage["id"]): str(stage["type"]) for stage in stages}

    assert stage_types_by_id["six_chip_architecture"] == "instrument.nayra_six_chip"
    assert stage_types_by_id["hwp"] == "polarization.hwp"
    assert stage_types_by_id["pbs"] == "polarization.pbs_split"
    assert stage_types_by_id["pm_fiber_jones"] == "polarization.pm_fiber"
    assert stage_types_by_id["background"] == "background.sky_surface_brightness"
    background = next(stage for stage in stages if str(stage["id"]) == "background")
    assert background["site_profile"] == "cerro_armazones"

    stage_ids = [str(stage["id"]) for stage in stages]
    assert stage_ids.index("injection") < stage_ids.index("hwp")
    assert stage_ids.index("hwp") < stage_ids.index("pbs")
    assert stage_ids.index("pbs") < stage_ids.index("transport")
    assert stage_ids.index("transport") < stage_ids.index("pm_fiber_jones")
    assert stage_ids.index("pm_fiber_jones") < stage_ids.index("chip_opt")
    chip_stage = next(stage for stage in stages if str(stage["id"]) == "chip")
    assert chip_stage["deployment_mode"] == "per_pol_arm"


def test_nayra_fringe_tracking_uses_eelt_subaperture_crossing_scale() -> None:
    config = load_config(str(CONFIG_PATH))
    stages = {
        str(stage.id): stage.model_dump(mode="python")
        for stage in config.pipeline.stages
    }

    ao_stage = stages["ao"]
    fringe_tracking = stages["fringe_tracking"]

    assert ao_stage["pupil_template"] == "eelt"
    assert ao_stage["n_subapertures_per_aperture"] == 18
    assert "effective_baseline_m" not in fringe_tracking
    assert fringe_tracking["use_full_piston_servo_residual"] is True
    assert fringe_tracking["control_model"] == "photon_kalman"
    assert fringe_tracking["inter_telescope_piston"] is False


def test_nayra_simultaneous_bands_match_six_chip_manifest() -> None:
    config = load_config(str(CONFIG_PATH))
    stages = {
        str(stage.id): stage.model_dump(mode="python")
        for stage in config.pipeline.stages
    }

    expected_bands = ["J", "H", "K"]
    assert stages["fringe_tracking"]["simultaneous_science_bands"] == expected_bands
    assert stages["sensitivity"]["simultaneous_science_bands"] == expected_bands


def test_nayra_six_chip_manifest_matches_instrument_quadrature_split() -> None:
    """N=18 quadrature: [instrument] is the single port-topology source.

    The old N=48 config pinned per-stage port counts through ~25 zipped
    sweep axes and carried a calibration.floor_lookup stage; both are gone.
    """

    config = load_config(str(CONFIG_PATH))
    stages = {
        str(stage.id): stage.model_dump(mode="python")
        for stage in config.pipeline.stages
    }

    instrument = config.instrument
    assert instrument is not None
    assert instrument.n_modes == 36
    assert instrument.n_input_modes == 18
    assert (instrument.n_bright_ports, instrument.n_dark_ports) == (18, 17)

    arch = stages["six_chip_architecture"]
    assert int(arch["n_modes"]) == 36
    assert (
        int(arch["n_bright_ports"]),
        int(arch["n_dark_ports"]),
        int(arch["n_spectro_ports"]),
    ) == (18, 17, 1)

    splitter = stages["quadrature_split"]
    assert int(splitter["copies"]) == 2

    assert "calibration" not in stages
    sweep_params = {axis.param for axis in config.sweep}
    assert sweep_params == {"scene.star.jmag"}

    cells = expand_sweeps(config)
    assert len(cells) == 5
