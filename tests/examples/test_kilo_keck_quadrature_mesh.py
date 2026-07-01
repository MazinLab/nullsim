"""Cheap checks for the KILO Keck duplicated-input Clements test config."""

from __future__ import annotations

from pathlib import Path

from nullsim.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "examples" / "test" / "kilo_keck_quadrature_mesh.toml"
KILO_KECK_6_CONFIG_PATH = PROJECT_ROOT / "examples" / "test" / "kilo_keck_6.toml"
KILO_KECK_6_JOINT_CRLB_CONFIG_PATH = (
    PROJECT_ROOT / "examples" / "test" / "kilo_keck_6_joint_crlb.toml"
)


def _stages_by_id() -> dict[str, object]:
    config = load_config(str(CONFIG_PATH))
    return {stage.id: stage for stage in config.pipeline.stages}


def test_kilo_keck_quadrature_mesh_inherits_and_inserts_splitter() -> None:
    config = load_config(str(CONFIG_PATH))
    stages = _stages_by_id()
    stage_ids = [stage.id for stage in config.pipeline.stages]

    assert config.run.name == "kilo_keck_quadrature_mesh"
    assert "chip_opt" in stages
    assert stage_ids.index("quadrature_split") < stage_ids.index("chip_opt")
    assert "calibration" not in stages
    assert "characterization" not in stages
    assert "characterization_curve" not in stages
    assert stages["quadrature_split"].type == "injection.mode_splitter"
    assert stages["quadrature_split"].copies == 2
    assert stages["chip_opt"].type == "chip_optimization.torch"
    assert stages["chip_opt"].n_modes == 48
    assert stages["chip_opt"].kernel_order == 3
    assert stages["chip_opt"].disk_n_rings == 3
    assert stages["chip_opt"].disk_n_azimuths == 4
    assert stages["chip_opt"].tracker_observability_weight == 1.0e-3
    assert stages["chip_opt"].tracker_observability_min_singular_per_nm == 1.0e-3
    assert stages["chip_opt"].tracker_observability_n_spectral_channels == 1
    assert stages["chip"].type == "chip.kernel_mzi_mesh"
    assert stages["chip"].n_modes == 48
    assert stages["chip"].n_bright_ports == 24
    assert stages["chip"].n_dark_ports == 23
    assert stages["chip"].n_spectro_ports == 1


def test_kilo_keck_quadrature_mesh_expands_output_port_counts() -> None:
    stages = _stages_by_id()

    assert stages["detector"].n_modes == 48
    assert stages["detector"].n_bright_ports == 24
    assert stages["detector"].n_dark_ports == 23
    assert stages["detector"].n_spectro_ports == 1
    assert stages["sensitivity"].n_bright_ports == 24
    assert stages["sensitivity"].n_dark_ports == 23
    assert stages["sensitivity"].n_spectro_ports == 1
    assert stages["fringe_observability"].n_spectral_channels == 1
    assert list(stages["fringe_observability"].gauss_newton_piston_rms_grid_nm) == [
        1.0,
        3.0,
        10.0,
        30.0,
        100.0,
    ]


def test_kilo_keck_quadrature_mesh_keeps_clements_diagnostics() -> None:
    config = load_config(str(CONFIG_PATH))

    assert "characterization_summary" not in config.output.tables
    assert {
        "chip_diagnostics",
        "cophasing_unitary",
        "contrast_curve",
        "null_transmission_map",
        "fringe_observability",
        "debug_report",
    }.issubset(set(config.output.plots))


def test_kilo_keck_6_characterization_retunes_against_finite_disk() -> None:
    """N=6 characterization suppresses the resolved-star spectro leakage."""

    config = load_config(str(KILO_KECK_6_CONFIG_PATH))
    stages = {stage.id: stage for stage in config.pipeline.stages}

    chip_opt = stages["chip_opt"]
    assert chip_opt.amplitude_observability_weight == 0.0
    assert chip_opt.joint_observability_crlb_weight == 1.0e-3
    assert chip_opt.joint_observability_science_null_guard_weight == 0.0
    assert chip_opt.disk_n_rings == 0
    assert chip_opt.disk_n_azimuths == 0

    characterization = stages["characterization"]
    assert characterization.n_restarts == 3
    assert characterization.max_iter == 5000
    # n_workers=1: the restart ThreadPool is GIL bound for this eager torch
    # optimizer and oversubscribes the sweep-cell pool (see config comment).
    assert characterization.n_workers == 1
    assert characterization.broadband_n_wavelengths == 1
    assert characterization.use_upstream_chip_params_start is False
    assert characterization.disk_n_rings == 3
    assert characterization.disk_n_azimuths == 4
    assert characterization.disk_integral_n_radial > 0
    assert characterization.disk_integral_n_pa > 0

    curve = stages["characterization_curve"]
    assert curve.n_anchors == 12
    assert curve.separation_min_mas == 0.5944725589892117
    assert curve.separation_max_mas == 300.0
    assert curve.n_restarts == 3
    assert curve.max_iter == 2000
    assert curve.n_workers == 1
    assert curve.seed == 2060
    assert curve.disk_n_rings == 3
    assert curve.disk_n_azimuths == 4
    assert curve.disk_integral_n_radial > 0
    assert curve.disk_integral_n_pa > 0
    assert not hasattr(curve, "tracker_observability_crlb_constraint_nm")
    assert curve.use_upstream_chip_params_start is False


def test_kilo_keck_6_joint_crlb_pins_high_snr_characterization() -> None:
    """Joint-CRLB experiment keeps the archived high-SNR char settings."""

    config = load_config(str(KILO_KECK_6_JOINT_CRLB_CONFIG_PATH))
    stages = {stage.id: stage for stage in config.pipeline.stages}

    chip_opt = stages["chip_opt"]
    assert chip_opt.amplitude_observability_weight == 0.0
    assert chip_opt.joint_observability_crlb_weight == 1.0e-3
    assert chip_opt.joint_observability_science_null_guard_weight == 0.0

    characterization = stages["characterization"]
    assert characterization.wl_design_um == 1.25375
    assert characterization.broadband_n_wavelengths == 1
    assert characterization.n_restarts == 3
    assert characterization.max_iter == 10000
    assert characterization.n_workers == 4
    assert characterization.use_upstream_chip_params_start is False

    curve = stages["characterization_curve"]
    assert curve.wl_design_um == 1.25375
    assert curve.n_anchors == 12
    assert curve.separation_min_mas == 0.5944725589892117
    assert curve.separation_max_mas == 300.0
    assert curve.n_restarts == 3
    assert curve.max_iter == 2000
    assert curve.n_workers == 4
    assert curve.seed == 2060
    assert not hasattr(curve, "tracker_observability_crlb_constraint_nm")
    assert curve.snr_proxy_weight == 4.0
    assert curve.broadband_n_wavelengths == 5
