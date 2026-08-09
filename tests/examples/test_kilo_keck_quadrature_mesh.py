"""Cheap checks for the KILO Keck duplicated-input Clements test config."""

from __future__ import annotations

from pathlib import Path

import pytest

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


@pytest.mark.parametrize(
    "config_path",
    [KILO_KECK_6_CONFIG_PATH, KILO_KECK_6_JOINT_CRLB_CONFIG_PATH],
    ids=["kilo_keck_6", "kilo_keck_6_joint_crlb"],
)
def test_tracked_kilo_keck_6_configs_load(config_path: Path) -> None:
    """Keep the tracked six-subaperture configs exercised at test time."""
    config = load_config(str(config_path))

    assert config.pipeline.stages


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
