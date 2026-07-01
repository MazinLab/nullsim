"""Config checks preventing duplicate SMF injection-coupling losses."""

from __future__ import annotations

from pathlib import Path

import pytest

from nullsim.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "config_name",
    [
        "hwo_space.toml",
        "kilo_keck.toml",
        "test/kilo_maunakea.toml",
        "nayra_eelt.toml",
        "test/nayra_eelt_baseline.toml",
    ],
)
def test_example_configs_apply_smf_injection_coupling_once(config_name: str) -> None:
    """A config may use either ideal-injection coupling or the SMF stage, not both."""

    config = load_config(str(PROJECT_ROOT / "examples" / config_name))
    stages = tuple(stage.model_dump(mode="python") for stage in config.pipeline.stages)

    has_smf_coupling_stage = any(
        stage.get("type") == "injection.single_mode_fiber"
        for stage in stages
    )
    coupled_ideal_stages = [
        stage
        for stage in stages
        if stage.get("type") == "injection.ideal"
        and bool(stage.get("apply_strehl_coupling", True))
    ]

    assert not (has_smf_coupling_stage and coupled_ideal_stages), (
        f"{config_name} applies Ruilier-Cassaing injection coupling both in "
        "injection.ideal and injection.single_mode_fiber."
    )
