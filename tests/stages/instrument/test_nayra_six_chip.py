"""Tests for the NAYRA six-chip architecture manifest stage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from nullsim.pipeline import RunContext, SimulationState
from nullsim.stages.instrument import NayraSixChipArchitecture


def _context(tmp_path: Path) -> RunContext:
    return RunContext(
        rng=np.random.default_rng(0),
        config_hash="nayra-six-chip",
        sweep_coord_tuple=(),
        instance_id="six_chip",
        run_dir=tmp_path,
    )


def test_nayra_six_chip_manifest_records_six_channels(tmp_path: Path) -> None:
    stage = NayraSixChipArchitecture({}, _context(tmp_path))

    result = stage.apply(SimulationState.empty({"wavelength_center_um": 1.25}))

    component = result.results.components["instrument.nayra_six_chip"]
    assert component["n_channels"] == 6
    assert component["n_band_designs"] == 3
    channels = component["channels"]
    assert {(ch["band"], ch["pol_arm"]) for ch in channels} == {
        ("Y", "A"),
        ("Y", "B"),
        ("J", "A"),
        ("J", "B"),
        ("H", "A"),
        ("H", "B"),
    }
    primary_channels = [ch for ch in channels if ch["is_primary_band"]]
    assert {(ch["band"], ch["pol_arm"]) for ch in primary_channels} == {
        ("J", "A"),
        ("J", "B"),
    }
    assert all(0.0 < ch["frontend_to_chip_transmission"] <= 1.0 for ch in channels)


def test_nayra_six_chip_rejects_port_count_mismatch() -> None:
    with pytest.raises(ValidationError):
        NayraSixChipArchitecture.Params(
            n_modes=18,
            n_bright_ports=4,
            n_dark_ports=12,
            n_spectro_ports=1,
        )
