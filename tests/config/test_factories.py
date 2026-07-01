from __future__ import annotations

import pytest

from nullsim.config import band_grid_sweep_specs, science_band


def test_science_band_returns_near_ir_grid() -> None:
    j_band = science_band("j")

    assert j_band.name == "J"
    assert j_band.wavelength_center_um == pytest.approx(1.25)
    assert j_band.wavelength_bandwidth_um == pytest.approx(0.16)


def test_band_grid_sweep_specs_builds_zipped_grid_and_stage_band_axes() -> None:
    sweeps = band_grid_sweep_specs(
        ("Y", "J", "H"),
        stage_band_params=(
            "pipeline.stages.detector.band",
            "pipeline.stages.background.band",
        ),
    )

    assert sweeps[0] == {
        "param": "grid.wavelength_center_um",
        "values": [1.02, 1.25, 1.65],
        "mode": "zip",
    }
    assert sweeps[1] == {
        "param": "grid.wavelength_bandwidth_um",
        "values": [0.10, 0.16, 0.30],
        "mode": "zip",
    }
    assert sweeps[2]["values"] == ["Y", "J", "H"]
    assert sweeps[3]["param"] == "pipeline.stages.background.band"
