"""Tests for the wavefront.space_residual stage."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from nullsim.pipeline import RunContext, SimulationState, Wavefront
from nullsim.stages.wavefront import SpaceWavefrontResidual


def _context(tmp_path: Path) -> RunContext:
    return RunContext(
        rng=np.random.default_rng(0),
        config_hash="space-wavefront-residual",
        sweep_coord_tuple=(),
        instance_id="space_wavefront",
        run_dir=tmp_path,
    )


def test_space_residual_updates_wavefront_and_opd_budget(tmp_path: Path) -> None:
    state = SimulationState.empty({"wavelength_grid_um": (0.5, 1.0)})
    stage = SpaceWavefrontResidual(
        {
            "static_wfe_rms_pm": 10.0,
            "dynamic_wfe_rms_pm": 5.0,
            "amplitude_jitter_fraction": 0.02,
            "pointing_jitter_rms_mas": 0.5,
        },
        _context(tmp_path),
    )

    out = stage.apply(state)
    payload = out.wavefront.components["wavefront.space_residual"]
    total_wfe_pm = math.hypot(10.0, 5.0)
    wavelengths_m = np.array([0.5e-6, 1.0e-6], dtype=np.float64)
    expected_strehl = np.exp(-((2.0 * math.pi * total_wfe_pm * 1.0e-12) / wavelengths_m) ** 2)

    np.testing.assert_allclose(out.wavefront.strehl_ratio, expected_strehl)
    np.testing.assert_allclose(out.wavefront.amplitude_jitter_fraction, 0.02)
    assert payload["total_wfe_rms_pm"] == pytest.approx(total_wfe_pm)
    assert payload["pointing_jitter_rms_mas"] == pytest.approx(0.5)
    assert out.opd_budget.total_rms_opd_nm == pytest.approx(total_wfe_pm * 1.0e-3)
    assert out.opd_budget.components["wavefront.space_residual"].rms_opd_nm == (
        pytest.approx(total_wfe_pm * 1.0e-3)
    )


def test_space_residual_combines_with_upstream_wavefront(tmp_path: Path) -> None:
    base = SimulationState.empty({"wavelength_grid_um": (1.0,)})
    state = replace(
        base,
        wavefront=Wavefront(
            wavelength_grid_um=(1.0,),
            strehl_ratio=(0.9,),
            amplitude_jitter_fraction=(0.03,),
        ),
    )
    stage = SpaceWavefrontResidual(
        {
            "static_wfe_rms_pm": 10.0,
            "amplitude_jitter_fraction": 0.04,
        },
        _context(tmp_path),
    )

    out = stage.apply(state)

    residual_strehl = math.exp(-((2.0 * math.pi * 10.0e-12) / 1.0e-6) ** 2)
    assert out.wavefront.strehl_ratio[0] == pytest.approx(0.9 * residual_strehl)
    assert out.wavefront.amplitude_jitter_fraction[0] == pytest.approx(
        math.hypot(0.03, 0.04)
    )


def test_space_residual_rejects_bad_params() -> None:
    with pytest.raises(ValidationError):
        SpaceWavefrontResidual.Params(static_wfe_rms_pm=-1.0)
    with pytest.raises(ValidationError):
        SpaceWavefrontResidual.Params(pointing_jitter_rms_mas=-1.0)
