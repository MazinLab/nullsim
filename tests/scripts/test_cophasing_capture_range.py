"""Tests for the cophasing capture-range script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "cophasing_capture_range.py"
)
_SPEC = importlib.util.spec_from_file_location("cophasing_capture_range", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

random_capture_sweep = _MODULE.random_capture_sweep
single_aperture_sweep = _MODULE.single_aperture_sweep
wrap_opd_m = _MODULE.wrap_opd_m


def test_wrap_opd_m_wraps_to_half_wave_interval() -> None:
    wavelength = 1.25e-6
    values = np.array([-0.75, -0.25, 0.25, 0.75]) * wavelength

    wrapped = wrap_opd_m(values, wavelength)

    assert np.all(wrapped >= -0.5 * wavelength)
    assert np.all(wrapped < 0.5 * wavelength)
    np.testing.assert_allclose(
        wrapped,
        np.array([0.25, -0.25, 0.25, -0.25]) * wavelength,
        atol=1.0e-18,
    )


def test_single_aperture_sweep_fails_after_phase_wrap() -> None:
    sweep = single_aperture_sweep(
        n_modes=4,
        wavelength_m=1.0e-6,
        offset_nm=np.array([100.0, 1000.0]),
    )

    assert sweep.residual_rms_nm[0] < 1.0e-9
    assert sweep.edge_wrap_fraction[0] == 0.0
    assert sweep.residual_rms_nm[1] > 100.0
    assert sweep.edge_wrap_fraction[1] > 0.0


def test_random_capture_sweep_reports_lock_loss_at_large_rms() -> None:
    sweep = random_capture_sweep(
        n_modes=6,
        wavelength_m=1.25e-6,
        injected_rms_nm=np.array([10.0, 1000.0]),
        n_trials=128,
        lock_threshold_nm=10.0,
        seed=5,
    )

    assert sweep.lock_fraction[0] > 0.9
    assert sweep.lock_fraction[1] < 0.1
    assert sweep.edge_wrap_fraction[1] > 0.5
