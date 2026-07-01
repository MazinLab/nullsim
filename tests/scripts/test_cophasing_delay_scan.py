"""Tests for the inter-telescope cophasing delay-scan script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "cophasing_delay_scan.py"
_SPEC = importlib.util.spec_from_file_location("cophasing_delay_scan", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

broadband_packet_metric = _MODULE.broadband_packet_metric
delay_capture_sweep = _MODULE.delay_capture_sweep
estimate_delay_by_scan = _MODULE.estimate_delay_by_scan
wavelength_grid_m = _MODULE.wavelength_grid_m


def test_broadband_packet_metric_peaks_at_zero_delay() -> None:
    wavelengths = wavelength_grid_m(n_wavelengths=64)
    residual = np.array([0.0, 20.0e-6])

    metric = broadband_packet_metric(residual, wavelengths_m=wavelengths)

    assert metric[0] == 1.0
    assert metric[1] < 0.2


def test_delay_scan_recovers_one_millimeter_offset() -> None:
    wavelengths = wavelength_grid_m(n_wavelengths=64)

    estimate = estimate_delay_by_scan(
        true_delay_m=1.0e-3,
        wavelengths_m=wavelengths,
        scan_half_range_m=1.2e-3,
        coarse_step_m=2.0e-6,
        fine_half_width_m=8.0e-6,
        fine_step_m=10.0e-9,
    )

    assert abs(estimate.residual_m) < 10.0e-9
    assert estimate.peak_metric > 0.99


def test_delay_scan_fails_when_true_delay_is_outside_scan_range() -> None:
    wavelengths = wavelength_grid_m(n_wavelengths=64)

    estimate = estimate_delay_by_scan(
        true_delay_m=1.0e-3,
        wavelengths_m=wavelengths,
        scan_half_range_m=100.0e-6,
        coarse_step_m=2.0e-6,
        fine_half_width_m=8.0e-6,
        fine_step_m=10.0e-9,
    )

    assert abs(estimate.residual_m) > 800.0e-6


def test_delay_capture_sweep_locks_across_millimeter_range() -> None:
    wavelengths = wavelength_grid_m(n_wavelengths=64)
    sweep = delay_capture_sweep(
        true_delay_m=np.linspace(-1.0e-3, 1.0e-3, 9),
        wavelengths_m=wavelengths,
        scan_half_range_m=1.2e-3,
        coarse_step_m=2.0e-6,
        fine_half_width_m=8.0e-6,
        fine_step_m=10.0e-9,
        fine_lock_threshold_m=10.0e-9,
    )

    assert np.all(sweep.locked_lambda_over_two)
    assert np.all(sweep.locked_fine)
