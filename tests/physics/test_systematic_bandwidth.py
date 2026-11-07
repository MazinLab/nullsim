"""Tests for ``effective_systematic_bandwidth_hz``.

The blend weights each process by its share of the band leakage
variance (review 2026-11-07 row 27), not by its input variance.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nullsim.physics.realized_null import realized_null_depth_mc
from nullsim.physics.systematic_bandwidth import (
    effective_systematic_bandwidth_hz,
)


def test_collapses_to_f_piston_when_piston_dominates() -> None:
    f_eff = effective_systematic_bandwidth_hz(
        piston_leakage_variance=1e-12,
        amplitude_leakage_variance=0.0,
        f_piston_hz=100.0,
        f_amp_hz=5000.0,
    )
    assert f_eff == pytest.approx(100.0, rel=1.0e-12)


def test_collapses_to_f_amp_when_scintillation_dominates() -> None:
    f_eff = effective_systematic_bandwidth_hz(
        piston_leakage_variance=0.0,
        amplitude_leakage_variance=1e-12,
        f_piston_hz=100.0,
        f_amp_hz=5000.0,
    )
    assert f_eff == pytest.approx(5000.0, rel=1.0e-12)


def test_blend_formula_matches_harmonic_weighting() -> None:
    """1/f_eff = w_phi/f_piston + w_I/f_amp with leakage-variance weights."""

    v_phi, v_I = 3.0e-13, 1.0e-13
    f_eff = effective_systematic_bandwidth_hz(
        piston_leakage_variance=v_phi,
        amplitude_leakage_variance=v_I,
        f_piston_hz=100.0,
        f_amp_hz=5000.0,
    )
    assert f_eff == pytest.approx(1.0 / (0.75 / 100.0 + 0.25 / 5000.0), rel=1e-12)


def test_zero_variance_falls_back_to_max_bandwidth() -> None:
    f_eff = effective_systematic_bandwidth_hz(
        piston_leakage_variance=0.0,
        amplitude_leakage_variance=0.0,
        f_piston_hz=100.0,
        f_amp_hz=5000.0,
    )
    assert f_eff == pytest.approx(5000.0, rel=1.0e-12)


def test_negative_variance_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        effective_systematic_bandwidth_hz(
            piston_leakage_variance=-1.0,
            amplitude_leakage_variance=0.0,
            f_piston_hz=100.0,
            f_amp_hz=5000.0,
        )


def test_equal_input_variances_give_unequal_leakage_weights() -> None:
    """Row 27 regression on a Bracewell pair in the quadratic regime.

    With equal input RMS sigma for piston phase and fractional intensity,
    the leakage is Delta phi^2 / 4 for piston (Delta phi ~ N(0, 2 sigma^2))
    and Delta a^2 / 4 for amplitude (Delta a ~ N(0, sigma^2 / 2)), so the
    leakage variances are 2 (sigma^2/2)^2 and 2 (sigma^2/8)^2: piston
    carries 16/17 of the variance, not the 1/2 input weighting gives.
    """

    combiner = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / math.sqrt(2.0)
    wavelength_m = 1.25e-6
    sigma = 1.0e-3
    common = dict(
        u_stack=combiner[:, :, np.newaxis],
        stellar_field=np.ones(2, dtype=np.complex128),
        dark_indices=np.array([1]),
        wavelengths_m=np.array([wavelength_m]),
        n_realizations_per_azimuth=40_000,
        seed=3,
    )
    piston = realized_null_depth_mc(
        ao_piston_opd_rms_per_sub_m=sigma * wavelength_m / (2.0 * math.pi), **common,
    )
    amplitude = realized_null_depth_mc(
        ao_piston_opd_rms_per_sub_m=0.0, scintillation_index_per_sub=sigma, **common,
    )
    v_phi = piston.null_fluctuation_rms**2
    v_I = amplitude.null_fluctuation_rms**2
    assert v_phi == pytest.approx(2.0 * (sigma**2 / 2.0) ** 2, rel=0.05)
    assert v_I == pytest.approx(2.0 * (sigma**2 / 8.0) ** 2, rel=0.05)

    f_eff = effective_systematic_bandwidth_hz(
        piston_leakage_variance=v_phi,
        amplitude_leakage_variance=v_I,
        f_piston_hz=100.0,
        f_amp_hz=5000.0,
    )
    leakage_weighted = 1.0 / ((16.0 / 17.0) / 100.0 + (1.0 / 17.0) / 5000.0)
    input_weighted = 1.0 / (0.5 / 100.0 + 0.5 / 5000.0)
    assert f_eff == pytest.approx(leakage_weighted, rel=0.01)
    assert f_eff < 0.6 * input_weighted
