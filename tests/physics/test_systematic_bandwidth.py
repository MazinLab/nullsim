"""Tests for ``effective_systematic_bandwidth_hz``.

These tests pin the analytic limits of the KILO blended-bandwidth
formula with the corrected per-mode averaging on both piston and
amplitude residuals.
"""

from __future__ import annotations

import pytest

from nullsim.physics.systematic_bandwidth import (
    effective_systematic_bandwidth_hz,
)


def test_collapses_to_f_piston_when_ao_dominates() -> None:
    """sigma_I → 0 ⇒ f_eff = f_piston (AO piston is the only contributor)."""
    f_eff = effective_systematic_bandwidth_hz(
        sigma_phi_piston_rad=0.01,
        sigma_I_residual=0.0,
        n_modes=24,
        f_piston_hz=100.0,
        f_amp_hz=5000.0,
    )
    assert f_eff == pytest.approx(100.0, rel=1.0e-12)


def test_collapses_to_f_amp_when_scintillation_dominates() -> None:
    """sigma_phi → 0 ⇒ f_eff = f_amp (amp servo is the only contributor)."""
    f_eff = effective_systematic_bandwidth_hz(
        sigma_phi_piston_rad=0.0,
        sigma_I_residual=0.04,
        n_modes=24,
        f_piston_hz=100.0,
        f_amp_hz=5000.0,
    )
    assert f_eff == pytest.approx(5000.0, rel=1.0e-12)


def test_blend_formula_matches_kilo_analytic() -> None:
    """Direct check against 1/f_eff = w_phi/f_piston + w_I/f_amp."""
    sigma_phi = 0.007  # rad
    sigma_I = 0.04
    n_modes = 24
    f_piston = 100.0
    f_amp = 5000.0

    f_eff = effective_systematic_bandwidth_hz(
        sigma_phi_piston_rad=sigma_phi,
        sigma_I_residual=sigma_I,
        n_modes=n_modes,
        f_piston_hz=f_piston,
        f_amp_hz=f_amp,
    )

    v_phi = sigma_phi ** 2 / n_modes
    v_I = sigma_I ** 2 / n_modes
    v_total = v_phi + v_I
    w_phi = v_phi / v_total
    w_I = v_I / v_total
    expected_f_eff = 1.0 / (w_phi / f_piston + w_I / f_amp)
    assert f_eff == pytest.approx(expected_f_eff, rel=1.0e-12)


def test_balanced_contributors_land_between_the_two_bandwidths() -> None:
    """When V_phi ≈ V_I, f_eff lands in the harmonic-mean ballpark."""
    sigma_phi = 0.04  # makes V_phi = V_I after both divide by n_modes
    sigma_I = 0.04
    f_eff = effective_systematic_bandwidth_hz(
        sigma_phi_piston_rad=sigma_phi,
        sigma_I_residual=sigma_I,
        n_modes=24,
        f_piston_hz=100.0,
        f_amp_hz=5000.0,
    )
    # Harmonic mean of f_piston and f_amp = 2 / (1/100 + 1/5000) ≈ 196.1 Hz
    expected = 2.0 / (1.0 / 100.0 + 1.0 / 5000.0)
    assert f_eff == pytest.approx(expected, rel=1.0e-12)
    assert 100.0 < f_eff < 5000.0


def test_zero_variance_falls_back_to_max_bandwidth() -> None:
    """If both contributors are zero, fall back to the widest bandwidth."""
    f_eff = effective_systematic_bandwidth_hz(
        sigma_phi_piston_rad=0.0,
        sigma_I_residual=0.0,
        n_modes=24,
        f_piston_hz=100.0,
        f_amp_hz=5000.0,
    )
    assert f_eff == pytest.approx(5000.0, rel=1.0e-12)


def test_invalid_n_modes_raises() -> None:
    with pytest.raises(ValueError, match="n_modes"):
        effective_systematic_bandwidth_hz(
            sigma_phi_piston_rad=0.01,
            sigma_I_residual=0.04,
            n_modes=0,
            f_piston_hz=100.0,
            f_amp_hz=5000.0,
        )


def test_effective_bandwidth_is_mode_count_invariant() -> None:
    """Both variances divide by M, so the weights — and f_eff — cannot
    depend on n_modes (review 2026-06-10 Issue 27c). Pin the invariance so
    a future change to the averaging semantics is a deliberate act."""

    reference = effective_systematic_bandwidth_hz(
        sigma_phi_piston_rad=0.0035,
        sigma_I_residual=0.02,
        n_modes=24,
        f_piston_hz=100.0,
        f_amp_hz=4000.0,
    )
    for n_modes in (1, 6, 48, 480):
        assert effective_systematic_bandwidth_hz(
            sigma_phi_piston_rad=0.0035,
            sigma_I_residual=0.02,
            n_modes=n_modes,
            f_piston_hz=100.0,
            f_amp_hz=4000.0,
        ) == reference
