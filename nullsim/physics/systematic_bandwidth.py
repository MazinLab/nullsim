"""Variance-weighted effective bandwidth for the null-instability noise term.

The chip-null leakage has two physically independent contributors:

- AO piston phase residual ``sigma_phi_piston`` (averaged over the
  full pupil; coherence time ~ 1 / f_piston, ~30-250 Hz at Keck J=4),
- Amp-servo scintillation residual ``sigma_I`` (per sub-aperture
  intensity-jitter; coherence time ~ 1 / f_amp, ~4-10 kHz at J=4).

The performance stage uses a single-bandwidth systematic-noise term

    sigma_sys^2 = eps^2 * leak^2 * t / (2 * f_eff)

This module provides :func:`effective_systematic_bandwidth_hz`, the
variance-weighted blend that lets the single-bandwidth formula match
the two-process integral. It follows KILO
``_effective_systematic_bandwidth_hz`` (kilo/studies/natural_design.py:180)
with the per-sub-aperture amplitude term averaged over chip modes.
"""

from __future__ import annotations


__all__ = ["effective_systematic_bandwidth_hz"]


def effective_systematic_bandwidth_hz(
    *,
    sigma_phi_piston_rad: float,
    sigma_I_residual: float,
    n_modes: int,
    f_piston_hz: float,
    f_amp_hz: float,
) -> float:
    """Variance-weighted f_eff for the null-instability systematic floor.

    KILO-derived formula with a corrected amplitude-variance term. The
    local KILO reference divides the piston variance by ``M`` but leaves
    the per-sub-aperture amplitude variance unaveraged; for independent
    sub-aperture residuals, both terms average over the chip modes.

    The chip-null leakage variance has two contributors:

        V_phi ≈ sigma_phi_piston^2 / M   (AO piston residual)
        V_I   ≈ sigma_I_residual^2 / M   (amp-servo residual)

    Each averages independently over an integration of length ``t`` with
    ``N_i = 2 f_i t`` independent samples. The total averaged variance,
    set equal to ``eps^2 * leak^2 * t / (2 f_eff)``, yields

        1 / f_eff = w_phi / f_piston + w_I / f_amp

    with ``w_phi = V_phi / (V_phi + V_I)`` and ``w_I = V_I / (V_phi + V_I)``.
    The result collapses to ``f_piston`` when the AO piston term
    dominates and to ``f_amp`` when scintillation dominates.

    Parameters
    ----------
    sigma_phi_piston_rad
        AO piston phase residual RMS in radians at the science wavelength
        (``mean(ao_opd_rms_per_sub) * 2π/λ``). This is the AO-side
        residual, NOT the bright-port fringe-tracking residual.
    sigma_I_residual
        Amp-servo residual fractional-intensity RMS per sub-aperture
        (dimensionless).
    n_modes
        Total sub-aperture count ``M`` (chip mesh dimension). Used as the
        averaging denominator on the AO and amp-servo variances because
        the chip null averages over ``M`` independent sub-apertures.
    f_piston_hz
        AO piston servo bandwidth (~30-250 Hz at Keck J=4; auto-optimized
        per telescope in KILO).
    f_amp_hz
        Amp-servo bandwidth (~4-10 kHz at J=4; photon-noise-limited
        optimum).

    Returns
    -------
    float
        Variance-weighted effective bandwidth in Hz. When either input
        variance or bandwidth is non-positive, falls back to
        ``max(f_piston_hz, f_amp_hz, 1.0)`` — the safe widest-bandwidth
        choice that does not zero the systematic term.
    """

    if n_modes < 1:
        raise ValueError(f"n_modes must be >= 1, got {n_modes}")

    # NOTE: because BOTH variances divide by M, the mode count cancels
    # exactly in the weights and f_eff is M-independent — only the
    # sigma_phi/sigma_I ratio matters. The parameter is retained for the
    # documented derivation and validation, but changing it cannot change
    # the result (review 2026-06-10 Issue 27c).
    v_phi = (sigma_phi_piston_rad ** 2) / float(n_modes)
    v_I = (sigma_I_residual ** 2) / float(n_modes)
    v_total = v_phi + v_I
    if v_total <= 0.0 or f_piston_hz <= 0.0 or f_amp_hz <= 0.0:
        return float(max(f_piston_hz, f_amp_hz, 1.0))
    w_phi = v_phi / v_total
    w_I = v_I / v_total
    inv_f_eff = w_phi / f_piston_hz + w_I / f_amp_hz
    if inv_f_eff <= 0.0:
        return float(max(f_piston_hz, f_amp_hz, 1.0))
    return float(1.0 / inv_f_eff)
