"""Leakage-variance-weighted effective bandwidth for the null-instability term.

The chip-null leakage fluctuates through two physically independent
processes with different correlation times:

- residual piston (AO plus fringe tracking; coherence time ~ 1 / f_piston),
- residual scintillation after the amplitude servo (~ 1 / f_amp).

The performance stage uses a single-bandwidth systematic-noise term

    sigma_sys^2 = sigma_L^2 * t / (2 * f_eff)

This module provides :func:`effective_systematic_bandwidth_hz`, the blend
that lets the single-bandwidth formula match the two-process integral. It
follows KILO ``_effective_systematic_bandwidth_hz``
(kilo/studies/natural_design.py:180) in form, but weights each process by
its share of the band LEAKAGE variance rather than by its input variance
(review 2026-11-07 row 27). The input variances are the wrong weights: a
static offset makes leakage linear in the perturbation, a deep null makes
it quadratic, and the chip couples piston and amplitude with different
strengths, so equal input variance does not mean equal leakage variance.
"""

from __future__ import annotations


__all__ = ["effective_systematic_bandwidth_hz"]


def effective_systematic_bandwidth_hz(
    *,
    piston_leakage_variance: float,
    amplitude_leakage_variance: float,
    f_piston_hz: float,
    f_amp_hz: float,
) -> float:
    """Harmonic blend of the two servo bandwidths, weighted by leakage variance.

    Each process averages independently over an integration of length
    ``t`` with ``N_i = 2 f_i t`` independent samples. Setting the summed
    averaged variance ``V_phi / (2 f_piston t) + V_I / (2 f_amp t)`` equal
    to ``(V_phi + V_I) / (2 f_eff t)`` gives

        1 / f_eff = w_phi / f_piston + w_I / f_amp,
        w_i = V_i / (V_phi + V_I).

    The result collapses to ``f_piston`` when piston dominates the leakage
    variance and to ``f_amp`` when scintillation does.

    Parameters
    ----------
    piston_leakage_variance
        Variance of the band leakage with piston as the only perturbation
        (realized-null MC run with amplitude off), null-depth units squared.
    amplitude_leakage_variance
        Variance of the band leakage with amplitude as the only
        perturbation (MC run with piston off), same units.
    f_piston_hz
        Piston servo bandwidth.
    f_amp_hz
        Amplitude servo bandwidth (or the scintillation correlation
        bandwidth for an open-loop amplitude term).

    Returns
    -------
    float
        Effective bandwidth in Hz. When both variances or either bandwidth
        is non-positive, falls back to ``max(f_piston_hz, f_amp_hz, 1.0)``,
        the widest bandwidth, which does not zero the systematic term.
    """

    v_phi = float(piston_leakage_variance)
    v_I = float(amplitude_leakage_variance)
    if v_phi < 0.0 or v_I < 0.0:
        raise ValueError(
            "leakage variances must be non-negative, got "
            f"{v_phi} (piston) and {v_I} (amplitude)."
        )
    v_total = v_phi + v_I
    if v_total <= 0.0 or f_piston_hz <= 0.0 or f_amp_hz <= 0.0:
        return float(max(f_piston_hz, f_amp_hz, 1.0))
    inv_f_eff = (v_phi / f_piston_hz + v_I / f_amp_hz) / v_total
    return float(1.0 / inv_f_eff)
