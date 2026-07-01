"""Polarimetric differential imaging (PDI) SNR primitives.

Port of KILO ``signal_to_noise_pdi`` (``kilo/performance.py:412-648``) —
SPHERE-IRDIS / GPI / ZIMPOL-style double-difference detection where the
planet's polarized Q+U signature is extracted from a 4-angle Mueller
modulation across two matched PBS-arm dark detectors. KILO_PARITY_PLAN
§5.2.

Static instrument leakage (chip null asymmetry, common-mode AO drift)
cancels to first order in each (A − B) difference; the residual scales
with the PBS axis offset, HWP retardance error, HWP rotation jitter,
and PBS extinction-ratio leakage power. Continuous modulation
(``modulation_period_s > 0``) breaks the leakage into N independent
cycle realizations and lowers the systematic floor accordingly. The
common-mode-rejection ratio (``cmrr_db``) captures the slow-drift
residual the 4-angle scheme cannot cancel.

Photon / dark / sky variance doubles when both Q and U Stokes
parameters are recovered (``recover_both_qu``) because each Stokes
component draws from its own half-integration noise and ``|P|`` =
``sqrt(Q² + U²)`` combines them in quadrature. The systematic floor
does NOT pick up that ``sqrt(2)`` penalty: the same instrument leakage
drives Q and U simultaneously (correlated, not independent).
"""

from __future__ import annotations

import math

__all__ = ["signal_to_noise_pdi"]


def signal_to_noise_pdi(
    *,
    planet_rate_total_hz: float,
    stellar_leakage_rate_total_hz: float,
    integration_time_s: float,
    pol_fraction: float,
    n_dark_ports_per_arm: int = 1,
    pbs_axis_offset_rad: float = 0.0,
    hwp_retardance_error_rad: float = 0.0,
    hwp_rotation_jitter_rad: float = 0.0,
    pbs_extinction_db: float = float("inf"),
    modulation_period_s: float = 0.0,
    cmrr_db: float = 30.0,
    recover_both_qu: bool = True,
    dark_rate_hz_per_port_per_arm: float = 0.0,
    sky_rate_hz_per_dark_port_per_arm: float = 0.0,
) -> float:
    """SNR for polarimetric detection of a polarized planet (KILO port).

    Implements the 4-angle (0, π/8, π/4, 3π/8) double-difference scheme

        D_Q = (N_A(0)    − N_B(0))    − (N_A(π/4)   − N_B(π/4))
        D_U = (N_A(π/8) − N_B(π/8)) − (N_A(3π/8) − N_B(3π/8))

    Args:
        planet_rate_total_hz: Total detected planet rate aggregated over
            both PBS arms, before polarization extraction.
        stellar_leakage_rate_total_hz: Total stellar-leakage rate
            aggregated over both PBS arms.
        integration_time_s: Total integration time across all 4 HWP angles.
        pol_fraction: Planet polarization fraction (0..1).
        n_dark_ports_per_arm: Dark detection ports per PBS arm.
        pbs_axis_offset_rad: PBS analyzer axis offset from nominal.
        hwp_retardance_error_rad: HWP retardance error vs nominal λ/2.
        hwp_rotation_jitter_rad: HWP rotation-angle 1σ jitter per cycle.
        pbs_extinction_db: PBS extinction ratio in dB (default ∞ = perfect).
        modulation_period_s: HWP modulation cycle period; 0 picks the
            single-cycle hard-ceiling variance bound.
        cmrr_db: Common-mode rejection ratio in dB; suppresses the raw
            imperfection amplitude by ``10^(−cmrr_db/20)``.
        recover_both_qu: When True, doubles the photon variance for the
            ``|P|`` = ``sqrt(Q² + U²)`` combination.
        dark_rate_hz_per_port_per_arm: Detector dark count rate.
        sky_rate_hz_per_dark_port_per_arm: Sky/thermal background rate.

    Returns:
        PDI SNR for detecting the planet's polarized component. ``0.0``
        when the variance is non-positive.
    """
    if integration_time_s <= 0.0:
        return 0.0
    if pol_fraction <= 0.0:
        return 0.0

    s_total = float(planet_rate_total_hz) * integration_time_s
    b_total = float(stellar_leakage_rate_total_hz) * integration_time_s
    d_total = (
        float(dark_rate_hz_per_port_per_arm)
        * int(n_dark_ports_per_arm)
        * 2
        * integration_time_s
    )
    sky_total = (
        float(sky_rate_hz_per_dark_port_per_arm)
        * int(n_dark_ports_per_arm)
        * 2
        * integration_time_s
    )

    signal = float(pol_fraction) * s_total
    photon_dark_variance = s_total + b_total + d_total + sky_total
    if recover_both_qu:
        photon_dark_variance *= 2.0

    pbs_leakage_power = (
        0.0
        if math.isinf(pbs_extinction_db)
        else 10.0 ** (-float(pbs_extinction_db) / 10.0)
    )
    eps_pdi = math.sqrt(
        (2.0 * float(pbs_axis_offset_rad)) ** 2
        + (0.5 * float(hwp_retardance_error_rad)) ** 2
        + (2.0 * float(hwp_rotation_jitter_rad)) ** 2
        + pbs_leakage_power
    )
    if cmrr_db > 0.0:
        eps_pdi *= 10.0 ** (-float(cmrr_db) / 20.0)

    systematic_variance = 0.0
    if eps_pdi > 0.0 and stellar_leakage_rate_total_hz > 0.0:
        if modulation_period_s > 0.0:
            # N = T_int / T_cycle independent cycles. Per-cycle leakage
            # residual: eps_pdi · b_rate · T_cycle. Total variance:
            # N · (per-cycle)² = eps_pdi² · b_rate² · T_int · T_cycle.
            systematic_variance = (
                eps_pdi ** 2
                * float(stellar_leakage_rate_total_hz) ** 2
                * integration_time_s
                * float(modulation_period_s)
            )
        else:
            # Single-cycle hard ceiling; use the uncapped leakage rate
            # (the null fluctuates at the true rate even when the
            # detector saturates).
            systematic_variance = (eps_pdi * b_total) ** 2

    variance = photon_dark_variance + systematic_variance
    if variance <= 0.0:
        return 0.0
    return signal / math.sqrt(variance)
