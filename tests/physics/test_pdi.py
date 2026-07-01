"""Pin the PDI SNR primitive against KILO ``signal_to_noise_pdi``."""

from __future__ import annotations

import math

import pytest

from nullsim.physics.pdi import signal_to_noise_pdi


def test_returns_zero_when_integration_time_is_zero() -> None:
    snr = signal_to_noise_pdi(
        planet_rate_total_hz=1.0,
        stellar_leakage_rate_total_hz=0.1,
        integration_time_s=0.0,
        pol_fraction=0.1,
    )
    assert snr == 0.0


def test_returns_zero_when_pol_fraction_is_zero() -> None:
    snr = signal_to_noise_pdi(
        planet_rate_total_hz=1.0,
        stellar_leakage_rate_total_hz=0.1,
        integration_time_s=3600.0,
        pol_fraction=0.0,
    )
    assert snr == 0.0


def test_perfect_optics_no_systematic_floor_matches_pol_amplified_poisson() -> None:
    """With ideal optics + CMRR off the formula reduces to Q+U Poisson SNR."""

    planet_rate = 5.0
    leak_rate = 50.0
    t = 1000.0
    pol = 0.5

    snr = signal_to_noise_pdi(
        planet_rate_total_hz=planet_rate,
        stellar_leakage_rate_total_hz=leak_rate,
        integration_time_s=t,
        pol_fraction=pol,
        pbs_extinction_db=float("inf"),
        cmrr_db=0.0,
        recover_both_qu=True,
    )
    signal = pol * planet_rate * t
    expected_variance = 2.0 * (planet_rate + leak_rate) * t
    expected_snr = signal / math.sqrt(expected_variance)
    assert snr == pytest.approx(expected_snr, rel=1.0e-12)


def test_recover_only_q_halves_variance() -> None:
    """recover_both_qu=False drops the 2× Q+U penalty."""

    kw = dict(
        planet_rate_total_hz=5.0,
        stellar_leakage_rate_total_hz=50.0,
        integration_time_s=1000.0,
        pol_fraction=0.5,
        pbs_extinction_db=float("inf"),
        cmrr_db=0.0,
    )
    snr_qu = signal_to_noise_pdi(recover_both_qu=True, **kw)
    snr_q = signal_to_noise_pdi(recover_both_qu=False, **kw)
    assert snr_q == pytest.approx(snr_qu * math.sqrt(2.0), rel=1.0e-12)


def test_pbs_extinction_db_lifts_systematic_floor() -> None:
    """Finite PBS extinction raises the systematic floor → lower SNR."""

    common = dict(
        planet_rate_total_hz=5.0,
        stellar_leakage_rate_total_hz=1.0e6,
        integration_time_s=1.0e6,  # long enough to be systematic-limited
        pol_fraction=0.5,
        modulation_period_s=0.0,
        cmrr_db=0.0,
    )
    snr_perfect = signal_to_noise_pdi(pbs_extinction_db=float("inf"), **common)
    snr_30db = signal_to_noise_pdi(pbs_extinction_db=30.0, **common)
    assert snr_30db < snr_perfect


def test_cmrr_db_attenuates_imperfection_amplitude() -> None:
    """20 dB CMRR drops eps_pdi by 10× → systematic variance by 100×."""

    common = dict(
        planet_rate_total_hz=5.0,
        stellar_leakage_rate_total_hz=1.0e6,
        integration_time_s=1.0e6,
        pol_fraction=0.5,
        pbs_extinction_db=30.0,
        modulation_period_s=0.0,
    )
    snr_no_cmrr = signal_to_noise_pdi(cmrr_db=0.0, **common)
    snr_20db = signal_to_noise_pdi(cmrr_db=20.0, **common)
    assert snr_20db > snr_no_cmrr


def test_hwp_retardance_uses_half_angle_quadrature() -> None:
    planet_rate = 5.0
    leak_rate = 1.0e6
    integration_time_s = 1.0e6
    hwp_error = 0.02

    snr = signal_to_noise_pdi(
        planet_rate_total_hz=planet_rate,
        stellar_leakage_rate_total_hz=leak_rate,
        integration_time_s=integration_time_s,
        pol_fraction=0.5,
        hwp_retardance_error_rad=hwp_error,
        pbs_axis_offset_rad=0.0,
        hwp_rotation_jitter_rad=0.0,
        pbs_extinction_db=float("inf"),
        modulation_period_s=0.0,
        cmrr_db=0.0,
        recover_both_qu=False,
    )

    signal = 0.5 * planet_rate * integration_time_s
    photon_variance = (planet_rate + leak_rate) * integration_time_s
    systematic_variance = ((0.5 * hwp_error) * leak_rate * integration_time_s) ** 2
    expected = signal / math.sqrt(photon_variance + systematic_variance)
    assert snr == pytest.approx(expected, rel=1.0e-12)


def test_modulation_period_lowers_systematic_floor() -> None:
    """Modulation breaks the leakage into N cycles, suppressing the floor."""

    common = dict(
        planet_rate_total_hz=5.0,
        stellar_leakage_rate_total_hz=1.0e6,
        integration_time_s=1.0e6,
        pol_fraction=0.5,
        pbs_extinction_db=30.0,
        cmrr_db=0.0,
    )
    snr_no_mod = signal_to_noise_pdi(modulation_period_s=0.0, **common)
    snr_fast_mod = signal_to_noise_pdi(modulation_period_s=1.0e-3, **common)
    assert snr_fast_mod > snr_no_mod
