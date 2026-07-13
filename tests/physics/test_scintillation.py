"""Tests for the atmospheric scintillation and amplitude-servo model.

``nullsim/physics/scintillation.py`` was re-anchored to the measured
aperture-averaged scintillation statistics in the 2026-07-13 audit
(Kornilov et al. 2012 anchor, achromatic index, aperture-crossing
timescale). These tests pin the corrected analytic results.
"""

from __future__ import annotations

import math

import pytest

from nullsim.physics.scintillation import (
    SCINTILLATION_SATURATION_INDEX,
    amplitude_servo_residual_sigma_i,
    scintillation_index,
    scintillation_timescale_for_servo_design,
)


def test_scintillation_index_matches_anchor_at_default_params() -> None:
    """At d_sub=5 m, zenith, the result equals the anchor (any wavelength)."""
    sigma = scintillation_index(
        wavelength_m=1.65e-6,
        sub_diameter_m=5.0,
        scintillation_index_h5m_zenith=0.0013,
        scintillation_height_km=10.0,
        airmass=1.0,
    )
    assert sigma == pytest.approx(0.0013, rel=1.0e-12)


def test_scintillation_index_default_anchor_is_kornilov_q1() -> None:
    """Default anchor = Kornilov+2012 Q1 Maunakea: sqrt(S_2^2 * 5^(-7/3))."""
    sigma = scintillation_index(
        wavelength_m=1.65e-6,
        sub_diameter_m=5.0,
    )
    kornilov_q1_s2_squared_m73 = 0.77e-4
    measured = math.sqrt(kornilov_q1_s2_squared_m73 * 5.0 ** (-7.0 / 3.0))
    assert sigma == pytest.approx(measured, rel=0.05)


def test_scintillation_index_is_achromatic_in_aperture_averaged_regime() -> None:
    """The large-aperture index is wavelength-independent (Kenyon 2006)."""
    sigma_h = scintillation_index(
        wavelength_m=1.65e-6,
        sub_diameter_m=5.0,
        scintillation_index_h5m_zenith=0.0013,
    )
    sigma_short = scintillation_index(
        wavelength_m=0.825e-6,
        sub_diameter_m=5.0,
        scintillation_index_h5m_zenith=0.0013,
    )
    assert sigma_short == pytest.approx(sigma_h, rel=1.0e-12)


def test_scintillation_index_subaperture_scaling_minus_7_over_6() -> None:
    """Doubling d_sub scales σ_I by 2^(-7/6) (large-aperture regime)."""
    sigma_small = scintillation_index(
        wavelength_m=1.65e-6,
        sub_diameter_m=5.0,
    )
    sigma_big = scintillation_index(
        wavelength_m=1.65e-6,
        sub_diameter_m=10.0,
    )
    assert sigma_big / sigma_small == pytest.approx(
        2.0 ** (-7.0 / 6.0), rel=1.0e-12
    )


def test_scintillation_index_airmass_scaling_pow_three_halves() -> None:
    """σ_I ∝ airmass^(3/2)."""
    sigma_zenith = scintillation_index(
        wavelength_m=1.65e-6,
        sub_diameter_m=5.0,
        airmass=1.0,
    )
    sigma_secz2 = scintillation_index(
        wavelength_m=1.65e-6,
        sub_diameter_m=5.0,
        airmass=2.0,
    )
    assert sigma_secz2 / sigma_zenith == pytest.approx(
        2.0 ** 1.5, rel=1.0e-12
    )


def test_scintillation_index_below_fresnel_radius_saturates() -> None:
    """Sub-aperture below the Fresnel radius clips to the saturation cap."""
    with pytest.warns(UserWarning, match="Fresnel radius"):
        sigma = scintillation_index(
            wavelength_m=1.65e-6,
            sub_diameter_m=0.01,  # well below r_F ≈ 0.13 m at H, h=10 km
            scintillation_index_h5m_zenith=10.0,  # force sigma above the cap
        )
    assert sigma == pytest.approx(SCINTILLATION_SATURATION_INDEX, rel=1.0e-12)


def test_scintillation_timescale_is_aperture_crossing_time() -> None:
    """τ = 0.5 d_sub / v_high; at 3.14 m, 35 m/s, ~45 ms (not r_F/v ~3.7 ms)."""
    tau = scintillation_timescale_for_servo_design(
        sub_diameter_m=3.14,
        high_altitude_wind_ms=35.0,
    )
    assert tau == pytest.approx(0.5 * 3.14 / 35.0, rel=1.0e-12)
    # Smoke check: tens of ms for a meter-class sub-aperture.
    assert 0.02 < tau < 0.12


def test_amplitude_servo_residual_photon_limited_for_faint_target() -> None:
    """At very low photon rate, σ_I_residual → 1/√N (photon floor)."""
    # Pick a wide servo bandwidth so τ_loop/τ_scint is tiny and the
    # temporal residual collapses; then the result is photon-limited.
    bandwidth = 1.0e4  # 10 kHz, τ_loop = 1e-4 s ≈ 1/37 of τ_scint
    rate = 1.0e3      # ph/s/sub → 0.1 photons / 1e-4 s sample
    res = amplitude_servo_residual_sigma_i(
        wavelength_m=1.65e-6,
        sub_diameter_m=1.5,
        servo_bandwidth_hz=bandwidth,
        photon_rate_per_sub_hz=rate,
        scintillation_index_h5m_zenith=0.041,
    )
    expected_photon = 1.0 / math.sqrt(rate / bandwidth)
    # Photon floor dominates here; residual ≈ σ_photon (capped at σ_open).
    assert res.sigma_I_photon == pytest.approx(expected_photon, rel=1.0e-12)
    assert res.sigma_I_residual == pytest.approx(
        min(expected_photon, res.sigma_I_open), rel=1.0e-12
    )


def test_amplitude_servo_residual_temporal_dominates_for_bright_target() -> None:
    """At high photon rate, σ_I_residual → (τ_loop/τ_scint)^(5/6) × σ_open / √g."""
    bandwidth = 2.0e3
    rate = 1.0e12  # photon-noise free
    res = amplitude_servo_residual_sigma_i(
        wavelength_m=1.65e-6,
        sub_diameter_m=1.5,
        servo_bandwidth_hz=bandwidth,
        photon_rate_per_sub_hz=rate,
        scintillation_index_h5m_zenith=0.041,
        predictive_gain_variance=20.0,
    )
    tau_loop = 1.0 / bandwidth
    tau_scint = scintillation_timescale_for_servo_design(
        sub_diameter_m=1.5,
        high_altitude_wind_ms=35.0,
    )
    expected_temporal = (
        res.sigma_I_open * (tau_loop / tau_scint) ** (5.0 / 6.0)
        / math.sqrt(20.0)
    )
    assert res.sigma_I_temporal == pytest.approx(expected_temporal, rel=1.0e-12)
    # Quadrature with the residual photon-noise floor shifts the combined
    # sigma from the pure temporal limit by ~(photon/temporal)^2/2 ~ 4e-4.
    assert res.sigma_I_residual == pytest.approx(expected_temporal, rel=2.0e-3)


def test_amplitude_servo_residual_saturates_at_open_loop() -> None:
    """Residual can never exceed the open-loop σ_I."""
    res = amplitude_servo_residual_sigma_i(
        wavelength_m=1.65e-6,
        sub_diameter_m=1.5,
        servo_bandwidth_hz=1.0,    # very slow servo, no correction
        photon_rate_per_sub_hz=1.0e-3,
        scintillation_index_h5m_zenith=0.041,
    )
    assert res.sigma_I_residual <= res.sigma_I_open
    assert res.sigma_I_residual == pytest.approx(res.sigma_I_open, rel=1.0e-12)


def test_amplitude_servo_residual_keck_keck_only_j_band_order_of_magnitude() -> None:
    """KILO Keck-only J-band smoke: σ_I_residual lands in ~0.01-0.10."""
    res = amplitude_servo_residual_sigma_i(
        wavelength_m=1.25e-6,
        sub_diameter_m=1.93,
        servo_bandwidth_hz=2.0e3,
        photon_rate_per_sub_hz=1.0e6,
        scintillation_index_h5m_zenith=0.041,
        predictive_gain_variance=20.0,
    )
    # KILO's value at this regime lands in the 0.04-0.08 range; require a
    # 2× tolerance for cross-version drift.
    assert 0.01 < res.sigma_I_residual < 0.20


def test_amplitude_servo_residual_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError, match="servo_bandwidth_hz"):
        amplitude_servo_residual_sigma_i(
            wavelength_m=1.65e-6,
            sub_diameter_m=1.5,
            servo_bandwidth_hz=0.0,
            photon_rate_per_sub_hz=1.0e6,
        )
    with pytest.raises(ValueError, match="photon_rate_per_sub_hz"):
        amplitude_servo_residual_sigma_i(
            wavelength_m=1.65e-6,
            sub_diameter_m=1.5,
            servo_bandwidth_hz=2.0e3,
            photon_rate_per_sub_hz=0.0,
        )
    with pytest.raises(ValueError, match="predictive_gain_variance"):
        amplitude_servo_residual_sigma_i(
            wavelength_m=1.65e-6,
            sub_diameter_m=1.5,
            servo_bandwidth_hz=2.0e3,
            photon_rate_per_sub_hz=1.0e6,
            predictive_gain_variance=0.5,
        )
