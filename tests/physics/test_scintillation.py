"""Tests for the atmospheric scintillation and amplitude-servo model.

The implementation in ``nullsim/physics/scintillation.py`` is a direct
port of KILO ``kilo/atmosphere.py`` (L95-481). These tests pin the
KILO-equivalent analytic results so any drift from KILO physics shows
up here.
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
    """At λ=1.65 µm, d_sub=5 m, zenith, the result equals the H-band anchor."""
    sigma = scintillation_index(
        wavelength_m=1.65e-6,
        sub_diameter_m=5.0,
        scintillation_index_h5m_zenith=0.041,
        scintillation_height_km=10.0,
        airmass=1.0,
    )
    assert sigma == pytest.approx(0.041, rel=1.0e-12)


def test_scintillation_index_wavelength_scaling_minus_7_over_12() -> None:
    """Halving λ (1.65→0.825 µm) scales σ_I by 2^(7/12)."""
    sigma_h = scintillation_index(
        wavelength_m=1.65e-6,
        sub_diameter_m=5.0,
        scintillation_index_h5m_zenith=0.041,
    )
    sigma_short = scintillation_index(
        wavelength_m=0.825e-6,
        sub_diameter_m=5.0,
        scintillation_index_h5m_zenith=0.041,
    )
    assert sigma_short / sigma_h == pytest.approx(
        2.0 ** (7.0 / 12.0), rel=1.0e-12
    )


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
            scintillation_index_h5m_zenith=0.041,
        )
    assert sigma == pytest.approx(SCINTILLATION_SATURATION_INDEX, rel=1.0e-12)


def test_scintillation_timescale_uses_jet_stream_wind() -> None:
    """τ = r_F / v_high; at H-band, h=10 km, v=35 m/s, ~3.7 ms."""
    tau = scintillation_timescale_for_servo_design(
        wavelength_m=1.65e-6,
        scintillation_height_km=10.0,
        high_altitude_wind_ms=35.0,
    )
    r_fresnel = math.sqrt(1.65e-6 * 10.0e3)
    expected_tau = r_fresnel / 35.0
    assert tau == pytest.approx(expected_tau, rel=1.0e-12)
    # Smoke check: expect ~3.7 ms.
    assert 3.0e-3 < tau < 4.5e-3


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
        wavelength_m=1.65e-6,
        scintillation_height_km=10.0,
        high_altitude_wind_ms=35.0,
    )
    expected_temporal = (
        res.sigma_I_open * (tau_loop / tau_scint) ** (5.0 / 6.0)
        / math.sqrt(20.0)
    )
    assert res.sigma_I_temporal == pytest.approx(expected_temporal, rel=1.0e-12)
    # Quadrature with the residual photon-noise floor (~3e-5 here vs
    # temporal ~7e-3) shifts the combined σ from the pure temporal limit
    # by ~2e-5. Tolerance loosened from 1e-6 → 1e-4.
    assert res.sigma_I_residual == pytest.approx(expected_temporal, rel=1.0e-4)


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
