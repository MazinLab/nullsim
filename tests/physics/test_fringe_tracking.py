"""Tests for nullsim.physics.fringe_tracking primitives.

Mirrors KILO's ``optimal_piston_servo_bandwidth`` derivation (see
``kilo/fringe_tracking.py`` L286-333).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nullsim.physics.fringe_tracking import (
    fusion_effective_broadband_rms_m,
    optimal_piston_servo_bandwidth,
    photon_kalman_residual_budget,
    photon_limited_setpoint_trim,
    piston_psd_knee_hz,
    piston_structure_function_m2,
    servo_residual_variance_flat_knee,
    temporal_kalman_piston_covariance,
    vibration_residual_rms_nm,
    von_karman_differential_piston_rms_m,
)


def test_optimal_bandwidth_disabled_when_starved() -> None:
    """Zero photon rate or zero visibility disables the servo."""

    assert optimal_piston_servo_bandwidth(
        sigma2_temporal_rad2=0.25,
        f_char_hz=0.7,
        rate_per_sub_hz=0.0,
        visibility=0.9,
    ) == 0.0
    assert optimal_piston_servo_bandwidth(
        sigma2_temporal_rad2=0.25,
        f_char_hz=0.7,
        rate_per_sub_hz=1e6,
        visibility=0.0,
    ) == 0.0


def test_optimal_bandwidth_disabled_for_invalid_pred() -> None:
    assert optimal_piston_servo_bandwidth(
        sigma2_temporal_rad2=0.25,
        f_char_hz=0.7,
        rate_per_sub_hz=1e6,
        visibility=0.9,
        predictive_lag_reduction=0.0,
    ) == 0.0


def test_optimal_bandwidth_matches_closed_form() -> None:
    """``f_opt = [(5/(3 pi)) * sigma2 * f_char^{5/3} * R * V^2 * pred]^{3/8}``."""

    sigma2 = 0.25
    f_char = 0.7
    rate = 1e7
    visibility = 0.9
    pred = 0.5

    coeff = (5.0 / (3.0 * math.pi)) * sigma2 * f_char ** (5.0 / 3.0)
    inner = coeff * rate * visibility ** 2 * pred
    expected = inner ** (3.0 / 8.0)

    f_opt = optimal_piston_servo_bandwidth(
        sigma2_temporal_rad2=sigma2,
        f_char_hz=f_char,
        rate_per_sub_hz=rate,
        visibility=visibility,
        predictive_lag_reduction=pred,
    )
    assert f_opt == pytest.approx(expected, rel=1e-12)


def test_optimal_bandwidth_scales_with_predictive_gain() -> None:
    """KILO docstring: predictor shifts f_opt down by pred^{3/8}."""

    reference = optimal_piston_servo_bandwidth(
        sigma2_temporal_rad2=0.25,
        f_char_hz=0.7,
        rate_per_sub_hz=1e7,
        visibility=0.9,
        predictive_lag_reduction=1.0,
    )
    predicted = optimal_piston_servo_bandwidth(
        sigma2_temporal_rad2=0.25,
        f_char_hz=0.7,
        rate_per_sub_hz=1e7,
        visibility=0.9,
        predictive_lag_reduction=0.0625,
    )
    assert predicted / reference == pytest.approx(0.0625 ** (3.0 / 8.0), rel=1e-12)


def test_optimal_bandwidth_in_kilo_keck_regime() -> None:
    """J=4 Keck-ish inputs put f_opt in the ~30-3000 Hz range."""

    f_opt = optimal_piston_servo_bandwidth(
        sigma2_temporal_rad2=0.25,
        f_char_hz=0.694,
        rate_per_sub_hz=1e7,
        visibility=0.9,
    )
    assert 30.0 < f_opt < 3000.0


def test_von_karman_differential_piston_saturates_at_outer_scale() -> None:
    """B >> L0 differential piston saturates with B replaced by L0."""

    sigma_m = von_karman_differential_piston_rms_m(
        r0_500nm_m=0.15,
        outer_scale_m=25.0,
    )

    assert sigma_m == pytest.approx(14.81e-6, rel=1.0e-3)


def test_piston_psd_knee_is_max_of_outer_scale_and_aperture_cutoffs() -> None:
    """Outer-scale and aperture averaging set the flat-PSD knee."""

    assert piston_psd_knee_hz(
        wind_speed_m_s=16.643,
        outer_scale_m=25.0,
        aperture_diameter_m=10.0,
    ) == pytest.approx(16.643 / 25.0, rel=1.0e-6)


def test_servo_residual_flat_knee_17_3_coefficient() -> None:
    """Flat-to-knee / f^(-17/3) PSD integrates to the 7/12 coefficient."""

    var = servo_residual_variance_flat_knee(
        sigma2=1.0,
        f_knee_hz=0.6656,
        f_3db_hz=50.0,
        predictive_gain_variance=1.0,
    )

    assert var == pytest.approx((7.0 / 12.0) * (0.6656 / 50.0) ** 2, rel=1.0e-6)


def test_inter_telescope_residual_end_to_end_number() -> None:
    """14.81 um open-loop piston at 0.6656 Hz leaves 150.6 nm after 50 Hz servo."""

    rms_nm = 1.0e9 * math.sqrt(
        servo_residual_variance_flat_knee(
            sigma2=(14.81e-6) ** 2,
            f_knee_hz=0.6656,
            f_3db_hz=50.0,
            predictive_gain_variance=1.0,
        )
    )

    assert rms_nm == pytest.approx(150.6, rel=2.0e-3)


def test_vibration_residual_lines_only() -> None:
    """Residual line power applies min(1, (f/f_c)^2) per line."""

    rms_nm = vibration_residual_rms_nm(
        line_frequencies_hz=[12, 29, 58, 88, 150, 200],
        line_rms_nm=[30, 40, 25, 20, 10, 10],
        broadband_rms_nm=0.0,
        broadband_max_hz=500.0,
        f_3db_hz=100.0,
    )

    assert rms_nm == pytest.approx(29.454, rel=1.0e-3)


def test_vibration_residual_with_broadband_floor() -> None:
    """Flat broadband vibration floor uses the analytic 0..f_max integral."""

    rms_nm = vibration_residual_rms_nm(
        line_frequencies_hz=[12, 29, 58, 88, 150, 200],
        line_rms_nm=[30, 40, 25, 20, 10, 10],
        broadband_rms_nm=30.0,
        broadband_max_hz=500.0,
        f_3db_hz=100.0,
    )

    assert rms_nm == pytest.approx(40.59, rel=1.0e-3)


def test_vibration_residual_broadband_floor_above_modeled_cutoff() -> None:
    """When f_3db exceeds f_max, the full flat floor scales as f_max^2 / f_c^2."""

    rms_nm = vibration_residual_rms_nm(
        line_frequencies_hz=[],
        line_rms_nm=[],
        broadband_rms_nm=30.0,
        broadband_max_hz=500.0,
        f_3db_hz=1000.0,
    )

    expected_fraction = 500.0**2 / (3.0 * 1000.0**2)
    assert rms_nm == pytest.approx(30.0 * math.sqrt(expected_fraction), rel=1.0e-12)


def test_fusion_effective_broadband_floor() -> None:
    """Accelerometer fusion suppresses only the coherent broadband floor."""

    assert fusion_effective_broadband_rms_m(
        broadband_rms_m=30.0e-9,
        coherent_fraction=0.7,
        suppression=2.5,
    ) == pytest.approx(19.25e-9, rel=1.0e-3)


def test_structure_function_small_tau_matches_analytic() -> None:
    """Flat-to-knee / f^(-17/3) PSD has the pinned small-tau limit."""

    value = piston_structure_function_m2(
        tau_s=1.0e-4,
        atm_sigma2_m2=1.0,
        atm_f_knee_hz=0.6656,
        broadband_sigma2_m2=0.0,
        broadband_max_hz=500.0,
    )

    assert value == pytest.approx(1.020e-7, rel=2.0e-2)


def test_structure_function_saturates_at_twice_variance() -> None:
    """For long lags, the structure function tends to twice the variance."""

    value = piston_structure_function_m2(
        tau_s=1.0e3,
        atm_sigma2_m2=1.0,
        atm_f_knee_hz=0.6656,
        broadband_sigma2_m2=0.0,
        broadband_max_hz=500.0,
    )

    assert value == pytest.approx(2.0, rel=5.0e-2)


def test_photon_kalman_zero_noise_limit_is_latency_floor() -> None:
    """With negligible measurement noise, the residual is the lag floor."""

    result = photon_kalman_residual_budget(
        atm_sigma2_m2=(14.81e-6) ** 2,
        atm_f_knee_hz=0.6656,
        broadband_sigma2_m2=0.0,
        broadband_max_hz=500.0,
        tracked_line_count=0,
        line_tracking_floor_m=0.0,
        crlb_m_sqrt_s=1.0e-18,
        latency_s=5.0e-5,
    )
    expected = (
        piston_structure_function_m2(
            tau_s=5.0e-5,
            atm_sigma2_m2=(14.81e-6) ** 2,
            atm_f_knee_hz=0.6656,
            broadband_sigma2_m2=0.0,
            broadband_max_hz=500.0,
        )
        / 2.0
    )

    assert result.total_m2 == pytest.approx(expected, rel=1.0e-2)


def test_photon_kalman_j4_defaults_land_in_band() -> None:
    """J=4-like defaults land near the analytic Phase B residual estimate."""

    result = photon_kalman_residual_budget(
        atm_sigma2_m2=(14.81e-6) ** 2,
        atm_f_knee_hz=0.6656,
        broadband_sigma2_m2=(19.25e-9) ** 2,
        broadband_max_hz=500.0,
        tracked_line_count=6,
        line_tracking_floor_m=2.0e-9,
        crlb_m_sqrt_s=1.29e-10,
        latency_s=5.0e-5,
    )

    assert 10.0e-9 < math.sqrt(result.total_m2) < 16.0e-9
    assert 5.0e-5 < result.window_s < 5.0e-4
    assert result.f_3db_equiv_hz == pytest.approx(
        1.0 / (2.0 * (result.window_s + 5.0e-5)),
        rel=1.0e-9,
    )


def test_temporal_kalman_covariance_uses_slow_long_baseline_modes() -> None:
    """Long-baseline modes decorrelate more slowly than intra-pupil modes."""

    measurement_nm2 = np.diag([0.0, 1.0, 1.0])
    measurement = measurement_nm2 * 1.0e-18
    # Mode 1 compares the close pair. Mode 2 compares the pair to the far
    # aperture, so the effective Conan/Di Folco baseline should be larger and
    # the AR(1) correlation coefficient should be closer to one.
    positions = np.array(
        [
            [0.0, 0.0],
            [4.0, 0.0],
            [86.0, 0.0],
        ],
        dtype=np.float64,
    )

    result = temporal_kalman_piston_covariance(
        measurement_covariance_m2=measurement,
        positions_m=positions,
        prior_rms_m=10.0e-9,
        sample_time_s=0.04,
        wind_speed_m_s=10.0,
        subaperture_diameter_m=4.0,
        residual_kind="posterior",
    )

    assert result.selected_covariance_m2.shape == (3, 3)
    assert result.mode_baseline_m[2] > result.mode_baseline_m[1]
    assert result.mode_ar1_rho[2] > result.mode_ar1_rho[1]
    posterior_std_nm = np.sqrt(result.mode_posterior_variance_m2) * 1.0e9
    assert posterior_std_nm[1] < 1.0
    assert posterior_std_nm[2] < 1.0


def test_temporal_kalman_one_step_prediction_is_not_overconfident() -> None:
    """One-step prediction includes process evolution after the measurement."""

    measurement = np.diag([0.0, 1.0, 1.0]) * 1.0e-18
    positions = np.array([[0.0, 0.0], [4.0, 0.0], [86.0, 0.0]], dtype=np.float64)

    posterior = temporal_kalman_piston_covariance(
        measurement_covariance_m2=measurement,
        positions_m=positions,
        prior_rms_m=10.0e-9,
        sample_time_s=0.04,
        wind_speed_m_s=10.0,
        subaperture_diameter_m=4.0,
        residual_kind="posterior",
    )
    prediction = temporal_kalman_piston_covariance(
        measurement_covariance_m2=measurement,
        positions_m=positions,
        prior_rms_m=10.0e-9,
        sample_time_s=0.04,
        wind_speed_m_s=10.0,
        subaperture_diameter_m=4.0,
        residual_kind="one_step_prediction",
    )

    assert np.trace(prediction.selected_covariance_m2) > np.trace(
        posterior.selected_covariance_m2
    )


def test_temporal_kalman_unobserved_mode_carries_full_process_variance() -> None:
    """An unobserved (zero-eigenvalue) mode with a real baseline must carry the
    full one-step process variance in its prediction, not zero.

    Regression for the guard that skipped ``evals[index] <= 0`` modes: a mode
    the bright ports cannot sense is uncorrected, so its predicted residual is
    the full process variance ``prior_variance * (1 - rho^2)``. Only the common
    piston (zero baseline) is legitimately excluded.
    """

    # Mode 0 (the zero-eigenvalue eigenvector e0) is unsensed but is a real
    # differential mode with a nonzero aperture baseline.
    measurement = np.diag([0.0, 1.0, 1.0]) * 1.0e-18
    positions = np.array([[0.0, 0.0], [4.0, 0.0], [86.0, 0.0]], dtype=np.float64)
    prior_rms_m = 10.0e-9

    result = temporal_kalman_piston_covariance(
        measurement_covariance_m2=measurement,
        positions_m=positions,
        prior_rms_m=prior_rms_m,
        sample_time_s=0.04,
        wind_speed_m_s=10.0,
        subaperture_diameter_m=4.0,
        residual_kind="one_step_prediction",
    )

    assert result.mode_measurement_variance_m2[0] == 0.0
    assert result.mode_baseline_m[0] > 0.0
    rho0 = result.mode_ar1_rho[0]
    expected_q = (prior_rms_m**2) * (1.0 - rho0**2)
    assert result.mode_prediction_variance_m2[0] == pytest.approx(expected_q, rel=1.0e-9)
    assert result.mode_prediction_variance_m2[0] > 0.0
    # The measurement update is vacuous for an unobserved mode.
    assert result.mode_posterior_variance_m2[0] == 0.0


def test_setpoint_trim_optimum_matches_j4_hand_derivation():
    """Pin against the 2026-06-10 hand derivation at the kilo_keck_6 J=4 point.

    S=5.15e8 cps, N_f=1.83e-5, f_trim=1 Hz, n=12, lambda=1.25 um:
    dwell = 1/24 s, k = sqrt(N_f/(S*dwell))/(sqrt2*G) = 2.612e-6,
    A_opt = (2n k^2)^(1/4) = 3.577e-3 rad (0.712 nm amplitude, 0.503 nm RMS),
    delta = k/A = 0.1453 nm per mode, total leakage G A^2 = 3.20e-6 with
    equal dither/bias partition.
    """
    out = photon_limited_setpoint_trim(
        stellar_rate_hz=5.15e8,
        dark_floor_null=1.83e-5,
        trim_loop_bandwidth_hz=1.0,
        n_trim_modes=12,
        wavelength_m=1.25e-6,
    )
    assert out.dither_amplitude_rad == pytest.approx(3.577e-3, rel=2e-3)
    assert out.dither_rms_nm == pytest.approx(0.503, rel=2e-3)
    assert out.setpoint_residual_nm_per_mode == pytest.approx(0.1453, rel=2e-3)
    assert out.total_leakage == pytest.approx(3.20e-6, rel=2e-3)
    assert out.dither_leakage == pytest.approx(out.bias_leakage, rel=1e-9)


def test_setpoint_trim_scales_with_flux_and_trim_bandwidth():
    """L_min = G sqrt(2n) k with k ~ sqrt(f_trim/S): total ~ 1/sqrt(S), ~ sqrt(f)."""
    kwargs = dict(
        stellar_rate_hz=5.15e8,
        dark_floor_null=1.83e-5,
        trim_loop_bandwidth_hz=1.0,
        n_trim_modes=12,
        wavelength_m=1.25e-6,
    )
    base = photon_limited_setpoint_trim(**kwargs)
    faint = photon_limited_setpoint_trim(**{**kwargs, "stellar_rate_hz": 5.15e8 / 40.0})
    assert faint.total_leakage == pytest.approx(
        base.total_leakage * math.sqrt(40.0), rel=1e-9
    )
    slow = photon_limited_setpoint_trim(
        **{**kwargs, "trim_loop_bandwidth_hz": 0.1}
    )
    assert slow.total_leakage == pytest.approx(
        base.total_leakage * math.sqrt(0.1), rel=1e-9
    )


def test_setpoint_trim_zero_floor_returns_zero_budget():
    out = photon_limited_setpoint_trim(
        stellar_rate_hz=5.15e8,
        dark_floor_null=0.0,
        trim_loop_bandwidth_hz=1.0,
        n_trim_modes=12,
        wavelength_m=1.25e-6,
    )
    assert out.total_leakage == 0.0
    assert out.dither_amplitude_rad == 0.0
