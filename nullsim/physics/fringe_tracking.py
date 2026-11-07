"""Piston-servo physics primitives.

Pulled out of the stage layer (``nullsim/stages/fringe_tracking/``) so the
formulas can be tested independently and reused by other stages (e.g.
``sensitivity.detection_curve`` for what-if calibration sweeps). Mirrors
the structure of ``kilo/fringe_tracking.py`` so the name ``optimal_piston
_servo_bandwidth`` lines up directly with KILO.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "PISTON_PSD_APERTURE_CUTOFF_COEFFICIENT",
    "PhotonKalmanResidual",
    "TemporalKalmanPistonCovariance",
    "VIBRATION_BROADBAND_CUTOFF_HZ",
    "VON_KARMAN_PISTON_COEFFICIENT",
    "fusion_effective_broadband_rms_m",
    "optimal_piston_servo_bandwidth",
    "photon_kalman_residual_budget",
    "piston_psd_knee_hz",
    "piston_structure_function_m2",
    "servo_residual_variance_flat_knee",
    "temporal_kalman_piston_covariance",
    "vibration_residual_rms_nm",
    "von_karman_differential_piston_rms_m",
]

VON_KARMAN_PISTON_COEFFICIENT = 12.0 * math.pi / 5.0 * 0.023
"""Saturated point-aperture differential piston, 0.1734 (L0/r0)^(5/3) rad^2."""

_VON_KARMAN_PSD_COEFFICIENT = 0.023
"""Von Karman phase PSD prefactor, W(f) = 0.023 r0^(-5/3) (f^2 + L0^-2)^(-11/6)."""

PISTON_PSD_APERTURE_CUTOFF_COEFFICIENT = 0.3
"""High-frequency aperture-averaging cutoff 0.3 v/D."""

VIBRATION_BROADBAND_CUTOFF_HZ = 500.0
"""Default upper frequency for the broadband vibration floor model."""

_STRUCTURE_FUNCTION_GRID_SIZE = 4096
_STRUCTURE_FUNCTION_SMALL_ARGUMENT = 1.0e-2
_STRUCTURE_FUNCTION_SATURATION_ARGUMENT = 100.0


@dataclass(frozen=True)
class TemporalKalmanPistonCovariance:
    """Mode-dependent piston covariance after temporal Kalman filtering.

    ``measurement_covariance_m2`` is the instantaneous bright-fringe
    measurement covariance. ``posterior_covariance_m2`` is the residual after
    updating with that frame's measurement. ``prediction_covariance_m2`` is the
    one-sample-ahead residual after propagating the posterior through the
    atmospheric prior.
    """

    measurement_covariance_m2: NDArray[np.float64]
    posterior_covariance_m2: NDArray[np.float64]
    prediction_covariance_m2: NDArray[np.float64]
    selected_covariance_m2: NDArray[np.float64]
    mode_measurement_variance_m2: NDArray[np.float64]
    mode_prior_variance_m2: NDArray[np.float64]
    mode_process_variance_m2: NDArray[np.float64]
    mode_posterior_variance_m2: NDArray[np.float64]
    mode_prediction_variance_m2: NDArray[np.float64]
    mode_baseline_m: NDArray[np.float64]
    mode_f1_hz: NDArray[np.float64]
    mode_f2_hz: NDArray[np.float64]
    mode_correlation_hz: NDArray[np.float64]
    mode_ar1_rho: NDArray[np.float64]
    sample_time_s: float
    prior_rms_m: float
    residual_kind: str
    model_note: str


@dataclass(frozen=True)
class PhotonKalmanResidual:
    """Frame-free photon-counting residual budget at the optimal window."""

    total_m2: float
    estimation_m2: float
    prediction_m2: float
    line_floor_m2: float
    window_s: float
    f_3db_equiv_hz: float


def von_karman_differential_piston_rms_m(
    *,
    r0_500nm_m: float,
    outer_scale_m: float,
    baseline_m: float = math.inf,
    aperture_diameter_m: float = 0.0,
) -> float:
    """Uncorrected differential piston RMS between two apertures (von Karman).

    Conan, Rousset & Madec (1995, JOSA A 12, 1559) aperture-filtered
    differential piston for two circular apertures of diameter ``D`` whose
    centers are ``B`` apart:

        sigma^2 = 2 int W_phi(f) [1 - J0(2 pi f B)] [2 J1(pi D f)/(pi D f)]^2 d^2f
        W_phi(f) = 0.023 r0^(-5/3) (f^2 + L0^(-2))^(-11/6)

    in rad^2 at 500 nm, converted to OPD with ``lambda / 2 pi``. Each
    aperture's piston is the average over its full clear disk (no central
    obstruction). Limits: ``B -> inf, D -> 0`` gives
    ``(12 pi / 5) 0.023 (L0/r0)^(5/3) = 0.1734 (L0/r0)^(5/3)``; ``B << L0,
    D -> 0`` gives the Kolmogorov phase structure function
    ``6.88 (B/r0)^(5/3)``. The defaults (``B = inf``, ``D = 0``) return the
    saturated point-aperture value, an upper bound for any finite geometry.

    Achromatic: ``r0 ~ lambda^(6/5)`` cancels the wavelength prefactor, so
    evaluating at 500 nm is exact for any science band.

    Args:
        r0_500nm_m: Fried parameter at 500 nm, in meters.
        outer_scale_m: Von Karman atmospheric outer scale, in meters.
        baseline_m: Aperture center separation in meters; ``inf`` for
            fully decorrelated apertures.
        aperture_diameter_m: Aperture diameter in meters; ``0`` for points.

    Returns:
        Differential piston RMS in meters.

    Raises:
        ValueError: If a physical scale is outside its valid range.
    """

    if r0_500nm_m <= 0.0 or not math.isfinite(float(r0_500nm_m)):
        raise ValueError("r0_500nm_m must be finite and positive.")
    if outer_scale_m <= 0.0 or not math.isfinite(float(outer_scale_m)):
        raise ValueError("outer_scale_m must be finite and positive.")
    if not baseline_m > 0.0:
        raise ValueError("baseline_m must be positive (inf allowed).")
    if aperture_diameter_m < 0.0 or not math.isfinite(float(aperture_diameter_m)):
        raise ValueError("aperture_diameter_m must be finite and non-negative.")

    sigma2_rad2 = (
        4.0 * math.pi * _VON_KARMAN_PSD_COEFFICIENT * float(r0_500nm_m) ** (-5.0 / 3.0)
        * _differential_piston_radial_integral(
            outer_scale_m=float(outer_scale_m),
            baseline_m=float(baseline_m),
            aperture_diameter_m=float(aperture_diameter_m),
        )
    )
    wavelength_500nm_m = 500.0e-9
    return float(math.sqrt(sigma2_rad2) * wavelength_500nm_m / (2.0 * math.pi))


@lru_cache(maxsize=64)
def _differential_piston_radial_integral(
    *,
    outer_scale_m: float,
    baseline_m: float,
    aperture_diameter_m: float,
) -> float:
    """``int_0^inf f (f^2 + k^2)^(-11/6) A(f) [1 - J0(2 pi f B)] df``.

    ``A(f)`` is the squared Airy piston filter. The grid is geometric
    (40 points per e-fold) below ``1/s`` and linear with step ``1/(40 s)``
    above it, ``s = max(B, D)``, so every Bessel oscillation is sampled.
    Above ``f_hi`` the ``J0`` term has averaged out; for points the
    remaining ``int f (f^2+k^2)^(-11/6) df`` tail is added in closed form,
    and for ``D > 0`` the ``f^(-17/3)`` filtered tail is negligible.
    """

    from scipy.special import j0, j1

    k2 = outer_scale_m**-2
    finite_baseline = math.isfinite(baseline_m)
    if not finite_baseline and aperture_diameter_m == 0.0:
        return 0.6 * k2 ** (-5.0 / 6.0)

    scale_m = max(baseline_m if finite_baseline else 0.0, aperture_diameter_m)
    f_lo = 1.0e-4 * min(1.0 / outer_scale_m, 1.0 / scale_m)
    if aperture_diameter_m > 0.0:
        f_hi = 200.0 / aperture_diameter_m
    else:
        f_hi = 200.0 * max(1.0 / outer_scale_m, 1.0 / baseline_m)
    f_knee = min(1.0 / scale_m, f_hi)
    f = np.concatenate(
        (
            np.geomspace(f_lo, f_knee, int(40.0 * math.log(f_knee / f_lo)) + 2),
            np.arange(f_knee, f_hi, 1.0 / (40.0 * scale_m))[1:],
        )
    )
    integrand = f * (f * f + k2) ** (-11.0 / 6.0)
    if finite_baseline:
        integrand = integrand * (1.0 - j0(2.0 * math.pi * f * baseline_m))
    if aperture_diameter_m > 0.0:
        x = math.pi * aperture_diameter_m * f
        integrand = integrand * (2.0 * j1(x) / x) ** 2
    total = float(np.trapezoid(integrand, f))
    # Analytic contribution below f_lo: the integrand is O(f^3) there when
    # B is finite and O(f) for B = inf.
    if not finite_baseline:
        total += 0.5 * f_lo**2 * k2 ** (-11.0 / 6.0)
    if aperture_diameter_m == 0.0:
        total += 0.6 * (f[-1] ** 2 + k2) ** (-5.0 / 6.0)
    return total


def piston_psd_knee_hz(
    *,
    wind_speed_m_s: float,
    outer_scale_m: float,
    aperture_diameter_m: float,
) -> float:
    """Knee of the decorrelated-piston temporal PSD.

    The simplified PSD is flat below the larger of the outer-scale flattening
    cutoff ``v / L0`` and the aperture-averaging cutoff ``0.3 v / D``, then
    falls approximately as ``f^(-17/3)``.

    Args:
        wind_speed_m_s: Effective wind speed in m/s.
        outer_scale_m: Atmospheric outer scale in meters.
        aperture_diameter_m: Primary aperture diameter in meters.

    Returns:
        Temporal PSD knee in Hz.

    Raises:
        ValueError: If any physical scale is not finite and positive.
    """

    if wind_speed_m_s <= 0.0 or not math.isfinite(float(wind_speed_m_s)):
        raise ValueError("wind_speed_m_s must be finite and positive.")
    if outer_scale_m <= 0.0 or not math.isfinite(float(outer_scale_m)):
        raise ValueError("outer_scale_m must be finite and positive.")
    if aperture_diameter_m <= 0.0 or not math.isfinite(float(aperture_diameter_m)):
        raise ValueError("aperture_diameter_m must be finite and positive.")

    wind = float(wind_speed_m_s)
    return float(
        max(
            wind / float(outer_scale_m),
            PISTON_PSD_APERTURE_CUTOFF_COEFFICIENT
            * wind
            / float(aperture_diameter_m),
        )
    )


def servo_residual_variance_flat_knee(
    *,
    sigma2: float,
    f_knee_hz: float,
    f_3db_hz: float,
    predictive_gain_variance: float,
) -> float:
    """Closed-loop residual of a flat-to-knee / f^(-17/3) disturbance.

    Rejection ``|1-H|^2 = (f/f_c)^2`` below ``f_c`` gives the closed form
    ``(7/12) sigma^2 (f_k/f_c)^2`` for ``f_c >> f_k``. The predictive gain
    divides the variance; a value of ``1`` means no predictive improvement.

    Args:
        sigma2: Uncorrected disturbance variance.
        f_knee_hz: Flat-PSD knee frequency in Hz.
        f_3db_hz: Closed-loop 3 dB bandwidth in Hz.
        predictive_gain_variance: Variance reduction factor.

    Returns:
        Residual variance in the same units as ``sigma2``.

    Raises:
        ValueError: If a finite physical input is outside its valid range.
    """

    if sigma2 < 0.0 or not math.isfinite(float(sigma2)):
        raise ValueError("sigma2 must be finite and non-negative.")
    if f_knee_hz < 0.0 or not math.isfinite(float(f_knee_hz)):
        raise ValueError("f_knee_hz must be finite and non-negative.")
    if f_3db_hz < 0.0 or not math.isfinite(float(f_3db_hz)):
        raise ValueError("f_3db_hz must be finite and non-negative.")
    if predictive_gain_variance <= 0.0 or not math.isfinite(
        float(predictive_gain_variance)
    ):
        raise ValueError("predictive_gain_variance must be finite and positive.")

    if sigma2 == 0.0:
        return 0.0
    if f_knee_hz == 0.0:
        return 0.0
    if f_3db_hz == 0.0:
        return float(sigma2)

    ratio = min(1.0, float(f_knee_hz) / float(f_3db_hz))
    return float(
        (7.0 / 12.0)
        * float(sigma2)
        * ratio**2
        / max(float(predictive_gain_variance), 1.0)
    )


def vibration_residual_rms_nm(
    *,
    line_frequencies_hz: Sequence[float],
    line_rms_nm: Sequence[float],
    broadband_rms_nm: float,
    broadband_max_hz: float,
    f_3db_hz: float,
) -> float:
    """Residual vibration OPD after the piston servo.

    Lines are residual-after-feedforward amplitudes. The servo additionally
    rejects each line by ``(f/f_c)^2`` below ``f_c``. The broadband floor is
    modeled as a flat PSD from 0 to ``broadband_max_hz``.

    Args:
        line_frequencies_hz: Vibration line frequencies in Hz.
        line_rms_nm: Residual line RMS amplitudes in nm.
        broadband_rms_nm: Broadband vibration floor RMS in nm.
        broadband_max_hz: Upper frequency of the flat broadband model.
        f_3db_hz: Closed-loop 3 dB bandwidth in Hz.

    Returns:
        Residual vibration RMS in nm.

    Raises:
        ValueError: If line arrays differ in length or any value is invalid.
    """

    if len(line_frequencies_hz) != len(line_rms_nm):
        raise ValueError("line_frequencies_hz and line_rms_nm must have equal length.")
    if broadband_rms_nm < 0.0 or not math.isfinite(float(broadband_rms_nm)):
        raise ValueError("broadband_rms_nm must be finite and non-negative.")
    if broadband_max_hz < 0.0 or not math.isfinite(float(broadband_max_hz)):
        raise ValueError("broadband_max_hz must be finite and non-negative.")
    if f_3db_hz < 0.0 or not math.isfinite(float(f_3db_hz)):
        raise ValueError("f_3db_hz must be finite and non-negative.")

    var_nm2 = 0.0
    for frequency_hz, rms_nm in zip(line_frequencies_hz, line_rms_nm):
        if frequency_hz < 0.0 or not math.isfinite(float(frequency_hz)):
            raise ValueError("line_frequencies_hz must be finite and non-negative.")
        if rms_nm < 0.0 or not math.isfinite(float(rms_nm)):
            raise ValueError("line_rms_nm must be finite and non-negative.")
        rejection = (
            min(1.0, (float(frequency_hz) / float(f_3db_hz)) ** 2)
            if f_3db_hz > 0.0
            else 1.0
        )
        var_nm2 += float(rms_nm) ** 2 * rejection

    if broadband_rms_nm > 0.0:
        if broadband_max_hz == 0.0:
            raise ValueError(
                "broadband_max_hz must be positive when broadband_rms_nm is non-zero."
            )
        f_3db = float(f_3db_hz)
        f_max = float(broadband_max_hz)
        if f_3db >= f_max:
            residual_fraction = f_max**2 / (3.0 * f_3db**2)
        else:
            residual_fraction = (f_3db / 3.0 + (f_max - f_3db)) / f_max
        var_nm2 += float(broadband_rms_nm) ** 2 * residual_fraction

    return float(math.sqrt(max(var_nm2, 0.0)))


def fusion_effective_broadband_rms_m(
    *,
    broadband_rms_m: float,
    coherent_fraction: float,
    suppression: float,
) -> float:
    """Accelerometer-feedforward fusion on the broadband vibration floor.

    The accelerometer-coherent fraction is suppressed by online-identified
    feedforward; the incoherent remainder passes through untouched.

    Args:
        broadband_rms_m: Unfused broadband vibration floor RMS in meters.
        coherent_fraction: Fraction of the floor coherent with accelerometer
            telemetry.
        suppression: RMS suppression factor applied to the coherent fraction.

    Returns:
        Effective broadband vibration RMS in meters.

    Raises:
        ValueError: If any input is outside its physical range.
    """

    if broadband_rms_m < 0.0 or not math.isfinite(float(broadband_rms_m)):
        raise ValueError("broadband_rms_m must be finite and non-negative.")
    if not 0.0 <= coherent_fraction <= 1.0:
        raise ValueError(f"coherent_fraction must be in [0, 1], got {coherent_fraction}")
    if suppression < 1.0 or not math.isfinite(float(suppression)):
        raise ValueError(f"suppression must be finite and >= 1, got {suppression}")
    return float(
        float(broadband_rms_m)
        * math.sqrt(
            float(coherent_fraction) / float(suppression) ** 2
            + (1.0 - float(coherent_fraction))
        )
    )


def piston_structure_function_m2(
    *,
    tau_s: float,
    atm_sigma2_m2: float,
    atm_f_knee_hz: float,
    broadband_sigma2_m2: float,
    broadband_max_hz: float,
) -> float:
    """Temporal piston structure function ``D(tau)`` in square meters.

    ``D(tau) = 2 int S(f) (1 - cos(2 pi f tau)) df``. The atmospheric PSD
    matches ``servo_residual_variance_flat_knee``: flat to ``f_knee`` and then
    proportional to ``f^(-17/3)``. The broadband vibration floor is flat from
    zero to ``broadband_max_hz``. The small-lag atmospheric limit is
    ``(7/12) sigma^2 (2 pi f_k tau)^2``.

    Args:
        tau_s: Temporal lag in seconds.
        atm_sigma2_m2: Atmospheric piston variance in m^2.
        atm_f_knee_hz: Atmospheric flat-PSD knee in Hz.
        broadband_sigma2_m2: Broadband vibration variance in m^2.
        broadband_max_hz: Upper frequency of the flat broadband floor.

    Returns:
        Structure-function variance in m^2.

    Raises:
        ValueError: If finite physical inputs are outside their valid range.
    """

    if tau_s < 0.0 or not math.isfinite(float(tau_s)):
        raise ValueError("tau_s must be finite and non-negative.")
    if atm_sigma2_m2 < 0.0 or not math.isfinite(float(atm_sigma2_m2)):
        raise ValueError("atm_sigma2_m2 must be finite and non-negative.")
    if atm_f_knee_hz < 0.0 or not math.isfinite(float(atm_f_knee_hz)):
        raise ValueError("atm_f_knee_hz must be finite and non-negative.")
    if broadband_sigma2_m2 < 0.0 or not math.isfinite(float(broadband_sigma2_m2)):
        raise ValueError("broadband_sigma2_m2 must be finite and non-negative.")
    if broadband_max_hz < 0.0 or not math.isfinite(float(broadband_max_hz)):
        raise ValueError("broadband_max_hz must be finite and non-negative.")
    if tau_s == 0.0:
        return 0.0
    if atm_sigma2_m2 > 0.0 and atm_f_knee_hz == 0.0:
        raise ValueError("atm_f_knee_hz must be positive when atm_sigma2_m2 is non-zero.")
    if broadband_sigma2_m2 > 0.0 and broadband_max_hz == 0.0:
        raise ValueError(
            "broadband_max_hz must be positive when broadband_sigma2_m2 is non-zero."
        )

    total_m2 = 0.0
    if atm_sigma2_m2 > 0.0:
        total_m2 += _atmospheric_structure_function_m2(
            tau_s=float(tau_s),
            sigma2_m2=float(atm_sigma2_m2),
            f_knee_hz=float(atm_f_knee_hz),
            broadband_max_hz=float(broadband_max_hz),
        )
    if broadband_sigma2_m2 > 0.0:
        argument = 2.0 * math.pi * float(broadband_max_hz) * float(tau_s)
        sinc = math.sin(argument) / argument if argument != 0.0 else 1.0
        total_m2 += 2.0 * float(broadband_sigma2_m2) * (1.0 - sinc)
    return float(max(total_m2, 0.0))


def photon_kalman_residual_budget(
    *,
    atm_sigma2_m2: float,
    atm_f_knee_hz: float,
    broadband_sigma2_m2: float,
    broadband_max_hz: float,
    tracked_line_count: int,
    line_tracking_floor_m: float,
    crlb_m_sqrt_s: float,
    latency_s: float,
) -> PhotonKalmanResidual:
    """Frame-free point-process piston residual: boxcar-estimate plus predict.

    The model minimizes over averaging window ``T``:

    ``crlb_m_sqrt_s^2 / T + D(T + latency_s) / 2 + n_lines * floor^2``.

    Tracked vibration lines are excluded from ``D`` because they are oscillator
    states in the filter (Menu et al. 2012; Poyneer et al. 2016) and contribute
    only their per-line tracking floor. This boxcar+predict model is
    conservative versus an optimal causal smoother by roughly 1.5-2x in
    variance, which keeps the classical closed-loop noise gain from being
    double-counted.

    Args:
        atm_sigma2_m2: Atmospheric piston variance in m^2.
        atm_f_knee_hz: Atmospheric flat-PSD knee in Hz.
        broadband_sigma2_m2: Broadband vibration variance in m^2.
        broadband_max_hz: Upper frequency of the broadband floor.
        tracked_line_count: Number of tracked narrowband vibration lines.
        line_tracking_floor_m: Residual floor per tracked line in meters.
        crlb_m_sqrt_s: Continuous-time measurement CRLB in m sqrt(s).
        latency_s: Estimator/actuator latency in seconds.

    Returns:
        Optimal residual budget.

    Raises:
        ValueError: If any input is outside its physical range.
    """

    if tracked_line_count < 0:
        raise ValueError("tracked_line_count must be non-negative.")
    if line_tracking_floor_m < 0.0 or not math.isfinite(float(line_tracking_floor_m)):
        raise ValueError("line_tracking_floor_m must be finite and non-negative.")
    if crlb_m_sqrt_s < 0.0 or not math.isfinite(float(crlb_m_sqrt_s)):
        raise ValueError("crlb_m_sqrt_s must be finite and non-negative.")
    if latency_s <= 0.0 or not math.isfinite(float(latency_s)):
        raise ValueError("latency_s must be finite and positive.")

    # MODELING ASSUMPTION (review 2026-06-10 Issue 16): the per-line
    # tracking floor is a fixed hardware-characterization input — it does
    # NOT scale with the configured line amplitudes and is not charged
    # against the photon budget. It represents the demonstrated residual
    # of a narrowband Kalman oscillator state locked onto a stable
    # vibration tone (Lozi+ 2018-class line tracking); at very faint flux
    # this is optimistic if the broadband CRLB exceeds the floor.
    line_floor_m2 = float(tracked_line_count) * float(line_tracking_floor_m) ** 2
    if float(crlb_m_sqrt_s) ** 2 <= 1.0e-32:
        prediction_m2 = (
            piston_structure_function_m2(
                tau_s=float(latency_s),
                atm_sigma2_m2=atm_sigma2_m2,
                atm_f_knee_hz=atm_f_knee_hz,
                broadband_sigma2_m2=broadband_sigma2_m2,
                broadband_max_hz=broadband_max_hz,
            )
            / 2.0
        )
        return PhotonKalmanResidual(
            total_m2=float(prediction_m2 + line_floor_m2),
            estimation_m2=0.0,
            prediction_m2=float(prediction_m2),
            line_floor_m2=float(line_floor_m2),
            window_s=0.0,
            f_3db_equiv_hz=float(1.0 / (2.0 * float(latency_s))),
        )

    windows_s = np.logspace(-5.0, math.log10(5.0e-2), 301)
    estimation_m2 = float(crlb_m_sqrt_s) ** 2 / windows_s
    prediction_m2 = np.array(
        [
            piston_structure_function_m2(
                tau_s=float(window_s) + float(latency_s),
                atm_sigma2_m2=atm_sigma2_m2,
                atm_f_knee_hz=atm_f_knee_hz,
                broadband_sigma2_m2=broadband_sigma2_m2,
                broadband_max_hz=broadband_max_hz,
            )
            / 2.0
            for window_s in windows_s
        ],
        dtype=np.float64,
    )
    totals_m2 = estimation_m2 + prediction_m2 + line_floor_m2
    index = int(np.argmin(totals_m2))
    window_s = float(windows_s[index])
    return PhotonKalmanResidual(
        total_m2=float(totals_m2[index]),
        estimation_m2=float(estimation_m2[index]),
        prediction_m2=float(prediction_m2[index]),
        line_floor_m2=float(line_floor_m2),
        window_s=window_s,
        f_3db_equiv_hz=float(1.0 / (2.0 * (window_s + float(latency_s)))),
    )


@dataclass(frozen=True)
class SetpointTrimBudget:
    """Photon-limited dark-port setpoint-trim operating point."""

    dither_amplitude_rad: float
    dither_rms_nm: float
    setpoint_residual_rad_per_mode: float
    setpoint_residual_nm_per_mode: float
    dither_leakage: float
    bias_leakage: float
    total_leakage: float


def photon_limited_setpoint_trim(
    *,
    stellar_rate_hz: float,
    dark_floor_null: float,
    trim_loop_bandwidth_hz: float,
    n_trim_modes: int,
    wavelength_m: float,
    null_curvature: float = 0.25,
) -> SetpointTrimBudget:
    """Jointly optimal trim dither amplitude and held setpoint residual.

    Sequential per-mode sinusoidal dither of amplitude A (rad) about a
    setpoint offset delta on the dark ports: the detected rate is
    ``S * (N_f + G (delta + A sin wt)^2)``, so the 1f lock-in signal rate is
    ``2 G A delta S`` against photon noise ``sqrt(N_f S / tau)`` over the
    per-mode dwell ``tau = 1/(2 f_trim n_modes)``. The photon-limited held
    setpoint is ``delta(A) = k / A`` with
    ``k = sqrt(N_f / (S tau)) / (sqrt(2) G)``, and the total added null

        L(A) = G A^2 / 2  +  n G delta(A)^2

    is minimized at ``A_opt^4 = 2 n k^2``, where the dither and bias terms
    partition equally. Smaller dithers are NOT free: they starve the lock-in
    and the setpoint drifts (delta ~ 1/A). Derivation 2026-06-10; see
    tests/physics/notes/fringe_disturbances.md.
    """
    if stellar_rate_hz <= 0.0 or trim_loop_bandwidth_hz <= 0.0 or n_trim_modes < 1:
        raise ValueError(
            "photon_limited_setpoint_trim needs positive stellar_rate_hz, "
            "trim_loop_bandwidth_hz, and n_trim_modes >= 1; got "
            f"{stellar_rate_hz}, {trim_loop_bandwidth_hz}, {n_trim_modes}"
        )
    if wavelength_m <= 0.0 or null_curvature <= 0.0:
        raise ValueError(
            "photon_limited_setpoint_trim needs positive wavelength_m and "
            f"null_curvature; got {wavelength_m}, {null_curvature}"
        )
    if dark_floor_null <= 0.0:
        return SetpointTrimBudget(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    dwell_s = 1.0 / (2.0 * trim_loop_bandwidth_hz * n_trim_modes)
    k = math.sqrt(dark_floor_null / (stellar_rate_hz * dwell_s)) / (
        math.sqrt(2.0) * null_curvature
    )
    amplitude_rad = (2.0 * n_trim_modes) ** 0.25 * math.sqrt(k)
    delta_rad = k / amplitude_rad
    dither_leakage = null_curvature * amplitude_rad**2 / 2.0
    bias_leakage = n_trim_modes * null_curvature * delta_rad**2
    rad_to_nm = wavelength_m / (2.0 * math.pi) * 1e9
    return SetpointTrimBudget(
        dither_amplitude_rad=amplitude_rad,
        dither_rms_nm=amplitude_rad / math.sqrt(2.0) * rad_to_nm,
        setpoint_residual_rad_per_mode=delta_rad,
        setpoint_residual_nm_per_mode=delta_rad * rad_to_nm,
        dither_leakage=dither_leakage,
        bias_leakage=bias_leakage,
        total_leakage=dither_leakage + bias_leakage,
    )


def _atmospheric_structure_function_m2(
    *,
    tau_s: float,
    sigma2_m2: float,
    f_knee_hz: float,
    broadband_max_hz: float,
) -> float:
    """Structure function for the flat-to-knee / f^(-17/3) piston PSD."""

    knee_lag = float(f_knee_hz) * float(tau_s)
    if 2.0 * math.pi * knee_lag < _STRUCTURE_FUNCTION_SMALL_ARGUMENT:
        return float(
            (7.0 / 12.0)
            * float(sigma2_m2)
            * (2.0 * math.pi * float(f_knee_hz) * float(tau_s)) ** 2
        )
    if knee_lag > _STRUCTURE_FUNCTION_SATURATION_ARGUMENT:
        return float(2.0 * float(sigma2_m2))

    f_min_hz = 1.0e-4
    f_max_hz = 100.0 * max(float(f_knee_hz), float(broadband_max_hz), 1.0)
    frequencies_hz = np.geomspace(
        f_min_hz,
        f_max_hz,
        _STRUCTURE_FUNCTION_GRID_SIZE,
        dtype=np.float64,
    )
    flat_level = float(sigma2_m2) * 14.0 / (17.0 * float(f_knee_hz))
    psd = np.where(
        frequencies_hz <= float(f_knee_hz),
        flat_level,
        flat_level * (frequencies_hz / float(f_knee_hz)) ** (-17.0 / 3.0),
    )
    response = 1.0 - np.cos(2.0 * math.pi * frequencies_hz * float(tau_s))
    return float(2.0 * np.trapezoid(psd * response, frequencies_hz))


def optimal_piston_servo_bandwidth(
    *,
    sigma2_temporal_rad2: float,
    f_char_hz: float,
    rate_per_sub_hz: float,
    visibility: float,
    predictive_lag_reduction: float = 1.0,
) -> float:
    """Analytic photon-limited optimum for the piston-servo bandwidth.

    Minimizes the sum of lag variance ``sigma_lag^2 ~ f_s^{-5/3}`` and
    photon-noise variance ``sigma_photon^2 ~ f_s``. The optimum is

        f_opt = [(5 / (3 pi)) * sigma2_temp * f_char^{5/3} * R * V^2
                 * predictive_lag_reduction] ^ (3/8)

    Derivation: set ``d/df_s [sigma_lag^2 + sigma_photon^2] = 0`` with
    ``sigma_lag^2 = predictive_lag_reduction * sigma2_temp * (f_char / f_s)^{5/3}``
    and ``sigma_photon^2 = pi * f_s / (R * V^2)``. Matches KILO
    ``optimal_piston_servo_bandwidth`` (``kilo/fringe_tracking.py``
    L286-333).

    Args:
        sigma2_temporal_rad2: Correctable temporal piston variance in
            rad^2 (the ``eta * sigma2_temp`` term already weighted by
            the piston-sensing efficiency, if the caller chooses to
            pre-multiply).
        f_char_hz: Sub-aperture piston crossing frequency in Hz
            (``0.134 * v_wind / d_sub``).
        rate_per_sub_hz: Detected bright-port photon rate per
            sub-aperture in counts/s. Returns ``0.0`` for non-positive
            rates so a starved configuration disables the servo rather
            than blowing up.
        visibility: Fringe visibility in [0, 1].
        predictive_lag_reduction: Variance-reduction multiplier on the
            lag term from a predictive (Kalman/LQG) controller. 1.0 ==
            leaky integrator (no prediction). Drives ``f_opt`` down by
            ``predictive_lag_reduction^(3/8)`` because lag becomes
            cheaper to reduce, so photon noise binds sooner.

    Returns:
        Optimal servo bandwidth in Hz, or 0.0 when any input would make
        the optimum non-physical (zero/negative rate or visibility).
    """

    if rate_per_sub_hz <= 0.0 or visibility <= 0.0:
        return 0.0
    if predictive_lag_reduction <= 0.0:
        return 0.0
    if sigma2_temporal_rad2 <= 0.0 or f_char_hz <= 0.0:
        return 0.0
    coeff = (5.0 / (3.0 * math.pi)) * sigma2_temporal_rad2 * f_char_hz ** (5.0 / 3.0)
    inner = coeff * rate_per_sub_hz * visibility ** 2 * predictive_lag_reduction
    if inner <= 0.0:
        return 0.0
    return float(inner ** (3.0 / 8.0))


def temporal_kalman_piston_covariance(
    *,
    measurement_covariance_m2: NDArray[np.floating],
    positions_m: NDArray[np.floating],
    prior_rms_m: float,
    sample_time_s: float,
    wind_speed_m_s: float,
    subaperture_diameter_m: float,
    residual_kind: str = "posterior",
) -> TemporalKalmanPistonCovariance:
    """Filter a piston measurement covariance with a temporal prior.

    The translation is intentionally conservative and explicit:

    * The bright-port CRLB is used as the per-frame measurement-noise
      covariance ``R``.
    * Each eigenmode of ``R`` is given an effective spatial baseline from the
      weighted RMS aperture separation of that piston mode.
    * That baseline is mapped to differential-piston PSD break frequencies
      using Conan, Rousset & Madec (1995, JOSA A 12, 1559) as summarized by
      Di Folco et al. (2002, SPIE 4838): ``f1 = 0.2 V / B`` and
      ``f2 = 0.3 V / D``, with PSD slopes ``-2/3``, ``-8/3``, and ``-17/3``.
      For compact sub-aperture modes where ``B < D``, ``B`` is clipped to
      ``D`` so the low-frequency break does not overtake the aperture break.
    * The geometric mean ``sqrt(f1 f2)`` is used as a one-pole AR(1)
      correlation frequency. That is not a full fitted PSD; it is the
      minimal state-space surrogate needed to capture that long-baseline
      differential piston evolves more slowly than short-baseline
      sub-aperture piston.
    * The scalar ``prior_rms_m`` supplies the stationary open-loop variance of
      each differential mode. Menu et al. (2012, A&A 541, A81), Lacour et al.
      (2019, A&A 624, A99), and the GRAVITY+ white-light controller describe
      this same practical pattern: an auto-regressive atmospheric/vibration
      disturbance model observed with additive fringe-measurement noise and
      estimated by a Kalman filter. This function implements the atmospheric
      AR(1) piece only; vibrations or measured telemetry PSDs should be added
      as extra state components when available.

    Args:
        measurement_covariance_m2: Square physical-mode covariance from the
            bright-port Fisher matrix, in m^2.
        positions_m: Physical aperture/sub-aperture positions, shape
            ``(n_modes, 2)`` or ``(n_modes, 3)``, in meters.
        prior_rms_m: Stationary RMS of a differential piston mode before the
            bright-port update. For KILO Keck N=6 this is resolved from the
            upstream AO piston residual, not from VLTI telemetry.
        sample_time_s: Fringe-tracker sample time.
        wind_speed_m_s: Effective turbulent-layer wind speed.
        subaperture_diameter_m: Physical sub-aperture diameter ``D``.
        residual_kind: ``"posterior"`` for measurement-updated residuals, or
            ``"one_step_prediction"`` for the one-frame-ahead predicted
            residual.

    Returns:
        TemporalKalmanPistonCovariance with full physical-mode matrices and
        per-mode diagnostics.

    Raises:
        ValueError: If dimensions or physical inputs are invalid.
    """

    if residual_kind not in {"posterior", "one_step_prediction"}:
        raise ValueError(
            "residual_kind must be 'posterior' or 'one_step_prediction'."
        )
    if prior_rms_m < 0.0 or not math.isfinite(float(prior_rms_m)):
        raise ValueError("prior_rms_m must be finite and non-negative.")
    if sample_time_s <= 0.0 or not math.isfinite(float(sample_time_s)):
        raise ValueError("sample_time_s must be finite and positive.")
    if wind_speed_m_s <= 0.0 or not math.isfinite(float(wind_speed_m_s)):
        raise ValueError("wind_speed_m_s must be finite and positive.")
    if (
        subaperture_diameter_m <= 0.0
        or not math.isfinite(float(subaperture_diameter_m))
    ):
        raise ValueError("subaperture_diameter_m must be finite and positive.")

    measurement = np.asarray(measurement_covariance_m2, dtype=np.float64)
    if measurement.ndim != 2 or measurement.shape[0] != measurement.shape[1]:
        raise ValueError("measurement_covariance_m2 must be a square matrix.")
    if not np.all(np.isfinite(measurement)):
        raise ValueError("measurement_covariance_m2 must be finite.")
    if not np.allclose(measurement, measurement.T, rtol=1.0e-10, atol=1.0e-30):
        raise ValueError("measurement_covariance_m2 must be symmetric.")
    measurement = 0.5 * (measurement + measurement.T)

    positions = np.asarray(positions_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[0] != measurement.shape[0]:
        raise ValueError(
            "positions_m must have shape (n_modes, 2) or (n_modes, 3) "
            "matching measurement_covariance_m2."
        )
    if positions.shape[1] not in {2, 3}:
        raise ValueError("positions_m must have two or three columns.")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions_m must be finite.")
    positions_xy = positions[:, :2]

    evals, evecs = np.linalg.eigh(measurement)
    scale = max(float(np.max(np.abs(evals))) if evals.size else 0.0, 1.0e-30)
    evals = np.where(evals > scale * 1.0e-12, evals, 0.0)

    n_modes = measurement.shape[0]
    prior_variance = float(prior_rms_m) ** 2
    prior_var = np.zeros(n_modes, dtype=np.float64)
    process_var = np.zeros(n_modes, dtype=np.float64)
    posterior_var = np.zeros(n_modes, dtype=np.float64)
    prediction_var = np.zeros(n_modes, dtype=np.float64)
    baselines = np.zeros(n_modes, dtype=np.float64)
    f1_hz = np.zeros(n_modes, dtype=np.float64)
    f2_hz = np.zeros(n_modes, dtype=np.float64)
    f_corr_hz = np.zeros(n_modes, dtype=np.float64)
    rho = np.ones(n_modes, dtype=np.float64)

    for index in range(n_modes):
        mode = evecs[:, index]
        baseline = _effective_mode_baseline_m(mode, positions_xy)
        baselines[index] = baseline
        # An unobserved mode (evals[index] == 0) must NOT be skipped here: with
        # zero measurement information its one-step prediction carries the full
        # process variance, not zero. _steady_state_scalar_kalman_variances
        # handles r_var == 0 correctly (posterior 0, prediction q_var). The
        # common-piston mode is still excluded by its zero baseline.
        if baseline <= 0.0 or prior_variance <= 0.0:
            continue

        baseline_for_f1 = max(baseline, float(subaperture_diameter_m))
        f1 = 0.2 * float(wind_speed_m_s) / baseline_for_f1
        f2 = 0.3 * float(wind_speed_m_s) / float(subaperture_diameter_m)
        f_corr = math.sqrt(max(f1 * f2, 0.0))
        mode_rho = math.exp(-2.0 * math.pi * f_corr * float(sample_time_s))
        mode_rho = min(max(mode_rho, 0.0), 1.0)

        prior_var[index] = prior_variance
        f1_hz[index] = f1
        f2_hz[index] = f2
        f_corr_hz[index] = f_corr
        rho[index] = mode_rho
        process_var[index] = prior_variance * max(1.0 - mode_rho**2, 0.0)
        posterior_var[index], prediction_var[index] = (
            _steady_state_scalar_kalman_variances(
                measurement_variance_m2=float(evals[index]),
                stationary_variance_m2=prior_variance,
                rho=mode_rho,
            )
        )

    posterior = _covariance_from_eigenmodes(evecs, posterior_var)
    prediction = _covariance_from_eigenmodes(evecs, prediction_var)
    selected = posterior if residual_kind == "posterior" else prediction
    note = (
        "AR(1) temporal Kalman surrogate: measurement covariance is the "
        "bright-port CRLB; modal correlation frequencies use the Conan/Di "
        "Folco differential-piston PSD breaks f1=0.2V/B and f2=0.3V/D; "
        "the selected residual is the "
        f"{residual_kind.replace('_', ' ')} covariance."
    )
    return TemporalKalmanPistonCovariance(
        measurement_covariance_m2=measurement.astype(np.float64, copy=True),
        posterior_covariance_m2=posterior,
        prediction_covariance_m2=prediction,
        selected_covariance_m2=selected.astype(np.float64, copy=True),
        mode_measurement_variance_m2=evals.astype(np.float64, copy=True),
        mode_prior_variance_m2=prior_var,
        mode_process_variance_m2=process_var,
        mode_posterior_variance_m2=posterior_var,
        mode_prediction_variance_m2=prediction_var,
        mode_baseline_m=baselines,
        mode_f1_hz=f1_hz,
        mode_f2_hz=f2_hz,
        mode_correlation_hz=f_corr_hz,
        mode_ar1_rho=rho,
        sample_time_s=float(sample_time_s),
        prior_rms_m=float(prior_rms_m),
        residual_kind=residual_kind,
        model_note=note,
    )


def _effective_mode_baseline_m(
    mode: NDArray[np.float64],
    positions_xy_m: NDArray[np.float64],
) -> float:
    """Return the two-aperture baseline with the same Kolmogorov variance.

    For a zero-sum piston mode ``c`` the phase variance is
    ``-1/2 sum_ij c_i c_j D(r_ij)`` with ``D(r) = 6.88 (r/r0)^(5/3)``, so

        B_eff = [sum_{i!=j} (-c_i c_j) r_ij^(5/3) / sum_i c_i^2]^(3/5)

    is the baseline whose two-aperture differential ``(1, -1)/sqrt(2)``
    has that variance per unit norm. The mean is projected out first, and
    apertures with zero coefficient contribute nothing.
    """

    coeff = np.asarray(mode, dtype=np.float64)
    coeff = coeff - float(np.mean(coeff))
    norm2 = float(np.sum(coeff * coeff))
    if norm2 <= 1.0e-24 * max(float(np.sum(np.asarray(mode) ** 2)), 1.0e-300):
        return 0.0
    delta = positions_xy_m[:, np.newaxis, :] - positions_xy_m[np.newaxis, :, :]
    dist_5_3 = np.sum(delta * delta, axis=2) ** (5.0 / 6.0)
    weighted = -float(coeff @ dist_5_3 @ coeff)
    return float(max(weighted / norm2, 0.0) ** 0.6)


def _steady_state_scalar_kalman_variances(
    *,
    measurement_variance_m2: float,
    stationary_variance_m2: float,
    rho: float,
) -> tuple[float, float]:
    """Return posterior and one-step prediction variances for an AR(1) mode."""

    r_var = max(float(measurement_variance_m2), 0.0)
    p_stationary = max(float(stationary_variance_m2), 0.0)
    if p_stationary == 0.0:
        return 0.0, 0.0
    if r_var == 0.0:
        q_var = p_stationary * max(1.0 - float(rho) ** 2, 0.0)
        return 0.0, q_var

    rho2 = min(max(float(rho), 0.0), 1.0) ** 2
    q_var = p_stationary * max(1.0 - rho2, 0.0)
    posterior = p_stationary
    for _ in range(256):
        prediction = rho2 * posterior + q_var
        gain = prediction / (prediction + r_var)
        updated = max((1.0 - gain) * prediction, 0.0)
        if abs(updated - posterior) <= max(p_stationary, r_var, 1.0e-30) * 1.0e-13:
            posterior = updated
            break
        posterior = updated
    prediction = rho2 * posterior + q_var
    return float(posterior), float(prediction)


def _covariance_from_eigenmodes(
    eigenvectors: NDArray[np.float64],
    variances: NDArray[np.float64],
) -> NDArray[np.float64]:
    covariance = (eigenvectors * variances[np.newaxis, :]) @ eigenvectors.T
    covariance = 0.5 * (covariance + covariance.T)
    return covariance.astype(np.float64, copy=True)
