"""Atmospheric scintillation and amplitude-servo residual model.

Direct port of KILO ``kilo/atmosphere.py`` functions:

- :func:`scintillation_index` (KILO L95-168): Roddier 1981 σ_I per
  sub-aperture, scaled from a Maunakea H-band/5 m/zenith anchor by
  λ^(-7/12), d_sub^(-7/6), airmass^(3/2). Saturation cap at 0.5.
- :func:`scintillation_timescale_for_servo_design` (KILO L195-228):
  r_F / v_high — jet-stream-driven timescale used to size the amplitude
  fringe tracker. **Not** the boundary-layer τ from
  ``scintillation_timescale``; that under-sizes the servo bandwidth.
- :func:`amplitude_servo_residual_sigma_i` (KILO L311-481): residual σ_I
  after a closed-loop amplitude servo. Combines a (τ_loop/τ_scint)^(5/6)
  temporal residual (variance-reduced by predictive control) with the
  1/√N photon-noise floor in quadrature, saturated at the open-loop
  value.

The residual σ_I per sub-aperture feeds
:func:`nullsim.physics.realized_null.realized_null_depth_mc` as the
``scintillation_index_per_sub`` parameter. The current KILO Keck baseline
auto-optimizes the amplitude-servo bandwidth per target and assumes a
predictive variance gain, so bright J-band cells can inject residuals near
the percent level even when the open-loop sub-aperture scintillation is much
larger. Diagnostics should report both values because the predictive gain is
an instrument assumption, not a property of the chip.

References:
    Roddier, F. "The effects of atmospheric turbulence in optical
        astronomy." Prog. Opt. 19, 281 (1981).
    Dravins, D. et al. "Atmospheric intensity scintillation of stars."
        PASP 109, 173 (1997).
    Tokovinin, A. & Kornilov, V. "Accurate seeing measurements with
        MASS and DIMM." MNRAS 381, 1179 (2007). Maunakea high-altitude
        wind statistics.
    Kornilov, V. et al. A&A 546, A41 (2012). Maunakea free-atmosphere
        S_3 statistics.
    Poyneer, L. A. et al. JOSA A 24, 2645 (2007). On-sky predictive
        Fourier AO at GPI.
    van Kooten, M. A. M. et al. A&A 636, A81 (2020). SPHERE on-sky LQG.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

__all__ = [
    "SCINTILLATION_SATURATION_INDEX",
    "AmplitudeServoResidual",
    "amplitude_servo_residual_sigma_i",
    "optimize_amplitude_servo_bandwidth",
    "scintillation_index",
    "scintillation_timescale_for_servo_design",
]


# Beer-Lambert intensity flickers are bounded in the 0.5..1.5 range; the
# small-aperture saturation cap on σ_I (RMS fractional intensity) caps
# the d^(-7/6) extrapolation when d_sub falls below the Fresnel radius.
SCINTILLATION_SATURATION_INDEX: float = 0.5


def scintillation_index(
    *,
    wavelength_m: float,
    sub_diameter_m: float,
    scintillation_index_h5m_zenith: float = 0.041,
    scintillation_height_km: float = 10.0,
    airmass: float = 1.0,
) -> float:
    """RMS fractional intensity per sub-aperture (open-loop).

    Direct port of KILO ``atmosphere.scintillation_index``. The
    aperture-averaging large-aperture (d_sub > r_F) regime scales as
    σ_I ∝ λ^(-7/12) × d_sub^(-7/6) × airmass^(3/2), anchored at
    Maunakea H-band, 5 m sub, zenith via
    ``scintillation_index_h5m_zenith`` (KILO default 0.041 — the
    Q1-night Kornilov+2012 anchor).

    Below the Fresnel radius r_F = √(λ h) the d^(-7/6) extrapolation
    is unphysical and the result is clipped at
    :data:`SCINTILLATION_SATURATION_INDEX`, with a warning.
    """
    if wavelength_m <= 0:
        raise ValueError(f"wavelength_m must be > 0, got {wavelength_m}")
    if sub_diameter_m <= 0:
        raise ValueError(f"sub_diameter_m must be > 0, got {sub_diameter_m}")
    if airmass < 1.0:
        raise ValueError(f"airmass must be >= 1.0, got {airmass}")

    wl_scale = (1.65e-6 / wavelength_m) ** (7.0 / 12.0)
    d_scale = (5.0 / sub_diameter_m) ** (7.0 / 6.0)
    am_scale = airmass ** 1.5
    sigma = scintillation_index_h5m_zenith * wl_scale * d_scale * am_scale

    r_fresnel = math.sqrt(wavelength_m * scintillation_height_km * 1000.0)
    if sub_diameter_m < r_fresnel:
        warnings.warn(
            f"sub_diameter_m={sub_diameter_m:.3f} m is below the Fresnel "
            f"radius r_F={r_fresnel:.3f} m at this wavelength; the "
            f"d^(-7/6) extrapolation over-predicts sigma_I. Clipping to "
            f"the saturation cap {SCINTILLATION_SATURATION_INDEX:.2f}.",
            stacklevel=2,
        )
    return float(min(sigma, SCINTILLATION_SATURATION_INDEX))


def scintillation_timescale_for_servo_design(
    *,
    wavelength_m: float,
    scintillation_height_km: float = 10.0,
    high_altitude_wind_ms: float = 35.0,
) -> float:
    """Scintillation timescale at the jet-stream wind (s).

    Direct port of KILO
    ``atmosphere.scintillation_timescale_for_servo_design``. Returns
    r_F / v_high — the dominant scintillation-generating layer at
    Maunakea is the tropopause jet stream (~10 km, 30-40 m/s), giving
    ~3.7 ms at H-band. Use this — **not** the boundary-layer τ — to
    size the amplitude fringe tracker.
    """
    if wavelength_m <= 0:
        raise ValueError(f"wavelength_m must be > 0, got {wavelength_m}")
    if high_altitude_wind_ms <= 0:
        raise ValueError(
            f"high_altitude_wind_ms must be > 0, got {high_altitude_wind_ms}"
        )
    r_fresnel = math.sqrt(wavelength_m * scintillation_height_km * 1000.0)
    return r_fresnel / high_altitude_wind_ms


@dataclass(frozen=True)
class AmplitudeServoResidual:
    """Residual amplitude-jitter breakdown after the amplitude fringe tracker.

    All fields are dimensionless (sigma_I has units of fractional intensity).
    """

    sigma_I_residual: float
    sigma_I_open: float
    sigma_I_temporal: float
    sigma_I_photon: float
    tau_loop_s: float
    tau_scint_s: float
    n_photons_per_sample: float
    predictive_gain_variance: float
    servo_bandwidth_hz: float


def optimize_amplitude_servo_bandwidth(
    *,
    wavelength_m: float,
    sub_diameter_m: float,
    photon_rate_per_sub_hz: float,
    scintillation_index_h5m_zenith: float = 0.041,
    scintillation_height_km: float = 10.0,
    airmass: float = 1.0,
    predictive_gain_variance: float = 20.0,
    high_altitude_wind_ms: float = 35.0,
    bandwidth_min_hz: float = 200.0,
    bandwidth_max_hz: float = 10_000.0,
) -> float:
    """Photon-limited optimum bandwidth for the amplitude FT servo.

    Direct port of KILO ``optimize_amplitude_servo_bandwidth``
    (``kilo/atmosphere.py:230-308``). Closed-form minimum of the sum

        V(f) = σ_open² / (g_pred · τ_scint^{5/3}) · f^{-5/3} + f / R

    where ``R`` is the per-sub WFS photon rate. Setting ``dV/df = 0``
    gives

        f_opt = ((5/3) · P · R)^{3/8},    P = σ_open² · τ_scint^{-5/3} / g_pred.

    SNSPDs and electrostatic phase shifters make multi-kHz amplitude
    loops electronically feasible; the binding constraint is photon
    noise per sample. Auto-optimizing per target avoids running a
    faster loop than the photon budget supports (which lifts the
    residual rather than lowering it). At J=4 Keck-only the optimum
    lands around 4-5 kHz; at J ≥ 8 it drops below 1 kHz, so the
    pre-Phase-2.1 single ``auto_amp_servo_bandwidth_hz=5000`` default
    over-budgets at faint J.

    See :func:`amplitude_servo_residual_sigma_i` for the matched
    residual computation that consumes the chosen bandwidth. Floor at
    ``bandwidth_min_hz`` (below which the servo can't outrun seeing
    variations) and cap at ``bandwidth_max_hz`` (past the optimum for
    even the brightest hosts and approaches SNSPD saturation).
    """
    if photon_rate_per_sub_hz <= 0:
        raise ValueError(
            f"photon_rate_per_sub_hz must be > 0, got {photon_rate_per_sub_hz}"
        )
    if predictive_gain_variance < 1.0:
        raise ValueError(
            "predictive_gain_variance must be >= 1.0 (1.0 = no prediction), "
            f"got {predictive_gain_variance}"
        )
    if bandwidth_min_hz <= 0 or bandwidth_max_hz <= bandwidth_min_hz:
        raise ValueError(
            "bandwidth bounds invalid: "
            f"min={bandwidth_min_hz}, max={bandwidth_max_hz}"
        )

    sigma_open = scintillation_index(
        wavelength_m=wavelength_m,
        sub_diameter_m=sub_diameter_m,
        scintillation_index_h5m_zenith=scintillation_index_h5m_zenith,
        scintillation_height_km=scintillation_height_km,
        airmass=airmass,
    )
    tau_scint = scintillation_timescale_for_servo_design(
        wavelength_m=wavelength_m,
        scintillation_height_km=scintillation_height_km,
        high_altitude_wind_ms=high_altitude_wind_ms,
    )

    p_coef = (sigma_open ** 2) / (
        predictive_gain_variance * (tau_scint ** (5.0 / 3.0))
    )
    f_opt = ((5.0 / 3.0) * p_coef * photon_rate_per_sub_hz) ** (3.0 / 8.0)
    return float(min(max(f_opt, bandwidth_min_hz), bandwidth_max_hz))


def amplitude_servo_residual_sigma_i(
    *,
    wavelength_m: float,
    sub_diameter_m: float,
    servo_bandwidth_hz: float,
    photon_rate_per_sub_hz: float,
    scintillation_index_h5m_zenith: float = 0.041,
    scintillation_height_km: float = 10.0,
    airmass: float = 1.0,
    predictive_gain_variance: float = 20.0,
    high_altitude_wind_ms: float = 35.0,
) -> AmplitudeServoResidual:
    """Residual σ_I per sub-aperture after the closed-loop amplitude servo.

    Direct port of KILO ``compute_amplitude_servo_residual_sigma_i``
    (kilo/atmosphere.py:311-481). Two quadrature terms:

    1. **Temporal lag residual** — uncorrected Kolmogorov amplitude
       spectrum above ~1/(2 τ_loop). Scales as
       σ_I_open × (τ_loop / τ_scint)^(5/6) / √(predictive_gain_variance).
    2. **Photon-noise floor** — 1/√(N_photons_per_servo_sample) sets a
       hard lower bound on the corrected residual.

    The returned residual is capped at σ_I_open (the controller can
    always choose to do nothing).
    """
    if servo_bandwidth_hz <= 0:
        raise ValueError(
            f"servo_bandwidth_hz must be > 0, got {servo_bandwidth_hz}"
        )
    if photon_rate_per_sub_hz <= 0:
        raise ValueError(
            f"photon_rate_per_sub_hz must be > 0, got {photon_rate_per_sub_hz}"
        )
    if predictive_gain_variance < 1.0:
        raise ValueError(
            "predictive_gain_variance must be >= 1.0 (1.0 = no prediction), "
            f"got {predictive_gain_variance}"
        )

    sigma_open = scintillation_index(
        wavelength_m=wavelength_m,
        sub_diameter_m=sub_diameter_m,
        scintillation_index_h5m_zenith=scintillation_index_h5m_zenith,
        scintillation_height_km=scintillation_height_km,
        airmass=airmass,
    )
    tau_loop = 1.0 / servo_bandwidth_hz
    tau_scint = scintillation_timescale_for_servo_design(
        wavelength_m=wavelength_m,
        scintillation_height_km=scintillation_height_km,
        high_altitude_wind_ms=high_altitude_wind_ms,
    )

    temporal_ratio = (tau_loop / tau_scint) ** (5.0 / 6.0)
    sigma_temporal = (
        sigma_open * temporal_ratio / math.sqrt(predictive_gain_variance)
    )

    n_photons_per_sample = photon_rate_per_sub_hz * tau_loop
    sigma_photon = (
        1.0 / math.sqrt(n_photons_per_sample)
        if n_photons_per_sample > 0
        else float("inf")
    )

    sigma_residual = math.sqrt(sigma_temporal ** 2 + sigma_photon ** 2)
    sigma_residual = min(sigma_residual, sigma_open)

    return AmplitudeServoResidual(
        sigma_I_residual=float(sigma_residual),
        sigma_I_open=float(sigma_open),
        sigma_I_temporal=float(sigma_temporal),
        sigma_I_photon=float(sigma_photon),
        tau_loop_s=float(tau_loop),
        tau_scint_s=float(tau_scint),
        n_photons_per_sample=float(n_photons_per_sample),
        predictive_gain_variance=float(predictive_gain_variance),
        servo_bandwidth_hz=float(servo_bandwidth_hz),
    )
