"""Atmospheric scintillation and amplitude-servo residual model.

Originally a direct port of KILO ``kilo/atmosphere.py``; re-anchored to the
measured aperture-averaged scintillation statistics in the 2026-07-13 audit
(NullSimTD task #44). The audit found the KILO law wrong in three ways for
meter-class sub-apertures, all inflating amplitude noise:

- The 0.041/0.06 anchor was ~30x the Kornilov et al. (2012) Maunakea
  measurement: their first-quartile S_2^2 = 0.77e-4 m^(7/3) gives
  σ_I(5 m, zenith) = 1.3e-3 for any wavelength.
- The λ^(-7/12) factor does not exist in the aperture-averaged regime —
  the large-aperture index is achromatic (Kenyon et al. 2006; Dravins
  et al. 1998; Kornilov et al. 2012 all state it explicitly).
- The r_F/v Fresnel-crossing timescale belongs to the point-source
  regime; aperture averaging removes exactly that fast structure, and the
  surviving fluctuations decorrelate on the aperture crossing time
  (~0.5 d/v; PSD knee at v/(π d), Dravins et al. 1998).

Current model:

- :func:`scintillation_index`: aperture-averaged Young/Roddier σ_I per
  sub-aperture, σ_I ∝ d_sub^(-7/6) × airmass^(3/2), wavelength-independent,
  anchored at the Kornilov Q1 Maunakea 5 m zenith value. Saturation cap
  at 0.5 below the Fresnel radius.
- :func:`scintillation_timescale_for_servo_design`: aperture crossing
  time 0.5 d_sub / v_high (measured ACF e-fold 0.47 d/v in the Fresnel MC
  against the kilo layer profile).
- :func:`amplitude_servo_residual_sigma_i`: residual σ_I after a
  closed-loop amplitude servo. Combines a (τ_loop/τ_scint)^(5/6)
  temporal residual (variance-reduced by predictive control) with the
  1/√N photon-noise floor in quadrature, saturated at the open-loop
  value.

The residual σ_I per sub-aperture feeds
:func:`nullsim.physics.realized_null.realized_null_depth_mc` as the
``scintillation_index_per_sub`` parameter. Diagnostics should report both
open- and closed-loop values because the predictive gain is an instrument
assumption, not a property of the chip.

References:
    Roddier, F. "The effects of atmospheric turbulence in optical
        astronomy." Prog. Opt. 19, 281 (1981).
    Dravins, D. et al. "Atmospheric intensity scintillation of stars III."
        PASP 110, 610 (1998). Large-aperture temporal statistics.
    Kenyon, S. L. et al. PASP 118, 924 (2006). Aperture-averaged
        σ_I² = 17.34 D^(-7/3) (cos γ)^(-3) ∫Cn²(h) h² dh, achromatic.
    Kornilov, V. et al. A&A 546, A41 (2012). Measured Maunakea S_2/S_3
        scintillation statistics (the anchor source).
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
    scintillation_index_h5m_zenith: float = 0.0013,
    scintillation_height_km: float = 10.0,
    airmass: float = 1.0,
) -> float:
    """RMS fractional intensity per sub-aperture (open-loop).

    Aperture-averaged Young/Roddier regime (d_sub > r_F):
    σ_I ∝ d_sub^(-7/6) × airmass^(3/2), **wavelength-independent**
    (Kenyon et al. 2006 Eq. 7; the λ² of the Fresnel filter cancels the
    k² prefactor once the aperture cuts off spatial frequencies below
    1/r_F). Anchored at Maunakea, 5 m sub, zenith via
    ``scintillation_index_h5m_zenith``: the default 1.3e-3 is the
    Kornilov et al. (2012) first-quartile measurement
    (S_2² = 0.77e-4 m^(7/3) → σ_I(5 m) = √(S_2² · 5^(-7/3))); their
    median is 1.7e-3. Equivalent turbulence moment:
    ∫Cn²h²dh = anchor² · 5^(7/3) / 17.34 ≈ 4.2e-6 m^(7/3) at the default.

    ``wavelength_m`` only sets the Fresnel radius r_F = √(λ h) guard:
    below r_F the d^(-7/6) extrapolation is unphysical and the result is
    clipped at :data:`SCINTILLATION_SATURATION_INDEX`, with a warning.
    """
    if wavelength_m <= 0:
        raise ValueError(f"wavelength_m must be > 0, got {wavelength_m}")
    if sub_diameter_m <= 0:
        raise ValueError(f"sub_diameter_m must be > 0, got {sub_diameter_m}")
    if airmass < 1.0:
        raise ValueError(f"airmass must be >= 1.0, got {airmass}")

    d_scale = (5.0 / sub_diameter_m) ** (7.0 / 6.0)
    am_scale = airmass ** 1.5
    sigma = scintillation_index_h5m_zenith * d_scale * am_scale

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


# Measured e-folding of the aperture-averaged intensity ACF in units of the
# aperture crossing time d/v: 0.47 in the 2026-07-13 Fresnel MC against the
# kilo 3-layer profile, consistent with the MTF-lobe half-width ~0.5 D/V of
# Dravins et al. (1998).
_CROSSING_TIME_FRACTION: float = 0.5


def scintillation_timescale_for_servo_design(
    *,
    sub_diameter_m: float,
    high_altitude_wind_ms: float = 35.0,
) -> float:
    """Aperture-averaged scintillation correlation time (s).

    Returns ``0.5 · d_sub / v_high`` — the aperture crossing time at the
    jet-stream wind. Aperture averaging removes the Fresnel-scale
    structure, so the surviving intensity fluctuations decorrelate when
    the frozen-flow pattern crosses the sub-aperture (PSD knee at
    v/(π d), Dravins et al. 1998), NOT at the point-source r_F/v
    timescale the KILO port used (~3.7 ms; ~13x too fast for meter-class
    sub-apertures — 2026-07-13 audit).
    """
    if sub_diameter_m <= 0:
        raise ValueError(f"sub_diameter_m must be > 0, got {sub_diameter_m}")
    if high_altitude_wind_ms <= 0:
        raise ValueError(
            f"high_altitude_wind_ms must be > 0, got {high_altitude_wind_ms}"
        )
    return _CROSSING_TIME_FRACTION * sub_diameter_m / high_altitude_wind_ms


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
    scintillation_index_h5m_zenith: float = 0.0013,
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
        sub_diameter_m=sub_diameter_m,
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
    scintillation_index_h5m_zenith: float = 0.0013,
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
        sub_diameter_m=sub_diameter_m,
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
