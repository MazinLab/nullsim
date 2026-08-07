"""Matched-filter cross-correlation (CCF) budget for high dispersion coronagraphy.

High dispersion coronagraphy (HDC) buys contrast by moving the planet detection
from a broadband photometric comparison to a spectral one: the planet signal is
a forest of narrow molecular lines, while the leaked starlight that dominates a
dark port is spectrally smooth. Continuum filtering removes the smooth part; the
surviving line forest is then correlated against a template. The framing and the
sqrt(N_lines) scaling follow Wang, Mawet, Ruane, Hu & Benneke 2017, AJ 153, 183
("Observing Exoplanets with High Dispersion Coronagraphy. I.").

Noise model
-----------
Work on ``n_ch`` spectrograph channels, each carrying rates in Hz: planet ``P``,
leaked starlight ``L``, sky ``S_sky``, and detector dark ``D``. Let ``m_i`` be a
telluric mask (1 where the atmospheric transmission clears a threshold) and let
``~`` denote continuum filtering (`continuum_filter`, a running-median high-pass).

- Per-channel noise-variance rate ``v_i = P_i + L_i + S_sky,i + D_i`` (Hz).
  Poisson statistics make this both the mean and the variance rate.
- Matched-filter weights for independent channels: ``w_i = m_i * Ptilde_i / v_i``.
- Signal ``S = t * sum_i w_i Ptilde_i`` (counts).
- Photon variance ``V_ph = t * sum_i w_i^2 v_i`` (counts^2). With the weights
  above this collapses to ``V_ph = S``, so ``SNR_photon = sqrt(t sum_i
  Ptilde_i^2 / v_i)`` -- the usual matched-filter result, and the broadband
  ``S/sqrt(S+B)`` when the template is flat.
- Systematic: the leaked starlight fluctuates coherently across the band with
  fractional amplitude ``eps`` (null instability). What that fluctuation injects
  into the statistic is not the raw band leak but its projection onto the
  template, ``B_w = sum_i w_i Ltilde_i`` (Hz). The two regimes mirror
  ``nullsim.stages.performance.standard._systematic_variance`` exactly, with
  ``B_w`` substituted for the raw dark-port leakage rate:

  * fringe tracking on -- the instability averages down over the servo band,
    ``V_sys = eps^2 B_w^2 t / (2 f_servo)`` (counts^2, grows like ``t``);
  * fringe tracking off -- the offset is static, ``V_sys = (eps B_w t)^2``
    (counts^2, grows like ``t^2``), so the SNR saturates.

- ``ccf_snr = S / sqrt(V_ph + V_sys)``; ``ccf_snr_photon_limited = S/sqrt(V_ph)``.

Two summary diagnostics come along for the ride:

- ``self_subtraction_factor = sqrt(sum_i Ptilde_i^2/v_i) / sqrt(sum_i P_i^2/v_i)``:
  the fraction of the achievable photon-limited *SNR* (not flux) that survives
  the continuum filter, over the unmasked channels.
- ``n_effective_lines = (sum_i |w_i|)^2 / sum_i w_i^2``: the participation ratio
  of the weights, i.e. how many channels the statistic actually leans on.

Pure numpy/scipy: no TOML, ``SimulationState``, or stage imports.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter

from nullsim.physics.spectra import (
    constant_r_grid,
    convolve_to_resolving_power,
    resample_flux_conserving,
)

__all__ = [
    "CcfBudget",
    "ChannelSpectra",
    "build_channel_edges",
    "ccf_vs_rv",
    "channelize",
    "continuum_filter",
    "matched_filter_ccf",
    "snr_vs_resolving_power",
]


@dataclass(frozen=True, eq=False)
class ChannelSpectra:
    """Per-channel photon rates on a spectrograph's channel grid.

    All rate arrays are per channel in Hz (already integrated over the channel
    bandpass, not densities). ``channel_edges_um`` has one more element than the
    rate arrays and must be strictly ascending.
    """

    channel_centers_um: NDArray[np.float64]
    channel_edges_um: NDArray[np.float64]
    planet_rate_hz: NDArray[np.float64]
    leak_rate_hz: NDArray[np.float64]
    sky_rate_hz: NDArray[np.float64]
    dark_rate_hz: NDArray[np.float64]

    def __post_init__(self) -> None:
        edges = np.asarray(self.channel_edges_um, dtype=np.float64)
        if edges.ndim != 1:
            raise ValueError(f"channel_edges_um must be 1-D, got {edges.shape}")
        if edges.size < 2:
            raise ValueError("channel_edges_um needs at least 2 edges")
        if not np.all(np.isfinite(edges)):
            raise ValueError("channel_edges_um must be finite")
        if not np.all(np.diff(edges) > 0.0):
            raise ValueError("channel_edges_um must be strictly ascending")
        n_ch = edges.size - 1

        arrays: dict[str, NDArray[np.float64]] = {}
        for name in (
            "channel_centers_um",
            "planet_rate_hz",
            "leak_rate_hz",
            "sky_rate_hz",
            "dark_rate_hz",
        ):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.ndim != 1:
                raise ValueError(f"{name} must be 1-D, got shape {values.shape}")
            if values.size != n_ch:
                raise ValueError(
                    f"{name} has length {values.size}, expected {n_ch} to match "
                    f"{edges.size} channel edges"
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be finite")
            if name != "channel_centers_um" and np.any(values < 0.0):
                raise ValueError(f"{name} must be non-negative")
            arrays[name] = values

        object.__setattr__(self, "channel_edges_um", edges)
        for name, values in arrays.items():
            object.__setattr__(self, name, values)

    @property
    def n_channels(self) -> int:
        """Number of spectrograph channels."""
        return int(self.channel_centers_um.size)

    def total_rate_hz(self) -> NDArray[np.float64]:
        """Per-channel total detected rate ``P + L + sky + dark`` (Hz)."""
        return (
            self.planet_rate_hz
            + self.leak_rate_hz
            + self.sky_rate_hz
            + self.dark_rate_hz
        )


@dataclass(frozen=True, eq=False)
class CcfBudget:
    """Matched-filter CCF signal, noise decomposition, and diagnostics.

    Attributes
    ----------
    ccf_snr:
        ``S / sqrt(V_ph + V_sys)``.
    ccf_snr_photon_limited:
        ``S / sqrt(V_ph)``: the same detection with the null instability off.
    signal_counts:
        Matched-filter signal ``S = t sum_i w_i Ptilde_i`` (counts).
    photon_var_counts, systematic_var_counts:
        The two noise variances (counts^2).
    systematic_ccf_leakage:
        ``B_w = sum_i w_i Ltilde_i`` (Hz), the template projection of the
        filtered stellar leak. This, not the raw band leak, sets the systematic
        amplitude.
    self_subtraction_factor:
        **SNR-retention ratio, not a flux ratio**:
        ``sqrt(sum_i Ptilde_i^2 / v_i) / sqrt(sum_i P_i^2 / v_i)`` over the
        unmasked channels. Non-negative by construction, and exactly 1.0 when
        continuum filtering is bypassed. It answers "what fraction of the
        achievable photon-limited SNR survives the continuum filter", which is
        the question a filter width is actually traded against. A weighted flux
        ratio would not do: it is unbounded above and changes sign for
        absorption-dominated templates, where the filtered template is negative.
    n_effective_lines:
        Participation ratio ``(sum_i |w_i|)^2 / sum_i w_i^2`` of the weights.
    n_channels_used:
        Channels with a non-zero weight budget (unmasked and ``v_i > 0``).
    template_weights:
        ``w_i``, zero where masked.
    """

    ccf_snr: float
    ccf_snr_photon_limited: float
    signal_counts: float
    photon_var_counts: float
    systematic_var_counts: float
    systematic_ccf_leakage: float
    self_subtraction_factor: float
    n_effective_lines: float
    n_channels_used: int
    template_weights: NDArray[np.float64]


def build_channel_edges(
    lambda_min_um: float, lambda_max_um: float, resolving_power: float
) -> NDArray[np.float64]:
    """Constant-resolving-power channel edges, one channel per resolution element.

    Thin wrapper on `nullsim.physics.spectra.constant_r_grid`: the returned array
    is a set of ``n_ch + 1`` edges spaced by ``d ln lambda = 1 / R``, so channel
    ``i`` spans one resolution element at ``R = resolving_power``.
    """
    return constant_r_grid(lambda_min_um, lambda_max_um, resolving_power)


def channelize(
    highres_wavelength_um: NDArray[np.float64],
    rate_density_hz_per_um: NDArray[np.float64],
    channel_edges_um: NDArray[np.float64],
    resolving_power_inst: float,
    *,
    r_in: float = math.inf,
) -> NDArray[np.float64]:
    """Degrade a high-resolution rate density to a channel rate in Hz per channel.

    Convolves with the instrument line-spread function *first* (on the
    log-uniform analysis grid, where a Gaussian LSF has a sample-constant width),
    then bin-averages onto ``channel_edges_um`` and multiplies by the channel
    widths. That order matters: `resample_flux_conserving` already applies a
    boxcar average over each bin, so resampling first would double-count part of
    the smoothing.

    ``r_in`` is the resolving power the input has already been broadened to
    (``inf`` for an unbroadened template); only the quadrature difference is
    applied.
    """
    wavelength = np.asarray(highres_wavelength_um, dtype=np.float64)
    density = np.asarray(rate_density_hz_per_um, dtype=np.float64)
    edges = np.asarray(channel_edges_um, dtype=np.float64)
    if wavelength.ndim != 1 or density.ndim != 1 or edges.ndim != 1:
        raise ValueError(
            "highres_wavelength_um, rate_density_hz_per_um, and channel_edges_um "
            "must be 1-D"
        )
    if wavelength.shape != density.shape:
        raise ValueError(
            "highres_wavelength_um and rate_density_hz_per_um must have the same "
            f"shape, got {wavelength.shape} and {density.shape}"
        )
    if edges.size < 2:
        raise ValueError("channel_edges_um needs at least 2 edges")

    smoothed = convolve_to_resolving_power(
        wavelength, density, resolving_power_inst, r_in=r_in
    )
    bin_means = resample_flux_conserving(wavelength, smoothed, edges)
    return bin_means * np.diff(edges)


def continuum_filter(
    values: NDArray[np.float64], width_channels: int
) -> NDArray[np.float64]:
    """High-pass a channel spectrum by subtracting a running median.

    ``values - median_filter(values, size=width, mode="nearest")`` with ``width``
    forced odd and at least 3 channels. The running median tracks the smooth
    stellar-leak continuum without being dragged around by the narrow planet
    lines, which is exactly the asymmetry HDC exploits.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"values must be 1-D, got shape {array.shape}")
    width = int(width_channels)
    if width < 3:
        width = 3
    if width % 2 == 0:
        width += 1
    return array - median_filter(array, size=width, mode="nearest")


def _filtered(
    values: NDArray[np.float64], width_channels: int
) -> NDArray[np.float64]:
    """`continuum_filter`, with ``width_channels <= 0`` meaning "no filtering"."""
    if int(width_channels) <= 0:
        return np.asarray(values, dtype=np.float64)
    return continuum_filter(values, width_channels)


def _telluric_mask(
    n_channels: int,
    telluric_transmission_per_channel: NDArray[np.float64] | None,
    telluric_mask_threshold: float,
) -> NDArray[np.bool_]:
    """Boolean keep-mask: channels whose telluric transmission clears threshold."""
    if telluric_transmission_per_channel is None:
        return np.ones(n_channels, dtype=bool)
    transmission = np.asarray(telluric_transmission_per_channel, dtype=np.float64)
    if transmission.ndim != 1 or transmission.size != n_channels:
        raise ValueError(
            "telluric_transmission_per_channel must be 1-D with length "
            f"{n_channels}, got shape {transmission.shape}"
        )
    if not np.all(np.isfinite(transmission)):
        raise ValueError("telluric_transmission_per_channel must be finite")
    return transmission >= float(telluric_mask_threshold)


def _masked_ratio(
    numerator: NDArray[np.float64],
    denominator: NDArray[np.float64],
    usable: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """``numerator / denominator`` where ``usable``, zero elsewhere."""
    out = np.zeros(numerator.shape, dtype=np.float64)
    np.divide(numerator, denominator, out=out, where=usable)
    return out


_GRID_RTOL = 1e-9


def _require_same_grid(
    trial: ChannelSpectra, reference: ChannelSpectra, rv_kms: float
) -> None:
    """Raise unless ``trial`` sits on the same wavelength grid as ``reference``.

    A trial template with the right length but a different grid would be
    correlated against the data index by index, silently producing a wrong
    answer instead of an error.
    """
    if trial.n_channels != reference.n_channels:
        raise ValueError(
            f"trial template at rv={rv_kms} has {trial.n_channels} channels, "
            f"expected {reference.n_channels} to match data_channels"
        )
    for name in ("channel_edges_um", "channel_centers_um"):
        trial_grid = getattr(trial, name)
        reference_grid = getattr(reference, name)
        if not np.allclose(trial_grid, reference_grid, rtol=_GRID_RTOL, atol=0.0):
            worst = int(np.argmax(np.abs(trial_grid - reference_grid)))
            raise ValueError(
                f"trial template at rv={rv_kms} has a different {name} than "
                f"data_channels: element {worst} is {trial_grid[worst]!r} vs "
                f"{reference_grid[worst]!r}. The trial closure must rebuild the "
                "template on the data's channel grid."
            )


def _weights(
    template_hz: NDArray[np.float64],
    variance_rate_hz: NDArray[np.float64],
    keep: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Matched-filter weights ``w = m * Ptilde / v``, zero on masked/empty channels."""
    usable = keep & (variance_rate_hz > 0.0)
    return _masked_ratio(template_hz, variance_rate_hz, usable), usable


def matched_filter_ccf(
    channels: ChannelSpectra,
    *,
    integration_time_s: float,
    eps_systematic: float,
    servo_bw_hz: float,
    with_fringe_tracking: bool,
    continuum_filter_width_channels: int,
    telluric_transmission_per_channel: NDArray[np.float64] | None,
    telluric_mask_threshold: float,
) -> CcfBudget:
    """Matched-filter CCF signal-to-noise for one channelized observation.

    See the module docstring for the noise model. ``continuum_filter_width_channels
    <= 0`` bypasses continuum filtering entirely (the template is used raw), which
    is the right setting for a deliberately broadband comparison.

    ``eps_systematic`` and ``servo_bw_hz`` carry the same meaning as in
    ``nullsim.stages.performance.standard``: a fractional null-instability
    amplitude on the leaked starlight, and the fringe-tracking servo bandwidth
    over which it averages down when ``with_fringe_tracking`` is true.
    """
    if not isinstance(channels, ChannelSpectra):
        raise TypeError(f"channels must be a ChannelSpectra, got {type(channels)!r}")
    integration_time_s = float(integration_time_s)
    eps_systematic = float(eps_systematic)
    servo_bw_hz = float(servo_bw_hz)
    if not math.isfinite(integration_time_s) or integration_time_s < 0.0:
        raise ValueError("integration_time_s must be finite and non-negative")
    if not math.isfinite(eps_systematic) or eps_systematic < 0.0:
        raise ValueError("eps_systematic must be finite and non-negative")
    if with_fringe_tracking and not (servo_bw_hz > 0.0 and math.isfinite(servo_bw_hz)):
        raise ValueError(
            "servo_bw_hz must be finite and positive when fringe tracking is on, "
            f"got {servo_bw_hz}"
        )

    n_ch = channels.n_channels
    keep = _telluric_mask(
        n_ch, telluric_transmission_per_channel, telluric_mask_threshold
    )
    variance_rate = channels.total_rate_hz()
    planet = channels.planet_rate_hz
    planet_filtered = _filtered(planet, continuum_filter_width_channels)
    leak_filtered = _filtered(channels.leak_rate_hz, continuum_filter_width_channels)

    weights, usable = _weights(planet_filtered, variance_rate, keep)

    signal_counts = float(np.sum(weights * planet_filtered) * integration_time_s)
    photon_var_counts = float(
        np.sum(weights * weights * variance_rate) * integration_time_s
    )
    systematic_ccf_leakage = float(np.sum(weights * leak_filtered))

    if with_fringe_tracking:
        systematic_var_counts = (
            eps_systematic
            * eps_systematic
            * systematic_ccf_leakage
            * systematic_ccf_leakage
            * integration_time_s
            / (2.0 * servo_bw_hz)
        )
    else:
        systematic_var_counts = (
            eps_systematic * systematic_ccf_leakage * integration_time_s
        ) ** 2

    total_var = photon_var_counts + systematic_var_counts
    ccf_snr = (
        signal_counts / math.sqrt(total_var)
        if total_var > 0.0 and signal_counts > 0.0
        else 0.0
    )
    ccf_snr_photon_limited = (
        signal_counts / math.sqrt(photon_var_counts)
        if photon_var_counts > 0.0 and signal_counts > 0.0
        else 0.0
    )

    # SNR-retention ratio, not a flux ratio: the photon-limited matched-filter
    # SNR of the filtered template over that of the unfiltered one, on the same
    # unmasked channels and the same noise budget.
    snr_sq_filtered = float(
        _masked_ratio(planet_filtered * planet_filtered, variance_rate, usable).sum()
    )
    snr_sq_unfiltered = float(
        _masked_ratio(planet * planet, variance_rate, usable).sum()
    )
    self_subtraction_factor = (
        math.sqrt(snr_sq_filtered / snr_sq_unfiltered)
        if snr_sq_unfiltered > 0.0
        else 0.0
    )

    abs_sum = float(np.sum(np.abs(weights)))
    sq_sum = float(np.sum(weights * weights))
    n_effective_lines = abs_sum * abs_sum / sq_sum if sq_sum > 0.0 else 0.0

    return CcfBudget(
        ccf_snr=ccf_snr,
        ccf_snr_photon_limited=ccf_snr_photon_limited,
        signal_counts=signal_counts,
        photon_var_counts=photon_var_counts,
        systematic_var_counts=float(systematic_var_counts),
        systematic_ccf_leakage=systematic_ccf_leakage,
        self_subtraction_factor=self_subtraction_factor,
        n_effective_lines=n_effective_lines,
        n_channels_used=int(np.count_nonzero(usable)),
        template_weights=weights,
    )


def snr_vs_resolving_power(
    build_channels_fn: Callable[[float], ChannelSpectra],
    r_values: Sequence[float] | NDArray[np.float64],
    **mf_kwargs: object,
) -> NDArray[np.float64]:
    """CCF SNR as a function of spectrograph resolving power.

    ``build_channels_fn(R)`` must rebuild the `ChannelSpectra` at resolving power
    ``R`` (new channel grid, re-convolved and re-channelized rates); the
    remaining keyword arguments are forwarded to `matched_filter_ccf`.
    """
    r_array = np.atleast_1d(np.asarray(r_values, dtype=np.float64))
    if r_array.ndim != 1:
        raise ValueError(f"r_values must be 1-D, got shape {r_array.shape}")
    return np.array(
        [
            matched_filter_ccf(build_channels_fn(float(r)), **mf_kwargs).ccf_snr  # type: ignore[arg-type]
            for r in r_array
        ],
        dtype=np.float64,
    )


def ccf_vs_rv(
    channels_at_trial_rv_fn: Callable[[float], ChannelSpectra],
    rv_grid_kms: Sequence[float] | NDArray[np.float64],
    *,
    data_channels: ChannelSpectra,
    continuum_filter_width_channels: int,
    telluric_transmission_per_channel: NDArray[np.float64] | None = None,
    telluric_mask_threshold: float = 0.0,
    integration_time_s: float = 1.0,
    **_ignored: object,
) -> NDArray[np.float64]:
    """Cross-correlation of a fixed observed spectrum against shifted templates.

    ``data_channels`` is the observation: planet at its true radial velocity plus
    leaked starlight, sky, and dark. ``channels_at_trial_rv_fn(rv)`` supplies the
    template at each trial radial velocity through its ``planet_rate_hz`` field;
    it must return channels on the *same* wavelength grid as ``data_channels``.
    The returned statistic at trial velocity ``rv`` is the template-normalized
    matched-filter score

    ``CCF(rv) = sqrt(t) * sum_i w_i(rv) Dtilde_i / sqrt(sum_i w_i(rv)^2 v_i)``,
    ``w_i(rv) = m_i Ttilde_i(rv) / v_i``,

    with ``Dtilde`` the continuum-filtered observed dark-port rate
    (planet + leak) and ``v_i`` the observed per-channel variance rate.

    The normalization is not cosmetic. The raw numerator ``T^T V^-1 D`` grows
    with the template norm, so a trial velocity that happens to park a strong
    template line on a weak data line can outscore the true velocity whenever the
    template's own strong lines have been removed by the telluric mask or pushed
    past a band edge. Dividing by ``sqrt(T^T V^-1 T)`` makes the statistic a
    Cauchy-Schwarz cosine in the ``V^-1`` metric, which is maximized only when
    the template matches the data, so the peak sits at the true radial velocity.

    Unrecognized keyword arguments are ignored so that the same ``mf_kwargs``
    mapping can be forwarded to both this function and `matched_filter_ccf`.
    """
    rv_array = np.atleast_1d(np.asarray(rv_grid_kms, dtype=np.float64))
    if rv_array.ndim != 1:
        raise ValueError(f"rv_grid_kms must be 1-D, got shape {rv_array.shape}")
    if not np.all(np.isfinite(rv_array)):
        raise ValueError("rv_grid_kms must be finite")
    integration_time_s = float(integration_time_s)
    if not math.isfinite(integration_time_s) or integration_time_s < 0.0:
        raise ValueError("integration_time_s must be finite and non-negative")

    n_ch = data_channels.n_channels
    keep = _telluric_mask(
        n_ch, telluric_transmission_per_channel, telluric_mask_threshold
    )
    variance_rate = data_channels.total_rate_hz()
    observed = data_channels.planet_rate_hz + data_channels.leak_rate_hz
    observed_filtered = _filtered(observed, continuum_filter_width_channels)
    sqrt_time = math.sqrt(integration_time_s)

    values = np.empty(rv_array.size, dtype=np.float64)
    for index, rv in enumerate(rv_array):
        trial = channels_at_trial_rv_fn(float(rv))
        _require_same_grid(trial, data_channels, rv)
        template = _filtered(trial.planet_rate_hz, continuum_filter_width_channels)
        weights, _ = _weights(template, variance_rate, keep)
        template_norm = math.sqrt(float(np.sum(weights * weights * variance_rate)))
        if template_norm <= 0.0:
            values[index] = 0.0
            continue
        values[index] = (
            sqrt_time * float(np.sum(weights * observed_filtered)) / template_norm
        )
    return values
