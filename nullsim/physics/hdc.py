"""Photon-weighted cross-correlation (CCF) budget for high dispersion coronagraphy.

High dispersion coronagraphy (HDC) buys contrast by moving the planet detection
from a broadband photometric comparison to a spectral one: the planet signal is
a forest of narrow molecular lines, while the leaked starlight that dominates a
dark port is spectrally smooth. Continuum filtering removes the smooth part; the
surviving line forest is then correlated against a template. The framing and the
sqrt(N_lines) scaling follow Wang, Mawet, Ruane, Hu & Benneke 2017, AJ 153, 183
("Observing Exoplanets with High Dispersion Coronagraphy. I.").

What this statistic is
----------------------
A **photon-weighted CCF with a fixed template**: the weights are
``w_i = m_i Ptilde_i / v_i``, set by the template and the photon variance alone.
It is *not* the optimal filter once a systematic is present. Under the rank-one
systematic covariance the optimum is the Sherman-Morrison filter
``C^-1 Ptilde`` with ``C = diag(v t) + K Ltilde Ltilde^T``, which actively
projects out the leak direction and can beat this statistic by an order of
magnitude in the systematics-dominated regime. That optimum is documented as an
upper bound only, not implemented: it assumes the leak spectrum is known
exactly, and buying a 10x paper gain against an assumption the instrument
cannot deliver is not the trade this module makes. See the derivation note.

Noise model
-----------
Work on ``n_ch`` spectrograph channels, each carrying rates in Hz: planet ``P``,
leaked starlight ``L``, sky ``S_sky``, and detector dark ``D``. Let ``m_i`` be a
telluric mask (1 where the atmospheric transmission clears a threshold), let
``H`` be the linear continuum high-pass (`continuum_filter`: identity minus a
normalized boxcar smoother), and write ``Ptilde = H P``, ``Ltilde = H L``.

- Per-channel noise-variance rate ``v_i = P_i + L_i + S_sky,i + D_i`` (Hz).
  Poisson statistics make this both the mean and the variance rate.
- Weights for the fixed-template filter: ``w_i = m_i * Ptilde_i / v_i``.
- The statistic is ``X = w . (H N)`` on observed counts ``N``, so
  ``S = t * sum_i w_i Ptilde_i`` (counts).
- Photon variance ``V_ph = t * w^T H diag(v) H^T w = t * sum_i u_i^2 v_i`` with
  ``u = H^T w`` (counts^2). The filter correlates neighbouring channels, so the
  raw diagonal ``t sum_i w_i^2 v_i`` is *not* the variance of this statistic --
  using it understates the noise by up to a factor of a few. ``H`` is linear
  precisely so this propagation is exact and closed-form; see
  `continuum_filter` for why the running median it replaced could not be.
  Only when filtering is bypassed does ``u = w`` and ``V_ph`` collapse to ``S``,
  recovering the textbook ``S/sqrt(S+B)`` for a flat template.
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

- ``self_subtraction_factor``: the photon-limited SNR of the filtered statistic
  over ``sqrt(t sum_i P_i^2/v_i)``, the unfiltered one, on the same unmasked
  channels. It now prices the whole cost of filtering -- signal lost to
  self-subtraction *and* noise gained from the correlation ``H`` introduces.
  Non-negative, exactly 1.0 when filtering is bypassed, usually below 1, but it
  can exceed 1 when absorption structure on a bright continuum turns into large
  signed excursions, which is real and must not be clamped.
- ``n_effective_channels = (sum_i |w_i|)^2 / sum_i w_i^2``: the participation
  ratio of the weights, i.e. how many channels the statistic actually leans on.
  (Exposed as ``n_effective_lines`` too, a deprecated alias: the ratio counts
  channels, and one line spans several of them at any oversampling.)

Detection mode
--------------
`ccf_contrast_at_target_snr` runs the same model backwards, per section 6 of
``docs/superpowers/specs/2026-08-07-hdc-ccf-design.md``: given a planet template
at unit contrast, solve ``SNR(c) = n`` for the contrast ``c``. Because the
planet enters the signal linearly and the variance affinely, and ``V_sys`` does
not depend on ``c`` in either regime, the condition is a quadratic with a
closed-form positive root -- no iteration, so a whole contrast curve costs one
pass per separation.

The weights are frozen at ``c = 0``, and the resulting error is

```text
SNR^2_true / SNR^2_frozen = 1 - 2 c Delta + O(c^2),
Delta = C2'/A - eta_ph E/B2 - eta_sys D/J0,
eta_sys = V_sys / (t B2 + V_sys),   eta_ph = 1 - eta_sys.
```

``Delta`` interpolates between a photon-limited coefficient and a systematics
one. The photon-limited end is small but, with a linear high-pass, no longer
identically zero. Both ends vanish for a flat template against a flat leak, and
when ``c P1_i / v0_i`` is channel-independent as well (flat spectra with
unstructured sky/dark) the frozen statistic is exact to all orders. So the bias
is set by spectral structure gated by the systematic share, not by the presence
of a systematic alone. See the function docstring for the measured behavior.

Pure numpy/scipy: no TOML, ``SimulationState``, or stage imports.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter1d

from nullsim.physics.spectra import (
    constant_r_grid,
    convolve_to_resolving_power,
    resample_flux_conserving,
)

__all__ = [
    "CcfBudget",
    "CcfContrastSolution",
    "ChannelSpectra",
    "build_channel_edges",
    "ccf_contrast_at_target_snr",
    "ccf_vs_rv",
    "channelize",
    "continuum_filter",
    "continuum_filter_adjoint",
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
    """Photon-weighted CCF signal, noise decomposition, and diagnostics.

    Attributes
    ----------
    ccf_snr:
        ``S / sqrt(V_ph + V_sys)``.
    ccf_snr_photon_limited:
        ``S / sqrt(V_ph)``: the same detection with the null instability off.
    signal_counts:
        Filter signal ``S = t sum_i w_i Ptilde_i`` (counts).
    photon_var_counts, systematic_var_counts:
        The two noise variances (counts^2).
    systematic_ccf_leakage:
        ``B_w = sum_i w_i Ltilde_i`` (Hz), the template projection of the
        filtered stellar leak. This, not the raw band leak, sets the systematic
        amplitude.
    self_subtraction_factor:
        **SNR-retention ratio, not a flux ratio**:
        ``sqrt(sum_i Ptilde_i^2 / v_i) / sqrt(sum_i P_i^2 / v_i)`` over the
        unmasked channels. It answers "what fraction of the achievable
        photon-limited SNR survives the continuum filter", which is the question
        a filter width is actually traded against.

        Non-negative by construction. Exactly 1.0 when continuum filtering is
        bypassed (``width <= 0``), unconditionally. 0.0 by convention when
        filtering is active and the template carries no signal at all.

        In practice at or below 1. The above-1 values the running median used
        to produce are gone: they came from dividing a filtered signal by an
        unfiltered diagonal variance, and pricing the filter's noise penalty
        removes the apparent gain (the pinned alternating case went from a
        reported 1.111 to an honest 0.637). A scan of 3000 random templates
        found nothing above 0.85. **No upper bound is proven, though**, so do
        not clamp it, range-validate it into [0, 1], or render it as a
        percentage capped at 100%.

        A weighted flux ratio would not do in its place: it changes sign for
        absorption-dominated templates, where the filtered template is negative,
        and its denominator can pass through zero.
    n_effective_channels:
        Participation ratio ``(sum_i |w_i|)^2 / sum_i w_i^2`` of the weights:
        how many channels the statistic actually leans on. Equals ``n`` for
        ``n`` equally weighted channels and 1 when a single channel dominates.
        Exposed as ``n_effective_lines`` too -- deprecated alias, kept so
        existing payload consumers keep working; prefer the channel name, since
        the ratio counts channels and one line spans several of them.
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
    n_effective_channels: float
    n_channels_used: int
    template_weights: NDArray[np.float64]

    @property
    def n_effective_lines(self) -> float:
        """Deprecated alias for `n_effective_channels`.

        The ratio is a participation count over *channels*; at any oversampling
        a single spectral line occupies several of them, so the old name
        overstated the line count. Kept for payload compatibility.
        """
        return self.n_effective_channels


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


def _odd_width(width_channels: int) -> int:
    """Coerce a filter width to odd and at least 3 channels."""
    width = int(width_channels)
    if width < 3:
        width = 3
    if width % 2 == 0:
        width += 1
    return width


def _boxcar(values: NDArray[np.float64], width: int) -> NDArray[np.float64]:
    """Zero-padded boxcar average: the symmetric banded operator ``K``.

    ``K_ij = 1/width`` for ``|i - j| <= width // 2`` and zero otherwise,
    truncated to the array. Restricting a symmetric Toeplitz matrix to a
    principal submatrix leaves it symmetric, so ``K = K^T`` exactly -- which is
    what makes the adjoint of the smoother cheap to write down.
    """
    return uniform_filter1d(
        values, size=width, mode="constant", cval=0.0, axis=-1
    )


def _smoother_normalization(n_channels: int, width: int) -> NDArray[np.float64]:
    """``c = K 1``: the in-array fraction of each window. 1 inside, <1 at edges."""
    return _boxcar(np.ones(n_channels, dtype=np.float64), width)


def _smooth(values: NDArray[np.float64], width: int) -> NDArray[np.float64]:
    """Apply the low-pass smoother ``S = diag(1/c) K``.

    Normalizing by the in-array window fraction makes ``S 1 = 1`` exactly, at
    the edges as well as the interior, so the high-pass ``H = I - S``
    annihilates a constant everywhere. A flat stellar continuum is therefore
    removed exactly rather than leaving an edge residual.
    """
    return _boxcar(values, width) / _smoother_normalization(values.shape[-1], width)


def _smooth_adjoint(values: NDArray[np.float64], width: int) -> NDArray[np.float64]:
    """Apply ``S^T = K^T diag(1/c) = K diag(1/c)``, using ``K = K^T``."""
    return _boxcar(
        values / _smoother_normalization(values.shape[-1], width), width
    )


def continuum_filter(
    values: NDArray[np.float64], width_channels: int
) -> NDArray[np.float64]:
    """High-pass a channel spectrum by subtracting a boxcar running mean.

    ``H x = x - S x`` with ``S`` the width-normalized boxcar smoother, ``width``
    forced odd and at least 3 channels. The smoother tracks the broad stellar
    continuum; subtracting it leaves the narrow planet lines, which is the
    asymmetry HDC exploits.

    **Why a linear smoother and not a running median.** A median rejects the
    lines themselves, so it estimates the continuum with less self-subtraction,
    and that is why it was the first choice here. But it is nonlinear, and the
    statistic built on top of it is a weighted sum of *filtered* counts whose
    covariance then has no closed form. Pairing a median-filtered signal model
    with a raw diagonal photon variance understates the noise badly -- 3.6x in
    variance, 1.9x in SNR, on the alternating-spectrum case pinned in
    ``tests/physics/test_hdc.py`` -- because the median of a window makes each
    filtered channel a difference of raw counts and correlates neighbours.
    A linear ``H`` costs more self-subtraction (the lines do drag a mean) but
    propagates exactly: ``Cov(H N) = H diag(v t) H^T``, so
    ``Var(w . H N) = t sum_i (H^T w)_i^2 v_i``. The self-subtraction is then
    honestly reported by ``self_subtraction_factor`` instead of being hidden in
    an unmodelled covariance. Boxcar rather than Gaussian because its support is
    exactly ``width`` channels, so the banded operator and its adjoint are exact
    with no kernel truncation to reason about.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"values must be 1-D, got shape {array.shape}")
    width = _odd_width(width_channels)
    return array - _smooth(array, width)


def continuum_filter_adjoint(
    values: NDArray[np.float64], width_channels: int
) -> NDArray[np.float64]:
    """Apply ``H^T``, the adjoint of `continuum_filter`.

    Needed to propagate photon noise through the filter: for independent
    per-channel counts with variance ``v_i t``, the statistic ``w . H N`` has
    variance ``t sum_i (H^T w)_i^2 v_i``. Same signature convention as
    `continuum_filter`.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"values must be 1-D, got shape {array.shape}")
    width = _odd_width(width_channels)
    return array - _smooth_adjoint(array, width)


def _filtered(
    values: NDArray[np.float64], width_channels: int
) -> NDArray[np.float64]:
    """`continuum_filter`, with ``width_channels <= 0`` meaning "no filtering"."""
    if int(width_channels) <= 0:
        return np.asarray(values, dtype=np.float64)
    return continuum_filter(values, width_channels)


def _filtered_adjoint(
    values: NDArray[np.float64], width_channels: int
) -> NDArray[np.float64]:
    """`continuum_filter_adjoint`, with ``width_channels <= 0`` the identity.

    Bypassed filtering means ``H = I``, so ``H^T = I`` too and the propagated
    variance collapses back to the plain diagonal ``sum_i w_i^2 v_i``.
    """
    if int(width_channels) <= 0:
        return np.asarray(values, dtype=np.float64)
    return continuum_filter_adjoint(values, width_channels)


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


def _validate_noise_params(
    integration_time_s: float,
    eps_systematic: float,
    servo_bw_hz: float,
    with_fringe_tracking: bool,
) -> tuple[float, float, float]:
    """Coerce and range-check the shared noise-model scalars.

    Shared by `matched_filter_ccf` and `ccf_contrast_at_target_snr` so the two
    entry points cannot drift apart on what they accept. NaN must be rejected
    explicitly: it slips through a bare ``<= 0`` test and would propagate into a
    silently zero result.
    """
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
    return integration_time_s, eps_systematic, servo_bw_hz


def _systematic_variance_counts(
    leakage_projection_hz: float,
    integration_time_s: float,
    eps_systematic: float,
    servo_bw_hz: float,
    with_fringe_tracking: bool,
) -> float:
    """``V_sys`` in counts^2 from the template-projected leak ``B_w``.

    The two regimes of
    ``nullsim.stages.performance.standard._systematic_variance`` with ``B_w``
    substituted for the raw dark-port leakage rate.
    """
    if with_fringe_tracking:
        return (
            eps_systematic
            * eps_systematic
            * leakage_projection_hz
            * leakage_projection_hz
            * integration_time_s
            / (2.0 * servo_bw_hz)
        )
    return (eps_systematic * leakage_projection_hz * integration_time_s) ** 2


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
    """Filter weights ``w = m * Ptilde / v``, zero on masked/empty channels."""
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
    """Photon-weighted CCF signal-to-noise for one channelized observation.

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
    integration_time_s, eps_systematic, servo_bw_hz = _validate_noise_params(
        integration_time_s, eps_systematic, servo_bw_hz, with_fringe_tracking
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
    # u = H^T w. The statistic is w . (H N) for independent Poisson counts N, so
    # its variance is t sum_i u_i^2 v_i, NOT the raw diagonal t sum_i w_i^2 v_i:
    # the filter correlates neighbouring channels.
    noise_weights = _filtered_adjoint(weights, continuum_filter_width_channels)

    signal_counts = float(np.sum(weights * planet_filtered) * integration_time_s)
    photon_var_counts = float(
        np.sum(noise_weights * noise_weights * variance_rate) * integration_time_s
    )
    systematic_ccf_leakage = float(np.sum(weights * leak_filtered))

    systematic_var_counts = _systematic_variance_counts(
        systematic_ccf_leakage,
        integration_time_s,
        eps_systematic,
        servo_bw_hz,
        with_fringe_tracking,
    )

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

    # SNR-retention ratio, not a flux ratio: the photon-limited SNR of the
    # filtered statistic over that of the unfiltered one, on the same unmasked
    # channels and the same noise budget. The numerator now carries the
    # H-propagated variance, so this measures the full cost of filtering --
    # signal lost to self-subtraction *and* noise gained from the correlation
    # the filter introduces. See the CcfBudget docstring for why it can exceed 1.
    if int(continuum_filter_width_channels) <= 0:
        # Filtering bypassed: H = I, so nothing was lost. Stated unconditionally
        # so an all-zero template gives 1.0 rather than falling into 0/0.
        self_subtraction_factor = 1.0
    else:
        snr_sq_unfiltered = float(
            _masked_ratio(planet * planet, variance_rate, usable).sum()
            * integration_time_s
        )
        # No template signal at all to retain: report 0.0 by convention.
        self_subtraction_factor = (
            ccf_snr_photon_limited / math.sqrt(snr_sq_unfiltered)
            if snr_sq_unfiltered > 0.0
            else 0.0
        )

    abs_sum = float(np.sum(np.abs(weights)))
    sq_sum = float(np.sum(weights * weights))
    n_effective_channels = abs_sum * abs_sum / sq_sum if sq_sum > 0.0 else 0.0

    return CcfBudget(
        ccf_snr=ccf_snr,
        ccf_snr_photon_limited=ccf_snr_photon_limited,
        signal_counts=signal_counts,
        photon_var_counts=photon_var_counts,
        systematic_var_counts=float(systematic_var_counts),
        systematic_ccf_leakage=systematic_ccf_leakage,
        self_subtraction_factor=self_subtraction_factor,
        n_effective_channels=n_effective_channels,
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
    """Whitened cross-correlation of a fixed observation against shifted templates.

    ``data_channels`` is the observation: planet at its true radial velocity plus
    leaked starlight, sky, and dark. ``channels_at_trial_rv_fn(rv)`` supplies the
    template at each trial radial velocity through its ``planet_rate_hz`` field;
    it must return channels on the *same* wavelength grid as ``data_channels``.

    The statistic is the generalized (whitened) matched filter for the filtered
    data ``y = H N``, whose covariance is ``C_y = t H diag(v) H^T``:

    ```text
    CCF(rv) = <Ttilde(rv), y>_{C_y^+} / sqrt(<Ttilde(rv), Ttilde(rv)>_{C_y^+}).
    ```

    Both ``Ttilde = H T`` and ``y`` lie in ``range(H) = range(C_y)``, so the
    pseudoinverse is well defined on them, and the score is a genuine cosine
    under one inner product. Cauchy-Schwarz then bounds it by
    ``sqrt(<y, y>_{C_y^+})`` with equality only when ``Ttilde(rv) ∝ y``, which
    puts the peak at the true radial velocity.

    **The pseudoinverse has a closed form here, so this stays O(n) per trial.**
    ``H = I - S`` with ``S`` row-stochastic and irreducible, so by
    Perron-Frobenius its unit eigenvalue is simple and ``null(H) = span{1}``
    exactly. Hence ``range(H^T) = 1-perp``, and maximizing
    ``(a.P)^2 / (t a^T diag(v) a)`` over ``a = H^T w`` in that subspace gives

    ```text
    a_i = m_i (T_i - lambda) / v_i,   lambda = sum_i m_i T_i/v_i / sum_i m_i/v_i,
    CCF(rv) = sqrt(t) * (a . R_obs) / sqrt(a^T diag(v) a),
    ```

    with ``R_obs`` the observed planet-plus-leak rate. No eigendecomposition and
    no dense matrix: ``a`` is the raw template inverse-variance weighted with a
    variance-weighted mean removed, which is precisely "blind to the constant
    mode and nothing else". A test pins this against the dense ``C_y^+`` form.

    **What changed and why.** This previously paired ``w = diag(v)^-1 Ttilde``
    weights with the ``C_y`` denominator ``sqrt(w^T H diag(v) H^T w)``. Those
    belong to different inner products, so the ratio was not a cosine and the
    Cauchy-Schwarz argument did not hold: on a ten-channel probe with a flat
    leak, a template shifted by eight channels scored 3.591 against 3.383 for
    the truth. Either half alone is self-consistent -- the old diagonal
    denominator was a valid cosine under ``diag(v)^-1`` -- but that normalizer
    is not the statistic's actual noise, and reporting a sigma-like score
    against a variance the data does not have is the same defect the covariance
    fix removed from `matched_filter_ccf`.

    **Caveat, and it is a real one.** Whitening by the filtered covariance
    necessarily undoes whatever suppression ``H`` applied, because the filter
    attenuated signal and noise together and the ratio is restored. The optimum
    is blind only to the constant mode, so unlike `matched_filter_ccf` -- which
    keeps its fixed-template weights on purpose, and with them the boxcar's
    suppression of smooth leak structure -- this scan is not protected against a
    non-constant leak continuum. That is the price of a peak guarantee, and it
    is why the two statistics differ: this one *locates* the radial velocity,
    `matched_filter_ccf` *scores* the detection. Do not read this score as the
    detection SNR.

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
    usable = keep & (variance_rate > 0.0)
    observed = data_channels.planet_rate_hz + data_channels.leak_rate_hz
    inverse_variance = _masked_ratio(np.ones(n_ch), variance_rate, usable)
    inverse_variance_total = float(inverse_variance.sum())
    sqrt_time = math.sqrt(integration_time_s)

    values = np.zeros(rv_array.size, dtype=np.float64)
    if inverse_variance_total <= 0.0:
        return values

    for index, rv in enumerate(rv_array):
        trial = channels_at_trial_rv_fn(float(rv))
        _require_same_grid(trial, data_channels, rv)
        template = trial.planet_rate_hz
        # Variance-weighted mean: the one direction H cannot see.
        offset = float(np.sum(inverse_variance * template)) / inverse_variance_total
        filter_vector = inverse_variance * (template - offset)
        filter_vector[~usable] = 0.0
        norm = math.sqrt(float(np.sum(filter_vector**2 * variance_rate)))
        if norm <= 0.0:
            continue
        values[index] = (
            sqrt_time * float(np.sum(filter_vector * observed)) / norm
        )
    return values


@dataclass(frozen=True, eq=False)
class CcfContrastSolution:
    """The band-integrated contrast a CCF observation reaches at a target SNR.

    Attributes
    ----------
    contrast_at_target_snr:
        The positive root ``c`` of ``SNR(c) = target_snr``. Dimensionless: the
        band-integrated planet/star contrast, in the same units as whatever
        normalization produced ``P1``.
    target_snr:
        The ``n`` that was solved for, echoed back.
    signal_coefficient:
        ``A = sum_i w_i Ptilde1_i`` (Hz per unit contrast). Analytically equal
        to ``B2`` (both are ``sum_i Ptilde1_i^2 / v0_i``), but computed by a
        different summation, so the two agree to rounding, not bit for bit.
    photon_var_zero_contrast:
        ``B2 = sum_i w_i^2 v0_i`` (Hz): the photon variance rate with no planet.
    planet_selfnoise_coefficient:
        ``C2 = sum_i w_i^2 P1_i`` (Hz per unit contrast): the planet's own shot
        noise, the term that makes the condition quadratic rather than linear.
    systematic_var_counts:
        ``V_sys`` (counts^2). Independent of ``c`` in both regimes, which is
        what keeps the root closed-form.
    template_weights:
        ``w_i`` frozen at zero contrast, zero where masked or ``v0_i = 0``.
    """

    contrast_at_target_snr: float
    target_snr: float
    signal_coefficient: float
    photon_var_zero_contrast: float
    planet_selfnoise_coefficient: float
    systematic_var_counts: float
    template_weights: NDArray[np.float64]


def _validate_rate_arrays(
    **named_arrays: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """Coerce per-channel rate arrays to float64 and check they agree in shape."""
    out: dict[str, NDArray[np.float64]] = {}
    length: int | None = None
    reference = ""
    for name, values in named_arrays.items():
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1:
            raise ValueError(f"{name} must be 1-D, got shape {array.shape}")
        if array.size < 1:
            raise ValueError(f"{name} must have at least one channel")
        if length is None:
            length, reference = array.size, name
        elif array.size != length:
            raise ValueError(
                f"{name} has length {array.size} but {reference} has length "
                f"{length}; all per-channel arrays must agree"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
        if np.any(array < 0.0):
            raise ValueError(f"{name} must be non-negative")
        out[name] = array
    return out


def ccf_contrast_at_target_snr(
    planet_rate_hz_per_channel_unit_contrast: NDArray[np.float64],
    leak_rate_hz_per_channel: NDArray[np.float64],
    sky_rate_hz_per_channel: NDArray[np.float64],
    dark_rate_hz_per_channel: NDArray[np.float64],
    *,
    integration_time_s: float,
    target_snr: float,
    eps_systematic: float,
    servo_bw_hz: float,
    with_fringe_tracking: bool,
    continuum_filter_width_channels: int,
    telluric_transmission_per_channel: NDArray[np.float64] | None,
    telluric_mask_threshold: float,
) -> CcfContrastSolution:
    """Solve ``SNR(c) = target_snr`` in closed form for the contrast ``c``.

    Detection mode, per section 6 of
    ``docs/superpowers/specs/2026-08-07-hdc-ccf-design.md``: instead of scoring
    a known planet, invert the matched filter to ask how faint a planet the
    same observation would still detect at ``target_snr``. That is what a CCF
    contrast curve is made of.

    The planet enters linearly in the contrast: the first argument is the
    per-channel planet rate at *unit* band-integrated contrast, so the actual
    planet rate is ``c * P1_i``. Writing ``v0_i = L_i + sky_i + dark_i`` for the
    planet-free variance rate, ``Ptilde1 = continuum_filter(P1)``,
    ``Ltilde = continuum_filter(L)``, and ``w_i = m_i Ptilde1_i / v0_i``,

    ```text
    A  = sum_i w_i Ptilde1_i     [Hz per unit contrast]
    B2 = sum_i w_i^2 v0_i        [Hz]
    C2 = sum_i w_i^2 P1_i        [Hz per unit contrast]
    B_w = sum_i w_i Ltilde_i     [Hz]

    SNR^2(c) = c^2 A^2 t^2 / (t B2 + c t C2 + V_sys)
    =>  (A^2 t^2) c^2 - (n^2 t C2) c - n^2 (t B2 + V_sys) = 0
    =>  c = [n^2 t C2 + sqrt(n^4 t^2 C2^2 + 4 A^2 t^2 n^2 (t B2 + V_sys))]
            / (2 A^2 t^2)
    ```

    ``V_sys`` follows the same two regimes as `matched_filter_ccf` and is
    independent of ``c``, which is what makes the root closed-form rather than
    an iteration.

    **Frozen-weights approximation, and where it degrades.** The weights are
    evaluated at ``c = 0``, so ``v0`` excludes the planet's own contribution to
    the *weighting* while ``C2`` still carries the planet shot noise into the
    denominator. Exact weights would depend on the answer, which is what the
    closed form buys its speed by avoiding. The accuracy is not uniform across
    the two noise regimes:

    - *Exactly photon-limited* (``V_sys = 0``): the residual slope is
      ``C2'/A - E/B2``, which the linear high-pass makes small but **no longer
      identically zero** -- ``H^T`` couples the weight perturbation into the
      variance, so the exact cancellation that held under a diagonal
      covariance does not survive. On the structured probe below it is
      ``-8.3e-05``, some 330x smaller than the systematics-driven slope and
      worth about a part in 1e5 on the returned contrast. Negative: the solver
      is very slightly *pessimistic* there.
    - *Nonzero systematic, structured spectra*: recomputed weights also change
      the leak projection ``B_w(c)``, and hence ``V_sys``, **generically at
      first order**. The general expansion is

      ```text
      SNR^2_true / SNR^2_frozen = 1 - 2 c Delta + O(c^2),
      Delta = C2'/A - eta_ph E/B2 - eta_sys D/J0,
      eta_sys = V_sys / (t B2 + V_sys),   eta_ph = 1 - eta_sys,
      C2' = sum_i Ptilde1_i^2 P1_i / v0_i^2,
      E   = sum_i u0_i (H^T g)_i v0_i,  g = Ptilde1 P1 / v0^2,
      D   = sum_i Ptilde1_i Ltilde_i P1_i / v0_i^2,   J0 = B_w.
      ```

      so the returned contrast is optimistic by ``c Delta`` in SNR. Without a
      filter ``u0 = w0``, ``C2' = E = C2``, and ``Delta`` collapses to the
      familiar ``eta_sys (C2/A - D/J0)``. ``eta_sys`` is just the systematic share of the zero-contrast
      variance: it tends to 1 in the systematics-dominated limit and falls
      smoothly to 0 as the budget becomes photon-limited, which is what makes
      the first-order term switch on gradually rather than at a threshold.
      The structural coefficient ``C2/A - D/H0`` is the other gate: it is
      identically zero for a flat template against a flat leak, and it grows
      with spectral structure, so the bias needs *both* a nonzero systematic
      and a structured spectrum. In the special case where the per-channel
      ratio ``c P1_i / v0_i`` is channel-independent (flat spectra *and*
      unstructured sky/dark, so that the ``(1 + c P1/v0)`` factors are common
      and cancel between numerator and denominator), the frozen and exact
      statistics coincide to *all* orders however large ``V_sys`` is. Flat
      ``P1`` and ``L`` alone do not suffice: structured sky or dark makes
      ``v0`` vary, the ratio with it, and a residual reappears at second order
      through the structural coefficient.

    Measured on an 80-channel structured probe with a width-11 filter,
    ``c Delta(eta_sys)`` reproduces the observed error to ~1% for
    ``eta_sys <= 0.9``: 0.09% optimistic at ``eta_sys = 0.17``, 1.3% at 0.89,
    5.1% at 0.99 (where higher-order terms cost the prediction ~6%).

    So the closed form is trustworthy when the planet is faint against the leak
    *and* the systematic share of the variance is small. Otherwise treat the
    value as optimistic at the percent level, or refine it with a single
    `matched_filter_ccf` evaluation at the returned contrast.

    Raises ``ValueError`` if ``A <= 0``: a template with no surviving line
    structure (a lineless planet spectrum, or one whose lines all fall in
    masked channels) has no contrast at which it reaches the target, and
    reporting ``inf`` would silently poison a contrast curve. Also raises
    ``ValueError`` -- not ``OverflowError`` -- if ``V_sys`` or the contrast root
    is not representable in float64, so a caller sweeping separation anchors has
    a single exception type to catch per anchor.
    """
    arrays = _validate_rate_arrays(
        planet_rate_hz_per_channel_unit_contrast=(
            planet_rate_hz_per_channel_unit_contrast
        ),
        leak_rate_hz_per_channel=leak_rate_hz_per_channel,
        sky_rate_hz_per_channel=sky_rate_hz_per_channel,
        dark_rate_hz_per_channel=dark_rate_hz_per_channel,
    )
    planet_unit = arrays["planet_rate_hz_per_channel_unit_contrast"]
    leak = arrays["leak_rate_hz_per_channel"]

    integration_time_s, eps_systematic, servo_bw_hz = _validate_noise_params(
        integration_time_s, eps_systematic, servo_bw_hz, with_fringe_tracking
    )
    if integration_time_s <= 0.0:
        raise ValueError("integration_time_s must be strictly positive to solve")
    target_snr = float(target_snr)
    if not math.isfinite(target_snr) or target_snr <= 0.0:
        raise ValueError(f"target_snr must be finite and positive, got {target_snr}")

    n_ch = planet_unit.size
    keep = _telluric_mask(
        n_ch, telluric_transmission_per_channel, telluric_mask_threshold
    )
    variance_rate_zero_contrast = (
        leak + arrays["sky_rate_hz_per_channel"] + arrays["dark_rate_hz_per_channel"]
    )
    planet_unit_filtered = _filtered(planet_unit, continuum_filter_width_channels)
    leak_filtered = _filtered(leak, continuum_filter_width_channels)

    weights, _usable = _weights(
        planet_unit_filtered, variance_rate_zero_contrast, keep
    )

    signal_coefficient = float(np.sum(weights * planet_unit_filtered))
    if signal_coefficient <= 0.0:
        raise ValueError(
            "no template signal survives continuum filtering and telluric "
            "masking (A <= 0), so no contrast reaches the target SNR"
        )
    # B2 and C2 are the two halves of the H-propagated photon variance rate,
    # v = v0 + c P1, split by their dependence on c. Using u = H^T w keeps the
    # variance exact; using w would revert to the diagonal approximation. The
    # split stays affine in c, so the target-SNR condition stays quadratic.
    noise_weights = _filtered_adjoint(weights, continuum_filter_width_channels)
    photon_var_zero_contrast = float(
        np.sum(noise_weights * noise_weights * variance_rate_zero_contrast)
    )
    planet_selfnoise_coefficient = float(
        np.sum(noise_weights * noise_weights * planet_unit)
    )
    leak_projection_hz = float(np.sum(weights * leak_filtered))
    # `matched_filter_ccf` lets an overflowing V_sys surface as it comes: the
    # no-fringe-tracking branch squares with `**`, which raises OverflowError,
    # while the fringe-tracking branch multiplies and quietly yields inf. This
    # entry point normalizes both to the ValueError it already documents for a
    # non-representable result, so a caller sweeping separation anchors has one
    # exception type to catch. Below the threshold nothing changes.
    try:
        systematic_var_counts = _systematic_variance_counts(
            leak_projection_hz,
            integration_time_s,
            eps_systematic,
            servo_bw_hz,
            with_fringe_tracking,
        )
    except OverflowError:
        systematic_var_counts = math.inf
    if not math.isfinite(systematic_var_counts):
        # Plain multiplication saturates to inf rather than raising, so this
        # is safe to evaluate even when the square of it is not.
        amplitude = eps_systematic * abs(leak_projection_hz) * integration_time_s
        raise ValueError(
            "systematic variance overflows float64; "
            f"eps*B_w*t ~ {amplitude!r} with eps={eps_systematic!r}, "
            f"B_w={leak_projection_hz!r}, t={integration_time_s!r}"
        )

    # Solve in scaled variables. Forming the dimensional coefficients directly
    # overflows or underflows well inside the range of physically valid inputs:
    # A ~ 1e150 already squares past the float64 ceiling, and A^2 t^2 flushes to
    # zero for rates around 1e-200. Substituting c = c0 x with the
    # photon-limited root c0 = n / sqrt(A t) reduces the quadratic to
    #
    #     x^2 - p x - q = 0,    p = c0 C2 / A,    q = B2/A + V_sys/(A t),
    #
    # which removes the A t and A^2 products. The reduced coefficients are not
    # themselves near unity -- p and q each still span the dynamic range of the
    # inputs, and q overflows on its own for a large V_sys over a small A. So q
    # is never materialized: only sqrt(q) is needed, and taking the square root
    # of each term first halves its exponent, which is what actually keeps the
    # arithmetic in range.
    sqrt_signal = math.sqrt(signal_coefficient)
    sqrt_time = math.sqrt(integration_time_s)
    contrast_scale = (target_snr / sqrt_signal) / sqrt_time
    reduced_linear = contrast_scale * (
        planet_selfnoise_coefficient / signal_coefficient
    )
    # sqrt(q) = hypot(sqrt(B2/A), sqrt(V_sys/(A t))), with the systematic term
    # grouped as sqrt(V_sys)/(sqrt(A) sqrt(t)): V_sys/A alone overflows for a
    # large systematic against a small signal coefficient.
    # sqrt(B2)/sqrt(A), never sqrt(B2/A): the covariance fix broke the A = B2
    # identity, so the ratio itself can overflow on inputs whose root is
    # perfectly representable (B2 ~ 1e126 against A ~ 1e-191).
    sqrt_reduced_constant = math.hypot(
        math.sqrt(photon_var_zero_contrast) / sqrt_signal,
        (math.sqrt(systematic_var_counts) / sqrt_signal) / sqrt_time,
    )
    # 0.5 (p + sqrt(p^2 + 4q)) == 0.5 p + hypot(p/2, sqrt(q)). Written this way
    # nothing is squared and 2*sqrt(q) is never formed, either of which would
    # overflow before the root does.
    reduced_root = 0.5 * reduced_linear + math.hypot(
        0.5 * reduced_linear, sqrt_reduced_constant
    )
    contrast = contrast_scale * reduced_root
    if not math.isfinite(contrast) or contrast <= 0.0:
        raise ValueError(
            f"the contrast solution is not representable (got {contrast!r}) for "
            f"A={signal_coefficient!r}, B2={photon_var_zero_contrast!r}, "
            f"C2={planet_selfnoise_coefficient!r}, V_sys={systematic_var_counts!r}, "
            f"t={integration_time_s!r}, n={target_snr!r}; rescale the input rates"
        )

    return CcfContrastSolution(
        contrast_at_target_snr=contrast,
        target_snr=target_snr,
        signal_coefficient=signal_coefficient,
        photon_var_zero_contrast=photon_var_zero_contrast,
        planet_selfnoise_coefficient=planet_selfnoise_coefficient,
        systematic_var_counts=systematic_var_counts,
        template_weights=weights,
    )
