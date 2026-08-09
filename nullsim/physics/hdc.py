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
It is *not* the optimal filter once a systematic is present. The optimum is the
generalized filter ``C_y^+ Ptilde`` for the covariance of the *filtered* data,

```text
C_y = t H diag(v) H^T + K Ltilde Ltilde^T,
```

whose rank-one systematic part Sherman-Morrison inverts in closed form. It
downweights the leak direction -- suppressing it by
``1/(1 + K Ltilde^T C_ph^+ Ltilde)``, so "projects out" is exact only in the
``K -> inf`` limit -- and can beat this statistic by an order of magnitude once
systematics dominate.

The filtered photon term matters: writing the photon part as the unfiltered
``diag(v t)`` does not give a bound at all. That version is
``sqrt(t sum_i Ptilde_i^2 / v_i) = sqrt(t A)`` against an implemented
``sqrt(t) A / sqrt(B2)``, so it falls *below* the implemented statistic whenever
``A > B2`` -- reachable, e.g. 6.390e-03 against 6.624e-03 on a 17-channel
width-5 probe, where the correct filtered optimum is 8.933e-03. ``C_y`` is
singular (``H 1 = 0``), so the inverse is a pseudoinverse taken on
``range(H)``, where both ``Ptilde`` and the data live.

That optimum is documented as an upper bound only, not implemented: it assumes
the leak spectrum is known exactly, and buying a 10x paper gain against an
assumption the instrument cannot deliver is not the trade this module makes.
See the derivation note.

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
  Non-negative and exactly 1.0 when filtering is bypassed. In practice at or
  below 1; the above-one values the running median produced were an artifact of
  the covariance defect, not a real gain. No upper bound is proven, so do not
  clamp it -- see `CcfBudget` for the full account.
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
    "rv_leak_contamination",
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
        **SNR-retention ratio, not a flux ratio**: the photon-limited SNR of the
        filtered statistic over the unfiltered one, on the same unmasked
        channels,

        ```text
        (t sum_i w_i Ptilde_i) / sqrt(t sum_i u_i^2 v_i)
        -------------------------------------------------,   u = H^T w.
                 sqrt(t sum_i P_i^2 / v_i)
        ```

        The numerator carries the propagated variance, so this prices both
        costs of filtering: signal lost to self-subtraction and noise gained
        from the correlation ``H`` introduces. (The diagonal-era form
        ``sqrt(sum Ptilde^2/v) / sqrt(sum P^2/v)`` is not this quantity -- it
        reads 0.818 on the pinned alternating case where the implemented value
        is 0.637.) It answers "what fraction of the achievable photon-limited
        SNR survives the continuum filter", which is the question a filter
        width is actually traded against.

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
        Number of **admitted** channels: unmasked with ``v_i > 0``. Not the
        number with a non-zero weight -- an admitted channel can carry zero
        weight (a flat island filters to zero), and the two-flat-segments
        regression reports 6 admitted channels, three per segment, with every
        weight zero.
    template_weights:
        ``w_i`` on the full channel grid, zero on excluded channels.
    admitted_mask:
        Boolean, full grid: the channels the statistic was allowed to read.
        Published because ``template_weights`` and the per-channel rates alone
        do not determine the statistic -- the filter runs blockwise on the
        admitted islands, so a consumer needs to know where those islands are
        to reproduce anything.
    noise_weights:
        ``u = H^T w`` on the full channel grid, zeros off the admitted set.
        This is what the photon variance is built from:
        ``V_ph = t sum_i u_i^2 v_i``. Applying a public adjoint to
        ``template_weights`` on the raw grid does *not* reproduce it, because
        that adjoint has no way to know the island boundaries.
    island_channel_counts:
        Length of each admitted island, in channels, in ascending wavelength
        order. Published because the filter acts within islands, not across
        the channel total: an island shorter than the filter width gets a
        boxcar renormalized over the whole block, which is a different filter
        from the one the width nominally asks for.
    island_self_subtraction_factors:
        `self_subtraction_factor` computed island by island (``H`` is block
        diagonal, so signal and propagated variance are both island sums and
        each island's retention stands alone). A one-channel island reports
        0.0, because a single sample cannot separate line from continuum.
        Beyond that the spread is large and *not* monotone in island length --
        an island shorter than the filter width has its boxcar renormalized
        over the whole block, so the filter removes only the island mean and
        can retain more than a long island does. Measured on the shipped
        systematics config: islands of 382/196/17 channels retain
        0.066/0.167/0.556 against a global 0.078. A photon-weighted average
        over that spread describes none of them, which is why both arrays are
        published.
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
    admitted_mask: NDArray[np.bool_]
    noise_weights: NDArray[np.float64]
    island_channel_counts: NDArray[np.intp]
    island_self_subtraction_factors: NDArray[np.float64]

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


def _scaled_product(
    factors: Sequence[float],
    divisors: Sequence[float] = (),
    exponent2: int = 0,
) -> float:
    """``prod(factors) / prod(divisors) * 2**exponent2``, materialized once.

    Every factor is split by `math.frexp` into a mantissa in ``[0.5, 1)`` and an
    integer power of two, so the running product cannot over- or underflow
    however extreme the individual factors are. Only the closing `math.ldexp`
    can, and by then the value genuinely is not representable.

    Exponent bookkeeping rather than logarithms: the split is exact, whereas
    ``exp(sum(log(x)))`` carries an absolute error in the log that becomes a
    relative error of ``|log x| * eps`` in the result -- 1e-13 at the magnitudes
    this module has to survive. Zero, infinity and NaN short-circuit, since
    ``frexp`` has nothing useful to say about them.
    """
    mantissa = 1.0
    exponent = int(exponent2)
    for value in factors:
        if value == 0.0 or not math.isfinite(value):
            mantissa *= value
            continue
        split_mantissa, split_exponent = math.frexp(value)
        mantissa *= split_mantissa
        exponent += split_exponent
        mantissa, extra = math.frexp(mantissa)
        exponent += extra
    for value in divisors:
        if value == 0.0:
            return math.copysign(math.inf, mantissa) if mantissa else math.nan
        if not math.isfinite(value):
            mantissa /= value
            continue
        split_mantissa, split_exponent = math.frexp(value)
        mantissa /= split_mantissa
        exponent -= split_exponent
        mantissa, extra = math.frexp(mantissa)
        exponent += extra
    if mantissa == 0.0 or not math.isfinite(mantissa):
        return mantissa
    try:
        return math.ldexp(mantissa, exponent)
    except OverflowError:
        return math.copysign(math.inf, mantissa)


def _reciprocal_split(
    values: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """``1 / values`` as ``(mantissa, exponent2)`` with the mantissa in ``(1, 2]``.

    A per-channel inverse variance can overflow float64 on its own (``v`` a
    subnormal), and no single linear or logarithmic normalization brings a
    ``5e-324``-to-``1e308`` span inside the exponent range. Keeping the power of
    two beside the mantissa does, and exactly.
    """
    mantissa, exponent = np.frexp(np.asarray(values, dtype=np.float64))
    return 1.0 / mantissa, -exponent


def _sum_split(
    mantissa: NDArray[np.float64], exponent: NDArray[np.int32]
) -> tuple[float, int]:
    """Sum ``mantissa * 2**exponent`` as a ``(mantissa, exponent2)`` pair."""
    nonzero = mantissa != 0.0
    if not bool(np.any(nonzero)):
        return 0.0, 0
    largest = int(np.max(exponent[nonzero]))
    shift = np.clip(exponent - largest, -1100, 0).astype(np.int32)
    with np.errstate(under="ignore"):
        total = float(np.sum(np.ldexp(mantissa, shift)))
    if total == 0.0 or not math.isfinite(total):
        return total, 0
    split_mantissa, split_exponent = math.frexp(total)
    return split_mantissa, largest + split_exponent


def _shifted(
    mantissa: NDArray[np.float64], exponent: NDArray[np.int32], reference: int
) -> NDArray[np.float64]:
    """``mantissa * 2**(exponent - reference)``, underflowing to zero when tiny."""
    shift = np.clip(exponent - reference, -1100, 1100).astype(np.int32)
    with np.errstate(over="ignore", under="ignore"):
        return np.ldexp(mantissa, shift)


# A filtered template that is nothing but rounding residue must not be allowed
# to become a detection. Measured worst case over flat arrays (lengths 1-4096,
# widths 3-1001): max|H x| / max|x| = 89 * eps. This threshold sits ~11x above
# that and ~3 orders of magnitude below the filtered amplitude of a 1e-10
# fractional line (4e5 * eps), so it separates roundoff from any structure a
# spectrum could physically carry.
_ROUNDOFF_RTOL = 1024.0 * float(np.finfo(np.float64).eps)


def _zero_roundoff_islands(
    filtered: NDArray[np.float64],
    raw: NDArray[np.float64],
    islands: list[slice],
    *,
    rtol: float = _ROUNDOFF_RTOL,
) -> NDArray[np.float64]:
    """Zero each island whose filtered template is pure rounding residue.

    ``H`` annihilates a constant analytically, but only to a few ulp in
    floating point: a flat island of ``1.03 Hz`` filters to ``2.2e-16`` rather
    than to zero. Every quantity downstream is scale-free in the template --
    weights are normalized, and the SNR is a ratio -- so that residue does not
    stay small. It is amplified into an ordinary-looking detection: 3 flat
    channels at 1.03 Hz against a 100 Hz leak report ``ccf_snr = 13.1`` at
    ``t = 3600 s``, from a spectrum with no line in it.

    The test has to be per island and relative to that island's own template
    magnitude. A global absolute threshold gets it wrong in both directions:
    it misses roundoff on a bright island and erases real structure on a faint
    one (a genuine residual on a 1 Hz island is pure noise to a threshold
    calibrated for a 1e12 Hz one).
    """
    if not islands:
        return filtered
    out = filtered
    for block in islands:
        scale = float(np.max(np.abs(raw[block]))) if raw[block].size else 0.0
        residual = float(np.max(np.abs(filtered[block]))) if scale > 0.0 else 0.0
        if scale <= 0.0 or residual <= rtol * scale:
            if out is filtered:
                out = filtered.copy()
            out[block] = 0.0
    return out


def _odd_width(width_channels: int) -> int:
    """Coerce a filter width to odd and at least 3 channels."""
    width = int(width_channels)
    if width < 3:
        width = 3
    if width % 2 == 0:
        width += 1
    return width


def _boxcar(values: NDArray[np.float64], width: int) -> NDArray[np.float64]:
    """Zero-padded boxcar average: the symmetric banded operator ``M``.

    ``M_ij = 1/width`` for ``|i - j| <= width // 2`` and zero otherwise,
    truncated to the array. Restricting a symmetric Toeplitz matrix to a
    principal submatrix leaves it symmetric, so ``M = M^T`` exactly -- which is
    what makes the adjoint of the smoother cheap to write down. (``M``, not
    ``K``: ``K`` is the systematic-amplitude scalar in the covariance.)
    """
    return uniform_filter1d(
        values, size=width, mode="constant", cval=0.0, axis=-1
    )


def _smoother_normalization(n_channels: int, width: int) -> NDArray[np.float64]:
    """``c = M 1``: the in-array fraction of each window. 1 inside, <1 at edges."""
    return _boxcar(np.ones(n_channels, dtype=np.float64), width)


def _smooth(values: NDArray[np.float64], width: int) -> NDArray[np.float64]:
    """Apply the low-pass smoother ``S = diag(1/c) M``.

    Normalizing by the in-array window fraction makes ``S 1 = 1`` exactly, at
    the edges as well as the interior, so the high-pass ``H = I - S``
    annihilates a constant everywhere. A flat stellar continuum is therefore
    removed exactly rather than leaving an edge residual.
    """
    return _boxcar(values, width) / _smoother_normalization(values.shape[-1], width)


def _smooth_adjoint(values: NDArray[np.float64], width: int) -> NDArray[np.float64]:
    """Apply ``S^T = M^T diag(1/c) = M diag(1/c)``, using ``M = M^T``."""
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


def _islands(admitted: NDArray[np.intp]) -> list[slice]:
    """Maximal runs of consecutive *raw* channels, as slices into the compact array.

    The continuum filter must never bridge a telluric gap. Two flat admitted
    segments either side of a masked band are not neighbours in wavelength, and
    smoothing across the gap invents continuum structure that is not there:
    ``[10,10,10]`` beside ``[100,100,100]``, compacted and boxcar-filtered at
    width 3, yields an inferred continuum of ``[10,10,40,70,100,100]`` and
    residuals of +-30 in a spectrum with no line anywhere.

    So ``H`` is block diagonal over these islands. Each block carries its own
    row-stochastic smoother, hence ``H 1 = 0`` *per island*, and an island
    narrower than the filter width simply gets its renormalized boxcar (a
    one-channel island gets ``H = 0``, which is correct: one sample cannot
    separate line from continuum).
    """
    if admitted.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(admitted) != 1) + 1
    starts = np.concatenate(([0], cuts))
    stops = np.concatenate((cuts, [admitted.size]))
    return [slice(int(a), int(b)) for a, b in zip(starts, stops, strict=True)]


def _filtered_blockwise(
    values: NDArray[np.float64], width_channels: int, islands: list[slice]
) -> NDArray[np.float64]:
    """`_filtered` applied independently on each contiguous island."""
    if int(width_channels) <= 0:
        return np.asarray(values, dtype=np.float64)
    out = np.empty_like(values)
    for block in islands:
        out[block] = continuum_filter(values[block], width_channels)
    return out


def _filtered_adjoint_blockwise(
    values: NDArray[np.float64], width_channels: int, islands: list[slice]
) -> NDArray[np.float64]:
    """`_filtered_adjoint` applied independently on each contiguous island.

    ``H`` is block diagonal, so ``H^T`` is block diagonal with each block
    transposed; applying the per-block adjoint is exactly ``H^T``.
    """
    if int(width_channels) <= 0:
        return np.asarray(values, dtype=np.float64)
    out = np.empty_like(values)
    for block in islands:
        out[block] = continuum_filter_adjoint(values[block], width_channels)
    return out


def _analysis_indices(
    keep: NDArray[np.bool_], variance_rate_hz: NDArray[np.float64]
) -> NDArray[np.intp]:
    """Channels the statistic is allowed to read, as an index array.

    **Masking semantics: raw-channel exclusion.** A telluric-masked channel
    carries no usable data, so no statistic may depend on its photons at all.
    Everything downstream therefore works on the compacted sub-vector of
    admitted channels, and the continuum filter is built on *that* vector: the
    statistic is ``a . N`` with ``a`` supported on this set, full stop.

    The alternative -- filter on the full vector and zero the filtered-output
    weights -- looks equivalent but is not, because ``H^T`` spreads support:
    ``a = H^T w`` is generally nonzero where ``w`` is zero, so masked photons
    leak back into the statistic. That is what `matched_filter_ccf` used to do,
    and it is why its SNR disagreed with the RV scan under a mask (0.1936
    against 0.0 on the review's five-channel probe).

    Channels with ``v_i <= 0`` are excluded on the same footing, rather than
    rejected with an error: a channel with no photons carries no information,
    and admitting it would make ``H diag(v) H^T`` rank-deficient relative to
    ``H``, breaking the ``range(C_y) = range(H)`` identity the RV optimum
    relies on. Excluding it restores that identity by construction on the
    reduced space, and degrades gracefully rather than failing a whole sweep
    because one anchor has an empty channel.
    """
    return np.flatnonzero(keep & (variance_rate_hz > 0.0))


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
    admitted = _analysis_indices(keep, variance_rate)

    # Raw-channel exclusion: compact to the admitted channels and build the
    # filter on that vector, so the statistic cannot read a masked photon.
    variance_admitted = variance_rate[admitted]
    planet_admitted = planet[admitted]
    islands = _islands(admitted)
    planet_filtered = _zero_roundoff_islands(
        _filtered_blockwise(
            planet_admitted, continuum_filter_width_channels, islands
        ),
        planet_admitted,
        islands,
    )
    leak_filtered = _filtered_blockwise(
        channels.leak_rate_hz[admitted], continuum_filter_width_channels, islands
    )

    admitted_weights = (
        planet_filtered / variance_admitted
        if admitted.size
        else np.zeros(0, dtype=np.float64)
    )
    weights = np.zeros(n_ch, dtype=np.float64)
    weights[admitted] = admitted_weights
    # u = H^T w. The statistic is w . (H N) for independent Poisson counts N, so
    # its variance is t sum_i u_i^2 v_i, NOT the raw diagonal t sum_i w_i^2 v_i:
    # the filter correlates neighbouring channels.
    noise_weights = _filtered_adjoint_blockwise(
        admitted_weights, continuum_filter_width_channels, islands
    )

    admitted_mask = np.zeros(n_ch, dtype=bool)
    admitted_mask[admitted] = True
    scattered_noise_weights = np.zeros(n_ch, dtype=np.float64)
    scattered_noise_weights[admitted] = noise_weights

    signal_counts = float(
        np.sum(admitted_weights * planet_filtered) * integration_time_s
    )
    photon_var_counts = float(
        np.sum(noise_weights * noise_weights * variance_admitted)
        * integration_time_s
    )
    systematic_ccf_leakage = float(np.sum(admitted_weights * leak_filtered))

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
            np.sum(planet_admitted * planet_admitted / variance_admitted)
            * integration_time_s
        ) if admitted.size else 0.0
        # No template signal at all to retain: report 0.0 by convention.
        self_subtraction_factor = (
            ccf_snr_photon_limited / math.sqrt(snr_sq_unfiltered)
            if snr_sq_unfiltered > 0.0
            else 0.0
        )

    abs_sum = float(np.sum(np.abs(admitted_weights)))
    sq_sum = float(np.sum(admitted_weights * admitted_weights))
    n_effective_channels = abs_sum * abs_sum / sq_sum if sq_sum > 0.0 else 0.0

    # Per island, because the global factor averages a crippled narrow island
    # against healthy wide ones and hides it. H is block diagonal, so the
    # signal and the propagated variance are both sums over islands and each
    # island's retention is well defined on its own.
    island_channel_counts = np.array(
        [block.stop - block.start for block in islands], dtype=np.intp
    )
    island_self_subtraction = np.zeros(len(islands), dtype=np.float64)
    for position, block in enumerate(islands):
        if int(continuum_filter_width_channels) <= 0:
            island_self_subtraction[position] = 1.0
            continue
        island_signal = float(
            np.sum(admitted_weights[block] * planet_filtered[block])
        )
        island_variance = float(
            np.sum(
                noise_weights[block]
                * noise_weights[block]
                * variance_admitted[block]
            )
        )
        island_unfiltered = float(
            np.sum(
                planet_admitted[block]
                * planet_admitted[block]
                / variance_admitted[block]
            )
        )
        if island_signal > 0.0 and island_variance > 0.0 and island_unfiltered > 0.0:
            island_self_subtraction[position] = (
                island_signal / math.sqrt(island_variance)
            ) / math.sqrt(island_unfiltered)

    return CcfBudget(
        ccf_snr=ccf_snr,
        ccf_snr_photon_limited=ccf_snr_photon_limited,
        signal_counts=signal_counts,
        photon_var_counts=photon_var_counts,
        systematic_var_counts=float(systematic_var_counts),
        systematic_ccf_leakage=systematic_ccf_leakage,
        self_subtraction_factor=self_subtraction_factor,
        n_effective_channels=n_effective_channels,
        n_channels_used=int(admitted.size),
        template_weights=weights,
        admitted_mask=admitted_mask,
        noise_weights=scattered_noise_weights,
        island_channel_counts=island_channel_counts,
        island_self_subtraction_factors=island_self_subtraction,
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
    under one inner product. Cauchy-Schwarz bounds it by
    ``sqrt(<y, y>_{C_y^+})``, with equality when ``Ttilde(rv)`` is parallel to
    ``y``.

    **That does not by itself put the peak at the true radial velocity, and an
    earlier version of this docstring wrongly said it did.** ``y`` is the
    *observed* spectrum -- planet plus leak -- not the planet template, so the
    scan peaks where the template best matches the observation. The two
    coincide only when the filtered leak projection is negligible, i.e. when
    the leak is smooth enough for ``H`` to annihilate it. Give the leak
    structure the filter cannot remove and the scan locks onto the leak: on a
    twelve-channel probe with a randomly structured leak the true velocity
    scores -5.99 while a four-channel shift scores 9.80. The guarantee is
    conditional on a smooth leak -- the design case, but an assumption rather
    than a theorem.

    **The pseudoinverse has a closed form here, so this stays O(n) per trial.**
    ``H`` is block diagonal over the admitted islands (see `_islands`), and each
    block is ``I - S`` with ``S`` row-stochastic and irreducible, so by
    Perron-Frobenius each block's unit eigenvalue is simple. Hence ``null(H)``
    is the span of the **island indicator vectors** -- not the single global
    constant -- and ``range(H^T)`` is its orthogonal complement. Maximizing
    ``(a.P)^2 / (t a^T diag(v) a)`` over that subspace is one Lagrange
    constraint *per island*:

    ```text
    a_i = m_i (T_i - lambda_k) / v_i   for channel i in island k,
    lambda_k = sum_{i in k} m_i T_i/v_i / sum_{i in k} m_i/v_i,
    CCF(rv) = sqrt(t) * (a . R_obs) / sqrt(a^T diag(v) a),
    ```

    with ``R_obs`` the observed planet-plus-leak rate. No eigendecomposition and
    no dense matrix: ``a`` is the raw template inverse-variance weighted with a
    variance-weighted mean removed island by island, which is precisely "blind
    to the per-island constant modes and nothing else". A test pins this against
    the dense ``C_y^+`` form.

    **How it is evaluated, and why not the obvious way.** Writing the offset out
    gives an equivalent pair-difference form that the implementation uses
    instead,

    ```text
    a_i = u_i (T_i - T_ref) - (u_i / s_k) sum_{j in k} u_j (T_j - T_ref),
    u = 1/v,   s_k = sum_{j in k} u_j,   ref = the largest-u channel of island k,
    ```

    which differs in two ways that matter. A template constant on an island
    gives ``T_i - T_ref = 0`` *exactly*, so such an island contributes exactly
    nothing rather than a rounding residue that normalization would amplify into
    an O(1) score; no threshold is needed, and none is applied, so genuine
    structure on a faint island survives beside a bright one. And every product
    is formed with `math.frexp` exponent bookkeeping, so a variance span wider
    than float64's exponent range -- ``5e-324`` beside ``1e308``, where ``1/v``
    overflows and no single normalization of ``u`` can hold both ends -- yields
    the correct antisymmetric filter instead of a NaN.

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
    is blind only to the per-island constant modes, so unlike
    `matched_filter_ccf` -- which keeps its fixed-template weights on purpose,
    and with them the boxcar's suppression of smooth leak structure -- this scan
    is not protected against a non-constant leak continuum. The two statistics
    are therefore not comparable in either direction: neither bounds the other
    (``P = [1,2,1]`` under ``L = [9,8,9]`` is flat in total, so this scan reads
    0.0 where the detection statistic reads 0.258, and a structured leak can
    push this one far above). Do not read this score as the detection SNR, and
    do not read its argmax as a radial velocity without checking
    `rv_leak_contamination` first.

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

    analysis = _RvAnalysis.build(
        data_channels,
        continuum_filter_width_channels=continuum_filter_width_channels,
        telluric_transmission_per_channel=telluric_transmission_per_channel,
        telluric_mask_threshold=telluric_mask_threshold,
    )
    values = np.zeros(rv_array.size, dtype=np.float64)
    if analysis is None:
        return values

    observed = (data_channels.planet_rate_hz + data_channels.leak_rate_hz)[
        analysis.admitted
    ]
    observed_scale = float(np.max(np.abs(observed)))
    if observed_scale <= 0.0:
        return values
    sqrt_time = math.sqrt(integration_time_s)

    for index, rv in enumerate(rv_array):
        trial = channels_at_trial_rv_fn(float(rv))
        _require_same_grid(trial, data_channels, rv)
        filter_vector = analysis.filter_vector(
            trial.planet_rate_hz[analysis.admitted]
        )
        if filter_vector is None:
            continue
        # Normalize the variance and the observation inside their sums: at
        # v ~ 1e308 the plain sum of f^2 v overflows and the score collapses to
        # -0.0 on a representable answer.
        norm = math.sqrt(
            float(
                np.sum(
                    filter_vector
                    * filter_vector
                    * (analysis.variance_admitted / analysis.variance_max)
                )
            )
        )
        if norm <= 0.0:
            continue
        numerator = float(np.sum(filter_vector * (observed / observed_scale)))
        values[index] = (
            sqrt_time
            * (observed_scale / math.sqrt(analysis.variance_max))
            * (numerator / norm)
        )
    return values


def rv_leak_contamination(
    trial_planet_rate_hz: NDArray[np.float64],
    *,
    data_channels: ChannelSpectra,
    leak_rate_hz_per_channel: NDArray[np.float64],
    continuum_filter_width_channels: int,
    telluric_transmission_per_channel: NDArray[np.float64] | None = None,
    telluric_mask_threshold: float = 0.0,
    **_ignored: object,
) -> float:
    """How much of the RV score a leak could contribute, relative to the planet.

    `ccf_vs_rv` peaks where the trial template best matches the *observed*
    spectrum, so its argmax is the planet's radial velocity only when the leak
    contributes little to the score. This is the size of that term:

    ```text
    contamination = |a . L| / |a . P_trial|,
    ```

    with ``a`` the whitened filter `ccf_vs_rv` builds for ``trial_planet_rate_hz``
    and ``L`` the leak spectrum on the same channel grid. It is the ratio of the
    leak's contribution to the score's numerator to the planet's, at one trial
    velocity -- the term that shifts the peak, in the units of the term that
    should set it. Well below 1 means the scan is measuring the planet; at or
    above 1 the peak location is not evidence about the radial velocity.

    ``L`` is passed separately from ``data_channels`` on purpose: a pipeline
    that subtracts a bright-port-measured leak mean before correlating has no
    leak left in its observation, and the honest question there is what the
    score would suffer if that subtraction were imperfect. Pass the leak
    spectrum the subtraction is meant to remove, and read the result as the
    contamination at 100% subtraction error; scale it by the fractional error
    for a realistic figure.

    Returns ``inf`` when the planet term vanishes (no filter, or a template the
    filter is blind to): a score with no planet content in it cannot localize
    anything.
    """
    analysis = _RvAnalysis.build(
        data_channels,
        continuum_filter_width_channels=continuum_filter_width_channels,
        telluric_transmission_per_channel=telluric_transmission_per_channel,
        telluric_mask_threshold=telluric_mask_threshold,
    )
    if analysis is None:
        return math.inf
    template = np.asarray(trial_planet_rate_hz, dtype=np.float64)
    if template.shape != (data_channels.n_channels,):
        raise ValueError(
            "trial_planet_rate_hz must have one entry per channel, got shape "
            f"{template.shape} for {data_channels.n_channels} channels"
        )
    leak = np.asarray(leak_rate_hz_per_channel, dtype=np.float64)
    if leak.shape != template.shape:
        raise ValueError(
            "leak_rate_hz_per_channel must have one entry per channel, got "
            f"shape {leak.shape} for {data_channels.n_channels} channels"
        )
    filter_vector = analysis.filter_vector(template[analysis.admitted])
    if filter_vector is None:
        return math.inf

    planet_admitted = template[analysis.admitted]
    leak_admitted = leak[analysis.admitted]
    planet_scale = float(np.max(np.abs(planet_admitted)))
    leak_scale = float(np.max(np.abs(leak_admitted)))
    planet_term = abs(
        float(np.sum(filter_vector * (planet_admitted / planet_scale)))
        if planet_scale > 0.0
        else 0.0
    )
    leak_term = abs(
        float(np.sum(filter_vector * (leak_admitted / leak_scale)))
        if leak_scale > 0.0
        else 0.0
    )
    if planet_term == 0.0 or planet_scale <= 0.0:
        return math.inf
    if leak_term == 0.0:
        return 0.0
    return _scaled_product((leak_term, leak_scale), (planet_term, planet_scale))


@dataclass(frozen=True, eq=False)
class _RvAnalysis:
    """The channel set, islands, and inverse variances the RV scan works on.

    Built once per observation and reused across trial velocities, which is
    what keeps the scan ``O(n)`` per trial. The inverse variances are carried as
    ``(mantissa, exponent2)`` pairs rather than floats: ``1/v`` overflows on its
    own for a subnormal ``v``, and the pair-difference filter needs products of
    a huge ``u`` with a tiny one, which no single normalization can hold.
    """

    admitted: NDArray[np.intp]
    variance_admitted: NDArray[np.float64]
    variance_max: float
    islands: list[slice]
    bypassed: bool
    inverse_mantissa: NDArray[np.float64]
    inverse_exponent: NDArray[np.int32]
    island_references: list[int]
    island_totals: list[tuple[float, int]]

    @classmethod
    def build(
        cls,
        data_channels: ChannelSpectra,
        *,
        continuum_filter_width_channels: int,
        telluric_transmission_per_channel: NDArray[np.float64] | None,
        telluric_mask_threshold: float,
    ) -> _RvAnalysis | None:
        """Return the analysis set, or ``None`` when no channel is admitted."""
        keep = _telluric_mask(
            data_channels.n_channels,
            telluric_transmission_per_channel,
            telluric_mask_threshold,
        )
        variance_rate = data_channels.total_rate_hz()
        admitted = _analysis_indices(keep, variance_rate)
        if admitted.size == 0:
            return None
        variance_admitted = variance_rate[admitted]
        islands = _islands(admitted)
        inverse_mantissa, inverse_exponent = _reciprocal_split(variance_admitted)
        references: list[int] = []
        totals: list[tuple[float, int]] = []
        for block in islands:
            # Largest inverse variance = smallest variance; comparing v avoids
            # having to order the (mantissa, exponent) pairs by hand.
            references.append(block.start + int(np.argmin(variance_admitted[block])))
            totals.append(
                _sum_split(inverse_mantissa[block], inverse_exponent[block])
            )
        return cls(
            admitted=admitted,
            variance_admitted=variance_admitted,
            variance_max=float(np.max(variance_admitted)),
            islands=islands,
            bypassed=int(continuum_filter_width_channels) <= 0,
            inverse_mantissa=inverse_mantissa,
            inverse_exponent=inverse_exponent,
            island_references=references,
            island_totals=totals,
        )

    def filter_vector(
        self, template_admitted: NDArray[np.float64]
    ) -> NDArray[np.float64] | None:
        """The whitened filter ``a`` for one trial template, up to a positive scale.

        Returned normalized to ``max|a| ~ 1``; the statistic is a ratio in
        ``a``, so the scale is free. ``None`` when the filter is identically
        zero -- an all-zero template, or a template constant on every island,
        which the filter genuinely cannot see.
        """
        template = np.asarray(template_admitted, dtype=np.float64)
        template_scale = float(np.max(np.abs(template)))
        if template_scale <= 0.0:
            return None
        template = template / template_scale

        size = template.size
        term_mantissa = np.zeros(size, dtype=np.float64)
        term_exponent = np.zeros(size, dtype=np.int32)
        offset_mantissa = np.zeros(size, dtype=np.float64)
        offset_exponent = np.zeros(size, dtype=np.int32)

        if self.bypassed:
            # H = I has no null space, so there is no zero-sum constraint and no
            # mean to remove: the optimum is the plain inverse-variance
            # weighting. Applying the constraint here would be wrong, and for a
            # single admitted channel it silently returns zero.
            mantissa, exponent = np.frexp(template)
            term_mantissa = self.inverse_mantissa * mantissa
            term_exponent = self.inverse_exponent + exponent
        else:
            for block, reference, (total_mantissa, total_exponent) in zip(
                self.islands,
                self.island_references,
                self.island_totals,
                strict=True,
            ):
                delta = template[block] - template[reference]
                mantissa, exponent = np.frexp(delta)
                block_mantissa = self.inverse_mantissa[block] * mantissa
                block_exponent = self.inverse_exponent[block] + exponent
                term_mantissa[block] = block_mantissa
                term_exponent[block] = block_exponent
                weighted_mantissa, weighted_exponent = _sum_split(
                    block_mantissa, block_exponent
                )
                if weighted_mantissa == 0.0 or total_mantissa == 0.0:
                    continue
                # u_i / s_k is at most 1 by construction, so it materializes
                # safely even when u_i itself does not.
                share = _shifted(
                    self.inverse_mantissa[block] / total_mantissa,
                    self.inverse_exponent[block],
                    total_exponent,
                )
                offset_mantissa[block] = -(share * weighted_mantissa)
                offset_exponent[block] = weighted_exponent

        largest: int | None = None
        for mantissa, exponent in (
            (term_mantissa, term_exponent),
            (offset_mantissa, offset_exponent),
        ):
            nonzero = mantissa != 0.0
            if bool(np.any(nonzero)):
                candidate = int(np.max(exponent[nonzero]))
                largest = candidate if largest is None else max(largest, candidate)
        if largest is None:
            return None
        return _shifted(term_mantissa, term_exponent, largest) + _shifted(
            offset_mantissa, offset_exponent, largest
        )


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
        ``A = sum_i w_i Ptilde1_i`` (Hz per unit contrast). Equal to ``B2`` only
        when continuum filtering is bypassed; with a filter active
        ``u0 = H^T w0 != w0`` and the two are different quantities (``A/B2``
        runs to 1.6 on the structured probe). Never assert them equal.
    photon_var_zero_contrast:
        ``B2 = sum_i u0_i^2 v0_i`` (Hz), ``u0 = H^T w``: the H-propagated photon
        variance rate with no planet.
    planet_selfnoise_coefficient:
        ``C2 = sum_i u0_i^2 P1_i`` (Hz per unit contrast): the planet's own shot
        noise, the term that makes the condition quadratic rather than linear.
    systematic_var_counts:
        ``V_sys`` (counts^2). Independent of ``c`` in both regimes, which is
        what keeps the root closed-form.
    template_weights:
        ``w_i = Ptilde1_i / v0_i`` frozen at zero contrast, zero on masked
        channels. **Not** zero where ``v0_i = 0``: those channels are where the
        frozen weight diverges, and the limit taken there (see
        `ccf_contrast_at_target_snr`) puts the whole weighting on them, leaving
        the finite-variance channels at zero instead. Only channels with
        neither variance nor template signal, and telluric-masked ones, are
        excluded outright.
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
    a known planet, invert the photon-weighted CCF to ask how faint a planet the
    same observation would still detect at ``target_snr``. That is what a CCF
    contrast curve is made of.

    The planet enters linearly in the contrast: the first argument is the
    per-channel planet rate at *unit* band-integrated contrast, so the actual
    planet rate is ``c * P1_i``. Writing ``v0_i = L_i + sky_i + dark_i`` for the
    planet-free variance rate, ``Ptilde1 = continuum_filter(P1)``,
    ``Ltilde = continuum_filter(L)``, and ``w_i = m_i Ptilde1_i / v0_i``,

    ```text
    A  = sum_i w_i Ptilde1_i     [Hz per unit contrast]
    B2 = sum_i u0_i^2 v0_i       [Hz],  u0 = H^T w  (H-propagated, not w^2)
    C2 = sum_i u0_i^2 P1_i       [Hz per unit contrast]
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
      The structural coefficient ``C2/A - D/J0`` is the other gate: it is
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

    **Noiseless channels.** A channel with ``v0_i = 0`` and ``P1_i > 0`` is not
    empty here, though it is for `matched_filter_ccf`: at contrast ``c`` its
    variance is ``c P1_i``, which the ``c C2`` term already carries. Its frozen
    weight diverges, so the weighting is taken in the limit ``v0 -> 0``, where
    those channels dominate completely and the finite-variance channels drop to
    zero weight. The *variance* sums keep every admitted channel:
    ``u = H^T w`` spreads onto the island's ordinary neighbours and ``B2``
    collects their photons. Island membership is decided on "has variance or
    has template signal", never on ``v0``, so a noiseless channel between two
    ordinary ones stays in their island -- ``P1 = [0,1,0]`` against
    ``v0 = [1,0,1]`` at width 3 and ``n = 5, t = 1`` solves to
    ``c = 25.4903810567...``, matching the same problem solved at
    ``v0[1] = 1e-6, 1e-12, 1e-100``.

    Raises ``ValueError`` if ``A <= 0``: a template with no surviving line
    structure (a lineless planet spectrum, one whose lines all fall in masked
    channels, or one that is flat on every island so that ``H P1`` is pure
    rounding residue) has no contrast at which it reaches the target, and
    reporting ``inf`` would silently poison a contrast curve. Also raises
    ``ValueError`` -- not ``OverflowError`` -- if the contrast root is not
    representable in float64, so a caller sweeping separation anchors has a
    single exception type to catch per anchor.

    ``systematic_var_counts`` is a *diagnostic*, and it saturates to ``inf``
    rather than aborting the solve: the root needs ``sqrt(V_sys)/t``, in which
    the exposure cancels analytically, so a ``V_sys`` too large to represent
    does not make the root unreachable. ``P1 = 1e10, L = 1e170, eps = 1`` at
    ``t = 1e200`` returns ``c = 5e160`` with ``systematic_var_counts = inf``,
    and that is deliberate.
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
    # The detection statistic drops v0 = 0 channels because they carry no
    # information at fixed contrast. The SOLVER must not: at contrast c such a
    # channel has variance c*P1 > 0, so it is informative through the planet's
    # own shot noise -- the c*C2 term the quadratic already carries. Only
    # v0 = 0 AND P1 = 0 is genuinely empty.
    #
    # Island topology is decided here, on "has any content at all", and never
    # on v0. A noiseless channel sitting between two ordinary ones belongs to
    # their island: it is adjacent in wavelength, and the continuum filter is a
    # statement about wavelength adjacency, not about noise. Splitting it out
    # into a singleton (which the previous version did) gave that island H = 0
    # and raised "no template signal" on a problem whose answer is an ordinary
    # number -- P1 = [0,1,0] against v0 = [1,0,1] at width 3 solves to
    # c = 25.4903810567..., the limit of the same solve as v0[1] -> 0.
    informative = (variance_rate_zero_contrast > 0.0) | (planet_unit > 0.0)
    admitted = np.flatnonzero(keep & informative)
    if admitted.size == 0:
        raise ValueError(
            "no template signal survives: every channel is telluric-masked or "
            "has neither variance nor template signal, so no channel is "
            "admitted and no contrast reaches the target SNR"
        )
    planet_admitted = planet_unit[admitted]
    variance_admitted = variance_rate_zero_contrast[admitted]

    # Accumulate the coefficients on NORMALIZED arrays. Forming A, B2 and C2
    # from the raw rates over/underflows on inputs whose root is perfectly
    # representable: P1 ~ 1e-200 against a 1e100 leak flushes A to zero,
    # P1 ~ 1e200 overflows C2, and a 1e-200 leak overflows w^2 while B2 is being
    # built. Two linear scales and one power of two keep every intermediate in
    # range:
    #
    #   alpha = max|P1|            -> |p| <= 1
    #   beta  = max v0             -> normalized variance <= 1
    #   2^G   = max|ptilde/v0|     -> |w_hat| <= 1  (G an exact integer)
    #
    # With kappa = alpha 2^G the scales come back out analytically,
    #
    #   A = kappa alpha A'',        B2 = kappa^2 beta B2'',
    #   C2 = kappa^2 alpha C2'',    B_w = kappa beta B_w'',
    #   V_sys = (kappa beta)^2 V_sys''    (quadratic in B_w in both regimes)
    #
    # and kappa cancels out of the root entirely.
    scale_planet = float(np.max(np.abs(planet_admitted)))
    if scale_planet <= 0.0:
        raise ValueError(
            "no template signal survives continuum filtering and telluric "
            "masking (A <= 0), so no contrast reaches the target SNR"
        )
    # All-noiseless problems have max v0 = 0; the normalization is then free
    # (B2 and the leak are identically zero) and 1.0 keeps the divisions valid.
    scale_rate = float(np.max(variance_admitted)) or 1.0
    normalized_planet = planet_admitted / scale_planet
    normalized_variance = variance_admitted / scale_rate

    islands = _islands(admitted)
    planet_unit_filtered = _zero_roundoff_islands(
        _filtered_blockwise(
            normalized_planet, continuum_filter_width_channels, islands
        ),
        normalized_planet,
        islands,
    )
    leak_filtered = _filtered_blockwise(
        leak[admitted] / scale_rate, continuum_filter_width_channels, islands
    )
    # Where an admitted channel has v0 = 0 the frozen weight P1tilde/v0
    # diverges, and in the limit those channels dominate the weighting
    # completely -- every finite-variance channel's weight is smaller by a
    # factor v0 -> 0 and drops out. Taking that limit: the true weight there is
    # P1tilde/(c P1), whose c cancels out of the scale-invariant statistic, so
    # the WEIGHTS restrict to the zero-variance set with denominator P1. The
    # variance sums do not restrict: u = H^T w spreads onto the island's
    # ordinary channels, and B2 = sum u^2 v0 collects their photons.
    #
    # The test is on the FILTERED template, not the raw one: a noiseless
    # channel the filter cannot see carries zero weight in the limit too
    # (P1tilde/(c P1) = 0 there), so it must not be allowed to displace the
    # ordinary channels that can still see something.
    dominant = (
        (variance_admitted <= 0.0)
        & (planet_admitted > 0.0)
        & (planet_unit_filtered != 0.0)
    )
    if np.any(dominant):
        weight_variance = np.where(dominant, planet_admitted, math.inf)
    else:
        weight_variance = np.where(
            variance_admitted > 0.0, variance_admitted, math.inf
        )

    # Weights with the power of two kept beside the mantissa. Dividing ptilde by
    # the variance overflows whenever v0 spans more than the exponent range --
    # 5e-324 against 1e308 is a span of 1e631, and no single linear scale can
    # bring both ends inside float64. Subtracting the largest exponent before
    # materializing bounds |w_hat| by ~1 whatever the span, and negligible
    # channels underflow to zero, which is the right weight for them. The split
    # is exact, so unlike accumulating logarithms it costs no precision at
    # extreme magnitudes.
    usable_weight = (weight_variance > 0.0) & np.isfinite(weight_variance)
    weight_mantissa, weight_exponent = np.frexp(planet_unit_filtered)
    variance_mantissa, variance_exponent = np.frexp(
        np.where(usable_weight, weight_variance, 1.0)
    )
    weight_mantissa = np.where(
        usable_weight, weight_mantissa / variance_mantissa, 0.0
    )
    weight_exponent = (weight_exponent - variance_exponent).astype(np.int32)
    nonzero_weight = weight_mantissa != 0.0
    if not bool(np.any(nonzero_weight)):
        raise ValueError(
            "no template signal survives continuum filtering and telluric "
            "masking (A <= 0), so no contrast reaches the target SNR"
        )
    weight_scale_exponent = int(np.max(weight_exponent[nonzero_weight]))
    admitted_weights = _shifted(
        weight_mantissa, weight_exponent, weight_scale_exponent
    )

    normalized_signal = float(np.sum(admitted_weights * planet_unit_filtered))
    if normalized_signal <= 0.0:
        raise ValueError(
            "no template signal survives continuum filtering and telluric "
            "masking (A <= 0), so no contrast reaches the target SNR"
        )
    # B2 and C2 are the two halves of the H-propagated photon variance rate,
    # v = v0 + c P1, split by their dependence on c. Using u = H^T w keeps the
    # variance exact; using w would revert to the diagonal approximation. The
    # split stays affine in c, so the target-SNR condition stays quadratic.
    noise_weights = _filtered_adjoint_blockwise(
        admitted_weights, continuum_filter_width_channels, islands
    )
    normalized_photon_var = float(
        np.sum(noise_weights * noise_weights * normalized_variance)
    )
    normalized_selfnoise = float(
        np.sum(noise_weights * noise_weights * normalized_planet)
    )
    normalized_leak_projection = float(np.sum(admitted_weights * leak_filtered))

    # V_sys in normalized units. The (kappa beta)^2 that turns it into counts^2
    # is applied only to the reported field, never inside the root, so a
    # diagnostic that saturates cannot make a representable root unreachable.
    # The root needs sqrt(V_sys)/t, not V_sys. Forming V_sys first squares
    # eps*B_w*t and then divides the square root by t again -- the t cancels
    # analytically, so squaring it is a pure overflow hazard (eps=1, B_w~1,
    # t=1e200 overflows a root that is t-independent). Take the square root
    # per regime instead:
    #     no fringe tracking:  sqrt(V_sys)/t = eps |B_w|
    #     fringe tracking:     sqrt(V_sys)/t = eps |B_w| / sqrt(2 f t)
    leak_magnitude = eps_systematic * abs(normalized_leak_projection)
    if with_fringe_tracking:
        systematic_root_over_time = leak_magnitude / math.sqrt(
            2.0 * servo_bw_hz * integration_time_s
        )
    else:
        systematic_root_over_time = leak_magnitude
    try:
        normalized_systematic = _systematic_variance_counts(
            normalized_leak_projection,
            integration_time_s,
            eps_systematic,
            servo_bw_hz,
            with_fringe_tracking,
        )
    except OverflowError:
        normalized_systematic = math.inf

    # kappa = alpha * 2^G is the physical weight scale, and it can overflow on
    # its own while every reported quantity is representable: P1 = 1e-300
    # against a 5e-324 leak puts 2^G at 2e323, yet the weight is 2.0e23 and
    # A = B2 = 2.0e-277. So the diagnostics are never formed by materializing
    # kappa and multiplying. Each is assembled in sign-and-exponent form by
    # `_scaled_product`, which folds every factor's power of two into an integer
    # and materializes once at the end -- the tiny planet scale cancels the huge
    # weight scale before anything is rounded to inf or NaN.
    weight_mantissa_scale, weight_planet_exponent = math.frexp(scale_planet)
    weights = np.zeros(n_ch, dtype=np.float64)
    weights[admitted] = _shifted(
        admitted_weights * weight_mantissa_scale,
        np.zeros(admitted.size, dtype=np.int32),
        -(weight_scale_exponent + weight_planet_exponent),
    )
    signal_coefficient = _scaled_product(
        (normalized_signal, scale_planet, scale_planet),
        exponent2=weight_scale_exponent,
    )
    photon_var_zero_contrast = _scaled_product(
        (normalized_photon_var, scale_planet, scale_planet, scale_rate),
        exponent2=2 * weight_scale_exponent,
    )
    planet_selfnoise_coefficient = _scaled_product(
        (normalized_selfnoise, scale_planet, scale_planet, scale_planet),
        exponent2=2 * weight_scale_exponent,
    )
    systematic_var_counts = _scaled_product(
        (
            normalized_systematic,
            scale_planet,
            scale_planet,
            scale_rate,
            scale_rate,
        ),
        exponent2=2 * weight_scale_exponent,
    )

    # Solve the normalized quadratic. Substituting the scale relations above
    # into (A^2 t^2) c^2 - (n^2 t C2) c - n^2 (t B2 + V_sys) = 0 and dividing
    # through leaves
    #
    #     c^2 - P c - Q = 0,
    #     P = g * (n C2''/(A'' t)),   g = n / (A'' alpha),
    #     Q = g^2 (beta B2''/t + beta^2 V_sys''/t^2),
    #
    # so n never appears squared inside one product (a 1e-200 target flushes
    # n^2 to zero) and beta t B2'' is never formed (it overflows for beta ~
    # 1e100 at t ~ 1e308). Only sqrt(Q) is needed, and every factor in it
    # enters at half exponent.
    # Both P and sqrt(Q) carry the same factor g = n/(alpha A''), and hypot is
    # homogeneous, so g factors out of the whole root. Keep it factored: g
    # overflows on its own for alpha ~ 1e-308 and underflows for a subnormal
    # target, on roots that are perfectly representable either way. No fixed
    # multiplication order rescues both ends -- `n * bracket` underflows to zero
    # for n = 1e-163 against a 3.2e-172 bracket whose quotient by alpha = 1e-278
    # is an ordinary 3.2e-57 -- so the closing product is assembled with
    # exponent bookkeeping and materialized once.
    sqrt_time = math.sqrt(integration_time_s)
    linear_term = (
        target_snr * normalized_selfnoise / normalized_signal / integration_time_s
    )
    constant_term = math.hypot(
        math.sqrt(scale_rate) * math.sqrt(normalized_photon_var) / sqrt_time,
        scale_rate * systematic_root_over_time,
    )
    # 0.5 (P + sqrt(P^2 + 4Q)) == 0.5 P + hypot(P/2, sqrt(Q)). Written this way
    # nothing is squared and 2*sqrt(Q) is never formed, either of which would
    # overflow before the root does.
    bracket = 0.5 * linear_term + math.hypot(0.5 * linear_term, constant_term)
    contrast = _scaled_product(
        (target_snr, bracket), (normalized_signal, scale_planet)
    )
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
