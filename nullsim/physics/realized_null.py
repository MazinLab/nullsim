"""Monte-Carlo AO-realized null depth for a kernel-nulled chip.

The chip-only null depth ``|U @ E_star_perfect|^2`` is the floor that the
BFGS optimizer reaches under perfect inputs. Real observations sit on a
much shallower floor set by AO-residual piston fluctuations between
sub-apertures. For each Monte-Carlo realization k:

    phi_k_sub ~ N(0, (sigma_OPD_sub)^2)         [meters of piston OPD]
    E_k_sub   = E_star_sub * exp(i * 2*pi * phi_k_sub / lambda)
    null_k    = sum_{i in dark} |U @ E_k|_i^2 / sum_i |U @ E_k|_i^2

The result is a *dark-power fraction* — the downstream
``performance.standard`` leakage-rate computation multiplies it by the
total stellar rate, so it must stay in fraction-of-power units. A
    KILO-convention scalar (``mean(dark)/mean(bright)``, matching MKID
    ``null_depth_raw`` and KILO ``multiband.J.null_depth_dark``) is also
    returned via ``RealizedNullResult.null_depth_kilo`` for diagnostics that
    need that historical convention.

The band statistics follow the detected photons. With photon weights
``w_lambda`` (detected stellar rate per bin over the band total), each
paired realization ``(a, r)`` (azimuth, draw; the draw is shared across
wavelengths) has band leakage ``N_ar = sum_lambda w_lambda null_ar,lambda``.
``null_depth_dark = mean(N)`` and the fluctuation is the within-azimuth
spread ``sigma_MC^2 = mean_a Var_r(N_ar)``: static azimuth-to-azimuth
differences are a slow mean shift, not instability. ``null_instability =
sigma_MC / mean(N)`` is the band CV that drives the systematic-noise term.

Port of KILO ``kilo/transport.py`` ``_apply_realization_perturbations``
(lines 1489-1577) and ``azimuth_averaged_realized_null_depth`` (lines
1739-1883). nullsim's version omits the azimuth axis (fiber transport
losses do not vary with telescope azimuth here) and can optionally group
post-split chip inputs that share one physical aperture so duplicated
inputs receive the same piston and scintillation draw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "RealizedNullResult",
    "band_leakage_statistics",
    "bias_cross_moments",
    "bias_cross_variance",
    "normalized_wavelength_weights",
    "piston_covariance_with_common_mode_m2",
    "realized_null_depth_mc",
    "_chip_null_depth_kilo",
]


@dataclass(frozen=True)

class RealizedNullResult:
    """Bundle of statistics returned by :func:`realized_null_depth_mc`.

    All ``*_null_depth`` fields are dark-port power fractions
    ``sum_{i in dark} |U @ E|_i^2 / sum_i |U @ E|_i^2``. The
    :attr:`null_depth_dark` property is an explicit reporting alias.

    ``mean_null_depth`` is the photon-weighted band leakage mean,
    ``null_fluctuation_rms`` its within-azimuth RMS, and
    ``null_instability`` their ratio (the band CV). ``median_null_depth``
    and ``std_null_depth`` are pooled over (wavelength, azimuth, draw)
    for diagnostics only.

    ``n_azimuths`` and ``n_realizations_per_azimuth`` document the MC
    structure (KILO ``azimuth_averaged_realized_null_depth`` defaults
    to 6 x 200). ``n_samples`` is the total pool size summed across
    azimuths and wavelengths.
    """

    mean_null_depth: float
    median_null_depth: float
    std_null_depth: float
    null_instability: float
    ideal_null_depth: float
    n_samples: int
    n_azimuths: int = 1
    n_realizations_per_azimuth: int = 0
    # KILO-convention mean(dark)/mean(bright) scalars — populated alongside
    # the dark-fraction metrics because MKID/SNSPD ``null_depth_raw`` uses
    # the same per-port-rate convention. The dark-fraction fields above stay
    # in the downstream contrast pipeline; the KILO-convention fields are
    # for reporting only. Zero defaults keep back-compat with hand-built
    # results (e.g. the MC-disabled fallback in detection_curve).
    ideal_null_depth_kilo: float = 0.0
    mean_null_depth_kilo: float = 0.0
    null_fluctuation_rms: float = 0.0
    samples_by_wavelength: NDArray[np.float64] | None = field(
        default=None, repr=False, compare=False,
    )
    samples_by_wavelength_kilo: NDArray[np.float64] | None = field(
        default=None, repr=False, compare=False,
    )
    # Only with ``return_dark_fields=True``: per-draw dark fields minus the
    # static dark fields, each normalized by the square root of its own total
    # output power, shape (n_wl, n_azimuths * n_draws, n_dark); and the
    # linear dark-field response to an input phase, c_dk = U_dk a_k /
    # sqrt(P_static), shape (n_azimuths, n_wl, n_dark, M).
    dark_field_deviations: NDArray[np.complex128] | None = field(
        default=None, repr=False, compare=False,
    )
    bias_dark_response: NDArray[np.complex128] | None = field(
        default=None, repr=False, compare=False,
    )

    @property
    def null_depth_dark(self) -> float:
        """Alias for :attr:`mean_null_depth` (paperv3 reporting name)."""

        return self.mean_null_depth


def normalized_wavelength_weights(
    wavelength_weights: NDArray[np.float64] | None,
    n_wl: int,
) -> NDArray[np.float64]:
    """Return band weights summing to one; ``None`` means equal weights."""

    if wavelength_weights is None:
        return np.full(n_wl, 1.0 / n_wl, dtype=np.float64)
    w = np.asarray(wavelength_weights, dtype=np.float64)
    if w.shape != (n_wl,):
        raise ValueError(
            f"wavelength_weights must have shape ({n_wl},); got {w.shape}."
        )
    total = float(np.sum(w))
    if not np.all(np.isfinite(w)) or np.any(w < 0.0) or total <= 0.0:
        raise ValueError(
            "wavelength_weights must be finite, non-negative, and not all zero."
        )
    return w / total


def band_leakage_statistics(
    samples_by_wavelength: NDArray[np.float64],
    wavelength_weights: NDArray[np.float64],
    n_azimuths: int,
) -> tuple[float, float]:
    """Mean and within-azimuth RMS of the photon-weighted band leakage.

    ``samples_by_wavelength`` has shape ``(n_wl, n_azimuths * n_draws)``
    with azimuth-major columns, the draws paired across wavelength. Each
    column gives ``N = sum_lambda w_lambda L_lambda``. The RMS is
    ``sqrt(mean_a Var_r(N_ar))``: only draw-to-draw variation within an
    azimuth counts as instability.
    """

    samples = np.asarray(samples_by_wavelength, dtype=np.float64)
    band = np.asarray(wavelength_weights, dtype=np.float64) @ samples
    per_azimuth = band.reshape(int(n_azimuths), -1)
    return float(np.mean(band)), float(
        np.sqrt(np.mean(np.var(per_azimuth, axis=1)))
    )


def bias_cross_moments(
    result: RealizedNullResult,
    wavelength_weights: NDArray[np.float64],
) -> tuple[float, float]:
    """Moments ``(S, D)`` that set the bias cross term (see below).

    A static phase bias ``b_k`` on chip input ``k`` adds the dark field
    ``i sum_k c_dk b_k`` to linear order. Against a fluctuation draw with
    static-subtracted dark field ``dE_d(x)``, the band leakage gains
    ``sum_l w_l 2 Re sum_d conj(i sum_k c_dk b_k) dE_d(x) = 2 sum_k b_k g_k(x)``
    with ``g_k(x) = sum_l w_l sum_d Im(conj(c_dk) dE_d(x))``. Returns
    ``S = mean_a sum_k Var_r g_k`` (within azimuth, like the leakage RMS) and
    ``D = mean_a sum_l w_l sum_dk |c_dk|^2``, the bias leakage per unit
    ``b_k^2``.
    """

    deviations = result.dark_field_deviations
    response = result.bias_dark_response
    if deviations is None or response is None:
        raise ValueError(
            "bias_cross_moments needs a realized_null_depth_mc result run "
            "with return_dark_fields=True."
        )
    weights = np.asarray(wavelength_weights, dtype=np.float64)
    n_az = int(result.n_azimuths)
    n_wl, n_samples, n_dark = deviations.shape
    per_az = deviations.reshape(n_wl, n_az, n_samples // n_az, n_dark)
    g = np.einsum("l,aldk,lard->ark", weights, response.conj(), per_az).imag
    s_moment = float(np.mean(np.sum(np.var(g, axis=1), axis=1)))
    d_moment = float(
        np.mean(np.einsum("l,aldk->a", weights, np.abs(response) ** 2))
    )
    return s_moment, d_moment


def bias_cross_variance(
    result: RealizedNullResult,
    wavelength_weights: NDArray[np.float64],
    bias_leakage: float,
) -> float:
    """Band leakage variance from a static phase bias beating with fluctuations.

    The setpoint residual and dither are modeled as an unknown static phase
    offset ``b``, independent across chip input modes with
    ``E[b_k b_l] = beta^2 delta_kl``. Averaged over ``b``, the cross term of
    :func:`bias_cross_moments` has variance ``4 beta^2 S``, and the bias
    leakage is ``L_b = beta^2 D``, so the cross variance is
    ``4 L_b S / D``. This is exact to linear order in ``b`` and in the
    fluctuation, needs no equal-eigenvalue assumption, and couples both
    phase and amplitude fluctuations as the output row geometry dictates
    (amplitude drops out only where it sits in the orthogonal quadrature,
    as in a real Bracewell row). Derivation:
    tests/physics/notes/sensitivity_null.md.
    """

    if bias_leakage <= 0.0:
        return 0.0
    s_moment, d_moment = bias_cross_moments(result, wavelength_weights)
    return 4.0 * float(bias_leakage) * s_moment / d_moment if d_moment > 0.0 else 0.0


def _chip_null_depth(
    u: NDArray[np.complex128],
    field: NDArray[np.complex128],
    dark_indices: NDArray[np.intp],
) -> float:
    """Null depth for one wavelength: ``|U @ E|^2_dark / total``.

    This is the *dark-power fraction* convention — leakage power at the
    dark ports as a fraction of total chip output power. Performance and
    sensitivity stages compose this with the stellar rate to compute the
    dark-port leakage rate, so it must stay in fraction-of-power units
    for those downstream consumers.

    The historical KILO ``null_depth_dark`` convention is
    ``mean(dark)/mean(bright)`` — see :func:`_chip_null_depth_kilo` and the
    sensitivity payload's ``ideal_chip_null_depth_kilo`` for that form.
    """

    out = u @ field
    power = np.abs(out) ** 2
    total = float(np.sum(power))
    if total <= 0.0:
        return 0.0
    return float(np.sum(power[dark_indices])) / total


def _chip_null_depth_kilo(
    u: NDArray[np.complex128],
    field: NDArray[np.complex128],
    dark_indices: NDArray[np.intp],
) -> float:
    """KILO-convention null depth: ``mean(dark)/mean(bright)``.

    Mirrors KILO's ``null_depth_dark`` (per-port-rate averaging) and the
    nullsim MKID/SNSPD ``null_depth_raw`` extractors. Differs from the
    dark-power-fraction
    :func:`_chip_null_depth` by ``n_dark/n_bright`` for deep nulls (~3x
    for the Keck 6-bright/18-dark allocation).
    """

    out = u @ field
    power = np.abs(out) ** 2
    if dark_indices.size == 0:
        return 0.0
    bright_mask = np.ones(power.shape[0], dtype=bool)
    bright_mask[dark_indices] = False
    if not bright_mask.any():
        return 0.0
    mean_bright = float(power[bright_mask].mean())
    if mean_bright <= 0.0:
        return 0.0
    return float(power[dark_indices].mean()) / mean_bright


def piston_covariance_with_common_mode_m2(
    *,
    sigma_iid_m: float,
    sigma_diff_m: float,
    telescope_signs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Per-mode piston covariance: IID AO term plus a rank-1 inter-telescope mode.

    The intra-pupil AO residual is modeled as independent per-mode piston with
    RMS ``sigma_iid_m``. The inter-telescope differential disturbance (von
    Karman piston + vibration lines from the fringe tracker) is fully
    correlated within each telescope: telescope ``+1`` carries ``+sigma_diff/2``
    and telescope ``-1`` carries ``-sigma_diff/2`` of one common Gaussian, so
    the differential OPD between the two telescopes has RMS ``sigma_diff_m``.

    Args:
        sigma_iid_m: Independent per-mode piston OPD RMS in meters.
        sigma_diff_m: Differential OPD RMS between the two telescopes in meters.
        telescope_signs: Per-mode sign vector with entries in ``{-1, +1}``,
            ``+1`` for modes on one telescope and ``-1`` for the other.

    Returns:
        Symmetric positive-semidefinite covariance of shape ``(M, M)`` in m^2,
        where ``M = telescope_signs.size``.

    Raises:
        ValueError: If ``telescope_signs`` contains a value other than +/-1.
    """

    signs = np.asarray(telescope_signs, dtype=np.float64)
    if signs.ndim != 1 or signs.size == 0:
        raise ValueError(
            "piston_covariance_with_common_mode_m2: telescope_signs must be a "
            f"non-empty 1-D array, got shape {signs.shape}."
        )
    if not np.all(np.isin(signs, (-1.0, 1.0))):
        raise ValueError(
            "piston_covariance_with_common_mode_m2: telescope_signs must be "
            "+/-1 per mode."
        )
    return np.eye(signs.size, dtype=np.float64) * float(sigma_iid_m) ** 2 + np.outer(
        signs, signs
    ) * (float(sigma_diff_m) ** 2 / 4.0)


def realized_null_depth_mc(
    *,
    u_stack: NDArray[np.complex128],
    stellar_field: NDArray[np.complex128],
    dark_indices: NDArray[np.intp],
    ao_piston_opd_rms_per_sub_m: float | NDArray[np.float64],
    wavelengths_m: NDArray[np.float64],
    ao_piston_opd_covariance_m2: NDArray[np.float64] | None = None,
    n_realizations: int | None = None,
    n_realizations_per_azimuth: int | None = None,
    seed: int | None = None,
    scintillation_index_per_sub: float | NDArray[np.float64] = 0.0,
    mode_group_indices: NDArray[np.intp] | None = None,
    fiber_phase_noise_rms_rad: float = 0.0,
    wavelength_weights: NDArray[np.float64] | None = None,
    return_dark_fields: bool = False,
) -> RealizedNullResult:
    """Monte-Carlo realized null depth on the dark ports.

    The MC marginalizes simultaneously over (azimuth, wavelength, realization).
    Each draw is shared across wavelengths, so the band leakage of a draw is
    the photon-weighted sum of its per-bin leakages. The reported mean and
    CV are those of that band sum, with the variance taken within each
    azimuth (see :func:`band_leakage_statistics`). This replaces the KILO
    ``azimuth_averaged_realized_null_depth`` convention (transport.py L1739,
    ``natural_design.py`` L1100-1106) of averaging per-wavelength CVs of the
    azimuth-pooled sample, which is not the CV of the detected leakage
    (review 2026-11-07 Issue 18). Pre-transported stellar fields per
    azimuth are supplied by the caller (see
    :func:`nullsim.stages.delay_lines.geometric.per_aperture_transmission`).

    Args:
        u_stack: Chip transfer stack of shape ``(N_ports, M, n_wl)``.
        stellar_field: Complex stellar field at the chip input. Shape
            ``(M,)`` for single-azimuth/back-compat, ``(n_az, M)`` to pool
            samples across pre-computed per-azimuth fields, or
            ``(n_az, n_wl, M)`` when the pre-transported field is itself
            wavelength dependent.
        dark_indices: Dark output-port indices, shape ``(n_dark,)``.
        ao_piston_opd_rms_per_sub_m: Per-sub-aperture piston OPD RMS in
            meters. Scalar or shape ``(M,)``. Ignored for piston draws when
            ``ao_piston_opd_covariance_m2`` is supplied, but still accepted
            for payload/back-compat callers that record a scalar summary.
        ao_piston_opd_covariance_m2: Optional physical piston covariance in
            m^2. Shape ``(M, M)`` or, when ``mode_group_indices`` is set,
            ``(n_groups, n_groups)``. This supersedes independent Gaussian
            draws from ``ao_piston_opd_rms_per_sub_m`` and allows the
            fringe-tracker CRLB covariance to drive the realized-null MC.
        wavelengths_m: Wavelengths for each bin in ``u_stack``,
            shape ``(n_wl,)``.
        n_realizations_per_azimuth: Independent OPD draws per
            (azimuth, wavelength) bin. KILO default 200. Either this or
            ``n_realizations`` must be supplied; if both are given,
            ``n_realizations_per_azimuth`` wins.
        n_realizations: Back-compat alias for
            ``n_realizations_per_azimuth`` used by the single-azimuth
            callers from before Phase 1.
        seed: Optional base RNG seed. Each azimuth gets a deterministic
            sub-seed derived from ``seed`` so runs are reproducible.
        scintillation_index_per_sub: Per-sub-aperture amplitude
            fluctuation RMS (dimensionless intensity fraction). Scalar
            or shape ``(M,)``. If ``mode_group_indices`` is set, this may
            also have shape ``(n_groups,)``.
        mode_group_indices: Optional integer group index for each chip input
            mode, shape ``(M,)``. Modes in the same group share the same
            piston and scintillation realization; use this for duplicated
            inputs produced by a splitter after a physical aperture.
        fiber_phase_noise_rms_rad: IID per-mode-per-realization residual
            fiber phase noise (KILO ``FiberConfig.phase_noise_rms_rad``).
        return_dark_fields: Also return the static-subtracted dark fields
            and the bias response used by :func:`bias_cross_moments`.
        wavelength_weights: Detected stellar photon rate per bin (any
            normalization), shape ``(n_wl,)``. ``None`` weights bins equally.

    Returns:
        A :class:`RealizedNullResult` whose ``mean_null_depth`` ==
        ``null_depth_dark`` is the photon-weighted band dark-port power
        fraction averaged over azimuths and draws, and whose
        ``null_instability`` is the band CV ``null_fluctuation_rms /
        mean_null_depth``.
    """

    u = np.asarray(u_stack)
    if u.ndim != 3:
        raise ValueError(
            f"realized_null_depth_mc: u_stack must be 3D (N_ports, M, n_wl); "
            f"got {u.shape}."
        )
    n_ports, n_modes, n_wl = u.shape
    dark = np.asarray(dark_indices, dtype=np.intp)
    if np.any((dark < 0) | (dark >= n_ports)):
        raise ValueError("realized_null_depth_mc: dark_indices out of range.")

    field_in = np.asarray(stellar_field, dtype=np.complex128)
    if field_in.ndim == 1:
        if field_in.shape != (n_modes,):
            raise ValueError(
                f"realized_null_depth_mc: 1-D stellar_field shape "
                f"{field_in.shape} must equal (M,) = ({n_modes},)."
            )
        fields_per_az_wl = np.broadcast_to(
            field_in[np.newaxis, np.newaxis, :],
            (1, n_wl, n_modes),
        )
    elif field_in.ndim == 2:
        if field_in.shape[1] != n_modes:
            raise ValueError(
                f"realized_null_depth_mc: 2-D stellar_field shape "
                f"{field_in.shape} must end in M = {n_modes}."
            )
        fields_per_az_wl = np.broadcast_to(
            field_in[:, np.newaxis, :],
            (field_in.shape[0], n_wl, n_modes),
        )
    elif field_in.ndim == 3:
        if field_in.shape[1:] != (n_wl, n_modes):
            raise ValueError(
                f"realized_null_depth_mc: 3-D stellar_field shape "
                f"{field_in.shape} must equal (n_az, n_wl, M) with "
                f"(n_wl, M)=({n_wl}, {n_modes})."
            )
        fields_per_az_wl = field_in
    else:
        raise ValueError(
            f"realized_null_depth_mc: stellar_field must be 1-D (M,), "
            f"2-D (n_az, M), or 3-D (n_az, n_wl, M); got ndim={field_in.ndim}."
        )
    n_azimuths = int(fields_per_az_wl.shape[0])

    wavelengths = np.asarray(wavelengths_m, dtype=np.float64)
    if wavelengths.shape != (n_wl,):
        raise ValueError(
            f"realized_null_depth_mc: wavelengths_m shape {wavelengths.shape} "
            f"must equal (n_wl,) = ({n_wl},)."
        )
    band_weights = normalized_wavelength_weights(wavelength_weights, n_wl)
    if fiber_phase_noise_rms_rad < 0.0:
        raise ValueError(
            f"fiber_phase_noise_rms_rad must be >= 0, got "
            f"{fiber_phase_noise_rms_rad}."
        )

    if n_realizations_per_azimuth is None:
        if n_realizations is None:
            raise ValueError(
                "realized_null_depth_mc: either n_realizations or "
                "n_realizations_per_azimuth must be supplied."
            )
        n_per_az = int(n_realizations)
    else:
        n_per_az = int(n_realizations_per_azimuth)
    if n_per_az < 0:
        raise ValueError(
            f"realized_null_depth_mc: n_realizations_per_azimuth must be "
            f">= 0 (got {n_per_az}). Use 0 explicitly to disable the MC pass."
        )

    group_indices = _validate_mode_group_indices(
        mode_group_indices,
        n_modes=n_modes,
    )
    piston_covariance = _validate_piston_covariance(
        ao_piston_opd_covariance_m2,
        n_modes=n_modes,
        group_indices=group_indices,
    )
    sigma, sigma_group_indices = _sigma_for_mode_or_group(
        ao_piston_opd_rms_per_sub_m,
        n_modes=n_modes,
        group_indices=group_indices,
        name="ao_piston_opd_rms_per_sub_m",
    )
    # Chip-only (ideal) null depth, per wavelength, averaged across azimuths.
    # Two conventions reported side-by-side: dark-power fraction (the
    # downstream contrast pipeline's input) and KILO mean(dark)/mean(bright)
    # (a diagnostic convention used by detector null-depth payloads).
    ideal_per_wl = np.array(
        [
            np.mean([
                _chip_null_depth(u[:, :, k], fields_per_az_wl[a, k], dark)
                for a in range(n_azimuths)
            ])
            for k in range(n_wl)
        ],
        dtype=np.float64,
    )
    ideal_per_wl_kilo = np.array(
        [
            np.mean([
                _chip_null_depth_kilo(u[:, :, k], fields_per_az_wl[a, k], dark)
                for a in range(n_azimuths)
            ])
            for k in range(n_wl)
        ],
        dtype=np.float64,
    )
    ideal_null = float(band_weights @ ideal_per_wl)
    ideal_null_kilo = float(band_weights @ ideal_per_wl_kilo)

    if n_per_az <= 0:
        return RealizedNullResult(
            mean_null_depth=ideal_null,
            median_null_depth=ideal_null,
            std_null_depth=0.0,
            null_instability=0.0,
            ideal_null_depth=ideal_null,
            ideal_null_depth_kilo=ideal_null_kilo,
            # Chip-only path: with no AO perturbation MC, the KILO-style
            # mean(dark)/mean(bright) realized null collapses to the
            # ideal KILO null. Default 0.0 (the dataclass default) would
            # silently mis-report the KILO-convention scalar as zero in the
            # no-AO path used by HWO and other space pipelines.
            mean_null_depth_kilo=ideal_null_kilo,
            n_samples=0,
            n_azimuths=n_azimuths,
            n_realizations_per_azimuth=0,
        )

    sigma_I, sigma_i_group_indices = _sigma_for_mode_or_group(
        scintillation_index_per_sub,
        n_modes=n_modes,
        group_indices=group_indices,
        name="scintillation_index_per_sub",
    )
    if np.any(sigma_I > 0.0):
        var_log = np.log1p(sigma_I ** 2)
        amp_std = np.sqrt(var_log)
        amp_mean = -0.5 * var_log
    else:
        amp_std = None
        amp_mean = None

    samples_by_wavelength = np.empty(
        (n_wl, n_azimuths * n_per_az),
        dtype=np.float64,
    )
    samples_by_wavelength_kilo = np.empty_like(samples_by_wavelength)
    dark_field_deviations = (
        np.empty((n_wl, n_azimuths * n_per_az, dark.size), dtype=np.complex128)
        if return_dark_fields
        else None
    )
    bias_dark_response = (
        np.empty((n_azimuths, n_wl, dark.size, n_modes), dtype=np.complex128)
        if return_dark_fields
        else None
    )
    null_per_wl_kilo = np.empty(n_wl, dtype=np.float64)
    # Accumulate all (wl, az, draw) samples for the *pooled* median/std.
    # Earlier the dataclass reported mean-of-per-wl-medians and mean-of-
    # per-wl-stds under the names ``median_null_depth`` / ``std_null_depth``
    # — semantically distinct from the pooled aggregates the field names
    # suggest. Cost: ~8 bytes × n_wl × n_az × n_per_az (negligible for
    # realistic MC sizes).
    pooled_all_samples: list[np.ndarray] = []
    total_pool_size = 0

    # Seed derivation: each azimuth gets a stable child stream. One physical
    # realization (an OPD draw, a fiber-phase draw, a scintillation draw)
    # is shared ACROSS wavelength bins — piston is an achromatic OPD, so a
    # single realization must produce correlated phase 2*pi*OPD/lambda_k at
    # every bin. The previous per-(wavelength, azimuth) draw made the
    # pooled std/median represent artificially independent wavelength
    # samples instead of broadband leakage fluctuation, and cost ~n_wl x
    # the RNG/draw work (review 2026-06-10 Issue 27d). The pairing is also
    # what makes the band-summed leakage of a draw well defined.
    _AZIMUTH_SEED_STRIDE = 10_007

    draws_per_az: list[
        tuple[np.ndarray, np.ndarray | None, np.ndarray | None]
    ] = []
    for az_idx in range(n_azimuths):
        sub_seed = (
            None
            if seed is None
            else int(seed) + az_idx * _AZIMUTH_SEED_STRIDE
        )
        rng = np.random.default_rng(sub_seed)
        if piston_covariance is None:
            opd_draw_shape = (n_per_az, sigma.shape[0])
            opd_draws = rng.normal(
                loc=0.0,
                scale=sigma[np.newaxis, :],
                size=opd_draw_shape,
            )
            if sigma_group_indices is not None:
                opd_draws = opd_draws[:, sigma_group_indices]
        else:
            opd_draws = _draw_correlated_pistons(
                rng=rng,
                covariance_m2=piston_covariance[0],
                n_draws=n_per_az,
            )
            if piston_covariance[1] is not None:
                opd_draws = opd_draws[:, piston_covariance[1]]
        fiber_phase_draws: np.ndarray | None = None
        if fiber_phase_noise_rms_rad > 0.0:
            fiber_phase_draws = rng.normal(
                loc=0.0,
                scale=fiber_phase_noise_rms_rad,
                size=(n_per_az, n_modes),
            )
        log_I_draws: np.ndarray | None = None
        if amp_std is not None:
            log_I_draws = rng.normal(
                loc=amp_mean[np.newaxis, :],
                scale=amp_std[np.newaxis, :],
                size=(n_per_az, amp_std.shape[0]),
            )
            if sigma_i_group_indices is not None:
                log_I_draws = log_I_draws[:, sigma_i_group_indices]
        draws_per_az.append((opd_draws, fiber_phase_draws, log_I_draws))

    for k in range(n_wl):
        u_k = u[:, :, k]
        wl = float(wavelengths[k])
        pooled = np.empty(n_azimuths * n_per_az, dtype=np.float64)
        pooled_kilo = np.empty_like(pooled)

        for az_idx in range(n_azimuths):
            field_az = fields_per_az_wl[az_idx, k]
            opd_draws, fiber_phase_draws, log_I_draws = draws_per_az[az_idx]
            phase = 2.0 * np.pi * opd_draws / wl
            if fiber_phase_draws is not None:
                phase = phase + fiber_phase_draws

            perturbed = field_az[np.newaxis, :] * np.exp(1j * phase)
            if log_I_draws is not None:
                perturbed = perturbed * np.exp(0.5 * log_I_draws)

            out = np.einsum("ij,nj->ni", u_k, perturbed)
            power = np.abs(out) ** 2
            total = np.sum(power, axis=1)
            dark_power = np.sum(power[:, dark], axis=1)
            ratio = np.where(
                total > 0.0,
                dark_power / np.where(total > 0.0, total, 1.0),
                0.0,
            )
            # KILO-convention ratio mean(dark)/mean(bright) per realization.
            bright_mask = np.ones(power.shape[1], dtype=bool)
            bright_mask[dark] = False
            if dark.size > 0 and bright_mask.any():
                mean_dark = power[:, dark].mean(axis=1)
                mean_bright = power[:, bright_mask].mean(axis=1)
                ratio_kilo = np.where(
                    mean_bright > 0.0,
                    mean_dark / np.where(mean_bright > 0.0, mean_bright, 1.0),
                    0.0,
                )
            else:
                ratio_kilo = np.zeros(power.shape[0], dtype=np.float64)
            start = az_idx * n_per_az
            pooled[start : start + n_per_az] = ratio
            pooled_kilo[start : start + n_per_az] = ratio_kilo
            if return_dark_fields:
                static_out = u_k @ field_az
                static_norm = math.sqrt(float(np.sum(np.abs(static_out) ** 2)))
                scale = np.where(total > 0.0, np.sqrt(np.where(total > 0.0, total, 1.0)), 1.0)
                dark_field_deviations[k, start : start + n_per_az] = (
                    out[:, dark] / scale[:, np.newaxis]
                    - static_out[dark][np.newaxis, :] / static_norm
                )
                bias_dark_response[az_idx, k] = (
                    u_k[dark, :] * field_az[np.newaxis, :] / static_norm
                )

        null_per_wl_kilo[k] = float(np.mean(pooled_kilo))
        total_pool_size += pooled.size
        samples_by_wavelength[k] = pooled
        samples_by_wavelength_kilo[k] = pooled_kilo
        pooled_all_samples.append(pooled.copy())

    all_samples = (
        np.concatenate(pooled_all_samples)
        if pooled_all_samples
        else np.empty(0, dtype=np.float64)
    )
    pooled_median = float(np.median(all_samples)) if all_samples.size else 0.0
    pooled_std = float(np.std(all_samples)) if all_samples.size else 0.0
    samples_by_wavelength.setflags(write=False)
    samples_by_wavelength_kilo.setflags(write=False)
    band_mean, band_rms = band_leakage_statistics(
        samples_by_wavelength, band_weights, n_azimuths,
    )

    return RealizedNullResult(
        mean_null_depth=band_mean,
        median_null_depth=pooled_median,
        std_null_depth=pooled_std,
        null_instability=band_rms / band_mean if band_mean > 0.0 else 0.0,
        null_fluctuation_rms=band_rms,
        ideal_null_depth=ideal_null,
        ideal_null_depth_kilo=ideal_null_kilo,
        mean_null_depth_kilo=float(band_weights @ null_per_wl_kilo),
        n_samples=total_pool_size,
        n_azimuths=n_azimuths,
        n_realizations_per_azimuth=n_per_az,
        samples_by_wavelength=samples_by_wavelength,
        samples_by_wavelength_kilo=samples_by_wavelength_kilo,
        dark_field_deviations=dark_field_deviations,
        bias_dark_response=bias_dark_response,
    )


def _validate_mode_group_indices(
    mode_group_indices: NDArray[np.intp] | None,
    *,
    n_modes: int,
) -> NDArray[np.intp] | None:
    if mode_group_indices is None:
        return None
    indices = np.asarray(mode_group_indices, dtype=np.intp)
    if indices.shape != (n_modes,):
        raise ValueError(
            "realized_null_depth_mc: mode_group_indices must have shape "
            f"(M,) = ({n_modes},); got {indices.shape}."
        )
    if np.any(indices < 0):
        raise ValueError(
            "realized_null_depth_mc: mode_group_indices must be non-negative."
        )
    unique = np.unique(indices)
    expected = np.arange(unique.size, dtype=np.intp)
    if not np.array_equal(unique, expected):
        raise ValueError(
            "realized_null_depth_mc: mode_group_indices must use contiguous "
            "group ids starting at 0."
        )
    return indices


def _validate_piston_covariance(
    covariance_m2: NDArray[np.float64] | None,
    *,
    n_modes: int,
    group_indices: NDArray[np.intp] | None,
) -> tuple[NDArray[np.float64], NDArray[np.intp] | None] | None:
    if covariance_m2 is None:
        return None
    raw = np.asarray(covariance_m2, dtype=np.float64)
    n_groups = int(np.max(group_indices)) + 1 if group_indices is not None else 0
    allowed_shapes = [(n_modes, n_modes)]
    if group_indices is not None:
        allowed_shapes.append((n_groups, n_groups))
    if raw.shape not in allowed_shapes:
        allowed = " or ".join(str(shape) for shape in allowed_shapes)
        raise ValueError(
            "realized_null_depth_mc: ao_piston_opd_covariance_m2 must have "
            f"shape {allowed}; got {raw.shape}."
        )
    if not np.all(np.isfinite(raw)):
        raise ValueError(
            "realized_null_depth_mc: ao_piston_opd_covariance_m2 must be finite."
        )
    covariance = 0.5 * (raw + raw.T)
    if not np.allclose(raw, raw.T, rtol=1.0e-10, atol=1.0e-30):
        raise ValueError(
            "realized_null_depth_mc: ao_piston_opd_covariance_m2 must be symmetric."
        )
    eigenvalues = np.linalg.eigvalsh(covariance)
    scale = max(float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0, 1.0e-300)
    if eigenvalues.size and float(np.min(eigenvalues)) < -1.0e-10 * scale:
        raise ValueError(
            "realized_null_depth_mc: ao_piston_opd_covariance_m2 must be "
            "positive semidefinite."
        )
    if group_indices is not None and raw.shape == (n_groups, n_groups):
        return covariance.astype(np.float64, copy=True), group_indices
    return covariance.astype(np.float64, copy=True), None


def _draw_correlated_pistons(
    *,
    rng: np.random.Generator,
    covariance_m2: NDArray[np.float64],
    n_draws: int,
) -> NDArray[np.float64]:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_m2)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    transform = eigenvectors * np.sqrt(eigenvalues)[np.newaxis, :]
    return rng.normal(0.0, 1.0, size=(n_draws, covariance_m2.shape[0])) @ transform.T


def _sigma_for_mode_or_group(
    value: float | NDArray[np.float64],
    *,
    n_modes: int,
    group_indices: NDArray[np.intp] | None,
    name: str,
) -> tuple[NDArray[np.float64], NDArray[np.intp] | None]:
    raw = np.asarray(value, dtype=np.float64)
    if group_indices is None:
        sigma = np.broadcast_to(raw, (n_modes,)).astype(np.float64, copy=False)
        _validate_sigma(sigma, name=name)
        return sigma, None

    n_groups = int(np.max(group_indices)) + 1 if group_indices.size else 0
    if raw.ndim == 0:
        sigma = np.full(n_groups, float(raw), dtype=np.float64)
    elif raw.shape == (n_groups,):
        sigma = raw.astype(np.float64, copy=False)
    elif raw.shape == (n_modes,):
        sigma = np.empty(n_groups, dtype=np.float64)
        for group_index in range(n_groups):
            values = raw[group_indices == group_index]
            if not np.allclose(values, values[0], rtol=1.0e-12, atol=0.0):
                raise ValueError(
                    f"realized_null_depth_mc: {name} values must match within "
                    "each mode_group_indices group."
                )
            sigma[group_index] = values[0]
    else:
        raise ValueError(
            f"realized_null_depth_mc: {name} must be scalar, shape "
            f"(M,) = ({n_modes},), or shape (n_groups,) = ({n_groups},); "
            f"got {raw.shape}."
        )
    _validate_sigma(sigma, name=name)
    return sigma, group_indices


def _validate_sigma(sigma: NDArray[np.float64], *, name: str) -> None:
    if not np.all(np.isfinite(sigma)):
        raise ValueError(f"realized_null_depth_mc: {name} must be finite.")
    if np.any(sigma < 0.0):
        raise ValueError(f"realized_null_depth_mc: {name} must be non-negative.")
