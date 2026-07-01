"""Kolmogorov turbulence primitives for the nullsim physics layer.

Pure-math primitives for the Kolmogorov atmospheric model used by downstream
atmosphere/AO stages. No SimulationState, no stage imports, no I/O. Function
signatures are dimensioned: all distances are meters, wavelengths are meters,
phases are radians.

The two canonical Kolmogorov constants pinned by this module are:

- ``KOLMOGOROV_PHASE_SF_COEFFICIENT = 6.88``: the phase structure function
  amplitude, ``D_phi(r) = 6.88 (r / r0)^(5/3)`` (Fried 1965; Roddier 1981).
  KILO uses the same numeric value as ``C_TEMP`` in its temporal-error
  formula (``kilo/atmosphere.py``); the coefficient arises from the same
  Kolmogorov spectrum.
- ``NOLL_TOTAL_VARIANCE_COEFFICIENT = 1.0299``: the piston-included
  full-aperture wavefront phase variance over a circular aperture of
  diameter D, ``sigma^2 = 1.0299 (D / r0)^(5/3)`` (Noll 1976, Table I).
  Neither KILO nor NAYRA expose this constant directly — both jump to
  budget breakdowns (fitting, temporal, WFS noise) for AO-residual work.
  See ``tests/physics/notes/kolmogorov.md`` for the rationale.

The Fried wavelength scaling ``r0 propto lambda^(6/5)`` matches KILO's
``scale_r0`` (KILO/NAYRA atmosphere.py, ``r0_500nm * (lambda/500e-9)^1.2``).

Non-finite r0 inputs (NaN) propagate as NaN in the output. Zero or negative
r0 is physically meaningless and raises ``ValueError``. Negative aperture
diameters and negative separations also raise ``ValueError``.

References:
    Noll, R. J. "Zernike polynomials and atmospheric turbulence."
        J. Opt. Soc. Am. 66, 207 (1976).
    Fried, D. L. "Statistics of a geometric representation of wavefront
        distortion." J. Opt. Soc. Am. 55, 1427 (1965).
    Roddier, F. "The effects of atmospheric turbulence in optical
        astronomy." Prog. Opt. 19, 281 (1981).
    Hardy, J. W. "Adaptive Optics for Astronomical Telescopes" (1998).
"""

from __future__ import annotations

from typing import TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

# ── Canonical normalization constants ────────────────────────────────────

#: Phase structure function coefficient (Fried 1965; Roddier 1981).
#:
#:     D_phi(r) = KOLMOGOROV_PHASE_SF_COEFFICIENT * (r / r0)^(5/3)   [rad^2]
#:
#: KILO/NAYRA reuse the same numeric value (``C_TEMP = 6.88``) in
#: ``kilo/atmosphere.py`` for the AO temporal-error formula; both stem
#: from the Kolmogorov spectrum.
KOLMOGOROV_PHASE_SF_COEFFICIENT: float = 6.88

#: Noll 1976 Table I piston-included total wavefront phase variance over
#: a circular aperture of diameter D (sum across all Zernike modes):
#:
#:     sigma^2 = NOLL_TOTAL_VARIANCE_COEFFICIENT * (D / r0)^(5/3)   [rad^2]
#:
#: This constant does not appear explicitly in KILO/NAYRA — both packages
#: track AO error sub-budgets (fitting, temporal, WFS noise) instead of
#: the raw Kolmogorov full-aperture variance. Adopting Noll's canonical
#: value here means downstream nullsim stages can decompose into the
#: same sub-budgets KILO uses, with this constant as the integral check.
NOLL_TOTAL_VARIANCE_COEFFICIENT: float = 1.0299

#: Noll 1976 Table II piston-removed coefficient (first Zernike removed).
#: Useful when the full aperture is implicitly piston-corrected (e.g. a
#: tip-tilt-only AO mode or a fringe-tracker that removes piston).
NOLL_PISTON_REMOVED_COEFFICIENT: float = 0.582

#: Reference wavelength for the Fried r0 scaling. 500 nm matches KILO's
#: ``scale_r0`` convention and the standard atmospheric site characterisation.
R0_REFERENCE_WAVELENGTH_M: float = 500.0e-9

#: Greenwood/Fried relation tau0 = 0.314 r0 / v_bar (Greenwood 1977).
TAU0_WIND_COEFFICIENT: float = 0.314

# Public exponent: r0(lambda) propto lambda^(6/5).
_FRIED_WAVELENGTH_EXPONENT: float = 6.0 / 5.0
_FIVE_THIRDS: float = 5.0 / 3.0


# ── Scalar/array type plumbing ───────────────────────────────────────────

T = TypeVar("T", bound=float | NDArray[np.floating])


def _as_floating(value: ArrayLike, name: str) -> float | NDArray[np.floating]:
    """Coerce ``value`` to a Python float or a floating-dtype ndarray.

    Preserves float32 inputs as float32 (dtype invariance). Python ints
    promote to float64 to keep arithmetic well-defined.

    Raises ValueError if ``value`` cannot be interpreted as numeric.
    """
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}.") from exc

    if array.dtype.kind not in ("f", "i", "u"):
        raise ValueError(
            f"{name} must be a real number or real-valued array, got dtype "
            f"{array.dtype}."
        )

    if array.dtype.kind in ("i", "u"):
        array = array.astype(np.float64)

    if array.ndim == 0:
        return float(array)
    return array


def _require_positive_r0(r0_m: ArrayLike) -> float | NDArray[np.floating]:
    """Validate r0: NaN propagates, zero/negative/±inf raises.

    The physical meaning of r0 <= 0 is undefined (negative Fried parameter
    has no physical interpretation; r0 = 0 corresponds to infinite phase
    variance). NaN is interpreted as "missing measurement" and propagates.
    ±inf has no physical interpretation either — pre-fix the validator
    only caught the ``isfinite + (v <= 0)`` case and let ±inf through,
    producing meaningless outputs (e.g. ``r0_scaled(-inf, ...) -> -inf``).
    Ultra-review Medium #4.
    """
    value = _as_floating(r0_m, "r0_m")

    if isinstance(value, np.ndarray):
        bad_mask = (~np.isnan(value)) & (~np.isfinite(value) | (value <= 0.0))
        if np.any(bad_mask):
            raise ValueError(
                "r0_m must be positive and finite where not NaN; got "
                "non-positive or non-finite (±inf) entries."
            )
    else:
        if not np.isnan(value) and (not np.isfinite(value) or value <= 0.0):
            raise ValueError(
                f"r0_m must be positive and finite (or NaN), got {value}."
            )
    return value


def _require_positive(value: ArrayLike, name: str) -> float | NDArray[np.floating]:
    """Validate a strictly-positive finite quantity (aperture, wavelength)."""
    arr = _as_floating(value, name)

    if isinstance(arr, np.ndarray):
        if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
            raise ValueError(
                f"{name} must be finite and positive everywhere; got entries "
                f"that are non-positive or non-finite."
            )
    else:
        if not np.isfinite(arr) or arr <= 0.0:
            raise ValueError(f"{name} must be finite and positive, got {arr}.")
    return arr


def _require_nonnegative(value: ArrayLike, name: str) -> float | NDArray[np.floating]:
    """Validate a non-negative finite quantity (e.g. separation)."""
    arr = _as_floating(value, name)
    if isinstance(arr, np.ndarray):
        if np.any(~np.isfinite(arr)) or np.any(arr < 0.0):
            raise ValueError(
                f"{name} must be finite and non-negative everywhere."
            )
    else:
        if not np.isfinite(arr) or arr < 0.0:
            raise ValueError(f"{name} must be finite and non-negative, got {arr}.")
    return arr


# ── Wavelength scaling ───────────────────────────────────────────────────

def r0_scaled(
    r0_ref_m: ArrayLike,
    wavelength_m: ArrayLike,
    wavelength_ref_m: ArrayLike = R0_REFERENCE_WAVELENGTH_M,
) -> float | NDArray[np.floating]:
    """Scale the Fried parameter to a target wavelength.

    Kolmogorov turbulence gives ``r0(lambda) = r0(lambda_ref)
    * (lambda / lambda_ref)^(6/5)``. Matches KILO's ``scale_r0`` /
    ``scale_tau0`` / ``scale_theta0`` convention (KILO ``atmosphere.py``).

    Args:
        r0_ref_m: Fried parameter at ``wavelength_ref_m`` in meters. NaN
            propagates; zero or negative raises ``ValueError``.
        wavelength_m: Target wavelength in meters.
        wavelength_ref_m: Reference wavelength for r0 in meters. Defaults
            to 500 nm to match KILO/NAYRA convention.

    Returns:
        Fried parameter at ``wavelength_m`` in meters. Returns a float
        when all inputs are scalars; otherwise a numpy array following
        numpy broadcasting rules. Preserves float32 inputs as float32.
    """
    r0_value = _require_positive_r0(r0_ref_m)
    wavelength_value = _require_positive(wavelength_m, "wavelength_m")
    wavelength_ref_value = _require_positive(wavelength_ref_m, "wavelength_ref_m")
    ratio = np.divide(wavelength_value, wavelength_ref_value)
    return r0_value * np.power(ratio, _FRIED_WAVELENGTH_EXPONENT)


def effective_wind_speed_m_s(*, r0_500nm_m: float, tau0_500nm_s: float) -> float:
    """5/3-moment effective wind speed implied by (r0, tau0) at 500 nm.

    Both r0 and tau0 scale as lambda^(6/5), so their ratio, and hence the
    effective wind, is wavelength independent.
    """

    if r0_500nm_m <= 0.0 or tau0_500nm_s <= 0.0:
        raise ValueError(
            "r0_500nm_m and tau0_500nm_s must be positive, "
            f"got {r0_500nm_m}, {tau0_500nm_s}"
        )
    return TAU0_WIND_COEFFICIENT * r0_500nm_m / tau0_500nm_s


# ── Phase structure function (Fried 1965 / Roddier 1981) ─────────────────

def phase_structure_function(
    separation_m: ArrayLike,
    r0_m: ArrayLike,
) -> float | NDArray[np.floating]:
    """Kolmogorov phase structure function (no outer scale).

    ``D_phi(r) = 6.88 (r / r0)^(5/3)`` rad^2 (Fried 1965; Roddier 1981).
    Assumes a single-layer Kolmogorov spectrum with an infinite outer
    scale L0. Finite-L0 corrections (von Karman) are an extension point
    for a later von Karman primitive — see ``notes/kolmogorov.md``.

    Args:
        separation_m: Two-point separation r in meters. Non-negative.
        r0_m: Fried parameter at the wavelength matching the returned
            phase variance (i.e. the user is responsible for calling
            ``r0_scaled`` first if r0 was measured at a different
            wavelength). NaN propagates; zero or negative raises.

    Returns:
        Phase structure function in rad^2 at the wavelength implied by
        r0_m. Preserves float32 inputs as float32.
    """
    separation_value = _require_nonnegative(separation_m, "separation_m")
    r0_value = _require_positive_r0(r0_m)
    ratio = np.divide(separation_value, r0_value)
    return KOLMOGOROV_PHASE_SF_COEFFICIENT * np.power(ratio, _FIVE_THIRDS)


# ── Phase variance over a circular aperture (Noll 1976) ──────────────────

def phase_variance(
    aperture_diameter_m: ArrayLike,
    r0_m: ArrayLike,
    *,
    piston_removed: bool = False,
) -> float | NDArray[np.floating]:
    """Wavefront phase variance over a circular aperture (Noll 1976).

    Piston-included form (default):
        ``sigma^2 = 1.0299 (D / r0)^(5/3)`` rad^2 (Noll 1976, Table I).

    Piston-removed form (``piston_removed=True``):
        ``sigma^2 = 0.582 (D / r0)^(5/3)`` rad^2 (Noll 1976, Table II,
        residual after the first Zernike is removed). Use this form when
        the aperture is implicitly piston-corrected (e.g. by a single-mode
        fringe tracker that removes the mean phase).

    Architect note: SPEC §15.2 lists the signature as
    ``phase_variance(r0_m, wavelength_m)``, but the aperture diameter
    is a load-bearing physical input (the variance is unbounded for an
    infinite aperture). This implementation takes
    ``(aperture_diameter_m, r0_m)`` instead; r0 already encodes the
    wavelength (use ``r0_scaled`` to wavelength-scale if needed). See
    ``tests/physics/notes/kolmogorov.md`` for the divergence rationale.

    Args:
        aperture_diameter_m: Circular aperture diameter D in meters.
            Must be finite and positive.
        r0_m: Fried parameter at the same wavelength the returned phase
            variance is in. NaN propagates; zero/negative raises.
        piston_removed: When True, use the piston-removed Noll coefficient
            (0.582). Default False uses the piston-included coefficient
            (1.0299).

    Returns:
        Phase variance in rad^2. ``sigma^2 -> 0`` as ``r0 -> infinity``
        for fixed D, as expected. Preserves float32 inputs as float32.
    """
    aperture_value = _require_positive(aperture_diameter_m, "aperture_diameter_m")
    r0_value = _require_positive_r0(r0_m)
    coefficient = (
        NOLL_PISTON_REMOVED_COEFFICIENT
        if piston_removed
        else NOLL_TOTAL_VARIANCE_COEFFICIENT
    )
    ratio = np.divide(aperture_value, r0_value)
    return coefficient * np.power(ratio, _FIVE_THIRDS)


# ── OPD RMS conversion ──────────────────────────────────────────────────

def opd_rms_m(
    aperture_diameter_m: ArrayLike,
    r0_m: ArrayLike,
    wavelength_m: ArrayLike,
    *,
    piston_removed: bool = False,
) -> float | NDArray[np.floating]:
    """Wavefront OPD RMS over a circular aperture (Kolmogorov, Noll 1976).

    Converts ``phase_variance`` to an optical-path-difference RMS in
    meters. Matches KILO's ``ao_piston_opd_rms`` conversion:
    ``sigma_OPD = sqrt(sigma_phi^2) * lambda / (2 pi)``.

    The OPD RMS is **wavelength-independent** for a given r0(wavelength)
    and aperture — the wavelength factor cancels the wavelength scaling
    of r0 (``r0 propto lambda^(6/5)``) under the (D / r0)^(5/3) term and
    the lambda/2pi conversion. This function accepts ``wavelength_m``
    explicitly so callers see the dimensioned conversion and so that
    when r0_m is passed at a wavelength that does not match
    wavelength_m, the result reflects the user's choice rather than
    silently going wrong.

    Architect note: SPEC §15.2 lists ``opd_rms_m(r0_m, wavelength_m)``;
    this implementation adds the required aperture diameter argument
    for the same reason as ``phase_variance`` above.

    Args:
        aperture_diameter_m: Circular aperture diameter D in meters.
        r0_m: Fried parameter at ``wavelength_m`` in meters.
        wavelength_m: Wavelength in meters for the phase-to-OPD
            conversion. Must be finite and positive.
        piston_removed: When True, use the piston-removed Noll
            coefficient (see ``phase_variance``).

    Returns:
        OPD RMS in meters.
    """
    wavelength_value = _require_positive(wavelength_m, "wavelength_m")
    sigma2_rad2 = phase_variance(
        aperture_diameter_m, r0_m, piston_removed=piston_removed,
    )
    return np.sqrt(sigma2_rad2) * wavelength_value / (2.0 * np.pi)


# ── Marechal Strehl approximation ───────────────────────────────────────

def strehl_marechal(
    sigma_opd_m: ArrayLike,
    wavelength_m: ArrayLike,
) -> float | NDArray[np.floating]:
    """Marechal Strehl approximation from an OPD RMS.

    ``S = exp(- (2 pi sigma_OPD / lambda)^2)``

    Equivalent to ``exp(-sigma_phi^2)`` (KILO's ``strehl_ratio``) with
    the phase variance written in terms of the OPD RMS. Valid for
    ``sigma_phi^2 < ~2 rad^2`` (S > ~0.14); beyond that the Marechal
    approximation underestimates the true Strehl but is still a useful
    monotonic indicator.

    Args:
        sigma_opd_m: OPD RMS in meters. Non-negative. NaN propagates.
        wavelength_m: Wavelength in meters. Must be finite and positive.

    Returns:
        Strehl ratio in [0, 1]. Preserves float32 inputs as float32.
    """
    sigma_value = _require_nonnegative_propagating(sigma_opd_m, "sigma_opd_m")
    wavelength_value = _require_positive(wavelength_m, "wavelength_m")
    phase_arg = 2.0 * np.pi * np.divide(sigma_value, wavelength_value)
    return np.exp(-(phase_arg * phase_arg))


def _require_nonnegative_propagating(
    value: ArrayLike, name: str
) -> float | NDArray[np.floating]:
    """Non-negative but allow NaN to propagate (for OPD RMS).

    Negative OPD RMS is meaningless (RMS is non-negative by construction)
    but NaN should propagate so downstream code can drop unobserved
    sub-apertures naturally. ±inf is rejected — pre-fix the validator
    only caught the ``isfinite + (v < 0)`` case and let ±inf through,
    producing meaningless outputs like ``strehl_marechal(-inf, ...) -> 0.0``.
    Ultra-review Medium #4.
    """
    arr = _as_floating(value, name)
    if isinstance(arr, np.ndarray):
        bad_mask = (~np.isnan(arr)) & (~np.isfinite(arr) | (arr < 0.0))
        if np.any(bad_mask):
            raise ValueError(
                f"{name} must be non-negative and finite where not NaN (RMS >= 0)."
            )
    else:
        if not np.isnan(arr) and (not np.isfinite(arr) or arr < 0.0):
            raise ValueError(
                f"{name} must be non-negative and finite (or NaN), got {arr}."
            )
    return arr


__all__ = [
    "KOLMOGOROV_PHASE_SF_COEFFICIENT",
    "NOLL_TOTAL_VARIANCE_COEFFICIENT",
    "NOLL_PISTON_REMOVED_COEFFICIENT",
    "R0_REFERENCE_WAVELENGTH_M",
    "TAU0_WIND_COEFFICIENT",
    "effective_wind_speed_m_s",
    "opd_rms_m",
    "phase_structure_function",
    "phase_variance",
    "r0_scaled",
    "strehl_marechal",
]
