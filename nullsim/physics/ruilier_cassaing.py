"""Ruilier-Cassaing single-mode fiber coupling primitives.

This module implements the pure math port called out in SPEC §15.2:
physics primitives stay independent of pipeline state, stages, config, and
transport losses. The coupling model follows the Gaussian-mode overlap used
by Ruilier (1999, SPIE 3350) and Ruilier & Cassaing (2001), with the same
Strehl scaling convention used by Coudé du Foresto et al. (2000).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import j0


OPTIMAL_PUPIL_RADIUS_TO_MODE_FIELD_RADIUS_RATIO = 1.120906422778534
"""Unobstructed optimum for aperture radius divided by Gaussian mode radius."""

UNOBSTRUCTED_GEOMETRIC_COUPLING_LIMIT = 0.8145287551781475
"""Maximum circular-aperture coupling into a matched Gaussian mode."""


FRESNEL_REFLECTION_LOSS = 0.035
"""Optical Fresnel-reflection loss per uncoated fiber tip (KILO convention).

KILO ``kilo/injection.py:25`` applies ``FRESNEL_LOSS = 0.035`` as the
uncoated SMF-input-tip reflection budget; nullsim's ``coupling_efficiency``
applies it when ``apply_fresnel_loss=True``. Anti-reflection-coated tips
land closer to 0.5-1%; the 3.5% value is the conservative paper-baseline.
"""


_OFF_AXIS_QUADRATURE_NODES = 96
"""Gauss-Legendre nodes for the off-axis overlap; matches adaptive quadrature
to ~1e-16 for ``theta D / lambda`` up to at least 80."""

_OFF_AXIS_CHUNK = 8192
"""Separations evaluated per block, bounding the (chunk, nodes) work array."""


def coupling_efficiency(
    strehl: float | ArrayLike,
    central_obscuration_ratio: float | ArrayLike = 0.0,
    *,
    apply_fresnel_loss: bool = False,
) -> float | NDArray[np.float64]:
    """Return on-axis single-mode fiber coupling efficiency.

    Parameters:
        strehl: Strehl ratio after AO/wavefront residuals. Values must be
            finite and within the closed interval [0, 1]. Array inputs are
            preserved and converted to float64.
        central_obscuration_ratio: Linear central-obscuration ratio epsilon.
            Values must be finite and in the half-open interval [0, 1).
        apply_fresnel_loss: When True, multiply by
            ``(1 - FRESNEL_REFLECTION_LOSS)`` to apply the uncoated-tip
            Fresnel reflection loss. Default False preserves the
            geometric-overlap-only convention. KILO ``injection.py:72``
            applies this loss to every coupled field; set this flag to
            True when that physical loss should be included.

    Returns:
        Coupling efficiency
        ``strehl * geometric_coupling_limit [* (1 - FRESNEL_REFLECTION_LOSS)]``.
        Scalar inputs return a Python float; array inputs return a float64
        ndarray with the broadcast shape of the inputs.

    Raises:
        ValueError: If Strehl or central obscuration values are non-finite or
            outside their valid intervals.
    """
    strehl_array = _as_float64_array(strehl, "strehl")
    _validate_closed_unit_interval(strehl_array, "strehl")

    geometric_limit = geometric_coupling_limit(central_obscuration_ratio)
    coupling = strehl_array * geometric_limit
    if apply_fresnel_loss:
        coupling = coupling * (1.0 - FRESNEL_REFLECTION_LOSS)
    return _as_scalar_if_zero_dim(np.asarray(coupling, dtype=np.float64))


def geometric_coupling_limit(
    central_obscuration_ratio: float | ArrayLike = 0.0,
    *,
    pupil_radius_to_mode_field_radius_ratio: float = (
        OPTIMAL_PUPIL_RADIUS_TO_MODE_FIELD_RADIUS_RATIO
    ),
) -> float | NDArray[np.float64]:
    """Return the geometric Gaussian-mode coupling limit.

    The overlap integral assumes a uniformly illuminated circular or annular
    aperture and a Gaussian fiber mode written in pupil-plane coordinates:

    `eta = 2 * (exp(-(rho epsilon)^2) - exp(-rho^2))^2
    / (rho^2 * (1 - epsilon^2))`

    where `rho` is the aperture-radius / Gaussian-mode-radius ratio and
    `epsilon` is the linear central-obscuration ratio.

    Parameters:
        central_obscuration_ratio: Linear central-obscuration ratio epsilon.
            Values must be finite and in the half-open interval [0, 1).
        pupil_radius_to_mode_field_radius_ratio: Dimensionless aperture-radius
            / mode-field-radius ratio in the pupil-plane overlap expression.
            The default is the unobstructed Ruilier optimum.

    Returns:
        Geometric coupling limit. Scalar inputs return a Python float; array
        inputs return a float64 ndarray preserving the obscuration shape.

    Raises:
        ValueError: If the central obscuration or mode-radius ratio is invalid.
    """
    obscuration = _as_float64_array(
        central_obscuration_ratio,
        "central_obscuration_ratio",
    )
    _validate_half_open_unit_interval(obscuration, "central_obscuration_ratio")
    rho = _validate_positive_scalar(
        pupil_radius_to_mode_field_radius_ratio,
        "pupil_radius_to_mode_field_radius_ratio",
    )

    outer_overlap = np.exp(-(rho**2))
    inner_overlap = np.exp(-((rho * obscuration) ** 2))
    numerator = 2.0 * (inner_overlap - outer_overlap) ** 2
    denominator = (rho**2) * (1.0 - obscuration**2)
    coupling_limit = numerator / denominator
    return _as_scalar_if_zero_dim(np.asarray(coupling_limit, dtype=np.float64))


def angular_coupling_amplitude(
    separation_rad: float | ArrayLike,
    sub_diameter_m: float | ArrayLike,
    wavelength_m: float | ArrayLike,
    central_obscuration_ratio: float = 0.0,
) -> float | NDArray[np.float64]:
    """Off-axis SMF coupling field amplitude relative to on-axis (signed).

    The coupled field is the overlap of a tilted plane wave on the clear
    (annular) sub-aperture with the fiber mode written in the pupil plane,
    the same Gaussian ``exp(-(rho r / R)^2)`` used for on-axis coupling.
    Azimuthal integration of the tilt phase gives J0, so with ``u = r / R``
    and ``s = theta D_sub / lambda``

        a(s) = int_eps^1 u exp(-rho^2 u^2) J0(pi s u) du
               / int_eps^1 u exp(-rho^2 u^2) du

    (the pupil-plane overlap of Ruilier & Cassaing 2001, JOSA A 18, 143,
    evaluated for a tilted input field; Shaklan & Roddier 1988, Appl. Opt.
    27, 2334). ``rho`` is ``OPTIMAL_PUPIL_RADIUS_TO_MODE_FIELD_RADIUS_RATIO``.
    The amplitude changes sign past its first zero at ``s = 1.4948``
    (unobstructed); the sign matters when modes of different diameter
    combine coherently.

    Args:
        separation_rad: Angular offset from on-axis in radians.
        sub_diameter_m: Sub-aperture diameter in meters.
        wavelength_m: Wavelength in meters.
        central_obscuration_ratio: Linear obstruction ratio of the receiving
            sub-aperture, in [0, 1). Packed sub-apertures are clear: 0.

    Returns:
        Amplitude ratio in ``[-1, 1]``, shaped by numpy broadcasting of the
        first three inputs.
    """

    s = np.abs(
        np.asarray(separation_rad, dtype=np.float64)
        * np.asarray(sub_diameter_m, dtype=np.float64)
        / np.asarray(wavelength_m, dtype=np.float64)
    )
    nodes, weights = _off_axis_quadrature(float(central_obscuration_ratio))
    flat = s.ravel()
    out = np.empty_like(flat)
    for start in range(0, flat.size, _OFF_AXIS_CHUNK):
        block = flat[start : start + _OFF_AXIS_CHUNK]
        out[start : start + block.size] = j0(np.pi * np.outer(block, nodes)) @ weights
    return _as_scalar_if_zero_dim(out.reshape(s.shape))


def angular_coupling_factor(
    separation_rad: float | ArrayLike,
    sub_diameter_m: float | ArrayLike,
    wavelength_m: float | ArrayLike,
    central_obscuration_ratio: float = 0.0,
) -> float | NDArray[np.float64]:
    """Off-axis SMF coupling power relative to on-axis, ``eta(theta)/eta(0)``.

    The square of :func:`angular_coupling_amplitude`, the exact overlap of a
    tilted plane wave with the fiber mode. At ``D_sub = 3.14 m`` and
    ``lambda = 1.25 um`` it is 0.971, 0.766, 0.323, 0.0205, and 0.0030 at
    10, 30, 60, 100, and 150 mas. This replaces KILO's Gaussian envelope
    ``exp(-(pi w theta / lambda)^2)``, which overstates coupling beyond about
    0.5 lambda/D_sub (see ``tests/physics/notes/ruilier_cassaing.md``).

    Returns:
        Power factor in ``[0, 1]``, shaped by numpy broadcasting.
    """

    amplitude = np.asarray(
        angular_coupling_amplitude(
            separation_rad, sub_diameter_m, wavelength_m, central_obscuration_ratio
        ),
        dtype=np.float64,
    )
    return _as_scalar_if_zero_dim(amplitude * amplitude)


@lru_cache(maxsize=8)
def _off_axis_quadrature(
    central_obscuration_ratio: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Gauss-Legendre nodes on ``[eps, 1]`` with weights that fold in the mode
    profile ``u exp(-rho^2 u^2)`` and are normalized to sum to one."""

    eps = central_obscuration_ratio
    _validate_half_open_unit_interval(np.asarray(eps), "central_obscuration_ratio")
    x, w = np.polynomial.legendre.leggauss(_OFF_AXIS_QUADRATURE_NODES)
    u = eps + 0.5 * (1.0 - eps) * (x + 1.0)
    rho = OPTIMAL_PUPIL_RADIUS_TO_MODE_FIELD_RADIUS_RATIO
    weights = w * u * np.exp(-((rho * u) ** 2))
    return u, weights / weights.sum()


def _as_float64_array(value: float | ArrayLike, name: str) -> NDArray[np.float64]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric and convertible to float64.") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validate_closed_unit_interval(array: NDArray[np.float64], name: str) -> None:
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must be in the interval [0, 1].")


def _validate_half_open_unit_interval(array: NDArray[np.float64], name: str) -> None:
    if np.any((array < 0.0) | (array >= 1.0)):
        raise ValueError(f"{name} must be in the interval [0, 1).")


def _validate_positive_scalar(value: float, name: str) -> float:
    array = _as_float64_array(value, name)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar.")
    scalar = float(array)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return scalar


def _as_scalar_if_zero_dim(array: NDArray[np.float64]) -> float | NDArray[np.float64]:
    if array.ndim == 0:
        return float(array)
    return array


__all__ = [
    "OPTIMAL_PUPIL_RADIUS_TO_MODE_FIELD_RADIUS_RATIO",
    "UNOBSTRUCTED_GEOMETRIC_COUPLING_LIMIT",
    "angular_coupling_amplitude",
    "angular_coupling_factor",
    "coupling_efficiency",
    "geometric_coupling_limit",
]
