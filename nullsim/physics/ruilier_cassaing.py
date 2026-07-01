"""Ruilier-Cassaing single-mode fiber coupling primitives.

This module implements the pure math port called out in SPEC §15.2:
physics primitives stay independent of pipeline state, stages, config, and
transport losses. The coupling model follows the Gaussian-mode overlap used
by Ruilier (1999, SPIE 3350) and Ruilier & Cassaing (2001), with the same
Strehl scaling convention used by Coudé du Foresto et al. (2000).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


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


KILO_MODE_FIELD_RADIUS_TO_SUBAP_RADIUS_RATIO = 0.6436
"""Mode-field radius / sub-aperture radius used by KILO's angular-coupling formula.

KILO defines ``w = 0.6436 * (D_sub / 2)`` in
``kilo/injection.py:106`` for the off-axis Gaussian-mode rolloff
``exp(-(pi w theta / lambda)^2)``. We preserve the same value so the
nullsim sensitivity stage reproduces KILO's planet-throughput-vs-
separation curve. This is a *different* w/R than the on-axis maximum-
overlap optimum encoded in
``OPTIMAL_PUPIL_RADIUS_TO_MODE_FIELD_RADIUS_RATIO`` above -- KILO uses
two distinct conventions for on-axis coupling vs angular envelope.
"""


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


def angular_coupling_factor(
    separation_rad: float | ArrayLike,
    sub_diameter_m: float | ArrayLike,
    wavelength_m: float | ArrayLike,
) -> float | NDArray[np.float64]:
    """Off-axis SMF coupling rolloff (power factor in ``[0, 1]``).

    A tilted wavefront across a sub-aperture reduces the overlap integral
    with the fiber's Gaussian mode:

        eta(theta) / eta(0) = exp(-(pi w theta / lambda)^2)

    where ``w = KILO_MODE_FIELD_RADIUS_TO_SUBAP_RADIUS_RATIO * D_sub / 2``.

    Args:
        separation_rad: Angular offset from on-axis in radians.
        sub_diameter_m: Sub-aperture diameter in meters.
        wavelength_m: Wavelength in meters.

    Returns:
        Multiplicative power factor in ``[0, 1]``. Shape follows numpy
        broadcasting of the inputs.

    Reference:
        Ruilier & Cassaing (2001), JOSA A 18, 143, Eqs. 18-20; KILO
        ``injection.angular_coupling_factor`` (``kilo/injection.py``
        L76-108).
    """

    w = (
        KILO_MODE_FIELD_RADIUS_TO_SUBAP_RADIUS_RATIO
        * np.asarray(sub_diameter_m, dtype=np.float64)
        / 2.0
    )
    x = (
        np.pi
        * w
        * np.asarray(separation_rad, dtype=np.float64)
        / np.asarray(wavelength_m, dtype=np.float64)
    )
    return _as_scalar_if_zero_dim(np.asarray(np.exp(-(x * x)), dtype=np.float64))


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
    "KILO_MODE_FIELD_RADIUS_TO_SUBAP_RADIUS_RATIO",
    "angular_coupling_factor",
    "coupling_efficiency",
    "geometric_coupling_limit",
]
