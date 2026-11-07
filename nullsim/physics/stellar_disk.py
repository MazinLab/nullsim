"""Stellar-disk leakage for a partially-resolved uniform-brightness source.

For a uniform-brightness disk of angular radius rho_0 = theta_star / 2,
the incoherent leakage reaching the dark ports is the area-weighted
average of the position-angle-averaged dark throughput T(r):

    <T>_disk = (2 / rho_0**2) * int_0^rho_0 r * T(r) dr

(Guyon et al. 2013: each sky point propagates coherently through the
chip, intensities add incoherently over the disk). The realized-null
Monte Carlo already evaluates the star at r = 0, so its mean contains
T(0). The disk floor added on top of it is therefore the excess

    disk_floor = <T(r) - T(0)>_disk,

not <T>_disk, which would count the static on-axis leakage twice
(review 2026-11-07 Issue 17).

The integral uses Gauss-Legendre nodes in r on [0, rho_0] with weight
2 r / rho_0**2, evaluating T on its own grid inside the stellar radius
rather than interpolating a coarse separation grid. The integrand is
r * T(r) with T(r) - T(0) ~ r**2 near axis, a low-order polynomial over
a disk much smaller than a fringe, so 16 nodes are exact to rounding.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "disk_radial_quadrature",
    "stellar_disk_null_floor",
    "effective_null_with_disk_floor",
]

ThroughputFn = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def disk_radial_quadrature(
    stellar_diameter_rad: float,
    n_nodes: int = 16,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Radial nodes and weights for a uniform-disk average.

    Returns ``(r_rad, weights)`` with ``sum(weights * f(r_rad))`` equal to
    ``(2 / rho_0**2) int_0^rho_0 r f(r) dr``. The weights sum to one.
    """

    if n_nodes < 1:
        raise ValueError(f"n_nodes must be >= 1, got {n_nodes}.")
    rho_0 = 0.5 * float(stellar_diameter_rad)
    x, gl_weights = np.polynomial.legendre.leggauss(int(n_nodes))
    r = 0.5 * rho_0 * (x + 1.0)
    # dr = rho_0/2 dx and 2 r / rho_0^2 dr = (r / rho_0) dx.
    return r, gl_weights * r / rho_0


def stellar_disk_null_floor(
    throughput_fn: ThroughputFn,
    stellar_diameter_rad: float,
    *,
    on_axis_throughput: float | None = None,
    n_nodes: int = 16,
) -> float:
    """Disk-averaged dark throughput in excess of the on-axis value.

    Args:
        throughput_fn: Maps separations in radians, shape ``(n,)``, to the
            position-angle-averaged dark throughput at each, shape ``(n,)``.
        stellar_diameter_rad: Stellar angular diameter in radians.
        on_axis_throughput: The T(0) already counted by the on-axis null.
            Defaults to ``throughput_fn(0)``. Pass the unsmoothed value when
            ``throughput_fn`` includes pointing-jitter smoothing, so the
            jitter leakage of the star is kept.
        n_nodes: Gauss-Legendre nodes in r.

    Returns:
        ``<T(r)>_disk - T(0)``. For a point star (zero diameter) this is
        ``throughput_fn(0) - on_axis_throughput``, zero by default.
    """

    if stellar_diameter_rad <= 0.0:
        r = np.zeros(1, dtype=np.float64)
        weights = np.ones(1, dtype=np.float64)
    else:
        r, weights = disk_radial_quadrature(stellar_diameter_rad, n_nodes)
    t = np.asarray(throughput_fn(np.concatenate(([0.0], r))), dtype=np.float64)
    if t.shape != (r.size + 1,):
        raise ValueError(
            "stellar_disk_null_floor: throughput_fn returned shape "
            f"{t.shape}; expected ({r.size + 1},)."
        )
    t_on_axis = float(t[0]) if on_axis_throughput is None else float(on_axis_throughput)
    return float(weights @ t[1:]) - t_on_axis


def effective_null_with_disk_floor(
    null_val_dark: float,
    throughput_fn: ThroughputFn,
    stellar_diameter_rad: float,
    *,
    n_nodes: int = 16,
) -> float:
    """Realized on-axis null plus the resolved-disk excess.

    Both feed the dark-port background linearly. ``null_val_dark`` already
    contains the on-axis T(0), so only the excess over it is added.
    """

    return float(null_val_dark) + stellar_disk_null_floor(
        throughput_fn, stellar_diameter_rad, n_nodes=n_nodes,
    )
