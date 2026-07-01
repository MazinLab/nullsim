"""Stellar-disk leakage for a partially-resolved uniform-brightness source.

For a uniform-brightness disk of angular diameter theta_star, the
incoherent leakage fraction reaching the dark ports is the area-weighted
integral of the dark-port throughput across the disk:

    null_disk = (1 / A_disk) * iint T(r) dA
              = (2 / rho_0**2) * int_0^rho_0 r * T(r) dr

where T(r) is the position-angle-averaged dark-port throughput and
rho_0 = theta_star / 2 is the disk radius. The AO-realized on-axis null
and this disk-floor add linearly to the dark-port background, so we
sum (not max) them in :func:`effective_null_with_disk_floor`.

Port of KILO ``kilo/performance.py`` ``stellar_disk_null_floor``
(lines 1130-1209) and ``effective_null_with_disk_floor`` (lines
1263-1298). KILO comment (lines 1267-1273) is explicit about why these
two contributions add linearly rather than max-combine: the AO residual
is the time-varying on-axis mean shift, and the disk integral is the
static off-axis throughput rolloff over the resolved disk -- both feed
the Poisson background.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["stellar_disk_null_floor", "effective_null_with_disk_floor"]


def stellar_disk_null_floor(
    throughputs: NDArray[np.float64] | tuple[float, ...] | list[float],
    separations_rad: NDArray[np.float64] | tuple[float, ...] | list[float],
    stellar_diameter_rad: float,
) -> float:
    """Stellar-leakage null floor from a partially-resolved uniform disk.

    Args:
        throughputs: Position-angle-averaged dark-port throughput at each
            separation (dimensionless, in ``[0, 1]``).
        separations_rad: Monotonically increasing angular separations in
            radians. Must cover ``[0, rho_0]`` where ``rho_0 =
            stellar_diameter_rad / 2``.
        stellar_diameter_rad: Stellar angular diameter in radians.

    Returns:
        The dimensionless disk-integrated null floor in ``[0, 1]``.

    Raises:
        ValueError: If ``stellar_diameter_rad / 2`` exceeds the largest
            separation in ``separations_rad``.
    """

    if stellar_diameter_rad <= 0.0:
        return 0.0

    rho_0 = stellar_diameter_rad / 2.0
    r = np.asarray(separations_rad, dtype=np.float64)
    t = np.asarray(throughputs, dtype=np.float64)
    if r.shape != t.shape:
        raise ValueError(
            "stellar_disk_null_floor: throughputs and separations_rad "
            f"shape mismatch (got {t.shape} vs {r.shape})."
        )
    if r.size == 0:
        return 0.0
    if r.size > 1 and not np.all(np.diff(r) > 0.0):
        raise ValueError(
            "stellar_disk_null_floor: separations_rad must be strictly "
            "monotonically increasing (np.interp on unsorted xp returns "
            f"silent garbage). Got {r.tolist()}."
        )
    if rho_0 > r.max():
        raise ValueError(
            "stellar_disk_null_floor: stellar disk radius rho_0="
            f"{rho_0:.3e} rad exceeds separations_rad.max()="
            f"{r.max():.3e} rad."
        )

    # Anchor (r=0, T=0): on-axis throughput is identically zero for a
    # nulling instrument by construction.
    r_full = np.concatenate(([0.0], r))
    t_full = np.concatenate(([0.0], t))
    inside = r_full < rho_0
    t_at_rho = float(np.interp(rho_0, r_full, t_full))
    r_eval = np.concatenate((r_full[inside], [rho_0]))
    t_eval = np.concatenate((t_full[inside], [t_at_rho]))
    order = np.argsort(r_eval)
    r_eval = r_eval[order]
    t_eval = t_eval[order]

    integral = float(np.trapezoid(2.0 * r_eval * t_eval, r_eval))
    return integral / (rho_0 * rho_0)


def effective_null_with_disk_floor(
    null_val_dark: float,
    throughputs: NDArray[np.float64] | tuple[float, ...] | list[float],
    separations_rad: NDArray[np.float64] | tuple[float, ...] | list[float],
    stellar_diameter_rad: float,
) -> float:
    """Combine the AO-realized on-axis null with the resolved-disk floor.

    Both contributions feed the dark-port Poisson background linearly:
    one is the time-varying on-axis mean shift from AO piston + servo
    residuals, the other is the static off-axis throughput rolloff
    integrated over the resolved disk.

    Args:
        null_val_dark: AO-realized on-axis null depth (dimensionless).
        throughputs: PA-averaged dark-port throughput vs separation.
        separations_rad: Monotonically increasing separations in radians.
        stellar_diameter_rad: Stellar angular diameter in radians.

    Returns:
        Effective null depth = ``null_val_dark + disk_floor``.
    """

    disk_floor = stellar_disk_null_floor(
        throughputs, separations_rad, stellar_diameter_rad,
    )
    return float(null_val_dark) + float(disk_floor)
