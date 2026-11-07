"""(u, v) coverage helpers for KILO-style baseline projection.

Ports the math from ``kilo/array.py`` (``compute_baselines``,
``baselines_to_uv``, ``compute_uv_tracks``) so nullsim can render the
paperv3 Fig 5 (u, v) coverage grid without depending on KILO at runtime.

References
----------
Thompson, Moran & Swenson, "Interferometry and Synthesis in Radio
Astronomy", Eqs. 4.1 (uv projection) and 4.3 (ENU → equatorial XYZ).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

# Maunakea — same default as kilo/array.py.
_MAUNAKEA_LAT_DEG = 19.8283


@dataclass(frozen=True)
class Baseline:
    """One sub-aperture pair, with telescope-grouping flag.

    Mirrors ``kilo.array.Baseline``. ``vector_enu_m`` is the global
    East-North-Up vector from sub-aperture ``i`` to sub-aperture ``j``.
    """

    i: int
    j: int
    vector_enu_m: tuple[float, float, float]
    is_intra_telescope: bool

    @property
    def length_m(self) -> float:
        e, n, u = self.vector_enu_m
        return float(np.sqrt(e * e + n * n + u * u))


def enumerate_subaperture_baselines(
    telescope_enu_m: NDArray[np.floating],
    subaperture_pupil_m: NDArray[np.floating],
    n_subapertures_per_telescope: int,
) -> list[Baseline]:
    """Build the full baseline list across a regular telescope+sub-aperture grid.

    Sub-apertures are assumed grouped: indices ``[0, n_sub)`` belong to
    telescope 0, ``[n_sub, 2*n_sub)`` to telescope 1, etc. The same
    pupil-frame ``(E, N)`` layout is broadcast to every telescope by
    adding the telescope's ENU origin.

    Args:
        telescope_enu_m: ``(n_telescopes, 3)`` array of telescope ENU positions (m).
        subaperture_pupil_m: ``(n_sub, 2)`` array of pupil-frame sub-aperture
            positions (m). The same layout is reused across telescopes.
        n_subapertures_per_telescope: must equal ``subaperture_pupil_m.shape[0]``.

    Returns:
        List of all ``C(n_total, 2)`` baselines.
    """

    if subaperture_pupil_m.shape[0] != n_subapertures_per_telescope:
        raise ValueError(
            f"subaperture_pupil_m has {subaperture_pupil_m.shape[0]} rows but "
            f"n_subapertures_per_telescope={n_subapertures_per_telescope}."
        )

    n_tel = telescope_enu_m.shape[0]
    n_sub = n_subapertures_per_telescope
    n_total = n_tel * n_sub

    # Broadcast pupil positions onto telescope ENU origins.
    sub_enu = np.empty((n_total, 3), dtype=np.float64)
    for t in range(n_tel):
        for s in range(n_sub):
            i = t * n_sub + s
            sub_enu[i, 0] = subaperture_pupil_m[s, 0] + telescope_enu_m[t, 0]
            sub_enu[i, 1] = subaperture_pupil_m[s, 1] + telescope_enu_m[t, 1]
            sub_enu[i, 2] = telescope_enu_m[t, 2]

    def _telescope_of(i: int) -> int:
        return i // n_sub

    baselines: list[Baseline] = []
    for i in range(n_total):
        for j in range(i + 1, n_total):
            vec = (
                float(sub_enu[j, 0] - sub_enu[i, 0]),
                float(sub_enu[j, 1] - sub_enu[i, 1]),
                float(sub_enu[j, 2] - sub_enu[i, 2]),
            )
            baselines.append(Baseline(
                i=i, j=j, vector_enu_m=vec,
                is_intra_telescope=_telescope_of(i) == _telescope_of(j),
            ))
    return baselines


def baselines_to_uv(
    baselines: Sequence[Baseline],
    hour_angle_rad: float | NDArray[np.floating],
    declination_rad: float,
    wavelength_m: float,
    latitude_rad: float = np.radians(_MAUNAKEA_LAT_DEG),
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Project baselines into (u, v) coordinates (Thompson, Moran & Swenson 4.1).

    Args:
        baselines: List of ``Baseline``.
        hour_angle_rad: Scalar or 1D array of hour angles (rad).
        declination_rad: Target declination (rad).
        wavelength_m: Observing wavelength (m).
        latitude_rad: Observatory latitude (rad). Default Maunakea.

    Returns:
        ``(u, v)`` arrays. If ``hour_angle_rad`` is scalar: shape ``(n_baselines,)``.
        If 1D of length ``T``: shape ``(T, n_baselines)``. Units: cycles per radian
        (baseline / wavelength).
    """

    ha = np.atleast_1d(hour_angle_rad)
    dec = float(declination_rad)
    sin_ha, cos_ha = np.sin(ha), np.cos(ha)
    sin_dec, cos_dec = np.sin(dec), np.cos(dec)
    sin_lat, cos_lat = np.sin(latitude_rad), np.cos(latitude_rad)

    bvec = np.array([b.vector_enu_m for b in baselines], dtype=np.float64)
    be, bn, bu = bvec[:, 0], bvec[:, 1], bvec[:, 2]

    # ENU → equatorial XYZ (Thompson, Moran & Swenson Eq. 4.3).
    bx = -sin_lat * bn + cos_lat * bu
    by = be
    bz = cos_lat * bn + sin_lat * bu

    u_arr = sin_ha[:, None] * bx[None, :] + cos_ha[:, None] * by[None, :]
    v_arr = (
        -sin_dec * cos_ha[:, None] * bx[None, :]
        + sin_dec * sin_ha[:, None] * by[None, :]
        + cos_dec * bz[None, :]
    )

    u_arr = u_arr / wavelength_m
    v_arr = v_arr / wavelength_m

    if u_arr.shape[0] == 1 and np.isscalar(hour_angle_rad):
        u_arr = u_arr.squeeze(axis=0)
        v_arr = v_arr.squeeze(axis=0)
    return u_arr, v_arr


def baselines_to_uv_with_origins(
    baselines: Sequence[Baseline],
    aperture_origins_enu_m: NDArray[np.floating],
    hour_angle_rad: float | NDArray[np.floating],
    declination_rad: float,
    wavelength_m: float,
    latitude_rad: float = np.radians(_MAUNAKEA_LAT_DEG),
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Project baselines into (u, v) with rigid intra-telescope pupil offsets.

    Only the ground baseline between telescope origins,
    ``origin_j - origin_i``, rotates with the Earth and is projected as in
    ``baselines_to_uv`` (Thompson, Moran & Swenson Eqs. 4.1, 4.3). The
    remainder of ``vector_enu_m`` is the pupil-frame sub-aperture offset;
    the pupil tracks the target, so that offset is added unprojected as
    ``(u, v) += (dE, dN) / wavelength``. Parallactic rotation of the pupil
    is ignored, the same snapshot convention as the stages'
    ``project_aperture_positions_for_pointing`` (review 2026-06-10
    Issue 20). With one aperture per telescope this equals
    ``baselines_to_uv``.

    Args:
        baselines: List of ``Baseline``; ``i`` and ``j`` index the rows of
            ``aperture_origins_enu_m``.
        aperture_origins_enu_m: ``(N, 3)`` ENU origin (m) of the telescope
            hosting each aperture.
        hour_angle_rad, declination_rad, wavelength_m, latitude_rad: as in
            ``baselines_to_uv``.

    Returns:
        ``(u, v)`` with the shapes of ``baselines_to_uv``, in cycles per radian.
    """

    origins = np.asarray(aperture_origins_enu_m, dtype=np.float64)
    i_idx = np.array([b.i for b in baselines], dtype=np.int64)
    j_idx = np.array([b.j for b in baselines], dtype=np.int64)
    origin_vec = origins[j_idx] - origins[i_idx]
    pupil_vec = np.array([b.vector_enu_m for b in baselines], dtype=np.float64) - origin_vec
    u, v = baselines_to_uv(
        [replace(b, vector_enu_m=tuple(float(x) for x in o))
         for b, o in zip(baselines, origin_vec)],
        hour_angle_rad, declination_rad, wavelength_m, latitude_rad,
    )
    return u + pupil_vec[:, 0] / wavelength_m, v + pupil_vec[:, 1] / wavelength_m


def compute_uv_tracks(
    baselines: Sequence[Baseline],
    declination_deg: float,
    ha_start_deg: float,
    ha_end_deg: float,
    n_time_steps: int,
    wavelength_m: float,
    latitude_deg: float = _MAUNAKEA_LAT_DEG,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Sample (u, v) tracks across an hour-angle range. Port of KILO ``compute_uv_tracks``."""

    ha_rad = np.linspace(
        np.radians(ha_start_deg), np.radians(ha_end_deg), n_time_steps
    )
    return baselines_to_uv(
        baselines, ha_rad, np.radians(declination_deg),
        wavelength_m, np.radians(latitude_deg),
    )


def project_positions_to_sky_plane_m(
    *,
    positions_enu_m: NDArray[np.floating],
    altitude_deg: float,
    azimuth_deg: float,
) -> NDArray[np.float64]:
    """Project ENU positions onto the plane perpendicular to the line of sight.

    Off-zenith, the array baselines foreshorten as seen from the source: the
    geometric delay toward a sky angle ``(alpha, beta)`` is
    ``2*pi/lambda * (x*alpha + y*beta)`` only after the aperture positions are
    projected onto the sky plane. This returns the projected ``(N, 2)``
    coordinates in that plane so the phase rows in
    ``planet_throughput`` / kernel-derivative / disk-sample builders stay valid
    when pointing away from zenith.

    Args:
        positions_enu_m: ``(N, 3)`` ENU positions (East, North, Up) in metres.
            ``(N, 2)`` input is treated as ``Up = 0``.
        altitude_deg: Line-of-sight altitude above the horizon (90 = zenith).
        azimuth_deg: Line-of-sight azimuth, measured from North toward East.

    Returns:
        ``(N, 2)`` sky-plane coordinates: axis 0 toward increasing azimuth,
        axis 1 in the in-plane direction that reduces to North at zenith (the
        negative of the increasing-altitude direction), completing the
        right-handed ``(e_az, e_alt, line-of-sight)`` triad. At
        ``altitude_deg == 90`` and ``azimuth_deg == 0`` this is the identity
        on ``(East, North)``.

    Raises:
        ValueError: if ``positions_enu_m`` does not have a trailing axis of
            length 2 or 3.
    """

    positions = np.asarray(positions_enu_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[-1] not in (2, 3):
        raise ValueError(
            "project_positions_to_sky_plane_m: positions_enu_m must have shape "
            f"(N, 2) or (N, 3); got {positions.shape}."
        )
    if positions.shape[-1] == 2:
        east = positions[:, 0]
        north = positions[:, 1]
        up = np.zeros_like(east)
    else:
        east = positions[:, 0]
        north = positions[:, 1]
        up = positions[:, 2]

    alt_rad = math.radians(float(altitude_deg))
    az_rad = math.radians(float(azimuth_deg))
    sin_alt = math.sin(alt_rad)
    cos_alt = math.cos(alt_rad)
    sin_az = math.sin(az_rad)
    cos_az = math.cos(az_rad)

    # Unit vector toward increasing azimuth (horizontal, perpendicular to the
    # azimuth meridian) and the in-plane direction that reduces to North at
    # zenith (negative of the increasing-altitude direction), both in ENU.
    e_az = np.array([cos_az, -sin_az, 0.0], dtype=np.float64)
    e_alt = np.array(
        [sin_alt * sin_az, sin_alt * cos_az, -cos_alt], dtype=np.float64
    )

    stacked = np.stack([east, north, up], axis=1)
    projected = np.empty((stacked.shape[0], 2), dtype=np.float64)
    projected[:, 0] = stacked @ e_az
    projected[:, 1] = stacked @ e_alt
    return projected


__all__ = [
    "Baseline",
    "baselines_to_uv",
    "baselines_to_uv_with_origins",
    "compute_uv_tracks",
    "enumerate_subaperture_baselines",
    "project_positions_to_sky_plane_m",
]
