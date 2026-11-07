"""Pupil geometry primitives + sub-aperture placement optimizer.

Direct port of KILO's ``kilo.pupil`` (constraint helpers) and
``kilo.array.optimize_subaperture_positions`` (the binary-search +
L-BFGS-B refinement that finds the largest sub-aperture diameter
``d_sub`` that fits ``N`` non-overlapping sub-apertures inside the
primary mirror boundary, outside the central obstruction, and clear
of spider vanes.

Source: ``kilo/pupil.py`` +
``kilo/array.py`` (KILO HEAD ``47607de``). The algorithm and
constants here are intentionally line-for-line faithful so that
nullsim's derived ``d_sub`` matches KILO's to 4 sig figs (Keck N=12 →
1.929 m, agreeing with KILO to ~1 mm).

Used by ``nullsim.stages.ao.pyramid_wfs.PyramidWFS`` to derive
``piston_subaperture_diameter_m`` from pupil geometry instead of
pinning a constant in TOML.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

__all__ = [
    "EELT_DIAMETER_M",
    "EELT_N_SPIDERS",
    "EELT_OBSCURATION_FRAC",
    "EELT_SPIDER_WIDTH_M",
    "PupilGeometry",
    "SpiderVane",
    "circle_avoids_vane",
    "circle_inside_circle",
    "circle_inside_convex_polygon",
    "circle_outside_circle",
    "eelt_pupil",
    "is_valid_position",
    "keck_pupil",
    "optimize_subaperture_positions",
    "placement_constraint_violation",
    "point_to_segment_distance",
]


# Small tolerance for floating-point geometry comparisons.
_GEOM_TOL = 1e-10

# ELT pupil constants. NAYRA's 0.11 obscuration fraction gave a 4.3 m central
# obstruction; ESO's telescope design gives 11.1 m (review 2026-11-07 Issue 7).
EELT_DIAMETER_M = 39.146
EELT_OBSCURATION_FRAC = 11.1 / EELT_DIAMETER_M
EELT_N_SPIDERS = 6
EELT_SPIDER_WIDTH_M = 0.5


@dataclass(frozen=True)
class SpiderVane:
    """A single spider vane modeled as a line segment with width.

    The vane extends from ``(x_start, y_start)`` to ``(x_end, y_end)`` in
    the pupil plane (meters), with a full width of ``width_m``.
    """

    x_start: float
    y_start: float
    x_end: float
    y_end: float
    width_m: float


@dataclass(frozen=True)
class PupilGeometry:
    """Pupil boundary + central obstruction + spider vanes.

    Attributes:
        central_obstruction_diameter_m: Diameter of the central obstruction
            (secondary + baffling) in meters.
        spider_vanes: Spider vane segments. Empty tuple if no vanes.
        boundary_vertices: Convex polygon vertices ``((x0, y0), (x1, y1),
            ...)`` defining the outer pupil boundary in meters. Used for
            Keck's hexagonal primary. ``None`` ⇒ circular boundary at
            ``diameter_m / 2``.
    """

    central_obstruction_diameter_m: float
    spider_vanes: tuple[SpiderVane, ...] = ()
    boundary_vertices: tuple[tuple[float, float], ...] | None = None


# ---------------------------------------------------------------------------
# Geometric primitives (ported from kilo/pupil.py).
# ---------------------------------------------------------------------------


def point_to_segment_distance(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return float(np.sqrt((px - x1) ** 2 + (py - y1) ** 2))
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return float(np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2))


def circle_inside_circle(
    cx: float, cy: float, r_inner: float, r_outer: float
) -> bool:
    return bool(np.sqrt(cx * cx + cy * cy) + r_inner <= r_outer + _GEOM_TOL)


def circle_outside_circle(
    cx: float, cy: float, r_sub: float, r_obstruction: float
) -> bool:
    return bool(np.sqrt(cx * cx + cy * cy) - r_sub >= r_obstruction - _GEOM_TOL)


def circle_avoids_vane(
    cx: float, cy: float, r_sub: float, vane: SpiderVane
) -> bool:
    dist = point_to_segment_distance(
        cx, cy, vane.x_start, vane.y_start, vane.x_end, vane.y_end
    )
    return bool(dist >= r_sub + vane.width_m / 2)


def _signed_distance_to_polygon_edge(
    px: float, py: float, vertices: tuple[tuple[float, float], ...]
) -> float:
    """Minimum signed distance from a point to the nearest edge of a CCW
    convex polygon. Positive inside, negative outside.
    """

    n = len(vertices)
    signed_area = sum(
        vertices[i][0] * vertices[(i + 1) % n][1]
        - vertices[(i + 1) % n][0] * vertices[i][1]
        for i in range(n)
    )
    if signed_area < 0:
        raise ValueError(
            "Polygon vertices must be in counter-clockwise order "
            "(got clockwise; signed area is negative)"
        )
    min_dist = np.inf
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        ex = x2 - x1
        ey = y2 - y1
        edge_len = np.sqrt(ex * ex + ey * ey)
        if edge_len == 0:
            continue
        nx = -ey / edge_len
        ny = ex / edge_len
        d = (px - x1) * nx + (py - y1) * ny
        min_dist = min(min_dist, d)
    return float(min_dist)


def circle_inside_convex_polygon(
    cx: float,
    cy: float,
    r: float,
    vertices: tuple[tuple[float, float], ...],
) -> bool:
    return bool(_signed_distance_to_polygon_edge(cx, cy, vertices) >= r)


def is_valid_position(
    x: float,
    y: float,
    r_sub: float,
    pupil: PupilGeometry,
    diameter_m: float,
    check_vanes: bool = True,
) -> bool:
    """Hard-constraint test for a single sub-aperture position."""

    if pupil.boundary_vertices is not None:
        if not circle_inside_convex_polygon(x, y, r_sub, pupil.boundary_vertices):
            return False
    else:
        if not circle_inside_circle(x, y, r_sub, diameter_m / 2):
            return False

    r_obs = pupil.central_obstruction_diameter_m / 2
    if r_obs > 0 and not circle_outside_circle(x, y, r_sub, r_obs):
        return False

    if check_vanes:
        for vane in pupil.spider_vanes:
            if not circle_avoids_vane(x, y, r_sub, vane):
                return False

    return True


def placement_constraint_violation(
    positions_flat: NDArray[np.float64],
    n_sub: int,
    r_sub: float,
    pupil: PupilGeometry,
    diameter_m: float,
) -> float:
    """Total squared constraint violation (0.0 ⇒ fully feasible)."""

    positions = positions_flat.reshape(n_sub, 2)
    violation = 0.0
    r_obs = pupil.central_obstruction_diameter_m / 2

    for i in range(n_sub):
        x, y = positions[i]
        r = np.sqrt(x * x + y * y)

        if pupil.boundary_vertices is not None:
            signed_dist = _signed_distance_to_polygon_edge(
                x, y, pupil.boundary_vertices
            )
            shortfall = r_sub - signed_dist
            if shortfall > 0:
                violation += shortfall * shortfall
        else:
            overshoot = (r + r_sub) - diameter_m / 2
            if overshoot > 0:
                violation += overshoot * overshoot

        if r_obs > 0:
            undershoot = r_obs - (r - r_sub)
            if undershoot > 0:
                violation += undershoot * undershoot

        for vane in pupil.spider_vanes:
            dist = point_to_segment_distance(
                x, y, vane.x_start, vane.y_start, vane.x_end, vane.y_end
            )
            clearance_needed = r_sub + vane.width_m / 2
            shortfall = clearance_needed - dist
            if shortfall > 0:
                violation += shortfall * shortfall

        for j in range(i + 1, n_sub):
            xj, yj = positions[j]
            sep = np.sqrt((x - xj) ** 2 + (y - yj) ** 2)
            overlap = 2 * r_sub - sep
            if overlap > 0:
                violation += overlap * overlap

    return float(violation)


# ---------------------------------------------------------------------------
# Placement search (ported from kilo/array.py).
# ---------------------------------------------------------------------------


def _effective_boundary_radius(pupil: PupilGeometry, diameter_m: float) -> float:
    if pupil.boundary_vertices is not None:
        return _signed_distance_to_polygon_edge(0.0, 0.0, pupil.boundary_vertices)
    return diameter_m / 2


def _ring_positions(
    n_sub: int, R: float, rotation_rad: float = 0.0
) -> NDArray[np.float64]:
    angles = np.linspace(0, 2 * np.pi, n_sub, endpoint=False) + rotation_rad
    return np.column_stack([R * np.cos(angles), R * np.sin(angles)])


def _max_ring_d_sub(
    n_sub: int, pupil: PupilGeometry, diameter_m: float,
) -> float:
    r_obs = pupil.central_obstruction_diameter_m / 2
    r_boundary = _effective_boundary_radius(pupil, diameter_m)
    if r_boundary <= r_obs:
        return 0.0

    s = np.sin(np.pi / n_sub) if n_sub >= 2 else 1.0

    candidates = [
        (r_boundary + r_obs) / 2,
        r_boundary / (1 + s),
    ]
    if s < 1.0:
        candidates.append(r_obs / (1 - s))

    best_d = 0.0
    for R in candidates:
        if r_obs > R or r_boundary < R:
            continue
        d_boundary = 2 * (r_boundary - R)
        d_obstruction = 2 * (R - r_obs)
        d_overlap = 2 * R * s
        d = min(d_boundary, d_obstruction, d_overlap)
        if d > best_d:
            best_d = d
    return float(best_d)


def _find_valid_ring(
    n_sub: int,
    r_sub: float,
    pupil: PupilGeometry,
    diameter_m: float,
) -> tuple[NDArray[np.float64] | None, int]:
    r_obs = pupil.central_obstruction_diameter_m / 2
    r_boundary = _effective_boundary_radius(pupil, diameter_m)

    R_min_overlap = r_sub / np.sin(np.pi / n_sub) if n_sub >= 2 else 0.0
    R_min = max(R_min_overlap, r_obs + r_sub)
    R_max = r_boundary - r_sub

    if R_min > R_max:
        return None, 0

    best_positions: NDArray[np.float64] | None = None
    best_n_valid = 0

    n_radii = 7
    n_rotations = 18
    for R in np.linspace(R_min, R_max, n_radii):
        for rot_idx in range(n_rotations):
            rotation = rot_idx * 2 * np.pi / (n_sub * n_rotations)
            positions = _ring_positions(n_sub, R, rotation)
            n_valid = sum(
                1 for x, y in positions
                if is_valid_position(x, y, r_sub, pupil, diameter_m)
            )
            if n_valid >= n_sub and best_n_valid < n_sub:
                best_positions = positions
                best_n_valid = n_valid
            elif n_valid > best_n_valid and best_n_valid < n_sub:
                best_n_valid = n_valid
                best_positions = positions

    return best_positions, best_n_valid


def _hexagonal_grid_positions(
    n_sub: int, diameter_m: float, r_sub: float, pupil: PupilGeometry,
) -> tuple[NDArray[np.float64], int]:
    d_sub = 2 * r_sub
    max_ring = int(np.ceil(np.sqrt(n_sub))) + 3

    best_seed: NDArray[np.float64] | None = None
    best_n_valid = -1

    n_rotations = 12 if pupil.spider_vanes else 1
    for rot_idx in range(n_rotations):
        angle = rot_idx * np.pi / (6 * n_rotations)
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        candidates: list[tuple[float, float, float]] = []
        for q in range(-max_ring, max_ring + 1):
            for r in range(-max_ring, max_ring + 1):
                xg = d_sub * (q + r * 0.5)
                yg = d_sub * (r * np.sqrt(3) / 2)
                x = cos_a * xg - sin_a * yg
                y = sin_a * xg + cos_a * yg
                radius = np.sqrt(x * x + y * y)
                candidates.append((x, y, radius))

        candidates.sort(key=lambda p: p[2])

        valid = [
            (x, y) for x, y, _ in candidates
            if is_valid_position(x, y, r_sub, pupil, diameter_m)
        ]

        if len(valid) >= n_sub and len(valid) > best_n_valid:
            best_n_valid = len(valid)
            best_seed = np.array(valid[:n_sub])
        elif len(valid) > best_n_valid:
            padded = list(valid)
            for x, y, _ in candidates:
                if len(padded) >= n_sub:
                    break
                if not any(
                    abs(x - vx) < 1e-12 and abs(y - vy) < 1e-12 for vx, vy in padded
                ):
                    padded.append((x, y))
            best_seed = np.array(padded[:n_sub])
            best_n_valid = len(valid)

    # Code Review 2026-05-16 Issue 53: ``assert`` is stripped under
    # ``python -O``; raise to keep this invariant under all settings.
    if best_seed is None:
        raise RuntimeError(
            "_hexagonal_grid_positions: no rotation produced a seed even "
            "after fallback padding — this should never happen for n_sub > 0."
        )
    return best_seed, best_n_valid


def _concentric_rings_positions(
    k_inner: int,
    k_outer: int,
    r_sub: float,
    pupil: PupilGeometry,
    diameter_m: float,
) -> NDArray[np.float64] | None:
    """Place ``k_inner`` subs on an inner ring + ``k_outer`` on an outer ring.

    Picks inner ring at the smallest valid radius (just outside the
    obstruction or the in-ring overlap bound) and outer ring at the
    largest valid radius (just inside the boundary). Sweeps inner radii
    and rotation offsets for both rings. Returns shape
    ``(k_inner + k_outer, 2)`` on success, ``None`` otherwise.
    """

    r_obs = pupil.central_obstruction_diameter_m / 2
    r_boundary = _effective_boundary_radius(pupil, diameter_m)
    if r_boundary <= r_obs or k_inner < 1 or k_outer < 1:
        return None

    s_inner = np.sin(np.pi / k_inner) if k_inner >= 2 else 1.0
    s_outer = np.sin(np.pi / k_outer) if k_outer >= 2 else 1.0

    R_in_min = max(r_sub / s_inner if k_inner >= 2 else 0.0, r_obs + r_sub)
    R_out = r_boundary - r_sub
    R_out_min_overlap = r_sub / s_outer if k_outer >= 2 else 0.0
    if R_out < R_out_min_overlap:
        return None
    R_in_max = R_out - 2 * r_sub
    if R_in_min > R_in_max:
        return None

    n_radii = 5
    n_rotations = 12
    half_inner = np.pi / k_inner if k_inner >= 2 else 0.0
    half_outer = np.pi / k_outer if k_outer >= 2 else 0.0
    for R_in in np.linspace(R_in_min, R_in_max, n_radii):
        for rot_in_idx in range(n_rotations):
            rot_in = rot_in_idx * (2 * half_inner) / n_rotations
            for rot_out_idx in range(n_rotations):
                rot_out = rot_out_idx * (2 * half_outer) / n_rotations
                inner = _ring_positions(k_inner, R_in, rot_in)
                outer = _ring_positions(k_outer, R_out, rot_out)
                positions = np.vstack([inner, outer])
                ok = all(
                    is_valid_position(x, y, r_sub, pupil, diameter_m)
                    for x, y in positions
                )
                if not ok:
                    continue
                diffs = positions[:, None, :] - positions[None, :, :]
                dist = np.sqrt((diffs ** 2).sum(axis=-1))
                np.fill_diagonal(dist, np.inf)
                if dist.min() < 2 * r_sub:
                    continue
                return positions
    return None


def _find_concentric_rings(
    n_sub: int, r_sub: float, pupil: PupilGeometry, diameter_m: float
) -> NDArray[np.float64] | None:
    """Try every (k_inner, k_outer) split summing to ``n_sub``, balanced first."""

    if n_sub < 2:
        return None
    splits = sorted(
        ((k_inner, n_sub - k_inner) for k_inner in range(1, n_sub)),
        key=lambda kp: (abs(kp[0] - kp[1]), -kp[1]),
    )
    for k_inner, k_outer in splits:
        positions = _concentric_rings_positions(
            k_inner, k_outer, r_sub, pupil, diameter_m
        )
        if positions is not None:
            return positions
    return None


def _wedge_hex_triangle_positions(
    n_sub: int,
    r_sub: float,
    pupil: PupilGeometry,
    diameter_m: float,
) -> NDArray[np.float64] | None:
    """Stack ``n_per_gap`` subs as a triangular hex pyramid per spider gap.

    For triangular ``n_per_gap = k*(k+1)/2`` (k >= 2: so 3, 6, 10, 15, ...),
    each gap holds k rows: base of k subs at the outer rim, ..., apex of
    1 sub closest to the obstruction. Hex spacing is computed in local
    Cartesian coordinates (gap centred along +y axis), then rotated to
    each gap's angular position — true 2*r_sub spacing between every
    pair (polar arithmetic would compress inner rows).
    """

    import math

    n_vanes = len(pupil.spider_vanes)
    if n_vanes < 2 or n_sub % n_vanes != 0:
        return None
    n_per_gap = n_sub // n_vanes
    k = int((math.sqrt(8 * n_per_gap + 1) - 1) / 2)
    if k < 2 or k * (k + 1) // 2 != n_per_gap:
        return None

    r_obs = pupil.central_obstruction_diameter_m / 2
    r_outer_boundary = _effective_boundary_radius(pupil, diameter_m)

    vane_angles = sorted(
        np.arctan2(v.y_end, v.x_end) % (2 * np.pi)
        for v in pupil.spider_vanes
    )
    gap_centres: list[float] = []
    for i in range(n_vanes):
        a1 = vane_angles[i]
        a2 = vane_angles[(i + 1) % n_vanes]
        if a2 < a1:
            a2 += 2 * np.pi
        gap_centres.append((a1 + a2) / 2)

    margin = max(0.05 * r_sub, 0.02)
    # Outer-row corner subs sit at local (x, R_out) with x = (k-1)*r_sub,
    # so radial distance is sqrt(R_out^2 + ((k-1)*r_sub)^2). Boundary
    # constraint: corner_radial + r_sub <= r_outer_boundary - margin.
    corner_x = (k - 1) * r_sub
    boundary_budget = r_outer_boundary - r_sub - margin
    if boundary_budget <= corner_x:
        return None
    R_out_max = np.sqrt(boundary_budget ** 2 - corner_x ** 2)
    # Apex (single sub at x=0) at y = R_out - (k-1)*sqrt(3)*r_sub must
    # clear the obstruction: y - r_sub >= r_obs + margin.
    R_out_min = r_obs + (1 + (k - 1) * np.sqrt(3)) * r_sub + margin
    if R_out_min > R_out_max:
        return None

    n_radii = 7
    for R_out in np.linspace(R_out_max, R_out_min, n_radii):
        local: list[tuple[float, float]] = []
        for j in range(k):
            n_in_row = k - j
            y = R_out - j * np.sqrt(3) * r_sub
            for i in range(n_in_row):
                x = (i - (n_in_row - 1) / 2) * 2 * r_sub
                local.append((x, y))

        positions: list[list[float]] = []
        for ang_centre in gap_centres:
            rot = ang_centre - np.pi / 2
            cos_r, sin_r = np.cos(rot), np.sin(rot)
            for x, y in local:
                gx = cos_r * x - sin_r * y
                gy = sin_r * x + cos_r * y
                positions.append([gx, gy])

        pos_array = np.array(positions, dtype=float)

        ok = all(
            is_valid_position(x, y, r_sub, pupil, diameter_m)
            for x, y in pos_array
        )
        if not ok:
            continue
        diffs = pos_array[:, None, :] - pos_array[None, :, :]
        dist = np.sqrt((diffs ** 2).sum(axis=-1))
        np.fill_diagonal(dist, np.inf)
        if dist.min() < 2 * r_sub - 1e-6:
            continue
        return pos_array

    return None


def _gap_triangle_positions(
    n_sub: int,
    r_sub: float,
    pupil: PupilGeometry,
    diameter_m: float,
) -> NDArray[np.float64] | None:
    n_vanes = len(pupil.spider_vanes)
    if n_vanes < 2:
        return None
    if n_sub % n_vanes != 0:
        return None
    n_per_gap = n_sub // n_vanes
    if n_per_gap not in (1, 2, 3):
        return None

    vane_angles = sorted(
        np.arctan2(v.y_end, v.x_end) % (2 * np.pi)
        for v in pupil.spider_vanes
    )

    gaps: list[tuple[float, float]] = []
    for i in range(n_vanes):
        a1 = vane_angles[i]
        a2 = vane_angles[(i + 1) % n_vanes]
        if a2 < a1:
            a2 += 2 * np.pi
        gaps.append(((a1 + a2) / 2, a2 - a1))

    r_obs = pupil.central_obstruction_diameter_m / 2
    r_outer = _effective_boundary_radius(pupil, diameter_m)

    margin = max(0.05 * r_sub, 0.02)
    r_inner_pos = r_obs + r_sub + margin
    r_outer_pos = r_outer - r_sub - margin
    r_mid = (r_inner_pos + r_outer_pos) / 2

    if r_inner_pos >= r_outer_pos:
        return None

    positions: list[list[float]] = []
    for ang_center, ang_width in gaps:
        if n_per_gap == 1:
            positions.append([
                r_mid * np.cos(ang_center),
                r_mid * np.sin(ang_center),
            ])
        elif n_per_gap == 2:
            for r in (r_inner_pos, r_outer_pos):
                positions.append([
                    r * np.cos(ang_center),
                    r * np.sin(ang_center),
                ])
        elif n_per_gap == 3:
            tangent_dang = (r_sub + margin) / r_outer_pos
            gap_half_dang = (ang_width / 2) * 0.7
            dang = min(tangent_dang, gap_half_dang)
            positions.append([
                r_inner_pos * np.cos(ang_center),
                r_inner_pos * np.sin(ang_center),
            ])
            for sign in (-1, 1):
                ang = ang_center + sign * dang
                positions.append([
                    r_outer_pos * np.cos(ang),
                    r_outer_pos * np.sin(ang),
                ])

    pos_array = np.array(positions, dtype=float)

    for x, y in pos_array:
        if not is_valid_position(x, y, r_sub, pupil, diameter_m):
            return None

    diffs = pos_array[:, None, :] - pos_array[None, :, :]
    dist = np.sqrt((diffs ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    if dist.min() < 2 * r_sub:
        return None

    return pos_array


def _generate_seeds(
    n_sub: int,
    r_sub: float,
    pupil: PupilGeometry,
    diameter_m: float,
    hex_seed: NDArray[np.float64] | None,
    ring_seed: NDArray[np.float64] | None,
) -> list[NDArray[np.float64]]:
    seeds: list[NDArray[np.float64]] = []

    gap_seed = _gap_triangle_positions(n_sub, r_sub, pupil, diameter_m)
    if gap_seed is not None:
        seeds.append(gap_seed)

    rings_seed = _find_concentric_rings(n_sub, r_sub, pupil, diameter_m)
    if rings_seed is not None:
        seeds.append(rings_seed)

    wedge_seed = _wedge_hex_triangle_positions(
        n_sub, r_sub, pupil, diameter_m
    )
    if wedge_seed is not None:
        seeds.append(wedge_seed)

    if hex_seed is not None:
        seeds.append(hex_seed)
    if ring_seed is not None:
        seeds.append(ring_seed)

    r_obs = pupil.central_obstruction_diameter_m / 2
    r_boundary = _effective_boundary_radius(pupil, diameter_m)
    R_min_overlap = r_sub / np.sin(np.pi / n_sub) if n_sub >= 2 else 0.0
    R_min = max(R_min_overlap, r_obs + r_sub)
    R_max = r_boundary - r_sub

    if R_min <= R_max:
        for R in [R_min, (R_min + R_max) / 2, R_max]:
            for rot_frac in [0.25, 0.5, 0.75]:
                rotation = rot_frac * 2 * np.pi / n_sub
                pos = _ring_positions(n_sub, R, rotation)
                all_valid = all(
                    is_valid_position(x, y, r_sub, pupil, diameter_m)
                    for x, y in pos
                )
                if all_valid:
                    seeds.append(pos)

    if not seeds:
        raise RuntimeError(
            f"No valid seed positions for n_sub={n_sub}, "
            f"r_sub={r_sub:.3f}m in {diameter_m:.1f}m pupil"
        )

    return seeds[:8]


def _refine_positions(
    positions: NDArray[np.float64],
    n_sub: int,
    r_sub: float,
    pupil: PupilGeometry,
    diameter_m: float,
) -> NDArray[np.float64]:
    if pupil.boundary_vertices is not None:
        r_bound = max(np.sqrt(x * x + y * y) for x, y in pupil.boundary_vertices)
    else:
        r_bound = diameter_m / 2
    bounds = [(-r_bound, r_bound)] * (2 * n_sub)

    hard_weight = 1e7

    def objective(flat_pos: NDArray[np.float64]) -> float:
        pos = flat_pos.reshape(n_sub, 2)
        hard_v = placement_constraint_violation(
            flat_pos, n_sub, r_sub, pupil, diameter_m
        )
        diffs = pos[:, None, :] - pos[None, :, :]
        # Code Review 2026-05-16 Issue 32: regularize the log term relative
        # to r_sub rather than a fixed 1e-10. For mm-scale subapertures
        # the absolute floor would dominate the true squared distance.
        eps_rel = 1.0e-6
        dist_floor_sq = (eps_rel * r_sub) ** 2
        dist_sq = (diffs ** 2).sum(axis=-1)
        idx = np.triu_indices(n_sub, k=1)
        log_dist_sum = np.log(
            np.sqrt(np.maximum(dist_sq[idx], dist_floor_sq))
        ).sum()
        return -log_dist_sum + hard_weight * hard_v

    init_violation = placement_constraint_violation(
        positions.ravel(), n_sub, r_sub, pupil, diameter_m
    )

    result = minimize(
        objective,
        positions.ravel(),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 300, "ftol": 1e-12},
    )
    refined = result.x.reshape(n_sub, 2)
    refined_violation = placement_constraint_violation(
        result.x, n_sub, r_sub, pupil, diameter_m
    )

    if refined_violation <= init_violation + 1e-8:
        return refined
    return positions


def _evaluate_placement(
    positions: NDArray[np.float64],
    n_sub: int,
    r_sub: float,
    pupil: PupilGeometry,
    diameter_m: float,
) -> float:
    flat = positions.ravel()
    violation = placement_constraint_violation(flat, n_sub, r_sub, pupil, diameter_m)
    if violation > 1e-6:
        return float(np.inf)

    diffs = positions[:, None, :] - positions[None, :, :]
    # Code Review 2026-05-16 Issue 32: relative regularization, same as
    # _refine_positions.
    eps_rel = 1.0e-6
    dist_floor_sq = (eps_rel * r_sub) ** 2
    dist_sq = (diffs ** 2).sum(axis=-1)
    idx = np.triu_indices(n_sub, k=1)
    log_dist_sum = np.log(
        np.sqrt(np.maximum(dist_sq[idx], dist_floor_sq))
    ).sum()
    return float(-log_dist_sum)


def _max_feasible_diameter_for_positions(
    positions: NDArray[np.float64],
    pupil: PupilGeometry,
    diameter_m: float,
) -> float:
    """Largest sub-aperture diameter allowed by fixed center positions."""

    positions = np.asarray(positions, dtype=np.float64)
    max_diameter = np.inf
    r_obs = pupil.central_obstruction_diameter_m / 2
    r_outer = diameter_m / 2

    for x, y in positions:
        radial = float(np.hypot(x, y))
        if pupil.boundary_vertices is not None:
            edge_clearance = _signed_distance_to_polygon_edge(
                float(x), float(y), pupil.boundary_vertices
            )
            max_diameter = min(max_diameter, 2.0 * edge_clearance)
        else:
            max_diameter = min(max_diameter, 2.0 * (r_outer - radial))

        if r_obs > 0.0:
            max_diameter = min(max_diameter, 2.0 * (radial - r_obs))

        for vane in pupil.spider_vanes:
            vane_clearance = point_to_segment_distance(
                float(x),
                float(y),
                vane.x_start,
                vane.y_start,
                vane.x_end,
                vane.y_end,
            ) - vane.width_m / 2
            max_diameter = min(max_diameter, 2.0 * vane_clearance)

    if positions.shape[0] > 1:
        diffs = positions[:, None, :] - positions[None, :, :]
        distances = np.sqrt((diffs ** 2).sum(axis=-1))
        distances = distances[np.triu_indices(positions.shape[0], k=1)]
        max_diameter = min(max_diameter, float(np.min(distances)))

    if not np.isfinite(max_diameter):
        return 0.0
    return max(0.0, float(max_diameter))


def _grow_refined_placement(
    positions: NDArray[np.float64],
    d_sub: float,
    d_sub_hi: float,
    n_sub: int,
    pupil: PupilGeometry,
    diameter_m: float,
) -> tuple[NDArray[np.float64], float]:
    """Grow a feasible placement using hard-constraint refinement.

    The seed search is intentionally conservative. For crowded pupils with
    spider vanes, such as E-ELT N=48, a feasible seed can often be enlarged by
    jointly relaxing all sub-aperture centers. This pass keeps the binary
    search's proven feasible layout as the starting point and only accepts a
    larger diameter when the existing hard-constraint score remains feasible.
    """

    if pupil.boundary_vertices is not None or not pupil.spider_vanes or n_sub < 24:
        return np.asarray(positions, dtype=np.float64), float(d_sub)

    best_positions = np.asarray(positions, dtype=np.float64)
    best_d_sub = float(d_sub)
    low = best_d_sub
    high = max(float(d_sub_hi), low)
    if high <= low:
        return best_positions, best_d_sub

    for _ in range(12):
        trial_d_sub = 0.5 * (low + high)
        trial_r_sub = 0.5 * trial_d_sub
        refined = _refine_positions(
            best_positions,
            n_sub,
            trial_r_sub,
            pupil,
            diameter_m,
        )
        max_diameter = _max_feasible_diameter_for_positions(
            refined,
            pupil,
            diameter_m,
        )
        accepted_d_sub = min(trial_d_sub, max_diameter)
        if accepted_d_sub > best_d_sub:
            best_positions = refined
            best_d_sub = accepted_d_sub
            low = accepted_d_sub
        else:
            high = trial_d_sub

    return best_positions, best_d_sub


def _place_single_subaperture(
    diameter_m: float, r_sub: float, pupil: PupilGeometry,
) -> NDArray[np.float64]:
    r_obs = pupil.central_obstruction_diameter_m / 2
    r_start = max(r_obs + r_sub + 0.01, 0.01)
    r_end = _effective_boundary_radius(pupil, diameter_m) - r_sub

    n_angles = 36
    n_radii = 20

    for r in np.linspace(r_start, r_end, n_radii):
        for angle_deg in np.linspace(0, 360, n_angles, endpoint=False):
            x = r * np.cos(np.radians(angle_deg))
            y = r * np.sin(np.radians(angle_deg))
            if is_valid_position(x, y, r_sub, pupil, diameter_m):
                return np.array([[x, y]])

    raise RuntimeError(
        f"Could not place a single sub-aperture in diameter={diameter_m:.1f}m pupil"
    )


def optimize_subaperture_positions(
    n_sub: int,
    diameter_m: float,
    pupil: PupilGeometry | None = None,
) -> tuple[NDArray[np.float64], float]:
    """Find optimal positions and size for ``n_sub`` sub-apertures in the pupil.

    Maximizes the sub-aperture diameter (collecting area) while satisfying
    hard constraints: inside the primary boundary, outside the central
    obstruction, no mutual overlap. Spider vane avoidance is a soft preference.

    Args:
        n_sub: Number of sub-apertures (>= 1).
        diameter_m: Effective primary mirror diameter in meters.
        pupil: Pupil geometry. If ``None``, a circular unobstructed
            aperture is assumed.

    Returns:
        ``(positions, d_sub)`` where ``positions`` has shape ``(n_sub, 2)``
        in meters and ``d_sub`` is the optimized sub-aperture diameter in
        meters.

    Raises:
        ValueError: If ``n_sub`` is not positive.
        RuntimeError: If no feasible placement can be found.
    """

    if n_sub <= 0:
        raise ValueError(f"n_sub must be positive, got {n_sub}")

    if pupil is None:
        pupil = PupilGeometry(central_obstruction_diameter_m=0.0)

    if n_sub == 1:
        r_obs = pupil.central_obstruction_diameter_m / 2
        r_boundary = _effective_boundary_radius(pupil, diameter_m)
        r_sub_max = (r_boundary - r_obs) / 2
        pos = _place_single_subaperture(diameter_m, r_sub_max, pupil)
        return pos, 2 * r_sub_max

    ring_d_max = _max_ring_d_sub(n_sub, pupil, diameter_m)
    d_sub_hi = max(diameter_m / np.sqrt(n_sub), ring_d_max) * 1.1
    d_sub_upper_bound = d_sub_hi
    d_sub_lo = 0.1
    best_d_sub = d_sub_lo

    for _ in range(15):
        d_trial = (d_sub_lo + d_sub_hi) / 2
        r_trial = d_trial / 2

        _, hex_valid = _hexagonal_grid_positions(
            n_sub, diameter_m, r_trial, pupil
        )
        _, ring_valid = _find_valid_ring(n_sub, r_trial, pupil, diameter_m)
        gap_seed = _gap_triangle_positions(n_sub, r_trial, pupil, diameter_m)
        gap_feasible = gap_seed is not None and gap_seed.shape[0] == n_sub
        rings_seed = _find_concentric_rings(n_sub, r_trial, pupil, diameter_m)
        rings_feasible = rings_seed is not None and rings_seed.shape[0] == n_sub
        wedge_seed = _wedge_hex_triangle_positions(
            n_sub, r_trial, pupil, diameter_m
        )
        wedge_feasible = wedge_seed is not None and wedge_seed.shape[0] == n_sub

        if (
            hex_valid >= n_sub
            or ring_valid >= n_sub
            or gap_feasible
            or rings_feasible
            or wedge_feasible
        ):
            best_d_sub = d_trial
            d_sub_lo = d_trial
        else:
            d_sub_hi = d_trial

    if best_d_sub <= 0.1:
        raise RuntimeError(
            f"Could not find feasible sub-aperture placement for n_sub={n_sub}, "
            f"diameter={diameter_m:.1f}m"
        )

    r_sub = best_d_sub / 2
    hex_pos, hex_valid = _hexagonal_grid_positions(
        n_sub, diameter_m, r_sub, pupil
    )
    ring_pos, ring_valid = _find_valid_ring(n_sub, r_sub, pupil, diameter_m)

    hex_seed = hex_pos if hex_valid >= n_sub else None
    ring_seed = ring_pos if ring_valid >= n_sub else None

    seeds = _generate_seeds(
        n_sub, r_sub, pupil, diameter_m, hex_seed, ring_seed
    )

    best_result: NDArray[np.float64] | None = None
    best_score = np.inf

    for seed in seeds:
        refined = _refine_positions(seed, n_sub, r_sub, pupil, diameter_m)
        score = _evaluate_placement(refined, n_sub, r_sub, pupil, diameter_m)
        if score < best_score:
            best_score = score
            best_result = refined

    if best_result is None:
        raise RuntimeError(
            f"All refinement attempts failed for n_sub={n_sub}, "
            f"diameter={diameter_m:.1f}m"
        )

    best_result, best_d_sub = _grow_refined_placement(
        best_result,
        best_d_sub,
        d_sub_upper_bound,
        n_sub,
        pupil,
        diameter_m,
    )

    return best_result, best_d_sub


# ---------------------------------------------------------------------------
# Named pupil templates.
# ---------------------------------------------------------------------------


def keck_pupil() -> PupilGeometry:
    """Pupil geometry for Keck I / II.

    Ported from ``kilo/config.py::_keck_pupil``. The Keck primary is a
    36-segment hexagonal mosaic with a 10.95 m corner-to-corner bounding
    circle, a 2.4 m central obstruction (24% of the 10 m effective
    diameter), and six 2.6 cm-wide spider vanes at 60° intervals running
    from the obstruction edge to the hexagon boundary. The dodecagonal
    outline is approximated as a regular hexagon inscribed in the 10.95 m
    bounding circle.
    """

    hex_ctc = 10.95  # m, Keck observer's guide
    hex_radius = hex_ctc / 2

    boundary_vertices = tuple(
        (
            float(hex_radius * np.cos(np.radians(a))),
            float(hex_radius * np.sin(np.radians(a))),
        )
        for a in range(0, 360, 60)
    )

    obs_diameter = 0.24 * 10.0  # 2.4 m

    vanes = tuple(
        SpiderVane(
            x_start=0.0,
            y_start=0.0,
            x_end=float(hex_radius * np.cos(np.radians(a))),
            y_end=float(hex_radius * np.sin(np.radians(a))),
            width_m=0.026,
        )
        for a in range(0, 360, 60)
    )

    return PupilGeometry(
        central_obstruction_diameter_m=obs_diameter,
        spider_vanes=vanes,
        boundary_vertices=boundary_vertices,
    )


def eelt_pupil() -> PupilGeometry:
    """Pupil geometry for the ELT primary used by NAYRA.

    Ported from nayra ``nayra.config.eelt_pupil_geometry``:
    39.146 m circular pupil, 11.1 m central obstruction (ESO; NAYRA used
    4.3 m), and six 0.5 m-wide
    radial spider vanes from the obscuration edge to the outer pupil.
    """

    obs_diameter = EELT_OBSCURATION_FRAC * EELT_DIAMETER_M
    obs_radius = obs_diameter / 2.0
    outer_radius = EELT_DIAMETER_M / 2.0

    vanes = tuple(
        SpiderVane(
            x_start=float(obs_radius * np.cos(np.radians(angle_deg))),
            y_start=float(obs_radius * np.sin(np.radians(angle_deg))),
            x_end=float(outer_radius * np.cos(np.radians(angle_deg))),
            y_end=float(outer_radius * np.sin(np.radians(angle_deg))),
            width_m=EELT_SPIDER_WIDTH_M,
        )
        for angle_deg in np.linspace(0.0, 360.0, EELT_N_SPIDERS, endpoint=False)
    )

    return PupilGeometry(
        central_obstruction_diameter_m=obs_diameter,
        spider_vanes=vanes,
        boundary_vertices=None,
    )
