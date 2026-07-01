"""Tests for nullsim.physics.pupil_geometry — the port of KILO's
sub-aperture placement optimizer.

Pinning tests confirm bit-equivalent agreement with KILO's
``kilo.array.optimize_subaperture_positions`` for the canonical Keck
N=12 configuration plus a circular-pupil reference. Edge cases cover
single sub-aperture, unobstructed circular pupil, and rejection of
malformed inputs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nullsim.physics.pupil_geometry import (
    EELT_DIAMETER_M,
    EELT_N_SPIDERS,
    EELT_OBSCURATION_FRAC,
    EELT_SPIDER_WIDTH_M,
    PupilGeometry,
    SpiderVane,
    _signed_distance_to_polygon_edge,
    circle_avoids_vane,
    circle_inside_circle,
    circle_inside_convex_polygon,
    circle_outside_circle,
    eelt_pupil,
    is_valid_position,
    keck_pupil,
    optimize_subaperture_positions,
    placement_constraint_violation,
    point_to_segment_distance,
)


# ---------------------------------------------------------------------------
# Geometric primitives.
# ---------------------------------------------------------------------------


def test_point_to_segment_distance_endpoint() -> None:
    # Point coincides with segment start.
    assert point_to_segment_distance(0.0, 0.0, 0.0, 0.0, 1.0, 0.0) == 0.0


def test_point_to_segment_distance_perpendicular() -> None:
    # Point at (0, 1), segment along x-axis from (-1, 0) to (1, 0).
    assert point_to_segment_distance(0.0, 1.0, -1.0, 0.0, 1.0, 0.0) == pytest.approx(1.0)


def test_point_to_segment_distance_zero_length_segment() -> None:
    # Degenerate segment (same start/end) reduces to point-to-point distance.
    assert point_to_segment_distance(3.0, 4.0, 0.0, 0.0, 0.0, 0.0) == pytest.approx(5.0)


def test_circle_inside_circle_strict() -> None:
    assert circle_inside_circle(0.0, 0.0, 1.0, 2.0) is True
    assert circle_inside_circle(0.5, 0.5, 1.0, 2.0) is True
    # Touching boundary is allowed up to _GEOM_TOL.
    assert circle_inside_circle(0.0, 1.0, 1.0, 2.0) is True
    assert circle_inside_circle(0.0, 1.5, 1.0, 2.0) is False


def test_circle_outside_circle() -> None:
    assert circle_outside_circle(3.0, 0.0, 1.0, 1.0) is True
    assert circle_outside_circle(1.5, 0.0, 1.0, 1.0) is False
    # Touching is allowed.
    assert circle_outside_circle(2.0, 0.0, 1.0, 1.0) is True


def test_circle_avoids_vane() -> None:
    vane = SpiderVane(0.0, 0.0, 1.0, 0.0, width_m=0.1)
    # Far from vane → OK.
    assert circle_avoids_vane(0.5, 1.0, 0.1, vane) is True
    # On the vane line → fails (distance < r_sub + width/2).
    assert circle_avoids_vane(0.5, 0.0, 0.1, vane) is False


def test_signed_distance_polygon_rejects_clockwise() -> None:
    # Clockwise triangle should raise.
    cw = ((0.0, 0.0), (1.0, 0.0), (0.5, -1.0))
    with pytest.raises(ValueError, match="counter-clockwise"):
        _signed_distance_to_polygon_edge(0.5, -0.3, cw)


def test_signed_distance_polygon_inside_outside() -> None:
    # CCW unit square centered at origin.
    sq = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
    # Center is 1.0 from each edge.
    assert _signed_distance_to_polygon_edge(0.0, 0.0, sq) == pytest.approx(1.0)
    # Outside the square.
    assert _signed_distance_to_polygon_edge(2.0, 0.0, sq) < 0.0


def test_circle_inside_convex_polygon() -> None:
    sq = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
    assert circle_inside_convex_polygon(0.0, 0.0, 0.5, sq) is True
    assert circle_inside_convex_polygon(0.0, 0.0, 1.1, sq) is False


# ---------------------------------------------------------------------------
# Position validity + constraint violation.
# ---------------------------------------------------------------------------


def test_is_valid_position_unobstructed_circular() -> None:
    pupil = PupilGeometry(central_obstruction_diameter_m=0.0)
    # 10 m diameter, place 1 m sub-aperture at (2, 0) → 1 m radius fits.
    assert is_valid_position(2.0, 0.0, 1.0, pupil, 10.0) is True
    # Place at (4.5, 0) → r=4.5, r+r_sub=5.5 > 5.0 → outside.
    assert is_valid_position(4.5, 0.0, 1.0, pupil, 10.0) is False


def test_is_valid_position_with_central_obstruction() -> None:
    pupil = PupilGeometry(central_obstruction_diameter_m=4.0)
    # At (1, 0), r=1, r-r_sub=0 < 2.0 → overlaps obstruction.
    assert is_valid_position(1.0, 0.0, 1.0, pupil, 10.0) is False
    # At (3.5, 0), r=3.5, r-r_sub=2.5 > 2.0 → outside obstruction.
    assert is_valid_position(3.5, 0.0, 1.0, pupil, 10.0) is True


def test_placement_constraint_violation_feasible_returns_zero() -> None:
    pupil = PupilGeometry(central_obstruction_diameter_m=0.0)
    positions = np.array([[2.0, 0.0], [-2.0, 0.0]]).ravel()
    assert placement_constraint_violation(positions, 2, 1.0, pupil, 10.0) == 0.0


def test_placement_constraint_violation_overlap_detected() -> None:
    pupil = PupilGeometry(central_obstruction_diameter_m=0.0)
    # Two unit-radius subs at distance 1.0 → overlap of 1.0.
    positions = np.array([[0.0, 0.0], [1.0, 0.0]]).ravel()
    v = placement_constraint_violation(positions, 2, 1.0, pupil, 10.0)
    assert v > 0.0


# ---------------------------------------------------------------------------
# optimize_subaperture_positions: KILO-style geometry + edge cases.
# ---------------------------------------------------------------------------


def test_optimize_keck_n12_matches_kilo() -> None:
    """The reason this physics module exists: derive Keck N=12 d_sub
    from pupil geometry instead of pinning a constant. KILO's
    keck_only_config reports d_sub = 1.929 m at N=12; this nullsim port
    should reproduce that to ~1 mm.
    """

    _, d_sub = optimize_subaperture_positions(12, 10.0, keck_pupil())
    assert d_sub == pytest.approx(1.929, abs=0.001)


def test_optimize_unobstructed_circular_single_sub() -> None:
    # KILO's n_sub==1 branch places the sub off-center at radius
    # r_sub_max = (r_boundary - r_obs) / 2, so d_sub = r_boundary - r_obs.
    # For a 10 m unobstructed pupil that's d_sub = 5 m (not 10 m — the
    # algorithm assumes the sub is placed off-center even when it didn't
    # need to be).
    pos, d_sub = optimize_subaperture_positions(
        1, 10.0, PupilGeometry(central_obstruction_diameter_m=0.0)
    )
    assert pos.shape == (1, 2)
    assert d_sub == pytest.approx(5.0, rel=1e-6)


def test_optimize_circular_unobstructed_n2() -> None:
    # 2 sub-apertures in a 10 m unobstructed pupil → each fits at radius
    # 2.5 m, so d_sub = 5 m.
    pupil = PupilGeometry(central_obstruction_diameter_m=0.0)
    pos, d_sub = optimize_subaperture_positions(2, 10.0, pupil)
    assert pos.shape == (2, 2)
    assert d_sub == pytest.approx(5.0, rel=0.01)


def test_optimize_default_pupil_is_unobstructed() -> None:
    # pupil=None defaults to unobstructed circular — matches passing one explicitly.
    _, d_default = optimize_subaperture_positions(2, 10.0, pupil=None)
    _, d_explicit = optimize_subaperture_positions(
        2, 10.0, PupilGeometry(central_obstruction_diameter_m=0.0)
    )
    assert d_default == pytest.approx(d_explicit, rel=1e-6)


def test_optimize_rejects_zero_subs() -> None:
    with pytest.raises(ValueError, match="n_sub must be positive"):
        optimize_subaperture_positions(0, 10.0)


def test_optimize_rejects_negative_subs() -> None:
    with pytest.raises(ValueError, match="n_sub must be positive"):
        optimize_subaperture_positions(-1, 10.0)


def test_optimize_returned_positions_are_feasible() -> None:
    """Optimizer output must satisfy hard constraints (no overlap, inside
    boundary, outside obstruction)."""

    pupil = keck_pupil()
    pos, d_sub = optimize_subaperture_positions(12, 10.0, pupil)
    r_sub = d_sub / 2

    # Every sub-aperture is feasible against vanes too (the optimizer
    # treats vanes as soft, but the Keck N=12 solution lands clear of them).
    for x, y in pos:
        assert is_valid_position(x, y, r_sub, pupil, 10.0, check_vanes=False), (
            f"sub-aperture at ({x:.3f}, {y:.3f}) violates hard constraints"
        )

    # No mutual overlap.
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            sep = math.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
            assert sep >= 2 * r_sub - 1e-6, (
                f"subs {i} and {j} overlap: sep={sep:.4f}, "
                f"required >= {2 * r_sub:.4f}"
            )


# ---------------------------------------------------------------------------
# Keck pupil template — structural pins.
# ---------------------------------------------------------------------------


def test_keck_pupil_has_six_vanes() -> None:
    p = keck_pupil()
    assert len(p.spider_vanes) == 6


def test_keck_pupil_central_obstruction_24pc() -> None:
    p = keck_pupil()
    # 24% of 10 m effective diameter.
    assert p.central_obstruction_diameter_m == pytest.approx(2.4)


def test_keck_pupil_boundary_is_hexagon() -> None:
    p = keck_pupil()
    assert p.boundary_vertices is not None
    assert len(p.boundary_vertices) == 6
    # All vertices at the same radius (regular hexagon, hex_ctc / 2).
    radii = [math.hypot(x, y) for x, y in p.boundary_vertices]
    assert max(radii) - min(radii) < 1e-9
    assert radii[0] == pytest.approx(10.95 / 2, abs=1e-9)


# ---------------------------------------------------------------------------
# E-ELT pupil template — NAYRA structural pins.
# ---------------------------------------------------------------------------


def test_eelt_pupil_matches_nayra_geometry_constants() -> None:
    p = eelt_pupil()
    assert p.boundary_vertices is None
    assert p.central_obstruction_diameter_m == pytest.approx(
        EELT_OBSCURATION_FRAC * EELT_DIAMETER_M
    )
    assert len(p.spider_vanes) == EELT_N_SPIDERS

    inner_radius = p.central_obstruction_diameter_m / 2.0
    outer_radius = EELT_DIAMETER_M / 2.0
    for vane in p.spider_vanes:
        assert vane.width_m == pytest.approx(EELT_SPIDER_WIDTH_M)
        assert math.hypot(vane.x_start, vane.y_start) == pytest.approx(inner_radius)
        assert math.hypot(vane.x_end, vane.y_end) == pytest.approx(outer_radius)


def test_optimize_eelt_n18_returns_feasible_subapertures() -> None:
    pupil = eelt_pupil()
    positions, d_sub = optimize_subaperture_positions(18, EELT_DIAMETER_M, pupil)
    assert positions.shape == (18, 2)
    assert d_sub > 0.0

    r_sub = d_sub / 2.0
    for x, y in positions:
        assert is_valid_position(x, y, r_sub, pupil, EELT_DIAMETER_M)


def test_optimize_eelt_n48_grows_packed_layout() -> None:
    pupil = eelt_pupil()
    positions, d_sub = optimize_subaperture_positions(48, EELT_DIAMETER_M, pupil)
    assert positions.shape == (48, 2)
    assert d_sub > 4.5

    r_sub = d_sub / 2.0
    for x, y in positions:
        assert is_valid_position(x, y, r_sub, pupil, EELT_DIAMETER_M)

    diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diffs * diffs, axis=2))
    distances = distances[np.triu_indices(positions.shape[0], k=1)]
    assert float(np.min(distances)) >= d_sub - 1.0e-8


def test_optimize_keck_n6_production_packing_pinned() -> None:
    """Pin the kilo_keck_6 production packing (N=6 per Keck): d_sub and
    feasibility. The N=12 KILO pin alone left the canonical paper config's
    geometry unprotected against seed-strategy changes (review 2026-06-10
    Issue 28c). Reference value from the 2026-06-10 implementation
    (d_sub = 3.140 m, effective area 6 x pi (d/2)^2 = 46.46 m^2/Keck)."""

    from nullsim.physics.pupil_geometry import (
        keck_pupil,
        optimize_subaperture_positions,
    )

    positions, d_sub = optimize_subaperture_positions(6, 10.0, keck_pupil())
    assert d_sub == pytest.approx(3.140, abs=2.0e-3)
    assert np.asarray(positions).shape == (6, 2)
    # Non-overlap feasibility: pairwise separations >= d_sub.
    pos = np.asarray(positions, dtype=np.float64)
    for i in range(6):
        for j in range(i + 1, 6):
            assert np.hypot(*(pos[i] - pos[j])) >= d_sub - 1.0e-9
