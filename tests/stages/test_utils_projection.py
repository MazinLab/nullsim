"""Off-zenith projection with pupil-mounted sub-aperture offsets."""

from __future__ import annotations

import math

import numpy as np
import pytest

from nullsim.stages._utils import project_aperture_positions_for_pointing


def test_pupil_offsets_do_not_foreshorten_with_origins() -> None:
    """Only the inter-telescope origin foreshortens off-zenith; intra-pupil
    offsets ride unprojected (pupil tracks the target — review 2026-06-10
    Issue 20)."""

    baseline_m = 80.0
    offset_m = 2.0
    # Two telescopes on a north-south baseline, one sub-aperture offset
    # north within each pupil. Pointing alt=60 deg, az=0 (north):
    # north components foreshorten by sin(alt).
    origins = np.array(
        [
            [0.0, -baseline_m / 2.0],
            [0.0, -baseline_m / 2.0],
            [0.0, +baseline_m / 2.0],
            [0.0, +baseline_m / 2.0],
        ]
    )
    positions = origins.copy()
    positions[1, 1] += offset_m
    positions[3, 1] += offset_m

    projected = project_aperture_positions_for_pointing(
        positions,
        altitude_deg=60.0,
        azimuth_deg=0.0,
        aperture_origins_m=origins,
    )
    sin_alt = math.sin(math.radians(60.0))
    # Inter-telescope baseline (origin separation) foreshortens...
    origin_sep = projected[2, 1] - projected[0, 1]
    assert origin_sep == pytest.approx(baseline_m * sin_alt, rel=1e-12)
    # ...while the intra-pupil offsets stay at their physical scale.
    assert projected[1, 1] - projected[0, 1] == pytest.approx(offset_m, rel=1e-12)
    assert projected[3, 1] - projected[2, 1] == pytest.approx(offset_m, rel=1e-12)

    # Legacy path (no origins): everything foreshortens — exact only when
    # each mode IS a telescope.
    legacy = project_aperture_positions_for_pointing(
        positions,
        altitude_deg=60.0,
        azimuth_deg=0.0,
    )
    assert legacy[1, 1] - legacy[0, 1] == pytest.approx(
        offset_m * sin_alt, rel=1e-12
    )


def test_origin_shape_mismatch_raises() -> None:
    positions = np.zeros((4, 2))
    with pytest.raises(ValueError, match="aperture_origins_m"):
        project_aperture_positions_for_pointing(
            positions,
            altitude_deg=60.0,
            azimuth_deg=0.0,
            aperture_origins_m=np.zeros((3, 2)),
        )
