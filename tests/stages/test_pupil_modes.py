"""Tests for monolithic-pupil mode placement."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nullsim.pipeline import Geometry, RunContext, SimulationState
from nullsim.stages.telescope.pupil_modes import PupilModes


def _context(tmp_path: Path) -> RunContext:
    return RunContext(
        rng=np.random.default_rng(0),
        config_hash="pupil-modes-test",
        sweep_coord_tuple=(),
        instance_id="pupil_modes",
        run_dir=tmp_path,
    )


def test_pupil_modes_places_linear_modes_across_diameter(tmp_path: Path) -> None:
    state = replace(
        SimulationState.empty(),
        geometry=Geometry(diameter_m=8.0, area_m2=np.pi * 4.0**2, n_apertures=1),
    )
    stage = PupilModes(
        params={"n_modes": 2, "span_fraction": 1.0},
        context=_context(tmp_path),
    )

    out = stage.apply(state)

    np.testing.assert_allclose(
        out.geometry.aperture_positions_pupil_m,
        np.array([[-4.0, 0.0], [4.0, 0.0]]),
    )
    np.testing.assert_allclose(out.geometry.baselines_m, np.array([8.0]))
    payload = out.geometry.components["pupil_modes"]
    assert payload["n_modes"] == 2
    assert payload["span_m"] == pytest.approx(8.0)
    assert payload["layout"] == "linear"
    assert payload["mode_diameter_m"] is None


def test_pupil_modes_places_packed_circular_modes(tmp_path: Path) -> None:
    state = replace(
        SimulationState.empty(),
        geometry=Geometry(diameter_m=8.0, area_m2=np.pi * 4.0**2, n_apertures=1),
    )
    stage = PupilModes(
        params={"n_modes": 12, "layout": "packed_circular"},
        context=_context(tmp_path),
    )

    out = stage.apply(state)

    positions = out.geometry.aperture_positions_pupil_m
    assert positions is not None
    assert positions.shape == (12, 2)
    assert np.max(np.sqrt(np.sum(positions * positions, axis=1))) < 4.0
    assert out.geometry.baselines_m is not None
    assert out.geometry.baselines_m.size == 66
    payload = out.geometry.components["pupil_modes"]
    assert payload["layout"] == "packed_circular"
    assert payload["mode_diameter_m"] > 0.0
    expected_area = math.pi * (payload["mode_diameter_m"] / 2.0) ** 2 * 12
    assert out.geometry.effective_collecting_area_m2 == pytest.approx(expected_area)


def test_pupil_modes_places_jittered_packed_modes(tmp_path: Path) -> None:
    state = replace(
        SimulationState.empty(),
        geometry=Geometry(diameter_m=8.0, area_m2=np.pi * 4.0**2, n_apertures=1),
    )
    packed = PupilModes(
        params={"n_modes": 18, "layout": "packed_circular"},
        context=_context(tmp_path),
    ).apply(state)
    jittered = PupilModes(
        params={"n_modes": 18, "layout": "packed_jittered"},
        context=_context(tmp_path),
    ).apply(state)

    packed_positions = packed.geometry.aperture_positions_pupil_m
    jittered_positions = jittered.geometry.aperture_positions_pupil_m
    assert packed_positions is not None
    assert jittered_positions is not None
    assert jittered_positions.shape == (18, 2)
    assert not np.allclose(jittered_positions, packed_positions)
    payload = jittered.geometry.components["pupil_modes"]
    assert payload["layout"] == "packed_jittered"
    _assert_holes_inside_and_nonoverlapping(jittered_positions, payload)


def test_pupil_modes_places_nonredundant_circular_modes(tmp_path: Path) -> None:
    state = replace(
        SimulationState.empty(),
        geometry=Geometry(diameter_m=8.0, area_m2=np.pi * 4.0**2, n_apertures=1),
    )
    stage = PupilModes(
        params={"n_modes": 18, "layout": "nonredundant_circular"},
        context=_context(tmp_path),
    )

    out = stage.apply(state)

    positions = out.geometry.aperture_positions_pupil_m
    assert positions is not None
    assert positions.shape == (18, 2)
    assert np.max(np.sqrt(np.sum(positions * positions, axis=1))) < 4.0
    payload = out.geometry.components["pupil_modes"]
    assert payload["layout"] == "nonredundant_circular"
    assert payload["mode_diameter_m"] > 0.0
    _assert_holes_inside_and_nonoverlapping(positions, payload)
    packed = PupilModes(
        params={"n_modes": 18, "layout": "packed_circular"},
        context=_context(tmp_path),
    ).apply(state)
    packed_positions = packed.geometry.aperture_positions_pupil_m
    assert packed_positions is not None
    assert not np.allclose(positions, packed_positions)


def test_pupil_modes_can_preserve_area_and_normalize_baseline(
    tmp_path: Path,
) -> None:
    full_area = math.pi * 4.0**2
    state = replace(
        SimulationState.empty(),
        geometry=Geometry(
            diameter_m=8.0,
            area_m2=full_area,
            collecting_area_m2=full_area,
            effective_collecting_area_m2=full_area,
            n_apertures=1,
        ),
    )
    stage = PupilModes(
        params={
            "n_modes": 12,
            "layout": "packed_circular",
            "normalize_max_baseline_to_span": True,
            "preserve_collecting_area": True,
        },
        context=_context(tmp_path),
    )

    out = stage.apply(state)

    assert out.geometry.baselines_m is not None
    assert np.max(out.geometry.baselines_m) == pytest.approx(8.0)
    assert out.geometry.effective_collecting_area_m2 == pytest.approx(full_area)
    payload = out.geometry.components["pupil_modes"]
    assert payload["normalize_max_baseline_to_span"] is True
    assert payload["preserve_collecting_area"] is True
    assert payload["effective_area_source"] == "upstream"
    assert payload["geometric_mode_area_m2"] != pytest.approx(full_area)


def test_pupil_modes_requires_upstream_diameter(tmp_path: Path) -> None:
    stage = PupilModes(params={"n_modes": 2}, context=_context(tmp_path))

    with pytest.raises(ValueError, match="geometry.diameter_m"):
        stage.apply(SimulationState.empty())


def test_pupil_modes_rejects_multi_telescope_array(tmp_path: Path) -> None:
    state = replace(
        SimulationState.empty(),
        geometry=Geometry(diameter_m=10.0, area_m2=np.pi * 5.0**2, n_apertures=4),
    )
    stage = PupilModes(params={"n_modes": 4}, context=_context(tmp_path))

    with pytest.raises(ValueError, match="n_apertures = 4"):
        stage.apply(state)


def test_pupil_modes_publishes_per_mode_vectors_and_origins(tmp_path: Path) -> None:
    """HWO-like case: a length-1 telescope vector upstream must not win the
    per-mode lookup over the pupil modes (review 2026-11-07 Issues 1, 22)."""

    from nullsim.stages._mode_splitter import (
        component_mode_areas_m2,
        component_mode_diameters_m,
    )
    from nullsim.stages._utils import mode_origin_positions_m

    geometry = Geometry(
        diameter_m=8.0, area_m2=np.pi * 4.0**2, n_apertures=1
    ).add_component(
        "telescope",
        {"mode_areas_m2": np.array([np.pi * 4.0**2]), "mode_diameters_m": np.array([8.0])},
    )
    state = replace(SimulationState.empty(), geometry=geometry)

    out = PupilModes(
        params={"n_modes": 12, "layout": "packed_circular"},
        context=_context(tmp_path),
    ).apply(state)

    d_mode = out.geometry.components["pupil_modes"]["mode_diameter_m"]
    areas, area_source = component_mode_areas_m2(out, 12)
    diameters, diameter_source = component_mode_diameters_m(out, 12)
    np.testing.assert_allclose(diameters, np.full(12, d_mode))
    np.testing.assert_allclose(areas, np.full(12, math.pi * (d_mode / 2.0) ** 2))
    assert area_source == diameter_source.replace("diameters_m", "areas_m2")
    assert area_source.startswith("geometry.pupil_modes")
    np.testing.assert_array_equal(
        mode_origin_positions_m(out.geometry, 12), np.zeros((12, 2))
    )


def _pairwise_distances(positions: np.ndarray) -> np.ndarray:
    diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=-1))
    return dists[np.triu_indices(positions.shape[0], k=1)]


def _assert_holes_inside_and_nonoverlapping(
    positions: np.ndarray,
    payload: dict[str, object],
) -> None:
    diameter = float(payload["mode_diameter_m"])
    radius = 0.5 * diameter
    center_radii = np.sqrt(np.sum(positions * positions, axis=1))
    assert float(np.max(center_radii + radius)) <= 4.0 + 1.0e-5
    assert float(np.min(_pairwise_distances(positions))) + 1.0e-5 >= diameter
