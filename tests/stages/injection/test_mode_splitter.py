"""Tests for the guided-mode fanout stage."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nullsim.pipeline import Field, Geometry, RunContext, SimulationState
from nullsim.stages.injection.mode_splitter import ModeSplitter


def _context(tmp_path: Path) -> RunContext:
    return RunContext(
        rng=np.random.default_rng(0),
        config_hash="mode-splitter",
        sweep_coord_tuple=(),
        instance_id="splitter",
        run_dir=tmp_path,
    )


def _state() -> SimulationState:
    wavelengths = (1.2, 1.3)
    amplitudes = np.arange(1, 9, dtype=np.float64).reshape(4, 2, 1).astype(
        np.complex128
    )
    state = SimulationState.empty({"wavelength_grid_um": wavelengths})
    return replace(
        state,
        geometry=Geometry(
            n_apertures=2,
            effective_collecting_area_m2=8.0,
            aperture_positions_pupil_m=np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [10.0, 0.0],
                    [11.0, 0.0],
                ],
                dtype=np.float64,
            ),
        ),
        field=Field(
            amplitudes=amplitudes,
            wavelength_grid_um=wavelengths,
            mode_labels=("a0", "a1", "b0", "b1"),
        ),
    )


def test_mode_splitter_preserves_detector_power_and_aperture_block_order(
    tmp_path: Path,
) -> None:
    stage = ModeSplitter(params={"copies": 2}, context=_context(tmp_path))

    out = stage.apply(_state())
    split = out.field.amplitudes
    assert split is not None

    original = _state().field.amplitudes
    assert original is not None
    assert split.shape == (8, 2, 1)
    np.testing.assert_allclose(split[0], original[0])
    np.testing.assert_allclose(split[2], original[0])
    payload = out.geometry.components["splitter"]
    original_area = 8.0 / 4.0
    split_area = float(payload["mode_area_m2"])
    assert split_area == pytest.approx(1.0)
    np.testing.assert_allclose(payload["mode_areas_m2"], np.full(8, 1.0))
    assert payload["mode_diameter_m"] == pytest.approx(np.sqrt(8.0 / np.pi))
    np.testing.assert_array_equal(
        payload["physical_mode_indices"],
        np.array([0, 1, 0, 1, 2, 3, 2, 3]),
    )
    assert np.sum(np.abs(split) ** 2) * split_area == pytest.approx(
        float(np.sum(np.abs(original) ** 2) * original_area)
    )
    np.testing.assert_allclose(
        out.geometry.aperture_positions_pupil_m,
        np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [10.0, 0.0],
                [11.0, 0.0],
                [10.0, 0.0],
                [11.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )
    assert out.field.mode_labels == (
        "a0_split0",
        "a1_split0",
        "a0_split1",
        "a1_split1",
        "b0_split0",
        "b1_split0",
        "b0_split1",
        "b1_split1",
    )


def test_mode_splitter_can_split_selected_modes_only(tmp_path: Path) -> None:
    stage = ModeSplitter(
        params={"copies": 2, "split_mode_indices": (0, 2)},
        context=_context(tmp_path),
    )

    out = stage.apply(_state())
    split = out.field.amplitudes
    assert split is not None
    original = _state().field.amplitudes
    assert original is not None

    assert split.shape == (6, 2, 1)
    split_scale = np.sqrt(6.0 / (4.0 * 2.0))
    unsplit_scale = np.sqrt(6.0 / 4.0)
    np.testing.assert_allclose(split[0], original[0] * split_scale)
    np.testing.assert_allclose(split[1], original[0] * split_scale)
    np.testing.assert_allclose(split[2], original[1] * unsplit_scale)
    np.testing.assert_allclose(split[3], original[2] * split_scale)
    np.testing.assert_allclose(split[4], original[2] * split_scale)
    np.testing.assert_allclose(split[5], original[3] * unsplit_scale)

    payload = out.geometry.components["splitter"]
    assert payload["mode_area_m2"] is None
    np.testing.assert_allclose(
        payload["mode_areas_m2"],
        np.array([1.0, 1.0, 2.0, 1.0, 1.0, 2.0], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        payload["physical_mode_indices"],
        np.array([0, 0, 1, 2, 2, 3]),
    )
    assert payload["field_area_reference_m2"] == pytest.approx(8.0 / 6.0)
    original_area = 8.0 / 4.0
    assert np.sum(np.abs(split[:, :, 0]) ** 2) * (8.0 / 6.0) == (
        pytest.approx(float(np.sum(np.abs(original[:, :, 0]) ** 2) * original_area))
    )
    np.testing.assert_allclose(
        out.geometry.aperture_positions_pupil_m,
        np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [10.0, 0.0],
                [10.0, 0.0],
                [11.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )
    assert out.field.mode_labels == (
        "a0_split0",
        "a0_split1",
        "a1",
        "b0_split0",
        "b0_split1",
        "b1",
    )


def test_mode_splitter_preserves_unequal_input_areas(tmp_path: Path) -> None:
    state = replace(
        _state(),
        geometry=replace(
            _state().geometry,
            effective_collecting_area_m2=10.0,
            components={
                "fixed_aperture": {
                    "mode_areas_m2": np.array([1.0, 3.0, 2.0, 4.0], dtype=np.float64),
                    "mode_diameters_m": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
                },
            },
        ),
    )
    stage = ModeSplitter(
        params={"copies": 2, "split_mode_indices": (0, 2)},
        context=_context(tmp_path),
    )

    out = stage.apply(state)
    split = out.field.amplitudes
    assert split is not None
    original = state.field.amplitudes
    assert original is not None

    payload = out.geometry.components["splitter"]
    np.testing.assert_allclose(
        payload["mode_areas_m2"],
        np.array([0.5, 0.5, 3.0, 1.0, 1.0, 4.0], dtype=np.float64),
    )
    np.testing.assert_allclose(
        payload["mode_diameters_m"],
        np.array([1.0, 1.0, 2.0, 3.0, 3.0, 4.0], dtype=np.float64),
    )
    assert payload["field_area_reference_m2"] == pytest.approx(10.0 / 6.0)
    branch_areas = np.asarray(payload["mode_areas_m2"], dtype=np.float64)
    ref_area = float(payload["field_area_reference_m2"])
    expected_scales = np.sqrt(branch_areas / ref_area)
    np.testing.assert_allclose(split[0], original[0] * expected_scales[0])
    np.testing.assert_allclose(split[2], original[1] * expected_scales[2])
    assert np.sum(np.abs(split[:, :, 0]) ** 2) * ref_area == pytest.approx(
        float(
            np.sum(
                np.abs(original[:, :, 0]) ** 2
                * np.array([1.0, 3.0, 2.0, 4.0], dtype=np.float64)[:, np.newaxis]
            )
        )
    )
