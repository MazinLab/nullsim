"""Tests for chip optimizer per-mode geometry helpers."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from nullsim.pipeline import Field, Geometry, SimulationState
from nullsim.stages.chip_optimization import (
    _mode_areas_m2,
    _mode_diameters_m,
    _require_field,
)


def test_chip_optimizer_uses_mode_splitter_geometry_overrides() -> None:
    state = SimulationState.empty({"wavelength_grid_um": (1.25,)})
    state = type(state)(
        scene=state.scene,
        geometry=Geometry(
            effective_collecting_area_m2=8.0,
            n_apertures=2,
            components={
                "pupil_modes": {
                    "mode_area_m2": 4.0,
                    "mode_diameter_m": 4.0,
                },
                "splitter": {
                    "mode_area_m2": 1.0,
                    "mode_diameter_m": 2.0,
                },
            },
        ),
        field=state.field,
        opd_budget=state.opd_budget,
        throughput=state.throughput,
        wavefront=state.wavefront,
        photon_rates=state.photon_rates,
        results=state.results,
    )

    np.testing.assert_allclose(_mode_areas_m2(state, 8), np.ones(8))
    np.testing.assert_allclose(_mode_diameters_m(state, 8), np.full(8, 2.0))


def test_chip_optimizer_uses_splitter_field_area_reference_once() -> None:
    wavelengths = (1.2, 1.3)
    amplitudes = np.array(
        [
            [[np.sqrt(0.5)], [np.sqrt(0.5)]],
            [[np.sqrt(2.0)], [np.sqrt(2.0)]],
        ],
        dtype=np.complex128,
    )
    state = SimulationState.empty({"wavelength_grid_um": wavelengths})
    state = replace(
        state,
        geometry=Geometry(
            effective_collecting_area_m2=3.0,
            components={
                "splitter": {
                    "n_output_modes": 2,
                    "mode_areas_m2": np.array([0.75, 3.0], dtype=np.float64),
                    "field_area_reference_m2": 1.5,
                },
            },
        ),
        field=Field(amplitudes=amplitudes, wavelength_grid_um=wavelengths),
    )

    optimizer_field = _require_field(state, 2, wl_design_um=1.2)

    expected = np.array([np.sqrt(0.75), np.sqrt(3.0)], dtype=np.float64)
    np.testing.assert_allclose(optimizer_field[:, 0, 0], expected)
