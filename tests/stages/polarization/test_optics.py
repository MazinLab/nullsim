"""Tests for polarization HWP/PBS/PM-fiber stages."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nullsim.pipeline import Field, RunContext, SimulationState, Throughput
from nullsim.stages.polarization import (
    HalfWavePlate,
    PMFiberJones,
    PolarizingBeamSplitter,
)


WAVELENGTH_GRID_UM = (1.20, 1.40)


def _context(tmp_path: Path, instance_id: str = "polarization") -> RunContext:
    return RunContext(
        rng=np.random.default_rng(123),
        config_hash="polarization-optics",
        sweep_coord_tuple=(),
        instance_id=instance_id,
        run_dir=tmp_path,
    )


def _state_from_amplitudes(amplitudes: np.ndarray) -> SimulationState:
    wavelength_grid_um = tuple(
        WAVELENGTH_GRID_UM[index] for index in range(amplitudes.shape[1])
    )
    state = SimulationState.empty({"wavelength_grid_um": wavelength_grid_um})
    field = Field(
        amplitudes=amplitudes.astype(np.complex128),
        wavelength_grid_um=wavelength_grid_um,
        mode_labels=tuple(f"mode_{index}" for index in range(amplitudes.shape[0])),
    )
    throughput = Throughput(
        wavelength_grid_um=wavelength_grid_um,
        transmission_fraction=tuple(1.0 for _ in wavelength_grid_um),
    )
    return replace(state, field=field, throughput=throughput)


def test_hwp_passes_unpolarized_scalar_field_through(tmp_path: Path) -> None:
    amplitudes = np.ones((2, 2, 1), dtype=np.complex128)
    state = _state_from_amplitudes(amplitudes)
    stage = HalfWavePlate(
        {"hwp_angle_rad": np.pi / 8.0, "retardance_error_rms_rad": 0.0},
        _context(tmp_path, "hwp"),
    )

    result = stage.apply(state)

    np.testing.assert_allclose(result.field.amplitudes, amplitudes, atol=1.0e-12)
    component = result.field.components["polarization.hwp"]
    assert component["input_n_pol"] == 1
    assert component["output_n_pol"] == 1


def test_hwp_rotates_two_pol_field(tmp_path: Path) -> None:
    amplitudes = np.zeros((1, 1, 2), dtype=np.complex128)
    amplitudes[0, 0, 0] = 1.0
    state = _state_from_amplitudes(amplitudes)
    stage = HalfWavePlate(
        {"hwp_angle_rad": np.pi / 4.0, "retardance_error_rms_rad": 0.0},
        _context(tmp_path, "hwp"),
    )

    result = stage.apply(state)

    output = result.field.amplitudes[0, 0]
    assert abs(output[0]) ** 2 < 1.0e-24
    assert abs(output[1]) ** 2 == pytest.approx(1.0, abs=1.0e-12)


def test_hwp_rejects_pbs_arm_basis(tmp_path: Path) -> None:
    amplitudes = np.ones((1, 1, 2), dtype=np.complex128)
    state = _state_from_amplitudes(amplitudes)
    state = replace(
        state,
        field=state.field.add_component(
            "polarization.pbs_split",
            {
                "polarization_labels": ("A", "B"),
                "polarization_basis": "pbs_arms",
            },
        ),
    )
    stage = HalfWavePlate(
        {"hwp_angle_rad": np.pi / 8.0},
        _context(tmp_path, "hwp"),
    )

    with pytest.raises(ValueError, match="PBS-arm"):
        stage.apply(state)


def test_pbs_split_unpolarized_field_into_equal_arms(tmp_path: Path) -> None:
    amplitudes = np.ones((2, 2, 1), dtype=np.complex128)
    state = _state_from_amplitudes(amplitudes)
    stage = PolarizingBeamSplitter(
        {"insertion_loss_db": 0.5},
        _context(tmp_path, "pbs"),
    )

    result = stage.apply(state)

    output = result.field.amplitudes
    assert output.shape == (2, 2, 2)
    np.testing.assert_allclose(np.abs(output[:, :, 0]), np.abs(output[:, :, 1]))
    assert np.sum(np.abs(output) ** 2) == pytest.approx(
        amplitudes.size * 10.0 ** (-0.5 / 10.0),
        rel=1.0e-12,
    )
    component = result.field.components["polarization.pbs_split"]
    assert component["polarization_labels"] == ("A", "B")
    assert component["input_interpretation"] == "unpolarized"


def test_pbs_split_jones_field_keeps_extinction_leakage(tmp_path: Path) -> None:
    amplitudes = np.zeros((1, 1, 2), dtype=np.complex128)
    amplitudes[0, 0, 1] = 1.0
    state = _state_from_amplitudes(amplitudes)
    stage = PolarizingBeamSplitter(
        {
            "input_interpretation": "jones",
            "extinction_ratio_db": 30.0,
            "axis_offset_rad": 0.0,
            "insertion_loss_db": 0.0,
        },
        _context(tmp_path, "pbs"),
    )

    result = stage.apply(state)

    output = result.field.amplitudes[0, 0]
    assert abs(output[0]) ** 2 == pytest.approx(1.0e-3, rel=1.0e-12)
    assert abs(output[1]) ** 2 == pytest.approx(1.0 - 1.0e-3, rel=1.0e-12)


def test_pm_fiber_projects_same_axis_power_and_records_jones(tmp_path: Path) -> None:
    amplitudes = np.ones((4, 2, 2), dtype=np.complex128)
    state = _state_from_amplitudes(amplitudes)
    stage = PMFiberJones(
        {
            "fiber_length_m": 30.0,
            "per_db_floor": 20.0,
            "slow_axis_alignment_rms_rad": 0.0,
            "seed": 7,
        },
        _context(tmp_path, "pm"),
    )

    result = stage.apply(state)

    expected_power = 1.0 - 10.0 ** (-20.0 / 10.0)
    np.testing.assert_allclose(
        np.abs(result.field.amplitudes) ** 2,
        expected_power,
        atol=1.0e-12,
    )
    component = result.field.components["polarization.pm_fiber"]
    assert component["cross_pol_leakage_power"] == pytest.approx(1.0e-2)
    assert component["arm_seeds"][0] != component["arm_seeds"][1]
    result_component = result.results.components["polarization.pm_fiber"]
    assert result_component["jones_matrices_arm_a"].shape == (4, 2, 2)
    assert result_component["jones_matrices_arm_b"].shape == (4, 2, 2)


def test_pm_fiber_requires_two_pbs_arms(tmp_path: Path) -> None:
    state = _state_from_amplitudes(np.ones((2, 2, 1), dtype=np.complex128))
    stage = PMFiberJones({"seed": 7}, _context(tmp_path, "pm"))

    with pytest.raises(ValueError, match="n_pol=2"):
        stage.apply(state)
