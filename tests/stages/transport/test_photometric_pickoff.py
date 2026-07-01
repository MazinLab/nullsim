"""Tests for the photometric amplitude-sensing pickoff stage."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nullsim.pipeline import (
    Field,
    RunContext,
    SimulationState,
    Throughput,
    build_instance,
    list_stages,
)
from nullsim.stages.transport import PhotometricPickoff


def _context(tmp_path: Path) -> RunContext:
    return RunContext(
        rng=np.random.default_rng(1),
        config_hash="photometric-pickoff",
        sweep_coord_tuple=(),
        instance_id="amplitude_pickoff",
        run_dir=tmp_path,
    )


def _state() -> SimulationState:
    wavelengths_um = (1.20, 1.25, 1.30)
    state = SimulationState.empty({"wavelength_grid_um": wavelengths_um})
    return replace(
        state,
        field=Field(
            amplitudes=2.0 * np.ones((4, len(wavelengths_um), 1), dtype=np.complex128),
            wavelength_grid_um=wavelengths_um,
        ),
        throughput=Throughput(
            wavelength_grid_um=wavelengths_um,
            transmission_fraction=tuple(0.5 for _ in wavelengths_um),
        ),
    )


def test_photometric_pickoff_registers_stage_contract(tmp_path: Path) -> None:
    stage = build_instance(
        {
            "id": "amplitude_pickoff",
            "type": "transport.photometric_pickoff",
            "tap_fraction": 0.30,
        },
        _context(tmp_path),
    )

    assert "transport.photometric_pickoff" in list_stages()
    assert isinstance(stage, PhotometricPickoff)
    assert stage.type_name == "transport.photometric_pickoff"
    assert stage.family == "transport"
    assert stage.consumes == frozenset({"field", "throughput"})
    assert stage.produces == frozenset({"field", "throughput"})


def test_photometric_pickoff_debits_field_without_double_counting_throughput(
    tmp_path: Path,
) -> None:
    stage = PhotometricPickoff(
        params={"tap_fraction": 0.30},
        context=_context(tmp_path),
    )

    result = stage.apply(_state())

    expected_amplitude = 2.0 * np.sqrt(0.70)
    np.testing.assert_allclose(result.field.amplitudes, expected_amplitude)
    assert result.throughput.transmission_fraction == pytest.approx((0.5, 0.5, 0.5))

    payload = result.throughput.components["transport.photometric_pickoff"]
    assert payload["tap_fraction"] == pytest.approx(0.30)
    assert payload["science_fraction"] == pytest.approx(0.70)
    assert payload["n_input_modes"] == 4
    assert payload["field_amplitude_applied"] is True
    assert payload["transmission_fraction"] == pytest.approx((0.70, 0.70, 0.70))
