"""Tests for the Ruilier-Cassaing single-mode fiber injection stage."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from nullsim.physics import ruilier_cassaing
from nullsim.pipeline import (
    Field,
    Geometry,
    RunContext,
    SimulationState,
    Throughput,
    Wavefront,
)
from nullsim.stages.injection.single_mode_fiber import SingleModeFiber


def _context(tmp_path: Path) -> RunContext:
    return RunContext(
        rng=np.random.default_rng(1),
        config_hash="single-mode-fiber-test",
        sweep_coord_tuple=(),
        instance_id="injection",
        run_dir=tmp_path,
    )


def _state(
    *,
    wavelength_grid_um: tuple[float, ...] = (1.2, 1.6, 2.0),
    strehl_ratio: tuple[float, ...] = (1.0, 1.0, 1.0),
    amplitudes: np.ndarray | None = None,
    transmission_fraction: tuple[float, ...] | None = None,
) -> SimulationState:
    n_wavelength_bins = len(wavelength_grid_um)
    if amplitudes is None:
        amplitudes = np.ones((2, n_wavelength_bins, 1), dtype=np.complex128)
    if transmission_fraction is None:
        transmission_fraction = tuple(1.0 for _ in wavelength_grid_um)

    state = SimulationState.empty({"wavelength_grid_um": wavelength_grid_um})
    geometry = Geometry(
        diameter_m=4.0,
        area_m2=math.pi * 2.0**2,
        n_apertures=1,
    )
    field = Field(
        amplitudes=amplitudes,
        wavelength_grid_um=wavelength_grid_um,
        mode_labels=("mode_0", "mode_1"),
    )
    throughput = Throughput(
        wavelength_grid_um=wavelength_grid_um,
        transmission_fraction=transmission_fraction,
    )
    wavefront = Wavefront(
        wavelength_grid_um=wavelength_grid_um,
        strehl_ratio=strehl_ratio,
        amplitude_jitter_fraction=tuple(0.0 for _ in wavelength_grid_um),
    )
    return replace(
        state,
        geometry=geometry,
        field=field,
        throughput=throughput,
        wavefront=wavefront,
    )


def _stage(tmp_path: Path, **params: object) -> SingleModeFiber:
    return SingleModeFiber(params, _context(tmp_path))


def _coupling_component(state: SimulationState) -> np.ndarray:
    component = state.throughput.components["single_mode_fiber"]
    return np.asarray(component["coupling_efficiency"], dtype=np.float64)


def test_single_mode_fiber_registers_stage_contract(tmp_path: Path) -> None:
    stage = _stage(tmp_path)

    assert stage.type_name == "injection.single_mode_fiber"
    assert stage.family == "injection"
    assert stage.consumes == frozenset({"field", "throughput", "wavefront", "geometry"})
    assert stage.produces == frozenset({"field", "throughput"})


@pytest.mark.parametrize("bad_obscuration", [-0.1, 1.0, math.inf])
def test_params_reject_invalid_central_obscuration_ratio(
    bad_obscuration: float,
) -> None:
    with pytest.raises(ValidationError, match="central_obscuration_ratio"):
        SingleModeFiber.Params(central_obscuration_ratio=bad_obscuration)


def test_params_reject_unknown_keys() -> None:
    with pytest.raises(ValidationError, match="unknown_key"):
        SingleModeFiber.Params(unknown_key=1)


def test_params_reject_nan() -> None:
    with pytest.raises(ValidationError, match="central_obscuration_ratio"):
        SingleModeFiber.Params(central_obscuration_ratio=math.nan)


def test_zero_strehl_zeroes_field_amplitude_only(tmp_path: Path) -> None:
    # Coupling loss is baked into field amplitudes (sqrt) and recorded as a
    # provenance component; it is NOT re-applied to throughput.transmission_fraction
    # (the detector multiplies |field|² × throughput, so double-application would
    # square the coupling factor — see Finding 1 fix).
    stage = _stage(tmp_path)
    state = _state(strehl_ratio=(0.0, 0.0, 0.0))

    result = stage.apply(state)

    assert result.field.amplitudes is not None
    np.testing.assert_allclose(result.field.amplitudes, 0.0 + 0.0j)
    np.testing.assert_allclose(_coupling_component(result), np.zeros(3))
    assert result.throughput.transmission_fraction == pytest.approx((1.0, 1.0, 1.0))


def test_unity_strehl_couples_at_geometric_limit(tmp_path: Path) -> None:
    stage = _stage(tmp_path)
    state = _state(wavelength_grid_um=(1.65,), strehl_ratio=(1.0,))

    result = stage.apply(state)

    expected_coupling = ruilier_cassaing.geometric_coupling_limit(0.0)
    assert _coupling_component(result)[0] == pytest.approx(expected_coupling)
    # Coupling is recorded as a provenance component but NOT folded into
    # transmission_fraction (avoids double-count at detector).
    assert result.throughput.transmission_fraction[0] == pytest.approx(1.0)
    assert result.field.amplitudes is not None
    np.testing.assert_allclose(result.field.amplitudes, math.sqrt(expected_coupling))


def test_nan_strehl_propagates_to_matching_wavelength_bin(tmp_path: Path) -> None:
    stage = _stage(tmp_path)
    state = _state(strehl_ratio=(1.0, math.nan, 0.5))

    result = stage.apply(state)

    assert result.field.amplitudes is not None
    coupling = _coupling_component(result)
    assert np.isfinite(coupling[0])
    assert math.isnan(coupling[1])
    assert np.isfinite(coupling[2])
    # NaN coupling propagates to field amplitude (sqrt(NaN)=NaN) and to the
    # coupling_efficiency provenance component, but transmission_fraction is
    # left untouched at the upstream value (no double-count).
    assert np.isnan(result.field.amplitudes[:, 1, :]).all()
    assert result.throughput.transmission_fraction[1] == pytest.approx(1.0)


def test_field_amplitude_uses_square_root_of_power_coupling(tmp_path: Path) -> None:
    stage = _stage(tmp_path, central_obscuration_ratio=0.0)
    state = _state(strehl_ratio=(0.5, 0.5, 0.5))

    result = stage.apply(state)

    expected_amplitude = math.sqrt(
        0.5 * ruilier_cassaing.UNOBSTRUCTED_GEOMETRIC_COUPLING_LIMIT,
    )
    assert result.field.amplitudes is not None
    np.testing.assert_allclose(
        result.field.amplitudes,
        expected_amplitude + 0.0j,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_missing_strehl_raises_stage_context_error(tmp_path: Path) -> None:
    stage = _stage(tmp_path)
    state = replace(_state(), wavefront=Wavefront(wavelength_grid_um=(1.2, 1.6, 2.0)))

    with pytest.raises(ValueError, match="injection.single_mode_fiber.*strehl_ratio"):
        stage.apply(state)


def test_throughput_component_records_geometry_provenance(tmp_path: Path) -> None:
    stage = _stage(
        tmp_path,
        central_obscuration_ratio=0.2,
        pupil_radius_to_mode_field_radius_ratio=1.3,
    )
    state = _state(wavelength_grid_um=(1.65,), strehl_ratio=(1.0,))

    result = stage.apply(state)

    component = result.throughput.components["single_mode_fiber"]
    assert component["aperture_diameter_m"] == 4.0
    assert component["central_obscuration_ratio"] == 0.2
    assert component["pupil_radius_to_mode_field_radius_ratio"] == 1.3


def test_two_instances_with_distinct_ids_do_not_collide(tmp_path: Path) -> None:
    """INS-9-005: distinct instance_ids must disambiguate component names."""
    first_context = RunContext(
        rng=np.random.default_rng(1),
        config_hash="single-mode-fiber-test",
        sweep_coord_tuple=(),
        instance_id="injection_a",
        run_dir=tmp_path,
    )
    second_context = RunContext(
        rng=np.random.default_rng(2),
        config_hash="single-mode-fiber-test",
        sweep_coord_tuple=(),
        instance_id="injection_b",
        run_dir=tmp_path,
    )

    first_stage = SingleModeFiber({}, first_context)
    second_stage = SingleModeFiber({}, second_context)

    state = _state(wavelength_grid_um=(1.65,), strehl_ratio=(1.0,))
    first_result = first_stage.apply(state)
    second_result = second_stage.apply(first_result)

    component_names = set(second_result.throughput.components)
    assert "single_mode_fiber" in component_names
    assert any(
        name != "single_mode_fiber" and name.startswith("single_mode_fiber")
        for name in component_names
    )
