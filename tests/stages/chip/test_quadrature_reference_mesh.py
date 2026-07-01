"""Tests for the ideal quadrature-reference mesh chip stage."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nullsim.pipeline import Field, RunContext, SimulationState, Throughput, list_stages
from nullsim.stages.chip import QuadratureReferenceMesh


def _context(tmp_path: Path) -> RunContext:
    return RunContext(
        rng=np.random.default_rng(0),
        config_hash="quadrature-reference-mesh",
        sweep_coord_tuple=(),
        instance_id="chip",
        run_dir=tmp_path,
    )


def _state(
    *,
    n_input_modes: int = 4,
    wavelength_grid_um: tuple[float, ...] = (1.2, 1.3),
) -> SimulationState:
    amplitudes = np.ones(
        (n_input_modes, len(wavelength_grid_um), 1),
        dtype=np.complex128,
    )
    state = SimulationState.empty({"wavelength_grid_um": wavelength_grid_um})
    return replace(
        state,
        field=Field(
            amplitudes=amplitudes,
            wavelength_grid_um=wavelength_grid_um,
            mode_labels=tuple(f"mode_{idx}" for idx in range(n_input_modes)),
        ),
        throughput=Throughput(
            wavelength_grid_um=wavelength_grid_um,
            transmission_fraction=tuple(1.0 for _ in wavelength_grid_um),
        ),
    )


def test_quadrature_reference_mesh_registers_stage_contract() -> None:
    assert "chip.quadrature_reference_mesh" in list_stages()
    assert QuadratureReferenceMesh.type_name == "chip.quadrature_reference_mesh"
    assert QuadratureReferenceMesh.family == "chip"
    assert QuadratureReferenceMesh.consumes == frozenset({"field", "throughput"})
    assert QuadratureReferenceMesh.produces == frozenset(
        {"field", "throughput", "results"}
    )


def test_quadrature_reference_mesh_publishes_rectangular_isometry(
    tmp_path: Path,
) -> None:
    stage = QuadratureReferenceMesh(
        params={
            "n_input_modes": 4,
            "n_bright_ports": 4,
            "n_dark_ports": 3,
            "n_spectro_ports": 1,
            "differential_coupling": 0.1,
            "wl_design_um": 1.25,
            "insertion_loss_db": 0.0,
        },
        context=_context(tmp_path),
    )

    out = stage.apply(_state())
    output_field = out.field.amplitudes
    assert output_field is not None
    assert output_field.shape == (8, 2, 1)

    payload = out.results.components["chip.kernel_mzi_mesh"]
    u_stack = np.asarray(payload["u_stack"])
    assert u_stack.shape == (8, 4, 2)
    assert payload["n_input_modes"] == 4
    assert payload["n_output_ports"] == 8
    assert payload["rectangular_isometry"] is True
    np.testing.assert_allclose(
        u_stack[:, :, 0].conj().T @ u_stack[:, :, 0],
        np.eye(4),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        output_field,
        np.einsum("ijw,jwp->iwp", u_stack, _state().field.amplitudes),
        atol=1.0e-14,
    )


def test_quadrature_reference_mesh_field_transfer_stack_includes_loss(
    tmp_path: Path,
) -> None:
    insertion_loss_db = -2.0
    transmission = 10.0 ** (insertion_loss_db / 10.0)
    stage = QuadratureReferenceMesh(
        params={
            "n_input_modes": 4,
            "n_bright_ports": 4,
            "n_dark_ports": 3,
            "n_spectro_ports": 1,
            "differential_coupling": 0.1,
            "wl_design_um": 1.25,
            "insertion_loss_db": insertion_loss_db,
        },
        context=_context(tmp_path),
    )

    input_state = _state()
    out = stage.apply(input_state)
    output_field = out.field.amplitudes
    assert output_field is not None
    payload = out.results.components["chip.kernel_mzi_mesh"]
    u_stack = np.asarray(payload["u_stack"])
    field_transfer_stack = np.asarray(payload["field_transfer_stack"])

    np.testing.assert_allclose(
        u_stack[:, :, 0].conj().T @ u_stack[:, :, 0],
        np.eye(4),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        field_transfer_stack,
        np.sqrt(transmission) * u_stack,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        output_field,
        np.einsum("ijw,jwp->iwp", field_transfer_stack, input_state.field.amplitudes),
        atol=1.0e-14,
    )


def test_quadrature_reference_mesh_science_rows_are_dark_to_star(
    tmp_path: Path,
) -> None:
    stage = QuadratureReferenceMesh(
        params={
            "n_input_modes": 6,
            "n_bright_ports": 6,
            "n_dark_ports": 5,
            "n_spectro_ports": 1,
            "differential_coupling": 0.003,
            "wl_design_um": 1.25,
            "insertion_loss_db": 0.0,
        },
        context=_context(tmp_path),
    )

    out = stage.apply(_state(n_input_modes=6, wavelength_grid_um=(1.25,)))
    output_field = out.field.amplitudes
    assert output_field is not None
    science_rows = output_field[6:, 0, 0]
    np.testing.assert_allclose(science_rows, 0.0, atol=1.0e-14)


def test_quadrature_reference_mesh_rejects_non_complementary_port_split(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="n_dark_ports \\+ n_spectro_ports"):
        QuadratureReferenceMesh(
            params={
                "n_input_modes": 4,
                "n_bright_ports": 4,
                "n_dark_ports": 4,
                "n_spectro_ports": 1,
                "differential_coupling": 0.1,
                "wl_design_um": 1.25,
            },
            context=_context(tmp_path),
        )
