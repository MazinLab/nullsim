"""Duplicated-input quadrature-reference chip stage.

This stage is an idealized architecture test for a 2N-output analysis mesh fed
by N physical aperture fields plus an internal quadrature reference copy. It is
not a Clements-parameter optimizer; it publishes the effective rectangular
``(2N outputs, N inputs)`` isometry seen by the science pipeline.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field as PydanticField, model_validator

from nullsim.physics.quadrature_fringe_tracker import quadrature_reference_mesh
from nullsim.pipeline import BaseStage, SimulationState, Throughput, register_stage
from nullsim.stages._utils import unique_component_name


@register_stage
class QuadratureReferenceMesh(BaseStage):
    """Apply an ideal quadrature-reference tracker/science isometry."""

    type_name = "chip.quadrature_reference_mesh"
    family = "chip"
    consumes = frozenset({"field", "throughput"})
    produces = frozenset({"field", "throughput", "results"})

    class Params(BaseModel):
        """Validated params for the quadrature-reference architecture test."""

        model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

        n_input_modes: int = PydanticField(gt=1)
        n_bright_ports: int = PydanticField(ge=1)
        n_dark_ports: int = PydanticField(ge=1)
        n_spectro_ports: int = PydanticField(default=0, ge=0)
        differential_coupling: float = PydanticField(ge=0.0, le=1.0)
        wl_design_um: float = PydanticField(gt=0.0)
        insertion_loss_db: float = PydanticField(default=-2.0, le=0.0)

        @model_validator(mode="after")
        def _check_port_split(self) -> QuadratureReferenceMesh.Params:
            if self.n_bright_ports != self.n_input_modes:
                raise ValueError(
                    "chip.quadrature_reference_mesh requires n_bright_ports "
                    "to equal n_input_modes so each tracker output carries one "
                    "uniform stellar reference channel."
                )
            n_science = self.n_dark_ports + self.n_spectro_ports
            if n_science != self.n_input_modes:
                raise ValueError(
                    "chip.quadrature_reference_mesh requires "
                    "n_dark_ports + n_spectro_ports == n_input_modes "
                    f"(got {self.n_dark_ports} + {self.n_spectro_ports} != "
                    f"{self.n_input_modes})."
                )
            return self

    def apply(self, state: SimulationState) -> SimulationState:
        params = self.params_model
        field_amplitudes = state.field.amplitudes
        if field_amplitudes is None:
            raise ValueError(
                "chip.quadrature_reference_mesh requires field amplitudes from "
                "an earlier stage."
            )
        field_array = np.asarray(field_amplitudes, dtype=np.complex128)
        if field_array.ndim != 3:
            raise ValueError(
                "chip.quadrature_reference_mesh requires field amplitudes with "
                "shape [n_input_modes, n_wavelength_bins, n_pol]."
            )
        if field_array.shape[0] != params.n_input_modes:
            raise ValueError(
                "chip.quadrature_reference_mesh field mode axis must match "
                f"n_input_modes ({field_array.shape[0]} != "
                f"{params.n_input_modes})."
            )

        wavelength_grid_um = _require_wavelength_grid_um(state, field_array)
        mesh = quadrature_reference_mesh(
            n_modes=int(params.n_input_modes),
            differential_coupling=float(params.differential_coupling),
        )
        transfer = np.vstack([mesh.tracker_matrix, mesh.science_matrix])
        u_stack = np.repeat(
            transfer[:, :, np.newaxis],
            len(wavelength_grid_um),
            axis=2,
        ).astype(np.complex128, copy=False)
        new_amplitudes = np.einsum("ij,jwp->iwp", transfer, field_array)

        transmission_fraction_chip = 10.0 ** (params.insertion_loss_db / 10.0)
        field_transfer_stack = u_stack
        if transmission_fraction_chip != 1.0:
            amplitude_loss = np.sqrt(transmission_fraction_chip)
            new_amplitudes = new_amplitudes * amplitude_loss
            field_transfer_stack = u_stack * amplitude_loss

        field = replace(
            state.field,
            amplitudes=new_amplitudes,
            wavelength_grid_um=wavelength_grid_um,
        )
        throughput = _updated_throughput(
            state=state,
            wavelength_grid_um=wavelength_grid_um,
            instance_id=self.context.instance_id,
            insertion_loss_db=float(params.insertion_loss_db),
            transmission_fraction_chip=float(transmission_fraction_chip),
        )

        components = dict(state.results.components)
        component_name = unique_component_name(
            "chip.kernel_mzi_mesh",
            components,
            self.context.instance_id,
        )
        components[component_name] = {
            "u_stack": u_stack.astype(np.complex128, copy=True),
            "field_transfer_stack": field_transfer_stack.astype(
                np.complex128,
                copy=True,
            ),
            "wavelength_grid_um": wavelength_grid_um,
            "wl_design_um": float(params.wl_design_um),
            "n_modes": int(params.n_input_modes),
            "n_input_modes": int(params.n_input_modes),
            "n_output_ports": int(transfer.shape[0]),
            "n_bright_ports": int(params.n_bright_ports),
            "n_dark_ports": int(params.n_dark_ports),
            "n_spectro_ports": int(params.n_spectro_ports),
            "differential_coupling": float(params.differential_coupling),
            "science_differential_throughput": float(
                1.0 - params.differential_coupling
            ),
            "tracker_differential_throughput": float(params.differential_coupling),
            "tracker_star_fraction": 1.0,
            "science_star_leakage": 0.0,
            "deployment_mode": "shared",
            "architecture": "quadrature_reference_mesh",
            "rectangular_isometry": True,
            "source_model": (
                "Ideal duplicated-input quadrature-reference architecture. The "
                "published u_stack is the lossless 2N-output by N-input "
                "rectangular isometry used by downstream observability "
                "diagnostics; field_transfer_stack includes scalar chip "
                "insertion loss and exactly reproduces the emitted field. N "
                "tracker bright rows carry the on-axis stellar reference and a "
                "tunable quadrature projection of differential piston modes; N "
                "science rows are the lossless complement and are exactly dark "
                "to the on-axis stellar mode in this ideal model."
            ),
        }
        return replace(
            state,
            field=field,
            throughput=throughput,
            results=replace(state.results, components=components),
        )


def _require_wavelength_grid_um(
    state: SimulationState,
    field_amplitudes: NDArray[np.complexfloating],
) -> tuple[float, ...]:
    wavelength_grid_um = (
        state.field.wavelength_grid_um or state.throughput.wavelength_grid_um
    )
    if not wavelength_grid_um:
        raise ValueError(
            "chip.quadrature_reference_mesh requires a non-empty wavelength grid."
        )
    if field_amplitudes.shape[1] != len(wavelength_grid_um):
        raise ValueError(
            "chip.quadrature_reference_mesh field wavelength axis must match "
            "the wavelength grid."
        )
    return tuple(float(value) for value in wavelength_grid_um)


def _updated_throughput(
    *,
    state: SimulationState,
    wavelength_grid_um: tuple[float, ...],
    instance_id: str,
    insertion_loss_db: float,
    transmission_fraction_chip: float,
) -> Throughput:
    n_wavelength_bins = len(wavelength_grid_um)
    current_transmission = _throughput_or_unity(state, n_wavelength_bins)
    component_payload = {
        "wavelength_grid_um": wavelength_grid_um,
        "transmission_fraction": tuple(
            transmission_fraction_chip for _ in range(n_wavelength_bins)
        ),
        "field_amplitude_applied": True,
        "insertion_loss_db": float(insertion_loss_db),
        "source_model": (
            "Quadrature-reference analysis mesh insertion loss "
            f"({insertion_loss_db:+.2f} dB -> "
            f"t = {transmission_fraction_chip:.6f}). Applied to field "
            "amplitudes only; recorded for the loss budget without folding "
            "into throughput.transmission_fraction to avoid double-counting."
        ),
    }
    components = dict(state.throughput.components)
    component_name = unique_component_name(
        "chip.quadrature_reference_mesh_loss",
        components,
        instance_id,
    )
    components[component_name] = component_payload
    return Throughput(
        wavelength_grid_um=wavelength_grid_um,
        transmission_fraction=tuple(float(v) for v in current_transmission),
        components=components,
    )


def _throughput_or_unity(
    state: SimulationState,
    n_wavelength_bins: int,
) -> NDArray[np.float64]:
    if state.throughput.transmission_fraction:
        if len(state.throughput.transmission_fraction) != n_wavelength_bins:
            raise ValueError(
                "chip.quadrature_reference_mesh throughput length must match "
                "the wavelength grid."
            )
        return np.asarray(state.throughput.transmission_fraction, dtype=np.float64)
    return np.ones(n_wavelength_bins, dtype=np.float64)


__all__ = ["QuadratureReferenceMesh"]
