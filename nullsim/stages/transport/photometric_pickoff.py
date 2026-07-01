"""Fixed photometric tap before the science analysis mesh."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from pydantic import BaseModel, ConfigDict, Field as PydanticField

from nullsim.pipeline import BaseStage, SimulationState, Throughput, register_stage
from nullsim.stages._utils import unique_component_name


@register_stage
class PhotometricPickoff(BaseStage):
    """Tap a fixed fraction of guided-mode light for amplitude sensing."""

    type_name = "transport.photometric_pickoff"
    family = "transport"
    consumes = frozenset({"field", "throughput"})
    produces = frozenset({"field", "throughput"})

    class Params(BaseModel):
        """Validated params for a scalar photometric pickoff."""

        model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

        tap_fraction: float = PydanticField(ge=0.0, lt=1.0)

    def apply(self, state: SimulationState) -> SimulationState:
        params = self.params_model
        field_amplitudes = state.field.amplitudes
        if field_amplitudes is None:
            raise ValueError("transport.photometric_pickoff requires field amplitudes.")
        field_array = np.asarray(field_amplitudes, dtype=np.complex128)
        if field_array.ndim != 3:
            raise ValueError(
                "transport.photometric_pickoff requires field amplitudes with "
                "shape [n_modes, n_wavelength_bins, n_pol]."
            )

        wavelength_grid_um = _require_wavelength_grid_um(state, field_array)
        tap_fraction = float(params.tap_fraction)
        science_fraction = 1.0 - tap_fraction
        new_amplitudes = field_array * np.sqrt(science_fraction)
        field = replace(state.field, amplitudes=new_amplitudes)
        throughput = _updated_throughput(
            state=state,
            wavelength_grid_um=wavelength_grid_um,
            tap_fraction=tap_fraction,
            science_fraction=science_fraction,
            n_input_modes=int(field_array.shape[0]),
            instance_id=self.context.instance_id,
        )
        return replace(state, field=field, throughput=throughput)


def _require_wavelength_grid_um(
    state: SimulationState,
    field_array: np.ndarray,
) -> tuple[float, ...]:
    wavelength_grid_um = (
        state.field.wavelength_grid_um or state.throughput.wavelength_grid_um
    )
    if not wavelength_grid_um:
        raise ValueError(
            "transport.photometric_pickoff requires a non-empty wavelength grid."
        )
    if len(wavelength_grid_um) != field_array.shape[1]:
        raise ValueError(
            "transport.photometric_pickoff field wavelength axis must match "
            f"the wavelength grid ({field_array.shape[1]} != "
            f"{len(wavelength_grid_um)})."
        )
    return tuple(float(value) for value in wavelength_grid_um)


def _updated_throughput(
    *,
    state: SimulationState,
    wavelength_grid_um: tuple[float, ...],
    tap_fraction: float,
    science_fraction: float,
    n_input_modes: int,
    instance_id: str,
) -> Throughput:
    """Record the pickoff as provenance without double-counting field loss."""

    n_wavelength_bins = len(wavelength_grid_um)
    if state.throughput.transmission_fraction:
        transmission = tuple(
            float(value) for value in state.throughput.transmission_fraction
        )
        if len(transmission) != n_wavelength_bins:
            raise ValueError(
                "transport.photometric_pickoff throughput.transmission_fraction "
                f"length {len(transmission)} does not match wavelength grid "
                f"length {n_wavelength_bins}."
            )
    else:
        transmission = tuple(1.0 for _ in wavelength_grid_um)

    component_name = unique_component_name(
        "transport.photometric_pickoff",
        state.throughput.components,
        instance_id,
    )
    component = {
        "wavelength_grid_um": wavelength_grid_um,
        "transmission_fraction": tuple(
            float(science_fraction) for _ in wavelength_grid_um
        ),
        "tap_fraction": float(tap_fraction),
        "science_fraction": float(science_fraction),
        "n_input_modes": int(n_input_modes),
        "field_amplitude_applied": True,
        "source_model": (
            "Fixed photometric tap after delay lines and before the quadrature "
            "fanout. The science-path debit is applied to field amplitudes as "
            "sqrt(1 - tap_fraction); throughput.transmission_fraction is left "
            "unchanged so detectors do not double-count the loss."
        ),
    }
    return replace(
        state.throughput,
        wavelength_grid_um=wavelength_grid_um,
        transmission_fraction=transmission,
    ).add_component(component_name, component)
