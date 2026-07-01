"""Architecture placeholder chip stage with an identity transfer matrix.

This module exists to exercise the vertical-slice pipeline from SPEC §15.1.
It is not a physics implementation of a Clements mesh, kernel nuller, or chip
optimization.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from pydantic import BaseModel, ConfigDict, Field as PydanticField

from nullsim.pipeline import BaseStage, SimulationState, register_stage


@register_stage
class IdentityChip(BaseStage):
    """Pass field amplitudes through, with an optional scalar insertion loss.

    The unitary is identity by construction; the only optical effect this
    stage models is a wavelength-flat fabrication insertion loss
    (``insertion_loss_db``, default ``-2.0`` dB) applied as ``sqrt(t)`` on
    field amplitudes, where ``t = 10 ** (insertion_loss_db / 10)``. The
    cumulative throughput spectrum is upstream-only; the chip component
    records ``t`` for diagnostics and loss-budget plots. Set
    ``insertion_loss_db = 0.0`` for a true lossless pass-through.
    """

    type_name = "chip.identity"
    family = "chip"
    consumes = frozenset({"field", "throughput"})
    produces = frozenset({"field", "throughput"})

    class Params(BaseModel):
        """Validated params for the placeholder identity chip."""

        model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True)

        insertion_loss_db: float = PydanticField(default=-2.0, le=0.0)

    def apply(self, state: SimulationState) -> SimulationState:
        params = self.params_model
        field_amplitudes = state.field.amplitudes
        if field_amplitudes is None:
            raise ValueError("chip.identity requires field amplitudes from an earlier stage.")
        if field_amplitudes.ndim != 3:
            raise ValueError(
                "chip.identity requires field amplitudes with shape "
                "[n_modes, n_wavelength_bins, n_pol]."
            )

        wavelength_grid_um = state.throughput.wavelength_grid_um
        n_wavelength_bins = len(wavelength_grid_um)
        if field_amplitudes.shape[1] != n_wavelength_bins:
            raise ValueError(
                "chip.identity field wavelength axis must match throughput wavelength grid."
            )

        transmission_fraction_chip = 10.0 ** (params.insertion_loss_db / 10.0)
        if transmission_fraction_chip != 1.0:
            new_amplitudes = field_amplitudes * np.sqrt(transmission_fraction_chip)
            field = replace(state.field, amplitudes=new_amplitudes)
        else:
            field = state.field

        # Chip loss is applied in field amplitudes only — the detector
        # picks it up via |field|². The cumulative transmission_fraction
        # records UPSTREAM throughput and must pass through unchanged;
        # the chip_identity component records the t factor for the
        # loss-budget plot. See chip.kernel_mzi_mesh._updated_throughput
        # for the same convention.
        component_payload = {
            "wavelength_grid_um": tuple(float(w) for w in wavelength_grid_um),
            "transmission_fraction": tuple(
                transmission_fraction_chip for _ in range(n_wavelength_bins)
            ),
            "field_amplitude_applied": True,
            "insertion_loss_db": float(params.insertion_loss_db),
            "source_model": (
                f"Identity unitary × scalar insertion loss "
                f"({params.insertion_loss_db:+.2f} dB → "
                f"t = {transmission_fraction_chip:.6f}). Applied to field "
                "amplitudes only (sqrt(t)); not folded into transmission_fraction."
            ),
        }
        throughput = state.throughput.add_component("chip_identity", component_payload)
        return replace(state, field=field, throughput=throughput)


def _throughput_or_unity(
    state: SimulationState,
    n_wavelength_bins: int,
) -> tuple[float, ...]:
    if state.throughput.transmission_fraction:
        if len(state.throughput.transmission_fraction) != n_wavelength_bins:
            raise ValueError(
                "chip.identity throughput length must match the wavelength grid."
            )
        return tuple(float(value) for value in state.throughput.transmission_fraction)
    return tuple(1.0 for _ in range(n_wavelength_bins))


__all__ = ["IdentityChip"]
