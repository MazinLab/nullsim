"""Dual-polarization HWP/PBS/PM-fiber stages (SPEC §15.2).

These stages wire the pure Jones primitives into the existing nullsim field
contract: ``Field.amplitudes`` is always ``(n_modes, n_wavelength_bins, n_pol)``.
The PBS output uses the polarization axis as two physical arms, ``A`` and
``B``. Downstream chip stages preserve that axis, and detector stages can
either emit per-arm physical channels or collapse it explicitly.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field as PydanticField

from nullsim.physics.jones import (
    PMFiberConfig,
    hwp_retardance,
    jones_hwp,
    jones_pm_fiber,
    polarizing_beam_splitter_arm,
    pm_fiber_per_at_length,
)
from nullsim.pipeline import BaseStage, Field, SimulationState, Throughput, register_stage
from nullsim.stages._utils import unique_component_name


@register_stage
class HalfWavePlate(BaseStage):
    """Apply an achromatic HWP Jones rotation across the field polarization axis."""

    type_name = "polarization.hwp"
    family = "polarization"
    consumes = frozenset({"field", "throughput"})
    produces = frozenset({"field", "throughput"})

    class Params(BaseModel):
        """Validated HWP stage parameters."""

        model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

        hwp_angle_rad: float = 0.0
        design_wavelength_um: float = PydanticField(default=1.25, gt=0.0)
        retardance_error_rms_rad: float = PydanticField(default=0.02, ge=0.0)
        band_half_width_lambda: float = PydanticField(default=0.25, gt=0.0)
        insertion_loss_db: float = PydanticField(default=0.0, ge=0.0)
        pass_unpolarized_scalar: bool = True

    def apply(self, state: SimulationState) -> SimulationState:
        params = self.params_model
        field_amplitudes = _require_field_amplitudes(state, self.type_name)
        wavelength_grid_um = _require_wavelength_grid_um(
            state,
            field_amplitudes,
            self.type_name,
        )

        n_pol = field_amplitudes.shape[2]
        if n_pol == 1:
            if not params.pass_unpolarized_scalar:
                raise ValueError(
                    "polarization.hwp received n_pol=1. Set "
                    "pass_unpolarized_scalar=true or run after a two-pol "
                    "field-producing stage."
                )
            new_amplitudes = field_amplitudes.astype(np.complex128, copy=True)
        elif n_pol == 2:
            if _polarization_basis(state) == "pbs_arms":
                raise ValueError(
                    "polarization.hwp received n_pol=2 PBS-arm amplitudes. "
                    "HWP expects a coherent Jones basis; place it before "
                    "polarization.pbs_split or convert the basis explicitly."
                )
            new_amplitudes = _apply_hwp_jones(
                field_amplitudes=field_amplitudes,
                wavelength_grid_um=wavelength_grid_um,
                hwp_angle_rad=float(params.hwp_angle_rad),
                design_wavelength_um=float(params.design_wavelength_um),
                retardance_error_rms_rad=float(params.retardance_error_rms_rad),
                band_half_width_lambda=float(params.band_half_width_lambda),
            )
        else:
            raise ValueError(
                "polarization.hwp requires field amplitudes with n_pol axis "
                f"1 or 2, got {n_pol}."
            )

        transmission_fraction = 10.0 ** (-float(params.insertion_loss_db) / 10.0)
        if transmission_fraction != 1.0:
            new_amplitudes = new_amplitudes * np.sqrt(transmission_fraction)

        field = _add_field_component(
            replace(state.field, amplitudes=new_amplitudes),
            self.type_name,
            state.field.components,
            self.context.instance_id,
            {
                "hwp_angle_rad": float(params.hwp_angle_rad),
                "design_wavelength_um": float(params.design_wavelength_um),
                "retardance_error_rms_rad": float(params.retardance_error_rms_rad),
                "band_half_width_lambda": float(params.band_half_width_lambda),
                "input_n_pol": int(n_pol),
                "output_n_pol": int(new_amplitudes.shape[2]),
                "polarization_basis": "scalar" if n_pol == 1 else "jones",
                "source_model": (
                    "Achromatic HWP Jones rotation using "
                    "nullsim.physics.jones.hwp_retardance/jones_hwp."
                ),
            },
        )
        throughput = _add_throughput_component(
            state=state,
            type_name=self.type_name,
            instance_id=self.context.instance_id,
            wavelength_grid_um=wavelength_grid_um,
            transmission_fraction=tuple(
                transmission_fraction for _ in wavelength_grid_um
            ),
            payload={
                "insertion_loss_db": float(params.insertion_loss_db),
                "field_amplitude_applied": True,
                "source_model": (
                    "HWP insertion loss applied to field amplitudes only; "
                    "top-level transmission_fraction is unchanged."
                ),
            },
        )
        return replace(state, field=field, throughput=throughput)


@register_stage
class PolarizingBeamSplitter(BaseStage):
    """Split a scalar or Jones field into PBS arms A/B on the n_pol axis."""

    type_name = "polarization.pbs_split"
    family = "polarization"
    consumes = frozenset({"field", "throughput"})
    produces = frozenset({"field", "throughput"})

    class Params(BaseModel):
        """Validated PBS stage parameters."""

        model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

        insertion_loss_db: float = PydanticField(default=0.5, ge=0.0)
        extinction_ratio_db: float = PydanticField(default=30.0, ge=0.0)
        axis_offset_rad: float = 1.0e-3
        input_interpretation: Literal["auto", "unpolarized", "jones"] = "auto"

    def apply(self, state: SimulationState) -> SimulationState:
        params = self.params_model
        field_amplitudes = _require_field_amplitudes(state, self.type_name)
        wavelength_grid_um = _require_wavelength_grid_um(
            state,
            field_amplitudes,
            self.type_name,
        )
        n_pol = field_amplitudes.shape[2]
        interpretation = _resolve_pbs_interpretation(
            str(params.input_interpretation), n_pol
        )

        if interpretation == "unpolarized":
            if n_pol != 1:
                raise ValueError(
                    "polarization.pbs_split input_interpretation='unpolarized' "
                    f"requires n_pol=1, got {n_pol}."
                )
            transmission_fraction = 10.0 ** (-float(params.insertion_loss_db) / 10.0)
            arm_amplitude = np.sqrt(transmission_fraction / 2.0)
            new_amplitudes = np.concatenate(
                [
                    field_amplitudes * arm_amplitude,
                    field_amplitudes * arm_amplitude,
                ],
                axis=2,
            ).astype(np.complex128, copy=False)
        else:
            if n_pol != 2:
                raise ValueError(
                    "polarization.pbs_split input_interpretation='jones' "
                    f"requires n_pol=2, got {n_pol}."
                )
            new_amplitudes = _project_jones_field_to_pbs_arms(
                field_amplitudes=field_amplitudes,
                extinction_ratio_db=float(params.extinction_ratio_db),
                axis_offset_rad=float(params.axis_offset_rad),
                insertion_loss_db=float(params.insertion_loss_db),
            )
            transmission_fraction = _mean_power_ratio(field_amplitudes, new_amplitudes)

        field = _add_field_component(
            replace(
                state.field,
                amplitudes=new_amplitudes,
            ),
            self.type_name,
            state.field.components,
            self.context.instance_id,
            {
                "input_n_pol": int(n_pol),
                "output_n_pol": 2,
                "polarization_labels": ("A", "B"),
                "polarization_basis": "pbs_arms",
                "input_interpretation": interpretation,
                "extinction_ratio_db": float(params.extinction_ratio_db),
                "axis_offset_rad": float(params.axis_offset_rad),
                "source_model": (
                    "Unpolarized inputs split 50/50; Jones inputs projected "
                    "with imperfect PBS arm matrices and NAYRA column-sum "
                    "arm projection."
                ),
            },
        )
        throughput = _add_throughput_component(
            state=state,
            type_name=self.type_name,
            instance_id=self.context.instance_id,
            wavelength_grid_um=wavelength_grid_um,
            transmission_fraction=tuple(
                float(transmission_fraction) for _ in wavelength_grid_um
            ),
            payload={
                "insertion_loss_db": float(params.insertion_loss_db),
                "extinction_ratio_db": float(params.extinction_ratio_db),
                "axis_offset_rad": float(params.axis_offset_rad),
                "field_amplitude_applied": True,
                "source_model": (
                    "PBS bulk loss applied in field amplitudes. Extinction "
                    "and axis offset affect Jones inputs; unpolarized inputs "
                    "remain a 50/50 split."
                ),
            },
        )
        return replace(state, field=field, throughput=throughput)


@register_stage
class PMFiberJones(BaseStage):
    """Project two PBS arms through PM-fiber Jones matrices."""

    type_name = "polarization.pm_fiber"
    family = "polarization"
    consumes = frozenset({"field", "throughput"})
    produces = frozenset({"field", "throughput", "results"})
    # With the default seed=None the birefringence realization is drawn
    # from context.rng (via _resolve_seed), so the cache key must fold in
    # the RNG identity — otherwise two runs differing only in [run].seed
    # cache-hit each other's stochastic draw (review 2026-06-10 Issue 25).
    uses_rng = True

    class Params(BaseModel):
        """Validated PM-fiber polarization parameters."""

        model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

        fiber_length_m: float = PydanticField(default=30.0, ge=0.0)
        per_db_per_100m: float = PydanticField(default=30.0, ge=0.0)
        per_db_floor: float = PydanticField(default=30.0, ge=0.0)
        slow_axis_alignment_rms_rad: float = PydanticField(default=17.0e-3, ge=0.0)
        seed: int | None = None

    def apply(self, state: SimulationState) -> SimulationState:
        params = self.params_model
        field_amplitudes = _require_field_amplitudes(state, self.type_name)
        wavelength_grid_um = _require_wavelength_grid_um(
            state,
            field_amplitudes,
            self.type_name,
        )
        if field_amplitudes.shape[2] != 2:
            raise ValueError(
                "polarization.pm_fiber requires n_pol=2 PBS-arm amplitudes; "
                f"got n_pol={field_amplitudes.shape[2]}."
            )
        if _polarization_basis(state) == "jones":
            raise ValueError(
                "polarization.pm_fiber received a coherent Jones-basis field. "
                "Run polarization.pbs_split first so n_pol=2 represents PBS arms."
            )

        config = PMFiberConfig(
            per_db_per_100m=float(params.per_db_per_100m),
            per_db_floor=float(params.per_db_floor),
            slow_axis_alignment_rms_rad=float(params.slow_axis_alignment_rms_rad),
        )
        seed = _resolve_seed(params.seed, getattr(self.context, "rng", None))
        seed_sequence = np.random.SeedSequence(seed)
        seed_a, seed_b = (
            int(child.generate_state(1, dtype=np.uint32)[0])
            for child in seed_sequence.spawn(2)
        )
        jones_matrices_a = jones_pm_fiber(
            field_amplitudes.shape[0],
            config,
            fiber_length_m=float(params.fiber_length_m),
            seed=seed_a,
        )
        jones_matrices_b = jones_pm_fiber(
            field_amplitudes.shape[0],
            config,
            fiber_length_m=float(params.fiber_length_m),
            seed=seed_b,
        )
        new_amplitudes = field_amplitudes.astype(np.complex128, copy=True)
        new_amplitudes[:, :, 0] *= jones_matrices_a[:, None, 0, 0]
        new_amplitudes[:, :, 1] *= jones_matrices_b[:, None, 0, 0]

        retained_power_fraction = _mean_power_ratio(field_amplitudes, new_amplitudes)
        leakage_power = pm_fiber_per_at_length(config, float(params.fiber_length_m))

        field = _add_field_component(
            replace(state.field, amplitudes=new_amplitudes),
            self.type_name,
            state.field.components,
            self.context.instance_id,
            {
                "input_n_pol": 2,
                "output_n_pol": 2,
                "polarization_labels": ("A", "B"),
                "polarization_basis": "pbs_arms",
                "retained_power_fraction_mean": float(retained_power_fraction),
                "cross_pol_leakage_power": float(leakage_power),
                "seed": seed,
                "arm_seeds": (seed_a, seed_b),
                # Per-mode, per-arm J00 this stage multiplied into the field,
                # shape (n_modes, n_wl, 2), for stages that rebuild fields
                # (review 2026-11-07 row 47).
                "static_field_factor": np.broadcast_to(
                    np.stack(
                        [jones_matrices_a[:, 0, 0], jones_matrices_b[:, 0, 0]],
                        axis=-1,
                    )[:, np.newaxis, :],
                    field_amplitudes.shape,
                ).astype(np.complex128, copy=True),
                "source_model": (
                    "Independent NAYRA PM-fiber Jones draws per PBS arm; each "
                    "arm keeps its same-axis chip-coupled component and treats "
                    "cross-polarized light as chip-coupling loss."
                ),
            },
        )
        throughput = _add_throughput_component(
            state=state,
            type_name=self.type_name,
            instance_id=self.context.instance_id,
            wavelength_grid_um=wavelength_grid_um,
            transmission_fraction=tuple(
                float(retained_power_fraction) for _ in wavelength_grid_um
            ),
            payload={
                "fiber_length_m": float(params.fiber_length_m),
                "per_db_per_100m": float(params.per_db_per_100m),
                "per_db_floor": float(params.per_db_floor),
                "slow_axis_alignment_rms_rad": float(
                    params.slow_axis_alignment_rms_rad
                ),
                "cross_pol_leakage_power": float(leakage_power),
                "field_amplitude_applied": True,
                "source_model": (
                    "Mean same-axis PM-fiber power retained by the two chip-coupled "
                    "PBS arms. Applied to field amplitudes only."
                ),
            },
        )
        results_name = unique_component_name(
            self.type_name,
            state.results.components,
            self.context.instance_id,
        )
        results = state.results.add_component(
            results_name,
            {
                "jones_matrices_arm_a": jones_matrices_a.copy(),
                "jones_matrices_arm_b": jones_matrices_b.copy(),
                "fiber_length_m": float(params.fiber_length_m),
                "cross_pol_leakage_power": float(leakage_power),
                "retained_power_fraction_mean": float(retained_power_fraction),
                "seed": seed,
                "arm_seeds": (seed_a, seed_b),
            },
        )
        return replace(state, field=field, throughput=throughput, results=results)


def _require_field_amplitudes(
    state: SimulationState,
    type_name: str,
) -> NDArray[np.complexfloating]:
    field_amplitudes = state.field.amplitudes
    if field_amplitudes is None:
        raise ValueError(f"{type_name} requires field amplitudes from an earlier stage.")
    field_array = np.asarray(field_amplitudes)
    if field_array.ndim != 3:
        raise ValueError(
            f"{type_name} requires field amplitudes with shape "
            "[n_modes, n_wavelength_bins, n_pol]."
        )
    return field_array


def _require_wavelength_grid_um(
    state: SimulationState,
    field_amplitudes: NDArray[np.complexfloating],
    type_name: str,
) -> tuple[float, ...]:
    field_grid_um = state.field.wavelength_grid_um
    throughput_grid_um = state.throughput.wavelength_grid_um
    wavelength_grid_um = field_grid_um or throughput_grid_um
    if not wavelength_grid_um:
        raise ValueError(f"{type_name} requires a non-empty wavelength grid.")
    if field_grid_um and throughput_grid_um and field_grid_um != throughput_grid_um:
        raise ValueError(f"{type_name} requires field and throughput grids to match.")
    if field_amplitudes.shape[1] != len(wavelength_grid_um):
        raise ValueError(f"{type_name} field wavelength axis must match the grid.")
    if state.throughput.transmission_fraction and (
        len(state.throughput.transmission_fraction) != len(wavelength_grid_um)
    ):
        raise ValueError(f"{type_name} throughput length must match the grid.")
    return tuple(float(value) for value in wavelength_grid_um)


def _apply_hwp_jones(
    *,
    field_amplitudes: NDArray[np.complexfloating],
    wavelength_grid_um: tuple[float, ...],
    hwp_angle_rad: float,
    design_wavelength_um: float,
    retardance_error_rms_rad: float,
    band_half_width_lambda: float,
) -> NDArray[np.complex128]:
    wavelengths_m = np.asarray(wavelength_grid_um, dtype=np.float64) * 1.0e-6
    retardance = hwp_retardance(
        wavelengths_m,
        design_wavelength_m=design_wavelength_um * 1.0e-6,
        retardance_error_rms_rad=retardance_error_rms_rad,
        band_half_width_lambda=band_half_width_lambda,
    )
    matrices = np.stack(
        [
            jones_hwp(hwp_angle_rad, float(retardance_for_wavelength))
            for retardance_for_wavelength in np.asarray(retardance, dtype=np.float64)
        ],
        axis=0,
    )
    return np.einsum(
        "wij,mwj->mwi",
        matrices,
        field_amplitudes.astype(np.complex128, copy=False),
        optimize=True,
    )


def _resolve_pbs_interpretation(interpretation: str, n_pol: int) -> str:
    if interpretation != "auto":
        return interpretation
    return "unpolarized" if n_pol == 1 else "jones"


def _polarization_basis(state: SimulationState) -> str | None:
    for payload in reversed(tuple(state.field.components.values())):
        if not isinstance(payload, dict) and not hasattr(payload, "get"):
            continue
        basis = payload.get("polarization_basis")
        if isinstance(basis, str):
            return basis
    return None


def _project_jones_field_to_pbs_arms(
    *,
    field_amplitudes: NDArray[np.complexfloating],
    extinction_ratio_db: float,
    axis_offset_rad: float,
    insertion_loss_db: float,
) -> NDArray[np.complex128]:
    arm_a_matrix = polarizing_beam_splitter_arm(
        "A",
        extinction_ratio_db=extinction_ratio_db,
        axis_offset_rad=axis_offset_rad,
        insertion_loss_db=insertion_loss_db,
    )
    arm_b_matrix = polarizing_beam_splitter_arm(
        "B",
        extinction_ratio_db=extinction_ratio_db,
        axis_offset_rad=axis_offset_rad,
        insertion_loss_db=insertion_loss_db,
    )
    arm_a_projection = arm_a_matrix.sum(axis=0)
    arm_b_projection = arm_b_matrix.sum(axis=0)
    field_array = field_amplitudes.astype(np.complex128, copy=False)
    arm_a = (
        arm_a_projection[0] * field_array[:, :, 0]
        + arm_a_projection[1] * field_array[:, :, 1]
    )
    arm_b = (
        arm_b_projection[0] * field_array[:, :, 0]
        + arm_b_projection[1] * field_array[:, :, 1]
    )
    return np.stack([arm_a, arm_b], axis=2)


def _mean_power_ratio(
    before: NDArray[np.complexfloating],
    after: NDArray[np.complexfloating],
) -> float:
    before_power = float(np.sum(np.abs(before) ** 2))
    if before_power <= 0.0:
        return 0.0
    return float(np.sum(np.abs(after) ** 2) / before_power)


def _add_field_component(
    field: Field,
    type_name: str,
    existing_components: object,
    instance_id: str,
    payload: dict[str, object],
) -> Field:
    component_name = unique_component_name(type_name, existing_components, instance_id)
    return field.add_component(component_name, payload)


def _add_throughput_component(
    *,
    state: SimulationState,
    type_name: str,
    instance_id: str,
    wavelength_grid_um: tuple[float, ...],
    transmission_fraction: tuple[float, ...],
    payload: dict[str, object],
) -> Throughput:
    component_name = unique_component_name(
        type_name,
        state.throughput.components,
        instance_id,
    )
    component_payload = {
        "wavelength_grid_um": tuple(float(value) for value in wavelength_grid_um),
        "transmission_fraction": tuple(float(value) for value in transmission_fraction),
        **payload,
    }
    return state.throughput.add_component(component_name, component_payload)


def _resolve_seed(seed: int | None, context_rng: np.random.Generator | None) -> int:
    if seed is not None:
        return int(seed)
    if context_rng is None:
        return 0
    return int(context_rng.integers(0, np.iinfo(np.uint32).max))


__all__ = ["HalfWavePlate", "PMFiberJones", "PolarizingBeamSplitter"]
