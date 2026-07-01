"""Fixed-aperture (optionally multi-telescope) geometry stage.

Originally a §15.1 vertical-slice placeholder that hardcoded one
circular aperture, this stage now also supports an N-telescope array
(``n_apertures > 1``) and populates ``geometry.aperture_positions_enu_m``
so downstream stages (``delay_lines.geometric``, multi-aperture field
constructors) have the ENU geometry they need. Array layouts are still
schematic — a real Keck/Subaru/Gemini geometry would import the
KILO-style geodetic-to-ENU conversion — but the linear / Y / manual
options cover a compact KILO-style four-telescope example without
pulling KILO into nullsim.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nullsim.pipeline import (
    BaseStage,
    Geometry,
    SimulationState,
    Throughput,
    register_stage,
)
from nullsim.stages._utils import unique_component_name


@register_stage
class FixedAperture(BaseStage):
    """Populate geometry with one or more circular apertures + ENU positions."""

    type_name = "telescope.fixed_aperture"
    family = "telescope"
    consumes = frozenset({"scene"})
    produces = frozenset({"geometry", "throughput"})
    # Folds its scalar transmission into the existing ``throughput`` content,
    # which it does not declare in ``consumes``.
    cache_key_extras = frozenset({"throughput"})

    class Params(BaseModel):
        """Validated params for the (multi-)aperture telescope stage."""

        model_config = ConfigDict(
            extra="forbid",
            frozen=True,
            allow_inf_nan=False,
        )

        diameter_m: float = Field(default=1.0, gt=0.0)
        aperture_diameters_m: tuple[float, ...] | None = None
        n_apertures: int = Field(default=1, ge=1)
        array_layout: Literal["linear", "y_array", "manual"] = "linear"
        spacing_m: float = Field(default=30.0, gt=0.0)
        aperture_positions_enu_m_explicit: tuple[tuple[float, float, float], ...] | None = (
            None
        )
        mirror_transmission: float | None = Field(
            default=None,
            gt=0.0,
            le=1.0,
            description=(
                "Band-averaged mirror-train transmission folded into the "
                "throughput chain. Site-typical values: Subaru-class 8 m "
                "Nasmyth ~0.80, E-ELT 5-mirror Nasmyth ~0.85, clean space "
                "optics ~0.95. None ⇒ no mirror transmission contribution "
                "(legacy behavior)."
            ),
        )
        ao_bench_transmission: float | None = Field(
            default=None,
            gt=0.0,
            le=1.0,
            description=(
                "Band-averaged transmission of the AO bench optics between "
                "the telescope and the photonic chip (dichroics, fold "
                "mirrors, science arm). Folded into the throughput chain "
                "as a separate factor from ``mirror_transmission`` so the "
                "throughput-breakdown plot can attribute the loss. KILO "
                "convention (``kilo/transport.py:175-244``): 0.85 for "
                "ground-based AO benches; ~1.0 for clean space optics. "
                "None ⇒ no AO-bench contribution (legacy behavior)."
            ),
        )
        central_obstruction_ratio: float = Field(
            default=0.0,
            ge=0.0,
            lt=1.0,
            description=(
                "Linear central-obstruction ratio ε = d_obs / D. The "
                "geometric collecting area is multiplied by (1 − ε²): "
                "Keck ε ≈ 0.24 → 5.8% area reduction, E-ELT ε ≈ 0.28 → "
                "7.8%, ideal space ε = 0.0. Matches the convention used "
                "by ``ao.pyramid_wfs.central_obstruction`` and the "
                "Ruilier–Cassaing ``geometric_coupling_limit``."
            ),
        )

        @field_validator("aperture_positions_enu_m_explicit")
        @classmethod
        def _validate_explicit_positions(
            cls,
            value: tuple[tuple[float, float, float], ...] | None,
        ) -> tuple[tuple[float, float, float], ...] | None:
            if value is None:
                return None
            for row in value:
                if len(row) != 3:
                    raise ValueError(
                        "aperture_positions_enu_m_explicit rows must have exactly "
                        "three entries (East, North, Up) in meters."
                    )
            return value

        @field_validator("aperture_diameters_m")
        @classmethod
        def _validate_aperture_diameters(
            cls,
            value: tuple[float, ...] | None,
        ) -> tuple[float, ...] | None:
            if value is None:
                return None
            diameters = tuple(float(item) for item in value)
            if not diameters:
                raise ValueError("aperture_diameters_m must not be empty.")
            if any(not math.isfinite(item) or item <= 0.0 for item in diameters):
                raise ValueError("aperture_diameters_m entries must be positive.")
            return diameters

        @model_validator(mode="after")
        def _check_manual_positions(self) -> FixedAperture.Params:
            if (
                self.aperture_diameters_m is not None
                and len(self.aperture_diameters_m) != self.n_apertures
            ):
                raise ValueError(
                    "telescope.fixed_aperture aperture_diameters_m must have "
                    f"n_apertures={self.n_apertures} entries "
                    f"(got {len(self.aperture_diameters_m)})."
                )
            if self.array_layout == "manual":
                if self.aperture_positions_enu_m_explicit is None:
                    raise ValueError(
                        "telescope.fixed_aperture array_layout='manual' requires "
                        "aperture_positions_enu_m_explicit."
                    )
                if len(self.aperture_positions_enu_m_explicit) != self.n_apertures:
                    raise ValueError(
                        "telescope.fixed_aperture aperture_positions_enu_m_explicit "
                        f"must have n_apertures={self.n_apertures} rows "
                        f"(got {len(self.aperture_positions_enu_m_explicit)})."
                    )
            return self

    def apply(self, state: SimulationState) -> SimulationState:
        params = self.params_model
        # Per-aperture obstructed light-bucket area: π·(D/2)²·(1 − ε²).
        # Keck ε ≈ 0.24 → ~5.8% reduction; E-ELT ε ≈ 0.28 → ~7.8%; HWO
        # space ε ≈ 0 → no reduction. ``ao.pyramid_wfs`` already applies
        # this factor to its WFS photon budget, so doing it here keeps
        # the absolute stellar-rate normalization consistent across the
        # AO and detector chains.
        aperture_diameters_m = _aperture_diameters_m(params)
        unobstructed_area_m2 = math.pi * (aperture_diameters_m / 2.0) ** 2
        obstruction_factor = 1.0 - params.central_obstruction_ratio ** 2
        aperture_areas_m2 = unobstructed_area_m2 * obstruction_factor
        area_m2 = float(np.mean(aperture_areas_m2))
        # Total collecting area summed over all apertures. Downstream
        # detector stages need this (not per-aperture ``area_m2``) so the
        # absolute photon-rate normalization across an N-aperture array is
        # right; matches KILO's ``effective_area`` convention. Without
        # the AO stage's sub-aperture packing refinement, the full-pupil
        # value is also the initial ``effective_collecting_area_m2``;
        # ``ao.pyramid_wfs`` updates that field with the packed
        # sub-aperture sum when ``pupil_template`` is set.
        collecting_area_m2 = float(np.sum(aperture_areas_m2))
        reference_diameter_m = (
            float(params.diameter_m)
            if params.aperture_diameters_m is None
            else float(np.max(aperture_diameters_m))
        )

        positions_enu_m: NDArray[np.float64] | None = _aperture_positions_enu_m(params)
        positions_pupil_m: NDArray[np.float64] | None = None
        if positions_enu_m is not None:
            positions_pupil_m = positions_enu_m[:, :2].astype(np.float64, copy=True)

        component_payload: dict[str, object] = {
            "diameter_m": reference_diameter_m,
            "aperture_diameters_m": aperture_diameters_m.copy(),
            "area_m2": area_m2,
            "aperture_areas_m2": aperture_areas_m2.copy(),
            "collecting_area_m2": collecting_area_m2,
            "n_apertures": params.n_apertures,
            "array_layout": params.array_layout,
            "central_obstruction_ratio": float(params.central_obstruction_ratio),
            "mode_areas_m2": aperture_areas_m2.copy(),
            "mode_diameters_m": aperture_diameters_m.copy(),
            "mode_area_m2": (
                float(aperture_areas_m2[0])
                if np.allclose(aperture_areas_m2, aperture_areas_m2[0])
                else None
            ),
            "mode_diameter_m": (
                float(aperture_diameters_m[0])
                if np.allclose(aperture_diameters_m, aperture_diameters_m[0])
                else None
            ),
        }
        if positions_enu_m is not None:
            component_payload["aperture_positions_enu_m"] = positions_enu_m.copy()
            component_payload["aperture_positions_pupil_m"] = positions_pupil_m.copy()
            component_payload["spacing_m"] = params.spacing_m

        geometry = Geometry(
            diameter_m=reference_diameter_m,
            area_m2=area_m2,
            collecting_area_m2=collecting_area_m2,
            effective_collecting_area_m2=collecting_area_m2,
            n_apertures=params.n_apertures,
            aperture_diameters_m=aperture_diameters_m.copy(),
            aperture_areas_m2=aperture_areas_m2.copy(),
            aperture_positions_enu_m=positions_enu_m,
            aperture_positions_pupil_m=positions_pupil_m,
        ).add_component("fixed_aperture", component_payload)

        throughput = _maybe_fold_mirror_transmission(
            state=state,
            mirror_transmission=params.mirror_transmission,
            ao_bench_transmission=params.ao_bench_transmission,
            instance_id=self.context.instance_id,
        )

        return replace(state, geometry=geometry, throughput=throughput)


def _aperture_positions_enu_m(
    params: FixedAperture.Params,
) -> NDArray[np.float64] | None:
    """Build (n_apertures, 3) ENU positions for the requested array layout.

    Returns ``None`` for the single-aperture case to preserve the original
    ``n_apertures=1`` semantics (geometry.aperture_positions_enu_m stays
    unset; downstream stages that don't need positions are unaffected).
    """

    if params.n_apertures == 1:
        return None

    if params.array_layout == "manual":
        # Validator already ensured shape and row count match n_apertures.
        assert params.aperture_positions_enu_m_explicit is not None
        return np.asarray(
            params.aperture_positions_enu_m_explicit, dtype=np.float64
        ).reshape(params.n_apertures, 3)

    if params.array_layout == "linear":
        return _linear_positions(params.n_apertures, params.spacing_m)

    if params.array_layout == "y_array":
        return _y_array_positions(params.n_apertures, params.spacing_m)

    raise ValueError(f"Unknown array_layout {params.array_layout!r}.")


def _aperture_diameters_m(params: FixedAperture.Params) -> NDArray[np.float64]:
    if params.aperture_diameters_m is None:
        return np.full(params.n_apertures, float(params.diameter_m), dtype=np.float64)
    return np.asarray(params.aperture_diameters_m, dtype=np.float64)


def _linear_positions(n_apertures: int, spacing_m: float) -> NDArray[np.float64]:
    """Equally spaced East-West array, centered on the ENU origin."""

    east_offsets = (np.arange(n_apertures, dtype=np.float64) - (n_apertures - 1) / 2.0) * spacing_m
    positions = np.zeros((n_apertures, 3), dtype=np.float64)
    positions[:, 0] = east_offsets
    return positions


def _y_array_positions(n_apertures: int, spacing_m: float) -> NDArray[np.float64]:
    """KILO Maunakea-style Y array (schematic): three arms at 0°, 120°, 240°.

    The first aperture sits at the array centroid; the remaining N-1 are
    distributed across the three arms in round-robin order. Distances along
    each arm increase in ``spacing_m`` steps. This is a stand-in for a real
    geodetic-to-ENU conversion — it preserves the right *order of magnitude*
    of baselines (a few × spacing_m) and the right *count* of independent
    baselines (n_apertures-choose-2) for geometry sanity checks.
    """

    positions = np.zeros((n_apertures, 3), dtype=np.float64)
    arm_angles_rad = np.array([0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0])
    arm_step = np.zeros(3, dtype=np.int64)  # step index along each arm
    for index in range(1, n_apertures):
        arm = (index - 1) % 3
        arm_step[arm] += 1
        radius = spacing_m * float(arm_step[arm])
        positions[index, 0] = radius * math.cos(arm_angles_rad[arm])
        positions[index, 1] = radius * math.sin(arm_angles_rad[arm])
    # Recenter on centroid so geometry is translation-invariant.
    positions -= positions.mean(axis=0, keepdims=True)
    return positions


def _float_tuple(values: NDArray[np.float64]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _maybe_fold_mirror_transmission(
    *,
    state: SimulationState,
    mirror_transmission: float | None,
    ao_bench_transmission: float | None,
    instance_id: str,
) -> Throughput:
    """Multiply the running throughput by band-averaged telescope-side factors.

    Applies the mirror-train and AO-bench transmissions as separate
    throughput components so the downstream throughput-breakdown plot
    attributes each loss correctly. A None factor is a no-op. Matches
    the KILO ``transport.py:175-244`` convention of accumulating
    ``Al^n_mirrors`` and ``ao_bench`` as distinct multiplicative
    factors rather than rolling them into a single scalar.
    """

    throughput = state.throughput
    if mirror_transmission is not None:
        throughput = _fold_scalar_transmission(
            throughput=throughput,
            transmission=float(mirror_transmission),
            component_root="telescope.mirror_transmission",
            payload_key="mirror_transmission",
            source_model=(
                "Band-averaged mirror-train transmission "
                "(telescope-typical scalar across all reflective surfaces)."
            ),
            instance_id=instance_id,
            error_prefix="telescope.fixed_aperture: mirror_transmission",
        )
    if ao_bench_transmission is not None:
        throughput = _fold_scalar_transmission(
            throughput=throughput,
            transmission=float(ao_bench_transmission),
            component_root="telescope.ao_bench_transmission",
            payload_key="ao_bench_transmission",
            source_model=(
                "Band-averaged AO-bench transmission "
                "(dichroics, fold mirrors, science arm between "
                "telescope and chip; KILO ao_bench default 0.85)."
            ),
            instance_id=instance_id,
            error_prefix="telescope.fixed_aperture: ao_bench_transmission",
        )
    return throughput


def _fold_scalar_transmission(
    *,
    throughput: Throughput,
    transmission: float,
    component_root: str,
    payload_key: str,
    source_model: str,
    instance_id: str,
    error_prefix: str,
) -> Throughput:
    wavelength_grid_um = throughput.wavelength_grid_um
    if not wavelength_grid_um:
        raise ValueError(
            f"{error_prefix} requires state.throughput.wavelength_grid_um "
            "to be populated."
        )
    n_bins = len(wavelength_grid_um)
    per_channel = np.full(n_bins, transmission, dtype=np.float64)
    current = (
        np.asarray(throughput.transmission_fraction, dtype=np.float64)
        if throughput.transmission_fraction
        else np.ones(n_bins, dtype=np.float64)
    )
    if current.size != n_bins:
        raise ValueError(
            f"{error_prefix}: throughput.transmission_fraction length "
            f"{current.size} does not match wavelength grid length {n_bins}."
        )
    updated = current * per_channel

    component_name = unique_component_name(
        component_root,
        throughput.components,
        instance_id,
    )
    return Throughput(
        wavelength_grid_um=wavelength_grid_um,
        transmission_fraction=_float_tuple(updated),
        components={
            **throughput.components,
            component_name: {
                "wavelength_grid_um": wavelength_grid_um,
                "transmission_fraction": _float_tuple(per_channel),
                payload_key: transmission,
                "source_model": source_model,
            },
        },
    )


__all__ = ["FixedAperture"]
