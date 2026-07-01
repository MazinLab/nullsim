"""Mode fanout for duplicated-input Clements mesh studies."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field as PydanticField, field_validator

from nullsim.pipeline import BaseStage, Field, Geometry, SimulationState, register_stage
from nullsim.stages._utils import mode_origin_positions_m


@register_stage
class ModeSplitter(BaseStage):
    """Duplicate selected physical guided modes into split-area mesh inputs."""

    type_name = "injection.mode_splitter"
    family = "injection"
    consumes = frozenset({"field", "geometry"})
    produces = frozenset({"field", "geometry"})

    class Params(BaseModel):
        """Validated params for a fixed guided-mode fanout."""

        model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

        copies: int = PydanticField(default=2, ge=2, le=4)
        split_mode_indices: tuple[int, ...] = ()

        @field_validator("split_mode_indices")
        @classmethod
        def _validate_split_mode_indices(
            cls,
            value: tuple[int, ...],
        ) -> tuple[int, ...]:
            indices = tuple(int(index) for index in value)
            if any(index < 0 for index in indices):
                raise ValueError("split_mode_indices must be non-negative.")
            if len(set(indices)) != len(indices):
                raise ValueError("split_mode_indices entries must be unique.")
            return indices

    def apply(self, state: SimulationState) -> SimulationState:
        params = self.params_model
        amplitudes = state.field.amplitudes
        if amplitudes is None:
            raise ValueError("injection.mode_splitter requires field amplitudes.")
        field_array = np.asarray(amplitudes, dtype=np.complex128)
        if field_array.ndim != 3:
            raise ValueError(
                "injection.mode_splitter requires field amplitudes with shape "
                "[n_modes, n_wavelength_bins, n_pol]."
            )
        n_input_modes = int(field_array.shape[0])
        n_apertures = int(state.geometry.n_apertures or 1)
        if n_input_modes % n_apertures != 0:
            raise ValueError(
                "injection.mode_splitter requires the input mode count to be a "
                "multiple of geometry.n_apertures."
            )

        split_mode_indices = _resolve_split_mode_indices(
            params.split_mode_indices,
            n_input_modes=n_input_modes,
        )
        split_amplitudes, physical_indices, copy_counts = _split_by_aperture_blocks(
            field_array,
            n_apertures=n_apertures,
            copies=int(params.copies),
            split_mode_indices=split_mode_indices,
        )
        split_positions = _split_positions(
            state.geometry.aperture_positions_pupil_m,
            n_input_modes=n_input_modes,
            n_apertures=n_apertures,
            copies=int(params.copies),
            split_mode_indices=split_mode_indices,
        )
        (
            mode_area_m2,
            mode_areas_m2,
            mode_diameter_m,
            mode_diameters_m,
            field_area_reference_m2,
        ) = (
            _split_mode_area_and_diameter(
                state.geometry,
                n_input_modes=n_input_modes,
                n_output_modes=int(split_amplitudes.shape[0]),
                copy_counts=copy_counts,
                physical_mode_indices=physical_indices,
            )
        )
        split_amplitudes = _apply_relative_area_scaling(
            split_amplitudes,
            mode_areas_m2=mode_areas_m2,
            field_area_reference_m2=field_area_reference_m2,
        )

        # Expand the per-mode telescope origins (when an upstream packer
        # published them) with the same row mapping as the amplitudes so
        # off-zenith projection stays correct downstream (review 2026-06-10
        # Issue 20).
        input_origins = mode_origin_positions_m(state.geometry, n_input_modes)
        split_origins = (
            input_origins[np.asarray(physical_indices, dtype=np.intp)]
            if input_origins is not None
            else None
        )

        components = dict(state.geometry.components)
        components[self.context.instance_id or "mode_splitter"] = {
            "n_input_modes": n_input_modes,
            "n_output_modes": int(split_amplitudes.shape[0]),
            "copies": int(params.copies),
            "split_mode_indices": np.asarray(
                sorted(split_mode_indices or ()), dtype=np.int64,
            ),
            "physical_mode_indices": physical_indices.astype(np.int64, copy=True),
            "mode_origin_positions_m": (
                split_origins.astype(np.float64, copy=True)
                if split_origins is not None
                else None
            ),
            "mode_area_m2": float(mode_area_m2) if mode_area_m2 is not None else None,
            "mode_areas_m2": (
                mode_areas_m2.astype(np.float64, copy=True)
                if mode_areas_m2 is not None
                else None
            ),
            "mode_diameter_m": (
                float(mode_diameter_m) if mode_diameter_m is not None else None
            ),
            "mode_diameters_m": (
                mode_diameters_m.astype(np.float64, copy=True)
                if mode_diameters_m is not None
                else None
            ),
            "field_area_reference_m2": (
                float(field_area_reference_m2)
                if field_area_reference_m2 is not None
                else None
            ),
            "source_model": (
                "Guided-mode fanout before the Clements mesh. Split field "
                "amplitudes are scaled by sqrt(mode_area / field_area_reference) "
                "so downstream unitary mixing sees the correct unequal branch "
                "weights. mode_areas_m2 carries the physical branch area for "
                "planet-throughput and optimizer diagnostics; detectors should "
                "use field_area_reference_m2 as the scalar output-port area."
            ),
        }

        labels = _split_mode_labels(
            state.field.mode_labels,
            n_input_modes=n_input_modes,
            n_apertures=n_apertures,
            copies=int(params.copies),
            split_mode_indices=split_mode_indices,
        )
        field = Field(
            amplitudes=split_amplitudes,
            wavelength_grid_um=state.field.wavelength_grid_um,
            mode_labels=labels,
            components=dict(state.field.components),
        )
        geometry = replace(
            state.geometry,
            aperture_positions_pupil_m=split_positions,
            components=components,
        )
        return replace(state, field=field, geometry=geometry)


def _split_by_aperture_blocks(
    field_array: NDArray[np.complex128],
    *,
    n_apertures: int,
    copies: int,
    split_mode_indices: frozenset[int] | None,
) -> tuple[NDArray[np.complex128], NDArray[np.intp], NDArray[np.int64]]:
    modes_per_aperture = field_array.shape[0] // n_apertures
    blocks: list[NDArray[np.complex128]] = []
    physical_indices: list[int] = []
    copy_counts: list[int] = []
    for aperture_index in range(n_apertures):
        start = aperture_index * modes_per_aperture
        stop = start + modes_per_aperture
        if split_mode_indices is None:
            block = field_array[start:stop]
            blocks.extend(block.copy() for _copy_index in range(copies))
            for _copy_index in range(copies):
                physical_indices.extend(range(start, stop))
            copy_counts.extend([copies] * modes_per_aperture)
            continue
        for mode_index in range(start, stop):
            n_copies = copies if mode_index in split_mode_indices else 1
            copy_counts.append(n_copies)
            for _copy_index in range(n_copies):
                blocks.append(field_array[mode_index : mode_index + 1].copy())
                physical_indices.append(mode_index)
    return (
        np.concatenate(blocks, axis=0).astype(np.complex128, copy=False),
        np.asarray(physical_indices, dtype=np.intp),
        np.asarray(copy_counts, dtype=np.int64),
    )


def _apply_relative_area_scaling(
    split_amplitudes: NDArray[np.complex128],
    *,
    mode_areas_m2: NDArray[np.float64] | None,
    field_area_reference_m2: float | None,
) -> NDArray[np.complex128]:
    """Scale copied branches by sqrt(area_j / reference_area).

    After this scaling, a detector can use the scalar reference area
    ``field_area_reference_m2`` for every post-chip output port. The input
    area heterogeneity has already been folded into the coherently mixed
    field amplitudes.
    """

    if mode_areas_m2 is None or field_area_reference_m2 is None:
        return split_amplitudes
    mode_areas = np.asarray(mode_areas_m2, dtype=np.float64)
    if mode_areas.shape != (split_amplitudes.shape[0],):
        raise ValueError(
            "injection.mode_splitter internal mode-area shape mismatch "
            f"({mode_areas.shape} != ({split_amplitudes.shape[0]},))."
        )
    reference = float(field_area_reference_m2)
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("injection.mode_splitter field_area_reference_m2 is invalid.")
    if not np.all(np.isfinite(mode_areas)) or np.any(mode_areas <= 0.0):
        raise ValueError("injection.mode_splitter mode_areas_m2 must be positive.")
    scale = np.sqrt(mode_areas / reference)
    return split_amplitudes * scale[:, np.newaxis, np.newaxis]


def _split_positions(
    positions_pupil_m: object,
    *,
    n_input_modes: int,
    n_apertures: int,
    copies: int,
    split_mode_indices: frozenset[int] | None,
) -> NDArray[np.float64] | None:
    if positions_pupil_m is None:
        return None
    positions = np.asarray(positions_pupil_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[0] != n_input_modes:
        raise ValueError(
            "injection.mode_splitter requires geometry.aperture_positions_pupil_m "
            f"with {n_input_modes} rows when positions are present."
        )
    modes_per_aperture = n_input_modes // n_apertures
    blocks: list[NDArray[np.float64]] = []
    for aperture_index in range(n_apertures):
        start = aperture_index * modes_per_aperture
        stop = start + modes_per_aperture
        if split_mode_indices is None:
            block = positions[start:stop]
            blocks.extend(block.copy() for _copy_index in range(copies))
            continue
        for mode_index in range(start, stop):
            n_copies = copies if mode_index in split_mode_indices else 1
            blocks.extend(
                positions[mode_index : mode_index + 1].copy()
                for _copy_index in range(n_copies)
            )
    return np.concatenate(blocks, axis=0).astype(np.float64, copy=False)


def _split_mode_area_and_diameter(
    geometry: Geometry,
    *,
    n_input_modes: int,
    n_output_modes: int,
    copy_counts: NDArray[np.int64],
    physical_mode_indices: NDArray[np.intp],
) -> tuple[
    float | None,
    NDArray[np.float64] | None,
    float | None,
    NDArray[np.float64] | None,
    float | None,
]:
    input_mode_areas, input_mode_diameters = _input_mode_area_and_diameter(
        geometry,
        n_input_modes=n_input_modes,
    )
    if input_mode_areas is None:
        return None, None, None, None, None
    total_area = float(np.sum(input_mode_areas))
    if not np.isfinite(total_area) or total_area <= 0.0:
        return None, None, None, None, None
    field_area_reference = total_area / float(n_output_modes)
    counts = np.asarray(copy_counts, dtype=np.float64)
    if counts.shape != (n_input_modes,):
        raise ValueError(
            "injection.mode_splitter internal copy-count shape mismatch "
            f"({counts.shape} != ({n_input_modes},))."
        )
    # Index by the authoritative per-row physical mode so the area/diameter
    # vectors share the amplitude rows' ordering on BOTH split paths. The
    # previous ``np.repeat(input/counts, counts)`` produced mode-interleaved
    # order while the default full-split path emits block-copy order —
    # misaligning areas vs amplitudes for heterogeneous sub-apertures
    # (review 2026-06-10 Issue 12).
    row_physical = np.asarray(physical_mode_indices, dtype=np.intp)
    if row_physical.shape != (n_output_modes,):
        raise ValueError(
            "injection.mode_splitter internal physical-index shape mismatch "
            f"({row_physical.shape} != ({n_output_modes},))."
        )
    mode_areas = input_mode_areas[row_physical] / counts[row_physical]
    mode_diameters = input_mode_diameters[row_physical]
    mode_area_scalar = (
        float(mode_areas[0])
        if mode_areas.size and np.allclose(mode_areas, mode_areas[0])
        else None
    )
    mode_diameter_scalar = (
        float(mode_diameters[0])
        if mode_diameters.size and np.allclose(mode_diameters, mode_diameters[0])
        else None
    )
    return (
        mode_area_scalar,
        mode_areas,
        mode_diameter_scalar,
        mode_diameters,
        field_area_reference,
    )


def _input_mode_area_and_diameter(
    geometry: Geometry,
    *,
    n_input_modes: int,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
    vector_area = _component_vector(geometry, "mode_areas_m2", n_input_modes)
    vector_diameter = _component_vector(geometry, "mode_diameters_m", n_input_modes)
    if vector_area is not None:
        diameters = (
            vector_diameter
            if vector_diameter is not None
            else np.sqrt(4.0 * vector_area / np.pi)
        )
        return vector_area, diameters

    if geometry.aperture_areas_m2 is not None:
        areas = _validated_vector(
            geometry.aperture_areas_m2,
            n_input_modes,
            "geometry.aperture_areas_m2",
        )
        if areas is not None:
            diameters = _validated_vector(
                geometry.aperture_diameters_m,
                n_input_modes,
                "geometry.aperture_diameters_m",
            )
            if diameters is None:
                diameters = np.sqrt(4.0 * areas / np.pi)
            return areas, diameters

    scalar_area = _component_scalar(geometry, "mode_area_m2")
    scalar_diameter = _component_scalar(geometry, "mode_diameter_m")
    if scalar_area is not None:
        areas = np.full(n_input_modes, scalar_area, dtype=np.float64)
        diameter = (
            scalar_diameter
            if scalar_diameter is not None
            else float(np.sqrt(4.0 * scalar_area / np.pi))
        )
        return areas, np.full(n_input_modes, diameter, dtype=np.float64)

    effective_area = geometry.effective_collecting_area_m2
    if effective_area is None:
        effective_area = geometry.collecting_area_m2
    if effective_area is None:
        return None, None
    total_area = float(effective_area)
    if not np.isfinite(total_area) or total_area <= 0.0:
        return None, None
    input_mode_area = total_area / float(n_input_modes)
    input_mode_diameter = float(np.sqrt(4.0 * input_mode_area / np.pi))
    return (
        np.full(n_input_modes, input_mode_area, dtype=np.float64),
        np.full(n_input_modes, input_mode_diameter, dtype=np.float64),
    )


def _component_vector(
    geometry: Geometry,
    key: str,
    n_modes: int,
) -> NDArray[np.float64] | None:
    for payload in reversed(tuple(geometry.components.values())):
        if not isinstance(payload, dict):
            continue
        arr = _validated_vector(payload.get(key), n_modes, f"geometry.{key}")
        if arr is not None:
            return arr
    return None


def _validated_vector(
    value: object,
    n_modes: int,
    name: str,
) -> NDArray[np.float64] | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (n_modes,):
        return None
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"injection.mode_splitter {name} must be positive.")
    return arr.astype(np.float64, copy=True)


def _component_scalar(geometry: Geometry, key: str) -> float | None:
    for payload in reversed(tuple(geometry.components.values())):
        if not isinstance(payload, dict):
            continue
        value = payload.get(key)
        if value is None:
            continue
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(scalar) and scalar > 0.0:
            return scalar
    return None


def _split_mode_labels(
    labels: tuple[str, ...],
    *,
    n_input_modes: int,
    n_apertures: int,
    copies: int,
    split_mode_indices: frozenset[int] | None,
) -> tuple[str, ...]:
    if len(labels) != n_input_modes:
        labels = tuple(f"mode_{index}" for index in range(n_input_modes))
    modes_per_aperture = n_input_modes // n_apertures
    out: list[str] = []
    for aperture_index in range(n_apertures):
        start = aperture_index * modes_per_aperture
        stop = start + modes_per_aperture
        block = labels[start:stop]
        if split_mode_indices is None:
            for copy_index in range(copies):
                out.extend(f"{label}_split{copy_index}" for label in block)
            continue
        for offset, label in enumerate(block):
            mode_index = start + offset
            n_copies = copies if mode_index in split_mode_indices else 1
            if n_copies == 1:
                out.append(label)
            else:
                out.extend(
                    f"{label}_split{copy_index}" for copy_index in range(n_copies)
                )
    return tuple(out)


def _resolve_split_mode_indices(
    indices: tuple[int, ...],
    *,
    n_input_modes: int,
) -> frozenset[int] | None:
    if not indices:
        return None
    selected = frozenset(int(index) for index in indices)
    max_index = max(selected)
    if max_index >= n_input_modes:
        raise ValueError(
            "injection.mode_splitter split_mode_indices contains index "
            f"{max_index}, but the input has only {n_input_modes} modes."
        )
    return selected


__all__ = ["ModeSplitter"]
