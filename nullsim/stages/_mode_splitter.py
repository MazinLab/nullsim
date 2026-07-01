"""Shared helpers for post-split chip input bookkeeping."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nullsim.pipeline import SimulationState


def splitter_physical_indices(
    state: SimulationState,
    n_chip_modes: int,
) -> tuple[NDArray[np.intp], str] | None:
    """Map post-split chip inputs back to pre-split physical modes."""

    for name, payload in reversed(tuple(state.geometry.components.items())):
        if not isinstance(payload, dict):
            continue
        n_input_raw = payload.get("n_input_modes")
        n_output_raw = payload.get("n_output_modes")
        copies_raw = payload.get("copies")
        if n_input_raw is None or n_output_raw is None or copies_raw is None:
            continue
        n_input = int(n_input_raw)
        n_output = int(n_output_raw)
        copies = int(copies_raw)
        if n_output != n_chip_modes:
            continue
        explicit_indices = _payload_vector(payload, "physical_mode_indices")
        if explicit_indices is not None:
            indices = np.asarray(explicit_indices, dtype=np.intp)
            if indices.shape != (n_chip_modes,):
                continue
            if np.any(indices < 0) or np.any(indices >= n_input):
                continue
            return indices, f"geometry.{name}"
        if n_input < 2 or copies < 2 or n_input * copies != n_output:
            continue
        n_apertures = int(state.geometry.n_apertures or 1)
        if n_input % n_apertures != 0:
            continue
        modes_per_aperture = n_input // n_apertures
        indices: list[int] = []
        for aperture_index in range(n_apertures):
            start = aperture_index * modes_per_aperture
            block = list(range(start, start + modes_per_aperture))
            for _copy_index in range(copies):
                indices.extend(block)
        if len(indices) != n_chip_modes:
            continue
        return np.asarray(indices, dtype=np.intp), f"geometry.{name}"
    return None


def component_mode_areas_m2(
    state: SimulationState,
    n_modes: int,
) -> tuple[NDArray[np.float64], str] | None:
    """Return per-chip-input collecting areas from geometry components."""

    return _component_vector(state, "mode_areas_m2", n_modes)


def component_mode_diameters_m(
    state: SimulationState,
    n_modes: int,
) -> tuple[NDArray[np.float64], str] | None:
    """Return per-chip-input physical diameters from geometry components."""

    return _component_vector(state, "mode_diameters_m", n_modes)


def _component_vector(
    state: SimulationState,
    key: str,
    n_modes: int,
) -> tuple[NDArray[np.float64], str] | None:
    for name, payload in reversed(tuple(state.geometry.components.items())):
        if not isinstance(payload, dict):
            continue
        vector = _payload_vector(payload, key)
        if vector is None:
            continue
        arr = np.asarray(vector, dtype=np.float64)
        if arr.shape != (n_modes,):
            continue
        if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
            continue
        return arr, f"geometry.{name}.{key}"
    return None


def _payload_vector(
    payload: dict,
    key: str,
) -> NDArray[np.float64] | None:
    value = payload.get(key)
    if value is None:
        return None
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 1:
        return None
    return arr
