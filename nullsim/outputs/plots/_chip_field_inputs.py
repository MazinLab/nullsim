"""Shared chip-field inputs for plot outputs.

This module centralizes the low-level component retrieval needed by plot
outputs that evaluate sky-dependent chip response. It deliberately stays below
the higher-level imaging-specific assembly in ``imaging_reconstruction``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from nullsim.physics.imaging import SubAperture
from nullsim.pipeline import SimulationState
from nullsim.stages._mode_splitter import (
    component_mode_areas_m2,
    component_mode_diameters_m,
)


@dataclass(frozen=True)
class ChipFieldInputs:
    """Inputs needed to evaluate chip response over sky coordinates."""

    u_stack: NDArray[np.complex128]
    wavelengths_m: NDArray[np.float64]
    wl_design_m: float
    positions_pupil_m: NDArray[np.float64]
    positions_enu_m: NDArray[np.float64]
    coupling_per_sub: NDArray[np.float64]
    sub_areas_m2: NDArray[np.float64]
    sub_diameters_m: NDArray[np.float64]
    dark_indices: NDArray[np.intp]
    unitary_arm_weights: NDArray[np.float64]


def build_chip_field_inputs(
    state: SimulationState,
    *,
    require_sensitivity: bool = False,
    error_prefix: str = "chip-field plot",
) -> ChipFieldInputs:
    """Return validated chip-field inputs for sky-response plots."""

    chip_payload = find_component(
        state,
        "chip.kernel_mzi_mesh",
        error_prefix=error_prefix,
    )
    u_stack = np.asarray(chip_payload.get("u_stack"), dtype=np.complex128)
    if u_stack.ndim != 3:
        raise ValueError(
            f"{error_prefix} requires chip.kernel_mzi_mesh to publish u_stack "
            f"with shape (n_ports, n_modes, n_wavelength); got {u_stack.shape}."
        )
    n_modes = int(u_stack.shape[1])
    n_ports = int(u_stack.shape[0])
    wavelengths_um = wavelength_grid_um(state, chip_payload, u_stack.shape[2])
    sensitivity = maybe_find_component(state, "sensitivity.detection_curve")
    if require_sensitivity and sensitivity is None:
        raise ValueError(
            f"{error_prefix} requires an upstream sensitivity.detection_curve "
            "component with chip-field normalization inputs."
        )
    base_dark_indices = dark_indices(chip_payload, sensitivity, n_ports)
    effective_stack, effective_dark_indices, arm_weights = effective_unitary_stack(
        state=state,
        chip_payload=chip_payload,
        u_stack=u_stack,
        base_dark_indices=base_dark_indices,
    )
    positions_pupil = mode_positions_pupil_m(state, n_modes)
    positions_enu = mode_positions_enu_m(state, n_modes)
    coupling, areas = coupling_and_areas(state, n_modes, sensitivity)
    return ChipFieldInputs(
        u_stack=effective_stack,
        wavelengths_m=np.asarray(wavelengths_um, dtype=np.float64) * 1.0e-6,
        wl_design_m=design_wavelength_um(chip_payload) * 1.0e-6,
        positions_pupil_m=positions_pupil,
        positions_enu_m=positions_enu,
        coupling_per_sub=coupling,
        sub_areas_m2=areas,
        sub_diameters_m=mode_diameters_m(state, n_modes, sensitivity),
        dark_indices=effective_dark_indices,
        unitary_arm_weights=arm_weights,
    )


def find_component(
    state: SimulationState,
    prefix: str,
    *,
    error_prefix: str = "chip-field plot",
) -> dict[str, Any]:
    """Return the first dict results component whose name starts with ``prefix``."""

    component = maybe_find_component(state, prefix)
    if component is None:
        raise ValueError(
            f"{error_prefix} requires a results component starting with {prefix!r}."
        )
    return component


def maybe_find_component(
    state: SimulationState,
    prefix: str,
) -> dict[str, Any] | None:
    """Return the newest dict results component whose name starts with ``prefix``."""

    for name, value in reversed(state.results.components.items()):
        if name.startswith(prefix) and isinstance(value, dict):
            return value
    return None


def wavelength_grid_um(
    state: SimulationState,
    chip_payload: dict[str, Any],
    n_wavelengths: int,
) -> tuple[float, ...]:
    """Resolve the wavelength grid attached to a chip stack."""

    raw = (
        chip_payload.get("wavelength_grid_um")
        or state.field.wavelength_grid_um
        or state.throughput.wavelength_grid_um
    )
    if raw is None:
        raise ValueError("chip-field plot requires a wavelength grid.")
    values = tuple(float(value) for value in raw)
    if len(values) != n_wavelengths:
        raise ValueError(
            "chip-field plot wavelength grid length does not match u_stack "
            f"wavelength axis ({len(values)} != {n_wavelengths})."
        )
    return values


def design_wavelength_um(chip_payload: dict[str, Any]) -> float:
    """Return the chip design wavelength in microns."""

    raw = chip_payload.get("wl_design_um")
    if raw is None:
        raise ValueError("chip.kernel_mzi_mesh payload is missing wl_design_um.")
    return float(raw)


def mode_positions_pupil_m(
    state: SimulationState,
    n_modes: int,
) -> NDArray[np.float64]:
    """Resolve 2-D pupil-plane mode positions in meters."""

    positions = state.geometry.aperture_positions_pupil_m
    if positions is None:
        positions = state.geometry.aperture_positions_enu_m
    if positions is None:
        raise ValueError(
            "chip-field plot requires geometry.aperture_positions_pupil_m "
            "or aperture_positions_enu_m."
        )
    arr = np.asarray(positions, dtype=np.float64)
    if arr.shape == (n_modes, 2):
        return arr.astype(np.float64, copy=True)
    if arr.shape == (n_modes, 3):
        return arr[:, :2].astype(np.float64, copy=True)
    raise ValueError(
        "chip-field plot requires per-mode positions matching the chip mode "
        f"count M={n_modes}; got {arr.shape}."
    )


def mode_positions_enu_m(
    state: SimulationState,
    n_modes: int,
) -> NDArray[np.float64]:
    """Resolve 3-D ENU mode positions in meters."""

    positions_enu = state.geometry.aperture_positions_enu_m
    positions_pupil = state.geometry.aperture_positions_pupil_m
    for positions in (positions_enu, positions_pupil):
        if positions is None:
            continue
        arr = np.asarray(positions, dtype=np.float64)
        if arr.shape == (n_modes, 2):
            out = np.zeros((n_modes, 3), dtype=np.float64)
            out[:, :2] = arr
            return out
        if arr.shape == (n_modes, 3):
            return arr.astype(np.float64, copy=True)
    if positions_enu is None and positions_pupil is None:
        raise ValueError(
            "chip-field plot requires geometry.aperture_positions_pupil_m "
            "or aperture_positions_enu_m."
        )
    shapes = [
        tuple(np.asarray(positions, dtype=np.float64).shape)
        for positions in (positions_enu, positions_pupil)
        if positions is not None
    ]
    raise ValueError(
        "chip-field plot requires per-mode positions matching the chip mode "
        f"count M={n_modes}; got {shapes}."
    )


def mode_diameters_m(
    state: SimulationState,
    n_modes: int,
    sensitivity: dict[str, Any] | None,
) -> NDArray[np.float64]:
    """Resolve one equivalent input-mode diameter per chip mode."""

    if sensitivity is not None and "sub_diameters_m" in sensitivity:
        diameters = np.asarray(sensitivity["sub_diameters_m"], dtype=np.float64)
        if diameters.shape == (n_modes,):
            return diameters.astype(np.float64, copy=True)
    vector = component_mode_diameters_m(state, n_modes)
    if vector is not None:
        return vector[0].copy()
    # Newest first: the oldest component is the telescope primary, which a
    # later pupil_modes or AO stage subdivides (review 2026-11-07 Issue 1).
    for payload in reversed(tuple(state.geometry.components.values())):
        if not isinstance(payload, dict):
            continue
        if payload.get("mode_diameter_m") is not None:
            return np.full(n_modes, float(payload["mode_diameter_m"]))
    areas = mode_areas_m2(state, n_modes, sensitivity)
    return np.sqrt(4.0 * areas / np.pi)


def stellar_amplitudes(
    state: SimulationState,
    n_modes: int,
    sensitivity: dict[str, Any] | None,
) -> NDArray[np.complex128]:
    """Return on-axis stellar amplitudes per input mode."""

    coupling, areas = coupling_and_areas(state, n_modes, sensitivity)
    return np.sqrt(np.maximum(coupling * areas, 0.0)).astype(np.complex128)


def coupling_and_areas(
    state: SimulationState,
    n_modes: int,
    sensitivity: dict[str, Any] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return per-mode coupling and collecting area arrays."""

    if (
        sensitivity is not None
        and "coupling_per_sub" in sensitivity
        and "sub_areas_m2" in sensitivity
    ):
        coupling = np.asarray(sensitivity["coupling_per_sub"], dtype=np.float64)
        areas = np.asarray(sensitivity["sub_areas_m2"], dtype=np.float64)
        if coupling.shape == (n_modes,) and areas.shape == (n_modes,):
            return coupling.astype(np.float64, copy=True), areas.astype(
                np.float64,
                copy=True,
            )
    return np.ones(n_modes, dtype=np.float64), mode_areas_m2(
        state,
        n_modes,
        sensitivity,
    )


def mode_areas_m2(
    state: SimulationState,
    n_modes: int,
    sensitivity: dict[str, Any] | None,
) -> NDArray[np.float64]:
    """Resolve one collecting area per input mode."""

    if sensitivity is not None and "sub_areas_m2" in sensitivity:
        areas = np.asarray(sensitivity["sub_areas_m2"], dtype=np.float64)
        if areas.shape == (n_modes,) and np.all(areas > 0.0):
            return areas.astype(np.float64, copy=True)
    vector = component_mode_areas_m2(state, n_modes)
    if vector is not None:
        return vector[0].copy()
    total = (
        state.geometry.effective_collecting_area_m2
        if state.geometry.effective_collecting_area_m2 is not None
        else state.geometry.collecting_area_m2
    )
    if total is None:
        if state.geometry.area_m2 is None:
            raise ValueError(
                "chip-field plot requires effective_collecting_area_m2, "
                "collecting_area_m2, or area_m2 to infer mode areas."
            )
        total = float(state.geometry.area_m2) * float(state.geometry.n_apertures or 1)
    if total <= 0.0:
        raise ValueError("mode collecting area must be positive.")
    return np.full(n_modes, float(total) / float(n_modes), dtype=np.float64)


def dark_indices(
    chip_payload: dict[str, Any],
    sensitivity: dict[str, Any] | None,
    n_ports: int,
) -> NDArray[np.intp]:
    """Return chip dark-port indices."""

    if sensitivity is not None and "dark_indices" in sensitivity:
        indices = np.asarray(sensitivity["dark_indices"], dtype=np.intp)
        if indices.ndim == 1 and indices.size > 0:
            if np.any((indices < 0) | (indices >= n_ports)):
                raise ValueError("sensitivity dark_indices are out of range.")
            return indices.astype(np.intp, copy=True)
    n_bright = int(chip_payload.get("n_bright_ports", 0))
    n_dark = int(chip_payload.get("n_dark_ports", 0))
    if n_bright < 0 or n_dark <= 0 or n_bright + n_dark > n_ports:
        raise ValueError("chip payload does not define a valid dark-port block.")
    return np.arange(n_bright, n_bright + n_dark, dtype=np.intp)


def effective_unitary_stack(
    *,
    state: SimulationState,
    chip_payload: dict[str, Any],
    u_stack: NDArray[np.complex128],
    base_dark_indices: NDArray[np.intp],
) -> tuple[NDArray[np.complex128], NDArray[np.intp], NDArray[np.float64]]:
    """Return the stack and dark rows used by static sky-response plots."""

    deployment_mode = str(chip_payload.get("deployment_mode", "shared"))
    if deployment_mode != "per_pol_arm":
        return (
            u_stack.astype(np.complex128, copy=True),
            base_dark_indices.astype(np.intp, copy=True),
            np.ones(1, dtype=np.float64),
        )

    stacks = per_arm_geometry_unitary_stacks(chip_payload)
    weights = unitary_arm_weights(state, stacks.shape[0])
    unitary_stack = np.stack(
        [
            weighted_dark_port_matrix(stacks[:, :, :, idx], base_dark_indices, weights)
            for idx in range(stacks.shape[3])
        ],
        axis=2,
    )
    dark = np.arange(unitary_stack.shape[0], dtype=np.intp)
    return unitary_stack, dark, weights


def unitary_matrices_for_imaging(
    *,
    state: SimulationState,
    chip_payload: dict[str, Any],
    u_stack: NDArray[np.complex128],
    base_dark_indices: NDArray[np.intp],
    wavelength_indices: tuple[int, ...],
    design_index: int,
    wavelengths_um: tuple[float, ...],
) -> tuple[
    NDArray[np.complex128],
    tuple[tuple[NDArray[np.complex128], float], ...],
    NDArray[np.intp],
]:
    """Return the design matrix and wavelength stack for imaging outputs."""

    effective_stack, dark, _weights = effective_unitary_stack(
        state=state,
        chip_payload=chip_payload,
        u_stack=u_stack,
        base_dark_indices=base_dark_indices,
    )
    return (
        effective_stack[:, :, design_index],
        tuple(
            (effective_stack[:, :, idx], float(wavelengths_um[idx]) * 1.0e-6)
            for idx in wavelength_indices
        ),
        dark,
    )


def per_arm_geometry_unitary_stacks(
    chip_payload: dict[str, Any],
) -> NDArray[np.complex128]:
    """Return per-arm stacks in the ideal geometry input phase frame."""

    raw = chip_payload.get("u_stack_per_arm")
    if raw is None:
        raise ValueError(
            "chip deployment_mode='per_pol_arm' requires chip.kernel_mzi_mesh "
            "to publish u_stack_per_arm."
        )
    deployed = np.asarray(raw, dtype=np.complex128)
    if deployed.ndim != 4:
        raise ValueError(
            "u_stack_per_arm must have shape (n_arm, n_modes, n_modes, n_wl); "
            f"got {deployed.shape}."
        )
    correction_raw = chip_payload.get("input_phase_correction")
    if correction_raw is None:
        return deployed
    correction = np.asarray(correction_raw, dtype=np.complex128)
    expected = (deployed.shape[1], deployed.shape[0])
    if correction.shape != expected:
        raise ValueError(
            "input_phase_correction shape does not match u_stack_per_arm "
            f"({correction.shape} != {expected})."
        )
    inverse_static_phase = np.conj(correction).T[:, np.newaxis, :, np.newaxis]
    return deployed * inverse_static_phase


def unitary_arm_weights(
    state: SimulationState,
    n_arms: int,
) -> NDArray[np.float64]:
    """Return field-power weights for deployed chip arms."""

    equal = np.full(n_arms, 1.0 / float(n_arms), dtype=np.float64)
    amplitudes = state.field.amplitudes
    if amplitudes is None:
        return equal
    field = np.asarray(amplitudes)
    if field.ndim != 3 or field.shape[2] != n_arms:
        return equal
    powers = np.sum(np.abs(field) ** 2, axis=(0, 1))
    total = float(np.sum(powers))
    if total <= 0.0 or not np.all(np.isfinite(powers)):
        return equal
    return np.asarray(powers, dtype=np.float64) / total


def weighted_dark_port_matrix(
    arm_unitaries: NDArray[np.complex128],
    dark_indices: NDArray[np.intp],
    arm_weights: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Stack arm dark-port rows with sqrt(weight) scaling."""

    rows: list[NDArray[np.complex128]] = []
    for arm_index, weight in enumerate(arm_weights):
        rows.append(np.sqrt(float(weight)) * arm_unitaries[arm_index, dark_indices, :])
    return np.concatenate(rows, axis=0)


def subapertures_from_positions(
    positions_enu_m: NDArray[np.float64],
    diameters_m: NDArray[np.float64],
) -> tuple[SubAperture, ...]:
    """Build imaging ``SubAperture`` objects from shared inputs."""

    positions = np.asarray(positions_enu_m, dtype=np.float64)
    diameters = np.asarray(diameters_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions_enu_m must have shape (M, 3).")
    if diameters.shape != (positions.shape[0],):
        raise ValueError("diameters_m must have shape (M,).")
    return tuple(
        SubAperture(
            position_enu_m=positions[idx].astype(np.float64, copy=True),
            diameter_m=float(diameters[idx]),
        )
        for idx in range(positions.shape[0])
    )
