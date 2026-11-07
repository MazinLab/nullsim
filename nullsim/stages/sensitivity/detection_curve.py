"""Compute planet throughput vs separation, disk null floor, and realized null.

This stage assembles the three physics ingredients needed to convert the
chip-only optimization result into a per-separation detection sensitivity
curve. It reads:

- The chip unitary stack persisted by ``chip.kernel_mzi_mesh``.
- Per-mode SMF coupling × Strehl and aperture areas (derived from
  ``state.geometry``, ``state.wavefront``, and the n_apertures partition).
- AO piston OPD RMS per sub-aperture (derived from
  ``state.opd_budget`` or the pyramid_wfs result component).
- The stellar disk angular diameter from ``state.scene``.

and writes:

- ``planet_throughputs`` shape (n_sep,) — PA- and band-averaged.
- ``separations_mas`` shape (n_sep,) — the evaluation grid.
- ``disk_null_floor`` scalar — resolved-disk leakage in excess of the
  on-axis value already in the realized null.
- ``realized_null_depth`` scalar — AO-piston-driven mean leakage.
- ``null_fluctuation_rms`` scalar — RMS of the leakage fluctuation in
  null-depth units; ``null_instability`` is it divided by the effective
  null, and ``null_instability_mc`` is the band CV of the Monte Carlo.
- ``residual_starlight_contrast`` shape (n_sep,) — optional external
  residual stellar-leakage profile.
- ``effective_null_with_disk_by_separation`` shape (n_sep,) — realized,
  disk, and residual-starlight leakage on the separation grid.
- ``effective_null_with_disk`` scalar — the scene-separation value for
  legacy consumers and headline SNR.

These are consumed by ``performance.standard`` to build the contrast
curve. Following the SPEC §6.2 component layout, all outputs live under
a single ``state.results.components`` entry; the headline scalars are
also surfaced in the typed ``Results`` channel where useful.

Reference: KILO ``natural_design._run_sensitivity`` (line 1652+) and
``performance.compute_planet_throughput_vs_separation`` (lines 1010-1128).
"""

from __future__ import annotations

import csv
import hashlib
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import (
    BaseModel,
    ConfigDict,
    model_validator,
)
from pydantic import (
    Field as PydanticField,
)

from nullsim.physics.fringe_tracking import (
    photon_limited_setpoint_trim,
    temporal_kalman_piston_covariance,
)
from nullsim.physics.planet_throughput import (
    compute_planet_throughput_vs_separation,
    stellar_input_field,
)
from nullsim.physics.realized_null import (
    RealizedNullResult,
    band_leakage_statistics,
    bias_cross_moments,
    piston_covariance_with_common_mode_m2,
    realized_null_depth_mc,
)
from nullsim.physics.ruilier_cassaing import coupling_efficiency
from nullsim.physics.scintillation import (
    AmplitudeServoResidual,
    amplitude_servo_residual_sigma_i,
    optimize_amplitude_servo_bandwidth,
)
from nullsim.physics.stellar_disk import stellar_disk_null_floor
from nullsim.physics.systematic_bandwidth import effective_systematic_bandwidth_hz
from nullsim.pipeline import (
    BaseStage,
    ExternalDependency,
    SimulationState,
    register_stage,
)
from nullsim.stages._mode_splitter import (
    component_mode_areas_m2,
    component_mode_diameters_m,
    splitter_physical_indices,
)
from nullsim.stages._science_bands import (
    infer_band_from_wavelength_um,
    simultaneous_band_rate_scale,
)
from nullsim.stages._utils import (
    mode_origin_positions_m,
    project_aperture_positions_for_pointing,
    unique_component_name,
)

__all__ = ["DetectionCurve"]


_CHIP_PREFIX = "chip.kernel_mzi_mesh"
_AO_PREFIX = "ao."
_FRINGE_OBSERVABILITY_PREFIXES = (
    "fringe_observability",
    "diagnostics.fringe_observability",
)
_INJECTION_COMPONENT_PREFIXES = ("single_mode_fiber", "injection_ideal")
_RAD_PER_MAS = 1.0 / (206265.0 * 1000.0)
_RESIDUAL_SEPARATION_COLUMNS = (
    "separation_mas",
    "angular_separation_mas",
    "rho_mas",
)
_RESIDUAL_CONTRAST_COLUMNS = (
    "contrast",
    "raw_contrast",
    "core_contrast",
    "residual_starlight_contrast",
)
_RESIDUAL_WAVELENGTH_COLUMNS = (
    "wavelength_um",
    "lambda_um",
    "wl_um",
)
_FRINGE_OBSERVABILITY_PISTON_METRICS = {
    "linear_monte_carlo_median": (("monte_carlo", "median_residual_rms_nm"),),
    "linear_monte_carlo_p95": (("monte_carlo", "p95_residual_rms_nm"),),
    "crlb": (("crlb_piston_rms_nm",),),
    "crlb_truncated": (("crlb_truncated_piston_rms_nm",),),
    "cophasing_handoff_science_median": (
        ("cophasing_handoff_monte_carlo", "median_science_residual_rms_nm"),
    ),
    "cophasing_handoff_science_p95": (
        ("cophasing_handoff_monte_carlo", "p95_science_residual_rms_nm"),
    ),
}
_FRINGE_OBSERVABILITY_AUTO_PISTON_METRICS = (
    ("monte_carlo", "median_residual_rms_nm"),
    ("crlb_truncated_piston_rms_nm",),
    ("crlb_piston_rms_nm",),
    ("cophasing_handoff_monte_carlo", "median_science_residual_rms_nm"),
)


def _find_chip_unitary_component(state: SimulationState) -> tuple[str, dict[str, Any]]:
    """Return (component_name, payload) for the chip unitary stack."""

    candidates = [
        (name, payload)
        for name, payload in state.results.components.items()
        if name.startswith(_CHIP_PREFIX) and isinstance(payload, dict)
        and "u_stack" in payload
    ]
    if not candidates:
        raise ValueError(
            "sensitivity.detection_curve requires a 'chip.kernel_mzi_mesh' "
            "result component with 'u_stack'. Add the chip stage upstream."
        )
    if len(candidates) > 1:
        raise ValueError(
            "sensitivity.detection_curve: multiple chip.kernel_mzi_mesh "
            f"components found ({[name for name, _ in candidates]}); "
            "the multi-chip case is not yet supported."
        )
    return candidates[0]


def _unitary_stacks_for_detection(
    *,
    state: SimulationState,
    chip_payload: dict[str, Any],
    u_stack: NDArray[np.complex128],
) -> tuple[NDArray[np.complex128], tuple[str, ...], NDArray[np.float64], str, bool]:
    """Return unitary stacks in the phase frame used by geometry fields."""

    deployment_mode = str(chip_payload.get("deployment_mode", "shared"))
    if deployment_mode != "per_pol_arm":
        return (
            u_stack[np.newaxis, :, :, :],
            ("shared",),
            np.ones(1, dtype=np.float64),
            deployment_mode,
            False,
        )

    if "u_stack_per_arm" not in chip_payload:
        raise ValueError(
            "sensitivity.detection_curve: chip.kernel_mzi_mesh reports "
            "deployment_mode='per_pol_arm' but did not publish u_stack_per_arm."
        )
    stacks = np.asarray(chip_payload["u_stack_per_arm"], dtype=np.complex128)
    if stacks.ndim != 4:
        raise ValueError(
            "sensitivity.detection_curve: u_stack_per_arm must have shape "
            f"(n_arm, n_modes, n_modes, n_wl); got {stacks.shape}."
        )
    if stacks.shape[0] <= 0:
        raise ValueError(
            "sensitivity.detection_curve: u_stack_per_arm must contain at "
            "least one deployed arm."
        )
    labels_raw = chip_payload.get("chip_channel_labels")
    labels = tuple(labels_raw) if labels_raw is not None else tuple(
        f"arm_{index}" for index in range(stacks.shape[0])
    )
    if len(labels) != stacks.shape[0]:
        raise ValueError(
            "sensitivity.detection_curve: chip_channel_labels length "
            f"{len(labels)} does not match u_stack_per_arm arm count "
            f"{stacks.shape[0]}."
        )
    geometry_stacks, phase_compensated = _geometry_field_unitary_stacks(
        chip_payload=chip_payload,
        deployed_stacks=stacks,
    )
    return (
        geometry_stacks,
        labels,
        _unitary_arm_weights(state, stacks.shape[0]),
        deployment_mode,
        phase_compensated,
    )


def _geometry_field_unitary_stacks(
    *,
    chip_payload: dict[str, Any],
    deployed_stacks: NDArray[np.complex128],
) -> tuple[NDArray[np.complex128], bool]:
    """Map deployed per-arm stacks into the ideal-geometry input phase frame.

    ``chip.kernel_mzi_mesh`` publishes ``u_stack_per_arm`` as the physical
    deployed stack, including each arm's diagonal input phase trim:
    ``U @ diag(correction)``. The sensitivity stage's ideal star/planet
    fields do not carry the static PM-fiber phases that the trim cancels.
    Multiplying the stack columns by ``conj(correction)`` makes
    ``U @ diag(correction)`` acting on the physical phased input
    equivalent to this returned stack acting on the real geometry field.
    """

    raw_correction = chip_payload.get("input_phase_correction")
    if raw_correction is None:
        return deployed_stacks, False
    correction = np.asarray(raw_correction, dtype=np.complex128)
    expected_shape = (deployed_stacks.shape[1], deployed_stacks.shape[0])
    if correction.shape != expected_shape:
        raise ValueError(
            "sensitivity.detection_curve: input_phase_correction has shape "
            f"{correction.shape}; expected {expected_shape} for u_stack_per_arm "
            f"shape {deployed_stacks.shape}."
        )
    inverse_static_phase = np.conj(correction).T[:, np.newaxis, :, np.newaxis]
    return deployed_stacks * inverse_static_phase, True


def _unitary_arm_weights(
    state: SimulationState,
    n_arms: int,
) -> NDArray[np.float64]:
    """Flux weights for incoherently combining deployed polarization arms."""

    equal = np.full(n_arms, 1.0 / float(n_arms), dtype=np.float64)
    amplitudes = state.field.amplitudes
    if amplitudes is None or amplitudes.ndim != 3 or amplitudes.shape[2] != n_arms:
        return equal
    powers = np.sum(np.abs(np.asarray(amplitudes)) ** 2, axis=(0, 1))
    total = float(np.sum(powers))
    if total <= 0.0 or not np.all(np.isfinite(powers)):
        return equal
    return np.asarray(powers, dtype=np.float64) / total


def _delay_line_diagnostic_payload(state: SimulationState) -> dict[str, Any] | None:
    """Return the newest geometric-delay diagnostic payload, if present."""

    for name, payload in reversed(state.throughput.components.items()):
        if not name.startswith("delay_line_loss"):
            continue
        if not isinstance(payload, dict):
            continue
        per_aperture = payload.get("per_aperture")
        if isinstance(per_aperture, dict):
            return per_aperture
    return None


def _has_residual_starlight_profile(params: Any) -> bool:
    """Return True when a residual-starlight profile is configured."""

    return (
        bool(params.residual_starlight_contrast)
        or params.residual_starlight_table_path is not None
    )


def _validate_residual_starlight_inline_params(params: Any) -> None:
    """Validate inline residual-starlight profile parameters."""

    has_table = params.residual_starlight_table_path is not None
    has_separations = bool(params.residual_starlight_separations_mas)
    has_contrasts = bool(params.residual_starlight_contrast)
    if has_table and (has_separations or has_contrasts):
        raise ValueError(
            "residual_starlight_table_path cannot be combined with inline "
            "residual_starlight_separations_mas/residual_starlight_contrast."
        )
    if has_separations and not has_contrasts:
        raise ValueError(
            "residual_starlight_separations_mas requires "
            "residual_starlight_contrast."
        )
    if not has_contrasts:
        return

    separations = np.asarray(params.residual_starlight_separations_mas, dtype=float)
    contrasts = np.asarray(params.residual_starlight_contrast, dtype=float)
    _validate_residual_starlight_samples(
        separations_mas=separations,
        contrast=contrasts,
        source="inline residual_starlight_* params",
        allow_constant_without_separation=True,
    )


def _validate_residual_starlight_samples(
    *,
    separations_mas: NDArray[np.float64],
    contrast: NDArray[np.float64],
    source: str,
    allow_constant_without_separation: bool,
) -> None:
    """Validate radial residual-starlight contrast samples."""

    if contrast.ndim != 1 or contrast.size == 0:
        raise ValueError(f"{source}: contrast samples must be a non-empty 1-D array.")
    if not np.all(np.isfinite(contrast)) or np.any(contrast < 0.0):
        raise ValueError(f"{source}: contrast samples must be finite and non-negative.")

    if separations_mas.size == 0:
        if allow_constant_without_separation and contrast.size == 1:
            return
        raise ValueError(
            f"{source}: multiple contrast samples require matching "
            "separation_mas samples."
        )
    if separations_mas.ndim != 1:
        raise ValueError(f"{source}: separation_mas samples must be a 1-D array.")
    if separations_mas.size != contrast.size:
        raise ValueError(
            f"{source}: separation_mas and contrast samples must have the same "
            f"length ({separations_mas.size} != {contrast.size})."
        )
    if not np.all(np.isfinite(separations_mas)) or np.any(separations_mas < 0.0):
        raise ValueError(
            f"{source}: separation_mas samples must be finite and non-negative."
        )
    if separations_mas.size > 1 and np.any(np.diff(separations_mas) <= 0.0):
        raise ValueError(
            f"{source}: separation_mas samples must be strictly increasing."
        )


def _residual_starlight_profile_for_grid(
    *,
    params: Any,
    separations_mas: NDArray[np.float64],
    wavelength_grid_um: NDArray[np.float64],
    wavelength_weights: NDArray[np.float64],
    config_dir: Path | None = None,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Return residual-starlight contrast sampled on the sensitivity grid."""

    if not _has_residual_starlight_profile(params):
        return np.zeros_like(separations_mas, dtype=np.float64), {
            "applied": False,
            "source": None,
            "table_path": None,
            "combine_mode": params.residual_starlight_combine_mode,
            "wavelengths_um": None,
            "wavelength_weights": None,
            "contrast_by_wavelength": None,
        }

    if params.residual_starlight_table_path is not None:
        table_path = _resolve_residual_starlight_table_path(
            params.residual_starlight_table_path,
            config_dir=config_dir,
        )
        (
            profile_separations,
            profile_wavelengths_um,
            profile_contrast,
        ) = _read_residual_starlight_table(table_path)
        source = f"table:{table_path}"
        table_path_out = str(table_path)
    else:
        profile_separations = np.asarray(
            params.residual_starlight_separations_mas,
            dtype=np.float64,
        )
        profile_wavelengths_um = np.array((), dtype=np.float64)
        profile_contrast = np.asarray(
            params.residual_starlight_contrast,
            dtype=np.float64,
        )
        source = "inline"
        table_path_out = None

    profile, contrast_by_wavelength = _interpolate_residual_starlight_profile(
        input_separations_mas=profile_separations,
        input_wavelengths_um=profile_wavelengths_um,
        input_contrast=profile_contrast,
        output_separations_mas=separations_mas,
        output_wavelengths_um=wavelength_grid_um,
        output_wavelength_weights=wavelength_weights,
    )
    return profile, {
        "applied": True,
        "source": source,
        "table_path": table_path_out,
        "combine_mode": params.residual_starlight_combine_mode,
        "interpolation": (
            "constant" if profile_contrast.size == 1
            else (
                "bilinear contrast vs separation_mas and wavelength_um "
                "with endpoint clamping"
                if profile_contrast.ndim == 2
                else "linear contrast vs separation_mas with endpoint clamping"
            )
        ),
        "wavelengths_um": (
            None if contrast_by_wavelength is None
            else tuple(float(v) for v in wavelength_grid_um)
        ),
        "wavelength_weights": (
            None if contrast_by_wavelength is None
            else tuple(float(v) for v in wavelength_weights)
        ),
        "contrast_by_wavelength": contrast_by_wavelength,
    }


def _angular_jitter_convolve_profile(
    profile: NDArray[np.float64],
    *,
    separations_mas: NDArray[np.float64],
    sigma_mas: float,
) -> NDArray[np.float64]:
    """Gaussian-smooth a radial profile by an RMS pointing jitter in mas."""

    values = np.asarray(profile, dtype=np.float64)
    if float(sigma_mas) <= 0.0:
        return values.astype(np.float64, copy=True)
    if values.shape[-1] != separations_mas.size:
        raise ValueError(
            "angular jitter convolution requires the last profile axis to "
            f"match separations_mas ({values.shape[-1]} != "
            f"{separations_mas.size})."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("angular jitter convolution profile values must be finite.")
    weights = _angular_jitter_kernel(
        separations_mas=separations_mas,
        sigma_mas=float(sigma_mas),
    )
    return np.tensordot(values, weights.T, axes=([-1], [0])).astype(
        np.float64,
        copy=False,
    )


def _angular_jitter_kernel(
    *,
    separations_mas: NDArray[np.float64],
    sigma_mas: float,
) -> NDArray[np.float64]:
    """Return the Gaussian smoothing matrix for a profile on a 1-D grid.

    The profile is linearly interpolated onto a fine grid (step sigma/8
    within 6 sigma of every node, plus the nodes), and the Gaussian is
    integrated there with trapezoid weights, so nonuniform node spacing
    does not bias the average toward the densely sampled side (review
    2026-11-07 row 30). Rows are normalized over the grid's own range.
    """

    separations = np.asarray(separations_mas, dtype=np.float64)
    if separations.ndim != 1 or separations.size == 0:
        raise ValueError("separations_mas must be a non-empty 1-D grid.")
    if not np.all(np.isfinite(separations)):
        raise ValueError("separations_mas must be finite.")
    if not math.isfinite(float(sigma_mas)) or sigma_mas <= 0.0:
        raise ValueError("sigma_mas must be finite and positive.")
    if separations.size == 1:
        return np.ones((1, 1), dtype=np.float64)
    sigma = float(sigma_mas)
    offsets = sigma * np.linspace(-6.0, 6.0, 97)
    fine = np.unique(np.concatenate([
        separations,
        np.clip(
            (separations[:, np.newaxis] + offsets[np.newaxis, :]).ravel(),
            separations[0],
            separations[-1],
        ),
    ]))
    spacing = np.diff(fine)
    trapezoid = 0.5 * (
        np.concatenate(([0.0], spacing)) + np.concatenate((spacing, [0.0]))
    )
    weights = (
        np.exp(-0.5 * ((separations[:, np.newaxis] - fine[np.newaxis, :]) / sigma) ** 2)
        * trapezoid[np.newaxis, :]
    )
    weights /= np.sum(weights, axis=1, keepdims=True)
    # Linear interpolation from the nodes onto the fine grid, column by node.
    interpolation = np.stack(
        [np.interp(fine, separations, column) for column in np.eye(separations.size)],
        axis=1,
    )
    return weights @ interpolation


def _resolve_pointing_jitter_rms_mas(
    *,
    state: SimulationState,
    params: Any,
) -> tuple[float, str]:
    """Resolve pointing jitter from params or upstream wavefront diagnostics."""

    configured = params.pointing_jitter_rms_mas
    if configured is not None:
        return float(configured), "params.pointing_jitter_rms_mas"
    for name, payload in reversed(tuple(state.wavefront.components.items())):
        if not isinstance(payload, dict) or "pointing_jitter_rms_mas" not in payload:
            continue
        value = float(payload["pointing_jitter_rms_mas"])
        if value < 0.0 or not math.isfinite(value):
            raise ValueError(
                "sensitivity.detection_curve: upstream wavefront component "
                f"{name!r} reported invalid pointing_jitter_rms_mas={value!r}."
            )
        return value, f"wavefront:{name}.pointing_jitter_rms_mas"
    return 0.0, "none"


def _interpolate_residual_starlight_profile(
    *,
    input_separations_mas: NDArray[np.float64],
    input_wavelengths_um: NDArray[np.float64],
    input_contrast: NDArray[np.float64],
    output_separations_mas: NDArray[np.float64],
    output_wavelengths_um: NDArray[np.float64],
    output_wavelength_weights: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Interpolate a radial residual-starlight profile onto a target grid."""

    if input_contrast.ndim == 1:
        _validate_residual_starlight_samples(
            separations_mas=input_separations_mas,
            contrast=input_contrast,
            source="residual starlight profile",
            allow_constant_without_separation=True,
        )
        if input_contrast.size == 1:
            return np.full_like(
                output_separations_mas,
                float(input_contrast[0]),
                dtype=np.float64,
            ), None
        return np.interp(
            output_separations_mas,
            input_separations_mas,
            input_contrast,
        ).astype(np.float64, copy=False), None

    _validate_residual_starlight_surface(
        separations_mas=input_separations_mas,
        wavelengths_um=input_wavelengths_um,
        contrast=input_contrast,
        source="residual starlight profile",
    )
    if output_wavelengths_um.ndim != 1 or output_wavelengths_um.size == 0:
        raise ValueError(
            "residual starlight wavelength interpolation requires a non-empty "
            "1-D output wavelength grid."
        )
    if output_wavelength_weights.shape != output_wavelengths_um.shape:
        raise ValueError(
            "residual starlight wavelength weights must match the output "
            "wavelength grid."
        )
    interp_by_input_wavelength = np.stack(
        [
            np.interp(
                output_separations_mas,
                input_separations_mas,
                input_contrast[:, wavelength_index],
            )
            for wavelength_index in range(input_wavelengths_um.size)
        ],
        axis=1,
    )
    contrast_by_wavelength = np.stack(
        [
            np.interp(
                output_wavelengths_um,
                input_wavelengths_um,
                interp_by_input_wavelength[sep_index, :],
            )
            for sep_index in range(output_separations_mas.size)
        ],
        axis=0,
    ).astype(np.float64, copy=False)
    band_profile = np.average(
        contrast_by_wavelength,
        axis=1,
        weights=output_wavelength_weights,
    ).astype(np.float64, copy=False)
    return band_profile, contrast_by_wavelength


def _validate_residual_starlight_surface(
    *,
    separations_mas: NDArray[np.float64],
    wavelengths_um: NDArray[np.float64],
    contrast: NDArray[np.float64],
    source: str,
) -> None:
    """Validate a wavelength-resolved residual-starlight contrast surface."""

    _validate_residual_starlight_samples(
        separations_mas=separations_mas,
        contrast=np.ones(separations_mas.size, dtype=np.float64),
        source=source,
        allow_constant_without_separation=False,
    )
    if wavelengths_um.ndim != 1 or wavelengths_um.size == 0:
        raise ValueError(
            f"{source}: wavelength_um samples must be a non-empty 1-D array."
        )
    if not np.all(np.isfinite(wavelengths_um)) or np.any(wavelengths_um <= 0.0):
        raise ValueError(f"{source}: wavelength_um samples must be finite and positive.")
    if wavelengths_um.size > 1 and np.any(np.diff(wavelengths_um) <= 0.0):
        raise ValueError(f"{source}: wavelength_um samples must be strictly increasing.")
    if contrast.shape != (separations_mas.size, wavelengths_um.size):
        raise ValueError(
            f"{source}: contrast surface shape {contrast.shape} must equal "
            f"(n_separation, n_wavelength)=({separations_mas.size}, "
            f"{wavelengths_um.size})."
        )
    if not np.all(np.isfinite(contrast)) or np.any(contrast < 0.0):
        raise ValueError(f"{source}: contrast surface must be finite and non-negative.")


def _residual_starlight_wavelength_weights(
    state: SimulationState,
    wavelength_grid_um: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return normalized spectral weights for band-averaging residual contrast."""

    wavelengths = np.asarray(wavelength_grid_um, dtype=np.float64)
    if wavelengths.ndim != 1 or wavelengths.size == 0:
        raise ValueError("residual starlight wavelength grid must be non-empty and 1-D.")
    weights = _wavelength_bin_widths_um(wavelengths)
    transmission = state.throughput.transmission_fraction
    if transmission and len(transmission) == wavelengths.size:
        transmission_arr = np.asarray(transmission, dtype=np.float64)
        finite = np.isfinite(transmission_arr) & (transmission_arr >= 0.0)
        if np.any(finite):
            weights = weights * np.where(finite, transmission_arr, 0.0)
    total = float(np.sum(weights))
    if not math.isfinite(total) or total <= 0.0:
        weights = np.ones(wavelengths.size, dtype=np.float64)
        total = float(weights.size)
    return weights / total


def _wavelength_bin_widths_um(wavelength_grid_um: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return positive bin widths for a wavelength grid in microns."""

    wavelengths = np.asarray(wavelength_grid_um, dtype=np.float64)
    if wavelengths.size == 1:
        return np.ones(1, dtype=np.float64)
    widths = np.gradient(wavelengths)
    widths = np.abs(np.asarray(widths, dtype=np.float64))
    if not np.all(np.isfinite(widths)) or np.any(widths <= 0.0):
        return np.ones(wavelengths.size, dtype=np.float64)
    return widths


def _apply_residual_starlight_to_effective_null(
    *,
    base_effective_null: float,
    residual_profile: NDArray[np.float64],
    combine_mode: Literal["max", "add", "replace"],
) -> NDArray[np.float64]:
    """Combine chip/disk leakage with an external residual-starlight profile."""

    base_profile = np.full_like(
        residual_profile,
        float(base_effective_null),
        dtype=np.float64,
    )
    if combine_mode == "max":
        return np.maximum(base_profile, residual_profile)
    if combine_mode == "add":
        return base_profile + residual_profile
    if combine_mode == "replace":
        return residual_profile.astype(np.float64, copy=True)
    raise ValueError(f"Unsupported residual_starlight_combine_mode: {combine_mode!r}")


def _opd_leakage_fraction(opd_rms_nm: float, wavelength_m: float) -> float:
    """Small-phase null leakage charged for dark-port trim residuals.

    Photon-counting focal-plane control heritage (e.g. MEC/SCExAO speckle
    nulling) motivates a slow science-port setpoint trim. Its residual bias
    and deliberate dither are known OPD terms, so their leakage is charged
    directly. They add no variance of their own, only the bias cross term
    with the random piston (see the null-instability block in ``apply``).
    """

    if opd_rms_nm < 0.0 or not math.isfinite(float(opd_rms_nm)):
        raise ValueError("opd_rms_nm must be finite and non-negative.")
    if wavelength_m <= 0.0 or not math.isfinite(float(wavelength_m)):
        raise ValueError("wavelength_m must be finite and positive.")
    phase_rad = 2.0 * math.pi * float(opd_rms_nm) * 1.0e-9 / float(wavelength_m)
    return float((phase_rad * phase_rad) / 4.0)


def _effective_null_scalar_from_curve(
    *,
    effective_null_curve: NDArray[np.float64],
    separations_mas: NDArray[np.float64],
    scene_planet_separation_mas: float | None,
) -> tuple[float, str]:
    """Return the scene-appropriate scalar null for legacy consumers."""

    if effective_null_curve.size == 0:
        raise ValueError("effective_null_curve must be non-empty.")
    if scene_planet_separation_mas is None:
        return (
            float(np.max(effective_null_curve)),
            "max(effective_null_with_disk_by_separation)",
        )
    return (
        float(
            np.interp(
                float(scene_planet_separation_mas),
                separations_mas,
                effective_null_curve,
            )
        ),
        "interpolated at scene.planet_separation_mas",
    )


def _read_residual_starlight_table(
    table_path: Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Read a residual-starlight contrast table from CSV."""

    try:
        with table_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(
                    f"residual starlight table {table_path} has no header row."
                )
            column_lookup = {
                name.strip().lower(): name for name in reader.fieldnames
            }
            separation_column = _first_present_column(
                column_lookup,
                _RESIDUAL_SEPARATION_COLUMNS,
                table_path,
            )
            contrast_column = _first_present_column(
                column_lookup,
                _RESIDUAL_CONTRAST_COLUMNS,
                table_path,
            )
            wavelength_column = _optional_present_column(
                column_lookup,
                _RESIDUAL_WAVELENGTH_COLUMNS,
            )
            rows = []
            for row in reader:
                separation = float(row[separation_column])
                contrast = float(row[contrast_column])
                if wavelength_column is None:
                    rows.append((separation, contrast))
                else:
                    rows.append((separation, float(row[wavelength_column]), contrast))
    except OSError as exc:
        raise OSError(
            f"Could not read residual starlight table {table_path}."
        ) from exc
    except ValueError as exc:
        raise ValueError(
            f"Invalid residual starlight table {table_path}: {exc}"
        ) from exc

    if not rows:
        raise ValueError(f"residual starlight table {table_path} contains no rows.")
    if len(rows[0]) == 3:
        return _residual_starlight_surface_from_rows(table_path, rows)

    ordered_rows = sorted(rows, key=lambda item: item[0])
    separations = np.asarray(
        [separation for separation, _contrast in ordered_rows],
        dtype=np.float64,
    )
    contrasts = np.asarray(
        [contrast for _separation, contrast in ordered_rows],
        dtype=np.float64,
    )
    _validate_residual_starlight_samples(
        separations_mas=separations,
        contrast=contrasts,
        source=f"residual starlight table {table_path}",
        allow_constant_without_separation=False,
    )
    return separations, np.array((), dtype=np.float64), contrasts


def _residual_starlight_surface_from_rows(
    table_path: Path,
    rows: list[tuple[float, float, float]],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return sorted separation/wavelength grids and contrast surface."""

    separations = np.asarray(sorted({row[0] for row in rows}), dtype=np.float64)
    wavelengths = np.asarray(sorted({row[1] for row in rows}), dtype=np.float64)
    contrast = np.full(
        (separations.size, wavelengths.size),
        np.nan,
        dtype=np.float64,
    )
    separation_index = {float(value): index for index, value in enumerate(separations)}
    wavelength_index = {float(value): index for index, value in enumerate(wavelengths)}
    for separation, wavelength, value in rows:
        i_sep = separation_index[float(separation)]
        i_wavelength = wavelength_index[float(wavelength)]
        if np.isfinite(contrast[i_sep, i_wavelength]):
            raise ValueError(
                "duplicate residual starlight table row for "
                f"separation_mas={separation} and wavelength_um={wavelength}."
            )
        contrast[i_sep, i_wavelength] = float(value)
    if np.any(~np.isfinite(contrast)):
        raise ValueError(
            "wavelength-resolved residual starlight tables must contain a "
            "complete separation_mas × wavelength_um grid."
        )
    _validate_residual_starlight_surface(
        separations_mas=separations,
        wavelengths_um=wavelengths,
        contrast=contrast,
        source=f"residual starlight table {table_path}",
    )
    return separations, wavelengths, contrast


def _first_present_column(
    column_lookup: dict[str, str],
    candidates: tuple[str, ...],
    table_path: Path,
) -> str:
    """Return the first supported CSV column present in a table."""

    for candidate in candidates:
        if candidate in column_lookup:
            return column_lookup[candidate]
    raise ValueError(
        f"{table_path} must include one of columns {', '.join(candidates)}."
    )


def _optional_present_column(
    column_lookup: dict[str, str],
    candidates: tuple[str, ...],
) -> str | None:
    """Return the first supported CSV column present, if any."""

    for candidate in candidates:
        if candidate in column_lookup:
            return column_lookup[candidate]
    return None


def _resolve_residual_starlight_table_path(
    path_str: str,
    *,
    config_dir: Path | None = None,
) -> Path:
    """Resolve a residual-starlight CSV path from config, cwd, or package roots."""

    candidate = Path(path_str).expanduser()
    if candidate.is_absolute():
        return candidate
    if config_dir is not None:
        config_relative = (config_dir / candidate).resolve()
        if config_relative.exists():
            return config_relative
    if candidate.exists():
        return candidate
    package_root = Path(__file__).resolve().parents[2]
    project_root = package_root.parent
    for base in (project_root, package_root):
        resolved = base / candidate
        if resolved.exists():
            return resolved
    return candidate


def _config_dir_from_context(context: Any) -> Path | None:
    source_path = getattr(context, "source_path", None)
    if source_path is None:
        return None
    return Path(source_path).expanduser().resolve().parent


def _ideal_realized_null_for_stack(
    *,
    u_stack: NDArray[np.complex128],
    stellar_field: NDArray[np.complex128],
    dark_indices: NDArray[np.intp],
    wavelength_weights: NDArray[np.float64],
) -> RealizedNullResult:
    """Chip-only realized-null summary for one deployed arm stack."""

    from nullsim.physics.realized_null import (  # noqa: WPS433
        _chip_null_depth as _cn,
    )
    from nullsim.physics.realized_null import (
        _chip_null_depth_kilo as _cn_kilo,
    )

    n_modes = u_stack.shape[1]
    n_wl = u_stack.shape[2]
    field = np.asarray(stellar_field, dtype=np.complex128)
    if field.ndim == 1:
        if field.shape != (n_modes,):
            raise ValueError(
                "sensitivity.detection_curve: stellar_field shape "
                f"{field.shape} must equal (M,) = ({n_modes},)."
            )
        fields_per_az_wl = np.broadcast_to(
            field[np.newaxis, np.newaxis, :],
            (1, n_wl, n_modes),
        )
    elif field.ndim == 2:
        if field.shape[1] != n_modes:
            raise ValueError(
                "sensitivity.detection_curve: stellar_field shape "
                f"{field.shape} must end in M = {n_modes}."
            )
        fields_per_az_wl = np.broadcast_to(
            field[:, np.newaxis, :],
            (field.shape[0], n_wl, n_modes),
        )
    elif field.ndim == 3:
        if field.shape[1:] != (n_wl, n_modes):
            raise ValueError(
                "sensitivity.detection_curve: stellar_field shape "
                f"{field.shape} must be (n_az, n_wl, M) with "
                f"(n_wl, M)=({n_wl}, {n_modes})."
            )
        fields_per_az_wl = field
    else:
        raise ValueError(
            "sensitivity.detection_curve: stellar_field must be 1-D, 2-D, or 3-D."
        )
    n_az = int(fields_per_az_wl.shape[0])
    ideal = float(wavelength_weights @ np.array([
        np.mean([
            _cn(u_stack[:, :, k], fields_per_az_wl[az_idx, k], dark_indices)
            for az_idx in range(n_az)
        ])
        for k in range(n_wl)
    ]))
    ideal_kilo = float(wavelength_weights @ np.array([
        np.mean([
            _cn_kilo(u_stack[:, :, k], fields_per_az_wl[az_idx, k], dark_indices)
            for az_idx in range(n_az)
        ])
        for k in range(n_wl)
    ]))
    return RealizedNullResult(
        mean_null_depth=ideal,
        median_null_depth=ideal,
        std_null_depth=0.0,
        null_instability=0.0,
        ideal_null_depth=ideal,
        ideal_null_depth_kilo=ideal_kilo,
        mean_null_depth_kilo=ideal_kilo,
        n_samples=0,
    )


def _combine_realized_null_results(
    results: Sequence[RealizedNullResult],
    weights: NDArray[np.float64],
    wavelength_weights: NDArray[np.float64],
) -> RealizedNullResult:
    """Incoherently combine per-arm realized-null distributions.

    Arm dark power adds (flux-weighted samples, paired by draw), then the
    band statistics follow :func:`band_leakage_statistics`.
    """

    if len(results) == 1:
        return results[0]
    weights_arr = np.asarray(weights, dtype=np.float64)
    samples = [result.samples_by_wavelength for result in results]
    if all(sample is not None for sample in samples):
        sample_arrays = [
            np.asarray(sample, dtype=np.float64)
            for sample in samples
            if sample is not None
        ]
        first_shape = sample_arrays[0].shape
        if all(sample.shape == first_shape for sample in sample_arrays):
            combined_samples = np.zeros(first_shape, dtype=np.float64)
            for weight, sample in zip(weights_arr, sample_arrays, strict=True):
                combined_samples += float(weight) * sample
            n_azimuths = max(result.n_azimuths for result in results)
            band_mean, band_rms = band_leakage_statistics(
                combined_samples, wavelength_weights, n_azimuths,
            )
            all_samples = combined_samples.reshape(-1)

            kilo_samples = [result.samples_by_wavelength_kilo for result in results]
            if all(sample is not None for sample in kilo_samples):
                kilo_arrays = [
                    np.asarray(sample, dtype=np.float64)
                    for sample in kilo_samples
                    if sample is not None
                ]
                if all(sample.shape == first_shape for sample in kilo_arrays):
                    combined_kilo = np.zeros(first_shape, dtype=np.float64)
                    for weight, sample in zip(weights_arr, kilo_arrays, strict=True):
                        combined_kilo += float(weight) * sample
                    mean_kilo = float(
                        wavelength_weights @ np.mean(combined_kilo, axis=1)
                    )
                else:
                    mean_kilo = float(sum(
                        float(weight) * float(result.mean_null_depth_kilo)
                        for weight, result in zip(weights_arr, results, strict=True)
                    ))
            else:
                mean_kilo = float(sum(
                    float(weight) * float(result.mean_null_depth_kilo)
                    for weight, result in zip(weights_arr, results, strict=True)
                ))

            return RealizedNullResult(
                mean_null_depth=band_mean,
                median_null_depth=float(np.median(all_samples)),
                std_null_depth=float(np.std(all_samples)),
                null_instability=band_rms / band_mean if band_mean > 0.0 else 0.0,
                null_fluctuation_rms=band_rms,
                ideal_null_depth=float(sum(
                    float(weight) * float(result.ideal_null_depth)
                    for weight, result in zip(weights_arr, results, strict=True)
                )),
                ideal_null_depth_kilo=float(sum(
                    float(weight) * float(result.ideal_null_depth_kilo)
                    for weight, result in zip(weights_arr, results, strict=True)
                )),
                mean_null_depth_kilo=mean_kilo,
                n_samples=int(all_samples.size),
                n_azimuths=n_azimuths,
                n_realizations_per_azimuth=max(
                    result.n_realizations_per_azimuth for result in results
                ),
            )

    def weighted(attr: str) -> float:
        return float(sum(
            float(weight) * float(getattr(result, attr))
            for weight, result in zip(weights_arr, results, strict=True)
        ))

    # Unpaired fallback (no stored samples): arms treated as fully
    # correlated, which bounds the combined RMS from above.
    mean = weighted("mean_null_depth")
    rms = weighted("null_fluctuation_rms")
    return RealizedNullResult(
        mean_null_depth=mean,
        median_null_depth=weighted("median_null_depth"),
        std_null_depth=weighted("std_null_depth"),
        null_instability=0.0 if mean <= 0.0 else rms / mean,
        null_fluctuation_rms=rms,
        ideal_null_depth=weighted("ideal_null_depth"),
        ideal_null_depth_kilo=weighted("ideal_null_depth_kilo"),
        mean_null_depth_kilo=weighted("mean_null_depth_kilo"),
        n_samples=max(result.n_samples for result in results),
        n_azimuths=max(result.n_azimuths for result in results),
        n_realizations_per_azimuth=max(
            result.n_realizations_per_azimuth for result in results
        ),
    )


def _find_ao_piston_rms_per_sub_m(state: SimulationState) -> float:
    """Return a scalar per-sub-aperture AO piston OPD RMS in meters.

    KILO's ``compute_servo_corrected_piston`` (fringe_tracking.py:473)
    USES the bright-port photon rate to feed a piston servo that
    subtracts the temporal-lag component of the AO piston. The result
    is a per-sub corrected piston ≈ irreducible (fitting+WFS noise
    piston-filtered) + lag-reduced temporal + photon shot noise. This
    corrected value REPLACES the raw AO piston for the realized-null
    MC; it is not summed with it.

    Tier 2.4 added the FT residual to the AO piston in quadrature,
    which is wrong: the FT residual is the corrected AO piston, not an
    additional independent noise term. Quadrature with AO at ~10 nm
    and FT residual at ~1 nm gives ~10 nm — same as raw AO — and so
    σ_OPD ends up 8-20× over KILO and realized null 100-400× over.

    Resolution policy now matches KILO:
      1. If fringe-tracking closed-loop residual is present, USE IT
         directly as the per-sub piston (the servo-corrected output).
      2. Otherwise fall back to the raw AO piston (space pipelines
         with no FT stage).
    """

    ao_piston_nm: float = 0.0
    ao_component = state.wavefront.components.get("ao_residual")
    if isinstance(ao_component, dict):
        piston_opd_nm = ao_component.get("piston_opd_nm", {})
        if isinstance(piston_opd_nm, dict):
            total = piston_opd_nm.get("total")
            if total is not None:
                arr = np.atleast_1d(np.asarray(total, dtype=np.float64))
                if arr.size > 0:
                    ao_piston_nm = float(np.mean(arr))

    ft_piston_nm: float = 0.0
    ft_prefix = "fringe_tracking.closed_loop_residual"
    # Newest wins, matching _find_fringe_tracking_common_split_nm.
    for name, component in reversed(tuple(state.opd_budget.components.items())):
        if name.startswith(ft_prefix):
            rms_nm = getattr(component, "rms_opd_nm", None)
            if rms_nm is not None and float(rms_nm) > 0.0:
                ft_piston_nm = float(rms_nm)
                break

    if ft_piston_nm > 0.0:
        return ft_piston_nm * 1e-9  # nm -> m, servo-corrected piston
    if ao_piston_nm > 0.0:
        return ao_piston_nm * 1e-9  # nm -> m, raw AO piston (no FT)
    return 0.0


def _find_fringe_tracking_common_split_nm(
    state: SimulationState,
) -> tuple[float, float] | None:
    """Return ``(iid_per_mode_nm, common_differential_nm)`` from the FT stage.

    The fringe tracker (``fringe_tracking.closed_loop``) publishes a per-mode
    IID residual and a common differential (inter-telescope piston + vibration)
    residual. The differential term does not average down across modes, so the
    realized-null MC must drive it through a rank-1 covariance rather than the
    scalar per-sub piston. Returns ``None`` when no FT results component carries
    the split.
    """

    for name, payload in reversed(tuple(state.results.components.items())):
        if not name.startswith("fringe_tracking.closed_loop"):
            continue
        if not isinstance(payload, dict):
            continue
        common_raw = payload.get("residual_common_differential_nm")
        iid_raw = payload.get("residual_iid_per_mode_nm")
        if common_raw is None or iid_raw is None:
            continue
        try:
            common_nm = float(common_raw)
            iid_nm = float(iid_raw)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(common_nm) and math.isfinite(iid_nm)):
            continue
        return iid_nm, common_nm
    return None


def _telescope_signs_for_modes(
    state: SimulationState,
    n_physical_modes: int,
) -> NDArray[np.float64]:
    """Build a +/-1 telescope-sign vector across the physical chip modes.

    Modes are ordered as contiguous equal blocks of ``n_modes / n_apertures``
    per aperture (the convention used by ``_split_apertures_to_modes`` and the
    fringe-observability covariance path). The two telescopes of a differential
    common mode are the first and second halves of the aperture list; apertures
    are mapped to ``+1`` / ``-1`` by their position relative to the midpoint.
    """

    n_apertures = int(state.geometry.n_apertures or 1)
    if n_apertures < 2:
        raise ValueError(
            "sensitivity.detection_curve: inter-telescope common-mode "
            "covariance requires at least two apertures; geometry.n_apertures "
            f"= {state.geometry.n_apertures!r}."
        )
    if n_physical_modes % n_apertures != 0:
        raise ValueError(
            "sensitivity.detection_curve: physical mode count "
            f"({n_physical_modes}) must be a multiple of n_apertures "
            f"({n_apertures}) for the common-mode sign vector."
        )
    modes_per_aperture = n_physical_modes // n_apertures
    midpoint = n_apertures / 2.0
    signs = np.empty(n_physical_modes, dtype=np.float64)
    for aperture_index in range(n_apertures):
        sign = 1.0 if aperture_index < midpoint else -1.0
        start = aperture_index * modes_per_aperture
        signs[start : start + modes_per_aperture] = sign
    return signs


def _inter_telescope_common_mode_covariance_m2(
    *,
    state: SimulationState,
    n_chip_modes: int,
    sigma_iid_m: float,
    sigma_common_m: float,
    stage_name: str,
) -> NDArray[np.float64] | None:
    """Build the rank-1 inter-telescope piston covariance for the realized null.

    The covariance is sized to the *physical* modes (pre-split) so the MC's
    splitter handling — driven by ``splitter_physical_indices`` at the call
    site, the same ``mode_group_indices`` the fringe-observability covariance
    path uses — expands it onto the duplicated chip inputs. Returns ``None``
    when the chip has a single physical mode (no inter-telescope baseline).
    """

    split = splitter_physical_indices(state, int(n_chip_modes))
    if split is None:
        n_physical_modes = int(n_chip_modes)
    else:
        mode_group_indices, _source = split
        n_physical_modes = int(mode_group_indices.max()) + 1
    if n_physical_modes < 2:
        return None
    signs = _telescope_signs_for_modes(state, n_physical_modes)
    covariance = piston_covariance_with_common_mode_m2(
        sigma_iid_m=float(sigma_iid_m),
        sigma_diff_m=float(sigma_common_m),
        telescope_signs=signs,
    )
    if covariance.shape != (n_physical_modes, n_physical_modes):
        raise ValueError(
            f"{stage_name}: inter-telescope covariance shape "
            f"{covariance.shape} does not match {n_physical_modes} physical "
            "modes."
        )
    return covariance


def _find_raw_ao_piston_rms_per_sub_m(state: SimulationState) -> float:
    """Return upstream AO residual piston before fringe-tracker replacement."""

    ao_component = state.wavefront.components.get("ao_residual")
    if not isinstance(ao_component, dict):
        return 0.0
    piston_opd_nm = ao_component.get("piston_opd_nm", {})
    if not isinstance(piston_opd_nm, dict):
        return 0.0
    total = piston_opd_nm.get("total")
    if total is None:
        return 0.0
    arr = np.atleast_1d(np.asarray(total, dtype=np.float64))
    finite = arr[np.isfinite(arr) & (arr > 0.0)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(finite)) * 1.0e-9


def _find_ao_piston_subaperture_diameter_m(
    state: SimulationState,
) -> tuple[float | None, str | None]:
    """Return the AO-published physical piston sub-aperture diameter."""

    ao_component = state.wavefront.components.get("ao_residual")
    if not isinstance(ao_component, dict):
        return None, None
    raw = ao_component.get("piston_subaperture_diameter_m")
    if raw is None:
        return None, None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None, None
    if not math.isfinite(value) or value <= 0.0:
        return None, None
    return value, "wavefront.ao_residual.piston_subaperture_diameter_m"


def _piston_temporal_subaperture_diameter_m(
    *,
    state: SimulationState,
    fallback_subaperture_diameter_m: float,
) -> tuple[float, str]:
    """Resolve the diameter used by the piston temporal-correlation model."""

    ao_value, ao_source = _find_ao_piston_subaperture_diameter_m(state)
    if ao_value is not None and ao_source is not None:
        return ao_value, ao_source
    return (
        float(fallback_subaperture_diameter_m),
        "sensitivity.geometry.mode_diameter_m",
    )


def _resolve_piston_rms_per_sub_m(
    *,
    state: SimulationState,
    params: Any,
    stage_name: str,
    n_chip_modes: int,
    subaperture_diameter_m: float | None = None,
) -> tuple[float, NDArray[np.float64] | None, dict[str, Any]]:
    """Resolve the piston residual model used by the realized-null Monte Carlo."""

    closed_loop_m = _find_ao_piston_rms_per_sub_m(state)
    source = str(params.ao_piston_residual_source)
    ao_piston_scale = float(params.ao_piston_scale)
    diagnostics: dict[str, Any] = {
        "source": source,
        "closed_loop_opd_budget_rms_m": float(closed_loop_m),
        "ao_piston_scale": ao_piston_scale,
    }
    if source == "opd_budget":
        split = _find_fringe_tracking_common_split_nm(state)
        if split is not None and split[1] > 0.0:
            iid_nm, common_nm = split
            sigma_iid_m = iid_nm * 1.0e-9 * ao_piston_scale
            sigma_common_m = common_nm * 1.0e-9 * ao_piston_scale
            covariance = _inter_telescope_common_mode_covariance_m2(
                state=state,
                n_chip_modes=int(n_chip_modes),
                sigma_iid_m=sigma_iid_m,
                sigma_common_m=sigma_common_m,
                stage_name=stage_name,
            )
            if covariance is not None:
                resolved_rms_m = _piston_covariance_rms_nm(covariance) * 1.0e-9
                diagnostics.update(
                    {
                        "opd_budget_iid_per_mode_rms_m": float(sigma_iid_m),
                        "opd_budget_common_differential_rms_m": float(
                            sigma_common_m
                        ),
                        "opd_budget_covariance_resolved_rms_m": float(
                            resolved_rms_m
                        ),
                        "opd_budget_common_mode_model": (
                            "Inter-telescope piston + vibration residual is a "
                            "rank-1 common differential mode that does not "
                            "average down across modes; routed through the "
                            "realized-null covariance MC. The per-mode IID AO "
                            "residual is the covariance diagonal."
                        ),
                    }
                )
                return float(resolved_rms_m), covariance, diagnostics
        return closed_loop_m * ao_piston_scale, None, diagnostics
    if source != "fringe_observability":
        raise ValueError(
            f"{stage_name}: unknown ao_piston_residual_source={source!r}."
        )

    component_name, payload = _find_fringe_observability_payload(state, stage_name)
    raw_nm, metric_label = _fringe_observability_piston_metric_nm(
        payload,
        metric=str(params.fringe_observability_piston_metric),
        stage_name=stage_name,
    )
    predictive_gain = float(params.fringe_observability_predictive_gain_variance)
    floor_nm = float(params.fringe_observability_piston_floor_nm)
    filtered_nm = raw_nm / math.sqrt(predictive_gain)
    resolved_nm = max(filtered_nm, floor_nm)
    raw_ao_m = _find_raw_ao_piston_rms_per_sub_m(state)
    covariance_m2, covariance_diagnostics = _fringe_observability_piston_covariance_m2(
        payload,
        mode=str(params.fringe_observability_piston_covariance),
        predictive_model=str(params.fringe_observability_predictive_model),
        predictive_gain_variance=predictive_gain,
        floor_nm=floor_nm,
        ao_piston_scale=float(params.ao_piston_scale),
        raw_ao_piston_rms_m=raw_ao_m,
        closed_loop_piston_rms_m=closed_loop_m,
        temporal_prior_rms_nm=params.fringe_observability_temporal_prior_rms_nm,
        temporal_residual_kind=str(
            params.fringe_observability_temporal_residual_kind
        ),
        temporal_wind_speed_m_s=float(
            params.fringe_observability_temporal_wind_speed_m_s
        ),
        temporal_sample_time_s=params.fringe_observability_temporal_sample_time_s,
        subaperture_diameter_m=subaperture_diameter_m,
        stage_name=stage_name,
    )
    if covariance_m2 is not None:
        resolved_nm = _piston_covariance_rms_nm(covariance_m2)
    diagnostics.update(
        {
            "fringe_observability_component": component_name,
            "fringe_observability_metric": metric_label,
            "fringe_observability_raw_rms_nm": float(raw_nm),
            "fringe_observability_predictive_gain_variance": predictive_gain,
            "fringe_observability_predictive_scalar_model": (
                "RMS is divided by sqrt(gain) to model a predictive/Kalman "
                "servo that averages temporally correlated piston estimates. "
                "gain=1 uses the instantaneous diagnostic residual."
            ),
            "fringe_observability_filtered_rms_nm": float(filtered_nm),
            "fringe_observability_floor_nm": floor_nm,
            "fringe_observability_resolved_rms_nm": float(resolved_nm),
            **covariance_diagnostics,
        }
    )
    if covariance_m2 is not None:
        return resolved_nm * 1.0e-9, covariance_m2, diagnostics
    return resolved_nm * 1.0e-9 * float(params.ao_piston_scale), None, diagnostics


def _find_fringe_observability_payload(
    state: SimulationState,
    stage_name: str,
) -> tuple[str, dict[str, Any]]:
    candidates: list[tuple[str, dict[str, Any]]] = []
    for name, payload in state.results.components.items():
        if not isinstance(payload, dict):
            continue
        if not name.startswith(_FRINGE_OBSERVABILITY_PREFIXES):
            continue
        if "n_differential_piston_modes" not in payload:
            continue
        candidates.append((name, payload))
    if not candidates:
        raise ValueError(
            f"{stage_name}: ao_piston_residual_source='fringe_observability' "
            "requires an upstream diagnostics.fringe_observability stage. "
            "Insert a diagnostic before sensitivity.detection_curve or use "
            "ao_piston_residual_source='opd_budget'."
        )
    return candidates[-1]


def _fringe_observability_piston_metric_nm(
    payload: dict[str, Any],
    *,
    metric: str,
    stage_name: str,
) -> tuple[float, str]:
    if metric == "auto":
        paths = _FRINGE_OBSERVABILITY_AUTO_PISTON_METRICS
    else:
        try:
            paths = _FRINGE_OBSERVABILITY_PISTON_METRICS[metric]
        except KeyError as exc:
            valid = tuple(sorted(_FRINGE_OBSERVABILITY_PISTON_METRICS)) + ("auto",)
            raise ValueError(
                f"{stage_name}: unknown fringe_observability_piston_metric="
                f"{metric!r}; expected one of {valid}."
            ) from exc
    for path in paths:
        value = _nested_payload_value(payload, path)
        if value is None:
            continue
        scalar = _positive_finite_float_or_none(value)
        if scalar is not None:
            return scalar, ".".join(path)
    raise ValueError(
        f"{stage_name}: fringe_observability payload does not contain a finite "
        f"positive piston metric for {metric!r}."
    )


def _nested_payload_value(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _positive_finite_float_or_none(value: Any) -> float | None:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            return None
        value = value.item()
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(scalar) or scalar <= 0.0:
        return None
    return scalar


def _fringe_observability_piston_covariance_m2(
    payload: dict[str, Any],
    *,
    mode: str,
    predictive_model: str,
    predictive_gain_variance: float,
    floor_nm: float,
    ao_piston_scale: float,
    raw_ao_piston_rms_m: float,
    closed_loop_piston_rms_m: float,
    temporal_prior_rms_nm: float | None,
    temporal_residual_kind: str,
    temporal_wind_speed_m_s: float,
    temporal_sample_time_s: float | None,
    subaperture_diameter_m: float | None,
    stage_name: str,
) -> tuple[NDArray[np.float64] | None, dict[str, Any]]:
    if mode == "disabled":
        return None, {"fringe_observability_piston_covariance": "disabled"}
    if mode not in {"auto", "crlb"}:
        raise ValueError(
            f"{stage_name}: unknown fringe_observability_piston_covariance="
            f"{mode!r}; expected 'auto', 'crlb', or 'disabled'."
        )
    finite = bool(payload.get("piston_crlb_covariance_finite", False))
    raw = payload.get("piston_crlb_covariance_m2")
    if raw is None or not finite:
        if mode == "crlb":
            raise ValueError(
                f"{stage_name}: fringe_observability_piston_covariance='crlb' "
                "requires a full-rank upstream piston_crlb_covariance_m2."
            )
        return None, {
            "fringe_observability_piston_covariance": "unavailable_scalar_fallback",
        }

    covariance = np.asarray(raw, dtype=np.float64)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(
            f"{stage_name}: piston_crlb_covariance_m2 must be a square matrix; "
            f"got {covariance.shape}."
        )
    if not np.all(np.isfinite(covariance)):
        raise ValueError(f"{stage_name}: piston_crlb_covariance_m2 must be finite.")
    if not np.allclose(covariance, covariance.T, rtol=1.0e-10, atol=1.0e-30):
        raise ValueError(f"{stage_name}: piston_crlb_covariance_m2 must be symmetric.")
    covariance = 0.5 * (covariance + covariance.T)

    raw_rms_nm = _piston_covariance_rms_nm(covariance)
    temporal_diagnostics: dict[str, Any] = {}
    if predictive_model == "scalar_gain":
        filtered = covariance / float(predictive_gain_variance)
        magnitude_model = (
            "full-rank bright-port CRLB covariance divided by predictive_gain, "
            "optionally rescaled to the RMS floor, then multiplied by "
            "ao_piston_scale^2; temporal lag, vibration, and estimator "
            "inefficiency are not included unless represented by the floor "
            "or scale"
        )
    elif predictive_model == "temporal_kalman":
        filtered, temporal_diagnostics = _temporal_kalman_covariance_m2(
            payload=payload,
            covariance_m2=covariance,
            raw_ao_piston_rms_m=float(raw_ao_piston_rms_m),
            closed_loop_piston_rms_m=float(closed_loop_piston_rms_m),
            temporal_prior_rms_nm=temporal_prior_rms_nm,
            temporal_residual_kind=temporal_residual_kind,
            temporal_wind_speed_m_s=temporal_wind_speed_m_s,
            temporal_sample_time_s=temporal_sample_time_s,
            subaperture_diameter_m=subaperture_diameter_m,
            stage_name=stage_name,
        )
        magnitude_model = (
            "full-rank bright-port CRLB covariance passed through an AR(1) "
            "temporal Kalman surrogate using a Conan/Di Folco piston-PSD "
            "correlation frequency, optionally rescaled to the RMS floor, "
            "then multiplied by ao_piston_scale^2"
        )
    else:
        raise ValueError(
            f"{stage_name}: unknown fringe_observability_predictive_model="
            f"{predictive_model!r}; expected 'scalar_gain' or 'temporal_kalman'."
        )
    filtered_rms_nm = _piston_covariance_rms_nm(filtered)
    floor_scale = 1.0
    if floor_nm > 0.0 and filtered_rms_nm < float(floor_nm):
        if filtered_rms_nm <= 0.0:
            filtered = np.eye(covariance.shape[0], dtype=np.float64) * (
                float(floor_nm) * 1.0e-9
            ) ** 2
        else:
            floor_scale = (float(floor_nm) / filtered_rms_nm) ** 2
            filtered = filtered * floor_scale
        filtered_rms_nm = _piston_covariance_rms_nm(filtered)

    resolved = filtered * (float(ao_piston_scale) ** 2)
    resolved_rms_nm = _piston_covariance_rms_nm(resolved)
    return resolved.astype(np.float64, copy=True), {
        "fringe_observability_piston_covariance": "crlb",
        "fringe_observability_covariance_raw_rms_nm": float(raw_rms_nm),
        "fringe_observability_covariance_filtered_rms_nm": float(filtered_rms_nm),
        "fringe_observability_covariance_floor_scale": float(floor_scale),
        "fringe_observability_covariance_resolved_rms_nm": float(resolved_rms_nm),
        "fringe_observability_predictive_model": predictive_model,
        "fringe_observability_covariance_magnitude_model": magnitude_model,
        **temporal_diagnostics,
    }


def _piston_covariance_rms_nm(covariance_m2: NDArray[np.float64]) -> float:
    covariance = np.asarray(covariance_m2, dtype=np.float64)
    trace = float(np.trace(covariance))
    return float(math.sqrt(max(trace, 0.0) / float(covariance.shape[0])) * 1.0e9)


def _temporal_kalman_covariance_m2(
    *,
    payload: dict[str, Any],
    covariance_m2: NDArray[np.float64],
    raw_ao_piston_rms_m: float,
    closed_loop_piston_rms_m: float,
    temporal_prior_rms_nm: float | None,
    temporal_residual_kind: str,
    temporal_wind_speed_m_s: float,
    temporal_sample_time_s: float | None,
    subaperture_diameter_m: float | None,
    stage_name: str,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    positions = _fringe_observability_positions_m(payload, stage_name=stage_name)
    sample_time_s = _fringe_observability_sample_time_s(
        payload,
        temporal_sample_time_s=temporal_sample_time_s,
        stage_name=stage_name,
    )
    diameter_m = _fringe_observability_subaperture_diameter_m(
        subaperture_diameter_m,
        stage_name=stage_name,
    )
    prior_m, prior_source = _temporal_prior_rms_m(
        temporal_prior_rms_nm=temporal_prior_rms_nm,
        raw_ao_piston_rms_m=raw_ao_piston_rms_m,
        closed_loop_piston_rms_m=closed_loop_piston_rms_m,
        covariance_m2=covariance_m2,
    )
    result = temporal_kalman_piston_covariance(
        measurement_covariance_m2=covariance_m2,
        positions_m=positions,
        prior_rms_m=prior_m,
        sample_time_s=sample_time_s,
        wind_speed_m_s=float(temporal_wind_speed_m_s),
        subaperture_diameter_m=diameter_m,
        residual_kind=temporal_residual_kind,
    )
    selected_rms_nm = _piston_covariance_rms_nm(result.selected_covariance_m2)
    return result.selected_covariance_m2, {
        "fringe_observability_temporal_prior_rms_nm": float(prior_m * 1.0e9),
        "fringe_observability_temporal_prior_source": prior_source,
        "fringe_observability_temporal_sample_time_s": float(sample_time_s),
        "fringe_observability_temporal_wind_speed_m_s": float(
            temporal_wind_speed_m_s
        ),
        "fringe_observability_temporal_subaperture_diameter_m": float(diameter_m),
        "fringe_observability_temporal_residual_kind": result.residual_kind,
        "fringe_observability_temporal_selected_rms_nm": float(selected_rms_nm),
        "fringe_observability_temporal_mode_baseline_m": (
            result.mode_baseline_m.astype(np.float64, copy=True)
        ),
        "fringe_observability_temporal_mode_correlation_hz": (
            result.mode_correlation_hz.astype(np.float64, copy=True)
        ),
        "fringe_observability_temporal_mode_ar1_rho": (
            result.mode_ar1_rho.astype(np.float64, copy=True)
        ),
        "fringe_observability_temporal_mode_measurement_std_nm": (
            np.sqrt(result.mode_measurement_variance_m2) * 1.0e9
        ).astype(np.float64, copy=True),
        "fringe_observability_temporal_mode_posterior_std_nm": (
            np.sqrt(result.mode_posterior_variance_m2) * 1.0e9
        ).astype(np.float64, copy=True),
        "fringe_observability_temporal_mode_prediction_std_nm": (
            np.sqrt(result.mode_prediction_variance_m2) * 1.0e9
        ).astype(np.float64, copy=True),
        "fringe_observability_temporal_model_note": result.model_note,
        "fringe_observability_temporal_literature": (
            "Conan, Rousset & Madec 1995 JOSA A 12, 1559; "
            "Di Folco et al. 2002 SPIE 4838; "
            "Menu et al. 2012 A&A 541, A81; "
            "Lacour et al. 2019 A&A 624, A99"
        ),
    }


def _fringe_observability_positions_m(
    payload: dict[str, Any],
    *,
    stage_name: str,
) -> NDArray[np.float64]:
    raw = payload.get("positions_pupil_m")
    if raw is None:
        raise ValueError(
            f"{stage_name}: fringe_observability_predictive_model="
            "'temporal_kalman' requires positions_pupil_m in the upstream "
            "diagnostics.fringe_observability payload."
        )
    positions = np.asarray(raw, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] not in {2, 3}:
        raise ValueError(
            f"{stage_name}: positions_pupil_m must have shape (n_modes, 2) "
            f"or (n_modes, 3); got {positions.shape}."
        )
    if not np.all(np.isfinite(positions)):
        raise ValueError(f"{stage_name}: positions_pupil_m must be finite.")
    return positions.astype(np.float64, copy=True)


def _fringe_observability_sample_time_s(
    payload: dict[str, Any],
    *,
    temporal_sample_time_s: float | None,
    stage_name: str,
) -> float:
    if temporal_sample_time_s is not None:
        return float(temporal_sample_time_s)
    raw = payload.get("frame_time_s")
    if raw is None:
        raise ValueError(
            f"{stage_name}: temporal_kalman piston covariance requires "
            "fringe_observability frame_time_s or "
            "fringe_observability_temporal_sample_time_s."
        )
    value = float(raw)
    if value <= 0.0 or not math.isfinite(value):
        raise ValueError(f"{stage_name}: fringe_observability frame_time_s invalid.")
    return value


def _fringe_observability_subaperture_diameter_m(
    subaperture_diameter_m: float | None,
    *,
    stage_name: str,
) -> float:
    if subaperture_diameter_m is None:
        raise ValueError(
            f"{stage_name}: temporal_kalman piston covariance requires "
            "subaperture_diameter_m from the sensitivity geometry path."
        )
    value = float(subaperture_diameter_m)
    if value <= 0.0 or not math.isfinite(value):
        raise ValueError(f"{stage_name}: subaperture_diameter_m must be positive.")
    return value


def _temporal_prior_rms_m(
    *,
    temporal_prior_rms_nm: float | None,
    raw_ao_piston_rms_m: float,
    closed_loop_piston_rms_m: float,
    covariance_m2: NDArray[np.float64],
) -> tuple[float, str]:
    if temporal_prior_rms_nm is not None:
        return float(temporal_prior_rms_nm) * 1.0e-9, (
            "params.fringe_observability_temporal_prior_rms_nm"
        )
    if raw_ao_piston_rms_m > 0.0:
        return float(raw_ao_piston_rms_m), "wavefront.ao_residual.piston_opd_nm.total"
    if closed_loop_piston_rms_m > 0.0:
        return float(closed_loop_piston_rms_m), (
            "opd_budget.fringe_tracking.closed_loop_residual"
        )
    return _piston_covariance_rms_nm(covariance_m2) * 1.0e-9, (
        "fringe_observability.piston_crlb_covariance_rms"
    )


def _resolve_scintillation_index(
    *,
    state: SimulationState,
    params: Any,
    wavelength_m: float,
    sub_diameter_m: float,
) -> tuple[float, dict[str, Any] | None]:
    """Resolve σ_I per sub-aperture for the realized-null MC.

    Returns ``(sigma_I, diagnostics)``. When
    ``params.auto_amplitude_servo_residual`` is False, starts from the manual
    ``params.scintillation_index_per_sub`` scalar. When True, derives σ_I from
    :func:`nullsim.physics.scintillation.amplitude_servo_residual_sigma_i`
    using the ``auto_*`` params. In both cases, any upstream
    ``wavefront.amplitude_jitter_fraction`` is treated as an independent
    fractional-intensity RMS and combined in quadrature so space-residual
    amplitude jitter is not a silent no-op.

    KILO_PARITY_PLAN §2.1: when ``auto_amp_servo_bandwidth_hz`` is
    ``None``, the bandwidth is set per-target by
    :func:`nullsim.physics.scintillation.optimize_amplitude_servo_bandwidth`
    (closed-form photon-limited optimum, ~4 kHz at J=4 / ~380 Hz at
    J=8). KILO_PARITY_PLAN §2.2: when
    ``auto_amp_servo_photon_rate_per_sub_hz`` is ``None``, the per-sub
    WFS photon rate is read from the upstream ``ao_residual`` component
    (``wfs_photons_per_subap_per_frame × loop_frequency_hz``).
    """

    wavefront_sigma_i, wavefront_payload = _wavefront_amplitude_jitter_payload(state)

    if not params.auto_amplitude_servo_residual:
        manual_sigma_i = float(params.scintillation_index_per_sub)
        injected_sigma_i = float(math.hypot(manual_sigma_i, wavefront_sigma_i))
        diagnostics = None
        if wavefront_sigma_i > 0.0:
            diagnostics = {
                "mode": "manual",
                "sigma_I_manual": manual_sigma_i,
                "sigma_I_injected_before_wavefront": manual_sigma_i,
                "sigma_I_injected": injected_sigma_i,
                "sigma_I_injected_source": (
                    "manual_scintillation_plus_wavefront_amplitude_jitter"
                ),
                **wavefront_payload,
            }
        return injected_sigma_i, diagnostics

    photon_rate_per_sub_hz = _resolve_amp_servo_photon_rate_hz(state, params)
    airmass = _resolve_scene_airmass(state)
    servo_bandwidth_hz = _resolve_amp_servo_bandwidth_hz(
        params=params,
        wavelength_m=wavelength_m,
        sub_diameter_m=sub_diameter_m,
        photon_rate_per_sub_hz=photon_rate_per_sub_hz,
        airmass=airmass,
    )

    residual: AmplitudeServoResidual = amplitude_servo_residual_sigma_i(
        wavelength_m=wavelength_m,
        sub_diameter_m=sub_diameter_m,
        servo_bandwidth_hz=servo_bandwidth_hz,
        photon_rate_per_sub_hz=photon_rate_per_sub_hz,
        scintillation_index_h5m_zenith=float(
            params.auto_scintillation_anchor_h5m_zenith
        ),
        scintillation_height_km=float(params.auto_scintillation_height_km),
        airmass=airmass,
        predictive_gain_variance=float(params.auto_amp_predictive_gain_variance),
        high_altitude_wind_ms=float(params.auto_high_altitude_wind_ms),
    )
    open_loop_bandwidth_hz = 0.5 / float(residual.tau_scint_s)
    mode = str(getattr(params, "auto_amplitude_servo_mode", "closed_loop"))
    # amplitude_servo_residual_sigma_i caps the closed-loop residual at the
    # open-loop scintillation (the servo can always do nothing). A capped
    # residual is open-loop scintillation, so it carries the open-loop
    # bandwidth 0.5/tau_scint, not the servo bandwidth.
    capped = residual.sigma_I_residual >= residual.sigma_I_open
    if mode == "closed_loop" and not capped:
        injected_sigma_I = float(residual.sigma_I_residual)
        injected_source = "closed_loop_residual"
        injected_bandwidth_hz = float(residual.servo_bandwidth_hz)
    elif mode == "closed_loop":
        injected_sigma_I = float(residual.sigma_I_open)
        injected_source = "closed_loop_capped_at_open_loop"
        injected_bandwidth_hz = float(open_loop_bandwidth_hz)
    elif mode == "open_loop":
        injected_sigma_I = float(residual.sigma_I_open)
        injected_source = "open_loop_scintillation"
        injected_bandwidth_hz = float(open_loop_bandwidth_hz)
    else:
        raise ValueError(
            "sensitivity.detection_curve: auto_amplitude_servo_mode must be "
            "'closed_loop' or 'open_loop'."
        )

    combined_sigma_I = float(math.hypot(injected_sigma_I, wavefront_sigma_i))
    combined_source = (
        injected_source
        if wavefront_sigma_i == 0.0
        else f"{injected_source}_plus_wavefront_amplitude_jitter"
    )
    diagnostics = {
        "sigma_I_residual": float(residual.sigma_I_residual),
        "sigma_I_open": float(residual.sigma_I_open),
        "sigma_I_injected_before_wavefront": float(injected_sigma_I),
        "sigma_I_injected": combined_sigma_I,
        "sigma_I_injected_source": combined_source,
        "sigma_I_temporal": float(residual.sigma_I_temporal),
        "sigma_I_photon": float(residual.sigma_I_photon),
        "tau_loop_s": float(residual.tau_loop_s),
        "tau_scint_s": float(residual.tau_scint_s),
        "open_loop_scintillation_bandwidth_hz": float(open_loop_bandwidth_hz),
        "injected_amplitude_bandwidth_hz": float(injected_bandwidth_hz),
        "n_photons_per_sample": float(residual.n_photons_per_sample),
        "predictive_gain_variance": float(residual.predictive_gain_variance),
        "servo_bandwidth_hz": float(residual.servo_bandwidth_hz),
        "photon_rate_per_sub_hz": float(photon_rate_per_sub_hz),
        "airmass": float(airmass),
        "mode": mode,
        **wavefront_payload,
    }
    return combined_sigma_I, diagnostics


def _resolve_scene_airmass(state: SimulationState) -> float:
    """Return the scene airmass used by scintillation calculations."""

    altitude_deg = state.scene.altitude_deg
    if altitude_deg is None:
        return 1.0
    altitude = float(altitude_deg)
    if not math.isfinite(altitude) or altitude <= 0.0 or altitude > 90.0:
        raise ValueError(
            "sensitivity.detection_curve could not derive airmass from "
            f"scene.altitude_deg={altitude_deg!r}; expected 0 < altitude_deg <= 90."
        )
    return max(1.0, 1.0 / math.sin(math.radians(altitude)))


def _wavefront_amplitude_jitter_payload(
    state: SimulationState,
) -> tuple[float, dict[str, Any]]:
    """Return scalarized upstream wavefront amplitude jitter diagnostics."""

    raw = state.wavefront.amplitude_jitter_fraction
    if len(raw) == 0:
        return 0.0, {
            "wavefront_amplitude_jitter_fraction_rms": 0.0,
            "wavefront_amplitude_jitter_fraction_max": 0.0,
            "wavefront_amplitude_jitter_model": "none",
        }
    jitter = np.asarray(raw, dtype=np.float64)
    if jitter.ndim != 1:
        raise ValueError(
            "sensitivity.detection_curve: wavefront.amplitude_jitter_fraction "
            f"must be 1-D; got shape {jitter.shape}."
        )
    if not np.all(np.isfinite(jitter)):
        raise ValueError(
            "sensitivity.detection_curve: wavefront.amplitude_jitter_fraction "
            "must be finite."
        )
    if np.any(jitter < 0.0):
        raise ValueError(
            "sensitivity.detection_curve: wavefront.amplitude_jitter_fraction "
            "must be non-negative."
        )
    rms = float(np.sqrt(np.mean(jitter * jitter))) if jitter.size else 0.0
    return rms, {
        "wavefront_amplitude_jitter_fraction_rms": rms,
        "wavefront_amplitude_jitter_fraction_max": (
            float(np.max(jitter)) if jitter.size else 0.0
        ),
        "wavefront_amplitude_jitter_fraction_by_wavelength": tuple(
            float(value) for value in jitter
        ),
        "wavefront_amplitude_jitter_model": (
            "band-RMS scalarized independent fractional-intensity jitter; "
            "combined in quadrature with scintillation_index_per_sub before "
            "realized_null_depth_mc"
            if rms > 0.0
            else "none"
        ),
    }


def _central_obstruction_ratio(state: SimulationState, default: float = 0.0) -> float:
    """Resolve the linear central-obstruction ratio epsilon = d_obs / D.

    Phase 4 (KILO unifies the obscuration through the stellar + planet
    coupling paths): the canonical source is the upstream telescope
    stage's diagnostic component (e.g. ``telescope.fixed_aperture``),
    which publishes ``central_obstruction_ratio``. Falls back to
    ``default`` when no telescope diagnostic is present (atmosphere-only
    smoke tests).
    """

    for payload in reversed(tuple(state.geometry.components.values())):
        if not isinstance(payload, dict):
            continue
        value = payload.get("central_obstruction_ratio")
        if value is None:
            continue
        try:
            ratio = float(value)
        except (TypeError, ValueError):
            continue
        if 0.0 <= ratio < 1.0:
            return ratio
    return float(default)


def _coupling_per_sub_from_state(
    state: SimulationState,
    params: Any,
    n_modes: int,
    obstruction_ratio: float,
) -> tuple[NDArray[np.float64], str]:
    """Return the deployed per-subaperture injection coupling.

    The injection stages carry SMF coupling in field amplitudes rather
    than top-level throughput so detectors do not double-count it. The
    sensitivity stage, however, reconstructs its own stellar input field
    for the AO-realized-null calculation, so it must read the
    injection-stage diagnostic directly when one exists.
    """

    upstream = _find_injection_coupling(state)
    if upstream is not None:
        coupling, source = upstream
        return np.full(n_modes, coupling, dtype=np.float64), source

    if params.apply_strehl_coupling:
        mean_strehl = _band_mean_strehl(state)
        eta_geom = float(
            coupling_efficiency(
                mean_strehl,
                obstruction_ratio,
                apply_fresnel_loss=params.apply_fresnel_injection_loss,
            )
        )
        source = "wavefront.strehl_ratio+geometry.central_obstruction_ratio"
    elif params.apply_fresnel_injection_loss:
        eta_geom = 1.0 - 0.035
        source = "params.apply_fresnel_injection_loss"
    else:
        eta_geom = 1.0
        source = "unity"
    return np.full(n_modes, eta_geom, dtype=np.float64), source


def _find_injection_coupling(state: SimulationState) -> tuple[float, str] | None:
    """Find a band-mean injection coupling published by upstream stages."""

    budgets = (
        ("throughput", state.throughput.components),
        ("wavefront", state.wavefront.components),
    )
    for budget_name, components in budgets:
        for component_name, payload in reversed(tuple(components.items())):
            if not component_name.startswith(_INJECTION_COMPONENT_PREFIXES):
                continue
            coupling = _coupling_from_component_payload(payload)
            if coupling is not None:
                field_name, value = coupling
                return value, f"{budget_name}:{component_name}.{field_name}"
    return None


def _coupling_from_component_payload(payload: Any) -> tuple[str, float] | None:
    if not isinstance(payload, dict):
        return None
    for field_name in ("coupling_efficiency", "transmission_fraction"):
        if field_name not in payload:
            continue
        raw_values = payload[field_name]
        if raw_values is None:
            continue
        try:
            values = np.asarray(raw_values, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            continue
        finite = values[np.isfinite(values) & (values >= 0.0)]
        if finite.size:
            return field_name, float(np.mean(finite))
    return None


def _chip_input_stellar_rate_hz(state: SimulationState, params: Any) -> float:
    """Total chip-input stellar photon rate in counts/s.

    Phase 3 (KILO ``natural_design.py`` L797): the photon budget feeding
    the amplitude WFS is the science flux at the chip input,

        ref_rate_at_chip = stellar_photon_rate(jmag, lambda, dlambda, area)
                            * throughput_to_chip

    summed over the wavelength grid. nullsim assembles this from
    ``state.scene.star_jmag``, ``state.field.wavelength_grid_um`` (with
    Delta-lambda per bin), ``state.geometry.area_m2`` or
    ``geometry.effective_collecting_area_m2``, and
    ``state.throughput.transmission_fraction`` (per-wavelength upstream
    throughput).
    """

    from nullsim.physics.photometry import photon_rate_per_m2_per_bin

    if state.scene.star_jmag is None:
        raise ValueError(
            "sensitivity.detection_curve (chip-input amp-servo rate): "
            "scene.star_jmag must be populated by a scene stage."
        )
    wavelength_grid_um = state.field.wavelength_grid_um
    if not wavelength_grid_um:
        raise ValueError(
            "sensitivity.detection_curve (chip-input amp-servo rate): "
            "field.wavelength_grid_um is empty."
        )
    area_m2 = (
        state.geometry.effective_collecting_area_m2
        if state.geometry.effective_collecting_area_m2 is not None
        else state.geometry.area_m2
    )
    if area_m2 is None or float(area_m2) <= 0.0:
        raise ValueError(
            "sensitivity.detection_curve (chip-input amp-servo rate): "
            "geometry.area_m2 (or effective_collecting_area_m2) must be "
            "populated by a telescope stage."
        )

    wl = np.asarray(wavelength_grid_um, dtype=np.float64)
    if wl.size == 1:
        # Mirror the MKID/SNSPD single-bin convention (mkid.py:275,
        # snspd.py:214): use ``wl[0]`` itself as a 100% fractional-
        # bandwidth stand-in. The previous zero-bandwidth path drove the
        # chip-input rate to 0 and then raised in
        # ``_resolve_amp_servo_photon_rate_hz``, hard-failing degenerate
        # single-wavelength sanity configs where the detectors would
        # have run cleanly.
        deltas = np.array([wl[0]], dtype=np.float64)
    else:
        diffs = np.diff(wl)
        deltas = np.concatenate(
            [[diffs[0]], (diffs[:-1] + diffs[1:]) / 2.0, [diffs[-1]]]
        )

    transmission_arr = _throughput_before_detector_qe(state, wl.size)
    field_loss_arr = _amp_servo_field_applied_transmission(state, wl.size)

    current_band = infer_band_from_wavelength_um(float(wl[len(wl) // 2]))
    rate_per_m2 = np.array(
        [
            photon_rate_per_m2_per_bin(
                float(state.scene.star_jmag),
                float(w),
                float(b),
                band=current_band,
            )
            for w, b in zip(wl, deltas)
        ],
        dtype=np.float64,
    )
    chip_input_per_m2_hz = float(
        np.sum(rate_per_m2 * transmission_arr * field_loss_arr)
    )
    scale = _simultaneous_science_band_scale(
        state=state,
        params=params,
        current_wavelength_grid_um=wavelength_grid_um,
    )
    return chip_input_per_m2_hz * float(area_m2) * scale


def _throughput_before_detector_qe(
    state: SimulationState,
    n_wavelength_bins: int,
) -> NDArray[np.float64]:
    if state.throughput.transmission_fraction:
        transmission = np.asarray(
            state.throughput.transmission_fraction,
            dtype=np.float64,
        )
        if transmission.shape != (n_wavelength_bins,):
            raise ValueError(
                "sensitivity.detection_curve (chip-input amp-servo rate): "
                "throughput.transmission_fraction length does not match "
                "field.wavelength_grid_um."
            )
    else:
        transmission = np.ones(n_wavelength_bins, dtype=np.float64)

    for name, payload in state.throughput.components.items():
        if not name.startswith(("detector.mkid_qe", "detector.snspd_qe")):
            continue
        if not isinstance(payload, Mapping):
            continue
        detector_qe = _component_transmission_array(
            payload,
            n_wavelength_bins,
            source=f"sensitivity.detection_curve detector QE component {name!r}",
        )
        if np.any(detector_qe <= 0.0):
            raise ValueError(
                "sensitivity.detection_curve detector QE throughput entries "
                "must be positive when deriving the pre-detector amp-servo rate."
            )
        transmission = transmission / detector_qe
    return transmission


def _amp_servo_field_applied_transmission(
    state: SimulationState,
    n_wavelength_bins: int,
) -> NDArray[np.float64]:
    factor = np.ones(n_wavelength_bins, dtype=np.float64)
    for name, payload in state.throughput.components.items():
        if name.startswith(("transport.photometric_pickoff", "chip.", "detector.")):
            break
        if not isinstance(payload, Mapping) or not _is_field_applied_throughput(payload):
            continue
        factor *= _component_transmission_array(
            payload,
            n_wavelength_bins,
            source=f"sensitivity.detection_curve field-applied component {name!r}",
        )
    return factor


def _is_field_applied_throughput(payload: Mapping[str, Any]) -> bool:
    for key in (
        "field_amplitude_applied",
        "field_applied_loss",
        "per_mode_field_amplitude_applied",
    ):
        if payload.get(key) is True:
            return True
    source_model = str(payload.get("source_model", "")).lower()
    return "applied to field" in source_model or "field amplitudes" in source_model


def _component_transmission_array(
    payload: Mapping[str, Any],
    n_wavelength_bins: int,
    *,
    source: str,
) -> NDArray[np.float64]:
    component_fraction = payload.get("transmission_fraction")
    if component_fraction is None and payload.get("qe") is not None:
        component_fraction = payload.get("qe")
    if component_fraction is None:
        return np.ones(n_wavelength_bins, dtype=np.float64)
    component = np.asarray(component_fraction, dtype=np.float64)
    if component.shape == ():
        component = np.full(n_wavelength_bins, float(component), dtype=np.float64)
    if component.shape != (n_wavelength_bins,):
        raise ValueError(
            f"{source} length {component.size} does not match wavelength grid "
            f"length {n_wavelength_bins}."
        )
    return component


def _simultaneous_science_band_scale(
    *,
    state: SimulationState,
    params: Any,
    current_wavelength_grid_um: tuple[float, ...],
) -> float:
    bands = tuple(getattr(params, "simultaneous_science_bands", ()) or ())
    if not bands:
        return 1.0
    if state.scene.star_jmag is None:
        raise ValueError(
            "sensitivity.detection_curve: simultaneous_science_bands requires "
            "scene.star_jmag so the extra band photon budget can be computed."
        )

    transmissions = tuple(
        getattr(params, "simultaneous_science_band_transmissions", ()) or ()
    )
    scale, _bands, _current_band = simultaneous_band_rate_scale(
        magnitude=float(state.scene.star_jmag),
        current_wavelength_grid_um=current_wavelength_grid_um,
        science_bands=bands,
        band_transmissions=transmissions,
        stage_name="sensitivity.detection_curve",
    )

    return float(scale)


def _find_photometric_pickoff_payload(state: SimulationState) -> dict[str, Any] | None:
    """Return the physical amplitude-tap payload when one is upstream."""

    for name, payload in reversed(tuple(state.throughput.components.items())):
        if not name.startswith("transport.photometric_pickoff"):
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _resolve_amp_servo_n_modes(
    state: SimulationState,
    params: Any,
    *,
    pickoff_payload: dict[str, Any] | None = None,
) -> int:
    """Pick the mode count that divides chip-input rate per amp WFS sub."""

    if pickoff_payload is not None:
        n_input_modes = pickoff_payload.get("n_input_modes")
        if n_input_modes is not None and int(n_input_modes) > 0:
            return int(n_input_modes)
    chip_payload = None
    for name, payload in reversed(state.results.components.items()):
        if name.startswith("chip.kernel_mzi_mesh") and isinstance(payload, dict):
            chip_payload = payload
            break
    if chip_payload is not None:
        n_modes_chip = chip_payload.get("n_modes")
        if n_modes_chip is not None and int(n_modes_chip) > 0:
            return int(n_modes_chip)
    n_bright = int(params.n_bright_ports)
    n_dark = int(params.n_dark_ports)
    n_spectro = int(getattr(params, "n_spectro_ports", 0))
    return n_bright + n_dark + n_spectro


def _resolve_amp_servo_photon_rate_hz(state: SimulationState, params: Any) -> float:
    """Return the per-sub photon rate driving the amplitude servo.

    Phase 3 (KILO ``natural_design.py`` L897): the amplitude WFS sees
    the full chip-input science flux, not the AO WFS photon budget.
    Priority:
      1. Explicit ``params.auto_amp_servo_photon_rate_per_sub_hz`` if
         positive.
      2. A physical ``transport.photometric_pickoff`` tap, when present.
         The tap fraction and pre-split physical mode count come from that
         stage so the amplitude WFS rate is not diluted by duplicated
         quadrature inputs.
      3. Chip-input rate ``ref_rate_at_chip`` (computed by
         :func:`_chip_input_stellar_rate_hz` from scene + geometry +
         throughput) multiplied by ``params.amp_wfs_pickoff_fraction``
         and divided by the chip mode count ``n_modes``.

    The pre-Phase-3 fallback read ``wfs_photons_per_subap_per_frame``
    from the AO ``ao_residual`` component; that is the GUIDE-star
    photon budget at the AO bench, not the science photons hitting the
    chip. Using it underestimates the amplitude-WFS rate by orders of
    magnitude on bright targets.
    """

    explicit = params.auto_amp_servo_photon_rate_per_sub_hz
    if explicit is not None:
        if explicit <= 0:
            raise ValueError(
                "sensitivity.detection_curve: "
                "auto_amp_servo_photon_rate_per_sub_hz must be > 0 "
                f"(got {explicit})."
            )
        return float(explicit)

    pickoff_payload = _find_photometric_pickoff_payload(state)
    pickoff = (
        float(pickoff_payload["tap_fraction"])
        if pickoff_payload is not None and "tap_fraction" in pickoff_payload
        else float(getattr(params, "amp_wfs_pickoff_fraction", 1.0))
    )
    if pickoff <= 0.0 or pickoff > 1.0:
        raise ValueError(
            "sensitivity.detection_curve: amplitude WFS pickoff fraction must "
            f"lie in (0, 1] (got {pickoff})."
        )

    chip_input_hz = _chip_input_stellar_rate_hz(state, params)
    n_modes = _resolve_amp_servo_n_modes(
        state,
        params,
        pickoff_payload=pickoff_payload,
    )
    rate_hz = pickoff * chip_input_hz / max(int(n_modes), 1)
    if rate_hz <= 0.0:
        raise ValueError(
            "sensitivity.detection_curve: chip-input amp-servo rate "
            f"resolved to {rate_hz} (must be > 0). Check scene.star_jmag, "
            "geometry.area_m2, and the upstream throughput chain."
        )
    return rate_hz


def _resolve_amp_servo_bandwidth_hz(
    *,
    params: Any,
    wavelength_m: float,
    sub_diameter_m: float,
    photon_rate_per_sub_hz: float,
    airmass: float,
) -> float:
    """Return the amplitude-servo bandwidth (fixed param or photon-optimum)."""

    explicit = params.auto_amp_servo_bandwidth_hz
    if explicit is not None:
        return float(explicit)
    return optimize_amplitude_servo_bandwidth(
        wavelength_m=wavelength_m,
        sub_diameter_m=sub_diameter_m,
        photon_rate_per_sub_hz=photon_rate_per_sub_hz,
        scintillation_index_h5m_zenith=float(
            params.auto_scintillation_anchor_h5m_zenith
        ),
        scintillation_height_km=float(params.auto_scintillation_height_km),
        airmass=float(airmass),
        predictive_gain_variance=float(params.auto_amp_predictive_gain_variance),
        high_altitude_wind_ms=float(params.auto_high_altitude_wind_ms),
        bandwidth_min_hz=float(params.auto_amp_servo_bandwidth_min_hz),
        bandwidth_max_hz=float(params.auto_amp_servo_bandwidth_max_hz),
    )


def _staged_background_rate_hz_per_dark_port(state: SimulationState) -> float:
    """Latest background.* component's per-dark-port rate, else 0.0."""

    for name in reversed(tuple(state.results.components)):
        if not name.startswith("background."):
            continue
        payload = state.results.components[name]
        if not isinstance(payload, dict):
            continue
        value = payload.get("background_rate_hz_per_dark_port")
        if (
            isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) >= 0.0
        ):
            return float(value)
    return 0.0


def _ft_published_f_3db_hz(state: SimulationState) -> float | None:
    """Return the closed-loop piston bandwidth the FT stage actually ran.

    Mirrors ``performance.standard._resolve_servo_bw``'s component lookup.
    Used as the piston leg of the f_eff variance blend: pre-fix the blend
    used the static ``auto_f_piston_hz`` default (100 Hz) while the real
    photon-Kalman loop runs at kHz-scale f_3db — inflating or deflating
    the systematic-noise term by up to ~10x across the J sweep (review
    2026-06-10 Issue 2).
    """

    # Newest first, like the residual lookup: a later fringe-tracking stage
    # supersedes an earlier one.
    for name, payload in reversed(tuple(state.results.components.items())):
        if not name.startswith("fringe_tracking."):
            continue
        if not isinstance(payload, dict):
            continue
        for key in ("f_3db_hz", "servo_bandwidth_hz"):
            value = payload.get(key)
            if (
                isinstance(value, (int, float))
                and math.isfinite(float(value))
                and float(value) > 0.0
            ):
                return float(value)
    return None


def _resolve_effective_bandwidth_hz(
    *,
    params: Any,
    piston_leakage_variance: float,
    amplitude_leakage_variance: float,
    resolved_amp_servo_bandwidth_hz: float | None = None,
    resolved_f_piston_hz: float | None = None,
) -> float | None:
    """Resolve the f_eff propagated to ``StandardPerformance``.

    Precedence:

    1. ``params.effective_systematic_bandwidth_hz_override`` (explicit).
    2. The leakage-variance-weighted blend
       (:func:`effective_systematic_bandwidth_hz`) when
       ``params.auto_amplitude_servo_residual=True`` (Tier 2.5b).
    3. ``None`` — performance stage uses its own default.

    KILO_PARITY_PLAN §2.1+§2.4: when the amplitude-servo bandwidth was
    photon-optimized in this run (``auto_amp_servo_bandwidth_hz=None``
    → ``optimize_amplitude_servo_bandwidth``), the *resolved* per-cell
    value is passed in via ``resolved_amp_servo_bandwidth_hz`` so the
    variance blend uses the bandwidth that actually drove the injected
    σ_I term rather than the raw param (which would be ``None`` here).
    For open-loop scintillation tests this value is the scintillation
    correlation bandwidth, not a control-loop bandwidth.
    """

    if params.effective_systematic_bandwidth_hz_override is not None:
        return float(params.effective_systematic_bandwidth_hz_override)
    if not params.auto_amplitude_servo_residual:
        return None
    if resolved_amp_servo_bandwidth_hz is not None:
        f_amp_hz = float(resolved_amp_servo_bandwidth_hz)
    elif params.auto_amp_servo_bandwidth_hz is not None:
        f_amp_hz = float(params.auto_amp_servo_bandwidth_hz)
    else:
        raise ValueError(
            "sensitivity.detection_curve: cannot resolve f_eff blend — "
            "auto_amp_servo_bandwidth_hz is None (photon-optimized) but "
            "the resolved bandwidth was not provided. This is a wiring "
            "bug in detection_curve; see KILO_PARITY_PLAN §2.4."
        )
    f_piston_hz = (
        float(resolved_f_piston_hz)
        if resolved_f_piston_hz is not None
        else float(params.auto_f_piston_hz)
    )
    return effective_systematic_bandwidth_hz(
        piston_leakage_variance=piston_leakage_variance,
        amplitude_leakage_variance=amplitude_leakage_variance,
        f_piston_hz=f_piston_hz,
        f_amp_hz=f_amp_hz,
    )


def _effective_bandwidth_diagnostics(
    *,
    params: Any,
    sigma_phi_piston_rad: float,
    sigma_I_residual: float,
    piston_leakage_variance: float,
    amplitude_leakage_variance: float,
    piston_covariance_m2: NDArray[np.float64] | None,
    resolved_amp_servo_bandwidth_hz: float | None,
    effective_bandwidth_hz: float | None,
    resolved_f_piston_hz: float | None = None,
) -> dict[str, Any]:
    """Return provenance for the scalar systematic-bandwidth bridge."""

    if params.effective_systematic_bandwidth_hz_override is not None:
        source = "params.effective_systematic_bandwidth_hz_override"
    elif params.auto_amplitude_servo_residual:
        source = "leakage_variance_weighted_piston_scintillation_blend"
    else:
        source = "performance.default_servo_bw_hz"
    piston_model = (
        "covariance_trace_rms_scalarized"
        if piston_covariance_m2 is not None
        else "scalar_rms"
    )
    return {
        "source": source,
        "effective_systematic_bandwidth_hz": (
            None if effective_bandwidth_hz is None else float(effective_bandwidth_hz)
        ),
        "sigma_phi_piston_rad": float(sigma_phi_piston_rad),
        "sigma_I_residual": float(sigma_I_residual),
        "piston_leakage_variance": float(piston_leakage_variance),
        "amplitude_leakage_variance": float(amplitude_leakage_variance),
        "piston_rms_model": piston_model,
        "piston_covariance_used_in_realized_null_mc": piston_covariance_m2 is not None,
        "piston_covariance_directionality_used_in_f_eff": (
            piston_covariance_m2 is not None and piston_leakage_variance > 0.0
        ),
        "resolved_amp_servo_bandwidth_hz": (
            None
            if resolved_amp_servo_bandwidth_hz is None
            else float(resolved_amp_servo_bandwidth_hz)
        ),
        "f_piston_hz": (
            float(resolved_f_piston_hz)
            if resolved_f_piston_hz is not None
            else float(params.auto_f_piston_hz)
        ),
        "f_piston_source": (
            "fringe_tracking results f_3db_hz"
            if resolved_f_piston_hz is not None
            else "params.auto_f_piston_hz (fallback)"
        ),
        "model": (
            "1/f_eff = w_phi/f_piston + w_I/f_amp with w_i the share of the "
            "photon-weighted band leakage variance from realized-null MC runs "
            "with only piston and only amplitude perturbations (same seeds). "
            "sigma_phi_piston_rad and sigma_I_residual are provenance only."
        ),
    }


def _split_apertures_to_modes(
    n_apertures: int,
    n_modes: int,
    full_aperture_area_m2: float,
    full_aperture_diameter_m: float,
    pupil_template: str | None = None,
    mode_diameter_m: float | NDArray[np.float64] | None = None,
    mode_area_m2: float | NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Distribute pupil area / diameter across sub-aperture modes.

    For the angular-coupling factor (Ruilier rolloff at large
    separations), the sub-aperture diameter matters a lot: a naive
    ``D_full / sqrt(n_sub_per_telescope)`` is the maximum
    *area-equivalent* diameter, but a real obstructed pupil packs
    smaller sub-apertures with gaps. KILO's keck_only_config uses
    ``optimize_subaperture_positions`` to pack 12 sub-apertures inside
    the hexagonal Keck primary avoiding the 2.4 m central obstruction;
    that gives d_sub ~ 1.93 m vs the naive 2.89 m. The 50% larger
    naive value would over-attenuate the planet field at 300 mas by
    ``exp(-(2.27)^2) / exp(-(3.4)^2) ~ 1e3``, sending the contrast
    curve to nonsense at that separation.

    When ``pupil_template`` resolves to a packed sub-aperture diameter,
    the per-sub *area* also uses ``π·(d_sub/2)²`` rather than the
    full-pupil ``A_full / n_sub`` share: KILO Keck packs ~44.7% of the
    obstructed primary into 12 d_sub circles, so the area-equivalent
    overcount is ~2.24× per mode (KILO_PARITY_PLAN §1.1).

    Returns (areas_m2 [M], diameters_m [M]).
    """

    if n_apertures <= 0:
        raise ValueError(
            f"sensitivity.detection_curve: n_apertures must be > 0 (got {n_apertures})."
        )
    if n_modes % n_apertures != 0:
        raise ValueError(
            "sensitivity.detection_curve: n_modes must be a multiple of "
            f"n_apertures (got {n_modes} % {n_apertures} != 0)."
        )
    n_sub_per_tel = n_modes // n_apertures

    mode_areas_array = _optional_mode_vector(
        mode_area_m2,
        n_modes=n_modes,
        name="mode_area_m2",
    )
    mode_diameters_array = _optional_mode_vector(
        mode_diameter_m,
        n_modes=n_modes,
        name="mode_diameter_m",
    )
    if mode_areas_array is not None:
        if mode_diameters_array is None:
            mode_diameters_array = np.sqrt(4.0 * mode_areas_array / math.pi)
        return mode_areas_array, mode_diameters_array

    sub_diameter: float | None = (
        float(mode_diameter_m) if mode_diameter_m is not None else None
    )
    sub_diameter_is_packed = mode_diameter_m is not None
    if sub_diameter is None and pupil_template is not None:
        try:
            from nullsim.physics.pupil_geometry import (
                eelt_pupil,
                keck_pupil,
                optimize_subaperture_positions,
            )
            if pupil_template == "keck":
                pupil = keck_pupil()
            elif pupil_template == "eelt":
                pupil = eelt_pupil()
            else:
                pupil = None
            _, d_sub_packed = optimize_subaperture_positions(
                n_sub_per_tel, full_aperture_diameter_m, pupil,
            )
            sub_diameter = float(d_sub_packed)
            sub_diameter_is_packed = True
        except (ValueError, RuntimeError) as exc:
            # Surface the degradation loudly: the naive D/sqrt(N) diameter
            # is ~1.5x off the packed value, worth ~1e3 in angular-coupling
            # attenuation at 300 mas (review 2026-06-10 — previously a
            # silent bare-except fallback).
            logging.warning(
                "sensitivity.detection_curve: sub-aperture packing failed "
                "(%s: %s); falling back to naive D/sqrt(N) diameter. "
                "Angular-coupling rolloff will be inaccurate.",
                type(exc).__name__,
                exc,
            )
            sub_diameter = None
    if sub_diameter is None:
        sub_diameter = full_aperture_diameter_m / math.sqrt(n_sub_per_tel)
    if mode_area_m2 is not None:
        sub_area = float(mode_area_m2)
    elif sub_diameter_is_packed:
        sub_area = math.pi * (sub_diameter / 2.0) ** 2
    else:
        sub_area = full_aperture_area_m2 / n_sub_per_tel

    areas = np.full(n_modes, sub_area, dtype=np.float64)
    diameters = np.full(n_modes, sub_diameter, dtype=np.float64)
    return areas, diameters


def _optional_mode_vector(
    value: float | NDArray[np.float64] | None,
    *,
    n_modes: int,
    name: str,
) -> NDArray[np.float64] | None:
    if value is None or np.isscalar(value):
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (n_modes,):
        raise ValueError(
            f"sensitivity.detection_curve: {name} must have shape "
            f"({n_modes},), got {arr.shape}."
        )
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(
            f"sensitivity.detection_curve: {name} must be positive and finite."
        )
    return arr.astype(np.float64, copy=True)


def _pupil_modes_mode_area_and_diameter_m(
    state: SimulationState,
) -> tuple[float | NDArray[np.float64] | None, float | NDArray[np.float64] | None]:
    n_modes = _current_mode_count(state)
    if n_modes is not None:
        area_component = component_mode_areas_m2(state, n_modes)
        diameter_component = component_mode_diameters_m(state, n_modes)
        area_array = None if area_component is None else area_component[0]
        diameter_array = None if diameter_component is None else diameter_component[0]
        if area_array is not None or diameter_array is not None:
            return area_array, diameter_array

    # Newest first: the oldest component is the telescope primary, which a
    # later pupil_modes or AO stage subdivides (review 2026-11-07 Issue 1).
    for payload in reversed(tuple(state.geometry.components.values())):
        if not isinstance(payload, dict):
            continue
        area_m2 = _positive_finite_payload_float(payload, "mode_area_m2")
        diameter_m = _positive_finite_payload_float(payload, "mode_diameter_m")
        if area_m2 is not None or diameter_m is not None:
            return area_m2, diameter_m
    return None, None


def _current_mode_count(state: SimulationState) -> int | None:
    if state.field.amplitudes is not None:
        return int(np.asarray(state.field.amplitudes).shape[0])
    if state.geometry.aperture_positions_pupil_m is not None:
        return int(np.asarray(state.geometry.aperture_positions_pupil_m).shape[0])
    return None


def _positive_finite_payload_float(
    payload: dict[str, Any],
    key: str,
) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(scalar) and scalar > 0.0:
        return scalar
    return None


def _pupil_modes_mode_diameter_m(state: SimulationState) -> float | None:
    _, diameter_m = _pupil_modes_mode_area_and_diameter_m(state)
    return diameter_m


def _band_mean_strehl(state: SimulationState) -> float:
    """Mean Strehl across the wavelength grid (clipped to [0, 1])."""

    strehls = np.asarray(state.wavefront.strehl_ratio, dtype=np.float64)
    if strehls.size == 0:
        return 1.0
    return float(np.clip(np.mean(strehls), 0.0, 1.0))


def _per_azimuth_stellar_fields(
    *,
    state: SimulationState,
    base_field: NDArray[np.complex128],
    n_azimuths: int,
    n_apertures: int,
) -> NDArray[np.complex128]:
    """Build per-azimuth stellar fields for the realized-null MC.

    KILO ``azimuth_averaged_realized_null_depth`` (transport.py L1739)
    samples ``n_azimuths`` uniformly in [0, 360°) and reapplies delay-line
    transport to the chip-input field at each one. When the upstream
    ``delay_lines.geometric`` diagnostics are present, we recover the geometry
    inputs from them and rebuild per-aperture transmissions and, when enabled,
    residual chromatic phase.

    Args:
        state: The current SimulationState.
        base_field: The chip-input stellar field BEFORE delay-line
            losses, shape ``(M,)``.
        n_azimuths: Number of uniformly-spaced azimuths in [0, 360°).
        n_apertures: Number of physical apertures (telescopes); the
            field has ``M = n_apertures * modes_per_aperture`` entries.

    Returns:
        Per-azimuth chip-input fields, shape ``(n_azimuths, M)`` when the
        delay-line field is wavelength independent, or ``(n_azimuths, n_wl, M)``
        when chromatic delay-line phase is present.
    """

    from nullsim.stages.delay_lines.geometric import (
        per_aperture_chromatic_phase_rad,
        per_aperture_transmission,
    )

    delay_diag = _delay_line_diagnostic_payload(state)

    field = np.asarray(base_field, dtype=np.complex128)
    if field.ndim != 1:
        raise ValueError(
            f"_per_azimuth_stellar_fields: base_field must be 1-D; "
            f"got shape {field.shape}."
        )

    if delay_diag is None or state.geometry.aperture_positions_enu_m is None:
        # No delay-line diagnostic upstream — fall back to broadcasting
        # the on-axis stellar field across azimuths (no transport-loss
        # azimuthal variation). This is the no-AO / no-delay-line case
        # (e.g. HWO space pipelines) and is the documented contract.
        return np.broadcast_to(field, (n_azimuths, field.size)).copy()

    aperture_positions = np.asarray(
        state.geometry.aperture_positions_enu_m, dtype=np.float64,
    )
    fiber_lengths_m = np.asarray(
        delay_diag.get("fiber_lengths_m", ()), dtype=np.float64,
    )
    if fiber_lengths_m.size != n_apertures:
        # Delay-line diagnostic is present but its fiber-length array
        # doesn't match the geometry's aperture count — a real config
        # mismatch (the delay-lines stage emitted N fibers, geometry
        # has M apertures). Earlier this silently returned the
        # no-transport broadcast, losing the azimuth-averaging physics.
        raise ValueError(
            "sensitivity.detection_curve: delay_line_loss diagnostic "
            f"reports {fiber_lengths_m.size} fiber lengths but geometry "
            f"has {n_apertures} apertures. Upstream stage misconfiguration."
        )

    n_eff = float(delay_diag.get("n_eff", 1.4682))
    phase_index_ref = float(delay_diag.get("phase_index_ref", n_eff))
    group_index_ref = float(delay_diag.get("group_index_ref", phase_index_ref))
    dispersion_ps_nm_km = float(delay_diag.get("dispersion_ps_nm_km", 0.0))
    chromatic_control = str(delay_diag.get("chromatic_control", "none"))
    wavelength_grid_um = tuple(float(value) for value in state.throughput.wavelength_grid_um)
    if not wavelength_grid_um:
        raise ValueError(
            "sensitivity.detection_curve: delay_line_loss diagnostic requires "
            "a non-empty throughput wavelength grid."
        )
    reference_wavelength_um = float(
        delay_diag.get("reference_wavelength_um", np.median(wavelength_grid_um))
    )
    attenuation_db_per_m = float(delay_diag.get("attenuation_db_per_m", 0.0))
    altitude_deg = float(
        delay_diag.get("altitude_deg", state.scene.altitude_deg or 90.0)
    )
    fixed_loss_db = delay_diag.get("fixed_delay_line_loss_db_per_telescope")
    if fixed_loss_db is not None:
        fixed_loss_db = tuple(
            None if v is None else float(v) for v in fixed_loss_db
        )
    fixed_loss_mode = str(delay_diag.get("fixed_delay_line_loss_mode", "replace_delay"))

    n_modes = field.size
    if n_modes % n_apertures != 0:
        # Mode count not divisible by aperture count — another upstream
        # misconfiguration that earlier silently fell back. The per-mode
        # broadcasting of per-aperture transmission requires an integer
        # modes-per-aperture ratio.
        raise ValueError(
            "sensitivity.detection_curve: field has "
            f"{n_modes} modes but geometry has {n_apertures} apertures "
            "(n_modes must be a multiple of n_apertures for per-aperture "
            "transport broadcasting)."
        )
    modes_per_aperture = n_modes // n_apertures

    azimuths_deg = np.linspace(0.0, 360.0, n_azimuths, endpoint=False)
    has_chromatic_phase = chromatic_control != "none"
    if has_chromatic_phase:
        fields_per_az_wl = np.empty(
            (n_azimuths, len(wavelength_grid_um), n_modes),
            dtype=np.complex128,
        )
    else:
        fields_per_az = np.empty((n_azimuths, n_modes), dtype=np.complex128)
    for az_idx, az in enumerate(azimuths_deg):
        transmission = per_aperture_transmission(
            aperture_positions_enu_m=aperture_positions,
            fiber_lengths_m=fiber_lengths_m,
            n_eff=phase_index_ref,
            attenuation_db_per_m=attenuation_db_per_m,
            altitude_deg=altitude_deg,
            azimuth_deg=float(az),
            fixed_loss_db=fixed_loss_db,
            fixed_loss_mode=fixed_loss_mode,
        )
        per_mode_scale = np.sqrt(np.repeat(transmission, modes_per_aperture))
        if has_chromatic_phase:
            phase = per_aperture_chromatic_phase_rad(
                aperture_positions_enu_m=aperture_positions,
                fiber_lengths_m=fiber_lengths_m,
                phase_index_ref=phase_index_ref,
                group_index_ref=group_index_ref,
                dispersion_ps_nm_km=dispersion_ps_nm_km,
                chromatic_control=chromatic_control,
                reference_wavelength_um=reference_wavelength_um,
                wavelength_grid_um=wavelength_grid_um,
                altitude_deg=altitude_deg,
                azimuth_deg=float(az),
                fixed_loss_db=fixed_loss_db,
                fixed_loss_mode=fixed_loss_mode,
            )
            per_mode_phase = np.repeat(phase, modes_per_aperture, axis=0).T
            fields_per_az_wl[az_idx] = (
                field[np.newaxis, :]
                * per_mode_scale[np.newaxis, :].astype(field.dtype)
                * np.exp(1j * per_mode_phase)
            )
        else:
            fields_per_az[az_idx] = field * per_mode_scale.astype(field.dtype)

    return fields_per_az_wl if has_chromatic_phase else fields_per_az


def _planet_input_transport_scales(
    *,
    state: SimulationState,
    n_azimuths: int,
    n_modes: int,
    n_wl: int,
    n_apertures: int,
) -> NDArray[np.complex128] | None:
    """Return delay-line transport scales for planet-throughput fields."""

    if (
        _delay_line_diagnostic_payload(state) is None
        or state.geometry.aperture_positions_enu_m is None
    ):
        return None
    unit_field = np.ones(n_modes, dtype=np.complex128)
    transported = _per_azimuth_stellar_fields(
        state=state,
        base_field=unit_field,
        n_azimuths=n_azimuths,
        n_apertures=n_apertures,
    )
    if transported.ndim == 2:
        return np.broadcast_to(
            transported[:, np.newaxis, :],
            (transported.shape[0], n_wl, transported.shape[1]),
        ).copy()
    if transported.ndim == 3:
        return transported
    raise ValueError(
        "sensitivity.detection_curve: delay-line planet transport scales "
        f"must be 2-D or 3-D; got {transported.shape}."
    )


def _static_field_factors(
    *,
    state: SimulationState,
    n_arms: int,
    n_wl: int,
    n_chip_modes: int,
    mode_group_indices: NDArray[np.intp] | None,
    design_index: int | None,
) -> tuple[NDArray[np.complex128] | None, tuple[str, ...]]:
    """Product of the upstream per-mode static field factors, per chip arm.

    Transport stages publish their per-mode GVD phase and
    ``polarization.pm_fiber`` its per-mode, per-arm Jones J00 as a field
    component ``static_field_factor`` of shape ``(n_modes, n_wl, n_pol)``
    (review 2026-11-07 Issue 10, row 47). The deployed chip sees them; the
    fields this stage rebuilds from coupling and area do not, so they are
    multiplied in here, as the delay-line chromatic phase is replayed.
    Factors published before ``injection.mode_splitter`` are mapped to the
    chip inputs through the splitter's physical-mode indices; a length-one
    polarization axis applies to every arm.

    With a per-arm input phase trim (``design_index`` set), the chip
    cancels each mode's design-bin phase, so each factor is taken relative
    to its own phase in that bin.

    Returns ``(factors, sources)`` with ``factors`` of shape
    ``(n_arms, n_wl, n_chip_modes)``, or ``(None, ())`` without any factor.
    """

    total: NDArray[np.complex128] | None = None
    sources: list[str] = []
    for name, payload in state.field.components.items():
        if not isinstance(payload, Mapping) or "static_field_factor" not in payload:
            continue
        factor = np.asarray(payload["static_field_factor"], dtype=np.complex128)
        if factor.ndim != 3 or factor.shape[1] != n_wl:
            raise ValueError(
                f"sensitivity.detection_curve: field component {name!r} "
                f"static_field_factor has shape {factor.shape}; expected "
                f"(n_modes, {n_wl}, n_pol)."
            )
        if factor.shape[0] != n_chip_modes:
            if mode_group_indices is None or factor.shape[0] <= int(
                np.max(mode_group_indices)
            ):
                raise ValueError(
                    f"sensitivity.detection_curve: field component {name!r} "
                    f"static_field_factor covers {factor.shape[0]} modes, which "
                    f"neither match the {n_chip_modes} chip inputs nor map "
                    "through an injection.mode_splitter."
                )
            factor = factor[mode_group_indices]
        n_pol = factor.shape[2]
        if n_pol != n_arms:
            if n_pol == 1:
                factor = np.broadcast_to(factor, (n_chip_modes, n_wl, n_arms))
            elif n_arms == 1 and np.allclose(factor, factor[:, :, :1]):
                factor = factor[:, :, :1]
            else:
                raise ValueError(
                    f"sensitivity.detection_curve: field component {name!r} "
                    f"carries {n_pol} polarization-dependent factors but the "
                    f"chip deploys {n_arms} arm(s); use "
                    "deployment_mode = 'per_pol_arm'."
                )
        total = factor.copy() if total is None else total * factor
        sources.append(f"field.{name}")
    if total is None:
        return None, ()
    if design_index is not None:
        design_phase = total[:, design_index, :] / np.abs(total[:, design_index, :])
        total = total * np.conj(design_phase)[:, np.newaxis, :]
    return np.transpose(total, (2, 1, 0)).copy(), tuple(sources)


def _photon_wavelength_weights(
    state: SimulationState,
    wavelength_grid_um: NDArray[np.float64],
) -> tuple[NDArray[np.float64], str]:
    """Detected stellar photon rate per wavelength bin, normalized to one.

    Reads the upstream detector's per-port, per-bin rates (the same rates
    whose port sums give the band totals that multiply the null), summed
    over ports, before saturation and MKID energy redistribution. Falls
    back to bin width times upstream transmission without a detector.
    """

    n_wl = int(np.asarray(wavelength_grid_um).size)
    detector = _find_detector_component(state)
    if detector is not None:
        for key in (
            "pre_energy_response_port_rate_hz_per_wavelength",
            "uncapped_port_rate_hz_per_wavelength",
        ):
            rates = detector.get(key)
            if rates is None:
                continue
            rates_arr = np.asarray(rates, dtype=np.float64)
            if rates_arr.ndim != 2 or rates_arr.shape[1] != n_wl:
                continue
            per_bin = np.sum(rates_arr, axis=0)
            if np.all(np.isfinite(per_bin)) and np.sum(per_bin) > 0.0:
                return per_bin / float(np.sum(per_bin)), f"detector:{key}"
    return (
        _residual_starlight_wavelength_weights(state, wavelength_grid_um),
        "bin_width_times_transmission",
    )


def _mode_piston_covariance_m2(
    *,
    sigma_opd_m: float,
    covariance_m2: NDArray[np.float64] | None,
    mode_group_indices: NDArray[np.intp] | None,
    n_modes: int,
) -> NDArray[np.float64] | None:
    """Per-chip-mode piston covariance matching the realized-null MC draws."""

    groups = (
        np.arange(n_modes, dtype=np.intp)
        if mode_group_indices is None
        else np.asarray(mode_group_indices, dtype=np.intp)
    )
    if covariance_m2 is not None:
        cov = np.asarray(covariance_m2, dtype=np.float64)
        if cov.shape == (n_modes, n_modes):
            return cov
        return cov[np.ix_(groups, groups)]
    if sigma_opd_m <= 0.0:
        return None
    same_group = groups[:, np.newaxis] == groups[np.newaxis, :]
    return float(sigma_opd_m) ** 2 * same_group.astype(np.float64)


def _detector_stellar_and_static_leakage_rates_hz(
    state: SimulationState,
) -> tuple[float | None, float | None]:
    """Resolve stellar normalization and static-chip leakage from the detector.

    Mirrors ``performance.standard``'s rate bookkeeping so the sensitivity
    stage can publish a realized leakage rate on the same normalization the
    SNR pipeline uses. Returns ``(total_stellar_rate_hz, static_chip_leakage_hz)``
    where ``total_stellar_rate_hz`` is the bright + dark + spectro port total
    of the detector's uncapped rates (``port_rates`` only for detectors that
    publish no uncapped rates, which do not saturate)
    and ``static_chip_leakage_hz`` is the detector dark-port total (the static
    chip null seen by the detector stage, before the AO / disk / setpoint
    realization the sensitivity MC adds).

    Returns ``(None, None)`` when no upstream ``detector.*`` component with
    port rates is present (sensitivity can run standalone in smoke tests).
    """

    detector = _find_detector_component(state)
    if detector is None:
        return None, None
    # Physical (pre-saturation) rates: saturation is a readout limit, not a
    # statement about the stellar flux (review 2026-11-07 Issue 9).
    port_rates_hz = np.asarray(
        detector.get("uncapped_port_rates", detector["port_rates"]),
        dtype=np.float64,
    )
    port_labels = tuple(str(label) for label in detector["port_labels"])
    if port_rates_hz.shape != (len(port_labels),):
        return None, None
    bright_mask = np.array(
        [label.startswith("bright") for label in port_labels], dtype=bool
    )
    dark_mask = np.array(
        [label.startswith("dark") for label in port_labels], dtype=bool
    )
    spectro_mask = np.array(
        [label.startswith("spectro_") for label in port_labels], dtype=bool
    )
    static_chip_leakage_hz = float(port_rates_hz[dark_mask].sum())
    total_stellar_rate_hz = float(
        port_rates_hz[bright_mask].sum()
        + static_chip_leakage_hz
        + port_rates_hz[spectro_mask].sum()
    )
    return total_stellar_rate_hz, static_chip_leakage_hz


def _find_detector_component(state: SimulationState) -> Mapping[str, Any] | None:
    """Return the newest ``detector.*`` results component with port rates."""

    for name, value in reversed(state.results.components.items()):
        if not name.startswith("detector."):
            continue
        if not isinstance(value, Mapping):
            continue
        if {"port_rates", "port_labels"}.issubset(value.keys()):
            return value
    return None


@register_stage
class DetectionCurve(BaseStage):
    """Planet throughput vs separation + stellar-disk null floor + realized null."""

    type_name = "sensitivity.detection_curve"
    family = "sensitivity"
    consumes = frozenset({
        "results",
        "geometry",
        "wavefront",
        "scene",
        "field",
        "throughput",
    })
    produces = frozenset({"results"})
    cache_key_extras = frozenset({"opd_budget"})

    class Params(BaseModel):
        """Validated params for the per-separation sensitivity stage."""

        model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

        n_bright_ports: int = PydanticField(default=2, ge=1)
        n_dark_ports: int = PydanticField(default=22, ge=1)
        n_spectro_ports: int = PydanticField(default=0, ge=0)

        separations_mas: tuple[float, ...] = (
            # 70 log-spaced separations matching KILO detection_space_cache.
            0.2, 0.2224, 0.2472, 0.2749, 0.3056, 0.3398, 0.3778, 0.4200,
            0.4670, 0.5192, 0.5772, 0.6417, 0.7135, 0.7933, 0.8820, 0.9806,
            1.0902, 1.2121, 1.3476, 1.4983, 1.6659, 1.8521, 2.0592, 2.2894,
            2.5454, 2.8300, 3.1464, 3.4982, 3.8894, 4.3242, 4.8077, 5.3453,
            5.9429, 6.6074, 7.3462, 8.1675, 9.0807, 10.0960, 11.2249, 12.4799,
            13.8753, 15.4267, 17.1515, 19.0692, 21.2014, 23.5719, 26.2074,
            29.1376, 32.3955, 36.0176, 40.0447, 44.5221, 49.5001, 55.0347,
            61.1881, 68.0294, 75.6358, 84.0926, 93.4949, 103.9485, 115.5709,
            128.4928, 142.8595, 158.8325, 176.5915, 196.3360, 218.2882,
            242.6949, 269.8304, 300.0,
        )
        n_pa_samples: int = PydanticField(default=36, ge=4)
        include_angular_coupling: bool = True
        # RMS line-of-sight pointing/vibration jitter. This applies a simple
        # Gaussian convolution to radial separation-dependent throughput and
        # residual-starlight profiles before performance/background stages
        # consume them. It is a 1-D radial surrogate, not a full image-plane
        # jitter propagation.
        pointing_jitter_rms_mas: float | None = PydanticField(default=None, ge=0.0)

        # Realized-null Monte Carlo. KILO paperv3 uses n_realizations=200
        # across all studies (natural_design.py L1050, sensitivity.py L176,
        # n_optimization.py L195, piston_servo.py L176, wavelength.py L189,
        # verification.py L1580). MC noise drops as sqrt(N): 2000 gives
        # only ~3× tighter CV than 200 at 10× the cost.
        n_mc_realizations: int = PydanticField(default=200, ge=0)
        # Phase 1: KILO ``azimuth_averaged_realized_null_depth`` defaults
        # to 6 azimuths x 200 realizations each (transport.py L1739).
        # Azimuth marginalization pulls in the delay-line-loss imbalance
        # the source sweeps through during an Earth-rotation track.
        n_azimuths: int = PydanticField(default=6, ge=1)
        # When None, falls back to ``n_mc_realizations`` for back-compat.
        # Set explicitly (paperv3 default: 200) to make the per-azimuth
        # realization budget independent of the legacy single-azimuth path.
        # ``0`` is the explicit "skip the MC pass" toggle; negative values
        # used to silently produce an optimistic ideal/no-instability result.
        n_realizations_per_azimuth: int | None = PydanticField(default=None, ge=0)
        mc_seed: int | None = PydanticField(default=0)
        # Scaling factor applied to the per-sub-aperture AO piston RMS
        # before it is passed to ``realized_null_depth_mc``. When the
        # pyramid_wfs piston is overestimated (e.g. the temporal-error
        # Conan formula gives ~10 nm at J=4 Keck where KILO's tuned
        # servo achieves ~4 nm), use 0.4 to scale into KILO's regime.
        # Default 1.0 (no scaling) preserves existing behaviour. Tier 2
        # closed_loop.py fixes drop the σ²_atmos multiplier from the Conan
        # lag; with the corrected upstream AO piston this scale should
        # fall toward 1.0 after Tier 2.4 (servo-corrected per-sub
        # piston) lands.
        ao_piston_scale: float = PydanticField(default=1.0, gt=0.0)
        # Piston residual source for the realized-null MC. ``opd_budget`` uses
        # the upstream fringe_tracking.closed_loop scalar. ``fringe_observability``
        # uses an upstream diagnostics.fringe_observability payload, useful for
        # quadrature-mesh tests where the actual bright-port reconstruction
        # should set the piston floor.
        ao_piston_residual_source: Literal[
            "opd_budget",
            "fringe_observability",
        ] = "opd_budget"
        # Which diagnostics.fringe_observability piston residual to use when
        # ``ao_piston_residual_source='fringe_observability'``. ``auto`` prefers
        # the science-unitary linear Monte Carlo median, then CRLB metrics, then
        # the dedicated cophasing handoff median if present.
        fringe_observability_piston_metric: Literal[
            "auto",
            "linear_monte_carlo_median",
            "linear_monte_carlo_p95",
            "crlb",
            "crlb_truncated",
            "cophasing_handoff_science_median",
            "cophasing_handoff_science_p95",
        ] = "auto"
        # When available, inject the full physical-mode CRLB covariance from
        # diagnostics.fringe_observability rather than an IID scalar piston RMS.
        fringe_observability_piston_covariance: Literal[
            "auto",
            "crlb",
            "disabled",
        ] = "auto"
        # Magnitude model applied to the upstream bright-port covariance.
        # ``scalar_gain`` is the legacy path: divide every mode's variance by
        # ``fringe_observability_predictive_gain_variance``. ``temporal_kalman``
        # uses a one-pole AR(1) Kalman surrogate whose correlation frequency
        # is derived from the Conan/Di Folco differential-piston PSD breaks and
        # whose measurement noise is the bright-port CRLB covariance.
        fringe_observability_predictive_model: Literal[
            "scalar_gain",
            "temporal_kalman",
        ] = "scalar_gain"
        # Predictive/Kalman servo assumption applied to the diagnostic residual:
        # variance is divided by this gain, so RMS is divided by sqrt(gain).
        # 1.0 uses the instantaneous diagnostic residual as-is.
        fringe_observability_predictive_gain_variance: float = PydanticField(
            default=1.0,
            ge=1.0,
        )
        # Optional RMS floor after predictive filtering, in nm.
        fringe_observability_piston_floor_nm: float = PydanticField(
            default=0.0,
            ge=0.0,
        )
        # Temporal-Kalman options. By default the prior RMS comes from the
        # upstream raw AO piston residual, the sample time comes from the
        # fringe-observability frame_time_s, and the sub-aperture diameter comes
        # from the sensitivity geometry path.
        fringe_observability_temporal_residual_kind: Literal[
            "posterior",
            "one_step_prediction",
        ] = "posterior"
        fringe_observability_temporal_prior_rms_nm: float | None = PydanticField(
            default=None,
            gt=0.0,
        )
        fringe_observability_temporal_sample_time_s: float | None = PydanticField(
            default=None,
            gt=0.0,
        )
        fringe_observability_temporal_wind_speed_m_s: float = PydanticField(
            default=10.0,
            gt=0.0,
        )
        # Scintillation index σ_I residual after the amp servo, per
        # sub-aperture. Passed to realized_null_depth_mc as
        # ``scintillation_index_per_sub`` (KILO transport L1639). Default
        # 0.0 means no amplitude jitter (back-compat). Auto-derived from
        # physics when ``auto_amplitude_servo_residual=True`` (see below);
        # the payload records both open-loop and injected residual values.
        scintillation_index_per_sub: float = PydanticField(default=0.0, ge=0.0)

        # Tier 4.4: IID per-mode residual fiber phase noise (KILO
        # ``FiberConfig.phase_noise_rms_rad`` convention; typical 0.01
        # rad for SMF28 + thermal drift). Passed through to
        # ``realized_null_depth_mc`` as ``fiber_phase_noise_rms_rad``.
        # Default 0.0 keeps the pre-Tier-4 behavior (no fiber noise).
        fiber_phase_noise_rms_rad: float = PydanticField(default=0.0, ge=0.0)

        # When True, derive both ``scintillation_index_per_sub`` AND the
        # variance-weighted f_eff blend from physics, ignoring the
        # manual ``scintillation_index_per_sub`` scalar. σ_I comes from
        # ``nullsim.physics.scintillation.amplitude_servo_residual_sigma_i``;
        # f_eff comes from
        # ``nullsim.physics.systematic_bandwidth.effective_systematic_bandwidth_hz``
        # (the variance-weighted KILO blend) and propagates as
        # ``effective_systematic_bandwidth_hz`` in the payload only if
        # ``effective_systematic_bandwidth_hz_override`` is None.
        # Disabled by default for back-compat; set to True in
        # examples/kilo_keck.toml to replace the manual override stack
        # with KILO physics.
        auto_amplitude_servo_residual: bool = False
        # Select which amplitude term the sensitivity MC injects when
        # ``auto_amplitude_servo_residual=True``. ``closed_loop`` uses the
        # predictive amplitude-servo residual. ``open_loop`` keeps the
        # same Roddier scintillation calculation and photon provenance but
        # injects the uncontrolled sub-aperture σ_I instead, with an
        # effective bandwidth set by the scintillation timescale.
        auto_amplitude_servo_mode: Literal["closed_loop", "open_loop"] = (
            "closed_loop"
        )

        # Maunakea defaults (KILO AtmosphereConfig).
        # ``auto_amp_servo_photon_rate_per_sub_hz`` is the per-sub-
        # aperture photon rate available to the amp WFS measurement.
        # When None and ``auto_amplitude_servo_residual=True``, derive
        # from the detector's bright-port rate divided by n_modes (the
        # bright_rate already includes all upstream throughput losses).
        # Kornilov et al. (2012) Q1 Maunakea measurement: S_2^2 = 0.77e-4
        # m^(7/3) -> sigma_I(5 m, zenith) = 1.3e-3, achromatic (2026-07-13
        # audit; the old 0.041 was ~30x the measured value).
        auto_scintillation_anchor_h5m_zenith: float = PydanticField(
            default=0.0013, gt=0.0,
        )
        auto_scintillation_height_km: float = PydanticField(default=10.0, gt=0.0)
        auto_high_altitude_wind_ms: float = PydanticField(default=35.0, gt=0.0)
        # ``None`` triggers the closed-form photon-limited optimum
        # ``optimize_amplitude_servo_bandwidth`` (KILO_PARITY_PLAN §2.1):
        # f_opt = ((5/3)·P·R)^(3/8), so the right bandwidth ~4 kHz at
        # J=4 and ~380 Hz at J=8 instead of the legacy fixed 5 kHz
        # default that over-runs faint hosts and lifts the residual.
        # Set explicitly for a fixed bandwidth (e.g. legacy 5000.0
        # behavior).
        auto_amp_servo_bandwidth_hz: float | None = PydanticField(
            default=None, gt=0.0,
        )
        auto_amp_predictive_gain_variance: float = PydanticField(default=20.0, ge=1.0)
        auto_amp_servo_bandwidth_min_hz: float = PydanticField(
            default=200.0, gt=0.0,
        )
        auto_amp_servo_bandwidth_max_hz: float = PydanticField(
            default=10_000.0, gt=0.0,
        )
        auto_amp_servo_photon_rate_per_sub_hz: float | None = PydanticField(
            default=None, gt=0.0,
        )
        # Phase 3: KILO ``natural_design.py`` L897 says the amplitude WFS
        # sees the science flux (chip-input rate / n_modes), optionally
        # attenuated by a pickoff fraction. When a physical
        # ``transport.photometric_pickoff`` stage is upstream, that stage
        # supplies both the fraction and the pre-split physical mode count;
        # this scalar remains the fallback for configs that do not carry an
        # explicit field debit.
        amp_wfs_pickoff_fraction: float = PydanticField(default=1.0, gt=0.0, le=1.0)
        # Instruments such as NAYRA can sense amplitude jitter with the
        # simultaneous science-band photon stream, even though this run's
        # science grid covers one primary band. When set, the amp-servo
        # photon rate scales from the current grid to the summed bandpasses.
        simultaneous_science_bands: tuple[str, ...] = ()
        simultaneous_science_band_transmissions: tuple[float, ...] = ()
        # FALLBACK piston servo bandwidth for the variance-weighted f_eff
        # blend, used only when no upstream fringe_tracking results
        # component publishes f_3db_hz/servo_bandwidth_hz (review
        # 2026-06-10 Issue 2: the blend now reads the FT-published
        # closed-loop bandwidth first).
        auto_f_piston_hz: float = PydanticField(default=100.0, gt=0.0)

        # When True, fold per-band Strehl into the coupling estimate used
        # for the planet field amplitude. The chip-input field already
        # accounts for atmospheric throughput, so this is the *injection*
        # coupling alone.
        apply_strehl_coupling: bool = True

        # When True, apply the 3.5% uncoated-fiber-tip Fresnel reflection
        # loss to the coupling efficiency (KILO ``injection.py:25,72``
        # ``FRESNEL_LOSS = 0.035`` convention). Default False preserves
        # the geometric-overlap-only convention; set True when that physical
        # loss should be included.
        apply_fresnel_injection_loss: bool = False

        # When set, overrides ``effective_null_with_disk`` with a fixed
        # value. Matches KILO's detection_space model where the dark-port
        # null is the calibration-limited floor (kilo/studies/sensitivity.py
        # L155 + L207) and not derived from the chip's own optimized null.
        # The leakage fluctuation ``null_fluctuation_rms`` (null-depth units)
        # comes from the MC and the trim bias regardless of this override;
        # ``null_instability`` is that RMS divided by the effective null in
        # force, so effective null x null_instability is the RMS either way.
        effective_null_override: float | None = PydanticField(default=None, gt=0.0)
        # Residual static OPD bias left by a slow dark-port trim loop. The loop
        # licenses a small bounded value; the leakage cost is charged directly
        # and enters the fluctuation only through its cross term with the
        # random piston.
        null_setpoint_residual_nm: float = PydanticField(default=0.0, ge=0.0)
        # RMS deliberate dark-port trim dither. This is a known modulation, so
        # its leakage is charged to the effective null and enters the
        # fluctuation only through its cross term with the random piston.
        setpoint_dither_rms_nm: float = PydanticField(default=0.0, ge=0.0)
        # Photon-limited auto mode: derive the trim dither amplitude and the
        # held setpoint jointly from this run's detected stellar rate and
        # dark-port floor (physics.fringe_tracking.photon_limited_setpoint_trim:
        # the 1f lock-in signal is ~ A*delta, so delta ~ 1/A and the total
        # leakage has a real optimum). Overrides the two fixed terms above and
        # makes the faint-end trim budget degrade honestly with flux.
        auto_setpoint_trim: bool = False
        trim_loop_bandwidth_hz: float = PydanticField(default=1.0, gt=0.0)
        # Trim DOFs sensed sequentially; None -> n_dark_ports + n_spectro_ports.
        n_trim_modes: int | None = PydanticField(default=None, ge=1)

        # Optional EXOSIMS-style residual-starlight contrast profile.
        # This represents coronagraph/nuller residual stellar leakage as a
        # radial contrast floor. A single contrast value is a flat floor;
        # multiple values require matching, strictly increasing
        # separations in mas. Table CSVs need a separation_mas-like column
        # and a contrast-like column; edits to table bytes enter the cache
        # key through external_dependencies().
        residual_starlight_separations_mas: tuple[float, ...] = ()
        residual_starlight_contrast: tuple[float, ...] = ()
        residual_starlight_table_path: str | None = None
        residual_starlight_combine_mode: Literal["max", "add", "replace"] = "max"

        # When set, propagates as the f_eff used by the performance
        # stage's systematic-noise term (eps² × leak² × t / (2 f_eff)).
        # When None and ``auto_amplitude_servo_residual=True``, f_eff is
        # computed via the variance-weighted blend from physics. When
        # None and ``auto_amplitude_servo_residual=False``, the
        # performance stage falls back to its own default_servo_bw_hz.
        # KILO's natural_design pipeline computes this via
        # _effective_systematic_bandwidth_hz from σ_φ_piston, σ_I_residual,
        # and the per-J piston/amp servo bandwidths; typical Keck
        # values are 1223 Hz at J=4 and 206 Hz at J=8.
        effective_systematic_bandwidth_hz_override: float | None = PydanticField(
            default=None, gt=0.0,
        )

        @model_validator(mode="after")
        def _validate_residual_starlight_profile(self) -> "DetectionCurve.Params":
            _validate_residual_starlight_inline_params(self)
            return self

    def external_dependencies(self) -> list[ExternalDependency]:
        """Declare residual-starlight profile tables for cache/run hashes."""

        path_str = self.params_model.residual_starlight_table_path
        if path_str is None:
            return []
        path = _resolve_residual_starlight_table_path(
            path_str,
            config_dir=_config_dir_from_context(self.context),
        )
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return [
                ExternalDependency(
                    source="sensitivity.residual_starlight_table",
                    path=path,
                )
            ]
        return [
            ExternalDependency(
                source="sensitivity.residual_starlight_table",
                path=path,
                content_hash=digest,
            )
        ]

    def apply(self, state: SimulationState) -> SimulationState:
        params = self.params_model

        chip_name, chip_payload = _find_chip_unitary_component(state)
        u_stack = np.asarray(chip_payload["u_stack"], dtype=np.complex128)
        chip_wavelengths_um = tuple(float(v) for v in chip_payload["wavelength_grid_um"])
        n_modes = int(chip_payload.get("n_input_modes", chip_payload["n_modes"]))
        n_output_ports = int(chip_payload.get("n_output_ports", u_stack.shape[0]))

        if u_stack.shape[0] != n_output_ports or u_stack.shape[1] != n_modes:
            raise ValueError(
                "sensitivity.detection_curve: chip transfer shape "
                f"{u_stack.shape} does not match n_output_ports="
                f"{n_output_ports}, n_input_modes={n_modes}."
            )
        (
            unitary_stacks,
            unitary_labels,
            unitary_arm_weights,
            chip_deployment_mode,
            input_phase_correction_applied,
        ) = _unitary_stacks_for_detection(
            state=state,
            chip_payload=chip_payload,
            u_stack=u_stack,
        )
        if unitary_stacks.shape[1:] != u_stack.shape:
            raise ValueError(
                "sensitivity.detection_curve: deployed unitary stack shape "
                f"{unitary_stacks.shape[1:]} does not match shared u_stack "
                f"shape {u_stack.shape}."
            )
        wavelengths_m = np.asarray(chip_wavelengths_um, dtype=np.float64) * 1.0e-6
        n_ports = (
            int(params.n_bright_ports)
            + int(params.n_dark_ports)
            + int(params.n_spectro_ports)
        )
        if n_ports != n_output_ports:
            raise ValueError(
                "sensitivity.detection_curve requires "
                "n_bright_ports + n_dark_ports + n_spectro_ports == chip output ports "
                f"(got {params.n_bright_ports} + {params.n_dark_ports} + "
                f"{params.n_spectro_ports} != {n_output_ports})."
            )

        if state.geometry.area_m2 is None or state.geometry.diameter_m is None:
            raise ValueError(
                "sensitivity.detection_curve: geometry.area_m2 and "
                "geometry.diameter_m are required."
            )
        n_apertures = state.geometry.n_apertures or 1
        # Pull pupil_template from the upstream pyramid_wfs component so
        # the packed-d_sub matches what the chip optimizer used. Falls
        # back to None (and the naive D/sqrt(n_sub) formula) when no AO
        # stage ran (e.g. space configs).
        pupil_template: str | None = None
        ao_payload = state.wavefront.components.get("ao_residual")
        if isinstance(ao_payload, dict):
            pupil_template_raw = ao_payload.get("pupil_template")
            if isinstance(pupil_template_raw, str):
                pupil_template = pupil_template_raw
        pupil_modes_mode_area_m2, pupil_modes_mode_diameter_m = (
            _pupil_modes_mode_area_and_diameter_m(state)
        )
        sub_areas_m2, sub_diameters_m = _split_apertures_to_modes(
            n_apertures=n_apertures,
            n_modes=n_modes,
            full_aperture_area_m2=float(state.geometry.area_m2),
            full_aperture_diameter_m=float(state.geometry.diameter_m),
            pupil_template=pupil_template,
            mode_diameter_m=pupil_modes_mode_diameter_m,
            mode_area_m2=pupil_modes_mode_area_m2,
        )

        positions = state.geometry.aperture_positions_pupil_m
        if positions is None:
            raise ValueError(
                "sensitivity.detection_curve: geometry.aperture_positions_pupil_m "
                "is required (the chip kernel-null needs per-mode positions)."
            )
        positions_m = np.asarray(positions, dtype=np.float64)
        if positions_m.shape != (n_modes, 2):
            raise ValueError(
                "sensitivity.detection_curve: aperture_positions_pupil_m has "
                f"shape {positions_m.shape}; expected ({n_modes}, 2)."
            )
        # Project onto the sky plane for off-zenith pointing so the planet
        # phase rows (2*pi/lambda * (x*alpha + y*beta)) foreshorten
        # correctly. With published per-mode origins only the
        # inter-telescope baseline foreshortens; pupil-mounted sub-aperture
        # offsets track the target (review 2026-06-10 Issue 20).
        positions_m = project_aperture_positions_for_pointing(
            positions_m,
            altitude_deg=state.scene.altitude_deg,
            azimuth_deg=state.scene.azimuth_deg,
            aperture_origins_m=mode_origin_positions_m(state.geometry, n_modes),
        )

        # Phase 4: read the central obstruction from the upstream
        # telescope diagnostic so the planet-leg coupling matches the
        # stellar-leg coupling that injection.ideal already applies
        # (KILO unifies these). Pre-Phase-4 this was hardcoded 0.0,
        # which made the planet coupling inconsistent with the stellar
        # one for any obstructed primary (Keck epsilon = 0.24).
        obstruction_ratio = _central_obstruction_ratio(state, default=0.0)
        coupling_per_sub, coupling_source = _coupling_per_sub_from_state(
            state=state,
            params=params,
            n_modes=n_modes,
            obstruction_ratio=obstruction_ratio,
        )

        dark_indices = np.arange(
            params.n_bright_ports,
            params.n_bright_ports + params.n_dark_ports,
            dtype=np.intp,
        )

        separations_mas = np.asarray(params.separations_mas, dtype=np.float64)
        separations_rad = separations_mas * _RAD_PER_MAS
        pointing_jitter_rms_mas, pointing_jitter_source = (
            _resolve_pointing_jitter_rms_mas(
                state=state,
                params=params,
            )
        )
        n_delay_apertures = int(state.geometry.aperture_positions_enu_m.shape[0]) if (
            state.geometry.aperture_positions_enu_m is not None
        ) else n_modes
        planet_transport_scales = _planet_input_transport_scales(
            state=state,
            n_azimuths=int(params.n_azimuths),
            n_modes=n_modes,
            n_wl=len(wavelengths_m),
            n_apertures=n_delay_apertures,
        )
        # One set of photon weights for every band average: realized null,
        # its fluctuation, planet throughput, disk floor, residual starlight.
        wavelength_weights, wavelength_weights_source = _photon_wavelength_weights(
            state,
            np.asarray(chip_wavelengths_um, dtype=np.float64),
        )
        (
            piston_temporal_subaperture_diameter_m,
            piston_temporal_subaperture_diameter_source,
        ) = _piston_temporal_subaperture_diameter_m(
            state=state,
            fallback_subaperture_diameter_m=float(sub_diameters_m[0]),
        )

        # AO-realized null Monte Carlo on the chip
        (
            sigma_OPD_m,
            piston_covariance_m2,
            piston_residual_diagnostics,
        ) = _resolve_piston_rms_per_sub_m(
            state=state,
            params=params,
            stage_name=self.type_name,
            n_chip_modes=int(unitary_stacks[0].shape[1]),
            subaperture_diameter_m=piston_temporal_subaperture_diameter_m,
        )
        if "fringe_observability_temporal_subaperture_diameter_m" in (
            piston_residual_diagnostics
        ):
            piston_residual_diagnostics[
                "fringe_observability_temporal_subaperture_diameter_source"
            ] = piston_temporal_subaperture_diameter_source
        mode_group_indices: NDArray[np.intp] | None = None
        mode_group_source: str | None = None
        split_indices = splitter_physical_indices(state, unitary_stacks[0].shape[1])
        if split_indices is not None:
            mode_group_indices, mode_group_source = split_indices
        static_field_factors, static_field_factor_sources = _static_field_factors(
            state=state,
            n_arms=int(unitary_stacks.shape[0]),
            n_wl=len(wavelengths_m),
            n_chip_modes=n_modes,
            mode_group_indices=mode_group_indices,
            design_index=(
                int(chip_payload.get(
                    "phase_reference_wavelength_index",
                    np.argmin(np.abs(wavelengths_m - float(chip_payload["wl_design_um"]) * 1e-6)),
                ))
                if input_phase_correction_applied
                else None
            ),
        )
        if static_field_factors is None:
            planet_scales_per_arm: tuple[NDArray[np.complex128] | None, ...] = (
                (planet_transport_scales,) * int(unitary_stacks.shape[0])
            )
        else:
            delay_scales = (
                np.ones((1, len(wavelengths_m), n_modes), dtype=np.complex128)
                if planet_transport_scales is None
                else planet_transport_scales
            )
            planet_scales_per_arm = tuple(
                delay_scales * arm_factor[np.newaxis] for arm_factor in static_field_factors
            )
        # The planet sees the same piston as the star (review 2026-11-07
        # row 36): average its dark power over the MC's piston covariance.
        mode_piston_covariance_m2 = _mode_piston_covariance_m2(
            sigma_opd_m=sigma_OPD_m,
            covariance_m2=piston_covariance_m2,
            mode_group_indices=mode_group_indices,
            n_modes=n_modes,
        )

        def planet_throughput_per_arm(seps_rad: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.stack(
                [
                    compute_planet_throughput_vs_separation(
                        u_stack=arm_stack,
                        wavelengths_m=wavelengths_m,
                        positions_m=positions_m,
                        dark_indices=dark_indices,
                        coupling_per_sub=coupling_per_sub,
                        sub_areas_m2=sub_areas_m2,
                        sub_diameters_m=(
                            sub_diameters_m if params.include_angular_coupling else None
                        ),
                        separations_rad=seps_rad,
                        n_pa_samples=params.n_pa_samples,
                        input_transport_scales=arm_scales,
                        wavelength_weights=wavelength_weights,
                        piston_covariance_m2=mode_piston_covariance_m2,
                    )
                    for arm_stack, arm_scales in zip(
                        unitary_stacks, planet_scales_per_arm, strict=True,
                    )
                ],
                axis=0,
            )

        def arm_averaged_throughput(seps_rad: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.average(
                planet_throughput_per_arm(seps_rad), axis=0, weights=unitary_arm_weights,
            )

        throughput_per_arm_before_pointing_jitter = planet_throughput_per_arm(
            separations_rad
        )
        throughput_per_arm = _angular_jitter_convolve_profile(
            throughput_per_arm_before_pointing_jitter,
            separations_mas=separations_mas,
            sigma_mas=pointing_jitter_rms_mas,
        )
        throughputs = np.average(
            throughput_per_arm,
            axis=0,
            weights=unitary_arm_weights,
        )

        # Tier 2.3b/2.5b: optionally derive σ_I and f_eff from physics
        # instead of using the manual scalar / override.
        sigma_I_per_sub, amp_servo_diagnostics = _resolve_scintillation_index(
            state=state,
            params=params,
            wavelength_m=float(wavelengths_m[len(wavelengths_m) // 2]),
            sub_diameter_m=float(sub_diameters_m[0]),
        )
        stellar_field = stellar_input_field(coupling_per_sub, sub_areas_m2)
        per_az_field = _per_azimuth_stellar_fields(
            state=state,
            base_field=stellar_field,
            n_azimuths=int(params.n_azimuths),
            n_apertures=n_delay_apertures,
        )
        if static_field_factors is None:
            stellar_fields_per_arm = (per_az_field,) * int(unitary_stacks.shape[0])
        else:
            per_az_field_wl = (
                per_az_field
                if per_az_field.ndim == 3
                else np.broadcast_to(
                    per_az_field[:, np.newaxis, :],
                    (per_az_field.shape[0], len(wavelengths_m), n_modes),
                )
            )
            stellar_fields_per_arm = tuple(
                per_az_field_wl * arm_factor[np.newaxis] for arm_factor in static_field_factors
            )
        # Phase 1 (KILO transport.py L1739): n_realizations_per_azimuth
        # is the per-azimuth budget; if unset, the back-compat alias
        # n_mc_realizations is divided across the n_azimuths slots so
        # the total sample count stays comparable to the pre-Phase-1
        # single-azimuth result.
        if params.n_realizations_per_azimuth is not None:
            n_per_az = int(params.n_realizations_per_azimuth)
        else:
            n_per_az = max(
                int(params.n_mc_realizations) // int(params.n_azimuths), 1
            )
        has_mc_perturbation = (
            sigma_OPD_m > 0.0
            or piston_covariance_m2 is not None
            or np.any(np.asarray(sigma_I_per_sub, dtype=np.float64) > 0.0)
            or float(params.fiber_phase_noise_rms_rad) > 0.0
        )
        mc_kwargs: dict[str, Any] = {
            "dark_indices": dark_indices,
            "ao_piston_opd_rms_per_sub_m": sigma_OPD_m,
            "ao_piston_opd_covariance_m2": piston_covariance_m2,
            "wavelengths_m": wavelengths_m,
            "n_realizations_per_azimuth": n_per_az,
            "seed": params.mc_seed,
            "scintillation_index_per_sub": sigma_I_per_sub,
            "mode_group_indices": mode_group_indices,
            "fiber_phase_noise_rms_rad": float(params.fiber_phase_noise_rms_rad),
            "wavelength_weights": wavelength_weights,
            "return_dark_fields": True,
        }

        def run_mc(**overrides: Any) -> tuple[tuple[RealizedNullResult, ...], RealizedNullResult]:
            per_arm = tuple(
                realized_null_depth_mc(
                    u_stack=arm_stack, stellar_field=arm_field, **{**mc_kwargs, **overrides},
                )
                for arm_stack, arm_field in zip(
                    unitary_stacks, stellar_fields_per_arm, strict=True,
                )
            )
            return per_arm, _combine_realized_null_results(
                per_arm, unitary_arm_weights, wavelength_weights,
            )

        run_realized_mc = n_per_az > 0 and has_mc_perturbation
        if run_realized_mc:
            mc_per_arm, mc = run_mc()
        else:
            mc_per_arm = tuple(
                _ideal_realized_null_for_stack(
                    u_stack=arm_stack,
                    stellar_field=arm_field,
                    dark_indices=dark_indices,
                    wavelength_weights=wavelength_weights,
                )
                for arm_stack, arm_field in zip(
                    unitary_stacks, stellar_fields_per_arm, strict=True,
                )
            )
            mc = _combine_realized_null_results(
                mc_per_arm, unitary_arm_weights, wavelength_weights,
            )

        # Stellar disk: the realized null already holds the on-axis T(0), so
        # only the resolved-disk excess <T(r) - T(0)> is added (review
        # 2026-11-07 Issue 17). T is evaluated on its own quadrature nodes
        # inside the stellar radius. With pointing jitter the disk sees the
        # smoothed T, while T(0) stays the unsmoothed value the MC contains,
        # so the jitter leakage of the star is kept.
        stellar_diameter_rad = 0.0
        if state.scene.star_angular_diameter_mas is not None:
            stellar_diameter_rad = float(state.scene.star_angular_diameter_mas) * _RAD_PER_MAS
        if pointing_jitter_rms_mas > 0.0:
            jitter_mas = float(pointing_jitter_rms_mas)

            def disk_throughput(r_rad: NDArray[np.float64]) -> NDArray[np.float64]:
                r_mas = np.asarray(r_rad, dtype=np.float64) / _RAD_PER_MAS
                grid_mas = np.unique(np.concatenate([
                    r_mas,
                    np.arange(0.0, float(np.max(r_mas)) + 6.0 * jitter_mas, jitter_mas / 8.0),
                ]))
                smoothed = _angular_jitter_convolve_profile(
                    arm_averaged_throughput(grid_mas * _RAD_PER_MAS),
                    separations_mas=grid_mas,
                    sigma_mas=jitter_mas,
                )
                return np.interp(r_mas, grid_mas, smoothed)

            on_axis_throughput: float | None = float(
                arm_averaged_throughput(np.zeros(1, dtype=np.float64))[0]
            )
        else:
            disk_throughput = arm_averaged_throughput
            on_axis_throughput = None
        disk_floor = stellar_disk_null_floor(
            disk_throughput,
            stellar_diameter_rad,
            on_axis_throughput=on_axis_throughput,
        )
        wavelength_center_m = float(wavelengths_m[len(wavelengths_m) // 2])
        base_effective_null = float(mc.mean_null_depth) + float(disk_floor)

        # When ``effective_null_override`` is set, replace only the AO-MC +
        # disk-floor base null with the supplied calibration-limited floor.
        # Residual-starlight profiles are applied below on top of this base
        # using their configured combine mode.
        if params.effective_null_override is not None:
            base_effective_null = float(params.effective_null_override)
            effective_null_scalar_source = "effective_null_override"
        else:
            effective_null_scalar_source = "chip_disk_scalar"

        if params.auto_setpoint_trim:
            trim_stellar_rate_hz, _ = _detector_stellar_and_static_leakage_rates_hz(
                state
            )
            if trim_stellar_rate_hz is None or trim_stellar_rate_hz <= 0.0:
                raise ValueError(
                    "sensitivity.detection_curve auto_setpoint_trim needs an "
                    "upstream detector.* component for the stellar rate; add a "
                    "detector stage or set auto_setpoint_trim = false."
                )
            n_trim_modes = (
                int(params.n_trim_modes)
                if params.n_trim_modes is not None
                else int(params.n_dark_ports) + int(params.n_spectro_ports)
            )
            # The 1f lock-in fights shot noise from EVERYTHING on the dark
            # ports: the pre-trim stellar floor, the sky background, and
            # the trim's own added leakage. The one-shot optimum used only
            # the pre-trim floor, understating the converged faint-end
            # budget by up to ~sqrt(2) (review 2026-06-10 Issue 27e).
            # Iterate to the fixed point (contraction: L ~ sqrt(N_f)).
            background_null_equiv = (
                _staged_background_rate_hz_per_dark_port(state)
                * float(n_trim_modes)
                / float(trim_stellar_rate_hz)
            )
            lockin_floor_null = base_effective_null + background_null_equiv
            trim_budget = photon_limited_setpoint_trim(
                stellar_rate_hz=trim_stellar_rate_hz,
                dark_floor_null=lockin_floor_null,
                trim_loop_bandwidth_hz=float(params.trim_loop_bandwidth_hz),
                n_trim_modes=n_trim_modes,
                wavelength_m=wavelength_center_m,
            )
            for _trim_iter in range(8):
                converged_floor = (
                    base_effective_null
                    + background_null_equiv
                    + float(trim_budget.total_leakage)
                )
                next_budget = photon_limited_setpoint_trim(
                    stellar_rate_hz=trim_stellar_rate_hz,
                    dark_floor_null=converged_floor,
                    trim_loop_bandwidth_hz=float(params.trim_loop_bandwidth_hz),
                    n_trim_modes=n_trim_modes,
                    wavelength_m=wavelength_center_m,
                )
                delta = abs(
                    float(next_budget.total_leakage)
                    - float(trim_budget.total_leakage)
                )
                trim_budget = next_budget
                if delta <= 1.0e-3 * max(float(next_budget.total_leakage), 1e-30):
                    break
            null_setpoint_leakage = float(trim_budget.bias_leakage)
            setpoint_dither_leakage = float(trim_budget.dither_leakage)
            # Equivalent single-offset OPD for the n-mode bias budget.
            setpoint_residual_nm_out = float(
                math.sqrt(n_trim_modes) * trim_budget.setpoint_residual_nm_per_mode
            )
            setpoint_dither_rms_nm_out = float(trim_budget.dither_rms_nm)
        else:
            n_trim_modes = None
            null_setpoint_leakage = _opd_leakage_fraction(
                float(params.null_setpoint_residual_nm),
                wavelength_center_m,
            )
            setpoint_dither_leakage = _opd_leakage_fraction(
                float(params.setpoint_dither_rms_nm),
                wavelength_center_m,
            )
            setpoint_residual_nm_out = float(params.null_setpoint_residual_nm)
            setpoint_dither_rms_nm_out = float(params.setpoint_dither_rms_nm)

        effective_null = (
            base_effective_null
            + null_setpoint_leakage
            + setpoint_dither_leakage
        )
        effective_null_before_residual_starlight = float(effective_null)
        (
            residual_starlight_profile_before_pointing_jitter,
            residual_starlight_diagnostics,
        ) = _residual_starlight_profile_for_grid(
            params=params,
            separations_mas=separations_mas,
            wavelength_grid_um=np.asarray(chip_wavelengths_um, dtype=np.float64),
            wavelength_weights=wavelength_weights,
            config_dir=_config_dir_from_context(self.context),
        )
        residual_starlight_profile = _angular_jitter_convolve_profile(
            residual_starlight_profile_before_pointing_jitter,
            separations_mas=separations_mas,
            sigma_mas=pointing_jitter_rms_mas,
        )
        residual_surface_before_pointing_jitter = (
            residual_starlight_diagnostics.get("contrast_by_wavelength")
        )
        if residual_surface_before_pointing_jitter is None:
            residual_surface = None
        else:
            surface_before = np.asarray(
                residual_surface_before_pointing_jitter,
                dtype=np.float64,
            )
            residual_surface = _angular_jitter_convolve_profile(
                surface_before.T,
                separations_mas=separations_mas,
                sigma_mas=pointing_jitter_rms_mas,
            ).T
        if residual_starlight_diagnostics["applied"]:
            effective_null_by_separation = _apply_residual_starlight_to_effective_null(
                base_effective_null=effective_null_before_residual_starlight,
                residual_profile=residual_starlight_profile,
                combine_mode=params.residual_starlight_combine_mode,
            )
            effective_null, effective_null_scalar_source = (
                _effective_null_scalar_from_curve(
                    effective_null_curve=effective_null_by_separation,
                    separations_mas=separations_mas,
                    scene_planet_separation_mas=state.scene.planet_separation_mas,
                )
            )
        else:
            effective_null_by_separation = np.full_like(
                separations_mas,
                effective_null_before_residual_starlight,
                dtype=np.float64,
            )

        # Null fluctuation (review 2026-11-07 Issue 19). The MC gives the
        # random part sigma_MC of the band leakage. The setpoint residual and
        # dither are modeled as a static phase offset, i.i.d. across chip
        # input modes, whose dark field beats linearly with each draw's
        # static-subtracted dark field (phase and amplitude, as the output
        # rows couple them); see bias_cross_moments. Pol arms are separate
        # chips with independent trim offsets, so their cross terms add with
        # squared flux weights. The disk floor is incoherent with the on-axis
        # star and adds no variance.
        bias_leakage = float(null_setpoint_leakage) + float(setpoint_dither_leakage)
        null_instability_mc = float(mc.null_instability)

        def arm_bias_cross(per_arm: Sequence[RealizedNullResult]) -> float:
            moments = [bias_cross_moments(r, wavelength_weights) for r in per_arm]
            s_moment = sum(
                float(w) ** 2 * m[0]
                for w, m in zip(unitary_arm_weights, moments, strict=True)
            )
            d_moment = sum(
                float(w) * m[1]
                for w, m in zip(unitary_arm_weights, moments, strict=True)
            )
            return 4.0 * bias_leakage * s_moment / d_moment if d_moment > 0.0 else 0.0

        bias_active = run_realized_mc and bias_leakage > 0.0
        bias_cross = arm_bias_cross(mc_per_arm) if bias_active else 0.0
        null_fluctuation_rms = math.sqrt(
            float(mc.null_fluctuation_rms) ** 2 + bias_cross
        )
        null_instability_out = (
            null_fluctuation_rms / float(effective_null) if effective_null > 0.0 else 0.0
        )

        # Resolve the f_eff for the performance stage's systematic-noise
        # term. Precedence:
        #   (1) explicit effective_systematic_bandwidth_hz_override,
        #   (2) leakage-variance-weighted blend when
        #       auto_amplitude_servo_residual is enabled (Tier 2.5b); the leg
        #       variances come from MC reruns with one perturbation each
        #       (same seeds, so the draws are those of the full MC); the bias
        #       cross term is split the same way, each part computed from its
        #       own leg's draws and averaging at that leg's bandwidth,
        #   (3) None (performance stage uses its own default).
        piston_leakage_variance = 0.0
        amplitude_leakage_variance = 0.0
        if (
            run_realized_mc
            and params.effective_systematic_bandwidth_hz_override is None
            and params.auto_amplitude_servo_residual
        ):
            piston_arms, piston_only = run_mc(
                scintillation_index_per_sub=0.0, fiber_phase_noise_rms_rad=0.0,
            )
            amplitude_arms, amplitude_only = run_mc(
                ao_piston_opd_rms_per_sub_m=0.0,
                ao_piston_opd_covariance_m2=None,
                fiber_phase_noise_rms_rad=0.0,
            )
            piston_leakage_variance = float(piston_only.null_fluctuation_rms) ** 2 + (
                arm_bias_cross(piston_arms) if bias_active else 0.0
            )
            amplitude_leakage_variance = float(
                amplitude_only.null_fluctuation_rms
            ) ** 2 + (arm_bias_cross(amplitude_arms) if bias_active else 0.0)
        resolved_amp_bw_hz = (
            None
            if amp_servo_diagnostics is None
            else _positive_finite_float_or_none(
                amp_servo_diagnostics.get("injected_amplitude_bandwidth_hz")
            )
        )
        sigma_phi_piston_rad = (
            sigma_OPD_m * (2.0 * math.pi)
            / float(wavelengths_m[len(wavelengths_m) // 2])
        )
        resolved_f_piston_hz = _ft_published_f_3db_hz(state)
        effective_bandwidth_hz = _resolve_effective_bandwidth_hz(
            params=params,
            piston_leakage_variance=piston_leakage_variance,
            amplitude_leakage_variance=amplitude_leakage_variance,
            resolved_amp_servo_bandwidth_hz=resolved_amp_bw_hz,
            resolved_f_piston_hz=resolved_f_piston_hz,
        )
        effective_bandwidth_diagnostics = _effective_bandwidth_diagnostics(
            params=params,
            sigma_phi_piston_rad=sigma_phi_piston_rad,
            sigma_I_residual=sigma_I_per_sub,
            piston_leakage_variance=piston_leakage_variance,
            amplitude_leakage_variance=amplitude_leakage_variance,
            piston_covariance_m2=piston_covariance_m2,
            resolved_amp_servo_bandwidth_hz=resolved_amp_bw_hz,
            effective_bandwidth_hz=effective_bandwidth_hz,
            resolved_f_piston_hz=resolved_f_piston_hz,
        )

        # Publish the realized leakage rate (effective null x total stellar
        # rate, the normalization performance.standard uses) alongside the
        # static-chip dark-port total the detector stage reported. The two
        # are different physical truths: the detector total is the static
        # chip null; the effective rate carries the AO / disk / setpoint
        # realization. SNR uses the effective value.
        (
            total_stellar_rate_hz_for_leakage,
            static_chip_leakage_hz,
        ) = _detector_stellar_and_static_leakage_rates_hz(state)
        if total_stellar_rate_hz_for_leakage is None:
            leakage_rate_hz_effective: float | None = None
        else:
            leakage_rate_hz_effective = float(
                effective_null * total_stellar_rate_hz_for_leakage
            )
        if (
            leakage_rate_hz_effective is None
            or static_chip_leakage_hz is None
            or static_chip_leakage_hz <= 0.0
        ):
            effective_to_static_leakage_ratio: float | None = None
        else:
            effective_to_static_leakage_ratio = float(
                leakage_rate_hz_effective / static_chip_leakage_hz
            )

        component_name = unique_component_name(
            self.type_name, state.results.components, self.context.instance_id,
        )
        payload: dict[str, Any] = {
            "separations_mas": tuple(float(s) for s in separations_mas),
            "planet_throughputs": throughputs.astype(np.float64, copy=True),
            "planet_throughputs_per_arm": throughput_per_arm.astype(
                np.float64, copy=True,
            ),
            "planet_throughputs_before_pointing_jitter": (
                np.average(
                    throughput_per_arm_before_pointing_jitter,
                    axis=0,
                    weights=unitary_arm_weights,
                ).astype(np.float64, copy=True)
            ),
            "planet_throughputs_per_arm_before_pointing_jitter": (
                throughput_per_arm_before_pointing_jitter.astype(
                    np.float64,
                    copy=True,
                )
            ),
            "planet_delay_transport_applied": planet_transport_scales is not None,
            "static_field_factor_sources": static_field_factor_sources,
            "wavelengths_m": tuple(float(w) for w in wavelengths_m),
            "photon_wavelength_weights": tuple(float(w) for w in wavelength_weights),
            "photon_wavelength_weights_source": wavelength_weights_source,
            "planet_throughput_piston_averaged": mode_piston_covariance_m2 is not None,
            "n_pa_samples": int(params.n_pa_samples),
            "include_angular_coupling": bool(params.include_angular_coupling),
            "pointing_jitter_rms_mas": float(pointing_jitter_rms_mas),
            "pointing_jitter_source": pointing_jitter_source,
            "pointing_jitter_applied": bool(pointing_jitter_rms_mas > 0.0),
            "coupling_per_sub": coupling_per_sub.astype(np.float64, copy=True),
            "coupling_source": coupling_source,
            "sub_areas_m2": sub_areas_m2.astype(np.float64, copy=True),
            "sub_diameters_m": sub_diameters_m.astype(np.float64, copy=True),
            "dark_indices": dark_indices.astype(np.intp, copy=True),
            "chip_n_input_modes": int(n_modes),
            "chip_n_output_ports": int(n_output_ports),
            "n_bright_ports": int(params.n_bright_ports),
            "n_dark_ports": int(params.n_dark_ports),
            "n_spectro_ports": int(params.n_spectro_ports),
            "ao_piston_opd_rms_per_sub_m": float(sigma_OPD_m),
            "ao_piston_opd_covariance_m2": (
                None
                if piston_covariance_m2 is None
                else piston_covariance_m2.astype(np.float64, copy=True)
            ),
            "ao_piston_residual": piston_residual_diagnostics,
            "stellar_diameter_mas": (
                None if state.scene.star_angular_diameter_mas is None
                else float(state.scene.star_angular_diameter_mas)
            ),
            "disk_null_floor": float(disk_floor),
            # Realized leakage rate (effective null x total stellar rate)
            # vs the static-chip dark-port total from the detector stage.
            "leakage_rate_hz_effective": leakage_rate_hz_effective,
            "leakage_rate_hz_static_chip": static_chip_leakage_hz,
            "effective_to_static_leakage_ratio": effective_to_static_leakage_ratio,
            "null_setpoint_leakage": float(null_setpoint_leakage),
            "setpoint_dither_leakage": float(setpoint_dither_leakage),
            "setpoint_trim_lockin_floor_model": (
                "fixed-point: base null + staged background + own trim "
                "leakage feed the 1f lock-in shot noise"
            ),
            "null_setpoint_residual_nm": float(setpoint_residual_nm_out),
            "setpoint_dither_rms_nm": float(setpoint_dither_rms_nm_out),
            "auto_setpoint_trim": bool(params.auto_setpoint_trim),
            "trim_loop_bandwidth_hz": float(params.trim_loop_bandwidth_hz),
            "n_trim_modes": None if n_trim_modes is None else int(n_trim_modes),
            "realized_null_depth": float(mc.mean_null_depth),
            "realized_null_depth_per_arm": tuple(
                float(arm_result.mean_null_depth) for arm_result in mc_per_arm
            ),
            "realized_null_median": float(mc.median_null_depth),
            "realized_null_std": float(mc.std_null_depth),
            "null_instability": null_instability_out,
            "null_instability_mc": null_instability_mc,
            "null_fluctuation_rms": float(null_fluctuation_rms),
            "null_fluctuation_rms_mc": float(mc.null_fluctuation_rms),
            "null_fluctuation_bias_leakage": float(bias_leakage),
            "null_fluctuation_bias_cross_variance": float(bias_cross),
            "ideal_chip_null_depth": float(mc.ideal_null_depth),
            "ideal_chip_null_depth_per_arm": tuple(
                float(arm_result.ideal_null_depth) for arm_result in mc_per_arm
            ),
            # KILO-convention chip-only null (mean(dark)/mean(bright)) —
            # directly comparable to KILO ``multiband.J.null_depth_dark``
            # and to MKID's ``null_depth_raw``. ``ideal_chip_null_depth``
            # above is the dark-power-fraction convention the downstream
            # contrast pipeline ingests; the two differ by ~n_dark/n_bright
            # (~3x on Keck) for a deep null.
            "ideal_chip_null_depth_kilo": float(mc.ideal_null_depth_kilo),
            # AO-realized null in KILO convention. Comparable to KILO's
            # paperv3 ``null_depth_dark`` reference and to KILO's
            # ``null_val_dark`` cached at ``warm_start_keck_only_j.npz``
            # (~9e-5 at J=4 Keck).
            "realized_null_depth_kilo": float(mc.mean_null_depth_kilo),
            "effective_null_with_disk": float(effective_null),
            "effective_null_before_residual_starlight": (
                float(effective_null_before_residual_starlight)
            ),
            "effective_null_with_disk_by_separation": (
                effective_null_by_separation.astype(np.float64, copy=True)
            ),
            "effective_null_scalar_source": effective_null_scalar_source,
            "residual_starlight_applied": bool(
                residual_starlight_diagnostics["applied"]
            ),
            "residual_starlight_contrast": residual_starlight_profile.astype(
                np.float64,
                copy=True,
            ),
            "residual_starlight_contrast_before_pointing_jitter": (
                residual_starlight_profile_before_pointing_jitter.astype(
                    np.float64,
                    copy=True,
                )
            ),
            "residual_starlight_wavelengths_um": residual_starlight_diagnostics[
                "wavelengths_um"
            ],
            "residual_starlight_wavelength_weights": residual_starlight_diagnostics[
                "wavelength_weights"
            ],
            "residual_starlight_contrast_by_wavelength": (
                None if residual_surface is None
                else residual_surface.astype(np.float64, copy=True)
            ),
            "residual_starlight_contrast_by_wavelength_before_pointing_jitter": (
                None if residual_surface_before_pointing_jitter is None
                else np.asarray(
                    residual_surface_before_pointing_jitter,
                    dtype=np.float64,
                ).copy()
            ),
            "residual_starlight_profile_source": residual_starlight_diagnostics[
                "source"
            ],
            "residual_starlight_table_path": residual_starlight_diagnostics[
                "table_path"
            ],
            "residual_starlight_combine_mode": residual_starlight_diagnostics[
                "combine_mode"
            ],
            "residual_starlight_interpolation": residual_starlight_diagnostics.get(
                "interpolation",
            ),
            "scintillation_index_per_sub": float(sigma_I_per_sub),
            "realized_null_mode_group_source": mode_group_source,
            "realized_null_mode_group_count": (
                None if mode_group_indices is None
                else int(np.unique(mode_group_indices).size)
            ),
            "realized_null_mode_group_indices": (
                None
                if mode_group_indices is None
                else mode_group_indices.astype(np.intp, copy=True)
            ),
            "amplitude_servo_residual": amp_servo_diagnostics,
            "n_azimuths": int(mc.n_azimuths),
            "n_realizations_per_azimuth": int(mc.n_realizations_per_azimuth),
            "mc_seed": None if params.mc_seed is None else int(params.mc_seed),
            "fiber_phase_noise_rms_rad": float(params.fiber_phase_noise_rms_rad),
            # f_eff for the performance stage's systematic-noise term.
            # None preserves the performance stage's own default.
            "effective_systematic_bandwidth_hz": (
                None if effective_bandwidth_hz is None else float(effective_bandwidth_hz)
            ),
            "effective_systematic_bandwidth_diagnostics": (
                effective_bandwidth_diagnostics
            ),
            "n_mc_samples": int(mc.n_samples),
            "chip_component": chip_name,
            "chip_deployment_mode": chip_deployment_mode,
            "input_phase_correction_applied": bool(input_phase_correction_applied),
            "chip_channel_labels": unitary_labels,
            "unitary_arm_weights": unitary_arm_weights.astype(
                np.float64, copy=True,
            ),
            "source_model": (
                "Port of KILO performance.compute_planet_throughput_vs_separation "
                "(lines 1010-1128) and azimuth_averaged_realized_null_depth "
                "(transport.py lines 1739-1883), with photon-weighted band "
                "statistics. The finite-star term follows the Guyon et al. "
                "(2013) nuller model: each sky point is propagated coherently "
                "through U, and intensities are integrated incoherently over "
                "the stellar disk; only the excess over the on-axis leakage "
                "is added to the realized null."
            ),
        }
        results = state.results.add_component(component_name, payload)
        return replace(state, results=results)
