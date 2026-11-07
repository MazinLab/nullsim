"""Populate pupil-frame mode positions for monolithic-pupil surrogates."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from nullsim.physics.pupil_geometry import optimize_subaperture_positions
from nullsim.pipeline import BaseStage, Geometry, SimulationState, register_stage


@register_stage
class PupilModes(BaseStage):
    """Place photonic input modes across a single circular pupil.

    This is a geometry bridge for first-pass single-primary configs, such as
    HWO, that intentionally do not run an AO/sub-aperture stage but still need
    per-mode pupil coordinates for separation-dependent planet throughput.
    It accepts exactly one upstream telescope; multi-telescope arrays get
    their per-telescope modes from ``ao.pyramid_wfs`` sub-aperture packing.
    """

    type_name = "telescope.pupil_modes"
    family = "telescope"
    consumes = frozenset({"geometry"})
    produces = frozenset({"geometry"})

    class Params(BaseModel):
        """Validated params for monolithic-pupil mode placement."""

        model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

        n_modes: int = Field(ge=1)
        layout: Literal[
            "linear",
            "packed_circular",
            "packed_jittered",
            "nonredundant_circular",
        ] = "linear"
        span_fraction: float = Field(
            default=1.0,
            gt=0.0,
            le=1.0,
            description=(
                "Fraction of telescope diameter spanned by the outermost "
                "mode centers. 1.0 places a two-mode surrogate at +/- D/2."
            ),
        )
        normalize_max_baseline_to_span: bool = Field(
            default=False,
            description=(
                "For packed layouts, rescale the mode-center coordinates so "
                "the largest center-to-center baseline equals span_fraction "
                "times the telescope diameter."
            ),
        )
        preserve_collecting_area: bool = Field(
            default=False,
            description=(
                "Keep the upstream effective_collecting_area_m2 instead of "
                "replacing it with the packed mode-circle area."
            ),
        )
        hole_diameter_scale: float | None = Field(
            default=None,
            gt=0.0,
            le=1.0,
            description=(
                "Optional scale applied to the densest packed hole diameter. "
                "If unset, symmetry-broken layouts use conservative defaults "
                "that trade a little fill fraction for placement freedom."
            ),
        )

    def apply(self, state: SimulationState) -> SimulationState:
        params = self.params_model
        diameter_m = _require_diameter_m(state)
        n_apertures = int(state.geometry.n_apertures or 1)
        if n_apertures > 1:
            raise ValueError(
                "telescope.pupil_modes places modes inside one monolithic pupil, "
                f"but the upstream geometry has n_apertures = {n_apertures}. "
                "Use a single telescope, or place per-telescope modes with "
                "ao.pyramid_wfs n_subapertures_per_aperture."
            )
        span_m = float(params.span_fraction) * diameter_m
        positions, mode_diameter_m = _positions_for_layout(params, span_m)
        geometry = _updated_geometry(
            state.geometry,
            positions,
            params,
            span_m,
            mode_diameter_m,
        )
        return replace(state, geometry=geometry)


def _require_diameter_m(state: SimulationState) -> float:
    diameter_m = state.geometry.diameter_m
    if diameter_m is None or float(diameter_m) <= 0.0:
        raise ValueError(
            "telescope.pupil_modes requires geometry.diameter_m from an "
            "upstream telescope.fixed_aperture stage."
        )
    return float(diameter_m)


def _linear_positions(n_modes: int, span_m: float) -> NDArray[np.float64]:
    positions = np.zeros((n_modes, 2), dtype=np.float64)
    if n_modes == 1:
        return positions
    positions[:, 0] = np.linspace(-span_m / 2.0, span_m / 2.0, n_modes)
    return positions


def _positions_for_layout(
    params: PupilModes.Params,
    span_m: float,
) -> tuple[NDArray[np.float64], float | None]:
    if params.layout == "linear":
        return _linear_positions(params.n_modes, span_m), None
    positions, mode_diameter_m = _packed_positions(params.n_modes, span_m)
    packed_mode_diameter_m = float(mode_diameter_m)
    hole_diameter_scale = _hole_diameter_scale(params)
    mode_diameter_m = packed_mode_diameter_m * hole_diameter_scale
    if params.layout == "packed_jittered":
        positions = _jitter_packed_positions(
            np.asarray(positions, dtype=np.float64),
            span_m=span_m,
            mode_diameter_m=mode_diameter_m,
            jitter_fraction=0.12,
            n_passes=1,
            seed_salt=1,
        )
    elif params.layout == "nonredundant_circular":
        positions = _nonredundant_circular_positions(
            np.asarray(positions, dtype=np.float64),
            params.n_modes,
            span_m=span_m,
            mode_diameter_m=mode_diameter_m,
        )
    elif params.layout != "packed_circular":
        raise ValueError(f"Unknown pupil_modes layout {params.layout!r}.")
    positions_array = np.asarray(positions, dtype=np.float64)
    if params.normalize_max_baseline_to_span:
        max_baseline = _max_pairwise_baseline(positions_array)
        if max_baseline <= 0.0:
            raise ValueError(
                "telescope.pupil_modes normalize_max_baseline_to_span requires "
                "at least two distinct mode centers."
            )
        scale = float(span_m) / max_baseline
        positions_array = positions_array * scale
        mode_diameter_m = float(mode_diameter_m) * scale
    return positions_array, float(mode_diameter_m)


def _hole_diameter_scale(params: PupilModes.Params) -> float:
    if params.hole_diameter_scale is not None:
        return float(params.hole_diameter_scale)
    if params.layout == "packed_jittered":
        return 0.95
    if params.layout == "nonredundant_circular":
        return 0.98
    return 1.0


def _packed_positions(
    n_modes: int,
    span_m: float,
) -> tuple[NDArray[np.float64], float]:
    positions, mode_diameter_m = optimize_subaperture_positions(
        n_modes,
        span_m,
        pupil=None,
    )
    return np.asarray(positions, dtype=np.float64), float(mode_diameter_m)


def _jitter_packed_positions(
    positions: NDArray[np.float64],
    *,
    span_m: float,
    mode_diameter_m: float,
    jitter_fraction: float,
    n_passes: int,
    seed_salt: int,
) -> NDArray[np.float64]:
    """Break packing symmetries without moving circles outside the span disk."""

    rng = np.random.default_rng(30_031 + 97 * positions.shape[0] + int(seed_salt))
    out = positions.astype(np.float64, copy=True)
    max_center_radius = max(0.0, 0.5 * float(span_m) - 0.5 * float(mode_diameter_m))
    min_separation = float(mode_diameter_m) * (1.0 - 1.0e-10)
    jitter_radius = float(jitter_fraction) * float(mode_diameter_m)

    for pass_index in range(int(n_passes)):
        pass_radius = jitter_radius / np.sqrt(pass_index + 1.0)
        for idx in rng.permutation(out.shape[0]):
            original = out[idx].copy()
            for _attempt in range(160):
                angle = rng.uniform(0.0, 2.0 * np.pi)
                radius = pass_radius * np.sqrt(rng.uniform(0.0, 1.0))
                candidate = original + radius * np.array([np.cos(angle), np.sin(angle)])
                if np.linalg.norm(candidate) > max_center_radius:
                    continue
                trial = out.copy()
                trial[idx] = candidate
                separations = np.sqrt(np.sum((trial - candidate) ** 2, axis=1))
                separations[idx] = np.inf
                if float(np.min(separations)) < min_separation:
                    continue
                out[idx] = candidate
                break
    return out


def _nonredundant_circular_positions(
    packed_positions: NDArray[np.float64],
    n_modes: int,
    *,
    span_m: float,
    mode_diameter_m: float,
) -> NDArray[np.float64]:
    """Deterministically perturb a full-aperture packing to break symmetries."""

    if n_modes == 1:
        return np.zeros((1, 2), dtype=np.float64)
    return _jitter_packed_positions(
        np.asarray(packed_positions, dtype=np.float64),
        span_m=span_m,
        mode_diameter_m=mode_diameter_m,
        jitter_fraction=0.32,
        n_passes=6,
        seed_salt=19,
    )


def _updated_geometry(
    geometry: Geometry,
    positions: NDArray[np.float64],
    params: PupilModes.Params,
    span_m: float,
    mode_diameter_m: float | None,
) -> Geometry:
    baselines_m = _pairwise_baselines(positions)
    effective_area_m2 = (
        None
        if mode_diameter_m is None
        else math.pi * (float(mode_diameter_m) / 2.0) ** 2 * int(params.n_modes)
    )
    preserve_area = bool(params.preserve_collecting_area)
    effective_collecting_area_m2 = (
        geometry.effective_collecting_area_m2
        if preserve_area or effective_area_m2 is None
        else effective_area_m2
    )
    n_modes = int(params.n_modes)
    mode_area_m2 = (
        None
        if effective_collecting_area_m2 is None
        else float(effective_collecting_area_m2) / n_modes
    )
    return replace(
        geometry,
        aperture_positions_pupil_m=positions,
        baselines_m=baselines_m,
        effective_collecting_area_m2=effective_collecting_area_m2,
    ).add_component(
        "pupil_modes",
        {
            "n_modes": int(params.n_modes),
            "layout": params.layout,
            "span_fraction": float(params.span_fraction),
            "normalize_max_baseline_to_span": bool(
                params.normalize_max_baseline_to_span
            ),
            "preserve_collecting_area": preserve_area,
            "span_m": float(span_m),
            "mode_diameter_m": (
                None if mode_diameter_m is None else float(mode_diameter_m)
            ),
            "geometric_mode_area_m2": (
                None if effective_area_m2 is None else float(effective_area_m2)
            ),
            "effective_collecting_area_m2": (
                None
                if effective_collecting_area_m2 is None
                else float(effective_collecting_area_m2)
            ),
            "mode_area_m2": mode_area_m2,
            "effective_area_source": (
                "upstream" if preserve_area else "packed_mode_circles"
            ),
            # Per-mode vectors (length n_modes) so readers take the vector
            # path instead of an upstream telescope's scalar (review
            # 2026-11-07 Issue 1).
            "mode_areas_m2": (
                None
                if mode_area_m2 is None
                else np.full(n_modes, mode_area_m2, dtype=np.float64)
            ),
            "mode_diameters_m": (
                None
                if mode_diameter_m is None
                else np.full(n_modes, float(mode_diameter_m), dtype=np.float64)
            ),
            # Every mode sits in the one telescope at the pupil-frame origin,
            # so sky projection keeps the pupil offsets rigid (Issue 22).
            "mode_origin_positions_m": np.zeros((n_modes, 2), dtype=np.float64),
            "aperture_positions_pupil_m": positions.copy(),
            "baselines_m": baselines_m.copy(),
            "source_model": (
                f"{params.layout} surrogate mode placement across a monolithic circular "
                "pupil for first-pass space-telescope sensitivity curves."
            ),
        },
    )


def _pairwise_baselines(positions: NDArray[np.float64]) -> NDArray[np.float64]:
    if positions.shape[0] < 2:
        return np.empty(0, dtype=np.float64)
    diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=-1))
    upper = np.triu_indices(positions.shape[0], k=1)
    return np.asarray(dists[upper], dtype=np.float64)


def _max_pairwise_baseline(positions: NDArray[np.float64]) -> float:
    baselines = _pairwise_baselines(positions)
    if baselines.size == 0:
        return 0.0
    return float(np.max(baselines))


__all__ = ["PupilModes"]
