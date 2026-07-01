"""Simple space-telescope wavefront residual stage.

The stage records non-atmospheric optical stability terms for space configs:
static WFE, dynamic WFE, amplitude jitter, and line-of-sight pointing jitter.
It intentionally stops short of Fourier-optics propagation. The WFE terms are
converted to a wavelength-dependent Marechal Strehl multiplier and the pointing
jitter is published for downstream angular-convolution consumers.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field as PydanticField

from nullsim.pipeline import (
    BaseStage,
    OPDComponent,
    OPDBudget,
    SimulationState,
    Wavefront,
    register_stage,
)
from nullsim.stages._utils import unique_component_name

_PM_TO_M = 1.0e-12
_PM_TO_NM = 1.0e-3


@register_stage
class SpaceWavefrontResidual(BaseStage):
    """Publish a scalar space-wavefront residual budget."""

    type_name = "wavefront.space_residual"
    family = "wavefront"
    consumes = frozenset()
    produces = frozenset({"wavefront", "opd_budget"})
    # Combines its residual with the upstream ``wavefront`` (Strehl, amplitude
    # jitter) and ``opd_budget`` content, neither declared in ``consumes``.
    cache_key_extras = frozenset({"wavefront", "opd_budget"})

    class Params(BaseModel):
        """Validated params for the space wavefront residual stage."""

        model_config = ConfigDict(
            extra="forbid",
            frozen=True,
            allow_inf_nan=False,
        )

        static_wfe_rms_pm: float = PydanticField(default=0.0, ge=0.0)
        dynamic_wfe_rms_pm: float = PydanticField(default=0.0, ge=0.0)
        amplitude_jitter_fraction: float = PydanticField(default=0.0, ge=0.0)
        pointing_jitter_rms_mas: float = PydanticField(default=0.0, ge=0.0)

    def apply(self, state: SimulationState) -> SimulationState:
        params = self.params_model
        wavelength_grid_um = _require_wavelength_grid_um(state)
        wavelength_grid_m = np.asarray(wavelength_grid_um, dtype=np.float64) * 1.0e-6

        residual_wfe_pm = float(
            math.hypot(params.static_wfe_rms_pm, params.dynamic_wfe_rms_pm)
        )
        residual_strehl = _marechal_strehl(residual_wfe_pm, wavelength_grid_m)
        upstream_strehl = _upstream_strehl_or_unity(state, wavelength_grid_um)
        combined_strehl = upstream_strehl * residual_strehl

        residual_amp_jitter = np.full(
            len(wavelength_grid_um),
            float(params.amplitude_jitter_fraction),
            dtype=np.float64,
        )
        upstream_amp_jitter = _upstream_amplitude_jitter_or_zero(
            state,
            wavelength_grid_um,
        )
        combined_amp_jitter = np.hypot(upstream_amp_jitter, residual_amp_jitter)

        component_name = unique_component_name(
            self.type_name,
            state.wavefront.components,
            self.context.instance_id,
        )
        component = {
            "wavelength_grid_um": wavelength_grid_um,
            "static_wfe_rms_pm": float(params.static_wfe_rms_pm),
            "dynamic_wfe_rms_pm": float(params.dynamic_wfe_rms_pm),
            "total_wfe_rms_pm": residual_wfe_pm,
            "total_wfe_rms_nm": residual_wfe_pm * _PM_TO_NM,
            "residual_strehl_ratio": tuple(float(v) for v in residual_strehl),
            "upstream_strehl_ratio": tuple(float(v) for v in upstream_strehl),
            "strehl_ratio": tuple(float(v) for v in combined_strehl),
            "amplitude_jitter_fraction": tuple(float(v) for v in combined_amp_jitter),
            "residual_amplitude_jitter_fraction": tuple(
                float(v) for v in residual_amp_jitter
            ),
            "pointing_jitter_rms_mas": float(params.pointing_jitter_rms_mas),
            "source_model": (
                "Scalar space-WFE residual: Strehl = exp[-(2*pi*sigma_OPD/lambda)^2]; "
                "pointing_jitter_rms_mas is consumed by sensitivity.detection_curve "
                "as a 1-D angular convolution width."
            ),
        }

        wavefront = _updated_wavefront(
            state=state,
            wavelength_grid_um=wavelength_grid_um,
            strehl_ratio=combined_strehl,
            amplitude_jitter_fraction=combined_amp_jitter,
            component_name=component_name,
            component=component,
        )
        opd_budget = _updated_opd_budget(
            state=state,
            component_name=component_name,
            residual_wfe_rms_nm=residual_wfe_pm * _PM_TO_NM,
        )
        return replace(state, wavefront=wavefront, opd_budget=opd_budget)


def _require_wavelength_grid_um(state: SimulationState) -> tuple[float, ...]:
    wavelength_grid_um = (
        state.wavefront.wavelength_grid_um
        or state.field.wavelength_grid_um
        or state.throughput.wavelength_grid_um
        or state.photon_rates.wavelength_grid_um
    )
    if not wavelength_grid_um:
        raise ValueError(
            "wavefront.space_residual requires a non-empty wavelength grid "
            "from the resolved grid config."
        )
    return wavelength_grid_um


def _marechal_strehl(
    wfe_rms_pm: float,
    wavelength_grid_m: NDArray[np.float64],
) -> NDArray[np.float64]:
    phase_rms_rad = 2.0 * math.pi * float(wfe_rms_pm) * _PM_TO_M / wavelength_grid_m
    return np.exp(-(phase_rms_rad * phase_rms_rad)).astype(np.float64, copy=False)


def _upstream_strehl_or_unity(
    state: SimulationState,
    wavelength_grid_um: tuple[float, ...],
) -> NDArray[np.float64]:
    if not state.wavefront.strehl_ratio:
        return np.ones(len(wavelength_grid_um), dtype=np.float64)
    strehl = np.asarray(state.wavefront.strehl_ratio, dtype=np.float64)
    if strehl.shape != (len(wavelength_grid_um),):
        raise ValueError(
            "wavefront.space_residual: upstream wavefront.strehl_ratio "
            "must match wavelength_grid_um length."
        )
    if not np.all(np.isfinite(strehl)) or np.any(strehl < 0.0):
        raise ValueError(
            "wavefront.space_residual: upstream wavefront.strehl_ratio "
            "must be finite and non-negative."
        )
    return strehl


def _upstream_amplitude_jitter_or_zero(
    state: SimulationState,
    wavelength_grid_um: tuple[float, ...],
) -> NDArray[np.float64]:
    if not state.wavefront.amplitude_jitter_fraction:
        return np.zeros(len(wavelength_grid_um), dtype=np.float64)
    jitter = np.asarray(state.wavefront.amplitude_jitter_fraction, dtype=np.float64)
    if jitter.shape != (len(wavelength_grid_um),):
        raise ValueError(
            "wavefront.space_residual: upstream amplitude_jitter_fraction "
            "must match wavelength_grid_um length."
        )
    if not np.all(np.isfinite(jitter)) or np.any(jitter < 0.0):
        raise ValueError(
            "wavefront.space_residual: upstream amplitude_jitter_fraction "
            "must be finite and non-negative."
        )
    return jitter


def _updated_wavefront(
    *,
    state: SimulationState,
    wavelength_grid_um: tuple[float, ...],
    strehl_ratio: NDArray[np.float64],
    amplitude_jitter_fraction: NDArray[np.float64],
    component_name: str,
    component: dict[str, object],
) -> Wavefront:
    return replace(
        state.wavefront,
        wavelength_grid_um=wavelength_grid_um,
        strehl_ratio=tuple(float(v) for v in strehl_ratio),
        amplitude_jitter_fraction=tuple(float(v) for v in amplitude_jitter_fraction),
    ).add_component(component_name, component)


def _updated_opd_budget(
    *,
    state: SimulationState,
    component_name: str,
    residual_wfe_rms_nm: float,
) -> OPDBudget:
    total_rms_opd_nm = residual_wfe_rms_nm
    if state.opd_budget.total_rms_opd_nm is not None:
        total_rms_opd_nm = float(
            math.hypot(state.opd_budget.total_rms_opd_nm, residual_wfe_rms_nm)
        )
    return replace(
        state.opd_budget,
        total_rms_opd_nm=total_rms_opd_nm,
    ).add_component(
        component_name,
        OPDComponent(
            rms_opd_nm=residual_wfe_rms_nm,
            reduction_stage=SpaceWavefrontResidual.type_name,
        ),
    )


__all__ = ["SpaceWavefrontResidual"]
