"""Kolmogorov atmosphere stage backed by nullsim physics primitives.

This is the first SPEC §15.3 atmosphere-stage pass: it wraps the pure
``nullsim.physics.kolmogorov`` functions into the pipeline state model without
adding AO, scintillation, or finite-outer-scale physics. The delivered
wavefront/Strehl budget uses the piston-removed Noll coefficient because a
constant aperture piston does not degrade image quality (and the
piston-included Kolmogorov variance diverges). The zenith r0 is degraded to
the pointing airmass as ``r0 * airmass^(-3/5)`` (Roddier 1981), the same
scaling ``ao.pyramid_wfs`` applies. ``L0_m`` is accepted
and recorded for provenance, but the current primitive assumes infinite outer
scale. The Kolmogorov/Noll normalization and KILO/NAYRA provenance live in
``nullsim/physics/kolmogorov.py`` and ``tests/physics/notes/kolmogorov.md``.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from nullsim.physics import kolmogorov
from nullsim.pipeline import (
    BaseStage,
    OPDComponent,
    OPDBudget,
    SimulationState,
    Throughput,
    Wavefront,
    register_stage,
)
from nullsim.stages._utils import unique_component_name

_MICRON_TO_M = 1.0e-6
_M_TO_NM = 1.0e9
_RAD_TO_ARCSEC = 206_264.806_247_096_36


@register_stage
class KolmogorovScreens(BaseStage):
    """Populate atmosphere OPD and wavefront budgets from Kolmogorov seeing."""

    type_name = "atmosphere.kolmogorov"
    family = "atmosphere"
    consumes = frozenset({"geometry"})
    produces = frozenset({"opd_budget", "wavefront", "throughput"})
    # Opportunistic scene read for the Beer-Lambert airmass scaling of the
    # science-path transmission; zenith (airmass 1) when no scene pointing
    # exists. Cache-key extra, not an enforced consume, so configs without
    # a scene stage keep validating (review 2026-06-10 Issue 28a).
    cache_key_extras = frozenset({"scene"})

    class Params(BaseModel):
        """Validated params for the Kolmogorov screen atmosphere stage."""

        model_config = ConfigDict(
            extra="forbid",
            frozen=True,
            allow_inf_nan=False,
        )

        r0_500nm_m: float = Field(allow_inf_nan=True)
        seeing_arcsec: float | None = Field(default=None, gt=0.0)
        wind_speed_m_s: float | None = Field(default=None, gt=0.0)
        L0_m: float | None = Field(default=None, gt=0.0)
        atmospheric_transmission: float | None = Field(
            default=None,
            gt=0.0,
            le=1.0,
            description=(
                "Band-averaged atmospheric transmission folded into the "
                "throughput chain. Site-typical values: Maunakea J-band "
                "~0.93, Paranal J-band ~0.95. None ⇒ no atmosphere "
                "transmission contribution (legacy behavior)."
            ),
        )

        @field_validator("r0_500nm_m")
        @classmethod
        def _validate_r0_500nm_m(cls, value: float) -> float:
            if math.isinf(value):
                raise ValueError("r0_500nm_m must be finite or NaN.")
            if math.isfinite(value) and value <= 0.0:
                raise ValueError("r0_500nm_m must be greater than 0 where finite.")
            return value

    def apply(self, state: SimulationState) -> SimulationState:
        params = self.params_model
        wavelength_grid_um = _require_wavelength_grid_um(state)
        aperture_diameter_m = _require_aperture_diameter_m(state)

        wavelength_grid_m = np.asarray(wavelength_grid_um, dtype=np.float64) * _MICRON_TO_M
        airmass = _airmass_from_scene(state)
        r0_scaled_m = kolmogorov.r0_scaled(
            params.r0_500nm_m,
            wavelength_grid_m,
            kolmogorov.R0_REFERENCE_WAVELENGTH_M,
        ) * airmass ** (-0.6)
        phase_variance_rad2 = kolmogorov.phase_variance(aperture_diameter_m, r0_scaled_m)
        opd_rms_m = kolmogorov.opd_rms_m(
            aperture_diameter_m,
            r0_scaled_m,
            wavelength_grid_m,
        )
        strehl_ratio = kolmogorov.strehl_marechal(opd_rms_m, wavelength_grid_m)

        amplitude_jitter_fraction = np.zeros_like(wavelength_grid_m, dtype=np.float64)
        representative_index = len(wavelength_grid_um) // 2
        representative_opd_nm = float(opd_rms_m[representative_index] * _M_TO_NM)

        wavefront = _updated_wavefront(
            state,
            wavelength_grid_um,
            strehl_ratio,
            amplitude_jitter_fraction,
            float(params.r0_500nm_m),
            _wavefront_component(
                params=params,
                aperture_diameter_m=aperture_diameter_m,
                wavelength_grid_um=wavelength_grid_um,
                r0_scaled_m=r0_scaled_m,
                airmass=airmass,
                phase_variance_rad2=phase_variance_rad2,
                opd_rms_m=opd_rms_m,
                strehl_ratio=strehl_ratio,
                amplitude_jitter_fraction=amplitude_jitter_fraction,
            ),
        )
        opd_budget = _updated_opd_budget(state, representative_opd_nm)

        throughput = _maybe_fold_atmospheric_transmission(
            state=state,
            wavelength_grid_um=wavelength_grid_um,
            atmospheric_transmission=params.atmospheric_transmission,
            airmass=airmass,
            instance_id=self.context.instance_id,
        )

        return replace(
            state,
            opd_budget=opd_budget,
            wavefront=wavefront,
            throughput=throughput,
        )


def _require_wavelength_grid_um(state: SimulationState) -> tuple[float, ...]:
    wavelength_grid_um = (
        state.field.wavelength_grid_um
        or state.wavefront.wavelength_grid_um
        or state.throughput.wavelength_grid_um
        or state.photon_rates.wavelength_grid_um
    )
    if not wavelength_grid_um:
        raise ValueError(
            "atmosphere.kolmogorov requires a non-empty wavelength grid "
            "from the resolved grid config."
        )
    return wavelength_grid_um


def _require_aperture_diameter_m(state: SimulationState) -> float:
    aperture_diameter_m = state.geometry.diameter_m
    if aperture_diameter_m is None:
        raise ValueError(
            "atmosphere.kolmogorov requires geometry.diameter_m from "
            "an earlier telescope stage."
        )
    if not math.isfinite(aperture_diameter_m) or aperture_diameter_m <= 0.0:
        raise ValueError(
            "atmosphere.kolmogorov requires a finite positive "
            f"geometry.diameter_m, got {aperture_diameter_m}."
        )
    return aperture_diameter_m


def _updated_wavefront(
    state: SimulationState,
    wavelength_grid_um: tuple[float, ...],
    strehl_ratio: np.ndarray,
    amplitude_jitter_fraction: np.ndarray,
    r0_at_500nm_m: float,
    component: dict[str, Any],
) -> Wavefront:
    return replace(
        state.wavefront,
        wavelength_grid_um=wavelength_grid_um,
        strehl_ratio=_float_tuple(strehl_ratio),
        amplitude_jitter_fraction=_float_tuple(amplitude_jitter_fraction),
        r0_at_500nm_m=r0_at_500nm_m,
    ).add_component("atmosphere", component)


def _wavefront_component(
    *,
    params: KolmogorovScreens.Params,
    aperture_diameter_m: float,
    wavelength_grid_um: tuple[float, ...],
    r0_scaled_m: np.ndarray,
    airmass: float,
    phase_variance_rad2: np.ndarray,
    opd_rms_m: np.ndarray,
    strehl_ratio: np.ndarray,
    amplitude_jitter_fraction: np.ndarray,
) -> dict[str, Any]:
    return {
        "aperture_diameter_m": aperture_diameter_m,
        "wavelength_grid_um": wavelength_grid_um,
        "r0_500nm_m": params.r0_500nm_m,
        # Wavelength- and airmass-scaled r0 that the variance uses.
        "r0_scaled_m": _float_tuple(r0_scaled_m),
        "airmass": airmass,
        "seeing_arcsec": (
            params.seeing_arcsec
            if params.seeing_arcsec is not None
            else _seeing_arcsec_from_r0_500nm(params.r0_500nm_m)
        ),
        "wind_speed_m_s": params.wind_speed_m_s,
        "L0_m": params.L0_m,
        "outer_scale_model": "infinite" if params.L0_m is None else "recorded_unused",
        "piston_removed": True,
        "phase_variance_rad2": _float_tuple(phase_variance_rad2),
        "opd_rms_nm": _float_tuple(opd_rms_m * _M_TO_NM),
        "strehl_ratio": _float_tuple(strehl_ratio),
        "amplitude_jitter_fraction": _float_tuple(amplitude_jitter_fraction),
    }


def _updated_opd_budget(
    state: SimulationState,
    representative_opd_nm: float,
) -> OPDBudget:
    total_rms_opd_nm = _quadrature_total(
        state.opd_budget.total_rms_opd_nm,
        representative_opd_nm,
    )
    return replace(
        state.opd_budget,
        total_rms_opd_nm=total_rms_opd_nm,
    ).add_component(
        "atmos_piston",
        OPDComponent(
            rms_opd_nm=representative_opd_nm,
            reduction_stage="kolmogorov_screens",
        ),
    )


def _quadrature_total(existing_opd_nm: float | None, added_opd_nm: float) -> float:
    if existing_opd_nm is None:
        return added_opd_nm
    return float(math.hypot(existing_opd_nm, added_opd_nm))


def _seeing_arcsec_from_r0_500nm(r0_500nm_m: float) -> float:
    if math.isnan(r0_500nm_m):
        return math.nan
    seeing_rad = 0.98 * kolmogorov.R0_REFERENCE_WAVELENGTH_M / r0_500nm_m
    return seeing_rad * _RAD_TO_ARCSEC


def _float_tuple(values: np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _airmass_from_scene(state: SimulationState) -> float:
    """Airmass from scene.altitude_deg; 1.0 (zenith) when unset."""

    altitude_deg = state.scene.altitude_deg
    if altitude_deg is None:
        return 1.0
    altitude = float(altitude_deg)
    if not math.isfinite(altitude) or altitude <= 0.0 or altitude > 90.0:
        raise ValueError(
            "atmosphere.kolmogorov could not derive airmass from "
            f"scene.altitude_deg={altitude_deg!r}; expected 0 < altitude_deg "
            "<= 90."
        )
    return max(1.0, 1.0 / math.sin(math.radians(altitude)))


def _maybe_fold_atmospheric_transmission(
    *,
    state: SimulationState,
    wavelength_grid_um: tuple[float, ...],
    atmospheric_transmission: float | None,
    airmass: float,
    instance_id: str,
) -> Throughput:
    """Multiply the running throughput by a band-averaged atmospheric factor.

    A None ``atmospheric_transmission`` is a no-op (legacy behavior preserved
    for HWO-class space configs and any ground config that has not opted in).
    A finite value is Beer-Lambert scaled to the pointing airmass
    (``T**airmass`` — the same convention as the WFS photon budget and
    KILO's science path; review 2026-06-10 Issue 28a), broadcast to every
    wavelength bin, multiplied into the running ``transmission_fraction``,
    and recorded as a Throughput component so downstream
    throughput-breakdown plots can attribute the loss.
    """

    if atmospheric_transmission is None:
        return state.throughput

    n_bins = len(wavelength_grid_um)
    scaled_transmission = float(atmospheric_transmission) ** float(airmass)
    per_channel = np.full(n_bins, scaled_transmission, dtype=np.float64)
    current = (
        np.asarray(state.throughput.transmission_fraction, dtype=np.float64)
        if state.throughput.transmission_fraction
        else np.ones(n_bins, dtype=np.float64)
    )
    if current.size != n_bins:
        raise ValueError(
            "atmosphere.kolmogorov: throughput.transmission_fraction "
            f"length {current.size} does not match wavelength grid length {n_bins}."
        )
    updated = current * per_channel

    component_name = unique_component_name(
        "atmosphere.transmission",
        state.throughput.components,
        instance_id,
    )
    return Throughput(
        wavelength_grid_um=wavelength_grid_um,
        transmission_fraction=_float_tuple(updated),
        components={
            **state.throughput.components,
            component_name: {
                "wavelength_grid_um": wavelength_grid_um,
                "transmission_fraction": _float_tuple(per_channel),
                "atmospheric_transmission": float(atmospheric_transmission),
                "airmass": float(airmass),
                "atmospheric_transmission_at_airmass": scaled_transmission,
                "source_model": (
                    "Band-averaged zenith atmospheric transmission, "
                    "Beer-Lambert scaled to the pointing airmass "
                    "(T**airmass)."
                ),
            },
        },
    )


__all__ = ["KolmogorovScreens"]
