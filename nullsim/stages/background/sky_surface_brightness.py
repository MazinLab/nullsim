"""Generic sky-surface-brightness background stage.

This is the site-profile version of ``background.nayra_sky``. It keeps the
same downstream payload keys used by performance and characterization stages,
but selects default sky brightness from a named site profile instead of a
NAYRA-only table.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField

from nullsim.pipeline import BaseStage, SimulationState, register_stage
from nullsim.stages._science_bands import infer_band_from_wavelength_um
from nullsim.stages._utils import unique_component_name
from nullsim.stages.background.nayra_sky import (
    _bandwidth_per_bin_um,
    _rate_throughput_or_unity,
    _require_wavelength_grid_um,
    _resolve_area_per_sub_m2,
    _resolve_n_subapertures,
    _sky_rate_per_bin_hz,
)

SITE_SKY_MAG_ARCSEC2: dict[str, dict[str, float]] = {
    "cerro_armazones": {
        "Y": 17.0,
        "J": 15.5,
        "H": 13.0,
        "K": 12.0,
    },
    # Representative dark Maunakea near-IR sky brightnesses in Vega
    # mag/arcsec^2. These are intentionally defaults, not a substitute for a
    # site-specific observing-condition file.
    "maunakea": {
        "Y": 17.3,
        "J": 16.0,
        "H": 14.0,
        "K": 13.0,
    },
}

# Legacy fixed sky-spaxel table (Vega mag/arcsec^2 receiver footprint).
# Retained only as a documented fallback for the case where the per-mode
# collecting area cannot be derived from geometry; the default path now uses
# the single-mode etendue (see ``_single_mode_spaxel_arcsec2``).
DEFAULT_SPAXEL_ARCSEC2 = {
    "Y": 0.0035,
    "J": 0.005,
    "H": 0.005,
    "K": 0.007,
}

# 1 sr = (180*3600/pi)^2 arcsec^2.
_ARCSEC2_PER_STERADIAN = (180.0 * 3600.0 / np.pi) ** 2


def _single_mode_spaxel_arcsec2(
    *,
    area_per_sub_m2: float,
    wavelength_center_m: float,
    n_polarizations: int,
) -> float:
    """Single-mode receiver sky spaxel from the A.Omega = n_pol.lambda^2 rule.

    A single-mode fiber/waveguide accepts the etendue A.Omega = n_pol.lambda^2,
    so the on-sky solid angle is Omega = n_pol.lambda^2 / A. Returns Omega in
    arcsec^2. The previous fixed 0.005 arcsec^2 table under-counted the sky by
    ~4-9x for Keck-scale subapertures.

    Args:
        area_per_sub_m2: Collecting area per chip input mode (m^2).
        wavelength_center_m: Band-center wavelength (m).
        n_polarizations: Polarization modes the receiver detects (2 for an
            MKID behind a PM fiber that carries both polarizations).

    Returns:
        Sky spaxel solid angle in arcsec^2.

    Raises:
        ValueError: if ``area_per_sub_m2`` or ``wavelength_center_m`` is not
            finite and positive, or ``n_polarizations`` is not positive.
    """

    if not (np.isfinite(area_per_sub_m2) and area_per_sub_m2 > 0.0):
        raise ValueError(
            "background.sky_surface_brightness: area_per_sub_m2 must be finite "
            f"and positive, got {area_per_sub_m2}."
        )
    if not (np.isfinite(wavelength_center_m) and wavelength_center_m > 0.0):
        raise ValueError(
            "background.sky_surface_brightness: wavelength_center_m must be "
            f"finite and positive, got {wavelength_center_m}."
        )
    if n_polarizations <= 0:
        raise ValueError(
            "background.sky_surface_brightness: n_polarizations must be "
            f"positive, got {n_polarizations}."
        )
    omega_sr = (
        float(n_polarizations) * wavelength_center_m ** 2 / float(area_per_sub_m2)
    )
    return omega_sr * _ARCSEC2_PER_STERADIAN


def _resolve_spaxel_arcsec2(
    *,
    spaxel_override_arcsec2: float | None,
    area_per_sub_m2: float,
    wavelength_grid_um: tuple[float, ...],
    n_polarizations: int,
    band: str,
) -> tuple[float, str]:
    """Resolve the sky spaxel, preferring the single-mode etendue.

    Precedence: explicit ``spaxel_arcsec2`` override, then the single-mode
    etendue from the per-mode collecting area, then the legacy fixed table
    (only if the per-mode area is unusable).
    """

    if spaxel_override_arcsec2 is not None:
        return float(spaxel_override_arcsec2), "param.spaxel_arcsec2"
    if np.isfinite(area_per_sub_m2) and area_per_sub_m2 > 0.0:
        wavelength_center_m = (
            float(wavelength_grid_um[len(wavelength_grid_um) // 2]) * 1.0e-6
        )
        spaxel = _single_mode_spaxel_arcsec2(
            area_per_sub_m2=area_per_sub_m2,
            wavelength_center_m=wavelength_center_m,
            n_polarizations=n_polarizations,
        )
        return spaxel, "single_mode_etendue"
    return DEFAULT_SPAXEL_ARCSEC2[band], "legacy_fixed_table"


@register_stage
class SkySurfaceBrightness(BaseStage):
    """Compute sky/background rate per dark/spectro port from a site profile."""

    type_name = "background.sky_surface_brightness"
    family = "background"
    consumes = frozenset({"geometry", "throughput", "field"})
    produces = frozenset({"results"})

    class Params(BaseModel):
        """Validated params for the generic sky-background calculator."""

        model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

        site_profile: Literal["cerro_armazones", "maunakea"] = "cerro_armazones"
        band: Literal["Y", "J", "H", "K"] | None = None
        sky_mag_arcsec2: float | None = PydanticField(default=None, gt=0.0)
        spaxel_arcsec2: float | None = PydanticField(default=None, gt=0.0)
        # Polarization modes the receiver detects (used only when
        # spaxel_arcsec2 is None, to set the single-mode etendue spaxel).
        n_polarizations: int = PydanticField(default=2, ge=1)
        n_subapertures: int | None = PydanticField(default=None, ge=1)
        area_source: Literal["effective", "collecting_area_per_mode"] = "effective"
        thermal_background_hz_per_port: float = PydanticField(default=0.0, ge=0.0)
        exozodi_background_hz_per_port: float = PydanticField(default=0.0, ge=0.0)

    def apply(self, state: SimulationState) -> SimulationState:
        params = self.params_model
        wavelength_grid_um = _require_wavelength_grid_um(state)
        band = _resolve_band(params.band, wavelength_grid_um)
        sky_mag_arcsec2 = (
            float(params.sky_mag_arcsec2)
            if params.sky_mag_arcsec2 is not None
            else SITE_SKY_MAG_ARCSEC2[str(params.site_profile)][band]
        )
        n_subapertures = _resolve_n_subapertures(state, params.n_subapertures)
        area_per_sub_m2 = _resolve_area_per_sub_m2(
            state=state,
            n_subapertures=n_subapertures,
            area_source=params.area_source,
        )
        spaxel_arcsec2, spaxel_source = _resolve_spaxel_arcsec2(
            spaxel_override_arcsec2=params.spaxel_arcsec2,
            area_per_sub_m2=area_per_sub_m2,
            wavelength_grid_um=wavelength_grid_um,
            n_polarizations=params.n_polarizations,
            band=band,
        )
        bandwidth_um = _bandwidth_per_bin_um(wavelength_grid_um)
        throughput = _rate_throughput_or_unity(state, len(wavelength_grid_um))
        sky_rate_per_bin_hz = _sky_rate_per_bin_hz(
            sky_mag_arcsec2=sky_mag_arcsec2,
            wavelength_grid_um=wavelength_grid_um,
            bandwidth_um=bandwidth_um,
            band=band,
            area_per_sub_m2=area_per_sub_m2,
            spaxel_arcsec2=spaxel_arcsec2,
            throughput=throughput,
        )
        sky_rate_hz_per_port = float(np.sum(sky_rate_per_bin_hz))
        total_background_hz_per_port = (
            sky_rate_hz_per_port
            + float(params.thermal_background_hz_per_port)
            + float(params.exozodi_background_hz_per_port)
        )

        component_name = unique_component_name(
            self.type_name,
            state.results.components,
            self.context.instance_id,
        )
        results = state.results.add_component(
            component_name,
            {
                "site_profile": str(params.site_profile),
                "band": band,
                "sky_mag_arcsec2": sky_mag_arcsec2,
                "spaxel_arcsec2": spaxel_arcsec2,
                "spaxel_source": spaxel_source,
                "n_polarizations": int(params.n_polarizations),
                "n_subapertures": int(n_subapertures),
                "area_source": params.area_source,
                "area_per_sub_m2": float(area_per_sub_m2),
                "wavelength_grid_um": tuple(float(v) for v in wavelength_grid_um),
                "bandwidth_um": bandwidth_um.astype(np.float64, copy=True),
                "throughput_fraction": throughput.astype(np.float64, copy=True),
                "throughput_mean": float(np.mean(throughput)),
                "sky_rate_hz_per_bin": sky_rate_per_bin_hz.astype(
                    np.float64,
                    copy=True,
                ),
                "sky_rate_hz_per_dark_port": sky_rate_hz_per_port,
                "sky_rate_hz_per_spectro_port": sky_rate_hz_per_port,
                "thermal_background_hz_per_port": float(
                    params.thermal_background_hz_per_port
                ),
                "exozodi_background_hz_per_port": float(
                    params.exozodi_background_hz_per_port
                ),
                "background_rate_hz_per_dark_port": total_background_hz_per_port,
                "background_rate_hz_per_spectro_port": total_background_hz_per_port,
                "source_model": (
                    "Generic site sky surface brightness × spaxel area × "
                    "active wavelength bins × sub-aperture area × upstream "
                    "throughput. Shares downstream payload keys with "
                    "background.nayra_sky."
                ),
            },
        )
        return replace(state, results=results)


def _resolve_band(
    band: str | None,
    wavelength_grid_um: tuple[float, ...],
) -> Literal["Y", "J", "H", "K"]:
    if band is not None:
        return band
    inferred = infer_band_from_wavelength_um(
        float(wavelength_grid_um[len(wavelength_grid_um) // 2])
    )
    if inferred in {"Y", "J", "H", "K"}:
        return inferred
    raise ValueError(
        "background.sky_surface_brightness could not infer a supported "
        f"near-IR band from the wavelength grid (got {inferred!r})."
    )


__all__ = ["DEFAULT_SPAXEL_ARCSEC2", "SITE_SKY_MAG_ARCSEC2", "SkySurfaceBrightness"]
