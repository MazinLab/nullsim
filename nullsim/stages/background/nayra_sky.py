"""NAYRA scalar sky/background rate calculator.

This stage ports the rate-level structure of nayra
``nayra.sky_background`` into nullsim's staged state: band-specific sky
surface brightness, a NAYRA dark-port spaxel area, the active wavelength
grid, upstream throughput, and the current sub-aperture collecting area
produce a per-port background rate. Thermal and exozodi terms are exposed
as additive scalar hooks until a full thermal/exozodi spectral model is
ported.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field as PydanticField

from nullsim.physics.photometry import photon_rate_per_m2_per_bin
from nullsim.pipeline import BaseStage, SimulationState, register_stage
from nullsim.stages._science_bands import infer_band_from_wavelength_um
from nullsim.stages._utils import unique_component_name


__all__ = ["NayraSkyBackground"]


_SKY_MAG_ARCSEC2 = {
    "Y": 17.0,
    "J": 15.5,
    "H": 13.0,
    "K": 12.0,
}

_DARK_PORT_SPAXEL_ARCSEC2 = {
    "Y": 0.0035,
    "J": 0.005,
    "H": 0.005,
    "K": 0.007,
}

@register_stage
class NayraSkyBackground(BaseStage):
    """Compute a NAYRA-style scalar sky/background rate per dark/spectro port."""

    type_name = "background.nayra_sky"
    family = "background"
    consumes = frozenset({"geometry", "throughput", "field"})
    produces = frozenset({"results"})

    class Params(BaseModel):
        """Validated params for the NAYRA sky/background calculator."""

        model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

        band: Literal["Y", "J", "H", "K"] | None = None
        sky_mag_arcsec2: float | None = PydanticField(default=None, gt=0.0)
        spaxel_arcsec2: float | None = PydanticField(default=None, gt=0.0)
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
            else _SKY_MAG_ARCSEC2[band]
        )
        spaxel_arcsec2 = (
            float(params.spaxel_arcsec2)
            if params.spaxel_arcsec2 is not None
            else _DARK_PORT_SPAXEL_ARCSEC2[band]
        )
        n_subapertures = _resolve_n_subapertures(state, params.n_subapertures)
        area_per_sub_m2 = _resolve_area_per_sub_m2(
            state=state,
            n_subapertures=n_subapertures,
            area_source=params.area_source,
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
                "band": band,
                "sky_mag_arcsec2": sky_mag_arcsec2,
                "spaxel_arcsec2": spaxel_arcsec2,
                "n_subapertures": int(n_subapertures),
                "area_source": params.area_source,
                "area_per_sub_m2": float(area_per_sub_m2),
                "wavelength_grid_um": tuple(float(v) for v in wavelength_grid_um),
                "bandwidth_um": bandwidth_um.astype(np.float64, copy=True),
                "throughput_fraction": throughput.astype(np.float64, copy=True),
                "throughput_mean": float(np.mean(throughput)),
                "sky_rate_hz_per_bin": sky_rate_per_bin_hz.astype(
                    np.float64, copy=True
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
                    "Rate-level port of nayra.sky_background: "
                    "band sky surface brightness × NAYRA spaxel area × "
                    "active wavelength bins × sub-aperture area × upstream "
                    "throughput. Thermal/exozodi terms are additive scalar "
                    "placeholders."
                ),
            },
        )
        return replace(state, results=results)


def _require_wavelength_grid_um(state: SimulationState) -> tuple[float, ...]:
    grid = (
        state.field.wavelength_grid_um
        or state.throughput.wavelength_grid_um
        or state.photon_rates.wavelength_grid_um
        or state.wavefront.wavelength_grid_um
    )
    if not grid:
        raise ValueError("background.nayra_sky requires a wavelength grid.")
    return tuple(float(v) for v in grid)


def _resolve_band(
    band: str | None,
    wavelength_grid_um: tuple[float, ...],
) -> Literal["Y", "J", "H", "K"]:
    if band is not None:
        return band
    inferred = infer_band_from_wavelength_um(
        float(wavelength_grid_um[len(wavelength_grid_um) // 2])
    )
    if inferred == "Y":
        return "Y"
    if inferred == "J":
        return "J"
    if inferred == "H":
        return "H"
    if inferred == "K":
        return "K"
    raise ValueError(
        "background.nayra_sky could not infer a supported NAYRA band "
        f"from the wavelength grid (got {inferred!r})."
    )


def _resolve_n_subapertures(
    state: SimulationState,
    configured: int | None,
) -> int:
    if configured is not None:
        return int(configured)
    if state.field.amplitudes is not None:
        return int(state.field.amplitudes.shape[0])
    if state.geometry.aperture_positions_pupil_m is not None:
        return int(np.asarray(state.geometry.aperture_positions_pupil_m).shape[0])
    if state.geometry.n_apertures is not None:
        return int(state.geometry.n_apertures)
    raise ValueError(
        "background.nayra_sky requires n_subapertures or a populated field/geometry."
    )


def _resolve_area_per_sub_m2(
    *,
    state: SimulationState,
    n_subapertures: int,
    area_source: str,
) -> float:
    if n_subapertures <= 0:
        raise ValueError("background.nayra_sky requires n_subapertures > 0.")
    if area_source == "effective" and state.geometry.effective_collecting_area_m2 is not None:
        return float(state.geometry.effective_collecting_area_m2) / n_subapertures
    if state.geometry.collecting_area_m2 is not None:
        return float(state.geometry.collecting_area_m2) / n_subapertures
    if state.geometry.area_m2 is not None:
        n_apertures = state.geometry.n_apertures or 1
        return float(state.geometry.area_m2) * float(n_apertures) / n_subapertures
    raise ValueError(
        "background.nayra_sky requires geometry.effective_collecting_area_m2, "
        "geometry.collecting_area_m2, or geometry.area_m2."
    )


def _bandwidth_per_bin_um(wavelength_grid_um: tuple[float, ...]) -> NDArray[np.float64]:
    wl = np.asarray(wavelength_grid_um, dtype=np.float64)
    if wl.size == 1:
        return np.array([wl[0]], dtype=np.float64)
    edges = np.empty(wl.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (wl[:-1] + wl[1:])
    edges[0] = wl[0] - 0.5 * (wl[1] - wl[0])
    edges[-1] = wl[-1] + 0.5 * (wl[-1] - wl[-2])
    return np.diff(edges)


def _throughput_or_unity(
    state: SimulationState,
    n_wavelength_bins: int,
) -> NDArray[np.float64]:
    if not state.throughput.transmission_fraction:
        return np.ones(n_wavelength_bins, dtype=np.float64)
    throughput = np.asarray(state.throughput.transmission_fraction, dtype=np.float64)
    if throughput.shape != (n_wavelength_bins,):
        raise ValueError(
            "background.nayra_sky: throughput.transmission_fraction length "
            f"{throughput.size} does not match wavelength grid length "
            f"{n_wavelength_bins}."
        )
    return throughput


def _rate_throughput_or_unity(
    state: SimulationState,
    n_wavelength_bins: int,
) -> NDArray[np.float64]:
    """Return throughput appropriate for per-physical-port background rates."""

    throughput = _throughput_or_unity(state, n_wavelength_bins)
    return (
        throughput
        * _field_applied_throughput_or_unity(state, n_wavelength_bins)
        * _physical_arm_fraction(state)
        * _splitter_input_fraction(state)
    )


def _splitter_input_fraction(state: SimulationState) -> float:
    """Sky-étendue correction for duplicated-input architectures.

    Sky enters through the PHYSICAL guided modes only; after an
    ``injection.mode_splitter`` fanout each of the n_output chip modes
    would otherwise be granted a full single-mode étendue, overcounting
    the total sky by ``n_output / n_input`` (2x for copies=2 in
    kilo_keck_6 — review 2026-06-10 Issue 4). For incoherent equal-power
    inputs through a unitary, each output port carries the per-input-mode
    power, i.e. ``n_input / n_output`` of a full étendue.
    """

    for payload in reversed(tuple(state.geometry.components.values())):
        if not isinstance(payload, Mapping):
            continue
        n_input = payload.get("n_input_modes")
        n_output = payload.get("n_output_modes")
        if n_input is None or n_output is None:
            continue
        try:
            fraction = float(n_input) / float(n_output)
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if np.isfinite(fraction) and 0.0 < fraction <= 1.0:
            return fraction
    return 1.0


def _field_applied_throughput_or_unity(
    state: SimulationState,
    n_wavelength_bins: int,
) -> NDArray[np.float64]:
    """Return losses recorded as throughput components but applied to fields."""

    factor = np.ones(n_wavelength_bins, dtype=np.float64)
    for name, payload in state.throughput.components.items():
        if not isinstance(payload, Mapping):
            continue
        if not _is_field_applied_throughput(payload):
            continue
        if payload.get("point_source_coupling_only") is True:
            # Strehl / obscuration overlap factors are point-source
            # wavefront couplings; a diffuse background's single-mode
            # coupling is étendue-fixed (review 2026-06-10 Issue 5).
            # Only the component's true transmission part (e.g. Fresnel)
            # applies, published as diffuse_transmission_fraction.
            component_fraction = payload.get("diffuse_transmission_fraction")
            if component_fraction is None:
                continue
            component = np.asarray(component_fraction, dtype=np.float64)
            if component.shape != (n_wavelength_bins,):
                raise ValueError(
                    "background.nayra_sky: diffuse transmission component "
                    f"{name!r} length {component.size} does not match "
                    f"wavelength grid length {n_wavelength_bins}."
                )
            factor *= component
            continue
        component_fraction = payload.get("transmission_fraction")
        if component_fraction is None:
            continue
        component = np.asarray(component_fraction, dtype=np.float64)
        if component.shape != (n_wavelength_bins,):
            raise ValueError(
                "background.nayra_sky: field-applied throughput component "
                f"{name!r} length {component.size} does not match wavelength "
                f"grid length {n_wavelength_bins}."
            )
        factor *= component
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


def _physical_arm_fraction(state: SimulationState) -> float:
    amplitudes = state.field.amplitudes
    if amplitudes is None or amplitudes.ndim != 3 or amplitudes.shape[2] <= 1:
        return 1.0
    n_pol = int(amplitudes.shape[2])
    for payload in state.field.components.values():
        if not isinstance(payload, Mapping):
            continue
        labels = payload.get("polarization_labels")
        if labels is None:
            continue
        try:
            resolved = tuple(labels)
        except TypeError:
            continue
        if len(resolved) == n_pol:
            return 1.0 / float(n_pol)
    return 1.0


def _sky_rate_per_bin_hz(
    *,
    sky_mag_arcsec2: float,
    wavelength_grid_um: tuple[float, ...],
    bandwidth_um: NDArray[np.float64],
    band: str,
    area_per_sub_m2: float,
    spaxel_arcsec2: float,
    throughput: NDArray[np.float64],
) -> NDArray[np.float64]:
    rates = np.array(
        [
            photon_rate_per_m2_per_bin(
                sky_mag_arcsec2,
                float(wl_um),
                float(delta_um),
                band=band,
            )
            for wl_um, delta_um in zip(wavelength_grid_um, bandwidth_um)
        ],
        dtype=np.float64,
    )
    return rates * float(area_per_sub_m2) * float(spaxel_arcsec2) * throughput
