"""Space-based local-zodi and exozodi background model."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import (
    BaseModel,
    ConfigDict,
    Field as PydanticField,
    model_validator,
)

from nullsim.physics.photometry import (
    photon_rate_per_m2_per_bin,
    stellar_band_magnitude_from_jmag,
)
from nullsim.pipeline import BaseStage, SimulationState, register_stage
from nullsim.stages._science_bands import infer_band_from_wavelength_um
from nullsim.stages._utils import unique_component_name
from nullsim.stages.background.nayra_sky import (
    _bandwidth_per_bin_um,
    _field_applied_throughput_or_unity,
    _resolve_area_per_sub_m2,
    _resolve_n_subapertures,
    _throughput_or_unity,
    _require_wavelength_grid_um,
)


__all__ = [
    "DEFAULT_EXOZODI_V_MAG_ARCSEC2",
    "DEFAULT_LOCAL_ZODI_V_MAG_ARCSEC2",
    "SOLAR_V_MAG_AT_10PC",
    "SpaceZodiBackground",
]


DEFAULT_LOCAL_ZODI_V_MAG_ARCSEC2 = 22.7
DEFAULT_EXOZODI_V_MAG_ARCSEC2 = 22.0
SOLAR_V_MAG_AT_10PC = 4.83
_SUPPORTED_BANDS = ("I", "Y", "J", "H", "K")
_SOLAR_V_MINUS_BAND = {
    # Approximate Vega-system solar colors. The zodi spectrum is treated as
    # scattered sunlight over nullsim's broad photometric proxy bands.
    "I": 0.69,
    "Y": 1.07,
    "J": 1.18,
    "H": 1.48,
    "K": 1.56,
}
_ARCSEC_PER_RAD = 206_264.80624709636


@dataclass(frozen=True)
class _RoutingThroughput:
    throughput: float
    source: str
    n_dark_ports: int
    separations_mas: NDArray[np.float64] | None = None
    throughputs_per_sep: NDArray[np.float64] | None = None


@register_stage
class SpaceZodiBackground(BaseStage):
    """Compute local-zodi and exozodi photon rates per physical dark port."""

    type_name = "background.space_zodi"
    # Draws the exozodi morphology realization from context.rng, so the result
    # depends on the config_hash-seeded stream.
    uses_rng = True
    family = "background"
    consumes = frozenset({"geometry", "throughput", "field", "scene", "results"})
    produces = frozenset({"results"})

    class Params(BaseModel):
        """Validated params for the space zodiacal-background calculator."""

        model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

        band: Literal["I", "Y", "J", "H", "K"] | None = None
        local_zodi_level: float = PydanticField(default=1.0, ge=0.0)
        exozodi_level_zodi: float | None = PydanticField(default=None, ge=0.0)
        exozodi_level_distribution: Literal["fixed", "hosts2020_lognormal"] = "fixed"
        exozodi_hosts2020_median_zodi: float = PydanticField(default=3.0, ge=0.0)
        exozodi_hosts2020_sigma_dex: float = PydanticField(
            default=math.log10(3.0),
            ge=0.0,
        )
        exozodi_level_min_zodi: float = PydanticField(default=0.0, ge=0.0)
        exozodi_level_max_zodi: float | None = PydanticField(default=None, ge=0.0)
        exozodi_morphology: Literal[
            "smooth_face_on",
            "smooth_inclined_disk",
        ] = "smooth_face_on"
        exozodi_inclination_deg: float = PydanticField(default=0.0, ge=0.0, lt=90.0)
        exozodi_inclination_distribution: Literal["fixed", "isotropic"] = "fixed"
        exozodi_max_inclination_deg: float = PydanticField(
            default=80.0,
            gt=0.0,
            lt=90.0,
        )
        exozodi_max_surface_brightness_multiplier: float = PydanticField(
            default=5.0,
            ge=1.0,
        )
        local_zodi_v_mag_arcsec2: float = PydanticField(
            default=DEFAULT_LOCAL_ZODI_V_MAG_ARCSEC2,
            gt=0.0,
        )
        exozodi_v_mag_arcsec2: float = PydanticField(
            default=DEFAULT_EXOZODI_V_MAG_ARCSEC2,
            gt=0.0,
        )
        derive_exozodi_mag_from_scene_jmag: bool = True
        default_spectral_type: str = "G"
        spaxel_arcsec2: float | None = PydanticField(default=None, gt=0.0)
        aperture_area_lam_over_d2: float = PydanticField(default=1.0, gt=0.0)
        routing: Literal["scene_planet_throughput", "dark_fraction"] = (
            "scene_planet_throughput"
        )
        n_subapertures: int | None = PydanticField(default=None, ge=1)
        area_source: Literal["effective", "collecting_area_per_mode"] = "effective"

        @model_validator(mode="after")
        def _validate_distribution_bounds(self) -> "SpaceZodiBackground.Params":
            if (
                self.exozodi_level_max_zodi is not None
                and self.exozodi_level_max_zodi < self.exozodi_level_min_zodi
            ):
                raise ValueError(
                    "exozodi_level_max_zodi must be >= exozodi_level_min_zodi."
                )
            return self

    def apply(self, state: SimulationState) -> SimulationState:
        params = self.params_model
        wavelength_grid_um = _require_wavelength_grid_um(state)
        band = _resolve_band(params.band, wavelength_grid_um)
        n_subapertures = _resolve_n_subapertures(state, params.n_subapertures)
        area_per_sub_m2 = _resolve_area_per_sub_m2(
            state=state,
            n_subapertures=n_subapertures,
            area_source=params.area_source,
        )
        spaxel_arcsec2 = _resolve_spaxel_arcsec2(
            state=state,
            wavelength_grid_um=wavelength_grid_um,
            configured=params.spaxel_arcsec2,
            aperture_area_lam_over_d2=params.aperture_area_lam_over_d2,
        )

        bandwidth_um = _bandwidth_per_bin_um(wavelength_grid_um)
        throughput = _space_rate_throughput_or_unity(state, len(wavelength_grid_um))
        routing = _resolve_routing_throughput(
            state,
            routing=params.routing,
        )
        collecting_area_m2 = float(area_per_sub_m2) * int(n_subapertures)
        local_mag = _surface_brightness_in_band(
            params.local_zodi_v_mag_arcsec2,
            band,
        )
        exozodi_mag, exozodi_mag_source, host_star_mag = _exozodi_mag_arcsec2(
            state=state,
            band=band,
            reference_v_mag_arcsec2=float(params.exozodi_v_mag_arcsec2),
            derive_from_scene=bool(params.derive_exozodi_mag_from_scene_jmag),
            default_spectral_type=str(params.default_spectral_type),
        )
        local_base_rate_per_bin_hz = _surface_brightness_rate_per_bin_hz(
            mag_arcsec2=local_mag,
            level=float(params.local_zodi_level),
            wavelength_grid_um=wavelength_grid_um,
            bandwidth_um=bandwidth_um,
            band=band,
            collecting_area_m2=collecting_area_m2,
            spaxel_arcsec2=spaxel_arcsec2,
            throughput=throughput,
            routing_throughput=1.0,
            n_dark_ports=routing.n_dark_ports,
        )
        exozodi_level, exozodi_level_source = _resolve_exozodi_level(
            state,
            params,
            self.context.rng,
        )
        exozodi_morphology = _resolve_exozodi_morphology(params, self.context.rng)
        exozodi_effective_level = (
            float(exozodi_level)
            * float(exozodi_morphology["surface_brightness_multiplier"])
        )
        exozodi_base_rate_per_bin_hz = _surface_brightness_rate_per_bin_hz(
            mag_arcsec2=exozodi_mag,
            level=exozodi_effective_level,
            wavelength_grid_um=wavelength_grid_um,
            bandwidth_um=bandwidth_um,
            band=band,
            collecting_area_m2=collecting_area_m2,
            spaxel_arcsec2=spaxel_arcsec2,
            throughput=throughput,
            routing_throughput=1.0,
            n_dark_ports=routing.n_dark_ports,
        )
        local_rate_per_bin_hz = local_base_rate_per_bin_hz * routing.throughput
        exozodi_rate_per_bin_hz = exozodi_base_rate_per_bin_hz * routing.throughput
        total_rate_per_bin_hz = local_rate_per_bin_hz + exozodi_rate_per_bin_hz
        local_rate_hz_per_port = float(np.sum(local_rate_per_bin_hz))
        exozodi_rate_hz_per_port = float(np.sum(exozodi_rate_per_bin_hz))
        background_rate_hz_per_port = float(np.sum(total_rate_per_bin_hz))
        curve_background_payload = _curve_background_payload(
            routing=routing,
            local_base_rate_per_bin_hz=local_base_rate_per_bin_hz,
            exozodi_base_rate_per_bin_hz=exozodi_base_rate_per_bin_hz,
        )

        component_name = unique_component_name(
            self.type_name,
            state.results.components,
            self.context.instance_id,
        )
        payload = {
            "band": band,
            "local_zodi_level": float(params.local_zodi_level),
            "exozodi_level_zodi": float(exozodi_level),
            "exozodi_effective_level_zodi": float(exozodi_effective_level),
            "exozodi_level_source": exozodi_level_source,
            "exozodi_level_distribution": params.exozodi_level_distribution,
            "exozodi_hosts2020_median_zodi": float(
                params.exozodi_hosts2020_median_zodi
            ),
            "exozodi_hosts2020_sigma_dex": float(
                params.exozodi_hosts2020_sigma_dex
            ),
            "exozodi_level_min_zodi": float(params.exozodi_level_min_zodi),
            "exozodi_level_max_zodi": (
                None if params.exozodi_level_max_zodi is None
                else float(params.exozodi_level_max_zodi)
            ),
            "exozodi_morphology": params.exozodi_morphology,
            "exozodi_inclination_deg": float(exozodi_morphology["inclination_deg"]),
            "exozodi_inclination_source": exozodi_morphology["inclination_source"],
            "exozodi_surface_brightness_multiplier": float(
                exozodi_morphology["surface_brightness_multiplier"]
            ),
            "exozodi_raw_surface_brightness_multiplier": float(
                exozodi_morphology["raw_surface_brightness_multiplier"]
            ),
            "local_zodi_v_mag_arcsec2": float(params.local_zodi_v_mag_arcsec2),
            "exozodi_v_mag_arcsec2": float(params.exozodi_v_mag_arcsec2),
            "local_zodi_mag_arcsec2": float(local_mag),
            "exozodi_mag_arcsec2": float(exozodi_mag),
            "exozodi_mag_source": exozodi_mag_source,
            "host_star_mag_for_exozodi": (
                None if host_star_mag is None else float(host_star_mag)
            ),
            "derive_exozodi_mag_from_scene_jmag": bool(
                params.derive_exozodi_mag_from_scene_jmag
            ),
            "spaxel_arcsec2": float(spaxel_arcsec2),
            "aperture_area_lam_over_d2": float(params.aperture_area_lam_over_d2),
            "n_subapertures": int(n_subapertures),
            "area_source": params.area_source,
            "area_per_sub_m2": float(area_per_sub_m2),
            "collecting_area_m2": float(collecting_area_m2),
            "wavelength_grid_um": tuple(float(v) for v in wavelength_grid_um),
            "bandwidth_um": bandwidth_um.astype(np.float64, copy=True),
            "throughput_fraction": throughput.astype(np.float64, copy=True),
            "throughput_mean": float(np.mean(throughput)),
            "routing": params.routing,
            "routing_throughput": float(routing.throughput),
            "routing_source": routing.source,
            "n_physical_dark_ports": int(routing.n_dark_ports),
            "local_zodi_rate_hz_per_bin": local_rate_per_bin_hz.astype(
                np.float64,
                copy=True,
            ),
            "exozodi_rate_hz_per_bin": exozodi_rate_per_bin_hz.astype(
                np.float64,
                copy=True,
            ),
            "zodi_rate_hz_per_bin": total_rate_per_bin_hz.astype(
                np.float64,
                copy=True,
            ),
            "local_zodi_rate_hz_per_dark_port": local_rate_hz_per_port,
            "exozodi_rate_hz_per_dark_port": exozodi_rate_hz_per_port,
            "zodi_rate_hz_per_dark_port": background_rate_hz_per_port,
            "background_rate_hz_per_dark_port": background_rate_hz_per_port,
            "background_rate_hz_per_spectro_port": background_rate_hz_per_port,
            # Backward-compatible aliases for performance/output consumers that
            # still label any background-like term as sky.
            "sky_rate_hz_per_dark_port": background_rate_hz_per_port,
            "sky_rate_hz_per_spectro_port": background_rate_hz_per_port,
            "source_model": (
                "Local zodiacal and exozodiacal surface brightness × "
                "lambda/D photometric aperture × effective per-mode "
                "collecting area × upstream throughput × off-axis "
                "dark-port routing throughput. Local zodi uses an absolute "
                "V-band surface brightness converted with approximate "
                "solar colors; exozodi defaults to a 1-zodi surface "
                "brightness scaled by the host-star band magnitude. Optional "
                "HOSTS-style zodi-level sampling and inclined smooth-disk "
                "surface-brightness scaling are applied before routing."
            ),
        }
        payload.update(curve_background_payload)
        results = state.results.add_component(component_name, payload)
        return replace(state, results=results)


def _resolve_band(
    band: str | None,
    wavelength_grid_um: tuple[float, ...],
) -> Literal["I", "Y", "J", "H", "K"]:
    if band is not None:
        return band
    inferred = infer_band_from_wavelength_um(
        float(wavelength_grid_um[len(wavelength_grid_um) // 2])
    )
    if inferred in _SUPPORTED_BANDS:
        return cast(Literal["I", "Y", "J", "H", "K"], inferred)
    supported = ", ".join(_SUPPORTED_BANDS)
    raise ValueError(
        "background.space_zodi could not infer a supported band from the "
        f"wavelength grid (got {inferred!r}; supported: {supported})."
    )


def _resolve_spaxel_arcsec2(
    *,
    state: SimulationState,
    wavelength_grid_um: tuple[float, ...],
    configured: float | None,
    aperture_area_lam_over_d2: float,
) -> float:
    if configured is not None:
        return float(configured)
    diameter_m = state.geometry.diameter_m
    if diameter_m is None or not math.isfinite(float(diameter_m)) or diameter_m <= 0.0:
        raise ValueError(
            "background.space_zodi requires geometry.diameter_m when "
            "spaxel_arcsec2 is not configured."
        )
    center_wavelength_um = float(
        np.mean(np.asarray(wavelength_grid_um, dtype=np.float64))
    )
    lam_over_d_arcsec = (
        center_wavelength_um * 1.0e-6 / float(diameter_m) * _ARCSEC_PER_RAD
    )
    return float(aperture_area_lam_over_d2) * lam_over_d_arcsec * lam_over_d_arcsec


def _surface_brightness_in_band(v_mag_arcsec2: float, band: str) -> float:
    return float(v_mag_arcsec2) - _SOLAR_V_MINUS_BAND[band]


def _space_rate_throughput_or_unity(
    state: SimulationState,
    n_wavelength_bins: int,
) -> NDArray[np.float64]:
    return (
        _throughput_or_unity(state, n_wavelength_bins)
        * _field_applied_throughput_or_unity(state, n_wavelength_bins)
        * _single_mode_coupling_or_unity(state, n_wavelength_bins)
    )


def _single_mode_coupling_or_unity(
    state: SimulationState,
    n_wavelength_bins: int,
) -> NDArray[np.float64]:
    factor = np.ones(n_wavelength_bins, dtype=np.float64)
    for name, payload in state.throughput.components.items():
        if not name.startswith("single_mode_fiber") or not isinstance(payload, dict):
            continue
        coupling = payload.get("coupling_efficiency")
        if coupling is None:
            continue
        component = np.asarray(coupling, dtype=np.float64)
        if component.shape != (n_wavelength_bins,):
            raise ValueError(
                "background.space_zodi: single-mode coupling component "
                f"{name!r} length {component.size} does not match wavelength "
                f"grid length {n_wavelength_bins}."
            )
        factor *= component
    return factor


def _resolve_routing_throughput(
    state: SimulationState,
    *,
    routing: str,
) -> _RoutingThroughput:
    n_dark_ports, n_total_ports = _detector_port_counts(state)
    if routing == "dark_fraction":
        return _RoutingThroughput(
            throughput=float(n_dark_ports) / float(n_total_ports),
            source="detector dark-port fraction",
            n_dark_ports=n_dark_ports,
        )
    if routing != "scene_planet_throughput":
        raise ValueError(f"background.space_zodi: unsupported routing={routing!r}.")
    if state.scene.planet_separation_mas is None:
        raise ValueError(
            "background.space_zodi routing='scene_planet_throughput' requires "
            "scene.planet_separation_mas."
        )
    sensitivity_name, sensitivity_payload = _find_sensitivity_component(state)
    separations = np.asarray(sensitivity_payload["separations_mas"], dtype=np.float64)
    throughputs = np.asarray(
        sensitivity_payload["planet_throughputs"], dtype=np.float64
    )
    if separations.shape != throughputs.shape or separations.size == 0:
        raise ValueError(
            "background.space_zodi: sensitivity separations and planet_throughputs "
            "must be non-empty arrays with matching shape."
        )
    if not np.all(np.isfinite(separations)) or not np.all(np.isfinite(throughputs)):
        raise ValueError(
            "background.space_zodi: sensitivity separations and planet_throughputs "
            "must be finite."
        )
    if np.any(throughputs < 0.0):
        raise ValueError(
            "background.space_zodi: sensitivity planet_throughputs must be "
            "non-negative."
        )
    order = np.argsort(separations)
    routed = float(
        np.interp(
            float(state.scene.planet_separation_mas),
            separations[order],
            throughputs[order],
        )
    )
    if not math.isfinite(routed) or routed < 0.0:
        raise ValueError(
            "background.space_zodi: interpolated scene planet throughput must be "
            f"finite and non-negative, got {routed!r}."
        )
    return _RoutingThroughput(
        throughput=routed,
        source=f"sensitivity:{sensitivity_name}.planet_throughputs",
        n_dark_ports=n_dark_ports,
        separations_mas=separations.astype(np.float64, copy=True),
        throughputs_per_sep=throughputs.astype(np.float64, copy=True),
    )


def _curve_background_payload(
    *,
    routing: _RoutingThroughput,
    local_base_rate_per_bin_hz: NDArray[np.float64],
    exozodi_base_rate_per_bin_hz: NDArray[np.float64],
) -> dict[str, object]:
    if routing.separations_mas is None or routing.throughputs_per_sep is None:
        return {}

    local_base_rate_hz_per_port = float(np.sum(local_base_rate_per_bin_hz))
    exozodi_base_rate_hz_per_port = float(np.sum(exozodi_base_rate_per_bin_hz))
    route = routing.throughputs_per_sep.astype(np.float64, copy=True)
    local_per_sep = route * local_base_rate_hz_per_port
    exozodi_per_sep = route * exozodi_base_rate_hz_per_port
    total_per_sep = local_per_sep + exozodi_per_sep
    return {
        "background_separations_mas": routing.separations_mas.astype(
            np.float64,
            copy=True,
        ),
        "background_routing_throughputs": route,
        "local_zodi_rate_hz_per_dark_port_per_sep": local_per_sep.astype(
            np.float64,
            copy=True,
        ),
        "exozodi_rate_hz_per_dark_port_per_sep": exozodi_per_sep.astype(
            np.float64,
            copy=True,
        ),
        "zodi_rate_hz_per_dark_port_per_sep": total_per_sep.astype(
            np.float64,
            copy=True,
        ),
        "background_rate_hz_per_dark_port_per_sep": total_per_sep.astype(
            np.float64,
            copy=True,
        ),
        "sky_rate_hz_per_dark_port_per_sep": total_per_sep.astype(
            np.float64,
            copy=True,
        ),
    }


def _detector_port_counts(state: SimulationState) -> tuple[int, int]:
    for name, payload in reversed(tuple(state.results.components.items())):
        if not name.startswith("detector.") or not isinstance(payload, dict):
            continue
        labels = payload.get("port_labels")
        if labels is not None:
            port_labels = tuple(str(label) for label in labels)
            n_dark = sum(label.startswith("dark") for label in port_labels)
            if n_dark > 0:
                return int(n_dark), len(port_labels)
        n_dark_value = payload.get("n_physical_dark_ports")
        n_bright_value = payload.get("n_physical_bright_ports")
        n_spectro_value = payload.get("n_physical_spectro_ports", 0)
        if n_dark_value is not None and n_bright_value is not None:
            n_dark = int(n_dark_value)
            n_total = n_dark + int(n_bright_value) + int(n_spectro_value)
            if n_dark > 0 and n_total > 0:
                return n_dark, n_total
    raise ValueError(
        "background.space_zodi requires an upstream detector component with "
        "dark-port labels or physical port counts."
    )


def _find_sensitivity_component(
    state: SimulationState,
) -> tuple[str, dict[str, object]]:
    for name, payload in reversed(tuple(state.results.components.items())):
        if not name.startswith("sensitivity.detection_curve"):
            continue
        if isinstance(payload, dict) and {
            "separations_mas",
            "planet_throughputs",
        }.issubset(payload):
            return name, payload
    raise ValueError(
        "background.space_zodi routing='scene_planet_throughput' requires an "
        "upstream sensitivity.detection_curve component."
    )


def _exozodi_mag_arcsec2(
    *,
    state: SimulationState,
    band: str,
    reference_v_mag_arcsec2: float,
    derive_from_scene: bool,
    default_spectral_type: str,
) -> tuple[float, str, float | None]:
    if not derive_from_scene:
        return (
            _surface_brightness_in_band(reference_v_mag_arcsec2, band),
            "absolute_v_mag_arcsec2_converted_with_solar_color",
            None,
        )
    if state.scene.star_jmag is None:
        raise ValueError(
            "background.space_zodi derive_exozodi_mag_from_scene_jmag=True "
            "requires scene.star_jmag."
        )
    host_star_mag = stellar_band_magnitude_from_jmag(
        float(state.scene.star_jmag),
        band,
        state.scene.star_spectral_type,
        derive_from_jmag=True,
        default_spectral_type=default_spectral_type,
    )
    zodi_offset_mag = float(reference_v_mag_arcsec2) - SOLAR_V_MAG_AT_10PC
    return (
        float(host_star_mag + zodi_offset_mag),
        (
            "scene.star_jmag converted to band, plus 1-zodi surface-brightness "
            "offset relative to the Sun at 10 pc"
        ),
        float(host_star_mag),
    )


def _resolve_exozodi_level(
    state: SimulationState,
    params: SpaceZodiBackground.Params,
    rng: np.random.Generator,
) -> tuple[float, str]:
    """Resolve or sample the physical exozodi dust level in zodis."""

    if params.exozodi_level_distribution == "fixed":
        if params.exozodi_level_zodi is not None:
            return float(params.exozodi_level_zodi), "params.exozodi_level_zodi"
        if state.scene.exozodi_level_zodi is None:
            return 0.0, "scene.exozodi_level_zodi missing"
        return float(state.scene.exozodi_level_zodi), "scene.exozodi_level_zodi"

    if params.exozodi_level_distribution != "hosts2020_lognormal":
        raise ValueError(
            "background.space_zodi: unsupported exozodi_level_distribution="
            f"{params.exozodi_level_distribution!r}."
        )
    median = float(params.exozodi_hosts2020_median_zodi)
    if median <= 0.0:
        sampled = 0.0
    elif float(params.exozodi_hosts2020_sigma_dex) == 0.0:
        sampled = median
    else:
        sampled = median * 10.0 ** float(
            rng.normal(0.0, float(params.exozodi_hosts2020_sigma_dex))
        )
    clipped = max(float(params.exozodi_level_min_zodi), float(sampled))
    if params.exozodi_level_max_zodi is not None:
        clipped = min(float(params.exozodi_level_max_zodi), clipped)
    return clipped, "hosts2020_lognormal"


def _resolve_exozodi_morphology(
    params: SpaceZodiBackground.Params,
    rng: np.random.Generator,
) -> dict[str, float | str]:
    """Resolve simple exozodi morphology diagnostics and rate multiplier."""

    if params.exozodi_morphology == "smooth_face_on":
        return {
            "inclination_deg": 0.0,
            "inclination_source": "smooth_face_on",
            "raw_surface_brightness_multiplier": 1.0,
            "surface_brightness_multiplier": 1.0,
        }
    if params.exozodi_morphology != "smooth_inclined_disk":
        raise ValueError(
            "background.space_zodi: unsupported exozodi_morphology="
            f"{params.exozodi_morphology!r}."
        )

    if params.exozodi_inclination_distribution == "isotropic":
        max_inclination_rad = math.radians(float(params.exozodi_max_inclination_deg))
        cos_i = float(rng.uniform(math.cos(max_inclination_rad), 1.0))
        inclination_deg = math.degrees(math.acos(cos_i))
        inclination_source = "isotropic"
    elif params.exozodi_inclination_distribution == "fixed":
        inclination_deg = float(params.exozodi_inclination_deg)
        inclination_source = "params.exozodi_inclination_deg"
    else:
        raise ValueError(
            "background.space_zodi: unsupported "
            "exozodi_inclination_distribution="
            f"{params.exozodi_inclination_distribution!r}."
        )

    cos_i = max(math.cos(math.radians(inclination_deg)), 1.0e-6)
    raw_multiplier = 1.0 / cos_i
    multiplier = min(
        raw_multiplier,
        float(params.exozodi_max_surface_brightness_multiplier),
    )
    return {
        "inclination_deg": inclination_deg,
        "inclination_source": inclination_source,
        "raw_surface_brightness_multiplier": raw_multiplier,
        "surface_brightness_multiplier": multiplier,
    }


def _surface_brightness_rate_per_bin_hz(
    *,
    mag_arcsec2: float,
    level: float,
    wavelength_grid_um: tuple[float, ...],
    bandwidth_um: NDArray[np.float64],
    band: str,
    collecting_area_m2: float,
    spaxel_arcsec2: float,
    throughput: NDArray[np.float64],
    routing_throughput: float,
    n_dark_ports: int,
) -> NDArray[np.float64]:
    if level <= 0.0:
        return np.zeros(len(wavelength_grid_um), dtype=np.float64)
    if n_dark_ports <= 0:
        raise ValueError("background.space_zodi requires n_dark_ports > 0.")
    rates = np.array(
        [
            photon_rate_per_m2_per_bin(
                mag_arcsec2,
                float(wl_um),
                float(delta_um),
                band=band,
            )
            for wl_um, delta_um in zip(wavelength_grid_um, bandwidth_um, strict=True)
        ],
        dtype=np.float64,
    )
    return (
        rates
        * float(level)
        * float(collecting_area_m2)
        * float(spaxel_arcsec2)
        * throughput
        * float(routing_throughput)
        / float(n_dark_ports)
    )
