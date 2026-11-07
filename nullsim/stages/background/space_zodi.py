"""Space-based local-zodi and exozodi background model.

Both terms are diffuse. Each single-mode input collects ``B lambda^2 eta_d``
per spectral bin from a surface brightness B (antenna theorem; see
``nayra_sky.DIFFUSE_SINGLE_MODE_ACCEPTANCE``), with no point-source coupling.
Local zodi is uniform across the field, so output j receives
``sum_k |(U S)_jk|^2`` of one input's share through the deployed chip.
Exozodi falls off with stellocentric radius, so its effective brightness is
the profile averaged with the position-angle-averaged dark transmission
T_dark(r) as weight.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from typing import Any, Literal, cast

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
from nullsim.physics.planet_throughput import compute_planet_throughput_vs_separation
from nullsim.pipeline import BaseStage, SimulationState, register_stage
from nullsim.stages._mode_splitter import (
    component_mode_areas_m2,
    component_mode_diameters_m,
)
from nullsim.stages._science_bands import infer_band_from_wavelength_um
from nullsim.stages._utils import unique_component_name
from nullsim.stages.background.nayra_sky import (
    ARCSEC2_PER_STERADIAN,
    DIFFUSE_SINGLE_MODE_ACCEPTANCE,
    _bandwidth_per_bin_um,
    _diffuse_port_routing,
    _latest_chip_transfer,
    _rate_throughput_or_unity,
    _require_wavelength_grid_um,
    _resolve_area_per_sub_m2,
    _resolve_n_subapertures,
)


__all__ = [
    "DEFAULT_EXOZODI_V_MAG_ARCSEC2",
    "DEFAULT_LOCAL_ZODI_V_MAG_ARCSEC2",
    "SOLAR_V_MAG_AT_10PC",
    "SpaceZodiBackground",
]

_LOG = logging.getLogger(__name__)

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
# Blackbody grain equilibrium temperature at 1 AU from the Sun, and the
# silicate sublimation temperature that sets the dust disk's inner edge.
_BLACKBODY_TEMPERATURE_1AU_K = 278.3
_DUST_SUBLIMATION_TEMPERATURE_K = 1500.0
# T(r) quadrature for the exozodi weight: log grid out to this many
# lambda_max / D_mode_min.
_EXOZODI_PROFILE_MAX_LAMBDA_OVER_D = 20.0
_EXOZODI_PROFILE_N_RADII = 240
_EXOZODI_PROFILE_N_PA = 36


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
        # 1-zodi exozodi surface brightness at 1 AU from a solar twin.
        exozodi_v_mag_arcsec2: float = PydanticField(
            default=DEFAULT_EXOZODI_V_MAG_ARCSEC2,
            gt=0.0,
        )
        derive_exozodi_mag_from_scene_jmag: bool = True
        default_spectral_type: str = "G"
        # Optional fixed solid angle per input; default is the single-mode
        # etendue eta_d lambda^2 / A per bin.
        spaxel_arcsec2: float | None = PydanticField(default=None, gt=0.0)
        # "chip_unitary" is the only model. The legacy values still load and
        # route the same way (review 2026-11-07 Issue 2a).
        routing: Literal[
            "chip_unitary",
            "scene_planet_throughput",
            "dark_fraction",
        ] = "chip_unitary"
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
        if params.routing != "chip_unitary":
            _LOG.warning(
                "background.space_zodi routing=%r is deprecated; zodi is routed "
                "through the deployed chip unitary.",
                params.routing,
            )
        wavelength_grid_um = _require_wavelength_grid_um(state)
        n_bins = len(wavelength_grid_um)
        band = _resolve_band(params.band, wavelength_grid_um)
        n_subapertures = _resolve_n_subapertures(state, params.n_subapertures)
        area_per_sub_m2 = _resolve_area_per_sub_m2(
            state=state,
            n_subapertures=n_subapertures,
            area_source=params.area_source,
        )
        spaxel_arcsec2 = _resolve_spaxel_arcsec2(
            wavelength_grid_um=wavelength_grid_um,
            area_per_sub_m2=area_per_sub_m2,
            configured=params.spaxel_arcsec2,
        )
        bandwidth_um = _bandwidth_per_bin_um(wavelength_grid_um)
        throughput = _rate_throughput_or_unity(state, n_bins)
        # m^2 arcsec^2 collected by one physical input (per PBS arm).
        per_input_etendue = float(area_per_sub_m2) * spaxel_arcsec2 * throughput
        routing, dark_indices, routing_source = _diffuse_port_routing(state, n_bins)
        route_dark = np.mean(routing[dark_indices], axis=0)
        route_all = np.mean(routing, axis=0)
        n_dark_ports, _n_total_ports = _detector_port_counts(state)

        local_mag = _surface_brightness_in_band(params.local_zodi_v_mag_arcsec2, band)
        local_sb_per_bin = float(params.local_zodi_level) * _photon_rate_per_bin(
            local_mag, wavelength_grid_um, bandwidth_um, band
        )

        exozodi = _exozodi_mag_at_1au(
            state=state,
            band=band,
            reference_v_mag_arcsec2=float(params.exozodi_v_mag_arcsec2),
            derive_from_scene=bool(params.derive_exozodi_mag_from_scene_jmag),
            default_spectral_type=str(params.default_spectral_type),
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
        exozodi_sb_1au_per_bin = exozodi_effective_level * _photon_rate_per_bin(
            exozodi["mag_arcsec2"], wavelength_grid_um, bandwidth_um, band
        )
        distance_pc = state.scene.star_distance_pc
        inner_radius_au = _dust_inner_radius_au(exozodi["luminosity_lsun"])
        if exozodi_effective_level > 0.0:
            if distance_pc is None or distance_pc <= 0.0:
                raise ValueError(
                    "background.space_zodi requires scene.star_distance_pc for a "
                    "nonzero exozodi level."
                )
            profile_dark, profile_all = (
                _exozodi_profile_weight(
                    state,
                    wavelength_grid_um=wavelength_grid_um,
                    port_indices=indices,
                    distance_pc=float(distance_pc),
                    inner_radius_au=inner_radius_au,
                )
                for indices in (dark_indices, np.arange(routing.shape[0]))
            )
        else:
            profile_dark = profile_all = 0.0

        local_rate_per_bin_hz = local_sb_per_bin * per_input_etendue * route_dark
        exozodi_rate_per_bin_hz = (
            exozodi_sb_1au_per_bin * profile_dark * per_input_etendue * route_dark
        )
        total_rate_per_bin_hz = local_rate_per_bin_hz + exozodi_rate_per_bin_hz
        local_rate_hz_per_port = float(np.sum(local_rate_per_bin_hz))
        exozodi_rate_hz_per_port = float(np.sum(exozodi_rate_per_bin_hz))
        background_rate_hz_per_port = float(np.sum(total_rate_per_bin_hz))
        spectro_rate_hz_per_port = float(
            np.sum(
                (local_sb_per_bin + exozodi_sb_1au_per_bin * profile_all)
                * per_input_etendue
                * route_all
            )
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
            # 1-zodi band surface brightness at 1 AU, before level scaling.
            "exozodi_mag_arcsec2": float(exozodi["mag_arcsec2"]),
            "exozodi_mag_source": exozodi["source"],
            "host_star_mag_for_exozodi": exozodi["apparent_mag"],
            "host_star_abs_mag_for_exozodi": exozodi["absolute_mag"],
            "star_distance_pc": None if distance_pc is None else float(distance_pc),
            "derive_exozodi_mag_from_scene_jmag": bool(
                params.derive_exozodi_mag_from_scene_jmag
            ),
            "exozodi_inner_radius_au": float(inner_radius_au),
            "exozodi_profile_weight_dark": float(profile_dark),
            "exozodi_profile_weight_all_ports": float(profile_all),
            # Unrouted, band-integrated brightness at the telescope entrance
            # (photons / s / m^2 / arcsec^2), before throughput and QE.
            "local_zodi_rate_hz_per_m2_per_arcsec2": float(np.sum(local_sb_per_bin)),
            "exozodi_rate_hz_per_m2_per_arcsec2_at_scene_separation": (
                _exozodi_sb_at_scene_separation(
                    state, float(np.sum(exozodi_sb_1au_per_bin)), inner_radius_au
                )
            ),
            # Band-center value for display; rates use the per-bin array.
            "spaxel_arcsec2": float(spaxel_arcsec2[n_bins // 2]),
            "spaxel_arcsec2_per_bin": spaxel_arcsec2.astype(np.float64, copy=True),
            "n_subapertures": int(n_subapertures),
            "area_source": params.area_source,
            "area_per_sub_m2": float(area_per_sub_m2),
            "collecting_area_m2": float(area_per_sub_m2) * int(n_subapertures),
            "wavelength_grid_um": tuple(float(v) for v in wavelength_grid_um),
            "bandwidth_um": bandwidth_um.astype(np.float64, copy=True),
            "throughput_fraction": throughput.astype(np.float64, copy=True),
            "throughput_mean": float(np.mean(throughput)),
            "routing": "chip_unitary",
            "routing_throughput": float(np.mean(route_dark)),
            "routing_fraction_dark_per_bin": route_dark.astype(np.float64, copy=True),
            "routing_source": routing_source,
            "n_physical_dark_ports": int(n_dark_ports),
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
            # Mean over all outputs, as in the ground sky stages: the
            # characterization stages retune the spectro rows downstream.
            "background_rate_hz_per_spectro_port": spectro_rate_hz_per_port,
            # Backward-compatible aliases for performance/output consumers that
            # still label any background-like term as sky.
            "sky_rate_hz_per_dark_port": background_rate_hz_per_port,
            "sky_rate_hz_per_spectro_port": spectro_rate_hz_per_port,
            "source_model": (
                "Diffuse local zodi and exozodi: surface brightness x "
                "single-mode etendue eta_d lambda^2 per input and bin x "
                "upstream throughput (no point-source coupling) x deployed "
                "chip routing sum_k |(U S)_jk|^2 averaged over dark ports. "
                "Local zodi is an absolute V surface brightness converted "
                "with solar colors. Exozodi is n zodis x 22 mag/arcsec^2 (V) "
                "at 1 AU scaled by the host's absolute band magnitude, "
                "falling as r^-2 outside the dust sublimation radius "
                "(Stark et al. 2014), averaged with the PA-averaged dark "
                "transmission T_dark(r) as weight. Optional HOSTS level "
                "sampling and inclined-disk brightening multiply the level."
            ),
        }
        payload.update(
            _curve_background_payload(
                state,
                local_rate_hz_per_port=local_rate_hz_per_port,
                exozodi_rate_hz_per_port=exozodi_rate_hz_per_port,
                routing_throughput=float(np.mean(route_dark)),
            )
        )
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
    wavelength_grid_um: tuple[float, ...],
    area_per_sub_m2: float,
    configured: float | None,
) -> NDArray[np.float64]:
    """Solid angle per input and bin: eta_d lambda^2 / A unless configured.

    The single-mode etendue, not the primary's (lambda/D)^2 shared over the
    dark ports (review 2026-11-07 Issue 2b). A times this solid angle is
    eta_d lambda^2, independent of the mode area.
    """

    wavelength_m = np.asarray(wavelength_grid_um, dtype=np.float64) * 1.0e-6
    if configured is not None:
        return np.full(wavelength_m.size, float(configured))
    return (
        DIFFUSE_SINGLE_MODE_ACCEPTANCE
        * wavelength_m**2
        / float(area_per_sub_m2)
        * ARCSEC2_PER_STERADIAN
    )


def _surface_brightness_in_band(v_mag_arcsec2: float, band: str) -> float:
    return float(v_mag_arcsec2) - _SOLAR_V_MINUS_BAND[band]


def _photon_rate_per_bin(
    mag_arcsec2: float,
    wavelength_grid_um: tuple[float, ...],
    bandwidth_um: NDArray[np.float64],
    band: str,
) -> NDArray[np.float64]:
    """Photons / s / m^2 / arcsec^2 per bin for a surface brightness."""

    return np.array(
        [
            photon_rate_per_m2_per_bin(mag_arcsec2, float(wl), float(dl), band=band)
            for wl, dl in zip(wavelength_grid_um, bandwidth_um, strict=True)
        ],
        dtype=np.float64,
    )


def _exozodi_mag_at_1au(
    *,
    state: SimulationState,
    band: str,
    reference_v_mag_arcsec2: float,
    derive_from_scene: bool,
    default_spectral_type: str,
) -> dict[str, Any]:
    """1-zodi band surface brightness at 1 AU, and the host band luminosity.

    Stark et al. (2014, ApJ 795, 122) scale the 1-zodi surface brightness
    (22 mag/arcsec^2 in V at 1 AU for a solar twin) by the host's luminosity
    in the band: ``SB = SB_ref + M_host - M_sun``. M_host is the absolute
    magnitude ``m - 5 log10(d / 10 pc)``; using the apparent magnitude made
    nearby hosts too bright and distant ones too faint (review 2026-11-07
    Issue 2c). The band luminosity ratio also sets the dust inner edge.
    """

    in_band = _surface_brightness_in_band(reference_v_mag_arcsec2, band)
    if not derive_from_scene:
        return {
            "mag_arcsec2": in_band,
            "source": "absolute_v_mag_arcsec2_converted_with_solar_color",
            "apparent_mag": None,
            "absolute_mag": None,
            "luminosity_lsun": 1.0,
        }
    if state.scene.star_jmag is None:
        raise ValueError(
            "background.space_zodi derive_exozodi_mag_from_scene_jmag=True "
            "requires scene.star_jmag."
        )
    distance_pc = state.scene.star_distance_pc
    if distance_pc is None or distance_pc <= 0.0:
        raise ValueError(
            "background.space_zodi derive_exozodi_mag_from_scene_jmag=True "
            "requires scene.star_distance_pc."
        )
    apparent_mag = float(
        stellar_band_magnitude_from_jmag(
            float(state.scene.star_jmag),
            band,
            state.scene.star_spectral_type,
            derive_from_jmag=True,
            default_spectral_type=default_spectral_type,
        )
    )
    absolute_mag = apparent_mag - 5.0 * math.log10(float(distance_pc) / 10.0)
    solar_band_mag = SOLAR_V_MAG_AT_10PC - _SOLAR_V_MINUS_BAND[band]
    return {
        "mag_arcsec2": in_band + absolute_mag - solar_band_mag,
        "source": (
            "scene.star_jmag converted to a band absolute magnitude with "
            "scene.star_distance_pc, relative to the Sun's"
        ),
        "apparent_mag": apparent_mag,
        "absolute_mag": absolute_mag,
        "luminosity_lsun": 10.0 ** (-0.4 * (absolute_mag - solar_band_mag)),
    }


def _dust_inner_radius_au(luminosity_lsun: float) -> float:
    """Blackbody-grain sublimation radius, (278.3 K / T_sub)^2 sqrt(L) AU."""

    ratio = _BLACKBODY_TEMPERATURE_1AU_K / _DUST_SUBLIMATION_TEMPERATURE_K
    return ratio * ratio * math.sqrt(float(luminosity_lsun))


def _exozodi_profile(
    radius_au: NDArray[np.float64],
    inner_radius_au: float,
) -> NDArray[np.float64]:
    """Exozodi surface brightness relative to 1 AU: r^-2 outside the inner edge."""

    radius = np.asarray(radius_au, dtype=np.float64)
    inside = radius < inner_radius_au
    return np.where(inside, 0.0, 1.0 / np.where(inside, 1.0, radius) ** 2)


def _exozodi_sb_at_scene_separation(
    state: SimulationState,
    sb_1au: float,
    inner_radius_au: float,
) -> float | None:
    separation_mas = state.scene.planet_separation_mas
    distance_pc = state.scene.star_distance_pc
    if separation_mas is None or distance_pc is None:
        return None
    radius_au = float(separation_mas) * 1.0e-3 * float(distance_pc)
    return sb_1au * float(_exozodi_profile(np.array([radius_au]), inner_radius_au)[0])


def _exozodi_profile_weight(
    state: SimulationState,
    *,
    wavelength_grid_um: tuple[float, ...],
    port_indices: NDArray[np.intp],
    distance_pc: float,
    inner_radius_au: float,
) -> float:
    """Exozodi profile averaged with the ports' T(r) as weight.

    ``<p>_T = int p(r) T(r) r dr / int T(r) r dr``, with p the brightness
    relative to 1 AU and T the PA- and band-averaged transmission into
    ``port_indices`` from the planet-throughput machinery. A uniform source
    (p = 1) gives exactly 1, so exozodi reduces to the local-zodi etendue rule
    and the weight carries only the shape of T, not the normalization of the
    angular-coupling envelope (review 2026-11-07 Issue 2a).
    """

    chip = _latest_chip_transfer(state)
    if chip is None:
        raise ValueError(
            "background.space_zodi needs an upstream chip transfer stack to "
            "route exozodi."
        )
    u_stack = np.asarray(chip[1]["u_stack"], dtype=np.complex128)
    n_modes = u_stack.shape[1]
    positions = state.geometry.aperture_positions_pupil_m
    if positions is None or np.asarray(positions).shape != (n_modes, 2):
        raise ValueError(
            "background.space_zodi requires geometry.aperture_positions_pupil_m "
            f"with shape ({n_modes}, 2) to route exozodi."
        )
    areas_m2, diameters_m = _mode_areas_and_diameters_m(state, n_modes)
    wavelengths_m = np.asarray(wavelength_grid_um, dtype=np.float64) * 1.0e-6
    theta_max_rad = (
        _EXOZODI_PROFILE_MAX_LAMBDA_OVER_D
        * float(np.max(wavelengths_m))
        / float(np.min(diameters_m))
    )
    theta_inner_rad = inner_radius_au / distance_pc / _ARCSEC_PER_RAD
    theta_rad = np.geomspace(
        1.0e-4 * theta_max_rad, theta_max_rad, _EXOZODI_PROFILE_N_RADII
    )
    if theta_inner_rad < theta_max_rad:
        theta_rad = np.union1d(theta_rad, [theta_inner_rad])
    transmission = compute_planet_throughput_vs_separation(
        u_stack=u_stack,
        wavelengths_m=wavelengths_m,
        positions_m=np.asarray(positions, dtype=np.float64),
        dark_indices=port_indices,
        coupling_per_sub=np.ones(n_modes, dtype=np.float64),
        sub_areas_m2=areas_m2,
        sub_diameters_m=diameters_m,
        separations_rad=theta_rad,
        n_pa_samples=_EXOZODI_PROFILE_N_PA,
    )
    weight = transmission * theta_rad
    norm = float(np.trapezoid(weight, theta_rad))
    if norm <= 0.0:
        return 0.0
    profile = _exozodi_profile(theta_rad * _ARCSEC_PER_RAD * distance_pc, inner_radius_au)
    return float(np.trapezoid(profile * weight, theta_rad)) / norm


def _mode_areas_and_diameters_m(
    state: SimulationState,
    n_modes: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-mode areas and diameters: vectors first, else the newest scalar.

    Geometry components are walked newest first, so the chip-input modes
    (``telescope.pupil_modes``) win over the primary that
    ``telescope.fixed_aperture`` recorded earlier (review 2026-11-07 Issue 1).
    """

    areas = component_mode_areas_m2(state, n_modes)
    diameters = component_mode_diameters_m(state, n_modes)
    area_vec = None if areas is None else areas[0]
    diameter_vec = None if diameters is None else diameters[0]
    if diameter_vec is None:
        diameter = _newest_component_scalar(state, "mode_diameter_m")
        area = _newest_component_scalar(state, "mode_area_m2")
        if diameter is not None:
            diameter_vec = np.full(n_modes, diameter)
        elif area_vec is not None:
            diameter_vec = np.sqrt(4.0 * area_vec / np.pi)
        elif area is not None:
            diameter_vec = np.full(n_modes, math.sqrt(4.0 * area / math.pi))
        else:
            raise ValueError(
                "background.space_zodi requires per-mode diameters or areas in "
                "the geometry components to route exozodi."
            )
    if area_vec is None:
        area_vec = np.pi * diameter_vec**2 / 4.0
    return area_vec, diameter_vec


def _newest_component_scalar(state: SimulationState, key: str) -> float | None:
    for payload in reversed(tuple(state.geometry.components.values())):
        if not isinstance(payload, dict) or payload.get(key) is None:
            continue
        scalar = float(payload[key])
        if math.isfinite(scalar) and scalar > 0.0:
            return scalar
    return None


def _curve_background_payload(
    state: SimulationState,
    *,
    local_rate_hz_per_port: float,
    exozodi_rate_hz_per_port: float,
    routing_throughput: float,
) -> dict[str, object]:
    """Per-separation copies on the sensitivity grid, kept for consumers.

    Both terms are diffuse and do not depend on where the planet sits, so the
    vectors are constant.
    """

    separations = _sensitivity_separations_mas(state)
    if separations is None:
        return {}
    ones = np.ones(separations.size, dtype=np.float64)
    total = local_rate_hz_per_port + exozodi_rate_hz_per_port
    return {
        "background_separations_mas": separations,
        "background_routing_throughputs": routing_throughput * ones,
        "local_zodi_rate_hz_per_dark_port_per_sep": local_rate_hz_per_port * ones,
        "exozodi_rate_hz_per_dark_port_per_sep": exozodi_rate_hz_per_port * ones,
        "zodi_rate_hz_per_dark_port_per_sep": total * ones,
        "background_rate_hz_per_dark_port_per_sep": total * ones,
        "sky_rate_hz_per_dark_port_per_sep": total * ones,
    }


def _sensitivity_separations_mas(state: SimulationState) -> NDArray[np.float64] | None:
    for name, payload in reversed(tuple(state.results.components.items())):
        if (
            name.startswith("sensitivity.detection_curve")
            and isinstance(payload, dict)
            and "separations_mas" in payload
        ):
            return np.asarray(payload["separations_mas"], dtype=np.float64).copy()
    return None


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
