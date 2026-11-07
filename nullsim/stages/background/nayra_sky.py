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
from nullsim.physics.ruilier_cassaing import (
    OPTIMAL_PUPIL_RADIUS_TO_MODE_FIELD_RADIUS_RATIO,
)
from nullsim.pipeline import BaseStage, SimulationState, register_stage
from nullsim.stages._mode_splitter import splitter_physical_indices
from nullsim.stages._science_bands import infer_band_from_wavelength_um
from nullsim.stages._utils import unique_component_name


__all__ = ["NayraSkyBackground"]


_SKY_MAG_ARCSEC2 = {
    "Y": 17.0,
    "J": 15.5,
    "H": 13.0,
    "K": 12.0,
}

# 1 sr = (180*3600/pi)^2 arcsec^2.
ARCSEC2_PER_STERADIAN = (180.0 * 3600.0 / np.pi) ** 2

DIFFUSE_SINGLE_MODE_ACCEPTANCE = 1.0 - np.exp(
    -2.0 * OPTIMAL_PUPIL_RADIUS_TO_MODE_FIELD_RADIUS_RATIO**2
)
"""Diffuse-light acceptance of a single-mode input behind a circular aperture.

By the antenna theorem (Siegman 1966, Proc. IEEE 54, 1350) a single guided
mode accepts the etendue A.Omega = lambda^2 of a uniform incoherent source of
total-intensity surface brightness B, over both polarizations. A circular
aperture of radius R in front of the input passes only the fraction
``1 - exp(-2 rho^2)`` of the back-propagated Gaussian mode of field radius w,
with ``rho = R / w``. At the Ruilier-Cassaing optimum rho = 1.1209 this is
0.919, so the input collects ``B lambda^2 * 0.919`` per spectral bin. The
point-source coupling eta = 0.8145 does not apply to diffuse light.
"""

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
        sky_rate_per_input_per_bin_hz = _sky_rate_per_bin_hz(
            sky_mag_arcsec2=sky_mag_arcsec2,
            wavelength_grid_um=wavelength_grid_um,
            bandwidth_um=bandwidth_um,
            band=band,
            area_per_sub_m2=area_per_sub_m2,
            spaxel_arcsec2=spaxel_arcsec2,
            throughput=throughput,
        )
        routed = _routed_sky_payload(state, sky_rate_per_input_per_bin_hz)
        additive_hz_per_port = float(params.thermal_background_hz_per_port) + float(
            params.exozodi_background_hz_per_port
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
                **routed,
                "thermal_background_hz_per_port": float(
                    params.thermal_background_hz_per_port
                ),
                "exozodi_background_hz_per_port": float(
                    params.exozodi_background_hz_per_port
                ),
                "background_rate_hz_per_dark_port": (
                    routed["sky_rate_hz_per_dark_port"] + additive_hz_per_port
                ),
                "background_rate_hz_per_spectro_port": (
                    routed["sky_rate_hz_per_spectro_port"] + additive_hz_per_port
                ),
                "source_model": (
                    "Rate-level port of nayra.sky_background: "
                    "band sky surface brightness × NAYRA spaxel area × "
                    "active wavelength bins × sub-aperture area × upstream "
                    "throughput, routed per output through the deployed chip "
                    "stack. Thermal/exozodi terms are additive scalar "
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
        # Count physical guided modes: sky enters before any splitter fanout,
        # whose copies _diffuse_port_routing accounts for.
        n_chip_inputs = int(state.field.amplitudes.shape[0])
        return _splitter_isometry(state, n_chip_inputs).shape[1]
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
    """Throughput from the sky to one physical input mode, per PBS arm.

    Excludes the chip routing, which ``_diffuse_port_routing`` applies per
    output port.
    """

    throughput = _throughput_or_unity(state, n_wavelength_bins)
    return (
        throughput
        * _field_applied_throughput_or_unity(state, n_wavelength_bins)
        * _physical_arm_fraction(state)
    )


def _diffuse_port_routing(
    state: SimulationState,
    n_wavelength_bins: int,
) -> tuple[NDArray[np.float64], NDArray[np.intp], str]:
    """Share of one physical input's diffuse power reaching each chip output.

    Diffuse sky enters every physical guided mode k as independent,
    equal-power light, so output j receives
    ``r_j = sum_k |(U S)_jk|^2`` times the per-input power, where U is the
    deployed chip stack (per wavelength bin, per PBS arm when the chip is
    deployed per arm) and S the ``injection.mode_splitter`` isometry that
    sends input k to its copies with amplitude ``1/sqrt(n_copies)``. Copies
    of one input are coherent, so the chip can steer them unequally: the
    uniform ``n_in / n_out`` share holds only on average over all outputs
    (review 2026-11-07 Issue 11). With no chip upstream U is the identity.

    Returns:
        ``(routing, dark_indices, source)``: routing has shape
        ``(n_outputs, n_wavelength_bins)``, averaged over PBS arms (each arm
        already carries ``1/n_arm`` of the power through
        ``_physical_arm_fraction``).
    """

    chip = _latest_chip_transfer(state)
    if chip is None:
        if state.field.amplitudes is None:
            raise ValueError("background stages require field amplitudes.")
        n_chip_inputs = int(np.asarray(state.field.amplitudes).shape[0])
        stacks = np.eye(n_chip_inputs, dtype=np.complex128)[
            np.newaxis, :, :, np.newaxis
        ]
        source = "identity (no chip transfer stack upstream)"
        n_bright, n_dark = 0, None
    else:
        name, payload = chip
        per_arm = payload.get("u_stack_per_arm")
        stacks = np.asarray(
            payload["u_stack"] if per_arm is None else per_arm,
            dtype=np.complex128,
        )
        if per_arm is None:
            stacks = stacks[np.newaxis]
        if stacks.shape[3] != n_wavelength_bins:
            raise ValueError(
                f"background: chip component {name!r} stack has "
                f"{stacks.shape[3]} wavelength bins, expected {n_wavelength_bins}."
            )
        n_chip_inputs = stacks.shape[2]
        source = f"results.{name}." + ("u_stack" if per_arm is None else "u_stack_per_arm")
        n_bright = int(payload.get("n_bright_ports", 0))
        n_dark = payload.get("n_dark_ports")

    splitter = _splitter_isometry(state, n_chip_inputs)
    routed = np.einsum("aoiw,ip->aopw", stacks, splitter)
    routing = np.mean(np.sum(np.abs(routed) ** 2, axis=2), axis=0)
    n_outputs = routing.shape[0]
    if n_dark:
        dark_indices = np.arange(n_bright, n_bright + int(n_dark), dtype=np.intp)
    else:
        dark_indices = np.arange(n_outputs, dtype=np.intp)
    if routing.shape[1] == 1 and n_wavelength_bins > 1:
        routing = np.repeat(routing, n_wavelength_bins, axis=1)
    return routing, dark_indices, source


def _latest_chip_transfer(
    state: SimulationState,
) -> tuple[str, Mapping[str, Any]] | None:
    for name, payload in reversed(tuple(state.results.components.items())):
        if name.startswith("chip.") and isinstance(payload, Mapping) and (
            "u_stack" in payload
        ):
            return name, payload
    return None


def _splitter_isometry(
    state: SimulationState,
    n_chip_inputs: int,
) -> NDArray[np.float64]:
    """Matrix S (n_chip_inputs, n_physical) of the guided-mode fanout."""

    found = splitter_physical_indices(state, n_chip_inputs)
    if found is None:
        return np.eye(n_chip_inputs, dtype=np.float64)
    physical = found[0]
    counts = np.bincount(physical)
    isometry = np.zeros((n_chip_inputs, counts.size), dtype=np.float64)
    isometry[np.arange(n_chip_inputs), physical] = 1.0 / np.sqrt(counts[physical])
    return isometry


def _routed_sky_payload(
    state: SimulationState,
    rate_per_input_per_bin_hz: NDArray[np.float64],
) -> dict[str, Any]:
    """Route a per-input diffuse rate through the chip to its output ports.

    The dark-port keys carry the mean over the deployed chip's dark rows.
    The spectro-port keys carry the mean over all outputs, which is exact
    for any unitary (``n_in / n_out`` of a full input), because the
    characterization stages retune the spectro rows downstream and the
    detection chip's spectro rows do not describe the port they observe.
    """

    routing, dark_indices, source = _diffuse_port_routing(
        state, rate_per_input_per_bin_hz.size,
    )
    dark_per_bin = rate_per_input_per_bin_hz * np.mean(routing[dark_indices], axis=0)
    all_port_per_bin = rate_per_input_per_bin_hz * np.mean(routing, axis=0)
    return {
        "sky_rate_hz_per_input_mode": float(np.sum(rate_per_input_per_bin_hz)),
        "sky_routing_fraction_per_port": routing.astype(np.float64, copy=True),
        "sky_routing_dark_port_indices": dark_indices.astype(np.int64, copy=True),
        "sky_routing_source": source,
        "sky_rate_hz_per_output_port": (
            routing @ rate_per_input_per_bin_hz
        ).astype(np.float64, copy=True),
        "sky_rate_hz_per_bin": dark_per_bin.astype(np.float64, copy=True),
        "sky_rate_hz_per_dark_port": float(np.sum(dark_per_bin)),
        "sky_rate_hz_per_spectro_port": float(np.sum(all_port_per_bin)),
    }


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
    spaxel_arcsec2: float | NDArray[np.float64],
    throughput: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Sky photon rate per bin into one physical input mode (before routing).

    ``spaxel_arcsec2`` is a scalar or one solid angle per wavelength bin.
    """

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
    return rates * float(area_per_sub_m2) * np.asarray(spaxel_arcsec2) * throughput
