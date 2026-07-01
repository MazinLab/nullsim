"""Architecture placeholder detector stage with a simple count-rate model.

This module exists to exercise the vertical-slice pipeline from SPEC §15.1.
It is not a physics implementation of MKIDs, SNSPDs, detector noise, or
photometric calibration.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from nullsim.pipeline import BaseStage, PhotonRates, SimulationState, register_stage
from nullsim.stages._utils import unique_component_name


CONTRAST_CURVE_DTYPE = np.dtype(
    [
        ("separation_mas", np.float64),
        ("contrast", np.float64),
    ]
)


@register_stage
class IdealCounter(BaseStage):
    """Convert placeholder stellar flux into one ideal detector count stream."""

    type_name = "detector.ideal_counter"
    family = "detector"
    consumes = frozenset({"field", "throughput", "scene", "geometry"})
    produces = frozenset({"photon_rates", "results"})

    class Params(BaseModel):
        """Validated params for the placeholder ideal counter."""

        model_config = ConfigDict(
            extra="forbid",
            frozen=True,
            allow_inf_nan=False,
        )

        zero_point_rate_hz_per_m2_per_bin: float = Field(default=1.0e10, gt=0.0)
        integration_time_s: float = Field(default=1.0, gt=0.0)

    def apply(self, state: SimulationState) -> SimulationState:
        params = self.params_model
        field_amplitudes = state.field.amplitudes
        if field_amplitudes is None:
            raise ValueError("detector.ideal_counter requires field amplitudes.")
        if field_amplitudes.ndim != 3:
            raise ValueError(
                "detector.ideal_counter requires field amplitudes with shape "
                "[n_modes, n_wavelength_bins, n_pol]."
            )

        throughput_fraction = _require_throughput_fraction(state)
        if field_amplitudes.shape[1] != throughput_fraction.size:
            raise ValueError(
                "detector.ideal_counter field wavelength axis must match throughput."
            )

        stellar_flux_hz_per_m2_per_bin = params.zero_point_rate_hz_per_m2_per_bin * (
            10.0 ** (-0.4 * _require_star_jmag(state))
        )
        field_power_per_wavelength = _field_power_per_wavelength(field_amplitudes)
        area_per_mode_m2 = _resolve_collecting_area_m2(state) / float(
            field_amplitudes.shape[0]
        )
        bright_rates_hz = (
            stellar_flux_hz_per_m2_per_bin
            * throughput_fraction
            * area_per_mode_m2
            * field_power_per_wavelength
        )
        planet_contrast = _require_planet_contrast(state)
        dark_rates_hz = planet_contrast * bright_rates_hz
        rates_hz = np.asarray([bright_rates_hz, dark_rates_hz], dtype=np.float64)
        photon_rates = PhotonRates(
            wavelength_grid_um=state.throughput.wavelength_grid_um,
            port_labels=("bright_0", "dark_0"),
            rates_hz=rates_hz,
        ).add_component(
            "detector.ideal_counter",
            {
                "integrated_rate_hz": float(np.sum(rates_hz)),
                "zero_point_rate_hz_per_m2_per_bin": (
                    params.zero_point_rate_hz_per_m2_per_bin
                ),
                "field_power_per_wavelength": field_power_per_wavelength.astype(
                    np.float64,
                    copy=True,
                ),
            },
        )

        planet_separation_mas = _require_planet_separation_mas(state)
        contrast_curve = np.array(
            [(planet_separation_mas, planet_contrast)],
            dtype=CONTRAST_CURVE_DTYPE,
        )
        component_name = unique_component_name(
            self.type_name,
            state.results.components,
            self.context.instance_id,
        )
        port_rates_hz = rates_hz.sum(axis=1)
        results = replace(
            state.results,
            null_depth=planet_contrast,
            contrast_curve=contrast_curve,
        ).add_component(
            component_name,
            {
                "null_depth_raw": float(planet_contrast),
                "null_depth": planet_contrast,
                "contrast_curve_rows": int(contrast_curve.size),
                "port_rates": port_rates_hz.astype(np.float64, copy=True),
                "port_labels": photon_rates.port_labels,
                "integration_time_s": float(params.integration_time_s),
                "integrated_counts_per_port": (
                    port_rates_hz * params.integration_time_s
                ).astype(np.float64, copy=True),
                "source_model": "ideal two-port placeholder detector",
            },
        )
        return replace(state, photon_rates=photon_rates, results=results)


def _require_star_jmag(state: SimulationState) -> float:
    if state.scene.star_jmag is None:
        raise ValueError("detector.ideal_counter requires scene.star_jmag.")
    return float(state.scene.star_jmag)


def _require_planet_contrast(state: SimulationState) -> float:
    if state.scene.planet_contrast is None:
        raise ValueError("detector.ideal_counter requires scene.planet_contrast.")
    return float(state.scene.planet_contrast)


def _require_planet_separation_mas(state: SimulationState) -> float:
    if state.scene.planet_separation_mas is None:
        raise ValueError(
            "detector.ideal_counter requires scene.planet_separation_mas."
        )
    return float(state.scene.planet_separation_mas)


def _require_area_m2(state: SimulationState) -> float:
    if state.geometry.area_m2 is None:
        raise ValueError("detector.ideal_counter requires geometry.area_m2.")
    return float(state.geometry.area_m2)


def _resolve_collecting_area_m2(state: SimulationState) -> float:
    """Total light-collecting area summed over all apertures."""
    if state.geometry.collecting_area_m2 is not None:
        return float(state.geometry.collecting_area_m2)
    area_m2 = _require_area_m2(state)
    if state.geometry.n_apertures is None:
        return area_m2
    return float(state.geometry.n_apertures) * area_m2


def _field_power_per_wavelength(field_amplitudes: np.ndarray) -> np.ndarray:
    n_modes = field_amplitudes.shape[0]
    if n_modes < 1:
        raise ValueError("detector.ideal_counter requires at least one field mode.")
    return np.sum(np.abs(field_amplitudes) ** 2, axis=(0, 2)).astype(
        np.float64,
        copy=False,
    )


def _require_throughput_fraction(state: SimulationState) -> np.ndarray:
    if not state.throughput.transmission_fraction:
        raise ValueError(
            "detector.ideal_counter requires throughput.transmission_fraction."
        )
    throughput_fraction = np.asarray(
        state.throughput.transmission_fraction,
        dtype=np.float64,
    )
    if throughput_fraction.ndim != 1:
        raise ValueError("detector.ideal_counter throughput must be a 1D spectrum.")
    return throughput_fraction


__all__ = ["CONTRAST_CURVE_DTYPE", "IdealCounter"]
