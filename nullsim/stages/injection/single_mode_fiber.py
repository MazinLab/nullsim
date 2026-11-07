"""Single-mode fiber injection stage backed by Ruilier-Cassaing coupling.

This SPEC §15.3 stage wraps the pure ``nullsim.physics.ruilier_cassaing``
primitive for on-axis stellar coupling into a guided mode. It follows the
Ruilier (1999, SPIE 3350) Gaussian-mode overlap convention: coupling
efficiency is a power transmission, so field amplitudes are scaled by the
square root of that spectrum.

The ``state.throughput`` contribution follows the canonical Throughput
component payload (SPEC §6.2): ``wavelength_grid_um``, ``transmission_fraction``,
plus provenance fields (``coupling_efficiency``, ``strehl_ratio``,
``aperture_diameter_m``, geometry parameters) so downstream throughput-
breakdown plot stages can type-dispatch uniformly across injection,
transport, and delay-line contributions.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field as PydanticField

from nullsim.pipeline import BaseStage, Field, SimulationState, Throughput, register_stage
from nullsim.physics import ruilier_cassaing
from nullsim.stages._utils import unique_component_name


@register_stage
class SingleModeFiber(BaseStage):
    """Apply Ruilier-Cassaing coupling to field amplitudes and throughput."""

    type_name = "injection.single_mode_fiber"
    family = "injection"
    # 'geometry' is consumed for diagnostics only (aperture_diameter_m audit field);
    # the Ruilier-Cassaing formula is dimensionless and does not use diameter_m.
    consumes = frozenset({"field", "throughput", "wavefront", "geometry"})
    produces = frozenset({"field", "throughput"})

    class Params(BaseModel):
        """Validated params for the Ruilier-Cassaing injection stage."""

        model_config = ConfigDict(
            extra="forbid",
            frozen=True,
            allow_inf_nan=False,
        )

        central_obscuration_ratio: float = PydanticField(default=0.0, ge=0.0, lt=1.0)
        pupil_radius_to_mode_field_radius_ratio: float = PydanticField(
            default=ruilier_cassaing.OPTIMAL_PUPIL_RADIUS_TO_MODE_FIELD_RADIUS_RATIO,
            gt=0.0,
        )

    def apply(self, state: SimulationState) -> SimulationState:
        params = self.params_model
        wavelength_grid_um = _require_wavelength_grid_um(state)
        field_amplitudes = _require_field_amplitudes(state, wavelength_grid_um)
        strehl_ratio = _require_strehl_ratio(state, wavelength_grid_um)
        aperture_diameter_m = _require_aperture_diameter_m(state)

        coupling_spectrum = _coupling_efficiency_spectrum(strehl_ratio, params)
        field = _updated_field(state, field_amplitudes, coupling_spectrum)
        throughput = _updated_throughput(
            state=state,
            wavelength_grid_um=wavelength_grid_um,
            coupling_spectrum=coupling_spectrum,
            strehl_ratio=strehl_ratio,
            aperture_diameter_m=aperture_diameter_m,
            params=params,
            instance_id=self.context.instance_id,
        )
        return replace(state, field=field, throughput=throughput)


def _require_wavelength_grid_um(state: SimulationState) -> tuple[float, ...]:
    wavelength_grid_um = state.field.wavelength_grid_um
    if not wavelength_grid_um:
        raise ValueError(
            "injection.single_mode_fiber requires a non-empty "
            "field.wavelength_grid_um from an earlier field-producing stage."
        )
    return wavelength_grid_um


def _require_field_amplitudes(
    state: SimulationState,
    wavelength_grid_um: tuple[float, ...],
) -> np.ndarray:
    field_amplitudes = state.field.amplitudes
    if field_amplitudes is None:
        raise ValueError(
            "injection.single_mode_fiber requires field amplitudes from "
            "an earlier stage."
        )
    field_array = np.asarray(field_amplitudes, dtype=np.complex128)
    if field_array.ndim != 3:
        raise ValueError(
            "injection.single_mode_fiber requires field amplitudes with shape "
            "[n_modes, n_wavelength_bins, n_pol]."
        )
    if field_array.shape[1] != len(wavelength_grid_um):
        raise ValueError(
            "injection.single_mode_fiber requires the field wavelength axis "
            "to match field.wavelength_grid_um."
        )
    return field_array


def _require_strehl_ratio(
    state: SimulationState,
    wavelength_grid_um: tuple[float, ...],
) -> np.ndarray:
    if not state.wavefront.strehl_ratio:
        raise ValueError(
            "injection.single_mode_fiber requires wavefront.strehl_ratio "
            "from an earlier atmosphere or AO stage."
        )
    try:
        strehl_ratio = np.asarray(state.wavefront.strehl_ratio, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "injection.single_mode_fiber requires wavefront.strehl_ratio "
            "to contain numeric values."
        ) from exc
    if strehl_ratio.ndim != 1 or strehl_ratio.shape[0] != len(wavelength_grid_um):
        raise ValueError(
            "injection.single_mode_fiber requires wavefront.strehl_ratio "
            "to match field.wavelength_grid_um length."
        )
    if np.any(np.isinf(strehl_ratio)):
        raise ValueError(
            "injection.single_mode_fiber requires wavefront.strehl_ratio "
            "values to be finite or NaN."
        )
    return strehl_ratio


def _require_aperture_diameter_m(state: SimulationState) -> float:
    # diameter_m is provenance-only — stored in the throughput component for audit
    # purposes; the coupling formula uses the dimensionless ρ ratio, not this value.
    aperture_diameter_m = state.geometry.diameter_m
    if aperture_diameter_m is None:
        raise ValueError(
            "injection.single_mode_fiber requires geometry.diameter_m from "
            "an earlier telescope stage."
        )
    if not math.isfinite(aperture_diameter_m) or aperture_diameter_m <= 0.0:
        raise ValueError(
            "injection.single_mode_fiber requires a finite positive "
            f"geometry.diameter_m, got {aperture_diameter_m}."
        )
    return aperture_diameter_m


def _coupling_efficiency_spectrum(
    strehl_ratio: np.ndarray,
    params: SingleModeFiber.Params,
) -> np.ndarray:
    coupling_spectrum = np.full(strehl_ratio.shape, np.nan, dtype=np.float64)
    finite_mask = np.isfinite(strehl_ratio)
    finite_strehl = strehl_ratio[finite_mask]
    finite_coupling = ruilier_cassaing.coupling_efficiency(
        finite_strehl,
        central_obscuration_ratio=params.central_obscuration_ratio,
    )
    coupling_spectrum[finite_mask] = _apply_mode_radius_override(
        np.asarray(finite_coupling, dtype=np.float64),
        params,
    )
    return coupling_spectrum


def _apply_mode_radius_override(
    default_coupling: np.ndarray,
    params: SingleModeFiber.Params,
) -> np.ndarray:
    default_ratio = ruilier_cassaing.OPTIMAL_PUPIL_RADIUS_TO_MODE_FIELD_RADIUS_RATIO
    if params.pupil_radius_to_mode_field_radius_ratio == default_ratio:
        return default_coupling

    default_limit = ruilier_cassaing.geometric_coupling_limit(
        params.central_obscuration_ratio,
    )
    override_limit = ruilier_cassaing.geometric_coupling_limit(
        params.central_obscuration_ratio,
        pupil_radius_to_mode_field_radius_ratio=(
            params.pupil_radius_to_mode_field_radius_ratio
        ),
    )
    return default_coupling * (float(override_limit) / float(default_limit))


def _updated_field(
    state: SimulationState,
    field_amplitudes: np.ndarray,
    coupling_spectrum: np.ndarray,
) -> Field:
    amplitude_scale = np.sqrt(coupling_spectrum)
    scaled_amplitudes = field_amplitudes * amplitude_scale[np.newaxis, :, np.newaxis]
    return replace(state.field, amplitudes=scaled_amplitudes)


def _updated_throughput(
    *,
    state: SimulationState,
    wavelength_grid_um: tuple[float, ...],
    coupling_spectrum: np.ndarray,
    strehl_ratio: np.ndarray,
    aperture_diameter_m: float,
    params: SingleModeFiber.Params,
    instance_id: str,
) -> Throughput:
    if (
        state.throughput.wavelength_grid_um
        and state.throughput.wavelength_grid_um != wavelength_grid_um
    ):
        raise ValueError(
            "injection.single_mode_fiber requires throughput.wavelength_grid_um "
            "to match field.wavelength_grid_um."
        )

    # Coupling loss is carried into the field amplitudes via sqrt(coupling)
    # (KILO injection convention, kilo/injection.py:222). Recording the same
    # factor in throughput.transmission_fraction would double-count it at the
    # detector (|field|² × throughput). We keep the existing transmission
    # untouched and only register the coupling spectrum as a provenance
    # component for throughput-breakdown reporting.
    transmission_fraction = _throughput_or_unity(state, len(wavelength_grid_um))
    component_name = unique_component_name(
        "single_mode_fiber",
        state.throughput.components,
        instance_id,
    )
    return replace(
        state.throughput,
        wavelength_grid_um=wavelength_grid_um,
        transmission_fraction=_float_tuple(transmission_fraction),
    ).add_component(
        component_name,
        _throughput_component(
            wavelength_grid_um=wavelength_grid_um,
            coupling_spectrum=coupling_spectrum,
            strehl_ratio=strehl_ratio,
            aperture_diameter_m=aperture_diameter_m,
            params=params,
        ),
    )


def _throughput_or_unity(
    state: SimulationState,
    n_wavelength_bins: int,
) -> tuple[float, ...]:
    if state.throughput.transmission_fraction:
        if len(state.throughput.transmission_fraction) != n_wavelength_bins:
            raise ValueError(
                "injection.single_mode_fiber throughput length must match "
                "field.wavelength_grid_um."
            )
        return tuple(float(value) for value in state.throughput.transmission_fraction)
    return tuple(1.0 for _ in range(n_wavelength_bins))


def _throughput_component(
    *,
    wavelength_grid_um: tuple[float, ...],
    coupling_spectrum: np.ndarray,
    strehl_ratio: np.ndarray,
    aperture_diameter_m: float,
    params: SingleModeFiber.Params,
) -> dict[str, Any]:
    coupling_tuple = _float_tuple(coupling_spectrum)
    return {
        "wavelength_grid_um": wavelength_grid_um,
        "transmission_fraction": coupling_tuple,
        "coupling_efficiency": coupling_tuple,
        "strehl_ratio": _float_tuple(strehl_ratio),
        "central_obscuration_ratio": params.central_obscuration_ratio,
        "pupil_radius_to_mode_field_radius_ratio": (
            params.pupil_radius_to_mode_field_radius_ratio
        ),
        "aperture_diameter_m": aperture_diameter_m,
        "source_model": "Ruilier 1999 Gaussian-mode overlap",
        # Coupling is carried in the field amplitudes (see _updated_throughput),
        # so flag this component field-applied: stages that reconstruct an
        # absolute stellar rate from the throughput budget
        # (characterization.spectro) must fold the coupling back in, while the
        # detector reads the top-level transmission_fraction and gets it from
        # |field|^2.
        "field_amplitude_applied": True,
        # The Strehl x overlap coupling is a point-source factor: a diffuse
        # background's single-mode coupling is fixed by the étendue, so
        # background stages read diffuse_transmission_fraction instead. This
        # coupling has no Fresnel term, so nothing is lost for diffuse light
        # (same exemption as injection.ideal; review 2026-11-07 row 42).
        "point_source_coupling_only": True,
        "diffuse_transmission_fraction": tuple(1.0 for _ in wavelength_grid_um),
    }


def _float_tuple(values: np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


__all__ = ["SingleModeFiber"]
