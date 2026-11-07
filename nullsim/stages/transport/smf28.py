"""SMF-28 transport stage for scalar fiber loss, phase, and dispersion.

This SPEC §15.3 first pass wraps KILO transport behavior into the nullsim
pipeline layer without promoting a separate transport physics module yet. The
reference implementation is KILO ``transport.py``: SMF-28e attenuation
interpolation at lines 38-48, ``fiber_phase`` at lines 311-370,
``chromatic_dispersion_phase`` at lines 373-450, and static fiber loss at
lines 434-470. NAYRA keeps the same scalar formulas for the single-bench case
with a different default dispersion constant.

The stage is scalar and unpolarized: it applies wavelength-dependent SMF-28e
power transmission, one common-mode residual OPD phase draw, and a chromatic
dispersion phase derived from a hardware length-mismatch realization drawn
from the stage's RNG. Per SPEC §6.5, two ``transport.smf28`` instances in
the same pipeline (e.g. ``fiber_to_chip`` and ``fiber_to_detector``) receive
independent deterministic draws via ``derive_rng(root_seed, config_hash,
sweep_coord_tuple, instance_id)``: their length-mismatch realizations are
distinct hardware spools. ``dispersion_seed`` overrides the per-instance RNG
when tests need a frozen dispersion draw.

The ``state.throughput`` contribution follows the canonical Throughput
component payload (SPEC §6.2): ``wavelength_grid_um``, ``transmission_fraction``,
plus provenance fields so downstream throughput-breakdown plot stages can
type-dispatch uniformly across transport-family contributions.

Polarization (PM980 + jones_pm_fiber + DualPolField) is deferred to SPEC
§15.3 Pass 3 alongside chip work.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field as PydanticField

from nullsim.pipeline import (
    BaseStage,
    OPDComponent,
    OPDBudget,
    SimulationState,
    Field as StateField,
    Throughput,
    register_stage,
)
from nullsim.stages._utils import unique_component_name

_MICRON_TO_M = 1.0e-6
_M_TO_NM = 1.0e9
_C_LIGHT_M_S = 2.998e8

SMF28_DISPERSION_PS_NM_KM = -40.0
SMF28_GROUP_INDEX = 1.4504
PM980_DISPERSION_PS_NM_KM = -29.5
PM980_GROUP_INDEX = 1.4682
SMF28E_WAVELENGTHS_UM = np.array(
    [0.85, 0.98, 1.00, 1.06, 1.10, 1.20, 1.25, 1.31, 1.383, 1.55, 1.625, 1.70],
    dtype=np.float64,
)
SMF28E_ATTENUATION_DB_PER_KM = np.array(
    [2.1, 1.5, 1.3, 0.9, 0.7, 0.4, 0.35, 0.33, 0.31, 0.19, 0.20, 0.25],
    dtype=np.float64,
)

# PM980-XP polarization-maintaining fiber datasheet curve across YJH.
# Single-mode operation requires wavelength > cutoff (~920 +/- 50 nm);
# values below 0.95 um should be treated as lower bounds. Source:
# kilo/transport.py::pm_fiber_attenuation (Coherent/Nufern datasheet).
# The ~1 dB/km excess over SMF-28e at common wavelengths reflects
# Rayleigh scattering off the boron-doped stress rods that induce the
# fiber's birefringence.
PM980XP_WAVELENGTHS_UM = np.array(
    [0.85, 0.98, 1.06, 1.10, 1.25, 1.31, 1.40, 1.55, 1.625, 1.70],
    dtype=np.float64,
)
PM980XP_ATTENUATION_DB_PER_KM = np.array(
    [3.5, 2.5, 2.0, 1.7, 1.4, 1.2, 1.6, 1.0, 0.95, 1.1],
    dtype=np.float64,
)

# PM780-HP is the visible/short-NIR PM fiber family appropriate for an
# 800 nm HWO band. Coherent/Nufern publishes an operating range of
# 770-1100 nm and a conservative core attenuation spec of <=4 dB/km at
# 850 nm; absent a full curve in nullsim, keep the tabulated helper flat
# at that upper bound.
PM780HP_WAVELENGTHS_UM = np.array([0.77, 0.85, 1.10], dtype=np.float64)
PM780HP_ATTENUATION_DB_PER_KM = np.array([4.0, 4.0, 4.0], dtype=np.float64)


@register_stage
class SMF28Transport(BaseStage):
    """Apply SMF-28 scalar loss, common-mode phase, and dispersion."""

    type_name = "transport.smf28"
    # Draws a per-instance dispersion length-mismatch and common-mode phase
    # from context.rng, so the result depends on the config_hash-seeded stream.
    uses_rng = True
    family = "transport"
    consumes = frozenset({"field", "throughput", "opd_budget"})
    produces = frozenset({"field", "throughput", "opd_budget"})

    class Params(BaseModel):
        """Validated params matching KILO/NAYRA SMF-28 transport names."""

        model_config = ConfigDict(
            extra="forbid",
            frozen=True,
            allow_inf_nan=False,
        )

        length_m: float = PydanticField(gt=0.0)
        attenuation_db_per_km: float | None = PydanticField(default=None, ge=0.0)
        attenuation_curve: Literal["smf28e", "pm980_xp", "pm780_hp"] = "smf28e"
        length_matching_error_m: float = PydanticField(default=0.001, ge=0.0)
        dispersion_ps_nm_km: float = SMF28_DISPERSION_PS_NM_KM
        phi_rms_rad: float = PydanticField(default=0.01, ge=0.0)
        rng_seed: int | None = None
        dispersion_seed: int | None = None

    def apply(self, state: SimulationState) -> SimulationState:
        params = self.params_model
        field_amplitudes = _require_field_amplitudes(state)
        wavelength_grid_um = _require_wavelength_grid_um(state, field_amplitudes)
        wavelength_grid_m = np.asarray(wavelength_grid_um, dtype=np.float64) * _MICRON_TO_M
        wavelength_center_m = _center_wavelength_m(wavelength_grid_m)

        transmission_fraction = _transport_transmission_fraction(params, wavelength_grid_m)
        common_phase_rad = _common_mode_phase_rad(
            params=params,
            wavelength_grid_m=wavelength_grid_m,
            wavelength_center_m=wavelength_center_m,
            context_rng=getattr(self.context, "rng", None),
        )
        dispersion_phase_rad = chromatic_dispersion_phase_rad(
            wavelength_grid_m,
            wavelength_center_m,
            n_modes=field_amplitudes.shape[0],
            length_matching_error_m=params.length_matching_error_m,
            dispersion_ps_nm_km=params.dispersion_ps_nm_km,
            rng=_dispersion_rng(
                dispersion_seed=params.dispersion_seed,
                context_rng=getattr(self.context, "rng", None),
            ),
        )

        field = _updated_field(
            state=state,
            field_amplitudes=field_amplitudes,
            wavelength_grid_um=wavelength_grid_um,
            common_phase_rad=common_phase_rad,
            dispersion_phase_rad=dispersion_phase_rad,
        ).add_component(
            unique_component_name(
                self.type_name, state.field.components, self.context.instance_id,
            ),
            _static_field_factor_payload(
                dispersion_phase_rad=dispersion_phase_rad,
                field_shape=field_amplitudes.shape,
                wavelength_grid_um=wavelength_grid_um,
            ),
        )
        source_model = _attenuation_source_model(params)
        throughput = _updated_throughput(
            state=state,
            wavelength_grid_um=wavelength_grid_um,
            transmission_fraction=transmission_fraction,
            instance_id=self.context.instance_id,
            type_name=self.type_name,
            source_model=source_model,
        )
        opd_budget = _updated_opd_budget(
            state=state,
            params=params,
            wavelength_grid_m=wavelength_grid_m,
            wavelength_center_m=wavelength_center_m,
            dispersion_phase_rad=dispersion_phase_rad,
            instance_id=self.context.instance_id,
        )
        return replace(state, field=field, throughput=throughput, opd_budget=opd_budget)


def smf28e_attenuation_db_per_km(
    wavelength_m: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """Return SMF-28e+ attenuation in dB/km using KILO's datasheet table."""

    wavelength_um = np.asarray(wavelength_m, dtype=np.float64) * 1.0e6
    _validate_curve_domain("SMF-28e", wavelength_um, SMF28E_WAVELENGTHS_UM)
    attenuation = np.interp(
        wavelength_um,
        SMF28E_WAVELENGTHS_UM,
        SMF28E_ATTENUATION_DB_PER_KM,
    )
    if np.ndim(wavelength_m) == 0:
        return float(attenuation)
    return np.asarray(attenuation, dtype=np.float64)


def chromatic_dispersion_phase_rad(
    wavelength_m: NDArray[np.float64],
    wavelength_center_m: float,
    *,
    n_modes: int,
    rng: np.random.Generator,
    length_matching_error_m: float = 0.001,
    dispersion_ps_nm_km: float = SMF28_DISPERSION_PS_NM_KM,
) -> NDArray[np.float64]:
    """Return KILO-style residual chromatic dispersion phase in radians.

    MODELING ASSUMPTION (review 2026-06-10 Issue 21b): only the
    second-order (beta2, GVD) term of the length-mismatch expansion is
    retained. The first-order group-delay term ``(n_g/c) dL domega`` of a
    0.1 mm physical mismatch is ~40+ rad at the J-band edge (OPD ~147 um,
    far beyond the coherence length) — the model therefore implicitly
    assumes per-fiber group delay is trimmed to ~um level by hardware
    (matched cleave lengths plus chip-phase absorption of the design-
    wavelength piston). That trimming requirement is load-bearing for any
    broadband null at the 1e-6 level.

    Args:
        wavelength_m: One-dimensional wavelength grid in meters.
        wavelength_center_m: Expansion wavelength in meters.
        n_modes: Number of scalar field modes.
        rng: Generator used to draw the per-mode length-mismatch vector. Each
            ``transport.smf28`` stage instance must pass its own generator so
            distinct hardware spools see distinct draws (SPEC §6.5). Callers
            that need a frozen draw should pass
            ``np.random.default_rng(42)`` explicitly.
        length_matching_error_m: RMS fiber length mismatch in meters.
        dispersion_ps_nm_km: Dispersion parameter D in ps/(nm km).

    Returns:
        Array with shape ``(n_wavelength_bins, n_modes)``.

    Raises:
        ValueError: If the wavelength grid is not one-dimensional or ``n_modes``
            is not positive.
    """

    wavelengths_m = np.asarray(wavelength_m, dtype=np.float64)
    if wavelengths_m.ndim != 1:
        raise ValueError("transport.smf28 dispersion requires a 1D wavelength grid.")
    if n_modes <= 0:
        raise ValueError("transport.smf28 dispersion requires n_modes > 0.")
    if length_matching_error_m == 0.0 or dispersion_ps_nm_km == 0.0:
        return np.zeros((wavelengths_m.size, n_modes), dtype=np.float64)

    d_si = dispersion_ps_nm_km * 1.0e-6
    beta2_s2_per_m = -(wavelength_center_m**2) * d_si / (2.0 * np.pi * _C_LIGHT_M_S)

    delta_length_m = np.zeros(n_modes, dtype=np.float64)
    if n_modes > 1:
        delta_length_m[1:] = rng.normal(
            0.0,
            length_matching_error_m,
            size=n_modes - 1,
        )

    delta_lambda_m = wavelengths_m - wavelength_center_m
    delta_omega_rad_s = -2.0 * np.pi * _C_LIGHT_M_S * delta_lambda_m / wavelength_center_m**2
    return beta2_s2_per_m * delta_length_m[None, :] * delta_omega_rad_s[:, None] ** 2 / 2.0


def _dispersion_rng(
    *,
    dispersion_seed: int | None,
    context_rng: np.random.Generator | None,
) -> np.random.Generator:
    """Return the RNG used for the dispersion length-mismatch draw.

    Order of precedence: explicit ``dispersion_seed`` (for tests with a frozen draw),
    otherwise the per-instance ``context.rng`` plumbed by the runner. Raises
    when neither is available — earlier this fell back to a fresh
    ``np.random.default_rng()`` seeded from OS entropy, which silently broke
    reproducibility for any caller constructing the stage outside the
    runner (notebooks, standalone tests). CLAUDE.md "pin random seeds where
    reproducibility matters" applies.
    """

    if dispersion_seed is not None:
        return np.random.default_rng(dispersion_seed)
    if context_rng is not None:
        return context_rng
    raise ValueError(
        "transport.smf28: dispersion draw needs a deterministic RNG. "
        "Pin ``dispersion_seed`` on Params or run the stage through the "
        "pipeline runner so context.rng is plumbed."
    )


def _require_field_amplitudes(state: SimulationState) -> NDArray[np.complexfloating]:
    field_amplitudes = state.field.amplitudes
    if field_amplitudes is None:
        raise ValueError("transport.smf28 requires field amplitudes from an earlier stage.")
    field_array = np.asarray(field_amplitudes)
    if field_array.ndim != 3:
        raise ValueError(
            "transport.smf28 requires field amplitudes with shape "
            "[n_modes, n_wavelength_bins, n_pol]."
        )
    return field_array


def _require_wavelength_grid_um(
    state: SimulationState,
    field_amplitudes: NDArray[np.complexfloating],
) -> tuple[float, ...]:
    field_grid_um = state.field.wavelength_grid_um
    throughput_grid_um = state.throughput.wavelength_grid_um
    wavelength_grid_um = field_grid_um or throughput_grid_um
    if not wavelength_grid_um:
        raise ValueError(
            "transport.smf28 requires a non-empty wavelength grid from "
            "the resolved grid config."
        )
    if field_grid_um and throughput_grid_um and not _same_wavelength_grid(
        field_grid_um,
        throughput_grid_um,
    ):
        raise ValueError(
            "transport.smf28 requires field and throughput wavelength grids to match."
        )
    if field_amplitudes.shape[1] != len(wavelength_grid_um):
        raise ValueError(
            "transport.smf28 field wavelength axis must match the wavelength grid."
        )
    if state.throughput.transmission_fraction and (
        len(state.throughput.transmission_fraction) != len(wavelength_grid_um)
    ):
        raise ValueError(
            "transport.smf28 throughput length must match the wavelength grid."
        )
    return wavelength_grid_um


def _same_wavelength_grid(
    first_grid_um: tuple[float, ...],
    second_grid_um: tuple[float, ...],
) -> bool:
    first = np.asarray(first_grid_um, dtype=np.float64)
    second = np.asarray(second_grid_um, dtype=np.float64)
    return bool(first.shape == second.shape and np.array_equal(first, second, equal_nan=True))


def _center_wavelength_m(wavelength_grid_m: NDArray[np.float64]) -> float:
    return float(wavelength_grid_m[len(wavelength_grid_m) // 2])


def _transport_transmission_fraction(
    params: SMF28Transport.Params,
    wavelength_grid_m: NDArray[np.float64],
) -> NDArray[np.float64]:
    attenuation_db_per_km = _attenuation_db_per_km(params, wavelength_grid_m)
    length_km = params.length_m / 1000.0
    return np.asarray(10.0 ** (-(attenuation_db_per_km * length_km) / 10.0), dtype=np.float64)


def _attenuation_db_per_km(
    params: SMF28Transport.Params,
    wavelength_grid_m: NDArray[np.float64],
) -> NDArray[np.float64]:
    if params.attenuation_db_per_km is not None:
        return np.full_like(wavelength_grid_m, params.attenuation_db_per_km, dtype=np.float64)
    if params.attenuation_curve == "pm980_xp":
        return np.asarray(
            pm980xp_attenuation_db_per_km(wavelength_grid_m), dtype=np.float64,
        )
    if params.attenuation_curve == "pm780_hp":
        return np.asarray(
            pm780hp_attenuation_db_per_km(wavelength_grid_m), dtype=np.float64,
        )
    return np.asarray(smf28e_attenuation_db_per_km(wavelength_grid_m), dtype=np.float64)


def _attenuation_source_model(params: SMF28Transport.Params) -> str:
    """Return provenance text for the configured attenuation model."""

    if params.attenuation_db_per_km is not None:
        return (
            f"Fixed {float(params.attenuation_db_per_km):.3g} dB/km "
            "attenuation × length"
        )
    return _curve_source_model(params.attenuation_curve)


def _curve_source_model(curve: str) -> str:
    """Return provenance text for the configured attenuation curve."""

    if curve == "pm980_xp":
        return "PM980-XP attenuation × length (Coherent/Nufern datasheet)"
    if curve == "pm780_hp":
        return "PM780-HP attenuation × length (Coherent/Nufern datasheet)"
    return "SMF-28e attenuation × length (KILO datasheet)"


def pm980xp_attenuation_db_per_km(
    wavelength_m: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """Return PM980-XP attenuation in dB/km from the Coherent/Nufern curve."""

    wavelength_um = np.asarray(wavelength_m, dtype=np.float64) * 1.0e6
    _validate_curve_domain("PM980-XP", wavelength_um, PM980XP_WAVELENGTHS_UM)
    attenuation = np.interp(
        wavelength_um,
        PM980XP_WAVELENGTHS_UM,
        PM980XP_ATTENUATION_DB_PER_KM,
    )
    if np.ndim(wavelength_m) == 0:
        return float(attenuation)
    return np.asarray(attenuation, dtype=np.float64)


def pm780hp_attenuation_db_per_km(
    wavelength_m: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """Return PM780-HP attenuation in dB/km from the conservative spec table."""

    wavelength_um = np.asarray(wavelength_m, dtype=np.float64) * 1.0e6
    _validate_curve_domain("PM780-HP", wavelength_um, PM780HP_WAVELENGTHS_UM)
    attenuation = np.interp(
        wavelength_um,
        PM780HP_WAVELENGTHS_UM,
        PM780HP_ATTENUATION_DB_PER_KM,
    )
    if np.ndim(wavelength_m) == 0:
        return float(attenuation)
    return np.asarray(attenuation, dtype=np.float64)


def _validate_curve_domain(
    curve_name: str,
    wavelength_um: NDArray[np.float64],
    table_wavelength_um: NDArray[np.float64],
) -> None:
    min_wavelength = float(np.min(table_wavelength_um))
    max_wavelength = float(np.max(table_wavelength_um))
    if np.any((wavelength_um < min_wavelength) | (wavelength_um > max_wavelength)):
        requested_min = float(np.min(wavelength_um))
        requested_max = float(np.max(wavelength_um))
        raise ValueError(
            f"{curve_name} attenuation requested outside tabulated wavelength "
            f"range [{min_wavelength:.3g}, {max_wavelength:.3g}] um: "
            f"[{requested_min:.3g}, {requested_max:.3g}] um."
        )


def _common_mode_phase_rad(
    *,
    params: SMF28Transport.Params,
    wavelength_grid_m: NDArray[np.float64],
    wavelength_center_m: float,
    context_rng: np.random.Generator | None,
) -> NDArray[np.float64]:
    if params.phi_rms_rad == 0.0:
        return np.zeros_like(wavelength_grid_m, dtype=np.float64)

    rng = np.random.default_rng(params.rng_seed) if params.rng_seed is not None else context_rng
    if rng is None:
        raise ValueError(
            "transport.smf28: common-mode phase draw needs a deterministic RNG. "
            "Pin ``rng_seed`` on Params or run the stage through the pipeline "
            "runner so context.rng is plumbed."
        )
    phi_center_rad = float(rng.normal(0.0, params.phi_rms_rad))
    opd_sample_m = phi_center_rad * wavelength_center_m / (2.0 * math.pi)
    return 2.0 * math.pi * opd_sample_m / wavelength_grid_m


def _updated_field(
    *,
    state: SimulationState,
    field_amplitudes: NDArray[np.complexfloating],
    wavelength_grid_um: tuple[float, ...],
    common_phase_rad: NDArray[np.float64],
    dispersion_phase_rad: NDArray[np.float64],
) -> StateField:
    # Transport loss is recorded in throughput.transmission_fraction and
    # applied at the detector (|field|² × throughput). Scaling field
    # amplitudes by sqrt(transmission) here as well would double-count it.
    # NaN wavelengths still propagate to NaN field via exp(1j·NaN).
    phase_rad = common_phase_rad[None, :, None] + dispersion_phase_rad.T[:, :, None]
    transported_amplitudes = (
        np.asarray(field_amplitudes, dtype=np.complex128)
        * np.exp(1j * phase_rad)
    )
    return replace(
        state.field,
        amplitudes=transported_amplitudes,
        wavelength_grid_um=wavelength_grid_um,
    )


def _static_field_factor_payload(
    *,
    dispersion_phase_rad: NDArray[np.float64],
    field_shape: tuple[int, ...],
    wavelength_grid_um: tuple[float, ...],
) -> dict[str, object]:
    """Per-mode static field factor this stage applied, for field rebuilders.

    ``sensitivity.detection_curve`` rebuilds its stellar and planet fields
    from coupling and area, so it multiplies them by this factor to see the
    same per-mode GVD phase the deployed chip sees (review 2026-11-07
    Issue 10). The common-mode phase is the same for every mode at each
    wavelength, a global phase the chip cannot see, so it is left out.
    """

    factor = np.exp(1j * dispersion_phase_rad.T)[:, :, np.newaxis]
    return {
        "static_field_factor": np.broadcast_to(factor, field_shape).astype(
            np.complex128, copy=True,
        ),
        "wavelength_grid_um": wavelength_grid_um,
        "source_model": (
            "Per-mode GVD phase exp(i phi_disp[mode, wl]) from the stage's "
            "length-mismatch draw, shape (n_modes, n_wl, n_pol)."
        ),
    }


def _updated_throughput(
    *,
    state: SimulationState,
    wavelength_grid_um: tuple[float, ...],
    transmission_fraction: NDArray[np.float64],
    instance_id: str,
    type_name: str,
    source_model: str,
) -> Throughput:
    current_transmission = _throughput_or_unity(state, transmission_fraction.size)
    updated_transmission = current_transmission * transmission_fraction
    components = dict(state.throughput.components)
    component_payload = {
        "wavelength_grid_um": wavelength_grid_um,
        "transmission_fraction": _float_tuple(transmission_fraction),
        "source_model": source_model,
    }
    component_name = unique_component_name(
        f"{type_name}_loss",
        components,
        instance_id,
    )
    components[component_name] = component_payload
    return Throughput(
        wavelength_grid_um=wavelength_grid_um,
        transmission_fraction=_float_tuple(updated_transmission),
        components=components,
    )


def _throughput_or_unity(
    state: SimulationState,
    n_wavelength_bins: int,
) -> NDArray[np.float64]:
    if state.throughput.transmission_fraction:
        return np.asarray(state.throughput.transmission_fraction, dtype=np.float64)
    return np.ones(n_wavelength_bins, dtype=np.float64)


def _updated_opd_budget(
    *,
    state: SimulationState,
    params: SMF28Transport.Params,
    wavelength_grid_m: NDArray[np.float64],
    wavelength_center_m: float,
    dispersion_phase_rad: NDArray[np.float64],
    instance_id: str,
) -> OPDBudget:
    components = dict(state.opd_budget.components)
    added_rms_values_nm: list[float] = []
    if params.phi_rms_rad > 0.0:
        common_opd_nm = params.phi_rms_rad * wavelength_center_m / (2.0 * math.pi) * _M_TO_NM
        # A phase COMMON to every mode is a global piston: it cannot affect
        # any interferometric observable (the chip unitary is insensitive
        # to a global phase). Recorded for provenance only and deliberately
        # NOT folded into total_rms_opd_nm — booking ~2 nm of inert piston
        # as a differential OPD error contaminated the debug-report
        # waterfall (review 2026-06-10 Issue 21a).
        components[
            unique_component_name("fiber_phase_common_mode", components, instance_id)
        ] = OPDComponent(
            rms_opd_nm=float(common_opd_nm),
            reduction_stage="smf28",
        )

    dispersion_opd_nm = _dispersion_rms_opd_nm(dispersion_phase_rad, wavelength_grid_m)
    components[
        unique_component_name("fiber_chromatic_dispersion", components, instance_id)
    ] = OPDComponent(
        rms_opd_nm=dispersion_opd_nm,
        reduction_stage="smf28",
    )
    added_rms_values_nm.append(dispersion_opd_nm)

    total_rms_opd_nm = state.opd_budget.total_rms_opd_nm
    for rms_opd_nm in added_rms_values_nm:
        total_rms_opd_nm = _quadrature_total(total_rms_opd_nm, rms_opd_nm)
    return replace(
        state.opd_budget,
        components=components,
        total_rms_opd_nm=total_rms_opd_nm,
    )


def _dispersion_rms_opd_nm(
    dispersion_phase_rad: NDArray[np.float64],
    wavelength_grid_m: NDArray[np.float64],
) -> float:
    """RMS OPD of the chromatic dispersion phase across non-reference modes.

    Mode 0 is the reference mode (``delta_length_m[0] = 0``) so its
    dispersion phase is identically zero. Pre-fix the RMS averaged over
    all ``(n_wl, n_modes)`` entries including the zero reference, biasing
    the reported dispersion OPD down by a factor of
    ``sqrt((n_modes - 1) / n_modes)`` — small for KILO 4T (~14%) but
    ~30% for ``n_modes=2``. Code Review 2026-05-16 Issue 11.
    """
    opd_nm = dispersion_phase_rad * wavelength_grid_m[:, None] / (2.0 * math.pi) * _M_TO_NM
    if opd_nm.shape[1] <= 1:
        return 0.0
    return float(np.sqrt(np.mean(np.square(opd_nm[:, 1:]))))


def _quadrature_total(existing_opd_nm: float | None, added_opd_nm: float) -> float:
    if existing_opd_nm is None:
        return added_opd_nm
    return float(math.hypot(existing_opd_nm, added_opd_nm))


def _float_tuple(values: NDArray[np.float64]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


@register_stage
class PM980Transport(SMF28Transport):
    """PM980-XP polarization-maintaining fiber transport.

    Same scalar loss + common-mode phase + dispersion model as
    :class:`SMF28Transport`; the differences are the registered
    ``type_name``, PM980-XP attenuation curve, and the NAYRA single-bench
    dispersion / group-index defaults. Use this stage for NAYRA / KILO
    PM-fiber bench runs
    (PBS-arm → cryostat chip, typically 5–30 m) where the
    polarization-maintaining stress-rod fiber's slightly higher loss
    (~1 dB/km extra at YJH) and birefringence matter physically even
    if the scalar loss formula is identical.
    """

    type_name = "transport.pm980"

    class Params(SMF28Transport.Params):
        attenuation_curve: Literal["smf28e", "pm980_xp", "pm780_hp"] = "pm980_xp"
        dispersion_ps_nm_km: float = PM980_DISPERSION_PS_NM_KM


@register_stage
class PM780Transport(SMF28Transport):
    """PM780-HP polarization-maintaining fiber transport for ~800 nm bands."""

    type_name = "transport.pm780"
    # The inherited apply() reads state.opd_budget.components before
    # producing the budget; declare the read like the parent does
    # (review 2026-06-10 nit — previously worked only because the budget
    # defaults to empty).
    consumes = frozenset({"field", "throughput", "opd_budget"})
    produces = frozenset({"field", "throughput", "opd_budget"})

    class Params(SMF28Transport.Params):
        attenuation_db_per_km: float | None = PydanticField(default=4.0, ge=0.0)
        attenuation_curve: Literal["smf28e", "pm980_xp", "pm780_hp"] = "pm780_hp"


__all__ = [
    "PM780HP_ATTENUATION_DB_PER_KM",
    "PM780HP_WAVELENGTHS_UM",
    "PM780Transport",
    "PM980XP_ATTENUATION_DB_PER_KM",
    "PM980XP_WAVELENGTHS_UM",
    "PM980_DISPERSION_PS_NM_KM",
    "PM980_GROUP_INDEX",
    "PM980Transport",
    "SMF28E_ATTENUATION_DB_PER_KM",
    "SMF28E_WAVELENGTHS_UM",
    "SMF28Transport",
    "chromatic_dispersion_phase_rad",
    "pm780hp_attenuation_db_per_km",
    "pm980xp_attenuation_db_per_km",
    "smf28e_attenuation_db_per_km",
]
