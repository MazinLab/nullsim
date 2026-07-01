"""Geometric delay-line stage for KILO-style path equalization.

This SPEC §15.3 stage wraps the KILO-only delay-line behavior from
``kilo/transport.py`` lines 541-651. It computes the pointing-dependent
photonic delay needed to equalize telescope paths and applies the resulting
power loss to the throughput budget. When explicitly configured, it also
applies the deterministic chromatic phase left by a fiber delay medium after
piston or group-delay control. NAYRA has no equivalent single-bench delay-line
path.

The stage produces a single named contribution on ``state.throughput`` with
the canonical Throughput component payload (SPEC §6.2): ``wavelength_grid_um``,
``transmission_fraction``, plus provenance fields (``per_aperture`` breakdown,
delay lengths, loss in dB, pointing). This makes the per-aperture diagnostic
data persistent through cache hits and keeps downstream throughput-breakdown
plots schema-stable across all transport-family stages.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nullsim.pipeline import BaseStage, SimulationState, Throughput, register_stage
from nullsim.stages._utils import unique_component_name

KILO_DEFAULT_FIBER_LENGTH_M = 780.0
KILO_SMF28_N_EFF = 1.4504
KILO_DELAY_LINE_LOSS_DB_PER_M = 0.034
KILO_FREE_SPACE_DELAY_LINE_LOSS_DB = 2.0
HOLLOW_CORE_PHASE_INDEX_REF = 1.0003
HOLLOW_CORE_GROUP_INDEX_REF = 1.0004
HOLLOW_CORE_DISPERSION_PS_NM_KM = 1.0
HOLLOW_CORE_1060_PM_DISPERSION_PS_NM_KM = 80.0
HOLLOW_CORE_1060_PM_MIN_WAVELENGTH_UM = 1.03
HOLLOW_CORE_1060_PM_MAX_WAVELENGTH_UM = 1.09
GLOPHOTONICS_HCPCF_1030_1064_60_ATTENUATION_DB_PER_M = 0.040
GLOPHOTONICS_HCPCF_1030_1064_60_DISPERSION_PS_NM_KM = -2.0
GLOPHOTONICS_HCPCF_1030_1064_60_MIN_WAVELENGTH_UM = 0.80
GLOPHOTONICS_HCPCF_1030_1064_60_MAX_WAVELENGTH_UM = 1.15
GLOPHOTONICS_HCPCF_1550_60_ATTENUATION_DB_PER_M = 0.050
GLOPHOTONICS_HCPCF_1550_60_DISPERSION_PS_NM_KM = -2.5
GLOPHOTONICS_HCPCF_1550_60_MIN_WAVELENGTH_UM = 1.40
GLOPHOTONICS_HCPCF_1550_60_MAX_WAVELENGTH_UM = 1.75
SOLID_CORE_GROUP_INDEX_REF = 1.4682
SOLID_CORE_DISPERSION_PS_NM_KM = -40.0
_MAX_DELAY_TOLERANCE_M = 1.0e-10
_MICRON_TO_M = 1.0e-6
_C_LIGHT_M_S = 299_792_458.0


@dataclass(frozen=True)
class _DelayMedium:
    name: str
    phase_index_ref: float
    group_index_ref: float
    dispersion_ps_nm_km: float
    polarization_model: str
    polarization_caveat: str | None = None
    dispersion_caveat: str | None = None
    default_attenuation_db_per_m: float | None = None
    min_wavelength_um: float | None = None
    max_wavelength_um: float | None = None


@register_stage
class GeometricDelayLines(BaseStage):
    """Apply KILO geometric delay-line loss from scene pointing and geometry."""

    type_name = "delay_lines.geometric"
    family = "delay_lines"
    # opd_budget intentionally omitted: the opt-in chromatic model applies a
    # deterministic wavelength-dependent field phase and persists diagnostics
    # on the throughput component; it does not add an RMS OPD-noise term.
    consumes = frozenset({"geometry", "throughput", "scene", "field"})
    produces = frozenset({"throughput", "field"})

    class Params(BaseModel):
        """Validated params for geometric delay-line path equalization."""

        model_config = ConfigDict(
            extra="forbid",
            frozen=True,
            allow_inf_nan=False,
        )

        max_delay_m: float = Field(gt=0.0)
        attenuation_db_per_m: float | None = Field(default=None, ge=0.0)
        fiber_lengths_m: tuple[float, ...] | None = None
        n_eff: float = Field(default=KILO_SMF28_N_EFF, gt=0.0)
        fixed_delay_line_loss_db_per_telescope: tuple[float | None, ...] | None = None
        fixed_delay_line_loss_mode: Literal[
            "replace_delay",
            "attenuator",
        ] = "replace_delay"
        delay_medium: Literal[
            "loss_only",
            "solid_core_fiber",
            "hollow_core_fiber",
            "hollow_core_1060_pm",
            "glophotonics_hcpcf_1030_1064_60",
            "glophotonics_hcpcf_1550_60",
            "custom",
        ] = "loss_only"
        chromatic_control: Literal[
            "none",
            "phase_reference",
            "multi_wavelength_group_delay",
        ] = "none"
        reference_wavelength_um: float | None = None
        phase_index_ref: float | None = None
        group_index_ref: float | None = None
        dispersion_ps_nm_km: float | None = None

        @field_validator("fiber_lengths_m")
        @classmethod
        def _validate_fiber_lengths_m(
            cls,
            value: tuple[float, ...] | None,
        ) -> tuple[float, ...] | None:
            if value is None:
                return None
            if not value:
                raise ValueError("fiber_lengths_m must be non-empty when provided.")
            for length_m in value:
                if length_m <= 0.0:
                    raise ValueError("fiber_lengths_m values must be greater than 0.")
            return value

        @field_validator("fixed_delay_line_loss_db_per_telescope")
        @classmethod
        def _validate_fixed_loss_db(
            cls,
            value: tuple[float | None, ...] | None,
        ) -> tuple[float | None, ...] | None:
            if value is None:
                return None
            for loss_db in value:
                if loss_db is not None and loss_db < 0.0:
                    raise ValueError(
                        "fixed_delay_line_loss_db_per_telescope values must be "
                        "non-negative or None."
                    )
            return value

        @model_validator(mode="after")
        def _validate_chromatic_params(self) -> "GeometricDelayLines.Params":
            if self.chromatic_control != "none" and self.delay_medium == "loss_only":
                raise ValueError(
                    "chromatic_control requires delay_medium to be "
                    "'solid_core_fiber', 'hollow_core_fiber', "
                    "'hollow_core_1060_pm', "
                    "'glophotonics_hcpcf_1030_1064_60', "
                    "'glophotonics_hcpcf_1550_60', or 'custom'."
                )
            if (
                self.reference_wavelength_um is not None
                and self.reference_wavelength_um <= 0.0
            ):
                raise ValueError("reference_wavelength_um must be greater than 0.")
            for field_name in ("phase_index_ref", "group_index_ref"):
                value = getattr(self, field_name)
                if value is not None and value <= 0.0:
                    raise ValueError(f"{field_name} must be greater than 0.")
            if self.delay_medium == "custom":
                missing = [
                    field_name
                    for field_name in (
                        "phase_index_ref",
                        "group_index_ref",
                        "dispersion_ps_nm_km",
                    )
                    if getattr(self, field_name) is None
                ]
                if missing:
                    joined = ", ".join(missing)
                    raise ValueError(
                        f"custom delay_medium requires {joined}."
                    )
            return self

    def apply(self, state: SimulationState) -> SimulationState:
        params = self.params_model
        aperture_positions_enu_m = _require_aperture_positions_enu_m(state)
        altitude_deg, azimuth_deg = _require_scene_pointing_deg(state)
        fiber_lengths_m = _fiber_lengths_m(params, aperture_positions_enu_m.shape[0])
        delay_medium = _resolve_delay_medium(params)
        attenuation_db_per_m = _attenuation_db_per_m(params, delay_medium)
        wavelength_grid_um = _require_wavelength_grid_um(state)
        _validate_delay_medium_wavelengths(delay_medium, wavelength_grid_um)
        reference_wavelength_um = _reference_wavelength_um(
            params.reference_wavelength_um,
            wavelength_grid_um,
        )

        geometric_delay_lengths_m = _compute_delay_line_lengths_m(
            aperture_positions_enu_m=aperture_positions_enu_m,
            fiber_lengths_m=fiber_lengths_m,
            n_eff=delay_medium.phase_index_ref,
            altitude_deg=altitude_deg,
            azimuth_deg=azimuth_deg,
        )
        delay_lengths_m, loss_db = _apply_loss_model(
            geometric_delay_lengths_m=geometric_delay_lengths_m,
            attenuation_db_per_m=attenuation_db_per_m,
            n_eff=delay_medium.phase_index_ref,
            fixed_loss_db=params.fixed_delay_line_loss_db_per_telescope,
            fixed_loss_mode=params.fixed_delay_line_loss_mode,
        )
        _validate_max_delay_m(
            delay_lengths_m=delay_lengths_m,
            max_delay_m=params.max_delay_m,
            altitude_deg=altitude_deg,
            azimuth_deg=azimuth_deg,
        )

        per_aperture_transmission = 10.0 ** (-loss_db / 10.0)
        # Code Review 2026-05-16 Issue 6 fix: apply per-aperture losses
        # to ``state.field.amplitudes`` per-mode (one mode per aperture
        # when shape matches), instead of mean-smearing into a scalar
        # ``throughput.transmission_fraction``. The smeared scalar
        # masked per-arm imbalance that the chip's unitary depends on —
        # the dark-port leakage from unequal arm transmission only
        # appears if the chip's input has the right per-mode amplitudes.
        n_apertures = aperture_positions_enu_m.shape[0]
        field_array = (
            np.asarray(state.field.amplitudes)
            if state.field.amplitudes is not None
            else None
        )
        n_modes = field_array.shape[0] if field_array is not None else 0
        # KILO_PARITY_PLAN §4.2: when n_modes is a multiple of n_apertures
        # (the sub-aperture broadcast case — e.g. KILO Keck 2 telescopes ×
        # 12 sub-apertures = 24 modes) each sub-aperture inherits its
        # parent aperture's delay-line transmission. The pre-Phase-4.2
        # fallback mean-smeared into a scalar throughput, dropping the
        # per-arm chromatic loss imbalance the chip's unitary depends on.
        can_apply_per_mode = (
            field_array is not None
            and field_array.ndim == 3
            and n_modes >= n_apertures
            and n_modes % n_apertures == 0
        )
        if params.chromatic_control != "none" and field_array is None:
            raise ValueError(
                "delay_lines.geometric chromatic_control requires field amplitudes "
                "from an earlier injection stage."
            )
        if (
            params.chromatic_control != "none"
            and field_array is not None
            and not can_apply_per_mode
        ):
            raise ValueError(
                "delay_lines.geometric cannot apply chromatic delay-line phase "
                "because field modes are not an integer multiple of apertures."
            )

        chromatic_phase_rad = _chromatic_phase_rad(
            wavelength_grid_um=wavelength_grid_um,
            reference_wavelength_um=reference_wavelength_um,
            delay_lengths_m=delay_lengths_m,
            medium=delay_medium,
            chromatic_control=params.chromatic_control,
        )
        if can_apply_per_mode:
            modes_per_aperture = n_modes // n_apertures
            amp_scale = np.sqrt(per_aperture_transmission)
            per_mode_scale = np.repeat(amp_scale, modes_per_aperture)
            per_mode_phase = np.repeat(
                chromatic_phase_rad,
                modes_per_aperture,
                axis=0,
            )
            new_amplitudes = (
                field_array * per_mode_scale[:, np.newaxis, np.newaxis]
                * np.exp(1j * per_mode_phase[:, :, np.newaxis])
            )
            field = replace(state.field, amplitudes=new_amplitudes)
            # Losses now live in the field. Keep the running throughput
            # unchanged for detector rates, but record the actual
            # field-applied loss in the component so background and
            # loss-budget outputs can debit it once.
            aggregate_transmission_for_diag = float(np.mean(per_aperture_transmission))
            throughput_loss_factor = 1.0
            component_transmission = aggregate_transmission_for_diag
            field_applied_loss = True
        else:
            # Fallback when the field's mode axis is not commensurate with
            # n_apertures (e.g. ragged sub-aperture counts, not yet
            # supported). Mean-smearing the per-aperture losses into a
            # scalar throughput keeps the run from crashing but loses the
            # per-arm imbalance the chip's unitary depends on; flagged in
            # the diagnostics component for downstream visibility.
            field = state.field
            aggregate_transmission_for_diag = float(np.mean(per_aperture_transmission))
            throughput_loss_factor = aggregate_transmission_for_diag
            component_transmission = throughput_loss_factor
            field_applied_loss = False

        per_aperture_diagnostics = _delay_line_diagnostics(
            altitude_deg=altitude_deg,
            azimuth_deg=azimuth_deg,
            fiber_lengths_m=fiber_lengths_m,
            n_eff=delay_medium.phase_index_ref,
            attenuation_db_per_m=attenuation_db_per_m,
            geometric_delay_lengths_m=geometric_delay_lengths_m,
            delay_lengths_m=delay_lengths_m,
            loss_db=loss_db,
            per_aperture_transmission=per_aperture_transmission,
            aggregate_transmission=aggregate_transmission_for_diag,
            fixed_loss_db=params.fixed_delay_line_loss_db_per_telescope,
            fixed_loss_mode=params.fixed_delay_line_loss_mode,
            delay_medium=delay_medium,
            chromatic_control=params.chromatic_control,
            reference_wavelength_um=reference_wavelength_um,
            chromatic_phase_rad=chromatic_phase_rad,
            wavelength_grid_um=wavelength_grid_um,
        )
        per_aperture_diagnostics["per_mode_field_amplitude_applied"] = bool(
            can_apply_per_mode
        )
        throughput = _updated_throughput(
            state=state,
            aggregate_transmission=throughput_loss_factor,
            component_transmission=component_transmission,
            per_aperture_diagnostics=per_aperture_diagnostics,
            field_applied_loss=field_applied_loss,
            instance_id=self.context.instance_id,
        )
        return replace(state, field=field, throughput=throughput)


def _require_aperture_positions_enu_m(
    state: SimulationState,
) -> NDArray[np.float64]:
    positions = state.geometry.aperture_positions_enu_m
    if positions is None:
        raise ValueError(
            "delay_lines.geometric requires geometry.aperture_positions_enu_m "
            "from an earlier telescope stage."
        )
    positions_array = np.asarray(positions, dtype=np.float64)
    if positions_array.ndim != 2 or positions_array.shape[1] != 3:
        raise ValueError(
            "delay_lines.geometric requires geometry.aperture_positions_enu_m "
            "with shape (n_apertures, 3) in ENU meters."
        )
    if positions_array.shape[0] == 0:
        raise ValueError(
            "delay_lines.geometric requires at least one aperture position."
        )
    if not np.all(np.isfinite(positions_array)):
        raise ValueError(
            "delay_lines.geometric requires finite aperture ENU positions."
        )
    if (
        state.geometry.n_apertures is not None
        and state.geometry.n_apertures != positions_array.shape[0]
    ):
        raise ValueError(
            "delay_lines.geometric geometry.n_apertures must match "
            "geometry.aperture_positions_enu_m."
        )
    return positions_array


def _require_scene_pointing_deg(state: SimulationState) -> tuple[float, float]:
    altitude_deg = state.scene.altitude_deg
    azimuth_deg = state.scene.azimuth_deg
    if altitude_deg is None or azimuth_deg is None:
        raise ValueError(
            "delay_lines.geometric requires scene.altitude_deg and scene.azimuth_deg "
            "for the science target; populate them on the Scene sub-budget from "
            "an earlier scene stage (e.g. scene.point_source)."
        )
    return (
        _as_float(altitude_deg, "scene.altitude_deg"),
        _as_float(azimuth_deg, "scene.azimuth_deg"),
    )


def _fiber_lengths_m(
    params: GeometricDelayLines.Params,
    n_apertures: int,
) -> NDArray[np.float64]:
    if params.fiber_lengths_m is None:
        return np.full(n_apertures, KILO_DEFAULT_FIBER_LENGTH_M, dtype=np.float64)

    fiber_lengths_m = np.asarray(params.fiber_lengths_m, dtype=np.float64)
    if fiber_lengths_m.shape != (n_apertures,):
        raise ValueError(
            "delay_lines.geometric fiber_lengths_m length "
            f"{fiber_lengths_m.size} does not match aperture count {n_apertures}."
        )
    return fiber_lengths_m


def _resolve_delay_medium(params: GeometricDelayLines.Params) -> _DelayMedium:
    if params.delay_medium == "loss_only":
        return _DelayMedium(
            name="loss_only",
            phase_index_ref=params.n_eff,
            group_index_ref=params.group_index_ref or params.n_eff,
            dispersion_ps_nm_km=params.dispersion_ps_nm_km or 0.0,
            polarization_model="not_applied",
        )
    if params.delay_medium == "solid_core_fiber":
        return _DelayMedium(
            name="solid_core_fiber",
            phase_index_ref=params.phase_index_ref or params.n_eff,
            group_index_ref=params.group_index_ref or SOLID_CORE_GROUP_INDEX_REF,
            dispersion_ps_nm_km=(
                SOLID_CORE_DISPERSION_PS_NM_KM
                if params.dispersion_ps_nm_km is None
                else params.dispersion_ps_nm_km
            ),
            polarization_model="scalar_common_phase",
        )
    if params.delay_medium == "hollow_core_fiber":
        return _DelayMedium(
            name="hollow_core_fiber",
            phase_index_ref=params.phase_index_ref or HOLLOW_CORE_PHASE_INDEX_REF,
            group_index_ref=params.group_index_ref or HOLLOW_CORE_GROUP_INDEX_REF,
            dispersion_ps_nm_km=(
                HOLLOW_CORE_DISPERSION_PS_NM_KM
                if params.dispersion_ps_nm_km is None
                else params.dispersion_ps_nm_km
            ),
            polarization_model="scalar_common_phase",
            polarization_caveat=(
                "Hollow-core polarization, modal coupling, and PMD are not "
                "modeled; the same scalar delay phase is applied to every "
                "polarization arm."
            ),
        )
    if params.delay_medium == "hollow_core_1060_pm":
        return _DelayMedium(
            name="hollow_core_1060_pm",
            phase_index_ref=params.phase_index_ref or HOLLOW_CORE_PHASE_INDEX_REF,
            group_index_ref=params.group_index_ref or HOLLOW_CORE_GROUP_INDEX_REF,
            dispersion_ps_nm_km=(
                HOLLOW_CORE_1060_PM_DISPERSION_PS_NM_KM
                if params.dispersion_ps_nm_km is None
                else params.dispersion_ps_nm_km
            ),
            polarization_model="catalog_pm_scalar_phase",
            polarization_caveat=(
                "HC-1060-PM group birefringence is represented only as a "
                "stable polarization basis; cross-coupling and PMD are not "
                "yet modeled."
            ),
            min_wavelength_um=HOLLOW_CORE_1060_PM_MIN_WAVELENGTH_UM,
            max_wavelength_um=HOLLOW_CORE_1060_PM_MAX_WAVELENGTH_UM,
        )
    if params.delay_medium == "glophotonics_hcpcf_1030_1064_60":
        return _DelayMedium(
            name="glophotonics_hcpcf_1030_1064_60",
            phase_index_ref=params.phase_index_ref or HOLLOW_CORE_PHASE_INDEX_REF,
            group_index_ref=params.group_index_ref or HOLLOW_CORE_GROUP_INDEX_REF,
            dispersion_ps_nm_km=(
                GLOPHOTONICS_HCPCF_1030_1064_60_DISPERSION_PS_NM_KM
                if params.dispersion_ps_nm_km is None
                else params.dispersion_ps_nm_km
            ),
            polarization_model="assumed_pm_custom_scalar_phase",
            polarization_caveat=(
                "GLOphotonics HCPCF-1030-1064-60 is treated as an assumed "
                "PM-custom scalar hollow-core delay medium; polarization "
                "cross-coupling, PDL, and PMD are not modeled. The public "
                "catalog SKU is not specified as PM."
            ),
            dispersion_caveat=(
                "Catalog dispersion is calculated over the transmission band "
                "as [-2, +1] ps/(nm km); the preset default uses -2.0 and "
                "should be overridden with measured spool data when available."
            ),
            default_attenuation_db_per_m=(
                GLOPHOTONICS_HCPCF_1030_1064_60_ATTENUATION_DB_PER_M
            ),
            min_wavelength_um=GLOPHOTONICS_HCPCF_1030_1064_60_MIN_WAVELENGTH_UM,
            max_wavelength_um=GLOPHOTONICS_HCPCF_1030_1064_60_MAX_WAVELENGTH_UM,
        )
    if params.delay_medium == "glophotonics_hcpcf_1550_60":
        return _DelayMedium(
            name="glophotonics_hcpcf_1550_60",
            phase_index_ref=params.phase_index_ref or HOLLOW_CORE_PHASE_INDEX_REF,
            group_index_ref=params.group_index_ref or HOLLOW_CORE_GROUP_INDEX_REF,
            dispersion_ps_nm_km=(
                GLOPHOTONICS_HCPCF_1550_60_DISPERSION_PS_NM_KM
                if params.dispersion_ps_nm_km is None
                else params.dispersion_ps_nm_km
            ),
            polarization_model="assumed_pm_custom_scalar_phase",
            polarization_caveat=(
                "GLOphotonics HCPCF-1550-60 is treated as an assumed "
                "PM-custom scalar hollow-core delay medium; polarization "
                "cross-coupling, PDL, and PMD are not modeled. The public "
                "catalog SKU is not specified as PM."
            ),
            dispersion_caveat=(
                "Catalog dispersion is calculated over the transmission band "
                "as [-2.5, +0.9] ps/(nm km); the preset default uses -2.5 "
                "and should be overridden with measured spool data when "
                "available."
            ),
            default_attenuation_db_per_m=(
                GLOPHOTONICS_HCPCF_1550_60_ATTENUATION_DB_PER_M
            ),
            min_wavelength_um=GLOPHOTONICS_HCPCF_1550_60_MIN_WAVELENGTH_UM,
            max_wavelength_um=GLOPHOTONICS_HCPCF_1550_60_MAX_WAVELENGTH_UM,
        )
    return _DelayMedium(
        name="custom",
        phase_index_ref=float(params.phase_index_ref),
        group_index_ref=float(params.group_index_ref),
        dispersion_ps_nm_km=float(params.dispersion_ps_nm_km),
        polarization_model="scalar_common_phase",
    )


def _attenuation_db_per_m(
    params: GeometricDelayLines.Params,
    medium: _DelayMedium,
) -> float:
    if params.attenuation_db_per_m is not None:
        return float(params.attenuation_db_per_m)
    if medium.default_attenuation_db_per_m is not None:
        return medium.default_attenuation_db_per_m
    if medium.name in {"loss_only", "solid_core_fiber"}:
        return KILO_DELAY_LINE_LOSS_DB_PER_M
    raise ValueError(
        "delay_lines.geometric attenuation_db_per_m must be set explicitly "
        f"for delay_medium={medium.name!r}; the KILO SMF default is only "
        "applied to loss_only and solid_core_fiber."
    )


def _validate_delay_medium_wavelengths(
    medium: _DelayMedium,
    wavelength_grid_um: tuple[float, ...],
) -> None:
    if medium.min_wavelength_um is None or medium.max_wavelength_um is None:
        return
    wavelengths = np.asarray(wavelength_grid_um, dtype=np.float64)
    below = wavelengths < medium.min_wavelength_um
    above = wavelengths > medium.max_wavelength_um
    if not (np.any(below) or np.any(above)):
        return
    raise ValueError(
        "delay_lines.geometric delay_medium="
        f"{medium.name!r} supports {medium.min_wavelength_um:.3f}-"
        f"{medium.max_wavelength_um:.3f} um; got wavelength_grid_um="
        f"{_float_tuple(wavelengths)}."
    )


def _reference_wavelength_um(
    configured_reference_um: float | None,
    wavelength_grid_um: tuple[float, ...],
) -> float:
    if configured_reference_um is not None:
        return float(configured_reference_um)
    return float(np.median(np.asarray(wavelength_grid_um, dtype=np.float64)))


def _chromatic_phase_rad(
    *,
    wavelength_grid_um: tuple[float, ...],
    reference_wavelength_um: float,
    delay_lengths_m: NDArray[np.float64],
    medium: _DelayMedium,
    chromatic_control: str,
) -> NDArray[np.float64]:
    """Residual delay-line phase after the configured chromatic control.

    ``delay_lengths_m`` is an optical-path-equivalent length at the reference
    wavelength. ``phase_reference`` removes only piston at that wavelength and
    leaves the first-order group-delay slope. ``multi_wavelength_group_delay``
    assumes a chromatic fringe tracker has also removed that slope, so only the
    beta2/GVD curvature remains.
    """

    wavelengths_m = np.asarray(wavelength_grid_um, dtype=np.float64) * _MICRON_TO_M
    delay_lengths = np.asarray(delay_lengths_m, dtype=np.float64)
    phase = np.zeros((delay_lengths.size, wavelengths_m.size), dtype=np.float64)
    if chromatic_control == "none" or delay_lengths.size == 0:
        return phase

    reference_wavelength_m = float(reference_wavelength_um) * _MICRON_TO_M
    physical_lengths_m = delay_lengths / medium.phase_index_ref
    delta_omega_rad_s = 2.0 * np.pi * _C_LIGHT_M_S * (
        1.0 / wavelengths_m - 1.0 / reference_wavelength_m
    )
    # Residual slope after phase-zeroing at the reference wavelength: the
    # compensated vacuum delay carries its own omega-linear phase, so only the
    # group-vs-phase index mismatch survives (review 2026-06-09 Issue 4).
    beta1_residual_s_per_m = (
        medium.group_index_ref - medium.phase_index_ref
    ) / _C_LIGHT_M_S
    dispersion_s_per_m2 = medium.dispersion_ps_nm_km * 1.0e-6
    beta2_s2_per_m = (
        -(reference_wavelength_m**2)
        * dispersion_s_per_m2
        / (2.0 * np.pi * _C_LIGHT_M_S)
    )

    if chromatic_control == "phase_reference":
        phase += (
            physical_lengths_m[:, np.newaxis]
            * beta1_residual_s_per_m
            * delta_omega_rad_s
        )
    elif chromatic_control != "multi_wavelength_group_delay":
        raise ValueError(f"Unknown chromatic_control {chromatic_control!r}.")

    phase += (
        0.5
        * physical_lengths_m[:, np.newaxis]
        * beta2_s2_per_m
        * delta_omega_rad_s[np.newaxis, :] ** 2
    )
    return phase


def per_aperture_transmission(
    *,
    aperture_positions_enu_m: NDArray[np.float64],
    fiber_lengths_m: NDArray[np.float64],
    n_eff: float,
    attenuation_db_per_m: float,
    altitude_deg: float,
    azimuth_deg: float,
    fixed_loss_db: tuple[float | None, ...] | None = None,
    fixed_loss_mode: Literal["replace_delay", "attenuator"] = "replace_delay",
) -> NDArray[np.float64]:
    """Per-aperture multiplicative delay-line transmission at a pointing.

    Exposes the geometry-loss-transmission pipeline used by
    :meth:`GeometricDelayLines.apply` so the realized-null MC can rebuild
    transmissions at additional azimuths (KILO
    ``azimuth_averaged_realized_null_depth``, transport.py L1739).

    Args:
        aperture_positions_enu_m: Per-aperture ENU coordinates, shape ``(N, 3)``.
        fiber_lengths_m: Per-aperture nominal fiber length in meters, shape ``(N,)``.
        n_eff: SMF-28 effective index.
        attenuation_db_per_m: Per-meter attenuation.
        altitude_deg: Source altitude in degrees.
        azimuth_deg: Source azimuth in degrees.
        fixed_loss_db: Optional per-telescope fixed-loss override.
        fixed_loss_mode: ``"replace_delay"`` keeps the historical fallback
            behavior where fixed-loss entries replace the physical delay
            model. ``"attenuator"`` applies the fixed loss while preserving
            geometric delay lengths for phase/replay diagnostics.

    Returns:
        Per-aperture multiplicative transmission, shape ``(N,)``.
    """

    geometric_delay_lengths_m = _compute_delay_line_lengths_m(
        aperture_positions_enu_m=aperture_positions_enu_m,
        fiber_lengths_m=fiber_lengths_m,
        n_eff=n_eff,
        altitude_deg=altitude_deg,
        azimuth_deg=azimuth_deg,
    )
    _, loss_db = _apply_loss_model(
        geometric_delay_lengths_m=geometric_delay_lengths_m,
        attenuation_db_per_m=attenuation_db_per_m,
        n_eff=n_eff,
        fixed_loss_db=fixed_loss_db,
        fixed_loss_mode=fixed_loss_mode,
    )
    return 10.0 ** (-loss_db / 10.0)


def per_aperture_chromatic_phase_rad(
    *,
    aperture_positions_enu_m: NDArray[np.float64],
    fiber_lengths_m: NDArray[np.float64],
    phase_index_ref: float,
    group_index_ref: float,
    dispersion_ps_nm_km: float,
    chromatic_control: str,
    reference_wavelength_um: float,
    wavelength_grid_um: tuple[float, ...],
    altitude_deg: float,
    azimuth_deg: float,
    fixed_loss_db: tuple[float | None, ...] | None = None,
    fixed_loss_mode: Literal["replace_delay", "attenuator"] = "replace_delay",
) -> NDArray[np.float64]:
    """Per-aperture residual delay-line phase at a pointing.

    This mirrors :func:`per_aperture_transmission` for consumers that rebuild
    azimuth-dependent delay-line transport outside the stage, notably the
    realized-null sensitivity pass. The return shape is
    ``(n_apertures, n_wavelength_bins)`` in radians.
    """

    medium = _DelayMedium(
        name="diagnostic_replay",
        phase_index_ref=float(phase_index_ref),
        group_index_ref=float(group_index_ref),
        dispersion_ps_nm_km=float(dispersion_ps_nm_km),
        polarization_model="scalar_common_phase",
    )
    geometric_delay_lengths_m = _compute_delay_line_lengths_m(
        aperture_positions_enu_m=aperture_positions_enu_m,
        fiber_lengths_m=fiber_lengths_m,
        n_eff=medium.phase_index_ref,
        altitude_deg=altitude_deg,
        azimuth_deg=azimuth_deg,
    )
    delay_lengths_m, _ = _apply_loss_model(
        geometric_delay_lengths_m=geometric_delay_lengths_m,
        attenuation_db_per_m=0.0,
        n_eff=medium.phase_index_ref,
        fixed_loss_db=fixed_loss_db,
        fixed_loss_mode=fixed_loss_mode,
    )
    return _chromatic_phase_rad(
        wavelength_grid_um=wavelength_grid_um,
        reference_wavelength_um=reference_wavelength_um,
        delay_lengths_m=delay_lengths_m,
        medium=medium,
        chromatic_control=chromatic_control,
    )


def _compute_delay_line_lengths_m(
    *,
    aperture_positions_enu_m: NDArray[np.float64],
    fiber_lengths_m: NDArray[np.float64],
    n_eff: float,
    altitude_deg: float,
    azimuth_deg: float,
) -> NDArray[np.float64]:
    altitude_rad = np.radians(altitude_deg)
    azimuth_rad = np.radians(azimuth_deg)
    source_unit_vector = np.array(
        [
            np.cos(altitude_rad) * np.sin(azimuth_rad),
            np.cos(altitude_rad) * np.cos(azimuth_rad),
            np.sin(altitude_rad),
        ],
        dtype=np.float64,
    )

    geometric_term_m = -aperture_positions_enu_m @ source_unit_vector
    effective_delay_m = geometric_term_m + fiber_lengths_m * n_eff
    return np.max(effective_delay_m) - effective_delay_m


def _apply_loss_model(
    *,
    geometric_delay_lengths_m: NDArray[np.float64],
    attenuation_db_per_m: float,
    n_eff: float,
    fixed_loss_db: tuple[float | None, ...] | None,
    fixed_loss_mode: Literal["replace_delay", "attenuator"] = "replace_delay",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    delay_lengths_m = np.asarray(geometric_delay_lengths_m, dtype=np.float64)
    loss_db = (delay_lengths_m / n_eff) * attenuation_db_per_m
    if fixed_loss_db is None:
        return delay_lengths_m, loss_db
    if fixed_loss_mode not in {"replace_delay", "attenuator"}:
        raise ValueError(
            "fixed_delay_line_loss_mode must be 'replace_delay' or 'attenuator'."
        )
    if len(fixed_loss_db) != delay_lengths_m.size:
        raise ValueError(
            "fixed_delay_line_loss_db_per_telescope length "
            f"{len(fixed_loss_db)} does not match telescope count "
            f"{delay_lengths_m.size}."
        )

    delay_lengths_m = np.array(delay_lengths_m, dtype=np.float64, copy=True)
    loss_db = np.array(loss_db, dtype=np.float64, copy=True)
    for index, override_loss_db in enumerate(fixed_loss_db):
        if override_loss_db is None:
            continue
        if fixed_loss_mode == "replace_delay":
            delay_lengths_m[index] = 0.0
        loss_db[index] = float(override_loss_db)
    return delay_lengths_m, loss_db


def _validate_max_delay_m(
    *,
    delay_lengths_m: NDArray[np.float64],
    max_delay_m: float,
    altitude_deg: float,
    azimuth_deg: float,
) -> None:
    required_delay_m = float(np.max(delay_lengths_m))
    if np.isnan(required_delay_m):
        return
    if required_delay_m <= max_delay_m + _MAX_DELAY_TOLERANCE_M:
        return
    raise ValueError(
        "delay_lines.geometric max_delay_m is insufficient for "
        f"altitude_deg={altitude_deg}, azimuth_deg={azimuth_deg}: "
        f"requires {required_delay_m:.6g} m, got {max_delay_m:.6g} m."
    )


def _updated_throughput(
    *,
    state: SimulationState,
    aggregate_transmission: float,
    component_transmission: float,
    per_aperture_diagnostics: dict[str, Any],
    field_applied_loss: bool,
    instance_id: str,
) -> Throughput:
    wavelength_grid_um = _require_wavelength_grid_um(state)
    current_transmission = _throughput_or_unity(state, len(wavelength_grid_um))
    updated_transmission = tuple(
        float(value * aggregate_transmission) for value in current_transmission
    )
    component_transmission = tuple(
        float(component_transmission) for _ in wavelength_grid_um
    )
    source_model = "KILO geometric delay-line path equalization"
    if field_applied_loss:
        source_model = (
            "KILO geometric delay-line path equalization; loss applied to "
            "field amplitudes and recorded here for background/loss-budget "
            "accounting, not folded into throughput.transmission_fraction."
        )
    component_payload: dict[str, Any] = {
        "wavelength_grid_um": wavelength_grid_um,
        "transmission_fraction": component_transmission,
        "field_applied_loss": bool(field_applied_loss),
        "field_amplitude_applied": bool(field_applied_loss),
        "per_aperture": per_aperture_diagnostics,
        "source_model": source_model,
    }
    components = dict(state.throughput.components)
    component_name = unique_component_name(
        "delay_line_loss",
        components,
        instance_id,
    )
    return Throughput(
        wavelength_grid_um=wavelength_grid_um,
        transmission_fraction=updated_transmission,
        components=components,
    ).add_component(component_name, component_payload)


def _delay_line_diagnostics(
    *,
    altitude_deg: float,
    azimuth_deg: float,
    fiber_lengths_m: NDArray[np.float64],
    n_eff: float,
    attenuation_db_per_m: float,
    geometric_delay_lengths_m: NDArray[np.float64],
    delay_lengths_m: NDArray[np.float64],
    loss_db: NDArray[np.float64],
    per_aperture_transmission: NDArray[np.float64],
    aggregate_transmission: float,
    fixed_loss_db: tuple[float | None, ...] | None,
    fixed_loss_mode: str,
    delay_medium: _DelayMedium,
    chromatic_control: str,
    reference_wavelength_um: float,
    chromatic_phase_rad: NDArray[np.float64],
    wavelength_grid_um: tuple[float, ...],
) -> dict[str, Any]:
    physical_delay_lengths_m = delay_lengths_m / delay_medium.phase_index_ref
    wavelengths_m = np.asarray(wavelength_grid_um, dtype=np.float64) * _MICRON_TO_M
    reference_wavelength_m = reference_wavelength_um * _MICRON_TO_M
    delta_lambda_m = wavelengths_m - reference_wavelength_m
    dispersion_s_per_m2 = delay_medium.dispersion_ps_nm_km * 1.0e-6
    gvd_delay_ps = (
        dispersion_s_per_m2
        * physical_delay_lengths_m[:, np.newaxis]
        * delta_lambda_m[np.newaxis, :]
        * 1.0e12
    )
    group_delay_ps = (
        delay_medium.group_index_ref
        * physical_delay_lengths_m
        / _C_LIGHT_M_S
        * 1.0e12
    )
    finite_phase = chromatic_phase_rad[np.isfinite(chromatic_phase_rad)]
    return {
        "altitude_deg": altitude_deg,
        "azimuth_deg": azimuth_deg,
        "fiber_lengths_m": _float_tuple(fiber_lengths_m),
        "n_eff": n_eff,
        "phase_index_ref": delay_medium.phase_index_ref,
        "group_index_ref": delay_medium.group_index_ref,
        "dispersion_ps_nm_km": delay_medium.dispersion_ps_nm_km,
        "delay_medium": delay_medium.name,
        "chromatic_control": chromatic_control,
        "reference_wavelength_um": reference_wavelength_um,
        "physical_delay_lengths_m": _float_tuple(physical_delay_lengths_m),
        "attenuation_db_per_m": attenuation_db_per_m,
        "geometric_delay_lengths_m": _float_tuple(geometric_delay_lengths_m),
        "delay_lengths_m": _float_tuple(delay_lengths_m),
        "loss_db": _float_tuple(loss_db),
        "per_aperture_transmission": _float_tuple(per_aperture_transmission),
        "transmission_fraction": aggregate_transmission,
        "fixed_delay_line_loss_db_per_telescope": fixed_loss_db,
        "fixed_delay_line_loss_mode": fixed_loss_mode,
        "polarization_model": delay_medium.polarization_model,
        "polarization_caveat": delay_medium.polarization_caveat,
        "dispersion_caveat": delay_medium.dispersion_caveat,
        "chromatic_phase_rad": _float_matrix_tuple(chromatic_phase_rad),
        "max_abs_chromatic_phase_rad": (
            float(np.max(np.abs(finite_phase))) if finite_phase.size else 0.0
        ),
        "max_group_delay_ps": float(np.max(group_delay_ps))
        if group_delay_ps.size
        else 0.0,
        "max_abs_gvd_delay_ps": float(np.max(np.abs(gvd_delay_ps)))
        if gvd_delay_ps.size
        else 0.0,
        "group_delay_removed_by_control": (
            chromatic_control == "multi_wavelength_group_delay"
        ),
    }


def _require_wavelength_grid_um(state: SimulationState) -> tuple[float, ...]:
    wavelength_grid_um = state.throughput.wavelength_grid_um
    if not wavelength_grid_um:
        raise ValueError(
            "delay_lines.geometric requires a non-empty throughput wavelength grid."
        )
    return wavelength_grid_um


def _throughput_or_unity(
    state: SimulationState,
    n_wavelength_bins: int,
) -> NDArray[np.float64]:
    if not state.throughput.transmission_fraction:
        return np.ones(n_wavelength_bins, dtype=np.float64)

    transmission_fraction = np.asarray(
        state.throughput.transmission_fraction,
        dtype=np.float64,
    )
    if transmission_fraction.shape != (n_wavelength_bins,):
        raise ValueError(
            "delay_lines.geometric throughput length must match the wavelength grid."
        )
    return transmission_fraction


def _as_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"delay_lines.geometric requires numeric {name}.") from exc


def _float_tuple(values: NDArray[np.float64]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _float_matrix_tuple(values: NDArray[np.float64]) -> tuple[tuple[float, ...], ...]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("delay_lines.geometric expected a 2D diagnostic matrix.")
    return tuple(_float_tuple(row) for row in matrix)


__all__ = [
    "GeometricDelayLines",
    "GLOPHOTONICS_HCPCF_1030_1064_60_ATTENUATION_DB_PER_M",
    "GLOPHOTONICS_HCPCF_1030_1064_60_DISPERSION_PS_NM_KM",
    "GLOPHOTONICS_HCPCF_1030_1064_60_MAX_WAVELENGTH_UM",
    "GLOPHOTONICS_HCPCF_1030_1064_60_MIN_WAVELENGTH_UM",
    "GLOPHOTONICS_HCPCF_1550_60_ATTENUATION_DB_PER_M",
    "GLOPHOTONICS_HCPCF_1550_60_DISPERSION_PS_NM_KM",
    "GLOPHOTONICS_HCPCF_1550_60_MAX_WAVELENGTH_UM",
    "GLOPHOTONICS_HCPCF_1550_60_MIN_WAVELENGTH_UM",
    "HOLLOW_CORE_1060_PM_DISPERSION_PS_NM_KM",
    "HOLLOW_CORE_1060_PM_MAX_WAVELENGTH_UM",
    "HOLLOW_CORE_1060_PM_MIN_WAVELENGTH_UM",
    "KILO_DEFAULT_FIBER_LENGTH_M",
    "KILO_DELAY_LINE_LOSS_DB_PER_M",
    "KILO_FREE_SPACE_DELAY_LINE_LOSS_DB",
    "KILO_SMF28_N_EFF",
    "HOLLOW_CORE_DISPERSION_PS_NM_KM",
    "HOLLOW_CORE_GROUP_INDEX_REF",
    "HOLLOW_CORE_PHASE_INDEX_REF",
    "per_aperture_chromatic_phase_rad",
    "per_aperture_transmission",
]
