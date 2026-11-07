"""Closed-loop fringe-tracking residual piston OPD stage (SPEC §15.3 Pass 4b).

The residual follows the full KILO ``piston_servo_residual`` model below.
The former legacy path (``use_full_piston_servo_residual=False``: bare
Shao-Colavita shot noise plus a standalone Conan lag term) is rejected at
validation: its lag variance ``(f_piston/f_servo)^(5/3)`` rad^2 carries no
r0 or upstream piston amplitude, and a fixed phase variance makes the OPD
scale with wavelength.

Full-physics path (KILO_PARITY_PLAN §2.3) — reproduces KILO ``piston_servo_residual``
(``kilo/fringe_tracking.py:335-471``) using the per-component AO piston
breakdown the upstream ``ao.pyramid_wfs`` publishes on
``state.wavefront.components['ao_residual']['piston_opd_nm']``:

    σ²_residual = η · σ²_temp · (f_char / f_s)^{5/3} · g_pred
                + σ²_photon
                + (1 − η) · σ²_temp
                + σ²_irreducible

with ``η`` from :func:`_piston_sensing_efficiency`, ``σ²_temp`` =
piston-filtered AO temporal variance, and ``σ²_irreducible`` =
piston-filtered AO fitting + WFS-noise variance. Capped at the
uncorrected total so the servo never makes things worse.

The result is written to ``state.opd_budget.components``
as the named component ``fringe_tracking.closed_loop_residual`` with

    rms_opd_nm = sqrt(σ²_residual) · λ_center_m / (2 π) · 1e9

The upstream piston components the tracker replaces are marked
``superseded=True`` via ``dataclasses.replace`` so they stay in
``components`` for plot history but stop counting toward the rebuilt
``total_rms_opd_nm`` (the same pattern Pass 1's ``pyramid_wfs`` uses on
``atmos_piston``; see SPEC §6.2).

The stage emits one global residual rather than per-arm components
because the current ``OPDBudget`` is a single global sub-budget.
KILO's multi-telescope ``compute_servo_corrected_piston`` likewise
returns a single per-telescope residual that the caller broadcasts;
per-aperture diagnostics are deferred to a future ``state.results``-
only stage.

References:

- Shao & Colavita 1992, A&A 262, 353 — shot-noise-limited fringe
  estimator.
- Conan, Rousset & Madec 1995, JOSA-A 12, 1559 — piston-specific
  Greenwood frequency ``f_piston = 0.134 · v_wind / D_eff``.
- KILO ``fringe_tracking.py`` lines 67-95 (Shao-Colavita primitive),
  335-471 (``piston_servo_residual`` full-physics reference).
- NAYRA ``fringe_tracking.py`` predictive-default overrides.
- ``tests/stages/notes/fringe_tracking.md`` for the full source survey.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator
from pydantic import Field as PydanticField

from nullsim.physics.fringe_tracking import (
    VIBRATION_BROADBAND_CUTOFF_HZ,
    fusion_effective_broadband_rms_m,
    photon_kalman_residual_budget,
    piston_psd_knee_hz,
    servo_residual_variance_flat_knee,
    vibration_residual_rms_nm,
    von_karman_differential_piston_rms_m,
)
from nullsim.pipeline import (
    BaseStage,
    OPDBudget,
    OPDComponent,
    SimulationState,
    register_stage,
)
from nullsim.stages._science_bands import simultaneous_band_rate_scale
from nullsim.stages._utils import detector_n_pol_arms, unique_component_name

_MICRON_TO_M = 1.0e-6
_M_TO_NM = 1.0e9
_CONAN_PISTON_COEFFICIENT = 0.134
_FIVE_THIRDS = 5.0 / 3.0
# Per-SNSPD count-rate capability (0.1e9 cps); trackers gang multiple
# SNSPDs per port for higher capacity via tracker_n_detectors_per_port.
# Matches the detector.snspd stage's 1e8 Hz device default.
_SNSPD_MAX_RATE_HZ = 1.0e8


@dataclass(frozen=True)
class _ResidualTermsRad2:
    iid_per_mode: float
    common_differential: float
    inter_telescope: float
    vibration: float

    @property
    def total(self) -> float:
        return self.iid_per_mode + self.common_differential


@dataclass(frozen=True)
class _TrackerPhotonBudget:
    photons_per_channel_per_frame: float
    rate_per_port_hz: float
    usable_rate_per_port_hz: float
    rate_cap_hz: float
    rate_clipped: bool


@register_stage
class ClosedLoopFringeTracker(BaseStage):
    """Add closed-loop fringe-tracking residual piston OPD to the budget."""

    type_name = "fringe_tracking.closed_loop"
    family = "fringe_tracking"
    consumes = frozenset({"photon_rates", "opd_budget", "wavefront", "geometry"})
    produces = frozenset({"opd_budget", "results"})
    # Reads ``scene.star_jmag`` (simultaneous-band rate scale) and the ``field``
    # mode count when those params are active, neither declared in ``consumes``.
    cache_key_extras = frozenset({"scene", "field"})

    class Params(BaseModel):
        """Validated params matching KILO/NAYRA fringe-tracking config names."""

        model_config = ConfigDict(
            extra="forbid",
            frozen=True,
            allow_inf_nan=False,
        )

        # Phase 2: None (default) triggers KILO's photon-limited optimum
        # (optimal_piston_servo_bandwidth, KILO fringe_tracking.py L286).
        # An explicit float pins the bandwidth as before (legacy configs).
        servo_bw_hz: float | None = PydanticField(default=None, gt=0.0)
        control_model: Literal["classical", "photon_kalman"] = "classical"
        fringe_visibility: float = PydanticField(default=0.9, gt=0.0, le=1.0)
        n_spectral_channels: int = PydanticField(default=10, ge=1)
        # n_spectral_eff is the KILO ``piston_servo_residual`` averaging
        # factor; used both by ``piston_sensing_efficiency`` (when
        # ``use_full_piston_servo_residual=True``) and reserved on the
        # params for back-compat with existing configs.
        n_spectral_eff: float = PydanticField(default=2.5, gt=0.0)
        predictive_lag_reduction: float = PydanticField(
            default=0.33, gt=0.0, le=1.0
        )
        inter_telescope_piston: bool = False
        outer_scale_m: float = PydanticField(default=25.0, gt=0.0)
        vibration_line_frequencies_hz: tuple[float, ...] = ()
        vibration_line_rms_nm: tuple[float, ...] = ()
        vibration_broadband_rms_nm: float = PydanticField(default=0.0, ge=0.0)
        loop_latency_s: float = PydanticField(default=5.0e-5, gt=0.0)
        accelerometer_fusion: bool = False
        # Hardware-characterization input: coherent broadband vibration fraction
        # observed in accelerometer telemetry, not a free fit parameter.
        fusion_coherent_fraction: float = PydanticField(default=0.7, ge=0.0, le=1.0)
        # Hardware-characterization input: feedforward RMS suppression on the
        # coherent broadband floor, not a free fit parameter.
        fusion_broadband_suppression: float = PydanticField(default=2.5, ge=1.0)
        line_tracking_residual_floor_nm: float = PydanticField(default=2.0, ge=0.0)
        tracker_n_detectors_per_port: int = PydanticField(default=1, ge=1)
        servo_bandwidth_fraction: float = PydanticField(default=0.05, gt=0.0, le=1.0)
        sample_rate_max_hz: float = PydanticField(default=10000.0, gt=20.0)
        # Per-channel dark-count rate at the FT readout (counts/s).
        # 0.0 reproduces KILO's photon-counting limit for MKID/SNSPD;
        # set to ``snspd_dark_rate_hz / n_spectral_channels`` to mirror
        # KILO ``fringe_tracking_phase_noise``'s dark-count contribution.
        dark_rate_hz_per_channel: float = PydanticField(default=0.0, ge=0.0)
        wind_speed_m_s: float = PydanticField(default=10.0, gt=0.0)
        # Phase 2: When None (default), f_char defaults to the
        # subaperture diameter published by upstream ao.pyramid_wfs
        # (``piston_subaperture_diameter_m``). Atmospheric piston between
        # sub-apertures is driven by independent wind crossings of each
        # sub-aperture, so the timescale is d_sub / v_wind (KILO
        # fringe_tracking.py L376). The pre-Phase-2 default fell back to
        # the full primary diameter and over-stated f_char by ~5x for
        # Keck (~10 m / ~1.93 m).
        effective_baseline_m: float | None = PydanticField(default=None, gt=0.0)
        superseded_components: tuple[str, ...] = PydanticField(
            default=(
                "atmos_piston",
                "ao_temporal",
                "ao_fitting",
                "ao_wfs_noise",
            ),
            min_length=1,
        )
        # KILO_PARITY_PLAN §2.3: the residual is the full KILO
        # ``piston_servo_residual`` decomposition (lag_reduced + photon +
        # uncorrectable_temporal + irreducible piston-filtered fitting+WFS
        # noise) read from the upstream ``ao_residual`` component. Kept as a
        # field so existing configs validate; False is rejected below.
        use_full_piston_servo_residual: bool = True
        # Total number of chip-input modes. Used by
        # ``piston_sensing_efficiency`` when the full residual mode is
        # enabled. Falls back to ``geometry.n_apertures × n_subapertures``
        # when None.
        n_modes: int | None = PydanticField(default=None, ge=2)
        # Instruments such as NAYRA observe multiple science bands
        # simultaneously and can feed all those bright-port photons to the
        # fringe tracker. The resolved run still carries one spectral grid;
        # this optional list scales the current-grid bright-port rate to the
        # summed science-band rate for servo photon noise and auto bandwidth.
        simultaneous_science_bands: tuple[str, ...] = ()
        simultaneous_science_band_transmissions: tuple[float, ...] = ()

        @model_validator(mode="after")
        def _validate_vibration_lines(self) -> Self:
            if not self.use_full_piston_servo_residual:
                raise ValueError(
                    "use_full_piston_servo_residual=false selected the removed "
                    "legacy servo-lag model, whose lag variance "
                    "(f_piston/f_servo)^(5/3) rad^2 ignores r0 and the upstream "
                    "AO piston and so scales the OPD residual with wavelength. "
                    "Drop the key (the full KILO piston_servo_residual model is "
                    "the only one) and add an ao.pyramid_wfs stage upstream."
                )
            if len(self.vibration_line_frequencies_hz) != len(self.vibration_line_rms_nm):
                raise ValueError(
                    "vibration_line_frequencies_hz and vibration_line_rms_nm "
                    "must have equal length."
                )
            if any(value < 0.0 for value in self.vibration_line_frequencies_hz):
                raise ValueError("vibration_line_frequencies_hz must be non-negative.")
            if any(value < 0.0 for value in self.vibration_line_rms_nm):
                raise ValueError("vibration_line_rms_nm must be non-negative.")
            return self

    def apply(self, state: SimulationState) -> SimulationState:
        params = self.params_model

        wavelength_grid_um = _require_wavelength_grid_um(state, self.type_name)
        wavelength_center_m = _representative_wavelength_m(wavelength_grid_um)

        upstream_name, upstream_component = _require_upstream_component(
            state, params.superseded_components, self.type_name
        )
        # σ²_atmos is still pulled (and rolled into provenance) so the
        # upstream-component contract stays the same, but the Conan
        # lag formula no longer multiplies by it (Tier 2.1 fix).
        _ = _opd_nm_to_phase_variance_rad2(
            upstream_component.rms_opd_nm, wavelength_center_m
        )

        effective_baseline_m = _effective_baseline_m(state, params, self.type_name)
        ao_payload = state.wavefront.components.get("ao_residual")
        if isinstance(ao_payload, dict):
            v_char_m_s = float(
                ao_payload.get("effective_wind_speed_m_s", params.wind_speed_m_s)
            )
        else:
            v_char_m_s = float(params.wind_speed_m_s)
        f_char_hz = _CONAN_PISTON_COEFFICIENT * v_char_m_s / effective_baseline_m
        if params.control_model == "photon_kalman":
            common_terms, photon_kalman_breakdown = _photon_kalman_common_terms_rad2(
                state=state,
                params=params,
                wavelength_center_m=wavelength_center_m,
                wind_speed_m_s=v_char_m_s,
                stage_name=self.type_name,
            )
            f_3db_hz = float(photon_kalman_breakdown["photon_kalman_f_3db_equiv_hz"])
            # Frame-free photon counting: the real update cadence is the
            # loop latency, not f_3db / servo_bandwidth_fraction (which
            # published a fictional f_sample_hz above sample_rate_max_hz
            # in the same payload — review 2026-06-10).
            sample_rate_hz = 1.0 / float(params.loop_latency_s)
            servo_bandwidth_source = "photon_kalman"
            # Accounting convention (review 2026-06-10 Issue 16):
            # ALL tracker measurement noise is charged to the rank-1
            # inter-telescope common mode (the photon-Kalman estimation
            # term inside common_terms); the per-mode IID budget keeps
            # sigma2_shot=0 under the assumption that per-mode pistons
            # are slow/small and tracked with long effective windows.
            # The lag term below still uses the fast common-mode
            # f_3db — conservative for slow per-mode loops since the
            # lag contribution is negligible at f_char << f_3db.
            sigma2_iid_per_mode, breakdown = _piston_servo_residual_rad2(
                state=state,
                params=params,
                wavelength_center_m=wavelength_center_m,
                f_3db_hz=f_3db_hz,
                f_char_hz=f_char_hz,
                sigma2_shot=0.0,
                stage_name=self.type_name,
            )
            breakdown.update(photon_kalman_breakdown)
        else:
            sample_rate_hz, servo_bandwidth_source = _resolve_sample_rate_hz(
                state=state,
                params=params,
                wavelength_center_m=wavelength_center_m,
                f_char_hz=f_char_hz,
                stage_name=self.type_name,
            )
            f_3db_hz = float(params.servo_bandwidth_fraction) * sample_rate_hz
            sigma2_shot = _photon_residual_variance_rad2(
                state=state,
                params=params,
                stage_name=self.type_name,
                sample_rate_hz=sample_rate_hz,
            )

            sigma2_iid_per_mode, breakdown = _piston_servo_residual_rad2(
                state=state,
                params=params,
                wavelength_center_m=wavelength_center_m,
                f_3db_hz=f_3db_hz,
                f_char_hz=f_char_hz,
                sigma2_shot=sigma2_shot,
                stage_name=self.type_name,
            )
            common_terms = _common_disturbance_terms_rad2(
                state=state,
                params=params,
                wavelength_center_m=wavelength_center_m,
                f_3db_hz=f_3db_hz,
                wind_speed_m_s=v_char_m_s,
                stage_name=self.type_name,
            )
        terms = _ResidualTermsRad2(
            iid_per_mode=float(sigma2_iid_per_mode),
            **common_terms,
        )
        tracker_budget = _tracker_photon_budget(
            state,
            params,
            self.type_name,
            sample_rate_hz=sample_rate_hz,
        )
        breakdown["control_model"] = params.control_model
        breakdown["tracker_rate_per_port_hz"] = float(tracker_budget.rate_per_port_hz)
        breakdown["tracker_usable_rate_per_port_hz"] = float(
            tracker_budget.usable_rate_per_port_hz
        )
        breakdown["tracker_rate_cap_hz"] = float(tracker_budget.rate_cap_hz)
        breakdown["tracker_rate_clipped"] = bool(tracker_budget.rate_clipped)
        sigma2_residual = terms.total
        breakdown["sigma2_residual_rad2"] = float(sigma2_residual)
        breakdown["sigma2_iid_per_mode_rad2"] = float(terms.iid_per_mode)
        breakdown["sigma2_common_differential_rad2"] = float(
            terms.common_differential
        )
        breakdown["sigma2_inter_telescope_rad2"] = float(terms.inter_telescope)
        if params.inter_telescope_piston:
            # The pair-averaged disturbance the servo acts on, for audit.
            breakdown["inter_telescope_uncorrected_piston_rms_m"] = math.sqrt(
                _inter_telescope_piston_variance_m2(state, params, self.type_name)
            )
        breakdown["sigma2_vibration_rad2"] = float(terms.vibration)
        breakdown["servo_bandwidth_hz"] = float(f_3db_hz)
        breakdown["f_sample_hz"] = float(sample_rate_hz)
        breakdown["f_3db_hz"] = float(f_3db_hz)
        breakdown["servo_bandwidth_fraction"] = float(params.servo_bandwidth_fraction)
        breakdown["sample_rate_max_hz"] = float(params.sample_rate_max_hz)
        breakdown["servo_bandwidth_source"] = servo_bandwidth_source
        (
            simultaneous_rate_scale,
            simultaneous_bands,
            current_band,
        ) = _simultaneous_science_band_scale(state, params, self.type_name)
        breakdown["current_science_band"] = current_band
        breakdown["simultaneous_science_bands"] = simultaneous_bands
        breakdown["simultaneous_science_band_rate_scale"] = float(
            simultaneous_rate_scale
        )
        breakdown["simultaneous_science_band_information_scale"] = float(
            _simultaneous_science_band_scale(
                state, params, self.type_name, information_weighted=True,
            )[0]
        )
        residual_rms_opd_nm = _phase_variance_rad2_to_opd_nm(
            sigma2_residual, wavelength_center_m
        )
        residual_iid_per_mode_nm = _phase_variance_rad2_to_opd_nm(
            terms.iid_per_mode,
            wavelength_center_m,
        )
        sigma_inter_telescope_residual_nm = _phase_variance_rad2_to_opd_nm(
            terms.inter_telescope,
            wavelength_center_m,
        )
        sigma_vibration_residual_nm = _phase_variance_rad2_to_opd_nm(
            terms.vibration,
            wavelength_center_m,
        )
        residual_common_differential_nm = _phase_variance_rad2_to_opd_nm(
            terms.common_differential,
            wavelength_center_m,
        )

        opd_budget = _updated_opd_budget(
            state=state,
            superseded_names=params.superseded_components,
            residual_rms_opd_nm=residual_rms_opd_nm,
            instance_id=self.context.instance_id,
        )
        # Publish the σ² breakdown as a results component so the debug
        # report can render the lag / photon / uncorrectable / irreducible
        # waterfall directly (KILO_PARITY_PLAN §2.3 visualization).
        component_name = unique_component_name(
            self.type_name,
            state.results.components,
            self.context.instance_id,
        )
        results = state.results.add_component(
            component_name,
            {
                **breakdown,
                "residual_rms_opd_nm": float(residual_rms_opd_nm),
                "residual_iid_per_mode_nm": float(residual_iid_per_mode_nm),
                "sigma_inter_telescope_residual_nm": float(
                    sigma_inter_telescope_residual_nm
                ),
                "sigma_vibration_residual_nm": float(sigma_vibration_residual_nm),
                "residual_common_differential_nm": float(
                    residual_common_differential_nm
                ),
                "wavelength_center_m": float(wavelength_center_m),
                "upstream_component": upstream_name,
                "source_model": (
                    "nullsim.stages.fringe_tracking.closed_loop "
                    f"({breakdown['mode']})."
                ),
            },
        )
        return replace(state, opd_budget=opd_budget, results=results)


def _require_wavelength_grid_um(
    state: SimulationState, stage_name: str
) -> tuple[float, ...]:
    wavelength_grid_um = (
        state.field.wavelength_grid_um
        or state.wavefront.wavelength_grid_um
        or state.throughput.wavelength_grid_um
        or state.photon_rates.wavelength_grid_um
    )
    if not wavelength_grid_um:
        raise ValueError(
            f"{stage_name} requires a non-empty wavelength grid from a "
            "Pass 1+ stage (field, wavefront, throughput, or photon_rates)."
        )
    return wavelength_grid_um


def _representative_wavelength_m(wavelength_grid_um: tuple[float, ...]) -> float:
    representative_index = len(wavelength_grid_um) // 2
    return float(wavelength_grid_um[representative_index]) * _MICRON_TO_M


def _require_upstream_component(
    state: SimulationState, names: tuple[str, ...], stage_name: str
) -> tuple[str, OPDComponent]:
    components = state.opd_budget.components
    for name in names:
        component = components.get(name)
        if isinstance(component, OPDComponent):
            break
    else:
        raise ValueError(
            f"{stage_name} requires at least one upstream OPDComponent from "
            f"{names!r} in state.opd_budget.components (set "
            "Params.superseded_components to the upstream piston components "
            "that the fringe tracker replaces)."
        )
    if not math.isfinite(component.rms_opd_nm) or component.rms_opd_nm < 0.0:
        raise ValueError(
            f"{stage_name} expects finite non-negative rms_opd_nm on "
            f"upstream component {name!r}, got {component.rms_opd_nm!r}."
        )
    return name, component


def _effective_baseline_m(
    state: SimulationState,
    params: ClosedLoopFringeTracker.Params,
    stage_name: str,
) -> float:
    """Resolve the wind-crossing scale used in the Conan f_char formula.

    Phase 2 (KILO ``fringe_tracking.py`` L376): the atmospheric piston
    timescale between two sub-apertures is set by independent wind
    crossings of each sub-aperture, so ``f_char = 0.134 * v_wind / d_sub``.
    Priority:
      1. Explicit ``params.effective_baseline_m`` (legacy override).
      2. ``piston_subaperture_diameter_m`` from the upstream
         ``ao_residual`` component.
      3. ``state.geometry.diameter_m`` — the pre-Phase-2 fallback,
         retained for atmosphere-only configs that omit ao.pyramid_wfs.
    """

    if params.effective_baseline_m is not None:
        return float(params.effective_baseline_m)

    ao_payload = state.wavefront.components.get("ao_residual")
    if isinstance(ao_payload, dict):
        sub_diameter = ao_payload.get("piston_subaperture_diameter_m")
        if sub_diameter is not None and math.isfinite(float(sub_diameter)) and float(sub_diameter) > 0.0:
            return float(sub_diameter)

    geometry_diameter_m = state.geometry.diameter_m
    if geometry_diameter_m is None or not math.isfinite(geometry_diameter_m):
        raise ValueError(
            f"{stage_name} requires either Params.effective_baseline_m, "
            "an upstream ao.pyramid_wfs publishing "
            "piston_subaperture_diameter_m, or geometry.diameter_m."
        )
    if geometry_diameter_m <= 0.0:
        raise ValueError(
            f"{stage_name} requires positive geometry.diameter_m, got "
            f"{geometry_diameter_m}."
        )
    return float(geometry_diameter_m)


def _resolve_sample_rate_hz(
    *,
    state: SimulationState,
    params: ClosedLoopFringeTracker.Params,
    wavelength_center_m: float,
    f_char_hz: float,
    stage_name: str,
) -> tuple[float, str]:
    """Resolve the fringe-tracker sample rate and a one-word source tag.

    ``params.servo_bw_hz`` remains the user-facing explicit 3 dB bandwidth.
    When pinned, the implied sample rate is ``f_3db / bandwidth_fraction``.
    When omitted, the stage grid-searches sample rate directly.
    """

    if params.servo_bw_hz is not None:
        return (
            float(params.servo_bw_hz) / float(params.servo_bandwidth_fraction),
            "params",
        )

    f_grid_hz = np.logspace(
        math.log10(20.0),
        math.log10(float(params.sample_rate_max_hz)),
        241,
    )
    return (
        _optimize_sample_rate_hz(
            state=state,
            params=params,
            wavelength_center_m=wavelength_center_m,
            f_char_hz=f_char_hz,
            f_grid_hz=f_grid_hz,
            stage_name=stage_name,
        ),
        "auto",
    )


def _optimize_sample_rate_hz(
    *,
    state: SimulationState,
    params: ClosedLoopFringeTracker.Params,
    wavelength_center_m: float,
    f_char_hz: float,
    f_grid_hz: np.ndarray,
    stage_name: str,
) -> float:
    """Grid-minimize total residual variance over fringe-tracker sample rate."""

    total_variances = np.array(
        [
            _total_residual_variance_for_sample_rate_rad2(
                state=state,
                params=params,
                wavelength_center_m=wavelength_center_m,
                f_char_hz=f_char_hz,
                sample_rate_hz=float(sample_rate_hz),
                stage_name=stage_name,
            )
            for sample_rate_hz in f_grid_hz
        ],
        dtype=np.float64,
    )
    finite = np.isfinite(total_variances)
    if not finite.any():
        raise ValueError(
            f"{stage_name}: sample-rate optimizer found no finite residual "
            "variance; pin servo_bw_hz explicitly or check the photon budget."
        )
    masked = np.where(finite, total_variances, np.inf)
    return float(f_grid_hz[int(np.argmin(masked))])


def _total_residual_variance_for_sample_rate_rad2(
    *,
    state: SimulationState,
    params: ClosedLoopFringeTracker.Params,
    wavelength_center_m: float,
    f_char_hz: float,
    sample_rate_hz: float,
    stage_name: str,
) -> float:
    f_3db_hz = float(params.servo_bandwidth_fraction) * float(sample_rate_hz)
    sigma2_photon = _photon_residual_variance_rad2(
        state=state,
        params=params,
        stage_name=stage_name,
        sample_rate_hz=sample_rate_hz,
    )
    sigma2_iid_per_mode, _breakdown = _piston_servo_residual_rad2(
        state=state,
        params=params,
        wavelength_center_m=wavelength_center_m,
        f_3db_hz=f_3db_hz,
        f_char_hz=f_char_hz,
        sigma2_shot=sigma2_photon,
        stage_name=stage_name,
    )

    common_terms = _common_disturbance_terms_rad2(
        state=state,
        params=params,
        wavelength_center_m=wavelength_center_m,
        f_3db_hz=f_3db_hz,
        wind_speed_m_s=_effective_wind_speed_m_s(state, params),
        stage_name=stage_name,
    )
    return float(sigma2_iid_per_mode + common_terms["common_differential"])


def _effective_wind_speed_m_s(
    state: SimulationState,
    params: ClosedLoopFringeTracker.Params,
) -> float:
    ao_payload = state.wavefront.components.get("ao_residual")
    if isinstance(ao_payload, dict):
        return float(ao_payload.get("effective_wind_speed_m_s", params.wind_speed_m_s))
    return float(params.wind_speed_m_s)


def _photon_residual_variance_rad2(
    *,
    state: SimulationState,
    params: ClosedLoopFringeTracker.Params,
    stage_name: str,
    sample_rate_hz: float,
) -> float:
    sigma2_shot_frame, _budget = _shot_frame_variance_and_budget_rad2(
        state,
        params,
        stage_name,
        sample_rate_hz=sample_rate_hz,
    )
    if not math.isfinite(sigma2_shot_frame):
        return sigma2_shot_frame
    return float(
        (math.pi / 2.0)
        * 2.0
        * float(params.servo_bandwidth_fraction)
        * sigma2_shot_frame
    )


def _shot_frame_variance_and_budget_rad2(
    state: SimulationState,
    params: ClosedLoopFringeTracker.Params,
    stage_name: str,
    *,
    sample_rate_hz: float,
) -> tuple[float, _TrackerPhotonBudget]:
    budget = _tracker_photon_budget(
        state,
        params,
        stage_name,
        sample_rate_hz=sample_rate_hz,
    )
    sigma2_per_channel = _shao_colavita_shot_variance_rad2(
        photons_per_channel_per_frame=budget.photons_per_channel_per_frame,
        dark_per_channel_per_frame=params.dark_rate_hz_per_channel
        / float(sample_rate_hz),
        visibility=params.fringe_visibility,
    )
    if not math.isfinite(sigma2_per_channel):
        return sigma2_per_channel, budget
    return sigma2_per_channel / float(params.n_spectral_channels), budget


def _tracker_photon_budget(
    state: SimulationState,
    params: ClosedLoopFringeTracker.Params,
    stage_name: str,
    *,
    sample_rate_hz: float,
) -> _TrackerPhotonBudget:
    rates_hz = state.photon_rates.rates_hz
    if rates_hz is None:
        raise ValueError(
            f"{stage_name} requires state.photon_rates.rates_hz populated by "
            "a Pass 4a detector stage (SNSPDDetector, MKIDDetector, or "
            "IdealCounter)."
        )
    rates = np.asarray(rates_hz, dtype=np.float64)
    if rates.ndim != 2:
        raise ValueError(
            f"{stage_name} expects photon_rates.rates_hz with shape "
            f"(n_ports, n_wavelength); got shape {rates.shape!r}."
        )
    if rates.size == 0:
        raise ValueError(
            f"{stage_name} received empty photon_rates.rates_hz; cannot "
            "compute shot-noise variance."
        )
    if not np.all(np.isfinite(rates)) or bool(np.any(rates < 0.0)):
        raise ValueError(
            f"{stage_name} expects finite non-negative photon_rates.rates_hz."
        )

    # FT shot noise is set by the BRIGHT-port fringe photons, not the dark-
    # port residuals (those would inflate noise on a bad chip without
    # actually carrying fringe signal). Mask on port_labels — same
    # convention as performance.standard._partition_port_rates.
    port_labels = state.photon_rates.port_labels
    if port_labels is None or len(port_labels) != rates.shape[0]:
        raise ValueError(
            f"{stage_name} requires photon_rates.port_labels matching "
            f"rates_hz axis 0 (got {len(port_labels) if port_labels else 0} "
            f"labels vs {rates.shape[0]} ports)."
        )
    bright_mask = np.array(
        [str(label).lower().startswith("bright") for label in port_labels],
        dtype=bool,
    )
    if not bright_mask.any():
        raise ValueError(
            f"{stage_name} found no bright ports in photon_rates.port_labels; "
            "fringe-tracking shot noise needs bright-port fringe photons."
        )

    # KILO_PARITY_PLAN §2.5: KILO ``fringe_tracking_phase_noise``
    # (kilo/fringe_tracking.py:159-162) distributes the total bright-
    # port rate evenly across ``n_bright_ports`` so the per-baseline
    # phase estimator sees per-port × per-channel photons, not the
    # total over all bright ports. Without this division the photon
    # rate per channel is n_bright_ports× too high — Keck 6-bright =
    # 6× over, √6 ≈ 2.4× SNR overestimate on the FT path.
    n_bright_channels = int(bright_mask.sum())
    bright_rate_hz = float(rates[bright_mask].sum())
    # Dual-pol PBS-arm detectors publish each tracker port as n_pol_arms
    # channels (bright_0_A, bright_0_B, ...). Both arms of one physical
    # port carry the same piston fringe, so the per-port estimator pools
    # their counts: divide the summed bright rate by PHYSICAL ports, not
    # detector channels (otherwise the per-port shot variance doubles).
    # Each arm chip carries its own tracker detector set, so the per-port
    # SNSPD capacity scales with the arm count.
    n_pol_arms = detector_n_pol_arms(state)
    if n_pol_arms > 1 and n_bright_channels % n_pol_arms == 0:
        n_bright_ports = n_bright_channels // n_pol_arms
    else:
        n_pol_arms = 1
        n_bright_ports = n_bright_channels
    # Physical photons set the detector cap; information-equivalent photons
    # at the reference wavelength set the phase noise (see
    # simultaneous_band_rate_scale). The cap clips every band alike.
    photon_scale, _bands, _current_band = _simultaneous_science_band_scale(
        state, params, stage_name,
    )
    information_scale, _bands, _current_band = _simultaneous_science_band_scale(
        state, params, stage_name, information_weighted=True,
    )
    rate_per_port_hz = bright_rate_hz * photon_scale / max(n_bright_ports, 1)
    information_rate_per_port_hz = (
        bright_rate_hz * information_scale / max(n_bright_ports, 1)
    )
    if params.control_model == "photon_kalman":
        rate_cap_hz = (
            float(params.tracker_n_detectors_per_port)
            * n_pol_arms
            * _SNSPD_MAX_RATE_HZ
        )
        rate_clipped = rate_per_port_hz > rate_cap_hz
        usable_rate_per_port_hz = information_rate_per_port_hz * min(
            1.0, rate_cap_hz / rate_per_port_hz if rate_per_port_hz > 0.0 else 1.0
        )
    else:
        rate_cap_hz = 0.0
        usable_rate_per_port_hz = information_rate_per_port_hz
        rate_clipped = False
    tau_frame_s = 1.0 / float(sample_rate_hz)
    rate_per_channel_hz = usable_rate_per_port_hz / params.n_spectral_channels
    return _TrackerPhotonBudget(
        photons_per_channel_per_frame=float(rate_per_channel_hz * tau_frame_s),
        rate_per_port_hz=float(rate_per_port_hz),
        usable_rate_per_port_hz=float(usable_rate_per_port_hz),
        rate_cap_hz=float(rate_cap_hz),
        rate_clipped=bool(rate_clipped),
    )


def _simultaneous_science_band_scale(
    state: SimulationState,
    params: ClosedLoopFringeTracker.Params,
    stage_name: str,
    *,
    information_weighted: bool = False,
) -> tuple[float, tuple[str, ...], str]:
    """Return bright-rate scaling for simultaneous science-band FT photons.

    ``information_weighted`` weights each band by ``(lambda_ref/lambda_b)^2``
    with ``lambda_ref`` the closed-loop reference wavelength (the grid
    middle bin), giving information-equivalent photons for OPD estimation.
    """

    if state.scene.star_jmag is None:
        if params.simultaneous_science_bands:
            raise ValueError(
                f"{stage_name}: simultaneous_science_bands requires "
                "scene.star_jmag so the extra band photon budget can be "
                "computed."
            )
        wavelength_grid_um = _require_wavelength_grid_um(state, stage_name)
        return 1.0, (), _infer_current_band(wavelength_grid_um, stage_name)
    wavelength_grid_um = (
        state.photon_rates.wavelength_grid_um
        or _require_wavelength_grid_um(state, stage_name)
    )
    if not params.simultaneous_science_bands:
        return 1.0, (), _infer_current_band(wavelength_grid_um, stage_name)
    reference_wavelength_um = (
        _representative_wavelength_m(_require_wavelength_grid_um(state, stage_name))
        / _MICRON_TO_M
        if information_weighted
        else None
    )
    return simultaneous_band_rate_scale(
        magnitude=float(state.scene.star_jmag),
        current_wavelength_grid_um=wavelength_grid_um,
        science_bands=params.simultaneous_science_bands,
        band_transmissions=params.simultaneous_science_band_transmissions,
        stage_name=stage_name,
        reference_wavelength_um=reference_wavelength_um,
    )


def _infer_current_band(
    wavelength_grid_um: tuple[float, ...],
    stage_name: str,
) -> str:
    from nullsim.stages._science_bands import infer_band_from_wavelength_um

    if not wavelength_grid_um:
        raise ValueError(f"{stage_name}: wavelength grid is empty.")
    return infer_band_from_wavelength_um(
        wavelength_grid_um[len(wavelength_grid_um) // 2]
    )


def _shao_colavita_shot_variance_rad2(
    *,
    photons_per_channel_per_frame: float,
    dark_per_channel_per_frame: float,
    visibility: float,
) -> float:
    """Per-frame Shao-Colavita 1992 shot-noise phase variance in rad^2.

    Direct port of KILO ``shot_noise_phase_variance_per_frame``
    (kilo/fringe_tracking.py:67-95):

        sigma2_phi_per_channel = (N_s + N_d) / (V^2 * N_s^2)

    where ``N_s`` is the per-channel signal photons per frame and
    ``N_d`` the corresponding dark counts per frame. Cross-channel
    averaging (division by ``n_spectral_channels``) is applied by the
    caller. Returns ``inf`` when the photon budget collapses.
    """

    if photons_per_channel_per_frame <= 0.0 or visibility <= 0.0:
        return float("inf")
    n_signal = photons_per_channel_per_frame
    n_dark = dark_per_channel_per_frame
    denom = visibility * visibility * n_signal * n_signal
    if denom <= 0.0:
        return float("inf")
    return float((n_signal + n_dark) / denom)


def _piston_sensing_efficiency(
    *,
    n_bright: int,
    n_spectral_eff: float,
    total_modes: int,
) -> float:
    """Fraction of independent piston DOFs observable from bright-port fringes.

    Direct port of KILO ``piston_sensing_efficiency``
    (``kilo/fringe_tracking.py:257``). Each bright port × spectral
    channel provides one complex visibility measurement; with
    ``n_bright`` ports and ``n_spectral_eff`` effective channels the
    number of observable piston DOFs is
    ``min(2 × n_bright × n_spectral_eff, M − 1)`` (one global piston is
    unobservable).
    """
    if n_bright <= 0 or total_modes <= 1:
        return 0.0
    n_dof = total_modes - 1
    n_observable = min(2.0 * n_bright * n_spectral_eff, float(n_dof))
    return min(1.0, n_observable / n_dof)


def _piston_servo_residual_rad2(
    *,
    state: SimulationState,
    params: ClosedLoopFringeTracker.Params,
    wavelength_center_m: float,
    f_3db_hz: float,
    f_char_hz: float,
    sigma2_shot: float,
    stage_name: str,
) -> tuple[float, dict[str, float]]:
    """Full KILO ``piston_servo_residual`` decomposition (KILO_PARITY_PLAN §2.3).

        σ²_residual = η · σ²_temp · (f_char / f_s)^{5/3} · g_pred
                    + σ²_photon
                    + (1 − η) · σ²_temp
                    + σ²_irreducible

    where ``σ²_temp``, ``σ²_irreducible = σ²_fitting + σ²_wfs_noise`` are
    the piston-filtered AO components published by the upstream
    ``ao.pyramid_wfs`` stage. ``η`` is the bright-port × spectral-
    channel piston sensing efficiency.

    Returns ``(sigma2_residual, breakdown_dict)``. The breakdown is
    used by the FT-stage's results component for the debug-report
    waterfall.
    """

    ao_payload = state.wavefront.components.get("ao_residual")
    if not isinstance(ao_payload, dict):
        raise ValueError(
            f"{stage_name}: the piston servo residual requires an "
            "upstream ao.pyramid_wfs stage that publishes the piston "
            "breakdown on the 'ao_residual' wavefront component."
        )
    piston = ao_payload.get("piston_opd_nm")
    if not isinstance(piston, dict):
        raise ValueError(
            f"{stage_name}: 'ao_residual' component is missing the "
            "piston_opd_nm dict (expected keys: temporal, fitting, "
            "wfs_noise, total). Re-run with a current ao.pyramid_wfs."
        )

    sigma2_temporal = _opd_nm_to_phase_variance_rad2(
        _scalar_opd_nm(piston, "temporal", stage_name),
        wavelength_center_m,
    )
    sigma2_fitting = _opd_nm_to_phase_variance_rad2(
        _scalar_opd_nm(piston, "fitting", stage_name),
        wavelength_center_m,
    )
    sigma2_wfs_noise = _opd_nm_to_phase_variance_rad2(
        _scalar_opd_nm(piston, "wfs_noise", stage_name),
        wavelength_center_m,
    )
    sigma2_irreducible = sigma2_fitting + sigma2_wfs_noise

    n_bright_ports = _resolve_n_bright_ports(state, stage_name)
    n_modes = _resolve_n_modes(state, params, stage_name)
    eta = _piston_sensing_efficiency(
        n_bright=n_bright_ports,
        n_spectral_eff=float(params.n_spectral_eff),
        total_modes=n_modes,
    )

    f_piston_hz = float(f_char_hz)
    bandwidth_ratio = f_piston_hz / f_3db_hz
    sigma2_temp_correctable = eta * sigma2_temporal
    sigma2_temp_uncorrectable = (1.0 - eta) * sigma2_temporal
    sigma2_lag = (
        params.predictive_lag_reduction
        * sigma2_temp_correctable
        * bandwidth_ratio ** _FIVE_THIRDS
    )

    sigma2_residual_raw = (
        sigma2_lag + sigma2_shot + sigma2_temp_uncorrectable + sigma2_irreducible
    )
    # The raw sum IS the published residual. The KILO-heritage cap at the
    # open-loop total (``piston_servo_residual`` lines 451-454) hid the
    # sensor shot noise from both the published residual and the sample-rate
    # optimizer: above the cap the iid term went rate-independent, so the
    # bandwidth scan always pinned at sample_rate_max while still claiming
    # full common-mode rejection (2026-06-10 classical-ablation flux-scaling
    # bug; J=12 residual understated ~7x). A loop forced faster than its
    # photon budget honestly does worse than open loop; the numeric
    # sample-rate optimizer avoids that regime on its own. The
    # ``shot_exceeds_open_loop`` flag remains as a diagnostic.
    sigma2_uncorrected = sigma2_temporal + sigma2_irreducible
    shot_exceeds_open_loop = sigma2_residual_raw > sigma2_uncorrected
    sigma2_residual = sigma2_residual_raw

    breakdown = {
        "mode": "full_piston_servo_residual",
        "sigma2_lag_reduced_rad2": float(sigma2_lag),
        "sigma2_photon_rad2": float(sigma2_shot),
        "sigma2_uncorrectable_temporal_rad2": float(sigma2_temp_uncorrectable),
        "sigma2_irreducible_rad2": float(sigma2_irreducible),
        "sigma2_residual_rad2": float(sigma2_residual),
        "sigma2_uncorrected_rad2": float(sigma2_uncorrected),
        "sigma2_temporal_rad2": float(sigma2_temporal),
        "sigma2_fitting_rad2": float(sigma2_fitting),
        "sigma2_wfs_noise_rad2": float(sigma2_wfs_noise),
        "piston_sensing_efficiency": float(eta),
        "f_piston_hz": float(f_piston_hz),
        "n_bright_ports": int(n_bright_ports),
        "n_modes": int(n_modes),
        "shot_exceeds_open_loop": bool(shot_exceeds_open_loop),
    }
    return sigma2_residual, breakdown


def _photon_kalman_common_terms_rad2(
    *,
    state: SimulationState,
    params: ClosedLoopFringeTracker.Params,
    wavelength_center_m: float,
    wind_speed_m_s: float,
    stage_name: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Return the Phase B frame-free common-mode residual budget."""

    rad_per_m = 2.0 * math.pi / float(wavelength_center_m)
    atm_sigma2_m2 = 0.0
    atm_f_knee_hz = 0.0
    if params.inter_telescope_piston:
        primary_diameter_m = _primary_diameter_m(state, stage_name)
        atm_sigma2_m2 = _inter_telescope_piston_variance_m2(state, params, stage_name)
        atm_f_knee_hz = piston_psd_knee_hz(
            wind_speed_m_s=float(wind_speed_m_s),
            outer_scale_m=float(params.outer_scale_m),
            aperture_diameter_m=primary_diameter_m,
        )

    raw_broadband_rms_m = float(params.vibration_broadband_rms_nm) / _M_TO_NM
    if params.accelerometer_fusion and raw_broadband_rms_m > 0.0:
        broadband_rms_m = fusion_effective_broadband_rms_m(
            broadband_rms_m=raw_broadband_rms_m,
            coherent_fraction=float(params.fusion_coherent_fraction),
            suppression=float(params.fusion_broadband_suppression),
        )
    else:
        broadband_rms_m = raw_broadband_rms_m

    crlb_sample_rate_hz = 1.0 / float(params.loop_latency_s)
    sigma2_shot_frame_rad2, tracker_budget = _shot_frame_variance_and_budget_rad2(
        state,
        params,
        stage_name,
        sample_rate_hz=crlb_sample_rate_hz,
    )
    if not math.isfinite(sigma2_shot_frame_rad2):
        raise ValueError(
            f"{stage_name}: photon_kalman control requires a finite bright-port "
            "shot-noise CRLB; check photon_rates and tracker rate caps."
        )
    sigma_shot_frame_m = (
        math.sqrt(max(float(sigma2_shot_frame_rad2), 0.0))
        * float(wavelength_center_m)
        / (2.0 * math.pi)
    )
    crlb_m_sqrt_s = sigma_shot_frame_m * math.sqrt(1.0 / crlb_sample_rate_hz)
    result = photon_kalman_residual_budget(
        atm_sigma2_m2=atm_sigma2_m2,
        atm_f_knee_hz=atm_f_knee_hz,
        broadband_sigma2_m2=broadband_rms_m**2,
        broadband_max_hz=VIBRATION_BROADBAND_CUTOFF_HZ,
        tracked_line_count=len(params.vibration_line_frequencies_hz),
        line_tracking_floor_m=float(params.line_tracking_residual_floor_nm) / _M_TO_NM,
        crlb_m_sqrt_s=crlb_m_sqrt_s,
        latency_s=float(params.loop_latency_s),
    )
    estimation_rad2 = result.estimation_m2 * rad_per_m**2
    prediction_rad2 = result.prediction_m2 * rad_per_m**2
    line_floor_rad2 = result.line_floor_m2 * rad_per_m**2
    total_rad2 = result.total_m2 * rad_per_m**2

    common_terms = {
        "common_differential": float(total_rad2),
        # Label caveat: in photon_kalman mode the prediction residual
        # covers BOTH the von Karman atmospheric piston and the broadband
        # vibration disturbance (both live inside the structure function);
        # the "vibration" bucket carries only the narrowband line floor
        # (review 2026-06-10). Totals are unaffected.
        "inter_telescope": float(prediction_rad2),
        "vibration": float(line_floor_rad2),
    }
    breakdown = {
        "pk_prediction_includes_broadband_vibration": True,
        "photon_kalman_window_s": float(result.window_s),
        "photon_kalman_f_3db_equiv_hz": float(result.f_3db_equiv_hz),
        "photon_kalman_estimation_nm": float(math.sqrt(result.estimation_m2) * _M_TO_NM),
        "photon_kalman_prediction_nm": float(math.sqrt(result.prediction_m2) * _M_TO_NM),
        "photon_kalman_line_floor_nm": float(math.sqrt(result.line_floor_m2) * _M_TO_NM),
        "photon_kalman_total_common_nm": float(math.sqrt(result.total_m2) * _M_TO_NM),
        "sigma2_photon_kalman_estimation_rad2": float(estimation_rad2),
        "sigma2_photon_kalman_prediction_rad2": float(prediction_rad2),
        "sigma2_photon_kalman_line_floor_rad2": float(line_floor_rad2),
        "sigma2_photon_kalman_total_common_rad2": float(total_rad2),
        "sigma2_photon_rad2": float(estimation_rad2),
        "loop_latency_s": float(params.loop_latency_s),
        "crlb_m_sqrt_s": float(crlb_m_sqrt_s),
        "crlb_sample_rate_hz": float(crlb_sample_rate_hz),
        "accelerometer_fusion": bool(params.accelerometer_fusion),
        "fusion_coherent_fraction": float(params.fusion_coherent_fraction),
        "fusion_broadband_suppression": float(params.fusion_broadband_suppression),
        "vibration_broadband_rms_effective_nm": float(broadband_rms_m * _M_TO_NM),
        "tracked_line_count": int(len(params.vibration_line_frequencies_hz)),
        "line_tracking_residual_floor_nm": float(
            params.line_tracking_residual_floor_nm
        ),
        "tracker_rate_per_port_hz": float(tracker_budget.rate_per_port_hz),
        "tracker_usable_rate_per_port_hz": float(
            tracker_budget.usable_rate_per_port_hz
        ),
        "tracker_rate_cap_hz": float(tracker_budget.rate_cap_hz),
        "tracker_rate_clipped": bool(tracker_budget.rate_clipped),
    }
    return common_terms, breakdown


def _common_disturbance_terms_rad2(
    *,
    state: SimulationState,
    params: ClosedLoopFringeTracker.Params,
    wavelength_center_m: float,
    f_3db_hz: float,
    wind_speed_m_s: float,
    stage_name: str,
) -> dict[str, float]:
    rad_per_m = 2.0 * math.pi / float(wavelength_center_m)
    sigma2_inter_rad2 = 0.0
    if params.inter_telescope_piston:
        primary_diameter_m = _primary_diameter_m(state, stage_name)
        sigma2_it_m2 = _inter_telescope_piston_variance_m2(state, params, stage_name)
        f_knee_hz = piston_psd_knee_hz(
            wind_speed_m_s=float(wind_speed_m_s),
            outer_scale_m=float(params.outer_scale_m),
            aperture_diameter_m=primary_diameter_m,
        )
        sigma2_inter_m2 = servo_residual_variance_flat_knee(
            sigma2=sigma2_it_m2,
            f_knee_hz=f_knee_hz,
            f_3db_hz=float(f_3db_hz),
            predictive_gain_variance=1.0 / float(params.predictive_lag_reduction),
        )
        sigma2_inter_rad2 = sigma2_inter_m2 * rad_per_m**2

    sigma2_vibration_rad2 = 0.0
    if params.vibration_line_frequencies_hz or params.vibration_broadband_rms_nm > 0.0:
        broadband_rms_nm = float(params.vibration_broadband_rms_nm)
        if params.accelerometer_fusion and broadband_rms_nm > 0.0:
            broadband_rms_m = fusion_effective_broadband_rms_m(
                broadband_rms_m=broadband_rms_nm / _M_TO_NM,
                coherent_fraction=float(params.fusion_coherent_fraction),
                suppression=float(params.fusion_broadband_suppression),
            )
            broadband_rms_nm = broadband_rms_m * _M_TO_NM
        sigma_vibration_nm = vibration_residual_rms_nm(
            line_frequencies_hz=params.vibration_line_frequencies_hz,
            line_rms_nm=params.vibration_line_rms_nm,
            broadband_rms_nm=broadband_rms_nm,
            broadband_max_hz=VIBRATION_BROADBAND_CUTOFF_HZ,
            f_3db_hz=float(f_3db_hz),
        )
        sigma2_vibration_rad2 = (sigma_vibration_nm / _M_TO_NM * rad_per_m) ** 2

    return {
        "common_differential": float(sigma2_inter_rad2 + sigma2_vibration_rad2),
        "inter_telescope": float(sigma2_inter_rad2),
        "vibration": float(sigma2_vibration_rad2),
    }


def _resolve_r0_500nm_m(state: SimulationState, stage_name: str) -> float:
    """Return r0 at 500 nm along the line of sight, ``r0_zenith X^(-3/5)``.

    The AO and atmosphere payloads store the zenith r0 beside the airmass
    they applied; the bare ``wavefront.r0_at_500nm_m`` fallback is zenith.
    """

    for name in ("ao_residual", "atmosphere"):
        payload = state.wavefront.components.get(name)
        if not isinstance(payload, dict) or payload.get("r0_500nm_m") is None:
            continue
        r0_500nm_m = float(payload["r0_500nm_m"])
        if math.isfinite(r0_500nm_m) and r0_500nm_m > 0.0:
            return r0_500nm_m * float(payload.get("airmass") or 1.0) ** -0.6
    if state.wavefront.r0_at_500nm_m is not None:
        r0_500nm_m = float(state.wavefront.r0_at_500nm_m)
        if math.isfinite(r0_500nm_m) and r0_500nm_m > 0.0:
            return r0_500nm_m
    raise ValueError(
        f"{stage_name}: inter_telescope_piston=True requires r0_500nm_m from "
        "ao_residual or wavefront.r0_at_500nm_m."
    )


def _inter_telescope_piston_variance_m2(
    state: SimulationState,
    params: ClosedLoopFringeTracker.Params,
    stage_name: str,
) -> float:
    """Uncorrected inter-telescope differential piston variance in m^2.

    Conan et al. (1995) aperture-filtered von Karman differential piston for
    each telescope pair, at its horizontal (E, N) baseline from
    ``geometry.aperture_positions_enu_m`` and the full primary diameter
    ``geometry.diameter_m`` (each telescope's piston averages its whole
    pupil), then averaged over pairs. Without aperture positions the pair is
    treated as fully decorrelated (B -> inf), the saturated upper bound.
    """

    r0_500nm_m = _resolve_r0_500nm_m(state, stage_name)
    diameter_m = _primary_diameter_m(state, stage_name)
    positions = state.geometry.aperture_positions_enu_m
    baselines_m: list[float] = [math.inf]
    if positions is not None and len(positions) >= 2:
        xy = np.asarray(positions, dtype=np.float64)[:, :2]
        i, j = np.triu_indices(xy.shape[0], k=1)
        baselines_m = [float(b) for b in np.linalg.norm(xy[i] - xy[j], axis=1)]
    return float(
        np.mean(
            [
                von_karman_differential_piston_rms_m(
                    r0_500nm_m=r0_500nm_m,
                    outer_scale_m=float(params.outer_scale_m),
                    baseline_m=baseline_m,
                    aperture_diameter_m=diameter_m,
                )
                ** 2
                for baseline_m in baselines_m
            ]
        )
    )


def _primary_diameter_m(state: SimulationState, stage_name: str) -> float:
    diameter_m = state.geometry.diameter_m
    if diameter_m is None:
        raise ValueError(
            f"{stage_name}: inter_telescope_piston=True requires geometry.diameter_m."
        )
    diameter_value = float(diameter_m)
    if not math.isfinite(diameter_value) or diameter_value <= 0.0:
        raise ValueError(
            f"{stage_name}: geometry.diameter_m must be finite and positive, "
            f"got {diameter_m}."
        )
    return diameter_value


def _scalar_opd_nm(piston: dict[str, Any], key: str, stage_name: str) -> float:
    value = piston.get(key)
    if value is None:
        raise ValueError(
            f"{stage_name}: piston_opd_nm missing required key {key!r}."
        )
    if isinstance(value, (tuple, list)):
        if not value:
            raise ValueError(
                f"{stage_name}: piston_opd_nm[{key!r}] is empty."
            )
        return float(value[len(value) // 2])
    return float(value)


def _resolve_n_bright_ports(state: SimulationState, stage_name: str) -> int:
    port_labels = state.photon_rates.port_labels
    if port_labels is None:
        raise ValueError(
            f"{stage_name}: photon_rates.port_labels missing; cannot resolve "
            "n_bright_ports for the piston sensing efficiency."
        )
    n_bright = sum(
        1 for label in port_labels if str(label).lower().startswith("bright")
    )
    if n_bright <= 0:
        raise ValueError(
            f"{stage_name}: photon_rates.port_labels has no 'bright' ports."
        )
    # Dual-pol PBS-arm detectors publish each logical port once per arm
    # (bright_0_A, bright_0_B, ...); the sensing efficiency counts logical
    # ports, the same pooling _tracker_photon_budget applies.
    n_pol_arms = detector_n_pol_arms(state)
    if n_pol_arms > 1 and n_bright % n_pol_arms == 0:
        return n_bright // n_pol_arms
    return n_bright


def _resolve_n_modes(
    state: SimulationState,
    params: ClosedLoopFringeTracker.Params,
    stage_name: str,
) -> int:
    if params.n_modes is not None:
        return int(params.n_modes)
    field = state.field
    if field is not None and field.amplitudes is not None:
        return int(field.amplitudes.shape[0])
    n_apertures = state.geometry.n_apertures
    if n_apertures is not None:
        ao_payload = state.wavefront.components.get("ao_residual")
        if isinstance(ao_payload, dict):
            n_sub = ao_payload.get("n_subapertures_per_aperture")
            if isinstance(n_sub, int) and n_sub >= 1:
                return int(n_apertures) * int(n_sub)
        return int(n_apertures)
    raise ValueError(
        f"{stage_name}: cannot resolve n_modes for the piston sensing "
        "efficiency. Set Params.n_modes explicitly, or run a field-aware "
        "upstream stage."
    )


def _opd_nm_to_phase_variance_rad2(
    rms_opd_nm: float, wavelength_center_m: float
) -> float:
    """Convert an RMS OPD in nanometers to phase variance in rad^2."""

    rms_opd_m = rms_opd_nm / _M_TO_NM
    sigma_rad = 2.0 * math.pi * rms_opd_m / wavelength_center_m
    return sigma_rad * sigma_rad


def _phase_variance_rad2_to_opd_nm(
    sigma2_rad2: float, wavelength_center_m: float
) -> float:
    """Convert a phase variance in rad^2 to an RMS OPD in nanometers."""

    if not math.isfinite(sigma2_rad2):
        return float("inf")
    sigma_rad = math.sqrt(max(sigma2_rad2, 0.0))
    return sigma_rad * wavelength_center_m / (2.0 * math.pi) * _M_TO_NM


def _updated_opd_budget(
    *,
    state: SimulationState,
    superseded_names: tuple[str, ...],
    residual_rms_opd_nm: float,
    instance_id: str,
) -> OPDBudget:
    # Mark upstream piston components superseded so they stay in
    # `components` for plot history but stop counting toward the rebuilt
    # quadrature total. Mirrors pyramid_wfs's treatment of `atmos_piston`.
    updated_components: dict[str, Any] = dict(state.opd_budget.components)
    for name in superseded_names:
        upstream = updated_components.get(name)
        if isinstance(upstream, OPDComponent) and not upstream.superseded:
            updated_components[name] = replace(upstream, superseded=True)

    residual_name = unique_component_name(
        "fringe_tracking.closed_loop_residual",
        updated_components,
        instance_id,
    )
    updated_components[residual_name] = OPDComponent(
        rms_opd_nm=float(residual_rms_opd_nm),
        reduction_stage="fringe_tracking.closed_loop",
    )

    total_rms_opd_nm = math.sqrt(
        sum(
            component.rms_opd_nm * component.rms_opd_nm
            for component in updated_components.values()
            if isinstance(component, OPDComponent) and not component.superseded
        )
    )
    return replace(
        state.opd_budget,
        components=updated_components,
        total_rms_opd_nm=total_rms_opd_nm,
    )


__all__ = ["ClosedLoopFringeTracker"]
