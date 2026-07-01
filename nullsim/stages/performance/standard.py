"""Standard terminal performance stage (SPEC §15.3 Pass 4c).

Reference implementation: KILO ``performance.py`` (1430+ lines) in
``kilo/performance.py``. The three
formulas the stage implements all originate there:

- ``signal_to_noise`` (KILO lines 204-292):
      SNR = S / sqrt(S + B + D + Sky + systematic_variance)
- ``integration_time_for_snr`` (KILO lines 802-894): closed-form
  inversion of the SNR formula for the requested ``target_snr``.
- ``contrast_curve`` (KILO lines 896-1008): per-separation solution of
  the quadratic ``S^2 - snr^2 S - snr^2 N = 0`` for the
  minimum-detectable planet count, divided by the stellar count to give
  a contrast ratio.

The systematic-variance term has two regimes:

    with FT averaging:   systematic = eps^2 * B_rate^2 * t / (2 * f_servo)   [linear in t]
    without FT:          systematic = (eps * B)^2                            [quadratic in t]

The without-FT regime has a hard SNR ceiling at ``S / (eps * B)``; the
with-FT regime removes that ceiling because the systematic is linear in
``t`` like Poisson noise. The crossover is documented in KILO's
``signal_to_noise`` docstring (lines 220-244). Provenance: Shao &
Colavita (1992) PASP for shot-noise piston variance; Conan et al.
(1995) for the servo-lag term.

The stage routes the planet/leakage rates through the detector's
``state.results.components['detector.<x>']['port_rates']`` array — the
same single-source-of-truth pattern as KILO's
``detected_stellar_rate_at_telescope`` consolidation (KILO commit
``47607de``). nullsim solves the problem at the architecture level: the
rate-at-detector is computed exactly once by the detector stage, and
the performance stage consumes the result without recomputing it.

**opd_budget is consumed optionally.** Space / formation-flying pipelines
(e.g. HWO) strip atmosphere, AO, and fringe-tracking stages, so no
upstream stage populates ``state.opd_budget``. When opd_budget is absent
(empty components) or has no ``fringe_tracking.closed_loop_residual``
entry, ``_resolve_servo_bw`` still falls back to
``Params.default_servo_bw_hz`` and ``with_fringe_tracking`` remains as
configured in Params. The result component records
``servo_bw_source="params.default_servo_bw_hz (no opd_budget)"`` so the
provenance is auditable. Configs that include a fringe-tracking stage
continue to use the FT-aware systematic term unchanged.

See ``tests/stages/notes/performance.md`` for the full source survey,
KILO/NAYRA divergence notes, and edge-case handling.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField

from nullsim.pipeline import (
    BaseStage,
    SimulationState,
    register_stage,
)
from nullsim.stages._utils import unique_component_name

__all__ = ["StandardPerformance"]


_DETECTOR_PREFIX = "detector."
_CALIBRATION_PREFIX = "calibration."
_BACKGROUND_PREFIX = "background."
_FRINGE_TRACKING_RESIDUAL_PREFIX = "fringe_tracking.closed_loop_residual"


@register_stage
class StandardPerformance(BaseStage):
    """Compute SNR, integration time, and a contrast curve from upstream state."""

    type_name = "performance.standard"
    family = "performance"
    # opd_budget is read opportunistically by ``_resolve_servo_bw`` for the
    # legacy "FT residual present?" audit signal. Modern configs read
    # servo_bandwidth_hz from the FT results component (already covered by
    # ``results`` in consumes), so opd_budget is treated as a cache-key
    # extra rather than a hard dependency — HWO and other no-AO/no-FT
    # pipelines have no opd_budget producer and must still run. Dropping
    # ``photon_rates`` from consumes: this stage reads per-port rates from
    # the detector results component, not from state.photon_rates.
    consumes = frozenset({"results", "scene"})
    produces = frozenset({"results"})
    cache_key_extras = frozenset({"opd_budget"})

    class Params(BaseModel):
        """Validated params for the standard performance stage."""

        model_config = ConfigDict(
            extra="forbid",
            frozen=True,
            allow_inf_nan=False,
        )

        # Fractional instability of residual stellar speckle/leakage counts.
        # This is a scalar first-pass analog of EXOSIMS/Nemati C_sp-style
        # speckle residuals: it increases the systematic variance term but
        # does not add another mean leakage floor.
        residual_speckle_stability_factor: float = PydanticField(
            default=0.0,
            ge=0.0,
            le=1.0,
        )
        # Photon-resolved null self-calibration gain applied only to the
        # instability/systematic term. Mean-leakage photon noise is untouched.
        nsc_systematic_reduction: float = PydanticField(default=1.0, ge=1.0)
        target_snr: float = PydanticField(default=5.0, gt=0.0)
        # Both rates are **per port** (KILO ``n_*`` convention). Totals
        # across the n_dark_ports detector layout are n_*_hz * n_dark_ports
        # * T. Configure as if you're setting one detector pixel.
        n_sky_background_hz: float = PydanticField(default=0.0, ge=0.0)
        n_dark_count_hz: float = PydanticField(default=0.0, ge=0.0)
        compute_contrast_curve: bool = True
        contrast_curve_separations_mas: tuple[float, ...] = ()
        contrast_curve_reference_time_s: float | None = PydanticField(
            default=None,
            gt=0.0,
        )
        with_fringe_tracking: bool = True
        default_servo_bw_hz: float = PydanticField(default=2000.0, gt=0.0)
        planet_dark_throughput: float = PydanticField(default=0.5, gt=0.0, le=1.0)

    def apply(self, state: SimulationState) -> SimulationState:
        params = self.params_model

        detector_name, detector_component = _find_detector_component(state)
        port_rates_hz = np.asarray(
            detector_component["port_rates"], dtype=np.float64
        )
        port_labels = tuple(detector_component["port_labels"])
        integration_time_s = float(detector_component["integration_time_s"])

        (
            bright_rate_hz,
            leakage_rate_hz_total,
            n_dark_ports,
            n_bright_ports,
        ) = _partition_port_rates(port_rates_hz, port_labels)
        spectro_rate_hz_total = _sum_port_rates_by_prefix(
            port_rates_hz,
            port_labels,
            "spectro",
        )
        total_stellar_rate_hz = (
            bright_rate_hz + leakage_rate_hz_total + spectro_rate_hz_total
        )

        null_depth_residual, null_source = _resolve_null_depth_residual(
            state, detector_component
        )
        calibration_floor, calibration_floor_source = _resolve_calibration_floor(state)
        calibration_floor_applied = False

        # If a sensitivity stage has run, use its effective null (which
        # adds the resolved-disk floor to the AO-realized null) and its
        # per-separation planet throughput. The detector/calibration
        # residual still bounds the effective null: a nonideal upstream
        # field perturbation seen by the detector cannot be hidden by an
        # idealized sensitivity calculation.
        sensitivity = _find_sensitivity_component(state)
        null_depth_residual_effective = _kilo_null_to_dark_power_fraction(
            null_depth_residual,
            bright_rate_hz=bright_rate_hz,
            total_stellar_rate_hz=total_stellar_rate_hz,
            n_dark_ports=n_dark_ports,
            n_bright_ports=n_bright_ports,
        )
        calibration_floor_effective = (
            None
            if calibration_floor is None
            else _kilo_null_to_dark_power_fraction(
                calibration_floor,
                bright_rate_hz=bright_rate_hz,
                total_stellar_rate_hz=total_stellar_rate_hz,
                n_dark_ports=n_dark_ports,
                n_bright_ports=n_bright_ports,
            )
        )
        if sensitivity is not None:
            sens_name, sens_payload = sensitivity
            sensitivity_null_with_disk = float(sens_payload["effective_null_with_disk"])
            if (
                not math.isfinite(sensitivity_null_with_disk)
                or sensitivity_null_with_disk < 0.0
            ):
                # A NaN scalar fell through _max_finite_nonnegative to 0.0
                # pre-fix — a corrupted null silently became a PERFECT null
                # (review 2026-06-10 Issue 27b). Mirror the by-separation
                # curve's strict validation.
                raise ValueError(
                    "performance.standard: sensitivity component "
                    f"{sens_name!r} published a non-finite or negative "
                    f"effective_null_with_disk ({sensitivity_null_with_disk!r})."
                )
            null_depth_effective = _max_finite_nonnegative(
                sensitivity_null_with_disk,
                null_depth_residual_effective,
            )
            if (
                calibration_floor_effective is not None
                and calibration_floor_effective > null_depth_effective
            ):
                warnings.warn(
                    "performance.standard applied calibration_floor because it "
                    "exceeds sensitivity.effective_null_with_disk "
                    f"({calibration_floor_effective:.6g} > "
                    f"{null_depth_effective:.6g}).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                null_depth_effective = calibration_floor_effective
                null_source = f"calibration_floor:{calibration_floor_source}"
                calibration_floor_applied = True
            else:
                if (
                    null_depth_residual_effective is not None
                    and null_depth_residual_effective > sensitivity_null_with_disk
                ):
                    null_source = f"{null_source} bounds sensitivity:{sens_name}"
                else:
                    null_source = f"sensitivity:{sens_name}"
            # With the effective null we recompute the dark leakage so it
            # carries the disk floor and uses every stellar output port in
            # the normalization.
            leakage_rate_hz = null_depth_effective * total_stellar_rate_hz
            null_depth_used = null_depth_effective
            sensitivity_throughputs = tuple(
                float(v) for v in np.asarray(sens_payload["planet_throughputs"])
            )
            sensitivity_separations_mas = tuple(
                float(s) for s in sens_payload["separations_mas"]
            )
            sensitivity_null_by_separation = (
                _resolve_sensitivity_effective_null_by_separation(
                    sens_payload=sens_payload,
                    sensitivity_separations_mas=sensitivity_separations_mas,
                )
            )
            instab_eps = float(sens_payload.get("null_instability", 0.0))
            null_instability_used = instab_eps
            eps_used = _combined_systematic_eps(
                null_instability=instab_eps / float(params.nsc_systematic_reduction),
                residual_speckle_stability_factor=(
                    params.residual_speckle_stability_factor
                ),
            )
            # Optional sensitivity-stage f_eff override. When the
            # sensitivity stage sets
            # ``effective_systematic_bandwidth_hz_override``, it forwards
            # that value here and the contrast_curve uses
            # eps² × leak² × t / (2 f_eff) with f_eff = override. Default
            # ``_resolve_servo_bw`` (FT residual or
            # ``Params.default_servo_bw_hz``) is preserved when the
            # override is absent.
            sens_bandwidth_hz = sens_payload.get(
                "effective_systematic_bandwidth_hz", None,
            )
        else:
            sens_name = None
            null_depth_effective = null_depth_residual
            null_depth_effective_fraction = null_depth_residual_effective
            if (
                calibration_floor is not None
                and calibration_floor > null_depth_effective
            ):
                warnings.warn(
                    "performance.standard applied calibration_floor because it "
                    "exceeds detector/calibration null depth "
                    f"({calibration_floor:.6g} > {null_depth_effective:.6g}).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                null_depth_effective = calibration_floor
                null_depth_effective_fraction = calibration_floor_effective
                null_source = f"calibration_floor:{calibration_floor_source}"
                calibration_floor_applied = True
            if null_source == "detector.null_depth_raw":
                leakage_rate_hz = leakage_rate_hz_total
            elif null_depth_effective_fraction is not None:
                # Convert the KILO mean(dark)/mean(bright) convention to a
                # dark/total fraction BEFORE multiplying by the total
                # stellar rate. The unconverted product underestimated the
                # dark leakage by n_dark x mean_bright/total (review
                # 2026-06-10 Issue 27a).
                leakage_rate_hz = (
                    null_depth_effective_fraction * total_stellar_rate_hz
                )
            else:
                leakage_rate_hz = null_depth_effective * total_stellar_rate_hz
            null_depth_used = null_depth_effective
            sensitivity_throughputs = None
            sensitivity_separations_mas = None
            sensitivity_null_by_separation = None
            null_instability_used = 0.0
            eps_used = _combined_systematic_eps(
                null_instability=0.0,
                residual_speckle_stability_factor=(
                    params.residual_speckle_stability_factor
                ),
            )
            sens_bandwidth_hz = None

        # KILO convention (kilo/performance.py L204-292 signal_to_noise):
        # signal and stellar-leakage rates are **aggregate over all dark
        # ports** -- KILO sums photons across dark ports before the
        # Poisson noise sum, and adds dark/sky as per-port × n_dark to
        # match. Pre-fix nullsim divided planet_rate_hz and leakage_rate_hz
        # by n_dark_ports while leaving _compute_snr's D/Sky terms as
        # per-port × n_dark, so SNR was wrong by √n_dark_ports for n_dark>1
        # (Code Review 2026-05-16 Issue 2).
        planet_contrast = _resolve_planet_contrast(state)
        (
            planet_dark_throughput_used,
            planet_dark_throughput_source,
            planet_separation_mas,
        ) = _resolve_planet_dark_throughput_for_scene(
            state=state,
            params=params,
            sensitivity_name=sens_name,
            sensitivity_separations_mas=sensitivity_separations_mas,
            sensitivity_throughputs=sensitivity_throughputs,
        )
        # Sensitivity-stage throughputs are normalized to total coupled
        # stellar power. The legacy flat planet_dark_throughput parameter is
        # retained as a bright-port routing scalar for configs without the
        # sensitivity handoff.
        planet_rate_base_hz = (
            total_stellar_rate_hz
            if sensitivity_throughputs is not None
            else bright_rate_hz
        )
        planet_rate_hz = (
            planet_rate_base_hz
            * planet_contrast
            * planet_dark_throughput_used
        )

        servo_bw_hz, servo_bw_source = _resolve_servo_bw(state, params)
        if sens_bandwidth_hz is not None:
            servo_bw_hz = float(sens_bandwidth_hz)
            servo_bw_source = (
                f"sensitivity:{sens_name}.effective_systematic_bandwidth_hz"
            )
        sky_rate_hz_per_port, sky_rate_source = _resolve_background_rate_hz_per_port(
            state=state,
            configured_rate_hz_per_port=params.n_sky_background_hz,
            port_kind="dark",
        )
        dark_rate_hz_per_port, dark_rate_source = _resolve_dark_rate_hz_per_port(
            detector_component,
            params.n_dark_count_hz,
        )

        snr_breakdown = _compute_snr(
            planet_rate_hz=planet_rate_hz,
            leakage_rate_hz=leakage_rate_hz,
            dark_rate_hz_per_port=dark_rate_hz_per_port,
            sky_rate_hz_per_port=sky_rate_hz_per_port,
            n_dark_ports=n_dark_ports,
            integration_time_s=integration_time_s,
            eps=eps_used,
            with_fringe_tracking=params.with_fringe_tracking,
            servo_bw_hz=servo_bw_hz,
        )

        integration_time_for_target_snr_s = _integration_time_for_snr(
            target_snr=params.target_snr,
            planet_rate_hz=planet_rate_hz,
            leakage_rate_hz=leakage_rate_hz,
            dark_rate_hz_per_port=dark_rate_hz_per_port,
            sky_rate_hz_per_port=sky_rate_hz_per_port,
            n_dark_ports=n_dark_ports,
            eps=eps_used,
            with_fringe_tracking=params.with_fringe_tracking,
            servo_bw_hz=servo_bw_hz,
        )

        if params.compute_contrast_curve:
            contrast_curve_time_s, contrast_curve_time_source = (
                _contrast_curve_reference_time_s(
                    configured_time_s=params.contrast_curve_reference_time_s,
                    detector_integration_time_s=integration_time_s,
                )
            )
            # When the sensitivity stage has supplied a per-separation
            # throughput curve, use its separations grid (and the matched
            # throughputs) directly. Otherwise, fall back to the Param
            # separations + flat planet_dark_throughput.
            if sensitivity_separations_mas is not None:
                separations_mas = sensitivity_separations_mas
                planet_throughput_per_sep = sensitivity_throughputs
            else:
                separations_mas = tuple(
                    float(s) for s in params.contrast_curve_separations_mas
                )
                planet_throughput_per_sep = None
            # KILO ``contrast_curve`` (kilo/performance.py L968-1008) uses the
            # *total* post-throughput stellar photon rate in the denominator
            # of ``c = signal / (stellar × throughput × t)``. Pre-Phase-1.2
            # nullsim used the bright-port rate alone, which for the Keck
            # 5 bright / 24 total split is ~4.8× tighter than KILO's
            # convention. We pass both rates into ``_contrast_curve``:
            # ``bright_rate_hz`` as a positivity guard for legacy callers and
            # ``stellar_rate_hz`` for the total-star contrast denominator.
            (
                contrast_sky_rate_hz_per_port,
                contrast_sky_rate_source,
            ) = _resolve_background_curve_rate_hz_per_port(
                state=state,
                configured_rate_hz_per_port=params.n_sky_background_hz,
                separations_mas=separations_mas,
                port_kind="dark",
            )
            contrast_leakage_rate_hz_per_sep = (
                _contrast_curve_leakage_rates_from_null_curve(
                    sensitivity_null_by_separation=sensitivity_null_by_separation,
                    total_stellar_rate_hz=total_stellar_rate_hz,
                    null_depth_residual=null_depth_residual_effective,
                    calibration_floor=calibration_floor_effective,
                )
            )
            contrast_value = _contrast_curve(
                separations_mas=separations_mas,
                bright_rate_hz=bright_rate_hz,
                stellar_rate_hz=total_stellar_rate_hz,
                leakage_rate_hz=leakage_rate_hz,
                leakage_rate_hz_per_sep=contrast_leakage_rate_hz_per_sep,
                dark_rate_hz_per_port=dark_rate_hz_per_port,
                sky_rate_hz_per_port=sky_rate_hz_per_port,
                sky_rate_hz_per_port_per_sep=contrast_sky_rate_hz_per_port,
                n_dark_ports=n_dark_ports,
                integration_time_s=contrast_curve_time_s,
                eps=eps_used,
                with_fringe_tracking=params.with_fringe_tracking,
                servo_bw_hz=servo_bw_hz,
                target_snr=params.target_snr,
                planet_throughput_per_sep=planet_throughput_per_sep,
                planet_dark_throughput=params.planet_dark_throughput,
            )
        else:
            separations_mas = ()
            contrast_value = ()
            contrast_curve_time_s, contrast_curve_time_source = (
                _contrast_curve_reference_time_s(
                    configured_time_s=params.contrast_curve_reference_time_s,
                    detector_integration_time_s=integration_time_s,
                )
            )
            contrast_sky_rate_hz_per_port = None
            contrast_sky_rate_source = None
            contrast_leakage_rate_hz_per_sep = None

        component_name = unique_component_name(
            self.type_name,
            state.results.components,
            self.context.instance_id,
        )
        component_payload: dict[str, Any] = {
            "snr": float(snr_breakdown["snr"]),
            "integration_time_for_target_snr_s": float(integration_time_for_target_snr_s),
            "target_snr": float(params.target_snr),
            "contrast_curve_mas": separations_mas,
            "contrast_curve_value": contrast_value,
            "S": float(snr_breakdown["S"]),
            "B": float(snr_breakdown["B"]),
            "D": float(snr_breakdown["D"]),
            "Sky": float(snr_breakdown["Sky"]),
            "systematic_var": float(snr_breakdown["systematic_var"]),
            "null_depth_residual": float(null_depth_residual),
            "null_depth_residual_effective_fraction": (
                None
                if null_depth_residual_effective is None
                else float(null_depth_residual_effective)
            ),
            # Backward-compatible alias for existing plot/report consumers.
            "null_depth_calibrated": float(null_depth_residual),
            "null_depth_effective": float(null_depth_effective),
            "planet_contrast": float(planet_contrast),
            "planet_rate_hz": float(planet_rate_hz),
            "planet_rate_base_hz": float(planet_rate_base_hz),
            "planet_dark_throughput_used": float(planet_dark_throughput_used),
            "planet_dark_throughput_source": planet_dark_throughput_source,
            "planet_separation_mas": planet_separation_mas,
            "bright_rate_hz": float(bright_rate_hz),
            "leakage_rate_hz": float(leakage_rate_hz),
            "leakage_rate_hz_total": float(leakage_rate_hz_total),
            "spectro_rate_hz_total": float(spectro_rate_hz_total),
            "stellar_rate_hz_total": float(total_stellar_rate_hz),
            "sky_rate_hz_per_port": float(sky_rate_hz_per_port),
            "sky_rate_source": sky_rate_source,
            "contrast_curve_sky_rate_source": contrast_sky_rate_source,
            "dark_rate_hz_per_port": float(dark_rate_hz_per_port),
            "dark_rate_source": dark_rate_source,
            "n_dark_ports": int(n_dark_ports),
            "integration_time_s": float(integration_time_s),
            "contrast_curve_reference_time_s": float(contrast_curve_time_s),
            "contrast_curve_reference_time_source": contrast_curve_time_source,
            "residual_speckle_stability_factor": float(
                params.residual_speckle_stability_factor
            ),
            "null_instability": float(null_instability_used),
            "eps_systematic_used": float(eps_used),
            "nsc_systematic_reduction": float(params.nsc_systematic_reduction),
            "calibration_floor": (
                None if calibration_floor is None else float(calibration_floor)
            ),
            "calibration_floor_source": calibration_floor_source,
            "calibration_floor_applied": bool(calibration_floor_applied),
            "with_fringe_tracking": bool(params.with_fringe_tracking),
            "servo_bw_hz": float(servo_bw_hz),
            "servo_bw_source": servo_bw_source,
            "upstream_detector_component": detector_name,
            "upstream_sensitivity_component": sens_name,
            "null_depth_source": null_source,
            "source_model": (
                "KILO performance.py signal_to_noise (lines 204-292), "
                "integration_time_for_snr (lines 802-894), and contrast_curve "
                "(lines 896-1008). Calibration consolidation: KILO commit "
                "47607de (detected_stellar_rate_at_telescope, lines 62-161)."
            ),
        }
        if separations_mas and sensitivity_throughputs is None:
            component_payload["contrast_curve_todo"] = (
                "Angular-throughput scaling uses Param "
                "planet_dark_throughput as a flat value; add a "
                "sensitivity.detection_curve stage upstream for a per-"
                "separation curve."
            )
        if contrast_sky_rate_hz_per_port is not None:
            component_payload["contrast_curve_sky_rate_hz_per_port"] = np.asarray(
                contrast_sky_rate_hz_per_port,
                dtype=np.float64,
            )
        if contrast_leakage_rate_hz_per_sep is not None:
            component_payload["contrast_curve_leakage_rate_hz"] = np.asarray(
                contrast_leakage_rate_hz_per_sep,
                dtype=np.float64,
            )

        results = state.results.add_component(component_name, component_payload)
        if separations_mas and contrast_value:
            # Build a structured array matching the SPEC §6.2 contract so
            # detection_space and other consumers can index ["separation_mas"]
            # and ["contrast"] uniformly.
            _cc_dtype = np.dtype(
                [("separation_mas", np.float64), ("contrast", np.float64)]
            )
            contrast_curve_arr: NDArray[Any] | None = np.array(
                list(zip(separations_mas, contrast_value)),
                dtype=_cc_dtype,
            )
        else:
            contrast_curve_arr = None
        results = replace(
            results,
            null_depth=float(null_depth_used),
            snr=float(snr_breakdown["snr"]),
            contrast_curve=contrast_curve_arr,
        )
        return replace(state, results=results)


_SENSITIVITY_PREFIX = "sensitivity.detection_curve"


def _find_sensitivity_component(
    state: SimulationState,
) -> tuple[str, dict[str, Any]] | None:
    """Return (name, payload) of an upstream sensitivity.detection_curve, if any."""

    for name, payload in state.results.components.items():
        if (
            name.startswith(_SENSITIVITY_PREFIX)
            and isinstance(payload, dict)
            and "planet_throughputs" in payload
            and "separations_mas" in payload
        ):
            return name, payload
    return None


def _find_detector_component(state: SimulationState) -> tuple[str, Any]:
    """Return the first ``detector.*`` component with port_rates + port_labels."""

    for name, value in state.results.components.items():
        if not name.startswith(_DETECTOR_PREFIX):
            continue
        if not isinstance(value, dict):
            continue
        if {"port_rates", "port_labels", "integration_time_s"}.issubset(value.keys()):
            return name, value
    raise ValueError(
        "performance.standard requires an upstream detector stage "
        "(detector.mkid / detector.snspd / detector.ideal_counter) to "
        "have written a 'detector.*' component with 'port_rates', "
        "'port_labels', and 'integration_time_s' fields."
    )


def _contrast_curve_reference_time_s(
    *,
    configured_time_s: float | None,
    detector_integration_time_s: float,
) -> tuple[float, str]:
    """Resolve the integration time attached to stored contrast curves."""

    if configured_time_s is not None:
        return float(configured_time_s), "performance.contrast_curve_reference_time_s"
    return float(detector_integration_time_s), "detector.integration_time_s"


def _partition_port_rates(
    port_rates_hz: np.ndarray, port_labels: tuple[str, ...]
) -> tuple[float, float, int, int]:
    """Partition detector port rates into bright/dark totals and counts.

    Detectors label ports ``bright_*`` / ``dark_*`` (see
    ``nullsim.stages.detector.mkid._port_labels``). Returns
    ``(bright_total_hz, dark_total_hz, n_dark_ports, n_bright_ports)``.

    Both totals are summed across all ports of their kind. KILO's
    ``kilo/performance.py`` ``signal_to_noise`` uses **per-total**
    semantics: signal and stellar-leakage rates flow into the Poisson
    noise sum as aggregates across dark ports, while detector dark
    counts and sky are multiplied by ``n_dark_ports`` from their
    per-port input rates. ``StandardPerformance.apply`` passes these
    totals through unchanged to ``_compute_snr`` /
    ``_integration_time_for_snr`` / ``_contrast_curve``.
    """

    if port_rates_hz.shape != (len(port_labels),):
        raise ValueError(
            f"performance.standard: port_rates shape {port_rates_hz.shape} "
            f"does not match {len(port_labels)} port labels."
        )
    bright_mask = np.array(
        [label.startswith("bright") for label in port_labels], dtype=bool
    )
    dark_mask = np.array(
        [label.startswith("dark") for label in port_labels], dtype=bool
    )
    if not bright_mask.any():
        raise ValueError(
            "performance.standard: detector reported no bright ports; "
            "cannot compute stellar leakage or planet signal."
        )
    if not dark_mask.any():
        raise ValueError(
            "performance.standard: detector reported no dark ports; "
            "cannot compute planet detection SNR."
        )
    bright_rate = float(port_rates_hz[bright_mask].sum())
    dark_rate = float(port_rates_hz[dark_mask].sum())
    n_dark_ports = int(dark_mask.sum())
    n_bright_ports = int(bright_mask.sum())
    return bright_rate, dark_rate, n_dark_ports, n_bright_ports


def _sum_port_rates_by_prefix(
    port_rates_hz: np.ndarray,
    port_labels: tuple[str, ...],
    prefix: str,
) -> float:
    mask = np.array(
        [label.startswith(f"{prefix}_") for label in port_labels],
        dtype=bool,
    )
    if not mask.any():
        return 0.0
    return float(port_rates_hz[mask].sum())


def _resolve_null_depth_residual(
    state: SimulationState, detector_component: dict[str, Any]
) -> tuple[float, str]:
    """Prefer the calibration stage's explicit residual, fall back to raw."""

    for name, value in state.results.components.items():
        if not name.startswith(_CALIBRATION_PREFIX):
            continue
        if not isinstance(value, dict):
            continue
        if "null_depth_residual" in value:
            return float(value["null_depth_residual"]), name
        if "null_depth_calibrated" in value:
            return float(value["null_depth_calibrated"]), name
    return float(detector_component["null_depth_raw"]), "detector.null_depth_raw"


def _kilo_null_to_dark_power_fraction(
    null_depth_kilo: float | None,
    *,
    bright_rate_hz: float,
    total_stellar_rate_hz: float,
    n_dark_ports: int,
    n_bright_ports: int,
) -> float | None:
    """Convert mean(dark)/mean(bright) null depth to dark/total fraction."""

    if null_depth_kilo is None:
        return None
    value = float(null_depth_kilo)
    if not math.isfinite(value) or value < 0.0:
        return None
    if (
        bright_rate_hz <= 0.0
        or total_stellar_rate_hz <= 0.0
        or n_dark_ports <= 0
        or n_bright_ports <= 0
    ):
        return None
    mean_bright_rate_hz = float(bright_rate_hz) / float(n_bright_ports)
    equivalent_dark_rate_hz = value * mean_bright_rate_hz * float(n_dark_ports)
    return float(equivalent_dark_rate_hz / float(total_stellar_rate_hz))


def _max_finite_nonnegative(*values: float | None) -> float:
    """Return the maximum finite non-negative value, ignoring missing bounds."""

    finite_values = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value)) and float(value) >= 0.0
    ]
    if not finite_values:
        return 0.0
    return max(finite_values)


def _resolve_calibration_floor(
    state: SimulationState,
) -> tuple[float | None, str | None]:
    """Return the first explicit calibration floor from an upstream calibration stage."""

    for name, value in state.results.components.items():
        if not name.startswith(_CALIBRATION_PREFIX):
            continue
        if not isinstance(value, dict):
            continue
        if "calibration_floor" not in value:
            continue
        floor = float(value["calibration_floor"])
        if not math.isfinite(floor) or floor < 0.0:
            raise ValueError(
                "performance.standard: calibration_floor must be finite and "
                f"non-negative (got {floor!r} from {name})."
            )
        return floor, name
    return None, None


def _resolve_planet_contrast(state: SimulationState) -> float:
    """Read planet contrast from the scene."""

    contrast = state.scene.planet_contrast
    if contrast is None:
        raise ValueError(
            "performance.standard requires scene.planet_contrast for SNR "
            "and integration-time outputs."
        )
    return float(contrast)


def _resolve_planet_dark_throughput_for_scene(
    *,
    state: SimulationState,
    params: StandardPerformance.Params,
    sensitivity_name: str | None,
    sensitivity_separations_mas: tuple[float, ...] | None,
    sensitivity_throughputs: tuple[float, ...] | None,
) -> tuple[float, str, float | None]:
    """Return the planet throughput at the configured planet separation."""

    fallback = float(params.planet_dark_throughput)
    if sensitivity_separations_mas is None or sensitivity_throughputs is None:
        return fallback, "params.planet_dark_throughput", None

    planet_separation_mas = state.scene.planet_separation_mas
    if planet_separation_mas is None:
        return (
            fallback,
            "params.planet_dark_throughput (scene.planet_separation_mas missing)",
            None,
        )

    separations = np.asarray(sensitivity_separations_mas, dtype=np.float64)
    throughputs = np.asarray(sensitivity_throughputs, dtype=np.float64)
    if separations.shape != throughputs.shape:
        raise ValueError(
            "performance.standard: sensitivity separations and throughputs "
            f"must have matching shape (got {separations.shape} and "
            f"{throughputs.shape})."
        )
    if separations.size == 0:
        return fallback, "params.planet_dark_throughput (empty sensitivity grid)", None
    if not np.all(np.isfinite(separations)) or not np.all(np.isfinite(throughputs)):
        raise ValueError(
            "performance.standard: sensitivity separations and throughputs "
            "must be finite."
        )
    order = np.argsort(separations)
    sorted_separations = separations[order]
    sorted_throughputs = throughputs[order]
    throughput = float(
        np.interp(
            float(planet_separation_mas),
            sorted_separations,
            sorted_throughputs,
        )
    )
    if throughput < 0.0:
        raise ValueError(
            "performance.standard: interpolated planet throughput must be "
            f"non-negative (got {throughput!r})."
        )
    source_name = sensitivity_name or "sensitivity.detection_curve"
    return (
        throughput,
        f"sensitivity:{source_name}.planet_throughputs",
        float(planet_separation_mas),
    )


def _resolve_sensitivity_effective_null_by_separation(
    *,
    sens_payload: dict[str, Any],
    sensitivity_separations_mas: tuple[float, ...],
) -> tuple[float, ...] | None:
    """Return optional per-separation effective null from sensitivity."""

    if "effective_null_with_disk_by_separation" not in sens_payload:
        return None
    effective_null = np.asarray(
        sens_payload["effective_null_with_disk_by_separation"],
        dtype=np.float64,
    )
    separations = np.asarray(sensitivity_separations_mas, dtype=np.float64)
    if effective_null.shape != separations.shape:
        raise ValueError(
            "performance.standard: sensitivity effective_null_with_disk_by_separation "
            "and separations_mas must have matching shape "
            f"(got {effective_null.shape} and {separations.shape})."
        )
    if not np.all(np.isfinite(effective_null)) or np.any(effective_null < 0.0):
        raise ValueError(
            "performance.standard: sensitivity effective_null_with_disk_by_separation "
            "entries must be finite and non-negative."
        )
    return tuple(float(v) for v in effective_null)


def _contrast_curve_leakage_rates_from_null_curve(
    *,
    sensitivity_null_by_separation: tuple[float, ...] | None,
    total_stellar_rate_hz: float,
    null_depth_residual: float | None,
    calibration_floor: float | None,
) -> tuple[float, ...] | None:
    """Convert an optional per-separation null curve to leakage rates."""

    if sensitivity_null_by_separation is None:
        return None
    effective_null = np.asarray(sensitivity_null_by_separation, dtype=np.float64)
    if null_depth_residual is not None and math.isfinite(float(null_depth_residual)):
        effective_null = np.maximum(effective_null, float(null_depth_residual))
    if calibration_floor is not None and math.isfinite(float(calibration_floor)):
        effective_null = np.maximum(effective_null, float(calibration_floor))
    return tuple(float(v * total_stellar_rate_hz) for v in effective_null)


def _resolve_background_rate_hz_per_port(
    *,
    state: SimulationState,
    configured_rate_hz_per_port: float,
    port_kind: str = "dark",
) -> tuple[float, str]:
    """Resolve per-port sky/background rate from params or a background stage.

    Explicit positive params win. A configured zero keeps legacy behavior
    unless an upstream ``background.*`` component is present, in which case
    the staged background product is used. This lets configs opt into the
    new background calculator by inserting the stage without changing older
    zero-background tests.
    """

    configured = float(configured_rate_hz_per_port)
    if configured > 0.0:
        return configured, "params.n_sky_background_hz"

    preferred_key = f"background_rate_hz_per_{port_kind}_port"
    fallback_key = f"sky_rate_hz_per_{port_kind}_port"
    for name, payload in reversed(tuple(state.results.components.items())):
        if not name.startswith(_BACKGROUND_PREFIX) or not isinstance(payload, dict):
            continue
        value = payload.get(preferred_key, payload.get(fallback_key))
        if value is None and port_kind != "dark":
            value = payload.get(
                "background_rate_hz_per_dark_port",
                payload.get("sky_rate_hz_per_dark_port"),
            )
        if value is None:
            continue
        rate = float(value)
        if rate < 0.0 or not math.isfinite(rate):
            raise ValueError(
                f"performance.standard: background component {name!r} "
                f"reported invalid {preferred_key}={value!r}."
            )
        return rate, name

    return configured, "params.n_sky_background_hz"


def _resolve_background_curve_rate_hz_per_port(
    *,
    state: SimulationState,
    configured_rate_hz_per_port: float,
    separations_mas: tuple[float, ...],
    port_kind: str = "dark",
) -> tuple[tuple[float, ...] | None, str | None]:
    """Resolve a per-separation background vector for contrast curves."""

    if not separations_mas:
        return None, None

    configured = float(configured_rate_hz_per_port)
    if configured > 0.0:
        return (
            tuple(configured for _ in separations_mas),
            "params.n_sky_background_hz",
        )

    preferred_key = f"background_rate_hz_per_{port_kind}_port_per_sep"
    fallback_key = f"sky_rate_hz_per_{port_kind}_port_per_sep"
    vector_keys = (preferred_key, fallback_key)
    if port_kind != "dark":
        vector_keys = (
            preferred_key,
            fallback_key,
            "background_rate_hz_per_dark_port_per_sep",
            "sky_rate_hz_per_dark_port_per_sep",
        )

    target_separations = np.asarray(separations_mas, dtype=np.float64)
    for name, payload in reversed(tuple(state.results.components.items())):
        if not name.startswith(_BACKGROUND_PREFIX) or not isinstance(payload, dict):
            continue
        vector_value = None
        vector_key = None
        for key in vector_keys:
            if key in payload:
                vector_value = payload[key]
                vector_key = key
                break
        if vector_value is None or vector_key is None:
            continue

        source_separations = payload.get(
            "background_separations_mas",
            payload.get("separations_mas"),
        )
        if source_separations is None:
            raise ValueError(
                f"performance.standard: background component {name!r} "
                f"reported {vector_key!r} without background_separations_mas."
            )
        source_sep_arr = np.asarray(source_separations, dtype=np.float64)
        if source_sep_arr.shape != target_separations.shape:
            raise ValueError(
                f"performance.standard: background component {name!r} "
                "per-separation background shape does not match the contrast "
                f"curve grid ({source_sep_arr.shape} != {target_separations.shape})."
            )
        if not np.allclose(source_sep_arr, target_separations, rtol=0.0, atol=1.0e-9):
            raise ValueError(
                f"performance.standard: background component {name!r} "
                "per-separation background grid is not aligned with the "
                "contrast curve separations."
            )

        rates = np.asarray(vector_value, dtype=np.float64)
        if rates.shape != target_separations.shape:
            raise ValueError(
                f"performance.standard: background component {name!r} "
                f"reported {vector_key!r} shape {rates.shape}, expected "
                f"{target_separations.shape}."
            )
        if not np.all(np.isfinite(rates)) or np.any(rates < 0.0):
            raise ValueError(
                f"performance.standard: background component {name!r} "
                f"reported invalid {vector_key!r}; rates must be finite "
                "and non-negative."
            )
        return tuple(float(v) for v in rates), f"{name}.{vector_key}"

    return None, None


def _combined_systematic_eps(
    *,
    null_instability: float,
    residual_speckle_stability_factor: float,
) -> float:
    """Quadrature of the physically derived systematic terms.

    The hand-dialed calibration residual knob was removed 2026-06-09: it had
    no physical basis and, in fringe-tracking mode, was silently averaged down
    as sqrt(t) anyway. The MC null-instability CV and the HWO
    speckle-stability factor are the remaining physical terms.
    """

    return math.hypot(float(null_instability), float(residual_speckle_stability_factor))


def _resolve_dark_rate_hz_per_port(
    detector_component: dict[str, Any],
    configured_rate_hz_per_port: float,
) -> tuple[float, str]:
    configured = float(configured_rate_hz_per_port)
    if configured > 0.0:
        return configured, "params.n_dark_count_hz"

    for key in ("dark_rate_hz_per_port", "snspd_dark_rate_hz"):
        value = detector_component.get(key)
        if value is None:
            continue
        rate = float(value)
        if rate < 0.0 or not math.isfinite(rate):
            raise ValueError(
                "performance.standard: detector component reported invalid "
                f"{key}={value!r}."
            )
        return rate, f"detector.{key}"

    return configured, "params.n_dark_count_hz"


def _resolve_servo_bw(
    state: SimulationState, params: StandardPerformance.Params
) -> tuple[float, str]:
    """Return the servo bandwidth used for the FT systematic term.

    Preference order:
      1. ``servo_bandwidth_hz`` published by an upstream
         ``fringe_tracking.closed_loop`` results component. The FT
         stage stores both the auto-resolved and pinned bandwidth
         in its component payload, so reading it here matches what
         the servo actually ran with — earlier revisions always
         returned ``params.default_servo_bw_hz`` regardless, silently
         discarding any tuning the FT stage did and biasing the
         systematic-noise term whenever a user retuned the FT
         bandwidth without remembering to mirror it on the
         performance stage.
      2. ``params.default_servo_bw_hz`` when no FT component is
         present (e.g. HWO and other no-AO pipelines, or when the
         only opd_budget contribution is atmosphere/AO with no
         downstream FT). The ``opd_budget`` check is preserved as a
         second-tier audit signal.
    """

    for name, payload in state.results.components.items():
        if not name.startswith("fringe_tracking.closed_loop"):
            continue
        bw = payload.get("servo_bandwidth_hz") if isinstance(payload, dict) else None
        if isinstance(bw, (int, float)) and bw > 0.0 and math.isfinite(float(bw)):
            return float(bw), f"fringe_tracking servo_bandwidth_hz from results[{name!r}]"

    if not state.opd_budget.components:
        return (
            params.default_servo_bw_hz,
            "params.default_servo_bw_hz (no opd_budget)",
        )

    for name in state.opd_budget.components:
        if name.startswith(_FRINGE_TRACKING_RESIDUAL_PREFIX):
            return (
                params.default_servo_bw_hz,
                f"params.default_servo_bw_hz (FT residual {name} present, no servo_bandwidth_hz published)",
            )
    return params.default_servo_bw_hz, "params.default_servo_bw_hz (no FT residual upstream)"


def _systematic_variance(
    *,
    leakage_rate_hz: float,
    integration_time_s: float,
    eps: float,
    with_fringe_tracking: bool,
    servo_bw_hz: float,
) -> float:
    """Return the systematic-noise variance term in counts^2.

    Both branches follow KILO ``signal_to_noise`` lines 280-289.
    ``leakage_rate_hz`` is the integrated dark-port leakage rate
    (counts/s across all dark ports), matching the KILO
    ``stellar_leakage_rate`` argument.
    """

    leakage_total_hz = leakage_rate_hz  # already summed across dark ports
    if with_fringe_tracking:
        return (
            eps * eps * leakage_total_hz * leakage_total_hz * integration_time_s
            / (2.0 * servo_bw_hz)
        )
    leakage_counts = leakage_total_hz * integration_time_s
    return (eps * leakage_counts) ** 2


def _compute_snr(
    *,
    planet_rate_hz: float,
    leakage_rate_hz: float,
    dark_rate_hz_per_port: float,
    sky_rate_hz_per_port: float,
    n_dark_ports: int,
    integration_time_s: float,
    eps: float,
    with_fringe_tracking: bool,
    servo_bw_hz: float,
) -> dict[str, float]:
    """Return the SNR and its photon/dark/sky/systematic contributions.

    Photons-noise-limited limit: ``SNR → sqrt(S)`` when ``S >> B + D +
    Sky + systematic_var``. Systematic-noise-limited limit: ``SNR →
    S / (eps * B)`` when ``with_fringe_tracking=False`` and
    ``systematic_var → ∞``.
    """

    S = planet_rate_hz * integration_time_s
    B = leakage_rate_hz * integration_time_s
    D = dark_rate_hz_per_port * n_dark_ports * integration_time_s
    Sky = sky_rate_hz_per_port * n_dark_ports * integration_time_s
    systematic_var = _systematic_variance(
        leakage_rate_hz=leakage_rate_hz,
        integration_time_s=integration_time_s,
        eps=eps,
        with_fringe_tracking=with_fringe_tracking,
        servo_bw_hz=servo_bw_hz,
    )
    noise_variance = S + B + D + Sky + systematic_var
    if noise_variance <= 0.0 or S <= 0.0:
        snr = 0.0
    else:
        snr = S / math.sqrt(noise_variance)
    return {
        "snr": snr,
        "S": S,
        "B": B,
        "D": D,
        "Sky": Sky,
        "systematic_var": systematic_var,
    }


def _integration_time_for_snr(
    *,
    target_snr: float,
    planet_rate_hz: float,
    leakage_rate_hz: float,
    dark_rate_hz_per_port: float,
    sky_rate_hz_per_port: float,
    n_dark_ports: int,
    eps: float,
    with_fringe_tracking: bool,
    servo_bw_hz: float,
) -> float:
    """Solve the SNR formula for the integration time at ``target_snr``.

    Mirrors KILO ``integration_time_for_snr`` (lines 802-894). Returns
    ``inf`` when ``planet_rate_hz <= 0`` or when the target SNR
    exceeds the hard ceiling of the no-FT regime.
    """

    if planet_rate_hz <= 0.0:
        return math.inf

    noise_rate = (
        planet_rate_hz
        + leakage_rate_hz
        + dark_rate_hz_per_port * n_dark_ports
        + sky_rate_hz_per_port * n_dark_ports
    )
    snr_sq = target_snr * target_snr
    planet_rate_sq = planet_rate_hz * planet_rate_hz

    if with_fringe_tracking:
        # systematic adds eps^2 * B^2 / (2 * f_servo) to the effective rate.
        effective_noise_rate = (
            noise_rate
            + eps * eps * leakage_rate_hz * leakage_rate_hz / (2.0 * servo_bw_hz)
        )
        return snr_sq * effective_noise_rate / planet_rate_sq

    denominator = planet_rate_sq - snr_sq * (eps * leakage_rate_hz) ** 2
    if denominator <= 0.0:
        return math.inf
    return snr_sq * noise_rate / denominator


def _contrast_curve(
    *,
    separations_mas: tuple[float, ...],
    bright_rate_hz: float,
    stellar_rate_hz: float,
    leakage_rate_hz: float,
    dark_rate_hz_per_port: float,
    sky_rate_hz_per_port: float,
    n_dark_ports: int,
    integration_time_s: float,
    eps: float,
    with_fringe_tracking: bool,
    servo_bw_hz: float,
    target_snr: float,
    planet_throughput_per_sep: tuple[float, ...] | None = None,
    planet_dark_throughput: float = 0.5,
    sky_rate_hz_per_port_per_sep: tuple[float, ...] | None = None,
    leakage_rate_hz_per_sep: tuple[float, ...] | None = None,
) -> tuple[float, ...]:
    """Return the minimum-detectable contrast at each separation.

    The contrast is defined relative to the **total** post-throughput
    stellar photon rate (KILO ``contrast_curve`` convention,
    ``kilo/performance.py:964`` and ``natural_design.py:1972-1975``),
    i.e. the sum of ``bright_rate_hz`` and ``leakage_rate_hz_total``
    rather than the bright-port rate alone. For a KILO Keck 6+18-port
    layout this is ~4× looser than the bright-only denominator
    (KILO_PARITY_PLAN §1.2). ``bright_rate_hz`` is currently used only
    as a positivity guard (``<= 0.0 => contrast = inf``); it does NOT
    enter the planet-leg signal_counts arithmetic. Earlier revisions
    used it for an explicit planet-leg conversion that was folded into
    ``planet_throughput_per_sep`` and removed. Kept in the signature so
    callers without a bright/total split can pass ``stellar_rate_hz``
    twice without surprises.

    When ``planet_throughput_per_sep`` is provided (one value per entry
    of ``separations_mas``), the per-separation planet routing
    efficiency is used directly -- this is the KILO ``contrast_curve``
    mode and the only path that reproduces detection_space.pdf.
    Otherwise a single flat ``planet_dark_throughput`` scalar is
    broadcast to every separation (legacy fallback).

    The quadratic-in-``S`` solution and the noise model mirror KILO
    ``contrast_curve`` lines 968-1008.
    """

    if not separations_mas:
        return ()

    if bright_rate_hz <= 0.0 or integration_time_s <= 0.0:
        return tuple(math.inf for _ in separations_mas)

    D = dark_rate_hz_per_port * n_dark_ports * integration_time_s
    snr_sq = target_snr * target_snr
    # Contrast = required planet counts / coupled stellar counts.
    # planet_throughput moves on the planet leg only (folds into
    # signal_counts via the planet-noise quadratic). Folding it into
    # stellar_counts as well was inflating every reported contrast by
    # ~1/planet_dark_throughput (Code Review 2026-05-16 Issue 1).
    stellar_counts = stellar_rate_hz * integration_time_s
    if stellar_counts <= 0.0:
        return tuple(math.inf for _ in separations_mas)

    if planet_throughput_per_sep is not None:
        if len(planet_throughput_per_sep) != len(separations_mas):
            raise ValueError(
                "_contrast_curve: planet_throughput_per_sep length "
                f"{len(planet_throughput_per_sep)} != separations_mas length "
                f"{len(separations_mas)}."
            )
        throughputs = planet_throughput_per_sep
    else:
        throughputs = tuple(planet_dark_throughput for _ in separations_mas)

    if sky_rate_hz_per_port_per_sep is not None:
        if len(sky_rate_hz_per_port_per_sep) != len(separations_mas):
            raise ValueError(
                "_contrast_curve: sky_rate_hz_per_port_per_sep length "
                f"{len(sky_rate_hz_per_port_per_sep)} != separations_mas length "
                f"{len(separations_mas)}."
            )
        sky_rates = sky_rate_hz_per_port_per_sep
    else:
        sky_rates = tuple(sky_rate_hz_per_port for _ in separations_mas)

    if leakage_rate_hz_per_sep is not None:
        if len(leakage_rate_hz_per_sep) != len(separations_mas):
            raise ValueError(
                "_contrast_curve: leakage_rate_hz_per_sep length "
                f"{len(leakage_rate_hz_per_sep)} != separations_mas length "
                f"{len(separations_mas)}."
            )
        leakage_rates = leakage_rate_hz_per_sep
    else:
        leakage_rates = tuple(leakage_rate_hz for _ in separations_mas)

    contrasts: list[float] = []
    for t, sky_rate, leakage_rate in zip(
        throughputs,
        sky_rates,
        leakage_rates,
        strict=True,
    ):
        if t <= 0.0:
            contrasts.append(math.inf)
            continue
        if sky_rate < 0.0 or not math.isfinite(float(sky_rate)):
            raise ValueError(
                "_contrast_curve: sky_rate_hz_per_port_per_sep entries must be "
                "finite and non-negative."
            )
        if leakage_rate < 0.0 or not math.isfinite(float(leakage_rate)):
            raise ValueError(
                "_contrast_curve: leakage_rate_hz_per_sep entries must be "
                "finite and non-negative."
            )
        B = float(leakage_rate) * integration_time_s
        systematic_var = _systematic_variance(
            leakage_rate_hz=float(leakage_rate),
            integration_time_s=integration_time_s,
            eps=eps,
            with_fringe_tracking=with_fringe_tracking,
            servo_bw_hz=servo_bw_hz,
        )
        Sky = float(sky_rate) * n_dark_ports * integration_time_s
        bg_variance = B + D + Sky + systematic_var
        signal_counts = 0.5 * (
            snr_sq + math.sqrt(snr_sq * snr_sq + 4.0 * snr_sq * bg_variance)
        )
        contrasts.append(signal_counts / (stellar_counts * t))
    return tuple(contrasts)
