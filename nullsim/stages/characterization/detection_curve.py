"""Characterization-mode detection-space curve.

Anchors ``optimize_characterization`` at N log-spaced separations and
emits a per-anchor 5σ contrast — the green curve in KILO paperv3's
``detection_space.pdf``. KILO equivalent:
``kilo/studies/natural_design.py:_char_mode_anchor_worker`` and
the per-J-mag char overlay loop around lines 2020-2240.

The stage consumes the realized null floor from
``sensitivity.detection_curve``, the detected stellar rate from the
detector + photometry stack, and produces a
``characterization.detection_curve`` component that the
``detection_space`` plot overlays.

Each anchor is one full SLSQP run; with the torch autograd-Jacobian
backend each takes ~30 s at M=24. 12 anchors give a smooth curve in
log-contrast vs log-separation. Disable by removing the stage from
the pipeline, or by sweeping J-mag cells you don't care about and
filtering at plot time.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator
from pydantic import Field as PydanticField

from nullsim.physics.characterization import (
    disk_sample_stellar_stack,
    optimize_characterization,
    spectro_disk_leak_from_stack,
)
from nullsim.physics.planet_throughput import planet_input_field
from nullsim.physics.ruilier_cassaing import angular_coupling_factor
from nullsim.pipeline import BaseStage, SimulationState, register_stage
from nullsim.stages._utils import torch_compile_env_enabled, unique_component_name
from nullsim.stages.characterization.spectro import (
    _RAD_PER_MAS,
    _band_evaluated_spectro_metrics,
    _characterization_wavelength_grid,
    _detected_stellar_rate_by_wavelength_hz,
    _find_sensitivity_component,
    _mc_piston_opd_covariance_m2,
    _optional_float,
    _payload_array,
    _port_indices,
    _realized_spectro_floor_for_stack,
    _representative_rate_weights,
    _require_positions,
    _resolve_band_bandwidth_m,
    _resolve_wavelengths_m,
    _result_u_stack_or_build,
    _servo_bw_hz,
    _spectro_null_instability,
    _spectro_sky_rate_hz,
)
from nullsim.stages.chip_optimization import _tracker_piston_basis_from_state
from nullsim.stages.performance.standard import (
    _combined_systematic_eps,
    _compute_snr,
    _contrast_curve,
    _find_detector_component,
)

_CHIP_OPT_PREFIX = "chip_optimization"

__all__ = ["CharacterizationDetectionCurve"]


@register_stage
class CharacterizationDetectionCurve(BaseStage):
    """Anchor-swept 5σ contrast curve in characterization mode."""

    type_name = "characterization.detection_curve"
    family = "characterization"
    consumes = frozenset({"results", "geometry", "scene", "throughput", "field"})
    produces = frozenset({"results"})

    class Params(BaseModel):
        """Validated parameters for the char-mode detection curve sweep."""

        model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

        n_modes: int = PydanticField(gt=0)
        n_bright_ports: int = PydanticField(ge=0)
        n_dark_ports: int = PydanticField(ge=0)
        n_spectro_ports: int = PydanticField(default=1, ge=1)
        wl_design_um: float | None = PydanticField(default=None, gt=0.0)
        null_floor: float = PydanticField(default=3.0e-5, gt=0.0)
        # Match detection mode (performance.standard): combine the speckle
        # stability systematic floor in quadrature with null_instability,
        # instead of using null_instability alone.
        residual_speckle_stability_factor: float = PydanticField(
            default=0.0, ge=0.0, le=1.0
        )
        # Photon-resolved null self-calibration gain applied only to the
        # instability/systematic term. Mean-leakage photon noise is untouched.
        nsc_systematic_reduction: float = PydanticField(default=1.0, ge=1.0)
        n_anchors: int = PydanticField(default=12, ge=1)
        separation_min_mas: float | None = PydanticField(default=None, gt=0.0)
        separation_max_mas: float | None = PydanticField(default=None, gt=0.0)
        position_angle_deg: float = 0.0
        n_restarts: int = PydanticField(default=3, ge=0)
        max_iter: int = PydanticField(default=5000, ge=1)
        n_workers: int = PydanticField(default=1, ge=1)
        seed: int = 42
        dac_bits: int | None = PydanticField(default=None, gt=0)
        backend: str = "auto"
        compile_objective: bool = False
        throughput_weight: float = PydanticField(default=1.0, ge=0.0)
        bright_weight: float = PydanticField(default=1.0, ge=0.0)
        snr_proxy_weight: float = PydanticField(default=0.0, ge=0.0)
        tracker_observability_weight: float = PydanticField(default=0.0, ge=0.0)
        tracker_observability_min_singular_per_nm: float = PydanticField(
            default=1.0e-3,
            gt=0.0,
        )
        tracker_observability_crlb_weight: float = PydanticField(
            default=0.0,
            ge=0.0,
        )
        tracker_observability_crlb_target_nm: float = PydanticField(
            default=100.0,
            gt=0.0,
        )
        tracker_observability_crlb_constraint_nm: float | None = PydanticField(
            default=None,
            gt=0.0,
        )
        tracker_observability_eps: float = PydanticField(default=1.0e-12, gt=0.0)
        target_snr: float = PydanticField(default=5.0, gt=0.0)
        band: str = "J"
        band_bandwidth_um: float | None = PydanticField(default=None, gt=0.0)
        broadband_n_wavelengths: int = PydanticField(default=1, ge=1)
        use_upstream_chip_params_start: bool = True
        preserve_bright_rows: bool = False
        enforce_null_constraint: bool = True
        optimization_source: str = "auto"

        disk_n_rings: int = PydanticField(default=3, ge=0)
        disk_n_azimuths: int = PydanticField(default=4, ge=0)
        disk_kernel_order: int = PydanticField(default=0, ge=0, le=4)
        disk_integral_n_radial: int = PydanticField(default=21, ge=2)
        disk_integral_n_pa: int = PydanticField(default=16, ge=1)

        n_dark_count_hz: float = PydanticField(default=0.0, ge=0.0)
        n_sky_background_hz: float = PydanticField(default=0.0, ge=0.0)
        with_fringe_tracking: bool = True
        default_servo_bw_hz: float = PydanticField(default=2000.0, gt=0.0)

        # Skip cells whose J-mag is in this set (treat the stage as a
        # no-op for those cells). Lets the user run a J-mag sweep but
        # only compute the expensive char curve at the cells whose
        # overlay matters (KILO plots J=4 and J=8 only).
        skip_jmags: tuple[float, ...] = ()

        @model_validator(mode="after")
        def _resolve_compile_env(self) -> CharacterizationDetectionCurve.Params:
            # Fold NULLSIM_TORCH_COMPILE into the validated param so the
            # executed objective path enters the stage cache key
            # (review 2026-06-10 Issue 25).
            if not self.compile_objective and torch_compile_env_enabled():
                object.__setattr__(self, "compile_objective", True)
            return self

        @model_validator(mode="after")
        def _check(self) -> CharacterizationDetectionCurve.Params:
            if (
                self.n_bright_ports + self.n_dark_ports + self.n_spectro_ports
                != self.n_modes
            ):
                raise ValueError(
                    "characterization.detection_curve port split mismatch: "
                    f"{self.n_bright_ports} + {self.n_dark_ports} + "
                    f"{self.n_spectro_ports} != {self.n_modes}."
                )
            if (
                self.separation_min_mas is not None
                and self.separation_max_mas is not None
                and self.separation_max_mas <= self.separation_min_mas
                and self.n_anchors > 1
            ):
                raise ValueError(
                    "separation_max_mas must be strictly greater than "
                    "separation_min_mas."
                )
            return self

    def apply(self, state: SimulationState) -> SimulationState:
        params = self.params_model

        # Cheap escape hatch for J-mag cells we don't care about. isclose
        # rather than exact equality so linspace/logspace-generated sweep
        # values still match a hand-typed skip list (review 2026-06-10).
        scene_jmag = state.scene.star_jmag
        if scene_jmag is not None and any(
            math.isclose(float(scene_jmag), float(skip), abs_tol=1.0e-9)
            for skip in params.skip_jmags
        ):
            return state

        sensitivity_name, sensitivity = _find_sensitivity_component(state)
        detector_name, detector = _find_detector_component(state)

        positions_m = _require_positions(state, params.n_modes)
        coupling_per_sub = _payload_array(
            sensitivity, "coupling_per_sub", params.n_modes, dtype=np.float64,
        )
        sub_areas_m2 = _payload_array(
            sensitivity, "sub_areas_m2", params.n_modes, dtype=np.float64,
        )
        sub_diameters_m = _payload_array(
            sensitivity, "sub_diameters_m", params.n_modes, dtype=np.float64,
        )
        base_amplitudes = np.sqrt(coupling_per_sub * sub_areas_m2)

        wavelengths_m = _resolve_wavelengths_m(state, sensitivity)
        wl_design_m = (
            float(params.wl_design_um) * 1.0e-6
            if params.wl_design_um is not None
            else float(wavelengths_m[len(wavelengths_m) // 2])
        )
        broadband_wavelengths_m = _characterization_wavelength_grid(
            wavelengths_m,
            params.broadband_n_wavelengths,
            center_wavelength_m=wl_design_m,
        )

        sep_min, sep_max = self._resolve_sep_bounds(sensitivity, params)
        separations_mas = np.geomspace(sep_min, sep_max, params.n_anchors)
        position_angle_rad = math.radians(float(params.position_angle_deg))

        stellar_radius_rad = 0.0
        if state.scene.star_angular_diameter_mas is not None:
            stellar_radius_rad = (
                float(state.scene.star_angular_diameter_mas) * _RAD_PER_MAS / 2.0
            )
        stellar_stack = disk_sample_stellar_stack(
            positions_m=positions_m,
            base_amplitudes=base_amplitudes,
            wavelength_m=wl_design_m,
            sub_diameters_m=sub_diameters_m,
            stellar_radius_rad=stellar_radius_rad,
            n_rings=params.disk_n_rings,
            n_azimuths=params.disk_n_azimuths,
            kernel_order=params.disk_kernel_order,
        )
        stellar_fields_per_wl = [
            disk_sample_stellar_stack(
                positions_m=positions_m,
                base_amplitudes=base_amplitudes,
                wavelength_m=float(wavelength_m),
                sub_diameters_m=sub_diameters_m,
                stellar_radius_rad=stellar_radius_rad,
                n_rings=params.disk_n_rings,
                n_azimuths=params.disk_n_azimuths,
                kernel_order=params.disk_kernel_order,
            )
            for wavelength_m in broadband_wavelengths_m
        ]
        selection_stellar_fields_per_wl = [
            disk_sample_stellar_stack(
                positions_m=positions_m,
                base_amplitudes=base_amplitudes,
                wavelength_m=float(wavelength_m),
                sub_diameters_m=sub_diameters_m,
                stellar_radius_rad=stellar_radius_rad,
                n_rings=params.disk_n_rings,
                n_azimuths=params.disk_n_azimuths,
                kernel_order=params.disk_kernel_order,
            )
            for wavelength_m in wavelengths_m
        ]

        bright_idx, dark_idx, spectro_idx = _port_indices(params)
        tracker_basis = _tracker_piston_basis_from_state(state, params.n_modes)
        tracker_active = (
            params.tracker_observability_weight > 0.0
            or params.tracker_observability_crlb_weight > 0.0
            or params.tracker_observability_crlb_constraint_nm is not None
        )
        initial_params = _initial_params(state, params)
        anchor_initial_params = initial_params

        detected_stellar_rate_by_wavelength_hz = (
            _detected_stellar_rate_by_wavelength_hz(
                state=state,
                detector_component=detector,
                wavelength_grid_um=tuple(float(w * 1.0e6) for w in wavelengths_m),
                band=params.band,
                band_bandwidth_um_override=params.band_bandwidth_um,
            )
        )
        detected_stellar_rate_hz = float(np.sum(detected_stellar_rate_by_wavelength_hz))
        optimization_stellar_weights = _representative_rate_weights(
            representative_wavelengths_m=broadband_wavelengths_m,
            full_wavelengths_m=wavelengths_m,
            full_rate_by_wavelength_hz=detected_stellar_rate_by_wavelength_hz,
        )
        aperture_weight = coupling_per_sub * sub_areas_m2
        aperture_weight_sum = float(np.sum(aperture_weight))
        if aperture_weight_sum <= 0.0:
            raise ValueError(
                "characterization.detection_curve aperture weights sum to zero."
            )

        servo_bw_hz = _servo_bw_hz(sensitivity, params)
        piston_opd_covariance_m2 = _mc_piston_opd_covariance_m2(
            sensitivity, params.n_modes,
        )
        fiber_phase_noise_rms_rad = float(
            _optional_float(sensitivity.get("fiber_phase_noise_rms_rad")) or 0.0
        )
        integration_time_s = float(detector["integration_time_s"])

        contrast_5sigma = np.empty(params.n_anchors, dtype=np.float64)
        throughput_per = np.empty(params.n_anchors, dtype=np.float64)
        on_axis_null_per = np.empty(params.n_anchors, dtype=np.float64)
        disk_leak_per = np.empty(params.n_anchors, dtype=np.float64)
        static_leak_per = np.empty(params.n_anchors, dtype=np.float64)
        realized_spectro_floor_per = np.empty(params.n_anchors, dtype=np.float64)
        leak_fraction_per = np.empty(params.n_anchors, dtype=np.float64)
        leakage_rate_per = np.empty(params.n_anchors, dtype=np.float64)
        snr_per = np.empty(params.n_anchors, dtype=np.float64)
        null_instability_per = np.empty(params.n_anchors, dtype=np.float64)
        sky_rate_per = np.empty(params.n_anchors, dtype=np.float64)
        sky_rate_per_arm = np.empty(params.n_anchors, dtype=np.float64)
        eps_systematic_per = np.empty(params.n_anchors, dtype=np.float64)
        optimization_success_per = np.empty(params.n_anchors, dtype=bool)
        constraint_margin_per = np.empty(params.n_anchors, dtype=np.float64)
        full_band_constraint_margin_per = np.empty(params.n_anchors, dtype=np.float64)
        tracker_crlb_rms_nm_per = np.full(
            params.n_anchors,
            np.nan,
            dtype=np.float64,
        )
        on_axis_null_per_wavelength = np.empty(
            (params.n_anchors, wavelengths_m.size),
            dtype=np.float64,
        )
        static_leak_per_wavelength = np.empty_like(on_axis_null_per_wavelength)
        planet_throughput_per_wavelength = np.empty_like(on_axis_null_per_wavelength)
        planet_to_star_ratio_per_wavelength = np.empty_like(on_axis_null_per_wavelength)

        for index, sep_mas in enumerate(separations_mas):
            separation_rad = float(sep_mas) * _RAD_PER_MAS
            planet_field = planet_input_field(
                positions_m=positions_m,
                coupling_per_sub=coupling_per_sub,
                sub_areas_m2=sub_areas_m2,
                sub_diameters_m=sub_diameters_m,
                wavelength_m=wl_design_m,
                separation_rad=separation_rad,
                position_angle_rad=position_angle_rad,
            )
            planet_fields_per_wl = [
                planet_input_field(
                    positions_m=positions_m,
                    coupling_per_sub=coupling_per_sub,
                    sub_areas_m2=sub_areas_m2,
                    sub_diameters_m=sub_diameters_m,
                    wavelength_m=float(wavelength_m),
                    separation_rad=separation_rad,
                    position_angle_rad=position_angle_rad,
                )
                for wavelength_m in broadband_wavelengths_m
            ]
            planet_weight_per_wl = np.asarray(
                [
                    float(
                        np.sum(
                            aperture_weight
                            * np.asarray(
                                angular_coupling_factor(
                                    separation_rad,
                                    sub_diameters_m,
                                    float(wavelength_m),
                                ),
                                dtype=np.float64,
                            )
                        )
                        / aperture_weight_sum
                    )
                    for wavelength_m in broadband_wavelengths_m
                ],
                dtype=np.float64,
            )
            optimization_planet_weights = (
                optimization_stellar_weights * planet_weight_per_wl
            )
            selection_planet_fields_per_wl = [
                planet_input_field(
                    positions_m=positions_m,
                    coupling_per_sub=coupling_per_sub,
                    sub_areas_m2=sub_areas_m2,
                    sub_diameters_m=sub_diameters_m,
                    wavelength_m=float(wavelength_m),
                    separation_rad=separation_rad,
                    position_angle_rad=position_angle_rad,
                )
                for wavelength_m in wavelengths_m
            ]
            selection_planet_weight_per_wl = np.asarray(
                [
                    float(
                        np.sum(
                            aperture_weight
                            * np.asarray(
                                angular_coupling_factor(
                                    separation_rad,
                                    sub_diameters_m,
                                    float(wavelength_m),
                                ),
                                dtype=np.float64,
                            )
                        )
                        / aperture_weight_sum
                    )
                    for wavelength_m in wavelengths_m
                ],
                dtype=np.float64,
            )
            selection_kwargs: dict[str, Any] = {
                "selection_wavelengths_m": wavelengths_m,
                "selection_stellar_fields_per_wl": selection_stellar_fields_per_wl,
                "selection_planet_fields_per_wl": selection_planet_fields_per_wl,
                "selection_stellar_weights_per_wl": (
                    detected_stellar_rate_by_wavelength_hz
                ),
                "selection_planet_weights_per_wl": (
                    detected_stellar_rate_by_wavelength_hz
                    * selection_planet_weight_per_wl
                ),
            }
            broadband_kwargs: dict[str, Any] = {}
            if params.broadband_n_wavelengths > 1:
                broadband_kwargs = {
                    "wavelengths_m": broadband_wavelengths_m,
                    "stellar_fields_per_wl": stellar_fields_per_wl,
                    "planet_fields_per_wl": planet_fields_per_wl,
                    "stellar_weights_per_wl": optimization_stellar_weights,
                    "planet_weights_per_wl": optimization_planet_weights,
                }
            result = optimize_characterization(
                stellar_field=stellar_stack,
                planet_field=planet_field,
                n_modes=params.n_modes,
                dark_port_indices=dark_idx,
                spectro_port_indices=spectro_idx,
                bright_port_indices=bright_idx,
                null_floor=params.null_floor,
                n_restarts=params.n_restarts,
                seed=int(params.seed + 1009 * index),
                max_iter=params.max_iter,
                n_workers=params.n_workers,
                throughput_weight=params.throughput_weight,
                bright_weight=params.bright_weight,
                snr_proxy_weight=params.snr_proxy_weight,
                initial_params=anchor_initial_params,
                wl_design_m=wl_design_m,
                dac_bits=params.dac_bits,
                backend=params.backend,
                compile_objective=params.compile_objective,
                tracker_piston_basis=(
                    tracker_basis.chip_basis if tracker_active else None
                ),
                tracker_field=stellar_stack[0] if tracker_active else None,
                tracker_wavelength_m=wl_design_m if tracker_active else None,
                tracker_observability_weight=params.tracker_observability_weight,
                tracker_observability_min_singular_per_nm=(
                    params.tracker_observability_min_singular_per_nm
                ),
                tracker_observability_crlb_weight=(
                    params.tracker_observability_crlb_weight
                ),
                tracker_observability_crlb_target_nm=(
                    params.tracker_observability_crlb_target_nm
                ),
                tracker_observability_crlb_constraint_nm=(
                    params.tracker_observability_crlb_constraint_nm
                ),
                tracker_observability_eps=params.tracker_observability_eps,
                preserve_bright_rows=params.preserve_bright_rows,
                enforce_null_constraint=params.enforce_null_constraint,
                deployment_wavelengths_m=wavelengths_m,
                **selection_kwargs,
                **broadband_kwargs,
            )
            u_stack = _result_u_stack_or_build(
                result=result,
                n_modes=params.n_modes,
                wavelengths_m=wavelengths_m,
                wl_design_m=wl_design_m,
            )
            _band_disk_leak, band_disk_leak_per_wl = spectro_disk_leak_from_stack(
                u_stack=u_stack,
                positions_m=positions_m,
                base_amplitudes=base_amplitudes,
                spectro_port_indices=spectro_idx,
                wavelengths_m=wavelengths_m,
                sub_diameters_m=sub_diameters_m,
                stellar_radius_mas=(
                    float(state.scene.star_angular_diameter_mas or 0.0) / 2.0
                ),
                n_radial=params.disk_integral_n_radial,
                n_pa=params.disk_integral_n_pa,
            )
            band_metrics = _band_evaluated_spectro_metrics(
                u_stack=u_stack,
                wavelengths_m=wavelengths_m,
                detected_stellar_rate_by_wavelength_hz=(
                    detected_stellar_rate_by_wavelength_hz
                ),
                positions_m=positions_m,
                coupling_per_sub=coupling_per_sub,
                sub_areas_m2=sub_areas_m2,
                sub_diameters_m=sub_diameters_m,
                base_amplitudes=base_amplitudes,
                spectro_idx=spectro_idx,
                separation_rad=separation_rad,
                position_angle_rad=position_angle_rad,
                planet_contrast=1.0,
                disk_leak_per_wavelength=band_disk_leak_per_wl,
                stellar_angular_diameter_mas=state.scene.star_angular_diameter_mas,
                piston_opd_covariance_m2=piston_opd_covariance_m2,
                fiber_phase_noise_rms_rad=fiber_phase_noise_rms_rad,
            )
            realized_spectro_floor, realized_spectro_floor_model, leak_rms = (
                _realized_spectro_floor_for_stack(
                    sensitivity=sensitivity,
                    u_stack=u_stack,
                    base_amplitudes=base_amplitudes,
                    spectro_idx=spectro_idx,
                    wavelengths_m=wavelengths_m,
                    stellar_rate_weights=detected_stellar_rate_by_wavelength_hz,
                    n_dark_ports=params.n_dark_ports,
                    n_spectro_ports=params.n_spectro_ports,
                )
            )
            on_axis_null = float(band_metrics["on_axis_null_depth"])
            static_leak_fraction = float(band_metrics["static_leak_fraction"])
            band_constraint_margin = float(params.null_floor - static_leak_fraction)
            if detected_stellar_rate_hz > 0.0:
                rate_weights = (
                    detected_stellar_rate_by_wavelength_hz / detected_stellar_rate_hz
                )
                disk_leak = float(np.sum(rate_weights * band_disk_leak_per_wl))
            else:
                disk_leak = float(np.mean(band_disk_leak_per_wl))
            leak_fraction = static_leak_fraction + realized_spectro_floor
            # Sky through this anchor's retuned spectro rows (Issue 11).
            n_pol_arms, sky_rate_hz_per_spectro_arm, sky_rate_hz_per_spectro_port, (
                sky_rate_source
            ) = _spectro_sky_rate_hz(
                state,
                detector,
                params.n_sky_background_hz,
                u_stack=u_stack,
                spectro_idx=spectro_idx,
            )
            leakage_rate_hz = leak_fraction * detected_stellar_rate_hz
            # Epsilon is this anchor's retuned spectro-port leakage CV.
            null_instability, null_instability_source = _spectro_null_instability(
                sensitivity=sensitivity,
                leak_rms=leak_rms,
                leak_fraction=leak_fraction,
            )
            eps_systematic = _combined_systematic_eps(
                null_instability=null_instability
                / float(params.nsc_systematic_reduction),
                residual_speckle_stability_factor=float(
                    params.residual_speckle_stability_factor
                ),
            )
            planet_rate_per_unit_contrast_hz = float(
                band_metrics["planet_signal_rate_hz"]
            )
            planet_rate_hz = (
                planet_rate_per_unit_contrast_hz
                * float(state.scene.planet_contrast or 0.0)
            )
            snr_payload = _compute_snr(
                planet_rate_hz=planet_rate_hz,
                leakage_rate_hz=leakage_rate_hz,
                dark_rate_hz_per_port=params.n_dark_count_hz,
                sky_rate_hz_per_port=sky_rate_hz_per_spectro_port,
                n_dark_ports=params.n_spectro_ports,
                integration_time_s=integration_time_s,
                eps=eps_systematic,
                with_fringe_tracking=params.with_fringe_tracking,
                servo_bw_hz=servo_bw_hz,
            )
            effective_throughput = (
                planet_rate_per_unit_contrast_hz / detected_stellar_rate_hz
                if detected_stellar_rate_hz > 0.0
                else 0.0
            )

            # Characterization mode does not carry a bright/dark port
            # split — ``detected_stellar_rate_hz`` is the post-throughput
            # total. Passing it as ``bright_rate_hz`` is safe: that arg
            # only acts as a positivity guard in ``_contrast_curve``
            # (see its docstring). ``stellar_rate_hz`` is the contrast
            # denominator, where total post-throughput stellar is the
            # KILO convention.
            contrast = _contrast_curve(
                separations_mas=(float(sep_mas),),
                bright_rate_hz=detected_stellar_rate_hz,
                stellar_rate_hz=detected_stellar_rate_hz,
                leakage_rate_hz=leakage_rate_hz,
                dark_rate_hz_per_port=params.n_dark_count_hz,
                sky_rate_hz_per_port=sky_rate_hz_per_spectro_port,
                n_dark_ports=params.n_spectro_ports,
                integration_time_s=integration_time_s,
                eps=eps_systematic,
                with_fringe_tracking=params.with_fringe_tracking,
                servo_bw_hz=servo_bw_hz,
                target_snr=params.target_snr,
                planet_throughput_per_sep=(effective_throughput,),
            )[0]

            contrast_5sigma[index] = float(contrast)
            throughput_per[index] = float(effective_throughput)
            on_axis_null_per[index] = float(on_axis_null)
            disk_leak_per[index] = float(disk_leak)
            static_leak_per[index] = float(static_leak_fraction)
            realized_spectro_floor_per[index] = float(realized_spectro_floor)
            leak_fraction_per[index] = float(leak_fraction)
            leakage_rate_per[index] = float(leakage_rate_hz)
            snr_per[index] = float(snr_payload["snr"])
            null_instability_per[index] = null_instability
            sky_rate_per[index] = sky_rate_hz_per_spectro_port
            sky_rate_per_arm[index] = sky_rate_hz_per_spectro_arm
            eps_systematic_per[index] = eps_systematic
            optimization_success_per[index] = bool(result.optimization_success)
            constraint_margin_per[index] = float(result.constraint_margin)
            full_band_constraint_margin_per[index] = band_constraint_margin
            if result.tracker_crlb_rms_nm is not None:
                tracker_crlb_rms_nm_per[index] = float(result.tracker_crlb_rms_nm)
            if result.optimization_success and np.all(np.isfinite(result.params)):
                anchor_initial_params = (np.asarray(result.params, dtype=np.float64),)
            on_axis_null_per_wavelength[index] = np.asarray(
                band_metrics["on_axis_null_per_wavelength"],
                dtype=np.float64,
            )
            static_leak_per_wavelength[index] = np.asarray(
                band_metrics["static_leak_fraction_per_wavelength"],
                dtype=np.float64,
            )
            planet_throughput_per_wavelength[index] = np.asarray(
                band_metrics["planet_throughput_per_wavelength"],
                dtype=np.float64,
            )
            planet_to_star_ratio_per_wavelength[index] = np.asarray(
                band_metrics["planet_to_star_ratio_per_wavelength"],
                dtype=np.float64,
            )

        band_bandwidth_m = _resolve_band_bandwidth_m(
            wavelengths_m, params.band_bandwidth_um,
        )

        component_name = unique_component_name(
            self.type_name,
            state.results.components,
            self.context.instance_id,
        )
        payload = {
            "separations_mas": separations_mas.astype(np.float64, copy=True),
            "contrast_5sigma": contrast_5sigma.astype(np.float64, copy=True),
            "spectro_throughput_per_anchor": throughput_per.astype(
                np.float64, copy=True,
            ),
            "spectro_on_axis_null_per_anchor": on_axis_null_per.astype(
                np.float64, copy=True,
            ),
            "spectro_disk_leak_per_anchor": disk_leak_per.astype(
                np.float64, copy=True,
            ),
            "spectro_static_leak_fraction_per_anchor": static_leak_per.astype(
                np.float64, copy=True,
            ),
            "realized_spectro_floor_per_anchor": realized_spectro_floor_per.astype(
                np.float64, copy=True,
            ),
            "spectro_leak_fraction_per_anchor": leak_fraction_per.astype(
                np.float64, copy=True,
            ),
            "spectro_on_axis_null_per_anchor_wavelength": (
                on_axis_null_per_wavelength.astype(np.float64, copy=True)
            ),
            "spectro_static_leak_fraction_per_anchor_wavelength": (
                static_leak_per_wavelength.astype(np.float64, copy=True)
            ),
            "spectro_planet_throughput_per_anchor_wavelength": (
                planet_throughput_per_wavelength.astype(np.float64, copy=True)
            ),
            "spectro_planet_to_star_ratio_per_anchor_wavelength": (
                planet_to_star_ratio_per_wavelength.astype(np.float64, copy=True)
            ),
            "leakage_rate_hz_per_anchor": leakage_rate_per.astype(
                np.float64, copy=True,
            ),
            "snr_per_anchor": snr_per.astype(np.float64, copy=True),
            "optimization_success_per_anchor": optimization_success_per.astype(
                bool, copy=True,
            ),
            "optimization_constraint_margin_per_anchor": (
                constraint_margin_per.astype(np.float64, copy=True)
            ),
            "full_band_constraint_margin_per_anchor": (
                full_band_constraint_margin_per.astype(np.float64, copy=True)
            ),
            "full_band_constraint_margin_model": (
                "full detector-grid static spectro leakage: "
                "null_floor - spectro_static_leak_fraction_per_anchor"
            ),
            "tracker_crlb_rms_nm_per_anchor": tracker_crlb_rms_nm_per.astype(
                np.float64,
                copy=True,
            ),
            "tracker_observability_crlb_constraint_nm": (
                None
                if params.tracker_observability_crlb_constraint_nm is None
                else float(params.tracker_observability_crlb_constraint_nm)
            ),
            "realized_spectro_floor": float(np.nanmedian(realized_spectro_floor_per)),
            "realized_spectro_floor_model": realized_spectro_floor_model,
            "detected_stellar_rate_hz": float(detected_stellar_rate_hz),
            "detected_stellar_rate_by_wavelength_hz": (
                detected_stellar_rate_by_wavelength_hz.astype(np.float64, copy=True)
            ),
            "integration_time_s": integration_time_s,
            "band_lambda_m": float(wl_design_m),
            "band_bandwidth_m": float(band_bandwidth_m),
            "wavelengths_m": tuple(float(w) for w in wavelengths_m),
            "broadband_wavelengths_m": tuple(
                float(w) for w in broadband_wavelengths_m
            ),
            "selection_wavelengths_m": tuple(float(w) for w in wavelengths_m),
            "candidate_selection_model": (
                "restart candidates ranked on the full detector wavelength grid "
                "using photon-rate-weighted characterization SNR proxy"
            ),
            # Anchor medians; the sky depends on each anchor's retuned rows.
            "sky_rate_hz_per_spectro_port": float(np.median(sky_rate_per)),
            "sky_rate_hz_per_spectro_port_per_anchor": sky_rate_per.astype(
                np.float64, copy=True,
            ),
            "sky_rate_hz_per_spectro_arm": float(np.median(sky_rate_per_arm)),
            "n_pol_arms": int(n_pol_arms),
            "sky_rate_source": sky_rate_source,
            "position_angle_deg": float(params.position_angle_deg),
            "target_snr": float(params.target_snr),
            "n_anchors": int(params.n_anchors),
            "n_workers": int(params.n_workers),
            "null_floor": float(params.null_floor),
            # Scalars are anchor medians, like realized_spectro_floor.
            "null_instability": float(np.nanmedian(null_instability_per)),
            "null_instability_per_anchor": null_instability_per.astype(
                np.float64, copy=True,
            ),
            "null_instability_source": null_instability_source,
            "eps_systematic": float(np.nanmedian(eps_systematic_per)),
            "eps_systematic_per_anchor": eps_systematic_per.astype(
                np.float64, copy=True,
            ),
            # 0 when with_fringe_tracking is false: the quasi-static branch.
            "servo_bw_hz": float(servo_bw_hz),
            "with_fringe_tracking": bool(params.with_fringe_tracking),
            "nsc_systematic_reduction": float(params.nsc_systematic_reduction),
            "throughput_weight": float(params.throughput_weight),
            "bright_weight": float(params.bright_weight),
            "snr_proxy_weight": float(params.snr_proxy_weight),
            "enforce_null_constraint": bool(params.enforce_null_constraint),
            "source_model": (
                "Anchors optimize on a representative wavelength grid and "
                "evaluate stellar leakage plus planet signal over the full "
                "detector wavelength grid using photon-rate weights."
            ),
            "upstream_sensitivity_component": sensitivity_name,
            "upstream_detector_component": detector_name,
        }
        results = state.results.add_component(component_name, payload)
        return replace(state, results=results)

    def _resolve_sep_bounds(
        self,
        sensitivity: dict[str, Any],
        params: CharacterizationDetectionCurve.Params,
    ) -> tuple[float, float]:
        if params.separation_min_mas is not None and params.separation_max_mas is not None:
            return float(params.separation_min_mas), float(params.separation_max_mas)
        sens_seps = sensitivity.get("separations_mas")
        if sens_seps is not None and len(sens_seps) >= 2:
            seps_arr = np.asarray(sens_seps, dtype=np.float64)
            seps_arr = seps_arr[seps_arr > 0.0]
            if seps_arr.size >= 2:
                return (
                    float(params.separation_min_mas) if params.separation_min_mas
                    else float(seps_arr.min()),
                    float(params.separation_max_mas) if params.separation_max_mas
                    else float(seps_arr.max()),
                )
        raise ValueError(
            "characterization.detection_curve: cannot resolve separation "
            "bounds. Either set separation_min_mas + separation_max_mas, or "
            "ensure the upstream sensitivity.detection_curve component "
            "publishes a separations_mas grid."
        )


def _initial_params(
    state: SimulationState,
    params: CharacterizationDetectionCurve.Params,
) -> tuple[np.ndarray, ...]:
    if not params.use_upstream_chip_params_start:
        return ()
    components = state.results.components
    if params.optimization_source.lower() == "auto":
        candidates = [
            payload
            for name, payload in components.items()
            if name.startswith(_CHIP_OPT_PREFIX)
            and isinstance(payload, dict)
            and "optimal_chip_params" in payload
        ]
        if not candidates:
            return ()
        raw = candidates[0]["optimal_chip_params"]
    else:
        payload = components.get(params.optimization_source)
        if not isinstance(payload, dict) or "optimal_chip_params" not in payload:
            raise KeyError(
                "characterization.detection_curve optimization_source "
                f"{params.optimization_source!r} did not resolve to a "
                "chip_optimization component with optimal_chip_params."
            )
        raw = payload["optimal_chip_params"]
    return (np.asarray(raw, dtype=np.float64),)
