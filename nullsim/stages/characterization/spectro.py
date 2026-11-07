"""Known-position spectro-port characterization stage."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator
from pydantic import Field as PydanticField

from nullsim.physics.characterization import (
    build_unitary_stack_from_params,
    disk_sample_stellar_stack,
    mode_piston_opd_covariance_m2,
    optimize_characterization,
    piston_averaged_port_power,
    spectro_disk_leak,
    spectro_disk_leak_from_stack,
)
from nullsim.physics.mkid import mkid_resolving_power
from nullsim.physics.photometry import photon_rate_per_m2_per_bin
from nullsim.physics.planet_throughput import planet_input_field
from nullsim.physics.realized_null import realized_null_depth_mc
from nullsim.physics.ruilier_cassaing import angular_coupling_factor
from nullsim.physics.spectroscopy import (
    AWGSpectrum,
    awg_spectrum,
    max_spectral_resolution,
    per_bin_snr,
)
from nullsim.pipeline import BaseStage, SimulationState, register_stage
from nullsim.stages._utils import (
    mode_origin_positions_m,
    project_aperture_positions_for_pointing,
    torch_compile_env_enabled,
    unique_component_name,
)
from nullsim.stages.chip_optimization import _tracker_piston_basis_from_state
from nullsim.stages.performance.standard import (
    _combined_systematic_eps,
    _compute_snr,
    _find_detector_component,
    _resolve_background_rate_hz_per_port,
)

__all__ = ["SpectroCharacterization"]

_RAD_PER_MAS = 1.0 / 206265e3
_SENSITIVITY_PREFIX = "sensitivity.detection_curve"
_CHIP_OPT_PREFIX = "chip_optimization"

# AWG channel count is a hardware design parameter, independent of the
# ``resolving_power_cap`` search bound used by ``max_spectral_resolution``.
# Defaulting it here (instead of reusing ``resolving_power_cap``) keeps a
# deliberately uncapped max-R search from inflating the AWG bin grid into a
# multi-gigabyte allocation.
_DEFAULT_AWG_RESOLVING_POWER = 10_000.0


@register_stage
class SpectroCharacterization(BaseStage):
    """Retune the chip for a known planet and compute spectro-port SNR."""

    type_name = "characterization.spectro"
    family = "characterization"
    consumes = frozenset({"results", "geometry", "scene", "throughput", "field"})
    produces = frozenset({"results"})

    class Params(BaseModel):
        """Validated parameters for spectro-port characterization."""

        model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

        n_modes: int = PydanticField(gt=0)
        n_bright_ports: int = PydanticField(ge=0)
        n_dark_ports: int = PydanticField(ge=0)
        n_spectro_ports: int = PydanticField(default=1, ge=1)
        wl_design_um: float | None = PydanticField(default=None, gt=0.0)
        null_floor: float = PydanticField(default=3.0e-5, gt=0.0)
        # Match detection mode (performance.standard): a known-position retune
        # carries the same speckle-stability systematic floor, combined in
        # quadrature with null_instability.
        residual_speckle_stability_factor: float = PydanticField(
            default=0.0, ge=0.0, le=1.0
        )
        # Photon-resolved null self-calibration gain applied only to the
        # instability/systematic term. Mean-leakage photon noise is untouched.
        nsc_systematic_reduction: float = PydanticField(default=1.0, ge=1.0)
        n_restarts: int = PydanticField(default=1, ge=0)
        max_iter: int = PydanticField(default=1000, ge=1)
        n_workers: int = PydanticField(default=1, ge=1)
        seed: int | None = None
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

        disk_n_rings: int = PydanticField(default=3, ge=0)
        disk_n_azimuths: int = PydanticField(default=4, ge=0)
        disk_kernel_order: int = PydanticField(default=0, ge=0, le=4)
        disk_integral_n_radial: int = PydanticField(default=21, ge=2)
        disk_integral_n_pa: int = PydanticField(default=16, ge=1)
        broadband_n_wavelengths: int = PydanticField(default=1, ge=1)

        use_upstream_chip_params_start: bool = True
        preserve_bright_rows: bool = False
        enforce_null_constraint: bool = True
        optimization_source: str = "auto"
        dac_bits: int | None = PydanticField(default=None, gt=0)

        snr_min_per_bin: float = PydanticField(default=3.0, gt=0.0)
        resolving_power_cap: float = PydanticField(default=10_000.0, gt=0.0)
        resolving_power_floor: float = PydanticField(default=1.0, gt=0.0)
        awg_resolving_power: float | None = PydanticField(default=None, gt=0.0)
        awg_throughput: float = PydanticField(default=1.0, ge=0.0, le=1.0)
        awg_free_spectral_range_fraction: float = PydanticField(
            default=1.0,
            gt=0.0,
            le=1.0,
        )
        awg_order: int = PydanticField(default=1, ge=1)
        awg_pixels_per_resolution_element: int = PydanticField(default=2, ge=1)
        spectrograph: Literal["awg", "mkid"] = "awg"
        band_bandwidth_um: float | None = PydanticField(default=None, gt=0.0)
        band: str = "J"
        n_spectro_pixels: int = PydanticField(default=1, ge=1)
        # Both rates are **per port** (KILO ``n_*`` convention from
        # signal_to_noise). The total count budget for the n_spectro_ports
        # spectro-detector layout is n_dark_count_hz * n_spectro_ports *
        # T (and likewise for sky). Set as if you're configuring a
        # single detector pixel/port.
        n_dark_count_hz: float = PydanticField(default=0.0, ge=0.0)
        n_sky_background_hz: float = PydanticField(default=0.0, ge=0.0)
        with_fringe_tracking: bool = True
        default_servo_bw_hz: float = PydanticField(default=2000.0, gt=0.0)

        backend: str = "auto"
        compile_objective: bool = False

        @model_validator(mode="after")
        def _resolve_compile_env(self) -> SpectroCharacterization.Params:
            # Fold NULLSIM_TORCH_COMPILE into the validated param so the
            # executed objective path enters the stage cache key
            # (review 2026-06-10 Issue 25).
            if not self.compile_objective and torch_compile_env_enabled():
                object.__setattr__(self, "compile_objective", True)
            return self

        @model_validator(mode="after")
        def _check_port_split(self) -> SpectroCharacterization.Params:
            if (
                self.n_bright_ports
                + self.n_dark_ports
                + self.n_spectro_ports
                != self.n_modes
            ):
                raise ValueError(
                    "characterization.spectro requires "
                    "n_bright_ports + n_dark_ports + n_spectro_ports == n_modes "
                    f"(got {self.n_bright_ports} + {self.n_dark_ports} + "
                    f"{self.n_spectro_ports} != {self.n_modes})."
                )
            if self.resolving_power_cap < self.resolving_power_floor:
                raise ValueError(
                    "resolving_power_cap must be >= resolving_power_floor."
                )
            if self.spectrograph == "mkid" and self.n_spectro_ports != 1:
                raise ValueError(
                    'characterization.spectro spectrograph="mkid" currently '
                    "supports exactly one spectro port; multi-port MKID "
                    "aggregation has not been defined yet."
                )
            return self

    def apply(self, state: SimulationState) -> SimulationState:
        params = self.params_model

        sensitivity_name, sensitivity = _find_sensitivity_component(state)
        if params.spectrograph == "mkid":
            detector_name, detector = _find_mkid_detector_component(state)
        else:
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

        separation_rad = _require_scene_float(
            state.scene.planet_separation_mas,
            "scene.planet_separation_mas",
        ) * _RAD_PER_MAS
        position_angle_rad = math.radians(
            _require_scene_float(
                state.scene.planet_position_angle_deg,
                "scene.planet_position_angle_deg",
            )
        )
        planet_field = planet_input_field(
            positions_m=positions_m,
            coupling_per_sub=coupling_per_sub,
            sub_areas_m2=sub_areas_m2,
            sub_diameters_m=sub_diameters_m,
            wavelength_m=wl_design_m,
            separation_rad=separation_rad,
            position_angle_rad=position_angle_rad,
        )

        detected_stellar_rate_by_wavelength_hz = _detected_stellar_rate_by_wavelength_hz(
            state=state,
            detector_component=detector,
            wavelength_grid_um=tuple(float(w * 1.0e6) for w in wavelengths_m),
            band=params.band,
            band_bandwidth_um_override=params.band_bandwidth_um,
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
            raise ValueError("characterization.spectro aperture weights sum to zero.")
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
            "selection_stellar_weights_per_wl": detected_stellar_rate_by_wavelength_hz,
            "selection_planet_weights_per_wl": (
                detected_stellar_rate_by_wavelength_hz
                * selection_planet_weight_per_wl
            ),
        }

        broadband_kwargs: dict[str, Any] = {}
        if params.broadband_n_wavelengths > 1:
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
            broadband_kwargs = {
                "wavelengths_m": broadband_wavelengths_m,
                "stellar_fields_per_wl": stellar_fields_per_wl,
                "planet_fields_per_wl": planet_fields_per_wl,
                "stellar_weights_per_wl": optimization_stellar_weights,
                "planet_weights_per_wl": optimization_planet_weights,
            }

        bright_idx, dark_idx, spectro_idx = _port_indices(params)
        initial_params = _initial_params(state, params)
        tracker_basis = _tracker_piston_basis_from_state(state, params.n_modes)
        tracker_active = (
            params.tracker_observability_weight > 0.0
            or params.tracker_observability_crlb_weight > 0.0
            or params.tracker_observability_crlb_constraint_nm is not None
        )
        result = optimize_characterization(
            stellar_field=stellar_stack,
            planet_field=planet_field,
            n_modes=params.n_modes,
            dark_port_indices=dark_idx,
            spectro_port_indices=spectro_idx,
            bright_port_indices=bright_idx,
            null_floor=params.null_floor,
            n_restarts=params.n_restarts,
            seed=params.seed,
            max_iter=params.max_iter,
            n_workers=params.n_workers,
            throughput_weight=params.throughput_weight,
            bright_weight=params.bright_weight,
            snr_proxy_weight=params.snr_proxy_weight,
            initial_params=initial_params,
            wl_design_m=wl_design_m,
            dac_bits=params.dac_bits,
            backend=params.backend,
            compile_objective=params.compile_objective,
            tracker_piston_basis=tracker_basis.chip_basis if tracker_active else None,
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
        design_index = int(np.argmin(np.abs(wavelengths_m - wl_design_m)))

        if params.broadband_n_wavelengths > 1:
            broadband_indices = _wavelength_indices(
                source_wavelengths_m=wavelengths_m,
                target_wavelengths_m=broadband_wavelengths_m,
            )
            disk_leak, disk_leak_per_wl = spectro_disk_leak_from_stack(
                u_stack=u_stack[:, :, broadband_indices],
                positions_m=positions_m,
                base_amplitudes=base_amplitudes,
                spectro_port_indices=spectro_idx,
                wavelengths_m=broadband_wavelengths_m,
                sub_diameters_m=sub_diameters_m,
                stellar_radius_mas=(
                    float(state.scene.star_angular_diameter_mas or 0.0) / 2.0
                ),
                n_radial=params.disk_integral_n_radial,
                n_pa=params.disk_integral_n_pa,
            )
        else:
            disk_leak = spectro_disk_leak(
                unitary=u_stack[:, :, design_index],
                positions_m=positions_m,
                base_amplitudes=base_amplitudes,
                spectro_port_indices=spectro_idx,
                wavelength_m=wl_design_m,
                sub_diameters_m=sub_diameters_m,
                stellar_radius_mas=(
                    float(state.scene.star_angular_diameter_mas or 0.0) / 2.0
                ),
                n_radial=params.disk_integral_n_radial,
                n_pa=params.disk_integral_n_pa,
            )
            disk_leak_per_wl = np.array([disk_leak], dtype=np.float64)

        planet_contrast = _require_scene_float(
            state.scene.planet_contrast,
            "scene.planet_contrast",
        )

        realized_spectro_floor, realized_spectro_floor_model, spectro_leak_rms = (
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
        band_disk_leak, band_disk_leak_per_wl = spectro_disk_leak_from_stack(
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
        unitary_at_design = u_stack[:, :, design_index]
        on_axis_field = stellar_stack[0]
        on_axis_total = float(np.sum(np.abs(on_axis_field) ** 2))
        if on_axis_total > 0.0:
            on_axis_output = unitary_at_design @ on_axis_field
            on_axis_null = float(
                np.sum(np.abs(on_axis_output[spectro_idx]) ** 2)
            ) / on_axis_total
        else:
            on_axis_null = 0.0
        static_leak_fraction = _static_spectro_leak_fraction(
            on_axis_null=on_axis_null,
            disk_leak=disk_leak,
            stellar_angular_diameter_mas=state.scene.star_angular_diameter_mas,
        )
        band_metrics = _band_evaluated_spectro_metrics(
            u_stack=u_stack,
            wavelengths_m=wavelengths_m,
            detected_stellar_rate_by_wavelength_hz=detected_stellar_rate_by_wavelength_hz,
            positions_m=positions_m,
            coupling_per_sub=coupling_per_sub,
            sub_areas_m2=sub_areas_m2,
            sub_diameters_m=sub_diameters_m,
            base_amplitudes=base_amplitudes,
            spectro_idx=spectro_idx,
            separation_rad=separation_rad,
            position_angle_rad=position_angle_rad,
            planet_contrast=planet_contrast,
            disk_leak_per_wavelength=band_disk_leak_per_wl,
            stellar_angular_diameter_mas=state.scene.star_angular_diameter_mas,
            piston_opd_covariance_m2=_mc_piston_opd_covariance_m2(
                sensitivity, params.n_modes,
            ),
            fiber_phase_noise_rms_rad=float(
                _optional_float(sensitivity.get("fiber_phase_noise_rms_rad")) or 0.0
            ),
        )
        static_leak_fraction_for_rates = float(
            band_metrics["static_leak_fraction"]
        )
        planet_signal_rate_hz = float(band_metrics["planet_signal_rate_hz"])
        spectro_design_leak_fraction = static_leak_fraction + realized_spectro_floor
        spectro_leak_fraction = (
            static_leak_fraction_for_rates + realized_spectro_floor
        )
        leakage_rate_hz = spectro_leak_fraction * detected_stellar_rate_hz

        null_instability, null_instability_source = _spectro_null_instability(
            sensitivity=sensitivity,
            leak_rms=spectro_leak_rms,
            leak_fraction=spectro_leak_fraction,
        )
        eps_systematic = _combined_systematic_eps(
            null_instability=null_instability
            / float(params.nsc_systematic_reduction),
            residual_speckle_stability_factor=float(
                params.residual_speckle_stability_factor
            ),
        )
        servo_bw_hz = _servo_bw_hz(sensitivity, params)
        n_pol_arms, sky_rate_hz_per_spectro_arm, sky_rate_hz_per_spectro_port, (
            sky_rate_source
        ) = _spectro_sky_rate_hz(state, detector, params.n_sky_background_hz)
        snr_breakdown = _compute_snr(
            planet_rate_hz=planet_signal_rate_hz,
            leakage_rate_hz=leakage_rate_hz,
            dark_rate_hz_per_port=params.n_dark_count_hz,
            sky_rate_hz_per_port=sky_rate_hz_per_spectro_port,
            n_dark_ports=params.n_spectro_ports,
            integration_time_s=float(detector["integration_time_s"]),
            eps=eps_systematic,
            with_fringe_tracking=params.with_fringe_tracking,
            servo_bw_hz=servo_bw_hz,
        )

        band_bandwidth_m = _resolve_band_bandwidth_m(
            wavelengths_m,
            params.band_bandwidth_um,
        )
        integration_time_s = float(detector["integration_time_s"])
        spectro_pixels_per_resolution_element = (
            1
            if params.spectrograph == "mkid"
            else int(params.awg_pixels_per_resolution_element)
        )
        # Use the SAME combined systematic eps as the headline SNR
        # (NSC-reduced null instability + speckle stability in quadrature).
        # Pre-fix the per-bin/max-R spectrograph metrics used the raw
        # null_instability — a 100x larger systematic variance in the same
        # payload row (review 2026-06-10 Issue 3).
        snr_limited_max_r = _snr_limited_max_spectral_resolution(
            params=params,
            planet_signal_rate_hz=planet_signal_rate_hz,
            leakage_rate_hz=leakage_rate_hz,
            wl_design_m=wl_design_m,
            band_bandwidth_m=band_bandwidth_m,
            integration_time_s=integration_time_s,
            null_instability=eps_systematic,
            servo_bw_hz=servo_bw_hz,
            sky_rate_hz_per_spectro_port=sky_rate_hz_per_spectro_port,
            pixels_per_resolution_element=spectro_pixels_per_resolution_element,
        )
        if params.spectrograph == "mkid":
            achievable_max_r, spectrograph_payload = _mkid_spectrograph_payload(
                params=params,
                detector=detector,
                planet_signal_rate_hz=planet_signal_rate_hz,
                leakage_rate_hz=leakage_rate_hz,
                wl_design_m=wl_design_m,
                band_bandwidth_m=band_bandwidth_m,
                integration_time_s=integration_time_s,
                null_instability=eps_systematic,
                servo_bw_hz=servo_bw_hz,
                sky_rate_hz_per_spectro_port=sky_rate_hz_per_spectro_port,
                snr_limited_max_r=snr_limited_max_r,
            )
        else:
            achievable_max_r, spectrograph_payload = _awg_spectrograph_payload(
                params=params,
                planet_signal_rate_hz=planet_signal_rate_hz,
                leakage_rate_hz=leakage_rate_hz,
                wl_design_m=wl_design_m,
                band_bandwidth_m=band_bandwidth_m,
                integration_time_s=integration_time_s,
                null_instability=eps_systematic,
                servo_bw_hz=servo_bw_hz,
                sky_rate_hz_per_spectro_port=sky_rate_hz_per_spectro_port,
                snr_limited_max_r=snr_limited_max_r,
            )

        component_name = unique_component_name(
            self.type_name,
            state.results.components,
            self.context.instance_id,
        )
        payload = {
            "spectro_throughput": float(
                band_metrics["planet_throughput_per_wavelength"][design_index]
            ),
            "spectro_null_depth": float(result.spectro_null_depth),
            "spectro_on_axis_null_depth": float(on_axis_null),
            "dark_null_depth": float(result.dark_null_depth),
            "bright_stellar_fraction": float(result.bright_stellar_fraction),
            "snr_proxy_weight": float(params.snr_proxy_weight),
            "tracker_observability_weight": float(
                params.tracker_observability_weight
            ),
            "tracker_observability_min_singular_per_nm": float(
                params.tracker_observability_min_singular_per_nm
            ),
            "tracker_observability_crlb_weight": float(
                params.tracker_observability_crlb_weight
            ),
            "tracker_observability_crlb_target_nm": float(
                params.tracker_observability_crlb_target_nm
            ),
            "tracker_observability_crlb_constraint_nm": (
                None
                if params.tracker_observability_crlb_constraint_nm is None
                else float(params.tracker_observability_crlb_constraint_nm)
            ),
            "preserve_bright_rows": bool(params.preserve_bright_rows),
            "enforce_null_constraint": bool(params.enforce_null_constraint),
            "tracker_observability_basis_source": tracker_basis.source,
            "tracker_observability_n_physical_modes": int(
                tracker_basis.n_physical_modes
            ),
            "tracker_observability_singular_values_per_nm": np.asarray(
                result.tracker_singular_values_per_nm,
                dtype=np.float64,
            )
            if result.tracker_singular_values_per_nm is not None
            else np.empty(0, dtype=np.float64),
            "tracker_observability_crlb_rms_nm": (
                None
                if result.tracker_crlb_rms_nm is None
                else float(result.tracker_crlb_rms_nm)
            ),
            "tracker_observability_loss": float(result.tracker_loss),
            "tracker_observability_crlb_loss": float(result.tracker_crlb_loss),
            "spectro_disk_leak": float(disk_leak),
            "spectro_disk_leak_per_wavelength": disk_leak_per_wl.astype(
                np.float64, copy=True,
            ),
            "spectro_band_disk_leak": float(band_disk_leak),
            "spectro_band_disk_leak_per_wavelength": band_disk_leak_per_wl.astype(
                np.float64, copy=True,
            ),
            "realized_spectro_floor": float(realized_spectro_floor),
            "realized_spectro_floor_model": realized_spectro_floor_model,
            "spectro_design_static_leak_fraction": float(static_leak_fraction),
            "spectro_design_leak_fraction": float(spectro_design_leak_fraction),
            "spectro_static_leak_fraction": static_leak_fraction_for_rates,
            "spectro_leak_fraction": float(spectro_leak_fraction),
            "spectro_band_static_leak_fraction": static_leak_fraction_for_rates,
            # Audit margin against the FULL detector grid, mirroring the
            # curve stage's full_band_constraint_margin_per_anchor: with the
            # cheap design-bin objective, optimization_constraint_margin
            # reflects design-bin feasibility only (review 2026-06-10
            # Issue 22).
            "full_band_constraint_margin": float(
                params.null_floor - float(static_leak_fraction_for_rates)
            ),
            "full_band_constraint_margin_model": (
                "full detector-grid static spectro leakage: "
                "null_floor - spectro_band_static_leak_fraction"
            ),
            "spectro_band_leak_fraction": float(spectro_leak_fraction),
            "spectro_band_on_axis_null_depth": float(
                band_metrics["on_axis_null_depth"]
            ),
            "spectro_on_axis_null_depth_per_wavelength": np.asarray(
                band_metrics["on_axis_null_per_wavelength"],
                dtype=np.float64,
            ),
            "spectro_static_leak_fraction_per_wavelength": np.asarray(
                band_metrics["static_leak_fraction_per_wavelength"],
                dtype=np.float64,
            ),
            "spectro_planet_throughput_per_wavelength": np.asarray(
                band_metrics["planet_throughput_per_wavelength"],
                dtype=np.float64,
            ),
            "spectro_planet_to_star_ratio_per_wavelength": np.asarray(
                band_metrics["planet_to_star_ratio_per_wavelength"],
                dtype=np.float64,
            ),
            "planet_signal_rate_hz": float(planet_signal_rate_hz),
            "leakage_rate_hz": float(leakage_rate_hz),
            "detected_stellar_rate_hz": float(detected_stellar_rate_hz),
            "detected_stellar_rate_by_wavelength_hz": (
                detected_stellar_rate_by_wavelength_hz.astype(np.float64, copy=True)
            ),
            "snr": float(snr_breakdown["snr"]),
            "S": float(snr_breakdown["S"]),
            "B": float(snr_breakdown["B"]),
            "D": float(snr_breakdown["D"]),
            "Sky": float(snr_breakdown["Sky"]),
            "sky_rate_hz_per_spectro_port": float(sky_rate_hz_per_spectro_port),
            "sky_rate_hz_per_spectro_arm": float(sky_rate_hz_per_spectro_arm),
            "n_pol_arms": int(n_pol_arms),
            "sky_rate_source": sky_rate_source,
            "systematic_var": float(snr_breakdown["systematic_var"]),
            "max_spectral_resolution": float(achievable_max_r),
            "snr_min_per_bin": float(params.snr_min_per_bin),
            "resolving_power_cap": float(params.resolving_power_cap),
            "band_lambda_m": float(wl_design_m),
            "band_bandwidth_m": float(band_bandwidth_m),
            "null_instability": float(null_instability),
            "null_instability_source": null_instability_source,
            "spectro_leak_rms_fraction": (
                None if spectro_leak_rms is None else float(spectro_leak_rms)
            ),
            "detection_null_instability": float(
                sensitivity.get("null_instability", 0.0)
            ),
            "eps_systematic": float(eps_systematic),
            "nsc_systematic_reduction": float(params.nsc_systematic_reduction),
            "null_floor": float(params.null_floor),
            # 0 when with_fringe_tracking is false: the quasi-static branch.
            "servo_bw_hz": float(servo_bw_hz),
            # Which branch of _systematic_variance produced systematic_var.
            # Downstream consumers (characterization.hdc_ccf) otherwise have to
            # infer it by matching the published variance against both closed
            # forms, which is ambiguous at 2 * servo_bw_hz * t == 1.
            "with_fringe_tracking": bool(params.with_fringe_tracking),
            "optimal_chip_params": tuple(float(v) for v in result.params),
            "optimal_u_stack": u_stack.astype(np.complex128, copy=True),
            "wavelengths_m": tuple(float(w) for w in wavelengths_m),
            "broadband_wavelengths_m": tuple(
                float(w) for w in broadband_wavelengths_m
            ),
            "selection_wavelengths_m": tuple(float(w) for w in wavelengths_m),
            "candidate_selection_model": (
                "restart candidates ranked on the full detector wavelength grid "
                "using photon-rate-weighted characterization SNR proxy"
            ),
            "bright_indices": bright_idx.astype(np.intp, copy=True),
            "dark_indices": dark_idx.astype(np.intp, copy=True),
            "spectro_indices": spectro_idx.astype(np.intp, copy=True),
            "n_restarts_completed": int(result.n_restarts_completed),
            "n_workers": int(params.n_workers),
            # Deployed-design metric recomputed at the design wavelength;
            # in broadband mode this is NOT the objective SLSQP minimized
            # (review 2026-06-10 Issue 22) — that is published separately.
            "objective_value": float(result.objective_value),
            "optimizer_objective_value": (
                None
                if result.optimizer_objective_value is None
                else float(result.optimizer_objective_value)
            ),
            "optimization_success": bool(result.optimization_success),
            "optimization_constraint_margin": float(result.constraint_margin),
            "upstream_detector_component": detector_name,
            "upstream_sensitivity_component": sensitivity_name,
            "source_model": (
                "Known-position characterization port of KILO "
                "optimize_characterization and spectro max-R primitives, "
                "wired to nullsim's sensitivity-stage physical floors. Rates "
                "are evaluated over the full wavelength grid using the deployed "
                "chromatic unitary; the optimizer may still use a smaller "
                "representative wavelength set."
            ),
        }
        payload.update(spectrograph_payload)
        results = state.results.add_component(component_name, payload)
        return replace(state, results=results)


def _snr_limited_max_spectral_resolution(
    *,
    params: SpectroCharacterization.Params,
    planet_signal_rate_hz: float,
    leakage_rate_hz: float,
    wl_design_m: float,
    band_bandwidth_m: float,
    integration_time_s: float,
    null_instability: float,
    servo_bw_hz: float,
    sky_rate_hz_per_spectro_port: float,
    pixels_per_resolution_element: int,
) -> float:
    # Rates are pre-spectrograph; the AWG's optical throughput applies to
    # every optical term, as in ``awg_spectrum``. Sky is summed over all
    # spectro ports to match the summed planet and leakage rates.
    return max_spectral_resolution(
        planet_rate_total_hz=planet_signal_rate_hz,
        leakage_rate_total_hz=leakage_rate_hz,
        spectro_throughput=(
            params.awg_throughput if params.spectrograph == "awg" else 1.0
        ),
        band_lambda_m=wl_design_m,
        band_bandwidth_m=band_bandwidth_m,
        integration_time_s=integration_time_s,
        dark_rate_hz_per_pixel=params.n_dark_count_hz,
        n_spectro_pixels=params.n_spectro_pixels,
        null_instability=null_instability,
        servo_bandwidth_hz=servo_bw_hz,
        snr_min=params.snr_min_per_bin,
        resolving_power_cap=params.resolving_power_cap,
        resolving_power_floor=params.resolving_power_floor,
        sky_rate_hz_per_spectro_port=(
            sky_rate_hz_per_spectro_port * params.n_spectro_ports
        ),
        pixels_per_resolution_element=pixels_per_resolution_element,
    )


def _servo_bw_hz(sensitivity: dict[str, Any], params: Any) -> float:
    """Systematic averaging bandwidth, or 0 (quasi-static) without fringe tracking.

    ``physics.spectroscopy`` treats a zero bandwidth as the unaveraged
    systematic branch, so every spectrograph SNR call honors
    ``with_fringe_tracking`` the way the headline ``_compute_snr`` does.
    """

    if not params.with_fringe_tracking:
        return 0.0
    return float(
        sensitivity.get("effective_systematic_bandwidth_hz")
        or params.default_servo_bw_hz
    )


def _spectro_sky_rate_hz(
    state: SimulationState,
    detector: dict[str, Any],
    configured_rate_hz_per_port: float,
) -> tuple[int, float, float, str]:
    """Return ``(n_pol_arms, sky per arm, sky per logical spectro port, source)``.

    Background stages and ``n_sky_background_hz`` give the sky per physical
    detector channel. A dual-pol pipeline feeds each logical spectro port
    from ``n_pol_arms`` PBS arms, each carrying that sky, while the spectro
    stellar and planet rates are arm totals; charge every arm, as
    ``performance.standard`` counts every dark channel.
    """

    sky_per_arm_hz, source = _resolve_background_rate_hz_per_port(
        state=state,
        configured_rate_hz_per_port=configured_rate_hz_per_port,
        port_kind="spectro",
    )
    n_pol_arms = max(int(detector.get("n_pol_arms") or 1), 1)
    return n_pol_arms, sky_per_arm_hz, sky_per_arm_hz * n_pol_arms, source


def _result_u_stack_or_build(
    *,
    result: Any,
    n_modes: int,
    wavelengths_m: NDArray[np.float64],
    wl_design_m: float,
) -> NDArray[np.complex128]:
    raw_stack = getattr(result, "optimal_u_stack", None)
    raw_wavelengths = getattr(result, "optimal_u_stack_wavelengths_m", None)
    if raw_stack is not None and raw_wavelengths is not None:
        stack = np.asarray(raw_stack, dtype=np.complex128)
        stack_wl = np.asarray(raw_wavelengths, dtype=np.float64)
        if stack.shape == (n_modes, n_modes, wavelengths_m.size) and np.allclose(
            stack_wl,
            wavelengths_m,
            rtol=0.0,
            atol=1.0e-15,
        ):
            return stack
    return build_unitary_stack_from_params(
        params=result.params,
        n_modes=n_modes,
        wavelengths_m=wavelengths_m,
        wl_design_m=wl_design_m,
        dac_bits=None,
    )


def _wavelength_indices(
    *,
    source_wavelengths_m: NDArray[np.float64],
    target_wavelengths_m: NDArray[np.float64],
) -> NDArray[np.intp]:
    source = np.asarray(source_wavelengths_m, dtype=np.float64)
    target = np.asarray(target_wavelengths_m, dtype=np.float64)
    indices = np.empty(target.size, dtype=np.intp)
    for index, wavelength_m in enumerate(target):
        source_index = int(np.argmin(np.abs(source - float(wavelength_m))))
        if not np.isclose(
            source[source_index],
            float(wavelength_m),
            rtol=0.0,
            atol=1.0e-15,
        ):
            raise ValueError(
                "characterization.spectro broadband wavelengths must be a "
                "subset of the deployed wavelength stack."
            )
        indices[index] = source_index
    return indices


def _mkid_spectrograph_payload(
    *,
    params: SpectroCharacterization.Params,
    detector: dict[str, Any],
    planet_signal_rate_hz: float,
    leakage_rate_hz: float,
    wl_design_m: float,
    band_bandwidth_m: float,
    integration_time_s: float,
    null_instability: float,
    servo_bw_hz: float,
    sky_rate_hz_per_spectro_port: float,
    snr_limited_max_r: float,
) -> tuple[float, dict[str, Any]]:
    mkid_r_ref = float(detector["resolving_power_ref"])
    mkid_lambda_ref_um = float(detector["lambda_ref_um"])
    mkid_r = float(
        mkid_resolving_power(
            np.array([wl_design_m * 1.0e6], dtype=np.float64),
            mkid_r_ref,
            mkid_lambda_ref_um,
        )[0]
    )
    mkid_snr_per_bin = per_bin_snr(
        planet_rate_total_hz=planet_signal_rate_hz,
        leakage_rate_total_hz=leakage_rate_hz,
        spectro_throughput=1.0,
        band_lambda_m=wl_design_m,
        band_bandwidth_m=band_bandwidth_m,
        resolving_power=mkid_r,
        integration_time_s=integration_time_s,
        dark_rate_hz_per_pixel=params.n_dark_count_hz,
        n_spectro_pixels=params.n_spectro_pixels,
        pixels_per_resolution_element=1,
        null_instability=null_instability,
        servo_bandwidth_hz=servo_bw_hz,
        sky_rate_hz_per_spectro_port=sky_rate_hz_per_spectro_port,
    )
    achievable_max_r = (
        min(float(snr_limited_max_r), mkid_r) if snr_limited_max_r > 0.0 else 0.0
    )
    # The current flat-spectrum MKID model yields one representative bin SNR.
    return achievable_max_r, {
        "spectrograph": "mkid",
        "mkid_snr_model": (
            "representative flat-spectrum bin using total band planet/leakage "
            "rates; does not propagate an MKID energy-response matrix"
        ),
        "max_spectral_resolution_snr_limited": float(snr_limited_max_r),
        "mkid_resolving_power": mkid_r,
        "mkid_resolving_power_ref": mkid_r_ref,
        "mkid_lambda_ref_um": mkid_lambda_ref_um,
        "mkid_pixels_per_resolution_element": 1,
        # Per-bin detector dark exactly as per_bin_snr charges it above
        # (dark_rate_hz_per_pixel * n_spectro_pixels). Published so downstream
        # consumers do not have to reconstruct it from the band-total D, which
        # scales with n_spectro_ports rather than n_spectro_pixels and
        # undercounts whenever n_spectro_pixels > 1. characterization.hdc_ccf
        # will need this once its phase-2 MKID path lands; it refuses MKID
        # payloads today.
        "mkid_n_spectro_pixels": int(params.n_spectro_pixels),
        "mkid_dark_rate_hz_per_bin_total": float(
            params.n_dark_count_hz * int(params.n_spectro_pixels)
        ),
        "mkid_snr_per_bin": float(mkid_snr_per_bin),
        "mkid_min_snr_per_bin": float(mkid_snr_per_bin),
    }


def _awg_spectrograph_payload(
    *,
    params: SpectroCharacterization.Params,
    planet_signal_rate_hz: float,
    leakage_rate_hz: float,
    wl_design_m: float,
    band_bandwidth_m: float,
    integration_time_s: float,
    null_instability: float,
    servo_bw_hz: float,
    sky_rate_hz_per_spectro_port: float,
    snr_limited_max_r: float,
) -> tuple[float, dict[str, Any]]:
    awg_resolving_power = (
        float(params.awg_resolving_power)
        if params.awg_resolving_power is not None
        else _DEFAULT_AWG_RESOLVING_POWER
    )
    awg = awg_spectrum(
        planet_rate_total_hz=planet_signal_rate_hz,
        leakage_rate_total_hz=leakage_rate_hz,
        sky_rate_total_hz=(sky_rate_hz_per_spectro_port * params.n_spectro_ports),
        throughput=params.awg_throughput,
        resolving_power=awg_resolving_power,
        band_lambda_m=wl_design_m,
        band_bandwidth_m=band_bandwidth_m,
        integration_time_s=integration_time_s,
        dark_rate_hz_per_pixel=params.n_dark_count_hz,
        n_detector_pixels_per_bin=(
            params.n_spectro_pixels * params.n_spectro_ports
        ),
        free_spectral_range_fraction=params.awg_free_spectral_range_fraction,
        order=params.awg_order,
        pixels_per_resolution_element=params.awg_pixels_per_resolution_element,
        null_instability=null_instability,
        servo_bandwidth_hz=servo_bw_hz,
    )
    awg_payload = _awg_payload(awg)
    return float(snr_limited_max_r), {
        "spectrograph": "awg",
        "max_spectral_resolution_snr_limited": float(snr_limited_max_r),
        "awg_resolving_power": float(awg.resolving_power),
        "awg_n_bins": int(awg.n_bins),
        "awg_min_snr_per_bin": float(awg_payload["min_snr_per_bin"]),
        "awg_mean_snr_per_bin": float(awg_payload["mean_snr_per_bin"]),
        "awg_total_planet_rate_hz": float(awg_payload["total_planet_rate_hz"]),
        "awg_total_leakage_rate_hz": float(awg_payload["total_leakage_rate_hz"]),
        "awg_total_sky_rate_hz": float(awg_payload["total_sky_rate_hz"]),
        "awg": awg_payload,
    }


def _find_sensitivity_component(state: SimulationState) -> tuple[str, dict[str, Any]]:
    for name, payload in state.results.components.items():
        if not name.startswith(_SENSITIVITY_PREFIX) or not isinstance(payload, dict):
            continue
        required = {
            "coupling_per_sub",
            "sub_areas_m2",
            "sub_diameters_m",
            "effective_null_with_disk",
            "wavelengths_m",
        }
        if required.issubset(payload):
            return name, payload
    raise ValueError(
        "characterization.spectro requires an upstream "
        "sensitivity.detection_curve component. Add the sensitivity stage before "
        "characterization so the spectro calculation reuses the same physical "
        "coupling, disk-floor, and realized-null budget."
    )


def _find_mkid_detector_component(
    state: SimulationState,
) -> tuple[str, dict[str, Any]]:
    missing_resolution: str | None = None
    for name, payload in state.results.components.items():
        if not name.startswith("detector.mkid") or not isinstance(payload, dict):
            continue
        required_base = {"port_rates", "port_labels", "integration_time_s"}
        if not required_base.issubset(payload):
            continue
        if (
            payload.get("resolving_power_ref") is not None
            and payload.get("lambda_ref_um") is not None
        ):
            return name, payload
        missing_resolution = name
    if missing_resolution is not None:
        raise ValueError(
            'characterization.spectro with spectrograph="mkid" requires the '
            f"upstream MKID detector component {missing_resolution!r} to be "
            "R-enabled. Set detector.mkid resolving_power_ref and lambda_ref_um."
        )
    raise ValueError(
        'characterization.spectro with spectrograph="mkid" requires an '
        "upstream detector.mkid stage with resolving_power_ref and lambda_ref_um."
    )


def _require_positions(
    state: SimulationState,
    n_modes: int,
) -> NDArray[np.float64]:
    if state.geometry.aperture_positions_pupil_m is None:
        raise ValueError(
            "characterization.spectro requires geometry.aperture_positions_pupil_m."
        )
    positions = np.asarray(state.geometry.aperture_positions_pupil_m, dtype=np.float64)
    if positions.shape != (n_modes, 2):
        raise ValueError(
            "characterization.spectro aperture_positions_pupil_m has shape "
            f"{positions.shape}; expected ({n_modes}, 2)."
        )
    # Project onto the sky plane so the known-position planet phase rows
    # foreshorten correctly off-zenith (inter-telescope component only when
    # per-mode origins are published; review 2026-06-10 Issue 20).
    return project_aperture_positions_for_pointing(
        positions,
        altitude_deg=state.scene.altitude_deg,
        azimuth_deg=state.scene.azimuth_deg,
        aperture_origins_m=mode_origin_positions_m(state.geometry, n_modes),
    )


def _payload_array(
    payload: dict[str, Any],
    key: str,
    n_modes: int,
    *,
    dtype: Any,
) -> NDArray[Any]:
    array = np.asarray(payload[key], dtype=dtype)
    if array.shape != (n_modes,):
        raise ValueError(
            f"characterization.spectro sensitivity payload {key!r} has shape "
            f"{array.shape}; expected ({n_modes},)."
        )
    return array


def _resolve_wavelengths_m(
    state: SimulationState,
    sensitivity: dict[str, Any],
) -> NDArray[np.float64]:
    if "wavelengths_m" in sensitivity:
        wavelengths = np.asarray(sensitivity["wavelengths_m"], dtype=np.float64)
    elif state.field.wavelength_grid_um:
        wavelengths = (
            np.asarray(state.field.wavelength_grid_um, dtype=np.float64) * 1e-6
        )
    else:
        raise ValueError("characterization.spectro requires a wavelength grid.")
    if wavelengths.ndim != 1 or wavelengths.size == 0:
        raise ValueError("characterization.spectro wavelength grid must be 1-D.")
    return wavelengths


def _characterization_wavelength_grid(
    wavelengths_m: NDArray[np.float64],
    n_wavelengths: int,
    *,
    center_wavelength_m: float | None = None,
) -> NDArray[np.float64]:
    wavelengths = np.asarray(wavelengths_m, dtype=np.float64)
    if wavelengths.ndim != 1 or wavelengths.size == 0:
        raise ValueError("characterization wavelength grid must be a non-empty 1-D array.")
    order = np.argsort(wavelengths)
    wavelengths = wavelengths[order]
    if center_wavelength_m is None:
        preferred_index = int(wavelengths.size // 2)
    else:
        preferred_index = int(
            np.argmin(np.abs(wavelengths - float(center_wavelength_m))),
        )
    if n_wavelengths <= 1 or wavelengths.size == 1:
        return np.array([float(wavelengths[preferred_index])])
    if int(n_wavelengths) >= wavelengths.size:
        return wavelengths.astype(np.float64, copy=True)
    indices = np.rint(
        np.linspace(0, wavelengths.size - 1, int(n_wavelengths)),
    ).astype(np.intp)
    indices = np.unique(indices)
    if preferred_index not in set(int(index) for index in indices):
        replace_at = int(np.argmin(np.abs(indices - preferred_index)))
        indices[replace_at] = preferred_index
        indices = np.unique(indices)
    return wavelengths[indices].astype(np.float64, copy=True)


def _require_scene_float(value: float | None, name: str) -> float:
    if value is None:
        raise ValueError(f"characterization.spectro requires {name}.")
    return float(value)


def _port_indices(
    params: SpectroCharacterization.Params,
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.intp]]:
    bright = np.arange(0, params.n_bright_ports, dtype=np.intp)
    dark = np.arange(
        params.n_bright_ports,
        params.n_bright_ports + params.n_dark_ports,
        dtype=np.intp,
    )
    spectro = np.arange(
        params.n_bright_ports + params.n_dark_ports,
        params.n_modes,
        dtype=np.intp,
    )
    return bright, dark, spectro


def _realized_spectro_floor(
    *,
    sensitivity: dict[str, Any],
    n_dark_ports: int,
    n_spectro_ports: int,
) -> float:
    """Return the upstream dynamic/systematic floor assigned to spectro ports.

    The characterization chip computes its own static stellar leakage
    (on-axis for an unresolved star, finite-disk integrated for a resolved
    star). Do not reuse ``effective_null_with_disk`` directly here: that
    scalar already includes the detection chip's static ideal/disk terms,
    which would double count stellar leakage after retuning.
    """

    realized_null = _optional_float(sensitivity.get("realized_null_depth"))
    ideal_null = _optional_float(sensitivity.get("ideal_chip_null_depth"))
    disk_floor = _optional_float(sensitivity.get("disk_null_floor")) or 0.0
    effective_null = _optional_float(sensitivity.get("effective_null_with_disk"))

    if realized_null is None:
        if effective_null is None:
            raise ValueError(
                "characterization.spectro requires sensitivity realized_null_depth "
                "or effective_null_with_disk."
            )
        dynamic_floor = max(effective_null - disk_floor, 0.0)
        realized_null_for_extra = dynamic_floor
    elif ideal_null is None:
        dynamic_floor = max(realized_null, 0.0)
        realized_null_for_extra = realized_null
    else:
        dynamic_floor = max(realized_null - ideal_null, 0.0)
        realized_null_for_extra = realized_null

    # Preserve explicit calibration/override floors that lift the effective
    # null above the physical realized+disk budget, while still removing the
    # detection chip's static disk contribution before retuned-chip leakage
    # is added below.
    if effective_null is not None:
        override_extra = max(
            effective_null - disk_floor - realized_null_for_extra,
            0.0,
        )
        dynamic_floor += override_extra

    # ``sensitivity.detection_curve`` reports realized leakage summed over
    # its dark-port set; spectro ports are not included in that upstream total.
    null_port_count = max(int(n_dark_ports), 1)
    spectro_port_count = max(int(n_spectro_ports), 0)
    return float(dynamic_floor * spectro_port_count / null_port_count)


def _realized_spectro_floor_for_stack(
    *,
    sensitivity: dict[str, Any],
    u_stack: NDArray[np.complex128],
    base_amplitudes: NDArray[np.float64],
    spectro_idx: NDArray[np.intp],
    wavelengths_m: NDArray[np.float64],
    stellar_rate_weights: NDArray[np.float64],
    n_dark_ports: int,
    n_spectro_ports: int,
) -> tuple[float, str, float | None]:
    """Return dynamic spectro leakage for the retuned characterization chip.

    Returns ``(floor, model, leak_rms)``. ``leak_rms`` is the RMS of the
    photon-weighted band leakage fraction of the retuned spectro ports, or
    ``None`` on the legacy path where the retuned MC cannot run.
    """

    retuned = _retuned_spectro_dynamic_floor(
        sensitivity=sensitivity,
        u_stack=u_stack,
        base_amplitudes=base_amplitudes,
        spectro_idx=spectro_idx,
        wavelengths_m=wavelengths_m,
        stellar_rate_weights=stellar_rate_weights,
    )
    override_extra = _spectro_override_extra(
        sensitivity=sensitivity,
        n_dark_ports=n_dark_ports,
        n_spectro_ports=n_spectro_ports,
    )
    if retuned is not None:
        retuned_floor, leak_rms = retuned
        return (
            float(retuned_floor + override_extra),
            "retuned spectro unitary realized-null MC using the upstream "
            "piston/scintillation covariance, with detection-mode static "
            "terms removed; photon-weighted over the band",
            leak_rms,
        )

    return (
        _realized_spectro_floor(
            sensitivity=sensitivity,
            n_dark_ports=n_dark_ports,
            n_spectro_ports=n_spectro_ports,
        ),
        "legacy upstream realized dynamic floor after removing detection-chip "
        "static terms, assigned to retuned spectro ports in proportion to "
        "their share of the detection dark-port set",
        None,
    )


def _spectro_null_instability(
    *,
    sensitivity: dict[str, Any],
    leak_rms: float | None,
    leak_fraction: float,
) -> tuple[float, str]:
    """Return the spectro port's leakage CV used as characterization epsilon.

    The CV is the retuned port's band-leakage RMS over its total (static plus
    dynamic) leak fraction, so ``eps * leakage_rate`` is the leakage-rate
    fluctuation in the same normalization as the rates. Without the retuned
    MC, fall back to the detection chip's published CV.
    """

    if leak_rms is None:
        return (
            float(sensitivity.get("null_instability", 0.0)),
            "sensitivity.detection_curve null_instability (retuned MC unavailable)",
        )
    cv = leak_rms / leak_fraction if leak_fraction > 0.0 else 0.0
    return float(cv), "retuned spectro port band-leakage RMS / spectro_leak_fraction"


def _mc_piston_opd_covariance_m2(
    sensitivity: dict[str, Any],
    n_modes: int,
) -> NDArray[np.float64] | None:
    """Per-mode piston covariance the retuned realized-floor MC draws from.

    ``None`` when that MC does not run, so planet and stellar leakage see the
    same piston (or none).
    """

    n_per_az = _optional_int(sensitivity.get("n_realizations_per_azimuth"))
    if n_per_az is None or n_per_az <= 0:
        return None
    return mode_piston_opd_covariance_m2(
        n_modes=n_modes,
        opd_rms_per_sub_m=_optional_array_or_scalar(
            sensitivity.get("ao_piston_opd_rms_per_sub_m"), default=0.0,
        ),
        opd_covariance_m2=_optional_matrix(
            sensitivity.get("ao_piston_opd_covariance_m2")
        ),
        mode_group_indices=_optional_index_array(
            sensitivity.get("realized_null_mode_group_indices"), n_modes=n_modes,
        ),
    )


def _retuned_spectro_dynamic_floor(
    *,
    sensitivity: dict[str, Any],
    u_stack: NDArray[np.complex128],
    base_amplitudes: NDArray[np.float64],
    spectro_idx: NDArray[np.intp],
    wavelengths_m: NDArray[np.float64],
    stellar_rate_weights: NDArray[np.float64],
) -> tuple[float, float] | None:
    """Return the retuned port's photon-weighted dynamic floor and leak RMS.

    Both use per-realization band leakage ``L_r = sum_k w_k N_kr`` with
    ``w`` the normalized detected stellar rate per bin: the floor is
    ``E[L] - L_static`` and the RMS is the within-azimuth standard deviation
    of ``L_r``. The MC shares one piston draw across wavelength bins, so the
    bins are paired realizations.
    """

    n_per_az = _optional_int(sensitivity.get("n_realizations_per_azimuth"))
    if n_per_az is None:
        return None

    n_modes = int(base_amplitudes.size)
    piston_covariance = _optional_matrix(
        sensitivity.get("ao_piston_opd_covariance_m2")
    )
    mode_group_indices = _optional_index_array(
        sensitivity.get("realized_null_mode_group_indices"),
        n_modes=n_modes,
    )
    mc = realized_null_depth_mc(
        u_stack=np.asarray(u_stack, dtype=np.complex128),
        stellar_field=np.asarray(base_amplitudes, dtype=np.complex128),
        dark_indices=np.asarray(spectro_idx, dtype=np.intp),
        ao_piston_opd_rms_per_sub_m=_optional_array_or_scalar(
            sensitivity.get("ao_piston_opd_rms_per_sub_m"),
            default=0.0,
        ),
        ao_piston_opd_covariance_m2=piston_covariance,
        wavelengths_m=np.asarray(wavelengths_m, dtype=np.float64),
        n_realizations_per_azimuth=n_per_az,
        seed=_optional_int(sensitivity.get("mc_seed")),
        scintillation_index_per_sub=_optional_array_or_scalar(
            sensitivity.get("scintillation_index_per_sub"),
            default=0.0,
        ),
        mode_group_indices=mode_group_indices,
        fiber_phase_noise_rms_rad=float(
            _optional_float(sensitivity.get("fiber_phase_noise_rms_rad")) or 0.0
        ),
    )
    if mc.samples_by_wavelength is None:  # MC disabled: no dynamic term
        return 0.0, 0.0
    weights = np.asarray(stellar_rate_weights, dtype=np.float64)
    weights = weights / np.sum(weights)
    out = np.einsum("pmk,m->kp", u_stack, base_amplitudes.astype(np.complex128))
    power = np.abs(out) ** 2
    ideal_per_wl = np.sum(power[:, spectro_idx], axis=1) / np.sum(power, axis=1)
    samples = mc.samples_by_wavelength.reshape(
        wavelengths_m.size, mc.n_azimuths, mc.n_realizations_per_azimuth,
    )
    band_leak = np.einsum("k,kar->ar", weights, samples)
    floor = max(float(np.mean(band_leak) - weights @ ideal_per_wl), 0.0)
    leak_rms = math.sqrt(float(np.mean(np.var(band_leak, axis=1))))
    return floor, leak_rms


def _spectro_override_extra(
    *,
    sensitivity: dict[str, Any],
    n_dark_ports: int,
    n_spectro_ports: int,
) -> float:
    override_extra = _detection_floor_override_extra(sensitivity)
    if override_extra <= 0.0:
        return 0.0
    null_port_count = max(int(n_dark_ports), 1)
    spectro_port_count = max(int(n_spectro_ports), 0)
    return float(override_extra * spectro_port_count / null_port_count)


def _detection_floor_override_extra(sensitivity: dict[str, Any]) -> float:
    realized_null = _optional_float(sensitivity.get("realized_null_depth"))
    disk_floor = _optional_float(sensitivity.get("disk_null_floor")) or 0.0
    effective_null = _optional_float(sensitivity.get("effective_null_with_disk"))
    if effective_null is None or realized_null is None:
        return 0.0
    return max(effective_null - disk_floor - realized_null, 0.0)


def _static_spectro_leak_fraction(
    *,
    on_axis_null: float,
    disk_leak: float,
    stellar_angular_diameter_mas: float | None,
) -> float:
    """Return static stellar leakage for the retuned characterization chip."""

    if (
        stellar_angular_diameter_mas is not None
        and float(stellar_angular_diameter_mas) > 0.0
    ):
        return float(disk_leak)
    return max(float(on_axis_null), float(disk_leak))


def _band_evaluated_spectro_metrics(
    *,
    u_stack: NDArray[np.complex128],
    wavelengths_m: NDArray[np.float64],
    detected_stellar_rate_by_wavelength_hz: NDArray[np.float64],
    positions_m: NDArray[np.float64],
    coupling_per_sub: NDArray[np.float64],
    sub_areas_m2: NDArray[np.float64],
    sub_diameters_m: NDArray[np.float64],
    base_amplitudes: NDArray[np.float64],
    spectro_idx: NDArray[np.intp],
    separation_rad: float,
    position_angle_rad: float,
    planet_contrast: float,
    disk_leak_per_wavelength: NDArray[np.float64],
    stellar_angular_diameter_mas: float | None,
    piston_opd_covariance_m2: NDArray[np.float64] | None = None,
    fiber_phase_noise_rms_rad: float = 0.0,
) -> dict[str, Any]:
    """Evaluate deployed spectro leakage and planet signal over the band.

    With ``piston_opd_covariance_m2`` (the realized-floor MC's per-mode
    covariance) the planet throughput is the piston expectation
    ``E|U_s(X exp(i phi))|^2`` with phase covariance ``k^2 C + sigma_f^2 I``,
    ``k = 2 pi / lambda``, matching the piston the stellar floor carries.
    """

    expected_shape = (base_amplitudes.size, base_amplitudes.size, wavelengths_m.size)
    if u_stack.shape != expected_shape:
        raise ValueError("characterization.spectro u_stack shape mismatch.")
    if detected_stellar_rate_by_wavelength_hz.shape != wavelengths_m.shape:
        raise ValueError(
            "characterization.spectro detected stellar-rate grid mismatch."
        )
    if disk_leak_per_wavelength.shape != wavelengths_m.shape:
        raise ValueError("characterization.spectro disk-leak grid mismatch.")

    total_stellar_rate_hz = float(np.sum(detected_stellar_rate_by_wavelength_hz))
    if total_stellar_rate_hz <= 0.0 or not math.isfinite(total_stellar_rate_hz):
        raise ValueError("characterization.spectro detected stellar rate is invalid.")

    on_axis_total = float(np.sum(np.abs(base_amplitudes) ** 2))
    if on_axis_total <= 0.0:
        raise ValueError("characterization.spectro on-axis field has zero power.")
    aperture_weight = coupling_per_sub * sub_areas_m2
    aperture_weight_sum = float(np.sum(aperture_weight))
    if aperture_weight_sum <= 0.0:
        raise ValueError("characterization.spectro aperture weights sum to zero.")

    on_axis_null = np.zeros(wavelengths_m.size, dtype=np.float64)
    static_leak = np.zeros(wavelengths_m.size, dtype=np.float64)
    planet_throughput = np.zeros(wavelengths_m.size, dtype=np.float64)
    planet_to_star_ratio = np.zeros(wavelengths_m.size, dtype=np.float64)
    planet_rate_by_wavelength = np.zeros(wavelengths_m.size, dtype=np.float64)
    n_modes = base_amplitudes.size
    if piston_opd_covariance_m2 is None:
        opd_covariance = np.zeros((n_modes, n_modes), dtype=np.float64)
        phase_noise_covariance = opd_covariance
    else:
        opd_covariance = np.asarray(piston_opd_covariance_m2, dtype=np.float64)
        phase_noise_covariance = fiber_phase_noise_rms_rad**2 * np.eye(n_modes)

    for index, wavelength_m in enumerate(wavelengths_m):
        unitary = u_stack[:, :, index]
        on_axis_output = unitary @ base_amplitudes.astype(np.complex128)
        on_axis_null[index] = (
            float(np.sum(np.abs(on_axis_output[spectro_idx]) ** 2)) / on_axis_total
        )
        static_leak[index] = _static_spectro_leak_fraction(
            on_axis_null=float(on_axis_null[index]),
            disk_leak=float(disk_leak_per_wavelength[index]),
            stellar_angular_diameter_mas=stellar_angular_diameter_mas,
        )
        planet_field = planet_input_field(
            positions_m=positions_m,
            coupling_per_sub=coupling_per_sub,
            sub_areas_m2=sub_areas_m2,
            sub_diameters_m=sub_diameters_m,
            wavelength_m=float(wavelength_m),
            separation_rad=separation_rad,
            position_angle_rad=position_angle_rad,
        )
        planet_total = float(np.sum(np.abs(planet_field) ** 2))
        if planet_total > 0.0:
            k = 2.0 * math.pi / float(wavelength_m)
            phase_covariance = k * k * opd_covariance + phase_noise_covariance
            planet_throughput[index] = piston_averaged_port_power(
                unitary=unitary,
                field=planet_field,
                port_indices=spectro_idx,
                phase_covariance_rad2=phase_covariance,
            ) / planet_total
        ang_power_planet = np.asarray(
            angular_coupling_factor(separation_rad, sub_diameters_m, wavelength_m),
            dtype=np.float64,
        )
        planet_to_star_ratio[index] = (
            float(np.sum(aperture_weight * ang_power_planet)) / aperture_weight_sum
        )
        planet_rate_by_wavelength[index] = (
            float(planet_contrast)
            * float(detected_stellar_rate_by_wavelength_hz[index])
            * planet_throughput[index]
            * planet_to_star_ratio[index]
        )

    weights = detected_stellar_rate_by_wavelength_hz / total_stellar_rate_hz
    return {
        "static_leak_fraction": float(np.sum(weights * static_leak)),
        "on_axis_null_depth": float(np.sum(weights * on_axis_null)),
        "planet_signal_rate_hz": float(np.sum(planet_rate_by_wavelength)),
        "on_axis_null_per_wavelength": on_axis_null,
        "static_leak_fraction_per_wavelength": static_leak,
        "planet_throughput_per_wavelength": planet_throughput,
        "planet_to_star_ratio_per_wavelength": planet_to_star_ratio,
        "planet_rate_by_wavelength_hz": planet_rate_by_wavelength,
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_matrix(value: Any) -> NDArray[np.float64] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        return None
    return array


def _optional_index_array(
    value: Any,
    *,
    n_modes: int,
) -> NDArray[np.intp] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.intp)
    if array.shape != (n_modes,):
        return None
    return array


def _optional_array_or_scalar(value: Any, *, default: float) -> Any:
    if value is None:
        return float(default)
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return float(array)
    return array


def _representative_rate_weights(
    *,
    representative_wavelengths_m: NDArray[np.float64],
    full_wavelengths_m: NDArray[np.float64],
    full_rate_by_wavelength_hz: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Interpolate full-grid photon rates onto optimizer wavelengths."""

    representatives = np.asarray(representative_wavelengths_m, dtype=np.float64)
    wavelengths = np.asarray(full_wavelengths_m, dtype=np.float64)
    rates = np.asarray(full_rate_by_wavelength_hz, dtype=np.float64)
    if representatives.ndim != 1 or wavelengths.ndim != 1 or rates.ndim != 1:
        raise ValueError("representative rate weights require 1-D arrays.")
    if wavelengths.shape != rates.shape:
        raise ValueError("full wavelength and rate arrays must have matching shape.")
    if representatives.size == 0 or wavelengths.size == 0:
        raise ValueError("representative rate weights require non-empty grids.")
    order = np.argsort(wavelengths)
    ordered_wavelengths = wavelengths[order]
    ordered_rates = rates[order]
    return np.interp(
        representatives,
        ordered_wavelengths,
        ordered_rates,
        left=float(ordered_rates[0]),
        right=float(ordered_rates[-1]),
    ).astype(np.float64, copy=False)


def _awg_payload(awg: AWGSpectrum) -> dict[str, Any]:
    snr_per_bin = awg.snr_per_bin.astype(np.float64, copy=True)
    if snr_per_bin.size:
        min_snr = float(np.min(snr_per_bin))
        mean_snr = float(np.mean(snr_per_bin))
        dark_rate_hz_per_bin = float(awg.dark_rate_hz_per_bin[0])
    else:
        min_snr = 0.0
        mean_snr = 0.0
        dark_rate_hz_per_bin = 0.0
    return {
        "resolving_power": float(awg.resolving_power),
        "throughput": float(awg.throughput),
        "free_spectral_range_fraction": float(awg.free_spectral_range_fraction),
        "order": int(awg.order),
        "pixels_per_resolution_element": int(awg.pixels_per_resolution_element),
        "n_detector_pixels_per_bin": int(awg.n_detector_pixels_per_bin),
        "n_bins": int(awg.n_bins),
        "bin_edges_m": awg.bin_edges_m.astype(np.float64, copy=True),
        "bin_centers_m": awg.bin_centers_m.astype(np.float64, copy=True),
        "planet_rate_hz_per_bin": awg.planet_rate_hz_per_bin.astype(
            np.float64,
            copy=True,
        ),
        "leakage_rate_hz_per_bin": awg.leakage_rate_hz_per_bin.astype(
            np.float64,
            copy=True,
        ),
        "sky_rate_hz_per_bin": awg.sky_rate_hz_per_bin.astype(
            np.float64,
            copy=True,
        ),
        "dark_rate_hz_per_bin": awg.dark_rate_hz_per_bin.astype(
            np.float64,
            copy=True,
        ),
        "snr_per_bin": snr_per_bin,
        "min_snr_per_bin": min_snr,
        "mean_snr_per_bin": mean_snr,
        "total_planet_rate_hz": float(np.sum(awg.planet_rate_hz_per_bin)),
        "total_leakage_rate_hz": float(np.sum(awg.leakage_rate_hz_per_bin)),
        "total_sky_rate_hz": float(np.sum(awg.sky_rate_hz_per_bin)),
        "dark_rate_hz_per_bin_total": dark_rate_hz_per_bin,
        "total_dark_rate_hz_across_bins": float(np.sum(awg.dark_rate_hz_per_bin)),
    }


def _initial_params(
    state: SimulationState,
    params: SpectroCharacterization.Params,
) -> tuple[NDArray[np.float64], ...]:
    if not params.use_upstream_chip_params_start:
        return ()
    components = state.results.components
    if params.optimization_source.lower() == "auto":
        candidates = [
            payload for name, payload in components.items()
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
                "characterization.spectro optimization_source "
                f"{params.optimization_source!r} did not resolve to a "
                "chip_optimization component with optimal_chip_params."
            )
        raw = payload["optimal_chip_params"]
    return (np.asarray(raw, dtype=np.float64),)


def _detected_stellar_rate_by_wavelength_hz(
    *,
    state: SimulationState,
    detector_component: dict[str, Any],
    wavelength_grid_um: tuple[float, ...],
    band: str,
    band_bandwidth_um_override: float | None = None,
) -> NDArray[np.float64]:
    if state.scene.star_jmag is None:
        raise ValueError("characterization.spectro requires scene.star_jmag.")
    area_m2 = _resolve_collecting_area_m2(state)
    bandwidth_um = _bandwidth_per_bin_um(
        wavelength_grid_um,
        total_band_bandwidth_um=band_bandwidth_um_override,
    )
    rate_per_m2 = np.array(
        [
            photon_rate_per_m2_per_bin(
                float(state.scene.star_jmag),
                float(wavelength_um),
                float(bin_width_um),
                band=band,
            )
            for wavelength_um, bin_width_um in zip(wavelength_grid_um, bandwidth_um)
        ],
        dtype=np.float64,
    )
    throughput = _throughput_or_unity(state, len(wavelength_grid_um))
    detector_qe = float(detector_component.get("detector_qe", 1.0))
    qe_factor = 1.0 if _throughput_includes_detector_qe(state) else detector_qe
    return (rate_per_m2 * throughput * area_m2 * qe_factor).astype(
        np.float64,
        copy=False,
    )


def _resolve_collecting_area_m2(state: SimulationState) -> float:
    if state.geometry.effective_collecting_area_m2 is not None:
        return float(state.geometry.effective_collecting_area_m2)
    if state.geometry.collecting_area_m2 is not None:
        return float(state.geometry.collecting_area_m2)
    if state.geometry.area_m2 is None:
        raise ValueError("characterization.spectro requires geometry area.")
    if state.geometry.n_apertures is None:
        return float(state.geometry.area_m2)
    return float(state.geometry.area_m2) * int(state.geometry.n_apertures)


def _throughput_or_unity(
    state: SimulationState,
    n_bins: int,
) -> NDArray[np.float64]:
    throughput = (
        np.asarray(state.throughput.transmission_fraction, dtype=np.float64)
        if state.throughput.transmission_fraction
        else np.ones(n_bins, dtype=np.float64)
    )
    if throughput.shape != (n_bins,):
        raise ValueError(
            "characterization.spectro throughput length does not match wavelength grid."
        )
    for name, payload in state.throughput.components.items():
        if not isinstance(payload, Mapping):
            continue
        if not _is_field_applied_throughput(payload):
            continue
        component_fraction = payload.get("transmission_fraction")
        if component_fraction is None:
            continue
        component = np.asarray(component_fraction, dtype=np.float64)
        if component.shape != (n_bins,):
            raise ValueError(
                "characterization.spectro field-applied throughput component "
                f"{name!r} length {component.size} does not match wavelength grid "
                f"length {n_bins}."
            )
        throughput = throughput * component
    return throughput


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


def _throughput_includes_detector_qe(state: SimulationState) -> bool:
    return any(name.startswith("detector.mkid_qe") for name in state.throughput.components)


def _bandwidth_per_bin_um(
    wavelength_grid_um: tuple[float, ...],
    total_band_bandwidth_um: float | None = None,
) -> NDArray[np.float64]:
    wavelengths = np.asarray(wavelength_grid_um, dtype=np.float64)
    if wavelengths.size == 1:
        if total_band_bandwidth_um is None:
            raise ValueError(
                "characterization.spectro requires band_bandwidth_um when the "
                "wavelength grid has a single bin (cannot infer bin width)."
            )
        return np.array([float(total_band_bandwidth_um)], dtype=np.float64)
    edges = np.empty(wavelengths.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (wavelengths[:-1] + wavelengths[1:])
    edges[0] = wavelengths[0] - 0.5 * (wavelengths[1] - wavelengths[0])
    edges[-1] = wavelengths[-1] + 0.5 * (wavelengths[-1] - wavelengths[-2])
    return np.diff(edges)


def _resolve_band_bandwidth_m(
    wavelengths_m: NDArray[np.float64],
    override_um: float | None,
) -> float:
    if override_um is not None:
        return float(override_um) * 1.0e-6
    widths_um = _bandwidth_per_bin_um(
        tuple(float(w * 1.0e6) for w in wavelengths_m),
        total_band_bandwidth_um=None,
    )
    return float(np.sum(widths_um) * 1.0e-6)
