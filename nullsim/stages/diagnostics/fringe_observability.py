"""Diagnose bright-port observability of differential piston modes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field as PydanticField

from nullsim.physics.fringe_observability import (
    active_coordinate_acquisition_monte_carlo,
    amplitude_postprocessing_monte_carlo,
    cophasing_handoff_monte_carlo,
    cramer_rao_bound_from_svd,
    differential_piston_basis,
    gauss_newton_reconstruction_monte_carlo,
    linearized_reconstruction_monte_carlo,
    piston_measurement_covariance_from_svd,
    reconstruction_monte_carlo,
    spectral_channel_fractional_intensity_response_matrix,
    spectral_channel_count_response_matrix,
    svd_observability,
    weakest_piston_modes,
    whiten_response_matrix,
)
from nullsim.pipeline import BaseStage, SimulationState, register_stage
from nullsim.stages._mode_splitter import splitter_physical_indices
from nullsim.stages._utils import detector_n_pol_arms, unique_component_name

_CHIP_PREFIX = "chip.kernel_mzi_mesh"
_FT_PREFIX = "fringe_tracking.closed_loop"
_M_TO_NM = 1.0e9


@dataclass(frozen=True)
class _PistonBasisSet:
    """Physical and chip-input piston bases for a deployed chip."""

    physical_basis: NDArray[np.float64]
    chip_basis: NDArray[np.float64]
    n_physical_modes: int
    positions_pupil_m: NDArray[np.float64] | None
    source: str


@register_stage
class FringeObservability(BaseStage):
    """Measure whether bright-port counts span the piston subspace."""

    type_name = "diagnostics.fringe_observability"
    family = "diagnostics"
    consumes = frozenset({"field", "photon_rates", "results", "geometry"})
    produces = frozenset({"results"})

    class Params(BaseModel):
        """Config for the piston-observability diagnostic."""

        model_config = ConfigDict(
            extra="forbid",
            frozen=True,
            allow_inf_nan=False,
        )

        bright_port_prefix: str = "bright"
        n_spectral_channels: int | None = PydanticField(default=None, ge=1)
        frame_time_s: float | None = PydanticField(default=None, gt=0.0)
        amplitude_frame_time_s: float | None = PydanticField(default=None, gt=0.0)
        amplitude_servo_bandwidth_hz: float | None = PydanticField(
            default=None,
            gt=0.0,
        )
        amplitude_estimator_port_set: Literal["bright", "bright_dark", "all"] = (
            "bright"
        )
        amplitude_postprocessing_n_monte_carlo: int = PydanticField(
            default=0,
            ge=0,
            le=20_000,
        )
        amplitude_postprocessing_sigma_i: float | None = PydanticField(
            default=None,
            ge=0.0,
        )
        amplitude_postprocessing_ideal_tap_counts_per_mode: float | None = (
            PydanticField(default=None, ge=0.0)
        )
        tracker_qe: float | None = PydanticField(default=None, gt=0.0, le=1.0)
        tracker_count_rate_cap_hz_per_channel: float | None = PydanticField(
            default=None,
            gt=0.0,
        )
        count_rate_source: Literal["auto", "photon_rates", "detector_uncapped"] = (
            "auto"
        )
        dark_rate_hz_per_channel: float = PydanticField(default=0.0, ge=0.0)
        relative_singular_threshold: float = PydanticField(
            default=1.0e-6,
            gt=0.0,
            lt=1.0,
        )
        absolute_singular_threshold: float = PydanticField(default=0.0, ge=0.0)
        noise_floor_counts: float = PydanticField(default=1.0, ge=0.0)
        n_monte_carlo: int = PydanticField(default=256, ge=0, le=20_000)
        piston_rms_nm_for_mc: float = PydanticField(default=50.0, ge=0.0)
        mc_seed: int = 0
        shot_noise: bool = True
        gauss_newton_max_iter: int = PydanticField(default=8, ge=0, le=100)
        gauss_newton_damping: float = PydanticField(default=1.0e-3, ge=0.0)
        gauss_newton_step_tolerance_nm: float = PydanticField(default=1.0e-3, ge=0.0)
        gauss_newton_cost_relative_tolerance: float = PydanticField(
            default=1.0e-6,
            ge=0.0,
        )
        gauss_newton_initialization: Literal["linear", "zero"] = "linear"
        gauss_newton_piston_rms_grid_nm: tuple[float, ...] = ()
        gauss_newton_sweep_n_monte_carlo: int = PydanticField(
            default=64,
            ge=0,
            le=20_000,
        )
        active_acquisition_n_monte_carlo: int = PydanticField(
            default=0,
            ge=0,
            le=20_000,
        )
        active_acquisition_n_passes: int = PydanticField(default=3, ge=1, le=20)
        active_acquisition_response_iterations: int = PydanticField(
            default=0,
            ge=0,
            le=50,
        )
        active_acquisition_probe_step_nm: float = PydanticField(
            default=10.0,
            gt=0.0,
        )
        active_acquisition_max_response_step_nm: float = PydanticField(
            default=80.0,
            gt=0.0,
        )
        active_acquisition_scan_half_range_nm: float = PydanticField(
            default=150.0,
            gt=0.0,
        )
        active_acquisition_scan_steps: int = PydanticField(
            default=61,
            ge=3,
            le=1001,
        )
        active_acquisition_reference_mode: int = PydanticField(default=0, ge=0)
        active_acquisition_lock_threshold_nm: float = PydanticField(
            default=10.0,
            ge=0.0,
        )
        active_acquisition_calibration_frames: int = PydanticField(
            default=1,
            ge=1,
            le=10_000,
        )
        cophasing_handoff_n_monte_carlo: int = PydanticField(
            default=0,
            ge=0,
            le=20_000,
        )
        cophasing_edges: tuple[tuple[int, int], ...] = ()
        cophasing_tracker_fraction: float = PydanticField(
            default=1.0,
            gt=0.0,
            le=1.0,
        )
        cophasing_visibility: float = PydanticField(
            default=1.0,
            gt=0.0,
            le=1.0,
        )
        cophasing_edge_throughput: float = PydanticField(
            default=1.0,
            gt=0.0,
            le=1.0,
        )
        cophasing_phase_offsets_rad: tuple[float, float, float, float] = (
            0.0,
            0.5 * np.pi,
            np.pi,
            1.5 * np.pi,
        )
        cophasing_edge_count_weights: tuple[float, ...] = ()
        cophasing_science_drift_rms_nm: float = PydanticField(default=2.0, ge=0.0)
        cophasing_lock_threshold_nm: float = PydanticField(default=10.0, ge=0.0)
        max_weak_modes: int = PydanticField(default=4, ge=0, le=12)
        include_response_matrix: bool = True

    def apply(self, state: SimulationState) -> SimulationState:
        params = self.params_model
        chip_payload = _find_component(state, _CHIP_PREFIX, self.type_name)
        u_stack = _require_u_stack(chip_payload, self.type_name)
        n_ports, n_modes, n_wavelengths = u_stack.shape
        wavelengths_um = _require_wavelength_grid_um(
            state,
            chip_payload,
            n_wavelengths,
            self.type_name,
        )
        wavelengths_m = np.asarray(wavelengths_um, dtype=np.float64) * 1.0e-6

        input_fields = _recover_chip_input_fields(state, u_stack, self.type_name)
        port_labels = _require_port_labels(state, n_ports, self.type_name)
        bright_indices, bright_labels = _bright_port_indices(
            port_labels,
            params.bright_port_prefix,
            self.type_name,
        )
        amplitude_indices, amplitude_labels = _amplitude_port_indices(
            port_labels,
            bright_indices=bright_indices,
            port_set=params.amplitude_estimator_port_set,
            stage_name=self.type_name,
        )
        wavelength_indices, wavelength_groups = _select_wavelength_channels(
            n_wavelengths,
            params.n_spectral_channels,
        )
        frame_time_s, frame_time_source = _frame_time_s(
            state,
            params.frame_time_s,
            self.type_name,
        )
        (
            signal_counts,
            variance_counts,
            background_counts,
            signal_counts_by_wavelength,
            background_counts_by_wavelength,
            count_metadata,
        ) = _expected_count_budget(
            state=state,
            port_indices=bright_indices,
            wavelength_groups=wavelength_groups,
            frame_time_s=frame_time_s,
            params=params,
            stage_name=self.type_name,
        )
        amplitude_frame_time_s, amplitude_frame_time_source = _amplitude_frame_time_s(
            state=state,
            params=params,
            piston_frame_time_s=frame_time_s,
            stage_name=self.type_name,
        )
        (
            amplitude_signal_counts,
            amplitude_variance_counts,
            _amplitude_background_counts,
            amplitude_signal_counts_by_wavelength,
            _amplitude_background_counts_by_wavelength,
            _amplitude_count_metadata,
        ) = _expected_count_budget(
            state=state,
            port_indices=amplitude_indices,
            wavelength_groups=wavelength_groups,
            frame_time_s=amplitude_frame_time_s,
            params=params,
            stage_name=self.type_name,
        )

        basis_set = _piston_basis_set(state, n_modes)
        response, projected_counts, whitened, svd, crlb = _linear_observability(
            u_stack=u_stack,
            input_fields=input_fields,
            wavelengths_m=wavelengths_m,
            bright_indices=bright_indices,
            wavelength_indices=wavelength_indices,
            wavelength_groups=wavelength_groups,
            piston_basis=basis_set.chip_basis,
            n_piston_modes=basis_set.n_physical_modes,
            signal_counts=signal_counts,
            signal_counts_by_wavelength=signal_counts_by_wavelength,
            variance_counts=variance_counts,
            params=params,
        )
        (
            amplitude_response,
            amplitude_projected_counts,
            amplitude_whitened,
            amplitude_svd,
            amplitude_crlb,
        ) = _amplitude_observability(
            u_stack=u_stack,
            input_fields=input_fields,
            wavelengths_m=wavelengths_m,
            port_indices=amplitude_indices,
            wavelength_groups=wavelength_groups,
            amplitude_basis=basis_set.chip_basis,
            n_amplitude_modes=basis_set.n_physical_modes,
            signal_counts=amplitude_signal_counts,
            signal_counts_by_wavelength=amplitude_signal_counts_by_wavelength,
            variance_counts=amplitude_variance_counts,
            params=params,
        )
        scintillation_assumptions = _scintillation_assumptions_payload(
            state=state,
            sigma_i_override=params.amplitude_postprocessing_sigma_i,
            amplitude_frame_time_s=amplitude_frame_time_s,
        )
        amplitude_postprocessing = _amplitude_postprocessing_payload(
            whitened=amplitude_whitened,
            basis_set=basis_set,
            scintillation_assumptions=scintillation_assumptions,
            amplitude_frame_time_s=amplitude_frame_time_s,
            estimator_port_set=params.amplitude_estimator_port_set,
            params=params,
        )
        weak_modes = weakest_piston_modes(
            right_singular_vectors=svd.right_singular_vectors,
            piston_basis=basis_set.physical_basis,
            n_weak_modes=int(params.max_weak_modes),
        )
        piston_covariance = piston_measurement_covariance_from_svd(
            svd,
            piston_basis=basis_set.physical_basis,
        )
        channel_sweep = _channel_sweep_payload(
            state=state,
            u_stack=u_stack,
            input_fields=input_fields,
            wavelengths_m=wavelengths_m,
            bright_indices=bright_indices,
            piston_basis=basis_set.chip_basis,
            n_piston_modes=basis_set.n_physical_modes,
            frame_time_s=frame_time_s,
            n_wavelengths=n_wavelengths,
            # Sweep up to the full wavelength-bin count (capped at 8) even
            # when the configured tracker has fewer channels: with
            # n_spectral_channels=1 the "spectral-diversity sweep" panel
            # otherwise degenerated to a single point and could not answer
            # whether 2-4 channels would lift the rank (review 2026-06-10
            # Issue 30c). The configured operating point stays marked via
            # n_spectral_channels in the payload.
            max_channels=max(int(wavelength_indices.size), min(n_wavelengths, 8)),
            params=params,
            stage_name=self.type_name,
        )
        # Per-estimator child seeds: the previous shared seed produced
        # streams that were neither matched piston realizations nor
        # independent (different per-trial draw counts), so estimator
        # comparisons mixed both regimes (review 2026-06-10 Issue 30e).
        linear_mc = linearized_reconstruction_monte_carlo(
            whitened_response_per_m=whitened,
            piston_basis=basis_set.physical_basis,
            piston_rms_m=float(params.piston_rms_nm_for_mc) / _M_TO_NM,
            n_trials=int(params.n_monte_carlo),
            rng=np.random.default_rng(int(params.mc_seed)),
            relative_threshold=float(params.relative_singular_threshold),
        )
        nonlinear_mc = reconstruction_monte_carlo(
            u_stack=u_stack,
            input_fields=input_fields,
            wavelengths_m=wavelengths_m,
            port_indices=bright_indices,
            wavelength_indices=wavelength_indices,
            piston_basis=basis_set.physical_basis,
            reference_power=response.reference_power,
            expected_counts=signal_counts,
            variance_counts=variance_counts,
            background_counts=background_counts,
            whitened_response_per_m=whitened,
            wavelength_groups=wavelength_groups,
            expected_counts_by_wavelength=signal_counts_by_wavelength,
            background_counts_by_wavelength=background_counts_by_wavelength,
            piston_rms_m=float(params.piston_rms_nm_for_mc) / _M_TO_NM,
            n_trials=int(params.n_monte_carlo),
            rng=np.random.default_rng(int(params.mc_seed) + 1),
            field_piston_basis=basis_set.chip_basis,
            relative_threshold=float(params.relative_singular_threshold),
            noise_floor_counts=float(params.noise_floor_counts),
            shot_noise=bool(params.shot_noise),
        )
        gauss_newton_mc = gauss_newton_reconstruction_monte_carlo(
            u_stack=u_stack,
            input_fields=input_fields,
            wavelengths_m=wavelengths_m,
            port_indices=bright_indices,
            wavelength_indices=wavelength_indices,
            piston_basis=basis_set.physical_basis,
            reference_power=response.reference_power,
            expected_counts=signal_counts,
            variance_counts=variance_counts,
            background_counts=background_counts,
            whitened_response_per_m=whitened,
            wavelength_groups=wavelength_groups,
            expected_counts_by_wavelength=signal_counts_by_wavelength,
            background_counts_by_wavelength=background_counts_by_wavelength,
            piston_rms_m=float(params.piston_rms_nm_for_mc) / _M_TO_NM,
            n_trials=int(params.n_monte_carlo),
            rng=np.random.default_rng(int(params.mc_seed) + 2),
            field_piston_basis=basis_set.chip_basis,
            relative_threshold=float(params.relative_singular_threshold),
            shot_noise=bool(params.shot_noise),
            max_iter=int(params.gauss_newton_max_iter),
            damping=float(params.gauss_newton_damping),
            step_tolerance_m=float(params.gauss_newton_step_tolerance_nm) / _M_TO_NM,
            cost_relative_tolerance=float(params.gauss_newton_cost_relative_tolerance),
            initialization=params.gauss_newton_initialization,
            noise_floor_counts=float(params.noise_floor_counts),
        )
        gauss_newton_sweep = _gauss_newton_capture_sweep_payload(
            u_stack=u_stack,
            input_fields=input_fields,
            wavelengths_m=wavelengths_m,
            bright_indices=bright_indices,
            wavelength_indices=wavelength_indices,
            wavelength_groups=wavelength_groups,
            basis_set=basis_set,
            reference_power=response.reference_power,
            signal_counts=signal_counts,
            signal_counts_by_wavelength=signal_counts_by_wavelength,
            variance_counts=variance_counts,
            background_counts=background_counts,
            background_counts_by_wavelength=background_counts_by_wavelength,
            whitened=whitened,
            params=params,
        )
        active_acquisition_mc = _active_acquisition_monte_carlo(
            u_stack=u_stack,
            input_fields=input_fields,
            wavelengths_m=wavelengths_m,
            bright_indices=bright_indices,
            wavelength_indices=wavelength_indices,
            wavelength_groups=wavelength_groups,
            basis_set=basis_set,
            reference_power=response.reference_power,
            signal_counts=signal_counts,
            signal_counts_by_wavelength=signal_counts_by_wavelength,
            variance_counts=variance_counts,
            background_counts=background_counts,
            background_counts_by_wavelength=background_counts_by_wavelength,
            params=params,
        )
        (
            cophasing_counts,
            cophasing_counts_by_wavelength,
            cophasing_dark_counts,
            cophasing_count_cap,
            cophasing_metadata,
        ) = _cophasing_count_budget(
            state=state,
            wavelength_groups=wavelength_groups,
            frame_time_s=frame_time_s,
            params=params,
            stage_name=self.type_name,
        )
        cophasing_handoff_mc = cophasing_handoff_monte_carlo(
            wavelengths_m=wavelengths_m,
            cophasing_counts_per_channel=cophasing_counts,
            cophasing_counts_by_wavelength=cophasing_counts_by_wavelength,
            wavelength_groups=wavelength_groups,
            piston_basis=basis_set.physical_basis,
            science_whitened_response_per_m=whitened,
            piston_rms_m=float(params.piston_rms_nm_for_mc) / _M_TO_NM,
            n_trials=int(params.cophasing_handoff_n_monte_carlo),
            rng=np.random.default_rng(int(params.mc_seed) + 30_000),
            edges=_cophasing_edges(params, basis_set.n_physical_modes),
            tracker_fraction=float(params.cophasing_tracker_fraction),
            visibility=float(params.cophasing_visibility),
            edge_throughput=float(params.cophasing_edge_throughput),
            phase_offsets_rad=np.asarray(
                params.cophasing_phase_offsets_rad,
                dtype=np.float64,
            ),
            edge_count_weights=(
                None
                if len(params.cophasing_edge_count_weights) == 0
                else np.asarray(params.cophasing_edge_count_weights, dtype=np.float64)
            ),
            dark_counts_per_output=float(cophasing_dark_counts),
            count_cap_per_output=cophasing_count_cap,
            science_drift_rms_m=float(params.cophasing_science_drift_rms_nm) / _M_TO_NM,
            lock_threshold_m=float(params.cophasing_lock_threshold_nm) / _M_TO_NM,
            shot_noise=bool(params.shot_noise),
            relative_threshold=float(params.relative_singular_threshold),
        )
        nonlinear_input = np.asarray(nonlinear_mc.input_rms_m, dtype=np.float64)
        input_rms = (
            float(np.median(nonlinear_input)) if nonlinear_input.size else float("nan")
        )
        nonlinear_ratio = (
            float(nonlinear_mc.median_residual_rms_m / input_rms)
            if input_rms > 0.0
            else float("nan")
        )
        gauss_newton_input = np.asarray(gauss_newton_mc.input_rms_m, dtype=np.float64)
        gauss_newton_input_rms = (
            float(np.median(gauss_newton_input))
            if gauss_newton_input.size
            else float("nan")
        )
        gauss_newton_ratio = (
            float(gauss_newton_mc.median_residual_rms_m / gauss_newton_input_rms)
            if gauss_newton_input_rms > 0.0
            else float("nan")
        )

        ft_payload = _find_component(state, _FT_PREFIX, self.type_name, required=False)
        payload: dict[str, Any] = {
            "n_modes": int(n_modes),
            "n_chip_input_modes": int(n_modes),
            "n_physical_piston_modes": int(basis_set.n_physical_modes),
            "n_differential_piston_modes": int(basis_set.physical_basis.shape[1]),
            "piston_basis_source": basis_set.source,
            "n_bright_ports": int(bright_indices.size),
            "bright_port_indices": bright_indices.astype(np.int64, copy=True),
            "bright_port_labels": tuple(bright_labels),
            "n_spectral_channels_requested": (
                None
                if params.n_spectral_channels is None
                else int(params.n_spectral_channels)
            ),
            "n_spectral_channels_used": int(wavelength_indices.size),
            "wavelength_indices": wavelength_indices.astype(np.int64, copy=True),
            "wavelength_channel_groups": tuple(
                tuple(int(i) for i in group) for group in wavelength_groups
            ),
            "wavelength_grid_um": tuple(float(v) for v in wavelengths_um),
            "frame_time_s": float(frame_time_s),
            "frame_time_source": frame_time_source,
            "expected_counts_per_sample": signal_counts.astype(np.float64, copy=True),
            "expected_signal_counts_per_sample": signal_counts.astype(
                np.float64,
                copy=True,
            ),
            "variance_counts_per_sample": variance_counts.astype(
                np.float64,
                copy=True,
            ),
            "background_counts_per_sample": background_counts.astype(
                np.float64,
                copy=True,
            ),
            "expected_signal_counts_by_wavelength_per_sample": (
                signal_counts_by_wavelength.astype(np.float64, copy=True)
            ),
            "background_counts_by_wavelength_per_sample": (
                background_counts_by_wavelength.astype(np.float64, copy=True)
            ),
            **count_metadata,
            "row_port_indices": response.row_port_indices.astype(np.int64, copy=True),
            "row_wavelength_indices": response.row_wavelength_indices.astype(
                np.int64,
                copy=True,
            ),
            "row_channel_indices": response.row_channel_indices.astype(
                np.int64,
                copy=True,
            ),
            "singular_values_per_m": svd.singular_values.astype(np.float64, copy=True),
            "singular_value_tolerance_per_m": float(svd.tolerance),
            "rank": int(svd.rank),
            "full_rank": bool(svd.full_rank),
            "observable_fraction": float(svd.observable_fraction),
            "condition_number": float(svd.condition_number),
            "truncated_condition_number": float(svd.truncated_condition_number),
            "crlb_piston_rms_nm": float(crlb.piston_rms_m * _M_TO_NM),
            "crlb_coordinate_rms_nm": float(crlb.coordinate_rms_m * _M_TO_NM),
            "crlb_truncated_piston_rms_nm": float(
                crlb.truncated_piston_rms_m * _M_TO_NM,
            ),
            "crlb_truncated_coordinate_rms_nm": float(
                crlb.truncated_coordinate_rms_m * _M_TO_NM,
            ),
            "crlb_singular_mode_std_nm": (
                crlb.singular_mode_std_m.astype(np.float64, copy=True) * _M_TO_NM
            ),
            "crlb_finite": bool(crlb.finite),
            "piston_crlb_covariance_m2": (
                piston_covariance.piston_covariance_m2.astype(np.float64, copy=True)
            ),
            "piston_crlb_coordinate_covariance_m2": (
                piston_covariance.coordinate_covariance_m2.astype(
                    np.float64,
                    copy=True,
                )
            ),
            "piston_crlb_covariance_finite": bool(piston_covariance.finite),
            "piston_crlb_covariance_rms_nm": float(
                piston_covariance.piston_rms_m * _M_TO_NM
            ),
            "amplitude_observability": _amplitude_observability_payload(
                response=amplitude_response,
                projected_counts=amplitude_projected_counts,
                whitened=amplitude_whitened,
                svd=amplitude_svd,
                crlb=amplitude_crlb,
                include_response_matrix=bool(params.include_response_matrix),
                frame_time_s=amplitude_frame_time_s,
                frame_time_source=amplitude_frame_time_source,
                expected_counts=amplitude_signal_counts,
                estimator_port_set=params.amplitude_estimator_port_set,
                estimator_port_indices=amplitude_indices,
                estimator_port_labels=amplitude_labels,
            ),
            "scintillation_assumptions": scintillation_assumptions,
            "amplitude_postprocessing_monte_carlo": amplitude_postprocessing,
            "amplitude_postprocessing_model": (
                "small-signal differential-intensity time-stream diagnostic: "
                "draw per-aperture scintillation at the injected sigma_I, "
                "compare no correction with selected-port linear and "
                "Gaussian-prior/Wiener post-processing, and report an ideal "
                "per-aperture photometric tap reference using the same "
                "amplitude-sample photon budget"
            ),
            "channel_sweep": channel_sweep,
            "weakest_piston_modes_unit_rms": weak_modes.astype(np.float64, copy=True),
            "monte_carlo": _monte_carlo_payload(linear_mc),
            "monte_carlo_model": (
                "linearized Poisson model: whitened_response @ piston_coords "
                "+ unit Gaussian count noise"
            ),
            "nonlinear_one_step_monte_carlo": _monte_carlo_payload(nonlinear_mc),
            "nonlinear_one_step_model": (
                "true nonlinear chip counts reconstructed with one "
                "zero-point linear inverse; diagnoses capture range and "
                "model-curvature amplification"
            ),
            "nonlinear_one_step_residual_ratio": nonlinear_ratio,
            "gauss_newton_monte_carlo": _gauss_newton_payload(gauss_newton_mc),
            "gauss_newton_model": (
                "true nonlinear chip counts reconstructed with damped "
                "Gauss-Newton iterations and refreshed intensity Jacobians"
            ),
            "gauss_newton_residual_ratio": gauss_newton_ratio,
            "gauss_newton_capture_sweep": gauss_newton_sweep,
            "active_acquisition_monte_carlo": _active_acquisition_payload(
                active_acquisition_mc,
            ),
            "active_acquisition_model": (
                "known delay-line coordinate scan: hold one aperture as the "
                "piston reference, scan trial corrections for each remaining "
                "physical aperture, measure the bright-port signature after "
                "each applied correction, and retain the correction closest to "
                "the zero-piston calibration; optional active response-matrix "
                "iterations then dither each aperture, solve a weighted local "
                "least-squares correction, and repeat"
            ),
            "active_acquisition_n_passes": int(params.active_acquisition_n_passes),
            "active_acquisition_response_iterations": int(
                params.active_acquisition_response_iterations
            ),
            "active_acquisition_probe_step_nm": float(
                params.active_acquisition_probe_step_nm
            ),
            "active_acquisition_max_response_step_nm": float(
                params.active_acquisition_max_response_step_nm
            ),
            "active_acquisition_scan_half_range_nm": float(
                params.active_acquisition_scan_half_range_nm
            ),
            "active_acquisition_scan_steps": int(params.active_acquisition_scan_steps),
            "active_acquisition_reference_mode": int(
                params.active_acquisition_reference_mode
            ),
            "active_acquisition_lock_threshold_nm": float(
                params.active_acquisition_lock_threshold_nm
            ),
            "active_acquisition_calibration_frames": int(
                params.active_acquisition_calibration_frames
            ),
            "cophasing_handoff_monte_carlo": _cophasing_handoff_payload(
                cophasing_handoff_mc,
            ),
            "cophasing_handoff_model": (
                "mode-switch acquisition: reconfigure the programmable mesh "
                "to a dedicated pairwise ABCD cophasing unitary, solve "
                "physical aperture pistons on a connected tree graph, apply "
                "upstream delay corrections, then hand off to the science "
                "unitary's small-signal bright-port estimator for maintenance. "
                "The pair graph, edge count allocation, ABCD phase offsets, "
                "visibility, throughput, detector dark counts, and output cap "
                "are explicit payload/config assumptions."
            ),
            "cophasing_handoff_n_monte_carlo": int(
                params.cophasing_handoff_n_monte_carlo
            ),
            "cophasing_tracker_fraction": float(params.cophasing_tracker_fraction),
            "cophasing_visibility": float(params.cophasing_visibility),
            "cophasing_edge_throughput": float(params.cophasing_edge_throughput),
            "cophasing_phase_offsets_rad": tuple(
                float(value) for value in params.cophasing_phase_offsets_rad
            ),
            "cophasing_edge_count_weights": tuple(
                float(value) for value in params.cophasing_edge_count_weights
            ),
            "cophasing_science_drift_rms_nm": float(
                params.cophasing_science_drift_rms_nm
            ),
            "cophasing_lock_threshold_nm": float(params.cophasing_lock_threshold_nm),
            "cophasing_counts_per_sample": cophasing_counts.astype(
                np.float64,
                copy=True,
            ),
            "cophasing_counts_by_wavelength_per_sample": (
                cophasing_counts_by_wavelength.astype(np.float64, copy=True)
            ),
            "cophasing_dark_counts_per_output": float(cophasing_dark_counts),
            "cophasing_count_cap_per_output": (
                None if cophasing_count_cap is None else float(cophasing_count_cap)
            ),
            **cophasing_metadata,
            "gauss_newton_max_iter": int(params.gauss_newton_max_iter),
            "gauss_newton_damping": float(params.gauss_newton_damping),
            "gauss_newton_step_tolerance_nm": float(
                params.gauss_newton_step_tolerance_nm
            ),
            "gauss_newton_cost_relative_tolerance": float(
                params.gauss_newton_cost_relative_tolerance
            ),
            "gauss_newton_initialization": params.gauss_newton_initialization,
            "scalar_piston_sensing_efficiency": _scalar_efficiency(ft_payload),
            "source_model": (
                "Bright-port intensity response matrix from the deployed "
                "chip.kernel_mzi_mesh u_stack. n_spectral_channels denotes "
                "independent tracker count channels across the observing band, "
                "not lambda/FWHM resolving power; channel rates, intensity "
                "responses, and nonlinear count models are incoherently "
                "summed over the detector wavelength bins assigned to each "
                "tracker channel before projection and whitening."
            ),
        }
        if basis_set.positions_pupil_m is not None:
            payload["positions_pupil_m"] = basis_set.positions_pupil_m.astype(
                np.float64,
                copy=True,
            )
        if params.include_response_matrix:
            payload["response_matrix_counts_per_m"] = projected_counts.astype(
                np.float64,
                copy=True,
            )
            payload["whitened_response_matrix_per_m"] = whitened.astype(
                np.float64,
                copy=True,
            )

        base_name = self.context.instance_id or "fringe_observability"
        component_name = unique_component_name(
            base_name,
            state.results.components,
            self.context.instance_id,
        )
        results = state.results.add_component(component_name, payload)
        return replace(state, results=results)


def _find_component(
    state: SimulationState,
    prefix: str,
    stage_name: str,
    *,
    required: bool = True,
) -> dict[str, Any] | None:
    components = state.results.components
    exact = components.get(prefix)
    if isinstance(exact, dict):
        return exact
    if exact is not None and required:
        raise ValueError(
            f"{stage_name} requires result component {prefix!r} to be a dict."
        )

    matches = [
        (name, value)
        for name, value in components.items()
        if name.startswith(f"{prefix}.") and isinstance(value, dict)
    ]
    if len(matches) == 1:
        return matches[0][1]
    if len(matches) > 1:
        names = ", ".join(name for name, _value in matches)
        raise ValueError(
            f"{stage_name} found multiple results components matching "
            f"{prefix!r}: {names}. Use a single shared component for this "
            "diagnostic or add a stage selector."
        )
    if required:
        raise ValueError(f"{stage_name} requires a results component named {prefix!r}.")
    return None


def _require_u_stack(
    chip_payload: dict[str, Any],
    stage_name: str,
) -> NDArray[np.complex128]:
    raw = chip_payload.get("u_stack")
    if raw is None:
        raise ValueError(f"{stage_name} requires chip payload key 'u_stack'.")
    u_stack = np.asarray(raw, dtype=np.complex128)
    if u_stack.ndim != 3 or u_stack.shape[0] <= 0 or u_stack.shape[1] <= 0:
        raise ValueError(
            f"{stage_name} requires u_stack with shape "
            f"(n_ports, n_input_modes, n_wavelength); got {u_stack.shape}."
        )
    return u_stack


def _require_wavelength_grid_um(
    state: SimulationState,
    chip_payload: dict[str, Any],
    n_wavelengths: int,
    stage_name: str,
) -> tuple[float, ...]:
    raw = (
        chip_payload.get("wavelength_grid_um")
        or state.field.wavelength_grid_um
        or state.photon_rates.wavelength_grid_um
    )
    if raw is None:
        raise ValueError(f"{stage_name} requires a wavelength grid.")
    values = tuple(float(v) for v in raw)
    if len(values) != n_wavelengths:
        raise ValueError(
            f"{stage_name} wavelength grid length {len(values)} does not match "
            f"u_stack wavelength axis {n_wavelengths}."
        )
    return values


def _recover_chip_input_fields(
    state: SimulationState,
    u_stack: NDArray[np.complex128],
    stage_name: str,
) -> NDArray[np.complex128]:
    raw = state.field.amplitudes
    if raw is None:
        raise ValueError(f"{stage_name} requires post-chip field amplitudes.")
    output = np.asarray(raw, dtype=np.complex128)
    if (
        output.ndim != 3
        or output.shape[0] != u_stack.shape[0]
        or output.shape[1] != u_stack.shape[2]
        or output.shape[2] < 1
    ):
        raise ValueError(
            f"{stage_name} requires post-chip field amplitudes with shape "
            f"({u_stack.shape[0]}, {u_stack.shape[2]}, n_pol); got {output.shape}."
        )
    if output.shape[2] > 1:
        # Dual-pol PBS-arm deployment: the arms are separate chips sharing the
        # design unitary with small per-arm diagonal phase trims, so their
        # fractional piston responses agree to first order. The audit models
        # the dominant-power arm's coherent response; absolute count budgets
        # come from photon_rates, which already sum both arms.
        arm_powers = np.sum(np.abs(output) ** 2, axis=(0, 1))
        dominant_arm = int(np.argmax(arm_powers))
        output = output[:, :, dominant_arm : dominant_arm + 1]
    recovered = np.zeros((u_stack.shape[1], u_stack.shape[2], 1), dtype=np.complex128)
    for wave in range(u_stack.shape[2]):
        recovered[:, wave, 0] = u_stack[:, :, wave].conj().T @ output[:, wave, 0]
    return recovered


def _require_port_labels(
    state: SimulationState,
    n_ports: int,
    stage_name: str,
) -> tuple[str, ...]:
    raw = state.photon_rates.port_labels
    labels = () if raw is None else tuple(str(label) for label in raw)
    n_pol_arms = detector_n_pol_arms(state)
    if n_pol_arms > 1 and len(labels) == n_ports * n_pol_arms:
        # Dual-pol PBS-arm detector channels are port-major, arm-minor
        # (e.g. bright_0_A, bright_0_B, bright_1_A, ...). The audit works in
        # physical chip ports, so collapse each per-arm group to its base
        # label; the matching rate rows are arm-summed in
        # _detector_signal_rate_matrix.
        collapsed: list[str] = []
        for port_i in range(n_ports):
            group = labels[port_i * n_pol_arms : (port_i + 1) * n_pol_arms]
            bases = {label.rsplit("_", 1)[0] for label in group}
            if len(bases) != 1:
                raise ValueError(
                    f"{stage_name} could not collapse per-arm port labels "
                    f"{group}; expected a common '<port>_<arm>' base."
                )
            collapsed.append(bases.pop())
        labels = tuple(collapsed)
    if len(labels) != n_ports:
        raise ValueError(
            f"{stage_name} requires photon_rates.port_labels length {n_ports}; "
            f"got {0 if raw is None else len(raw)}."
        )
    return labels


def _sum_polarization_arm_rows(
    arr: NDArray[np.float64],
    n_pol_arms: int,
    stage_name: str,
) -> NDArray[np.float64]:
    if n_pol_arms <= 1:
        return arr
    if arr.shape[0] % n_pol_arms != 0:
        raise ValueError(
            f"{stage_name} cannot sum polarization arms: rate rows "
            f"({arr.shape[0]}) are not a multiple of n_pol_arms ({n_pol_arms})."
        )
    n_physical = arr.shape[0] // n_pol_arms
    return arr.reshape(n_physical, n_pol_arms, arr.shape[1]).sum(axis=1)


def _bright_port_indices(
    port_labels: tuple[str, ...],
    prefix: str,
    stage_name: str,
) -> tuple[NDArray[np.intp], tuple[str, ...]]:
    indices = np.array(
        [i for i, label in enumerate(port_labels) if label.startswith(prefix)],
        dtype=np.intp,
    )
    if indices.size == 0:
        raise ValueError(
            f"{stage_name} found no ports whose labels start with {prefix!r}."
        )
    labels = tuple(port_labels[int(i)] for i in indices)
    return indices, labels


def _amplitude_port_indices(
    port_labels: tuple[str, ...],
    *,
    bright_indices: NDArray[np.intp],
    port_set: Literal["bright", "bright_dark", "all"],
    stage_name: str,
) -> tuple[NDArray[np.intp], tuple[str, ...]]:
    """Return measured ports used by the amplitude estimator."""

    if port_set == "bright":
        indices = np.asarray(bright_indices, dtype=np.intp)
    elif port_set == "bright_dark":
        dark_indices = np.array(
            [i for i, label in enumerate(port_labels) if label.startswith("dark")],
            dtype=np.intp,
        )
        indices = np.unique(
            np.concatenate((np.asarray(bright_indices, dtype=np.intp), dark_indices))
        ).astype(np.intp, copy=False)
    elif port_set == "all":
        indices = np.arange(len(port_labels), dtype=np.intp)
    else:
        raise ValueError(f"{stage_name} unknown amplitude port set {port_set!r}.")
    if indices.size == 0:
        raise ValueError(
            f"{stage_name} amplitude_estimator_port_set={port_set!r} selected "
            "no measured ports."
        )
    labels = tuple(port_labels[int(i)] for i in indices)
    return indices, labels


def _piston_basis_set(
    state: SimulationState,
    n_chip_modes: int,
) -> _PistonBasisSet:
    """Return the physical piston basis and its chip-input expansion."""

    splitter = splitter_physical_indices(state, n_chip_modes)
    if splitter is None:
        basis = differential_piston_basis(n_chip_modes)
        return _PistonBasisSet(
            physical_basis=basis,
            chip_basis=basis,
            n_physical_modes=n_chip_modes,
            positions_pupil_m=_mode_positions_pupil_m(state, n_chip_modes),
            source="chip_input_modes",
        )

    physical_indices, source = splitter
    n_physical_modes = int(np.max(physical_indices)) + 1
    physical_basis = differential_piston_basis(n_physical_modes)
    expansion = np.zeros((n_chip_modes, n_physical_modes), dtype=np.float64)
    expansion[np.arange(n_chip_modes, dtype=np.intp), physical_indices] = 1.0
    return _PistonBasisSet(
        physical_basis=physical_basis,
        chip_basis=expansion @ physical_basis,
        n_physical_modes=n_physical_modes,
        positions_pupil_m=_physical_positions_from_splitter(
            state,
            physical_indices=physical_indices,
            n_physical_modes=n_physical_modes,
        ),
        source=source,
    )


def _select_wavelength_channels(
    n_wavelengths: int,
    n_channels: int | None,
) -> tuple[NDArray[np.intp], tuple[NDArray[np.intp], ...]]:
    """Return representative wavelength indices and contiguous count groups.

    ``n_channels`` means independent tracker count channels spanning the
    available band. It is intentionally not the spectrograph resolving power
    ``lambda / FWHM(lambda)``.
    """

    if n_wavelengths <= 0:
        raise ValueError("n_wavelengths must be positive.")
    if n_channels is None or n_channels >= n_wavelengths:
        indices = np.arange(n_wavelengths, dtype=np.intp)
        groups = tuple(np.array([i], dtype=np.intp) for i in indices)
        return indices, groups

    groups = tuple(
        np.asarray(group, dtype=np.intp)
        for group in np.array_split(
            np.arange(n_wavelengths, dtype=np.intp), int(n_channels)
        )
        if len(group) > 0
    )
    indices = np.array(
        [int(group[(len(group) - 1) // 2]) for group in groups],
        dtype=np.intp,
    )
    return indices, groups


def _frame_time_s(
    state: SimulationState,
    frame_time_override_s: float | None,
    stage_name: str,
) -> tuple[float, str]:
    if frame_time_override_s is not None:
        return float(frame_time_override_s), "params.frame_time_s"
    ft_payload = _find_component(state, _FT_PREFIX, stage_name, required=False)
    if isinstance(ft_payload, dict):
        servo = ft_payload.get("servo_bandwidth_hz")
        if isinstance(servo, (float, int)) and float(servo) > 0.0:
            return 1.0 / (2.0 * float(servo)), "fringe_tracking.servo_bandwidth"
    raise ValueError(
        f"{stage_name} requires params.frame_time_s when no upstream "
        "fringe_tracking.closed_loop servo_bandwidth_hz is available; it does "
        "not assume a 1 s fringe-tracker sample."
    )


def _amplitude_frame_time_s(
    *,
    state: SimulationState,
    params: FringeObservability.Params,
    piston_frame_time_s: float,
    stage_name: str,
) -> tuple[float, str]:
    if params.amplitude_frame_time_s is not None:
        return float(params.amplitude_frame_time_s), "params.amplitude_frame_time_s"
    if params.amplitude_servo_bandwidth_hz is not None:
        return (
            1.0 / (2.0 * float(params.amplitude_servo_bandwidth_hz)),
            "params.amplitude_servo_bandwidth_hz",
        )

    for name, payload in state.results.components.items():
        if not name.startswith("sensitivity.detection_curve"):
            continue
        if not isinstance(payload, dict):
            continue
        residual = payload.get("amplitude_servo_residual")
        if not isinstance(residual, dict):
            continue
        bandwidth = residual.get("servo_bandwidth_hz")
        if isinstance(bandwidth, (float, int)) and float(bandwidth) > 0.0:
            return (
                1.0 / (2.0 * float(bandwidth)),
                f"{name}.amplitude_servo_residual.servo_bandwidth_hz",
            )

    return (
        float(piston_frame_time_s),
        (
            "piston_frame_time_fallback; place diagnostics.fringe_observability "
            "after sensitivity.detection_curve or set amplitude_frame_time_s / "
            "amplitude_servo_bandwidth_hz for amplitude-servo CRLBs"
        ),
    )


def _expected_count_budget(
    *,
    state: SimulationState,
    port_indices: NDArray[np.intp],
    wavelength_groups: tuple[NDArray[np.intp], ...],
    frame_time_s: float,
    params: FringeObservability.Params,
    stage_name: str,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    dict[str, Any],
]:
    signal_rates, signal_rates_by_wavelength, metadata = _signal_rate_channels_hz(
        state=state,
        port_indices=port_indices,
        wavelength_groups=wavelength_groups,
        params=params,
        stage_name=stage_name,
    )
    signal_counts = (signal_rates.T.reshape(-1) * float(frame_time_s)).astype(
        np.float64,
        copy=False,
    )
    background_counts = np.full_like(
        signal_counts,
        float(params.dark_rate_hz_per_channel) * float(frame_time_s),
        dtype=np.float64,
    )
    signal_counts_by_wavelength = (
        signal_rates_by_wavelength * float(frame_time_s)
    ).astype(np.float64, copy=False)
    background_counts_by_wavelength = _background_counts_by_wavelength(
        n_ports=port_indices.size,
        n_wavelengths=signal_rates_by_wavelength.shape[1],
        wavelength_groups=wavelength_groups,
        dark_counts_per_channel=float(params.dark_rate_hz_per_channel)
        * float(frame_time_s),
    )
    variance_counts = signal_counts + background_counts
    metadata = {
        **metadata,
        "dark_rate_hz_per_channel": float(params.dark_rate_hz_per_channel),
        "count_frame_time_s": float(frame_time_s),
    }
    return (
        signal_counts,
        variance_counts,
        background_counts,
        signal_counts_by_wavelength,
        background_counts_by_wavelength,
        metadata,
    )


def _cophasing_count_budget(
    *,
    state: SimulationState,
    wavelength_groups: tuple[NDArray[np.intp], ...],
    frame_time_s: float,
    params: FringeObservability.Params,
    stage_name: str,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    float,
    float | None,
    dict[str, Any],
]:
    """Return total count budget available to the switched cophasing unitary."""

    rate_arr, source_metadata = _detector_signal_rate_matrix(
        state,
        stage_name=stage_name,
        source_mode=str(params.count_rate_source),
    )
    if len(wavelength_groups) == 0:
        raise ValueError(f"{stage_name} requires at least one spectral channel.")

    total_rates_by_wavelength = np.asarray(rate_arr, dtype=np.float64).sum(axis=0)
    total_rates = np.zeros(len(wavelength_groups), dtype=np.float64)
    for channel_i, group in enumerate(wavelength_groups):
        wave_idx = np.asarray(group, dtype=np.intp)
        if wave_idx.ndim != 1 or wave_idx.size == 0:
            raise ValueError(f"{stage_name} spectral channel groups must be non-empty.")
        if np.any(wave_idx < 0) or np.any(wave_idx >= rate_arr.shape[1]):
            raise ValueError(f"{stage_name} selected wavelength index is invalid.")
        total_rates[channel_i] = float(total_rates_by_wavelength[wave_idx].sum())

    if not np.all(np.isfinite(total_rates)) or np.any(total_rates < 0.0):
        raise ValueError(f"{stage_name} expects finite non-negative cophasing rates.")

    upstream_qe, detector_component = _upstream_detector_qe(state)
    if params.tracker_qe is None:
        qe_scale = 1.0
        tracker_qe = upstream_qe
        count_source = f"{source_metadata['rate_source']} summed over all science ports"
    else:
        if upstream_qe is None or upstream_qe <= 0.0:
            raise ValueError(
                f"{stage_name} tracker_qe requires an upstream detector payload "
                "with positive detector_qe so photon_rates can be rescaled."
            )
        tracker_qe = float(params.tracker_qe)
        qe_scale = tracker_qe / upstream_qe
        total_rates *= qe_scale
        total_rates_by_wavelength *= qe_scale
        count_source = (
            f"{source_metadata['rate_source']} summed over all science ports and "
            "scaled to tracker_qe"
        )

    dark_counts = float(params.dark_rate_hz_per_channel) * float(frame_time_s)
    cap = params.tracker_count_rate_cap_hz_per_channel
    cap_counts = None if cap is None else float(cap) * float(frame_time_s)
    metadata = {
        "cophasing_count_source": count_source,
        "cophasing_upstream_detector_component": detector_component,
        "cophasing_upstream_detector_qe": (
            None if upstream_qe is None else float(upstream_qe)
        ),
        "cophasing_tracker_qe": None if tracker_qe is None else float(tracker_qe),
        "cophasing_tracker_qe_scale": float(qe_scale),
        **{
            f"cophasing_{key}": value
            for key, value in source_metadata.items()
            if key != "rate_source"
        },
    }
    return (
        (total_rates * float(frame_time_s)).astype(np.float64, copy=False),
        (total_rates_by_wavelength * float(frame_time_s)).astype(
            np.float64,
            copy=False,
        ),
        dark_counts,
        cap_counts,
        metadata,
    )


def _signal_rate_channels_hz(
    *,
    state: SimulationState,
    port_indices: NDArray[np.intp],
    wavelength_groups: tuple[NDArray[np.intp], ...],
    params: FringeObservability.Params,
    stage_name: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
    rate_arr, source_metadata = _detector_signal_rate_matrix(
        state,
        stage_name=stage_name,
        source_mode=str(params.count_rate_source),
    )
    if np.any(port_indices < 0) or np.any(port_indices >= rate_arr.shape[0]):
        raise ValueError(f"{stage_name} selected port index is invalid.")
    if len(wavelength_groups) == 0:
        raise ValueError(f"{stage_name} requires at least one spectral channel.")

    selected_by_wavelength = np.asarray(rate_arr[port_indices, :], dtype=np.float64)
    selected = np.zeros((port_indices.size, len(wavelength_groups)), dtype=np.float64)
    for channel_i, group in enumerate(wavelength_groups):
        wave_idx = np.asarray(group, dtype=np.intp)
        if wave_idx.ndim != 1 or wave_idx.size == 0:
            raise ValueError(f"{stage_name} spectral channel groups must be non-empty.")
        if np.any(wave_idx < 0) or np.any(wave_idx >= rate_arr.shape[1]):
            raise ValueError(f"{stage_name} selected wavelength index is invalid.")
        selected[:, channel_i] = selected_by_wavelength[:, wave_idx].sum(axis=1)

    if not np.all(np.isfinite(selected)) or np.any(selected < 0.0):
        raise ValueError(f"{stage_name} expects finite non-negative bright rates.")

    upstream_qe, detector_component = _upstream_detector_qe(state)
    if params.tracker_qe is None:
        tracker_qe = upstream_qe
        qe_scale = 1.0
        count_source = source_metadata["rate_source"]
    else:
        if upstream_qe is None or upstream_qe <= 0.0:
            raise ValueError(
                f"{stage_name} tracker_qe requires an upstream detector payload "
                "with positive detector_qe so photon_rates can be rescaled."
            )
        tracker_qe = float(params.tracker_qe)
        qe_scale = tracker_qe / upstream_qe
        selected = selected * qe_scale
        selected_by_wavelength = selected_by_wavelength * qe_scale
        count_source = f"{source_metadata['rate_source']} scaled to tracker_qe"

    cap = params.tracker_count_rate_cap_hz_per_channel
    saturated = np.zeros_like(selected, dtype=bool)
    if cap is not None:
        saturated = selected > float(cap)
        for channel_i, group in enumerate(wavelength_groups):
            channel_rates = selected[:, channel_i]
            with np.errstate(divide="ignore", invalid="ignore"):
                scale = np.where(
                    channel_rates > float(cap),
                    float(cap) / channel_rates,
                    1.0,
                )
            scale = np.where(np.isfinite(scale), scale, 1.0)
            selected_by_wavelength[:, group] *= scale[:, np.newaxis]
            selected[:, channel_i] = np.minimum(channel_rates, float(cap))

    metadata: dict[str, Any] = {
        "count_source": count_source,
        "tracker_detector_model": (
            "explicit_tracker_qe"
            if params.tracker_qe is not None
            else "upstream_detector"
        ),
        # MODELING ASSUMPTIONS (review 2026-06-10 Issue 19): (1) the
        # tracker QE / dark rate / count cap apply to EVERY selected
        # estimator port — with amplitude_estimator_port_set="bright_dark"
        # the MKID-read dark science ports are treated as if SNSPD-read,
        # ~20% optimistic in counts with the wrong saturation ceiling;
        # (2) the count-rate cap is modeled as a perfect achromatic
        # attenuator (proportional spectrum rescale, Fisher shape
        # preserved), whereas the cophasing MC in this same stage models
        # the cap as hard clipping of the ABCD outputs.
        "tracker_readout_assumption": (
            "tracker qe/dark/cap applied to all selected estimator ports; "
            "count cap modeled as upstream achromatic attenuation"
        ),
        "upstream_detector_component": detector_component,
        "upstream_detector_qe": None if upstream_qe is None else float(upstream_qe),
        "tracker_qe": None if tracker_qe is None else float(tracker_qe),
        "tracker_qe_scale": float(qe_scale),
        "tracker_count_rate_cap_hz_per_channel": (None if cap is None else float(cap)),
        "tracker_saturated_channels": saturated.T.reshape(-1).astype(bool, copy=True),
        "n_tracker_saturated_channels": int(np.count_nonzero(saturated)),
        **source_metadata,
    }
    return (
        selected.astype(np.float64, copy=False),
        selected_by_wavelength.astype(np.float64, copy=False),
        metadata,
    )


def _upstream_detector_qe(state: SimulationState) -> tuple[float | None, str | None]:
    for name, payload in state.results.components.items():
        if not name.startswith(("detector.mkid", "detector.snspd")):
            continue
        if not isinstance(payload, dict):
            continue
        value = payload.get("detector_qe")
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            return float(value), str(name)
    return None, None


def _detector_signal_rate_matrix(
    state: SimulationState,
    *,
    stage_name: str,
    source_mode: str = "auto",
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Return the best available per-port, per-wavelength signal-rate budget."""

    if source_mode not in {"auto", "photon_rates", "detector_uncapped"}:
        raise ValueError(
            f"{stage_name} count_rate_source={source_mode!r} is invalid; "
            "expected 'auto', 'photon_rates', or 'detector_uncapped'."
        )

    detector_name: str | None = None
    if source_mode != "photon_rates":
        for name, payload in state.results.components.items():
            if not name.startswith(("detector.mkid", "detector.snspd")):
                continue
            if not isinstance(payload, dict):
                continue
            detector_name = str(name)
            for key in (
                "uncapped_port_rate_hz_per_wavelength",
                "true_rates_hz_per_port_wavelength",
            ):
                raw = payload.get(key)
                if raw is None:
                    continue
                arr = np.asarray(raw, dtype=np.float64)
                if arr.ndim != 2:
                    raise ValueError(
                        f"{stage_name} expected detector payload {name!r} key "
                        f"{key!r} to have shape (n_ports, n_wavelength), got "
                        f"{arr.shape}."
                    )
                n_pol_arms = detector_n_pol_arms(state)
                arr = _sum_polarization_arm_rows(arr, n_pol_arms, stage_name)
                return arr, {
                    "rate_source": f"{name}.{key}",
                    "rate_source_component": detector_name,
                    "rate_source_is_uncapped": True,
                    "count_rate_source_mode": source_mode,
                    "polarization_arms_summed": int(n_pol_arms),
                }
        if source_mode == "detector_uncapped":
            raise ValueError(
                f"{stage_name} count_rate_source='detector_uncapped' requires an "
                "upstream detector payload with per-wavelength uncapped/true rates."
            )

    rates = state.photon_rates.rates_hz
    if rates is None:
        raise ValueError(
            f"{stage_name} requires photon_rates.rates_hz or detector "
            "uncapped per-wavelength rates."
        )
    arr = np.asarray(rates, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"{stage_name} expects photon_rates.rates_hz with shape "
            f"(n_ports, n_wavelength); got {arr.shape}."
        )
    n_pol_arms = detector_n_pol_arms(state)
    arr = _sum_polarization_arm_rows(arr, n_pol_arms, stage_name)
    return arr, {
        "rate_source": "state.photon_rates.rates_hz",
        "rate_source_component": detector_name,
        "rate_source_is_uncapped": False,
        "count_rate_source_mode": source_mode,
        "polarization_arms_summed": int(n_pol_arms),
    }


def _background_counts_by_wavelength(
    *,
    n_ports: int,
    n_wavelengths: int,
    wavelength_groups: tuple[NDArray[np.intp], ...],
    dark_counts_per_channel: float,
) -> NDArray[np.float64]:
    if n_wavelengths <= 0 or len(wavelength_groups) <= 0:
        raise ValueError("background counts require positive wavelength/group counts.")
    background = np.zeros((int(n_ports), int(n_wavelengths)), dtype=np.float64)
    for group in wavelength_groups:
        if group.size == 0:
            raise ValueError("background count groups must be non-empty.")
        background[:, group] = float(dark_counts_per_channel) / float(group.size)
    return background


def _linear_observability(
    *,
    u_stack: NDArray[np.complex128],
    input_fields: NDArray[np.complex128],
    wavelengths_m: NDArray[np.float64],
    bright_indices: NDArray[np.intp],
    wavelength_indices: NDArray[np.intp],
    wavelength_groups: tuple[NDArray[np.intp], ...],
    piston_basis: NDArray[np.float64],
    n_piston_modes: int,
    signal_counts: NDArray[np.float64],
    signal_counts_by_wavelength: NDArray[np.float64],
    variance_counts: NDArray[np.float64],
    params: FringeObservability.Params,
):
    response = spectral_channel_count_response_matrix(
        u_stack=u_stack,
        input_fields=input_fields,
        wavelengths_m=wavelengths_m,
        port_indices=bright_indices,
        wavelength_groups=wavelength_groups,
        expected_counts_by_wavelength=signal_counts_by_wavelength,
    )
    if response.expected_counts.shape != signal_counts.shape:
        raise ValueError("grouped response count rows do not match signal counts.")
    projected_counts = response.response_counts_per_m @ piston_basis
    whitened = whiten_response_matrix(
        response_counts_per_m=projected_counts,
        variance_counts=variance_counts,
        noise_floor_counts=float(params.noise_floor_counts),
    )
    svd = svd_observability(
        whitened,
        relative_threshold=float(params.relative_singular_threshold),
        absolute_threshold=float(params.absolute_singular_threshold),
    )
    crlb = cramer_rao_bound_from_svd(svd, n_piston_modes=int(n_piston_modes))
    return response, projected_counts, whitened, svd, crlb


def _amplitude_observability(
    *,
    u_stack: NDArray[np.complex128],
    input_fields: NDArray[np.complex128],
    wavelengths_m: NDArray[np.float64],
    port_indices: NDArray[np.intp],
    wavelength_groups: tuple[NDArray[np.intp], ...],
    amplitude_basis: NDArray[np.float64],
    n_amplitude_modes: int,
    signal_counts: NDArray[np.float64],
    signal_counts_by_wavelength: NDArray[np.float64],
    variance_counts: NDArray[np.float64],
    params: FringeObservability.Params,
):
    response = spectral_channel_fractional_intensity_response_matrix(
        u_stack=u_stack,
        input_fields=input_fields,
        wavelengths_m=wavelengths_m,
        port_indices=port_indices,
        wavelength_groups=wavelength_groups,
        expected_counts_by_wavelength=signal_counts_by_wavelength,
    )
    if response.expected_counts.shape != signal_counts.shape:
        raise ValueError("grouped amplitude response count rows do not match counts.")
    projected_counts = response.response_counts_per_fractional_intensity @ amplitude_basis
    whitened = whiten_response_matrix(
        response_counts_per_m=projected_counts,
        variance_counts=variance_counts,
        noise_floor_counts=float(params.noise_floor_counts),
    )
    svd = svd_observability(
        whitened,
        relative_threshold=float(params.relative_singular_threshold),
        absolute_threshold=float(params.absolute_singular_threshold),
    )
    crlb = cramer_rao_bound_from_svd(svd, n_piston_modes=int(n_amplitude_modes))
    return response, projected_counts, whitened, svd, crlb


def _amplitude_observability_payload(
    *,
    response,
    projected_counts: NDArray[np.float64],
    whitened: NDArray[np.float64],
    svd,
    crlb,
    include_response_matrix: bool,
    frame_time_s: float,
    frame_time_source: str,
    expected_counts: NDArray[np.float64],
    estimator_port_set: str,
    estimator_port_indices: NDArray[np.intp],
    estimator_port_labels: tuple[str, ...],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": (
            "small-signal selected-port response to differential physical-aperture "
            "fractional intensity perturbations; common flux changes are "
            "removed because they do not break the null"
        ),
        "coordinate": "fractional_intensity",
        "estimator_port_set": str(estimator_port_set),
        "estimator_port_indices": estimator_port_indices.astype(
            np.int64,
            copy=True,
        ),
        "estimator_port_labels": tuple(estimator_port_labels),
        "n_estimator_ports": int(estimator_port_indices.size),
        "n_differential_amplitude_modes": int(svd.right_singular_vectors.shape[1]),
        "frame_time_s": float(frame_time_s),
        "frame_time_source": str(frame_time_source),
        "expected_signal_counts_per_sample": expected_counts.astype(
            np.float64,
            copy=True,
        ),
        "row_port_indices": response.row_port_indices.astype(np.int64, copy=True),
        "row_wavelength_indices": response.row_wavelength_indices.astype(
            np.int64,
            copy=True,
        ),
        "row_channel_indices": response.row_channel_indices.astype(
            np.int64,
            copy=True,
        ),
        "singular_values_per_fractional_intensity": svd.singular_values.astype(
            np.float64,
            copy=True,
        ),
        "singular_value_tolerance_per_fractional_intensity": float(svd.tolerance),
        "rank": int(svd.rank),
        "full_rank": bool(svd.full_rank),
        "observable_fraction": float(svd.observable_fraction),
        "condition_number": float(svd.condition_number),
        "truncated_condition_number": float(svd.truncated_condition_number),
        "crlb_fractional_intensity_rms": float(crlb.piston_rms_m),
        "crlb_fractional_intensity_percent": float(100.0 * crlb.piston_rms_m),
        "crlb_truncated_fractional_intensity_rms": float(
            crlb.truncated_piston_rms_m
        ),
        "crlb_truncated_fractional_intensity_percent": float(
            100.0 * crlb.truncated_piston_rms_m
        ),
        "crlb_coordinate_rms": float(crlb.coordinate_rms_m),
        "crlb_truncated_coordinate_rms": float(crlb.truncated_coordinate_rms_m),
        "crlb_singular_mode_std": crlb.singular_mode_std_m.astype(
            np.float64,
            copy=True,
        ),
        "crlb_finite": bool(crlb.finite),
    }
    if include_response_matrix:
        payload["response_matrix_counts_per_fractional_intensity"] = (
            projected_counts.astype(np.float64, copy=True)
        )
        payload["whitened_response_matrix_per_fractional_intensity"] = (
            whitened.astype(np.float64, copy=True)
        )
    return payload


def _scintillation_assumptions_payload(
    *,
    state: SimulationState,
    sigma_i_override: float | None,
    amplitude_frame_time_s: float,
) -> dict[str, Any]:
    """Return the scintillation residual being tested by this diagnostic."""

    sensitivity_name, sensitivity = _find_sensitivity_payload(state)
    residual = (
        sensitivity.get("amplitude_servo_residual")
        if isinstance(sensitivity, dict)
        else None
    )
    payload: dict[str, Any] = {
        "enabled": False,
        "source": "none",
        "amplitude_frame_time_s": float(amplitude_frame_time_s),
    }

    if isinstance(residual, dict):
        sigma_residual = _finite_float_or_none(residual.get("sigma_I_residual"))
        sigma_open = _finite_float_or_none(residual.get("sigma_I_open"))
        sigma_temporal = _finite_float_or_none(residual.get("sigma_I_temporal"))
        sigma_photon = _finite_float_or_none(residual.get("sigma_I_photon"))
        tau_loop_s = _finite_float_or_none(residual.get("tau_loop_s"))
        tau_scint_s = _finite_float_or_none(residual.get("tau_scint_s"))
        predictive_gain = _finite_float_or_none(
            residual.get("predictive_gain_variance")
        )
        photon_rate = _finite_float_or_none(residual.get("photon_rate_per_sub_hz"))
        servo_bandwidth = _finite_float_or_none(residual.get("servo_bandwidth_hz"))
        sigma_no_prediction = _no_prediction_sigma_i(
            sigma_open=sigma_open,
            sigma_photon=sigma_photon,
            tau_loop_s=tau_loop_s,
            tau_scint_s=tau_scint_s,
        )
        injected = (
            float(sigma_i_override)
            if sigma_i_override is not None
            else (0.0 if sigma_residual is None else float(sigma_residual))
        )
        payload.update(
            {
                "enabled": True,
                "source": (
                    "params.amplitude_postprocessing_sigma_i"
                    if sigma_i_override is not None
                    else f"{sensitivity_name}.amplitude_servo_residual"
                ),
                "scintillation_sigma_i_injected": float(injected),
                "sigma_I_residual": (
                    None if sigma_residual is None else float(sigma_residual)
                ),
                "sigma_I_open": None if sigma_open is None else float(sigma_open),
                "sigma_I_temporal": (
                    None if sigma_temporal is None else float(sigma_temporal)
                ),
                "sigma_I_photon": (
                    None if sigma_photon is None else float(sigma_photon)
                ),
                "sigma_I_no_prediction_same_bandwidth": sigma_no_prediction,
                "predictive_gain_variance": (
                    None if predictive_gain is None else float(predictive_gain)
                ),
                "servo_bandwidth_hz": (
                    None if servo_bandwidth is None else float(servo_bandwidth)
                ),
                "tau_loop_s": None if tau_loop_s is None else float(tau_loop_s),
                "tau_scint_s": None if tau_scint_s is None else float(tau_scint_s),
                "photon_rate_per_sub_hz": (
                    None if photon_rate is None else float(photon_rate)
                ),
                "ideal_tap_counts_per_mode_from_rate": (
                    None
                    if photon_rate is None
                    else float(photon_rate) * float(amplitude_frame_time_s)
                ),
                "model_note": (
                    "sigma_I_injected is the residual used by the realized-null "
                    "Monte Carlo. sigma_I_no_prediction_same_bandwidth removes "
                    "only the predictive variance gain while keeping the same "
                    "servo bandwidth and photon-noise term."
                ),
            }
        )
        return payload

    manual_sigma = None
    if isinstance(sensitivity, dict):
        manual_sigma = _finite_float_or_none(sensitivity.get("scintillation_index_per_sub"))
    if sigma_i_override is not None or manual_sigma is not None:
        injected = float(sigma_i_override) if sigma_i_override is not None else manual_sigma
        payload.update(
            {
                "enabled": True,
                "source": (
                    "params.amplitude_postprocessing_sigma_i"
                    if sigma_i_override is not None
                    else f"{sensitivity_name}.scintillation_index_per_sub"
                ),
                "scintillation_sigma_i_injected": float(injected),
                "sigma_I_residual": float(injected),
                "model_note": (
                    "Manual scintillation scalar; no open-loop or servo "
                    "breakdown was available from sensitivity.detection_curve."
                ),
            }
        )
    return payload


def _amplitude_postprocessing_payload(
    *,
    whitened: NDArray[np.float64],
    basis_set: _PistonBasisSet,
    scintillation_assumptions: dict[str, Any],
    amplitude_frame_time_s: float,
    estimator_port_set: str,
    params: FringeObservability.Params,
) -> dict[str, Any]:
    n_trials = int(params.amplitude_postprocessing_n_monte_carlo)
    sigma_i = _finite_float_or_none(
        scintillation_assumptions.get("scintillation_sigma_i_injected")
    )
    if n_trials == 0 or sigma_i is None:
        return {
            "enabled": False,
            "n_trials": n_trials,
            "estimator_port_set": str(estimator_port_set),
            "disabled_reason": (
                "n_trials is zero"
                if n_trials == 0
                else "no scintillation sigma available"
            ),
        }

    ideal_counts = params.amplitude_postprocessing_ideal_tap_counts_per_mode
    ideal_count_source = "params.amplitude_postprocessing_ideal_tap_counts_per_mode"
    if ideal_counts is None:
        ideal_counts = _finite_float_or_none(
            scintillation_assumptions.get("ideal_tap_counts_per_mode_from_rate")
        )
        ideal_count_source = (
            "sensitivity.detection_curve.amplitude_servo_residual."
            "photon_rate_per_sub_hz * amplitude_frame_time_s"
        )
    if ideal_counts is None:
        ideal_counts = 0.0
        ideal_count_source = "unavailable"

    mc = amplitude_postprocessing_monte_carlo(
        whitened_response_per_fractional_intensity=whitened,
        amplitude_basis=basis_set.physical_basis,
        scintillation_sigma_i=float(sigma_i),
        ideal_tap_counts_per_mode=float(ideal_counts),
        n_trials=n_trials,
        rng=np.random.default_rng(int(params.mc_seed) + 40_000),
        relative_threshold=float(params.relative_singular_threshold),
    )
    return {
        "enabled": True,
        "n_trials": n_trials,
        "estimator_port_set": str(estimator_port_set),
        "scintillation_sigma_i_injected": float(sigma_i),
        "amplitude_frame_time_s": float(amplitude_frame_time_s),
        "ideal_tap_counts_per_mode": float(ideal_counts),
        "ideal_tap_count_source": ideal_count_source,
        "input_rms_percent": (
            mc.input_rms_fractional_intensity.astype(np.float64, copy=True) * 100.0
        ),
        "bright_linear_residual_rms_percent": (
            mc.bright_linear_residual_rms_fractional_intensity.astype(
                np.float64,
                copy=True,
            )
            * 100.0
        ),
        "linear_residual_rms_percent": (
            mc.bright_linear_residual_rms_fractional_intensity.astype(
                np.float64,
                copy=True,
            )
            * 100.0
        ),
        "bright_wiener_residual_rms_percent": (
            mc.bright_wiener_residual_rms_fractional_intensity.astype(
                np.float64,
                copy=True,
            )
            * 100.0
        ),
        "wiener_residual_rms_percent": (
            mc.bright_wiener_residual_rms_fractional_intensity.astype(
                np.float64,
                copy=True,
            )
            * 100.0
        ),
        "ideal_tap_residual_rms_percent": (
            mc.ideal_tap_residual_rms_fractional_intensity.astype(
                np.float64,
                copy=True,
            )
            * 100.0
        ),
        "median_input_rms_percent": (
            100.0 * mc.median_input_rms_fractional_intensity
        ),
        "median_bright_linear_residual_rms_percent": (
            100.0 * mc.median_bright_linear_residual_rms_fractional_intensity
        ),
        "median_linear_residual_rms_percent": (
            100.0 * mc.median_bright_linear_residual_rms_fractional_intensity
        ),
        "p95_bright_linear_residual_rms_percent": (
            100.0 * mc.p95_bright_linear_residual_rms_fractional_intensity
        ),
        "p95_linear_residual_rms_percent": (
            100.0 * mc.p95_bright_linear_residual_rms_fractional_intensity
        ),
        "median_bright_wiener_residual_rms_percent": (
            100.0 * mc.median_bright_wiener_residual_rms_fractional_intensity
        ),
        "median_wiener_residual_rms_percent": (
            100.0 * mc.median_bright_wiener_residual_rms_fractional_intensity
        ),
        "p95_bright_wiener_residual_rms_percent": (
            100.0 * mc.p95_bright_wiener_residual_rms_fractional_intensity
        ),
        "p95_wiener_residual_rms_percent": (
            100.0 * mc.p95_bright_wiener_residual_rms_fractional_intensity
        ),
        "median_ideal_tap_residual_rms_percent": (
            100.0 * mc.median_ideal_tap_residual_rms_fractional_intensity
        ),
        "p95_ideal_tap_residual_rms_percent": (
            100.0 * mc.p95_ideal_tap_residual_rms_fractional_intensity
        ),
    }


def _find_sensitivity_payload(
    state: SimulationState,
) -> tuple[str, dict[str, Any] | None]:
    for name, payload in state.results.components.items():
        if name.startswith("sensitivity.detection_curve") and isinstance(payload, dict):
            return str(name), payload
    return "sensitivity.detection_curve", None


def _finite_float_or_none(value: Any) -> float | None:
    if not isinstance(value, (float, int)):
        return None
    out = float(value)
    return out if np.isfinite(out) else None


def _no_prediction_sigma_i(
    *,
    sigma_open: float | None,
    sigma_photon: float | None,
    tau_loop_s: float | None,
    tau_scint_s: float | None,
) -> float | None:
    if (
        sigma_open is None
        or sigma_photon is None
        or tau_loop_s is None
        or tau_scint_s is None
        or tau_scint_s <= 0.0
        or tau_loop_s <= 0.0
    ):
        return None
    sigma_temporal_no_prediction = float(sigma_open) * (
        float(tau_loop_s) / float(tau_scint_s)
    ) ** (5.0 / 6.0)
    sigma = np.sqrt(sigma_temporal_no_prediction ** 2 + float(sigma_photon) ** 2)
    return float(min(sigma, float(sigma_open)))


def _channel_sweep_payload(
    *,
    state: SimulationState,
    u_stack: NDArray[np.complex128],
    input_fields: NDArray[np.complex128],
    wavelengths_m: NDArray[np.float64],
    bright_indices: NDArray[np.intp],
    piston_basis: NDArray[np.float64],
    n_piston_modes: int,
    frame_time_s: float,
    n_wavelengths: int,
    max_channels: int,
    params: FringeObservability.Params,
    stage_name: str,
) -> dict[str, Any]:
    channels = np.arange(1, max(1, int(max_channels)) + 1, dtype=np.int64)
    used = np.zeros(channels.shape, dtype=np.int64)
    rank = np.zeros(channels.shape, dtype=np.int64)
    observable = np.zeros(channels.shape, dtype=np.float64)
    condition = np.full(channels.shape, np.inf, dtype=np.float64)
    truncated_condition = np.full(channels.shape, np.inf, dtype=np.float64)
    crlb_piston_nm = np.full(channels.shape, np.inf, dtype=np.float64)
    crlb_truncated_piston_nm = np.full(channels.shape, np.inf, dtype=np.float64)

    for i, n_channels in enumerate(channels):
        wave_idx, groups = _select_wavelength_channels(n_wavelengths, int(n_channels))
        (
            signal_counts,
            variance_counts,
            _background,
            signal_counts_by_wavelength,
            _background_by_wavelength,
            _metadata,
        ) = _expected_count_budget(
            state=state,
            port_indices=bright_indices,
            wavelength_groups=groups,
            frame_time_s=frame_time_s,
            params=params,
            stage_name=stage_name,
        )
        _response, _projected, _whitened, svd, crlb = _linear_observability(
            u_stack=u_stack,
            input_fields=input_fields,
            wavelengths_m=wavelengths_m,
            bright_indices=bright_indices,
            wavelength_indices=wave_idx,
            wavelength_groups=groups,
            piston_basis=piston_basis,
            n_piston_modes=n_piston_modes,
            signal_counts=signal_counts,
            signal_counts_by_wavelength=signal_counts_by_wavelength,
            variance_counts=variance_counts,
            params=params,
        )
        used[i] = int(wave_idx.size)
        rank[i] = int(svd.rank)
        observable[i] = float(svd.observable_fraction)
        condition[i] = float(svd.condition_number)
        truncated_condition[i] = float(svd.truncated_condition_number)
        crlb_piston_nm[i] = float(crlb.piston_rms_m * _M_TO_NM)
        crlb_truncated_piston_nm[i] = float(crlb.truncated_piston_rms_m * _M_TO_NM)

    return {
        "n_spectral_channels": channels,
        "n_spectral_channels_used": used,
        "rank": rank,
        "observable_fraction": observable,
        "condition_number": condition,
        "truncated_condition_number": truncated_condition,
        "crlb_piston_rms_nm": crlb_piston_nm,
        "crlb_truncated_piston_rms_nm": crlb_truncated_piston_nm,
        "single_channel_rank": int(rank[0]) if rank.size else 0,
    }


def _gauss_newton_capture_sweep_payload(
    *,
    u_stack: NDArray[np.complex128],
    input_fields: NDArray[np.complex128],
    wavelengths_m: NDArray[np.float64],
    bright_indices: NDArray[np.intp],
    wavelength_indices: NDArray[np.intp],
    wavelength_groups: tuple[NDArray[np.intp], ...],
    basis_set: _PistonBasisSet,
    reference_power: NDArray[np.float64],
    signal_counts: NDArray[np.float64],
    signal_counts_by_wavelength: NDArray[np.float64],
    variance_counts: NDArray[np.float64],
    background_counts: NDArray[np.float64],
    background_counts_by_wavelength: NDArray[np.float64],
    whitened: NDArray[np.float64],
    params: FringeObservability.Params,
) -> dict[str, Any]:
    """Run an optional capture-range sweep over injected piston RMS."""

    grid_nm = np.asarray(params.gauss_newton_piston_rms_grid_nm, dtype=np.float64)
    if grid_nm.size == 0 or int(params.gauss_newton_sweep_n_monte_carlo) == 0:
        return {
            "piston_rms_nm": grid_nm.astype(np.float64, copy=True),
            "n_trials": int(params.gauss_newton_sweep_n_monte_carlo),
            "enabled": False,
        }
    if np.any(~np.isfinite(grid_nm)) or np.any(grid_nm < 0.0):
        raise ValueError(
            "diagnostics.fringe_observability gauss_newton_piston_rms_grid_nm "
            "must contain finite non-negative values."
        )

    med = np.zeros(grid_nm.size, dtype=np.float64)
    p95 = np.zeros(grid_nm.size, dtype=np.float64)
    conv = np.zeros(grid_nm.size, dtype=np.float64)
    worse = np.zeros(grid_nm.size, dtype=np.float64)
    median_iter = np.zeros(grid_nm.size, dtype=np.float64)
    for i, rms_nm in enumerate(grid_nm):
        mc = gauss_newton_reconstruction_monte_carlo(
            u_stack=u_stack,
            input_fields=input_fields,
            wavelengths_m=wavelengths_m,
            port_indices=bright_indices,
            wavelength_indices=wavelength_indices,
            piston_basis=basis_set.physical_basis,
            reference_power=reference_power,
            expected_counts=signal_counts,
            variance_counts=variance_counts,
            background_counts=background_counts,
            whitened_response_per_m=whitened,
            wavelength_groups=wavelength_groups,
            expected_counts_by_wavelength=signal_counts_by_wavelength,
            background_counts_by_wavelength=background_counts_by_wavelength,
            piston_rms_m=float(rms_nm) / _M_TO_NM,
            n_trials=int(params.gauss_newton_sweep_n_monte_carlo),
            rng=np.random.default_rng(int(params.mc_seed) + 10_000 + i),
            field_piston_basis=basis_set.chip_basis,
            relative_threshold=float(params.relative_singular_threshold),
            shot_noise=bool(params.shot_noise),
            max_iter=int(params.gauss_newton_max_iter),
            damping=float(params.gauss_newton_damping),
            step_tolerance_m=float(params.gauss_newton_step_tolerance_nm) / _M_TO_NM,
            cost_relative_tolerance=float(params.gauss_newton_cost_relative_tolerance),
            initialization=params.gauss_newton_initialization,
            noise_floor_counts=float(params.noise_floor_counts),
        )
        med[i] = float(mc.median_residual_rms_m * _M_TO_NM)
        p95[i] = float(mc.p95_residual_rms_m * _M_TO_NM)
        conv[i] = float(mc.convergence_fraction)
        worse[i] = float(mc.worse_than_input_fraction)
        median_iter[i] = float(np.median(mc.iterations)) if mc.iterations.size else 0.0
    return {
        "enabled": True,
        "n_trials": int(params.gauss_newton_sweep_n_monte_carlo),
        "piston_rms_nm": grid_nm.astype(np.float64, copy=True),
        "median_residual_rms_nm": med,
        "p95_residual_rms_nm": p95,
        "convergence_fraction": conv,
        "worse_than_input_fraction": worse,
        "median_iterations": median_iter,
    }


def _active_acquisition_monte_carlo(
    *,
    u_stack: NDArray[np.complex128],
    input_fields: NDArray[np.complex128],
    wavelengths_m: NDArray[np.float64],
    bright_indices: NDArray[np.intp],
    wavelength_indices: NDArray[np.intp],
    wavelength_groups: tuple[NDArray[np.intp], ...],
    basis_set: _PistonBasisSet,
    reference_power: NDArray[np.float64],
    signal_counts: NDArray[np.float64],
    signal_counts_by_wavelength: NDArray[np.float64],
    variance_counts: NDArray[np.float64],
    background_counts: NDArray[np.float64],
    background_counts_by_wavelength: NDArray[np.float64],
    params: FringeObservability.Params,
):
    """Run the optional active aperture-by-aperture acquisition diagnostic."""

    half_range_nm = float(params.active_acquisition_scan_half_range_nm)
    scan_steps = int(params.active_acquisition_scan_steps)
    scan_offsets_m = (
        np.linspace(-half_range_nm, half_range_nm, scan_steps, dtype=np.float64)
        / _M_TO_NM
    )
    return active_coordinate_acquisition_monte_carlo(
        u_stack=u_stack,
        input_fields=input_fields,
        wavelengths_m=wavelengths_m,
        port_indices=bright_indices,
        wavelength_indices=wavelength_indices,
        piston_basis=basis_set.physical_basis,
        reference_power=reference_power,
        expected_counts=signal_counts,
        variance_counts=variance_counts,
        background_counts=background_counts,
        wavelength_groups=wavelength_groups,
        expected_counts_by_wavelength=signal_counts_by_wavelength,
        background_counts_by_wavelength=background_counts_by_wavelength,
        calibration_frames=int(params.active_acquisition_calibration_frames),
        piston_rms_m=float(params.piston_rms_nm_for_mc) / _M_TO_NM,
        n_trials=int(params.active_acquisition_n_monte_carlo),
        rng=np.random.default_rng(int(params.mc_seed) + 20_000),
        scan_offsets_m=scan_offsets_m,
        n_passes=int(params.active_acquisition_n_passes),
        reference_mode=int(params.active_acquisition_reference_mode),
        lock_threshold_m=float(params.active_acquisition_lock_threshold_nm) / _M_TO_NM,
        response_iterations=int(params.active_acquisition_response_iterations),
        probe_step_m=float(params.active_acquisition_probe_step_nm) / _M_TO_NM,
        max_response_step_m=(
            float(params.active_acquisition_max_response_step_nm) / _M_TO_NM
        ),
        field_piston_basis=basis_set.chip_basis,
        shot_noise=bool(params.shot_noise),
        noise_floor_counts=float(params.noise_floor_counts),
        relative_threshold=float(params.relative_singular_threshold),
    )


def _physical_positions_from_splitter(
    state: SimulationState,
    *,
    physical_indices: NDArray[np.intp],
    n_physical_modes: int,
) -> NDArray[np.float64] | None:
    raw = state.geometry.aperture_positions_pupil_m
    if raw is None:
        raw = state.geometry.aperture_positions_enu_m
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float64)
    if arr.shape == (n_physical_modes, 2):
        return arr.astype(np.float64, copy=True)
    if arr.shape == (n_physical_modes, 3):
        return arr[:, :2].astype(np.float64, copy=True)
    if arr.shape == (physical_indices.size, 3):
        arr = arr[:, :2]
    if arr.shape == (physical_indices.size, 2):
        positions = np.zeros((n_physical_modes, 2), dtype=np.float64)
        seen = np.zeros(n_physical_modes, dtype=bool)
        for chip_index, physical_index in enumerate(physical_indices):
            if not seen[int(physical_index)]:
                positions[int(physical_index)] = arr[chip_index]
                seen[int(physical_index)] = True
        if np.all(seen):
            return positions
    return None


def _mode_positions_pupil_m(
    state: SimulationState,
    n_modes: int,
) -> NDArray[np.float64] | None:
    raw = state.geometry.aperture_positions_pupil_m
    if raw is None:
        raw = state.geometry.aperture_positions_enu_m
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float64)
    if arr.shape == (n_modes, 2):
        return arr.astype(np.float64, copy=True)
    if arr.shape == (n_modes, 3):
        return arr[:, :2].astype(np.float64, copy=True)
    return None


def _scalar_efficiency(ft_payload: dict[str, Any] | None) -> float | None:
    if not isinstance(ft_payload, dict):
        return None
    value = ft_payload.get("piston_sensing_efficiency")
    if isinstance(value, (int, float)) and np.isfinite(float(value)):
        return float(value)
    return None


def _cophasing_edges(
    params: FringeObservability.Params,
    n_physical_modes: int,
) -> NDArray[np.int64] | None:
    if not params.cophasing_edges:
        return None
    arr = np.asarray(params.cophasing_edges, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            "diagnostics.fringe_observability cophasing_edges must have "
            "shape (n_edges, 2)."
        )
    if np.any(arr < 0) or np.any(arr >= int(n_physical_modes)):
        raise ValueError(
            "diagnostics.fringe_observability cophasing_edges contains an "
            "aperture index outside the physical piston basis."
        )
    return arr


def _monte_carlo_payload(mc: Any) -> dict[str, Any]:
    return {
        "input_rms_nm": np.asarray(mc.input_rms_m, dtype=np.float64) * _M_TO_NM,
        "recovered_rms_nm": (
            np.asarray(mc.recovered_rms_m, dtype=np.float64) * _M_TO_NM
        ),
        "residual_rms_nm": np.asarray(mc.residual_rms_m, dtype=np.float64) * _M_TO_NM,
        "median_residual_rms_nm": float(mc.median_residual_rms_m * _M_TO_NM),
        "p95_residual_rms_nm": float(mc.p95_residual_rms_m * _M_TO_NM),
        "median_recovered_rms_nm": float(mc.median_recovered_rms_m * _M_TO_NM),
    }


def _gauss_newton_payload(mc: Any) -> dict[str, Any]:
    payload = _monte_carlo_payload(mc)
    payload.update(
        {
            "converged": np.asarray(mc.converged, dtype=bool).copy(),
            "iterations": np.asarray(mc.iterations, dtype=np.int64).copy(),
            "final_weighted_cost": np.asarray(
                mc.final_weighted_cost,
                dtype=np.float64,
            ).copy(),
            "convergence_fraction": float(mc.convergence_fraction),
            "max_iter_fraction": float(mc.max_iter_fraction),
            "worse_than_input_fraction": float(mc.worse_than_input_fraction),
        }
    )
    return payload


def _active_acquisition_payload(mc: Any) -> dict[str, Any]:
    payload = _monte_carlo_payload(mc)
    payload.update(
        {
            "locked": np.asarray(mc.locked, dtype=bool).copy(),
            "evaluations": np.asarray(mc.evaluations, dtype=np.int64).copy(),
            "final_weighted_cost": np.asarray(
                mc.final_weighted_cost,
                dtype=np.float64,
            ).copy(),
            "lock_fraction": float(mc.lock_fraction),
            "worse_than_input_fraction": float(mc.worse_than_input_fraction),
        }
    )
    return payload


def _cophasing_handoff_payload(mc: Any) -> dict[str, Any]:
    return {
        "input_rms_nm": np.asarray(mc.input_rms_m, dtype=np.float64) * _M_TO_NM,
        "acquisition_residual_rms_nm": (
            np.asarray(mc.acquisition_residual_rms_m, dtype=np.float64) * _M_TO_NM
        ),
        "science_input_rms_nm": (
            np.asarray(mc.science_input_rms_m, dtype=np.float64) * _M_TO_NM
        ),
        "science_residual_rms_nm": (
            np.asarray(mc.science_residual_rms_m, dtype=np.float64) * _M_TO_NM
        ),
        "median_acquisition_residual_rms_nm": float(
            mc.median_acquisition_residual_rms_m * _M_TO_NM
        ),
        "p95_acquisition_residual_rms_nm": float(
            mc.p95_acquisition_residual_rms_m * _M_TO_NM
        ),
        "median_science_residual_rms_nm": float(
            mc.median_science_residual_rms_m * _M_TO_NM
        ),
        "p95_science_residual_rms_nm": float(mc.p95_science_residual_rms_m * _M_TO_NM),
        "median_science_input_rms_nm": float(mc.median_science_input_rms_m * _M_TO_NM),
        "p95_science_input_rms_nm": float(mc.p95_science_input_rms_m * _M_TO_NM),
        "acquisition_locked": np.asarray(mc.acquisition_locked, dtype=bool).copy(),
        "science_locked": np.asarray(mc.science_locked, dtype=bool).copy(),
        "acquisition_lock_fraction": float(mc.acquisition_lock_fraction),
        "science_lock_fraction": float(mc.science_lock_fraction),
        "edges": np.asarray(mc.edges, dtype=np.int64).copy(),
        "edge_wrap_fraction": float(mc.edge_wrap_fraction),
        "max_edge_opd_over_lambda": float(mc.max_edge_opd_over_lambda),
    }


__all__ = ["FringeObservability"]
