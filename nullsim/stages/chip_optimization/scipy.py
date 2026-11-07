"""Scipy backend for chip null optimization.

SPEC §15.3 Pass 3 WP3. Reference implementation: KILO ``chip.py``
``optimize_null`` (lines 639-757) and ``_null_objective`` (lines 304-331);
WP1 notes at ``tests/stages/notes/chip_optimization.md``.

Differences from KILO ``optimize_null`` retained intentionally:

* Method is ``L-BFGS-B`` (the WP3 spec line), not KILO's default ``BFGS``.
  L-BFGS-B accepts an explicit ``maxiter`` cap that is honored on every
  problem size (the unconstrained BFGS in scipy can keep stepping past
  ``maxiter`` in the inner line search). For small meshes the basin
      behaviour is the same; if a future optimizer study needs the BFGS
      basin, re-add it as a polish step.
* Gradient is scipy's default 2-point finite-difference. Analytic
  gradients live in the torch backend.
* Optional ``polish='SLSQP'`` flag re-runs from the L-BFGS-B optimum
  to clean up the final iterate when L-BFGS-B exits early due to the
  step-size tolerance.

Loss: see ``nullsim.stages.chip_optimization`` for the broadband leakage
formula and spectral reduction modes.
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from nullsim.pipeline import BaseStage, SimulationState, register_stage
from nullsim.stages._utils import (
    mode_origin_positions_m,
    project_aperture_positions_for_pointing,
)
from nullsim.stages.chip_optimization import (
    ChipOptimizationParams,
    _chip_chromaticity_metadata,
    _dark_indices,
    _delay_state_current_transport_scale_from_state,
    _delay_state_fields_from_state,
    _delay_state_metadata,
    _delay_state_transport_scales_from_state,
    _loss_meets_target,
    _loss_numpy,
    _mode_diameters_m,
    _n_params,
    _normalization_field_array,
    _normalized_loss_for_target,
    _null_indices,
    _optimizer_null_field_stack,
    _optimizer_wavelength_metadata,
    _optimizer_wavelength_subset,
    _quantize_phases,
    _reported_null_depth_denominator,
    _require_field,
    _require_wavelength_grid_um,
    _spectral_loss_metadata,
    _stellar_radius_rad,
    _throughput_penalty_fields,
    _uses_disk_samples,
    _write_results,
)

_PROCESS_POOL_UNAVAILABLE_ERRORS = (OSError, NotImplementedError)

_History = tuple[tuple[int, float], ...] | None
_RestartResult = tuple[
    float, NDArray[np.float64], NDArray[np.float64], int, _History, _History
]


def _run_one_restart_worker(
    args: tuple,
) -> _RestartResult:
    """Top-level worker for the chip-restart ProcessPoolExecutor.

    Must be top-level (not a closure) so it pickles cleanly under the
    spawn multiprocessing context. Mirrors KILO ``_process_worker_run_restart``
    (kilo/chip.py L762).

    Returns ``(loss, x_final, x_unquantized, nit, history, null_history)``.
    With history capture on, ``history`` holds ``(nfev, objective)`` and
    ``null_history`` holds ``(nfev, reported null)``: the leakage term
    alone over ``null_denominator``, the convention published as
    ``optimization_normalized_null_final`` (review 2026-11-07 Issue 23).
    """

    (
        x0,
        field_array,
        wavelength_grid_um,
        params,
        null_idx,
        dark_idx,
        capture_convergence_history,
        aperture_positions_pupil_m,
        sub_diameters_m,
        stellar_radius_rad,
        delay_state_fields,
        throughput_penalty_fields,
        null_denominator,
    ) = args

    nfev_counter = [0]

    # Optimize on the SMOOTH (unquantized) objective. Quantizing inside
    # the loss makes it piecewise-constant on the DAC grid (step
    # 2*pi/2**dac_bits ~ 1e-4) while L-BFGS-B's finite-difference step is
    # ~1.5e-8: every FD gradient component is exactly zero and the
    # optimizer exits at nit=0 with the random init — a silent no-op
    # optimization (review 2026-06-10 Issue 13). Quantization is applied
    # post-hoc to the final iterate and the loss re-evaluated there,
    # matching the torch backend's contract.
    smooth_params = (
        params
        if params.dac_bits is None
        else params.model_copy(update={"dac_bits": None})
    )

    def _loss(
        x: NDArray[np.float64],
        loss_params: ChipOptimizationParams,
        tp_fields: NDArray[np.complex128] | None,
    ) -> float:
        return _loss_numpy(
            x, field_array=field_array,
            wavelength_grid_um=wavelength_grid_um,
            params=loss_params, null_idx=null_idx, dark_idx=dark_idx,
            aperture_positions_pupil_m=aperture_positions_pupil_m,
            sub_diameters_m=sub_diameters_m,
            stellar_radius_rad=stellar_radius_rad,
            delay_state_fields=delay_state_fields,
            throughput_penalty_fields=tp_fields,
        )

    def _obj(x: NDArray[np.float64]) -> float:
        nfev_counter[0] += 1
        return _loss(x, smooth_params, throughput_penalty_fields)

    history: list[tuple[int, float]] | None = None
    null_history: list[tuple[int, float]] | None = None
    callback = None
    if capture_convergence_history:
        history = []
        null_history = []

        def callback(xk: NDArray[np.float64], _h=history, _hn=null_history) -> None:
            _h.append((nfev_counter[0], _loss(xk, smooth_params, throughput_penalty_fields)))
            _hn.append((nfev_counter[0], _loss(xk, smooth_params, None) / null_denominator))

        callback(x0)

    result = minimize(
        _obj, x0,
        method="L-BFGS-B",
        options={"maxiter": params.max_iter},
        callback=callback,
    )
    nit = int(getattr(result, "nit", 0))
    x_unquantized = np.asarray(result.x, dtype=np.float64)
    x_final = x_unquantized
    if params.dac_bits is not None:
        x_final = _quantize_phases(x_final, params.dac_bits)
    # Re-evaluate at the (possibly quantized) final iterate with the
    # ORIGINAL params so the returned loss matches the deployed chip.
    loss = _loss(x_final, params, throughput_penalty_fields)
    if history is not None:
        if history[-1][0] != nfev_counter[0]:
            history.append((nfev_counter[0], loss))
            null_history.append(
                (nfev_counter[0], _loss(x_final, params, None) / null_denominator)
            )
        return (
            loss, x_final, x_unquantized, nit, tuple(history), tuple(null_history),
        )
    return loss, x_final, x_unquantized, nit, None, None


@register_stage
class ChipOptimizationScipy(BaseStage):
    """Find Clements params that minimize broadband dark-port leakage (scipy backend)."""

    type_name = "chip_optimization.scipy"
    family = "chip_optimization"
    consumes = frozenset({"field", "throughput", "geometry"})
    produces = frozenset({"results"})
    cache_key_extras = frozenset({"scene"})

    Params = ChipOptimizationParams

    def required_consumes(self) -> frozenset[str]:
        required = set(self.consumes)
        if _uses_disk_samples(self.params_model):
            required.add("scene")
        return frozenset(required)

    def apply(self, state: SimulationState) -> SimulationState:
        params: ChipOptimizationParams = self.params_model
        if (
            params.tracker_observability_weight > 0.0
            or params.tracker_observability_crlb_weight > 0.0
            or params.tracker_observability_rank_priority
        ):
            raise ValueError(
                "tracker_observability is only implemented for "
                "chip_optimization.torch; use the torch backend for the "
                "bright-port piston Fisher regularizer."
            )
        if params.amplitude_observability_weight > 0.0:
            raise ValueError(
                "amplitude_observability is only implemented for "
                "chip_optimization.torch; use the torch backend for the "
                "selected-port amplitude Fisher regularizer."
            )
        if (
            params.joint_observability_weight > 0.0
            or params.joint_observability_crlb_weight > 0.0
        ):
            raise ValueError(
                "joint_observability is only implemented for "
                "chip_optimization.torch; use the torch backend for the "
                "joint phase+amplitude Fisher regularizer."
            )

        field_array = _require_field(
            state,
            params.n_modes,
            wl_design_um=params.wl_design_um,
        )
        wavelength_grid_um = _require_wavelength_grid_um(state, field_array)
        full_wavelength_grid_um = tuple(float(w) for w in wavelength_grid_um)
        field_array, wavelength_grid_um, optimizer_wavelength_indices = (
            _optimizer_wavelength_subset(field_array, wavelength_grid_um, params)
        )
        null_idx = _null_indices(params)
        dark_idx = _dark_indices(params)
        positions_pupil = (
            project_aperture_positions_for_pointing(
                np.asarray(
                    state.geometry.aperture_positions_pupil_m, dtype=np.float64
                ),
                altitude_deg=state.scene.altitude_deg,
                azimuth_deg=state.scene.azimuth_deg,
                aperture_origins_m=mode_origin_positions_m(
                    state.geometry,
                    int(
                        np.asarray(
                            state.geometry.aperture_positions_pupil_m
                        ).shape[0]
                    ),
                ),
            )
            if state.geometry.aperture_positions_pupil_m is not None
            else None
        )
        sub_diameters_m = (
            _mode_diameters_m(state, params.n_modes)
            if _uses_disk_samples(params)
            else None
        )
        stellar_radius_rad = _stellar_radius_rad(state)
        delay_state_fields = _delay_state_fields_from_state(
            state=state,
            field_array=field_array,
            params=params,
            wavelength_grid_um=wavelength_grid_um,
        )
        delay_state_transport_scales = _delay_state_transport_scales_from_state(
            state=state,
            field_shape=tuple(int(value) for value in field_array.shape),
            params=params,
            wavelength_grid_um=wavelength_grid_um,
        )
        delay_state_current_transport_scale = (
            _delay_state_current_transport_scale_from_state(
                state=state,
                field_shape=tuple(int(value) for value in field_array.shape),
                params=params,
                wavelength_grid_um=wavelength_grid_um,
            )
        )
        throughput_penalty_fields = _throughput_penalty_fields(
            state=state,
            field_array=field_array,
            wavelength_grid_um=wavelength_grid_um,
            params=params,
            aperture_positions_pupil_m=positions_pupil,
            delay_state_transport_scales=delay_state_transport_scales,
            delay_state_current_transport_scale=delay_state_current_transport_scale,
        )
        target_field_array = _normalization_field_array(
            field_array,
            delay_state_fields,
        )
        if (
            params.kernel_order > 0
            or (_uses_disk_samples(params) and stellar_radius_rad > 0.0)
            or delay_state_fields is not None
        ):
            n_objective_rows = int(
                _optimizer_null_field_stack(
                    on_axis_field=field_array,
                    params=params,
                    aperture_positions_pupil_m=positions_pupil,
                    wavelength_grid_um=wavelength_grid_um,
                    sub_diameters_m=sub_diameters_m,
                    stellar_radius_rad=stellar_radius_rad,
                    delay_state_fields=delay_state_fields,
                ).shape[0]
            )
        else:
            n_objective_rows = 1
        # Reported-null convention counts rows per delay state (review
        # 2026-06-10 Issue 1).
        n_delay_states = (
            int(delay_state_fields.shape[0]) if delay_state_fields is not None else 1
        )
        null_denominator = _reported_null_depth_denominator(
            target_field_array,
            n_objective_rows=n_objective_rows,
            n_states=n_delay_states,
        )

        seed_base = 0 if params.seed is None else int(params.seed)
        n_params_total = _n_params(params.n_modes)

        # Pre-generate restart x0 vectors so the parallel and serial
        # paths share the same per-restart determinism (KILO chip.py
        # L1118-1122 pattern: restart seeds drawn from a master RNG so
        # both paths get bit-identical inits).
        master_rng = np.random.default_rng(seed_base)
        restart_seeds = [
            int(master_rng.integers(0, 2**31 - 1)) for _ in range(params.n_restarts)
        ]
        init_x0_list = [
            np.random.default_rng(s).uniform(0.0, 2.0 * np.pi, n_params_total)
            for s in restart_seeds
        ]

        worker_args = [
            (
                init_x0_list[k],
                field_array,
                wavelength_grid_um,
                params,
                null_idx,
                dark_idx,
                params.capture_convergence_history,
                positions_pupil,
                sub_diameters_m,
                stellar_radius_rad,
                delay_state_fields,
                throughput_penalty_fields,
                null_denominator,
            )
            for k in range(params.n_restarts)
        ]

        best_convergence_history: tuple[tuple[int, float], ...] | None = None
        best_null_history: tuple[tuple[int, float], ...] | None = None
        best_params: NDArray[np.float64] | None = None
        best_params_unquantized: NDArray[np.float64] | None = None
        best_loss = math.inf
        best_iterations = 0
        # Phase 9: count how many restarts actually executed so the
        # diagnostic can assert "patience=0 ran all of them" without
        # peeking at internal stage state.
        n_restarts_completed = 0

        if params.n_workers > 1 and params.n_restarts > 1:
            # KILO chip.py L1239 pattern: spawn context so workers re-import
            # cleanly under macOS torch/matplotlib. No mid-pool early exit
            # — all restarts are in-flight before we can inspect any.
            n_workers = min(params.n_workers, params.n_restarts)
            results = _run_restarts_with_pool_or_serial(worker_args, n_workers)
            n_restarts_completed = len(results)
            for loss, x_out, x_raw, nit, history, null_history in results:
                if loss < best_loss:
                    best_loss = loss
                    best_params = x_out
                    best_params_unquantized = x_raw
                    best_iterations = nit
                    best_convergence_history = history
                    best_null_history = null_history
        else:
            # Serial path with KILO-style stall detection across restarts.
            # KILO empirically: deepest basin is found in the first 1-5
            # restarts at M=24; further restarts usually just explore
            # shallower basins.
            n_no_improve = 0
            for worker_arg in worker_args:
                loss, x_out, x_raw, nit, history, null_history = (
                    _run_one_restart_worker(worker_arg)
                )
                n_restarts_completed += 1
                improved = loss < best_loss * (1.0 - params.restart_patience_reltol)
                if loss < best_loss:
                    best_loss = loss
                    best_params = x_out
                    best_params_unquantized = x_raw
                    best_iterations = nit
                    best_convergence_history = history
                    best_null_history = null_history
                if improved:
                    n_no_improve = 0
                else:
                    n_no_improve += 1
                # Target-null early exit uses dimensionless null depth,
                # while the optimizer objective remains KILO's absolute
                # leakage convention.
                if _loss_meets_target(
                    best_loss,
                    target_field_array,
                    params.target_null_depth,
                    n_objective_rows=n_objective_rows,
                    n_states=n_delay_states,
                ):
                    break
                # Patience early exit: bail when K consecutive restarts
                # don't improve over the running best.
                if params.restart_patience > 0 and n_no_improve >= params.restart_patience:
                    break

        # Code Review 2026-05-16 Issue 53: ``assert`` would be stripped
        # under ``python -O``. Raise explicitly.
        if best_params is None:
            raise RuntimeError(
                "chip_optimization.scipy produced no candidate after all "
                "restarts; check that n_restarts >= 1 and seed/x0 sweep "
                "is non-degenerate."
            )
        optimal_params = best_params
        loss_final = best_loss
        iterations = best_iterations

        if not _loss_meets_target(
            loss_final,
            target_field_array,
            params.target_null_depth,
            n_objective_rows=n_objective_rows,
            n_states=n_delay_states,
        ):
            smooth_params = (
                params
                if params.dac_bits is None
                else params.model_copy(update={"dac_bits": None})
            )

            def _polish_obj(x: NDArray[np.float64]) -> float:
                # Smooth (unquantized) objective; see _run_one_restart_worker
                # for the FD-gradient rationale (review 2026-06-10 Issue 13).
                return _loss_numpy(
                    x, field_array=field_array,
                    wavelength_grid_um=wavelength_grid_um,
                    params=smooth_params, null_idx=null_idx, dark_idx=dark_idx,
                    aperture_positions_pupil_m=positions_pupil,
                    sub_diameters_m=sub_diameters_m,
                    stellar_radius_rad=stellar_radius_rad,
                    delay_state_fields=delay_state_fields,
                    throughput_penalty_fields=throughput_penalty_fields,
                )
            polish = minimize(
                _polish_obj,
                optimal_params,
                method="SLSQP",
                options={"maxiter": max(1, params.max_iter // 4)},
            )
            polish_x_unquantized = np.asarray(polish.x, dtype=np.float64)
            polish_x = polish_x_unquantized
            if params.dac_bits is not None:
                polish_x = _quantize_phases(polish_x, params.dac_bits)
            polish_loss = float(
                _loss_numpy(
                    polish_x, field_array=field_array,
                    wavelength_grid_um=wavelength_grid_um,
                    params=params, null_idx=null_idx, dark_idx=dark_idx,
                    aperture_positions_pupil_m=positions_pupil,
                    sub_diameters_m=sub_diameters_m,
                    stellar_radius_rad=stellar_radius_rad,
                    delay_state_fields=delay_state_fields,
                    throughput_penalty_fields=throughput_penalty_fields,
                )
            )
            if polish_loss < loss_final:
                optimal_params = polish_x
                best_params_unquantized = polish_x_unquantized
                loss_final = polish_loss
                iterations += int(getattr(polish, "nit", 0))

        # Publish the null from the leakage term alone: with
        # throughput_penalty_alpha > 0 the objective carries the negative
        # throughput reward, which is not a null depth (Issue 23).
        leakage_final = _loss_numpy(
            optimal_params, field_array=field_array,
            wavelength_grid_um=wavelength_grid_um,
            params=params, null_idx=null_idx, dark_idx=dark_idx,
            aperture_positions_pupil_m=positions_pupil,
            sub_diameters_m=sub_diameters_m,
            stellar_radius_rad=stellar_radius_rad,
            delay_state_fields=delay_state_fields,
        )
        return _write_results(
            state=state,
            instance_id=self.context.instance_id,
            optimal_params=optimal_params,
            optimal_params_unquantized=best_params_unquantized,
            loss_final=loss_final,
            iterations=iterations,
            backend_name="scipy",
            # scipy uses physics.clements layout natively — broadband
            # consistent with the kernel's per-wavelength rescaling.
            kernel_rescales_per_wavelength=True,
            normalized_loss_final=leakage_final / null_denominator,
            convergence_history=best_convergence_history,
            convergence_history_normalized_null=best_null_history,
            n_restarts_completed=n_restarts_completed,
            backend_metadata={
                "optimization_normalized_null_legacy_final": (
                    _normalized_loss_for_target(leakage_final, target_field_array)
                ),
                "disk_null_floor_note": (
                    "chip-realization dependent"
                    if params.objective_is_flux_scale_dependent
                    else "flux-scale invariant"
                ),
                **_chip_chromaticity_metadata(params),
                **_spectral_loss_metadata(params),
                **_optimizer_wavelength_metadata(
                    full_wavelength_grid_um=full_wavelength_grid_um,
                    optimizer_wavelength_grid_um=wavelength_grid_um,
                    optimizer_wavelength_indices=optimizer_wavelength_indices,
                ),
                **_delay_state_metadata(params, delay_state_fields),
            },
        )


def _run_restarts_with_pool_or_serial(
    worker_args: list[tuple],
    n_workers: int,
) -> list[_RestartResult]:
    """Run all restarts in a spawn pool, falling back to serial if unavailable."""

    ctx = mp.get_context("spawn")
    try:
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            return list(pool.map(_run_one_restart_worker, worker_args))
    except _PROCESS_POOL_UNAVAILABLE_ERRORS as exc:
        logging.warning(
            "chip_optimization.scipy: process pool unavailable "
            "(%s: %s); running restarts serially.",
            type(exc).__name__,
            exc,
        )
        return [_run_one_restart_worker(worker_arg) for worker_arg in worker_args]


__all__ = ["ChipOptimizationScipy"]
