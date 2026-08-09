"""Tests for the chip-restart loop controls (KILO optimizer tricks)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nullsim.pipeline import RunContext
from nullsim.pipeline.state import Field, SimulationState
from nullsim.stages.chip_optimization import ChipOptimizationParams
from nullsim.stages.chip_optimization.scipy import ChipOptimizationScipy


def _state(n_modes: int = 4, n_wl: int = 3, wl_design: float = 1.25) -> SimulationState:
    state = SimulationState.empty(grid={
        "wavelength_grid_um": tuple(np.linspace(wl_design - 0.05, wl_design + 0.05, n_wl).tolist()),
    })
    amps = np.ones((n_modes, n_wl, 1), dtype=np.complex128) / np.sqrt(n_modes)
    return replace(state, field=Field(
        amplitudes=amps,
        wavelength_grid_um=tuple(np.linspace(wl_design - 0.05, wl_design + 0.05, n_wl).tolist()),
    ))


def _ctx() -> RunContext:
    return RunContext(
        rng=np.random.default_rng(0),
        config_hash="abc",
        sweep_coord_tuple=(),
        instance_id="chip_opt",
        run_dir=Path("/tmp"),
    )


def test_restart_patience_disabled_runs_all_restarts() -> None:
    """patience=0 disables early exit: all n_restarts must run when the
    target is unreachable. Phase 9: prior assertion only verified
    ``optimal_chip_params is not None`` which a single restart would
    also pass; we now read n_restarts_completed from the payload."""
    params = ChipOptimizationParams(
        n_modes=4, n_bright_ports=2, n_dark_ports=2,
        wl_design_um=1.25, target_null_depth=1e-20,  # unreachable
        restart_patience=0, n_restarts=3, seed=0, max_iter=10,
    )
    stage = ChipOptimizationScipy(params=params.model_dump(), context=_ctx())
    out = stage.apply(_state())
    payload = next(iter(out.results.components.values()))
    assert payload["optimal_chip_params"] is not None
    assert payload["n_restarts_completed"] == 3, (
        "patience=0 with unreachable target must execute every restart; "
        f"got n_restarts_completed={payload['n_restarts_completed']}."
    )


def test_restart_patience_enabled_can_short_circuit() -> None:
    """Companion to the above: when restart_patience > 0 and no restart
    improves, the optimizer bails before n_restarts. Pins the early-exit
    contract that Phase 9 exposed via n_restarts_completed."""
    params = ChipOptimizationParams(
        n_modes=4, n_bright_ports=2, n_dark_ports=2,
        wl_design_um=1.25, target_null_depth=1e-20,  # unreachable
        restart_patience=1, n_restarts=10, seed=0, max_iter=10,
    )
    stage = ChipOptimizationScipy(params=params.model_dump(), context=_ctx())
    out = stage.apply(_state())
    payload = next(iter(out.results.components.values()))
    assert payload["n_restarts_completed"] < 10, (
        "restart_patience=1 must short-circuit once no restart improves; "
        f"got n_restarts_completed={payload['n_restarts_completed']}."
    )


def test_n_workers_pool_gives_same_or_better_loss_than_serial() -> None:
    """Pool mode dispatches all restarts; result must not be WORSE than serial."""
    base_kwargs = dict(
        n_modes=4, n_bright_ports=2, n_dark_ports=2, wl_design_um=1.25,
        target_null_depth=1e-20,  # forces all restarts to run
        n_restarts=3, seed=7, max_iter=50, restart_patience=0,
    )
    p_serial = ChipOptimizationParams(**base_kwargs, n_workers=1)
    p_pool = ChipOptimizationParams(**base_kwargs, n_workers=2)

    out_serial = ChipOptimizationScipy(params=p_serial.model_dump(), context=_ctx()).apply(_state())
    out_pool = ChipOptimizationScipy(params=p_pool.model_dump(), context=_ctx()).apply(_state())

    loss_serial = next(iter(out_serial.results.components.values()))["optimization_loss_final"]
    loss_pool = next(iter(out_pool.results.components.values()))["optimization_loss_final"]
    # Both modes use the same restart seeds so the best loss is identical
    # bit-for-bit if all restarts complete (target unreachable, no early exit).
    assert loss_pool == pytest.approx(loss_serial, rel=1e-12, abs=1e-15)


def test_target_null_early_exit_still_works() -> None:
    """Trivial 2-mode case: optimizer finds a deep null in restart 0 → bail."""
    params = ChipOptimizationParams(
        n_modes=2, n_bright_ports=1, n_dark_ports=1,
        wl_design_um=1.25, target_null_depth=1e-3,
        n_restarts=10, seed=0, max_iter=200,
    )
    stage = ChipOptimizationScipy(params=params.model_dump(), context=_ctx())
    out = stage.apply(_state(n_modes=2))
    payload = next(iter(out.results.components.values()))
    assert payload["optimization_normalized_null_final"] < params.target_null_depth
    assert payload["n_restarts_completed"] < params.n_restarts


def test_scipy_honors_n_wl_optimize(monkeypatch: pytest.MonkeyPatch) -> None:
    """The scipy backend must use the same wavelength-subset contract as torch."""

    import nullsim.stages.chip_optimization.scipy as scipy_backend

    def fake_minimize(fun, x0, **_kwargs):
        return SimpleNamespace(fun=float(fun(x0)), x=np.asarray(x0), nit=0)

    monkeypatch.setattr(scipy_backend, "minimize", fake_minimize)
    params = ChipOptimizationParams(
        n_modes=4,
        n_bright_ports=2,
        n_dark_ports=2,
        wl_design_um=1.25,
        target_null_depth=1.0e9,
        n_restarts=1,
        max_iter=1,
        n_wl_optimize=2,
    )

    out = ChipOptimizationScipy(params=params.model_dump(), context=_ctx()).apply(
        _state(n_modes=4, n_wl=3, wl_design=1.25)
    )
    payload = next(iter(out.results.components.values()))

    assert payload["optimizer_wavelength_grid_um"] == pytest.approx((1.20, 1.30))
    assert payload["optimizer_wavelength_indices"] == (0, 2)
