"""Tests for the chip-optimizer convergence figure (paperv3 Fig 9, scoped down)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import nullsim  # noqa: F401
from nullsim.outputs.plots.calibration_time import (
    _ConvergenceTrace,
    plot_calibration_time,
)


def _trace(label: str, color: str, n_points: int = 10) -> _ConvergenceTrace:
    return _ConvergenceTrace(
        record_label=label, color=color,
        nfev=np.arange(n_points) * 50,
        loss=np.logspace(0, -10, n_points),
    )


def test_render_one_trace(tmp_path: Path) -> None:
    out = tmp_path / "calib_one.pdf"
    fig = plot_calibration_time([_trace("J", "C1")], save_path=out)
    assert out.exists()
    assert len(fig.axes) == 1


def test_render_empty_returns_placeholder(tmp_path: Path) -> None:
    out = tmp_path / "calib_empty.pdf"
    fig = plot_calibration_time([], save_path=out)
    assert out.exists()
    assert len(fig.axes) == 1


def test_capture_convergence_history_end_to_end() -> None:
    """Wire the optimizer flag through scipy backend → state.results.components."""

    from dataclasses import replace
    from nullsim.pipeline import RunContext
    from nullsim.pipeline.state import Field, SimulationState
    from nullsim.stages.chip_optimization import ChipOptimizationParams
    from nullsim.stages.chip_optimization.scipy import ChipOptimizationScipy

    params = ChipOptimizationParams(
        n_modes=4, n_bright_ports=2, n_dark_ports=2,
        wl_design_um=1.25, target_null_depth=1e-6, max_iter=200,
        seed=7, capture_convergence_history=True,
    )
    ctx = RunContext(
        rng=np.random.default_rng(0),
        config_hash="abc",
        sweep_coord_tuple=(),
        instance_id="chip_opt",
        run_dir=Path("/tmp"),
    )
    stage = ChipOptimizationScipy(params=params.model_dump(), context=ctx)

    state = SimulationState.empty(grid={"wavelength_grid_um": (1.20, 1.25, 1.30)})
    amps = np.ones((4, 3, 1), dtype=np.complex128) / np.sqrt(4)
    state = replace(
        state,
        field=Field(amplitudes=amps, wavelength_grid_um=(1.20, 1.25, 1.30)),
    )

    out_state = stage.apply(state)
    payload = next(iter(out_state.results.components.values()))
    history = payload["convergence_history"]
    assert len(history) >= 2  # initial point + at least one callback
    nfev_first, loss_first = history[0]
    nfev_last, loss_last = history[-1]
    assert nfev_first == 0
    assert nfev_last > nfev_first
    assert loss_last < loss_first  # optimizer actually progressed


def test_default_flag_off_keeps_payload_clean() -> None:
    """capture_convergence_history defaults to False — no extra payload key."""

    from dataclasses import replace
    from nullsim.pipeline import RunContext
    from nullsim.pipeline.state import Field, SimulationState
    from nullsim.stages.chip_optimization import ChipOptimizationParams
    from nullsim.stages.chip_optimization.scipy import ChipOptimizationScipy

    params = ChipOptimizationParams(
        n_modes=4, n_bright_ports=2, n_dark_ports=2,
        wl_design_um=1.25, target_null_depth=1e-6, max_iter=100, seed=0,
    )
    ctx = RunContext(
        rng=np.random.default_rng(0),
        config_hash="abc",
        sweep_coord_tuple=(),
        instance_id="chip_opt",
        run_dir=Path("/tmp"),
    )
    stage = ChipOptimizationScipy(params=params.model_dump(), context=ctx)

    state = SimulationState.empty(grid={"wavelength_grid_um": (1.20, 1.25, 1.30)})
    amps = np.ones((4, 3, 1), dtype=np.complex128) / np.sqrt(4)
    state = replace(
        state,
        field=Field(amplitudes=amps, wavelength_grid_um=(1.20, 1.25, 1.30)),
    )
    out_state = stage.apply(state)
    payload = next(iter(out_state.results.components.values()))
    assert "convergence_history" not in payload
