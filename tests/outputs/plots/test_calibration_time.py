"""Tests for the chip-optimizer convergence figure (paperv3 Fig 9, scoped down)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

import nullsim  # noqa: F401
from nullsim.outputs.plots.calibration_time import (
    _ConvergenceTrace,
    _extract_traces,
    plot_calibration_time,
)


def _trace(label: str, color: str, n_points: int = 10) -> _ConvergenceTrace:
    return _ConvergenceTrace(
        record_label=label, color=color,
        nfev=np.arange(n_points) * 50,
        normalized_null=np.logspace(0, -10, n_points),
        final_null=1e-10,
    )


def test_plot_uses_normalized_null_not_objective() -> None:
    """The objective goes negative with the throughput reward; the log axis
    must show the published normalized null instead (review Issue 23)."""

    payload = {
        "convergence_history": ((0, 1e-3), (10, -5e-3)),
        "convergence_history_normalized_null": ((0, 1e-2), (10, 4e-6)),
        "optimization_normalized_null_final": 3.5e-6,
    }
    record = SimpleNamespace(
        result=SimpleNamespace(state=SimpleNamespace(results=SimpleNamespace(
            components={"chip_optimization": payload},
        ))),
        config=SimpleNamespace(grid=SimpleNamespace(wavelength_center_um=1.25)),
    )

    (trace,) = _extract_traces(SimpleNamespace(records=[record]))
    np.testing.assert_array_equal(trace.normalized_null, [1e-2, 4e-6])
    assert trace.final_null == 3.5e-6

    fig = plot_calibration_time([trace])
    ax = fig.axes[0]
    assert "Normalized null" in ax.get_ylabel()
    floor = [line for line in ax.get_lines() if line.get_label().startswith("Best floor")]
    assert floor and floor[0].get_ydata()[0] == 3.5e-6


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
    null_history = payload["convergence_history_normalized_null"]
    assert [n for n, _ in null_history] == [n for n, _ in history]


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
    assert "convergence_history_normalized_null" not in payload
