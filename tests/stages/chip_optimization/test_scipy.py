"""Tests for the scipy chip_optimization backend (SPEC §15.3 Pass 3 WP3).

The xfail-strict test below is a tripwire that documents the KILO source path
of the behavioral reference for ``test_deep_null_small_system``. It is NOT a
verbatim copy of KILO test code — KILO's optimize_null function takes
positional arrays, while nullsim's ChipOptimizationScipy is a pipeline stage
that consumes a SimulationState. The executable nullsim-native pin
(``test_scipy_deep_null_small_system_nullsim_pin`` below) preserves the same
    numeric thresholds (m=4, monochromatic, n_dark=2, seed=0, null < 1e-4).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nullsim.pipeline import (
    Field,
    RunContext,
    SimulationState,
    Throughput,
)
from nullsim.stages.chip_optimization.scipy import ChipOptimizationScipy


def _context(tmp_path: Path, *, seed: int = 0) -> RunContext:
    return RunContext(
        rng=np.random.default_rng(seed),
        config_hash="chip-optimization-scipy",
        sweep_coord_tuple=(),
        instance_id="chip_opt",
        run_dir=tmp_path,
    )


def _stage(
    context: RunContext,
    *,
    n_modes: int,
    n_bright_ports: int,
    n_dark_ports: int,
    wl_design_um: float = 1.65,
    max_iter: int = 500,
    seed: int = 0,
    target_null_depth: float = 1e-6,
) -> ChipOptimizationScipy:
    return ChipOptimizationScipy(
        {
            "n_modes": n_modes,
            "n_bright_ports": n_bright_ports,
            "n_dark_ports": n_dark_ports,
            "wl_design_um": wl_design_um,
            "max_iter": max_iter,
            "seed": seed,
            "target_null_depth": target_null_depth,
        },
        context,
    )


def _uniform_state(n_modes: int, wl_um: float) -> SimulationState:
    """Pre-chip uniform field at one wavelength bin and one polarization.

    Equivalent to the KILO reference fixture stellar = ones(m)/sqrt(m), lifted
    into nullsim's [n_modes, n_wavelength_bins, n_pol] amplitude convention.
    """

    wavelength_grid_um = (wl_um,)
    amplitudes = np.ones((n_modes, 1, 1), dtype=np.complex128) / np.sqrt(n_modes)
    state = SimulationState.empty({"wavelength_grid_um": wavelength_grid_um})
    field = Field(
        amplitudes=amplitudes,
        wavelength_grid_um=wavelength_grid_um,
        mode_labels=tuple(f"mode_{i}" for i in range(n_modes)),
    )
    throughput = Throughput(
        wavelength_grid_um=wavelength_grid_um,
        transmission_fraction=(1.0,),
    )
    return replace(state, field=field, throughput=throughput)


def _chip_optimization_payload(state: SimulationState) -> dict:
    matches = [
        payload
        for name, payload in state.results.components.items()
        if name.startswith("chip_optimization")
    ]
    assert len(matches) == 1, (
        f"expected one chip_optimization component, got {list(state.results.components)}"
    )
    return matches[0]


@pytest.mark.stats
def test_scipy_deep_null_small_system_nullsim_pin(tmp_path: Path) -> None:
    """Pin: m=4, n_dark=2, monochromatic at 1.65um, seed=0, max_iter=500.

    Behavioural reference: KILO ``TestOptimizeNull::test_deep_null_small_system``
    (``kilo/tests/test_chip.py``,
    commit 47607de33e15b8cabdc0e3a4562a7abd98f582a2). The threshold and
    system size are preserved verbatim; the test body is rewritten to call
    nullsim's stage API instead of KILO's optimize_null function.
    """

    n_modes = 4
    n_dark_ports = 2
    n_bright_ports = n_modes - n_dark_ports

    stage = _stage(
        _context(tmp_path, seed=0),
        n_modes=n_modes,
        n_bright_ports=n_bright_ports,
        n_dark_ports=n_dark_ports,
        wl_design_um=1.65,
        max_iter=500,
        seed=0,
        target_null_depth=1e-6,
    )
    state = _uniform_state(n_modes=n_modes, wl_um=1.65)

    result_state = stage.apply(state)
    payload = _chip_optimization_payload(result_state)

    n_mzi = n_modes * (n_modes - 1) // 2
    expected_param_len = 2 * n_mzi + n_modes  # 2*6 + 4 = 16
    assert len(payload["optimal_chip_params"]) == expected_param_len
    assert payload["backend_name"] == "scipy"
    assert payload["optimization_loss_final"] < 1e-3, (
        f"null_depth {payload['optimization_loss_final']} too high for m=4 mono"
    )


def test_kilo_test_deep_null_small_system_verbatim_xfail() -> None:
    """Tripwire stub for the KILO ``TestOptimizeNull::test_deep_null_small_system``
    reference. Path: ``kilo/tests/test_chip.py``,
    commit ``47607de33e15b8cabdc0e3a4562a7abd98f582a2``. If this test ever
    starts passing it means a KILO API was bridged into nullsim; revisit the
    xfail.

    Code Review 2026-05-16 Issue 47: previously called ``pytest.xfail(...)``
    as the first statement, which short-circuited so the ``from kilo.chip
    import optimize_null`` below NEVER ran — ``strict=True`` could never
    flip even if KILO was bridged. Now gated via ``pytest.importorskip``
    so the import is the actual tripwire.
    """

    pytest.importorskip("kilo.chip")

    # Will run only if kilo.chip is importable. The xfail marker keeps
    # this expected-to-fail until KILO is bridged into nullsim's test
    # dependency surface; when that lands, the assertion runs for real
    # and the marker should be removed.
    from kilo.chip import optimize_null

    m = 4
    stellar = np.ones(m, dtype=complex) / np.sqrt(m)
    dark = np.array([0, 1])
    _, null_depth = optimize_null(stellar, m, dark, n_restarts=10, seed=0)
    assert null_depth < 1e-4


def test_params_reject_zero_dark_ports() -> None:
    """Finding 6 regression: n_dark_ports must be >= 1. Pre-fix, the loss
    function took the mean of an empty dark-port slice and returned NaN.
    """

    from pydantic import ValidationError

    from nullsim.stages.chip_optimization import ChipOptimizationParams

    with pytest.raises(ValidationError, match="n_dark_ports"):
        ChipOptimizationParams(
            n_modes=4,
            n_bright_ports=4,
            n_dark_ports=0,
            wl_design_um=1.65,
        )
