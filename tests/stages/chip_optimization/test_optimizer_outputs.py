"""Published optimizer outputs: null history and pre-DAC parameters."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nullsim.pipeline import Field, Geometry, RunContext, SimulationState, Throughput
from nullsim.stages.chip_optimization import (
    _build_unitary_stack_numpy,
    _quantize_phases,
    _require_field,
)
from nullsim.stages.chip_optimization.scipy import ChipOptimizationScipy

_N_MODES = 4
_WL_UM = (1.20, 1.25, 1.30)
_POSITIONS = np.array([[-1.0, -0.5], [0.5, -0.25], [1.25, 0.75], [2.0, 1.0]])


def _context(tmp_path: Path, instance_id: str) -> RunContext:
    return RunContext(
        rng=np.random.default_rng(0),
        config_hash=f"chip-opt-outputs-{instance_id}",
        sweep_coord_tuple=(),
        instance_id=instance_id,
        run_dir=tmp_path,
    )


def _state() -> SimulationState:
    rng = np.random.default_rng(3)
    amps = (
        rng.normal(size=(_N_MODES, len(_WL_UM), 1))
        + 1j * rng.normal(size=(_N_MODES, len(_WL_UM), 1))
    ) / 4.0
    state = SimulationState.empty({"wavelength_grid_um": _WL_UM})
    return replace(
        state,
        field=Field(amplitudes=amps.astype(np.complex128), wavelength_grid_um=_WL_UM),
        throughput=Throughput(
            wavelength_grid_um=_WL_UM, transmission_fraction=(1.0,) * len(_WL_UM)
        ),
        geometry=Geometry(
            aperture_positions_pupil_m=_POSITIONS, effective_collecting_area_m2=4.0
        ),
    )


def _params(**extra: object) -> dict:
    return {
        "n_modes": _N_MODES, "n_bright_ports": 2, "n_dark_ports": 2,
        "wl_design_um": 1.25, "seed": 7, "max_iter": 40, "n_restarts": 1,
        "capture_convergence_history": True, **extra,
    }


def _payload(state: SimulationState) -> dict:
    return next(
        payload for name, payload in state.results.components.items()
        if name.startswith("chip_optimization")
    )


def _direct_null(params_vec: np.ndarray, field: np.ndarray, dac_bits: int) -> float:
    """Dark-port power over input power, band-averaged, built from the unitary."""

    u = _build_unitary_stack_numpy(
        params_vec=params_vec, n_modes=_N_MODES, wavelength_grid_um=_WL_UM,
        wl_design_um=1.25, dac_bits=dac_bits,
    )
    out = np.einsum("ijw,jwp->iwp", u, field)
    dark = np.sum(np.abs(out[2:]) ** 2, axis=(0, 2))
    total = np.sum(np.abs(field) ** 2, axis=(0, 2))
    return float(np.mean(dark) / np.mean(total))


def test_scipy_publishes_leakage_null_not_objective_with_throughput_reward(
    tmp_path: Path,
) -> None:
    # A large reward drives the objective negative; the published null and
    # its history must stay the leakage-only null depth (Issue 23).
    state = _state()
    stage = ChipOptimizationScipy(
        _params(throughput_penalty_alpha=10.0, throughput_penalty_seps_mas=[200.0],
                throughput_penalty_n_pa=4, dac_bits=6),
        _context(tmp_path, "scipy_tp"),
    )
    payload = _payload(stage.apply(state))

    objective = np.asarray(payload["convergence_history"])
    nulls = np.asarray(payload["convergence_history_normalized_null"])
    assert objective[-1, 1] < 0.0
    np.testing.assert_array_equal(nulls[:, 0], objective[:, 0])
    assert np.all(nulls[:, 1] > 0.0)

    # The optimizer works on the phase-stripped, area-scaled field.
    field = _require_field(state, _N_MODES, wl_design_um=1.25)
    opt = np.asarray(payload["optimal_chip_params"])
    expected = _direct_null(opt, field, dac_bits=6)
    assert payload["optimization_normalized_null_final"] == pytest.approx(expected, rel=1e-10)


def test_scipy_publishes_unquantized_optimum(tmp_path: Path) -> None:
    stage = ChipOptimizationScipy(_params(dac_bits=6), _context(tmp_path, "scipy_dac"))
    payload = _payload(stage.apply(_state()))

    raw = payload["optimal_chip_params_unquantized"]
    assert isinstance(raw, np.ndarray) and raw.dtype == np.float64
    np.testing.assert_array_equal(
        _quantize_phases(raw, 6), np.asarray(payload["optimal_chip_params"])
    )
    assert not np.allclose(raw, payload["optimal_chip_params"])


def test_torch_records_normalized_null_history_and_unquantized_optimum(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from nullsim.stages.chip_optimization.torch import ChipOptimizationTorch

    stage = ChipOptimizationTorch(
        _params(single_lambda_bootstrap=False, throughput_penalty_alpha=10.0,
                throughput_penalty_seps_mas=[200.0], throughput_penalty_n_pa=4),
        _context(tmp_path, "torch_tp"),
    )
    payload = _payload(stage.apply(_state()))

    objective = np.asarray(payload["convergence_history"])
    nulls = np.asarray(payload["convergence_history_normalized_null"])
    assert objective[-1, 1] < 0.0
    np.testing.assert_array_equal(nulls[:, 0], objective[:, 0])
    assert np.all(nulls[:, 1] > 0.0)
    # No DAC: the last iterate is the deployed chip, so its null is the
    # published one.
    assert nulls[-1, 1] == pytest.approx(
        payload["optimization_normalized_null_final"], rel=1e-9
    )
    raw = payload["optimal_chip_params_unquantized"]
    assert raw.dtype == np.float64
    np.testing.assert_allclose(raw, payload["optimal_chip_params"], atol=1e-12)
