"""Edge-case tests for the chip.kernel_mzi_mesh stage (SPEC §15.3 Pass 3)."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from nullsim.pipeline import (
    Field,
    RunContext,
    SimulationState,
    Throughput,
)
from nullsim.stages.chip import KernelMZIMesh


def _context(tmp_path: Path) -> RunContext:
    return RunContext(
        rng=np.random.default_rng(0),
        config_hash="kernel-mzi-mesh-edge",
        sweep_coord_tuple=(),
        instance_id="chip",
        run_dir=tmp_path,
    )


def _n_param_total(n_modes: int) -> int:
    n_mzi = n_modes * (n_modes - 1) // 2
    return 2 * n_mzi + n_modes


def _zero_params(n_modes: int) -> tuple[float, ...]:
    return tuple(0.0 for _ in range(_n_param_total(n_modes)))


def _state_from_amplitudes(
    amplitudes: np.ndarray,
    *,
    wavelength_grid_um: tuple[float, ...],
) -> SimulationState:
    state = SimulationState.empty({"wavelength_grid_um": wavelength_grid_um})
    n_modes = amplitudes.shape[0] if amplitudes.ndim >= 1 else 0
    field = Field(
        amplitudes=amplitudes,
        wavelength_grid_um=wavelength_grid_um,
        mode_labels=tuple(f"mode_{i}" for i in range(n_modes)),
    )
    throughput = Throughput(
        wavelength_grid_um=wavelength_grid_um,
        transmission_fraction=tuple(1.0 for _ in wavelength_grid_um),
    )
    return replace(state, field=field, throughput=throughput)


# --- (c) NaN propagation -----------------------------------------------------


def test_params_reject_nan() -> None:
    """allow_inf_nan=False on the Params model rejects NaN entries."""
    n_modes = 4
    bad = list(_zero_params(n_modes))
    bad[0] = math.nan
    with pytest.raises(ValidationError):
        KernelMZIMesh.Params(
            n_modes=n_modes,
            n_bright_ports=3,
            n_dark_ports=1,
            wl_design_um=1.65,
            params=tuple(bad),
        )


def test_field_with_nan_amplitude_propagates_to_output(tmp_path: Path) -> None:
    """A NaN entry in the input field amplitudes propagates to the output —
    the stage does not raise; downstream stages decide how to handle NaNs."""
    n_modes = 4
    amplitudes = np.ones((n_modes, 1, 1), dtype=np.complex128)
    amplitudes[2, 0, 0] = complex(math.nan, 0.0)
    stage = KernelMZIMesh(
        {
            "n_modes": n_modes,
            "n_bright_ports": 3,
            "n_dark_ports": 1,
            "wl_design_um": 1.65,
            "params": _zero_params(n_modes),
            "insertion_loss_db": 0.0,
        },
        _context(tmp_path),
    )
    state = _state_from_amplitudes(amplitudes, wavelength_grid_um=(1.65,))

    result = stage.apply(state)
    out = result.field.amplitudes
    assert out is not None
    assert np.isnan(out).any()


# --- (d) Pydantic validation rejections --------------------------------------


def test_params_reject_n_modes_zero() -> None:
    with pytest.raises(ValidationError, match="n_modes"):
        KernelMZIMesh.Params(
            n_modes=0,
            n_bright_ports=0,
            n_dark_ports=0,
            wl_design_um=1.65,
            params=(),
        )


def test_params_reject_n_modes_negative() -> None:
    with pytest.raises(ValidationError, match="n_modes"):
        KernelMZIMesh.Params(
            n_modes=-1,
            n_bright_ports=0,
            n_dark_ports=0,
            wl_design_um=1.65,
            params=(),
        )


def test_params_reject_port_split_mismatch() -> None:
    with pytest.raises(ValidationError, match="n_bright_ports|n_dark_ports|n_modes"):
        KernelMZIMesh.Params(
            n_modes=4,
            n_bright_ports=2,
            n_dark_ports=1,  # 2 + 1 != 4
            wl_design_um=1.65,
            params=_zero_params(4),
        )


def test_params_reject_wl_design_negative() -> None:
    with pytest.raises(ValidationError, match="wl_design_um"):
        KernelMZIMesh.Params(
            n_modes=4,
            n_bright_ports=3,
            n_dark_ports=1,
            wl_design_um=-1.0,
            params=_zero_params(4),
        )


def test_params_reject_wrong_params_length() -> None:
    with pytest.raises(ValidationError, match="params"):
        KernelMZIMesh.Params(
            n_modes=4,
            n_bright_ports=3,
            n_dark_ports=1,
            wl_design_um=1.65,
            params=(0.0, 0.0, 0.0),  # too short
        )


def test_params_reject_dac_bits_zero() -> None:
    """dac_bits=0 either fails Params validation or fails inside _quantize_phases.

    Either path is acceptable (zero-bit DAC is meaningless), so the test
    accepts both ValidationError at construction and ValueError at apply.
    """
    n_modes = 4
    try:
        params = KernelMZIMesh.Params(
            n_modes=n_modes,
            n_bright_ports=3,
            n_dark_ports=1,
            wl_design_um=1.65,
            params=_zero_params(n_modes),
            dac_bits=0,
        )
    except ValidationError:
        return
    # Construction succeeded (no ge=1 constraint on dac_bits), so the runtime
    # check inside _quantize_phases must reject it at apply time.
    assert params.dac_bits == 0
    stage = KernelMZIMesh(
        {
            "n_modes": n_modes,
            "n_bright_ports": 3,
            "n_dark_ports": 1,
            "wl_design_um": 1.65,
            "params": _zero_params(n_modes),
            "dac_bits": 0,
            "insertion_loss_db": 0.0,
        },
        RunContext(
            rng=np.random.default_rng(0),
            config_hash="dac-bits-zero",
            sweep_coord_tuple=(),
            instance_id="chip",
            run_dir=Path("."),
        ),
    )
    state = _state_from_amplitudes(
        np.ones((n_modes, 1, 1), dtype=np.complex128),
        wavelength_grid_um=(1.65,),
    )
    with pytest.raises(ValueError, match="dac_bits"):
        stage.apply(state)


# --- Field tensor shape mismatch --------------------------------------------


def test_field_2d_shape_rejected(tmp_path: Path) -> None:
    """A 2-D field tensor (missing the polarization axis) is rejected."""
    n_modes = 4
    amplitudes = np.ones((n_modes, 1), dtype=np.complex128)  # 2-D, not 3-D
    stage = KernelMZIMesh(
        {
            "n_modes": n_modes,
            "n_bright_ports": 3,
            "n_dark_ports": 1,
            "wl_design_um": 1.65,
            "params": _zero_params(n_modes),
            "insertion_loss_db": 0.0,
        },
        _context(tmp_path),
    )
    state = _state_from_amplitudes(amplitudes, wavelength_grid_um=(1.65,))
    with pytest.raises(ValueError, match="shape|n_modes"):
        stage.apply(state)


def test_field_n_modes_axis_mismatch_rejected(tmp_path: Path) -> None:
    """Field with n_modes axis differing from params.n_modes is rejected."""
    n_modes_params = 4
    n_modes_field = 5
    amplitudes = np.ones((n_modes_field, 1, 1), dtype=np.complex128)
    stage = KernelMZIMesh(
        {
            "n_modes": n_modes_params,
            "n_bright_ports": 3,
            "n_dark_ports": 1,
            "wl_design_um": 1.65,
            "params": _zero_params(n_modes_params),
            "insertion_loss_db": 0.0,
        },
        _context(tmp_path),
    )
    state = _state_from_amplitudes(amplitudes, wavelength_grid_um=(1.65,))
    with pytest.raises(ValueError, match="n_modes"):
        stage.apply(state)
