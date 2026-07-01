"""Regression test for injection.ideal wavefront carry-over (Bug #3)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from nullsim.pipeline import RunContext
from nullsim.pipeline.state import Field, SimulationState, Wavefront
from nullsim.stages.injection.ideal import IdealInjection


def _ctx() -> RunContext:
    return RunContext(
        rng=np.random.default_rng(0),
        config_hash="abc",
        sweep_coord_tuple=(),
        instance_id="injection",
        run_dir=Path("/tmp"),
    )


def _state_with_wavefront_history(n_modes: int = 2) -> SimulationState:
    """A pre-injection state carrying atmosphere + ao_residual wavefront components."""

    wl_grid = (1.2, 1.25, 1.3)
    state = SimulationState.empty(grid={"wavelength_grid_um": wl_grid})
    amps = np.ones((n_modes, len(wl_grid), 1), dtype=np.complex128)
    state = replace(state, field=Field(amplitudes=amps, wavelength_grid_um=wl_grid))
    wavefront = Wavefront(
        wavelength_grid_um=wl_grid,
        strehl_ratio=(0.8, 0.82, 0.85),
        amplitude_jitter_fraction=(0.02, 0.02, 0.02),
    ).add_component(
        "atmosphere", {"r0_500nm_m": 0.15, "tau0_ms": 2.83},
    ).add_component(
        "ao_residual", {"residual_rms_nm": 95.0},
    )
    state = replace(state, wavefront=wavefront)
    return state


def test_injection_ideal_preserves_upstream_wavefront_components() -> None:
    state = _state_with_wavefront_history()
    assert "atmosphere" in state.wavefront.components
    assert "ao_residual" in state.wavefront.components

    stage = IdealInjection(params={"n_modes": 2}, context=_ctx())
    out = stage.apply(state)

    # All three components must be present after injection.ideal runs.
    assert "atmosphere" in out.wavefront.components
    assert "ao_residual" in out.wavefront.components
    assert "injection_ideal" in out.wavefront.components

    # Upstream payloads must be unchanged.
    assert out.wavefront.components["atmosphere"]["r0_500nm_m"] == 0.15
    assert out.wavefront.components["ao_residual"]["residual_rms_nm"] == 95.0
