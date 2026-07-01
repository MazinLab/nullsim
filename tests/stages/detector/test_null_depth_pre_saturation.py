"""Regression test for null_depth_raw masking by detector saturation (Bug #1)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nullsim.pipeline import RunContext
from nullsim.pipeline.state import Field, SimulationState, Throughput, Geometry
from nullsim.stages.detector.mkid import MKIDDetector


def _ctx() -> RunContext:
    return RunContext(
        rng=np.random.default_rng(0),
        config_hash="abc",
        sweep_coord_tuple=(),
        instance_id="detector",
        run_dir=Path("/tmp"),
    )


def _state_for_bright_star(
    n_bright: int, n_dark: int, n_wl: int = 3,
    bright_amp: complex = 1.0, dark_amp: complex = 1e-3,
) -> SimulationState:
    """State with a known chip-output: bright ports loud, dark ports dim by 1e-6 in power."""

    wl_grid = tuple(np.linspace(1.20, 1.30, n_wl).tolist())
    n_modes = n_bright + n_dark
    # Field amplitudes: bright ports at amp 1.0, dark at 1e-3 → null = 1e-6.
    amps = np.zeros((n_modes, n_wl, 1), dtype=np.complex128)
    amps[:n_bright, :, 0] = bright_amp
    amps[n_bright:, :, 0] = dark_amp

    state = SimulationState.empty(grid={"wavelength_grid_um": wl_grid})
    state = replace(state, field=Field(amplitudes=amps, wavelength_grid_um=wl_grid))
    # Throughput unit so per-port photon rate ~ |amp|^2 × stellar_rate × area
    state = replace(state, throughput=Throughput(
        wavelength_grid_um=wl_grid,
        transmission_fraction=tuple(1.0 for _ in wl_grid),
    ))
    state = replace(state, geometry=Geometry(area_m2=78.5))
    return state


def test_null_depth_unaffected_by_saturation_cap() -> None:
    """null_depth_raw should reflect the optical |U|^2 ratio, not the saturated rates."""

    # 1 bright + 3 dark; bright at amp 1.0, dark at amp 1e-3 → optical null = 1e-6.
    state = _state_for_bright_star(n_bright=1, n_dark=3)

    # First: very bright host (J=2-ish): both bright and dark heavily saturated.
    params_bright = {
        "n_modes": 4,
        "n_bright_ports": 1,
        "n_dark_ports": 3,
        "band": "J",
        "qe": 0.9,
        "mkid_max_count_rate_hz": 5.0e4,
        "n_pixels_bright": 10,  # bright cap = 10 * 50kHz = 5e5/s
        "n_pixels_dark": 10,
        "integration_time_s": 3600.0,
    }
    state_with_scene = replace(state, scene=replace(state.scene, star_jmag=2.0))
    stage = MKIDDetector(params=params_bright, context=_ctx())
    out_bright = stage.apply(state_with_scene)
    comp = next(iter(out_bright.results.components.values()))
    null_bright = comp["null_depth_raw"]

    # Second: dim host: same ratio, no saturation.
    state_dim = replace(state, scene=replace(state.scene, star_jmag=15.0))
    stage_dim = MKIDDetector(params=params_bright, context=_ctx())
    out_dim = stage_dim.apply(state_dim)
    null_dim = next(iter(out_dim.results.components.values()))["null_depth_raw"]

    # Both should report the same null (the |amplitude|^2 ratio of the
    # chip output), because the detector saturation/QE/area/stellar-rate
    # all cancel in dark/bright. The pre-fix code reported saturation-
    # dependent ratios that drifted with J-mag.
    assert null_bright == pytest.approx(null_dim, rel=1e-9)
    # And the value should be near 1e-6 (|1e-3|^2 / |1|^2), not the
    # pixel-count ratio.
    assert null_bright == pytest.approx(1e-6, rel=1e-6)
