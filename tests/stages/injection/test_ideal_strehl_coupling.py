"""Regression tests for injection.ideal Strehl-coupling fix (Phase 0.3).

Before this fix, ``IdealInjection.apply`` seeded unit-amplitude field
amplitudes regardless of the upstream wavefront Strehl, and overwrote
``state.wavefront.strehl_ratio`` with 1.0. Both behaviors hid the
photon-budget Strehl factor from the downstream MKID stage (which
derives count rates from ``|field|²``) and from
``sensitivity.detection_curve``'s planet-throughput coupling.

These tests pin the corrected behavior: the field amplitudes carry
``√(η_opt × S × (1 − R_Fresnel))`` and the wavefront preserves the
upstream Strehl spectrum.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nullsim.physics.ruilier_cassaing import (
    FRESNEL_REFLECTION_LOSS,
    UNOBSTRUCTED_GEOMETRIC_COUPLING_LIMIT,
    coupling_efficiency,
)
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


def _state(n_modes: int = 2, strehl: tuple[float, ...] | None = None) -> SimulationState:
    wl_grid = (1.2, 1.25, 1.3)
    state = SimulationState.empty(grid={"wavelength_grid_um": wl_grid})
    amps = np.ones((n_modes, len(wl_grid), 1), dtype=np.complex128)
    state = replace(state, field=Field(amplitudes=amps, wavelength_grid_um=wl_grid))
    wavefront = Wavefront(
        wavelength_grid_um=wl_grid,
        strehl_ratio=strehl if strehl is not None else (1.0, 1.0, 1.0),
        amplitude_jitter_fraction=(0.0, 0.0, 0.0),
    )
    return replace(state, wavefront=wavefront)


def test_default_field_amplitudes_carry_strehl_x_eta_opt() -> None:
    """|field|² per mode = S × η_opt(0) at each wavelength bin."""

    strehl = (0.30, 0.50, 0.70)
    stage = IdealInjection(params={"n_modes": 2}, context=_ctx())
    out = stage.apply(_state(strehl=strehl))

    # Expected η = S × η_opt(0) (no Fresnel by default).
    expected_eta = np.array(
        [coupling_efficiency(s, 0.0, apply_fresnel_loss=False) for s in strehl]
    )
    power_per_mode = np.abs(out.field.amplitudes) ** 2  # shape (M, n_wl, 1)
    assert power_per_mode.shape == (2, 3, 1)
    # Every mode carries the same coupling factor (on-axis source).
    for m in range(2):
        np.testing.assert_allclose(power_per_mode[m, :, 0], expected_eta)


def test_injection_ideal_throughput_component_is_field_applied() -> None:
    """The injection_ideal throughput component is flagged field-applied.

    The coupling loss is carried in the field amplitudes, so stages that
    reconstruct an absolute stellar rate from the throughput budget
    (characterization.spectro) must fold it in. Before this flag the rate
    reconstruction omitted injection coupling, inflating characterization SNR
    by ~sqrt(1/eta_inj) (growing toward faint guide stars as Strehl collapses).
    """

    from nullsim.stages.characterization.spectro import _is_field_applied_throughput

    stage = IdealInjection(params={"n_modes": 2}, context=_ctx())
    out = stage.apply(_state(strehl=(0.30, 0.50, 0.70)))

    component = out.throughput.components["injection_ideal"]
    assert component["field_amplitude_applied"] is True
    assert _is_field_applied_throughput(component) is True


def test_apply_strehl_coupling_false_seeds_unit_amplitude() -> None:
    """Opt-out path: amplitudes are unit, regardless of upstream Strehl."""

    strehl = (0.30, 0.50, 0.70)
    stage = IdealInjection(
        params={"n_modes": 2, "apply_strehl_coupling": False},
        context=_ctx(),
    )
    out = stage.apply(_state(strehl=strehl))
    np.testing.assert_allclose(np.abs(out.field.amplitudes) ** 2, 1.0)


def test_fresnel_loss_multiplies_field_power() -> None:
    """Enabling Fresnel multiplies coupling by (1 − R_Fresnel) per port."""

    strehl = (1.0, 1.0, 1.0)
    stage = IdealInjection(
        params={"n_modes": 1, "apply_fresnel_injection_loss": True},
        context=_ctx(),
    )
    out = stage.apply(_state(n_modes=1, strehl=strehl))
    expected = np.array(
        [coupling_efficiency(s, 0.0, apply_fresnel_loss=True) for s in strehl]
    )
    np.testing.assert_allclose(np.abs(out.field.amplitudes[0, :, 0]) ** 2, expected)
    # Sanity: with Strehl=1, η_opt × (1 − R_Fresnel) ≈ 0.786.
    np.testing.assert_allclose(
        expected,
        UNOBSTRUCTED_GEOMETRIC_COUPLING_LIMIT * (1.0 - FRESNEL_REFLECTION_LOSS),
    )


def test_wavefront_strehl_is_preserved_not_overwritten() -> None:
    """Upstream Strehl must survive ``injection.ideal`` (bug #3 follow-on).

    The MKID stage uses ``|field|²`` for photon rate; the
    ``sensitivity.detection_curve`` stage reads ``wavefront.strehl_ratio``
    to compute its planet-throughput coupling. Both must see the AO/
    atmosphere-residual Strehl, not 1.0.
    """

    strehl = (0.30, 0.50, 0.70)
    stage = IdealInjection(params={"n_modes": 2}, context=_ctx())
    out = stage.apply(_state(strehl=strehl))
    np.testing.assert_allclose(out.wavefront.strehl_ratio, strehl)


def test_unit_strehl_with_strehl_coupling_off_and_fresnel_off_is_unit() -> None:
    """Sanity: with all the toggles off the seed is exactly unit-amplitude."""

    stage = IdealInjection(
        params={
            "n_modes": 1,
            "apply_strehl_coupling": False,
            "apply_fresnel_injection_loss": False,
        },
        context=_ctx(),
    )
    out = stage.apply(_state(n_modes=1, strehl=(1.0, 1.0, 1.0)))
    np.testing.assert_allclose(np.abs(out.field.amplitudes), 1.0)


def test_missing_upstream_strehl_falls_back_to_unity() -> None:
    """Without any upstream wavefront-quality stage, Strehl defaults to 1.

    The vertical-slice tests run injection.ideal without an atmosphere
    or AO stage; that path must continue to seed unit-coupled fields.
    """

    state = _state(n_modes=2, strehl=None)
    # Drop the strehl_ratio entirely (simulate no upstream wavefront stage).
    state = replace(
        state,
        wavefront=Wavefront(
            wavelength_grid_um=state.wavefront.wavelength_grid_um,
            strehl_ratio=(),
            amplitude_jitter_fraction=state.wavefront.amplitude_jitter_fraction,
        ),
    )
    stage = IdealInjection(params={"n_modes": 2}, context=_ctx())
    out = stage.apply(state)
    expected = np.array(
        [coupling_efficiency(1.0, 0.0, apply_fresnel_loss=False)] * 3
    )
    np.testing.assert_allclose(np.abs(out.field.amplitudes[0, :, 0]) ** 2, expected)


def test_mismatched_strehl_shape_raises() -> None:
    wl_grid = (1.2, 1.25, 1.3)
    state = SimulationState.empty(grid={"wavelength_grid_um": wl_grid})
    state = replace(
        state,
        wavefront=Wavefront(
            wavelength_grid_um=wl_grid,
            strehl_ratio=(0.5, 0.5),  # 2 bins, not 3
            amplitude_jitter_fraction=(0.0, 0.0, 0.0),
        ),
    )
    stage = IdealInjection(params={"n_modes": 2}, context=_ctx())
    with pytest.raises(ValueError, match="strehl_ratio"):
        stage.apply(state)


def test_injection_ideal_cache_key_changes_with_upstream_strehl(
    tmp_path: Path,
) -> None:
    """injection.ideal reads upstream Strehl opportunistically but historically did not fold
    state.wavefront into its cache key. AO-only sweeps could reuse a
    stale injected field. ``cache_key_extras = {"wavefront"}`` closes
    that gap without breaking HWO-space (no upstream AO) pipelines.
    """
    from nullsim.pipeline.cache import ContentCache
    from nullsim.pipeline.context import RunContext
    from nullsim.pipeline.state import SimulationState, Wavefront

    cache = ContentCache(cache_dir=tmp_path / "cache", cache_version=1)
    ctx = RunContext(
        rng=np.random.default_rng(0),
        config_hash="cfg",
        sweep_coord_tuple=(),
        instance_id="inst",
        run_dir=tmp_path,
    )
    stage = IdealInjection(
        {"apply_strehl_coupling": True, "n_modes": 4},
        ctx,
    )

    wl = (1.25,)
    base = SimulationState.empty({"wavelength_grid_um": wl})
    state_lo = replace(
        base,
        wavefront=Wavefront(
            wavelength_grid_um=wl, strehl_ratio=(0.5,)
        ),
    )
    state_hi = replace(
        base,
        wavefront=Wavefront(
            wavelength_grid_um=wl, strehl_ratio=(0.95,)
        ),
    )

    assert cache.key_for(stage, state_lo) != cache.key_for(stage, state_hi)
