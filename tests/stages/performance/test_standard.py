"""Tests for the StandardPerformance stage (SPEC §15.3 Pass 4c).

Reference invariants come from KILO ``performance.py`` lines 204-1008 in
``kilo/performance.py``:

- ``signal_to_noise`` (lines 204-292) — headline SNR formula
- ``integration_time_for_snr`` (lines 802-894) — inverse solver
- ``contrast_curve`` (lines 896-1008) — per-separation minimum detectable

See ``tests/stages/notes/performance.md`` for the full source survey
including the KILO/NAYRA divergence and the
``detected_stellar_rate_at_telescope`` consolidation lesson.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from nullsim.pipeline import (
    OPDBudget,
    OPDComponent,
    PhotonRates,
    RunContext,
    Scene,
    SimulationState,
)
from nullsim.stages.performance import StandardPerformance


def _context(tmp_path: Path) -> RunContext:
    return RunContext(
        rng=np.random.default_rng(0),
        config_hash="performance-standard-pinning",
        sweep_coord_tuple=(),
        instance_id="performance",
        run_dir=tmp_path,
    )


def _state(
    *,
    bright_rate_hz: float = 1.0e8,
    leakage_rate_hz: float = 1.0e2,
    n_dark_ports: int = 3,
    n_spectro_ports: int = 0,
    spectro_rate_hz: float = 0.0,
    integration_time_s: float = 3600.0,
    planet_contrast: float = 1.0e-6,
    planet_separation_mas: float | None = 100.0,
    star_jmag: float = 5.0,
    include_fringe_residual: bool = False,
    include_calibration_component: bool = False,
    null_depth_calibrated: float = 1.0e-7,
    null_depth_raw: float = 1.5e-7,
) -> SimulationState:
    state = SimulationState.empty({"wavelength_grid_um": (1.25,)})
    state = replace(
        state,
        scene=Scene(star_jmag=star_jmag, planet_contrast=planet_contrast),
    )
    state = replace(
        state,
        scene=replace(
            state.scene,
            planet_separation_mas=planet_separation_mas,
        ),
    )

    # Single bright port carrying bright_rate_hz; dark ports each carry
    # leakage_rate_hz / n_dark_ports so the total is leakage_rate_hz.
    port_rates = np.empty(1 + n_dark_ports + n_spectro_ports, dtype=np.float64)
    port_rates[0] = bright_rate_hz
    port_rates[1 : 1 + n_dark_ports] = leakage_rate_hz / n_dark_ports
    if n_spectro_ports:
        port_rates[1 + n_dark_ports :] = spectro_rate_hz / n_spectro_ports
    port_labels = (
        ("bright_0",)
        + tuple(f"dark_{i}" for i in range(n_dark_ports))
        + tuple(f"spectro_{i}" for i in range(n_spectro_ports))
    )

    photon_rates = PhotonRates(
        wavelength_grid_um=(1.25,),
        port_labels=port_labels,
        rates_hz=port_rates.reshape(-1, 1),
    )
    state = replace(state, photon_rates=photon_rates)

    results = state.results.add_component(
        "detector.mkid",
        {
            "null_depth_raw": float(null_depth_raw),
            "integration_time_s": float(integration_time_s),
            "port_rates": port_rates,
            "port_labels": port_labels,
        },
    )
    if include_calibration_component:
        results = results.add_component(
            "calibration.floor_lookup",
            {
                "null_depth_raw": float(null_depth_raw),
                "null_depth_residual": float(null_depth_calibrated),
                "null_depth_calibrated": float(null_depth_calibrated),
                "calibration_floor": float(null_depth_raw - null_depth_calibrated),
                "band": "J",
            },
        )
    state = replace(state, results=results)

    if include_fringe_residual:
        opd_budget = OPDBudget(
            components={
                "fringe_tracking.closed_loop_residual": OPDComponent(
                    rms_opd_nm=100.0,
                    reduction_stage="fringe_tracking.closed_loop",
                )
            },
            total_rms_opd_nm=100.0,
        )
        state = replace(state, opd_budget=opd_budget)

    return state


def _stage(
    context: RunContext,
    **overrides: object,
) -> StandardPerformance:
    return StandardPerformance(dict(overrides), context)


def test_performance_writes_snr_and_integration_time(tmp_path: Path) -> None:
    state = _state(
        bright_rate_hz=1.0e8,
        leakage_rate_hz=1.0e2,
        integration_time_s=600.0,
    )
    stage = _stage(_context(tmp_path))
    result = stage.apply(state)
    component = result.results.components["performance.standard"]

    assert math.isfinite(component["snr"])
    assert component["snr"] >= 0.0
    assert component["integration_time_for_target_snr_s"] > 0.0
    assert math.isfinite(component["integration_time_for_target_snr_s"])
    # Results dataclass fields also mirror the headline scalars.
    assert result.results.snr == pytest.approx(component["snr"])
    assert result.results.null_depth == pytest.approx(component["null_depth_residual"])


def test_performance_ignores_spectro_port_for_dark_snr_partition(
    tmp_path: Path,
) -> None:
    state = _state(
        bright_rate_hz=1.0e8,
        leakage_rate_hz=30.0,
        n_dark_ports=3,
        n_spectro_ports=1,
        spectro_rate_hz=1.0e9,
        integration_time_s=10.0,
    )
    stage = _stage(
        _context(tmp_path),
        n_sky_background_hz=0.0,
        n_dark_count_hz=0.0,
        compute_contrast_curve=False,
    )

    result = stage.apply(state)
    component = result.results.components["performance.standard"]

    assert component["n_dark_ports"] == 3
    assert component["leakage_rate_hz_total"] == pytest.approx(30.0)
    assert component["spectro_rate_hz_total"] == pytest.approx(1.0e9)
    assert component["stellar_rate_hz_total"] == pytest.approx(1.1e9 + 30.0)
    assert component["B"] == pytest.approx(300.0)


def test_performance_contrast_curve_includes_spectro_stellar_rate(
    tmp_path: Path,
) -> None:
    state = _state(
        bright_rate_hz=1.0e10,
        leakage_rate_hz=0.0,
        n_dark_ports=1,
        n_spectro_ports=1,
        spectro_rate_hz=1.0e10,
        integration_time_s=3600.0,
        null_depth_raw=0.0,
    )
    stage = _stage(
        _context(tmp_path),
        contrast_curve_separations_mas=(100.0,),
        n_sky_background_hz=0.0,
        n_dark_count_hz=0.0,
        planet_dark_throughput=0.5,
        target_snr=5.0,
        with_fringe_tracking=False,
    )

    component = stage.apply(state).results.components["performance.standard"]
    actual_contrast = float(component["contrast_curve_value"][0])

    expected_contrast = 25.0 / ((2.0e10) * 0.5 * 3600.0)
    assert component["stellar_rate_hz_total"] == pytest.approx(2.0e10)
    np.testing.assert_allclose(actual_contrast, expected_contrast, rtol=1.0e-9)


def test_performance_uses_upstream_background_component(tmp_path: Path) -> None:
    state = _state(
        bright_rate_hz=1.0e8,
        leakage_rate_hz=0.0,
        n_dark_ports=3,
        integration_time_s=10.0,
    )
    results = state.results.add_component(
        "background.nayra_sky",
        {"background_rate_hz_per_dark_port": 2.0},
    )
    state = replace(state, results=results)
    stage = _stage(
        _context(tmp_path),
        n_sky_background_hz=0.0,
        n_dark_count_hz=0.0,
        compute_contrast_curve=False,
    )

    result = stage.apply(state)
    component = result.results.components["performance.standard"]

    assert component["Sky"] == pytest.approx(2.0 * 3 * 10.0)
    assert component["sky_rate_hz_per_port"] == pytest.approx(2.0)
    assert component["sky_rate_source"] == "background.nayra_sky"


def test_performance_contrast_curve_uses_per_separation_background(
    tmp_path: Path,
) -> None:
    state = _state(
        bright_rate_hz=1.0e8,
        leakage_rate_hz=0.0,
        n_dark_ports=1,
        integration_time_s=10.0,
        null_depth_raw=0.0,
    )
    results = state.results.add_component(
        "background.space_zodi",
        {
            "background_rate_hz_per_dark_port": 0.0,
            "background_separations_mas": (50.0, 100.0),
            "background_rate_hz_per_dark_port_per_sep": np.array(
                [0.0, 1.0e6],
                dtype=np.float64,
            ),
        },
    )
    state = replace(state, results=results)
    stage = _stage(
        _context(tmp_path),
        contrast_curve_separations_mas=(50.0, 100.0),
        contrast_curve_reference_time_s=10.0,
        n_sky_background_hz=0.0,
        n_dark_count_hz=0.0,
        planet_dark_throughput=0.5,
        target_snr=5.0,
        with_fringe_tracking=False,
    )

    component = stage.apply(state).results.components["performance.standard"]
    contrasts = np.asarray(component["contrast_curve_value"], dtype=np.float64)

    assert component["Sky"] == pytest.approx(0.0)
    assert (
        component["contrast_curve_sky_rate_source"]
        == "background.space_zodi.background_rate_hz_per_dark_port_per_sep"
    )
    np.testing.assert_allclose(
        component["contrast_curve_sky_rate_hz_per_port"],
        [0.0, 1.0e6],
    )
    assert contrasts[1] > contrasts[0]


def test_performance_uses_detector_dark_rate_when_param_zero(tmp_path: Path) -> None:
    state = _state(
        leakage_rate_hz=0.0,
        n_dark_ports=3,
        integration_time_s=10.0,
    )
    detector = dict(state.results.components["detector.mkid"])
    detector["dark_rate_hz_per_port"] = 7.0
    components = dict(state.results.components)
    components["detector.mkid"] = detector
    state = replace(state, results=replace(state.results, components=components))
    stage = _stage(
        _context(tmp_path),
        n_dark_count_hz=0.0,
        n_sky_background_hz=0.0,
        compute_contrast_curve=False,
    )

    component = stage.apply(state).results.components["performance.standard"]

    assert component["D"] == pytest.approx(7.0 * 3 * 10.0)
    assert component["dark_rate_hz_per_port"] == pytest.approx(7.0)
    assert component["dark_rate_source"] == "detector.dark_rate_hz_per_port"


@pytest.mark.stats
def test_performance_photon_noise_limited_regime(tmp_path: Path) -> None:
    # High planet rate, ~zero leakage / sky / dark, eps=very small so the
    # systematic is negligible. The SNR should converge to sqrt(S).
    state = _state(
        bright_rate_hz=1.0e10,
        leakage_rate_hz=0.0,
        integration_time_s=1.0,
        planet_contrast=1.0,  # planet ~= 0.5e10 counts/s after dark throughput
    )
    stage = _stage(
        _context(tmp_path),
        n_sky_background_hz=0.0,
        n_dark_count_hz=0.0,
        compute_contrast_curve=False,
        with_fringe_tracking=False,
    )
    result = stage.apply(state)
    component = result.results.components["performance.standard"]
    expected_S = component["S"]
    expected_snr = math.sqrt(expected_S)
    assert component["snr"] == pytest.approx(expected_snr, rel=1.0e-6)


@pytest.mark.stats
def test_performance_systematic_noise_limited_ceiling(tmp_path: Path) -> None:
    # High leakage rate, no fringe tracking, large integration time —
    # SNR plateaus at S / (eps * B), independent of t.
    eps = 1.0e-3
    bright = 1.0e8
    leak = 1.0e6
    planet_contrast = 1.0e-4
    state = _state(
        bright_rate_hz=bright,
        leakage_rate_hz=leak,
        integration_time_s=1.0e6,
        planet_contrast=planet_contrast,
    )
    stage = _stage(
        _context(tmp_path),
        residual_speckle_stability_factor=eps,
        with_fringe_tracking=False,
        compute_contrast_curve=False,
    )
    result = stage.apply(state)
    component = result.results.components["performance.standard"]

    # In the ceiling-dominated limit, SNR -> S_count / (eps * B_count) where
    # both grow with t. After cancellation: S_rate / (eps * B_rate).
    planet_rate_default = bright * planet_contrast * 0.5  # planet_dark_throughput default
    expected_ceiling = planet_rate_default / (eps * leak)
    assert component["snr"] == pytest.approx(expected_ceiling, rel=1.0e-3)


def test_performance_residual_speckle_stability_adds_systematic_variance(
    tmp_path: Path,
) -> None:
    state = _state(
        bright_rate_hz=1.0e8,
        leakage_rate_hz=100.0,
        integration_time_s=10.0,
        planet_contrast=1.0e-6,
    )
    stage = _stage(
        _context(tmp_path),
        residual_speckle_stability_factor=1.0e-2,
        compute_contrast_curve=False,
        with_fringe_tracking=False,
        n_sky_background_hz=0.0,
        n_dark_count_hz=0.0,
    )

    component = stage.apply(state).results.components["performance.standard"]

    assert component["eps_systematic_used"] == pytest.approx(1.0e-2)
    assert component["residual_speckle_stability_factor"] == pytest.approx(1.0e-2)
    assert component["systematic_var"] == pytest.approx((1.0e-2 * 1000.0) ** 2)


@pytest.mark.stats
def test_performance_integration_time_monotonic_in_target_snr(tmp_path: Path) -> None:
    state = _state(
        bright_rate_hz=1.0e8,
        leakage_rate_hz=1.0e2,
        integration_time_s=1.0,
    )
    times: list[float] = []
    for target in (3.0, 5.0, 7.0, 10.0):
        stage = _stage(
            _context(tmp_path),
            target_snr=target,
            compute_contrast_curve=False,
        )
        result = stage.apply(state)
        times.append(
            result.results.components["performance.standard"]["integration_time_for_target_snr_s"]
        )
    for earlier, later in zip(times, times[1:]):
        assert later > earlier


@pytest.mark.stats
def test_performance_ft_reduces_systematic_variance(tmp_path: Path) -> None:
    state = _state(
        bright_rate_hz=1.0e8,
        leakage_rate_hz=1.0e3,
        integration_time_s=3600.0,
    )
    systematic_eps = 1.0e-3
    stage_no_ft = _stage(
        _context(tmp_path),
        residual_speckle_stability_factor=systematic_eps,
        with_fringe_tracking=False,
    )
    stage_with_ft = _stage(
        _context(tmp_path),
        residual_speckle_stability_factor=systematic_eps,
        with_fringe_tracking=True,
        default_servo_bw_hz=2000.0,
    )
    sys_no_ft = stage_no_ft.apply(state).results.components[
        "performance.standard"
    ]["systematic_var"]
    sys_with_ft = stage_with_ft.apply(state).results.components[
        "performance.standard"
    ]["systematic_var"]
    assert sys_with_ft < sys_no_ft


def test_performance_empty_separation_tuple(tmp_path: Path) -> None:
    state = _state()
    stage = _stage(_context(tmp_path), contrast_curve_separations_mas=())
    component = stage.apply(state).results.components["performance.standard"]
    assert component["contrast_curve_value"] == ()
    assert component["contrast_curve_mas"] == ()


def test_performance_contrast_curve_uses_aggregate_stellar_rate(
    tmp_path: Path,
) -> None:
    """Regression test for Tier 1.2 contrast-curve denominator fix.

    KILO ``contrast_curve`` (kilo/performance.py L968-1008) uses the *total*
    stellar photon rate in the denominator ``c = signal / (stellar × throughput × t)``.
    nullsim previously divided by ``n_dark_ports``, giving contrasts
    ``n_dark_ports`` × tighter (~18× for Keck-only). The fix passes the
    aggregate bright_rate_hz directly. This test pins that convention by
    asserting the analytic photon-noise-limited contrast.
    """
    integration_time_s = 3600.0
    bright_rate_hz = 1.0e10  # photon-noise-limited regime
    leakage_rate_hz = 0.0
    n_dark_ports = 3
    planet_dark_throughput = 0.5
    target_snr = 5.0
    state = _state(
        bright_rate_hz=bright_rate_hz,
        leakage_rate_hz=leakage_rate_hz,
        n_dark_ports=n_dark_ports,
        integration_time_s=integration_time_s,
        null_depth_raw=0.0,
    )
    stage = _stage(
        _context(tmp_path),
        contrast_curve_separations_mas=(100.0,),
        planet_dark_throughput=planet_dark_throughput,
        target_snr=target_snr,
        with_fringe_tracking=False,
    )
    component = stage.apply(state).results.components["performance.standard"]
    actual_contrast = float(component["contrast_curve_value"][0])

    # Photon-limited analytic: bg_variance ≈ 0 → signal_counts = snr²,
    #   contrast = snr² / (bright × throughput × t).
    expected_contrast = (target_snr * target_snr) / (
        bright_rate_hz * planet_dark_throughput * integration_time_s
    )
    np.testing.assert_allclose(actual_contrast, expected_contrast, rtol=1.0e-9)


def test_performance_contrast_curve_uses_reference_time(tmp_path: Path) -> None:
    state = _state(
        bright_rate_hz=1.0e10,
        leakage_rate_hz=0.0,
        integration_time_s=60.0,
        null_depth_raw=0.0,
    )
    stage = _stage(
        _context(tmp_path),
        contrast_curve_separations_mas=(100.0,),
        contrast_curve_reference_time_s=3600.0,
        planet_dark_throughput=0.5,
        target_snr=5.0,
        with_fringe_tracking=False,
    )

    component = stage.apply(state).results.components["performance.standard"]
    actual_contrast = float(component["contrast_curve_value"][0])

    expected_contrast = 25.0 / (1.0e10 * 0.5 * 3600.0)
    assert component["integration_time_s"] == pytest.approx(60.0)
    assert component["contrast_curve_reference_time_s"] == pytest.approx(3600.0)
    np.testing.assert_allclose(actual_contrast, expected_contrast, rtol=1.0e-9)


def test_performance_contrast_curve_defaults_to_detector_integration_time(
    tmp_path: Path,
) -> None:
    state = _state(
        bright_rate_hz=1.0e10,
        leakage_rate_hz=0.0,
        integration_time_s=14400.0,
        null_depth_raw=0.0,
    )
    stage = _stage(
        _context(tmp_path),
        contrast_curve_separations_mas=(100.0,),
        planet_dark_throughput=0.5,
        target_snr=5.0,
        with_fringe_tracking=False,
    )

    component = stage.apply(state).results.components["performance.standard"]

    expected_contrast = 25.0 / (1.0e10 * 0.5 * 14400.0)
    assert component["contrast_curve_reference_time_s"] == pytest.approx(14400.0)
    assert component["contrast_curve_reference_time_source"] == (
        "detector.integration_time_s"
    )
    np.testing.assert_allclose(
        float(component["contrast_curve_value"][0]),
        expected_contrast,
        rtol=1.0e-9,
    )


def test_performance_uses_calibration_component_when_present(tmp_path: Path) -> None:
    state = _state(
        include_calibration_component=True,
        null_depth_calibrated=2.5e-7,
        null_depth_raw=3.0e-7,
    )
    stage = _stage(_context(tmp_path))
    result = stage.apply(state)
    component = result.results.components["performance.standard"]
    assert component["null_depth_residual"] == pytest.approx(2.5e-7)
    assert component["null_depth_calibrated"] == pytest.approx(2.5e-7)
    assert component["null_depth_source"] == "calibration.floor_lookup"
    assert component["leakage_rate_hz_total"] == pytest.approx(1.0e2)
    # KILO mean(dark)/mean(bright) converted to a dark/total fraction
    # before the rate product: 2.5e-7 x mean_bright(1e8/1) x n_dark(3)
    # (review 2026-06-10 Issue 27a; the old pin tested the unconverted
    # mixed-convention product).
    assert component["leakage_rate_hz"] == pytest.approx(
        2.5e-7 * 1.0e8 * 3.0,
    )


def test_performance_calibration_floor_bounds_detector_null_without_sensitivity(
    tmp_path: Path,
) -> None:
    state = _state(
        include_calibration_component=True,
        null_depth_calibrated=2.5e-7,
        null_depth_raw=8.0e-7,
    )
    stage = _stage(_context(tmp_path))

    with pytest.warns(RuntimeWarning, match="applied calibration_floor"):
        result = stage.apply(state)
    component = result.results.components["performance.standard"]

    assert component["null_depth_residual"] == pytest.approx(2.5e-7)
    assert component["null_depth_effective"] == pytest.approx(5.5e-7)
    assert component["calibration_floor"] == pytest.approx(5.5e-7)
    assert component["calibration_floor_source"] == "calibration.floor_lookup"
    assert component["calibration_floor_applied"] is True
    assert (
        component["null_depth_source"]
        == "calibration_floor:calibration.floor_lookup"
    )
    # Converted convention (review 2026-06-10 Issue 27a): 5.5e-7 x
    # mean_bright(1e8/1) x n_dark(3).
    assert component["leakage_rate_hz"] == pytest.approx(
        5.5e-7 * 1.0e8 * 3.0,
    )
    assert result.results.null_depth == pytest.approx(5.5e-7)


def test_performance_calibration_floor_bounds_sensitivity_null(
    tmp_path: Path,
) -> None:
    state = _state(
        include_calibration_component=True,
        null_depth_calibrated=5.0e-7,
        null_depth_raw=1.5e-6,
    )
    results = state.results.add_component(
        "sensitivity.detection_curve",
        {
            "effective_null_with_disk": 1.0e-7,
            "planet_throughputs": np.array([0.5], dtype=np.float64),
            "separations_mas": (100.0,),
            "null_instability": 0.0,
        },
    )
    state = replace(state, results=results)
    stage = _stage(
        _context(tmp_path),
        compute_contrast_curve=False,
        n_sky_background_hz=0.0,
        n_dark_count_hz=0.0,
    )

    with pytest.warns(RuntimeWarning, match="applied calibration_floor"):
        result = stage.apply(state)
    component = result.results.components["performance.standard"]

    expected_floor_fraction = 1.0e-6 * 1.0e8 * 3.0 / (1.0e8 + 1.0e2)
    assert component["null_depth_residual"] == pytest.approx(5.0e-7)
    assert component["null_depth_effective"] == pytest.approx(expected_floor_fraction)
    assert component["calibration_floor"] == pytest.approx(1.0e-6)
    assert component["calibration_floor_source"] == "calibration.floor_lookup"
    assert component["calibration_floor_applied"] is True
    assert (
        component["null_depth_source"]
        == "calibration_floor:calibration.floor_lookup"
    )
    assert component["leakage_rate_hz"] == pytest.approx(
        expected_floor_fraction * (1.0e8 + 1.0e2),
    )
    assert result.results.null_depth == pytest.approx(expected_floor_fraction)


def test_performance_detector_residual_bounds_sensitivity_null(
    tmp_path: Path,
) -> None:
    state = _state(null_depth_raw=3.0e-6, include_calibration_component=False)
    results = state.results.add_component(
        "sensitivity.detection_curve",
        {
            "effective_null_with_disk": 1.0e-7,
            "planet_throughputs": np.array([0.5], dtype=np.float64),
            "separations_mas": (100.0,),
            "null_instability": 0.0,
        },
    )
    state = replace(state, results=results)
    stage = _stage(
        _context(tmp_path),
        compute_contrast_curve=False,
        n_sky_background_hz=0.0,
        n_dark_count_hz=0.0,
    )

    component = stage.apply(state).results.components["performance.standard"]

    expected_detector_fraction = 3.0e-6 * 1.0e8 * 3.0 / (1.0e8 + 1.0e2)
    assert component["null_depth_effective"] == pytest.approx(
        expected_detector_fraction
    )
    assert component["leakage_rate_hz"] == pytest.approx(
        expected_detector_fraction * (1.0e8 + 1.0e2),
    )
    assert (
        component["null_depth_source"]
        == "detector.null_depth_raw bounds sensitivity:sensitivity.detection_curve"
    )


def test_performance_ignores_nan_detector_residual_when_sensitivity_present(
    tmp_path: Path,
) -> None:
    state = _state(
        leakage_rate_hz=0.0,
        null_depth_raw=float("nan"),
        include_calibration_component=False,
    )
    results = state.results.add_component(
        "sensitivity.detection_curve",
        {
            "effective_null_with_disk": 2.0e-7,
            "planet_throughputs": np.array([0.5], dtype=np.float64),
            "separations_mas": (100.0,),
            "null_instability": 0.0,
        },
    )
    state = replace(state, results=results)
    stage = _stage(
        _context(tmp_path),
        compute_contrast_curve=False,
        n_sky_background_hz=0.0,
        n_dark_count_hz=0.0,
    )

    component = stage.apply(state).results.components["performance.standard"]

    assert component["null_depth_effective"] == pytest.approx(2.0e-7)
    assert component["null_depth_residual_effective_fraction"] is None
    assert component["null_depth_source"] == "sensitivity:sensitivity.detection_curve"


def test_performance_contrast_curve_allows_missing_detector_residual(
    tmp_path: Path,
) -> None:
    state = _state(
        bright_rate_hz=1.0e8,
        leakage_rate_hz=0.0,
        integration_time_s=3600.0,
        planet_contrast=1.0e-6,
        planet_separation_mas=10.0,
        null_depth_raw=float("nan"),
        include_calibration_component=False,
    )
    results = state.results.add_component(
        "sensitivity.detection_curve",
        {
            "effective_null_with_disk": 1.0e-10,
            "effective_null_with_disk_by_separation": np.array(
                [1.0e-10, 1.0e-6],
                dtype=np.float64,
            ),
            "planet_throughputs": np.array([0.5, 0.5], dtype=np.float64),
            "separations_mas": (10.0, 100.0),
            "null_instability": 0.0,
        },
    )
    state = replace(state, results=results)
    stage = _stage(
        _context(tmp_path),
        compute_contrast_curve=True,
        contrast_curve_reference_time_s=3600.0,
        n_sky_background_hz=0.0,
        n_dark_count_hz=0.0,
        with_fringe_tracking=False,
    )

    component = stage.apply(state).results.components["performance.standard"]

    np.testing.assert_allclose(
        component["contrast_curve_leakage_rate_hz"],
        [1.0e-2, 1.0e2],
        rtol=1.0e-12,
    )


def test_performance_headline_snr_uses_sensitivity_throughput_at_planet_sep(
    tmp_path: Path,
) -> None:
    state = _state(
        bright_rate_hz=1.0e8,
        leakage_rate_hz=0.0,
        n_spectro_ports=1,
        spectro_rate_hz=3.0e8,
        integration_time_s=1.0,
        planet_contrast=1.0,
        planet_separation_mas=75.0,
        null_depth_raw=0.0,
    )
    results = state.results.add_component(
        "sensitivity.detection_curve",
        {
            "effective_null_with_disk": 0.0,
            "planet_throughputs": np.array([0.2, 0.6], dtype=np.float64),
            "separations_mas": (50.0, 100.0),
            "null_instability": 0.0,
        },
    )
    state = replace(state, results=results)
    stage = _stage(
        _context(tmp_path),
        planet_dark_throughput=0.9,
        compute_contrast_curve=False,
        n_sky_background_hz=0.0,
        n_dark_count_hz=0.0,
        with_fringe_tracking=False,
    )

    component = stage.apply(state).results.components["performance.standard"]

    assert component["planet_dark_throughput_used"] == pytest.approx(0.4)
    assert component["planet_separation_mas"] == pytest.approx(75.0)
    assert (
        component["planet_dark_throughput_source"]
        == "sensitivity:sensitivity.detection_curve.planet_throughputs"
    )
    assert component["planet_rate_base_hz"] == pytest.approx(4.0e8)
    assert component["planet_rate_hz"] == pytest.approx(1.6e8)
    assert component["S"] == pytest.approx(1.6e8)
    assert component["snr"] == pytest.approx(math.sqrt(1.6e8), rel=1.0e-9)


def test_performance_contrast_curve_uses_sensitivity_null_by_separation(
    tmp_path: Path,
) -> None:
    state = _state(
        bright_rate_hz=1.0e8,
        leakage_rate_hz=0.0,
        integration_time_s=3600.0,
        planet_contrast=1.0e-6,
        planet_separation_mas=10.0,
        null_depth_raw=0.0,
    )
    results = state.results.add_component(
        "sensitivity.detection_curve",
        {
            "effective_null_with_disk": 1.0e-10,
            "effective_null_with_disk_by_separation": np.array(
                [1.0e-10, 1.0e-6],
                dtype=np.float64,
            ),
            "planet_throughputs": np.array([0.5, 0.5], dtype=np.float64),
            "separations_mas": (10.0, 100.0),
            "null_instability": 0.0,
        },
    )
    state = replace(state, results=results)
    stage = _stage(
        _context(tmp_path),
        compute_contrast_curve=True,
        contrast_curve_reference_time_s=3600.0,
        n_sky_background_hz=0.0,
        n_dark_count_hz=0.0,
        with_fringe_tracking=False,
    )

    component = stage.apply(state).results.components["performance.standard"]

    np.testing.assert_allclose(
        component["contrast_curve_leakage_rate_hz"],
        [1.0e-2, 1.0e2],
        rtol=1.0e-12,
    )
    contrast = np.asarray(component["contrast_curve_value"], dtype=np.float64)
    assert contrast[1] > 50.0 * contrast[0]


def test_performance_combines_speckle_eps_with_null_instability(
    tmp_path: Path,
) -> None:
    state = _state(
        bright_rate_hz=1.0e8,
        leakage_rate_hz=1.0e2,
        integration_time_s=10.0,
    )
    results = state.results.add_component(
        "sensitivity.detection_curve",
        {
            "effective_null_with_disk": 1.0e-6,
            "planet_throughputs": np.array([0.5], dtype=np.float64),
            "separations_mas": (100.0,),
            "null_instability": 3.0e-3,
        },
    )
    state = replace(state, results=results)
    stage = _stage(
        _context(tmp_path),
        residual_speckle_stability_factor=4.0e-3,
        compute_contrast_curve=False,
        with_fringe_tracking=False,
    )

    component = stage.apply(state).results.components["performance.standard"]

    assert component["residual_speckle_stability_factor"] == pytest.approx(4.0e-3)
    assert component["null_instability"] == pytest.approx(3.0e-3)
    assert component["eps_systematic_used"] == pytest.approx(5.0e-3)


def test_nsc_reduction_divides_instability_only(tmp_path: Path) -> None:
    state = _state(
        bright_rate_hz=1.0e8,
        leakage_rate_hz=1.0e2,
        integration_time_s=10.0,
    )
    results = state.results.add_component(
        "sensitivity.detection_curve",
        {
            "effective_null_with_disk": 1.0e-6,
            "planet_throughputs": np.array([0.5], dtype=np.float64),
            "separations_mas": (100.0,),
            "null_instability": 5.0e-3,
        },
    )
    state = replace(state, results=results)
    stage = _stage(
        _context(tmp_path),
        residual_speckle_stability_factor=0.0,
        nsc_systematic_reduction=10.0,
        compute_contrast_curve=False,
        with_fringe_tracking=False,
    )

    component = stage.apply(state).results.components["performance.standard"]

    assert component["eps_systematic_used"] == pytest.approx(5.0e-4, rel=1.0e-6)
    assert component["null_instability"] == pytest.approx(5.0e-3)
    assert component["nsc_systematic_reduction"] == pytest.approx(10.0)


def test_performance_falls_back_to_raw_null_when_no_calibration(tmp_path: Path) -> None:
    state = _state(null_depth_raw=4.2e-7, include_calibration_component=False)
    stage = _stage(_context(tmp_path))
    result = stage.apply(state)
    component = result.results.components["performance.standard"]
    assert component["null_depth_residual"] == pytest.approx(4.2e-7)
    assert component["null_depth_calibrated"] == pytest.approx(4.2e-7)
    assert component["null_depth_source"] == "detector.null_depth_raw"


def test_performance_records_ft_residual_provenance(tmp_path: Path) -> None:
    state = _state(include_fringe_residual=True)
    stage = _stage(_context(tmp_path))
    result = stage.apply(state)
    component = result.results.components["performance.standard"]
    assert "FT residual" in component["servo_bw_source"]


def test_performance_zero_planet_signal_returns_zero_snr_inf_time(tmp_path: Path) -> None:
    # planet_contrast = 0 → planet_rate = 0 → SNR = 0 and t_for_target = inf.
    state = _state(planet_contrast=0.0)
    stage = _stage(_context(tmp_path))
    result = stage.apply(state)
    component = result.results.components["performance.standard"]
    assert component["snr"] == 0.0
    assert math.isinf(component["integration_time_for_target_snr_s"])


def test_performance_missing_planet_contrast_raises(tmp_path: Path) -> None:
    state = _state()
    state = replace(state, scene=replace(state.scene, planet_contrast=None))
    stage = _stage(_context(tmp_path))

    with pytest.raises(ValueError, match="scene.planet_contrast"):
        stage.apply(state)


def test_performance_missing_detector_component_raises(tmp_path: Path) -> None:
    state = SimulationState.empty({"wavelength_grid_um": (1.25,)})
    stage = _stage(_context(tmp_path))
    with pytest.raises(ValueError, match="detector"):
        stage.apply(state)


def test_performance_target_snr_above_ceiling_returns_inf(tmp_path: Path) -> None:
    # No FT, very high leakage, low planet rate, target SNR above the
    # ceiling S/(eps*B) → integration time = inf.
    state = _state(
        bright_rate_hz=1.0e8,
        leakage_rate_hz=1.0e6,
        integration_time_s=1.0,
        planet_contrast=1.0e-6,
    )
    stage = _stage(
        _context(tmp_path),
        residual_speckle_stability_factor=1.0e-2,
        target_snr=1.0e6,
        with_fringe_tracking=False,
        compute_contrast_curve=False,
    )
    component = stage.apply(state).results.components["performance.standard"]
    assert math.isinf(component["integration_time_for_target_snr_s"])


def test_performance_bad_residual_speckle_stability_rejected_by_params() -> None:
    with pytest.raises(ValidationError):
        StandardPerformance.Params(residual_speckle_stability_factor=-1.0e-6)
    with pytest.raises(ValidationError):
        StandardPerformance.Params(residual_speckle_stability_factor=1.01)


def test_performance_bad_target_snr_rejected_by_params() -> None:
    with pytest.raises(ValidationError):
        StandardPerformance.Params(target_snr=0.0)


def test_performance_bad_contrast_curve_reference_time_rejected_by_params() -> None:
    with pytest.raises(ValidationError):
        StandardPerformance.Params(contrast_curve_reference_time_s=0.0)


def test_performance_negative_sky_background_rejected_by_params() -> None:
    with pytest.raises(ValidationError):
        StandardPerformance.Params(n_sky_background_hz=-1.0)


def test_performance_end_to_end_smoke_chain(tmp_path: Path) -> None:
    # MKID detector component → calibration → performance, all wired
    # through state.results.components. Confirms the cross-stage
    # contract.
    from nullsim.stages.calibration import CalibrationFloorLookup

    state = _state(
        null_depth_raw=5.0e-6,
        integration_time_s=1800.0,
    )
    cal_stage = CalibrationFloorLookup(
        {"default_floor_log10": -6.0, "band": "J"},
        _context(tmp_path),
    )
    cal_state = cal_stage.apply(state)
    perf_stage = _stage(_context(tmp_path))
    perf_state = perf_stage.apply(cal_state)
    component = perf_state.results.components["performance.standard"]

    assert math.isfinite(component["snr"])
    assert component["integration_time_for_target_snr_s"] > 0.0
    assert component["null_depth_source"] == "calibration.floor_lookup"
    assert component["null_depth_residual"] == pytest.approx(4.0e-6, rel=1e-12)
    assert component["null_depth_calibrated"] == pytest.approx(4.0e-6, rel=1e-12)


@pytest.mark.parametrize("n_dark_ports", [1, 2, 4, 8])
def test_performance_snr_invariant_in_n_dark_ports_total_semantics(
    tmp_path: Path, n_dark_ports: int
) -> None:
    """KILO convention (kilo/performance.py L204-292) uses **per-total**
    semantics: signal and leakage rates are aggregate across dark ports.
    Holding the total leakage and total signal constant, SNR must NOT
    depend on n_dark_ports (since D and Sky are per-port × n_dark, but
    if they're zero they don't introduce scaling either).

    Pre-fix (Code Review 2026-05-16 Issue 2) nullsim divided S and B by
    n_dark_ports while keeping D + Sky as per-port × n_dark, so SNR
    drifted with n_dark_ports.
    """

    state = _state(
        bright_rate_hz=1.0e8,
        leakage_rate_hz=1.0e2,  # total leakage held constant
        n_dark_ports=n_dark_ports,
    )
    stage = _stage(
        _context(tmp_path),
        target_snr=5.0,
        with_fringe_tracking=False,
        n_dark_count_hz=0.0,
        n_sky_background_hz=0.0,
    )

    result = stage.apply(state)
    snr = result.results.components["performance.standard"]["snr"]

    # Reference: n_dark=1 SNR for the same total rates.
    state_ref = _state(
        bright_rate_hz=1.0e8,
        leakage_rate_hz=1.0e2,
        n_dark_ports=1,
    )
    stage_ref = _stage(
        _context(tmp_path),
        target_snr=5.0,
        with_fringe_tracking=False,
        n_dark_count_hz=0.0,
        n_sky_background_hz=0.0,
    )
    snr_ref = stage_ref.apply(state_ref).results.components[
        "performance.standard"
    ]["snr"]

    # Per KILO per-total semantics with D=Sky=0, SNR is independent of
    # how many dark ports we split the total leakage across.
    assert snr == pytest.approx(snr_ref, rel=1.0e-9), (
        f"SNR drifted with n_dark_ports={n_dark_ports}: "
        f"got {snr}, expected ~{snr_ref}"
    )
