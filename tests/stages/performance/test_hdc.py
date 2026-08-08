"""Tests for the performance.hdc detection-mode CCF contrast-curve stage."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nullsim.pipeline import (
    ExternalDependency,
    RunContext,
    Scene,
    SimulationState,
    list_stages,
)
from nullsim.stages.performance.hdc import HdcPerformanceStage
from nullsim.stages.performance.standard import _contrast_curve

_N_WAVELENGTHS = 8
_WL_MIN_UM = 1.15
_WL_MAX_UM = 1.35
_N_ANCHORS = 6
_SEPARATIONS_MAS = (20.0, 35.0, 60.0, 100.0, 170.0, 300.0)
_THROUGHPUTS = (0.05, 0.12, 0.22, 0.30, 0.32, 0.31)
_STELLAR_RATE_HZ = 8.0e6
_LEAKAGE_RATE_HZ = 800.0
_EPS_SYSTEMATIC = 1.0e-3
_SERVO_BW_HZ = 100.0
_INTEGRATION_TIME_S = 3600.0
_PLANET_RV_KMS = 30.0
_TARGET_SNR = 5.0
# Deliberately different from _INTEGRATION_TIME_S and from the 5.0 default:
# the broadband curve is stored at its own reference time and target, and the
# stage must inherit BOTH or the gain divides mismatched observations.
_CONTRAST_CURVE_TIME_S = 1800.0
_BROADBAND_TARGET_SNR = 7.0
_SKY_RATE_HZ_PER_PORT = 1.0e-3
_DARK_RATE_HZ_PER_PORT = 1.0e-4
_N_DARK_PORTS = 3


def _context(tmp_path: Path) -> RunContext:
    return RunContext(
        rng=np.random.default_rng(0),
        config_hash="performance-hdc",
        sweep_coord_tuple=(),
        instance_id="hdc",
        run_dir=tmp_path,
    )


def _performance_payload(
    *,
    eps_systematic: float = _EPS_SYSTEMATIC,
    leakage_rate_hz: float = _LEAKAGE_RATE_HZ,
    with_fringe_tracking: bool | None = True,
) -> dict[str, object]:
    """A synthetic ``performance.standard`` payload."""

    # Build the broadband curve with the real performance.standard helper
    # rather than fabricating one, so it responds to eps and the fringe-
    # tracking branch exactly as the upstream stage's own curve would. A
    # fabricated curve would make every gain assertion vacuous.
    contrast_value = _contrast_curve(
        separations_mas=_SEPARATIONS_MAS,
        bright_rate_hz=_STELLAR_RATE_HZ,
        stellar_rate_hz=_STELLAR_RATE_HZ,
        leakage_rate_hz=leakage_rate_hz,
        dark_rate_hz_per_port=_DARK_RATE_HZ_PER_PORT,
        sky_rate_hz_per_port=_SKY_RATE_HZ_PER_PORT,
        n_dark_ports=_N_DARK_PORTS,
        integration_time_s=_CONTRAST_CURVE_TIME_S,
        eps=eps_systematic,
        with_fringe_tracking=bool(with_fringe_tracking),
        servo_bw_hz=_SERVO_BW_HZ,
        target_snr=_BROADBAND_TARGET_SNR,
        planet_throughput_per_sep=_THROUGHPUTS,
    )
    payload: dict[str, object] = {
        "leakage_rate_hz": leakage_rate_hz,
        "stellar_rate_hz_total": _STELLAR_RATE_HZ,
        "sky_rate_hz_per_port": _SKY_RATE_HZ_PER_PORT,
        "dark_rate_hz_per_port": _DARK_RATE_HZ_PER_PORT,
        "n_dark_ports": _N_DARK_PORTS,
        "eps_systematic_used": eps_systematic,
        "servo_bw_hz": _SERVO_BW_HZ,
        "integration_time_for_target_snr_s": _INTEGRATION_TIME_S,
        "integration_time_s": _INTEGRATION_TIME_S,
        "contrast_curve_reference_time_s": _CONTRAST_CURVE_TIME_S,
        "target_snr": _BROADBAND_TARGET_SNR,
        "contrast_curve_mas": _SEPARATIONS_MAS,
        "contrast_curve_value": contrast_value,
        "upstream_sensitivity_component": "sensitivity.detection_curve",
    }
    if with_fringe_tracking is not None:
        payload["with_fringe_tracking"] = with_fringe_tracking
    return payload


def _sensitivity_payload(
    *, throughputs: tuple[float, ...] = _THROUGHPUTS
) -> dict[str, object]:
    wl_um = np.linspace(_WL_MIN_UM, _WL_MAX_UM, _N_WAVELENGTHS)
    return {
        "separations_mas": _SEPARATIONS_MAS,
        "planet_throughputs": np.asarray(throughputs, dtype=np.float64),
        "wavelengths_m": tuple(float(w) * 1.0e-6 for w in wl_um),
    }


def _state(
    *,
    with_performance: bool = True,
    with_sensitivity: bool = True,
    performance_overrides: dict[str, object] | None = None,
    sensitivity_overrides: dict[str, object] | None = None,
) -> SimulationState:
    wl_um = tuple(
        float(w) for w in np.linspace(_WL_MIN_UM, _WL_MAX_UM, _N_WAVELENGTHS)
    )
    state = SimulationState.empty({"wavelength_grid_um": wl_um})
    state = replace(
        state,
        scene=Scene(
            star_jmag=5.0,
            star_spectral_type="M2V",
            planet_contrast=1.0e-6,
            planet_radial_velocity_kms=_PLANET_RV_KMS,
        ),
    )
    results = state.results
    if with_sensitivity:
        results = results.add_component(
            "sensitivity.detection_curve",
            _sensitivity_payload(**(sensitivity_overrides or {})),  # type: ignore[arg-type]
        )
    if with_performance:
        results = results.add_component(
            "performance.standard",
            _performance_payload(**(performance_overrides or {})),  # type: ignore[arg-type]
        )
    return replace(state, results=results)


def _stage(context: RunContext, **overrides: object) -> HdcPerformanceStage:
    params: dict[str, object] = {}
    params.update(overrides)
    return HdcPerformanceStage(params, context)


@pytest.fixture(scope="module")
def payload(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    context = _context(tmp_path_factory.mktemp("perf_hdc"))
    result = _stage(context).apply(_state())
    return result.results.components["performance.hdc"]


def test_performance_hdc_registers_stage_contract(tmp_path: Path) -> None:
    stage = _stage(_context(tmp_path))

    assert "performance.hdc" in list_stages()
    assert stage.type_name == "performance.hdc"
    assert stage.family == "performance"
    assert stage.consumes == frozenset({"results", "scene"})
    assert stage.produces == frozenset({"results"})


def test_performance_hdc_publishes_expected_payload_keys(
    payload: dict[str, object],
) -> None:
    expected = {
        "separations_mas",
        "ccf_contrast_5sigma",
        "broadband_contrast_5sigma",
        "ccf_contrast_gain",
        "target_snr",
        "integration_time_s",
        "integration_time_source",
        "comparison_time_s",
        "comparison_target_snr",
        "comparison_model",
        "post_correction_residual_per_anchor",
        "sky_rate_hz_per_anchor",
        "ccf_leak_rate_hz_per_channel",
        "ccf_planet_rate_hz_per_channel_unit_contrast",
        "upstream_sensitivity_link_source",
        "resolving_power_used",
        "n_channels",
        "n_effective_lines_per_anchor",
        "systematic_var_counts_per_anchor",
        "photon_var_zero_contrast_per_anchor",
        "refined_snr_per_anchor",
        "solver_optimism_per_anchor",
        "corrected_anchor_count",
        "non_detection_reasons",
        "n_non_detections",
        "planet_template_provenance",
        "star_template_provenance",
        "telluric_template_provenance",
        "planet_template_sha256",
        "star_template_sha256",
        "telluric_template_sha256",
        "eps_systematic_used",
        "servo_bw_hz_used",
        "with_fringe_tracking_used",
        "ccf_model",
        "detection_budget_model",
        "leak_shape_model",
        "upstream_performance_component",
        "upstream_sensitivity_component",
    }
    missing = expected - set(payload)
    assert not missing, f"missing payload keys: {sorted(missing)}"


def test_performance_hdc_curve_shapes_and_finiteness(
    payload: dict[str, object],
) -> None:
    separations = np.asarray(payload["separations_mas"], dtype=np.float64)
    ccf = np.asarray(payload["ccf_contrast_5sigma"], dtype=np.float64)
    broadband = np.asarray(payload["broadband_contrast_5sigma"], dtype=np.float64)
    gain = np.asarray(payload["ccf_contrast_gain"], dtype=np.float64)

    assert separations.shape == ccf.shape == broadband.shape == gain.shape
    assert separations.size == _N_ANCHORS
    assert np.all(np.isfinite(ccf)) and np.all(ccf > 0.0)
    assert int(payload["n_non_detections"]) == 0
    # gain = broadband / ccf, by construction.
    assert np.allclose(gain, broadband / ccf, rtol=1.0e-12)


def test_ccf_contrast_gain_exceeds_one_when_systematics_dominate(
    tmp_path: Path,
) -> None:
    """A dominant null instability is what HDC is for: it must beat broadband.

    Without fringe tracking the instability is a static offset, so the
    broadband variance grows like t^2 and its curve saturates. The CCF barely
    notices: continuum filtering leaves the smooth leak almost orthogonal to
    the planet template, so the leak projection that eps multiplies is tiny.
    That asymmetry is the entire HDC systematic gain.
    """

    result = _stage(_context(tmp_path)).apply(
        _state(
            performance_overrides={
                "eps_systematic": 0.01,
                "with_fringe_tracking": False,
            }
        )
    )
    payload = result.results.components["performance.hdc"]
    gain = np.asarray(payload["ccf_contrast_gain"], dtype=np.float64)

    assert np.all(np.isfinite(gain))
    assert np.all(gain > 1.0), f"expected HDC to win at every anchor, got {gain}"


def test_ccf_contrast_gain_near_one_when_photon_limited(tmp_path: Path) -> None:
    """With no systematic, continuum filtering can only cost signal."""

    result = _stage(_context(tmp_path)).apply(
        _state(performance_overrides={"eps_systematic": 0.0})
    )
    payload = result.results.components["performance.hdc"]
    gain = np.asarray(payload["ccf_contrast_gain"], dtype=np.float64)

    assert np.all(np.isfinite(gain))
    # The CCF loses to broadband here: with no systematic to reject, continuum
    # filtering only discards signal. How much it loses is set by the
    # self-subtraction factor (~0.2 for this band and filter width), so the
    # floor is a sanity bound, not a physical claim.
    assert np.all(gain > 0.15)
    assert np.all(gain < 1.05)


def test_contrast_curve_monotonicity_with_throughput(tmp_path: Path) -> None:
    """Half the planet throughput at an anchor must need a brighter planet."""

    halved = list(_THROUGHPUTS)
    halved[3] = _THROUGHPUTS[3] / 2.0

    baseline = _stage(_context(tmp_path)).apply(_state())
    reduced = _stage(_context(tmp_path)).apply(
        _state(sensitivity_overrides={"throughputs": tuple(halved)})
    )
    left = np.asarray(
        baseline.results.components["performance.hdc"]["ccf_contrast_5sigma"]
    )
    right = np.asarray(
        reduced.results.components["performance.hdc"]["ccf_contrast_5sigma"]
    )

    assert right[3] > left[3]
    # Every other anchor is untouched.
    untouched = [i for i in range(_N_ANCHORS) if i != 3]
    assert np.allclose(right[untouched], left[untouched], rtol=1.0e-12)


def test_refined_snr_lands_on_target(payload: dict[str, object]) -> None:
    """The published contrast must actually reach the target SNR."""

    refined = np.asarray(payload["refined_snr_per_anchor"], dtype=np.float64)
    target = float(payload["target_snr"])
    finite = np.isfinite(refined)
    assert finite.any()
    # After the correction step every anchor is within 2% of target.
    assert np.all(np.abs(refined[finite] / target - 1.0) < 0.02)


def test_errors_clearly_without_performance_component(tmp_path: Path) -> None:
    stage = _stage(_context(tmp_path))
    with pytest.raises(ValueError, match="performance.standard"):
        stage.apply(_state(with_performance=False))


def test_errors_clearly_without_sensitivity_component(tmp_path: Path) -> None:
    stage = _stage(_context(tmp_path))
    with pytest.raises(ValueError, match="sensitivity.detection_curve"):
        stage.apply(_state(with_sensitivity=False))


def test_explicit_upstream_selector_and_most_recent_default(
    tmp_path: Path,
) -> None:
    state = _state()
    state = replace(
        state,
        results=state.results.add_component(
            "performance.standard.b",
            _performance_payload(leakage_rate_hz=4.0 * _LEAKAGE_RATE_HZ),
        ),
    )

    # Default: the most recently added compatible component.
    default = _stage(_context(tmp_path)).apply(state)
    default_payload = default.results.components["performance.hdc"]
    assert (
        default_payload["upstream_performance_component"]
        == "performance.standard.b"
    )

    explicit = _stage(
        _context(tmp_path), upstream_performance="performance.standard"
    ).apply(state)
    explicit_payload = explicit.results.components["performance.hdc"]
    assert (
        explicit_payload["upstream_performance_component"] == "performance.standard"
    )

    # More leak needs a brighter planet, so the two selections must differ.
    assert not np.allclose(
        np.asarray(default_payload["ccf_contrast_5sigma"]),
        np.asarray(explicit_payload["ccf_contrast_5sigma"]),
    )

    missing = _stage(_context(tmp_path), upstream_performance="performance.nope")
    with pytest.raises(ValueError, match="is not present on state.results"):
        missing.apply(state)


def test_explicit_upstream_sensitivity_selector(tmp_path: Path) -> None:
    state = _state()
    result = _stage(
        _context(tmp_path), upstream_sensitivity="sensitivity.detection_curve"
    ).apply(state)
    payload = result.results.components["performance.hdc"]
    assert (
        payload["upstream_sensitivity_component"] == "sensitivity.detection_curve"
    )


def test_deterministic(tmp_path: Path) -> None:
    state = _state()
    first = _stage(_context(tmp_path)).apply(state)
    second = _stage(_context(tmp_path)).apply(state)
    left = first.results.components["performance.hdc"]
    right = second.results.components["performance.hdc"]

    assert np.array_equal(
        np.asarray(left["ccf_contrast_5sigma"]),
        np.asarray(right["ccf_contrast_5sigma"]),
    )
    assert np.array_equal(
        np.asarray(left["refined_snr_per_anchor"]),
        np.asarray(right["refined_snr_per_anchor"]),
    )


def test_non_detection_anchor_records_nan_and_a_reason(tmp_path: Path) -> None:
    """A masked-out template must not abort the whole curve."""

    # Threshold above 1 masks every channel, so no template signal survives.
    result = _stage(
        _context(tmp_path),
        telluric_template="maunakea",
        telluric_mask_threshold=1.0,
    ).apply(_state())
    payload = result.results.components["performance.hdc"]

    ccf = np.asarray(payload["ccf_contrast_5sigma"], dtype=np.float64)
    reasons = payload["non_detection_reasons"]
    assert np.all(np.isnan(ccf))
    assert int(payload["n_non_detections"]) == _N_ANCHORS
    assert len(reasons) == _N_ANCHORS
    assert any("no template signal" in str(reason) for reason in reasons)


def test_declares_hashed_template_dependencies(tmp_path: Path) -> None:
    stage = _stage(_context(tmp_path), telluric_template="maunakea")
    dependencies = stage.external_dependencies()

    assert dependencies
    assert all(isinstance(dep, ExternalDependency) for dep in dependencies)
    sources = {dep.source for dep in dependencies}
    assert "performance.hdc.planet_template" in sources
    assert "performance.hdc.telluric_template" in sources


def test_comparison_basis_is_inherited_from_the_broadband_curve(
    payload: dict[str, object],
) -> None:
    """The gain must divide curves computed at the same time and threshold.

    The fixture stores the broadband curve at 1800 s / SNR 7 while publishing a
    3600 s integration_time_for_target_snr_s and while the stage's own default
    target would be 5. Inheriting both is what keeps the ratio meaningful.
    """

    assert float(payload["comparison_time_s"]) == pytest.approx(
        _CONTRAST_CURVE_TIME_S
    )
    assert float(payload["comparison_target_snr"]) == pytest.approx(
        _BROADBAND_TARGET_SNR
    )
    assert float(payload["integration_time_s"]) == pytest.approx(
        _CONTRAST_CURVE_TIME_S
    )
    assert float(payload["target_snr"]) == pytest.approx(_BROADBAND_TARGET_SNR)
    assert "contrast_curve_reference_time_s" in str(payload["comparison_model"])


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"integration_time_s": 3600.0}, "contrast_curve_reference_time_s"),
        ({"target_snr": 5.0}, "target_snr"),
    ],
)
def test_mismatched_override_is_refused(
    tmp_path: Path, override: dict[str, object], message: str
) -> None:
    stage = _stage(_context(tmp_path), **override)
    with pytest.raises(ValueError, match=message):
        stage.apply(_state())


def test_matching_override_is_accepted(tmp_path: Path) -> None:
    result = _stage(
        _context(tmp_path),
        integration_time_s=_CONTRAST_CURVE_TIME_S,
        target_snr=_BROADBAND_TARGET_SNR,
    ).apply(_state())
    payload = result.results.components["performance.hdc"]
    assert float(payload["comparison_time_s"]) == pytest.approx(
        _CONTRAST_CURVE_TIME_S
    )



def test_refine_contrast_corrects_a_shortfall_and_reports_it() -> None:
    """The correction step must actually pull a short contrast onto target.

    It does not fire in the fixtures above -- the CCF's systematic share is
    tiny by construction, which is the whole point of HDC -- so the path is
    exercised directly by handing it a contrast that is deliberately too faint.
    """

    from nullsim.physics.hdc import ChannelSpectra
    from nullsim.stages.performance.hdc import _refine_contrast

    rng = np.random.default_rng(3)
    n = 64
    edges = np.linspace(1.15, 1.35, n + 1)
    planet_unit = rng.uniform(0.5, 1.5, n)
    leak = rng.uniform(80.0, 120.0, n)
    channels = ChannelSpectra(
        channel_centers_um=0.5 * (edges[:-1] + edges[1:]),
        channel_edges_um=edges,
        planet_rate_hz=planet_unit,
        leak_rate_hz=leak,
        sky_rate_hz=np.zeros(n),
        dark_rate_hz=np.zeros(n),
    )
    solver_kwargs = {
        "integration_time_s": 3600.0,
        "eps_systematic": 0.0,
        "servo_bw_hz": 100.0,
        "with_fringe_tracking": True,
        "continuum_filter_width_channels": 30,
        "telluric_transmission_per_channel": None,
        "telluric_mask_threshold": 0.8,
    }
    target = 5.0

    def refine(solution: object) -> tuple[float, float, float, bool]:
        return _refine_contrast(
            solution,
            planet_unit_rate_hz=planet_unit,
            channels=channels,
            leak=leak,
            sky=np.zeros(n),
            dark=np.zeros(n),
            target_snr=target,
            solver_kwargs=solver_kwargs,
        )

    # A solution that lands on target, then an optimistic one at half of it.
    from dataclasses import replace as dc_replace

    from nullsim.physics.hdc import ccf_contrast_at_target_snr

    solution = ccf_contrast_at_target_snr(
        planet_unit,
        leak,
        np.zeros(n),
        np.zeros(n),
        target_snr=target,
        **{k: v for k, v in solver_kwargs.items()},
    )
    optimistic = dc_replace(
        solution,
        contrast_at_target_snr=0.5 * solution.contrast_at_target_snr,
    )

    value, snr, first_snr, corrected = refine(optimistic)
    assert corrected is True
    assert value > optimistic.contrast_at_target_snr
    assert snr == pytest.approx(target, rel=0.02)
    # The reported optimism is the PRE-correction evaluation.
    assert first_snr == pytest.approx(0.5 * target, rel=0.05)

    # Starting on target, nothing is corrected and the two agree.
    value, snr, first_snr, corrected = refine(solution)
    assert corrected is False
    assert snr == pytest.approx(target, rel=0.02)
    assert first_snr == snr


def test_chromatic_residual_structures_the_leak_spectrum(tmp_path: Path) -> None:
    """A chromatic null must reach the leak's channel spectrum, not be greyed."""

    wl_um = np.linspace(_WL_MIN_UM, _WL_MAX_UM, 12)
    # A strong ramp with a notch: unmistakable structure.
    row = 1.0 + 4.0 * (wl_um - _WL_MIN_UM) / (_WL_MAX_UM - _WL_MIN_UM)
    row[5:7] *= 0.1
    surface = np.tile(row, (_N_ANCHORS, 1))

    sensitivity = _sensitivity_payload()
    sensitivity["residual_starlight_contrast_by_wavelength"] = surface
    sensitivity["residual_starlight_wavelengths_um"] = tuple(
        float(w) for w in wl_um
    )

    base = _state()
    chromatic = replace(
        base,
        results=base.results.add_component(
            "sensitivity.detection_curve.chromatic", sensitivity
        ),
    )
    # Point the performance link at the chromatic sensitivity component.
    performance = dict(
        chromatic.results.components["performance.standard"]
    )
    performance["upstream_sensitivity_component"] = (
        "sensitivity.detection_curve.chromatic"
    )
    comps = dict(chromatic.results.components)
    comps["performance.standard"] = performance
    from nullsim.pipeline.state import Results

    chromatic = replace(chromatic, results=Results(components=comps))

    grey_payload = _stage(_context(tmp_path)).apply(base).results.components[
        "performance.hdc"
    ]
    chromatic_payload = _stage(_context(tmp_path)).apply(
        chromatic
    ).results.components["performance.hdc"]

    assert "GREY" in str(grey_payload["leak_shape_model"])
    assert "CHROMATIC" in str(chromatic_payload["leak_shape_model"])

    grey_leak = np.asarray(grey_payload["ccf_leak_rate_hz_per_channel"][0])
    chromatic_leak = np.asarray(
        chromatic_payload["ccf_leak_rate_hz_per_channel"][0]
    )
    # Same band total (the shape is band-mean normalized)...
    assert float(np.sum(chromatic_leak)) == pytest.approx(
        float(np.sum(grey_leak)), rel=0.02
    )
    # ...but a genuinely different spectrum: the ramp must show up as a rising
    # red/blue ratio that the grey leak does not have.
    half = chromatic_leak.size // 2
    grey_ratio = float(np.sum(grey_leak[half:]) / np.sum(grey_leak[:half]))
    chromatic_ratio = float(
        np.sum(chromatic_leak[half:]) / np.sum(chromatic_leak[:half])
    )
    assert chromatic_ratio > 1.5 * grey_ratio


def test_sky_carries_the_awg_throughput(tmp_path: Path) -> None:
    full = _stage(_context(tmp_path), awg_throughput=1.0).apply(_state())
    lossy = _stage(_context(tmp_path), awg_throughput=0.5).apply(_state())

    full_sky = np.asarray(
        full.results.components["performance.hdc"]["sky_rate_hz_per_anchor"]
    )
    lossy_sky = np.asarray(
        lossy.results.components["performance.hdc"]["sky_rate_hz_per_anchor"]
    )
    assert np.allclose(lossy_sky, 0.5 * full_sky)


def test_zero_scalar_sky_does_not_zero_a_positive_per_anchor_sky(
    tmp_path: Path,
) -> None:
    """The reachable bug: a unit sky reference must not collapse to zero."""

    state = _state()
    performance = dict(state.results.components["performance.standard"])
    performance["sky_rate_hz_per_port"] = 0.0
    performance["contrast_curve_sky_rate_hz_per_port"] = tuple(
        2.0e-3 for _ in _SEPARATIONS_MAS
    )
    from nullsim.pipeline.state import Results

    comps = dict(state.results.components)
    comps["performance.standard"] = performance
    state = replace(state, results=Results(components=comps))

    payload = _stage(_context(tmp_path)).apply(state).results.components[
        "performance.hdc"
    ]
    sky = np.asarray(payload["sky_rate_hz_per_anchor"], dtype=np.float64)
    assert np.all(sky > 0.0)
    assert np.all(np.isfinite(payload["ccf_contrast_5sigma"]))


def test_per_anchor_curves_interpolate_by_coordinate_not_by_index(
    tmp_path: Path,
) -> None:
    """Equal-length grids with different coordinates must not misalign."""

    state = _state()
    performance = dict(state.results.components["performance.standard"])
    # Same length as the anchors, deliberately different coordinates.
    other_grid = tuple(s * 2.0 for s in _SEPARATIONS_MAS)
    leak_curve = tuple(
        _LEAKAGE_RATE_HZ * (1.0 + 0.5 * index) for index in range(_N_ANCHORS)
    )
    performance["contrast_curve_mas"] = other_grid
    performance["contrast_curve_leakage_rate_hz"] = leak_curve
    from nullsim.pipeline.state import Results

    comps = dict(state.results.components)
    comps["performance.standard"] = performance
    state = replace(state, results=Results(components=comps))

    payload = _stage(_context(tmp_path)).apply(state).results.components[
        "performance.hdc"
    ]
    used = np.asarray(payload["leak_rate_hz_per_anchor"], dtype=np.float64)
    expected = np.interp(
        np.asarray(_SEPARATIONS_MAS), np.asarray(other_grid), np.asarray(leak_curve)
    ) * float(payload["awg_throughput_used"])

    assert np.allclose(used, expected)
    # The index-matched reading would have been the raw curve; reject it.
    index_matched = np.asarray(leak_curve) * float(payload["awg_throughput_used"])
    assert not np.allclose(used, index_matched)


def test_sensitivity_resolved_through_the_performance_link(
    tmp_path: Path,
) -> None:
    """The link wins over recency so the budgets describe one solution."""

    state = _state()
    later = _sensitivity_payload(throughputs=tuple(0.9 for _ in _THROUGHPUTS))
    state = replace(
        state,
        results=state.results.add_component(
            "sensitivity.detection_curve.later", later
        ),
    )

    payload = _stage(_context(tmp_path)).apply(state).results.components[
        "performance.hdc"
    ]
    # performance.standard names the first one; recency would have taken the
    # later one.
    assert (
        payload["upstream_sensitivity_component"] == "sensitivity.detection_curve"
    )
    assert "link" in str(payload["upstream_sensitivity_link_source"])


def test_missing_linked_sensitivity_component_raises(tmp_path: Path) -> None:
    state = _state()
    performance = dict(state.results.components["performance.standard"])
    performance["upstream_sensitivity_component"] = "sensitivity.gone"
    from nullsim.pipeline.state import Results

    comps = dict(state.results.components)
    comps["performance.standard"] = performance
    state = replace(state, results=Results(components=comps))

    with pytest.raises(ValueError, match="is not present on state.results"):
        _stage(_context(tmp_path)).apply(state)


def test_refinement_side_error_is_caught_per_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ValueError raised during refinement must not abort the curve."""

    import nullsim.stages.performance.hdc as module

    def boom(*_args: object, **_kwargs: object) -> None:
        raise ValueError("synthetic refinement failure")

    monkeypatch.setattr(module, "matched_filter_ccf", boom)

    payload = _stage(_context(tmp_path)).apply(_state()).results.components[
        "performance.hdc"
    ]
    assert int(payload["n_non_detections"]) == _N_ANCHORS
    assert np.all(np.isnan(np.asarray(payload["ccf_contrast_5sigma"])))
    assert all(
        "synthetic refinement failure" in str(reason)
        for reason in payload["non_detection_reasons"]
    )
