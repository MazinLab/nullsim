"""Tests for the ClosedLoopFringeTracker stage (SPEC §15.3 Pass 4b).

Reference invariants come from KILO ``fringe_tracking.py`` (Shao-Colavita
1992 shot noise primitive, Conan 1995 piston-specific Greenwood
frequency, and ``FringeTrackingConfig`` defaults). See
``tests/stages/notes/fringe_tracking.md`` for the source survey.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from nullsim.pipeline import (
    Geometry,
    OPDBudget,
    OPDComponent,
    PhotonRates,
    RunContext,
    Scene,
    SimulationState,
    Wavefront,
)
from nullsim.stages.fringe_tracking import ClosedLoopFringeTracker


_DEFAULT_WAVELENGTH_GRID_UM: tuple[float, ...] = (1.20, 1.25, 1.30)
_DEFAULT_WAVELENGTH_CENTER_M: float = 1.25e-6


def _context(tmp_path: Path, *, seed: int = 0) -> RunContext:
    return RunContext(
        rng=np.random.default_rng(seed),
        config_hash="fringe-tracking-closed-loop",
        sweep_coord_tuple=(),
        instance_id="fringe_tracker",
        run_dir=tmp_path,
    )


def _stage(
    context: RunContext, **overrides: object
) -> ClosedLoopFringeTracker:
    # Phase 2: the production default is servo_bw_hz=None (auto-derive
    # from KILO's photon-limited optimum), which requires an upstream
    # ao.pyramid_wfs publishing 'ao_residual'. The unit tests in this
    # file build minimal states with no upstream AO stage, so we pin
    # the bandwidth to the pre-Phase-2 default here.
    overrides.setdefault("servo_bw_hz", 2000.0)
    # The piston servo residual needs the chip mode count for its sensing
    # efficiency; these minimal states carry no field.
    overrides.setdefault("n_modes", 24)
    return ClosedLoopFringeTracker(dict(overrides), context)


def _fringe_tracking_payload(result: SimulationState) -> dict[str, object]:
    for name, value in result.results.components.items():
        if name.startswith("fringe_tracking.closed_loop"):
            return dict(value)
    raise AssertionError("missing fringe tracking result component")


def _state(
    *,
    wavelength_grid_um: tuple[float, ...] = _DEFAULT_WAVELENGTH_GRID_UM,
    atmos_piston_rms_opd_nm: float | None = 100.0,
    upstream_name: str = "atmos_piston",
    extra_components: dict[str, OPDComponent] | None = None,
    diameter_m: float | None = 10.0,
    bright_port_rate_hz: float = 1.0e8,
    n_ports: int = 4,
    rates_hz: np.ndarray | None = None,
) -> SimulationState:
    state = SimulationState.empty({"wavelength_grid_um": wavelength_grid_um})

    if diameter_m is None:
        geometry = Geometry()
    else:
        geometry = Geometry(
            diameter_m=diameter_m,
            area_m2=math.pi * (diameter_m / 2.0) ** 2,
        )
    wavefront = Wavefront(
        wavelength_grid_um=wavelength_grid_um,
        r0_at_500nm_m=0.15,
    )

    components: dict[str, OPDComponent] = {}
    if atmos_piston_rms_opd_nm is not None:
        components[upstream_name] = OPDComponent(
            rms_opd_nm=atmos_piston_rms_opd_nm,
            reduction_stage="kolmogorov_screens",
        )
    if extra_components is not None:
        components.update(extra_components)

    total = (
        math.sqrt(
            sum(
                component.rms_opd_nm * component.rms_opd_nm
                for component in components.values()
                if not component.superseded
            )
        )
        if components
        else None
    )
    opd_budget = OPDBudget(components=components, total_rms_opd_nm=total)

    if rates_hz is None:
        rates_hz = np.full(
            (n_ports, len(wavelength_grid_um)),
            bright_port_rate_hz / (n_ports * len(wavelength_grid_um)),
            dtype=np.float64,
        )
    # Use the canonical bright_*/dark_* label convention emitted by the
    # detector stages (mkid._port_labels, snspd._port_labels). The FT stage
    # masks on these labels to extract bright-port shot-noise photons.
    # Default partition: one bright fringe-port, rest dark.
    n_p = rates_hz.shape[0]
    photon_rates = PhotonRates(
        wavelength_grid_um=wavelength_grid_um,
        port_labels=("bright_0",)
        + tuple(f"dark_{idx}" for idx in range(n_p - 1)),
        rates_hz=rates_hz,
    )

    return replace(
        state,
        geometry=geometry,
        wavefront=wavefront,
        opd_budget=opd_budget,
        photon_rates=photon_rates,
    )


def _with_ao_residual(
    state: SimulationState,
    *,
    temporal_nm: float,
    fitting_nm: float,
    wfs_noise_nm: float,
    subaperture_diameter_m: float = 1.93,
    wind_speed_m_s: float = 16.643,
    r0_500nm_m: float = 0.15,
) -> SimulationState:
    ao_payload = {
        "piston_opd_nm": {
            "temporal": temporal_nm,
            "fitting": fitting_nm,
            "wfs_noise": wfs_noise_nm,
            "total": math.sqrt(temporal_nm**2 + fitting_nm**2 + wfs_noise_nm**2),
        },
        "piston_subaperture_diameter_m": subaperture_diameter_m,
        "effective_wind_speed_m_s": wind_speed_m_s,
        "r0_500nm_m": r0_500nm_m,
    }
    return replace(
        state,
        wavefront=replace(state.wavefront).add_component("ao_residual", ao_payload),
    )


def _with_small_ao(state: SimulationState) -> SimulationState:
    return _with_ao_residual(state, temporal_nm=10.0, fitting_nm=3.0, wfs_noise_nm=2.0)


def test_closed_loop_supersedes_upstream_and_adds_residual_component(
    tmp_path: Path,
) -> None:
    stage = _stage(_context(tmp_path))
    state = _with_small_ao(_state())

    result = stage.apply(state)

    components = result.opd_budget.components
    assert "atmos_piston" in components
    assert components["atmos_piston"].superseded is True
    assert "fringe_tracking.closed_loop_residual" in components
    residual = components["fringe_tracking.closed_loop_residual"]
    assert residual.reduction_stage == "fringe_tracking.closed_loop"
    assert math.isfinite(residual.rms_opd_nm)
    assert residual.rms_opd_nm > 0.0


def test_closed_loop_total_rms_excludes_superseded_atmos_piston(
    tmp_path: Path,
) -> None:
    stage = _stage(_context(tmp_path))
    state = _with_small_ao(_state(atmos_piston_rms_opd_nm=200.0))

    result = stage.apply(state)

    residual_nm = result.opd_budget.components[
        "fringe_tracking.closed_loop_residual"
    ].rms_opd_nm
    expected_total = residual_nm  # only the new residual contributes
    assert result.opd_budget.total_rms_opd_nm == pytest.approx(expected_total)
    assert result.opd_budget.total_rms_opd_nm < 200.0


def test_closed_loop_respects_explicit_superseded_component(
    tmp_path: Path,
) -> None:
    """When AO already superseded atmos_piston, FT corrects the AO residual."""

    extra = {
        "ao_temporal": OPDComponent(
            rms_opd_nm=10.6,
            reduction_stage="pyramid_wfs",
        ),
    }
    state = _state(
        atmos_piston_rms_opd_nm=100.0,
        extra_components=extra,
    )
    # Mark atmos_piston superseded as pyramid_wfs would
    components = dict(state.opd_budget.components)
    components["atmos_piston"] = replace(components["atmos_piston"], superseded=True)
    state = _with_small_ao(
        replace(
            state,
            opd_budget=replace(state.opd_budget, components=components),
        )
    )

    stage = _stage(
        _context(tmp_path),
        superseded_components=("ao_temporal",),
    )
    result = stage.apply(state)

    assert result.opd_budget.components["ao_temporal"].superseded is True
    # atmos_piston stays superseded from upstream stage
    assert result.opd_budget.components["atmos_piston"].superseded is True
    assert (
        result.opd_budget.components[
            "fringe_tracking.closed_loop_residual"
        ].rms_opd_nm
        > 0.0
    )


def test_closed_loop_supersedes_all_replaced_ao_piston_components(
    tmp_path: Path,
) -> None:
    extra = {
        "ao_temporal": OPDComponent(rms_opd_nm=10.0, reduction_stage="pyramid_wfs"),
        "ao_fitting": OPDComponent(rms_opd_nm=2.0, reduction_stage="pyramid_wfs"),
        "ao_wfs_noise": OPDComponent(rms_opd_nm=3.0, reduction_stage="pyramid_wfs"),
        "fiber_chromatic_dispersion": OPDComponent(
            rms_opd_nm=4.0,
            reduction_stage="transport.smf28",
        ),
    }
    state = _with_small_ao(
        _state(
            atmos_piston_rms_opd_nm=100.0,
            extra_components=extra,
        )
    )
    stage = _stage(_context(tmp_path))

    result = stage.apply(state)
    components = result.opd_budget.components

    for name in ("atmos_piston", "ao_temporal", "ao_fitting", "ao_wfs_noise"):
        assert components[name].superseded is True
    residual_nm = components["fringe_tracking.closed_loop_residual"].rms_opd_nm
    expected_total = math.sqrt(
        residual_nm**2 + components["fiber_chromatic_dispersion"].rms_opd_nm**2
    )
    assert result.opd_budget.total_rms_opd_nm == pytest.approx(expected_total)


@pytest.mark.stats
def test_closed_loop_high_bandwidth_drives_residual_to_shot_noise_floor(
    tmp_path: Path,
) -> None:
    """In the photon-rich limit, raising servo_bw cuts the lag-dominated total."""

    base_state = _with_ao_residual(
        _state(
            atmos_piston_rms_opd_nm=500.0,
            bright_port_rate_hz=1.0e18,  # photon-rich so shot noise is negligible
        ),
        temporal_nm=500.0,
        fitting_nm=0.0,
        wfs_noise_nm=0.0,
    )

    stage_high_bw = _stage(
        _context(tmp_path),
        servo_bw_hz=1.0e6,
        predictive_lag_reduction=1.0,
    )
    stage_baseline = _stage(
        _context(tmp_path),
        servo_bw_hz=2.0e3,
        predictive_lag_reduction=1.0,
    )

    residual_high_bw = stage_high_bw.apply(base_state).opd_budget.components[
        "fringe_tracking.closed_loop_residual"
    ].rms_opd_nm
    residual_baseline = stage_baseline.apply(base_state).opd_budget.components[
        "fringe_tracking.closed_loop_residual"
    ].rms_opd_nm

    assert residual_high_bw < residual_baseline
    assert residual_high_bw > 0.0  # shot-noise floor remains positive


@pytest.mark.stats
def test_closed_loop_shot_noise_vanishes_in_photon_rich_limit(
    tmp_path: Path,
) -> None:
    """Very high photon rate should leave only the servo-lag floor."""

    state_rich = _with_small_ao(
        _state(
            atmos_piston_rms_opd_nm=500.0,
            bright_port_rate_hz=1.0e18,  # photon-rich regime
        )
    )
    state_starved = _with_small_ao(
        _state(
            atmos_piston_rms_opd_nm=500.0,
            bright_port_rate_hz=1.0e2,  # photon-starved regime
        )
    )

    stage = _stage(_context(tmp_path))

    residual_rich = stage.apply(state_rich).opd_budget.components[
        "fringe_tracking.closed_loop_residual"
    ].rms_opd_nm
    residual_starved = stage.apply(state_starved).opd_budget.components[
        "fringe_tracking.closed_loop_residual"
    ].rms_opd_nm

    assert residual_rich < residual_starved
    assert residual_rich > 0.0  # servo lag still binds


@pytest.mark.stats
def test_closed_loop_predictive_kalman_reduces_residual_in_lag_regime(
    tmp_path: Path,
) -> None:
    """predictive_lag_reduction=0.05 cuts residual sqrt(20)~4.47x in lag regime."""

    # Lag-only regime: photon-rich, no irreducible AO piston, and n_modes=2
    # so the sensing efficiency is 1 and no temporal piston is uncorrectable.
    state = _with_ao_residual(
        _state(
            atmos_piston_rms_opd_nm=500.0,
            bright_port_rate_hz=1.0e18,
        ),
        temporal_nm=500.0,
        fitting_nm=0.0,
        wfs_noise_nm=0.0,
    )

    stage_leaky = _stage(
        _context(tmp_path),
        n_modes=2,
        predictive_lag_reduction=1.0,
    )
    stage_kalman = _stage(
        _context(tmp_path),
        n_modes=2,
        predictive_lag_reduction=0.05,
    )

    residual_leaky = stage_leaky.apply(state).opd_budget.components[
        "fringe_tracking.closed_loop_residual"
    ].rms_opd_nm
    residual_kalman = stage_kalman.apply(state).opd_budget.components[
        "fringe_tracking.closed_loop_residual"
    ].rms_opd_nm

    # Variance ratio = 0.05 => sigma ratio = sqrt(20) ~= 4.472.
    ratio = residual_leaky / residual_kalman
    assert ratio == pytest.approx(math.sqrt(20.0), rel=1.0e-6)


@pytest.mark.stats
def test_closed_loop_shot_variance_halves_when_photon_rate_doubles(
    tmp_path: Path,
) -> None:
    """Doubling the photon rate halves sigma^2_shot (Shao-Colavita)."""

    # Zero AO piston breakdown: the residual is pure shot noise.
    state_low = _with_ao_residual(
        _state(atmos_piston_rms_opd_nm=0.0, bright_port_rate_hz=1.0e6),
        temporal_nm=0.0,
        fitting_nm=0.0,
        wfs_noise_nm=0.0,
    )
    state_high = _with_ao_residual(
        _state(atmos_piston_rms_opd_nm=0.0, bright_port_rate_hz=2.0e6),
        temporal_nm=0.0,
        fitting_nm=0.0,
        wfs_noise_nm=0.0,
    )

    stage = _stage(_context(tmp_path))

    residual_low = stage.apply(state_low).opd_budget.components[
        "fringe_tracking.closed_loop_residual"
    ].rms_opd_nm
    residual_high = stage.apply(state_high).opd_budget.components[
        "fringe_tracking.closed_loop_residual"
    ].rms_opd_nm

    # sigma^2_shot halves -> sigma_shot reduces by sqrt(2).
    ratio = residual_low / residual_high
    assert ratio == pytest.approx(math.sqrt(2.0), rel=1.0e-9)


@pytest.mark.stats
def test_closed_loop_lag_variance_drops_by_2_pow_5over3_when_servo_doubles(
    tmp_path: Path,
) -> None:
    """Doubling f_servo reduces sigma^2_lag by 2^(5/3) ~= 3.175x (Conan)."""

    # Lag-dominated regime: photon-rich, large piston, modest baseline servo.
    # bright_port_rate_hz is spread across 4 ports in _state but only the
    # `bright_0` port now contributes to FT shot noise (after the bright-port
    # masking fix), so bump the rate enough that shot-noise contribution stays
    # negligible vs the lag-dominated 2^(5/3) variance ratio.
    state = _with_ao_residual(
        _state(
            atmos_piston_rms_opd_nm=500.0,
            bright_port_rate_hz=1.0e18,
        ),
        temporal_nm=500.0,
        fitting_nm=0.0,
        wfs_noise_nm=0.0,
    )

    stage_baseline = _stage(
        _context(tmp_path),
        servo_bw_hz=1.0e3,
        n_modes=2,
        predictive_lag_reduction=1.0,
    )
    stage_doubled = _stage(
        _context(tmp_path),
        servo_bw_hz=2.0e3,
        n_modes=2,
        predictive_lag_reduction=1.0,
    )

    residual_baseline = stage_baseline.apply(state).opd_budget.components[
        "fringe_tracking.closed_loop_residual"
    ].rms_opd_nm
    residual_doubled = stage_doubled.apply(state).opd_budget.components[
        "fringe_tracking.closed_loop_residual"
    ].rms_opd_nm

    # variance ratio = 2^(5/3), sigma ratio = 2^(5/6)
    sigma_ratio = residual_baseline / residual_doubled
    assert sigma_ratio == pytest.approx(2.0 ** (5.0 / 6.0), rel=1.0e-6)



def test_closed_loop_rejects_legacy_servo_lag_model(tmp_path: Path) -> None:
    """use_full_piston_servo_residual=false selected a lag model with no r0.

    Its (f_piston/f_servo)^(5/3) rad^2 lag variance ignored r0 and the
    upstream piston, so the OPD residual scaled with wavelength. The flag now
    defaults to true and false is refused with the reason.
    """

    assert _stage(_context(tmp_path)).params_model.use_full_piston_servo_residual
    with pytest.raises(ValidationError, match="ignores r0"):
        _stage(_context(tmp_path), use_full_piston_servo_residual=False)


def test_closed_loop_shao_colavita_matches_kilo_formula(tmp_path: Path) -> None:
    """Regression test for Tier 2.2 Shao-Colavita fix.

    KILO ``shot_noise_phase_variance_per_frame`` (kilo/fringe_tracking.py:67-95)
    gives ``(N_s + N_d) / (V² × N_s²)`` per channel and ``/n_spectral_channels``
    for cross-channel averaging. Pre-fix nullsim used
    ``1 / (V² × n_spectral_eff × N_s × 2π²)`` which is off by ~158× from
    KILO for typical KILO Keck params.
    """
    # ``_state`` spreads ``bright_port_rate_hz`` uniformly across n_ports;
    # one port (``bright_0``) feeds the FT shot-noise budget so the actual
    # bright-rate seen by the stage is ``bright_port_rate_hz / n_ports``.
    total_rate_hz = 1.0e7
    n_ports = 4
    bright_rate_hz = total_rate_hz / n_ports
    visibility = 0.9
    n_spectral_channels = 10
    servo_bw = 2000.0
    state = _with_ao_residual(
        _state(
            atmos_piston_rms_opd_nm=0.0,
            bright_port_rate_hz=total_rate_hz,
            n_ports=n_ports,
        ),
        temporal_nm=0.0,  # zero AO piston: the residual is pure shot noise
        fitting_nm=0.0,
        wfs_noise_nm=0.0,
    )
    stage = _stage(
        _context(tmp_path),
        servo_bw_hz=servo_bw,
        fringe_visibility=visibility,
        n_spectral_channels=n_spectral_channels,
        predictive_lag_reduction=1.0,
    )
    residual_nm = stage.apply(state).opd_budget.components[
        "fringe_tracking.closed_loop_residual"
    ].rms_opd_nm

    # KILO analytic: N_s = rate_per_channel × tau
    tau = 1.0 / (2.0 * servo_bw)
    n_signal = (bright_rate_hz / n_spectral_channels) * tau
    sigma2_per_channel = n_signal / (visibility ** 2 * n_signal ** 2)
    # KILO_PARITY_PLAN §2.5: the closed_loop stage now multiplies the
    # per-channel-averaged shot variance by π/2 (KILO
    # ``piston_servo_residual`` noise-equivalent gain integral,
    # ``kilo/fringe_tracking.py:439-443``).
    sigma2_shot_rad2 = (math.pi / 2.0) * sigma2_per_channel / n_spectral_channels

    expected_rms_nm = math.sqrt(sigma2_shot_rad2) * (
        _representative_wavelength_m_default() * 1.0e9 / (2.0 * math.pi)
    )
    assert residual_nm == pytest.approx(expected_rms_nm, rel=1.0e-6)


def test_closed_loop_fraction_half_reproduces_legacy_full_residual(
    tmp_path: Path,
) -> None:
    """With f_3dB=f_sample/2, the explicit-bandwidth path is pre-change exact."""

    servo_bw = 2000.0
    visibility = 0.9
    n_spectral_channels = 20
    n_spectral_eff = 2.5
    n_modes = 24
    predictive_lag_reduction = 1.0
    state = _state(
        atmos_piston_rms_opd_nm=200.0,
        diameter_m=10.0,
        bright_port_rate_hz=1.0e8,
        n_ports=4,
    )
    ao_payload = {
        "piston_opd_nm": {
            "temporal": 100.0,
            "fitting": 30.0,
            "wfs_noise": 20.0,
            "total": math.sqrt(100.0**2 + 30.0**2 + 20.0**2),
        },
        "piston_subaperture_diameter_m": 1.93,
        "effective_wind_speed_m_s": 10.0,
        "r0_500nm_m": 0.15,
    }
    state = replace(
        state,
        wavefront=replace(state.wavefront).add_component("ao_residual", ao_payload),
    )
    stage = _stage(
        _context(tmp_path),
        servo_bw_hz=servo_bw,
        servo_bandwidth_fraction=0.5,
        use_full_piston_servo_residual=True,
        n_modes=n_modes,
        n_spectral_channels=n_spectral_channels,
        n_spectral_eff=n_spectral_eff,
        fringe_visibility=visibility,
        predictive_lag_reduction=predictive_lag_reduction,
    )

    result = stage.apply(state)
    residual_nm = result.opd_budget.components[
        "fringe_tracking.closed_loop_residual"
    ].rms_opd_nm
    payload = _fringe_tracking_payload(result)

    bright_rate_hz = 1.0e8 / 4.0
    tau = 1.0 / (2.0 * servo_bw)
    n_signal = (bright_rate_hz / n_spectral_channels) * tau
    sigma2_per_channel = n_signal / (visibility**2 * n_signal**2)
    sigma2_photon = (math.pi / 2.0) * sigma2_per_channel / n_spectral_channels
    eta = min(2.0 * 1.0 * n_spectral_eff, float(n_modes - 1)) / float(n_modes - 1)
    sigma2_temporal = (2.0 * math.pi * 100.0e-9 / _DEFAULT_WAVELENGTH_CENTER_M) ** 2
    sigma2_fitting = (2.0 * math.pi * 30.0e-9 / _DEFAULT_WAVELENGTH_CENTER_M) ** 2
    sigma2_wfs = (2.0 * math.pi * 20.0e-9 / _DEFAULT_WAVELENGTH_CENTER_M) ** 2
    f_piston_hz = 0.134 * 10.0 / 1.93
    sigma2_lag = (
        predictive_lag_reduction
        * eta
        * sigma2_temporal
        * (f_piston_hz / servo_bw) ** (5.0 / 3.0)
    )
    sigma2_iid = (
        sigma2_lag
        + sigma2_photon
        + (1.0 - eta) * sigma2_temporal
        + sigma2_fitting
        + sigma2_wfs
    )
    sigma2_uncorrected = sigma2_temporal + sigma2_fitting + sigma2_wfs
    expected_nm = math.sqrt(min(sigma2_iid, sigma2_uncorrected)) * (
        _DEFAULT_WAVELENGTH_CENTER_M * 1.0e9 / (2.0 * math.pi)
    )

    assert residual_nm == pytest.approx(expected_nm, rel=1.0e-6)
    assert payload["f_sample_hz"] == pytest.approx(2.0 * servo_bw)
    assert payload["f_3db_hz"] == pytest.approx(servo_bw)
    assert payload["residual_common_differential_nm"] == pytest.approx(0.0)


def test_closed_loop_disturbances_publish_common_mode_split(
    tmp_path: Path,
) -> None:
    """Inter-telescope piston and vibrations are common differential terms.

    With a small intra-pupil AO piston residual, the published total is
    dominated by the common differential disturbances and the vibration
    lines outweigh the inter-telescope piston after the servo.
    """

    state = _state(
        atmos_piston_rms_opd_nm=200.0,
        diameter_m=10.0,
        bright_port_rate_hz=1.0e8,
        n_ports=4,
    )
    ao_payload = {
        "piston_opd_nm": {
            "temporal": 16.0,
            "fitting": 8.0,
            "wfs_noise": 6.0,
            "total": math.sqrt(16.0**2 + 8.0**2 + 6.0**2),
        },
        "piston_subaperture_diameter_m": 1.93,
        "effective_wind_speed_m_s": 16.643,
        "r0_500nm_m": 0.15,
    }
    state = replace(
        state,
        wavefront=replace(state.wavefront).add_component("ao_residual", ao_payload),
    )
    stage = ClosedLoopFringeTracker(
        {
            "servo_bw_hz": None,
            "use_full_piston_servo_residual": True,
            "n_modes": 24,
            "n_spectral_channels": 20,
            "fringe_visibility": 0.9,
            "predictive_lag_reduction": 0.33,
            "inter_telescope_piston": True,
            "outer_scale_m": 25.0,
            "vibration_line_frequencies_hz": (12.0, 29.0, 58.0, 88.0, 150.0, 200.0),
            "vibration_line_rms_nm": (30.0, 40.0, 25.0, 20.0, 10.0, 10.0),
            "vibration_broadband_rms_nm": 30.0,
        },
        _context(tmp_path),
    )

    result = stage.apply(state)
    payload = _fringe_tracking_payload(result)

    assert payload["servo_bandwidth_source"] == "auto"
    assert payload["f_3db_hz"] == pytest.approx(
        payload["f_sample_hz"] * payload["servo_bandwidth_fraction"]
    )
    assert 20.0 <= payload["f_sample_hz"] <= payload["sample_rate_max_hz"]
    assert 25.0 < payload["residual_rms_opd_nm"] < 60.0
    assert payload["residual_common_differential_nm"] > 0.0
    assert payload["sigma_vibration_residual_nm"] > payload[
        "sigma_inter_telescope_residual_nm"
    ]
    assert payload["residual_common_differential_nm"] > payload[
        "residual_iid_per_mode_nm"
    ]


def test_closed_loop_sample_rate_optimizer_lands_interior(tmp_path: Path) -> None:
    """The numeric sample-rate scan finds an interior photon/disturbance optimum.

    A faint bright-port budget makes the per-frame photon term grow with sample
    rate while the disturbance rejection improves, so the total residual
    variance has a true interior minimum (not pinned to either grid endpoint).
    """

    state = _state(
        atmos_piston_rms_opd_nm=200.0,
        diameter_m=10.0,
        bright_port_rate_hz=2.0e5,
        n_ports=4,
    )
    ao_payload = {
        "piston_opd_nm": {
            "temporal": 100.0,
            "fitting": 30.0,
            "wfs_noise": 20.0,
            "total": math.sqrt(100.0**2 + 30.0**2 + 20.0**2),
        },
        "piston_subaperture_diameter_m": 1.93,
        "effective_wind_speed_m_s": 16.643,
        "r0_500nm_m": 0.15,
    }
    state = replace(
        state,
        wavefront=replace(state.wavefront).add_component("ao_residual", ao_payload),
    )
    stage = ClosedLoopFringeTracker(
        {
            "servo_bw_hz": None,
            "use_full_piston_servo_residual": True,
            "n_modes": 24,
            "n_spectral_channels": 20,
            "fringe_visibility": 0.9,
            "predictive_lag_reduction": 0.33,
            "sample_rate_max_hz": 10000.0,
            "inter_telescope_piston": True,
            "outer_scale_m": 25.0,
            "vibration_line_frequencies_hz": (12.0, 29.0, 58.0, 88.0, 150.0, 200.0),
            "vibration_line_rms_nm": (30.0, 40.0, 25.0, 20.0, 10.0, 10.0),
            "vibration_broadband_rms_nm": 30.0,
        },
        _context(tmp_path),
    )

    result = stage.apply(state)
    payload = _fringe_tracking_payload(result)

    f_sample = payload["f_sample_hz"]
    f_sample_max = payload["sample_rate_max_hz"]
    assert payload["servo_bandwidth_source"] == "auto"
    assert 20.0 < f_sample < f_sample_max

    residual_opt_nm = payload["residual_rms_opd_nm"]

    low_payload = _fringe_tracking_payload(
        ClosedLoopFringeTracker(
            {
                "servo_bw_hz": 0.05 * 20.0,
                "use_full_piston_servo_residual": True,
                "n_modes": 24,
                "n_spectral_channels": 20,
                "fringe_visibility": 0.9,
                "predictive_lag_reduction": 0.33,
                "inter_telescope_piston": True,
                "outer_scale_m": 25.0,
                "vibration_line_frequencies_hz": (
                    12.0,
                    29.0,
                    58.0,
                    88.0,
                    150.0,
                    200.0,
                ),
                "vibration_line_rms_nm": (30.0, 40.0, 25.0, 20.0, 10.0, 10.0),
                "vibration_broadband_rms_nm": 30.0,
            },
            _context(tmp_path),
        ).apply(state)
    )
    high_payload = _fringe_tracking_payload(
        ClosedLoopFringeTracker(
            {
                "servo_bw_hz": 0.05 * f_sample_max,
                "use_full_piston_servo_residual": True,
                "n_modes": 24,
                "n_spectral_channels": 20,
                "fringe_visibility": 0.9,
                "predictive_lag_reduction": 0.33,
                "inter_telescope_piston": True,
                "outer_scale_m": 25.0,
                "vibration_line_frequencies_hz": (
                    12.0,
                    29.0,
                    58.0,
                    88.0,
                    150.0,
                    200.0,
                ),
                "vibration_line_rms_nm": (30.0, 40.0, 25.0, 20.0, 10.0, 10.0),
                "vibration_broadband_rms_nm": 30.0,
            },
            _context(tmp_path),
        ).apply(state)
    )

    assert residual_opt_nm <= low_payload["residual_rms_opd_nm"]
    assert residual_opt_nm <= high_payload["residual_rms_opd_nm"]


def test_closed_loop_control_model_classical_preserves_default_results(
    tmp_path: Path,
) -> None:
    state = _with_ao_residual(
        _state(atmos_piston_rms_opd_nm=200.0, bright_port_rate_hz=1.0e8),
        temporal_nm=16.0,
        fitting_nm=8.0,
        wfs_noise_nm=6.0,
    )
    common_params = {
        "servo_bw_hz": 1000.0,
        "use_full_piston_servo_residual": True,
        "n_modes": 24,
        "n_spectral_channels": 20,
        "fringe_visibility": 0.9,
        "predictive_lag_reduction": 0.33,
        "inter_telescope_piston": True,
        "outer_scale_m": 25.0,
        "vibration_line_frequencies_hz": (12.0, 29.0, 58.0, 88.0, 150.0, 200.0),
        "vibration_line_rms_nm": (30.0, 40.0, 25.0, 20.0, 10.0, 10.0),
        "vibration_broadband_rms_nm": 30.0,
    }

    default_payload = _fringe_tracking_payload(
        ClosedLoopFringeTracker(common_params, _context(tmp_path)).apply(state)
    )
    classical_payload = _fringe_tracking_payload(
        ClosedLoopFringeTracker(
            {**common_params, "control_model": "classical"},
            _context(tmp_path),
        ).apply(state)
    )

    for key in (
        "residual_rms_opd_nm",
        "residual_common_differential_nm",
        "residual_iid_per_mode_nm",
        "servo_bandwidth_hz",
        "sigma2_photon_rad2",
    ):
        assert classical_payload[key] == pytest.approx(default_payload[key], rel=1e-12)


def test_closed_loop_classical_accelerometer_fusion_reduces_vibration_variance(
    tmp_path: Path,
) -> None:
    state = _with_small_ao(
        _state(
            atmos_piston_rms_opd_nm=200.0,
            diameter_m=10.0,
            bright_port_rate_hz=1.0e8,
            n_ports=4,
        )
    )
    common_params = {
        "servo_bw_hz": 100.0,
        "control_model": "classical",
        "vibration_broadband_rms_nm": 30.0,
        "fusion_coherent_fraction": 0.7,
        "fusion_broadband_suppression": 10.0,
        "n_modes": 24,
    }

    raw_payload = _fringe_tracking_payload(
        ClosedLoopFringeTracker(common_params, _context(tmp_path)).apply(state)
    )
    fused_payload = _fringe_tracking_payload(
        ClosedLoopFringeTracker(
            {**common_params, "accelerometer_fusion": True},
            _context(tmp_path),
        ).apply(state)
    )

    expected_variance_fraction = 0.7 / 10.0**2 + 0.3
    assert fused_payload["sigma2_vibration_rad2"] < raw_payload[
        "sigma2_vibration_rad2"
    ]
    assert (
        fused_payload["sigma2_vibration_rad2"]
        / raw_payload["sigma2_vibration_rad2"]
    ) == pytest.approx(expected_variance_fraction, rel=1.0e-12)


def test_closed_loop_photon_kalman_j4_defaults_land_in_band(
    tmp_path: Path,
) -> None:
    per_port_rate_hz = (
        _DEFAULT_WAVELENGTH_CENTER_M / (2.0 * math.pi * 0.9 * 1.29e-10)
    ) ** 2
    state = _with_ao_residual(
        _state(
            atmos_piston_rms_opd_nm=200.0,
            diameter_m=10.0,
            bright_port_rate_hz=4.0 * per_port_rate_hz,
            n_ports=4,
        ),
        temporal_nm=1.0,
        fitting_nm=1.0,
        wfs_noise_nm=1.0,
    )
    stage = ClosedLoopFringeTracker(
        {
            "control_model": "photon_kalman",
            "use_full_piston_servo_residual": True,
            "n_modes": 24,
            "n_spectral_channels": 20,
            "fringe_visibility": 0.9,
            "inter_telescope_piston": True,
            "outer_scale_m": 25.0,
            "vibration_line_frequencies_hz": (12.0, 29.0, 58.0, 88.0, 150.0, 200.0),
            "vibration_line_rms_nm": (30.0, 40.0, 25.0, 20.0, 10.0, 10.0),
            "vibration_broadband_rms_nm": 30.0,
            "accelerometer_fusion": True,
        },
        _context(tmp_path),
    )

    payload = _fringe_tracking_payload(stage.apply(state))

    assert payload["control_model"] == "photon_kalman"
    assert 10.0 < payload["residual_common_differential_nm"] < 16.0
    assert 5.0e-5 < payload["photon_kalman_window_s"] < 5.0e-4
    assert payload["servo_bandwidth_hz"] == pytest.approx(
        1.0 / (2.0 * (payload["photon_kalman_window_s"] + 5.0e-5)),
        rel=1.0e-9,
    )
    assert payload["photon_kalman_estimation_nm"] > 0.0
    assert payload["photon_kalman_prediction_nm"] > 0.0
    assert payload["photon_kalman_line_floor_nm"] == pytest.approx(
        math.sqrt(6.0) * 2.0,
        rel=1.0e-12,
    )
    assert payload["tracker_rate_clipped"] is False


def test_closed_loop_photon_kalman_starved_rate_slows_loop(
    tmp_path: Path,
) -> None:
    per_port_rate_hz = (
        _DEFAULT_WAVELENGTH_CENTER_M / (2.0 * math.pi * 0.9 * 1.29e-10)
    ) ** 2
    base_state = _with_ao_residual(
        _state(
            atmos_piston_rms_opd_nm=200.0,
            diameter_m=10.0,
            bright_port_rate_hz=4.0 * per_port_rate_hz,
            n_ports=4,
        ),
        temporal_nm=1.0,
        fitting_nm=1.0,
        wfs_noise_nm=1.0,
    )
    starved_state = _with_ao_residual(
        _state(
            atmos_piston_rms_opd_nm=200.0,
            diameter_m=10.0,
            bright_port_rate_hz=4.0 * per_port_rate_hz / 1000.0,
            n_ports=4,
        ),
        temporal_nm=1.0,
        fitting_nm=1.0,
        wfs_noise_nm=1.0,
    )
    params = {
        "control_model": "photon_kalman",
        "use_full_piston_servo_residual": True,
        "n_modes": 24,
        "n_spectral_channels": 20,
        "fringe_visibility": 0.9,
        "inter_telescope_piston": True,
        "outer_scale_m": 25.0,
        "vibration_line_frequencies_hz": (12.0, 29.0, 58.0, 88.0, 150.0, 200.0),
        "vibration_line_rms_nm": (30.0, 40.0, 25.0, 20.0, 10.0, 10.0),
        "vibration_broadband_rms_nm": 30.0,
        "accelerometer_fusion": True,
    }

    baseline_payload = _fringe_tracking_payload(
        ClosedLoopFringeTracker(params, _context(tmp_path)).apply(base_state)
    )
    starved_payload = _fringe_tracking_payload(
        ClosedLoopFringeTracker(params, _context(tmp_path)).apply(starved_state)
    )

    assert starved_payload["residual_common_differential_nm"] > baseline_payload[
        "residual_common_differential_nm"
    ]
    assert starved_payload["photon_kalman_window_s"] > baseline_payload[
        "photon_kalman_window_s"
    ]


def test_closed_loop_photon_kalman_snspd_rate_cap(
    tmp_path: Path,
) -> None:
    state = _with_ao_residual(
        _state(
            atmos_piston_rms_opd_nm=200.0,
            diameter_m=10.0,
            bright_port_rate_hz=16.0 * 4.3e7,
            n_ports=4,
        ),
        temporal_nm=1.0,
        fitting_nm=1.0,
        wfs_noise_nm=1.0,
    )
    common_params = {
        "control_model": "photon_kalman",
        "use_full_piston_servo_residual": True,
        "n_modes": 24,
        "n_spectral_channels": 20,
        "fringe_visibility": 0.9,
        "inter_telescope_piston": True,
        "outer_scale_m": 25.0,
        "vibration_line_frequencies_hz": (12.0, 29.0, 58.0, 88.0, 150.0, 200.0),
        "vibration_line_rms_nm": (30.0, 40.0, 25.0, 20.0, 10.0, 10.0),
        "vibration_broadband_rms_nm": 30.0,
        "accelerometer_fusion": True,
    }

    clipped = _fringe_tracking_payload(
        ClosedLoopFringeTracker(
            {**common_params, "tracker_n_detectors_per_port": 1},
            _context(tmp_path),
        ).apply(state)
    )
    unclipped = _fringe_tracking_payload(
        ClosedLoopFringeTracker(
            {**common_params, "tracker_n_detectors_per_port": 4},
            _context(tmp_path),
        ).apply(state)
    )

    assert clipped["tracker_rate_clipped"] is True
    assert clipped["tracker_rate_per_port_hz"] == pytest.approx(1.72e8)
    assert clipped["tracker_usable_rate_per_port_hz"] == pytest.approx(1.0e8)
    assert unclipped["tracker_rate_clipped"] is False
    assert unclipped["tracker_usable_rate_per_port_hz"] == pytest.approx(1.72e8)
    assert clipped["residual_common_differential_nm"] > unclipped[
        "residual_common_differential_nm"
    ]


def _representative_wavelength_m_default() -> float:
    """Match the default ``_state`` wavelength grid centroid (1.25 µm)."""
    return 1.25e-6


def test_closed_loop_uses_effective_baseline_param_override(
    tmp_path: Path,
) -> None:
    """Effective baseline Param overrides geometry.diameter_m for f_piston."""

    state = _with_ao_residual(
        _state(
            atmos_piston_rms_opd_nm=500.0,
            diameter_m=10.0,
            bright_port_rate_hz=1.0e18,
        ),
        temporal_nm=500.0,
        fitting_nm=0.0,
        wfs_noise_nm=0.0,
        subaperture_diameter_m=10.0,
    )

    stage_geom = _stage(_context(tmp_path), n_modes=2, predictive_lag_reduction=1.0)
    stage_override = _stage(
        _context(tmp_path),
        effective_baseline_m=20.0,  # 2x baseline => f_piston halved
        n_modes=2,
        predictive_lag_reduction=1.0,
    )

    residual_geom = stage_geom.apply(state).opd_budget.components[
        "fringe_tracking.closed_loop_residual"
    ].rms_opd_nm
    residual_override = stage_override.apply(state).opd_budget.components[
        "fringe_tracking.closed_loop_residual"
    ].rms_opd_nm

    # f_piston halved => sigma^2_lag drops by 2^(5/3); sigma by 2^(5/6).
    assert residual_geom / residual_override == pytest.approx(
        2.0 ** (5.0 / 6.0), rel=1.0e-6
    )


def test_closed_loop_raises_when_upstream_component_missing(
    tmp_path: Path,
) -> None:
    state = _state(atmos_piston_rms_opd_nm=None)
    stage = _stage(_context(tmp_path))

    with pytest.raises(ValueError, match="atmos_piston"):
        stage.apply(state)


def test_closed_loop_raises_when_photon_rates_missing(tmp_path: Path) -> None:
    state = _state()
    state = replace(
        state,
        photon_rates=PhotonRates(
            wavelength_grid_um=_DEFAULT_WAVELENGTH_GRID_UM,
            port_labels=(),
            rates_hz=None,
        ),
    )
    stage = _stage(_context(tmp_path))

    with pytest.raises(ValueError, match="photon_rates"):
        stage.apply(state)


def test_closed_loop_raises_when_geometry_diameter_missing(tmp_path: Path) -> None:
    state = _state(diameter_m=None)
    state = replace(state, geometry=Geometry())
    stage = _stage(_context(tmp_path))

    with pytest.raises(ValueError, match="effective_baseline_m"):
        stage.apply(state)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"servo_bw_hz": -1.0}, "servo_bw_hz"),
        ({"fringe_visibility": 1.5}, "fringe_visibility"),
        ({"fringe_visibility": 0.0}, "fringe_visibility"),
        ({"n_spectral_channels": 0}, "n_spectral_channels"),
        ({"n_spectral_eff": -2.5}, "n_spectral_eff"),
        ({"predictive_lag_reduction": 0.0}, "predictive_lag_reduction"),
        ({"predictive_lag_reduction": 1.5}, "predictive_lag_reduction"),
        ({"wind_speed_m_s": -10.0}, "wind_speed_m_s"),
        ({"superseded_components": ()}, "superseded_components"),
    ],
)
def test_closed_loop_params_validation(
    tmp_path: Path, overrides: dict[str, object], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        _stage(_context(tmp_path), **overrides)


def test_closed_loop_auto_bandwidth_uses_subaperture_f_char(tmp_path: Path) -> None:
    """Phase 2: servo_bw_hz=None auto-derives from KILO's photon-limited
    optimum and f_char uses the subaperture diameter published by
    upstream ao.pyramid_wfs."""

    # Build a state with an ao_residual component that mirrors what
    # pyramid_wfs publishes.
    state = _state(
        atmos_piston_rms_opd_nm=200.0,
        diameter_m=10.0,
        bright_port_rate_hz=1.0e8,
        n_ports=4,
    )
    ao_payload = {
        "piston_opd_nm": {
            "temporal": 100.0,
            "fitting": 30.0,
            "wfs_noise": 20.0,
            "total": math.sqrt(100.0**2 + 30.0**2 + 20.0**2),
        },
        "piston_subaperture_diameter_m": 1.93,
    }
    wavefront = replace(
        state.wavefront,
        strehl_ratio=tuple(0.9 for _ in state.field.wavelength_grid_um),
    ).add_component("ao_residual", ao_payload)
    state = replace(state, wavefront=wavefront)

    stage = ClosedLoopFringeTracker(
        {
            "servo_bw_hz": None,
            "use_full_piston_servo_residual": True,
            "n_modes": 24,
            "n_spectral_channels": 20,
            "fringe_visibility": 0.9,
            "wind_speed_m_s": 10.0,
            "predictive_lag_reduction": 1.0,
        },
        _context(tmp_path),
    )
    result = stage.apply(state)

    payload: dict[str, float] | None = None
    for name, value in result.results.components.items():
        if name.startswith("fringe_tracking.closed_loop"):
            payload = value
            break
    assert payload is not None
    assert payload["servo_bandwidth_source"] == "auto"
    # f_char = 0.134 * 10 / 1.93 ~ 0.694 (subaperture, not full aperture).
    assert payload["f_piston_hz"] == pytest.approx(0.134 * 10.0 / 1.93, rel=1e-12)
    # Auto-bandwidth in the J=4 Keck regime is ~10 Hz - 1 kHz.
    assert 5.0 < payload["servo_bandwidth_hz"] < 5000.0


def test_closed_loop_f_char_prefers_ao_effective_wind_payload(tmp_path: Path) -> None:
    state = _state(
        atmos_piston_rms_opd_nm=200.0,
        diameter_m=10.0,
        bright_port_rate_hz=1.0e8,
        n_ports=4,
    )
    ao_payload = {
        "piston_opd_nm": {
            "temporal": 100.0,
            "fitting": 30.0,
            "wfs_noise": 20.0,
            "total": math.sqrt(100.0**2 + 30.0**2 + 20.0**2),
        },
        "piston_subaperture_diameter_m": 2.0,
        "effective_wind_speed_m_s": 25.0,
    }
    wavefront = replace(state.wavefront).add_component("ao_residual", ao_payload)
    state = replace(state, wavefront=wavefront)

    stage = _stage(
        _context(tmp_path),
        use_full_piston_servo_residual=True,
        n_modes=24,
        n_spectral_channels=20,
        wind_speed_m_s=10.0,
        predictive_lag_reduction=1.0,
    )
    result = stage.apply(state)

    payload = next(
        value
        for name, value in result.results.components.items()
        if name.startswith("fringe_tracking.closed_loop")
    )
    assert payload["f_piston_hz"] == pytest.approx(0.134 * 25.0 / 2.0, rel=1e-12)


def test_closed_loop_auto_bandwidth_ignores_dark_port_rates(tmp_path: Path) -> None:
    """Only bright_* ports should enter the photon-limited bandwidth estimate."""

    bright_spectrum = np.full(3, 1.0e8 / 3.0, dtype=np.float64)
    no_dark_rates = np.vstack(
        [
            bright_spectrum,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
        ],
    )
    bright_with_large_dark_rates = np.vstack(
        [
            bright_spectrum,
            np.full(3, 1.0e12, dtype=np.float64),
            np.full(3, 1.0e12, dtype=np.float64),
        ],
    )

    def _with_auto_ao(state: SimulationState) -> SimulationState:
        ao_payload = {
            "piston_opd_nm": {
                "temporal": 100.0,
                "fitting": 30.0,
                "wfs_noise": 20.0,
                "total": math.sqrt(100.0**2 + 30.0**2 + 20.0**2),
            },
            "piston_subaperture_diameter_m": 1.93,
        }
        wavefront = replace(
            state.wavefront,
            strehl_ratio=tuple(0.9 for _ in state.field.wavelength_grid_um),
        ).add_component("ao_residual", ao_payload)
        return replace(state, wavefront=wavefront)

    def _auto_bandwidth_hz(state: SimulationState) -> float:
        stage = ClosedLoopFringeTracker(
            {
                "servo_bw_hz": None,
                "use_full_piston_servo_residual": True,
                "n_modes": 24,
                "n_spectral_channels": 20,
                "fringe_visibility": 0.9,
                "wind_speed_m_s": 10.0,
                "predictive_lag_reduction": 1.0,
            },
            _context(tmp_path),
        )
        result = stage.apply(state)
        for name, value in result.results.components.items():
            if name.startswith("fringe_tracking.closed_loop"):
                return float(value["servo_bandwidth_hz"])
        raise AssertionError("missing fringe tracking result component")

    baseline = _auto_bandwidth_hz(_with_auto_ao(_state(rates_hz=no_dark_rates)))
    with_dark = _auto_bandwidth_hz(
        _with_auto_ao(_state(rates_hz=bright_with_large_dark_rates)),
    )

    assert with_dark == pytest.approx(baseline, rel=1e-12)


def test_closed_loop_pinned_bandwidth_marks_source_as_params(tmp_path: Path) -> None:
    """Pinning servo_bw_hz explicitly bypasses the sample-rate optimizer."""

    state = _state(
        atmos_piston_rms_opd_nm=200.0,
        diameter_m=10.0,
        bright_port_rate_hz=1.0e8,
        n_ports=4,
    )
    ao_payload = {
        "piston_opd_nm": {
            "temporal": 100.0,
            "fitting": 30.0,
            "wfs_noise": 20.0,
            "total": math.sqrt(100.0**2 + 30.0**2 + 20.0**2),
        },
        "piston_subaperture_diameter_m": 1.93,
        "effective_wind_speed_m_s": 10.0,
        "r0_500nm_m": 0.15,
    }
    state = replace(
        state,
        wavefront=replace(state.wavefront).add_component("ao_residual", ao_payload),
    )
    common_params = {
        "use_full_piston_servo_residual": True,
        "n_modes": 24,
        "n_spectral_channels": 20,
        "fringe_visibility": 0.9,
        "predictive_lag_reduction": 0.33,
    }
    pinned_stage = ClosedLoopFringeTracker(
        {
            **common_params,
            "servo_bw_hz": 1500.0,
        },
        _context(tmp_path),
    )
    auto_stage = ClosedLoopFringeTracker(
        {
            **common_params,
            "servo_bw_hz": None,
        },
        _context(tmp_path),
    )

    payload = _fringe_tracking_payload(pinned_stage.apply(state))
    auto_payload = _fringe_tracking_payload(auto_stage.apply(state))

    assert payload["servo_bandwidth_source"] == "params"
    assert payload["servo_bandwidth_hz"] == pytest.approx(1500.0)
    assert payload["f_3db_hz"] == pytest.approx(1500.0)
    assert payload["f_sample_hz"] == pytest.approx(
        1500.0 / payload["servo_bandwidth_fraction"]
    )
    assert auto_payload["servo_bandwidth_source"] == "auto"
    assert auto_payload["f_3db_hz"] != pytest.approx(payload["f_3db_hz"])


def test_closed_loop_simultaneous_bands_scale_ft_photon_budget(
    tmp_path: Path,
) -> None:
    """J/H/K simultaneous operation improves the FT shot-noise budget."""

    state = _with_ao_residual(
        _state(atmos_piston_rms_opd_nm=0.0, bright_port_rate_hz=1.0e6),
        temporal_nm=0.0,
        fitting_nm=0.0,
        wfs_noise_nm=0.0,
    )
    state = replace(state, scene=Scene(star_jmag=5.0))

    single = _stage(_context(tmp_path), servo_bw_hz=2000.0)
    multi = _stage(
        _context(tmp_path),
        servo_bw_hz=2000.0,
        simultaneous_science_bands=("J", "H", "K"),
        simultaneous_science_band_transmissions=(0.985, 0.97, 0.92),
    )

    single_result = single.apply(state)
    multi_result = multi.apply(state)
    single_residual = single_result.opd_budget.components[
        "fringe_tracking.closed_loop_residual"
    ].rms_opd_nm
    multi_residual = multi_result.opd_budget.components[
        "fringe_tracking.closed_loop_residual"
    ].rms_opd_nm

    payload = next(
        value
        for name, value in multi_result.results.components.items()
        if name.startswith("fringe_tracking.closed_loop")
    )
    assert payload["simultaneous_science_bands"] == ("J", "H", "K")
    assert payload["simultaneous_science_band_rate_scale"] > 1.0
    assert multi_residual < single_residual

    # OPD information per photon scales as 1/lambda^2 (row 44): each band
    # counts N_b T_b (lambda_ref / lambda_b)^2 photons at the 1.25 um
    # reference, over the unweighted current-grid rate.
    from nullsim.physics.photometry import photon_rate_per_m2_per_bin
    from nullsim.stages._science_bands import grid_photon_rate_per_m2

    current = grid_photon_rate_per_m2(
        magnitude=5.0,
        wavelength_grid_um=_DEFAULT_WAVELENGTH_GRID_UM,
        band="J",
        stage_name="test",
    ) * 0.985
    information = sum(
        photon_rate_per_m2_per_bin(5.0, center_um, width_um, band=band)
        * transmission
        * (1.25 / center_um) ** 2
        for band, center_um, width_um, transmission in (
            ("J", 1.25, 0.16, 0.985),
            ("H", 1.65, 0.30, 0.97),
            ("K", 2.20, 0.40, 0.92),
        )
    )
    assert payload["simultaneous_science_band_information_scale"] == pytest.approx(
        information / current, rel=1e-12
    )
    # Pure shot noise: the residual variance ratio is the inverse scale.
    assert (single_residual / multi_residual) ** 2 == pytest.approx(
        payload["simultaneous_science_band_information_scale"], rel=1e-9
    )


def test_closed_loop_classical_starved_photons_keep_shot_noise_in_residual(
    tmp_path: Path,
) -> None:
    """A photon-starved classical loop must not hide its shot noise.

    The pre-fix code capped the full residual at the open-loop AO total,
    which (a) dropped the sensor shot noise from the published residual at
    faint magnitudes and (b) made the iid term sample-rate-independent so
    the bandwidth optimizer always pinned at sample_rate_max (2026-06-10
    classical-ablation flux-scaling bug). The published residual must equal
    the raw lag + shot + uncorrectable + irreducible sum, the loop must
    slow down at low flux, and the residual must degrade with flux.
    """

    ao_payload = {
        "piston_opd_nm": {
            "temporal": 10.0,
            "fitting": 0.3,
            "wfs_noise": 1.5,
            "total": math.sqrt(10.0**2 + 0.3**2 + 1.5**2),
        },
        "piston_subaperture_diameter_m": 1.93,
        "effective_wind_speed_m_s": 16.643,
        "r0_500nm_m": 0.15,
    }
    params = {
        "servo_bw_hz": None,
        "use_full_piston_servo_residual": True,
        "control_model": "classical",
        "n_modes": 24,
        "n_spectral_channels": 20,
        "fringe_visibility": 0.9,
        "predictive_lag_reduction": 0.33,
        "inter_telescope_piston": True,
        "outer_scale_m": 25.0,
        "vibration_line_frequencies_hz": (12.0, 29.0, 58.0, 88.0, 150.0, 200.0),
        "vibration_line_rms_nm": (30.0, 40.0, 25.0, 20.0, 10.0, 10.0),
        "vibration_broadband_rms_nm": 30.0,
    }

    payloads = {}
    for label, rate_hz in (("rich", 1.0e8), ("starved", 2.0e4)):
        state = _state(
            atmos_piston_rms_opd_nm=200.0,
            diameter_m=10.0,
            bright_port_rate_hz=rate_hz,
            n_ports=4,
        )
        state = replace(
            state,
            wavefront=replace(state.wavefront).add_component(
                "ao_residual", ao_payload
            ),
        )
        stage = ClosedLoopFringeTracker(dict(params), _context(tmp_path))
        payloads[label] = _fringe_tracking_payload(stage.apply(state))

    starved = payloads["starved"]
    rich = payloads["rich"]

    # Published residual is the raw component sum — shot noise included,
    # no open-loop cap — plus the common differential disturbance term.
    assert starved["sigma2_residual_rad2"] == pytest.approx(
        starved["sigma2_lag_reduced_rad2"]
        + starved["sigma2_photon_rad2"]
        + starved["sigma2_uncorrectable_temporal_rad2"]
        + starved["sigma2_irreducible_rad2"]
        + starved["sigma2_common_differential_rad2"],
        rel=1e-9,
    )
    # The optimizer trades shot noise against disturbance rejection: a
    # starved loop runs slower than a photon-rich one instead of pinning
    # at sample_rate_max.
    assert starved["f_sample_hz"] < rich["f_sample_hz"]
    assert starved["f_sample_hz"] < 0.99 * starved["sample_rate_max_hz"]
    # And the residual honestly degrades with flux.
    assert (
        starved["residual_rms_opd_nm"] > 3.0 * rich["residual_rms_opd_nm"]
    )


def test_tracker_photon_budget_pools_dual_pol_arm_channels(tmp_path: Path) -> None:
    """Dual-pol arm channels of one tracker port pool into one estimator.

    The detector publishes each physical bright port as n_pol_arms
    channels carrying half the photons each. Both arms see the same
    piston fringe, so the per-port budget must equal the scalar
    single-channel case — dividing by detector channels instead would
    double the assumed shot variance.
    """

    from nullsim.stages.fringe_tracking.closed_loop import _tracker_photon_budget

    wavelengths = _DEFAULT_WAVELENGTH_GRID_UM
    total_bright_hz = 1.0e8
    params = ClosedLoopFringeTracker(
        params={"control_model": "photon_kalman"},
        context=_context(tmp_path),
    ).params_model

    scalar_state = _state(bright_port_rate_hz=total_bright_hz, n_ports=4)
    scalar_budget = _tracker_photon_budget(
        scalar_state,
        params,
        "fringe_tracking.closed_loop",
        sample_rate_hz=1000.0,
    )

    n_wl = len(wavelengths)
    dual_rates = np.zeros((8, n_wl), dtype=np.float64)
    # Port-major, arm-minor: rows 0/1 are bright_0's two arms, each with
    # half the scalar fixture's bright photons (which live in row 0 only).
    dual_rates[0, :] = total_bright_hz / (2.0 * n_wl * 4)
    dual_rates[1, :] = total_bright_hz / (2.0 * n_wl * 4)
    dual_labels = ("bright_0_A", "bright_0_B") + tuple(
        f"dark_{idx}_{arm}" for idx in range(3) for arm in ("A", "B")
    )
    dual_state = _state(bright_port_rate_hz=total_bright_hz, n_ports=4)
    dual_state = replace(
        dual_state,
        photon_rates=PhotonRates(
            wavelength_grid_um=wavelengths,
            port_labels=dual_labels,
            rates_hz=dual_rates,
        ),
        results=dual_state.results.add_component(
            "detector.mkid",
            {"n_pol_arms": 2},
        ),
    )
    dual_budget = _tracker_photon_budget(
        dual_state,
        params,
        "fringe_tracking.closed_loop",
        sample_rate_hz=1000.0,
    )

    assert dual_budget.rate_per_port_hz == pytest.approx(
        scalar_budget.rate_per_port_hz
    )
    # Two arm chips carry two tracker detector sets, so the per-port
    # capacity doubles.
    assert dual_budget.rate_cap_hz == pytest.approx(2.0 * scalar_budget.rate_cap_hz)


def test_inter_telescope_piston_uses_geometry_baseline_and_primary(
    tmp_path: Path,
) -> None:
    """Keck-Keck (85 m, 10 m primaries, r0 0.15 m, L0 25 m): 1.688 um RMS.

    The value is the independent Conan et al. (1995) quadrature of review
    2026-11-07 Issue 3; the old code returned 14.81 um. Without aperture
    positions the pair is treated as decorrelated (B -> inf).
    """

    from nullsim.stages.fringe_tracking.closed_loop import (
        _inter_telescope_piston_variance_m2,
    )

    params = _stage(_context(tmp_path), inter_telescope_piston=True).params_model
    state = _with_small_ao(_state(diameter_m=10.0))
    keck = replace(
        state,
        geometry=replace(
            state.geometry,
            aperture_positions_enu_m=np.array([[0.0, 0.0, 0.0], [85.0, 0.0, 1.0]]),
        ),
    )

    sigma_um = math.sqrt(
        _inter_telescope_piston_variance_m2(keck, params, "test")
    ) * 1.0e6
    assert sigma_um == pytest.approx(1.688, abs=1.5e-3)
    unknown_um = math.sqrt(
        _inter_telescope_piston_variance_m2(state, params, "test")
    ) * 1.0e6
    assert unknown_um == pytest.approx(sigma_um, rel=1e-3)

    # The stage publishes the same disturbance for the debug report.
    stage = _stage(_context(tmp_path), inter_telescope_piston=True)
    published = stage.apply(keck).results.components["fringe_tracking.closed_loop"]
    published_um = published["inter_telescope_uncorrected_piston_rms_m"] * 1.0e6
    assert published_um == pytest.approx(sigma_um, rel=1e-12)


def test_closed_loop_counts_logical_bright_ports_for_dual_pol_arms(
    tmp_path: Path,
) -> None:
    """Row 45: bright_0_A/bright_0_B are one logical port, not two."""

    n_wl = len(_DEFAULT_WAVELENGTH_GRID_UM)
    labels = ("bright_0_A", "bright_0_B", "bright_1_A", "bright_1_B") + tuple(
        f"dark_{idx}_{arm}" for idx in range(2) for arm in ("A", "B")
    )
    state = _with_small_ao(_state())
    state = replace(
        state,
        photon_rates=PhotonRates(
            wavelength_grid_um=_DEFAULT_WAVELENGTH_GRID_UM,
            port_labels=labels,
            rates_hz=np.full((len(labels), n_wl), 1.0e6),
        ),
        results=state.results.add_component("detector.mkid", {"n_pol_arms": 2}),
    )

    payload = _fringe_tracking_payload(
        _stage(_context(tmp_path), n_modes=24).apply(state)
    )

    assert payload["n_bright_ports"] == 2
    assert payload["piston_sensing_efficiency"] == pytest.approx(
        min(2.0 * 2 * 2.5, 23.0) / 23.0
    )
