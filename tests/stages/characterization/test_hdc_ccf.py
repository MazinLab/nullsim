"""Tests for the characterization.hdc_ccf cross-correlation stage."""

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
from nullsim.stages.characterization.hdc_ccf import HdcCcfStage

_N_WAVELENGTHS = 8
_WL_MIN_UM = 1.15
_WL_MAX_UM = 1.35
_STELLAR_RATE_HZ_PER_BIN = 1.0e6
_THROUGHPUT = 0.3
_NULL_DEPTH = 1.0e-4
_PLANET_CONTRAST = 1.0e-6
_EPS_SYSTEMATIC = 1.0e-3
_SERVO_BW_HZ = 100.0
_INTEGRATION_TIME_S = 3600.0
_PLANET_RV_KMS = 30.0


def _context(tmp_path: Path) -> RunContext:
    return RunContext(
        rng=np.random.default_rng(0),
        config_hash="characterization-hdc-ccf",
        sweep_coord_tuple=(),
        instance_id="hdc",
        run_dir=tmp_path,
    )


def _spectro_payload() -> dict[str, object]:
    """A synthetic ``characterization.spectro`` payload on a flat 8-bin band."""

    wl_um = np.linspace(_WL_MIN_UM, _WL_MAX_UM, _N_WAVELENGTHS)
    stellar = np.full(_N_WAVELENGTHS, _STELLAR_RATE_HZ_PER_BIN)
    throughput = np.full(_N_WAVELENGTHS, _THROUGHPUT)
    null = np.full(_N_WAVELENGTHS, _NULL_DEPTH)
    ratio = np.ones(_N_WAVELENGTHS)

    stellar_total = float(stellar.sum())
    planet_rate_hz = _PLANET_CONTRAST * stellar_total * _THROUGHPUT
    leak_rate_hz = _NULL_DEPTH * stellar_total
    systematic_var = (
        _EPS_SYSTEMATIC**2
        * leak_rate_hz**2
        * _INTEGRATION_TIME_S
        / (2.0 * _SERVO_BW_HZ)
    )
    return {
        "wavelengths_m": tuple(float(w) * 1.0e-6 for w in wl_um),
        "detected_stellar_rate_by_wavelength_hz": stellar,
        "detected_stellar_rate_hz": stellar_total,
        "spectro_planet_throughput_per_wavelength": throughput,
        "spectro_on_axis_null_depth_per_wavelength": null,
        "spectro_static_leak_fraction_per_wavelength": null,
        "spectro_band_disk_leak_per_wavelength": np.zeros(_N_WAVELENGTHS),
        "spectro_planet_to_star_ratio_per_wavelength": ratio,
        "realized_spectro_floor": 0.0,
        "planet_signal_rate_hz": planet_rate_hz,
        "leakage_rate_hz": leak_rate_hz,
        "eps_systematic": _EPS_SYSTEMATIC,
        "servo_bw_hz": _SERVO_BW_HZ,
        "S": planet_rate_hz * _INTEGRATION_TIME_S,
        "B": leak_rate_hz * _INTEGRATION_TIME_S,
        "D": 1.0,
        "Sky": 10.0,
        "systematic_var": systematic_var,
        "spectrograph": "awg",
        "awg_resolving_power": 3000.0,
        "awg_n_bins": 200,
    }


def _state(*, with_spectro: bool = True) -> SimulationState:
    wavelengths_um = tuple(
        float(w) for w in np.linspace(_WL_MIN_UM, _WL_MAX_UM, _N_WAVELENGTHS)
    )
    state = SimulationState.empty({"wavelength_grid_um": wavelengths_um})
    state = replace(
        state,
        scene=Scene(
            star_jmag=4.0,
            star_spectral_type="M2V",
            planet_contrast=_PLANET_CONTRAST,
            planet_separation_mas=10.0,
            planet_position_angle_deg=90.0,
            planet_radial_velocity_kms=_PLANET_RV_KMS,
        ),
    )
    if not with_spectro:
        return state
    results = state.results.add_component(
        "characterization.spectro", _spectro_payload()
    )
    return replace(state, results=results)


def _stage(context: RunContext, **overrides: object) -> HdcCcfStage:
    params: dict[str, object] = {
        "integration_time_s": _INTEGRATION_TIME_S,
        "snr_vs_r_points": 6,
    }
    params.update(overrides)
    return HdcCcfStage(params, context)


@pytest.fixture(scope="module")
def payload(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    context = _context(tmp_path_factory.mktemp("hdc"))
    result = _stage(context).apply(_state())
    return result.results.components["characterization.hdc_ccf"]


def test_hdc_ccf_registers_stage_contract(tmp_path: Path) -> None:
    stage = _stage(_context(tmp_path))

    assert "characterization.hdc_ccf" in list_stages()
    assert stage.type_name == "characterization.hdc_ccf"
    assert stage.family == "characterization"
    assert stage.consumes == frozenset({"results", "scene"})
    assert stage.produces == frozenset({"results"})


def test_hdc_ccf_publishes_expected_payload_keys(payload: dict[str, object]) -> None:
    expected = {
        "ccf_snr",
        "ccf_snr_photon_limited",
        "systematic_ccf_leakage",
        "ccf_systematic_rejection",
        "n_effective_lines",
        "n_channels_used",
        "telluric_masked_fraction",
        "snr_vs_resolving_power",
        "snr_vs_resolving_power_r",
        "snr_vs_rv",
        "snr_vs_rv_kms",
        "integration_time_for_target_snr_s",
        "resolving_power_used",
        "planet_template_provenance",
        "star_template_provenance",
        "leak_normalization",
        "planet_normalization",
        "self_subtraction_factor",
        "ccf_model",
    }
    assert expected <= set(payload)


def test_hdc_ccf_snr_is_finite_positive_and_photon_limit_bounds_it(
    payload: dict[str, object],
) -> None:
    snr = float(payload["ccf_snr"])
    photon_limited = float(payload["ccf_snr_photon_limited"])
    assert np.isfinite(snr)
    assert 0.0 < snr <= photon_limited


def test_hdc_ccf_rv_scan_peaks_at_planet_rv(payload: dict[str, object]) -> None:
    rv_kms = np.asarray(payload["snr_vs_rv_kms"], dtype=np.float64)
    ccf = np.asarray(payload["snr_vs_rv"], dtype=np.float64)
    assert rv_kms.size == ccf.size > 1
    idx = int(np.argmax(ccf))
    assert rv_kms[idx] == pytest.approx(_PLANET_RV_KMS, abs=10.0)


def test_hdc_ccf_normalizations_near_unity(payload: dict[str, object]) -> None:
    assert float(payload["leak_normalization"]) == pytest.approx(1.0, rel=0.2)
    assert float(payload["planet_normalization"]) == pytest.approx(1.0, rel=0.2)


def test_hdc_ccf_snr_vs_r_axis_is_ascending_and_finite(
    payload: dict[str, object],
) -> None:
    r_axis = np.asarray(payload["snr_vs_resolving_power_r"], dtype=np.float64)
    snr = np.asarray(payload["snr_vs_resolving_power"], dtype=np.float64)
    assert r_axis.size == snr.size == 6
    assert np.all(np.diff(r_axis) > 0.0)
    assert np.all(np.isfinite(snr))


def test_hdc_ccf_telluric_template_masks_channels(tmp_path: Path) -> None:
    result = _stage(_context(tmp_path), telluric_template="maunakea").apply(_state())
    payload = result.results.components["characterization.hdc_ccf"]

    masked_fraction = float(payload["telluric_masked_fraction"])
    assert 0.0 < masked_fraction < 1.0
    assert int(payload["n_channels_used"]) < int(payload["n_channels"])
    assert "SkyCalc" in str(payload["telluric_template_provenance"])


def test_hdc_ccf_is_deterministic(tmp_path: Path) -> None:
    state = _state()
    first = _stage(_context(tmp_path), snr_vs_r_points=0, rv_scan_points=0).apply(
        state
    )
    second = _stage(_context(tmp_path), snr_vs_r_points=0, rv_scan_points=0).apply(
        state
    )
    left = first.results.components["characterization.hdc_ccf"]
    right = second.results.components["characterization.hdc_ccf"]
    for key in (
        "ccf_snr",
        "ccf_snr_photon_limited",
        "systematic_ccf_leakage",
        "self_subtraction_factor",
        "leak_normalization",
        "planet_normalization",
    ):
        assert left[key] == right[key]


def test_hdc_ccf_errors_clearly_without_spectro_component(tmp_path: Path) -> None:
    stage = _stage(_context(tmp_path))
    with pytest.raises(ValueError, match="characterization.spectro"):
        stage.apply(_state(with_spectro=False))


def test_hdc_ccf_declares_hashed_template_dependencies(tmp_path: Path) -> None:
    stage = _stage(_context(tmp_path), telluric_template="maunakea")
    dependencies = stage.external_dependencies()

    assert dependencies
    assert all(isinstance(dep, ExternalDependency) for dep in dependencies)
    sources = {dep.source for dep in dependencies}
    assert "characterization.hdc_ccf.planet_template" in sources
    assert "characterization.hdc_ccf.telluric_template" in sources
    for dependency in dependencies:
        assert dependency.path is not None
        assert Path(dependency.path).exists()
        assert isinstance(dependency.content_hash, str)
        assert len(dependency.content_hash) == 64


def test_hdc_ccf_rejects_unknown_template_name(tmp_path: Path) -> None:
    stage = _stage(_context(tmp_path), planet_template="jupiter")
    with pytest.raises((KeyError, ValueError, FileNotFoundError)):
        stage.apply(_state())
