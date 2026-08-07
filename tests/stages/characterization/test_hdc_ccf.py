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


def _spectro_payload(
    *,
    awg_throughput: float = 1.0,
    free_spectral_range_fraction: float = 1.0,
    dark_rate_hz_per_bin: float = 0.0,
    with_fringe_tracking: bool | None = True,
) -> dict[str, object]:
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
    # Mirror awg_spectrum: the AWG throughput and the free-spectral-range
    # fraction multiply every optical rate before binning, and dark is charged
    # per bin with two bins per resolution element.
    detected = awg_throughput * free_spectral_range_fraction
    payload: dict[str, object] = {
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
        "awg_total_planet_rate_hz": planet_rate_hz * detected,
        "awg_total_leakage_rate_hz": leak_rate_hz * detected,
        "awg_total_sky_rate_hz": (10.0 / _INTEGRATION_TIME_S) * detected,
        "awg": {
            "n_bins": 200,
            "pixels_per_resolution_element": 2,
            "dark_rate_hz_per_bin_total": dark_rate_hz_per_bin,
            "throughput": awg_throughput,
            "free_spectral_range_fraction": free_spectral_range_fraction,
        },
    }
    if with_fringe_tracking is not None:
        payload["with_fringe_tracking"] = with_fringe_tracking
    return payload


def _state(
    *, with_spectro: bool = True, **payload_overrides: object
) -> SimulationState:
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
        "characterization.spectro",
        _spectro_payload(**payload_overrides),  # type: ignore[arg-type]
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
        # Headline CCF budget.
        "ccf_snr",
        "ccf_snr_photon_limited",
        "ccf_signal_counts",
        "ccf_photon_var_counts",
        "ccf_systematic_var_counts",
        "systematic_ccf_leakage",
        "ccf_systematic_rejection",
        "self_subtraction_factor",
        "n_effective_lines",
        "n_channels_used",
        "n_channels",
        "telluric_masked_fraction",
        # Per-channel arrays.
        "ccf_channel_centers_um",
        "ccf_planet_rate_hz_per_channel",
        "ccf_leak_rate_hz_per_channel",
        "ccf_variance_rate_hz_per_channel",
        "template_weights_per_channel",
        # Scans.
        "snr_vs_resolving_power",
        "snr_vs_resolving_power_photon_limited",
        "snr_vs_resolving_power_r",
        "snr_vs_rv",
        "snr_vs_rv_kms",
        # Time and configuration.
        "integration_time_for_target_snr_s",
        "integration_time_s",
        "target_snr",
        "resolving_power_used",
        "resolving_power_source",
        "analysis_grid_resolving_power",
        "content_resolving_power",
        "grid_oversampling",
        "continuum_filter_width_res_elements",
        "telluric_mask_threshold",
        "planet_radial_velocity_kms",
        # Detection budget and audit.
        "detected_planet_rate_hz",
        "detected_leak_rate_hz",
        "sky_rate_hz_total",
        "dark_rate_hz_per_channel",
        "detection_budget_source",
        "channel_band_coverage_fraction",
        "leak_normalization",
        "planet_normalization",
        "eps_systematic_used",
        "servo_bw_hz_used",
        "with_fringe_tracking",
        "systematic_branch_source",
        "upstream_spectro_component",
        # Provenance.
        "planet_template_provenance",
        "star_template_provenance",
        "telluric_template_provenance",
        "ccf_model",
        "snr_vs_resolving_power_model",
        "snr_vs_rv_model",
        "source_model",
    }
    missing = expected - set(payload)
    assert not missing, f"missing payload keys: {sorted(missing)}"


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


def test_hdc_ccf_variance_rate_is_the_matched_filter_denominator(
    payload: dict[str, object],
) -> None:
    """v_i = P_i + L_i + S_sky,i + D_i is what the weights divide by, and it is
    what a consumer needs to turn w_i into a per-channel SNR contribution."""

    variance = np.asarray(
        payload["ccf_variance_rate_hz_per_channel"], dtype=np.float64
    )
    planet = np.asarray(payload["ccf_planet_rate_hz_per_channel"], dtype=np.float64)
    leak = np.asarray(payload["ccf_leak_rate_hz_per_channel"], dtype=np.float64)
    sky = np.asarray(payload["ccf_sky_rate_hz_per_channel"], dtype=np.float64)
    dark = float(payload["dark_rate_hz_per_channel"])

    assert variance.size == int(payload["n_channels"])
    assert variance == pytest.approx(planet + leak + sky + dark)
    assert np.all(variance > 0.0)


def test_hdc_ccf_photon_limited_scan_bounds_the_realized_scan(
    payload: dict[str, object],
) -> None:
    """The photon limit is R dependent, so it is scanned alongside the realized
    SNR: same axis, same NaN pattern, and pointwise no smaller."""

    snr = np.asarray(payload["snr_vs_resolving_power"], dtype=np.float64)
    photon = np.asarray(
        payload["snr_vs_resolving_power_photon_limited"], dtype=np.float64
    )

    assert photon.size == snr.size
    assert np.array_equal(np.isnan(photon), np.isnan(snr))
    finite = np.isfinite(snr)
    assert np.all(photon[finite] >= snr[finite] - 1.0e-12)


def test_hdc_ccf_photon_limited_scan_is_r_dependent(
    payload: dict[str, object],
) -> None:
    """The whole point of scanning the photon limit: it varies with resolving
    power, so it cannot be drawn as the operating-point scalar held flat."""

    photon = np.asarray(
        payload["snr_vs_resolving_power_photon_limited"], dtype=np.float64
    )
    finite = photon[np.isfinite(photon)]

    assert finite.size > 1
    assert float(np.ptp(finite)) > 0.01 * float(np.max(finite))


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


def test_hdc_ccf_external_dependencies_fall_back_when_file_unreadable(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "not_there" / "planet.csv"
    stage = _stage(_context(tmp_path), planet_template=str(missing))
    dependencies = stage.external_dependencies()

    planet = next(
        dep
        for dep in dependencies
        if dep.source == "characterization.hdc_ccf.planet_template"
    )
    assert planet.content_hash is None
    assert Path(planet.path) == missing
    # Every readable template still gets its hash.
    star_candidates = [
        dep for dep in dependencies if "star_template_candidate" in dep.source
    ]
    assert len(star_candidates) == 3
    assert all(dep.content_hash is not None for dep in star_candidates)


def test_hdc_ccf_normalizes_to_awg_detected_rates(tmp_path: Path) -> None:
    """AWG throughput and FSR losses must reach the CCF budget."""

    full = _stage(_context(tmp_path), snr_vs_r_points=0, rv_scan_points=0).apply(
        _state()
    )
    lossy = _stage(_context(tmp_path), snr_vs_r_points=0, rv_scan_points=0).apply(
        _state(awg_throughput=0.5, free_spectral_range_fraction=0.5)
    )
    full_payload = full.results.components["characterization.hdc_ccf"]
    lossy_payload = lossy.results.components["characterization.hdc_ccf"]

    # 0.5 throughput x 0.5 FSR = a quarter of the photons on the detector.
    assert float(lossy_payload["detected_planet_rate_hz"]) == pytest.approx(
        0.25 * float(full_payload["detected_planet_rate_hz"])
    )
    assert float(lossy_payload["detected_leak_rate_hz"]) == pytest.approx(
        0.25 * float(full_payload["detected_leak_rate_hz"])
    )
    assert float(lossy_payload["ccf_snr"]) < float(full_payload["ccf_snr"])


def test_hdc_ccf_zero_awg_throughput_gives_no_signal(tmp_path: Path) -> None:
    result = _stage(
        _context(tmp_path), snr_vs_r_points=0, rv_scan_points=0
    ).apply(_state(awg_throughput=0.0))
    payload = result.results.components["characterization.hdc_ccf"]

    assert float(payload["detected_planet_rate_hz"]) == 0.0
    assert float(payload["ccf_snr"]) == 0.0
    assert float(payload["ccf_snr_photon_limited"]) == 0.0


def test_hdc_ccf_dark_is_fixed_per_channel_not_diluted(tmp_path: Path) -> None:
    """AWG dark is per detector bin: two bins per resolution element."""

    result = _stage(_context(tmp_path), snr_vs_r_points=0, rv_scan_points=0).apply(
        _state(dark_rate_hz_per_bin=0.25)
    )
    payload = result.results.components["characterization.hdc_ccf"]
    assert float(payload["dark_rate_hz_per_channel"]) == pytest.approx(0.5)


def test_hdc_ccf_channels_stay_inside_the_detector_band(
    payload: dict[str, object],
) -> None:
    """No partial channel hanging off the red end of the band."""

    coverage = float(payload["channel_band_coverage_fraction"])
    assert 0.99 < coverage <= 1.0
    centers = np.asarray(payload["ccf_channel_centers_um"], dtype=np.float64)
    assert centers[0] >= _WL_MIN_UM - 0.02
    assert centers[-1] <= _WL_MAX_UM + 0.02


def test_hdc_ccf_band_totals_survive_the_telluric_shape(tmp_path: Path) -> None:
    """Telluric is a shape: the published detected totals must be preserved.

    The upstream rates already carry a band-averaged atmospheric transmission
    from ``atmosphere.kolmogorov``, so the template may only add line
    structure, not a second broadband loss.
    """

    result = _stage(
        _context(tmp_path),
        telluric_template="maunakea",
        snr_vs_r_points=0,
        rv_scan_points=0,
    ).apply(_state())
    payload = result.results.components["characterization.hdc_ccf"]

    planet_total = float(
        np.sum(np.asarray(payload["ccf_planet_rate_hz_per_channel"]))
    )
    leak_total = float(np.sum(np.asarray(payload["ccf_leak_rate_hz_per_channel"])))
    assert planet_total == pytest.approx(
        float(payload["detected_planet_rate_hz"]), rel=0.02
    )
    assert leak_total == pytest.approx(
        float(payload["detected_leak_rate_hz"]), rel=0.02
    )
    assert float(payload["leak_normalization"]) == pytest.approx(1.0, rel=0.05)
    assert float(payload["planet_normalization"]) == pytest.approx(1.0, rel=0.05)


def test_hdc_ccf_r_scan_reaches_the_template_resolution(tmp_path: Path) -> None:
    result = _stage(
        _context(tmp_path), snr_vs_r_points=8, rv_scan_points=0
    ).apply(_state())
    payload = result.results.components["characterization.hdc_ccf"]

    r_axis = np.asarray(payload["snr_vs_resolving_power_r"], dtype=np.float64)
    assert r_axis[0] == pytest.approx(100.0)
    assert r_axis[-1] == pytest.approx(30_000.0)
    assert np.all(np.isfinite(payload["snr_vs_resolving_power"]))


def test_hdc_ccf_headline_snr_is_independent_of_the_diagnostic_scans(
    tmp_path: Path,
) -> None:
    """Turning the R or RV scans on must not move the headline number."""

    state = _state()
    without = _stage(
        _context(tmp_path), snr_vs_r_points=0, rv_scan_points=0
    ).apply(state)
    with_scans = _stage(
        _context(tmp_path), snr_vs_r_points=8, rv_scan_points=21
    ).apply(state)

    left = without.results.components["characterization.hdc_ccf"]
    right = with_scans.results.components["characterization.hdc_ccf"]
    for key in ("ccf_snr", "systematic_ccf_leakage", "planet_normalization"):
        assert left[key] == right[key]


def _mkid_payload(*, n_spectro_pixels: int, per_pixel_dark_hz: float) -> dict:
    payload = _spectro_payload()
    for key in (
        "awg_total_planet_rate_hz",
        "awg_total_leakage_rate_hz",
        "awg_total_sky_rate_hz",
        "awg",
        "awg_resolving_power",
        "awg_n_bins",
    ):
        payload.pop(key)
    payload["spectrograph"] = "mkid"
    payload["mkid_resolving_power"] = 3000.0
    payload["mkid_pixels_per_resolution_element"] = 1
    payload["mkid_n_spectro_pixels"] = n_spectro_pixels
    # Mirrors per_bin_snr: dark_rate_hz_per_pixel * n_spectro_pixels per bin.
    payload["mkid_dark_rate_hz_per_bin_total"] = per_pixel_dark_hz * n_spectro_pixels
    # D scales with n_spectro_ports (== 1 on the MKID branch), NOT with pixels.
    payload["D"] = per_pixel_dark_hz * _INTEGRATION_TIME_S
    return payload


def test_hdc_ccf_mkid_dark_scales_with_spectro_pixels(tmp_path: Path) -> None:
    """MKID dark follows n_spectro_pixels, which the band-total D does not."""

    per_channel = []
    for n_pixels in (1, 4):
        state = SimulationState.empty({"wavelength_grid_um": (1.25,)})
        state = replace(
            state,
            scene=Scene(
                star_jmag=4.0,
                star_spectral_type="M2V",
                planet_contrast=_PLANET_CONTRAST,
                planet_radial_velocity_kms=_PLANET_RV_KMS,
            ),
            results=SimulationState.empty({}).results.add_component(
                "characterization.spectro",
                _mkid_payload(n_spectro_pixels=n_pixels, per_pixel_dark_hz=0.5),
            ),
        )
        result = _stage(
            _context(tmp_path), snr_vs_r_points=0, rv_scan_points=0
        ).apply(state)
        payload = result.results.components["characterization.hdc_ccf"]
        per_channel.append(float(payload["dark_rate_hz_per_channel"]))
        assert "mkid_dark_rate_hz_per_bin_total" in str(
            payload["detection_budget_source"]
        )

    assert per_channel == [0.5, 2.0]


@pytest.mark.parametrize("resolving_power", [100.0, 300.0, 1000.0, 3000.0])
def test_hdc_ccf_sky_loses_the_same_uncovered_band_share(
    tmp_path: Path, resolving_power: float
) -> None:
    """Sky must forfeit the dropped channel's share like planet and leak do.

    Sky is carried as a rate density and channelized through the identical
    path, so it tracks the leak -- the other smooth continuum -- essentially
    exactly. Before the fix it was allocated by an area rule renormalized into
    the retained span and kept 100% while everything else lost its share.
    """

    result = _stage(
        _context(tmp_path),
        resolving_power=resolving_power,
        snr_vs_r_points=0,
        rv_scan_points=0,
    ).apply(_state())
    payload = result.results.components["characterization.hdc_ccf"]

    assert float(payload["channel_band_coverage_fraction"]) < 1.0

    def retained(channel_key: str, total_key: str) -> float:
        return float(
            np.sum(np.asarray(payload[channel_key], dtype=np.float64))
        ) / float(payload[total_key])

    leak_fraction = retained(
        "ccf_leak_rate_hz_per_channel", "detected_leak_rate_hz"
    )
    sky_fraction = retained("ccf_sky_rate_hz_per_channel", "sky_rate_hz_total")

    assert sky_fraction < 1.0
    assert sky_fraction == pytest.approx(leak_fraction, rel=2.0e-3)

    if resolving_power >= 3000.0:
        # At a sane kernel width every component agrees. At very coarse R the
        # planet's structured spectrum interacts differently with the LSF's
        # edge clamp than a smooth continuum does, which is an artifact of the
        # convolution's edge mode, not of the sky allocation.
        planet_fraction = retained(
            "ccf_planet_rate_hz_per_channel", "detected_planet_rate_hz"
        )
        assert sky_fraction == pytest.approx(planet_fraction, rel=1.0e-3)


def test_hdc_ccf_narrow_band_scan_records_nan_instead_of_raising(
    tmp_path: Path,
) -> None:
    """A fixed 100..30000 scan axis must not fail a viable narrow-band run."""

    wl_um = np.linspace(1.150, 1.155, _N_WAVELENGTHS)
    narrow = _spectro_payload()
    narrow["wavelengths_m"] = tuple(float(w) * 1.0e-6 for w in wl_um)
    state = SimulationState.empty({"wavelength_grid_um": tuple(float(w) for w in wl_um)})
    state = replace(
        state,
        scene=Scene(
            star_jmag=4.0,
            star_spectral_type="M2V",
            planet_contrast=_PLANET_CONTRAST,
            planet_radial_velocity_kms=_PLANET_RV_KMS,
        ),
        results=state.results.add_component("characterization.spectro", narrow),
    )

    result = _stage(_context(tmp_path), snr_vs_r_points=12).apply(state)
    payload = result.results.components["characterization.hdc_ccf"]

    snr = np.asarray(payload["snr_vs_resolving_power"], dtype=np.float64)
    r_axis = np.asarray(payload["snr_vs_resolving_power_r"], dtype=np.float64)
    # The band spans ln(1.155/1.150) = 4.3e-3, so R below ~230 fits no channel.
    infeasible = r_axis < 200.0
    assert np.any(infeasible) and np.all(np.isnan(snr[infeasible]))
    assert np.all(np.isfinite(snr[r_axis > 1000.0]))
    # The headline at R=3000 is unaffected.
    assert float(payload["ccf_snr"]) > 0.0


def test_hdc_ccf_narrow_band_headline_still_raises_below_one_channel(
    tmp_path: Path,
) -> None:
    """The headline path keeps its hard error; only the scan degrades."""

    wl_um = np.linspace(1.150, 1.155, _N_WAVELENGTHS)
    narrow = _spectro_payload()
    narrow["wavelengths_m"] = tuple(float(w) * 1.0e-6 for w in wl_um)
    state = SimulationState.empty({"wavelength_grid_um": tuple(float(w) for w in wl_um)})
    state = replace(
        state,
        scene=Scene(
            star_jmag=4.0,
            star_spectral_type="M2V",
            planet_contrast=_PLANET_CONTRAST,
            planet_radial_velocity_kms=_PLANET_RV_KMS,
        ),
        results=state.results.add_component("characterization.spectro", narrow),
    )
    stage = _stage(_context(tmp_path), resolving_power=100.0, snr_vs_r_points=0)
    with pytest.raises(ValueError, match="narrower than one resolution element"):
        stage.apply(state)


def test_hdc_ccf_reads_with_fringe_tracking_from_the_payload(
    tmp_path: Path,
) -> None:
    declared = _stage(_context(tmp_path), snr_vs_r_points=0, rv_scan_points=0).apply(
        _state(with_fringe_tracking=False)
    )
    payload = declared.results.components["characterization.hdc_ccf"]
    assert payload["with_fringe_tracking"] is False
    assert "payload key" in str(payload["systematic_branch_source"])

    inferred = _stage(_context(tmp_path), snr_vs_r_points=0, rv_scan_points=0).apply(
        _state(with_fringe_tracking=None)
    )
    inferred_payload = inferred.results.components["characterization.hdc_ccf"]
    assert inferred_payload["with_fringe_tracking"] is True
    assert "fallback inference" in str(inferred_payload["systematic_branch_source"])


def test_hdc_ccf_rv_statistic_matches_photon_limited_snr_at_true_rv(
    payload: dict[str, object],
) -> None:
    """The RV statistic is a matched-filter score, not a bounded cosine."""

    rv_kms = np.asarray(payload["snr_vs_rv_kms"], dtype=np.float64)
    ccf = np.asarray(payload["snr_vs_rv"], dtype=np.float64)
    at_true_rv = float(ccf[int(np.argmin(np.abs(rv_kms - _PLANET_RV_KMS)))])
    assert at_true_rv == pytest.approx(
        float(payload["ccf_snr_photon_limited"]), rel=1e-6
    )
    assert "matched-filter score" in str(payload["snr_vs_rv_model"])
