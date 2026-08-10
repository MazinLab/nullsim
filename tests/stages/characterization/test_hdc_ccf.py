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
# Band edges as the stage derives them: midpoint edges around the coarse grid.
_BIN_WIDTH_UM = (_WL_MAX_UM - _WL_MIN_UM) / (_N_WAVELENGTHS - 1)
_BAND_LO_UM = _WL_MIN_UM - 0.5 * _BIN_WIDTH_UM
_BAND_HI_UM = _WL_MAX_UM + 0.5 * _BIN_WIDTH_UM
_BAND_CENTRE_UM = 0.5 * (_BAND_LO_UM + _BAND_HI_UM)
_BAND_WIDTH_UM = _BAND_HI_UM - _BAND_LO_UM


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
    planet_rate_scale: float = 1.0,
    with_fringe_tracking: bool | None = True,
    wl_min_um: float = _WL_MIN_UM,
    wl_max_um: float = _WL_MAX_UM,
    stellar_rate_hz_per_bin: float = _STELLAR_RATE_HZ_PER_BIN,
    optimization_success: bool | None = None,
    enforce_null_constraint: bool | None = None,
    optimization_constraint_margin: float | None = None,
) -> dict[str, object]:
    """A synthetic ``characterization.spectro`` payload on a flat 8-bin band.

    ``wl_min_um`` / ``wl_max_um`` / ``stellar_rate_hz_per_bin`` exist so a test
    can build the SAME spectral density over a narrower detector band, which is
    what makes a partial free spectral range comparable to an explicitly
    narrow band.
    """

    wl_um = np.linspace(wl_min_um, wl_max_um, _N_WAVELENGTHS)
    bin_width_um = (wl_max_um - wl_min_um) / (_N_WAVELENGTHS - 1)
    band_lo_um = wl_min_um - 0.5 * bin_width_um
    band_hi_um = wl_max_um + 0.5 * bin_width_um
    band_centre_um = 0.5 * (band_lo_um + band_hi_um)
    band_width_um = band_hi_um - band_lo_um
    stellar = np.full(_N_WAVELENGTHS, stellar_rate_hz_per_bin)
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
        "awg_total_planet_rate_hz": planet_rate_hz * detected * planet_rate_scale,
        "awg_total_leakage_rate_hz": leak_rate_hz * detected,
        "awg_total_sky_rate_hz": (10.0 / _INTEGRATION_TIME_S) * detected,
        "awg": {
            "n_bins": 200,
            "pixels_per_resolution_element": 2,
            "dark_rate_hz_per_bin_total": dark_rate_hz_per_bin,
            "throughput": awg_throughput,
            "free_spectral_range_fraction": free_spectral_range_fraction,
            # awg_wavelength_grid centres a span of
            # band_bandwidth x free_spectral_range_fraction on the design
            # wavelength; mirror that so the stage can restrict channels to it.
            "bin_edges_m": np.linspace(
                (band_centre_um - 0.5 * band_width_um * free_spectral_range_fraction)
                * 1.0e-6,
                (band_centre_um + 0.5 * band_width_um * free_spectral_range_fraction)
                * 1.0e-6,
                201,
            ),
        },
    }
    if with_fringe_tracking is not None:
        payload["with_fringe_tracking"] = with_fringe_tracking
    if optimization_success is not None:
        payload["optimization_success"] = optimization_success
    if enforce_null_constraint is not None:
        payload["enforce_null_constraint"] = enforce_null_constraint
    if optimization_constraint_margin is not None:
        payload["optimization_constraint_margin"] = optimization_constraint_margin
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


def _mkid_payload() -> dict:
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
    return payload


def _flat_flambda_stellar_csv(tmp_path: Path) -> Path:
    """A PHOENIX-like snapshot whose F_lambda is exactly flat.

    The packaged files store a PHOTON-flux shape ``F_lambda * lambda``, so a
    flat-F_lambda star is stored as values proportional to lambda.
    """

    from nullsim.physics.spectra import constant_r_grid

    grid = constant_r_grid(1.0, 1.5, 30_000.0)
    path = tmp_path / "flat_flambda.csv"
    lines = [
        "# kind: stellar_flux",
        "# resolving_power: 30000",
        "# source: synthetic flat-F_lambda test template",
        "wavelength_um,value",
    ]
    lines += [f"{w:.10e},{w:.10e}" for w in grid]
    path.write_text("\n".join(lines))
    return path


def test_hdc_ccf_stellar_template_does_not_retilt_the_coarse_density(
    tmp_path: Path,
) -> None:
    """No double lambda tilt: photon-flux shape must be converted to F_lambda.

    The coarse detected rates already carry the photon-rate lambda tilt (the
    zero-point model is flat in F_lambda), so multiplying by a stored
    photon-flux shape would give lambda^2 * F_lambda. With a flat-F_lambda
    star and a flat coarse rate density, the channelized leak density must come
    out flat.
    """

    result = _stage(
        _context(tmp_path),
        star_template=str(_flat_flambda_stellar_csv(tmp_path)),
        snr_vs_r_points=0,
        rv_scan_points=0,
    ).apply(_state())
    payload = result.results.components["characterization.hdc_ccf"]

    centers = np.asarray(payload["ccf_channel_centers_um"], dtype=np.float64)
    leak = np.asarray(payload["ccf_leak_rate_hz_per_channel"], dtype=np.float64)
    # Channel widths grow like lambda on a log grid; divide them out to recover
    # the underlying rate density, which must be flat.
    edges = np.asarray(payload["ccf_channel_edges_um"], dtype=np.float64)
    density = leak / np.diff(edges)
    interior = slice(2, -2)  # skip LSF edge roll-off
    density = density[interior] / float(np.mean(density[interior]))
    slope = np.polyfit(centers[interior], density, 1)[0]
    tilt_across_band = abs(slope) * (centers[-1] - centers[0])
    assert tilt_across_band < 0.01, (
        f"stellar shape retilted the coarse density by {tilt_across_band:.3%} "
        "across the band; expected <1%"
    )


def test_hdc_ccf_systematic_rejection_is_scale_invariant(tmp_path: Path) -> None:
    """The rejection factor must not move with the template normalization.

    Exact invariance is a property of the metric itself, so it is pinned on the
    helper with the weights scaled directly. End to end, scaling the planet
    rate by 10 also feeds back weakly through the per-channel variance (the
    planet is part of v), so that leg only checks the factor stays put to a
    few percent and never goes negative.
    """

    from nullsim.physics.hdc import ChannelSpectra
    from nullsim.stages.characterization.hdc_ccf import _ccf_systematic_rejection

    rng = np.random.default_rng(0)
    n = 32
    edges = np.linspace(1.2, 1.3, n + 1)
    channels = ChannelSpectra(
        channel_centers_um=0.5 * (edges[:-1] + edges[1:]),
        channel_edges_um=edges,
        planet_rate_hz=rng.uniform(0.1, 1.0, n),
        leak_rate_hz=rng.uniform(10.0, 20.0, n),
        sky_rate_hz=np.zeros(n),
        dark_rate_hz=np.zeros(n),
    )
    weights = rng.uniform(-1.0, 1.0, n)

    class _Budget:
        def __init__(self, scale: float) -> None:
            self.template_weights = weights * scale
            self.systematic_ccf_leakage = float(
                np.sum(self.template_weights * channels.leak_rate_hz * 0.3)
            )

    unscaled = _ccf_systematic_rejection(_Budget(1.0), channels)
    scaled = _ccf_systematic_rejection(_Budget(10.0), channels)
    assert unscaled >= 0.0
    assert scaled == pytest.approx(unscaled, rel=1.0e-12)

    baseline = _stage(
        _context(tmp_path), snr_vs_r_points=0, rv_scan_points=0
    ).apply(_state())
    brighter = _stage(
        _context(tmp_path), snr_vs_r_points=0, rv_scan_points=0
    ).apply(_state(planet_rate_scale=10.0))
    left = float(
        baseline.results.components["characterization.hdc_ccf"][
            "ccf_systematic_rejection"
        ]
    )
    right = float(
        brighter.results.components["characterization.hdc_ccf"][
            "ccf_systematic_rejection"
        ]
    )
    # Non-zero: a degenerate denominator would silently collapse this to 0.
    assert left > 0.0 and right > 0.0
    assert right == pytest.approx(left, rel=0.05)

def test_hdc_ccf_partial_fsr_restricts_the_channel_span(tmp_path: Path) -> None:
    """With FSR < 1 the detector never sees the blocked wavelengths."""

    full = _stage(_context(tmp_path), snr_vs_r_points=0, rv_scan_points=0).apply(
        _state()
    )
    half = _stage(_context(tmp_path), snr_vs_r_points=0, rv_scan_points=0).apply(
        _state(free_spectral_range_fraction=0.5)
    )
    full_payload = full.results.components["characterization.hdc_ccf"]
    half_payload = half.results.components["characterization.hdc_ccf"]

    full_lo, full_hi = full_payload["observed_band_um"]
    half_lo, half_hi = half_payload["observed_band_um"]
    assert (half_hi - half_lo) == pytest.approx(
        0.5 * _BAND_WIDTH_UM, rel=1.0e-6
    )
    assert half_lo > full_lo and half_hi < full_hi
    # Channels only tile the dispersed span, so there are about half as many
    # and the dark budget scales with them instead of the full band.
    assert int(half_payload["n_channels"]) == pytest.approx(
        0.5 * int(full_payload["n_channels"]), rel=0.02
    )
    centers = np.asarray(
        half_payload["ccf_channel_centers_um"], dtype=np.float64
    )
    assert centers.min() >= half_lo and centers.max() <= half_hi


def test_hdc_ccf_binds_each_instance_to_its_own_spectro(tmp_path: Path) -> None:
    """spectro_a -> hdc_a -> spectro_b -> hdc_b must not both bind spectro_a."""

    state = _state()
    first = _stage(
        _context(tmp_path), snr_vs_r_points=0, rv_scan_points=0
    ).apply(state)
    # A second spectro component lands after the first HDC payload.
    with_second = replace(
        first,
        results=first.results.add_component(
            "characterization.spectro.b", _spectro_payload(awg_throughput=0.5)
        ),
    )
    second = _stage(
        _context(tmp_path), snr_vs_r_points=0, rv_scan_points=0
    ).apply(with_second)

    payloads = [
        payload
        for name, payload in second.results.components.items()
        if name.startswith("characterization.hdc_ccf")
    ]
    assert len(payloads) == 2
    assert payloads[0]["upstream_spectro_component"] == "characterization.spectro"
    assert payloads[1]["upstream_spectro_component"] == "characterization.spectro.b"


def test_hdc_ccf_upstream_spectro_param_selects_explicitly(tmp_path: Path) -> None:
    state = _state()
    state = replace(
        state,
        results=state.results.add_component(
            "characterization.spectro.b", _spectro_payload(awg_throughput=0.5)
        ),
    )
    result = _stage(
        _context(tmp_path),
        upstream_spectro="characterization.spectro",
        snr_vs_r_points=0,
        rv_scan_points=0,
    ).apply(state)
    payload = result.results.components["characterization.hdc_ccf"]
    assert payload["upstream_spectro_component"] == "characterization.spectro"

    missing = _stage(_context(tmp_path), upstream_spectro="characterization.nope")
    with pytest.raises(ValueError, match="is not present on state.results"):
        missing.apply(state)


def test_hdc_ccf_rejects_mkid_native_spectroscopy(tmp_path: Path) -> None:
    """Phase 1 refuses MKID rather than applying a constant-R LSF to it."""

    state = SimulationState.empty({"wavelength_grid_um": (1.25,)})
    state = replace(
        state,
        scene=Scene(
            star_jmag=4.0,
            star_spectral_type="M2V",
            planet_contrast=_PLANET_CONTRAST,
            planet_radial_velocity_kms=_PLANET_RV_KMS,
        ),
        results=state.results.add_component(
            "characterization.spectro", _mkid_payload()
        ),
    )
    stage = _stage(_context(tmp_path))
    with pytest.raises(ValueError, match="does not support MKID"):
        stage.apply(state)


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

    # The bug being pinned: sky used to keep 100%. It now forfeits the same
    # share the other components do, to within the LSF's edge-clamp asymmetry
    # (the leak carries the stellar tilt, sky is flat, and at coarse R the
    # kernel is a sizeable fraction of the band).
    assert sky_fraction < 1.0
    assert sky_fraction == pytest.approx(leak_fraction, rel=5.0e-3)

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


def test_hdc_ccf_rv_statistic_peaks_at_the_true_radial_velocity(
    payload: dict[str, object],
) -> None:
    """The RV scan localizes the planet, and dominates the fixed-template SNR.

    Re-pinned when the RV statistic became the whitened (C_y-pseudoinverse)
    matched filter. It used to equal ``ccf_snr_photon_limited`` at the true RV,
    because it shared that statistic's fixed-template weights; it no longer
    does, and must not. Whitening recovers the signal the continuum filter
    attenuated, so the localization score sits at or above the detection SNR
    for a smooth leak like this one.

    Not strictly above, and not a theorem. The optimum dominating any fixed
    filter in its subspace would give ``>=`` only if both correlated against
    the same vector, and they do not: the scan correlates against the observed
    planet-plus-leak spectrum while the detection signal is the planet alone.
    A leak that cancels the planet's structure (``P = [1, 2, 1]`` under
    ``L = [9, 8, 9]``, total flat) leaves the observation featureless, and the
    scan correctly reports 0.0 where detection reports 0.258. Equality is also
    reachable, so this asserts ``>=`` with a tolerance.
    """

    rv_kms = np.asarray(payload["snr_vs_rv_kms"], dtype=np.float64)
    ccf = np.asarray(payload["snr_vs_rv"], dtype=np.float64)
    at_true_rv = float(ccf[int(np.argmin(np.abs(rv_kms - _PLANET_RV_KMS)))])

    assert rv_kms[int(np.argmax(ccf))] == pytest.approx(_PLANET_RV_KMS, abs=1.0)
    assert at_true_rv == pytest.approx(float(np.max(ccf)), rel=1e-9)
    photon_limited = float(payload["ccf_snr_photon_limited"])
    assert at_true_rv >= photon_limited * (1.0 - 1e-9)
    assert "whitened cross-correlation score" in str(payload["snr_vs_rv_model"])
    assert "LOCALIZATION" in str(payload["snr_vs_rv_model"])


def test_partial_fsr_matches_an_explicit_narrow_band(tmp_path: Path) -> None:
    """A partial free spectral range must charge the band fraction ONCE.

    Two descriptions of the same instrument. Either the AWG disperses a
    fraction ``f`` of a wide detector band, or the detector band simply IS that
    dispersed span. Same photons per wavelength, same span, so the two must
    give the same per-channel rates.

    The construction makes them the same density by hand: with ``N`` grid
    points the explicit band's bin width comes out ``f`` times the wide band's,
    so scaling the per-bin stellar rate by ``f`` leaves the rate DENSITY
    identical. Anything that applied ``f`` a second time -- once as the flat
    band-fraction inside the AWG totals and again by restricting the analysis
    band -- would show up as the partial-FSR channels sitting a factor ``f``
    below the explicit-band ones over the same wavelengths.
    """

    fraction = 0.6
    half_span_um = 0.5 * fraction * _BAND_WIDTH_UM
    lo_um = _BAND_CENTRE_UM - half_span_um
    hi_um = _BAND_CENTRE_UM + half_span_um
    # Grid endpoints inset by half a bin so the explicit band's EDGES land on
    # the dispersed span rather than its grid points.
    inset_um = (hi_um - lo_um) / (2 * _N_WAVELENGTHS)

    partial = _stage(_context(tmp_path), snr_vs_r_points=0, rv_scan_points=0).apply(
        _state(free_spectral_range_fraction=fraction)
    ).results.components["characterization.hdc_ccf"]
    explicit = _stage(_context(tmp_path), snr_vs_r_points=0, rv_scan_points=0).apply(
        _state(
            wl_min_um=lo_um + inset_um,
            wl_max_um=hi_um - inset_um,
            stellar_rate_hz_per_bin=fraction * _STELLAR_RATE_HZ_PER_BIN,
        )
    ).results.components["characterization.hdc_ccf"]

    assert explicit["observed_band_um"] == pytest.approx(
        partial["observed_band_um"], rel=1.0e-9
    )
    assert int(explicit["n_channels"]) == int(partial["n_channels"])

    partial_planet = np.asarray(
        partial["ccf_planet_rate_hz_per_channel"], dtype=np.float64
    )
    explicit_planet = np.asarray(
        explicit["ccf_planet_rate_hz_per_channel"], dtype=np.float64
    )
    assert partial_planet.sum() > 0.0
    assert partial_planet == pytest.approx(explicit_planet, rel=1.0e-6)
    partial_leak = np.asarray(
        partial["ccf_leak_rate_hz_per_channel"], dtype=np.float64
    )
    explicit_leak = np.asarray(
        explicit["ccf_leak_rate_hz_per_channel"], dtype=np.float64
    )
    assert partial_leak == pytest.approx(explicit_leak, rel=1.0e-6)


def test_rejected_spectro_solution_publishes_nan_and_a_reason(
    tmp_path: Path,
) -> None:
    """A retune the pipeline rejected must not yield a plausible CCF number."""

    from nullsim.stages.characterization.acceptance import REASON_OPTIMIZER_FAILED

    payload = _stage(_context(tmp_path)).apply(
        _state(optimization_success=False)
    ).results.components["characterization.hdc_ccf"]

    assert payload["spectro_solution_accepted"] is False
    assert payload["invalidity_reason"] == REASON_OPTIMIZER_FAILED
    assert np.isnan(float(payload["ccf_snr"]))
    assert np.isnan(float(payload["ccf_snr_photon_limited"]))
    assert np.isnan(float(payload["n_effective_lines"]))
    assert int(payload["n_channels"]) == 0
    assert (
        np.asarray(payload["ccf_planet_rate_hz_per_channel"], dtype=np.float64).size
        == 0
    )
    assert payload["upstream_spectro_component"] == "characterization.spectro"


def test_violated_null_constraint_is_rejected_with_its_own_code(
    tmp_path: Path,
) -> None:
    """The two rejection routes are distinguishable without parsing prose."""

    from nullsim.stages.characterization.acceptance import (
        REASON_NULL_CONSTRAINT_VIOLATED,
    )

    payload = _stage(_context(tmp_path)).apply(
        _state(
            enforce_null_constraint=True,
            optimization_constraint_margin=-1.0,
        )
    ).results.components["characterization.hdc_ccf"]

    assert payload["spectro_solution_accepted"] is False
    assert payload["invalidity_reason"] == REASON_NULL_CONSTRAINT_VIOLATED
    assert np.isnan(float(payload["ccf_snr"]))


def test_accepted_spectro_solution_is_marked_accepted(
    payload: dict[str, object],
) -> None:
    """The acceptance verdict is stamped on every payload, not only failures."""

    assert payload["spectro_solution_accepted"] is True
    assert payload["invalidity_reason"] is None
    assert float(payload["ccf_snr"]) > 0.0


ACCEPTED_ONLY_SCALAR_KEYS: tuple[str, ...] = (
    "analysis_grid_resolving_power",
    "ccf_model",
    "content_resolving_power",
    "continuum_filter_width_res_elements",
    "detection_budget_source",
    "eps_systematic_used",
    "grid_oversampling",
    "integration_time_s",
    "observed_band_um",
    "planet_radial_velocity_kms",
    "planet_template_provenance",
    "planet_template_sha256",
    "resolving_power_source",
    "rv_scan_localization_model",
    "servo_bw_hz_used",
    "snr_vs_resolving_power_model",
    "snr_vs_rv_model",
    "star_template_provenance",
    "star_template_sha256",
    "systematic_branch_source",
    "target_snr",
    "telluric_mask_threshold",
    "telluric_template_provenance",
    "telluric_template_sha256",
    "with_fringe_tracking",
)
"""Keys an accepted payload has and a rejected one deliberately does not.

Every one describes a quantity the rejected path never computed (a resolving
power, a template it never read) or prose about a model it never ran. Pinned
as a list because for a payload the schema IS the behavior: a consumer reads
keys, so a key appearing or disappearing is a contract change and should have
to be written down here.
"""


def test_payload_schema_does_not_depend_on_acceptance(
    payload: dict[str, object], tmp_path: Path
) -> None:
    """A rejected retune publishes the same arrays, emptied -- not fewer keys.

    A consumer decides what to read from the payload schema, and the schema
    cannot depend on whether the optimizer happened to converge: published
    per-channel arrays existed only on the accepted path, so a plot iterating
    them raised on rejection instead of drawing a hole. The contract asserted
    here in both directions is: every accepted array key exists on the rejected
    path as an empty array of the same dtype kind; the rejected path adds
    nothing of its own; and the scalar keys it omits are exactly the pinned
    list above.
    """

    rejected = _stage(_context(tmp_path)).apply(
        _state(optimization_success=False)
    ).results.components["characterization.hdc_ccf"]

    accepted_arrays = {
        key for key, value in payload.items() if isinstance(value, np.ndarray)
    }
    assert sorted(accepted_arrays - set(rejected)) == []
    for key in sorted(accepted_arrays):
        empty = rejected[key]
        assert isinstance(empty, np.ndarray), key
        assert empty.size == 0, key
        assert empty.dtype.kind == np.asarray(payload[key]).dtype.kind, key

    # Reverse direction: rejection may not invent keys an accepted run lacks.
    assert sorted(set(rejected) - set(payload)) == []

    # And the scalars it does omit are exactly the documented set.
    assert sorted(set(payload) - set(rejected)) == sorted(
        ACCEPTED_ONLY_SCALAR_KEYS
    )

    assert np.isnan(float(rejected["rv_leak_contamination"]))
    assert np.isnan(float(rejected["rv_scan_worst_case_margin_sigma"]))
    assert rejected["rv_scan_localization_valid_modeled_leak"] is False
    assert rejected["rv_scan_localization_grade_modeled_leak"] == "not evaluated"


def test_rv_localization_verdict_is_a_grid_wide_margin_comparison(
    payload: dict[str, object],
) -> None:
    """The verdict compares two sigma quantities over the whole scan.

    A projection at the true velocity alone cannot validate an argmax: a leak
    orthogonal to the filter at the truth can align with the filter a few
    trials away. So the published verdict has to be a function of every trial's
    shift, and it has to be reproducible from the published fields.
    """

    shifts = np.asarray(payload["rv_leak_score_shift_sigma"], dtype=np.float64)
    scores = np.asarray(payload["snr_vs_rv"], dtype=np.float64)
    grid = np.asarray(payload["snr_vs_rv_kms"], dtype=np.float64)
    assert shifts.shape == scores.shape == grid.shape
    assert shifts.size > 1

    truth = int(np.argmin(np.abs(grid - float(payload["planet_radial_velocity_kms"]))))
    off_truth = np.ones(scores.size, dtype=bool)
    off_truth[truth] = False
    margin = float(scores[truth] - scores[off_truth].max())
    shift_truth = float(abs(shifts[truth]))
    shift_off = float(np.abs(shifts[off_truth]).max())

    assert float(payload["rv_scan_margin_sigma"]) == pytest.approx(margin)
    assert float(payload["rv_scan_leak_shift_at_truth_sigma"]) == pytest.approx(
        shift_truth
    )
    assert float(
        payload["rv_scan_max_off_truth_leak_shift_sigma"]
    ) == pytest.approx(shift_off)
    assert float(payload["rv_scan_worst_case_margin_sigma"]) == pytest.approx(
        margin - shift_truth - shift_off
    )

    grade = payload["rv_scan_localization_grade_modeled_leak"]
    assert grade in ("reliable", "marginal", "unreliable")
    assert payload["rv_scan_localization_valid_modeled_leak"] is (
        grade == "reliable"
    )
    assert (grade == "reliable") is (
        margin > 0.0 and margin - shift_truth - shift_off > 0.0
    )

    # The shifts are the modeled residual, so they scale with eps_systematic:
    # a config with no null instability has nothing to be contaminated by.
    if float(payload["eps_systematic_used"]) == 0.0:
        assert np.all(shifts == 0.0)

    # The verdict's scope is stated where it cannot be missed.
    model = str(payload["rv_scan_localization_model"])
    assert "eps_systematic * L" in model
    assert "static subtraction error" in model
    # The truth-only ratio is published but explicitly not the verdict.
    assert float(payload["rv_leak_contamination"]) >= 0.0
    assert "deliberately not used as the verdict" in model


def test_island_diagnostics_expose_what_the_global_factor_averages_over(
    payload: dict[str, object],
) -> None:
    """Per-island retention: the global factor is a different quantity."""

    counts = np.asarray(payload["ccf_island_channel_counts"], dtype=np.int64)
    factors = np.asarray(
        payload["ccf_island_self_subtraction_factors"], dtype=np.float64
    )
    assert counts.size == factors.size
    assert counts.size >= 1
    assert int(counts.sum()) == int(payload["n_channels_used"])
    assert np.all(counts >= 1)
    assert np.all(factors >= 0.0)
    # A one-channel island cannot separate line from continuum, ever.
    assert np.all(factors[counts == 1] == 0.0)
