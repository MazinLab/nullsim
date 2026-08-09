"""Tests for the MKIDDetector stage (SPEC §15.3 Pass 4a).

Reference invariants come from KILO ``detect_mkid``
(``kilo/detectors.py`` L256-287). See
``tests/stages/notes/detector_mkid.md`` for the source survey.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from nullsim.physics.photometry import jmag_to_band_mag, photon_rate_per_m2_per_bin
from nullsim.pipeline import (
    Field,
    Geometry,
    RunContext,
    Scene,
    SimulationState,
    Throughput,
)
from nullsim.stages.detector import MKIDDetector


def _context(tmp_path: Path) -> RunContext:
    return RunContext(
        rng=np.random.default_rng(0),
        config_hash="detector-mkid-pinning",
        sweep_coord_tuple=(),
        instance_id="detector",
        run_dir=tmp_path,
    )


def _state(
    *,
    n_modes: int = 4,
    n_wavelength: int = 8,
    wl_center_um: float = 1.25,
    wl_half_width_um: float = 0.05,
    bright_amplitude_sq: float = 0.999,
    dark_amplitude_sq: float = 1.0e-7,
    transmission: float = 1.0,
    star_jmag: float = 5.0,
    star_spectral_type: str | None = None,
    area_m2: float = 50.0,
) -> SimulationState:
    wl_um = tuple(
        np.linspace(
            wl_center_um - wl_half_width_um,
            wl_center_um + wl_half_width_um,
            n_wavelength,
        )
    )
    amps = np.zeros((n_modes, n_wavelength, 1), dtype=np.complex128)
    # Convention: port 0 is bright, ports 1..n_modes-1 are dark.
    amps[0, :, 0] = np.sqrt(bright_amplitude_sq)
    amps[1:, :, 0] = np.sqrt(dark_amplitude_sq)
    field = Field(
        amplitudes=amps,
        wavelength_grid_um=wl_um,
        mode_labels=tuple(f"port_{p}" for p in range(n_modes)),
    )
    throughput = Throughput(
        wavelength_grid_um=wl_um,
        transmission_fraction=tuple(transmission for _ in wl_um),
    )
    scene = Scene(star_jmag=star_jmag, star_spectral_type=star_spectral_type)
    geometry = Geometry(area_m2=area_m2)
    state = SimulationState.empty({"wavelength_grid_um": wl_um})
    return replace(
        state,
        field=field,
        throughput=throughput,
        scene=scene,
        geometry=geometry,
    )


def _stage(
    context: RunContext,
    *,
    n_modes: int = 4,
    n_bright_ports: int = 1,
    n_dark_ports: int = 3,
    n_spectro_ports: int = 0,
    qe: float = 1.0,
    mkid_max_count_rate_hz: float = 1.0e15,
    n_pixels_bright: int = 1,
    n_pixels_dark: int = 1,
    integration_time_s: float = 1.0,
    band: str = "J",
    derive_band_mag_from_scene_jmag: bool = False,
    resolving_power_ref: float | None = None,
    lambda_ref_um: float | None = None,
) -> MKIDDetector:
    params = {
        "qe": qe,
        "mkid_max_count_rate_hz": mkid_max_count_rate_hz,
        "n_pixels_bright": n_pixels_bright,
        "n_pixels_dark": n_pixels_dark,
        "n_bright_ports": n_bright_ports,
        "n_dark_ports": n_dark_ports,
        "n_spectro_ports": n_spectro_ports,
        "n_modes": n_modes,
        "integration_time_s": integration_time_s,
        "band": band,
        "derive_band_mag_from_scene_jmag": derive_band_mag_from_scene_jmag,
    }
    if resolving_power_ref is not None:
        params["resolving_power_ref"] = resolving_power_ref
    if lambda_ref_um is not None:
        params["lambda_ref_um"] = lambda_ref_um
    return MKIDDetector(params, context)


def test_mkid_unit_qe_matches_analytical_field_squared_rate(tmp_path: Path) -> None:
    # With QE=1, no saturation, and constant unity transmission, the per-port,
    # per-wavelength rate must equal
    #     |field|² × stellar_rate_per_m2_per_bin × (collecting_area / n_modes).
    # The per-mode area = collecting_area / n_modes follows from the chip's
    # input convention (one sub-aperture per mode); for the legacy
    # single-aperture test setup (n_apertures=None) this collapses to
    # ``area_m2 / n_modes``.
    transmission = 1.0
    star_jmag = 5.0
    area_m2 = 50.0
    state = _state(
        transmission=transmission,
        star_jmag=star_jmag,
        area_m2=area_m2,
    )
    stage = _stage(
        _context(tmp_path),
        qe=1.0,
        mkid_max_count_rate_hz=1.0e15,
        n_pixels_bright=1,
        n_pixels_dark=1,
    )

    result = stage.apply(state)
    rates = result.photon_rates.rates_hz

    wl_um = np.asarray(state.field.wavelength_grid_um, dtype=np.float64)
    bandwidth_um = (wl_um[-1] - wl_um[0]) / (wl_um.size - 1)
    expected_stellar = np.array(
        [
            photon_rate_per_m2_per_bin(star_jmag, float(wl), bandwidth_um, band="J")
            for wl in wl_um
        ],
        dtype=np.float64,
    )
    power_per_mode = np.sum(
        np.abs(state.field.amplitudes) ** 2,
        axis=2,
    ).astype(np.float64)
    n_modes = power_per_mode.shape[0]
    area_per_mode_m2 = area_m2 / float(n_modes)
    expected = power_per_mode * (expected_stellar * area_per_mode_m2)[np.newaxis, :]

    np.testing.assert_allclose(rates, expected, rtol=1.0e-12)


def test_mkid_does_not_apply_input_area_vector_to_output_ports(
    tmp_path: Path,
) -> None:
    """Input-mode areas must be folded into fields before unitary mixing.

    A post-chip detector cannot treat a per-input area vector as if it were
    attached to output-port rows. Partially split meshes encode unequal input
    branch areas in the field amplitudes before the chip and then use the usual
    scalar area-per-output convention here.
    """

    area_m2 = 6.0
    state = _state(
        n_modes=2,
        n_wavelength=3,
        bright_amplitude_sq=1.0,
        dark_amplitude_sq=1.0,
        transmission=1.0,
        star_jmag=5.0,
        area_m2=area_m2,
    )
    state = replace(
        state,
        geometry=replace(
            state.geometry,
            components={
                "splitter": {
                    "n_output_modes": 2,
                    "mode_areas_m2": np.array([1.0, 5.0], dtype=np.float64),
                    "field_area_reference_m2": 3.0,
                },
            },
        ),
    )
    stage = _stage(
        _context(tmp_path),
        n_modes=2,
        n_bright_ports=1,
        n_dark_ports=1,
        qe=1.0,
        mkid_max_count_rate_hz=1.0e15,
    )

    result = stage.apply(state)

    wl_um = np.asarray(state.field.wavelength_grid_um, dtype=np.float64)
    bandwidth_um = (wl_um[-1] - wl_um[0]) / (wl_um.size - 1)
    expected_stellar = np.array(
        [
            photon_rate_per_m2_per_bin(5.0, float(wl), bandwidth_um, band="J")
            for wl in wl_um
        ],
        dtype=np.float64,
    )
    expected = np.ones((2, wl_um.size), dtype=np.float64) * (
        expected_stellar * (area_m2 / 2.0)
    )[np.newaxis, :]
    np.testing.assert_allclose(result.photon_rates.rates_hz, expected, rtol=1.0e-12)


def test_mkid_can_derive_optical_band_magnitude_from_scene_jmag(
    tmp_path: Path,
) -> None:
    """HWO uses J catalog magnitudes but an 0.8 um detector band."""

    star_jmag = 8.0
    state = _state(
        n_wavelength=1,
        wl_center_um=0.806,
        wl_half_width_um=0.0,
        star_jmag=star_jmag,
        star_spectral_type="M3V",
    )
    stage = _stage(
        _context(tmp_path),
        qe=1.0,
        mkid_max_count_rate_hz=1.0e15,
        band="I",
        derive_band_mag_from_scene_jmag=True,
    )

    result = stage.apply(state)
    component = result.results.components["detector.mkid"]
    expected_imag = jmag_to_band_mag(star_jmag, "I", "M3V")

    assert component["photometric_band"] == "I"
    assert component["derive_band_mag_from_scene_jmag"] is True
    assert component["star_magnitude"] == pytest.approx(expected_imag)
    assert component["star_magnitude"] > star_jmag


def test_mkid_multi_aperture_uses_collecting_area(tmp_path: Path) -> None:
    """Regression test for Tier 1.1 detector over-normalization fix.

    With ``n_apertures`` and ``collecting_area_m2`` set, the detector must
    use the total collecting area summed across apertures (not per-aperture
    area). For a unitary chip + unit-amplitude injection, the total port
    count rate equals
    ``collecting_area_m2 × Σ_wl stellar_rate_per_m2_per_bin × qe``.

    Pre-fix nullsim multiplied by ``n_modes × area_per_aperture`` — an
    ``n_modes / n_apertures`` over-count (~12× for Keck-only with M=24,
    N=2). KILO uses the same total-collecting-area convention
    (``effective_area = sum(coupling × sub_areas)`` in
    ``kilo/studies/natural_design.py``).
    """
    n_modes = 4
    star_jmag = 5.0
    area_per_aperture_m2 = 50.0
    n_apertures = 2
    collecting_area_m2 = float(n_apertures) * area_per_aperture_m2

    state = _state(
        transmission=1.0,
        star_jmag=star_jmag,
        area_m2=area_per_aperture_m2,
        n_modes=n_modes,
    )
    # Unit amplitude on every chip mode (one sub-aperture per mode), and
    # the collecting-area field that the post-fix detector consumes.
    amps = np.ones_like(state.field.amplitudes)
    state = replace(
        state,
        field=replace(state.field, amplitudes=amps),
        geometry=replace(
            state.geometry,
            n_apertures=n_apertures,
            collecting_area_m2=collecting_area_m2,
        ),
    )

    stage = _stage(
        _context(tmp_path),
        qe=1.0,
        mkid_max_count_rate_hz=1.0e18,
        n_modes=n_modes,
        n_pixels_bright=1,
        n_pixels_dark=1,
    )
    result = stage.apply(state)
    total_rate_hz = float(result.photon_rates.rates_hz.sum())

    wl_um = np.asarray(state.field.wavelength_grid_um, dtype=np.float64)
    bandwidth_um = (wl_um[-1] - wl_um[0]) / (wl_um.size - 1)
    stellar_rate_per_m2 = np.array(
        [
            photon_rate_per_m2_per_bin(star_jmag, float(wl), bandwidth_um, band="J")
            for wl in wl_um
        ],
        dtype=np.float64,
    )
    # Per-mode area × n_modes = collecting_area; with M unit-amp inputs and a
    # unitary chip, total port rate = collecting_area × Σ_wl stellar_rate.
    expected_total_rate_hz = float(collecting_area_m2 * stellar_rate_per_m2.sum())

    np.testing.assert_allclose(total_rate_hz, expected_total_rate_hz, rtol=1.0e-12)


def test_mkid_collecting_area_falls_back_to_n_apertures_times_area(tmp_path: Path) -> None:
    """When ``collecting_area_m2`` isn't set, the detector must derive it from
    ``n_apertures × area_m2`` (matches the telescope.fixed_aperture default).
    """
    n_modes = 4
    star_jmag = 5.0
    area_per_aperture_m2 = 50.0
    n_apertures = 2

    state = _state(
        transmission=1.0,
        star_jmag=star_jmag,
        area_m2=area_per_aperture_m2,
        n_modes=n_modes,
    )
    amps = np.ones_like(state.field.amplitudes)
    state = replace(
        state,
        field=replace(state.field, amplitudes=amps),
        geometry=replace(
            state.geometry,
            n_apertures=n_apertures,
            collecting_area_m2=None,  # exercise the fallback
        ),
    )

    stage = _stage(
        _context(tmp_path),
        qe=1.0,
        mkid_max_count_rate_hz=1.0e18,
        n_modes=n_modes,
    )
    result = stage.apply(state)
    total_rate_hz = float(result.photon_rates.rates_hz.sum())

    wl_um = np.asarray(state.field.wavelength_grid_um, dtype=np.float64)
    bandwidth_um = (wl_um[-1] - wl_um[0]) / (wl_um.size - 1)
    stellar_rate_per_m2 = np.array(
        [
            photon_rate_per_m2_per_bin(star_jmag, float(wl), bandwidth_um, band="J")
            for wl in wl_um
        ],
        dtype=np.float64,
    )
    expected = float(n_apertures) * area_per_aperture_m2 * stellar_rate_per_m2.sum()
    np.testing.assert_allclose(total_rate_hz, expected, rtol=1.0e-12)


def test_mkid_saturation_cap_per_pixel(tmp_path: Path) -> None:
    # Drive the bright port well above the cap; the PORT-TOTAL rate
    # (summed across all wavelength bins) is capped at cap_hz × n_pixels,
    # NOT cap_hz × n_pixels × n_wavelength. Per-bin rates are rescaled
    # proportionally so the spectral shape is preserved. Pre-fix the
    # cap was applied to each (port, bin) cell independently then summed,
    # so the port total was 8× too high here (Code Review 2026-05-16
    # Ultra-review High #4).
    cap_hz = 5.0e4
    n_wavelength = 8
    state = _state(
        n_wavelength=n_wavelength,
        bright_amplitude_sq=1.0,
        dark_amplitude_sq=0.0,
        star_jmag=-10.0,  # very bright → guarantees we exceed cap
    )
    stage = _stage(
        _context(tmp_path),
        qe=1.0,
        mkid_max_count_rate_hz=cap_hz,
        n_pixels_bright=1,
        n_pixels_dark=1,
    )

    result = stage.apply(state)
    bright_rates = result.photon_rates.rates_hz[0, :]

    # Per-bin rates rescaled proportionally so the SUM equals cap_hz × n_pixels.
    # Per-bin values are not uniform because the J-band stellar
    # photon-rate spectrum is wavelength-dependent.
    assert bright_rates.sum() == pytest.approx(cap_hz)  # n_pixels=1
    component = result.results.components["detector.mkid"]
    assert component["saturated_per_port"][0] is np.True_ or bool(
        component["saturated_per_port"][0]
    )
    assert component["port_rates"][0] == pytest.approx(cap_hz)


def test_mkid_per_pixel_cap_scales_with_n_pixels(tmp_path: Path) -> None:
    # With 4 pixels per bright port and the same saturating drive, the
    # total port rate must be 4× cap_hz (n_pixels × per-pixel cap). The
    # n_wavelength factor does NOT appear — port saturation is at the
    # port-total rate level, not per-bin (see Code Review 2026-05-16
    # Ultra-review High #4).
    cap_hz = 5.0e4
    n_wavelength = 8
    n_pixels = 4
    state = _state(
        n_wavelength=n_wavelength,
        bright_amplitude_sq=1.0,
        dark_amplitude_sq=0.0,
        star_jmag=-10.0,
    )
    stage = _stage(
        _context(tmp_path),
        qe=1.0,
        mkid_max_count_rate_hz=cap_hz,
        n_pixels_bright=n_pixels,
        n_pixels_dark=1,
    )

    result = stage.apply(state)
    component = result.results.components["detector.mkid"]
    assert component["port_rates"][0] == pytest.approx(n_pixels * cap_hz)


def test_mkid_resolution_disabled_preserves_default_payload(tmp_path: Path) -> None:
    state = _state()
    context = _context(tmp_path)
    baseline = _stage(context).apply(state)
    explicit_disabled = MKIDDetector(
        {
            "qe": 1.0,
            "mkid_max_count_rate_hz": 1.0e15,
            "n_pixels_bright": 1,
            "n_pixels_dark": 1,
            "n_bright_ports": 1,
            "n_dark_ports": 3,
            "n_spectro_ports": 0,
            "n_modes": 4,
            "integration_time_s": 1.0,
            "band": "J",
            "derive_band_mag_from_scene_jmag": False,
            "resolving_power_ref": None,
            "lambda_ref_um": None,
        },
        context,
    ).apply(state)

    baseline_component = baseline.results.components["detector.mkid"]
    disabled_component = explicit_disabled.results.components["detector.mkid"]
    assert disabled_component.keys() == baseline_component.keys()
    np.testing.assert_allclose(
        explicit_disabled.photon_rates.rates_hz,
        baseline.photon_rates.rates_hz,
    )
    for key in (
        "resolving_power_ref",
        "lambda_ref_um",
        "resolving_power_per_bin",
        "true_rates_hz_per_port_wavelength",
        "energy_response_matrix",
        "energy_response_tail_loss_per_true_bin",
    ):
        assert key not in disabled_component


def test_mkid_resolution_redistributes_spectrum_and_conserves_totals(
    tmp_path: Path,
) -> None:
    state = _state(
        n_modes=2,
        n_wavelength=11,
        wl_center_um=1.0,
        wl_half_width_um=0.1,
        bright_amplitude_sq=0.0,
        dark_amplitude_sq=0.0,
    )
    amplitudes = np.zeros((2, 11, 1), dtype=np.complex128)
    center_index = 5
    amplitudes[0, center_index, 0] = 1.0
    amplitudes[1, center_index, 0] = np.sqrt(1.0e-4)
    state = replace(state, field=replace(state.field, amplitudes=amplitudes))
    stage = _stage(
        _context(tmp_path),
        n_modes=2,
        n_bright_ports=1,
        n_dark_ports=1,
        qe=1.0,
        mkid_max_count_rate_hz=1.0e18,
        resolving_power_ref=30.0,
        lambda_ref_um=1.0,
    )

    result = stage.apply(state)
    component = result.results.components["detector.mkid"]
    measured = result.photon_rates.rates_hz
    true_rates = component["true_rates_hz_per_port_wavelength"]

    assert np.count_nonzero(true_rates[0]) == 1
    assert np.count_nonzero(measured[0] > 0.0) > 1
    np.testing.assert_allclose(measured.sum(axis=1), true_rates.sum(axis=1))
    np.testing.assert_allclose(component["port_rates"], true_rates.sum(axis=1))
    assert component["null_depth_raw"] == pytest.approx(1.0e-4)
    np.testing.assert_allclose(
        component["energy_response_matrix"].sum(axis=0),
        np.ones(11),
    )
    assert component["energy_response_tail_loss_per_true_bin"][0] > 0.0
    assert component["resolving_power_per_bin"][center_index] == pytest.approx(30.0)


def test_mkid_resolution_applies_cap_after_redistribution(tmp_path: Path) -> None:
    cap_hz = 5.0e4
    state = _state(
        n_modes=2,
        n_wavelength=11,
        wl_center_um=1.0,
        wl_half_width_um=0.1,
        bright_amplitude_sq=0.0,
        dark_amplitude_sq=0.0,
        star_jmag=-10.0,
    )
    amplitudes = np.zeros((2, 11, 1), dtype=np.complex128)
    amplitudes[0, 5, 0] = 1.0
    state = replace(state, field=replace(state.field, amplitudes=amplitudes))
    stage = _stage(
        _context(tmp_path),
        n_modes=2,
        n_bright_ports=1,
        n_dark_ports=1,
        qe=1.0,
        mkid_max_count_rate_hz=cap_hz,
        n_pixels_bright=1,
        n_pixels_dark=1,
        resolving_power_ref=30.0,
        lambda_ref_um=1.0,
    )

    result = stage.apply(state)
    component = result.results.components["detector.mkid"]

    assert component["port_rates"][0] == pytest.approx(cap_hz)
    assert result.photon_rates.rates_hz[0].sum() == pytest.approx(cap_hz)
    assert bool(component["saturated_per_port"][0])


def test_mkid_resolution_requires_multiple_wavelength_bins(tmp_path: Path) -> None:
    state = _state(n_wavelength=1, wl_center_um=1.0, wl_half_width_um=0.0)
    stage = _stage(
        _context(tmp_path),
        resolving_power_ref=30.0,
        lambda_ref_um=1.0,
    )

    with pytest.raises(ValueError, match="grid.n_wavelength_bins"):
        stage.apply(state)


def test_mkid_null_depth_uses_bright_dark_partition(tmp_path: Path) -> None:
    # With bright port carrying ~1 and dark ports carrying 1e-8 of power each,
    # null_depth_raw = mean(dark)/mean(bright) ≈ 1e-8 / 1.0.
    state = _state(
        bright_amplitude_sq=1.0,
        dark_amplitude_sq=1.0e-8,
    )
    stage = _stage(
        _context(tmp_path),
        qe=1.0,
        mkid_max_count_rate_hz=1.0e15,
    )

    result = stage.apply(state)
    component = result.results.components["detector.mkid"]
    bright_rate = component["port_rates"][0]
    dark_mean = np.mean(component["port_rates"][1:])
    expected_null = dark_mean / bright_rate
    assert component["null_depth_raw"] == pytest.approx(expected_null, rel=1.0e-12)


def test_mkid_labels_spectro_port_without_counting_it_as_dark(
    tmp_path: Path,
) -> None:
    state = _state(
        n_modes=4,
        bright_amplitude_sq=1.0,
        dark_amplitude_sq=1.0e-8,
    )
    stage = _stage(
        _context(tmp_path),
        n_modes=4,
        n_bright_ports=1,
        n_dark_ports=2,
        n_spectro_ports=1,
        qe=1.0,
        mkid_max_count_rate_hz=1.0e15,
    )

    result = stage.apply(state)
    component = result.results.components["detector.mkid"]

    assert component["port_labels"] == ("bright_0", "dark_0", "dark_1", "spectro_0")
    assert component["n_spectro_ports"] == 1
    bright_rate = component["port_rates"][0]
    dark_mean = np.mean(component["port_rates"][1:3])
    assert component["null_depth_raw"] == pytest.approx(dark_mean / bright_rate)


def test_mkid_preserves_two_pbs_arms_as_physical_channels(tmp_path: Path) -> None:
    n_modes = 4
    state = _state(n_modes=n_modes, n_wavelength=2, star_jmag=5.0, area_m2=50.0)
    powers = np.array(
        [
            [0.25, 0.75],
            [1.0e-8, 4.0e-8],
            [2.0e-8, 5.0e-8],
            [3.0e-8, 6.0e-8],
        ],
        dtype=np.float64,
    )
    amps = np.zeros((n_modes, 2, 2), dtype=np.complex128)
    amps[:, :, :] = np.sqrt(powers)[:, np.newaxis, :]
    state = replace(
        state,
        field=replace(
            state.field,
            amplitudes=amps,
            components={"polarization.pbs_split": {"polarization_labels": ("A", "B")}},
        ),
    )
    stage = _stage(
        _context(tmp_path),
        n_modes=n_modes,
        n_bright_ports=1,
        n_dark_ports=3,
        qe=1.0,
        mkid_max_count_rate_hz=1.0e18,
    )

    result = stage.apply(state)

    assert result.photon_rates.port_labels == (
        "bright_0_A",
        "bright_0_B",
        "dark_0_A",
        "dark_0_B",
        "dark_1_A",
        "dark_1_B",
        "dark_2_A",
        "dark_2_B",
    )
    wl_um = np.asarray(state.field.wavelength_grid_um, dtype=np.float64)
    bandwidth_um = (wl_um[-1] - wl_um[0]) / (wl_um.size - 1)
    stellar_rate_per_m2 = np.array(
        [
            photon_rate_per_m2_per_bin(5.0, float(wl), bandwidth_um, band="J")
            for wl in wl_um
        ],
        dtype=np.float64,
    )
    power_per_physical_port = np.transpose(np.abs(amps) ** 2, (0, 2, 1)).reshape(
        n_modes * 2,
        2,
    )
    expected = power_per_physical_port * (stellar_rate_per_m2 * 50.0 / n_modes)
    np.testing.assert_allclose(result.photon_rates.rates_hz, expected, rtol=1.0e-12)


def test_mkid_caps_two_pol_bright_arms_with_bright_pixel_pool(
    tmp_path: Path,
) -> None:
    cap_hz = 5.0e4
    state = _state(
        n_modes=2,
        n_wavelength=2,
        star_jmag=-10.0,
        area_m2=50.0,
    )
    amps = np.zeros((2, 2, 2), dtype=np.complex128)
    amps[0, :, :] = 1.0
    state = replace(
        state,
        field=replace(
            state.field,
            amplitudes=amps,
            components={"polarization.pbs_split": {"polarization_labels": ("A", "B")}},
        ),
    )
    stage = _stage(
        _context(tmp_path),
        n_modes=2,
        n_bright_ports=1,
        n_dark_ports=1,
        qe=1.0,
        mkid_max_count_rate_hz=cap_hz,
        n_pixels_bright=4,
        n_pixels_dark=1,
    )

    result = stage.apply(state)
    component = result.results.components["detector.mkid"]

    assert component["port_labels"][:2] == ("bright_0_A", "bright_0_B")
    assert component["port_rates"][0] == pytest.approx(4.0 * cap_hz)
    assert component["port_rates"][1] == pytest.approx(4.0 * cap_hz)
    assert bool(component["saturated_per_port"][0])
    assert bool(component["saturated_per_port"][1])


def test_mkid_throughput_multiplies_into_rate(tmp_path: Path) -> None:
    # Halving throughput halves the rate (linear in transmission).
    state_full = _state(transmission=1.0)
    state_half = _state(transmission=0.5)
    stage = _stage(_context(tmp_path), qe=1.0, mkid_max_count_rate_hz=1.0e15)

    rates_full = stage.apply(state_full).photon_rates.rates_hz
    rates_half = stage.apply(state_half).photon_rates.rates_hz

    np.testing.assert_allclose(rates_half, 0.5 * rates_full, rtol=1.0e-12)


def test_mkid_qe_above_one_rejected_by_params() -> None:
    with pytest.raises(ValidationError, match="qe"):
        MKIDDetector.Params(
            qe=1.5,
            n_pixels_bright=1,
            n_pixels_dark=1,
            n_bright_ports=1,
            n_dark_ports=3,
            n_modes=4,
            integration_time_s=1.0,
        )


def test_mkid_qe_zero_rejected_by_params() -> None:
    with pytest.raises(ValidationError, match="qe"):
        MKIDDetector.Params(
            qe=0.0,
            n_pixels_bright=1,
            n_pixels_dark=1,
            n_bright_ports=1,
            n_dark_ports=3,
            n_modes=4,
            integration_time_s=1.0,
        )


def test_mkid_resolution_requires_lambda_ref_when_enabled() -> None:
    with pytest.raises(ValidationError, match="lambda_ref_um"):
        MKIDDetector.Params(
            qe=1.0,
            n_pixels_bright=1,
            n_pixels_dark=1,
            n_bright_ports=1,
            n_dark_ports=3,
            n_modes=4,
            integration_time_s=1.0,
            resolving_power_ref=30.0,
        )


def test_mkid_port_count_mismatch_rejected_by_params() -> None:
    with pytest.raises(ValidationError, match="n_bright_ports"):
        MKIDDetector.Params(
            qe=0.7,
            n_pixels_bright=1,
            n_pixels_dark=1,
            n_bright_ports=2,
            n_dark_ports=2,
            n_modes=5,
            integration_time_s=1.0,
        )


def test_mkid_n_pixels_bright_zero_rejected_by_params() -> None:
    with pytest.raises(ValidationError, match="n_pixels_bright"):
        MKIDDetector.Params(
            qe=0.7,
            n_pixels_bright=0,
            n_pixels_dark=1,
            n_bright_ports=1,
            n_dark_ports=3,
            n_modes=4,
            integration_time_s=1.0,
        )


def test_mkid_n_pixels_dark_zero_rejected_by_params() -> None:
    with pytest.raises(ValidationError, match="n_pixels_dark"):
        MKIDDetector.Params(
            qe=0.7,
            n_pixels_bright=1,
            n_pixels_dark=0,
            n_bright_ports=1,
            n_dark_ports=3,
            n_modes=4,
            integration_time_s=1.0,
        )


def test_mkid_no_throughput_modifications(tmp_path: Path) -> None:
    # The detector consumes throughput and now folds the MKID QE scalar
    # into the multiplicative spectrum so downstream throughput-breakdown
    # plots see a QE-aware transmission. With
    # qe=1.0 (the fixture default) the spectrum stays bit-identical but a
    # single ``detector.mkid_qe`` component is added.
    state = _state()
    stage = _stage(_context(tmp_path))
    result = stage.apply(state)
    assert result.throughput.transmission_fraction == state.throughput.transmission_fraction
    extra_components = set(result.throughput.components) - set(state.throughput.components)
    assert extra_components == {"detector.mkid_qe"}
    qe_payload = result.throughput.components["detector.mkid_qe"]
    assert qe_payload["qe"] == 1.0
