"""Tests for the SNSPDDetector stage (SPEC §15.3 Pass 4a).

Reference invariants come from KILO ``detect_snspd``
(``kilo/detectors.py`` L289-309). See
``tests/stages/notes/detector_snspd.md`` for the source survey.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from nullsim.physics.photometry import photon_rate_per_m2_per_bin
from nullsim.pipeline import (
    Field,
    Geometry,
    RunContext,
    Scene,
    SimulationState,
    Throughput,
    build_instance,
)
from nullsim.stages.detector import SNSPDDetector


def _context(tmp_path: Path) -> RunContext:
    return RunContext(
        rng=np.random.default_rng(0),
        config_hash="detector-snspd-pinning",
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
    scene = Scene(star_jmag=star_jmag)
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
    snspd_max_count_rate_hz: float = 1.0e15,
    snspd_dark_rate_hz: float = 0.0,
    snspd_timing_resolution_s: float = 50.0e-12,
    integration_time_s: float = 1.0,
    band: str = "J",
) -> SNSPDDetector:
    stage = build_instance(
        {
            "id": "detector",
            "type": "detector.snspd",
            "qe": qe,
            "snspd_max_count_rate_hz": snspd_max_count_rate_hz,
            "snspd_dark_rate_hz": snspd_dark_rate_hz,
            "snspd_timing_resolution_s": snspd_timing_resolution_s,
            "n_bright_ports": n_bright_ports,
            "n_dark_ports": n_dark_ports,
            "n_spectro_ports": n_spectro_ports,
            "n_modes": n_modes,
            "integration_time_s": integration_time_s,
            "band": band,
        },
        context,
    )
    assert isinstance(stage, SNSPDDetector)
    return stage


def test_snspd_unit_qe_no_dark_matches_analytical_rate(tmp_path: Path) -> None:
    # See ``test_mkid_unit_qe_matches_analytical_field_squared_rate``: per-port
    # rate = |field|² × stellar_rate × (collecting_area / n_modes). With the
    # single-aperture legacy fallback this is area_m2 / n_modes.
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
        snspd_max_count_rate_hz=1.0e15,
        snspd_dark_rate_hz=0.0,
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


def test_snspd_does_not_apply_input_area_vector_to_output_ports(
    tmp_path: Path,
) -> None:
    """Input-mode area vectors are not output-port area vectors post-chip."""

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
        snspd_max_count_rate_hz=1.0e15,
        snspd_dark_rate_hz=0.0,
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


def test_snspd_saturation_cap_is_fixed_per_port(tmp_path: Path) -> None:
    cap_hz = 1.0e8
    n_wavelength = 8
    state = _state(
        n_wavelength=n_wavelength,
        bright_amplitude_sq=1.0,
        dark_amplitude_sq=0.0,
        star_jmag=-15.0,  # crank past 1e8 even with low per-bin rate
    )
    stage = _stage(
        _context(tmp_path),
        qe=1.0,
        snspd_max_count_rate_hz=cap_hz,
        snspd_dark_rate_hz=0.0,
    )

    result = stage.apply(state)
    bright_rates = result.photon_rates.rates_hz[0, :]

    # Cap is on the PORT-TOTAL rate, not per (port, bin). Per-bin rates
    # are rescaled proportionally so the SUM equals cap_hz (Code Review
    # 2026-05-16 Ultra-review High #4). Per-bin values are not uniform
    # because the J-band stellar photon-rate spectrum is wavelength-dependent.
    assert bright_rates.sum() == pytest.approx(cap_hz)
    component = result.results.components["detector.snspd"]
    assert bool(component["saturated_per_port"][0])
    assert component["port_rates"][0] == pytest.approx(cap_hz)


def test_snspd_dark_counts_reported_separately_with_zero_field(tmp_path: Path) -> None:
    # All ports get the dark rate in total diagnostics, while stellar
    # ``port_rates`` remain dark-free for downstream contrast normalization.
    state = _state(
        bright_amplitude_sq=0.0,
        dark_amplitude_sq=0.0,
    )
    dark_rate_hz = 10.0
    stage = _stage(
        _context(tmp_path),
        qe=1.0,
        snspd_max_count_rate_hz=1.0e15,
        snspd_dark_rate_hz=dark_rate_hz,
    )

    result = stage.apply(state)
    component = result.results.components["detector.snspd"]

    np.testing.assert_allclose(component["port_rates"], 0.0)
    np.testing.assert_allclose(component["total_port_rates_with_dark"], dark_rate_hz)
    np.testing.assert_allclose(result.photon_rates.rates_hz.sum(axis=1), 0.0)


def test_snspd_null_depth_uses_bright_dark_partition(tmp_path: Path) -> None:
    state = _state(
        bright_amplitude_sq=1.0,
        dark_amplitude_sq=1.0e-8,
    )
    stage = _stage(
        _context(tmp_path),
        qe=1.0,
        snspd_max_count_rate_hz=1.0e15,
        snspd_dark_rate_hz=0.0,
    )

    result = stage.apply(state)
    component = result.results.components["detector.snspd"]
    bright_rate = component["port_rates"][0]
    dark_mean = np.mean(component["port_rates"][1:])
    expected_null = dark_mean / bright_rate
    assert component["null_depth_raw"] == pytest.approx(expected_null, rel=1.0e-12)


def test_snspd_labels_spectro_port_without_counting_it_as_dark(
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
        snspd_max_count_rate_hz=1.0e15,
        snspd_dark_rate_hz=0.0,
    )

    result = stage.apply(state)
    component = result.results.components["detector.snspd"]

    assert component["port_labels"] == ("bright_0", "dark_0", "dark_1", "spectro_0")
    assert component["n_spectro_ports"] == 1
    bright_rate = component["port_rates"][0]
    dark_mean = np.mean(component["port_rates"][1:3])
    assert component["null_depth_raw"] == pytest.approx(dark_mean / bright_rate)


def test_snspd_throughput_multiplies_into_rate(tmp_path: Path) -> None:
    state_full = _state(transmission=1.0)
    state_half = _state(transmission=0.5)
    stage = _stage(
        _context(tmp_path),
        qe=1.0,
        snspd_max_count_rate_hz=1.0e15,
        snspd_dark_rate_hz=0.0,
    )

    rates_full = stage.apply(state_full).photon_rates.rates_hz
    rates_half = stage.apply(state_half).photon_rates.rates_hz

    np.testing.assert_allclose(rates_half, 0.5 * rates_full, rtol=1.0e-12)


def test_snspd_preserves_two_pbs_arms_as_physical_channels(tmp_path: Path) -> None:
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
            components={
                "polarization.pbs_split": {
                    "polarization_labels": ("A", "B"),
                    "polarization_basis": "pbs_arms",
                }
            },
        ),
    )
    stage = _stage(
        _context(tmp_path),
        n_modes=n_modes,
        n_bright_ports=1,
        n_dark_ports=3,
        qe=1.0,
        snspd_max_count_rate_hz=1.0e18,
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


def test_snspd_qe_above_one_rejected_by_params() -> None:
    with pytest.raises(ValidationError, match="qe"):
        SNSPDDetector.Params(
            qe=1.5,
            n_bright_ports=1,
            n_dark_ports=3,
            n_modes=4,
            integration_time_s=1.0,
        )


def test_snspd_negative_qe_rejected_by_params() -> None:
    with pytest.raises(ValidationError, match="qe"):
        SNSPDDetector.Params(
            qe=-0.1,
            n_bright_ports=1,
            n_dark_ports=3,
            n_modes=4,
            integration_time_s=1.0,
        )


def test_snspd_negative_dark_rate_rejected_by_params() -> None:
    with pytest.raises(ValidationError, match="snspd_dark_rate_hz"):
        SNSPDDetector.Params(
            qe=0.85,
            snspd_dark_rate_hz=-1.0,
            n_bright_ports=1,
            n_dark_ports=3,
            n_modes=4,
            integration_time_s=1.0,
        )


def test_snspd_port_count_mismatch_rejected_by_params() -> None:
    with pytest.raises(ValidationError, match="n_bright_ports"):
        SNSPDDetector.Params(
            qe=0.85,
            n_bright_ports=2,
            n_dark_ports=2,
            n_modes=5,
            integration_time_s=1.0,
        )


def test_snspd_folds_qe_into_throughput(tmp_path: Path) -> None:
    state = _state()
    stage = _stage(_context(tmp_path), qe=0.8)
    result = stage.apply(state)
    expected = tuple(0.8 * value for value in state.throughput.transmission_fraction)
    assert result.throughput.transmission_fraction == expected
    qe_payload = result.throughput.components["detector.snspd_qe"]
    assert qe_payload["qe"] == pytest.approx(0.8)


@pytest.mark.stats
def test_snspd_dark_counts_add_linearly_to_signal(tmp_path: Path) -> None:
    # Total rate with dark = signal rate + n_wavelengths × per-bin dark distrib.
    # The per-bin dark = dark_rate / n_wavelengths, so summed across bins =
    # dark_rate exactly. Verify across all ports.
    state = _state()
    no_dark = _stage(
        _context(tmp_path),
        qe=1.0,
        snspd_max_count_rate_hz=1.0e15,
        snspd_dark_rate_hz=0.0,
    )
    with_dark = _stage(
        _context(tmp_path),
        qe=1.0,
        snspd_max_count_rate_hz=1.0e15,
        snspd_dark_rate_hz=10.0,
    )

    no_dark_payload = no_dark.apply(state).results.components["detector.snspd"]
    with_dark_payload = with_dark.apply(state).results.components["detector.snspd"]

    np.testing.assert_allclose(
        with_dark_payload["port_rates"],
        no_dark_payload["port_rates"],
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        with_dark_payload["total_port_rates_with_dark"]
        - no_dark_payload["total_port_rates_with_dark"],
        10.0,
        rtol=1.0e-12,
    )


def test_snspd_prefers_effective_collecting_area(tmp_path: Path) -> None:
    """SNSPD must scale per-mode rates with ``effective_collecting_area_m2``.

    KILO sub-aperture pipelines populate the effective area as the sum of
    packed sub-aperture footprints (smaller than the obstructed primary).
    Without the prefer-step, SNSPD overcounts per-mode rates by the
    area-equivalent overcount factor (state.py:183 contract).
    """
    state = _state(area_m2=50.0, transmission=1.0, star_jmag=5.0)
    # Refine the geometry as a sub-aperture stage (e.g. ao.pyramid_wfs) would:
    # effective area = 0.5 × collecting area (a smaller packed footprint).
    state = replace(
        state,
        geometry=replace(
            state.geometry,
            collecting_area_m2=50.0,
            effective_collecting_area_m2=25.0,
        ),
    )
    stage = _stage(
        _context(tmp_path),
        qe=1.0,
        snspd_max_count_rate_hz=1.0e15,
        snspd_dark_rate_hz=0.0,
    )

    rates = stage.apply(state).photon_rates.rates_hz

    wl_um = np.asarray(state.field.wavelength_grid_um, dtype=np.float64)
    bandwidth_um = (wl_um[-1] - wl_um[0]) / (wl_um.size - 1)
    expected_stellar = np.array(
        [
            photon_rate_per_m2_per_bin(5.0, float(wl), bandwidth_um, band="J")
            for wl in wl_um
        ],
        dtype=np.float64,
    )
    power_per_mode = np.sum(
        np.abs(state.field.amplitudes) ** 2, axis=2,
    ).astype(np.float64)
    n_modes = power_per_mode.shape[0]
    # Effective area, not collecting area or per-aperture area.
    area_per_mode_m2 = 25.0 / float(n_modes)
    expected = power_per_mode * (expected_stellar * area_per_mode_m2)[np.newaxis, :]

    np.testing.assert_allclose(rates, expected, rtol=1.0e-12)
