"""Tests for characterization spectroscopy primitives."""

from __future__ import annotations

import pytest

from nullsim.physics.spectroscopy import (
    awg_bin_count,
    awg_spectrum,
    max_spectral_resolution,
    per_bin_snr,
)


def test_per_bin_snr_decreases_with_resolving_power() -> None:
    kwargs = {
        "planet_rate_total_hz": 1000.0,
        "leakage_rate_total_hz": 100.0,
        "spectro_throughput": 1.0,
        "band_lambda_m": 1.25e-6,
        "band_bandwidth_m": 0.15e-6,
        "integration_time_s": 3600.0,
        "dark_rate_hz_per_pixel": 0.0,
    }
    assert per_bin_snr(resolving_power=100.0, **kwargs) > per_bin_snr(
        resolving_power=1000.0,
        **kwargs,
    )


def test_per_bin_snr_honors_pixels_per_resolution_element() -> None:
    kwargs = {
        "planet_rate_total_hz": 1000.0,
        "leakage_rate_total_hz": 100.0,
        "spectro_throughput": 1.0,
        "band_lambda_m": 1.25e-6,
        "band_bandwidth_m": 0.15e-6,
        "resolving_power": 100.0,
        "integration_time_s": 3600.0,
        "dark_rate_hz_per_pixel": 0.0,
    }

    native = per_bin_snr(pixels_per_resolution_element=1, **kwargs)
    nyquist = per_bin_snr(pixels_per_resolution_element=2, **kwargs)

    assert native > nyquist
    assert native / nyquist == pytest.approx(2.0**0.5, rel=1.0e-12)


def test_max_spectral_resolution_returns_cap_when_cap_is_feasible() -> None:
    max_r = max_spectral_resolution(
        planet_rate_total_hz=1.0e8,
        leakage_rate_total_hz=1.0,
        spectro_throughput=1.0,
        band_lambda_m=1.25e-6,
        band_bandwidth_m=0.15e-6,
        integration_time_s=3600.0,
        snr_min=3.0,
        resolving_power_cap=5000.0,
    )
    assert max_r == pytest.approx(5000.0)


def test_max_spectral_resolution_returns_zero_when_floor_is_infeasible() -> None:
    max_r = max_spectral_resolution(
        planet_rate_total_hz=1.0e-9,
        leakage_rate_total_hz=1.0,
        spectro_throughput=1.0,
        band_lambda_m=1.25e-6,
        band_bandwidth_m=0.15e-6,
        integration_time_s=1.0,
        snr_min=3.0,
        resolving_power_floor=1.0,
        resolving_power_cap=10.0,
    )
    assert max_r == 0.0


def test_max_spectral_resolution_honors_pixels_per_resolution_element() -> None:
    common = {
        "planet_rate_total_hz": 1.0e3,
        "leakage_rate_total_hz": 100.0,
        "spectro_throughput": 1.0,
        "band_lambda_m": 1.25e-6,
        "band_bandwidth_m": 0.15e-6,
        "integration_time_s": 3600.0,
        "snr_min": 100.0,
        "resolving_power_floor": 1.0,
        "resolving_power_cap": 10_000.0,
    }

    native = max_spectral_resolution(pixels_per_resolution_element=1, **common)
    nyquist = max_spectral_resolution(pixels_per_resolution_element=2, **common)

    assert native > nyquist


def test_per_bin_snr_sky_splits_with_bin_fraction() -> None:
    """Sky enters the spectro port through the band-pass and disperses
    over bins — so per-bin sky counts must scale with ``bin_fraction``.
    In the sky-dominated limit, doubling the resolving power should
    drop SNR by exactly ``sqrt(2)``: signal/bin halves linearly while
    sky/bin halves too, so noise/bin drops as ``sqrt(1/2)``.
    """

    common = {
        "planet_rate_total_hz": 1.0,
        "leakage_rate_total_hz": 0.0,
        "spectro_throughput": 1.0,
        "band_lambda_m": 1.25e-6,
        "band_bandwidth_m": 0.15e-6,
        "integration_time_s": 1.0,
        "sky_rate_hz_per_spectro_port": 1.0e6,
    }
    snr_low = per_bin_snr(resolving_power=100.0, **common)
    snr_high = per_bin_snr(resolving_power=200.0, **common)
    assert snr_low > snr_high
    assert snr_low / snr_high == pytest.approx(2.0 ** 0.5, rel=1.0e-3)


def test_awg_bin_count_uses_nyquist_sampling_and_fsr() -> None:
    assert awg_bin_count(
        resolving_power=1000.0,
        band_lambda_m=1.25e-6,
        band_bandwidth_m=0.15e-6,
        free_spectral_range_fraction=0.5,
        pixels_per_resolution_element=2,
    ) == 120


def test_awg_spectrum_applies_optical_throughput_and_fsr() -> None:
    spectrum = awg_spectrum(
        planet_rate_total_hz=1000.0,
        leakage_rate_total_hz=100.0,
        sky_rate_total_hz=50.0,
        throughput=0.5,
        resolving_power=100.0,
        band_lambda_m=1.25e-6,
        band_bandwidth_m=0.15e-6,
        integration_time_s=10.0,
        dark_rate_hz_per_pixel=2.0,
        n_detector_pixels_per_bin=3,
        free_spectral_range_fraction=0.5,
        pixels_per_resolution_element=2,
    )

    assert spectrum.n_bins == 12
    assert spectrum.bin_edges_m[0] == pytest.approx(1.25e-6 - 0.5 * 0.075e-6)
    assert spectrum.bin_edges_m[-1] == pytest.approx(1.25e-6 + 0.5 * 0.075e-6)
    assert spectrum.planet_rate_hz_per_bin.sum() == pytest.approx(250.0)
    assert spectrum.leakage_rate_hz_per_bin.sum() == pytest.approx(25.0)
    assert spectrum.sky_rate_hz_per_bin.sum() == pytest.approx(12.5)
    assert spectrum.dark_rate_hz_per_bin[0] == pytest.approx(6.0)
    assert spectrum.snr_per_bin.min() > 0.0
