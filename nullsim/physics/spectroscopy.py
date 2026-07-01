"""Spectroscopic SNR primitives for a characterization spectro port."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scipy.optimize import brentq

__all__ = [
    "AWGSpectrum",
    "awg_bin_count",
    "awg_spectrum",
    "awg_wavelength_grid",
    "max_spectral_resolution",
    "per_bin_snr",
]


@dataclass(frozen=True)
class AWGSpectrum:
    """Rate-binned AWG spectrograph output for one spectro channel.

    Rates are per spectral bin after AWG optical throughput and free spectral
    range losses. Dark rates are per bin because detector noise does not split
    with wavelength coverage.
    """

    resolving_power: float
    throughput: float
    free_spectral_range_fraction: float
    order: int
    pixels_per_resolution_element: int
    n_detector_pixels_per_bin: int
    bin_edges_m: NDArray[np.float64]
    bin_centers_m: NDArray[np.float64]
    planet_rate_hz_per_bin: NDArray[np.float64]
    leakage_rate_hz_per_bin: NDArray[np.float64]
    sky_rate_hz_per_bin: NDArray[np.float64]
    dark_rate_hz_per_bin: NDArray[np.float64]
    snr_per_bin: NDArray[np.float64]

    @property
    def n_bins(self) -> int:
        """Number of Nyquist-sampled wavelength bins."""

        return int(self.bin_centers_m.size)


def awg_bin_count(
    *,
    resolving_power: float,
    band_lambda_m: float,
    band_bandwidth_m: float,
    free_spectral_range_fraction: float = 1.0,
    pixels_per_resolution_element: int = 2,
) -> int:
    """Return the Nyquist-sampled AWG bin count across the covered band."""

    _validate_awg_optics(
        resolving_power=resolving_power,
        band_lambda_m=band_lambda_m,
        band_bandwidth_m=band_bandwidth_m,
        free_spectral_range_fraction=free_spectral_range_fraction,
        pixels_per_resolution_element=pixels_per_resolution_element,
    )
    raw_n_bins = (
        int(pixels_per_resolution_element)
        * resolving_power
        * band_bandwidth_m
        * free_spectral_range_fraction
        / band_lambda_m
    )
    nearest = round(raw_n_bins)
    if math.isclose(raw_n_bins, nearest, rel_tol=1.0e-12, abs_tol=1.0e-12):
        n_bins = int(nearest)
    else:
        n_bins = math.ceil(raw_n_bins)
    return max(1, int(n_bins))


def awg_wavelength_grid(
    *,
    resolving_power: float,
    band_lambda_m: float,
    band_bandwidth_m: float,
    free_spectral_range_fraction: float = 1.0,
    pixels_per_resolution_element: int = 2,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return AWG bin edges and centers for a band centered on ``band_lambda_m``."""

    n_bins = awg_bin_count(
        resolving_power=resolving_power,
        band_lambda_m=band_lambda_m,
        band_bandwidth_m=band_bandwidth_m,
        free_spectral_range_fraction=free_spectral_range_fraction,
        pixels_per_resolution_element=pixels_per_resolution_element,
    )
    covered_bandwidth_m = band_bandwidth_m * free_spectral_range_fraction
    lower_m = band_lambda_m - 0.5 * covered_bandwidth_m
    upper_m = band_lambda_m + 0.5 * covered_bandwidth_m
    if lower_m <= 0.0:
        raise ValueError("AWG wavelength grid lower edge must be positive.")
    bin_edges_m = np.linspace(lower_m, upper_m, n_bins + 1, dtype=np.float64)
    bin_centers_m = 0.5 * (bin_edges_m[:-1] + bin_edges_m[1:])
    return bin_edges_m, bin_centers_m


def awg_spectrum(
    *,
    planet_rate_total_hz: float,
    leakage_rate_total_hz: float,
    sky_rate_total_hz: float,
    throughput: float,
    resolving_power: float,
    band_lambda_m: float,
    band_bandwidth_m: float,
    integration_time_s: float,
    dark_rate_hz_per_pixel: float = 0.0,
    n_detector_pixels_per_bin: int = 1,
    free_spectral_range_fraction: float = 1.0,
    order: int = 1,
    pixels_per_resolution_element: int = 2,
    null_instability: float = 0.0,
    servo_bandwidth_hz: float = 0.0,
) -> AWGSpectrum:
    """Return the binned AWG spectrum and per-bin SNR.

    ``planet_rate_total_hz``, ``leakage_rate_total_hz``, and
    ``sky_rate_total_hz`` are band-integrated rates entering the AWG. The AWG
    throughput applies to all optical rates; detector dark current is added per
    bin after dispersion.
    """

    _validate_awg_rates(
        planet_rate_total_hz=planet_rate_total_hz,
        leakage_rate_total_hz=leakage_rate_total_hz,
        sky_rate_total_hz=sky_rate_total_hz,
        dark_rate_hz_per_pixel=dark_rate_hz_per_pixel,
    )
    if not 0.0 <= throughput <= 1.0:
        raise ValueError("AWG throughput must be between 0 and 1.")
    if integration_time_s <= 0.0:
        raise ValueError("integration_time_s must be positive.")
    if int(n_detector_pixels_per_bin) < 1:
        raise ValueError("n_detector_pixels_per_bin must be >= 1.")
    if int(order) < 1:
        raise ValueError("AWG order must be >= 1.")
    if null_instability < 0.0:
        raise ValueError("null_instability must be non-negative.")
    if servo_bandwidth_hz < 0.0:
        raise ValueError("servo_bandwidth_hz must be non-negative.")

    bin_edges_m, bin_centers_m = awg_wavelength_grid(
        resolving_power=resolving_power,
        band_lambda_m=band_lambda_m,
        band_bandwidth_m=band_bandwidth_m,
        free_spectral_range_fraction=free_spectral_range_fraction,
        pixels_per_resolution_element=pixels_per_resolution_element,
    )
    n_bins = int(bin_centers_m.size)
    bin_fraction = free_spectral_range_fraction / n_bins

    planet_rate_hz = planet_rate_total_hz * throughput * bin_fraction
    leakage_rate_hz = leakage_rate_total_hz * throughput * bin_fraction
    sky_rate_hz = sky_rate_total_hz * throughput * bin_fraction
    dark_rate_hz = dark_rate_hz_per_pixel * int(n_detector_pixels_per_bin)

    planet_rate_hz_per_bin = np.full(n_bins, planet_rate_hz, dtype=np.float64)
    leakage_rate_hz_per_bin = np.full(n_bins, leakage_rate_hz, dtype=np.float64)
    sky_rate_hz_per_bin = np.full(n_bins, sky_rate_hz, dtype=np.float64)
    dark_rate_hz_per_bin = np.full(n_bins, dark_rate_hz, dtype=np.float64)

    signal_counts = planet_rate_hz_per_bin * integration_time_s
    leakage_counts = leakage_rate_hz_per_bin * integration_time_s
    sky_counts = sky_rate_hz_per_bin * integration_time_s
    dark_counts = dark_rate_hz_per_bin * integration_time_s
    noise_variance = signal_counts + leakage_counts + sky_counts + dark_counts
    if null_instability > 0.0:
        if servo_bandwidth_hz > 0.0:
            noise_variance += (
                null_instability
                * null_instability
                * leakage_rate_hz_per_bin
                * leakage_rate_hz_per_bin
                * integration_time_s
                / (2.0 * servo_bandwidth_hz)
            )
        else:
            noise_variance += (null_instability * leakage_counts) ** 2

    snr_per_bin = np.divide(
        signal_counts,
        np.sqrt(noise_variance),
        out=np.zeros_like(signal_counts),
        where=(signal_counts > 0.0) & (noise_variance > 0.0),
    )
    return AWGSpectrum(
        resolving_power=float(resolving_power),
        throughput=float(throughput),
        free_spectral_range_fraction=float(free_spectral_range_fraction),
        order=int(order),
        pixels_per_resolution_element=int(pixels_per_resolution_element),
        n_detector_pixels_per_bin=int(n_detector_pixels_per_bin),
        bin_edges_m=bin_edges_m,
        bin_centers_m=bin_centers_m,
        planet_rate_hz_per_bin=planet_rate_hz_per_bin,
        leakage_rate_hz_per_bin=leakage_rate_hz_per_bin,
        sky_rate_hz_per_bin=sky_rate_hz_per_bin,
        dark_rate_hz_per_bin=dark_rate_hz_per_bin,
        snr_per_bin=snr_per_bin,
    )


def per_bin_snr(
    *,
    planet_rate_total_hz: float,
    leakage_rate_total_hz: float,
    spectro_throughput: float,
    band_lambda_m: float,
    band_bandwidth_m: float,
    resolving_power: float,
    integration_time_s: float,
    dark_rate_hz_per_pixel: float = 0.0,
    n_spectro_pixels: int = 1,
    pixels_per_resolution_element: int = 2,
    null_instability: float = 0.0,
    servo_bandwidth_hz: float = 0.0,
    sky_rate_hz_per_spectro_port: float = 0.0,
) -> float:
    """Return SNR in one spectral bin.

    The full-band planet, leakage, and sky rates are split over
    ``pixels_per_resolution_element * R * Delta_lambda / lambda`` bins.
    AWG callers use ``pixels_per_resolution_element=2`` for Nyquist
    sampling; native MKID callers use ``1``. Detector dark counts are per
    spectro pixel and do not split with resolving power.
    ``sky_rate_hz_per_spectro_port`` is interpreted as the band-integrated
    sky rate through one spectro port, matching the convention used by
    ``_compute_snr`` in ``performance.standard``.
    """

    if resolving_power <= 0.0 or integration_time_s <= 0.0:
        return 0.0
    if band_lambda_m <= 0.0 or band_bandwidth_m <= 0.0:
        return 0.0
    if int(pixels_per_resolution_element) < 1:
        raise ValueError("pixels_per_resolution_element must be >= 1.")

    n_bins = (
        float(int(pixels_per_resolution_element))
        * resolving_power
        * band_bandwidth_m
        / band_lambda_m
    )
    if n_bins <= 0.0:
        return 0.0
    bin_fraction = 1.0 / n_bins

    signal_rate_hz = planet_rate_total_hz * spectro_throughput * bin_fraction
    leakage_rate_hz = leakage_rate_total_hz * spectro_throughput * bin_fraction
    sky_rate_hz = sky_rate_hz_per_spectro_port * spectro_throughput * bin_fraction

    signal_counts = signal_rate_hz * integration_time_s
    leakage_counts = leakage_rate_hz * integration_time_s
    dark_counts = (
        dark_rate_hz_per_pixel * int(n_spectro_pixels) * integration_time_s
    )
    sky_counts = sky_rate_hz * integration_time_s

    noise_variance = signal_counts + leakage_counts + dark_counts + sky_counts
    if null_instability > 0.0:
        if servo_bandwidth_hz > 0.0:
            noise_variance += (
                null_instability
                * null_instability
                * leakage_rate_hz
                * leakage_rate_hz
                * integration_time_s
                / (2.0 * servo_bandwidth_hz)
            )
        else:
            noise_variance += (null_instability * leakage_counts) ** 2

    if signal_counts <= 0.0 or noise_variance <= 0.0:
        return 0.0
    return signal_counts / math.sqrt(noise_variance)


def max_spectral_resolution(
    *,
    planet_rate_total_hz: float,
    leakage_rate_total_hz: float,
    spectro_throughput: float,
    band_lambda_m: float,
    band_bandwidth_m: float,
    integration_time_s: float,
    dark_rate_hz_per_pixel: float = 0.0,
    n_spectro_pixels: int = 1,
    null_instability: float = 0.0,
    servo_bandwidth_hz: float = 0.0,
    snr_min: float = 3.0,
    resolving_power_cap: float = 10_000.0,
    resolving_power_floor: float = 1.0,
    pixels_per_resolution_element: int = 2,
    sky_rate_hz_per_spectro_port: float = 0.0,
) -> float:
    """Return the largest resolving power with per-bin SNR >= ``snr_min``."""

    def excess(resolving_power: float) -> float:
        return per_bin_snr(
            planet_rate_total_hz=planet_rate_total_hz,
            leakage_rate_total_hz=leakage_rate_total_hz,
            spectro_throughput=spectro_throughput,
            band_lambda_m=band_lambda_m,
            band_bandwidth_m=band_bandwidth_m,
            resolving_power=resolving_power,
            integration_time_s=integration_time_s,
            dark_rate_hz_per_pixel=dark_rate_hz_per_pixel,
            n_spectro_pixels=n_spectro_pixels,
            pixels_per_resolution_element=pixels_per_resolution_element,
            null_instability=null_instability,
            servo_bandwidth_hz=servo_bandwidth_hz,
            sky_rate_hz_per_spectro_port=sky_rate_hz_per_spectro_port,
        ) - snr_min

    if resolving_power_cap <= 0.0 or resolving_power_floor <= 0.0:
        raise ValueError("resolving power bounds must be positive.")
    if resolving_power_cap < resolving_power_floor:
        raise ValueError("resolving_power_cap must be >= resolving_power_floor.")
    if snr_min <= 0.0:
        raise ValueError("snr_min must be positive.")
    if int(pixels_per_resolution_element) < 1:
        raise ValueError("pixels_per_resolution_element must be >= 1.")

    if excess(resolving_power_cap) >= 0.0:
        return float(resolving_power_cap)
    if excess(resolving_power_floor) <= 0.0:
        return 0.0
    return float(
        brentq(
            excess,
            resolving_power_floor,
            resolving_power_cap,
            xtol=1.0,
            rtol=1.0e-4,
        )
    )


def _validate_awg_optics(
    *,
    resolving_power: float,
    band_lambda_m: float,
    band_bandwidth_m: float,
    free_spectral_range_fraction: float,
    pixels_per_resolution_element: int,
) -> None:
    if resolving_power <= 0.0:
        raise ValueError("AWG resolving_power must be positive.")
    if band_lambda_m <= 0.0:
        raise ValueError("band_lambda_m must be positive.")
    if band_bandwidth_m <= 0.0:
        raise ValueError("band_bandwidth_m must be positive.")
    if not 0.0 < free_spectral_range_fraction <= 1.0:
        raise ValueError("free_spectral_range_fraction must be in (0, 1].")
    if int(pixels_per_resolution_element) < 1:
        raise ValueError("pixels_per_resolution_element must be >= 1.")


def _validate_awg_rates(
    *,
    planet_rate_total_hz: float,
    leakage_rate_total_hz: float,
    sky_rate_total_hz: float,
    dark_rate_hz_per_pixel: float,
) -> None:
    if planet_rate_total_hz < 0.0:
        raise ValueError("planet_rate_total_hz must be non-negative.")
    if leakage_rate_total_hz < 0.0:
        raise ValueError("leakage_rate_total_hz must be non-negative.")
    if sky_rate_total_hz < 0.0:
        raise ValueError("sky_rate_total_hz must be non-negative.")
    if dark_rate_hz_per_pixel < 0.0:
        raise ValueError("dark_rate_hz_per_pixel must be non-negative.")
