from __future__ import annotations

import math

import numpy as np
import pytest

from nullsim.physics.photometry import (
    estimate_r_mag_from_j,
    h_band_zero_point_flux_density,
    i_band_zero_point_flux_density,
    j_band_zero_point_flux_density,
    jmag_to_band_mag,
    k_band_zero_point_flux_density,
    y_band_zero_point_flux_density,
    magnitude_to_flux_density,
    photon_rate_per_m2_per_bin,
)


def test_j_band_zero_point_flux_density_pins_cohen_2003_si_value() -> None:
    """Cohen+ 2003 list 3.129e-13 W cm^-2 um^-1 = 3.129e-9 SI."""
    assert j_band_zero_point_flux_density() == pytest.approx(3.129e-9, rel=0.01)


def test_h_band_zero_point_pins_cohen_2003_si_value() -> None:
    """Phase 4.5: H-band Vega zero point (2MASS, Cohen+ 2003)."""

    assert h_band_zero_point_flux_density() == pytest.approx(1.133e-9, rel=0.01)


def test_k_band_zero_point_pins_cohen_2003_si_value() -> None:
    """Phase 4.5: Ks-band Vega zero point (2MASS, Cohen+ 2003)."""

    assert k_band_zero_point_flux_density() == pytest.approx(4.283e-10, rel=0.01)


def test_y_band_zero_point_pins_hewett_2006_si_value() -> None:
    """Y-band Vega zero point, Hewett+ 2006 Table 7: 5.71e-13 W cm^-2 um^-1.

    Independent check: Table 7's 2026 Jy converted with
    F_lambda = F_nu c / lambda_eff^2 at lambda_eff = 1.0305 um.
    """

    assert y_band_zero_point_flux_density() == pytest.approx(5.71e-9, rel=1e-12)
    f_lambda_from_jy = 2026.0e-26 * 2.99792458e8 / (1.0305e-6) ** 2 * 1.0e-6
    assert y_band_zero_point_flux_density() == pytest.approx(
        f_lambda_from_jy, rel=5e-3
    )


def test_i_band_zero_point_pins_bessell_1998_si_value() -> None:
    """HWO optical proxy: Cousins I Vega zero point near 0.806 um."""

    assert i_band_zero_point_flux_density() == pytest.approx(1.12e-8, rel=0.02)


@pytest.mark.parametrize(
    "band, wavelength_um",
    [("I", 0.806), ("Y", 1.020), ("J", 1.235), ("H", 1.662), ("K", 2.159)],
)
def test_supported_bands_return_finite_photon_rate(
    band: str, wavelength_um: float
) -> None:
    """Smoke test: jmag=8 in any supported band returns a finite rate."""

    rate = photon_rate_per_m2_per_bin(
        magnitude=8.0,
        wavelength_um=wavelength_um,
        bandwidth_um=0.16,
        band=band,
    )
    assert math.isfinite(rate)
    assert rate > 0.0


def test_magnitude_to_flux_density_scales_by_vega_magnitude_factor() -> None:
    zero_point_flux_density = j_band_zero_point_flux_density()

    assert magnitude_to_flux_density(0.0, zero_point_flux_density) == pytest.approx(
        zero_point_flux_density
    )
    assert magnitude_to_flux_density(5.0, zero_point_flux_density) == pytest.approx(
        0.01 * zero_point_flux_density
    )


def test_photon_rate_for_jmag_zero_matches_hand_traced_bin_conversion() -> None:
    wavelength_um = 1.235
    bandwidth_um = 0.162
    photon_energy_j = 6.62607015e-34 * 299_792_458.0 / (wavelength_um * 1.0e-6)
    expected_rate_hz_per_m2_per_bin = (
        3.129e-9 * bandwidth_um / photon_energy_j
    )

    assert photon_rate_per_m2_per_bin(
        0.0,
        wavelength_um,
        bandwidth_um,
    ) == pytest.approx(expected_rate_hz_per_m2_per_bin, rel=0.01)


def test_photon_rate_scales_by_five_magnitudes() -> None:
    rate_j0_hz_per_m2_per_bin = photon_rate_per_m2_per_bin(
        0.0,
        1.235,
        0.162,
    )
    rate_j5_hz_per_m2_per_bin = photon_rate_per_m2_per_bin(
        5.0,
        1.235,
        0.162,
    )

    assert rate_j5_hz_per_m2_per_bin == pytest.approx(
        0.01 * rate_j0_hz_per_m2_per_bin
    )


def test_estimate_r_mag_from_j_uses_spectral_type_color() -> None:
    assert estimate_r_mag_from_j(5.0, "F5V") == pytest.approx(5.4)
    assert estimate_r_mag_from_j(5.0, "M3V") == pytest.approx(7.0)


def test_jmag_to_band_mag_defaults_to_f_star_when_type_missing() -> None:
    assert jmag_to_band_mag(5.0, "R", None) == pytest.approx(5.4)


def test_jmag_to_i_mag_uses_spectral_type_color() -> None:
    assert jmag_to_band_mag(8.0, "I", "M3V") == pytest.approx(9.4)


def test_jmag8_j_band_bin_sanity() -> None:
    bin_width_um = 0.30 / 32.0
    rate_hz_per_m2_per_bin = photon_rate_per_m2_per_bin(
        8.0,
        1.65,
        bin_width_um,
    )
    full_band_rate_hz_per_m2 = photon_rate_per_m2_per_bin(8.0, 1.65, 0.30)
    placeholder_rate_hz_per_m2 = 1.0e10 * 10.0 ** (-0.4 * 8.0)

    assert rate_hz_per_m2_per_bin == pytest.approx(
        full_band_rate_hz_per_m2 / 32.0
    )
    assert full_band_rate_hz_per_m2 / placeholder_rate_hz_per_m2 == pytest.approx(
        1.0,
        rel=0.30,
    )
    assert rate_hz_per_m2_per_bin == pytest.approx(1.53739e5, rel=1.0e-4)


def test_negative_magnitude_is_valid_and_brighter_than_vega() -> None:
    vega_rate_hz_per_m2_per_bin = photon_rate_per_m2_per_bin(0.0, 1.235, 0.162)
    bright_rate_hz_per_m2_per_bin = photon_rate_per_m2_per_bin(-1.0, 1.235, 0.162)

    assert bright_rate_hz_per_m2_per_bin > vega_rate_hz_per_m2_per_bin


def test_nan_magnitude_propagates() -> None:
    """NaN magnitude propagates as NaN (catalog rows with missing photometry).

    Per the module-level NaN policy: non-finite magnitudes flow through the
    Pogson 10^(-0.4 m) computation rather than raising. NaN propagates as
    NaN; +inf gives 0; -inf gives +inf. Callers with catalog data must
    filter rows with missing photometry before interpreting results.
    """
    rate_hz_per_m2_per_bin = photon_rate_per_m2_per_bin(
        math.nan, 1.235, 0.162,
    )
    assert math.isnan(rate_hz_per_m2_per_bin)


def test_nan_magnitude_propagates_through_magnitude_to_flux_density() -> None:
    flux_density_w_m2_um = magnitude_to_flux_density(math.nan, 3.129e-9)
    assert math.isnan(flux_density_w_m2_um)


def test_plus_inf_magnitude_gives_zero_rate() -> None:
    """An infinitely faint source has zero photon rate."""
    rate_hz_per_m2_per_bin = photon_rate_per_m2_per_bin(
        math.inf, 1.235, 0.162,
    )
    assert rate_hz_per_m2_per_bin == 0.0


def test_minus_inf_magnitude_gives_inf_rate() -> None:
    """An infinitely bright source has +inf photon rate."""
    rate_hz_per_m2_per_bin = photon_rate_per_m2_per_bin(
        -math.inf, 1.235, 0.162,
    )
    assert math.isinf(rate_hz_per_m2_per_bin)
    assert rate_hz_per_m2_per_bin > 0.0


@pytest.mark.parametrize("bad_bandwidth_um", [0.0, -0.1, math.nan, math.inf])
def test_invalid_bandwidth_raises(bad_bandwidth_um: float) -> None:
    with pytest.raises(ValueError, match="bandwidth_um"):
        photon_rate_per_m2_per_bin(0.0, 1.235, bad_bandwidth_um)


@pytest.mark.parametrize("bad_wavelength_um", [0.0, -1.235, math.nan, math.inf])
def test_invalid_wavelength_raises(bad_wavelength_um: float) -> None:
    with pytest.raises(ValueError, match="wavelength_um"):
        photon_rate_per_m2_per_bin(0.0, bad_wavelength_um, 0.162)


def test_unsupported_band_raises() -> None:
    """Phase 4.5: Y/J/H/K are now supported; only nonsense band names raise."""

    with pytest.raises(ValueError, match="Unsupported photometric band"):
        photon_rate_per_m2_per_bin(0.0, 1.235, 0.162, band="L")


def test_numpy_scalar_dtype_inputs_return_plain_float() -> None:
    rate_hz_per_m2_per_bin = photon_rate_per_m2_per_bin(
        np.float32(5.0),
        np.float64(1.235),
        np.float32(0.162),
    )

    assert isinstance(rate_hz_per_m2_per_bin, float)
    assert rate_hz_per_m2_per_bin > 0.0
