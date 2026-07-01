"""Tests for the stellar-disk leakage integral."""

from __future__ import annotations


import numpy as np
import pytest

from nullsim.physics.stellar_disk import (
    effective_null_with_disk_floor,
    stellar_disk_null_floor,
)


_ARCSEC_PER_RAD = 206265.0
_MAS_PER_ARCSEC = 1000.0
_RAD_PER_MAS = 1.0 / (_ARCSEC_PER_RAD * _MAS_PER_ARCSEC)


def _r_grid_mas(n: int = 100, r_max_mas: float = 5.0) -> np.ndarray:
    """Linearly sampled separations in radians from 0 to r_max."""

    r_mas = np.linspace(0.0, r_max_mas, n + 1)[1:]  # drop r=0; helper anchors it
    return r_mas * _RAD_PER_MAS


def test_zero_diameter_returns_zero() -> None:
    r = _r_grid_mas(50, 1.0)
    t = 0.1 * np.ones_like(r)
    assert stellar_disk_null_floor(t, r, 0.0) == 0.0


def test_constant_throughput_returns_input_value() -> None:
    """If T(r) = T_0 everywhere, the disk integral collapses to T_0."""

    r = _r_grid_mas(200, 5.0)
    t = 0.123 * np.ones_like(r)
    disk_diameter_rad = 2.0 * _RAD_PER_MAS  # 2 mas diameter => rho_0=1 mas
    value = stellar_disk_null_floor(t, r, disk_diameter_rad)
    assert value == pytest.approx(0.123, rel=1e-10)


def test_quadratic_throughput_matches_analytic() -> None:
    """For T(r) = a r^2, the disk integral is a rho_0^2 / 2."""

    a = 4.2  # dimensionless rad^-2
    r = _r_grid_mas(500, 10.0)
    t = a * (r ** 2)
    rho_0_rad = 3.0 * _RAD_PER_MAS
    disk_diameter_rad = 2.0 * rho_0_rad
    # (2 / rho_0^2) * int_0^rho_0 r * (a r^2) dr = (2 / rho_0^2) * (a rho_0^4 / 4)
    #                                            = a rho_0^2 / 2
    expected = a * rho_0_rad * rho_0_rad / 2.0
    value = stellar_disk_null_floor(t, r, disk_diameter_rad)
    assert value == pytest.approx(expected, rel=5e-3)


def test_sixth_order_matches_analytic() -> None:
    """KILO kernel_order=3 has T(r) ~ r^6 near origin: integral = a rho_0^6 / 4."""

    a = 1.5e10  # rad^-6
    r = _r_grid_mas(1000, 5.0)
    t = a * (r ** 6)
    rho_0_rad = 0.5 * _RAD_PER_MAS  # 0.5 mas => 1.0 mas disk
    disk_diameter_rad = 2.0 * rho_0_rad
    # (2 / rho_0^2) * int_0^rho_0 r * a r^6 dr = (2 / rho_0^2) * a rho_0^8 / 8
    #                                          = a rho_0^6 / 4
    expected = a * (rho_0_rad ** 6) / 4.0
    value = stellar_disk_null_floor(t, r, disk_diameter_rad)
    assert value == pytest.approx(expected, rel=5e-3)


def test_disk_radius_beyond_max_separation_raises() -> None:
    r = _r_grid_mas(50, 1.0)
    t = 0.1 * np.ones_like(r)
    disk_diameter_rad = 4.0 * _RAD_PER_MAS  # rho_0 = 2 mas > 1 mas max
    with pytest.raises(ValueError, match="exceeds"):
        stellar_disk_null_floor(t, r, disk_diameter_rad)


def test_shape_mismatch_raises() -> None:
    r = _r_grid_mas(10, 1.0)
    t = np.zeros(5)
    with pytest.raises(ValueError, match="shape mismatch"):
        stellar_disk_null_floor(t, r, _RAD_PER_MAS)


def test_effective_null_sums_linearly() -> None:
    r = _r_grid_mas(200, 5.0)
    t = 0.05 * np.ones_like(r)
    disk_diameter_rad = 2.0 * _RAD_PER_MAS
    null_ao = 1.4e-4
    effective = effective_null_with_disk_floor(
        null_ao, t, r, disk_diameter_rad,
    )
    assert effective == pytest.approx(null_ao + 0.05, rel=1e-10)


def test_effective_null_handles_zero_diameter() -> None:
    """Point source => effective null is just the AO floor."""

    r = _r_grid_mas(50, 1.0)
    t = 0.5 * np.ones_like(r)
    null_ao = 7.28e-5
    effective = effective_null_with_disk_floor(null_ao, t, r, 0.0)
    assert effective == null_ao


def test_kilo_keck_j_band_disk_floor_scale() -> None:
    """Sanity check against KILO's J=4 cache: rho_0=0.5 mas, throughput ~1e-4 near origin."""

    # Reproduce a plausible kernel-order-3 throughput profile T(r) ~ r^6 near 0,
    # transitioning to broadband ~0.05 at r >> lambda/D ~ 2.7 mas.
    r = _r_grid_mas(500, 10.0)
    r_mas = r / _RAD_PER_MAS
    # KILO J=4 cache shows T ~ 4.4e-5 at 0.2 mas and ~0.05 at 5 mas.
    # Fit a r^6 / (1 + (r/r_break)^6) sigmoid.
    r_break_mas = 2.0
    t_floor = 0.05
    t = t_floor * (r_mas / r_break_mas) ** 6 / (1 + (r_mas / r_break_mas) ** 6)
    disk_diameter_rad = 1.0 * _RAD_PER_MAS
    value = stellar_disk_null_floor(t, r, disk_diameter_rad)
    # For a 1 mas disk with this T(r), the integral should be << t_floor (we're
    # well inside the kernel-suppressed r^6 region).
    assert 1e-12 < value < 1e-3, (
        f"disk floor {value:.3e} outside plausible range for kernel-3"
    )
