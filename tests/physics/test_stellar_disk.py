"""Tests for the stellar-disk leakage integral."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import j0

from nullsim.physics.planet_throughput import compute_planet_throughput_vs_separation
from nullsim.physics.stellar_disk import (
    disk_radial_quadrature,
    effective_null_with_disk_floor,
    stellar_disk_null_floor,
)


_RAD_PER_MAS = math.pi / (180.0 * 3600.0 * 1000.0)


def test_disk_quadrature_weights_integrate_uniform_disk() -> None:
    """Weights sum to one and reproduce <r^2> = rho_0^2 / 2 on a disk."""

    rho_0 = 0.7 * _RAD_PER_MAS
    r, w = disk_radial_quadrature(2.0 * rho_0)
    assert r.size == 16
    assert np.all((r > 0.0) & (r < rho_0))
    assert np.sum(w) == pytest.approx(1.0, rel=1e-14)
    assert w @ r**2 == pytest.approx(rho_0**2 / 2.0, rel=1e-14)


def test_zero_diameter_returns_zero() -> None:
    assert stellar_disk_null_floor(lambda r: 0.1 + r, 0.0) == 0.0


def test_zero_diameter_keeps_smoothed_on_axis_excess() -> None:
    """A point star still leaks the difference between smoothed and bare T(0)."""

    value = stellar_disk_null_floor(
        lambda r: np.full(r.shape, 3e-6), 0.0, on_axis_throughput=1e-6,
    )
    assert value == pytest.approx(2e-6, rel=1e-14)


def test_constant_throughput_has_no_excess() -> None:
    """T(r) = T_0 everywhere: the on-axis null already holds all of it."""

    value = stellar_disk_null_floor(
        lambda r: np.full(r.shape, 0.123), 2.0 * _RAD_PER_MAS,
    )
    assert value == pytest.approx(0.0, abs=1e-15)


def test_quadratic_rolloff_matches_analytic_excess() -> None:
    """T(r) = T_0 + a r^2 gives a rho_0^2 / 2, independent of T_0."""

    a = 4.2e17  # rad^-2
    rho_0 = 3.0 * _RAD_PER_MAS
    value = stellar_disk_null_floor(lambda r: 7.2e-7 + a * r**2, 2.0 * rho_0)
    assert value == pytest.approx(a * rho_0**2 / 2.0, rel=1e-12)


def test_sixth_order_matches_analytic() -> None:
    """Kernel-order-3 rolloff T ~ a r^6: excess = a rho_0^6 / 4."""

    a = 1.5e10 / _RAD_PER_MAS**6
    rho_0 = 0.5 * _RAD_PER_MAS
    value = stellar_disk_null_floor(lambda r: a * r**6, 2.0 * rho_0)
    assert value == pytest.approx(a * rho_0**6 / 4.0, rel=1e-12)


def test_effective_null_does_not_double_count_on_axis_leakage() -> None:
    """Issue 17 regression: the on-axis T(0) is in the realized null already."""

    t_on_axis = 7.2e-7
    a = 1.0e17
    rho_0 = 0.5 * _RAD_PER_MAS
    realized_null = 7.12e-6  # contains t_on_axis
    effective = effective_null_with_disk_floor(
        realized_null, lambda r: t_on_axis + a * r**2, 2.0 * rho_0,
    )
    assert effective == pytest.approx(realized_null + a * rho_0**2 / 2.0, rel=1e-12)


def test_bracewell_disk_floor_matches_independent_quadrature() -> None:
    """Imbalanced Bracewell pair through the planet-throughput machinery.

    Amplitudes a1, a2 on a baseline B with a dark port (E1 - E2)/sqrt(2).
    The PA-averaged dark fraction is
    T(r) = (a1^2 + a2^2 - 2 a1 a2 J0(2 pi B r / lambda)) / (2 (a1^2 + a2^2)),
    so T(0) = (a1 - a2)^2 / (2 (a1^2 + a2^2)) != 0. The disk floor must be
    the disk average of T(r) - T(0), here by adaptive scipy quadrature.
    """

    baseline_m, wavelength_m = 85.0, 1.25e-6
    a1, a2 = 1.0, 0.9
    combiner = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / math.sqrt(2.0)

    def throughput(r_rad: np.ndarray) -> np.ndarray:
        return compute_planet_throughput_vs_separation(
            u_stack=combiner[:, :, np.newaxis],
            wavelengths_m=[wavelength_m],
            positions_m=np.array([[-0.5 * baseline_m, 0.0], [0.5 * baseline_m, 0.0]]),
            dark_indices=[1],
            coupling_per_sub=np.array([a1**2, a2**2]),
            sub_areas_m2=np.ones(2),
            sub_diameters_m=None,
            separations_rad=r_rad,
        )

    def analytic(r: float) -> float:
        z = 2.0 * math.pi * baseline_m * r / wavelength_m
        return (a1**2 + a2**2 - 2.0 * a1 * a2 * j0(z)) / (2.0 * (a1**2 + a2**2))

    rho_0 = 1.0 * _RAD_PER_MAS  # 2 pi B rho_0 / lambda = 2.07 rad
    direct, _ = quad(
        lambda r: 2.0 * r / rho_0**2 * (analytic(r) - analytic(0.0)), 0.0, rho_0,
        epsabs=0.0, epsrel=1e-13,
    )
    value = stellar_disk_null_floor(throughput, 2.0 * rho_0)
    assert analytic(0.0) > 1e-3
    assert value == pytest.approx(direct, rel=1e-9)
