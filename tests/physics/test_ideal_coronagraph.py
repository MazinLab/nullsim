"""Pin the Guyon 2006 optimal-coronagraph bound primitives."""

from __future__ import annotations

import numpy as np
import pytest

from nullsim.physics.ideal_coronagraph import (
    MAS_TO_RAD,
    ideal_coronagraph_contrast_curve,
    planet_mode_throughput,
    uniform_disk_leakage,
)

WAVELENGTH_M = 0.8e-6
DIAMETER_M = 8.0
LAMBDA_OVER_D_RAD = WAVELENGTH_M / DIAMETER_M


def test_planet_throughput_zero_on_axis() -> None:
    assert planet_mode_throughput(0.0, WAVELENGTH_M, DIAMETER_M)[0] == 0.0


def test_planet_throughput_half_power_near_half_lambda_over_d() -> None:
    # Guyon 2006 headline: useful throughput reaches 50% near 0.5 lambda/D.
    separations = np.linspace(0.01, 1.5, 2000) * LAMBDA_OVER_D_RAD
    throughput = planet_mode_throughput(separations, WAVELENGTH_M, DIAMETER_M)
    half_power = separations[np.argmin(np.abs(throughput - 0.5))]
    assert 0.45 * LAMBDA_OVER_D_RAD < half_power < 0.60 * LAMBDA_OVER_D_RAD


def test_planet_throughput_near_unity_far_off_axis() -> None:
    throughput = planet_mode_throughput(
        10.0 * LAMBDA_OVER_D_RAD, WAVELENGTH_M, DIAMETER_M
    )
    assert throughput[0] > 0.99


def test_planet_throughput_small_angle_quadratic() -> None:
    # T(theta) ~ x^2 / 4 for x = pi D theta / lambda << 1.
    theta = 1.0e-3 * LAMBDA_OVER_D_RAD
    x = np.pi * DIAMETER_M * theta / WAVELENGTH_M
    throughput = planet_mode_throughput(theta, WAVELENGTH_M, DIAMETER_M)[0]
    assert throughput == pytest.approx(x**2 / 4.0, rel=1.0e-4)


def test_uniform_disk_leakage_small_disk_limit() -> None:
    # L -> (pi D rho / lambda)^2 / 8 for a small uniform disk.
    rho = 1.0e-2 * LAMBDA_OVER_D_RAD
    x_edge = np.pi * DIAMETER_M * rho / WAVELENGTH_M
    leakage = uniform_disk_leakage(rho, WAVELENGTH_M, DIAMETER_M)
    assert leakage == pytest.approx(x_edge**2 / 8.0, rel=1.0e-3)


def test_uniform_disk_leakage_zero_radius() -> None:
    assert uniform_disk_leakage(0.0, WAVELENGTH_M, DIAMETER_M) == 0.0


def test_contrast_curve_scales_with_sqrt_time() -> None:
    separations = np.array([50.0, 200.0])
    kwargs = dict(
        wavelengths_m=np.array([WAVELENGTH_M]),
        diameter_m=DIAMETER_M,
        stellar_radius_mas=0.93,
        stellar_rate_hz=1.0e9,
        background_rate_hz=10.0,
        snr_target=5.0,
    )
    one_day = ideal_coronagraph_contrast_curve(
        separations, integration_time_s=86400.0, **kwargs
    )
    four_days = ideal_coronagraph_contrast_curve(
        separations, integration_time_s=4.0 * 86400.0, **kwargs
    )
    np.testing.assert_allclose(four_days, one_day / 2.0, rtol=1.0e-12)


def test_contrast_curve_inner_separations_are_harder() -> None:
    separations = np.array([5.0, 20.6, 200.0])  # 0.24, 1, 9.7 lambda/D
    curve = ideal_coronagraph_contrast_curve(
        separations,
        wavelengths_m=np.array([WAVELENGTH_M]),
        diameter_m=DIAMETER_M,
        stellar_radius_mas=0.93,
        stellar_rate_hz=1.0e9,
        background_rate_hz=10.0,
        integration_time_s=86400.0,
    )
    assert curve[0] > curve[1] > curve[2]
    assert np.all(np.isfinite(curve))


def test_contrast_curve_on_axis_is_infinite() -> None:
    curve = ideal_coronagraph_contrast_curve(
        np.array([0.0]),
        wavelengths_m=np.array([WAVELENGTH_M]),
        diameter_m=DIAMETER_M,
        stellar_radius_mas=0.93,
        stellar_rate_hz=1.0e9,
        background_rate_hz=0.0,
        integration_time_s=86400.0,
    )
    assert np.isinf(curve[0])


def test_order4_throughput_small_angle_quartic() -> None:
    # T4(theta) ~ x^4 / 64 for x << 1.
    theta = 3.0e-2 * LAMBDA_OVER_D_RAD
    x = np.pi * DIAMETER_M * theta / WAVELENGTH_M
    throughput = planet_mode_throughput(
        theta, WAVELENGTH_M, DIAMETER_M, order=4
    )[0]
    assert throughput == pytest.approx(x**4 / 64.0, rel=1.0e-2)


def test_order4_half_power_near_one_lambda_over_d() -> None:
    # Guyon 2006: the 4th-order coronagraph trades IWA (~1 lambda/D) for
    # stellar-size resistance.
    separations = np.linspace(0.01, 3.0, 4000) * LAMBDA_OVER_D_RAD
    throughput = planet_mode_throughput(
        separations, WAVELENGTH_M, DIAMETER_M, order=4
    )
    half_power = separations[np.argmin(np.abs(throughput - 0.5))]
    assert 0.8 * LAMBDA_OVER_D_RAD < half_power < 1.3 * LAMBDA_OVER_D_RAD


def test_order4_disk_leakage_small_disk_limit() -> None:
    # L4 -> (pi D rho / lambda)^4 / 192 for a small uniform disk.
    rho = 3.0e-2 * LAMBDA_OVER_D_RAD
    x_edge = np.pi * DIAMETER_M * rho / WAVELENGTH_M
    leakage = uniform_disk_leakage(rho, WAVELENGTH_M, DIAMETER_M, order=4)
    assert leakage == pytest.approx(x_edge**4 / 192.0, rel=1.0e-2)


def test_invalid_order_raises() -> None:
    with pytest.raises(ValueError, match="order"):
        planet_mode_throughput(1.0e-7, WAVELENGTH_M, DIAMETER_M, order=3)


def test_solar_analog_at_5pc_is_stellar_size_walled_at_order_2() -> None:
    # 8 m HWO-class telescope, G2V at 5 pc (~1e10 cps through perfect
    # optics), 1.86 mas stellar diameter, 1 day at 200 mas (1 AU). The
    # strict Guyon order-2 optimal coronagraph leaks ~2.5e-3 of the
    # resolved stellar disk and cannot reach the 1e-10 Earth analog; the
    # order-4 version suppresses the disk by ~3 more decades and gets
    # within an order of magnitude in a single day. This is the Guyon 2006
    # argument for higher-order nulls on resolved stars.
    kwargs = dict(
        wavelengths_m=np.array([WAVELENGTH_M]),
        diameter_m=DIAMETER_M,
        stellar_radius_mas=0.93,
        stellar_rate_hz=1.0e10,
        background_rate_hz=100.0,
        integration_time_s=86400.0,
    )
    order2 = ideal_coronagraph_contrast_curve(np.array([200.0]), **kwargs)
    order4 = ideal_coronagraph_contrast_curve(
        np.array([200.0]), order=4, **kwargs
    )
    sep_rad = 200.0 * MAS_TO_RAD
    assert sep_rad > 5 * LAMBDA_OVER_D_RAD  # planet throughput ~ 1 there
    assert 1.0e-9 < order2[0] < 3.0e-8
    assert order4[0] < order2[0] / 10.0
    assert order4[0] < 1.0e-9
