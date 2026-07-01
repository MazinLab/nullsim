"""Pinning tests for future Fresnel propagation primitives.

KILO and NAYRA do not contain a Fresnel or angular-spectrum propagator to lift.
This test therefore pins the literature-derived nullsim contract documented in
tests/physics/notes/fresnel.md. It skips until nullsim.physics.fresnel exists.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_evanescent_suppressed_angular_spectrum_round_trips_low_na_gaussian() -> None:
    """Forward then backward propagation should recover a low-NA Gaussian."""
    fresnel = pytest.importorskip("nullsim.physics.fresnel")
    propagate_angular_spectrum = getattr(
        fresnel,
        "propagate_angular_spectrum",
        None,
    )
    if propagate_angular_spectrum is None:
        pytest.fail(
            "nullsim.physics.fresnel must expose propagate_angular_spectrum",
        )

    n_pixels = 128
    dx_m = 5.0e-6
    wavelength_m = 1.55e-6
    z_m = 2.0e-3
    waist_m = 90.0e-6

    coords_m = (np.arange(n_pixels, dtype=np.float64) - n_pixels / 2) * dx_m
    x_m, y_m = np.meshgrid(coords_m, coords_m, indexing="xy")
    input_field = np.exp(-((x_m**2 + y_m**2) / waist_m**2)).astype(np.complex128)

    forward_field = propagate_angular_spectrum(
        input_field,
        wavelength_m=wavelength_m,
        dx_m=dx_m,
        z_m=z_m,
    )
    round_trip_field = propagate_angular_spectrum(
        forward_field,
        wavelength_m=wavelength_m,
        dx_m=dx_m,
        z_m=-z_m,
    )

    assert forward_field.shape == input_field.shape
    assert round_trip_field.shape == input_field.shape
    assert forward_field.dtype == np.complex128
    assert round_trip_field.dtype == np.complex128

    input_power = float(np.sum(np.abs(input_field) ** 2))
    forward_power = float(np.sum(np.abs(forward_field) ** 2))

    assert forward_power == pytest.approx(input_power, rel=1.0e-12)
    np.testing.assert_allclose(
        round_trip_field,
        input_field,
        rtol=1.0e-8,
        atol=1.0e-10,
    )
