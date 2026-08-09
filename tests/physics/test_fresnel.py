"""Tests for nullsim.physics.fresnel.

These tests cover the literature-derived angular-spectrum propagator contract:
analytical limits, stable round trips, band-limited evanescent suppression,
input validation, and the package-level Fresnel export.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nullsim.physics.fresnel import angular_spectrum_propagate


def test_zero_distance_returns_original_for_propagating_spectrum() -> None:
    rng = np.random.default_rng(12345)
    real_part = rng.normal(size=(32, 32)).astype(np.float32)
    imag_part = rng.normal(size=(32, 32)).astype(np.float32)
    field = real_part + 1j * imag_part

    propagated = angular_spectrum_propagate(
        field,
        distance_m=0.0,
        wavelength_um=1.55,
        pixel_pitch_um=5.0,
    )

    assert propagated.dtype == np.complex128
    np.testing.assert_allclose(propagated, field.astype(np.complex128), atol=1.0e-12)


def test_uniform_plane_wave_accumulates_global_phase() -> None:
    field = np.ones((32, 32), dtype=np.complex64)
    wavelength_um = 1.55
    distance_m = 2.0e-3
    expected_phase = np.exp(1j * (2.0 * np.pi / (wavelength_um * 1.0e-6)) * distance_m)

    # Uniform plane wave has zero spatial-frequency content, so the
    # angular-spectrum sampling limit (Code Review 2026-05-16 Issue 26)
    # is irrelevant here. Disable the check.
    propagated = angular_spectrum_propagate(
        field,
        distance_m=distance_m,
        wavelength_um=wavelength_um,
        pixel_pitch_um=5.0,
        check_sampling=False,
    )

    assert propagated.dtype == np.complex128
    np.testing.assert_allclose(propagated, expected_phase * field, rtol=1.0e-12)


def test_forward_backward_low_na_gaussian_round_trip() -> None:
    n_pixels = 128
    pixel_pitch_um = 5.0
    wavelength_um = 1.55
    distance_m = 2.0e-3
    waist_um = 90.0

    coords_um = (np.arange(n_pixels, dtype=np.float64) - n_pixels / 2) * pixel_pitch_um
    x_um, y_um = np.meshgrid(coords_um, coords_um, indexing="xy")
    input_field = np.exp(-((x_um**2 + y_um**2) / waist_um**2)).astype(np.complex128)

    forward_field = angular_spectrum_propagate(
        input_field,
        distance_m=distance_m,
        wavelength_um=wavelength_um,
        pixel_pitch_um=pixel_pitch_um,
    )
    round_trip_field = angular_spectrum_propagate(
        forward_field,
        distance_m=-distance_m,
        wavelength_um=wavelength_um,
        pixel_pitch_um=pixel_pitch_um,
    )

    input_power = float(np.sum(np.abs(input_field) ** 2))
    forward_power = float(np.sum(np.abs(forward_field) ** 2))

    assert forward_power == pytest.approx(input_power, rel=1.0e-12)
    np.testing.assert_allclose(round_trip_field, input_field, rtol=1.0e-8, atol=1.0e-10)


def test_suppress_evanescent_zeros_out_evanescent_checkerboard() -> None:
    indices = np.indices((16, 16)).sum(axis=0)
    field = np.where(indices % 2 == 0, 1.0, -1.0).astype(np.complex128)

    propagated = angular_spectrum_propagate(
        field,
        distance_m=1.0e-8,
        wavelength_um=1.0,
        pixel_pitch_um=0.1,
        suppress_evanescent=True,
    )

    assert np.max(np.abs(propagated)) < 1.0e-12


def test_evanescent_retained_when_suppression_disabled() -> None:
    indices = np.indices((16, 16)).sum(axis=0)
    field = np.where(indices % 2 == 0, 1.0, -1.0).astype(np.complex128)

    propagated = angular_spectrum_propagate(
        field,
        distance_m=1.0e-8,
        wavelength_um=1.0,
        pixel_pitch_um=0.1,
        suppress_evanescent=False,
    )

    peak_amplitude = float(np.max(np.abs(propagated)))
    assert 1.0e-6 < peak_amplitude < 1.0


@pytest.mark.parametrize(
    "bad_field",
    [
        np.ones(4),
        np.ones((4, 6)),
        np.ones((5, 5)),
        np.ones((0, 0)),
        np.array([[1.0, np.nan], [0.0, 1.0]]),
        np.array([[1.0, math.inf], [0.0, 1.0]]),
    ],
)
def test_invalid_field_shape_or_values_raise(bad_field: np.ndarray) -> None:
    with pytest.raises(ValueError, match="field"):
        angular_spectrum_propagate(
            bad_field,
            distance_m=0.0,
            wavelength_um=1.55,
            pixel_pitch_um=5.0,
        )


@pytest.mark.parametrize("bad_wavelength_um", [0.0, -1.0, math.inf, math.nan])
def test_invalid_wavelength_raises(bad_wavelength_um: float) -> None:
    with pytest.raises(ValueError, match="wavelength_um"):
        angular_spectrum_propagate(
            np.ones((4, 4)),
            distance_m=0.0,
            wavelength_um=bad_wavelength_um,
            pixel_pitch_um=5.0,
        )


@pytest.mark.parametrize("bad_pixel_pitch_um", [0.0, -1.0, math.inf, math.nan])
def test_invalid_pixel_pitch_raises(bad_pixel_pitch_um: float) -> None:
    with pytest.raises(ValueError, match="pixel_pitch_um"):
        angular_spectrum_propagate(
            np.ones((4, 4)),
            distance_m=0.0,
            wavelength_um=1.55,
            pixel_pitch_um=bad_pixel_pitch_um,
        )


@pytest.mark.parametrize("bad_distance_m", [math.inf, math.nan])
def test_invalid_distance_raises(bad_distance_m: float) -> None:
    with pytest.raises(ValueError, match="distance_m"):
        angular_spectrum_propagate(
            np.ones((4, 4)),
            distance_m=bad_distance_m,
            wavelength_um=1.55,
            pixel_pitch_um=5.0,
        )


def test_negative_distance_is_allowed() -> None:
    """Code Review 2026-05-16 Issue 33: pre-fix this test only asserted
    shape and dtype — a pure no-op (return input) would pass. Verify
    backward-then-forward propagation actually round-trips a known
    Gaussian field to high precision.
    """
    # Make a centered Gaussian so the angular-spectrum propagator stays
    # well inside the Nyquist sampling band at this distance.
    n = 32
    pitch_um = 5.0
    coords = (np.arange(n) - n / 2 + 0.5) * pitch_um * 1.0e-6
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    sigma_m = 20.0e-6
    gaussian = np.exp(-(xx**2 + yy**2) / (2.0 * sigma_m**2)).astype(np.complex128)

    distance_m = 1.0e-4  # 0.1 mm: low-NA regime, well-sampled
    forward = angular_spectrum_propagate(
        gaussian,
        distance_m=distance_m,
        wavelength_um=1.55,
        pixel_pitch_um=pitch_um,
    )
    round_trip = angular_spectrum_propagate(
        forward,
        distance_m=-distance_m,
        wavelength_um=1.55,
        pixel_pitch_um=pitch_um,
    )
    assert round_trip.shape == gaussian.shape
    assert round_trip.dtype == np.complex128
    np.testing.assert_allclose(round_trip, gaussian, atol=1.0e-10, rtol=1.0e-8)


def test_float32_and_complex64_inputs_return_complex128() -> None:
    real_field = np.ones((8, 8), dtype=np.float32)
    complex_field = (1.0 + 1.0j) * np.ones((8, 8), dtype=np.complex64)

    real_output = angular_spectrum_propagate(
        real_field,
        distance_m=1.0e-4,
        wavelength_um=1.55,
        pixel_pitch_um=5.0,
    )
    complex_output = angular_spectrum_propagate(
        complex_field,
        distance_m=1.0e-4,
        wavelength_um=1.55,
        pixel_pitch_um=5.0,
    )

    assert real_output.dtype == np.complex128
    assert complex_output.dtype == np.complex128
