"""Fresnel propagation primitives for scalar validation fields.

This module implements the angular-spectrum propagation kernel used for
near-field scalar wave-optics validation. It is pure math: no pipeline,
configuration, or state imports. The sampled field is a 2D complex amplitude
on a square, even grid; wavelengths and pixel pitches are supplied in
micrometers at the public API boundary.

References:
    SPEC §15.2.
    Goodman, J. W. Introduction to Fourier Optics (3rd ed., 2005), §4.2.
    Voelz, D. Computational Fourier Optics (SPIE 2011), §5.1.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def angular_spectrum_propagate(
    field: np.ndarray,
    distance_m: float,
    wavelength_um: float,
    pixel_pitch_um: float,
    *,
    suppress_evanescent: bool = True,
    check_sampling: bool = True,
) -> NDArray[np.complex128]:
    """Propagate a scalar field using the angular-spectrum method.

    The transfer function is ``H = exp(+i kz z)`` with
    ``kz = sqrt(k^2 - kx^2 - ky^2)`` for propagating components. Evanescent
    components use ``kz = i sqrt(kx^2 + ky^2 - k^2)`` when
    ``suppress_evanescent=False`` and are zeroed when
    ``suppress_evanescent=True`` (the default) to ensure physically causal
    propagation and to avoid exponentially growing terms during backward
    propagation.

    Code Review 2026-05-16 Issue 26: enforces the Voelz §5.1 sampling
    criterion ``|z| <= N · pitch² / λ`` by default. Warns at half that
    limit, raises beyond it. Callers who know their field has bounded
    spatial-frequency support (e.g. uniform plane waves) can pass
    ``check_sampling=False`` to skip the check.

    Args:
        field: Two-dimensional square complex-amplitude field. The side length
            must be even and all entries must be finite.
        distance_m: Propagation distance in meters. Negative values perform
            backward propagation; the value must be finite.
        wavelength_um: Vacuum wavelength in micrometers. Must be finite and
            positive.
        pixel_pitch_um: Square-grid sample pitch in micrometers. Must be
            finite and positive.
        suppress_evanescent: If True, zero spatial-frequency components with
            ``|k_perp| > k`` (evanescent waves). Defaults to True.
        check_sampling: If True (default), enforce the Nyquist sampling
            limit ``|z| <= N · pitch² / λ`` and raise on violation. Set
            to False when the field's spatial spectrum is known to be
            narrow enough that aliasing won't affect the propagated
            output.

    Returns:
        Propagated field with the same shape as ``field`` and dtype
        ``complex128``.

    Raises:
        ValueError: If the field shape, field values, or scalar physical inputs
            are invalid, OR if ``check_sampling`` is True and the
            propagation distance exceeds the angular-spectrum sampling
            limit.
    """
    import warnings

    complex_field = _as_valid_field(field)
    propagation_distance_m = _as_finite_real_scalar(distance_m, "distance_m")
    wavelength_m = _as_positive_um_as_m(wavelength_um, "wavelength_um")
    pixel_pitch_m = _as_positive_um_as_m(pixel_pitch_um, "pixel_pitch_um")

    n = complex_field.shape[0]
    if check_sampling:
        nyquist_distance_m = n * pixel_pitch_m * pixel_pitch_m / wavelength_m
        abs_distance_m = abs(propagation_distance_m)
        if abs_distance_m > nyquist_distance_m:
            raise ValueError(
                f"angular_spectrum_propagate: |distance|={abs_distance_m:.3e} m "
                f"exceeds the Voelz §5.1 sampling limit "
                f"(N * pitch² / λ = {nyquist_distance_m:.3e} m). The transfer "
                "function aliases above this distance; increase N, pitch, or "
                "use a Fresnel direct-integration propagator instead. "
                "Pass check_sampling=False if your field's spatial spectrum "
                "is narrow enough that aliasing is harmless."
            )
        if abs_distance_m > 0.5 * nyquist_distance_m:
            warnings.warn(
                f"angular_spectrum_propagate: |distance|={abs_distance_m:.3e} m "
                f"exceeds half the sampling limit "
                f"({0.5 * nyquist_distance_m:.3e} m); aliasing may be present "
                "in the propagated field.",
                RuntimeWarning,
                stacklevel=2,
            )

    wavenumber_rad_per_m = 2.0 * np.pi / wavelength_m
    transfer_function = _transfer_function(
        n,
        propagation_distance_m,
        wavenumber_rad_per_m,
        pixel_pitch_m,
        suppress_evanescent=suppress_evanescent,
    )

    spectrum = np.fft.fft2(complex_field)
    propagated = np.fft.ifft2(spectrum * transfer_function)
    return np.asarray(propagated, dtype=np.complex128)


def propagate_angular_spectrum(
    field: np.ndarray,
    *,
    wavelength_m: float,
    dx_m: float,
    z_m: float,
    suppress_evanescent: bool = True,
) -> NDArray[np.complex128]:
    """Compatibility wrapper for the WP5 pinning-test API.

    New code should call :func:`angular_spectrum_propagate`, whose public
    parameters use explicit micrometer suffixes for wavelength and pixel pitch.
    """
    return angular_spectrum_propagate(
        field,
        distance_m=z_m,
        wavelength_um=wavelength_m * 1.0e6,
        pixel_pitch_um=dx_m * 1.0e6,
        suppress_evanescent=suppress_evanescent,
    )


def _as_valid_field(field: np.ndarray) -> NDArray[np.complex128]:
    """Validate and coerce a scalar propagation field."""
    try:
        complex_field = np.asarray(field, dtype=np.complex128)
    except (TypeError, ValueError) as exc:
        raise ValueError("field must be numeric and convertible to complex128.") from exc

    if complex_field.ndim != 2:
        raise ValueError("field must be a two-dimensional array.")

    n_y, n_x = complex_field.shape
    if n_y == 0 or n_x == 0:
        raise ValueError("field must be non-empty.")
    if n_y != n_x:
        raise ValueError("field must be square.")
    if n_y % 2 != 0:
        raise ValueError("field side length must be even.")
    if not np.all(np.isfinite(complex_field)):
        raise ValueError("field must contain only finite values.")

    return complex_field


def _as_finite_real_scalar(value: float, name: str) -> np.float64:
    """Return ``value`` as a finite float64 scalar."""
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in ("i", "u", "f"):
        raise ValueError(f"{name} must be a finite real scalar.")

    scalar = np.float64(array)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite.")
    return scalar


def _as_positive_um_as_m(value_um: float, name: str) -> np.float64:
    """Validate a positive scalar in micrometers and return meters."""
    value = _as_finite_real_scalar(value_um, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return np.float64(value * 1.0e-6)


def _transfer_function(
    side_length: int,
    distance_m: np.float64,
    wavenumber_rad_per_m: np.float64,
    pixel_pitch_m: np.float64,
    *,
    suppress_evanescent: bool,
) -> NDArray[np.complex128]:
    """Build the angular-spectrum transfer function on NumPy FFT ordering."""
    spatial_frequencies_rad_per_m = (
        np.fft.fftfreq(side_length, d=pixel_pitch_m) * 2.0 * np.pi
    )
    kx_rad_per_m, ky_rad_per_m = np.meshgrid(
        spatial_frequencies_rad_per_m,
        spatial_frequencies_rad_per_m,
        indexing="xy",
    )
    transverse_wavenumber_squared = kx_rad_per_m**2 + ky_rad_per_m**2
    wavenumber_squared = wavenumber_rad_per_m * wavenumber_rad_per_m
    propagating = transverse_wavenumber_squared <= wavenumber_squared

    transfer = np.zeros((side_length, side_length), dtype=np.complex128)
    kz_propagating = np.sqrt(
        wavenumber_squared - transverse_wavenumber_squared[propagating],
        dtype=np.float64,
    )
    transfer[propagating] = np.exp(1j * kz_propagating * distance_m)

    if suppress_evanescent:
        return transfer

    evanescent = ~propagating
    kz_evanescent = 1j * np.sqrt(
        transverse_wavenumber_squared[evanescent] - wavenumber_squared,
        dtype=np.float64,
    )
    transfer[evanescent] = np.exp(1j * kz_evanescent * distance_m)
    return transfer


__all__ = [
    "angular_spectrum_propagate",
]
