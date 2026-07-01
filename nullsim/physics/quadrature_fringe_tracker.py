"""Quadrature-reference mesh model for integrated fringe tracking.

The model here answers a reduced architecture question: can one lossless
analysis mesh route the on-axis stellar reference into tracker outputs while
leaving science outputs dark, and still retain a tunable fraction of
differential-mode planet throughput?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nullsim.physics.fringe_observability import (
    cramer_rao_bound_from_svd,
    differential_piston_basis,
    intensity_response_matrix,
    scale_response_to_counts,
    svd_observability,
    whiten_response_matrix,
)

_TWO_PI = 2.0 * np.pi
_M_TO_NM = 1.0e9


@dataclass(frozen=True)
class QuadratureReferenceMesh:
    """Tracker/science isometry induced by a duplicated-input 2N mesh."""

    tracker_matrix: NDArray[np.complex128]
    science_matrix: NDArray[np.complex128]
    star_mode: NDArray[np.float64]
    piston_basis: NDArray[np.float64]
    differential_coupling: float


@dataclass(frozen=True)
class QuadratureReferenceObservability:
    """Small-signal observability of the quadrature-reference tracker."""

    rank: int
    full_rank: bool
    condition_number: float
    crlb_piston_rms_nm: float
    crlb_coordinate_rms_nm: float
    singular_values_per_m: NDArray[np.float64]
    tracker_channel_count: int
    science_channel_count: int
    tracker_star_fraction: float
    science_star_leakage: float
    science_differential_throughput: float
    tracker_differential_throughput: float
    expected_counts_per_tracker_channel: NDArray[np.float64]


def quadrature_reference_mesh(
    *,
    n_modes: int,
    differential_coupling: float,
) -> QuadratureReferenceMesh:
    """Build a 2N-output quadrature-reference fringe-tracker isometry.

    The tracker submatrix maps the on-axis stellar mode fully into ``N``
    tracker outputs, with a uniform reference amplitude at each tracker output.
    Differential piston modes are mixed into those same tracker outputs in
    quadrature with amplitude ``sqrt(differential_coupling)``. The science
    matrix is the lossless complement: it is exactly dark to the on-axis star
    and transmits ``1 - differential_coupling`` of every differential mode.

    Args:
        n_modes: Number of physical aperture/subaperture inputs.
        differential_coupling: Fraction of differential-mode power routed to
            the tracker outputs. The science complement keeps the remainder.

    Returns:
        Matrices with shape ``(n_modes, n_modes)`` for tracker and science
        outputs. Stacking them yields a ``(2*n_modes, n_modes)`` isometry that
        can be completed to a full ``2*n_modes`` unitary/Clements mesh.
    """

    if n_modes < 2:
        raise ValueError("n_modes must be at least 2.")
    if not 0.0 <= differential_coupling <= 1.0:
        raise ValueError("differential_coupling must be in [0, 1].")

    star = np.ones(n_modes, dtype=np.float64) / np.sqrt(float(n_modes))
    piston_basis = differential_piston_basis(n_modes)
    tracker_outputs_basis = differential_piston_basis(n_modes)
    tracker_matrix = np.outer(star, star).astype(np.complex128)
    tracker_matrix += (
        1j
        * np.sqrt(float(differential_coupling))
        * (tracker_outputs_basis @ piston_basis.T)
    )
    science_matrix = (
        np.sqrt(1.0 - float(differential_coupling))
        * (tracker_outputs_basis @ piston_basis.T)
    ).astype(np.complex128)
    return QuadratureReferenceMesh(
        tracker_matrix=tracker_matrix.astype(np.complex128, copy=False),
        science_matrix=science_matrix.astype(np.complex128, copy=False),
        star_mode=star,
        piston_basis=piston_basis,
        differential_coupling=float(differential_coupling),
    )


def evaluate_quadrature_reference_observability(
    *,
    n_modes: int,
    wavelengths_m: NDArray[np.floating],
    stellar_counts_per_channel: float | NDArray[np.floating],
    differential_coupling: float,
    dark_counts_per_channel: float = 0.0,
    relative_threshold: float = 1.0e-6,
    noise_floor_counts: float = 1.0,
) -> QuadratureReferenceObservability:
    """Evaluate Fisher/CRLB metrics for the quadrature-reference mesh.

    ``stellar_counts_per_channel`` is the total on-axis stellar count budget
    entering the photonic mesh in each independent tracker spectral channel.
    The quadrature-reference tracker routes that stellar reference uniformly
    over ``n_modes`` tracker outputs.
    """

    wavelengths = np.asarray(wavelengths_m, dtype=np.float64)
    if wavelengths.ndim != 1 or wavelengths.size == 0:
        raise ValueError("wavelengths_m must be a non-empty 1-D array.")
    if not np.all(np.isfinite(wavelengths)) or np.any(wavelengths <= 0.0):
        raise ValueError("wavelengths_m must contain finite positive values.")
    if dark_counts_per_channel < 0.0:
        raise ValueError("dark_counts_per_channel must be non-negative.")

    stellar_counts = _stellar_counts_array(
        stellar_counts_per_channel,
        n_wavelengths=int(wavelengths.size),
    )
    mesh = quadrature_reference_mesh(
        n_modes=n_modes,
        differential_coupling=differential_coupling,
    )
    star_fields = np.repeat(
        mesh.star_mode[:, np.newaxis, np.newaxis],
        wavelengths.size,
        axis=1,
    ).astype(np.complex128, copy=False)
    tracker_stack = np.repeat(
        mesh.tracker_matrix[:, :, np.newaxis],
        wavelengths.size,
        axis=2,
    )
    response = intensity_response_matrix(
        u_stack=tracker_stack,
        input_fields=star_fields,
        wavelengths_m=wavelengths,
        port_indices=np.arange(n_modes, dtype=np.intp),
    )
    row_channel_counts = stellar_counts[response.row_wavelength_indices]
    signal_counts = response.reference_power * row_channel_counts
    variance_counts = signal_counts + float(dark_counts_per_channel)
    projected_power = response.response_power_per_m @ mesh.piston_basis
    projected_counts = scale_response_to_counts(
        response_power_per_m=projected_power,
        reference_power=response.reference_power,
        expected_counts=signal_counts,
    )
    whitened = whiten_response_matrix(
        response_counts_per_m=projected_counts,
        variance_counts=variance_counts,
        noise_floor_counts=float(noise_floor_counts),
    )
    svd = svd_observability(
        whitened,
        relative_threshold=float(relative_threshold),
    )
    crlb = cramer_rao_bound_from_svd(svd, n_piston_modes=n_modes)
    science_gram = mesh.science_matrix.conj().T @ mesh.science_matrix
    tracker_gram = mesh.tracker_matrix.conj().T @ mesh.tracker_matrix
    star = mesh.star_mode
    science_star_leakage = float(np.real(star.conj().T @ science_gram @ star))
    tracker_star_fraction = float(np.real(star.conj().T @ tracker_gram @ star))

    return QuadratureReferenceObservability(
        rank=int(svd.rank),
        full_rank=bool(svd.full_rank),
        condition_number=float(svd.condition_number),
        crlb_piston_rms_nm=float(crlb.piston_rms_m * _M_TO_NM),
        crlb_coordinate_rms_nm=float(crlb.coordinate_rms_m * _M_TO_NM),
        singular_values_per_m=svd.singular_values.astype(np.float64, copy=True),
        tracker_channel_count=int(n_modes * wavelengths.size),
        science_channel_count=int(n_modes * wavelengths.size),
        tracker_star_fraction=tracker_star_fraction,
        science_star_leakage=science_star_leakage,
        science_differential_throughput=float(1.0 - differential_coupling),
        tracker_differential_throughput=float(differential_coupling),
        expected_counts_per_tracker_channel=signal_counts.astype(
            np.float64,
            copy=True,
        ),
    )


def analytic_quadrature_crlb_piston_rms_nm(
    *,
    n_modes: int,
    wavelengths_m: NDArray[np.floating],
    stellar_counts_per_channel: float | NDArray[np.floating],
    differential_coupling: float,
) -> float:
    """Return the shot-noise-only CRLB for the ideal quadrature mesh.

    This closed form is useful for checking numerical studies. With uniform
    stellar reference over the tracker outputs, the Fisher singular values add
    in quadrature over wavelength channels.
    """

    if n_modes < 2:
        raise ValueError("n_modes must be at least 2.")
    if not 0.0 < differential_coupling <= 1.0:
        raise ValueError("differential_coupling must be in (0, 1].")
    wavelengths = np.asarray(wavelengths_m, dtype=np.float64)
    counts = _stellar_counts_array(
        stellar_counts_per_channel,
        n_wavelengths=int(wavelengths.size),
    )
    if wavelengths.shape != counts.shape:
        raise ValueError("wavelengths_m and stellar_counts_per_channel mismatch.")
    singular_squared = np.sum(
        (2.0 * _TWO_PI / wavelengths) ** 2
        * float(differential_coupling)
        * counts
        / float(n_modes)
    )
    if singular_squared <= 0.0:
        return float("inf")
    coord_std = 1.0 / np.sqrt(float(singular_squared))
    piston_std = coord_std * np.sqrt(float(n_modes - 1) / float(n_modes))
    return float(piston_std * _M_TO_NM)


def _stellar_counts_array(
    value: float | NDArray[np.floating],
    *,
    n_wavelengths: int,
) -> NDArray[np.float64]:
    if n_wavelengths <= 0:
        raise ValueError("n_wavelengths must be positive.")
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(n_wavelengths, float(arr), dtype=np.float64)
    if arr.shape != (n_wavelengths,):
        raise ValueError(
            "stellar_counts_per_channel must be scalar or have shape "
            f"({n_wavelengths},)."
        )
    if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError("stellar_counts_per_channel must be finite and non-negative.")
    return arr.astype(np.float64, copy=False)


__all__ = [
    "QuadratureReferenceMesh",
    "QuadratureReferenceObservability",
    "analytic_quadrature_crlb_piston_rms_nm",
    "evaluate_quadrature_reference_observability",
    "quadrature_reference_mesh",
]
