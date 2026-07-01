"""MKID energy-resolution response primitives."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr

__all__ = [
    "apply_energy_response",
    "energy_response_matrix",
    "energy_response_tail_loss",
    "mkid_resolving_power",
]

_FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def mkid_resolving_power(
    wavelengths_um: NDArray[np.float64] | tuple[float, ...] | list[float],
    r_ref: float,
    lambda_ref_um: float,
) -> NDArray[np.float64]:
    """Return constant-delta-E MKID resolving power on a wavelength grid.

    Args:
        wavelengths_um: Positive wavelength centers in microns.
        r_ref: Resolving power at ``lambda_ref_um``.
        lambda_ref_um: Reference wavelength in microns.

    Returns:
        Resolving power values ``R(lambda) = r_ref * lambda_ref_um / lambda``.

    Raises:
        ValueError: If inputs are non-physical.
    """

    wavelengths = _validate_wavelength_values(wavelengths_um, "wavelengths_um")
    _validate_positive_scalar(r_ref, "r_ref")
    _validate_positive_scalar(lambda_ref_um, "lambda_ref_um")
    return np.asarray(r_ref * lambda_ref_um / wavelengths, dtype=np.float64)


def energy_response_matrix(
    true_grid_um: NDArray[np.float64] | tuple[float, ...] | list[float],
    measured_grid_um: NDArray[np.float64] | tuple[float, ...] | list[float],
    r_ref: float,
    lambda_ref_um: float,
) -> NDArray[np.float64]:
    """Build the column-normalized true-to-measured MKID energy response.

    Columns correspond to true-wavelength bins and rows correspond to measured
    wavelength bins. Integration is performed in energy units proportional to
    ``1 / lambda``; the constant ``hc`` cancels because the response depends
    only on relative energy widths.

    The returned matrix deliberately renormalizes each true-bin column to the
    finite measured grid. Energy-response tails that would scatter outside the
    modeled band are therefore kept in band so detector-stage redistribution
    conserves per-port broadband counts. Use ``energy_response_tail_loss`` to
    inspect the pre-normalization off-grid probability when edge-bin loss is
    scientifically relevant.

    Raises:
        ValueError: If grids or parameters are non-physical.
    """

    raw_response = _raw_energy_response(
        true_grid_um,
        measured_grid_um,
        r_ref,
        lambda_ref_um,
    )
    column_sums = raw_response.sum(axis=0)
    if np.any(column_sums <= 0.0) or not np.all(np.isfinite(column_sums)):
        raise ValueError(
            "energy response has zero in-band probability for at least one "
            "true wavelength bin."
        )
    return np.asarray(raw_response / column_sums[np.newaxis, :], dtype=np.float64)


def energy_response_tail_loss(
    true_grid_um: NDArray[np.float64] | tuple[float, ...] | list[float],
    measured_grid_um: NDArray[np.float64] | tuple[float, ...] | list[float],
    r_ref: float,
    lambda_ref_um: float,
) -> NDArray[np.float64]:
    """Return pre-normalization probability outside the measured grid."""

    raw_response = _raw_energy_response(
        true_grid_um,
        measured_grid_um,
        r_ref,
        lambda_ref_um,
    )
    return np.asarray(np.clip(1.0 - raw_response.sum(axis=0), 0.0, 1.0))


def apply_energy_response(
    rates_hz: NDArray[np.float64],
    response: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply an MKID energy response across the last rate axis.

    Args:
        rates_hz: Rates with true-wavelength bins on the last axis.
        response: Response matrix with shape ``(n_measured, n_true)``.

    Returns:
        Rates with the last axis replaced by measured-wavelength bins.

    Raises:
        ValueError: If shapes are inconsistent.
    """

    rates = np.asarray(rates_hz, dtype=np.float64)
    matrix = np.asarray(response, dtype=np.float64)
    if rates.ndim < 1:
        raise ValueError("rates_hz must have at least one dimension.")
    if matrix.ndim != 2:
        raise ValueError("response must have shape (n_measured, n_true).")
    if rates.shape[-1] != matrix.shape[1]:
        raise ValueError(
            "rates_hz last axis must match response true-bin axis "
            f"({rates.shape[-1]} != {matrix.shape[1]})."
        )
    if not np.all(np.isfinite(rates)):
        raise ValueError("rates_hz must contain only finite values.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("response must contain only finite values.")
    return np.asarray(np.tensordot(rates, matrix.T, axes=([-1], [0])))


def _raw_energy_response(
    true_grid_um: NDArray[np.float64] | tuple[float, ...] | list[float],
    measured_grid_um: NDArray[np.float64] | tuple[float, ...] | list[float],
    r_ref: float,
    lambda_ref_um: float,
) -> NDArray[np.float64]:
    true_grid = _validate_wavelength_grid(true_grid_um, "true_grid_um", min_size=1)
    measured_grid = _validate_wavelength_grid(
        measured_grid_um,
        "measured_grid_um",
        min_size=2,
    )
    _validate_positive_scalar(r_ref, "r_ref")
    _validate_positive_scalar(lambda_ref_um, "lambda_ref_um")

    measured_edges_um = _wavelength_bin_edges_um(measured_grid)
    true_energy = 1.0 / true_grid
    measured_energy_edges = 1.0 / measured_edges_um
    energy_lower = np.minimum(measured_energy_edges[:-1], measured_energy_edges[1:])
    energy_upper = np.maximum(measured_energy_edges[:-1], measured_energy_edges[1:])

    sigma_energy = (1.0 / float(lambda_ref_um)) / float(r_ref) * _FWHM_TO_SIGMA
    z_upper = (energy_upper[:, np.newaxis] - true_energy[np.newaxis, :]) / sigma_energy
    z_lower = (energy_lower[:, np.newaxis] - true_energy[np.newaxis, :]) / sigma_energy
    response = ndtr(z_upper) - ndtr(z_lower)
    return np.asarray(response, dtype=np.float64)


def _wavelength_bin_edges_um(
    wavelengths_um: NDArray[np.float64],
) -> NDArray[np.float64]:
    if wavelengths_um.size < 2:
        raise ValueError("at least two measured wavelengths are required.")
    edges = np.empty(wavelengths_um.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (wavelengths_um[:-1] + wavelengths_um[1:])
    edges[0] = wavelengths_um[0] - 0.5 * (wavelengths_um[1] - wavelengths_um[0])
    edges[-1] = wavelengths_um[-1] + 0.5 * (wavelengths_um[-1] - wavelengths_um[-2])
    if edges[0] <= 0.0:
        raise ValueError("wavelength bin edges must be positive.")
    return edges


def _validate_wavelength_values(
    values: NDArray[np.float64] | tuple[float, ...] | list[float],
    name: str,
) -> NDArray[np.float64]:
    wavelengths = np.asarray(values, dtype=np.float64)
    if wavelengths.ndim != 1:
        raise ValueError(f"{name} must be a 1-D wavelength array.")
    if wavelengths.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(wavelengths)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(wavelengths <= 0.0):
        raise ValueError(f"{name} must contain only positive wavelengths.")
    return wavelengths


def _validate_wavelength_grid(
    values: NDArray[np.float64] | tuple[float, ...] | list[float],
    name: str,
    *,
    min_size: int,
) -> NDArray[np.float64]:
    wavelengths = _validate_wavelength_values(values, name)
    if wavelengths.size < min_size:
        raise ValueError(f"{name} must contain at least {min_size} wavelengths.")
    if wavelengths.size > 1 and np.any(np.diff(wavelengths) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    return wavelengths


def _validate_positive_scalar(value: float, name: str) -> None:
    scalar = float(value)
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
