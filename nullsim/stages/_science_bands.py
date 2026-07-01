"""Shared science-band photon-budget helpers for simultaneous modes."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from nullsim.physics.photometry import photon_rate_per_m2_per_bin


BANDPASS_UM: dict[str, tuple[float, float]] = {
    "I": (0.806, 0.15),
    "Y": (1.02, 0.12),
    "J": (1.25, 0.16),
    "H": (1.65, 0.30),
    "K": (2.20, 0.40),
}


def infer_band_from_wavelength_um(wavelength_um: float) -> str:
    """Return the nearest supported science band for a wavelength."""

    wavelength = float(wavelength_um)
    if not math.isfinite(wavelength) or wavelength <= 0.0:
        raise ValueError(f"wavelength_um must be finite and positive, got {wavelength!r}.")
    return min(
        BANDPASS_UM,
        key=lambda band: abs(BANDPASS_UM[band][0] - wavelength),
    )


def simultaneous_band_rate_scale(
    *,
    magnitude: float,
    current_wavelength_grid_um: Sequence[float],
    science_bands: Sequence[str],
    band_transmissions: Sequence[float],
    stage_name: str,
) -> tuple[float, tuple[str, ...], str]:
    """Scale a single-band photon budget to a simultaneous-band budget.

    The current run carries one wavelength grid, but instruments such as
    NAYRA can feed a servo with photons from several science bands at once.
    This helper returns the multiplicative rate scale:

        sum(rate_per_m2(band) * transmission_band)
        / (current_grid_rate_per_m2 * current_band_transmission)

    The current-grid denominator uses the inferred photometric band for
    its zero point and the explicit bin widths from the resolved grid.
    When no transmissions are supplied, all transmissions are treated as
    unity.
    """

    normalized_bands = _normalize_bands(science_bands, stage_name)
    if not normalized_bands:
        return 1.0, (), _current_band(current_wavelength_grid_um, stage_name)

    transmissions = _normalize_transmissions(
        normalized_bands,
        band_transmissions,
        stage_name,
    )
    current_band = _current_band(current_wavelength_grid_um, stage_name)
    transmission_by_band = dict(
        zip(normalized_bands, transmissions, strict=True)
    )
    current_rate = grid_photon_rate_per_m2(
        magnitude=magnitude,
        wavelength_grid_um=current_wavelength_grid_um,
        band=current_band,
        stage_name=stage_name,
    ) * transmission_by_band.get(current_band, 1.0)
    if current_rate <= 0.0:
        raise ValueError(
            f"{stage_name}: current-grid photon rate is {current_rate}; "
            "cannot scale to simultaneous science bands."
        )

    total_rate = 0.0
    for band, transmission in zip(normalized_bands, transmissions, strict=True):
        center_um, bandwidth_um = BANDPASS_UM[band]
        total_rate += (
            photon_rate_per_m2_per_bin(
                magnitude,
                center_um,
                bandwidth_um,
                band=band,
            )
            * transmission
        )
    return float(total_rate / current_rate), normalized_bands, current_band


def grid_photon_rate_per_m2(
    *,
    magnitude: float,
    wavelength_grid_um: Sequence[float],
    band: str,
    stage_name: str,
) -> float:
    """Return rectangular-bin photon rate per m2 for an existing grid."""

    wl = np.asarray(tuple(float(value) for value in wavelength_grid_um), dtype=np.float64)
    if wl.size == 0:
        raise ValueError(f"{stage_name}: wavelength grid is empty.")
    if not np.all(np.isfinite(wl)) or np.any(wl <= 0.0):
        raise ValueError(f"{stage_name}: wavelength grid must be finite and positive.")
    if wl.size == 1:
        bandwidth_um = np.array([wl[0]], dtype=np.float64)
    else:
        edges = np.empty(wl.size + 1, dtype=np.float64)
        edges[1:-1] = 0.5 * (wl[:-1] + wl[1:])
        edges[0] = wl[0] - 0.5 * (wl[1] - wl[0])
        edges[-1] = wl[-1] + 0.5 * (wl[-1] - wl[-2])
        bandwidth_um = np.diff(edges)
    return float(
        np.sum(
            [
                photon_rate_per_m2_per_bin(
                    magnitude,
                    float(wavelength),
                    float(width),
                    band=band,
                )
                for wavelength, width in zip(wl, bandwidth_um, strict=True)
            ]
        )
    )


def _current_band(wavelength_grid_um: Sequence[float], stage_name: str) -> str:
    values = tuple(float(value) for value in wavelength_grid_um)
    if not values:
        raise ValueError(f"{stage_name}: wavelength grid is empty.")
    return infer_band_from_wavelength_um(values[len(values) // 2])


def _normalize_bands(science_bands: Sequence[Any], stage_name: str) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in science_bands:
        band = str(value).upper()
        if band not in BANDPASS_UM:
            supported = ", ".join(sorted(BANDPASS_UM))
            raise ValueError(
                f"{stage_name}: unsupported simultaneous science band {value!r}; "
                f"supported bands are {supported}."
            )
        normalized.append(band)
    return tuple(normalized)


def _normalize_transmissions(
    science_bands: tuple[str, ...],
    band_transmissions: Sequence[float],
    stage_name: str,
) -> tuple[float, ...]:
    if len(band_transmissions) == 0:
        return tuple(1.0 for _ in science_bands)
    if len(band_transmissions) != len(science_bands):
        raise ValueError(
            f"{stage_name}: simultaneous_science_band_transmissions length "
            f"({len(band_transmissions)}) must match simultaneous_science_bands "
            f"length ({len(science_bands)})."
        )
    out = tuple(float(value) for value in band_transmissions)
    if any((not math.isfinite(value)) or value <= 0.0 for value in out):
        raise ValueError(
            f"{stage_name}: simultaneous science-band transmissions must be "
            "finite and positive."
        )
    return out


__all__ = [
    "BANDPASS_UM",
    "grid_photon_rate_per_m2",
    "infer_band_from_wavelength_um",
    "simultaneous_band_rate_scale",
]
