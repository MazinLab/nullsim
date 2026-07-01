"""Tests for MKID energy-resolution physics primitives."""

from __future__ import annotations

import numpy as np
import pytest

from nullsim.physics.mkid import (
    apply_energy_response,
    energy_response_matrix,
    energy_response_tail_loss,
    mkid_resolving_power,
)


def test_mkid_resolving_power_constant_delta_e_scaling() -> None:
    wavelengths_um = np.array([0.5, 1.0, 2.0], dtype=np.float64)

    resolving_power = mkid_resolving_power(wavelengths_um, 30.0, 1.0)

    np.testing.assert_allclose(resolving_power, [60.0, 30.0, 15.0])


def test_energy_response_columns_conserve_counts_after_normalization() -> None:
    grid_um = np.linspace(0.9, 1.1, 11)

    response = energy_response_matrix(grid_um, grid_um, 30.0, 1.0)
    tail_loss = energy_response_tail_loss(grid_um, grid_um, 30.0, 1.0)

    np.testing.assert_allclose(response.sum(axis=0), np.ones(grid_um.size))
    assert response.shape == (grid_um.size, grid_um.size)
    assert tail_loss.shape == (grid_um.size,)
    assert np.all(tail_loss >= 0.0)
    assert tail_loss[0] > tail_loss[grid_um.size // 2]


def test_energy_response_high_resolving_power_approaches_identity() -> None:
    grid_um = np.array([0.9, 1.0, 1.1], dtype=np.float64)

    response = energy_response_matrix(grid_um, grid_um, 1.0e9, 1.0)

    np.testing.assert_allclose(response, np.eye(grid_um.size), atol=1.0e-12)


def test_energy_response_line_width_matches_fwhm_in_energy_space() -> None:
    measured_grid_um = np.linspace(0.9, 1.12, 1001)
    true_grid_um = np.array([1.0], dtype=np.float64)
    r_ref = 20.0

    response = energy_response_matrix(true_grid_um, measured_grid_um, r_ref, 1.0)
    profile = response[:, 0]
    energy_centers = 1.0 / measured_grid_um
    half_max = 0.5 * float(np.max(profile))
    above_half = energy_centers[profile >= half_max]
    measured_fwhm_energy = float(np.max(above_half) - np.min(above_half))

    assert measured_fwhm_energy == pytest.approx(1.0 / r_ref, rel=0.08)


def test_energy_response_preserves_energy_axis_ordering() -> None:
    measured_grid_um = np.array([0.9, 1.0, 1.1], dtype=np.float64)
    true_grid_um = np.array([0.91], dtype=np.float64)

    response = energy_response_matrix(true_grid_um, measured_grid_um, 1.0e4, 1.0)

    assert int(np.argmax(response[:, 0])) == 0


def test_apply_energy_response_uses_last_axis() -> None:
    rates_hz = np.array(
        [
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
        ],
        dtype=np.float64,
    )
    response = np.array(
        [
            [0.5, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    redistributed = apply_energy_response(rates_hz, response)

    np.testing.assert_allclose(
        redistributed,
        [
            [5.0, 5.0, 0.0],
            [0.0, 10.0, 0.0],
        ],
    )


def test_apply_energy_response_accepts_arbitrary_leading_axes() -> None:
    rates_hz = np.array(
        [
            [[10.0, 0.0], [0.0, 4.0]],
            [[2.0, 2.0], [6.0, 0.0]],
        ],
        dtype=np.float64,
    )
    response = np.array(
        [
            [0.25, 0.0],
            [0.75, 1.0],
        ],
        dtype=np.float64,
    )

    redistributed = apply_energy_response(rates_hz, response)

    assert redistributed.shape == rates_hz.shape
    np.testing.assert_allclose(redistributed.sum(axis=-1), rates_hz.sum(axis=-1))
    np.testing.assert_allclose(redistributed[0, 0], [2.5, 7.5])


@pytest.mark.parametrize(
    "true_grid_um, measured_grid_um, match",
    [
        ((1.0, 0.9), (0.9, 1.0), "true_grid_um"),
        ((1.0,), (1.0,), "measured_grid_um"),
        ((1.0,), (-0.9, 1.0), "positive"),
    ],
)
def test_energy_response_rejects_invalid_grids(
    true_grid_um: tuple[float, ...],
    measured_grid_um: tuple[float, ...],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        energy_response_matrix(true_grid_um, measured_grid_um, 30.0, 1.0)
