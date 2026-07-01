"""Tests for the quadrature-reference fringe-tracker mesh model."""

from __future__ import annotations

import numpy as np

from nullsim.physics.quadrature_fringe_tracker import (
    analytic_quadrature_crlb_piston_rms_nm,
    evaluate_quadrature_reference_observability,
    quadrature_reference_mesh,
)


def test_quadrature_mesh_is_lossless_isometry_for_physical_inputs() -> None:
    n_modes = 8
    beta = 0.2

    mesh = quadrature_reference_mesh(
        n_modes=n_modes,
        differential_coupling=beta,
    )
    isometry = np.vstack([mesh.tracker_matrix, mesh.science_matrix])

    np.testing.assert_allclose(
        isometry.conj().T @ isometry,
        np.eye(n_modes),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        mesh.tracker_matrix @ mesh.star_mode,
        mesh.star_mode,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        mesh.science_matrix @ mesh.star_mode,
        np.zeros(n_modes),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        mesh.tracker_matrix.conj().T @ mesh.tracker_matrix @ mesh.piston_basis,
        beta * mesh.piston_basis,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        mesh.science_matrix.conj().T @ mesh.science_matrix @ mesh.piston_basis,
        (1.0 - beta) * mesh.piston_basis,
        atol=1.0e-14,
    )


def test_quadrature_mesh_is_full_rank_with_one_spectral_channel() -> None:
    result = evaluate_quadrature_reference_observability(
        n_modes=8,
        wavelengths_m=np.array([1.25e-6]),
        stellar_counts_per_channel=1.0e7,
        differential_coupling=0.1,
    )

    assert result.rank == 7
    assert result.full_rank
    assert result.condition_number < 1.0 + 1.0e-10
    assert result.tracker_channel_count == 8
    assert result.science_channel_count == 8
    assert abs(result.tracker_star_fraction - 1.0) < 1.0e-14
    assert abs(result.science_star_leakage) < 1.0e-14
    assert abs(result.science_differential_throughput - 0.9) < 1.0e-14
    assert result.crlb_piston_rms_nm < 1.0


def test_quadrature_mesh_numerical_crlb_matches_closed_form() -> None:
    wavelengths = np.array([1.18e-6, 1.25e-6, 1.32e-6])
    counts = np.array([2.0e6, 2.5e6, 2.0e6])

    result = evaluate_quadrature_reference_observability(
        n_modes=10,
        wavelengths_m=wavelengths,
        stellar_counts_per_channel=counts,
        differential_coupling=0.03,
    )
    analytic = analytic_quadrature_crlb_piston_rms_nm(
        n_modes=10,
        wavelengths_m=wavelengths,
        stellar_counts_per_channel=counts,
        differential_coupling=0.03,
    )

    np.testing.assert_allclose(result.crlb_piston_rms_nm, analytic, rtol=1.0e-12)


def test_quadrature_mesh_crlb_improves_with_differential_coupling() -> None:
    low = evaluate_quadrature_reference_observability(
        n_modes=12,
        wavelengths_m=np.array([1.25e-6]),
        stellar_counts_per_channel=5.0e6,
        differential_coupling=0.01,
    )
    high = evaluate_quadrature_reference_observability(
        n_modes=12,
        wavelengths_m=np.array([1.25e-6]),
        stellar_counts_per_channel=5.0e6,
        differential_coupling=0.1,
    )

    assert high.crlb_piston_rms_nm < low.crlb_piston_rms_nm
    assert high.science_differential_throughput < low.science_differential_throughput
