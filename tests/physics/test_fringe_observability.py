"""Tests for fringe-tracker piston observability primitives."""

from __future__ import annotations

import numpy as np
import pytest

from nullsim.physics.fringe_observability import (
    active_coordinate_acquisition_monte_carlo,
    amplitude_postprocessing_monte_carlo,
    balanced_binary_tree_edges,
    cophasing_handoff_monte_carlo,
    cramer_rao_bound_from_svd,
    differential_piston_basis,
    fractional_intensity_response_matrix,
    gauss_newton_reconstruction_monte_carlo,
    intensity_response_matrix,
    intensity_response_matrix_at_pistons,
    linearized_reconstruction_monte_carlo,
    piston_measurement_covariance_from_svd,
    reconstruction_monte_carlo,
    scale_response_to_counts,
    spectral_channel_count_response_matrix,
    spectral_channel_fractional_intensity_response_matrix,
    svd_observability,
    weakest_piston_modes,
    whiten_response_matrix,
)
from nullsim.physics.quadrature_fringe_tracker import quadrature_reference_mesh


def _random_unitary(n_modes: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n_modes, n_modes)) + 1j * rng.normal(size=(n_modes, n_modes))
    q, r = np.linalg.qr(z)
    phase = np.diag(r) / np.abs(np.diag(r))
    return (q * phase[np.newaxis, :]).astype(np.complex128)


def _joint_piston_amplitude_response(
    *,
    u_stack: np.ndarray,
    input_fields: np.ndarray,
    wavelengths_m: np.ndarray,
    port_indices: np.ndarray,
    wavelength_groups: tuple[np.ndarray, ...],
    expected_counts_by_wavelength: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    piston = spectral_channel_count_response_matrix(
        u_stack=u_stack,
        input_fields=input_fields,
        wavelengths_m=wavelengths_m,
        port_indices=port_indices,
        wavelength_groups=wavelength_groups,
        expected_counts_by_wavelength=expected_counts_by_wavelength,
    )
    amplitude = spectral_channel_fractional_intensity_response_matrix(
        u_stack=u_stack,
        input_fields=input_fields,
        wavelengths_m=wavelengths_m,
        port_indices=port_indices,
        wavelength_groups=wavelength_groups,
        expected_counts_by_wavelength=expected_counts_by_wavelength,
    )
    joint = np.hstack(
        [
            piston.response_counts_per_m @ basis * 1.0e-9,
            amplitude.response_counts_per_fractional_intensity @ basis,
        ]
    )
    return whiten_response_matrix(
        response_counts_per_m=joint,
        expected_counts=piston.expected_counts,
    )


def test_differential_piston_basis_removes_common_piston() -> None:
    basis = differential_piston_basis(6)

    assert basis.shape == (6, 5)
    np.testing.assert_allclose(basis.T @ basis, np.eye(5), atol=1.0e-14)
    np.testing.assert_allclose(np.ones(6) @ basis, np.zeros(5), atol=1.0e-14)


def test_balanced_binary_tree_edges_groups_halves_before_bridge() -> None:
    edges = balanced_binary_tree_edges(12)

    np.testing.assert_array_equal(
        edges,
        np.array(
            [
                [0, 1],
                [1, 2],
                [0, 3],
                [3, 4],
                [4, 5],
                [0, 6],
                [6, 7],
                [7, 8],
                [6, 9],
                [9, 10],
                [10, 11],
            ],
            dtype=np.int64,
        ),
    )


def test_intensity_response_matches_finite_difference() -> None:
    n_modes = 4
    n_wl = 3
    rng = np.random.default_rng(12)
    u = np.stack([_random_unitary(n_modes, seed=i) for i in range(n_wl)], axis=2)
    fields = (
        rng.normal(size=(n_modes, n_wl, 1)) + 1j * rng.normal(size=(n_modes, n_wl, 1))
    ).astype(np.complex128)
    wavelengths = np.array([1.20e-6, 1.25e-6, 1.30e-6])
    ports = np.array([0, 2], dtype=np.intp)
    waves = np.array([0, 2], dtype=np.intp)

    response = intensity_response_matrix(
        u_stack=u,
        input_fields=fields,
        wavelengths_m=wavelengths,
        port_indices=ports,
        wavelength_indices=waves,
    )

    eps_m = 1.0e-12
    finite = np.zeros_like(response.response_power_per_m)
    for mode in range(n_modes):
        plus = fields.copy()
        minus = fields.copy()
        for wave in range(n_wl):
            phase = np.exp(1j * 2.0 * np.pi * eps_m / wavelengths[wave])
            plus[mode, wave, :] *= phase
            minus[mode, wave, :] *= np.conj(phase)
        plus_resp = intensity_response_matrix(
            u_stack=u,
            input_fields=plus,
            wavelengths_m=wavelengths,
            port_indices=ports,
            wavelength_indices=waves,
        )
        minus_resp = intensity_response_matrix(
            u_stack=u,
            input_fields=minus,
            wavelengths_m=wavelengths,
            port_indices=ports,
            wavelength_indices=waves,
        )
        finite[:, mode] = (plus_resp.reference_power - minus_resp.reference_power) / (
            2.0 * eps_m
        )

    np.testing.assert_allclose(response.response_power_per_m, finite, rtol=1.0e-5)


def test_fractional_intensity_response_matches_finite_difference() -> None:
    n_modes = 4
    n_wl = 3
    rng = np.random.default_rng(22)
    u = np.stack([_random_unitary(n_modes, seed=i) for i in range(n_wl)], axis=2)
    fields = (
        rng.normal(size=(n_modes, n_wl, 1)) + 1j * rng.normal(size=(n_modes, n_wl, 1))
    ).astype(np.complex128)
    ports = np.array([0, 2], dtype=np.intp)
    waves = np.array([0, 2], dtype=np.intp)

    response = fractional_intensity_response_matrix(
        u_stack=u,
        input_fields=fields,
        port_indices=ports,
        wavelength_indices=waves,
    )

    eps = 1.0e-6
    finite = np.zeros_like(response.response_power_per_fractional_intensity)
    for mode in range(n_modes):
        plus = fields.copy()
        minus = fields.copy()
        plus[mode, :, :] *= np.sqrt(1.0 + eps)
        minus[mode, :, :] *= np.sqrt(1.0 - eps)
        plus_resp = fractional_intensity_response_matrix(
            u_stack=u,
            input_fields=plus,
            port_indices=ports,
            wavelength_indices=waves,
        )
        minus_resp = fractional_intensity_response_matrix(
            u_stack=u,
            input_fields=minus,
            port_indices=ports,
            wavelength_indices=waves,
        )
        finite[:, mode] = (plus_resp.reference_power - minus_resp.reference_power) / (
            2.0 * eps
        )

    np.testing.assert_allclose(
        response.response_power_per_fractional_intensity,
        finite,
        rtol=1.0e-6,
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        response.response_power_per_fractional_intensity @ np.ones(n_modes),
        response.reference_power,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_amplitude_postprocessing_compares_bright_ports_to_ideal_taps() -> None:
    basis = differential_piston_basis(4)
    sigma_i = 0.02
    weak_response = np.zeros((4, basis.shape[1]), dtype=np.float64)

    weak = amplitude_postprocessing_monte_carlo(
        whitened_response_per_fractional_intensity=weak_response,
        amplitude_basis=basis,
        scintillation_sigma_i=sigma_i,
        ideal_tap_counts_per_mode=1.0e6,
        n_trials=128,
        rng=np.random.default_rng(10),
    )

    assert weak.median_bright_wiener_residual_rms_fractional_intensity == (
        pytest.approx(weak.median_input_rms_fractional_intensity)
    )
    assert weak.median_ideal_tap_residual_rms_fractional_intensity < 0.002

    strong_response = np.eye(basis.shape[1], dtype=np.float64) * 1.0e3
    strong = amplitude_postprocessing_monte_carlo(
        whitened_response_per_fractional_intensity=strong_response,
        amplitude_basis=basis,
        scintillation_sigma_i=sigma_i,
        ideal_tap_counts_per_mode=1.0e6,
        n_trials=128,
        rng=np.random.default_rng(11),
    )

    assert strong.median_bright_wiener_residual_rms_fractional_intensity < 0.002


def test_intensity_response_at_pistons_matches_finite_difference() -> None:
    n_modes = 4
    n_wl = 2
    rng = np.random.default_rng(13)
    u = np.stack([_random_unitary(n_modes, seed=i) for i in range(n_wl)], axis=2)
    fields = (
        rng.normal(size=(n_modes, n_wl, 1)) + 1j * rng.normal(size=(n_modes, n_wl, 1))
    ).astype(np.complex128)
    wavelengths = np.array([1.15e-6, 1.35e-6])
    ports = np.array([0, 1, 3], dtype=np.intp)
    pistons = np.array([20.0, -35.0, 12.0, 3.0], dtype=np.float64) * 1.0e-9

    response = intensity_response_matrix_at_pistons(
        u_stack=u,
        input_fields=fields,
        wavelengths_m=wavelengths,
        port_indices=ports,
        pistons_m=pistons,
    )

    eps_m = 1.0e-12
    finite = np.zeros_like(response.response_power_per_m)
    for mode in range(n_modes):
        plus = pistons.copy()
        minus = pistons.copy()
        plus[mode] += eps_m
        minus[mode] -= eps_m
        plus_resp = intensity_response_matrix_at_pistons(
            u_stack=u,
            input_fields=fields,
            wavelengths_m=wavelengths,
            port_indices=ports,
            pistons_m=plus,
        )
        minus_resp = intensity_response_matrix_at_pistons(
            u_stack=u,
            input_fields=fields,
            wavelengths_m=wavelengths,
            port_indices=ports,
            pistons_m=minus,
        )
        finite[:, mode] = (plus_resp.reference_power - minus_resp.reference_power) / (
            2.0 * eps_m
        )

    np.testing.assert_allclose(response.response_power_per_m, finite, rtol=1.0e-5)


def test_common_piston_has_zero_intensity_response() -> None:
    n_modes = 5
    n_wl = 2
    rng = np.random.default_rng(7)
    u = np.stack([_random_unitary(n_modes, seed=i) for i in range(n_wl)], axis=2)
    fields = rng.normal(size=(n_modes, n_wl, 1)) + 1j * rng.normal(
        size=(n_modes, n_wl, 1)
    )
    response = intensity_response_matrix(
        u_stack=u,
        input_fields=fields,
        wavelengths_m=np.array([1.0e-6, 1.2e-6]),
        port_indices=np.arange(n_modes),
    )

    np.testing.assert_allclose(
        response.response_power_per_m @ np.ones(n_modes),
        np.zeros(n_modes * n_wl),
        atol=1.0e-7,
    )


def test_spectral_channel_response_sums_wavelength_counts() -> None:
    n_modes = 3
    n_wl = 4
    rng = np.random.default_rng(18)
    u = np.stack(
        [_random_unitary(n_modes, seed=30 + i) for i in range(n_wl)],
        axis=2,
    )
    fields = (
        rng.normal(size=(n_modes, n_wl, 1)) + 1j * rng.normal(size=(n_modes, n_wl, 1))
    ).astype(np.complex128)
    wavelengths = np.linspace(1.0e-6, 1.3e-6, n_wl)
    ports = np.array([0, 2], dtype=np.intp)
    groups = (np.array([0, 1], dtype=np.intp), np.array([2, 3], dtype=np.intp))
    counts_by_wave = np.array(
        [[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]],
        dtype=np.float64,
    )

    grouped = spectral_channel_count_response_matrix(
        u_stack=u,
        input_fields=fields,
        wavelengths_m=wavelengths,
        port_indices=ports,
        wavelength_groups=groups,
        expected_counts_by_wavelength=counts_by_wave,
    )
    fine = intensity_response_matrix(
        u_stack=u,
        input_fields=fields,
        wavelengths_m=wavelengths,
        port_indices=ports,
    )
    fine_counts = scale_response_to_counts(
        response_power_per_m=fine.response_power_per_m,
        reference_power=fine.reference_power,
        expected_counts=counts_by_wave.T.reshape(-1),
    )

    expected = np.vstack(
        [
            fine_counts[0:2] + fine_counts[2:4],
            fine_counts[4:6] + fine_counts[6:8],
        ]
    )
    np.testing.assert_allclose(grouped.response_counts_per_m, expected)
    np.testing.assert_allclose(grouped.expected_counts, [30.0, 110.0, 70.0, 150.0])
    np.testing.assert_array_equal(grouped.row_channel_indices, [0, 0, 1, 1])


def test_spectral_channel_fractional_intensity_response_sums_counts() -> None:
    n_modes = 3
    n_wl = 4
    rng = np.random.default_rng(23)
    u = np.stack(
        [_random_unitary(n_modes, seed=40 + i) for i in range(n_wl)],
        axis=2,
    )
    fields = (
        rng.normal(size=(n_modes, n_wl, 1)) + 1j * rng.normal(size=(n_modes, n_wl, 1))
    ).astype(np.complex128)
    wavelengths = np.linspace(1.0e-6, 1.3e-6, n_wl)
    ports = np.array([0, 2], dtype=np.intp)
    groups = (np.array([0, 1], dtype=np.intp), np.array([2, 3], dtype=np.intp))
    counts_by_wave = np.array(
        [[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]],
        dtype=np.float64,
    )

    grouped = spectral_channel_fractional_intensity_response_matrix(
        u_stack=u,
        input_fields=fields,
        wavelengths_m=wavelengths,
        port_indices=ports,
        wavelength_groups=groups,
        expected_counts_by_wavelength=counts_by_wave,
    )
    fine = fractional_intensity_response_matrix(
        u_stack=u,
        input_fields=fields,
        port_indices=ports,
    )
    fine_counts = scale_response_to_counts(
        response_power_per_m=fine.response_power_per_fractional_intensity,
        reference_power=fine.reference_power,
        expected_counts=counts_by_wave.T.reshape(-1),
    )

    expected = np.vstack(
        [
            fine_counts[0:2] + fine_counts[2:4],
            fine_counts[4:6] + fine_counts[6:8],
        ]
    )
    np.testing.assert_allclose(
        grouped.response_counts_per_fractional_intensity,
        expected,
    )
    np.testing.assert_allclose(grouped.expected_counts, [30.0, 110.0, 70.0, 150.0])
    np.testing.assert_array_equal(grouped.row_channel_indices, [0, 0, 1, 1])


def test_bright_port_spectral_channels_can_lift_joint_phase_amplitude_rank() -> None:
    n_modes = 3
    n_wl = 2
    rng = np.random.default_rng(101)
    u = np.stack(
        [_random_unitary(n_modes, seed=1 + 10 * wave) for wave in range(n_wl)],
        axis=2,
    )
    fields = (
        rng.normal(size=(n_modes, n_wl, 1)) + 1j * rng.normal(size=(n_modes, n_wl, 1))
    ).astype(np.complex128)
    wavelengths = np.array([1.20e-6, 1.32e-6], dtype=np.float64)
    ports = np.array([0, 1], dtype=np.intp)
    counts_by_wave = np.full((ports.size, n_wl), 1.0e6, dtype=np.float64)
    basis = differential_piston_basis(n_modes)

    broadband = _joint_piston_amplitude_response(
        u_stack=u,
        input_fields=fields,
        wavelengths_m=wavelengths,
        port_indices=ports,
        wavelength_groups=(np.array([0, 1], dtype=np.intp),),
        expected_counts_by_wavelength=counts_by_wave,
        basis=basis,
    )
    spectral = _joint_piston_amplitude_response(
        u_stack=u,
        input_fields=fields,
        wavelengths_m=wavelengths,
        port_indices=ports,
        wavelength_groups=(
            np.array([0], dtype=np.intp),
            np.array([1], dtype=np.intp),
        ),
        expected_counts_by_wavelength=counts_by_wave,
        basis=basis,
    )

    assert broadband.shape == (2, 4)
    assert spectral.shape == (4, 4)
    assert svd_observability(broadband, relative_threshold=1.0e-9).rank == 2
    assert svd_observability(spectral, relative_threshold=1.0e-9).rank == 4


def test_svd_observability_sees_degenerate_identity_chip() -> None:
    n_modes = 4
    n_wl = 5
    u = np.repeat(np.eye(n_modes, dtype=np.complex128)[:, :, np.newaxis], n_wl, axis=2)
    fields = np.ones((n_modes, n_wl, 1), dtype=np.complex128)
    response = intensity_response_matrix(
        u_stack=u,
        input_fields=fields,
        wavelengths_m=np.linspace(1.0e-6, 1.4e-6, n_wl),
        port_indices=np.arange(n_modes),
    )
    basis = differential_piston_basis(n_modes)
    projected = response.response_power_per_m @ basis
    whitened = whiten_response_matrix(
        response_counts_per_m=scale_response_to_counts(
            response_power_per_m=response.response_power_per_m,
            reference_power=response.reference_power,
            expected_counts=np.full(response.reference_power.shape, 100.0),
        )
        @ basis,
        expected_counts=np.full(response.reference_power.shape, 100.0),
    )

    assert np.allclose(projected, 0.0)
    svd = svd_observability(whitened)
    crlb = cramer_rao_bound_from_svd(svd, n_piston_modes=n_modes)
    assert svd.rank == 0
    assert not svd.full_rank
    assert not crlb.finite
    assert np.isinf(crlb.piston_rms_m)
    assert np.isinf(crlb.truncated_piston_rms_m)


def test_random_chip_can_be_full_rank_in_differential_piston_space() -> None:
    n_modes = 4
    n_wl = 4
    rng = np.random.default_rng(19)
    u = np.stack(
        [_random_unitary(n_modes, seed=20 + i) for i in range(n_wl)],
        axis=2,
    )
    fields = rng.normal(size=(n_modes, n_wl, 1)) + 1j * rng.normal(
        size=(n_modes, n_wl, 1)
    )
    response = intensity_response_matrix(
        u_stack=u,
        input_fields=fields,
        wavelengths_m=np.linspace(1.0e-6, 1.3e-6, n_wl),
        port_indices=np.arange(n_modes),
    )
    basis = differential_piston_basis(n_modes)
    counts_response = scale_response_to_counts(
        response_power_per_m=response.response_power_per_m @ basis,
        reference_power=response.reference_power,
        expected_counts=np.full(response.reference_power.shape, 1.0e5),
    )
    whitened = whiten_response_matrix(
        response_counts_per_m=counts_response,
        expected_counts=np.full(response.reference_power.shape, 1.0e5),
    )

    svd = svd_observability(whitened)
    weak = weakest_piston_modes(
        right_singular_vectors=svd.right_singular_vectors,
        piston_basis=basis,
        n_weak_modes=2,
    )
    crlb = cramer_rao_bound_from_svd(svd, n_piston_modes=n_modes)

    assert svd.rank == n_modes - 1
    assert svd.full_rank
    assert np.isfinite(svd.condition_number)
    assert crlb.finite
    assert np.isfinite(crlb.piston_rms_m)
    assert weak.shape == (2, n_modes)
    np.testing.assert_allclose(weak.mean(axis=1), np.zeros(2), atol=1.0e-14)


def test_piston_measurement_covariance_matches_crlb_trace() -> None:
    n_modes = 4
    n_wl = 4
    rng = np.random.default_rng(21)
    u = np.stack(
        [_random_unitary(n_modes, seed=70 + i) for i in range(n_wl)],
        axis=2,
    )
    fields = rng.normal(size=(n_modes, n_wl, 1)) + 1j * rng.normal(
        size=(n_modes, n_wl, 1)
    )
    response = intensity_response_matrix(
        u_stack=u,
        input_fields=fields,
        wavelengths_m=np.linspace(1.0e-6, 1.3e-6, n_wl),
        port_indices=np.arange(n_modes),
    )
    basis = differential_piston_basis(n_modes)
    counts = np.full(response.reference_power.shape, 1.0e6)
    counts_response = scale_response_to_counts(
        response_power_per_m=response.response_power_per_m @ basis,
        reference_power=response.reference_power,
        expected_counts=counts,
    )
    whitened = whiten_response_matrix(
        response_counts_per_m=counts_response,
        expected_counts=counts,
    )
    svd = svd_observability(whitened)
    crlb = cramer_rao_bound_from_svd(svd, n_piston_modes=n_modes)

    covariance = piston_measurement_covariance_from_svd(svd, piston_basis=basis)

    assert covariance.finite
    assert covariance.piston_covariance_m2.shape == (n_modes, n_modes)
    np.testing.assert_allclose(
        covariance.piston_covariance_m2,
        covariance.piston_covariance_m2.T,
        atol=1.0e-30,
    )
    assert covariance.piston_rms_m == pytest.approx(crlb.piston_rms_m)
    assert np.min(np.linalg.eigvalsh(covariance.piston_covariance_m2)) > -1.0e-24


def test_reconstruction_monte_carlo_recovers_small_full_rank_pistons() -> None:
    n_modes = 4
    n_wl = 4
    rng = np.random.default_rng(41)
    u = np.stack(
        [_random_unitary(n_modes, seed=50 + i) for i in range(n_wl)],
        axis=2,
    )
    fields = rng.normal(size=(n_modes, n_wl, 1)) + 1j * rng.normal(
        size=(n_modes, n_wl, 1)
    )
    wavelengths = np.linspace(1.0e-6, 1.3e-6, n_wl)
    response = intensity_response_matrix(
        u_stack=u,
        input_fields=fields,
        wavelengths_m=wavelengths,
        port_indices=np.arange(n_modes),
    )
    basis = differential_piston_basis(n_modes)
    counts = np.full(response.reference_power.shape, 1.0e8)
    counts_response = scale_response_to_counts(
        response_power_per_m=response.response_power_per_m @ basis,
        reference_power=response.reference_power,
        expected_counts=counts,
    )
    whitened = whiten_response_matrix(
        response_counts_per_m=counts_response,
        expected_counts=counts,
    )

    mc = reconstruction_monte_carlo(
        u_stack=u,
        input_fields=fields,
        wavelengths_m=wavelengths,
        port_indices=np.arange(n_modes),
        wavelength_indices=np.arange(n_wl),
        piston_basis=basis,
        reference_power=response.reference_power,
        expected_counts=counts,
        whitened_response_per_m=whitened,
        piston_rms_m=1.0e-12,
        n_trials=48,
        rng=np.random.default_rng(42),
        shot_noise=False,
    )

    assert mc.median_residual_rms_m < 0.02 * float(np.median(mc.input_rms_m))


def test_reconstruction_monte_carlo_rejects_negative_noise_floor() -> None:
    n_modes = 3
    n_wl = 1
    u = np.eye(n_modes, dtype=np.complex128)[:, :, np.newaxis]
    fields = np.ones((n_modes, n_wl, 1), dtype=np.complex128)
    basis = differential_piston_basis(n_modes)

    with pytest.raises(ValueError, match="noise_floor_counts"):
        reconstruction_monte_carlo(
            u_stack=u,
            input_fields=fields,
            wavelengths_m=np.array([1.0e-6]),
            port_indices=np.arange(n_modes),
            wavelength_indices=np.arange(n_wl),
            piston_basis=basis,
            reference_power=np.ones(n_modes),
            expected_counts=np.ones(n_modes),
            whitened_response_per_m=np.ones((n_modes, n_modes - 1)),
            piston_rms_m=1.0e-12,
            n_trials=1,
            rng=np.random.default_rng(1),
            noise_floor_counts=-1.0,
        )


def test_gauss_newton_reconstruction_improves_finite_piston_fit() -> None:
    n_modes = 4
    n_wl = 5
    rng = np.random.default_rng(47)
    u = np.stack(
        [_random_unitary(n_modes, seed=70 + i) for i in range(n_wl)],
        axis=2,
    )
    fields = rng.normal(size=(n_modes, n_wl, 1)) + 1j * rng.normal(
        size=(n_modes, n_wl, 1)
    )
    wavelengths = np.linspace(1.0e-6, 1.35e-6, n_wl)
    response = intensity_response_matrix(
        u_stack=u,
        input_fields=fields,
        wavelengths_m=wavelengths,
        port_indices=np.arange(n_modes),
    )
    basis = differential_piston_basis(n_modes)
    counts = np.full(response.reference_power.shape, 1.0e9)
    counts_response = scale_response_to_counts(
        response_power_per_m=response.response_power_per_m @ basis,
        reference_power=response.reference_power,
        expected_counts=counts,
    )
    whitened = whiten_response_matrix(
        response_counts_per_m=counts_response,
        expected_counts=counts,
    )
    seed = 123

    one_step = reconstruction_monte_carlo(
        u_stack=u,
        input_fields=fields,
        wavelengths_m=wavelengths,
        port_indices=np.arange(n_modes),
        wavelength_indices=np.arange(n_wl),
        piston_basis=basis,
        reference_power=response.reference_power,
        expected_counts=counts,
        whitened_response_per_m=whitened,
        piston_rms_m=75.0e-9,
        n_trials=24,
        rng=np.random.default_rng(seed),
        shot_noise=False,
    )
    gn = gauss_newton_reconstruction_monte_carlo(
        u_stack=u,
        input_fields=fields,
        wavelengths_m=wavelengths,
        port_indices=np.arange(n_modes),
        wavelength_indices=np.arange(n_wl),
        piston_basis=basis,
        reference_power=response.reference_power,
        expected_counts=counts,
        whitened_response_per_m=whitened,
        piston_rms_m=75.0e-9,
        n_trials=24,
        rng=np.random.default_rng(seed),
        shot_noise=False,
        max_iter=12,
        damping=1.0e-3,
        step_tolerance_m=1.0e-13,
    )

    assert gn.convergence_fraction > 0.8
    assert gn.median_residual_rms_m < 0.25 * one_step.median_residual_rms_m
    assert gn.median_residual_rms_m < 0.10 * float(np.median(gn.input_rms_m))


def test_active_coordinate_acquisition_locks_from_50_nm_residuals() -> None:
    n_modes = 4
    mesh = quadrature_reference_mesh(n_modes=n_modes, differential_coupling=0.1)
    u = mesh.tracker_matrix[:, :, np.newaxis]
    fields = np.ones((n_modes, 1, 1), dtype=np.complex128)
    wavelengths = np.array([1.25e-6])
    response = intensity_response_matrix(
        u_stack=u,
        input_fields=fields,
        wavelengths_m=wavelengths,
        port_indices=np.arange(n_modes),
    )
    basis = differential_piston_basis(n_modes)

    mc = active_coordinate_acquisition_monte_carlo(
        u_stack=u,
        input_fields=fields,
        wavelengths_m=wavelengths,
        port_indices=np.arange(n_modes),
        wavelength_indices=np.array([0], dtype=np.intp),
        piston_basis=basis,
        reference_power=response.reference_power,
        expected_counts=np.full(response.reference_power.shape, 1.0e8),
        piston_rms_m=50.0e-9,
        n_trials=16,
        rng=np.random.default_rng(51),
        scan_offsets_m=np.linspace(-150.0, 150.0, 61) * 1.0e-9,
        n_passes=3,
        calibration_frames=4,
        lock_threshold_m=10.0e-9,
        shot_noise=True,
    )

    assert mc.lock_fraction >= 0.75
    assert mc.median_residual_rms_m < 0.25 * float(np.median(mc.input_rms_m))
    assert mc.p95_residual_rms_m < 10.0e-9
    assert np.all(mc.evaluations == 3 * (n_modes - 1) * 61)


def test_cophasing_handoff_locks_with_dedicated_pairwise_unitary() -> None:
    n_modes = 8
    basis = differential_piston_basis(n_modes)
    n_dof = basis.shape[1]
    science_response = np.eye(n_dof, dtype=np.float64) * 2.0e9

    mc = cophasing_handoff_monte_carlo(
        wavelengths_m=np.array([1.25e-6]),
        cophasing_counts_per_channel=np.array([2.0e7]),
        piston_basis=basis,
        science_whitened_response_per_m=science_response,
        piston_rms_m=50.0e-9,
        n_trials=32,
        rng=np.random.default_rng(52),
        edges=balanced_binary_tree_edges(n_modes),
        science_drift_rms_m=2.0e-9,
        lock_threshold_m=10.0e-9,
        shot_noise=True,
    )

    assert mc.edges.shape == (n_modes - 1, 2)
    assert mc.acquisition_lock_fraction > 0.8
    assert mc.science_lock_fraction > 0.8
    assert mc.median_acquisition_residual_rms_m < 10.0e-9
    assert mc.median_science_residual_rms_m < 10.0e-9


def test_linearized_reconstruction_monte_carlo_tracks_crlb_scale() -> None:
    n_modes = 4
    n_wl = 4
    rng = np.random.default_rng(44)
    u = np.stack(
        [_random_unitary(n_modes, seed=60 + i) for i in range(n_wl)],
        axis=2,
    )
    fields = rng.normal(size=(n_modes, n_wl, 1)) + 1j * rng.normal(
        size=(n_modes, n_wl, 1)
    )
    wavelengths = np.linspace(1.0e-6, 1.3e-6, n_wl)
    response = intensity_response_matrix(
        u_stack=u,
        input_fields=fields,
        wavelengths_m=wavelengths,
        port_indices=np.arange(n_modes),
    )
    basis = differential_piston_basis(n_modes)
    counts = np.full(response.reference_power.shape, 1.0e6)
    counts_response = scale_response_to_counts(
        response_power_per_m=response.response_power_per_m @ basis,
        reference_power=response.reference_power,
        expected_counts=counts,
    )
    whitened = whiten_response_matrix(
        response_counts_per_m=counts_response,
        expected_counts=counts,
    )
    svd = svd_observability(whitened)
    crlb = cramer_rao_bound_from_svd(svd, n_piston_modes=n_modes)

    mc = linearized_reconstruction_monte_carlo(
        whitened_response_per_m=whitened,
        piston_basis=basis,
        piston_rms_m=50.0e-9,
        n_trials=2048,
        rng=np.random.default_rng(45),
    )

    ratio = mc.median_residual_rms_m / crlb.piston_rms_m
    assert 0.4 < ratio < 1.8


def test_reconstruction_monte_carlo_leaves_degenerate_modes_unrecovered() -> None:
    n_modes = 4
    n_wl = 3
    u = np.repeat(np.eye(n_modes, dtype=np.complex128)[:, :, np.newaxis], n_wl, axis=2)
    fields = np.ones((n_modes, n_wl, 1), dtype=np.complex128)
    wavelengths = np.linspace(1.0e-6, 1.3e-6, n_wl)
    response = intensity_response_matrix(
        u_stack=u,
        input_fields=fields,
        wavelengths_m=wavelengths,
        port_indices=np.arange(n_modes),
    )
    basis = differential_piston_basis(n_modes)
    counts = np.full(response.reference_power.shape, 1.0e6)
    whitened = np.zeros((response.reference_power.size, n_modes - 1), dtype=np.float64)

    mc = reconstruction_monte_carlo(
        u_stack=u,
        input_fields=fields,
        wavelengths_m=wavelengths,
        port_indices=np.arange(n_modes),
        wavelength_indices=np.arange(n_wl),
        piston_basis=basis,
        reference_power=response.reference_power,
        expected_counts=counts,
        whitened_response_per_m=whitened,
        piston_rms_m=5.0e-12,
        n_trials=24,
        rng=np.random.default_rng(43),
        shot_noise=False,
    )

    np.testing.assert_allclose(mc.residual_rms_m, mc.input_rms_m, rtol=1.0e-12)
