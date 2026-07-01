from __future__ import annotations

import numpy as np
import pytest

from nullsim.physics.imaging import (
    SubAperture,
    build_response_matrix,
    build_wavelength_resolved_response_matrix,
    create_scene,
    phase_chop_unitary_pair,
    reconstruct_nnls,
    reconstruct_snr_map,
    simulate_observation,
    simulate_phase_chopped_observation,
)


def _two_aperture_setup() -> tuple[
    tuple[SubAperture, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    subapertures = (
        SubAperture(np.array([-5.0, 0.0, 0.0], dtype=np.float64), 1.0),
        SubAperture(np.array([5.0, 0.0, 0.0], dtype=np.float64), 1.0),
    )
    amplitudes = np.ones(2, dtype=np.complex128)
    unitary = np.array(
        [[1.0, 1.0], [1.0, -1.0]],
        dtype=np.complex128,
    ) / np.sqrt(2.0)
    dark_indices = np.array([1], dtype=np.intp)
    return subapertures, amplitudes, unitary, dark_indices


def test_broadband_response_changes_with_wavelength() -> None:
    subapertures, amplitudes, unitary, dark_indices = _two_aperture_setup()
    scene = create_scene(
        n_pix=24,
        fov_mas=40.0,
        planets=[(20.0, 90.0, 1.0e-7)],
    )
    hour_angles = np.array([0.0], dtype=np.float64)

    response_center = build_response_matrix(
        subapertures=subapertures,
        amplitudes=amplitudes,
        unitary=unitary,
        dark_port_indices=dark_indices,
        hour_angles_rad=hour_angles,
        declination_rad=0.0,
        wavelength_m=1.0e-6,
        l_coords=scene.l_coords,
        m_coords=scene.m_coords,
    )
    response_broad = build_response_matrix(
        subapertures=subapertures,
        amplitudes=amplitudes,
        unitary=unitary,
        dark_port_indices=dark_indices,
        hour_angles_rad=hour_angles,
        declination_rad=0.0,
        wavelength_m=1.0e-6,
        l_coords=scene.l_coords,
        m_coords=scene.m_coords,
        unitaries_by_wavelength=((unitary, 0.9e-6), (unitary, 1.1e-6)),
    )

    assert response_center.shape == response_broad.shape
    assert not np.allclose(response_center, response_broad)


def test_wavelength_resolved_response_keeps_bins_as_rows() -> None:
    subapertures, amplitudes, unitary, dark_indices = _two_aperture_setup()
    scene = create_scene(
        n_pix=16,
        fov_mas=40.0,
        planets=[(20.0, 90.0, 1.0e-7)],
    )
    hour_angles = np.array([0.0, 0.1], dtype=np.float64)

    response = build_wavelength_resolved_response_matrix(
        subapertures=subapertures,
        amplitudes=amplitudes,
        unitary=unitary,
        dark_port_indices=dark_indices,
        hour_angles_rad=hour_angles,
        declination_rad=0.0,
        wavelength_m=1.0e-6,
        l_coords=scene.l_coords,
        m_coords=scene.m_coords,
        unitaries_by_wavelength=((unitary, 0.9e-6), (unitary, 1.1e-6)),
    )

    assert response.shape == (4, scene.n_pix * scene.n_pix)
    assert not np.allclose(response[0], response[1])


def test_spacecraft_roll_changes_response_rows() -> None:
    subapertures, amplitudes, unitary, dark_indices = _two_aperture_setup()
    scene = create_scene(
        n_pix=24,
        fov_mas=40.0,
        planets=[(20.0, 90.0, 1.0e-7)],
    )
    response = build_response_matrix(
        subapertures=subapertures,
        amplitudes=amplitudes,
        unitary=unitary,
        dark_port_indices=dark_indices,
        hour_angles_rad=np.array([0.0, 0.0], dtype=np.float64),
        roll_angles_rad=np.array([0.0, np.pi / 2.0], dtype=np.float64),
        declination_rad=0.0,
        wavelength_m=1.0e-6,
        l_coords=scene.l_coords,
        m_coords=scene.m_coords,
    )

    assert response.shape == (2, scene.n_pix * scene.n_pix)
    assert not np.allclose(response[0], response[1])


def test_phase_chop_unitary_pair_combines_dark_ports() -> None:
    _subapertures, _amplitudes, unitary, _dark_indices = _two_aperture_setup()

    plus, minus, chop_indices = phase_chop_unitary_pair(
        unitary,
        np.array([0, 1], dtype=np.intp),
    )

    expected_plus = (unitary[0] + 1j * unitary[1]) / np.sqrt(2.0)
    expected_minus = (unitary[0] - 1j * unitary[1]) / np.sqrt(2.0)
    np.testing.assert_allclose(plus, expected_plus[np.newaxis, :])
    np.testing.assert_allclose(minus, expected_minus[np.newaxis, :])
    np.testing.assert_array_equal(chop_indices, np.array([0], dtype=np.intp))


def test_phase_chopped_observation_uses_signed_difference_variance() -> None:
    subapertures, amplitudes, unitary, _dark_indices = _two_aperture_setup()
    dark_indices = np.array([0, 1], dtype=np.intp)
    scene = create_scene(
        n_pix=24,
        fov_mas=50.0,
        planets=[(20.0, 90.0, 2.0e-6)],
    )

    observation = simulate_phase_chopped_observation(
        scene=scene,
        subapertures=subapertures,
        amplitudes=amplitudes,
        unitary=unitary,
        dark_port_indices=dark_indices,
        hour_angles_rad=np.array([0.0], dtype=np.float64),
        declination_rad=0.0,
        wavelength_m=1.0e-6,
        stellar_photon_rate_hz=4.0e9,
        null_depth=1.0e-8,
        integration_time_per_step_s=30.0,
        sky_rate_hz_per_port=0.2,
        keep_response_matrix=True,
        poisson=False,
    )

    assert observation.measurements.shape == (1, 1)
    assert observation.response_matrix is not None
    assert observation.response_matrix.shape == (1, scene.n_pix * scene.n_pix)
    assert observation.baseline_counts is not None
    assert observation.variance_counts is not None
    assert float(np.max(observation.variance_counts)) > abs(
        float(np.max(observation.baseline_counts))
    )
    assert not np.allclose(observation.response_matrix, 0.0)


@pytest.mark.parametrize("wavelength_resolved", [False, True])
def test_phase_chopped_null_instability_is_common_mode(
    wavelength_resolved: bool,
) -> None:
    subapertures, amplitudes, unitary, _dark_indices = _two_aperture_setup()
    dark_indices = np.array([0, 1], dtype=np.intp)
    scene = create_scene(n_pix=16, fov_mas=40.0, planets=[])
    wavelengths = ((unitary, 0.9e-6), (unitary, 1.1e-6))

    observation = simulate_phase_chopped_observation(
        scene=scene,
        subapertures=subapertures,
        amplitudes=amplitudes,
        unitary=unitary,
        dark_port_indices=dark_indices,
        hour_angles_rad=np.array([0.0, 0.1, 0.2], dtype=np.float64),
        declination_rad=0.0,
        wavelength_m=1.0e-6,
        stellar_photon_rate_hz=4.0e9,
        stellar_photon_rate_hz_by_wavelength=(
            np.array([1.6e9, 2.4e9], dtype=np.float64)
            if wavelength_resolved
            else None
        ),
        null_depth=1.0e-5,
        integration_time_per_step_s=30.0,
        null_instability=0.5,
        unitaries_by_wavelength=wavelengths if wavelength_resolved else (),
        wavelength_resolved=wavelength_resolved,
        poisson=False,
    )

    assert observation.baseline_counts is not None
    assert observation.variance_counts is not None
    np.testing.assert_allclose(observation.baseline_counts, 0.0, atol=1.0e-9)
    np.testing.assert_allclose(observation.measurements, 0.0, atol=1.0e-9)
    assert float(np.max(observation.variance_counts)) > 0.0


@pytest.mark.parametrize("wavelength_resolved", [False, True])
def test_simulate_observation_variance_counts_track_expected_counts(
    wavelength_resolved: bool,
) -> None:
    subapertures, amplitudes, unitary, dark_indices = _two_aperture_setup()
    scene = create_scene(
        n_pix=20,
        fov_mas=50.0,
        planets=[(20.0, 90.0, 2.0e-6)],
    )
    wavelengths = ((unitary, 0.9e-6), (unitary, 1.1e-6))

    observation = simulate_observation(
        scene=scene,
        subapertures=subapertures,
        amplitudes=amplitudes,
        unitary=unitary,
        dark_port_indices=dark_indices,
        hour_angles_rad=np.array([0.0, 0.1], dtype=np.float64),
        declination_rad=0.0,
        wavelength_m=1.0e-6,
        stellar_photon_rate_hz=4.0e9,
        stellar_photon_rate_hz_by_wavelength=(
            np.array([1.6e9, 2.4e9], dtype=np.float64)
            if wavelength_resolved
            else None
        ),
        dark_count_rate_hz_per_port_by_wavelength=(
            np.array([0.1, 0.2], dtype=np.float64) if wavelength_resolved else None
        ),
        sky_rate_hz_per_port_by_wavelength=(
            np.array([0.3, 0.4], dtype=np.float64) if wavelength_resolved else None
        ),
        null_depth=1.0e-8,
        integration_time_per_step_s=30.0,
        dark_count_rate_hz_per_port=0.3,
        sky_rate_hz_per_port=0.7,
        unitaries_by_wavelength=wavelengths if wavelength_resolved else (),
        wavelength_resolved=wavelength_resolved,
        poisson=False,
    )

    assert observation.variance_counts is not None
    np.testing.assert_allclose(observation.variance_counts, observation.measurements)


def test_wavelength_rates_honor_zero_total_with_supplied_spectrum() -> None:
    subapertures, amplitudes, unitary, dark_indices = _two_aperture_setup()
    scene = create_scene(
        n_pix=20,
        fov_mas=50.0,
        planets=[(20.0, 90.0, 2.0e-6)],
    )
    wavelengths = ((unitary, 0.9e-6), (unitary, 1.1e-6))

    observation = simulate_observation(
        scene=scene,
        subapertures=subapertures,
        amplitudes=amplitudes,
        unitary=unitary,
        dark_port_indices=dark_indices,
        hour_angles_rad=np.array([0.0, 0.1], dtype=np.float64),
        declination_rad=0.0,
        wavelength_m=1.0e-6,
        stellar_photon_rate_hz=0.0,
        stellar_photon_rate_hz_by_wavelength=np.array([1.6e9, 2.4e9]),
        null_depth=1.0e-8,
        integration_time_per_step_s=30.0,
        unitaries_by_wavelength=wavelengths,
        wavelength_resolved=True,
        poisson=False,
    )

    assert observation.variance_counts is not None
    np.testing.assert_allclose(observation.measurements, 0.0)
    np.testing.assert_allclose(observation.variance_counts, 0.0)
    np.testing.assert_allclose(observation.response_scale_per_row, 0.0)


def test_matched_filter_snr_is_positive_at_injected_planet() -> None:
    subapertures, amplitudes, unitary, dark_indices = _two_aperture_setup()
    scene = create_scene(
        n_pix=32,
        fov_mas=50.0,
        planets=[(20.0, 90.0, 5.0e-5)],
    )
    hour_angles = np.linspace(-0.05, 0.05, 5)

    observation = simulate_observation(
        scene=scene,
        subapertures=subapertures,
        amplitudes=amplitudes,
        unitary=unitary,
        dark_port_indices=dark_indices,
        hour_angles_rad=hour_angles,
        declination_rad=0.0,
        wavelength_m=1.0e-6,
        stellar_photon_rate_hz=2.0e9,
        null_depth=1.0e-8,
        integration_time_per_step_s=20.0,
        sky_rate_hz_per_port=0.2,
        poisson=False,
    )
    reconstruction = reconstruct_snr_map(
        data=observation,
        subapertures=subapertures,
        amplitudes=amplitudes,
        unitary=unitary,
        declination_rad=0.0,
        wavelength_m=1.0e-6,
        l_coords=scene.l_coords,
        m_coords=scene.m_coords,
    )

    coords_mas = scene.l_coords * 206_264_806.247
    col = int(np.argmin(np.abs(coords_mas - 20.0)))
    row = int(np.argmin(np.abs(coords_mas)))
    assert float(reconstruction.image[row, col]) > 5.0


def test_matched_filter_snr_uses_inverse_variance_weights() -> None:
    subapertures, amplitudes, unitary, _dark_indices = _two_aperture_setup()
    dark_indices = np.array([0, 1], dtype=np.intp)
    scene = create_scene(
        n_pix=32,
        fov_mas=50.0,
        planets=[(20.0, 90.0, 5.0e-5)],
    )
    hour_angles = np.array([0.0], dtype=np.float64)
    observation = simulate_observation(
        scene=scene,
        subapertures=subapertures,
        amplitudes=amplitudes,
        unitary=unitary,
        dark_port_indices=dark_indices,
        hour_angles_rad=hour_angles,
        declination_rad=0.0,
        wavelength_m=1.0e-6,
        stellar_photon_rate_hz=2.0e9,
        null_depth=1.0e-6,
        integration_time_per_step_s=20.0,
        sky_rate_hz_per_port=0.2,
        poisson=False,
    )
    reconstruction = reconstruct_snr_map(
        data=observation,
        subapertures=subapertures,
        amplitudes=amplitudes,
        unitary=unitary,
        declination_rad=0.0,
        wavelength_m=1.0e-6,
        l_coords=scene.l_coords,
        m_coords=scene.m_coords,
    )
    response = build_response_matrix(
        subapertures=subapertures,
        amplitudes=amplitudes,
        unitary=unitary,
        dark_port_indices=dark_indices,
        hour_angles_rad=hour_angles,
        declination_rad=0.0,
        wavelength_m=1.0e-6,
        l_coords=scene.l_coords,
        m_coords=scene.m_coords,
    )
    coords_mas = scene.l_coords * 206_264_806.247
    col = int(np.argmin(np.abs(coords_mas - 20.0)))
    row = int(np.argmin(np.abs(coords_mas)))
    flat_index = row * scene.n_pix + col
    baseline = observation.baseline_counts.ravel()
    variance = observation.variance_counts.ravel()
    residual = observation.measurements.ravel() - baseline
    inv_variance = 1.0 / variance
    sqrt_inv_variance = np.sqrt(inv_variance)
    response_scale = observation.response_scale_per_row.ravel()
    design = response * response_scale[:, None] * sqrt_inv_variance[:, None]
    rhs = residual * sqrt_inv_variance
    common = np.mean(design, axis=1)
    common_norm_sq = float(common @ common)
    design = design - common[:, None] * ((common @ design) / common_norm_sq)[
        None,
        :,
    ]
    rhs = rhs - common * (float(common @ rhs) / common_norm_sq)
    expected_snr = float(
        np.sum(rhs * design[:, flat_index])
        / np.sqrt(np.sum(design[:, flat_index] ** 2))
    )

    assert reconstruction.image[row, col] == pytest.approx(expected_snr)


def test_wavelength_resolved_nnls_preserves_contrast_units() -> None:
    subapertures, amplitudes, unitary, dark_indices = _two_aperture_setup()
    scene = create_scene(
        n_pix=20,
        fov_mas=50.0,
        planets=[(20.0, 90.0, 2.0e-6)],
    )
    hour_angles = np.linspace(-0.05, 0.05, 3)
    wavelengths = ((unitary, 0.9e-6), (unitary, 1.1e-6))

    observation = simulate_observation(
        scene=scene,
        subapertures=subapertures,
        amplitudes=amplitudes,
        unitary=unitary,
        dark_port_indices=dark_indices,
        hour_angles_rad=hour_angles,
        declination_rad=0.0,
        wavelength_m=1.0e-6,
        stellar_photon_rate_hz=4.0e9,
        stellar_photon_rate_hz_by_wavelength=np.array([1.6e9, 2.4e9]),
        null_depth=1.0e-8,
        integration_time_per_step_s=30.0,
        sky_rate_hz_per_port=0.2,
        keep_response_matrix=True,
        unitaries_by_wavelength=wavelengths,
        wavelength_resolved=True,
        poisson=False,
    )
    reconstruction = reconstruct_nnls(
        data=observation,
        subapertures=subapertures,
        amplitudes=amplitudes,
        unitary=unitary,
        declination_rad=0.0,
        wavelength_m=1.0e-6,
        l_coords=scene.l_coords,
        m_coords=scene.m_coords,
        unitaries_by_wavelength=wavelengths,
    )

    assert observation.measurements.shape == (3, 2, 1)
    assert observation.response_matrix is not None
    assert observation.response_matrix.shape == (6, scene.n_pix * scene.n_pix)
    np.testing.assert_allclose(np.max(reconstruction.image), np.max(scene.image))
    np.testing.assert_allclose(np.sum(reconstruction.image), np.sum(scene.image))
