from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from nullsim.physics import imaging
from nullsim.physics.imaging import (
    SubAperture,
    _solve_nnls,
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


def _hadamard_setup() -> tuple[tuple[SubAperture, ...], np.ndarray, np.ndarray]:
    subapertures = tuple(
        SubAperture(np.array([x, 0.0, 0.0], dtype=np.float64), 1.0)
        for x in (-7.5, -2.5, 2.5, 7.5)
    )
    hadamard = 0.5 * np.array(
        [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]],
        dtype=np.complex128,
    )
    return subapertures, hadamard, np.array([1, 2, 3], dtype=np.intp)


def _leakage_only_observation(**kwargs) -> object:
    defaults = dict(
        scene=create_scene(n_pix=8, fov_mas=40.0, planets=[]),
        hour_angles_rad=np.array([0.0], dtype=np.float64),
        declination_rad=0.0,
        wavelength_m=1.0e-6,
        stellar_photon_rate_hz=1.0e10,
        integration_time_per_step_s=1.0,
        poisson=False,
    )
    defaults.update(kwargs)
    return simulate_observation(**defaults)


def test_leakage_excess_follows_iid_piston_shape_not_coherent_residual() -> None:
    # A perfect Hadamard null has no coherent residual, so the old code took
    # the leakage shape from roundoff: a 1e-10 rad rotation sent it all to one
    # port. Equal |U_dm|^2 |a_m|^2 makes the IID-piston shape uniform, so the
    # 1e5 realized counts split 1e5/3 per dark port whatever the rotation.
    subapertures, hadamard, dark = _hadamard_setup()
    generator = np.zeros((4, 4), dtype=np.complex128)
    generator[1, 2] = generator[2, 1] = 1.0
    rotated = hadamard @ (np.eye(4) + 1j * 1.0e-10 * generator)

    observation = _leakage_only_observation(
        subapertures=subapertures,
        amplitudes=np.ones(4, dtype=np.complex128),
        unitary=rotated,
        dark_port_indices=dark,
        null_depth=1.0e-5,
    )

    np.testing.assert_allclose(observation.stellar_leakage_per_port, 1.0e5 / 3.0, rtol=1e-9)


def test_static_chip_leakage_lands_in_its_own_port() -> None:
    # a = (1, 0.9, 0.8, 1): the Hadamard dark rows give coherent amplitudes
    # (-0.05, 0.05, 0.15), so static powers (0.0025, 0.0025, 0.0225) / A with
    # A = sum|a|^2 = 3.45. With the null set to twice the static total, the
    # excess S equals the static total and splits evenly (|H_dm|^2 = 1/4).
    subapertures, hadamard, dark = _hadamard_setup()
    amplitudes = np.array([1.0, 0.9, 0.8, 1.0], dtype=np.complex128)
    static = np.array([0.0025, 0.0025, 0.0225]) / 3.45
    total = float(np.sum(static))

    observation = _leakage_only_observation(
        subapertures=subapertures,
        amplitudes=amplitudes,
        unitary=hadamard,
        dark_port_indices=dark,
        null_depth=2.0 * total,
    )

    np.testing.assert_allclose(
        observation.stellar_leakage_per_port,
        1.0e10 * (static + total / 3.0),
        rtol=1e-12,
    )


@pytest.mark.parametrize("wavelength_resolved", [False, True])
def test_phase_chop_states_share_one_realized_dark_leakage(
    wavelength_resolved: bool,
) -> None:
    # The plus and minus rows are a unitary recombination of the dark pair, so
    # over one step (two half-step states) they collect the dark leakage once:
    # null * F * dt_state = 1e-5 * 4e9 * 15 s = 6e5 counts, not 1.2e6.
    subapertures, amplitudes, unitary, _dark_indices = _two_aperture_setup()
    wavelengths = ((unitary, 0.9e-6), (unitary, 1.1e-6))

    observation = simulate_phase_chopped_observation(
        scene=create_scene(n_pix=8, fov_mas=40.0, planets=[]),
        subapertures=subapertures,
        amplitudes=amplitudes,
        unitary=unitary,
        dark_port_indices=np.array([0, 1], dtype=np.intp),
        hour_angles_rad=np.array([0.0], dtype=np.float64),
        declination_rad=0.0,
        wavelength_m=1.0e-6,
        stellar_photon_rate_hz=4.0e9,
        null_depth=1.0e-5,
        integration_time_per_step_s=30.0,
        unitaries_by_wavelength=wavelengths,
        wavelength_resolved=wavelength_resolved,
        poisson=False,
    )

    assert float(np.sum(observation.stellar_leakage_per_port)) == pytest.approx(6.0e5)


def test_broadband_observation_weights_bins_by_stellar_spectrum() -> None:
    # Broadband counts are the resolved counts summed over wavelength; with
    # all the starlight in the first bin, the second bin's beam must not count.
    subapertures, amplitudes, unitary, dark_indices = _two_aperture_setup()
    scene = create_scene(n_pix=20, fov_mas=50.0, planets=[(20.0, 90.0, 2.0e-6)])
    common = dict(
        scene=scene,
        subapertures=subapertures,
        amplitudes=amplitudes,
        unitary=unitary,
        dark_port_indices=dark_indices,
        hour_angles_rad=np.array([0.0, 0.1], dtype=np.float64),
        declination_rad=0.0,
        wavelength_m=1.0e-6,
        stellar_photon_rate_hz=4.0e9,
        stellar_photon_rate_hz_by_wavelength=np.array([1.0, 0.0]),
        null_depth=1.0e-8,
        integration_time_per_step_s=30.0,
        sky_rate_hz_per_port=0.2,
        unitaries_by_wavelength=((unitary, 0.9e-6), (unitary, 1.1e-6)),
        poisson=False,
    )

    broadband = simulate_observation(**common)
    resolved = simulate_observation(**common, wavelength_resolved=True)

    np.testing.assert_allclose(
        broadband.measurements,
        resolved.measurements.sum(axis=1),
        rtol=1e-12,
    )


def test_snr_map_rebuilds_response_with_band_weights() -> None:
    subapertures, amplitudes, unitary, dark_indices = _two_aperture_setup()
    scene = create_scene(n_pix=20, fov_mas=50.0, planets=[(20.0, 90.0, 5.0e-5)])
    wavelengths = ((unitary, 0.9e-6), (unitary, 1.1e-6))
    common = dict(
        scene=scene,
        subapertures=subapertures,
        amplitudes=amplitudes,
        unitary=unitary,
        dark_port_indices=dark_indices,
        hour_angles_rad=np.linspace(-0.05, 0.05, 3),
        declination_rad=0.0,
        wavelength_m=1.0e-6,
        stellar_photon_rate_hz=2.0e9,
        stellar_photon_rate_hz_by_wavelength=np.array([3.0, 1.0]),
        null_depth=1.0e-8,
        integration_time_per_step_s=20.0,
        unitaries_by_wavelength=wavelengths,
        poisson=False,
    )
    maps = [
        reconstruct_snr_map(
            data=simulate_observation(**common, keep_response_matrix=keep),
            subapertures=subapertures,
            amplitudes=amplitudes,
            unitary=unitary,
            declination_rad=0.0,
            wavelength_m=1.0e-6,
            l_coords=scene.l_coords,
            m_coords=scene.m_coords,
            unitaries_by_wavelength=wavelengths,
        ).image
        for keep in (True, False)
    ]

    np.testing.assert_allclose(maps[1], maps[0], rtol=1e-12)


def test_bounded_nnls_reports_iterations_and_convergence() -> None:
    # b = A (2, -1): the bound binds, x2 = 0, and x1 = (3 + 1) / 2 = 2 from the
    # first column (the x2 gradient at (2, 0) is +3, so KKT holds).
    design = np.array([[1.0, -1.0], [1.0, 1.0], [0.0, 1.0]])
    solve = _solve_nnls(design, design @ np.array([2.0, -1.0]))

    np.testing.assert_allclose(solve.x, [2.0, 0.0], atol=1e-6)
    assert solve.converged
    assert solve.status > 0
    assert solve.n_iterations > 0


def test_bounded_nnls_retries_normalized_then_flags_failure(monkeypatch) -> None:
    calls: list[tuple[np.ndarray, int]] = []

    def stalled(design, rhs, **kwargs):
        calls.append((np.linalg.norm(design, axis=0), kwargs["max_iter"]))
        return OptimizeResult(
            x=np.ones(design.shape[1]), status=0, nit=kwargs["max_iter"],
            message="The maximum number of iterations is exceeded.",
        )

    monkeypatch.setattr(imaging, "lsq_linear", stalled)
    design = np.hstack((np.full((40, 15), -2.0), np.full((40, 15), 3.0)))

    with pytest.warns(RuntimeWarning, match="did not converge"):
        solve = _solve_nnls(design, np.ones(40))

    assert [max_iter for _norms, max_iter in calls] == [200, 600]
    np.testing.assert_allclose(calls[1][0], 1.0)
    assert not solve.converged
    assert solve.status == 0
    assert solve.n_iterations == 800
    assert "maximum number of iterations" in solve.message
