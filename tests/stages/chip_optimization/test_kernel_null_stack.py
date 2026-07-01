"""Tests for the kernel-null field stack (KILO injection.py L226-330 port)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nullsim.pipeline import RunContext
from nullsim.pipeline.state import (
    Field,
    Geometry,
    Scene,
    SimulationState,
    Throughput,
)
from nullsim.stages.chip_optimization import (
    ChipOptimizationParams,
    _build_unitary_stack_numpy,
    _kernel_null_field_stack,
    _delay_state_current_transport_scale_from_state,
    _delay_state_fields_from_state,
    _delay_state_transport_scales_from_state,
    _loss_meets_target,
    _loss_numpy,
    _normalization_field_array,
    _normalized_loss_for_target,
    _null_indices,
    _optimizer_null_field_stack,
    _quantize_phases,
    _reduce_spectral_leakage_numpy,
    _reported_null_depth_for_target,
    _tracker_piston_basis_from_state,
)
from nullsim.stages.chip_optimization.ideal_kernel import (
    _complete_orthonormal_basis,
    _decompose_real_clements,
    _stellar_subspace_basis,
)
from nullsim.stages.chip_optimization.scipy import ChipOptimizationScipy


def test_kernel_order_0_returns_one_row() -> None:
    on_axis = np.array([1.0 + 1.0j, 2.0 + 0.5j, 0.5 + 1.5j], dtype=np.complex128)
    positions = np.array([[1.0, 0.0], [2.0, 1.0], [-1.0, 1.0]])
    stack = _kernel_null_field_stack(on_axis, positions, order=0)
    assert stack.shape == (1, 3)
    np.testing.assert_array_equal(stack[0], on_axis)


def test_kernel_order_2_returns_three_rows_l2_equalized() -> None:
    on_axis = np.array([1.0 + 1.0j, 2.0 + 0.5j, 0.5 + 1.5j], dtype=np.complex128)
    positions = np.array([[1.0, 0.5], [2.0, 1.0], [-1.0, 1.0]])
    stack = _kernel_null_field_stack(on_axis, positions, order=2)
    assert stack.shape == (3, 3)
    # All rows must share the same L2 norm (KILO L2-equalization).
    norms = np.linalg.norm(stack, axis=1)
    np.testing.assert_allclose(norms, norms[0], rtol=1e-12)


def test_kernel_order_3_returns_six_rows() -> None:
    """All 6 derivative rows are kept when no weight vector vanishes."""

    on_axis = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.complex128)
    # Positions with non-zero x*y, x^2, y^2, x, y for every row.
    positions = np.array([[1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]])
    stack = _kernel_null_field_stack(on_axis, positions, order=3)
    assert stack.shape == (6, 4)


def test_kernel_order_3_drops_zero_weight_rows() -> None:
    """Cross-axis positions drop the xy row since each position has x=0 or y=0."""

    on_axis = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.complex128)
    positions = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    stack = _kernel_null_field_stack(on_axis, positions, order=3)
    # on-axis + x + y + x^2 + y^2 = 5 (xy drops because every position has x*y=0).
    assert stack.shape == (5, 4)


def test_kernel_order_4_returns_ten_rows() -> None:
    on_axis = np.array([1.0 + 0.1j] * 5, dtype=np.complex128)
    rng = np.random.default_rng(0)
    positions = rng.normal(size=(5, 2))
    stack = _kernel_null_field_stack(on_axis, positions, order=4)
    assert stack.shape == (10, 5)


def test_kernel_skips_zero_weight_rows() -> None:
    """A row whose weight vector is all-zero should be silently dropped."""

    on_axis = np.array([1.0 + 0j, 1.0 + 0j, 1.0 + 0j], dtype=np.complex128)
    # All sub-apertures on the y-axis only — x weights all zero.
    positions = np.array([[0.0, 1.0], [0.0, -1.0], [0.0, 0.5]])
    stack = _kernel_null_field_stack(on_axis, positions, order=2)
    # order=2: on-axis + x-weighted (zero norm, dropped) + y-weighted = 2 rows
    assert stack.shape == (2, 3)


def test_kernel_invalid_order_rejected() -> None:
    on_axis = np.array([1.0 + 0j], dtype=np.complex128)
    positions = np.zeros((1, 2))
    with pytest.raises(ValueError, match="kernel_order must be 0, 2, 3, or 4"):
        _kernel_null_field_stack(on_axis, positions, order=1)


def test_real_clements_decomposition_uses_zero_internal_phase() -> None:
    theta = 0.37
    c = np.cos(theta)
    s = np.sin(theta)
    target = np.array(
        [
            [c, s, 0.0],
            [-s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.complex128,
    )

    decomp = _decompose_real_clements(target, atol=1.0e-12)

    np.testing.assert_allclose(decomp.phi, 0.0, atol=0.0)
    reconstructed = _build_unitary_stack_numpy(
        params_vec=np.concatenate([decomp.theta, decomp.phi, decomp.alpha]),
        n_modes=3,
        wavelength_grid_um=(0.95, 1.0, 1.05),
        wl_design_um=1.0,
        dac_bits=None,
        chromaticity_params=ChipOptimizationParams(
            n_modes=3,
            n_bright_ports=1,
            n_dark_ports=2,
            wl_design_um=1.0,
            chip_chromaticity_model="parametric",
            phase_chromaticity_model="fixed_opd_neff",
        ),
    )
    for wavelength_index in range(reconstructed.shape[2]):
        np.testing.assert_allclose(
            np.abs(reconstructed[:, :, wavelength_index]),
            np.abs(target),
            atol=1.0e-12,
        )


def test_ideal_kernel_basis_completion_is_canonical() -> None:
    primary = np.array(
        [
            [1.0 / np.sqrt(2.0), 0.0],
            [-1.0 / np.sqrt(2.0), 0.0],
            [0.0, 1.0],
        ],
        dtype=np.complex128,
    )

    basis_a = _complete_orthonormal_basis(primary, n_modes=3, atol=1.0e-12)
    basis_b = _complete_orthonormal_basis(-primary, n_modes=3, atol=1.0e-12)

    np.testing.assert_allclose(basis_a.conj().T @ basis_a, np.eye(3), atol=1.0e-12)
    np.testing.assert_allclose(np.abs(basis_a), np.abs(basis_b), atol=1.0e-12)
    np.testing.assert_allclose(basis_a[:, 2], np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0))


def test_ideal_kernel_basis_uses_svd_principal_components() -> None:
    design_field = np.array([1.0, 0.7, 1.3, 0.5], dtype=np.complex128)[:, np.newaxis]
    positions = np.array(
        [[-1.0, -0.4], [-0.2, 0.7], [0.8, -0.1], [1.5, 0.9]],
        dtype=np.float64,
    )
    stack = _kernel_null_field_stack(design_field, positions, order=2)
    vectors = stack.reshape(stack.shape[0], stack.shape[1], -1)
    matrix = np.moveaxis(vectors, 1, 0).reshape(stack.shape[1], -1)
    expected_u, expected_singular_values, _vh = np.linalg.svd(
        matrix,
        full_matrices=True,
    )

    basis, singular_values, rank = _stellar_subspace_basis(
        design_field=design_field,
        positions=positions,
        kernel_order=2,
        rank_rtol=1.0e-12,
    )

    assert rank == 3
    np.testing.assert_allclose(singular_values, expected_singular_values)
    expected_projector = expected_u[:, :rank] @ expected_u[:, :rank].conj().T
    actual_projector = basis[:, :rank] @ basis[:, :rank].conj().T
    np.testing.assert_allclose(actual_projector, expected_projector, atol=1.0e-12)
    assert abs(np.vdot(expected_u[:, 0], basis[:, 0])) == pytest.approx(1.0)


def test_real_clements_decomposition_rejects_complex_target() -> None:
    target = np.eye(3, dtype=np.complex128)
    target[0, 0] = 1.0j

    with pytest.raises(ValueError, match="real-valued target unitary"):
        _decompose_real_clements(target, atol=1.0e-12)


def test_kernel_multi_wavelength_pol_preserved() -> None:
    """Stack broadcasts over (n_wl, n_pol) trailing axes correctly."""

    on_axis = np.ones((4, 3, 2), dtype=np.complex128)
    positions = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    stack = _kernel_null_field_stack(on_axis, positions, order=2)
    assert stack.shape == (3, 4, 3, 2)
    # Row 0 should be the unmodified on-axis field.
    np.testing.assert_array_equal(stack[0], on_axis)


def test_tracker_piston_basis_expands_mode_splitter_physical_modes() -> None:
    state = replace(
        SimulationState.empty({"wavelength_grid_um": (1.25,)}),
        geometry=Geometry(
            n_apertures=2,
            components={
                "quadrature_split": {
                    "n_input_modes": 4,
                    "n_output_modes": 8,
                    "copies": 2,
                }
            },
        ),
    )

    basis = _tracker_piston_basis_from_state(state, n_chip_modes=8)

    assert basis.source == "geometry.quadrature_split"
    assert basis.n_physical_modes == 4
    assert basis.physical_basis.shape == (4, 3)
    assert basis.chip_basis.shape == (8, 3)
    np.testing.assert_allclose(basis.chip_basis[0], basis.chip_basis[2])
    np.testing.assert_allclose(basis.chip_basis[1], basis.chip_basis[3])
    np.testing.assert_allclose(basis.chip_basis[4], basis.chip_basis[6])
    np.testing.assert_allclose(basis.chip_basis[5], basis.chip_basis[7])


def test_tracker_piston_basis_uses_partial_splitter_physical_indices() -> None:
    state = replace(
        SimulationState.empty({"wavelength_grid_um": (1.25,)}),
        geometry=Geometry(
            n_apertures=2,
            components={
                "partial_split": {
                    "n_input_modes": 4,
                    "n_output_modes": 6,
                    "copies": 2,
                    "physical_mode_indices": np.array([0, 0, 1, 2, 2, 3]),
                }
            },
        ),
    )

    basis = _tracker_piston_basis_from_state(state, n_chip_modes=6)

    assert basis.source == "geometry.partial_split"
    assert basis.n_physical_modes == 4
    assert basis.physical_basis.shape == (4, 3)
    assert basis.chip_basis.shape == (6, 3)
    np.testing.assert_allclose(basis.chip_basis[0], basis.chip_basis[1])
    np.testing.assert_allclose(basis.chip_basis[3], basis.chip_basis[4])


def test_optimizer_stack_adds_weighted_disk_sample_rows() -> None:
    on_axis = np.ones((4, 2, 1), dtype=np.complex128) / 2.0
    positions = np.array(
        [[0.0, 0.0], [1.0, 0.5], [-0.7, 1.2], [0.2, -1.1]],
        dtype=np.float64,
    )
    params = ChipOptimizationParams(
        n_modes=4,
        n_bright_ports=2,
        n_dark_ports=2,
        wl_design_um=1.25,
        kernel_order=2,
        disk_n_rings=1,
        disk_n_azimuths=2,
        disk_weight=0.25,
    )

    stack = _optimizer_null_field_stack(
        on_axis_field=on_axis,
        params=params,
        aperture_positions_pupil_m=positions,
        wavelength_grid_um=(1.20, 1.30),
        sub_diameters_m=np.ones(4, dtype=np.float64),
        stellar_radius_rad=1.0e-7,
    )

    assert stack.shape == (7, 4, 2, 1)
    on_axis_norm = np.linalg.norm(stack[0])
    # The three kernel rows stay unit-weighted; four disk rows are scaled
    # by sqrt(disk_weight).
    kernel_norms = np.sqrt(np.sum(np.abs(stack[:3]) ** 2, axis=(1, 2, 3)))
    disk_norms = np.sqrt(np.sum(np.abs(stack[3:]) ** 2, axis=(1, 2, 3)))
    np.testing.assert_allclose(kernel_norms, on_axis_norm)
    np.testing.assert_allclose(
        disk_norms,
        0.5 * on_axis_norm,
        rtol=1.0e-12,
    )


def test_delay_state_fields_replay_geometric_delay_phase() -> None:
    wavelengths_um = (1.50, 1.55, 1.60)
    field = np.ones((2, 3, 1), dtype=np.complex128)
    aperture_positions_enu_m = np.array(
        [[0.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
        dtype=np.float64,
    )
    delay_diag = {
        "fiber_lengths_m": (1.0, 1.0),
        "n_eff": 1.0003,
        "phase_index_ref": 1.0003,
        "group_index_ref": 1.0004,
        "dispersion_ps_nm_km": 1.0,
        "attenuation_db_per_m": 0.0,
        "altitude_deg": 0.0,
        "azimuth_deg": 0.0,
        "fixed_delay_line_loss_db_per_telescope": None,
        "chromatic_control": "multi_wavelength_group_delay",
        "reference_wavelength_um": 1.55,
        "per_mode_field_amplitude_applied": True,
    }
    state = SimulationState.empty({"wavelength_grid_um": wavelengths_um})
    state = replace(
        state,
        scene=Scene(star_jmag=4.0, altitude_deg=0.0, azimuth_deg=0.0),
        geometry=Geometry(
            n_apertures=2,
            aperture_positions_enu_m=aperture_positions_enu_m,
        ),
        field=Field(amplitudes=field, wavelength_grid_um=wavelengths_um),
        throughput=Throughput(
            wavelength_grid_um=wavelengths_um,
            transmission_fraction=(1.0, 1.0, 1.0),
            components={
                "delay_line_loss": {
                    "wavelength_grid_um": wavelengths_um,
                    "transmission_fraction": (1.0, 1.0, 1.0),
                    "per_aperture": delay_diag,
                },
            },
        ),
    )
    params = ChipOptimizationParams(
        n_modes=2,
        n_bright_ports=1,
        n_dark_ports=1,
        wl_design_um=1.55,
        delay_state_n_azimuths=2,
    )

    states = _delay_state_fields_from_state(
        state=state,
        field_array=field,
        params=params,
        wavelength_grid_um=wavelengths_um,
    )

    assert states is not None
    assert states.shape == (2, 2, 3, 1)
    scales = _delay_state_transport_scales_from_state(
        state=state,
        field_shape=field.shape,
        params=params,
        wavelength_grid_um=wavelengths_um,
    )
    assert scales is not None
    np.testing.assert_allclose(states[0, :, :, 0], scales[0].T, atol=1.0e-12)
    assert not np.allclose(states[0], field)
    assert not np.allclose(states[1], field)


def test_delay_state_transport_scales_are_absolute() -> None:
    wavelengths_um = (1.50, 1.55)
    aperture_positions_enu_m = np.array(
        [[0.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
        dtype=np.float64,
    )
    delay_diag = {
        "fiber_lengths_m": (1.0, 1.0),
        "n_eff": 1.0003,
        "phase_index_ref": 1.0003,
        "group_index_ref": 1.0004,
        "dispersion_ps_nm_km": 0.0,
        "attenuation_db_per_m": 0.0,
        "altitude_deg": 0.0,
        "azimuth_deg": 0.0,
        "fixed_delay_line_loss_db_per_telescope": (0.0, 6.0),
        "chromatic_control": "none",
        "reference_wavelength_um": 1.55,
        "per_mode_field_amplitude_applied": True,
    }
    state = SimulationState.empty({"wavelength_grid_um": wavelengths_um})
    state = replace(
        state,
        scene=Scene(star_jmag=4.0, altitude_deg=0.0, azimuth_deg=0.0),
        geometry=Geometry(
            n_apertures=2,
            aperture_positions_enu_m=aperture_positions_enu_m,
        ),
        field=Field(
            amplitudes=np.ones((2, 2, 1), dtype=np.complex128),
            wavelength_grid_um=wavelengths_um,
        ),
        throughput=Throughput(
            wavelength_grid_um=wavelengths_um,
            transmission_fraction=(1.0, 1.0),
            components={
                "delay_line_loss": {
                    "wavelength_grid_um": wavelengths_um,
                    "transmission_fraction": (1.0, 1.0),
                    "per_aperture": delay_diag,
                },
            },
        ),
    )
    params = ChipOptimizationParams(
        n_modes=2,
        n_bright_ports=1,
        n_dark_ports=1,
        wl_design_um=1.55,
        delay_state_n_azimuths=2,
    )

    scales = _delay_state_transport_scales_from_state(
        state=state,
        field_shape=(2, 2, 1),
        params=params,
        wavelength_grid_um=wavelengths_um,
    )
    current_scale = _delay_state_current_transport_scale_from_state(
        state=state,
        field_shape=(2, 2, 1),
        params=params,
        wavelength_grid_um=wavelengths_um,
    )

    assert scales is not None
    assert current_scale is not None
    assert scales.shape == (2, 2, 2)
    assert current_scale.shape == (2, 2)
    np.testing.assert_allclose(np.abs(scales[:, :, 1]), np.sqrt(10.0 ** (-0.6)))
    np.testing.assert_allclose(np.abs(current_scale[:, 1]), np.sqrt(10.0 ** (-0.6)))


def test_delay_state_transport_scales_keep_attenuator_chromatic_phase() -> None:
    wavelengths_um = (1.50, 1.55, 1.60)
    aperture_positions_enu_m = np.array(
        [[0.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
        dtype=np.float64,
    )
    delay_diag = {
        "fiber_lengths_m": (1.0, 1.0),
        "n_eff": 1.0003,
        "phase_index_ref": 1.0003,
        "group_index_ref": 1.0004,
        "dispersion_ps_nm_km": 1.0,
        "attenuation_db_per_m": 0.0,
        "altitude_deg": 0.0,
        "azimuth_deg": 0.0,
        "fixed_delay_line_loss_db_per_telescope": (2.0, 2.0),
        "fixed_delay_line_loss_mode": "attenuator",
        "chromatic_control": "multi_wavelength_group_delay",
        "reference_wavelength_um": 1.55,
        "per_mode_field_amplitude_applied": True,
    }
    state = SimulationState.empty({"wavelength_grid_um": wavelengths_um})
    state = replace(
        state,
        scene=Scene(star_jmag=4.0, altitude_deg=0.0, azimuth_deg=0.0),
        geometry=Geometry(
            n_apertures=2,
            aperture_positions_enu_m=aperture_positions_enu_m,
        ),
        field=Field(
            amplitudes=np.ones((2, 3, 1), dtype=np.complex128),
            wavelength_grid_um=wavelengths_um,
        ),
        throughput=Throughput(
            wavelength_grid_um=wavelengths_um,
            transmission_fraction=(1.0, 1.0, 1.0),
            components={
                "delay_line_loss": {
                    "wavelength_grid_um": wavelengths_um,
                    "transmission_fraction": (1.0, 1.0, 1.0),
                    "per_aperture": delay_diag,
                },
            },
        ),
    )
    params = ChipOptimizationParams(
        n_modes=2,
        n_bright_ports=1,
        n_dark_ports=1,
        wl_design_um=1.55,
        delay_state_n_azimuths=2,
    )

    scales = _delay_state_transport_scales_from_state(
        state=state,
        field_shape=(2, 3, 1),
        params=params,
        wavelength_grid_um=wavelengths_um,
    )

    assert scales is not None
    expected_amp = np.sqrt(10.0 ** (-0.2))
    np.testing.assert_allclose(np.abs(scales), expected_amp)
    assert not np.allclose(scales[0], scales[1])
    assert abs(scales[0, 0, 1] - expected_amp) > 1.0e-6


def test_optimizer_stack_averages_delay_states_by_field_amplitude() -> None:
    on_axis = np.ones((2, 2, 1), dtype=np.complex128)
    delay_states = np.stack(
        [
            on_axis,
            2.0 * on_axis,
        ],
        axis=0,
    )
    params = ChipOptimizationParams(
        n_modes=2,
        n_bright_ports=1,
        n_dark_ports=1,
        wl_design_um=1.25,
        delay_state_n_azimuths=2,
    )

    stack = _optimizer_null_field_stack(
        on_axis_field=on_axis,
        params=params,
        aperture_positions_pupil_m=None,
        wavelength_grid_um=(1.20, 1.30),
        sub_diameters_m=None,
        stellar_radius_rad=0.0,
        delay_state_fields=delay_states,
    )
    normalization_field = _normalization_field_array(on_axis, delay_states)

    assert stack.shape == (2, 2, 2, 1)
    np.testing.assert_allclose(stack[0], delay_states[0] / np.sqrt(2.0))
    np.testing.assert_allclose(stack[1], delay_states[1] / np.sqrt(2.0))
    assert np.sum(np.abs(normalization_field) ** 2) == pytest.approx(
        np.mean(np.sum(np.abs(delay_states) ** 2, axis=(1, 2, 3)))
    )


def test_loss_numpy_kernel_order_3_requires_positions() -> None:
    """kernel_order>0 with positions=None raises a clear error."""

    p = ChipOptimizationParams(
        n_modes=4, n_bright_ports=2, n_dark_ports=2,
        wl_design_um=1.25, kernel_order=3,
    )
    field = np.ones((4, 1, 1), dtype=np.complex128) / 2
    null_idx = _null_indices(p)
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 2 * np.pi, 4 * 3 + 4)
    with pytest.raises(ValueError, match="aperture_positions_pupil_m"):
        _loss_numpy(x, field_array=field, wavelength_grid_um=(1.25,),
                    params=p, null_idx=null_idx)


def test_loss_numpy_disk_samples_require_positions_for_resolved_star() -> None:
    p = ChipOptimizationParams(
        n_modes=4,
        n_bright_ports=2,
        n_dark_ports=2,
        wl_design_um=1.25,
        disk_n_rings=1,
        disk_n_azimuths=2,
    )
    field = np.ones((4, 1, 1), dtype=np.complex128) / 2
    null_idx = _null_indices(p)
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 2 * np.pi, 4 * 3 + 4)
    with pytest.raises(ValueError, match="aperture_positions_pupil_m"):
        _loss_numpy(
            x,
            field_array=field,
            wavelength_grid_um=(1.25,),
            params=p,
            null_idx=null_idx,
            sub_diameters_m=np.ones(4, dtype=np.float64),
            stellar_radius_rad=1.0e-7,
        )


def test_loss_numpy_kernel_order_3_matches_loss_shape() -> None:
    """With positions populated, kernel-order-3 loss runs and returns a finite scalar."""

    p = ChipOptimizationParams(
        n_modes=4, n_bright_ports=2, n_dark_ports=2,
        wl_design_um=1.25, kernel_order=3,
    )
    rng = np.random.default_rng(0)
    field = rng.normal(size=(4, 3, 1)) + 1j * rng.normal(size=(4, 3, 1))
    field = field.astype(np.complex128)
    positions = rng.normal(size=(4, 2))
    null_idx = _null_indices(p)
    x = rng.uniform(0, 2 * np.pi, 4 * 3 + 4)
    loss = _loss_numpy(
        x, field_array=field, wavelength_grid_um=(1.20, 1.25, 1.30),
        params=p, null_idx=null_idx, aperture_positions_pupil_m=positions,
    )
    assert np.isfinite(loss)
    assert loss > 0


def test_loss_numpy_kernel_order_sums_rows_over_wavelength_mean() -> None:
    p = ChipOptimizationParams(
        n_modes=4,
        n_bright_ports=2,
        n_dark_ports=2,
        wl_design_um=1.25,
        kernel_order=3,
    )
    rng = np.random.default_rng(5)
    field = rng.normal(size=(4, 3, 1)) + 1j * rng.normal(size=(4, 3, 1))
    field = field.astype(np.complex128)
    positions = rng.normal(size=(4, 2))
    null_idx = _null_indices(p)
    x = rng.uniform(0, 2 * np.pi, 4 * 3 + 4)

    loss = _loss_numpy(
        x,
        field_array=field,
        wavelength_grid_um=(1.20, 1.25, 1.30),
        params=p,
        null_idx=null_idx,
        aperture_positions_pupil_m=positions,
    )
    u_stack = _build_unitary_stack_numpy(
        params_vec=x,
        n_modes=p.n_modes,
        wavelength_grid_um=(1.20, 1.25, 1.30),
        wl_design_um=p.wl_design_um,
        dac_bits=p.dac_bits,
    )
    field_stack = _optimizer_null_field_stack(
        on_axis_field=field,
        params=p,
        aperture_positions_pupil_m=positions,
        wavelength_grid_um=(1.20, 1.25, 1.30),
        sub_diameters_m=None,
        stellar_radius_rad=0.0,
    )
    output = np.einsum("ijw,rjwp->riwp", u_stack, field_stack)
    leak_per_wl_per_row = np.sum(
        np.abs(output[:, null_idx, :, :]) ** 2,
        axis=(1, 3),
    )
    expected = float(np.sum(leak_per_wl_per_row, axis=0).mean())

    assert field_stack.shape[0] > 1
    assert loss == pytest.approx(expected, rel=1.0e-12)
    assert loss == pytest.approx(
        float(np.mean(leak_per_wl_per_row)) * field_stack.shape[0],
        rel=1.0e-12,
    )


def test_spectral_loss_p_norm_emphasizes_worst_wavelength() -> None:
    leakage = np.array([1.0e-6, 1.0e-6, 1.0e-3], dtype=np.float64)
    p_off = ChipOptimizationParams(
        n_modes=4,
        n_bright_ports=2,
        n_dark_ports=2,
        wl_design_um=1.25,
    )
    p_norm = ChipOptimizationParams(
        n_modes=4,
        n_bright_ports=2,
        n_dark_ports=2,
        wl_design_um=1.25,
        spectral_loss_mode="p_norm",
        spectral_loss_p=8.0,
    )

    mean_loss = _reduce_spectral_leakage_numpy(leakage, p_off)
    p_norm_loss = _reduce_spectral_leakage_numpy(leakage, p_norm)
    expected = float(
        np.max(leakage)
        * np.mean((leakage / np.max(leakage)) ** 8.0) ** (1.0 / 8.0)
    )

    assert mean_loss == pytest.approx(float(np.mean(leakage)))
    assert p_off.spectral_loss_mode == "off"
    assert p_norm_loss == pytest.approx(expected, rel=1.0e-12)
    assert p_norm_loss > mean_loss


def test_spectral_loss_edge_weighted_mean_emphasizes_band_edges() -> None:
    wavelength_grid_um = (1.20, 1.25, 1.30)
    leakage = np.array([1.0e-3, 1.0e-6, 1.0e-6], dtype=np.float64)
    params = ChipOptimizationParams(
        n_modes=4,
        n_bright_ports=2,
        n_dark_ports=2,
        wl_design_um=1.25,
        spectral_loss_mode="edge_weighted_mean",
        spectral_edge_weight_alpha=4.0,
        spectral_edge_weight_power=2.0,
    )

    weighted_loss = _reduce_spectral_leakage_numpy(
        leakage,
        params,
        wavelength_grid_um=wavelength_grid_um,
    )
    weights = np.array([5.0, 1.0, 5.0], dtype=np.float64)
    expected = float(np.sum(weights * leakage) / np.sum(weights))

    assert weighted_loss == pytest.approx(expected, rel=1.0e-12)
    assert weighted_loss > float(np.mean(leakage))


def test_target_null_depth_uses_reported_null_convention() -> None:
    field = np.full((4, 2, 1), 2.0 + 0.0j, dtype=np.complex128)
    loss = 8.0

    normalized = _normalized_loss_for_target(loss, field)
    reported = _reported_null_depth_for_target(loss, field)

    assert normalized == pytest.approx(loss / np.sum(np.abs(field) ** 2))
    assert reported == pytest.approx(loss / 16.0)
    assert _loss_meets_target(loss, field, target_null_depth=0.60)
    assert not _loss_meets_target(loss, field, target_null_depth=0.40)


def test_reported_null_depth_uses_per_wavelength_row_convention() -> None:
    field = np.full((4, 2, 1), 2.0 + 0.0j, dtype=np.complex128)
    loss = 8.0

    reported = _reported_null_depth_for_target(
        loss,
        field,
        n_objective_rows=3,
    )

    mean_input_power_per_wavelength = 4 * 4.0
    assert reported == pytest.approx(loss / (3.0 * mean_input_power_per_wavelength))


def test_loss_numpy_disk_samples_runs_when_positions_are_present() -> None:
    p = ChipOptimizationParams(
        n_modes=4,
        n_bright_ports=2,
        n_dark_ports=2,
        wl_design_um=1.25,
        disk_n_rings=1,
        disk_n_azimuths=2,
    )
    rng = np.random.default_rng(0)
    field = np.ones((4, 2, 1), dtype=np.complex128) / 2
    positions = rng.normal(size=(4, 2))
    null_idx = _null_indices(p)
    x = rng.uniform(0, 2 * np.pi, 4 * 3 + 4)
    loss = _loss_numpy(
        x,
        field_array=field,
        wavelength_grid_um=(1.20, 1.30),
        params=p,
        null_idx=null_idx,
        aperture_positions_pupil_m=positions,
        sub_diameters_m=np.ones(4, dtype=np.float64),
        stellar_radius_rad=1.0e-7,
    )
    assert np.isfinite(loss)
    assert loss > 0


def test_scipy_stage_with_kernel_order_3_runs_end_to_end(tmp_path: Path) -> None:
    """The scipy chip stage accepts kernel_order=3 and produces an optimum."""

    params = ChipOptimizationParams(
        n_modes=4, n_bright_ports=2, n_dark_ports=2,
        wl_design_um=1.25, target_null_depth=1e-12,
        max_iter=200, n_restarts=2, seed=0, kernel_order=3,
    )
    ctx = RunContext(
        rng=np.random.default_rng(0), config_hash="abc",
        sweep_coord_tuple=(), instance_id="chip_opt", run_dir=tmp_path,
    )
    stage = ChipOptimizationScipy(params=params.model_dump(), context=ctx)

    wl_grid = (1.20, 1.25, 1.30)
    state = SimulationState.empty(grid={"wavelength_grid_um": wl_grid})
    amps = np.ones((4, 3, 1), dtype=np.complex128) / 2
    positions = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    state = replace(state, field=Field(amplitudes=amps, wavelength_grid_um=wl_grid))
    state = replace(state, geometry=Geometry(aperture_positions_pupil_m=positions))

    out = stage.apply(state)
    payload = next(iter(out.results.components.values()))
    # The optimizer should at least produce a candidate; the value will be
    # higher than on-axis-only because we're enforcing 5 additional
    # derivative constraints on a 4-mode mesh (which has only 16 DOFs).
    assert payload["optimization_iterations"] > 0
    assert np.isfinite(payload["optimization_loss_final"])


def test_scipy_stage_with_disk_samples_runs_end_to_end(tmp_path: Path) -> None:
    params = ChipOptimizationParams(
        n_modes=4,
        n_bright_ports=2,
        n_dark_ports=2,
        wl_design_um=1.25,
        target_null_depth=1e-12,
        max_iter=40,
        n_restarts=1,
        seed=0,
        disk_n_rings=1,
        disk_n_azimuths=2,
    )
    ctx = RunContext(
        rng=np.random.default_rng(0), config_hash="abc",
        sweep_coord_tuple=(), instance_id="chip_opt", run_dir=tmp_path,
    )
    stage = ChipOptimizationScipy(params=params.model_dump(), context=ctx)

    wl_grid = (1.20, 1.25, 1.30)
    state = SimulationState.empty(grid={"wavelength_grid_um": wl_grid})
    amps = np.ones((4, 3, 1), dtype=np.complex128) / 2
    positions = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    state = replace(
        state,
        scene=Scene(star_angular_diameter_mas=1.0),
        field=Field(amplitudes=amps, wavelength_grid_um=wl_grid),
        geometry=Geometry(
            aperture_positions_pupil_m=positions,
            effective_collecting_area_m2=4.0,
        ),
    )

    out = stage.apply(state)
    payload = next(iter(out.results.components.values()))
    assert payload["optimization_iterations"] > 0
    assert np.isfinite(payload["optimization_loss_final"])


def test_chip_opt_quantize_no_wrap_only_dac_grid() -> None:
    """chip_optimization._quantize_phases ROUNDS to the DAC grid but does
    NOT wrap to ``[0, 2*pi)``.

    Wrapping happens at the call site that also applies the chromatic
    rescaling (``_build_unitary_stack_numpy`` for the scipy backend, the
    kernel's own ``_quantize_phases`` for the deployment side). The
    torch backend rounds its final iterate to the DAC grid via
    this helper without wrapping, because they publish an
    ``optimal_u_stack`` built from the same unwrapped phases the
    autograd objective minimized — wrapping here would shift the
    deployed chromatic chip away from the optimizer's operating point.
    """

    rng = np.random.default_rng(7)
    # BFGS-perturbed iterate that ranges outside [0, 2*pi).
    raw = rng.uniform(-2.0 * np.pi, 4.0 * np.pi, size=24)
    dac_bits = 8

    q_opt = _quantize_phases(raw, dac_bits)

    # On the DAC grid:
    step = 2.0 * np.pi / (1 << dac_bits)
    np.testing.assert_allclose(q_opt, step * np.round(raw / step), atol=0.0, rtol=0.0)
    # NOT wrapped: negative inputs stay negative, super-2pi stays super-2pi.
    assert q_opt.min() < 0.0
    assert q_opt.max() > 2.0 * np.pi


def test_chip_opt_numpy_stack_matches_kernel_offdesign() -> None:
    """The scipy backend's numpy reference unitary must match the kernel
    stack at off-design wavelengths once the optimizer chooses negative
    phases. The wrap that makes this true lives in
    ``_build_unitary_stack_numpy`` (after DAC quantization, before the
    per-wavelength rescaling), mirroring the kernel's own
    quantize-then-wrap-then-scale path."""

    from nullsim.stages.chip.kernel_mzi_mesh import (
        _build_unitary_stack as kernel_build_stack,
        _quantize_phases as kernel_quantize,
    )
    from nullsim.stages.chip_optimization import _clements_order

    rng = np.random.default_rng(11)
    n_modes = 4
    n_mzi = n_modes * (n_modes - 1) // 2
    params_vec = rng.uniform(-2.0 * np.pi, 4.0 * np.pi, size=2 * n_mzi + n_modes)
    wl_design_um = 1.25
    wl_grid_um = (1.10, 1.25, 1.40)
    dac_bits = 8

    stack_opt = _build_unitary_stack_numpy(
        params_vec=params_vec,
        n_modes=n_modes,
        wavelength_grid_um=wl_grid_um,
        wl_design_um=wl_design_um,
        dac_bits=dac_bits,
    )

    # Drive the kernel through its own quantize (which wraps), then build.
    kernel_quantized = kernel_quantize(params_vec, dac_bits)
    theta = kernel_quantized[0:n_mzi]
    phi = kernel_quantized[n_mzi : 2 * n_mzi]
    alpha = kernel_quantized[2 * n_mzi : 2 * n_mzi + n_modes]
    order = _clements_order(n_modes)
    stack_kernel = kernel_build_stack(
        theta=theta, phi=phi, alpha=alpha, order=order, n_modes=n_modes,
        wavelength_grid_um=wl_grid_um, wl_design_um=wl_design_um,
        rescale_per_wavelength=True,
    )

    # All wavelengths — including off-design — must agree to numerical
    # precision. Without the wrap inside _build_unitary_stack_numpy, the
    # off-design slices diverge by O(1).
    np.testing.assert_allclose(stack_opt, stack_kernel, atol=1e-12, rtol=1e-12)
