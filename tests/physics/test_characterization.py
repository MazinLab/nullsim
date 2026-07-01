"""Tests for known-position characterization physics helpers."""

from __future__ import annotations

import numpy as np
import pytest

from nullsim.physics.characterization import (
    _candidate_selection_key,
    _snr_proxy,
    build_unitary_from_params,
    build_unitary_stack_from_params,
    disk_sample_stellar_stack,
    optimize_characterization,
    spectro_disk_leak,
    spectro_disk_leak_from_stack,
)
from nullsim.physics.characterization_torch import (
    _quantize_native_params,
    _spectro_row_space_canonical_start,
    _systematic_snr_proxy_numpy,
)
from nullsim.physics.fringe_observability import differential_piston_basis


def test_candidate_selection_key_prioritizes_feasibility_then_null_margin() -> None:
    feasible = _candidate_selection_key(
        objective_value=10.0,
        constraint_margin=0.0,
        feasibility_tol=1.0e-8,
    )
    infeasible_high_snr = _candidate_selection_key(
        objective_value=-100.0,
        constraint_margin=-1.0e-3,
        feasibility_tol=1.0e-8,
    )
    infeasible_lower_snr = _candidate_selection_key(
        objective_value=100.0,
        constraint_margin=-1.0e-6,
        feasibility_tol=1.0e-8,
    )

    assert feasible < infeasible_lower_snr
    assert infeasible_lower_snr < infeasible_high_snr


def test_disk_sample_stellar_stack_adds_real_imag_disk_rows() -> None:
    positions = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    base = np.ones(3, dtype=np.float64)
    diameters = np.ones(3, dtype=np.float64)

    stack = disk_sample_stellar_stack(
        positions_m=positions,
        base_amplitudes=base,
        wavelength_m=1.25e-6,
        sub_diameters_m=diameters,
        stellar_radius_rad=0.5 / 206265e3,
        n_rings=2,
        n_azimuths=3,
        kernel_order=0,
    )

    assert stack.shape[0] > 1
    assert stack.shape[1] == 3
    norms = np.linalg.norm(stack, axis=1)
    np.testing.assert_allclose(norms, norms[0], rtol=1e-12)


def test_spectro_disk_leak_zero_for_unresolved_star() -> None:
    unitary = np.eye(2, dtype=np.complex128)
    leak = spectro_disk_leak(
        unitary=unitary,
        positions_m=np.zeros((2, 2)),
        base_amplitudes=np.ones(2),
        spectro_port_indices=np.array([1], dtype=np.intp),
        wavelength_m=1.25e-6,
        sub_diameters_m=np.ones(2),
        stellar_radius_mas=0.0,
    )
    assert leak == 0.0


def test_spectro_disk_leak_from_stack_uses_deployed_wavelength_slices() -> None:
    identity = np.eye(2, dtype=np.complex128)
    swap = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    stack = np.stack((identity, swap), axis=2)
    positions = np.array([[0.0, 0.0], [6.0, 0.0]], dtype=np.float64)
    base = np.ones(2, dtype=np.float64)
    diameters = np.ones(2, dtype=np.float64)
    wavelengths = np.array([1.2e-6, 1.3e-6], dtype=np.float64)

    _, per_wl = spectro_disk_leak_from_stack(
        u_stack=stack,
        positions_m=positions,
        base_amplitudes=base,
        spectro_port_indices=np.array([1], dtype=np.intp),
        wavelengths_m=wavelengths,
        sub_diameters_m=diameters,
        stellar_radius_mas=0.5,
        n_radial=4,
        n_pa=5,
    )

    expected = [
        spectro_disk_leak(
            unitary=stack[:, :, index],
            positions_m=positions,
            base_amplitudes=base,
            spectro_port_indices=np.array([1], dtype=np.intp),
            wavelength_m=float(wavelength),
            sub_diameters_m=diameters,
            stellar_radius_mas=0.5,
            n_radial=4,
            n_pa=5,
        )
        for index, wavelength in enumerate(wavelengths)
    ]
    np.testing.assert_allclose(per_wl, expected, rtol=0.0, atol=0.0)


def test_snr_proxy_saturates_at_null_floor() -> None:
    # Leak below the configured floor is worthless at deployment (the
    # realized piston floor dominates), so the optimizer reward must
    # saturate there: an unbounded sqrt reward traded planet throughput
    # for un-cashable null depth (2026-06-10 follow-up).
    floor_proxy = _snr_proxy(
        planet_throughput=0.1,
        stellar_leak=1.0e-6,
        null_floor=1.0e-6,
    )
    deeper_proxy = _snr_proxy(
        planet_throughput=0.1,
        stellar_leak=1.0e-8,
        null_floor=1.0e-6,
    )
    leaky_proxy = _snr_proxy(
        planet_throughput=0.1,
        stellar_leak=1.0e-4,
        null_floor=1.0e-6,
    )

    assert floor_proxy == pytest.approx(0.1)
    assert deeper_proxy == pytest.approx(floor_proxy)
    assert leaky_proxy < floor_proxy


def test_torch_snr_proxy_matches_numpy_saturation() -> None:
    torch = pytest.importorskip("torch")
    from nullsim.physics.characterization_torch import _t_snr_proxy

    for leak in (1.0e-8, 1.0e-6, 1.0e-4):
        torch_value = float(
            _t_snr_proxy(
                torch.tensor(0.1, dtype=torch.float64),
                torch.tensor(leak, dtype=torch.float64),
                1.0e-6,
                torch_mod=torch,
            )
        )
        numpy_value = _snr_proxy(
            planet_throughput=0.1,
            stellar_leak=leak,
            null_floor=1.0e-6,
        )
        assert torch_value == pytest.approx(numpy_value)


def test_systematic_snr_proxy_clamps_at_null_floor() -> None:
    at_floor = _systematic_snr_proxy_numpy(
        planet_throughput=0.1,
        stellar_leak=1.0e-6,
        null_floor=1.0e-6,
    )
    deeper = _systematic_snr_proxy_numpy(
        planet_throughput=0.1,
        stellar_leak=1.0e-8,
        null_floor=1.0e-6,
    )
    leaky = _systematic_snr_proxy_numpy(
        planet_throughput=0.1,
        stellar_leak=1.0e-4,
        null_floor=1.0e-6,
    )

    assert deeper == pytest.approx(at_floor)
    assert leaky < at_floor


def test_torch_native_quantization_preserves_fixed_path_phase_wraps() -> None:
    raw = np.array([-0.1, 0.0, 2.0 * np.pi + 0.1, 17.0 * np.pi], dtype=np.float64)

    quantized = _quantize_native_params(raw, dac_bits=20)

    assert quantized[0] < 0.0
    assert quantized[2] > 2.0 * np.pi
    assert quantized[3] > 2.0 * np.pi
    np.testing.assert_allclose(quantized, raw, atol=2.0 * np.pi / (1 << 20))


def test_spectro_row_space_start_nulls_star_and_routes_planet() -> None:
    star = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    planet = np.array([1.0, -1.0], dtype=np.complex128) / np.sqrt(2.0)

    params = _spectro_row_space_canonical_start(
        n_modes=2,
        spectro_idx=np.array([1], dtype=np.intp),
        bright_idx=np.array([], dtype=np.intp),
        stellar_fields=(star,),
        planet_fields=(planet,),
        stellar_weights=None,
        planet_weights=None,
        initial_params=None,
        build_unitary_from_params=build_unitary_from_params,
    )

    assert params is not None
    unitary = build_unitary_from_params(params, 2)
    star_out = unitary @ star
    planet_out = unitary @ planet
    assert float(np.abs(star_out[1]) ** 2) < 1.0e-12
    assert float(np.abs(planet_out[1]) ** 2) > 0.99


def test_optimize_characterization_returns_feasible_two_mode_solution() -> None:
    star = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    planet = np.array([1.0, -1.0], dtype=np.complex128) / np.sqrt(2.0)

    result = optimize_characterization(
        stellar_field=star,
        planet_field=planet,
        n_modes=2,
        dark_port_indices=np.array([], dtype=np.intp),
        spectro_port_indices=np.array([1], dtype=np.intp),
        bright_port_indices=np.array([0], dtype=np.intp),
        null_floor=1.0e-4,
        n_restarts=4,
        seed=3,
        max_iter=300,
        backend="scipy",
    )

    assert result.spectro_null_depth < 1.01e-4
    assert result.spectro_throughput > 0.5
    assert result.n_restarts_completed >= 1


def test_optimize_characterization_torch_parallel_restarts_complete() -> None:
    pytest.importorskip("torch")

    star = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    planet = np.array([1.0, -1.0], dtype=np.complex128) / np.sqrt(2.0)

    result = optimize_characterization(
        stellar_field=star,
        planet_field=planet,
        n_modes=2,
        dark_port_indices=np.array([], dtype=np.intp),
        spectro_port_indices=np.array([1], dtype=np.intp),
        bright_port_indices=np.array([0], dtype=np.intp),
        null_floor=1.0e-4,
        n_restarts=2,
        seed=3,
        max_iter=2,
        backend="torch",
        enforce_null_constraint=False,
        n_workers=2,
        wl_design_m=1.0e-6,
    )

    assert result.n_restarts_completed >= 2
    assert np.all(np.isfinite(result.params))


def test_optimize_characterization_uses_selection_grid_for_restart_choice() -> None:
    design_star = np.array([1.0, 0.0], dtype=np.complex128)
    design_planet = np.array([0.0, 1.0], dtype=np.complex128)
    full_band_star = np.array([0.0, 1.0], dtype=np.complex128)
    full_band_planet = np.array([1.0, 0.0], dtype=np.complex128)
    identity_params = np.zeros(4, dtype=np.float64)
    swap_params = np.array([np.pi / 2.0, 0.0, 0.0, 0.0], dtype=np.float64)

    result = optimize_characterization(
        stellar_field=design_star,
        planet_field=design_planet,
        n_modes=2,
        dark_port_indices=np.array([], dtype=np.intp),
        spectro_port_indices=np.array([1], dtype=np.intp),
        bright_port_indices=np.array([0], dtype=np.intp),
        null_floor=1.0e-4,
        n_restarts=0,
        seed=3,
        max_iter=1,
        backend="scipy",
        initial_params=(identity_params, swap_params),
        throughput_weight=0.0,
        bright_weight=0.0,
        snr_proxy_weight=0.0,
        enforce_null_constraint=False,
        wl_design_m=1.0e-6,
        selection_wavelengths_m=np.array([1.0e-6], dtype=np.float64),
        selection_stellar_fields_per_wl=(full_band_star,),
        selection_planet_fields_per_wl=(full_band_planet,),
        selection_stellar_weights_per_wl=np.array([1.0], dtype=np.float64),
        selection_planet_weights_per_wl=np.array([1.0], dtype=np.float64),
    )

    assert result.n_restarts_completed == 2
    np.testing.assert_allclose(result.params, swap_params, atol=1.0e-9)


def test_optimize_characterization_can_regularize_tracker_observability() -> None:
    pytest.importorskip("torch")

    star = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    planet = np.array([1.0, -1.0], dtype=np.complex128) / np.sqrt(2.0)

    result = optimize_characterization(
        stellar_field=star,
        planet_field=planet,
        n_modes=2,
        dark_port_indices=np.array([], dtype=np.intp),
        spectro_port_indices=np.array([1], dtype=np.intp),
        bright_port_indices=np.array([0], dtype=np.intp),
        null_floor=1.0e-4,
        n_restarts=3,
        seed=3,
        max_iter=300,
        backend="torch",
        tracker_piston_basis=differential_piston_basis(2),
        tracker_field=star,
        tracker_wavelength_m=1.25e-6,
        tracker_observability_weight=1.0e-3,
        tracker_observability_crlb_weight=1.0e-4,
    )

    assert result.spectro_null_depth < 1.01e-4
    assert result.tracker_singular_values_per_nm is not None
    assert result.tracker_singular_values_per_nm.shape == (1,)
    assert np.all(np.isfinite(result.tracker_singular_values_per_nm))
    assert result.tracker_crlb_rms_nm is not None
    assert np.isfinite(result.tracker_crlb_rms_nm)


def test_torch_broadband_characterization_accepts_canonical_warm_start() -> None:
    pytest.importorskip("torch")

    star = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    planet = np.array([1.0, -1.0], dtype=np.complex128) / np.sqrt(2.0)
    warm_start = np.array([np.pi / 4.0, np.pi, 0.0, 0.0], dtype=np.float64)
    wavelengths_m = np.array([1.25e-6, 1.25e-6], dtype=np.float64)

    result = optimize_characterization(
        stellar_field=star,
        planet_field=planet,
        n_modes=2,
        dark_port_indices=np.array([], dtype=np.intp),
        spectro_port_indices=np.array([1], dtype=np.intp),
        bright_port_indices=np.array([0], dtype=np.intp),
        null_floor=1.0e-4,
        n_restarts=0,
        seed=3,
        max_iter=30,
        backend="torch",
        initial_params=(warm_start,),
        wl_design_m=1.25e-6,
        wavelengths_m=wavelengths_m,
        stellar_fields_per_wl=(star, star),
        planet_fields_per_wl=(planet, planet),
        tracker_piston_basis=differential_piston_basis(2),
        tracker_field=star,
        tracker_wavelength_m=1.25e-6,
        tracker_observability_weight=1.0e-3,
        tracker_observability_crlb_weight=1.0e-4,
        tracker_observability_crlb_target_nm=1.0e6,
        tracker_observability_crlb_constraint_nm=2.0e6,
    )

    assert result.optimization_success
    assert result.n_restarts_completed == 1
    assert result.spectro_null_depth < 1.01e-4
    assert result.spectro_throughput > 0.5
    assert result.tracker_singular_values_per_nm is not None


def test_torch_broadband_characterization_margin_matches_deployed_stack() -> None:
    pytest.importorskip("torch")

    star = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    planet = np.array([1.0, -1.0], dtype=np.complex128) / np.sqrt(2.0)
    warm_start = np.array([np.pi / 4.0, np.pi, 0.0, 0.0], dtype=np.float64)
    wl_design_m = 1.25e-6
    wavelengths_m = np.array([1.18e-6, 1.25e-6, 1.32e-6], dtype=np.float64)
    weights = np.array([0.2, 0.5, 0.3], dtype=np.float64)
    null_floor = 1.0e-3

    result = optimize_characterization(
        stellar_field=star,
        planet_field=planet,
        n_modes=2,
        dark_port_indices=np.array([], dtype=np.intp),
        spectro_port_indices=np.array([1], dtype=np.intp),
        bright_port_indices=np.array([0], dtype=np.intp),
        null_floor=null_floor,
        n_restarts=0,
        seed=3,
        max_iter=80,
        backend="torch",
        initial_params=(warm_start,),
        wl_design_m=wl_design_m,
        wavelengths_m=wavelengths_m,
        stellar_fields_per_wl=(star, star, star),
        planet_fields_per_wl=(planet, planet, planet),
        stellar_weights_per_wl=weights,
        planet_weights_per_wl=weights,
    )

    assert result.optimal_u_stack is not None
    stack = result.optimal_u_stack
    deployed_leak = 0.0
    total_star = float(np.sum(np.abs(star) ** 2))
    for index, weight in enumerate(weights / np.sum(weights)):
        output = stack[:, :, index] @ star
        deployed_leak += float(weight) * (
            float(np.sum(np.abs(output[[1]]) ** 2)) / total_star
        )

    assert result.constraint_margin == pytest.approx(
        null_floor - deployed_leak,
        rel=1.0e-9,
        abs=1.0e-12,
    )


def test_preserve_bright_rows_keeps_inherited_tracker_outputs() -> None:
    rng = np.random.default_rng(12)
    n_modes = 4
    n_mzi = n_modes * (n_modes - 1) // 2
    warm_start = rng.uniform(0.0, 2.0 * np.pi, 2 * n_mzi + n_modes)
    inherited = build_unitary_from_params(warm_start, n_modes)
    star = np.ones(n_modes, dtype=np.complex128)
    planet = np.exp(1j * np.linspace(0.0, 1.0, n_modes))

    result = optimize_characterization(
        stellar_field=star,
        planet_field=planet,
        n_modes=n_modes,
        dark_port_indices=np.array([2], dtype=np.intp),
        spectro_port_indices=np.array([3], dtype=np.intp),
        bright_port_indices=np.array([0, 1], dtype=np.intp),
        null_floor=1.0e-3,
        n_restarts=1,
        seed=4,
        max_iter=80,
        backend="scipy",
        initial_params=(warm_start,),
        preserve_bright_rows=True,
        deployment_wavelengths_m=np.array(
            [1.2e-6, 1.25e-6, 1.3e-6],
            dtype=np.float64,
        ),
        wl_design_m=1.25e-6,
    )

    retuned = build_unitary_from_params(result.params, n_modes)
    np.testing.assert_allclose(retuned[:2], inherited[:2], atol=1.0e-10)
    assert result.optimal_u_stack is not None
    deployed_stack = build_unitary_stack_from_params(
        params=result.params,
        n_modes=n_modes,
        wavelengths_m=np.array([1.2e-6, 1.25e-6, 1.3e-6], dtype=np.float64),
        wl_design_m=1.25e-6,
    )
    np.testing.assert_allclose(
        result.optimal_u_stack,
        deployed_stack,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        result.optimal_u_stack[:2, :, 1],
        inherited[:2],
        atol=1.0e-10,
    )
    assert result.n_restarts_completed == 2


def test_preserve_bright_rows_stack_matches_quantized_physical_mesh() -> None:
    rng = np.random.default_rng(14)
    n_modes = 4
    n_mzi = n_modes * (n_modes - 1) // 2
    warm_start = rng.uniform(0.0, 2.0 * np.pi, 2 * n_mzi + n_modes)
    star = np.ones(n_modes, dtype=np.complex128)
    planet = np.exp(1j * np.linspace(0.0, 1.0, n_modes))
    wavelengths_m = np.array([1.2e-6, 1.25e-6, 1.3e-6], dtype=np.float64)

    result = optimize_characterization(
        stellar_field=star,
        planet_field=planet,
        n_modes=n_modes,
        dark_port_indices=np.array([2], dtype=np.intp),
        spectro_port_indices=np.array([3], dtype=np.intp),
        bright_port_indices=np.array([0, 1], dtype=np.intp),
        null_floor=1.0e-3,
        n_restarts=0,
        seed=4,
        max_iter=50,
        backend="scipy",
        initial_params=(warm_start,),
        preserve_bright_rows=True,
        deployment_wavelengths_m=wavelengths_m,
        wl_design_m=1.25e-6,
        dac_bits=6,
    )

    assert result.optimal_u_stack is not None
    deployed_stack = build_unitary_stack_from_params(
        params=result.params,
        n_modes=n_modes,
        wavelengths_m=wavelengths_m,
        wl_design_m=1.25e-6,
        dac_bits=None,
    )
    np.testing.assert_allclose(result.optimal_u_stack, deployed_stack, atol=1.0e-10)


def test_preserve_bright_rows_uses_physical_torch_optimizer_when_requested() -> None:
    pytest.importorskip("torch")

    star = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    planet = np.array([1.0, -1.0], dtype=np.complex128) / np.sqrt(2.0)
    warm_start = np.array([np.pi / 4.0, np.pi, 0.0, 0.0], dtype=np.float64)
    wavelengths_m = np.array([1.24e-6, 1.25e-6, 1.26e-6], dtype=np.float64)

    result = optimize_characterization(
        stellar_field=star,
        planet_field=planet,
        n_modes=2,
        dark_port_indices=np.array([], dtype=np.intp),
        spectro_port_indices=np.array([1], dtype=np.intp),
        bright_port_indices=np.array([0], dtype=np.intp),
        null_floor=1.0e-3,
        n_restarts=0,
        seed=3,
        max_iter=40,
        backend="torch",
        initial_params=(warm_start,),
        preserve_bright_rows=True,
        wl_design_m=1.25e-6,
        wavelengths_m=wavelengths_m,
        stellar_fields_per_wl=(star, star, star),
        planet_fields_per_wl=(planet, planet, planet),
        deployment_wavelengths_m=wavelengths_m,
        tracker_piston_basis=differential_piston_basis(2),
        tracker_field=star,
        tracker_wavelength_m=1.25e-6,
        tracker_observability_crlb_constraint_nm=2.0e6,
    )

    assert result.optimization_success
    assert result.optimal_u_stack is not None
    assert result.optimal_u_stack.shape == (2, 2, 3)
    assert result.tracker_crlb_rms_nm is not None
    assert np.isfinite(result.tracker_crlb_rms_nm)


def test_torch_preserve_bright_rows_can_start_fresh() -> None:
    pytest.importorskip("torch")

    star = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    planet = np.array([1.0, -1.0], dtype=np.complex128) / np.sqrt(2.0)

    result = optimize_characterization(
        stellar_field=star,
        planet_field=planet,
        n_modes=2,
        dark_port_indices=np.array([], dtype=np.intp),
        spectro_port_indices=np.array([1], dtype=np.intp),
        bright_port_indices=np.array([0], dtype=np.intp),
        null_floor=1.0e-3,
        n_restarts=1,
        seed=7,
        max_iter=30,
        backend="torch",
        initial_params=(),
        preserve_bright_rows=True,
        wl_design_m=1.25e-6,
        wavelengths_m=np.array([1.25e-6], dtype=np.float64),
        stellar_fields_per_wl=(star,),
        planet_fields_per_wl=(planet,),
        deployment_wavelengths_m=np.array([1.25e-6], dtype=np.float64),
        tracker_piston_basis=differential_piston_basis(2),
        tracker_field=star,
        tracker_wavelength_m=1.25e-6,
        tracker_observability_crlb_constraint_nm=2.0e6,
    )

    assert result.n_restarts_completed >= 1
    assert result.optimal_u_stack is not None
    assert result.tracker_crlb_rms_nm is not None
