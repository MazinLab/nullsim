from __future__ import annotations

import numpy as np
import pytest

from nullsim.physics import (
    DualPolField,
    HWPConfig,
    PBSConfig,
    PMFiberConfig,
    apply_hwp,
    apply_jones,
    apply_pbs,
    half_wave_plate,
    hwp_retardance,
    identity_jones,
    jones_hwp,
    jones_pm_fiber,
    jones_to_stokes,
    linear_polarizer,
    pdi_imperfection_factor,
    pm_fiber_leakage_power,
    pm_fiber_per_at_length,
    polarizing_beam_splitter_arm,
    quarter_wave_plate,
    random_birefringence,
    rotator,
    waveplate,
)


def _assert_same_state_up_to_global_phase(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    atol: float = 1.0e-12,
) -> None:
    actual_norm = np.linalg.norm(actual)
    expected_norm = np.linalg.norm(expected)
    assert actual_norm == pytest.approx(expected_norm, abs=atol)
    if expected_norm == 0.0:
        np.testing.assert_allclose(actual, expected, atol=atol)
        return

    overlap = np.vdot(expected / expected_norm, actual / actual_norm)
    assert abs(overlap) == pytest.approx(1.0, abs=atol)


def test_linear_polarizer_zero_blocks_vertical() -> None:
    vertical = np.array([0.0, 1.0], dtype=np.complex128)

    output = linear_polarizer(0.0) @ vertical

    np.testing.assert_allclose(output, [0.0, 0.0], atol=1.0e-12)


def test_linear_polarizer_projects_onto_transmission_axis() -> None:
    angle_rad = np.pi / 3.0
    axis = np.array([np.cos(angle_rad), np.sin(angle_rad)], dtype=np.complex128)
    input_vector = np.array([1.0 + 0.25j, -0.5 + 0.75j], dtype=np.complex128)
    expected = axis * np.vdot(axis, input_vector)

    output = apply_jones(linear_polarizer(angle_rad), input_vector)

    np.testing.assert_allclose(output, expected, atol=1.0e-12)


def test_rotator_pi_over_two_rotates_horizontal_to_vertical() -> None:
    horizontal = np.array([1.0, 0.0], dtype=np.complex128)

    output = apply_jones(rotator(np.pi / 2.0), horizontal)

    np.testing.assert_allclose(output, [0.0, 1.0], atol=1.0e-12)


def test_waveplate_uses_requested_similarity_transform() -> None:
    retardance_rad = np.pi / 3.0
    fast_axis_rad = np.pi / 7.0
    expected = (
        rotator(fast_axis_rad)
        @ np.diag([1.0 + 0.0j, np.exp(1j * retardance_rad)])
        @ rotator(-fast_axis_rad)
    )

    matrix = waveplate(retardance_rad, fast_axis_rad)

    np.testing.assert_allclose(matrix, expected, atol=1.0e-12)


def test_quarter_wave_plate_zero_adds_phase_to_vertical_component() -> None:
    vertical = np.array([0.0, 1.0], dtype=np.complex128)

    output = apply_jones(quarter_wave_plate(0.0), vertical)

    np.testing.assert_allclose(output, [0.0, 1.0j], atol=1.0e-12)


def test_half_wave_plate_at_45deg_flips_vertical_to_horizontal() -> None:
    vertical = np.array([0.0, 1.0], dtype=np.complex128)
    expected = np.array([1.0, 0.0], dtype=np.complex128)

    output = apply_jones(half_wave_plate(np.pi / 4.0), vertical)

    _assert_same_state_up_to_global_phase(output, expected)


def test_apply_jones_broadcasts_over_leading_axes() -> None:
    fields = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [1.0j, -1.0j]],
        ],
        dtype=np.complex64,
    )

    output = apply_jones(rotator(np.pi / 2.0), fields)

    assert output.shape == fields.shape
    assert output.dtype == np.complex128
    np.testing.assert_allclose(output[..., 0], -fields[..., 1], atol=1.0e-12)
    np.testing.assert_allclose(output[..., 1], fields[..., 0], atol=1.0e-12)


def test_jones_to_stokes_horizontal_returns_iqvu_order() -> None:
    stokes = jones_to_stokes(np.array([1.0, 0.0], dtype=np.complex128))

    np.testing.assert_allclose(stokes, [1.0, 1.0, 0.0, 0.0], atol=1.0e-12)


def test_jones_to_stokes_batch_shape_and_circular_sign() -> None:
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    fields = np.array(
        [
            [0.0, 1.0],
            [inv_sqrt2, inv_sqrt2],
            [inv_sqrt2, 1j * inv_sqrt2],
        ],
        dtype=np.complex64,
    )

    stokes = jones_to_stokes(fields)

    assert stokes.shape == (3, 4)
    assert stokes.dtype == np.float64
    np.testing.assert_allclose(
        stokes,
        [
            [1.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
        ],
        atol=1.0e-12,
    )


def test_constructors_return_complex128_for_float32_angles() -> None:
    matrices = (
        linear_polarizer(np.float32(0.0)),
        waveplate(np.float32(np.pi / 3.0), np.float32(0.2)),
        half_wave_plate(np.float32(np.pi / 4.0)),
        quarter_wave_plate(np.float32(np.pi / 4.0)),
        rotator(np.float32(0.0)),
    )

    for matrix in matrices:
        assert matrix.dtype == np.complex128


def test_hwp_retardance_is_pi_at_design_wavelength_and_matches_rms() -> None:
    cfg = HWPConfig()
    wavelengths_m = np.linspace(
        cfg.design_wavelength_m * (1.0 - cfg.band_half_width_lambda),
        cfg.design_wavelength_m * (1.0 + cfg.band_half_width_lambda),
        1001,
    )

    design_retardance = hwp_retardance(
        cfg.design_wavelength_m,
        cfg.design_wavelength_m,
        cfg.retardance_error_rms_rad,
        cfg.band_half_width_lambda,
    )
    retardance = hwp_retardance(
        wavelengths_m,
        cfg.design_wavelength_m,
        cfg.retardance_error_rms_rad,
        cfg.band_half_width_lambda,
    )

    assert design_retardance == pytest.approx(np.pi, abs=1.0e-12)
    assert retardance.shape == wavelengths_m.shape
    rms_error = float(np.sqrt(np.mean((retardance - np.pi) ** 2)))
    assert rms_error == pytest.approx(cfg.retardance_error_rms_rad, rel=0.02)


def test_hwp_retardance_default_preserves_kilo_band_width() -> None:
    cfg = HWPConfig()

    excursion = float(hwp_retardance(0.97e-6, cfg.design_wavelength_m)) - np.pi

    assert cfg.band_half_width_lambda == pytest.approx(0.25)
    assert abs(excursion) > 0.5 * cfg.retardance_error_rms_rad


def test_identity_jones_returns_complex_identity_per_mode() -> None:
    matrices = identity_jones(3)

    assert matrices.shape == (3, 2, 2)
    assert matrices.dtype == np.complex128
    for matrix in matrices:
        np.testing.assert_allclose(matrix, np.eye(2), atol=1.0e-12)


def test_random_birefringence_is_reproducible_and_unitary() -> None:
    matrices = random_birefringence(4, retardance_rms_rad=0.2, seed=123)
    matrices_repeat = random_birefringence(4, retardance_rms_rad=0.2, seed=123)

    np.testing.assert_allclose(matrices, matrices_repeat, atol=0.0)
    for matrix in matrices:
        np.testing.assert_allclose(
            matrix @ matrix.conj().T,
            np.eye(2),
            atol=1.0e-12,
        )


def test_pbs_arm_extinction_and_insertion_loss() -> None:
    vertical = np.array([0.0, 1.0], dtype=np.complex128)
    horizontal = np.array([1.0, 0.0], dtype=np.complex128)

    arm_a = polarizing_beam_splitter_arm(
        "A",
        extinction_ratio_db=30.0,
        insertion_loss_db=0.0,
    )
    leaked = apply_jones(arm_a, vertical)
    passed = apply_jones(arm_a, horizontal)

    assert np.sum(np.abs(leaked) ** 2) == pytest.approx(1.0e-3, rel=1.0e-12)
    assert np.sum(np.abs(passed) ** 2) == pytest.approx(1.0 - 1.0e-3, rel=1.0e-12)

    lossy_arm = polarizing_beam_splitter_arm(
        "A",
        extinction_ratio_db=np.inf,
        insertion_loss_db=0.5,
    )
    lossy_output = apply_jones(lossy_arm, horizontal)
    assert np.sum(np.abs(lossy_output) ** 2) == pytest.approx(
        10.0 ** (-0.5 / 10.0),
        rel=1.0e-12,
    )


def test_pbs_axis_offset_rotates_pass_axis_into_lab_frame() -> None:
    theta = 0.1
    horizontal = np.array([1.0, 0.0], dtype=np.complex128)
    arm_a = polarizing_beam_splitter_arm(
        "A",
        extinction_ratio_db=np.inf,
        insertion_loss_db=0.0,
        axis_offset_rad=theta,
    )

    output = apply_jones(arm_a, horizontal)

    expected = np.array(
        [np.cos(theta) ** 2, np.sin(theta) * np.cos(theta)],
        dtype=np.complex128,
    )
    np.testing.assert_allclose(output, expected, atol=1.0e-12)


def test_pbs_arm_rejects_bad_arm_name() -> None:
    with pytest.raises(ValueError, match="arm"):
        polarizing_beam_splitter_arm("X")


def test_apply_pbs_splits_unpolarized_power_equally() -> None:
    field = np.ones((2, 3), dtype=np.complex128)
    config = PBSConfig(insertion_loss_db=0.5, extinction_ratio_db=30.0)

    dual_pol = apply_pbs(field, config)

    assert isinstance(dual_pol, DualPolField)
    np.testing.assert_allclose(np.abs(dual_pol.arm_a), np.abs(dual_pol.arm_b))
    assert dual_pol.total_power == pytest.approx(
        float(field.size) * 10.0 ** (-0.5 / 10.0),
        rel=1.0e-12,
    )


def test_dual_pol_field_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="shape"):
        DualPolField(arm_a=np.zeros(2), arm_b=np.zeros(3))


def test_apply_hwp_passes_unpolarized_scalar_field_through() -> None:
    field = np.array([1.0 + 0.0j, 2.0 + 1.0j])

    output = apply_hwp(field, np.pi / 8.0, HWPConfig())

    assert output.dtype == np.complex128
    np.testing.assert_allclose(output, field, atol=1.0e-12)


def test_apply_hwp_rotates_dual_pol_field() -> None:
    dual_pol = DualPolField(
        arm_a=np.array([1.0 + 0.0j]),
        arm_b=np.array([0.0 + 0.0j]),
    )

    output = apply_hwp(dual_pol, np.pi / 4.0, HWPConfig())

    assert np.sum(np.abs(output.arm_a) ** 2) < 1.0e-24
    assert np.sum(np.abs(output.arm_b) ** 2) == pytest.approx(1.0, abs=1.0e-12)


def test_jones_hwp_alias_matches_half_wave_plate_up_to_global_phase() -> None:
    horizontal = np.array([1.0, 0.0], dtype=np.complex128)

    alias_output = apply_jones(jones_hwp(np.pi / 8.0), horizontal)
    canonical_output = apply_jones(half_wave_plate(np.pi / 8.0), horizontal)

    _assert_same_state_up_to_global_phase(alias_output, canonical_output)


def test_pm_fiber_leakage_floor_and_length_scaling() -> None:
    config = PMFiberConfig(per_db_per_100m=30.0, per_db_floor=30.0)

    assert pm_fiber_per_at_length(config, 5.0) == pytest.approx(1.0e-3)
    assert pm_fiber_leakage_power(30.0, 30.0, 30.0) == pytest.approx(1.0e-3)
    assert pm_fiber_per_at_length(config, 200.0) == pytest.approx(2.0e-3)


def test_jones_pm_fiber_is_unitary_and_alignment_matters() -> None:
    zero_alignment = PMFiberConfig(slow_axis_alignment_rms_rad=0.0)
    large_alignment = PMFiberConfig(slow_axis_alignment_rms_rad=0.1)

    matrices_zero = jones_pm_fiber(32, zero_alignment, fiber_length_m=30.0, seed=7)
    matrices_large = jones_pm_fiber(32, large_alignment, fiber_length_m=30.0, seed=7)

    for matrix in matrices_large:
        np.testing.assert_allclose(
            matrix @ matrix.conj().T,
            np.eye(2),
            atol=1.0e-12,
        )
    offdiag_zero = float(np.mean(np.abs(matrices_zero[:, 0, 1])))
    offdiag_large = float(np.mean(np.abs(matrices_large[:, 0, 1])))
    assert offdiag_large > 1.3 * offdiag_zero


def test_jones_pm_fiber_diagonal_power_matches_per_at_zero_alignment() -> None:
    config = PMFiberConfig(per_db_floor=20.0, slow_axis_alignment_rms_rad=0.0)

    matrices = jones_pm_fiber(4, config, fiber_length_m=30.0, seed=0)

    expected = 1.0 - pm_fiber_per_at_length(config, 30.0)
    np.testing.assert_allclose(np.abs(matrices[:, 0, 0]) ** 2, expected, atol=1.0e-12)


def test_pdi_imperfection_factor_matches_nayra_quadrature() -> None:
    pbs = PBSConfig(axis_offset_rad=1.0e-3, extinction_ratio_db=30.0)
    hwp = HWPConfig(retardance_error_rms_rad=0.02, rotation_jitter_rad=1.0e-4)
    pm_fiber = PMFiberConfig(per_db_floor=30.0)

    epsilon = pdi_imperfection_factor(
        pbs,
        hwp,
        pm_fiber,
        pm_fiber_length_m=30.0,
        sigma_i_per_arm=0.05,
    )

    expected = np.sqrt(
        (2.0e-3) ** 2
        + (0.02 / 2.0) ** 2
        + (2.0e-4) ** 2
        + 10.0 ** (-30.0 / 10.0)
        + 10.0 ** (-30.0 / 10.0)
        + 0.05**2
    )
    assert epsilon == pytest.approx(expected, rel=1.0e-12)


@pytest.mark.parametrize("bad_angle", [np.nan, np.inf, -np.inf, 1.0 + 0.0j, [0.0]])
def test_angle_inputs_must_be_finite_real_scalars(bad_angle: object) -> None:
    with pytest.raises(ValueError):
        rotator(bad_angle)


@pytest.mark.parametrize(
    "bad_vector",
    [
        np.array(1.0),
        np.array([1.0, 0.0, 0.0]),
        np.array([[1.0], [0.0]]),
    ],
)
def test_apply_jones_rejects_non_jones_vector_shapes(bad_vector: np.ndarray) -> None:
    with pytest.raises(ValueError, match="E"):
        apply_jones(np.eye(2, dtype=np.complex128), bad_vector)


@pytest.mark.parametrize(
    "bad_vector",
    [
        np.array([np.nan, 0.0]),
        np.array([np.inf, 0.0]),
        np.array([1.0 + np.nan * 1j, 0.0]),
    ],
)
def test_apply_jones_rejects_nonfinite_vectors(bad_vector: np.ndarray) -> None:
    with pytest.raises(ValueError, match="E"):
        apply_jones(np.eye(2, dtype=np.complex128), bad_vector)


def test_apply_jones_rejects_bad_matrix_shape() -> None:
    with pytest.raises(ValueError, match="M"):
        apply_jones(np.eye(3, dtype=np.complex128), np.array([1.0, 0.0]))


def test_apply_jones_rejects_nonfinite_matrix() -> None:
    matrix = np.eye(2, dtype=np.complex128)
    matrix[0, 0] = np.nan

    with pytest.raises(ValueError, match="M"):
        apply_jones(matrix, np.array([1.0, 0.0]))


def test_jones_to_stokes_rejects_non_jones_vector_shape() -> None:
    with pytest.raises(ValueError, match="E"):
        jones_to_stokes(np.array([1.0, 0.0, 0.0, 0.0]))


def test_jones_to_stokes_rejects_nan_input() -> None:
    with pytest.raises(ValueError, match="E"):
        jones_to_stokes(np.array([1.0, np.nan]))
