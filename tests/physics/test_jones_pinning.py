"""Pinning tests for Jones-matrix behavior lifted from KILO/NAYRA.

These tests intentionally skip until ``nullsim.physics.jones`` exists. They
pin behavior from KILO ``tests/test_transport.py`` and NAYRA
``tests/test_polarization.py`` while targeting the planned nullsim pure-physics
API. See ``tests/physics/notes/jones.md`` for the source survey.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(name="jones")
def fixture_jones_module() -> object:
    """Import the future Jones module, or skip tests until WP4 implements it."""
    return pytest.importorskip("nullsim.physics.jones")


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


def test_half_wave_plate_at_45deg_flips_vertical_to_horizontal(
    jones: object,
) -> None:
    """Source behavior: ideal HWP at 45 deg swaps H and V up to phase."""
    vertical = np.array([0.0, 1.0], dtype=np.complex128)

    matrix = jones.half_wave_plate(np.pi / 4.0)
    output = jones.apply_jones(matrix, vertical)

    horizontal = np.array([1.0, 0.0], dtype=np.complex128)
    _assert_same_state_up_to_global_phase(output, horizontal)


def test_rotator_zero_is_identity_for_any_jones_vector(jones: object) -> None:
    """KILO/NAYRA use the real Jones-basis rotation with theta=0 as identity."""
    vector = np.array([0.25 + 0.5j, -0.75 + 0.125j], dtype=np.complex128)

    matrix = jones.rotator(0.0)
    output = jones.apply_jones(matrix, vector)

    np.testing.assert_allclose(matrix, np.eye(2, dtype=np.complex128), atol=1e-12)
    np.testing.assert_allclose(output, vector, atol=1e-12)


def test_linear_polarizer_projects_onto_axis_angle(jones: object) -> None:
    """An ideal polarizer at theta projects the input onto e_theta."""
    theta_rad = np.pi / 6.0
    input_vector = np.array([1.0 + 0.25j, -0.5 + 0.75j], dtype=np.complex128)
    axis = np.array([np.cos(theta_rad), np.sin(theta_rad)], dtype=np.complex128)
    expected = axis * np.vdot(axis, input_vector)

    matrix = jones.linear_polarizer(theta_rad)
    output = jones.apply_jones(matrix, input_vector)

    np.testing.assert_allclose(output, expected, atol=1e-12)


def test_zero_retardance_waveplate_is_identity(jones: object) -> None:
    """Lifted from both source test suites for jones_linear_retarder."""
    matrix = jones.waveplate(retardance_rad=0.0, fast_axis_rad=np.pi / 3.0)

    np.testing.assert_allclose(matrix, np.eye(2, dtype=np.complex128), atol=1e-12)


def test_half_wave_plate_rotates_linear_polarization_by_twice_angle(
    jones: object,
) -> None:
    """Lifted KILO/NAYRA HWP test: 22.5 deg gives equal H/V powers."""
    horizontal = np.array([1.0, 0.0], dtype=np.complex128)

    output = jones.apply_jones(jones.half_wave_plate(np.pi / 8.0), horizontal)

    np.testing.assert_allclose(np.abs(output) ** 2, [0.5, 0.5], atol=1e-12)
    assert np.angle(output[0]) == pytest.approx(np.angle(output[1]), abs=1e-12)


def test_jones_to_stokes_uses_iqvu_order_and_nullsim_v_sign(
    jones: object,
) -> None:
    """KILO/NAYRA imply Q/U; nullsim pins full I,Q,U,V order here."""
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    states = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [inv_sqrt2, inv_sqrt2],
            [inv_sqrt2, 1j * inv_sqrt2],
        ],
        dtype=np.complex128,
    )
    expected = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    stokes = jones.jones_to_stokes(states)

    np.testing.assert_allclose(stokes, expected, atol=1e-12)


def test_public_matrix_constructors_return_complex128(jones: object) -> None:
    """nullsim uses complex128 even when a source identity was real-valued."""
    matrices = (
        jones.rotator(0.0),
        jones.linear_polarizer(0.0),
        jones.waveplate(retardance_rad=np.pi / 3.0, fast_axis_rad=0.2),
        jones.half_wave_plate(np.pi / 4.0),
    )

    for matrix in matrices:
        assert matrix.dtype == np.complex128
