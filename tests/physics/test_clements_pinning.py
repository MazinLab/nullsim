"""Pinning tests for nullsim.physics.clements behavior.

These tests pin the behavioral contract of the Clements decomposition
independent of implementation details.  They skip if the module is absent so
the file can be committed before the implementation ships.

Pinned behaviors:
  * Haar-random round trip (N=4): decompose + reconstruct recovers original U
  * Identity case: decomposition of I_4 reconstructs to the identity
  * complex64 input is promoted to complex128 output
  * Non-unitary input raises ValueError mentioning 'unitary'

See tests/physics/notes/jones.md for the template this file follows.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(name="clements")
def fixture_clements_module() -> object:
    """Import nullsim.physics.clements, or skip tests until it is implemented."""
    return pytest.importorskip("nullsim.physics.clements")


def _haar_unitary(n_modes: int, seed: int) -> np.ndarray:
    """Draw a Haar-distributed N×N unitary via QR with diagonal-phase fix.

    Reference: Mezzadri, Notices AMS 54, 592 (2007), Eq. (5.12).
    """
    rng = np.random.default_rng(seed)
    z = (rng.standard_normal(size=(n_modes, n_modes)) + 1j * rng.standard_normal(size=(n_modes, n_modes))) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    return q @ np.diag(np.diag(r) / np.abs(np.diag(r)))


def test_clements_haar_round_trip_n4(clements: object) -> None:
    """decompose then reconstruct returns the original N=4 Haar-random unitary."""
    U = _haar_unitary(4, seed=4246)

    decomp = clements.decompose_clements(U)
    U_reconstructed = clements.reconstruct_clements(decomp)

    np.testing.assert_allclose(U_reconstructed, U, atol=1.0e-12)


def test_clements_identity_reconstructs_to_identity(clements: object) -> None:
    """Decomposing I_4 and reconstructing must recover the identity."""
    identity = np.eye(4, dtype=np.complex128)

    reconstructed = clements.reconstruct_clements(clements.decompose_clements(identity))

    np.testing.assert_allclose(reconstructed, identity, atol=1.0e-12)


def test_clements_complex64_input_promoted_to_complex128(clements: object) -> None:
    """complex64 input must be promoted; output dtype is complex128."""
    U_f64 = _haar_unitary(4, seed=2025)
    U_f32 = U_f64.astype(np.complex64)

    decomp = clements.decompose_clements(U_f32, atol=1.0e-6)
    U_reconstructed = clements.reconstruct_clements(decomp)

    assert decomp.theta.dtype == np.float64
    assert U_reconstructed.dtype == np.complex128


def test_clements_rejects_non_unitary_matrix(clements: object) -> None:
    """A perturbed identity that is not unitary must raise ValueError."""
    bad_U = np.eye(4, dtype=np.complex128) * 1.5  # clearly not unitary

    with pytest.raises(ValueError, match="unitary"):
        clements.decompose_clements(bad_U)
