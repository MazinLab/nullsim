"""Clements-style unitary mesh decomposition primitives.

The functions here expose the package-level Clements import surface used by
chip-stage work. The implementation factors a unitary matrix into nearest-row
complex Givens rotations plus an output phase screen, which is the numerical
core required for an MZI mesh representation. It is pure math: no pipeline,
configuration, state, or stage imports.

References:
    SPEC §15.2.
    Clements et al., "Optimal design for universal multiport interferometers,"
        Optica 3, 1460 (2016), Eqs. (1)-(3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ClementsDecomposition:
    """Nearest-neighbor unitary mesh factors.

    Attributes:
        theta: Rotation angles in radians, shape ``(N * (N - 1) / 2,)``.
        phi: Relative phase angles in radians, same shape as ``theta``.
        alpha: Output phase-screen angles in radians, shape ``(N,)``.
        order: Row-pair order for each rotation as ``(upper_row, lower_row)``.
        n_modes: Number of optical modes in the decomposed unitary.
    """

    theta: NDArray[np.float64]
    phi: NDArray[np.float64]
    alpha: NDArray[np.float64]
    order: list[tuple[int, int]]
    n_modes: int


def decompose_clements(U: np.ndarray, atol: float = 1.0e-10) -> ClementsDecomposition:
    """Decompose a unitary matrix into nearest-neighbor mesh rotations.

    Args:
        U: Square unitary matrix. Values are cast to ``complex128``.
        atol: Absolute tolerance for the unitarity check.

    Returns:
        A frozen ``ClementsDecomposition`` whose reconstruction matches ``U`` to
        numerical precision.

    Raises:
        ValueError: If ``U`` is non-square, non-finite, or not unitary within
            ``atol``.
    """
    matrix = _as_unitary_matrix(U, atol)
    n_modes = matrix.shape[0]

    working = matrix.copy()
    theta_values: list[float] = []
    phi_values: list[float] = []
    order: list[tuple[int, int]] = []

    for column_index in range(n_modes - 1):
        for lower_row in range(n_modes - 1, column_index, -1):
            upper_row = lower_row - 1
            theta_rad, phi_rad = _rotation_angles(
                working[upper_row, column_index],
                working[lower_row, column_index],
            )
            rotation = _rotation_block(theta_rad, phi_rad)
            working[[upper_row, lower_row], :] = (
                rotation @ working[[upper_row, lower_row], :]
            )
            theta_values.append(theta_rad)
            phi_values.append(phi_rad)
            order.append((upper_row, lower_row))

    # Code Review 2026-05-16 Issue 28: after triangularization, ``working``
    # must be diagonal — otherwise the decomposition silently failed and
    # we'd return garbage angles. Verify and raise.
    off_diag = working - np.diag(np.diag(working))
    off_diag_max = float(np.max(np.abs(off_diag))) if off_diag.size else 0.0
    decomp_tol = max(n_modes * atol, 1.0e-10)
    if off_diag_max > decomp_tol:
        raise ValueError(
            "decompose_clements: triangularization failed; off-diagonal "
            f"residual {off_diag_max:.3e} exceeds tolerance {decomp_tol:.3e}. "
            "The input matrix may not be Clements-decomposable in float64."
        )

    return ClementsDecomposition(
        theta=np.asarray(theta_values, dtype=np.float64),
        phi=np.asarray(phi_values, dtype=np.float64),
        alpha=np.angle(np.diag(working)).astype(np.float64, copy=False),
        order=order,
        n_modes=n_modes,
    )


def reconstruct_clements(decomp: ClementsDecomposition) -> NDArray[np.complex128]:
    """Reconstruct a unitary matrix from a Clements decomposition.

    Args:
        decomp: Decomposition returned by :func:`decompose_clements`.

    Returns:
        Reconstructed unitary matrix as ``complex128``.

    Raises:
        ValueError: If the decomposition arrays have inconsistent shapes or row
            indices.
    """
    _validate_decomposition(decomp)
    matrix = np.diag(np.exp(1j * decomp.alpha)).astype(np.complex128)

    for theta_rad, phi_rad, (upper_row, lower_row) in reversed(
        list(zip(decomp.theta, decomp.phi, decomp.order, strict=True))
    ):
        rotation = _rotation_block(theta_rad, phi_rad)
        matrix[[upper_row, lower_row], :] = (
            rotation.conjugate().T @ matrix[[upper_row, lower_row], :]
        )

    return matrix.astype(np.complex128, copy=False)


def _as_unitary_matrix(U: np.ndarray, atol: float) -> NDArray[np.complex128]:
    """Validate and coerce an input unitary matrix."""
    atol_value = _as_nonnegative_finite_scalar(atol, "atol")
    try:
        matrix = np.asarray(U, dtype=np.complex128)
    except (TypeError, ValueError) as exc:
        raise ValueError("U must be numeric and convertible to complex128.") from exc

    if matrix.ndim != 2:
        raise ValueError("U must be a two-dimensional matrix.")
    n_rows, n_columns = matrix.shape
    if n_rows != n_columns:
        raise ValueError("U must be square.")
    if n_rows == 0:
        raise ValueError("U must be non-empty.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("U must contain only finite values.")

    identity = np.eye(n_rows, dtype=np.complex128)
    if not np.allclose(matrix @ matrix.conjugate().T, identity, atol=atol_value, rtol=0.0):
        raise ValueError("input matrix is not unitary within atol.")

    return matrix


def _as_nonnegative_finite_scalar(value: float, name: str) -> np.float64:
    """Return a finite non-negative float64 scalar."""
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in ("i", "u", "f"):
        raise ValueError(f"{name} must be a finite real scalar.")

    scalar = np.float64(array)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return scalar


def _rotation_angles(a: np.complex128, b: np.complex128) -> tuple[float, float]:
    """Return angles for a complex Givens rotation that zeros ``b``.

    Code Review 2026-05-16 Issue 27: the previous early-exit branches
    used ``np.isclose(b, 0.0, atol=0.0, rtol=0.0)`` which collapses to
    bit-exact equality (``b == 0.0``). Sub-eps inputs went through the
    general path anyway, so the branches were dead. ``np.arctan2`` and
    ``np.angle`` handle the zero cases correctly without special-casing.
    """
    theta_rad = float(np.arctan2(np.abs(b), np.abs(a)))
    phi_rad = float(np.angle(a) - np.angle(b))
    return theta_rad, phi_rad


def _rotation_block(theta_rad: float, phi_rad: float) -> NDArray[np.complex128]:
    """Return the 2x2 complex Givens/MZI rotation block."""
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    phase = np.exp(1j * phi_rad)
    return np.array(
        [
            [cos_theta, phase * sin_theta],
            [-np.conjugate(phase) * sin_theta, cos_theta],
        ],
        dtype=np.complex128,
    )


def _validate_decomposition(decomp: ClementsDecomposition) -> None:
    """Validate decomposition shape consistency before reconstruction."""
    if not isinstance(decomp, ClementsDecomposition):
        raise ValueError("decomp must be a ClementsDecomposition.")
    if decomp.n_modes <= 0:
        raise ValueError("decomp.n_modes must be positive.")

    expected_rotations = decomp.n_modes * (decomp.n_modes - 1) // 2
    if decomp.theta.shape != (expected_rotations,):
        raise ValueError("decomp.theta has the wrong shape.")
    if decomp.phi.shape != (expected_rotations,):
        raise ValueError("decomp.phi has the wrong shape.")
    if decomp.alpha.shape != (decomp.n_modes,):
        raise ValueError("decomp.alpha has the wrong shape.")
    if len(decomp.order) != expected_rotations:
        raise ValueError("decomp.order has the wrong length.")
    if (
        not np.all(np.isfinite(decomp.theta))
        or not np.all(np.isfinite(decomp.phi))
        or not np.all(np.isfinite(decomp.alpha))
    ):
        raise ValueError("decomposition angles must be finite.")

    for upper_row, lower_row in decomp.order:
        if lower_row != upper_row + 1:
            raise ValueError("decomp.order must contain nearest-neighbor row pairs.")
        if upper_row < 0 or lower_row >= decomp.n_modes:
            raise ValueError("decomp.order contains an out-of-range row index.")


__all__ = [
    "ClementsDecomposition",
    "decompose_clements",
    "reconstruct_clements",
]
