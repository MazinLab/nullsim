"""Ideal achromatic kernel-null projector for clean upper-bound studies.

This stage is intentionally not a Clements mesh optimizer. It constructs the
Guyon et al. (2013) PCA/SVD upper-bound unitary for the requested stellar
Taylor subspace and publishes that achromatic ``U_stack`` through the same
``chip_optimization`` payload consumed by ``chip.kernel_mzi_mesh``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator
from pydantic import Field as PydanticField

from nullsim.physics.clements import ClementsDecomposition, decompose_clements
from nullsim.pipeline import BaseStage, SimulationState, register_stage
from nullsim.stages._utils import (
    mode_origin_positions_m,
    project_aperture_positions_for_pointing,
)
from nullsim.stages.chip_optimization import (
    _kernel_null_field_stack,
    _n_params,
    _normalized_loss_for_target,
    _null_indices,
    _require_field,
    _require_wavelength_grid_um,
    _write_results,
)


@register_stage
class IdealKernelProjector(BaseStage):
    """Publish an ideal achromatic kernel-null unitary.

    The construction is the linear-algebra upper bound for a nuller with
    ``n_bright_ports`` dump outputs: build the order-K stellar Taylor rows,
    use SVD to order the principal stellar modes by residual starlight, and
    use those modes as the first rows of the chip unitary. Any remaining bright
    rows carry the next-lowest singular modes; the dark rows are orthogonal to
    the stellar subspace by construction.
    """

    type_name = "chip_optimization.ideal_kernel"
    family = "chip_optimization"
    consumes = frozenset({"field", "throughput", "geometry"})
    produces = frozenset({"results"})
    cache_key_extras = frozenset({"scene"})

    class Params(BaseModel):
        """Validated params for the ideal kernel projector."""

        model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

        n_modes: int = PydanticField(gt=0)
        n_bright_ports: int = PydanticField(ge=1)
        n_dark_ports: int = PydanticField(ge=1)
        n_spectro_ports: int = PydanticField(default=0, ge=0)
        wl_design_um: float = PydanticField(gt=0.0)
        kernel_order: int = PydanticField(default=3, ge=0, le=4)
        target_null_depth: float = PydanticField(default=1.0e-10, gt=0.0)
        rank_rtol: float = PydanticField(default=1.0e-12, gt=0.0)
        deployment: Literal[
            "achromatic_ustack",
            "physical_clements",
            "physical_real_clements",
        ] = "achromatic_ustack"

        @model_validator(mode="after")
        def _check_ports_and_order(self) -> IdealKernelProjector.Params:
            if (
                self.n_bright_ports
                + self.n_dark_ports
                + self.n_spectro_ports
                != self.n_modes
            ):
                raise ValueError(
                    "chip_optimization.ideal_kernel requires "
                    "n_bright_ports + n_dark_ports + n_spectro_ports == n_modes "
                    f"(got {self.n_bright_ports} + {self.n_dark_ports} + "
                    f"{self.n_spectro_ports} != {self.n_modes})."
                )
            if self.kernel_order not in (0, 2, 3, 4):
                raise ValueError(
                    "chip_optimization.ideal_kernel kernel_order must be "
                    f"0, 2, 3, or 4 (got {self.kernel_order})."
                )
            return self

    def apply(self, state: SimulationState) -> SimulationState:
        params: IdealKernelProjector.Params = self.params_model
        field_array = _require_field(
            state,
            params.n_modes,
            wl_design_um=params.wl_design_um,
        )
        wavelength_grid_um = _require_wavelength_grid_um(state, field_array)
        positions = _require_positions(state, params)

        design_index = int(
            np.argmin(
                np.abs(
                    np.asarray(wavelength_grid_um, dtype=np.float64)
                    - float(params.wl_design_um)
                )
            )
        )
        design_field = field_array[:, design_index, :]
        basis, singular_values, rank = _stellar_subspace_basis(
            design_field=design_field,
            positions=positions,
            kernel_order=int(params.kernel_order),
            rank_rtol=float(params.rank_rtol),
        )
        if rank > int(params.n_bright_ports):
            raise ValueError(
                "chip_optimization.ideal_kernel needs at least one bright port "
                "per independent stellar Taylor row: "
                f"rank={rank}, n_bright_ports={params.n_bright_ports}."
            )

        u_matrix = basis.conj().T.astype(np.complex128, copy=False)
        u_stack = _achromatic_stack(u_matrix, len(wavelength_grid_um))
        loss_final = _kernel_leakage_loss(
            u_stack=u_stack,
            field_array=field_array,
            positions=positions,
            params=params,
        )
        optimal_params, deployed_u_stack, rescales_per_wavelength = (
            _deployment_payload(u_matrix, params, len(wavelength_grid_um))
        )

        return _write_results(
            state=state,
            instance_id=self.context.instance_id,
            optimal_params=optimal_params,
            loss_final=loss_final,
            iterations=0,
            backend_name="ideal_kernel_projector",
            kernel_rescales_per_wavelength=rescales_per_wavelength,
            normalized_loss_final=_normalized_loss_for_target(
                loss_final,
                field_array,
            ),
            optimal_u_stack=deployed_u_stack,
            backend_metadata={
                "kernel_order": int(params.kernel_order),
                "stellar_subspace_rank": int(rank),
                "rank_rtol": float(params.rank_rtol),
                "deployment": str(params.deployment),
                "singular_values": tuple(float(v) for v in singular_values),
                "torch_wavelength_batch_size": int(len(wavelength_grid_um)),
                "source_model": (
                    "Ideal achromatic kernel-null projector: Guyon et al. "
                    "(2013) PCA/SVD stellar Taylor modes are routed into "
                    "bright ports in decreasing singular-value order. "
                    "deployment='achromatic_ustack' is the upper-bound HWO "
                    "baseline; deployment='physical_clements' decomposes the "
                    "center-wavelength unitary into a dispersive Clements mesh "
                    "for finite-bandwidth degradation studies; "
                    "deployment='physical_real_clements' uses a signed-real "
                    "Givens decomposition when the target unitary is real, "
                    "avoiding internal pi phase flips. The remaining row "
                    "sign phases are treated as achromatic output relabeling "
                    "rather than fixed-OPD phase shifters."
                ),
            },
        )


def _require_positions(
    state: SimulationState,
    params: IdealKernelProjector.Params,
) -> NDArray[np.float64]:
    if params.kernel_order == 0:
        # The order-0 helper does not use positions, but keep the shape valid.
        return np.zeros((params.n_modes, 2), dtype=np.float64)
    positions = state.geometry.aperture_positions_pupil_m
    if positions is None:
        raise ValueError(
            "chip_optimization.ideal_kernel with kernel_order > 0 requires "
            "geometry.aperture_positions_pupil_m from an upstream stage."
        )
    positions_array = np.asarray(positions, dtype=np.float64)
    if positions_array.shape != (params.n_modes, 2):
        raise ValueError(
            "chip_optimization.ideal_kernel aperture_positions_pupil_m has "
            f"shape {positions_array.shape}; expected ({params.n_modes}, 2)."
        )
    # Project onto the sky plane so the kernel-derivative phase rows
    # foreshorten correctly off-zenith (inter-telescope component only when
    # per-mode origins are published; review 2026-06-10 Issue 20).
    return project_aperture_positions_for_pointing(
        positions_array,
        altitude_deg=state.scene.altitude_deg,
        azimuth_deg=state.scene.azimuth_deg,
        aperture_origins_m=mode_origin_positions_m(
            state.geometry, int(positions_array.shape[0])
        ),
    )


def _stellar_subspace_basis(
    *,
    design_field: NDArray[np.complex128],
    positions: NDArray[np.float64],
    kernel_order: int,
    rank_rtol: float,
) -> tuple[NDArray[np.complex128], NDArray[np.float64], int]:
    stack = _kernel_null_field_stack(
        design_field,
        positions,
        order=kernel_order,
    )
    # Columns are the independent stellar vectors whose span must be dumped.
    vectors = stack.reshape(stack.shape[0], stack.shape[1], -1)
    matrix = np.moveaxis(vectors, 1, 0).reshape(stack.shape[1], -1)
    u_matrix, singular_values, _vh = np.linalg.svd(matrix, full_matrices=True)
    if singular_values.size == 0:
        rank = 0
        threshold = 0.0
    else:
        threshold = float(rank_rtol) * float(singular_values[0])
        rank = int(np.count_nonzero(singular_values > threshold))
    if rank == 0:
        raise ValueError(
            "chip_optimization.ideal_kernel found zero-rank stellar input "
            "subspace; check upstream field amplitudes."
        )
    basis = _canonicalize_basis_phases(u_matrix[:, : matrix.shape[0]])
    identity = np.eye(matrix.shape[0], dtype=np.complex128)
    if not np.allclose(basis.conj().T @ basis, identity, atol=1.0e-10, rtol=0.0):
        raise ValueError("chip_optimization.ideal_kernel SVD basis is not unitary.")
    return basis, singular_values, rank


def _canonicalize_basis_phases(
    basis: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Fix arbitrary per-column SVD phases without changing the subspaces."""

    return np.column_stack(
        [
            _canonicalize_vector_phase(column)
            for column in np.asarray(basis, dtype=np.complex128).T
        ]
    ).astype(np.complex128, copy=False)


def _canonicalize_vector_phase(
    vector: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    pivot_index = int(np.argmax(np.abs(vector)))
    pivot = complex(vector[pivot_index])
    if abs(pivot) == 0.0:
        return vector
    phase = pivot / abs(pivot)
    canonical = vector / phase
    if canonical[pivot_index].real < 0.0:
        canonical = -canonical
    return canonical


def _kernel_leakage_loss(
    *,
    u_stack: NDArray[np.complex128],
    field_array: NDArray[np.complex128],
    positions: NDArray[np.float64],
    params: IdealKernelProjector.Params,
) -> float:
    field_stack = _kernel_null_field_stack(
        field_array,
        positions,
        order=int(params.kernel_order),
    )
    null_idx = _null_indices(params)
    output = np.einsum("ijw,rjwp->riwp", u_stack, field_stack)
    leak_per_wavelength = np.sum(
        np.abs(output[:, null_idx, :, :]) ** 2,
        axis=(0, 1, 3),
    )
    return float(np.mean(leak_per_wavelength))


def _achromatic_stack(
    u_matrix: NDArray[np.complex128],
    n_wavelengths: int,
) -> NDArray[np.complex128]:
    return np.repeat(
        u_matrix[:, :, np.newaxis],
        int(n_wavelengths),
        axis=2,
    ).astype(np.complex128, copy=False)


def _deployment_payload(
    u_matrix: NDArray[np.complex128],
    params: IdealKernelProjector.Params,
    n_wavelengths: int,
) -> tuple[NDArray[np.float64], NDArray[np.complex128] | None, bool]:
    if params.deployment == "achromatic_ustack":
        return (
            np.zeros(_n_params(params.n_modes), dtype=np.float64),
            _achromatic_stack(u_matrix, n_wavelengths),
            False,
        )
    if params.deployment == "physical_clements":
        decomp = decompose_clements(u_matrix, atol=1.0e-5)
        optimal_params = np.concatenate(
            [
                np.asarray(decomp.theta, dtype=np.float64),
                np.asarray(decomp.phi, dtype=np.float64),
                np.asarray(decomp.alpha, dtype=np.float64),
            ]
        )
        return optimal_params, None, True
    if params.deployment == "physical_real_clements":
        decomp = _decompose_real_clements(u_matrix, atol=1.0e-10)
        optimal_params = np.concatenate(
            [
                np.asarray(decomp.theta, dtype=np.float64),
                np.asarray(decomp.phi, dtype=np.float64),
                np.asarray(decomp.alpha, dtype=np.float64),
            ]
        )
        return optimal_params, None, False
    raise ValueError(f"Unknown ideal-kernel deployment {params.deployment!r}.")


def _decompose_real_clements(
    u_matrix: NDArray[np.complex128],
    *,
    atol: float,
) -> ClementsDecomposition:
    """Decompose a real orthogonal unitary with signed rotations and phi=0.

    The generic complex Clements decomposition represents sign choices with
    internal ``phi = pi`` phase shifts. Under a fixed-OPD model those internal
    pi shifts become chromatic away from the design wavelength. For real HWO
    kernel projectors, the same unitary can be implemented as a real Givens
    mesh with signed theta rotations, zero internal phi, and only output-row
    sign phases. Output phases do not change per-port power or null depth.
    """

    matrix_complex = np.asarray(u_matrix, dtype=np.complex128)
    if matrix_complex.ndim != 2 or matrix_complex.shape[0] != matrix_complex.shape[1]:
        raise ValueError("physical_real_clements requires a square matrix.")
    if not np.all(np.isfinite(matrix_complex)):
        raise ValueError("physical_real_clements requires finite matrix values.")
    if float(np.max(np.abs(matrix_complex.imag))) > float(atol):
        raise ValueError(
            "physical_real_clements requires a real-valued target unitary; "
            "use deployment='physical_clements' for complex targets."
        )

    matrix = matrix_complex.real.astype(np.float64, copy=True)
    n_modes = int(matrix.shape[0])
    identity = np.eye(n_modes, dtype=np.float64)
    if not np.allclose(matrix @ matrix.T, identity, atol=float(atol), rtol=0.0):
        raise ValueError("physical_real_clements target is not orthogonal.")

    working = matrix.copy()
    theta_values: list[float] = []
    phi_values: list[float] = []
    order: list[tuple[int, int]] = []
    for column_index in range(n_modes - 1):
        for lower_row in range(n_modes - 1, column_index, -1):
            upper_row = lower_row - 1
            theta_rad = float(
                np.arctan2(
                    working[lower_row, column_index],
                    working[upper_row, column_index],
                )
            )
            cos_theta = np.cos(theta_rad)
            sin_theta = np.sin(theta_rad)
            rotation = np.array(
                [
                    [cos_theta, sin_theta],
                    [-sin_theta, cos_theta],
                ],
                dtype=np.float64,
            )
            working[[upper_row, lower_row], :] = (
                rotation @ working[[upper_row, lower_row], :]
            )
            theta_values.append(theta_rad)
            phi_values.append(0.0)
            order.append((upper_row, lower_row))

    off_diag = working - np.diag(np.diag(working))
    off_diag_max = float(np.max(np.abs(off_diag))) if off_diag.size else 0.0
    decomp_tol = max(n_modes * float(atol), 1.0e-12)
    if off_diag_max > decomp_tol:
        raise ValueError(
            "physical_real_clements triangularization failed; off-diagonal "
            f"residual {off_diag_max:.3e} exceeds tolerance {decomp_tol:.3e}."
        )

    return ClementsDecomposition(
        theta=np.asarray(theta_values, dtype=np.float64),
        phi=np.asarray(phi_values, dtype=np.float64),
        alpha=np.angle(np.diag(working).astype(np.complex128)).astype(
            np.float64,
            copy=False,
        ),
        order=order,
        n_modes=n_modes,
    )


__all__ = ["IdealKernelProjector"]
