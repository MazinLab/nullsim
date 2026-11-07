"""Known-position characterization chip optimization primitives.

Characterization mode retunes the photonic chip for a known planet
position: stellar light is constrained at one or more spectro ports while
planet light is routed to those same ports for high-resolution follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from nullsim.physics.clements import (
    ClementsDecomposition,
    decompose_clements,
    reconstruct_clements,
)
from nullsim.physics.fringe_observability import intensity_response_matrix
from nullsim.physics.realized_null import (
    _sigma_for_mode_or_group,
    _validate_mode_group_indices,
    _validate_piston_covariance,
)
from nullsim.physics.ruilier_cassaing import angular_coupling_factor

__all__ = [
    "CharacterizationResult",
    "build_unitary_from_params",
    "build_unitary_stack_from_params",
    "disk_sample_stellar_stack",
    "mode_piston_opd_covariance_m2",
    "optimize_characterization",
    "piston_averaged_port_power",
    "spectro_disk_leak",
    "spectro_disk_leak_from_stack",
]

_TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class CharacterizationResult:
    """Optimized known-position chip state and headline chip metrics."""

    params: NDArray[np.float64]
    spectro_null_depth: float
    dark_null_depth: float
    spectro_throughput: float
    bright_stellar_fraction: float
    objective_value: float
    n_restarts_completed: int
    optimization_success: bool = True
    constraint_margin: float = 0.0
    tracker_singular_values_per_nm: NDArray[np.float64] | None = None
    tracker_crlb_rms_nm: float | None = None
    tracker_loss: float = 0.0
    tracker_crlb_loss: float = 0.0
    optimal_u_stack: NDArray[np.complex128] | None = None
    optimal_u_stack_wavelengths_m: NDArray[np.float64] | None = None
    # The objective the optimizer actually minimized (the broadband
    # photon-weighted objective in broadband mode). ``objective_value``
    # is the deployed-design metric recomputed at the design wavelength;
    # the two differ in broadband mode (review 2026-06-10 Issue 22).
    optimizer_objective_value: float | None = None


def build_unitary_from_params(
    params: NDArray[np.float64] | Sequence[float],
    n_modes: int,
) -> NDArray[np.complex128]:
    """Build a Clements unitary from nullsim canonical params."""

    params_arr = np.asarray(params, dtype=np.float64)
    theta, phi, alpha = _split_params(params_arr, n_modes)
    decomp = ClementsDecomposition(
        theta=theta,
        phi=phi,
        alpha=alpha,
        order=_clements_order(n_modes),
        n_modes=n_modes,
    )
    return reconstruct_clements(decomp)


def build_unitary_stack_from_params(
    *,
    params: NDArray[np.float64] | Sequence[float],
    n_modes: int,
    wavelengths_m: Sequence[float],
    wl_design_m: float,
    dac_bits: int | None = None,
) -> NDArray[np.complex128]:
    """Build ``(n_modes, n_modes, n_wl)`` with KILO phase rescaling."""

    raw = np.asarray(params, dtype=np.float64)
    if dac_bits is not None:
        raw = _quantize_phases(raw, dac_bits)
    theta, phi, alpha = _split_params(raw, n_modes)
    order = _clements_order(n_modes)

    wavelengths = np.asarray(wavelengths_m, dtype=np.float64)
    stack = np.empty((n_modes, n_modes, wavelengths.size), dtype=np.complex128)
    for index, wavelength_m in enumerate(wavelengths):
        scale = float(wl_design_m) / float(wavelength_m)
        decomp = ClementsDecomposition(
            theta=theta,
            phi=phi * scale,
            alpha=alpha * scale,
            order=order,
            n_modes=n_modes,
        )
        stack[:, :, index] = reconstruct_clements(decomp)
    return stack


def disk_sample_stellar_stack(
    *,
    positions_m: NDArray[np.float64],
    base_amplitudes: NDArray[np.float64],
    wavelength_m: float,
    sub_diameters_m: NDArray[np.float64],
    stellar_radius_rad: float,
    n_rings: int = 3,
    n_azimuths: int = 4,
    kernel_order: int = 0,
) -> NDArray[np.complex128]:
    """Return stellar field rows sampling the resolved host disk.

    Row 0 is the on-axis field. Optional kernel rows enforce low-order
    Taylor derivatives near the origin. Disk rows sample real and
    imaginary parts around rings on the stellar disk, L2-equalized to the
    on-axis row so the optimizer treats each constraint comparably.
    """

    positions = np.asarray(positions_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise ValueError("positions_m must have shape (n_modes, >=2).")
    base = np.asarray(base_amplitudes, dtype=np.float64)
    diameters = np.asarray(sub_diameters_m, dtype=np.float64)
    if positions.shape[0] != base.size or diameters.shape != base.shape:
        raise ValueError("positions_m, base_amplitudes, and sub_diameters_m mismatch.")

    on_axis = base.astype(np.complex128)
    on_axis_norm = float(np.linalg.norm(on_axis))
    rows: list[NDArray[np.complex128]] = [on_axis]

    if kernel_order >= 2:
        x_m = positions[:, 0]
        y_m = positions[:, 1]
        weights = [x_m, y_m]
        if kernel_order >= 3:
            weights += [x_m * x_m, y_m * y_m, x_m * y_m]
        if kernel_order >= 4:
            weights += [
                x_m * x_m * x_m,
                y_m * y_m * y_m,
                x_m * x_m * y_m,
                x_m * y_m * y_m,
            ]
        for weight in weights:
            row = weight * on_axis
            norm = float(np.linalg.norm(row))
            if norm > 0.0:
                rows.append((row * (on_axis_norm / norm)).astype(np.complex128))

    if stellar_radius_rad > 0.0 and n_rings > 0 and n_azimuths > 0:
        for ring_index in range(n_rings):
            radius = stellar_radius_rad * (ring_index + 1) / n_rings
            angular_power = np.asarray(
                angular_coupling_factor(radius, diameters, wavelength_m),
                dtype=np.float64,
            )
            amplitudes = base * np.sqrt(angular_power)
            for azimuth_index in range(n_azimuths):
                angle = _TWO_PI * azimuth_index / n_azimuths
                alpha = radius * np.sin(angle)
                beta = radius * np.cos(angle)
                opd_m = positions[:, 0] * alpha + positions[:, 1] * beta
                phase = _TWO_PI * opd_m / wavelength_m
                sample = amplitudes * np.exp(1j * phase)
                for part in (sample.real, sample.imag):
                    norm = float(np.linalg.norm(part))
                    if norm > 0.0:
                        rows.append(
                            (part * (on_axis_norm / norm)).astype(np.complex128)
                        )

    return np.stack(rows, axis=0)


def spectro_disk_leak(
    *,
    unitary: NDArray[np.complex128],
    positions_m: NDArray[np.float64],
    base_amplitudes: NDArray[np.float64],
    spectro_port_indices: NDArray[np.intp] | Sequence[int],
    wavelength_m: float,
    sub_diameters_m: NDArray[np.float64],
    stellar_radius_mas: float,
    n_radial: int = 21,
    n_pa: int = 16,
) -> float:
    """Return PA-averaged disk leakage fraction into spectro ports."""

    if stellar_radius_mas <= 0.0:
        return 0.0
    if n_radial < 2 or n_pa < 1:
        raise ValueError("n_radial must be >= 2 and n_pa must be >= 1.")

    positions = np.asarray(positions_m, dtype=np.float64)
    base = np.asarray(base_amplitudes, dtype=np.float64)
    diameters = np.asarray(sub_diameters_m, dtype=np.float64)
    spectro_idx = np.asarray(spectro_port_indices, dtype=np.intp)
    total_power = float(np.sum(base * base))
    if total_power <= 0.0:
        raise ValueError("base_amplitudes have zero total power.")

    radius_grid_mas = np.linspace(0.0, stellar_radius_mas, n_radial)
    pa_grid = np.linspace(0.0, _TWO_PI, n_pa, endpoint=False)
    profile = np.zeros_like(radius_grid_mas)

    for radius_index, radius_mas in enumerate(radius_grid_mas):
        radius_rad = radius_mas / 206265e3
        angular_power = np.asarray(
            angular_coupling_factor(radius_rad, diameters, wavelength_m),
            dtype=np.float64,
        )
        amplitudes = base * np.sqrt(angular_power)
        accum = 0.0
        for pa in pa_grid:
            alpha = radius_rad * np.sin(pa)
            beta = radius_rad * np.cos(pa)
            opd_m = positions[:, 0] * alpha + positions[:, 1] * beta
            phase = _TWO_PI * opd_m / wavelength_m
            field = amplitudes * np.exp(1j * phase)
            output = unitary @ field
            accum += float(np.sum(np.abs(output[spectro_idx]) ** 2))
        profile[radius_index] = accum / (n_pa * total_power)

    integral = np.trapezoid(radius_grid_mas * profile, radius_grid_mas)
    return float((2.0 / stellar_radius_mas**2) * integral)


def spectro_disk_leak_from_stack(
    *,
    u_stack: NDArray[np.complex128],
    positions_m: NDArray[np.float64],
    base_amplitudes: NDArray[np.float64],
    spectro_port_indices: NDArray[np.intp] | Sequence[int],
    wavelengths_m: Sequence[float],
    sub_diameters_m: NDArray[np.float64],
    stellar_radius_mas: float,
    n_radial: int = 21,
    n_pa: int = 16,
) -> tuple[float, NDArray[np.float64]]:
    """Return spectro disk leakage for an already deployed chromatic stack."""

    wavelengths = np.asarray(wavelengths_m, dtype=np.float64)
    stack = np.asarray(u_stack, dtype=np.complex128)
    if stack.shape != (base_amplitudes.size, base_amplitudes.size, wavelengths.size):
        raise ValueError("u_stack shape does not match wavelength grid.")
    leak_per_wl = np.empty(wavelengths.size, dtype=np.float64)
    for index, wavelength_m in enumerate(wavelengths):
        leak_per_wl[index] = spectro_disk_leak(
            unitary=stack[:, :, index],
            positions_m=positions_m,
            base_amplitudes=base_amplitudes,
            spectro_port_indices=spectro_port_indices,
            wavelength_m=float(wavelength_m),
            sub_diameters_m=sub_diameters_m,
            stellar_radius_mas=stellar_radius_mas,
            n_radial=n_radial,
            n_pa=n_pa,
        )
    return float(np.mean(leak_per_wl)), leak_per_wl


def mode_piston_opd_covariance_m2(
    *,
    n_modes: int,
    opd_rms_per_sub_m: float | NDArray[np.float64] = 0.0,
    opd_covariance_m2: NDArray[np.float64] | None = None,
    mode_group_indices: NDArray[np.intp] | None = None,
) -> NDArray[np.float64]:
    """Per-chip-mode piston OPD covariance drawn by :func:`realized_null_depth_mc`.

    Expands the realized-null MC's piston inputs to an ``(M, M)`` covariance
    in m^2: an explicit covariance wins (expanded from groups to modes when
    ``mode_group_indices`` is set); otherwise independent per-mode or
    per-group RMS values, with modes of one group fully correlated.
    """

    groups = _validate_mode_group_indices(mode_group_indices, n_modes=n_modes)
    covariance = _validate_piston_covariance(
        opd_covariance_m2, n_modes=n_modes, group_indices=groups,
    )
    if covariance is not None:
        matrix, cov_groups = covariance
        return matrix if cov_groups is None else matrix[np.ix_(cov_groups, cov_groups)]
    sigma, sigma_groups = _sigma_for_mode_or_group(
        opd_rms_per_sub_m,
        n_modes=n_modes,
        group_indices=groups,
        name="ao_piston_opd_rms_per_sub_m",
    )
    if sigma_groups is None:
        return np.diag(sigma**2)
    same_group = sigma_groups[:, None] == sigma_groups[None, :]
    return np.where(same_group, sigma[sigma_groups][:, None] ** 2, 0.0)


def piston_averaged_port_power(
    *,
    unitary: NDArray[np.complex128],
    field: NDArray[np.complex128],
    port_indices: NDArray[np.intp] | Sequence[int],
    phase_covariance_rad2: NDArray[np.float64],
) -> float:
    """Expected output power ``E sum_s |U_s (X * exp(i phi))|^2`` over Gaussian phase.

    For zero-mean jointly Gaussian input phases ``phi`` with covariance
    ``Sigma``, ``E exp(i(phi_m - phi_n)) = exp(-(Sigma_mm + Sigma_nn -
    2 Sigma_mn) / 2)`` (the Gaussian characteristic function), so the
    expectation is ``sum_s a_s^T G conj(a_s)`` with ``a_s = U_s * X``.
    Piston scatters a planet's coherent output between ports the same way
    it scatters starlight into the null.
    """

    rows = np.asarray(unitary, dtype=np.complex128)[np.asarray(port_indices, dtype=np.intp)]
    amplitudes = rows * np.asarray(field, dtype=np.complex128)[None, :]
    cov = np.asarray(phase_covariance_rad2, dtype=np.float64)
    var = np.diag(cov)
    coherence = np.exp(-0.5 * (var[:, None] + var[None, :] - 2.0 * cov))
    return float(np.real(np.einsum("sm,mn,sn->", amplitudes, coherence, amplitudes.conj())))


def optimize_characterization(
    *,
    stellar_field: NDArray[np.complex128],
    planet_field: NDArray[np.complex128],
    n_modes: int,
    dark_port_indices: NDArray[np.intp] | Sequence[int],
    spectro_port_indices: NDArray[np.intp] | Sequence[int],
    bright_port_indices: NDArray[np.intp] | Sequence[int] | None = None,
    null_floor: float = 1.0e-6,
    n_restarts: int = 5,
    seed: int | None = None,
    max_iter: int = 1000,
    throughput_weight: float = 1.0,
    bright_weight: float = 1.0,
    snr_proxy_weight: float = 0.0,
    initial_params: Sequence[NDArray[np.float64] | Sequence[float]] = (),
    wl_design_m: float | None = None,
    wavelengths_m: Sequence[float] | None = None,
    stellar_fields_per_wl: Sequence[NDArray[np.complex128]] | None = None,
    planet_fields_per_wl: Sequence[NDArray[np.complex128]] | None = None,
    stellar_weights_per_wl: Sequence[float] | None = None,
    planet_weights_per_wl: Sequence[float] | None = None,
    deployment_wavelengths_m: Sequence[float] | None = None,
    selection_wavelengths_m: Sequence[float] | None = None,
    selection_stellar_fields_per_wl: Sequence[NDArray[np.complex128]] | None = None,
    selection_planet_fields_per_wl: Sequence[NDArray[np.complex128]] | None = None,
    selection_stellar_weights_per_wl: Sequence[float] | None = None,
    selection_planet_weights_per_wl: Sequence[float] | None = None,
    dac_bits: int | None = None,
    backend: str = "auto",
    tracker_piston_basis: NDArray[np.float64] | None = None,
    tracker_field: NDArray[np.complex128] | None = None,
    tracker_wavelength_m: float | None = None,
    tracker_observability_weight: float = 0.0,
    tracker_observability_min_singular_per_nm: float = 1.0e-3,
    tracker_observability_crlb_weight: float = 0.0,
    tracker_observability_crlb_target_nm: float = 100.0,
    tracker_observability_crlb_constraint_nm: float | None = None,
    tracker_observability_eps: float = 1.0e-12,
    preserve_bright_rows: bool = False,
    enforce_null_constraint: bool = True,
    n_workers: int = 1,
    compile_objective: bool = False,
) -> CharacterizationResult:
    """Optimize a known-position spectro-port chip solution.

    When ``dac_bits`` is set, the best continuous solution is quantized
    onto the DAC grid before the returned ``params`` and metrics are
    computed, so the reported throughput/null depth correspond to the
    deployed quantized chip rather than the unconstrained optimizer
    output.

    ``backend`` selects the optimization driver:
      * ``"scipy"`` — SLSQP with 2-point finite-difference Jacobians
        (numpy only, no extras required).
      * ``"torch"`` — SLSQP with torch-autograd Jacobians.
        Upstream canonical detection-chip params are fitted into the
        torch-native mesh layout before retuning.
      * ``"auto"`` — use ``"torch"`` when torch is importable; otherwise
        ``"scipy"``.

    Broadband optimization uses photon-weighted mean leakage/throughput when
    ``*_weights_per_wl`` are provided. Tracker observability is evaluated as a
    design-wavelength regularizer or constraint.

    When ``preserve_bright_rows`` is true with the torch backend, the physical
    fixed-phase optimizer relies on the tracker-observability terms to keep the
    bright-port piston-sensing subspace usable while retuning the deployed mesh;
    it may use either supplied initial params or fresh random starts. The exact
    row-copying implementation is retained only for the scipy fallback path.
    """

    if null_floor <= 0.0:
        raise ValueError("null_floor must be positive.")
    if n_restarts < 1 and not initial_params:
        raise ValueError("n_restarts must be >= 1 when initial_params is empty.")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1.")
    if n_workers < 1:
        raise ValueError("n_workers must be >= 1.")
    if backend not in ("auto", "scipy", "torch"):
        raise ValueError(f"backend must be 'auto', 'scipy', or 'torch'; got {backend!r}.")
    if tracker_observability_weight < 0.0:
        raise ValueError("tracker_observability_weight must be non-negative.")
    if snr_proxy_weight < 0.0:
        raise ValueError("snr_proxy_weight must be non-negative.")
    if tracker_observability_min_singular_per_nm <= 0.0:
        raise ValueError("tracker_observability_min_singular_per_nm must be positive.")
    if tracker_observability_crlb_weight < 0.0:
        raise ValueError("tracker_observability_crlb_weight must be non-negative.")
    if tracker_observability_crlb_target_nm <= 0.0:
        raise ValueError("tracker_observability_crlb_target_nm must be positive.")
    if (
        tracker_observability_crlb_constraint_nm is not None
        and tracker_observability_crlb_constraint_nm <= 0.0
    ):
        raise ValueError(
            "tracker_observability_crlb_constraint_nm must be positive when set."
        )
    if tracker_observability_eps <= 0.0:
        raise ValueError("tracker_observability_eps must be positive.")

    tracker_active = (
        tracker_observability_weight > 0.0
        or tracker_observability_crlb_weight > 0.0
        or tracker_observability_crlb_constraint_nm is not None
    )
    if tracker_active:
        if tracker_piston_basis is None:
            raise ValueError(
                "tracker_piston_basis is required when characterization tracker "
                "observability regularization is enabled."
            )
        if tracker_wavelength_m is None:
            raise ValueError(
                "tracker_wavelength_m is required when characterization tracker "
                "observability regularization is enabled."
            )
        if tracker_wavelength_m <= 0.0:
            raise ValueError("tracker_wavelength_m must be positive.")

    spectro_idx = np.asarray(spectro_port_indices, dtype=np.intp)
    dark_idx = np.asarray(dark_port_indices, dtype=np.intp)
    if bright_port_indices is None:
        null_ports = set(dark_idx.tolist()) | set(spectro_idx.tolist())
        bright_idx = np.array(
            [i for i in range(n_modes) if i not in null_ports],
            dtype=np.intp,
        )
    else:
        bright_idx = np.asarray(bright_port_indices, dtype=np.intp)

    broadband_mode = (
        wl_design_m is not None
        and wavelengths_m is not None
        and stellar_fields_per_wl is not None
        and planet_fields_per_wl is not None
    )

    use_torch = backend == "torch"
    if backend == "auto":
        try:
            import torch  # noqa: F401, PLC0415
            use_torch = True
        except ImportError:
            use_torch = False

    if preserve_bright_rows and not use_torch:
        if not initial_params:
            raise ValueError("preserve_bright_rows requires initial_params.")
        return _optimize_preserving_bright_rows(
            stellar_field=stellar_field,
            planet_field=planet_field,
            n_modes=n_modes,
            dark_idx=dark_idx,
            spectro_idx=spectro_idx,
            bright_idx=bright_idx,
            null_floor=null_floor,
            n_restarts=n_restarts,
            seed=seed,
            max_iter=max_iter,
            throughput_weight=throughput_weight,
            snr_proxy_weight=snr_proxy_weight,
            initial_params=np.asarray(initial_params[0], dtype=np.float64),
            wl_design_m=wl_design_m,
            wavelengths_m=wavelengths_m,
            stellar_fields_per_wl=stellar_fields_per_wl,
            planet_fields_per_wl=planet_fields_per_wl,
            stellar_weights_per_wl=stellar_weights_per_wl,
            planet_weights_per_wl=planet_weights_per_wl,
            deployment_wavelengths_m=deployment_wavelengths_m,
            dac_bits=dac_bits,
            tracker_active=tracker_active,
            tracker_piston_basis=tracker_piston_basis,
            tracker_field=tracker_field,
            fallback_tracker_field=np.atleast_2d(
                np.asarray(stellar_field, dtype=np.complex128)
            )[0],
            tracker_wavelength_m=tracker_wavelength_m,
            tracker_observability_min_singular_per_nm=(
                tracker_observability_min_singular_per_nm
            ),
            tracker_observability_crlb_target_nm=tracker_observability_crlb_target_nm,
            tracker_observability_crlb_constraint_nm=(
                tracker_observability_crlb_constraint_nm
            ),
            tracker_observability_eps=tracker_observability_eps,
        )

    if use_torch:
        from nullsim.physics.characterization_torch import optimize_torch
        return optimize_torch(
            stellar_field=stellar_field,
            planet_field=planet_field,
            n_modes=n_modes,
            dark_idx=dark_idx,
            spectro_idx=spectro_idx,
            bright_idx=bright_idx,
            null_floor=null_floor,
            n_restarts=n_restarts,
            seed=seed,
            max_iter=max_iter,
            throughput_weight=throughput_weight,
            bright_weight=bright_weight,
            snr_proxy_weight=snr_proxy_weight,
            initial_params=initial_params,
            n_params=_n_params(n_modes),
            build_unitary_from_params=build_unitary_from_params,
            throughput_fn=_throughput,
            null_depth_fn=_null_depth,
            total_power_fn=_total_power,
            result_cls=CharacterizationResult,
            dac_bits=dac_bits,
            wl_design_m=wl_design_m,
            wavelengths_m=wavelengths_m,
            stellar_fields_per_wl=stellar_fields_per_wl,
            planet_fields_per_wl=planet_fields_per_wl,
            stellar_weights_per_wl=stellar_weights_per_wl,
            planet_weights_per_wl=planet_weights_per_wl,
            deployment_wavelengths_m=deployment_wavelengths_m,
            selection_wavelengths_m=selection_wavelengths_m,
            selection_stellar_fields_per_wl=selection_stellar_fields_per_wl,
            selection_planet_fields_per_wl=selection_planet_fields_per_wl,
            selection_stellar_weights_per_wl=selection_stellar_weights_per_wl,
            selection_planet_weights_per_wl=selection_planet_weights_per_wl,
            tracker_piston_basis=tracker_piston_basis,
            tracker_field=tracker_field,
            tracker_wavelength_m=tracker_wavelength_m,
            tracker_observability_weight=tracker_observability_weight,
            tracker_observability_min_singular_per_nm=(
                tracker_observability_min_singular_per_nm
            ),
            tracker_observability_crlb_weight=tracker_observability_crlb_weight,
            tracker_observability_crlb_target_nm=tracker_observability_crlb_target_nm,
            tracker_observability_crlb_constraint_nm=(
                tracker_observability_crlb_constraint_nm
            ),
            tracker_observability_eps=tracker_observability_eps,
            enforce_null_constraint=enforce_null_constraint,
            n_workers=n_workers,
            compile_objective=compile_objective,
        )
    star_fields = np.atleast_2d(np.asarray(stellar_field, dtype=np.complex128))
    planet_fields = np.atleast_2d(np.asarray(planet_field, dtype=np.complex128))
    total_stellar = _total_power(star_fields, "stellar_field")
    total_planet = _total_power(planet_fields, "planet_field")

    rng = np.random.default_rng(seed)

    if broadband_mode:
        wavelengths = np.asarray(wavelengths_m, dtype=np.float64)
        wl_scales = float(wl_design_m) / wavelengths
        star_fields_bb = [
            np.atleast_2d(np.asarray(field, dtype=np.complex128))
            for field in stellar_fields_per_wl or ()
        ]
        planet_fields_bb = [
            np.atleast_2d(np.asarray(field, dtype=np.complex128))
            for field in planet_fields_per_wl or ()
        ]
        if len(star_fields_bb) != wavelengths.size:
            raise ValueError("stellar_fields_per_wl length must match wavelengths_m.")
        if len(planet_fields_bb) != wavelengths.size:
            raise ValueError("planet_fields_per_wl length must match wavelengths_m.")
        total_stellar_bb = np.asarray(
            [_total_power(field, "stellar_fields_per_wl") for field in star_fields_bb],
            dtype=np.float64,
        )
        total_planet_bb = np.asarray(
            [_total_power(field, "planet_fields_per_wl") for field in planet_fields_bb],
            dtype=np.float64,
        )
        stellar_weights = _normalized_weights(
            stellar_weights_per_wl,
            wavelengths.size,
            name="stellar_weights_per_wl",
        )
        planet_weights = _normalized_weights(
            planet_weights_per_wl,
            wavelengths.size,
            name="planet_weights_per_wl",
        )

        def objective(x: NDArray[np.float64]) -> float:
            base = _throughput_objective_broadband(
                x, n_modes, planet_fields_bb, star_fields_bb, wl_scales,
                spectro_idx, bright_idx, total_planet_bb, total_stellar_bb,
                throughput_weight, bright_weight, snr_proxy_weight, null_floor,
                planet_weights, stellar_weights,
            )
            return base + _tracker_observability_penalty_numpy(
                x,
                n_modes=n_modes,
                bright_idx=bright_idx,
                tracker_active=tracker_active,
                tracker_piston_basis=tracker_piston_basis,
                tracker_field=tracker_field,
                fallback_tracker_field=star_fields[0],
                tracker_wavelength_m=tracker_wavelength_m,
                tracker_observability_weight=tracker_observability_weight,
                tracker_observability_min_singular_per_nm=(
                    tracker_observability_min_singular_per_nm
                ),
                tracker_observability_crlb_weight=tracker_observability_crlb_weight,
                tracker_observability_crlb_target_nm=(
                    tracker_observability_crlb_target_nm
                ),
                tracker_observability_eps=tracker_observability_eps,
            )

        def constraint(x: NDArray[np.float64]) -> float:
            return _null_constraint_broadband(
                x, n_modes, star_fields_bb, wl_scales, spectro_idx,
                total_stellar_bb, stellar_weights, null_floor,
            )

    else:

        def objective(x: NDArray[np.float64]) -> float:
            base = _throughput_objective(
                x, n_modes, planet_fields, star_fields, spectro_idx, bright_idx,
                total_planet, total_stellar, throughput_weight, bright_weight,
                snr_proxy_weight, null_floor,
            )
            return base + _tracker_observability_penalty_numpy(
                x,
                n_modes=n_modes,
                bright_idx=bright_idx,
                tracker_active=tracker_active,
                tracker_piston_basis=tracker_piston_basis,
                tracker_field=tracker_field,
                fallback_tracker_field=star_fields[0],
                tracker_wavelength_m=tracker_wavelength_m,
                tracker_observability_weight=tracker_observability_weight,
                tracker_observability_min_singular_per_nm=(
                    tracker_observability_min_singular_per_nm
                ),
                tracker_observability_crlb_weight=tracker_observability_crlb_weight,
                tracker_observability_crlb_target_nm=(
                    tracker_observability_crlb_target_nm
                ),
                tracker_observability_eps=tracker_observability_eps,
            )

        def constraint(x: NDArray[np.float64]) -> float:
            return _null_constraint(
                x, n_modes, star_fields, spectro_idx, total_stellar, null_floor,
            )

    feasibility_tol = 0.01 * null_floor
    selection_args = (
        selection_wavelengths_m,
        selection_stellar_fields_per_wl,
        selection_planet_fields_per_wl,
    )
    selection_mode = any(arg is not None for arg in selection_args)
    if selection_mode and not all(arg is not None for arg in selection_args):
        raise ValueError(
            "selection_wavelengths_m, selection_stellar_fields_per_wl, and "
            "selection_planet_fields_per_wl must be provided together."
        )
    if selection_mode and wl_design_m is None:
        raise ValueError("selection_wavelengths_m requires wl_design_m.")
    if selection_mode:
        selection_wavelengths = np.asarray(selection_wavelengths_m, dtype=np.float64)
        if (
            selection_wavelengths.ndim != 1
            or selection_wavelengths.size < 1
            or np.any(~np.isfinite(selection_wavelengths))
            or np.any(selection_wavelengths <= 0.0)
        ):
            raise ValueError("selection_wavelengths_m must be a positive 1-D array.")
        selection_wl_scales = float(wl_design_m) / selection_wavelengths
        selection_star_fields = [
            np.atleast_2d(np.asarray(field, dtype=np.complex128))
            for field in selection_stellar_fields_per_wl or ()
        ]
        selection_planet_fields = [
            np.atleast_2d(np.asarray(field, dtype=np.complex128))
            for field in selection_planet_fields_per_wl or ()
        ]
        if len(selection_star_fields) != selection_wavelengths.size:
            raise ValueError(
                "selection_stellar_fields_per_wl length must match "
                "selection_wavelengths_m."
            )
        if len(selection_planet_fields) != selection_wavelengths.size:
            raise ValueError(
                "selection_planet_fields_per_wl length must match "
                "selection_wavelengths_m."
            )
        selection_total_stellar = np.asarray(
            [
                _total_power(field, "selection_stellar_fields_per_wl")
                for field in selection_star_fields
            ],
            dtype=np.float64,
        )
        selection_total_planet = np.asarray(
            [
                _total_power(field, "selection_planet_fields_per_wl")
                for field in selection_planet_fields
            ],
            dtype=np.float64,
        )
        selection_stellar_weights = _normalized_weights(
            selection_stellar_weights_per_wl,
            selection_wavelengths.size,
            name="selection_stellar_weights_per_wl",
        )
        selection_planet_weights = _normalized_weights(
            selection_planet_weights_per_wl,
            selection_wavelengths.size,
            name="selection_planet_weights_per_wl",
        )
    else:
        selection_wl_scales = np.empty(0, dtype=np.float64)
        selection_star_fields = []
        selection_planet_fields = []
        selection_total_stellar = np.empty(0, dtype=np.float64)
        selection_total_planet = np.empty(0, dtype=np.float64)
        selection_stellar_weights = np.empty(0, dtype=np.float64)
        selection_planet_weights = np.empty(0, dtype=np.float64)

    def science_score(x: NDArray[np.float64]) -> float:
        if not selection_mode:
            return 0.0
        throughput, leak = _weighted_science_metrics_broadband(
            x,
            n_modes=n_modes,
            planet_fields_per_wl=selection_planet_fields,
            stellar_fields_per_wl=selection_star_fields,
            wl_scales=selection_wl_scales,
            spectro_idx=spectro_idx,
            total_planet_per_wl=selection_total_planet,
            total_stellar_per_wl=selection_total_stellar,
            planet_weights=selection_planet_weights,
            stellar_weights=selection_stellar_weights,
        )
        return _systematic_snr_proxy(throughput, leak, null_floor)

    def candidate_key(
        *,
        objective_value: float,
        constraint_margin: float,
        science_score_value: float,
    ) -> tuple[int, float, float, float]:
        score = float(science_score_value) if np.isfinite(science_score_value) else 0.0
        constraint_ok = (
            (not enforce_null_constraint) or constraint_margin >= -feasibility_tol
        )
        if constraint_ok:
            return (0, -score, float(objective_value), 0.0)
        return (1, float(-constraint_margin), -score, float(objective_value))

    def tracker_constraint(x: NDArray[np.float64]) -> float:
        if tracker_observability_crlb_constraint_nm is None:
            return 1.0
        _, crlb_rms_nm, _, _ = _tracker_observability_metrics_numpy(
            x,
            n_modes=n_modes,
            bright_idx=bright_idx,
            tracker_active=tracker_active,
            tracker_piston_basis=tracker_piston_basis,
            tracker_field=tracker_field,
            fallback_tracker_field=star_fields[0],
            tracker_wavelength_m=tracker_wavelength_m,
            tracker_observability_min_singular_per_nm=(
                tracker_observability_min_singular_per_nm
            ),
            tracker_observability_crlb_target_nm=tracker_observability_crlb_target_nm,
            tracker_observability_eps=tracker_observability_eps,
        )
        if crlb_rms_nm is None or not np.isfinite(crlb_rms_nm):
            return -np.inf
        return float(tracker_observability_crlb_constraint_nm) - float(crlb_rms_nm)

    n_params = _n_params(n_modes)
    starting_points: list[NDArray[np.float64]] = [
        np.asarray(p, dtype=np.float64).copy() for p in initial_params
    ]
    for p in starting_points:
        if p.shape != (n_params,):
            raise ValueError(
                f"initial_params entries must have shape ({n_params},), got {p.shape}."
            )
    for _ in range(n_restarts):
        starting_points.append(rng.uniform(0.0, _TWO_PI, n_params))

    best_params = starting_points[0].copy()
    best_objective = objective(best_params)
    n_completed = 0
    n_feasible = 0
    best_margin = float(constraint(best_params))
    best_is_feasible = (not enforce_null_constraint) or best_margin >= -feasibility_tol
    best_science_score = science_score(best_params)
    best_key = candidate_key(
        objective_value=best_objective,
        constraint_margin=best_margin,
        science_score_value=best_science_score,
    )
    constraints = []
    if enforce_null_constraint:
        constraints.append({"type": "ineq", "fun": constraint})
    if tracker_observability_crlb_constraint_nm is not None:
        constraints.append({"type": "ineq", "fun": tracker_constraint})
    for x0 in starting_points:
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            constraints=constraints,
            options={"maxiter": max_iter, "ftol": 1.0e-9},
        )
        n_completed += 1
        result_x = np.asarray(result.x, dtype=np.float64)
        result_fun = float(objective(result_x))
        result_margin = float(constraint(result_x))
        result_science_score = science_score(result_x)
        if (
            not np.isfinite(result_fun)
            or not np.isfinite(result_margin)
            or not np.isfinite(result_science_score)
        ):
            continue
        result_is_feasible = (
            (not enforce_null_constraint) or result_margin >= -feasibility_tol
        )
        if result_is_feasible:
            n_feasible += 1
        result_key = candidate_key(
            objective_value=result_fun,
            constraint_margin=result_margin,
            science_score_value=result_science_score,
        )
        if result_key < best_key:
            best_key = result_key
            best_objective = result_fun
            best_margin = result_margin
            best_science_score = result_science_score
            best_params = result_x
            best_is_feasible = result_is_feasible

    if dac_bits is not None:
        best_params = _quantize_phases(best_params, dac_bits)
        best_objective = float(objective(best_params))
        best_margin = float(constraint(best_params))
        best_is_feasible = best_margin >= -feasibility_tol

    unitary = build_unitary_from_params(best_params, n_modes)
    spectro_null = _null_depth(unitary, star_fields, spectro_idx)
    dark_null = _null_depth(unitary, star_fields, dark_idx)
    spectro_throughput = _throughput(unitary, planet_fields, spectro_idx)
    bright_fraction = _throughput(unitary, star_fields, bright_idx)
    (
        tracker_singular,
        tracker_crlb_rms_nm,
        tracker_loss,
        tracker_crlb_loss,
    ) = _tracker_observability_metrics_numpy(
        best_params,
        n_modes=n_modes,
        bright_idx=bright_idx,
        tracker_active=tracker_active,
        tracker_piston_basis=tracker_piston_basis,
        tracker_field=tracker_field,
        fallback_tracker_field=star_fields[0],
        tracker_wavelength_m=tracker_wavelength_m,
        tracker_observability_min_singular_per_nm=(
            tracker_observability_min_singular_per_nm
        ),
        tracker_observability_crlb_target_nm=tracker_observability_crlb_target_nm,
        tracker_observability_eps=tracker_observability_eps,
    )

    return CharacterizationResult(
        params=best_params,
        spectro_null_depth=float(spectro_null),
        dark_null_depth=float(dark_null),
        spectro_throughput=float(spectro_throughput),
        bright_stellar_fraction=float(bright_fraction),
        objective_value=float(best_objective),
        n_restarts_completed=n_completed,
        optimization_success=bool(
            (not enforce_null_constraint or (n_feasible > 0 and best_is_feasible))
        ),
        constraint_margin=float(best_margin),
        tracker_singular_values_per_nm=tracker_singular,
        tracker_crlb_rms_nm=tracker_crlb_rms_nm,
        tracker_loss=tracker_loss,
        tracker_crlb_loss=tracker_crlb_loss,
    )


def _optimize_preserving_bright_rows(
    *,
    stellar_field: NDArray[np.complex128],
    planet_field: NDArray[np.complex128],
    n_modes: int,
    dark_idx: NDArray[np.intp],
    spectro_idx: NDArray[np.intp],
    bright_idx: NDArray[np.intp],
    null_floor: float,
    n_restarts: int,
    seed: int | None,
    max_iter: int,
    throughput_weight: float,
    snr_proxy_weight: float,
    initial_params: NDArray[np.float64],
    wl_design_m: float | None,
    wavelengths_m: Sequence[float] | None,
    stellar_fields_per_wl: Sequence[NDArray[np.complex128]] | None,
    planet_fields_per_wl: Sequence[NDArray[np.complex128]] | None,
    stellar_weights_per_wl: Sequence[float] | None,
    planet_weights_per_wl: Sequence[float] | None,
    deployment_wavelengths_m: Sequence[float] | None,
    dac_bits: int | None,
    tracker_active: bool,
    tracker_piston_basis: NDArray[np.float64] | None,
    tracker_field: NDArray[np.complex128] | None,
    fallback_tracker_field: NDArray[np.complex128],
    tracker_wavelength_m: float | None,
    tracker_observability_min_singular_per_nm: float,
    tracker_observability_crlb_target_nm: float,
    tracker_observability_crlb_constraint_nm: float | None,
    tracker_observability_eps: float,
) -> CharacterizationResult:
    if spectro_idx.size != 1:
        raise ValueError("preserve_bright_rows currently supports one spectro port.")
    if initial_params.shape != (_n_params(n_modes),):
        raise ValueError(
            "preserve_bright_rows initial_params must have shape "
            f"({_n_params(n_modes)},), got {initial_params.shape}."
        )

    u0 = build_unitary_from_params(initial_params, n_modes)
    null_idx = np.concatenate((dark_idx, spectro_idx)).astype(np.intp, copy=False)
    if np.intersect1d(bright_idx, null_idx).size:
        raise ValueError("bright and dark/spectro ports must be disjoint.")
    null_rows_design = u0[null_idx, :]
    n_null_rows = int(null_idx.size)

    star_fields = np.atleast_2d(np.asarray(stellar_field, dtype=np.complex128))
    planet_fields = np.atleast_2d(np.asarray(planet_field, dtype=np.complex128))
    total_stellar = _total_power(star_fields, "stellar_field")
    total_planet = _total_power(planet_fields, "planet_field")

    broadband_mode = (
        wl_design_m is not None
        and wavelengths_m is not None
        and stellar_fields_per_wl is not None
        and planet_fields_per_wl is not None
    )
    if broadband_mode:
        wavelengths = np.asarray(wavelengths_m, dtype=np.float64)
        stack0 = build_unitary_stack_from_params(
            params=initial_params,
            n_modes=n_modes,
            wavelengths_m=wavelengths,
            wl_design_m=float(wl_design_m),
            dac_bits=None,
        )
        null_rows_per_wl = [stack0[null_idx, :, i] for i in range(wavelengths.size)]
        star_fields_bb = [
            np.atleast_2d(np.asarray(field, dtype=np.complex128))
            for field in stellar_fields_per_wl or ()
        ]
        planet_fields_bb = [
            np.atleast_2d(np.asarray(field, dtype=np.complex128))
            for field in planet_fields_per_wl or ()
        ]
        if len(star_fields_bb) != wavelengths.size:
            raise ValueError("stellar_fields_per_wl length must match wavelengths_m.")
        if len(planet_fields_bb) != wavelengths.size:
            raise ValueError("planet_fields_per_wl length must match wavelengths_m.")
        total_stellar_bb = np.asarray(
            [_total_power(field, "stellar_fields_per_wl") for field in star_fields_bb],
            dtype=np.float64,
        )
        total_planet_bb = np.asarray(
            [_total_power(field, "planet_fields_per_wl") for field in planet_fields_bb],
            dtype=np.float64,
        )
        stellar_weights = _normalized_weights(
            stellar_weights_per_wl,
            wavelengths.size,
            name="stellar_weights_per_wl",
        )
        planet_weights = _normalized_weights(
            planet_weights_per_wl,
            wavelengths.size,
            name="planet_weights_per_wl",
        )
    else:
        null_rows_per_wl = [null_rows_design]
        star_fields_bb = [star_fields]
        planet_fields_bb = [planet_fields]
        total_stellar_bb = np.asarray([total_stellar], dtype=np.float64)
        total_planet_bb = np.asarray([total_planet], dtype=np.float64)
        stellar_weights = np.ones(1, dtype=np.float64)
        planet_weights = np.ones(1, dtype=np.float64)

    def coeffs_from_x(x: NDArray[np.float64]) -> NDArray[np.complex128]:
        raw = np.asarray(x[:n_null_rows] + 1j * x[n_null_rows:], dtype=np.complex128)
        norm = float(np.linalg.norm(raw))
        if norm <= 0.0 or not np.isfinite(norm):
            return np.full(n_null_rows, np.nan + 1j * np.nan, dtype=np.complex128)
        return raw / norm

    def row_power_fraction(
        row: NDArray[np.complex128],
        fields: NDArray[np.complex128],
        total_power: float,
    ) -> float:
        output = np.atleast_2d(fields) @ row.T
        return float(np.sum(np.abs(output) ** 2) / total_power)

    def stack_port_power_fraction(
        stack: NDArray[np.complex128],
        fields_per_wl: Sequence[NDArray[np.complex128]],
        total_power_per_wl: NDArray[np.float64],
        weights_per_wl: NDArray[np.float64],
    ) -> float:
        if stack.shape[2] != len(fields_per_wl):
            raise ValueError("stack wavelength count must match field count.")
        weighted = 0.0
        for i, fields in enumerate(fields_per_wl):
            output = np.atleast_2d(fields) @ stack[spectro_idx, :, i].T
            weighted += float(weights_per_wl[i]) * (
                float(np.sum(np.abs(output) ** 2)) / float(total_power_per_wl[i])
            )
        return float(weighted)

    def objective(x: NDArray[np.float64]) -> float:
        coeffs = coeffs_from_x(x)
        if np.any(~np.isfinite(coeffs)):
            return np.inf
        throughput_term = 0.0
        leak_term = 0.0
        for i, rows in enumerate(null_rows_per_wl):
            spectro_row = coeffs @ rows
            planet_throughput = row_power_fraction(
                spectro_row,
                planet_fields_bb[i],
                float(total_planet_bb[i]),
            )
            stellar_leak = row_power_fraction(
                spectro_row,
                star_fields_bb[i],
                float(total_stellar_bb[i]),
            )
            throughput_term += float(planet_weights[i]) * planet_throughput
            leak_term += float(stellar_weights[i]) * stellar_leak
        return float(
            -float(throughput_weight) * throughput_term
            - float(snr_proxy_weight)
            * _snr_proxy(throughput_term, leak_term, null_floor)
        )

    def constraint(x: NDArray[np.float64]) -> float:
        coeffs = coeffs_from_x(x)
        if np.any(~np.isfinite(coeffs)):
            return -np.inf
        null = 0.0
        for i, rows in enumerate(null_rows_per_wl):
            spectro_row = coeffs @ rows
            null += (
                float(stellar_weights[i])
                * row_power_fraction(
                    spectro_row,
                    star_fields_bb[i],
                    float(total_stellar_bb[i]),
                )
            )
        return float(null_floor - null)

    rng = np.random.default_rng(seed)
    starts: list[NDArray[np.float64]] = []
    existing_spectro = np.zeros(n_null_rows, dtype=np.complex128)
    existing_spectro[-1] = 1.0
    starts.append(np.concatenate((existing_spectro.real, existing_spectro.imag)))
    for _ in range(n_restarts):
        raw = rng.normal(size=n_null_rows) + 1j * rng.normal(size=n_null_rows)
        raw /= np.linalg.norm(raw)
        starts.append(np.concatenate((raw.real, raw.imag)).astype(np.float64))

    best_x = starts[0].copy()
    best_objective = float(objective(best_x))
    best_margin = float(constraint(best_x))
    feasibility_tol = 0.01 * null_floor
    best_key = _candidate_selection_key(
        objective_value=best_objective,
        constraint_margin=best_margin,
        feasibility_tol=feasibility_tol,
    )
    n_completed = 0
    n_feasible = 0
    for x0 in starts:
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            constraints=[{"type": "ineq", "fun": constraint}],
            options={"maxiter": max_iter, "ftol": 1.0e-10},
        )
        n_completed += 1
        x = np.asarray(result.x, dtype=np.float64)
        value = float(objective(x))
        margin = float(constraint(x))
        if not np.isfinite(value) or not np.isfinite(margin):
            continue
        if margin >= -feasibility_tol:
            n_feasible += 1
        key = _candidate_selection_key(
            objective_value=value,
            constraint_margin=margin,
            feasibility_tol=feasibility_tol,
        )
        if key < best_key:
            best_x = x
            best_objective = value
            best_margin = margin
            best_key = key

    coeffs = coeffs_from_x(best_x)
    rotation = _row_rotation_with_last_row(coeffs)
    target_u = u0.copy()
    target_u[null_idx, :] = rotation @ null_rows_design
    decomp = decompose_clements(target_u, atol=1.0e-5)
    best_params = np.concatenate(
        (
            np.asarray(decomp.theta, dtype=np.float64),
            np.asarray(decomp.phi, dtype=np.float64),
            np.asarray(decomp.alpha, dtype=np.float64),
        )
    )
    if dac_bits is not None:
        best_params = _quantize_phases(best_params, dac_bits)
        best_margin = float(constraint(best_x))

    unitary = build_unitary_from_params(best_params, n_modes)
    stack_wavelengths = _resolve_deployment_wavelengths(
        deployment_wavelengths_m=deployment_wavelengths_m,
        wavelengths_m=wavelengths_m,
        wl_design_m=wl_design_m,
    )
    stack_design_m = (
        float(wl_design_m)
        if wl_design_m is not None
        else float(stack_wavelengths[stack_wavelengths.size // 2])
    )
    optimal_u_stack = build_unitary_stack_from_params(
        params=best_params,
        n_modes=n_modes,
        wavelengths_m=stack_wavelengths,
        wl_design_m=stack_design_m,
        dac_bits=None,
    )
    if broadband_mode:
        constraint_wavelengths = wavelengths
        constraint_star_fields = star_fields_bb
        constraint_planet_fields = planet_fields_bb
        constraint_star_totals = total_stellar_bb
        constraint_planet_totals = total_planet_bb
        constraint_star_weights = stellar_weights
        constraint_planet_weights = planet_weights
    else:
        constraint_wavelengths = np.asarray([stack_design_m], dtype=np.float64)
        constraint_star_fields = [star_fields]
        constraint_planet_fields = [planet_fields]
        constraint_star_totals = np.asarray([total_stellar], dtype=np.float64)
        constraint_planet_totals = np.asarray([total_planet], dtype=np.float64)
        constraint_star_weights = np.ones(1, dtype=np.float64)
        constraint_planet_weights = np.ones(1, dtype=np.float64)
    constraint_stack = build_unitary_stack_from_params(
        params=best_params,
        n_modes=n_modes,
        wavelengths_m=constraint_wavelengths,
        wl_design_m=stack_design_m,
        dac_bits=None,
    )
    deployed_leak = stack_port_power_fraction(
        constraint_stack,
        constraint_star_fields,
        constraint_star_totals,
        constraint_star_weights,
    )
    deployed_throughput = stack_port_power_fraction(
        constraint_stack,
        constraint_planet_fields,
        constraint_planet_totals,
        constraint_planet_weights,
    )
    best_margin = float(null_floor - deployed_leak)
    best_objective = float(
        -float(throughput_weight) * deployed_throughput
        - float(snr_proxy_weight)
        * _snr_proxy(deployed_throughput, deployed_leak, null_floor)
    )
    spectro_null = _null_depth(unitary, star_fields, spectro_idx)
    dark_null = _null_depth(unitary, star_fields, dark_idx)
    spectro_throughput = _throughput(unitary, planet_fields, spectro_idx)
    bright_fraction = _throughput(unitary, star_fields, bright_idx)
    (
        tracker_singular,
        tracker_crlb_rms_nm,
        tracker_loss,
        tracker_crlb_loss,
    ) = _tracker_observability_metrics_numpy(
        best_params,
        n_modes=n_modes,
        bright_idx=bright_idx,
        tracker_active=tracker_active,
        tracker_piston_basis=tracker_piston_basis,
        tracker_field=tracker_field,
        fallback_tracker_field=fallback_tracker_field,
        tracker_wavelength_m=tracker_wavelength_m,
        tracker_observability_min_singular_per_nm=(
            tracker_observability_min_singular_per_nm
        ),
        tracker_observability_crlb_target_nm=tracker_observability_crlb_target_nm,
        tracker_observability_eps=tracker_observability_eps,
    )
    tracker_constraint_met = True
    if tracker_observability_crlb_constraint_nm is not None:
        tracker_constraint_met = (
            tracker_crlb_rms_nm is not None
            and np.isfinite(tracker_crlb_rms_nm)
            and float(tracker_crlb_rms_nm)
            <= float(tracker_observability_crlb_constraint_nm) * (1.0 + 1.0e-9)
        )
    return CharacterizationResult(
        params=best_params,
        spectro_null_depth=float(spectro_null),
        dark_null_depth=float(dark_null),
        spectro_throughput=float(spectro_throughput),
        bright_stellar_fraction=float(bright_fraction),
        objective_value=float(best_objective),
        n_restarts_completed=n_completed,
        optimization_success=bool(
            n_feasible > 0
            and best_margin >= -feasibility_tol
            and tracker_constraint_met
        ),
        constraint_margin=float(best_margin),
        tracker_singular_values_per_nm=tracker_singular,
        tracker_crlb_rms_nm=tracker_crlb_rms_nm,
        tracker_loss=tracker_loss,
        tracker_crlb_loss=tracker_crlb_loss,
        optimal_u_stack=optimal_u_stack,
        optimal_u_stack_wavelengths_m=stack_wavelengths,
    )


def _resolve_deployment_wavelengths(
    *,
    deployment_wavelengths_m: Sequence[float] | None,
    wavelengths_m: Sequence[float] | None,
    wl_design_m: float | None,
) -> NDArray[np.float64]:
    if deployment_wavelengths_m is not None:
        raw = deployment_wavelengths_m
    elif wavelengths_m is not None:
        raw = wavelengths_m
    elif wl_design_m is not None:
        raw = (float(wl_design_m),)
    else:
        raw = (1.0,)
    wavelengths = np.asarray(raw, dtype=np.float64)
    if wavelengths.ndim != 1 or wavelengths.size < 1:
        raise ValueError("deployment wavelengths must be a non-empty 1-D array.")
    if np.any(~np.isfinite(wavelengths)) or np.any(wavelengths <= 0.0):
        raise ValueError("deployment wavelengths must be positive and finite.")
    return wavelengths


def _row_rotation_with_last_row(
    last_row: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    row = np.asarray(last_row, dtype=np.complex128).reshape(-1)
    norm = float(np.linalg.norm(row))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("last_row must have finite nonzero norm.")
    row = row / norm
    rows: list[NDArray[np.complex128]] = []
    for basis_index in range(row.size):
        candidate = np.zeros(row.size, dtype=np.complex128)
        candidate[basis_index] = 1.0
        for existing in (*rows, row):
            candidate = candidate - np.vdot(existing, candidate) * existing
        candidate_norm = float(np.linalg.norm(candidate))
        if candidate_norm > 1.0e-12:
            rows.append(candidate / candidate_norm)
        if len(rows) == row.size - 1:
            break
    if len(rows) != row.size - 1:
        raise ValueError("could not construct a unitary row completion.")
    rotation = np.vstack((*rows, row))
    if not np.allclose(rotation @ rotation.conj().T, np.eye(row.size), atol=1.0e-8):
        raise ValueError("row completion is not unitary.")
    return rotation


def _throughput_objective(
    params: NDArray[np.float64],
    n_modes: int,
    planet_fields: NDArray[np.complex128],
    stellar_fields: NDArray[np.complex128],
    spectro_idx: NDArray[np.intp],
    bright_idx: NDArray[np.intp],
    total_planet: float,
    total_stellar: float,
    throughput_weight: float,
    bright_weight: float,
    snr_proxy_weight: float,
    null_floor: float,
) -> float:
    unitary = build_unitary_from_params(params, n_modes)
    planet_throughput = _throughput(unitary, planet_fields, spectro_idx, total_planet)
    stellar_leak = _null_depth(unitary, stellar_fields, spectro_idx, total_stellar)
    snr_proxy = _snr_proxy(planet_throughput, stellar_leak, null_floor)
    return (
        -throughput_weight * planet_throughput
        - bright_weight * _throughput(unitary, stellar_fields, bright_idx, total_stellar)
        - snr_proxy_weight * snr_proxy
    )


def _null_constraint(
    params: NDArray[np.float64],
    n_modes: int,
    stellar_fields: NDArray[np.complex128],
    spectro_idx: NDArray[np.intp],
    total_stellar: float,
    threshold: float,
) -> float:
    unitary = build_unitary_from_params(params, n_modes)
    return float(threshold - _null_depth(unitary, stellar_fields, spectro_idx,
                                         total_stellar))


def _throughput_objective_broadband(
    params: NDArray[np.float64],
    n_modes: int,
    planet_fields_per_wl: Sequence[NDArray[np.complex128]],
    stellar_fields_per_wl: Sequence[NDArray[np.complex128]],
    wl_scales: NDArray[np.float64],
    spectro_idx: NDArray[np.intp],
    bright_idx: NDArray[np.intp],
    total_planet_per_wl: NDArray[np.float64],
    total_stellar_per_wl: NDArray[np.float64],
    throughput_weight: float,
    bright_weight: float,
    snr_proxy_weight: float,
    null_floor: float,
    planet_weights: NDArray[np.float64],
    stellar_weights: NDArray[np.float64],
) -> float:
    total = 0.0
    weighted_planet = 0.0
    weighted_leak = 0.0
    for index, scale in enumerate(wl_scales):
        scaled = _apply_wavelength_scale(params, n_modes, float(scale))
        unitary = build_unitary_from_params(scaled, n_modes)
        planet_throughput = _throughput(
            unitary, planet_fields_per_wl[index], spectro_idx,
            total_planet_per_wl[index],
        )
        stellar_leak = _null_depth(
            unitary, stellar_fields_per_wl[index], spectro_idx,
            total_stellar_per_wl[index],
        )
        weighted_planet += float(planet_weights[index]) * planet_throughput
        weighted_leak += float(stellar_weights[index]) * stellar_leak
        total += (
            -throughput_weight
            * float(planet_weights[index])
            * planet_throughput
            - bright_weight
            * float(stellar_weights[index])
            * _throughput(
                unitary, stellar_fields_per_wl[index], bright_idx,
                total_stellar_per_wl[index],
            )
        )
    total -= snr_proxy_weight * _snr_proxy(weighted_planet, weighted_leak, null_floor)
    return float(total)


def _snr_proxy(planet_throughput: float, stellar_leak: float, null_floor: float) -> float:
    """Return a dimensionless characterization SNR proxy.

    Approaching the configured leak floor from above is rewarded as
    ``1 / sqrt(leak)`` because stellar leakage dominates characterization
    noise in the KILO Keck regime. The reward saturates AT the floor:
    static leak below ``null_floor`` is worthless at deployment (the
    realized piston/scintillation floor dominates there), and an unbounded
    reward made the optimizer trade away planet throughput for un-cashable
    null depth (review follow-up 2026-06-10).
    """

    leak = max(float(stellar_leak), float(null_floor))
    return float(planet_throughput) * float(np.sqrt(float(null_floor) / leak))


def _systematic_snr_proxy(
    planet_throughput: float,
    stellar_leak: float,
    null_floor: float,
) -> float:
    """Rank restart candidates by throughput after the requested leak floor."""

    leak = max(float(stellar_leak), float(null_floor))
    throughput = float(planet_throughput)
    if not np.isfinite(leak) or leak <= 0.0:
        return 0.0
    if not np.isfinite(throughput) or throughput <= 0.0:
        return 0.0
    return throughput / leak


def _weighted_science_metrics_broadband(
    params: NDArray[np.float64],
    *,
    n_modes: int,
    planet_fields_per_wl: Sequence[NDArray[np.complex128]],
    stellar_fields_per_wl: Sequence[NDArray[np.complex128]],
    wl_scales: NDArray[np.float64],
    spectro_idx: NDArray[np.intp],
    total_planet_per_wl: NDArray[np.float64],
    total_stellar_per_wl: NDArray[np.float64],
    planet_weights: NDArray[np.float64],
    stellar_weights: NDArray[np.float64],
) -> tuple[float, float]:
    weighted_planet = 0.0
    weighted_leak = 0.0
    for index, scale in enumerate(wl_scales):
        scaled = _apply_wavelength_scale(params, n_modes, float(scale))
        unitary = build_unitary_from_params(scaled, n_modes)
        weighted_planet += float(planet_weights[index]) * _throughput(
            unitary,
            planet_fields_per_wl[index],
            spectro_idx,
            total_planet_per_wl[index],
        )
        weighted_leak += float(stellar_weights[index]) * _null_depth(
            unitary,
            stellar_fields_per_wl[index],
            spectro_idx,
            total_stellar_per_wl[index],
        )
    return float(weighted_planet), float(weighted_leak)


def _null_constraint_broadband(
    params: NDArray[np.float64],
    n_modes: int,
    stellar_fields_per_wl: Sequence[NDArray[np.complex128]],
    wl_scales: NDArray[np.float64],
    spectro_idx: NDArray[np.intp],
    total_stellar_per_wl: NDArray[np.float64],
    stellar_weights: NDArray[np.float64],
    threshold: float,
) -> float:
    mean_null = 0.0
    for index, scale in enumerate(wl_scales):
        scaled = _apply_wavelength_scale(params, n_modes, float(scale))
        unitary = build_unitary_from_params(scaled, n_modes)
        mean_null += float(stellar_weights[index]) * _null_depth(
            unitary, stellar_fields_per_wl[index], spectro_idx,
            total_stellar_per_wl[index],
        )
    return float(threshold - mean_null)


def _normalized_weights(
    weights: Sequence[float] | None,
    n_items: int,
    *,
    name: str,
) -> NDArray[np.float64]:
    if n_items <= 0:
        raise ValueError("weight length must be positive.")
    if weights is None:
        return np.full(n_items, 1.0 / float(n_items), dtype=np.float64)
    arr = np.asarray(weights, dtype=np.float64)
    if arr.shape != (n_items,):
        raise ValueError(f"{name} must have shape ({n_items},), got {arr.shape}.")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError(f"{name} must contain finite non-negative weights.")
    total = float(np.sum(arr))
    if total <= 0.0:
        raise ValueError(f"{name} must contain at least one positive weight.")
    return (arr / total).astype(np.float64, copy=False)


def _candidate_selection_key(
    *,
    objective_value: float,
    constraint_margin: float,
    feasibility_tol: float,
) -> tuple[int, float, float]:
    if constraint_margin >= -feasibility_tol:
        return (0, float(objective_value), 0.0)
    # Failed characterization retunes must stay conservative: a high-throughput
    # spectro row with a large broadband stellar leak is worse than a lower-SNR
    # row that remains close to the requested null floor.
    return (1, float(-constraint_margin), float(objective_value))


def _tracker_observability_penalty_numpy(
    params: NDArray[np.float64],
    *,
    n_modes: int,
    bright_idx: NDArray[np.intp],
    tracker_active: bool,
    tracker_piston_basis: NDArray[np.float64] | None,
    tracker_field: NDArray[np.complex128] | None,
    fallback_tracker_field: NDArray[np.complex128],
    tracker_wavelength_m: float | None,
    tracker_observability_weight: float,
    tracker_observability_min_singular_per_nm: float,
    tracker_observability_crlb_weight: float,
    tracker_observability_crlb_target_nm: float,
    tracker_observability_eps: float,
) -> float:
    if not tracker_active:
        return 0.0
    _singular, _crlb_rms, tracker_loss, tracker_crlb_loss = (
        _tracker_observability_metrics_numpy(
            params,
            n_modes=n_modes,
            bright_idx=bright_idx,
            tracker_active=tracker_active,
            tracker_piston_basis=tracker_piston_basis,
            tracker_field=tracker_field,
            fallback_tracker_field=fallback_tracker_field,
            tracker_wavelength_m=tracker_wavelength_m,
            tracker_observability_min_singular_per_nm=(
                tracker_observability_min_singular_per_nm
            ),
            tracker_observability_crlb_target_nm=tracker_observability_crlb_target_nm,
            tracker_observability_eps=tracker_observability_eps,
        )
    )
    return float(
        tracker_observability_weight * tracker_loss
        + tracker_observability_crlb_weight * tracker_crlb_loss
    )


def _tracker_observability_metrics_numpy(
    params: NDArray[np.float64],
    *,
    n_modes: int,
    bright_idx: NDArray[np.intp],
    tracker_active: bool,
    tracker_piston_basis: NDArray[np.float64] | None,
    tracker_field: NDArray[np.complex128] | None,
    fallback_tracker_field: NDArray[np.complex128],
    tracker_wavelength_m: float | None,
    tracker_observability_min_singular_per_nm: float,
    tracker_observability_crlb_target_nm: float,
    tracker_observability_eps: float,
) -> tuple[NDArray[np.float64] | None, float | None, float, float]:
    if not tracker_active:
        return None, None, 0.0, 0.0
    if tracker_piston_basis is None or tracker_wavelength_m is None:
        raise ValueError(
            "tracker_piston_basis and tracker_wavelength_m are required "
            "when characterization tracker observability is enabled."
        )
    matrix = _tracker_observability_matrix_numpy(
        params,
        n_modes=n_modes,
        bright_idx=bright_idx,
        tracker_piston_basis=tracker_piston_basis,
        tracker_field=tracker_field,
        fallback_tracker_field=fallback_tracker_field,
        tracker_wavelength_m=tracker_wavelength_m,
        eps=tracker_observability_eps,
    )
    singular = np.linalg.svd(matrix, compute_uv=False)
    regularized = _regularized_singular_values_numpy(
        matrix,
        eps=tracker_observability_eps,
    )
    target = float(tracker_observability_min_singular_per_nm)
    tiny = float(tracker_observability_eps)
    log_deficit = np.maximum(
        np.log(target) - np.log(regularized + tiny),
        0.0,
    )
    n_dof = int(matrix.shape[1])
    tracker_loss = float(np.sum(log_deficit**2) / float(n_dof))
    std_nm = 1.0 / np.maximum(regularized, tiny)
    tracker_crlb_loss = float(
        np.mean((std_nm / float(tracker_observability_crlb_target_nm)) ** 2)
    )
    return (
        singular.astype(np.float64, copy=True),
        _tracker_crlb_rms_nm_from_singular_values(
            singular,
            eps=tracker_observability_eps,
        ),
        tracker_loss,
        tracker_crlb_loss,
    )


def _tracker_observability_matrix_numpy(
    params: NDArray[np.float64],
    *,
    n_modes: int,
    bright_idx: NDArray[np.intp],
    tracker_piston_basis: NDArray[np.float64],
    tracker_field: NDArray[np.complex128] | None,
    fallback_tracker_field: NDArray[np.complex128],
    tracker_wavelength_m: float,
    eps: float,
) -> NDArray[np.float64]:
    basis = np.asarray(tracker_piston_basis, dtype=np.float64)
    if basis.ndim != 2 or basis.shape[0] != n_modes:
        raise ValueError(
            "tracker_piston_basis must have shape "
            f"({n_modes}, n_differential_modes)."
        )
    field = (
        np.asarray(tracker_field, dtype=np.complex128)
        if tracker_field is not None
        else np.asarray(fallback_tracker_field, dtype=np.complex128)
    )
    if field.ndim == 1:
        field_by_wl = field[:, np.newaxis, np.newaxis]
    elif field.ndim == 2 and field.shape[0] == n_modes:
        field_by_wl = field[:, np.newaxis, :]
    else:
        raise ValueError(
            "tracker_field must have shape (n_modes,) or "
            f"({n_modes}, n_states)."
        )
    unitary = build_unitary_from_params(params, n_modes)
    response = intensity_response_matrix(
        u_stack=unitary[:, :, np.newaxis],
        input_fields=field_by_wl,
        wavelengths_m=np.array([float(tracker_wavelength_m)], dtype=np.float64),
        port_indices=bright_idx,
    )
    denom = np.sqrt(np.maximum(response.reference_power, float(eps)))
    return (
        (response.response_power_per_m @ basis)
        / denom[:, np.newaxis]
        * 1.0e-9
    )


def _regularized_singular_values_numpy(
    matrix: NDArray[np.float64],
    *,
    eps: float,
) -> NDArray[np.float64]:
    gram = matrix.T @ matrix
    evals = np.linalg.eigvalsh(gram + (float(eps) ** 2) * np.eye(gram.shape[0]))
    return np.sqrt(np.maximum(evals, float(eps) ** 2))


def _tracker_crlb_rms_nm_from_singular_values(
    singular_values_per_nm: NDArray[np.float64],
    *,
    eps: float,
) -> float:
    values = np.asarray(singular_values_per_nm, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return float("inf")
    clipped = np.maximum(values, float(eps))
    return float(np.sqrt(np.mean(1.0 / (clipped * clipped))))


def _throughput(
    unitary: NDArray[np.complex128],
    fields: NDArray[np.complex128],
    indices: NDArray[np.intp],
    total_power: float | None = None,
) -> float:
    fields_2d = np.atleast_2d(fields)
    denominator = _total_power(fields_2d, "fields") if total_power is None else total_power
    output = fields_2d @ unitary.T
    return float(np.sum(np.abs(output[:, indices]) ** 2) / denominator)


def _null_depth(
    unitary: NDArray[np.complex128],
    stellar_fields: NDArray[np.complex128],
    indices: NDArray[np.intp],
    total_stellar: float | None = None,
) -> float:
    return _throughput(unitary, stellar_fields, indices, total_stellar)


def _total_power(fields: NDArray[np.complex128], name: str) -> float:
    total = float(np.sum(np.abs(fields) ** 2))
    if total <= 0.0:
        raise ValueError(f"{name} has zero total power.")
    return total


def _n_params(n_modes: int) -> int:
    n_mzi = n_modes * (n_modes - 1) // 2
    return 2 * n_mzi + n_modes


def _split_params(
    params: NDArray[np.float64],
    n_modes: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    n_mzi = n_modes * (n_modes - 1) // 2
    expected = 2 * n_mzi + n_modes
    if params.shape != (expected,):
        raise ValueError(f"params must have shape ({expected},), got {params.shape}.")
    return (
        params[0:n_mzi],
        params[n_mzi : 2 * n_mzi],
        params[2 * n_mzi : 2 * n_mzi + n_modes],
    )


def _clements_order(n_modes: int) -> list[tuple[int, int]]:
    return [
        (lower_row - 1, lower_row)
        for column_index in range(n_modes - 1)
        for lower_row in range(n_modes - 1, column_index, -1)
    ]


def _apply_wavelength_scale(
    params: NDArray[np.float64],
    n_modes: int,
    scale: float,
) -> NDArray[np.float64]:
    n_mzi = n_modes * (n_modes - 1) // 2
    out = np.asarray(params, dtype=np.float64).copy()
    out[n_mzi : 2 * n_mzi] *= scale
    out[2 * n_mzi : 2 * n_mzi + n_modes] *= scale
    return out


def _quantize_phases(
    raw_params: NDArray[np.float64],
    dac_bits: int,
) -> NDArray[np.float64]:
    if dac_bits <= 0:
        raise ValueError("dac_bits must be positive when set.")
    step = _TWO_PI / (1 << dac_bits)
    return np.mod(step * np.round(raw_params / step), _TWO_PI)
