"""Torch autograd Jacobian backend for ``optimize_characterization``.

The characterization optimizer runs scipy SLSQP with both objective and
constraint Jacobians supplied by torch autograd. It evaluates the same
native Clements mesh as the detection optimizer and returns the chromatic
unitary stack it optimized so downstream characterization rates use the
same deployed mesh model.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from nullsim.physics.clements import decompose_clements
from nullsim.stages.chip_optimization import _canonicalize_clements_params
from nullsim.stages.chip_optimization.torch import (
    _clements_unitary_ri,
    _clements_unitary_ri_batched,
    _tracker_crlb_rms_nm_from_singular_values,
    _tracker_observability_crlb_loss_torch,
    _tracker_observability_loss_torch,
    _tracker_observability_singular_values_torch,
    _lazy_import_torch,
    _torch_compile_enabled,
)

_TWO_PI = 2.0 * np.pi


def optimize_torch(
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
    bright_weight: float,
    snr_proxy_weight: float,
    initial_params: Sequence[NDArray[np.float64] | Sequence[float]],
    n_params: int,
    build_unitary_from_params: Any,
    throughput_fn: Any,
    null_depth_fn: Any,
    total_power_fn: Any,
    result_cls: Any,
    dac_bits: int | None,
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
    tracker_piston_basis: NDArray[np.float64] | None = None,
    tracker_field: NDArray[np.complex128] | None = None,
    tracker_wavelength_m: float | None = None,
    tracker_observability_weight: float = 0.0,
    tracker_observability_min_singular_per_nm: float = 1.0e-3,
    tracker_observability_crlb_weight: float = 0.0,
    tracker_observability_crlb_target_nm: float = 100.0,
    tracker_observability_crlb_constraint_nm: float | None = None,
    tracker_observability_eps: float = 1.0e-12,
    enforce_null_constraint: bool = True,
    n_workers: int = 1,
    compile_objective: bool = False,
) -> Any:
    """Drive scipy SLSQP with torch-autograd objective/constraint Jacobians.

    The optimizer runs in the fast native torch Clements ordering. The result
    includes an ``optimal_u_stack`` evaluated from those same native params;
    characterization stages use that stack for broadband rates instead of
    rebuilding chromatic behavior from canonical design-wavelength params.

    ``build_unitary_from_params``, ``throughput_fn``, ``null_depth_fn``,
    ``total_power_fn``, ``result_cls`` are injected so this module does not
    need to reach back into ``characterization.py`` and create a circular
    import.
    """
    if n_workers < 1:
        raise ValueError("n_workers must be >= 1.")

    torch = _lazy_import_torch()

    star_fields = np.atleast_2d(np.asarray(stellar_field, dtype=np.complex128))
    planet_fields = np.atleast_2d(np.asarray(planet_field, dtype=np.complex128))
    total_stellar = total_power_fn(star_fields, "stellar_field")
    total_planet = total_power_fn(planet_fields, "planet_field")
    broadband_mode = (
        wl_design_m is not None
        and wavelengths_m is not None
        and stellar_fields_per_wl is not None
        and planet_fields_per_wl is not None
    )

    star_r_t = torch.from_numpy(np.ascontiguousarray(star_fields.real)).to(torch.float64)
    star_i_t = torch.from_numpy(np.ascontiguousarray(star_fields.imag)).to(torch.float64)
    planet_r_t = torch.from_numpy(np.ascontiguousarray(planet_fields.real)).to(torch.float64)
    planet_i_t = torch.from_numpy(np.ascontiguousarray(planet_fields.imag)).to(torch.float64)
    spectro_t = torch.from_numpy(np.ascontiguousarray(spectro_idx)).to(torch.long)
    bright_t = torch.from_numpy(np.ascontiguousarray(bright_idx)).to(torch.long)
    total_stellar_t = torch.tensor(total_stellar, dtype=torch.float64)
    total_planet_t = torch.tensor(total_planet, dtype=torch.float64)
    if broadband_mode:
        wavelengths_np = np.asarray(wavelengths_m, dtype=np.float64)
        if np.any(~np.isfinite(wavelengths_np)) or np.any(wavelengths_np <= 0.0):
            raise ValueError("wavelengths_m must contain positive finite values.")
        wl_scales_np = float(wl_design_m) / wavelengths_np
        star_fields_bb = [
            np.atleast_2d(np.asarray(field, dtype=np.complex128))
            for field in stellar_fields_per_wl or ()
        ]
        planet_fields_bb = [
            np.atleast_2d(np.asarray(field, dtype=np.complex128))
            for field in planet_fields_per_wl or ()
        ]
        if len(star_fields_bb) != wavelengths_np.size:
            raise ValueError("stellar_fields_per_wl length must match wavelengths_m.")
        if len(planet_fields_bb) != wavelengths_np.size:
            raise ValueError("planet_fields_per_wl length must match wavelengths_m.")
        star_r_wl_t = [
            torch.from_numpy(np.ascontiguousarray(field.real)).to(torch.float64)
            for field in star_fields_bb
        ]
        star_i_wl_t = [
            torch.from_numpy(np.ascontiguousarray(field.imag)).to(torch.float64)
            for field in star_fields_bb
        ]
        planet_r_wl_t = [
            torch.from_numpy(np.ascontiguousarray(field.real)).to(torch.float64)
            for field in planet_fields_bb
        ]
        planet_i_wl_t = [
            torch.from_numpy(np.ascontiguousarray(field.imag)).to(torch.float64)
            for field in planet_fields_bb
        ]
        total_stellar_wl_t = torch.tensor(
            [total_power_fn(field, "stellar_fields_per_wl") for field in star_fields_bb],
            dtype=torch.float64,
        )
        total_planet_wl_t = torch.tensor(
            [total_power_fn(field, "planet_fields_per_wl") for field in planet_fields_bb],
            dtype=torch.float64,
        )
        stellar_weights_t = torch.from_numpy(
            _normalized_weights_torch_input(
                stellar_weights_per_wl,
                wavelengths_np.size,
                name="stellar_weights_per_wl",
            )
        ).to(torch.float64)
        planet_weights_t = torch.from_numpy(
            _normalized_weights_torch_input(
                planet_weights_per_wl,
                wavelengths_np.size,
                name="planet_weights_per_wl",
            )
        ).to(torch.float64)
        wl_scales_t = torch.from_numpy(wl_scales_np).to(torch.float64)
    else:
        star_r_wl_t = []
        star_i_wl_t = []
        planet_r_wl_t = []
        planet_i_wl_t = []
        total_stellar_wl_t = torch.empty((0,), dtype=torch.float64)
        total_planet_wl_t = torch.empty((0,), dtype=torch.float64)
        stellar_weights_t = torch.empty((0,), dtype=torch.float64)
        planet_weights_t = torch.empty((0,), dtype=torch.float64)
        wl_scales_t = torch.empty((0,), dtype=torch.float64)

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
        selection_wavelengths_np = np.asarray(
            selection_wavelengths_m,
            dtype=np.float64,
        )
        if (
            selection_wavelengths_np.ndim != 1
            or selection_wavelengths_np.size < 1
            or np.any(~np.isfinite(selection_wavelengths_np))
            or np.any(selection_wavelengths_np <= 0.0)
        ):
            raise ValueError("selection_wavelengths_m must be a positive 1-D array.")
        selection_star_fields = [
            np.atleast_2d(np.asarray(field, dtype=np.complex128))
            for field in selection_stellar_fields_per_wl or ()
        ]
        selection_planet_fields = [
            np.atleast_2d(np.asarray(field, dtype=np.complex128))
            for field in selection_planet_fields_per_wl or ()
        ]
        if len(selection_star_fields) != selection_wavelengths_np.size:
            raise ValueError(
                "selection_stellar_fields_per_wl length must match "
                "selection_wavelengths_m."
            )
        if len(selection_planet_fields) != selection_wavelengths_np.size:
            raise ValueError(
                "selection_planet_fields_per_wl length must match "
                "selection_wavelengths_m."
            )
        selection_star_r_wl_t = [
            torch.from_numpy(np.ascontiguousarray(field.real)).to(torch.float64)
            for field in selection_star_fields
        ]
        selection_star_i_wl_t = [
            torch.from_numpy(np.ascontiguousarray(field.imag)).to(torch.float64)
            for field in selection_star_fields
        ]
        selection_planet_r_wl_t = [
            torch.from_numpy(np.ascontiguousarray(field.real)).to(torch.float64)
            for field in selection_planet_fields
        ]
        selection_planet_i_wl_t = [
            torch.from_numpy(np.ascontiguousarray(field.imag)).to(torch.float64)
            for field in selection_planet_fields
        ]
        selection_total_stellar_wl_t = torch.tensor(
            [
                total_power_fn(field, "selection_stellar_fields_per_wl")
                for field in selection_star_fields
            ],
            dtype=torch.float64,
        )
        selection_total_planet_wl_t = torch.tensor(
            [
                total_power_fn(field, "selection_planet_fields_per_wl")
                for field in selection_planet_fields
            ],
            dtype=torch.float64,
        )
        selection_stellar_weights_t = torch.from_numpy(
            _normalized_weights_torch_input(
                selection_stellar_weights_per_wl,
                selection_wavelengths_np.size,
                name="selection_stellar_weights_per_wl",
            )
        ).to(torch.float64)
        selection_planet_weights_t = torch.from_numpy(
            _normalized_weights_torch_input(
                selection_planet_weights_per_wl,
                selection_wavelengths_np.size,
                name="selection_planet_weights_per_wl",
            )
        ).to(torch.float64)
        selection_wl_scales_t = torch.from_numpy(
            (float(wl_design_m) / selection_wavelengths_np).astype(np.float64),
        ).to(torch.float64)
    else:
        selection_star_r_wl_t = []
        selection_star_i_wl_t = []
        selection_planet_r_wl_t = []
        selection_planet_i_wl_t = []
        selection_total_stellar_wl_t = torch.empty((0,), dtype=torch.float64)
        selection_total_planet_wl_t = torch.empty((0,), dtype=torch.float64)
        selection_stellar_weights_t = torch.empty((0,), dtype=torch.float64)
        selection_planet_weights_t = torch.empty((0,), dtype=torch.float64)
        selection_wl_scales_t = torch.empty((0,), dtype=torch.float64)
    tracker_active = (
        tracker_observability_weight > 0.0
        or tracker_observability_crlb_weight > 0.0
        or tracker_observability_crlb_constraint_nm is not None
    )
    if tracker_active:
        if tracker_piston_basis is None or tracker_wavelength_m is None:
            raise ValueError(
                "tracker_piston_basis and tracker_wavelength_m are required "
                "when characterization tracker observability is enabled."
            )
        basis_np = np.asarray(tracker_piston_basis, dtype=np.float64)
        if basis_np.ndim != 2 or basis_np.shape[0] != n_modes:
            raise ValueError(
                "tracker_piston_basis must have shape "
                f"({n_modes}, n_differential_modes)."
            )
        field_np = (
            np.asarray(tracker_field, dtype=np.complex128)
            if tracker_field is not None
            else np.asarray(star_fields[0], dtype=np.complex128)
        )
        if field_np.ndim == 1:
            field_np = field_np[:, np.newaxis]
        if field_np.ndim != 2 or field_np.shape[0] != n_modes:
            raise ValueError(
                "tracker_field must have shape (n_modes,) or "
                f"({n_modes}, n_states)."
            )
        tracker_field_r_t = torch.from_numpy(
            np.ascontiguousarray(field_np.real[np.newaxis, :, :], dtype=np.float64)
        )
        tracker_field_i_t = torch.from_numpy(
            np.ascontiguousarray(field_np.imag[np.newaxis, :, :], dtype=np.float64)
        )
        tracker_basis_t = torch.from_numpy(np.ascontiguousarray(basis_np)).to(
            torch.float64
        )
        tracker_wavelengths_m_t = torch.tensor(
            [float(tracker_wavelength_m)],
            dtype=torch.float64,
        )
    else:
        tracker_field_r_t = None
        tracker_field_i_t = None
        tracker_basis_t = None
        tracker_wavelengths_m_t = None

    def _t_throughput(U_r, U_i, F_r, F_i, idx, total):
        # output = F @ U.T (rows of F dot rows of U); complex split:
        #   out_r = F_r @ U_r.T - F_i @ U_i.T
        #   out_i = F_r @ U_i.T + F_i @ U_r.T
        out_r = F_r @ U_r.T - F_i @ U_i.T
        out_i = F_r @ U_i.T + F_i @ U_r.T
        sel_r = out_r[:, idx]
        sel_i = out_i[:, idx]
        return (sel_r * sel_r + sel_i * sel_i).sum() / total

    def _objective_t(params_t):
        if broadband_mode:
            u_r_wl, u_i_wl = _native_clements_unitary_stack_ri(
                params_t,
                n_modes,
                wl_scales_t,
                torch_mod=torch,
            )
            t_planet = _t_weighted_throughput_stack(
                u_r_wl,
                u_i_wl,
                planet_r_wl_t,
                planet_i_wl_t,
                spectro_t,
                total_planet_wl_t,
                planet_weights_t,
            )
            f_bright = _t_weighted_throughput_stack(
                u_r_wl,
                u_i_wl,
                star_r_wl_t,
                star_i_wl_t,
                bright_t,
                total_stellar_wl_t,
                stellar_weights_t,
            )
            stellar_leak = _t_weighted_throughput_stack(
                u_r_wl,
                u_i_wl,
                star_r_wl_t,
                star_i_wl_t,
                spectro_t,
                total_stellar_wl_t,
                stellar_weights_t,
            )
        else:
            U_r, U_i = _clements_unitary_ri(
                params_t,
                n_modes,
                torch_mod=torch,
            )
            t_planet = _t_throughput(
                U_r, U_i, planet_r_t, planet_i_t, spectro_t, total_planet_t,
            )
            f_bright = _t_throughput(
                U_r, U_i, star_r_t, star_i_t, bright_t, total_stellar_t,
            )
            stellar_leak = _t_throughput(
                U_r, U_i, star_r_t, star_i_t, spectro_t, total_stellar_t,
            )
        tracker_loss = torch.zeros((), dtype=torch.float64)
        tracker_crlb_loss = torch.zeros((), dtype=torch.float64)
        if tracker_active:
            u_r_design, u_i_design = _clements_unitary_ri(
                params_t,
                n_modes,
                torch_mod=torch,
            )
            u_r_tracker = u_r_design.unsqueeze(0)
            u_i_tracker = u_i_design.unsqueeze(0)
            tracker_loss = _tracker_observability_loss_torch(
                u_r=u_r_tracker,
                u_i=u_i_tracker,
                field_r_by_wl=tracker_field_r_t,
                field_i_by_wl=tracker_field_i_t,
                wavelengths_m=tracker_wavelengths_m_t,
                bright_indices=bright_t,
                piston_basis=tracker_basis_t,
                min_singular_per_nm=float(
                    tracker_observability_min_singular_per_nm
                ),
                eps=float(tracker_observability_eps),
                torch_mod=torch,
            )
            if tracker_observability_crlb_weight > 0.0:
                tracker_crlb_loss = _tracker_observability_crlb_loss_torch(
                    u_r=u_r_tracker,
                    u_i=u_i_tracker,
                    field_r_by_wl=tracker_field_r_t,
                    field_i_by_wl=tracker_field_i_t,
                    wavelengths_m=tracker_wavelengths_m_t,
                    bright_indices=bright_t,
                    piston_basis=tracker_basis_t,
                    target_rms_nm=float(tracker_observability_crlb_target_nm),
                    eps=float(tracker_observability_eps),
                    torch_mod=torch,
                )
        return (
            -(throughput_weight * t_planet + bright_weight * f_bright)
            - float(snr_proxy_weight)
            * _t_snr_proxy(t_planet, stellar_leak, null_floor, torch_mod=torch)
            + float(tracker_observability_weight) * tracker_loss
            + float(tracker_observability_crlb_weight) * tracker_crlb_loss
        )

    def _constraint_t(params_t):
        if broadband_mode:
            u_r_wl, u_i_wl = _native_clements_unitary_stack_ri(
                params_t,
                n_modes,
                wl_scales_t,
                torch_mod=torch,
            )
            null = _t_weighted_throughput_stack(
                u_r_wl,
                u_i_wl,
                star_r_wl_t,
                star_i_wl_t,
                spectro_t,
                total_stellar_wl_t,
                stellar_weights_t,
            )
        else:
            U_r, U_i = _clements_unitary_ri(
                params_t,
                n_modes,
                torch_mod=torch,
            )
            null = _t_throughput(
                U_r, U_i, star_r_t, star_i_t, spectro_t, total_stellar_t,
            )
        return null_floor - null

    def _objective_design_t(params_t):
        U_r, U_i = _clements_unitary_ri(
            params_t,
            n_modes,
            torch_mod=torch,
        )
        t_planet = _t_throughput(
            U_r, U_i, planet_r_t, planet_i_t, spectro_t, total_planet_t,
        )
        f_bright = _t_throughput(
            U_r, U_i, star_r_t, star_i_t, bright_t, total_stellar_t,
        )
        stellar_leak = _t_throughput(
            U_r, U_i, star_r_t, star_i_t, spectro_t, total_stellar_t,
        )
        tracker_loss = torch.zeros((), dtype=torch.float64)
        tracker_crlb_loss = torch.zeros((), dtype=torch.float64)
        if tracker_active:
            u_r_tracker = U_r.unsqueeze(0)
            u_i_tracker = U_i.unsqueeze(0)
            tracker_loss = _tracker_observability_loss_torch(
                u_r=u_r_tracker,
                u_i=u_i_tracker,
                field_r_by_wl=tracker_field_r_t,
                field_i_by_wl=tracker_field_i_t,
                wavelengths_m=tracker_wavelengths_m_t,
                bright_indices=bright_t,
                piston_basis=tracker_basis_t,
                min_singular_per_nm=float(
                    tracker_observability_min_singular_per_nm
                ),
                eps=float(tracker_observability_eps),
                torch_mod=torch,
            )
            if tracker_observability_crlb_weight > 0.0:
                tracker_crlb_loss = _tracker_observability_crlb_loss_torch(
                    u_r=u_r_tracker,
                    u_i=u_i_tracker,
                    field_r_by_wl=tracker_field_r_t,
                    field_i_by_wl=tracker_field_i_t,
                    wavelengths_m=tracker_wavelengths_m_t,
                    bright_indices=bright_t,
                    piston_basis=tracker_basis_t,
                    target_rms_nm=float(tracker_observability_crlb_target_nm),
                    eps=float(tracker_observability_eps),
                    torch_mod=torch,
                )
        return (
            -(throughput_weight * t_planet + bright_weight * f_bright)
            - float(snr_proxy_weight)
            * _t_snr_proxy(t_planet, stellar_leak, null_floor, torch_mod=torch)
            + float(tracker_observability_weight) * tracker_loss
            + float(tracker_observability_crlb_weight) * tracker_crlb_loss
        )

    def _constraint_design_t(params_t):
        U_r, U_i = _clements_unitary_ri(
            params_t,
            n_modes,
            torch_mod=torch,
        )
        null = _t_throughput(
            U_r, U_i, star_r_t, star_i_t, spectro_t, total_stellar_t,
        )
        return null_floor - null

    def _tracker_constraint_t(params_t):
        if tracker_observability_crlb_constraint_nm is None:
            return torch.tensor(1.0, dtype=torch.float64)
        if not tracker_active:
            raise ValueError(
                "tracker_observability_crlb_constraint_nm requires tracker "
                "observability inputs."
            )
        u_r, u_i = _clements_unitary_ri(
            params_t,
            n_modes,
            torch_mod=torch,
        )
        crlb_loss = _tracker_observability_crlb_loss_torch(
            u_r=u_r.unsqueeze(0),
            u_i=u_i.unsqueeze(0),
            field_r_by_wl=tracker_field_r_t,
            field_i_by_wl=tracker_field_i_t,
            wavelengths_m=tracker_wavelengths_m_t,
            bright_indices=bright_t,
            piston_basis=tracker_basis_t,
            target_rms_nm=float(tracker_observability_crlb_constraint_nm),
            eps=float(tracker_observability_eps),
            torch_mod=torch,
        )
        return torch.tensor(
            1.0,
            device=crlb_loss.device,
            dtype=crlb_loss.dtype,
        ) - crlb_loss

    # torch.compile fuses the per-column Clements loop and the per-wavelength
    # power sum into one graph (chip_optimization.torch does the same: ~15x
    # faster per grad-eval at M=24 on Mac CPU, one-time ~30-55s compile). Gated
    # by compile_objective or NULLSIM_TORCH_COMPILE; default off so small runs
    # and existing tests are unchanged. Compiling also moves the per-step work
    # into a fused graph that releases the GIL, so restart parallelism can scale
    # once enabled. The two warm-start objectives are compiled separately,
    # mirroring chip_opt's bootstrap/broadband split.
    _compile_enabled = compile_objective or _torch_compile_enabled()
    if _compile_enabled:
        _obj_fn = torch.compile(_objective_t, mode="reduce-overhead", dynamic=False)
        _design_obj_fn = torch.compile(
            _objective_design_t, mode="reduce-overhead", dynamic=False
        )
    else:
        _obj_fn = _objective_t
        _design_obj_fn = _objective_design_t

    def _obj_np(x: NDArray[np.float64]) -> float:
        with torch.no_grad():
            p_t = torch.from_numpy(np.asarray(x, dtype=np.float64))
            return float(_obj_fn(p_t).item())

    def _obj_jac_np(x: NDArray[np.float64]) -> NDArray[np.float64]:
        p_t = torch.from_numpy(np.asarray(x, dtype=np.float64)).requires_grad_(True)
        loss = _obj_fn(p_t)
        loss.backward()
        return p_t.grad.detach().numpy().astype(np.float64)

    def _cons_np(x: NDArray[np.float64]) -> float:
        with torch.no_grad():
            p_t = torch.from_numpy(np.asarray(x, dtype=np.float64))
            return float(_constraint_t(p_t).item())

    def _cons_jac_np(x: NDArray[np.float64]) -> NDArray[np.float64]:
        p_t = torch.from_numpy(np.asarray(x, dtype=np.float64)).requires_grad_(True)
        val = _constraint_t(p_t)
        val.backward()
        return p_t.grad.detach().numpy().astype(np.float64)

    def _design_obj_np(x: NDArray[np.float64]) -> float:
        with torch.no_grad():
            p_t = torch.from_numpy(np.asarray(x, dtype=np.float64))
            return float(_design_obj_fn(p_t).item())

    def _design_obj_jac_np(x: NDArray[np.float64]) -> NDArray[np.float64]:
        p_t = torch.from_numpy(np.asarray(x, dtype=np.float64)).requires_grad_(True)
        loss = _design_obj_fn(p_t)
        loss.backward()
        return p_t.grad.detach().numpy().astype(np.float64)

    def _design_cons_np(x: NDArray[np.float64]) -> float:
        with torch.no_grad():
            p_t = torch.from_numpy(np.asarray(x, dtype=np.float64))
            return float(_constraint_design_t(p_t).item())

    def _design_cons_jac_np(x: NDArray[np.float64]) -> NDArray[np.float64]:
        p_t = torch.from_numpy(np.asarray(x, dtype=np.float64)).requires_grad_(True)
        val = _constraint_design_t(p_t)
        val.backward()
        return p_t.grad.detach().numpy().astype(np.float64)

    def _tracker_cons_np(x: NDArray[np.float64]) -> float:
        with torch.no_grad():
            p_t = torch.from_numpy(np.asarray(x, dtype=np.float64))
            return float(_tracker_constraint_t(p_t).item())

    def _tracker_cons_jac_np(x: NDArray[np.float64]) -> NDArray[np.float64]:
        p_t = torch.from_numpy(np.asarray(x, dtype=np.float64)).requires_grad_(True)
        val = _tracker_constraint_t(p_t)
        val.backward()
        return p_t.grad.detach().numpy().astype(np.float64)

    def _tracker_metrics_np(
        p_np: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64] | None, float | None, float, float]:
        if not tracker_active:
            return None, None, 0.0, 0.0
        with torch.no_grad():
            p_t = torch.from_numpy(np.asarray(p_np, dtype=np.float64)).to(torch.float64)
            u_r, u_i = _clements_unitary_ri(
                p_t,
                n_modes,
                torch_mod=torch,
            )
            u_r_tracker = u_r.unsqueeze(0)
            u_i_tracker = u_i.unsqueeze(0)
            singular_t = _tracker_observability_singular_values_torch(
                u_r=u_r_tracker,
                u_i=u_i_tracker,
                field_r_by_wl=tracker_field_r_t,
                field_i_by_wl=tracker_field_i_t,
                wavelengths_m=tracker_wavelengths_m_t,
                bright_indices=bright_t,
                piston_basis=tracker_basis_t,
                eps=float(tracker_observability_eps),
                torch_mod=torch,
            )
            loss_t = _tracker_observability_loss_torch(
                u_r=u_r_tracker,
                u_i=u_i_tracker,
                field_r_by_wl=tracker_field_r_t,
                field_i_by_wl=tracker_field_i_t,
                wavelengths_m=tracker_wavelengths_m_t,
                bright_indices=bright_t,
                piston_basis=tracker_basis_t,
                min_singular_per_nm=float(
                    tracker_observability_min_singular_per_nm
                ),
                eps=float(tracker_observability_eps),
                torch_mod=torch,
            )
            crlb_loss_t = torch.zeros((), dtype=torch.float64)
            if tracker_observability_crlb_weight > 0.0:
                crlb_loss_t = _tracker_observability_crlb_loss_torch(
                    u_r=u_r_tracker,
                    u_i=u_i_tracker,
                    field_r_by_wl=tracker_field_r_t,
                    field_i_by_wl=tracker_field_i_t,
                    wavelengths_m=tracker_wavelengths_m_t,
                    bright_indices=bright_t,
                    piston_basis=tracker_basis_t,
                    target_rms_nm=float(tracker_observability_crlb_target_nm),
                    eps=float(tracker_observability_eps),
                    torch_mod=torch,
                )
            singular = singular_t.detach().numpy().astype(np.float64, copy=False)
        return (
            singular,
            _tracker_crlb_rms_nm_from_singular_values(
                singular,
                eps=float(tracker_observability_eps),
            ),
            float(loss_t.item()),
            float(crlb_loss_t.item()),
        )

    def _science_metrics_np(p_np: NDArray[np.float64]) -> tuple[float, float]:
        with torch.no_grad():
            p_t = torch.from_numpy(np.asarray(p_np, dtype=np.float64)).to(torch.float64)
            if selection_mode:
                u_r_wl, u_i_wl = _native_clements_unitary_stack_ri(
                    p_t,
                    n_modes,
                    selection_wl_scales_t,
                    torch_mod=torch,
                )
                t_planet = _t_weighted_throughput_stack(
                    u_r_wl,
                    u_i_wl,
                    selection_planet_r_wl_t,
                    selection_planet_i_wl_t,
                    spectro_t,
                    selection_total_planet_wl_t,
                    selection_planet_weights_t,
                )
                stellar_leak = _t_weighted_throughput_stack(
                    u_r_wl,
                    u_i_wl,
                    selection_star_r_wl_t,
                    selection_star_i_wl_t,
                    spectro_t,
                    selection_total_stellar_wl_t,
                    selection_stellar_weights_t,
                )
            elif broadband_mode:
                u_r_wl, u_i_wl = _native_clements_unitary_stack_ri(
                    p_t,
                    n_modes,
                    wl_scales_t,
                    torch_mod=torch,
                )
                t_planet = _t_weighted_throughput_stack(
                    u_r_wl,
                    u_i_wl,
                    planet_r_wl_t,
                    planet_i_wl_t,
                    spectro_t,
                    total_planet_wl_t,
                    planet_weights_t,
                )
                stellar_leak = _t_weighted_throughput_stack(
                    u_r_wl,
                    u_i_wl,
                    star_r_wl_t,
                    star_i_wl_t,
                    spectro_t,
                    total_stellar_wl_t,
                    stellar_weights_t,
                )
            else:
                U_r, U_i = _clements_unitary_ri(
                    p_t,
                    n_modes,
                    torch_mod=torch,
                )
                t_planet = _t_throughput(
                    U_r, U_i, planet_r_t, planet_i_t, spectro_t, total_planet_t,
                )
                stellar_leak = _t_throughput(
                    U_r, U_i, star_r_t, star_i_t, spectro_t, total_stellar_t,
                )
        return float(t_planet.item()), float(stellar_leak.item())

    rng = np.random.default_rng(seed)
    provided_starts: list[NDArray[np.float64]] = [
        np.asarray(p, dtype=np.float64).copy() for p in initial_params
    ]
    for p in provided_starts:
        if p.shape != (n_params,):
            raise ValueError(
                f"initial_params entries must have shape ({n_params},), got {p.shape}."
            )
    analytic_start = _spectro_row_space_canonical_start(
        n_modes=n_modes,
        spectro_idx=spectro_idx,
        bright_idx=bright_idx,
        stellar_fields=star_fields_bb if broadband_mode else [star_fields],
        planet_fields=planet_fields_bb if broadband_mode else [planet_fields],
        stellar_weights=(
            stellar_weights_t.detach().numpy() if broadband_mode else None
        ),
        planet_weights=(
            planet_weights_t.detach().numpy() if broadband_mode else None
        ),
        initial_params=provided_starts[0] if provided_starts else None,
        build_unitary_from_params=build_unitary_from_params,
    )
    starting_points: list[NDArray[np.float64]] = []
    if analytic_start is not None:
        starting_points.append(
            _fit_native_params_to_canonical_start(
                analytic_start,
                n_modes,
                torch_mod=torch,
                build_unitary_from_params=build_unitary_from_params,
                row_weights=_char_start_fit_row_weights(
                    n_modes=n_modes,
                    bright_idx=bright_idx,
                    spectro_idx=spectro_idx,
                ),
            )
        )
    else:
        starting_points.extend(
            _fit_native_params_to_canonical_start(
                p,
                n_modes,
                torch_mod=torch,
                build_unitary_from_params=build_unitary_from_params,
            )
            for p in provided_starts
        )
    for _ in range(n_restarts):
        starting_points.append(rng.uniform(0.0, _TWO_PI, n_params))
    if not starting_points:
        starting_points.append(rng.uniform(0.0, _TWO_PI, n_params))

    n_completed = 0
    feasibility_tol = 0.01 * null_floor
    tracker_feasibility_tol = 1.0e-6

    def _constraint_margins(
        p_np: NDArray[np.float64],
    ) -> tuple[float, float]:
        null_margin = float(_cons_np(p_np))
        tracker_margin = (
            float(_tracker_cons_np(p_np))
            if tracker_observability_crlb_constraint_nm is not None
            else float("inf")
        )
        return null_margin, tracker_margin

    def _is_feasible(null_margin: float, tracker_margin: float) -> bool:
        null_ok = (not enforce_null_constraint) or null_margin >= -feasibility_tol
        tracker_ok = (
            tracker_observability_crlb_constraint_nm is None
            or tracker_margin >= -tracker_feasibility_tol
        )
        return bool(null_ok and tracker_ok)

    def _candidate_science_score(p_np: NDArray[np.float64]) -> float:
        throughput, leak = _science_metrics_np(p_np)
        return _systematic_snr_proxy_numpy(throughput, leak, null_floor)

    def _selection_key(
        *,
        objective_value: float,
        null_margin: float,
        tracker_margin: float,
        science_score: float,
    ) -> tuple[int, float, float, float, float]:
        score = float(science_score) if np.isfinite(science_score) else 0.0
        if _is_feasible(null_margin, tracker_margin):
            return (0, -score, float(objective_value), 0.0, 0.0)
        null_deficit = (
            max(0.0, -float(null_margin))
            if enforce_null_constraint
            else 0.0
        )
        tracker_deficit = (
            max(0.0, -float(tracker_margin))
            if tracker_observability_crlb_constraint_nm is not None
            else 0.0
        )
        # If the spectro null floor is not reachable, deploy the best
        # systematic-leak-aware science candidate among tracker-valid solutions.
        # The normal objective's sqrt(leak) proxy can over-reward very bright but
        # leaky rows when bright-star systematic leakage dominates the final SNR.
        # Tracker failure remains higher priority because piston sensing is a hard
        # requirement for the KILO quadrature mesh.
        return (
            1,
            float(tracker_deficit),
            -score,
            float(null_deficit),
            float(objective_value),
        )

    best_params = starting_points[0].copy()
    best_objective = _obj_np(best_params)
    best_margin, best_tracker_margin = _constraint_margins(best_params)
    best_science_score = _candidate_science_score(best_params)
    best_is_feasible = _is_feasible(best_margin, best_tracker_margin)
    best_key = _selection_key(
        objective_value=best_objective,
        null_margin=best_margin,
        tracker_margin=best_tracker_margin,
        science_score=best_science_score,
    )
    n_feasible = 0
    constraints_list = []
    if enforce_null_constraint:
        constraints_list.append(
            {"type": "ineq", "fun": _cons_np, "jac": _cons_jac_np}
        )
    if tracker_observability_crlb_constraint_nm is not None:
        constraints_list.append(
            {
                "type": "ineq",
                "fun": _tracker_cons_np,
                "jac": _tracker_cons_jac_np,
            }
        )
    design_constraints_list = []
    if enforce_null_constraint:
        design_constraints_list.append(
            {
                "type": "ineq",
                "fun": _design_cons_np,
                "jac": _design_cons_jac_np,
            }
        )
    if tracker_observability_crlb_constraint_nm is not None:
        design_constraints_list.append(
            {
                "type": "ineq",
                "fun": _tracker_cons_np,
                "jac": _tracker_cons_jac_np,
            }
        )
    continuation_max_iter = min(max_iter, max(40, max_iter // 2))

    def _run_start(
        x0: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.float64],
        float,
        float,
        float,
        float,
        bool,
        tuple[int, float, float, float, float],
    ] | None:
        candidate = np.asarray(x0, dtype=np.float64).copy()
        if broadband_mode:
            design_result = minimize(
                _design_obj_np,
                candidate,
                jac=_design_obj_jac_np,
                method="SLSQP",
                constraints=design_constraints_list,
                options={"maxiter": continuation_max_iter, "ftol": 1.0e-9},
            )
            design_x = np.asarray(design_result.x, dtype=np.float64)
            if np.all(np.isfinite(design_x)):
                candidate = design_x
        result = minimize(
            _obj_np,
            candidate,
            jac=_obj_jac_np,
            method="SLSQP",
            constraints=constraints_list,
            options={"maxiter": max_iter, "ftol": 1.0e-9},
        )
        result_x = np.asarray(result.x, dtype=np.float64)
        result_fun = float(_obj_np(result_x))
        result_margin, result_tracker_margin = _constraint_margins(result_x)
        result_science_score = _candidate_science_score(result_x)
        if (
            not np.isfinite(result_fun)
            or not np.isfinite(result_margin)
            or not np.isfinite(result_science_score)
            or (
                tracker_observability_crlb_constraint_nm is not None
                and not np.isfinite(result_tracker_margin)
            )
        ):
            return None
        result_is_feasible = _is_feasible(result_margin, result_tracker_margin)
        result_key = _selection_key(
            objective_value=result_fun,
            null_margin=result_margin,
            tracker_margin=result_tracker_margin,
            science_score=result_science_score,
        )
        return (
            result_x,
            result_fun,
            result_margin,
            result_tracker_margin,
            result_science_score,
            result_is_feasible,
            result_key,
        )

    # torch.compile'd objectives crash when called concurrently from restart
    # threads (torch._dynamo MetricsContext is not thread safe), and restart
    # parallelism is GIL bound for this optimizer anyway, so run restarts
    # serially whenever the objective is compiled.
    restart_workers = min(int(n_workers), len(starting_points))
    if restart_workers > 1 and not _compile_enabled:
        with ThreadPoolExecutor(max_workers=restart_workers) as executor:
            restart_results = list(executor.map(_run_start, starting_points))
    else:
        restart_results = [_run_start(x0) for x0 in starting_points]

    n_completed = len(starting_points)
    for restart_result in restart_results:
        if restart_result is None:
            continue
        (
            result_x,
            result_fun,
            result_margin,
            result_tracker_margin,
            result_science_score,
            result_is_feasible,
            result_key,
        ) = restart_result
        if result_is_feasible:
            n_feasible += 1
        if result_key < best_key:
            best_key = result_key
            best_objective = result_fun
            best_margin = result_margin
            best_tracker_margin = result_tracker_margin
            best_science_score = result_science_score
            best_params = result_x
            best_is_feasible = result_is_feasible

    if dac_bits is not None:
        best_params = _quantize_native_params(best_params, dac_bits)
        best_margin, best_tracker_margin = _constraint_margins(best_params)
        best_science_score = _candidate_science_score(best_params)
        best_is_feasible = _is_feasible(best_margin, best_tracker_margin)

    native_unitary = _native_unitary_numpy(best_params, n_modes, torch_mod=torch)
    canonical_params = _canonicalize_clements_params(
        best_params,
        n_modes,
        build_unitary_complex=lambda p, n: _native_unitary_numpy(
            p,
            n,
            torch_mod=torch,
        ),
    )

    spectro_null = null_depth_fn(native_unitary, star_fields, spectro_idx)
    dark_null = null_depth_fn(native_unitary, star_fields, dark_idx)
    spectro_throughput = throughput_fn(native_unitary, planet_fields, spectro_idx)
    bright_fraction = throughput_fn(native_unitary, star_fields, bright_idx)
    final_objective = -(
        throughput_weight * spectro_throughput + bright_weight * bright_fraction
    )
    final_objective -= float(snr_proxy_weight) * _snr_proxy_numpy(
        spectro_throughput,
        spectro_null,
        null_floor,
    )
    (
        tracker_singular_values,
        tracker_crlb_rms_nm,
        tracker_loss,
        tracker_crlb_loss,
    ) = _tracker_metrics_np(best_params)
    if tracker_active:
        final_objective += (
            float(tracker_observability_weight) * tracker_loss
            + float(tracker_observability_crlb_weight) * tracker_crlb_loss
        )
    stack_wavelengths = _deployment_wavelengths(
        deployment_wavelengths_m=deployment_wavelengths_m,
        wavelengths_m=wavelengths_m,
        wl_design_m=wl_design_m if wl_design_m is not None else tracker_wavelength_m,
    )
    optimal_u_stack = _native_u_stack_numpy(
        best_params,
        n_modes,
        stack_wavelengths,
        wl_design_m=wl_design_m,
        torch_mod=torch,
    )

    return result_cls(
        params=canonical_params,
        spectro_null_depth=float(spectro_null),
        dark_null_depth=float(dark_null),
        spectro_throughput=float(spectro_throughput),
        bright_stellar_fraction=float(bright_fraction),
        objective_value=float(final_objective),
        n_restarts_completed=n_completed,
        optimization_success=bool(
            (not enforce_null_constraint or (n_feasible > 0 and best_is_feasible))
        ),
        constraint_margin=float(best_margin),
        tracker_singular_values_per_nm=tracker_singular_values,
        tracker_crlb_rms_nm=tracker_crlb_rms_nm,
        tracker_loss=tracker_loss,
        tracker_crlb_loss=tracker_crlb_loss,
        optimal_u_stack=optimal_u_stack,
        optimal_u_stack_wavelengths_m=stack_wavelengths,
        optimizer_objective_value=float(best_objective),
    )


def _native_clements_unitary_stack_ri(
    params_t: Any,
    n_modes: int,
    wl_scales_t: Any,
    *,
    torch_mod: Any,
) -> tuple[Any, Any]:
    """Build native torch Clements unitaries after chromatic phase scaling."""

    n_mzi = n_modes * (n_modes - 1) // 2
    batch = params_t.unsqueeze(0).repeat(int(wl_scales_t.numel()), 1).clone()
    batch[:, n_mzi : 2 * n_mzi] *= wl_scales_t.unsqueeze(1)
    batch[:, 2 * n_mzi : 2 * n_mzi + n_modes] *= wl_scales_t.unsqueeze(1)
    return _clements_unitary_ri_batched(batch, n_modes, torch_mod=torch_mod)


def _native_unitary_numpy(
    params: NDArray[np.float64],
    n_modes: int,
    *,
    torch_mod: Any,
) -> NDArray[np.complex128]:
    with torch_mod.no_grad():
        p_t = torch_mod.from_numpy(np.asarray(params, dtype=np.float64)).to(
            torch_mod.float64
        )
        u_r, u_i = _clements_unitary_ri(p_t, n_modes, torch_mod=torch_mod)
    return (u_r.numpy() + 1j * u_i.numpy()).astype(np.complex128, copy=False)


def _native_u_stack_numpy(
    params: NDArray[np.float64],
    n_modes: int,
    wavelengths_m: NDArray[np.float64],
    *,
    wl_design_m: float | None,
    torch_mod: Any,
) -> NDArray[np.complex128]:
    design_m = float(wl_design_m) if wl_design_m is not None else float(wavelengths_m[0])
    scales = design_m / np.asarray(wavelengths_m, dtype=np.float64)
    with torch_mod.no_grad():
        p_t = torch_mod.from_numpy(np.asarray(params, dtype=np.float64)).to(
            torch_mod.float64
        )
        s_t = torch_mod.from_numpy(scales.astype(np.float64)).to(torch_mod.float64)
        u_r, u_i = _native_clements_unitary_stack_ri(
            p_t,
            n_modes,
            s_t,
            torch_mod=torch_mod,
        )
    return (u_r.numpy() + 1j * u_i.numpy()).transpose(1, 2, 0).astype(
        np.complex128,
        copy=False,
    )


def _quantize_native_params(
    raw_params: NDArray[np.float64],
    dac_bits: int,
) -> NDArray[np.float64]:
    """Round native torch phases to the DAC grid without wrapping."""

    if dac_bits <= 0:
        raise ValueError("dac_bits must be positive when set.")
    step = _TWO_PI / (1 << dac_bits)
    return step * np.round(np.asarray(raw_params, dtype=np.float64) / step)


def _fit_native_params_to_canonical_start(
    canonical_params: NDArray[np.float64],
    n_modes: int,
    *,
    torch_mod: Any,
    build_unitary_from_params: Any,
    row_weights: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Fit native torch params to a canonical Clements warm-start unitary."""

    target = build_unitary_from_params(canonical_params, n_modes)
    target_r = torch_mod.from_numpy(target.real.astype(np.float64))
    target_i = torch_mod.from_numpy(target.imag.astype(np.float64))
    if row_weights is None:
        weights_np = np.ones(n_modes, dtype=np.float64)
    else:
        weights_np = np.asarray(row_weights, dtype=np.float64)
        if weights_np.shape != (n_modes,):
            raise ValueError(
                f"row_weights must have shape ({n_modes},), got {weights_np.shape}."
            )
        weights_np = np.where(np.isfinite(weights_np), weights_np, 0.0)
        weights_np = np.maximum(weights_np, 0.0)
        if float(np.sum(weights_np)) <= 0.0:
            weights_np = np.ones(n_modes, dtype=np.float64)
    weights_t = torch_mod.from_numpy(weights_np[:, np.newaxis])
    norm_t = torch_mod.tensor(float(np.sum(weights_np) * n_modes), dtype=torch_mod.float64)

    def _fit_objective(params_t: Any) -> Any:
        u_r, u_i = _clements_unitary_ri(params_t, n_modes, torch_mod=torch_mod)
        row_error = (u_r - target_r) ** 2 + (u_i - target_i) ** 2
        return torch_mod.sum(weights_t * row_error) / norm_t

    def _fit_obj_np(x: NDArray[np.float64]) -> float:
        with torch_mod.no_grad():
            p_t = torch_mod.from_numpy(np.asarray(x, dtype=np.float64))
            return float(_fit_objective(p_t).item())

    def _fit_jac_np(x: NDArray[np.float64]) -> NDArray[np.float64]:
        p_t = torch_mod.from_numpy(np.asarray(x, dtype=np.float64)).requires_grad_(True)
        loss = _fit_objective(p_t)
        loss.backward()
        return p_t.grad.detach().numpy().astype(np.float64)

    result = minimize(
        _fit_obj_np,
        np.asarray(canonical_params, dtype=np.float64),
        jac=_fit_jac_np,
        method="BFGS",
        options={"maxiter": 500, "gtol": 1.0e-10},
    )
    if not np.isfinite(float(result.fun)):
        return np.asarray(canonical_params, dtype=np.float64).copy()
    fit = np.asarray(result.x, dtype=np.float64)
    if fit.shape != np.asarray(canonical_params, dtype=np.float64).shape:
        return np.asarray(canonical_params, dtype=np.float64).copy()
    return fit


def _char_start_fit_row_weights(
    *,
    n_modes: int,
    bright_idx: NDArray[np.intp],
    spectro_idx: NDArray[np.intp],
) -> NDArray[np.float64]:
    """Prioritize rows whose physical meaning survives the analytic start."""

    weights = np.zeros(n_modes, dtype=np.float64)
    bright = np.asarray(bright_idx, dtype=np.intp)
    spectro = np.asarray(spectro_idx, dtype=np.intp)
    if bright.size:
        weights[bright[(bright >= 0) & (bright < n_modes)]] = 1.0
    if spectro.size:
        weights[spectro[(spectro >= 0) & (spectro < n_modes)]] = 2.0
    if float(np.sum(weights)) <= 0.0:
        weights[:] = 1.0
    return weights


def _spectro_row_space_canonical_start(
    *,
    n_modes: int,
    spectro_idx: NDArray[np.intp],
    bright_idx: NDArray[np.intp],
    stellar_fields: Sequence[NDArray[np.complex128]],
    planet_fields: Sequence[NDArray[np.complex128]],
    stellar_weights: Sequence[float] | None,
    planet_weights: Sequence[float] | None,
    initial_params: NDArray[np.float64] | None,
    build_unitary_from_params: Any,
) -> NDArray[np.float64] | None:
    """Construct a canonical start by solving the ideal spectro-row problem.

    This is a warm start only. It keeps inherited bright rows fixed when an
    upstream detection mesh is available, chooses the spectro row inside the
    remaining row space to null the stellar samples and maximize weighted planet
    coupling, then decomposes the completed unitary into Clements parameters.
    The fixed-phase broadband optimizer still decides the deployed solution.
    """

    spectro = np.asarray(spectro_idx, dtype=np.intp)
    if spectro.shape != (1,):
        return None
    bright = np.asarray(bright_idx, dtype=np.intp)
    if np.any(bright < 0) or np.any(bright >= n_modes):
        return None
    if int(spectro[0]) < 0 or int(spectro[0]) >= n_modes:
        return None

    fixed_rows: list[tuple[int, NDArray[np.complex128]]] = []
    if initial_params is not None and bright.size:
        try:
            inherited = np.asarray(
                build_unitary_from_params(initial_params, n_modes),
                dtype=np.complex128,
            )
        except (TypeError, ValueError):
            return None
        if inherited.shape != (n_modes, n_modes):
            return None
        fixed_rows.extend(
            (int(index), inherited[int(index)].astype(np.complex128, copy=True))
            for index in bright
        )

    stellar_rows = _flatten_field_rows(stellar_fields, n_modes)
    planet_rows = _flatten_field_rows(planet_fields, n_modes)
    if not stellar_rows or not planet_rows:
        return None

    constraints: list[NDArray[np.complex128]] = [
        row.conjugate() for _, row in fixed_rows
    ]
    constraints.extend(stellar_rows)
    constraint_matrix = np.vstack(constraints).astype(np.complex128, copy=False)
    nullspace = _right_nullspace(constraint_matrix)
    if nullspace.shape[1] == 0:
        return None

    p_weights = _normalized_row_weights(planet_weights, len(planet_rows))
    gram = np.zeros((nullspace.shape[1], nullspace.shape[1]), dtype=np.complex128)
    for weight, planet in zip(p_weights, planet_rows, strict=True):
        projection = planet @ nullspace
        gram += float(weight) * np.outer(projection.conjugate(), projection)
    if np.all(np.isfinite(gram)):
        eigvals, eigvecs = np.linalg.eigh(gram)
        coeff = eigvecs[:, int(np.argmax(eigvals))]
    else:
        coeff = np.zeros(nullspace.shape[1], dtype=np.complex128)
        coeff[0] = 1.0
    spectro_row = (nullspace @ coeff).astype(np.complex128, copy=False)
    spectro_row = _orthonormalized_row(
        spectro_row,
        [row for _, row in fixed_rows],
    )
    if spectro_row is None:
        return None
    fixed_rows.append((int(spectro[0]), spectro_row))

    target = _complete_unitary_from_fixed_rows(n_modes, fixed_rows)
    if target is None:
        return None
    try:
        decomp = decompose_clements(target, atol=1.0e-8)
    except ValueError:
        return None
    return np.concatenate([decomp.theta, decomp.phi, decomp.alpha]).astype(
        np.float64,
        copy=False,
    )


def _flatten_field_rows(
    fields: Sequence[NDArray[np.complex128]],
    n_modes: int,
) -> list[NDArray[np.complex128]]:
    rows: list[NDArray[np.complex128]] = []
    for field in fields:
        arr = np.atleast_2d(np.asarray(field, dtype=np.complex128))
        if arr.shape[1] != n_modes:
            return []
        rows.extend(arr[index].copy() for index in range(arr.shape[0]))
    return rows


def _normalized_row_weights(
    weights: Sequence[float] | None,
    n_rows: int,
) -> NDArray[np.float64]:
    if weights is None:
        return np.full(n_rows, 1.0 / float(n_rows), dtype=np.float64)
    arr = np.asarray(weights, dtype=np.float64)
    if arr.shape == (n_rows,):
        row_weights = arr
    else:
        row_weights = np.repeat(arr, int(np.ceil(n_rows / max(arr.size, 1))))[:n_rows]
    if row_weights.shape != (n_rows,) or np.any(~np.isfinite(row_weights)):
        return np.full(n_rows, 1.0 / float(n_rows), dtype=np.float64)
    row_weights = np.maximum(row_weights, 0.0)
    total = float(np.sum(row_weights))
    if total <= 0.0:
        return np.full(n_rows, 1.0 / float(n_rows), dtype=np.float64)
    return (row_weights / total).astype(np.float64, copy=False)


def _right_nullspace(matrix: NDArray[np.complex128]) -> NDArray[np.complex128]:
    if matrix.size == 0:
        return np.eye(matrix.shape[1], dtype=np.complex128)
    _, singular, vh = np.linalg.svd(matrix, full_matrices=True)
    scale = float(singular[0]) if singular.size else 1.0
    tol = max(matrix.shape) * np.finfo(np.float64).eps * max(scale, 1.0)
    rank = int(np.sum(singular > tol))
    return vh[rank:].conjugate().T.astype(np.complex128, copy=False)


def _orthonormalized_row(
    row: NDArray[np.complex128],
    previous_rows: Sequence[NDArray[np.complex128]],
) -> NDArray[np.complex128] | None:
    candidate = np.asarray(row, dtype=np.complex128).copy()
    for previous in previous_rows:
        candidate -= (candidate @ previous.conjugate()) * previous
    norm = float(np.linalg.norm(candidate))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        return None
    return (candidate / norm).astype(np.complex128, copy=False)


def _complete_unitary_from_fixed_rows(
    n_modes: int,
    fixed_rows: Sequence[tuple[int, NDArray[np.complex128]]],
) -> NDArray[np.complex128] | None:
    used_indices = [index for index, _ in fixed_rows]
    if len(set(used_indices)) != len(used_indices):
        return None
    fixed_matrix = np.vstack([row for _, row in fixed_rows]).astype(
        np.complex128,
        copy=False,
    )
    if fixed_matrix.shape[0] > n_modes:
        return None
    if not np.allclose(
        fixed_matrix @ fixed_matrix.conjugate().T,
        np.eye(fixed_matrix.shape[0]),
        atol=1.0e-8,
        rtol=0.0,
    ):
        return None
    q, _ = np.linalg.qr(fixed_matrix.conjugate().T, mode="complete")
    complement_rows = q[:, fixed_matrix.shape[0] :].conjugate().T

    target = np.empty((n_modes, n_modes), dtype=np.complex128)
    occupied = set()
    for index, row in fixed_rows:
        target[index] = row
        occupied.add(index)
    remaining_indices = [index for index in range(n_modes) if index not in occupied]
    if len(remaining_indices) != complement_rows.shape[0]:
        return None
    for index, row in zip(remaining_indices, complement_rows, strict=True):
        target[index] = row
    if not np.allclose(
        target @ target.conjugate().T,
        np.eye(n_modes),
        atol=1.0e-8,
        rtol=0.0,
    ):
        return None
    return target


def _deployment_wavelengths(
    *,
    deployment_wavelengths_m: Sequence[float] | None,
    wavelengths_m: Sequence[float] | None,
    wl_design_m: float | None,
) -> NDArray[np.float64]:
    if deployment_wavelengths_m is not None:
        wavelengths = np.asarray(deployment_wavelengths_m, dtype=np.float64)
    elif wavelengths_m is not None:
        wavelengths = np.asarray(wavelengths_m, dtype=np.float64)
    elif wl_design_m is not None:
        wavelengths = np.asarray([float(wl_design_m)], dtype=np.float64)
    else:
        raise ValueError("wl_design_m or deployment_wavelengths_m is required.")
    if wavelengths.ndim != 1 or wavelengths.size < 1:
        raise ValueError("deployment wavelengths must be a non-empty vector.")
    if np.any(~np.isfinite(wavelengths)) or np.any(wavelengths <= 0.0):
        raise ValueError("deployment wavelengths must be positive finite values.")
    return wavelengths.astype(np.float64, copy=True)


def _canonical_clements_order(n_modes: int) -> list[tuple[int, int]]:
    return [
        (lower_row - 1, lower_row)
        for column_index in range(n_modes - 1)
        for lower_row in range(n_modes - 1, column_index, -1)
    ]


def _t_weighted_throughput_stack(
    u_r_wl,
    u_i_wl,
    fields_r_wl,
    fields_i_wl,
    idx,
    total_wl,
    weights_wl,
):
    total = u_r_wl.new_zeros(())
    for wl_index, (fields_r, fields_i) in enumerate(zip(fields_r_wl, fields_i_wl)):
        total = total + weights_wl[wl_index] * _t_throughput(
            u_r_wl[wl_index],
            u_i_wl[wl_index],
            fields_r,
            fields_i,
            idx,
            total_wl[wl_index],
        )
    return total


def _t_snr_proxy(planet_throughput, stellar_leak, null_floor: float, *, torch_mod):
    # Saturate the reward at the configured floor: static leak below
    # null_floor is worthless at deployment (the realized piston floor
    # dominates), and an unbounded sqrt reward trades planet throughput
    # for un-cashable null depth. Mirrors _snr_proxy in characterization.py.
    leak = torch_mod.clamp(stellar_leak, min=float(null_floor))
    scale = torch_mod.sqrt(
        torch_mod.tensor(float(null_floor), dtype=leak.dtype, device=leak.device) / leak
    )
    return planet_throughput * scale


def _snr_proxy_numpy(
    planet_throughput: float,
    stellar_leak: float,
    null_floor: float,
) -> float:
    leak = max(float(stellar_leak), float(null_floor))
    return float(planet_throughput) * float(np.sqrt(float(null_floor) / leak))


def _systematic_snr_proxy_numpy(
    planet_throughput: float,
    stellar_leak: float,
    null_floor: float,
) -> float:
    """Rank final candidates by throughput once they meet the leak floor."""

    leak = max(float(stellar_leak), float(null_floor))
    if not np.isfinite(leak) or leak <= 0.0:
        return 0.0
    throughput = float(planet_throughput)
    if not np.isfinite(throughput) or throughput <= 0.0:
        return 0.0
    return throughput / leak


def _t_throughput(U_r, U_i, F_r, F_i, idx, total):
    # output = F @ U.T (rows of F dot rows of U); complex split:
    #   out_r = F_r @ U_r.T - F_i @ U_i.T
    #   out_i = F_r @ U_i.T + F_i @ U_r.T
    out_r = F_r @ U_r.T - F_i @ U_i.T
    out_i = F_r @ U_i.T + F_i @ U_r.T
    sel_r = out_r[:, idx]
    sel_i = out_i[:, idx]
    return (sel_r * sel_r + sel_i * sel_i).sum() / total


def _normalized_weights_torch_input(
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
    return (1, float(-constraint_margin), float(objective_value))
