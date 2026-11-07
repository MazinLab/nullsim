"""Torch backend for chip null optimization.

SPEC §15.3 Pass 3 WP3. Reference implementation: KILO ``chip_torch.py``
``_clements_unitary_ri`` (lines 63-134) and
``make_torch_null_objective_and_grad`` (lines 137-191); WP1 notes at
``tests/stages/notes/chip_optimization.md`` (§2).

Optional extra: install with ``pip install nullsim[torch]``. The torch
import is lazy (inside the stage methods) so importing this module on a
system without torch only fails when the stage is actually instantiated.

KILO hard-won lesson reproduced here: TF32 mode on Ampere / Blackwell GPUs
truncates fp32 mantissas to 10 bits, perturbing BFGS gradients enough to
flip the optimizer into a different basin (~1.4e-4 broadband null on RTX
5090 fp32+TF32 vs ~6e-5 on CPU at the same seed). We therefore (a) use
float64 everywhere and (b) disable TF32 explicitly at first import as a
belt-and-braces guard against any future float32 path.

Loss: see ``nullsim.stages.chip_optimization`` for the broadband
dark-port leakage formula and spectral reduction modes. The objective
is differentiated via torch autograd against a real/imag split unitary,
with scipy ``BFGS`` driving the line search.
"""

from __future__ import annotations

import math
import os
import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from nullsim.pipeline import BaseStage, SimulationState, register_stage
from nullsim.stages._utils import (
    mode_origin_positions_m,
    project_aperture_positions_for_pointing,
)
from nullsim.stages.chip_optimization import (
    ChipOptimizationParams,
    _canonicalize_clements_params,
    _chip_chromaticity_metadata,
    _dark_indices,
    _delay_state_current_transport_scale_from_state,
    _delay_state_fields_from_state,
    _delay_state_metadata,
    _delay_state_transport_scales_from_state,
    _edge_weighted_spectral_weights_numpy,
    _loss_meets_target,
    _mode_diameters_m,
    _n_params,
    _normalization_field_array,
    _normalized_loss_for_target,
    _null_indices,
    _optimizer_null_field_stack,
    _optimizer_wavelength_metadata,
    _optimizer_wavelength_subset,
    _phase_scale_factors_numpy,
    _quantize_phases,
    _reported_null_depth_denominator,
    _reported_null_depth_for_target,
    _require_field,
    _require_wavelength_grid_um,
    _spectral_loss_metadata,
    _stellar_radius_rad,
    _theta_offsets_numpy,
    _throughput_penalty_fields,
    _tracker_piston_basis_from_state,
    _uses_disk_samples,
    _write_results,
)


def _torch_compile_enabled() -> bool:
    """Return True iff NULLSIM_TORCH_COMPILE env var is truthy.

    Mirrors KILO's KILO_TORCH_COMPILE gate (chip_torch.py L352-358).
    Off by default — torch.compile pays a one-time tracing cost (~30s
    for the M=24 Clements graph) but speeds up subsequent grad-evals
    by ~15x on Mac CPU per KILO's measurement. Worth the cost for
    runs that hit >2000 BFGS iterations.
    """
    return os.environ.get("NULLSIM_TORCH_COMPILE", "0").lower() in {
        "1", "true", "yes", "on",
    }

_TORCH_INIT_DONE = False
_TORCH_MODULE: Any = None


def _lazy_import_torch() -> Any:
    """Import torch lazily so the package installs without the optional extra.

    Raises a clear ``ImportError`` pointing at the ``nullsim[torch]``
    extra when torch is not installed. Disables TF32 on first import
    (see module docstring).
    """

    global _TORCH_INIT_DONE, _TORCH_MODULE
    if _TORCH_MODULE is not None:
        return _TORCH_MODULE
    try:
        import torch  # noqa: PLC0415 — lazy by design
    except ImportError as exc:  # pragma: no cover - exercised in extras test
        raise ImportError(
            "ChipOptimizationTorch requires the optional 'torch' extra. "
            "Install with: pip install 'nullsim[torch]'"
        ) from exc

    if not _TORCH_INIT_DONE:
        # KILO chip_torch.py:43-45 — disable TF32 on CUDA matmul + cuDNN.
        # BFGS basin flips on Ampere/Blackwell GPUs with TF32 enabled.
        torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
        _TORCH_INIT_DONE = True

    _TORCH_MODULE = torch
    return torch


def _select_torch_device(torch_mod: Any, requested: str) -> Any:
    """Resolve the torch device for the float64 chip optimizer."""

    token = requested.lower()
    if token == "auto":
        token = "cuda" if torch_mod.cuda.is_available() else "cpu"
    if token == "cuda" and not torch_mod.cuda.is_available():
        raise ValueError("chip_optimization.torch requested CUDA, but CUDA is unavailable.")
    if token not in {"cpu", "cuda"}:
        raise ValueError("torch_device must be one of 'auto', 'cpu', or 'cuda'.")
    return torch_mod.device(token)


def _clements_unitary_ri(
    params_t: Any,
    n_modes: int,
    *,
    torch_mod: Any,
) -> tuple[Any, Any]:
    """Build the Clements M x M unitary as two real (M, M) tensors.

    Direct port of KILO ``chip_torch._clements_unitary_ri`` (lines 63-134).
    Vectorized per Clements column: each column applies its active MZI
    pairs as a single broadcast op rather than sequential row updates,
    cutting CUDA kernel launches from O(M^2) to O(M).

    Parameter layout matches the WP1 notes (§1):
        params[:n_mzi]              -> MZI internal phases theta
        params[n_mzi:2*n_mzi]        -> MZI external phases phi
        params[2*n_mzi:2*n_mzi + m]  -> diagonal output phases alpha
    """

    n_mzi = n_modes * (n_modes - 1) // 2
    thetas = params_t[:n_mzi]
    phis = params_t[n_mzi : 2 * n_mzi]
    diag = params_t[2 * n_mzi : 2 * n_mzi + n_modes]

    device = params_t.device
    dtype = params_t.dtype
    U_r = torch_mod.eye(n_modes, device=device, dtype=dtype)
    U_i = torch_mod.zeros((n_modes, n_modes), device=device, dtype=dtype)

    idx = 0
    for col in range(n_modes):
        start_row = col % 2
        n_active = (n_modes - start_row) // 2
        if n_active == 0:
            continue

        t = thetas[idx : idx + n_active]
        p = phis[idx : idx + n_active]
        idx += n_active

        ct = torch_mod.cos(t).unsqueeze(1)
        st = torch_mod.sin(t).unsqueeze(1)
        cp = torch_mod.cos(p).unsqueeze(1)
        sp = torch_mod.sin(p).unsqueeze(1)

        t00_r = cp * st
        t00_i = sp * st
        t01_r = ct
        t10_r = cp * ct
        t10_i = sp * ct
        t11_r = -st

        end_row = start_row + 2 * n_active
        rows_r = U_r[start_row:end_row].view(n_active, 2, n_modes)
        rows_i = U_i[start_row:end_row].view(n_active, 2, n_modes)
        r0 = rows_r[:, 0, :]
        i0 = rows_i[:, 0, :]
        r1 = rows_r[:, 1, :]
        i1 = rows_i[:, 1, :]

        nr0 = t00_r * r0 - t00_i * i0 + t01_r * r1
        ni0 = t00_r * i0 + t00_i * r0 + t01_r * i1
        nr1 = t10_r * r0 - t10_i * i0 + t11_r * r1
        ni1 = t10_r * i0 + t10_i * r0 + t11_r * i1

        new_r = torch_mod.stack([nr0, nr1], dim=1).reshape(2 * n_active, n_modes)
        new_i = torch_mod.stack([ni0, ni1], dim=1).reshape(2 * n_active, n_modes)

        U_r = torch_mod.cat([U_r[:start_row], new_r, U_r[end_row:]], dim=0)
        U_i = torch_mod.cat([U_i[:start_row], new_i, U_i[end_row:]], dim=0)

    cp_d = torch_mod.cos(diag).unsqueeze(1)
    sp_d = torch_mod.sin(diag).unsqueeze(1)
    new_U_r = cp_d * U_r - sp_d * U_i
    new_U_i = cp_d * U_i + sp_d * U_r
    return new_U_r, new_U_i


def _clements_unitary_ri_batched(
    params_t: Any,
    n_modes: int,
    *,
    torch_mod: Any,
) -> tuple[Any, Any]:
    """Build a batch of Clements unitaries as real/imag tensors.

    ``params_t`` has shape ``(batch, 2*n_mzi + n_modes)``. The Clements
    column loop is still explicit, but every active MZI in every batch element
    is updated by one broadcasted tensor operation.
    """

    if params_t.dim() != 2:
        raise ValueError(
            "batched Clements params must have shape "
            f"(batch, 2*n_mzi + n_modes); got {tuple(params_t.shape)}."
        )
    batch_size = int(params_t.shape[0])
    n_mzi = n_modes * (n_modes - 1) // 2
    expected = 2 * n_mzi + n_modes
    if int(params_t.shape[1]) != expected:
        raise ValueError(
            f"batched Clements params length must be {expected}; "
            f"got {int(params_t.shape[1])}."
        )

    thetas = params_t[:, :n_mzi]
    phis = params_t[:, n_mzi : 2 * n_mzi]
    diag = params_t[:, 2 * n_mzi : 2 * n_mzi + n_modes]

    device = params_t.device
    dtype = params_t.dtype
    eye = torch_mod.eye(n_modes, device=device, dtype=dtype)
    u_r = eye.unsqueeze(0).expand(batch_size, n_modes, n_modes).contiguous()
    u_i = torch_mod.zeros(
        (batch_size, n_modes, n_modes),
        device=device,
        dtype=dtype,
    )

    idx = 0
    for col in range(n_modes):
        start_row = col % 2
        n_active = (n_modes - start_row) // 2
        if n_active == 0:
            continue

        theta = thetas[:, idx : idx + n_active]
        phi = phis[:, idx : idx + n_active]
        idx += n_active

        ct = torch_mod.cos(theta).unsqueeze(2)
        st = torch_mod.sin(theta).unsqueeze(2)
        cp = torch_mod.cos(phi).unsqueeze(2)
        sp = torch_mod.sin(phi).unsqueeze(2)

        t00_r = cp * st
        t00_i = sp * st
        t01_r = ct
        t10_r = cp * ct
        t10_i = sp * ct
        t11_r = -st

        end_row = start_row + 2 * n_active
        rows_r = u_r[:, start_row:end_row, :].view(
            batch_size,
            n_active,
            2,
            n_modes,
        )
        rows_i = u_i[:, start_row:end_row, :].view(
            batch_size,
            n_active,
            2,
            n_modes,
        )
        r0 = rows_r[:, :, 0, :]
        i0 = rows_i[:, :, 0, :]
        r1 = rows_r[:, :, 1, :]
        i1 = rows_i[:, :, 1, :]

        nr0 = t00_r * r0 - t00_i * i0 + t01_r * r1
        ni0 = t00_r * i0 + t00_i * r0 + t01_r * i1
        nr1 = t10_r * r0 - t10_i * i0 + t11_r * r1
        ni1 = t10_r * i0 + t10_i * r0 + t11_r * i1

        new_r = torch_mod.stack([nr0, nr1], dim=2).reshape(
            batch_size,
            2 * n_active,
            n_modes,
        )
        new_i = torch_mod.stack([ni0, ni1], dim=2).reshape(
            batch_size,
            2 * n_active,
            n_modes,
        )

        u_r = torch_mod.cat(
            [u_r[:, :start_row, :], new_r, u_r[:, end_row:, :]],
            dim=1,
        )
        u_i = torch_mod.cat(
            [u_i[:, :start_row, :], new_i, u_i[:, end_row:, :]],
            dim=1,
        )

    cp_d = torch_mod.cos(diag).unsqueeze(2)
    sp_d = torch_mod.sin(diag).unsqueeze(2)
    new_u_r = cp_d * u_r - sp_d * u_i
    new_u_i = cp_d * u_i + sp_d * u_r
    return new_u_r, new_u_i


@register_stage
class ChipOptimizationTorch(BaseStage):
    """Find Clements params that minimize broadband dark-port leakage (torch backend).

    Flux-scale dependence: the base broadband null objective (absolute
    dark-port leakage plus the throughput penalty) is scale-equivariant, so
    the reported null depth and the stellar-disk null floor are flux-scale
    invariant. The tracker / joint / amplitude observability terms carry
    absolute physical targets — the per-nm intensity-response singular floor
    (``tracker_observability_min_singular_per_nm``) and the CRLB targets in nm
    (``tracker_observability_crlb_target_nm``,
    ``joint_observability_crlb_target_per_scaled_mode``) — that are genuinely
    photon-rate dependent: a fringe tracker's piston estimability scales with
    the actual photon flux. When any of those terms is active the optimum, and
    hence the realized disk floor, depends on the flux the chip was optimized
    at. The payload advertises this honestly via the ``disk_null_floor_note``
    key (``ChipOptimizationParams.objective_is_flux_scale_dependent``); the
    per-flux disk-floor spread is then chip-realization dependent rather than a
    single fixed number.
    """

    type_name = "chip_optimization.torch"
    family = "chip_optimization"
    consumes = frozenset({"field", "throughput", "geometry"})
    produces = frozenset({"results"})
    cache_key_extras = frozenset({"scene"})

    Params = ChipOptimizationParams

    def required_consumes(self) -> frozenset[str]:
        required = set(self.consumes)
        if _uses_disk_samples(self.params_model):
            required.add("scene")
        if self.params_model.warm_start_source is not None:
            required.add("results")
        return frozenset(required)

    def apply(self, state: SimulationState) -> SimulationState:
        params: ChipOptimizationParams = self.params_model
        torch_mod = _lazy_import_torch()

        field_array = _require_field(
            state,
            params.n_modes,
            wl_design_um=params.wl_design_um,
        )
        wavelength_grid_um = _require_wavelength_grid_um(state, field_array)

        # The optimizer subsamples for the loss function (cheaper iters,
        # smooth chip response across the band) but the DEPLOYED u_stack
        # must cover every detector-grid bin or chip.kernel_mzi_mesh
        # raises a shape mismatch (Tier 3.1 contract). Keep the full
        # grid for the post-optimization u_stack build below.
        full_wavelength_grid_um = tuple(float(w) for w in wavelength_grid_um)

        # KILO uses 5 wavelengths for the broadband chip optimization
        # (natural_design.py L474, ``KILO_BB_N_WL`` default 5) while nullsim
        # input grids often have 20+ bins for detector accuracy. Sub-
        # sampling the optimization keeps the per-iteration cost
        # proportional to n_wl_optimize and the broadband optimum is
        # essentially unchanged because the chip's wavelength response
        # is smooth across the J-band.
        field_array, wavelength_grid_um, optimizer_wavelength_indices = (
            _optimizer_wavelength_subset(field_array, wavelength_grid_um, params)
        )
        delay_state_fields = _delay_state_fields_from_state(
            state=state,
            field_array=field_array,
            params=params,
            wavelength_grid_um=wavelength_grid_um,
        )
        delay_state_transport_scales = _delay_state_transport_scales_from_state(
            state=state,
            field_shape=tuple(int(value) for value in field_array.shape),
            params=params,
            wavelength_grid_um=wavelength_grid_um,
        )
        delay_state_current_transport_scale = (
            _delay_state_current_transport_scale_from_state(
                state=state,
                field_shape=tuple(int(value) for value in field_array.shape),
                params=params,
                wavelength_grid_um=wavelength_grid_um,
            )
        )
        target_field_array = _normalization_field_array(
            field_array,
            delay_state_fields,
        )
        # KILO splits the chip-output port indices into two sets that
        # play different roles in the objective:
        #   null_idx  = dark + spectro = ``_null_indices(params)``
        #               (constrained to zero stellar leakage)
        #   dark_idx  = dark only      = ``_dark_indices(params)``
        #               (planet-throughput maximization)
        # See natural_design.py:390 and :643.
        null_idx = _null_indices(params)
        dark_idx = _dark_indices(params)

        device = _select_torch_device(torch_mod, params.torch_device)

        # Build the stellar null field stack (n_rows, M, n_wl, n_pol) once.
        # For kernel_order=0 and disk samples disabled, this is just
        # on_axis_field[None, ...]. Kernel rows enforce Taylor derivative
        # constraints; disk rows sample the resolved host disk. The stack is
        # constant w.r.t. optimizer params, so it stays outside autograd.
        positions_pupil = (
            project_aperture_positions_for_pointing(
                np.asarray(
                    state.geometry.aperture_positions_pupil_m, dtype=np.float64
                ),
                altitude_deg=state.scene.altitude_deg,
                azimuth_deg=state.scene.azimuth_deg,
                aperture_origins_m=mode_origin_positions_m(
                    state.geometry,
                    int(
                        np.asarray(
                            state.geometry.aperture_positions_pupil_m
                        ).shape[0]
                    ),
                ),
            )
            if state.geometry.aperture_positions_pupil_m is not None
            else None
        )
        stellar_radius_rad = _stellar_radius_rad(state)
        active_disk_samples = _uses_disk_samples(params) and stellar_radius_rad > 0.0
        active_throughput_penalty = params.throughput_penalty_alpha > 0.0
        if (
            params.kernel_order > 0 or active_disk_samples or active_throughput_penalty
        ) and positions_pupil is None:
            raise ValueError(
                f"chip_optimization.torch with kernel_order={params.kernel_order} "
                "or disk/throughput-penalty samples requires "
                "state.geometry.aperture_positions_pupil_m to be populated "
                "upstream. Set kernel_order=0, disable disk samples, and set "
                "throughput_penalty_alpha=0 for on-axis-only."
            )
        if (
            params.kernel_order > 0
            or active_disk_samples
            or delay_state_fields is not None
        ):
            sub_diameters_m = (
                _mode_diameters_m(state, params.n_modes)
                if active_disk_samples
                else None
            )
            field_stack = _optimizer_null_field_stack(
                on_axis_field=field_array,
                params=params,
                aperture_positions_pupil_m=positions_pupil,
                wavelength_grid_um=wavelength_grid_um,
                sub_diameters_m=sub_diameters_m,
                stellar_radius_rad=stellar_radius_rad,
                delay_state_fields=delay_state_fields,
            )
        else:
            field_stack = field_array[np.newaxis, ...]

        # Per-wavelength field tensors and rescaling factors. We split the
        # complex field into real and imag tensors so the optimizer's
        # autograd graph stays in pure real arithmetic. Stack axis 0 is
        # n_rows (kernel-null derivative constraints).
        stack_r = torch_mod.from_numpy(
            np.ascontiguousarray(field_stack.real, dtype=np.float64),
        ).to(device)
        stack_i = torch_mod.from_numpy(
            np.ascontiguousarray(field_stack.imag, dtype=np.float64),
        ).to(device)
        stack_r_by_wl = stack_r.permute(2, 0, 1, 3).contiguous()
        stack_i_by_wl = stack_i.permute(2, 0, 1, 3).contiguous()
        phase_scales_np = _phase_scale_factors_numpy(
            wavelength_grid_um=wavelength_grid_um,
            wl_design_um=params.wl_design_um,
            chromaticity_params=params,
            rescale_per_wavelength=True,
        )
        theta_offsets_np = _theta_offsets_numpy(
            wavelength_grid_um=wavelength_grid_um,
            wl_design_um=params.wl_design_um,
            chromaticity_params=params,
            apply_chromaticity=True,
        )
        phase_scales_t = torch_mod.from_numpy(phase_scales_np).to(device)
        wavelengths_m_all_t = torch_mod.from_numpy(
            np.ascontiguousarray(
                np.asarray(wavelength_grid_um, dtype=np.float64) * 1.0e-6,
            )
        ).to(device)
        theta_offsets_t = torch_mod.from_numpy(theta_offsets_np).to(device)
        idx_null_t = torch_mod.from_numpy(
            np.ascontiguousarray(null_idx),
        ).to(device)
        idx_dark_t = torch_mod.from_numpy(
            np.ascontiguousarray(dark_idx),
        ).to(device)

        n_mzi = params.n_modes * (params.n_modes - 1) // 2
        n_wl = len(wavelength_grid_um)
        n_rows = field_stack.shape[0]
        # Delay-state replay stacks the same objective rows once per state
        # with 1/sqrt(n_states) weights; the reported-null convention
        # counts rows PER STATE (review 2026-06-10 Issue 1).
        n_delay_states = (
            int(delay_state_fields.shape[0]) if delay_state_fields is not None else 1
        )
        target_null_loss = float(params.target_null_depth) * (
            _reported_null_depth_denominator(
                target_field_array,
                n_objective_rows=n_rows,
                n_states=n_delay_states,
            )
        )
        spectral_weights_t = None
        if params.spectral_loss_mode == "edge_weighted_mean":
            weights_np = _edge_weighted_spectral_weights_numpy(
                wavelength_grid_um,
                params,
            )
            spectral_weights_t = torch_mod.from_numpy(
                np.ascontiguousarray(weights_np, dtype=np.float64),
            ).to(device)

        tracker_metadata: dict[str, Any] = {}
        tracker_active = (
            params.tracker_observability_weight > 0.0
            or params.tracker_observability_crlb_weight > 0.0
            or bool(params.tracker_observability_rank_priority)
        )
        if tracker_active:
            tracker_basis = _tracker_piston_basis_from_state(state, params.n_modes)
            tracker_wave_idx_np = _tracker_observability_wavelength_indices(
                len(wavelength_grid_um),
                int(params.tracker_observability_n_spectral_channels),
            )
            if not params.broadband_refine and tracker_wave_idx_np.size > 1:
                raise ValueError(
                    "chip_optimization.torch: tracker_observability_n_spectral_channels "
                    "selects multiple wavelengths, but broadband_refine=False only "
                    "optimizes the bootstrap design wavelength. Set "
                    "tracker_observability_n_spectral_channels=1 or enable "
                    "broadband_refine."
                )
            tracker_idx_bright_t = torch_mod.arange(
                int(params.n_bright_ports),
                device=device,
                dtype=torch_mod.long,
            )
            tracker_basis_t = torch_mod.from_numpy(
                np.ascontiguousarray(tracker_basis.chip_basis, dtype=np.float64),
            ).to(device)
            if delay_state_fields is None:
                tracker_field_np = np.moveaxis(
                    field_array[:, tracker_wave_idx_np, :],
                    1,
                    0,
                )
                tracker_eval_wave_idx_np = tracker_wave_idx_np
                tracker_field_source = "nominal_on_axis_field"
                tracker_n_field_states = 1
            else:
                tracker_field_np = delay_state_fields[:, :, tracker_wave_idx_np, :]
                tracker_field_np = np.transpose(
                    tracker_field_np,
                    (2, 0, 1, 3),
                ).reshape(
                    tracker_wave_idx_np.size * delay_state_fields.shape[0],
                    params.n_modes,
                    delay_state_fields.shape[3],
                )
                tracker_eval_wave_idx_np = np.repeat(
                    tracker_wave_idx_np,
                    delay_state_fields.shape[0],
                ).astype(np.int64, copy=False)
                tracker_field_source = "delay_state_fields"
                tracker_n_field_states = int(delay_state_fields.shape[0])
            tracker_eval_wave_idx_t = torch_mod.from_numpy(
                np.ascontiguousarray(tracker_eval_wave_idx_np, dtype=np.int64),
            ).to(device)
            tracker_wavelengths_m_t = torch_mod.from_numpy(
                np.ascontiguousarray(
                    np.asarray(wavelength_grid_um, dtype=np.float64)[
                        tracker_eval_wave_idx_np
                    ]
                    * 1.0e-6,
                    dtype=np.float64,
                )
            ).to(device)
            tracker_field_r_sel = torch_mod.from_numpy(
                np.ascontiguousarray(tracker_field_np.real, dtype=np.float64),
            ).to(device)
            tracker_field_i_sel = torch_mod.from_numpy(
                np.ascontiguousarray(tracker_field_np.imag, dtype=np.float64),
            ).to(device)
            tracker_metadata = {
                "tracker_observability_weight": float(
                    params.tracker_observability_weight
                ),
                "tracker_observability_min_singular_per_nm": float(
                    params.tracker_observability_min_singular_per_nm
                ),
                "tracker_observability_crlb_weight": float(
                    params.tracker_observability_crlb_weight
                ),
                "tracker_observability_crlb_target_nm": float(
                    params.tracker_observability_crlb_target_nm
                ),
                "tracker_observability_rank_priority": bool(
                    params.tracker_observability_rank_priority
                ),
                "tracker_observability_n_spectral_channels": int(
                    params.tracker_observability_n_spectral_channels
                ),
                "tracker_observability_wavelength_indices": tuple(
                    int(value) for value in tracker_wave_idx_np
                ),
                "tracker_observability_basis_source": tracker_basis.source,
                "tracker_observability_n_physical_modes": int(
                    tracker_basis.n_physical_modes
                ),
                "tracker_observability_field_source": tracker_field_source,
                "tracker_observability_n_field_states": tracker_n_field_states,
                "tracker_observability_model": (
                    "Count-free topology proxy: representative-wavelength, "
                    "reference-power-weighted bright-port intensity-response "
                    "singular values per nm of physical piston. The floor "
                    "hinge prevents dropped modes; the optional CRLB term "
                    "minimizes sqrt(mean(1/s_i^2)). Each delay state is a "
                    "separate pointing; the hinge, CRLB, and rank act on the "
                    "worst state. The diagnostics."
                    "fringe_observability stage computes the count-weighted "
                    "grouped-channel Fisher/CRLB model."
                ),
            }
        else:
            tracker_idx_bright_t = None
            tracker_basis_t = None
            tracker_eval_wave_idx_t = None
            tracker_wavelengths_m_t = None
            tracker_field_r_sel = None
            tracker_field_i_sel = None

        def _selected_field_tensors(
            wave_idx_np: NDArray[np.integer],
        ) -> tuple[Any, Any, Any, tuple[int, ...], str, int]:
            if delay_state_fields is None:
                selected_field_np = np.moveaxis(
                    field_array[:, wave_idx_np, :],
                    1,
                    0,
                )
                eval_wave_idx_np = np.asarray(wave_idx_np, dtype=np.int64)
                field_source = "nominal_on_axis_field"
                n_field_states = 1
            else:
                selected_field_np = delay_state_fields[:, :, wave_idx_np, :]
                selected_field_np = np.transpose(
                    selected_field_np,
                    (2, 0, 1, 3),
                ).reshape(
                    len(wave_idx_np) * delay_state_fields.shape[0],
                    params.n_modes,
                    delay_state_fields.shape[3],
                )
                eval_wave_idx_np = np.repeat(
                    np.asarray(wave_idx_np, dtype=np.int64),
                    delay_state_fields.shape[0],
                ).astype(np.int64, copy=False)
                field_source = "delay_state_fields"
                n_field_states = int(delay_state_fields.shape[0])
            eval_wave_idx_t = torch_mod.from_numpy(
                np.ascontiguousarray(eval_wave_idx_np, dtype=np.int64),
            ).to(device)
            field_r_t = torch_mod.from_numpy(
                np.ascontiguousarray(selected_field_np.real, dtype=np.float64),
            ).to(device)
            field_i_t = torch_mod.from_numpy(
                np.ascontiguousarray(selected_field_np.imag, dtype=np.float64),
            ).to(device)
            return (
                eval_wave_idx_t,
                field_r_t,
                field_i_t,
                tuple(int(value) for value in wave_idx_np),
                field_source,
                n_field_states,
            )

        amplitude_metadata: dict[str, Any] = {}
        amplitude_mismatch_active = params.amplitude_mismatch_weight > 0.0
        amplitude_observability_active = params.amplitude_observability_weight > 0.0
        joint_observability_active = (
            params.joint_observability_weight > 0.0
            or params.joint_observability_crlb_weight > 0.0
        )
        amplitude_active = (
            amplitude_mismatch_active
            or amplitude_observability_active
            or joint_observability_active
        )
        if amplitude_active:
            amplitude_basis = (
                tracker_basis
                if tracker_active
                else _tracker_piston_basis_from_state(state, params.n_modes)
            )
            amplitude_basis_t = torch_mod.from_numpy(
                np.ascontiguousarray(amplitude_basis.chip_basis, dtype=np.float64),
            ).to(device)
        else:
            amplitude_basis_t = None
            amplitude_basis = None

        if amplitude_mismatch_active:
            amplitude_mismatch_wave_idx_np = _tracker_observability_wavelength_indices(
                len(wavelength_grid_um),
                int(params.amplitude_mismatch_n_spectral_channels),
            )
            if not params.broadband_refine and amplitude_mismatch_wave_idx_np.size > 1:
                raise ValueError(
                    "chip_optimization.torch: amplitude_mismatch_n_spectral_channels "
                    "selects multiple wavelengths, but broadband_refine=False only "
                    "optimizes the bootstrap design wavelength. Set "
                    "amplitude_mismatch_n_spectral_channels=1 or enable "
                    "broadband_refine."
                )
            amplitude_dump_index = (
                int(params.n_bright_ports) - 1
                if params.amplitude_dump_port_index is None
                else int(params.amplitude_dump_port_index)
            )
            amplitude_dump_idx_t = torch_mod.tensor(
                [amplitude_dump_index],
                device=device,
                dtype=torch_mod.long,
            )
            (
                amplitude_mismatch_eval_wave_idx_t,
                amplitude_mismatch_field_r_sel,
                amplitude_mismatch_field_i_sel,
                amplitude_mismatch_wavelength_indices,
                amplitude_mismatch_field_source,
                amplitude_mismatch_n_field_states,
            ) = _selected_field_tensors(amplitude_mismatch_wave_idx_np)
            amplitude_metadata.update(
                {
                "amplitude_mismatch_weight": float(params.amplitude_mismatch_weight),
                "amplitude_mismatch_n_spectral_channels": int(
                    params.amplitude_mismatch_n_spectral_channels
                ),
                "amplitude_mismatch_probe_sigma_i": float(
                    params.amplitude_mismatch_probe_sigma_i
                ),
                "amplitude_mismatch_wavelength_indices": (
                    amplitude_mismatch_wavelength_indices
                ),
                "amplitude_mismatch_basis_source": amplitude_basis.source,
                "amplitude_mismatch_n_physical_modes": int(
                    amplitude_basis.n_physical_modes
                ),
                "amplitude_mismatch_field_source": amplitude_mismatch_field_source,
                "amplitude_mismatch_n_field_states": (
                    amplitude_mismatch_n_field_states
                ),
                "amplitude_dump_port_index": int(amplitude_dump_index),
                "amplitude_dump_min_fraction": float(
                    params.amplitude_dump_min_fraction
                ),
                "amplitude_mismatch_model": (
                    "Count-free topology proxy: evaluate the dark-port fraction "
                    "under symmetric physical fractional-intensity perturbations "
                    "about the nominal stellar field, normalize by probe sigma^2, "
                    "and optionally require a selected bright port to carry a "
                    "minimum fraction of the linear perturbation power."
                ),
                }
            )
        else:
            amplitude_mismatch_eval_wave_idx_t = None
            amplitude_mismatch_field_r_sel = None
            amplitude_mismatch_field_i_sel = None
            amplitude_dump_idx_t = None

        if amplitude_observability_active:
            amplitude_observability_wave_idx_np = (
                _tracker_observability_wavelength_indices(
                    len(wavelength_grid_um),
                    int(params.amplitude_observability_n_spectral_channels),
                )
            )
            if (
                not params.broadband_refine
                and amplitude_observability_wave_idx_np.size > 1
            ):
                raise ValueError(
                    "chip_optimization.torch: "
                    "amplitude_observability_n_spectral_channels selects multiple "
                    "wavelengths, but broadband_refine=False only optimizes the "
                    "bootstrap design wavelength. Set "
                    "amplitude_observability_n_spectral_channels=1 or enable "
                    "broadband_refine."
                )
            amplitude_observability_idx_np = _amplitude_observability_port_indices(
                params
            )
            amplitude_observability_idx_t = torch_mod.from_numpy(
                np.ascontiguousarray(amplitude_observability_idx_np, dtype=np.int64),
            ).to(device)
            (
                amplitude_observability_eval_wave_idx_t,
                amplitude_observability_field_r_sel,
                amplitude_observability_field_i_sel,
                amplitude_observability_wavelength_indices,
                amplitude_observability_field_source,
                amplitude_observability_n_field_states,
            ) = _selected_field_tensors(amplitude_observability_wave_idx_np)
            amplitude_metadata.update(
                {
                    "amplitude_observability_weight": float(
                        params.amplitude_observability_weight
                    ),
                    "amplitude_observability_min_singular_per_fractional_intensity": (
                        float(
                            params.amplitude_observability_min_singular_per_fractional_intensity
                        )
                    ),
                    "amplitude_observability_port_set": (
                        params.amplitude_observability_port_set
                    ),
                    "amplitude_observability_port_indices": tuple(
                        int(value) for value in amplitude_observability_idx_np
                    ),
                    "amplitude_observability_n_spectral_channels": int(
                        params.amplitude_observability_n_spectral_channels
                    ),
                    "amplitude_observability_wavelength_indices": (
                        amplitude_observability_wavelength_indices
                    ),
                    "amplitude_observability_basis_source": amplitude_basis.source,
                    "amplitude_observability_n_physical_modes": int(
                        amplitude_basis.n_physical_modes
                    ),
                    "amplitude_observability_field_source": (
                        amplitude_observability_field_source
                    ),
                    "amplitude_observability_n_field_states": (
                        amplitude_observability_n_field_states
                    ),
                    "amplitude_observability_model": (
                        "Count-free topology proxy: maximize the selected-port "
                        "reference-power-weighted response singular values to "
                        "physical differential fractional-intensity modes. This "
                        "regularizer rewards estimability instead of making the "
                        "science dark ports insensitive to amplitude perturbations. "
                        "Each delay state is a separate pointing; the hinge acts "
                        "on the worst state."
                    ),
                }
            )
        else:
            amplitude_observability_eval_wave_idx_t = None
            amplitude_observability_field_r_sel = None
            amplitude_observability_field_i_sel = None
            amplitude_observability_idx_t = None

        joint_metadata: dict[str, Any] = {}
        if joint_observability_active:
            joint_wave_idx_np = _tracker_observability_wavelength_indices(
                len(wavelength_grid_um),
                int(params.joint_observability_n_spectral_channels),
            )
            if not params.broadband_refine and joint_wave_idx_np.size > 1:
                raise ValueError(
                    "chip_optimization.torch: "
                    "joint_observability_n_spectral_channels selects multiple "
                    "wavelengths, but broadband_refine=False only optimizes the "
                    "bootstrap design wavelength. Set "
                    "joint_observability_n_spectral_channels=1 or enable "
                    "broadband_refine."
                )
            joint_idx_np = _joint_observability_port_indices(params)
            joint_idx_t = torch_mod.from_numpy(
                np.ascontiguousarray(joint_idx_np, dtype=np.int64),
            ).to(device)
            (
                joint_eval_wave_idx_t,
                joint_field_r_sel,
                joint_field_i_sel,
                joint_wavelength_indices,
                joint_field_source,
                joint_n_field_states,
            ) = _selected_field_tensors(joint_wave_idx_np)
            # Rows per delay state: each state is its own response matrix.
            joint_n_rows = int(joint_idx_np.size) * int(joint_wave_idx_np.size)
            joint_n_columns = 2 * int(amplitude_basis.chip_basis.shape[1])
            if joint_n_rows < joint_n_columns:
                raise ValueError(
                    "chip_optimization.torch joint observability is rank-deficient "
                    "by construction: selected ports × spectral channels gives "
                    f"{joint_n_rows} rows per delay state for {joint_n_columns} "
                    "phase+amplitude modes. Increase "
                    "joint_observability_n_spectral_channels, select more ports, "
                    "or disable the joint observability regularizer."
                )
            joint_wavelengths_m_t = wavelengths_m_all_t.index_select(
                0,
                joint_eval_wave_idx_t,
            )
            joint_metadata = {
                "joint_observability_weight": float(
                    params.joint_observability_weight
                ),
                "joint_observability_min_singular_per_scaled_mode": float(
                    params.joint_observability_min_singular_per_scaled_mode
                ),
                "joint_observability_crlb_weight": float(
                    params.joint_observability_crlb_weight
                ),
                "joint_observability_crlb_target_per_scaled_mode": float(
                    params.joint_observability_crlb_target_per_scaled_mode
                ),
                "joint_observability_science_null_guard_weight": float(
                    params.joint_observability_science_null_guard_weight
                ),
                "joint_observability_science_null_guard_factor": float(
                    params.joint_observability_science_null_guard_factor
                ),
                "joint_observability_port_set": params.joint_observability_port_set,
                "joint_observability_port_indices": tuple(
                    int(value) for value in joint_idx_np
                ),
                "joint_observability_n_spectral_channels": int(
                    params.joint_observability_n_spectral_channels
                ),
                "joint_observability_wavelength_indices": joint_wavelength_indices,
                "joint_observability_amplitude_scale": float(
                    params.joint_observability_amplitude_scale
                ),
                "joint_observability_basis_source": amplitude_basis.source,
                "joint_observability_n_physical_modes": int(
                    amplitude_basis.n_physical_modes
                ),
                "joint_observability_field_source": joint_field_source,
                "joint_observability_n_field_states": joint_n_field_states,
                "joint_observability_n_rows": joint_n_rows,
                "joint_observability_n_columns": joint_n_columns,
                "joint_observability_model": (
                    "Count-free Poisson Fisher topology proxy: stack "
                    "reference-power-weighted selected-port piston columns "
                    "per nm with differential fractional-intensity columns "
                    "scaled by joint_observability_amplitude_scale. The hard "
                    "hinge penalizes weak singular modes; the optional CRLB "
                    "term softly minimizes sqrt(mean(1/s_i^2)); the optional "
                    "science-null guard penalizes candidates above the target "
                    "null basin. Each delay state is a separate pointing; the "
                    "hinge and CRLB act on the worst state, and n_rows counts "
                    "rows per state."
                ),
            }
        else:
            joint_eval_wave_idx_t = None
            joint_wavelengths_m_t = None
            joint_field_r_sel = None
            joint_field_i_sel = None
            joint_idx_t = None

        def _objective_components(
            params_t: Any,
        ) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
            # KILO chip_torch.py L312-352: per-wavelength absolute leakage
            # sum at null ports. With kernel_order>0 the same per-wavelength
            # loop runs across the n_rows kernel-null stack rows enforcing
            # derivative constraints at dark ports. The final spectral reducer
            # is configurable: historical mean, smooth p-norm, or edge-weighted
            # mean.
            #
            # Tier 3.3 (2026-05-17): when throughput_penalty_alpha > 0 the
            # broadband objective also subtracts
            # ``alpha × <planet_throughput>_{sep, PA, wl}``, matching KILO
            # ``natural_design.py:KILO_BB_THROUGHPUT_PENALTY_ALPHA``
            # convention. Pre-fix the penalty lived only in the bootstrap
            # objective (single-wavelength); broadband_refine basins then
            # drifted toward narrow-band null-only solutions that lost
            # the kernel-null r^6 throughput rolloff at the band edges.
            scaled = params_t.unsqueeze(0).expand(n_wl, -1).clone()
            scaled[:, :n_mzi] = scaled[:, :n_mzi] + theta_offsets_t[:, None]
            scaled[:, n_mzi : 2 * n_mzi] = (
                scaled[:, n_mzi : 2 * n_mzi] * phase_scales_t[:, None]
            )
            scaled[:, 2 * n_mzi : 2 * n_mzi + params.n_modes] = (
                scaled[:, 2 * n_mzi : 2 * n_mzi + params.n_modes]
                * phase_scales_t[:, None]
            )
            u_r, u_i = _clements_unitary_ri_batched(
                scaled,
                params.n_modes,
                torch_mod=torch_mod,
            )
            out_r = torch_mod.einsum(
                "wij,wrjp->wrip", u_r, stack_r_by_wl
            ) - torch_mod.einsum("wij,wrjp->wrip", u_i, stack_i_by_wl)
            out_i = torch_mod.einsum(
                "wij,wrjp->wrip", u_r, stack_i_by_wl
            ) + torch_mod.einsum("wij,wrjp->wrip", u_i, stack_r_by_wl)
            leakage_power = out_r[:, :, idx_null_t, :] ** 2
            leakage_power = leakage_power + out_i[:, :, idx_null_t, :] ** 2
            leak_per_wl = torch_mod.sum(leakage_power, dim=(1, 2, 3))
            leakage_loss = _reduce_spectral_leakage_torch(
                leak_per_wl,
                params,
                spectral_weights=spectral_weights_t,
                torch_mod=torch_mod,
            )
            throughput_reward = torch_mod.zeros(
                (),
                device=device,
                dtype=params_t.dtype,
            )
            if tp_field_r_t is not None:
                # Per sample s: out_r_s = Re((U @ z_s)),
                #               out_i_s = Im((U @ z_s)).
                pf_out_r = torch_mod.einsum(
                    "wsm,wim->wsi", tp_field_r_t, u_r
                ) - torch_mod.einsum("wsm,wim->wsi", tp_field_i_t, u_i)
                pf_out_i = torch_mod.einsum(
                    "wsm,wim->wsi", tp_field_r_t, u_i
                ) + torch_mod.einsum("wsm,wim->wsi", tp_field_i_t, u_r)
                dark_power = (
                    pf_out_r[:, :, idx_dark_t] ** 2 + pf_out_i[:, :, idx_dark_t] ** 2
                )
                # KILO averages the planet-throughput term over PA/separation
                # samples only; dark-port power remains summed across ports.
                throughput_per_sample = torch_mod.sum(dark_power, dim=2)
                mean_throughput = torch_mod.mean(throughput_per_sample, dim=1)
                throughput_reward = torch_mod.sum(mean_throughput) / float(n_wl)
            tracker_loss = torch_mod.zeros(
                (),
                device=device,
                dtype=params_t.dtype,
            )
            tracker_crlb_loss = torch_mod.zeros(
                (),
                device=device,
                dtype=params_t.dtype,
            )
            if tracker_active:
                u_r_tracker = u_r.index_select(0, tracker_eval_wave_idx_t)
                u_i_tracker = u_i.index_select(0, tracker_eval_wave_idx_t)
                tracker_loss, tracker_crlb_loss = (
                    _tracker_worst_state_losses_torch(
                        n_states=tracker_n_field_states,
                        u_r=u_r_tracker,
                        u_i=u_i_tracker,
                        field_r_by_wl=tracker_field_r_sel,
                        field_i_by_wl=tracker_field_i_sel,
                        wavelengths_m=tracker_wavelengths_m_t,
                        bright_indices=tracker_idx_bright_t,
                        piston_basis=tracker_basis_t,
                        min_singular_per_nm=float(
                            params.tracker_observability_min_singular_per_nm
                        ),
                        target_rms_nm=float(
                            params.tracker_observability_crlb_target_nm
                        ),
                        compute_crlb=(
                            params.tracker_observability_crlb_weight > 0.0
                        ),
                        eps=float(params.tracker_observability_eps),
                        torch_mod=torch_mod,
                    )
                )
            amplitude_mismatch_loss = torch_mod.zeros(
                (),
                device=device,
                dtype=params_t.dtype,
            )
            if amplitude_mismatch_active:
                u_r_amp = u_r.index_select(0, amplitude_mismatch_eval_wave_idx_t)
                u_i_amp = u_i.index_select(0, amplitude_mismatch_eval_wave_idx_t)
                amplitude_mismatch_loss = _amplitude_mismatch_loss_torch(
                    u_r=u_r_amp,
                    u_i=u_i_amp,
                    field_r_by_wl=amplitude_mismatch_field_r_sel,
                    field_i_by_wl=amplitude_mismatch_field_i_sel,
                    null_indices=idx_dark_t,
                    dump_indices=amplitude_dump_idx_t,
                    amplitude_basis=amplitude_basis_t,
                    min_dump_fraction=float(params.amplitude_dump_min_fraction),
                    probe_sigma_i=float(params.amplitude_mismatch_probe_sigma_i),
                    eps=float(params.amplitude_mismatch_eps),
                    torch_mod=torch_mod,
                )
            amplitude_observability_loss = torch_mod.zeros(
                (),
                device=device,
                dtype=params_t.dtype,
            )
            if amplitude_observability_active:
                u_r_amp_obs = u_r.index_select(
                    0,
                    amplitude_observability_eval_wave_idx_t,
                )
                u_i_amp_obs = u_i.index_select(
                    0,
                    amplitude_observability_eval_wave_idx_t,
                )
                amplitude_observability_loss = _worst_delay_state_loss_torch(
                    _amplitude_observability_loss_torch,
                    n_delay_states,
                    u_r=u_r_amp_obs,
                    u_i=u_i_amp_obs,
                    field_r_by_wl=amplitude_observability_field_r_sel,
                    field_i_by_wl=amplitude_observability_field_i_sel,
                    port_indices=amplitude_observability_idx_t,
                    amplitude_basis=amplitude_basis_t,
                    min_singular_per_fractional_intensity=float(
                        params.amplitude_observability_min_singular_per_fractional_intensity
                    ),
                    eps=float(params.amplitude_observability_eps),
                    torch_mod=torch_mod,
                )
            joint_observability_loss = torch_mod.zeros(
                (),
                device=device,
                dtype=params_t.dtype,
            )
            joint_observability_crlb_loss = torch_mod.zeros(
                (),
                device=device,
                dtype=params_t.dtype,
            )
            if joint_observability_active:
                u_r_joint = u_r.index_select(0, joint_eval_wave_idx_t)
                u_i_joint = u_i.index_select(0, joint_eval_wave_idx_t)
                if params.joint_observability_weight > 0.0:
                    joint_observability_loss = (
                        _worst_delay_state_loss_torch(
                            _joint_phase_amplitude_observability_loss_torch,
                            n_delay_states,
                            u_r=u_r_joint,
                            u_i=u_i_joint,
                            field_r_by_wl=joint_field_r_sel,
                            field_i_by_wl=joint_field_i_sel,
                            wavelengths_m=joint_wavelengths_m_t,
                            port_indices=joint_idx_t,
                            piston_basis=amplitude_basis_t,
                            amplitude_basis=amplitude_basis_t,
                            amplitude_scale=float(
                                params.joint_observability_amplitude_scale
                            ),
                            min_singular_per_scaled_mode=float(
                                params.joint_observability_min_singular_per_scaled_mode
                            ),
                            eps=float(params.joint_observability_eps),
                            torch_mod=torch_mod,
                        )
                    )
                if params.joint_observability_crlb_weight > 0.0:
                    joint_observability_crlb_loss = (
                        _worst_delay_state_loss_torch(
                            _joint_phase_amplitude_observability_crlb_loss_torch,
                            n_delay_states,
                            u_r=u_r_joint,
                            u_i=u_i_joint,
                            field_r_by_wl=joint_field_r_sel,
                            field_i_by_wl=joint_field_i_sel,
                            wavelengths_m=joint_wavelengths_m_t,
                            port_indices=joint_idx_t,
                            piston_basis=amplitude_basis_t,
                            amplitude_basis=amplitude_basis_t,
                            amplitude_scale=float(
                                params.joint_observability_amplitude_scale
                            ),
                            target_per_scaled_mode=float(
                                params.joint_observability_crlb_target_per_scaled_mode
                            ),
                            eps=float(params.joint_observability_eps),
                            torch_mod=torch_mod,
                        )
                    )
            joint_science_guard_loss = torch_mod.zeros(
                (),
                device=device,
                dtype=params_t.dtype,
            )
            if params.joint_observability_science_null_guard_weight > 0.0:
                joint_science_guard_loss = _science_null_guard_loss_torch(
                    leakage_loss=leakage_loss,
                    target_loss=target_null_loss,
                    guard_factor=float(
                        params.joint_observability_science_null_guard_factor
                    ),
                    eps=float(params.joint_observability_eps),
                    torch_mod=torch_mod,
                )
            total = (
                leakage_loss
                - tp_alpha * throughput_reward
                + float(params.tracker_observability_weight) * tracker_loss
                + float(params.tracker_observability_crlb_weight)
                * tracker_crlb_loss
                + float(params.amplitude_mismatch_weight) * amplitude_mismatch_loss
                + float(params.amplitude_observability_weight)
                * amplitude_observability_loss
                + float(params.joint_observability_weight) * joint_observability_loss
                + float(params.joint_observability_crlb_weight)
                * joint_observability_crlb_loss
                + float(params.joint_observability_science_null_guard_weight)
                * joint_science_guard_loss
            )
            return (
                total,
                leakage_loss,
                throughput_reward,
                tracker_loss,
                tracker_crlb_loss,
                amplitude_mismatch_loss,
                amplitude_observability_loss,
                joint_observability_loss,
                joint_observability_crlb_loss,
                joint_science_guard_loss,
            )

        def _objective(params_t: Any) -> Any:
            (
                total,
                _leakage_loss,
                _throughput_reward,
                _tracker_loss,
                _tracker_crlb_loss,
                _amp_mismatch_loss,
                _amp_observability_loss,
                _joint_observability_loss,
                _joint_observability_crlb_loss,
                _joint_science_guard_loss,
            ) = _objective_components(params_t)
            return total

        # KILO chip_torch.py L352-358: torch.compile fuses the per-column
        # Clements loop + per-wavelength sum into one graph. KILO benched
        # 15.5x faster per-grad-eval at M=24 / n_wl=5 on Mac CPU, 4.2x
        # end-to-end through scipy BFGS. One-time ~30-55s compile cost.
        # Gated by params.compile_objective OR the NULLSIM_TORCH_COMPILE
        # env var; default off so existing tests + small runs are
        # unchanged.
        if params.compile_objective or _torch_compile_enabled():
            _obj_fn = torch_mod.compile(
                _objective, mode="reduce-overhead", dynamic=False,
            )
        else:
            _obj_fn = _objective

        def _obj_np(x: NDArray[np.float64]) -> float:
            with torch_mod.no_grad():
                p_t = torch_mod.from_numpy(np.asarray(x, dtype=np.float64)).to(device)
                return float(_obj_fn(p_t).item())

        def _objective_components_np(x: NDArray[np.float64]) -> dict[str, float]:
            with torch_mod.no_grad():
                p_t = torch_mod.from_numpy(np.asarray(x, dtype=np.float64)).to(device)
                (
                    total,
                    leakage_loss,
                    throughput_reward,
                    tracker_loss,
                    tracker_crlb_loss,
                    amp_mismatch_loss,
                    amp_observability_loss,
                    joint_observability_loss,
                    joint_observability_crlb_loss,
                    joint_science_guard_loss,
                ) = _objective_components(p_t)
                return {
                    "optimization_objective_final": float(total.item()),
                    "optimization_leakage_loss_final": float(leakage_loss.item()),
                    "optimization_throughput_reward_final": float(
                        throughput_reward.item()
                    ),
                    "optimization_tracker_loss_final": float(tracker_loss.item()),
                    "optimization_tracker_crlb_loss_final": float(
                        tracker_crlb_loss.item()
                    ),
                    "optimization_amplitude_mismatch_loss_final": float(
                        amp_mismatch_loss.item()
                    ),
                    "optimization_amplitude_observability_loss_final": float(
                        amp_observability_loss.item()
                    ),
                    "optimization_joint_observability_loss_final": float(
                        joint_observability_loss.item()
                    ),
                    "optimization_joint_observability_crlb_loss_final": float(
                        joint_observability_crlb_loss.item()
                    ),
                    "optimization_joint_science_guard_loss_final": float(
                        joint_science_guard_loss.item()
                    ),
                }

        def _obj_and_grad_np(
            x: NDArray[np.float64],
        ) -> tuple[float, NDArray[np.float64]]:
            # scipy BFGS evaluates fun and jac at the same iterate; a fused
            # (loss, grad) from ONE forward+backward saves the duplicate
            # forward — ~33% of the dominant cost at ~35k grad evals/cell
            # (review 2026-06-10 Issue 31).
            p_t = (
                torch_mod.from_numpy(np.asarray(x, dtype=np.float64))
                .to(device)
                .requires_grad_(True)
            )
            loss = _obj_fn(p_t)
            loss.backward()
            return (
                float(loss.item()),
                p_t.grad.detach().cpu().numpy().astype(np.float64),
            )

        def _tracker_singular_values_for_params_np(
            x: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            if not tracker_active:
                return np.empty(0, dtype=np.float64)
            with torch_mod.no_grad():
                p_t = torch_mod.from_numpy(np.asarray(x, dtype=np.float64)).to(device)
                scaled = p_t.unsqueeze(0).expand(n_wl, -1).clone()
                scaled[:, :n_mzi] = scaled[:, :n_mzi] + theta_offsets_t[:, None]
                scaled[:, n_mzi : 2 * n_mzi] = (
                    scaled[:, n_mzi : 2 * n_mzi] * phase_scales_t[:, None]
                )
                scaled[:, 2 * n_mzi : 2 * n_mzi + params.n_modes] = (
                    scaled[:, 2 * n_mzi : 2 * n_mzi + params.n_modes]
                    * phase_scales_t[:, None]
                )
                u_r_eval, u_i_eval = _clements_unitary_ri_batched(
                    scaled,
                    params.n_modes,
                    torch_mod=torch_mod,
                )
                singular_t = _tracker_per_state_singular_values_torch(
                    n_states=tracker_n_field_states,
                    u_r=u_r_eval.index_select(0, tracker_eval_wave_idx_t),
                    u_i=u_i_eval.index_select(0, tracker_eval_wave_idx_t),
                    field_r_by_wl=tracker_field_r_sel,
                    field_i_by_wl=tracker_field_i_sel,
                    wavelengths_m=tracker_wavelengths_m_t,
                    bright_indices=tracker_idx_bright_t,
                    piston_basis=tracker_basis_t,
                    eps=float(params.tracker_observability_eps),
                    torch_mod=torch_mod,
                )
                return singular_t.detach().cpu().numpy().astype(np.float64, copy=False)

        def _candidate_selection_key(
            x: NDArray[np.float64],
            objective_loss: float,
            leakage_loss: float,
            null_field_array: NDArray[np.complex128],
            n_objective_rows: int,
        ) -> tuple[float, ...]:
            if not params.tracker_observability_rank_priority or not tracker_active:
                return (float(objective_loss),)
            singular_values = _tracker_singular_values_for_params_np(x)
            floor = float(params.tracker_observability_min_singular_per_nm)
            n_dof = int(tracker_basis.chip_basis.shape[1])
            if not np.all(np.isfinite(singular_values)):
                return (
                    float(n_dof),
                    1.0,
                    float("inf"),
                    float("inf"),
                    float(objective_loss),
                    float("inf"),
                    float("inf"),
                )
            reported_null = _reported_null_depth_for_target(
                leakage_loss,
                null_field_array,
                n_objective_rows=n_objective_rows,
                n_states=n_delay_states,
            )
            return _tracker_rank_priority_selection_key(
                singular_values_per_nm=singular_values,
                n_dof=n_dof,
                min_singular_per_nm=floor,
                crlb_target_nm=float(params.tracker_observability_crlb_target_nm),
                objective_loss=objective_loss,
                normalized_null=reported_null,
                target_null_depth=float(params.target_null_depth),
                eps=float(params.tracker_observability_eps),
            )

        # KILO chip.py L1100-1160 restart-loop pattern (now also used by
        # the scipy backend, see test_scipy_restarts.py). Master RNG draws
        # one seed per restart so each restart is deterministic. Patience
        # early-exit bails after K consecutive no-improvement restarts.
        seed_base = 0 if params.seed is None else int(params.seed)
        master_rng = np.random.default_rng(seed_base)
        # PyTorch global RNG: numpy seed alone leaves torch.randn /
        # torch internal RNG seeded from system entropy, which makes
        # restart init non-reproducible across cells. Pinning the
        # torch RNG (cheap) gives identical inits; torch CPU MKL
        # reductions are not pinned (the multi-thread perf cost
        # would be ~10×) but their non-determinism is small enough
        # that BFGS basins are stable from identical inits.
        torch_mod.manual_seed(seed_base)
        n_params_total = _n_params(params.n_modes)
        restart_seeds = [
            int(master_rng.integers(0, 2**31 - 1)) for _ in range(params.n_restarts)
        ]

        tp_field_np = _throughput_penalty_fields(
            state=state,
            field_array=field_array,
            wavelength_grid_um=wavelength_grid_um,
            params=params,
            aperture_positions_pupil_m=positions_pupil,
            delay_state_transport_scales=delay_state_transport_scales,
            delay_state_current_transport_scale=delay_state_current_transport_scale,
        )
        if tp_field_np is not None:
            tp_field_r_t = torch_mod.from_numpy(
                tp_field_np.real.astype(np.float64),
            ).to(device)
            tp_field_i_t = torch_mod.from_numpy(
                tp_field_np.imag.astype(np.float64),
            ).to(device)
            tp_alpha = float(params.throughput_penalty_alpha)
        else:
            tp_field_r_t = None
            tp_field_i_t = None
            tp_alpha = 0.0

        warm_start_arr, warm_start_metadata = _resolve_external_warm_start(
            state=state,
            params=params,
            wavelength_grid_um=wavelength_grid_um,
            torch_mod=torch_mod,
            device=device,
        )

        # Single-lambda bootstrap (KILO natural_design.py L555-585).
        # At M=24 with kernel-null rows, BFGS-from-Haar lands in shallow
        # broadband basins because the deep on-axis valley is hard to
        # find from random init when one has to be consistent across n_wl
        # wavelengths simultaneously. The bootstrap finds the deep valley
        # at wl_design alone (no per-wavelength rescaling), then uses
        # those params as the warm-start so the broadband solve only has
        # to refine the band edges.
        if params.single_lambda_bootstrap and n_wl > 1:
            # Build a SINGLE-wavelength version of the objective: same
            # field stack, same dark indices, scale=1 (no rescaling).
            bs_inv_norm = 1.0
            bs_wl_idx = int(
                np.argmin(
                    np.abs(np.asarray(wavelength_grid_um) - float(params.wl_design_um))
                )
            )
            bs_target_field_array = field_array[:, bs_wl_idx : bs_wl_idx + 1, :]
            bs_target_null_loss = float(params.target_null_depth) * float(
                _reported_null_depth_denominator(
                    bs_target_field_array,
                    n_objective_rows=n_rows,
                    n_states=n_delay_states,
                )
            )

            if tp_field_r_t is not None:
                # Bootstrap-wavelength planet fields (n_sep*n_PA, M).
                tp_field_r_bs = tp_field_r_t[bs_wl_idx]
                tp_field_i_bs = tp_field_i_t[bs_wl_idx]
            if tracker_active:
                if delay_state_fields is None:
                    bs_tracker_field_np = np.moveaxis(
                        field_array[:, bs_wl_idx : bs_wl_idx + 1, :],
                        1,
                        0,
                    )
                else:
                    bs_tracker_field_np = delay_state_fields[:, :, bs_wl_idx, :]
                bs_tracker_n_channels = int(bs_tracker_field_np.shape[0])
                bs_tracker_field_r_t = torch_mod.from_numpy(
                    np.ascontiguousarray(
                        bs_tracker_field_np.real,
                        dtype=np.float64,
                    )
                ).to(device)
                bs_tracker_field_i_t = torch_mod.from_numpy(
                    np.ascontiguousarray(
                        bs_tracker_field_np.imag,
                        dtype=np.float64,
                    )
                ).to(device)
                bs_tracker_wavelengths_m_t = torch_mod.full(
                    (bs_tracker_n_channels,),
                    float(wavelength_grid_um[bs_wl_idx]) * 1.0e-6,
                    device=device,
                    dtype=bs_tracker_field_r_t.dtype,
                )
            if amplitude_active:
                if delay_state_fields is None:
                    bs_amplitude_field_np = np.moveaxis(
                        field_array[:, bs_wl_idx : bs_wl_idx + 1, :],
                        1,
                        0,
                    )
                else:
                    bs_amplitude_field_np = delay_state_fields[:, :, bs_wl_idx, :]
                bs_amplitude_n_channels = int(bs_amplitude_field_np.shape[0])
                bs_amplitude_field_r_t = torch_mod.from_numpy(
                    np.ascontiguousarray(
                        bs_amplitude_field_np.real,
                        dtype=np.float64,
                    )
                ).to(device)
                bs_amplitude_field_i_t = torch_mod.from_numpy(
                    np.ascontiguousarray(
                        bs_amplitude_field_np.imag,
                        dtype=np.float64,
                    )
                ).to(device)
            if joint_observability_active:
                bs_joint_wavelengths_m_t = torch_mod.full(
                    (bs_amplitude_n_channels,),
                    float(wavelength_grid_um[bs_wl_idx]) * 1.0e-6,
                    device=device,
                    dtype=bs_amplitude_field_r_t.dtype,
                )

            def _bs_objective_components(
                params_t: Any,
            ) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
                u_r, u_i = _clements_unitary_ri(
                    params_t,
                    params.n_modes,
                    torch_mod=torch_mod,
                )
                leakage_loss = torch_mod.zeros((), device=device, dtype=params_t.dtype)
                for r in range(n_rows):
                    f_r = stack_r[r, :, bs_wl_idx, :]
                    f_i = stack_i[r, :, bs_wl_idx, :]
                    out_r = u_r @ f_r - u_i @ f_i
                    out_i = u_r @ f_i + u_i @ f_r
                    leak = torch_mod.sum(
                        out_r[idx_null_t, :] ** 2 + out_i[idx_null_t, :] ** 2,
                    )
                    leakage_loss = leakage_loss + leak * bs_inv_norm
                throughput_reward = torch_mod.zeros(
                    (),
                    device=device,
                    dtype=params_t.dtype,
                )
                if tp_field_r_t is not None:
                    # KILO physical planet fields: sqrt(coupling * area)
                    # with angular-coupling rolloff. Average over
                    # n_sep * n_PA samples; subtract alpha * <dark power>.
                    out_re = tp_field_r_bs @ u_r.t() - tp_field_i_bs @ u_i.t()
                    out_im = tp_field_r_bs @ u_i.t() + tp_field_i_bs @ u_r.t()
                    dark_power = out_re[:, idx_dark_t] ** 2 + out_im[:, idx_dark_t] ** 2
                    # Keep the bootstrap penalty on the same scale as the
                    # broadband objective: average over PA/separation samples,
                    # not over dark ports.
                    throughput_per_sample = torch_mod.sum(dark_power, dim=1)
                    throughput_reward = torch_mod.mean(throughput_per_sample)
                tracker_loss = torch_mod.zeros(
                    (),
                    device=device,
                    dtype=params_t.dtype,
                )
                tracker_crlb_loss = torch_mod.zeros(
                    (),
                    device=device,
                    dtype=params_t.dtype,
                )
                if tracker_active:
                    tracker_loss, tracker_crlb_loss = (
                        _tracker_worst_state_losses_torch(
                            n_states=tracker_n_field_states,
                            u_r=u_r.unsqueeze(0).expand(
                                bs_tracker_n_channels,
                                -1,
                                -1,
                            ),
                            u_i=u_i.unsqueeze(0).expand(
                                bs_tracker_n_channels,
                                -1,
                                -1,
                            ),
                            field_r_by_wl=bs_tracker_field_r_t,
                            field_i_by_wl=bs_tracker_field_i_t,
                            wavelengths_m=bs_tracker_wavelengths_m_t,
                            bright_indices=tracker_idx_bright_t,
                            piston_basis=tracker_basis_t,
                            min_singular_per_nm=float(
                                params.tracker_observability_min_singular_per_nm
                            ),
                            target_rms_nm=float(
                                params.tracker_observability_crlb_target_nm
                            ),
                            compute_crlb=(
                                params.tracker_observability_crlb_weight > 0.0
                            ),
                            eps=float(params.tracker_observability_eps),
                            torch_mod=torch_mod,
                        )
                    )
                amplitude_mismatch_loss = torch_mod.zeros(
                    (),
                    device=device,
                    dtype=params_t.dtype,
                )
                if amplitude_mismatch_active:
                    amplitude_mismatch_loss = _amplitude_mismatch_loss_torch(
                        u_r=u_r.unsqueeze(0).expand(
                            bs_amplitude_n_channels, -1, -1
                        ),
                        u_i=u_i.unsqueeze(0).expand(
                            bs_amplitude_n_channels, -1, -1
                        ),
                        field_r_by_wl=bs_amplitude_field_r_t,
                        field_i_by_wl=bs_amplitude_field_i_t,
                        null_indices=idx_dark_t,
                        dump_indices=amplitude_dump_idx_t,
                        amplitude_basis=amplitude_basis_t,
                        min_dump_fraction=float(params.amplitude_dump_min_fraction),
                        probe_sigma_i=float(params.amplitude_mismatch_probe_sigma_i),
                        eps=float(params.amplitude_mismatch_eps),
                        torch_mod=torch_mod,
                    )
                amplitude_observability_loss = torch_mod.zeros(
                    (),
                    device=device,
                    dtype=params_t.dtype,
                )
                if amplitude_observability_active:
                    amplitude_observability_loss = (
                        _worst_delay_state_loss_torch(
                            _amplitude_observability_loss_torch,
                            n_delay_states,
                            u_r=u_r.unsqueeze(0).expand(
                                bs_amplitude_n_channels,
                                -1,
                                -1,
                            ),
                            u_i=u_i.unsqueeze(0).expand(
                                bs_amplitude_n_channels,
                                -1,
                                -1,
                            ),
                            field_r_by_wl=bs_amplitude_field_r_t,
                            field_i_by_wl=bs_amplitude_field_i_t,
                            port_indices=amplitude_observability_idx_t,
                            amplitude_basis=amplitude_basis_t,
                            min_singular_per_fractional_intensity=float(
                                params.amplitude_observability_min_singular_per_fractional_intensity
                            ),
                            eps=float(params.amplitude_observability_eps),
                            torch_mod=torch_mod,
                        )
                    )
                joint_observability_loss = torch_mod.zeros(
                    (),
                    device=device,
                    dtype=params_t.dtype,
                )
                joint_observability_crlb_loss = torch_mod.zeros(
                    (),
                    device=device,
                    dtype=params_t.dtype,
                )
                if joint_observability_active:
                    if params.joint_observability_weight > 0.0:
                        joint_observability_loss = (
                            _worst_delay_state_loss_torch(
                                _joint_phase_amplitude_observability_loss_torch,
                                n_delay_states,
                                u_r=u_r.unsqueeze(0).expand(
                                    bs_amplitude_n_channels,
                                    -1,
                                    -1,
                                ),
                                u_i=u_i.unsqueeze(0).expand(
                                    bs_amplitude_n_channels,
                                    -1,
                                    -1,
                                ),
                                field_r_by_wl=bs_amplitude_field_r_t,
                                field_i_by_wl=bs_amplitude_field_i_t,
                                wavelengths_m=bs_joint_wavelengths_m_t,
                                port_indices=joint_idx_t,
                                piston_basis=amplitude_basis_t,
                                amplitude_basis=amplitude_basis_t,
                                amplitude_scale=float(
                                    params.joint_observability_amplitude_scale
                                ),
                                min_singular_per_scaled_mode=float(
                                    params.joint_observability_min_singular_per_scaled_mode
                                ),
                                eps=float(params.joint_observability_eps),
                                torch_mod=torch_mod,
                            )
                        )
                    if params.joint_observability_crlb_weight > 0.0:
                        joint_observability_crlb_loss = (
                            _worst_delay_state_loss_torch(
                                _joint_phase_amplitude_observability_crlb_loss_torch,
                                n_delay_states,
                                u_r=u_r.unsqueeze(0).expand(
                                    bs_amplitude_n_channels,
                                    -1,
                                    -1,
                                ),
                                u_i=u_i.unsqueeze(0).expand(
                                    bs_amplitude_n_channels,
                                    -1,
                                    -1,
                                ),
                                field_r_by_wl=bs_amplitude_field_r_t,
                                field_i_by_wl=bs_amplitude_field_i_t,
                                wavelengths_m=bs_joint_wavelengths_m_t,
                                port_indices=joint_idx_t,
                                piston_basis=amplitude_basis_t,
                                amplitude_basis=amplitude_basis_t,
                                amplitude_scale=float(
                                    params.joint_observability_amplitude_scale
                                ),
                                target_per_scaled_mode=float(
                                    params.joint_observability_crlb_target_per_scaled_mode
                                ),
                                eps=float(params.joint_observability_eps),
                                torch_mod=torch_mod,
                            )
                        )
                joint_science_guard_loss = torch_mod.zeros(
                    (),
                    device=device,
                    dtype=params_t.dtype,
                )
                if params.joint_observability_science_null_guard_weight > 0.0:
                    joint_science_guard_loss = _science_null_guard_loss_torch(
                        leakage_loss=leakage_loss,
                        target_loss=bs_target_null_loss,
                        guard_factor=float(
                            params.joint_observability_science_null_guard_factor
                        ),
                        eps=float(params.joint_observability_eps),
                        torch_mod=torch_mod,
                    )
                total = (
                    leakage_loss
                    - tp_alpha * throughput_reward
                    + float(params.tracker_observability_weight) * tracker_loss
                    + float(params.tracker_observability_crlb_weight)
                    * tracker_crlb_loss
                    + float(params.amplitude_mismatch_weight) * amplitude_mismatch_loss
                    + float(params.amplitude_observability_weight)
                    * amplitude_observability_loss
                    + float(params.joint_observability_weight)
                    * joint_observability_loss
                    + float(params.joint_observability_crlb_weight)
                    * joint_observability_crlb_loss
                    + float(params.joint_observability_science_null_guard_weight)
                    * joint_science_guard_loss
                )
                return (
                    total,
                    leakage_loss,
                    throughput_reward,
                    tracker_loss,
                    tracker_crlb_loss,
                    amplitude_mismatch_loss,
                    amplitude_observability_loss,
                    joint_observability_loss,
                    joint_observability_crlb_loss,
                    joint_science_guard_loss,
                )

            def _bs_objective(params_t: Any) -> Any:
                (
                    total,
                    _leakage_loss,
                    _throughput_reward,
                    _tracker_loss,
                    _tracker_crlb_loss,
                    _amp_mismatch_loss,
                    _amp_observability_loss,
                    _joint_observability_loss,
                    _joint_observability_crlb_loss,
                    _joint_science_guard_loss,
                ) = _bs_objective_components(params_t)
                return total

            # Honour params.compile_objective / NULLSIM_TORCH_COMPILE for the
            # bootstrap objective too. The broadband `_objective` above is
            # never invoked when broadband_refine=False (the production
            # path), so without this the compile flag had no effect on the
            # actual hot loop.
            if params.compile_objective or _torch_compile_enabled():
                _bs_obj_fn = torch_mod.compile(
                    _bs_objective, mode="reduce-overhead", dynamic=False,
                )
            else:
                _bs_obj_fn = _bs_objective

            def _bs_components_np(x: NDArray[np.float64]) -> dict[str, float]:
                with torch_mod.no_grad():
                    p_t = torch_mod.from_numpy(
                        np.asarray(x, dtype=np.float64),
                    ).to(device)
                    (
                        total,
                        leakage_loss,
                        throughput_reward,
                        tracker_loss,
                        tracker_crlb_loss,
                        amp_mismatch_loss,
                        amp_observability_loss,
                        joint_observability_loss,
                        joint_observability_crlb_loss,
                        joint_science_guard_loss,
                    ) = _bs_objective_components(p_t)
                    return {
                        "optimization_objective_final": float(total.item()),
                        "optimization_leakage_loss_final": float(leakage_loss.item()),
                        "optimization_throughput_reward_final": float(
                            throughput_reward.item()
                        ),
                        "optimization_tracker_loss_final": float(tracker_loss.item()),
                        "optimization_tracker_crlb_loss_final": float(
                            tracker_crlb_loss.item()
                        ),
                        "optimization_amplitude_mismatch_loss_final": float(
                            amp_mismatch_loss.item()
                        ),
                        "optimization_amplitude_observability_loss_final": float(
                            amp_observability_loss.item()
                        ),
                        "optimization_joint_observability_loss_final": float(
                            joint_observability_loss.item()
                        ),
                        "optimization_joint_observability_crlb_loss_final": float(
                            joint_observability_crlb_loss.item()
                        ),
                        "optimization_joint_science_guard_loss_final": float(
                            joint_science_guard_loss.item()
                        ),
                    }

            def _bs_obj_and_grad_np(
                x: NDArray[np.float64],
            ) -> tuple[float, NDArray[np.float64]]:
                # Fused (loss, grad) — see _obj_and_grad_np above.
                p_t = (
                    torch_mod.from_numpy(np.asarray(x, dtype=np.float64))
                    .to(device)
                    .requires_grad_(True)
                )
                loss = _bs_obj_fn(p_t)
                loss.backward()
                return (
                    float(loss.item()),
                    p_t.grad.detach().cpu().numpy().astype(np.float64),
                )

            bs_seeds = [
                int(master_rng.integers(0, 2**31 - 1))
                for _ in range(params.bootstrap_n_restarts)
            ]
            bs_best_loss = math.inf
            bs_best_leakage_loss = math.inf
            bs_best_x: NDArray[np.float64] | None = None
            bs_best_selection_key: tuple[float, ...] | None = None
            bs_n_no_improve = 0
            for bs_index, bs_seed in enumerate(bs_seeds):
                if bs_index == 0 and warm_start_arr is not None:
                    x0 = warm_start_arr
                else:
                    x0 = np.random.default_rng(bs_seed).uniform(
                        0.0, 2.0 * np.pi, n_params_total,
                    )
                bs_res = minimize(
                    _bs_obj_and_grad_np,
                    x0,
                    jac=True,
                    # KILO uses full-Hessian BFGS (kilo/chip.py:1180,1212);
                    # the L-BFGS-B low-memory approximation lands the
                    # 576-dim multi-modal kernel-null basin selection
                    # in a different place from the same init, which is
                    # the dominant cause of "same algorithm, different
                    # null floor" runs. Switched in Tier 3.2.
                    method="BFGS",
                    options={"maxiter": params.bootstrap_max_iter},
                )
                loss = float(bs_res.fun)
                candidate_x = np.asarray(bs_res.x, dtype=np.float64)
                if not np.isfinite(loss) or not np.all(np.isfinite(candidate_x)):
                    continue
                candidate_components = _bs_components_np(candidate_x)
                candidate_leakage_loss = float(
                    candidate_components["optimization_leakage_loss_final"]
                )
                if not np.isfinite(candidate_leakage_loss):
                    continue
                candidate_key = _candidate_selection_key(
                    candidate_x,
                    loss,
                    candidate_leakage_loss,
                    bs_target_field_array,
                    n_rows,
                )
                patience_improved = (
                    bs_best_selection_key is None
                    or loss < bs_best_loss * (1.0 - params.restart_patience_reltol)
                )
                # KILO-style patience: bail the restart loop when the deep
                # basin has been found and subsequent restarts aren't
                # meaningfully improving (matches the broadband-refine
                # check below and KILO chip.py L1257-1269). Without this,
                # bootstrap_n_restarts=5 always runs all 5 even after
                # restart 1 has landed the deep basin.
                improved = (
                    bs_best_selection_key is None
                    or candidate_key < bs_best_selection_key
                )
                if improved:
                    bs_best_selection_key = candidate_key
                    bs_best_loss = loss
                    bs_best_x = candidate_x
                    bs_best_leakage_loss = _bs_components_np(bs_best_x)[
                        "optimization_leakage_loss_final"
                    ]
                bs_n_no_improve = (
                    0
                    if (patience_improved or params.tracker_observability_rank_priority and improved)
                    else bs_n_no_improve + 1
                )
                if _loss_meets_target(
                    bs_best_leakage_loss,
                    bs_target_field_array,
                    params.target_null_depth,
                    n_objective_rows=n_rows,
                    n_states=n_delay_states,
                ) and (
                    not params.tracker_observability_rank_priority
                    or bs_best_selection_key is None
                    or bs_best_selection_key[0] == 0.0
                ):
                    break
                if (
                    params.restart_patience > 0
                    and bs_n_no_improve >= params.restart_patience
                ):
                    break
            warm_start_arr = bs_best_x

        # Validation: broadband_refine=False only makes sense when
        # there's a bootstrap warm-start to fall back on. Otherwise we'd
        # have no chip at all.
        if not params.broadband_refine and warm_start_arr is None:
            raise ValueError(
                "chip_optimization.torch: broadband_refine=False "
                "requires single_lambda_bootstrap=True and n_wl>1 so a "
                "bootstrap warm-start is available as the final params."
            )

        # Fast-path: when the kernel will deploy monochromatically
        # (kernel_rescales_per_wavelength=False below) the broadband
        # restart loop optimizes the wrong objective. The bootstrap
        # already minimized the single-wavelength leakage at wl_design,
        # which is exactly what the kernel applies. Skip the broadband
        # restart loop and use the bootstrap warm-start directly.
        best_selection_key: tuple[float, ...] | None = None
        best_success = False
        best_message = ""
        best_convergence_history: tuple[tuple[int, float], ...] | None = None
        best_null_history: tuple[tuple[int, float], ...] | None = None
        n_restarts_completed = 0
        if not params.broadband_refine:
            best_x = warm_start_arr
            if "bs_best_loss" in locals():
                best_loss = float(bs_best_loss)
                best_iterations = params.bootstrap_max_iter
            else:
                best_loss = _obj_np(warm_start_arr)
                best_iterations = 0
            best_success = True
            best_message = "bootstrap_warm_start"
        else:
            best_x = None
            best_loss = math.inf
            best_leakage_loss = math.inf
            best_iterations = 0
            n_no_improve = 0
            # DAC quantization is applied at the design wavelength only; the torch
            # autograd path skips it inside the loss (non-differentiable). We
            # quantize the final iterate post-hoc, matching the kernel-stage rule.
            for k, seed in enumerate(restart_seeds):
                if k == 0 and warm_start_arr is not None:
                    x0 = warm_start_arr
                else:
                    x0 = np.random.default_rng(seed).uniform(
                        0.0, 2.0 * np.pi, n_params_total,
                    )
                # capture_convergence_history: record (nfev, loss) per BFGS
                # iteration. Pre-fix the torch backend silently ignored the
                # flag and the debug-report panel always claimed "history
                # not captured" (review 2026-06-10 Issue 9).
                # The objective is dimensional and goes negative with the
                # throughput reward, so the published null (leakage only,
                # reported convention) is recorded beside it (Issue 23).
                restart_history: list[tuple[int, float]] | None = None
                restart_null_history: list[tuple[int, float]] | None = None
                restart_callback = None
                restart_obj = _obj_and_grad_np
                if params.capture_convergence_history:
                    nfev_counter = [0]
                    restart_history = []
                    restart_null_history = []

                    def restart_obj(
                        x: NDArray[np.float64],
                        _counter: list[int] = nfev_counter,
                    ) -> tuple[float, NDArray[np.float64]]:
                        _counter[0] += 1
                        return _obj_and_grad_np(x)

                    def restart_callback(
                        xk: NDArray[np.float64],
                        _h: list[tuple[int, float]] = restart_history,
                        _hn: list[tuple[int, float]] = restart_null_history,
                        _counter: list[int] = nfev_counter,
                    ) -> None:
                        components = _objective_components_np(xk)
                        _h.append(
                            (_counter[0], components["optimization_objective_final"])
                        )
                        _hn.append((
                            _counter[0],
                            _reported_null_depth_for_target(
                                components["optimization_leakage_loss_final"],
                                target_field_array,
                                n_objective_rows=n_rows,
                                n_states=n_delay_states,
                            ),
                        ))

                    restart_callback(x0)

                result = minimize(
                    restart_obj,
                    x0,
                    jac=True,
                    # See bootstrap call above — Tier 3.2 BFGS switch.
                    method="BFGS",
                    options={"maxiter": params.max_iter},
                    callback=restart_callback,
                )
                loss = float(result.fun)
                candidate_x = np.asarray(result.x, dtype=np.float64)
                n_restarts_completed = k + 1
                if not math.isfinite(loss) or not np.all(np.isfinite(candidate_x)):
                    n_no_improve += 1
                    continue
                candidate_components = _objective_components_np(candidate_x)
                candidate_leakage_loss = float(
                    candidate_components["optimization_leakage_loss_final"]
                )
                if not math.isfinite(candidate_leakage_loss):
                    n_no_improve += 1
                    continue
                candidate_key = _candidate_selection_key(
                    candidate_x,
                    loss,
                    candidate_leakage_loss,
                    target_field_array,
                    n_rows,
                )
                patience_improved = (
                    best_selection_key is None
                    or loss < best_loss * (1.0 - params.restart_patience_reltol)
                )
                improved = (
                    best_selection_key is None or candidate_key < best_selection_key
                )
                if improved:
                    best_selection_key = candidate_key
                    best_loss = loss
                    best_x = candidate_x
                    best_leakage_loss = candidate_leakage_loss
                    best_iterations = int(getattr(result, "nit", 0))
                    best_success = bool(getattr(result, "success", False))
                    best_convergence_history = (
                        tuple(restart_history)
                        if restart_history is not None
                        else None
                    )
                    best_null_history = (
                        tuple(restart_null_history)
                        if restart_null_history is not None
                        else None
                    )
                    best_message = str(getattr(result, "message", ""))
                n_no_improve = (
                    0
                    if (
                        patience_improved
                        or params.tracker_observability_rank_priority
                        and improved
                    )
                    else n_no_improve + 1
                )
                if _loss_meets_target(
                    best_leakage_loss,
                    target_field_array,
                    params.target_null_depth,
                    n_objective_rows=n_rows,
                    n_states=n_delay_states,
                ) and (
                    not params.tracker_observability_rank_priority
                    or best_selection_key is None
                    or best_selection_key[0] == 0.0
                ):
                    break
                if (
                    params.restart_patience > 0
                    and n_no_improve >= params.restart_patience
                ):
                    break

        if best_x is None:
            raise RuntimeError(
                "chip_optimization.torch produced no candidate after all "
                "restarts; check n_restarts >= 1 and the input field."
            )
        optimal_params = _quantize_phases(best_x, params.dac_bits)
        # Re-evaluate the loss at the quantized iterate so the recorded
        # ``optimization_loss_final`` matches the returned params exactly.
        loss_final = _obj_np(optimal_params)
        objective_component_metadata = _objective_components_np(optimal_params)
        objective_component_metadata["optimization_objective_final"] = loss_final
        legacy_normalized_null_final = _normalized_loss_for_target(
            objective_component_metadata["optimization_leakage_loss_final"],
            target_field_array,
        )
        reported_null_final = _reported_null_depth_for_target(
            objective_component_metadata["optimization_leakage_loss_final"],
            target_field_array,
            n_objective_rows=n_rows,
            n_states=n_delay_states,
        )
        target_met_final = reported_null_final <= float(params.target_null_depth)
        objective_component_metadata.update(
            {
                "optimization_target_null_depth": float(params.target_null_depth),
                "optimization_normalized_null_legacy_final": float(
                    legacy_normalized_null_final
                ),
                "optimization_reported_null_depth_final": float(reported_null_final),
                "optimization_reported_null_depth_model": (
                    "spectrally reduced leakage divided by mean input power per "
                    "wavelength and objective row count"
                ),
                "optimization_null_target_met_final": bool(target_met_final),
                "optimization_selection_key_final": (
                    None if best_selection_key is None else tuple(best_selection_key)
                ),
                "optimization_best_result_success": bool(best_success),
                "optimization_best_result_message": best_message,
            }
        )
        if not target_met_final:
            warnings.warn(
                "chip_optimization.torch deployed a chip whose reported null "
                f"{reported_null_final:.6g} exceeds target_null_depth="
                f"{float(params.target_null_depth):.6g}.",
                RuntimeWarning,
                stacklevel=2,
            )
        iterations = best_iterations

        u_r_opt_diag = None
        u_i_opt_diag = None
        if tracker_active or amplitude_active:
            with torch_mod.no_grad():
                p_t = torch_mod.from_numpy(
                    np.asarray(optimal_params, dtype=np.float64),
                ).to(device)
                scaled = p_t.unsqueeze(0).expand(n_wl, -1).clone()
                scaled[:, :n_mzi] = scaled[:, :n_mzi] + theta_offsets_t[:, None]
                scaled[:, n_mzi : 2 * n_mzi] = (
                    scaled[:, n_mzi : 2 * n_mzi] * phase_scales_t[:, None]
                )
                scaled[:, 2 * n_mzi : 2 * n_mzi + params.n_modes] = (
                    scaled[:, 2 * n_mzi : 2 * n_mzi + params.n_modes]
                    * phase_scales_t[:, None]
                )
                u_r_opt, u_i_opt = _clements_unitary_ri_batched(
                    scaled,
                    params.n_modes,
                    torch_mod=torch_mod,
                )
                u_r_opt_diag = u_r_opt
                u_i_opt_diag = u_i_opt

        tracker_final_metadata: dict[str, Any] = {}
        if tracker_active:
            with torch_mod.no_grad():
                singular_t = _tracker_per_state_singular_values_torch(
                    n_states=tracker_n_field_states,
                    u_r=u_r_opt_diag.index_select(0, tracker_eval_wave_idx_t),
                    u_i=u_i_opt_diag.index_select(0, tracker_eval_wave_idx_t),
                    field_r_by_wl=tracker_field_r_sel,
                    field_i_by_wl=tracker_field_i_sel,
                    wavelengths_m=tracker_wavelengths_m_t,
                    bright_indices=tracker_idx_bright_t,
                    piston_basis=tracker_basis_t,
                    eps=float(params.tracker_observability_eps),
                    torch_mod=torch_mod,
                )
                per_state_np = (
                    singular_t.detach().cpu().numpy().astype(np.float64, copy=False)
                )
            # Headline values describe the worst delay state (Issue 21); the
            # CRLB is the largest over states.
            tracker = _worst_state_singular_summary(
                per_state_np,
                float(params.tracker_observability_min_singular_per_nm),
                int(tracker_basis.chip_basis.shape[1]),
            )
            crlb_rms_nm = max(
                _tracker_crlb_rms_nm_from_singular_values(
                    values, eps=float(params.tracker_observability_eps)
                )
                for values in per_state_np
            )
            crlb_loss = (
                crlb_rms_nm / float(params.tracker_observability_crlb_target_nm)
            ) ** 2
            tracker_final_metadata = {
                "tracker_observability_singular_values_per_nm_final": (
                    tracker["singular_values"]
                ),
                "tracker_observability_singular_values_per_nm_per_state_final": (
                    tracker["singular_values_per_state"]
                ),
                "tracker_observability_min_singular_per_nm_final": (
                    tracker["min_singular"]
                ),
                "tracker_observability_condition_final": tracker["condition"],
                "tracker_observability_crlb_rms_nm_final": float(crlb_rms_nm),
                "tracker_observability_crlb_loss_final": float(crlb_loss),
                "tracker_observability_rank_at_floor_final": tracker["rank_at_floor"],
                "tracker_observability_floor_met_final": tracker["floor_met"],
            }

        amplitude_final_metadata: dict[str, Any] = {}
        if amplitude_mismatch_active:
            with torch_mod.no_grad():
                null_fraction_t, dump_fraction_t = (
                    _amplitude_mismatch_fractions_torch(
                        u_r=u_r_opt_diag.index_select(
                            0,
                            amplitude_mismatch_eval_wave_idx_t,
                        ),
                        u_i=u_i_opt_diag.index_select(
                            0,
                            amplitude_mismatch_eval_wave_idx_t,
                        ),
                        field_r_by_wl=amplitude_mismatch_field_r_sel,
                        field_i_by_wl=amplitude_mismatch_field_i_sel,
                        null_indices=idx_dark_t,
                        dump_indices=amplitude_dump_idx_t,
                        amplitude_basis=amplitude_basis_t,
                        probe_sigma_i=float(params.amplitude_mismatch_probe_sigma_i),
                        eps=float(params.amplitude_mismatch_eps),
                        torch_mod=torch_mod,
                    )
                )
                null_fraction_np = (
                    null_fraction_t.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=False)
                )
                dump_fraction_np = (
                    dump_fraction_t.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=False)
                )
            amplitude_final_metadata = {
                "amplitude_mismatch_null_fraction_mean_final": float(
                    np.mean(null_fraction_np)
                ),
                "amplitude_mismatch_null_fraction_max_final": float(
                    np.max(null_fraction_np)
                ),
                "amplitude_dump_fraction_mean_final": float(np.mean(dump_fraction_np)),
                "amplitude_dump_fraction_min_final": float(np.min(dump_fraction_np)),
                "amplitude_dump_fraction_target_met_final": bool(
                    np.all(dump_fraction_np >= float(params.amplitude_dump_min_fraction))
                ),
            }
        if amplitude_observability_active:
            with torch_mod.no_grad():
                singular_t = _per_delay_state_singular_values_torch(
                    _amplitude_observability_singular_values_torch,
                    n_delay_states,
                    u_r=u_r_opt_diag.index_select(
                        0,
                        amplitude_observability_eval_wave_idx_t,
                    ),
                    u_i=u_i_opt_diag.index_select(
                        0,
                        amplitude_observability_eval_wave_idx_t,
                    ),
                    field_r_by_wl=amplitude_observability_field_r_sel,
                    field_i_by_wl=amplitude_observability_field_i_sel,
                    port_indices=amplitude_observability_idx_t,
                    amplitude_basis=amplitude_basis_t,
                    eps=float(params.amplitude_observability_eps),
                    torch_mod=torch_mod,
                )
                amp_singular_np = (
                    singular_t.detach().cpu().numpy().astype(np.float64, copy=False)
                )
            amp = _worst_state_singular_summary(
                amp_singular_np,
                float(params.amplitude_observability_min_singular_per_fractional_intensity),
                int(amplitude_basis.chip_basis.shape[1]),
            )
            amplitude_final_metadata.update(
                {
                    "amplitude_observability_singular_values_final": (
                        amp["singular_values"]
                    ),
                    "amplitude_observability_singular_values_per_state_final": (
                        amp["singular_values_per_state"]
                    ),
                    "amplitude_observability_min_singular_final": amp["min_singular"],
                    "amplitude_observability_condition_final": amp["condition"],
                    "amplitude_observability_rank_at_floor_final": amp["rank_at_floor"],
                    "amplitude_observability_floor_met_final": amp["floor_met"],
                }
            )

        joint_final_metadata: dict[str, Any] = {}
        if joint_observability_active:
            with torch_mod.no_grad():
                singular_t = _per_delay_state_singular_values_torch(
                    _joint_phase_amplitude_observability_singular_values_torch,
                    n_delay_states,
                    u_r=u_r_opt_diag.index_select(0, joint_eval_wave_idx_t),
                    u_i=u_i_opt_diag.index_select(0, joint_eval_wave_idx_t),
                    field_r_by_wl=joint_field_r_sel,
                    field_i_by_wl=joint_field_i_sel,
                    wavelengths_m=joint_wavelengths_m_t,
                    port_indices=joint_idx_t,
                    piston_basis=amplitude_basis_t,
                    amplitude_basis=amplitude_basis_t,
                    amplitude_scale=float(params.joint_observability_amplitude_scale),
                    eps=float(params.joint_observability_eps),
                    torch_mod=torch_mod,
                )
                joint_singular_np = (
                    singular_t.detach().cpu().numpy().astype(np.float64, copy=False)
                )
            joint_n_modes = 2 * int(amplitude_basis.chip_basis.shape[1])
            joint = _worst_state_singular_summary(
                joint_singular_np,
                float(params.joint_observability_min_singular_per_scaled_mode),
                joint_n_modes,
            )
            joint_final_metadata = {
                "joint_observability_singular_values_final": joint["singular_values"],
                "joint_observability_singular_values_per_state_final": (
                    joint["singular_values_per_state"]
                ),
                "joint_observability_min_singular_final": joint["min_singular"],
                "joint_observability_condition_final": joint["condition"],
                "joint_observability_n_modes_final": joint_n_modes,
                "joint_observability_rank_at_floor_final": joint["rank_at_floor"],
                # Only meaningful when the hinge term enforced the floor;
                # in CRLB-only runs (weight=0, crlb_weight>0) the floor was
                # never part of the objective (review 2026-06-10).
                "joint_observability_floor_enforced": bool(
                    params.joint_observability_weight > 0.0
                ),
                "joint_observability_floor_met_final": joint["floor_met"],
            }

        # Finding 3 fix: the torch backend's _clements_unitary_ri uses
        # KILO's column-alternating MZI ordering, but the downstream
        # chip.kernel_mzi_mesh reconstructs via physics.clements (a
        # different lexicographic ordering). Without canonicalization, the
        # kernel rebuilds a *different* unitary from the same param vector,
        # silently breaking the optimizer→kernel handoff. We build the
        # optimized unitary in torch's native ordering, decompose it via
        # physics.clements, and store the re-packed canonical params so
        # the kernel reproduces the optimized unitary bit-for-bit.
        def _torch_build_unitary_numpy(
            p_np: NDArray[np.float64], n_modes_: int
        ) -> NDArray[np.complex128]:
            with torch_mod.no_grad():
                p_t = torch_mod.from_numpy(
                    np.asarray(p_np, dtype=np.float64)
                ).to(device)
                u_r, u_i = _clements_unitary_ri(
                    p_t, n_modes_, torch_mod=torch_mod
                )
                return (u_r.cpu().numpy() + 1j * u_i.cpu().numpy()).astype(
                    np.complex128
                )

        canonical_params = _canonicalize_clements_params(
            optimal_params,
            params.n_modes,
            build_unitary_complex=_torch_build_unitary_numpy,
        )
        # Pre-DAC optimum in the same physics.clements layout, for the DAC
        # bit-depth diagnostic (review 2026-11-07 row 60).
        canonical_params_unquantized = _canonicalize_clements_params(
            np.asarray(best_x, dtype=np.float64),
            params.n_modes,
            build_unitary_complex=_torch_build_unitary_numpy,
        )

        # Tier 3.1 (option 2): publish the chromatic U_stack the optimizer
        # actually evaluated, so the chip stage can deploy verbatim
        # instead of going through a canonicalize → physics.clements
        # rebuild + per-λ rescale. The rebuild path lost the chromaticity
        # because the two Clements layouts produce different unitaries
        # when phases are rescaled (Ultra-review High #5). Building
        # U(λ) here uses the same KILO column-alternating layout and the same
        # phase rescaling the broadband objective used, so the deployed
        # chip matches the optimizer at every detector-grid wavelength.
        #
        # IMPORTANT: build U(λ) over the FULL detector-grid wavelengths
        # (full_wavelength_grid_um), not the subsampled optimization
        # grid. chip.kernel_mzi_mesh asserts u_stack.shape[2] equals the
        # state's wavelength grid length; supplying the 5-bin
        # optimization stack triggers a ValueError at handoff after the
        # entire (expensive) optimization completes. The optimized
        # params describe a single physical chip whose response is
        # evaluable at any wavelength; re-evaluating at all 20 bins for
        # deployment costs ~20 quick unitary builds, trivial vs the
        # optimizer's thousands of grad evals.
        n_mzi = params.n_modes * (params.n_modes - 1) // 2
        n_wl_full = len(full_wavelength_grid_um)
        full_phase_scales = _phase_scale_factors_numpy(
            wavelength_grid_um=full_wavelength_grid_um,
            wl_design_um=params.wl_design_um,
            chromaticity_params=params,
            rescale_per_wavelength=True,
        )
        full_theta_offsets = _theta_offsets_numpy(
            wavelength_grid_um=full_wavelength_grid_um,
            wl_design_um=params.wl_design_um,
            chromaticity_params=params,
            apply_chromaticity=True,
        )
        scaled_full = np.repeat(
            optimal_params[np.newaxis, :],
            n_wl_full,
            axis=0,
        )
        scaled_full[:, :n_mzi] += full_theta_offsets[:, np.newaxis]
        scaled_full[:, n_mzi : 2 * n_mzi] *= full_phase_scales[:, np.newaxis]
        scaled_full[:, 2 * n_mzi : 2 * n_mzi + params.n_modes] *= (
            full_phase_scales[:, np.newaxis]
        )
        with torch_mod.no_grad():
            scaled_t = torch_mod.from_numpy(scaled_full).to(device)
            u_r_full, u_i_full = _clements_unitary_ri_batched(
                scaled_t,
                params.n_modes,
                torch_mod=torch_mod,
            )
            u_stack = (
                u_r_full.cpu().numpy() + 1j * u_i_full.cpu().numpy()
            ).transpose(1, 2, 0).astype(np.complex128, copy=False)
        if n_wl_full > 1:
            u_stack_edge_delta_max = float(
                np.max(np.abs(u_stack[:, :, 0] - u_stack[:, :, -1]))
            )
        else:
            u_stack_edge_delta_max = 0.0

        return _write_results(
            state=state,
            instance_id=self.context.instance_id,
            optimal_params=canonical_params,
            loss_final=loss_final,
            iterations=iterations,
            backend_name="torch",
            # With the precomputed U_stack the chip stage uses it
            # verbatim and ignores the legacy rescale flag. We keep the
            # flag at False for back-compat with diagnostics that read
            # the field; semantically it now means "I shipped a
            # precomputed chromatic U_stack instead of expecting you to
            # rescale my canonical params".
            kernel_rescales_per_wavelength=False,
            n_restarts_completed=n_restarts_completed,
            normalized_loss_final=reported_null_final,
            convergence_history=best_convergence_history,
            convergence_history_normalized_null=best_null_history,
            optimal_params_unquantized=canonical_params_unquantized,
            optimal_u_stack=u_stack,
            backend_metadata={
                "torch_device": str(device),
                "torch_unitary_builder": "batched_clements_ri",
                "torch_wavelength_batch_size": int(n_wl_full),
                "torch_u_stack_edge_delta_max": u_stack_edge_delta_max,
                "disk_null_floor_note": (
                    "chip-realization dependent"
                    if params.objective_is_flux_scale_dependent
                    else "flux-scale invariant"
                ),
                **_chip_chromaticity_metadata(params),
                **warm_start_metadata,
                **_spectral_loss_metadata(params),
                **_optimizer_wavelength_metadata(
                    full_wavelength_grid_um=full_wavelength_grid_um,
                    optimizer_wavelength_grid_um=wavelength_grid_um,
                    optimizer_wavelength_indices=optimizer_wavelength_indices,
                ),
                **_delay_state_metadata(params, delay_state_fields),
                **tracker_metadata,
                **tracker_final_metadata,
                **amplitude_metadata,
                **amplitude_final_metadata,
                **joint_metadata,
                **joint_final_metadata,
                **objective_component_metadata,
            },
        )


def _resolve_external_warm_start(
    *,
    state: SimulationState,
    params: ChipOptimizationParams,
    wavelength_grid_um: tuple[float, ...],
    torch_mod: Any,
    device: Any,
) -> tuple[NDArray[np.float64] | None, dict[str, Any]]:
    source = params.warm_start_source
    if source is None:
        return None, {}

    source_key, payload = _warm_start_payload(state, source)
    raw_u_stack = payload.get("optimal_u_stack")
    if raw_u_stack is not None:
        u_stack = np.asarray(raw_u_stack, dtype=np.complex128)
        if u_stack.shape[:2] != (params.n_modes, params.n_modes):
            raise ValueError(
                "chip_optimization.torch warm_start_source "
                f"{source_key!r} optimal_u_stack shape {u_stack.shape} does "
                f"not match n_modes={params.n_modes}."
            )
        if u_stack.shape[2] == len(wavelength_grid_um):
            target_index = int(
                np.argmin(
                    np.abs(
                        np.asarray(wavelength_grid_um, dtype=np.float64)
                        - float(params.wl_design_um)
                    )
                )
            )
        else:
            target_index = int(u_stack.shape[2] // 2)
        fit_params, fit_metadata = _fit_native_params_to_unitary(
            target_u=u_stack[:, :, target_index],
            params=params,
            torch_mod=torch_mod,
            device=device,
        )
        return fit_params, {
            "warm_start_source": source_key,
            "warm_start_kind": "fit_optimal_u_stack",
            "warm_start_wavelength_index": int(target_index),
            **fit_metadata,
        }

    raw_params = payload.get("optimal_chip_params")
    if raw_params is not None and str(payload.get("backend_name")) == "torch":
        warm = np.asarray(raw_params, dtype=np.float64)
        if warm.shape != (_n_params(params.n_modes),):
            raise ValueError(
                "chip_optimization.torch warm_start_source "
                f"{source_key!r} optimal_chip_params length {warm.size} does "
                f"not match n_modes={params.n_modes}."
            )
        return warm, {
            "warm_start_source": source_key,
            "warm_start_kind": "torch_native_params",
        }

    raise ValueError(
        "chip_optimization.torch warm_start_source must publish "
        "optimal_u_stack, or torch-native optimal_chip_params. "
        f"Source {source_key!r} has keys {sorted(payload)}."
    )


def _warm_start_payload(
    state: SimulationState,
    source: str,
) -> tuple[str, dict[str, Any]]:
    components = state.results.components
    if source.lower() == "auto":
        matches = [
            key for key in components
            if key.lower().startswith("chip_optimization")
        ]
        if not matches:
            raise KeyError(
                "chip_optimization.torch warm_start_source='auto' found no "
                "chip_optimization.* component."
            )
        if len(matches) > 1:
            raise ValueError(
                "chip_optimization.torch warm_start_source='auto' is "
                f"ambiguous; matching components: {matches}."
            )
        source_key = matches[0]
    else:
        source_key = source
        if source_key not in components:
            raise KeyError(
                "chip_optimization.torch warm_start_source="
                f"{source!r} not found. Available keys: {sorted(components)}."
            )
    payload = components[source_key]
    if not isinstance(payload, dict):
        raise TypeError(
            f"chip_optimization.torch warm_start_source {source_key!r} "
            f"payload must be a dict, got {type(payload).__name__}."
        )
    return source_key, payload


def _fit_native_params_to_unitary(
    *,
    target_u: NDArray[np.complex128],
    params: ChipOptimizationParams,
    torch_mod: Any,
    device: Any,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    target = np.asarray(target_u, dtype=np.complex128)
    if target.shape != (params.n_modes, params.n_modes):
        raise ValueError(
            "chip_optimization.torch warm-start target unitary shape "
            f"{target.shape} does not match n_modes={params.n_modes}."
        )
    target_r = torch_mod.from_numpy(target.real.astype(np.float64)).to(device)
    target_i = torch_mod.from_numpy(target.imag.astype(np.float64)).to(device)

    def _fit_objective(params_t: Any) -> Any:
        u_r, u_i = _clements_unitary_ri(
            params_t,
            params.n_modes,
            torch_mod=torch_mod,
        )
        return torch_mod.mean((u_r - target_r) ** 2 + (u_i - target_i) ** 2)

    if params.compile_objective or _torch_compile_enabled():
        _fit_obj_fn = torch_mod.compile(
            _fit_objective,
            mode="reduce-overhead",
            dynamic=False,
        )
    else:
        _fit_obj_fn = _fit_objective

    def _fit_obj_np(x: NDArray[np.float64]) -> float:
        with torch_mod.no_grad():
            p_t = torch_mod.from_numpy(np.asarray(x, dtype=np.float64)).to(device)
            return float(_fit_obj_fn(p_t).item())

    def _fit_grad_np(x: NDArray[np.float64]) -> NDArray[np.float64]:
        p_t = (
            torch_mod.from_numpy(np.asarray(x, dtype=np.float64))
            .to(device)
            .requires_grad_(True)
        )
        loss = _fit_obj_fn(p_t)
        loss.backward()
        return p_t.grad.detach().cpu().numpy().astype(np.float64)

    x0 = np.zeros(_n_params(params.n_modes), dtype=np.float64)
    result = minimize(
        _fit_obj_np,
        x0,
        jac=_fit_grad_np,
        method="BFGS",
        options={"maxiter": int(params.warm_start_fit_max_iter)},
    )
    return np.asarray(result.x, dtype=np.float64), {
        "warm_start_fit_loss": float(result.fun),
        "warm_start_fit_iterations": int(getattr(result, "nit", 0)),
        "warm_start_fit_success": bool(result.success),
    }


__all__ = [
    "ChipOptimizationTorch",
    "_clements_unitary_ri",
    "_clements_unitary_ri_batched",
    "_amplitude_mismatch_fractions_torch",
    "_amplitude_mismatch_loss_torch",
    "_amplitude_observability_loss_torch",
    "_amplitude_observability_singular_values_torch",
    "_joint_phase_amplitude_observability_loss_torch",
    "_joint_phase_amplitude_observability_crlb_loss_torch",
    "_joint_phase_amplitude_observability_singular_values_torch",
    "_tracker_observability_crlb_loss_torch",
    "_tracker_observability_loss_torch",
    "_tracker_observability_singular_values_torch",
    "_tracker_observability_wavelength_indices",
]


def _tracker_observability_wavelength_indices(
    n_wavelengths: int,
    n_spectral_channels: int,
) -> NDArray[np.int64]:
    """Representative optimizer wavelengths for the tracker-topology proxy."""

    if n_wavelengths <= 0:
        raise ValueError("tracker observability requires at least one wavelength.")
    if n_spectral_channels <= 0 or n_spectral_channels >= n_wavelengths:
        return np.arange(n_wavelengths, dtype=np.int64)
    groups = np.array_split(
        np.arange(n_wavelengths, dtype=np.int64),
        n_spectral_channels,
    )
    return np.asarray(
        [int(group[(len(group) - 1) // 2]) for group in groups if len(group) > 0],
        dtype=np.int64,
    )


def _amplitude_observability_port_indices(
    params: ChipOptimizationParams,
) -> NDArray[np.int64]:
    """Return output-port indices used by the amplitude observability proxy."""

    return _port_indices_for_set(
        params,
        params.amplitude_observability_port_set,
        option_name="amplitude_observability_port_set",
    )


def _joint_observability_port_indices(
    params: ChipOptimizationParams,
) -> NDArray[np.int64]:
    """Return output-port indices used by the joint observability proxy."""

    return _port_indices_for_set(
        params,
        params.joint_observability_port_set,
        option_name="joint_observability_port_set",
    )


def _port_indices_for_set(
    params: ChipOptimizationParams,
    port_set: str,
    *,
    option_name: str,
) -> NDArray[np.int64]:
    if port_set == "bright":
        return np.arange(int(params.n_bright_ports), dtype=np.int64)
    if port_set == "bright_dark":
        return np.arange(
            int(params.n_bright_ports) + int(params.n_dark_ports),
            dtype=np.int64,
        )
    if port_set == "all":
        return np.arange(int(params.n_modes), dtype=np.int64)
    raise ValueError(f"Unknown {option_name} {port_set!r}.")


def _tracker_regularized_singular_values_torch(
    *,
    u_r: Any,
    u_i: Any,
    field_r_by_wl: Any,
    field_i_by_wl: Any,
    wavelengths_m: Any,
    bright_indices: Any,
    piston_basis: Any,
    eps: float,
    torch_mod: Any,
) -> Any:
    """Regularized singular values of the tracker Fisher-proxy matrix."""

    matrix = _tracker_observability_matrix_torch(
        u_r=u_r,
        u_i=u_i,
        field_r_by_wl=field_r_by_wl,
        field_i_by_wl=field_i_by_wl,
        wavelengths_m=wavelengths_m,
        bright_indices=bright_indices,
        piston_basis=piston_basis,
        eps=eps,
        torch_mod=torch_mod,
    )
    return _regularized_singular_values_torch(
        matrix,
        eps=eps,
        torch_mod=torch_mod,
    )


def _tracker_hinge_loss_from_singular_torch(
    singular_values: Any,
    *,
    n_modes: int,
    min_singular_per_nm: float,
    eps: float,
    torch_mod: Any,
) -> Any:
    target = torch_mod.tensor(
        float(min_singular_per_nm),
        device=singular_values.device,
        dtype=singular_values.dtype,
    )
    tiny = torch_mod.tensor(
        float(eps),
        device=singular_values.device,
        dtype=singular_values.dtype,
    )
    log_deficit = torch_mod.relu(
        torch_mod.log(target) - torch_mod.log(singular_values + tiny)
    )
    loss = torch_mod.sum(log_deficit**2)
    missing = n_modes - int(singular_values.numel())
    if missing > 0:
        missing_deficit = torch_mod.log(target) - torch_mod.log(tiny)
        loss = loss + float(missing) * missing_deficit**2
    return loss / float(n_modes)


def _tracker_crlb_loss_from_singular_torch(
    singular_values: Any,
    *,
    n_modes: int,
    target_rms_nm: float,
    eps: float,
    torch_mod: Any,
) -> Any:
    tiny = torch_mod.tensor(
        float(eps),
        device=singular_values.device,
        dtype=singular_values.dtype,
    )
    target = torch_mod.tensor(
        float(target_rms_nm),
        device=singular_values.device,
        dtype=singular_values.dtype,
    )
    std_nm = 1.0 / torch_mod.clamp(singular_values, min=tiny)
    missing = n_modes - int(singular_values.numel())
    if missing > 0:
        missing_std = torch_mod.full(
            (missing,),
            1.0 / float(eps),
            device=std_nm.device,
            dtype=std_nm.dtype,
        )
        std_nm = torch_mod.cat((std_nm, missing_std))
    return torch_mod.mean((std_nm / target) ** 2)


def _tracker_observability_losses_torch(
    *,
    u_r: Any,
    u_i: Any,
    field_r_by_wl: Any,
    field_i_by_wl: Any,
    wavelengths_m: Any,
    bright_indices: Any,
    piston_basis: Any,
    min_singular_per_nm: float,
    target_rms_nm: float,
    compute_crlb: bool,
    eps: float,
    torch_mod: Any,
) -> tuple[Any, Any]:
    """Hinge and CRLB tracker penalties from ONE eigendecomposition.

    With both tracker weights active the pre-fix objective rebuilt the
    identical response matrix and Gram eigendecomposition twice per
    evaluation (review 2026-06-10 Issue 31).
    """

    singular_values = _tracker_regularized_singular_values_torch(
        u_r=u_r,
        u_i=u_i,
        field_r_by_wl=field_r_by_wl,
        field_i_by_wl=field_i_by_wl,
        wavelengths_m=wavelengths_m,
        bright_indices=bright_indices,
        piston_basis=piston_basis,
        eps=eps,
        torch_mod=torch_mod,
    )
    n_modes = int(piston_basis.shape[1])
    hinge = _tracker_hinge_loss_from_singular_torch(
        singular_values,
        n_modes=n_modes,
        min_singular_per_nm=min_singular_per_nm,
        eps=eps,
        torch_mod=torch_mod,
    )
    if not compute_crlb:
        return hinge, torch_mod.zeros(
            (),
            device=singular_values.device,
            dtype=singular_values.dtype,
        )
    crlb = _tracker_crlb_loss_from_singular_torch(
        singular_values,
        n_modes=n_modes,
        target_rms_nm=target_rms_nm,
        eps=eps,
        torch_mod=torch_mod,
    )
    return hinge, crlb


_CHANNEL_AXIS_KWARGS = ("u_r", "u_i", "field_r_by_wl", "field_i_by_wl", "wavelengths_m")


def _per_delay_state_torch(fn: Any, n_states: int, **kwargs: Any) -> list[Any]:
    """Evaluate an observability helper once per delay state.

    Observability channel axes are wavelength-major with ``n_states`` delay
    states innermost, so ``x[state::n_states]`` selects one state. The
    fringe tracker sees one pointing at a time: the states are alternatives,
    not simultaneous measurements. Stacking them into one response matrix
    scaled every singular value by ``sqrt(n_states)`` for identical states
    and could hide a state that is unobservable alone (review 2026-11-07
    Issue 21), so each state gets its own matrix and callers act on the
    worst one.
    """

    return [
        fn(**{
            key: (value[state::n_states] if key in _CHANNEL_AXIS_KWARGS else value)
            for key, value in kwargs.items()
        })
        for state in range(int(n_states))
    ]


def _worst_delay_state_loss_torch(fn: Any, n_states: int, **kwargs: Any) -> Any:
    """Largest per-state value of a scalar observability penalty."""

    return kwargs["torch_mod"].stack(_per_delay_state_torch(fn, n_states, **kwargs)).max()


def _per_delay_state_singular_values_torch(
    fn: Any, n_states: int, **kwargs: Any
) -> Any:
    """Per-state singular values, shape ``(n_states, n_modes)``."""

    return kwargs["torch_mod"].stack(_per_delay_state_torch(fn, n_states, **kwargs))


def _tracker_worst_state_losses_torch(*, n_states: int, **kwargs: Any) -> tuple[Any, Any]:
    """Tracker hinge and CRLB penalties of the worst delay state."""

    hinges, crlbs = zip(*_per_delay_state_torch(
        _tracker_observability_losses_torch, n_states, **kwargs
    ))
    torch_mod = kwargs["torch_mod"]
    return torch_mod.stack(hinges).max(), torch_mod.stack(crlbs).max()


def _tracker_per_state_singular_values_torch(*, n_states: int, **kwargs: Any) -> Any:
    """Tracker singular values per delay state, shape ``(n_states, n_dof)``."""

    return _per_delay_state_singular_values_torch(
        _tracker_observability_singular_values_torch, n_states, **kwargs
    )


def _worst_state_singular_summary(
    per_state: NDArray[np.float64],
    floor: float,
    n_modes: int,
) -> dict[str, Any]:
    """Headline observability values of the worst delay state.

    The worst state is the one with the weakest mode; rank at floor is the
    minimum over states and the floor is met only if every state meets it.
    """

    if per_state.shape[1]:
        values = per_state[int(np.argmin(per_state.min(axis=1)))]
        min_singular = float(np.min(values))
        condition = (
            float(np.max(values) / min_singular) if min_singular > 0.0 else float("inf")
        )
    else:
        values = per_state[0]
        min_singular = 0.0
        condition = float("inf")
    return {
        "singular_values": tuple(float(value) for value in values),
        "singular_values_per_state": tuple(
            tuple(float(value) for value in row) for row in per_state
        ),
        "min_singular": min_singular,
        "condition": condition,
        "rank_at_floor": int(np.min(np.count_nonzero(per_state >= floor, axis=1))),
        "floor_met": bool(per_state.shape[1] >= n_modes and np.all(per_state >= floor)),
    }


def _tracker_observability_loss_torch(
    *,
    u_r: Any,
    u_i: Any,
    field_r_by_wl: Any,
    field_i_by_wl: Any,
    wavelengths_m: Any,
    bright_indices: Any,
    piston_basis: Any,
    min_singular_per_nm: float,
    eps: float,
    torch_mod: Any,
) -> Any:
    """Penalty for bright-port piston modes with too little intensity response.

    The response matrix rows are a count-free Fisher proxy: bright-port
    intensity derivatives divided by the square root of the nominal reference
    power, per nm of physical piston. Its columns are common-piston-free
    physical piston coordinates expanded onto the chip input modes. A log-hinge
    keeps weak singular modes above ``min_singular_per_nm`` without rewarding
    already strong modes.
    """

    singular_values = _tracker_regularized_singular_values_torch(
        u_r=u_r,
        u_i=u_i,
        field_r_by_wl=field_r_by_wl,
        field_i_by_wl=field_i_by_wl,
        wavelengths_m=wavelengths_m,
        bright_indices=bright_indices,
        piston_basis=piston_basis,
        eps=eps,
        torch_mod=torch_mod,
    )
    return _tracker_hinge_loss_from_singular_torch(
        singular_values,
        n_modes=int(piston_basis.shape[1]),
        min_singular_per_nm=min_singular_per_nm,
        eps=eps,
        torch_mod=torch_mod,
    )


def _tracker_observability_crlb_loss_torch(
    *,
    u_r: Any,
    u_i: Any,
    field_r_by_wl: Any,
    field_i_by_wl: Any,
    wavelengths_m: Any,
    bright_indices: Any,
    piston_basis: Any,
    target_rms_nm: float,
    eps: float,
    torch_mod: Any,
) -> Any:
    """CRLB-style penalty for weak bright-port piston Fisher modes."""

    singular_values = _tracker_regularized_singular_values_torch(
        u_r=u_r,
        u_i=u_i,
        field_r_by_wl=field_r_by_wl,
        field_i_by_wl=field_i_by_wl,
        wavelengths_m=wavelengths_m,
        bright_indices=bright_indices,
        piston_basis=piston_basis,
        eps=eps,
        torch_mod=torch_mod,
    )
    return _tracker_crlb_loss_from_singular_torch(
        singular_values,
        n_modes=int(piston_basis.shape[1]),
        target_rms_nm=target_rms_nm,
        eps=eps,
        torch_mod=torch_mod,
    )


def _tracker_observability_singular_values_torch(
    *,
    u_r: Any,
    u_i: Any,
    field_r_by_wl: Any,
    field_i_by_wl: Any,
    wavelengths_m: Any,
    bright_indices: Any,
    piston_basis: Any,
    eps: float,
    torch_mod: Any,
) -> Any:
    """Singular values for the tracker Fisher-proxy response matrix."""

    matrix = _tracker_observability_matrix_torch(
        u_r=u_r,
        u_i=u_i,
        field_r_by_wl=field_r_by_wl,
        field_i_by_wl=field_i_by_wl,
        wavelengths_m=wavelengths_m,
        bright_indices=bright_indices,
        piston_basis=piston_basis,
        eps=eps,
        torch_mod=torch_mod,
    )
    return torch_mod.linalg.svdvals(matrix)


def _regularized_singular_values_torch(
    matrix: Any,
    *,
    eps: float,
    torch_mod: Any,
) -> Any:
    """Differentiable singular values via a regularized Gram matrix.

    Public diagnostics use ``torch.linalg.svdvals`` so reported values remain
    the literal response-matrix singular values. Optimizer losses use this
    helper because the SVD backward pass is fragile near repeated or zero
    singular values; ``eigvalsh(M.T @ M + eps^2 I)`` gives the same values away
    from the numerical floor and avoids zero-mode infinities in gradients.
    """

    gram = matrix.transpose(0, 1) @ matrix
    tiny = torch_mod.tensor(
        float(eps) ** 2,
        device=gram.device,
        dtype=gram.dtype,
    )
    eye = torch_mod.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    evals = torch_mod.linalg.eigvalsh(gram + tiny * eye)
    return torch_mod.sqrt(torch_mod.clamp(evals, min=tiny))


def _tracker_observability_matrix_torch(
    *,
    u_r: Any,
    u_i: Any,
    field_r_by_wl: Any,
    field_i_by_wl: Any,
    wavelengths_m: Any,
    bright_indices: Any,
    piston_basis: Any,
    eps: float,
    torch_mod: Any,
) -> Any:
    """Reference-power-weighted bright-port piston response per nm."""

    u_b_r = u_r.index_select(1, bright_indices)
    u_b_i = u_i.index_select(1, bright_indices)

    out_r = torch_mod.einsum("wbm,wmp->wbp", u_b_r, field_r_by_wl) - torch_mod.einsum(
        "wbm,wmp->wbp", u_b_i, field_i_by_wl
    )
    out_i = torch_mod.einsum("wbm,wmp->wbp", u_b_r, field_i_by_wl) + torch_mod.einsum(
        "wbm,wmp->wbp", u_b_i, field_r_by_wl
    )

    phase_per_m = (2.0 * math.pi / wavelengths_m).view(-1, 1, 1, 1)
    basis = piston_basis.view(1, piston_basis.shape[0], 1, piston_basis.shape[1])
    d_field_r = -field_i_by_wl.unsqueeze(3) * phase_per_m * basis
    d_field_i = field_r_by_wl.unsqueeze(3) * phase_per_m * basis

    d_out_r = torch_mod.einsum("wbm,wmpc->wbpc", u_b_r, d_field_r) - torch_mod.einsum(
        "wbm,wmpc->wbpc", u_b_i, d_field_i
    )
    d_out_i = torch_mod.einsum("wbm,wmpc->wbpc", u_b_r, d_field_i) + torch_mod.einsum(
        "wbm,wmpc->wbpc", u_b_i, d_field_r
    )

    response_power_per_m = 2.0 * torch_mod.sum(
        out_r.unsqueeze(3) * d_out_r + out_i.unsqueeze(3) * d_out_i,
        dim=2,
    )
    reference_power = torch_mod.sum(out_r**2 + out_i**2, dim=2)
    tiny = torch_mod.tensor(
        float(eps),
        device=reference_power.device,
        dtype=reference_power.dtype,
    )
    response_per_nm = response_power_per_m / torch_mod.sqrt(
        torch_mod.clamp(
            reference_power.unsqueeze(2),
            min=tiny,
        )
    )
    response_per_nm = response_per_nm * 1.0e-9
    return response_per_nm.reshape(-1, piston_basis.shape[1])


def _tracker_crlb_rms_nm_from_singular_values(
    singular_values_per_nm: NDArray[np.float64],
    *,
    eps: float,
) -> float:
    """Return sqrt(mean(1/s_i^2)) for singular values in 1/nm."""

    values = np.asarray(singular_values_per_nm, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return float("inf")
    clipped = np.maximum(values, float(eps))
    return float(np.sqrt(np.mean(1.0 / (clipped * clipped))))


def _tracker_rank_priority_selection_key(
    *,
    singular_values_per_nm: NDArray[np.float64],
    n_dof: int,
    min_singular_per_nm: float,
    crlb_target_nm: float,
    objective_loss: float,
    normalized_null: float,
    target_null_depth: float,
    eps: float,
) -> tuple[float, ...]:
    """Rank restart candidates with tracker observability as a constraint.

    ``singular_values_per_nm`` is ``(n_dof,)`` for one pointing or
    ``(n_states, n_dof)`` per delay state. Rank and CRLB are judged on the
    worst state, since the tracker sees one pointing at a time (Issue 21).
    """

    singular_values = np.atleast_2d(
        np.asarray(singular_values_per_nm, dtype=np.float64)
    )
    if not np.all(np.isfinite(singular_values)):
        return (
            float(n_dof),
            1.0,
            float("inf"),
            float("inf"),
            float(objective_loss),
            float("inf"),
            float("inf"),
        )
    rank_at_floor = int(
        np.min(
            np.count_nonzero(singular_values >= float(min_singular_per_nm), axis=1)
        )
    )
    rank_deficit = max(int(n_dof) - rank_at_floor, 0)
    null_miss = 0.0 if float(normalized_null) <= float(target_null_depth) else 1.0
    crlb_rms_nm = max(
        _tracker_crlb_rms_nm_from_singular_values(values, eps=float(eps))
        for values in singular_values
    )
    crlb_excess = max(crlb_rms_nm / float(crlb_target_nm) - 1.0, 0.0)
    crlb_miss = 1.0 if crlb_excess > 0.0 else 0.0
    return (
        float(rank_deficit),
        float(null_miss),
        float(crlb_miss),
        float(crlb_excess),
        # Once the candidate has enough rank and meets the CRLB target, let the
        # science objective choose the basin. The objective already carries the
        # leakage-vs-throughput trade.
        float(objective_loss),
        float(normalized_null),
        float(crlb_rms_nm),
    )


def _amplitude_observability_loss_torch(
    *,
    u_r: Any,
    u_i: Any,
    field_r_by_wl: Any,
    field_i_by_wl: Any,
    port_indices: Any,
    amplitude_basis: Any,
    min_singular_per_fractional_intensity: float,
    eps: float,
    torch_mod: Any,
) -> Any:
    """Hinge penalty for weak selected-port amplitude Fisher modes."""

    matrix = _amplitude_observability_matrix_torch(
        u_r=u_r,
        u_i=u_i,
        field_r_by_wl=field_r_by_wl,
        field_i_by_wl=field_i_by_wl,
        port_indices=port_indices,
        amplitude_basis=amplitude_basis,
        eps=eps,
        torch_mod=torch_mod,
    )
    singular_values = _regularized_singular_values_torch(
        matrix,
        eps=eps,
        torch_mod=torch_mod,
    )
    target = torch_mod.tensor(
        float(min_singular_per_fractional_intensity),
        device=singular_values.device,
        dtype=singular_values.dtype,
    )
    deficit = torch_mod.relu(target - singular_values) / target
    n_modes = int(amplitude_basis.shape[1])
    missing = n_modes - int(singular_values.numel())
    if missing > 0:
        deficit = torch_mod.cat(
            (
                deficit,
                torch_mod.ones(
                    (missing,),
                    device=deficit.device,
                    dtype=deficit.dtype,
                ),
            )
        )
    return torch_mod.mean(deficit**2)


def _amplitude_observability_singular_values_torch(
    *,
    u_r: Any,
    u_i: Any,
    field_r_by_wl: Any,
    field_i_by_wl: Any,
    port_indices: Any,
    amplitude_basis: Any,
    eps: float,
    torch_mod: Any,
) -> Any:
    """Singular values for selected-port differential-amplitude response."""

    matrix = _amplitude_observability_matrix_torch(
        u_r=u_r,
        u_i=u_i,
        field_r_by_wl=field_r_by_wl,
        field_i_by_wl=field_i_by_wl,
        port_indices=port_indices,
        amplitude_basis=amplitude_basis,
        eps=eps,
        torch_mod=torch_mod,
    )
    return torch_mod.linalg.svdvals(matrix)


def _amplitude_observability_matrix_torch(
    *,
    u_r: Any,
    u_i: Any,
    field_r_by_wl: Any,
    field_i_by_wl: Any,
    port_indices: Any,
    amplitude_basis: Any,
    eps: float,
    torch_mod: Any,
) -> Any:
    """Reference-power-weighted selected-port amplitude response.

    Coordinates are differential fractional-intensity modes. A coordinate
    changes field amplitude by 0.5 * delta_I because optical power is the
    square of field amplitude.
    """

    u_sel_r = u_r.index_select(1, port_indices)
    u_sel_i = u_i.index_select(1, port_indices)
    out_r = torch_mod.einsum("wom,wmp->wop", u_sel_r, field_r_by_wl) - (
        torch_mod.einsum("wom,wmp->wop", u_sel_i, field_i_by_wl)
    )
    out_i = torch_mod.einsum("wom,wmp->wop", u_sel_r, field_i_by_wl) + (
        torch_mod.einsum("wom,wmp->wop", u_sel_i, field_r_by_wl)
    )

    basis = amplitude_basis.view(
        1,
        amplitude_basis.shape[0],
        1,
        amplitude_basis.shape[1],
    )
    d_field_r = 0.5 * field_r_by_wl.unsqueeze(3) * basis
    d_field_i = 0.5 * field_i_by_wl.unsqueeze(3) * basis

    d_out_r = torch_mod.einsum("wom,wmpc->wopc", u_sel_r, d_field_r) - (
        torch_mod.einsum("wom,wmpc->wopc", u_sel_i, d_field_i)
    )
    d_out_i = torch_mod.einsum("wom,wmpc->wopc", u_sel_r, d_field_i) + (
        torch_mod.einsum("wom,wmpc->wopc", u_sel_i, d_field_r)
    )

    response_per_fraction = 2.0 * torch_mod.sum(
        out_r.unsqueeze(3) * d_out_r + out_i.unsqueeze(3) * d_out_i,
        dim=2,
    )
    reference_power = torch_mod.sum(out_r**2 + out_i**2, dim=2)
    tiny = torch_mod.tensor(
        float(eps),
        device=reference_power.device,
        dtype=reference_power.dtype,
    )
    response_per_fraction = response_per_fraction / torch_mod.sqrt(
        torch_mod.clamp(reference_power.unsqueeze(2), min=tiny)
    )
    return response_per_fraction.reshape(-1, amplitude_basis.shape[1])


def _joint_phase_amplitude_observability_loss_torch(
    *,
    u_r: Any,
    u_i: Any,
    field_r_by_wl: Any,
    field_i_by_wl: Any,
    wavelengths_m: Any,
    port_indices: Any,
    piston_basis: Any,
    amplitude_basis: Any,
    amplitude_scale: float,
    min_singular_per_scaled_mode: float,
    eps: float,
    torch_mod: Any,
) -> Any:
    """Hinge penalty for weak joint piston+amplitude Fisher modes."""

    matrix = _joint_phase_amplitude_observability_matrix_torch(
        u_r=u_r,
        u_i=u_i,
        field_r_by_wl=field_r_by_wl,
        field_i_by_wl=field_i_by_wl,
        wavelengths_m=wavelengths_m,
        port_indices=port_indices,
        piston_basis=piston_basis,
        amplitude_basis=amplitude_basis,
        amplitude_scale=amplitude_scale,
        eps=eps,
        torch_mod=torch_mod,
    )
    singular_values = _regularized_singular_values_torch(
        matrix,
        eps=eps,
        torch_mod=torch_mod,
    )
    target = torch_mod.tensor(
        float(min_singular_per_scaled_mode),
        device=singular_values.device,
        dtype=singular_values.dtype,
    )
    deficit = torch_mod.relu(target - singular_values) / target
    n_modes = int(piston_basis.shape[1]) + int(amplitude_basis.shape[1])
    missing = n_modes - int(singular_values.numel())
    if missing > 0:
        deficit = torch_mod.cat(
            (
                deficit,
                torch_mod.ones(
                    (missing,),
                    device=deficit.device,
                    dtype=deficit.dtype,
                ),
            )
        )
    return torch_mod.mean(deficit**2)


def _joint_phase_amplitude_observability_crlb_loss_torch(
    *,
    u_r: Any,
    u_i: Any,
    field_r_by_wl: Any,
    field_i_by_wl: Any,
    wavelengths_m: Any,
    port_indices: Any,
    piston_basis: Any,
    amplitude_basis: Any,
    amplitude_scale: float,
    target_per_scaled_mode: float,
    eps: float,
    torch_mod: Any,
) -> Any:
    """CRLB-style penalty for weak joint phase+amplitude Fisher modes."""

    matrix = _joint_phase_amplitude_observability_matrix_torch(
        u_r=u_r,
        u_i=u_i,
        field_r_by_wl=field_r_by_wl,
        field_i_by_wl=field_i_by_wl,
        wavelengths_m=wavelengths_m,
        port_indices=port_indices,
        piston_basis=piston_basis,
        amplitude_basis=amplitude_basis,
        amplitude_scale=amplitude_scale,
        eps=eps,
        torch_mod=torch_mod,
    )
    singular_values = _regularized_singular_values_torch(
        matrix,
        eps=eps,
        torch_mod=torch_mod,
    )
    tiny = torch_mod.tensor(
        float(eps),
        device=singular_values.device,
        dtype=singular_values.dtype,
    )
    target = torch_mod.tensor(
        float(target_per_scaled_mode),
        device=singular_values.device,
        dtype=singular_values.dtype,
    )
    std = 1.0 / torch_mod.clamp(singular_values, min=tiny)
    n_modes = int(piston_basis.shape[1]) + int(amplitude_basis.shape[1])
    missing = n_modes - int(singular_values.numel())
    if missing > 0:
        missing_std = torch_mod.full(
            (missing,),
            1.0 / float(eps),
            device=std.device,
            dtype=std.dtype,
        )
        std = torch_mod.cat((std, missing_std))
    return torch_mod.mean((std / target) ** 2)


def _joint_phase_amplitude_observability_singular_values_torch(
    *,
    u_r: Any,
    u_i: Any,
    field_r_by_wl: Any,
    field_i_by_wl: Any,
    wavelengths_m: Any,
    port_indices: Any,
    piston_basis: Any,
    amplitude_basis: Any,
    amplitude_scale: float,
    eps: float,
    torch_mod: Any,
) -> Any:
    """Singular values for the joint selected-port phase+amplitude proxy."""

    matrix = _joint_phase_amplitude_observability_matrix_torch(
        u_r=u_r,
        u_i=u_i,
        field_r_by_wl=field_r_by_wl,
        field_i_by_wl=field_i_by_wl,
        wavelengths_m=wavelengths_m,
        port_indices=port_indices,
        piston_basis=piston_basis,
        amplitude_basis=amplitude_basis,
        amplitude_scale=amplitude_scale,
        eps=eps,
        torch_mod=torch_mod,
    )
    return torch_mod.linalg.svdvals(matrix)


def _joint_phase_amplitude_observability_matrix_torch(
    *,
    u_r: Any,
    u_i: Any,
    field_r_by_wl: Any,
    field_i_by_wl: Any,
    wavelengths_m: Any,
    port_indices: Any,
    piston_basis: Any,
    amplitude_basis: Any,
    amplitude_scale: float,
    eps: float,
    torch_mod: Any,
) -> Any:
    """Reference-power-weighted matrix for mixed piston and amplitude modes."""

    piston_matrix = _tracker_observability_matrix_torch(
        u_r=u_r,
        u_i=u_i,
        field_r_by_wl=field_r_by_wl,
        field_i_by_wl=field_i_by_wl,
        wavelengths_m=wavelengths_m,
        bright_indices=port_indices,
        piston_basis=piston_basis,
        eps=eps,
        torch_mod=torch_mod,
    )
    amplitude_matrix = _amplitude_observability_matrix_torch(
        u_r=u_r,
        u_i=u_i,
        field_r_by_wl=field_r_by_wl,
        field_i_by_wl=field_i_by_wl,
        port_indices=port_indices,
        amplitude_basis=amplitude_basis,
        eps=eps,
        torch_mod=torch_mod,
    )
    scale = torch_mod.tensor(
        float(amplitude_scale),
        device=amplitude_matrix.device,
        dtype=amplitude_matrix.dtype,
    )
    return torch_mod.cat((piston_matrix, amplitude_matrix * scale), dim=1)


def _science_null_guard_loss_torch(
    *,
    leakage_loss: Any,
    target_loss: float,
    guard_factor: float,
    eps: float,
    torch_mod: Any,
) -> Any:
    """Quadratic penalty for leaving the target science-null basin."""

    target = torch_mod.tensor(
        float(target_loss) * float(guard_factor),
        device=leakage_loss.device,
        dtype=leakage_loss.dtype,
    )
    tiny = torch_mod.tensor(
        float(eps),
        device=leakage_loss.device,
        dtype=leakage_loss.dtype,
    )
    return (
        torch_mod.relu(leakage_loss - target) / torch_mod.clamp(target, min=tiny)
    ) ** 2


def _amplitude_mismatch_loss_torch(
    *,
    u_r: Any,
    u_i: Any,
    field_r_by_wl: Any,
    field_i_by_wl: Any,
    null_indices: Any,
    dump_indices: Any,
    amplitude_basis: Any,
    min_dump_fraction: float,
    probe_sigma_i: float,
    eps: float,
    torch_mod: Any,
) -> Any:
    """Penalty for amplitude-perturbed dark fraction and dump routing."""

    null_fraction, dump_fraction = _amplitude_mismatch_fractions_torch(
        u_r=u_r,
        u_i=u_i,
        field_r_by_wl=field_r_by_wl,
        field_i_by_wl=field_i_by_wl,
        null_indices=null_indices,
        dump_indices=dump_indices,
        amplitude_basis=amplitude_basis,
        probe_sigma_i=probe_sigma_i,
        eps=eps,
        torch_mod=torch_mod,
    )
    probe_var = torch_mod.tensor(
        float(probe_sigma_i) ** 2,
        device=null_fraction.device,
        dtype=null_fraction.dtype,
    )
    tiny = torch_mod.tensor(
        float(eps),
        device=null_fraction.device,
        dtype=null_fraction.dtype,
    )
    loss = torch_mod.mean(null_fraction) / torch_mod.clamp(probe_var, min=tiny)
    if min_dump_fraction > 0.0:
        target = torch_mod.tensor(
            float(min_dump_fraction),
            device=dump_fraction.device,
            dtype=dump_fraction.dtype,
        )
        loss = loss + torch_mod.mean(torch_mod.relu(target - dump_fraction) ** 2)
    return loss


def _amplitude_mismatch_fractions_torch(
    *,
    u_r: Any,
    u_i: Any,
    field_r_by_wl: Any,
    field_i_by_wl: Any,
    null_indices: Any,
    dump_indices: Any,
    amplitude_basis: Any,
    probe_sigma_i: float,
    eps: float,
    torch_mod: Any,
) -> tuple[Any, Any]:
    """Return dark fractions and dump fractions for amplitude perturbations.

    Coordinates are physical differential fractional-intensity modes. The dark
    fraction is the average of positive and negative perturbations about the
    nominal stellar field, using ``field * exp(0.5 * delta_I)`` so intensity
    stays positive. The dump response remains the linear perturbation-power
    fraction so ``amplitude_dump_min_fraction`` stays a routing fraction.
    """

    basis = amplitude_basis.view(
        1,
        amplitude_basis.shape[0],
        1,
        amplitude_basis.shape[1],
    )
    tiny = torch_mod.tensor(
        float(eps),
        device=field_r_by_wl.device,
        dtype=field_r_by_wl.dtype,
    )
    probe = torch_mod.tensor(
        float(probe_sigma_i),
        device=field_r_by_wl.device,
        dtype=field_r_by_wl.dtype,
    )
    u_null_r = u_r.index_select(1, null_indices)
    u_null_i = u_i.index_select(1, null_indices)

    def _dark_fraction_for_scale(scale: Any) -> Any:
        perturbed_r = field_r_by_wl.unsqueeze(3) * scale
        perturbed_i = field_i_by_wl.unsqueeze(3) * scale
        null_r = torch_mod.einsum("won,wnpc->wopc", u_null_r, perturbed_r) - (
            torch_mod.einsum("won,wnpc->wopc", u_null_i, perturbed_i)
        )
        null_i = torch_mod.einsum("won,wnpc->wopc", u_null_r, perturbed_i) + (
            torch_mod.einsum("won,wnpc->wopc", u_null_i, perturbed_r)
        )
        dark_power = torch_mod.sum(null_r**2 + null_i**2, dim=(1, 2))
        total_power = torch_mod.sum(perturbed_r**2 + perturbed_i**2, dim=(1, 2))
        return dark_power / torch_mod.clamp(total_power, min=tiny)

    delta_i = probe * basis
    null_fraction = 0.5 * (
        _dark_fraction_for_scale(torch_mod.exp(0.5 * delta_i))
        + _dark_fraction_for_scale(torch_mod.exp(-0.5 * delta_i))
    )

    d_field_r = 0.5 * field_r_by_wl.unsqueeze(3) * basis
    d_field_i = 0.5 * field_i_by_wl.unsqueeze(3) * basis

    u_dump_r = u_r.index_select(1, dump_indices)
    u_dump_i = u_i.index_select(1, dump_indices)
    dump_r = torch_mod.einsum("won,wnpc->wopc", u_dump_r, d_field_r) - (
        torch_mod.einsum("won,wnpc->wopc", u_dump_i, d_field_i)
    )
    dump_i = torch_mod.einsum("won,wnpc->wopc", u_dump_r, d_field_i) + (
        torch_mod.einsum("won,wnpc->wopc", u_dump_i, d_field_r)
    )
    dump_power = torch_mod.sum(dump_r**2 + dump_i**2, dim=(1, 2))

    perturbation_power = torch_mod.sum(d_field_r**2 + d_field_i**2, dim=(1, 2))
    perturbation_power = torch_mod.clamp(perturbation_power, min=tiny)
    return null_fraction, dump_power / perturbation_power


def _reduce_spectral_leakage_torch(
    leak_per_wl: Any,
    params: ChipOptimizationParams,
    *,
    spectral_weights: Any | None = None,
    wavelength_grid_um: tuple[float, ...] | NDArray[np.floating] | None = None,
    torch_mod: Any,
) -> Any:
    """Torch equivalent of the shared spectral leakage reducer."""

    if params.spectral_loss_mode in {"off", "mean"}:
        return torch_mod.mean(leak_per_wl)
    if params.spectral_loss_mode == "p_norm":
        max_leak = torch_mod.max(leak_per_wl)
        scale = torch_mod.clamp(
            max_leak,
            min=torch_mod.finfo(leak_per_wl.dtype).tiny,
        )
        p = float(params.spectral_loss_p)
        return max_leak * torch_mod.mean(
            torch_mod.clamp(leak_per_wl / scale, min=0.0) ** p
        ) ** (1.0 / p)
    if params.spectral_loss_mode == "edge_weighted_mean":
        weights = spectral_weights
        if weights is None:
            if wavelength_grid_um is None:
                raise ValueError(
                    "edge_weighted_mean spectral loss requires "
                    "wavelength_grid_um or spectral_weights."
                )
            weights_np = _edge_weighted_spectral_weights_numpy(
                wavelength_grid_um,
                params,
            )
            weights = torch_mod.from_numpy(
                np.ascontiguousarray(weights_np, dtype=np.float64),
            ).to(leak_per_wl.device)
        return torch_mod.sum(weights * leak_per_wl) / torch_mod.sum(weights)
    raise ValueError(f"Unknown spectral_loss_mode {params.spectral_loss_mode!r}.")
