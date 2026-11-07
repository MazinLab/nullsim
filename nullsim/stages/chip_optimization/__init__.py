"""Chip-optimization stages: scipy / torch BFGS-based null-mesh solvers.

SPEC §15.3 Pass 3 WP3. Each backend produces the same payload shape under
``state.results.components`` so downstream stages can swap
backends without code changes.

The shared ``Params`` model below captures the inputs common to all three
backends (mesh sizing, dark-port count, design wavelength, target null
depth, iteration cap, DAC bit depth, optional seed). Per the WP1 notes
(``tests/stages/notes/chip_optimization.md``), each backend keeps its own
non-trivial conventions:

* scipy: numpy float64 throughout, scipy ``minimize`` with method
  ``L-BFGS-B`` (and optional ``SLSQP`` polish), 2-point finite-difference
  gradient.
* torch: float64, real/imag split unitary, TF32 disabled at import time
  (the KILO hard-won lesson is recorded inline).

The optional torch backend imports torch lazily, so missing the extra
(``pip install nullsim[torch]``) does not break the package import path.
The lazy import raises a clear ``ImportError`` pointing at the right
extra when the stage is actually instantiated.

The default loss minimized in both backends is the broadband mean dark-port
leakage. Configs can opt into smooth spectral reducers to weight the
worst wavelength bins or band edges more strongly:

    L(params) = (1 / (n_wl * n_pol * total_input_power))
                  * sum_{wl, pol} sum_{k in dark_ports}
                      |(U(wl) @ field(:, wl, pol))_k|^2

where the dark-port subset is the last ``n_dark_ports`` rows of the
output and ``U(wl)`` is built by ``nullsim.physics.clements`` with the
KILO ``phi``+``alpha`` wavelength rescaling rule (notes §1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator
from pydantic import Field as PydanticField

from nullsim.physics.clements import ClementsDecomposition, reconstruct_clements
from nullsim.physics.fringe_observability import differential_piston_basis
from nullsim.physics.planet_throughput import planet_input_field
from nullsim.physics.ruilier_cassaing import angular_coupling_amplitude
from nullsim.pipeline import SimulationState
from nullsim.stages._mode_splitter import (
    component_mode_areas_m2 as _component_mode_areas_m2,
)
from nullsim.stages._mode_splitter import (
    component_mode_diameters_m as _component_mode_diameters_m,
)
from nullsim.stages._mode_splitter import (
    splitter_physical_indices as _splitter_physical_indices,
)
from nullsim.stages._utils import (
    torch_compile_env_enabled as _torch_compile_env_enabled,
)
from nullsim.stages._utils import unique_component_name

_RAD_PER_MAS = 1.0 / 206265e3
_DEFAULT_PHASE_NEFF_REF = 2.4
_DEFAULT_PHASE_GROUP_INDEX = 2.4


class ChipOptimizationParams(BaseModel):
    """Validated params shared by all three chip-optimization backends.

    Per WP3 spec (SPEC §15.3 Pass 3 WP3): bright + dark port counts must
    sum to ``n_modes``; per-wavelength rescaling and DAC quantization
    follow the same conventions as ``chip.kernel_mzi_mesh``.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    n_modes: int = PydanticField(gt=0)
    n_bright_ports: int = PydanticField(ge=0)
    n_dark_ports: int = PydanticField(ge=1)
    # KILO_PARITY_PLAN §3.1: number of spectroscopy ports. KILO
    # ``assign_ports`` (kilo/chip.py:2049) lays the M chip outputs out
    # as ``[bright | dark | spectro]`` with ``n_bright + n_dark +
    # n_spectro == M``. The optimizer's NULL CONSTRAINT SET is
    # ``concat(dark_idx, spectro_idx)`` — the spectro ports are pulled
    # to zero stellar leakage alongside the dark ports — but the
    # PLANET-THROUGHPUT MAXIMIZATION penalty sums over JUST the dark
    # indices (KILO natural_design.py L643:
    # ``throughput_port_indices_eff = dark_idx``). nullsim mirrors that
    # split via ``_null_indices`` (dark + spectro) and ``_dark_indices``
    # (dark only). Default 0 reproduces the pre-Phase-3.1 layout where
    # there is no spectro port at all (n_bright + n_dark = n_modes).
    n_spectro_ports: int = PydanticField(default=0, ge=0)
    wl_design_um: float = PydanticField(gt=0.0)
    target_null_depth: float = PydanticField(default=1e-6, gt=0.0)
    # KILO paperv3 kernel_null kernel_order=3 broadband uses maxiter=5000
    # (kilo/studies/natural_design.py L433). 500 is much too few for a
    # 576-param kernel-null BFGS and forces every kernel-null config to
    # override the default.
    max_iter: int = PydanticField(default=5000, ge=1)
    seed: int | None = None
    dac_bits: int | None = None
    capture_convergence_history: bool = False
    # torch backend only: wrap the autograd objective in torch.compile
    # (mode='reduce-overhead') for ~15x per-grad-eval speedup. Composes
    # with the NULLSIM_TORCH_COMPILE env var (either source enables it).
    # One-time tracing cost ~30-55s — only worth it for runs with
    # >2000 BFGS iterations. KILO chip_torch.py L352 convention.
    compile_objective: bool = False
    # Torch backend only: "auto" picks CUDA when available and CPU otherwise.
    # Explicit "cuda" raises if CUDA is unavailable; explicit "cpu" keeps
    # runs on host memory. MPS is intentionally not accepted because the
    # null-depth objective uses float64 throughout and PyTorch MPS does
    # not support that dtype reliably.
    torch_device: str = "auto"
    # Kernel-null order (KILO injection.py L226-330 conventions). 0 =
    # null only the on-axis field (current default; back-compat).
    # 2 = G(r) ~ r^4 near origin, 3 = G(r) ~ r^6, 4 = G(r) ~ r^8.
    # Higher orders need n_apertures(M) sub-aperture positions on
    # state.geometry.aperture_positions_pupil_m — the optimizer raises
    # a clear error if kernel_order > 0 and positions aren't populated.
    kernel_order: int = PydanticField(default=0, ge=0, le=4)
    # Optional resolved-stellar-disk null constraints. These mirror the
    # characterization stages' disk-sampling controls, but default off for
    # back-compat because they add many rows to the optimizer objective.
    disk_n_rings: int = PydanticField(default=0, ge=0)
    disk_n_azimuths: int = PydanticField(default=0, ge=0)
    # Scales disk-sample row amplitudes by sqrt(disk_weight), so the objective
    # contribution scales linearly with disk_weight. 0 disables disk rows.
    disk_weight: float = PydanticField(default=1.0, ge=0.0)
    # Restart-loop controls (KILO chip.py L1100-1160 conventions). The 5
    # default matches KILO's optimize_null default; KILO's empirical
    # comment: "the deepest basin is found in the first 1-5 restarts at
    # M=24 with maxiter>=5000; the remaining 19 in a 24-restart budget
    # are usually wasted exploring shallower basins."
    n_restarts: int = PydanticField(default=5, ge=1)
    # Stop the restart loop after this many consecutive restarts produce
    # < restart_patience_reltol improvement over the running best.
    # Set to 0 to disable. KILO ``KILO_BB_PATIENCE`` env var defaults
    # to 0 (chip.py L1158), so KILO production also runs with patience
    # disabled. An earlier nullsim default of 3 fired premature bail-
    # outs on ~25% of broadband restart loops at the default
    # ``restart_patience_reltol=0.01``. Keep the param exposed so
    # configs can opt back in to early termination.
    restart_patience: int = PydanticField(default=0, ge=0)
    restart_patience_reltol: float = PydanticField(default=0.01, gt=0.0)
    # Chip-restart process-pool fan-out. Composes with [run].n_workers
    # (outer sweep-cell pool). Total cores in flight ~= outer * inner.
    # Default 1 = sequential (current behavior).
    n_workers: int = PydanticField(default=1, ge=1)
    # Single-lambda bootstrap (KILO_BB_SINGLE_LAMBDA_BOOTSTRAP=1
    # convention, kilo/studies/natural_design.py L555-585). At M=24 with
    # kernel-null rows, BFGS-from-Haar-init lands in shallow basins
    # (~1e-3 per-row mean leakage at best). Running a single-wavelength
    # optimize_null first finds the deep on-axis basin (target null
    # ~1e-7), then using those params as the warm-start for the first
    # restart of the broadband solve lets BFGS refine for band edges
    # instead of having to find the deep basin from random init.
    # Default True since the paperv3 detection_space curves we replicate
    # were produced with KILO_BB_SINGLE_LAMBDA_BOOTSTRAP=1.
    single_lambda_bootstrap: bool = True
    # Iterations and restarts for the bootstrap single-wavelength solve.
    # KILO uses n_restarts=1 for the single-lambda bootstrap
    # (natural_design.py L577), relying on the warm-start to land the
    # deepest basin at the design wavelength. nullsim defaults to 5
    # because the M=24/order=3 kernel-null basin is multi-modal and a
    # single cold start reaches the target null only ~10% of the time.
    # maxiter=2000 is enough to converge each single-wavelength restart.
    bootstrap_max_iter: int = PydanticField(default=2000, ge=1)
    bootstrap_n_restarts: int = PydanticField(default=5, ge=1)
    # When the kernel deploys monochromatically (torch backend's
    # canonicalized params, kernel_rescales_per_wavelength=False), the
    # broadband restart loop optimizes a per-wavelength rescaled
    # objective that does NOT match what the kernel actually applies.
    # Empirically the broadband refine drifts AWAY from the bootstrap's
    # deep design-wavelength basin (chip_ideal floor 2.7e-4 vs the
    # bootstrap's reachable 1e-5 at M=24, kernel_order=3). When False,
    # skip the broadband restart loop entirely and use the bootstrap
    # warm-start as the final params. Only valid when single_lambda_
    # bootstrap is True; raises at run if bootstrap is disabled.
    # Default True preserves existing behaviour for the scipy backend.
    broadband_refine: bool = True
    # Spectral reduction for the broadband dark-port leakage. ``off`` keeps
    # the historical KILO/nullsim mean objective. ``mean`` is accepted as a
    # backward-compatible alias. ``p_norm`` emphasizes the worst wavelength
    # bin while staying smooth enough for BFGS; larger p is closer to
    # max(leak_per_wl). ``edge_weighted_mean`` keeps a mean-like objective
    # but weights wavelengths farther from wl_design_um more strongly:
    # w = 1 + alpha * (|wl - wl_design| / max_edge_distance) ** power.
    spectral_loss_mode: Literal[
        "off",
        "mean",
        "p_norm",
        "edge_weighted_mean",
    ] = "off"
    spectral_loss_p: float = PydanticField(default=8.0, ge=1.0)
    spectral_edge_weight_alpha: float = PydanticField(default=4.0, ge=0.0)
    spectral_edge_weight_power: float = PydanticField(default=2.0, gt=0.0)
    # KILO uses 5 wavelengths for the broadband chip optimization
    # (kilo/studies/natural_design.py L474, ``KILO_BB_N_WL`` default 5) while
    # nullsim's input field grid often has 20+ bins for detector
    # photon-rate accuracy. Sub-sampling the optimization to a coarser
    # grid keeps the per-iteration cost down (each restart's inner loop
    # is O(n_wl_optimize)) and the broadband optimum is essentially
    # unchanged because the chip's wavelength response is smooth.
    # Default 0 means "use full grid" (back-compat); set to e.g. 5 to
    # match KILO. Evaluated at uniformly-spaced indices across the
    # input grid (closest indices to the linspace targets).
    n_wl_optimize: int = PydanticField(default=0, ge=0)
    # Optional delay-line robustness term. When >0 and a prior
    # ``delay_lines.geometric`` stage left per-aperture diagnostics, the
    # optimizer reconstructs the pre-delay stellar field and replays the
    # delay-line phase/loss at uniformly spaced azimuths. The mesh settings
    # remain one shared physical parameter vector; these rows only add
    # representative geometric-delay states to the broadband objective.
    delay_state_n_azimuths: int = PydanticField(default=0, ge=0)
    # KILO_BB_THROUGHPUT_PENALTY_ALPHA convention (kilo/studies/natural_design.py
    # L590-625). The chip optimizer's null-only objective drives chip leakage
    # toward zero at the cost of planet throughput at all separations -- the
    # deeper-null basin sharpens the kernel-null modulation and collapses the
    # signal at small separations (empirically: chip_ideal 4e-6 → planet
    # throughput at 30 mas ~0.006, vs KILO's null × throughput product 2.5e-5
    # with chip 7e-5 + throughput 0.35). Adding a soft penalty
    # ``-alpha × planet_throughput_at_rep_sep`` to the bootstrap objective
    # reshapes the loss surface to favor basins that preserve the kernel
    # null's r^6 throughput rolloff. KILO default 1e-3; 0 disables.
    throughput_penalty_alpha: float = PydanticField(default=0.0, ge=0.0)
    # Representative separations (in mas) for the throughput penalty PA
    # average. KILO default 3 mas (a single sep); can be multi-valued to
    # anchor across the photon-noise-limited inner working angle. Each
    # separation gets weighted equally in the penalty term.
    throughput_penalty_seps_mas: tuple[float, ...] = PydanticField(
        default=(3.0,)
    )
    # PA samples per separation (KILO default 8).
    throughput_penalty_n_pa: int = PydanticField(default=8, ge=1)
    # Torch backend only: soft regularizer that preserves the bright-port
    # small-signal piston Fisher subspace while optimizing the dark/kernel rows.
    # The singular values are computed from a reference-power-weighted bright-port
    # intensity-response proxy per nm of physical piston. 0 disables the term.
    tracker_observability_weight: float = PydanticField(default=0.0, ge=0.0)
    tracker_observability_min_singular_per_nm: float = PydanticField(
        default=1.0e-3,
        gt=0.0,
    )
    # CRLB-style tracker regularizer. The singular values are in 1/nm, so
    # sqrt(mean(1/s_i^2)) is the count-free piston RMS implied by the bright
    # ports. This complements the floor hinge above: the hinge prevents
    # dropped modes; the CRLB term keeps all modes well measured.
    tracker_observability_crlb_weight: float = PydanticField(default=0.0, ge=0.0)
    tracker_observability_crlb_target_nm: float = PydanticField(
        default=5.0,
        gt=0.0,
    )
    # When enabled, restart selection first prefers candidates with fewer
    # tracker modes below the singular-value floor, then lower CRLB loss,
    # then the usual scalar objective. This prevents a good science-null
    # basin from silently shipping an under-ranked fringe-tracker subspace.
    tracker_observability_rank_priority: bool = False
    # Number of independent tracker count channels to enforce in the regularizer.
    # 0 uses every optimizer wavelength; 1 enforces single-channel sensitivity.
    tracker_observability_n_spectral_channels: int = PydanticField(
        default=0,
        ge=0,
    )
    tracker_observability_eps: float = PydanticField(default=1.0e-12, gt=0.0)
    # Torch backend only: optional regularizer for scintillation / differential
    # amplitude robustness. It minimizes the first-order field response of
    # null ports to physical fractional-intensity modes and can require a
    # selected bright port to act as an amplitude dump. The dump index is in
    # the bright-port block; None means the last bright port.
    amplitude_mismatch_weight: float = PydanticField(default=0.0, ge=0.0)
    amplitude_mismatch_n_spectral_channels: int = PydanticField(default=0, ge=0)
    amplitude_mismatch_probe_sigma_i: float = PydanticField(default=0.02, gt=0.0)
    amplitude_dump_port_index: int | None = PydanticField(default=None, ge=0)
    amplitude_dump_min_fraction: float = PydanticField(default=0.0, ge=0.0, le=1.0)
    amplitude_mismatch_eps: float = PydanticField(default=1.0e-12, gt=0.0)
    # Torch backend only: optional regularizer for amplitude observability.
    # This is complementary to amplitude_mismatch_weight. Instead of forcing
    # dark ports to be insensitive to amplitude perturbations, it rewards an
    # invertible selected-port response matrix so post-processing or a future
    # servo can estimate differential photometric errors.
    amplitude_observability_weight: float = PydanticField(default=0.0, ge=0.0)
    amplitude_observability_min_singular_per_fractional_intensity: float = (
        PydanticField(default=1.0e-2, gt=0.0)
    )
    amplitude_observability_port_set: Literal["bright", "bright_dark", "all"] = (
        "bright_dark"
    )
    amplitude_observability_n_spectral_channels: int = PydanticField(default=0, ge=0)
    amplitude_observability_eps: float = PydanticField(default=1.0e-12, gt=0.0)
    # Torch backend only: joint small-signal Fisher regularizer for phase and
    # amplitude observability. Rows use the same reference-power Poisson proxy
    # as the separate tracker/amplitude terms; columns combine piston-per-nm
    # coordinates with differential fractional-intensity coordinates scaled by
    # joint_observability_amplitude_scale.
    joint_observability_weight: float = PydanticField(default=0.0, ge=0.0)
    joint_observability_min_singular_per_scaled_mode: float = PydanticField(
        default=1.0e-2,
        gt=0.0,
    )
    joint_observability_crlb_weight: float = PydanticField(default=0.0, ge=0.0)
    joint_observability_crlb_target_per_scaled_mode: float = PydanticField(
        default=100.0,
        gt=0.0,
    )
    joint_observability_science_null_guard_weight: float = PydanticField(
        default=0.0,
        ge=0.0,
    )
    joint_observability_science_null_guard_factor: float = PydanticField(
        default=1.0,
        gt=0.0,
    )
    joint_observability_port_set: Literal["bright", "bright_dark", "all"] = (
        "bright_dark"
    )
    joint_observability_n_spectral_channels: int = PydanticField(default=0, ge=0)
    joint_observability_amplitude_scale: float = PydanticField(
        default=0.01,
        gt=0.0,
    )
    joint_observability_eps: float = PydanticField(default=1.0e-12, gt=0.0)
    # Shared chip chromaticity model used by the optimizer and deployment
    # stages. ``legacy`` preserves the KILO/nullsim convention exactly:
    # theta is achromatic and phi/alpha scale as wl_design / wl. ``parametric``
    # enables the optional theta-polynomial and phase-dispersion knobs below.
    chip_chromaticity_model: Literal["legacy", "parametric"] = "legacy"
    theta_chromaticity_slope_rad_per_frac: float = 0.0
    theta_chromaticity_quadratic_rad_per_frac2: float = 0.0
    phase_chromaticity_model: Literal[
        "legacy_inverse_lambda",
        "fixed_opd_neff",
    ] = "legacy_inverse_lambda"
    phase_neff_ref: float = PydanticField(
        default=_DEFAULT_PHASE_NEFF_REF,
        gt=0.0,
    )
    phase_group_index: float = PydanticField(
        default=_DEFAULT_PHASE_GROUP_INDEX,
        gt=0.0,
    )
    # Torch backend only: optional upstream chip_optimization component used
    # to initialize the first restart. If the source publishes an
    # ``optimal_u_stack`` (for example chip_optimization.ideal_kernel), the
    # torch backend fits that design-wavelength unitary into its native
    # Clements parameter order before broadband refinement.
    warm_start_source: str | None = None
    warm_start_fit_max_iter: int = PydanticField(default=500, ge=1)

    @model_validator(mode="after")
    def _resolve_compile_env(self) -> ChipOptimizationParams:
        # Fold the NULLSIM_TORCH_COMPILE env override into the validated
        # param so the executed objective path enters the stage cache key
        # (review 2026-06-10 Issue 25). "Either source enables it"
        # semantics are unchanged.
        if not self.compile_objective and _torch_compile_env_enabled():
            object.__setattr__(self, "compile_objective", True)
        return self

    @model_validator(mode="after")
    def _check_port_split(self) -> ChipOptimizationParams:
        if (
            self.n_bright_ports
            + self.n_dark_ports
            + self.n_spectro_ports
            != self.n_modes
        ):
            raise ValueError(
                "chip_optimization requires "
                "n_bright_ports + n_dark_ports + n_spectro_ports == n_modes "
                f"(got {self.n_bright_ports} + {self.n_dark_ports} + "
                f"{self.n_spectro_ports} != {self.n_modes}). KILO layout: "
                "[bright | dark | spectro] disjoint partitions."
            )
        if self.torch_device.lower() not in {"auto", "cpu", "cuda"}:
            raise ValueError("torch_device must be one of 'auto', 'cpu', or 'cuda'.")
        if (
            self.tracker_observability_weight > 0.0
            or self.tracker_observability_crlb_weight > 0.0
            or self.tracker_observability_rank_priority
        ) and self.n_bright_ports <= 0:
            raise ValueError(
                "tracker_observability requires at least one bright port."
            )
        if self.amplitude_mismatch_weight > 0.0:
            if self.n_bright_ports <= 0:
                raise ValueError(
                    "amplitude_mismatch_weight requires at least one bright port."
                )
            if (
                self.amplitude_dump_port_index is not None
                and self.amplitude_dump_port_index >= self.n_bright_ports
            ):
                raise ValueError(
                    "amplitude_dump_port_index is indexed within the bright-port "
                    f"block and must be < n_bright_ports ({self.n_bright_ports})."
                )
        if self.amplitude_observability_weight > 0.0:
            if (
                self.amplitude_observability_port_set == "bright"
                and self.n_bright_ports <= 0
            ):
                raise ValueError(
                    "amplitude_observability_port_set='bright' requires at least "
                    "one bright port."
                )
            if (
                self.amplitude_observability_port_set == "bright_dark"
                and self.n_bright_ports + self.n_dark_ports <= 0
            ):
                raise ValueError(
                    "amplitude_observability_port_set='bright_dark' selected no "
                    "ports."
                )
        if (
            self.joint_observability_weight > 0.0
            or self.joint_observability_crlb_weight > 0.0
        ):
            # The CRLB-only activation (weight=0, crlb_weight>0) runs the
            # same port-set machinery, so it needs the same validation
            # (review 2026-06-10).
            if (
                self.joint_observability_port_set == "bright"
                and self.n_bright_ports <= 0
            ):
                raise ValueError(
                    "joint_observability_port_set='bright' requires at least "
                    "one bright port."
                )
            if (
                self.joint_observability_port_set == "bright_dark"
                and self.n_bright_ports + self.n_dark_ports <= 0
            ):
                raise ValueError(
                    "joint_observability_port_set='bright_dark' selected no "
                    "ports."
                )
        _validate_chip_chromaticity_params(self)
        return self

    @property
    def objective_is_flux_scale_dependent(self) -> bool:
        """Whether the objective optimum depends on absolute input flux.

        The base broadband null objective (absolute dark-port leakage plus the
        throughput penalty) is scale-equivariant: scaling every input field
        amplitude by a constant multiplies the loss by a constant and leaves
        the argmin unchanged, so the reported null depth and the stellar-disk
        null floor it drives are flux-scale invariant.

        The tracker / joint / amplitude observability terms, however, carry
        absolute physical targets — per-nm intensity-response singular floors
        and CRLB targets in nm — that are genuinely photon-rate dependent (a
        fringe tracker's piston estimability scales with the actual photon
        flux). When any of those terms is active the optimum, and therefore
        the realized stellar-disk null floor, depends on which flux the chip
        was optimized at, so the per-flux disk floor is chip-realization
        dependent rather than a single fixed number.
        """

        return (
            self.tracker_observability_weight > 0.0
            or self.tracker_observability_crlb_weight > 0.0
            or bool(self.tracker_observability_rank_priority)
            or self.amplitude_observability_weight > 0.0
            or self.joint_observability_weight > 0.0
            or self.joint_observability_crlb_weight > 0.0
        )


# ---------------------------------------------------------------------------
# Shared helpers used by all three backends. Factored here per DRY — the
# field validation, wavelength rescaling helpers, and result-writing path
# are identical across scipy / torch; only the optimizer call and the
# Clements unitary builder differ.
# ---------------------------------------------------------------------------


def _mode_areas_m2(
    state: SimulationState,
    n_modes: int,
) -> NDArray[np.float64] | None:
    """Return the physical collecting area represented by each chip mode.

    KILO's optimizer fields are expressed as ``sqrt(coupling * area)``.
    nullsim's built-in injection stages keep area out of
    ``state.field.amplitudes`` because the detector stages apply it
    downstream, so the chip optimizer has to reintroduce the per-mode
    area locally to match KILO's objective scale.
    """

    component_areas = _component_mode_areas_m2(state, n_modes)
    if component_areas is not None:
        areas, _source = component_areas
        return areas.astype(np.float64, copy=True)

    component_area = _mode_geometry_scalar(state, "mode_area_m2")
    if component_area is not None:
        return np.full(n_modes, component_area, dtype=np.float64)

    area_total = state.geometry.effective_collecting_area_m2
    if area_total is None:
        area_total = state.geometry.collecting_area_m2
    if area_total is None and state.geometry.area_m2 is not None:
        if state.geometry.n_apertures is not None:
            area_total = (
                float(state.geometry.area_m2)
                * int(state.geometry.n_apertures)
            )
        else:
            area_total = float(state.geometry.area_m2)
    if area_total is None:
        return None
    area_total = float(area_total)
    if not np.isfinite(area_total) or area_total <= 0.0:
        return None
    return np.full(n_modes, area_total / float(n_modes), dtype=np.float64)


def _field_area_reference_m2(
    state: SimulationState,
    n_modes: int,
) -> float | None:
    """Return scalar area reference already carried in field amplitudes."""

    for payload in reversed(tuple(state.geometry.components.values())):
        if not isinstance(payload, dict):
            continue
        n_output_raw = payload.get("n_output_modes")
        if n_output_raw is not None and int(n_output_raw) != n_modes:
            continue
        value = payload.get("field_area_reference_m2")
        if value is None:
            continue
        try:
            area = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(area) and area > 0.0:
            return area
    return None


def _mode_diameters_m(
    state: SimulationState,
    n_modes: int,
) -> NDArray[np.float64] | None:
    """Return per-mode physical diameters for off-axis coupling.

    Published geometry diameters win; the equal-area circle is only the
    fallback when no stage published one.
    """

    component_diameters = _component_mode_diameters_m(state, n_modes)
    if component_diameters is not None:
        diameters, _source = component_diameters
        return diameters.astype(np.float64, copy=True)

    component_diameter = _mode_geometry_scalar(state, "mode_diameter_m")
    if component_diameter is not None:
        return np.full(n_modes, component_diameter, dtype=np.float64)

    mode_areas = _mode_areas_m2(state, n_modes)
    if mode_areas is not None:
        return np.sqrt(4.0 * mode_areas / math.pi)
    if state.geometry.diameter_m is None:
        return None
    n_apertures = int(state.geometry.n_apertures or 1)
    if n_apertures <= 0 or n_modes % n_apertures != 0:
        return None
    n_sub_per_aperture = n_modes // n_apertures
    if n_sub_per_aperture <= 0:
        return None
    sub_diameter = float(state.geometry.diameter_m) / math.sqrt(n_sub_per_aperture)
    return np.full(n_modes, sub_diameter, dtype=np.float64)


def _mode_geometry_scalar(
    state: SimulationState,
    key: str,
) -> float | None:
    """Return a positive scalar per-mode geometry override if present."""

    for payload in reversed(tuple(state.geometry.components.values())):
        if not isinstance(payload, dict):
            continue
        value = payload.get(key)
        if value is None:
            continue
        scalar = float(value)
        if np.isfinite(scalar) and scalar > 0.0:
            return scalar
    return None


def _stellar_radius_rad(state: SimulationState) -> float:
    if state.scene.star_angular_diameter_mas is None:
        return 0.0
    return float(state.scene.star_angular_diameter_mas) * _RAD_PER_MAS / 2.0


def _require_field(
    state: SimulationState,
    n_modes: int,
    *,
    wl_design_um: float | None = None,
) -> NDArray[np.complex128]:
    """Return the on-axis nominal stellar field for chip optimization.

    KILO ``injection.py:stellar_input_field`` (line 194-225) returns
    ``sqrt(coupling × area)`` — a real, on-axis-phase-zero field. The chip
    is designed at the NOMINAL operating point: phases that come from
    upstream AO residual + transport delays do NOT belong in the
    optimizer's input because the chip is a fixed photonic device, not an
    adaptive nuller.

    Pre-fix this returned ``state.field.amplitudes`` verbatim, which
    carries phases from atmospheric piston + transport (range up to ±π
    across the band at J=4 Keck). Optimizing against those phases drove
    the broadband solver into a shallow Pareto basin (mean dark/bright
    ~1e-3 at the band edges) — the chromatic chip could not simultaneously
    null a phase-perturbed input at five wavelengths. With the phases
    stripped (KILO convention), the same optimizer now reaches mean
    dark/bright ~1e-5 at J=4 Keck, comparable to KILO's theoretical
    chip floor.

    Implementation choice: take the PER-MODE magnitude near the design
    wavelength from ``state.field.amplitudes`` and convert it to KILO's
    absolute ``sqrt(coupling*area)`` convention. For ordinary unsplit fields,
    that means multiplying by ``sqrt(area_per_mode)``. For a partially split
    field, ``injection.mode_splitter`` has already encoded relative branch
    areas in the amplitudes, so only its scalar ``field_area_reference_m2`` is
    applied here.
    """
    field_amplitudes = state.field.amplitudes
    if field_amplitudes is None:
        raise ValueError(
            "chip_optimization requires field amplitudes from an earlier stage."
        )
    field_array = np.asarray(field_amplitudes)
    if field_array.ndim != 3:
        raise ValueError(
            "chip_optimization requires field amplitudes with shape "
            "[n_modes, n_wavelength_bins, n_pol]."
        )
    if field_array.shape[0] != n_modes:
        raise ValueError(
            "chip_optimization field n_modes axis must match params.n_modes "
            f"(field has {field_array.shape[0]}, params has {n_modes})."
        )
    # Strip upstream phases and collapse per-mode magnitude to a single
    # design-wavelength value, then broadcast back to (n_modes, n_wl, n_pol).
    n_wl = field_array.shape[1]
    n_pol = field_array.shape[2]
    wavelength_grid_um = (
        state.field.wavelength_grid_um
        or state.throughput.wavelength_grid_um
    )
    if (
        wl_design_um is not None
        and wavelength_grid_um
        and len(wavelength_grid_um) == n_wl
    ):
        grid = np.asarray(wavelength_grid_um, dtype=np.float64)
        design_idx = int(np.argmin(np.abs(grid - float(wl_design_um))))
        per_mode_mag = np.abs(field_array[:, design_idx, :])
    else:
        per_mode_mag = np.abs(field_array).mean(axis=1)  # (n_modes, n_pol)

    field_area_reference = _field_area_reference_m2(state, n_modes)
    if field_area_reference is not None:
        per_mode_mag = per_mode_mag * math.sqrt(field_area_reference)
    else:
        mode_areas = _mode_areas_m2(state, n_modes)
        if mode_areas is not None:
            per_mode_mag = per_mode_mag * np.sqrt(mode_areas)[:, np.newaxis]

    return np.broadcast_to(
        per_mode_mag[:, np.newaxis, :], (n_modes, n_wl, n_pol),
    ).astype(np.complex128, copy=True)


def _require_wavelength_grid_um(
    state: SimulationState,
    field_amplitudes: NDArray[np.complex128],
) -> tuple[float, ...]:
    field_grid_um = state.field.wavelength_grid_um
    throughput_grid_um = state.throughput.wavelength_grid_um
    wavelength_grid_um = field_grid_um or throughput_grid_um
    if not wavelength_grid_um:
        raise ValueError(
            "chip_optimization requires a non-empty wavelength grid from "
            "the resolved grid config."
        )
    if field_amplitudes.shape[1] != len(wavelength_grid_um):
        raise ValueError(
            "chip_optimization field wavelength axis must match the wavelength grid."
        )
    return tuple(float(value) for value in wavelength_grid_um)


def _optimizer_wavelength_subset(
    field_array: NDArray[np.complex128],
    wavelength_grid_um: tuple[float, ...],
    params: ChipOptimizationParams,
) -> tuple[NDArray[np.complex128], tuple[float, ...], NDArray[np.int64]]:
    """Return the wavelength grid actually used by the optimizer objective."""

    if params.n_wl_optimize <= 0 or params.n_wl_optimize >= len(wavelength_grid_um):
        indices = np.arange(len(wavelength_grid_um), dtype=np.int64)
        return field_array, wavelength_grid_um, indices

    full_wl = np.asarray(wavelength_grid_um, dtype=np.float64)
    target_wl = np.linspace(full_wl.min(), full_wl.max(), params.n_wl_optimize)
    indices = np.array(
        [int(np.argmin(np.abs(full_wl - target))) for target in target_wl],
        dtype=np.int64,
    )
    indices = np.unique(indices)
    return (
        field_array[:, indices, :],
        tuple(float(full_wl[index]) for index in indices),
        indices,
    )


def _optimizer_wavelength_metadata(
    *,
    full_wavelength_grid_um: tuple[float, ...],
    optimizer_wavelength_grid_um: tuple[float, ...],
    optimizer_wavelength_indices: NDArray[np.integer],
) -> dict[str, Any]:
    """Diagnostic payload for wavelength-subsampled optimization."""

    return {
        "optimizer_wavelength_grid_um": tuple(
            float(value) for value in optimizer_wavelength_grid_um
        ),
        "optimizer_wavelength_indices": tuple(
            int(value) for value in optimizer_wavelength_indices
        ),
        "optimizer_n_wavelengths": int(len(optimizer_wavelength_grid_um)),
        "optimizer_full_n_wavelengths": int(len(full_wavelength_grid_um)),
    }


def _n_params(n_modes: int) -> int:
    n_mzi = n_modes * (n_modes - 1) // 2
    return 2 * n_mzi + n_modes


def _clements_order(n_modes: int) -> list[tuple[int, int]]:
    """Return the (upper_row, lower_row) MZI pair sequence consumed by ``reconstruct_clements``."""

    return [
        (lower_row - 1, lower_row)
        for column_index in range(n_modes - 1)
        for lower_row in range(n_modes - 1, column_index, -1)
    ]


_TWO_PI = 2.0 * math.pi


def _quantize_phases(
    raw_params: NDArray[np.float64],
    dac_bits: int | None,
) -> NDArray[np.float64]:
    """DAC quantization (round-to-nearest at ``step = 2*pi / 2**bits``).

    Does NOT wrap to ``[0, 2*pi)``. Wrapping belongs at the call site that
    also applies the per-wavelength ``wl_design / wl_target`` rescaling
    (see ``_build_unitary_stack_numpy`` below for the scipy reference, and
    ``chip.kernel_mzi_mesh._quantize_phases`` for the kernel deployment).
    The torch backends use this helper to round their final iterate
    onto the DAC grid before publishing an ``optimal_u_stack`` built from
    the SAME unwrapped phases the autograd objective minimized — wrapping
    here would shift the deployed chromatic chip away from the optimizer's
    operating point and inflate the reported loss by O(1).
    """

    if dac_bits is None:
        return raw_params
    if dac_bits <= 0:
        raise ValueError(
            f"chip_optimization dac_bits must be positive when set (got {dac_bits})."
        )
    step = _TWO_PI / (1 << dac_bits)
    return step * np.round(raw_params / step)


def _validate_chip_chromaticity_params(params: Any) -> None:
    """Validate the shared chip-chromaticity fields on a params-like object."""

    model = str(getattr(params, "chip_chromaticity_model", "legacy"))
    if model == "legacy":
        if not math.isclose(
            float(getattr(params, "theta_chromaticity_slope_rad_per_frac", 0.0)),
            0.0,
            abs_tol=0.0,
        ):
            raise ValueError(
                "theta_chromaticity_slope_rad_per_frac requires "
                "chip_chromaticity_model='parametric'."
            )
        if not math.isclose(
            float(
                getattr(
                    params,
                    "theta_chromaticity_quadratic_rad_per_frac2",
                    0.0,
                )
            ),
            0.0,
            abs_tol=0.0,
        ):
            raise ValueError(
                "theta_chromaticity_quadratic_rad_per_frac2 requires "
                "chip_chromaticity_model='parametric'."
            )
        if (
            str(getattr(params, "phase_chromaticity_model", "legacy_inverse_lambda"))
            != "legacy_inverse_lambda"
        ):
            raise ValueError(
                "phase_chromaticity_model values other than "
                "'legacy_inverse_lambda' require "
                "chip_chromaticity_model='parametric'."
            )
    elif model != "parametric":
        raise ValueError(
            "chip_chromaticity_model must be 'legacy' or 'parametric'."
        )


def _phase_scale_factors_numpy(
    *,
    wavelength_grid_um: tuple[float, ...],
    wl_design_um: float,
    chromaticity_params: Any | None,
    rescale_per_wavelength: bool,
) -> NDArray[np.float64]:
    """Return per-wavelength phase scales for Clements phi/alpha blocks.

    ``legacy_inverse_lambda`` is the historical KILO/nullsim rule. The
    ``fixed_opd_neff`` option treats stored phase values as the design-
    wavelength phase from a physical path-length difference and applies a
    first-order effective-index dispersion model:

        n_eff(lambda) = n_eff0 + (n_eff0 - n_g) * (lambda/lambda0 - 1)
        phase(lambda) / phase(lambda0)
            = (lambda0/lambda) * n_eff(lambda) / n_eff0
    """

    wavelengths = np.asarray(wavelength_grid_um, dtype=np.float64)
    if not rescale_per_wavelength:
        return np.ones_like(wavelengths, dtype=np.float64)
    wl_design = float(wl_design_um)
    legacy_scale = wl_design / wavelengths
    if not _chip_chromaticity_is_parametric(chromaticity_params):
        return legacy_scale
    phase_model = str(
        getattr(
            chromaticity_params,
            "phase_chromaticity_model",
            "legacy_inverse_lambda",
        )
    )
    if phase_model == "legacy_inverse_lambda":
        return legacy_scale
    if phase_model == "fixed_opd_neff":
        n_eff_ref = float(getattr(chromaticity_params, "phase_neff_ref"))
        n_group = float(getattr(chromaticity_params, "phase_group_index"))
        detuning = wavelengths / wl_design - 1.0
        n_eff = n_eff_ref + (n_eff_ref - n_group) * detuning
        if np.any(n_eff <= 0.0):
            raise ValueError(
                "fixed_opd_neff produced non-positive effective index over "
                "the wavelength grid; reduce phase_group_index or bandwidth."
            )
        return legacy_scale * (n_eff / n_eff_ref)
    raise ValueError(f"Unknown phase_chromaticity_model {phase_model!r}.")


def _theta_offsets_numpy(
    *,
    wavelength_grid_um: tuple[float, ...],
    wl_design_um: float,
    chromaticity_params: Any | None,
    apply_chromaticity: bool,
) -> NDArray[np.float64]:
    """Return additive splitter-angle offsets for the Clements theta block."""

    wavelengths = np.asarray(wavelength_grid_um, dtype=np.float64)
    if (
        not apply_chromaticity
        or not _chip_chromaticity_is_parametric(chromaticity_params)
    ):
        return np.zeros_like(wavelengths, dtype=np.float64)
    detuning = wavelengths / float(wl_design_um) - 1.0
    linear = float(
        getattr(chromaticity_params, "theta_chromaticity_slope_rad_per_frac")
    )
    quadratic = float(
        getattr(chromaticity_params, "theta_chromaticity_quadratic_rad_per_frac2")
    )
    return linear * detuning + quadratic * detuning * detuning


def _chip_chromaticity_is_parametric(chromaticity_params: Any | None) -> bool:
    if chromaticity_params is None:
        return False
    return str(getattr(chromaticity_params, "chip_chromaticity_model", "legacy")) == (
        "parametric"
    )


def _chip_chromaticity_metadata(params: Any) -> dict[str, float | str]:
    """Return scalar metadata describing the chip chromaticity model."""

    return {
        "chip_chromaticity_model": str(
            getattr(params, "chip_chromaticity_model", "legacy")
        ),
        "theta_chromaticity_slope_rad_per_frac": float(
            getattr(params, "theta_chromaticity_slope_rad_per_frac", 0.0)
        ),
        "theta_chromaticity_quadratic_rad_per_frac2": float(
            getattr(params, "theta_chromaticity_quadratic_rad_per_frac2", 0.0)
        ),
        "phase_chromaticity_model": str(
            getattr(params, "phase_chromaticity_model", "legacy_inverse_lambda")
        ),
        "phase_neff_ref": float(
            getattr(params, "phase_neff_ref", _DEFAULT_PHASE_NEFF_REF)
        ),
        "phase_group_index": float(
            getattr(params, "phase_group_index", _DEFAULT_PHASE_GROUP_INDEX)
        ),
    }


def _build_unitary_stack_numpy(
    *,
    params_vec: NDArray[np.float64],
    n_modes: int,
    wavelength_grid_um: tuple[float, ...],
    wl_design_um: float,
    dac_bits: int | None,
    chromaticity_params: Any | None = None,
) -> NDArray[np.complex128]:
    """Numpy reference Clements stack used by the scipy backend.

    Quantization is applied at the design wavelength BEFORE the
    ``phi``+``alpha`` rescaling (notes §1). The quantized phases are
    wrapped to ``[0, 2*pi)`` here — NOT inside ``_quantize_phases`` —
    because the wrap is a property of the per-wavelength rescaling
    (negative or super-2pi phases collide at the design wavelength but
    diverge at off-design once multiplied by ``wl_design / wl_target``).
    The torch backends bypass this helper and publish a
    ``u_stack`` built from unwrapped phases the autograd objective
    actually minimized; the kernel uses that ``u_stack`` verbatim so
    no extra wrap is required there.
    """

    quantized = _quantize_phases(np.asarray(params_vec, dtype=np.float64), dac_bits)
    quantized = np.mod(quantized, _TWO_PI)
    n_mzi = n_modes * (n_modes - 1) // 2
    theta = quantized[0:n_mzi]
    phi = quantized[n_mzi : 2 * n_mzi]
    alpha = quantized[2 * n_mzi : 2 * n_mzi + n_modes]
    order = _clements_order(n_modes)

    n_wl = len(wavelength_grid_um)
    phase_scales = _phase_scale_factors_numpy(
        wavelength_grid_um=wavelength_grid_um,
        wl_design_um=wl_design_um,
        chromaticity_params=chromaticity_params,
        rescale_per_wavelength=True,
    )
    theta_offsets = _theta_offsets_numpy(
        wavelength_grid_um=wavelength_grid_um,
        wl_design_um=wl_design_um,
        chromaticity_params=chromaticity_params,
        apply_chromaticity=True,
    )
    stack = np.empty((n_modes, n_modes, n_wl), dtype=np.complex128)
    for k, scale in enumerate(phase_scales):
        decomp = ClementsDecomposition(
            theta=theta + theta_offsets[k],
            phi=phi * scale,
            alpha=alpha * scale,
            order=order,
            n_modes=n_modes,
        )
        stack[:, :, k] = reconstruct_clements(decomp)
    return stack


def _canonicalize_clements_params(
    native_params: NDArray[np.float64],
    n_modes: int,
    *,
    build_unitary_complex: Any,
    atol: float = 1.0e-5,
) -> NDArray[np.float64]:
    """Re-pack a backend-native Clements param vector into the canonical layout.

    Each chip_optimization backend uses its own MZI ordering while
    optimizing — the scipy backend uses ``physics.clements`` lexicographic
    order ``(lower_row - 1, lower_row)``, while the torch backends
    use KILO's column-alternating layout (``col % 2`` start row).
    The downstream ``chip.kernel_mzi_mesh`` stage reconstructs the unitary
    via ``physics.clements.reconstruct_clements``, which is locked to the
    lexicographic layout.

    Finding 3 was that the same param vector produced different unitaries
    between the torch optimizer (used during the search) and the
    kernel reconstruction (used downstream), so the chip never matched
    what the optimizer minimized. The fix here: after the optimizer
    converges, build the optimal unitary in the backend's native
    ordering, ``decompose_clements`` it to get the canonical
    ``(theta, phi, alpha)`` triplet, and store *those* angles. The
    kernel's reconstruction then bit-exactly reproduces the optimized
    unitary regardless of which backend ran.

    Args:
        native_params: Backend-native Clements params (theta || phi || alpha).
        n_modes: Number of optical modes (M).
        build_unitary_complex: Callable ``(params, n_modes) -> ndarray``
            that returns the (M, M) complex unitary built using the
            backend's own ordering.
        atol: Tolerance for the unitarity check during decomposition.
            Loosened from physics.clements's default 1e-10 to 1e-5 to
            absorb numerical drift in the float64 backend Clements
            reconstruction. The downstream quantization and
            reconstruction round-trip absorbs any deviation at this
            scale.

    Returns:
        Canonical params, same shape as ``native_params``, in the layout
        ``physics.clements`` expects.
    """

    from nullsim.physics.clements import decompose_clements

    U = build_unitary_complex(np.asarray(native_params, dtype=np.float64), int(n_modes))
    decomp = decompose_clements(U, atol=atol)
    return np.concatenate(
        [
            np.asarray(decomp.theta, dtype=np.float64),
            np.asarray(decomp.phi, dtype=np.float64),
            np.asarray(decomp.alpha, dtype=np.float64),
        ]
    )


def _dark_indices(params: ChipOptimizationParams) -> NDArray[np.int64]:
    """Output rows that count as planet-detection dark ports.

    KILO ``assign_ports`` (kilo/chip.py:2049) lays the M chip outputs
    out as ``[bright(0..n_b-1) | dark(n_b..n_b+n_d-1) | spectro(n_b+n_d..M-1)]``.
    This helper returns just the dark slice — the indices KILO's
    throughput-penalty term sums over (natural_design.py L643:
    ``throughput_port_indices_eff = dark_idx``).

    For the null-constraint set (dark + spectro), use
    ``_null_indices``.
    """

    start = params.n_bright_ports
    stop = params.n_bright_ports + params.n_dark_ports
    return np.arange(start, stop, dtype=np.int64)


@dataclass(frozen=True)
class _TrackerPistonBasis:
    """Physical piston basis expanded onto chip input modes."""

    physical_basis: NDArray[np.float64]
    chip_basis: NDArray[np.float64]
    n_physical_modes: int
    source: str


def _tracker_piston_basis_from_state(
    state: SimulationState,
    n_chip_modes: int,
) -> _TrackerPistonBasis:
    """Return the common-piston-free physical basis seen by chip inputs.

    For ordinary meshes the chip input modes are the physical piston modes.
    For ``injection.mode_splitter`` fanout studies, multiple chip inputs map
    back to one physical aperture/subaperture piston; the basis is expanded
    through that splitter map so the optimizer regularizes real piston DOFs,
    not artificial split-copy phase DOFs.
    """

    splitter = _splitter_physical_indices(state, n_chip_modes)
    if splitter is None:
        basis = differential_piston_basis(n_chip_modes)
        return _TrackerPistonBasis(
            physical_basis=basis,
            chip_basis=basis,
            n_physical_modes=n_chip_modes,
            source="chip_input_modes",
        )

    physical_indices, source = splitter
    n_physical_modes = int(np.max(physical_indices)) + 1
    physical_basis = differential_piston_basis(n_physical_modes)
    expansion = np.zeros((n_chip_modes, n_physical_modes), dtype=np.float64)
    expansion[np.arange(n_chip_modes, dtype=np.intp), physical_indices] = 1.0
    return _TrackerPistonBasis(
        physical_basis=physical_basis,
        chip_basis=expansion @ physical_basis,
        n_physical_modes=n_physical_modes,
        source=source,
    )


def _null_indices(params: ChipOptimizationParams) -> NDArray[np.int64]:
    """Output rows the optimizer forces to zero stellar leakage.

    KILO ``natural_design.py:390``:
    ``null_idx = np.concatenate([dark_idx, spectro_idx])`` — the
    spectro ports are treated as dark (zero stellar leakage) for the
    chip null objective but are NOT in the throughput-penalty sum.
    Returns ``[n_bright_ports, n_modes)`` (i.e. dark + spectro,
    contiguous because KILO's port layout is ``[bright | dark |
    spectro]``).
    """

    return np.arange(
        params.n_bright_ports, params.n_modes, dtype=np.int64,
    )


def _loss_numpy(
    params_vec: NDArray[np.float64],
    *,
    field_array: NDArray[np.complex128],
    wavelength_grid_um: tuple[float, ...],
    params: ChipOptimizationParams,
    null_idx: NDArray[np.int64],
    dark_idx: NDArray[np.int64] | None = None,
    aperture_positions_pupil_m: NDArray[np.floating] | None = None,
    sub_diameters_m: NDArray[np.floating] | None = None,
    stellar_radius_rad: float = 0.0,
    delay_state_fields: NDArray[np.complex128] | None = None,
    throughput_penalty_fields: NDArray[np.complex128] | None = None,
) -> float:
    """Broadband absolute dark-port leakage (KILO numerical convention).

    Direct port of KILO ``make_torch_broadband_null_objective_and_grad``
    (chip_torch.py L194-385). Per-wavelength leakage at null ports is
    summed across polarization, then weighted-averaged across wavelengths:

        L = sum_k w_k * sum_{i in dark, p} |U_k @ field_k|^2 [i, p]
        w_k = 1 / n_wl   (uniform)

    Pre-fix this returned ``mean(dark) / mean(bright)``, which has a
    less-tractable basin geometry for L-BFGS-B because the bright
    denominator can drift toward 0 along directions that don't reduce
    the dark numerator. KILO regularly drives the absolute-leakage
    form to 1e-11 (unquantized single-wavelength) on M=24 with autograd;
    the ratio form caps at ~1e-1 on the same hardware/seed. See the
    chip_torch.py docstring at L207 for the original derivation.
    """

    u_stack = _build_unitary_stack_numpy(
        params_vec=params_vec,
        n_modes=params.n_modes,
        wavelength_grid_um=wavelength_grid_um,
        wl_design_um=params.wl_design_um,
        dac_bits=params.dac_bits,
        chromaticity_params=params,
    )

    active_disk_samples = _uses_disk_samples(params) and stellar_radius_rad > 0.0
    active_delay_states = delay_state_fields is not None
    if params.kernel_order == 0 and not active_disk_samples and not active_delay_states:
        # output[i, wl, pol] = sum_j U[i, j, wl] * field[j, wl, pol]
        output = np.einsum("ijw,jwp->iwp", u_stack, field_array)
        leak_per_wl = np.sum(np.abs(output[null_idx, :, :]) ** 2, axis=(0, 2))
        loss = _reduce_spectral_leakage_numpy(
            leak_per_wl,
            params,
            wavelength_grid_um=wavelength_grid_um,
        )
        return _apply_throughput_penalty_numpy(
            loss,
            u_stack=u_stack,
            params=params,
            dark_idx=dark_idx,
            throughput_penalty_fields=throughput_penalty_fields,
        )

    if (params.kernel_order > 0 or active_disk_samples) and aperture_positions_pupil_m is None:
        raise ValueError(
            f"chip_optimization with kernel_order={params.kernel_order} "
            "or disk samples requires "
            "state.geometry.aperture_positions_pupil_m to be populated by an "
            "upstream stage (e.g. ao.pyramid_wfs). Set kernel_order=0 to "
            "fall back to on-axis-only nulling and disable disk samples."
        )
    field_stack = _optimizer_null_field_stack(
        on_axis_field=field_array,
        params=params,
        aperture_positions_pupil_m=aperture_positions_pupil_m,
        wavelength_grid_um=wavelength_grid_um,
        sub_diameters_m=sub_diameters_m,
        stellar_radius_rad=stellar_radius_rad,
        delay_state_fields=delay_state_fields,
    )
    # output_stack[r, i, wl, pol] = sum_j U[i, j, wl] * field_stack[r, j, wl, pol]
    output = np.einsum("ijw,rjwp->riwp", u_stack, field_stack)
    leak_per_wl_per_row = np.sum(
        np.abs(output[:, null_idx, :, :]) ** 2, axis=(1, 3)
    )  # (n_rows, n_wl)
    leak_per_wl = np.sum(leak_per_wl_per_row, axis=0)
    loss = _reduce_spectral_leakage_numpy(
        leak_per_wl,
        params,
        wavelength_grid_um=wavelength_grid_um,
    )
    return _apply_throughput_penalty_numpy(
        loss,
        u_stack=u_stack,
        params=params,
        dark_idx=dark_idx,
        throughput_penalty_fields=throughput_penalty_fields,
    )


def _apply_throughput_penalty_numpy(
    loss: float,
    *,
    u_stack: NDArray[np.complex128],
    params: ChipOptimizationParams,
    dark_idx: NDArray[np.int64] | None,
    throughput_penalty_fields: NDArray[np.complex128] | None,
) -> float:
    if throughput_penalty_fields is None or params.throughput_penalty_alpha <= 0.0:
        return float(loss)
    if dark_idx is None:
        raise ValueError("throughput penalty requires dark_idx.")
    fields = np.asarray(throughput_penalty_fields, dtype=np.complex128)
    if fields.ndim != 3 or fields.shape[0] != u_stack.shape[2]:
        raise ValueError(
            "throughput_penalty_fields must have shape "
            f"(n_wl, n_samples, n_modes); got {fields.shape}."
        )
    output = np.einsum("ijw,wsj->wsi", u_stack, fields)
    dark_power = np.sum(np.abs(output[:, :, dark_idx]) ** 2, axis=2)
    mean_throughput_per_wl = np.mean(dark_power, axis=1)
    mean_throughput = float(np.sum(mean_throughput_per_wl) / float(u_stack.shape[2]))
    return float(loss) - float(params.throughput_penalty_alpha) * mean_throughput


def _edge_weighted_spectral_weights_numpy(
    wavelength_grid_um: tuple[float, ...] | NDArray[np.floating],
    params: ChipOptimizationParams,
) -> NDArray[np.float64]:
    """Return edge-emphasis weights for wavelength-bin leakage."""

    wavelengths = np.asarray(wavelength_grid_um, dtype=np.float64)
    if wavelengths.ndim != 1:
        raise ValueError(
            "edge-weighted spectral loss expects a 1-D wavelength grid; "
            f"got shape {wavelengths.shape}."
        )
    if wavelengths.size == 0:
        raise ValueError("edge-weighted spectral loss requires wavelengths.")
    edge_distance = np.max(np.abs(wavelengths - float(params.wl_design_um)))
    if edge_distance <= 0.0:
        return np.ones_like(wavelengths, dtype=np.float64)
    normalized_distance = np.abs(wavelengths - float(params.wl_design_um))
    normalized_distance = normalized_distance / edge_distance
    return 1.0 + float(params.spectral_edge_weight_alpha) * (
        normalized_distance ** float(params.spectral_edge_weight_power)
    )


def _reduce_spectral_leakage_numpy(
    leak_per_wl: NDArray[np.floating],
    params: ChipOptimizationParams,
    *,
    wavelength_grid_um: tuple[float, ...] | NDArray[np.floating] | None = None,
) -> float:
    """Reduce per-wavelength leakage to the scalar optimizer objective."""

    leakage = np.asarray(leak_per_wl, dtype=np.float64)
    if leakage.ndim != 1:
        raise ValueError(
            "_reduce_spectral_leakage_numpy expects a 1-D wavelength vector; "
            f"got shape {leakage.shape}."
        )
    if params.spectral_loss_mode in {"off", "mean"}:
        return float(np.mean(leakage))
    if params.spectral_loss_mode == "p_norm":
        max_leak = float(np.max(leakage))
        if max_leak <= 0.0:
            return 0.0
        scaled = np.clip(leakage / max_leak, 0.0, None)
        p = float(params.spectral_loss_p)
        return float(max_leak * np.mean(scaled ** p) ** (1.0 / p))
    if params.spectral_loss_mode == "edge_weighted_mean":
        if wavelength_grid_um is None:
            raise ValueError(
                "edge_weighted_mean spectral loss requires wavelength_grid_um."
            )
        weights = _edge_weighted_spectral_weights_numpy(wavelength_grid_um, params)
        if weights.shape != leakage.shape:
            raise ValueError(
                "edge_weighted_mean spectral loss requires leakage and "
                f"wavelength vectors with the same shape; got {leakage.shape} "
                f"and {weights.shape}."
            )
        return float(np.sum(weights * leakage) / np.sum(weights))
    raise ValueError(f"Unknown spectral_loss_mode {params.spectral_loss_mode!r}.")


def _spectral_loss_metadata(params: ChipOptimizationParams) -> dict[str, float | str]:
    """Payload fields describing the spectral objective reducer."""

    return {
        "spectral_loss_mode": str(params.spectral_loss_mode),
        "spectral_loss_p": float(params.spectral_loss_p),
        "spectral_edge_weight_alpha": float(params.spectral_edge_weight_alpha),
        "spectral_edge_weight_power": float(params.spectral_edge_weight_power),
    }


def _kernel_null_field_stack(
    on_axis_field: NDArray[np.complex128],
    aperture_positions_pupil_m: NDArray[np.floating],
    order: int,
) -> NDArray[np.complex128]:
    """Build the kernel-null field stack for the requested order.

    Direct port of KILO ``injection.py:kernel_null_field_stack`` L226-330.
    Returns an ``(n_rows, M, ...)`` stack where row 0 is the on-axis
    field and subsequent rows are pupil-position-weighted variants
    enforcing first / second / third angular derivatives via Taylor
    expansion.

    Each derivative row is L2-equalized to the on-axis row so the
    summed-leakage objective treats all constraints with equal weight.

    Args:
        on_axis_field: shape ``(M,)`` or ``(M, ...)`` — the chip-input
            stellar field on each mode. Extra leading dims are
            preserved (e.g. wavelength, polarization).
        aperture_positions_pupil_m: ``(M, >=2)`` — pupil-frame ENU
            positions for each mode (column 0 = E, 1 = N).
        order: 0 (on-axis only) / 2 (r^4) / 3 (r^6) / 4 (r^8).

    Returns:
        Stack of shape ``(n_rows, M, ...)``. ``n_rows`` is 1 / 3 / 6 /
        10 for order 0 / 2 / 3 / 4 respectively (rows with zero norm
        are skipped, matching KILO).
    """

    if order not in (0, 2, 3, 4):
        raise ValueError(f"kernel_order must be 0, 2, 3, or 4, got {order}")

    on_axis = np.asarray(on_axis_field)
    if on_axis.shape[0] != aperture_positions_pupil_m.shape[0]:
        raise ValueError(
            f"on_axis_field axis 0 ({on_axis.shape[0]}) must match "
            f"aperture_positions_pupil_m rows "
            f"({aperture_positions_pupil_m.shape[0]})."
        )

    if order == 0:
        return on_axis[np.newaxis, ...]

    x_m = np.asarray(aperture_positions_pupil_m[:, 0], dtype=np.float64)
    y_m = np.asarray(aperture_positions_pupil_m[:, 1], dtype=np.float64)
    on_axis_norm = float(np.linalg.norm(on_axis))
    if on_axis_norm <= 0.0:
        return on_axis[np.newaxis, ...]

    weights = [x_m, y_m]
    if order >= 3:
        weights += [x_m * x_m, y_m * y_m, x_m * y_m]
    if order >= 4:
        weights += [
            x_m * x_m * x_m,
            y_m * y_m * y_m,
            x_m * x_m * y_m,
            x_m * y_m * y_m,
        ]

    rows = [on_axis]
    # Broadcast weight (M,) onto on_axis shape (M, ...) — multiply along axis 0.
    weight_shape: tuple[int, ...] = (on_axis.shape[0],) + (1,) * (on_axis.ndim - 1)
    for w in weights:
        row = w.reshape(weight_shape) * on_axis
        norm = float(np.linalg.norm(row))
        if norm <= 0.0:
            # Trivially satisfied (sub-apertures all zero along this axis).
            continue
        rows.append(row * (on_axis_norm / norm))
    return np.stack(rows, axis=0)


def _uses_disk_samples(params: ChipOptimizationParams) -> bool:
    return (
        params.disk_weight > 0.0
        and params.disk_n_rings > 0
        and params.disk_n_azimuths > 0
    )


def _delay_line_diagnostic_payload(state: SimulationState) -> dict[str, Any] | None:
    """Return the newest geometric-delay diagnostic payload, if present."""

    for name, payload in reversed(state.throughput.components.items()):
        if not name.startswith("delay_line_loss"):
            continue
        if not isinstance(payload, dict):
            continue
        per_aperture = payload.get("per_aperture")
        if isinstance(per_aperture, dict):
            return per_aperture
    return None


def _delay_state_azimuths_deg(n_azimuths: int) -> tuple[float, ...]:
    if n_azimuths <= 0:
        return ()
    return tuple(
        float(value)
        for value in np.linspace(0.0, 360.0, int(n_azimuths), endpoint=False)
    )


def _fixed_loss_tuple(raw: Any) -> tuple[float | None, ...] | None:
    if raw is None:
        return None
    return tuple(None if value is None else float(value) for value in raw)


@dataclass(frozen=True)
class _DelayStateReplay:
    n_azimuths: int
    n_modes: int
    n_wavelengths: int
    wavelength_grid_um: tuple[float, ...]
    aperture_positions_enu_m: NDArray[np.float64]
    fiber_lengths_m: NDArray[np.float64]
    modes_per_aperture: int
    phase_index_ref: float
    group_index_ref: float
    dispersion_ps_nm_km: float
    chromatic_control: str
    reference_wavelength_um: float
    attenuation_db_per_m: float
    altitude_deg: float
    current_azimuth_deg: float
    fixed_loss_db: tuple[float | None, ...] | None
    fixed_loss_mode: str


def _delay_state_transport_scale(
    *,
    aperture_positions_enu_m: NDArray[np.float64],
    fiber_lengths_m: NDArray[np.float64],
    modes_per_aperture: int,
    wavelength_grid_um: tuple[float, ...],
    phase_index_ref: float,
    group_index_ref: float,
    dispersion_ps_nm_km: float,
    chromatic_control: str,
    reference_wavelength_um: float,
    attenuation_db_per_m: float,
    altitude_deg: float,
    azimuth_deg: float,
    fixed_loss_db: tuple[float | None, ...] | None,
    fixed_loss_mode: str,
) -> NDArray[np.complex128]:
    """Return per-mode complex transport scale for one delay-line state."""

    from nullsim.stages.delay_lines.geometric import (
        per_aperture_chromatic_phase_rad,
        per_aperture_transmission,
    )

    transmission = per_aperture_transmission(
        aperture_positions_enu_m=aperture_positions_enu_m,
        fiber_lengths_m=fiber_lengths_m,
        n_eff=phase_index_ref,
        attenuation_db_per_m=attenuation_db_per_m,
        altitude_deg=altitude_deg,
        azimuth_deg=azimuth_deg,
        fixed_loss_db=fixed_loss_db,
        fixed_loss_mode=fixed_loss_mode,
    )
    phase = per_aperture_chromatic_phase_rad(
        aperture_positions_enu_m=aperture_positions_enu_m,
        fiber_lengths_m=fiber_lengths_m,
        phase_index_ref=phase_index_ref,
        group_index_ref=group_index_ref,
        dispersion_ps_nm_km=dispersion_ps_nm_km,
        chromatic_control=chromatic_control,
        reference_wavelength_um=reference_wavelength_um,
        wavelength_grid_um=wavelength_grid_um,
        altitude_deg=altitude_deg,
        azimuth_deg=azimuth_deg,
        fixed_loss_db=fixed_loss_db,
        fixed_loss_mode=fixed_loss_mode,
    )
    per_mode_amp = np.sqrt(np.repeat(transmission, modes_per_aperture))
    per_mode_phase = np.repeat(phase, modes_per_aperture, axis=0)
    return (
        per_mode_amp[:, np.newaxis]
        * np.exp(1j * per_mode_phase)
    ).astype(np.complex128)


def _delay_state_replay_from_state(
    *,
    state: SimulationState,
    field_shape: tuple[int, int, int],
    params: ChipOptimizationParams,
    wavelength_grid_um: tuple[float, ...],
) -> _DelayStateReplay | None:
    n_azimuths = int(params.delay_state_n_azimuths)
    if n_azimuths <= 0:
        return None

    if len(field_shape) != 3:
        raise ValueError(
            "delay-state chip optimization requires field amplitudes with "
            f"shape (n_modes, n_wl, n_pol); got {field_shape}."
        )
    if field_shape[1] != len(wavelength_grid_um):
        raise ValueError(
            "delay-state chip optimization requires wavelength_grid_um to "
            "match the field wavelength axis."
        )

    delay_diag = _delay_line_diagnostic_payload(state)
    if delay_diag is None:
        raise ValueError(
            "chip_optimization delay_state_n_azimuths > 0 requires an "
            "upstream delay_lines.geometric diagnostic component."
        )
    if delay_diag.get("per_mode_field_amplitude_applied") is False:
        raise ValueError(
            "chip_optimization delay-state replay requires delay_lines.geometric "
            "to apply per-mode field amplitudes; aggregate-smearing diagnostics "
            "cannot be replayed."
        )

    aperture_positions = state.geometry.aperture_positions_enu_m
    if aperture_positions is None:
        raise ValueError(
            "chip_optimization delay-state replay requires "
            "geometry.aperture_positions_enu_m."
        )
    aperture_positions_enu_m = np.asarray(aperture_positions, dtype=np.float64)
    if aperture_positions_enu_m.ndim != 2 or aperture_positions_enu_m.shape[1] != 3:
        raise ValueError(
            "chip_optimization delay-state replay requires "
            "geometry.aperture_positions_enu_m with shape (n_apertures, 3)."
        )
    n_apertures = aperture_positions_enu_m.shape[0]
    if field_shape[0] % n_apertures != 0:
        raise ValueError(
            "chip_optimization delay-state replay requires n_modes to be an "
            "integer multiple of the physical aperture count."
        )
    modes_per_aperture = field_shape[0] // n_apertures

    fiber_lengths_m = np.asarray(
        delay_diag.get("fiber_lengths_m", ()),
        dtype=np.float64,
    )
    if fiber_lengths_m.shape != (n_apertures,):
        raise ValueError(
            "chip_optimization delay-state replay requires delay_line_loss "
            f"fiber_lengths_m shape ({n_apertures},); got {fiber_lengths_m.shape}."
        )

    phase_index_ref = float(
        delay_diag.get("phase_index_ref", delay_diag.get("n_eff", 1.4682))
    )
    group_index_ref = float(delay_diag.get("group_index_ref", phase_index_ref))
    return _DelayStateReplay(
        n_azimuths=n_azimuths,
        n_modes=field_shape[0],
        n_wavelengths=field_shape[1],
        wavelength_grid_um=wavelength_grid_um,
        aperture_positions_enu_m=aperture_positions_enu_m,
        fiber_lengths_m=fiber_lengths_m,
        modes_per_aperture=modes_per_aperture,
        phase_index_ref=phase_index_ref,
        group_index_ref=group_index_ref,
        dispersion_ps_nm_km=float(delay_diag.get("dispersion_ps_nm_km", 0.0)),
        chromatic_control=str(delay_diag.get("chromatic_control", "none")),
        reference_wavelength_um=float(
            delay_diag.get("reference_wavelength_um", np.median(wavelength_grid_um))
        ),
        attenuation_db_per_m=float(delay_diag.get("attenuation_db_per_m", 0.0)),
        altitude_deg=float(
            delay_diag.get(
                "altitude_deg",
                90.0 if state.scene.altitude_deg is None else state.scene.altitude_deg,
            )
        ),
        current_azimuth_deg=float(
            delay_diag.get(
                "azimuth_deg",
                0.0 if state.scene.azimuth_deg is None else state.scene.azimuth_deg,
            )
        ),
        fixed_loss_db=_fixed_loss_tuple(
            delay_diag.get("fixed_delay_line_loss_db_per_telescope")
        ),
        fixed_loss_mode=str(
            delay_diag.get("fixed_delay_line_loss_mode", "replace_delay")
        ),
    )


def _delay_state_transport_scale_from_replay(
    replay: _DelayStateReplay,
    azimuth_deg: float,
) -> NDArray[np.complex128]:
    return _delay_state_transport_scale(
        aperture_positions_enu_m=replay.aperture_positions_enu_m,
        fiber_lengths_m=replay.fiber_lengths_m,
        modes_per_aperture=replay.modes_per_aperture,
        wavelength_grid_um=replay.wavelength_grid_um,
        phase_index_ref=replay.phase_index_ref,
        group_index_ref=replay.group_index_ref,
        dispersion_ps_nm_km=replay.dispersion_ps_nm_km,
        chromatic_control=replay.chromatic_control,
        reference_wavelength_um=replay.reference_wavelength_um,
        attenuation_db_per_m=replay.attenuation_db_per_m,
        altitude_deg=replay.altitude_deg,
        azimuth_deg=azimuth_deg,
        fixed_loss_db=replay.fixed_loss_db,
        fixed_loss_mode=replay.fixed_loss_mode,
    )


def _delay_state_transport_scales_from_state(
    *,
    state: SimulationState,
    field_shape: tuple[int, int, int],
    params: ChipOptimizationParams,
    wavelength_grid_um: tuple[float, ...],
) -> NDArray[np.complex128] | None:
    """Return absolute delay-line transport scales at optimizer delay states."""

    replay = _delay_state_replay_from_state(
        state=state,
        field_shape=field_shape,
        params=params,
        wavelength_grid_um=wavelength_grid_um,
    )
    if replay is None:
        return None
    scales = np.empty(
        (replay.n_azimuths, replay.n_wavelengths, replay.n_modes),
        dtype=np.complex128,
    )
    for state_index, azimuth_deg in enumerate(
        _delay_state_azimuths_deg(replay.n_azimuths)
    ):
        scale = _delay_state_transport_scale_from_replay(replay, azimuth_deg)
        scales[state_index] = scale.T
    return scales


def _delay_state_current_transport_scale_from_state(
    *,
    state: SimulationState,
    field_shape: tuple[int, int, int],
    params: ChipOptimizationParams,
    wavelength_grid_um: tuple[float, ...],
) -> NDArray[np.complex128] | None:
    """Return the current pointing's delay-line transport scale."""

    replay = _delay_state_replay_from_state(
        state=state,
        field_shape=field_shape,
        params=params,
        wavelength_grid_um=wavelength_grid_um,
    )
    if replay is None:
        return None
    scale = _delay_state_transport_scale_from_replay(
        replay,
        replay.current_azimuth_deg,
    )
    return scale.T


def _throughput_penalty_fields(
    *,
    state: SimulationState,
    field_array: NDArray[np.complex128],
    wavelength_grid_um: tuple[float, ...],
    params: ChipOptimizationParams,
    aperture_positions_pupil_m: NDArray[np.floating] | None,
    delay_state_transport_scales: NDArray[np.complex128] | None = None,
    delay_state_current_transport_scale: NDArray[np.complex128] | None = None,
) -> NDArray[np.complex128] | None:
    """Build planet input fields for the optimizer throughput penalty."""

    if params.throughput_penalty_alpha <= 0.0:
        return None
    if aperture_positions_pupil_m is None:
        raise ValueError(
            "chip_optimization throughput_penalty_alpha > 0 requires "
            "state.geometry.aperture_positions_pupil_m."
        )

    rad_per_mas = math.pi / (180.0 * 3600.0 * 1000.0)
    pa_grid = np.linspace(0.0, 2.0 * np.pi, params.throughput_penalty_n_pa, endpoint=False)
    sep_rad_grid = (
        np.asarray(params.throughput_penalty_seps_mas, dtype=np.float64)
        * rad_per_mas
    )
    mode_areas = _mode_areas_m2(state, params.n_modes)
    if mode_areas is None:
        mode_areas = np.ones(params.n_modes, dtype=np.float64)
        mode_diameters = None
    else:
        # Physical mode diameters set the angular-coupling envelope
        # (Ruilier & Cassaing 2001); a mode splitter divides area between
        # copies but keeps the diameter, so the equal-area circle would
        # overstate off-axis coupling (review 2026-11-07 Issue 20).
        mode_diameters = _mode_diameters_m(state, params.n_modes)

    design_wl_idx = int(
        np.argmin(
            np.abs(
                np.asarray(wavelength_grid_um, dtype=np.float64)
                - float(params.wl_design_um)
            )
        )
    )
    # Sum over polarization arms, matching the leakage term, so a PBS does
    # not halve the reward relative to the leakage (review 2026-11-07 row 29).
    field_power = np.sum(
        np.abs(field_array[:, design_wl_idx, :]) ** 2,
        axis=1,
    )
    if delay_state_current_transport_scale is not None:
        current_transport_power = (
            np.abs(delay_state_current_transport_scale[design_wl_idx]) ** 2
        )
        field_power = np.divide(
            field_power,
            current_transport_power,
            out=np.zeros_like(field_power, dtype=np.float64),
            where=current_transport_power > 0.0,
        )
    coupling_per_mode = np.divide(
        field_power,
        mode_areas,
        out=np.zeros_like(field_power, dtype=np.float64),
        where=mode_areas > 0.0,
    )

    planet_fields_per_wl: list[NDArray[np.complex128]] = []
    positions = np.asarray(aperture_positions_pupil_m, dtype=np.float64)
    for wavelength_index, wl_um in enumerate(wavelength_grid_um):
        rows = []
        wl_m = float(wl_um) * 1.0e-6
        for sep_rad in sep_rad_grid:
            for pa in pa_grid:
                rows.append(
                    planet_input_field(
                        positions[:, :2],
                        coupling_per_mode,
                        mode_areas,
                        mode_diameters,
                        wl_m,
                        float(sep_rad),
                        float(pa),
                    )
                )
        rows_array = np.stack(rows, axis=0)
        if delay_state_transport_scales is not None:
            transport_at_wl = delay_state_transport_scales[:, wavelength_index, :]
            rows_array = (
                rows_array[np.newaxis, :, :]
                * transport_at_wl[:, np.newaxis, :]
            ).reshape(-1, params.n_modes)
        planet_fields_per_wl.append(rows_array)

    return np.ascontiguousarray(
        np.stack(planet_fields_per_wl, axis=0),
        dtype=np.complex128,
    )


def _delay_state_fields_from_state(
    *,
    state: SimulationState,
    field_array: NDArray[np.complex128],
    params: ChipOptimizationParams,
    wavelength_grid_um: tuple[float, ...],
) -> NDArray[np.complex128] | None:
    """Replay geometric-delay transport at representative azimuth states.

    The optimizer input has already passed through ``_require_field()``, which
    intentionally strips upstream phases but keeps the current pointing's
    delay-line attenuation in the per-mode magnitudes. Reconstruct the nominal
    pre-delay amplitude by removing only that attenuation, then apply the
    absolute delay-line complex scale at each sampled azimuth. Dividing by the
    current complex phase would train the mesh on phase ratios relative to the
    current pointing rather than the physical fields the deployed chip sees.
    """

    n_azimuths = int(params.delay_state_n_azimuths)
    if n_azimuths <= 0:
        return None

    field = np.asarray(field_array, dtype=np.complex128)
    replay = _delay_state_replay_from_state(
        state=state,
        field_shape=tuple(int(value) for value in field.shape),
        params=params,
        wavelength_grid_um=wavelength_grid_um,
    )
    if replay is None:
        return None

    current_scale = _delay_state_transport_scale_from_replay(
        replay,
        replay.current_azimuth_deg,
    )
    current_amplitude = np.abs(current_scale)
    if np.any(current_amplitude <= 0.0):
        raise ValueError(
            "chip_optimization delay-state replay cannot reconstruct the "
            "pre-delay field because at least one current transport scale is zero."
        )
    pre_delay_field = field / current_amplitude[:, :, np.newaxis]

    state_fields = np.empty((replay.n_azimuths,) + field.shape, dtype=np.complex128)
    for state_index, azimuth_deg in enumerate(
        _delay_state_azimuths_deg(replay.n_azimuths)
    ):
        scale = _delay_state_transport_scale_from_replay(replay, azimuth_deg)
        state_fields[state_index] = pre_delay_field * scale[:, :, np.newaxis]
    return state_fields


def _normalization_field_array(
    field_array: NDArray[np.complex128],
    delay_state_fields: NDArray[np.complex128] | None,
) -> NDArray[np.complex128]:
    if delay_state_fields is None:
        return field_array
    n_states = int(delay_state_fields.shape[0])
    if n_states <= 0:
        raise ValueError("delay_state_fields must contain at least one state.")
    return delay_state_fields / math.sqrt(float(n_states))


def _delay_state_metadata(
    params: ChipOptimizationParams,
    delay_state_fields: NDArray[np.complex128] | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "delay_state_n_azimuths": int(params.delay_state_n_azimuths),
    }
    if delay_state_fields is None:
        return metadata
    metadata["delay_state_objective"] = "uniform_azimuth_average"
    metadata["delay_state_azimuths_deg"] = _delay_state_azimuths_deg(
        int(delay_state_fields.shape[0])
    )
    metadata["delay_state_field_shape"] = tuple(
        int(value) for value in delay_state_fields.shape
    )
    return metadata


def _optimizer_null_field_stack(
    *,
    on_axis_field: NDArray[np.complex128],
    params: ChipOptimizationParams,
    aperture_positions_pupil_m: NDArray[np.floating] | None,
    wavelength_grid_um: tuple[float, ...],
    sub_diameters_m: NDArray[np.floating] | None,
    stellar_radius_rad: float,
    delay_state_fields: NDArray[np.complex128] | None = None,
) -> NDArray[np.complex128]:
    """Return derivative and finite-disk stellar rows for det-mode nulling."""

    on_axis = np.asarray(on_axis_field, dtype=np.complex128)
    if delay_state_fields is None:
        source_fields = on_axis[np.newaxis, ...]
        state_weight = 1.0
    else:
        source_fields = np.asarray(delay_state_fields, dtype=np.complex128)
        expected_shape = (source_fields.shape[0],) + on_axis.shape
        if source_fields.shape != expected_shape:
            raise ValueError(
                "delay_state_fields must have shape "
                f"(n_states, {on_axis.shape}); got {source_fields.shape}."
            )
        if source_fields.shape[0] <= 0:
            raise ValueError("delay_state_fields must contain at least one state.")
        state_weight = 1.0 / math.sqrt(float(source_fields.shape[0]))

    stacks = [
        _single_optimizer_null_field_stack(
            on_axis_field=source_field,
            params=params,
            aperture_positions_pupil_m=aperture_positions_pupil_m,
            wavelength_grid_um=wavelength_grid_um,
            sub_diameters_m=sub_diameters_m,
            stellar_radius_rad=stellar_radius_rad,
        )
        * state_weight
        for source_field in source_fields
    ]
    return np.concatenate(stacks, axis=0)


def _single_optimizer_null_field_stack(
    *,
    on_axis_field: NDArray[np.complex128],
    params: ChipOptimizationParams,
    aperture_positions_pupil_m: NDArray[np.floating] | None,
    wavelength_grid_um: tuple[float, ...],
    sub_diameters_m: NDArray[np.floating] | None,
    stellar_radius_rad: float,
) -> NDArray[np.complex128]:
    active_disk_samples = _uses_disk_samples(params) and stellar_radius_rad > 0.0
    if (params.kernel_order > 0 or active_disk_samples) and aperture_positions_pupil_m is None:
        raise ValueError(
            f"chip_optimization with kernel_order={params.kernel_order} "
            "or disk samples requires "
            "state.geometry.aperture_positions_pupil_m to be populated by an "
            "upstream stage (e.g. ao.pyramid_wfs). Set kernel_order=0 to "
            "fall back to on-axis-only nulling and disable disk samples."
        )
    if params.kernel_order == 0:
        kernel_stack = np.asarray(on_axis_field, dtype=np.complex128)[np.newaxis, ...]
    else:
        kernel_stack = _kernel_null_field_stack(
            on_axis_field,
            np.asarray(aperture_positions_pupil_m, dtype=np.float64),
            order=params.kernel_order,
        )
    disk_stack = _stellar_disk_field_stack(
        on_axis_field=on_axis_field,
        aperture_positions_pupil_m=aperture_positions_pupil_m,
        wavelength_grid_um=wavelength_grid_um,
        sub_diameters_m=sub_diameters_m,
        stellar_radius_rad=stellar_radius_rad,
        n_rings=params.disk_n_rings,
        n_azimuths=params.disk_n_azimuths,
        disk_weight=params.disk_weight,
    )
    if disk_stack.shape[0] == 0:
        return kernel_stack
    return np.concatenate([kernel_stack, disk_stack], axis=0)


def _stellar_disk_field_stack(
    *,
    on_axis_field: NDArray[np.complex128],
    aperture_positions_pupil_m: NDArray[np.floating],
    wavelength_grid_um: tuple[float, ...],
    sub_diameters_m: NDArray[np.floating] | None,
    stellar_radius_rad: float,
    n_rings: int,
    n_azimuths: int,
    disk_weight: float,
) -> NDArray[np.complex128]:
    """Return L2-equalized finite stellar-disk rows for det-mode nulling."""

    on_axis = np.asarray(on_axis_field, dtype=np.complex128)
    empty_shape = (0,) + on_axis.shape
    if (
        stellar_radius_rad <= 0.0
        or n_rings <= 0
        or n_azimuths <= 0
        or disk_weight <= 0.0
    ):
        return np.empty(empty_shape, dtype=np.complex128)

    positions = np.asarray(aperture_positions_pupil_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[0] != on_axis.shape[0] or positions.shape[1] < 2:
        raise ValueError(
            "stellar disk samples require aperture_positions_pupil_m with shape "
            f"({on_axis.shape[0]}, >=2)."
        )
    if sub_diameters_m is None:
        raise ValueError("stellar disk samples require per-mode sub_diameters_m.")
    diameters = np.asarray(sub_diameters_m, dtype=np.float64)
    if diameters.shape != (on_axis.shape[0],):
        raise ValueError(
            "stellar disk samples require sub_diameters_m with shape "
            f"({on_axis.shape[0]},)."
        )
    if len(wavelength_grid_um) != on_axis.shape[1]:
        raise ValueError(
            "stellar disk samples require wavelength_grid_um to match the "
            "field wavelength axis."
        )

    on_axis_norm = float(np.linalg.norm(on_axis))
    if on_axis_norm <= 0.0:
        return np.empty(empty_shape, dtype=np.complex128)

    weight_scale = math.sqrt(float(disk_weight))
    rows: list[NDArray[np.complex128]] = []
    for ring_index in range(int(n_rings)):
        radius_rad = stellar_radius_rad * (ring_index + 1) / int(n_rings)
        for azimuth_index in range(int(n_azimuths)):
            angle = _TWO_PI * azimuth_index / int(n_azimuths)
            alpha_rad = radius_rad * math.sin(angle)
            beta_rad = radius_rad * math.cos(angle)
            sample = np.empty_like(on_axis, dtype=np.complex128)
            for wavelength_index, wavelength_um in enumerate(wavelength_grid_um):
                wavelength_m = float(wavelength_um) * 1.0e-6
                angular_amplitude = np.asarray(
                    angular_coupling_amplitude(radius_rad, diameters, wavelength_m),
                    dtype=np.float64,
                )
                opd_m = positions[:, 0] * alpha_rad + positions[:, 1] * beta_rad
                phase = _TWO_PI * opd_m / wavelength_m
                sample[:, wavelength_index, :] = (
                    on_axis[:, wavelength_index, :]
                    * angular_amplitude[:, np.newaxis]
                    * np.exp(1j * phase)[:, np.newaxis]
                )
            for part in (sample.real, sample.imag):
                norm = float(np.linalg.norm(part))
                if norm > 0.0:
                    rows.append(
                        (weight_scale * part * (on_axis_norm / norm)).astype(
                            np.complex128
                        )
                    )
    if not rows:
        return np.empty(empty_shape, dtype=np.complex128)
    return np.stack(rows, axis=0)


def _total_input_power(field_array: NDArray[np.complex128]) -> float:
    """Total stellar power summed over (mode, wavelength, pol)."""

    total = float(np.sum(np.abs(field_array) ** 2))
    if total <= 0.0:
        raise ValueError(
            "chip_optimization input field has zero total power; "
            "cannot define a normalized null depth."
        )
    return total


def _normalized_loss_for_target(
    loss: float,
    field_array: NDArray[np.complex128],
) -> float:
    """Dimensionless null-depth value used for target-null comparisons."""

    return float(loss) / _total_input_power(field_array)


def _reported_null_depth_for_target(
    loss: float,
    field_array: NDArray[np.complex128],
    *,
    n_objective_rows: int = 1,
    n_states: int = 1,
) -> float:
    """Normalize a spectrally reduced leakage objective as a null depth."""

    denominator = _reported_null_depth_denominator(
        field_array,
        n_objective_rows=n_objective_rows,
        n_states=n_states,
    )
    return float(loss) / denominator


def _reported_null_depth_denominator(
    field_array: NDArray[np.complex128],
    *,
    n_objective_rows: int = 1,
    n_states: int = 1,
) -> float:
    """Return the denominator for the reported target-null convention.

    With delay-state replay active the objective rows are weighted by
    ``1/sqrt(n_states)`` (so the summed leakage loss is a per-state MEAN,
    not a sum over states) and the 4-d normalization field is divided by
    ``sqrt(n_states)`` (so its power is the mean per-state power). The
    denominator must therefore count rows PER STATE: using the
    state-multiplied stacked row count double-counted the state
    multiplicity and deflated every reported null by exactly
    ``n_states`` — 6x optimistic in kilo_keck_6 (review 2026-06-10
    Issue 1).
    """

    field = np.asarray(field_array, dtype=np.complex128)
    if field.ndim == 3:
        power_per_wavelength = np.sum(np.abs(field) ** 2, axis=(0, 2))
    elif field.ndim == 4:
        power_per_wavelength = np.sum(np.abs(field) ** 2, axis=(0, 1, 3))
    else:
        raise ValueError(
            "_reported_null_depth_for_target expects field_array with shape "
            "(n_modes, n_wavelengths, n_pol) or "
            f"(n_states, n_modes, n_wavelengths, n_pol); got {field.shape}."
        )
    if int(n_objective_rows) < 1:
        raise ValueError("n_objective_rows must be >= 1.")
    if int(n_states) < 1:
        raise ValueError("n_states must be >= 1.")
    if int(n_objective_rows) % int(n_states) != 0:
        raise ValueError(
            f"n_objective_rows ({n_objective_rows}) must be a multiple of "
            f"n_states ({n_states}): the objective stack holds the same row "
            "set repeated once per delay state."
        )
    rows_per_state = int(n_objective_rows) // int(n_states)
    mean_power = float(np.mean(power_per_wavelength))
    if mean_power <= 0.0:
        raise ValueError(
            "chip_optimization input field has zero mean per-wavelength power; "
            "cannot define a reported null depth."
        )
    return mean_power * float(rows_per_state)


def _loss_meets_target(
    loss: float,
    field_array: NDArray[np.complex128],
    target_null_depth: float,
    *,
    n_objective_rows: int = 1,
    n_states: int = 1,
) -> bool:
    return (
        _reported_null_depth_for_target(
            loss,
            field_array,
            n_objective_rows=n_objective_rows,
            n_states=n_states,
        )
        < float(target_null_depth)
    )


def _write_results(
    *,
    state: SimulationState,
    instance_id: str,
    optimal_params: NDArray[np.float64],
    loss_final: float,
    iterations: int,
    backend_name: str,
    kernel_rescales_per_wavelength: bool,
    convergence_history: tuple[tuple[int, float], ...] | None = None,
    convergence_history_normalized_null: tuple[tuple[int, float], ...] | None = None,
    optimal_params_unquantized: NDArray[np.float64] | None = None,
    optimal_u_stack: NDArray[np.complex128] | None = None,
    n_restarts_completed: int | None = None,
    normalized_loss_final: float | None = None,
    backend_metadata: dict[str, Any] | None = None,
) -> SimulationState:
    """Persist optimizer output to ``state.results.components``.

    The ``kernel_rescales_per_wavelength`` flag controls how
    ``chip.kernel_mzi_mesh`` consumes the params. When ``True`` (scipy
    backend, whose params are valid under the kernel's per-wavelength
    ``phi * wl_design/wl`` rescaling because both use the same Clements
    layout), the kernel scales angles per wavelength as usual. When
    ``False`` (torch backends, whose canonical params produced by
    ``_canonicalize_clements_params`` only match the optimized unitary
    at ``wl_design``), the kernel uses the angles verbatim across all
    bins — chip behaves monochromatic. Ultra-review High #5 fix.

    Tier 3.1 (option 2): when ``optimal_u_stack`` is provided, the
    kernel **prefers it** and ignores the rescale flag. The U_stack is
    the chromatic chip the optimizer actually evaluated (KILO column-
    alternating layout + per-λ phase rescaling), so the deployed chip matches
    the optimizer at every detector wavelength.

    ``convergence_history`` holds ``(nfev, objective)`` pairs; the objective
    is dimensional and can be negative. ``convergence_history_normalized_null``
    holds ``(nfev, null)`` pairs in the convention of
    ``optimization_normalized_null_final``. ``optimal_params_unquantized`` is
    the pre-DAC optimum in the same layout as ``optimal_chip_params``.
    """

    components = dict(state.results.components)
    payload: dict[str, Any] = {
        "optimal_chip_params": tuple(float(value) for value in optimal_params),
        "optimization_loss_final": float(loss_final),
        "optimization_iterations": int(iterations),
        "backend_name": backend_name,
        "kernel_rescales_per_wavelength": bool(kernel_rescales_per_wavelength),
    }
    if normalized_loss_final is not None:
        payload["optimization_normalized_null_final"] = float(normalized_loss_final)
    if n_restarts_completed is not None:
        payload["n_restarts_completed"] = int(n_restarts_completed)
    if optimal_u_stack is not None:
        payload["optimal_u_stack"] = np.asarray(
            optimal_u_stack, dtype=np.complex128,
        ).copy()
    if backend_metadata is not None:
        payload.update(dict(backend_metadata))
    if convergence_history is not None:
        payload["convergence_history"] = tuple(
            (int(nfev), float(loss)) for nfev, loss in convergence_history
        )
    if convergence_history_normalized_null is not None:
        payload["convergence_history_normalized_null"] = tuple(
            (int(nfev), float(null)) for nfev, null in convergence_history_normalized_null
        )
    if optimal_params_unquantized is not None:
        payload["optimal_chip_params_unquantized"] = np.array(
            optimal_params_unquantized, dtype=np.float64,
        )
    name = unique_component_name("chip_optimization", components, instance_id)
    components[name] = payload
    results = replace(state.results, components=components)
    return replace(state, results=results)


__all__ = [
    "ChipOptimizationParams",
    "_build_unitary_stack_numpy",
    "_clements_order",
    "_chip_chromaticity_metadata",
    "_dark_indices",
    "_delay_state_current_transport_scale_from_state",
    "_delay_state_fields_from_state",
    "_delay_state_metadata",
    "_delay_state_transport_scales_from_state",
    "_null_indices",
    "_optimizer_null_field_stack",
    "_phase_scale_factors_numpy",
    "_kernel_null_field_stack",
    "_loss_numpy",
    "_mode_diameters_m",
    "_mode_areas_m2",
    "_n_params",
    "_optimizer_wavelength_metadata",
    "_optimizer_wavelength_subset",
    "_loss_meets_target",
    "_normalized_loss_for_target",
    "_reported_null_depth_for_target",
    "_reported_null_depth_denominator",
    "_normalization_field_array",
    "_edge_weighted_spectral_weights_numpy",
    "_reduce_spectral_leakage_numpy",
    "_quantize_phases",
    "_require_field",
    "_require_wavelength_grid_um",
    "_spectral_loss_metadata",
    "_stellar_radius_rad",
    "_throughput_penalty_fields",
    "_total_input_power",
    "_theta_offsets_numpy",
    "_validate_chip_chromaticity_params",
    "_uses_disk_samples",
    "_write_results",
]
