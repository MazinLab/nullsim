"""Bakeoff: compare nullsim's chip-optimization backends on this machine.

The chip optimizer is the slowest stage in nullsim and has two backends —
numpy/scipy and PyTorch (CPU or CUDA) — whose published speedups have
historically been hard to compare because each backend uses a different
gradient method, dtype, and (in some cases) loss formulation.

This script benchmarks every available backend on the same problem (same RNG
seed, same field stack, same wavelength grid) and reports two complementary
timings:

  Tier 1 (microbench): a single objective + gradient evaluation, median of N
    repeats with warmup. Reveals raw kernel speed independent of optimizer.

  Tier 2 (end-to-end): the actual ChipOptimization*.apply() call to a fixed
    iter cap. Reveals what the user experiences in practice; includes
    one-time compile-tracing cost for torch.compile paths.

CUDA rows are emitted automatically when torch.cuda.is_available(); otherwise
skipped.

Run from the project root:

    conda run -n py313 python scripts/bench_chip_optimizer.py
    conda run -n py313 python scripts/bench_chip_optimizer.py --modes 24 \\
        --n-wl 5 --max-iter 50 --repeats 20

Defaults match the Keck broadband problem (M=24 modes, n_wl=5 wavelengths,
kernel_order=0). For kernel_order > 0 see ``--kernel-order``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from nullsim.pipeline import Field, RunContext, SimulationState, Throughput
from nullsim.stages.chip_optimization import (
    ChipOptimizationParams,
    _dark_indices,
    _loss_numpy,
    _n_params,
)
from nullsim.stages.chip_optimization.scipy import ChipOptimizationScipy


# ---------------------------------------------------------------------------
# availability detection
# ---------------------------------------------------------------------------


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        from nullsim.stages.chip_optimization import torch as _backend  # noqa: F401
    except Exception:
        return False
    return True


def _torch_cuda_available() -> bool:
    if not _torch_available():
        return False
    import torch

    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# problem definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchProblem:
    n_modes: int = 24
    n_dark_ports: int = 18
    n_wl: int = 5
    wl_design_um: float = 1.65
    wl_min_um: float = 1.55
    wl_max_um: float = 1.75
    kernel_order: int = 0
    seed: int = 0

    @property
    def n_bright_ports(self) -> int:
        return self.n_modes - self.n_dark_ports

    @property
    def n_params(self) -> int:
        return _n_params(self.n_modes)


def _build_state(prob: BenchProblem) -> SimulationState:
    """Multi-wavelength uniform-amplitude state (Keck-shaped, on-axis)."""
    wavelength_grid_um = tuple(
        float(w) for w in np.linspace(prob.wl_min_um, prob.wl_max_um, prob.n_wl)
    )
    amplitudes = np.ones(
        (prob.n_modes, prob.n_wl, 1), dtype=np.complex128
    ) / np.sqrt(prob.n_modes)
    state = SimulationState.empty({"wavelength_grid_um": wavelength_grid_um})
    field_obj = Field(
        amplitudes=amplitudes,
        wavelength_grid_um=wavelength_grid_um,
        mode_labels=tuple(f"mode_{i}" for i in range(prob.n_modes)),
    )
    throughput = Throughput(
        wavelength_grid_um=wavelength_grid_um,
        transmission_fraction=tuple(1.0 for _ in wavelength_grid_um),
    )
    return replace(state, field=field_obj, throughput=throughput)


def _params_kwargs(
    prob: BenchProblem, *, max_iter: int, n_restarts: int
) -> dict[str, Any]:
    return dict(
        n_modes=prob.n_modes,
        n_bright_ports=prob.n_bright_ports,
        n_dark_ports=prob.n_dark_ports,
        wl_design_um=prob.wl_design_um,
        max_iter=max_iter,
        n_restarts=n_restarts,
        seed=prob.seed,
        kernel_order=prob.kernel_order,
        single_lambda_bootstrap=False,  # bench the broadband solve directly
    )


def _params_for_loss(prob: BenchProblem) -> ChipOptimizationParams:
    return ChipOptimizationParams(
        **_params_kwargs(prob, max_iter=1, n_restarts=1)
    )


def _run_context(instance_id: str, *, run_dir: Path, seed: int) -> RunContext:
    return RunContext(
        rng=np.random.default_rng(seed),
        config_hash=f"bench-{instance_id}",
        sweep_coord_tuple=(),
        instance_id=instance_id,
        run_dir=run_dir,
    )


# ---------------------------------------------------------------------------
# tier-1 closures: one (obj, grad) pair per backend, kernel_order=0 only
# ---------------------------------------------------------------------------


def _build_scipy_obj_grad(
    prob: BenchProblem, state: SimulationState
) -> tuple[Callable[[NDArray[np.float64]], float], Callable[[NDArray[np.float64]], NDArray[np.float64]]]:
    """numpy + 2-point finite-difference (mirrors scipy backend's FD)."""
    from scipy.optimize import approx_fprime

    params = _params_for_loss(prob)
    field_array = np.ascontiguousarray(state.field.amplitudes)
    wl_grid = tuple(float(w) for w in state.field.wavelength_grid_um)
    dark_idx = _dark_indices(params)
    eps = math.sqrt(np.finfo(np.float64).eps)

    def obj(x: NDArray[np.float64]) -> float:
        return _loss_numpy(
            x,
            field_array=field_array,
            wavelength_grid_um=wl_grid,
            params=params,
            dark_idx=dark_idx,
            aperture_positions_pupil_m=None,
        )

    def grad(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return approx_fprime(x, obj, eps)

    return obj, grad


def _build_torch_obj_grad(
    prob: BenchProblem,
    state: SimulationState,
    *,
    device: str,
    compile_mode: bool,
):
    """Mirrors the kernel_order=0 path of ChipOptimizationTorch.apply()."""
    import torch
    from nullsim.stages.chip_optimization.torch import _clements_unitary_ri

    torch_device = torch.device(device)
    field_array = np.ascontiguousarray(state.field.amplitudes)
    wl_grid = tuple(float(w) for w in state.field.wavelength_grid_um)
    n_modes = prob.n_modes
    n_mzi = n_modes * (n_modes - 1) // 2
    n_wl = len(wl_grid)
    dark_idx_np = np.arange(prob.n_bright_ports, n_modes, dtype=np.int64)

    field_r = torch.from_numpy(
        np.ascontiguousarray(field_array.real, dtype=np.float64)
    ).to(torch_device)
    field_i = torch.from_numpy(
        np.ascontiguousarray(field_array.imag, dtype=np.float64)
    ).to(torch_device)
    scales = torch.tensor(
        [prob.wl_design_um / float(wl) for wl in wl_grid],
        dtype=torch.float64,
        device=torch_device,
    )
    idx_t = torch.from_numpy(dark_idx_np).to(torch_device)
    inv_norm = 1.0 / float(n_wl)

    def _objective(params_t: Any) -> Any:
        total = torch.zeros((), device=torch_device, dtype=params_t.dtype)
        for k in range(n_wl):
            scale = scales[k]
            scaled = params_t.clone()
            scaled[n_mzi : 2 * n_mzi] = scaled[n_mzi : 2 * n_mzi] * scale
            scaled[2 * n_mzi : 2 * n_mzi + n_modes] = (
                scaled[2 * n_mzi : 2 * n_mzi + n_modes] * scale
            )
            u_r, u_i = _clements_unitary_ri(scaled, n_modes, torch_mod=torch)
            f_r = field_r[:, k, :]
            f_i = field_i[:, k, :]
            out_r = u_r @ f_r - u_i @ f_i
            out_i = u_r @ f_i + u_i @ f_r
            leak = torch.sum(out_r[idx_t, :] ** 2 + out_i[idx_t, :] ** 2)
            total = total + leak * inv_norm
        return total

    if compile_mode:
        _obj_fn = torch.compile(_objective, mode="reduce-overhead", dynamic=False)
    else:
        _obj_fn = _objective

    def _sync() -> None:
        if device == "cuda":
            torch.cuda.synchronize()

    def obj(x: NDArray[np.float64]) -> float:
        with torch.no_grad():
            p_t = torch.from_numpy(np.asarray(x, dtype=np.float64)).to(torch_device)
            val = _obj_fn(p_t)
            _sync()
            return float(val.item())

    def grad(x: NDArray[np.float64]) -> NDArray[np.float64]:
        p_t = (
            torch.from_numpy(np.asarray(x, dtype=np.float64))
            .to(torch_device)
            .requires_grad_(True)
        )
        loss = _obj_fn(p_t)
        loss.backward()
        _sync()
        return p_t.grad.detach().cpu().numpy().astype(np.float64)

    return obj, grad



# ---------------------------------------------------------------------------
# timing helpers
# ---------------------------------------------------------------------------


def _time_grad(
    grad_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x0: NDArray[np.float64],
    *,
    n_warmup: int,
    n_repeat: int,
) -> dict[str, float]:
    """Return cold-first, median, IQR (s) for n_repeat grad evals after warmup.

    Reports the very first call separately as ``cold_first_s`` so torch.compile
    tracing cost doesn't contaminate the steady-state median.
    """
    rng = np.random.default_rng(0)

    t0 = time.perf_counter()
    _ = grad_fn(x0)
    cold_first_s = time.perf_counter() - t0

    for _ in range(max(0, n_warmup - 1)):
        x_w = rng.uniform(0.0, 2.0 * math.pi, x0.shape).astype(np.float64)
        _ = grad_fn(x_w)

    samples_s: list[float] = []
    for _ in range(n_repeat):
        x_r = rng.uniform(0.0, 2.0 * math.pi, x0.shape).astype(np.float64)
        t = time.perf_counter()
        _ = grad_fn(x_r)
        samples_s.append(time.perf_counter() - t)

    samples_s.sort()
    med = statistics.median(samples_s)
    q1 = samples_s[len(samples_s) // 4]
    q3 = samples_s[(3 * len(samples_s)) // 4]
    return {
        "cold_first_s": cold_first_s,
        "median_s": med,
        "iqr_s": max(0.0, q3 - q1),
        "min_s": min(samples_s),
        "max_s": max(samples_s),
    }


def _final_loss_from_state(state: SimulationState) -> tuple[float, int]:
    matches = [
        payload
        for name, payload in state.results.components.items()
        if name.startswith("chip_optimization")
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one chip_optimization payload, found {len(matches)}"
        )
    payload = matches[0]
    return (
        float(payload["optimization_loss_final"]),
        int(payload.get("optimization_iterations", 0)),
    )


# ---------------------------------------------------------------------------
# backend registry
# ---------------------------------------------------------------------------


@dataclass
class BackendSpec:
    name: str
    device: str
    dtype: str
    compile_mode: str
    build_tier1: Callable[[BenchProblem, SimulationState], tuple[Callable, Callable]] | None
    end_to_end_stage: Callable[..., Any] | None
    end_to_end_overrides: dict[str, Any]
    available_fn: Callable[[], bool]
    skipped_reason: str | None = None


@dataclass
class BackendResult:
    name: str
    device: str
    dtype: str
    compile_mode: str
    available: bool
    skipped_reason: str | None = None
    cold_first_s: float | None = None
    median_per_grad_s: float | None = None
    iqr_per_grad_s: float | None = None
    end_to_end_s: float | None = None
    final_loss: float | None = None
    iterations: int | None = None


def _build_backend_specs(prob: BenchProblem) -> list[BackendSpec]:
    # Lazy: importing the torch backend at module scope would make the whole
    # benchmark script unimportable on a machine without torch.
    from nullsim.stages.chip_optimization.torch import ChipOptimizationTorch

    specs: list[BackendSpec] = []

    specs.append(
        BackendSpec(
            name="numpy/scipy",
            device="cpu",
            dtype="f64",
            compile_mode="—",
            build_tier1=_build_scipy_obj_grad,
            end_to_end_stage=ChipOptimizationScipy,
            end_to_end_overrides={},
            available_fn=lambda: True,
        )
    )

    if _torch_available():
        specs.append(
            BackendSpec(
                name="torch",
                device="cpu",
                dtype="f64",
                compile_mode="eager",
                build_tier1=lambda p, s: _build_torch_obj_grad(
                    p, s, device="cpu", compile_mode=False
                ),
                end_to_end_stage=ChipOptimizationTorch,
                end_to_end_overrides={"compile_objective": False},
                available_fn=_torch_available,
            )
        )
        specs.append(
            BackendSpec(
                name="torch",
                device="cpu",
                dtype="f64",
                compile_mode="torch.compile",
                build_tier1=lambda p, s: _build_torch_obj_grad(
                    p, s, device="cpu", compile_mode=True
                ),
                end_to_end_stage=ChipOptimizationTorch,
                end_to_end_overrides={"compile_objective": True},
                available_fn=_torch_available,
            )
        )
    else:
        specs.append(
            BackendSpec(
                name="torch",
                device="cpu",
                dtype="f64",
                compile_mode="eager",
                build_tier1=None,
                end_to_end_stage=None,
                end_to_end_overrides={},
                available_fn=lambda: False,
                skipped_reason="PyTorch not importable",
            )
        )

    if _torch_cuda_available():
        specs.append(
            BackendSpec(
                name="torch",
                device="cuda",
                dtype="f64",
                compile_mode="eager",
                build_tier1=lambda p, s: _build_torch_obj_grad(
                    p, s, device="cuda", compile_mode=False
                ),
                end_to_end_stage=ChipOptimizationTorch,
                end_to_end_overrides={"compile_objective": False},
                available_fn=_torch_cuda_available,
            )
        )
        specs.append(
            BackendSpec(
                name="torch",
                device="cuda",
                dtype="f64",
                compile_mode="torch.compile",
                build_tier1=lambda p, s: _build_torch_obj_grad(
                    p, s, device="cuda", compile_mode=True
                ),
                end_to_end_stage=ChipOptimizationTorch,
                end_to_end_overrides={"compile_objective": True},
                available_fn=_torch_cuda_available,
            )
        )
    else:
        specs.append(
            BackendSpec(
                name="torch",
                device="cuda",
                dtype="f64",
                compile_mode="eager",
                build_tier1=None,
                end_to_end_stage=None,
                end_to_end_overrides={},
                available_fn=lambda: False,
                skipped_reason="CUDA not available",
            )
        )

    return specs


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def _filter_specs(
    specs: list[BackendSpec], wanted: list[str] | None
) -> list[BackendSpec]:
    if not wanted:
        return specs
    norm = {w.lower() for w in wanted}

    def label(s: BackendSpec) -> str:
        return f"{s.name}-{s.device}-{s.compile_mode}".lower()

    return [
        s for s in specs if s.name.lower() in norm or label(s) in norm
    ]


def _run_microbench(
    spec: BackendSpec,
    prob: BenchProblem,
    state: SimulationState,
    *,
    n_warmup: int,
    n_repeat: int,
) -> dict[str, float]:
    obj, grad = spec.build_tier1(prob, state)
    rng = np.random.default_rng(prob.seed)
    x0 = rng.uniform(0.0, 2.0 * math.pi, prob.n_params).astype(np.float64)
    # sanity: call obj once to make sure forward path runs
    _ = obj(x0)
    return _time_grad(grad, x0, n_warmup=n_warmup, n_repeat=n_repeat)


def _run_end_to_end(
    spec: BackendSpec,
    prob: BenchProblem,
    *,
    max_iter: int,
    n_restarts: int,
    run_dir: Path,
) -> dict[str, float]:
    kwargs = _params_kwargs(prob, max_iter=max_iter, n_restarts=n_restarts)
    kwargs.update(spec.end_to_end_overrides)
    label = f"{spec.name}-{spec.device}-{spec.compile_mode}".replace(".", "_")
    ctx = _run_context(label, run_dir=run_dir, seed=prob.seed)
    stage = spec.end_to_end_stage(kwargs, ctx)
    state = _build_state(prob)

    t = time.perf_counter()
    out = stage.apply(state)
    wall_s = time.perf_counter() - t

    loss, iters = _final_loss_from_state(out)
    return {"wall_s": wall_s, "final_loss": loss, "iterations": iters}


def run_bakeoff(
    prob: BenchProblem,
    *,
    max_iter: int,
    n_restarts: int,
    n_warmup: int,
    n_repeat: int,
    do_microbench: bool,
    do_end_to_end: bool,
    backends_filter: list[str] | None,
    quiet: bool,
) -> list[BackendResult]:
    specs = _filter_specs(_build_backend_specs(prob), backends_filter)
    state = _build_state(prob)

    results: list[BackendResult] = []
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        for spec in specs:
            r = BackendResult(
                name=spec.name,
                device=spec.device,
                dtype=spec.dtype,
                compile_mode=spec.compile_mode,
                available=spec.available_fn(),
                skipped_reason=spec.skipped_reason,
            )
            label = f"{spec.name}/{spec.device}/{spec.compile_mode}"
            if not r.available:
                if not quiet:
                    print(f"[skip] {label}: {r.skipped_reason}", file=sys.stderr)
                results.append(r)
                continue

            if do_microbench and spec.build_tier1 is not None:
                if not quiet:
                    print(f"[tier1] {label} ...", file=sys.stderr, flush=True)
                try:
                    t1 = _run_microbench(
                        spec,
                        prob,
                        state,
                        n_warmup=n_warmup,
                        n_repeat=n_repeat,
                    )
                    r.cold_first_s = t1["cold_first_s"]
                    r.median_per_grad_s = t1["median_s"]
                    r.iqr_per_grad_s = t1["iqr_s"]
                except Exception as e:
                    r.skipped_reason = f"tier1 failed: {e!r}"
                    if not quiet:
                        print(f"  → FAIL: {e!r}", file=sys.stderr)

            if do_end_to_end and spec.end_to_end_stage is not None:
                if not quiet:
                    print(f"[tier2] {label} ...", file=sys.stderr, flush=True)
                try:
                    t2 = _run_end_to_end(
                        spec,
                        prob,
                        max_iter=max_iter,
                        n_restarts=n_restarts,
                        run_dir=run_dir,
                    )
                    r.end_to_end_s = t2["wall_s"]
                    r.final_loss = t2["final_loss"]
                    r.iterations = t2["iterations"]
                except Exception as e:
                    if r.skipped_reason is None:
                        r.skipped_reason = f"tier2 failed: {e!r}"
                    if not quiet:
                        print(f"  → FAIL: {e!r}", file=sys.stderr)

            results.append(r)
    return results


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def _fmt_s(v: float | None) -> str:
    if v is None:
        return "—"
    if v < 1e-3:
        return f"{v * 1e6:.1f} μs"
    if v < 1.0:
        return f"{v * 1e3:.2f} ms"
    return f"{v:.2f} s"


def _fmt_loss(v: float | None) -> str:
    return "—" if v is None else f"{v:.3e}"


def print_table(results: list[BackendResult], prob: BenchProblem) -> None:
    print()
    print(f"  Problem: M={prob.n_modes}, n_wl={prob.n_wl}, "
          f"kernel_order={prob.kernel_order}, n_dark={prob.n_dark_ports}, "
          f"n_params={prob.n_params}")
    print()
    hdr = (
        "  backend / device / compile      dtype  cold-first   per-grad (med)  ±IQR        "
        "end-to-end   final loss   iters"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in results:
        label = f"{r.name}/{r.device}/{r.compile_mode}"
        if not r.available or r.skipped_reason:
            reason = r.skipped_reason or "unavailable"
            print(f"  {label:32s} {r.dtype:5s}  [skipped] {reason}")
            continue
        print(
            f"  {label:32s} {r.dtype:5s}  "
            f"{_fmt_s(r.cold_first_s):10s}  "
            f"{_fmt_s(r.median_per_grad_s):14s}  "
            f"{_fmt_s(r.iqr_per_grad_s):10s}  "
            f"{_fmt_s(r.end_to_end_s):10s}  "
            f"{_fmt_loss(r.final_loss):11s}  "
            f"{r.iterations if r.iterations is not None else '—'}"
        )
    print()


def make_recommendation(results: list[BackendResult]) -> str:
    runnable_any = [r for r in results if r.available and r.skipped_reason is None]
    if not runnable_any:
        return "No runnable backends — check installation."

    lines: list[str] = []

    with_grad = [r for r in runnable_any if r.median_per_grad_s is not None]
    if with_grad:
        fastest_grad = min(with_grad, key=lambda r: r.median_per_grad_s)
        lines.append(
            f"  Fastest per-grad: {fastest_grad.name}/{fastest_grad.device}/"
            f"{fastest_grad.compile_mode} at {_fmt_s(fastest_grad.median_per_grad_s)}"
        )

    e2e = [r for r in runnable_any if r.end_to_end_s is not None]
    if e2e:
        fastest_e2e = min(e2e, key=lambda r: r.end_to_end_s)
        lines.append(
            f"  Fastest end-to-end: {fastest_e2e.name}/{fastest_e2e.device}/"
            f"{fastest_e2e.compile_mode} at {_fmt_s(fastest_e2e.end_to_end_s)}"
        )

    # Crossover heuristic: for each torch.compile row, find the matching eager
    # row on the same device and estimate how many grad evals justify the
    # cold-first compile cost.
    compile_rows = [r for r in with_grad if "compile" in r.compile_mode]
    for cr in compile_rows:
        peer = next(
            (
                r
                for r in with_grad
                if r.name == cr.name
                and r.device == cr.device
                and "compile" not in r.compile_mode
            ),
            None,
        )
        if (
            peer is not None
            and cr.cold_first_s is not None
            and peer.median_per_grad_s > cr.median_per_grad_s
        ):
            diff = peer.median_per_grad_s - cr.median_per_grad_s
            breakeven = cr.cold_first_s / max(diff, 1e-9)
            lines.append(
                f"  {cr.name}/{cr.device}/{cr.compile_mode}: cold-first "
                f"{_fmt_s(cr.cold_first_s)} pays back vs eager after "
                f"~{breakeven:.0f} grad evals "
                f"({_fmt_s(cr.median_per_grad_s)} vs {_fmt_s(peer.median_per_grad_s)} steady-state). "
                "Inductor caches the compiled graph to disk, so the cold "
                "cost is incurred ONCE per machine, not per process."
            )

    return "\n".join(lines)


def check_correctness(
    results: list[BackendResult], *, rel_tol: float = 0.10, abs_floor: float = 1e-6
) -> str | None:
    """Compare final losses across backends; return warning string if any disagree.

    Skips numpy/scipy at low iter caps: scipy uses finite-difference gradients
    and genuinely converges slower per BFGS iteration than the autodiff
    backends. Disagreement there is expected, not a bug.
    """
    with_loss = [r for r in results if r.final_loss is not None]
    if len(with_loss) < 2:
        return None

    f64_autodiff = [
        r.final_loss
        for r in with_loss
        if r.dtype == "f64" and r.name != "numpy/scipy"
    ]
    if not f64_autodiff:
        return None
    best = min(f64_autodiff)
    warnings: list[str] = []
    for r in with_loss:
        if r.name == "numpy/scipy":
            continue  # scipy w/ FD expected to lag — flagged in notes, not here
        if r.dtype == "f32" and r.final_loss > max(abs_floor, best * 100):
            warnings.append(
                f"  {r.name}/{r.device} (f32) loss {r.final_loss:.3e} far from "
                f"best f64 {best:.3e} — check dtype floor / kernel-null"
            )
        elif r.dtype == "f64":
            if best > 0 and (r.final_loss - best) / best > rel_tol:
                warnings.append(
                    f"  {r.name}/{r.device}/{r.compile_mode} loss "
                    f"{r.final_loss:.3e} >{rel_tol*100:.0f}% above best f64 "
                    f"{best:.3e}"
                )
    return "\n".join(warnings) if warnings else None


def scipy_note(results: list[BackendResult]) -> str | None:
    """If scipy ran and lags the autodiff backends, explain why."""
    scipy_r = next((r for r in results if r.name == "numpy/scipy"), None)
    if scipy_r is None or scipy_r.final_loss is None:
        return None
    autodiff = [
        r.final_loss
        for r in results
        if r.name != "numpy/scipy"
        and r.dtype == "f64"
        and r.final_loss is not None
    ]
    if not autodiff:
        return None
    best = min(autodiff)
    if scipy_r.final_loss <= best * 2:
        return None
    return (
        f"  numpy/scipy reached {scipy_r.final_loss:.3e} vs autodiff "
        f"{best:.3e}. Expected at low --max-iter: scipy uses 2-point FD, "
        "so each BFGS iter costs ~(N+1) forward passes and the noisier "
        "gradient hurts BFGS line-search basin selection. Raise --max-iter "
        "or use an autodiff backend."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Chip-optimizer backend bakeoff for nullsim.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--modes", type=int, default=24)
    ap.add_argument("--n-wl", type=int, default=5)
    ap.add_argument("--n-dark", type=int, default=18)
    ap.add_argument("--wl-design-um", type=float, default=1.65)
    ap.add_argument("--wl-min-um", type=float, default=1.55)
    ap.add_argument("--wl-max-um", type=float, default=1.75)
    ap.add_argument("--kernel-order", type=int, default=0, choices=(0, 2, 3, 4))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-iter", type=int, default=50,
                    help="end-to-end iter cap per restart")
    ap.add_argument("--n-restarts", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--backends", nargs="+", default=None,
                    help="filter by name(s); e.g. 'torch-cpu-eager' or 'numpy/scipy'")
    ap.add_argument("--no-microbench", action="store_true")
    ap.add_argument("--no-end-to-end", action="store_true")
    ap.add_argument("--output-json", type=Path, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.kernel_order != 0:
        print(
            "warning: kernel_order>0 requires aperture_positions_pupil_m on the "
            "state and is not supported by this bench script's synthetic "
            "uniform state — only the scipy/torch end-to-end paths will be "
            "exercised, and even then only at kernel_order=0. Run with "
            "--kernel-order 0 for the bakeoff.",
            file=sys.stderr,
        )

    prob = BenchProblem(
        n_modes=args.modes,
        n_dark_ports=args.n_dark,
        n_wl=args.n_wl,
        wl_design_um=args.wl_design_um,
        wl_min_um=args.wl_min_um,
        wl_max_um=args.wl_max_um,
        kernel_order=0,  # bench script always at 0 (synthetic uniform state)
        seed=args.seed,
    )

    t0 = time.perf_counter()
    results = run_bakeoff(
        prob,
        max_iter=args.max_iter,
        n_restarts=args.n_restarts,
        n_warmup=args.warmup,
        n_repeat=args.repeats,
        do_microbench=not args.no_microbench,
        do_end_to_end=not args.no_end_to_end,
        backends_filter=args.backends,
        quiet=args.quiet,
    )
    elapsed_s = time.perf_counter() - t0

    print_table(results, prob)
    rec = make_recommendation(results)
    if rec:
        print("Recommendation:")
        print(rec)
        print()
    warn = check_correctness(results)
    if warn:
        print("Cross-backend correctness warnings:")
        print(warn)
        print()
    snote = scipy_note(results)
    if snote:
        print("scipy gap (expected):")
        print(snote)
        print()
    print(f"  Total bakeoff wall time: {elapsed_s:.1f} s")

    if args.output_json:
        payload = {
            "problem": asdict(prob),
            "config": {
                "max_iter": args.max_iter,
                "n_restarts": args.n_restarts,
                "n_warmup": args.warmup,
                "n_repeat": args.repeats,
            },
            "results": [asdict(r) for r in results],
            "wall_time_s": elapsed_s,
        }
        args.output_json.write_text(json.dumps(payload, indent=2))
        print(f"  JSON written to {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
