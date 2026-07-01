# Chip Optimization Notes

## KILO source surveyed

Files inspected:

- `kilo/chip.py` lines 304-331:
  `_null_objective` — monochromatic/power-weighted leakage sum.
- `kilo/chip.py` lines 639-757:
  `optimize_null` — BFGS with optional analytic gradient backends.
- `kilo/chip.py` lines 903-960:
  `optimize_null_broadband` signature and docstring.
- `kilo/chip.py` lines 761-806:
  `_process_worker_run_restart` — ProcessPool worker for multi-process BFGS.
- `kilo/chip_torch.py` lines 1-46:
  module docstring and TF32-disable block (lines 43-45).
- `kilo/chip_torch.py` lines 63-134:
  `_clements_unitary_ri` real/imag-split implementation.
- `kilo/chip_torch.py` lines 137-191:
  `make_torch_null_objective_and_grad` — single-wavelength factory.
- `kilo/chip_torch.py` lines 194-383:
  `make_torch_broadband_null_objective_and_grad` — broadband factory.
- `kilo/chip_mlx.py` lines 1-17:
  module docstring, 240x speedup claim, real/imag split rationale.
- `kilo/chip_mlx.py` lines 107-159:
  `make_mlx_null_objective_and_grad` — optimizer choice (BFGS via scipy).
- `kilo/chip_mlx.py` lines 163-260:
  `make_mlx_broadband_null_objective_and_grad`.
- `kilo/tests/test_chip.py` lines 134-145:
  `TestOptimizeNull::test_deep_null_small_system`.
- `kilo/tests/test_chip_backends.py` lines 62-99:
  `TestClementsBackendsAgree`.
- `nayra/chip.py` lines 306-465:
  `optimize_null` — NAYRA fork with remote dispatch and thread-pool fan-out.
- `nayra/chip.py` lines 565-776:
  `optimize_null_broadband` — broadband optimizer with NAYRA-specific defaults.
- `nayra/chip_torch.py` lines 1-50:
  thin re-export of kilo.chip_torch with float64 default dtype.
- `nayra/chip_mlx.py` lines 1-23:
  thin re-export of kilo.chip_mlx.

## Non-obvious invariants

### 1. Scipy optimizer: loss function, method, seed, and iterations

`optimize_null` (chip.py:639-757) minimizes `_null_objective` (chip.py:304-331):

```
L = sum_{wl} sum_{k in dark} |( U @ field_wl )_k|^2
```

This is the raw (unnormalised) leakage power summed over all dark ports and
all input field rows. The returned `null_depth` is `L / total_power`
(chip.py:756).

Method: defaults to `method="BFGS"` (chip.py:645). BFGS is preferred over
L-BFGS-B for large meshes because BFGS uses the full Hessian approximation
and does not get trapped in the same basins (chip.py:674).

Init: `x0 = rng.uniform(0, 2*pi, n_params)` for each restart (chip.py:728).
Seed: `rng = np.random.default_rng(seed)` (chip.py:697).

Restarts: `n_restarts=5` default. Each restart uses a fresh random `x0`.
Early exit once `null_depth <= null_depth_target` (chip.py:752-754).
`null_depth_target` defaults to `1e-11`.

Max iterations: `maxiter=10000` default (chip.py:648).

Gradient: numerical (scipy 2-point finite difference) unless an analytic
backend is requested. Analytic backends are triggered by `use_torch=True` or
`use_mlx=True`, and only when `fields.imag` is negligible (< 1e-12);
complex fields silently fall back to numerical gradient (chip.py:712).

For broadband: `optimize_null_broadband` (chip.py:903-960) uses the
objective:
```
L = sum_i w_i * sum_{k in null} |( U_i @ field_i )_k|^2
```
where `U_i` is built with `wl_scales[i] = wl_center / wl_i` applied to
phi + diag only. Default backend is `"torch"` and default method is `BFGS`
via the torch analytic-gradient factory.

### 2. chip_torch.py: real/imag split and TF32 disable

`_clements_unitary_ri` (chip_torch.py:63-134) builds the Clements unitary as
two float64 tensors `(U_r, U_i)` rather than a single complex tensor. This
avoids complex autograd entirely — KILO's on-axis and kernel-null field stacks
are all real, so `|U @ f|^2 = (U_r @ f)^2 + (U_i @ f)^2`.

The vectorised per-column Clements loop reduces CUDA kernel launches from
`O(M^2)` (one per MZI) to `O(M)` (one per column), critical for M=24 on CUDA.

**TF32 disable lines (chip_torch.py:43-45):**

```python
torch.backends.cuda.matmul.allow_tf32 = False
if hasattr(torch.backends, "cudnn"):
    torch.backends.cudnn.allow_tf32 = False
```

These are module-level statements, executed at import time. They apply
globally to all torch operations in the process.

**Why this matters (KILO hard-won lesson):** on Ampere and Blackwell GPUs,
TF32 mode truncates float32 mantissas to 10 bits before matmul. At float32
precision this perturbs BFGS gradients enough to flip the optimizer into a
different basin: KILO observed ~1.4e-4 broadband null on an RTX 5090 at fp32
with TF32 enabled vs ~6e-5 on CPU at the same seed. fp64 sidesteps both TF32
and non-deterministic CUDA reductions (chip_torch.py:27-31).

All tensors in the torch backend are materialized as **float64** (chip_torch.py:24).

The `KILO_TORCH_COMPILE=1` env var wraps the objective in `torch.compile`
(mode='reduce-overhead'): 15.5x per-grad speedup at M=24/n_wl=5 on Mac CPU,
4.2x end-to-end. Off by default (chip_torch.py:358-366).

### 3. chip_mlx.py: float32, real/imag split, optimizer, and speedup

`chip_mlx.py` uses **float32** throughout (chip_mlx.py:135, 218, 249). This
is a deliberate trade-off: Metal dispatch overhead dominates at small M; fp32
is acceptable for intermediate null depths but loses precision below ~1e-7.

Real/imag split rationale (chip_mlx.py:11-17): MLX 0.31 has broken autograd
for `abs(complex)**2` and no VJP for `conjugate`; tracking U as `(U_r, U_i)`
keeps the objective in pure real arithmetic that MLX can differentiate.

The per-column vectorization collapses `O(M^2)` Metal kernel launches to
`O(M)` per BFGS call (chip_mlx.py:39-45). This is the key to the 240x
speedup claimed in the module docstring: "For M=20 the speedup over numerical
gradients is ~240x" (chip_mlx.py:8).

**Optimizer:** scipy BFGS via the factory pattern. `make_mlx_null_objective_and_grad`
(chip_mlx.py:107-159) returns `(obj_np, grad_np)` — numpy-in / numpy-out
callables suitable for `scipy.optimize.minimize(obj, x0, jac=grad,
method='BFGS')`. The MLX backend does **not** run its own optimizer; scipy
BFGS drives the line search and `_clements_unitary_ri` + `mx.grad` provide
the gradient.

Both `_objective` and `mx.grad(_objective)` are wrapped in `mx.compile` at
factory time (chip_mlx.py:144-145) so the per-column ops fuse into a single
Metal kernel per call.

At M<=24 with multi-threaded MKL BLAS, the torch CPU path is faster than MLX
(per chip.py:953-957 comment and the NAYRA 2026-05-13 resync note). MLX wins
on much larger meshes where Metal compute dominates dispatch.

### 4. test_chip.py: TestOptimizeNull::test_deep_null_small_system

```python
def test_deep_null_small_system(self):
    m = 4
    stellar = np.ones(m, dtype=complex) / np.sqrt(m)
    dark = np.array([0, 1])

    _, null_depth = optimize_null(
        stellar, m, dark, n_restarts=10, seed=0
    )

    assert null_depth < 1e-4, f"Null depth {null_depth} too high for m=4 mono"
```

Key parameters: `m=4`, seed=0, `n_restarts=10`, dark ports `[0, 1]`.
Assertion threshold: **null < 1e-4** (comment says "should achieve at least
1e-6"; the hard assertion is 1e-4 to be robust across scipy versions).

This test verifies that scipy BFGS with numerical gradients (no torch/mlx)
finds a deep monochromatic null for a 4-mode mesh within 10 restarts.

### 5. test_chip_backends.py: TestClementsBackendsAgree

```python
@pytest.fixture(scope="module")
def random_params():
    rng = np.random.default_rng(0)
    for m in (4, 8, 12):
        n_p = n_params_for_unitary(m)
        for _ in range(20):
            params = rng.uniform(0.0, 2 * np.pi, n_p)
            cases.append((m, params))
    return cases
```

m values tested: **{4, 8, 12}**, 20 random parameter vectors each.

Tolerance: **`atol=1e-4`** for both MLX and torch vs numpy reference
(test_chip_backends.py:74, 88). This tolerance accounts for float32
accumulation error in the backends vs float64 numpy reference.

Float32 is used for the backend parameter arrays (`params.astype(np.float32)`)
while numpy computes in float64. The 1e-4 tolerance is specific to this
float32/float64 cross-check; it is not a statement about null-depth accuracy.

MLX test skipped if `mlx` is not installed (uses `importlib.util.find_spec`
guard before any MLX runtime calls, to avoid hard-abort on incompatible Metal
stacks — test_chip_backends.py:26-37).

### 6. NAYRA backends diff vs KILO

**chip_torch.py (nayra/chip_torch.py):** thin re-export of `kilo.chip_torch`.
Zero independent implementation. Adds one behaviour change at module load:

```python
if _torch.get_default_dtype() is _torch.float32:
    _torch.set_default_dtype(_torch.float64)
```

(nayra/chip_torch.py:21-22). This is a process-global side effect; any other
torch code in the same process will also use float64 after this import.
KILO does not set default dtype — it always passes `np.float64` explicitly
to tensor constructors.

**chip_mlx.py (nayra/chip_mlx.py):** thin re-export of `kilo.chip_mlx`.
No differences.

**optimize_null (nayra/chip.py:306-465):** adds remote Ray dispatch via
`_should_use_remote()` / `NAYRA_OPT_BACKEND=remote` (nayra/chip.py:356-363).
Default backend as of 2026-05-13 resync: `use_torch=True` when
`NAYRA_OPT_BACKEND` is unset (nayra/chip.py:368-371). Adds thread-pool
fan-out across restarts via `NAYRA_OPT_N_THREADS` env var (default 1).

**optimize_null_broadband (nayra/chip.py:565-776):** adds remote dispatch,
thread-pool fan-out (`NAYRA_BB_N_THREADS`), and `NAYRA_BB_OPTIMIZER=batched_adam`
path (batched Adam over all restarts simultaneously — for saturating a CUDA
GPU where per-restart BFGS is launch-overhead-bound). No KILO equivalent of
the batched Adam path.

**nullsim scope:** nullsim should implement the non-remote, single-machine
BFGS path only. Remote dispatch and batched Adam are NAYRA-specific cluster
features not needed in the nullsim pipeline.

## Sign / dtype conventions

- `_null_objective` returns total leakage power (not normalised). The caller
  divides by `total_power` to get `null_depth`. Never pass `null_depth`
  directly to scipy — the optimizer sees raw leakage.
- Gradient sign: all objectives minimised, so negative gradient points toward
  lower leakage. SLSQP constraint returns `threshold - spectro_null >= 0`
  (positive = constraint satisfied).
- Wavelength scale convention: `scale = wl_center / wl_target`, so longer
  wavelength → scale < 1 → smaller phi/diag phases (physically correct for
  thermal phase shifters).

## Edge cases

- `optimize_null` with `use_torch=True` or `use_mlx=True` and complex fields
  silently falls back to scipy numerical gradient (chip.py:712). There is no
  warning. Callers with complex fields get no analytic-gradient speedup.
- Early-exit logic (chip.py:752-754) checks the best null from all completed
  restarts, not just the current one. A single very good restart stops the
  remaining ones.
- `best_params` is `None` if `n_restarts=0`; the function would return
  `(None, 0.0)`. No guard in KILO; nullsim should validate `n_restarts >= 1`.
- The broadband objective weights default to `weights = np.ones(n_wl) / n_wl`
  (uniform mean). Passing wavelength-dependent weights (e.g. instrument
  throughput × stellar SED) is supported but not the default.
- MLX compile (`mx.compile`) is applied at factory-construction time, not per
  call. Reconstructing the factory inside a restart loop (instead of once
  before the loop) wastes ~tens of milliseconds of Metal recompilation per
  restart.
