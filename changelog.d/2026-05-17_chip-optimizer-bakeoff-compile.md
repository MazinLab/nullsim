---
date: 2026-05-17
---
### Chip-optimizer bakeoff script + kilo_keck torch.compile (commit c1a8780)

**New script: `scripts/bench_chip_optimizer.py`**
Two-tier benchmark comparing all available chip-optimizer backends on the
same problem (same RNG seed, same field stack, same wavelength grid):

- Tier 1 (microbench): median of N repeats of a single objective + gradient
  evaluation with warmup. Isolates raw kernel speed from optimizer overhead.
- Tier 2 (end-to-end): full `ChipOptimization*.apply()` to a fixed iteration
  cap. Includes one-time compile-tracing cost for `torch.compile` / `mlx.compile`.

CUDA rows auto-detected; MLX rows excluded for `kernel_order > 0`.

Measured on Apple Silicon at M=24 / n_wl=5:
- numpy/scipy FD: 2.96 s/grad
- torch eager CPU: 26 ms/grad
- torch.compile CPU: 1.6 ms/grad
- MLX Metal: 26 ms/grad
- compile-vs-eager crossover: ~78 grad evals

**`examples/kilo_keck.toml`: compile_objective false → true**
With ~35,000 grad evals per cell (bootstrap 5×2000 + broadband 5×5000)
end-to-end time drops from ~15 min to ~1 min per cell. The one-time
~30 s Inductor cache miss persists in `~/.cache/torch/inductor` across
processes.

Default in `ChipOptimizationParams` stays `false` so small-problem configs
(hwo_space M=2, kilo_maunakea M=4, nayra_eelt M=2 — all scipy backend at
`max_iter=100`) are unaffected.
