---
date: 2026-05-16
---

## Chip-optimizer KILO port: kernel-null stack, absolute-leakage objective, torch restart loop, debug_report (commits 22fb83a, 06dda42, 8d25657, c28aac8, 64e7d05)

**plots: debug_report.pdf canonical throughput payload (22fb83a, 06dda42):**
`nullsim/outputs/plots/debug_report.py` renders a multi-page PDF walking the pipeline stage-by-stage; each page covers one stage's contribution to throughput/OPD/wavefront/results. Commits 22fb83a (initial port) and 06dda42 fix the injection throughput payload to the canonical dict shape `{wavelength_grid_um, transmission_fraction, +provenance}` consumed by `debug_report` and `loss_budget`.

**perf(torch): restart loop + patience + torch.compile gate (8d25657):**
`nullsim/stages/chip_optimization/torch.py` gains KILO's restart-loop pattern: `n_restarts` outer restarts with `restart_patience` / `restart_patience_reltol` early-exit, and a `torch.compile` gate (enabled only when `torch.cuda.is_available()` or the user opts in via `compile_objective`) that avoids JIT overhead on CPU-only runs.

**perf(chip_opt): KILO absolute-leakage objective (c28aac8):**
`ChipOptimizationParams` gains an `objective` field (default `"absolute_leakage"`). All three backends (scipy, torch, mlx) now minimize `sum |U[dark_ports, :] @ field_stack|²` — the absolute leakage summed over dark ports and wavelengths — matching KILO's chip.py objective exactly. The prior relative-leakage formulation is retained as `"relative_leakage"` for backward compatibility.

**feat: kernel-null field stack (64e7d05):**
`ChipOptimizationParams` gains `kernel_order` (int, default 0, valid: 0/2/3/4). When `kernel_order > 0`, `chip_optimization` builds a kernel-null field stack (`_kernel_null_field_stack`: direct port of KILO `injection.py:kernel_null_field_stack` L226-330) and passes it as the optimizer target instead of the on-axis stellar field. Kernel order 2 → G(r)~r⁴ quadratic null, order 4 → G(r)~r⁸ quartic null. `ao.pyramid_wfs` now writes `state.geometry.aperture_positions_pupil_m` (shape M×2, pupil-frame ENU positions derived from `optimize_subaperture_positions`); `chip_optimization` reads this array when `kernel_order > 0` and raises a clear error if it is absent.
