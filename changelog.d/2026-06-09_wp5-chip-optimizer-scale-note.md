---
date: 2026-06-09
---
### Chip optimizer: honest disk-floor presentation under flux-dependent objectives

The chip-optimizer disk-floor scatter across the J sweep was traced to the
objective, not a bug. The base broadband null objective (absolute dark-port
leakage plus the throughput penalty) is scale-equivariant: the reported null
depth and the stellar-disk null floor it drives are flux-scale invariant (pinned
in `tests/stages/chip_optimization/test_scale_invariance.py`). The tracker,
joint, and amplitude observability terms, however, carry absolute physical
targets — the per-nm intensity-response singular floor and the CRLB targets in
nm — that are genuinely photon-rate dependent (a fringe tracker's piston
estimability scales with the actual photon flux). When those terms are active
the optimum, and hence the realized disk floor, depends on the flux the chip
was optimized at.

Rather than forcing a normalization that would corrupt the photon-rate-dependent
CRLB targets, this surfaces the dependence honestly:
`ChipOptimizationParams.objective_is_flux_scale_dependent` reports whether any
observability term is active; the chip-optimization payload now carries
`disk_null_floor_note` ("chip-realization dependent" or "flux-scale invariant");
the `ChipOptimizationTorch` docstring names the intentionally scale-dependent
terms; and `performance_summary` now emits `disk_null_floor` and
`disk_null_floor_note` columns so the per-flux disk-floor spread is visible
across sweep cells.
