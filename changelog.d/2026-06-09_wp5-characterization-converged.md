---
date: 2026-06-09
---
### Characterization outputs: rename `feasible` -> `converged`

The characterization summary table column `feasible` is now `converged`, the
`performance_summary` column `char_feasible` is now `char_converged`, and the
helpers `_characterization_payload_is_feasible` (tables) and
`_characterization_component_is_feasible` (paper bundle) are now
`_characterization_payload_converged` and
`_characterization_component_converged`. The logic is unchanged: it reports
optimizer convergence (success plus null-constraint margin within tolerance),
which is what the flag has always measured. The per-anchor curve feasibility
masks (`_characterization_curve_feasible_mask`) are a distinct quantity and are
left as-is.
