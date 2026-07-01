---
date: 2026-06-09
---
### kilo_keck_6: realistic det piston floor + restore the char-mode curve

Two regressions in the kilo_keck_6 detection-space figure (both dating to the
June 5-8 work) are fixed.

**Det null was ~10x too shallow.** The `sensitivity` stage was using the
fringe-observability temporal-Kalman `one_step_prediction` piston surrogate
(~3.1 nm at J=4 / 6.3 nm at J=8), which was built on the *open-loop* AO prior
(10.4 nm) and used the closed-loop servo bandwidth (11.5 Hz) as the prediction
interval, overestimating the piston ~4x. The config now uses
`ao_piston_residual_source = "opd_budget"`, i.e. the actual closed-loop
`fringe_tracking.closed_loop` residual (~0.71 nm at J=4 / 3.94 nm at J=8), which
restores a floor-limited null (~1e-5 at J=4). The `fringe_observability` stage
still runs for the single-frame CRLB / observability diagnostics. Applied to
both `examples/kilo_keck_6.toml` and `examples/test/kilo_keck_6.toml`.

**Char-mode curve collapsed to ~2 points.** `detection_space`'s char-curve
plotter gated anchors on a second, stricter `full_band_constraint_margin`
(static null <= null_floor across every detector bin) that the optimizer never
targeted — it nulls to `null_floor` only on the broadband design grid. That
dropped 8 of 12 feasible anchors. The plot now gates only on
`optimization_constraint_margin_per_anchor` (the margin the optimizer actually
solved against), restoring the full characterization curve. Anchors that fail
the optimizer's own constraint are still dropped.
