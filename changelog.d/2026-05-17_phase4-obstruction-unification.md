---
date: 2026-05-17
---
### Phase 4 — unify obstruction ratio between stellar and planet coupling

Per the KILO parity audit (2026-05-17), nullsim's planet coupling in
`sensitivity.detection_curve` hardcoded `central_obscuration_ratio=0.0`
while `injection.ideal` used the configured value (Keck `epsilon=0.24`).
The result: stellar light saw a 6.5% reduction in geometric coupling
that planet light did not, creating an artificial flux mismatch.

- New helper `_central_obstruction_ratio(state)` in
  `nullsim/stages/sensitivity/detection_curve.py` reads
  `central_obstruction_ratio` from the upstream telescope diagnostic
  (`telescope.fixed_aperture` component on `state.geometry.components`).
  Falls back to 0.0 only when no telescope stage is upstream.
- `detection_curve.apply()` now passes the resolved obstruction to
  `coupling_efficiency()` instead of the hardcoded 0.0, so the planet
  leg matches the stellar leg.
- New regression test
  `test_detection_curve_planet_coupling_picks_up_obstruction`: pins
  the ratio of `coupling(epsilon=0.24) / coupling(epsilon=0)` against
  the Ruilier-Cassaing physics primitive.

Out of scope for Phase 4: the chip-optimizer throughput-penalty fields
(`chip_optimization/torch.py` L400) still use unit-amplitude phase
patterns. For the identical-telescope Keck case the ratio cancels and
the optimizer is correct; for NAYRA's heterogeneous E-ELT this would
need physical amplitudes. Tracked as a separate Phase-4.5 task.

`null_depth_dark_jband` xfail in `test_kilo_keck_paperv3_parity_metric`
left in place pending the end-to-end run; the obscuration unification
shifts the absolute coupling level by ~14% which propagates to the
realized null depth.
