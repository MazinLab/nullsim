---
date: 2026-05-17
---

### KILO Keck-only parity — phases 0–4 (architectural / physics)

Comprehensive parity pass targeting factor-of-2 agreement with the KILO
paperv3 detection-space figure across J=4–12. Covers four independent
areas: cache infrastructure, photon-budget wiring, AO+servo physics, and
chip-optimization architecture.

#### Phase 0.1 — Package-wide cache source digest

`nullsim/pipeline/cache.py`: the cache key's source-code component is now
a package-wide SHA-256 digest of all `.py` files under `nullsim/` (sorted,
computed once per process). Replaces the earlier per-stage source-file hash.
**Any code change in the package invalidates all stage caches.** Prevents
stale hits from edits to helper modules outside the stage's own `.py` file.

#### Phase 0.3 — `IdealInjection` now applies Strehl × η_opt by default

`injection.ideal.apply_strehl_coupling` defaults to `True`. Pre-phase-0.3,
`IdealInjection` seeded unit amplitudes so the MKID never saw Strehl
attenuation — the stellar rate was 3.7× too high at J=4, 7.8× at J=8.
**Callers that depended on unit-amplitude field seeds must set
`apply_strehl_coupling = false` explicitly.**

New companion param `apply_fresnel_injection_loss: bool = False` — when True,
additionally scales amplitudes by √(1 − FRESNEL_REFLECTION_LOSS). Default off
for back-compatibility.

#### Phase 1.2 — Contrast denominator: total stellar rate, not bright-port rate

`StandardPerformance` contrast formula now uses the aggregate post-throughput
stellar photon rate (matching KILO `performance.py:964`). Pre-fix the contrast
was divided by bright-port rate only, making it ~4× too pessimistic on Keck
(18 dark / 24 total modes).

#### Phase 1.4 — `central_obstruction_ratio` multiplies into `area_m2`

`telescope.fixed_aperture` applies `(1 − ε²)` to `collecting_area_m2` when
`central_obstruction_ratio > 0`. Pre-fix Keck's ε=0.24 was ignored → 5.8%
area overcount in all photon-rate stages.

New param `ao_bench_transmission: float | None` inserts a multiplicative
throughput component representing the AO relay. When set, it is applied
immediately after the aperture area step and published to
`state.throughput.components`.

#### Phase 2.5 — FT closed-loop shot variance: ×π/2 and /n_bright_ports

`fringe_tracking/closed_loop.py`: the per-frame shot-noise variance is now
multiplied by π/2 (KILO `piston_servo_residual` convention,
`kilo/fringe_tracking.py:439-443`) and the total bright-port photon rate is
divided by `n_bright_ports` before computing per-baseline signal counts
(KILO `fringe_tracking_phase_noise`, `kilo/fringe_tracking.py:159-162`).
Pre-fix the FT photon noise was ~2.8× underestimated on Keck-only (6 bright
ports).

#### Phase 3.1 — `n_spectro_ports` in chip_optimization

`ChipOptimizationParams.n_spectro_ports` (default 0): ports occupying the
last `n_spectro_ports` slots of the bright partition that are nulled by the
optimizer but not counted as bright-throughput ports. Closes the KILO
Keck-only port-accounting gap (5 bright + 18 dark + 1 spectro = 24 modes;
optimizer must suppress 19 ports, detect on 18).

#### Phase 3.4 — `restart_patience` default 3 → 0

`ChipOptimizationParams.restart_patience` default changed from 3 to 0
(matching KILO `KILO_BB_PATIENCE=0`). The previous default caused premature
termination of broadband restart loops without improving the basin.
