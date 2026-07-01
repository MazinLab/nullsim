---
date: 2026-05-17
---
### Phase 2 — piston servo: auto-BW + subaperture crossing scale

Per the KILO parity audit (2026-05-17), nullsim's fringe-tracker pinned
`servo_bw_hz = 2000.0` and used the full primary diameter for the
Conan crossing scale `f_char`. KILO auto-optimizes the servo bandwidth
(photon-limited `f_opt`, `kilo/fringe_tracking.py` L286) and uses the
sub-aperture diameter (`d_sub`) for `f_char`, because the atmospheric
piston between sub-apertures is driven by independent wind crossings of
each sub-aperture (timescale `d_sub / v_wind`, not `D_primary / v_wind`).

- New physics primitive
  `nullsim/physics/fringe_tracking.optimal_piston_servo_bandwidth` —
  ports KILO's closed-form
  `f_opt = [(5/(3π)) · σ²_temp · f_char^{5/3} · R · V² · pred]^{3/8}`.
- `closed_loop.py`:
  - `servo_bw_hz: float | None = None` (default → auto). Explicit
    floats still pin the bandwidth as before.
  - `_resolve_servo_bandwidth_hz` reads upstream `ao_residual` for the
    temporal piston variance + the bright-port photon rate from
    `state.photon_rates`, and calls the new physics primitive. Returns
    a `(bandwidth_hz, source)` tuple so diagnostics can surface whether
    the value came from params or auto-optimization.
  - `_effective_baseline_m` now prefers
    `ao_residual["piston_subaperture_diameter_m"]` over
    `state.geometry.diameter_m`, falling back only when no AO stage is
    upstream. The pre-Phase-2 default used the full primary, which
    over-stated `f_char` by ~5× for Keck.
  - `_piston_servo_residual_rad2` (and `_photons_per_channel_per_frame`)
    now take the resolved `servo_bandwidth_hz` and `f_char_hz` as
    explicit parameters instead of reading `params.servo_bw_hz` and
    deriving `f_char` internally — single source of truth for the
    resolved bandwidth and crossing scale.
  - The diagnostic component published under
    `state.results.components["fringe_tracking.closed_loop"]` now
    includes a `breakdown` of the σ² terms (lag / photon /
    uncorrectable / irreducible) and a `servo_bandwidth_source` tag,
    powering the debug-report waterfall visualisation.
- `examples/kilo_keck.toml`: removed the manual `servo_bw_hz = 2000.0`
  pin so the auto-optimum kicks in.
- Two new regression tests in
  `tests/stages/fringe_tracking/test_closed_loop.py` verify the
  auto-bandwidth path uses sub-aperture `f_char` and reports
  `servo_bandwidth_source` correctly.
- Five new tests in `tests/physics/test_fringe_tracking.py` pin the
  closed-form formula and KILO-regime sanity bounds.

`sigma_phi_rad_jband` xfail in `test_kilo_keck_paperv3_parity_metric`
left in place pending an end-to-end re-run; Phase 2 is necessary but
the parity baseline may shift once Phases 3/4 also land.
