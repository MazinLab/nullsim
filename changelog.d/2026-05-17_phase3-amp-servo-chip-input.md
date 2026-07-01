---
date: 2026-05-17
---
### Phase 3 — amplitude servo photon source = chip-input / n_modes

Per the KILO parity audit (2026-05-17), nullsim's amplitude-servo
photon budget was derived from the AO WFS payload
(`wfs_photons_per_subap_per_frame × loop_frequency_hz`). That value is
the GUIDE-star photon budget at the AO bench, not the science photons
hitting the photonic chip. KILO `natural_design.py` L897 uses
`ref_rate_at_chip / n_modes` — the chip-input science flux distributed
across the chip's modes, optionally attenuated by a dichroic pickoff
fraction.

- `nullsim/stages/sensitivity/detection_curve.py`:
  - New helper `_chip_input_stellar_rate_hz(state)` assembles
    `ref_rate_at_chip = sum_lambda photon_rate_per_m2_per_bin(jmag, lambda,
    dlambda) × area × transmission(lambda)` from
    `state.scene.star_jmag`, `state.field.wavelength_grid_um`,
    `state.geometry.area_m2` (or `effective_collecting_area_m2`), and
    `state.throughput.transmission_fraction`.
  - New helper `_resolve_amp_servo_n_modes(state, params)` picks the
    mode count from the upstream chip stage, falling back to
    `n_bright_ports + n_dark_ports`.
  - `_resolve_amp_servo_photon_rate_hz` now defaults to
    `pickoff_frac × chip_input_rate / n_modes`. Explicit
    `auto_amp_servo_photon_rate_per_sub_hz` still wins. The AO-WFS-
    payload fallback is removed (the wrong-by-orders-of-magnitude
    rate it produced is no longer reachable).
  - New params `amp_wfs_pickoff_fraction` (default 1.0, KILO
    `KILO_AMP_WFS_PICKOFF_FRAC`) and `amp_wfs_n_modes_override`
    (default None) for explicit overrides.
- Two new regression tests in `test_detection_curve.py`:
  - Chip-input fallback produces a rate in the expected J=4 Keck range.
  - `amp_wfs_pickoff_fraction` linearly scales the per-sub rate.
- The pre-Phase-3 "auto-amp-servo requires photon rate" test is updated
  to assert the new failure mode (missing `scene.star_jmag`).

`null_instability_jband` xfail in `test_kilo_keck_paperv3_parity_metric`
left in place pending the end-to-end run; Phase 3 contributes but
sigma_I drives instability jointly with Phase 1's azimuth + transport
marginalization.
