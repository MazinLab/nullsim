---
date: 2026-05-17
---

### KILO Keck-only parity — phases 5–6 (calibration floor + PDI + polish)

#### Phase 5.1 — Bundled KILO M24 J-band calibration-time table

`nullsim/data/kilo_reference/calibration_time_J_M24.json` now ships with
the package. `calibration.floor_lookup` resolves `lookup_table_path` first
against the working directory, then against the `nullsim` package parent,
so the bundled table works from any cwd without an absolute path in TOML.

Without this table `nullsim` returned a null floor of 0 at J=4 long
separations rather than KILO's ~1.135e-7 — the dominant long-separation
contrast contribution for the M24 design point.

#### Phase 5.2 — Polarimetric differential imaging (PDI) SNR primitive and stage

New `nullsim.physics.pdi.signal_to_noise_pdi` ports KILO
`signal_to_noise_pdi` (`kilo/performance.py:412-648`): SPHERE-IRDIS / GPI
/ ZIMPOL 4-angle Mueller double-difference scheme with photon/dark/sky
variance (×2 under Q+U recovery), imperfection coupling `ε_pdi` (PBS axis
offset, HWP retardance, HWP rotation jitter, PBS extinction-ratio leakage),
modulation-cycle averaging, and common-mode rejection ratio.

New `performance.pdi` stage (`PdiPerformance`) drives the primitive from
upstream `detector.*` ports. Registered in `stages/_autoload.py`. Not
enabled in `kilo_keck.toml` (scalar single-arm study); opt in via
`[[pipeline.stages]]`.

#### Phase 6.1 — `fringe_tracking/closed_loop.py` module docstring corrected

Module docstring rewritten to describe both the legacy
`fringe_tracking_phase_noise` path (shot × π/2 + standalone Conan lag) and
the full KILO `piston_servo_residual` decomposition gated by
`use_full_piston_servo_residual`. Previous text described a
pre-Tier-2a formula that did not reflect the π/2 multiplier or the
full-physics branch added in Phase 2.

#### Phase 6.4 — `kilo_keck.toml` DAC-bits comment corrected

`dac_bits = 16` comment now cites both sources: KILO default is 20
(`kilo/config.py:231`); paperv3 `chip_diagnostics.pdf` was generated at 16.
The TOML pins 16 to track the published figure.
