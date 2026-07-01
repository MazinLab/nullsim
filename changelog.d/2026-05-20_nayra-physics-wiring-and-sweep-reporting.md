---
date: 2026-05-20
---

## NAYRA physics wiring and sweep reporting fixes

- Added `nullsim/data/nayra_reference/calibration_time_J_M36.json` — N=36
  sub-aperture calibration floor reference, wired into `calibration.floor_lookup`
  alongside the existing M18 file.
- Added `examples/nayra_eelt_baseline.toml` — fixed-point NAYRA E-ELT config
  with no sweeps, for use as a stable regression baseline.
- `background.nayra_sky`: sky losses now applied at the field level, preserving
  dual-pol detector channel structure across NAYRA pipelines.
- `calibration.floor_lookup`: validates calibration table metadata on load;
  supports M36 reference data.
- `detector.mkid`: preserves dual-pol channels in NAYRA per-arm deployments.
