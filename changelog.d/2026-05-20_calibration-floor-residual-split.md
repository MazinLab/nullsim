---
date: 2026-05-20
---
### Calibration floor/residual split

`calibration.floor_lookup` now records the KILO/NAYRA calibration scalar as
`calibration_floor` and the post-subtraction contrast as
`null_depth_residual = max(null_depth_raw - calibration_floor, 0)`. The old
`null_depth_calibrated` key remains as a compatibility alias for the residual.

`performance.standard`, debug-report summaries, and KILO/NAYRA parity metrics
prefer the explicit residual key while keeping compatibility with older cached
run records that only contain `null_depth_calibrated`.
