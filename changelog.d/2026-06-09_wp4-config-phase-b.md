---
date: 2026-06-09
---
### KILO Keck config: Phase B fringe-tracking wiring

`examples/kilo_keck.toml` now enables the photon-Kalman fringe-tracking control
model with inter-telescope piston, Keck vibration lines, accelerometer fusion,
and SNSPD tracker multiplicity. The Phase B science-stage mitigation knobs are
also wired: dark-port setpoint trim leakage in sensitivity and photon-resolved
NSC reduction for performance and characterization.

`examples/kilo_keck_6.toml` drops the unused pre-sensitivity
`fringe_observability` stage and relies on the OPD-budget branch that already
returns before reading fringe-observability payloads.
