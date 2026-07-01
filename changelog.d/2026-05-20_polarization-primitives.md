---
date: 2026-05-20
---
### Polarization primitives

Added KILO/NAYRA Jones-layer primitives to `nullsim.physics.jones`: PBS/HWP/PM
fiber hardware dataclasses, achromatic HWP retardance, imperfect PBS arm
matrices, unpolarized PBS splitting, unitary PM-fiber Jones matrices with a
short-fiber PER floor, random birefringence, and the NAYRA PDI imperfection
quadrature factor.

The new helpers are pure physics functions with package-level exports; pipeline
stages remain scalar until the follow-on dual-polarization phase wires these
primitives into `SimulationState.field`.
