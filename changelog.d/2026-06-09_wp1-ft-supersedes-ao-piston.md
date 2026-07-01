---
date: 2026-06-09
---
### Fringe tracking: supersede replaced AO piston terms

`fringe_tracking.closed_loop` now supersedes the atmospheric piston and all
AO piston sub-budget components it replaces (`ao_temporal`, `ao_fitting`, and
`ao_wfs_noise`) before rebuilding the active OPD quadrature total.
