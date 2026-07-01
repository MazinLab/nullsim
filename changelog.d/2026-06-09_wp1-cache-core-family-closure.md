---
date: 2026-06-09
---
### Stage cache: core-reachable stage families

The stage cache now scans cache-relevant non-stage package code for imports
into `nullsim.stages.<family>`. Any stage family reachable from shared core
code, such as the `physics` to `chip_optimization` edge, is folded into every
stage source digest so shared-core imports cannot serve stale cached results.
