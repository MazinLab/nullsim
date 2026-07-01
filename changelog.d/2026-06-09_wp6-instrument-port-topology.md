---
date: 2026-06-09
---
### Config: `[instrument]` port-topology section

Added a strict top-level `[instrument]` section for shared chip/input-mode and
port-count topology. The loader materializes those values into port-bearing
stages while preserving explicit stage-level overrides. The root KILO Keck
configs now declare the 48-port and 24-port layouts once and rely on
materialization for chip, detector, sensitivity, characterization, injection,
and fringe-tracking stage params.
