---
date: 2026-05-20
---
### NAYRA six-chip architecture

Added `instrument.nayra_six_chip`, a NAYRA-specific architecture manifest stage
that records the Y/J/H × A/B channel map, shared per-band chip-design IDs, and
per-channel frontend-to-chip transmission factors.

`examples/nayra_eelt.toml` now includes the six-chip manifest and propagates the
J-primary HWP/PBS/PM-fiber dual-polarization chain before the chip optimizer.
The sub-aperture-count sweep keeps the architecture manifest's port counts
zipped with the optical stages.
