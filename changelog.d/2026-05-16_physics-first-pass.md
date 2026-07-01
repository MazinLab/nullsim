---
date: 2026-05-16
---
### Physics first scoped pass — §15.2 (commit 35409d3)

Three `nullsim/physics/` modules shipped, each with a notes file in `tests/physics/notes/` capturing cross-check decisions:

- **`kolmogorov.py`** — Kolmogorov phase PSD, structure function, r₀ wavelength scaling. Cross-checked between KILO and NAYRA atmosphere implementations.
- **`photometry.py`** — J-band Vega zero-point, magnitude-to-photon-rate conversion (Cohen+ 2003).
- **`ruilier_cassaing.py`** — single-mode-fiber coupling efficiency vs. Strehl with optional central obscuration; Ruilier 1999 analytical limit pinned at η ≈ 0.8145 for matched circular aperture.

Each module has a pure unit-test suite in `tests/physics/`. Remaining §15.2 modules (clements, jones, fresnel, chip optimizer backends) are deferred to subsequent passes.
