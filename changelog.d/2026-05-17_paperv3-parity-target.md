---
date: 2026-05-17
---
### Phase 0 — re-pin parity target to paperv3/figures

The KILO parity audit (2026-05-17) found that nullsim's pinned reference
file (`kilo_maunakea_reference.json`) was built against a single-aperture
analytic surrogate rather than the paperv3 publication artifacts.
`paperv3/figures/keck_only_results.json` and
`paperv3/figures/detection_space_cache.json` disagree on the J-band dark
null by ~2× (1.39e-04 vs 7.28e-05) — both live alongside the paper, but
each was generated with slightly different MC settings.

This phase establishes paperv3/figures/keck_only_results.json as the
canonical Keck-only parity target:

- Added `tests/examples/references/kilo_keck_paperv3_reference.json`
  capturing the per-band (Y/J/H — J pinned for now) summary values
  from the paperv3 publication artifacts, with `_drift_notes` recording
  the detection-space disagreement for Phase 1 cross-checks.
- Added `tests/examples/test_kilo_keck_paperv3_parity.py` exercising
  `examples/kilo_keck.toml` at the J=4 fiducial cell and comparing 7
  scalar metrics to the paperv3 reference. Metrics gated on Phases 1,
  2, 3, 6 are `xfail`'d with phase-tagged reasons so the dict shrinks
  one entry at a time as fixes land.
- Added `scripts/generate_paperv3_reference.py` so the reference file
  is refreshable from the published artifacts (no hand-editing).
- Annotated `kilo_maunakea_reference.json` as the smaller surrogate
  config, pointing at the new paperv3 reference for the publication
  target.

KILO HEAD `47607de33e15b8cabdc0e3a4562a7abd98f582a2` matches the
paperv3 publication state; no KILO regen needed.
