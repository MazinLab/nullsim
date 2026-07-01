---
date: 2026-05-16
---

### §15.4 — detection_space output + catalog versioning

- New package `nullsim/catalogs/` with `exoplanets.py` (versioned snapshot loader,
  classification rule v1) and `nullsim/config/catalog.py` (`[catalog.exoplanets]`
  TOML schema bridge).
- New plot `nullsim/outputs/plots/detection_space.py` — the headline detection-space
  figure: contrast curve family + known exoplanet population overlay, colored by
  planet type (Rocky / Super-Earth / Neptune-like / Gas Giant), shaped by discovery
  method (RV / Transit / Imaging / Other).
- `nullsim/cli.py` `_run_hash_with_externals` now folds the catalog snapshot
  `content_hash()` into `run_hash` so a catalog refresh produces a new output
  directory and a new manifest entry rather than silently shifting the points.
- New example config `examples/detection_space_example.toml`.
- Test count: 446+3xfail → 494+21xfail (+48 passing, +18 xfail including 17
  honest pipeline-gap markers across KILO/NAYRA/HWO scenarios).

### §15.5 — kilo_maunakea.toml + parity infrastructure (KILO HEAD 47607de)

- `examples/kilo_maunakea.toml` ships as the canonical four-telescope Maunakea
  KILO config.
- Parity infrastructure: `tests/examples/references/kilo_maunakea_reference.json`
  (frozen scalar metrics at KILO HEAD 47607de) and matching parity test suite.
- `scripts/generate_kilo_reference.py` regenerates the reference snapshot
  deliberately; not run automatically.

### §15.6 — nayra_eelt.toml + parity infrastructure (NAYRA HEAD 48ece53)

- `examples/nayra_eelt.toml` ships as the canonical single-pupil E-ELT NAYRA config.
- Parity infrastructure: `tests/examples/references/nayra_eelt_reference.json`
  (frozen scalar metrics at NAYRA HEAD 48ece53) and matching parity test suite.
- `scripts/generate_nayra_reference.py` regenerates the reference snapshot
  deliberately.

### §15.7 — hwo_space.toml — first new-instrument config

- `examples/hwo_space.toml` ships as the first config with no prior reference
  package: HWO-class space nuller, no atmosphere stage, formation-flying delay
  budget, MKID detector.
- `examples/kilo_vs_hwo_study.toml` ships as the study-layer config that overlays
  KILO and HWO configs in one detection-space figure.
- xfail markers in parity tests are honest pipeline-gap markers (17 across KILO /
  NAYRA / HWO scenarios) rather than temporary skips; each is annotated with the
  specific stage or output not yet wired.
