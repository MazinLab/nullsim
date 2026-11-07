### Fixed

- `config.sweep.resolved.toml` now holds the merged, pre-materialization
  config (extends and presets resolved, `[[sweep]]` intact) and reloads
  and re-expands to the same cells. It previously held the materialized
  config, whose injected stage values read as explicit overrides, so the
  kilo_keck_6, nayra_eelt, and hwo_space artifacts failed with "sets
  'star_jmag' explicitly" (review 2026-11-07 row 67).
  `config.resolved.toml` stays the materialized cell-0 config.
- A relative `[catalog.exoplanets].snapshot_path` is rebased onto the
  directory of the file that wrote it while `extends` chains load, and
  the resolved TOML records the absolute path, so it reloads from any
  directory and an inherited path no longer resolves against the child
  (row 68). The absolute path enters `config_hash`, so the hwo_space and
  detection_space_example config hashes now depend on checkout location.
- The state content digest hashes components in insertion order instead
  of sorted by name. Pipelines that differ only in stage order (e.g. two
  background stages swapped ahead of `performance.standard`) no longer
  share a cache key (Issue 25). Existing cache entries for states with two
  or more components miss once.
