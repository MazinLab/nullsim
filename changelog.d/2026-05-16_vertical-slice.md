---
date: 2026-05-16
---
### Vertical slice implemented — §15.1 complete (commits 3424290, a0551c1)

Implemented the §15.1 architecture-validation vertical slice end-to-end:

**Pipeline layer** (`nullsim/pipeline/`): stage protocol, `SimulationState`, runner with dependency validation, conservative content-hash cache (`_canonical.py`, `cache.py`), stage/plot/table registry, RNG derivation (`rng.py`), run manifest, `RunContext`.

**Config layer** (`nullsim/config/`): TOML loader with preset merge and environment overrides, pydantic v2 schemas, sweep expansion (grid/zip), config hashing, stage materialization, validation errors.

**Stage layer** (`nullsim/stages/`): five trivial stages — `scene.point_source`, `telescope.fixed_aperture`, `injection.ideal`, `chip.identity`, `detector.ideal_counter`. Auto-loader wires all built-in stages at import time.

**Output layer** (`nullsim/outputs/`): plot/table stubs, sweep-results parquet writer, publication matplotlib styling.

**CLI** (`nullsim/cli.py`, `nullsim/__main__.py`): `nullsim run|validate|list-stages|inspect|cache` subcommands.

**Tests**: 60+ files covering config, pipeline, stages, outputs, and end-to-end vertical-slice example. `examples/vertical_slice.toml` runs, cache-hits on re-run, sweep produces `sweep_results.parquet`.
