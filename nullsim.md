# nullsim — A Generalized Photonic Nulling Instrument Simulator

**Status:** design draft
**Date:** 2026-05-15
**Author:** Ben Mazin (with brainstorming assistance)
**Supersedes:** the bespoke KILO and NAYRA simulation packages, which become reference examples.

---

## 1. Summary

`nullsim` is a Python package and command-line pipeline for simulating photonic nulling instruments. A user describes an instrument and a study in a single TOML file. The pipeline composes physics modules (atmosphere, AO, injection, fiber transport, delay lines, photonic chip, detectors, post-processing) as an ordered sequence of stage instances with declared dependencies validated before execution, runs the requested simulation, optionally sweeps over parameter axes, and writes a chosen subset of standardized plots and tables. Stages, plots, and tables are pluggable: a user adds a custom thermal-drift stage or a custom output by registering one Python file. The same package targets ground-based interferometric nullers (KILO-class multi-telescope arrays), single-pupil ELTs (NAYRA-class), space-based nullers (HWO and successors), and lab-only chip characterization, by enabling or disabling stages rather than by editing code.

## 2. Motivation

The KILO and NAYRA codebases each implement a photonic nulling simulator that produces noise budgets, contrast curves, integration-time estimates, and a detection-space figure that overlays the achievable contrast on the known exoplanet population. The two packages share roughly seventy-five percent of their physics: NAYRA already routes its simulations through a KILO adapter, and the two packages reimplement parallel modules for atmosphere, AO, injection, fiber transport, chip optimization, detectors, and plotting.

Continuing to maintain two packages with a partial adapter between them is the wrong shape going forward. The next instrument target, a Habitable Worlds Observatory (HWO) successor, will reuse the chip and detector physics but discard the atmospheric stages entirely. Adding it as a third sibling repository would compound the problem.

`nullsim` extracts the shared physics into a generic pipeline with one configurable entry point. The package was initially built **guided by the KILO and NAYRA source trees, not by working from papers alone.** KILO in particular encodes a large amount of debugged-by-fire knowledge — sign conventions, dtype choices, edge cases in coupling formulae, baseline-indexing fixes, calibration ordering — that does not appear in any reference. It is fine, and often desirable, to rewrite the code cleanly to match `nullsim` pipeline standards (typed stages, sub-budget contributions, no global state); the rule is that hard-won physical behavior should be captured by nullsim-native tests and invariants. The KILO and NAYRA repositories themselves are unmodified by this process and continue to serve as paper-reproducibility archives.

## 3. Goals and non-goals

**Goals**

- One package and one CLI that simulate any photonic nulling instrument whose physics is expressible as a sequence of stages between a stellar field and a detector.
- TOML configuration as the user-facing surface. Hand-editable, diff-friendly, version-controllable.
- An ordered sequence of stage instances: the user lists stage instances (each with its own `id` and `type`) in execution order, the runner validates that consumes/produces dependencies are satisfied before running, then executes in order.
- First-class parameter sweeps inside the config file. A study is one TOML file.
- A thin study layer above that, for comparing across configs (e.g. KILO vs. HWO on a shared exoplanet catalog).
- A standardized catalog of plots and tables, with `detection_space` (planet/star contrast vs. angular separation overlaid with the known exoplanet population) as the headline output.
- Stage-level result caching keyed by a conservative content hash so re-runs of unchanged studies are near-instant. Distinguish `config_hash` (resolved config only) from `run_hash` (config + code + data + catalog snapshots) so output identity is honest.
- Extensibility through two paths: an in-tree extension module listed in the config, and an installable plugin discovered via Python entry points.
- Reproducibility: each run writes its resolved config, a manifest of package and dependency versions, and a human-readable timestamped output directory by default (``results/{run.name}/{timestamp}``; ``{run_hash}`` and ``{config_hash}`` remain available as alternate template variables in ``output.dir``).

**Non-goals**

- `nullsim` is not a Fourier-optics simulator. It does not replace HCIPy, Poppy, or Prysm. If ground-truth wavefront propagation cross-checks are needed later, HCIPy is the natural fit; nothing in the core depends on it today.
- `nullsim` is not a job scheduler. Sweep parallelism is in-process via `concurrent.futures`; users running studies that exceed a single workstation use external schedulers around the CLI.
- `nullsim` does not aim for bit-identical reproduction of KILO or NAYRA paper figures. Those repositories are static design references; nullsim's CI checks its own physics, stage contracts, and example configs.

## 4. Architecture overview

```
config.toml --> [config layer]      pydantic schema, preset merge, sweep expansion
                       |
                       v
                [pipeline layer]    dependency validation, runner, caching, telemetry
                       |
                       v
                [stage layer]       physics stages consume/produce SimulationState
                       |
                       v
                [output layer]      registered plots and tables, sweep-aware
                       |
                       v
                results/<run>/<YYYYMMDD-HHMMSS>/  plots, tables, manifest, resolved config
```

Three guiding splits:

- **Math vs. pipeline.** Pure mathematical primitives (Kolmogorov spectrum, Clements decomposition, Ruilier–Cassaing coupling formula, Jones matrices, photometric zero points) live in `nullsim/physics/`. They have no knowledge of `SimulationState` or TOML and can be imported and tested in isolation. Stages in `nullsim/stages/` wrap that math into the pipeline.
- **Stages vs. families.** Stage families (atmosphere, AO, injection, transport, delay lines, chip, detector, postprocess) are organizational conventions, not Python inheritance hierarchies. Each family directory holds multiple alternative implementations selectable by `type = "..."` in TOML.
- **Catalog vs. registry.** A standardized catalog of named plots and tables ships with the package. The registry that backs the catalog is the same one user-defined outputs register into; user code and built-in code are indistinguishable to the runner.

## 5. Package layout

```
nullsim/
├── pyproject.toml
├── nullsim/
│   ├── __init__.py
│   ├── config/
│   │   ├── schema.py          # pydantic v2 models per stage family + top level
│   │   ├── loader.py          # TOML parse, preset merge, environment overrides
│   │   ├── sweeps.py          # grid/zip expansion, sweep coordinate algebra
│   │   └── presets/           # built-in instrument presets (TOML)
│   ├── pipeline/
│   │   ├── stage.py           # Stage protocol, StageParams, RunContext
│   │   ├── state.py           # SimulationState and sub-budget dataclasses
│   │   ├── registry.py        # stage/plot/table registry, plugin discovery
│   │   ├── runner.py          # dependency validation, execution, sweep dispatch
│   │   ├── cache.py           # content-hash keyed disk cache
│   │   └── _canonical.py      # type-tagged canonical-bytes hashing (used by state, rng, cache)
│   ├── stages/
│   │   ├── scene/             # StandardStarPlanet, PointSource, ExozodiModel
│   │   ├── telescope/         # ArrayGeometry, SingleAperture, MultiAperture
│   │   ├── atmosphere/        # KolmogorovAtmosphere, FixedStrehl, NoAtmosphere
│   │   ├── ao/                # PyramidWFS, ShackHartmann, NoAO
│   │   ├── injection/         # SingleModeFiber, MultiModeFiber, Ideal
│   │   ├── transport/         # SMF28, PM980, ZeroLossFiber
│   │   ├── delay_lines/       # GeometricDelayLine, NoDelay
│   │   ├── chip/              # KernelMZIMesh, ClementsMesh, Identity
│   │   ├── detector/          # MKID, SNSPD, EMCCD, IdealCounter
│   │   ├── sensitivity/       # DetectionCurve (per-sep throughput + disk floor + realized null)
│   │   └── postprocess/       # ChipOptimization, FringeTracking, Calibration,
│   │                          #   ImageReconstruction, PerformanceMetrics
│   ├── physics/
│   │   ├── kolmogorov.py
│   │   ├── ruilier_cassaing.py
│   │   ├── clements.py
│   │   ├── jones.py
│   │   ├── photometry.py
│   │   ├── pupil_geometry.py
│   │   ├── stellar_disk.py        # uniform-disk null floor integral
│   │   ├── planet_throughput.py   # PA-averaged dark-port throughput vs separation
│   │   ├── realized_null.py       # AO-piston Monte Carlo realized null + instability
│   │   ├── uv_coverage.py
│   │   └── fresnel.py
│   ├── sites/                 # built-in site database (TOML)
│   │   ├── maunakea.toml
│   │   ├── cerro_armazones.toml
│   │   └── l2_halo.toml
│   ├── targets/               # exoplanet catalogs, target-list helpers
│   ├── outputs/
│   │   ├── plots/             # registered plot functions
│   │   ├── tables/            # registered table writers
│   │   └── styles.py          # publication matplotlib styling
│   ├── data/                  # shipped reference curves and snapshots
│   │   ├── atran/             # atmospheric transmission per site
│   │   ├── fibers/            # SMF-28e, PM980-XP attenuation and birefringence
│   │   ├── photometry/        # Vega zero-points, filter curves
│   │   ├── kilo_reference/    # KILO paperv3 cache snapshots for plot overlays
│   │   └── loader.py          # path resolution, override via TOML
│   ├── study/                 # cross-config study layer
│   │   └── runner.py
│   └── cli.py                 # nullsim run|validate|list-stages|inspect|cache
├── examples/
│   ├── kilo_maunakea.toml
│   ├── nayra_eelt.toml
│   ├── hwo_space.toml
│   └── chip_only_lab.toml
└── tests/
    ├── physics/
    ├── stages/
    ├── outputs/
    └── examples/              # end-to-end config and smoke tests
```

## 6. Pipeline and stage model

### 6.1 Stage protocol

A `Stage` is the *type* (the implementation). A `StageInstance` is a *named occurrence* in the pipeline — one TOML entry with an `id`. The same `Stage` type can be instantiated multiple times (two `transport.smf28` instances, one for the sub-aperture-to-chip run and one for the chip-to-detector run).

```python
class Stage(Protocol):
    type_name: ClassVar[str]                  # registry key, e.g. "transport.smf28"
    family: ClassVar[str]                     # "atmosphere" | "transport" | ...
    consumes: ClassVar[frozenset[StateKey]]   # references resolved to instance IDs
    produces: ClassVar[frozenset[StateKey]]

    def __init__(self, params: StageParams, context: RunContext) -> None: ...
    def apply(self, state: SimulationState) -> SimulationState: ...
    def diagnostics(self) -> dict: ...
    def external_dependencies(self) -> list[ExternalDep]: ...
```

A `StateKey` is a dotted path like `"field"`, `"opd_budget.atmos_piston"`, `"geometry.baselines"`, or `"diagnostics.<instance_id>.<key>"`. The runner validates the pipeline before execution by checking that every consumed key is produced by some earlier instance. Mistakes such as listing chip before injection fail at validation time.

`diagnostics()` returns optional per-instance data keyed by instance ID so multiple instances of the same stage type stay distinguishable. The atmosphere instance exposes its r₀, AO Strehl per wavelength, and OPD variance components; the chip instance exposes its phase matrix and optimization history.

`external_dependencies()` declares external resources the stage reads (shipped data files, fetched catalog snapshots, external optimizer configurations). The runner hashes each declared dependency and folds it into the cache key. Stages do **not** define their own cache-key logic; the cache layer derives the key from a fixed recipe (see §6.4). This is the conservative-by-default choice — stages cannot accidentally under-include inputs.

### 6.2 SimulationState

`SimulationState` is a typed payload that flows through the pipeline. It is not one monolithic blob; it is a handful of named sub-budgets that stage instances opt into reading and writing:

| Sub-budget | Type | Producer families |
|---|---|---|
| `scene` | `Scene` — astrophysical inputs: star SED + angular diameter + distance, planet(s) contrast spectrum + (separation, PA) + optional phase, exozodi model, background sources, sky background spectrum | scene (source factory stages) |
| `geometry` | `Geometry` — array geometry: aperture positions in ENU and on the pupil, sub-aperture mapping, baselines as a function of time/hour angle, parallactic angle, UV coverage trajectory | telescope, array_geometry stages |
| `field` | `Field` — complex amplitudes of shape `[n_modes, n_wavelength_bins, n_pol]` with metadata for the wavelength grid and the mode-to-aperture mapping | injection, chip |
| `opd_budget` | `dict[str, OPDComponent]` — named RMS contributions (atmos_piston, fiber_dispersion, thermal_drift, ...) | atmosphere, transport, delay_lines, postprocess |
| `throughput` | `Throughput` — multiplicative transmission spectrum on the wavelength grid, with a per-component log | atmosphere, transport, chip |
| `wavefront` | `Wavefront` — Strehl and amplitude jitter per wavelength, per aperture | atmosphere, ao |
| `photon_rates` | `PhotonRates` — per-port, per-wavelength rates with named contributors (star, sky, dark, planet, zodi) | sky_background, detector |
| `results` | `Results` — final products: null depth, contrast curve, SNR, calibration time | postprocess |

`scene` and `geometry` are populated by dedicated stages at the front of the pipeline (`scene` and `telescope` family stages). They are load-bearing inputs to `injection`, `chip`, `detector`, and to outputs such as `detection_space`, `uv_coverage`, and `contrast_curve`. Stellar-diameter leakage, exozodi photon backgrounds, planet position-dependent throughput, and hour-angle-dependent UV coverage all read from these two sub-budgets.

Each sub-budget is a frozen dataclass with an `add_component(name, value)` constructor so contributions are traceable by name. A throughput stacked-area plot decomposed by component requires no extra bookkeeping inside individual stages.

### 6.3 Runner

```python
def run(config: ResolvedConfig) -> RunResult:
    state = SimulationState.empty(config.grid)
    instances = [registry.build_instance(spec, context) for spec in config.pipeline.stages]
    validate_dependencies(instances)            # consumes/produces check, not a topology pass
    for inst in instances:
        key = cache.key_for(inst, state, context)
        if cache.has(key):
            state, diag = cache.load(key)
        else:
            with telemetry(inst.id):
                state = inst.apply(state)
                diag = inst.diagnostics()
            cache.store(key, state, diag)
    return RunResult(state=state, diagnostics=collect(instances))
```

Stage execution is an explicit ordered sequence: the TOML lists stage instances in execution order, the runner respects it, and dependency validation only checks that every consumed key has been produced by an earlier instance. There is no topological reordering. Explicit beats clever for a configuration file that researchers will diff against published versions.

### 6.4 Caching

Stage-level caching is on by default and content-addressed with a **conservative default key recipe**. Stages do not define their own cache-key function; the cache layer computes the key from a fixed formula. This is the load-bearing safety property: a stage author cannot accidentally produce a stale cache hit by forgetting to include an input.

The cache key for a stage instance is the hash of:

1. The instance's resolved params (frozen, sorted).
2. A digest of every sub-budget the instance declares it `consumes`. The state object hashes its own sub-budgets by content.
3. A hash of the stage type's source file (the `.py` containing the class).
4. The package version.
5. The hashes of all `external_dependencies()` the stage declares (e.g. an ATRAN transmission curve, a NASA Exoplanet Archive snapshot, a fiber attenuation table). Stages opt **into** declaring additional externals — they do not opt out of the consumed-state digests.
6. The user-bumpable `cache.version` field in the config (escape hatch for forcing a rebuild).

Two related hashes the cache layer also computes for the run as a whole:

- **`config_hash`** = hash of `config.resolved.toml` only. Identifies "what the user wrote."
- **`run_hash`** = hash of `config_hash` + package version + every `external_dependencies()` hash across all instances + the runtime Python/numpy/scipy/torch versions. Identifies "what actually got produced."

`run_hash` is what names the output directory. Two runs with the same `config_hash` but different `run_hash` (because the exoplanet catalog snapshot changed, or because torch was upgraded) write to *different* directories. The manifest records both hashes.

Cache hits restore the post-instance `SimulationState` and diagnostics. Misses execute normally and write to disk. The cache lives in `.nullsim_cache/` next to the config by default. CLI flags `--no-cache` and `--clear-cache` provide the escape hatches.

Sweeps benefit directly: a sweep over `chip.n_modes` caches each unique value once. Re-running the same sweep, or a different sweep touching the same chip values, hits the cache.

### 6.5 Stochastic stages and RNG determinism

Most stages today are analytic noise-budget calculators that return deterministic outputs. Some physics — fringe-tracking residuals, thermal drift over an observation, telescope vibrations — naturally wants Monte Carlo. A stage opts in by setting `mode = StageMode.MONTE_CARLO` and reading `context.rng`.

The RNG handed to each stage instance is derived deterministically so that **the result of a stochastic run is independent of how the runner is parallelized**. `numpy.random.SeedSequence` only accepts integer entropy, but the inputs we want to mix in (`config_hash` is a hex string, `sweep_coord_tuple` is `tuple[(str, Any), ...]`, `instance_id` is a string) are not integers. The recipe is therefore a two-step canonicalization:

```python
# 1. Type-tagged canonical encoding of the non-int inputs (prevents collisions
#    between e.g. the string "1" and the integer 1), then SHA-256, then split
#    the digest into eight little-endian uint32 entropy words.
canon = canonicalize((config_hash, sweep_coord_tuple, instance_id))
digest = hashlib.sha256(canon).digest()
entropy_words = struct.unpack("<8I", digest)

# 2. Pass (root_seed, *entropy_words) to numpy SeedSequence.
ss = np.random.SeedSequence((int(root_seed), *entropy_words))
context.rng = np.random.default_rng(ss)
```

`root_seed` comes from `[run] seed`. `config_hash` and `sweep_coord_tuple` are known at config-load time. `instance_id` is the stage's TOML id. The resulting RNG stream is reproducible across runs and unchanged whether the run executes with `--workers 1` or `--workers 32`. The same trick gives every stage its own independent stream without manual seed plumbing.

The canonicalization helper, the SHA-256 step, and the LE-uint32 split are implementation details; the load-bearing property is that `derive_rng(root_seed, config_hash, sweep_coord_tuple, instance_id)` is a pure function of its arguments and that those four arguments uniquely identify a stage instance within a sweep cell. See `nullsim/pipeline/rng.py` for the source of truth.

### 6.6 Chip optimization as a separate stage

The chip optimizer is its own postprocess stage rather than a hidden side effect of the chip stage. The TOML lists `chip` followed by `chip_optimization` explicitly. Two reasons:

- A/B comparison of optimizers becomes a one-line config change.
- A user studying a perfectly-tuned chip versus one calibrated on photon counts can swap or omit the optimization stage without touching the chip stage.

The optimizer dispatches across backends (`scipy`, `torch`) through separate stage classes (`ChipOptimizationScipy`, `ChipOptimizationTorch`) sharing a common `ChipOptimizationParams` base, consolidating the chip optimization code that was previously spread across multiple files in KILO and NAYRA.

## 7. Configuration and sweeps

### 7.1 TOML schema

A complete configuration file:

```toml
# ─── Run metadata ─────────────────────────────────────────────
[run]
name = "kilo_sensitivity_paper_fig3"
description = "Contrast vs J-mag, 4 Maunakea telescopes, H-band"
seed = 12345
extensions = ["my_lab_package.stages"]

# ─── Spectral / spatial grid ──────────────────────────────────
[grid]
wavelength_center_um = 1.65
wavelength_bandwidth_um = 0.30
n_wavelength_bins = 32

# ─── Site & telescope (preset + override pattern) ─────────────
[site]
preset = "maunakea"

[telescope]
preset = "keck_pair_plus_subaru_gemini"

# ─── Scene (source models, populated into state.scene) ────────
[scene.star]
target = "tau_Ceti"        # resolves via astropy/SIMBAD, can override below
jmag = 4.5
angular_diameter_mas = 2.08
distance_pc = 3.65

[scene.planet]
contrast = 1.0e-7
separation_mas = 100.0
position_angle_deg = 45.0

[scene.exozodi]
level_zodi = 3.0           # in units of solar-system zodi

# ─── Observation geometry ─────────────────────────────────────
[observation]
hour_angle_h = 0.0
integration_time_s = 3600

# ─── Pipeline (ordered instance list, each with id + type) ────
[[pipeline.stages]]
id = "scene"
type = "scene.standard_star_planet"

[[pipeline.stages]]
id = "array"
type = "telescope.array_geometry"

[[pipeline.stages]]
id = "atmosphere"
type = "atmosphere.kolmogorov"

[[pipeline.stages]]
id = "ao"
type = "ao.pyramid_wfs"
n_actuators = 3500

[[pipeline.stages]]
id = "injection"
type = "injection.single_mode_fiber"

[[pipeline.stages]]
id = "fiber_to_chip"
type = "transport.smf28"
length_m = 30.0

[[pipeline.stages]]
id = "delay_lines"
type = "delay_lines.geometric"
length_m = 50.0

[[pipeline.stages]]
id = "chip"
type = "chip.kernel_mzi_mesh"
n_modes = 4
n_bright_ports = 3
n_dark_ports = 1

[[pipeline.stages]]
id = "chip_opt"
type = "postprocess.chip_optimization"
algorithm = "broadband_bfgs"
backend = "torch"
max_iter = 500

[[pipeline.stages]]
id = "fiber_to_detector"
type = "transport.smf28"
length_m = 5.0

[[pipeline.stages]]
id = "fringe_tracking"
type = "postprocess.fringe_tracking"

[[pipeline.stages]]
id = "detector_bright"
type = "detector.snspd"
ports = "bright"

[[pipeline.stages]]
id = "detector_dark"
type = "detector.mkid"
ports = "dark"
max_count_rate_hz = 50_000

[[pipeline.stages]]
id = "performance"
type = "postprocess.performance_metrics"

# ─── Sweeps (first-class) ─────────────────────────────────────
[[sweep]]
param = "scene.star.jmag"
linspace = { start = 4, stop = 12, num = 9 }
mode = "grid"

[[sweep]]
param = "pipeline.stages.chip.n_modes"
values = [3, 4, 5, 6, 8]
mode = "grid"

# ─── Outputs ──────────────────────────────────────────────────
[output]
dir = "results/{run.name}/{timestamp}"  # human-readable; {run_hash}/{config_hash} also supported
plots = ["detection_space", "contrast_curve", "snr_vs_jmag",
         "throughput_breakdown", "null_depth_vs_nmodes", "uv_coverage"]
tables = ["target_detectability", "throughput_budget", "performance_summary"]
formats.plots = ["pdf", "png"]
formats.tables = ["csv", "json", "parquet"]
sweep_table = "parquet"                  # tidy long-format sweep results

# ─── Catalog snapshot (versioned) ─────────────────────────────
[catalog.exoplanets]
source = "nasa_exoplanet_archive"
query  = "default"
snapshot = "2026-05-01"                  # resolves to data/catalogs/<snapshot>.parquet
classification = "bins_v2"               # named binning rule for Rocky/SE/Neptune/GG

# ─── Cache ────────────────────────────────────────────────────
[cache]
enabled = true
dir = ".nullsim_cache"
version = 1
```

Notes on the schema:

- **Stage instances** are addressed by `id` (`chip`, `fiber_to_chip`, `detector_bright`). The `type` field selects the implementation. Multiple instances of the same type are allowed (two `transport.smf28` instances, two detector instances on different port groups).
- **Sweep params** reference instance IDs (`pipeline.stages.chip.n_modes`) or scene/observation paths.
- The example shows the multi-instance shape that KILO actually needs (separate fibers before and after the chip, separate detectors on bright vs. dark ports), which the previous single-stage-per-family schema couldn't express.

### 7.2 Presets and overrides

`[site]`, `[telescope]`, `[scene]`, and `[observation]` accept `preset = "name"` which loads from the built-in TOML database, then deep-merges any inline fields the user provides. The user can override individual fields without copy-pasting a full preset. Pipeline stage lists do not use presets — they are explicit by design — but a stage's `params` table can reference a preset for stage-internal defaults.

### 7.3 Validation

Each stage class ships a pydantic v2 `Params` model. The loader:

1. Parses the TOML.
2. Merges presets for `[site]`, `[telescope]`, `[scene]`, `[observation]`.
3. Validates each `[[pipeline.stages]]` entry: the `type` resolves through the registry, and the remaining fields are validated against that type's `Params` model. Unknown keys are errors. Type mismatches surface as path-aware errors (`pipeline.stages[chip].n_modes: expected int, got "four"`).
4. Validates `[catalog.*]` snapshots resolve to on-disk files (or are fetchable).
5. Expands sweeps into a cross product of resolved configs.
6. Calls `validate_dependencies()` on the resolved pipeline (every `consumes` key matches some earlier instance's `produces`).

`nullsim validate config.toml` runs the full validation pipeline without executing anything.

### 7.4 Sweep semantics

Each `[[sweep]]` table declares one parameter axis. Values can be given as `values = [...]`, `range = {start, stop, step}`, `linspace = {start, stop, num}`, or `logspace = {start, stop, num}`. Mode is `grid` (default) or `zip`.

- All `grid`-mode sweeps form a Cartesian product.
- All `zip`-mode sweeps iterate in parallel; their joint length must agree, and they contribute one axis (the zipped tuple) to the cross product.

Output products are indexed by a tuple of sweep coordinates. Plot functions decide which axes to render against. Sweep cells are independent; the runner parallelizes via `concurrent.futures` with `--workers N`.

### 7.5 Reproducibility manifest

Every run writes alongside its outputs:

- `config.toml` — the original.
- `config.resolved.toml` — presets expanded, defaults filled in.
- `manifest.json` — `config_hash` and `run_hash`, package version, git commit (when in a repo), Python and dependency versions, RNG seeds, host info, per-instance timings, cache hit/miss summary, and per-instance `external_dependencies` (source + retrieval date + content hash for each).

Output directories are named by `run_hash`, not `config_hash`. That guarantees re-running with the same TOML but a different package version, dependency stack, or catalog snapshot writes to a different directory and does not silently overwrite earlier results. Both hashes are recorded in the manifest so you can group runs by either dimension.

## 8. Outputs

### 8.1 Standardized catalog

The catalog is the single source of named outputs. Each entry is a registered function `(RunCollection, RunContext) -> Figure` or `... -> DataFrame` with declared input requirements that are validated at config-load time. A request for `uv_coverage` when no stage produces baselines fails before the run starts, not in the plotting step.

**Plots (tier 1):**

| Name | What it shows | Sweep-aware |
|---|---|---|
| `detection_space` | Headline figure. Planet/star contrast vs. angular separation, one detection-limit curve per (instrument config × J-mag × integration time), filled region above each curve = detectable. Overlays the known exoplanet population: color = planet type (Rocky, Super-Earth, Neptune-like, Gas Giant), marker shape = discovery method (RV, Transit, Imaging, Other), optional text labels for notable systems. Stellar diameter assumption in the caption. Consumes contrast curves from one or more sweep cells, plus a *versioned* exoplanet catalog snapshot declared in `[catalog.exoplanets]` (source, query, retrieval date, snapshot hash, classification rule). The catalog snapshot hash enters `run_hash` so a catalog refresh produces a new output directory and a new manifest entry rather than silently shifting the points. | yes |
| `contrast_curve` | Achievable contrast vs. angular separation, one curve or a small set. | optional |
| `contrast_vs_<param>` | Contrast at fixed separation as a function of any swept parameter. | yes |
| `snr_vs_jmag` | SNR (or integration time to 5σ) vs. stellar J-magnitude. | yes |
| `null_depth_spectrum` | Instantaneous and calibrated null depth vs. wavelength. | no |
| `throughput_breakdown` | Stacked-area transmission vs. wavelength, decomposed by named component. | no |
| `opd_budget_bar` | Bar chart of OPD RMS contributions, named, stacked by reduction stage (open-loop → AO → FT). | no |
| `photon_rate_budget` | Per-port photon rates broken down by source (star, sky, dark). | no |
| `uv_coverage` | (u, v) coverage for the array and observation. | no |
| `pupil_layout` | Sub-aperture geometry on the pupil. | no |
| `chip_phase_matrix` | Optimized MZI phase settings (heatmap). | no |
| `chip_optimization_history` | Optimizer loss vs. iteration. | no |
| `calibration_time_vs_null` | Calibration time required to reach a target null depth. | yes |
| `target_detectability_map` | Sky scatter of exoplanet targets colored by achievable SNR. | no |

**Tables (tier 1):**

| Name | Contents |
|---|---|
| `throughput_budget` | Per-component transmission contributions, per wavelength bin. |
| `opd_budget` | Named OPD RMS contributions and their reduction stages. |
| `photon_rate_budget` | Per-port photon rates, star / sky / dark breakdown. |
| `target_detectability` | Per-target: RA, Dec, magnitude, planet contrast, achieved SNR, integration time to 5σ. |
| `chip_parameters` | Final MZI phase and amplitude settings for each cell. |
| `performance_summary` | Top-level scalar metrics: best null depth, contrast at 1λ/D, calibration time. |
| `run_manifest` | Tabular form of the JSON manifest, for paper inclusion. |

Plots default to PDF and PNG; tables default to CSV and JSON.

### 8.1.1 Sweep result storage

Sweeps are first-class, so sweep-aggregated output is a first-class concern, not a v2 extension:

- **`sweep_results.parquet`** — a tidy long-format table with one row per (sweep cell × output metric × wavelength bin, where applicable). Columns: every sweep coordinate, the metric name, the value, plus a `cell_run_hash` linking back to the per-cell directory. This is the file figures load from; it makes ad-hoc analysis in pandas trivial.
- **`sweep_manifest.json`** — collection-level metadata: the sweep axes, mode (grid/zip), cell count, cell `run_hash` list, total wall time, cache hit rate.
- **Per-cell directories** still exist under `cells/<cell_run_hash>/` for runs where you need per-cell plots or diagnostics, but the standard outputs never need to walk them.

Tables also write to CSV/JSON by default; Parquet is enabled via `formats.tables = ["csv", "parquet"]` or by setting `sweep_table = "parquet"`. For studies with 10³+ cells this is the difference between a usable artifact and a directory tree the OS struggles to list.

### 8.2 detection_space is load-bearing

The detection-space plot is the headline output and drives two design choices:

- Plot functions take a `RunCollection`, not a single `RunResult`. Detection space consumes contrast curves from many sweep cells (J-mag axis) at once.
- Cross-config comparison needs a thin **study layer**. Within one TOML, sweeps cover one config × many param values. Comparing across configs (KILO vs. HWO vs. NAYRA) is the next level up.

### 8.3 Study layer

A `study.toml` references multiple run configs by path and tells the output system which runs to overlay:

```toml
[study]
name = "kilo_vs_hwo_paper_fig"
runs = [
  { config = "kilo_maunakea.toml", label = "KILO Maunakea" },
  { config = "hwo_space.toml",     label = "HWO baseline"  },
]

[output]
plots = ["detection_space", "contrast_curve_overlay"]
formats.plots = ["pdf", "png"]
```

`nullsim study run study.toml` executes each referenced config (cache-aware), then dispatches outputs against the combined `RunCollection`. No new pipeline machinery is required; the study layer is a thin multiplexer.

## 9. Extensibility

Two paths, intentionally redundant:

**In-tree (one-off custom code).** Write a Python module that exposes `register(registry)`. List it under `[run] extensions = [...]`. The loader imports it and calls its `register()` function. Lowest barrier; right for a paper-specific custom stage.

```python
# my_thermal_drift.py
from nullsim.pipeline import Stage, register_stage

@register_stage
class ThermalDrift(Stage):
    name = "thermal_drift"
    family = "postprocess"
    consumes = frozenset({"opd_budget", "chip.diagnostics"})
    produces = frozenset({"opd_budget.thermal"})
    ...
```

**Out-of-tree (installable plugin).** Declare a `nullsim.stages` (or `.plots` or `.tables`) entry point in your package's `pyproject.toml`. The registry discovers it at import time. Right for code that more than one project depends on.

```toml
# external pyproject.toml
[project.entry-points."nullsim.stages"]
my_custom_chip = "my_pkg.stages:MyChip"
```

Both paths produce stages, plots, and tables that are indistinguishable from built-ins to the runner: same schema validation, same caching, same diagnostics. The CLI `nullsim list-stages` and `nullsim list-outputs` show built-ins and user-registered entries in one list.

## 10. CLI

```
nullsim run    config.toml [--out DIR] [--workers N] [--no-cache] [--dry-run]
nullsim study  run study.toml [--workers N] [--no-cache]
nullsim validate config.toml                  # schema + dependency validation, no execution
nullsim list-stages [--family chip]           # show registered stages
nullsim list-outputs                          # show registered plots & tables
nullsim inspect config.toml                   # resolved config + dependency visualization
nullsim cache info|clear [--config config.toml]
```

`nullsim run` is the main entry point. `nullsim inspect` is the discoverability tool: it prints the resolved config, the stage instance list with consumes/produces annotations as ASCII art (or a Graphviz file with `--graphviz`), and a table of which outputs each instance feeds. New users learn the system through `inspect`.

## 11. Testing strategy

Three layers:

1. **Physics primitives** in `nullsim/physics/` get pure unit tests. Kolmogorov spectrum normalization, Clements decomposition invertibility, Ruilier–Cassaing analytical limits.
2. **Each stage** gets a contract test: build a minimal `SimulationState` containing only what the stage `consumes`, run the stage, assert that every key in `produces` appears and that conservation properties hold (throughput ≤ 1, photon rates non-negative, OPD components RMS-stack correctly).
3. **Example and integration tests** run shipped configs far enough to catch wiring mistakes, missing components, and broken output contracts. KILO/NAYRA-derived formulas still get local unit or stage tests when their physical assumptions are load-bearing, but CI does not compare nullsim against frozen external scalar outputs.

A future `nullsim/validation/` package could wrap HCIPy for ground-truth checks of injection coupling and atmospheric OPD stages. None ship today; HCIPy is intentionally absent from the dependency surface until a real cross-check is written.

## 12. Shipped examples

Four example configs ship with the package and anchor the design. They double as executable examples for end-to-end config checks.

| File | What it reproduces |
|---|---|
| `kilo_maunakea.toml` | Four-telescope Maunakea KILO-style J-mag sweep. |
| `nayra_eelt.toml` | Single-pupil E-ELT NAYRA-style study. |
| `hwo_space.toml` | HWO-class space nuller. No atmosphere stage, formation-flying delay budget instead of AO + atmosphere. MKID detector. |
| `chip_only_lab.toml` | Bare chip + detector. No telescope, no atmosphere. Lab characterization mode. |

A fifth file, `examples/kilo_vs_hwo_study.toml`, is a study-layer config that overlays the KILO and HWO configs in one detection-space figure.

## 13. Dependency policy

Python target: **`requires-python = ">=3.12"`**. `tomllib` is stdlib from 3.11+; no third-party TOML parser needed. Dev environment for this project is conda `py313`; the package itself does not require 3.13.

Core dependencies (required for `pip install nullsim`):

- `numpy`, `scipy`, `matplotlib`, `astropy`, `pandas`, `pyarrow` (Parquet), `pydantic >= 2`
- `tqdm` (sweep progress)

Optional dependencies (extras):

- `torch` — chip optimizer fast path on CPU/CUDA
- `pyvo` — live NASA Exoplanet Archive TAP queries (ships with cached CSV fallback)

`kilo` and `nayra` are not runtime dependencies — `nullsim` does not `import kilo` at runtime, and `pip install nullsim` does not pull them in. They are however useful source references for physical models, kept in separate repositories outside this tree. New `nullsim/physics/` and stage code may be a clean rewrite to fit pipeline conventions, but every hard-won physical lesson should land in `nullsim` as a local test, not as folklore.

## 14. Open questions

These are not blocking the design but want answers before the implementation plan is final.

1. **Site and telescope preset ownership.** The built-in presets ship with the package. Should new preset additions (e.g. a new VLT configuration) live in the user's project or upstream in `nullsim/sites/`? My recommendation: presets that correspond to real, published instruments ship upstream; experimental presets live in user projects.
2. **Polarization Jones tracking.** The `field` sub-budget has a `n_pol` axis. NAYRA's `polarization.py` is currently stub-level. Should Jones-matrix propagation be a first-class stage family (`polarization`), or does it live inside transport/chip stages? My recommendation: a `polarization` family that defaults to a `scalar` no-op stage; dual-pol instruments swap in a `jones_chain` stage.
3. **Time-domain output mode.** Stochastic stages return distributions today. If multiple stages are stochastic and the user wants a true time series of detector counts, that is a separate output type the catalog does not yet describe. Defer to a v2 design pass.
4. **Catalog snapshot distribution.** Exoplanet catalog snapshots are versioned and hashed, but the *distribution channel* is open: ship a frozen snapshot with the package and refresh on release, or fetch on first use and cache locally? My recommendation: ship a frozen snapshot (last 30 days before release), allow `nullsim catalog refresh` to fetch a fresher one explicitly. Avoids first-run network dependency for reproducibility.

## 15. Implementation roadmap

Out of scope for this design doc, but to anchor expectations. The first milestone is a **vertical slice** that exercises the architecture end-to-end before any real physics gets ported — that way the interfaces get pressure-tested while they are still cheap to change.

1. **Vertical slice (architecture validation).** Build the minimum pieces of every layer at once:
   - `nullsim/pipeline/` skeleton: stage protocol, `SimulationState` with `scene`/`geometry`/`field`/`throughput`/`photon_rates`/`results`, runner, conservative cache, registry, RNG derivation.
   - `nullsim/config/` skeleton: TOML loader, pydantic schemas, sweep expansion, preset merge.
   - 4–5 *trivial* stages: `scene.point_source`, `telescope.fixed_aperture`, `injection.ideal`, `chip.identity`, `detector.ideal_counter`.
   - One plot (`contrast_curve`) and one table (`performance_summary`).
   - Manifest with `config_hash` and `run_hash`.
   - End-to-end test: a 5-line `vertical_slice.toml` runs, writes outputs, cache hits on re-run, sweep over `scene.star.jmag` produces a `sweep_results.parquet`.
2. Build `nullsim/physics/` and `nullsim/optimization/` guided by the KILO and NAYRA source trees. The math may be rewritten cleanly to fit `nullsim` conventions (typed signatures, no global state, explicit units, dependency injection at boundaries), but the hard-won lessons must come along. For each target module (kolmogorov, ruilier_cassaing, clements, jones, photometry, fresnel, chip optimizer backends), the workflow is: (a) read the KILO/NAYRA implementation in full, including comments and any `# fixme/# note` markers, and write a short notes file capturing the non-obvious invariants (sign conventions, dtype choices, edge cases, ordering constraints); (b) lift the corresponding KILO/NAYRA tests into `tests/physics/` first so those invariants are pinned before any rewrite; (c) write the new `nullsim` implementation, refactoring naming and signatures, and verify every lifted test still passes; (d) add new tests for any behavior the original lacked coverage for. A clean rewrite that drops a debugged behavior because "the paper doesn't say so" is the failure mode this guards against.

   **Step 2 first scoped pass (2026-05-15):** three modules shipped — `nullsim/physics/kolmogorov.py` (Kolmogorov phase PSD + structure function + r0 wavelength scaling, cross-checked between KILO and NAYRA atmosphere implementations), `nullsim/physics/photometry.py` (J-band Vega zero-point + magnitude→photon-rate per Cohen+ 2003), and `nullsim/physics/ruilier_cassaing.py` (single-mode-fiber coupling efficiency vs Strehl with optional central obscuration, Ruilier 1999 analytical limit pinned at η ≈ 0.8145 for matched circular aperture). Each module ships with a notes file in `tests/physics/notes/<module>.md` capturing the cross-check decisions and adopted constants. Remaining step-2 modules (clements, jones, fresnel, chip optimizer backends) ship in subsequent passes.
3. Port the real stages: atmosphere (Kolmogorov), AO (pyramid WFS), injection (Ruilier–Cassaing), transport (SMF28, PM980), delay lines, chip (kernel MZI mesh), chip_optimization, detectors (MKID, SNSPD), fringe tracking, calibration, performance. Each stage wraps the already-built `nullsim/physics/` primitives. Where KILO's structure crosses what is now a stage boundary (e.g. a single KILO function that does atmosphere + AO), split along the boundary but verify behavior end-to-end against the corresponding KILO run.

   **Step 3 first scoped pass (2026-05-16):** two stages shipped — `nullsim/stages/atmosphere/kolmogorov_screens.py` (KolmogorovScreens wraps the Kolmogorov physics primitive; writes `OPDBudget["atmos_piston"]` and a typed `Wavefront.r0_at_500nm_m` channel) and `nullsim/stages/ao/pyramid_wfs.py` (PyramidWFS sums KILO's three residual terms — fitting, temporal, WFS-noise — in quadrature; marks the atmosphere contribution as superseded via `OPDComponent.superseded`). KILO numeric references pinned at rel=0.01 in the stage tests. Remaining step-3 stages (injection real, transport SMF28/PM980, delay lines, chip kernel MZI mesh, chip_optimization, detectors MKID/SNSPD, fringe tracking, calibration, performance) ship in subsequent passes.

   **Step 3 second scoped pass (2026-05-16):** three stages shipped — `nullsim/stages/injection/single_mode_fiber.py` (SingleModeFiber wraps the Ruilier–Cassaing physics primitive; multiplies the field amplitude by √coupling and writes a per-wavelength `injection.smf_coupling` throughput component), `nullsim/stages/transport/smf28.py` (SMF28 applies fiber loss + chromatic dispersion phase + per-telescope common-mode phase noise; writes `transport.smf28` throughput component and `transport.smf28_phase` OPD component), and `nullsim/stages/delay_lines/geometric.py` (GeometricDelayLines applies pointing-dependent path-equalization delays; writes `delay_lines.geometric` throughput component and `delay_lines.geometric_residual` OPD component). All three components adopt the canonical `Throughput.components` payload shape `{wavelength_grid_um, transmission_fraction, +provenance}`, and `Scene` gains typed `altitude_deg`/`azimuth_deg` fields replacing the prior string-keyed `components` lookup. Polarization-aware variants (PM980 + Jones-fiber) deferred to step 3 third pass alongside chip work.

   **Step 3 third scoped pass (2026-05-16):** chip stage + optimizer backends shipped — `nullsim/stages/chip/kernel_mzi_mesh.py` (KernelMZIMesh wraps the Clements physics primitive; applies a per-wavelength M×M unitary to the field tensor on the n_modes axis with `theta` unscaled and `phi`/`alpha` rescaled by `wl_design/wl_target`; DAC-quantization applied BEFORE wavelength rescaling; writes a unit-transmission `chip.kernel_mzi_mesh_loss` throughput component since unitaries preserve power) replaces the `chip.identity` placeholder, plus three optimizer-backend stages under `nullsim/stages/chip_optimization/{scipy,torch,mlx}.py` (scipy uses L-BFGS-B + optional SLSQP polish; torch uses lazy-import + real/imag-split autograd with TF32 disabled per KILO's hard-won basin-flip lesson; mlx uses Apple Metal float32 + same real/imag trick). Optimizers write `optimal_chip_params` into `state.results.components` rather than threading chip params through `SimulationState` (Pass 3 design decision per Simplifier YAGNI verdict — `state.py` is locked, optimal params are read out by the user and baked into a follow-up TOML). Optional dependency extras `[torch]` and `[mlx]` added to `pyproject.toml`. KILO pinning tests `TestOptimizeNull::test_deep_null_small_system` and `TestClementsBackendsAgree` lifted as xfail-strict tripwires; nullsim-native cross-backend agreement tests run at 1% relative tolerance. Polarization-aware variants (DualPolField + Jones-fiber + PM980) deferred to step 3 fourth pass alongside polarization-aware injection/transport.

   **Step 3 fourth scoped pass (2026-05-16):** detector + post-detection stages shipped — `nullsim/stages/detector/mkid.py` (MKIDDetector wraps `nullsim/physics/photometry.py` to convert chip-output `|field|²` × throughput into per-port photon counts; per-pixel saturation cap default 50 kHz KILO / 100 kHz NAYRA; no MKID dark counts), `nullsim/stages/detector/snspd.py` (SNSPDDetector adds 10 Hz dark rate and 1e8 Hz fixed cap), `nullsim/stages/fringe_tracking/closed_loop.py` (ClosedLoopFringeTracker quadrature-sums Shao-Colavita 1992 shot-noise + Conan 1995 servo-lag piston RMS; emits `fringe_tracking.closed_loop_residual` OPDComponent and marks the upstream `atmos_piston` as superseded via the Pass 1 INS-8-001 pattern; KILO `predictive_lag_reduction=1.0` default vs NAYRA Kalman 0.05 captured as a Param), `nullsim/stages/calibration/floor_lookup.py` (CalibrationFloorLookup applies a default ~1e-6 systematic floor or interpolates a pre-computed JSON lookup table; clamps `null_depth_calibrated = max(raw - floor, 0)` since physical nulls cannot go negative), and `nullsim/stages/performance/standard.py` (StandardPerformance terminal stage; `SNR = S / √(S + B + D + Sky + systematic)` with the systematic term `eps²B²·t/(2·f_servo)` when fringe tracking is in the pipeline or `(eps·B)²` without — KILO `signal_to_noise` lines 204-292; reads port rates once from the detector component to honour KILO's `detected_stellar_rate_at_telescope` consolidation lesson). 446 passed + 3 xfailed (was 367 + 3; +79 tests). Polarization-aware variants and per-arm fringe-tracking diagnostics deferred to a later pass.
4. Build out the standardized output catalog including `detection_space` and the catalog-versioning machinery.

   **Step 4 first scoped sub-pass (2026-05-16):** headline `detection_space` plot + catalog-versioning machinery shipped — `nullsim/outputs/plots/detection_space.py` (contrast-vs-separation curves with detectable-region fill, per-(config×J-mag×integration-time) cell, type/method-coded exoplanet scatter overlay, footer caption with stellar diameter + catalog snapshot SHA-256 + retrieval date), `nullsim/catalogs/exoplanets.py` (`ExoplanetCatalogConfig`, `Exoplanet` pydantic models, `load_catalog` for IPAC/NASA-Exoplanet-Archive CSV, `classify_planet` versioned rule with Rocky/Super-Earth/Neptune-like/Gas-Giant thresholds, `compute_snapshot_hash` for SHA-256 reproducibility), `nullsim/config/catalog.py` (TOML schema for `[catalog.exoplanets]`). `nullsim/cli.py` `_run_hash_with_externals` now folds the catalog content hash into `run_hash`, so a catalog refresh produces a NEW output directory rather than silently shifting points on the headline figure. 479 passed + 3 xfailed (was 446 + 3; +33 tests, including 5-planet fixture catalog covering all four planet types and discovery methods). Remaining sub-passes covered supporting plots, supporting tables, and the study-layer multi-config multiplexer.
5. Ship `kilo_maunakea.toml` and functional tests for the KILO-style sensitivity example.

   **Step 5 first scoped pass (2026-05-16):** `examples/kilo_maunakea.toml` shipped as a 4-telescope KILO-style Maunakea config with a J-band sweep. Early development used external scalar references; those reference-comparison tests were later removed once nullsim's own physics/stage contracts became the active test surface.
6. Ship `nayra_eelt.toml` and functional tests for the NAYRA-style E-ELT example.

   **Step 6 first scoped pass (2026-05-16):** `examples/nayra_eelt.toml` shipped as a single-aperture 39 m E-ELT config with no delay-line stage, MKID cap 100 kHz, `predictive_lag_reduction=0.05` for the Kalman/LQG fringe tracker, and a J-mag sweep.
7. Ship `hwo_space.toml` as the first config that did not exist before the new package.

   **Step 7 first scoped pass (2026-05-16):** HWO-class space nuller config shipped — `examples/hwo_space.toml` (6 m monolithic primary at Y-band λ=1.0 µm, 1-day integration, M-dwarf-like host at 10 pc, planet contrast=1e-10 at 100 mas, exozodi 3× solar, pipeline strips atmosphere/AO/fringe-tracking/delay-lines, calibration floor 10⁻¹⁰), `tests/examples/test_hwo_space_sanity.py` (sanity asserts include end-to-end pipeline run, design-floor pin at 10⁻¹⁰, no atmosphere/AO/FT components in the OPD budget, contrast-sweep monotonicity, detection_space PNG smoke), and `examples/kilo_vs_hwo_study.toml` (cross-config study-layer placeholder per SPEC §8.3 — runner deferred per task #36 sub-pass 36d, but the TOML schema is in place).

   **HWO photon-baseline and finite-bandwidth warm-start pass (2026-05-21):** `examples/hwo_space.toml` now models an 8 m unobstructed HWO-class primary at 0.8 µm with a 10% band, 24 pupil modes, six bright/dump ports, 18 dark ports, all-sky observability, dual-pol PBS/PM-fiber transport, and an ideal order-3 kernel projector for the photon-noise upper bound. `nullsim/stages/chip_optimization/ideal_kernel.py` adds the `chip_optimization.ideal_kernel` stage, which routes the stellar Taylor subspace into bright ports and can either publish an achromatic `optimal_u_stack` or decompose the center-wavelength unitary into a physical Clements seed. `chip_optimization.torch` can now warm-start from an upstream `chip_optimization` component via `warm_start_source`, enabling physical Clements refinement from the ideal HWO kernel instead of cold-starting the 576-parameter mesh.

   **Shared chip-chromaticity pass (2026-05-21):** `chip.kernel_mzi_mesh`, `chip_optimization.scipy`, and `chip_optimization.torch` gained opt-in parametric chromaticity fields. Default `chip_chromaticity_model = "legacy"` preserves KILO/NAYRA behavior: theta is wavelength-flat and phi/alpha scale as `wl_design / wl`. Setting `chip_chromaticity_model = "parametric"` enables additive directional-coupler theta drift (`theta_chromaticity_*`) and a selectable phase model, including `phase_chromaticity_model = "fixed_opd_neff"` with first-order effective-index dispersion from `phase_neff_ref` and `phase_group_index`.

   **HWO fixed-OPD deployment pass (2026-05-21):** `examples/hwo_space.toml` now deploys the ideal order-3 kernel seed as a signed-real physical Clements decomposition (`deployment = "physical_real_clements"`) and propagates it through `chip.kernel_mzi_mesh` with `chip_chromaticity_model = "parametric"` and `phase_chromaticity_model = "fixed_opd_neff"`. This keeps the default HWO example on a physical mesh while avoiding internal pi phase flips that would otherwise dominate finite-bandwidth fixed-OPD leakage for the real-valued HWO kernel projector.

   **HWO physics-path cleanup pass (2026-05-21):** the HWO pupil surrogate now preserves the full 8 m unobstructed collecting area and rescales its 24 mode-center coordinates so the maximum surrogate baseline is 8 m. The photon-limited baseline retains the PBS split and per-arm chip deployment but removes the nonideal PM-fiber Jones/PER perturbation; PM780 propagation loss remains. `performance.standard` now prevents a sensitivity-stage ideal null from hiding detector/calibration residual leakage and uses the sensitivity-stage throughput interpolated at `scene.planet.separation_mas` for headline SNR and integration-time calculations.

   **HWO optical photometry pass (2026-05-21):** `nullsim.physics.photometry` gained a Cousins I-band Vega zero point and compact I-J color conversion so HWO's 0.8 µm photon budget no longer uses the near-IR Y/J placeholder. Detector stages now expose an opt-in `derive_band_mag_from_scene_jmag` switch; KILO/NAYRA defaults remain unchanged, while `examples/hwo_space.toml` sets `band = "I"`, `scene.star.spectral_type = "M3V"`, and derives the I-band magnitude from the catalog J magnitude.
8. Documentation, CI, packaging.

   **Step 8 first scoped pass (2026-05-16):** closing polish — `README.md` (elevator pitch, install table covering base/`[torch]`/`[mlx]`/`[validation]` extras, quick-start table over all six example configs, CLI cheat sheet, architecture summary linking back to nullsim.md and SPEC.md, v0.1 ship list, MIT license note), `CONTRIBUTING.md` (~140 lines covering branch-PR workflow, build pipeline pointer, pre-commit guards including spec-discipline + librarian backlog ≤5, pytest marker reference, and fragment-changelog → Librarian discipline). `pyproject.toml` extended with `readme = "README.md"`, MIT license, authors/keywords/classifiers, `[project.urls]` placeholder, and `[validation] = ["hcipy>=0.7"]` extra for the optional ground-truth atmospheric/injection cross-check. `.github/workflows/ci.yml` ships the test matrix on Ubuntu × Python {3.12, 3.13}; pip-cached via `actions/cache@v4` keyed on the pyproject hash. With this pass the SPEC §15 roadmap is complete: vertical slice + physics primitives + four scoped stage passes + detection_space + KILO/NAYRA-derived examples + HWO + docs/CI/packaging.

   **Chip-optimizer KILO tricks pass (2026-05-16):** `nullsim/stages/chip_optimization/scipy.py` restart loop refactored from a fixed 16-restart sequential bank into KILO's leaner pattern (kilo/chip.py L1100-1160). `ChipOptimizationParams` gains four new fields: `n_restarts` (default 5, was hardcoded 16 — matches KILO empirical "deepest basin found in first 1-5 restarts at M=24"), `restart_patience` (default 3, KILO `KILO_BB_PATIENCE` analogue — bail when K consecutive restarts don't improve over running best by `restart_patience_reltol`), `restart_patience_reltol` (default 0.01 = 1%/restart), and `n_workers` (default 1 — chip-restart ProcessPoolExecutor over BFGS restarts with spawn context, top-level worker `_run_one_restart_worker` for pickle safety). Pool mode dispatches all restarts in one shot (no mid-pool early exit, matching KILO's process-pool semantics); serial mode keeps both target-null and patience early exits. SLSQP polish step gets its own local `_polish_obj` closure now that the restart loop's `_obj` lives inside the worker. `examples/kilo_keck.toml` updated: `n_restarts=5`, `restart_patience=3`, `n_workers=2` → 5 outer sweep cells × 2 inner chip-restart workers = 10 cores in flight. 605→611 tests passing.

   **Debug-report pass (2026-05-16):** new `nullsim/outputs/plots/debug_report.py` writes a multi-page `debug_report.pdf` that walks the executed pipeline stage-by-stage. Each page covers one stage's contribution to throughput / OPD / wavefront / results sub-budgets (read from the `components` dicts on the final state — no runner-level snapshots needed for v1). Cumulative pages at the end show running throughput, OPD quadrature sum, and per-port detector rates. Intent: a single-click sanity check that surfaces every load-bearing quantity the pipeline produces, so silent regressions like the post-figure-port bugs become visible without a deep dive. Added to `examples/kilo_keck.toml` `[output.plots]` so any run automatically renders it. 613→615 tests.

   **Figure-port pass (2026-05-16):** sanity-check figure ports from `kilo/studies/paperv3_plots.py` so the headline `detection_space` plot isn't the only visual output. `nullsim/outputs/styles.py` PUBLICATION_RC replaced with KILO's verbatim `_setup_style()` + `_configure_matplotlib()` (cmr10 serif, font sizes 12/14/11, lines.linewidth=1.5, ticks-in, figure 8x6 @ 150 dpi) and auto-applied at import. Six figure ports shipped: `nullsim/outputs/plots/subaperture_layout.py` (KILO paperv3 Fig 3 — per-telescope panel showing pupil boundary, central obstruction, spider vanes, optimized sub-aperture circles), `nullsim/outputs/plots/uv_coverage_grid.py` (KILO paperv3 Fig 5 — 4-dec x 3-integration grid of (u,v) tracks with intra-/inter-telescope coloring), `nullsim/outputs/plots/chip_diagnostics.py` (KILO paperv3 Fig 7 — 4-panel chip diagnostic: per-port stellar power distribution, |U| + arg(U) heatmaps, null vs DAC bit depth, broadband null spectrum; chip-input field recovered via U^H @ output to avoid new diagnostic state), and `nullsim/outputs/plots/contrast_curves_multiband.py` (KILO paperv3 Fig 10 — `1 × n_integration_times` panels of detectable-contrast vs angular-separation, with one curve per band; bands derived from each record's `grid.wavelength_center_um`, times from `observation.integration_time_s`, so the user controls the sweep axes in TOML), `nullsim/outputs/plots/loss_budget.py` (KILO paperv3 Fig 4 — per-band stacked-dB pre-chip loss bars; scalar adaptation since nullsim has no dual-polarization arms yet — drops KILO's scalar-vs-dual comparison axis, keeps the per-stage decomposition read out of `state.throughput.components`), and `nullsim/outputs/plots/calibration_time.py` (KILO paperv3 Fig 9, scoped down — chip optimizer convergence curve: null depth vs objective evaluations. Single panel only since KILO's right panel needs photon-noise-aware optimization + wall-clock mapping that nullsim doesn't have yet. Optimizer convergence is captured via a new `capture_convergence_history` flag on `ChipOptimizationParams` plumbed through the scipy backend). New physics primitive `nullsim/physics/uv_coverage.py` ports KILO's `Baseline` + `baselines_to_uv` + `compute_uv_tracks` + `enumerate_subaperture_baselines`. The Fig 5 port reproduces `kilo_keck.toml`'s 276 baselines (132 intra + 144 inter) and the 80-96 m inter-Keck range. Same pass added `[run].n_workers` + `[run].threads_per_worker` + `nullsim run --workers N` for ProcessPoolExecutor-based sweep-cell parallelism (KILO chip.py L1239 pattern: spawn context, top-level worker, as_completed streaming). Defaults to `min(n_cells, cpu_count)`. Five-cell `kilo_keck.toml` runs 5x faster end-to-end. 605 passed (was 570; +35 tests). Remaining figure ports — Fig 8 char_mode (deferred until characterization-mode chip lands) — queued. Fig 9 right panel (null vs calibration budget at multiple J-mags) also deferred until photon-noise-aware chip optimization is wired in.

   **Post-§15 hardening pass (2026-05-16):** independent review surfaced six correctness bugs that all landed fixes in one pass: (1) `injection.single_mode_fiber` + `transport.smf28` double-counted optical loss (scaled both field amplitude AND `throughput.transmission_fraction`, then the detector multiplied them — squared every detector rate by the loss factor); (2) `nullsim/cli.py` `_handle_run` used the pre-materialized resolved config for `run_hash` but the materialized cells[0] hash in the manifest, so output directory and manifest disagreed about which config ran; (3) `chip_optimization/torch.py` + `chip_optimization/mlx.py` produced Clements params in KILO's column-alternating layout while `chip.kernel_mzi_mesh` reconstructed via the lexicographic `physics.clements` layout — same params → different unitary → broken optimizer-to-kernel handoff; (4) `calibration.floor_lookup` read its lookup table at runtime but did not declare it as an external dependency, so an edit to the table reused the cache and the same `run_hash`; (5) `nullsim/catalogs/exoplanets.py` `_autocompute_snapshot_hash` short-circuited on pinned hashes without verifying them against file bytes; (6) `ChipOptimizationParams.n_dark_ports` accepted 0 and silently returned NaN loss. Same pass added `nullsim/physics/pupil_geometry.py` (line-for-line port of KILO's `kilo.array.optimize_subaperture_positions` + `kilo.pupil` constraint helpers) so `PyramidWFS` can derive `piston_subaperture_diameter_m` from the pupil geometry (`n_subapertures_per_aperture` + `pupil_template`) instead of pinning a constant in TOML.


   **Post-figure-port bugfix pass (2026-05-16, commits e4d84f0 + ba5095d):** two correctness bugs surfaced after the figure-port pass. (1) `injection.ideal` constructed a fresh `Wavefront()` without copying upstream `wavefront.components`, silently wiping any OPD or amplitude contributions written by atmosphere/AO stages — fixed by passing `components=dict(state.wavefront.components)` to the constructor. (2) `detector.mkid` and `detector.snspd` computed `null_depth_raw` from per-port rates AFTER the saturation cap, QE scaling, and dark-count addition — when both bright and dark MKID ports saturated, the ratio collapsed to the pixel-count ratio (~1e-3) regardless of the actual chip null; fixed by computing `null_depth_raw` from `port_rate_hz_per_wavelength` BEFORE saturation/QE/darks, matching KILO's `null_depth()` convention (`kilo/performance.py` L164).
   **Chip-optimizer KILO port pass (2026-05-16, commits 22fb83a, 06dda42, 8d25657, c28aac8, 64e7d05):** four changes landing together complete the chip optimizer's structural alignment with KILO. (1) The `debug_report.pdf` plot (`nullsim/outputs/plots/debug_report.py`) is now wired into `kilo_keck.toml` `[output.plots]` and a follow-up fix (06dda42) corrects the injection-stage throughput payload to the canonical dict shape `{wavelength_grid_um, transmission_fraction, +provenance}` so `debug_report` and `loss_budget` read it without a key error. (2) `ChipOptimizationParams` gains an `objective` field (default `"absolute_leakage"`); all three backends now minimize `sum |U[dark_ports,:] @ field_stack|²` — the absolute leakage summed over dark ports and wavelengths — exactly matching KILO's chip optimizer objective (c28aac8). The prior relative-leakage formulation is retained as `"relative_leakage"` for backward compatibility. (3) The torch backend (8d25657) gains KILO's restart-loop pattern: `n_restarts` outer restarts driven by `restart_patience`/`restart_patience_reltol` early-exit, and a `torch.compile` gate that engages only when CUDA is available or the user opts in — avoids JIT overhead on CPU-only developer runs. (4) `ChipOptimizationParams` gains `kernel_order` (int 0/2/3/4, default 0); when non-zero the optimizer targets a kernel-null field stack instead of the on-axis stellar field (`_kernel_null_field_stack`: direct port of KILO `injection.py:kernel_null_field_stack` L226-330, kernel order 2 → G(r)~r⁴, order 4 → G(r)~r⁸). `ao.pyramid_wfs` now writes `state.geometry.aperture_positions_pupil_m` (M×2, pupil-frame ENU positions derived from `optimize_subaperture_positions`); `chip_optimization` reads this when `kernel_order > 0` and raises a clear error if absent (64e7d05).

   **Sensitivity correctness pass (2026-05-16, commits 118c5dc, 1fda201):** two correctness fixes to `nullsim/stages/sensitivity/detection_curve.py` that together close a ~10000x contrast error at 300 mas. (1) `_find_ao_piston_rms_per_sub_m` now prefers the raw AO-piston residual in `state.wavefront.components['ao_residual'].piston_opd_nm.total` (KILO `transport.realized_null_depth ao_piston_opd_rms_per_sub` convention); the fringe tracker corrects only the common inter-telescope piston, so the within-telescope differential piston that drives the chip null floor is the AO-stage output, not the FT residual. Falls back to the closed-loop FT residual in `state.opd_budget.components` (key prefix `fringe_tracking.closed_loop_residual`) for space pipelines where no AO stage runs; `DetectionCurve.consumes` extended to include `opd_budget`. (2) `_split_apertures_to_modes` now uses the pupil-packed sub-aperture diameter instead of the naive area-equivalent value.

   **Chip-optimizer wavelength subsampling (2026-05-16, commit 910f2de):** `ChipOptimizationParams` gains `n_wl_optimize` (int, default 0 = full grid, back-compat). KILO's broadband null factory uses 5 wavelengths (`natural_design.py` L611, `n_wl_broadband=5`) while nullsim's detector grid typically has 20+ bins; running the optimizer on the full grid imposes 4× more per-iteration cost and the broadband basin converges to a shallower minimum because the optimizer must satisfy more constraints simultaneously. When `n_wl_optimize > 0`, the torch backend subsamples the input field array at that many uniformly-spaced wavelength indices for the BFGS/restart loop only; `chip.kernel_mzi_mesh` then applies the optimized unitary at every bin of the original grid. `examples/kilo_keck.toml` sets `n_wl_optimize=5` to match KILO's `keck_only_config`, and also bumps `max_iter=15000` / `n_restarts=10` / `restart_patience=4` to allow deeper convergence.

   **Chip-optimizer broadband_refine fast-path + scene wiring pass (2026-05-16, commits 70b5068, a91c6c2, c181979, bba89f8):** three correctness fixes that together push kilo_keck chip_ideal from ~3–9e-4 to 1.7–5e-6, below KILO's reference 7.3e-5. (1) `ChipOptimizationParams` gains `broadband_refine` (bool, default `True` for back-compat). The torch backend deploys its canonicalized params monochromatically (`kernel_rescales_per_wavelength=False`), so the broadband restart loop was minimizing the wrong objective — per-wavelength rescaled-unitary leakages that the kernel never applies — and empirically drifted away from the bootstrap's deep design-wavelength basin. When `False`, the bootstrap warm-start is used directly as the final params; raises `ValueError` if the bootstrap was not run or produced no warm-start. (2) `scene.point_source.Params` gains `star_angular_diameter_mas` (`float | None`, default `None`). Previously the field was parsed by the TOML schema but silently dropped before reaching `state.scene`, so `sensitivity.detection_curve`'s stellar-disk-leakage integral always saw `disk_floor=0` even when `[scene.star].angular_diameter_mas` was set in TOML. `stage_materialize.py` now binds `scene.star.angular_diameter_mas` → `scene.point_source.star_angular_diameter_mas`. (3) `ao.pyramid_wfs.Params` gains `derive_r_mag_from_scene_jmag` (bool, default `False`) and `r_minus_j_offset` (float, default `1.0`). When the flag is set, the WFS noise term uses `r_mag = state.scene.star_jmag + r_minus_j_offset` instead of the static `guide_star_R_mag`, enabling natural-guide-star configs (host star = WFS reference) to sweep AO performance with J-magnitude. `PyramidWFS.consumes` gains `"scene"`. The effective R-mag is recorded in the wavefront component payload. `kilo_keck.toml` sets `derive_r_mag_from_scene_jmag=true`, `bootstrap_n_restarts=30`, and `broadband_refine=false`.

   **Throughput-penalty + AO-piston-scale tuning pass (2026-05-17, commits 1764ae0, 7063adf, 1407e01, dd16ee05, 6f485003):** Three additions to tune chip-optimizer basin selection and realized-null estimation to KILO's regime. (1) `ChipOptimizationParams` gains `throughput_penalty_alpha` (float, default 0.0 = disabled; KILO_BB_THROUGHPUT_PENALTY_ALPHA convention, `kilo/studies/natural_design.py` L590-625), `throughput_penalty_seps_mas` (tuple of floats, default `(3.0,)`), and `throughput_penalty_n_pa` (int, default 8). When `throughput_penalty_alpha > 0`, the bootstrap objective adds a soft penalty `−alpha × <planet_throughput at rep seps>` to bias the optimizer toward basins that preserve the kernel-null's r^6 throughput rolloff; without the penalty the optimizer tends to find ultra-deep-null basins that collapse planet throughput at small separations (chip_ideal 4e-6 but throughput at 30 mas ~0.006 vs KILO's balanced chip 7e-5 / throughput 0.35). `kilo_keck.toml` sets `throughput_penalty_alpha=5e-2` and `throughput_penalty_seps_mas=[3.0, 10.0, 30.0]`. (2) `DetectionCurve.Params` gains `ao_piston_scale` (float, default 1.0 = no scaling). The raw pyramid-WFS piston (Conan 1995 temporal-error formula) can overestimate KILO's tuned-servo value by ~2-3x at Keck J=4; `ao_piston_scale` multiplies `_find_ao_piston_rms_per_sub_m`'s return before it is passed to `realized_null_depth_mc`. `kilo_keck.toml` sets `ao_piston_scale=0.4`. (3) `DetectionCurve.Params` gains `scintillation_index_per_sub` (float, default 0.0 = disabled) and `realized_null_depth_mc` gains a matching `scintillation_index_per_sub` parameter. Per-sub multiplicative amplitude jitter is drawn log-normal with E[I]=1, Var[I]=sigma_I^2 (Roddier 1981); this is the residual after the amplitude servo, NOT open-loop scintillation (KILO transport L1639). `kilo_keck.toml` sets `scintillation_index_per_sub=0.02`.

   **chip_optimization.torch RNG pin (2026-05-17, commit 0e52d8f2):** `ChipOptimizationTorch.apply` now calls `torch.manual_seed(seed_base)` immediately after constructing the numpy master RNG. Without this, PyTorch's internal RNG was seeded from system entropy each run, making restart-init non-reproducible even when `params.seed` was fixed. The pin is cheap (CPU-only) and does not affect multi-thread MKL non-determinism, which is small enough that BFGS basins are stable from identical inits.

   **eps_calibration noise-model port (2026-05-17, commits c9feec43, d968b86e, fbcf59de):** Port of KILO `studies/natural_design.py` L1240 `_effective_systematic_bandwidth_hz` into nullsim's two-stage sensitivity + performance pipeline. (1) `DetectionCurve.Params` gains `effective_systematic_bandwidth_hz_override` (`float | None`, default `None`, gt=0). When set, it is forwarded to the performance stage as `effective_systematic_bandwidth_hz` in the sensitivity component payload, causing `_systematic_variance` to apply `eps² × leak² × t / (2 f_eff)` with `f_eff = override` instead of the performance stage's own `default_servo_bw_hz`. (2) `StandardPerformance.apply` reads `sens_payload["effective_systematic_bandwidth_hz"]` and, when present and non-None, overrides `servo_bw_hz` before calling `_compute_snr` and `_contrast_curve`. `_resolve_servo_bw` is unchanged; its output is now superseded by the sensitivity stage's f_eff when provided.

   **Tier 2a — Fringe-tracking formula fixes + scintillation/f_eff physics ports (2026-05-17):** Three closed_loop.py bugs and two missing KILO physics modules that block Keck-only reproduction. (1) `_conan_servo_lag_variance_rad2` no longer multiplies by the upstream atmospheric piston variance — KILO `fringe_tracking_phase_noise` (`kilo/fringe_tracking.py`:180-182) uses the standalone `(f_piston / f_servo)^(5/3)` form, since the bandwidth-ratio factor already carries the absolute lag variance; the pre-fix `× σ²_atmos` multiplier inflated lag by 2-3 orders of magnitude at J-band Keck. (2) `_shao_colavita_shot_variance_rad2` now matches KILO `shot_noise_phase_variance_per_frame` (`kilo/fringe_tracking.py`:67-95) exactly — `σ²_per_channel = (N_s + N_d) / (V² × N_s²)`, with cross-channel averaging by `n_spectral_channels` applied in `apply()` — replacing the pre-fix `1 / (V² × n_spectral_eff × N_s × 2π²)` (off by ~158× from KILO). New `Params.dark_rate_hz_per_channel` carries the FT-readout dark term. (3) New `nullsim/physics/scintillation.py` and `nullsim/physics/systematic_bandwidth.py` cover the scintillation and effective-bandwidth pieces. Three existing FT scaling-test tolerances loosened modestly to absorb the small residual shot contribution now that lag is no longer artificially amplified.

   **Tier 2b — Sensitivity-stage wiring for σ_I + f_eff + AO/FT quadrature (2026-05-17):** Pipeline-side wiring for the Tier 2a physics ports. (1) `DetectionCurve.Params` gains `auto_amplitude_servo_residual: bool = False`; when True σ_I per sub-aperture is derived from `nullsim/physics/scintillation.py:amplitude_servo_residual_sigma_i` (KILO `atmosphere.py:311-481` port) instead of the manual `scintillation_index_per_sub` scalar. Companion KILO Maunakea-default params (`auto_scintillation_anchor_h5m_zenith=0.041`, `auto_scintillation_height_km=10.0`, `auto_high_altitude_wind_ms=35.0`, `auto_amp_servo_bandwidth_hz=5000.0`, `auto_amp_predictive_gain_variance=20.0`, `auto_amp_servo_photon_rate_per_sub_hz` required when auto-mode is on). The payload's new `amplitude_servo_residual` dict carries the full `AmplitudeServoResidual` breakdown (open-loop, temporal-lag, photon-floor). (2) `DetectionCurve.Params` gains `auto_f_piston_hz: float = 100.0` (AO piston servo bandwidth used by the variance-weighted f_eff blend). `apply()` resolves f_eff with precedence: explicit `effective_systematic_bandwidth_hz_override` → variance-weighted blend (`nullsim/physics/systematic_bandwidth.py:effective_systematic_bandwidth_hz` — KILO `studies/natural_design.py:180-238` port) when auto-mode is on → None (performance stage default). (3) `_find_ao_piston_rms_per_sub_m` now combines the AO `piston_opd_nm.total` and the `fringe_tracking.closed_loop_residual` OPD in quadrature (KILO model: FT corrects only common-mode inter-telescope piston; AO-differential ⊕ FT-residual feeds the chip null). Falls back to AO-only or FT-only when one stage is absent. Full per-telescope `(M,)` array support (KILO `compute_servo_corrected_piston`) deferred until pyramid_wfs emits per-tel piston OPD. `kilo_keck.toml` retains the manual override stack for now (`auto_amplitude_servo_residual=False`); flipping it to the new auto-mode is a Tier 4 config decision once the per-sub photon rate plumbing is in. Two new regression tests pin the auto-mode payload against direct physics-function calls and the clear error when auto-mode is requested without a photon rate.

   **Tier 3 — Chip fidelity (BFGS + throughput penalty + Fresnel loss; 2026-05-17):** Three of four chip-side fixes. (1) `nullsim/stages/chip_optimization/torch.py:432,472` both bootstrap and broadband BFGS calls switched from scipy `L-BFGS-B` (low-memory) to full-Hessian `BFGS` to match KILO `chip.py:1180,1212`. On a 576-dim multi-modal kernel-null landscape the two land in different basins from identical inits. (2) The throughput penalty `−α × <planet_throughput>_{sep, PA, wl}` now runs inside the broadband objective at every wavelength bin (KILO `KILO_BB_THROUGHPUT_PENALTY_ALPHA` convention in `kilo/studies/natural_design.py:606`); pre-fix it lived only in the single-λ bootstrap, so `broadband_refine` drifted toward narrow-band null-only basins. (3) `nullsim/physics/ruilier_cassaing.py` gains `FRESNEL_REFLECTION_LOSS = 0.035` constant and `apply_fresnel_loss: bool = False` kwarg on `coupling_efficiency`; KILO `injection.py:25,72` applies a 3.5% uncoated-tip Fresnel loss per coupled field. The exact annular-overlap `geometric_coupling_limit` is kept because it is the more accurate physical formula.

   **Tier 4 — Config mismatches + realized-null MC physics (2026-05-17):** Two config slices and two MC-physics ports. (1) `examples/kilo_keck.toml`: `[observation].integration_time_s` and `[detector.mkid].integration_time_s` bumped 3600.0 → 14400.0 to match KILO paperv3 headline 4 h on-source. (2) `mkid_max_count_rate_hz` 5.0e15 → 5.0e4 (KILO physical 50 kHz; previous value was a unit-test override that disabled the cap); `n_pixels_dark` 100 → 360 = `n_dark_ports × n_wavelength_bins` so each (port, bin) MKID is its own pixel (KILO per-bin readout convention). (3) `dac_bits` 20 → 16 to match KILO's paperv3 chip-diagnostics figure; 20 bits gave ~100× tighter phase quantization than KILO. (4) `nullsim/physics/realized_null.py:realized_null_depth_mc` gains `fiber_phase_noise_rms_rad: float = 0.0` — IID per-mode-per-realization fiber phase noise. (5) The MC now draws independent AO-piston OPD perturbations per wavelength bin. Three new realized-null regression tests pin the fiber-noise lift, input validation, and per-λ-then-averaged CV.

   **Tier 3.1 — Chromatic chip pass-through, option 2 (2026-05-17):** Closes the chromaticity gap deferred from Tier 3. The torch and MLX backends now publish the chromatic `U_stack` they actually evaluated (KILO column-alternating layout + per-wavelength phase rescaling, post-DAC-quantization) in a new `optimal_u_stack` field on the chip_optimization payload. `chip.kernel_mzi_mesh` checks for `optimal_u_stack` first; when present, the chip stage uses it verbatim. The chip is now genuinely broadband — its deployed unitary varies across wavelength the way a real physical chip's does — and matches the optimizer at every detector bin.

   **CLI results/latest symlink (2026-05-17, commit 8278022):** `_handle_run` now calls `_update_latest_symlink(output_dir)` after writing all outputs. `_update_latest_symlink` creates a relative symlink `results/<name>/latest → <run-dir>` so users have a stable alias regardless of the output directory naming scheme (timestamp, run_hash, or custom template). The symlink is created as a relative path so the parent `results/` directory can be moved without breaking it. If a non-symlink file or directory already occupies `latest`, it is left untouched. All errors are swallowed with `except OSError: pass`; the run's artifacts are already on disk and the alias is cosmetic.

   **Six-bug correctness pass (2026-05-17, commit 0e54bf7):** External code-review surfaced six bugs across four subsystems, all fixed. [HIGH] `chip_optimization._quantize_phases` did not wrap the rounded result to `[0, 2π)`, causing off-design-wavelength unitary mismatches between the optimizer and the deployed chip; fixed with `np.mod(quantized, 2π)` shared by all backends. [MEDIUM] `detector.snspd._resolve_collecting_area_m2` skipped `state.geometry.effective_collecting_area_m2`, overcounting photon rates by ~2.24× in sub-aperture pipelines. [MEDIUM] `physics.realized_null.realized_null_depth_mc` silently accepted negative `n_realizations_per_azimuth`; now raises `ValueError`. [MEDIUM] `sensitivity.detection_curve._chip_input_stellar_rate_hz` used `deltas=[wl[0] * 0.0]` for single-wavelength grids, zeroing the chip-input rate. [LOW] `ao.pyramid_wfs._resolve_pupil_geometry` returned `None` for `pupil_template="circular"` when a central obstruction was present. [LOW] `cli._resolve_output_dir` surfaced format-string errors as raw `AttributeError`. Parity test extractors corrected for `strehl_jband`, `mean_coupling_jband`, `resolution_mas_jband`, and `null_depth_calibrated`.

   **Chip-optimizer bakeoff script + kilo_keck torch.compile (2026-05-17, commit c1a8780):** `scripts/bench_chip_optimizer.py` ships as a two-tier benchmark comparing available chip-optimizer backends on the same problem with the same RNG seed. Tier 1 times a single objective+gradient evaluation; Tier 2 times the full `ChipOptimization*.apply()`. On M-series Apple Silicon at M=24/n_wl=5: numpy/scipy FD 2.96 s/grad, torch eager CPU 26 ms/grad, `torch.compile` CPU 1.6 ms/grad. `examples/kilo_keck.toml` flips `compile_objective` to `true`: end-to-end time drops from ~15 min to ~1 min per cell. Default in `ChipOptimizationParams` stays `false` so small-problem configs are unaffected.

   **Torch-path wrap fix + KILO-convention null scalars (2026-05-17/18, commits d31d64b, 167d5e3, 3fb003b, c137256):** Four targeted fixes. (1) `_quantize_phases` now only rounds to the DAC grid; the `[0, 2π)` wrap moves to `_build_unitary_stack_numpy` alongside the per-wavelength rescaling, keeping the torch path's operating point consistent with its autograd loop. (2) `_require_field` collapses `state.field.amplitudes` to the wavelength-mean per-mode magnitude and broadcasts as real-valued, matching KILO `injection.py:stellar_input_field`. (3) Two new fields added to `RealizedNullResult` — `ideal_null_depth_kilo` (chip-only, mean(dark)/mean(bright)) and `mean_null_depth_kilo` (AO-realized MC mean) — surfaced in sensitivity payloads as `ideal_chip_null_depth_kilo` and `realized_null_depth_kilo`. The existing dark-power-fraction fields stay as the contrast pipeline input. (4) `ClosedLoopFringeTracker.Params.predictive_lag_reduction` default changed from 1.0 to 0.05, matching KILO production convention.

   **Timestamp output dirs + chip_diagnostics fix + kilo_keck fiber noise correction (2026-05-18, commits ec8d4f3, ce1d4a2, 0c5a1d1):** Three targeted fixes. (1) `OutputConfig.dir` default changed to `"results/{run.name}/{timestamp}"`; `ResolvedConfig` gains a `timestamp` field (wall-clock, `YYYYMMDD-HHMMSS`, computed once at `resolve_config` so CLI and runner resolve to the same path). `{run_hash}` remains supported. (2) `chip_diagnostics._plot_broadband_null` constructs a clean nominal stellar field (design-wavelength magnitudes, phases stripped) matching `chip_optimization._require_field`, replacing the prior post-chip-output field read that showed AO/transport-phase-contaminated realized null. Panel title updated to "Broadband null spectrum (chip only, clean input)". (3) `examples/kilo_keck.toml` sets `fiber_phase_noise_rms_rad = 0.0`; the prior 0.01 rad was double-counting the fringe-tracker servo residual. Effect: realized null at J=4 drops from 1.80e-4 to 1.32e-4.

   **Spectro-port disjoint partitions, physical planet fields, basin knobs + MLX retirement (2026-05-18, commits 3e77b10, 639085c):** Five basin-aligning changes against KILO's `natural_design.py` reference plus retirement of the MLX backend. (1) `chip_optimization` schema updated to KILO's disjoint `[bright | dark | spectro]` layout; `_null_indices` = dark + spectro (null objective), `_dark_indices` = dark only (throughput-penalty regularizer). (2) `_require_field` picks per-mode magnitude near design wavelength × `sqrt(area_per_mode)`; throughput-penalty term builds full `planet_input_field(...)` samples per optimized wavelength. (3) Basin knobs in `kilo_keck.toml`: `seed = 42`, `throughput_penalty_seps_mas = [3.0]`, `5/18/1` chip_optimization port counts, `5/19` downstream, Keck ENU geodetic coordinates. (4) `nullsim/stages/chip_optimization/mlx.py` deleted; `nullsim[mlx]` pyproject extra and all MLX-aware test/bench code removed.

   **chip_diagnostics: deployed u_stack + disjoint null_idx + comment audit (2026-05-18, commit c6e71fb):** `_plot_broadband_null` now prefers `optimal_u_stack` from the torch backend over rebuilding from `optimal_chip_params`. `null_idx` updated to `arange(n_bright_ports, n_modes)` (19 indices, dark + spectro) from `arange(n_modes - n_dark_ports, n_modes)` (18 indices). Comment audit: `_require_field` docstring, `sensitivity/detection_curve.py` KILO null reference, and `performance/standard.py` contrast-curve ratio updated.

   **Pipeline contracts: per-instance required_consumes + stage fixes (2026-05-18, commit 52c6d1e):** (1) `Stage` gains `required_consumes(self) -> set[str]`; `validate_dependencies` and `ContentCache.key_for` route through `consumed_state_keys()`. `chip.kernel_mzi_mesh` overrides `required_consumes` to include `results` only when `params.params is None`. (2) `--no-cache` skips cache-key computation entirely. (3) `detector.ideal_counter` preserves `state.results`, publishes a `detector.ideal_counter` component via `unique_component_name`, and emits `bright_0` + `dark_0` ports with proper rates and integrated counts. (4) `fringe_tracking.closed_loop._bright_port_photon_rate_hz` matches ports starting with `"bright"` instead of requiring an exact label, fixing ~5× SNR underestimate with the new ideal-counter port layout.

   **CI test selection cleanup (2026-05-21):** frozen external reference-comparison tests were removed; pytest and CI now run the active nullsim test suite without marker filtering.

   **Packaging + correctness sweep (2026-05-18, commits 4827b30–abeb943):** Ten targeted fixes across packaging, sweep correctness, stage contracts, numerical bugs, and CLI robustness. (1) Package data: `pyproject.toml` extended to bundle `data/**/*.csv` and `config/presets/*.toml`; wheel installs now work without falling back to source tree. `[mlx]` extra replaced by `[dev]` (`pytest>=8`) in `pyproject.toml` and `README.md`. (2) `chip.kernel_mzi_mesh._quantize_phases` wraps to `[0, 2π)` even when `dac_bits is None`, matching the scipy optimizer's wrap-then-scale convention. (3) Sweep mirror-overwrite fixed: `materialize_pipeline_stages` accepts `skip_stage_params` so sweep-written params are not clobbered by re-materialization; `_run_hash_with_externals` now folds all per-cell externals into run identity, not just cell-0. (4) SNR docstrings in `characterization.spectro`, `performance.standard`, and `characterization.detection_curve` clarified — no numerical changes. (5) `performance.standard._resolve_servo_bw` reads live FT bandwidth from results; `fringe_tracking.closed_loop` double-`eta` removed; bright-port mask capitalization normalized. (6) Stage contracts corrected: `performance.standard.consumes` removes `photon_rates`; `sensitivity.detection_curve.consumes` adds `throughput`; `injection.ideal` raises explicitly on ambiguous Strehl fallback. (7) Numerical fixes: `realized_null` pooled median/std; `stellar_disk` monotonic-sort validation; `transport.smf28` deterministic zero phase and dispersion-seed raise; `sensitivity.detection_curve._per_azimuth_stellar_fields` mismatch guards. (8) `detection_space` plot loads user-configured catalog snapshot (not always bundled CSV); `_handle_run`/`_handle_validate` raise on zero-cell sweep; `RunCollection` uses materialized cell-0 config. (9) Lazy outputs autoload: plot/table registration deferred to `_autoload_outputs()` on the `run` path; `validate` no longer imports matplotlib. Cache digest invalidates on file mtime changes. (10) `detection_space` legend lower-left; x-axis upper bound from curve data.

The implementation plan is a separate document.

---

## Approval

This design is ready for implementation planning once the open questions in section 14 are decided. Implementation will be specified in a follow-on plan document.
