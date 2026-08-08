# nullsim Code Map

Navigation guide for the `nullsim` package: load-bearing modules, data flow,
and the tests that pin each area. Hand-maintained — if this map contradicts
the source, trust the source and flag the drift. For parameter-level stage
detail, use `nullsim list-stages` and the stage module docstrings.

## Big Picture

`nullsim` is a Python package and CLI for simulating photonic nulling
instruments from a single TOML config, organized around an ordered pipeline:

```text
TOML config
  -> config load / preset merge / sweep expansion
  -> ordered stage validation and execution
  -> typed SimulationState budgets
  -> registered plots and tables
  -> results/<run>/<YYYYMMDD-HHMMSS>/
```

CLI surface (`nullsim ...` or `python -m nullsim`):

- `run <config>`: execute, expand sweeps, write outputs. `--dry-run` prints
  the expanded plan, output names, and dependency graph without executing.
- `validate <config>`: schema, output registry, sweep, and stage dependency
  check, no execution.
- `inspect <config>`: non-executing run plan with hashes, output dir, stage
  consumes/produces, and physics/trust notes.
- `list-stages [--family ...]` / `list-outputs [--kind ...]`: registry
  introspection.

## Repository Map

```text
nullsim/
  __init__.py                 import side effects: stage and output autoload
  __main__.py                 python -m nullsim entry point
  cli.py                      argparse CLI, sweep execution, manifests, outputs

  config/                     TOML schema, loader, hashing, sweep expansion
  pipeline/                   stage protocol, registry, state, runner, cache
  stages/                     built-in stage implementations
  physics/                    pure math / scientific primitives
  outputs/                    registered plot and table writers
  catalogs/                   versioned external catalog loaders
  data/                       packaged reference data and catalog snapshots

examples/                     canonical user-facing TOML configs
tests/                        unit, stage, output, and example tests
scripts/                      maintenance, reference-generation, benchmarks
changelog.d/                  fragment changelog entries

README.md                     user quick start and CLI overview
nullsim.md                    design document and roadmap
CONTRIBUTING.md               development workflow
pyproject.toml                package metadata, deps, pytest/ruff config
```

## Runtime Flow

1. `nullsim/__init__.py`
   - Sets `__version__`; imports `nullsim.stages._autoload` so stage
     registration decorators run.
   - Plot/table registration is deferred: `nullsim.outputs._autoload_outputs()`
     runs on output-writing/listing paths and when validation/inspection needs
     to check requested output names.

2. `nullsim/cli.py`
   - Builds the subcommands; loads configs via `config.load_config`; expands
     sweeps via `config.expand_sweeps`.
   - Validates requested plot/table names before expensive stage execution.
   - Runs each cell through `pipeline.run`; parallel sweep cells use a
     spawn-based `ProcessPoolExecutor`.
   - Writes `config.toml`, `config.resolved.toml`, `manifest.json`,
     `sweep_results.*`, registered outputs, and the opt-in paper bundle when
     `[output].paper = true`.

3. `nullsim/config/loader.py`
   - Parses TOML with `tomllib`; merges presets from `config/presets/`;
     materializes top-level fields into stage params via
     `stage_materialize.py`; validates with Pydantic models from `schema.py`.
   - Computes `config_hash`, initial `run_hash`, and a timestamp.

4. `nullsim/config/sweeps.py`
   - Expands `[[sweep]]` axes into independent `SweepCell` objects;
     re-materializes stage params after each sweep coordinate is applied.

5. `nullsim/pipeline/runner.py`
   - Builds stage instances in the declared TOML order (no topological
     reordering); validates that every required `consumes` key was produced
     by an earlier stage; wraps each stage in content-addressed cache
     lookup/store.
   - Returns `RunResult(state, diagnostics, timings, cache_hit_count,
     config_hash)`.

6. `nullsim/outputs/`
   - Receives a `RunCollection` of `RunRecord`s; writes selected plots/tables
     by registry lookup.

## Config Layer

- `nullsim/config/schema.py` — Pydantic models for the TOML surface.
  Top-level sections: `run`, `grid`, `site`, `telescope`, `scene`,
  `observation`, `instrument`, `pipeline`, `sweep`, `output`, `cache`,
  `catalog`. `[instrument]` is the shared port-topology section
  (`n_modes = n_bright_ports + n_dark_ports + n_spectro_ports` when all set;
  `n_input_modes` counts physical guided modes before any duplicated-input
  fanout). `StageSpec` allows extra keys (each stage owns its own params
  model); `PipelineConfig` rejects duplicate stage IDs.
- `nullsim/config/loader.py` — public `load_config(path)` entry point.
- `nullsim/config/stage_materialize.py` — bridges user-facing sections into
  stage params: `[scene.*]`, merged `[site]`/`[telescope]` presets, and
  `[instrument]` flow into the stages that need them, preserving explicit
  stage-level overrides.
- `nullsim/config/sweeps.py` — sweep params target `scene.*`,
  `observation.*`, or `pipeline.stages.<id>.<field>`; supports `values`,
  `range`, `linspace`, `logspace`, grid axes, and zip axes.
- `nullsim/config/hashes.py` — deterministic TOML serialization;
  `config_hash` from resolved config text, `run_hash` adds package/runtime
  identity, dependency versions, and external dependency hashes.
- `nullsim/config/catalog.py` — validates `[catalog.exoplanets]` and resolves
  relative snapshot paths.
- `nullsim/config/factories.py` — band-sweep dict helpers for cross-band
  studies. (Site sky brightness tables live in
  `stages/background/sky_surface_brightness.py`, not here.)

Tests: `tests/config/test_loader.py`, `test_sweeps.py`, `test_hashes.py`.

## Pipeline Layer

- `nullsim/pipeline/stage.py` — `Stage` protocol, `BaseStage`,
  `ExternalDependency`, `consumed_state_keys()`. Stage classes declare
  `type_name`, `family`, `consumes`, `produces`, optional `cache_key_extras`;
  an inner `Params` Pydantic model is validated once at construction.
- `nullsim/pipeline/state.py` — immutable `SimulationState` with eight
  budgets: `scene`, `geometry`, `field`, `opd_budget`, `throughput`,
  `wavefront`, `photon_rates`, `results`. Each budget supports named
  `components` and deterministic content hashing.
- `nullsim/pipeline/registry.py` — shared singleton registry;
  `@register_stage`, `@register_plot`, `@register_table` are the extension
  mechanism.
- `nullsim/pipeline/runner.py` — deterministic per-instance RNG contexts,
  dotted-path `consumes`/`produces` validation, declared-order execution,
  timings/diagnostics/cache-hit tracking.
- `nullsim/pipeline/cache.py` — content-addressed stage cache. Keys fold in
  validated stage params, consumed state digest, a per-stage-family source
  digest (shared core + the family's import closure; output and CLI code
  excluded), class/instance identity, RNG identity, package version, external
  dependencies, and cache version. Entries are pickles under
  `.nullsim_cache`.
- `nullsim/pipeline/rng.py` — RNG derived from root seed, the stage's own
  spec digest, sweep coordinate, and stage instance ID. Stage-local by
  design: config edits that do not touch a stage's spec cannot change its
  draws.
- `nullsim/pipeline/manifest.py` — writes `manifest.json` (hashes,
  dependency versions, host info, seed, timings, cache stats, external
  dependencies).

Tests: `tests/pipeline/test_cache.py`, `test_registry.py`, `test_rng.py`,
`test_runner_validation.py`, `test_state.py`.

## Built-In Stages

Built-ins are imported by `nullsim/stages/_autoload.py`. Stage type names are
the strings used in TOML under `[[pipeline.stages]]`.

| Type | File | Consumes | Produces | Purpose |
|---|---|---|---|---|
| `scene.point_source` | `stages/scene/point_source.py` | none | `scene` | Source properties and pointing. |
| `telescope.fixed_aperture` | `stages/telescope/fixed_aperture.py` | `scene` | `geometry`, `throughput` | Aperture area, ENU positions, baselines, telescope throughput. |
| `telescope.pupil_modes` | `stages/telescope/pupil_modes.py` | `geometry` | `geometry` | Photonic input mode placement on a single circular pupil (single-primary geometry bridge). |
| `atmosphere.kolmogorov` | `stages/atmosphere/kolmogorov_screens.py` | `geometry` | `opd_budget`, `wavefront`, `throughput` | Seeing, atmospheric transmission, OPD/wavefront budgets. |
| `ao.pyramid_wfs` | `stages/ao/pyramid_wfs.py` | `geometry`, `opd_budget`, `wavefront`, `scene` | `opd_budget`, `wavefront`, `geometry` | KILO-style pyramid WFS residuals and sub-aperture geometry. |
| `wavefront.space_residual` | `stages/wavefront/space_residual.py` | none | `wavefront`, `opd_budget` | Scalar space-telescope WFE/Strehl and jitter residual budget. |
| `injection.ideal` | `stages/injection/ideal.py` | `scene`, `geometry` | `field`, `throughput`, `wavefront` | Seed ideal field amplitudes, optionally using upstream Strehl. |
| `injection.mode_splitter` | `stages/injection/mode_splitter.py` | `field`, `geometry` | `field`, `geometry` | Guided-mode fanout for duplicated-input Clements mesh studies. |
| `injection.single_mode_fiber` | `stages/injection/single_mode_fiber.py` | `field`, `throughput`, `wavefront`, `geometry` | `field`, `throughput` | Ruilier-Cassaing SMF coupling. |
| `delay_lines.geometric` | `stages/delay_lines/geometric.py` | `geometry`, `throughput`, `scene`, `field` | `throughput`, `field` | Delay-line path equalization loss; opt-in chromatic phase diagnostics. |
| `transport.smf28` | `stages/transport/smf28.py` | `field`, `throughput`, `opd_budget` | `field`, `throughput`, `opd_budget` | SMF-28 loss, phase, and dispersion. |
| `transport.pm980` | `stages/transport/smf28.py` | `field`, `throughput`, `opd_budget` | `field`, `throughput`, `opd_budget` | PM980-XP PM fiber (SMF28Transport subclass). |
| `transport.pm780` | `stages/transport/smf28.py` | `field`, `throughput`, `opd_budget` | `field`, `throughput`, `opd_budget` | PM780-HP PM fiber for ~800 nm bands (SMF28Transport subclass). |
| `transport.photometric_pickoff` | `stages/transport/photometric_pickoff.py` | `field`, `throughput` | `field`, `throughput` | Fixed photometric tap before the science mesh for amplitude sensing. |
| `background.nayra_sky` | `stages/background/nayra_sky.py` | `geometry`, `throughput`, `field` | `results` | Per-port sky background rates for single-pupil ELT (NAYRA) pipelines. |
| `background.sky_surface_brightness` | `stages/background/sky_surface_brightness.py` | `geometry`, `throughput`, `field` | `results` | Generic site sky surface brightness background. |
| `background.space_zodi` | `stages/background/space_zodi.py` | `geometry`, `throughput`, `field`, `scene`, `results` | `results` | Space local-zodi and exozodi background rates per dark/spectro port. |
| `polarization.hwp` | `stages/polarization/optics.py` | `field`, `throughput` | `field`, `throughput` | Half-wave plate Jones rotation per port. |
| `polarization.pbs_split` | `stages/polarization/optics.py` | `field`, `throughput` | `field`, `throughput` | Polarizing beam splitter into two orthogonal polarization arms. |
| `polarization.pm_fiber` | `stages/polarization/optics.py` | `field`, `throughput` | `field`, `throughput`, `results` | PM fiber Jones propagation with birefringence and leakage. |
| `instrument.nayra_six_chip` | `stages/instrument/nayra_six_chip.py` | none | `results` | NAYRA Y/J/H × dual-pol six-chip channel manifest. |
| `chip.identity` | `stages/chip/identity.py` | `field`, `throughput` | `field`, `throughput` | Pass-through chip for smoke tests. |
| `chip.kernel_mzi_mesh` | `stages/chip/kernel_mzi_mesh.py` | `field`, `throughput` (+ `results` when `optimization_source` is set) | `field`, `throughput`, `results` | Kernel-nulling Clements MZI mesh; supports per-pol-arm deployment. |
| `chip.quadrature_reference_mesh` | `stages/chip/quadrature_reference_mesh.py` | `field`, `throughput` | `field`, `throughput`, `results` | Ideal 2N-output quadrature-reference mesh (N tracker + N science rows). |
| `chip_optimization.scipy` | `stages/chip_optimization/scipy.py` | `field`, `throughput`, `geometry` (+ `scene` when disk samples enabled) | `results` | Scipy broadband chip optimization with optional finite-disk constraints. |
| `chip_optimization.torch` | `stages/chip_optimization/torch.py` | `field`, `throughput`, `geometry` (+ `scene` when disk samples enabled) | `results` | Torch/autograd chip optimization: batched wavelength solves, disk rows, tracker/amplitude observability regularization. |
| `chip_optimization.ideal_kernel` | `stages/chip_optimization/ideal_kernel.py` | `field`, `throughput`, `geometry` | `results` | Guyon et al. (2013) ideal achromatic kernel-null projector; warm-start source for Clements refinements. |
| `detector.ideal_counter` | `stages/detector/ideal_counter.py` | `field`, `throughput`, `scene`, `geometry` | `photon_rates`, `results` | Simple ideal detector for the vertical slice. |
| `detector.mkid` | `stages/detector/mkid.py` | `field`, `throughput`, `scene`, `geometry` | `photon_rates`, `results`, `throughput` | MKID QE, saturation, count rates, opt-in energy-resolution redistribution. |
| `detector.snspd` | `stages/detector/snspd.py` | `field`, `throughput`, `scene`, `geometry` | `photon_rates`, `results`, `throughput` | SNSPD QE, saturation, dark counts. |
| `fringe_tracking.closed_loop` | `stages/fringe_tracking/closed_loop.py` | `photon_rates`, `opd_budget`, `wavefront`, `geometry` | `opd_budget`, `results` | Closed-loop piston residuals, classical or photon-Kalman; optional von Karman piston and vibration disturbances. |
| `calibration.floor_lookup` | `stages/calibration/floor_lookup.py` | `results`, `scene` | `results` | Calibration systematic floor lookup/subtraction. |
| `sensitivity.detection_curve` | `stages/sensitivity/detection_curve.py` | `results`, `geometry`, `wavefront`, `scene`, `field`, `throughput` | `results` | Planet throughput curve, stellar-disk floor, realized/effective null, leakage rates. |
| `performance.standard` | `stages/performance/standard.py` | `results`, `scene` | `results` | SNR, integration time, and contrast curve. |
| `performance.pdi` | `stages/performance/pdi.py` | `results` | `results` | Polarimetric differential imaging SNR. |
| `performance.hdc` | `stages/performance/hdc.py` | `results`, `scene` | `results` | Detection-mode high dispersion coronagraphy: inverts the matched filter at each `sensitivity.detection_curve` separation anchor for a CCF contrast curve, alongside the broadband curve and their ratio. Needs an upstream `performance.standard`. |
| `characterization.spectro` | `stages/characterization/spectro.py` | `results`, `geometry`, `scene`, `throughput`, `field` | `results` | Known-position spectro-port retuning and SNR (AWG or MKID-native accounting). |
| `characterization.detection_curve` | `stages/characterization/detection_curve.py` | `results`, `geometry`, `scene`, `throughput`, `field` | `results` | Characterization-mode contrast curve anchors (shares Params with `characterization.spectro`). |
| `characterization.hdc_ccf` | `stages/characterization/hdc_ccf.py` | `results`, `scene` | `results` | High dispersion coronagraphy matched-filter cross-correlation on the spectro port (Wang et al. 2017); needs an upstream AWG-mode `characterization.spectro`. Rejects MKID-native spectro payloads (phase 1). |
| `diagnostics.fringe_observability` | `stages/diagnostics/fringe_observability.py` | `field`, `photon_rates`, `results`, `geometry` | `results` | Fringe-tracker piston/amplitude observability audit (SVD/CRLB, acquisition, cophasing handoff). |

Contract notes (subtleties that are easy to get wrong):

- `sensitivity.detection_curve` and `performance.standard` read `opd_budget`
  opportunistically as a `cache_key_extras` entry, not a validated consume.
- `sensitivity.detection_curve` raises if the separation grid does not cover
  the configured stellar radius.
- `performance.standard` recomputes leakage from the sensitivity-published
  effective null (the published leakage-rate keys are debug-report inputs)
  and applies `nsc_systematic_reduction` only to the null-instability
  systematic term, not mean-leakage photon noise.
- `characterization.spectro` and `characterization.detection_curve` (which
  shares its `Params`) honor `NULLSIM_TORCH_COMPILE` for `compile_objective`
  and force serial restarts when compilation is on.
  `characterization.hdc_ccf` runs no optimizer and has neither knob.

Stage-family tests live under `tests/stages/`, with additional integration
coverage in `tests/examples/`.

## Physics Layer

`nullsim/physics/` should remain independent of TOML, `SimulationState`, and
stage execution except where explicitly documented. Derivation notes and
pinning rationale live in `tests/physics/notes/*.md`.

| File | Responsibility |
|---|---|
| `kolmogorov.py` | Kolmogorov turbulence constants, r0 scaling, phase variance, effective wind speed. |
| `ruilier_cassaing.py` | Single-mode fiber coupling and angular coupling factors. |
| `clements.py` | Clements unitary decomposition and reconstruction. |
| `jones.py` | Jones matrices, waveplates, Stokes conversions, PM fiber birefringence/leakage, PBS arms, `DualPolField`. |
| `photometry.py` | Vega zero points and photon-rate conversions. |
| `pupil_geometry.py` | Keck/KILO/E-ELT sub-aperture packing, pupil constraints, seed strategies. |
| `fresnel.py` | Angular-spectrum scalar propagation. |
| `uv_coverage.py` | Baseline enumeration, ENU-to-UV projection, off-zenith sky-plane projection. |
| `stellar_disk.py` | Uniform-disk null-floor integration. |
| `planet_throughput.py` | Off-axis input fields, chip throughput vs separation, dark-port transmission maps. |
| `realized_null.py` | Monte Carlo AO-realized null-depth statistics, including IID + common-mode piston covariance. |
| `quadrature_fringe_tracker.py` | Quadrature-reference mesh primitives and analytic observability/CRLB estimates. |
| `scintillation.py` | Scintillation and amplitude-servo residual model. |
| `fringe_tracking.py` | Piston-servo bandwidth, disturbance terms, photon-Kalman residual budget, AR(1) temporal-Kalman covariance surrogate. |
| `systematic_bandwidth.py` | Effective bandwidth for null-instability noise. |
| `spectroscopy.py` | Spectro-port SNR, max resolving power, AWG spectrograph model. |
| `mkid.py` | MKID resolving power and energy-response matrix primitives. |
| `spectra.py` | Template spectrum loading, constant-R grids, Doppler shift, resolving-power convolution, flux-conserving resampling. |
| `hdc.py` | High dispersion coronagraphy matched-filter CCF budget: channelization, template weights, self-subtraction, RV scan, and the inverse solve `ccf_contrast_at_target_snr` (faintest contrast reaching a target SNR). |
| `pdi.py` | Polarimetric differential imaging SNR primitives. |
| `characterization.py` | Known-position chip optimization for spectro mode. |
| `characterization_torch.py` | Torch autograd-Jacobian helper for characterization optimization. |
| `imaging.py` | Aperture-synthesis image reconstruction: beam patterns, photon-count observations, SNR maps, NNLS. |
| `fringe_observability.py` | Piston/amplitude observability primitives: response matrices, SVD/CRLB, Monte Carlo estimation, Gauss-Newton capture range, acquisition scans, cophasing handoff. |
| `reflected_light.py` | Reflected-light planet/star contrast (albedo × Lambertian phase × (R_p/a)²). |
| `ideal_coronagraph.py` | Guyon et al. (2006) ideal coronagraph bound and photon-limited 5σ contrast curve. |

Tests: `tests/physics/test_*.py`.

## Output Layer

The output layer is registry-driven: configs name plots/tables in `[output]`,
and the CLI resolves names against the shared registry.

- `nullsim/outputs/__init__.py` — `RunRecord` (one sweep cell or single run)
  and `RunCollection` (all records plus output directory).
- `nullsim/outputs/sweep_results.py` — long-format `sweep_results.parquet`
  with scalar results, spectra, contrast curves, and sensitivity/
  characterization scalars.
- `nullsim/outputs/paper.py` — opt-in `[output].paper = true` bundle writer.
  Creates `paper/` beside `plots/` and `tables/`; reuses registered writers
  and adds paper-only schematic PDFs and LaTeX result tables. The full
  emitted set is recorded in `paper_manifest.json`.

Registered plots:

| Name | File | Purpose |
|---|---|---|
| `contrast_curve` | `outputs/plots/__init__.py` | Basic contrast curve plot. |
| `detection_space` | `outputs/plots/detection_space.py` | Exoplanet population overlay with contrast curves; overlays `performance.hdc` CCF curves as dash-dot lines of the matching colour, labelled with each record's J/time/R identity. |
| `subaperture_layout` | `outputs/plots/subaperture_layout.py` | Pupil/sub-aperture layout panels. |
| `uv_coverage_grid` | `outputs/plots/uv_coverage_grid.py` | KILO-style UV coverage grid. |
| `chip_diagnostics` | `outputs/plots/chip_diagnostics.py` | Detection-chip power, matrix, DAC, and spectrum diagnostics. |
| `chip_diagnostics_characterization` | `outputs/plots/chip_diagnostics.py` | Characterization retune diagnostics. |
| `cophasing_unitary` | `outputs/plots/cophasing_unitary.py` | Cophasing-mode ABCD, pair graph, OPD-incidence, and handoff diagnostics. |
| `loss_budget` | `outputs/plots/loss_budget.py` | Throughput/loss breakdown. |
| `contrast_curves_multiband` | `outputs/plots/contrast_curves_multiband.py` | Multi-band/multi-time contrast curves. |
| `calibration_time` | `outputs/plots/calibration_time.py` | Optimizer convergence / calibration-time plot. |
| `debug_report` | `outputs/plots/debug_report.py` | Multi-page pipeline audit report. |
| `imaging_reconstruction` | `outputs/plots/imaging_reconstruction.py` | Truth/SNR/beam imaging artifacts for space nuller reconstruction. |
| `null_transmission_map` | `outputs/plots/null_transmission_map.py` | Summed dark-port transmission map over sky coordinates. |
| `fringe_observability` | `outputs/plots/fringe_observability.py` | Fringe-tracker observability diagnostic panels. |
| `hdc_ccf` | `outputs/plots/hdc_ccf.py` | HDC cross-correlation diagnostics: channelized spectra, template weights, SNR vs resolving power, SNR vs trial radial velocity. |

Registered tables:

| Name | File | Purpose |
|---|---|---|
| `performance_summary` | `outputs/tables/__init__.py` | One row per run with headline scalar metrics. |
| `characterization_summary` | `outputs/tables/__init__.py` | One row per `characterization.spectro` result component. |

Tests: `tests/outputs/test_outputs.py`, `test_end_to_end.py`, `test_paper.py`,
`tests/outputs/plots/test_*.py`.

## Catalogs and Data

- `nullsim/catalogs/exoplanets.py` — validates NASA Exoplanet Archive-style
  CSV snapshots, computes/verifies snapshot hashes, classifies planets with
  versioned rules, derives separation in mas when needed.
- `nullsim/data/exoplanet_catalogs/nasa_exoplanet_archive_paperv3.csv` —
  packaged snapshot used by detection-space outputs.
- `nullsim/data/kilo_reference/*.json` — KILO reference data for plot
  overlays and calibration floors.
- `nullsim/data/nayra_reference/*.json` — NAYRA E-ELT reference data
  (`calibration_time_J_M18.json`, `calibration_time_J_M36.json`).
- `nullsim/data/spectra/*.csv` — HDC template snapshots on constant-R grids
  (R = 30000, 0.5–1.8 um): planet albedos (`earth_albedo`, `venus_albedo`)
  from the NASA Planetary Spectrum Generator, PHOENIX stellar photospheres
  (`stellar_g2v`, `stellar_k2v`, `stellar_m2v`), and ESO SkyCalc telluric
  transmission (`telluric_maunakea`, `telluric_armazones`). Each carries a
  provenance header. `scripts/generate_hdc_spectra.py` regenerates all seven,
  each from its own upstream: the two albedos from the PSG web API driven by
  the frozen configs in `scripts/psg_config/*.cfg`, the photospheres from the
  PHOENIX HiRes FITS archive, the tellurics from the SkyCalc REST service via
  `skycalc_cli`. `characterization.hdc_ccf` declares the ones it reads as
  `ExternalDependency` entries, so their bytes enter the run hash.

Tests: `tests/catalogs/test_exoplanets.py` with
`tests/catalogs/fixtures/sample_catalog.csv`.

## Examples

Reference/instrument configs in the root of `examples/`:

- `examples/kilo_keck.toml` — KILO Keck-only paper-baseline pipeline with
  `[output].paper = true`.
- `examples/kilo_keck_6.toml` — default KILO Keck config: reduced
  6-subaperture-per-pupil quadrature mesh extending `kilo_keck.toml`; source
  of the phasing-paper bundle.
- `examples/nayra_eelt.toml` — single-pupil E-ELT NAYRA at Cerro Armazones;
  N=48 sub-aperture J-primary baseline.
- `examples/hwo_space.toml` — HWO-class space nuller, no atmosphere/AO
  stages.

Regression and smoke-test configs under `examples/test/`:

- `vertical_slice.toml` — fast architecture smoke test with a J-mag sweep.
- `kilo_maunakea.toml` — compact four-telescope Maunakea demonstrator.
- `nayra_eelt_baseline.toml` — fixed-point NAYRA regression reference (no
  sweeps).
- `detection_space_example.toml` — minimal detection-space output exercise.
- `mkid_native_spectro.toml` — MKID-native spectroscopy smoke config.
- `hdc_ccf_smoke.toml` — HDC cross-correlation smoke config: four modes, M2V
  host, AWG spectro port at R = 5000, `characterization.hdc_ccf` + `hdc_ccf`
  plot. Runs in seconds. Photon limited, so the CCF loses to the broadband
  band SNR — correctly, there being no systematic to reject.
- `hdc_ccf_systematics.toml` — the same chain with a quasi-static leak
  instability (`residual_speckle_stability_factor = 1e-3`,
  `with_fringe_tracking = false`), the regime where characterization-mode HDC
  wins. Companion to `hdc_ccf_smoke.toml`; the config header carries the
  budget and the caveats on the asserted correlation time.
- `hdc_detection_smoke.toml` — detection-mode HDC: same systematics knobs on
  `performance.standard`, `performance.hdc` over seven separation anchors, and
  the `detection_space` overlay. `ccf_contrast_gain` 7.5 with the knobs, 0.16
  without.
- `kilo_vs_hwo_study.toml` — cross-config comparison study.
- `kilo_keck_quadrature_mesh.toml` — KILO Keck with a 48-mode
  duplicated-input Clements mesh and cophasing diagnostics.

Example tests live under `tests/examples/` and exercise config loading,
structural invariants, and runnable end-to-end slices.

## Adding a Stage

1. Create a module under the appropriate `nullsim/stages/<family>/` directory.
2. Subclass `BaseStage`.
3. Add class variables:
   - `type_name = "<family>.<name>"`
   - `family = "<family>"`
   - `consumes = frozenset({...})`
   - `produces = frozenset({...})`
4. Add an inner `Params(BaseModel)` if the stage has parameters.
5. Implement `apply(self, state: SimulationState) -> SimulationState`.
6. Return diagnostics from `diagnostics()` if the stage has audit values.
7. Return `ExternalDependency` objects from `external_dependencies()` for data
   files or snapshots that affect results.
8. Decorate the class with `@register_stage`.
9. Add the module path to `_STAGE_MODULES` in `nullsim/stages/_autoload.py`.
10. Add focused tests under `tests/stages/<family>/` and, if relevant, an
    example-level smoke test.

Important contracts:

- Do not mutate existing state budgets in place; return replaced budgets.
- Preserve existing components when adding a component to a budget.
- Declare every required state read in `consumes`.
- Use `cache_key_extras` only for opportunistic reads that should affect cache
  identity but should not be enforced by dependency validation.

## Adding an Output

1. Add a function that accepts `(collection, output_dir, formats)`.
2. Decorate it with `@register_plot` or `@register_table`.
3. Import the module from `outputs/plots/__init__.py` or
   `outputs/tables/__init__.py` so registration runs.
4. Add the output name to an example config if it is user-facing.
5. Add a focused test under `tests/outputs/`.

## Test Map

```text
tests/config/              config loading, hashing, sweeps
tests/pipeline/            registry, runner validation, cache, RNG, state
tests/physics/             pure mathematical primitives
tests/stages/              stage behavior and stage integration
tests/outputs/             output writers and plot modules
tests/catalogs/            exoplanet catalog loader
tests/examples/            end-to-end config checks
```

Default test command:

```bash
conda run -n py313 pytest
```

Target subsets by appending a path: `tests/pipeline`, `tests/stages`,
`tests/examples/test_vertical_slice.py`.

## Common Navigation Tasks

- To understand a config field: start in `config/schema.py`, then inspect the
  stage `Params` model that consumes it.
- To understand why a pipeline fails validation: inspect the failing stage's
  `consumes`, the earlier stages' `produces`, and `pipeline/runner.py`.
- To understand stale or surprising results: inspect `pipeline/cache.py`,
  `external_dependencies()`, and the run's `manifest.json`.
- To understand a plotted quantity: inspect the output module first, then trace
  the `state.results.components` or budget field it reads.
- To add a physics formula: put it in `physics/`, test it in `tests/physics/`,
  then wrap it with a stage only after the primitive is pinned.
- To debug an end-to-end config: start with the relevant `examples/*.toml`, then
  inspect the corresponding structure or smoke test under `tests/examples/`.
