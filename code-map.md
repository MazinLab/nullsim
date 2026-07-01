# nullsim Code Map

This file is a navigation guide for the `nullsim` package. It points to the
load-bearing modules, the data flow between them, and the tests that pin each
area.

## Big Picture

`nullsim` is a Python package and CLI for simulating photonic nulling
instruments from a single TOML config. The package is organized around an
ordered pipeline:

```text
TOML config
  -> config load / preset merge / sweep expansion
  -> ordered stage validation and execution
  -> typed SimulationState budgets
  -> registered plots and tables
  -> results/<run>/<YYYYMMDD-HHMMSS>/
```

The public command surface is:

- `nullsim run <config>`: execute a config, expand sweeps, write outputs.
- `nullsim run --dry-run <config>`: validate the expanded run plan, output
  names, and dependency graph without creating outputs.
- `nullsim validate <config>`: schema, output registry, sweep, and stage
  dependency check, no execution.
- `nullsim inspect <config>`: non-executing run plan with hashes, output dir,
  stage consumes/produces, and physics/trust notes.
- `nullsim list-stages [--family ...]`: inspect registered stage types.
- `nullsim list-outputs [--kind ...]`: inspect registered plot/table outputs.
- `python -m nullsim`: same CLI entry point.

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
   - Sets `__version__`.
   - Imports `nullsim.stages._autoload` so stage registration decorators run.
   - Imports `nullsim.outputs` for `RunCollection`/`RunRecord` dataclasses.
     Plot/table registration is deferred: `nullsim.outputs._autoload_outputs()`
     is called by output-writing/listing paths and by validation/inspection
     only when requested output names need to be checked.

2. `nullsim/cli.py`
   - Builds the `run`, `validate`, `inspect`, `list-stages`, and
     `list-outputs` subcommands.
   - Loads configs through `config.load_config`.
   - Expands sweep cells with `config.expand_sweeps`.
   - Validates requested plot/table names before expensive stage execution.
   - `run --dry-run` and `inspect` print a non-executing plan with run hashes,
     output location, stage consumes/produces, and trust notes.
   - Runs each cell through `pipeline.run`.
   - Uses a spawn-based `ProcessPoolExecutor` for parallel sweep cells.
   - Writes `config.toml`, `config.resolved.toml`, `manifest.json`,
     `sweep_results.*`, registered outputs, and the opt-in paper bundle when
     `[output].paper = true`.

3. `nullsim/config/loader.py`
   - Parses TOML with `tomllib`.
   - Merges supported presets from `config/presets/`.
   - Materializes top-level config fields into stage params through
     `stage_materialize.py`.
   - Validates with Pydantic models from `schema.py`.
   - Computes `config_hash`, initial `run_hash`, and a timestamp.

4. `nullsim/config/sweeps.py`
   - Expands `[[sweep]]` axes into independent `SweepCell` objects.
   - Supports explicit `values`, numeric `range`, `linspace`, `logspace`.
   - Supports grid axes and zip axes.
   - Re-materializes stage params after each sweep coordinate is applied.

5. `nullsim/pipeline/runner.py`
   - Builds stage instances from `pipeline.stages` in the declared TOML order.
   - Validates that every stage's required `consumes` keys were produced by an
     earlier stage.
   - Runs stages in order. There is no topological reordering.
   - Wraps each stage in content-addressed cache lookup/store.
   - Returns `RunResult(state, diagnostics, timings, cache_hit_count,
     config_hash)`.

6. `nullsim/outputs/`
   - Receives a `RunCollection` of one or more `RunRecord`s.
   - Writes selected plots/tables by looking up names in the shared registry.

## Config Layer

Key files:

- `nullsim/config/schema.py`
  - Pydantic models for the TOML surface.
  - Top-level sections: `run`, `grid`, `site`, `telescope`, `scene`,
    `observation`, `instrument`, `pipeline`, `sweep`, `output`, `cache`,
    `catalog`.
  - `[instrument]` is the shared port-topology section. It validates
    `n_modes = n_bright_ports + n_dark_ports + n_spectro_ports` when all are
    set and carries `n_input_modes` for physical guided-mode counts before any
    duplicated-input chip fanout.
  - `StageSpec` allows extra keys because each stage owns its own parameter
    model.
  - `PipelineConfig` rejects duplicate stage IDs.

- `nullsim/config/loader.py`
  - Public `load_config(path)` entry point.
  - Applies presets and stage materialization before producing
    `ResolvedConfig`.

- `nullsim/config/stage_materialize.py`
  - Bridges user-facing sections into stage params.
  - Stages pull shared physical defaults from top-level sections:
    `scene.point_source` from `[scene.*]`, telescope / atmosphere / AO /
    injection / background / fringe-tracking stages from merged `[site]`
    and `[telescope]` presets, and port-bearing stages from `[instrument]`.
    Instrument values materialize into chip, chip-optimization, detector,
    sensitivity, characterization, `injection.ideal`, and
    `fringe_tracking.closed_loop` params while preserving explicit stage-level
    overrides.

- `nullsim/config/sweeps.py`
  - Turns one `ResolvedConfig` into executable sweep cells.
  - Sweep params can target `scene.*`, `observation.*`, or
    `pipeline.stages.<id>.<field>`.

- `nullsim/config/hashes.py`
  - Deterministic TOML serialization.
  - Computes `config_hash` from resolved config text.
  - Computes `run_hash` from config identity, package/runtime identity,
    dependency versions, and external dependency hashes.

- `nullsim/config/catalog.py`
  - Validates the `[catalog.exoplanets]` TOML block and resolves relative
    snapshot paths.

- `nullsim/config/factories.py`
  - Band-sweep dict helpers for cross-band study configs. (Site sky
    brightness tables live in
    `stages/background/sky_surface_brightness.py`, not here.)

Tests:

- `tests/config/test_loader.py`
- `tests/config/test_sweeps.py`
- `tests/config/test_hashes.py`

## Pipeline Layer

Key files:

- `nullsim/pipeline/stage.py`
  - Defines the `Stage` protocol, `BaseStage`, `ExternalDependency`, and
    `consumed_state_keys()`.
  - Stage classes declare `type_name`, `family`, `consumes`, `produces`, and
    optional `cache_key_extras`.
  - If a stage defines an inner `Params` Pydantic model, `BaseStage` validates
    params once at construction time.

- `nullsim/pipeline/state.py`
  - Defines the immutable `SimulationState`.
  - Defines the eight state budgets:
    `scene`, `geometry`, `field`, `opd_budget`, `throughput`, `wavefront`,
    `photon_rates`, `results`.
  - Each sub-budget supports named `components` and deterministic content
    hashing.

- `nullsim/pipeline/registry.py`
  - Shared singleton registry for stages, plots, and tables.
  - `@register_stage`, `@register_plot`, and `@register_table` are the
    extension mechanism used by built-ins.
  - Validates stage-specific params via `stage_cls.Params` when present.

- `nullsim/pipeline/runner.py`
  - Instantiates stages with deterministic per-instance RNG contexts.
  - Validates dotted-path `consumes`/`produces` contracts.
  - Executes the declared stage order.
  - Tracks timings, diagnostics, and cache hit count.

- `nullsim/pipeline/cache.py`
  - Content-addressed stage cache.
  - Cache keys fold in validated stage params, consumed state digest,
    a per-stage-family source digest (shared core + the stage family's
    import closure; output and CLI code excluded), class and instance
    identity, RNG identity,
    package version, external dependencies, and cache version.
  - Cache entries are pickle files under `.nullsim_cache` by default.

- `nullsim/pipeline/rng.py`
  - Deterministic RNG derivation from root seed, the stage's own spec digest,
    sweep coordinate, and stage instance ID. Stage-local by design: config
    edits that do not touch a stage's spec cannot change its draws (review
    2026-06-10 Issue 7).

- `nullsim/pipeline/manifest.py`
  - Writes `manifest.json` with config/run hashes, dependency versions, host
    info, RNG seed, timings, cache stats, and external dependencies.

Tests:

- `tests/pipeline/test_cache.py`
- `tests/pipeline/test_registry.py`
- `tests/pipeline/test_rng.py`
- `tests/pipeline/test_runner_validation.py`
- `tests/pipeline/test_state.py`

## Built-In Stages

Built-ins are imported by `nullsim/stages/_autoload.py`. Stage type names are
the strings used in TOML under `[[pipeline.stages]]`.

| Type | File | Consumes | Produces | Purpose |
|---|---|---|---|---|
| `scene.point_source` | `stages/scene/point_source.py` | none | `scene` | Populate source properties and pointing. |
| `telescope.fixed_aperture` | `stages/telescope/fixed_aperture.py` | `scene` | `geometry`, `throughput` | Aperture area, ENU positions, baselines, telescope throughput. |
| `telescope.pupil_modes` | `stages/telescope/pupil_modes.py` | `geometry` | `geometry` | Place photonic input modes across a single circular pupil; geometry bridge for single-primary configs (HWO) that skip AO but need per-mode positions for planet throughput. |
| `atmosphere.kolmogorov` | `stages/atmosphere/kolmogorov_screens.py` | `geometry` | `opd_budget`, `wavefront`, `throughput` | Seeing, atmospheric transmission, OPD/wavefront budgets. |
| `ao.pyramid_wfs` | `stages/ao/pyramid_wfs.py` | `geometry`, `opd_budget`, `wavefront`, `scene` | `opd_budget`, `wavefront`, `geometry` | KILO-style pyramid WFS residuals and sub-aperture geometry. |
| `wavefront.space_residual` | `stages/wavefront/space_residual.py` | none | `wavefront`, `opd_budget` | Scalar space-telescope WFE/Strehl, amplitude-jitter, and pointing-jitter residual budget. |
| `injection.ideal` | `stages/injection/ideal.py` | `scene`, `geometry` | `field`, `throughput`, `wavefront` | Seed ideal field amplitudes, optionally using upstream Strehl. |
| `injection.mode_splitter` | `stages/injection/mode_splitter.py` | `field`, `geometry` | `field`, `geometry` | Area-normalized guided-mode fanout for duplicated-input Clements mesh architecture studies; preserves physical pupil positions while expanding the chip input mode count. |
| `injection.single_mode_fiber` | `stages/injection/single_mode_fiber.py` | `field`, `throughput`, `wavefront`, `geometry` | `field`, `throughput` | Ruilier-Cassaing SMF coupling. |
| `delay_lines.geometric` | `stages/delay_lines/geometric.py` | `geometry`, `throughput`, `scene`, `field` | `throughput`, `field` | Geometric delay-line path equalization loss, with opt-in fiber-medium chromatic phase diagnostics. |
| `transport.smf28` | `stages/transport/smf28.py` | `field`, `throughput`, `opd_budget` | `field`, `throughput`, `opd_budget` | SMF-28 loss, phase, and dispersion. |
| `transport.pm980` | `stages/transport/smf28.py` | `field`, `throughput`, `opd_budget` | `field`, `throughput`, `opd_budget` | PM980-XP polarization-maintaining fiber; subclass of SMF28Transport with PM980-XP attenuation defaults. |
| `transport.pm780` | `stages/transport/smf28.py` | `field`, `throughput`, `opd_budget` | `field`, `throughput`, `opd_budget` | PM780-HP polarization-maintaining fiber for ~800 nm bands; subclass of SMF28Transport with PM780-HP attenuation defaults. |
| `transport.photometric_pickoff` | `stages/transport/photometric_pickoff.py` | `field`, `throughput` | `field`, `throughput` | Fixed scalar photometric tap before the science mesh: routes `tap_fraction` of each guided mode to a photometric monitor for amplitude sensing and scales the surviving field/throughput. |
| `background.nayra_sky` | `stages/background/nayra_sky.py` | `geometry`, `throughput`, `field` | `results` | Per-port sky-background photon rates for single-pupil ELT (NAYRA) pipelines. |
| `background.sky_surface_brightness` | `stages/background/sky_surface_brightness.py` | `geometry`, `throughput`, `field` | `results` | Generic sky surface brightness background; uses site sky spectrum from config factories. |
| `background.space_zodi` | `stages/background/space_zodi.py` | `geometry`, `throughput`, `field`, `scene`, `results` | `results` | Space local-zodiacal and exozodiacal background rates per physical dark/spectro port. |
| `polarization.hwp` | `stages/polarization/optics.py` | `field`, `throughput` | `field`, `throughput` | Half-wave plate Jones rotation applied per-port. |
| `polarization.pbs_split` | `stages/polarization/optics.py` | `field`, `throughput` | `field`, `throughput` | Polarizing beam splitter: splits into two orthogonal linear polarization arms. |
| `polarization.pm_fiber` | `stages/polarization/optics.py` | `field`, `throughput` | `field`, `throughput`, `results` | PM fiber Jones propagation with birefringence and leakage; records polarization diagnostics. |
| `instrument.nayra_six_chip` | `stages/instrument/nayra_six_chip.py` | none | `results` | Records NAYRA Y/J/H × dual-pol six-chip channel manifest into results. |
| `chip.identity` | `stages/chip/identity.py` | `field`, `throughput` | `field`, `throughput` | Pass-through chip for smoke tests. |
| `chip.kernel_mzi_mesh` | `stages/chip/kernel_mzi_mesh.py` | `field`, `throughput` (+ `results` when `optimization_source` is set) | `field`, `throughput`, `results` | Kernel-nulling Clements MZI mesh; supports `deployment_mode="per_pol_arm"` for NAYRA dual-PBS-arm front end (independent chip per arm with diagonal phase correction). |
| `chip.quadrature_reference_mesh` | `stages/chip/quadrature_reference_mesh.py` | `field`, `throughput` | `field`, `throughput`, `results` | Ideal 2N-output quadrature-reference architecture test: N tracker rows plus N science rows, published as a rectangular effective `chip.kernel_mzi_mesh` stack for downstream diagnostics. |
| `chip_optimization.scipy` | `stages/chip_optimization/scipy.py` | `field`, `throughput`, `geometry` (+ `scene` when disk samples enabled) | `results` | Scipy broadband chip optimization; supports `disk_n_rings`/`disk_n_azimuths`/`disk_weight` for finite stellar-disk null constraints. |
| `chip_optimization.torch` | `stages/chip_optimization/torch.py` | `field`, `throughput`, `geometry` (+ `scene` when disk samples enabled) | `results` | Torch/autograd chip optimization backend; batches wavelength solves across the MZI mesh for speed; supports finite-disk rows, bright-port piston tracker regularization through singular-value floor/count-free CRLB/tracker-rank-priority restart selection, passive amplitude-mismatch suppression, selected-port amplitude-observability regularization, and a joint phase+amplitude Fisher proxy for mixed piston/scintillation estimability. |
| `chip_optimization.ideal_kernel` | `stages/chip_optimization/ideal_kernel.py` | `field`, `throughput`, `geometry` | `results` | Ideal achromatic kernel-null projector; builds the Guyon et al. (2013) PCA/SVD stellar-mode upper bound and publishes the chip unitary without iterative optimization. Warm-start source for physical Clements refinements. |
| `detector.ideal_counter` | `stages/detector/ideal_counter.py` | `field`, `throughput`, `scene`, `geometry` | `photon_rates`, `results` | Simple ideal detector for vertical slice. |
| `detector.mkid` | `stages/detector/mkid.py` | `field`, `throughput`, `scene`, `geometry` | `photon_rates`, `results`, `throughput` | MKID QE, saturation, count rates, and opt-in finite energy-resolution redistribution. |
| `detector.snspd` | `stages/detector/snspd.py` | `field`, `throughput`, `scene`, `geometry` | `photon_rates`, `results`, `throughput` | SNSPD QE, saturation, dark counts; folds detector QE into throughput budget. |
| `fringe_tracking.closed_loop` | `stages/fringe_tracking/closed_loop.py` | `photon_rates`, `opd_budget`, `wavefront`, `geometry` | `opd_budget`, `results` | Closed-loop piston residuals with classical or `control_model="photon_kalman"` operation. The stage supersedes replaced AO piston components, separates detector sample rate `f_sample_hz` from equivalent closed-loop `f_3db_hz`, can add inter-telescope von Karman piston and vibration disturbances, and publishes IID/common residual splits for realized-null covariance propagation. |
| `calibration.floor_lookup` | `stages/calibration/floor_lookup.py` | `results`, `scene` | `results` | Calibration systematic floor lookup/subtraction. |
| `sensitivity.detection_curve` | `stages/sensitivity/detection_curve.py` | `results`, `geometry`, `wavefront`, `scene`, `field`, `throughput` | `results` | Planet throughput curve, finite stellar-disk floor, realized null, optional closed-loop/open-loop scintillation injection, setpoint-trim and dither leakage, and optional fringe-observability piston covariance propagation through either scalar predictive gain or the temporal-Kalman model. `opd_budget` is a cache-key extra for optional OPD-budget piston reads, not a dependency-validation consume. Publishes `effective_null_by_separation`, `leakage_rate_hz_effective`, `leakage_rate_hz_static_chip`, and `effective_to_static_leakage_ratio`; raises if the separation grid does not cover the configured stellar radius. |
| `performance.standard` | `stages/performance/standard.py` | `results`, `scene` | `results` | SNR, integration time, and contrast curve. Recomputes leakage from the sensitivity-published effective null (the published leakage-rate keys are debug-report inputs), reads the FT-published `servo_bandwidth_hz` equivalent to `f_3db_hz`, keeps `opd_budget` as a cache-key extra for the legacy residual audit, and applies `nsc_systematic_reduction` only to the null-instability systematic term, not mean-leakage photon noise. |
| `performance.pdi` | `stages/performance/pdi.py` | `results` | `results` | Polarimetric differential imaging SNR. |
| `characterization.spectro` | `stages/characterization/spectro.py` | `results`, `geometry`, `scene`, `throughput`, `field` | `results` | Known-position spectro-port retuning and SNR; supports AWG or MKID-native spectrograph accounting plus optional bright-port piston-observability regularization for characterization units that must also fringe-track. Params include `compile_objective` (also honored via `NULLSIM_TORCH_COMPILE`; forces serial restarts when on), `residual_speckle_stability_factor`, `nsc_systematic_reduction`, and `null_floor`. The stage emits `eps_systematic`, `nsc_systematic_reduction`, and `null_floor`; the old hand-tuned calibration-epsilon knob is gone. |
| `characterization.detection_curve` | `stages/characterization/detection_curve.py` | `results`, `geometry`, `scene`, `throughput`, `field` | `results` | Characterization-mode contrast curve anchors using the same optional bright-port piston-observability regularization as `characterization.spectro`. Shares `compile_objective`, `residual_speckle_stability_factor`, `nsc_systematic_reduction`, and `null_floor` Params, and emits `null_floor`, `eps_systematic`, and `nsc_systematic_reduction`; the old hand-tuned calibration-epsilon knob is gone. |
| `diagnostics.fringe_observability` | `stages/diagnostics/fringe_observability.py` | `field`, `photon_rates`, `results`, `geometry` | `results` | Fringe-tracker piston and amplitude observability audit: bright-port piston intensity response, selectable bright/bright+dark/all-port amplitude response, SVD/Fisher observability spectrum, differential fractional-intensity CRLB, scintillation-assumption and amplitude post-processing checks, spectral-diversity sweep, linear/Gauss-Newton residuals, active coordinate-scan acquisition, and dedicated cophasing-unitary handoff Monte Carlo diagnostics. |

Stage-family tests live under `tests/stages/`, with additional integration
coverage in `tests/examples/`.

## Physics Layer

`nullsim/physics/` should remain independent of TOML, `SimulationState`, and
stage execution except where explicitly documented.

| File | Responsibility |
|---|---|
| `kolmogorov.py` | Kolmogorov turbulence constants, r0 scaling, phase variance, and `effective_wind_speed_m_s` from Greenwood/Fried `tau0 = 0.314 r0 / v_eff`. |
| `ruilier_cassaing.py` | Single-mode fiber coupling and angular coupling factors. |
| `clements.py` | Clements unitary decomposition and reconstruction. |
| `jones.py` | Jones matrices, waveplates, Stokes conversions; PM fiber birefringence and leakage (`jones_pm_fiber`, `pm_fiber_leakage_power`); polarizing beam splitter arm matrices (`polarizing_beam_splitter_arm`); `DualPolField` container. |
| `photometry.py` | Vega zero points and photon-rate conversions. |
| `pupil_geometry.py` | Keck/KILO/E-ELT sub-aperture packing and pupil constraints; `keck_pupil()`, `eelt_pupil()`, concentric-ring and wedge-hex seed strategies. |
| `fresnel.py` | Angular-spectrum scalar propagation. |
| `uv_coverage.py` | Baseline enumeration, ENU-to-UV projection, and `project_positions_to_sky_plane_m` for off-zenith aperture-coordinate projection before sky-angle phase rows. |
| `stellar_disk.py` | Uniform-disk null-floor integration: area-weighted incoherent intensity integral over a finite stellar disk. |
| `planet_throughput.py` | Off-axis coherent input fields, chip throughput vs separation, and 2-D summed dark-port transmission maps. |
| `realized_null.py` | Monte Carlo AO-realized null-depth statistics, including `piston_covariance_with_common_mode_m2` for an IID per-mode piston term plus a rank-1 inter-telescope common differential mode. |
| `quadrature_fringe_tracker.py` | Ideal duplicated-input quadrature-reference mesh primitives and analytic observability/CRLB estimates. |
| `scintillation.py` | Scintillation and amplitude-servo residual model. |
| `fringe_tracking.py` | Piston-servo bandwidth primitives; disturbance terms `von_karman_differential_piston_rms_m`, `piston_psd_knee_hz`, `servo_residual_variance_flat_knee`, and `vibration_residual_rms_nm`; accelerometer-fusion and structure-function helpers; `photon_kalman_residual_budget` / `PhotonKalmanResidual` for the frame-free photon-counting control model; and the AR(1) temporal-Kalman covariance surrogate used to turn bright-port CRLB measurements into mode-dependent residual piston covariance. |
| `systematic_bandwidth.py` | Effective bandwidth for null-instability noise. |
| `spectroscopy.py` | Characterization spectro-port SNR and max resolving power; AWG spectrograph model (`AWGSpectrum`, `awg_spectrum`, `awg_wavelength_grid`, `per_bin_snr`). |
| `mkid.py` | MKID constant-delta-E resolving power and true-to-measured energy-response matrix primitives. |
| `pdi.py` | Polarimetric differential imaging SNR primitives. |
| `characterization.py` | Known-position chip optimization for spectro mode. |
| `characterization_torch.py` | Torch autograd-Jacobian helper for characterization optimization. |
| `imaging.py` | Aperture-synthesis image reconstruction: dark-port beam patterns, simulated photon-count observations, matched-filter SNR maps, and wavelength-resolved NNLS reconstruction. |
| `fringe_observability.py` | Fringe-tracker piston and amplitude observability primitives: intensity response matrices at zero and nonzero piston, selectable-port fractional-intensity response matrices for scintillation sensing, SVD-based observability rank and singular values, CRLB estimates, linear Monte Carlo piston-estimation RMSE, amplitude post-processing Monte Carlo comparisons against ideal photometric taps, damped Gauss-Newton reconstruction/capture-range diagnostics, active acquisition scans, and pairwise ABCD cophasing-handoff simulations. |
| `reflected_light.py` | Reflected-light planet/star contrast: Lambertian phase function and `reflected_light_contrast` (geometric albedo × phase × `(R_p/a)²`). |
| `ideal_coronagraph.py` | Guyon et al. (2006) theoretically optimal coronagraph bound: order-2/4 removed-stellar-subspace planet throughput, uniform-disk leakage, and the photon-limited 5σ contrast curve drawn by the opt-in `detection_space` overlay (`[output.detection_space].ideal_coronagraph_orders`). |

Tests:

- `tests/physics/test_*.py`
- `tests/physics/notes/*.md` for derivation notes and pinning rationale.

## Output Layer

The output layer is registry-driven. Configs name plots/tables in `[output]`,
and the CLI resolves names against the shared registry.

Core containers:

- `nullsim/outputs/__init__.py`
  - `RunRecord`: one sweep cell or single run.
  - `RunCollection`: all records plus output directory.

- `nullsim/outputs/sweep_results.py`
  - Writes long-format `sweep_results.parquet`.
  - Includes scalar results, photon-rate spectra, contrast curves, sensitivity
    scalars, and characterization scalars.

- `nullsim/outputs/paper.py`
  - Opt-in `[output].paper = true` bundle writer.
  - Creates a `paper/` directory beside `plots/` and `tables/`.
  - Reuses registered plot/table writers for KILO-paper-equivalent artifacts
    and adds paper-only products: schematic PDFs (`signal_chain.pdf`,
    `array_layout.pdf`, `baseline_histogram.pdf`, `cophasing_scheme.pdf`,
    `dirty_beams.pdf`, plus dual-pol layout and signal-chain variants) and
    LaTeX result tables (`exoplanet_table.tex`, `characterization_table.tex`,
    `pdi_exoplanet_table.tex`). The full emitted set, including intentionally
    out-of-scope products, is recorded in `paper_manifest.json`.

Registered plots:

| Name | File | Purpose |
|---|---|---|
| `contrast_curve` | `outputs/plots/__init__.py` | Basic contrast curve plot. |
| `detection_space` | `outputs/plots/detection_space.py` | Exoplanet population overlay with contrast curves. |
| `subaperture_layout` | `outputs/plots/subaperture_layout.py` | Pupil/sub-aperture layout panels. |
| `uv_coverage_grid` | `outputs/plots/uv_coverage_grid.py` | KILO-style UV coverage grid. |
| `chip_diagnostics` | `outputs/plots/chip_diagnostics.py` | Detection-chip power, matrix, DAC, and spectrum diagnostics. |
| `chip_diagnostics_characterization` | `outputs/plots/chip_diagnostics.py` | Characterization retune power, matrix, DAC, and spectro-port broadband-null diagnostics. |
| `cophasing_unitary` | `outputs/plots/cophasing_unitary.py` | Dedicated cophasing-mode ABCD analysis matrix, pair graph, OPD-incidence, and handoff summary diagnostics. |
| `loss_budget` | `outputs/plots/loss_budget.py` | Throughput/loss breakdown. |
| `contrast_curves_multiband` | `outputs/plots/contrast_curves_multiband.py` | Multi-band/multi-time contrast curves. |
| `calibration_time` | `outputs/plots/calibration_time.py` | Optimizer convergence / calibration-time style plot. |
| `debug_report` | `outputs/plots/debug_report.py` | Multi-page pipeline audit report. |
| `imaging_reconstruction` | `outputs/plots/imaging_reconstruction.py` | Truth/SNR/beam imaging artifacts for HWO-class space nuller image reconstruction. |
| `null_transmission_map` | `outputs/plots/null_transmission_map.py` | Optional design-wavelength and band-averaged summed dark-port transmission map over sky coordinates; shares chip-field retrieval via `outputs/plots/_chip_field_inputs.py`. |
| `fringe_observability` | `outputs/plots/fringe_observability.py` | Fringe-tracker piston/amplitude observability diagnostic panels: singular-value spectrum, amplitude rank/CRLB summary, scintillation and amplitude-post-processing summary, spectral-diversity rank/CRLB sweep, weakest-mode shape, science-unitary reconstruction residuals, active/cophasing acquisition and handoff residuals, capture sweep, and compact health summary. |

Registered tables:

| Name | File | Purpose |
|---|---|---|
| `performance_summary` | `outputs/tables/__init__.py` | One row per run with headline scalar metrics. |
| `characterization_summary` | `outputs/tables/__init__.py` | One row per `characterization.spectro` result component. |

Tests:

- `tests/outputs/test_outputs.py`
- `tests/outputs/test_end_to_end.py`
- `tests/outputs/test_paper.py`
- `tests/outputs/plots/test_*.py`

## Catalogs and Data

Key files:

- `nullsim/catalogs/exoplanets.py`
  - Validates NASA Exoplanet Archive-style CSV snapshots.
  - Computes or verifies snapshot hashes.
  - Classifies planets with versioned rules.
  - Derives separation in mas from semi-major axis and distance when needed.

- `nullsim/data/exoplanet_catalogs/nasa_exoplanet_archive_paperv3.csv`
  - Packaged exoplanet snapshot used by detection-space outputs.

- `nullsim/data/kilo_reference/*.json`
  - KILO reference data used for optional plot overlays and calibration floors.

- `nullsim/data/nayra_reference/*.json`
  - NAYRA E-ELT reference data; `calibration_time_J_M18.json` (N=18 sub-aperture baseline) and `calibration_time_J_M36.json` (N=36 sub-aperture baseline).

Tests:

- `tests/catalogs/test_exoplanets.py`
- `tests/catalogs/fixtures/sample_catalog.csv`

## Examples

Reference/instrument configs in the root of `examples/`:

- `examples/kilo_keck.toml`
  - KILO Keck-only paper-baseline pipeline; enables `[output].paper = true` so
    runs also produce the paper artifact bundle.
- `examples/kilo_keck_6.toml`
  - Default KILO Keck config: reduced 6-subaperture-per-pupil quadrature mesh
    (24 Clements inputs -> 12 tracker + 11 dark + 1 spectro) extending
    `kilo_keck.toml`. Run name `kilo_keck_6`; source of the phasing-paper
    bundle.
- `examples/nayra_eelt.toml`
  - Single-pupil E-ELT NAYRA at Cerro Armazones; N=48 sub-aperture J-primary baseline with PM980 fiber transport, generic sky background, sensitivity, characterization, and diagnostic plot outputs.
- `examples/hwo_space.toml`
  - HWO-class space nuller with no atmosphere/AO stages.

Auxiliary regression and smoke-test configs under `examples/test/`:

- `examples/test/vertical_slice.toml`
  - Fast architecture smoke test with simple stages and a J-mag sweep.
- `examples/test/kilo_maunakea.toml`
  - Compact four-telescope Maunakea KILO-style demonstrator.
- `examples/test/nayra_eelt_baseline.toml`
  - Fixed-point NAYRA E-ELT baseline config (no sweeps); used as a stable regression reference.
- `examples/test/detection_space_example.toml`
  - Minimal standalone detection-space output exercise.
- `examples/test/mkid_native_spectro.toml`
  - MKID-native spectroscopy smoke config.
- `examples/test/kilo_vs_hwo_study.toml`
  - Cross-config comparison study.
- `examples/test/kilo_keck_quadrature_mesh.toml`
  - KILO Keck with a 48-mode duplicated-input Clements mesh (mode splitter + quadrature reference stage + fringe observability, active acquisition, and cophasing handoff diagnostics).

Example tests live under `tests/examples/` and exercise config loading,
structural invariants, and runnable end-to-end slices:

```bash
conda run -n py313 pytest
```

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

Targeted examples:

```bash
conda run -n py313 pytest tests/pipeline
conda run -n py313 pytest tests/stages
conda run -n py313 pytest tests/examples/test_vertical_slice.py
```

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
