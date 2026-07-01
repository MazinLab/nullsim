# nullsim

`nullsim` is a Python package and CLI for simulating photonic nulling
instruments. Configure an instrument in a single TOML file and the pipeline
composes physics modules — atmosphere, AO, injection, fiber transport, delay
lines, photonic chip, detectors, post-processing — as an ordered sequence of
stage instances with declared `consumes`/`produces` dependencies validated
before execution. The same package targets ground-based interferometric
nullers (KILO-class multi-telescope arrays), single-pupil ELTs (NAYRA-class),
space-based nullers (HWO and successors), and lab-only chip characterization,
by enabling or disabling stages rather than by editing code.

## Installation

```bash
pip install -e .              # base install
pip install -e .[torch]       # add Torch chip-optimization backend
pip install -e .[dev]         # add pytest and parallel test runner
```

Requires Python 3.12 or newer. The base install pulls only `numpy`, `scipy`,
`matplotlib`, `pandas`, `pyarrow`, `pydantic>=2`, and `tqdm`; the chip-
optimization backends are optional extras.

## Quick start

The shipped baseline configs in `examples/` are the canonical entry points:

| Config | What it does |
|---|---|
| `examples/kilo_keck.toml` | KILO Keck-only paper-baseline J-band example. |
| `examples/kilo_keck_6.toml` | Default KILO Keck config: reduced N=6-per-Keck quadrature mesh extending `kilo_keck.toml`; source config for the current KILO phasing-paper bundle. |
| `examples/nayra_eelt.toml` | Single-pupil NAYRA-style E-ELT example. |
| `examples/hwo_space.toml` | HWO-class space nuller; no atmosphere, no AO, MKID detector. First non-KILO/NAYRA instrument. |

Auxiliary regression and smoke-test configs live under `examples/test/`,
including the minimal vertical slice, compact KILO Maunakea demonstrator, MKID
native-spectroscopy demo, study-layer examples, and the KILO duplicated-input
quadrature mesh diagnostic used for fringe-tracker acquisition and cophasing
handoff studies.

Run any config end-to-end:

```bash
nullsim validate examples/test/vertical_slice.toml       # schema, outputs, and dependency check
nullsim inspect  examples/test/vertical_slice.toml       # resolved run plan + trust notes
nullsim run --dry-run examples/test/vertical_slice.toml  # same plan through the run path
nullsim run       examples/test/vertical_slice.toml      # execute and write outputs
nullsim list-stages                                # show registered stage types
nullsim list-outputs                               # show registered plots and tables
```

Outputs land in the directory declared by `[output].dir` in the TOML
(default `./outputs/{run.name}/{run_hash}/`), alongside the resolved
config, a reproducibility manifest, and any requested plots/tables. Unknown
plot or table names are rejected before any expensive stages run.

## Pipeline architecture

The CLI loads a TOML config, validates it through a Pydantic schema,
expands declared parameter sweeps, then runs an ordered sequence of stage
instances. Each stage declares which `SimulationState` keys it consumes
and produces; the runner refuses to start if any consumed key is not
produced by an earlier instance. Results are cached on disk keyed by a
conservative content hash, so re-runs of unchanged studies are
near-instant. The `run_hash` folds in package version, key dependency
versions, and external catalog snapshots so output directories are honest
about identity. See [`nullsim.md`](nullsim.md) for the full design
document.

## Physics sanity checks

The finite-star nulling path follows the Guyon et al. (2013) beam-combiner
model: each sky direction is propagated coherently through the chip unitary,
then intensities are integrated incoherently over the stellar disk.
`chip_optimization.ideal_kernel` uses the SVD/PCA stellar modes ordered by
singular value for clean upper-bound studies; the iterative scipy/torch chip
optimizers may use different objectives, but they consume the same coherent
field and dark-port leakage conventions. `sensitivity.detection_curve` requires
the separation grid to cover the stellar radius when
`scene.star.angular_diameter_mas` is non-zero, so finite-disk leakage cannot
silently collapse to zero on an undersampled grid.

## Running the tests

```bash
pytest tests/             # full suite
pytest -m stats           # only stochastic / statistical-validation tests
```

The test suite uses the markers declared in `pyproject.toml`. Tests
marked `xfail(strict=True)` are pinning known invariants — they fail
loudly if they ever start passing.

## What ships in v0.1

- 15+ stages spanning atmosphere → AO → injection → transport → delay
  lines → chip → chip optimization → detectors → fringe tracking →
  calibration → performance.
- Root baseline configs for KILO Keck, NAYRA E-ELT, and HWO space, plus
  auxiliary regression configs under `examples/test/`.
- Fringe-tracker observability diagnostics for small-signal science-unitary
  tracking, active acquisition, and mode-switched cophasing handoff studies.
- Functional tests for the example configs and core physics/stage contracts.
- The headline `detection_space` output with versioned exoplanet
  catalog snapshots; refreshing the snapshot bumps `run_hash` so
  published figures cannot drift silently.
- Reproducibility manifest with `config_hash` and `run_hash` separated
  per SPEC §6.4.
- CLI: `nullsim run`, `nullsim run --dry-run`, `nullsim validate`,
  `nullsim inspect`, `nullsim list-stages`, `nullsim list-outputs`.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the development workflow,
coding standards, and test guidance, and [`nullsim.md`](nullsim.md) for the
design document and implementation roadmap.

## License

BSD 3-Clause. See [`LICENSE`](LICENSE).
