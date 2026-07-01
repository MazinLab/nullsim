"""Command-line entry points for the nullsim vertical slice (SPEC §10).

The CLI exposes the subcommands required to drive and inspect a nullsim
pipeline from a single TOML file:

* ``run``          — load, validate, expand sweeps, execute, and write outputs.
* ``validate``     — load and validate a TOML config without executing.
* ``inspect``      — print the resolved execution/output/trust plan.
* ``list-stages``  — list registered stages, optionally filtered by family.
* ``list-outputs`` — list registered plot and table outputs.

Cache-management subcommands described in SPEC §10 remain out of scope until a
downstream WP needs them.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import shutil
import sys
from collections.abc import Iterable, Sequence
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

import nullsim  # noqa: F401  # ensures stage + output autoload
from nullsim.config import (
    NullsimConfigError,
    ResolvedConfig,
    expand_sweeps,
    load_config,
)
from nullsim.outputs import RunCollection, RunRecord, _autoload_outputs
from nullsim.outputs.paper import write_paper_outputs
from nullsim.outputs.sweep_results import write_sweep_results
from nullsim.pipeline import (
    RunResult,
    build_stage_instances,
    external_dependency_digest,
    registry,
    run,
    validate_dependencies,
)
from nullsim.pipeline.manifest import write_manifest
from nullsim.pipeline.stage import consumed_state_keys

_PROCESS_POOL_UNAVAILABLE_ERRORS = (OSError, NotImplementedError)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point used by both the console-script and ``python -m nullsim``."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 2
    try:
        return int(handler(args) or 0)
    except NullsimConfigError as exc:
        print(f"nullsim: config error: {exc}", file=sys.stderr)
        return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nullsim",
        description="Photonic nulling instrument simulator.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run",
        help="Execute a TOML config and write the requested outputs.",
    )
    run_parser.add_argument("config", type=Path, help="Path to nullsim TOML config.")
    run_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Override the output directory declared in [output].dir.",
    )
    run_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the disk cache for this run.",
    )
    run_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of parallel sweep-cell workers. Overrides [run].n_workers. "
            "Defaults to min(n_cells, cpu_count())."
        ),
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate the expanded run, output names, and stage dependency "
            "plan without executing stages or creating outputs."
        ),
    )
    run_parser.set_defaults(handler=_handle_run)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a TOML config without executing the pipeline.",
    )
    validate_parser.add_argument("config", type=Path)
    validate_parser.set_defaults(handler=_handle_validate)

    list_parser = subparsers.add_parser(
        "list-stages",
        help="List stage type names registered in the runtime registry.",
    )
    list_parser.add_argument(
        "--family",
        default=None,
        help="Filter stages by family (scene, telescope, injection, ...).",
    )
    list_parser.set_defaults(handler=_handle_list_stages)

    output_parser = subparsers.add_parser(
        "list-outputs",
        help="List registered plot and table output names.",
    )
    output_parser.add_argument(
        "--kind",
        choices=("all", "plots", "tables"),
        default="all",
        help="Filter by output kind.",
    )
    output_parser.set_defaults(handler=_handle_list_outputs)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Validate and print a non-executing run plan for a TOML config.",
    )
    inspect_parser.add_argument("config", type=Path)
    inspect_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Override the output directory when showing the planned run.",
    )
    inspect_parser.add_argument(
        "--resolved",
        action="store_true",
        help="Also print the canonical resolved TOML.",
    )
    inspect_parser.set_defaults(handler=_handle_inspect)

    return parser


def _run_one_cell_worker(
    payload: tuple[int, ResolvedConfig, tuple],
) -> tuple[int, RunResult, tuple]:
    """Top-level ProcessPoolExecutor worker: run one sweep cell.

    Mirrors KILO's ``_process_worker_run_restart`` pattern (kilo/chip.py:762).
    Top-level (not a closure) so it pickles cleanly under the ``spawn``
    multiprocessing context. ``import nullsim`` re-triggers stage / output
    autoload inside the worker, since spawn'd workers start from a clean
    interpreter.
    """

    import nullsim  # noqa: F401
    from nullsim.pipeline import run as _run

    cell_idx, cell_config, coord_tuple = payload
    cell_result = _run(cell_config)
    return cell_idx, cell_result, coord_tuple


def _handle_run(args: argparse.Namespace) -> int:
    resolved = _load_config_friendly(args.config)
    cells = expand_sweeps(resolved)
    if not cells:
        raise NullsimConfigError(
            f"{args.config}: sweep expansion produced zero cells. "
            "Check [[sweep]] axes for empty value lists or degenerate "
            "linspace/logspace specs."
        )
    _validate_requested_outputs(resolved)

    # The "executed" config is the first cell's materialized form (with
    # [scene.*] defaults injected into stage params via the config-layer
    # materializer). For no-sweep runs cells[0].config is the only one;
    # for sweeps it is representative of what actually ran. Pinning
    # *everything* downstream (run_hash, output_dir, manifest) to the
    # materialized config means the manifest.config_hash and the hash
    # folded into run_hash both reference the same digest — otherwise
    # the run directory and the manifest disagree about which config
    # was executed (Finding 2).
    primary_resolved = cells[0].config
    primary_config_hash = primary_resolved.config_hash

    # Build instances for every cell so we can collect each cell's
    # external_dependencies() and fold them into the run hash. The
    # pipeline list is identical across cells, but stage params can
    # carry external-dependency paths (e.g. calibration_floor lookup
    # tables) that sweeps may mutate. Using cell-0 only would let a
    # sweep over an external-dep path produce a run_hash blind to the
    # files later cells actually read — silently breaking the SPEC §6.4
    # contract.
    cell0_instances = build_stage_instances(primary_resolved)
    all_cell_instances: list[Sequence[Any]] = [cell0_instances]
    for cell in cells[1:]:
        all_cell_instances.append(build_stage_instances(cell.config))
    for instances in all_cell_instances:
        validate_dependencies(instances)
    cell_config_hashes = tuple(cell.config.config_hash for cell in cells)
    primary_run_hash = _run_hash_with_externals(
        primary_resolved,
        all_cell_instances,
        sweep_config_hashes=cell_config_hashes,
    )
    cell_run_hashes = tuple(
        _run_hash_with_externals(cell.config, instances)
        for cell, instances in zip(cells, all_cell_instances, strict=True)
    )

    output_dir = _resolve_output_dir(primary_resolved, args.out, run_hash=primary_run_hash)

    if args.dry_run:
        _print_inspection_report(
            title="nullsim dry-run",
            resolved=primary_resolved,
            cells=cells,
            instances=cell0_instances,
            output_dir=output_dir,
            run_hash=primary_run_hash,
        )
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    # Persist the canonical resolved TOML next to outputs — the
    # materialized form, so a reader can reproduce the run from this
    # file alone without re-running the materializer.
    resolved.write_toml(output_dir / "config.sweep.resolved.toml")
    primary_resolved.write_toml(output_dir / "config.resolved.toml")
    if args.config.is_file():
        # Code Review 2026-05-16 Issue 23: don't silently swallow OSError
        # — a missing-permissions / disk-full / I/O failure on the user's
        # original config is a real problem the user should see. We still
        # tolerate the case where the input path isn't a regular file (no
        # copy to make).
        shutil.copyfile(args.config, output_dir / "config.toml")

    # Materialize per-cell configs and output dirs in the parent so the
    # workers do no filesystem setup (avoids races on shared paths and
    # keeps the worker payload pure compute).
    cell_configs: list[ResolvedConfig] = []
    cell_dirs: list[Path] = []
    for cell, cell_run_hash in zip(cells, cell_run_hashes, strict=True):
        cell_base_config = cell.config.model_copy(update={"run_hash": cell_run_hash})
        cell_config = _maybe_disable_cache(cell_base_config, no_cache=args.no_cache)
        cell_dir = _cell_output_dir(output_dir, cell_base_config, len(cells))
        cell_dir.mkdir(parents=True, exist_ok=True)
        cell_config = _with_runtime_output_dir(cell_config, cell_dir)
        cell_configs.append(cell_config)
        cell_dirs.append(cell_dir)

    # CLI --workers overrides [run].n_workers; default cell-bounded cpu_count.
    n_workers = args.workers if args.workers is not None else primary_resolved.run.n_workers
    if n_workers is None:
        n_workers = min(len(cells), os.cpu_count() or 1)
    n_workers = max(1, min(n_workers, len(cells)))

    if n_workers > 1 and len(cells) > 1:
        print(
            f"nullsim: running {len(cells)} sweep cells on {n_workers} worker(s)",
            flush=True,
        )
        records = _try_run_cells_parallel(
            cells=cells,
            cell_configs=cell_configs,
            cell_dirs=cell_dirs,
            n_workers=n_workers,
        )
        if records is None:
            records = _run_cells_serial(cells, cell_configs, cell_dirs)
    else:
        records = _run_cells_serial(cells, cell_configs, cell_dirs)

    # Use ``primary_resolved`` (the materialized cell-0 config the run
    # actually executed against), not ``resolved`` (the pre-materialize
    # view from load_config). Earlier plots that read
    # ``collection.config.scene.star.*`` to recover stage params saw
    # different values than the manifest and the on-disk
    # config.resolved.toml, because materialization injects [scene.*]
    # defaults into stage specs.
    output_config = primary_resolved.model_copy(update={"run_hash": primary_run_hash})
    collection = RunCollection(
        config=output_config,
        records=tuple(records),
        output_dir=output_dir,
    )

    # Trigger plot/table registration only on the run path so
    # ``nullsim validate`` and ``nullsim list-stages`` don't pull
    # matplotlib into the process unnecessarily.
    _autoload_outputs()
    _write_registered_outputs(resolved, collection, output_dir)
    if resolved.output.paper:
        write_paper_outputs(collection, output_dir)
    write_sweep_results(collection, output_dir, resolved.output.sweep_table)
    _write_run_manifest(
        output_dir=output_dir,
        resolved=primary_resolved,
        records=records,
        instances=(
            all_cell_instances if len(all_cell_instances) > 1 else cell0_instances
        ),
        run_hash=primary_run_hash,
        config_hash=primary_config_hash,
    )

    # Repoint a "latest" symlink next to the run dir so the hex-hash
    # directory names aren't the only handle the user has on a result.
    # Best-effort; the run is already on disk.
    _update_latest_symlink(output_dir)

    print(f"nullsim: wrote outputs to {output_dir}")
    return 0


def _update_latest_symlink(output_dir: Path) -> None:
    """Repoint ``<output_dir>.parent/latest`` to ``output_dir``.

    Created as a relative symlink so the parent results dir can be moved
    without breaking the alias. If a non-symlink file or directory
    already occupies that path, leaves it untouched (the user may have
    named a real run "latest" on purpose). Symlink creation failures
    are swallowed because the run's actual artifacts are already on
    disk and the alias is cosmetic.
    """
    latest = output_dir.parent / "latest"
    try:
        if latest.is_symlink():
            latest.unlink()
        elif latest.exists():
            return
        latest.symlink_to(output_dir.name)
    except OSError:
        pass


def _try_run_cells_parallel(
    *,
    cells: Sequence[Any],
    cell_configs: Sequence[ResolvedConfig],
    cell_dirs: Sequence[Path],
    n_workers: int,
) -> list[RunRecord] | None:
    """Run sweep cells with a spawn pool, or return None if pools are unavailable."""

    # Coordinate the intra-op thread budget across the outer cell pool. Each
    # spawned worker otherwise defaults torch / BLAS to a full-core thread pool,
    # so N concurrent cells oversubscribe the CPU and thrash (measured: the
    # kilo_keck_6 sweep ran far slower than its single-cell floor). Cap per
    # worker threads to cores / n_workers and set it in the parent env so each
    # spawn'd child inherits the cap before numpy / torch initialize their pools.
    # setdefault so an explicit user OMP_NUM_THREADS (etc.) still wins.
    intra_op_threads = max(1, (os.cpu_count() or 1) // max(1, n_workers))
    for _thread_var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(_thread_var, str(intra_op_threads))

    # KILO chip.py L1242: spawn (not fork) so each worker is a clean
    # interpreter — avoids fork-with-active-torch / matplotlib hangs on macOS.
    payloads = [
        (idx, cfg, cell.coord_tuple)
        for idx, (cell, cfg) in enumerate(zip(cells, cell_configs))
    ]
    ctx = mp.get_context("spawn")
    records: list[RunRecord | None] = [None] * len(cells)
    # Only pool CREATION / submission failures mean "pools are unavailable
    # in this environment". Exceptions re-raised from fut.result() come
    # from INSIDE a worker cell — a stage raising FileNotFoundError (an
    # OSError) used to be misclassified as pool unavailability, triggering
    # a silent full serial re-run with a misleading diagnostic (review
    # 2026-06-10 Issue 14). Worker exceptions now propagate with the cell
    # index attached; only BrokenExecutor (a worker process died abruptly)
    # still falls back to the serial path so the real error surfaces
    # in-process.
    try:
        pool = ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx)
    except _PROCESS_POOL_UNAVAILABLE_ERRORS as exc:
        _warn_process_pool_unavailable("sweep cells", exc)
        return None
    with pool:
        try:
            futures = {pool.submit(_run_one_cell_worker, p): p[0] for p in payloads}
        except _PROCESS_POOL_UNAVAILABLE_ERRORS as exc:
            _warn_process_pool_unavailable("sweep cells", exc)
            return None
        for completed, fut in enumerate(as_completed(futures), start=1):
            submitted_idx = futures[fut]
            try:
                cell_idx, cell_result, coord_tuple = fut.result()
            except BrokenExecutor as exc:
                _warn_process_pool_unavailable("sweep cells", exc)
                return None
            except Exception as exc:
                exc.add_note(
                    f"nullsim: raised while running sweep cell {submitted_idx} "
                    f"(coord {payloads[submitted_idx][2]!r})."
                )
                raise
            records[cell_idx] = RunRecord(
                config=cell_configs[cell_idx],
                result=cell_result,
                run_dir=cell_dirs[cell_idx],
                coord_tuple=coord_tuple,
            )
            print(
                f"  [{completed}/{len(cells)}] cell {cell_idx} done",
                flush=True,
            )

    if any(record is None for record in records):
        missing = [idx for idx, record in enumerate(records) if record is None]
        raise RuntimeError(f"nullsim: cells {missing} produced no result")
    return [record for record in records if record is not None]


def _run_cells_serial(
    cells: Sequence[Any],
    cell_configs: Sequence[ResolvedConfig],
    cell_dirs: Sequence[Path],
) -> list[RunRecord]:
    """Run sweep cells in-process."""

    records: list[RunRecord] = []
    for idx, (cell, cfg) in enumerate(zip(cells, cell_configs)):
        cell_result = run(cfg)
        records.append(
            RunRecord(
                config=cfg,
                result=cell_result,
                run_dir=cell_dirs[idx],
                coord_tuple=cell.coord_tuple,
            )
        )
    return records


def _warn_process_pool_unavailable(scope: str, exc: BaseException) -> None:
    """Print the restricted-environment process-pool fallback."""

    message = (
        f"process pool unavailable for {scope} "
        f"({type(exc).__name__}: {exc}); running serially"
    )
    print(f"nullsim: {message}", file=sys.stderr, flush=True)


def _handle_validate(args: argparse.Namespace) -> int:
    resolved = _load_config_friendly(args.config)
    cells = expand_sweeps(resolved)
    if not cells:
        raise NullsimConfigError(
            f"{args.config}: sweep expansion produced zero cells. "
            "Check [[sweep]] axes for empty value lists or degenerate "
            "linspace/logspace specs."
        )
    _validate_requested_outputs(resolved)
    # SPEC §7.3 step 6: validate consume/produce dependency graph before
    # any execution. Some stage dependencies are param-dependent, and sweeps
    # can mutate those params, so every expanded cell must be checked.
    for cell in cells:
        instances = build_stage_instances(cell.config)
        validate_dependencies(instances)
    print(
        f"nullsim: config OK ({len(cells)} sweep cell(s), "
        f"config_hash={resolved.config_hash[:12]}, run_hash={resolved.run_hash[:12]})"
    )
    return 0


def _handle_list_stages(args: argparse.Namespace) -> int:
    type_names = registry.list_stages(family=args.family)
    if not type_names:
        print("nullsim: no stages registered.")
        return 0
    for type_name in type_names:
        stage_cls = registry.stages[type_name]
        family = getattr(stage_cls, "family", "?")
        print(f"{type_name}\t({family})")
    return 0


def _handle_list_outputs(args: argparse.Namespace) -> int:
    """List output names from the live registry."""

    _autoload_outputs()
    if args.kind in {"all", "plots"}:
        print("plots:")
        for name in sorted(registry.plots):
            print(f"  {name}")
    if args.kind in {"all", "tables"}:
        print("tables:")
        for name in sorted(registry.tables):
            print(f"  {name}")
    return 0


def _handle_inspect(args: argparse.Namespace) -> int:
    """Print a cheap, non-executing run plan."""

    resolved = _load_config_friendly(args.config)
    cells = expand_sweeps(resolved)
    if not cells:
        raise NullsimConfigError(
            f"{args.config}: sweep expansion produced zero cells. "
            "Check [[sweep]] axes for empty value lists or degenerate "
            "linspace/logspace specs."
        )
    _validate_requested_outputs(resolved)

    primary_resolved = cells[0].config
    all_cell_instances = [build_stage_instances(cell.config) for cell in cells]
    for instances in all_cell_instances:
        validate_dependencies(instances)
    primary_run_hash = _run_hash_with_externals(
        primary_resolved,
        all_cell_instances,
        sweep_config_hashes=tuple(cell.config.config_hash for cell in cells),
    )
    output_dir = _resolve_output_dir(primary_resolved, args.out, run_hash=primary_run_hash)
    _print_inspection_report(
        title="nullsim inspect",
        resolved=primary_resolved,
        cells=cells,
        instances=all_cell_instances[0],
        output_dir=output_dir,
        run_hash=primary_run_hash,
    )
    if args.resolved:
        print()
        print("[resolved TOML]")
        print(primary_resolved.resolved_toml_text, end="")
    return 0


def _load_config_friendly(path: Path) -> ResolvedConfig:
    """Load a TOML config, converting filesystem errors into config errors.

    ``load_config()`` opens the file with ``pathlib.Path.open``; missing,
    directory-shaped, or unreadable paths surface as ``FileNotFoundError``,
    ``IsADirectoryError``, or ``PermissionError`` respectively.  The CLI is
    a user surface (SPEC §10), so those are funnelled through
    :class:`NullsimConfigError` to get the friendly ``nullsim: config error:
    ...`` rendering and exit code 2 already wired up in :func:`main`.
    """

    try:
        return load_config(path)
    except (FileNotFoundError, IsADirectoryError, PermissionError) as exc:
        raise NullsimConfigError(str(exc)) from exc


def _validate_requested_outputs(resolved: ResolvedConfig) -> None:
    """Reject unknown requested outputs before any expensive pipeline work."""

    if not resolved.output.plots and not resolved.output.tables:
        return
    _autoload_outputs()
    unknown_plots = [
        plot_name for plot_name in resolved.output.plots if plot_name not in registry.plots
    ]
    unknown_tables = [
        table_name
        for table_name in resolved.output.tables
        if table_name not in registry.tables
    ]
    if not unknown_plots and not unknown_tables:
        return

    parts: list[str] = []
    if unknown_plots:
        parts.append(
            "unknown plot output(s): "
            f"{', '.join(repr(name) for name in unknown_plots)}"
        )
        parts.append(
            "registered plots: " + _format_names(sorted(registry.plots.keys()))
        )
    if unknown_tables:
        parts.append(
            "unknown table output(s): "
            f"{', '.join(repr(name) for name in unknown_tables)}"
        )
        parts.append(
            "registered tables: " + _format_names(sorted(registry.tables.keys()))
        )
    raise NullsimConfigError("; ".join(parts))


def _print_inspection_report(
    *,
    title: str,
    resolved: ResolvedConfig,
    cells: Sequence[Any],
    instances: Sequence[Any],
    output_dir: Path,
    run_hash: str,
) -> None:
    """Render a human-readable run plan."""

    source = resolved.source_path if resolved.source_path is not None else "<memory>"
    print(f"{title}: {source}")
    print(f"run: {resolved.run.name}")
    if resolved.run.description:
        print(f"description: {resolved.run.description}")
    print(f"sweep cells: {len(cells)}")
    print(f"config_hash: {resolved.config_hash[:12]}")
    print(f"run_hash: {run_hash[:12]}")
    print(f"output_dir: {output_dir}")
    print(
        "outputs: "
        f"plots={_format_names(resolved.output.plots)}; "
        f"tables={_format_names(resolved.output.tables)}; "
        f"paper={resolved.output.paper}"
    )
    if cells and getattr(cells[0], "coord_tuple", ()):
        print("first cell coordinates:")
        for axis_name, axis_value in cells[0].coord_tuple:
            print(f"  {axis_name} = {axis_value}")

    print("pipeline:")
    for index, instance in enumerate(instances, start=1):
        instance_id = getattr(instance.context, "instance_id", f"stage_{index}")
        type_name = getattr(type(instance), "type_name", type(instance).__name__)
        family = getattr(type(instance), "family", "?")
        consumes = _format_names(sorted(consumed_state_keys(instance)))
        produces = _format_names(sorted(getattr(instance, "produces", ())))
        print(f"  {index}. {instance_id}: {type_name} ({family})")
        print(f"     consumes: {consumes}")
        print(f"     produces: {produces}")

    notes = _physics_trust_notes(resolved)
    if notes:
        print("physics/trust notes:")
        for note in notes:
            print(f"  - {note}")
    else:
        print("physics/trust notes: none")


def _physics_trust_notes(resolved: ResolvedConfig) -> list[str]:
    """Return lightweight warnings about known surrogate or placeholder physics."""

    notes: list[str] = []
    for spec in resolved.pipeline.stages:
        stage_id = _stage_config_value(spec, "id", "<unknown>")
        type_name = _stage_config_value(spec, "type", "<unknown>")
        if type_name == "scene.point_source":
            notes.append(
                f"{stage_id}: scene.point_source is a scalar one-star/one-planet "
                "source model; target-list, orbit, spectrum, and multi-planet "
                "effects are outside this run."
            )
        elif type_name == "telescope.pupil_modes":
            notes.append(
                f"{stage_id}: telescope.pupil_modes is a monolithic-pupil "
                "surrogate geometry bridge, not a pupil-resolved propagation model."
            )
        elif type_name == "wavefront.space_residual":
            notes.append(
                f"{stage_id}: wavefront.space_residual records scalar WFE/Strehl "
                "and pointing jitter; it does not generate residual speckle maps."
            )
        elif type_name == "calibration.floor_lookup":
            lookup_path = _stage_config_value(spec, "lookup_table_path")
            default_floor = float(_stage_config_value(spec, "default_floor_log10", -6.0))
            if lookup_path is None:
                notes.append(
                    f"{stage_id}: calibration.floor_lookup uses "
                    f"default_floor_log10={default_floor:+.2f}; no calibration-time "
                    "lookup table is configured."
                )
        elif type_name == "sensitivity.detection_curve":
            if _stage_config_value(spec, "residual_starlight_table_path") is not None:
                notes.append(
                    f"{stage_id}: residual starlight comes from a radial/table "
                    "profile rather than a wavefront-derived image-plane model."
                )
            n_realizations = int(_stage_config_value(spec, "n_mc_realizations", 0) or 0)
            n_per_az = int(
                _stage_config_value(spec, "n_realizations_per_azimuth", 0) or 0
            )
            if n_realizations <= 0 and n_per_az <= 0:
                notes.append(
                    f"{stage_id}: realized-null Monte Carlo is disabled for this "
                    "sensitivity calculation."
                )
        elif type_name == "detector.mkid":
            cap = float(_stage_config_value(spec, "mkid_max_count_rate_hz", 5.0e4))
            if cap >= 1.0e9:
                notes.append(
                    f"{stage_id}: MKID saturation cap is {cap:.3g} Hz per pixel, "
                    "effectively disabling detector saturation."
                )
        elif type_name == "instrument.nayra_six_chip":
            notes.append(
                f"{stage_id}: instrument.nayra_six_chip records the multichip "
                "architecture manifest; it does not itself orchestrate all bands."
            )
        elif type_name == "characterization.spectro":
            n_wl = int(_stage_config_value(spec, "broadband_n_wavelengths", 1) or 1)
            if n_wl <= 1:
                notes.append(
                    f"{stage_id}: characterization.spectro is configured as a "
                    "single-wavelength characterization solve."
                )
    if "imaging_reconstruction" in resolved.output.plots:
        notes.append(
            "imaging_reconstruction is an output-side reconstruction product, "
            "not a validated pipeline science stage."
        )
    return notes


def _stage_config_value(spec: Any, key: str, default: Any = None) -> Any:
    """Return a stage spec field from pydantic, object, or mapping inputs."""

    if hasattr(spec, "model_dump"):
        data = spec.model_dump(mode="python")
        return data.get(key, default)
    if isinstance(spec, dict):
        return spec.get(key, default)
    return getattr(spec, key, default)


def _format_names(names: Sequence[str]) -> str:
    if not names:
        return "none"
    return ", ".join(str(name) for name in names)


def _maybe_disable_cache(config: ResolvedConfig, *, no_cache: bool) -> ResolvedConfig:
    if not no_cache:
        return config
    cache_override = config.cache.model_copy(update={"enabled": False})
    return config.model_copy(update={"cache": cache_override})


def _with_runtime_output_dir(config: ResolvedConfig, output_dir: Path) -> ResolvedConfig:
    """Return a config whose stage RunContext resolves to the actual output dir.

    This deliberately does not recompute config_hash/run_hash: CLI ``--out``
    and per-cell output placement are runtime destinations, not scientific
    configuration identity.
    """

    output_override = config.output.model_copy(update={"dir": str(output_dir)})
    return config.model_copy(update={"output": output_override})


def _run_hash_with_externals(
    resolved: ResolvedConfig,
    instances: Sequence[Any] | Sequence[Sequence[Any]],
    *,
    sweep_config_hashes: Sequence[str] = (),
) -> str:
    """Recompute run_hash folding in every stage's external_dependencies.

    The loader computes ``resolved.run_hash`` with ``external_dep_hashes=()``
    because stage instances don't exist yet at config-load time. SPEC §6.4
    says ``run_hash`` must include external dependency hashes, so a catalog
    refresh produces a new output directory rather than silently shifting
    the points. This helper closes the loop after instances are built.

    SPEC §15.4 extension: a ``[catalog.exoplanets]`` block declared in the
    config is an external dependency of the same flavor — its content_hash
    enters the run identity here so refreshing the snapshot produces a new
    output directory and a fresh manifest entry, never a silent edit to an
    already-published ``detection_space`` figure.

    ``instances`` accepts either a flat list of stage instances (single
    cell) or a list-of-lists (one entry per sweep cell). Sweeps that
    mutate external-dependency paths produce per-cell externals that
    must all enter the run identity — see _handle_run.
    """
    from nullsim.config.catalog import parse_catalog_block
    from nullsim.config.hashes import (
        compute_run_hash,
        runtime_dependency_versions,
        runtime_python_version,
    )

    if instances and isinstance(instances[0], (list, tuple)):
        instance_groups: Sequence[Sequence[Any]] = instances  # type: ignore[assignment]
    else:
        instance_groups = (instances,)  # type: ignore[assignment]

    external_hashes: list[str] = []
    seen: set[str] = set()
    for config_hash in sweep_config_hashes:
        digest = f"sweep-cell-config:{config_hash}"
        if digest in seen:
            continue
        seen.add(digest)
        external_hashes.append(digest)
    for group in instance_groups:
        for instance in group:
            # Code Review 2026-05-16 Issue 51: narrow the catch. A stage's
            # external_dependencies() returning an unexpected type would
            # raise AttributeError/TypeError; a misbehaving stage class
            # raising something else should not be silently swallowed —
            # log so failures are visible. Matching the manifest writer's
            # policy.
            try:
                externals = instance.external_dependencies() or []
            except (AttributeError, TypeError, NotImplementedError) as exc:
                logging.warning(
                    "Stage %r external_dependencies() raised %s: %s — skipping.",
                    getattr(instance, "context", None),
                    type(exc).__name__,
                    exc,
                )
                continue
            for dep in externals:
                digest = external_dependency_digest(dep).hex()
                # Deduplicate so a cell that shares deps with cell 0
                # (the common case) doesn't inflate the hash.
                if digest in seen:
                    continue
                seen.add(digest)
                external_hashes.append(digest)

    raw_config = resolved.raw_config or {}
    catalog_block = None
    if isinstance(raw_config, dict):
        catalog_block = raw_config.get("catalog", {}).get("exoplanets")
    if catalog_block is not None:
        config_dir = resolved.source_path.parent if resolved.source_path is not None else None
        try:
            catalog_config = parse_catalog_block(catalog_block, config_dir=config_dir)
        except (NullsimConfigError, ValueError, TypeError) as exc:
            raise NullsimConfigError(
                f"Invalid [catalog.exoplanets] block: {exc}"
            ) from exc
        if catalog_config is not None:
            external_hashes.append(catalog_config.content_hash())

    return compute_run_hash(
        config_hash=resolved.config_hash,
        package_version=nullsim.__version__,
        external_deps_hashes_iterable=tuple(external_hashes),
        runtime_python_version=runtime_python_version(),
        key_dep_versions=runtime_dependency_versions(),
    )


def _resolve_output_dir(
    resolved: ResolvedConfig,
    override: Path | None,
    *,
    run_hash: str | None = None,
) -> Path:
    """Resolve [output].dir, substituting template variables.

    Supported: ``{run.<key>}``, ``{run_hash}``, ``{config_hash}``,
    ``{timestamp}``. The caller may pass a recomputed ``run_hash`` (one
    that includes stage external_dependencies — see
    ``_run_hash_with_externals``); falling back to ``resolved.run_hash``
    is correct only when no stages declare externals (the loader's
    default).
    """
    if override is not None:
        return Path(override)
    raw = resolved.output.dir
    try:
        formatted = raw.format(
            run=resolved.run,
            run_hash=run_hash or resolved.run_hash,
            config_hash=resolved.config_hash,
            timestamp=resolved.timestamp,
        )
    except (KeyError, AttributeError, IndexError) as exc:
        # Mirror ``nullsim.pipeline.runner._run_dir`` so a typo in
        # output.dir (e.g. ``{run.missing}``) surfaces as a friendly
        # config error in ``main()`` instead of an uncaught
        # AttributeError.
        raise NullsimConfigError(
            f"output.dir template contains an unresolved variable "
            f"({exc!r}); template = {raw!r}. Supported template fields: "
            "{run.<key>}, {run_hash}, {config_hash}, {timestamp}."
        ) from exc
    return Path(formatted)


def _cell_output_dir(
    output_dir: Path,
    cell_config: ResolvedConfig,
    cell_count: int,
) -> Path:
    if cell_count <= 1:
        return output_dir
    cell_hash = cell_config.run_hash or cell_config.config_hash or "cell"
    return output_dir / "cells" / cell_hash


def _write_registered_outputs(
    resolved: ResolvedConfig,
    collection: RunCollection,
    output_dir: Path,
) -> None:
    plot_formats = tuple(resolved.output.formats.plots)
    table_formats = tuple(resolved.output.formats.tables)

    for plot_name in resolved.output.plots:
        plot_callable = registry.plots.get(plot_name)
        if plot_callable is None:
            raise NullsimConfigError(f"Unknown plot output {plot_name!r}.")
        plot_callable(collection, output_dir, plot_formats)

    for table_name in resolved.output.tables:
        table_callable = registry.tables.get(table_name)
        if table_callable is None:
            raise NullsimConfigError(f"Unknown table output {table_name!r}.")
        table_callable(collection, output_dir, table_formats)


def _write_run_manifest(
    *,
    output_dir: Path,
    resolved: ResolvedConfig,
    records: Iterable[RunRecord],
    instances: Sequence[Any] | Sequence[Sequence[Any]],
    run_hash: str | None = None,
    config_hash: str | None = None,
) -> None:
    record_list = list(records)
    primary_result: RunResult | None = (
        record_list[0].result if record_list else None
    )
    if primary_result is None:
        return

    extra: dict[str, Any] = {}
    if len(record_list) > 1:
        extra["sweep"] = {
            "cell_count": len(record_list),
            "axes": _sweep_axes(record_list),
            "cells": [_cell_summary(record) for record in record_list],
        }

    write_manifest(
        output_dir / "manifest.json",
        config_hash=config_hash or resolved.config_hash,
        run_hash=run_hash or resolved.run_hash,
        rng_root_seed=resolved.run.seed,
        result=primary_result,
        results=tuple(record.result for record in record_list),
        instances=instances,
        extra=extra,
    )


def _sweep_axes(records: list[RunRecord]) -> list[str]:
    axes: list[str] = []
    seen: set[str] = set()
    for record in records:
        for axis_name, _ in record.coord_tuple:
            if axis_name not in seen:
                seen.add(axis_name)
                axes.append(axis_name)
    return axes


def _cell_summary(record: RunRecord) -> dict[str, Any]:
    return {
        "coord_tuple": [
            [str(axis_name), _json_safe(axis_value)]
            for axis_name, axis_value in record.coord_tuple
        ],
        "config_hash": getattr(record.config, "config_hash", ""),
        "run_hash": getattr(record.config, "run_hash", ""),
        "cache_hit_count": int(record.result.cache_hit_count),
        "stage_count": len(record.result.timings),
    }


def _json_safe(value: Any) -> Any:
    """Convert a value into a JSON-serializable form for the manifest.

    Code Review 2026-05-16 Issue 58: pre-fix the catch-all fell back to
    ``repr(value)``, so a ``datetime.date(2026, 5, 16)`` became the
    Python source string ``'datetime.date(2026, 5, 16)'`` in the
    manifest. Handle the common types explicitly and raise on anything
    else so opaque types surface rather than leak repr() strings.
    """

    import datetime as _dt

    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (_dt.date, _dt.datetime)):
        return value.isoformat()
    raise TypeError(
        f"_json_safe: cannot serialize {type(value).__name__} "
        f"to JSON; extend the manifest serializer if needed."
    )


__all__ = ["main"]
