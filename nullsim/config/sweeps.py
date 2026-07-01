"""Sweep expansion for SPEC §7 configs while preserving SPEC §6.4 identity.

The expander turns one resolved config containing ``[[sweep]]`` axes into
independent executable cells, each with its own resolved parameter values,
coordinate tuple, canonical TOML, ``config_hash``, and ``run_hash``.
"""

from __future__ import annotations

import copy
import itertools
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from nullsim.config.errors import NullsimConfigError
from nullsim.config.loader import resolve_config
from nullsim.config.schema import ResolvedConfig, SweepSpec
from nullsim.config.stage_materialize import (
    bound_top_level_paths,
    explicit_stage_params,
    materialize_pipeline_stages,
    stage_binding_sources,
)


@dataclass(frozen=True)
class SweepCell:
    """One expanded sweep coordinate and executable resolved config."""

    coord_tuple: tuple[tuple[str, Any], ...]
    config: ResolvedConfig

    def __iter__(self) -> Any:
        """Allow ``coord_tuple, config = cell`` unpacking."""

        yield self.coord_tuple
        yield self.config


def expand_sweeps(resolved_config: ResolvedConfig) -> list[SweepCell]:
    """Expand grid and zip sweep axes into independent resolved configs."""

    if not resolved_config.sweep:
        materialized = _materialized(resolved_config)
        return [SweepCell(materialized.sweep_coord_tuple, materialized)]

    grid_axes: list[list[tuple[tuple[str, Any], ...]]] = []
    zip_specs: list[SweepSpec] = []
    for sweep_spec in resolved_config.sweep:
        values = _values_for_spec(sweep_spec)
        axis = [((sweep_spec.param, value),) for value in values]
        if sweep_spec.mode == "grid":
            grid_axes.append(axis)
        else:
            zip_specs.append(sweep_spec)

    zip_axis = _zip_axis(zip_specs)
    product_axes = list(grid_axes)
    if zip_axis:
        product_axes.append(zip_axis)

    cells: list[SweepCell] = []
    for coord_groups in itertools.product(*product_axes):
        coord_tuple = tuple(
            coord
            for coord_group in coord_groups
            for coord in coord_group
        )
        cell_config = _config_for_coord_tuple(resolved_config, coord_tuple)
        cells.append(SweepCell(coord_tuple, cell_config))
    return cells


def _values_for_spec(sweep_spec: SweepSpec) -> tuple[Any, ...]:
    if sweep_spec.values is not None:
        return sweep_spec.values
    if sweep_spec.range is not None:
        return _range_values(
            sweep_spec.range.start,
            sweep_spec.range.stop,
            sweep_spec.range.step,
        )
    if sweep_spec.linspace is not None:
        return tuple(
            float(value)
            for value in np.linspace(
                sweep_spec.linspace.start,
                sweep_spec.linspace.stop,
                sweep_spec.linspace.num,
            )
        )
    if sweep_spec.logspace is not None:
        return tuple(
            float(value)
            for value in np.logspace(
                sweep_spec.logspace.start,
                sweep_spec.logspace.stop,
                sweep_spec.logspace.num,
            )
        )
    raise NullsimConfigError(f"Sweep {sweep_spec.param!r} has no value source.")


def _range_values(start: float, stop: float, step: float) -> tuple[float, ...]:
    """Half-open range [start, stop) with step, computed without drift.

    Computes each value as ``start + i * step`` (one ULP off the
    analytically-correct point regardless of how many steps fit in the
    interval) — Python's own builtin ``range`` rejects floats for the
    drift reason.

    Code Review 2026-05-16 Issue 19: the post-floor corrective ``while``
    loops are now bounded by an explicit iteration cap so float
    absorption (``start = 1e16, step = 1.0`` ⇒ ``start + n*step == start``
    for all n) raises rather than hangs.
    """
    import math

    if step == 0.0:
        raise NullsimConfigError("range sweep step must be non-zero.")
    if not (math.isfinite(start) and math.isfinite(stop) and math.isfinite(step)):
        raise NullsimConfigError(
            f"range sweep requires finite start/stop/step (got "
            f"start={start!r}, stop={stop!r}, step={step!r})."
        )
    span = stop - start
    if (step > 0.0 and span <= 0.0) or (step < 0.0 and span >= 0.0):
        raise NullsimConfigError("range sweep produced no values.")
    # Detect mantissa absorption upfront: if start + step is bit-identical
    # to start (loss-of-precision regime), no integer multiple of step
    # will ever reach stop. Pre-fix the post-floor while-loops never
    # terminated in this case.
    if start + step == start:
        raise NullsimConfigError(
            f"range sweep step {step!r} is lost in the floating-point "
            f"mantissa of start {start!r} (start + step == start). "
            "Use a larger step or smaller start."
        )
    n = int(math.floor(span / step))
    # span/step is exclusive of stop in float arithmetic; clamp the count
    # down by 1 if start + n*step would round past stop, and up by 1 if
    # the floor came in short. This is the standard "snap to integer
    # bounds, then verify" pattern. Bounded to a few iterations to guard
    # against unexpected float pathologies.
    _MAX_CORRECTIONS = 8
    if step > 0.0:
        corrections = 0
        while n > 0 and start + n * step >= stop and corrections < _MAX_CORRECTIONS:
            n -= 1
            corrections += 1
        corrections = 0
        while start + (n + 1) * step < stop and corrections < _MAX_CORRECTIONS:
            n += 1
            corrections += 1
    else:
        corrections = 0
        while n > 0 and start + n * step <= stop and corrections < _MAX_CORRECTIONS:
            n -= 1
            corrections += 1
        corrections = 0
        while start + (n + 1) * step > stop and corrections < _MAX_CORRECTIONS:
            n += 1
            corrections += 1
    n += 1  # half-open => count includes the start point
    if n <= 0:
        raise NullsimConfigError("range sweep produced no values.")
    return tuple(start + i * step for i in range(n))


def _zip_axis(zip_specs: list[SweepSpec]) -> list[tuple[tuple[str, Any], ...]]:
    if not zip_specs:
        return []

    value_axes = [_values_for_spec(sweep_spec) for sweep_spec in zip_specs]
    lengths = {len(values) for values in value_axes}
    if len(lengths) != 1:
        params = ", ".join(sweep_spec.param for sweep_spec in zip_specs)
        raise NullsimConfigError(f"zip sweep axes must have equal lengths: {params}.")

    return [
        tuple(
            (sweep_spec.param, values[index])
            for sweep_spec, values in zip(zip_specs, value_axes, strict=True)
        )
        for index in range(next(iter(lengths)))
    ]


def _config_for_coord_tuple(
    resolved_config: ResolvedConfig,
    coord_tuple: tuple[tuple[str, Any], ...],
) -> ResolvedConfig:
    config_data = _mutable_copy(resolved_config.config_mapping())
    for param_path, value in coord_tuple:
        _set_param_path(config_data, param_path, value)
    config_data["sweep"] = []
    # Force overwrite of stage-spec bindings: the prior load_config
    # materialize baked in the un-swept value (e.g. scene.star.jmag=4),
    # and the deep-copy preserves it. Without overwrite, the cell's
    # swept top-level value (scene.star.jmag=12) would silently fail
    # to reach the stage spec.
    #
    # Exclude any (stage_id, param) the sweep wrote directly: a sweep
    # over ``pipeline.stages.<id>.<param>`` is authoritative, and the
    # re-materializer would otherwise clobber it from the un-swept
    # top-level mirror, producing N identical cells with distinct
    # sweep_coord_tuples — silent wrong result.
    skips: set[tuple[str, str]] = set()
    for param_path, _value in coord_tuple:
        parts = param_path.split(".")
        if (
            len(parts) == 4
            and parts[0] == "pipeline"
            and parts[1] == "stages"
        ):
            skips.add((parts[2], parts[3]))
    # Also exclude every (stage_id, param) the user wrote explicitly on a
    # stage spec in the raw config. The no-sweep load path preserves
    # explicit stage-level overrides (overwrite=False); without this skip
    # the sweep path's overwrite=True re-copies the top-level binding and
    # the same config silently produces different stage params with and
    # without a [[sweep]] block (review 2026-06-10 Issue 6). An explicit
    # stage value that mirrors a *swept* top-level source is a genuine
    # conflict (the sweep would be silently inert for that stage), so it
    # raises instead.
    user_explicit = explicit_stage_params(resolved_config.raw_config)
    if user_explicit:
        swept_top_level_paths = {
            param_path
            for param_path, _value in coord_tuple
            if not param_path.startswith("pipeline.")
        }
        if swept_top_level_paths:
            _raise_on_explicit_override_of_swept_source(
                config_data,
                user_explicit,
                swept_top_level_paths,
            )
        skips.update(user_explicit)
    materialize_pipeline_stages(
        config_data,
        overwrite=True,
        skip_stage_params=frozenset(skips),
    )
    return resolve_config(
        config_data,
        raw_config=resolved_config.raw_config,
        source_path=resolved_config.source_path,
        sweep_coord_tuple=coord_tuple,
    )


def _raise_on_explicit_override_of_swept_source(
    config_data: Mapping[str, Any],
    user_explicit: frozenset[tuple[str, str]],
    swept_top_level_paths: set[str],
) -> None:
    """Reject explicit stage params that mirror a swept top-level source.

    An explicit ``star_jmag = 4`` on the scene stage combined with a sweep
    over ``scene.star.jmag`` would make every cell carry the explicit value
    — N identical cells with distinct coord tuples. Fail loudly instead.
    """

    pipeline = config_data.get("pipeline")
    stages = pipeline.get("stages") if isinstance(pipeline, Mapping) else None
    if not isinstance(stages, (list, tuple)):
        return
    for spec in stages:
        if not isinstance(spec, Mapping):
            continue
        stage_id = spec.get("id")
        if not isinstance(stage_id, str):
            continue
        for param_name, source_path in stage_binding_sources(spec.get("type")).items():
            if (
                (stage_id, param_name) in user_explicit
                and source_path in swept_top_level_paths
            ):
                raise NullsimConfigError(
                    f"Stage {stage_id!r} sets {param_name!r} explicitly, but "
                    f"the sweep varies its top-level source {source_path!r}. "
                    "The explicit stage value would make every sweep cell "
                    "identical for that parameter. Remove the stage-level "
                    f"value (it materializes from [{source_path.rsplit('.', 1)[0]}]) "
                    "or sweep the stage param directly via "
                    f"pipeline.stages.{stage_id}.{param_name}."
                )


def _materialized(resolved_config: ResolvedConfig) -> ResolvedConfig:
    """Re-resolve a config after baking stage-spec defaults — no sweep coord."""
    config_data = _mutable_copy(resolved_config.config_mapping())
    materialize_pipeline_stages(config_data)
    return resolve_config(
        config_data,
        raw_config=resolved_config.raw_config,
        source_path=resolved_config.source_path,
        sweep_coord_tuple=resolved_config.sweep_coord_tuple,
    )


def _set_param_path(
    config_data: MutableMapping[str, Any],
    param_path: str,
    value: Any,
) -> None:
    parts = param_path.split(".")
    if len(parts) < 2:
        raise NullsimConfigError(f"Sweep param {param_path!r} is not a dotted path.")

    root = parts[0]
    if root == "observation" and param_path not in bound_top_level_paths():
        # [observation] fields only reach the physics through explicit
        # stage-param bindings (stage_materialize). Sweeping an unbound
        # field would produce N cells with identical stage physics —
        # silent wrong results (review 2026-06-10 Issue 15).
        raise NullsimConfigError(
            f"Sweep param {param_path!r} is not materialized into any "
            "stage, so sweeping it would not change results. Bind it in "
            "stage_materialize._STAGE_CONFIG_BINDINGS or sweep the stage "
            "param directly via pipeline.stages.<id>.<field>."
        )
    if root in {"grid", "scene", "observation"}:
        _set_nested(config_data[root], parts[1:], value, param_path)
        return

    if root == "pipeline" and len(parts) >= 4 and parts[1] == "stages":
        stage_id = parts[2]
        stage = _stage_by_id(config_data, stage_id, param_path)
        _set_nested(stage, parts[3:], value, param_path)
        return

    raise NullsimConfigError(
        "Sweep params must reference grid.*, scene.*, observation.*, or "
        f"pipeline.stages.<id>.<field>; got {param_path!r}."
    )


def _stage_by_id(
    config_data: Mapping[str, Any],
    stage_id: str,
    param_path: str,
) -> MutableMapping[str, Any]:
    try:
        stages = config_data["pipeline"]["stages"]
    except KeyError as exc:
        raise NullsimConfigError("Sweep config does not define pipeline.stages.") from exc

    for stage in stages:
        if isinstance(stage, MutableMapping) and stage.get("id") == stage_id:
            return stage
    raise NullsimConfigError(
        f"Sweep param {param_path!r} references unknown stage id {stage_id!r}."
    )


def _set_nested(
    target: MutableMapping[str, Any],
    parts: list[str],
    value: Any,
    param_path: str,
) -> None:
    if not parts:
        raise NullsimConfigError(f"Sweep param {param_path!r} has no leaf field.")

    current: MutableMapping[str, Any] = target
    for part in parts[:-1]:
        nested = current.get(part)
        if not isinstance(nested, MutableMapping):
            raise NullsimConfigError(
                f"Sweep param {param_path!r} cannot descend through {part!r}."
            )
        current = nested
    current[parts[-1]] = value


def _mutable_copy(value: Any) -> Any:
    """Return a deep-copy of ``value`` with no tuple-to-list rebuild.

    Code Review 2026-05-16 Issue 63: the previous implementation called
    ``copy.deepcopy`` and then walked the structure to convert tuples to
    lists. Pydantic accepts both, so the rebuild was redundant and
    doubled the per-cell allocation. Returning ``copy.deepcopy(value)``
    directly halves sweep-expansion memory pressure.
    """
    return copy.deepcopy(value)
