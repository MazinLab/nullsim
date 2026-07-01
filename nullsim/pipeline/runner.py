"""Ordered stage validation and execution for SPEC §§6.3 and 6.4."""

from __future__ import annotations

import logging
import time

from nullsim.pipeline._canonical import canonical_digest
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Mapping

from nullsim.config import NullsimConfigError
from nullsim.pipeline.cache import CacheLoadError, ContentCache
from nullsim.pipeline.context import RunContext
from nullsim.pipeline.registry import build_instance, stage_id_from_spec
from nullsim.pipeline.rng import derive_rng
from nullsim.pipeline.stage import Stage, consumed_state_keys
from nullsim.pipeline.state import SimulationState


@dataclass(frozen=True)
class StageTiming:
    """Execution metadata for one stage instance."""

    instance_id: str
    elapsed_s: float
    cache_hit: bool


@dataclass(frozen=True)
class RunResult:
    """Result returned by the pipeline runner."""

    state: SimulationState
    diagnostics: dict[str, dict[str, Any]]
    timings: tuple[StageTiming, ...]
    cache_hit_count: int
    config_hash: str


def build_stage_instances(resolved_config: Any) -> list[Stage]:
    """Instantiate a resolved config's stage list without applying stages.

    Args:
        resolved_config: Mapping or object exposing ``pipeline.stages`` plus
            optional ``run.seed``, ``config_hash``, and ``sweep_coord_tuple``.

    Returns:
        Stage instances in declared execution order.

    Raises:
        NullsimConfigError: If any stage spec is malformed or references an
            unknown stage type.
    """

    config_hash = _config_hash(resolved_config)
    sweep_coord_tuple = _sweep_coord_tuple(resolved_config)
    root_seed = _root_seed(resolved_config)
    run_dir = _run_dir(resolved_config)
    source_path = _source_path(resolved_config)

    instances: list[Stage] = []
    for spec in _stage_specs(resolved_config):
        instance_id = stage_id_from_spec(spec)
        # RNG identity is the stage's OWN spec digest, not the full
        # config_hash: seeding from the whole resolved config made every
        # stochastic draw (and, through the input-state digests, every
        # downstream cache entry) change on cosmetic config edits such as
        # adding a plot name (review 2026-06-10 Issue 7). A stage's draw
        # now changes only when its own spec, the sweep coordinate, the
        # root seed, or its instance id changes.
        rng_identity = canonical_digest(_canonical_config(spec)).hex()
        context = RunContext(
            rng=derive_rng(root_seed, rng_identity, sweep_coord_tuple, instance_id),
            config_hash=config_hash,
            sweep_coord_tuple=sweep_coord_tuple,
            instance_id=instance_id,
            run_dir=run_dir,
            root_seed=root_seed,
            source_path=source_path,
            rng_identity=rng_identity,
        )
        instances.append(build_instance(spec, context))
    return instances


def validate_dependencies(instances: list[Stage] | tuple[Stage, ...]) -> None:
    """Validate consume/produce dependencies, with dotted-path semantics.

    SPEC §6.1 lets stages declare ``StateKey``s as either bare sub-budget
    names (``"field"``, ``"opd_budget"``) or dotted paths into a sub-budget
    (``"opd_budget.atmos_piston"``, ``"diagnostics.<instance_id>.<key>"``).
    A consume is satisfied when an earlier produce either:

      - matches it exactly (``produces={"opd_budget.atmos_piston"}`` covers
        ``consumes={"opd_budget.atmos_piston"}``), or
      - is broader, naming an ancestor in the dotted path
        (``produces={"opd_budget"}`` covers any ``opd_budget.*`` consume).

    A narrower produce does NOT cover a broader consume — declaring
    ``produces={"opd_budget.atmos_piston"}`` alone does not satisfy
    ``consumes={"opd_budget"}`` because the consumer wants the whole
    sub-budget. This matches the principle that producers must be at
    least as informative as consumers expect.

    Args:
        instances: Stage instances in TOML declaration order.

    Raises:
        NullsimConfigError: If any stage consumes a key that is not
            covered (exactly or broadly) by some earlier stage's produces.
    """

    produced_keys: set[str] = set()
    for instance in instances:
        unsatisfied = sorted(
            consumed
            for consumed in consumed_state_keys(instance)
            if not _consume_is_satisfied(consumed, produced_keys)
        )
        if unsatisfied:
            instance_id = _instance_id(instance)
            missing = ", ".join(unsatisfied)
            raise NullsimConfigError(
                f"Pipeline stage {instance_id!r} consumes missing state key(s): {missing}."
            )
        produced_keys.update(instance.produces)


def _consume_is_satisfied(consumed: str, produced_keys: set[str]) -> bool:
    """A consumed dotted path is covered by exact-match or broader ancestor."""
    if consumed in produced_keys:
        return True
    parts = consumed.split(".")
    # Walk ancestors from broadest to narrowest (just-shy-of-self).
    # Example: consumed="opd_budget.atmos_piston.rms_nm" checks
    # "opd_budget" then "opd_budget.atmos_piston".
    return any(".".join(parts[:i]) in produced_keys for i in range(1, len(parts)))


def run(resolved_config: Any) -> RunResult:
    """Execute a resolved config's stage list in declared order.

    Args:
        resolved_config: Mapping or object exposing ``pipeline.stages`` plus
            optional ``grid``, ``run.seed``, ``cache``, ``output.dir``,
            ``config_hash``, and ``sweep_coord_tuple`` values.

    Returns:
        Final state, diagnostics, per-instance timings, and cache-hit count.

    Raises:
        NullsimConfigError: If pipeline specs or dependencies are invalid.
    """

    config_hash = _config_hash(resolved_config)
    instances = build_stage_instances(resolved_config)

    validate_dependencies(instances)

    state = SimulationState.empty(_config_value(resolved_config, "grid", default=None))
    cache = ContentCache(_cache_dir(resolved_config), _cache_version(resolved_config))
    cache_enabled = _cache_enabled(resolved_config)
    diagnostics_by_instance: dict[str, dict[str, Any]] = {}
    timings: list[StageTiming] = []
    cache_hit_count = 0

    for instance in instances:
        instance_id = _instance_id(instance)
        cache_key = cache.key_for(instance, state) if cache_enabled else None
        started_s = time.perf_counter()
        cache_hit = False
        if cache_key is not None and cache.has(cache_key):
            try:
                state, diagnostics = cache.load(cache_key)
                cache_hit = True
                cache_hit_count += 1
            except CacheLoadError:
                # Corrupted entry was already deleted by cache.load; fall
                # through to recompute. This matches the conservative
                # cache contract: a bad slot becomes a miss, never a crash.
                cache_hit = False
        if not cache_hit:
            # Code Review 2026-05-16 Issue 24 (latent): snapshot the
            # produced sub-budgets' component sets BEFORE apply, then
            # after apply verify that nothing got wiped. SubBudget
            # ``add_component`` already guards against same-key
            # overwrites within a sub-budget, but whole-budget
            # replacement via ``replace(state, opd_budget=fresh_budget)``
            # silently dropped earlier contributions — only a warning
            # because some stages legitimately reset state.
            pre_apply_component_names = _produced_component_names(
                state, instance.produces
            )
            state = instance.apply(state)
            _warn_if_components_dropped(
                instance_id,
                pre_apply_component_names,
                _produced_component_names(state, instance.produces),
            )
            diagnostics = instance.diagnostics()
            if cache_key is not None:
                cache.store(cache_key, state, diagnostics)
        elapsed_s = time.perf_counter() - started_s
        diagnostics_by_instance[instance_id] = diagnostics
        timings.append(StageTiming(instance_id, elapsed_s, cache_hit))

    return RunResult(
        state=state,
        diagnostics=diagnostics_by_instance,
        timings=tuple(timings),
        cache_hit_count=cache_hit_count,
        config_hash=config_hash,
    )


# _budget_roots removed: validate_dependencies now uses dotted-path
# semantics (see _consume_is_satisfied). Sub-budget root extraction for
# the cache layer happens in state.SimulationState.content_digest_for.


def _instance_id(instance: Stage) -> str:
    context = getattr(instance, "context", None)
    instance_id = getattr(context, "instance_id", None)
    if isinstance(instance_id, str) and instance_id:
        return instance_id
    return getattr(instance, "type_name", type(instance).__name__)


def _stage_specs(resolved_config: Any) -> list[Any]:
    pipeline = _config_value(resolved_config, "pipeline", default=None)
    stages = _config_value(pipeline, "stages", default=None)
    if stages is None:
        stages = _config_value(resolved_config, "stages", default=None)
    if stages is None:
        raise NullsimConfigError("Resolved config must define pipeline.stages.")
    return list(stages)


# INS-3-002: removed — stage-spec materialization moved to
# nullsim.config.stage_materialize. The runner is now type-agnostic.


# INS-3-002: removed — was only used by _materialized_stage_spec.


def _produced_component_names(
    state: Any, produces: frozenset[str]
) -> dict[str, set[str]]:
    """Return ``{sub_budget_name: {component_names}}`` for each produced
    sub-budget. Sub-budgets named via dotted paths use their root name.
    """

    snapshot: dict[str, set[str]] = {}
    for produce_key in produces:
        root = str(produce_key).split(".", 1)[0]
        sub_budget = getattr(state, root, None)
        components = getattr(sub_budget, "components", None)
        if components is None:
            continue
        snapshot[root] = set(components)
    return snapshot


def _warn_if_components_dropped(
    instance_id: str,
    before: dict[str, set[str]],
    after: dict[str, set[str]],
) -> None:
    """Log a warning when a stage's apply() silently dropped a previously-
    accumulated component from a sub-budget it claimed to produce.
    """

    for root, before_names in before.items():
        after_names = after.get(root, set())
        dropped = before_names - after_names
        if dropped:
            logging.warning(
                "Stage %r dropped %d component(s) from %s after apply(): %s. "
                "This usually means the stage replaced the whole sub-budget "
                "instead of composing additively.",
                instance_id,
                len(dropped),
                root,
                sorted(dropped),
            )


def _config_hash(resolved_config: Any) -> str:
    explicit_hash = _config_value(resolved_config, "config_hash", default=None)
    if isinstance(explicit_hash, str) and explicit_hash:
        return explicit_hash
    # ``_canonical_config`` already produces deterministic nested
    # dict/tuple/str structures. Wrapping it in ``pickle.dumps`` (the
    # pre-fix path) re-introduced pickle-protocol-dependent bytes — the
    # exact drift the cache module's canonical encoder exists to avoid.
    # Code Review 2026-05-16 Issue 7.
    return canonical_digest(_canonical_config(resolved_config)).hex()


def _sweep_coord_tuple(resolved_config: Any) -> tuple[tuple[str, Any], ...]:
    value = _config_value(resolved_config, "sweep_coord_tuple", default=())
    return tuple((str(axis_name), axis_value) for axis_name, axis_value in value)


def _root_seed(resolved_config: Any) -> int:
    run_config = _config_value(resolved_config, "run", default={})
    return int(_config_value(run_config, "seed", default=0))


def _source_path(resolved_config: Any) -> Path | None:
    value = _config_value(resolved_config, "source_path", default=None)
    if value is None:
        return None
    return Path(value)


def _run_dir(resolved_config: Any) -> Path:
    """Resolve [output].dir, substituting template variables.

    Supported: ``{run.<key>}``, ``{run_hash}``, ``{config_hash}``,
    ``{timestamp}``. Without substitution, ``RunContext.run_dir`` would
    be the literal template string because the config-load run_hash
    includes the unsubstituted dir verbatim. Any stage that writes
    artifacts via ``context.run_dir`` would then write to a path
    containing literal ``{...}`` characters.
    """
    output_config = _config_value(resolved_config, "output", default={})
    raw = _config_value(output_config, "dir", default=".")
    if not isinstance(raw, str) or "{" not in raw:
        return Path(raw)
    run_section = _config_value(resolved_config, "run", default={})
    run_hash = _config_value(resolved_config, "run_hash", default="")
    config_hash = _config_value(resolved_config, "config_hash", default="")
    timestamp = _config_value(resolved_config, "timestamp", default="")
    try:
        formatted = raw.format(
            run=run_section,
            run_hash=run_hash,
            config_hash=config_hash,
            timestamp=timestamp,
        )
    except (KeyError, AttributeError, IndexError) as exc:
        # Code Review 2026-05-16 Issue 25: an unresolved template variable
        # used to be silently returned with literal ``{var}`` braces in
        # the path, so a typo in ``output.dir`` (e.g. ``{run.missing}``)
        # would silently corrupt the reproducibility-critical output
        # directory name without any error.
        raise NullsimConfigError(
            f"output.dir template contains an unresolved variable "
            f"({exc!r}); template = {raw!r}. Supported template fields: "
            "{run.<key>}, {run_hash}, {config_hash}, {timestamp}."
        ) from exc
    return Path(formatted)


def _cache_enabled(resolved_config: Any) -> bool:
    cache_config = _config_value(resolved_config, "cache", default={})
    return bool(_config_value(cache_config, "enabled", default=True))


def _cache_dir(resolved_config: Any) -> Path:
    cache_config = _config_value(resolved_config, "cache", default={})
    return Path(_config_value(cache_config, "dir", default=".nullsim_cache"))


def _cache_version(resolved_config: Any) -> int:
    cache_config = _config_value(resolved_config, "cache", default={})
    return int(_config_value(cache_config, "version", default=1))


def _config_value(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _canonical_config(config: Any) -> Any:
    if isinstance(config, Mapping):
        return tuple(
            (str(key), _canonical_config(value))
            for key, value in sorted(config.items(), key=lambda item: str(item[0]))
        )
    if isinstance(config, Path):
        return str(config)
    if isinstance(config, tuple | list):
        return tuple(_canonical_config(value) for value in config)
    if hasattr(config, "model_dump"):
        return _canonical_config(config.model_dump())
    if hasattr(config, "__dict__"):
        return _canonical_config(vars(config))
    return config
