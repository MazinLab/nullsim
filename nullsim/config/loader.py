"""TOML loading, preset merge, and validation for SPEC §7 and §6.4.

The loader parses user TOML, merges only the preset-bearing sections used by the
vertical slice, validates with pydantic, and attaches canonical ``config_hash``
and ``run_hash`` metadata for the pipeline/cache layers.
"""

from __future__ import annotations

import copy
import tomllib
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

import nullsim
from nullsim.config import presets
from nullsim.config.errors import NullsimConfigError
from nullsim.config.stage_materialize import materialize_pipeline_stages
from nullsim.config.hashes import (
    canonical_toml_dumps,
    compute_config_hash,
    compute_run_hash,
    runtime_python_version,
)
from nullsim.config.schema import ResolvedConfig, TopLevelConfig

_PRESET_SECTIONS = ("site", "telescope", "scene", "observation")


def load_config(path: Path | str) -> ResolvedConfig:
    """Load, preset-merge, materialize, validate, and hash a nullsim TOML config.

    Pre-fix, the public API path (``run(load_config(path))``) skipped
    ``materialize_pipeline_stages``, which is the step that injects
    top-level section defaults into stage params. Public-API users silently
    got each stage's own ``Params`` defaults (e.g.
    ``scene.point_source.star_jmag=5.0``) instead of the TOML's
    ``[scene.star].jmag``, while the CLI worked correctly because
    ``expand_sweeps`` materialized before resolving each cell. See Code
    Review 2026-05-16 Ultra-review High #2.

    Materialization is in-place + idempotent (``stage_materialize.py``
    skips fields already set), so the CLI's later call inside
    ``_config_for_coord_tuple`` is a safe no-op for the fields filled
    here.
    """

    config_path = Path(path)
    raw_config = _load_raw_config(config_path)

    merged_config = merge_presets(raw_config)
    materialize_pipeline_stages(merged_config)
    return resolve_config(
        merged_config,
        raw_config=raw_config,
        source_path=config_path,
        sweep_coord_tuple=(),
    )


def _load_raw_config(path: Path, seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    """Load a TOML config, resolving a single-parent ``extends`` chain.

    Merge semantics: ``[[pipeline.stages]]`` lists merge by stage id (see
    ``_merge_stage_lists``); every other list — including ``[[sweep]]`` —
    is replaced wholesale by the child (last writer wins). A child that
    declares any ``[[sweep]]`` block therefore drops ALL parent sweep
    axes and must restate the ones it wants to keep.
    """

    config_path = path.expanduser().resolve()
    if config_path in seen:
        chain = " -> ".join(str(p) for p in (*seen, config_path))
        raise NullsimConfigError(f"Config extends cycle detected: {chain}")
    with config_path.open("rb") as handle:
        raw_config = tomllib.load(handle)

    extends = raw_config.pop("extends", None)
    if extends is None:
        return raw_config
    if not isinstance(extends, str) or not extends:
        raise NullsimConfigError("Top-level extends must be a non-empty string path.")
    base_path = Path(extends)
    if not base_path.is_absolute():
        base_path = config_path.parent / base_path
    base_config = _load_raw_config(base_path, seen | frozenset({config_path}))
    return _deep_merge(base_config, raw_config)


def merge_presets(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep copy of ``raw_config`` with supported presets merged."""

    merged_config = copy.deepcopy(dict(raw_config))
    for section_name in _PRESET_SECTIONS:
        section_value = merged_config.get(section_name)
        if not isinstance(section_value, Mapping):
            continue
        preset_name = section_value.get("preset")
        if not isinstance(preset_name, str) or not preset_name:
            continue
        preset_data = presets.load_preset(preset_name, section=section_name)
        merged_config[section_name] = _deep_merge(preset_data, section_value)
    return merged_config


def resolve_config(
    config_data: Mapping[str, Any],
    *,
    raw_config: Mapping[str, Any] | None = None,
    source_path: Path | str | None = None,
    sweep_coord_tuple: tuple[tuple[str, Any], ...] = (),
    external_dep_hashes: tuple[str | bytes, ...] = (),
    package_version: str | None = None,
) -> ResolvedConfig:
    """Validate an already preset-merged config mapping and attach hashes."""

    try:
        top_level_config = TopLevelConfig.model_validate(config_data)
    except ValidationError as exc:
        source_label = f" {source_path}" if source_path is not None else ""
        raise NullsimConfigError(
            f"Invalid nullsim config{source_label}: {exc}"
        ) from exc

    # SPEC §9 in-tree extension loading is part of a later WP. Until then,
    # silently accepting `[run].extensions` would be the worst failure
    # mode — user copies a SPEC §7.1 example, sees no error, no effect.
    if top_level_config.run.extensions:
        raise NullsimConfigError(
            "[run].extensions is reserved for the SPEC §9 in-tree extension "
            "loader and is not yet wired up. Remove the field for now; "
            "tracking the implementation in a follow-up WP."
        )

    config_mapping = top_level_config.model_dump(mode="python", exclude_none=True)
    resolved_toml_text = canonical_toml_dumps(config_mapping)
    config_hash = compute_config_hash(resolved_toml_text)

    # Fold the [catalog.exoplanets] snapshot hash into ``run_hash`` so
    # the library-path identity matches the CLI/manifest. Pre-fix the
    # library always passed ``external_dep_hashes=()`` and the CLI then
    # overrode the manifest with the externals-aware value — so a user
    # calling ``run(load_config(path)).run_hash`` got a value that
    # disagreed with the on-disk manifest (Code Review 2026-05-16 Issue 4).
    # Stage-declared externals (``stage.external_dependencies()``) still
    # require building stage instances and remain a CLI-side fold; the
    # ``external_dep_hashes`` arg accepts caller-supplied additions.
    catalog_hash = _catalog_block_hash(raw_config or config_data, source_path)
    if catalog_hash is not None:
        external_dep_hashes = tuple(external_dep_hashes) + (catalog_hash,)

    run_hash = compute_run_hash(
        config_hash=config_hash,
        package_version=package_version or nullsim.__version__,
        external_deps_hashes_iterable=external_dep_hashes,
        runtime_python_version=runtime_python_version(),
    )
    resolved_payload = {
        **config_mapping,
        "raw_config": copy.deepcopy(dict(raw_config or config_data)),
        "source_path": Path(source_path) if source_path is not None else None,
        "resolved_toml_text": resolved_toml_text,
        "config_hash": config_hash,
        "run_hash": run_hash,
        # Wall-clock stamp for human-readable output directories
        # (``{timestamp}`` in ``output.dir``). Computed once at load time
        # so the CLI's ``_resolve_output_dir`` and the runner's
        # ``_run_dir`` resolve to the same path during one process.
        "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "sweep_coord_tuple": sweep_coord_tuple,
    }
    return ResolvedConfig.model_validate(resolved_payload)


def _deep_merge(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
    *,
    path: tuple[str, ...] = (),
) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, override_value in override.items():
        base_value = merged.get(key)
        if (
            key == "stages"
            and path == ("pipeline",)
            and isinstance(base_value, list)
            and isinstance(override_value, list)
        ):
            merged[key] = _merge_stage_lists(base_value, override_value)
            continue
        if isinstance(base_value, Mapping) and isinstance(override_value, Mapping):
            # TODO: If preset ownership broadens, restrict this recursive merge by
            # section family schema instead of applying it to every nested dict.
            merged[key] = _deep_merge(
                base_value,
                override_value,
                path=(*path, str(key)),
            )
        else:
            merged[key] = copy.deepcopy(override_value)
    return merged


def _merge_stage_lists(
    base_stages: list[Any],
    override_stages: list[Any],
) -> list[Any]:
    """Merge ``[[pipeline.stages]]`` blocks by stage id.

    An override stage with ``remove = true`` deletes the inherited stage. An
    override stage with ``replace = true`` or a changed ``type`` replaces the
    inherited block. Otherwise matching stages are deep-merged, which is useful
    for small parameter overrides on detector/background/sensitivity stages.
    """

    merged = [copy.deepcopy(stage) for stage in base_stages]
    for raw_override in override_stages:
        if not isinstance(raw_override, Mapping):
            merged.append(copy.deepcopy(raw_override))
            continue
        override = copy.deepcopy(dict(raw_override))
        stage_id = override.get("id")
        if not isinstance(stage_id, str) or not stage_id:
            merged.append(override)
            continue
        remove = bool(override.pop("remove", False))
        replace = bool(override.pop("replace", False))
        before = override.pop("before", None)
        after = override.pop("after", None)
        if before is not None and after is not None:
            raise NullsimConfigError(
                f"Stage override {stage_id!r} cannot specify both before and after."
            )
        match_index = next(
            (
                index
                for index, stage in enumerate(merged)
                if isinstance(stage, Mapping) and stage.get("id") == stage_id
            ),
            None,
        )
        if remove:
            if match_index is None:
                raise NullsimConfigError(
                    f"Stage override {stage_id!r} requested remove=true, "
                    "but no inherited stage with that id exists."
                )
            del merged[match_index]
            continue
        if match_index is None:
            if "type" not in override:
                # A typeless block that matches no inherited id is almost
                # always a typo'd parameter override. Pre-fix it was
                # appended as a new stage and failed much later in pydantic
                # with "type Field required" at a stage index — useless for
                # locating the typo (review 2026-06-10 Issue 29).
                inherited_ids = sorted(
                    str(stage.get("id"))
                    for stage in merged
                    if isinstance(stage, Mapping) and stage.get("id")
                )
                raise NullsimConfigError(
                    f"[[pipeline.stages]] override id={stage_id!r} matches no "
                    "inherited stage and declares no type. If this was meant "
                    "to override an inherited stage, fix the id; inherited "
                    f"ids: {', '.join(inherited_ids)}."
                )
            insert_index = _stage_insert_index(
                merged,
                before=before,
                after=after,
                stage_id=stage_id,
            )
            if insert_index is None:
                merged.append(override)
            else:
                merged.insert(insert_index, override)
            continue
        if before is not None or after is not None:
            raise NullsimConfigError(
                f"Stage override {stage_id!r} matches an inherited stage and "
                "cannot also specify before/after. Use replace=true to update "
                "the stage in place, or remove it and insert a new stage with "
                "a different id."
            )
        base_stage = merged[match_index]
        type_changed = (
            isinstance(base_stage, Mapping)
            and "type" in override
            and override.get("type") != base_stage.get("type")
        )
        if replace or type_changed or not isinstance(base_stage, Mapping):
            merged[match_index] = override
        else:
            merged[match_index] = _deep_merge(base_stage, override)
    return merged


def _stage_insert_index(
    stages: list[Any],
    *,
    before: Any,
    after: Any,
    stage_id: str,
) -> int | None:
    """Return an insertion index for a new inherited stage override."""

    target = before if before is not None else after
    if target is None:
        return None
    if not isinstance(target, str) or not target:
        raise NullsimConfigError(
            f"Stage override {stage_id!r} before/after must name a stage id."
        )
    match_index = next(
        (
            index
            for index, stage in enumerate(stages)
            if isinstance(stage, Mapping) and stage.get("id") == target
        ),
        None,
    )
    if match_index is None:
        raise NullsimConfigError(
            f"Stage override {stage_id!r} requested insertion relative to "
            f"unknown stage id {target!r}."
        )
    return match_index if before is not None else match_index + 1


def _catalog_block_hash(
    raw_config: Mapping[str, Any] | None,
    source_path: Path | str | None,
) -> str | None:
    """Return the catalog snapshot content_hash for ``run_hash`` mixing.

    Returns ``None`` if no ``[catalog.exoplanets]`` block is present.
    Malformed declared catalog blocks fail at load time so run identity
    cannot silently omit a user-specified external dependency.
    """

    from nullsim.config.catalog import parse_catalog_block

    if not isinstance(raw_config, Mapping):
        return None
    catalog_section = raw_config.get("catalog")
    if not isinstance(catalog_section, Mapping):
        return None
    catalog_block = catalog_section.get("exoplanets")
    if catalog_block is None:
        return None
    config_dir = Path(source_path).parent if source_path is not None else None
    try:
        catalog_config = parse_catalog_block(catalog_block, config_dir=config_dir)
    except (NullsimConfigError, ValidationError, ValueError, TypeError) as exc:
        raise NullsimConfigError(f"Invalid [catalog.exoplanets] block: {exc}") from exc
    return catalog_config.content_hash()
