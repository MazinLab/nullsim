"""Run manifest writer (SPEC §7.5).

``manifest.json`` records the identity and runtime context of one run so
output directories are self-describing. Fields:

* ``config_hash`` / ``run_hash`` — SPEC §6.4 identity hashes.
* ``package_version`` — installed ``nullsim`` version via ``importlib.metadata``.
* ``dependency_versions`` — key dependency versions that enter ``run_hash``.
* ``source_identity`` — git commit/digest identity that enters ``run_hash``.
* ``host`` — Python implementation, version, and platform identifier.
* ``rng_root_seed`` — root seed for SPEC §6.5 RNG derivation.
* ``timings`` — per-instance wall time and cache-hit flag.
* ``cache`` — hit/miss counts.
* ``external_dependencies`` — declared external resources per instance.
"""

from __future__ import annotations

import json
import platform
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from nullsim.config.hashes import source_identity
from nullsim.pipeline.runner import RunResult, StageTiming
from nullsim.pipeline.stage import ExternalDependency, Stage

MANIFEST_DEPENDENCY_NAMES: tuple[str, ...] = (
    "numpy",
    "scipy",
    "pandas",
    "pyarrow",
    "pydantic",
    "matplotlib",
)


def write_manifest(
    output_path: Path | str,
    *,
    config_hash: str,
    run_hash: str,
    rng_root_seed: int,
    result: RunResult,
    results: Sequence[RunResult] | None = None,
    instances: Sequence[Stage] | Sequence[Sequence[Stage]] = (),
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``manifest.json`` describing this run's identity and runtime.

    Args:
        output_path: Destination file (typically ``<run_dir>/manifest.json``).
        config_hash: SPEC §6.4 config-only hash.
        run_hash: SPEC §6.4 run-identity hash (names output directory).
        rng_root_seed: Root seed for SPEC §6.5 RNG derivation.
        result: Completed primary :class:`RunResult` with timings and cache counts.
        results: Optional full set of run results for aggregate cache counts
            across sweep cells. Defaults to ``(result,)``.
        instances: Stage instances in declared order, or one such sequence per
            sweep cell, used to record each stage's declared
            ``external_dependencies()``.
        extra: Optional additional top-level keys (e.g. sweep manifest data).

    Returns:
        The path to the written manifest.
    """

    manifest = build_manifest(
        config_hash=config_hash,
        run_hash=run_hash,
        rng_root_seed=rng_root_seed,
        result=result,
        results=results,
        instances=instances,
        extra=extra,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    return destination


def build_manifest(
    *,
    config_hash: str,
    run_hash: str,
    rng_root_seed: int,
    result: RunResult,
    results: Sequence[RunResult] | None = None,
    instances: Sequence[Stage] | Sequence[Sequence[Stage]] = (),
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the manifest payload without writing it to disk."""

    timings_payload = [_timing_payload(timing) for timing in result.timings]
    cache_results = tuple(results) if results is not None else (result,)
    stage_count = sum(len(cache_result.timings) for cache_result in cache_results)
    hits = sum(
        1
        for cache_result in cache_results
        for timing in cache_result.timings
        if timing.cache_hit
    )
    misses = stage_count - hits
    manifest: dict[str, Any] = {
        "config_hash": config_hash,
        "run_hash": run_hash,
        "package_version": _package_version("nullsim"),
        "dependency_versions": _dependency_versions(MANIFEST_DEPENDENCY_NAMES),
        "source_identity": source_identity(),
        "host": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "rng_root_seed": int(rng_root_seed),
        "timings": timings_payload,
        "cache": {
            "hits": hits,
            "misses": misses,
            "stage_count": stage_count,
        },
        "external_dependencies": _external_dependencies_payload(instances),
    }
    if extra:
        for key, value in extra.items():
            manifest[key] = value
    return manifest


def _timing_payload(timing: StageTiming) -> dict[str, Any]:
    return {
        "instance_id": timing.instance_id,
        "elapsed_s": float(timing.elapsed_s),
        "cache_hit": bool(timing.cache_hit),
    }


def _external_dependencies_payload(
    instances: Sequence[Stage] | Sequence[Sequence[Stage]],
) -> list[dict[str, Any]]:
    instance_groups = _instance_groups(instances)
    if len(instance_groups) > 1:
        payload: list[dict[str, Any]] = []
        for cell_index, group in enumerate(instance_groups):
            for entry in _external_dependencies_payload_for_group(group):
                payload.append({"cell_index": cell_index, **entry})
        return payload
    if not instance_groups:
        return []
    return _external_dependencies_payload_for_group(instance_groups[0])


def _instance_groups(
    instances: Sequence[Stage] | Sequence[Sequence[Stage]],
) -> tuple[Sequence[Stage], ...]:
    if not instances:
        return ()
    first = instances[0]
    if isinstance(first, (list, tuple)):
        return tuple(instances)  # type: ignore[return-value]
    return (instances,)  # type: ignore[return-value]


def _external_dependencies_payload_for_group(
    instances: Sequence[Stage],
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for instance in instances:
        instance_id = _instance_id(instance)
        try:
            declared = list(instance.external_dependencies() or [])
        except Exception as exc:  # noqa: BLE001 — manifest must not crash a run
            payload.append(
                {
                    "instance_id": instance_id,
                    "error": f"external_dependencies() raised {type(exc).__name__}: {exc}",
                }
            )
            continue
        payload.append(
            {
                "instance_id": instance_id,
                "dependencies": [
                    _dependency_payload(dependency) for dependency in declared
                ],
            }
        )
    return payload


def _dependency_payload(dependency: ExternalDependency | Any) -> dict[str, Any]:
    if isinstance(dependency, ExternalDependency):
        return {
            "source": dependency.source,
            "path": None if dependency.path is None else str(dependency.path),
            "content_hash": (
                None
                if dependency.content_hash is None
                else (
                    dependency.content_hash.hex()
                    if isinstance(dependency.content_hash, bytes)
                    else dependency.content_hash
                )
            ),
        }
    return {"source": repr(dependency)}


def _instance_id(instance: Stage) -> str:
    context = getattr(instance, "context", None)
    instance_id = getattr(context, "instance_id", None)
    if isinstance(instance_id, str) and instance_id:
        return instance_id
    return getattr(instance, "type_name", type(instance).__name__)


def _dependency_versions(names: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "<missing>"
    return versions


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "<missing>"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="python")
    return repr(value)


__all__ = [
    "MANIFEST_DEPENDENCY_NAMES",
    "build_manifest",
    "write_manifest",
]
