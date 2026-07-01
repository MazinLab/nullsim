"""Run-identity hashing for SPEC §6.4 using SPEC §7 resolved TOML text.

``config_hash`` is derived only from canonical resolved TOML. ``run_hash`` adds
the package version, runtime identity, key dependency versions, source-code
identity, and declared external dependency digests so output directories
distinguish what was produced.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

KEY_DEPENDENCY_NAMES = ("numpy", "scipy", "pyarrow", "torch")

_SOURCE_IDENTITY_UNAVAILABLE = {
    "source_control": "unavailable",
    "source_commit": None,
    "source_dirty": None,
    "source_digest": "source-unavailable",
}


def canonical_toml_dumps(config: Mapping[str, Any] | Any) -> str:
    """Serialize a resolved config mapping into deterministic TOML text."""

    plain_config = _to_plain(config)
    if not isinstance(plain_config, Mapping):
        raise TypeError("canonical_toml_dumps expects a mapping or config model.")

    lines: list[str] = []
    _write_table(lines, (), plain_config)
    return f"{'\n'.join(lines).rstrip()}\n"


def compute_config_hash(resolved_config_toml_text: str) -> str:
    """Return sha256 hex over canonical resolved TOML text."""

    return hashlib.sha256(resolved_config_toml_text.encode("utf-8")).hexdigest()


def compute_run_hash(
    config_hash: str,
    package_version: str,
    external_deps_hashes_iterable: Iterable[str | bytes],
    runtime_python_version: str,
    key_dep_versions: Mapping[str, str] | None = None,
    source_digest: str | None = None,
) -> str:
    """Return sha256 hex over the run identity recipe from SPEC §6.4."""

    dependency_versions = key_dep_versions or runtime_dependency_versions()
    resolved_source_digest = source_digest or source_identity()["source_digest"]
    digest = hashlib.sha256()
    _update_labeled(digest, "config_hash", config_hash)
    _update_labeled(digest, "package_version", package_version)
    _update_labeled(digest, "runtime_python_version", runtime_python_version)
    _update_labeled(digest, "source_digest", resolved_source_digest)
    for dependency_name, dependency_version in sorted(dependency_versions.items()):
        _update_labeled(digest, f"dependency:{dependency_name}", dependency_version)
    for external_digest in sorted(_coerce_digest_text(item) for item in external_deps_hashes_iterable):
        _update_labeled(digest, "external_dependency", external_digest)
    return digest.hexdigest()


def source_identity(repo_root: Path | None = None) -> dict[str, Any]:
    """Return the git source identity that enters ``run_hash``.

    The identity is intentionally based on the committed tree plus tracked
    uncommitted diffs. Untracked local scratch files do not affect the digest,
    which keeps run hashes stable in common exploratory workflows while still
    distinguishing results produced by different source code.
    """

    root = repo_root or Path(__file__).resolve().parents[2]
    try:
        commit = _git_stdout(root, "rev-parse", "--verify", "HEAD")
        status = _git_stdout(root, "status", "--porcelain", "--untracked-files=no")
        diff = _git_bytes(root, "diff", "--binary", "HEAD", "--")
    except (OSError, subprocess.CalledProcessError):
        return dict(_SOURCE_IDENTITY_UNAVAILABLE)

    digest = hashlib.sha256()
    _update_labeled(digest, "git_commit", commit)
    _update_labeled(digest, "tracked_status", status)
    digest.update(b"tracked_diff\0")
    digest.update(diff)
    return {
        "source_control": "git",
        "source_commit": commit,
        "source_dirty": bool(status.strip()),
        "source_digest": digest.hexdigest(),
    }


def runtime_python_version() -> str:
    """Return ``"<implementation> <version>"`` for run hashing.

    Pre-fix this returned ``sys.version``, which embeds the compile date,
    OS name, and compiler banner (e.g. ``"3.13.0 (main, Oct 7 2024,
    18:18:54) [Clang 16.0.0]"``). Two machines running the same Python
    point release would produce different ``run_hash`` values for
    byte-identical inputs. SPEC §6.4 calls for "runtime Python version,"
    not the build banner. Code Review 2026-05-16 Issue 5.
    """

    return f"{platform.python_implementation()} {platform.python_version()}"


def runtime_dependency_versions(
    package_names: Iterable[str] = KEY_DEPENDENCY_NAMES,
) -> dict[str, str]:
    """Return installed versions for dependency names that enter ``run_hash``."""

    versions: dict[str, str] = {}
    for package_name in package_names:
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = "<missing>"
    return versions


def _git_stdout(root: Path, *args: str) -> str:
    return _git_bytes(root, *args).decode("utf-8", errors="replace").strip()


def _git_bytes(root: Path, *args: str) -> bytes:
    return subprocess.check_output(
        ("git", "-C", str(root), *args),
        stderr=subprocess.DEVNULL,
    )


def write_resolved_config_toml(config: Mapping[str, Any] | Any, path: Path | str) -> str:
    """Write deterministic resolved TOML and return the written text."""

    toml_text = canonical_toml_dumps(config)
    Path(path).write_text(toml_text, encoding="utf-8")
    return toml_text


def _write_table(lines: list[str], path: tuple[str, ...], table: Mapping[str, Any]) -> None:
    scalar_items: list[tuple[str, Any]] = []
    table_items: list[tuple[str, Mapping[str, Any]]] = []
    array_table_items: list[tuple[str, list[Mapping[str, Any]]]] = []

    for key, value in _ordered_items(path, table):
        plain_value = _to_plain(value)
        if _is_array_of_tables(plain_value):
            array_table_items.append((key, list(plain_value)))
        elif isinstance(plain_value, Mapping):
            table_items.append((key, plain_value))
        elif plain_value is not None:
            scalar_items.append((key, plain_value))

    if path and (scalar_items or table_items or array_table_items):
        _append_header(lines, f"[{'.'.join(path)}]")
    for key, value in scalar_items:
        lines.append(f"{key} = {_format_toml_value(value)}")
    for key, nested_table in table_items:
        _write_table(lines, (*path, key), nested_table)
    for key, array_items in array_table_items:
        _write_array_table(lines, (*path, key), array_items)


def _write_array_table(
    lines: list[str],
    path: tuple[str, ...],
    items: list[Mapping[str, Any]],
) -> None:
    for item in items:
        _append_header(lines, f"[[{'.'.join(path)}]]")
        for key, value in _ordered_items(path, item):
            if value is not None:
                lines.append(f"{key} = {_format_toml_value(value)}")


def _append_header(lines: list[str], header: str) -> None:
    if lines:
        lines.append("")
    lines.append(header)


def _ordered_items(path: tuple[str, ...], table: Mapping[str, Any]) -> list[tuple[str, Any]]:
    preferred = _preferred_order(path)
    ordered_keys = [key for key in preferred if key in table]
    ordered_keys.extend(sorted(str(key) for key in table if str(key) not in ordered_keys))
    return [(key, table[key]) for key in ordered_keys]


def _preferred_order(path: tuple[str, ...]) -> tuple[str, ...]:
    return {
        (): (
            "run",
            "grid",
            "site",
            "telescope",
            "scene",
            "observation",
            "pipeline",
            "sweep",
            "output",
            "cache",
        ),
        ("run",): ("name", "description", "seed", "extensions"),
        ("grid",): (
            "wavelength_center_um",
            "wavelength_bandwidth_um",
            "n_wavelength_bins",
        ),
        ("scene",): ("star", "planet", "exozodi"),
        ("scene", "star"): (
            "target",
            "jmag",
            "angular_diameter_mas",
            "distance_pc",
        ),
        ("scene", "planet"): ("contrast", "separation_mas", "position_angle_deg"),
        ("scene", "exozodi"): ("level_zodi",),
        ("observation",): ("hour_angle_h", "integration_time_s"),
        ("pipeline",): ("stages",),
        ("pipeline", "stages"): ("id", "type"),
        ("sweep",): ("param", "mode", "values", "range", "linspace", "logspace"),
        ("output",): ("dir", "plots", "tables", "formats", "sweep_table"),
        ("output", "formats"): ("plots", "tables"),
        ("cache",): ("enabled", "dir", "version"),
    }.get(path, ())


def _format_toml_value(value: Any) -> str:
    plain_value = _to_plain(value)
    if isinstance(plain_value, str):
        return json.dumps(plain_value)
    if isinstance(plain_value, bool):
        return "true" if plain_value else "false"
    if isinstance(plain_value, int):
        return str(plain_value)
    if isinstance(plain_value, float):
        return format(plain_value, ".17g")
    if isinstance(plain_value, Mapping):
        return _format_inline_table(plain_value)
    if isinstance(plain_value, tuple | list):
        return "[" + ", ".join(_format_toml_value(item) for item in plain_value) + "]"
    raise TypeError(f"Cannot serialize {type(value).__name__} as TOML.")


def _format_inline_table(table: Mapping[str, Any]) -> str:
    parts = [
        f"{key} = {_format_toml_value(value)}"
        for key, value in _ordered_items((), table)
        if value is not None
    ]
    return "{ " + ", ".join(parts) + " }"


def _is_array_of_tables(value: Any) -> bool:
    return isinstance(value, tuple | list) and bool(value) and all(
        isinstance(item, Mapping) for item in value
    )


def _to_plain(value: Any) -> Any:
    if hasattr(value, "config_mapping"):
        return value.config_mapping()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="python", exclude_none=True)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item_value) for key, item_value in value.items()}
    if isinstance(value, tuple | list):
        return [_to_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _update_labeled(digest: Any, label: str, value: str) -> None:
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(value.encode("utf-8"))
    digest.update(b"\0")


def _coerce_digest_text(value: str | bytes) -> str:
    return value.hex() if isinstance(value, bytes) else str(value)
