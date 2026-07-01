"""Content-addressed stage cache using the conservative recipe in SPEC §6.4.

Threat model
------------
Cache entries are stored as ``pickle`` files. ``pickle.load`` is *arbitrary
code execution* — any byte stream a malicious actor can write into the
cache directory becomes RCE on the next ``cache.load(key)`` call.

This is acceptable when the cache directory has the same trust level as the
project source checkout (the default ``./.nullsim_cache`` does). It is
**not safe** when the cache is shared (HPC scratch, NFS, NFS-backed CI
artifacts, a checked-in ``.nullsim_cache/`` from an untrusted contributor).

Mitigations applied here:
  * cache keys must be 64-hex-character SHA-256 digests (validated by
    :func:`_validate_cache_key`), so an attacker cannot path-traverse out
    of ``cache_dir`` to overwrite arbitrary files;
  * :meth:`ContentCache.load` is wrapped in try/except — any unpickle
    failure raises :class:`CacheLoadError` and deletes the corrupted file
    so a subsequent run gets a clean miss instead of repeatedly crashing;
  * the loaded payload is shape-checked (must be a 2-tuple of
    ``(SimulationState, dict)``) before being returned.

These mitigations stop *malformed* and *path-traversal* attacks. They do
not stop a determined attacker who can write a valid pickle into a
correctly-named slot. Migration off pickle (e.g. ``.npz`` plus JSON, or
HMAC-signed pickle) is tracked as future work for any deployment where
the cache directory crosses a trust boundary.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import pickle
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any
from uuid import uuid4

import nullsim
from nullsim.config.hashes import (
    runtime_dependency_versions,
    runtime_python_version,
)
from nullsim.pipeline._canonical import canonical_digest as _canonical_digest
from nullsim.pipeline.stage import (
    ExternalDependency,
    Stage,
    external_dependency_digest,
)
from nullsim.pipeline.state import SimulationState

_CACHE_SUFFIX = ".pickle"


class CacheLoadError(RuntimeError):
    """Raised when a cache entry exists but cannot be safely loaded.

    The corresponding file has already been deleted by the time this is
    raised, so the caller can fall back to recomputing the entry from
    scratch.
    """


class ContentCache:
    """Disk cache for post-stage ``(SimulationState, diagnostics)`` payloads."""

    def __init__(self, cache_dir: Path | str = ".nullsim_cache", cache_version: int = 1) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_version = int(cache_version)

    def key_for(self, instance: Stage, state: SimulationState) -> str:
        """Return the conservative content key for ``instance`` on ``state``.

        Folds in:
          * canonical-encoded stage params,
          * the full input-state digest. The cached value is the full
            post-stage ``SimulationState`` (the runner adopts it wholesale on
            a hit), so the key must cover every budget of the input state —
            not only the stage's consumed slices. A consumed-slice key let a
            hit on a scene-independent stage (``atmosphere.kolmogorov``)
            replace the whole state with another sweep cell's, silently
            swapping ``scene.star_jmag`` (kilo_keck_6 J-mag sweep regression,
            2026-06-09),
          * the per-stage source digest: the shared core (pipeline, config,
            physics, catalogs, shared ``stages/_*`` helpers) plus this stage
            family's transitive import closure of other stage families; output
            and CLI edits never invalidate stage results. Physics→stages
            imports resolve toward inclusion: any family reachable from core
            code is folded into the core digest for every stage,
          * the stage class's qualname (so two classes defined in the
            same source file still get distinct keys),
          * the stage's instance identity (SPEC §6.1's
            ``transport.smf28`` two-instance pattern, SPEC §6.5's
            per-instance RNG derivation),
          * the RNG identity bytes — but ONLY for stages that draw from
            ``context.rng`` (``uses_rng``); RNG-independent stages omit
            it so an unrelated config edit reuses their cached result
            (Ultra-review High #1 still holds for the ``uses_rng`` stages),
          * the nullsim ``__version__``,
          * the Python runtime and key dependency versions,
          * the stage's external-dependencies digest,
          * the cache version.
        """

        digest = hashlib.sha256()
        digest.update(_canonical_digest(_stage_params(instance)))
        # The cached VALUE is the full post-stage state (runner.py adopts it
        # wholesale on a hit), so the key must digest the full input state —
        # not just the consumed slices. A consumed-slice key let a hit on a
        # stage that does not consume `scene` resurrect another sweep cell's
        # scene budget (different star_jmag), poisoning every downstream
        # stage: the 2026-06-09 kilo_keck_6 missing char-curve regression.
        # The full digest also covers opportunistic `cache_key_extras` reads
        # and the run-global [grid] wavelength grid carried on the budgets.
        digest.update(state.content_digest())
        digest.update(_stage_source_digest(getattr(instance, "family", None)))
        digest.update(type(instance).__qualname__.encode("utf-8"))
        digest.update(_instance_identity(instance).encode("utf-8"))
        # The spec-digest-seeded RNG identity only affects the RESULT of stages
        # that actually draw from context.rng (uses_rng=True). For every other
        # stage the result is RNG-independent, so omitting it lets an
        # unrelated config edit reuse the cached result. Correctness then rests
        # on params + the full input-state digest + the source digest fully
        # capturing the stage's inputs.
        if getattr(instance, "uses_rng", False):
            digest.update(_rng_identity_bytes(instance))
        else:
            digest.update(b"<config-independent-rng>")
        digest.update(nullsim.__version__.encode("utf-8"))
        digest.update(runtime_python_version().encode("utf-8"))
        digest.update(_canonical_digest(runtime_dependency_versions()))
        digest.update(_external_dependencies_digest(instance.external_dependencies()))
        digest.update(str(self.cache_version).encode("utf-8"))
        return digest.hexdigest()

    def has(self, key: str) -> bool:
        """Return whether a cache entry exists for ``key``."""

        return self._path_for(key).is_file()

    def load(self, key: str) -> tuple[SimulationState, dict[str, Any]]:
        """Load a cached ``(state, diagnostics)`` tuple for ``key``.

        Wraps ``pickle.load`` in a recovery shell: any deserialization
        failure (corrupted file, schema drift across versions, malformed
        payload) raises :class:`CacheLoadError` after deleting the bad
        entry so a subsequent run sees a clean miss. See module docstring
        for the threat model — pickle is RCE on a shared cache.
        """

        cache_path = self._path_for(key)
        try:
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception as exc:
            # Code Review 2026-05-16 Issue 21: pre-fix the catch listed
            # specific exception types (UnpicklingError, EOFError,
            # ValueError, TypeError, AttributeError, ModuleNotFoundError)
            # which missed ImportError (parent of ModuleNotFoundError)
            # and KeyError from ``pickle.Unpickler.find_class``. Since
            # this is a recovery shell that discards the cache entry
            # either way, widen to ``Exception`` and log so failures are
            # visible.
            logging.warning(
                "Cache entry %r failed to unpickle (%s: %s); discarding.",
                key, type(exc).__name__, exc,
            )
            self._discard(cache_path)
            raise CacheLoadError(
                f"Cache entry {key!r} unpicklable ({type(exc).__name__}: "
                f"{exc}); discarded."
            ) from exc

        if (
            not isinstance(payload, tuple)
            or len(payload) != 2
            or not isinstance(payload[0], SimulationState)
            or not isinstance(payload[1], dict)
        ):
            self._discard(cache_path)
            raise CacheLoadError(
                f"Cache entry {key!r} has unexpected shape "
                f"({type(payload).__name__}); discarded."
            )

        return payload[0], payload[1]

    @staticmethod
    def _discard(cache_path: Path) -> None:
        """Delete a corrupted cache entry; never raises."""
        with suppress(OSError):
            cache_path.unlink(missing_ok=True)

    def store(
        self,
        key: str,
        state: SimulationState,
        diagnostics: dict[str, Any],
    ) -> None:
        """Store a post-stage ``(state, diagnostics)`` tuple under ``key``."""

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._path_for(key)
        # Use a unique temp file per writer. Under pytest-xdist and
        # multi-process sweeps, two workers can legitimately compute the
        # same stage key at the same time. A fixed ``<key>.pickle.tmp`` path
        # lets one worker atomically replace the temp file before another
        # worker reaches its own replace call.
        temp_path = cache_path.with_name(f"{cache_path.name}.{uuid4().hex}.tmp")
        try:
            with temp_path.open("wb") as handle:
                pickle.dump((state, diagnostics), handle, protocol=5)
            temp_path.replace(cache_path)
        finally:
            with suppress(OSError):
                temp_path.unlink(missing_ok=True)

    def clear(self) -> None:
        """Remove all entries from this cache directory."""

        if not self.cache_dir.exists():
            return
        for cache_file in self.cache_dir.glob(f"*{_CACHE_SUFFIX}"):
            cache_file.unlink()

    def _path_for(self, key: str) -> Path:
        _validate_cache_key(key)
        return self.cache_dir / f"{key}{_CACHE_SUFFIX}"


def _stage_params(instance: Stage) -> Any:
    return getattr(instance, "params", {})


def _instance_identity(instance: Stage) -> str:
    """Return the stage's per-instance identity for the cache key.

    Prefer the RunContext-derived ``instance_id`` (matches the RNG
    derivation in SPEC §6.5). Fall back to ``type_name`` if the
    instance has no context (defensive — shouldn't happen in normal
    pipeline runs).
    """
    context = getattr(instance, "context", None)
    instance_id = getattr(context, "instance_id", None)
    if isinstance(instance_id, str) and instance_id:
        return instance_id
    return getattr(instance, "type_name", type(instance).__name__)


def _rng_identity_bytes(instance: Stage) -> bytes:
    """Return RNG-identity bytes for the cache key.

    The per-instance RNG is derived from ``(root_seed, rng_identity,
    sweep_coord_tuple, instance_id)`` where ``rng_identity`` is the
    stage's own spec digest (review 2026-06-10 Issue 7 — seeding from
    the full config_hash made unrelated config edits change every
    stochastic draw and bust every downstream cache entry). The cache
    key must include the same tuple — otherwise a stochastic stage's
    second run with a different seed (or different sweep cell, or
    different stage spec) would cache-hit the first run's entry and
    silently reuse the wrong RNG path. Ultra-review High #1.

    Contexts built by hand without ``rng_identity`` (older tests, direct
    RunContext construction) fall back to ``config_hash``, which is
    strictly more conservative. The function raises ``ValueError`` on
    missing components instead of producing degenerate identity bytes.
    The single legitimate "no context" path (low-level unit tests
    instantiating a stage outside the runner) still returns the
    ``<no-context>`` sentinel.
    """

    context = getattr(instance, "context", None)
    if context is None:
        return b"<no-context>"

    root_seed = getattr(context, "root_seed", None)
    rng_identity = getattr(context, "rng_identity", None)
    if not (isinstance(rng_identity, str) and rng_identity):
        rng_identity = getattr(context, "config_hash", None)
    sweep_coord_tuple = getattr(context, "sweep_coord_tuple", None)

    if root_seed is None:
        raise ValueError(
            f"{_instance_identity(instance)}: cache key identity is "
            "missing root_seed (RunContext is incomplete). The RNG "
            "derivation requires (root_seed, rng_identity, "
            "sweep_coord_tuple, instance_id) — None components would "
            "collide across runs."
        )
    if rng_identity is None:
        raise ValueError(
            f"{_instance_identity(instance)}: cache key identity is "
            "missing both rng_identity and config_hash (RunContext is "
            "incomplete)."
        )
    if sweep_coord_tuple is None:
        # Empty tuple is fine — that means a single-cell run; None is
        # the bug.
        raise ValueError(
            f"{_instance_identity(instance)}: cache key identity is "
            "missing sweep_coord_tuple (RunContext is incomplete)."
        )

    parts: list[bytes] = []
    parts.append(f"root_seed:{root_seed!r}".encode())
    parts.append(b"\0")
    parts.append(f"rng_identity:{rng_identity!r}".encode())
    parts.append(b"\0")
    parts.append(_canonical_digest(tuple(sweep_coord_tuple)))
    return b"".join(parts)





# Source subtrees that never participate in stage execution: the pure output
# writers (``outputs/`` — plots, tables, the paper bundle, sweep_results, and
# the RunRecord/RunCollection containers) and the CLI entry points (``cli.py``,
# ``__main__.py``) all run *after* the pipeline has produced ``SimulationState``
# and never feed back into a stage's result. Editing them must not invalidate
# the (often multi-hour) cached stage optimizations. Everything else —
# pipeline, config, physics, stages, catalogs — is hashed conservatively, so
# any edit that could change a stage result still busts the cache.
_CACHE_IRRELEVANT_PREFIXES: tuple[str, ...] = ("outputs/",)
_CACHE_IRRELEVANT_FILES: frozenset[str] = frozenset({"cli.py", "__main__.py"})


def _affects_stage_state(rel_posix: str) -> bool:
    """Whether a package-relative ``*.py`` path can affect a stage's result.

    Used by BOTH the source-digest computation and the mtime memoization scan
    in :func:`_max_package_mtime_ns`; they must apply the identical filter, or
    the memoization could fail to notice an edit to a tracked file and serve a
    stale digest.
    """

    if rel_posix in _CACHE_IRRELEVANT_FILES:
        return False
    return not any(
        rel_posix.startswith(prefix) for prefix in _CACHE_IRRELEVANT_PREFIXES
    )


# --- Per-stage-family source digesting (Tier 2 granularity) ----------------
#
# A stage's result depends only on shared core code (pipeline, config, physics,
# catalogs, the shared ``stages/_*.py`` helpers) plus its own stage family and
# any other stage family that family imports. Hashing only that closure -- not
# every ``*.py`` under ``nullsim/`` -- lets edits to an unrelated stage family
# reuse this stage's cached (often multi-hour) optimizations. The det path
# (``chip_optimization``, ``sensitivity``) imports no part of the char path, so
# char-code edits no longer bust the det cache and vice versa.
#
# Safety: the digest only narrows along *proven* absolute-import edges. If a
# family file fails to parse or uses an unresolved relative import, that family
# degrades to "depends on every family"; if the stage exposes no known family
# the digest includes every family. Over-inclusion costs a cold re-run;
# under-inclusion would silently serve stale science, so every ambiguity
# resolves toward inclusion. Core code (pipeline, config, physics, catalogs,
# and shared stage helpers) is also scanned for absolute imports into stage
# families; any family reachable from core code is folded into every stage's
# digest because editing that family can change shared-core behavior.


def _stage_family_of_source(rel_posix: str) -> str | None:
    """Stage-family name for a ``stages/<family>/...`` path, else ``None``.

    Files directly under ``stages/`` (e.g. ``stages/_utils.py``,
    ``stages/__init__.py``) are shared core, not a family, and return ``None``.
    """

    parts = rel_posix.split("/")
    if len(parts) >= 3 and parts[0] == "stages":
        return parts[1]
    return None


def _imported_stage_families(path: Path) -> set[str] | None:
    """Stage families imported by ``path`` via ``nullsim.stages.<family>``.

    Uses the AST so only real import statements count, never docstring or
    comment mentions. Returns ``None`` if the file cannot be parsed or uses a
    relative import we will not resolve, signalling the caller to treat that
    family as depending on every family (conservative).
    """

    try:
        tree = ast.parse(path.read_bytes())
    except (SyntaxError, ValueError, OSError):
        return None
    module_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level > 0:
                return None
            if node.module:
                module_names.append(node.module)
        elif isinstance(node, ast.Import):
            module_names.extend(alias.name for alias in node.names)
    families: set[str] = set()
    for module in module_names:
        parts = module.split(".")
        if len(parts) >= 3 and parts[0] == "nullsim" and parts[1] == "stages":
            families.add(parts[2])
    return families


def _compute_stage_family_closures() -> dict[str, frozenset[str]] | None:
    """Map each stage family to the transitive set of families it depends on.

    Returns ``None`` only when the ``stages`` package is missing; per-file
    analysis failures degrade that single family to "depends on every family"
    rather than disabling the optimisation everywhere.
    """

    nullsim_root = Path(nullsim.__file__).parent
    stages_root = nullsim_root / "stages"
    if not stages_root.is_dir():
        return None
    families = {
        child.name
        for child in stages_root.iterdir()
        if child.is_dir() and not child.name.startswith("__")
    }
    direct: dict[str, set[str]] = {family: set() for family in families}
    for path in stages_root.rglob("*.py"):
        rel = path.relative_to(nullsim_root).as_posix()
        family = _stage_family_of_source(rel)
        if family is None or family not in families:
            continue
        imported = _imported_stage_families(path)
        if imported is None:
            direct[family] = set(families)
            continue
        direct[family].update(imported & families)
    closures: dict[str, frozenset[str]] = {}
    for family in families:
        seen: set[str] = set()
        stack = [family]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(direct.get(current, set()) - seen)
        closures[family] = frozenset(seen)
    return closures


_CORE_REACHABLE_STAGE_FAMILIES: frozenset[str] | None = None
_CORE_REACHABLE_STAGE_FAMILIES_MTIME_NS: int | None = None


def _core_reachable_stage_families() -> frozenset[str]:
    """Stage families imported by cache-relevant non-stage package code."""

    global _CORE_REACHABLE_STAGE_FAMILIES, _CORE_REACHABLE_STAGE_FAMILIES_MTIME_NS
    latest_mtime_ns = _max_package_mtime_ns()
    if (
        _CORE_REACHABLE_STAGE_FAMILIES is None
        or _CORE_REACHABLE_STAGE_FAMILIES_MTIME_NS != latest_mtime_ns
    ):
        _CORE_REACHABLE_STAGE_FAMILIES = _compute_core_reachable_stage_families()
        _CORE_REACHABLE_STAGE_FAMILIES_MTIME_NS = latest_mtime_ns
    return _CORE_REACHABLE_STAGE_FAMILIES


def _compute_core_reachable_stage_families() -> frozenset[str]:
    nullsim_root = Path(nullsim.__file__).parent
    stages_root = nullsim_root / "stages"
    if not stages_root.is_dir():
        return frozenset()
    families = {
        child.name
        for child in stages_root.iterdir()
        if child.is_dir() and not child.name.startswith("__")
    }
    reachable: set[str] = set()
    for path in nullsim_root.rglob("*.py"):
        rel = path.relative_to(nullsim_root).as_posix()
        if rel.startswith("stages/") or not _affects_stage_state(rel):
            continue
        imported = _imported_stage_families(path)
        if imported is None:
            return frozenset(families)
        reachable.update(imported & families)
    return frozenset(reachable)


def _compute_source_digests() -> (
    tuple[bytes, dict[str, bytes], dict[str, frozenset[str]] | None]
):
    """Compute the shared-core digest, per-family digests, and family closures.

    Walks the state-relevant ``*.py`` tree once (``_affects_stage_state`` still
    excludes output and CLI code), bucketing each file into the shared core or
    its stage family.
    """

    nullsim_root = Path(nullsim.__file__).parent
    core = hashlib.sha256()
    family_hashers: dict[str, Any] = {}
    for path in sorted(nullsim_root.rglob("*.py")):
        rel = path.relative_to(nullsim_root).as_posix()
        if not _affects_stage_state(rel):
            continue
        family = _stage_family_of_source(rel)
        if family is None:
            target = core
        else:
            target = family_hashers.setdefault(family, hashlib.sha256())
        target.update(rel.encode("utf-8"))
        target.update(b"\0")
        target.update(path.read_bytes())
        target.update(b"\0")
    family_digests = {
        family: hasher.digest() for family, hasher in family_hashers.items()
    }
    return core.digest(), family_digests, _compute_stage_family_closures()


_SOURCE_DIGESTS: (
    tuple[bytes, dict[str, bytes], dict[str, frozenset[str]] | None] | None
) = None
_SOURCE_DIGESTS_MTIME_NS: int | None = None


def _max_package_mtime_ns() -> int:
    """Max mtime across cache-relevant .py files in the package.

    Reading mtimes is ~100x cheaper than re-hashing every file. The
    full hash is only recomputed when *any* tracked file changes since
    the last memoization. This lets long-running interactive sessions
    (notebooks, watchers) see source edits invalidate the cache instead
    of using a stale, hours-old digest.

    Applies the same :func:`_affects_stage_state` filter as the digest so the
    two never drift: a stricter filter here would miss edits to tracked files
    and keep serving a stale digest.
    """

    nullsim_root = Path(nullsim.__file__).parent
    latest = 0
    for path in nullsim_root.rglob("*.py"):
        rel = path.relative_to(nullsim_root).as_posix()
        if not _affects_stage_state(rel):
            continue
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            continue
        if mtime > latest:
            latest = mtime
    return latest


def _source_digests() -> (
    tuple[bytes, dict[str, bytes], dict[str, frozenset[str]] | None]
):
    """Return the memoized (core digest, per-family digests, family closures).

    Recomputed when any tracked *.py file's mtime changes, so an editor save
    during a long interactive session invalidates the cache instead of serving
    a stale digest for the rest of the process.
    """

    global _SOURCE_DIGESTS, _SOURCE_DIGESTS_MTIME_NS
    latest_mtime_ns = _max_package_mtime_ns()
    if _SOURCE_DIGESTS is None or _SOURCE_DIGESTS_MTIME_NS != latest_mtime_ns:
        _SOURCE_DIGESTS = _compute_source_digests()
        _SOURCE_DIGESTS_MTIME_NS = latest_mtime_ns
    return _SOURCE_DIGESTS


def _stage_source_digest(family: str | None) -> bytes:
    """SHA-256 over the shared core plus the import closure of ``family``.

    Folded into :meth:`ContentCache.key_for`. Falls back to including every
    stage family (== the conservative whole-package digest) when the family is
    unknown or the closures are unavailable, so an unresolved dependency can
    only over-invalidate, never serve stale results.
    """

    core_digest, family_digests, closures = _source_digests()
    digest = hashlib.sha256()
    digest.update(core_digest)
    if family is None or closures is None or family not in closures:
        for family_name in sorted(family_digests):
            digest.update(family_digests[family_name])
        return digest.digest()
    included_families = set(_core_reachable_stage_families())
    included_families.update(closures[family])
    for family_name in sorted(included_families):
        family_digest = family_digests.get(family_name)
        if family_digest is not None:
            digest.update(family_digest)
    return digest.digest()


def _external_dependencies_digest(dependencies: Sequence[ExternalDependency]) -> bytes:
    digest = hashlib.sha256()
    for dependency in dependencies:
        digest.update(_dependency_digest(dependency))
        digest.update(b"\0")
    return digest.digest()


def _dependency_digest(dependency: ExternalDependency) -> bytes:
    return external_dependency_digest(dependency)


def _validate_cache_key(key: str) -> None:
    if len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
        raise ValueError("Cache keys must be sha256 hex digests.")
