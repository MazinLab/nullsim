"""Stage protocol and base implementation for SPEC §6.1 pipeline stages."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol
from collections.abc import Mapping

from nullsim.pipeline.context import RunContext
from nullsim.pipeline.state import SimulationState

StateKey = str
StageParams = Mapping[str, Any]


@dataclass(frozen=True, kw_only=True)
class ExternalDependency:
    """External resource included in conservative cache key derivation.

    Args:
        source: Stable human-readable dependency identifier, such as a catalog
            snapshot name or data file label.
        path: Optional local file whose bytes should be hashed.
        content_hash: Optional precomputed content hash when the resource is not
            directly readable from local disk.
    """

    source: str
    path: Path | str | None = None
    content_hash: str | bytes | None = None

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("ExternalDependency.source must be non-empty.")
        if self.path is not None:
            object.__setattr__(self, "path", Path(self.path))


def external_dependency_digest(dependency: ExternalDependency) -> bytes:
    """Return a conservative digest for a declared external dependency."""

    if not isinstance(dependency, ExternalDependency):
        raise TypeError(
            "external_dependencies() must return ExternalDependency instances; "
            f"got {type(dependency).__name__}. "
            "See nullsim.pipeline.stage.ExternalDependency."
        )

    digest = hashlib.sha256()
    digest.update(dependency.source.encode("utf-8"))
    digest.update(b"\0")
    if dependency.path is not None:
        dependency_path = Path(dependency.path)
        digest.update(str(dependency_path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(dependency_path.read_bytes())
    if dependency.content_hash is not None:
        content_hash = dependency.content_hash
        digest.update(
            content_hash
            if isinstance(content_hash, bytes)
            else content_hash.encode("utf-8")
        )
    return digest.digest()


class Stage(Protocol):
    """Protocol implemented by all nullsim pipeline stage classes."""

    type_name: ClassVar[str]
    family: ClassVar[str]
    consumes: ClassVar[frozenset[StateKey]]
    produces: ClassVar[frozenset[StateKey]]
    # State keys the stage reads OPPORTUNISTICALLY: they participate
    # in the cache key when present but are NOT enforced by the
    # validator (so configs that omit the upstream producer still
    # run). Default empty.
    cache_key_extras: ClassVar[frozenset[StateKey]]
    # True only for stages whose RESULT depends on the per-instance
    # ``context.rng`` (which is seeded from ``config_hash``). Such stages keep
    # ``config_hash`` in their cache key; every other stage is
    # config_hash-independent and may reuse a cached result across unrelated
    # config edits. Default False.
    uses_rng: ClassVar[bool]

    def __init__(self, params: StageParams, context: RunContext) -> None:
        """Initialize a stage instance from validated params and run context."""

    def apply(self, state: SimulationState) -> SimulationState:
        """Return the post-stage simulation state."""

    def diagnostics(self) -> dict[str, Any]:
        """Return optional per-instance diagnostics."""

    def external_dependencies(self) -> list[ExternalDependency]:
        """Return external resources that must enter the cache key."""

    def required_consumes(self) -> frozenset[StateKey]:
        """Return state keys required by this configured stage instance."""


class BaseStage(ABC):
    """Abstract base class for stateless stage implementations.

    If a subclass declares an inner ``Params`` pydantic model,
    ``BaseStage.__init__`` validates the user-supplied ``params`` dict
    against it once and exposes the typed model as ``self.params_model``.
    Stages should read ``self.params_model.<field>`` rather than calling
    ``self.Params.model_validate(self.params)`` again inside ``apply()``
    — repeated validation is wasted work and risks silent re-binding if
    defaults shift between the two calls.
    """

    type_name: ClassVar[str]
    family: ClassVar[str]
    consumes: ClassVar[frozenset[StateKey]] = frozenset()
    produces: ClassVar[frozenset[StateKey]] = frozenset()
    # See :class:`Stage`. Default empty so existing stages opt in
    # without breaking back-compat.
    cache_key_extras: ClassVar[frozenset[StateKey]] = frozenset()
    # See :class:`Stage`. False by default; only stages that draw from
    # ``context.rng`` (``transport.smf28`` and subclasses, ``space_zodi``)
    # set it True.
    uses_rng: ClassVar[bool] = False

    params: StageParams
    params_model: Any  # subclass's validated Params instance, if it has one
    context: RunContext

    def __init__(self, params: StageParams, context: RunContext) -> None:
        params_cls = getattr(type(self), "Params", None)
        if params_cls is not None and isinstance(params_cls, type):
            try:
                self.params_model = params_cls.model_validate(params)
            except AttributeError:
                # Subclass's Params isn't a pydantic model — leave the
                # raw params dict as the only access path.
                self.params_model = None
                self.params = dict(params)
            else:
                # Persist the model_dump as `params` so downstream code
                # (cache key derivation, diagnostics) sees the validated +
                # default-applied form rather than the raw input.
                self.params = self.params_model.model_dump(mode="python")
        else:
            self.params_model = None
            self.params = dict(params)
        self.context = context

    @abstractmethod
    def apply(self, state: SimulationState) -> SimulationState:
        """Return the post-stage simulation state."""

    def diagnostics(self) -> dict[str, Any]:
        """Return optional per-instance diagnostics."""

        return {}

    def external_dependencies(self) -> list[ExternalDependency]:
        """Return external resources that must enter the cache key."""

        return []

    def required_consumes(self) -> frozenset[StateKey]:
        """Return state keys required by this configured stage instance.

        Most stages have static dependencies and can rely on ``consumes``.
        Stages whose params select an optional upstream handoff can override
        this so validation and cache keys see the same configured contract.
        """

        return frozenset(self.consumes)


def consumed_state_keys(instance: Stage) -> frozenset[StateKey]:
    """Return required consume keys for ``instance`` with legacy fallback."""

    required_consumes = getattr(instance, "required_consumes", None)
    if callable(required_consumes):
        return frozenset(required_consumes())
    return frozenset(instance.consumes)
