"""In-memory registries for stages and outputs described in SPEC §§6.1 and 8."""

from __future__ import annotations

from collections.abc import Callable
from types import MappingProxyType
from typing import Any, TypeVar
from collections.abc import Mapping

from pydantic import ValidationError

from nullsim.config import NullsimConfigError
from nullsim.pipeline.context import RunContext
from nullsim.pipeline.stage import Stage, StageParams

StageType = TypeVar("StageType", bound=type[Stage])
CallableType = TypeVar("CallableType", bound=Callable[..., Any])


class Registry:
    """In-memory registry for stage, plot, and table callables."""

    def __init__(self) -> None:
        self._stages: dict[str, type[Stage]] = {}
        self._plots: dict[str, Callable[..., Any]] = {}
        self._tables: dict[str, Callable[..., Any]] = {}

    @property
    def stages(self) -> Mapping[str, type[Stage]]:
        """Registered stage classes keyed by ``type_name``."""

        return MappingProxyType(self._stages)

    @property
    def plots(self) -> Mapping[str, Callable[..., Any]]:
        """Registered plot callables keyed by output name."""

        return MappingProxyType(self._plots)

    @property
    def tables(self) -> Mapping[str, Callable[..., Any]]:
        """Registered table callables keyed by output name."""

        return MappingProxyType(self._tables)

    def register_stage(self, stage_cls: StageType) -> StageType:
        """Register a stage class by its ``type_name`` class variable."""

        type_name = _required_type_name(stage_cls)
        if type_name in self._stages:
            raise ValueError(f"Stage type {type_name!r} is already registered.")
        self._stages[type_name] = stage_cls
        return stage_cls

    def list_stages(self, family: str | None = None) -> tuple[str, ...]:
        """Return registered stage type names, optionally filtered by family."""

        stage_names = (
            type_name
            for type_name, stage_cls in self._stages.items()
            if family is None or getattr(stage_cls, "family", None) == family
        )
        return tuple(sorted(stage_names))

    def register_plot(self, plot_func: CallableType) -> CallableType:
        """Register a plot callable by ``type_name``, ``name``, or function name."""

        name = _callable_registry_name(plot_func)
        if name in self._plots:
            raise ValueError(f"Plot {name!r} is already registered.")
        self._plots[name] = plot_func
        return plot_func

    def register_table(self, table_func: CallableType) -> CallableType:
        """Register a table callable by ``type_name``, ``name``, or function name."""

        name = _callable_registry_name(table_func)
        if name in self._tables:
            raise ValueError(f"Table {name!r} is already registered.")
        self._tables[name] = table_func
        return table_func

    def build_instance(self, spec: Any, context: RunContext) -> Stage:
        """Instantiate the registered stage described by one pipeline spec.

        Args:
            spec: Mapping or object containing at least ``id`` and ``type``.
            context: Per-stage execution context.

        Returns:
            A stage instance initialized with resolved params.

        Raises:
            NullsimConfigError: If the type field is missing or unregistered.
        """

        type_name = stage_type_from_spec(spec)
        try:
            stage_cls = self._stages[type_name]
        except KeyError as exc:
            raise NullsimConfigError(f"Unknown stage type {type_name!r}.") from exc
        _validate_params_model_type(stage_cls)
        params = stage_params_from_spec(spec)
        try:
            return stage_cls(params, context)
        except ValidationError as exc:
            raise _stage_params_error(stage_cls, spec, exc) from exc


registry = Registry()


def register_stage(stage_cls: StageType) -> StageType:
    """Register a stage class in the module-level singleton registry."""

    return registry.register_stage(stage_cls)


def register_plot(plot_func: CallableType) -> CallableType:
    """Register a plot callable in the module-level singleton registry."""

    return registry.register_plot(plot_func)


def register_table(table_func: CallableType) -> CallableType:
    """Register a table callable in the module-level singleton registry."""

    return registry.register_table(table_func)


def build_instance(spec: Any, context: RunContext) -> Stage:
    """Instantiate a stage from the module-level singleton registry."""

    return registry.build_instance(spec, context)


def list_stages(family: str | None = None) -> tuple[str, ...]:
    """Return stage type names registered in the module-level singleton registry."""

    return registry.list_stages(family)


def stage_id_from_spec(spec: Any) -> str:
    """Return the instance id from a mapping- or object-shaped stage spec."""

    stage_id = _value_from_spec(spec, "id")
    if not isinstance(stage_id, str) or not stage_id:
        raise NullsimConfigError("Pipeline stage specs must define a non-empty 'id'.")
    return stage_id


def stage_type_from_spec(spec: Any) -> str:
    """Return the stage type name from a mapping- or object-shaped stage spec."""

    type_name = _value_from_spec(spec, "type")
    if not isinstance(type_name, str) or not type_name:
        stage_id = _value_from_spec(spec, "id") or "<unknown>"
        raise NullsimConfigError(
            f"Pipeline stage {stage_id!r} must define a non-empty 'type'."
        )
    return type_name


def stage_params_from_spec(spec: Any) -> StageParams:
    """Return stage params after removing pipeline-level ``id`` and ``type`` keys.

    The top-level ``params`` key is special-cased to support stages whose
    ``Params`` model has a literal ``params`` field (e.g.
    ``chip.kernel_mzi_mesh``): a Mapping value is merged in as keyword
    overrides, while a non-Mapping value (list / tuple / scalar) is passed
    through unchanged under the ``params`` key so the stage's pydantic
    model can validate it.
    """

    params: dict[str, Any] = {}
    explicit_params = _value_from_spec(spec, "params")
    if isinstance(explicit_params, Mapping):
        params.update(explicit_params)
    elif explicit_params is not None:
        params["params"] = explicit_params

    for key, value in _items_from_spec(spec):
        if key not in {"id", "type", "params"}:
            params[key] = value
    return params


def validate_stage_params(
    stage_cls: type[Stage],
    params: StageParams,
    spec: Any,
) -> StageParams:
    """Validate stage params with ``stage_cls.Params`` when the class defines it."""

    params_model = getattr(stage_cls, "Params", None)
    if params_model is None:
        return params
    if not hasattr(params_model, "model_validate"):
        raise TypeError(f"{stage_cls.__name__}.Params must be a pydantic v2 model.")

    try:
        validated_params = params_model.model_validate(params)
    except ValidationError as exc:
        raise _stage_params_error(stage_cls, spec, exc) from exc
    return validated_params.model_dump()


def _validate_params_model_type(stage_cls: type[Stage]) -> None:
    params_model = getattr(stage_cls, "Params", None)
    if params_model is not None and not hasattr(params_model, "model_validate"):
        raise TypeError(f"{stage_cls.__name__}.Params must be a pydantic v2 model.")


def _stage_params_error(
    stage_cls: type[Stage],
    spec: Any,
    exc: ValidationError,
) -> NullsimConfigError:
    stage_id = _value_from_spec(spec, "id") or "<unknown>"
    type_name = _required_type_name(stage_cls)
    return NullsimConfigError(
        f"Invalid params for pipeline stage {stage_id!r} "
        f"of type {type_name!r}: {exc}"
    )


def _required_type_name(stage_cls: type[Stage]) -> str:
    type_name = getattr(stage_cls, "type_name", None)
    if not isinstance(type_name, str) or not type_name:
        raise ValueError("Stage classes must define a non-empty type_name.")
    return type_name


def _callable_registry_name(func: Callable[..., Any]) -> str:
    name = getattr(func, "type_name", None) or getattr(func, "name", None) or func.__name__
    if not isinstance(name, str) or not name:
        raise ValueError("Registered callables must have a non-empty registry name.")
    return name


def _value_from_spec(spec: Any, key: str) -> Any:
    if isinstance(spec, Mapping):
        return spec.get(key)
    return getattr(spec, key, None)


def _items_from_spec(spec: Any) -> list[tuple[str, Any]]:
    if isinstance(spec, Mapping):
        return list(spec.items())
    if hasattr(spec, "model_dump"):
        return list(spec.model_dump().items())
    if hasattr(spec, "__dict__"):
        return list(vars(spec).items())
    return []
