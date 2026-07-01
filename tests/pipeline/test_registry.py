from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict

from nullsim.config import NullsimConfigError
from nullsim.pipeline import BaseStage, Registry, RunContext, SimulationState


class LocalStage(BaseStage):
    type_name = "tests.registry.local"
    family = "scene"
    produces = frozenset({"scene"})

    def apply(self, state: SimulationState) -> SimulationState:
        return state


def _context(tmp_path: Path) -> RunContext:
    return RunContext(
        rng=np.random.default_rng(1),
        config_hash="config",
        sweep_coord_tuple=(),
        instance_id="local",
        run_dir=tmp_path,
    )


def test_register_stage_decorator_builds_instance_with_params(tmp_path: Path) -> None:
    local_registry = Registry()

    registered_cls = local_registry.register_stage(LocalStage)
    instance = local_registry.build_instance(
        {"id": "source", "type": "tests.registry.local", "star_jmag": 5.0},
        _context(tmp_path),
    )

    assert registered_cls is LocalStage
    assert isinstance(instance, LocalStage)
    assert instance.params == {"star_jmag": 5.0}


def test_register_stage_rejects_duplicate_type_name() -> None:
    local_registry = Registry()
    local_registry.register_stage(LocalStage)

    with pytest.raises(ValueError, match="already registered"):
        local_registry.register_stage(LocalStage)


def test_build_instance_raises_for_unknown_stage_type(tmp_path: Path) -> None:
    local_registry = Registry()

    with pytest.raises(NullsimConfigError, match="Unknown stage type"):
        local_registry.build_instance(
            {"id": "missing", "type": "tests.registry.missing"},
            _context(tmp_path),
        )


def test_build_instance_validates_pydantic_params_once(tmp_path: Path) -> None:
    local_registry = Registry()
    validate_calls = {"n": 0}

    class CountingParams(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)
        value: int = 0

        @classmethod
        def model_validate(cls, *args: Any, **kwargs: Any):  # type: ignore[override]
            validate_calls["n"] += 1
            return super().model_validate(*args, **kwargs)

    class CountingStage(BaseStage):
        type_name = "tests.registry.counting"
        family = "scene"
        produces = frozenset({"scene"})
        Params = CountingParams

        def apply(self, state: SimulationState) -> SimulationState:
            return state

    local_registry.register_stage(CountingStage)
    instance = local_registry.build_instance(
        {"id": "counting", "type": "tests.registry.counting", "value": 7},
        _context(tmp_path),
    )

    assert validate_calls["n"] == 1
    assert instance.params == {"value": 7}


def test_register_plot_and_table_use_callable_names() -> None:
    local_registry = Registry()

    def contrast_curve(*args: Any, **kwargs: Any) -> None:
        return None

    def performance_summary(*args: Any, **kwargs: Any) -> None:
        return None

    local_registry.register_plot(contrast_curve)
    local_registry.register_table(performance_summary)

    assert local_registry.plots["contrast_curve"] is contrast_curve
    assert local_registry.tables["performance_summary"] is performance_summary
