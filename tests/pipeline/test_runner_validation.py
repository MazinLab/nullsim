from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nullsim.config import NullsimConfigError
from nullsim.pipeline import BaseStage, RunContext, SimulationState, validate_dependencies


class SceneProducer(BaseStage):
    type_name = "tests.validation.scene_producer"
    family = "scene"
    # Produces the whole `scene` sub-budget; the dotted-path validator
    # treats this as covering any `scene.*` consume by an ancestor rule.
    produces = frozenset({"scene"})

    def apply(self, state: SimulationState) -> SimulationState:
        return state


class SceneConsumer(BaseStage):
    type_name = "tests.validation.scene_consumer"
    family = "detector"
    consumes = frozenset({"scene.planet"})
    produces = frozenset({"results"})

    def apply(self, state: SimulationState) -> SimulationState:
        return state


class SceneStarProducer(BaseStage):
    """Sibling-produce-only: writes scene.star but not scene.planet."""
    type_name = "tests.validation.scene_star_producer"
    family = "scene"
    produces = frozenset({"scene.star"})

    def apply(self, state: SimulationState) -> SimulationState:
        return state


class FieldConsumer(BaseStage):
    type_name = "tests.validation.field_consumer"
    family = "chip"
    consumes = frozenset({"field.amplitudes"})
    produces = frozenset({"results"})

    def apply(self, state: SimulationState) -> SimulationState:
        return state


def _stage(stage_cls: type[BaseStage], instance_id: str, tmp_path: Path) -> BaseStage:
    return stage_cls(
        {},
        RunContext(
            rng=np.random.default_rng(1),
            config_hash="config",
            sweep_coord_tuple=(),
            instance_id=instance_id,
            run_dir=tmp_path,
        ),
    )


def test_validate_dependencies_accepts_earlier_produced_budget_root(
    tmp_path: Path,
) -> None:
    producer = _stage(SceneProducer, "scene", tmp_path)
    consumer = _stage(SceneConsumer, "consumer", tmp_path)

    validate_dependencies([producer, consumer])


def test_validate_dependencies_respects_declaration_order(tmp_path: Path) -> None:
    producer = _stage(SceneProducer, "scene", tmp_path)
    consumer = _stage(SceneConsumer, "consumer", tmp_path)

    with pytest.raises(NullsimConfigError, match="consumer.*scene"):
        validate_dependencies([consumer, producer])


def test_validate_dependencies_reports_missing_budget_and_instance_id(
    tmp_path: Path,
) -> None:
    field_consumer = _stage(FieldConsumer, "chip", tmp_path)

    with pytest.raises(NullsimConfigError, match="chip.*field"):
        validate_dependencies([field_consumer])



def test_validate_rejects_sibling_only_produce_for_dotted_consume(
    tmp_path: Path,
) -> None:
    """SPEC §6.1: a producer of only `scene.star` does NOT cover a
    consumer of `scene.planet`. They are siblings, not ancestors.

    The previous root-fold semantics conflated "produces in the same
    sub-budget root" with "produces a covering dotted path." This test
    pins the new strict-but-fair rule: only an exact match or a broader
    ancestor produce satisfies a dotted-path consume.
    """
    sibling_producer = _stage(SceneStarProducer, "scene_star", tmp_path)
    consumer = _stage(SceneConsumer, "consumer", tmp_path)

    with pytest.raises(NullsimConfigError, match="consumer.*scene\\.planet"):
        validate_dependencies([sibling_producer, consumer])


def test_validate_accepts_exact_match_dotted_produce(tmp_path: Path) -> None:
    """produces={'scene.planet'} satisfies consumes={'scene.planet'} exactly."""

    class ScenePlanetProducer(BaseStage):
        type_name = "tests.validation.scene_planet_producer"
        family = "scene"
        produces = frozenset({"scene.planet"})

        def apply(self, state: SimulationState) -> SimulationState:
            return state

    producer = _stage(ScenePlanetProducer, "scene_planet", tmp_path)
    consumer = _stage(SceneConsumer, "consumer", tmp_path)

    validate_dependencies([producer, consumer])  # must not raise



def test_run_dir_template_is_substituted_in_context(tmp_path: Path) -> None:
    """Issue #4: RunContext.run_dir must contain a real path, not a literal
    template like ``results/{run_hash}``.

    Before this fix, build_stage_instances built a RunContext with the
    raw ``output.dir`` template string, so any stage using
    ``context.run_dir`` would write to a path containing literal `{...}`
    characters. The fix substitutes `{run_hash}` and `{config_hash}` at
    instance-build time.
    """
    from nullsim.pipeline.runner import build_stage_instances

    # Use a registered trivial stage so build_stage_instances can
    # actually construct it. The test isn't about which stage runs —
    # only about the run_dir template substitution in its RunContext.
    config = {
        "config_hash": "abc123",
        "run_hash": "def456",
        "run": {"seed": 0},
        "output": {"dir": "results/{run_hash}/{config_hash}"},
        "pipeline": {
            "stages": [
                {
                    "id": "scene",
                    "type": "scene.point_source",
                    "star_jmag": 5.0,
                    "planet_contrast": 1.0e-7,
                    "planet_separation_mas": 100.0,
                },
            ],
        },
    }
    instances = build_stage_instances(config)
    rendered = str(instances[0].context.run_dir)
    assert "{" not in rendered, (
        f"run_dir template must be substituted, got {rendered!r}"
    )
    assert rendered.endswith("def456/abc123")


def test_run_dir_template_supports_timestamp(tmp_path: Path) -> None:
    """``{timestamp}`` resolves to the load-time wall-clock stamp.

    The CLI's ``_resolve_output_dir`` and the runner's ``_run_dir`` both
    read ``ResolvedConfig.timestamp`` so multi-cell sweeps agree on the
    same parent directory for one process.
    """
    from nullsim.pipeline.runner import build_stage_instances

    config = {
        "config_hash": "abc123",
        "run_hash": "def456",
        "timestamp": "20260518-090000",
        "run": {"seed": 0},
        "output": {"dir": "results/{timestamp}"},
        "pipeline": {
            "stages": [
                {
                    "id": "scene",
                    "type": "scene.point_source",
                    "star_jmag": 5.0,
                    "planet_contrast": 1.0e-7,
                    "planet_separation_mas": 100.0,
                },
            ],
        },
    }
    instances = build_stage_instances(config)
    rendered = str(instances[0].context.run_dir)
    assert rendered.endswith("results/20260518-090000"), rendered
