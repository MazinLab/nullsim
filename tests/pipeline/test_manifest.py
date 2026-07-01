from __future__ import annotations

from types import SimpleNamespace

from nullsim.pipeline import ExternalDependency, RunResult, SimulationState, StageTiming
from nullsim.pipeline.manifest import build_manifest


class DependencyStage:
    type_name = "tests.manifest.dependency"
    family = "test"
    consumes = frozenset()
    produces = frozenset()

    def __init__(self, instance_id: str, content_hash: str) -> None:
        self.context = SimpleNamespace(instance_id=instance_id)
        self.content_hash = content_hash

    def external_dependencies(self) -> list[ExternalDependency]:
        return [
            ExternalDependency(
                source=f"dependency:{self.context.instance_id}",
                content_hash=self.content_hash,
            )
        ]


def test_build_manifest_records_external_dependencies_for_all_sweep_cells() -> None:
    result = RunResult(
        state=SimulationState.empty(),
        diagnostics={},
        timings=(),
        cache_hit_count=0,
        config_hash="cfg",
    )

    manifest = build_manifest(
        config_hash="cfg",
        run_hash="run",
        rng_root_seed=1,
        result=result,
        instances=(
            (DependencyStage("calibration", "hash-cell-0"),),
            (DependencyStage("calibration", "hash-cell-1"),),
        ),
    )

    dependencies = manifest["external_dependencies"]
    assert dependencies == [
        {
            "cell_index": 0,
            "instance_id": "calibration",
            "dependencies": [
                {
                    "source": "dependency:calibration",
                    "path": None,
                    "content_hash": "hash-cell-0",
                }
            ],
        },
        {
            "cell_index": 1,
            "instance_id": "calibration",
            "dependencies": [
                {
                    "source": "dependency:calibration",
                    "path": None,
                    "content_hash": "hash-cell-1",
                }
            ],
        },
    ]


def test_build_manifest_cache_counts_can_aggregate_sweep_results() -> None:
    primary = RunResult(
        state=SimulationState.empty(),
        diagnostics={},
        timings=(
            StageTiming("scene", 0.1, False),
            StageTiming("detector", 0.1, True),
        ),
        cache_hit_count=1,
        config_hash="cfg-0",
    )
    second = RunResult(
        state=SimulationState.empty(),
        diagnostics={},
        timings=(
            StageTiming("scene", 0.1, True),
            StageTiming("detector", 0.1, True),
        ),
        cache_hit_count=2,
        config_hash="cfg-1",
    )

    manifest = build_manifest(
        config_hash="cfg",
        run_hash="run",
        rng_root_seed=1,
        result=primary,
        results=(primary, second),
    )

    assert manifest["cache"] == {
        "hits": 3,
        "misses": 1,
        "stage_count": 4,
    }
    assert "source_digest" in manifest["source_identity"]
    assert manifest["timings"] == [
        {"instance_id": "scene", "elapsed_s": 0.1, "cache_hit": False},
        {"instance_id": "detector", "elapsed_s": 0.1, "cache_hit": True},
    ]
