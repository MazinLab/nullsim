"""Unit tests for ``nullsim.cli`` user-surface error handling.

These tests guard the SPEC §10 contract that the CLI never returns a raw
Python traceback for the failure modes an ordinary user is going to hit
(typoed path, directory instead of file, permission-denied).  Each failure
must reach the existing :class:`NullsimConfigError` handler in
:func:`nullsim.cli.main` so the user sees ``nullsim: config error: ...``
and the process exits 2.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ConfigDict

import nullsim.cli as cli
from nullsim.cli import (
    _resolve_output_dir,
    _run_hash_with_externals,
    _with_runtime_output_dir,
    _write_run_manifest,
    main,
)
from nullsim.config import load_config
from nullsim.outputs import RunRecord
from nullsim.pipeline import (
    BaseStage,
    ExternalDependency,
    RunResult,
    SimulationState,
    StageTiming,
    build_stage_instances,
    register_stage,
)


@register_stage
class CliValidationProducer(BaseStage):
    type_name = "tests.cli_validation.producer"
    family = "tests"
    produces = frozenset({"field"})

    def apply(self, state: SimulationState) -> SimulationState:
        return state


@register_stage
class CliValidationConsumer(BaseStage):
    type_name = "tests.cli_validation.consumer"
    family = "tests"
    consumes = frozenset({"field"})
    produces = frozenset({"results"})

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)

        needs_results: bool = False

    def required_consumes(self) -> frozenset[str]:
        required = set(self.consumes)
        if self.params_model.needs_results:
            required.add("results")
        return frozenset(required)

    def apply(self, state: SimulationState) -> SimulationState:
        return state


def test_run_with_missing_config_emits_friendly_error(
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    """``nullsim run`` on a non-existent path must exit 2 with no traceback."""

    missing = tmp_path / "definitely-not-here.toml"
    assert not missing.exists()

    exit_code = main(["run", str(missing)])
    captured = capsys.readouterr()

    assert exit_code == 2, f"expected exit 2, got {exit_code!r}"
    assert "config error" in captured.err
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out


def test_validate_with_missing_config_emits_friendly_error(
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    """``nullsim validate`` must mirror the same handling as ``run``."""

    missing = tmp_path / "still-not-here.toml"
    exit_code = main(["validate", str(missing)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "config error" in captured.err
    assert "Traceback" not in captured.err


def test_run_with_directory_path_emits_friendly_error(
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    """Pointing ``run`` at a directory must surface as a config error, not
    a raw ``IsADirectoryError`` traceback."""

    exit_code = main(["run", str(tmp_path)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "config error" in captured.err
    assert "Traceback" not in captured.err



def test_validate_catches_misordered_pipeline(
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    """`nullsim validate` runs validate_dependencies (SPEC §7.3 step 6).

    Before this check, validate only verified the schema and sweep
    expansion — a user with a misordered pipeline (chip before scene)
    would see "config OK" and then a runtime failure on `nullsim run`.
    """

    config = tmp_path / "broken.toml"
    config.write_text(
        """
[run]
seed = 0

[grid]
wavelength_center_um = 1.65
wavelength_bandwidth_um = 0.30
n_wavelength_bins = 3

[[pipeline.stages]]
id = "detector"
type = "detector.ideal_counter"

[[pipeline.stages]]
id = "scene"
type = "scene.point_source"
star_jmag = 5.0
planet_contrast = 1.0e-7
planet_separation_mas = 100.0

[[pipeline.stages]]
id = "telescope"
type = "telescope.fixed_aperture"
diameter_m = 10.0

[[pipeline.stages]]
id = "injection"
type = "injection.ideal"

[[pipeline.stages]]
id = "chip"
type = "chip.identity"
insertion_loss_db = 0.0
""",
        encoding="utf-8",
    )

    exit_code = main(["validate", str(config)])
    captured = capsys.readouterr()

    assert exit_code == 2, "misordered pipeline must NOT report config OK"
    assert "config OK" not in captured.out
    assert "Traceback" not in captured.err


def test_validate_checks_dependency_contract_for_every_sweep_cell(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """A later sweep cell can change a stage's configured dependency set."""

    config = tmp_path / "cell-dependent.toml"
    config.write_text(
        """
[run]
seed = 0

[grid]
wavelength_center_um = 1.65
wavelength_bandwidth_um = 0.30
n_wavelength_bins = 3

[scene.star]
jmag = 5.0
angular_diameter_mas = 1.0
distance_pc = 5.0

[scene.planet]
contrast = 1.0e-7
separation_mas = 100.0
position_angle_deg = 0.0

[observation]
hour_angle_h = 0.0
integration_time_s = 10.0

[[pipeline.stages]]
id = "producer"
type = "tests.cli_validation.producer"

[[pipeline.stages]]
id = "consumer"
type = "tests.cli_validation.consumer"

[[sweep]]
param = "pipeline.stages.consumer.needs_results"
values = [false, true]
""",
        encoding="utf-8",
    )

    exit_code = main(["validate", str(config)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "consumer" in captured.err
    assert "results" in captured.err
    assert "config OK" not in captured.out


def test_validate_rejects_unknown_output_before_execution(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Requested plot/table names are part of validation, not post-run cleanup."""

    config = tmp_path / "unknown-output.toml"
    config.write_text(
        """
[run]
seed = 0

[grid]
wavelength_center_um = 1.65
wavelength_bandwidth_um = 0.30
n_wavelength_bins = 3

[scene.star]
jmag = 5.0
angular_diameter_mas = 1.0
distance_pc = 5.0

[scene.planet]
contrast = 1.0e-7
separation_mas = 100.0
position_angle_deg = 0.0

[observation]
hour_angle_h = 0.0
integration_time_s = 10.0

[[pipeline.stages]]
id = "scene"
type = "scene.point_source"

[output]
plots = ["not_a_real_plot"]
""",
        encoding="utf-8",
    )

    exit_code = main(["validate", str(config)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "unknown plot output" in captured.err
    assert "not_a_real_plot" in captured.err
    assert "config OK" not in captured.out
    assert "Traceback" not in captured.err


def test_list_outputs_shows_registered_outputs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["list-outputs", "--kind", "tables"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "tables:" in captured.out
    assert "performance_summary" in captured.out
    assert "Traceback" not in captured.err


def test_inspect_prints_pipeline_and_trust_notes(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    config = tmp_path / "inspect.toml"
    config.write_text(
        """
[run]
name = "inspect_smoke"
seed = 0

[grid]
wavelength_center_um = 1.0
wavelength_bandwidth_um = 0.1
n_wavelength_bins = 3

[scene.star]
jmag = 5.0
angular_diameter_mas = 1.0
distance_pc = 5.0

[scene.planet]
contrast = 1.0e-7
separation_mas = 100.0
position_angle_deg = 0.0

[observation]
hour_angle_h = 0.0
integration_time_s = 10.0

[[pipeline.stages]]
id = "scene"
type = "scene.point_source"

[[pipeline.stages]]
id = "telescope"
type = "telescope.fixed_aperture"
diameter_m = 8.0

[[pipeline.stages]]
id = "pupil_modes"
type = "telescope.pupil_modes"
n_modes = 4
layout = "packed_circular"
""",
        encoding="utf-8",
    )

    exit_code = main(["inspect", str(config)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "sweep cells: 1" in captured.out
    assert "pupil_modes: telescope.pupil_modes" in captured.out
    assert "physics/trust notes:" in captured.out
    assert "surrogate" in captured.out
    assert "Traceback" not in captured.err


def test_run_dry_run_does_not_create_outputs(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    config = tmp_path / "dry-run.toml"
    output_dir = tmp_path / "planned-output"
    config.write_text(
        """
[run]
name = "dry_run_smoke"
seed = 0

[grid]
wavelength_center_um = 1.65
wavelength_bandwidth_um = 0.30
n_wavelength_bins = 3

[scene.star]
jmag = 5.0
angular_diameter_mas = 1.0
distance_pc = 5.0

[scene.planet]
contrast = 1.0e-7
separation_mas = 100.0
position_angle_deg = 0.0

[observation]
hour_angle_h = 0.0
integration_time_s = 10.0

[[pipeline.stages]]
id = "scene"
type = "scene.point_source"
""",
        encoding="utf-8",
    )

    exit_code = main(["run", str(config), "--out", str(output_dir), "--dry-run"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "nullsim dry-run" in captured.out
    assert f"output_dir: {output_dir}" in captured.out
    assert not output_dir.exists()
    assert "Traceback" not in captured.err


def test_default_output_dir_uses_timestamp(tmp_path: Path) -> None:
    config = tmp_path / "default-output.toml"
    config.write_text(
        """
[run]
name = "default_output"
seed = 0

[grid]
wavelength_center_um = 1.65
wavelength_bandwidth_um = 0.30
n_wavelength_bins = 3

[scene.star]
jmag = 5.0
angular_diameter_mas = 1.0
distance_pc = 5.0

[scene.planet]
contrast = 1.0e-7
separation_mas = 100.0
position_angle_deg = 0.0

[observation]
hour_angle_h = 0.0
integration_time_s = 10.0

[[pipeline.stages]]
id = "scene"
type = "scene.point_source"
""",
        encoding="utf-8",
    )

    resolved = load_config(config)
    output_dir = _resolve_output_dir(resolved, None)

    assert output_dir == Path(f"results/default_output/{resolved.timestamp}")


def test_runtime_output_dir_reaches_stage_context(tmp_path: Path) -> None:
    config = tmp_path / "run-dir.toml"
    config.write_text(
        """
[run]
seed = 0

[grid]
wavelength_center_um = 1.65
wavelength_bandwidth_um = 0.30
n_wavelength_bins = 3

[scene.star]
jmag = 5.0
angular_diameter_mas = 1.0
distance_pc = 5.0

[scene.planet]
contrast = 1.0e-7
separation_mas = 100.0
position_angle_deg = 0.0

[observation]
hour_angle_h = 0.0
integration_time_s = 10.0

[[pipeline.stages]]
id = "scene"
type = "scene.point_source"
""",
        encoding="utf-8",
    )
    resolved = load_config(config)
    runtime_dir = tmp_path / "actual-output"

    runtime_config = _with_runtime_output_dir(resolved, runtime_dir)
    instances = build_stage_instances(runtime_config)

    assert instances[0].context.run_dir == runtime_dir
    assert instances[0].context.source_path == config
    assert runtime_config.config_hash == resolved.config_hash
    assert runtime_config.run_hash == resolved.run_hash


def test_paper_collection_uses_aggregate_sweep_run_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "paper-hash.toml"
    output_dir = tmp_path / "out"
    config.write_text(
        """
[run]
name = "paper_hash"
seed = 0
n_workers = 1

[grid]
wavelength_center_um = 1.65
wavelength_bandwidth_um = 0.30
n_wavelength_bins = 3

[scene.star]
jmag = 5.0
angular_diameter_mas = 1.0
distance_pc = 5.0

[scene.planet]
contrast = 1.0e-7
separation_mas = 100.0
position_angle_deg = 0.0

[observation]
hour_angle_h = 0.0
integration_time_s = 10.0

[[pipeline.stages]]
id = "scene"
type = "scene.point_source"

[[sweep]]
param = "scene.star.jmag"
values = [5.0, 6.0]

[output]
paper = true
""",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_write_paper_outputs(collection, _output_dir):
        captured["collection_run_hash"] = collection.config.run_hash
        captured["record_run_hashes"] = tuple(
            record.config.run_hash for record in collection.records
        )
        return ()

    monkeypatch.setattr(cli, "write_paper_outputs", fake_write_paper_outputs)

    exit_code = main(["run", str(config), "--out", str(output_dir), "--workers", "1"])

    assert exit_code == 0
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert captured["collection_run_hash"] == manifest["run_hash"]
    assert captured["collection_run_hash"] != captured["record_run_hashes"][0]


def test_run_hash_with_externals_hashes_path_only_dependency(tmp_path) -> None:
    dependency_path = tmp_path / "dependency.txt"
    dependency_path.write_text("first", encoding="utf-8")

    class PathDependencyStage:
        context = SimpleNamespace(instance_id="path_dependency")

        def external_dependencies(self) -> list[ExternalDependency]:
            return [
                ExternalDependency(
                    source="test_path_only_dependency",
                    path=dependency_path,
                )
            ]

    resolved = SimpleNamespace(config_hash="cfg", raw_config={}, source_path=None)
    first_hash = _run_hash_with_externals(resolved, [PathDependencyStage()])

    dependency_path.write_text("second", encoding="utf-8")
    second_hash = _run_hash_with_externals(resolved, [PathDependencyStage()])

    assert second_hash != first_hash


def test_write_run_manifest_aggregates_sweep_cache_counts(tmp_path: Path) -> None:
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
    records = [
        RunRecord(
            config=SimpleNamespace(config_hash="cfg-0", run_hash="run-0"),
            result=primary,
            run_dir=tmp_path / "cell-0",
            coord_tuple=(("scene.star.jmag", 5.0),),
        ),
        RunRecord(
            config=SimpleNamespace(config_hash="cfg-1", run_hash="run-1"),
            result=second,
            run_dir=tmp_path / "cell-1",
            coord_tuple=(("scene.star.jmag", 6.0),),
        ),
    ]
    resolved = SimpleNamespace(
        config_hash="cfg",
        run_hash="run",
        run=SimpleNamespace(seed=123),
    )

    _write_run_manifest(
        output_dir=tmp_path,
        resolved=resolved,
        records=records,
        instances=(),
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cache"] == {"hits": 3, "misses": 1, "stage_count": 4}
    assert [cell["cache_hit_count"] for cell in manifest["sweep"]["cells"]] == [1, 2]
