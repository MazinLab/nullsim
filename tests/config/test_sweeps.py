from __future__ import annotations

from pathlib import Path

import pytest

from nullsim.config import NullsimConfigError, expand_sweeps, load_config


def _write_config(tmp_path: Path, sweep_text: str) -> Path:
    config_path = tmp_path / "sweep.toml"
    config_path.write_text(_base_config(sweep_text), encoding="utf-8")
    return config_path


def _base_config(sweep_text: str) -> str:
    return f"""
[run]
name = "sweep_test"
seed = 123

[grid]
wavelength_center_um = 1.65
wavelength_bandwidth_um = 0.3
n_wavelength_bins = 3

[scene.star]
jmag = 4.5
angular_diameter_mas = 2.0
distance_pc = 3.6

[scene.planet]
contrast = 1.0e-7
separation_mas = 100.0
position_angle_deg = 45.0

[observation]
hour_angle_h = 0.0
integration_time_s = 10.0

[[pipeline.stages]]
id = "scene"
type = "scene.point_source"

[[pipeline.stages]]
id = "chip"
type = "chip.identity"
n_modes = 2

{sweep_text}
"""


def test_grid_expansion_uses_cartesian_product(tmp_path: Path) -> None:
    resolved_config = load_config(
        _write_config(
            tmp_path,
            """
[[sweep]]
param = "scene.star.jmag"
values = [4.0, 5.0]
mode = "grid"

[[sweep]]
param = "pipeline.stages.chip.n_modes"
values = [3, 4, 5]
mode = "grid"
""",
        )
    )

    cells = expand_sweeps(resolved_config)

    assert len(cells) == 6
    assert {
        (cell.config.scene.star.jmag, cell.config.pipeline.stages[1].n_modes)
        for cell in cells
    } == {
        (4.0, 3),
        (4.0, 4),
        (4.0, 5),
        (5.0, 3),
        (5.0, 4),
        (5.0, 5),
    }


def test_grid_section_can_be_swept_for_band_factories(tmp_path: Path) -> None:
    resolved_config = load_config(
        _write_config(
            tmp_path,
            """
[[sweep]]
param = "grid.wavelength_center_um"
values = [1.02, 1.25, 1.65]
mode = "zip"

[[sweep]]
param = "grid.wavelength_bandwidth_um"
values = [0.10, 0.16, 0.30]
mode = "zip"
""",
        )
    )

    cells = expand_sweeps(resolved_config)

    assert [
        (
            cell.config.grid.wavelength_center_um,
            cell.config.grid.wavelength_bandwidth_um,
        )
        for cell in cells
    ] == [(1.02, 0.10), (1.25, 0.16), (1.65, 0.30)]


def test_zip_expansion_checks_equal_lengths(tmp_path: Path) -> None:
    resolved_config = load_config(
        _write_config(
            tmp_path,
            """
[[sweep]]
param = "scene.star.jmag"
values = [4.0, 5.0]
mode = "zip"

[[sweep]]
param = "observation.integration_time_s"
values = [10.0]
mode = "zip"
""",
        )
    )

    with pytest.raises(NullsimConfigError, match="equal lengths"):
        expand_sweeps(resolved_config)


def test_sweep_cells_do_not_alias_resolved_values(tmp_path: Path) -> None:
    resolved_config = load_config(
        _write_config(
            tmp_path,
            """
[[sweep]]
param = "pipeline.stages.chip.n_modes"
values = [3, 4]
mode = "grid"
""",
        )
    )

    first_cell, second_cell = expand_sweeps(resolved_config)

    assert first_cell.config is not second_cell.config
    assert first_cell.config.pipeline.stages is not second_cell.config.pipeline.stages
    assert first_cell.config.pipeline.stages[1].n_modes == 3
    assert second_cell.config.pipeline.stages[1].n_modes == 4
    assert first_cell.config.sweep == ()
    assert second_cell.config.sweep == ()



def test_range_sweep_does_not_accumulate_float_drift() -> None:
    """The fix for issue #4: ``current += step`` accumulated rounding error.

    The classic case is `range(0.0, 1.0, 0.1)` — the previous accumulating
    impl returned a final value `0.9000000000000001`. Computing each
    point as `start + i*step` keeps every draw within one ULP of the
    analytically-correct point.
    """
    from nullsim.config.sweeps import _range_values

    values = _range_values(0.0, 1.0, 0.1)
    assert len(values) == 10
    assert values[-1] == 0.9  # exact, not 0.9000000000000001
    # Negative step: half-open [start, stop) honored.
    assert _range_values(1.0, 0.0, -0.25) == (1.0, 0.75, 0.5, 0.25)
    # Integer-clean range still emits exact values.
    assert _range_values(4.0, 12.0, 2.0) == (4.0, 6.0, 8.0, 10.0)
    # Zero step is a config error, not an infinite loop.
    try:
        _range_values(0.0, 1.0, 0.0)
    except Exception as exc:
        assert "non-zero" in str(exc)
    else:
        raise AssertionError("expected NullsimConfigError on step=0")


def test_sweep_cells_preserve_explicit_stage_overrides(tmp_path: Path) -> None:
    """Sweep-path re-materialization must not clobber user-explicit stage
    params with their top-level binding source (review 2026-06-10 Issue 6:
    the same config produced different stage params with and without a
    [[sweep]] block)."""

    config_path = tmp_path / "sweep.toml"
    config_path.write_text(
        _base_config(
            """
[[sweep]]
param = "scene.star.jmag"
values = [4.0, 8.0]
mode = "grid"
"""
        ).replace(
            'id = "scene"\ntype = "scene.point_source"',
            'id = "scene"\ntype = "scene.point_source"\nstar_distance_pc = 7.7',
        ),
        encoding="utf-8",
    )

    cells = expand_sweeps(load_config(config_path))
    assert len(cells) == 2
    for cell in cells:
        scene_stage = next(
            stage for stage in cell.config.pipeline.stages if stage.id == "scene"
        )
        dumped = scene_stage.model_dump(mode="python")
        # The explicit stage-level override survives the sweep cell's
        # overwrite=True re-materialization...
        assert dumped["star_distance_pc"] == 7.7
    # ...while the swept top-level value still reaches the stage spec.
    jmags = sorted(
        next(
            stage for stage in cell.config.pipeline.stages if stage.id == "scene"
        ).model_dump(mode="python")["star_jmag"]
        for cell in cells
    )
    assert jmags == [4.0, 8.0]


def test_sweep_over_source_of_explicit_stage_param_raises(tmp_path: Path) -> None:
    """An explicit stage param mirroring a swept top-level source would make
    every cell identical for that param — must fail loudly."""

    config_path = tmp_path / "sweep.toml"
    config_path.write_text(
        _base_config(
            """
[[sweep]]
param = "scene.star.jmag"
values = [4.0, 8.0]
mode = "grid"
"""
        ).replace(
            'id = "scene"\ntype = "scene.point_source"',
            'id = "scene"\ntype = "scene.point_source"\nstar_jmag = 5.5',
        ),
        encoding="utf-8",
    )

    with pytest.raises(NullsimConfigError, match="scene.star.jmag"):
        expand_sweeps(load_config(config_path))


def test_observation_sweep_requires_bound_field(tmp_path: Path) -> None:
    """Sweeping an [observation] field with no stage binding would produce N
    cells with identical physics (review 2026-06-10 Issue 15)."""

    config_path = _write_config(
        tmp_path,
        """
[[sweep]]
param = "observation.hour_angle_h"
values = [0.0, 1.0]
mode = "grid"
""",
    )

    with pytest.raises(NullsimConfigError, match="not materialized"):
        expand_sweeps(load_config(config_path))


def test_observation_integration_time_materializes_and_sweeps(
    tmp_path: Path,
) -> None:
    """[observation].integration_time_s reaches detector stages through the
    binding, and sweeping it propagates per cell."""

    config_path = tmp_path / "sweep.toml"
    config_path.write_text(
        _base_config(
            """
[[sweep]]
param = "observation.integration_time_s"
values = [10.0, 20.0]
mode = "grid"
"""
        ).replace(
            'id = "chip"\ntype = "chip.identity"\nn_modes = 2',
            'id = "chip"\ntype = "chip.identity"\nn_modes = 2\n\n'
            '[[pipeline.stages]]\nid = "det"\ntype = "detector.ideal_counter"',
        ),
        encoding="utf-8",
    )

    cells = expand_sweeps(load_config(config_path))
    times = sorted(
        next(
            stage for stage in cell.config.pipeline.stages if stage.id == "det"
        ).model_dump(mode="python")["integration_time_s"]
        for cell in cells
    )
    assert times == [10.0, 20.0]
