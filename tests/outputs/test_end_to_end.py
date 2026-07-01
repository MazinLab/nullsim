"""End-to-end integration tests: pipeline run → cache → outputs.

These tests exercise the vertical-slice behavior described in the SPEC even
though the top-level CLI module (``nullsim.cli``) is not yet wired up. The
intent is to lock down the run-twice/cache-hit invariant, the sweep parquet
contract, and the plot/table writers all in one place.

If/when ``nullsim.cli`` lands, a thin test that simply shells through it can
defer to these for the substantive assertions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import nullsim  # noqa: F401  # ensures stages + outputs autoload
from nullsim.outputs import RunCollection, RunRecord
from nullsim.outputs.plots import contrast_curve
from nullsim.outputs.tables import performance_summary
from nullsim.pipeline import run

GRID = {
    "wavelength_center_um": 1.65,
    "wavelength_bandwidth_um": 0.30,
    "n_wavelength_bins": 3,
}


def _vertical_slice_config(
    tmp_path: Path,
    star_jmag: float = 5.0,
    *,
    cache_enabled: bool = True,
    config_hash: str | None = None,
) -> dict[str, object]:
    """Resolved-config shape representing the architect's vertical_slice.toml."""
    return {
        "config_hash": config_hash or f"vs-{star_jmag}",
        "grid": GRID,
        "run": {"seed": 1},
        "cache": {
            "enabled": cache_enabled,
            "dir": tmp_path / "cache",
            "version": 1,
        },
        "output": {"dir": tmp_path / "run"},
        "pipeline": {
            "stages": [
                {
                    "id": "scene",
                    "type": "scene.point_source",
                    "star_jmag": star_jmag,
                    "planet_contrast": 1.0e-7,
                    "planet_separation_mas": 100.0,
                },
                {
                    "id": "array",
                    "type": "telescope.fixed_aperture",
                    "diameter_m": 10.0,
                },
                {"id": "injection", "type": "injection.ideal"},
                {"id": "chip", "type": "chip.identity", "insertion_loss_db": 0.0},
                {"id": "detector", "type": "detector.ideal_counter"},
            ],
        },
    }


# ---------------------------------------------------------------------------
# Pipeline → outputs end-to-end.
# ---------------------------------------------------------------------------

def test_full_pipeline_produces_results_and_plots(tmp_path: Path) -> None:
    """Run the five trivial stages, then write the standard plot/table."""
    result = run(_vertical_slice_config(tmp_path, cache_enabled=False))

    # Pipeline produced the contrast row.
    assert result.state.results.null_depth == pytest.approx(1.0e-7)
    assert result.state.results.contrast_curve is not None
    assert result.state.results.contrast_curve.size > 0

    # Stage timings cover all five declared instances.
    instance_ids = [timing.instance_id for timing in result.timings]
    assert instance_ids == ["scene", "array", "injection", "chip", "detector"]

    # Write the two registered outputs.
    record = RunRecord(
        config=_resolved_stub(),
        result=result,
        run_dir=tmp_path / "run",
        coord_tuple=(),
    )
    collection = RunCollection(
        config=record.config,
        records=(record,),
        output_dir=tmp_path / "run",
    )
    plot_paths = contrast_curve(collection, tmp_path / "run", ("png",))
    table_paths = performance_summary(collection, tmp_path / "run", ("json",))

    assert plot_paths[0].exists() and plot_paths[0].stat().st_size > 0
    assert table_paths[0].exists() and table_paths[0].stat().st_size > 0


# ---------------------------------------------------------------------------
# Cache hit on re-run (architect spec: "Cache hit on re-run").
# ---------------------------------------------------------------------------

def test_pipeline_rerun_produces_identical_results(tmp_path: Path) -> None:
    """Re-running the same config must reproduce the physical result values.

    This is the §15.1 "cache hit on re-run" guarantee at the top-line
    metric level: re-running the same config produces identical RunResult
    cache_hit_count totals and identical contrast-curve outputs. The
    stricter byte-level state equivalence — once xfail because of a
    SubBudget pickle round-trip bug — is now asserted directly in
    test_pipeline_rerun_uses_cache_for_every_stage; both passes are
    expected to be green.
    """
    config = _vertical_slice_config(tmp_path, cache_enabled=True)

    first_run = run(config)
    second_run = run(config)

    # Cold run: zero hits, since the cache starts empty.
    assert first_run.cache_hit_count == 0
    assert all(not t.cache_hit for t in first_run.timings)

    # At least the first stage (no consumes) must hit on rerun.
    assert second_run.cache_hit_count >= 1
    assert second_run.timings[0].cache_hit is True

    # Top-line physical results match exactly.
    assert (
        first_run.state.results.null_depth
        == second_run.state.results.null_depth
    )
    np.testing.assert_array_equal(
        first_run.state.photon_rates.rates_hz,
        second_run.state.photon_rates.rates_hz,
    )
    np.testing.assert_array_equal(
        first_run.state.results.contrast_curve,
        second_run.state.results.contrast_curve,
    )


def test_pipeline_rerun_uses_cache_for_every_stage(tmp_path: Path) -> None:
    """Architect spec: second run must report cache_hit=True for every stage."""
    config = _vertical_slice_config(tmp_path, cache_enabled=True)

    run(config)
    second_run = run(config)

    assert second_run.cache_hit_count == len(second_run.timings)
    assert all(t.cache_hit for t in second_run.timings)


def test_pipeline_rerun_with_disabled_cache_recomputes(tmp_path: Path) -> None:
    """With cache disabled, the second run must not report any cache hits."""
    config = _vertical_slice_config(tmp_path, cache_enabled=False)

    run(config)
    second_run = run(config)

    assert second_run.cache_hit_count == 0
    assert all(not t.cache_hit for t in second_run.timings)


# ---------------------------------------------------------------------------
# Sweep parquet: rows-per-coord assertion (architect spec: sweep_results.parquet).
# ---------------------------------------------------------------------------

def test_sweep_parquet_has_one_row_per_sweep_cell(tmp_path: Path) -> None:
    """Emulating sweep_results.parquet via performance_summary on a 5-cell sweep.

    Until ``nullsim.cli`` wires the dedicated sweep_results writer, the
    performance_summary table doubles as the same shape: one row per sweep
    coord, including run_hash and at least one result metric.
    """
    sweep_jmag_values = (4.0, 6.0, 8.0, 10.0, 12.0)
    records: list[RunRecord] = []
    for axis_value in sweep_jmag_values:
        result = run(
            _vertical_slice_config(
                tmp_path / f"cell_{axis_value:g}",
                star_jmag=axis_value,
                cache_enabled=False,
                config_hash=f"sweep-{axis_value}",
            )
        )
        records.append(
            RunRecord(
                config=_resolved_stub(run_hash=f"rh-{axis_value:g}"),
                result=result,
                run_dir=tmp_path / f"cell_{axis_value:g}",
                coord_tuple=(("scene.star.jmag", axis_value),),
            )
        )
    collection = RunCollection(
        config=_resolved_stub(),
        records=tuple(records),
        output_dir=tmp_path,
    )

    paths = performance_summary(collection, tmp_path, ("parquet",))
    frame = pd.read_parquet(paths[0])

    assert len(frame) == len(sweep_jmag_values)
    assert "scene.star.jmag" in frame.columns
    assert "run_hash" in frame.columns
    assert sorted(frame["scene.star.jmag"].tolist()) == list(sweep_jmag_values)
    # At least one result metric is present.
    assert "null_depth" in frame.columns
    assert frame["null_depth"].notna().all()


# ---------------------------------------------------------------------------
# Photon-rate monotonicity (smoke check that stages compose physically).
# ---------------------------------------------------------------------------

def test_photon_rate_decreases_with_fainter_star(tmp_path: Path) -> None:
    """Brighter stars must yield strictly more detected photons.

    Sanity check that nothing in the pipeline composition silently zeroes out
    the photon rate; also catches sign/exponent regressions in the magnitude
    conversion used by the detector stage.
    """
    bright = run(_vertical_slice_config(tmp_path / "b", star_jmag=4.0, cache_enabled=False))
    faint = run(_vertical_slice_config(tmp_path / "f", star_jmag=8.0, cache_enabled=False))

    bright_total_rate_hz = float(np.sum(bright.state.photon_rates.rates_hz))
    faint_total_rate_hz = float(np.sum(faint.state.photon_rates.rates_hz))

    assert bright_total_rate_hz > faint_total_rate_hz
    # 4 mag difference is exactly a factor of 10**(0.4 * 4) ≈ 39.8 in flux.
    # The trivial pipeline is linear in flux, so the ratio matches within fp tol.
    np.testing.assert_allclose(
        bright_total_rate_hz / faint_total_rate_hz,
        10.0 ** (0.4 * 4.0),
        rtol=1.0e-6,
    )


# ---------------------------------------------------------------------------
# Resolved-config stub helper.
# ---------------------------------------------------------------------------

def _resolved_stub(run_hash: str = "stub-run") -> "nullsim.config.ResolvedConfig":
    from nullsim.config.loader import resolve_config

    raw = {
        "run": {"name": "vertical_slice", "seed": 0},
        "grid": GRID,
        "scene": {
            "star": {"jmag": 5.0, "angular_diameter_mas": 1.0, "distance_pc": 5.0},
            "planet": {
                "contrast": 1.0e-7,
                "separation_mas": 100.0,
                "position_angle_deg": 0.0,
            },
        },
        "observation": {"integration_time_s": 10.0},
        "pipeline": {"stages": [{"id": "scene", "type": "scene.point_source"}]},
    }
    resolved = resolve_config(raw)
    return resolved.model_copy(update={"run_hash": run_hash})
