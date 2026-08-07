"""Tests for plot/table outputs registered in ``nullsim.outputs``.

Covers WP5: the contrast-curve plot and performance-summary table writers.
The full sweep_results.parquet / manifest.json end-to-end orchestration lives in
``tests/outputs/test_end_to_end.py``.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import nullsim  # noqa: F401  # ensures stage + output autoload
from nullsim.config import ResolvedConfig
from nullsim.outputs import RunCollection, RunRecord
from nullsim.outputs.plots import contrast_curve
from nullsim.outputs.sweep_results import write_sweep_results
from nullsim.outputs.tables import characterization_summary, performance_summary
from nullsim.pipeline import RunResult, SimulationState, registry, run


GRID = {
    "wavelength_center_um": 1.65,
    "wavelength_bandwidth_um": 0.30,
    "n_wavelength_bins": 3,
}


def _trivial_config(
    tmp_path: Path,
    star_jmag: float = 5.0,
    planet_contrast: float = 2.0e-6,
    planet_separation_mas: float = 50.0,
) -> dict[str, object]:
    """Minimal in-memory config that drives the five trivial stages.

    Returned dict mirrors the resolved-config shape consumed by ``run``.
    """
    return {
        "config_hash": f"out-{star_jmag}-{planet_contrast}-{planet_separation_mas}",
        "grid": GRID,
        "run": {"seed": 1},
        "cache": {"enabled": False, "dir": tmp_path / "cache", "version": 1},
        "output": {"dir": tmp_path / "run"},
        "pipeline": {
            "stages": [
                {
                    "id": "scene",
                    "type": "scene.point_source",
                    "star_jmag": star_jmag,
                    "planet_contrast": planet_contrast,
                    "planet_separation_mas": planet_separation_mas,
                },
                {
                    "id": "array",
                    "type": "telescope.fixed_aperture",
                    "diameter_m": 2.0,
                },
                {"id": "injection", "type": "injection.ideal"},
                {"id": "chip", "type": "chip.identity", "insertion_loss_db": 0.0},
                {"id": "detector", "type": "detector.ideal_counter"},
            ],
        },
    }


def _resolved_stub(
    config_hash: str = "stub-config",
    run_hash: str = "stub-run",
    run_name: str = "stub_run",
) -> ResolvedConfig:
    """Build a minimal ResolvedConfig used as a label/metadata carrier."""
    from nullsim.config.loader import resolve_config

    raw = {
        "run": {"name": run_name, "seed": 0},
        "grid": GRID,
        "scene": {
            "star": {"jmag": 5.0, "angular_diameter_mas": 1.0, "distance_pc": 5.0},
            "planet": {
                "contrast": 2.0e-6,
                "separation_mas": 50.0,
                "position_angle_deg": 0.0,
            },
        },
        "observation": {"integration_time_s": 10.0},
        "pipeline": {"stages": [{"id": "scene", "type": "scene.point_source"}]},
    }
    resolved = resolve_config(raw)
    # ResolvedConfig is frozen, so produce a copy with overridden hashes via
    # model_copy. This keeps the metadata pieces predictable for assertions.
    return resolved.model_copy(
        update={"config_hash": config_hash, "run_hash": run_hash},
    )


def _record(tmp_path: Path, star_jmag: float = 5.0) -> RunRecord:
    config = _resolved_stub()
    result = run(_trivial_config(tmp_path, star_jmag=star_jmag))
    return RunRecord(
        config=config,
        result=result,
        run_dir=tmp_path,
        coord_tuple=(("scene.star.jmag", star_jmag),),
    )


# ---------------------------------------------------------------------------
# Registration: both outputs autoload at import time and live in the registry.
# ---------------------------------------------------------------------------

def test_contrast_curve_and_performance_summary_are_registered() -> None:
    """Importing nullsim must auto-register both default outputs."""
    assert "contrast_curve" in registry.plots
    assert "performance_summary" in registry.tables
    assert registry.plots["contrast_curve"] is contrast_curve
    assert registry.tables["performance_summary"] is performance_summary


def test_sweep_results_writes_all_component_numeric_scalars(tmp_path: Path) -> None:
    """Result components should not need output-specific allow lists."""

    state = SimulationState.empty()
    results = state.results
    for name, payload in (
        ("performance.pdi", {"snr": 12.5, "eps_pdi": np.float64(0.03)}),
        ("fringe_tracking.closed_loop", {"residual_opd_nm": 1.2}),
        (
            "polarization.pm_fiber",
            {"nested": {"phase_rms_rad": 0.04}, "jones_matrices": np.ones((2, 2, 2))},
        ),
        (
            "detector.snspd",
            {"n_physical_dark_ports": 4, "calibration_floor_applied": True},
        ),
    ):
        results = results.add_component(name, payload)
    state = replace(state, results=results)
    result = RunResult(
        state=state,
        diagnostics={},
        timings=(),
        cache_hit_count=0,
        config_hash="cfg",
    )
    config = _resolved_stub(run_hash="cell-run")
    collection = RunCollection(
        config=config,
        records=(
            RunRecord(
                config=config,
                result=result,
                run_dir=tmp_path / "cell",
                coord_tuple=(("scene.star.jmag", 5.0),),
            ),
        ),
        output_dir=tmp_path,
    )

    output_path = write_sweep_results(collection, tmp_path, "csv")
    frame = pd.read_csv(output_path)
    rows = {
        (str(row.stage_instance), str(row.metric_name)): float(row.metric_value)
        for row in frame.itertuples()
    }

    assert rows[("performance.pdi", "snr")] == pytest.approx(12.5)
    assert rows[("performance.pdi", "eps_pdi")] == pytest.approx(0.03)
    assert rows[("fringe_tracking.closed_loop", "residual_opd_nm")] == pytest.approx(1.2)
    assert rows[("polarization.pm_fiber", "nested.phase_rms_rad")] == pytest.approx(0.04)
    assert rows[("detector.snspd", "n_physical_dark_ports")] == pytest.approx(4.0)
    assert ("polarization.pm_fiber", "jones_matrices") not in rows
    assert ("detector.snspd", "calibration_floor_applied") not in rows
    assert set(frame["cell_run_hash"]) == {"cell-run"}


def test_sweep_results_uses_separation_column_for_contrast_rows(
    tmp_path: Path,
) -> None:
    config = _resolved_stub(run_hash="cell-run")
    record = _record(tmp_path)
    collection = RunCollection(
        config=config,
        records=(
            RunRecord(
                config=config,
                result=record.result,
                run_dir=tmp_path / "cell",
                coord_tuple=(("scene.star.jmag", 5.0),),
            ),
        ),
        output_dir=tmp_path,
    )

    output_path = write_sweep_results(collection, tmp_path, "csv")
    frame = pd.read_csv(output_path)
    contrast_rows = frame[frame["stage_instance"] == "contrast_curve"]

    assert not contrast_rows.empty
    assert contrast_rows["separation_mas"].notna().all()
    assert contrast_rows["wavelength_bin_um"].isna().all()


# ---------------------------------------------------------------------------
# Plot output: contrast_curve.
# ---------------------------------------------------------------------------

def test_contrast_curve_writes_requested_formats(tmp_path: Path) -> None:
    """Each requested file format must produce a non-empty plot file."""
    collection = RunCollection(
        config=_resolved_stub(),
        records=(_record(tmp_path),),
        output_dir=tmp_path,
    )

    written_paths = contrast_curve(collection, tmp_path, ("png", "pdf"))

    assert {p.name for p in written_paths} == {"contrast_curve.png", "contrast_curve.pdf"}
    for path in written_paths:
        assert path.exists()
        assert path.stat().st_size > 0
        assert path.parent.name == "plots"


def test_contrast_curve_handles_empty_records_gracefully(tmp_path: Path) -> None:
    """No records still produces a placeholder file (test that it does not crash)."""
    collection = RunCollection(
        config=_resolved_stub(),
        records=(),
        output_dir=tmp_path,
    )

    written_paths = contrast_curve(collection, tmp_path, ("png",))

    assert len(written_paths) == 1
    assert written_paths[0].exists()


def test_contrast_curve_multi_record_legend_uses_coord(tmp_path: Path) -> None:
    """Multi-record runs label curves with the sweep coordinate tuple."""
    records = (_record(tmp_path / "a", 5.0), _record(tmp_path / "b", 6.0))
    collection = RunCollection(
        config=_resolved_stub(),
        records=records,
        output_dir=tmp_path,
    )

    written_paths = contrast_curve(collection, tmp_path, ("png",))

    # Both records had non-empty contrast curves, so the legend code path runs.
    assert written_paths[0].exists()
    assert written_paths[0].stat().st_size > 0


# ---------------------------------------------------------------------------
# Table output: performance_summary.
# ---------------------------------------------------------------------------

def test_performance_summary_csv_row_per_record(tmp_path: Path) -> None:
    """CSV output must have one row per RunRecord and include core columns."""
    records = (_record(tmp_path / "a", 5.0), _record(tmp_path / "b", 6.0))
    collection = RunCollection(
        config=_resolved_stub(run_hash="rh"),
        records=records,
        output_dir=tmp_path,
    )

    paths = performance_summary(collection, tmp_path, ("csv",))

    assert len(paths) == 1
    frame = pd.read_csv(paths[0])
    assert len(frame) == 2
    for expected_column in ("config_hash", "run_hash", "null_depth", "cache_hit_count"):
        assert expected_column in frame.columns
    # Sweep coord must also appear as a column.
    assert "scene.star.jmag" in frame.columns
    assert sorted(frame["scene.star.jmag"].tolist()) == [5.0, 6.0]


def test_performance_summary_json_is_sorted_and_parseable(tmp_path: Path) -> None:
    """JSON output must be sorted (deterministic) and parse to list of dicts."""
    collection = RunCollection(
        config=_resolved_stub(),
        records=(_record(tmp_path),),
        output_dir=tmp_path,
    )

    paths = performance_summary(collection, tmp_path, ("json",))

    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert len(payload) == 1
    assert payload[0]["null_depth"] == pytest.approx(2.0e-6)
    # sort_keys=True must produce alphabetical key order.
    first = payload[0]
    assert list(first.keys()) == sorted(first.keys())


def test_performance_summary_parquet_round_trip(tmp_path: Path) -> None:
    """Parquet output must round-trip through pandas without loss."""
    collection = RunCollection(
        config=_resolved_stub(),
        records=(_record(tmp_path),),
        output_dir=tmp_path,
    )

    paths = performance_summary(collection, tmp_path, ("parquet",))

    frame = pd.read_parquet(paths[0])
    assert len(frame) == 1
    assert "null_depth" in frame.columns


def test_characterization_summaries_prefer_band_evaluated_leakage(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path)
    state = record.result.state
    results = state.results.add_component(
        "characterization.spectro",
        {
            "snr": 1.0,
            "max_spectral_resolution": 0.0,
            "spectro_throughput": 0.1,
            "spectro_null_depth": 1.0e-6,
            "spectro_band_on_axis_null_depth": 2.0e-5,
            "spectro_disk_leak": 2.0e-6,
            "spectro_band_disk_leak": 1.0e-3,
            "spectro_leak_fraction": 1.2e-3,
            "optimization_success": True,
            "optimization_constraint_margin": 1.0e-7,
        },
    )
    patched_result = RunResult(
        state=replace(state, results=results),
        diagnostics=record.result.diagnostics,
        timings=record.result.timings,
        cache_hit_count=record.result.cache_hit_count,
        config_hash=record.result.config_hash,
    )
    patched_record = RunRecord(
        config=record.config,
        result=patched_result,
        run_dir=record.run_dir,
        coord_tuple=record.coord_tuple,
    )
    collection = RunCollection(
        config=_resolved_stub(),
        records=(patched_record,),
        output_dir=tmp_path,
    )

    performance_path = performance_summary(collection, tmp_path, ("json",))[0]
    characterization_path = characterization_summary(collection, tmp_path, ("json",))[0]

    performance = json.loads(performance_path.read_text(encoding="utf-8"))[0]
    characterization = json.loads(characterization_path.read_text(encoding="utf-8"))[0]
    assert performance["char_spectro_null_depth"] == pytest.approx(2.0e-5)
    assert performance["char_spectro_disk_leak"] == pytest.approx(1.0e-3)
    assert performance["char_spectro_design_null_depth"] == pytest.approx(1.0e-6)
    assert characterization["spectro_null_depth"] == pytest.approx(2.0e-5)
    assert characterization["spectro_disk_leak"] == pytest.approx(1.0e-3)
    assert characterization["spectro_design_null_depth"] == pytest.approx(1.0e-6)
    assert characterization["spectro_design_disk_leak"] == pytest.approx(2.0e-6)


def test_characterization_summaries_mask_infeasible_snr_and_rates(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path)
    state = record.result.state
    results = state.results.add_component(
        "characterization.spectro",
        {
            "snr": 9.0,
            "max_spectral_resolution": 100.0,
            "planet_signal_rate_hz": 1.0,
            "leakage_rate_hz": 2.0,
            "optimization_success": False,
        },
    )
    patched_record = RunRecord(
        config=record.config,
        result=RunResult(
            state=replace(state, results=results),
            diagnostics=record.result.diagnostics,
            timings=record.result.timings,
            cache_hit_count=record.result.cache_hit_count,
            config_hash=record.result.config_hash,
        ),
        run_dir=record.run_dir,
        coord_tuple=record.coord_tuple,
    )
    collection = RunCollection(
        config=_resolved_stub(),
        records=(patched_record,),
        output_dir=tmp_path,
    )

    performance_path = performance_summary(collection, tmp_path, ("json",))[0]
    characterization_path = characterization_summary(collection, tmp_path, ("json",))[0]
    performance = json.loads(performance_path.read_text(encoding="utf-8"))[0]
    characterization = json.loads(characterization_path.read_text(encoding="utf-8"))[0]

    assert performance["char_converged"] is False
    assert characterization["converged"] is False
    assert math.isnan(performance["char_snr"])
    assert math.isnan(characterization["snr"])
    assert math.isnan(performance["char_planet_signal_rate_hz"])
    assert math.isnan(characterization["planet_signal_rate_hz"])


# One distinct sentinel per published hdc_* column, so a test that asserts all
# five is asserting five independent values rather than five views of one.
_HDC_SENTINELS: dict[str, tuple[str, float]] = {
    "hdc_ccf_snr": ("ccf_snr", 12.5),
    "hdc_ccf_snr_photon_limited": ("ccf_snr_photon_limited", 18.0),
    "hdc_n_effective_lines": ("n_effective_lines", 340.0),
    "hdc_resolving_power": ("resolving_power_used", 25000.0),
    "hdc_systematic_rejection": ("ccf_systematic_rejection", 3.0e-3),
}


def _hdc_sentinel_payload(
    upstream: str = "characterization.spectro",
) -> dict[str, object]:
    """A fully populated hdc_ccf payload: every published column has a value."""
    payload: dict[str, object] = {
        key: value for key, value in _HDC_SENTINELS.values()
    }
    payload["upstream_spectro_component"] = upstream
    return payload


def _characterization_record(
    tmp_path: Path,
    extra_components: dict[str, dict[str, object]] | None = None,
    optimization_success: bool = True,
) -> RunRecord:
    """A record carrying a characterization.spectro row plus optional extras."""
    record = _record(tmp_path)
    state = record.result.state
    results = state.results.add_component(
        "characterization.spectro",
        {
            "snr": 5.0,
            "max_spectral_resolution": 100.0,
            "optimization_success": optimization_success,
        },
    )
    for name, payload in (extra_components or {}).items():
        results = results.add_component(name, payload)
    return RunRecord(
        config=record.config,
        result=RunResult(
            state=replace(state, results=results),
            diagnostics=record.result.diagnostics,
            timings=record.result.timings,
            cache_hit_count=record.result.cache_hit_count,
            config_hash=record.result.config_hash,
        ),
        run_dir=record.run_dir,
        coord_tuple=record.coord_tuple,
    )


def test_characterization_summary_publishes_hdc_ccf_columns(tmp_path: Path) -> None:
    """HDC CCF scalars must surface as hdc_* columns on the spectro row."""
    record = _characterization_record(
        tmp_path, {"characterization.hdc_ccf": _hdc_sentinel_payload()}
    )
    collection = RunCollection(
        config=_resolved_stub(), records=(record,), output_dir=tmp_path
    )

    path = characterization_summary(collection, tmp_path, ("json",))[0]
    row = json.loads(path.read_text(encoding="utf-8"))[0]

    for column, (_key, expected) in _HDC_SENTINELS.items():
        assert row[column] == pytest.approx(expected), column


def test_characterization_summary_hdc_columns_nan_without_component(
    tmp_path: Path,
) -> None:
    """Without an hdc_ccf component the hdc_* columns degrade to NaN."""
    collection = RunCollection(
        config=_resolved_stub(),
        records=(_characterization_record(tmp_path),),
        output_dir=tmp_path,
    )

    path = characterization_summary(collection, tmp_path, ("json",))[0]
    row = json.loads(path.read_text(encoding="utf-8"))[0]

    for column in _HDC_SENTINELS:
        assert math.isnan(row[column]), column


def test_characterization_summary_hdc_columns_mask_on_spectro_nonconvergence(
    tmp_path: Path,
) -> None:
    """HDC scalars are built from the spectro solution, so they mask with it.

    A failed retune makes a finite CCF number describe a candidate solution the
    pipeline never accepted, so the hdc_* columns follow the same masking as
    ``snr`` and the per-port rates on that row.

    Both branches run off the same fully populated payload, so the NaNs are
    attributable to the convergence gate alone: every one of the five columns
    demonstrably carries a value when the spectro row converges.
    """
    rows: dict[bool, dict[str, object]] = {}
    for converged in (True, False):
        record = _characterization_record(
            tmp_path,
            {"characterization.hdc_ccf": _hdc_sentinel_payload()},
            optimization_success=converged,
        )
        collection = RunCollection(
            config=_resolved_stub(), records=(record,), output_dir=tmp_path
        )
        path = characterization_summary(collection, tmp_path, ("json",))[0]
        rows[converged] = json.loads(path.read_text(encoding="utf-8"))[0]

    assert rows[True]["converged"] is True
    assert rows[False]["converged"] is False
    for column, (_key, expected) in _HDC_SENTINELS.items():
        assert rows[True][column] == pytest.approx(expected), column
        assert math.isnan(rows[False][column]), column


def test_characterization_summary_hdc_columns_join_on_upstream_component(
    tmp_path: Path,
) -> None:
    """Each spectro row takes only the HDC result computed from that row."""
    record = _record(tmp_path)
    state = record.result.state
    # Second-instance naming follows stages/_utils.unique_component_name:
    # base name first, then base.<instance_id>.
    results = state.results
    for suffix, snr in (("", 5.0), (".spectro_b", 6.0)):
        results = results.add_component(
            f"characterization.spectro{suffix}",
            {"snr": snr, "optimization_success": True},
        )
    for suffix, ccf_snr in (("", 11.0), (".hdc_b", 22.0)):
        results = results.add_component(
            f"characterization.hdc_ccf{suffix}",
            {
                "ccf_snr": ccf_snr,
                "upstream_spectro_component": (
                    "characterization.spectro"
                    if suffix == ""
                    else "characterization.spectro.spectro_b"
                ),
            },
        )
    patched = RunRecord(
        config=record.config,
        result=RunResult(
            state=replace(state, results=results),
            diagnostics=record.result.diagnostics,
            timings=record.result.timings,
            cache_hit_count=record.result.cache_hit_count,
            config_hash=record.result.config_hash,
        ),
        run_dir=record.run_dir,
        coord_tuple=record.coord_tuple,
    )
    collection = RunCollection(
        config=_resolved_stub(), records=(patched,), output_dir=tmp_path
    )

    path = characterization_summary(collection, tmp_path, ("json",))[0]
    rows = json.loads(path.read_text(encoding="utf-8"))

    by_component = {row["component"]: row for row in rows}
    assert set(by_component) == {
        "characterization.spectro",
        "characterization.spectro.spectro_b",
    }
    assert by_component["characterization.spectro"]["hdc_ccf_snr"] == pytest.approx(
        11.0
    )
    assert by_component["characterization.spectro.spectro_b"][
        "hdc_ccf_snr"
    ] == pytest.approx(22.0)


def test_characterization_summary_hdc_columns_fall_back_without_upstream_key(
    tmp_path: Path,
) -> None:
    """Payloads predating upstream_spectro_component still populate the row."""
    record = _characterization_record(
        tmp_path, {"characterization.hdc_ccf": {"ccf_snr": 9.0}}
    )
    collection = RunCollection(
        config=_resolved_stub(), records=(record,), output_dir=tmp_path
    )

    path = characterization_summary(collection, tmp_path, ("json",))[0]
    row = json.loads(path.read_text(encoding="utf-8"))[0]

    assert row["hdc_ccf_snr"] == pytest.approx(9.0)


def test_performance_summary_rejects_unknown_format(tmp_path: Path) -> None:
    """An unsupported file format must raise a clear ValueError."""
    collection = RunCollection(
        config=_resolved_stub(),
        records=(_record(tmp_path),),
        output_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="xlsx"):
        performance_summary(collection, tmp_path, ("xlsx",))


def test_performance_summary_directory_is_tables_subdir(tmp_path: Path) -> None:
    """Tables are written under <output_dir>/tables (matching the SPEC layout)."""
    collection = RunCollection(
        config=_resolved_stub(),
        records=(_record(tmp_path),),
        output_dir=tmp_path,
    )

    paths = performance_summary(collection, tmp_path, ("json",))

    assert paths[0].parent == tmp_path / "tables"
    assert paths[0].parent.is_dir()


# ---------------------------------------------------------------------------
# RunRecord helpers.
# ---------------------------------------------------------------------------

def test_run_record_coord_mapping_round_trips_tuple() -> None:
    """RunRecord.coord_mapping must materialize the coord tuple as a dict."""
    config = _resolved_stub()
    record = RunRecord(
        config=config,
        result=None,  # type: ignore[arg-type]  # not used in this assertion
        run_dir=Path("/tmp"),
        coord_tuple=(("x", 1), ("y", 2.5)),
    )

    assert record.coord_mapping == {"x": 1, "y": 2.5}


def test_run_collection_records_are_iterable_tuple(tmp_path: Path) -> None:
    """RunCollection.records is a tuple so order is preserved across iterations."""
    records = (_record(tmp_path / "a"), _record(tmp_path / "b"))
    collection = RunCollection(
        config=_resolved_stub(),
        records=records,
        output_dir=tmp_path,
    )

    assert isinstance(collection.records, tuple)
    assert len(collection.records) == 2
    assert collection.records[0] is records[0]


def test_band_avg_throughput_means_finite_per_wavelength() -> None:
    """Band-averaged throughput is the mean over finite per-wavelength values;
    None when the per-wavelength throughput is unavailable."""

    from nullsim.outputs.tables import _band_avg_throughput

    assert _band_avg_throughput(
        {"spectro_planet_throughput_per_wavelength": [0.2, 0.4, 0.6]}
    ) == pytest.approx(0.4)
    assert _band_avg_throughput({}) is None
    assert _band_avg_throughput(
        {"spectro_planet_throughput_per_wavelength": [float("nan"), 0.5]}
    ) == pytest.approx(0.5)
