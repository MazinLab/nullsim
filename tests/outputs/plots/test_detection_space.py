"""Smoke tests for the headline ``detection_space`` plot (SPEC §8.1, §8.2, §15.4)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib import pyplot as plt

import nullsim  # noqa: F401  # ensures plot registration runs at import time
from nullsim.config import ResolvedConfig
from nullsim.config.loader import resolve_config
from nullsim.outputs import RunCollection, RunRecord
from nullsim.outputs.plots._kilo_planet_overlay import PlotPlanet
from nullsim.outputs.plots.detection_space import (
    detection_space,
    _attach_legends,
    _contrast_curve_legend_title,
    _decorate_axes,
    _load_planet_overlay,
    _max_baseline_m,
    _observability_config,
    _plot_characterization_curves,
    _plot_detection_curves,
    _plot_hdc_curves,
    _valid_curve_segments,
)
from nullsim.pipeline import RunResult, registry
from nullsim.pipeline.state import Results, SimulationState
from nullsim.stages.detector.ideal_counter import CONTRAST_CURVE_DTYPE


_FIXTURE_CATALOG = (
    Path(__file__).resolve().parents[1].parent
    / "catalogs"
    / "fixtures"
    / "sample_catalog.csv"
)


_GRID = {
    "wavelength_center_um": 1.65,
    "wavelength_bandwidth_um": 0.30,
    "n_wavelength_bins": 3,
}


def _resolved_with_catalog(snapshot_path: Path | None = None) -> ResolvedConfig:
    """Build a ResolvedConfig stub carrying a [catalog.exoplanets] block.

    The plot reads the catalog block off ``raw_config``, so the stub must
    populate that field — model_validate's mode='python' returns a frozen
    instance we then ``model_copy`` to inject the catalog.
    """
    raw: dict[str, object] = {
        "run": {"name": "detection_space_test", "seed": 0},
        "grid": _GRID,
        "scene": {
            "star": {"jmag": 5.0, "angular_diameter_mas": 0.5, "distance_pc": 5.0},
            "planet": {
                "contrast": 1.0e-6,
                "separation_mas": 50.0,
                "position_angle_deg": 0.0,
            },
        },
        "observation": {"integration_time_s": 3600.0},
        "pipeline": {"stages": [{"id": "scene", "type": "scene.point_source"}]},
    }
    if snapshot_path is not None:
        raw["catalog"] = {
            "exoplanets": {
                "retrieval_date": "2026-04-01",
                "snapshot_path": str(snapshot_path),
            }
        }
    resolved = resolve_config(raw)
    if snapshot_path is not None:
        # raw_config is the field the plot reads — overwrite with the
        # catalog-bearing dict (resolve_config strips the unknown
        # ``catalog`` table during validation).
        resolved = resolved.model_copy(update={"raw_config": raw})
    return resolved


def _resolved_all_sky_with_catalog(snapshot_path: Path) -> ResolvedConfig:
    raw: dict[str, object] = {
        "run": {"name": "hwo_detection_space_test", "seed": 0},
        "site": {"observability": {"all_sky": True}},
        "grid": _GRID,
        "scene": {
            "star": {"jmag": 5.0, "angular_diameter_mas": 0.5, "distance_pc": 5.0},
            "planet": {
                "contrast": 1.0e-6,
                "separation_mas": 50.0,
                "position_angle_deg": 0.0,
            },
        },
        "observation": {"integration_time_s": 3600.0},
        "catalog": {
            "exoplanets": {
                "retrieval_date": "2026-04-01",
                "snapshot_path": str(snapshot_path),
            }
        },
        "pipeline": {"stages": [{"id": "scene", "type": "scene.point_source"}]},
    }
    return resolve_config(raw).model_copy(update={"raw_config": raw})


def _synthetic_record(jmag: float, contrast_floor: float) -> RunRecord:
    """Build a RunRecord whose Results carries a synthetic contrast curve."""
    separations_mas = np.geomspace(20.0, 1000.0, 25)
    contrasts = contrast_floor * (50.0 / separations_mas) ** 2
    contrasts = np.clip(contrasts, 1e-9, 1e-2)
    curve = np.empty(separations_mas.size, dtype=CONTRAST_CURVE_DTYPE)
    curve["separation_mas"] = separations_mas
    curve["contrast"] = contrasts

    base = SimulationState.empty()
    state = SimulationState(
        scene=base.scene,
        geometry=base.geometry,
        field=base.field,
        opd_budget=base.opd_budget,
        throughput=base.throughput,
        wavefront=base.wavefront,
        photon_rates=base.photon_rates,
        results=Results(
            null_depth=contrast_floor,
            contrast_curve=curve,
            components={
                "performance.standard": {
                    "contrast_curve_reference_time_s": 3600.0,
                },
            },
        ),
    )
    result = RunResult(
        state=state,
        diagnostics={},
        timings=(),
        cache_hit_count=0,
        config_hash="synthetic",
    )
    config = _resolved_with_catalog(_FIXTURE_CATALOG)
    return RunRecord(
        config=config,
        result=result,
        run_dir=Path("."),
        coord_tuple=(("scene.star.jmag", jmag),),
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_detection_space_is_registered() -> None:
    """Importing nullsim must auto-register the headline plot."""
    assert "detection_space" in registry.plots
    assert registry.plots["detection_space"] is detection_space


# ---------------------------------------------------------------------------
# Smoke: 3 sweep cells × 5 catalog points → real PNG/PDF on disk.
# ---------------------------------------------------------------------------


def test_detection_space_writes_png_with_overlay(tmp_path: Path) -> None:
    """Three contrast curves + the 5-planet fixture produce a >10 kB PNG."""
    config = _resolved_with_catalog(_FIXTURE_CATALOG)
    records = tuple(
        _synthetic_record(jmag=jmag, contrast_floor=floor)
        for jmag, floor in ((5.0, 1.0e-7), (8.0, 5.0e-7), (11.0, 5.0e-6))
    )
    collection = RunCollection(config=config, records=records, output_dir=tmp_path)

    paths = detection_space(collection, tmp_path, ("png", "pdf"))

    assert {p.name for p in paths} == {"detection_space.png", "detection_space.pdf"}
    for path in paths:
        assert path.exists()
        assert path.parent == tmp_path / "plots"
    # PNG should comfortably exceed 10 kB given a multi-curve plot + caption.
    png_path = next(p for p in paths if p.suffix == ".png")
    assert png_path.stat().st_size > 10_000


def test_detection_space_works_without_catalog(tmp_path: Path) -> None:
    """No [catalog.exoplanets] block → curves still draw, no scatter overlay."""
    config = _resolved_with_catalog(None)
    record = _synthetic_record(jmag=5.0, contrast_floor=1.0e-7)
    record = RunRecord(
        config=config,
        result=record.result,
        run_dir=record.run_dir,
        coord_tuple=record.coord_tuple,
    )
    collection = RunCollection(config=config, records=(record,), output_dir=tmp_path)

    paths = detection_space(collection, tmp_path, ("png",))

    assert len(paths) == 1
    assert paths[0].exists()
    assert paths[0].stat().st_size > 5_000


def test_detection_space_drops_unphysical_contrast_points() -> None:
    """Contrast tails above 1 are diagnostics, not plottable planet limits."""

    import matplotlib.pyplot as plt

    record = _synthetic_record(jmag=5.0, contrast_floor=1.0e-7)
    curve = np.empty(3, dtype=CONTRAST_CURVE_DTYPE)
    curve["separation_mas"] = (10.0, 30.0, 100.0)
    curve["contrast"] = (1.0e-6, 2.0, np.nan)
    state = replace(
        record.result.state,
        results=Results(null_depth=1.0e-7, contrast_curve=curve),
    )
    record = RunRecord(
        config=record.config,
        result=RunResult(
            state=state,
            diagnostics={},
            timings=(),
            cache_hit_count=0,
            config_hash="synthetic",
        ),
        run_dir=record.run_dir,
        coord_tuple=record.coord_tuple,
    )
    collection = RunCollection(
        config=record.config,
        records=(record,),
        output_dir=Path("."),
    )
    fig, axes = plt.subplots()

    with pytest.warns(UserWarning, match="unphysical"):
        _plot_detection_curves(axes, collection)

    assert len(axes.lines) == 1
    np.testing.assert_allclose(axes.lines[0].get_xdata(), [10.0])
    np.testing.assert_allclose(axes.lines[0].get_ydata(), [1.0e-6])
    plt.close(fig)


def test_detection_space_segments_characterization_anchor_gaps() -> None:
    """Failed char anchors must not be bridged by a misleading curve line."""

    separations = np.asarray([30.0, 1.0, 10.0, 3.0, 100.0], dtype=np.float64)
    contrasts = np.asarray([3.0e-7, 1.0e-5, 1.0e-7, 2.0e-6, 1.0e-6])
    valid = np.asarray([True, True, False, True, True])

    segments = _valid_curve_segments(separations, contrasts, valid)

    assert len(segments) == 2
    np.testing.assert_allclose(segments[0][0], [1.0, 3.0])
    np.testing.assert_allclose(segments[0][1], [1.0e-5, 2.0e-6])
    np.testing.assert_allclose(segments[1][0], [30.0, 100.0])
    np.testing.assert_allclose(segments[1][1], [3.0e-7, 1.0e-6])


def _record_with_char_payload(record: RunRecord, payload: dict) -> RunRecord:
    """Return a copy of ``record`` carrying a synthetic char detection curve."""

    state = replace(
        record.result.state,
        results=replace(
            record.result.state.results,
            components={
                **record.result.state.results.components,
                "characterization.detection_curve": payload,
            },
        ),
    )
    return RunRecord(
        config=record.config,
        result=RunResult(
            state=state,
            diagnostics={},
            timings=(),
            cache_hit_count=0,
            config_hash="synthetic",
        ),
        run_dir=record.run_dir,
        coord_tuple=record.coord_tuple,
    )


def test_detection_space_keeps_anchors_failing_only_full_band() -> None:
    """Anchors that pass the optimizer's design-grid constraint margin but fail
    the stricter full-band static-null margin must still be drawn: the plot
    gates only on the constraint the optimizer actually solved against."""

    import warnings

    record = _synthetic_record(jmag=4.0, contrast_floor=1.0e-7)
    payload = {
        "separations_mas": np.asarray([1.0, 3.0, 10.0], dtype=np.float64),
        "contrast_5sigma": np.asarray([1.0e-5, 1.0e-6, 1.0e-7]),
        "optimization_success_per_anchor": np.asarray([True, True, True]),
        "optimization_constraint_margin_per_anchor": np.asarray(
            [1.0e-8, 1.0e-8, 1.0e-8],
        ),
        "full_band_constraint_margin_per_anchor": np.asarray(
            [-1.0e-6, -2.0e-6, -1.0e-6],
        ),
        "enforce_null_constraint": True,
        "null_floor": 1.0e-6,
    }
    record = _record_with_char_payload(record, payload)
    collection = RunCollection(
        config=record.config,
        records=(record,),
        output_dir=Path("."),
    )
    fig, axes = plt.subplots()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _plot_characterization_curves(axes, collection)
    assert not [w for w in caught if "characterization anchor" in str(w.message)]
    assert len(axes.lines) == 1  # all three anchors form one contiguous curve
    plt.close(fig)


def test_detection_space_still_gates_design_grid_margin() -> None:
    """An anchor that fails the optimizer's own constraint margin is dropped."""

    record = _synthetic_record(jmag=4.0, contrast_floor=1.0e-7)
    payload = {
        "separations_mas": np.asarray([1.0, 3.0, 10.0], dtype=np.float64),
        "contrast_5sigma": np.asarray([1.0e-5, 1.0e-6, 1.0e-7]),
        "optimization_success_per_anchor": np.asarray([True, True, True]),
        "optimization_constraint_margin_per_anchor": np.asarray(
            [1.0e-8, -1.0e-6, 1.0e-8],
        ),
        "enforce_null_constraint": True,
        "null_floor": 1.0e-6,
    }
    record = _record_with_char_payload(record, payload)
    collection = RunCollection(
        config=record.config,
        records=(record,),
        output_dir=Path("."),
    )
    fig, axes = plt.subplots()
    with pytest.warns(UserWarning, match="characterization anchor"):
        _plot_characterization_curves(axes, collection)
    plt.close(fig)


def _record_with_hdc_payload(record: RunRecord, payload: dict) -> RunRecord:
    """Return a copy of ``record`` carrying a synthetic performance.hdc payload."""

    state = replace(
        record.result.state,
        results=replace(
            record.result.state.results,
            components={
                **record.result.state.results.components,
                "performance.hdc": payload,
            },
        ),
    )
    return RunRecord(
        config=record.config,
        result=RunResult(
            state=state,
            diagnostics={},
            timings=(),
            cache_hit_count=0,
            config_hash="synthetic",
        ),
        run_dir=record.run_dir,
        coord_tuple=record.coord_tuple,
    )


def _hdc_payload(**overrides: object) -> dict:
    payload = {
        "separations_mas": np.asarray([1.0, 3.0, 10.0], dtype=np.float64),
        "ccf_contrast_5sigma": np.asarray([1.0e-5, 1.0e-6, 1.0e-7]),
        "broadband_contrast_5sigma": np.asarray([2.0e-5, 2.0e-6, 2.0e-7]),
        # Per-anchor array, as production publishes it. Finite values below 1
        # are the normal photon-limited result, not an error.
        "ccf_contrast_gain": np.asarray([2.0, 0.5, 0.5]),
        "resolving_power_used": 3000.0,
        "comparison_time_s": 3600.0,
        "comparison_target_snr": 5.0,
    }
    payload.update(overrides)
    return payload


def _hdc_collection(payload: dict | None) -> RunCollection:
    record = _synthetic_record(jmag=4.0, contrast_floor=1.0e-7)
    if payload is not None:
        record = _record_with_hdc_payload(record, payload)
    return RunCollection(
        config=record.config,
        records=(record,),
        output_dir=Path("."),
    )


def _legend_labels(axes, collection) -> list[str]:
    """Legend labels after attaching, with a detection curve present so the
    legend exists at all."""

    _plot_detection_curves(axes, collection)
    _attach_legends(axes, (), collection)
    return [text.get_text() for text in axes.get_legend().get_texts()]


def test_detection_space_overlays_hdc_ccf_curve() -> None:
    """The HDC CCF curve is drawn dash-dot, distinct from the dashed
    characterization overlay it sits beside."""

    collection = _hdc_collection(_hdc_payload())
    fig, axes = plt.subplots()

    _plot_hdc_curves(axes, collection)

    assert len(axes.lines) == 1
    line = axes.lines[0]
    assert line.get_linestyle() == "-."
    np.testing.assert_allclose(line.get_xdata(), [1.0, 3.0, 10.0])
    np.testing.assert_allclose(line.get_ydata(), [1.0e-5, 1.0e-6, 1.0e-7])
    plt.close(fig)


def test_detection_space_hdc_legend_label_carries_resolving_power() -> None:
    collection = _hdc_collection(_hdc_payload(resolving_power_used=2500.0))
    fig, axes = plt.subplots()

    _plot_detection_curves(axes, collection)
    _plot_hdc_curves(axes, collection)
    _attach_legends(axes, (), collection)

    labels = [text.get_text() for text in axes.get_legend().get_texts()]
    assert "HDC CCF (R=2500)" in labels
    plt.close(fig)


def test_detection_space_without_hdc_component_is_unchanged() -> None:
    """Runs with no performance.hdc stage draw no HDC overlay and no legend
    entry, and raise nothing."""

    collection = _hdc_collection(None)
    fig, axes = plt.subplots()

    _plot_hdc_curves(axes, collection)
    _plot_detection_curves(axes, collection)
    _attach_legends(axes, (), collection)

    assert len(axes.lines) == 1  # the detection curve alone
    labels = [text.get_text() for text in axes.get_legend().get_texts()]
    assert not [label for label in labels if "HDC CCF" in label]
    plt.close(fig)


def _plotted_separations(axes) -> list[float]:
    """Every drawn x coordinate, from lines and from isolated-anchor scatters."""

    values: list[float] = []
    for line in axes.lines:
        values.extend(np.asarray(line.get_xdata(), dtype=float).tolist())
    for collection in axes.collections:
        offsets = np.asarray(collection.get_offsets(), dtype=float)
        if offsets.size:
            values.extend(offsets[:, 0].tolist())
    return sorted(values)


def test_detection_space_drops_non_finite_hdc_anchors() -> None:
    """NaN CCF anchors are non-detections by design, not failures: drop them
    with a counted warning, plot the rest, and do not bridge the gap."""

    payload = _hdc_payload(
        ccf_contrast_5sigma=np.asarray([1.0e-5, np.nan, 1.0e-7]),
    )
    collection = _hdc_collection(payload)
    fig, axes = plt.subplots()

    with pytest.warns(UserWarning, match="HDC CCF anchor"):
        _plot_hdc_curves(axes, collection)

    assert _plotted_separations(axes) == [1.0, 10.0]
    # The dropped middle anchor splits the curve rather than joining 1 to 10.
    assert not axes.lines
    plt.close(fig)


def test_detection_space_hdc_all_nan_anchors_draws_nothing() -> None:
    """An all-NaN CCF curve (every anchor a non-detection) is not an error."""

    payload = _hdc_payload(ccf_contrast_5sigma=np.full(3, np.nan))
    collection = _hdc_collection(payload)
    fig, axes = plt.subplots()

    with pytest.warns(UserWarning, match="HDC CCF anchor"):
        _plot_hdc_curves(axes, collection)

    assert not axes.lines
    assert not axes.collections
    # No curve drawn means no legend entry: the entry must follow the artist,
    # not the mere presence of a performance.hdc component.
    assert not [label for label in _legend_labels(axes, collection) if "HDC CCF" in label]
    plt.close(fig)


def test_detection_space_hdc_skips_payload_missing_keys(caplog) -> None:
    """A payload without the curve keys is skipped -- logged, no crash, and no
    ghost legend entry."""

    collection = _hdc_collection({"resolving_power_used": 3000.0})
    fig, axes = plt.subplots()

    with caplog.at_level("INFO", logger="root"):
        _plot_hdc_curves(axes, collection)

    assert "no usable separations_mas/ccf_contrast_5sigma pair" in caplog.text
    assert not axes.lines
    assert not axes.collections
    assert not [label for label in _legend_labels(axes, collection) if "HDC CCF" in label]
    plt.close(fig)


def test_detection_space_hdc_requires_a_finite_resolving_power(caplog) -> None:
    """The R-bearing label is the binding contract, so a payload with no finite
    resolving_power_used is a missing required key -- a logged skip, not a
    curve carrying a vaguer label."""

    for missing in (None, float("nan")):
        collection = _hdc_collection(_hdc_payload(resolving_power_used=missing))
        fig, axes = plt.subplots()

        caplog.clear()
        with caplog.at_level("INFO", logger="root"):
            _plot_hdc_curves(axes, collection)

        assert "no finite resolving_power_used" in caplog.text
        assert not axes.lines
        assert not axes.collections
        assert not [
            label for label in _legend_labels(axes, collection) if "HDC CCF" in label
        ]
        plt.close(fig)


def _hdc_collection_from(payloads: tuple[tuple[float, dict], ...]) -> RunCollection:
    records = tuple(
        _record_with_hdc_payload(
            _synthetic_record(jmag=jmag, contrast_floor=1.0e-7), payload
        )
        for jmag, payload in payloads
    )
    return RunCollection(
        config=records[0].config,
        records=records,
        output_dir=Path("."),
    )


def test_detection_space_hdc_r_sweep_draws_one_curve_per_resolving_power() -> None:
    """An R sweep at fixed J and integration time must not have its later cells
    suppressed by a coarser dedupe identity than the legend uses."""

    collection = _hdc_collection_from(
        (
            (4.0, _hdc_payload(resolving_power_used=3000.0)),
            (4.0, _hdc_payload(resolving_power_used=5000.0)),
        )
    )
    fig, axes = plt.subplots()

    _plot_hdc_curves(axes, collection)
    hdc_lines = [
        line for line in axes.lines if str(line.get_label()).startswith("HDC CCF")
    ]
    labels = [label for label in _legend_labels(axes, collection) if "HDC CCF" in label]

    assert len(hdc_lines) == 2
    assert labels == ["HDC CCF (R=3000)", "HDC CCF (R=5000)"]
    plt.close(fig)


def test_detection_space_hdc_records_sharing_r_draw_once() -> None:
    """Two cells at the same resolving power are indistinguishable once drawn,
    so they collapse to one curve and one legend entry -- not one entry with
    two overlapping curves behind it."""

    collection = _hdc_collection_from(
        (
            (4.0, _hdc_payload(resolving_power_used=3000.0)),
            (8.0, _hdc_payload(resolving_power_used=3000.0)),
        )
    )
    fig, axes = plt.subplots()

    _plot_hdc_curves(axes, collection)
    hdc_lines = [
        line for line in axes.lines if str(line.get_label()).startswith("HDC CCF")
    ]
    labels = [label for label in _legend_labels(axes, collection) if "HDC CCF" in label]

    assert len(hdc_lines) == 1
    assert labels == ["HDC CCF (R=3000)"]
    plt.close(fig)


def test_detection_space_uses_paper_jmag_styles() -> None:
    import matplotlib.pyplot as plt

    record_a = _synthetic_record(jmag=4.0, contrast_floor=1.0e-7)
    record_b = _synthetic_record(jmag=8.0, contrast_floor=2.0e-7)
    collection = RunCollection(
        config=record_a.config,
        records=(record_a, record_b),
        output_dir=Path("."),
    )
    fig, axes = plt.subplots()

    _plot_detection_curves(axes, collection)

    assert len(axes.lines) == 2
    assert axes.lines[0].get_color() == axes.lines[1].get_color()
    assert axes.lines[0].get_linestyle() != axes.lines[1].get_linestyle()
    plt.close(fig)


def test_detection_space_x_axis_matches_paper_view() -> None:
    """The paper view starts at half the assumed stellar diameter and ends at 300 mas."""

    import matplotlib.pyplot as plt

    record = _synthetic_record(jmag=5.0, contrast_floor=1.0e-7)
    collection = RunCollection(
        config=record.config,
        records=(record,),
        output_dir=Path("."),
    )
    fig, axes = plt.subplots()

    _plot_detection_curves(axes, collection)
    _decorate_axes(axes, stellar_diameter_mas=0.5)

    assert axes.get_xlim() == pytest.approx((0.25, 300.0))
    plt.close(fig)


def test_detection_space_baseline_prefers_enu_over_pupil() -> None:
    geometry = SimpleNamespace(
        baselines_m=None,
        aperture_positions_pupil_m=np.array([[0.0, 0.0], [10.0, 0.0]]),
        aperture_positions_enu_m=np.array([[0.0, 0.0], [85.0, 0.0]]),
    )

    assert _max_baseline_m(geometry) == pytest.approx(85.0)


def test_detection_space_legend_title_includes_contrast_reference_time(
    tmp_path: Path,
) -> None:
    record = _synthetic_record(5.0, 1.0e-6)
    collection = RunCollection(
        config=record.config,
        records=(record,),
        output_dir=tmp_path,
    )

    title = _contrast_curve_legend_title(collection)

    assert "3600 s" in title


def test_detection_space_handles_empty_records(tmp_path: Path) -> None:
    """Zero sweep cells must still produce a placeholder figure (no crash)."""
    config = _resolved_with_catalog(_FIXTURE_CATALOG)
    collection = RunCollection(config=config, records=(), output_dir=tmp_path)

    paths = detection_space(collection, tmp_path, ("png",))

    assert len(paths) == 1
    assert paths[0].exists()


def test_detection_space_all_sky_observability_keeps_southern_planets(
    tmp_path: Path,
) -> None:
    """Space observatories should not inherit the Mauna Kea declination cut."""

    ground_config = _resolved_with_catalog(_FIXTURE_CATALOG)
    ground_record = _synthetic_record(jmag=5.0, contrast_floor=1.0e-7)
    ground_record = RunRecord(
        config=ground_config,
        result=ground_record.result,
        run_dir=ground_record.run_dir,
        coord_tuple=ground_record.coord_tuple,
    )
    ground_collection = RunCollection(
        config=ground_config,
        records=(ground_record,),
        output_dir=tmp_path,
    )
    ground_obs = _observability_config(ground_collection)
    ground_planets = _load_planet_overlay(
        _FIXTURE_CATALOG,
        required=True,
        observability=ground_obs,
    )

    space_config = _resolved_all_sky_with_catalog(_FIXTURE_CATALOG)
    space_record = RunRecord(
        config=space_config,
        result=ground_record.result,
        run_dir=ground_record.run_dir,
        coord_tuple=ground_record.coord_tuple,
    )
    space_collection = RunCollection(
        config=space_config,
        records=(space_record,),
        output_dir=tmp_path,
    )
    space_obs = _observability_config(space_collection)
    space_planets = _load_planet_overlay(
        _FIXTURE_CATALOG,
        required=True,
        observability=space_obs,
    )

    assert len(ground_planets) == 3
    assert len(space_planets) == 5
    assert space_obs.min_altitude_deg is None


# ---------------------------------------------------------------------------
# run_hash propagation: catalog refresh → new run_hash → new output dir.
# ---------------------------------------------------------------------------


def test_run_hash_changes_with_catalog_snapshot(tmp_path: Path) -> None:
    """Same TOML, two different catalog snapshots → two different run_hash values."""
    from nullsim.cli import _run_hash_with_externals

    snap_a = tmp_path / "snap_a.csv"
    snap_b = tmp_path / "snap_b.csv"
    snap_a.write_text(
        "pl_name,pl_rade,separation_mas,contrast_estimate,discoverymethod\n"
        "Foo,1.0,100.0,1e-7,Imaging\n",
        encoding="utf-8",
    )
    snap_b.write_text(
        "pl_name,pl_rade,separation_mas,contrast_estimate,discoverymethod\n"
        "Foo,1.0,100.0,1e-7,Imaging\nBar,5.0,200.0,1e-6,Transit\n",
        encoding="utf-8",
    )

    resolved_a = _resolved_with_catalog(snap_a)
    resolved_b = _resolved_with_catalog(snap_b)
    hash_a = _run_hash_with_externals(resolved_a, instances=())
    hash_b = _run_hash_with_externals(resolved_b, instances=())
    assert hash_a != hash_b


def test_run_hash_unchanged_without_catalog_block() -> None:
    """No catalog block → behavior identical to the previous baseline (no extra hash mixed in)."""
    from nullsim.cli import _run_hash_with_externals

    resolved_no_catalog = _resolved_with_catalog(None)
    hash_no_catalog = _run_hash_with_externals(resolved_no_catalog, instances=())
    # Re-call must be deterministic.
    assert hash_no_catalog == _run_hash_with_externals(
        resolved_no_catalog, instances=()
    )


def test_detection_space_legend_uses_paper_prose_class_labels(
    tmp_path: Path,
) -> None:
    """Legend shows "Super Earth" / "Ice Giant"; internal names stay KILO-1:1."""
    config = _resolved_with_catalog(_FIXTURE_CATALOG)
    records = (_synthetic_record(jmag=5.0, contrast_floor=1.0e-7),)
    collection = RunCollection(config=config, records=records, output_dir=tmp_path)
    planets = [
        PlotPlanet(
            name="se",
            separation_mas=10.0,
            contrast=1.0e-7,
            planet_type="Super-Earth",
            discovery_method="Transit",
            radius_rearth=2.0,
            radius_is_estimated=False,
        ),
        PlotPlanet(
            name="ig",
            separation_mas=20.0,
            contrast=1.0e-6,
            planet_type="Neptune-like",
            discovery_method="Radial Velocity",
            radius_rearth=4.0,
            radius_is_estimated=False,
        ),
    ]

    fig, axes = plt.subplots()
    try:
        _attach_legends(axes, planets, collection)
        legend = axes.get_legend()
        assert legend is not None
        texts = {text.get_text() for text in legend.get_texts()}
        assert "Super Earth" in texts
        assert "Ice Giant" in texts
        assert "Super-Earth" not in texts
        assert "Neptune-like" not in texts
    finally:
        plt.close(fig)


def test_detection_space_omits_design_context_and_caption_text(tmp_path: Path) -> None:
    """In-figure design-context and caption text were removed 2026-06-12:
    the paper figure caption carries that information instead."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    config = _resolved_with_catalog(_FIXTURE_CATALOG)
    records = (_synthetic_record(jmag=5.0, contrast_floor=1.0e-7),)
    collection = RunCollection(config=config, records=records, output_dir=tmp_path)

    paths = detection_space(collection, tmp_path, ("svg",))
    assert len(paths) == 1
    svg_text = paths[0].read_text(encoding="utf-8")
    assert "uniform disk" not in svg_text
    assert "Assumed stellar diameter" not in svg_text
    assert "exoplanets within" not in svg_text
    plt.close("all")
