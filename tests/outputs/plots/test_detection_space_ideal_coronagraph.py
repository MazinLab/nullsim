"""Guyon 2006 ideal-coronagraph overlay on the detection_space plot."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from matplotlib import pyplot as plt

import nullsim  # noqa: F401  # plot registration side effect
from nullsim.config.loader import resolve_config
from nullsim.outputs import RunCollection, RunRecord
from nullsim.outputs.plots.detection_space import (
    _IDEAL_CORONAGRAPH_COLOR,
    _ideal_coronagraph_curve_inputs,
    _plot_ideal_coronagraph_curves,
)
from nullsim.pipeline import RunResult
from nullsim.pipeline.state import Results, SimulationState
from nullsim.stages.detector.ideal_counter import CONTRAST_CURVE_DTYPE


def _record(
    *,
    orders: tuple[int, ...] = (2, 4),
    n_apertures: int = 1,
) -> RunRecord:
    wavelengths_um = (0.76, 0.80, 0.84)
    raw: dict[str, object] = {
        "run": {"name": "hwo_ideal_corono_test", "seed": 0},
        "grid": {
            "wavelength_center_um": 0.8,
            "wavelength_bandwidth_um": 0.08,
            "n_wavelength_bins": 3,
        },
        "telescope": {
            "n_apertures": n_apertures,
            "aperture": {"diameter_m": 8.0, "obscuration_fraction": 0.0},
        },
        "scene": {
            "star": {"jmag": 2.13, "angular_diameter_mas": 1.86, "distance_pc": 5.0},
            "planet": {
                "contrast": 1.0e-10,
                "separation_mas": 200.0,
                "position_angle_deg": 0.0,
            },
        },
        "observation": {"integration_time_s": 86400.0},
        "output": {
            "detection_space": {"ideal_coronagraph_orders": list(orders)},
        },
        "pipeline": {"stages": [{"id": "scene", "type": "scene.point_source"}]},
    }
    config = resolve_config(raw)

    separations_mas = np.geomspace(5.0, 1000.0, 20)
    curve = np.empty(separations_mas.size, dtype=CONTRAST_CURVE_DTYPE)
    curve["separation_mas"] = separations_mas
    curve["contrast"] = 1.0e-9 * (50.0 / separations_mas)

    base = SimulationState.empty()
    state = replace(
        base,
        scene=replace(base.scene, star_angular_diameter_mas=1.86),
        throughput=replace(base.throughput, wavelength_grid_um=wavelengths_um),
        results=Results(
            null_depth=1.0e-9,
            contrast_curve=curve,
            components={
                "detector.mkid": {
                    "star_magnitude": 1.5,
                    "photometric_band": "I",
                    "detector_qe": 0.9,
                },
                "background.space_zodi": {
                    "zodi_rate_hz_per_dark_port": 2.0,
                    "n_physical_dark_ports": 26,
                },
                "performance.standard": {
                    "contrast_curve_reference_time_s": 86400.0,
                    "target_snr": 5.0,
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
    return RunRecord(
        config=config,
        result=result,
        run_dir=Path("."),
        coord_tuple=(("scene.planet.contrast", 1.0e-10),),
    )


def _collection(record: RunRecord) -> RunCollection:
    return RunCollection(
        config=record.config, records=(record,), output_dir=Path(".")
    )


def test_overlay_draws_one_curve_per_order() -> None:
    record = _record(orders=(2, 4))
    figure, axes = plt.subplots()
    try:
        _plot_ideal_coronagraph_curves(axes, _collection(record))
        overlay = [
            line
            for line in axes.lines
            if line.get_color() == _IDEAL_CORONAGRAPH_COLOR
        ]
        assert len(overlay) == 2
        # Order 4 admits more stellar-disk leakage suppression: at a wide
        # separation its bound must sit below order 2's.
        far_values = []
        for line in overlay:
            x = np.asarray(line.get_xdata())
            y = np.asarray(line.get_ydata())
            far_values.append(float(np.interp(500.0, x, y)))
        assert min(far_values) < max(far_values) / 10.0
    finally:
        plt.close(figure)


def test_overlay_disabled_by_default() -> None:
    record = _record(orders=())
    figure, axes = plt.subplots()
    try:
        _plot_ideal_coronagraph_curves(axes, _collection(record))
        assert not axes.lines
    finally:
        plt.close(figure)


def test_overlay_skips_multi_aperture_configs() -> None:
    record = _record(orders=(2,), n_apertures=2)
    figure, axes = plt.subplots()
    try:
        with pytest.warns(UserWarning, match="single clear primary"):
            _plot_ideal_coronagraph_curves(axes, _collection(record))
        assert not axes.lines
    finally:
        plt.close(figure)


def test_inputs_assemble_perfect_optics_rate() -> None:
    record = _record()
    with pytest.warns(UserWarning, match="zodi surface brightness"):
        inputs = _ideal_coronagraph_curve_inputs(_collection(record))
    assert inputs is not None
    assert inputs["diameter_m"] == pytest.approx(8.0)
    assert inputs["stellar_radius_mas"] == pytest.approx(0.93)
    # Without the surface brightness keys: the routed dark-port fallback.
    assert inputs["background_rate_hz"] == pytest.approx(52.0)
    assert inputs["integration_time_s"] == pytest.approx(86400.0)
    # I=1.5 solar analog through a clear 8 m primary and QE 0.9 lands at
    # a few 1e10 cps; pin the order of magnitude, not the zero point.
    assert 1.0e9 < inputs["stellar_rate_hz"] < 1.0e12


def test_background_is_zodi_through_one_resolution_element_of_the_primary() -> None:
    """Review 2026-11-07 row 56: (local + exozodi) surface brightness times
    primary area, (lambda/D)^2, and QE, not the nuller's routed zodi.

    Hand value: lambda/D = 0.8e-6 / 8 rad = 0.0206265 arcsec, squared
    4.2545e-4 arcsec^2; area pi*16 = 50.265 m^2; (30 + 10) * 50.265 *
    4.2545e-4 * 0.9 = 0.76985 Hz.
    """

    record = _record()
    state = record.result.state
    zodi = {
        **state.results.components["background.space_zodi"],
        "local_zodi_rate_hz_per_m2_per_arcsec2": 30.0,
        "exozodi_rate_hz_per_m2_per_arcsec2_at_scene_separation": 10.0,
    }
    components = {**state.results.components, "background.space_zodi": zodi}
    state = replace(state, results=replace(state.results, components=components))
    record = RunRecord(
        config=record.config,
        result=replace(record.result, state=state),
        run_dir=record.run_dir,
        coord_tuple=record.coord_tuple,
    )

    inputs = _ideal_coronagraph_curve_inputs(_collection(record))

    assert inputs["background_rate_hz"] == pytest.approx(0.76985, rel=1.0e-4)


def test_degenerate_sweep_draws_one_curve_and_one_legend_entry() -> None:
    """Cells identical in label and curve collapse to a single line/entry.

    The HWO config sweeps scene.planet.contrast, which the 5-sigma contrast
    curve does not depend on: pre-fix the figure stacked five identical
    curves and legend entries.
    """

    from nullsim.outputs.plots.detection_space import (
        _attach_legends,
        _plot_detection_curves,
    )

    records = tuple(_record(orders=()) for _ in range(5))
    collection = RunCollection(
        config=records[0].config, records=records, output_dir=Path(".")
    )
    figure, axes = plt.subplots()
    try:
        _plot_detection_curves(axes, collection)
        assert len(axes.lines) == 1
        _attach_legends(axes, (), collection)
        legend = axes.get_legend()
        labels = [text.get_text() for text in legend.get_texts()]
        assert len([lab for lab in labels if "HWO" in lab]) == 1
    finally:
        plt.close(figure)
