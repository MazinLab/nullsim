"""Tests for the contrast_curves_multiband figure (KILO paperv3 Fig 10)."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

import nullsim  # noqa: F401
from nullsim.outputs.plots.contrast_curves_multiband import (
    _RecordView,
    _band_label,
    _sweep_label,
    _time_label,
    plot_contrast_curves_multiband,
)


def test_band_label_jh_recognized() -> None:
    name, color, sort_key = _band_label(1.25)
    assert name == "J"
    assert color == "C1"
    assert sort_key == pytest.approx(1.25)


def test_band_label_y_recognized() -> None:
    assert _band_label(1.0)[0] == "Y"


def test_band_label_i_recognized() -> None:
    assert _band_label(0.8)[0] == "I"


def test_band_label_h_recognized() -> None:
    assert _band_label(1.65)[0] == "H"


def test_band_label_unknown_returns_wavelength() -> None:
    name, _, _ = _band_label(0.5)
    assert "um" in name


def test_time_label_hours_round() -> None:
    assert _time_label(3600.0) == "1h"
    assert _time_label(14400.0) == "4h"
    assert _time_label(28800.0) == "8h"


def test_time_label_fractional_hour() -> None:
    assert _time_label(5400.0) == "1.5h"


def _make_view(
    band: str,
    color: str,
    wl: float,
    t_s: float,
    contrast_floor: float,
    *,
    sweep_label: str = "",
    sweep_sort_key: float = 0.0,
):
    sep = np.logspace(np.log10(0.3), np.log10(300), 20)
    return _RecordView(
        band_label=band, band_color=color, band_sort_key=wl,
        time_label=_time_label(t_s), time_sort_key=t_s,
        sweep_label=sweep_label, sweep_sort_key=sweep_sort_key,
        separations_mas=sep, contrast=contrast_floor * np.ones_like(sep),
    )


def test_render_multi_band_multi_time_grid(tmp_path: Path) -> None:
    views = [
        _make_view(band, color, wl, t, floor)
        for band, color, wl in (("Y", "C0", 1.0), ("J", "C1", 1.25), ("H", "C3", 1.65))
        for t in (3600.0, 14400.0, 28800.0)
        for floor in (1e-6,)  # one curve per (band, time)
    ]
    out = tmp_path / "multiband.pdf"
    fig = plot_contrast_curves_multiband(views, save_path=out)
    assert out.exists()
    assert len(fig.axes) == 3  # three integration times


def test_render_single_record_degenerate(tmp_path: Path) -> None:
    views = [_make_view("J", "C1", 1.25, 3600.0, 1e-6)]
    out = tmp_path / "single.pdf"
    fig = plot_contrast_curves_multiband(views, save_path=out)
    assert out.exists()
    assert len(fig.axes) == 1


def test_render_no_views_returns_placeholder(tmp_path: Path) -> None:
    out = tmp_path / "empty.pdf"
    fig = plot_contrast_curves_multiband([], save_path=out)
    assert out.exists()
    assert len(fig.axes) == 1


def test_duplicate_band_time_records_median_collapsed(tmp_path: Path) -> None:
    """When two records share (band, time), the curve is the median."""
    sep = np.linspace(1, 100, 10)
    v1 = _RecordView(
        "J", "C1", 1.25, "1h", 3600.0, "J=4", 4.0, sep, np.full_like(sep, 1e-5)
    )
    v2 = _RecordView(
        "J", "C1", 1.25, "1h", 3600.0, "J=4", 4.0, sep, np.full_like(sep, 1e-7)
    )
    out = tmp_path / "median.pdf"
    fig = plot_contrast_curves_multiband([v1, v2], save_path=out)
    assert out.exists()
    line_ys = fig.axes[0].lines[0].get_ydata()
    assert np.allclose(line_ys, np.median([1e-5, 1e-7]))


def test_jmag_sweep_records_are_not_median_collapsed(tmp_path: Path) -> None:
    sep = np.linspace(1, 100, 10)
    v1 = _RecordView(
        "J", "C1", 1.25, "1h", 3600.0, "J=4", 4.0, sep, np.full_like(sep, 1e-5)
    )
    v2 = _RecordView(
        "J", "C1", 1.25, "1h", 3600.0, "J=8", 8.0, sep, np.full_like(sep, 1e-7)
    )
    out = tmp_path / "jmag_split.pdf"
    fig = plot_contrast_curves_multiband([v1, v2], save_path=out)
    assert out.exists()
    assert len(fig.axes[0].lines) == 2
    labels = [line.get_label() for line in fig.axes[0].lines]
    assert labels == ["J-band J=4", "J-band J=8"]


def test_sweep_label_reads_jmag_coordinate() -> None:
    class Record:
        @property
        def coord_mapping(self) -> dict[str, float]:
            return {"scene.star.jmag": 6.0}

    assert _sweep_label(Record()) == ("J=6", 6.0)


def test_unphysical_contrast_tail_masked_without_nan_warning(tmp_path: Path) -> None:
    sep = np.array([1.0, 10.0, 100.0])
    contrast = np.array([1.0e-6, 2.0, np.nan])
    view = _RecordView("J", "C1", 1.25, "1h", 3600.0, "", 0.0, sep, contrast)
    out = tmp_path / "masked_tail.pdf"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig = plot_contrast_curves_multiband([view], save_path=out)

    assert out.exists()
    assert caught == []
    np.testing.assert_allclose(fig.axes[0].lines[0].get_xdata(), [1.0])
    np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), [1.0e-6])
