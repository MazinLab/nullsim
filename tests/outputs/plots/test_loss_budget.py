"""Tests for the per-band loss-budget figure (paperv3 Fig 4, scalar adaptation)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import nullsim  # noqa: F401
from nullsim.outputs.plots.loss_budget import (
    _BandRow,
    _component_loss_db,
    plot_loss_budget,
)


def test_component_loss_db_picks_band_center() -> None:
    payload = {
        "wavelength_grid_um": [1.0, 1.25, 1.5],
        "transmission_fraction": [0.5, 0.8, 0.9],
    }
    assert _component_loss_db(payload, 1.25) == pytest.approx(
        -10.0 * np.log10(0.8)
    )


def test_component_loss_db_handles_missing_fields() -> None:
    assert _component_loss_db({}, 1.25) == 0.0
    assert _component_loss_db({"wavelength_grid_um": [1.0]}, 1.25) == 0.0


def test_component_loss_db_zero_transmission_is_inf() -> None:
    payload = {
        "wavelength_grid_um": [1.25],
        "transmission_fraction": [0.0],
    }
    assert _component_loss_db(payload, 1.25) == float("inf")


def _make_row(band: str, wl: float, losses_db: dict[str, float]) -> _BandRow:
    total_db = sum(v for v in losses_db.values() if np.isfinite(v))
    return _BandRow(
        band_label=band, wavelength_um=wl,
        component_loss_db=losses_db,
        total_transmission=10.0 ** (-total_db / 10.0),
    )


def test_render_three_band_stack(tmp_path: Path) -> None:
    rows = [
        _make_row("Y", 1.0, {"transport": 1.5, "delay": 5.1}),
        _make_row("J", 1.25, {"transport": 1.2, "delay": 5.1}),
        _make_row("H", 1.65, {"transport": 0.9, "delay": 5.1}),
    ]
    out = tmp_path / "lb.pdf"
    fig = plot_loss_budget(rows, save_path=out)
    assert out.exists()
    assert len(fig.axes) >= 1


def test_delay_lines_sort_between_transport_and_chip(tmp_path: Path) -> None:
    rows = [
        _make_row(
            "J",
            1.25,
            {
                "transport.pm980": 1.0,
                "delay_lines.geometric": 2.0,
                "chip.kernel_mzi_mesh": 3.0,
            },
        )
    ]
    out = tmp_path / "lb_delay_order.pdf"
    fig = plot_loss_budget(rows, save_path=out)
    assert out.exists()
    legend = fig.axes[0].get_legend()
    assert legend is not None
    labels = [text.get_text() for text in legend.get_texts()]
    assert labels == [
        "chip.kernel_mzi_mesh",
        "delay_lines.geometric",
        "transport.pm980",
    ]


def test_render_no_rows_returns_placeholder(tmp_path: Path) -> None:
    out = tmp_path / "lb_empty.pdf"
    fig = plot_loss_budget([], save_path=out)
    assert out.exists()
    assert len(fig.axes) == 1


def test_duplicate_band_records_collapsed_median(tmp_path: Path) -> None:
    rows = [
        _make_row("J", 1.25, {"transport": 1.0}),
        _make_row("J", 1.25, {"transport": 3.0}),
    ]
    out = tmp_path / "lb_dup.pdf"
    fig = plot_loss_budget(rows, save_path=out)
    assert out.exists()
    # Stacked-bar height (transport segment) should be the median = 2.0 dB
    bar = fig.axes[0].patches[0]
    assert bar.get_height() == pytest.approx(2.0)


def test_render_all_nan_component_logged_and_marked(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Phase 9: an all-NaN component at a band must (a) emit a warning,
    (b) render an explicit 'n/a' marker instead of silently collapsing
    to 0 dB."""
    rows = [
        _make_row("J", 1.25, {"transport": 1.0, "broken_stage": float("nan")}),
    ]
    out = tmp_path / "lb_nan.pdf"
    with caplog.at_level("WARNING", logger="nullsim.outputs.plots.loss_budget"):
        fig = plot_loss_budget(rows, save_path=out)
    assert out.exists()
    # Warning message points at the broken component + band.
    msgs = [record.getMessage() for record in caplog.records]
    assert any("broken_stage" in m and "J" in m for m in msgs), (
        f"expected an all-NaN warning for broken_stage at J; got {msgs!r}"
    )
    # n/a marker text shows up on the axis.
    ax = fig.axes[0]
    annotated = [t.get_text() for t in ax.texts]
    assert any("n/a: broken_stage" in t for t in annotated), (
        f"missing 'n/a' marker; axis texts = {annotated!r}"
    )
