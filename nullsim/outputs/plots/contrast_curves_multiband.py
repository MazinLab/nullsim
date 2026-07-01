"""Multi-band contrast curves figure (KILO paperv3 Fig 10).

Direct port of ``kilo/studies/natural_design.py:contrast_curves_multiband``
layout: ``1 × n_integration_times`` panels, with one contrast curve per
band overlaid in each panel. Bands derive from each record's
``grid.wavelength_center_um``; integration times from
``observation.integration_time_s``. The user supplies the sweep over
either / both axes in their TOML.

When only one (band, time) pair is present, the figure degenerates to a
single panel with a single curve — still useful as a sanity check on
top of a non-sweep run.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

import matplotlib  # noqa: F401  # backend choice managed by styles module
import numpy as np

from nullsim.outputs.styles import _ensure_headless_backend  # noqa: F401  # side-effect

from matplotlib import pyplot as plt

from nullsim.outputs import RunCollection
from nullsim.pipeline import register_plot


# Band ranges in micrometers (KILO paperv3 conventions). Outside these
# ranges the wavelength is used directly as the label.
_BAND_RANGES_UM: tuple[tuple[str, str, float, float], ...] = (
    ("I", "C2", 0.72, 0.88),
    ("Y", "C0", 0.95, 1.10),
    ("J", "C1", 1.15, 1.35),
    ("H", "C3", 1.50, 1.80),
    ("K", "C4", 2.00, 2.40),
)


class _RecordView(NamedTuple):
    band_label: str
    band_color: str
    band_sort_key: float
    time_label: str
    time_sort_key: float
    sweep_label: str
    sweep_sort_key: float
    separations_mas: np.ndarray
    contrast: np.ndarray


def _band_label(wavelength_um: float) -> tuple[str, str, float]:
    for name, color, lo, hi in _BAND_RANGES_UM:
        if lo <= wavelength_um <= hi:
            return name, color, wavelength_um
    return f"{wavelength_um:.2f} um", "C7", wavelength_um


def _time_label(seconds: float) -> str:
    hours = seconds / 3600.0
    if abs(hours - round(hours)) < 0.05:
        return f"{int(round(hours))}h"
    return f"{hours:.1f}h"


def _sweep_label(record: object) -> tuple[str, float]:
    """Return an optional sweep label that should split plotted series."""

    coord_mapping = getattr(record, "coord_mapping", {})
    if not isinstance(coord_mapping, dict):
        return "", 0.0
    for key, value in coord_mapping.items():
        if key != "scene.star.jmag" and not key.endswith(".jmag"):
            continue
        try:
            jmag = float(value)
        except (TypeError, ValueError):
            return f"{key}={value}", 0.0
        return f"J={jmag:g}", jmag
    return "", 0.0


def _summarize_records(collection: RunCollection) -> list[_RecordView]:
    views: list[_RecordView] = []
    for record in collection.records:
        curve = record.result.state.results.contrast_curve
        if curve is None or curve.size == 0:
            continue
        wl_um = float(record.config.grid.wavelength_center_um)
        t_s = float(record.config.observation.integration_time_s)
        band_name, band_color, band_sort = _band_label(wl_um)
        sweep_label, sweep_sort = _sweep_label(record)
        views.append(_RecordView(
            band_label=band_name,
            band_color=band_color,
            band_sort_key=band_sort,
            time_label=_time_label(t_s),
            time_sort_key=t_s,
            sweep_label=sweep_label,
            sweep_sort_key=sweep_sort,
            separations_mas=np.asarray(curve["separation_mas"], dtype=np.float64),
            contrast=np.asarray(curve["contrast"], dtype=np.float64),
        ))
    return views


def plot_contrast_curves_multiband(
    views: list[_RecordView],
    save_path: Path | None = None,
) -> plt.Figure:
    """Render the 1 x n_times multi-band contrast curve panel."""

    if not views:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.text(0.5, 0.5, "no contrast curves to plot",
                ha="center", va="center", transform=ax.transAxes)
        if save_path is not None:
            fig.savefig(save_path)
        return fig

    times_sorted = sorted({(v.time_sort_key, v.time_label) for v in views})
    series_sorted = sorted(
        {
            (
                v.band_sort_key,
                v.sweep_sort_key,
                v.band_label,
                v.band_color,
                v.sweep_label,
            )
            for v in views
        }
    )
    n_panels = len(times_sorted)

    # Color: one hue per band when only bands vary. When a magnitude (or other)
    # sweep splits the series, every curve shares its band color and the plot
    # collapses to a single hue, so shade each series along a sequential
    # colormap (ordered by band then sweep value) instead.
    distinct_sweeps = {s[4] for s in series_sorted if s[4]}
    if len(distinct_sweeps) > 1:
        cmap = plt.get_cmap("viridis")
        n_series = len(series_sorted)
        series_colors = {
            series: cmap(0.05 + 0.80 * (idx / max(n_series - 1, 1)))
            for idx, series in enumerate(series_sorted)
        }
    else:
        series_colors = {series: series[3] for series in series_sorted}

    fig, axes = plt.subplots(
        1, n_panels, figsize=(6 * n_panels, 6),
        sharey=True, squeeze=False,
    )

    y_min = np.inf
    y_max = -np.inf
    for j, (t_key, t_label) in enumerate(times_sorted):
        ax = axes[0, j]
        for series in series_sorted:
            _, sweep_key, band_label, band_color, sweep_label = series
            matched = [
                v for v in views
                if (
                    v.time_sort_key == t_key
                    and v.band_label == band_label
                    and v.sweep_label == sweep_label
                    and v.sweep_sort_key == sweep_key
                )
            ]
            if not matched:
                continue
            sep = matched[0].separations_mas
            stack = np.stack([v.contrast for v in matched])
            stack = np.where(np.isfinite(stack), stack, np.nan)
            stack = np.where((stack > 0.0) & (stack <= 1.0), stack, np.nan)
            finite_columns = np.isfinite(stack).any(axis=0)
            ct = np.full(stack.shape[1], np.nan, dtype=np.float64)
            if finite_columns.any():
                ct[finite_columns] = np.nanmedian(
                    stack[:, finite_columns], axis=0,
                )
            mask = np.isfinite(sep) & (sep > 0.0) & np.isfinite(ct)
            if not mask.any():
                continue
            order = np.argsort(sep[mask])
            label = f"{band_label}-band"
            if sweep_label:
                label = f"{label} {sweep_label}"
            ax.semilogy(sep[mask][order], ct[mask][order], "-",
                        color=series_colors[series], label=label, linewidth=2)
            y_min = min(y_min, float(np.nanmin(ct[mask])))
            y_max = max(y_max, float(np.nanmax(ct[mask])))

        ax.set_xlabel("Angular separation (mas)")
        if j == 0:
            ax.set_ylabel(r"Detectable contrast ($5\sigma$)")
        if n_panels > 1:
            ax.set_title(f"{t_label} integration")
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    if np.isfinite(y_min) and np.isfinite(y_max):
        lo_decade = np.floor(np.log10(max(y_min, 1.0e-12)))
        hi_decade = np.ceil(np.log10(min(y_max, 1.0)))
        for ax in axes[0]:
            ax.set_ylim(10 ** (lo_decade - 0.5), 10 ** (hi_decade + 0.5))

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


@register_plot
def contrast_curves_multiband(
    collection: RunCollection,
    output_dir: Path,
    formats: Iterable[str],
) -> tuple[Path, ...]:
    """Registered plot: ``contrast_curves_multiband.<fmt>`` per requested format."""

    views = _summarize_records(collection)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig = plot_contrast_curves_multiband(views, save_path=None)

    written: list[Path] = []
    for fmt in formats:
        out = plot_dir / f"contrast_curves_multiband.{fmt}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        written.append(out)
    plt.close(fig)
    return tuple(written)
