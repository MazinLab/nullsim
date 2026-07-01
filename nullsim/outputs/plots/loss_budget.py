"""Per-band end-to-end throughput loss budget figure.

KILO paperv3 Fig 14 (``dualpol_loss_budget.pdf``) is a dual-polarization
budget: scalar SMF vs PM per fiber vs dual-pol per arm vs dual-pol
combined, three bars per band. nullsim is single-polarization today, so
this port renders the scalar equivalent: one stacked-loss bar per band,
with each stage's throughput contribution as a separate segment.

Bars come from each record's ``state.throughput.components`` (which the
transport, telescope, injection, and chip stages all populate). Bands
derive from ``grid.wavelength_center_um`` — the same recipe used by
``contrast_curves_multiband``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

import matplotlib  # noqa: F401  # backend choice managed by styles module
import numpy as np

_log = logging.getLogger(__name__)

from nullsim.outputs.styles import _ensure_headless_backend  # noqa: E402, F401  # side-effect

from matplotlib import pyplot as plt  # noqa: E402

from nullsim.outputs import RunCollection  # noqa: E402
from nullsim.outputs.plots.contrast_curves_multiband import _band_label  # noqa: E402
from nullsim.pipeline import register_plot  # noqa: E402


class _BandRow(NamedTuple):
    band_label: str
    wavelength_um: float
    component_loss_db: dict[str, float]  # name → dB loss (>=0)
    total_transmission: float


def _component_loss_db(payload: dict, wavelength_um: float) -> float:
    """Compute dB loss for one stage's throughput component at the band center."""

    if not isinstance(payload, dict):
        # Stages should write {wavelength_grid_um, transmission_fraction}
        # dicts per SPEC §6.2; a bare array means the stage uses an older
        # payload shape and we can't extract a band-aware loss.
        return 0.0
    grid = payload.get("wavelength_grid_um")
    fractions = payload.get("transmission_fraction")
    if not grid or not fractions:
        return 0.0
    arr_wl = np.asarray(grid, dtype=np.float64)
    arr_t = np.asarray(fractions, dtype=np.float64)
    idx = int(np.argmin(np.abs(arr_wl - wavelength_um)))
    t = float(arr_t[idx])
    if t <= 0:
        return float("inf")
    return -10.0 * np.log10(t)


def _summarize_records(collection: RunCollection) -> list[_BandRow]:
    rows: list[_BandRow] = []
    for record in collection.records:
        wl_um = float(record.config.grid.wavelength_center_um)
        band_name, _band_color, _band_sort = _band_label(wl_um)
        components = record.result.state.throughput.components or {}
        per_stage: dict[str, float] = {}
        total_t = 1.0
        for name, payload in components.items():
            db = _component_loss_db(payload, wl_um)
            per_stage[name] = db
            t = 10.0 ** (-db / 10.0) if np.isfinite(db) else 0.0
            total_t *= t
        rows.append(_BandRow(
            band_label=band_name,
            wavelength_um=wl_um,
            component_loss_db=per_stage,
            total_transmission=total_t,
        ))
    return rows


def plot_loss_budget(
    rows: list[_BandRow],
    save_path: Path | None = None,
) -> plt.Figure:
    """Render the per-band stacked-loss bar chart."""

    if not rows:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.text(0.5, 0.5, "no throughput components to plot",
                ha="center", va="center", transform=ax.transAxes)
        if save_path is not None:
            fig.savefig(save_path)
        return fig

    # Deduplicate rows per band (median collapse on dB) so a multi-axis
    # sweep doesn't double-stack.
    by_band: dict[str, list[_BandRow]] = {}
    for r in rows:
        by_band.setdefault(r.band_label, []).append(r)
    band_keys = sorted(by_band.keys(), key=lambda b: by_band[b][0].wavelength_um)

    # Union of all component names, ordered along the physical light
    # path: atmosphere -> telescope -> injection -> transport ->
    # delay_lines -> chip -> detector. Inside each family, alphabetical.
    # Components whose family prefix is not in the canonical order land
    # at the end so a new stage type still renders even if we haven't
    # ranked it.
    _LIGHT_PATH_ORDER = (
        "atmosphere",
        "telescope",
        "injection",
        "transport",
        "delay_lines",
        "chip",
        "detector",
    )
    _LIGHT_PATH_RANK = {name: i for i, name in enumerate(_LIGHT_PATH_ORDER)}
    all_components: set[str] = set()
    for r in rows:
        all_components.update(r.component_loss_db.keys())

    def _component_sort_key(name: str) -> tuple[int, str]:
        prefix = name.split(".", 1)[0]
        return (_LIGHT_PATH_RANK.get(prefix, len(_LIGHT_PATH_ORDER)), name)

    component_order = sorted(all_components, key=_component_sort_key)

    colormap = plt.get_cmap("tab20")
    component_colors = {
        name: colormap(i % colormap.N) for i, name in enumerate(component_order)
    }

    fig, ax = plt.subplots(figsize=(max(6, 2.0 * len(band_keys) + 4), 5.5))
    x = np.arange(len(band_keys))
    width = 0.6
    bottoms = np.zeros(len(band_keys), dtype=np.float64)

    # Phase 9: track which (component, band) cells dropped to NaN so
    # we can render them with a hatched 'n/a' marker instead of
    # silently collapsing to 0 dB. A broken stage that reports all-NaN
    # transmission used to be indistinguishable from a perfectly
    # transmitting one — worst-possible failure mode for a diagnostic.
    missing_cells: list[tuple[int, str]] = []
    for name in component_order:
        heights = np.zeros(len(band_keys), dtype=np.float64)
        for j, band in enumerate(band_keys):
            band_rows = by_band[band]
            vals = [r.component_loss_db.get(name, 0.0) for r in band_rows]
            finite = [v for v in vals if np.isfinite(v)]
            if finite:
                heights[j] = float(np.median(finite))
            elif vals:
                missing_cells.append((j, name))
                heights[j] = 0.0
                _log.warning(
                    "loss_budget: all-NaN transmission for component %r at "
                    "band %r; rendering as 'n/a' (no contribution to the "
                    "stacked bar).",
                    name, band,
                )
        if heights.sum() <= 0:
            continue
        ax.bar(
            x, heights, width, bottom=bottoms,
            color=component_colors[name], label=name, edgecolor="white", linewidth=0.5,
        )
        bottoms += heights

    # Phase 9: explicit 'n/a' markers at the top of each band's bar for
    # every component that dropped to NaN. The markers are stacked above
    # the bar so they don't overlap the totals annotation.
    if missing_cells:
        marker_height = 0.25
        per_band_count: dict[int, int] = {}
        for j, _name in missing_cells:
            per_band_count[j] = per_band_count.get(j, 0) + 1
        for j, name in missing_cells:
            slot = per_band_count[j] - 1
            y = bottoms[j] + 0.4 + slot * marker_height
            ax.bar(
                [j], [marker_height], width, bottom=[y],
                facecolor="white", edgecolor=component_colors[name],
                linewidth=0.5, hatch="///",
            )
            ax.text(
                j, y + marker_height / 2.0,
                f"n/a: {name}",
                ha="center", va="center", fontsize=7, color=component_colors[name],
            )
            per_band_count[j] -= 1

    # Annotate total transmission above each bar.
    for j, band in enumerate(band_keys):
        band_rows = by_band[band]
        med_t = float(np.median([r.total_transmission for r in band_rows]))
        ax.text(
            j, bottoms[j] + 0.15,
            f"T = {med_t * 100:.1f}%",
            ha="center", fontsize=9, color="0.2",
        )

    band_labels = [
        f"{band}\n({by_band[band][0].wavelength_um:.2f} um)" for band in band_keys
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(band_labels)
    ax.set_ylabel("Throughput loss (dB)")
    n_collapsed = max(len(v) for v in by_band.values()) if by_band else 1
    title = "End-to-end throughput budget per band"
    if n_collapsed > 1:
        # State the collapse convention on the figure: each segment is the
        # per-component MEDIAN over the sweep cells sharing the band, so
        # the stacked bar corresponds to no single cell (review 2026-06-10
        # Issue 23c).
        title += f"  (median over {n_collapsed} sweep cells)"
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    # Reverse legend so the top entry matches the topmost bar segment
    # (matplotlib's default order is bottom-of-stack first).
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1],
              loc="upper left", fontsize=8, bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
    return fig


@register_plot
def loss_budget(
    collection: RunCollection,
    output_dir: Path,
    formats: Iterable[str],
) -> tuple[Path, ...]:
    """Registered plot: ``loss_budget.<fmt>`` per requested format."""

    rows = _summarize_records(collection)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig = plot_loss_budget(rows, save_path=None)

    written: list[Path] = []
    for fmt in formats:
        out = plot_dir / f"loss_budget.{fmt}"
        fig.savefig(out)
        written.append(out)
    plt.close(fig)
    return tuple(written)
