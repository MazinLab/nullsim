"""Chip-optimization convergence figure (KILO paperv3 Fig 9, scoped down).

KILO paperv3 Fig 9 (``calibration_time_J.pdf``) has two panels: (1) loss
vs wall-clock time for individual restarts + best-of-k envelope, and
(2) achievable null depth vs calibration time budget at three calibration
star magnitudes. The second panel maps iteration count → wall-clock time
via a photon-noise-aware objective-evaluation rate; this nullsim port
keeps the first panel only since:

* nullsim's chip optimizer minimizes mathematical leakage (the textbook
  noise-free Clements objective), not a photon-noise-weighted SNR;
* there is no per-J-mag wall-clock mapping primitive in nullsim today.

The single panel still gives a useful sanity check: it confirms the
optimizer is actually converging, and how many objective evaluations
the best restart needed to reach the realized null depth.

To populate the convergence history during a run, set
``capture_convergence_history = true`` under the chip optimizer stage in
the TOML. Both backends record, at every iteration of the best restart,
the objective (``convergence_history``) and the normalized null
(``convergence_history_normalized_null``). The figure plots the normalized
null, the quantity published as ``optimization_normalized_null_final``.
The objective is not shown: it is dimensional and goes negative once the
throughput reward is on, so it has no place on a log null-depth axis
(review 2026-11-07 Issue 23).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib  # noqa: F401  # backend choice managed by styles module
import numpy as np

from nullsim.outputs.styles import _ensure_headless_backend  # noqa: F401  # side-effect

from matplotlib import pyplot as plt

from nullsim.outputs import RunCollection
from nullsim.outputs.plots.contrast_curves_multiband import _band_label
from nullsim.pipeline import register_plot


class _ConvergenceTrace(NamedTuple):
    record_label: str
    color: str
    nfev: np.ndarray
    normalized_null: np.ndarray
    final_null: float


def _extract_traces(collection: RunCollection) -> list[_ConvergenceTrace]:
    traces: list[_ConvergenceTrace] = []
    for record in collection.records:
        components: dict[str, Any] = record.result.state.results.components or {}
        for name, payload in components.items():
            if not name.startswith("chip_optimization"):
                continue
            history = payload.get("convergence_history_normalized_null")
            if not history:
                continue
            arr = np.asarray(history, dtype=np.float64)
            wl_um = float(record.config.grid.wavelength_center_um)
            band, color, _ = _band_label(wl_um)
            traces.append(_ConvergenceTrace(
                record_label=f"{band} ({wl_um:.2f} um)",
                color=color,
                nfev=arr[:, 0],
                normalized_null=arr[:, 1],
                final_null=float(
                    payload.get("optimization_normalized_null_final", arr[-1, 1])
                ),
            ))
    return traces


def plot_calibration_time(
    traces: list[_ConvergenceTrace],
    save_path: Path | None = None,
) -> plt.Figure:
    """Single-panel chip-optimizer convergence figure."""

    fig, ax = plt.subplots(figsize=(8, 5))

    if not traces:
        ax.text(
            0.5, 0.5,
            "no convergence history captured\n"
            "(set capture_convergence_history = true under "
            "the chip optimizer stage to populate this plot)",
            ha="center", va="center", transform=ax.transAxes,
        )
        if save_path is not None:
            fig.savefig(save_path)
        return fig

    best_null = float("inf")
    for trace in traces:
        ax.semilogy(
            trace.nfev, trace.normalized_null,
            "-o", color=trace.color, markersize=3, linewidth=1.5,
            label=trace.record_label,
        )
        best_null = min(best_null, trace.final_null)

    ax.set_xlabel("Objective evaluations")
    ax.set_ylabel(
        "Normalized null (dark-port leakage / input power per objective row)"
    )
    ax.set_title("Chip optimizer convergence")
    ax.grid(True, which="both", alpha=0.3)
    if np.isfinite(best_null) and best_null > 0.0:
        ax.axhline(
            best_null, color="0.4", linestyle="--", linewidth=1,
            label=f"Best floor: {best_null:.2e}",
        )
    ax.legend(fontsize=9)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
    return fig


@register_plot
def calibration_time(
    collection: RunCollection,
    output_dir: Path,
    formats: Iterable[str],
) -> tuple[Path, ...]:
    """Registered plot: ``calibration_time.<fmt>`` per requested format."""

    traces = _extract_traces(collection)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig = plot_calibration_time(traces, save_path=None)

    written: list[Path] = []
    for fmt in formats:
        out = plot_dir / f"calibration_time.{fmt}"
        fig.savefig(out)
        written.append(out)
    plt.close(fig)
    return tuple(written)
