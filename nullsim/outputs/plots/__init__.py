"""Registered plot outputs for nullsim run collections."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib  # noqa: F401  # backend choice managed by styles module
import numpy as np

from nullsim.outputs.styles import _ensure_headless_backend  # noqa: F401  # side-effect

from matplotlib import pyplot as plt

from nullsim.outputs import RunCollection, RunRecord
from nullsim.pipeline import register_plot


@register_plot
def contrast_curve(
    collection: RunCollection,
    output_dir: Path,
    formats: Iterable[str],
) -> tuple[Path, ...]:
    """Write the placeholder contrast-curve plot for a run collection."""

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
    plotted_any = False
    y_min = np.inf
    y_max = -np.inf
    for record in collection.records:
        curve = record.result.state.results.contrast_curve
        if curve is None or curve.size == 0:
            continue
        sep = np.asarray(curve["separation_mas"], dtype=float)
        contrast = np.asarray(curve["contrast"], dtype=float)
        mask = (
            np.isfinite(sep)
            & np.isfinite(contrast)
            & (sep > 0.0)
            & (contrast > 0.0)
            & (contrast <= 1.0)
        )
        if not mask.any():
            continue
        order = np.argsort(sep[mask])
        plot_sep = sep[mask][order]
        plot_contrast = contrast[mask][order]
        axes.plot(
            plot_sep,
            plot_contrast,
            marker="o",
            linewidth=1.5,
            label=_record_label(record),
        )
        plotted_any = True
        y_min = min(y_min, float(plot_contrast.min()))
        y_max = max(y_max, float(plot_contrast.max()))

    axes.set_xlabel("Angular separation (mas)")
    axes.set_ylabel("Planet/star contrast")
    axes.set_title("Contrast curve")
    axes.grid(True, alpha=0.3)
    if plotted_any:
        axes.set_yscale("log")
        # Auto-fit y-range to data with half-decade padding on each side;
        # cap upper bound at 1.0 (contrast > 1 is unphysical) and floor
        # the lower bound at 1e-12 to handle pathological zero-floors.
        lo_decade = np.floor(np.log10(max(y_min, 1.0e-12)))
        hi_decade = np.ceil(np.log10(min(y_max, 1.0)))
        axes.set_ylim(10 ** (lo_decade - 0.5), 10 ** (hi_decade + 0.5))
    if len(collection.records) > 1:
        axes.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=7.5,
            frameon=True,
            framealpha=0.92,
            borderpad=0.55,
            labelspacing=0.28,
        )

    written_paths: list[Path] = []
    for file_format in formats:
        output_path = plot_dir / f"contrast_curve.{file_format}"
        figure.savefig(output_path)
        written_paths.append(output_path)
    plt.close(figure)
    return tuple(written_paths)


def _record_label(record: RunRecord) -> str:
    from nullsim.outputs.plots._axis_labels import compact_record_label
    return compact_record_label(record.coord_tuple, record.config.run.name)


def _short_axis_name(axis_name: str) -> str | None:
    """Backwards-compatible re-export of the shared helper."""

    from nullsim.outputs.plots._axis_labels import short_axis_name
    return short_axis_name(axis_name)


from nullsim.outputs.plots.calibration_time import calibration_time  # noqa: E402,F401
from nullsim.outputs.plots.chip_diagnostics import (  # noqa: E402,F401
    chip_diagnostics,
    chip_diagnostics_characterization,
)
from nullsim.outputs.plots.cophasing_unitary import cophasing_unitary  # noqa: E402,F401
from nullsim.outputs.plots.contrast_curves_multiband import contrast_curves_multiband  # noqa: E402,F401
from nullsim.outputs.plots.debug_report import debug_report  # noqa: E402,F401
from nullsim.outputs.plots.detection_space import detection_space  # noqa: E402,F401
from nullsim.outputs.plots.fringe_observability import fringe_observability  # noqa: E402,F401
from nullsim.outputs.plots.imaging_reconstruction import imaging_reconstruction  # noqa: E402,F401
from nullsim.outputs.plots.loss_budget import loss_budget  # noqa: E402,F401
from nullsim.outputs.plots.null_transmission_map import null_transmission_map  # noqa: E402,F401
from nullsim.outputs.plots.subaperture_layout import subaperture_layout  # noqa: E402,F401
from nullsim.outputs.plots.uv_coverage_grid import uv_coverage_grid  # noqa: E402,F401

__all__ = [
    "calibration_time",
    "chip_diagnostics",
    "chip_diagnostics_characterization",
    "cophasing_unitary",
    "contrast_curve",
    "contrast_curves_multiband",
    "debug_report",
    "detection_space",
    "fringe_observability",
    "imaging_reconstruction",
    "loss_budget",
    "null_transmission_map",
    "subaperture_layout",
    "uv_coverage_grid",
]
