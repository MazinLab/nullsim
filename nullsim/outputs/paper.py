"""Opt-in paper artifact bundle for nullsim run outputs.

The KILO paper figure directory is broader than a single nullsim plot
request: it contains the main run diagnostics, selected sidecar tables,
and a few paper-only schematics.  This module keeps that bundle as a
thin orchestration layer over the normal output registry so the canonical
plot implementations stay in ``nullsim.outputs.plots``.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from nullsim.outputs import RunCollection, RunRecord
from nullsim.pipeline import registry

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _OutputProduct:
    registry_name: str
    source_name: str
    target_name: str
    formats: tuple[str, ...]
    sidecars: tuple[tuple[str, str], ...] = ()


_PLOT_PRODUCTS: tuple[_OutputProduct, ...] = (
    _OutputProduct(
        "subaperture_layout", "subaperture_layout", "subaperture_layout", ("pdf",)
    ),
    _OutputProduct(
        "uv_coverage_grid", "uv_coverage_grid", "uv_coverage_grid", ("pdf",)
    ),
    _OutputProduct(
        "chip_diagnostics", "chip_diagnostics", "chip_diagnostics", ("pdf",)
    ),
    _OutputProduct(
        "calibration_time", "calibration_time", "calibration_time_J", ("pdf",)
    ),
    _OutputProduct(
        "contrast_curves_multiband",
        "contrast_curves_multiband",
        "contrast_curves_multiband",
        ("pdf",),
    ),
    _OutputProduct(
        "detection_space",
        "detection_space",
        "detection_space",
        ("pdf",),
    ),
    _OutputProduct(
        "imaging_reconstruction",
        "imaging_reconstruction",
        "imaging_reconstruction",
        ("pdf",),
        (("tables/imaging_reconstruction.json", "imaging_results.json"),),
    ),
    _OutputProduct("loss_budget", "loss_budget", "dualpol_loss_budget", ("pdf",)),
    _OutputProduct(
        "null_transmission_map",
        "null_transmission_map",
        "null_transmission_map",
        ("pdf",),
        (("plots/null_transmission_map.npz", "null_transmission_map.npz"),),
    ),
    _OutputProduct(
        "fringe_observability", "fringe_observability", "fringe_observability", ("pdf",)
    ),
    _OutputProduct(
        "cophasing_unitary", "cophasing_unitary", "cophasing_unitary", ("pdf",)
    ),
)

_TABLE_PRODUCTS: tuple[_OutputProduct, ...] = (
    _OutputProduct(
        "performance_summary",
        "performance_summary",
        "performance_summary",
        ("json", "csv"),
    ),
    _OutputProduct(
        "characterization_summary",
        "characterization_summary",
        "characterization_summary",
        ("json", "csv"),
    ),
)

_OUT_OF_SCOPE_PRODUCTS: tuple[tuple[str, str], ...] = (
    (
        "natural_array_layout.pdf",
        "Run the four-telescope Maunakea design as a separate nullsim config.",
    ),
    (
        "natural_detection_space.pdf",
        "Run the four-telescope Maunakea design as a separate nullsim config.",
    ),
    (
        "natural_detection_space_cache.json",
        "Run the four-telescope Maunakea design as a separate nullsim config.",
    ),
    (
        "natural_subaperture_layout.pdf",
        "Run the four-telescope Maunakea design as a separate nullsim config.",
    ),
)

_POL_MODEL_CROSSOVER_REARTH = 1.5
_POL_MODEL_BLEND_HI_REARTH = 2.0
_EARTHLIKE_PEAK_POL_FRAC = 0.40
_CLOUDY_GIANT_PEAK_POL_FRAC = 0.15
_ORBIT_AVERAGED_POL_SHAPE_FACTOR = 0.5268
_DUAL_POL_THROUGHPUT_VS_SCALAR = 0.87
_PDI_HWP_ROTATION_RATE_HZ = 1000.0
_LOG_SCATTER_FLOOR = 1.0e-2


def write_paper_outputs(
    collection: RunCollection, output_dir: Path
) -> tuple[Path, ...]:
    """Write the opt-in paper artifact bundle beside ``plots/`` and ``tables/``.

    Args:
        collection: Run collection already produced by the CLI.
        output_dir: Run output directory.

    Returns:
        Paths written under ``output_dir / "paper"``.
    """

    paper_dir = output_dir / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "source": "nullsim",
        "run_name": collection.config.run.name,
        "config_hash": collection.config.config_hash,
        "run_hash": collection.config.run_hash,
        "record_count": len(collection.records),
        "generated": [],
        "skipped": [],
    }
    written: list[Path] = []

    for filename, writer in (
        ("signal_chain.pdf", _write_signal_chain),
        ("array_layout.pdf", _write_array_layout),
        ("baseline_histogram.pdf", _write_baseline_histogram),
        ("cophasing_scheme.pdf", _write_cophasing_scheme),
        ("dirty_beams.pdf", _write_dirty_beams),
        (
            "chip_diagnostics_characterization.pdf",
            _write_chip_diagnostics_characterization,
        ),
        ("planet_polarization_phase.pdf", _write_planet_polarization_phase),
        ("dualpol_optical_layout.pdf", _write_dualpol_optical_layout),
        ("signal_chain_dualpol.pdf", _write_signal_chain_dualpol),
    ):
        _run_custom_writer(
            filename=filename,
            writer=writer,
            collection=collection,
            paper_dir=paper_dir,
            manifest=manifest,
            written=written,
        )

    for product in _PLOT_PRODUCTS:
        _copy_registered_product(
            product=product,
            kind="plot",
            collection=collection,
            output_dir=output_dir,
            paper_dir=paper_dir,
            manifest=manifest,
            written=written,
        )
    for product in _TABLE_PRODUCTS:
        _copy_registered_product(
            product=product,
            kind="table",
            collection=collection,
            output_dir=output_dir,
            paper_dir=paper_dir,
            manifest=manifest,
            written=written,
        )

    for label, writer in (
        ("exoplanet_table", _write_exoplanet_survey),
        ("pdi_products", _write_pdi_products),
    ):
        _run_multi_writer(
            label=label,
            writer=writer,
            collection=collection,
            paper_dir=paper_dir,
            manifest=manifest,
            written=written,
        )

    manifest["out_of_scope"] = [
        {"file": filename, "reason": reason}
        for filename, reason in _OUT_OF_SCOPE_PRODUCTS
    ]

    _write_sidecar_summaries(collection, paper_dir, manifest, written)
    manifest_path = paper_dir / "paper_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    written.append(manifest_path)
    return tuple(written)


def _run_custom_writer(
    *,
    filename: str,
    writer: Callable[[RunCollection, Path], Path],
    collection: RunCollection,
    paper_dir: Path,
    manifest: dict[str, Any],
    written: list[Path],
) -> None:
    try:
        path = writer(collection, paper_dir)
    except Exception as exc:  # pragma: no cover - defensive manifest path
        LOGGER.warning("paper output %s failed: %s", filename, exc)
        _record_skip(manifest, filename, f"{type(exc).__name__}: {exc}")
        return
    if path.exists():
        written.append(path)
        _record_generated(manifest, path.name, "paper")
    else:
        _record_skip(manifest, filename, "writer produced no file")


def _run_multi_writer(
    *,
    label: str,
    writer: Callable[[RunCollection, Path], tuple[Path, ...]],
    collection: RunCollection,
    paper_dir: Path,
    manifest: dict[str, Any],
    written: list[Path],
) -> None:
    try:
        paths = writer(collection, paper_dir)
    except Exception as exc:  # pragma: no cover - defensive manifest path
        LOGGER.warning("paper output study %s failed: %s", label, exc)
        _record_skip(manifest, label, f"{type(exc).__name__}: {exc}")
        return
    if not paths:
        _record_skip(manifest, label, "writer produced no files")
        return
    for path in paths:
        if path.exists():
            written.append(path)
            _record_generated(manifest, path.name, "paper")
        else:
            _record_skip(manifest, path.name, "writer returned a missing file")


def _copy_registered_product(
    *,
    product: _OutputProduct,
    kind: str,
    collection: RunCollection,
    output_dir: Path,
    paper_dir: Path,
    manifest: dict[str, Any],
    written: list[Path],
) -> None:
    writer = (registry.plots if kind == "plot" else registry.tables).get(
        product.registry_name
    )
    if writer is None:
        _record_skip(
            manifest,
            f"{product.target_name}.{product.formats[0]}",
            f"{kind} {product.registry_name!r} is not registered",
        )
        return

    source_root = "plots" if kind == "plot" else "tables"
    with tempfile.TemporaryDirectory(prefix=".paper-", dir=str(output_dir)) as tmp_name:
        tmp_dir = Path(tmp_name)
        try:
            produced = tuple(
                Path(path)
                for path in writer(collection, tmp_dir, tuple(product.formats))
            )
        except Exception as exc:  # pragma: no cover - product-specific skip path
            LOGGER.warning(
                "paper output %s %s failed: %s",
                kind,
                product.registry_name,
                exc,
            )
            _record_skip(
                manifest,
                f"{product.target_name}.{product.formats[0]}",
                f"{kind} {product.registry_name!r} failed: {type(exc).__name__}: {exc}",
            )
            return

        produced_by_name = {path.name: path for path in produced}
        for file_format in product.formats:
            source_name = f"{product.source_name}.{file_format}"
            source_path = (
                produced_by_name.get(source_name) or tmp_dir / source_root / source_name
            )
            target_path = paper_dir / f"{product.target_name}.{file_format}"
            if _copy_if_exists(source_path, target_path):
                written.append(target_path)
                _record_generated(manifest, target_path.name, kind)
            else:
                _record_skip(
                    manifest,
                    target_path.name,
                    f"{kind} {product.registry_name!r} produced no {source_name}",
                )

        for source_rel, target_name in product.sidecars:
            source_path = tmp_dir / source_rel
            target_path = paper_dir / target_name
            if _copy_if_exists(source_path, target_path):
                written.append(target_path)
                _record_generated(manifest, target_path.name, f"{kind} sidecar")
            else:
                _record_skip(
                    manifest,
                    target_path.name,
                    f"{kind} {product.registry_name!r} produced no {source_rel}",
                )


def _copy_if_exists(source_path: Path, target_path: Path) -> bool:
    if not source_path.exists():
        return False
    shutil.copy2(source_path, target_path)
    return True


def _write_sidecar_summaries(
    collection: RunCollection,
    paper_dir: Path,
    manifest: dict[str, Any],
    written: list[Path],
) -> None:
    for filename, payload in (
        ("chip_diagnostics.json", _chip_payload_summary(collection)),
        ("uv_coverage_results.json", _uv_coverage_summary(collection)),
        ("keck_only_results.json", _run_payload_summary(collection)),
        ("dualpol_loss_budget.json", _loss_budget_summary(collection)),
    ):
        path = paper_dir / filename
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        written.append(path)
        _record_generated(manifest, filename, "paper sidecar")


def _write_signal_chain(collection: RunCollection, paper_dir: Path) -> Path:
    from matplotlib import pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    from nullsim.outputs.styles import publication_style

    first = _first_record(collection)
    chip = _find_component(first, "chip.") if first is not None else None
    n_inputs = _int_from_payload(chip, "n_input_modes") or _int_from_payload(
        chip, "n_modes"
    )
    n_bright = _int_from_payload(chip, "n_bright_ports")
    n_dark = _int_from_payload(chip, "n_dark_ports")
    n_spectro = _int_from_payload(chip, "n_spectro_ports")
    mesh_label = (
        f"Photonic mesh\n({n_inputs} inputs)"
        if n_inputs is not None
        else "Photonic mesh"
    )

    with publication_style():
        fig, ax = plt.subplots(figsize=(14.5, 4.8))
        ax.set_xlim(0.0, 14.5)
        ax.set_ylim(0.0, 5.0)
        ax.axis("off")

        colors = {
            "telescope": "#4A90D9",
            "fiber": "#6DBB75",
            "chip": "#E5A432",
            "detector": "#C84F4F",
            "control": "#7B6FD6",
        }

        def box(x: float, y: float, w: float, h: float, label: str, color: str) -> None:
            patch = FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.12",
                facecolor=color,
                edgecolor="black",
                alpha=0.88,
                linewidth=1.0,
            )
            ax.add_patch(patch)
            ax.text(
                x + w / 2.0,
                y + h / 2.0,
                label,
                ha="center",
                va="center",
                fontsize=9.5,
                fontweight="bold",
                color="white",
            )

        def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
            ax.add_patch(
                FancyArrowPatch(
                    (x1, y1),
                    (x2, y2),
                    arrowstyle="-|>",
                    mutation_scale=13,
                    color="0.25",
                    linewidth=1.25,
                    shrinkA=5,
                    shrinkB=5,
                )
            )

        main = [
            (0.25, "Keck\napertures", "telescope"),
            (1.65, "Adaptive\noptics", "telescope"),
            (3.05, "Sub-aperture\nfibers", "fiber"),
            (4.45, "Delay\nlines", "fiber"),
            (5.85, mesh_label, "chip"),
        ]
        for x, label, key in main:
            box(x, 2.05, 1.1, 0.9, label, colors[key])
        for left, right in zip(main[:-1], main[1:], strict=True):
            arrow(left[0] + 1.1, 2.5, right[0], 2.5)

        branch_x = main[-1][0] + 1.1
        branches = [
            (7.55, 3.65, f"Bright tracker\n{_count_label(n_bright, 'port')}", "chip"),
            (7.55, 2.05, f"Science dark\n{_count_label(n_dark, 'port')}", "chip"),
            (7.55, 0.65, f"Spectroscopy\n{_count_label(n_spectro, 'port')}", "chip"),
        ]
        for x, y, label, key in branches:
            arrow(branch_x, 2.5, x, y + 0.45)
            box(x, y, 1.35, 0.9, label, colors[key])

        box(9.55, 3.65, 1.25, 0.9, "SNSPD\ncounts", colors["detector"])
        box(9.55, 2.05, 1.25, 0.9, "MKID\nphotons", colors["detector"])
        box(9.55, 0.65, 1.25, 0.9, "MKID\nspectrum", colors["detector"])
        for _, y, _label, _key in branches:
            arrow(8.9, y + 0.45, 9.55, y + 0.45)

        box(11.65, 3.65, 1.55, 0.9, "Piston\nestimator", colors["control"])
        arrow(10.8, 4.1, 11.65, 4.1)
        arrow(12.4, 3.65, 12.4, 2.95)
        ax.text(
            12.55,
            3.28,
            "cophasing\nhandoff",
            ha="left",
            va="center",
            fontsize=9,
            color="0.3",
        )

        ax.text(
            9.6,
            4.75,
            "fringe tracking",
            ha="center",
            va="center",
            fontsize=10,
            color="0.35",
            style="italic",
        )
        ax.text(
            9.6,
            1.75,
            "planet detection and characterization",
            ha="center",
            va="center",
            fontsize=10,
            color="0.35",
            style="italic",
        )
        ax.set_title("KILO signal chain in nullsim")
        path = paper_dir / "signal_chain.pdf"
        fig.savefig(path)
        plt.close(fig)
    return path


def _write_array_layout(collection: RunCollection, paper_dir: Path) -> Path:
    from matplotlib import pyplot as plt

    from nullsim.outputs.styles import publication_style

    positions = _array_positions_enu_m(collection)
    with publication_style():
        fig, ax = plt.subplots(figsize=(7.0, 7.0))
        if positions is None or positions.shape[0] < 1:
            ax.text(
                0.5,
                0.5,
                "no array positions available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
        else:
            xy = positions[:, :2]
            ax.scatter(xy[:, 0], xy[:, 1], s=180, c="C0", edgecolors="black", zorder=10)
            for idx, (east, north) in enumerate(xy):
                ax.annotate(
                    _aperture_label(idx, collection),
                    (east, north),
                    xytext=(8, 7),
                    textcoords="offset points",
                    fontsize=10,
                    ha="left",
                    va="bottom",
                )
            for i, j in combinations(range(xy.shape[0]), 2):
                dx, dy = xy[j] - xy[i]
                length = float(np.hypot(dx, dy))
                ax.plot(
                    [xy[i, 0], xy[j, 0]],
                    [xy[i, 1], xy[j, 1]],
                    color="0.15",
                    alpha=0.28,
                    linewidth=1.0,
                )
                if length > 0:
                    mid = 0.5 * (xy[i] + xy[j])
                    angle = float(np.degrees(np.arctan2(dy, dx)))
                    label_angle = angle if -90.0 <= angle <= 90.0 else angle + 180.0
                    ax.text(
                        mid[0],
                        mid[1],
                        f"{length:.0f} m",
                        fontsize=8,
                        color="0.35",
                        ha="center",
                        va="center",
                        rotation=label_angle,
                        bbox={
                            "facecolor": "white",
                            "edgecolor": "none",
                            "alpha": 0.7,
                            "pad": 1.5,
                        },
                    )
            ax.set_xlabel("East (m)")
            ax.set_ylabel("North (m)")
            ax.set_title("KILO array layout")
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True, alpha=0.25)
        fig.tight_layout()
        path = paper_dir / "array_layout.pdf"
        fig.savefig(path)
        plt.close(fig)
    return path


def _write_baseline_histogram(collection: RunCollection, paper_dir: Path) -> Path:
    from matplotlib import pyplot as plt

    from nullsim.outputs.styles import publication_style

    input_positions = _input_mode_positions_m(collection)
    array_positions = _array_positions_enu_m(collection)
    input_lengths = _pairwise_lengths(input_positions)
    telescope_lengths = _pairwise_lengths(
        array_positions[:, :2] if array_positions is not None else None
    )

    with publication_style():
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), width_ratios=[1.15, 1.0])
        if input_lengths.size:
            axes[0].hist(
                input_lengths,
                bins=min(30, max(6, int(np.sqrt(input_lengths.size)))),
                color="C0",
                alpha=0.82,
                edgecolor="white",
            )
            axes[0].set_title(f"Input-mode baselines ({input_lengths.size})")
            axes[0].set_xlabel("Baseline length (m)")
            axes[0].set_ylabel("Count")
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].text(
                0.5,
                0.5,
                "no input-mode baselines",
                ha="center",
                va="center",
                transform=axes[0].transAxes,
            )
            axes[0].set_axis_off()

        if telescope_lengths.size:
            axes[1].hist(
                telescope_lengths,
                bins=min(20, max(4, telescope_lengths.size)),
                color="C3",
                alpha=0.82,
                edgecolor="white",
            )
            axes[1].set_title(f"Array baselines ({telescope_lengths.size})")
            axes[1].set_xlabel("Baseline length (m)")
            axes[1].set_ylabel("Count")
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(
                0.5,
                0.5,
                "no array baselines",
                ha="center",
                va="center",
                transform=axes[1].transAxes,
            )
            axes[1].set_axis_off()
        fig.suptitle("Baseline distribution")
        fig.tight_layout()
        path = paper_dir / "baseline_histogram.pdf"
        fig.savefig(path)
        plt.close(fig)
    return path


def _write_cophasing_scheme(collection: RunCollection, paper_dir: Path) -> Path:
    from matplotlib import pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    from nullsim.outputs.styles import publication_style

    first = _first_record(collection)
    chip = _find_component(first, "chip.") if first is not None else None
    fringe = (
        _find_component(first, "fringe_observability") if first is not None else None
    )

    n_inputs = _int_from_payload(chip, "n_input_modes") or _int_from_payload(
        chip, "n_modes"
    )
    n_physical = _int_from_payload(
        fringe, "n_physical_piston_modes"
    ) or _int_from_payload(
        first.result.state.geometry.components.get("pupil_modes") if first else None,
        "n_modes",
    )
    n_bright = _int_from_payload(chip, "n_bright_ports") or _int_from_payload(
        fringe, "n_bright_ports"
    )
    n_dark = _int_from_payload(chip, "n_dark_ports")
    rank = _int_from_payload(fringe, "rank")
    dof = _int_from_payload(fringe, "n_differential_piston_modes")
    crlb = _float_from_payload(fringe, "crlb_piston_rms_nm")
    condition = _float_from_payload(fringe, "condition_number")
    gn = _nested_payload(fringe, "gauss_newton_monte_carlo")
    gn_median = _float_from_payload(gn, "median_residual_rms_nm")
    handoff = _nested_payload(fringe, "cophasing_handoff_monte_carlo")
    handoff_science = _float_from_payload(handoff, "median_science_residual_rms_nm")
    handoff_lock = _float_from_payload(handoff, "science_lock_fraction")
    handoff_wrap = _float_from_payload(handoff, "edge_wrap_fraction")

    with publication_style():
        fig, ax = plt.subplots(figsize=(12.0, 5.8))
        ax.set_xlim(0.0, 12.0)
        ax.set_ylim(0.0, 6.0)
        ax.axis("off")

        colors = {
            "input": "#4A90D9",
            "mesh": "#E5A432",
            "tracker": "#7B6FD6",
            "science": "#6DBB75",
            "diagnostic": "#C84F4F",
        }

        def box(x: float, y: float, w: float, h: float, label: str, color: str) -> None:
            patch = FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.14",
                facecolor=color,
                edgecolor="black",
                alpha=0.88,
                linewidth=1.0,
            )
            ax.add_patch(patch)
            ax.text(
                x + w / 2.0,
                y + h / 2.0,
                label,
                ha="center",
                va="center",
                fontsize=9.5,
                fontweight="bold",
                color="white",
            )

        def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
            ax.add_patch(
                FancyArrowPatch(
                    (x1, y1),
                    (x2, y2),
                    arrowstyle="-|>",
                    mutation_scale=13,
                    color="0.25",
                    linewidth=1.2,
                    shrinkA=5,
                    shrinkB=5,
                )
            )

        box(
            0.35,
            2.45,
            1.65,
            1.1,
            f"Aperture\npistons\n{_count_label(n_physical, 'mode')}",
            colors["input"],
        )
        box(
            2.65,
            2.45,
            1.95,
            1.1,
            f"Reference\nmesh\n{_count_label(n_inputs, 'input')}",
            colors["mesh"],
        )
        box(
            5.25,
            3.85,
            1.75,
            1.0,
            f"Bright\ntracker\n{_count_label(n_bright, 'port')}",
            colors["tracker"],
        )
        box(
            5.25,
            1.15,
            1.75,
            1.0,
            f"Dark science\n{_count_label(n_dark, 'port')}",
            colors["science"],
        )
        box(7.75, 3.85, 1.8, 1.0, "SNSPD\ncount stream", colors["diagnostic"])
        box(7.75, 2.45, 1.8, 1.0, "Response\nmatrix", colors["diagnostic"])
        box(9.95, 2.45, 1.65, 1.0, "Piston\nsolution", colors["tracker"])

        arrow(2.0, 3.0, 2.65, 3.0)
        arrow(4.6, 3.0, 5.25, 4.35)
        arrow(4.6, 3.0, 5.25, 1.65)
        arrow(7.0, 4.35, 7.75, 4.35)
        arrow(7.0, 4.35, 7.75, 2.95)
        arrow(9.55, 2.95, 9.95, 2.95)
        arrow(10.75, 2.45, 10.75, 1.35)
        ax.text(
            10.95, 1.9, "feed-forward\ncophasing", fontsize=9, color="0.3", va="center"
        )

        metric_lines = [
            f"rank: {_rank_label(rank, dof)}",
            f"condition: {_number_label(condition)}",
            f"Cramer-Rao lower bound: {_nm_label(crlb)}",
            f"Gauss-Newton median: {_nm_label(gn_median)}",
        ]
        if handoff_science is not None:
            metric_lines.append(
                f"science residual after handoff: {_nm_label(handoff_science)}"
            )
        if handoff_lock is not None:
            metric_lines.append(
                f"science lock fraction: {_percent_label(handoff_lock)}"
            )
        if handoff_wrap is not None:
            metric_lines.append(f"edge wrap fraction: {_percent_label(handoff_wrap)}")
        ax.text(
            0.45,
            0.45,
            "\n".join(metric_lines),
            ha="left",
            va="bottom",
            fontsize=10,
            family="monospace",
            bbox={
                "facecolor": "0.96",
                "edgecolor": "0.75",
                "boxstyle": "round,pad=0.35",
            },
        )
        ax.set_title("Cophasing and fringe-tracker information flow")
        path = paper_dir / "cophasing_scheme.pdf"
        fig.savefig(path)
        plt.close(fig)
    return path


def _write_dirty_beams(collection: RunCollection, paper_dir: Path) -> Path:
    from matplotlib import pyplot as plt

    from nullsim.outputs.styles import publication_style

    positions = _input_mode_positions_m(collection)
    wl_center_m = float(collection.config.grid.wavelength_center_um) * 1.0e-6
    wavelengths_m = _wavelength_grid_m(collection)
    with publication_style():
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7), squeeze=False)
        axes = axes[0]
        if positions is None or positions.shape[0] < 2:
            for ax in axes:
                ax.text(
                    0.5,
                    0.5,
                    "no input-mode baselines",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_axis_off()
        else:
            fov_mas = _dirty_beam_fov_mas(positions, wl_center_m)
            coords_mas = np.linspace(-fov_mas, fov_mas, 241)
            design = _dirty_beam(positions, (wl_center_m,), coords_mas)
            band = _dirty_beam(positions, tuple(wavelengths_m), coords_mas)
            for ax, beam, title in (
                (axes[0], design, f"Design wavelength ({wl_center_m * 1e6:.3f} um)"),
                (axes[1], band, "Band averaged"),
            ):
                im = ax.imshow(
                    beam,
                    origin="lower",
                    extent=(-fov_mas, fov_mas, -fov_mas, fov_mas),
                    cmap="inferno",
                    vmin=0.0,
                    vmax=min(1.0, float(np.percentile(beam, 99.7))),
                )
                ax.set_title(title)
                ax.set_xlabel(r"$\Delta\alpha$ (mas; East left)")
                ax.set_ylabel(r"$\Delta\delta$ (mas)")
                ax.grid(alpha=0.15, color="white")
                fig.colorbar(
                    im, ax=ax, fraction=0.046, pad=0.04, label="Normalized response"
                )
        fig.suptitle("Dirty beam from nullsim input-mode baselines")
        fig.tight_layout()
        path = paper_dir / "dirty_beams.pdf"
        fig.savefig(path)
        plt.close(fig)
    return path


def _write_chip_diagnostics_characterization(
    collection: RunCollection,
    paper_dir: Path,
) -> Path:
    from nullsim.outputs.plots.chip_diagnostics import (
        plot_characterization_chip_diagnostics_for_record,
    )

    record = _first_record(collection)
    if record is not None:
        path = paper_dir / "chip_diagnostics_characterization.pdf"
        fig = plot_characterization_chip_diagnostics_for_record(record)
        if fig is not None:
            fig.savefig(path)
            from matplotlib import pyplot as plt

            plt.close(fig)
            return path

    from matplotlib import pyplot as plt

    from nullsim.outputs.styles import publication_style

    char = _find_component(record, "characterization.spectro") if record else None
    chip = _find_component(record, "chip.") if record else None
    sensitivity = (
        _find_component(record, "sensitivity.detection_curve") if record else None
    )
    n_bright = _int_from_payload(chip, "n_bright_ports") or 0
    n_dark = _int_from_payload(chip, "n_dark_ports") or 0
    n_spectro = _int_from_payload(chip, "n_spectro_ports") or 0
    n_ports = n_bright + n_dark + n_spectro
    port_index = np.arange(n_ports, dtype=int)

    if sensitivity is not None and "planet_throughputs" in sensitivity:
        throughput = np.asarray(sensitivity["planet_throughputs"], dtype=np.float64)
        detection_planet = float(np.nanmedian(throughput)) if throughput.size else 0.0
    else:
        detection_planet = float("nan")
    detection_leak = _float_from_payload(sensitivity, "effective_null_with_disk")
    char_planet = _first_float(
        char,
        "spectro_throughput",
        "char_spectro_throughput",
        default=float("nan"),
    )
    char_leak = _first_float(
        char,
        "spectro_leak_fraction",
        "spectro_disk_leak",
        "spectro_null_depth",
        default=float("nan"),
    )

    with publication_style():
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
        ax_ports, ax_summary = axes
        if n_ports > 0:
            colors = (
                ["#e4aa2f"] * n_bright + ["#2f78b7"] * n_dark + ["#7b6fd6"] * n_spectro
            )
            values = np.full(n_ports, np.nan)
            if chip is not None and chip.get("stellar_power_per_port") is not None:
                raw = np.asarray(chip["stellar_power_per_port"], dtype=np.float64)
                values[: min(raw.size, n_ports)] = raw[: min(raw.size, n_ports)]
            else:
                values[:n_bright] = 1.0 / max(n_bright, 1)
                values[n_bright : n_bright + n_dark] = max(
                    detection_leak or 0.0, 1.0e-12
                )
                values[n_bright + n_dark :] = max(char_leak, 1.0e-12)
            ax_ports.bar(port_index, np.maximum(values, 1.0e-12), color=colors)
            ax_ports.set_yscale("log")
            ax_ports.set_xlabel("Output port index")
            ax_ports.set_ylabel("Fractional stellar power")
            ax_ports.set_title("Detection-chip port allocation")
            ax_ports.grid(axis="y", alpha=0.3)
        else:
            ax_ports.text(
                0.5,
                0.5,
                "no chip payload",
                ha="center",
                va="center",
                transform=ax_ports.transAxes,
            )
            ax_ports.set_axis_off()

        labels = [
            "Detection\nplanet",
            "Detection\nleak",
            "Spectro\nplanet",
            "Spectro\nleak",
        ]
        vals = [detection_planet, detection_leak, char_planet, char_leak]
        ax_summary.bar(
            np.arange(len(vals)),
            [
                max(float(v), 1.0e-12) if v is not None and np.isfinite(v) else np.nan
                for v in vals
            ],
            color=["#6dbb75", "#c84f4f", "#6dbb75", "#c84f4f"],
        )
        ax_summary.set_yscale("log")
        ax_summary.set_xticks(np.arange(len(vals)))
        ax_summary.set_xticklabels(labels)
        ax_summary.set_ylabel("Fraction")
        ax_summary.set_title("Detection vs characterization channels")
        ax_summary.grid(axis="y", alpha=0.3)
        if char is None:
            ax_summary.text(
                0.5,
                0.92,
                "No characterization.spectro component in this run",
                ha="center",
                va="center",
                transform=ax_summary.transAxes,
                fontsize=9,
                color="0.35",
            )
        fig.tight_layout()
        path = paper_dir / "chip_diagnostics_characterization.pdf"
        fig.savefig(path)
        plt.close(fig)
    return path


def _write_planet_polarization_phase(
    _collection: RunCollection,
    paper_dir: Path,
) -> Path:
    from matplotlib import pyplot as plt

    from nullsim.outputs.styles import publication_style

    phase_deg = np.linspace(0.0, 180.0, 361)
    phase_rad = np.radians(phase_deg)
    with publication_style():
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.plot(
            phase_deg,
            _planet_polarization_fraction(phase_rad, 1.0),
            color="#3377aa",
            linewidth=2,
            label=(
                rf"Earth-like ($R < {_POL_MODEL_CROSSOVER_REARTH:.1f}\,R_\oplus$, "
                rf"$|P|_\mathrm{{max}} = {_EARTHLIKE_PEAK_POL_FRAC:.2f}$)"
            ),
        )
        ax.plot(
            phase_deg,
            _planet_polarization_fraction(phase_rad, 5.0),
            color="#cc5544",
            linewidth=2,
            label=(
                rf"Cloudy giant ($R \geq {_POL_MODEL_CROSSOVER_REARTH:.1f}\,R_\oplus$, "
                rf"$|P|_\mathrm{{max}} = {_CLOUDY_GIANT_PEAK_POL_FRAC:.2f}$)"
            ),
        )
        ax.axvline(90.0, color="0.45", linestyle="--", alpha=0.6, label="quadrature")
        ax.set_xlabel("Phase angle (degrees)")
        ax.set_ylabel(r"Linear polarization fraction $|P|$")
        ax.set_xlim(0.0, 180.0)
        ax.set_ylim(0.0, 0.55)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        path = paper_dir / "planet_polarization_phase.pdf"
        fig.savefig(path)
        plt.close(fig)
    return path


def _write_dualpol_optical_layout(
    _collection: RunCollection,
    paper_dir: Path,
) -> Path:
    from matplotlib import pyplot as plt

    from nullsim.outputs.styles import publication_style

    with publication_style():
        fig, ax = plt.subplots(figsize=(9.0, 5.5))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 8)
        ax.axis("off")

        def box(x, y, w, h, label, color):
            ax.add_patch(
                plt.Rectangle(
                    (x, y), w, h, facecolor=color, edgecolor="black", linewidth=1.1
                )
            )
            ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)

        def arrow(x1, y1, x2, y2, label=None):
            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops={"arrowstyle": "->", "lw": 1.1},
            )
            if label:
                ax.text(
                    (x1 + x2) / 2,
                    (y1 + y2) / 2 + 0.18,
                    label,
                    ha="center",
                    fontsize=8,
                    style="italic",
                )

        box(0.2, 3.5, 1.6, 1.0, "AO-corrected\npupil", "#cce5ff")
        box(2.4, 3.5, 1.4, 1.0, "Re-imaged\npupil", "#cce5ff")
        box(4.2, 3.5, 0.8, 1.0, "HWP", "#fff2cc")
        box(5.4, 3.5, 0.8, 1.0, "PBS", "#fff2cc")
        arrow(1.8, 4.0, 2.4, 4.0)
        arrow(3.8, 4.0, 4.2, 4.0)
        arrow(5.0, 4.0, 5.4, 4.0)
        for y, arm in ((5.5, "A"), (1.5, "B")):
            box(7.0, y, 1.6, 1.0, f"PM fiber {arm}", "#ffe5cc")
            box(9.0, y, 1.6, 1.0, f"Delay line {arm}", "#ffe5cc")
            box(11.0, y, 1.8, 1.0, f"KILOPIC\nchip {arm}", "#ffe5cc")
            box(13.0, y, 0.8, 1.0, f"MKIDs\n{arm}", "#ffcccc")
            arrow(6.2, 4.5 if arm == "A" else 3.5, 7.0, y + 0.5, label=f"pol {arm}")
            arrow(8.6, y + 0.5, 9.0, y + 0.5)
            arrow(10.6, y + 0.5, 11.0, y + 0.5)
            arrow(12.8, y + 0.5, 13.0, y + 0.5)
        ax.annotate(
            "",
            xy=(13.4, 5.5),
            xytext=(13.4, 2.5),
            arrowprops={"arrowstyle": "<->", "lw": 1.0, "color": "purple"},
        )
        ax.text(
            13.7,
            4.0,
            "PDI:\nA - B\nat 4 f_HWP",
            color="purple",
            fontsize=8,
            ha="left",
            va="center",
        )
        ax.set_title("Dual-polarization optical layout")
        path = paper_dir / "dualpol_optical_layout.pdf"
        fig.savefig(path)
        plt.close(fig)
    return path


def _write_signal_chain_dualpol(
    _collection: RunCollection,
    paper_dir: Path,
) -> Path:
    from matplotlib import pyplot as plt

    from nullsim.outputs.styles import publication_style

    with publication_style():
        fig, ax = plt.subplots(figsize=(11.0, 3.2))
        ax.set_xlim(0, 22)
        ax.set_ylim(0, 4)
        ax.axis("off")

        def stage(x, label, color="#cce5ff", w=2.2):
            ax.add_patch(
                plt.Rectangle(
                    (x, 1.3), w, 1.4, facecolor=color, edgecolor="black", linewidth=1.0
                )
            )
            ax.text(x + w / 2, 2.0, label, ha="center", va="center", fontsize=8.5)
            return x + w

        def arrow(x1, x2):
            ax.annotate(
                "",
                xy=(x2, 2.0),
                xytext=(x1, 2.0),
                arrowprops={"arrowstyle": "->", "lw": 1.0},
            )

        x = 0.2
        for label, color, width in (
            ("Telescope\n+ AO", "#cce5ff", 1.9),
            ("Re-imaged\npupil", "#cce5ff", 1.8),
            ("HWP", "#fff2cc", 1.2),
            ("PBS\nsplit", "#fff2cc", 1.5),
            ("PM fibers\nx2", "#ffe5cc", 1.8),
            ("Delay lines\nx2", "#ffe5cc", 1.9),
            ("Photonic\nchips x2", "#ffe5cc", 2.0),
            ("MKIDs", "#ffcccc", 1.5),
            ("PDI signal\n4 f_HWP", "#e8d5e8", 2.0),
        ):
            end = stage(x, label, color, width)
            if label != "PDI signal\n4 f_HWP":
                arrow(end, end + 0.35)
            x = end + 0.35
        ax.set_title("KILO signal chain with dual-polarization differential imaging")
        path = paper_dir / "signal_chain_dualpol.pdf"
        fig.savefig(path)
        plt.close(fig)
    return path


def _write_exoplanet_survey(
    collection: RunCollection,
    paper_dir: Path,
) -> tuple[Path, ...]:
    import csv

    from nullsim.outputs.plots.detection_space import _BUNDLED_PLANET_CATALOG

    rows = _survey_rows(collection)
    archive_path = paper_dir / "nasa_exoplanet_archive.csv"
    shutil.copy2(_BUNDLED_PLANET_CATALOG, archive_path)

    csv_path = paper_dir / "exoplanet_table.csv"
    fieldnames = _exoplanet_table_columns()
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    results = {
        "config": _survey_config_summary(collection),
        "all_rows": rows,
        "top_by_category": _top_rows_by_category(rows, n=10),
        "top_by_category_char": _top_rows_by_char_snr(rows, n=10),
        "source_model": (
            "nullsim paper survey: bundled NASA archive filtered like detection_space; "
            "SNR is scaled from the active run's 5-sigma contrast curves."
        ),
    }
    json_path = paper_dir / "exoplanet_table_results.json"
    json_path.write_text(
        json.dumps(_json_safe(results), indent=2, sort_keys=True), encoding="utf-8"
    )

    tex_path = paper_dir / "exoplanet_table.tex"
    tex_path.write_text(
        _latex_exoplanet_table(results["top_by_category"]), encoding="utf-8"
    )
    char_tex_path = paper_dir / "characterization_table.tex"
    char_tex_path.write_text(
        _latex_characterization_table(results["top_by_category_char"]),
        encoding="utf-8",
    )
    return archive_path, csv_path, tex_path, char_tex_path, json_path


def _write_pdi_products(
    collection: RunCollection,
    paper_dir: Path,
) -> tuple[Path, ...]:
    import csv

    exoplanet_json = paper_dir / "exoplanet_table_results.json"
    if exoplanet_json.exists():
        data = json.loads(exoplanet_json.read_text(encoding="utf-8"))
        base_rows = list(data.get("all_rows", ()))
    else:
        base_rows = _survey_rows(collection)

    pdi_rows = [_pdi_row(row, collection) for row in base_rows]
    pdi_rows = [row for row in pdi_rows if row is not None]

    # Sort by the SNR metric the table actually displays (detection-mode PDI),
    # sinking non-finite/missing values to the bottom so they never displace
    # finite rows from the top-10 LaTeX/JSON selections. The full CSV/JSON
    # exports still retain every planet.
    pdi_rows.sort(key=_pdi_displayed_snr, reverse=True)

    json_payload = {
        "config": {
            "integration_time_s": _integration_time_s(collection),
            "dual_pol_throughput_vs_scalar": _scalar_to_dual_pol_rate_scale(collection),
            "hwp_rotation_rate_hz": _PDI_HWP_ROTATION_RATE_HZ,
            "pol_model_crossover_rearth": _POL_MODEL_CROSSOVER_REARTH,
        },
        "all_rows": pdi_rows,
        "top_rows": pdi_rows[:10],
    }
    pdi_json = paper_dir / "pdi_exoplanet_table.json"
    pdi_json.write_text(
        json.dumps(_json_safe(json_payload), indent=2, sort_keys=True), encoding="utf-8"
    )

    pdi_csv = paper_dir / "pdi_exoplanet_table.csv"
    fieldnames = _pdi_table_columns()
    with pdi_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in pdi_rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    pdi_tex = paper_dir / "pdi_exoplanet_table.tex"
    pdi_tex.write_text(_latex_pdi_table(pdi_rows[:10]), encoding="utf-8")

    real_rates = paper_dir / "pdi_planet_table_real_rates.json"
    real_rates.write_text(
        json.dumps(_json_safe(pdi_rows), indent=2, sort_keys=True), encoding="utf-8"
    )

    paths = [pdi_json, pdi_csv, pdi_tex, real_rates]
    paths.append(_write_pdi_sensitivity_curves(collection, paper_dir, pdi_rows))
    paths.append(_write_hwp_rate_sweep(collection, paper_dir, pdi_rows))
    paths.append(_write_pdi_planet_table_scatter(collection, paper_dir, pdi_rows))
    return tuple(paths)


def _write_pdi_sensitivity_curves(
    collection: RunCollection,
    paper_dir: Path,
    rows: list[dict[str, Any]],
) -> Path:
    from matplotlib import pyplot as plt

    from nullsim.outputs.styles import publication_style

    targets = _named_or_top_rows(rows, ("tau Boo b", "tau Cet h"), n=2)
    times_s = np.array([60.0, 600.0, 3600.0, 3.6e4, 3.6e5, 3.6e6])
    n_dark = _n_dark_ports_per_arm(collection)
    rate_scale = _scalar_to_dual_pol_rate_scale(collection)
    with publication_style():
        fig, axes = plt.subplots(
            1, max(len(targets), 1), figsize=(11.0, 4.5), squeeze=False
        )
        axes = axes[0]
        if not targets:
            axes[0].text(
                0.5,
                0.5,
                "no PDI survey rows",
                ha="center",
                va="center",
                transform=axes[0].transAxes,
            )
            axes[0].set_axis_off()
        for ax, row in zip(axes, targets, strict=False):
            scalar = []
            ideal = []
            continuous = []
            stepped = []
            for t in times_s:
                scalar.append(_scale_snr_time(float(row["snr_det_4h"]), t, collection))
                ideal.append(
                    _pdi_snr(
                        row,
                        t,
                        modulation_period_s=0.0,
                        n_dark_ports=n_dark,
                        rate_scale=rate_scale,
                        ideal=True,
                    )
                )
                continuous.append(
                    _pdi_snr(
                        row,
                        t,
                        modulation_period_s=1.0 / _PDI_HWP_ROTATION_RATE_HZ,
                        n_dark_ports=n_dark,
                        rate_scale=rate_scale,
                    )
                )
                stepped.append(
                    _pdi_snr(
                        row,
                        t,
                        modulation_period_s=60.0,
                        n_dark_ports=n_dark,
                        rate_scale=rate_scale,
                    )
                )
            t_hr = times_s / 3600.0
            ax.plot(t_hr, scalar, "o-", color="0.35", label="scalar detection")
            ax.plot(t_hr, ideal, "s--", color="#88aa55", label="PDI ideal")
            ax.plot(t_hr, stepped, "v-", color="#cc5544", label="PDI stepped")
            ax.plot(t_hr, continuous, "^-", color="#3377aa", label="PDI continuous")
            ax.axhline(5.0, color="black", linestyle=":", alpha=0.5)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Integration time (hours)")
            ax.set_ylabel("SNR")
            ax.set_title(
                rf"{row['name']}" "\n"
                rf"$R={float(row['radius_rearth']):.1f}\,R_\oplus$, "
                rf"$|P|={float(row['pol_fraction_orbit_avg']):.2f}$",
                fontsize=9,
            )
            ax.grid(alpha=0.3, which="both")
            ax.legend(loc="lower right", fontsize=7)
        fig.suptitle("Polarimetric differential imaging SNR vs integration time")
        fig.tight_layout()
        path = paper_dir / "pdi_sensitivity_curves.pdf"
        fig.savefig(path)
        plt.close(fig)
    return path


def _write_hwp_rate_sweep(
    collection: RunCollection,
    paper_dir: Path,
    rows: list[dict[str, Any]],
) -> Path:
    from matplotlib import pyplot as plt

    from nullsim.outputs.styles import publication_style

    targets = _named_or_top_rows(rows, ("tau Boo b", "tau Cet h"), n=2)
    rates_hz = np.logspace(0.0, 4.0, 41)
    tint = _integration_time_s(collection)
    n_dark = _n_dark_ports_per_arm(collection)
    rate_scale = _scalar_to_dual_pol_rate_scale(collection)
    with publication_style():
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        if not targets:
            ax.text(
                0.5,
                0.5,
                "no PDI survey rows",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        for row, color in zip(targets, ("#cc5544", "#3377aa"), strict=False):
            snrs = [
                _pdi_snr(
                    row,
                    tint,
                    modulation_period_s=1.0 / float(rate),
                    n_dark_ports=n_dark,
                    rate_scale=rate_scale,
                )
                for rate in rates_hz
            ]
            ax.semilogx(
                rates_hz, snrs, "-", color=color, linewidth=2, label=row["name"]
            )
            ax.axhline(
                _pdi_snr(
                    row,
                    tint,
                    modulation_period_s=60.0,
                    n_dark_ports=n_dark,
                    rate_scale=rate_scale,
                ),
                linestyle="--",
                alpha=0.6,
                color=color,
                label=f"{row['name']} stepped",
            )
        ax.axhline(5.0, color="black", linestyle=":", alpha=0.5)
        ax.set_xlabel("HWP rotation rate (Hz)")
        ax.set_ylabel("PDI SNR")
        ax.set_title("PDI SNR vs HWP modulation rate")
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        path = paper_dir / "hwp_rate_sweep.pdf"
        fig.savefig(path)
        plt.close(fig)
    return path


def _write_pdi_planet_table_scatter(
    _collection: RunCollection,
    paper_dir: Path,
    rows: list[dict[str, Any]],
) -> Path:
    from matplotlib import pyplot as plt

    from nullsim.outputs.styles import publication_style

    with publication_style():
        fig, ax = plt.subplots(figsize=(7.2, 5.8))
        if rows:
            det = np.asarray(
                [float(row["snr_det_4h"]) for row in rows], dtype=np.float64
            )
            pdi = np.asarray(
                [float(row["snr_pdi_continuous_4h"]) for row in rows], dtype=np.float64
            )
            radii = np.asarray(
                [float(row["radius_rearth"]) for row in rows], dtype=np.float64
            )
            earthlike = radii < _POL_MODEL_CROSSOVER_REARTH
            ax.scatter(
                np.maximum(det[~earthlike], _LOG_SCATTER_FLOOR),
                np.maximum(pdi[~earthlike], _LOG_SCATTER_FLOOR),
                s=18,
                color="#cc5544",
                alpha=0.5,
                label=f"Cloudy giants, n={int((~earthlike).sum())}",
            )
            ax.scatter(
                np.maximum(det[earthlike], _LOG_SCATTER_FLOOR),
                np.maximum(pdi[earthlike], _LOG_SCATTER_FLOOR),
                s=26,
                color="#3377aa",
                alpha=0.7,
                label=f"Earth-like, n={int(earthlike.sum())}",
            )
            hi = max(float(np.nanmax(det)), float(np.nanmax(pdi)), 5.0) * 1.3
            ax.plot(
                [_LOG_SCATTER_FLOOR, hi],
                [_LOG_SCATTER_FLOOR, hi],
                "k--",
                alpha=0.4,
                linewidth=1,
            )
            ax.set_xlim(_LOG_SCATTER_FLOOR, hi)
            ax.set_ylim(_LOG_SCATTER_FLOOR, hi)
            ax.set_xscale("log")
            ax.set_yscale("log")
        else:
            ax.text(
                0.5,
                0.5,
                "no PDI survey rows",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.axhline(5.0, color="green", linestyle=":", alpha=0.5)
        ax.axvline(5.0, color="green", linestyle=":", alpha=0.5)
        ax.set_xlabel("Scalar detection SNR")
        ax.set_ylabel("Continuous-mode PDI SNR")
        ax.set_title("PDI survey: scalar vs polarized detection")
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        path = paper_dir / "pdi_planet_table_scatter.pdf"
        fig.savefig(path)
        plt.close(fig)
    return path


def _array_positions_enu_m(collection: RunCollection) -> np.ndarray | None:
    record = _first_record(collection)
    if record is None:
        return None
    geometry = record.result.state.geometry
    for value in geometry.components.values():
        if (
            isinstance(value, Mapping)
            and value.get("aperture_positions_enu_m") is not None
        ):
            arr = np.asarray(value["aperture_positions_enu_m"], dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] in {2, 3}:
                return _ensure_enu(arr)
    if geometry.aperture_positions_enu_m is not None:
        arr = np.asarray(geometry.aperture_positions_enu_m, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] in {2, 3}:
            return _ensure_enu(arr)
    return None


def _input_mode_positions_m(collection: RunCollection) -> np.ndarray | None:
    record = _first_record(collection)
    if record is None:
        return None
    state = record.result.state
    if state.geometry.aperture_positions_pupil_m is not None:
        arr = np.asarray(state.geometry.aperture_positions_pupil_m, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, :2].copy()
    if state.geometry.aperture_positions_enu_m is not None:
        arr = np.asarray(state.geometry.aperture_positions_enu_m, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, :2].copy()
    return None


def _ensure_enu(arr: np.ndarray) -> np.ndarray:
    if arr.shape[1] == 3:
        return arr.copy()
    out = np.zeros((arr.shape[0], 3), dtype=np.float64)
    out[:, :2] = arr[:, :2]
    return out


def _pairwise_lengths(positions: np.ndarray | None) -> np.ndarray:
    if positions is None or positions.shape[0] < 2:
        return np.array([], dtype=np.float64)
    lengths = [
        float(np.linalg.norm(positions[j, :2] - positions[i, :2]))
        for i, j in combinations(range(positions.shape[0]), 2)
    ]
    return np.asarray(lengths, dtype=np.float64)


def _first_record(collection: RunCollection) -> RunRecord | None:
    return collection.records[0] if collection.records else None


def _find_component(record: RunRecord | None, prefix: str) -> dict[str, Any] | None:
    """The newest component whose name starts with ``prefix``.

    Newest first, like the stages' selectors: a later stage instance
    supersedes an earlier one whether or not its name carries a suffix.
    """

    name = None if record is None else _component_name(record, prefix)
    return None if name is None else dict(record.result.state.results.components[name])


def _chip_payload_summary(collection: RunCollection) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for record in collection.records:
        chip = _find_component(record, "chip.")
        if chip is None:
            continue
        summaries.append(
            {
                "coord": dict(record.coord_tuple),
                "component": _component_name(record, "chip."),
                "payload": _json_safe(chip),
            }
        )
    return {"records": summaries}


def _uv_coverage_summary(collection: RunCollection) -> dict[str, Any]:
    positions = _array_positions_enu_m(collection)
    input_positions = _input_mode_positions_m(collection)
    array_lengths = _pairwise_lengths(
        positions[:, :2] if positions is not None else None
    )
    input_lengths = _pairwise_lengths(input_positions)
    return {
        "array_positions_enu_m": _json_safe(positions),
        "n_array_baselines": int(array_lengths.size),
        "array_baseline_range_m": _range_or_none(array_lengths),
        "n_input_mode_baselines": int(input_lengths.size),
        "input_mode_baseline_range_m": _range_or_none(input_lengths),
    }


def _run_payload_summary(collection: RunCollection) -> dict[str, Any]:
    rows = []
    for record in collection.records:
        state = record.result.state
        rows.append(
            {
                "coord": dict(record.coord_tuple),
                "config_hash": record.config.config_hash,
                "run_hash": record.config.run_hash,
                "null_depth": state.results.null_depth,
                "snr": state.results.snr,
                "calibration_time_s": state.results.calibration_time_s,
                "cache_hit_count": record.result.cache_hit_count,
            }
        )
    return {"records": _json_safe(rows)}


def _loss_budget_summary(collection: RunCollection) -> dict[str, Any]:
    rows = []
    for record in collection.records:
        rows.append(
            {
                "coord": dict(record.coord_tuple),
                "components": _json_safe(record.result.state.throughput.components),
                "total_transmission_fraction": _json_safe(
                    record.result.state.throughput.transmission_fraction
                ),
            }
        )
    return {"records": rows}


def _survey_rows(collection: RunCollection) -> list[dict[str, Any]]:
    import pandas as pd

    from nullsim.outputs.plots._kilo_planet_overlay import (
        GEOMETRIC_ALBEDO_DEFAULT,
        LAT_KECK_DEG,
        chen_kipping_2017_radius,
        classify_by_radius,
        lambertian_orbit_averaged_phase,
        load_planets,
    )
    from nullsim.outputs.plots.detection_space import _BUNDLED_PLANET_CATALOG

    planets = load_planets(
        _BUNDLED_PLANET_CATALOG,
        latitude_deg=LAT_KECK_DEG,
        min_altitude_deg=45.0,
        geometric_albedo=GEOMETRIC_ALBEDO_DEFAULT,
        max_distance_pc=50.0,
    )
    source = pd.read_csv(_BUNDLED_PLANET_CATALOG, comment="#")
    by_name = {
        str(row.get("pl_name") or "").strip(): row for _, row in source.iterrows()
    }
    rows: list[dict[str, Any]] = []
    phi = lambertian_orbit_averaged_phase()
    # Whether this run produces separation-resolved characterization curves at
    # all (some cells may be skipped via skip_jmags); used to suppress the
    # single-scene scalar char SNR fallback on cells whose curve was skipped.
    curve_expected = any(
        _find_component(cell, "characterization.detection_curve") is not None
        for cell in collection.records
    )
    sweep_jmags = [
        jmag for cell in collection.records if (jmag := _record_jmag(cell)) is not None
    ]
    for planet in planets:
        raw = by_name.get(planet.name)
        record = _record_for_jmag(collection, _float_value(raw, "sy_jmag"))
        sensitivity = _find_component(record, "sensitivity.detection_curve")
        # The sweep simulates a discrete J-mag grid; _record_for_jmag picks the
        # nearest cell. Rescale that cell's stellar photon rate to the planet's
        # actual host J mag (first-order flux scaling 10^(-0.4*dJ)). This fixes
        # the reported photon-rate columns and characterization SNR magnitudes;
        # snr_det_4h is contrast-curve-based and still uses the nearest cell.
        record_jmag = _record_jmag(record)
        target_jmag = _float_value(raw, "sy_jmag")
        jmag_flux_scale = (
            10.0 ** (-0.4 * (target_jmag - record_jmag))
            if record_jmag is not None and target_jmag is not None
            else 1.0
        )
        reflected_contrast = float(planet.contrast)
        limit = _contrast_limit_at(record, planet.separation_mas)
        snr_det = (
            0.0
            if limit is None or not np.isfinite(limit) or limit <= 0
            else 5.0 * reflected_contrast / limit
        )
        throughput_dark = _interp_payload(
            sensitivity,
            "separations_mas",
            "planet_throughputs",
            planet.separation_mas,
            default=1.0,
        )
        realized_null = _first_float(
            sensitivity,
            "realized_null_depth_kilo",
            "realized_null_depth",
            default=0.0,
        )
        effective_null = _interp_payload(
            sensitivity,
            "separations_mas",
            "effective_null_with_disk_by_separation",
            planet.separation_mas,
            default=_first_float(
                sensitivity, "effective_null_with_disk", default=realized_null
            ),
        )
        stellar_rate = _stellar_rate_hz(record) * jmag_flux_scale
        planet_rate = stellar_rate * reflected_contrast * max(throughput_dark, 0.0)
        leakage_rate = stellar_rate * max(effective_null, 0.0)
        char = _find_component(record, "characterization.spectro")
        char_curve = _find_component(record, "characterization.detection_curve")
        char_record = record
        char_rate_scale = jmag_flux_scale
        char_floor_scaled_from_jmag = float("nan")
        if char_curve is None and curve_expected:
            # The nearest J-mag cell skipped its characterization curve
            # (skip_jmags); borrow the nearest curve-bearing cell's chip
            # (throughput, static leak) and rescale the stellar rate to the
            # host magnitude. The realized floor, eps, and systematic
            # bandwidth depend on J, so they come from the host's own cell.
            curve_records = [
                r
                for r in collection.records
                if _find_component(r, "characterization.detection_curve")
                is not None
            ]
            if curve_records and target_jmag is not None:
                char_record = min(
                    curve_records,
                    key=lambda r: abs(
                        (
                            target_jmag
                            if _record_jmag(r) is None
                            else _record_jmag(r)
                        )
                        - target_jmag
                    ),
                )
                char_curve = _char_curve_rescaled_to_cell(
                    _find_component(char_record, "characterization.detection_curve"),
                    donor_spectro=_find_component(
                        char_record, "characterization.spectro"
                    ),
                    cell_spectro=char,
                )
                char = _find_component(char_record, "characterization.spectro")
                char_record_jmag = _record_jmag(char_record)
                if char_record_jmag is not None:
                    char_floor_scaled_from_jmag = char_record_jmag
                char_rate_scale = (
                    10.0 ** (-0.4 * (target_jmag - char_record_jmag))
                    if char_record_jmag is not None
                    else 1.0
                )
        char_planet, char_leakage, char_snr, char_max_r = (
            _characterization_rates_for_planet(
            record=char_record,
            char=char,
            char_curve=char_curve,
            planet_contrast=reflected_contrast,
            planet_separation_mas=float(planet.separation_mas),
            fallback_planet_rate_hz=planet_rate,
            fallback_leakage_rate_hz=leakage_rate,
            stellar_rate_scale=char_rate_scale,
            curve_expected=curve_expected,
        ))
        char_source = char if char is not None else char_curve
        n_spectro_ports = 0 if char_source is None else _spectro_port_count(char_source)
        radius = float(planet.radius_rearth)
        if not np.isfinite(radius) and raw is not None:
            mass = _float_value(raw, "pl_bmasse")
            radius = (
                chen_kipping_2017_radius(mass) if mass and mass > 0 else float("nan")
            )
        row = {
            "name": planet.name,
            "hostname": "" if raw is None else str(raw.get("hostname") or ""),
            "category": planet.planet_type
            if np.isfinite(radius)
            else classify_by_radius(radius),
            "dist_pc": _float_value(raw, "sy_dist"),
            "host_jmag": _float_value(raw, "sy_jmag"),
            "spectral_type": "" if raw is None else str(raw.get("st_spectype") or ""),
            "assumed_star_disk_mas": _assumed_star_disk_mas(collection),
            "radius_rearth": radius,
            "radius_is_estimated": int(bool(planet.radius_is_estimated)),
            "sma_au": _float_value(raw, "pl_orbsmax"),
            "ang_sep_mas": float(planet.separation_mas),
            "reflected_contrast": reflected_contrast,
            "throughput_dark": float(throughput_dark),
            "realized_null": float(realized_null),
            "effective_null": float(effective_null),
            "photon_rate_at_chip": float(stellar_rate),
            "leakage_rate_dark": float(leakage_rate),
            "planet_signal_rate_dark": float(planet_rate),
            "snr_det_4h": float(snr_det),
            "null_instability": _first_float(
                sensitivity, "null_instability", default=0.0
            ),
            "ft_servo_bandwidth_hz": _frame_rate_hz(record),
            "sky_rate_per_dark_port": _sky_rate_per_dark_port(record),
            "dark_rate_per_dark_port": _first_float(
                _find_component(record, "performance.standard"),
                "dark_rate_hz_per_port",
                default=0.0,
            ),
            # Hosts beyond the simulated J range take the edge cell's floor,
            # eps, and bandwidth without extrapolation.
            "host_jmag_outside_sweep": int(
                target_jmag is not None
                and bool(sweep_jmags)
                and not min(sweep_jmags) <= target_jmag <= max(sweep_jmags)
            ),
            "char_floor_scaled_from_jmag": float(char_floor_scaled_from_jmag),
            # The donor had no realized floor: every anchor took the host
            # cell's scene floor.
            "char_floor_from_host_scene": int(
                bool(char_curve is not None and char_curve.get("floor_from_host_scene"))
            ),
            "char_spectro_throughput": _first_float(
                char, "spectro_throughput", default=0.0
            ),
            "char_spectro_null_depth": _first_float(
                char, "spectro_band_on_axis_null_depth", "spectro_null_depth",
                default=0.0,
            ),
            "char_spectro_design_null_depth": _first_float(
                char, "spectro_null_depth", default=0.0
            ),
            "char_dark_null_depth": _first_float(char, "dark_null_depth", default=0.0),
            "char_planet_signal_rate": float(char_planet),
            "char_leakage_rate": float(char_leakage),
            "char_disk_leak": _first_float(
                char, "spectro_band_disk_leak", "spectro_disk_leak", default=0.0
            ),
            "char_design_disk_leak": _first_float(
                char, "spectro_disk_leak", default=0.0
            ),
            "char_spectro_leak_fraction": _first_float(
                char, "spectro_leak_fraction", default=0.0
            ),
            "char_snr_4h": float(char_snr),
            # Spectro-port background for the characterization PDI: sky is the
            # arm total per logical port, dark the per-port rate in the
            # stage's D budget.
            "char_n_spectro_ports": n_spectro_ports,
            "char_sky_rate_per_spectro_port": _first_float(
                char_curve,
                "sky_rate_hz_per_spectro_port",
                default=_first_float(char, "sky_rate_hz_per_spectro_port", default=0.0),
            ),
            "char_dark_rate_per_spectro_port": (
                _char_dark_rate_hz_per_port(
                    char,
                    integration_time_s=_integration_time_s_for_record(char_record),
                    n_spectro_ports=n_spectro_ports,
                )
                if char is not None
                else 0.0
            ),
            # Largest R with per-bin SNR >= 3, computed per planet from its
            # characterization rates (previously this column copied the
            # scene-level max_spectral_resolution, identical for every row).
            "char_max_R": float(char_max_r),
            "orbit_averaged_lambert_phase": float(phi),
        }
        rows.append(row)
    rows.sort(key=lambda item: float(item.get("snr_det_4h") or 0.0), reverse=True)
    return rows


def _survey_config_summary(collection: RunCollection) -> dict[str, Any]:
    return {
        "wavelength_m": float(collection.config.grid.wavelength_center_um) * 1.0e-6,
        "bandwidth_m": float(collection.config.grid.wavelength_bandwidth_um) * 1.0e-6,
        "integration_time_s": _integration_time_s(collection),
        "top_n": 10,
        "run_name": collection.config.run.name,
        "config_hash": collection.config.config_hash,
        "snr_model": "SNR = 5 * planet_contrast / contrast_curve_limit",
    }


def _exoplanet_table_columns() -> list[str]:
    return [
        "name",
        "hostname",
        "category",
        "dist_pc",
        "host_jmag",
        "spectral_type",
        "assumed_star_disk_mas",
        "radius_rearth",
        "radius_is_estimated",
        "sma_au",
        "ang_sep_mas",
        "reflected_contrast",
        "throughput_dark",
        "realized_null",
        "effective_null",
        "photon_rate_at_chip",
        "leakage_rate_dark",
        "planet_signal_rate_dark",
        "snr_det_4h",
        "null_instability",
        "ft_servo_bandwidth_hz",
        "sky_rate_per_dark_port",
        "dark_rate_per_dark_port",
        "host_jmag_outside_sweep",
        "char_floor_scaled_from_jmag",
        "char_floor_from_host_scene",
        "char_spectro_throughput",
        "char_spectro_null_depth",
        "char_spectro_design_null_depth",
        "char_dark_null_depth",
        "char_planet_signal_rate",
        "char_leakage_rate",
        "char_disk_leak",
        "char_design_disk_leak",
        "char_spectro_leak_fraction",
        "char_snr_4h",
        "char_max_R",
        "char_n_spectro_ports",
        "char_sky_rate_per_spectro_port",
        "char_dark_rate_per_spectro_port",
    ]


def _pdi_table_columns() -> list[str]:
    return [
        "name",
        "hostname",
        "category",
        "dist_pc",
        "host_jmag",
        "spectral_type",
        "radius_rearth",
        "radius_is_estimated",
        "sma_au",
        "ang_sep_mas",
        "pol_fraction_orbit_avg",
        "leakage_rate_dark",
        "planet_signal_rate_dark",
        "sky_rate_per_dark_port",
        "dark_rate_per_dark_port",
        "char_planet_signal_rate_pdi",
        "char_leakage_rate_pdi",
        "snr_det_4h",
        "snr_pdi_continuous_4h",
        "snr_char_pdi_continuous_4h",
        "pdi_advantage_4h",
        "t5sigma_pdi_continuous_h",
        "t5sigma_char_pdi_continuous_h",
    ]


def _top_rows_by_category(
    rows: list[dict[str, Any]], *, n: int
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(str(row.get("category") or "Other"), []).append(row)
    for category, category_rows in out.items():
        out[category] = sorted(
            category_rows,
            key=lambda item: float(item.get("snr_det_4h") or 0.0),
            reverse=True,
        )[:n]
    return out


# Display labels for the LaTeX survey tables. The catalog classifier keeps its
# versioned internal names; the paper prose says "Super Earth" / "Ice Giant".
_CATEGORY_DISPLAY_LABELS = {
    "Super-Earth": "Super Earth",
    "Neptune-like": "Ice Giant",
}


def _display_category(category: str) -> str:
    return _CATEGORY_DISPLAY_LABELS.get(category, category)


def _latex_exoplanet_table(top_by_category: Mapping[str, list[dict[str, Any]]]) -> str:
    rows = [
        row
        for category in sorted(top_by_category)
        for row in top_by_category[category][:3]
    ]
    lines = [
        r"\begin{table*}",
        r"\centering",
        r"\caption{Detection mode ranking of confirmed exoplanets within "
        r"50\,pc visible from Maunakea. Columns give the planet class, host "
        r"$J$ magnitude, angular separation, assumed reflected light "
        r"contrast ($A_g = 0.3$), and the predicted KILO detection SNR in a "
        r"4\,h integration.}",
        r"\label{tab:exoplanet_survey}",
        r"\begin{tabular}{llrrrr}",
        r"\hline",
        r"Planet & Class & $J$ & Sep. (mas) & Contrast & SNR$_{4h}$ \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            f"{_tex_escape(row['name'])} & "
            f"{_tex_escape(_display_category(row['category']))} & "
            f"{float(row.get('host_jmag') or 0):.2f} & "
            f"{float(row.get('ang_sep_mas') or 0):.1f} & "
            f"{float(row.get('reflected_contrast') or 0):.1e} & "
            f"{float(row.get('snr_det_4h') or 0):.2f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def _top_rows_by_char_snr(
    rows: list[dict[str, Any]],
    n: int,
) -> dict[str, list[dict[str, Any]]]:
    """Top-n rows per category ranked by finite characterization SNR."""

    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        snr = float(row.get("char_snr_4h") or 0.0)
        if not np.isfinite(snr) or snr <= 0.0:
            continue
        out.setdefault(str(row.get("category") or "Unknown"), []).append(row)
    return {
        category: sorted(
            category_rows,
            key=lambda item: float(item.get("char_snr_4h") or 0.0),
            reverse=True,
        )[:n]
        for category, category_rows in out.items()
    }


def _format_char_max_r(value: float) -> str:
    if not np.isfinite(value) or value <= 0.0:
        return "--"
    if value >= 9999.5:
        return r"$\geq 10^4$"
    return f"{value:.0f}"


def _latex_characterization_table(
    top_by_category: Mapping[str, list[dict[str, Any]]],
) -> str:
    rows = [
        row
        for category in sorted(top_by_category)
        for row in top_by_category[category]
    ]
    notes = ""
    if any(_float_value(row, "char_floor_scaled_from_jmag") is not None for row in rows):
        notes += (
            r" Hosts whose nearest simulated $J$ cell has no characterization "
            r"curve use the chip throughput and static leak of the nearest "
            r"cell that has one, with its realized leakage floor scaled by "
            r"the ratio of the two cells' retuned spectro floors, and with "
            r"the systematic $\epsilon$ and bandwidth of the host's own "
            r"retuned spectro port."
        )
    if any(row.get("host_jmag_outside_sweep") for row in rows):
        notes += (
            r" Hosts outside the simulated $J$ range use the nearest cell "
            r"without extrapolation."
        )
    lines = [
        r"\begin{table*}",
        r"\centering",
        r"\caption{Characterization mode ranking of confirmed exoplanets "
        r"within 50\,pc visible from Maunakea: the ten highest "
        r"characterization SNR planets of each class. SNR$_{\rm char}$ is "
        r"the integrated one spectro port characterization SNR in a 4\,h "
        r"integration; $R_{\rm SNR \geq 3}$ is the largest spectral "
        r"resolving power at which each Nyquist sampled bin retains "
        r"SNR $\geq 3$, capped at the $R = 10^4$ AWG design limit." + notes + "}",
        r"\label{tab:characterization_survey}",
        r"\begin{tabular}{llrrrrr}",
        r"\hline",
        r"Planet & Class & $J$ & Sep. (mas) & Contrast & "
        r"SNR$_{\rm char}$ & $R_{\rm SNR \geq 3}$ \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            f"{_tex_escape(row['name'])} & "
            f"{_tex_escape(_display_category(row['category']))} & "
            f"{float(row.get('host_jmag') or 0):.2f} & "
            f"{float(row.get('ang_sep_mas') or 0):.1f} & "
            f"{float(row.get('reflected_contrast') or 0):.1e} & "
            f"{float(row.get('char_snr_4h') or 0):.1f} & "
            f"{_format_char_max_r(float(row.get('char_max_R') or 0))} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def _latex_pdi_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table*}",
        r"\centering",
        r"\caption{Polarimetric differential imaging survey of catalogued "
        r"cloudy giants ($R_p \geq 1.5\,R_\oplus$, orbit averaged "
        r"$|P| \approx 0.08$). Columns give the angular separation, "
        r"polarization fraction, detection mode SNR, detection PDI SNR at "
        r"4\,h, and the integration time at which the PDI channel crosses "
        r"$5\sigma$.}",
        r"\label{tab:pdi_survey}",
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        r"Planet & Sep. (mas) & $|P|$ & SNR$_{\rm det}$ & SNR$_{\rm PDI}$ & t$_{5\sigma}$ (h) \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            f"{_tex_escape(row['name'])} & "
            f"{float(row.get('ang_sep_mas') or 0):.1f} & "
            f"{float(row.get('pol_fraction_orbit_avg') or 0):.2f} & "
            f"{float(row.get('snr_det_4h') or 0):.2f} & "
            f"{float(row.get('snr_pdi_continuous_4h') or 0):.2f} & "
            f"{float(row.get('t5sigma_pdi_continuous_h') or 0):.2f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def _pdi_row(row: dict[str, Any], collection: RunCollection) -> dict[str, Any] | None:
    radius = float(row.get("radius_rearth") or 0.0)
    if radius <= 0.0:
        return None
    pol = _orbit_averaged_polarization_fraction(radius)
    tint = _integration_time_s(collection)
    scale = _scalar_to_dual_pol_rate_scale(collection)
    planet_raw = float(row.get("planet_signal_rate_dark") or 0.0)
    leakage_raw = float(row.get("leakage_rate_dark") or 0.0)
    # ``is None`` checks: a legitimate characterization rate of 0.0 must
    # not silently fall back to the detection-mode rate (review 2026-06-10
    # nit). The fallback takes the unscaled rate so the factor applies once.
    char_planet_raw = row.get("char_planet_signal_rate")
    char_leakage_raw = row.get("char_leakage_rate")
    planet = planet_raw * scale
    leakage = leakage_raw * scale
    char_planet = scale * float(
        planet_raw if char_planet_raw is None else char_planet_raw
    )
    char_leakage = scale * float(
        leakage_raw if char_leakage_raw is None else char_leakage_raw
    )
    sky = float(row.get("sky_rate_per_dark_port") or 0.0)
    dark = float(row.get("dark_rate_per_dark_port") or 0.0)
    n_dark = _n_dark_ports_per_arm(collection)
    snr_pdi = _signal_to_noise_pdi_rates(
        planet, leakage, tint, pol, n_dark, sky_rate_hz=sky, dark_rate_hz=dark
    )
    # Characterization rates are spectro port rates: charge the spectro
    # ports' background. The primitive sums per-arm rates over 2 arms, so the
    # arm-total sky and the per-port dark are split evenly across the arms.
    n_spectro = int(row.get("char_n_spectro_ports") or 0)
    if n_spectro > 0:
        snr_char = _signal_to_noise_pdi_rates(
            char_planet,
            char_leakage,
            tint,
            pol,
            n_spectro,
            sky_rate_hz=0.5 * float(row.get("char_sky_rate_per_spectro_port") or 0.0),
            dark_rate_hz=0.5 * float(row.get("char_dark_rate_per_spectro_port") or 0.0),
        )
    else:
        snr_char = _signal_to_noise_pdi_rates(
            char_planet, char_leakage, tint, pol, n_dark, sky_rate_hz=sky, dark_rate_hz=dark
        )
    det = float(row.get("snr_det_4h") or 0.0)
    return {
        "name": row.get("name"),
        "hostname": row.get("hostname"),
        "category": row.get("category"),
        "dist_pc": row.get("dist_pc"),
        "host_jmag": row.get("host_jmag"),
        "spectral_type": row.get("spectral_type"),
        "radius_rearth": radius,
        "radius_is_estimated": row.get("radius_is_estimated"),
        "sma_au": row.get("sma_au"),
        "ang_sep_mas": row.get("ang_sep_mas"),
        "pol_fraction_orbit_avg": float(pol),
        "leakage_rate_dark": leakage_raw,
        "planet_signal_rate_dark": planet_raw,
        "sky_rate_per_dark_port": sky,
        "dark_rate_per_dark_port": dark,
        "char_planet_signal_rate_pdi": float(char_planet),
        "char_leakage_rate_pdi": float(char_leakage),
        "snr_det_4h": float(det),
        "snr_pdi_continuous_4h": float(snr_pdi),
        "snr_char_pdi_continuous_4h": float(snr_char),
        "pdi_advantage_4h": float(snr_pdi / det) if det > 0 else float("inf"),
        "t5sigma_pdi_continuous_h": _time_to_snr5_h(snr_pdi, tint),
        "t5sigma_char_pdi_continuous_h": _time_to_snr5_h(snr_char, tint),
    }


def _pdi_displayed_snr(row: Mapping[str, Any]) -> float:
    """Sort key for the PDI survey table: the displayed detection-mode PDI SNR.

    The published table displays ``snr_pdi_continuous_4h``; sorting by it keeps
    the top-of-table LaTeX/JSON selections consistent with what is shown.
    Non-finite or missing values are mapped to ``-inf`` so they sort last and
    never displace finite rows from the top.
    """

    try:
        value = float(row["snr_pdi_continuous_4h"])
    except (TypeError, ValueError, KeyError):
        return float("-inf")
    return value if np.isfinite(value) else float("-inf")


def _scalar_to_dual_pol_rate_scale(collection: RunCollection) -> float:
    """Factor converting the survey's detector rates to dual polarization.

    ``_DUAL_POL_THROUGHPUT_VS_SCALAR`` stands in for the HWP, PBS, and PM
    fiber losses of a scalar pipeline. A pipeline with polarization stages
    already carries those losses in its detector rates, so the factor is 1.
    """

    has_polarization_stages = any(
        stage.type.startswith("polarization.")
        for stage in collection.config.pipeline.stages
    )
    return 1.0 if has_polarization_stages else _DUAL_POL_THROUGHPUT_VS_SCALAR


def _signal_to_noise_pdi_rates(
    planet_rate_hz: float,
    leakage_rate_hz: float,
    integration_time_s: float,
    pol_fraction: float,
    n_dark_ports: int,
    *,
    sky_rate_hz: float = 0.0,
    dark_rate_hz: float = 0.0,
) -> float:
    from nullsim.physics.pdi import signal_to_noise_pdi

    return float(
        signal_to_noise_pdi(
            planet_rate_total_hz=planet_rate_hz,
            stellar_leakage_rate_total_hz=leakage_rate_hz,
            integration_time_s=integration_time_s,
            pol_fraction=pol_fraction,
            n_dark_ports_per_arm=n_dark_ports,
            pbs_axis_offset_rad=1.0e-3,
            hwp_retardance_error_rad=0.02,
            hwp_rotation_jitter_rad=1.0e-4,
            pbs_extinction_db=30.0,
            modulation_period_s=1.0 / _PDI_HWP_ROTATION_RATE_HZ,
            cmrr_db=30.0,
            recover_both_qu=True,
            dark_rate_hz_per_port_per_arm=dark_rate_hz,
            sky_rate_hz_per_dark_port_per_arm=sky_rate_hz,
        )
    )


def _pdi_snr(
    row: Mapping[str, Any],
    integration_time_s: float,
    *,
    modulation_period_s: float,
    n_dark_ports: int,
    rate_scale: float,
    ideal: bool = False,
) -> float:
    from nullsim.physics.pdi import signal_to_noise_pdi

    return float(
        signal_to_noise_pdi(
            planet_rate_total_hz=float(row["planet_signal_rate_dark"]) * rate_scale,
            stellar_leakage_rate_total_hz=float(row["leakage_rate_dark"]) * rate_scale,
            dark_rate_hz_per_port_per_arm=float(row.get("dark_rate_per_dark_port") or 0.0),
            sky_rate_hz_per_dark_port_per_arm=float(
                row.get("sky_rate_per_dark_port") or 0.0
            ),
            integration_time_s=integration_time_s,
            pol_fraction=float(row["pol_fraction_orbit_avg"]),
            n_dark_ports_per_arm=n_dark_ports,
            pbs_axis_offset_rad=0.0 if ideal else 1.0e-3,
            hwp_retardance_error_rad=0.0 if ideal else 0.02,
            hwp_rotation_jitter_rad=0.0 if ideal else 1.0e-4,
            pbs_extinction_db=float("inf") if ideal else 30.0,
            modulation_period_s=modulation_period_s,
            cmrr_db=30.0,
            recover_both_qu=True,
        )
    )


def _named_or_top_rows(
    rows: list[dict[str, Any]],
    names: tuple[str, ...],
    *,
    n: int,
) -> list[dict[str, Any]]:
    selected = []
    for name in names:
        match = next((row for row in rows if str(row.get("name")) == name), None)
        if match is not None:
            selected.append(match)
    if len(selected) >= n:
        return selected[:n]
    for row in rows:
        if row not in selected:
            selected.append(row)
        if len(selected) >= n:
            break
    return selected


def _scale_snr_time(
    snr_reference: float, integration_time_s: float, collection: RunCollection
) -> float:
    reference = _integration_time_s(collection)
    if reference <= 0 or integration_time_s <= 0 or snr_reference <= 0:
        return 0.0
    return float(snr_reference) * float(np.sqrt(integration_time_s / reference))


def _time_to_snr5_h(snr: float, integration_time_s: float) -> float:
    if snr <= 0.0:
        return float("inf")
    return float((5.0 / snr) ** 2 * integration_time_s / 3600.0)


def _integration_time_s(collection: RunCollection) -> float:
    return float(
        getattr(collection.config.observation, "integration_time_s", 4.0 * 3600.0)
    )


def _n_dark_ports_per_arm(collection: RunCollection) -> int:
    record = _first_record(collection)
    sensitivity = (
        _find_component(record, "sensitivity.detection_curve") if record else None
    )
    n_dark = _int_from_payload(sensitivity, "n_dark_ports")
    return int(n_dark or 18)


def _record_for_jmag(
    collection: RunCollection, target_jmag: float | None
) -> RunRecord | None:
    if not collection.records:
        return None
    if target_jmag is None:
        return collection.records[0]
    return min(
        collection.records,
        # ``is None`` rather than falsy: a legitimate jmag of 0.0 must not
        # collapse to distance 0 from every target (review 2026-06-10 nit).
        key=lambda record: abs(
            (
                target_jmag
                if _record_jmag(record) is None
                else _record_jmag(record)
            )
            - target_jmag
        ),
    )


def _record_jmag(record: RunRecord) -> float | None:
    for axis_name, axis_value in record.coord_tuple:
        if axis_name == "scene.star.jmag":
            try:
                return float(axis_value)
            except (TypeError, ValueError):
                return None
    try:
        return float(record.config.scene.star.jmag)
    except (AttributeError, TypeError, ValueError):
        return None


def _contrast_limit_at(record: RunRecord | None, separation_mas: float) -> float | None:
    if record is None:
        return None
    curve = record.result.state.results.contrast_curve
    if curve is None or getattr(curve, "size", 0) == 0:
        return None
    sep = np.asarray(curve["separation_mas"], dtype=np.float64)
    contrast = np.asarray(curve["contrast"], dtype=np.float64)
    mask = np.isfinite(sep) & np.isfinite(contrast) & (sep > 0) & (contrast > 0)
    if not mask.any():
        return None
    return _interp_gapped(sep, contrast, mask, separation_mas)


def _interp_gapped(
    xs: np.ndarray,
    ys: np.ndarray,
    valid: np.ndarray,
    x: float,
) -> float:
    """Linear interpolation that treats invalid samples as gaps.

    Returns NaN outside the sampled range and between two samples when either
    is invalid, so a failed anchor or an infinite contrast is never bridged
    and a lone valid sample counts only at its own abscissa.
    """

    keep = np.isfinite(xs)
    order = np.argsort(xs[keep])
    xs, ys, valid = xs[keep][order], ys[keep][order], valid[keep][order]
    x = float(x)
    if xs.size == 0 or not xs[0] <= x <= xs[-1]:
        return float("nan")
    hit = xs == x
    if hit.any():
        index = np.flatnonzero(hit & valid)
        return float(ys[index[0]]) if index.size else float("nan")
    upper = int(np.searchsorted(xs, x))
    if not (valid[upper - 1] and valid[upper]):
        return float("nan")
    return float(np.interp(x, xs[upper - 1 : upper + 1], ys[upper - 1 : upper + 1]))


def _interp_payload(
    payload: Mapping[str, Any] | None,
    x_key: str,
    y_key: str,
    x: float,
    *,
    default: float,
) -> float:
    if payload is None or payload.get(x_key) is None or payload.get(y_key) is None:
        return float(default)
    xs = np.asarray(payload[x_key], dtype=np.float64)
    ys = np.asarray(payload[y_key], dtype=np.float64)
    if xs.ndim != 1 or ys.ndim != 1 or xs.size != ys.size or xs.size == 0:
        return float(default)
    return _interp_gapped(xs, ys, np.isfinite(ys), x)


def _bright_rate_hz(record: RunRecord | None) -> float:
    if record is None:
        return 1.0e8
    for payload in record.result.state.results.components.values():
        if not isinstance(payload, Mapping):
            continue
        if payload.get("port_rates") is None or payload.get("port_labels") is None:
            continue
        rates = np.asarray(payload["port_rates"], dtype=np.float64)
        labels = tuple(str(label) for label in payload["port_labels"])
        if rates.shape != (len(labels),):
            continue
        mask = np.array([label.startswith("bright") for label in labels], dtype=bool)
        if mask.any():
            return float(np.sum(rates[mask]))
    return 1.0e8


def _stellar_rate_hz(record: RunRecord | None) -> float:
    if record is None:
        return 1.0e8
    performance = _find_component(record, "performance.standard")
    rate = _first_float(performance, "stellar_rate_hz_total", default=float("nan"))
    if np.isfinite(rate) and rate > 0.0:
        return float(rate)
    for payload in record.result.state.results.components.values():
        if not isinstance(payload, Mapping):
            continue
        if payload.get("port_rates") is None or payload.get("port_labels") is None:
            continue
        rates = np.asarray(payload["port_rates"], dtype=np.float64)
        labels = tuple(str(label) for label in payload["port_labels"])
        if rates.shape != (len(labels),):
            continue
        mask = np.array(
            [
                label.startswith(("bright", "dark", "spectro"))
                for label in labels
            ],
            dtype=bool,
        )
        if mask.any():
            return float(np.sum(rates[mask]))
    return _bright_rate_hz(record)


def _char_max_resolving_power(
    record: RunRecord | None,
    *,
    planet_rate_hz: float,
    leakage_rate_hz: float,
    dark_rate_hz_per_port: float,
    sky_rate_hz_per_port: float,
    n_spectro_ports: int,
    integration_time_s: float,
    eps_systematic: float,
    servo_bw_hz: float,
    char: Mapping[str, Any] | None,
) -> float:
    """Largest resolving power with per-bin SNR >= ``snr_min_per_bin``.

    Mirrors ``characterization.spectro``'s own max-R call, read back from its
    payload: the rates are pre-spectrograph, so the AWG optical throughput
    applies in AWG mode; sky is summed over the spectro ports; the detector
    pixels per port and the pixels per resolution element (1 for MKID) are
    the stage's.
    """

    from nullsim.physics.spectroscopy import max_spectral_resolution

    if not (np.isfinite(planet_rate_hz) and planet_rate_hz > 0.0):
        return float("nan")
    band_lambda_m = 0.0
    band_bandwidth_m = 0.0
    if record is not None:
        grid = getattr(record.config, "grid", None)
        if grid is not None:
            band_lambda_m = float(grid.wavelength_center_um) * 1.0e-6
            band_bandwidth_m = float(grid.wavelength_bandwidth_um) * 1.0e-6
    if band_lambda_m <= 0.0 or band_bandwidth_m <= 0.0:
        return float("nan")
    n_ports = max(int(n_spectro_ports), 1)
    spectrograph = None if char is None else char.get("spectrograph")
    if spectrograph == "awg":
        awg = char.get("awg") or {}
        n_spectro_pixels = int(awg.get("n_detector_pixels_per_bin", n_ports)) // n_ports
        pixels_per_resolution_element = int(awg.get("pixels_per_resolution_element", 2))
    elif spectrograph == "mkid":
        n_spectro_pixels = int(char.get("mkid_n_spectro_pixels", 1))
        pixels_per_resolution_element = 1
    else:
        n_spectro_pixels, pixels_per_resolution_element = 1, 2
    return max_spectral_resolution(
        planet_rate_total_hz=float(planet_rate_hz),
        leakage_rate_total_hz=max(float(leakage_rate_hz), 0.0),
        spectro_throughput=_awg_throughput(char),
        band_lambda_m=band_lambda_m,
        band_bandwidth_m=band_bandwidth_m,
        integration_time_s=float(integration_time_s),
        dark_rate_hz_per_pixel=max(float(dark_rate_hz_per_port), 0.0),
        n_spectro_pixels=max(n_spectro_pixels, 1),
        null_instability=max(float(eps_systematic), 0.0),
        servo_bandwidth_hz=max(float(servo_bw_hz), 0.0),
        snr_min=_first_float(char, "snr_min_per_bin", default=3.0),
        resolving_power_cap=_first_float(char, "resolving_power_cap", default=1.0e4),
        pixels_per_resolution_element=pixels_per_resolution_element,
        sky_rate_hz_per_spectro_port=max(float(sky_rate_hz_per_port), 0.0) * n_ports,
    )


def _awg_throughput(char: Mapping[str, Any] | None) -> float:
    """AWG optical throughput of a ``characterization.spectro`` payload, else 1."""

    if char is None or char.get("spectrograph") != "awg":
        return 1.0
    return _first_float(char.get("awg"), "throughput", default=1.0)


def _characterization_rates_for_planet(
    *,
    record: RunRecord | None,
    char: Mapping[str, Any] | None,
    char_curve: Mapping[str, Any] | None = None,
    planet_contrast: float,
    planet_separation_mas: float | None = None,
    fallback_planet_rate_hz: float,
    fallback_leakage_rate_hz: float,
    stellar_rate_scale: float = 1.0,
    curve_expected: bool = False,
) -> tuple[float, float, float, float]:
    """Return (planet_rate_hz, leakage_rate_hz, char_snr, char_max_R)."""

    curve_estimate = _characterization_curve_estimate_for_planet(
        record=record,
        char=char,
        char_curve=char_curve,
        planet_contrast=planet_contrast,
        planet_separation_mas=planet_separation_mas,
        stellar_rate_scale=stellar_rate_scale,
    )
    if curve_estimate is not None:
        return curve_estimate

    if curve_expected and char_curve is None:
        # A separation-resolved characterization curve is produced for other
        # cells in this run but was skipped for this one (skip_jmags). The
        # single-scene scalar retune is not valid at arbitrary planet
        # separations, so keep the detection-mode rates but report NaN for the
        # per-planet characterization SNR.
        return (
            float(fallback_planet_rate_hz),
            float(fallback_leakage_rate_hz),
            float("nan"),
            float("nan"),
        )

    if char is None:
        return (
            float(fallback_planet_rate_hz),
            float(fallback_leakage_rate_hz),
            0.0,
            float("nan"),
        )
    if not _characterization_component_converged(char):
        return float("nan"), float("nan"), float("nan"), float("nan")

    stellar_rate_hz = float(stellar_rate_scale) * _first_float(
        char,
        "detected_stellar_rate_hz",
        default=_stellar_rate_hz(record),
    )
    throughput = _first_float(char, "spectro_throughput", default=0.0)
    planet_to_star_ratio = _spectro_design_planet_to_star_ratio(char)
    leak_fraction = _first_float(
        char,
        "spectro_band_leak_fraction",
        "spectro_leak_fraction",
        default=float("nan"),
    )
    if not np.isfinite(leak_fraction):
        leak_fraction = (
            float(fallback_leakage_rate_hz) / stellar_rate_hz
            if stellar_rate_hz > 0.0
            else 0.0
        )
    planet_rate_hz = (
        stellar_rate_hz
        * float(planet_contrast)
        * max(throughput, 0.0)
        * max(planet_to_star_ratio, 0.0)
    )
    leakage_rate_hz = stellar_rate_hz * max(float(leak_fraction), 0.0)

    integration_time_s = _integration_time_s_for_record(record)
    n_spectro_ports = _spectro_port_count(char)
    dark_rate_hz_per_port = _char_dark_rate_hz_per_port(
        char,
        integration_time_s=integration_time_s,
        n_spectro_ports=n_spectro_ports,
    )
    sky_rate_hz_per_port = _first_float(
        char,
        "sky_rate_hz_per_spectro_port",
        default=0.0,
    )
    null_instability = _first_float(char, "null_instability", default=0.0)
    eps_systematic = _first_float(char, "eps_systematic", default=null_instability)
    servo_bw_hz = _first_float(char, "servo_bw_hz", default=_frame_rate_hz(record))
    with_fringe_tracking = servo_bw_hz > 0.0

    from nullsim.stages.performance.standard import _compute_snr

    snr = _compute_snr(
        planet_rate_hz=planet_rate_hz,
        leakage_rate_hz=leakage_rate_hz,
        dark_rate_hz_per_port=dark_rate_hz_per_port,
        sky_rate_hz_per_port=sky_rate_hz_per_port,
        n_dark_ports=n_spectro_ports,
        integration_time_s=integration_time_s,
        eps=eps_systematic,
        with_fringe_tracking=with_fringe_tracking,
        servo_bw_hz=servo_bw_hz if servo_bw_hz > 0.0 else 1.0,
    )["snr"]
    max_resolving_power = _char_max_resolving_power(
        record,
        planet_rate_hz=planet_rate_hz,
        leakage_rate_hz=leakage_rate_hz,
        dark_rate_hz_per_port=dark_rate_hz_per_port,
        sky_rate_hz_per_port=sky_rate_hz_per_port,
        n_spectro_ports=n_spectro_ports,
        integration_time_s=integration_time_s,
        eps_systematic=eps_systematic,
        servo_bw_hz=servo_bw_hz,
        char=char,
    )
    return (
        float(planet_rate_hz),
        float(leakage_rate_hz),
        float(snr),
        float(max_resolving_power),
    )


def _characterization_curve_estimate_for_planet(
    *,
    record: RunRecord | None,
    char: Mapping[str, Any] | None,
    char_curve: Mapping[str, Any] | None,
    planet_contrast: float,
    planet_separation_mas: float | None,
    stellar_rate_scale: float = 1.0,
) -> tuple[float, float, float, float] | None:
    if char_curve is None or planet_separation_mas is None:
        return None
    estimate = _characterization_curve_throughput_and_leak(
        char_curve,
        float(planet_separation_mas),
    )
    if estimate is None:
        return None
    throughput, leak_fraction, is_feasible = estimate
    if not is_feasible:
        return float("nan"), float("nan"), float("nan"), float("nan")

    stellar_rate_hz = float(stellar_rate_scale) * _first_float(
        char_curve,
        "detected_stellar_rate_hz",
        default=_first_float(char, "detected_stellar_rate_hz", default=_stellar_rate_hz(record)),
    )
    planet_rate_hz = (
        stellar_rate_hz * float(planet_contrast) * max(float(throughput), 0.0)
    )
    leakage_rate_hz = stellar_rate_hz * max(float(leak_fraction), 0.0)

    integration_time_s = _integration_time_s_for_record(record)
    n_spectro_ports = _spectro_port_count(char or char_curve)
    dark_rate_hz_per_port = (
        _char_dark_rate_hz_per_port(
            char,
            integration_time_s=integration_time_s,
            n_spectro_ports=n_spectro_ports,
        )
        if char is not None
        else 0.0
    )
    sky_rate_hz_per_port = _first_float(
        char_curve,
        "sky_rate_hz_per_spectro_port",
        default=_first_float(char, "sky_rate_hz_per_spectro_port", default=0.0),
    )
    null_instability = _first_float(
        char_curve,
        "null_instability",
        default=_first_float(char, "null_instability", default=0.0),
    )
    eps_systematic = _first_float(
        char_curve,
        "eps_systematic",
        default=_first_float(char, "eps_systematic", default=null_instability),
    )
    # The curve publishes each anchor's retuned-port leak CV; use the one at
    # the planet's separation when it is defined there.
    eps_per_anchor = _payload_vector(char_curve, "eps_systematic_per_anchor")
    seps = _payload_vector(char_curve, "separations_mas")
    if eps_per_anchor is not None and seps is not None and seps.shape == eps_per_anchor.shape:
        eps_at = _interp_gapped(
            seps, eps_per_anchor, np.isfinite(eps_per_anchor), planet_separation_mas
        )
        if np.isfinite(eps_at):
            eps_systematic = eps_at
    servo_bw_hz = _first_float(
        char_curve,
        "servo_bw_hz",
        "effective_systematic_bandwidth_hz",
        default=_first_float(char, "servo_bw_hz", default=_frame_rate_hz(record)),
    )
    with_fringe_tracking = servo_bw_hz > 0.0

    from nullsim.stages.performance.standard import _compute_snr

    snr = _compute_snr(
        planet_rate_hz=planet_rate_hz,
        leakage_rate_hz=leakage_rate_hz,
        dark_rate_hz_per_port=dark_rate_hz_per_port,
        sky_rate_hz_per_port=sky_rate_hz_per_port,
        n_dark_ports=n_spectro_ports,
        integration_time_s=integration_time_s,
        eps=eps_systematic,
        with_fringe_tracking=with_fringe_tracking,
        servo_bw_hz=servo_bw_hz if servo_bw_hz > 0.0 else 1.0,
    )["snr"]
    max_resolving_power = _char_max_resolving_power(
        record,
        planet_rate_hz=planet_rate_hz,
        leakage_rate_hz=leakage_rate_hz,
        dark_rate_hz_per_port=dark_rate_hz_per_port,
        sky_rate_hz_per_port=sky_rate_hz_per_port,
        n_spectro_ports=n_spectro_ports,
        integration_time_s=integration_time_s,
        eps_systematic=eps_systematic,
        servo_bw_hz=servo_bw_hz,
        char=char,
    )
    return (
        float(planet_rate_hz),
        float(leakage_rate_hz),
        float(snr),
        float(max_resolving_power),
    )


def _char_curve_rescaled_to_cell(
    donor_curve: Mapping[str, Any],
    *,
    donor_spectro: Mapping[str, Any] | None,
    cell_spectro: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Carry a characterization curve from a donor J cell to the host's cell.

    The retuned chip's throughput and static leak transfer between cells. The
    realized floor, the leakage fluctuation, and the systematic bandwidth do
    not: they follow the fringe tracker's photon budget and the setpoint
    trim. Each cell's ``characterization.spectro`` retunes the chip at its
    own J, so:

    - floor: each anchor's floor is scaled by the ratio of the cells' spectro
      ``realized_spectro_floor``. Both zero gives ratio 1; a zero donor floor
      under a nonzero host floor puts the host's scene floor at every anchor
      and sets ``floor_from_host_scene``.
    - eps: a CV relative to each anchor's own leakage, so the fluctuation
      amplitude is what moves. The stage forms ``eps = hypot(ni / nsc, s)``
      with ``ni = leak_rms / leak``; each anchor's ``leak_rms`` (from
      ``null_instability_per_anchor`` times its leakage) is scaled by the
      ratio of the cells' spectro ``spectro_leak_rms_fraction``, the speckle
      CV ``s`` is kept (its amplitude follows the new leakage), and eps is
      recomposed over the new leakage. Without those keys, the host spectro's
      scalar eps replaces the curve's.
    - bandwidth: the host cell's spectro ``servo_bw_hz`` (0 without fringe
      tracking) and ``with_fringe_tracking``.
    """

    curve = dict(donor_curve)
    if donor_spectro is None or cell_spectro is None:
        return curve
    for key in ("servo_bw_hz", "with_fringe_tracking"):
        if cell_spectro.get(key) is not None:
            curve[key] = cell_spectro[key]

    old_leak = _payload_vector(curve, "spectro_leak_fraction_per_anchor")
    new_leak = old_leak
    floor = _payload_vector(curve, "realized_spectro_floor_per_anchor")
    static = _payload_vector(curve, "spectro_static_leak_fraction_per_anchor")
    donor_floor = _first_float(donor_spectro, "realized_spectro_floor")
    cell_floor = _first_float(cell_spectro, "realized_spectro_floor")
    if floor is not None and static is not None and np.isfinite(donor_floor + cell_floor):
        if donor_floor > 0.0:
            new_floor = floor * (cell_floor / donor_floor)
        elif cell_floor == 0.0:
            new_floor = floor
        else:
            new_floor = np.full_like(floor, cell_floor)
            curve["floor_from_host_scene"] = True
        new_leak = static + new_floor
        curve["realized_spectro_floor_per_anchor"] = new_floor
        curve["spectro_leak_fraction_per_anchor"] = new_leak
        curve["realized_spectro_floor"] = float(np.nanmedian(new_floor))

    ni = _payload_vector(curve, "null_instability_per_anchor")
    eps = _payload_vector(curve, "eps_systematic_per_anchor")
    donor_rms = _first_float(donor_spectro, "spectro_leak_rms_fraction")
    cell_rms = _first_float(cell_spectro, "spectro_leak_rms_fraction")
    rms_ratio = (
        cell_rms / donor_rms
        if donor_rms > 0.0
        else (1.0 if cell_rms == 0.0 else float("nan"))
    )
    if (
        old_leak is not None
        and ni is not None
        and eps is not None
        and ni.shape == eps.shape == old_leak.shape
        and np.isfinite(rms_ratio)
    ):
        nsc = _first_float(curve, "nsc_systematic_reduction", default=1.0)
        speckle_cv = np.sqrt(np.maximum(eps**2 - (ni / nsc) ** 2, 0.0))
        leak_rms = ni * old_leak * rms_ratio
        with np.errstate(divide="ignore", invalid="ignore"):
            new_ni = np.where(new_leak > 0.0, leak_rms / new_leak, 0.0)
        new_eps = np.hypot(new_ni / nsc, speckle_cv)
        curve["null_instability_per_anchor"] = new_ni
        curve["eps_systematic_per_anchor"] = new_eps
        curve["eps_systematic"] = float(np.nanmedian(new_eps))
        return curve

    if cell_spectro.get("eps_systematic") is not None:
        curve["eps_systematic"] = cell_spectro["eps_systematic"]
    curve.pop("eps_systematic_per_anchor", None)
    return curve


def _characterization_curve_throughput_and_leak(
    payload: Mapping[str, Any],
    separation_mas: float,
) -> tuple[float, float, bool] | None:
    seps = _payload_vector(payload, "separations_mas")
    throughput = _payload_vector(payload, "spectro_throughput_per_anchor")
    leak = _payload_vector(payload, "spectro_leak_fraction_per_anchor")
    if seps is None or throughput is None or leak is None:
        return None
    if seps.shape != throughput.shape or seps.shape != leak.shape or seps.size == 0:
        return None

    mask = (
        np.isfinite(seps)
        & np.isfinite(throughput)
        & np.isfinite(leak)
        & (throughput >= 0.0)
        & (leak >= 0.0)
    )
    mask &= _characterization_curve_feasible_mask(payload, seps.shape)
    if not np.any(mask):
        return float("nan"), float("nan"), False

    throughput_at = _interp_gapped(seps, throughput, mask, separation_mas)
    leak_at = _interp_gapped(seps, leak, mask, separation_mas)
    if not (np.isfinite(throughput_at) and np.isfinite(leak_at)):
        return float("nan"), float("nan"), False
    return throughput_at, leak_at, True


def _characterization_curve_feasible_mask(
    payload: Mapping[str, Any],
    shape: tuple[int, ...],
) -> np.ndarray:
    mask = np.ones(shape, dtype=bool)
    success = payload.get("optimization_success_per_anchor")
    if success is not None:
        success_mask = np.asarray(success, dtype=bool)
        if success_mask.shape == shape:
            mask &= success_mask

    if bool(payload.get("enforce_null_constraint", False)):
        tolerance = _characterization_feasibility_tolerance(payload)
        for key in (
            "optimization_constraint_margin_per_anchor",
            "full_band_constraint_margin_per_anchor",
        ):
            margins = _payload_vector(payload, key)
            if margins is not None and margins.shape == shape:
                mask &= margins >= -tolerance
    return mask


def _characterization_component_converged(payload: Mapping[str, Any]) -> bool:
    success = payload.get("optimization_success")
    if success is not None and not bool(success):
        return False
    if not bool(payload.get("enforce_null_constraint", False)):
        return True
    margin = _first_float(
        payload,
        "optimization_constraint_margin",
        default=float("nan"),
    )
    if not np.isfinite(margin):
        return True
    return bool(margin >= -_characterization_feasibility_tolerance(payload))


def _characterization_feasibility_tolerance(payload: Mapping[str, Any]) -> float:
    """Margin slack below which an anchor counts as infeasible.

    The optimizer's hard constraint holds the leak at ``null_floor`` on its
    own (coarse) disk quadrature; the published full-band margin re-integrates
    on a finer quadrature and can land a factor of ~1.5 above the floor while
    the deployed leakage still sits below the realized piston/scintillation
    floor, where extra static suppression buys nothing on sky. Gate
    feasibility on the realized floor when it is published: static leak
    between ``null_floor`` and the realized floor is deployable, not failed.
    """

    null_floor = _first_float(payload, "null_floor", default=0.0)
    tolerance = 0.01 * null_floor if null_floor > 0.0 else 0.0
    realized_floor = _first_float(
        payload,
        "realized_spectro_floor",
        default=float("nan"),
    )
    if np.isfinite(realized_floor) and realized_floor > null_floor > 0.0:
        tolerance = max(tolerance, realized_floor - null_floor)
    return tolerance


def _spectro_design_planet_to_star_ratio(payload: Mapping[str, Any]) -> float:
    values = _payload_vector(payload, "spectro_planet_to_star_ratio_per_wavelength")
    if values is None or values.size == 0:
        return 1.0
    index = _payload_design_wavelength_index(payload, values.size)
    value = float(values[index])
    return value if np.isfinite(value) else 1.0


def _payload_design_wavelength_index(
    payload: Mapping[str, Any],
    size: int,
) -> int:
    wavelengths = _payload_vector(payload, "wavelengths_m")
    band_lambda = _first_float(payload, "band_lambda_m", default=float("nan"))
    if wavelengths is None or wavelengths.size != size or not np.isfinite(band_lambda):
        return 0
    return int(np.argmin(np.abs(wavelengths - band_lambda)))


def _payload_vector(
    payload: Mapping[str, Any],
    key: str,
) -> np.ndarray | None:
    raw = payload.get(key)
    if raw is None:
        return None
    try:
        values = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if values.ndim != 1:
        return None
    return values


def _integration_time_s_for_record(record: RunRecord | None) -> float:
    if record is None:
        return 4.0 * 3600.0
    for payload in record.result.state.results.components.values():
        if isinstance(payload, Mapping) and payload.get("integration_time_s") is not None:
            return float(payload["integration_time_s"])
    return 4.0 * 3600.0


def _spectro_port_count(char: Mapping[str, Any]) -> int:
    if char.get("spectro_indices") is not None:
        return max(int(np.asarray(char["spectro_indices"]).size), 1)
    return 1


def _char_dark_rate_hz_per_port(
    char: Mapping[str, Any],
    *,
    integration_time_s: float,
    n_spectro_ports: int,
) -> float:
    if integration_time_s <= 0.0 or n_spectro_ports <= 0:
        return 0.0
    dark_counts = _first_float(char, "D", default=0.0)
    return max(float(dark_counts), 0.0) / (
        float(integration_time_s) * float(n_spectro_ports)
    )


def _sky_rate_per_dark_port(record: RunRecord | None) -> float:
    """Background per dark port: sky plus the thermal and exozodi additions.

    ``performance.standard`` publishes the rate its SNR charged (an explicit
    parameter, else the newest background stage's
    ``background_rate_hz_per_dark_port``, else its sky). Without it, read the
    background stages the same way, newest first.
    """

    if record is None:
        return 0.0
    rate = _first_float(
        _find_component(record, "performance.standard"),
        "sky_rate_hz_per_port",
        default=float("nan"),
    )
    if np.isfinite(rate):
        return float(rate)
    payloads = [
        payload
        for payload in reversed(record.result.state.results.components.values())
        if isinstance(payload, Mapping)
    ]
    for key in (
        "background_rate_hz_per_dark_port",
        "sky_rate_hz_per_dark_port",
        "sky_rate_per_dark_port_hz",
    ):
        for payload in payloads:
            if payload.get(key) is not None:
                return float(payload[key])
    return 0.0


def _frame_rate_hz(record: RunRecord | None) -> float:
    if record is None:
        return 0.0
    fringe = _find_component(record, "fringe_tracking.closed_loop")
    return _first_float(
        fringe, "servo_bandwidth_hz", "closed_loop_bandwidth_hz", default=0.0
    )


def _assumed_star_disk_mas(collection: RunCollection) -> float:
    """Angular diameter (mas) of the simulated reference star.

    The bundled catalog snapshot carries no per-host stellar radius, so the
    contrast and characterization floors are computed for the single configured
    scene star and applied to every overlay planet. This column is therefore
    the *assumed* stellar disk used in the floor, not each catalog host's true
    angular diameter.
    """

    try:
        return float(collection.config.scene.star.angular_diameter_mas)
    except (AttributeError, TypeError, ValueError):
        return 0.0


def _float_value(row: Any, key: str) -> float | None:
    if row is None:
        return None
    try:
        value = float(row.get(key))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _first_float(
    payload: Mapping[str, Any] | None,
    *keys: str,
    default: float | None = None,
) -> float:
    if payload is not None:
        for key in keys:
            if payload.get(key) is None:
                continue
            try:
                value = float(payload[key])
            except (TypeError, ValueError):
                continue
            return value
    return float("nan") if default is None else float(default)


def _wavelength_grid_m(collection: RunCollection) -> np.ndarray:
    if collection.records:
        state_grid = collection.records[0].result.state.field.wavelength_grid_um
        if state_grid:
            return np.asarray(state_grid, dtype=np.float64) * 1.0e-6
    center = float(collection.config.grid.wavelength_center_um)
    width = float(collection.config.grid.wavelength_bandwidth_um)
    n = int(collection.config.grid.n_wavelength_bins)
    if n <= 1 or width <= 0.0:
        return np.asarray([center], dtype=np.float64) * 1.0e-6
    return np.linspace(center - width / 2.0, center + width / 2.0, n) * 1.0e-6


def _dirty_beam_fov_mas(positions_m: np.ndarray, wavelength_m: float) -> float:
    baselines = _pairwise_lengths(positions_m)
    if baselines.size == 0:
        return 100.0
    max_baseline = max(float(np.max(baselines)), 1.0)
    resolution_mas = wavelength_m / max_baseline * 206265.0e3
    return float(np.clip(12.0 * resolution_mas, 20.0, 300.0))


def _dirty_beam(
    positions_m: np.ndarray,
    wavelengths_m: tuple[float, ...],
    coords_mas: np.ndarray,
) -> np.ndarray:
    xy_rad = coords_mas / 206265.0e3
    l_grid, m_grid = np.meshgrid(xy_rad, xy_rad)
    n_apertures = int(positions_m.shape[0])
    if n_apertures < 2:
        return np.ones_like(l_grid)
    beam = np.zeros_like(l_grid, dtype=np.float64)
    # Power (dirty) beam |sum_k exp(i 2*pi/lambda (x_k l + y_k m))|^2,
    # computed directly from the aperture positions. The previous
    # pair-sum version used a DC term of 1 instead of N and then squared
    # the already-power-domain pattern, shrinking half-power contours
    # (true 0.5 -> plotted ~0.21), quartically suppressing sidelobes,
    # and creating spurious bright lobes where 1 + sum(2cos) < 0
    # (review 2026-06-10 Issue 10).
    for wavelength_m in wavelengths_m:
        phase = (
            2.0
            * np.pi
            / float(wavelength_m)
            * (
                positions_m[:, 0, np.newaxis, np.newaxis] * l_grid
                + positions_m[:, 1, np.newaxis, np.newaxis] * m_grid
            )
        )
        response = np.sum(np.exp(1j * phase), axis=0)
        beam += np.abs(response) ** 2 / float(n_apertures) ** 2
    beam /= max(len(wavelengths_m), 1)
    peak = float(np.max(beam))
    return beam / peak if peak > 0.0 else beam


def _planet_polarization_fraction(
    phase_angle_rad: Any, radius_rearth: float
) -> np.ndarray:
    alpha = np.asarray(phase_angle_rad, dtype=np.float64)
    shape = np.sin(alpha) ** 2 / (1.0 + np.cos(alpha) ** 2)
    return _blended_peak_polarization(radius_rearth) * shape


def _orbit_averaged_polarization_fraction(radius_rearth: float) -> float:
    return float(
        _blended_peak_polarization(radius_rearth) * _ORBIT_AVERAGED_POL_SHAPE_FACTOR
    )


def _blended_peak_polarization(radius_rearth: float) -> float:
    radius = float(radius_rearth)
    if radius <= _POL_MODEL_CROSSOVER_REARTH:
        return _EARTHLIKE_PEAK_POL_FRAC
    if radius >= _POL_MODEL_BLEND_HI_REARTH:
        return _CLOUDY_GIANT_PEAK_POL_FRAC
    t = (radius - _POL_MODEL_CROSSOVER_REARTH) / (
        _POL_MODEL_BLEND_HI_REARTH - _POL_MODEL_CROSSOVER_REARTH
    )
    return (1.0 - t) * _EARTHLIKE_PEAK_POL_FRAC + t * _CLOUDY_GIANT_PEAK_POL_FRAC


def _tex_escape(value: Any) -> str:
    text = str(value)
    for old, new in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    ):
        text = text.replace(old, new)
    return text


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _array_summary(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            return str(value)
        return value
    return str(value)


def _array_summary(value: np.ndarray) -> Any:
    arr = np.asarray(value)
    if arr.size <= 128:
        return _json_safe(arr.tolist())
    finite = np.isfinite(np.abs(arr)) if np.iscomplexobj(arr) else np.isfinite(arr)
    magnitude = np.abs(arr) if np.iscomplexobj(arr) else arr
    summary: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }
    if finite.any():
        summary["min"] = float(np.nanmin(magnitude[finite]))
        summary["max"] = float(np.nanmax(magnitude[finite]))
    return summary


def _range_or_none(values: np.ndarray) -> list[float] | None:
    if values.size == 0:
        return None
    return [float(np.min(values)), float(np.max(values))]


def _component_name(record: RunRecord, prefix: str) -> str | None:
    for name, payload in reversed(record.result.state.results.components.items()):
        if name.startswith(prefix) and isinstance(payload, Mapping):
            return name
    return None


def _int_from_payload(payload: Mapping[str, Any] | None, key: str) -> int | None:
    if payload is None or payload.get(key) is None:
        return None
    try:
        return int(payload[key])
    except (TypeError, ValueError):
        return None


def _float_from_payload(payload: Mapping[str, Any] | None, key: str) -> float | None:
    if payload is None or payload.get(key) is None:
        return None
    try:
        value = float(payload[key])
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _nested_payload(
    payload: Mapping[str, Any] | None, key: str
) -> Mapping[str, Any] | None:
    if payload is None:
        return None
    value = payload.get(key)
    return value if isinstance(value, Mapping) else None


def _count_label(count: int | None, noun: str) -> str:
    if count is None:
        return noun + "s"
    return f"{count} {noun}" + ("" if count == 1 else "s")


def _rank_label(rank: int | None, dof: int | None) -> str:
    if rank is None and dof is None:
        return "n/a"
    if rank is None:
        return f"?/{dof}"
    if dof is None:
        return str(rank)
    return f"{rank}/{dof}"


def _number_label(value: float | None) -> str:
    if value is None:
        return "n/a"
    if not np.isfinite(value):
        if value != value:
            return "nan"
        return "inf" if value > 0 else "-inf"
    return f"{value:.3g}"


def _percent_label(value: float | None) -> str:
    if value is None:
        return "n/a"
    if not np.isfinite(value):
        return _number_label(value)
    return f"{100.0 * value:.1f}%"


def _nm_label(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3g} nm"


def _aperture_label(index: int, collection: RunCollection) -> str:
    stage_labels = _telescope_labels_from_config(collection.config)
    if index < len(stage_labels):
        return stage_labels[index]
    return f"Aperture {index}"


def _telescope_labels_from_config(config: Any) -> tuple[str, ...]:
    labels: list[str] = []
    for stage in getattr(config.pipeline, "stages", ()):
        if getattr(stage, "type", None) != "telescope.fixed_aperture":
            continue
        raw = getattr(stage, "aperture_names", None) or getattr(stage, "names", None)
        if isinstance(raw, Iterable) and not isinstance(raw, (str, bytes)):
            labels.extend(str(item) for item in raw)
    return tuple(labels)


def _record_generated(manifest: dict[str, Any], filename: str, source: str) -> None:
    manifest["generated"].append({"file": filename, "source": source})


def _record_skip(manifest: dict[str, Any], filename: str, reason: str) -> None:
    manifest["skipped"].append({"file": filename, "reason": reason})
