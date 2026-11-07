"""Plot the dedicated pairwise ABCD cophasing analysis matrix."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib  # noqa: F401  # backend choice managed by styles module
import numpy as np
from numpy.typing import NDArray

from nullsim.outputs.styles import _ensure_headless_backend  # noqa: F401  # side-effect

from matplotlib.patches import FancyArrowPatch
from matplotlib import pyplot as plt

from nullsim.outputs import RunCollection, RunRecord
from nullsim.pipeline import register_plot


_PHASE_LABELS = ("A", "B", "C", "D")


@register_plot
def cophasing_unitary(
    collection: RunCollection,
    output_dir: Path,
    formats: Iterable[str],
) -> tuple[Path, ...]:
    """Write ``cophasing_unitary`` plots for the first record with payload."""

    payload = None
    matched_record = None
    for record in collection.records:
        payload = _find_payload(record)
        if payload is not None:
            matched_record = record
            break
    if payload is None:
        return ()
    record_label = None
    if matched_record is not None and len(collection.records) > 1:
        from nullsim.outputs.plots._axis_labels import compact_record_label

        record_label = compact_record_label(
            matched_record.coord_tuple,
            matched_record.config.run.name,
        )

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for file_format in formats:
        path = plot_dir / f"cophasing_unitary.{file_format}"
        fig = plot_cophasing_unitary(payload, record_label=record_label)
        fig.savefig(path)
        plt.close(fig)
        written.append(path)
    return tuple(written)


def plot_cophasing_unitary(
    payload: dict[str, Any],
    record_label: str | None = None,
) -> plt.Figure:
    """Return a diagnostic figure for the dedicated cophasing unitary."""

    model = _CophasingModel.from_payload(payload)
    fig = plt.figure(figsize=(13.4, 9.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=(1.1, 1.0, 1.0))
    ax_power = fig.add_subplot(grid[0, 0])
    ax_abs = fig.add_subplot(grid[0, 1])
    ax_phase = fig.add_subplot(grid[0, 2])
    ax_graph = fig.add_subplot(grid[1, 0])
    ax_incidence = fig.add_subplot(grid[1, 1])
    ax_summary = fig.add_subplot(grid[1, 2])

    _plot_output_allocation(ax_power, model)
    _plot_matrix_abs(ax_abs, model)
    _plot_matrix_phase(ax_phase, model)
    _plot_pair_graph(ax_graph, model)
    _plot_incidence(ax_incidence, model)
    _plot_summary(ax_summary, payload, model)
    suffix = f"  ·  {record_label}" if record_label else ""
    fig.suptitle(f"Dedicated cophasing unitary (pairwise ABCD mode){suffix}", y=1.02)
    return fig


class _CophasingModel:
    def __init__(
        self,
        *,
        edges: NDArray[np.int64],
        phase_offsets_rad: NDArray[np.float64],
        edge_weights: NDArray[np.float64],
        edge_throughput: float,
        tracker_fraction: float,
        visibility: float,
        positions_pupil_m: NDArray[np.float64] | None,
    ) -> None:
        self.edges = edges
        self.phase_offsets_rad = phase_offsets_rad
        self.edge_weights = edge_weights
        self.edge_throughput = float(edge_throughput)
        self.tracker_fraction = float(tracker_fraction)
        self.visibility = float(visibility)
        self.positions_pupil_m = positions_pupil_m
        self.n_modes = int(np.max(edges)) + 1 if edges.size else 0
        self.analysis_matrix = _analysis_matrix(
            edges=edges,
            phase_offsets_rad=phase_offsets_rad,
            edge_weights=edge_weights,
            edge_throughput=float(edge_throughput),
        )
        self.output_count_fraction = np.repeat(
            edge_weights * float(edge_throughput) * float(tracker_fraction) / 4.0,
            phase_offsets_rad.size,
        )

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "_CophasingModel":
        cophasing = payload.get("cophasing_handoff_monte_carlo")
        if not isinstance(cophasing, dict):
            raise ValueError("cophasing_unitary requires cophasing handoff payload.")
        edges = np.asarray(cophasing.get("edges", ()), dtype=np.int64)
        if edges.ndim != 2 or edges.shape[1] != 2 or edges.shape[0] == 0:
            raise ValueError("cophasing_unitary requires non-empty pair edges.")
        phase_offsets = np.asarray(
            payload.get(
                "cophasing_phase_offsets_rad",
                (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi),
            ),
            dtype=np.float64,
        )
        if phase_offsets.shape != (4,) or not np.all(np.isfinite(phase_offsets)):
            raise ValueError("cophasing_unitary requires four finite phase offsets.")
        weights = _edge_weights(
            payload.get("cophasing_edge_count_weights"),
            n_edges=edges.shape[0],
        )
        positions = _positions(payload.get("positions_pupil_m"), edges)
        display_order = _cluster_display_order(edges, positions)
        edges = edges[display_order]
        weights = weights[display_order]
        return cls(
            edges=edges,
            phase_offsets_rad=phase_offsets,
            edge_weights=weights,
            edge_throughput=float(payload.get("cophasing_edge_throughput", 1.0)),
            tracker_fraction=float(payload.get("cophasing_tracker_fraction", 1.0)),
            visibility=float(payload.get("cophasing_visibility", 1.0)),
            positions_pupil_m=positions,
        )


def _find_payload(record: RunRecord) -> dict[str, Any] | None:
    candidates: list[tuple[str, dict[str, Any]]] = []
    for name, value in record.result.state.results.components.items():
        if not isinstance(value, dict):
            continue
        if name.startswith(("fringe_observability", "diagnostics.fringe_observability")):
            if isinstance(value.get("cophasing_handoff_monte_carlo"), dict):
                candidates.append((name, value))
            continue
        if isinstance(value.get("cophasing_handoff_monte_carlo"), dict):
            candidates.append((name, value))
    if not candidates:
        return None
    for name, value in candidates:
        if name in {"fringe_observability", "diagnostics.fringe_observability"}:
            return value
    return candidates[-1][1]


def _analysis_matrix(
    *,
    edges: NDArray[np.int64],
    phase_offsets_rad: NDArray[np.float64],
    edge_weights: NDArray[np.float64],
    edge_throughput: float,
) -> NDArray[np.complex128]:
    n_modes = int(np.max(edges)) + 1
    matrix = np.zeros((edges.shape[0] * phase_offsets_rad.size, n_modes), dtype=np.complex128)
    for edge_i, (mode_a, mode_b) in enumerate(edges):
        scale = 0.5 * np.sqrt(float(edge_weights[edge_i]) * float(edge_throughput))
        for phase_i, phase in enumerate(phase_offsets_rad):
            row = edge_i * phase_offsets_rad.size + phase_i
            matrix[row, int(mode_a)] = scale
            # e^{-i theta} on mode_b reproduces the measurement model
            # 1 + V cos(phi_a - phi_b + theta) of
            # physics.fringe_observability._measure_abcd_pair_edges.
            matrix[row, int(mode_b)] = scale * np.exp(-1j * float(phase))
    return matrix


def _edge_weights(raw: Any, *, n_edges: int) -> NDArray[np.float64]:
    if raw is None or (isinstance(raw, (tuple, list)) and len(raw) == 0):
        return np.full(n_edges, 1.0 / float(n_edges), dtype=np.float64)
    weights = np.asarray(raw, dtype=np.float64)
    if weights.shape != (n_edges,):
        return np.full(n_edges, 1.0 / float(n_edges), dtype=np.float64)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        return np.full(n_edges, 1.0 / float(n_edges), dtype=np.float64)
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.full(n_edges, 1.0 / float(n_edges), dtype=np.float64)
    return (weights / total).astype(np.float64, copy=False)


def _positions(raw: Any, edges: NDArray[np.int64]) -> NDArray[np.float64]:
    n_modes = int(np.max(edges)) + 1
    if raw is not None:
        positions = np.asarray(raw, dtype=np.float64)
        if positions.shape == (n_modes, 2) and np.all(np.isfinite(positions)):
            return positions.astype(np.float64, copy=True)
    theta = np.linspace(0.0, 2.0 * np.pi, n_modes, endpoint=False)
    return np.column_stack((np.cos(theta), np.sin(theta))).astype(np.float64)


def _cluster_display_order(
    edges: NDArray[np.int64],
    positions_pupil_m: NDArray[np.float64],
) -> NDArray[np.intp]:
    """Place local cluster edges around the longest bridge edge for display."""

    if edges.shape[0] < 3:
        return np.arange(edges.shape[0], dtype=np.intp)
    bridge_index = _longest_edge_index(edges, positions_pupil_m)
    if bridge_index < 0:
        return np.arange(edges.shape[0], dtype=np.intp)
    bridge = tuple(int(value) for value in edges[bridge_index])
    n_modes = int(np.max(edges)) + 1
    components = _components_without_bridge(edges, n_modes=n_modes, bridge=bridge)
    if len(components) != 2:
        return np.arange(edges.shape[0], dtype=np.intp)

    comp_a, comp_b = sorted(components, key=lambda members: min(members))
    left: list[int] = []
    right: list[int] = []
    for edge_i, (mode_a, mode_b) in enumerate(edges):
        if edge_i == bridge_index:
            continue
        a = int(mode_a)
        b = int(mode_b)
        if a in comp_a and b in comp_a:
            left.append(edge_i)
        elif a in comp_b and b in comp_b:
            right.append(edge_i)
        else:
            return np.arange(edges.shape[0], dtype=np.intp)
    return np.asarray(left + [bridge_index] + right, dtype=np.intp)


def _components_without_bridge(
    edges: NDArray[np.int64],
    *,
    n_modes: int,
    bridge: tuple[int, int],
) -> list[set[int]]:
    neighbors: list[set[int]] = [set() for _ in range(n_modes)]
    bridge_set = frozenset(bridge)
    for mode_a, mode_b in edges:
        a = int(mode_a)
        b = int(mode_b)
        if frozenset((a, b)) == bridge_set:
            continue
        neighbors[a].add(b)
        neighbors[b].add(a)

    unseen = set(range(n_modes))
    components: list[set[int]] = []
    while unseen:
        root = min(unseen)
        stack = [root]
        members: set[int] = set()
        unseen.remove(root)
        while stack:
            node = stack.pop()
            members.add(node)
            for neighbor in sorted(neighbors[node]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(members)
    return components


def _plot_output_allocation(ax: plt.Axes, model: _CophasingModel) -> None:
    x = np.arange(model.output_count_fraction.size)
    colors = np.take(
        np.array(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]),
        x % model.phase_offsets_rad.size,
    )
    ax.bar(x, model.output_count_fraction, color=colors, width=0.82, linewidth=0.0)
    for edge in range(1, model.edges.shape[0]):
        ax.axvline(edge * model.phase_offsets_rad.size - 0.5, color="0.85", linewidth=0.5)
    for phase_i, label in enumerate(_phase_labels(model.phase_offsets_rad)):
        ax.bar([], [], color=colors[phase_i], label=label)
    ax.set_xlabel("Cophasing output index")
    ax.set_ylabel("Fraction of tracker counts")
    ax.set_title("ABCD output allocation")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, ncols=2, framealpha=0.9)


def _plot_matrix_abs(ax: plt.Axes, model: _CophasingModel) -> None:
    image = ax.imshow(
        np.abs(model.analysis_matrix),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
    )
    ax.set_xlabel("Input aperture index")
    ax.set_ylabel("ABCD output row")
    ax.set_title(r"$|U_{\rm coph,ij}|$")
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def _plot_matrix_phase(ax: plt.Axes, model: _CophasingModel) -> None:
    phase = np.ma.array(
        np.angle(model.analysis_matrix),
        mask=np.abs(model.analysis_matrix) <= 0.0,
    )
    cmap = plt.get_cmap("twilight_shifted").copy()
    cmap.set_bad("0.94")
    image = ax.imshow(
        phase,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=-np.pi,
        vmax=np.pi,
    )
    ax.set_xlabel("Input aperture index")
    ax.set_ylabel("ABCD output row")
    ax.set_title(r"$\arg(U_{\rm coph,ij})$")
    cbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([-np.pi, -0.5 * np.pi, 0.0, 0.5 * np.pi, np.pi])
    cbar.set_ticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])


def _plot_pair_graph(ax: plt.Axes, model: _CophasingModel) -> None:
    pos = model.positions_pupil_m
    edge_weights = model.edge_weights
    finite = np.isfinite(edge_weights)
    max_weight = float(np.max(edge_weights[finite])) if np.any(finite) else 1.0
    uniform_weights = _is_uniform(edge_weights)
    bridge_index = _longest_edge_index(model.edges, pos)
    cmap = plt.get_cmap("magma")
    for edge_i, (mode_a, mode_b) in enumerate(model.edges):
        weight = float(edge_weights[edge_i])
        normalized = weight / max(max_weight, np.finfo(float).tiny)
        is_bridge = edge_i == bridge_index
        color = "#2f5f93" if uniform_weights else cmap(0.25 + 0.7 * normalized)
        linewidth = (
            (2.1 if is_bridge else 1.45)
            if uniform_weights
            else 0.9 + 3.4 * normalized
        )
        patch = FancyArrowPatch(
            posA=(pos[int(mode_a), 0], pos[int(mode_a), 1]),
            posB=(pos[int(mode_b), 0], pos[int(mode_b), 1]),
            arrowstyle="-",
            connectionstyle=f"arc3,rad={_edge_curve(edge_i, is_bridge=is_bridge)}",
            shrinkA=8.0 if is_bridge else 0.0,
            shrinkB=8.0 if is_bridge else 0.0,
            color=color,
            linewidth=linewidth,
            alpha=0.86 if is_bridge else 0.9,
            zorder=6 if not is_bridge else 4,
        )
        ax.add_patch(patch)
        if is_bridge:
            mid = 0.5 * (pos[int(mode_a)] + pos[int(mode_b)])
            ax.text(
                mid[0],
                mid[1],
                f"bridge {int(mode_a)}-{int(mode_b)}",
                fontsize=7,
                color="#2f5f93",
                ha="center",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "fc": "white",
                    "ec": "none",
                    "alpha": 0.75,
                },
                zorder=7,
            )
    ax.scatter(
        pos[:, 0],
        pos[:, 1],
        s=58,
        color="#d9ecff",
        edgecolor="0.20",
        linewidth=0.8,
        alpha=0.72,
        zorder=5,
    )
    for index, (x, y) in enumerate(pos):
        ax.text(
            x,
            y,
            str(index),
            ha="center",
            va="center",
            fontsize=7,
            color="0.10",
            zorder=6,
        )
    if bridge_index >= 0:
        bridge_nodes = model.edges[bridge_index].astype(np.intp, copy=False)
        ax.scatter(
            pos[bridge_nodes, 0],
            pos[bridge_nodes, 1],
            s=118,
            facecolor="none",
            edgecolor="#b45309",
            linewidth=1.7,
            zorder=8,
        )
        upper = int(bridge_nodes[np.argmax(pos[bridge_nodes, 1])])
        offset = _bridge_label_offset(pos, upper)
        ax.text(
            pos[upper, 0] + offset[0],
            pos[upper, 1] + offset[1],
            f"bridge endpoint {upper}",
            fontsize=7,
            color="#8a3a00",
            ha="left" if offset[0] >= 0.0 else "right",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.12",
                "fc": "white",
                "ec": "none",
                "alpha": 0.75,
            },
            zorder=9,
        )
    lim = max(float(np.nanmax(np.abs(pos))) * 1.12, 1.0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Pupil x (m)")
    ax.set_ylabel("Pupil y (m)")
    ax.set_title("Pairwise cophasing graph")
    ax.grid(True, alpha=0.2)


def _longest_edge_index(
    edges: NDArray[np.int64],
    positions_pupil_m: NDArray[np.float64],
) -> int:
    if edges.size == 0:
        return -1
    starts = positions_pupil_m[edges[:, 0]]
    stops = positions_pupil_m[edges[:, 1]]
    lengths = np.linalg.norm(stops - starts, axis=1)
    return int(np.argmax(lengths))


def _edge_curve(edge_index: int, *, is_bridge: bool) -> float:
    if is_bridge:
        return 0.0
    return 0.18 if edge_index % 2 == 0 else -0.18


def _bridge_label_offset(
    positions_pupil_m: NDArray[np.float64],
    upper_endpoint_index: int,
) -> NDArray[np.float64]:
    center = np.mean(positions_pupil_m, axis=0)
    endpoint = positions_pupil_m[int(upper_endpoint_index)]
    direction = endpoint - center
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        return np.array([2.0, 1.4], dtype=np.float64)
    return 3.0 * direction / norm


def _plot_incidence(ax: plt.Axes, model: _CophasingModel) -> None:
    incidence = np.zeros((model.edges.shape[0], model.n_modes), dtype=np.float64)
    for edge_i, (mode_a, mode_b) in enumerate(model.edges):
        incidence[edge_i, int(mode_a)] = 1.0
        incidence[edge_i, int(mode_b)] = -1.0
    image = ax.imshow(
        incidence,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )
    ax.set_xlabel("Aperture index")
    ax.set_ylabel("Pair edge index")
    ax.set_title("OPD measurement incidence")
    cbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([-1.0, 0.0, 1.0])
    cbar.set_ticklabels(["-1", "0", "+1"])


def _plot_summary(
    ax: plt.Axes,
    payload: dict[str, Any],
    model: _CophasingModel,
) -> None:
    ax.set_axis_off()
    cophasing = payload.get("cophasing_handoff_monte_carlo")
    assert isinstance(cophasing, dict)
    rows: list[tuple[str, str | None]] = [
        ("Model", None),
        ("Physical apertures", str(model.n_modes)),
        ("Pair edges", str(model.edges.shape[0])),
        ("ABCD outputs", str(model.analysis_matrix.shape[0])),
        ("Phase offsets", ", ".join(_phase_labels(model.phase_offsets_rad))),
        ("Tracker fraction", _fmt_float(model.tracker_fraction)),
        ("Edge throughput", _fmt_float(model.edge_throughput)),
        ("Visibility", _fmt_float(model.visibility)),
        ("Count allocation", "uniform" if _is_uniform(model.edge_weights) else "weighted"),
        ("Diagnostics", None),
        (
            "Acq. median",
            _fmt_quantity(cophasing.get("median_acquisition_residual_rms_nm"), "nm"),
        ),
        ("Acq. lock frac.", _fmt_float(cophasing.get("acquisition_lock_fraction"))),
        (
            "Science maint.",
            _fmt_quantity(cophasing.get("median_science_residual_rms_nm"), "nm"),
        ),
        ("Maint. lock frac.", _fmt_float(cophasing.get("science_lock_fraction"))),
        ("ABCD wrap frac.", _fmt_float(cophasing.get("edge_wrap_fraction"))),
        ("Max OPD / lambda", _fmt_float(cophasing.get("max_edge_opd_over_lambda"))),
    ]
    ax.text(
        0.02,
        0.98,
        "Summary",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    y = 0.89
    for label, value in rows:
        if value is None:
            ax.text(
                0.02,
                y,
                label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9.4,
                fontweight="bold",
                color="0.18",
            )
            y -= 0.055
            continue
        ax.text(
            0.02,
            y,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.8,
            color="0.35",
        )
        ax.text(
            0.98,
            y,
            value,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.8,
            family="monospace",
            color="0.05",
        )
        y -= 0.047
    ax.text(
        0.02,
        0.03,
        "Rows are the effective ABCD analysis outputs used by the "
        "mode-switch cophasing diagnostic.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.6,
        color="0.45",
        wrap=True,
    )


def _phase_labels(phases: NDArray[np.float64]) -> list[str]:
    labels: list[str] = []
    for index, phase in enumerate(phases):
        prefix = _PHASE_LABELS[index] if index < len(_PHASE_LABELS) else f"P{index}"
        labels.append(f"{prefix} {_fmt_phase(float(phase))}")
    return labels


def _fmt_phase(value: float) -> str:
    wrapped = float(np.mod(value, 2.0 * np.pi))
    if np.isclose(wrapped, 0.0) or np.isclose(wrapped, 2.0 * np.pi):
        return "0"
    if np.isclose(wrapped, 0.5 * np.pi):
        return "pi/2"
    if np.isclose(wrapped, np.pi):
        return "pi"
    if np.isclose(wrapped, 1.5 * np.pi):
        return "3pi/2"
    return f"{value:.2g} rad"


def _fmt_float(value: Any) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(x):
        return "inf" if x > 0 else "-inf" if x < 0 else "nan"
    return f"{x:.3g}"


def _fmt_quantity(value: Any, unit: str) -> str:
    """Value with unit, or bare "n/a" — avoids rendering "n/a nm"."""

    text = _fmt_float(value)
    return text if text == "n/a" else f"{text} {unit}"


def _is_uniform(values: NDArray[np.float64]) -> bool:
    if values.size == 0:
        return True
    return bool(np.allclose(values, np.full_like(values, float(np.mean(values)))))


__all__ = ["cophasing_unitary", "plot_cophasing_unitary"]
