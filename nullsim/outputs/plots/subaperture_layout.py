"""Sub-aperture layout figure (KILO paperv3 Fig 3).

Direct port of ``kilo/plotting.py::plot_subaperture_layout``. One panel per
telescope showing the outer pupil boundary, central obstruction, spider
vanes, and the optimized sub-aperture circles (with index labels).

The figure is config-derivable — it does not need a completed pipeline
run. The registered-plot wrapper reads telescope diameter / aperture
count / pupil template / N_sub from the resolved config's stage params.

Today the ``"keck"`` and ``"eelt"`` pupil templates are supported; ``"circular"``
and ``"full_pupil"`` fall back to the plain circular aperture panel.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib  # noqa: F401  # backend choice managed by styles module
import numpy as np

from nullsim.outputs.styles import _ensure_headless_backend  # noqa: F401  # side-effect

from matplotlib import pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from numpy.typing import NDArray

from nullsim.outputs import RunCollection
from nullsim.physics.pupil_geometry import (
    PupilGeometry,
    eelt_pupil,
    keck_pupil,
    optimize_subaperture_positions,
)
from nullsim.pipeline import register_plot


@dataclass(frozen=True)
class TelescopePanel:
    """One panel's worth of geometry for the subaperture_layout figure."""

    name: str
    diameter_m: float
    pupil: PupilGeometry | None
    mode_positions_m: NDArray[np.float64] | None = None
    mode_diameter_m: float | None = None
    layout_name: str | None = None


_SUPPORTED_PUPIL_TEMPLATES: dict[str, Callable[[], PupilGeometry]] = {
    "eelt": eelt_pupil,
    "keck": keck_pupil,
}


def _build_pupil_template(name: str | None) -> PupilGeometry | None:
    """Reconstruct a ``PupilGeometry`` from a template name (or ``None``)."""

    if name is None or name in {"circular", "full_pupil"}:
        return None
    try:
        factory = _SUPPORTED_PUPIL_TEMPLATES[name]
    except KeyError:
        raise ValueError(
            f"subaperture_layout: pupil template {name!r} not supported. "
            f"Known templates: {sorted(_SUPPORTED_PUPIL_TEMPLATES)}."
        )
    return factory()


def _draw_spider_vanes(ax: plt.Axes, pupil: PupilGeometry) -> None:
    """Draw spider vanes as filled rectangles (port of KILO ``_draw_spider_vanes``)."""

    for vane in pupil.spider_vanes:
        dx = vane.x_end - vane.x_start
        dy = vane.y_end - vane.y_start
        length = np.hypot(dx, dy)
        if length == 0:
            continue
        nx = -dy / length * vane.width_m / 2
        ny = dx / length * vane.width_m / 2
        corners = [
            (vane.x_start + nx, vane.y_start + ny),
            (vane.x_end + nx, vane.y_end + ny),
            (vane.x_end - nx, vane.y_end - ny),
            (vane.x_start - nx, vane.y_start - ny),
        ]
        ax.add_patch(MplPolygon(corners, facecolor="0.5", edgecolor="0.4", alpha=0.6))


def plot_subaperture_layout(
    telescopes: Sequence[TelescopePanel],
    n_sub: int,
    save_path: Path | None = None,
) -> plt.Figure:
    """Render one panel per telescope. Direct port of KILO ``plot_subaperture_layout``."""

    n_tel = len(telescopes)
    fig, axes = plt.subplots(1, n_tel, figsize=(4 * n_tel, 4.5))
    if n_tel == 1:
        axes = [axes]

    for ax, tel in zip(axes, telescopes):
        diameter = tel.diameter_m
        pupil = tel.pupil

        if pupil is not None and pupil.boundary_vertices:
            verts = list(pupil.boundary_vertices) + [pupil.boundary_vertices[0]]
            xs = [v[0] for v in verts]
            ys = [v[1] for v in verts]
            ax.plot(xs, ys, "k-", linewidth=1.5)
            max_r = max(np.hypot(v[0], v[1]) for v in pupil.boundary_vertices)
            display_diameter = 2.0 * max_r
        else:
            ax.add_patch(plt.Circle(
                (0, 0), diameter / 2,
                fill=False, edgecolor="k", linewidth=1.5,
            ))
            display_diameter = diameter

        obs_r = pupil.central_obstruction_diameter_m / 2 if pupil is not None else 0.0
        if obs_r > 0:
            ax.add_patch(plt.Circle(
                (0, 0), obs_r,
                fill=True, facecolor="0.85", edgecolor="k", linewidth=1.0,
            ))

        if pupil is not None and pupil.spider_vanes:
            _draw_spider_vanes(ax, pupil)

        positions, d_sub = _panel_positions(tel, n_sub)

        for i, (x, y) in enumerate(positions):
            if d_sub > 0.0:
                ax.add_patch(plt.Circle(
                    (x, y), d_sub / 2,
                    fill=True, facecolor="C0", edgecolor="C0",
                    alpha=0.35, linewidth=1.0,
                ))
            else:
                ax.plot(x, y, "o", color="C0", alpha=0.7, markersize=4)
            ax.text(x, y, str(i), ha="center", va="center", fontsize=7, color="k")

        margin = diameter * 0.6
        ax.set_xlim(-margin, margin)
        ax.set_ylim(-margin, margin)
        ax.set_aspect("equal")
        layout_suffix = "" if tel.layout_name is None else f"\n{tel.layout_name}"
        ax.set_title(
            f"{tel.name}\n{display_diameter:.2f} m, {n_sub} subs{layout_suffix}",
            fontsize=11,
        )
        ax.set_xlabel("E (m)")
        ax.set_ylabel("N (m)")
        ax.grid(True, alpha=0.2)
        if d_sub > 0.0:
            ax.text(
                0.02, 0.02,
                r"$d_\mathrm{sub}$" + f" = {d_sub:.2f} m",
                transform=ax.transAxes, fontsize=9, verticalalignment="bottom",
            )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)

    return fig


def _panel_positions(
    tel: TelescopePanel,
    n_sub: int,
) -> tuple[NDArray[np.float64], float]:
    if tel.mode_positions_m is not None:
        positions = np.asarray(tel.mode_positions_m, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(
                "TelescopePanel.mode_positions_m must have shape (n_sub, 2)."
            )
        diameter = 0.0 if tel.mode_diameter_m is None else float(tel.mode_diameter_m)
        return positions, diameter
    positions, d_sub = optimize_subaperture_positions(n_sub, tel.diameter_m, tel.pupil)
    return np.asarray(positions, dtype=np.float64), float(d_sub)


# ---------------------------------------------------------------------------
# Registered-plot wrapper: extract inputs from RunCollection.config
# ---------------------------------------------------------------------------


def _stage_extra(stage_spec, key: str, default=None):
    """Read a parameter from a ``StageSpec`` (which uses ``extra='allow'``)."""

    extra = getattr(stage_spec, "model_extra", None) or {}
    return extra.get(key, default)


def _find_stage(config, type_name: str):
    for stage in config.pipeline.stages:
        if stage.type == type_name:
            return stage
    return None


def _extract_telescopes(config, state=None) -> tuple[TelescopePanel, ...]:
    tel_stage = _find_stage(config, "telescope.fixed_aperture")
    if tel_stage is None:
        raise ValueError(
            "subaperture_layout requires a 'telescope.fixed_aperture' stage."
        )
    ao_stage = _find_stage(config, "ao.pyramid_wfs")
    pupil_template_name = _stage_extra(ao_stage, "pupil_template") if ao_stage else None

    diameter_m = float(_stage_extra(tel_stage, "diameter_m", 1.0))
    n_apertures = int(_stage_extra(tel_stage, "n_apertures", 1))
    pupil = _build_pupil_template(pupil_template_name)

    mode_positions, mode_diameter, layout_name = _state_pupil_mode_layout(state)
    if mode_positions is not None and n_apertures == 1:
        return (
            TelescopePanel(
                name="Aperture 0",
                diameter_m=diameter_m,
                pupil=pupil,
                mode_positions_m=mode_positions,
                mode_diameter_m=mode_diameter,
                layout_name=layout_name,
            ),
        )

    return tuple(
        TelescopePanel(
            name=f"Aperture {i}",
            diameter_m=diameter_m,
            pupil=pupil,
        )
        for i in range(n_apertures)
    )


def _state_pupil_mode_layout(
    state,
) -> tuple[NDArray[np.float64] | None, float | None, str | None]:
    if state is None:
        return None, None, None
    payload = state.geometry.components.get("pupil_modes")
    if not isinstance(payload, dict):
        return None, None, None
    raw_positions = payload.get("aperture_positions_pupil_m")
    if raw_positions is None:
        return None, None, None
    positions = np.asarray(raw_positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2:
        return None, None, None
    mode_diameter = payload.get("mode_diameter_m")
    return (
        positions,
        None if mode_diameter is None else float(mode_diameter),
        str(payload.get("layout", "pupil_modes")),
    )


def _extract_n_sub(config) -> int:
    ao_stage = _find_stage(config, "ao.pyramid_wfs")
    if ao_stage is not None:
        return int(_stage_extra(ao_stage, "n_subapertures_per_aperture", 1))
    pupil_modes_stage = _find_stage(config, "telescope.pupil_modes")
    if pupil_modes_stage is not None:
        return int(_stage_extra(pupil_modes_stage, "n_modes", 1))
    return 1


@register_plot
def subaperture_layout(
    collection: RunCollection,
    output_dir: Path,
    formats: Iterable[str],
) -> tuple[Path, ...]:
    """Registered plot: emit ``subaperture_layout.<fmt>`` per requested format."""

    state = collection.records[0].result.state if collection.records else None
    telescopes = _extract_telescopes(collection.config, state)
    n_sub = _extract_n_sub(collection.config)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig = plot_subaperture_layout(telescopes, n_sub, save_path=None)

    written: list[Path] = []
    for fmt in formats:
        out = plot_dir / f"subaperture_layout.{fmt}"
        fig.savefig(out)
        written.append(out)
    plt.close(fig)
    return tuple(written)
