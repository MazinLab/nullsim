"""(u, v) coverage grid figure (KILO paperv3 Fig 5).

Direct port of ``kilo/studies/uv_coverage.run_study`` (the
``uv_coverage_grid.pdf`` panel). A ``len(declinations) × len(integration_configs)``
grid of (u, v) tracks at Y-band, intra-telescope baselines colored ``C0``,
inter-telescope baselines colored ``C3``. Both ``±(u, v)`` are scattered
to show the Hermitian symmetry.

The plot is config-derivable: it reads telescope ENU positions,
sub-aperture count, pupil template, and Maunakea latitude from the
resolved config.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib  # noqa: F401  # backend choice managed by styles module
import numpy as np

from nullsim.outputs.styles import _ensure_headless_backend  # noqa: F401  # side-effect

from matplotlib import pyplot as plt

from nullsim.outputs import RunCollection
from nullsim.outputs.plots.subaperture_layout import (
    _build_pupil_template,
    _extract_n_sub,
    _find_stage,
    _stage_extra,
)
from nullsim.physics.pupil_geometry import optimize_subaperture_positions
from nullsim.physics.uv_coverage import (
    Baseline,
    compute_uv_tracks,
    enumerate_subaperture_baselines,
)
from nullsim.pipeline import register_plot


_DEFAULT_WAVELENGTHS_M: dict[str, float] = {
    "Y": 1.00e-6,
    "J": 1.25e-6,
    "H": 1.65e-6,
}

_DEFAULT_DECLINATIONS_DEG: tuple[float, ...] = (0.0, 20.0, 40.0, 60.0)

_DEFAULT_INTEGRATION_CONFIGS: tuple[tuple[str, float, float], ...] = (
    ("1h", -7.5, 7.5),
    ("4h", -30.0, 30.0),
    ("8h", -60.0, 60.0),
)


def plot_uv_coverage_grid(
    baselines: Sequence[Baseline],
    *,
    wavelength_m: float,
    declinations_deg: Sequence[float] = _DEFAULT_DECLINATIONS_DEG,
    integration_configs: Sequence[tuple[str, float, float]] = _DEFAULT_INTEGRATION_CONFIGS,
    latitude_deg: float = 19.8283,
    save_path: Path | None = None,
) -> plt.Figure:
    """Render the (u, v) coverage grid figure. Direct port of KILO ``run_study``."""

    # Global axis limit from the largest baseline + widest HA sweep.
    uv_limit = 0.0
    for dec in declinations_deg:
        for _, ha_start, ha_end in integration_configs:
            n_steps = max(50, int((ha_end - ha_start) * 4))
            u, v = compute_uv_tracks(
                baselines, dec, ha_start, ha_end, n_steps,
                wavelength_m, latitude_deg,
            )
            uv_limit = max(
                uv_limit,
                float(np.max(np.abs(u / 1e6))),
                float(np.max(np.abs(v / 1e6))),
            )
    uv_limit *= 1.1

    n_rows = len(declinations_deg)
    n_cols = len(integration_configs)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        sharex=True, sharey=True,
        squeeze=False,
    )

    intra_mask = np.array([b.is_intra_telescope for b in baselines])

    for i, dec in enumerate(declinations_deg):
        for j, (label, ha_start, ha_end) in enumerate(integration_configs):
            n_steps = max(50, int((ha_end - ha_start) * 4))
            u, v = compute_uv_tracks(
                baselines, dec, ha_start, ha_end, n_steps,
                wavelength_m, latitude_deg,
            )

            ax = axes[i, j]
            u_inter = u[:, ~intra_mask].ravel()
            v_inter = v[:, ~intra_mask].ravel()
            u_intra = u[:, intra_mask].ravel()
            v_intra = v[:, intra_mask].ravel()

            ax.scatter(u_inter / 1e6, v_inter / 1e6, s=0.3, c="C3", alpha=0.3, rasterized=True)
            ax.scatter(-u_inter / 1e6, -v_inter / 1e6, s=0.3, c="C3", alpha=0.3, rasterized=True)
            ax.scatter(u_intra / 1e6, v_intra / 1e6, s=0.3, c="C0", alpha=0.3, rasterized=True)
            ax.scatter(-u_intra / 1e6, -v_intra / 1e6, s=0.3, c="C0", alpha=0.3, rasterized=True)

            ax.set_xlim(-uv_limit, uv_limit)
            ax.set_ylim(-uv_limit, uv_limit)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=11)
            if i == 0:
                ax.set_title(f"{label}", fontsize=14)
            if j == 0:
                ax.set_ylabel(f"Dec={int(dec)}" + r"$^\circ$" + "\n" + r"$v$ (M$\lambda$)", fontsize=13)
            if i == n_rows - 1:
                ax.set_xlabel(r"$u$ (M$\lambda$)", fontsize=13)
            ax.grid(True, alpha=0.2)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def _placeholder_figure(message: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    ax.axis("off")
    ax.text(
        0.5,
        0.55,
        "UV coverage unavailable",
        ha="center",
        va="center",
        fontsize=14,
        weight="bold",
    )
    ax.text(
        0.5,
        0.42,
        message,
        ha="center",
        va="center",
        fontsize=9,
        wrap=True,
    )
    return fig


# ---------------------------------------------------------------------------
# Registered-plot wrapper
# ---------------------------------------------------------------------------


def _extract_baselines_from_config(config) -> list[Baseline]:
    tel_stage = _find_stage(config, "telescope.fixed_aperture")
    if tel_stage is None:
        raise ValueError(
            "uv_coverage_grid requires a 'telescope.fixed_aperture' stage."
        )
    enu_explicit = _stage_extra(tel_stage, "aperture_positions_enu_m_explicit")
    if enu_explicit is None:
        raise ValueError(
            "uv_coverage_grid requires telescope.fixed_aperture."
            "aperture_positions_enu_m_explicit (per-telescope ENU positions)."
        )
    telescope_enu_m = np.asarray(enu_explicit, dtype=np.float64)

    diameter_m = float(_stage_extra(tel_stage, "diameter_m", 1.0))
    n_sub = _extract_n_sub(config)
    ao_stage = _find_stage(config, "ao.pyramid_wfs")
    pupil_template_name = _stage_extra(ao_stage, "pupil_template") if ao_stage else None
    pupil = _build_pupil_template(pupil_template_name)

    sub_positions_pupil, _d_sub = optimize_subaperture_positions(
        n_sub, diameter_m, pupil,
    )
    return enumerate_subaperture_baselines(
        telescope_enu_m=telescope_enu_m,
        subaperture_pupil_m=np.asarray(sub_positions_pupil, dtype=np.float64),
        n_subapertures_per_telescope=n_sub,
    )


@register_plot
def uv_coverage_grid(
    collection: RunCollection,
    output_dir: Path,
    formats: Iterable[str],
) -> tuple[Path, ...]:
    """Registered plot: ``uv_coverage_grid.<fmt>`` per requested format."""

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    try:
        baselines = _extract_baselines_from_config(collection.config)
    except ValueError as exc:
        fig = _placeholder_figure(str(exc))
    else:
        # Use the run's band-center wavelength for the M-lambda axes. The
        # old hardcoded Y-band 1.0 um value mis-scaled the axes by 25% for
        # the J-band kilo_keck runs and stated the wavelength nowhere
        # (review 2026-06-10 Issue 23b).
        grid = getattr(collection.config, "grid", None)
        wavelength_um = float(
            getattr(grid, "wavelength_center_um", 0.0) or 0.0
        )
        wavelength_m = (
            wavelength_um * 1.0e-6
            if wavelength_um > 0.0
            else _DEFAULT_WAVELENGTHS_M["Y"]
        )
        fig = plot_uv_coverage_grid(
            baselines,
            wavelength_m=wavelength_m,
            save_path=None,
        )
        fig.suptitle(
            f"UV coverage  (λ = {wavelength_m * 1e6:.2f} µm)",
            fontsize=10,
        )

    written: list[Path] = []
    for fmt in formats:
        out = plot_dir / f"uv_coverage_grid.{fmt}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        written.append(out)
    plt.close(fig)
    return tuple(written)
