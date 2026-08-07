"""Plot high-dispersion coronagraphy cross-correlation diagnostics.

Renders the ``characterization.hdc_ccf`` payload: the channelized planet and
leaked-starlight spectra, the matched-filter template weights, the resolving
power scan, and the radial-velocity scan.

Two labelling contracts inherited from the physics reviews:

* ``self_subtraction_factor`` can legitimately exceed 1 (continuum filtering
  can sharpen as well as erode the template), so it is reported as a bare
  factor, never as a capped percentage.
* ``snr_vs_rv`` is an unbounded sigma-like matched-filter score, not a bounded
  correlation coefficient; the axis says so.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib  # noqa: F401  # backend choice managed by styles module
import numpy as np

from nullsim.outputs.styles import _ensure_headless_backend  # noqa: F401  # side-effect

from matplotlib import pyplot as plt

from nullsim.outputs import RunCollection, RunRecord
from nullsim.pipeline import register_plot

_COMPONENT_PREFIXES = ("characterization.hdc_ccf", "hdc_ccf")
_IDENTIFYING_KEYS = frozenset(
    {"ccf_snr", "ccf_channel_centers_um", "template_weights_per_channel"}
)


@register_plot
def hdc_ccf(
    collection: RunCollection,
    output_dir: Path,
    formats: Iterable[str],
) -> tuple[Path, ...]:
    """Write ``hdc_ccf`` plots for the first record carrying the payload."""

    payload = None
    matched_record = None
    for record in collection.records:
        payload = _find_payload(record)
        if payload is not None:
            matched_record = record
            break
    if payload is None:
        return ()

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    record_label = None
    if matched_record is not None and len(collection.records) > 1:
        from nullsim.outputs.plots._axis_labels import compact_record_label

        record_label = compact_record_label(
            matched_record.coord_tuple,
            matched_record.config.run.name,
        )
    written: list[Path] = []
    for file_format in formats:
        path = plot_dir / f"hdc_ccf.{file_format}"
        fig = plot_hdc_ccf(payload, record_label=record_label)
        fig.savefig(path)
        plt.close(fig)
        written.append(path)
    return tuple(written)


def plot_hdc_ccf(
    payload: dict[str, Any],
    record_label: str | None = None,
) -> plt.Figure:
    """Return the HDC cross-correlation diagnostic figure."""

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.0), constrained_layout=True)
    _plot_channel_spectra(axes[0, 0], payload)
    _plot_template_weights(axes[0, 1], payload)
    _plot_snr_vs_r(axes[1, 0], payload)
    _plot_snr_vs_rv(axes[1, 1], payload)
    suffix = f"  |  {record_label}" if record_label else ""
    fig.suptitle(_title(payload) + suffix, y=1.03)
    return fig


def _title(payload: dict[str, Any]) -> str:
    parts = [f"HDC CCF: SNR {_fmt_float(payload.get('ccf_snr'))}"]
    photon = payload.get("ccf_snr_photon_limited")
    if _finite_scalar(photon) is not None:
        parts.append(f"photon-limited {_fmt_float(photon)}")
    parts.append(f"R = {_fmt_float(payload.get('resolving_power_used'))}")
    parts.append(f"N_eff = {_fmt_float(payload.get('n_effective_lines'))}")
    factor = _finite_scalar(payload.get("self_subtraction_factor"))
    if factor is not None:
        # Bare factor on purpose: values above 1 are physical.
        parts.append(f"self-subtraction x{factor:.3g}")
    return "  |  ".join(parts)


def _find_payload(record: RunRecord) -> dict[str, Any] | None:
    for name, value in record.result.state.results.components.items():
        if not isinstance(value, dict):
            continue
        if name.startswith(_COMPONENT_PREFIXES):
            return value
        if _IDENTIFYING_KEYS.issubset(value):
            return value
    return None


def _plot_channel_spectra(ax: plt.Axes, payload: dict[str, Any]) -> None:
    centers = _array(payload.get("ccf_channel_centers_um"))
    planet = _array(payload.get("ccf_planet_rate_hz_per_channel"))
    leak = _array(payload.get("ccf_leak_rate_hz_per_channel"))
    sky = _array(payload.get("ccf_sky_rate_hz_per_channel"))
    if centers.size == 0 or planet.shape != centers.shape:
        _blank(ax, "no channelized spectra")
        return

    planet_positive = _positive(planet)
    if planet_positive.size:
        ax.semilogy(
            centers,
            _mask_nonpositive(planet),
            "-",
            color="#1f77b4",
            linewidth=1.3,
            label="planet template",
        )
    ax.set_xlabel("Channel center (um)")
    ax.set_ylabel("Planet rate (Hz / channel)")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.grid(True, which="both", alpha=0.22)

    handles, labels = ax.get_legend_handles_labels()
    if leak.shape == centers.shape and _positive(leak).size:
        ax2 = ax.twinx()
        ax2.semilogy(
            centers,
            _mask_nonpositive(leak),
            "--",
            color="#d62728",
            linewidth=1.1,
            label="leaked starlight",
        )
        if sky.shape == centers.shape and _positive(sky).size:
            ax2.semilogy(
                centers,
                _mask_nonpositive(sky),
                ":",
                color="#7f7f7f",
                linewidth=1.0,
                label="sky",
            )
        ax2.set_ylabel("Leak / sky rate (Hz / channel)")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        extra_handles, extra_labels = ax2.get_legend_handles_labels()
        handles += extra_handles
        labels += extra_labels

    used = payload.get("n_channels_used")
    total = payload.get("n_channels")
    title = "Channelized spectra"
    if used is not None and total is not None:
        title += f" ({int(used)}/{int(total)} channels used)"
    ax.set_title(title)
    if handles:
        ax.legend(handles, labels, loc="lower left", fontsize=8.0, framealpha=0.85)


def _plot_template_weights(ax: plt.Axes, payload: dict[str, Any]) -> None:
    weights = _array(payload.get("template_weights_per_channel"))
    centers = _array(payload.get("ccf_channel_centers_um"))
    if weights.size == 0:
        _blank(ax, "no template weights")
        return
    x = centers if centers.shape == weights.shape else np.arange(weights.size, dtype=float)
    x_label = (
        "Channel center (um)"
        if centers.shape == weights.shape
        else "Channel index"
    )

    width = _bar_width(x)
    ax.bar(x, np.nan_to_num(weights, nan=0.0), width=width, color="#2ca02c", alpha=0.8)
    ax.axhline(0.0, color="0.3", linewidth=0.8)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Matched-filter weight (per channel)")
    ax.grid(True, axis="y", alpha=0.22)

    # Per-channel SNR contribution: the matched filter's SNR adds in
    # quadrature across channels, so channel i contributes a fraction
    # w_i^2 / sum_j w_j^2 of the total SNR^2.
    finite = np.isfinite(weights)
    total = float(np.sum(weights[finite] ** 2))
    if total > 0.0:
        share = np.full(weights.size, np.nan, dtype=np.float64)
        share[finite] = weights[finite] ** 2 / total
        ax2 = ax.twinx()
        ax2.plot(x, share, "o-", color="#9467bd", linewidth=1.1, markersize=3.2)
        ax2.set_ylim(bottom=0.0)
        ax2.set_ylabel("Share of total SNR$^2$")
        ax2.tick_params(axis="y", labelcolor="#9467bd")
    ax.set_title("Template weights and SNR contribution")


def _plot_snr_vs_r(ax: plt.Axes, payload: dict[str, Any]) -> None:
    r_values = _array(payload.get("snr_vs_resolving_power_r"))
    snr = _array(payload.get("snr_vs_resolving_power"))
    if r_values.size == 0 or snr.shape != r_values.shape:
        _blank(ax, "no resolving-power scan")
        return
    valid = np.isfinite(r_values) & (r_values > 0.0) & np.isfinite(snr)
    if not np.any(valid):
        _blank(ax, "no feasible resolving-power points")
        return

    # NaN entries are infeasible R values on a narrow band; matplotlib skips
    # them, and every limit/annotation below is computed on the finite subset.
    ax.semilogx(
        r_values,
        np.where(np.isfinite(snr), snr, np.nan),
        "o-",
        color="#1f77b4",
        linewidth=1.5,
        markersize=4,
        label="CCF SNR",
    )
    photon = _finite_scalar(payload.get("ccf_snr_photon_limited"))
    if photon is not None:
        ax.axhline(
            photon,
            color="#ff7f0e",
            linestyle="--",
            linewidth=1.1,
            label=f"photon limit {photon:.3g}",
        )
    realized = _finite_scalar(payload.get("ccf_snr"))
    r_used = _finite_scalar(payload.get("resolving_power_used"))
    if realized is not None:
        ax.axhline(
            realized,
            color="#2ca02c",
            linestyle=":",
            linewidth=1.1,
            label=f"realized {realized:.3g}",
        )
    if r_used is not None and r_used > 0.0:
        ax.axvline(r_used, color="0.45", linestyle=":", linewidth=1.0)
        if realized is not None:
            ax.plot([r_used], [realized], "*", color="#2ca02c", markersize=11)

    ax.set_xlabel("Resolving power R")
    ax.set_ylabel("CCF SNR")
    ax.set_title("CCF SNR vs resolving power")
    ax.grid(True, which="both", alpha=0.22)
    n_masked = int(np.sum(~np.isfinite(snr)))
    if n_masked:
        ax.text(
            0.02,
            0.96,
            f"{n_masked} infeasible R point(s) omitted",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.8,
            color="0.45",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.5},
        )
    ax.legend(loc="lower right", fontsize=8.5)


def _plot_snr_vs_rv(ax: plt.Axes, payload: dict[str, Any]) -> None:
    rv_kms = _array(payload.get("snr_vs_rv_kms"))
    score = _array(payload.get("snr_vs_rv"))
    if rv_kms.size == 0 or score.shape != rv_kms.shape:
        _blank(ax, "no radial-velocity scan")
        return
    ax.plot(rv_kms, score, "-", color="#1f77b4", linewidth=1.4)
    ax.axhline(0.0, color="0.35", linewidth=0.8)
    scene_rv = _finite_scalar(payload.get("planet_radial_velocity_kms"))
    if scene_rv is not None:
        ax.axvline(
            scene_rv,
            color="#d62728",
            linestyle="--",
            linewidth=1.1,
            label=f"scene RV {scene_rv:.3g} km/s",
        )
    finite = np.isfinite(score)
    if np.any(finite):
        peak = int(np.argmax(np.where(finite, score, -np.inf)))
        ax.plot(
            [rv_kms[peak]],
            [score[peak]],
            "o",
            color="#2ca02c",
            markersize=6,
            label=f"peak {score[peak]:.3g} at {rv_kms[peak]:.3g} km/s",
        )
    ax.set_xlabel("Trial radial velocity (km/s)")
    # Unbounded matched-filter score, not a correlation coefficient.
    ax.set_ylabel("Matched-filter detection statistic (sigma)")
    ax.set_title("CCF detection statistic vs trial RV")
    ax.grid(True, alpha=0.22)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", fontsize=8.5)


def _array(values: Any) -> np.ndarray:
    if values is None:
        return np.empty(0, dtype=np.float64)
    try:
        return np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return np.empty(0, dtype=np.float64)


def _positive(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values) & (values > 0.0)]


def _mask_nonpositive(values: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(values) & (values > 0.0), values, np.nan)


def _bar_width(x: np.ndarray) -> float:
    if x.size < 2:
        return 1.0
    spacing = float(np.min(np.diff(np.sort(x))))
    return spacing * 0.9 if spacing > 0.0 else 1.0


def _blank(ax: plt.Axes, message: str) -> None:
    ax.set_axis_off()
    ax.text(0.5, 0.5, message, ha="center", va="center")


def _finite_scalar(value: Any) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if np.isfinite(resolved) else None


def _fmt_float(value: Any) -> str:
    resolved = _finite_scalar(value)
    if resolved is None:
        return "n/a"
    if resolved == 0.0:
        return "0"
    if 1.0e-2 <= abs(resolved) < 1.0e5:
        return f"{resolved:.4g}"
    return f"{resolved:.3e}"


__all__ = ["hdc_ccf", "plot_hdc_ccf"]
