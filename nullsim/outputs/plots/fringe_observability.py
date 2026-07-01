"""Plot fringe-tracker piston observability diagnostics."""

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


@register_plot
def fringe_observability(
    collection: RunCollection,
    output_dir: Path,
    formats: Iterable[str],
) -> tuple[Path, ...]:
    """Write ``fringe_observability`` plots for the first record with payload."""

    payload = None
    matched_record = None
    for record in collection.records:
        payload = _find_payload(record)
        if payload is not None:
            matched_record = record
            break
    if payload is None:
        return ()

    # Surface the realized temporal-lag piston floor (what actually enters the
    # contrast curve) so the single-frame CRLB observability on the summary
    # panel is not misread as the operative tracking floor.
    realized_floor_nm = _realized_piston_floor_nm(matched_record)
    if realized_floor_nm is not None:
        payload = {**payload, "realized_piston_floor_nm": realized_floor_nm}

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    # Identify WHICH sweep cell this single-record figure shows — the chip
    # realization is flux dependent, so an unlabeled J=4 figure is
    # ambiguous in a J sweep (review 2026-06-10 Issue 23d).
    record_label = None
    if matched_record is not None and len(collection.records) > 1:
        from nullsim.outputs.plots._axis_labels import compact_record_label

        record_label = compact_record_label(
            matched_record.coord_tuple,
            matched_record.config.run.name,
        )
    written: list[Path] = []
    for file_format in formats:
        path = plot_dir / f"fringe_observability.{file_format}"
        fig = plot_fringe_observability(payload, record_label=record_label)
        fig.savefig(path)
        plt.close(fig)
        written.append(path)
    return tuple(written)


def plot_fringe_observability(
    payload: dict[str, Any],
    record_label: str | None = None,
) -> plt.Figure:
    """Return the piston-observability diagnostic figure."""

    fig = plt.figure(figsize=(16.0, 8.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, width_ratios=(1.0, 1.0, 1.0, 0.92))
    ax_svd = fig.add_subplot(grid[0, 0])
    ax_sweep = fig.add_subplot(grid[0, 1])
    ax_mode = fig.add_subplot(grid[0, 2])
    ax_text = fig.add_subplot(grid[:, 3])
    ax_mc = fig.add_subplot(grid[1, 0])
    ax_acquisition = fig.add_subplot(grid[1, 1])
    ax_capture = fig.add_subplot(grid[1, 2])
    _plot_singular_values(ax_svd, payload)
    _plot_channel_sweep(ax_sweep, payload)
    _plot_weakest_mode(ax_mode, payload)
    _plot_monte_carlo(ax_mc, payload)
    _plot_acquisition_handoff(ax_acquisition, payload)
    _plot_summary(ax_text, payload)
    _plot_capture_sweep(ax_capture, payload)
    suffix = f"  ·  {record_label}" if record_label else ""
    fig.suptitle(f"Fringe-tracker piston observability{suffix}", y=1.02)
    return fig


def _find_payload(record: RunRecord) -> dict[str, Any] | None:
    candidates: list[tuple[str, dict[str, Any]]] = []
    for name, value in record.result.state.results.components.items():
        if not isinstance(value, dict):
            continue
        if name.startswith(
            ("fringe_observability", "diagnostics.fringe_observability")
        ):
            candidates.append((name, value))
            continue
        if {"singular_values_per_m", "weakest_piston_modes_unit_rms"}.issubset(value):
            candidates.append((name, value))
    return _preferred_fringe_payload(candidates)


def _realized_piston_floor_nm(record: RunRecord | None) -> float | None:
    """Realized one-step piston floor (nm) that enters the contrast curve.

    This is the temporal-lag-filtered piston RMS published by
    ``sensitivity.detection_curve`` (covariance-resolved, then temporal-Kalman
    selected, then scalar-resolved). It is distinct from, and typically much
    larger than, the single-frame CRLB observability shown on the same figure.
    Returns ``None`` when no detection-curve piston residual is available.
    """

    if record is None:
        return None
    for name, payload in record.result.state.results.components.items():
        if not isinstance(payload, dict):
            continue
        if not name.startswith("sensitivity.detection_curve"):
            continue
        residual = payload.get("ao_piston_residual")
        if not isinstance(residual, dict):
            continue
        # nm-valued keys from the fringe_observability residual source,
        # then the meter-valued keys from the opd_budget source —
        # kilo_keck_6 uses ao_piston_residual_source="opd_budget", and the
        # old nm-only list rendered "Realized floor: n/a" in exactly the
        # config this row was built for (review 2026-06-10 Issue 11).
        for key, scale in (
            ("fringe_observability_covariance_resolved_rms_nm", 1.0),
            ("fringe_observability_temporal_selected_rms_nm", 1.0),
            ("fringe_observability_resolved_rms_nm", 1.0),
            ("opd_budget_covariance_resolved_rms_m", 1.0e9),
            ("closed_loop_opd_budget_rms_m", 1.0e9),
        ):
            value = residual.get(key)
            if value is None:
                continue
            try:
                resolved = float(value) * scale
            except (TypeError, ValueError):
                continue
            if np.isfinite(resolved):
                return resolved
    return None


def _preferred_fringe_payload(
    candidates: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any] | None:
    if not candidates:
        return None
    for name, value in candidates:
        if name in {"fringe_observability", "diagnostics.fringe_observability"}:
            return value
    return candidates[-1][1]


def _plot_singular_values(ax: plt.Axes, payload: dict[str, Any]) -> None:
    singular = np.asarray(payload.get("singular_values_per_m", ()), dtype=np.float64)
    if singular.size == 0:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no singular values", ha="center", va="center")
        return
    x = np.arange(1, singular.size + 1)
    normalized = singular / max(float(singular[0]), np.finfo(float).tiny)
    ax.semilogy(
        x,
        normalized,
        "o-",
        color="#1f77b4",
        linewidth=1.6,
        markersize=4,
        label="piston",
    )
    amp = _payload_dict(payload, "amplitude_observability")
    amp_singular = np.asarray(
        amp.get("singular_values_per_fractional_intensity", ()),
        dtype=np.float64,
    )
    if amp_singular.size:
        amp_x = np.arange(1, amp_singular.size + 1)
        amp_norm = amp_singular / max(float(amp_singular[0]), np.finfo(float).tiny)
        ax.semilogy(
            amp_x,
            amp_norm,
            "s--",
            color="#ff7f0e",
            linewidth=1.2,
            markersize=3.4,
            label="amplitude",
        )
    tol = float(payload.get("singular_value_tolerance_per_m", 0.0))
    if tol > 0.0:
        ax.axhline(
            tol / max(float(singular[0]), np.finfo(float).tiny),
            color="#d62728",
            linestyle="--",
            linewidth=1.0,
            label="rank threshold",
        )
    n_dof = int(payload.get("n_differential_piston_modes", singular.size))
    ax.axvline(n_dof, color="0.55", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Singular mode index")
    ax.set_ylabel("Singular value / max")
    title = f"Piston rank {int(payload.get('rank', 0))}/{n_dof}"
    if amp_singular.size:
        amp_rank = int(amp.get("rank", 0))
        amp_dof = int(amp.get("n_differential_amplitude_modes", amp_singular.size))
        title += f"; amp {amp_rank}/{amp_dof}"
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    if tol > 0.0 or amp_singular.size:
        ax.legend(loc="lower left", fontsize=9)


def _plot_channel_sweep(ax: plt.Axes, payload: dict[str, Any]) -> None:
    sweep = payload.get("channel_sweep")
    if not isinstance(sweep, dict):
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no channel sweep", ha="center", va="center")
        return
    channels = np.asarray(sweep.get("n_spectral_channels", ()), dtype=np.float64)
    rank = np.asarray(sweep.get("rank", ()), dtype=np.float64)
    crlb = np.asarray(sweep.get("crlb_piston_rms_nm", ()), dtype=np.float64)
    if channels.size == 0 or rank.size != channels.size:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no channel sweep", ha="center", va="center")
        return

    n_dof = int(payload.get("n_differential_piston_modes", 0))
    ax.plot(channels, rank, "o-", color="#1f77b4", linewidth=1.6, markersize=4)
    if n_dof > 0:
        ax.axhline(n_dof, color="0.35", linestyle=":", linewidth=1.0, label="full rank")
    ax.set_xlabel("Independent spectral count channels")
    ax.set_ylabel("Rank")
    ax.set_ylim(bottom=0, top=max(n_dof, np.nanmax(rank)) * 1.08 if rank.size else 1)
    ax.grid(True, alpha=0.25)

    finite = np.isfinite(crlb) & (crlb > 0.0)
    if crlb.size == channels.size and np.any(finite):
        ax2 = ax.twinx()
        ax2.semilogy(
            channels[finite],
            crlb[finite],
            "s--",
            color="#d62728",
            linewidth=1.1,
            markersize=3.5,
            label="full-rank CRLB",
        )
        ax2.set_ylabel("Full-rank CRLB piston RMS (nm)")
        ax2.tick_params(axis="y", labelcolor="#d62728")
    ax.set_title("Rank vs spectral diversity")


def _plot_weakest_mode(ax: plt.Axes, payload: dict[str, Any]) -> None:
    modes = np.asarray(
        payload.get("weakest_piston_modes_unit_rms", ()), dtype=np.float64
    )
    if modes.ndim != 2 or modes.shape[0] == 0:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no weak-mode vectors", ha="center", va="center")
        return
    mode = modes[-1]
    positions = payload.get("positions_pupil_m")
    if positions is not None:
        pos = np.asarray(positions, dtype=np.float64)
        if pos.shape == (mode.size, 2):
            lim = float(np.nanmax(np.abs(pos))) if pos.size else 1.0
            lim = max(lim * 1.12, 1.0)
            scatter = ax.scatter(
                pos[:, 0],
                pos[:, 1],
                c=mode,
                s=95,
                cmap="coolwarm",
                edgecolor="0.25",
                linewidth=0.6,
            )
            for i, (x, y) in enumerate(pos):
                ax.text(x, y, str(i), ha="center", va="center", fontsize=7)
            fig = ax.figure
            fig.colorbar(scatter, ax=ax, label="Unit-RMS piston coefficient")
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_xlabel("Pupil x (m)")
            ax.set_ylabel("Pupil y (m)")
            ax.set_title("Weakest piston mode")
            ax.grid(True, alpha=0.2)
            return
    ax.bar(np.arange(mode.size), mode, color="#9467bd", alpha=0.85)
    ax.axhline(0.0, color="0.25", linewidth=0.8)
    ax.set_xlabel("Aperture / mode index")
    ax.set_ylabel("Unit-RMS coefficient")
    ax.set_title("Weakest piston mode")
    ax.grid(True, axis="y", alpha=0.25)


def _plot_monte_carlo(ax: plt.Axes, payload: dict[str, Any]) -> None:
    mc = _payload_dict(payload, "monte_carlo")
    residual = _finite_array(mc.get("residual_rms_nm", ()))
    nonlinear = _payload_dict(payload, "nonlinear_one_step_monte_carlo")
    nonlinear_residual = _finite_array(nonlinear.get("residual_rms_nm", ()))
    gn = _payload_dict(payload, "gauss_newton_monte_carlo")
    gn_residual = _finite_array(gn.get("residual_rms_nm", ()))
    residual_sets = (residual, nonlinear_residual, gn_residual)
    if all(values.size == 0 for values in residual_sets):
        ax.set_axis_off()
        ax.text(0.5, 0.5, "Monte Carlo disabled", ha="center", va="center")
        return
    bins, use_log_x = _histogram_bins(residual_sets)
    if residual.size:
        ax.hist(residual, bins=bins, color="#2ca02c", alpha=0.45, label="linear")
        median = float(np.median(residual))
        ax.axvline(
            median,
            color="#2ca02c",
            linestyle="-",
            linewidth=1.2,
            label=f"linear median {median:.2g} nm",
        )
    if nonlinear_residual.size:
        ax.hist(
            nonlinear_residual,
            bins=bins,
            color="#7f7f7f",
            alpha=0.33,
            label="one-step nonlinear",
        )
        median = float(np.median(nonlinear_residual))
        ax.axvline(
            median,
            color="#7f7f7f",
            linestyle=":",
            linewidth=1.1,
            label=f"one-step median {median:.2g} nm",
        )
    if gn_residual.size:
        ax.hist(
            gn_residual,
            bins=bins,
            color="#1f77b4",
            alpha=0.45,
            label="Gauss-Newton",
        )
        median = float(np.median(gn_residual))
        p95 = float(np.percentile(gn_residual, 95.0))
        ax.axvline(
            median,
            color="#1f77b4",
            linestyle="-",
            linewidth=1.2,
            label=f"GN median {median:.2g} nm",
        )
        ax.axvline(
            p95,
            color="#d62728",
            linestyle="--",
            linewidth=1.1,
            label=f"GN p95 {p95:.2g} nm",
        )
    if use_log_x:
        ax.set_xscale("log")
    ax.set_xlabel("Residual piston RMS (nm)")
    ax.set_ylabel("Trials")
    ax.set_title("Science-unitary reconstruction")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8.5)


def _plot_acquisition_handoff(ax: plt.Axes, payload: dict[str, Any]) -> None:
    active = _payload_dict(payload, "active_acquisition_monte_carlo")
    active_residual = _finite_array(active.get("residual_rms_nm", ()))
    cophasing = _payload_dict(payload, "cophasing_handoff_monte_carlo")
    acquisition_residual = _finite_array(
        cophasing.get("acquisition_residual_rms_nm", ())
    )
    science_residual = _finite_array(cophasing.get("science_residual_rms_nm", ()))
    residual_sets = (active_residual, acquisition_residual, science_residual)
    if all(values.size == 0 for values in residual_sets):
        ax.set_axis_off()
        ax.text(0.5, 0.5, "acquisition diagnostics disabled", ha="center", va="center")
        return

    bins, use_log_x = _histogram_bins(residual_sets)
    if active_residual.size:
        ax.hist(
            active_residual,
            bins=bins,
            color="#ff7f0e",
            alpha=0.42,
            label=_with_lock_fraction(
                "active acquisition", active.get("lock_fraction")
            ),
        )
        median = float(np.median(active_residual))
        p95 = float(np.percentile(active_residual, 95.0))
        ax.axvline(
            median,
            color="#ff7f0e",
            linestyle="-",
            linewidth=1.2,
            label=f"active median {median:.2g} nm",
        )
        ax.axvline(
            p95,
            color="#ff7f0e",
            linestyle="--",
            linewidth=1.0,
            label=f"active p95 {p95:.2g} nm",
        )
    if acquisition_residual.size:
        ax.hist(
            acquisition_residual,
            bins=bins,
            color="#9467bd",
            alpha=0.40,
            label=_with_lock_fraction(
                "cophasing acquisition",
                cophasing.get("acquisition_lock_fraction"),
            ),
        )
        median = float(np.median(acquisition_residual))
        p95 = float(np.percentile(acquisition_residual, 95.0))
        ax.axvline(
            median,
            color="#9467bd",
            linestyle="-",
            linewidth=1.2,
            label=f"cophasing median {median:.2g} nm",
        )
        ax.axvline(
            p95,
            color="#9467bd",
            linestyle="--",
            linewidth=1.0,
            label=f"cophasing p95 {p95:.2g} nm",
        )
    if science_residual.size:
        ax.hist(
            science_residual,
            bins=bins,
            color="#17becf",
            alpha=0.38,
            label=_with_lock_fraction(
                "science-maintenance residual",
                cophasing.get("science_lock_fraction"),
            ),
        )
        median = float(np.median(science_residual))
        p95 = float(np.percentile(science_residual, 95.0))
        ax.axvline(
            median,
            color="#17becf",
            linestyle="-",
            linewidth=1.2,
            label=f"maintenance median {median:.2g} nm",
        )
        ax.axvline(
            p95,
            color="#17becf",
            linestyle="--",
            linewidth=1.0,
            label=f"maintenance p95 {p95:.2g} nm",
        )
    threshold = _first_finite(
        payload.get("cophasing_lock_threshold_nm"),
        payload.get("active_acquisition_lock_threshold_nm"),
    )
    if threshold is not None and threshold > 0.0:
        ax.axvline(
            threshold,
            color="0.25",
            linestyle=":",
            linewidth=1.1,
            label=f"lock threshold {threshold:.2g} nm",
        )
    if use_log_x:
        ax.set_xscale("log")
    ax.set_xlabel("Residual piston RMS (nm)")
    ax.set_ylabel("Trials")
    ax.set_title("Acquisition and science maintenance")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8.0)


def _plot_capture_sweep(ax: plt.Axes, payload: dict[str, Any]) -> None:
    sweep = payload.get("gauss_newton_capture_sweep")
    if not isinstance(sweep, dict) or not bool(sweep.get("enabled", False)):
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no capture sweep", ha="center", va="center")
        return

    injected = np.asarray(sweep.get("piston_rms_nm", ()), dtype=np.float64)
    median = np.asarray(sweep.get("median_residual_rms_nm", ()), dtype=np.float64)
    p95 = np.asarray(sweep.get("p95_residual_rms_nm", ()), dtype=np.float64)
    convergence = np.asarray(sweep.get("convergence_fraction", ()), dtype=np.float64)
    valid = (
        np.isfinite(injected) & (injected > 0.0) & np.isfinite(median) & (median > 0.0)
    )
    if not np.any(valid):
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no finite capture sweep", ha="center", va="center")
        return

    x = injected[valid]
    med = median[valid]
    ax.loglog(
        x,
        med,
        "o-",
        color="#1f77b4",
        linewidth=1.5,
        markersize=4,
        label="median residual",
    )
    p95_valid = p95[valid] if p95.shape == injected.shape else np.zeros(0)
    finite_p95 = np.isfinite(p95_valid) & (p95_valid > 0.0)
    if p95_valid.size and np.any(finite_p95):
        ax.loglog(
            x[finite_p95],
            p95_valid[finite_p95],
            "s--",
            color="#d62728",
            linewidth=1.1,
            markersize=3.5,
            label="p95 residual",
        )

    reference = np.array([np.nanmin(x), np.nanmax(x)], dtype=np.float64)
    if np.all(np.isfinite(reference)) and reference[0] < reference[1]:
        ax.loglog(reference, reference, color="0.45", linestyle=":", linewidth=1.0)
        ax.text(
            0.97,
            0.08,
            "residual = injected",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.5,
            color="0.35",
        )

    ax.set_xlabel("Injected piston RMS (nm)")
    ax.set_ylabel("Residual piston RMS (nm)")
    ax.set_title("Gauss-Newton capture sweep")
    ax.grid(True, which="both", alpha=0.25)

    if convergence.shape == injected.shape:
        conv = convergence[valid]
        finite_conv = np.isfinite(conv)
        if np.any(finite_conv):
            ax2 = ax.twinx()
            ax2.plot(
                x[finite_conv],
                conv[finite_conv],
                "D-.",
                color="#9467bd",
                linewidth=1.1,
                markersize=3.5,
                label="convergence",
            )
            ax2.set_ylim(-0.05, 1.05)
            ax2.set_ylabel("Convergence fraction")
            ax2.tick_params(axis="y", labelcolor="#9467bd")

    ax.legend(loc="upper left", fontsize=8.5)


def _plot_summary(ax: plt.Axes, payload: dict[str, Any]) -> None:
    ax.set_axis_off()
    n_dof = int(payload.get("n_differential_piston_modes", 0))
    n_chip_modes = payload.get("n_chip_input_modes", payload.get("n_modes"))
    n_physical_modes = payload.get("n_physical_piston_modes", n_chip_modes)
    scalar_eta = payload.get("scalar_piston_sensing_efficiency")
    counts = np.asarray(payload.get("expected_counts_per_sample", ()), dtype=np.float64)
    mc = (
        payload.get("monte_carlo")
        if isinstance(payload.get("monte_carlo"), dict)
        else {}
    )
    nonlinear = (
        payload.get("nonlinear_one_step_monte_carlo")
        if isinstance(payload.get("nonlinear_one_step_monte_carlo"), dict)
        else {}
    )
    gauss_newton = (
        payload.get("gauss_newton_monte_carlo")
        if isinstance(payload.get("gauss_newton_monte_carlo"), dict)
        else {}
    )
    active = (
        payload.get("active_acquisition_monte_carlo")
        if isinstance(payload.get("active_acquisition_monte_carlo"), dict)
        else {}
    )
    cophasing = (
        payload.get("cophasing_handoff_monte_carlo")
        if isinstance(payload.get("cophasing_handoff_monte_carlo"), dict)
        else {}
    )
    scintillation = _payload_dict(payload, "scintillation_assumptions")
    amp_post = _payload_dict(payload, "amplitude_postprocessing_monte_carlo")
    amplitude = _payload_dict(payload, "amplitude_observability")
    amplitude_counts = np.asarray(
        amplitude.get("expected_signal_counts_per_sample", ()),
        dtype=np.float64,
    )
    estimator_port_set = str(amplitude.get("estimator_port_set", "bright"))
    estimator_label = {
        "bright": "Bright",
        "bright_dark": "Bright+dark",
        "all": "All-port",
    }.get(estimator_port_set, estimator_port_set)
    rows: list[tuple[str, str | None]] = [
        ("Observability", None),
        ("Bright ports", str(payload.get("n_bright_ports", "?"))),
        ("Spectral channels", str(payload.get("n_spectral_channels_used", "?"))),
        ("Chip input modes", str(n_chip_modes)),
        ("Physical piston modes", str(n_physical_modes)),
        ("Differential piston modes", str(n_dof)),
        (
            "Observable fraction",
            f"{float(payload.get('observable_fraction', 0.0)):.3f}",
        ),
        ("Condition number", _fmt_float(payload.get("condition_number"))),
        ("Truncated condition", _fmt_float(payload.get("truncated_condition_number"))),
        (
            "CRLB RMS (1 frame)",
            _fmt_float(payload.get("crlb_piston_rms_nm")) + " nm",
        ),
        (
            "Trunc. CRLB (1 frame)",
            _fmt_float(payload.get("crlb_truncated_piston_rms_nm")) + " nm",
        ),
        (
            "Realized floor (temporal)",
            (
                "n/a"
                if payload.get("realized_piston_floor_nm") is None
                else _fmt_float(payload.get("realized_piston_floor_nm")) + " nm"
            ),
        ),
        (
            "Scalar eta in FT model",
            "n/a" if scalar_eta is None else f"{float(scalar_eta):.3f}",
        ),
        ("Amplitude", None),
        (
            "Amp. estimator",
            f"{estimator_label} ({amplitude.get('n_estimator_ports', '?')})",
        ),
        ("Amp. rank", _rank_text(amplitude)),
        (
            "Amp. condition",
            _fmt_float(amplitude.get("condition_number")),
        ),
        (
            "Amp. CRLB",
            _fmt_float(amplitude.get("crlb_fractional_intensity_percent")) + "%",
        ),
        (
            "Amp. trunc. CRLB",
            _fmt_float(
                amplitude.get("crlb_truncated_fractional_intensity_percent")
            )
            + "%",
        ),
        (
            "Amp. frame time",
            (
                "n/a"
                if not amplitude
                else f"{float(amplitude.get('frame_time_s', 0.0)):.3e} s"
            ),
        ),
        (
            "Amp. med. counts",
            (
                "n/a"
                if amplitude_counts.size == 0
                else _fmt_float(np.median(amplitude_counts))
            ),
        ),
        ("Scintillation", None),
        (
            "Injected sigma_I",
            _fmt_percent(scintillation.get("scintillation_sigma_i_injected")),
        ),
        ("Open-loop sigma_I", _fmt_percent(scintillation.get("sigma_I_open"))),
        (
            "No-pred. sigma_I",
            _fmt_percent(scintillation.get("sigma_I_no_prediction_same_bandwidth")),
        ),
        (
            "Predictive gain",
            _fmt_float(scintillation.get("predictive_gain_variance")),
        ),
        ("Amp. post-processing", None),
        (
            "No correction",
            _fmt_percent(amp_post.get("median_input_rms_percent"), already=True),
        ),
        (
            f"{estimator_label} linear",
            _fmt_percent(
                amp_post.get(
                    "median_linear_residual_rms_percent",
                    amp_post.get("median_bright_linear_residual_rms_percent"),
                ),
                already=True,
            ),
        ),
        (
            f"{estimator_label} Wiener",
            _fmt_percent(
                amp_post.get(
                    "median_wiener_residual_rms_percent",
                    amp_post.get("median_bright_wiener_residual_rms_percent"),
                ),
                already=True,
            ),
        ),
        (
            "Ideal taps",
            _fmt_percent(
                amp_post.get("median_ideal_tap_residual_rms_percent"),
                already=True,
            ),
        ),
        ("Frame time", f"{float(payload.get('frame_time_s', 0.0)):.3e} s"),
        (
            "Median frame counts",
            "n/a" if counts.size == 0 else _fmt_float(np.median(counts)),
        ),
    ]
    if _payload_has_finite_array(mc, "residual_rms_nm"):
        rows.extend(
            [
                ("Science-unitary MC", None),
                (
                    "Linear MC median",
                    _fmt_float(mc.get("median_residual_rms_nm")) + " nm",
                ),
                ("Linear MC p95", _fmt_float(mc.get("p95_residual_rms_nm")) + " nm"),
            ]
        )
    if _payload_has_finite_array(nonlinear, "residual_rms_nm"):
        rows.append(
            (
                "Nonlinear one-step",
                _fmt_float(nonlinear.get("median_residual_rms_nm")) + " nm",
            )
        )
    if _payload_has_finite_array(gauss_newton, "residual_rms_nm"):
        if not any(label == "Science-unitary MC" for label, _value in rows):
            rows.append(("Science-unitary MC", None))
        rows.extend(
            [
                (
                    "Gauss-Newton median",
                    _fmt_float(gauss_newton.get("median_residual_rms_nm")) + " nm",
                ),
                (
                    "Gauss-Newton conv.",
                    _fmt_float(gauss_newton.get("convergence_fraction")),
                ),
            ]
        )
    if _payload_has_finite_array(active, "residual_rms_nm") or _payload_has_finite_array(
        cophasing,
        "science_residual_rms_nm",
    ):
        rows.append(("Acquisition", None))
    if _payload_has_finite_array(active, "residual_rms_nm"):
        rows.extend(
            [
                (
                    "Active acq. median",
                    _fmt_float(active.get("median_residual_rms_nm")) + " nm",
                ),
                (
                    "Active acq. p95",
                    _fmt_float(active.get("p95_residual_rms_nm")) + " nm",
                ),
                ("Active lock frac.", _fmt_float(active.get("lock_fraction"))),
            ]
        )
    if _payload_has_finite_array(cophasing, "science_residual_rms_nm"):
        rows.extend(
            [
                (
                    "Cophasing acq.",
                    _fmt_float(cophasing.get("median_acquisition_residual_rms_nm"))
                    + " nm",
                ),
                (
                    "Science input",
                    _fmt_float(cophasing.get("median_science_input_rms_nm")) + " nm",
                ),
                (
                    "Science maint.",
                    _fmt_float(cophasing.get("median_science_residual_rms_nm"))
                    + " nm",
                ),
                (
                    "Maint. lock frac.",
                    _fmt_float(cophasing.get("science_lock_fraction")),
                ),
                (
                    "ABCD wrap frac.",
                    _fmt_float(cophasing.get("edge_wrap_fraction")),
                ),
            ]
        )
    y = 0.98
    row_step = min(0.030, max(0.021, 0.86 / max(len(rows), 1)))
    section_step = 0.95 * row_step
    row_font = 7.7 if row_step < 0.026 else 8.0
    section_font = 8.1 if row_step < 0.026 else 8.4
    ax.text(
        0.02,
        y,
        "Summary",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    y -= 0.07
    for label, value in rows:
        if value is None:
            ax.text(
                0.02,
                y,
                label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=section_font,
                fontweight="bold",
                color="0.18",
            )
            y -= section_step
            continue
        ax.text(
            0.02,
            y,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=row_font,
            color="0.35",
        )
        ax.text(
            0.72,
            y,
            value,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=row_font,
            family="monospace",
        )
        y -= row_step

    source_parts = [
        str(payload.get("frame_time_source", "")),
        str(payload.get("count_source", "")),
    ]
    source = " - ".join(part for part in source_parts if part)
    if source:
        ax.text(
            0.02,
            0.01,
            source,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.8,
            color="0.55",
            wrap=True,
        )


def _payload_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _payload_has_finite_array(payload: dict[str, Any], key: str) -> bool:
    if not payload:
        return False
    try:
        values = np.asarray(payload.get(key, ()), dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.size and np.any(np.isfinite(values)))


def _rank_text(payload: dict[str, Any]) -> str:
    if not payload:
        return "n/a"
    rank = payload.get("rank")
    n_modes = payload.get("n_differential_amplitude_modes")
    if rank is None or n_modes is None:
        return "n/a"
    return f"{int(rank)}/{int(n_modes)}"


def _finite_array(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def _histogram_bins(values: Iterable[np.ndarray]) -> tuple[np.ndarray | int, bool]:
    finite = np.concatenate([value for value in values if value.size])
    if finite.size == 0:
        return 8, False
    n_bins = min(30, max(8, int(finite.size // 4)))
    positive = finite[finite > 0.0]
    if positive.size == finite.size:
        min_positive = float(np.nanmin(positive))
        max_positive = float(np.nanmax(positive))
        if min_positive > 0.0 and max_positive / min_positive > 50.0:
            return np.geomspace(min_positive, max_positive, n_bins + 1), True
    return n_bins, False


def _with_lock_fraction(label: str, value: Any) -> str:
    try:
        lock_fraction = float(value)
    except (TypeError, ValueError):
        return label
    if not np.isfinite(lock_fraction):
        return label
    return f"{label} ({lock_fraction:.0%} lock)"


def _first_finite(*values: Any) -> float | None:
    for value in values:
        try:
            finite = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(finite):
            return finite
    return None


def _fmt_float(value: Any) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(val):
        if np.isnan(val):
            return "nan"
        return "inf" if val > 0.0 else "-inf"
    if val == 0.0:
        return "0"
    if 1.0e-2 <= abs(val) < 1.0e4:
        return f"{val:.3g}"
    return f"{val:.3e}"


def _fmt_percent(value: Any, *, already: bool = False) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(val):
        return _fmt_float(val)
    if not already:
        val *= 100.0
    return f"{val:.3g}%"


__all__ = ["fringe_observability", "plot_fringe_observability"]
