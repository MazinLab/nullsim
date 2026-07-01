"""Study a duplicated-input quadrature mesh fringe-tracker concept.

This is an intentionally reduced model: the 48x48 Clements mesh is represented
by its action on the 24 physical aperture fields. The tracker half carries the
stellar reference and a tunable fraction of differential piston modes; the
science half is the lossless complement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from nullsim.physics.quadrature_fringe_tracker import (
    evaluate_quadrature_reference_observability,
)

DEFAULT_TOTAL_BAND_COUNTS_PER_FRAME = 2.08e7
DEFAULT_CURRENT_CRLB_NM = 1.96e3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-modes", type=int, default=24)
    parser.add_argument(
        "--n-lambda",
        type=int,
        nargs="+",
        default=(1, 2, 3, 5, 10),
        help="Independent tracker spectral channels across the observing band.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        nargs="+",
        default=(0.003, 0.01, 0.03, 0.1, 0.3),
        help="Differential-mode power fraction routed to tracker outputs.",
    )
    parser.add_argument("--lambda-min-um", type=float, default=1.18)
    parser.add_argument("--lambda-max-um", type=float, default=1.32)
    parser.add_argument(
        "--total-band-counts-per-frame",
        type=float,
        default=DEFAULT_TOTAL_BAND_COUNTS_PER_FRAME,
        help=(
            "Total on-axis stellar counts entering the mesh per fringe-tracker "
            "sample across the whole observing band. Default is the KILO-scale "
            "count budget inferred from the latest 5-port diagnostic."
        ),
    )
    parser.add_argument(
        "--current-crlb-nm",
        type=float,
        default=DEFAULT_CURRENT_CRLB_NM,
        help="Reference CRLB from the current 5-bright-port diagnostic.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/fringe_tracker_quadrature_mesh"),
    )
    args = parser.parse_args()

    rows = _run_study(args)
    _write_outputs(rows, args)


def _run_study(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n_lambda in args.n_lambda:
        wavelengths_m = _wavelength_grid_m(
            n_lambda=int(n_lambda),
            lambda_min_um=float(args.lambda_min_um),
            lambda_max_um=float(args.lambda_max_um),
        )
        counts_per_channel = (
            float(args.total_band_counts_per_frame) / float(n_lambda)
        )
        for beta in args.beta:
            result = evaluate_quadrature_reference_observability(
                n_modes=int(args.n_modes),
                wavelengths_m=wavelengths_m,
                stellar_counts_per_channel=np.full(
                    int(n_lambda),
                    counts_per_channel,
                    dtype=np.float64,
                ),
                differential_coupling=float(beta),
            )
            rows.append(
                {
                    "n_modes": int(args.n_modes),
                    "n_lambda": int(n_lambda),
                    "differential_coupling_beta": float(beta),
                    "tracker_channels": int(result.tracker_channel_count),
                    "science_channels": int(result.science_channel_count),
                    "rank": int(result.rank),
                    "condition_number": float(result.condition_number),
                    "crlb_piston_rms_nm": float(result.crlb_piston_rms_nm),
                    "tracker_star_fraction": float(result.tracker_star_fraction),
                    "science_star_leakage": float(result.science_star_leakage),
                    "tracker_differential_throughput": float(
                        result.tracker_differential_throughput,
                    ),
                    "science_differential_throughput": float(
                        result.science_differential_throughput,
                    ),
                    "median_counts_per_tracker_channel": float(
                        np.median(result.expected_counts_per_tracker_channel),
                    ),
                },
            )
    return rows


def _write_outputs(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    plot_dir = Path(args.out_dir) / "plots"
    table_dir = Path(args.out_dir) / "tables"
    plot_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    json_path = table_dir / "quadrature_mesh_study.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fig = _plot_rows(rows, current_crlb_nm=float(args.current_crlb_nm))
    png_path = plot_dir / "quadrature_mesh_study.png"
    pdf_path = plot_dir / "quadrature_mesh_study.pdf"
    fig.savefig(png_path, bbox_inches="tight", facecolor="white", dpi=180)
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print(f"wrote {json_path}")
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")
    print(_summary(rows))


def _plot_rows(rows: list[dict[str, Any]], *, current_crlb_nm: float):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax = axes[0]
    n_lambda_values = sorted({int(row["n_lambda"]) for row in rows})
    for n_lambda in n_lambda_values:
        sub = [row for row in rows if int(row["n_lambda"]) == n_lambda]
        sub.sort(key=lambda row: float(row["differential_coupling_beta"]))
        beta = np.array([float(row["differential_coupling_beta"]) for row in sub])
        crlb = np.array([float(row["crlb_piston_rms_nm"]) for row in sub])
        channels = int(sub[0]["tracker_channels"])
        ax.loglog(beta, crlb, marker="o", label=f"{n_lambda} bins ({channels} SNSPD)")
    ax.axhline(current_crlb_nm, color="0.35", linestyle="--", linewidth=1.2)
    ax.text(
        0.0032,
        current_crlb_nm * 1.08,
        "current 5-port CRLB",
        fontsize=8.5,
        color="0.35",
    )
    ax.axhline(10.0, color="#c53030", linestyle=":", linewidth=1.2)
    ax.text(0.0032, 11.0, "10 nm", fontsize=8.5, color="#c53030")
    ax.text(
        0.98,
        0.04,
        "fixed total band counts:\nextra spectral bins mostly add SNSPDs",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.2,
        color="0.35",
    )
    ax.set_xlabel("tracker differential coupling beta")
    ax.set_ylabel("piston CRLB RMS (nm)")
    ax.set_title("Linear piston information")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    ax2 = axes[1]
    beta_values = sorted({float(row["differential_coupling_beta"]) for row in rows})
    science = np.array([1.0 - beta for beta in beta_values])
    tracker = np.array(beta_values)
    ax2.semilogx(beta_values, science, marker="o", label="science differential")
    ax2.semilogx(beta_values, tracker, marker="s", label="tracker differential")
    ax2.axhline(1.0, color="0.6", linewidth=0.8)
    ax2.set_ylim(0.0, 1.04)
    ax2.set_xlabel("tracker differential coupling beta")
    ax2.set_ylabel("power fraction")
    ax2.set_title("Lossless throughput split")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(fontsize=8)

    fig.suptitle("Duplicated-input quadrature mesh fringe tracker")
    return fig


def _summary(rows: list[dict[str, Any]]) -> str:
    candidates = [
        row
        for row in rows
        if float(row["crlb_piston_rms_nm"]) <= 10.0
        and float(row["science_differential_throughput"]) >= 0.9
    ]
    if not candidates:
        best = min(rows, key=lambda row: float(row["crlb_piston_rms_nm"]))
        return (
            "best: "
            f"n_lambda={best['n_lambda']}, beta={best['differential_coupling_beta']}, "
            f"CRLB={best['crlb_piston_rms_nm']:.3g} nm"
        )
    best = min(candidates, key=lambda row: int(row["tracker_channels"]))
    return (
        "lowest-channel <=10 nm and >=90% science-differential throughput: "
        f"n_lambda={best['n_lambda']}, beta={best['differential_coupling_beta']}, "
        f"channels={best['tracker_channels']}, "
        f"CRLB={best['crlb_piston_rms_nm']:.3g} nm"
    )


def _wavelength_grid_m(
    *,
    n_lambda: int,
    lambda_min_um: float,
    lambda_max_um: float,
) -> np.ndarray:
    if n_lambda <= 0:
        raise ValueError("n_lambda must be positive.")
    if n_lambda == 1:
        return np.array([(lambda_min_um + lambda_max_um) * 0.5e-6], dtype=np.float64)
    return np.linspace(lambda_min_um, lambda_max_um, n_lambda, dtype=np.float64) * 1e-6


if __name__ == "__main__":
    main()
