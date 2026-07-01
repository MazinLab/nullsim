"""Estimate the capture range of the pairwise ABCD cophasing unitary.

This script isolates the large-signal ambiguity of the dedicated cophasing
mode. It uses the same connected tree graph convention as the nullsim
fringe-observability diagnostic, but strips away the science unitary and photon
noise by default so the plot answers a clean question: when do pairwise ABCD
phase measurements wrap and make the piston graph solve ambiguous?
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from nullsim.physics.fringe_observability import (
    balanced_binary_tree_edges,
    differential_piston_basis,
)

_M_TO_NM = 1.0e9


@dataclass(frozen=True)
class CaptureSweep:
    """Random common-piston-free piston capture sweep."""

    injected_rms_nm: NDArray[np.float64]
    median_residual_rms_nm: NDArray[np.float64]
    p95_residual_rms_nm: NDArray[np.float64]
    lock_fraction: NDArray[np.float64]
    edge_wrap_fraction: NDArray[np.float64]


@dataclass(frozen=True)
class SingleApertureSweep:
    """Single-aperture offset capture sweep."""

    offset_nm: NDArray[np.float64]
    input_rms_nm: NDArray[np.float64]
    residual_rms_nm: NDArray[np.float64]
    edge_wrap_fraction: NDArray[np.float64]


def wrap_opd_m(opd_m: NDArray[np.floating] | float, wavelength_m: float) -> NDArray[np.float64]:
    """Wrap OPD into the single-wavelength ABCD interval [-lambda/2, lambda/2)."""

    if wavelength_m <= 0.0:
        raise ValueError("wavelength_m must be positive.")
    opd = np.asarray(opd_m, dtype=np.float64)
    wrapped = np.mod(opd + 0.5 * float(wavelength_m), float(wavelength_m))
    return (wrapped - 0.5 * float(wavelength_m)).astype(np.float64, copy=False)


def solve_wrapped_pair_graph(
    *,
    pistons_m: NDArray[np.floating],
    edges: NDArray[np.integer],
    wavelength_m: float,
) -> NDArray[np.float64]:
    """Solve piston offsets from noiseless wrapped pairwise ABCD measurements."""

    pistons = np.asarray(pistons_m, dtype=np.float64)
    edge_arr = np.asarray(edges, dtype=np.int64)
    incidence = _incidence_matrix(edge_arr, n_modes=pistons.size)
    true_edge_opd = incidence @ pistons
    measured_edge_opd = wrap_opd_m(true_edge_opd, wavelength_m)
    solution = np.linalg.lstsq(incidence, measured_edge_opd, rcond=None)[0]
    return (solution - float(np.mean(solution))).astype(np.float64, copy=False)


def random_capture_sweep(
    *,
    n_modes: int,
    wavelength_m: float,
    injected_rms_nm: NDArray[np.floating],
    n_trials: int,
    lock_threshold_nm: float,
    seed: int,
) -> CaptureSweep:
    """Sweep random piston RMS and report wrapped-ABCD acquisition success."""

    if n_trials <= 0:
        raise ValueError("n_trials must be positive.")
    if n_modes < 2:
        raise ValueError("n_modes must be at least 2.")
    if lock_threshold_nm < 0.0:
        raise ValueError("lock_threshold_nm must be non-negative.")
    rms_nm = np.asarray(injected_rms_nm, dtype=np.float64)
    if rms_nm.ndim != 1 or rms_nm.size == 0:
        raise ValueError("injected_rms_nm must be a non-empty 1-D array.")
    if not np.all(np.isfinite(rms_nm)) or np.any(rms_nm < 0.0):
        raise ValueError("injected_rms_nm must be finite and non-negative.")

    rng = np.random.default_rng(int(seed))
    basis = differential_piston_basis(int(n_modes))
    edges = balanced_binary_tree_edges(int(n_modes))
    n_dof = basis.shape[1]
    lock_threshold_m = float(lock_threshold_nm) / _M_TO_NM
    med = np.zeros_like(rms_nm)
    p95 = np.zeros_like(rms_nm)
    lock = np.zeros_like(rms_nm)
    wrap = np.zeros_like(rms_nm)

    for index, value_nm in enumerate(rms_nm):
        target_rms_m = float(value_nm) / _M_TO_NM
        coord_sigma = target_rms_m * np.sqrt(float(n_modes) / float(n_dof))
        residuals = np.zeros(int(n_trials), dtype=np.float64)
        wrapped_edges = 0
        total_edges = 0
        for trial in range(int(n_trials)):
            coords = rng.normal(0.0, coord_sigma, size=n_dof)
            pistons = basis @ coords
            estimate = solve_wrapped_pair_graph(
                pistons_m=pistons,
                edges=edges,
                wavelength_m=float(wavelength_m),
            )
            residual = pistons - estimate
            residual -= float(np.mean(residual))
            residuals[trial] = _rms(residual)
            edge_opd = _incidence_matrix(edges, n_modes=int(n_modes)) @ pistons
            wrapped_edges += int(np.count_nonzero(np.abs(edge_opd) > 0.5 * wavelength_m))
            total_edges += int(edge_opd.size)
        med[index] = float(np.median(residuals) * _M_TO_NM)
        p95[index] = float(np.percentile(residuals, 95.0) * _M_TO_NM)
        lock[index] = float(np.mean(residuals <= lock_threshold_m))
        wrap[index] = float(wrapped_edges) / float(total_edges)

    return CaptureSweep(
        injected_rms_nm=rms_nm.astype(np.float64, copy=True),
        median_residual_rms_nm=med,
        p95_residual_rms_nm=p95,
        lock_fraction=lock,
        edge_wrap_fraction=wrap,
    )


def single_aperture_sweep(
    *,
    n_modes: int,
    wavelength_m: float,
    offset_nm: NDArray[np.floating],
    aperture_index: int = 0,
) -> SingleApertureSweep:
    """Sweep one aperture offset while all other apertures start at zero piston."""

    if n_modes < 2:
        raise ValueError("n_modes must be at least 2.")
    if aperture_index < 0 or aperture_index >= int(n_modes):
        raise ValueError("aperture_index is outside the mode range.")
    offsets_nm = np.asarray(offset_nm, dtype=np.float64)
    if offsets_nm.ndim != 1 or offsets_nm.size == 0:
        raise ValueError("offset_nm must be a non-empty 1-D array.")
    if not np.all(np.isfinite(offsets_nm)) or np.any(offsets_nm < 0.0):
        raise ValueError("offset_nm must be finite and non-negative.")

    edges = balanced_binary_tree_edges(int(n_modes))
    incidence = _incidence_matrix(edges, n_modes=int(n_modes))
    input_rms = np.zeros_like(offsets_nm)
    residual_rms = np.zeros_like(offsets_nm)
    wrap_fraction = np.zeros_like(offsets_nm)
    for index, value_nm in enumerate(offsets_nm):
        pistons = np.zeros(int(n_modes), dtype=np.float64)
        pistons[int(aperture_index)] = float(value_nm) / _M_TO_NM
        pistons -= float(np.mean(pistons))
        estimate = solve_wrapped_pair_graph(
            pistons_m=pistons,
            edges=edges,
            wavelength_m=float(wavelength_m),
        )
        residual = pistons - estimate
        residual -= float(np.mean(residual))
        edge_opd = incidence @ pistons
        input_rms[index] = _rms(pistons) * _M_TO_NM
        residual_rms[index] = _rms(residual) * _M_TO_NM
        wrap_fraction[index] = float(
            np.mean(np.abs(edge_opd) > 0.5 * float(wavelength_m))
        )
    return SingleApertureSweep(
        offset_nm=offsets_nm.astype(np.float64, copy=True),
        input_rms_nm=input_rms,
        residual_rms_nm=residual_rms,
        edge_wrap_fraction=wrap_fraction,
    )


def plot_capture_range(
    *,
    random_sweep: CaptureSweep,
    single_sweep: SingleApertureSweep,
    wavelength_m: float,
    lock_threshold_nm: float,
    output_path: Path,
) -> None:
    """Write the capture-range figure."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    half_wave_nm = 0.5 * float(wavelength_m) * _M_TO_NM
    fig = plt.figure(figsize=(12.8, 8.4))
    grid = fig.add_gridspec(2, 2)
    ax_lock = fig.add_subplot(grid[0, 0])
    ax_resid = fig.add_subplot(grid[0, 1])
    ax_wrap = fig.add_subplot(grid[1, 0])
    ax_single = fig.add_subplot(grid[1, 1])

    x = random_sweep.injected_rms_nm
    ax_lock.semilogx(x, random_sweep.lock_fraction, "o-", color="#1f77b4")
    ax_lock.axvline(half_wave_nm, color="0.35", linestyle=":", label="lambda/2")
    _mark_reference_offsets(ax_lock)
    ax_lock.set_ylim(-0.05, 1.05)
    ax_lock.set_xlabel("Injected random piston RMS (nm)")
    ax_lock.set_ylabel("Lock fraction")
    ax_lock.set_title("Random piston capture probability")
    ax_lock.grid(True, which="both", alpha=0.25)
    ax_lock.legend(loc="lower left", fontsize=8.5)

    ax_resid.loglog(x, random_sweep.median_residual_rms_nm, "o-", label="median")
    ax_resid.loglog(x, random_sweep.p95_residual_rms_nm, "s--", label="p95")
    ax_resid.loglog(x, x, color="0.5", linestyle=":", label="residual = input")
    ax_resid.axhline(lock_threshold_nm, color="#d62728", linestyle="--", label="lock threshold")
    ax_resid.axvline(half_wave_nm, color="0.35", linestyle=":")
    _mark_reference_offsets(ax_resid)
    ax_resid.set_xlabel("Injected random piston RMS (nm)")
    ax_resid.set_ylabel("Residual piston RMS (nm)")
    ax_resid.set_title("Residual after wrapped graph solve")
    ax_resid.grid(True, which="both", alpha=0.25)
    ax_resid.legend(loc="upper left", fontsize=8.5)

    ax_wrap.semilogx(x, random_sweep.edge_wrap_fraction, "o-", color="#9467bd")
    ax_wrap.axvline(half_wave_nm, color="0.35", linestyle=":")
    _mark_reference_offsets(ax_wrap)
    ax_wrap.set_ylim(-0.05, 1.05)
    ax_wrap.set_xlabel("Injected random piston RMS (nm)")
    ax_wrap.set_ylabel("Fraction of wrapped pair measurements")
    ax_wrap.set_title("Pairwise ABCD ambiguity")
    ax_wrap.grid(True, which="both", alpha=0.25)

    offset = single_sweep.offset_nm
    ax_single.loglog(offset, single_sweep.residual_rms_nm, "-", color="#2ca02c", label="residual")
    ax_single.loglog(offset, single_sweep.input_rms_nm, color="0.5", linestyle=":", label="input RMS")
    ax_single.axhline(lock_threshold_nm, color="#d62728", linestyle="--", label="lock threshold")
    ax_single.axvline(half_wave_nm, color="0.35", linestyle=":", label="lambda/2")
    _mark_reference_offsets(ax_single)
    ax_single.set_xlabel("Single-aperture piston offset (nm)")
    ax_single.set_ylabel("Common-piston-free RMS (nm)")
    ax_single.set_title("One aperture offset while others are zero")
    ax_single.grid(True, which="both", alpha=0.25)
    ax_single.legend(loc="upper left", fontsize=8.5)

    fig.suptitle(
        "Cophasing capture range for single-wavelength pairwise ABCD",
        y=0.985,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path)
    plt.close(fig)


def _incidence_matrix(edges: NDArray[np.integer], *, n_modes: int) -> NDArray[np.float64]:
    edge_arr = np.asarray(edges, dtype=np.int64)
    incidence = np.zeros((edge_arr.shape[0], int(n_modes)), dtype=np.float64)
    for row, (mode_a, mode_b) in enumerate(edge_arr):
        incidence[row, int(mode_a)] = 1.0
        incidence[row, int(mode_b)] = -1.0
    return incidence


def _rms(values: NDArray[np.floating]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(arr))))


def _mark_reference_offsets(ax: plt.Axes) -> None:
    for value, label, color in (
        (1.0e3, "1 um", "#ff7f0e"),
        (1.0e6, "1 mm", "#8c564b"),
    ):
        ax.axvline(value, color=color, linestyle="-.", linewidth=1.0, label=label)


def _nearest_index(values: NDArray[np.float64], target: float) -> int:
    return int(np.argmin(np.abs(np.asarray(values, dtype=np.float64) - float(target))))


def _print_summary(
    random_sweep: CaptureSweep,
    single_sweep: SingleApertureSweep,
    *,
    wavelength_m: float,
) -> None:
    print(f"wavelength_um={wavelength_m * 1.0e6:.4g}")
    print(f"single_edge_unambiguous_half_wave_nm={0.5 * wavelength_m * _M_TO_NM:.4g}")
    for target in (1.0e3, 1.0e6):
        i = _nearest_index(random_sweep.injected_rms_nm, target)
        j = _nearest_index(single_sweep.offset_nm, target)
        print(
            "random_rms_nm={:.4g}: lock_fraction={:.3g}, "
            "median_residual_nm={:.4g}, edge_wrap_fraction={:.3g}".format(
                random_sweep.injected_rms_nm[i],
                random_sweep.lock_fraction[i],
                random_sweep.median_residual_rms_nm[i],
                random_sweep.edge_wrap_fraction[i],
            )
        )
        print(
            "single_offset_nm={:.4g}: residual_rms_nm={:.4g}, "
            "edge_wrap_fraction={:.3g}".format(
                single_sweep.offset_nm[j],
                single_sweep.residual_rms_nm[j],
                single_sweep.edge_wrap_fraction[j],
            )
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-modes", type=int, default=24)
    parser.add_argument("--wavelength-um", type=float, default=1.25)
    parser.add_argument("--min-rms-nm", type=float, default=1.0)
    parser.add_argument("--max-rms-nm", type=float, default=1.0e6)
    parser.add_argument("--n-grid", type=int, default=61)
    parser.add_argument("--n-trials", type=int, default=2048)
    parser.add_argument("--lock-threshold-nm", type=float, default=10.0)
    parser.add_argument("--single-aperture-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cophasing_capture_range/cophasing_capture_range.png"),
        help="Output PNG/PDF/SVG path. Parent directories are created.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    wavelength_m = float(args.wavelength_um) * 1.0e-6
    grid = np.logspace(
        np.log10(float(args.min_rms_nm)),
        np.log10(float(args.max_rms_nm)),
        int(args.n_grid),
    )
    random_sweep = random_capture_sweep(
        n_modes=int(args.n_modes),
        wavelength_m=wavelength_m,
        injected_rms_nm=grid,
        n_trials=int(args.n_trials),
        lock_threshold_nm=float(args.lock_threshold_nm),
        seed=int(args.seed),
    )
    single_sweep = single_aperture_sweep(
        n_modes=int(args.n_modes),
        wavelength_m=wavelength_m,
        offset_nm=grid,
        aperture_index=int(args.single_aperture_index),
    )
    plot_capture_range(
        random_sweep=random_sweep,
        single_sweep=single_sweep,
        wavelength_m=wavelength_m,
        lock_threshold_nm=float(args.lock_threshold_nm),
        output_path=Path(args.output),
    )
    _print_summary(random_sweep, single_sweep, wavelength_m=wavelength_m)
    print(f"wrote={Path(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
