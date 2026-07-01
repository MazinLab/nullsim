"""Simulate inter-telescope cophasing by scanning a physical delay line.

This diagnostic assumes each telescope pupil has already been internally
cophasing-locked to the single-wavelength ABCD range. The remaining large
unknown is one scalar: the telescope-to-telescope optical path delay. A
broadband ABCD output summed over wavelength has a white-light fringe packet,
so a delay-line scan can recover millimeter-scale offsets without knowing the
single-wavelength fringe order in advance.
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

_M_TO_NM = 1.0e9
_M_TO_UM = 1.0e6


@dataclass(frozen=True)
class DelayEstimate:
    """One coarse-plus-fine delay-line estimate."""

    true_delay_m: float
    coarse_delay_m: float
    estimated_delay_m: float
    residual_m: float
    peak_metric: float
    coarse_command_delay_m: NDArray[np.float64]
    coarse_metric: NDArray[np.float64]
    fine_command_delay_m: NDArray[np.float64]
    fine_metric: NDArray[np.float64]


@dataclass(frozen=True)
class DelaySweep:
    """Delay-line capture sweep over injected inter-telescope offsets."""

    true_delay_m: NDArray[np.float64]
    estimated_delay_m: NDArray[np.float64]
    residual_m: NDArray[np.float64]
    peak_metric: NDArray[np.float64]
    locked_lambda_over_two: NDArray[np.bool_]
    locked_fine: NDArray[np.bool_]


def wavelength_grid_m(
    *,
    min_um: float = 1.18,
    max_um: float = 1.32,
    n_wavelengths: int = 128,
) -> NDArray[np.float64]:
    """Return a positive wavelength grid in meters."""

    if min_um <= 0.0 or max_um <= min_um:
        raise ValueError("wavelength bounds must be finite with max_um > min_um > 0.")
    if n_wavelengths < 2:
        raise ValueError("n_wavelengths must be at least 2.")
    return np.linspace(float(min_um), float(max_um), int(n_wavelengths)) * 1.0e-6


def coherence_length_m(wavelengths_m: NDArray[np.floating]) -> float:
    """Return the approximate full-band white-light packet scale."""

    wavelengths = _validate_wavelengths(wavelengths_m)
    wavenumber_span = float(1.0 / wavelengths.min() - 1.0 / wavelengths.max())
    return float("inf") if wavenumber_span <= 0.0 else 1.0 / wavenumber_span


def broadband_packet_metric(
    residual_delay_m: NDArray[np.floating] | float,
    *,
    wavelengths_m: NDArray[np.floating],
    weights: NDArray[np.floating] | None = None,
) -> NDArray[np.float64]:
    """Return normalized broadband ABCD coherence versus residual delay.

    For each commanded delay, the residual delay is ``true - commanded``. The
    ABCD quadratures summed over wavelength are proportional to
    ``sum_lambda weight_lambda * exp(i 2*pi*residual/lambda)``. The normalized
    magnitude of that vector is the fringe-packet metric scanned here.
    """

    wavelengths = _validate_wavelengths(wavelengths_m)
    wave_weights = _weights(weights, n_wavelengths=wavelengths.size)
    residual = np.asarray(residual_delay_m, dtype=np.float64)
    phase = 2.0 * np.pi * residual[..., np.newaxis] / wavelengths[np.newaxis, :]
    complex_sum = np.sum(wave_weights[np.newaxis, :] * np.exp(1j * phase), axis=-1)
    return (np.abs(complex_sum) / float(np.sum(wave_weights))).astype(
        np.float64,
        copy=False,
    )


def estimate_delay_by_scan(
    *,
    true_delay_m: float,
    wavelengths_m: NDArray[np.floating],
    scan_half_range_m: float,
    coarse_step_m: float,
    fine_half_width_m: float,
    fine_step_m: float,
    weights: NDArray[np.floating] | None = None,
) -> DelayEstimate:
    """Recover one inter-telescope delay with a coarse and fine delay scan."""

    if scan_half_range_m <= 0.0:
        raise ValueError("scan_half_range_m must be positive.")
    if coarse_step_m <= 0.0 or fine_step_m <= 0.0:
        raise ValueError("scan steps must be positive.")
    if fine_half_width_m <= 0.0:
        raise ValueError("fine_half_width_m must be positive.")

    wavelengths = _validate_wavelengths(wavelengths_m)
    command = _regular_grid(
        -float(scan_half_range_m),
        float(scan_half_range_m),
        float(coarse_step_m),
    )
    coarse_metric = scan_metric(
        true_delay_m=float(true_delay_m),
        command_delay_m=command,
        wavelengths_m=wavelengths,
        weights=weights,
    )
    coarse_delay = float(command[int(np.argmax(coarse_metric))])

    fine_min = max(-float(scan_half_range_m), coarse_delay - float(fine_half_width_m))
    fine_max = min(float(scan_half_range_m), coarse_delay + float(fine_half_width_m))
    fine_command = _regular_grid(fine_min, fine_max, float(fine_step_m))
    fine_metric = scan_metric(
        true_delay_m=float(true_delay_m),
        command_delay_m=fine_command,
        wavelengths_m=wavelengths,
        weights=weights,
    )
    estimated = _quadratic_peak(fine_command, fine_metric)
    residual = float(true_delay_m) - estimated
    return DelayEstimate(
        true_delay_m=float(true_delay_m),
        coarse_delay_m=coarse_delay,
        estimated_delay_m=estimated,
        residual_m=residual,
        peak_metric=float(np.max(fine_metric)),
        coarse_command_delay_m=command,
        coarse_metric=coarse_metric.astype(np.float64, copy=True),
        fine_command_delay_m=fine_command,
        fine_metric=fine_metric.astype(np.float64, copy=True),
    )


def scan_metric(
    *,
    true_delay_m: float,
    command_delay_m: NDArray[np.floating],
    wavelengths_m: NDArray[np.floating],
    weights: NDArray[np.floating] | None = None,
) -> NDArray[np.float64]:
    """Return the fringe-packet metric for a grid of commanded delays."""

    command = np.asarray(command_delay_m, dtype=np.float64)
    return broadband_packet_metric(
        float(true_delay_m) - command,
        wavelengths_m=wavelengths_m,
        weights=weights,
    )


def delay_capture_sweep(
    *,
    true_delay_m: NDArray[np.floating],
    wavelengths_m: NDArray[np.floating],
    scan_half_range_m: float,
    coarse_step_m: float,
    fine_half_width_m: float,
    fine_step_m: float,
    fine_lock_threshold_m: float,
    weights: NDArray[np.floating] | None = None,
) -> DelaySweep:
    """Sweep injected inter-telescope delays through the scan estimator."""

    true_delays = np.asarray(true_delay_m, dtype=np.float64)
    if true_delays.ndim != 1 or true_delays.size == 0:
        raise ValueError("true_delay_m must be a non-empty 1-D array.")
    if not np.all(np.isfinite(true_delays)):
        raise ValueError("true_delay_m must be finite.")
    if fine_lock_threshold_m < 0.0:
        raise ValueError("fine_lock_threshold_m must be non-negative.")

    wavelengths = _validate_wavelengths(wavelengths_m)
    estimated = np.zeros_like(true_delays)
    residual = np.zeros_like(true_delays)
    peak = np.zeros_like(true_delays)
    half_wave = 0.5 * float(np.mean(wavelengths))
    for index, delay in enumerate(true_delays):
        result = estimate_delay_by_scan(
            true_delay_m=float(delay),
            wavelengths_m=wavelengths,
            scan_half_range_m=float(scan_half_range_m),
            coarse_step_m=float(coarse_step_m),
            fine_half_width_m=float(fine_half_width_m),
            fine_step_m=float(fine_step_m),
            weights=weights,
        )
        estimated[index] = result.estimated_delay_m
        residual[index] = result.residual_m
        peak[index] = result.peak_metric
    return DelaySweep(
        true_delay_m=true_delays.astype(np.float64, copy=True),
        estimated_delay_m=estimated,
        residual_m=residual,
        peak_metric=peak,
        locked_lambda_over_two=np.abs(residual) < half_wave,
        locked_fine=np.abs(residual) <= float(fine_lock_threshold_m),
    )


def plot_delay_scan_diagnostic(
    *,
    wavelengths_m: NDArray[np.floating],
    examples: Sequence[DelayEstimate],
    sweep: DelaySweep,
    fine_lock_threshold_m: float,
    output_path: Path,
) -> None:
    """Write the delay-scan diagnostic figure."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wavelengths = _validate_wavelengths(wavelengths_m)
    packet_um = coherence_length_m(wavelengths) * _M_TO_UM
    half_wave_nm = 0.5 * float(np.mean(wavelengths)) * _M_TO_NM

    fig = plt.figure(figsize=(13.2, 8.8))
    grid = fig.add_gridspec(2, 2)
    ax_packet = fig.add_subplot(grid[0, 0])
    ax_zoom = fig.add_subplot(grid[0, 1])
    ax_recovery = fig.add_subplot(grid[1, 0])
    ax_residual = fig.add_subplot(grid[1, 1])

    colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")
    for index, result in enumerate(examples):
        color = colors[index % len(colors)]
        label = f"true {result.true_delay_m * _M_TO_UM:.4g} um"
        ax_packet.plot(
            result.coarse_command_delay_m * _M_TO_UM,
            result.coarse_metric,
            color=color,
            linewidth=1.2,
            label=label,
        )
        ax_packet.axvline(
            result.true_delay_m * _M_TO_UM,
            color=color,
            linestyle=":",
            linewidth=0.9,
        )
    ax_packet.set_xlabel("Commanded delay (um)")
    ax_packet.set_ylabel("Normalized broadband coherence")
    ax_packet.set_title("Coarse delay-line packet search")
    ax_packet.grid(True, alpha=0.25)
    ax_packet.legend(loc="upper left", fontsize=8.2)

    largest = max(examples, key=lambda item: abs(item.true_delay_m))
    ax_zoom.plot(
        largest.fine_command_delay_m * _M_TO_UM,
        largest.fine_metric,
        color="#2ca02c",
        linewidth=1.4,
    )
    ax_zoom.axvline(largest.true_delay_m * _M_TO_UM, color="0.25", linestyle=":", label="true")
    ax_zoom.axvline(
        largest.estimated_delay_m * _M_TO_UM,
        color="#d62728",
        linestyle="--",
        label="estimate",
    )
    ax_zoom.set_xlabel("Commanded delay (um)")
    ax_zoom.set_ylabel("Normalized broadband coherence")
    ax_zoom.set_title("Fine scan around the largest injected delay")
    ax_zoom.grid(True, alpha=0.25)
    ax_zoom.legend(loc="upper left", fontsize=8.5)

    true_um = sweep.true_delay_m * _M_TO_UM
    estimated_um = sweep.estimated_delay_m * _M_TO_UM
    ax_recovery.plot(true_um, estimated_um, ".", color="#1f77b4", markersize=4)
    lim = max(float(np.nanmax(np.abs(true_um))), float(np.nanmax(np.abs(estimated_um))), 1.0)
    ax_recovery.plot([-lim, lim], [-lim, lim], color="0.4", linestyle=":", label="estimate = true")
    ax_recovery.set_xlim(-lim, lim)
    ax_recovery.set_ylim(-lim, lim)
    ax_recovery.set_xlabel("Injected inter-telescope delay (um)")
    ax_recovery.set_ylabel("Recovered delay (um)")
    ax_recovery.set_title("Recovered delay across scan range")
    ax_recovery.grid(True, alpha=0.25)
    ax_recovery.legend(loc="upper left", fontsize=8.5)

    residual_nm = np.abs(sweep.residual_m) * _M_TO_NM
    ax_residual.semilogy(true_um, residual_nm, ".", color="#9467bd", markersize=4)
    ax_residual.axhline(half_wave_nm, color="0.35", linestyle=":", label="lambda/2")
    ax_residual.axhline(
        float(fine_lock_threshold_m) * _M_TO_NM,
        color="#d62728",
        linestyle="--",
        label="fine threshold",
    )
    ax_residual.set_xlabel("Injected inter-telescope delay (um)")
    ax_residual.set_ylabel("Absolute residual (nm)")
    ax_residual.set_title("Residual after coarse + fine scan")
    ax_residual.grid(True, which="both", alpha=0.25)
    ax_residual.legend(loc="upper left", fontsize=8.5)

    fig.suptitle(
        "Ideal inter-telescope delay scan after intra-pupil cophasing "
        f"(J band packet ~{packet_um:.2g} um)",
        y=0.985,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path)
    plt.close(fig)


def _validate_wavelengths(wavelengths_m: NDArray[np.floating]) -> NDArray[np.float64]:
    wavelengths = np.asarray(wavelengths_m, dtype=np.float64)
    if wavelengths.ndim != 1 or wavelengths.size < 2:
        raise ValueError("wavelengths_m must be a 1-D array with at least two values.")
    if not np.all(np.isfinite(wavelengths)) or np.any(wavelengths <= 0.0):
        raise ValueError("wavelengths_m must contain finite positive values.")
    return wavelengths


def _weights(
    weights: NDArray[np.floating] | None,
    *,
    n_wavelengths: int,
) -> NDArray[np.float64]:
    if weights is None:
        return np.ones(int(n_wavelengths), dtype=np.float64)
    arr = np.asarray(weights, dtype=np.float64)
    if arr.shape != (int(n_wavelengths),):
        raise ValueError("weights must match the wavelength grid.")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError("weights must be finite and non-negative.")
    if float(np.sum(arr)) <= 0.0:
        raise ValueError("weights must have positive sum.")
    return arr


def _regular_grid(start: float, stop: float, step: float) -> NDArray[np.float64]:
    if stop < start:
        raise ValueError("grid stop must be >= start.")
    n_steps = int(np.floor((float(stop) - float(start)) / float(step)))
    values = float(start) + float(step) * np.arange(n_steps + 1, dtype=np.float64)
    if values.size == 0 or values[-1] < stop:
        values = np.append(values, float(stop))
    return values.astype(np.float64, copy=False)


def _quadratic_peak(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
) -> float:
    values_x = np.asarray(x, dtype=np.float64)
    values_y = np.asarray(y, dtype=np.float64)
    index = int(np.argmax(values_y))
    if index == 0 or index == values_y.size - 1:
        return float(values_x[index])
    x3 = values_x[index - 1 : index + 2]
    y3 = values_y[index - 1 : index + 2]
    coeff = np.polyfit(x3 - values_x[index], y3, deg=2)
    curvature = float(coeff[0])
    if curvature >= 0.0:
        return float(values_x[index])
    vertex = values_x[index] - float(coeff[1]) / (2.0 * curvature)
    if x3[0] <= vertex <= x3[-1]:
        return float(vertex)
    return float(values_x[index])


def _example_delays_m() -> tuple[float, ...]:
    return (0.0, 1.0e-6, 1.0e-3)


def _print_summary(
    *,
    wavelengths_m: NDArray[np.floating],
    examples: Sequence[DelayEstimate],
    sweep: DelaySweep,
    fine_lock_threshold_m: float,
) -> None:
    print(f"wavelength_min_um={float(np.min(wavelengths_m)) * _M_TO_UM:.4g}")
    print(f"wavelength_max_um={float(np.max(wavelengths_m)) * _M_TO_UM:.4g}")
    print(f"white_light_packet_um={coherence_length_m(wavelengths_m) * _M_TO_UM:.4g}")
    print(f"fine_lock_threshold_nm={fine_lock_threshold_m * _M_TO_NM:.4g}")
    for result in examples:
        print(
            "true_delay_um={:.6g}: estimated_um={:.6g}, residual_nm={:.4g}, "
            "peak_metric={:.4g}".format(
                result.true_delay_m * _M_TO_UM,
                result.estimated_delay_m * _M_TO_UM,
                result.residual_m * _M_TO_NM,
                result.peak_metric,
            )
        )
    print(
        "sweep_lambda_over_two_lock_fraction={:.4g}".format(
            float(np.mean(sweep.locked_lambda_over_two))
        )
    )
    print(
        "sweep_fine_lock_fraction={:.4g}".format(float(np.mean(sweep.locked_fine)))
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wavelength-min-um", type=float, default=1.18)
    parser.add_argument("--wavelength-max-um", type=float, default=1.32)
    parser.add_argument("--n-wavelengths", type=int, default=128)
    parser.add_argument("--scan-half-range-um", type=float, default=1200.0)
    parser.add_argument("--coarse-step-um", type=float, default=2.0)
    parser.add_argument("--fine-half-width-um", type=float, default=8.0)
    parser.add_argument("--fine-step-nm", type=float, default=10.0)
    parser.add_argument("--fine-lock-threshold-nm", type=float, default=10.0)
    parser.add_argument("--sweep-max-delay-um", type=float, default=1000.0)
    parser.add_argument("--n-sweep", type=int, default=397)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cophasing_delay_scan/cophasing_delay_scan.png"),
        help="Output PNG/PDF/SVG path. Parent directories are created.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    wavelengths = wavelength_grid_m(
        min_um=float(args.wavelength_min_um),
        max_um=float(args.wavelength_max_um),
        n_wavelengths=int(args.n_wavelengths),
    )
    scan_half_range_m = float(args.scan_half_range_um) * 1.0e-6
    coarse_step_m = float(args.coarse_step_um) * 1.0e-6
    fine_half_width_m = float(args.fine_half_width_um) * 1.0e-6
    fine_step_m = float(args.fine_step_nm) / _M_TO_NM
    fine_lock_threshold_m = float(args.fine_lock_threshold_nm) / _M_TO_NM

    examples = tuple(
        estimate_delay_by_scan(
            true_delay_m=delay_m,
            wavelengths_m=wavelengths,
            scan_half_range_m=scan_half_range_m,
            coarse_step_m=coarse_step_m,
            fine_half_width_m=fine_half_width_m,
            fine_step_m=fine_step_m,
        )
        for delay_m in _example_delays_m()
    )
    sweep_delays = np.linspace(
        -float(args.sweep_max_delay_um) * 1.0e-6,
        float(args.sweep_max_delay_um) * 1.0e-6,
        int(args.n_sweep),
    )
    sweep = delay_capture_sweep(
        true_delay_m=sweep_delays,
        wavelengths_m=wavelengths,
        scan_half_range_m=scan_half_range_m,
        coarse_step_m=coarse_step_m,
        fine_half_width_m=fine_half_width_m,
        fine_step_m=fine_step_m,
        fine_lock_threshold_m=fine_lock_threshold_m,
    )
    output = Path(args.output)
    plot_delay_scan_diagnostic(
        wavelengths_m=wavelengths,
        examples=examples,
        sweep=sweep,
        fine_lock_threshold_m=fine_lock_threshold_m,
        output_path=output,
    )
    _print_summary(
        wavelengths_m=wavelengths,
        examples=examples,
        sweep=sweep,
        fine_lock_threshold_m=fine_lock_threshold_m,
    )
    print(f"wrote={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
