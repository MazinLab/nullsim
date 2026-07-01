"""Temporary KILO N=6 char-curve tuning helper."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from nullsim.pipeline.context import RunContext
from nullsim.stages.characterization.detection_curve import (
    CharacterizationDetectionCurve,
)


BASE_CACHE = Path(
    ".nullsim_cache/"
    "c0bcc7cbbc44d9da76bc4986d72ca04acd78b57db686e62d3a18913686cc4882.pickle"
)
PERF_CACHE = Path(
    ".nullsim_cache/"
    "109533bb6fee3d5db415097f1c6498a29c20de6db1f47b7f8a8a3b9c8f34dd0d.pickle"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-anchors", type=int, default=6)
    parser.add_argument("--sep-min", type=float, default=0.5944725589892117)
    parser.add_argument("--sep-max", type=float, default=9.054956197195102)
    parser.add_argument("--seed", type=int, default=2060)
    parser.add_argument("--n-restarts", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--snr-proxy-weight", type=float, default=4.0)
    parser.add_argument("--throughput-weight", type=float, default=2.0)
    parser.add_argument("--bright-weight", type=float, default=0.05)
    parser.add_argument("--broadband-n-wavelengths", type=int, default=5)
    parser.add_argument("--n-workers", type=int, default=4)
    args = parser.parse_args()

    base_state = pickle.loads(BASE_CACHE.read_bytes())[0]
    perf_state = pickle.loads(PERF_CACHE.read_bytes())[0]
    params = {
        "n_modes": 24,
        "n_bright_ports": 12,
        "n_dark_ports": 11,
        "n_spectro_ports": 1,
        "wl_design_um": 1.25375,
        "null_floor": 1.0e-6,
        "n_anchors": args.n_anchors,
        "separation_min_mas": args.sep_min,
        "separation_max_mas": args.sep_max,
        "position_angle_deg": 0.0,
        "n_restarts": args.n_restarts,
        "max_iter": args.max_iter,
        "n_workers": args.n_workers,
        "seed": args.seed,
        "throughput_weight": args.throughput_weight,
        "bright_weight": args.bright_weight,
        "snr_proxy_weight": args.snr_proxy_weight,
        "tracker_observability_weight": 1.0e-2,
        "tracker_observability_min_singular_per_nm": 1.0e-3,
        "tracker_observability_crlb_weight": 1.0e-3,
        "tracker_observability_crlb_target_nm": 100.0,
        "enforce_null_constraint": True,
        "use_upstream_chip_params_start": False,
        "dac_bits": 16,
        "backend": "torch",
        "target_snr": 5.0,
        "band": "J",
        "broadband_n_wavelengths": args.broadband_n_wavelengths,
        "disk_n_rings": 3,
        "disk_n_azimuths": 4,
        "disk_kernel_order": 0,
        "disk_integral_n_radial": 21,
        "disk_integral_n_pa": 16,
        "n_dark_count_hz": 0.0,
        "n_sky_background_hz": 0.0,
        "with_fringe_tracking": True,
    }
    context = RunContext(
        rng=np.random.default_rng(0),
        config_hash="direct",
        sweep_coord_tuple=(("scene.star.jmag", 4.0),),
        instance_id="characterization_curve",
        run_dir=Path("/private/tmp"),
        source_path=Path("examples/test/kilo_keck_6.toml"),
    )
    out = CharacterizationDetectionCurve(params, context).apply(base_state)
    payload = out.results.components["characterization.detection_curve"]
    perf = perf_state.results.components["performance.standard"]

    separations = np.asarray(payload["separations_mas"], dtype=np.float64)
    char = np.asarray(payload["contrast_5sigma"], dtype=np.float64)
    det = 10 ** np.interp(
        np.log10(separations),
        np.log10(np.asarray(perf["contrast_curve_mas"], dtype=np.float64)),
        np.log10(np.asarray(perf["contrast_curve_value"], dtype=np.float64)),
    )
    ratio = det / char
    print(f"n_restarts={args.n_restarts}")
    print(f"max_iter={args.max_iter}")
    print(f"n_workers={args.n_workers}")
    print(f"snr_proxy_weight={args.snr_proxy_weight}")
    print(f"throughput_weight={args.throughput_weight}")
    print(f"bright_weight={args.bright_weight}")
    print(f"broadband_n_wavelengths={args.broadband_n_wavelengths}")
    print("separations_mas", separations)
    print("contrast_5sigma", char)
    print("snr_per_anchor", np.asarray(payload["snr_per_anchor"]))
    print("throughput", np.asarray(payload["spectro_throughput_per_anchor"]))
    print(
        "static_leak",
        np.asarray(payload["spectro_static_leak_fraction_per_anchor"]),
    )
    print(
        "full_margin",
        np.asarray(payload["full_band_constraint_margin_per_anchor"]),
    )
    print("success", np.asarray(payload["optimization_success_per_anchor"]))
    print("det_over_char", ratio)
    print("min_ratio", float(np.min(ratio)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
