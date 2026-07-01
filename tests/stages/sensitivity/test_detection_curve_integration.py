"""End-to-end smoke: chip.kernel_mzi_mesh → sensitivity → performance.

Builds a minimal SimulationState with:
- A 4-mode chip_optimization.scipy result so chip.kernel_mzi_mesh can
  build a real (M, M, n_wl) unitary stack.
- An ao_residual wavefront component with a finite piston OPD.
- A point-source scene + telescope geometry (Keck-like 86 m baseline).
- A populated detector component so performance can compute SNR.

Verifies that performance.standard's contrast curve consumes the
sensitivity-stage outputs (per-separation throughput, effective null
with disk floor, MC instability).
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from nullsim.pipeline import (
    Field,
    Geometry,
    RunContext,
    Scene,
    SimulationState,
    Throughput,
    Wavefront,
)
from nullsim.pipeline.state import Results
from nullsim.stages.chip.kernel_mzi_mesh import KernelMZIMesh
from nullsim.stages.chip_optimization import ChipOptimizationParams
from nullsim.stages.chip_optimization.scipy import ChipOptimizationScipy
from nullsim.stages.sensitivity import DetectionCurve
from nullsim.stages.performance import StandardPerformance


def _context(tmp_path: Path, instance: str) -> RunContext:
    return RunContext(
        rng=np.random.default_rng(0),
        config_hash="sensitivity-integration",
        sweep_coord_tuple=(),
        instance_id=instance,
        run_dir=tmp_path,
    )


def test_chip_sensitivity_performance_pipeline_end_to_end(tmp_path: Path) -> None:
    """Wire the three stages and confirm the contrast curve uses sensitivity throughputs."""

    n_modes = 4
    wavelengths_um = (1.20, 1.25, 1.30)
    positions = np.array(
        [
            [43.0, 0.0],
            [44.5, 0.0],
            [-43.0, 0.0],
            [-44.5, 0.0],
        ]
    )

    state = SimulationState.empty({"wavelength_grid_um": wavelengths_um})
    state = replace(
        state,
        scene=Scene(
            star_jmag=4.0,
            star_angular_diameter_mas=1.0,
            planet_contrast=1.0e-6,
            planet_separation_mas=10.0,
        ),
        geometry=Geometry(
            diameter_m=10.0,
            area_m2=math.pi * 5.0 ** 2,
            n_apertures=2,
            aperture_positions_pupil_m=positions,
        ),
        field=Field(
            amplitudes=np.ones((n_modes, len(wavelengths_um), 1), dtype=np.complex128),
            wavelength_grid_um=wavelengths_um,
        ),
        throughput=Throughput(
            wavelength_grid_um=wavelengths_um,
            transmission_fraction=tuple(0.5 for _ in wavelengths_um),
        ),
        wavefront=Wavefront(
            wavelength_grid_um=wavelengths_um,
            strehl_ratio=tuple(0.93 for _ in wavelengths_um),
            amplitude_jitter_fraction=tuple(0.0 for _ in wavelengths_um),
            components={
                "ao_residual": {
                    "piston_opd_nm": {"total": 1.35},
                },
            },
        ),
    )

    # Run a short chip optimization to populate optimal_chip_params.
    chip_opt_params = ChipOptimizationParams(
        n_modes=n_modes,
        n_bright_ports=1,
        n_dark_ports=3,
        wl_design_um=1.25,
        target_null_depth=1.0e-3,
        max_iter=100,
        n_restarts=1,
        seed=0,
    )
    chip_opt = ChipOptimizationScipy(params=chip_opt_params.model_dump(), context=_context(tmp_path, "chip_opt"))
    state = chip_opt.apply(state)

    # Apply the chip mesh; it now also persists the u_stack to results.
    chip_mesh = KernelMZIMesh(
        params={
            "n_modes": n_modes,
            "n_bright_ports": 1,
            "n_dark_ports": 3,
            "wl_design_um": 1.25,
            "optimization_source": "auto",
            "insertion_loss_db": 0.0,
        },
        context=_context(tmp_path, "chip"),
    )
    state = chip_mesh.apply(state)
    assert any(name.startswith("chip.kernel_mzi_mesh") for name in state.results.components)

    # Inject a synthetic detector component so performance has bright/dark rates.
    bright_rate = 1.0e8
    leakage_rate_total = 1.0e4
    n_dark = 3
    n_bright = 1
    port_rates = np.array(
        [bright_rate]
        + [leakage_rate_total / n_dark for _ in range(n_dark)],
        dtype=np.float64,
    )
    port_labels = ("bright_0",) + tuple(f"dark_{i}" for i in range(n_dark))
    detector_payload = {
        "port_rates": port_rates,
        "port_labels": port_labels,
        "integration_time_s": 3600.0,
        "null_depth_raw": 1.0e-4,
    }
    components = dict(state.results.components)
    components["detector.mkid"] = detector_payload
    state = replace(state, results=Results(components=components))

    # Run sensitivity. With piston_opd_nm.total=1.35 (-> ~7 mrad at J),
    # the realized null floor should be of order 1e-5 - 1e-4.
    sensitivity = DetectionCurve(
        params={
            "n_bright_ports": n_bright,
            "n_dark_ports": n_dark,
            "separations_mas": (1.0, 10.0, 100.0),
            "n_pa_samples": 36,
            "n_mc_realizations": 500,
            "mc_seed": 0,
        },
        context=_context(tmp_path, "sensitivity"),
    )
    state = sensitivity.apply(state)
    sens_payload = state.results.components["sensitivity.detection_curve"]
    assert sens_payload["realized_null_depth"] > 0.0
    assert sens_payload["null_instability"] > 0.0

    # Performance.standard should now consume the sensitivity payload.
    perf = StandardPerformance(
        params={
            "target_snr": 5.0,
            "compute_contrast_curve": True,
        },
        context=_context(tmp_path, "performance"),
    )
    state = perf.apply(state)

    perf_payload = next(
        v for k, v in state.results.components.items()
        if k.startswith("performance.standard")
    )
    # Sensitivity-derived fields should appear.
    assert perf_payload["upstream_sensitivity_component"] == "sensitivity.detection_curve"
    assert perf_payload["null_depth_effective"] == sens_payload["effective_null_with_disk"]
    # The contrast curve length must match the sensitivity separations grid.
    assert len(perf_payload["contrast_curve_value"]) == 3
    assert state.results.contrast_curve is not None
    assert state.results.contrast_curve["separation_mas"].tolist() == [1.0, 10.0, 100.0]
    # Contrasts should all be finite (the sensitivity throughputs are >0
    # at these separations for a generic chip).
    assert all(np.isfinite(state.results.contrast_curve["contrast"]))


def test_performance_falls_back_to_flat_throughput_without_sensitivity(tmp_path: Path) -> None:
    """When no sensitivity stage runs, performance uses Param flat throughput."""

    state = SimulationState.empty({"wavelength_grid_um": (1.25,)})
    state = replace(state, scene=Scene(star_jmag=4.0, planet_contrast=1.0e-6))
    components = dict(state.results.components)
    components["detector.mkid"] = {
        "port_rates": np.array([1e8, 1e4]),
        "port_labels": ("bright_0", "dark_0"),
        "integration_time_s": 3600.0,
        "null_depth_raw": 1.0e-4,
    }
    state = replace(state, results=Results(components=components))
    perf = StandardPerformance(
        params={
            "compute_contrast_curve": True,
            "contrast_curve_separations_mas": (1.0, 10.0, 100.0),
        },
        context=_context(tmp_path, "performance"),
    )
    state = perf.apply(state)
    perf_payload = next(
        v for k, v in state.results.components.items()
        if k.startswith("performance.standard")
    )
    assert perf_payload["upstream_sensitivity_component"] is None
    # Flat throughput means the contrast value is constant across separations.
    values = perf_payload["contrast_curve_value"]
    assert len(values) == 3
    assert values[0] == values[1] == values[2]
