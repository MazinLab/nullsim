"""Smoke tests for the per-stage debug_report PDF."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

import nullsim  # noqa: F401
from nullsim.config import ResolvedConfig
from nullsim.outputs import RunCollection, RunRecord
from nullsim.outputs.plots.debug_report import (
    _component_label,
    _human_time,
    _result_item_rank,
    _result_value_repr,
    _sweep_summary,
    debug_report,
)
from nullsim.pipeline import RunResult
from nullsim.pipeline.runner import StageTiming
from nullsim.pipeline.state import (
    Field,
    OPDBudget,
    OPDComponent,
    PhotonRates,
    Results,
    SimulationState,
    Throughput,
    Wavefront,
)


def _synthetic_state() -> SimulationState:
    wl_grid = (1.20, 1.25, 1.30)
    state = SimulationState.empty(grid={"wavelength_grid_um": wl_grid})

    # Throughput: three components composing multiplicatively.
    throughput = Throughput(
        wavelength_grid_um=wl_grid,
        transmission_fraction=(0.5, 0.6, 0.55),
        components={
            "telescope": {
                "wavelength_grid_um": wl_grid,
                "transmission_fraction": (0.9, 0.9, 0.9),
            },
            "injection": {
                "wavelength_grid_um": wl_grid,
                "transmission_fraction": (0.8, 0.85, 0.82),
            },
            "transport.smf28": {
                "wavelength_grid_um": wl_grid,
                "transmission_fraction": (0.7, 0.78, 0.75),
            },
        },
    )

    # OPD: two active + one superseded.
    opd_budget = OPDBudget(
        total_rms_opd_nm=120.0,
        components={
            "atmosphere": OPDComponent(rms_opd_nm=400.0, superseded=True),
            "ao_pyramid": OPDComponent(rms_opd_nm=95.0, superseded=False),
            "transport_phase": OPDComponent(rms_opd_nm=12.0, superseded=False),
        },
    )

    wavefront = Wavefront(
        wavelength_grid_um=wl_grid,
        strehl_ratio=(0.80, 0.82, 0.85),
        components={
            "atmosphere": {
                "strehl_ratio": (0.40, 0.42, 0.45),
                "amplitude_jitter_fraction": (0.02, 0.02, 0.02),
            },
            "ao_pyramid": {
                "strehl_ratio": (0.80, 0.82, 0.85),
                "amplitude_jitter_fraction": (0.02, 0.02, 0.02),
            },
            "injection": {
                "strehl_ratio": (1.0, 1.0, 1.0),
            },
        },
    )

    rates = np.ones((4, 3, 1), dtype=np.float64) * 1e6
    rates[0, :, 0] *= 1000  # bright port hotter
    photon_rates = PhotonRates(
        wavelength_grid_um=wl_grid,
        port_labels=("bright_0", "dark_0", "dark_1", "dark_2"),
        rates_hz=rates,
    )

    results = Results(components={
        "chip_optimization_chip_opt": {
            "optimal_chip_params": (0.1, 0.2, 0.3),
            "optimization_loss_final": 3.2e-7,
            "optimization_iterations": 145,
            "backend_name": "scipy",
            "kernel_rescales_per_wavelength": True,
        },
        "detector_mkid": {
            "null_depth_raw": 4.5e-7,
            "saturated_ports": [False, False, False, False],
        },
        "calibration": {
            "null_depth_raw": 4.5e-7,
            "null_depth_calibrated": 0.0,
            "calibration_floor": 1.0e-6,
        },
        "performance": {
            "snr": 18.7,
            "integration_time_s": 3600.0,
        },
    })

    field = Field(
        amplitudes=np.ones((4, 3, 1), dtype=np.complex128) / 2,
        wavelength_grid_um=wl_grid,
    )

    return replace(
        state,
        field=field,
        throughput=throughput,
        opd_budget=opd_budget,
        wavefront=wavefront,
        photon_rates=photon_rates,
        results=results,
    )


def test_human_time_handles_nonfinite_values() -> None:
    assert _human_time(float("nan")) == "n/a"
    assert _human_time(float("inf")) == "n/a"


def test_result_value_repr_summarizes_arrays_for_report_tables() -> None:
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    text = _result_value_repr("planet_throughputs", values)

    assert "shape=(3, 4)" in text
    assert "min=0.000e+00" in text
    assert "max=1.100e+01" in text
    assert "array([[" not in text


def test_result_rows_prioritize_scalar_health_metrics() -> None:
    rows = [
        ("planet_throughputs", np.arange(70, dtype=np.float64)),
        ("realized_null_depth", 5.0e-5),
        ("optimal_u_stack", np.ones((24, 24, 20), dtype=np.complex128)),
        ("snr", 18.6),
    ]

    ordered = [key for key, _ in sorted(rows, key=_result_item_rank)]

    assert ordered[:2] == ["realized_null_depth", "snr"]
    assert ordered[-1] == "optimal_u_stack"


def test_component_label_shortens_long_pipeline_identifiers() -> None:
    assert (
        _component_label("fringe_tracking.closed_loop_residual")
        == "fringe_tracking_residual"
    )
    assert len(_component_label("very.long.component.name.with.many.sections")) <= 32


def _synthetic_resolved_config() -> ResolvedConfig:
    raw = {
        "run": {"name": "debug_test", "seed": 1},
        "grid": {
            "wavelength_center_um": 1.25,
            "wavelength_bandwidth_um": 0.10,
            "n_wavelength_bins": 3,
        },
        "scene": {
            "star": {"jmag": 5.0, "angular_diameter_mas": 1.0, "distance_pc": 10.0},
            "planet": {
                "contrast": 1e-6, "separation_mas": 5.0,
                "position_angle_deg": 0.0,
            },
        },
        "observation": {"integration_time_s": 3600.0},
        "pipeline": {
            "stages": [
                {"id": "telescope", "type": "telescope.fixed_aperture"},
                {"id": "atmosphere", "type": "atmosphere.kolmogorov"},
                {"id": "ao_pyramid", "type": "ao.pyramid_wfs"},
                {"id": "injection", "type": "injection.ideal"},
                {"id": "transport.smf28", "type": "transport.smf28"},
                {"id": "chip_opt", "type": "chip_optimization.scipy"},
                {"id": "chip", "type": "chip.kernel_mzi_mesh", "insertion_loss_db": 0.0},
                {"id": "detector_mkid", "type": "detector.mkid"},
                {"id": "calibration", "type": "calibration.floor_lookup"},
                {"id": "performance", "type": "performance.standard"},
            ],
        },
        "output": {"dir": "/tmp", "plots": ["debug_report"], "tables": []},
    }
    return ResolvedConfig.model_validate(
        {**raw, "raw_config": raw, "source_path": None,
         "config_hash": "abc123", "run_hash": "def456",
         "sweep_coord_tuple": ()},
    )


def test_debug_report_writes_multipage_pdf(tmp_path: Path) -> None:
    config = _synthetic_resolved_config()
    state = _synthetic_state()
    timings = tuple(
        StageTiming(instance_id=s.id, elapsed_s=0.5, cache_hit=False)
        for s in config.pipeline.stages
    )
    record = RunRecord(
        config=config,
        result=RunResult(
            state=state, diagnostics={}, timings=timings,
            cache_hit_count=0, config_hash="abc123",
        ),
        run_dir=tmp_path,
        coord_tuple=(),
    )
    collection = RunCollection(
        config=config, records=(record,), output_dir=tmp_path,
    )

    written = debug_report(collection, tmp_path, formats=("pdf",))
    assert len(written) == 1
    out = written[0]
    assert out.exists()
    assert out.stat().st_size > 5000

    # The PDF should be multi-page: summary + N stage pages + 3 summary pages.
    # A single-page PDF is ~2-3 KB; >10 pages worth of content easily exceeds 30 KB.
    assert out.stat().st_size > 30 * 1024


def test_debug_report_empty_collection_returns_empty(tmp_path: Path) -> None:
    config = _synthetic_resolved_config()
    collection = RunCollection(config=config, records=(), output_dir=tmp_path)
    written = debug_report(collection, tmp_path, formats=("pdf",))
    assert written == ()


def test_sweep_summary_prefers_final_science_null(tmp_path: Path) -> None:
    """Detector raw null is diagnostic; the report headline uses science null."""

    config = _synthetic_resolved_config()
    state = _synthetic_state()
    sensitivity = {
        "ideal_chip_null_depth": 9.72e-6,
        "realized_null_depth": 5.75e-5,
        "disk_null_floor": 2.37e-5,
        "effective_null_with_disk": 8.12e-5,
    }
    calibration = {
        "null_depth_raw": 7.16e-4,
        "null_depth_residual": 7.157e-4,
        "calibration_floor": 1.20e-7,
    }
    performance = {
        "null_depth_effective": 8.12e-5,
        "snr": 47.84,
        "integration_time_for_target_snr_s": 315.0,
        "target_snr": 5.0,
    }
    state = replace(
        state,
        results=Results(
            null_depth=8.12e-5,
            components={
                **state.results.components,
                "sensitivity": sensitivity,
                "calibration": calibration,
                "performance": performance,
            },
        ),
    )
    record = RunRecord(
        config=config,
        result=RunResult(
            state=state, diagnostics={}, timings=(),
            cache_hit_count=0, config_hash="abc123",
        ),
        run_dir=tmp_path,
        coord_tuple=(("scene.star.jmag", 3.0),),
    )

    summary = _sweep_summary([record])

    assert summary["final_null"] == [8.12e-5]


def _synthetic_state_with_sweep_metrics(jmag: float, chip_ideal: float) -> SimulationState:
    """Synthetic state with the result components the sweep PDF reads."""

    state = _synthetic_state()
    sens_payload = {
        "ideal_chip_null_depth": chip_ideal,
        "realized_null_depth": chip_ideal * 1.05,
        "disk_null_floor": 3.0e-3,
        "effective_null_with_disk": chip_ideal * 1.05 + 3.0e-3,
        "ao_piston_opd_rms_per_sub_m": 1.3e-9 + 0.1e-9 * jmag,
        "separations_mas": (3.0, 10.0, 30.0, 100.0, 300.0),
        "planet_throughputs": (1.0e-3, 5.0e-3, 1.0e-2, 1.0e-2, 5.0e-3),
    }
    calib_payload = {
        "null_depth_raw": chip_ideal * 1.05 + 3.0e-3,
        "calibration_floor": 1.0e-6,
        "null_depth_calibrated": chip_ideal * 1.05 + 3.0e-3 - 1.0e-6,
    }
    perf_payload = {
        "snr": max(0.1, 30.0 / max(jmag - 2.0, 1.0)),
        "integration_time_for_target_snr_s": 3600.0 * (jmag - 2.0) ** 2,
        "target_snr": 5.0,
        "contrast_curve_mas": (3.0, 10.0, 30.0, 100.0, 300.0),
        "contrast_curve_value": (
            5.0e-5 * (jmag / 4.0) ** 2,
            1.0e-6 * (jmag / 4.0) ** 2,
            1.0e-7 * (jmag / 4.0) ** 2,
            5.0e-7 * (jmag / 4.0) ** 2,
            1.0e-4 * (jmag / 4.0) ** 2,
        ),
    }
    results = Results(
        components={
            **state.results.components,
            "sensitivity": sens_payload,
            "calibration": calib_payload,
            "performance": perf_payload,
            "fringe_tracking.closed_loop": {
                "mode": "full_piston_servo_residual",
                "sigma2_lag_reduced_rad2": 1.0e-4,
                "sigma2_photon_rad2": 5.0e-5,
                "sigma2_uncorrectable_temporal_rad2": 2.0e-5,
                "sigma2_irreducible_rad2": 3.0e-4,
                "sigma2_residual_rad2": 4.7e-4,
                "sigma2_uncorrected_rad2": 8.0e-4,
                "piston_sensing_efficiency": 0.95,
                "f_piston_hz": 100.0,
                "servo_bandwidth_hz": 2000.0,
                "n_bright_ports": 6,
                "n_modes": 24,
                "shot_exceeds_open_loop": False,
                "wavelength_center_m": 1.25e-6,
                "servo_bandwidth_source": "params",
            },
        },
    )
    return replace(state, results=results)


def test_debug_report_writes_per_cell_pdfs_and_sweep(tmp_path: Path) -> None:
    """Multi-cell sweep emits one PDF per cell + a sweep-overview PDF."""

    config = _synthetic_resolved_config()
    jmags = (4.0, 8.0, 12.0)
    records = []
    for jmag, chip_ideal in zip(jmags, (3.0e-5, 5.0e-4, 1.0e-3)):
        cell_dir = tmp_path / "cells" / f"j{int(jmag)}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        state = _synthetic_state_with_sweep_metrics(jmag, chip_ideal)
        timings = tuple(
            StageTiming(instance_id=s.id, elapsed_s=0.5, cache_hit=False)
            for s in config.pipeline.stages
        )
        records.append(RunRecord(
            config=config,
            result=RunResult(
                state=state, diagnostics={}, timings=timings,
                cache_hit_count=0, config_hash="abc123",
            ),
            run_dir=cell_dir,
            coord_tuple=(("scene.star.jmag", jmag),),
        ))

    collection = RunCollection(
        config=config, records=tuple(records), output_dir=tmp_path,
    )
    written = debug_report(collection, tmp_path, formats=("pdf",))

    # Per-cell PDFs (3 cells) + sweep root mirror + sweep overview.
    assert (tmp_path / "plots" / "debug_report.pdf").resolve() in [p.resolve() for p in written]
    assert (tmp_path / "plots" / "debug_report_sweep.pdf").resolve() in [p.resolve() for p in written]
    for cell_dir in (tmp_path / "cells").iterdir():
        assert (cell_dir / "plots" / "debug_report.pdf").exists()
    # Sweep PDF should have meaningful content (multi-page).
    sweep_pdf = tmp_path / "plots" / "debug_report_sweep.pdf"
    assert sweep_pdf.exists()
    assert sweep_pdf.stat().st_size > 20 * 1024


def _photon_kalman_ft_payload() -> dict:
    return {
        "control_model": "photon_kalman",
        "mode": "full_piston_servo_residual",
        "wavelength_center_m": 1.25e-6,
        "photon_kalman_window_s": 1.1e-4,
        "photon_kalman_f_3db_equiv_hz": 3150.0,
        "photon_kalman_estimation_nm": 7.8,
        "photon_kalman_prediction_nm": 6.6,
        "photon_kalman_line_floor_nm": 4.9,
        "photon_kalman_total_common_nm": 11.3,
        "loop_latency_s": 5e-5,
        "crlb_m_sqrt_s": 8.1e-11,
        "inter_telescope_uncorrected_piston_rms_m": 1.688e-6,
        "tracked_line_count": 6,
        "line_tracking_residual_floor_nm": 2.0,
        "accelerometer_fusion": True,
        "fusion_coherent_fraction": 0.7,
        "fusion_broadband_suppression": 2.5,
        "vibration_broadband_rms_effective_nm": 19.3,
        "tracker_usable_rate_per_port_hz": 6.7e6,
        "tracker_rate_clipped": False,
        "residual_rms_opd_nm": 11.4,
        "residual_iid_per_mode_nm": 1.56,
        "residual_common_differential_nm": 11.3,
        "sigma_inter_telescope_residual_nm": 6.6,
        "sigma_vibration_residual_nm": 4.9,
        "n_modes": 12,
        "sigma2_residual_rad2": 3.3e-3,
        "sigma2_iid_per_mode_rad2": 6.1e-5,
        "sigma2_common_differential_rad2": 3.2e-3,
        "sigma2_photon_kalman_estimation_rad2": 1.5e-3,
        "sigma2_photon_kalman_prediction_rad2": 1.1e-3,
        "sigma2_photon_kalman_line_floor_rad2": 6.0e-4,
        "sigma2_irreducible_rad2": 6.1e-5,
        "sigma2_uncorrected_rad2": 6.4e-3,
    }


def _record_with_phase_b_payloads(tmp_path: Path) -> RunRecord:
    config = _synthetic_resolved_config()
    state = _synthetic_state()
    wf = state.wavefront
    wavefront = replace(wf, components={
        **dict(wf.components),
        "ao_residual": {
            "effective_wind_speed_m_s": 16.6,
            "r0_500nm_m": 0.15,
            "tau0_500nm_s": 2.83e-3,
        },
    })
    results = Results(components={
        "fringe_tracking.closed_loop": _photon_kalman_ft_payload(),
        "sensitivity.detection_curve": {
            "realized_null_depth": 5.0e-5,
            "disk_null_floor": 5.9e-6,
            "null_setpoint_leakage": 6.3e-6,
            "setpoint_dither_leakage": 3.9e-5,
            "effective_null_with_disk": 1.0e-4,
            "leakage_rate_hz_static_chip": 1.07e3,
            "leakage_rate_hz_effective": 8.16e3,
            "effective_to_static_leakage_ratio": 7.6,
            "null_instability": 0.385,
            "stellar_diameter_mas": 1.0,
            "chip_component": "chip.kernel_mzi_mesh",
            "chip_deployment_mode": "shared",
            "chip_n_input_modes": 24,
            "chip_n_output_ports": 24,
        },
        "performance.standard": {
            "eps_systematic_used": 0.0385,
            "nsc_systematic_reduction": 10.0,
            "residual_speckle_stability_factor": 0.0,
            "calibration_floor_applied": False,
            "calibration_floor_source": None,
            "upstream_detector_component": "detector.mkid",
            "snr": 4.6,
        },
    })
    state = replace(state, wavefront=wavefront, results=results)
    timings = tuple(
        StageTiming(instance_id=s.id, elapsed_s=0.5, cache_hit=False)
        for s in config.pipeline.stages
    )
    return RunRecord(
        config=config,
        result=RunResult(
            state=state, diagnostics={}, timings=timings,
            cache_hit_count=0, config_hash="abc123",
        ),
        run_dir=tmp_path,
        coord_tuple=(),
    )


def test_assumptions_page_pulls_live_payload_values(tmp_path: Path) -> None:
    """The closing limits page states the FT, systematics, and topology facts."""
    from matplotlib import pyplot as plt

    from nullsim.outputs.plots.debug_report import _assumptions_page

    record = _record_with_phase_b_payloads(tmp_path)
    fig = _assumptions_page(record, 1, 1)
    try:
        blob = "\n".join(t.get_text() for t in fig.texts)
    finally:
        plt.close(fig)

    assert "photon_kalman" in blob
    assert "no time-domain servo simulation" in blob
    assert "supersedes" in blob and "atmosphere" in blob
    assert "1.00e-04" in blob  # effective-null assembly result
    assert "NSC" in blob and "÷" not in blob  # prose, not the card glyph
    assert "Greenwood" in blob and "16.6" in blob
    assert "No [instrument] section" in blob  # synthetic config has none


def test_ft_disturbance_panel_renders_photon_kalman_payload(tmp_path: Path) -> None:
    """The FT text panel surfaces control, disturbance, and decomposition rows."""
    from matplotlib import pyplot as plt

    from nullsim.config.schema import StageSpec
    from nullsim.outputs.plots.debug_report import _panel_ft_disturbances

    record = _record_with_phase_b_payloads(tmp_path)
    spec = StageSpec.model_validate({
        "id": "fringe_tracking",
        "type": "fringe_tracking.closed_loop",
        "inter_telescope_piston": True,
        "outer_scale_m": 25.0,
        "wind_speed_m_s": 10.0,
        "vibration_line_frequencies_hz": [12.0, 29.0],
        "vibration_line_rms_nm": [30.0, 40.0],
        "vibration_broadband_rms_nm": 30.0,
    })
    breakdown = _photon_kalman_ft_payload()

    fig = plt.figure(figsize=(8.5, 11.0))
    ax = fig.add_axes((0.06, 0.10, 0.88, 0.42))
    try:
        _panel_ft_disturbances(ax, breakdown, record, spec)
        blob = "\n".join(t.get_text() for t in ax.texts)
    finally:
        plt.close(fig)

    assert "photon_kalman" in blob
    assert "Greenwood" in blob and "16.60 m/s" in blob and "10.00 m/s" in blob
    assert "12/30, 29/40" in blob          # vibration line spectrum
    assert "uncorr sigma 1688 nm" in blob  # the stage's published value
    assert "19.3 nm effective" in blob     # post-fusion broadband floor
    assert "atmosphere" in blob            # superseded OPD component listed


def test_pre_detector_transmission_divides_out_detector_qe(tmp_path: Path) -> None:
    """Review 2026-11-07 row 64: the cumulative curve stops before the detector."""
    from matplotlib import pyplot as plt

    from nullsim.outputs.plots.debug_report import _spectral_loss_page

    record = _record_with_phase_b_payloads(tmp_path)
    state = record.result.state
    wl = state.throughput.wavelength_grid_um
    pre_detector = np.array([0.5, 0.6, 0.55])
    throughput = replace(
        state.throughput,
        transmission_fraction=tuple(pre_detector * 0.7),
        components={
            **state.throughput.components,
            "detector.mkid_qe": {
                "wavelength_grid_um": wl,
                "transmission_fraction": (0.7, 0.7, 0.7),
                "qe": 0.7,
            },
        },
    )
    record = replace(
        record,
        result=replace(record.result, state=replace(state, throughput=throughput)),
    )
    fig = _spectral_loss_page(record, 1, 1)
    try:
        ax = next(ax for ax in fig.axes if ax.get_ylabel() == "Transmission (%)")
        plotted = np.asarray(ax.lines[0].get_ydata(), dtype=float)
    finally:
        plt.close(fig)
    np.testing.assert_allclose(plotted, pre_detector * 100.0, rtol=1e-12)
