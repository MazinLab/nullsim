"""Tests for the fringe-observability plot output."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nullsim.outputs import RunCollection, RunRecord
from nullsim.outputs.plots.fringe_observability import (
    _fmt_float,
    _find_payload,
    fringe_observability,
)
from nullsim.pipeline import Results, RunResult, SimulationState


def _payload() -> dict:
    return {
        "n_modes": 4,
        "n_differential_piston_modes": 3,
        "n_bright_ports": 4,
        "n_spectral_channels_used": 5,
        "singular_values_per_m": np.array([1.0e9, 2.0e8, 5.0e7]),
        "singular_value_tolerance_per_m": 1.0e3,
        "rank": 3,
        "full_rank": True,
        "observable_fraction": 1.0,
        "condition_number": 20.0,
        "truncated_condition_number": 20.0,
        "crlb_piston_rms_nm": 1.4,
        "crlb_truncated_piston_rms_nm": 1.4,
        "scalar_piston_sensing_efficiency": 1.0,
        "amplitude_observability": {
            "n_differential_amplitude_modes": 3,
            "singular_values_per_fractional_intensity": np.array([30.0, 8.0, 2.0]),
            "rank": 3,
            "condition_number": 15.0,
            "crlb_fractional_intensity_percent": 0.2,
            "crlb_truncated_fractional_intensity_percent": 0.2,
            "frame_time_s": 1.0e-4,
            "expected_signal_counts_per_sample": np.full(20, 300.0),
        },
        "scintillation_assumptions": {
            "scintillation_sigma_i_injected": 0.01,
            "sigma_I_open": 0.10,
            "sigma_I_no_prediction_same_bandwidth": 0.04,
            "predictive_gain_variance": 20.0,
        },
        "amplitude_postprocessing_monte_carlo": {
            "enabled": True,
            "median_input_rms_percent": 0.9,
            "median_bright_linear_residual_rms_percent": 12.0,
            "median_bright_wiener_residual_rms_percent": 0.9,
            "median_ideal_tap_residual_rms_percent": 0.05,
        },
        "frame_time_s": 5.0e-4,
        "frame_time_source": "fringe_tracking.servo_bandwidth",
        "count_source": "state.photon_rates.rates_hz scaled to tracker_qe",
        "expected_counts_per_sample": np.full(20, 1500.0),
        "channel_sweep": {
            "n_spectral_channels": np.arange(1, 6),
            "rank": np.array([2, 3, 3, 3, 3]),
            "crlb_piston_rms_nm": np.array([np.inf, 3.0, 2.0, 1.6, 1.4]),
            "crlb_truncated_piston_rms_nm": np.array([8.0, 3.0, 2.0, 1.6, 1.4]),
            "single_channel_rank": 2,
        },
        "positions_pupil_m": np.array(
            [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]
        ),
        "weakest_piston_modes_unit_rms": np.array(
            [[1.0, -1.0, 0.5, -0.5], [1.0, 1.0, -1.0, -1.0]]
        ),
        "monte_carlo": {
            "residual_rms_nm": np.linspace(2.0, 8.0, 24),
            "median_residual_rms_nm": 5.0,
            "p95_residual_rms_nm": 7.7,
        },
        "nonlinear_one_step_monte_carlo": {
            "residual_rms_nm": np.linspace(30.0, 80.0, 24),
            "median_residual_rms_nm": 55.0,
            "p95_residual_rms_nm": 77.0,
        },
        "gauss_newton_monte_carlo": {
            "residual_rms_nm": np.linspace(3.0, 12.0, 24),
            "median_residual_rms_nm": 7.0,
            "p95_residual_rms_nm": 11.0,
            "convergence_fraction": 0.92,
        },
        "active_acquisition_monte_carlo": {
            "residual_rms_nm": np.linspace(0.6, 3.0, 24),
            "median_residual_rms_nm": 1.4,
            "p95_residual_rms_nm": 2.8,
            "lock_fraction": 1.0,
        },
        "cophasing_handoff_monte_carlo": {
            "acquisition_residual_rms_nm": np.linspace(0.8, 4.0, 24),
            "science_residual_rms_nm": np.linspace(1.0, 5.0, 24),
            "median_acquisition_residual_rms_nm": 1.8,
            "median_science_residual_rms_nm": 2.1,
            "science_lock_fraction": 1.0,
        },
        "gauss_newton_capture_sweep": {
            "enabled": True,
            "n_trials": 16,
            "piston_rms_nm": np.array([1.0, 3.0, 10.0, 30.0, 100.0]),
            "median_residual_rms_nm": np.array([1.2, 2.0, 12.0, 80.0, 500.0]),
            "p95_residual_rms_nm": np.array([2.0, 5.0, 30.0, 200.0, 900.0]),
            "convergence_fraction": np.array([1.0, 1.0, 0.75, 0.25, 0.0]),
        },
    }


def test_fmt_float_distinguishes_nonfinite_values() -> None:
    assert _fmt_float(float("nan")) == "nan"
    assert _fmt_float(float("inf")) == "inf"
    assert _fmt_float(float("-inf")) == "-inf"


def test_find_payload_prefers_full_diagnostic_over_prepass() -> None:
    full_payload = _payload()
    state = SimulationState.empty({"wavelength_grid_um": (1.2, 1.3)})
    state = SimulationState(
        scene=state.scene,
        geometry=state.geometry,
        field=state.field,
        opd_budget=state.opd_budget,
        throughput=state.throughput,
        wavefront=state.wavefront,
        photon_rates=state.photon_rates,
        results=Results(
            components={
                "fringe_observability_pre_sensitivity": {"rank": 0},
                "fringe_observability": full_payload,
            },
        ),
    )
    record = RunRecord(
        config=SimpleNamespace(),
        result=RunResult(
            state=state,
            diagnostics={},
            timings=(),
            cache_hit_count=0,
            config_hash="plot",
        ),
        run_dir=Path("."),
        coord_tuple=(),
    )

    selected = _find_payload(record)

    assert selected is not None
    assert selected["rank"] == full_payload["rank"]
    assert selected["n_differential_piston_modes"] == 3


def test_realized_piston_floor_prefers_covariance_resolved() -> None:
    """The realized temporal-lag floor is read from the detection-curve
    ao_piston_residual, preferring the covariance-resolved value over the
    temporal-selected and scalar-resolved fallbacks."""

    from nullsim.outputs.plots.fringe_observability import _realized_piston_floor_nm

    residual = {
        "fringe_observability_covariance_resolved_rms_nm": 3.0,
        "fringe_observability_temporal_selected_rms_nm": 2.5,
        "fringe_observability_resolved_rms_nm": 0.17,
    }
    components = {"sensitivity.detection_curve": {"ao_piston_residual": residual}}
    record = SimpleNamespace(
        result=SimpleNamespace(
            state=SimpleNamespace(results=SimpleNamespace(components=components))
        )
    )
    assert _realized_piston_floor_nm(record) == 3.0
    assert _realized_piston_floor_nm(None) is None
    # Falls back to the temporal-selected value when covariance is absent.
    residual.pop("fringe_observability_covariance_resolved_rms_nm")
    assert _realized_piston_floor_nm(record) == 2.5


def test_registered_fringe_observability_writes_plot(tmp_path: Path) -> None:
    state = SimulationState.empty({"wavelength_grid_um": (1.2, 1.3)})
    state = SimulationState(
        scene=state.scene,
        geometry=state.geometry,
        field=state.field,
        opd_budget=state.opd_budget,
        throughput=state.throughput,
        wavefront=state.wavefront,
        photon_rates=state.photon_rates,
        results=Results(components={"fringe_observability": _payload()}),
    )
    record = RunRecord(
        config=SimpleNamespace(),
        result=RunResult(
            state=state,
            diagnostics={},
            timings=(),
            cache_hit_count=0,
            config_hash="plot",
        ),
        run_dir=tmp_path,
        coord_tuple=(),
    )
    collection = RunCollection(
        config=SimpleNamespace(),
        records=(record,),
        output_dir=tmp_path,
    )

    written = fringe_observability(collection, tmp_path, ("png",))

    out = tmp_path / "plots" / "fringe_observability.png"
    assert out in written
    assert out.exists()
    assert out.stat().st_size > 3_000


def test_realized_piston_floor_reads_opd_budget_source_keys() -> None:
    """kilo_keck_6 runs with ao_piston_residual_source="opd_budget", whose
    residual payload publishes METER-valued keys. The summary row must
    resolve them (x1e9 -> nm) instead of rendering "n/a" in exactly the
    flagship config it was built for (review 2026-06-10 Issue 11)."""

    from nullsim.outputs.plots.fringe_observability import _realized_piston_floor_nm

    residual = {
        "opd_budget_covariance_resolved_rms_m": 3.99e-9,
        "closed_loop_opd_budget_rms_m": 5.0e-9,
    }
    components = {"sensitivity.detection_curve": {"ao_piston_residual": residual}}
    record = SimpleNamespace(
        result=SimpleNamespace(
            state=SimpleNamespace(results=SimpleNamespace(components=components))
        )
    )
    assert _realized_piston_floor_nm(record) == pytest.approx(3.99)
    residual.pop("opd_budget_covariance_resolved_rms_m")
    assert _realized_piston_floor_nm(record) == pytest.approx(5.0)
