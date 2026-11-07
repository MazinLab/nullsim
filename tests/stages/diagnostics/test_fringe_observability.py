"""Stage tests for the fringe-observability diagnostic."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nullsim.pipeline import (
    Field,
    Geometry,
    PhotonRates,
    Results,
    RunContext,
    SimulationState,
    build_instance,
)
from nullsim.physics.quadrature_fringe_tracker import quadrature_reference_mesh
from nullsim.stages.diagnostics import FringeObservability
from nullsim.stages.diagnostics.fringe_observability import (
    _detector_signal_rate_matrix,
)


def _context(tmp_path: Path) -> RunContext:
    return RunContext(
        rng=np.random.default_rng(0),
        config_hash="fringe-observability",
        sweep_coord_tuple=(),
        instance_id="fringe_observability",
        run_dir=tmp_path,
    )


def _random_unitary(n_modes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n_modes, n_modes)) + 1j * rng.normal(size=(n_modes, n_modes))
    q, r = np.linalg.qr(z)
    phase = np.diag(r) / np.abs(np.diag(r))
    return (q * phase[np.newaxis, :]).astype(np.complex128)


def _state() -> SimulationState:
    n_modes = 4
    n_wl = 5
    wavelengths = tuple(np.linspace(1.10, 1.35, n_wl))
    rng = np.random.default_rng(22)
    u_stack = np.stack(
        [_random_unitary(n_modes, seed=100 + i) for i in range(n_wl)],
        axis=2,
    )
    input_fields = (
        rng.normal(size=(n_modes, n_wl, 1)) + 1j * rng.normal(size=(n_modes, n_wl, 1))
    ).astype(np.complex128)
    output_fields = np.einsum("ijw,jwp->iwp", u_stack, input_fields)
    powers = np.sum(np.abs(output_fields) ** 2, axis=2)
    rates_hz = powers * 1.0e8
    state = SimulationState.empty({"wavelength_grid_um": wavelengths})
    return SimulationState(
        scene=state.scene,
        geometry=Geometry(
            n_apertures=n_modes,
            aperture_positions_pupil_m=tuple(
                (float(i), float(i % 2)) for i in range(n_modes)
            ),
        ),
        field=Field(amplitudes=output_fields, wavelength_grid_um=wavelengths),
        opd_budget=state.opd_budget,
        throughput=state.throughput,
        wavefront=state.wavefront,
        photon_rates=PhotonRates(
            wavelength_grid_um=wavelengths,
            port_labels=tuple(f"bright_{i}" for i in range(n_modes)),
            rates_hz=rates_hz,
        ),
        results=Results(
            components={
                "chip.kernel_mzi_mesh": {
                    "u_stack": u_stack,
                    "wavelength_grid_um": wavelengths,
                    "wl_design_um": 1.25,
                    "n_modes": n_modes,
                    "n_bright_ports": n_modes,
                    "n_dark_ports": 0,
                    "n_spectro_ports": 0,
                },
                "fringe_tracking.closed_loop": {
                    "servo_bandwidth_hz": 1000.0,
                    "piston_sensing_efficiency": 1.0,
                },
                "detector.mkid": {
                    "detector_qe": 0.7,
                    "uncapped_port_rate_hz_per_wavelength": rates_hz,
                },
            },
        ),
    )


def _rectangular_state() -> SimulationState:
    n_modes = 4
    n_wl = 3
    wavelengths = tuple(np.linspace(1.18, 1.32, n_wl))
    mesh = quadrature_reference_mesh(n_modes=n_modes, differential_coupling=0.05)
    transfer = np.vstack([mesh.tracker_matrix, mesh.science_matrix])
    u_stack = np.repeat(transfer[:, :, np.newaxis], n_wl, axis=2)
    input_fields = np.ones((n_modes, n_wl, 1), dtype=np.complex128)
    output_fields = np.einsum("ijw,jwp->iwp", u_stack, input_fields)
    powers = np.sum(np.abs(output_fields) ** 2, axis=2)
    rates_hz = powers * 1.0e8
    state = SimulationState.empty({"wavelength_grid_um": wavelengths})
    return SimulationState(
        scene=state.scene,
        geometry=Geometry(
            n_apertures=n_modes,
            aperture_positions_pupil_m=tuple(
                (float(i), float(i % 2)) for i in range(n_modes)
            ),
        ),
        field=Field(amplitudes=output_fields, wavelength_grid_um=wavelengths),
        opd_budget=state.opd_budget,
        throughput=state.throughput,
        wavefront=state.wavefront,
        photon_rates=PhotonRates(
            wavelength_grid_um=wavelengths,
            port_labels=tuple(f"bright_{i}" for i in range(n_modes))
            + tuple(f"dark_{i}" for i in range(n_modes)),
            rates_hz=rates_hz,
        ),
        results=Results(
            components={
                "chip.kernel_mzi_mesh": {
                    "u_stack": u_stack,
                    "wavelength_grid_um": wavelengths,
                    "wl_design_um": 1.25,
                    "n_modes": n_modes,
                    "n_input_modes": n_modes,
                    "n_output_ports": 2 * n_modes,
                    "n_bright_ports": n_modes,
                    "n_dark_ports": n_modes,
                    "n_spectro_ports": 0,
                },
                "fringe_tracking.closed_loop": {
                    "servo_bandwidth_hz": 1000.0,
                    "piston_sensing_efficiency": 1.0,
                },
                "detector.snspd": {
                    "detector_qe": 0.85,
                    "uncapped_port_rate_hz_per_wavelength": rates_hz,
                },
            },
        ),
    )


def _splitter_state() -> SimulationState:
    n_physical = 4
    n_chip = 8
    n_wl = 3
    wavelengths = tuple(np.linspace(1.18, 1.32, n_wl))
    rng = np.random.default_rng(44)
    u_stack = np.stack(
        [_random_unitary(n_chip, seed=200 + i) for i in range(n_wl)],
        axis=2,
    )
    input_fields = (
        rng.normal(size=(n_chip, n_wl, 1)) + 1j * rng.normal(size=(n_chip, n_wl, 1))
    ).astype(np.complex128)
    output_fields = np.einsum("ijw,jwp->iwp", u_stack, input_fields)
    powers = np.sum(np.abs(output_fields) ** 2, axis=2)
    rates_hz = powers * 1.0e8
    state = SimulationState.empty({"wavelength_grid_um": wavelengths})
    return SimulationState(
        scene=state.scene,
        geometry=Geometry(
            n_apertures=2,
            aperture_positions_pupil_m=np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [10.0, 0.0],
                    [11.0, 0.0],
                    [10.0, 0.0],
                    [11.0, 0.0],
                ],
                dtype=np.float64,
            ),
            components={
                "quadrature_split": {
                    "n_input_modes": n_physical,
                    "n_output_modes": n_chip,
                    "copies": 2,
                },
            },
        ),
        field=Field(amplitudes=output_fields, wavelength_grid_um=wavelengths),
        opd_budget=state.opd_budget,
        throughput=state.throughput,
        wavefront=state.wavefront,
        photon_rates=PhotonRates(
            wavelength_grid_um=wavelengths,
            port_labels=tuple(f"bright_{i}" for i in range(n_chip)),
            rates_hz=rates_hz,
        ),
        results=Results(
            components={
                "chip.kernel_mzi_mesh": {
                    "u_stack": u_stack,
                    "wavelength_grid_um": wavelengths,
                    "wl_design_um": 1.25,
                    "n_modes": n_chip,
                    "n_bright_ports": n_chip,
                    "n_dark_ports": 0,
                    "n_spectro_ports": 0,
                },
                "fringe_tracking.closed_loop": {
                    "servo_bandwidth_hz": 1000.0,
                    "piston_sensing_efficiency": 1.0,
                },
                "detector.snspd": {
                    "detector_qe": 0.85,
                    "uncapped_port_rate_hz_per_wavelength": rates_hz,
                },
            },
        ),
    )


def test_detector_signal_rate_matrix_can_be_forced_to_photon_rates() -> None:
    state = _state()
    base_rates = np.asarray(state.photon_rates.rates_hz, dtype=np.float64)
    photon_rates = np.full_like(base_rates, 2.0)
    uncapped_rates = np.full_like(base_rates, 9.0)
    state = replace(
        state,
        photon_rates=PhotonRates(
            wavelength_grid_um=state.photon_rates.wavelength_grid_um,
            port_labels=state.photon_rates.port_labels,
            rates_hz=photon_rates,
        ),
        results=Results(
            components={
                **state.results.components,
                "detector.mkid": {
                    "detector_qe": 0.7,
                    "uncapped_port_rate_hz_per_wavelength": uncapped_rates,
                },
            },
        ),
    )

    auto, auto_meta = _detector_signal_rate_matrix(state, stage_name="test")
    forced, forced_meta = _detector_signal_rate_matrix(
        state,
        stage_name="test",
        source_mode="photon_rates",
    )

    np.testing.assert_allclose(auto, uncapped_rates)
    assert auto_meta["rate_source_is_uncapped"] is True
    np.testing.assert_allclose(forced, photon_rates)
    assert forced_meta["rate_source"] == "state.photon_rates.rates_hz"
    assert forced_meta["rate_source_is_uncapped"] is False


def test_fringe_observability_stage_honors_count_rate_source(
    tmp_path: Path,
) -> None:
    state = _state()
    base_rates = np.asarray(state.photon_rates.rates_hz, dtype=np.float64)
    photon_rates = np.full_like(base_rates, 2.0e6)
    uncapped_rates = np.full_like(base_rates, 9.0e6)
    state = replace(
        state,
        photon_rates=PhotonRates(
            wavelength_grid_um=state.photon_rates.wavelength_grid_um,
            port_labels=state.photon_rates.port_labels,
            rates_hz=photon_rates,
        ),
        results=Results(
            components={
                **state.results.components,
                "detector.mkid": {
                    "detector_qe": 0.7,
                    "uncapped_port_rate_hz_per_wavelength": uncapped_rates,
                },
            },
        ),
    )
    stage = FringeObservability(
        params={
            "n_spectral_channels": 1,
            "n_monte_carlo": 0,
            "count_rate_source": "photon_rates",
            "tracker_qe": 0.85,
            "tracker_count_rate_cap_hz_per_channel": 1.0e12,
        },
        context=_context(tmp_path),
    )

    payload = stage.apply(state).results.components["fringe_observability"]

    assert payload["count_rate_source_mode"] == "photon_rates"
    assert payload["rate_source_is_uncapped"] is False
    assert payload["count_source"].startswith("state.photon_rates.rates_hz")
    assert payload["upstream_detector_qe"] == pytest.approx(0.7)
    assert payload["tracker_qe"] == pytest.approx(0.85)
    np.testing.assert_allclose(
        payload["expected_signal_counts_by_wavelength_per_sample"],
        photon_rates * (0.85 / 0.7) * payload["frame_time_s"],
    )


def test_fringe_observability_stage_publishes_rank_payload(tmp_path: Path) -> None:
    stage = build_instance(
        {
            "id": "fringe_observability",
            "type": "diagnostics.fringe_observability",
            "n_spectral_channels": 4,
            "n_monte_carlo": 16,
            "piston_rms_nm_for_mc": 10.0,
            "mc_seed": 4,
            "tracker_qe": 0.85,
            "tracker_count_rate_cap_hz_per_channel": 1.0e12,
            "dark_rate_hz_per_channel": 10.0,
        },
        _context(tmp_path),
    )
    assert isinstance(stage, FringeObservability)

    out = stage.apply(_state())
    payload = out.results.components["fringe_observability"]

    assert payload["n_differential_piston_modes"] == 3
    assert payload["n_spectral_channels_used"] == 4
    assert payload["frame_time_source"] == "fringe_tracking.servo_bandwidth"
    assert payload["count_source"].endswith(
        "uncapped_port_rate_hz_per_wavelength scaled to tracker_qe"
    )
    assert payload["tracker_qe"] == pytest.approx(0.85)
    assert payload["upstream_detector_qe"] == pytest.approx(0.7)
    assert payload["rate_source_is_uncapped"] is True
    assert payload["full_rank"] is True
    assert payload["rank"] == 3
    assert payload["observable_fraction"] == 1.0
    assert payload["scalar_piston_sensing_efficiency"] == 1.0
    assert payload["response_matrix_counts_per_m"].shape == (16, 3)
    assert payload["expected_signal_counts_by_wavelength_per_sample"].shape == (4, 5)
    assert payload["channel_sweep"]["single_channel_rank"] <= 3
    assert payload["channel_sweep"]["rank"][-1] == 3
    assert np.isfinite(payload["crlb_truncated_piston_rms_nm"])
    assert payload["piston_crlb_covariance_finite"] is True
    assert payload["piston_crlb_covariance_m2"].shape == (4, 4)
    assert payload["piston_crlb_coordinate_covariance_m2"].shape == (3, 3)
    assert payload["piston_crlb_covariance_rms_nm"] == pytest.approx(
        payload["crlb_piston_rms_nm"]
    )
    amplitude = payload["amplitude_observability"]
    assert amplitude["n_differential_amplitude_modes"] == 3
    assert amplitude["rank"] <= 3
    assert np.isfinite(amplitude["crlb_truncated_fractional_intensity_rms"])
    assert amplitude["frame_time_s"] == pytest.approx(payload["frame_time_s"])
    assert amplitude["frame_time_source"].startswith("piston_frame_time_fallback")
    assert amplitude["expected_signal_counts_per_sample"].shape == (16,)
    assert (
        amplitude["response_matrix_counts_per_fractional_intensity"].shape == (16, 3)
    )
    assert payload["weakest_piston_modes_unit_rms"].shape == (3, 4)
    assert payload["positions_pupil_m"].shape == (4, 2)
    assert payload["monte_carlo"]["residual_rms_nm"].shape == (16,)
    assert payload["monte_carlo_model"].startswith("linearized Poisson")
    assert payload["nonlinear_one_step_monte_carlo"]["residual_rms_nm"].shape == (16,)
    assert "capture range" in payload["nonlinear_one_step_model"]
    assert np.isfinite(payload["nonlinear_one_step_residual_ratio"])
    assert payload["gauss_newton_monte_carlo"]["residual_rms_nm"].shape == (16,)
    assert payload["gauss_newton_monte_carlo"]["iterations"].shape == (16,)
    assert np.isfinite(payload["gauss_newton_residual_ratio"])
    assert "Gauss-Newton" in payload["gauss_newton_model"]
    assert payload["gauss_newton_capture_sweep"]["enabled"] is False


def test_fringe_observability_publishes_amplitude_postprocessing(
    tmp_path: Path,
) -> None:
    residual = {
        "sigma_I_residual": 0.01,
        "sigma_I_open": 0.10,
        "sigma_I_temporal": 0.01,
        "sigma_I_photon": 0.002,
        "tau_loop_s": 1.0e-3,
        "tau_scint_s": 4.0e-3,
        "n_photons_per_sample": 1.0e3,
        "predictive_gain_variance": 20.0,
        "servo_bandwidth_hz": 1000.0,
        "photon_rate_per_sub_hz": 1.0e6,
    }
    state = _state()
    state = replace(
        state,
        results=state.results.add_component(
            "sensitivity.detection_curve",
            {
                "scintillation_index_per_sub": 0.01,
                "amplitude_servo_residual": residual,
            },
        ),
    )
    stage = FringeObservability(
        params={
            "n_spectral_channels": 2,
            "n_monte_carlo": 0,
            "amplitude_postprocessing_n_monte_carlo": 8,
            "tracker_qe": 0.85,
            "shot_noise": False,
        },
        context=_context(tmp_path),
    )

    out = stage.apply(state)
    payload = out.results.components["fringe_observability"]
    scintillation = payload["scintillation_assumptions"]
    post = payload["amplitude_postprocessing_monte_carlo"]

    assert payload["amplitude_observability"]["frame_time_s"] == pytest.approx(5e-4)
    assert payload["amplitude_observability"]["frame_time_source"].endswith(
        "amplitude_servo_residual.servo_bandwidth_hz"
    )
    assert scintillation["scintillation_sigma_i_injected"] == pytest.approx(0.01)
    assert scintillation["sigma_I_open"] == pytest.approx(0.10)
    assert scintillation["sigma_I_no_prediction_same_bandwidth"] > 0.01
    assert scintillation["predictive_gain_variance"] == pytest.approx(20.0)
    assert post["enabled"] is True
    assert post["n_trials"] == 8
    assert post["ideal_tap_counts_per_mode"] == pytest.approx(500.0)
    assert post["input_rms_percent"].shape == (8,)
    assert np.isfinite(post["median_bright_wiener_residual_rms_percent"])


def test_fringe_observability_can_use_dark_ports_for_amplitude_estimator(
    tmp_path: Path,
) -> None:
    residual = {
        "sigma_I_residual": 0.01,
        "sigma_I_open": 0.10,
        "sigma_I_temporal": 0.01,
        "sigma_I_photon": 0.002,
        "tau_loop_s": 1.0e-3,
        "tau_scint_s": 4.0e-3,
        "n_photons_per_sample": 1.0e3,
        "predictive_gain_variance": 20.0,
        "servo_bandwidth_hz": 1000.0,
        "photon_rate_per_sub_hz": 1.0e6,
    }
    state = _rectangular_state()
    state = replace(
        state,
        results=state.results.add_component(
            "sensitivity.detection_curve",
            {
                "scintillation_index_per_sub": 0.01,
                "amplitude_servo_residual": residual,
            },
        ),
    )
    stage = FringeObservability(
        params={
            "n_spectral_channels": 1,
            "n_monte_carlo": 0,
            "amplitude_estimator_port_set": "bright_dark",
            "amplitude_postprocessing_n_monte_carlo": 4,
            "tracker_qe": 0.85,
            "shot_noise": False,
        },
        context=_context(tmp_path),
    )

    out = stage.apply(state)
    payload = out.results.components["fringe_observability"]
    amplitude = payload["amplitude_observability"]
    post = payload["amplitude_postprocessing_monte_carlo"]

    assert amplitude["estimator_port_set"] == "bright_dark"
    assert amplitude["n_estimator_ports"] == 8
    assert amplitude["estimator_port_labels"][-1].startswith("dark_")
    assert amplitude["expected_signal_counts_per_sample"].shape == (8,)
    assert post["estimator_port_set"] == "bright_dark"
    assert post["linear_residual_rms_percent"].shape == (4,)
    assert np.isfinite(post["median_wiener_residual_rms_percent"])


def test_fringe_observability_active_acquisition_payload(tmp_path: Path) -> None:
    stage = FringeObservability(
        params={
            "n_spectral_channels": 1,
            "n_monte_carlo": 0,
            "active_acquisition_n_monte_carlo": 8,
            "active_acquisition_n_passes": 3,
            "active_acquisition_response_iterations": 2,
            "active_acquisition_probe_step_nm": 10.0,
            "active_acquisition_max_response_step_nm": 80.0,
            "active_acquisition_scan_half_range_nm": 150.0,
            "active_acquisition_scan_steps": 61,
            "active_acquisition_lock_threshold_nm": 10.0,
            "cophasing_handoff_n_monte_carlo": 8,
            "cophasing_science_drift_rms_nm": 2.0,
            "cophasing_lock_threshold_nm": 10.0,
            "tracker_qe": 0.85,
            "shot_noise": False,
        },
        context=_context(tmp_path),
    )

    out = stage.apply(_rectangular_state())
    payload = out.results.components["fringe_observability"]
    active = payload["active_acquisition_monte_carlo"]

    assert active["residual_rms_nm"].shape == (8,)
    assert active["evaluations"].shape == (8,)
    assert active["lock_fraction"] >= 0.5
    assert payload["active_acquisition_reference_mode"] == 0
    assert payload["active_acquisition_response_iterations"] == 2
    assert payload["active_acquisition_probe_step_nm"] == 10.0
    assert payload["active_acquisition_scan_steps"] == 61
    assert "coordinate scan" in payload["active_acquisition_model"]
    cophasing = payload["cophasing_handoff_monte_carlo"]
    assert cophasing["acquisition_residual_rms_nm"].shape == (8,)
    assert cophasing["science_residual_rms_nm"].shape == (8,)
    assert cophasing["edges"].shape == (3, 2)
    assert cophasing["science_lock_fraction"] >= 0.5
    assert "mode-switch acquisition" in payload["cophasing_handoff_model"]


def test_fringe_observability_gauss_newton_capture_sweep(tmp_path: Path) -> None:
    stage = FringeObservability(
        params={
            "n_monte_carlo": 4,
            "piston_rms_nm_for_mc": 10.0,
            "gauss_newton_piston_rms_grid_nm": (1.0, 10.0, 50.0),
            "gauss_newton_sweep_n_monte_carlo": 3,
            "frame_time_s": 0.01,
            "shot_noise": False,
        },
        context=_context(tmp_path),
    )

    out = stage.apply(_state())
    payload = out.results.components["fringe_observability"]
    sweep = payload["gauss_newton_capture_sweep"]

    assert sweep["enabled"] is True
    np.testing.assert_allclose(sweep["piston_rms_nm"], np.array([1.0, 10.0, 50.0]))
    assert sweep["median_residual_rms_nm"].shape == (3,)
    assert sweep["convergence_fraction"].shape == (3,)
    assert sweep["n_trials"] == 3


def test_fringe_observability_accepts_rectangular_isometry(tmp_path: Path) -> None:
    stage = FringeObservability(
        params={
            "n_spectral_channels": 1,
            "n_monte_carlo": 0,
            "tracker_qe": 0.85,
        },
        context=_context(tmp_path),
    )

    out = stage.apply(_rectangular_state())
    payload = out.results.components["fringe_observability"]

    assert payload["n_modes"] == 4
    assert payload["n_bright_ports"] == 4
    assert payload["rank"] == 3
    assert payload["full_rank"] is True
    assert payload["response_matrix_counts_per_m"].shape == (4, 3)


def test_fringe_observability_uses_physical_basis_after_mode_splitter(
    tmp_path: Path,
) -> None:
    stage = FringeObservability(
        params={
            "n_spectral_channels": 1,
            "n_monte_carlo": 4,
            "tracker_qe": 0.85,
        },
        context=_context(tmp_path),
    )

    out = stage.apply(_splitter_state())
    payload = out.results.components["fringe_observability"]

    assert payload["n_modes"] == 8
    assert payload["n_chip_input_modes"] == 8
    assert payload["n_physical_piston_modes"] == 4
    assert payload["n_differential_piston_modes"] == 3
    assert payload["piston_basis_source"] == "geometry.quadrature_split"
    assert payload["response_matrix_counts_per_m"].shape == (8, 3)
    assert payload["weakest_piston_modes_unit_rms"].shape == (3, 4)
    assert payload["positions_pupil_m"].shape == (4, 2)
    assert payload["rank"] <= 3


def test_fringe_observability_rejects_missing_chip_payload(tmp_path: Path) -> None:
    stage = FringeObservability(params={}, context=_context(tmp_path))
    state = SimulationState.empty({"wavelength_grid_um": (1.2, 1.3)})

    try:
        stage.apply(state)
    except ValueError as exc:
        assert "chip.kernel_mzi_mesh" in str(exc)
    else:
        raise AssertionError("missing chip payload should raise ValueError")


def test_fringe_observability_requires_explicit_frame_time_without_ft(
    tmp_path: Path,
) -> None:
    stage = FringeObservability(params={"n_monte_carlo": 0}, context=_context(tmp_path))
    state = _state()
    components = dict(state.results.components)
    components.pop("fringe_tracking.closed_loop")
    state = SimulationState(
        scene=state.scene,
        geometry=state.geometry,
        field=state.field,
        opd_budget=state.opd_budget,
        throughput=state.throughput,
        wavefront=state.wavefront,
        photon_rates=state.photon_rates,
        results=Results(components=components),
    )

    with pytest.raises(ValueError, match="frame_time_s"):
        stage.apply(state)


def test_fringe_observability_collapses_dual_pol_arms(tmp_path: Path) -> None:
    params = {
        "n_spectral_channels": 4,
        "n_monte_carlo": 8,
        "piston_rms_nm_for_mc": 10.0,
        "mc_seed": 4,
        "tracker_qe": 0.85,
        "tracker_count_rate_cap_hz_per_channel": 1.0e12,
        "dark_rate_hz_per_channel": 10.0,
    }
    base = _state()
    wavelengths = base.field.wavelength_grid_um
    arm0 = np.asarray(base.field.amplitudes, dtype=np.complex128)
    n_ports = arm0.shape[0]

    # Two PBS arms: dominant arm 0 plus a half-amplitude arm 1. Detector
    # channels are port-major, arm-minor with per-arm labels and rate rows.
    dual_amplitudes = np.concatenate([arm0, 0.5 * arm0], axis=2)
    rates_per_arm = (np.abs(dual_amplitudes) ** 2) * 1.0e8
    dual_rates = np.transpose(rates_per_arm, (0, 2, 1)).reshape(
        n_ports * 2, len(wavelengths)
    )
    dual_labels = tuple(
        f"bright_{i}_{arm}" for i in range(n_ports) for arm in ("A", "B")
    )
    dual_components = dict(base.results.components)
    dual_components["detector.mkid"] = {
        **dual_components["detector.mkid"],
        "n_pol_arms": 2,
        "uncapped_port_rate_hz_per_wavelength": dual_rates,
    }
    dual_pol = SimulationState(
        scene=base.scene,
        geometry=base.geometry,
        field=Field(amplitudes=dual_amplitudes, wavelength_grid_um=wavelengths),
        opd_budget=base.opd_budget,
        throughput=base.throughput,
        wavefront=base.wavefront,
        photon_rates=PhotonRates(
            wavelength_grid_um=wavelengths,
            port_labels=dual_labels,
            rates_hz=dual_rates,
        ),
        results=Results(components=dual_components),
    )

    # Single-pol reference: dominant arm 0 field with the arm-summed rates.
    summed_rates = rates_per_arm.sum(axis=2)
    reference_components = dict(base.results.components)
    reference_components["detector.mkid"] = {
        **reference_components["detector.mkid"],
        "uncapped_port_rate_hz_per_wavelength": summed_rates,
    }
    reference = SimulationState(
        scene=base.scene,
        geometry=base.geometry,
        field=Field(amplitudes=arm0, wavelength_grid_um=wavelengths),
        opd_budget=base.opd_budget,
        throughput=base.throughput,
        wavefront=base.wavefront,
        photon_rates=PhotonRates(
            wavelength_grid_um=wavelengths,
            port_labels=tuple(f"bright_{i}" for i in range(n_ports)),
            rates_hz=summed_rates,
        ),
        results=Results(components=reference_components),
    )

    dual_payload = (
        FringeObservability(params=params, context=_context(tmp_path))
        .apply(dual_pol)
        .results.components["fringe_observability"]
    )
    reference_payload = (
        FringeObservability(params=params, context=_context(tmp_path))
        .apply(reference)
        .results.components["fringe_observability"]
    )

    assert dual_payload["rank"] == reference_payload["rank"]
    assert dual_payload["crlb_piston_rms_nm"] == pytest.approx(
        reference_payload["crlb_piston_rms_nm"]
    )
    assert dual_payload["piston_crlb_covariance_m2"] == pytest.approx(
        reference_payload["piston_crlb_covariance_m2"]
    )


def _with_detectors(
    state: SimulationState, detectors: dict[str, dict[str, object]]
) -> SimulationState:
    components = {
        name: payload
        for name, payload in state.results.components.items()
        if not name.startswith("detector.")
    }
    components.update(detectors)
    return replace(state, results=Results(components=components))


_OBSERVABILITY_PARAMS = {
    "n_spectral_channels": 4,
    "n_monte_carlo": 8,
    "piston_rms_nm_for_mc": 10.0,
    "mc_seed": 4,
}


def _observability_payload(state: SimulationState, tmp_path: Path) -> dict:
    return (
        FringeObservability(params=_OBSERVABILITY_PARAMS, context=_context(tmp_path))
        .apply(state)
        .results.components["fringe_observability"]
    )


def test_fringe_observability_ignores_mkid_energy_redistribution(
    tmp_path: Path,
) -> None:
    """Issue 6: the tracker response pairs with the true per-wavelength rates.

    detector.mkid with resolving_power_ref stores the energy-redistributed
    spectrum under uncapped_port_rate_hz_per_wavelength. The CRLB must equal
    the one computed from the true rates alone.
    """

    base = _state()
    true_rates = np.asarray(
        base.results.components["detector.mkid"]["uncapped_port_rate_hz_per_wavelength"]
    )
    redistributed = np.repeat(
        true_rates.mean(axis=1, keepdims=True), true_rates.shape[1], axis=1
    )
    energy_resolving = _with_detectors(
        base,
        {
            "detector.mkid": {
                "detector_qe": 0.7,
                "resolving_power_ref": 30.0,
                "uncapped_port_rate_hz_per_wavelength": redistributed,
                "true_rates_hz_per_port_wavelength": true_rates,
            }
        },
    )

    payload = _observability_payload(energy_resolving, tmp_path)
    reference = _observability_payload(base, tmp_path)

    assert payload["rate_source"] == "detector.mkid.true_rates_hz_per_port_wavelength"
    assert payload["crlb_piston_rms_nm"] == pytest.approx(
        reference["crlb_piston_rms_nm"], rel=1e-12
    )


def test_fringe_observability_takes_qe_and_rates_from_newest_detector(
    tmp_path: Path,
) -> None:
    """Issue 25: an older detector's QE never pairs with a newer one's rates."""

    base = _state()
    rates = np.asarray(
        base.results.components["detector.mkid"]["uncapped_port_rate_hz_per_wavelength"]
    )
    state = _with_detectors(
        base,
        {
            "detector.mkid": {
                "detector_qe": 0.7,
                "uncapped_port_rate_hz_per_wavelength": 3.0 * rates,
            },
            "detector.snspd": {
                "detector_qe": 0.85,
                "uncapped_port_rate_hz_per_wavelength": rates,
            },
        },
    )
    params = {**_OBSERVABILITY_PARAMS, "tracker_qe": 0.85}

    payload = (
        FringeObservability(params=params, context=_context(tmp_path))
        .apply(state)
        .results.components["fringe_observability"]
    )

    assert payload["upstream_detector_component"] == "detector.snspd"
    assert payload["upstream_detector_qe"] == pytest.approx(0.85)
    assert payload["tracker_qe_scale"] == pytest.approx(1.0)
    assert payload["rate_source"] == "detector.snspd.uncapped_port_rate_hz_per_wavelength"


def test_fringe_observability_sums_fisher_over_distinct_pol_arms(
    tmp_path: Path,
) -> None:
    """Row 49: each arm's own field and rates; the Fisher matrices add.

    Arm B carries a different input-field pattern, so pairing the dominant
    arm's field with arm-summed rates is wrong. The dual-arm CRLB precision
    must equal the sum of the two single-arm precisions, each computed from
    a scalar state built from that arm alone.
    """

    base = _state()
    wavelengths = base.field.wavelength_grid_um
    u_stack = np.asarray(base.results.components["chip.kernel_mzi_mesh"]["u_stack"])
    arm_a = np.asarray(base.field.amplitudes, dtype=np.complex128)
    rng = np.random.default_rng(5)
    input_b = (
        rng.normal(size=(4, len(wavelengths), 1))
        + 1j * rng.normal(size=(4, len(wavelengths), 1))
    )
    arm_b = 0.7 * np.einsum("ijw,jwp->iwp", u_stack, input_b)
    n_ports = arm_a.shape[0]

    def scalar_state(arm: np.ndarray) -> SimulationState:
        rates = np.abs(arm[:, :, 0]) ** 2 * 1.0e8
        return replace(
            _with_detectors(
                base,
                {
                    "detector.mkid": {
                        "detector_qe": 0.7,
                        "uncapped_port_rate_hz_per_wavelength": rates,
                    }
                },
            ),
            field=Field(amplitudes=arm, wavelength_grid_um=wavelengths),
            photon_rates=PhotonRates(
                wavelength_grid_um=wavelengths,
                port_labels=base.photon_rates.port_labels,
                rates_hz=rates,
            ),
        )

    dual_amplitudes = np.concatenate([arm_a, arm_b], axis=2)
    dual_rates = np.transpose(
        np.abs(dual_amplitudes) ** 2 * 1.0e8, (0, 2, 1)
    ).reshape(2 * n_ports, len(wavelengths))
    dual = replace(
        _with_detectors(
            base,
            {
                "detector.mkid": {
                    "detector_qe": 0.7,
                    "n_pol_arms": 2,
                    "uncapped_port_rate_hz_per_wavelength": dual_rates,
                }
            },
        ),
        field=Field(amplitudes=dual_amplitudes, wavelength_grid_um=wavelengths),
        photon_rates=PhotonRates(
            wavelength_grid_um=wavelengths,
            port_labels=tuple(
                f"bright_{i}_{arm}" for i in range(n_ports) for arm in ("A", "B")
            ),
            rates_hz=dual_rates,
        ),
    )

    dual_payload = _observability_payload(dual, tmp_path)
    precision = [
        np.linalg.pinv(
            _observability_payload(scalar_state(arm), tmp_path)[
                "piston_crlb_covariance_m2"
            ]
        )
        for arm in (arm_a, arm_b)
    ]

    assert dual_payload["piston_fisher_polarization_arms"] == 2
    np.testing.assert_allclose(
        np.linalg.pinv(dual_payload["piston_crlb_covariance_m2"]),
        precision[0] + precision[1],
        rtol=1e-8,
        atol=1e-8 * float(np.max(np.abs(precision[0] + precision[1]))),
    )
