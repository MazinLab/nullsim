"""Tests for the characterization.spectro stage."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from nullsim.pipeline import (
    Field,
    Geometry,
    RunContext,
    Scene,
    SimulationState,
    Throughput,
)
from nullsim.pipeline.state import Results
from nullsim.stages.characterization import (
    CharacterizationDetectionCurve,
    SpectroCharacterization,
)
from nullsim.stages.characterization.spectro import (
    _band_evaluated_spectro_metrics,
    _characterization_wavelength_grid,
    _find_sensitivity_component,
    _mc_piston_opd_covariance_m2,
    _realized_spectro_floor_for_stack,
    _realized_spectro_floor,
    _spectro_sky_rate_hz,
    _static_spectro_leak_fraction,
)
import nullsim.stages.characterization.detection_curve as detection_curve_module


def _context(tmp_path: Path) -> RunContext:
    return RunContext(
        rng=np.random.default_rng(0),
        config_hash="characterization-spectro",
        sweep_coord_tuple=(),
        instance_id="characterization",
        run_dir=tmp_path,
    )


def _state() -> SimulationState:
    wavelengths_um = (1.20, 1.25, 1.30)
    state = SimulationState.empty({"wavelength_grid_um": wavelengths_um})
    n_modes = 2
    components = {
        "sensitivity.detection_curve": {
            "coupling_per_sub": np.ones(n_modes),
            "sub_areas_m2": np.ones(n_modes),
            "sub_diameters_m": np.ones(n_modes),
            "effective_null_with_disk": 1.0e-4,
            "null_instability": 0.0,
            "effective_systematic_bandwidth_hz": 1000.0,
            "wavelengths_m": tuple(w * 1.0e-6 for w in wavelengths_um),
        },
        "detector.mkid": {
            "port_rates": np.array([1000.0, 1.0]),
            "port_labels": ("bright_0", "dark_0"),
            "integration_time_s": 100.0,
            "detector_qe": 1.0,
        },
    }
    return replace(
        state,
        scene=Scene(
            star_jmag=4.0,
            star_angular_diameter_mas=0.0,
            planet_contrast=1.0e-6,
            planet_separation_mas=10.0,
            planet_position_angle_deg=90.0,
        ),
        geometry=Geometry(
            diameter_m=1.0,
            area_m2=math.pi / 4.0,
            collecting_area_m2=math.pi / 4.0,
            effective_collecting_area_m2=2.0,
            n_apertures=1,
            aperture_positions_pupil_m=np.array([[0.0, 0.0], [10.0, 0.0]]),
        ),
        field=Field(
            amplitudes=np.ones((n_modes, len(wavelengths_um), 1), dtype=np.complex128),
            wavelength_grid_um=wavelengths_um,
        ),
        throughput=Throughput(
            wavelength_grid_um=wavelengths_um,
            transmission_fraction=(1.0, 1.0, 1.0),
        ),
        results=Results(components=components),
    )


def test_spectro_characterization_writes_payload(tmp_path: Path) -> None:
    stage = SpectroCharacterization(
        params={
            "n_modes": 2,
            "n_bright_ports": 1,
            "n_dark_ports": 0,
            "n_spectro_ports": 1,
            "null_floor": 1.0,
            "n_restarts": 1,
            "max_iter": 50,
            "seed": 0,
            "broadband_n_wavelengths": 1,
        },
        context=_context(tmp_path),
    )

    out = stage.apply(_state())
    payload = out.results.components["characterization.spectro"]

    assert payload["snr"] >= 0.0
    assert payload["max_spectral_resolution"] >= 0.0
    assert payload["spectro_throughput"] >= 0.0
    assert payload["leakage_rate_hz"] > 0.0
    assert payload["optimal_u_stack"].shape == (2, 2, 3)
    assert payload["detected_stellar_rate_by_wavelength_hz"].shape == (3,)
    assert payload["spectro_static_leak_fraction"] == pytest.approx(
        payload["spectro_band_static_leak_fraction"]
    )
    assert payload["spectro_leak_fraction"] == pytest.approx(
        payload["spectro_band_leak_fraction"]
    )
    assert payload["spectro_band_on_axis_null_depth"] >= 0.0
    assert payload["spectro_planet_throughput_per_wavelength"].shape == (3,)
    assert payload["spectro_planet_to_star_ratio_per_wavelength"].shape == (3,)
    assert isinstance(payload["optimization_success"], bool)
    assert np.isfinite(payload["optimization_constraint_margin"])
    assert payload["spectrograph"] == "awg"
    assert payload["awg_n_bins"] == payload["awg"]["n_bins"]
    assert payload["awg"]["bin_centers_m"].shape == (payload["awg_n_bins"],)


def test_spectro_characterization_writes_awg_spectrum(tmp_path: Path) -> None:
    stage = SpectroCharacterization(
        params={
            "n_modes": 2,
            "n_bright_ports": 1,
            "n_dark_ports": 0,
            "n_spectro_ports": 1,
            "null_floor": 1.0,
            "n_restarts": 1,
            "max_iter": 50,
            "seed": 0,
            "broadband_n_wavelengths": 1,
            "awg_resolving_power": 100.0,
            "awg_throughput": 0.5,
            "awg_free_spectral_range_fraction": 0.5,
            "awg_pixels_per_resolution_element": 2,
            "n_dark_count_hz": 2.0,
            "n_sky_background_hz": 100.0,
            "n_spectro_pixels": 3,
        },
        context=_context(tmp_path),
    )

    payload = stage.apply(_state()).results.components["characterization.spectro"]
    awg = payload["awg"]

    assert payload["awg_resolving_power"] == pytest.approx(100.0)
    assert payload["awg_n_bins"] == 12
    assert awg["throughput"] == pytest.approx(0.5)
    assert awg["free_spectral_range_fraction"] == pytest.approx(0.5)
    assert np.sum(awg["planet_rate_hz_per_bin"]) == pytest.approx(
        payload["planet_signal_rate_hz"] * 0.25,
    )
    assert np.sum(awg["leakage_rate_hz_per_bin"]) == pytest.approx(
        payload["leakage_rate_hz"] * 0.25,
    )
    assert np.sum(awg["sky_rate_hz_per_bin"]) == pytest.approx(25.0)
    assert awg["dark_rate_hz_per_bin"][0] == pytest.approx(6.0)
    assert payload["awg_min_snr_per_bin"] == pytest.approx(
        np.min(awg["snr_per_bin"]),
    )


def test_awg_resolving_power_defaults_independently_of_resolving_power_cap(
    tmp_path: Path,
) -> None:
    """A deliberately uncapped ``resolving_power_cap`` (used to free the
    SNR-limited max-R search) must NOT drive the AWG channel count. Before the
    fix the unset ``awg_resolving_power`` fell back to ``resolving_power_cap``,
    so a 1e9 cap built a ~1e8-bin AWG grid and allocated tens of GB, OOM-killing
    CI runners."""

    stage = SpectroCharacterization(
        params={
            "n_modes": 2,
            "n_bright_ports": 1,
            "n_dark_ports": 0,
            "n_spectro_ports": 1,
            "null_floor": 1.0,
            "n_restarts": 1,
            "max_iter": 50,
            "seed": 0,
            "broadband_n_wavelengths": 1,
            # Uncap the max-R search; leave awg_resolving_power unset.
            "resolving_power_cap": 1.0e9,
        },
        context=_context(tmp_path),
    )

    payload = stage.apply(_state()).results.components["characterization.spectro"]

    # AWG resolving power is the bounded hardware default, not the search cap.
    assert payload["awg_resolving_power"] == pytest.approx(10_000.0)
    assert payload["awg_n_bins"] == payload["awg"]["n_bins"]
    # Sanity: the bin grid is small, nowhere near the ~1e8 bins the bug built.
    assert payload["awg_n_bins"] < 100_000


def test_spectro_characterization_mkid_spectrograph_uses_detector_r(
    tmp_path: Path,
) -> None:
    state = _state()
    components = dict(state.results.components)
    detector = dict(components["detector.mkid"])
    detector["resolving_power_ref"] = 30.0
    detector["lambda_ref_um"] = 1.0
    components["detector.mkid"] = detector
    state = replace(state, results=Results(components=components))
    stage = SpectroCharacterization(
        params={
            "n_modes": 2,
            "n_bright_ports": 1,
            "n_dark_ports": 0,
            "n_spectro_ports": 1,
            "null_floor": 1.0,
            "n_restarts": 1,
            "max_iter": 50,
            "seed": 0,
            "broadband_n_wavelengths": 1,
            "spectrograph": "mkid",
            "resolving_power_cap": 1000.0,
        },
        context=_context(tmp_path),
    )

    payload = stage.apply(state).results.components["characterization.spectro"]

    assert payload["spectrograph"] == "mkid"
    assert payload["upstream_detector_component"] == "detector.mkid"
    assert payload["mkid_resolving_power"] == pytest.approx(30.0 / 1.25)
    assert payload["mkid_lambda_ref_um"] == pytest.approx(1.0)
    assert payload["mkid_pixels_per_resolution_element"] == 1
    assert payload["mkid_min_snr_per_bin"] == pytest.approx(
        payload["mkid_snr_per_bin"]
    )
    assert payload["max_spectral_resolution"] <= payload["mkid_resolving_power"]
    assert "awg" not in payload


def test_spectro_characterization_mkid_requires_mkid_detector(
    tmp_path: Path,
) -> None:
    state = _state()
    components = dict(state.results.components)
    components["detector.snspd"] = components.pop("detector.mkid")
    state = replace(state, results=Results(components=components))
    stage = SpectroCharacterization(
        params={
            "n_modes": 2,
            "n_bright_ports": 1,
            "n_dark_ports": 0,
            "n_spectro_ports": 1,
            "null_floor": 1.0,
            "n_restarts": 1,
            "max_iter": 50,
            "seed": 0,
            "broadband_n_wavelengths": 1,
            "spectrograph": "mkid",
        },
        context=_context(tmp_path),
    )

    with pytest.raises(ValueError, match="detector.mkid"):
        stage.apply(state)


def test_spectro_characterization_mkid_requires_enabled_resolution(
    tmp_path: Path,
) -> None:
    stage = SpectroCharacterization(
        params={
            "n_modes": 2,
            "n_bright_ports": 1,
            "n_dark_ports": 0,
            "n_spectro_ports": 1,
            "null_floor": 1.0,
            "n_restarts": 1,
            "max_iter": 50,
            "seed": 0,
            "broadband_n_wavelengths": 1,
            "spectrograph": "mkid",
        },
        context=_context(tmp_path),
    )

    with pytest.raises(ValueError, match="R-enabled"):
        stage.apply(_state())


def test_spectro_characterization_mkid_rejects_multiple_spectro_ports() -> None:
    with pytest.raises(ValidationError, match="exactly one spectro port"):
        SpectroCharacterization.Params(
            n_modes=3,
            n_bright_ports=1,
            n_dark_ports=0,
            n_spectro_ports=2,
            spectrograph="mkid",
        )


def test_spectro_characterization_enforces_null_floor(tmp_path: Path) -> None:
    """A real null_floor must be honoured by the optimizer at design λ.

    The deployed chromatic stack can have larger off-design leakage; that
    is reported separately in the band-evaluated leakage fields.
    """

    null_floor = 1.0e-3
    stage = SpectroCharacterization(
        params={
            "n_modes": 2,
            "n_bright_ports": 1,
            "n_dark_ports": 0,
            "n_spectro_ports": 1,
            "null_floor": null_floor,
            "n_restarts": 4,
            "max_iter": 400,
            "seed": 1,
            "broadband_n_wavelengths": 1,
        },
        context=_context(tmp_path),
    )
    payload = stage.apply(_state()).results.components["characterization.spectro"]
    assert payload["spectro_null_depth"] <= 1.01 * null_floor


def test_spectro_characterization_publishes_tracker_observability(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    stage = SpectroCharacterization(
        params={
            "n_modes": 2,
            "n_bright_ports": 1,
            "n_dark_ports": 0,
            "n_spectro_ports": 1,
            "null_floor": 1.0,
            "n_restarts": 1,
            "max_iter": 80,
            "seed": 0,
            "backend": "torch",
            "tracker_observability_weight": 1.0e-3,
            "tracker_observability_crlb_weight": 1.0e-4,
            "tracker_observability_crlb_target_nm": 100.0,
        },
        context=_context(tmp_path),
    )

    payload = stage.apply(_state()).results.components["characterization.spectro"]

    singular = payload["tracker_observability_singular_values_per_nm"]
    assert singular.shape == (1,)
    assert singular[0] > 0.0
    assert payload["tracker_observability_crlb_rms_nm"] > 0.0
    assert payload["tracker_observability_basis_source"] == "chip_input_modes"


def test_characterization_curve_accepts_tracker_observability(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    stage = CharacterizationDetectionCurve(
        params={
            "n_modes": 2,
            "n_bright_ports": 1,
            "n_dark_ports": 0,
            "n_spectro_ports": 1,
            "null_floor": 1.0,
            "n_anchors": 2,
            "separation_min_mas": 1.0,
            "separation_max_mas": 2.0,
            "n_restarts": 1,
            "max_iter": 80,
            "seed": 0,
            "backend": "torch",
            "tracker_observability_weight": 1.0e-3,
            "tracker_observability_crlb_weight": 1.0e-4,
            "tracker_observability_crlb_target_nm": 100.0,
        },
        context=_context(tmp_path),
    )

    payload = stage.apply(_state()).results.components[
        "characterization.detection_curve"
    ]

    assert payload["contrast_5sigma"].shape == (2,)
    assert np.all(np.isfinite(payload["contrast_5sigma"]))


def test_characterization_curve_evaluates_full_wavelength_grid(
    tmp_path: Path,
) -> None:
    stage = CharacterizationDetectionCurve(
        params={
            "n_modes": 2,
            "n_bright_ports": 1,
            "n_dark_ports": 0,
            "n_spectro_ports": 1,
            "null_floor": 1.0,
            "n_anchors": 2,
            "separation_min_mas": 1.0,
            "separation_max_mas": 2.0,
            "n_restarts": 1,
            "max_iter": 80,
            "seed": 0,
            "backend": "scipy",
            "broadband_n_wavelengths": 2,
        },
        context=_context(tmp_path),
    )

    payload = stage.apply(_state()).results.components[
        "characterization.detection_curve"
    ]

    assert len(payload["broadband_wavelengths_m"]) == 2
    assert payload["spectro_on_axis_null_per_anchor_wavelength"].shape == (2, 3)
    assert payload["spectro_static_leak_fraction_per_anchor_wavelength"].shape == (
        2,
        3,
    )
    assert payload["detected_stellar_rate_by_wavelength_hz"].shape == (3,)
    assert "full detector wavelength grid" in payload["source_model"]


def test_characterization_curve_allows_single_anchor_equal_bounds() -> None:
    base_params = {
        "n_modes": 2,
        "n_bright_ports": 1,
        "n_dark_ports": 0,
        "n_spectro_ports": 1,
        "n_anchors": 1,
        "separation_min_mas": 5.0,
        "separation_max_mas": 5.0,
    }

    params = CharacterizationDetectionCurve.Params.model_validate(base_params)

    assert params.n_anchors == 1
    assert params.separation_min_mas == params.separation_max_mas
    with pytest.raises(ValidationError):
        CharacterizationDetectionCurve.Params.model_validate(
            base_params | {"n_anchors": 2},
        )


def test_characterization_wavelength_grid_uses_detector_bins() -> None:
    wavelengths_m = np.linspace(1.18e-6, 1.32e-6, 20)

    representatives = _characterization_wavelength_grid(
        wavelengths_m,
        5,
        center_wavelength_m=float(wavelengths_m[9]),
    )

    assert representatives.shape == (5,)
    assert set(representatives.tolist()).issubset(set(wavelengths_m.tolist()))
    assert float(wavelengths_m[9]) in set(representatives.tolist())
    np.testing.assert_allclose(
        _characterization_wavelength_grid(wavelengths_m, 20),
        wavelengths_m,
    )


def test_characterization_curve_reports_full_band_leak_margin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer_calls: list[dict[str, object]] = []

    def fake_optimize_characterization(**kwargs: object) -> SimpleNamespace:
        optimizer_calls.append(kwargs)
        wavelengths_m = np.asarray(kwargs["deployment_wavelengths_m"], dtype=np.float64)
        u_stack = np.repeat(
            np.eye(2, dtype=np.complex128)[:, :, np.newaxis],
            wavelengths_m.size,
            axis=2,
        )
        returned_params = np.full(4, float(len(optimizer_calls)), dtype=np.float64)
        return SimpleNamespace(
            params=returned_params,
            optimization_success=True,
            constraint_margin=1.0e-3,
            tracker_crlb_rms_nm=None,
            optimal_u_stack=u_stack,
            optimal_u_stack_wavelengths_m=wavelengths_m,
        )

    def fake_spectro_disk_leak_from_stack(**kwargs: object) -> tuple[float, np.ndarray]:
        wavelengths_m = np.asarray(kwargs["wavelengths_m"], dtype=np.float64)
        return 0.0, np.zeros(wavelengths_m.shape, dtype=np.float64)

    def fake_band_metrics(**kwargs: object) -> dict[str, object]:
        wavelengths_m = np.asarray(kwargs["wavelengths_m"], dtype=np.float64)
        return {
            "on_axis_null_depth": 0.0,
            "static_leak_fraction": 2.0e-3,
            "planet_signal_rate_hz": 10.0,
            "on_axis_null_per_wavelength": np.zeros_like(wavelengths_m),
            "static_leak_fraction_per_wavelength": np.full_like(
                wavelengths_m,
                2.0e-3,
            ),
            "planet_throughput_per_wavelength": np.ones_like(wavelengths_m),
            "planet_to_star_ratio_per_wavelength": np.ones_like(wavelengths_m),
        }

    floor_calls: list[object] = []

    def fake_realized_spectro_floor_for_stack(
        **kwargs: object,
    ) -> tuple[float, str, float | None]:
        floor_calls.append(kwargs["u_stack"])
        return 1.0e-6 * len(floor_calls), "fake retuned floor", None

    monkeypatch.setattr(
        detection_curve_module,
        "optimize_characterization",
        fake_optimize_characterization,
    )
    monkeypatch.setattr(
        detection_curve_module,
        "spectro_disk_leak_from_stack",
        fake_spectro_disk_leak_from_stack,
    )
    monkeypatch.setattr(
        detection_curve_module,
        "_band_evaluated_spectro_metrics",
        fake_band_metrics,
    )
    monkeypatch.setattr(
        detection_curve_module,
        "_realized_spectro_floor_for_stack",
        fake_realized_spectro_floor_for_stack,
    )
    stage = CharacterizationDetectionCurve(
        params={
            "n_modes": 2,
            "n_bright_ports": 1,
            "n_dark_ports": 0,
            "n_spectro_ports": 1,
            "null_floor": 1.0e-3,
            "n_anchors": 2,
            "separation_min_mas": 1.0,
            "separation_max_mas": 2.0,
            "n_restarts": 0,
            "max_iter": 1,
            "n_workers": 3,
            "seed": 0,
        },
        context=_context(tmp_path),
    )

    payload = stage.apply(_state()).results.components[
        "characterization.detection_curve"
    ]

    np.testing.assert_allclose(
        payload["optimization_constraint_margin_per_anchor"],
        [1.0e-3, 1.0e-3],
    )
    np.testing.assert_allclose(
        payload["full_band_constraint_margin_per_anchor"],
        [-1.0e-3, -1.0e-3],
    )
    assert np.all(payload["optimization_success_per_anchor"])
    assert payload["snr_per_anchor"].shape == (2,)
    np.testing.assert_allclose(
        payload["realized_spectro_floor_per_anchor"],
        [1.0e-6, 2.0e-6],
    )
    assert payload["realized_spectro_floor"] == pytest.approx(1.5e-6)
    assert payload["realized_spectro_floor_model"] == "fake retuned floor"
    assert len(floor_calls) == 2
    assert payload["n_workers"] == 3
    assert len(optimizer_calls) == 2
    first_call = optimizer_calls[0]
    assert first_call["n_workers"] == 3
    assert first_call["initial_params"] == ()
    second_initial_params = optimizer_calls[1]["initial_params"]
    assert len(second_initial_params) == 1
    np.testing.assert_allclose(
        np.asarray(second_initial_params[0], dtype=np.float64),
        np.ones(4, dtype=np.float64),
    )
    np.testing.assert_allclose(
        np.asarray(first_call["selection_wavelengths_m"], dtype=np.float64),
        np.asarray(payload["wavelengths_m"], dtype=np.float64),
    )
    assert len(first_call["selection_stellar_fields_per_wl"]) == len(
        payload["wavelengths_m"],
    )
    assert len(first_call["selection_planet_fields_per_wl"]) == len(
        payload["wavelengths_m"],
    )
    np.testing.assert_allclose(
        np.asarray(first_call["selection_stellar_weights_per_wl"], dtype=np.float64),
        np.asarray(payload["detected_stellar_rate_by_wavelength_hz"], dtype=np.float64),
    )


def test_static_spectro_leak_uses_disk_integral_for_resolved_star() -> None:
    """Resolved-disk leakage already includes static on-axis chip leakage."""

    assert _static_spectro_leak_fraction(
        on_axis_null=6.0e-5,
        disk_leak=5.0e-5,
        stellar_angular_diameter_mas=1.0,
    ) == pytest.approx(5.0e-5)


def test_static_spectro_leak_uses_on_axis_for_unresolved_star() -> None:
    assert _static_spectro_leak_fraction(
        on_axis_null=6.0e-5,
        disk_leak=0.0,
        stellar_angular_diameter_mas=0.0,
    ) == pytest.approx(6.0e-5)


def test_realized_spectro_floor_removes_detection_static_terms_and_shares_null_floor() -> None:
    sensitivity = {
        "ideal_chip_null_depth": 2.0e-5,
        "realized_null_depth": 7.0e-5,
        "disk_null_floor": 3.0e-5,
        "effective_null_with_disk": 1.0e-4,
    }

    assert _realized_spectro_floor(
        sensitivity=sensitivity,
        n_dark_ports=9,
        n_spectro_ports=1,
    ) == pytest.approx(5.0e-5 / 9.0)


def test_realized_spectro_floor_replays_retuned_unitary_mc() -> None:
    wavelengths_m = np.array([1.25e-6], dtype=np.float64)
    base = np.ones(2, dtype=np.float64)
    hadamard = np.array(
        [[1.0, 1.0], [1.0, -1.0]],
        dtype=np.complex128,
    ) / np.sqrt(2.0)
    identity = np.eye(2, dtype=np.complex128)
    sensitivity = {
        "ao_piston_opd_rms_per_sub_m": 20.0e-9,
        "scintillation_index_per_sub": 0.0,
        "n_realizations_per_azimuth": 128,
        "mc_seed": 3,
        "fiber_phase_noise_rms_rad": 0.0,
    }

    interferometric_floor, model, _ = _realized_spectro_floor_for_stack(
        sensitivity=sensitivity,
        u_stack=hadamard[:, :, np.newaxis],
        base_amplitudes=base,
        spectro_idx=np.array([1], dtype=np.intp),
        wavelengths_m=wavelengths_m,
        stellar_rate_weights=np.ones(1),
        n_dark_ports=1,
        n_spectro_ports=1,
    )
    identity_floor, _, _ = _realized_spectro_floor_for_stack(
        sensitivity=sensitivity,
        u_stack=identity[:, :, np.newaxis],
        base_amplitudes=base,
        spectro_idx=np.array([1], dtype=np.intp),
        wavelengths_m=wavelengths_m,
        stellar_rate_weights=np.ones(1),
        n_dark_ports=1,
        n_spectro_ports=1,
    )

    assert "retuned spectro unitary" in model
    assert interferometric_floor > 0.0
    assert identity_floor == pytest.approx(0.0)


def test_spectro_characterization_dac_metrics_match_quantized_chip(
    tmp_path: Path,
) -> None:
    """With DAC quantization on, the reported throughput must equal the
    throughput recomputed from the deployed quantized unitary at the
    design wavelength. Catches metric/u_stack mismatch."""

    stage = SpectroCharacterization(
        params={
            "n_modes": 2,
            "n_bright_ports": 1,
            "n_dark_ports": 0,
            "n_spectro_ports": 1,
            "null_floor": 1.0e-3,
            "n_restarts": 2,
            "max_iter": 200,
            "seed": 2,
            "broadband_n_wavelengths": 1,
            "dac_bits": 6,
        },
        context=_context(tmp_path),
    )
    payload = stage.apply(_state()).results.components["characterization.spectro"]

    wavelengths_m = np.asarray(payload["wavelengths_m"], dtype=np.float64)
    design_index = int(np.argmin(np.abs(wavelengths_m - payload["band_lambda_m"])))
    u_design = payload["optimal_u_stack"][:, :, design_index]

    # planet field on-axis for a single-baseline 2-aperture array with
    # planet at separation_mas=10, PA=0 along east. East offset on the
    # second aperture only: phase = 2π * (x*alpha) / λ.
    positions = np.array([[0.0, 0.0], [10.0, 0.0]])
    planet_sep_rad = 10.0 / 206265e3
    wl = float(payload["band_lambda_m"])
    phase = 2.0 * np.pi * positions[:, 0] * planet_sep_rad / wl
    # base amplitudes from coupling*area = 1 (the test fixture)
    planet_input = np.exp(1j * phase)
    planet_total = float(np.sum(np.abs(planet_input) ** 2))
    planet_output = u_design @ planet_input
    spectro_idx = np.asarray(payload["spectro_indices"], dtype=np.intp)
    recomputed_throughput = (
        float(np.sum(np.abs(planet_output[spectro_idx]) ** 2)) / planet_total
    )
    assert recomputed_throughput == pytest.approx(
        payload["spectro_throughput"], rel=1.0e-9, abs=1.0e-12,
    )


def test_spectro_characterization_requires_band_bandwidth_for_single_bin(
    tmp_path: Path,
) -> None:
    """Single-wavelength grids cannot infer bin width from neighbours,
    so the stage must require an explicit ``band_bandwidth_um``."""

    wavelengths_um = (1.25,)
    state = replace(
        _state(),
        field=Field(
            amplitudes=np.ones((2, 1, 1), dtype=np.complex128),
            wavelength_grid_um=wavelengths_um,
        ),
        throughput=Throughput(
            wavelength_grid_um=wavelengths_um,
            transmission_fraction=(1.0,),
        ),
    )
    sensitivity = dict(state.results.components["sensitivity.detection_curve"])
    sensitivity["wavelengths_m"] = (1.25e-6,)
    components = dict(state.results.components)
    components["sensitivity.detection_curve"] = sensitivity
    state = replace(state, results=Results(components=components))

    stage_missing = SpectroCharacterization(
        params={
            "n_modes": 2,
            "n_bright_ports": 1,
            "n_dark_ports": 0,
            "n_spectro_ports": 1,
            "null_floor": 1.0,
            "n_restarts": 1,
            "max_iter": 50,
            "seed": 0,
            "broadband_n_wavelengths": 1,
        },
        context=_context(tmp_path),
    )
    with pytest.raises(ValueError, match="band_bandwidth_um"):
        stage_missing.apply(state)

    stage_with_bw = SpectroCharacterization(
        params={
            "n_modes": 2,
            "n_bright_ports": 1,
            "n_dark_ports": 0,
            "n_spectro_ports": 1,
            "null_floor": 1.0,
            "n_restarts": 1,
            "max_iter": 50,
            "seed": 0,
            "broadband_n_wavelengths": 1,
            "band_bandwidth_um": 0.16,
        },
        context=_context(tmp_path),
    )
    payload = stage_with_bw.apply(state).results.components[
        "characterization.spectro"
    ]
    assert payload["band_bandwidth_m"] == pytest.approx(0.16e-6)


def test_spectro_characterization_broadband_runs(tmp_path: Path) -> None:
    """Broadband mode (``broadband_n_wavelengths > 1``) must run and
    populate the per-wavelength disk-leak vector."""

    stage = SpectroCharacterization(
        params={
            "n_modes": 2,
            "n_bright_ports": 1,
            "n_dark_ports": 0,
            "n_spectro_ports": 1,
            "null_floor": 1.0e-3,
            "n_restarts": 1,
            "max_iter": 100,
            "seed": 3,
            "broadband_n_wavelengths": 3,
        },
        context=_context(tmp_path),
    )
    payload = stage.apply(_state()).results.components["characterization.spectro"]
    assert payload["spectro_disk_leak_per_wavelength"].shape == (3,)
    assert payload["spectro_on_axis_null_depth"] >= 0.0


def test_per_bin_metrics_share_headline_eps_convention(tmp_path: Path) -> None:
    """max_spectral_resolution must use the SAME NSC-reduced combined eps as
    the headline SNR. Pre-fix the spectrograph paths received the raw
    null_instability — a 100x larger systematic variance inside the same
    payload row (review 2026-06-10 Issue 3)."""

    def _payload(nsc: float) -> dict:
        state = _state()
        components = dict(state.results.components)
        sens = dict(components["sensitivity.detection_curve"])
        sens["null_instability"] = 0.5  # systematic-dominated regime
        components["sensitivity.detection_curve"] = sens
        det = dict(components["detector.mkid"])
        det["port_rates"] = np.array([1.0e7, 1.0e2])
        components["detector.mkid"] = det
        state = replace(
            state,
            results=replace(state.results, components=components),
        )
        # Bright planet so per-bin SNR is photon-capable and the max-R
        # solve binds on the systematic term.
        state = replace(
            state,
            scene=replace(state.scene, planet_contrast=1.0e-3),
        )
        stage = SpectroCharacterization(
            params={
                "n_modes": 2,
                "n_bright_ports": 1,
                "n_dark_ports": 0,
                "n_spectro_ports": 1,
                "null_floor": 1.0,
                "n_restarts": 1,
                "max_iter": 50,
                "seed": 0,
                "broadband_n_wavelengths": 1,
                "nsc_systematic_reduction": nsc,
                # Uncap so the SNR-limited solve binds on the systematic
                # term instead of saturating at the default cap.
                "resolving_power_cap": 1.0e9,
            },
            context=_context(tmp_path),
        )
        return stage.apply(state).results.components["characterization.spectro"]

    raw = _payload(1.0)
    reduced = _payload(100.0)
    # The headline SNR improves with NSC...
    assert reduced["snr"] > raw["snr"]
    # ...and the SNR-limited max resolving power must improve with it too
    # (same eps convention), instead of staying pinned at the raw value.
    assert reduced["max_spectral_resolution"] > raw["max_spectral_resolution"]


_HADAMARD = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)


def _spectro_stage(tmp_path: Path, **overrides: object) -> SpectroCharacterization:
    params: dict[str, object] = {
        "n_modes": 2,
        "n_bright_ports": 1,
        "n_dark_ports": 0,
        "n_spectro_ports": 1,
        "null_floor": 1.0,
        "n_restarts": 1,
        "max_iter": 50,
        "seed": 0,
        "broadband_n_wavelengths": 1,
    }
    params.update(overrides)
    return SpectroCharacterization(params=params, context=_context(tmp_path))


def _with_sensitivity(state: SimulationState, **updates: object) -> SimulationState:
    components = dict(state.results.components)
    sens = dict(components["sensitivity.detection_curve"])
    sens.update(updates)
    components["sensitivity.detection_curve"] = sens
    return replace(state, results=replace(state.results, components=components))


def test_retuned_spectro_floor_is_photon_weighted_band_leakage() -> None:
    """Hadamard pair with IID OPD sigma: the difference port leaks
    (1 - cos(k dOPD))/2 with dOPD ~ N(0, 2 sigma^2). The band leakage
    L = sum_k w_k (1 - cos(k_k dOPD))/2 is a function of one Gaussian, so
    its mean and RMS follow from Gauss-Hermite quadrature."""

    wavelengths_m = np.array([1.0e-6, 2.0e-6])
    rates = np.array([3.0, 1.0])
    sigma_opd_m = 60.0e-9
    floor, _, leak_rms = _realized_spectro_floor_for_stack(
        sensitivity={
            "ao_piston_opd_rms_per_sub_m": sigma_opd_m,
            "n_realizations_per_azimuth": 40_000,
            "mc_seed": 11,
        },
        u_stack=np.repeat(_HADAMARD[:, :, None], 2, axis=2),
        base_amplitudes=np.ones(2),
        spectro_idx=np.array([1], dtype=np.intp),
        wavelengths_m=wavelengths_m,
        stellar_rate_weights=rates,
        n_dark_ports=1,
        n_spectro_ports=1,
    )

    nodes, gh_weights = np.polynomial.hermite_e.hermegauss(60)
    gh_weights = gh_weights / gh_weights.sum()
    d_opd = np.sqrt(2.0) * sigma_opd_m * nodes
    k = 2.0 * np.pi / wavelengths_m
    band = (rates / rates.sum()) @ (0.5 * (1.0 - np.cos(k[:, None] * d_opd[None, :])))
    expected_mean = float(gh_weights @ band)
    expected_rms = float(np.sqrt(gh_weights @ (band - expected_mean) ** 2))
    unweighted_mean = float(np.mean(0.5 * (1.0 - np.exp(-(k * sigma_opd_m) ** 2))))

    assert floor == pytest.approx(expected_mean, rel=0.03)
    assert abs(floor - unweighted_mean) > 0.2 * expected_mean
    assert leak_rms == pytest.approx(expected_rms, rel=0.03)


def test_spectro_eps_is_retuned_port_cv_not_detection_null_instability(
    tmp_path: Path,
) -> None:
    state = _with_sensitivity(
        _state(),
        null_instability=0.5,
        ao_piston_opd_rms_per_sub_m=30.0e-9,
        n_realizations_per_azimuth=500,
        mc_seed=1,
    )
    payload = _spectro_stage(tmp_path).apply(state).results.components[
        "characterization.spectro"
    ]

    expected_cv = payload["spectro_leak_rms_fraction"] / payload["spectro_leak_fraction"]
    assert payload["spectro_leak_rms_fraction"] > 0.0
    assert payload["null_instability"] == pytest.approx(expected_cv)
    assert payload["eps_systematic"] == pytest.approx(expected_cv)
    assert payload["detection_null_instability"] == pytest.approx(0.5)


def test_band_metrics_planet_throughput_includes_piston_expectation() -> None:
    """Planet arms differ in phase by psi = 2 pi B theta / lambda; with IID
    phase sigma the difference port receives (1 - exp(-sigma^2) cos psi)/2."""

    wavelength_m = 1.25e-6
    baseline_m, separation_rad = 10.0, 30.0 / 206265e3
    sigma_opd_m = 80.0e-9
    sensitivity = {
        "ao_piston_opd_rms_per_sub_m": sigma_opd_m,
        "n_realizations_per_azimuth": 10,
    }
    covariance = _mc_piston_opd_covariance_m2(sensitivity, 2)
    metrics = _band_evaluated_spectro_metrics(
        u_stack=_HADAMARD[:, :, None],
        wavelengths_m=np.array([wavelength_m]),
        detected_stellar_rate_by_wavelength_hz=np.array([1.0e6]),
        positions_m=np.array([[0.0, 0.0], [baseline_m, 0.0]]),
        coupling_per_sub=np.ones(2),
        sub_areas_m2=np.ones(2),
        sub_diameters_m=np.ones(2),
        base_amplitudes=np.ones(2),
        spectro_idx=np.array([1], dtype=np.intp),
        separation_rad=separation_rad,
        position_angle_rad=math.pi / 2.0,
        planet_contrast=1.0,
        disk_leak_per_wavelength=np.zeros(1),
        stellar_angular_diameter_mas=0.0,
        piston_opd_covariance_m2=covariance,
    )

    psi = 2.0 * np.pi * baseline_m * separation_rad / wavelength_m
    sigma_rad = 2.0 * np.pi * sigma_opd_m / wavelength_m
    assert metrics["planet_throughput_per_wavelength"][0] == pytest.approx(
        0.5 * (1.0 - np.exp(-sigma_rad**2) * np.cos(psi))
    )
    assert _mc_piston_opd_covariance_m2(
        {**sensitivity, "n_realizations_per_azimuth": 0}, 2
    ) is None


def test_spectrograph_snr_uses_quasi_static_branch_without_fringe_tracking(
    tmp_path: Path,
) -> None:
    """Without fringe tracking a single AWG bin cannot beat the band SNR, and
    each bin's systematic is (eps * leakage counts)^2, not servo averaged."""

    state = _with_sensitivity(_state(), null_instability=0.2)
    payload = _spectro_stage(
        tmp_path, with_fringe_tracking=False, awg_resolving_power=100.0,
    ).apply(state).results.components["characterization.spectro"]
    awg = payload["awg"]
    t_s = 100.0
    signal = awg["planet_rate_hz_per_bin"] * t_s
    leak = awg["leakage_rate_hz_per_bin"] * t_s
    expected = signal / np.sqrt(signal + leak + (payload["eps_systematic"] * leak) ** 2)

    assert payload["servo_bw_hz"] == 0.0
    np.testing.assert_allclose(awg["snr_per_bin"], expected, rtol=1.0e-12)
    assert payload["awg_min_snr_per_bin"] < payload["snr"]


def test_max_spectral_resolution_applies_awg_throughput(tmp_path: Path) -> None:
    """At the returned R the per-bin SNR, with AWG throughput 0.5 on every
    optical rate and n_bins = 2 R dlambda / lambda, equals snr_min."""

    state = replace(_state(), scene=replace(_state().scene, planet_contrast=1.0e-4))
    payload = _spectro_stage(
        tmp_path,
        awg_throughput=0.5,
        resolving_power_cap=1.0e9,
        snr_min_per_bin=5.0,
    ).apply(state).results.components["characterization.spectro"]
    r_max = payload["max_spectral_resolution"]
    n_bins = 2.0 * r_max * payload["band_bandwidth_m"] / payload["band_lambda_m"]
    signal = payload["planet_signal_rate_hz"] * 0.5 * 100.0 / n_bins
    leak = payload["leakage_rate_hz"] * 0.5 * 100.0 / n_bins

    assert 1.0 < r_max < 1.0e9
    assert signal / math.sqrt(signal + leak) == pytest.approx(5.0, rel=1.0e-3)


def test_spectro_sky_charges_every_polarization_arm(tmp_path: Path) -> None:
    state = _state()
    components = dict(state.results.components)
    components["detector.mkid"] = {**components["detector.mkid"], "n_pol_arms": 2}
    state = replace(state, results=replace(state.results, components=components))
    payload = _spectro_stage(
        tmp_path, n_sky_background_hz=100.0, awg_throughput=0.5,
    ).apply(state).results.components["characterization.spectro"]

    assert payload["n_pol_arms"] == 2
    assert payload["sky_rate_hz_per_spectro_arm"] == pytest.approx(100.0)
    assert payload["Sky"] == pytest.approx(200.0 * 100.0)
    assert payload["awg_total_sky_rate_hz"] == pytest.approx(200.0 * 0.5)


def test_spectro_sky_is_routed_through_the_retuned_rows() -> None:
    """Two physical inputs split into four coherent copies (physical index
    [0, 1, 0, 1]). Spectro row 0 adds input 0's copies in phase in bin 0
    (share 1) and takes one copy in bin 1 (share 1/2). Per-input sky is
    [30, 70] Hz, recovered from the published dark sky [15, 17.5] Hz over a
    dark share [0.5, 0.25]. Per arm: 30 * 1 + 70 * 0.5 + 3 Hz additive = 68 Hz,
    where the published all-output mean is 50 + 3 Hz."""

    r = 1.0 / np.sqrt(2.0)
    bin0 = np.array(
        [[r, 0, r, 0], [r, 0, -r, 0], [0, r, 0, r], [0, r, 0, -r]],
        dtype=np.complex128,
    )
    u_stack = np.stack([bin0, np.eye(4, dtype=np.complex128)], axis=2)
    detection_routing = np.array(
        [[0.5, 0.75], [0.5, 0.25], [0.5, 0.5], [0.5, 0.5]]
    )
    state = replace(
        SimulationState.empty({"wavelength_grid_um": (1.2, 1.3)}),
        geometry=Geometry(
            components={
                "injection.mode_splitter": {
                    "n_input_modes": 2,
                    "n_output_modes": 4,
                    "copies": 2,
                    "physical_mode_indices": np.array([0, 1, 0, 1]),
                }
            }
        ),
        field=Field(
            amplitudes=np.ones((4, 2, 1), dtype=np.complex128),
            wavelength_grid_um=(1.2, 1.3),
        ),
        results=Results(
            components={
                "background.sky_surface_brightness": {
                    "sky_rate_hz_per_input_mode": 100.0,
                    "sky_rate_hz_per_bin": np.array([15.0, 17.5]),
                    "sky_routing_fraction_per_port": detection_routing,
                    "sky_routing_dark_port_indices": np.array([1]),
                    "sky_rate_hz_per_spectro_port": 50.0,
                    "background_rate_hz_per_spectro_port": 53.0,
                }
            }
        ),
    )

    n_arms, per_arm, per_port, source = _spectro_sky_rate_hz(
        state,
        {"n_pol_arms": 2},
        0.0,
        u_stack=u_stack,
        spectro_idx=np.array([0], dtype=np.intp),
    )

    assert per_arm == pytest.approx(68.0)
    assert per_port == pytest.approx(2 * 68.0)
    assert n_arms == 2
    assert "retuned" in source


def test_spectro_leak_rms_includes_setpoint_bias_cross_term() -> None:
    """Small-phase Bracewell null L = (D + beta)^2 / 4 with D ~ N(0, 2 s^2):
    Var L = s^4 / 2 + beta^2 s^2 / 2 exactly, and L_b = beta^2 / 4 is the
    setpoint leakage the stage reads."""

    wavelength_m = 1.25e-6
    sigma_rad, beta_rad = 0.02, 0.2
    sensitivity = {
        "ao_piston_opd_rms_per_sub_m": sigma_rad * wavelength_m / (2.0 * np.pi),
        "n_realizations_per_azimuth": 40_000,
        "mc_seed": 5,
        "null_setpoint_leakage": beta_rad**2 / 4.0,
        "setpoint_dither_leakage": 0.0,
    }
    kwargs = dict(
        u_stack=_HADAMARD[:, :, None],
        base_amplitudes=np.ones(2),
        spectro_idx=np.array([1], dtype=np.intp),
        wavelengths_m=np.array([wavelength_m]),
        stellar_rate_weights=np.ones(1),
        n_dark_ports=1,
        n_spectro_ports=1,
    )

    _, _, leak_rms = _realized_spectro_floor_for_stack(sensitivity=sensitivity, **kwargs)

    assert leak_rms**2 == pytest.approx(
        sigma_rad**4 / 2.0 + beta_rad**2 * sigma_rad**2 / 2.0, rel=0.04,
    )

    # The bias is a phase offset: scintillation must not change the cross term
    # (same seed, so the phase draws are shared).
    def cross(extra: dict) -> float:
        biased = _realized_spectro_floor_for_stack(
            sensitivity={**sensitivity, **extra}, **kwargs,
        )[2]
        unbiased = _realized_spectro_floor_for_stack(
            sensitivity={**sensitivity, **extra, "null_setpoint_leakage": 0.0},
            **kwargs,
        )[2]
        return biased**2 - unbiased**2

    assert cross({"scintillation_index_per_sub": 0.1}) == pytest.approx(
        cross({}), rel=1.0e-12,
    )


def test_find_sensitivity_component_prefers_newest() -> None:
    payload = {
        "coupling_per_sub": np.ones(2),
        "sub_areas_m2": np.ones(2),
        "sub_diameters_m": np.ones(2),
        "effective_null_with_disk": 1.0e-4,
        "wavelengths_m": (1.25e-6,),
    }
    state = replace(
        SimulationState.empty({}),
        results=Results(
            components={
                "sensitivity.detection_curve": payload,
                "sensitivity.detection_curve.late": {**payload, "effective_null_with_disk": 2.0e-4},
            }
        ),
    )

    name, _ = _find_sensitivity_component(state)

    assert name == "sensitivity.detection_curve.late"
