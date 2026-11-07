from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

import nullsim  # noqa: F401  # ensures output registration side effects are available
from nullsim.config.loader import resolve_config
from nullsim.config.schema import ImagingPlanetConfig
from nullsim.outputs import RunCollection, RunRecord
from nullsim.outputs.plots.imaging_reconstruction import (
    _build_imaging_inputs,
    _planet_views,
    _stellar_rates_by_wavelength,
    imaging_reconstruction,
)
from nullsim.physics.reflected_light import reflected_light_contrast
from nullsim.pipeline import (
    Geometry,
    PhotonRates,
    Results,
    RunResult,
    Scene,
    SimulationState,
)


def test_imaging_reconstruction_writes_plot_json_and_arrays(tmp_path: Path) -> None:
    config = resolve_config(
        {
            "run": {"name": "imaging_test", "seed": 3},
            "grid": {
                "wavelength_center_um": 1.0,
                "wavelength_bandwidth_um": 0.1,
                "n_wavelength_bins": 3,
            },
            "site": {"latitude_deg": 0.0},
            "scene": {
                "star": {
                    "jmag": 5.0,
                    "angular_diameter_mas": 1.0,
                    "distance_pc": 5.0,
                },
                "planet": {
                    "contrast": 5.0e-7,
                    "separation_mas": 20.0,
                    "position_angle_deg": 90.0,
                },
            },
            "observation": {"integration_time_s": 60.0},
            "pipeline": {"stages": [{"id": "scene", "type": "scene.point_source"}]},
            "output": {
                "plots": ["imaging_reconstruction"],
                "formats": {"plots": ["png"], "tables": ["json"]},
                "imaging_reconstruction": {
                    "n_pix": 24,
                    "fov_mas": 45.0,
                    "n_time": 3,
                    "hour_angle_span_h": 0.2,
                    "rotation_mode": "spacecraft_roll",
                    "roll_span_deg": 60.0,
                    "n_roll_steps": 3,
                    "declination_deg": 0.0,
                    "latitude_deg": 0.0,
                    "max_wavelengths": 2,
                    "poisson": False,
                    "save_arrays": True,
                    "compute_nnls": True,
                    "nnls_wavelength_mode": "resolved",
                },
            },
        }
    )
    state = _imaging_state()
    result = RunResult(
        state=state,
        diagnostics={},
        timings=(),
        cache_hit_count=0,
        config_hash="cfg",
    )
    collection = RunCollection(
        config=config,
        records=(
            RunRecord(
                config=config,
                result=result,
                run_dir=tmp_path / "cell",
                coord_tuple=(),
            ),
        ),
        output_dir=tmp_path,
    )

    written = imaging_reconstruction(collection, tmp_path, ("png",))

    plot_path = tmp_path / "plots" / "imaging_reconstruction.png"
    nnls_plot_path = tmp_path / "plots" / "imaging_reconstruction_nnls.png"
    json_path = tmp_path / "tables" / "imaging_reconstruction.json"
    array_path = tmp_path / "imaging" / "imaging_reconstruction.npz"
    assert plot_path in written
    assert plot_path.exists()
    assert plot_path.stat().st_size > 5_000
    assert nnls_plot_path in written
    assert nnls_plot_path.exists()
    assert nnls_plot_path.stat().st_size > 5_000
    assert json_path.exists()
    assert array_path.exists()
    arrays = np.load(array_path)
    assert arrays["truth_image"].shape == (24, 24)
    assert arrays["point_source_truth_image"].shape == (24, 24)
    assert arrays["exozodi_image"].shape == (24, 24)
    assert float(np.sum(arrays["exozodi_image"])) > 0.0
    assert float(np.sum(arrays["truth_image"])) > float(
        np.sum(arrays["point_source_truth_image"])
    )
    assert arrays["snr_map"].shape == (24, 24)
    assert arrays["matched_filter_beam"].shape == (24, 24)
    assert arrays["nnls_image"].shape == (24, 24)
    assert arrays["nnls_snr_map"].shape == (24, 24)
    assert arrays["nnls_measurements"].shape == (3, 2, 1)
    assert arrays["nnls_wavelengths_m"].shape == (2,)
    assert arrays["roll_angles_rad"].shape == (3,)
    assert float(np.nanmax(arrays["snr_map"])) > 0.0
    assert float(np.nanmax(arrays["nnls_image"])) > 0.0
    assert float(np.nanmax(arrays["nnls_snr_map"])) > 0.0
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    record = summary["records"][0]
    assert record["include_exozodi_glow"] is True
    assert record["sky_rate_hz_per_port"] == pytest.approx(0.06)
    assert record["exozodi_glow_total_contrast"] > 0.0
    assert record["nnls_converged"] is True


def test_imaging_planet_reflected_light_model_derives_contrast(tmp_path: Path) -> None:
    config = resolve_config(
        {
            "run": {"name": "imaging_reflected_test", "seed": 3},
            "grid": {
                "wavelength_center_um": 1.0,
                "wavelength_bandwidth_um": 0.1,
                "n_wavelength_bins": 3,
            },
            "scene": {
                "star": {
                    "jmag": 5.0,
                    "angular_diameter_mas": 1.0,
                    "distance_pc": 5.0,
                },
                "planet": {
                    "contrast": 5.0e-7,
                    "separation_mas": 20.0,
                    "position_angle_deg": 90.0,
                },
            },
            "observation": {"integration_time_s": 60.0},
            "pipeline": {"stages": [{"id": "scene", "type": "scene.point_source"}]},
            "output": {
                "imaging_reconstruction": {
                    "planets": [
                        {
                            "label": "Earth",
                            "separation_mas": 200.0,
                            "position_angle_deg": 60.0,
                            "flux_model": "reflected_light",
                            "radius_earth": 1.0,
                            "semi_major_axis_au": 1.0,
                            "geometric_albedo": 0.2,
                            "phase_angle_deg": 90.0,
                        }
                    ],
                },
            },
        }
    )
    result = RunResult(
        state=_imaging_state(),
        diagnostics={},
        timings=(),
        cache_hit_count=0,
        config_hash="cfg",
    )
    record = RunRecord(
        config=config,
        result=result,
        run_dir=tmp_path / "cell",
        coord_tuple=(),
    )

    planets = _planet_views(record, config.output.imaging_reconstruction)

    expected = reflected_light_contrast(
        radius_earth=1.0,
        semi_major_axis_au=1.0,
        geometric_albedo=0.2,
        phase_angle_rad=np.pi / 2.0,
    )
    assert len(planets) == 1
    assert planets[0].label == "Earth"
    assert planets[0].flux_model == "reflected_light"
    assert planets[0].contrast == pytest.approx(expected)
    assert planets[0].contrast_source == "reflected_light"


def test_reflected_light_imaging_planet_requires_semi_major_axis() -> None:
    with pytest.raises(ValidationError, match="semi_major_axis_au"):
        ImagingPlanetConfig.model_validate(
            {
                "label": "Earth",
                "separation_mas": 200.0,
                "position_angle_deg": 60.0,
                "flux_model": "reflected_light",
                "radius_earth": 1.0,
                "geometric_albedo": 0.2,
                "phase_angle_deg": 90.0,
            }
        )


@pytest.mark.parametrize(
    ("with_fringe_tracking", "expected_servo_bw_hz"),
    [(False, 0.0), (True, 2000.0)],
)
def test_imaging_inputs_use_physical_star_and_performance_systematic(
    tmp_path: Path,
    with_fringe_tracking: bool,
    expected_servo_bw_hz: float,
) -> None:
    # Saturated bright ports: the capped readout total (1e8 Hz) understates
    # the star; the planet and leakage scale with the physical 3.8e8 Hz. The
    # systematic is performance's combined eps (0.01), not the bare
    # sensitivity null_instability (0), and it is averaged over servo cycles
    # only when performance charged it that way.
    config = resolve_config(
        {
            "run": {"name": "imaging_inputs", "seed": 3},
            "grid": {
                "wavelength_center_um": 1.0,
                "wavelength_bandwidth_um": 0.1,
                "n_wavelength_bins": 3,
            },
            "scene": {
                "star": {"jmag": 5.0, "angular_diameter_mas": 1.0, "distance_pc": 5.0},
                "planet": {
                    "contrast": 5.0e-7,
                    "separation_mas": 20.0,
                    "position_angle_deg": 90.0,
                },
            },
            "observation": {"integration_time_s": 60.0},
            "pipeline": {"stages": [{"id": "scene", "type": "scene.point_source"}]},
            "output": {"imaging_reconstruction": {"n_pix": 8, "fov_mas": 45.0}},
        }
    )
    state = _imaging_state(
        performance_overrides={
            "stellar_rate_hz_total_uncapped": 3.8e8,
            "eps_systematic_used": 0.01,
            "with_fringe_tracking": with_fringe_tracking,
            "servo_bw_hz": 2000.0,
        }
    )
    record = RunRecord(
        config=config,
        result=RunResult(
            state=state, diagnostics={}, timings=(), cache_hit_count=0, config_hash="cfg"
        ),
        run_dir=tmp_path,
        coord_tuple=(),
    )

    inputs = _build_imaging_inputs(record, config.output.imaging_reconstruction)

    assert inputs.stellar_photon_rate_hz == pytest.approx(3.8e8)
    assert float(np.sum(inputs.stellar_rate_hz_by_wavelength)) == pytest.approx(3.8e8)
    assert inputs.eps_systematic == pytest.approx(0.01)
    assert inputs.servo_bw_hz == expected_servo_bw_hz


def _imaging_state(performance_overrides: dict | None = None) -> SimulationState:
    state = SimulationState.empty({"wavelength_grid_um": (0.95, 1.0, 1.05)})
    geometry = Geometry(
        diameter_m=1.0,
        area_m2=1.0,
        collecting_area_m2=2.0,
        effective_collecting_area_m2=2.0,
        n_apertures=2,
        aperture_positions_pupil_m=np.array(
            [[-5.0, 0.0], [5.0, 0.0]],
            dtype=np.float64,
        ),
    )
    scene = Scene(
        star_jmag=5.0,
        star_angular_diameter_mas=1.0,
        star_distance_pc=5.0,
        planet_contrast=5.0e-7,
        planet_separation_mas=20.0,
        planet_position_angle_deg=90.0,
    )
    unitary = np.array(
        [[1.0, 1.0], [1.0, -1.0]],
        dtype=np.complex128,
    ) / np.sqrt(2.0)
    u_stack = np.stack([unitary, unitary, unitary], axis=2)
    results = Results()
    for name, payload in (
        (
            "chip.kernel_mzi_mesh",
            {
                "u_stack": u_stack,
                "wavelength_grid_um": (0.95, 1.0, 1.05),
                "wl_design_um": 1.0,
                "n_modes": 2,
                "n_bright_ports": 1,
                "n_dark_ports": 1,
                "n_spectro_ports": 0,
                "active_chip_count": 1,
            },
        ),
        (
            "sensitivity.detection_curve",
            {
                "coupling_per_sub": np.ones(2, dtype=np.float64),
                "sub_areas_m2": np.ones(2, dtype=np.float64),
                "sub_diameters_m": np.ones(2, dtype=np.float64),
                "dark_indices": np.array([1], dtype=np.intp),
            },
        ),
        (
            "detector.ideal_counter",
            {
                "port_rates": np.array([1.0e8, 1.0e3], dtype=np.float64),
                "port_labels": ("bright_0", "dark_0"),
                "integration_time_s": 60.0,
                "null_depth_raw": 1.0e-5,
            },
        ),
        (
            "background.space_zodi",
            {
                "local_zodi_rate_hz_per_bin": np.array(
                    [0.03, 0.02, 0.01],
                    dtype=np.float64,
                ),
                "exozodi_rate_hz_per_bin": np.array(
                    [0.02, 0.015, 0.005],
                    dtype=np.float64,
                ),
                "zodi_rate_hz_per_bin": np.array(
                    [0.05, 0.035, 0.015],
                    dtype=np.float64,
                ),
                "local_zodi_rate_hz_per_dark_port": 0.06,
                "exozodi_rate_hz_per_dark_port": 0.04,
                "zodi_rate_hz_per_dark_port": 0.10,
                "background_rate_hz_per_dark_port": 0.10,
                "exozodi_effective_level_zodi": 2.0,
                "exozodi_inclination_deg": 35.0,
            },
        ),
        (
            "performance.standard",
            {
                "integration_time_s": 60.0,
                "stellar_rate_hz_total": 1.0e8,
                "null_depth_effective": 1.0e-5,
                "dark_rate_hz_per_port": 0.01,
                "sky_rate_hz_per_port": 0.1,
                "null_instability": 0.0,
                "servo_bw_hz": 0.0,
                **(performance_overrides or {}),
            },
        ),
    ):
        results = results.add_component(name, payload)
    photon_rates = PhotonRates(
        wavelength_grid_um=(0.95, 1.0, 1.05),
        port_labels=("bright_0", "dark_0"),
        rates_hz=np.array(
            [[5.0e7, 4.0e7, 1.0e7], [5.0e2, 4.0e2, 1.0e2]],
            dtype=np.float64,
        ),
    )
    return replace(
        state,
        geometry=geometry,
        scene=scene,
        photon_rates=photon_rates,
        results=results,
    )


def test_stellar_weights_use_true_wavelength_rates_not_mkid_redistributed() -> None:
    """Optical responses are evaluated at true wavelengths, so the spectral
    weights come from the detector's pre-redistribution, uncapped rates."""

    state = SimulationState.empty({"wavelength_grid_um": (0.95, 1.0, 1.05)})
    true_rates = np.array([[3.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    state = replace(
        state,
        photon_rates=PhotonRates(rates_hz=np.ones((2, 3))),  # MKID-smeared
        results=Results().add_component(
            "detector.mkid",
            {"pre_energy_response_port_rate_hz_per_wavelength": true_rates},
        ),
    )

    weights = _stellar_rates_by_wavelength(
        state=state, wavelength_indices=(0, 1, 2), total_rate_hz=6.0e3
    )

    np.testing.assert_allclose(weights, [4.0e3, 2.0e3, 0.0])
