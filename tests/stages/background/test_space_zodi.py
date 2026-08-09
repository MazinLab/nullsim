"""Tests for the space zodiacal-background stage."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nullsim.physics.photometry import photon_rate_per_m2_per_bin
from nullsim.pipeline import (
    Field,
    Geometry,
    RunContext,
    Scene,
    SimulationState,
    Throughput,
)
from nullsim.stages.background import SpaceZodiBackground


def _context(tmp_path: Path) -> RunContext:
    return RunContext(
        rng=np.random.default_rng(0),
        config_hash="background-space-zodi",
        sweep_coord_tuple=(),
        instance_id="zodi",
        run_dir=tmp_path,
    )


def _state() -> SimulationState:
    wavelength_grid_um = (0.78, 0.82)
    state = SimulationState.empty({"wavelength_grid_um": wavelength_grid_um})
    return replace(
        state,
        scene=Scene(
            star_jmag=8.0,
            star_spectral_type="M3V",
            planet_separation_mas=100.0,
            exozodi_level_zodi=2.0,
        ),
        field=Field(
            amplitudes=np.ones((2, 2, 1), dtype=np.complex128),
            wavelength_grid_um=wavelength_grid_um,
        ),
        geometry=Geometry(
            diameter_m=8.0,
            area_m2=30.0,
            collecting_area_m2=30.0,
            effective_collecting_area_m2=20.0,
            n_apertures=1,
        ),
        throughput=Throughput(
            wavelength_grid_um=wavelength_grid_um,
            transmission_fraction=(0.5, 0.25),
        ),
        results=(
            state.results.add_component(
                "detector.mkid",
                {"port_labels": ("bright_0", "dark_0")},
            ).add_component(
                "sensitivity.detection_curve",
                {
                    "separations_mas": (100.0,),
                    "planet_throughputs": np.array([0.2], dtype=np.float64),
                },
            )
        ),
    )


def test_space_zodi_uses_local_and_scene_exozodi_levels(tmp_path: Path) -> None:
    state = _state()
    stage = SpaceZodiBackground(
        {
            "band": "I",
            "local_zodi_level": 1.5,
            "local_zodi_v_mag_arcsec2": 23.0,
            "exozodi_v_mag_arcsec2": 22.0,
            "spaxel_arcsec2": 0.01,
        },
        _context(tmp_path),
    )

    result = stage.apply(state)
    payload = result.results.components["background.space_zodi"]

    local_mag_i = 23.0 - 0.69
    host_star_i_mag = 9.4
    exozodi_mag_i = host_star_i_mag + (22.0 - 4.83)
    local_rate = np.array(
        [
            photon_rate_per_m2_per_bin(local_mag_i, 0.78, 0.04, band="I"),
            photon_rate_per_m2_per_bin(local_mag_i, 0.82, 0.04, band="I"),
        ],
        dtype=np.float64,
    )
    exozodi_rate = np.array(
        [
            photon_rate_per_m2_per_bin(exozodi_mag_i, 0.78, 0.04, band="I"),
            photon_rate_per_m2_per_bin(exozodi_mag_i, 0.82, 0.04, band="I"),
        ],
        dtype=np.float64,
    )
    common = 20.0 * 0.01 * np.array([0.5, 0.25]) * 0.2
    expected_local = local_rate * 1.5 * common
    expected_exozodi = exozodi_rate * 2.0 * common

    np.testing.assert_allclose(payload["local_zodi_rate_hz_per_bin"], expected_local)
    np.testing.assert_allclose(payload["exozodi_rate_hz_per_bin"], expected_exozodi)
    assert payload["host_star_mag_for_exozodi"] == pytest.approx(host_star_i_mag)
    assert payload["derive_exozodi_mag_from_scene_jmag"] is True
    assert payload["local_zodi_rate_hz_per_dark_port"] == pytest.approx(
        float(expected_local.sum())
    )
    assert payload["exozodi_rate_hz_per_dark_port"] == pytest.approx(
        float(expected_exozodi.sum())
    )
    assert payload["background_rate_hz_per_dark_port"] == pytest.approx(
        float(expected_local.sum() + expected_exozodi.sum())
    )
    np.testing.assert_allclose(payload["background_separations_mas"], [100.0])
    np.testing.assert_allclose(payload["background_routing_throughputs"], [0.2])
    np.testing.assert_allclose(
        payload["local_zodi_rate_hz_per_dark_port_per_sep"],
        [float(expected_local.sum())],
    )
    np.testing.assert_allclose(
        payload["exozodi_rate_hz_per_dark_port_per_sep"],
        [float(expected_exozodi.sum())],
    )
    np.testing.assert_allclose(
        payload["background_rate_hz_per_dark_port_per_sep"],
        [float(expected_local.sum() + expected_exozodi.sum())],
    )


def test_space_zodi_can_sample_hosts_level_and_inclined_morphology(
    tmp_path: Path,
) -> None:
    state = _state()
    stage = SpaceZodiBackground(
        {
            "band": "I",
            "local_zodi_level": 0.0,
            "exozodi_level_distribution": "hosts2020_lognormal",
            "exozodi_hosts2020_median_zodi": 3.0,
            "exozodi_hosts2020_sigma_dex": 0.0,
            "exozodi_morphology": "smooth_inclined_disk",
            "exozodi_inclination_deg": 60.0,
            "spaxel_arcsec2": 0.01,
        },
        _context(tmp_path),
    )

    result = stage.apply(state)
    payload = result.results.components["background.space_zodi"]

    host_star_i_mag = 9.4
    exozodi_mag_i = host_star_i_mag + (22.0 - 4.83)
    exozodi_rate = np.array(
        [
            photon_rate_per_m2_per_bin(exozodi_mag_i, 0.78, 0.04, band="I"),
            photon_rate_per_m2_per_bin(exozodi_mag_i, 0.82, 0.04, band="I"),
        ],
        dtype=np.float64,
    )
    common = 20.0 * 0.01 * np.array([0.5, 0.25]) * 0.2
    expected_exozodi = exozodi_rate * 3.0 * 2.0 * common

    assert payload["exozodi_level_source"] == "hosts2020_lognormal"
    assert payload["exozodi_level_zodi"] == pytest.approx(3.0)
    assert payload["exozodi_inclination_deg"] == pytest.approx(60.0)
    assert payload["exozodi_surface_brightness_multiplier"] == pytest.approx(2.0)
    assert payload["exozodi_effective_level_zodi"] == pytest.approx(6.0)
    np.testing.assert_allclose(payload["exozodi_rate_hz_per_bin"], expected_exozodi)


def test_space_zodi_publishes_background_vector_on_sensitivity_grid(
    tmp_path: Path,
) -> None:
    state = _state()
    sensitivity_payload = {
        "separations_mas": (50.0, 100.0),
        "planet_throughputs": np.array([0.1, 0.5], dtype=np.float64),
    }
    components = dict(state.results.components)
    components["sensitivity.detection_curve"] = sensitivity_payload
    state = replace(
        state,
        scene=replace(state.scene, planet_separation_mas=75.0),
        results=replace(state.results, components=components),
    )
    stage = SpaceZodiBackground(
        {
            "band": "I",
            "local_zodi_v_mag_arcsec2": 23.0,
            "exozodi_level_zodi": 0.0,
            "spaxel_arcsec2": 0.01,
        },
        _context(tmp_path),
    )

    result = stage.apply(state)
    payload = result.results.components["background.space_zodi"]

    vector = np.asarray(
        payload["background_rate_hz_per_dark_port_per_sep"],
        dtype=np.float64,
    )
    np.testing.assert_allclose(payload["background_separations_mas"], [50.0, 100.0])
    np.testing.assert_allclose(payload["background_routing_throughputs"], [0.1, 0.5])
    assert payload["routing_throughput"] == pytest.approx(0.3)
    assert payload["background_rate_hz_per_dark_port"] == pytest.approx(
        float(vector[0]) * 3.0
    )
    assert float(vector[1]) == pytest.approx(float(vector[0]) * 5.0)


def test_space_zodi_can_use_absolute_exozodi_surface_brightness(
    tmp_path: Path,
) -> None:
    state = _state()
    stage = SpaceZodiBackground(
        {
            "band": "I",
            "local_zodi_level": 0.0,
            "exozodi_level_zodi": 1.0,
            "exozodi_v_mag_arcsec2": 22.0,
            "derive_exozodi_mag_from_scene_jmag": False,
            "spaxel_arcsec2": 0.01,
        },
        _context(tmp_path),
    )

    result = stage.apply(state)
    payload = result.results.components["background.space_zodi"]

    exozodi_mag_i = 22.0 - 0.69
    expected = np.array(
        [
            photon_rate_per_m2_per_bin(exozodi_mag_i, 0.78, 0.04, band="I"),
            photon_rate_per_m2_per_bin(exozodi_mag_i, 0.82, 0.04, band="I"),
        ],
        dtype=np.float64,
    )
    expected *= 20.0 * 0.01 * np.array([0.5, 0.25]) * 0.2

    assert payload["derive_exozodi_mag_from_scene_jmag"] is False
    assert payload["host_star_mag_for_exozodi"] is None
    np.testing.assert_allclose(payload["exozodi_rate_hz_per_bin"], expected)


def test_space_zodi_derives_lambda_over_d_spaxel(tmp_path: Path) -> None:
    stage = SpaceZodiBackground(
        {"band": "I", "local_zodi_level": 0.0, "exozodi_level_zodi": 0.0},
        _context(tmp_path),
    )

    result = stage.apply(_state())
    payload = result.results.components["background.space_zodi"]

    expected_lam_over_d_arcsec = 0.8e-6 / 8.0 * 206_264.80624709636
    assert payload["spaxel_arcsec2"] == pytest.approx(
        expected_lam_over_d_arcsec**2
    )
    assert payload["background_rate_hz_per_dark_port"] == pytest.approx(0.0)


def test_space_zodi_does_not_double_count_dual_pol_field_losses(tmp_path: Path) -> None:
    state = _state()
    state = replace(
        state,
        field=replace(
            state.field,
            amplitudes=np.ones((2, 2, 2), dtype=np.complex128),
            components={"polarization.pbs_split": {"polarization_labels": ("A", "B")}},
        ),
        throughput=replace(
            state.throughput,
            transmission_fraction=(1.0, 1.0),
            components={
                "polarization.pbs_split": {
                    "transmission_fraction": (0.8, 0.6),
                },
                "chip.kernel_mzi_mesh_loss": {
                    "transmission_fraction": (0.5, 0.5),
                    "source_model": "applied to field amplitudes",
                },
            },
        ),
    )
    stage = SpaceZodiBackground(
        {
            "band": "I",
            "local_zodi_v_mag_arcsec2": 23.0,
            "exozodi_level_zodi": 0.0,
            "spaxel_arcsec2": 0.01,
        },
        _context(tmp_path),
    )

    result = stage.apply(state)
    payload = result.results.components["background.space_zodi"]

    local_mag_i = 23.0 - 0.69
    expected = np.array(
        [
            photon_rate_per_m2_per_bin(local_mag_i, 0.78, 0.04, band="I"),
            photon_rate_per_m2_per_bin(local_mag_i, 0.82, 0.04, band="I"),
        ],
        dtype=np.float64,
    )
    expected *= 20.0 * 0.01 * 0.5 * 0.2

    np.testing.assert_allclose(payload["throughput_fraction"], [0.5, 0.5])
    np.testing.assert_allclose(payload["zodi_rate_hz_per_bin"], expected)
