"""Tests for the generic site-profile sky-background stage."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nullsim.physics.photometry import photon_rate_per_m2_per_bin
from nullsim.pipeline import Field, Geometry, RunContext, SimulationState, Throughput
from nullsim.stages.background import SkySurfaceBrightness
from nullsim.stages.background.sky_surface_brightness import (
    _single_mode_spaxel_arcsec2,
)


def _context(tmp_path: Path) -> RunContext:
    return RunContext(
        rng=np.random.default_rng(0),
        config_hash="background-sky-surface-brightness",
        sweep_coord_tuple=(),
        instance_id="background",
        run_dir=tmp_path,
    )


def _state() -> SimulationState:
    wavelength_grid_um = (1.20, 1.30)
    state = SimulationState.empty({"wavelength_grid_um": wavelength_grid_um})
    return replace(
        state,
        field=Field(
            amplitudes=np.ones((2, 2, 1), dtype=np.complex128),
            wavelength_grid_um=wavelength_grid_um,
        ),
        geometry=Geometry(
            area_m2=30.0,
            collecting_area_m2=30.0,
            effective_collecting_area_m2=20.0,
            n_apertures=1,
        ),
        throughput=Throughput(
            wavelength_grid_um=wavelength_grid_um,
            transmission_fraction=(0.5, 0.25),
        ),
    )


def test_generic_sky_stage_matches_explicit_surface_brightness(tmp_path: Path) -> None:
    state = _state()
    stage = SkySurfaceBrightness(
        {
            "site_profile": "cerro_armazones",
            "band": "J",
            "sky_mag_arcsec2": 15.5,
            "spaxel_arcsec2": 0.005,
        },
        _context(tmp_path),
    )

    result = stage.apply(state)
    payload = result.results.components["background.sky_surface_brightness"]

    expected = np.array(
        [
            photon_rate_per_m2_per_bin(15.5, 1.20, 0.10, band="J"),
            photon_rate_per_m2_per_bin(15.5, 1.30, 0.10, band="J"),
        ],
        dtype=np.float64,
    )
    expected *= 10.0 * 0.005 * np.array([0.5, 0.25])

    np.testing.assert_allclose(payload["sky_rate_hz_per_bin"], expected)
    assert payload["site_profile"] == "cerro_armazones"
    assert payload["sky_rate_hz_per_dark_port"] == pytest.approx(float(expected.sum()))


def test_generic_sky_stage_does_not_reapply_field_losses_per_physical_pol_arm(
    tmp_path: Path,
) -> None:
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
                },
            },
        ),
    )
    stage = SkySurfaceBrightness(
        {
            "site_profile": "cerro_armazones",
            "band": "J",
            "sky_mag_arcsec2": 15.5,
            "spaxel_arcsec2": 0.005,
        },
        _context(tmp_path),
    )

    result = stage.apply(state)
    payload = result.results.components["background.sky_surface_brightness"]

    expected = np.array(
        [
            photon_rate_per_m2_per_bin(15.5, 1.20, 0.10, band="J"),
            photon_rate_per_m2_per_bin(15.5, 1.30, 0.10, band="J"),
        ],
        dtype=np.float64,
    )
    expected *= 10.0 * 0.005 * 0.5

    np.testing.assert_allclose(payload["throughput_fraction"], [0.5, 0.5])
    np.testing.assert_allclose(payload["sky_rate_hz_per_bin"], expected)


def test_generic_sky_stage_debits_field_applied_delay_loss(tmp_path: Path) -> None:
    state = _state()
    state = replace(
        state,
        throughput=replace(
            state.throughput,
            transmission_fraction=(1.0, 1.0),
            components={
                "delay_line_loss": {
                    "transmission_fraction": (0.7, 0.7),
                    "source_model": (
                        "KILO geometric delay-line path equalization; "
                        "loss applied to field amplitudes"
                    ),
                },
            },
        ),
    )
    stage = SkySurfaceBrightness(
        {
            "site_profile": "cerro_armazones",
            "band": "J",
            "sky_mag_arcsec2": 15.5,
            "spaxel_arcsec2": 0.005,
        },
        _context(tmp_path),
    )

    result = stage.apply(state)
    payload = result.results.components["background.sky_surface_brightness"]

    expected = np.array(
        [
            photon_rate_per_m2_per_bin(15.5, 1.20, 0.10, band="J"),
            photon_rate_per_m2_per_bin(15.5, 1.30, 0.10, band="J"),
        ],
        dtype=np.float64,
    )
    expected *= 10.0 * 0.005 * 0.7

    np.testing.assert_allclose(payload["throughput_fraction"], [0.7, 0.7])
    np.testing.assert_allclose(payload["sky_rate_hz_per_bin"], expected)


def test_site_profile_changes_default_sky_brightness(tmp_path: Path) -> None:
    state = _state()
    cerro = SkySurfaceBrightness(
        {"site_profile": "cerro_armazones", "band": "J"},
        _context(tmp_path),
    ).apply(state)
    maunakea = SkySurfaceBrightness(
        {"site_profile": "maunakea", "band": "J"},
        _context(tmp_path),
    ).apply(state)

    cerro_payload = cerro.results.components["background.sky_surface_brightness"]
    maunakea_payload = maunakea.results.components["background.sky_surface_brightness"]
    assert maunakea_payload["sky_mag_arcsec2"] > cerro_payload["sky_mag_arcsec2"]
    assert (
        maunakea_payload["sky_rate_hz_per_dark_port"]
        < cerro_payload["sky_rate_hz_per_dark_port"]
    )


def test_single_mode_spaxel_etendue_keck_subaperture() -> None:
    # Omega = n_pol * lambda^2 / A; A = 2.93 m^2, lambda = 1.25 um, n_pol = 2.
    # 1 sr = (206264.806)^2 arcsec^2 = 4.25452e10.
    spaxel = _single_mode_spaxel_arcsec2(
        area_per_sub_m2=2.93,
        wavelength_center_m=1.25e-6,
        n_polarizations=2,
    )
    expected = 2 * (1.25e-6) ** 2 / 2.93 * (206264.806) ** 2
    assert spaxel == pytest.approx(expected, rel=1e-6)


def test_default_spaxel_is_single_mode_etendue(tmp_path: Path) -> None:
    """With no spaxel override the stage uses the single-mode etendue."""

    state = _state()  # effective_collecting_area_m2 = 20.0, n_apertures = 1
    stage = SkySurfaceBrightness(
        {"site_profile": "cerro_armazones", "band": "J"},
        _context(tmp_path),
    )
    payload = stage.apply(state).results.components[
        "background.sky_surface_brightness"
    ]

    # area_per_sub = effective_collecting_area / n_subapertures.
    # n_subapertures defaults to the number of chip input modes (2 here).
    area_per_sub_m2 = payload["area_per_sub_m2"]
    wavelength_center_m = 1.30e-6  # central bin of (1.20, 1.30)
    expected = _single_mode_spaxel_arcsec2(
        area_per_sub_m2=area_per_sub_m2,
        wavelength_center_m=wavelength_center_m,
        n_polarizations=2,
    )
    assert payload["spaxel_source"] == "single_mode_etendue"
    assert payload["spaxel_arcsec2"] == pytest.approx(expected, rel=1e-9)
    assert payload["n_polarizations"] == 2


def test_explicit_spaxel_override_takes_precedence(tmp_path: Path) -> None:
    state = _state()
    stage = SkySurfaceBrightness(
        {"site_profile": "cerro_armazones", "band": "J", "spaxel_arcsec2": 0.005},
        _context(tmp_path),
    )
    payload = stage.apply(state).results.components[
        "background.sky_surface_brightness"
    ]
    assert payload["spaxel_source"] == "param.spaxel_arcsec2"
    assert payload["spaxel_arcsec2"] == pytest.approx(0.005)


def test_sky_rate_scaled_by_splitter_input_fraction(tmp_path: Path) -> None:
    """After an injection.mode_splitter fanout, the total sky across chip
    modes must equal the sky entering the physical fibers: each port gets
    n_input/n_output of a full single-mode étendue (review 2026-06-10
    Issue 4)."""

    base_state = _state()
    stage_args = {
        "site_profile": "maunakea",
        "band": "J",
        "sky_mag_arcsec2": 16.0,
        "spaxel_arcsec2": 0.005,
    }

    baseline = SkySurfaceBrightness(stage_args, _context(tmp_path)).apply(base_state)
    base_rate = baseline.results.components[
        "background.sky_surface_brightness"
    ]["sky_rate_hz_per_dark_port"]

    split_state = replace(
        base_state,
        geometry=replace(
            base_state.geometry,
            components={
                "quadrature_split": {
                    "n_input_modes": 2,
                    "n_output_modes": 4,
                    "copies": 2,
                }
            },
        ),
    )
    split = SkySurfaceBrightness(stage_args, _context(tmp_path)).apply(split_state)
    split_rate = split.results.components[
        "background.sky_surface_brightness"
    ]["sky_rate_hz_per_dark_port"]

    assert split_rate == pytest.approx(0.5 * base_rate)


def test_sky_rate_not_debited_by_stellar_strehl(tmp_path: Path) -> None:
    """Point-source coupling factors (Strehl x obscuration overlap) must not
    debit the diffuse sky; only the component's true transmission part
    (Fresnel) applies (review 2026-06-10 Issue 5)."""

    base_state = _state()
    stage_args = {
        "site_profile": "maunakea",
        "band": "J",
        "sky_mag_arcsec2": 16.0,
        "spaxel_arcsec2": 0.005,
    }
    baseline = SkySurfaceBrightness(stage_args, _context(tmp_path)).apply(base_state)
    base_rate = baseline.results.components[
        "background.sky_surface_brightness"
    ]["sky_rate_hz_per_dark_port"]

    injected = replace(
        base_state,
        throughput=base_state.throughput.add_component(
            "injection_ideal",
            {
                "wavelength_grid_um": (1.20, 1.30),
                # Strehl-dominated coupling that must NOT reach the sky...
                "transmission_fraction": (0.30, 0.35),
                "field_amplitude_applied": True,
                "point_source_coupling_only": True,
                # ...while the Fresnel part does.
                "diffuse_transmission_fraction": (0.96, 0.96),
            },
        ),
    )
    debited = SkySurfaceBrightness(stage_args, _context(tmp_path)).apply(injected)
    debited_rate = debited.results.components[
        "background.sky_surface_brightness"
    ]["sky_rate_hz_per_dark_port"]

    assert debited_rate == pytest.approx(0.96 * base_rate)
