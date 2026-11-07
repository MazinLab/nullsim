"""Tests for the generic site-profile sky-background stage."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nullsim.physics.photometry import photon_rate_per_m2_per_bin
from nullsim.pipeline import (
    Field,
    Geometry,
    RunContext,
    SimulationState,
    Throughput,
    Wavefront,
)
from nullsim.stages.background import SkySurfaceBrightness
from nullsim.stages.background.sky_surface_brightness import (
    _single_mode_spaxel_arcsec2,
)
from nullsim.stages.injection.single_mode_fiber import SingleModeFiber


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


# Fraction of a Gaussian mode (aperture radius / mode radius = 1.1209) that a
# circular aperture passes: the diffuse single-mode acceptance.
_ETA_DIFFUSE = 1.0 - math.exp(-2.0 * 1.120906422778534**2)


def test_single_mode_spaxel_etendue_keck_subaperture() -> None:
    # Omega = eta_d * lambda^2 / A; A = 2.93 m^2, lambda = 1.25 um.
    # B lambda^2 already counts both polarizations (Issue 12).
    spaxel = _single_mode_spaxel_arcsec2(
        area_per_sub_m2=2.93,
        wavelength_m=1.25e-6,
        n_polarizations=1,
    )
    expected = 0.9190 * (1.25e-6) ** 2 / 2.93 * (206264.806) ** 2
    assert spaxel == pytest.approx(expected, rel=1e-4)


def test_default_spaxel_is_single_mode_etendue_per_bin(tmp_path: Path) -> None:
    """With no override each bin uses its own lambda^2 (row 40), one
    polarization count (Issue 12), and the 0.919 acceptance (Issue 2b)."""

    state = _state()  # effective_collecting_area_m2 = 20.0, 2 input modes
    stage = SkySurfaceBrightness(
        {"site_profile": "cerro_armazones", "band": "J", "sky_mag_arcsec2": 15.5},
        _context(tmp_path),
    )
    payload = stage.apply(state).results.components[
        "background.sky_surface_brightness"
    ]

    wavelength_m = np.array([1.20e-6, 1.30e-6])
    np.testing.assert_allclose(
        payload["spaxel_arcsec2_per_bin"],
        _ETA_DIFFUSE * wavelength_m**2 / 10.0 * 206264.80624709636**2,
        rtol=1e-12,
    )
    assert payload["spaxel_source"] == "single_mode_etendue"
    assert payload["n_polarizations"] == 1

    # Area cancels: each input collects B * eta_d * lambda^2 per bin.
    expected = np.array(
        [
            photon_rate_per_m2_per_bin(15.5, 1.20, 0.10, band="J"),
            photon_rate_per_m2_per_bin(15.5, 1.30, 0.10, band="J"),
        ]
    ) * _ETA_DIFFUSE * wavelength_m**2 * 206264.80624709636**2 * np.array([0.5, 0.25])
    np.testing.assert_allclose(payload["sky_rate_hz_per_bin"], expected, rtol=1e-12)


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


def _split_state(chip: dict | None) -> SimulationState:
    """Two physical inputs fanned out to four chip modes [0, 1, 0, 1]."""

    base = _state()
    results = base.results
    if chip is not None:
        results = results.add_component("chip.kernel_mzi_mesh", chip)
    return replace(
        base,
        field=replace(base.field, amplitudes=np.ones((4, 2, 1), dtype=np.complex128)),
        geometry=replace(
            base.geometry,
            components={
                "quadrature_split": {
                    "n_input_modes": 2,
                    "n_output_modes": 4,
                    "copies": 2,
                }
            },
        ),
        results=results,
    )


_SKY_ARGS = {
    "site_profile": "maunakea",
    "band": "J",
    "sky_mag_arcsec2": 16.0,
    "spaxel_arcsec2": 0.005,
}


def _sky(state: SimulationState, tmp_path: Path) -> dict:
    return SkySurfaceBrightness(_SKY_ARGS, _context(tmp_path)).apply(
        state
    ).results.components["background.sky_surface_brightness"]


def test_sky_without_chip_splits_each_input_over_its_copies(tmp_path: Path) -> None:
    """No chip: each copy carries 1/copies of its physical input's sky."""

    base_rate = _sky(_state(), tmp_path)["sky_rate_hz_per_dark_port"]
    split = _sky(_split_state(None), tmp_path)
    assert split["sky_rate_hz_per_dark_port"] == pytest.approx(0.5 * base_rate)
    np.testing.assert_allclose(split["sky_routing_fraction_per_port"], 0.5)


def test_sky_routes_coherent_copies_through_the_chip(tmp_path: Path) -> None:
    """Copies of one input are coherent, so sky in output j is
    sum_k |(U S)_jk|^2, not n_in/n_out (Issue 11). Rows that add the two
    copies of an input in phase collect the whole input; rows that subtract
    them collect none."""

    root_half = 1.0 / math.sqrt(2.0)
    unitary = root_half * np.array(
        [
            [1.0, 0.0, 1.0, 0.0],   # input 0 copies in phase -> 1
            [0.0, 1.0, 0.0, 1.0],   # input 1 copies in phase -> 1
            [1.0, 0.0, -1.0, 0.0],  # -> 0
            [0.0, 1.0, 0.0, -1.0],  # -> 0
        ],
        dtype=np.complex128,
    )
    chip = {
        "u_stack": np.repeat(unitary[:, :, np.newaxis], 2, axis=2),
        "n_bright_ports": 0,
        "n_dark_ports": 2,
    }
    base_rate = _sky(_state(), tmp_path)["sky_rate_hz_per_dark_port"]
    payload = _sky(_split_state(chip), tmp_path)

    np.testing.assert_allclose(
        payload["sky_routing_fraction_per_port"][:, 0], [1.0, 1.0, 0.0, 0.0], atol=1e-12
    )
    assert payload["sky_rate_hz_per_dark_port"] == pytest.approx(base_rate)
    # Spectro keys carry the all-output mean, n_in / n_out for a unitary.
    assert payload["sky_rate_hz_per_spectro_port"] == pytest.approx(0.5 * base_rate)

    # Per-PBS-arm deployment: a pi trim on one copy in arm B swaps which rows
    # add in phase, so the arm-averaged dark share is 1/2.
    trim = np.diag([1.0, 1.0, -1.0, 1.0]).astype(np.complex128)
    arm_stacks = np.stack([unitary, unitary @ trim])[:, :, :, np.newaxis]
    per_arm = chip | {"u_stack_per_arm": np.repeat(arm_stacks, 2, axis=3)}
    arm_payload = _sky(_split_state(per_arm), tmp_path)
    np.testing.assert_allclose(
        arm_payload["sky_routing_fraction_per_port"][:, 0],
        [0.5, 1.0, 0.5, 0.0],
        atol=1e-12,
    )


def test_sky_rate_not_debited_by_smf_coupling(tmp_path: Path) -> None:
    """injection.single_mode_fiber's Strehl x eta coupling is point-source
    only and must not reach the diffuse sky (row 42)."""

    base = _state()
    injected = SingleModeFiber({}, _context(tmp_path)).apply(
        replace(
            base,
            geometry=replace(base.geometry, diameter_m=1.0),
            wavefront=Wavefront(wavelength_grid_um=(1.20, 1.30), strehl_ratio=(0.3, 0.4)),
        )
    )
    assert _sky(injected, tmp_path)["sky_rate_hz_per_dark_port"] == pytest.approx(
        _sky(base, tmp_path)["sky_rate_hz_per_dark_port"]
    )


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
