"""Tests for the NAYRA sky/background stage."""

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
    SimulationState,
    Throughput,
    list_stages,
)
from nullsim.stages.background import NayraSkyBackground


def _context(tmp_path: Path) -> RunContext:
    return RunContext(
        rng=np.random.default_rng(0),
        config_hash="background-nayra-sky",
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


def test_nayra_sky_registers_stage_contract(tmp_path: Path) -> None:
    stage = NayraSkyBackground({"band": "J"}, _context(tmp_path))

    assert "background.nayra_sky" in list_stages()
    assert stage.type_name == "background.nayra_sky"
    assert stage.family == "background"
    assert stage.consumes == frozenset({"geometry", "throughput", "field"})
    assert stage.produces == frozenset({"results"})


def test_nayra_sky_rate_uses_effective_area_grid_and_throughput(
    tmp_path: Path,
) -> None:
    state = _state()
    stage = NayraSkyBackground(
        {
            "band": "J",
            "sky_mag_arcsec2": 15.5,
            "spaxel_arcsec2": 0.005,
        },
        _context(tmp_path),
    )

    result = stage.apply(state)
    payload = result.results.components["background.nayra_sky"]

    # Two wavelength centers have bin widths 0.1 um each. Effective area
    # 20 m^2 over two sub-apertures gives 10 m^2 per sub-aperture.
    expected = np.array(
        [
            photon_rate_per_m2_per_bin(15.5, 1.20, 0.10, band="J"),
            photon_rate_per_m2_per_bin(15.5, 1.30, 0.10, band="J"),
        ],
        dtype=np.float64,
    )
    expected *= 10.0 * 0.005 * np.array([0.5, 0.25])

    np.testing.assert_allclose(payload["sky_rate_hz_per_bin"], expected)
    assert payload["area_per_sub_m2"] == pytest.approx(10.0)
    assert payload["sky_rate_hz_per_dark_port"] == pytest.approx(float(expected.sum()))
    assert payload["background_rate_hz_per_spectro_port"] == pytest.approx(
        float(expected.sum())
    )


def test_nayra_sky_does_not_reapply_field_losses(tmp_path: Path) -> None:
    state = _state()
    state = replace(
        state,
        field=Field(
            amplitudes=np.ones((2, 2, 2), dtype=np.complex128),
            wavelength_grid_um=(1.20, 1.30),
            components={
                "polarization.pbs_split": {
                    "polarization_labels": ("A", "B"),
                },
            },
        ),
        throughput=Throughput(
            wavelength_grid_um=(1.20, 1.30),
            transmission_fraction=(0.5, 0.25),
            components={
                "polarization.pm_fiber": {"transmission_fraction": 0.25},
                "chip.kernel_mzi_mesh_loss": {
                    "transmission_fraction": (0.5, 0.5),
                    "field_amplitude_applied": True,
                    "source_model": "chip scalar insertion loss",
                },
            },
        ),
    )
    stage = NayraSkyBackground(
        {
            "band": "J",
            "sky_mag_arcsec2": 15.5,
            "spaxel_arcsec2": 0.005,
        },
        _context(tmp_path),
    )

    result = stage.apply(state)
    payload = result.results.components["background.nayra_sky"]

    expected = np.array(
        [
            photon_rate_per_m2_per_bin(15.5, 1.20, 0.10, band="J"),
            photon_rate_per_m2_per_bin(15.5, 1.30, 0.10, band="J"),
        ],
        dtype=np.float64,
    )
    expected *= 10.0 * 0.005 * np.array([0.5, 0.25]) * 0.5 * 0.5

    np.testing.assert_allclose(payload["sky_rate_hz_per_bin"], expected)


def test_nayra_sky_adds_thermal_and_exozodi_terms(tmp_path: Path) -> None:
    state = _state()
    stage = NayraSkyBackground(
        {
            "band": "J",
            "thermal_background_hz_per_port": 2.0,
            "exozodi_background_hz_per_port": 3.0,
        },
        _context(tmp_path),
    )

    result = stage.apply(state)
    payload = result.results.components["background.nayra_sky"]

    assert payload["background_rate_hz_per_dark_port"] == pytest.approx(
        payload["sky_rate_hz_per_dark_port"] + 5.0
    )
