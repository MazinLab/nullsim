"""Cross-stage smoke tests for the first real stage pass."""

from __future__ import annotations

import importlib
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
    Stage,
    build_instance,
)

WAVELENGTH_GRID_UM = (1.2, 1.4, 1.6, 1.8)
PASS2_AUTOLOAD_STAGE_TYPES = {
    "ao.pyramid_wfs",
    "atmosphere.kolmogorov",
    "injection.single_mode_fiber",
    "transport.smf28",
    "delay_lines.geometric",
}


def test_atmosphere_injection_transport_delay_lines_smoke(tmp_path: Path) -> None:
    importlib.import_module("nullsim.stages._autoload")
    state = _synthetic_two_aperture_state()

    atmosphere = _stage(
        "atmosphere",
        "atmosphere.kolmogorov",
        tmp_path,
        r0_500nm_m=0.5,
        seeing_arcsec=0.2,
        wind_speed_m_s=8.0,
    )
    injection = _stage("injection", "injection.single_mode_fiber", tmp_path)
    transport = _stage(
        "fiber_to_chip",
        "transport.smf28",
        tmp_path,
        length_m=10.0,
        attenuation_db_per_km=0.25,
        phi_rms_rad=0.02,
        rng_seed=7,
        length_matching_error_m=0.001,
    )
    delay_lines = _stage(
        "delay_lines",
        "delay_lines.geometric",
        tmp_path,
        max_delay_m=20.0,
    )

    atmosphere_state = atmosphere.apply(state)
    assert "atmosphere" in atmosphere_state.wavefront.components
    assert "atmos_piston" in atmosphere_state.opd_budget.components

    injected_state = injection.apply(atmosphere_state)
    assert injected_state.field.amplitudes is not None
    assert np.all(np.abs(injected_state.field.amplitudes) <= 1.0)
    assert "single_mode_fiber" in injected_state.throughput.components

    transported_state = transport.apply(injected_state)
    assert "fiber_phase_common_mode" in transported_state.opd_budget.components
    assert "fiber_chromatic_dispersion" in transported_state.opd_budget.components

    delayed_state = delay_lines.apply(transported_state)
    assert "delay_line_loss" in delayed_state.throughput.components
    assert delayed_state.throughput.transmission_fraction


def _stage(
    instance_id: str,
    type_name: str,
    tmp_path: Path,
    **params: object,
) -> Stage:
    return build_instance(
        {"id": instance_id, "type": type_name, **params},
        _context(tmp_path, instance_id),
    )


def _context(tmp_path: Path, instance_id: str) -> RunContext:
    return RunContext(
        rng=np.random.default_rng(11),
        config_hash="pass2-smoke",
        sweep_coord_tuple=(),
        instance_id=instance_id,
        run_dir=tmp_path,
    )


def _synthetic_two_aperture_state() -> SimulationState:
    aperture_diameter_m = 2.0
    aperture_positions_enu_m = np.array(
        [
            [-5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    field_amplitudes = np.ones(
        (2, len(WAVELENGTH_GRID_UM), 1),
        dtype=np.complex128,
    )
    state = SimulationState.empty({"wavelength_grid_um": WAVELENGTH_GRID_UM})
    scene = Scene(
        star_jmag=5.0,
        altitude_deg=60.0,
        azimuth_deg=90.0,
    ).add_component(
        "science_target",
        {
            "star_jmag": 5.0,
            "altitude_deg": 60.0,
            "azimuth_deg": 90.0,
        },
    )
    geometry = Geometry(
        diameter_m=aperture_diameter_m,
        area_m2=math.pi * (aperture_diameter_m / 2.0) ** 2,
        n_apertures=2,
        aperture_positions_enu_m=aperture_positions_enu_m,
    ).add_component(
        "synthetic_array",
        {
            "diameter_m": aperture_diameter_m,
            "n_apertures": 2,
            "aperture_positions_enu_m": aperture_positions_enu_m,
        },
    )
    field = Field(
        amplitudes=field_amplitudes,
        wavelength_grid_um=WAVELENGTH_GRID_UM,
        mode_labels=("aperture_0", "aperture_1"),
    )
    return replace(state, scene=scene, geometry=geometry, field=field)
