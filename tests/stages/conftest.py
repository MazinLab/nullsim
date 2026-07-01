from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nullsim.pipeline import Geometry, RunContext, Scene, SimulationState


@pytest.fixture
def stage_run_context(tmp_path: Path) -> RunContext:
    """Run context for one atmosphere stage instance in pinning tests."""

    return RunContext(
        rng=np.random.default_rng(1),
        config_hash="kolmogorov-screens-pinning",
        sweep_coord_tuple=(),
        instance_id="atmosphere",
        run_dir=tmp_path,
    )


@pytest.fixture
def maunakea_scene_geometry_state() -> SimulationState:
    """Minimal state consumed by the future KolmogorovScreens stage."""

    diameter_m = 4.0
    area_m2 = math.pi * (diameter_m / 2.0) ** 2
    state = SimulationState.empty({"wavelength_grid_um": (1.65,)})
    scene = Scene(star_jmag=8.0).add_component(
        "pinning_scene",
        {"star_jmag": 8.0},
    )
    geometry = Geometry(
        diameter_m=diameter_m,
        area_m2=area_m2,
        n_apertures=1,
    ).add_component(
        "pinning_aperture",
        {
            "diameter_m": diameter_m,
            "area_m2": area_m2,
            "n_apertures": 1,
        },
    )
    return replace(state, scene=scene, geometry=geometry)
