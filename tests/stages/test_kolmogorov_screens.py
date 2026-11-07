"""Tests for the KolmogorovScreens atmosphere stage."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from nullsim.pipeline import (
    Geometry,
    OPDComponent,
    RunContext,
    Scene,
    SimulationState,
    build_instance,
    validate_dependencies,
)
from nullsim.stages.atmosphere import KolmogorovScreens
from nullsim.stages.telescope.fixed_aperture import FixedAperture

ATMOSPHERE_STAGE_SPEC: dict[str, object] = {
    "id": "atmosphere",
    "type": "atmosphere.kolmogorov",
    "r0_500nm_m": 0.16,
}


def _build_kolmogorov_screens(
    context: RunContext,
    **overrides: object,
) -> KolmogorovScreens:
    spec = {**ATMOSPHERE_STAGE_SPEC, **overrides}
    stage = build_instance(spec, context)
    assert isinstance(stage, KolmogorovScreens)
    return stage


def _scene_geometry_state(
    *,
    wavelength_grid_um: tuple[float, ...],
    diameter_m: float,
) -> SimulationState:
    area_m2 = math.pi * (diameter_m / 2.0) ** 2
    state = SimulationState.empty({"wavelength_grid_um": wavelength_grid_um})
    scene = Scene(star_jmag=8.0).add_component(
        "test_scene",
        {"star_jmag": 8.0},
    )
    geometry = Geometry(
        diameter_m=diameter_m,
        area_m2=area_m2,
        n_apertures=1,
    ).add_component(
        "test_aperture",
        {
            "diameter_m": diameter_m,
            "area_m2": area_m2,
            "n_apertures": 1,
        },
    )
    return replace(state, scene=scene, geometry=geometry)


def _expected_stage_arrays(
    *,
    r0_500nm_m: float,
    diameter_m: float,
    wavelength_grid_um: tuple[float, ...],
    airmass: float = 1.0,
) -> dict[str, np.ndarray]:
    """Hand formulas: r0 = r0_500 (lambda/500 nm)^(6/5) X^(-3/5) and the
    Noll (1976) piston-removed variance 1.0299 (D/r0)^(5/3)."""
    wavelength_grid_m = np.asarray(wavelength_grid_um, dtype=np.float64) * 1.0e-6
    r0_scaled_m = r0_500nm_m * (wavelength_grid_m / 500.0e-9) ** 1.2 * airmass**-0.6
    phase_variance_rad2 = 1.0299 * (diameter_m / r0_scaled_m) ** (5.0 / 3.0)
    opd_rms_m = np.sqrt(phase_variance_rad2) * wavelength_grid_m / (2.0 * math.pi)
    strehl_ratio = np.exp(-phase_variance_rad2)
    return {
        "r0_scaled_m": np.asarray(r0_scaled_m),
        "phase_variance_rad2": np.asarray(phase_variance_rad2),
        "opd_rms_nm": np.asarray(opd_rms_m) * 1.0e9,
        "strehl_ratio": np.asarray(strehl_ratio),
    }


def test_kolmogorov_screens_matches_physics_primitives(
    maunakea_scene_geometry_state: SimulationState,
    stage_run_context: RunContext,
) -> None:
    stage = _build_kolmogorov_screens(stage_run_context)

    state = stage.apply(maunakea_scene_geometry_state)

    expected = _expected_stage_arrays(
        r0_500nm_m=0.16,
        diameter_m=4.0,
        wavelength_grid_um=(1.65,),
    )
    assert state.wavefront.wavelength_grid_um == (1.65,)
    assert state.wavefront.strehl_ratio == pytest.approx(
        tuple(expected["strehl_ratio"]),
    )
    assert state.wavefront.amplitude_jitter_fraction == pytest.approx((0.0,))

    wavefront_component = state.wavefront.components["atmosphere"]
    assert wavefront_component["piston_removed"] is True
    assert wavefront_component["r0_scaled_m"] == pytest.approx(
        tuple(expected["r0_scaled_m"]),
    )
    assert wavefront_component["phase_variance_rad2"] == pytest.approx(
        tuple(expected["phase_variance_rad2"]),
    )
    assert wavefront_component["opd_rms_nm"] == pytest.approx(
        tuple(expected["opd_rms_nm"]),
    )

    atmosphere_opd = state.opd_budget.components["atmos_piston"]
    assert isinstance(atmosphere_opd, OPDComponent)
    assert atmosphere_opd.rms_opd_nm == pytest.approx(expected["opd_rms_nm"][0])
    assert atmosphere_opd.reduction_stage == "kolmogorov_screens"
    assert state.opd_budget.total_rms_opd_nm == pytest.approx(
        expected["opd_rms_nm"][0],
    )


def test_kolmogorov_screens_degrades_r0_with_airmass(
    stage_run_context: RunContext,
) -> None:
    """At altitude 30 deg (airmass 2) r0 shrinks by 2^(-3/5), as in the AO
    stage, and the variance grows by 2^(5/3 * 3/5) = 2."""

    state = _scene_geometry_state(wavelength_grid_um=(1.25, 1.65), diameter_m=4.0)
    state = replace(state, scene=replace(state.scene, altitude_deg=30.0))
    stage = _build_kolmogorov_screens(stage_run_context)

    component = stage.apply(state).wavefront.components["atmosphere"]

    expected = _expected_stage_arrays(
        r0_500nm_m=0.16,
        diameter_m=4.0,
        wavelength_grid_um=(1.25, 1.65),
        airmass=2.0,
    )
    zenith = _expected_stage_arrays(
        r0_500nm_m=0.16, diameter_m=4.0, wavelength_grid_um=(1.25, 1.65),
    )
    assert component["airmass"] == pytest.approx(2.0)
    assert component["r0_scaled_m"] == pytest.approx(tuple(expected["r0_scaled_m"]))
    assert component["phase_variance_rad2"] == pytest.approx(
        tuple(2.0 * zenith["phase_variance_rad2"])
    )


def test_kolmogorov_screens_records_optional_provenance_params(
    maunakea_scene_geometry_state: SimulationState,
    stage_run_context: RunContext,
) -> None:
    stage = _build_kolmogorov_screens(
        stage_run_context,
        seeing_arcsec=0.7,
        wind_speed_m_s=12.0,
        L0_m=25.0,
    )

    state = stage.apply(maunakea_scene_geometry_state)

    component = state.wavefront.components["atmosphere"]
    assert component["seeing_arcsec"] == 0.7
    assert component["wind_speed_m_s"] == 12.0
    assert component["L0_m"] == 25.0
    assert component["outer_scale_model"] == "recorded_unused"


def test_strehl_monotonically_increases_with_wavelength(
    stage_run_context: RunContext,
) -> None:
    wavelength_grid_um = (1.1, 1.3, 1.65, 2.2)
    state = _scene_geometry_state(wavelength_grid_um=wavelength_grid_um, diameter_m=2.0)
    stage = _build_kolmogorov_screens(stage_run_context, r0_500nm_m=0.3)

    result = stage.apply(state)

    strehl_ratio = np.asarray(result.wavefront.strehl_ratio)
    assert np.all(np.diff(strehl_ratio) > 0.0)


def test_opd_rms_is_wavelength_independent_when_r0_scales_with_wavelength(
    stage_run_context: RunContext,
) -> None:
    wavelength_grid_um = (1.1, 1.3, 1.65, 2.2)
    state = _scene_geometry_state(wavelength_grid_um=wavelength_grid_um, diameter_m=2.0)
    stage = _build_kolmogorov_screens(stage_run_context, r0_500nm_m=0.3)

    result = stage.apply(state)

    component = result.wavefront.components["atmosphere"]
    opd_rms_nm = np.asarray(component["opd_rms_nm"])
    np.testing.assert_allclose(opd_rms_nm, opd_rms_nm[0], rtol=1.0e-10)


def test_strehl_is_bounded_over_wavelength_grid(stage_run_context: RunContext) -> None:
    state = _scene_geometry_state(wavelength_grid_um=(1.1, 1.3, 1.65, 2.2), diameter_m=2.0)
    stage = _build_kolmogorov_screens(stage_run_context, r0_500nm_m=0.3)

    result = stage.apply(state)

    strehl_ratio = np.asarray(result.wavefront.strehl_ratio)
    assert np.all(strehl_ratio >= 0.0)
    assert np.all(strehl_ratio <= 1.0)


def test_bad_seeing_limit_has_low_j_band_strehl(stage_run_context: RunContext) -> None:
    state = _scene_geometry_state(wavelength_grid_um=(1.25,), diameter_m=10.0)
    stage = _build_kolmogorov_screens(stage_run_context, r0_500nm_m=0.05)

    result = stage.apply(state)

    assert result.wavefront.strehl_ratio[0] < 0.01


def test_good_seeing_limit_has_high_h_band_strehl(stage_run_context: RunContext) -> None:
    # r0(H) = 2.1 m, so D = 1 m gives S = exp(-1.0299 * 0.29) = 0.74.
    state = _scene_geometry_state(wavelength_grid_um=(1.65,), diameter_m=1.0)
    stage = _build_kolmogorov_screens(stage_run_context, r0_500nm_m=0.5)

    result = stage.apply(state)

    assert result.wavefront.strehl_ratio[0] > 0.5


def test_nan_r0_propagates_to_strehl_and_opd(stage_run_context: RunContext) -> None:
    state = _scene_geometry_state(wavelength_grid_um=(1.25, 1.65), diameter_m=2.0)
    stage = _build_kolmogorov_screens(stage_run_context, r0_500nm_m=math.nan)

    result = stage.apply(state)

    component = result.wavefront.components["atmosphere"]
    assert all(math.isnan(value) for value in result.wavefront.strehl_ratio)
    assert all(math.isnan(value) for value in component["opd_rms_nm"])
    atmosphere_opd = result.opd_budget.components["atmos_piston"]
    assert isinstance(atmosphere_opd, OPDComponent)
    assert math.isnan(atmosphere_opd.rms_opd_nm)
    assert math.isnan(result.opd_budget.total_rms_opd_nm)


@pytest.mark.parametrize("bad_r0_500nm_m", [0.0, -0.1])
def test_zero_or_negative_r0_rejected_by_params(
    bad_r0_500nm_m: float,
) -> None:
    with pytest.raises(ValidationError, match="r0_500nm_m"):
        KolmogorovScreens.Params(r0_500nm_m=bad_r0_500nm_m)


def test_validate_dependencies_accepts_scene_geometry_then_atmosphere(
    tmp_path: Path,
    stage_run_context: RunContext,
) -> None:
    telescope = FixedAperture(
        {"diameter_m": 2.0},
        RunContext(
            rng=np.random.default_rng(1),
            config_hash="kolmogorov-dependencies",
            sweep_coord_tuple=(),
            instance_id="array",
            run_dir=tmp_path,
        ),
    )
    atmosphere = _build_kolmogorov_screens(stage_run_context)

    validate_dependencies([_SceneProducer(), telescope, atmosphere])


class _SceneProducer:
    type_name = "tests.kolmogorov.scene"
    family = "scene"
    consumes = frozenset()
    produces = frozenset({"scene"})
    context = None

    def apply(self, state: SimulationState) -> SimulationState:
        return state

    def diagnostics(self) -> dict[str, Any]:
        return {}

    def external_dependencies(self) -> list[Any]:
        return []
