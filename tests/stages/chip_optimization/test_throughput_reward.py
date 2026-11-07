"""Planet fields behind the chip optimizer's throughput reward."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from nullsim.physics.ruilier_cassaing import (
    KILO_MODE_FIELD_RADIUS_TO_SUBAP_RADIUS_RATIO as MFD_RATIO,
)
from nullsim.pipeline import Geometry, SimulationState
from nullsim.stages.chip_optimization import (
    ChipOptimizationParams,
    _throughput_penalty_fields,
)

_WL_UM = 1.25
_SEP_MAS = 100.0


def _params() -> ChipOptimizationParams:
    return ChipOptimizationParams(
        n_modes=2, n_bright_ports=1, n_dark_ports=1, wl_design_um=_WL_UM,
        throughput_penalty_alpha=1.0, throughput_penalty_seps_mas=(_SEP_MAS,),
        throughput_penalty_n_pa=1,
    )


def _state(mode_area_m2: float, mode_diameter_m: float) -> SimulationState:
    state = SimulationState.empty({"wavelength_grid_um": (_WL_UM,)})
    return replace(
        state,
        geometry=Geometry(
            components={
                "splitter": {
                    "mode_area_m2": mode_area_m2,
                    "mode_diameter_m": mode_diameter_m,
                },
            },
        ),
    )


def _planet_power(state: SimulationState, field: np.ndarray) -> np.ndarray:
    fields = _throughput_penalty_fields(
        state=state,
        field_array=field,
        wavelength_grid_um=(_WL_UM,),
        params=_params(),
        aperture_positions_pupil_m=np.zeros((2, 2)),
    )
    return np.abs(fields[0, 0]) ** 2


def test_throughput_reward_uses_physical_mode_diameter() -> None:
    # A splitter halves area but keeps the 2 m diameter; the equal-area
    # circle (1.13 m) would widen the coupling envelope (Issue 20).
    area_m2, diameter_m = 1.0, 2.0
    field = np.ones((2, 1, 1), dtype=np.complex128)

    power = _planet_power(_state(area_m2, diameter_m), field)

    theta = _SEP_MAS * math.pi / (180.0 * 3600.0 * 1000.0)
    x = math.pi * MFD_RATIO * diameter_m / 2.0 * theta / (_WL_UM * 1e-6)
    np.testing.assert_allclose(power, math.exp(-x * x), rtol=1e-12)
    assert math.exp(-x * x) < 0.9  # the envelope matters at this separation


def test_throughput_reward_sums_polarization_arms() -> None:
    # Two PBS arms carrying half the power each give the same planet field
    # power as the scalar field, matching the leakage sum (row 29).
    state = _state(1.0, 2.0)
    scalar = np.full((2, 1, 1), 0.8, dtype=np.complex128)
    arms = np.full((2, 1, 2), 0.8 / math.sqrt(2.0), dtype=np.complex128)

    np.testing.assert_allclose(
        _planet_power(state, arms), _planet_power(state, scalar), rtol=1e-12
    )
