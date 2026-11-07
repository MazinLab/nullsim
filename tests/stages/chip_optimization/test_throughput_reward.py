"""Planet fields behind the chip optimizer's throughput reward."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from scipy.integrate import quad
from scipy.special import j0

from nullsim.physics.ruilier_cassaing import (
    OPTIMAL_PUPIL_RADIUS_TO_MODE_FIELD_RADIUS_RATIO as RHO,
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


def _exact_coupling(separation_rad: float, diameter_m: float) -> float:
    """Off-axis SMF coupling ratio by direct quadrature of the overlap."""

    s = separation_rad * diameter_m / (_WL_UM * 1e-6)
    num = quad(lambda u: u * math.exp(-RHO**2 * u**2) * j0(math.pi * s * u), 0.0, 1.0)[0]
    den = quad(lambda u: u * math.exp(-RHO**2 * u**2), 0.0, 1.0)[0]
    return (num / den) ** 2


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
    expected = _exact_coupling(theta, diameter_m)
    np.testing.assert_allclose(power, expected, rtol=1e-9)
    equal_area = _exact_coupling(theta, math.sqrt(4.0 * area_m2 / math.pi))
    assert expected < 0.9 and equal_area - expected > 0.05


def test_throughput_reward_sums_polarization_arms() -> None:
    # Two PBS arms carrying half the power each give the same planet field
    # power as the scalar field, matching the leakage sum (row 29).
    state = _state(1.0, 2.0)
    scalar = np.full((2, 1, 1), 0.8, dtype=np.complex128)
    arms = np.full((2, 1, 2), 0.8 / math.sqrt(2.0), dtype=np.complex128)

    np.testing.assert_allclose(
        _planet_power(state, arms), _planet_power(state, scalar), rtol=1e-12
    )
