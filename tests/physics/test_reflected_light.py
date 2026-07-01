from __future__ import annotations

import math

import pytest

from nullsim.physics.reflected_light import (
    R_EARTH_AU,
    lambertian_phase,
    projected_separation_to_au,
    reflected_light_contrast,
)


def test_lambertian_phase_limits() -> None:
    assert lambertian_phase(0.0) == pytest.approx(1.0)
    assert lambertian_phase(math.pi / 2.0) == pytest.approx(1.0 / math.pi)
    assert lambertian_phase(math.pi) == pytest.approx(0.0, abs=1.0e-15)


def test_reflected_light_contrast_matches_quadrature_formula() -> None:
    contrast = reflected_light_contrast(
        radius_earth=1.0,
        semi_major_axis_au=1.0,
        geometric_albedo=0.3,
        phase_angle_rad=math.pi / 2.0,
    )

    assert contrast == pytest.approx(0.3 * (1.0 / math.pi) * R_EARTH_AU**2)


def test_projected_separation_to_au() -> None:
    assert projected_separation_to_au(
        separation_mas=200.0,
        distance_pc=5.0,
    ) == pytest.approx(1.0)
