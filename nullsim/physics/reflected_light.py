"""Reflected-light planet/star contrast helpers."""

from __future__ import annotations

import math

R_EARTH_AU = 4.2587e-5


def lambertian_phase(phase_angle_rad: float) -> float:
    """Return the Lambertian phase function for phase angle ``alpha``.

    ``Phi(alpha) = [sin(alpha) + (pi - alpha) cos(alpha)] / pi`` with
    ``alpha = 0`` at full phase and ``alpha = pi`` at new phase.
    """

    alpha = float(phase_angle_rad)
    if not math.isfinite(alpha) or alpha < 0.0 or alpha > math.pi:
        raise ValueError("phase_angle_rad must be finite and in [0, pi].")
    return (math.sin(alpha) + (math.pi - alpha) * math.cos(alpha)) / math.pi


def reflected_light_contrast(
    *,
    radius_earth: float,
    semi_major_axis_au: float,
    geometric_albedo: float,
    phase_angle_rad: float,
) -> float:
    """Return reflected-light planet/star contrast for a circular orbit.

    Args:
        radius_earth: Planet radius in Earth radii.
        semi_major_axis_au: Orbital radius or semi-major axis in AU.
        geometric_albedo: Geometric albedo ``A_g``.
        phase_angle_rad: Star-planet-observer phase angle in radians.

    Returns:
        ``A_g * Phi(alpha) * (R_p / a)^2``.
    """

    radius = float(radius_earth)
    semi_major_axis = float(semi_major_axis_au)
    albedo = float(geometric_albedo)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius_earth must be positive and finite.")
    if not math.isfinite(semi_major_axis) or semi_major_axis <= 0.0:
        raise ValueError("semi_major_axis_au must be positive and finite.")
    if not math.isfinite(albedo) or albedo < 0.0:
        raise ValueError("geometric_albedo must be non-negative and finite.")
    phase = lambertian_phase(phase_angle_rad)
    radius_au = radius * R_EARTH_AU
    return float(albedo * phase * (radius_au / semi_major_axis) ** 2)


def projected_separation_to_au(
    *,
    separation_mas: float,
    distance_pc: float,
) -> float:
    """Convert angular separation to projected separation in AU."""

    separation = float(separation_mas)
    distance = float(distance_pc)
    if not math.isfinite(separation) or separation < 0.0:
        raise ValueError("separation_mas must be non-negative and finite.")
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError("distance_pc must be positive and finite.")
    return separation * distance / 1000.0


__all__ = [
    "R_EARTH_AU",
    "lambertian_phase",
    "projected_separation_to_au",
    "reflected_light_contrast",
]
