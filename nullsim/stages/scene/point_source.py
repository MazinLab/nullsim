"""Architecture placeholder scene stage for a star plus one planet.

This module exists to exercise the vertical-slice pipeline from SPEC §15.1.
It is not a physics implementation of stellar spectra, phase functions, or
background sources.
"""

from __future__ import annotations

from dataclasses import replace

from pydantic import BaseModel, ConfigDict, Field

from nullsim.pipeline import BaseStage, Scene, SimulationState, register_stage


@register_stage
class PointSource(BaseStage):
    """Populate the scene budget with scalar placeholder source properties."""

    type_name = "scene.point_source"
    family = "scene"
    consumes = frozenset()
    produces = frozenset({"scene"})

    class Params(BaseModel):
        """Validated params for the placeholder point-source scene."""

        model_config = ConfigDict(
            extra="forbid",
            frozen=True,
            allow_inf_nan=False,
        )

        star_jmag: float = 5.0
        # Apparent stellar angular diameter in milli-arcseconds. Used
        # downstream by sensitivity.detection_curve to evaluate the
        # stellar-disk leakage floor (uniform-disk integration against
        # the chip throughput grid). Defaults to None which leaves
        # state.scene.star_angular_diameter_mas unset (the sensitivity
        # stage then treats the source as a point and disk_floor=0).
        star_angular_diameter_mas: float | None = Field(default=None, ge=0.0)
        star_distance_pc: float | None = Field(default=None, gt=0.0)
        star_spectral_type: str | None = None
        planet_contrast: float = Field(default=1.0e-6, ge=0.0)
        planet_separation_mas: float = Field(default=100.0, ge=0.0)
        position_angle_deg: float = 0.0
        exozodi_level_zodi: float = Field(default=0.0, ge=0.0)
        altitude_deg: float | None = None
        azimuth_deg: float | None = None

    def apply(self, state: SimulationState) -> SimulationState:
        params = self.params_model
        scene = Scene(
            star_jmag=params.star_jmag,
            star_angular_diameter_mas=params.star_angular_diameter_mas,
            star_distance_pc=params.star_distance_pc,
            star_spectral_type=params.star_spectral_type,
            planet_contrast=params.planet_contrast,
            planet_separation_mas=params.planet_separation_mas,
            planet_position_angle_deg=params.position_angle_deg,
            exozodi_level_zodi=params.exozodi_level_zodi,
            altitude_deg=params.altitude_deg,
            azimuth_deg=params.azimuth_deg,
        ).add_component(
            "point_source",
            {
                "star_jmag": params.star_jmag,
                "star_angular_diameter_mas": params.star_angular_diameter_mas,
                "star_distance_pc": params.star_distance_pc,
                "star_spectral_type": params.star_spectral_type,
                "planet_contrast": params.planet_contrast,
                "planet_separation_mas": params.planet_separation_mas,
                "position_angle_deg": params.position_angle_deg,
                "exozodi_level_zodi": params.exozodi_level_zodi,
                "altitude_deg": params.altitude_deg,
                "azimuth_deg": params.azimuth_deg,
            },
        )
        return replace(state, scene=scene)


__all__ = ["PointSource"]
