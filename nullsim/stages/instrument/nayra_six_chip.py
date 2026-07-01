"""NAYRA six-chip architecture manifest stage (SPEC §15.6).

NAYRA observes with six physical photonic chips: three science bands
(``Y``, ``J``, ``H``) times two PBS polarization arms (``A``, ``B``). This
stage records that architecture in ``state.results`` so NAYRA configs no
longer describe the warm front end as a scalar single-chip surrogate. The
active nullsim run can still be a J-primary optical propagation; the manifest
identifies the sibling chips and the per-channel frontend-to-chip throughput
needed by follow-on factory/multiband orchestration.
"""

from __future__ import annotations

from dataclasses import replace

from pydantic import BaseModel, ConfigDict, Field as PydanticField, model_validator

from nullsim.stages._utils import unique_component_name
from nullsim.stages.transport.smf28 import pm980xp_attenuation_db_per_km
from nullsim.pipeline import BaseStage, SimulationState, register_stage


DEFAULT_BANDS = ("Y", "J", "H")
DEFAULT_POL_ARMS = ("A", "B")
DEFAULT_BAND_CENTERS_UM = (1.02, 1.25, 1.65)
DEFAULT_BANDWIDTHS_UM = (0.10, 0.16, 0.30)


@register_stage
class NayraSixChipArchitecture(BaseStage):
    """Record NAYRA's Y/J/H × dual-pol chip/channel manifest."""

    type_name = "instrument.nayra_six_chip"
    family = "instrument"
    produces = frozenset({"results"})

    class Params(BaseModel):
        """Validated NAYRA six-chip manifest parameters."""

        model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

        bands: tuple[str, ...] = DEFAULT_BANDS
        pol_arms: tuple[str, ...] = DEFAULT_POL_ARMS
        wavelength_center_um: tuple[float, ...] = DEFAULT_BAND_CENTERS_UM
        wavelength_bandwidth_um: tuple[float, ...] = DEFAULT_BANDWIDTHS_UM
        primary_band: str = "J"
        n_modes: int = PydanticField(default=18, gt=0)
        n_bright_ports: int = PydanticField(default=4, ge=0)
        n_dark_ports: int = PydanticField(default=13, ge=0)
        n_spectro_ports: int = PydanticField(default=1, ge=0)
        frontend_bulk_transmission: float = PydanticField(default=0.85, gt=0.0, le=1.0)
        pbs_insertion_loss_db: float = PydanticField(default=0.5, ge=0.0)
        pm_fiber_length_m: float = PydanticField(default=5.0, ge=0.0)
        edge_coupler_loss_db: float = PydanticField(default=0.3, ge=0.0)

        @model_validator(mode="after")
        def _validate_channel_axes(self) -> NayraSixChipArchitecture.Params:
            if len(set(self.bands)) != len(self.bands):
                raise ValueError("bands must be unique.")
            if len(set(self.pol_arms)) != len(self.pol_arms):
                raise ValueError("pol_arms must be unique.")
            if self.primary_band not in self.bands:
                raise ValueError("primary_band must be present in bands.")
            if len(self.wavelength_center_um) != len(self.bands):
                raise ValueError("wavelength_center_um length must match bands.")
            if len(self.wavelength_bandwidth_um) != len(self.bands):
                raise ValueError("wavelength_bandwidth_um length must match bands.")
            n_ports = self.n_bright_ports + self.n_dark_ports + self.n_spectro_ports
            if n_ports != self.n_modes:
                raise ValueError(
                    "n_bright_ports + n_dark_ports + n_spectro_ports must "
                    f"equal n_modes (got {n_ports} != {self.n_modes})."
                )
            return self

    def apply(self, state: SimulationState) -> SimulationState:
        params = self.params_model
        channels = _channel_manifest(params)
        component_name = unique_component_name(
            self.type_name,
            state.results.components,
            self.context.instance_id,
        )
        results = state.results.add_component(
            component_name,
            {
                "bands": tuple(params.bands),
                "pol_arms": tuple(params.pol_arms),
                "primary_band": str(params.primary_band),
                "n_channels": len(channels),
                "n_band_designs": len(params.bands),
                "n_modes": int(params.n_modes),
                "n_bright_ports": int(params.n_bright_ports),
                "n_dark_ports": int(params.n_dark_ports),
                "n_spectro_ports": int(params.n_spectro_ports),
                "channels": tuple(channels),
                "source_model": (
                    "NAYRA six-chip architecture: one band-specific chip design "
                    "for each of Y/J/H, duplicated across PBS arms A/B. The active "
                    "pipeline may propagate one primary band; this manifest records "
                    "the sibling channels for factory/multiband orchestration."
                ),
            },
        )
        return replace(state, results=results)


def _channel_manifest(params: NayraSixChipArchitecture.Params) -> list[dict[str, object]]:
    channels: list[dict[str, object]] = []
    pbs_arm_transmission = 0.5 * 10.0 ** (-float(params.pbs_insertion_loss_db) / 10.0)
    edge_transmission = 10.0 ** (-float(params.edge_coupler_loss_db) / 10.0)
    for band, center_um, bandwidth_um in zip(
        params.bands,
        params.wavelength_center_um,
        params.wavelength_bandwidth_um,
        strict=True,
    ):
        pm_attenuation_db_per_km = float(pm980xp_attenuation_db_per_km(center_um * 1.0e-6))
        pm_fiber_transmission = 10.0 ** (
            -pm_attenuation_db_per_km * (float(params.pm_fiber_length_m) / 1000.0) / 10.0
        )
        frontend_to_chip = (
            float(params.frontend_bulk_transmission)
            * pbs_arm_transmission
            * pm_fiber_transmission
            * edge_transmission
        )
        for pol_arm in params.pol_arms:
            channels.append(
                {
                    "band": str(band),
                    "pol_arm": str(pol_arm),
                    "channel_id": f"{band}_{pol_arm}",
                    "band_design_id": f"{band}_chip_design",
                    "wavelength_center_um": float(center_um),
                    "wavelength_bandwidth_um": float(bandwidth_um),
                    "is_primary_band": band == params.primary_band,
                    "frontend_bulk_transmission": float(
                        params.frontend_bulk_transmission
                    ),
                    "pbs_arm_transmission": float(pbs_arm_transmission),
                    "pm980_attenuation_db_per_km": float(pm_attenuation_db_per_km),
                    "pm_fiber_transmission": float(pm_fiber_transmission),
                    "edge_coupler_transmission": float(edge_transmission),
                    "frontend_to_chip_transmission": float(frontend_to_chip),
                }
            )
    return channels


__all__ = [
    "DEFAULT_BANDS",
    "DEFAULT_BANDWIDTHS_UM",
    "DEFAULT_BAND_CENTERS_UM",
    "DEFAULT_POL_ARMS",
    "NayraSixChipArchitecture",
]
