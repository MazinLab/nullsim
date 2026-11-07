"""Polarimetric differential imaging (PDI) performance stage.

KILO_PARITY_PLAN §5.2. Companion to ``performance.standard`` that
computes the SNR for detecting a polarized planet via the SPHERE-IRDIS
/ GPI / ZIMPOL 4-angle Mueller modulation, calling
:func:`nullsim.physics.pdi.signal_to_noise_pdi` with the upstream
detector totals.

The stage assumes the detector port_rates already aggregate over the
two PBS arms (``planet_rate_total_hz`` =
``planet_dark_throughput`` × ``bright_rate_hz`` × ``contrast_target``,
``stellar_leakage_rate_total_hz`` = sum over dark-port rates). Run
alongside ``performance.standard`` when the user wants both the
single-arm contrast curve and the PDI SNR on the same simulation.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field as PydanticField

from nullsim.physics.pdi import signal_to_noise_pdi
from nullsim.pipeline import (
    BaseStage,
    SimulationState,
    register_stage,
)
from nullsim.stages._utils import unique_component_name


_DETECTOR_PREFIX = "detector."


@register_stage
class PdiPerformance(BaseStage):
    """Polarimetric differential imaging SNR from upstream detector totals.

    Reads the same ``detector.*`` component that
    ``performance.standard`` consumes (port_rates, port_labels,
    integration_time_s) and writes a ``performance.pdi`` results
    component carrying the PDI SNR and its breakdown for any planet
    contrast in ``contrasts``.
    """

    type_name = "performance.pdi"
    family = "performance"
    consumes = frozenset({"results"})
    produces = frozenset({"results"})

    class Params(BaseModel):
        """Validated params for the PDI performance stage."""

        model_config = ConfigDict(
            extra="forbid",
            frozen=True,
            allow_inf_nan=False,
        )

        pol_fraction: float = PydanticField(default=0.01, gt=0.0, le=1.0)
        contrasts: tuple[float, ...] = PydanticField(
            default=(1.0e-6, 1.0e-7, 1.0e-8),
        )
        planet_dark_throughput: float = PydanticField(default=0.5, gt=0.0, le=1.0)
        n_dark_ports_per_arm: int = PydanticField(default=1, ge=1)

        # Imperfection budget (KILO PBSConfig / HWPConfig defaults).
        pbs_axis_offset_rad: float = PydanticField(default=0.0, ge=0.0)
        hwp_retardance_error_rad: float = PydanticField(default=0.0, ge=0.0)
        hwp_rotation_jitter_rad: float = PydanticField(default=0.0, ge=0.0)
        pbs_extinction_db: float = PydanticField(default=30.0, gt=0.0)
        modulation_period_s: float = PydanticField(default=0.0, ge=0.0)
        cmrr_db: float = PydanticField(default=30.0, ge=0.0)
        recover_both_qu: bool = True

        # Background per arm (same defaults as performance.standard).
        n_sky_background_hz_per_dark_port_per_arm: float = PydanticField(
            default=0.0, ge=0.0,
        )
        n_dark_count_hz_per_port_per_arm: float = PydanticField(
            default=0.0, ge=0.0,
        )

    def apply(self, state: SimulationState) -> SimulationState:
        params = self.params_model

        detector_name, detector_component = _find_detector_component(state)
        port_rates_hz = np.asarray(
            detector_component["port_rates"], dtype=np.float64
        )
        port_labels = tuple(detector_component["port_labels"])
        integration_time_s = float(detector_component["integration_time_s"])

        bright_rate_hz, leakage_rate_hz_total = _partition_bright_dark(
            port_rates_hz, port_labels
        )

        snr_curve: list[dict[str, float]] = []
        for contrast in params.contrasts:
            planet_rate_total_hz = (
                bright_rate_hz
                * float(contrast)
                * float(params.planet_dark_throughput)
            )
            snr = signal_to_noise_pdi(
                planet_rate_total_hz=planet_rate_total_hz,
                stellar_leakage_rate_total_hz=leakage_rate_hz_total,
                integration_time_s=integration_time_s,
                pol_fraction=float(params.pol_fraction),
                n_dark_ports_per_arm=int(params.n_dark_ports_per_arm),
                pbs_axis_offset_rad=float(params.pbs_axis_offset_rad),
                hwp_retardance_error_rad=float(params.hwp_retardance_error_rad),
                hwp_rotation_jitter_rad=float(params.hwp_rotation_jitter_rad),
                pbs_extinction_db=float(params.pbs_extinction_db),
                modulation_period_s=float(params.modulation_period_s),
                cmrr_db=float(params.cmrr_db),
                recover_both_qu=bool(params.recover_both_qu),
                dark_rate_hz_per_port_per_arm=float(
                    params.n_dark_count_hz_per_port_per_arm
                ),
                sky_rate_hz_per_dark_port_per_arm=float(
                    params.n_sky_background_hz_per_dark_port_per_arm
                ),
            )
            snr_curve.append(
                {
                    "contrast": float(contrast),
                    "snr": float(snr),
                    "planet_rate_total_hz": float(planet_rate_total_hz),
                }
            )

        component_name = unique_component_name(
            self.type_name, state.results.components, self.context.instance_id,
        )
        payload: dict[str, Any] = {
            "snr_curve": tuple(snr_curve),
            "pol_fraction": float(params.pol_fraction),
            "planet_dark_throughput": float(params.planet_dark_throughput),
            "n_dark_ports_per_arm": int(params.n_dark_ports_per_arm),
            "integration_time_s": float(integration_time_s),
            "bright_rate_hz": float(bright_rate_hz),
            "leakage_rate_hz_total": float(leakage_rate_hz_total),
            "upstream_detector_component": detector_name,
            "imperfection_budget": {
                "pbs_axis_offset_rad": float(params.pbs_axis_offset_rad),
                "hwp_retardance_error_rad": float(params.hwp_retardance_error_rad),
                "hwp_rotation_jitter_rad": float(params.hwp_rotation_jitter_rad),
                "pbs_extinction_db": float(params.pbs_extinction_db),
                "modulation_period_s": float(params.modulation_period_s),
                "cmrr_db": float(params.cmrr_db),
                "recover_both_qu": bool(params.recover_both_qu),
            },
            "source_model": (
                "nullsim.physics.pdi.signal_to_noise_pdi "
                "(KILO port; kilo/performance.py:412-648)."
            ),
        }
        results = state.results.add_component(component_name, payload)
        return replace(state, results=results)


def _find_detector_component(state: SimulationState) -> tuple[str, Any]:
    """Return the newest ``detector.*`` component with port_rates+port_labels."""

    for name, value in reversed(state.results.components.items()):
        if not name.startswith(_DETECTOR_PREFIX):
            continue
        if not isinstance(value, dict):
            continue
        if {"port_rates", "port_labels", "integration_time_s"}.issubset(value.keys()):
            return name, value
    raise ValueError(
        "performance.pdi requires an upstream detector stage that wrote a "
        "'detector.*' component with port_rates, port_labels, and "
        "integration_time_s. See performance.standard for the same contract."
    )


def _partition_bright_dark(
    port_rates_hz: np.ndarray, port_labels: tuple[str, ...]
) -> tuple[float, float]:
    """Return ``(bright_total_hz, dark_total_hz)`` from labeled port rates."""

    if port_rates_hz.shape != (len(port_labels),):
        raise ValueError(
            f"performance.pdi: port_rates shape {port_rates_hz.shape} "
            f"does not match {len(port_labels)} port labels."
        )
    bright_mask = np.array(
        [label.startswith("bright") for label in port_labels], dtype=bool
    )
    dark_mask = np.array(
        [label.startswith("dark") for label in port_labels], dtype=bool
    )
    bright = float(port_rates_hz[bright_mask].sum())
    dark = float(port_rates_hz[dark_mask].sum())
    return bright, dark
