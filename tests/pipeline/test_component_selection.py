"""Component selectors take the newest matching component (review 2026-11-07 Issue 25)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from nullsim.outputs.plots import _chip_field_inputs
from nullsim.outputs.plots.imaging_reconstruction import (
    _find_detector_component as imaging_find_detector,
)
from nullsim.pipeline import Results, SimulationState, Throughput
from nullsim.stages import chip_optimization
from nullsim.stages.calibration import floor_lookup
from nullsim.stages.diagnostics import fringe_observability
from nullsim.stages.performance import pdi
from nullsim.stages.sensitivity import detection_curve

_DETECTOR = {
    "port_rates": (1.0,),
    "port_labels": ("dark_0",),
    "integration_time_s": 1.0,
    "null_depth_raw": 1e-5,
}


def _state(results: dict[str, Any], throughput: dict[str, Any]) -> SimulationState:
    state = SimulationState.empty({"wavelength_grid_um": (1.25,)})
    return replace(
        state,
        results=Results(components=results),
        throughput=Throughput(components=throughput),
    )


def _two(prefix: str, payload: dict[str, Any], key: str) -> dict[str, Any]:
    return {
        f"{prefix}": {**payload, key: "old"},
        f"{prefix}.second": {**payload, key: "new"},
    }


def _unwrap(found: Any) -> dict[str, Any]:
    return found[1] if isinstance(found, tuple) else found


_RESULTS_SELECTORS: list[tuple[str, str, Callable[[SimulationState], Any]]] = [
    ("detector.mkid", "sensitivity", detection_curve._find_detector_component),
    ("detector.mkid", "floor_lookup", floor_lookup._find_detector_component),
    ("detector.mkid", "pdi", pdi._find_detector_component),
    ("detector.mkid", "imaging", imaging_find_detector),
    (
        "sensitivity.detection_curve",
        "fringe_observability",
        fringe_observability._find_sensitivity_payload,
    ),
    (
        "sensitivity.detection_curve",
        "chip_field_inputs",
        lambda state: _chip_field_inputs.maybe_find_component(
            state, "sensitivity.detection_curve"
        ),
    ),
]


@pytest.mark.parametrize(
    ("prefix", "consumer", "select"),
    _RESULTS_SELECTORS,
    ids=[consumer for _, consumer, _ in _RESULTS_SELECTORS],
)
def test_results_selectors_take_newest_component(
    prefix: str, consumer: str, select: Callable[[SimulationState], Any]
) -> None:
    state = _state(_two(prefix, _DETECTOR, "tag"), {})
    assert _unwrap(select(state))["tag"] == "new", consumer


@pytest.mark.parametrize(
    "select",
    [
        chip_optimization._delay_line_diagnostic_payload,
        detection_curve._delay_line_diagnostic_payload,
    ],
    ids=["chip_optimization", "sensitivity"],
)
def test_delay_line_selectors_take_newest_component(
    select: Callable[[SimulationState], Any],
) -> None:
    throughput = {
        "delay_line_loss": {"per_aperture": {"tag": "old"}},
        "delay_line_loss.second": {"per_aperture": {"tag": "new"}},
    }
    assert select(_state({}, throughput))["tag"] == "new"


def test_amp_servo_mode_count_takes_newest_chip() -> None:
    results = {
        "chip.kernel_mzi_mesh": {"n_modes": 4},
        "chip.kernel_mzi_mesh.second": {"n_modes": 6},
    }
    params = SimpleNamespace(n_bright_ports=1, n_dark_ports=1, n_spectro_ports=0)
    n_modes = detection_curve._resolve_amp_servo_n_modes(_state(results, {}), params)
    assert n_modes == 6
