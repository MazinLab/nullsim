"""Tests for the cophasing-unitary plot output."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nullsim.outputs import RunCollection, RunRecord
from nullsim.outputs.plots.cophasing_unitary import (
    _find_payload,
    cophasing_unitary,
)
from nullsim.pipeline import Results, RunResult, SimulationState


def _payload() -> dict:
    return {
        "n_physical_piston_modes": 4,
        "positions_pupil_m": np.array(
            [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]],
            dtype=np.float64,
        ),
        "cophasing_tracker_fraction": 0.9,
        "cophasing_visibility": 0.95,
        "cophasing_edge_throughput": 0.8,
        "cophasing_phase_offsets_rad": (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi),
        "cophasing_edge_count_weights": (0.2, 0.3, 0.5),
        "cophasing_handoff_monte_carlo": {
            "edges": np.array([[0, 2], [0, 1], [2, 3]], dtype=np.int64),
            "median_acquisition_residual_rms_nm": 0.4,
            "acquisition_lock_fraction": 1.0,
            "median_science_residual_rms_nm": 3.1,
            "science_lock_fraction": 0.94,
            "edge_wrap_fraction": 0.0,
            "max_edge_opd_over_lambda": 0.18,
        },
    }


def test_find_payload_prefers_full_diagnostic_over_prepass() -> None:
    full_payload = _payload()
    state = SimulationState.empty({"wavelength_grid_um": (1.2, 1.3)})
    state = SimulationState(
        scene=state.scene,
        geometry=state.geometry,
        field=state.field,
        opd_budget=state.opd_budget,
        throughput=state.throughput,
        wavefront=state.wavefront,
        photon_rates=state.photon_rates,
        results=Results(
            components={
                "fringe_observability_pre_sensitivity": {
                    "cophasing_handoff_monte_carlo": {"disabled": True},
                },
                "fringe_observability": full_payload,
            },
        ),
    )
    record = RunRecord(
        config=SimpleNamespace(),
        result=RunResult(
            state=state,
            diagnostics={},
            timings=(),
            cache_hit_count=0,
            config_hash="plot",
        ),
        run_dir=Path("."),
        coord_tuple=(),
    )

    selected = _find_payload(record)

    assert selected is not None
    assert selected["n_physical_piston_modes"] == full_payload["n_physical_piston_modes"]
    assert selected["cophasing_handoff_monte_carlo"]["science_lock_fraction"] == 0.94


def test_registered_cophasing_unitary_writes_plot(tmp_path: Path) -> None:
    state = SimulationState.empty({"wavelength_grid_um": (1.2, 1.3)})
    state = SimulationState(
        scene=state.scene,
        geometry=state.geometry,
        field=state.field,
        opd_budget=state.opd_budget,
        throughput=state.throughput,
        wavefront=state.wavefront,
        photon_rates=state.photon_rates,
        results=Results(components={"fringe_observability": _payload()}),
    )
    record = RunRecord(
        config=SimpleNamespace(),
        result=RunResult(
            state=state,
            diagnostics={},
            timings=(),
            cache_hit_count=0,
            config_hash="plot",
        ),
        run_dir=tmp_path,
        coord_tuple=(),
    )
    collection = RunCollection(
        config=SimpleNamespace(),
        records=(record,),
        output_dir=tmp_path,
    )

    written = cophasing_unitary(collection, tmp_path, ("png",))

    out = tmp_path / "plots" / "cophasing_unitary.png"
    assert out in written
    assert out.exists()
    assert out.stat().st_size > 3_000
