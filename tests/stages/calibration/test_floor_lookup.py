"""Tests for the CalibrationFloorLookup stage (SPEC §15.3 Pass 4c).

Reference invariants come from KILO
``kilo.studies.calibration_time.lookup_calibration_null`` in
``kilo/studies/calibration_time.py``
(lines 81-188). See ``tests/stages/notes/calibration_floor.md`` for the
source survey.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from nullsim.pipeline import (
    RunContext,
    Scene,
    SimulationState,
    list_stages,
)
from nullsim.stages.calibration import CalibrationFloorLookup


def _context(tmp_path: Path, *, source_path: Path | None = None) -> RunContext:
    return RunContext(
        rng=np.random.default_rng(0),
        config_hash="calibration-floor-lookup-pinning",
        sweep_coord_tuple=(),
        instance_id="calibration",
        run_dir=tmp_path,
        source_path=source_path,
    )


def _state_with_detector(
    *,
    null_depth_raw: float = 1.0e-5,
    integration_time_s: float = 1800.0,
    star_jmag: float | None = 5.0,
    n_modes: int | None = None,
    n_bright_ports: int | None = None,
    n_dark_ports: int | None = None,
    n_spectro_ports: int | None = None,
) -> SimulationState:
    state = SimulationState.empty({"wavelength_grid_um": (1.25,)})
    if star_jmag is not None:
        state = replace(state, scene=Scene(star_jmag=star_jmag))
    detector_payload = {
        "null_depth_raw": float(null_depth_raw),
        "integration_time_s": float(integration_time_s),
        "port_rates": np.array([1.0e6, 1.0, 1.0, 1.0], dtype=np.float64),
        "port_labels": ("bright_0", "dark_0", "dark_1", "dark_2"),
    }
    if n_modes is not None:
        detector_payload["n_modes"] = int(n_modes)
    if n_bright_ports is not None:
        detector_payload["n_bright_ports"] = int(n_bright_ports)
    if n_dark_ports is not None:
        detector_payload["n_dark_ports"] = int(n_dark_ports)
    if n_spectro_ports is not None:
        detector_payload["n_spectro_ports"] = int(n_spectro_ports)
    results = state.results.add_component(
        "detector.mkid",
        detector_payload,
    )
    return replace(state, results=results)


def _stage(
    context: RunContext,
    *,
    lookup_table_path: str | None = None,
    default_floor_log10: float = -6.0,
    band: str = "J",
) -> CalibrationFloorLookup:
    return CalibrationFloorLookup(
        {
            "lookup_table_path": lookup_table_path,
            "default_floor_log10": default_floor_log10,
            "band": band,
        },
        context,
    )


def test_calibration_registers_stage_contract(tmp_path: Path) -> None:
    stage = _stage(_context(tmp_path))

    assert "calibration.floor_lookup" in list_stages()
    assert stage.type_name == "calibration.floor_lookup"
    assert stage.family == "calibration"
    assert stage.consumes == frozenset({"results", "scene"})
    assert stage.produces == frozenset({"results"})


def test_calibration_default_floor_subtracts_one_microniull(tmp_path: Path) -> None:
    # raw = 1e-5, floor = 10**-6 = 1e-6, residual = 9e-6.
    state = _state_with_detector(null_depth_raw=1.0e-5)
    stage = _stage(_context(tmp_path))
    result = stage.apply(state)
    component = result.results.components["calibration.floor_lookup"]
    assert component["calibration_floor"] == pytest.approx(1.0e-6)
    assert component["null_depth_residual"] == pytest.approx(9.0e-6, rel=1e-12)
    assert component["null_depth_calibrated"] == pytest.approx(9.0e-6, rel=1e-12)
    assert component["null_depth_raw"] == pytest.approx(1.0e-5)


def test_calibration_clamps_at_zero(tmp_path: Path) -> None:
    # raw = 5e-7 < floor = 1e-6, so calibrated = 0 (not negative).
    state = _state_with_detector(null_depth_raw=5.0e-7)
    stage = _stage(_context(tmp_path))
    result = stage.apply(state)
    component = result.results.components["calibration.floor_lookup"]
    assert component["null_depth_residual"] == 0.0
    assert component["null_depth_calibrated"] == 0.0


def test_calibration_custom_default_floor(tmp_path: Path) -> None:
    # Negative log10 corresponds to a smaller absolute floor;
    # raw = 1e-5, floor = 10**-7 = 1e-7, calibrated = 9.9e-6.
    state = _state_with_detector(null_depth_raw=1.0e-5)
    stage = _stage(_context(tmp_path), default_floor_log10=-7.0)
    result = stage.apply(state)
    component = result.results.components["calibration.floor_lookup"]
    assert component["calibration_floor"] == pytest.approx(1.0e-7)
    assert component["null_depth_residual"] == pytest.approx(9.9e-6, rel=1e-12)
    assert component["null_depth_calibrated"] == pytest.approx(9.9e-6, rel=1e-12)


def test_calibration_lookup_exact_match(tmp_path: Path) -> None:
    # Build a tiny calibration_time_J.json analogue and confirm the
    # stage retrieves the exact (J5, 30min) cell.
    table_path = tmp_path / "calibration_time_J.json"
    table_path.write_text(
        json.dumps(
            {
                "analysis": {
                    "J5": {
                        "30min": {"best_null_depth": 1.2e-6},
                        "60min": {"best_null_depth": 8.7e-7},
                    },
                    "J7": {
                        "30min": {"best_null_depth": 2.1e-6},
                        "60min": {"best_null_depth": 1.6e-6},
                    },
                }
            }
        )
    )
    state = _state_with_detector(
        null_depth_raw=5.0e-6, integration_time_s=1800.0, star_jmag=5.0
    )
    stage = _stage(_context(tmp_path), lookup_table_path=str(table_path))
    result = stage.apply(state)
    component = result.results.components["calibration.floor_lookup"]
    assert component["calibration_floor"] == pytest.approx(1.2e-6)
    assert component["null_depth_residual"] == pytest.approx(5.0e-6 - 1.2e-6, rel=1e-12)
    assert component["null_depth_calibrated"] == pytest.approx(5.0e-6 - 1.2e-6, rel=1e-12)
    assert "exact match" in component["source_model"]


def test_calibration_lookup_resolves_table_relative_to_config(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    table_dir = config_dir / "tables"
    table_dir.mkdir(parents=True)
    table_path = table_dir / "calibration_time_J.json"
    table_path.write_text(
        json.dumps(
            {
                "analysis": {
                    "J5": {
                        "30min": {"best_null_depth": 1.2e-6},
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    state = _state_with_detector(
        null_depth_raw=5.0e-6,
        integration_time_s=1800.0,
        star_jmag=5.0,
    )
    stage = _stage(
        _context(tmp_path, source_path=config_dir / "config.toml"),
        lookup_table_path="tables/calibration_time_J.json",
    )

    dependencies = stage.external_dependencies()
    result = stage.apply(state)
    component = result.results.components["calibration.floor_lookup"]

    assert dependencies[0].path == table_path.resolve()
    assert component["calibration_floor"] == pytest.approx(1.2e-6)
    assert component["null_depth_residual"] == pytest.approx(3.8e-6)


def test_calibration_lookup_reads_bundled_nayra_table(tmp_path: Path) -> None:
    state = _state_with_detector(
        null_depth_raw=2.0e-6,
        integration_time_s=28800.0,
        star_jmag=5.0,
        n_modes=18,
        n_bright_ports=4,
        n_dark_ports=13,
        n_spectro_ports=1,
    )
    stage = _stage(
        _context(tmp_path),
        lookup_table_path="nullsim/data/nayra_reference/calibration_time_J_M18.json",
    )

    result = stage.apply(state)
    component = result.results.components["calibration.floor_lookup"]

    assert component["calibration_floor"] == pytest.approx(2.0e-7)
    assert component["null_depth_residual"] == pytest.approx(1.8e-6)
    assert component["null_depth_calibrated"] == pytest.approx(1.8e-6)


def test_calibration_lookup_rejects_detector_table_metadata_mismatch(
    tmp_path: Path,
) -> None:
    state = _state_with_detector(
        null_depth_raw=2.0e-6,
        integration_time_s=28800.0,
        star_jmag=5.0,
        n_modes=36,
        n_bright_ports=9,
        n_dark_ports=26,
        n_spectro_ports=1,
    )
    stage = _stage(
        _context(tmp_path),
        lookup_table_path="nullsim/data/nayra_reference/calibration_time_J_M18.json",
    )

    with pytest.raises(ValueError, match="metadata does not match upstream detector"):
        stage.apply(state)


def test_calibration_lookup_interpolates_jmag(tmp_path: Path) -> None:
    # J=6 is midway between J5 and J7; log(30) on the J5 row gives
    # 1.2e-6 and on the J7 row 2.1e-6, so the linear interpolant in
    # jmag is 1.65e-6 (matching the geometric-grid sanity check below).
    table_path = tmp_path / "calibration_time_J.json"
    table_path.write_text(
        json.dumps(
            {
                "analysis": {
                    "J5": {
                        "30min": {"best_null_depth": 1.2e-6},
                        "60min": {"best_null_depth": 8.7e-7},
                    },
                    "J7": {
                        "30min": {"best_null_depth": 2.1e-6},
                        "60min": {"best_null_depth": 1.6e-6},
                    },
                }
            }
        )
    )
    state = _state_with_detector(
        null_depth_raw=5.0e-6, integration_time_s=1800.0, star_jmag=6.0
    )
    stage = _stage(_context(tmp_path), lookup_table_path=str(table_path))
    result = stage.apply(state)
    component = result.results.components["calibration.floor_lookup"]
    # Linear interpolation in jmag at 30 min:
    expected = 0.5 * (1.2e-6 + 2.1e-6)
    assert component["calibration_floor"] == pytest.approx(expected, rel=1e-12)


def test_calibration_lookup_interpolates_budget(tmp_path: Path) -> None:
    # 45 min sits between 30 min and 60 min in log space:
    # log(45) = 0.5 * (log(30) + log(60)) - 1/2 * log(2/1) actually,
    # log(45) - log(30) = log(45/30) = log(1.5), and
    # log(60) - log(30) = log(2). Weight = log(1.5)/log(2) ~ 0.585.
    import math as _m
    table_path = tmp_path / "calibration_time_J.json"
    table_path.write_text(
        json.dumps(
            {
                "analysis": {
                    "J5": {
                        "30min": {"best_null_depth": 1.0e-6},
                        "60min": {"best_null_depth": 5.0e-7},
                    },
                }
            }
        )
    )
    state = _state_with_detector(
        null_depth_raw=5.0e-6, integration_time_s=2700.0, star_jmag=5.0
    )
    stage = _stage(_context(tmp_path), lookup_table_path=str(table_path))
    result = stage.apply(state)
    component = result.results.components["calibration.floor_lookup"]
    weight = _m.log(45.0 / 30.0) / _m.log(60.0 / 30.0)
    expected = (1.0 - weight) * 1.0e-6 + weight * 5.0e-7
    assert component["calibration_floor"] == pytest.approx(expected, rel=1e-12)


def test_calibration_lookup_falls_back_outside_grid(tmp_path: Path) -> None:
    # Requesting jmag = 9 (outside [5, 7]) falls back to the default floor.
    table_path = tmp_path / "calibration_time_J.json"
    table_path.write_text(
        json.dumps(
            {
                "analysis": {
                    "J5": {
                        "30min": {"best_null_depth": 1.2e-6},
                    },
                    "J7": {
                        "30min": {"best_null_depth": 2.1e-6},
                    },
                }
            }
        )
    )
    state = _state_with_detector(
        null_depth_raw=5.0e-6, integration_time_s=1800.0, star_jmag=9.0
    )
    stage = _stage(_context(tmp_path), lookup_table_path=str(table_path))
    result = stage.apply(state)
    component = result.results.components["calibration.floor_lookup"]
    assert component["calibration_floor"] == pytest.approx(1.0e-6)
    assert "fallback" in component["source_model"]


def test_calibration_lookup_missing_table_raises(tmp_path: Path) -> None:
    state = _state_with_detector()
    stage = _stage(_context(tmp_path), lookup_table_path=str(tmp_path / "no_such.json"))
    with pytest.raises(FileNotFoundError):
        stage.apply(state)


def test_calibration_lookup_bad_schema_raises(tmp_path: Path) -> None:
    table_path = tmp_path / "bad.json"
    table_path.write_text(json.dumps({"unexpected_root": {}}))
    state = _state_with_detector()
    stage = _stage(_context(tmp_path), lookup_table_path=str(table_path))
    with pytest.raises(ValueError, match="analysis"):
        stage.apply(state)


def test_calibration_lookup_requires_star_jmag(tmp_path: Path) -> None:
    table_path = tmp_path / "calibration_time_J.json"
    table_path.write_text(json.dumps({"analysis": {"J5": {"30min": {"best_null_depth": 1e-6}}}}))
    state = _state_with_detector(star_jmag=None)
    stage = _stage(_context(tmp_path), lookup_table_path=str(table_path))
    with pytest.raises(ValueError, match="star_jmag"):
        stage.apply(state)


def test_calibration_missing_detector_component_raises(tmp_path: Path) -> None:
    # No upstream detector component → the stage cannot find a raw null.
    state = SimulationState.empty({"wavelength_grid_um": (1.25,)})
    stage = _stage(_context(tmp_path))
    with pytest.raises(ValueError, match="detector"):
        stage.apply(state)


def test_calibration_bad_band_rejected_by_params() -> None:
    with pytest.raises(ValidationError):
        CalibrationFloorLookup.Params(band="K")


def test_calibration_positive_default_floor_log10_rejected() -> None:
    # default_floor_log10 must be <= 0 (it's an *exponent*; +1 means
    # the floor is 10× the raw null which makes no physical sense).
    with pytest.raises(ValidationError):
        CalibrationFloorLookup.Params(default_floor_log10=1.0)


def test_calibration_records_provenance(tmp_path: Path) -> None:
    state = _state_with_detector(null_depth_raw=2.0e-6)
    stage = _stage(_context(tmp_path))
    result = stage.apply(state)
    component = result.results.components["calibration.floor_lookup"]
    assert component["upstream_detector_component"] == "detector.mkid"
    assert component["band"] == "J"
    assert "default floor" in component["source_model"]
