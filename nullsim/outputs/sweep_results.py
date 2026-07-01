"""Tidy long-format sweep result writer (SPEC §8.1.1).

The :func:`write_sweep_results` function emits one row per
(sweep cell × metric × wavelength bin) combination, with sweep coordinates and
a ``cell_run_hash`` linking back to the per-cell output directory.

The table stores headline ``Results`` scalars, every numeric scalar leaf from
stage result components, and spectra that need one row per physical bin.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from typing import Literal

import numpy as np
import pandas as pd

from nullsim.outputs import RunCollection, RunRecord

SWEEP_RESULTS_COLUMNS: tuple[str, ...] = (
    "sweep_coord",
    "cell_run_hash",
    "stage_instance",
    "metric_name",
    "metric_value",
    "wavelength_bin_um",
    "separation_mas",
)

_SCALAR_METRICS: tuple[tuple[str, str], ...] = (
    # (results.<attr>, stage_instance)
    ("null_depth", "results"),
    ("snr", "results"),
    ("calibration_time_s", "results"),
)


def write_sweep_results(
    collection: RunCollection,
    output_dir: Path,
    table_format: Literal["csv", "json", "parquet"] = "parquet",
) -> Path:
    """Write ``sweep_results`` for one run collection.

    Args:
        collection: Run records aggregated from a sweep (or a single cell).
        output_dir: Directory in which to write the sweep results table.
        table_format: Output format selected by ``[output].sweep_table``.

    Returns:
        The path to the written parquet file.
    """

    rows = list(_iter_rows(collection))
    frame = pd.DataFrame(rows, columns=list(SWEEP_RESULTS_COLUMNS))
    output_path = Path(output_dir) / f"sweep_results.{table_format}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if table_format == "parquet":
        frame.to_parquet(output_path, index=False)
    elif table_format == "csv":
        frame.to_csv(output_path, index=False)
    elif table_format == "json":
        frame.to_json(output_path, orient="records", indent=2)
    else:
        raise ValueError(f"Unsupported sweep table format {table_format!r}.")
    return output_path


def _iter_rows(collection: RunCollection) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in collection.records:
        coord_str = _serialize_coord(record.coord_tuple)
        cell_run_hash = _cell_run_hash(record)
        results = record.result.state.results
        for attr_name, stage_instance in _SCALAR_METRICS:
            value = getattr(results, attr_name, None)
            if value is None:
                continue
            rows.append(
                {
                    "sweep_coord": coord_str,
                    "cell_run_hash": cell_run_hash,
                    "stage_instance": stage_instance,
                    "metric_name": attr_name,
                    "metric_value": float(value),
                    "wavelength_bin_um": None,
                    "separation_mas": None,
                }
            )
        rows.extend(_photon_rate_rows(record, coord_str, cell_run_hash))
        rows.extend(_contrast_curve_rows(record, coord_str, cell_run_hash))
        rows.extend(_component_scalar_rows(record, coord_str, cell_run_hash))
    return rows


def _contrast_curve_rows(
    record: RunRecord,
    coord_str: str,
    cell_run_hash: str,
) -> list[dict[str, Any]]:
    """One row per (cell, separation) of the contrast curve.

    Contrast rows use the dedicated ``separation_mas`` column. Keeping
    ``wavelength_bin_um`` null avoids mixing length units in the table schema.
    """

    curve = record.result.state.results.contrast_curve
    if curve is None or curve.size == 0:
        return []
    seps = np.asarray(curve["separation_mas"], dtype=np.float64)
    contrasts = np.asarray(curve["contrast"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for sep, contrast in zip(seps, contrasts):
        rows.append(
            {
                "sweep_coord": coord_str,
                "cell_run_hash": cell_run_hash,
                "stage_instance": "contrast_curve",
                "metric_name": "contrast_at_separation_mas",
                "metric_value": float(contrast),
                "wavelength_bin_um": None,
                "separation_mas": float(sep),
            }
        )
    return rows


def _component_scalar_rows(
    record: RunRecord,
    coord_str: str,
    cell_run_hash: str,
) -> list[dict[str, Any]]:
    """Rows for every numeric scalar leaf in ``state.results.components``."""

    rows: list[dict[str, Any]] = []
    for name, payload in record.result.state.results.components.items():
        for key, value_float in _iter_numeric_leaves(payload):
            rows.append(
                {
                    "sweep_coord": coord_str,
                    "cell_run_hash": cell_run_hash,
                    "stage_instance": name,
                    "metric_name": key,
                    "metric_value": value_float,
                    "wavelength_bin_um": None,
                    "separation_mas": None,
                }
            )
    return rows


def _iter_numeric_leaves(
    value: Any,
    *,
    prefix: str = "",
) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_str = str(key)
            child_prefix = key_str if not prefix else f"{prefix}.{key_str}"
            rows.extend(_iter_numeric_leaves(child, prefix=child_prefix))
        return rows
    if not prefix:
        return rows
    scalar = _numeric_scalar(value)
    if scalar is None:
        return rows
    return [(prefix, scalar)]


def _numeric_scalar(value: Any) -> float | None:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            return None
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return None


def _photon_rate_rows(
    record: RunRecord,
    coord_str: str,
    cell_run_hash: str,
) -> list[dict[str, Any]]:
    photon_rates = record.result.state.photon_rates
    rates_hz = photon_rates.rates_hz
    if rates_hz is None or rates_hz.size == 0:
        return []
    wavelength_grid_um = photon_rates.wavelength_grid_um
    if not wavelength_grid_um:
        return []
    array = np.asarray(rates_hz, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    n_ports, n_bins = array.shape[:2]
    if n_bins != len(wavelength_grid_um):
        return []
    port_labels = photon_rates.port_labels or tuple(
        f"port_{port_index}" for port_index in range(n_ports)
    )
    rows: list[dict[str, Any]] = []
    for port_index in range(n_ports):
        stage_instance = (
            port_labels[port_index] if port_index < len(port_labels) else f"port_{port_index}"
        )
        for bin_index, wavelength_um in enumerate(wavelength_grid_um):
            rows.append(
                {
                    "sweep_coord": coord_str,
                    "cell_run_hash": cell_run_hash,
                    "stage_instance": f"photon_rates.{stage_instance}",
                    "metric_name": "rate_hz",
                    "metric_value": float(array[port_index, bin_index]),
                    "wavelength_bin_um": float(wavelength_um),
                    "separation_mas": None,
                }
            )
    return rows


def _serialize_coord(coord_tuple: tuple[tuple[str, Any], ...]) -> str:
    return json.dumps(
        [[str(axis_name), _json_safe(axis_value)] for axis_name, axis_value in coord_tuple],
        sort_keys=False,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    return repr(value)


def _cell_run_hash(record: RunRecord) -> str:
    config = record.config
    cell_hash = getattr(config, "run_hash", None) or getattr(config, "config_hash", None) or ""
    return str(cell_hash)


__all__ = ["SWEEP_RESULTS_COLUMNS", "write_sweep_results"]
