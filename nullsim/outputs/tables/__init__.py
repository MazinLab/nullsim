"""Registered table outputs for nullsim run collections."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nullsim.outputs import RunCollection, RunRecord
from nullsim.pipeline import register_table
from nullsim.stages.characterization.acceptance import spectro_solution_accepted


@register_table
def performance_summary(
    collection: RunCollection,
    output_dir: Path,
    formats: Iterable[str],
) -> tuple[Path, ...]:
    """Write scalar performance metrics for each run collection record."""

    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    rows = [_summary_row(record) for record in collection.records]
    frame = pd.DataFrame(rows)

    written_paths: list[Path] = []
    for file_format in formats:
        output_path = table_dir / f"performance_summary.{file_format}"
        match file_format:
            case "csv":
                frame.to_csv(output_path, index=False)
            case "json":
                output_path.write_text(
                    json.dumps(rows, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            case "parquet":
                frame.to_parquet(output_path, index=False)
            case _:
                raise ValueError(
                    f"performance_summary does not support {file_format!r} output."
                )
        written_paths.append(output_path)
    return tuple(written_paths)


def _summary_row(record: RunRecord) -> dict[str, Any]:
    characterization = _find_characterization_component(record)
    row = record.coord_mapping
    row.update(
        {
            "config_hash": record.config.config_hash,
            "run_hash": record.config.run_hash,
            "null_depth": record.result.state.results.null_depth,
            "snr": record.result.state.results.snr,
            "calibration_time_s": record.result.state.results.calibration_time_s,
            "cache_hit_count": record.result.cache_hit_count,
            # Stellar-disk null floor and whether it is chip-realization
            # dependent (flux-scale-dependent observability objective). Read
            # the column across sweep cells to see the per-flux disk-floor
            # spread the chip optimizer's photon-rate-dependent terms induce.
            "disk_null_floor": _find_payload_value(
                record, "sensitivity.detection_curve", "disk_null_floor"
            ),
            "disk_null_floor_note": _find_payload_value(
                record, "chip_optimization", "disk_null_floor_note"
            ),
        }
    )
    if characterization is not None:
        _name, payload = characterization
        converged = spectro_solution_accepted(payload)
        row.update(
            {
                "char_converged": converged,
                "char_snr": payload.get("snr") if converged else float("nan"),
                "char_max_spectral_resolution": payload.get(
                    "max_spectral_resolution"
                )
                if converged
                else float("nan"),
                "char_spectro_throughput_design_bin": payload.get(
                    "spectro_throughput"
                ),
                "char_spectro_throughput_band_avg": _band_avg_throughput(payload),
                "char_spectro_null_depth": _payload_first(
                    payload, "spectro_band_on_axis_null_depth", "spectro_null_depth"
                ),
                "char_spectro_disk_leak": _payload_first(
                    payload, "spectro_band_disk_leak", "spectro_disk_leak"
                ),
                "char_spectro_leak_fraction": payload.get("spectro_leak_fraction"),
                "char_spectro_design_null_depth": payload.get("spectro_null_depth"),
                "char_spectro_design_disk_leak": payload.get("spectro_disk_leak"),
                "char_optimization_success": payload.get("optimization_success"),
                "char_optimization_constraint_margin": payload.get(
                    "optimization_constraint_margin"
                ),
                "char_planet_signal_rate_hz": (
                    payload.get("planet_signal_rate_hz") if converged else float("nan")
                ),
                "char_leakage_rate_hz": (
                    payload.get("leakage_rate_hz") if converged else float("nan")
                ),
                "char_spectrograph": payload.get("spectrograph"),
                "char_awg_resolving_power": payload.get("awg_resolving_power"),
                "char_awg_n_bins": payload.get("awg_n_bins"),
                "char_awg_min_snr_per_bin": payload.get("awg_min_snr_per_bin"),
                "char_mkid_resolving_power": payload.get("mkid_resolving_power"),
                "char_mkid_min_snr_per_bin": payload.get("mkid_min_snr_per_bin"),
            }
        )
    return row


def _band_avg_throughput(payload: dict[str, Any]) -> float | None:
    """Band-averaged planet chip throughput.

    Mean of the per-wavelength planet throughput across the characterization
    passband. The published characterization SNR is band-integrated, so this is
    the bandwidth-consistent throughput figure; the companion design-bin column
    is the chip efficiency at the single design wavelength only. Returns
    ``None`` when the per-wavelength throughput is unavailable.
    """

    values = payload.get("spectro_planet_throughput_per_wavelength")
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return None
    return float(finite.mean())


def _find_characterization_component(
    record: RunRecord,
) -> tuple[str, dict[str, Any]] | None:
    for name, payload in record.result.state.results.components.items():
        if name.startswith("characterization.spectro") and isinstance(payload, dict):
            return name, payload
    return None


def _find_payload_value(record: RunRecord, prefix: str, key: str) -> Any:
    """Return ``key`` from the first results component named ``prefix*``."""

    for name, payload in record.result.state.results.components.items():
        if name.startswith(prefix) and isinstance(payload, dict) and key in payload:
            return payload[key]
    return None


@register_table
def characterization_summary(
    collection: RunCollection,
    output_dir: Path,
    formats: Iterable[str],
) -> tuple[Path, ...]:
    """Write one row per characterization.spectro component."""

    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for record in collection.records:
        for name, payload in record.result.state.results.components.items():
            if not name.startswith("characterization.spectro"):
                continue
            if not isinstance(payload, dict):
                continue
            converged = spectro_solution_accepted(payload)
            row = record.coord_mapping
            row.update(
                {
                    "component": name,
                    "config_hash": record.config.config_hash,
                    "run_hash": record.config.run_hash,
                    "converged": converged,
                    "snr": payload.get("snr") if converged else float("nan"),
                    "max_spectral_resolution": payload.get(
                        "max_spectral_resolution"
                    )
                    if converged
                    else float("nan"),
                    "spectro_throughput_design_bin": payload.get(
                        "spectro_throughput"
                    ),
                    "spectro_throughput_band_avg": _band_avg_throughput(payload),
                    "spectro_null_depth": _payload_first(
                        payload, "spectro_band_on_axis_null_depth", "spectro_null_depth"
                    ),
                    "spectro_design_null_depth": payload.get("spectro_null_depth"),
                    "spectro_on_axis_null_depth": payload.get(
                        "spectro_on_axis_null_depth"
                    ),
                    "dark_null_depth": payload.get("dark_null_depth"),
                    "spectro_disk_leak": _payload_first(
                        payload, "spectro_band_disk_leak", "spectro_disk_leak"
                    ),
                    "spectro_design_disk_leak": payload.get("spectro_disk_leak"),
                    "spectro_band_on_axis_null_depth": payload.get(
                        "spectro_band_on_axis_null_depth"
                    ),
                    "spectro_band_disk_leak": payload.get("spectro_band_disk_leak"),
                    "spectro_leak_fraction": payload.get("spectro_leak_fraction"),
                    "optimization_success": payload.get("optimization_success"),
                    "optimization_constraint_margin": payload.get(
                        "optimization_constraint_margin"
                    ),
                    "planet_signal_rate_hz": (
                        payload.get("planet_signal_rate_hz")
                        if converged
                        else float("nan")
                    ),
                    "leakage_rate_hz": (
                        payload.get("leakage_rate_hz") if converged else float("nan")
                    ),
                    "detected_stellar_rate_hz": payload.get(
                        "detected_stellar_rate_hz"
                    ),
                    "spectrograph": payload.get("spectrograph"),
                    "awg_resolving_power": payload.get("awg_resolving_power"),
                    "awg_n_bins": payload.get("awg_n_bins"),
                    "awg_min_snr_per_bin": payload.get("awg_min_snr_per_bin"),
                    "awg_mean_snr_per_bin": payload.get("awg_mean_snr_per_bin"),
                    "awg_total_planet_rate_hz": payload.get(
                        "awg_total_planet_rate_hz"
                    ),
                    "awg_total_leakage_rate_hz": payload.get(
                        "awg_total_leakage_rate_hz"
                    ),
                    "awg_total_sky_rate_hz": payload.get("awg_total_sky_rate_hz"),
                    "mkid_resolving_power": payload.get("mkid_resolving_power"),
                    "mkid_resolving_power_ref": payload.get(
                        "mkid_resolving_power_ref"
                    ),
                    "mkid_lambda_ref_um": payload.get("mkid_lambda_ref_um"),
                    "mkid_min_snr_per_bin": payload.get("mkid_min_snr_per_bin"),
                }
            )
            row.update(_hdc_columns(record, name, converged))
            rows.append(row)
    frame = pd.DataFrame(rows)

    written_paths: list[Path] = []
    for file_format in formats:
        output_path = table_dir / f"characterization_summary.{file_format}"
        match file_format:
            case "csv":
                frame.to_csv(output_path, index=False)
            case "json":
                output_path.write_text(
                    json.dumps(rows, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            case "parquet":
                frame.to_parquet(output_path, index=False)
            case _:
                raise ValueError(
                    "characterization_summary does not support "
                    f"{file_format!r} output."
                )
        written_paths.append(output_path)
    return tuple(written_paths)


_HDC_DETECTION_PREFIX = "performance.hdc"


@register_table
def hdc_detection(
    collection: RunCollection,
    output_dir: Path,
    formats: Iterable[str],
) -> tuple[Path, ...]:
    """Write one row per ``performance.hdc`` separation anchor.

    Detection-mode HDC used to survive only as dash-dot lines inside the
    ``detection_space`` figure, which means the numbers behind the headline
    claim -- how much the cross-correlation buys over the broadband curve, at
    which separation, at what threshold and exposure -- could not be read back
    without re-running. Long format (one row per anchor) rather than one row
    per component, because the gain is a curve, not a scalar.
    """

    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for record in collection.records:
        for name, payload in record.result.state.results.components.items():
            if not name.startswith(_HDC_DETECTION_PREFIX):
                continue
            if not isinstance(payload, dict):
                continue
            rows.extend(_hdc_detection_rows(record, name, payload))
    frame = pd.DataFrame(rows)

    written_paths: list[Path] = []
    for file_format in formats:
        output_path = table_dir / f"hdc_detection.{file_format}"
        match file_format:
            case "csv":
                frame.to_csv(output_path, index=False)
            case "json":
                output_path.write_text(
                    json.dumps(rows, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            case "parquet":
                frame.to_parquet(output_path, index=False)
            case _:
                raise ValueError(
                    f"hdc_detection does not support {file_format!r} output."
                )
        written_paths.append(output_path)
    return tuple(written_paths)


def _hdc_detection_rows(
    record: RunRecord, name: str, payload: dict[str, Any]
) -> list[dict[str, Any]]:
    """Anchor rows for one ``performance.hdc`` payload.

    Anchors the stage published as non-detections keep their row: the
    separation was asked about and the answer was "no contrast reaches the
    threshold", which is information. Their contrast columns are NaN and
    ``non_detection_reason`` carries the stage's own explanation.
    """

    separations = _float_array(payload.get("separations_mas"))
    if separations.size == 0:
        return []
    ccf = _aligned(payload.get("ccf_contrast_at_target"), separations.size)
    broadband = _aligned(
        payload.get("broadband_contrast_at_target"), separations.size
    )
    gain = _aligned(payload.get("ccf_contrast_gain"), separations.size)
    reasons = payload.get("non_detection_reasons")
    reason_list = list(reasons) if isinstance(reasons, (list, tuple)) else []

    target_snr = _finite_or_nan(payload.get("comparison_target_snr"))
    if not np.isfinite(target_snr):
        target_snr = _finite_or_nan(payload.get("target_snr"))
    integration_time_s = _finite_or_nan(payload.get("comparison_time_s"))
    if not np.isfinite(integration_time_s):
        integration_time_s = _finite_or_nan(payload.get("integration_time_s"))

    rows: list[dict[str, Any]] = []
    for index in range(separations.size):
        row = record.coord_mapping
        row.update(
            {
                "component": name,
                "config_hash": record.config.config_hash,
                "run_hash": record.config.run_hash,
                "separation_mas": float(separations[index]),
                "ccf_contrast": float(ccf[index]),
                "broadband_contrast": float(broadband[index]),
                "ccf_contrast_gain": float(gain[index]),
                "target_snr": float(target_snr),
                "integration_time_s": float(integration_time_s),
                "resolving_power": _finite_or_nan(
                    payload.get("resolving_power_used")
                ),
                "non_detection_reason": (
                    reason_list[index] if index < len(reason_list) else None
                ),
            }
        )
        rows.append(row)
    return rows


def _float_array(values: Any) -> np.ndarray:
    if values is None:
        return np.empty(0, dtype=np.float64)
    array = np.asarray(values, dtype=np.float64)
    return array if array.ndim == 1 else np.empty(0, dtype=np.float64)


def _aligned(values: Any, size: int) -> np.ndarray:
    """``values`` as a float array of length ``size``, NaN-filled when absent."""

    array = _float_array(values)
    if array.size == size:
        return array
    return np.full(size, np.nan, dtype=np.float64)


def _finite_or_nan(value: Any) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


_HDC_PREFIX = "characterization.hdc_ccf"

# Table column -> hdc_ccf payload key.
_HDC_COLUMNS: tuple[tuple[str, str], ...] = (
    ("hdc_ccf_snr", "ccf_snr"),
    ("hdc_ccf_snr_photon_limited", "ccf_snr_photon_limited"),
    ("hdc_n_effective_lines", "n_effective_lines"),
    ("hdc_resolving_power", "resolving_power_used"),
    ("hdc_systematic_rejection", "ccf_systematic_rejection"),
)


def _find_hdc_payload(
    record: RunRecord, spectro_name: str
) -> dict[str, Any] | None:
    """Return the hdc_ccf payload computed from ``spectro_name``.

    The HDC stage records which spectro component it consumed in
    ``upstream_spectro_component``; match on it so a run with several spectro
    components does not smear one HDC result across unrelated rows. Payloads
    predating that key cannot be associated, so they fall back to first-match.
    """

    fallback: dict[str, Any] | None = None
    for name, payload in record.result.state.results.components.items():
        if not name.startswith(_HDC_PREFIX) or not isinstance(payload, dict):
            continue
        upstream = payload.get("upstream_spectro_component")
        if upstream == spectro_name:
            return payload
        if upstream is None and fallback is None:
            fallback = payload
    return fallback


def _hdc_columns(
    record: RunRecord, spectro_name: str, converged: bool
) -> dict[str, float]:
    """HDC cross-correlation scalars for one spectro row.

    NaN when no HDC component belongs to this row. Also NaN when the upstream
    spectro optimization did not converge: the CCF metrics are built from that
    solution's throughput, leakage, and photon budgets, so a failed or
    constraint-violating retune makes a finite CCF number describe a candidate
    the pipeline never accepted. Same reasoning as the masked ``snr`` and
    per-port rate columns on this row.
    """

    payload = _find_hdc_payload(record, spectro_name) if converged else None
    columns: dict[str, float] = {}
    for column, key in _HDC_COLUMNS:
        value = payload.get(key) if payload is not None else None
        columns[column] = float("nan") if value is None else float(value)
    return columns


def _payload_first(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


__all__ = ["characterization_summary", "hdc_detection", "performance_summary"]
