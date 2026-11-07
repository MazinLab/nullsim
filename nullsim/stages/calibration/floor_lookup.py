"""Calibration-floor lookup stage (SPEC §15.3 Pass 4c).

Reference implementation: KILO
``kilo.studies.calibration_time.lookup_calibration_null`` lines 81-188 in
``kilo/studies/calibration_time.py``.
Calibration is *not* a real-time stage in KILO — the trade study
``kilo.studies.calibration_time`` runs many BFGS restarts on a synthetic
calibration star and writes a ``calibration_time_{Y,J,H}.json`` table of
``best_null_depth`` keyed on ``(jmag, budget_min)``. The runtime helper
``lookup_calibration_null`` reads the table for the requested band and
interpolates inside it. NAYRA reuses the KILO J-band table directly.

The nullsim stage exposes both paths via a single Param:

- ``lookup_table_path is not None``: read the JSON, find the calibration
  floor for ``(scene.star_jmag, integration_time_s)`` in the requested
  band. Exact-match first, linear interpolation in ``(jmag,
  log(budget_min))`` between bracketing entries when the exact cell is
  missing, fallback to ``10 ** default_floor_log10`` when the point
  falls outside the tabulated grid.
- ``lookup_table_path is None``: use ``10 ** default_floor_log10`` (the
  pre-2026-05 paperv2 design floor of 1e-6 by default).

The stage reports two distinct calibration quantities:

* ``calibration_floor``: the calibration-systematics floor looked up from
  KILO/NAYRA calibration products, or the configured default.
* ``null_depth_residual``: ``max(null_depth_raw - calibration_floor, 0.0)``.

The legacy key ``null_depth_calibrated`` is retained as an alias for
``null_depth_residual`` so older plots and example tests keep working, but
new code should read the explicit names above. Keeping the floor scalar
separate avoids conflating KILO's published calibration floor with nullsim's
post-subtraction residual.

See ``tests/stages/notes/calibration_floor.md`` for the full reference
survey, JSON schema, and default-floor justification.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field as PydanticField

import hashlib

from nullsim.pipeline import (
    BaseStage,
    ExternalDependency,
    SimulationState,
    register_stage,
)
from nullsim.stages._utils import unique_component_name


__all__ = ["CalibrationFloorLookup"]


@register_stage
class CalibrationFloorLookup(BaseStage):
    """Subtract a calibration-systematic null floor from the raw null depth."""

    type_name = "calibration.floor_lookup"
    family = "calibration"
    consumes = frozenset({"results", "scene"})
    produces = frozenset({"results"})

    class Params(BaseModel):
        """Validated params for the calibration-floor lookup stage."""

        model_config = ConfigDict(
            extra="forbid",
            frozen=True,
            allow_inf_nan=False,
        )

        lookup_table_path: str | None = None
        default_floor_log10: float = PydanticField(default=-6.0, le=0.0)
        band: Literal["I", "Y", "J", "H"] = "J"

    def apply(self, state: SimulationState) -> SimulationState:
        params = self.params_model

        detector_name, detector_component = _find_detector_component(state)
        null_depth_raw = float(detector_component["null_depth_raw"])
        integration_time_s = float(detector_component["integration_time_s"])

        floor_default = 10.0 ** params.default_floor_log10
        if params.lookup_table_path is None:
            calibration_floor = floor_default
            source_model = (
                f"nullsim CalibrationFloorLookup default floor "
                f"(default_floor_log10={params.default_floor_log10:+.2f}); no "
                "lookup table supplied. Reference: KILO "
                "kilo/studies/calibration_time.py lookup_calibration_null "
                "(lines 81-188)."
            )
        else:
            calibration_floor, source_model = _lookup_floor_from_table(
                table_path=_resolve_lookup_table_path(
                    params.lookup_table_path,
                    config_dir=_config_dir_from_context(self.context),
                ),
                band=params.band,
                jmag=_require_star_jmag(state),
                integration_time_s=integration_time_s,
                fallback_floor=floor_default,
                default_floor_log10=params.default_floor_log10,
                expected_metadata=_detector_expected_metadata(detector_component),
            )

        # Subtractive floor: raw - calibration. When the raw is deeper
        # than the floor (raw < floor), clip to zero rather than going
        # negative. Earlier this clipped silently — but raw < floor is
        # diagnostically meaningful (the chip beat the calibration
        # limit) and downstream consumers treating null=0 as "infinite
        # SNR" would be misled. Log when the clip fires.
        raw_minus_floor = null_depth_raw - calibration_floor
        if raw_minus_floor < 0.0:
            logging.warning(
                "calibration.floor_lookup: null_depth_raw=%.3e is deeper "
                "than calibration_floor=%.3e; clipping null_depth_residual "
                "to zero. Downstream SNR consumers will see no leakage "
                "from this stage — verify the calibration table is right "
                "for the observation regime.",
                null_depth_raw, calibration_floor,
            )
        null_depth_residual = max(raw_minus_floor, 0.0)

        component_name = unique_component_name(
            self.type_name,
            state.results.components,
            self.context.instance_id,
        )
        results = state.results.add_component(
            component_name,
            {
                "null_depth_raw": null_depth_raw,
                "calibration_floor": float(calibration_floor),
                "null_depth_residual": float(null_depth_residual),
                # Backward-compatible alias. New consumers should use
                # ``null_depth_residual`` for the post-subtraction value and
                # ``calibration_floor`` for the KILO/NAYRA calibration scalar.
                "null_depth_calibrated": float(null_depth_residual),
                "band": params.band,
                "upstream_detector_component": detector_name,
                "integration_time_s": integration_time_s,
                "source_model": source_model,
            },
        )
        return replace(state, results=results)

    def external_dependencies(self) -> list[ExternalDependency]:
        """Declare the lookup table file so its bytes enter the cache key
        AND the manifest's run_hash (Finding 4 fix).

        Pre-fix, the table was read at runtime via params.lookup_table_path
        but not declared as an external dependency, so an edit to the
        on-disk table would silently reuse a cached null_depth_residual
        and yield the same run_hash.

        We precompute the SHA-256 of the file bytes for a stable manifest
        fingerprint; the shared dependency digest also hashes local path
        bytes when present, so cache keys and CLI run hashes see the same
        external resource identity.
        """

        path_str = self.params_model.lookup_table_path
        if path_str is None:
            return []
        path = _resolve_lookup_table_path(
            path_str,
            config_dir=_config_dir_from_context(self.context),
        )
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            # Missing/unreadable file: declare the path with no content hash.
            # The cache will fail to read it at digest time and surface the
            # error; the run hash will be unchanged but downstream code
            # will hit the same OSError when apply() tries to read it.
            return [ExternalDependency(source="calibration.floor_lookup_table", path=path)]
        return [
            ExternalDependency(
                source="calibration.floor_lookup_table",
                path=path,
                content_hash=digest,
            )
        ]


def _find_detector_component(state: SimulationState) -> tuple[str, Any]:
    """Return the newest ``detector.*`` component on ``state.results``.

    Matches the convention from ``nullsim.stages.detector.mkid`` and
    ``nullsim.stages.detector.snspd`` which both write a single
    ``detector.<type>`` component with ``null_depth_raw`` and
    ``integration_time_s`` keys.
    """

    for name, value in reversed(state.results.components.items()):
        if not name.startswith("detector."):
            continue
        if not isinstance(value, dict):
            continue
        if "null_depth_raw" in value and "integration_time_s" in value:
            return name, value
    raise ValueError(
        "calibration.floor_lookup requires an upstream detector stage "
        "(detector.mkid or detector.snspd) to have written a "
        "'detector.*' component to state.results with 'null_depth_raw' "
        "and 'integration_time_s' fields."
    )


def _require_star_jmag(state: SimulationState) -> float:
    """Read the calibration-relevant J magnitude from ``state.scene``.

    KILO's calibration_time study keys on the calibration star's J-band
    apparent magnitude. The nullsim stage assumes ``scene.star_jmag``
    refers to the same star (or to a comparable calibrator) — presets
    that calibrate on a different source can override the scene before
    the stage runs.
    """

    jmag = state.scene.star_jmag
    if jmag is None:
        raise ValueError(
            "calibration.floor_lookup with lookup_table_path requires "
            "scene.star_jmag to be set so the table can be indexed by "
            "calibration star J-band magnitude."
        )
    return float(jmag)


def _detector_expected_metadata(detector_component: Any) -> dict[str, int]:
    if not isinstance(detector_component, dict):
        return {}

    expected: dict[str, int] = {}
    for key in (
        "n_modes",
        "n_bright_ports",
        "n_dark_ports",
        "n_spectro_ports",
    ):
        value = detector_component.get(key)
        if value is not None:
            expected[key] = int(value)

    if "n_modes" not in expected:
        port_count_keys = ("n_bright_ports", "n_dark_ports", "n_spectro_ports")
        if all(key in expected for key in port_count_keys):
            expected["n_modes"] = sum(expected[key] for key in port_count_keys)

    return expected


def _lookup_floor_from_table(
    *,
    table_path: Path,
    band: str,
    jmag: float,
    integration_time_s: float,
    fallback_floor: float,
    default_floor_log10: float,
    expected_metadata: dict[str, int] | None = None,
) -> tuple[float, str]:
    """Look up a calibration floor in the KILO calibration_time JSON.

    Returns ``(calibration_floor, source_model_string)``. Falls back to
    ``fallback_floor`` (and an explanatory ``source_model``) when the
    table does not cover the requested ``(jmag, budget_min)`` point.
    """

    if not table_path.exists():
        raise FileNotFoundError(
            f"calibration.floor_lookup: lookup table {table_path} does "
            "not exist. Either supply a path to a "
            "calibration_time_{band}.json produced by KILO "
            "kilo.studies.calibration_time, or omit lookup_table_path to "
            "use the default_floor_log10 fallback."
        )

    with table_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    _validate_table_metadata(
        table_path,
        data.get("metadata"),
        expected_metadata or {},
    )

    analysis = data.get("analysis")
    if not isinstance(analysis, dict):
        raise ValueError(
            f"calibration.floor_lookup: {table_path} does not have the "
            "expected schema (root key 'analysis' missing or not a "
            "mapping). See tests/stages/notes/calibration_floor.md for "
            "the accepted format."
        )

    budget_min = integration_time_s / 60.0
    # Code Review 2026-05-16 Issue 16 / Ultra-review Medium #3:
    # only take the exact-cell fast path when ``jmag`` and ``budget_min``
    # are integer-aligned (within numerical tolerance). Pre-fix
    # ``int(round(jmag))`` silently snapped 5.4 → 5, returning the
    # exact J5/30min floor instead of interpolating between the
    # surrounding grid cells.
    exact_mag_key = f"J{int(round(jmag))}"
    exact_budget_key = f"{int(round(budget_min))}min"
    near_int = (
        abs(jmag - round(jmag)) < 1.0e-9
        and abs(budget_min - round(budget_min)) < 1.0e-9
    )
    if near_int:
        mag_cell = analysis.get(exact_mag_key)
        if isinstance(mag_cell, dict):
            exact_cell = mag_cell.get(exact_budget_key)
            if isinstance(exact_cell, dict) and "best_null_depth" in exact_cell:
                return (
                    float(exact_cell["best_null_depth"]),
                    _table_source_model(
                        table_path,
                        band,
                        exact_mag_key,
                        exact_budget_key,
                        exact=True,
                    ),
                )

    floor = _interpolate_floor(
        analysis=analysis,
        jmag=jmag,
        budget_min=budget_min,
    )
    if floor is not None:
        return (
            float(floor),
            _table_source_model(
                table_path, band, exact_mag_key, exact_budget_key, exact=False
            ),
        )

    return (
        fallback_floor,
        (
            f"nullsim CalibrationFloorLookup fallback: requested "
            f"({exact_mag_key}, {exact_budget_key}) is outside the "
            f"tabulated grid of {table_path}; using "
            f"10**default_floor_log10={fallback_floor:.3e} "
            f"(default_floor_log10={default_floor_log10:+.2f})."
        ),
    )


def _validate_table_metadata(
    table_path: Path,
    metadata: Any,
    expected_metadata: dict[str, int],
) -> None:
    if not expected_metadata or not isinstance(metadata, dict):
        return

    mismatches: list[str] = []
    for key in (
        "n_modes",
        "n_bright_ports",
        "n_dark_ports",
        "n_spectro_ports",
    ):
        if key not in expected_metadata or key not in metadata:
            continue
        try:
            actual = int(metadata[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"calibration.floor_lookup: {table_path} metadata key {key!r} "
                f"must be an integer, got {metadata[key]!r}."
            ) from exc
        expected = int(expected_metadata[key])
        if actual != expected:
            mismatches.append(f"{key}: table={actual}, detector={expected}")

    if mismatches:
        joined = "; ".join(mismatches)
        raise ValueError(
            f"calibration.floor_lookup: lookup table {table_path} metadata "
            f"does not match upstream detector ({joined}). Use a calibration "
            "table generated for this mode/port layout, or omit "
            "lookup_table_path to use default_floor_log10."
        )


def _resolve_lookup_table_path(
    path_str: str,
    *,
    config_dir: Path | None = None,
) -> Path:
    """Resolve ``lookup_table_path`` to a real filesystem path.

    Absolute paths are returned as-is. Relative paths are tried first against
    the config file's directory when known, then against the current working
    directory, then against the ``nullsim`` package parent directory so bundled
    tables under ``nullsim/data/kilo_reference/`` resolve regardless of where
    the user is running from.
    """

    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate
    if config_dir is not None:
        config_relative = (config_dir / candidate).resolve()
        if config_relative.exists():
            return config_relative
    if candidate.exists():
        return candidate
    import nullsim

    package_root = Path(nullsim.__file__).resolve().parent.parent
    bundled = package_root / candidate
    if bundled.exists():
        return bundled
    # Neither location worked — return the cwd-relative path; callers
    # will surface the missing-file error with a useful message.
    return candidate


def _config_dir_from_context(context: Any) -> Path | None:
    source_path = getattr(context, "source_path", None)
    if source_path is None:
        return None
    return Path(source_path).expanduser().resolve().parent


def _table_source_model(
    table_path: Path, band: str, mag_key: str, budget_key: str, *, exact: bool
) -> str:
    mode = "exact match" if exact else "linear interpolation in (jmag, log(budget_min))"
    return (
        f"KILO kilo/studies/calibration_time.py lookup_calibration_null "
        f"(lines 81-188), band {band}, {mode} for ({mag_key}, {budget_key}) "
        f"in {table_path.name}."
    )


def _parse_mag(key: str) -> float | None:
    if not key.startswith("J"):
        return None
    try:
        return float(key[1:])
    except ValueError:
        return None


def _parse_budget(key: str) -> float | None:
    if not key.endswith("min"):
        return None
    try:
        return float(key[:-3])
    except ValueError:
        return None


def _interpolate_floor(
    *,
    analysis: dict[str, Any],
    jmag: float,
    budget_min: float,
) -> float | None:
    """Linearly interpolate the best_null_depth surface in (jmag, log(budget)).

    Returns ``None`` when the requested point falls outside the convex
    hull of the tabulated grid (caller falls back to the default floor).
    Exact-axis endpoints are accepted (the grid is half-open, not open).
    """

    if budget_min <= 0:
        return None

    # Per-magnitude budget curves: each magnitude row interpolates in
    # log(budget_min) first, then the two bracketing magnitudes are
    # linearly combined.
    mag_rows: list[tuple[float, dict[str, Any]]] = []
    for key, value in analysis.items():
        mag = _parse_mag(key)
        if mag is None or not isinstance(value, dict):
            continue
        mag_rows.append((mag, value))
    if not mag_rows:
        return None
    mag_rows.sort(key=lambda item: item[0])

    log_budget = math.log(budget_min)
    interpolated_rows: list[tuple[float, float]] = []
    for mag, row in mag_rows:
        floor_at_mag = _interpolate_budget_row(row, log_budget)
        if floor_at_mag is not None:
            interpolated_rows.append((mag, floor_at_mag))
    if not interpolated_rows:
        return None

    # Single magnitude row: only acceptable if jmag matches it exactly,
    # otherwise the caller falls back. Strict-outside check on the
    # multi-row case mirrors KILO's lookup which refuses extrapolation.
    if len(interpolated_rows) == 1:
        only_mag, only_floor = interpolated_rows[0]
        if jmag == only_mag:
            return only_floor
        return None

    if jmag < interpolated_rows[0][0] or jmag > interpolated_rows[-1][0]:
        return None

    for (mag_lo, floor_lo), (mag_hi, floor_hi) in zip(
        interpolated_rows, interpolated_rows[1:]
    ):
        if mag_lo <= jmag <= mag_hi:
            if mag_hi == mag_lo:
                return floor_lo
            weight = (jmag - mag_lo) / (mag_hi - mag_lo)
            return (1.0 - weight) * floor_lo + weight * floor_hi
    return None


def _interpolate_budget_row(row: dict[str, Any], log_budget: float) -> float | None:
    """Interpolate a single magnitude's budget curve in log(budget_min).

    Exact-axis endpoints are accepted. A single-point row only returns
    that point when ``log_budget`` matches it exactly.
    """

    points: list[tuple[float, float]] = []
    for budget_key, cell in row.items():
        budget_min = _parse_budget(budget_key)
        if budget_min is None or budget_min <= 0:
            continue
        if not isinstance(cell, dict):
            continue
        floor = cell.get("best_null_depth")
        if floor is None:
            continue
        points.append((math.log(budget_min), float(floor)))
    if not points:
        return None
    points.sort(key=lambda item: item[0])

    if len(points) == 1:
        only_log, only_floor = points[0]
        if log_budget == only_log:
            return only_floor
        return None

    if log_budget < points[0][0] or log_budget > points[-1][0]:
        return None

    for (lo_x, lo_y), (hi_x, hi_y) in zip(points, points[1:]):
        if lo_x <= log_budget <= hi_x:
            if hi_x == lo_x:
                return lo_y
            weight = (log_budget - lo_x) / (hi_x - lo_x)
            return (1.0 - weight) * lo_y + weight * hi_y
    return None
