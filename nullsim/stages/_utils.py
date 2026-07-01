"""Shared helpers for built-in stage implementations.

These helpers exist to keep stage modules consistent on cross-cutting
conventions — currently the disambiguation rule for named contributions
written into a :class:`nullsim.pipeline.state.SubBudget`. SPEC §6.2 requires
each contribution name to be unique inside a sub-budget; when two instances
of the same stage type write to the same budget, the second instance's
contribution name must be suffixed with the instance id so neither
``add_component`` call raises.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from nullsim.physics.uv_coverage import project_positions_to_sky_plane_m


def torch_compile_env_enabled() -> bool:
    """Return True iff the NULLSIM_TORCH_COMPILE env var is truthy.

    Mirrors KILO's KILO_TORCH_COMPILE gate. Stage Params models resolve
    this at validation time (ORing it into ``compile_objective``) so the
    executed objective path is part of the validated params and therefore
    of the stage cache key — pre-fix, toggling the env var changed results
    without changing any cache key (review 2026-06-10 Issue 25).
    """

    return os.environ.get("NULLSIM_TORCH_COMPILE", "0").lower() in {
        "1", "true", "yes", "on",
    }


def unique_component_name(
    base_name: str,
    components: Mapping[str, Any],
    instance_id: str,
) -> str:
    """Return a contribution name that does not collide with existing keys.

    Args:
        base_name: Canonical name for the contribution. Used unchanged when
            no collision exists, so the common single-instance case stays
            greppable.
        components: Existing components mapping on the target sub-budget.
        instance_id: Pipeline instance id used as the disambiguating suffix.

    Returns:
        ``base_name`` if it is unused; otherwise ``base_name.<instance_id>``
        optionally suffixed with a counter to guarantee uniqueness.
    """

    if base_name not in components:
        return base_name
    suffix = instance_id or "instance"
    candidate = f"{base_name}.{suffix}"
    counter = 2
    while candidate in components:
        candidate = f"{base_name}.{suffix}.{counter}"
        counter += 1
    return candidate


def mode_origin_positions_m(
    geometry: Any,
    n_modes: int,
) -> NDArray[np.float64] | None:
    """Per-mode telescope-origin positions published by upstream stages.

    Scans geometry components (latest first) for a
    ``mode_origin_positions_m`` array with ``n_modes`` rows: each row is
    the ENU origin of the telescope whose pupil hosts that mode. Producers:
    ``ao.pyramid_wfs`` sub-aperture packing and ``injection.mode_splitter``
    (which expands origins alongside positions). Returns ``None`` when no
    producer ran (e.g. one mode per telescope), in which case the full
    legacy projection applies.
    """

    for payload in reversed(tuple(geometry.components.values())):
        if not isinstance(payload, Mapping):
            continue
        origins = payload.get("mode_origin_positions_m")
        if origins is None:
            continue
        arr = np.asarray(origins, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[0] == n_modes and arr.shape[1] >= 2:
            return arr[:, :2].astype(np.float64, copy=True)
    return None


def project_aperture_positions_for_pointing(
    positions_m: NDArray[np.floating],
    *,
    altitude_deg: float | None,
    azimuth_deg: float | None,
    aperture_origins_m: NDArray[np.floating] | None = None,
) -> NDArray[np.float64]:
    """Project aperture positions onto the sky plane for the pointing.

    Sky-angle phase rows (``2*pi/lambda * (x*alpha + y*beta)``) are only valid
    once the aperture positions are projected onto the plane perpendicular to
    the line of sight, which foreshortens baselines off-zenith. Stages call
    this at their boundary before handing positions to the planet-throughput /
    kernel-derivative / disk-sample builders.

    PHYSICS (review 2026-06-10 Issue 20): only the INTER-telescope ground
    baseline foreshortens. Sub-apertures are mounted in the telescope
    pupil, which tracks the target and stays perpendicular to the line of
    sight — intra-telescope pupil offsets must NOT foreshorten. When
    ``aperture_origins_m`` provides each mode's telescope origin, the
    origin is projected and the pupil offset is added unprojected. Pupil
    rotation relative to the sky (parallactic angle) is ignored — the
    snapshot convention this pipeline uses throughout. Without origins the
    legacy behavior (project everything) is retained, which is exact when
    each mode IS a telescope. KILO projects sub-aperture offsets too; this
    is a deliberate parity break in favor of the correct construction.

    When neither ``altitude_deg`` nor ``azimuth_deg`` is set (no pointing
    specified, i.e. effectively zenith), the positions are returned unchanged
    so legacy zenith configs are unaffected. A missing altitude or azimuth
    defaults to zenith (90 deg) / due North (0 deg) respectively.

    Args:
        positions_m: ``(M, 2)`` or ``(M, 3)`` pupil-frame ENU positions (m).
            ``(M, 2)`` is treated as ``Up = 0``.
        altitude_deg: Line-of-sight altitude (90 = zenith) or ``None``.
        azimuth_deg: Line-of-sight azimuth from North toward East or ``None``.
        aperture_origins_m: Optional ``(M, 2)`` or ``(M, 3)`` per-mode
            telescope-origin ENU positions (m), same frame as
            ``positions_m``.

    Returns:
        ``(M, 2)`` sky-plane positions (axis 0 azimuth, axis 1 altitude).
        Unchanged ``(M, 2)`` input when no pointing is specified.
    """

    positions = np.asarray(positions_m, dtype=np.float64)
    if altitude_deg is None and azimuth_deg is None:
        if positions.ndim == 2 and positions.shape[-1] >= 2:
            return positions[:, :2].astype(np.float64, copy=True)
        return positions.astype(np.float64, copy=True)
    resolved_altitude_deg = 90.0 if altitude_deg is None else float(altitude_deg)
    resolved_azimuth_deg = 0.0 if azimuth_deg is None else float(azimuth_deg)
    if aperture_origins_m is None:
        return project_positions_to_sky_plane_m(
            positions_enu_m=positions,
            altitude_deg=resolved_altitude_deg,
            azimuth_deg=resolved_azimuth_deg,
        )
    origins = np.asarray(aperture_origins_m, dtype=np.float64)
    if origins.ndim != 2 or origins.shape[0] != positions.shape[0]:
        raise ValueError(
            "project_aperture_positions_for_pointing: aperture_origins_m "
            f"shape {origins.shape} does not match positions "
            f"{positions.shape}."
        )
    projected_origins = project_positions_to_sky_plane_m(
        positions_enu_m=origins,
        altitude_deg=resolved_altitude_deg,
        azimuth_deg=resolved_azimuth_deg,
    )
    pupil_offsets = positions[:, :2] - origins[:, :2]
    return (projected_origins + pupil_offsets).astype(np.float64, copy=False)


def detector_n_pol_arms(state: Any) -> int:
    """Return the polarization-arm channel multiplicity the detector published.

    Dual-pol PBS-arm pipelines record each chip port as ``n_pol_arms``
    physical detector channels (port-major, arm-minor: ``bright_0_A,
    bright_0_B, ...``). Stages that reason in physical chip ports use this
    to pool or collapse the per-arm channels. Returns 1 when no detector
    payload declares polarization arms (scalar pipelines).
    """

    for name, payload in reversed(tuple(state.results.components.items())):
        if not name.startswith(("detector.mkid", "detector.snspd")):
            continue
        if not isinstance(payload, Mapping):
            continue
        value = payload.get("n_pol_arms")
        if isinstance(value, (int, float)) and int(value) >= 1:
            return int(value)
    return 1


__all__ = [
    "detector_n_pol_arms",
    "mode_origin_positions_m",
    "project_aperture_positions_for_pointing",
    "torch_compile_env_enabled",
    "unique_component_name",
]
