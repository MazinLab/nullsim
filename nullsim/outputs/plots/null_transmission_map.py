"""Optional null-transmission map plot output."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import matplotlib  # noqa: F401  # backend choice managed by styles module
import numpy as np

from nullsim.outputs.styles import _ensure_headless_backend  # noqa: F401

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from numpy.typing import NDArray

from nullsim.config.schema import NullTransmissionMapOutputConfig
from nullsim.outputs import RunCollection, RunRecord
from nullsim.outputs.plots._chip_field_inputs import (
    ChipFieldInputs,
    build_chip_field_inputs,
)
from nullsim.physics.imaging import MAS_PER_RAD, MAUNAKEA_LAT_RAD
from nullsim.physics.planet_throughput import dark_port_transmission_map
from nullsim.pipeline import register_plot
from nullsim.stages._utils import (
    mode_origin_positions_m,
    project_aperture_positions_for_pointing,
)

LOGGER = logging.getLogger(__name__)

_DEG_PER_HOUR = 15.0
_DEFAULT_DECLINATION_DEG = 20.0


@dataclass(frozen=True)
class _SkyProjection:
    """Mode positions and metadata for a static sky-response map."""

    positions_m: NDArray[np.float64]
    mode: str
    hour_angle_rad: float | None
    declination_rad: float | None
    latitude_rad: float | None


@register_plot
def null_transmission_map(
    collection: RunCollection,
    output_dir: Path,
    formats: Iterable[str],
) -> tuple[Path, ...]:
    """Write static summed-dark-port transmission maps."""

    if not collection.records:
        return ()

    config = (
        collection.config.output.null_transmission_map
        or NullTransmissionMapOutputConfig()
    )
    records = collection.records[: int(config.max_records)]
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for index, record in enumerate(records):
        try:
            maps = _maps_for_record(record, config)
        except ValueError as exc:
            LOGGER.warning("null_transmission_map: skipping record %s: %s", index, exc)
            continue

        stem = (
            "null_transmission_map"
            if len(records) == 1
            else f"null_transmission_map_{index:03d}"
        )
        fig = _plot_maps(
            design_map=maps["design_map"],
            band_averaged_map=maps["band_averaged_map"],
            l_coords_mas=maps["l_coords_mas"],
            m_coords_mas=maps["m_coords_mas"],
            design_wavelength_um=float(maps["design_wavelength_m"]) * 1.0e6,
            config=config,
            record=record,
        )
        for fmt in formats:
            path = plot_dir / f"{stem}.{fmt}"
            fig.savefig(path)
            written.append(path)
        plt.close(fig)

        if config.save_arrays:
            array_path = plot_dir / f"{stem}.npz"
            np.savez_compressed(array_path, **maps)
            written.append(array_path)

    return tuple(written)


def _maps_for_record(
    record: RunRecord,
    config: NullTransmissionMapOutputConfig,
) -> dict[str, Any]:
    inputs = build_chip_field_inputs(
        record.result.state,
        require_sensitivity=False,
        error_prefix="null_transmission_map",
    )
    fov_rad = float(config.fov_mas) / MAS_PER_RAD
    l_coords_rad = np.linspace(-fov_rad, fov_rad, int(config.n_pix))
    m_coords_rad = np.linspace(-fov_rad, fov_rad, int(config.n_pix))
    design_index = int(np.argmin(np.abs(inputs.wavelengths_m - inputs.wl_design_m)))
    sub_diameters = (
        inputs.sub_diameters_m if bool(config.include_angular_coupling) else None
    )
    projection = _sky_projection_for_record(record, config, inputs)
    design_map = dark_port_transmission_map(
        u_stack=inputs.u_stack,
        wavelengths_m=inputs.wavelengths_m,
        positions_m=projection.positions_m,
        dark_indices=inputs.dark_indices,
        coupling_per_sub=inputs.coupling_per_sub,
        sub_areas_m2=inputs.sub_areas_m2,
        sub_diameters_m=sub_diameters,
        l_coords_rad=l_coords_rad,
        m_coords_rad=m_coords_rad,
        wavelength_indices=(design_index,),
    )
    wavelength_indices = tuple(range(inputs.wavelengths_m.size))
    band_averaged_map = dark_port_transmission_map(
        u_stack=inputs.u_stack,
        wavelengths_m=inputs.wavelengths_m,
        positions_m=projection.positions_m,
        dark_indices=inputs.dark_indices,
        coupling_per_sub=inputs.coupling_per_sub,
        sub_areas_m2=inputs.sub_areas_m2,
        sub_diameters_m=sub_diameters,
        l_coords_rad=l_coords_rad,
        m_coords_rad=m_coords_rad,
        wavelength_indices=wavelength_indices,
    )
    return {
        "design_map": design_map.astype(np.float64, copy=True),
        "band_averaged_map": band_averaged_map.astype(np.float64, copy=True),
        # Display-ready (East-left) maps consistent with ``display_x_mas`` and
        # the plotted PDF. The raw ``design_map``/``band_averaged_map`` arrays
        # pair with ``l_coords_mas`` in native (RA-increasing) orientation; a
        # naive re-plot against ``display_x_mas`` must use these flipped maps.
        "design_map_display": design_map[:, ::-1].astype(np.float64, copy=True),
        "band_averaged_map_display": (
            band_averaged_map[:, ::-1].astype(np.float64, copy=True)
        ),
        "l_coords_rad": l_coords_rad.astype(np.float64, copy=True),
        "m_coords_rad": m_coords_rad.astype(np.float64, copy=True),
        "l_coords_mas": (l_coords_rad * MAS_PER_RAD).astype(np.float64, copy=True),
        "m_coords_mas": (m_coords_rad * MAS_PER_RAD).astype(np.float64, copy=True),
        "display_x_mas": (-l_coords_rad[::-1] * MAS_PER_RAD).astype(
            np.float64,
            copy=True,
        ),
        "wavelengths_m": inputs.wavelengths_m.astype(np.float64, copy=True),
        "design_wavelength_m": float(inputs.wavelengths_m[design_index]),
        "design_wavelength_index": int(design_index),
        "dark_indices": inputs.dark_indices.astype(np.intp, copy=True),
        "positions_pupil_m": inputs.positions_pupil_m.astype(np.float64, copy=True),
        "positions_projected_m": projection.positions_m.astype(
            np.float64,
            copy=True,
        ),
        "sub_diameters_m": inputs.sub_diameters_m.astype(np.float64, copy=True),
        "sub_areas_m2": inputs.sub_areas_m2.astype(np.float64, copy=True),
        "coupling_per_sub": inputs.coupling_per_sub.astype(np.float64, copy=True),
        "unitary_arm_weights": inputs.unitary_arm_weights.astype(
            np.float64,
            copy=True,
        ),
        "include_angular_coupling": np.asarray(bool(config.include_angular_coupling)),
        "projection_mode": np.asarray(projection.mode),
        "projection_hour_angle_rad": _optional_float_array(projection.hour_angle_rad),
        "projection_declination_rad": _optional_float_array(projection.declination_rad),
        "projection_latitude_rad": _optional_float_array(projection.latitude_rad),
    }


def _sky_projection_for_record(
    record: RunRecord,
    config: NullTransmissionMapOutputConfig,
    inputs: ChipFieldInputs,
) -> _SkyProjection:
    """Return positions in the map's instantaneous sky-response frame."""

    requested_mode = str(config.projection_mode)
    if requested_mode == "pupil":
        return _pupil_projection(inputs.positions_pupil_m)

    should_project = requested_mode == "sky" or _auto_should_project(record, config)
    if not should_project:
        return _pupil_projection(inputs.positions_pupil_m)

    latitude_rad = _projection_latitude_rad(record, config)
    hour_angle_rad, declination_rad = _projection_pointing_rad(
        record,
        config,
        latitude_rad=latitude_rad,
    )
    altitude_rad, azimuth_rad = _hour_angle_declination_to_altaz_rad(
        hour_angle_rad=hour_angle_rad,
        declination_rad=declination_rad,
        latitude_rad=latitude_rad,
    )
    # Project exactly as the stages do (review 2026-06-10 Issue 20): only the
    # per-mode telescope origins foreshorten off-zenith; intra-telescope pupil
    # offsets ride unprojected because the pupil tracks the target. The map
    # then displays the same chip response the optimization and sensitivity
    # stages computed.
    pupil_positions = np.asarray(inputs.positions_pupil_m, dtype=np.float64)
    positions = project_aperture_positions_for_pointing(
        pupil_positions,
        altitude_deg=float(np.degrees(altitude_rad)),
        azimuth_deg=float(np.degrees(azimuth_rad)),
        aperture_origins_m=mode_origin_positions_m(
            record.result.state.geometry,
            pupil_positions.shape[0],
        ),
    )
    return _SkyProjection(
        positions_m=positions,
        mode="sky",
        hour_angle_rad=hour_angle_rad,
        declination_rad=declination_rad,
        latitude_rad=latitude_rad,
    )


def _pupil_projection(positions_pupil_m: NDArray[np.float64]) -> _SkyProjection:
    return _SkyProjection(
        positions_m=np.asarray(positions_pupil_m, dtype=np.float64).copy(),
        mode="pupil",
        hour_angle_rad=None,
        declination_rad=None,
        latitude_rad=None,
    )


def _auto_should_project(
    record: RunRecord,
    config: NullTransmissionMapOutputConfig,
) -> bool:
    """Auto-project when the config describes a ground pointing."""

    if (
        config.declination_deg is not None
        or config.latitude_deg is not None
        or config.hour_angle_h is not None
    ):
        return True
    return (
        record.result.state.scene.altitude_deg is not None
        and record.result.state.scene.azimuth_deg is not None
        and _site_latitude_deg(record) is not None
    )


def _projection_latitude_rad(
    record: RunRecord,
    config: NullTransmissionMapOutputConfig,
) -> float:
    latitude_deg = (
        float(config.latitude_deg)
        if config.latitude_deg is not None
        else _site_latitude_deg(record)
    )
    if latitude_deg is None:
        latitude_deg = float(np.degrees(MAUNAKEA_LAT_RAD))
    return float(np.radians(latitude_deg))


def _site_latitude_deg(record: RunRecord) -> float | None:
    raw = getattr(record.config.site, "latitude_deg", None)
    if raw is None:
        return None
    return float(raw)


def _projection_pointing_rad(
    record: RunRecord,
    config: NullTransmissionMapOutputConfig,
    *,
    latitude_rad: float,
) -> tuple[float, float]:
    if config.declination_deg is not None:
        declination_rad = float(np.radians(float(config.declination_deg)))
        hour_angle_rad = _configured_hour_angle_rad(record, config)
        return hour_angle_rad, declination_rad

    scene = record.result.state.scene
    if scene.altitude_deg is not None and scene.azimuth_deg is not None:
        derived_hour_angle, declination_rad = _altaz_to_hour_angle_declination_rad(
            altitude_rad=float(np.radians(float(scene.altitude_deg))),
            azimuth_rad=float(np.radians(float(scene.azimuth_deg))),
            latitude_rad=latitude_rad,
        )
        hour_angle_rad = (
            _configured_hour_angle_rad(record, config)
            if config.hour_angle_h is not None
            else derived_hour_angle
        )
        return hour_angle_rad, declination_rad

    imaging_config = record.config.output.imaging_reconstruction
    declination_deg = (
        float(imaging_config.declination_deg)
        if imaging_config is not None
        else _DEFAULT_DECLINATION_DEG
    )
    return _configured_hour_angle_rad(record, config), float(np.radians(declination_deg))


def _configured_hour_angle_rad(
    record: RunRecord,
    config: NullTransmissionMapOutputConfig,
) -> float:
    hour_angle_h = (
        float(config.hour_angle_h)
        if config.hour_angle_h is not None
        else float(record.config.observation.hour_angle_h)
    )
    return float(np.radians(hour_angle_h * _DEG_PER_HOUR))


def _altaz_to_hour_angle_declination_rad(
    *,
    altitude_rad: float,
    azimuth_rad: float,
    latitude_rad: float,
) -> tuple[float, float]:
    """Convert local alt/az pointing to HA/Dec.

    Azimuth follows the stage convention: 0 deg is North and 90 deg is East.
    """

    sin_alt = np.sin(altitude_rad)
    cos_alt = np.cos(altitude_rad)
    sin_lat = np.sin(latitude_rad)
    cos_lat = np.cos(latitude_rad)
    sin_az = np.sin(azimuth_rad)
    cos_az = np.cos(azimuth_rad)

    sin_dec = sin_lat * sin_alt + cos_lat * cos_alt * cos_az
    declination_rad = float(np.arcsin(np.clip(sin_dec, -1.0, 1.0)))
    cos_dec = float(np.cos(declination_rad))
    if abs(cos_dec) < 1.0e-15 or abs(cos_lat) < 1.0e-15:
        return 0.0, declination_rad

    sin_hour_angle = -sin_az * cos_alt / cos_dec
    cos_hour_angle = (sin_alt - sin_lat * sin_dec) / (cos_lat * cos_dec)
    hour_angle_rad = float(
        np.arctan2(
            np.clip(sin_hour_angle, -1.0, 1.0),
            np.clip(cos_hour_angle, -1.0, 1.0),
        )
    )
    return hour_angle_rad, declination_rad


def _hour_angle_declination_to_altaz_rad(
    *,
    hour_angle_rad: float,
    declination_rad: float,
    latitude_rad: float,
) -> tuple[float, float]:
    """Convert HA/Dec pointing to local alt/az.

    Inverse of :func:`_altaz_to_hour_angle_declination_rad`; azimuth is 0 deg
    North, 90 deg East, matching the stage convention.
    """

    sin_dec = np.sin(declination_rad)
    cos_dec = np.cos(declination_rad)
    sin_lat = np.sin(latitude_rad)
    cos_lat = np.cos(latitude_rad)
    sin_ha = np.sin(hour_angle_rad)
    cos_ha = np.cos(hour_angle_rad)

    sin_alt = sin_lat * sin_dec + cos_lat * cos_dec * cos_ha
    altitude_rad = float(np.arcsin(np.clip(sin_alt, -1.0, 1.0)))
    cos_alt = float(np.cos(altitude_rad))
    if abs(cos_alt) < 1.0e-15 or abs(cos_lat) < 1.0e-15:
        return altitude_rad, 0.0

    sin_az = -sin_ha * cos_dec / cos_alt
    cos_az = (sin_dec - sin_lat * sin_alt) / (cos_lat * cos_alt)
    azimuth_rad = float(
        np.arctan2(
            np.clip(sin_az, -1.0, 1.0),
            np.clip(cos_az, -1.0, 1.0),
        )
    )
    return altitude_rad, azimuth_rad


def _optional_float_array(value: float | None) -> NDArray[np.float64]:
    if value is None:
        return np.asarray(np.nan, dtype=np.float64)
    return np.asarray(float(value), dtype=np.float64)


def _plot_maps(
    *,
    design_map: NDArray[np.float64],
    band_averaged_map: NDArray[np.float64],
    l_coords_mas: NDArray[np.float64],
    m_coords_mas: NDArray[np.float64],
    design_wavelength_um: float,
    config: NullTransmissionMapOutputConfig,
    record: RunRecord,
) -> plt.Figure:
    display_x_mas = -np.asarray(l_coords_mas, dtype=np.float64)[::-1]
    display_maps = (
        np.asarray(design_map, dtype=np.float64)[:, ::-1],
        np.asarray(band_averaged_map, dtype=np.float64)[:, ::-1],
    )
    norm, levels, line_levels, plot_maps = _display_scale(display_maps, config)
    x_limits, y_limits = _display_limits_mas(
        maps=display_maps,
        x_coords_mas=display_x_mas,
        y_coords_mas=m_coords_mas,
        config=config,
        planet_position_mas=(
            _planet_display_position_mas(record) if config.overlay_planet else None
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    titles = (
        f"Design wavelength ({design_wavelength_um:.3f} um)",
        "Band averaged",
    )
    mappable = None
    for ax, title, values in zip(axes, titles, plot_maps, strict=True):
        mappable = ax.contourf(
            display_x_mas,
            m_coords_mas,
            values,
            levels=levels,
            norm=norm,
            cmap="magma",
            extend="both",
        )
        _add_labeled_contours(
            ax=ax,
            x_coords_mas=display_x_mas,
            y_coords_mas=m_coords_mas,
            values=values,
            levels=line_levels,
            log_scale=bool(config.log_scale),
        )
        ax.set_title(title)
        ax.set_xlabel(r"$-\Delta\alpha$ (mas; East left)")
        ax.set_ylabel(r"$\Delta\delta$ (mas)")
        ax.axhline(0.0, color="white", linestyle=":", linewidth=0.5, alpha=0.45)
        ax.axvline(0.0, color="white", linestyle=":", linewidth=0.5, alpha=0.45)
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_aspect("equal", adjustable="box")
        if config.overlay_planet:
            _overlay_planet(ax, record)

    if mappable is not None:
        colorbar = fig.colorbar(mappable, ax=axes, shrink=0.86)
        colorbar.set_label("dark-port transmission fraction")
        if config.log_scale and line_levels.size:
            colorbar.set_ticks(line_levels)
            colorbar.set_ticklabels(
                [_format_contour_label(float(level), log_scale=True) for level in line_levels]
            )
    return fig


def _display_scale(
    maps: tuple[NDArray[np.float64], NDArray[np.float64]],
    config: NullTransmissionMapOutputConfig,
) -> tuple[
    Normalize,
    NDArray[np.float64],
    NDArray[np.float64],
    tuple[NDArray[np.float64], ...],
]:
    finite_max = max(float(np.nanmax(values)) for values in maps if values.size)
    if config.log_scale:
        floor = float(config.log_floor)
        vmax = max(finite_max, floor * 10.0)
        levels = _nice_log_levels(floor, vmax, int(config.n_contour_levels))
        line_levels = _labeled_log_levels(floor, vmax)
        norm = LogNorm(vmin=floor, vmax=vmax)
        plot_maps = tuple(
            np.maximum(np.asarray(values, dtype=np.float64), floor) for values in maps
        )
        return norm, levels, line_levels, plot_maps
    vmax = max(finite_max, 1.0e-15)
    levels = np.linspace(0.0, vmax, int(config.n_contour_levels))
    line_levels = np.linspace(0.25 * vmax, 0.75 * vmax, 3)
    return Normalize(vmin=0.0, vmax=vmax), levels, line_levels, maps


def _nice_log_levels(
    floor: float,
    vmax: float,
    max_levels: int,
) -> NDArray[np.float64]:
    """Return readable log-spaced filled-contour levels."""

    lo_exp = int(np.floor(np.log10(floor)))
    hi_exp = int(np.ceil(np.log10(vmax)))
    values: list[float] = []
    for exponent in range(lo_exp, hi_exp + 1):
        for mantissa in (1.0, 2.0, 5.0):
            value = mantissa * 10.0**exponent
            if floor <= value <= vmax:
                values.append(float(value))
    levels = np.asarray(sorted({floor, vmax, *values}), dtype=np.float64)
    if levels.size < 2:
        return np.asarray([floor, vmax], dtype=np.float64)
    if levels.size > max_levels:
        return np.geomspace(floor, vmax, max_levels)
    return levels


def _labeled_log_levels(
    floor: float,
    vmax: float,
) -> NDArray[np.float64]:
    """Return sparse, readable contour-label levels."""

    lo_exp = int(np.floor(np.log10(floor))) + 1
    hi_exp = int(np.floor(np.log10(vmax)))
    levels = [
        10.0**exponent
        for exponent in range(lo_exp, hi_exp + 1)
        if exponent % 3 == 0
    ]
    top_decade = 10.0 ** int(np.floor(np.log10(vmax)))
    if top_decade not in levels and floor < top_decade <= vmax:
        levels.append(top_decade)
    return np.asarray(sorted(set(levels)), dtype=np.float64)


def _display_limits_mas(
    *,
    maps: tuple[NDArray[np.float64], NDArray[np.float64]],
    x_coords_mas: NDArray[np.float64],
    y_coords_mas: NDArray[np.float64],
    config: NullTransmissionMapOutputConfig,
    planet_position_mas: tuple[float, float] | None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Choose plot limits from the useful non-floor transmission footprint."""

    x_coords = np.asarray(x_coords_mas, dtype=np.float64)
    y_coords = np.asarray(y_coords_mas, dtype=np.float64)
    full_x = max(abs(float(x_coords[0])), abs(float(x_coords[-1])))
    full_y = max(abs(float(y_coords[0])), abs(float(y_coords[-1])))
    if not config.auto_scale_to_owa:
        return (-full_x, full_x), (-full_y, full_y)

    threshold = (
        _log_crop_floor(maps, config)
        if config.log_scale
        else _linear_crop_floor(maps)
    )
    x_values: list[float] = []
    y_values: list[float] = []
    for values in maps:
        arr = np.asarray(values, dtype=np.float64)
        mask = np.isfinite(arr) & (arr >= threshold)
        rows, cols = np.where(mask)
        if cols.size == 0:
            continue
        x_values.extend(float(value) for value in x_coords[cols])
        y_values.extend(float(value) for value in y_coords[rows])

    if planet_position_mas is not None:
        planet_x_mas, planet_y_mas = planet_position_mas
        x_values.append(float(planet_x_mas))
        y_values.append(float(planet_y_mas))

    if not x_values or not y_values:
        return (-full_x, full_x), (-full_y, full_y)

    half_width = max(
        max(abs(value) for value in x_values),
        max(abs(value) for value in y_values),
    )
    padded = half_width * (1.0 + float(config.display_padding_fraction))
    padded = min(max(padded, _minimum_display_half_width(x_coords, y_coords)), full_x, full_y)
    return (-padded, padded), (-padded, padded)


def _linear_crop_floor(
    maps: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> float:
    finite_max = max(float(np.nanmax(values)) for values in maps if values.size)
    return max(finite_max * 1.0e-3, 0.0)


def _log_crop_floor(
    maps: tuple[NDArray[np.float64], NDArray[np.float64]],
    config: NullTransmissionMapOutputConfig,
) -> float:
    """Return the useful-footprint floor for log-scaled map auto-cropping."""

    finite_max = max(float(np.nanmax(values)) for values in maps if values.size)
    return max(float(config.log_floor), finite_max * 1.0e-3)


def _minimum_display_half_width(
    x_coords_mas: NDArray[np.float64],
    y_coords_mas: NDArray[np.float64],
) -> float:
    x_span = abs(float(x_coords_mas[-1]) - float(x_coords_mas[0]))
    y_span = abs(float(y_coords_mas[-1]) - float(y_coords_mas[0]))
    return max(x_span, y_span) * 0.05


def _add_labeled_contours(
    *,
    ax: plt.Axes,
    x_coords_mas: NDArray[np.float64],
    y_coords_mas: NDArray[np.float64],
    values: NDArray[np.float64],
    levels: NDArray[np.float64],
    log_scale: bool,
) -> None:
    """Draw sparse labeled contour lines on top of filled contours."""

    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return
    usable = [
        float(level)
        for level in np.asarray(levels, dtype=np.float64)
        if float(np.nanmin(finite)) < float(level) < float(np.nanmax(finite))
    ]
    if not usable:
        return
    contours = ax.contour(
        x_coords_mas,
        y_coords_mas,
        arr,
        levels=usable,
        colors="white",
        linewidths=0.55,
        alpha=0.78,
    )
    labels = ax.clabel(
        contours,
        inline=True,
        fontsize=7,
        fmt={level: _format_contour_label(level, log_scale=log_scale) for level in usable},
    )
    for label in labels:
        label.set_bbox(
            {
                "facecolor": "black",
                "edgecolor": "none",
                "alpha": 0.55,
                "pad": 0.8,
            }
        )
        label.set_color("white")


def _format_contour_label(
    value: float,
    *,
    log_scale: bool,
) -> str:
    if log_scale:
        exponent = int(round(np.log10(value)))
        if np.isclose(value, 10.0**exponent):
            return rf"$10^{{{exponent}}}$"
    if value >= 1.0e-2:
        return f"{value:.2g}"
    return f"{value:.0e}"


def _overlay_planet(ax: plt.Axes, record: RunRecord) -> None:
    position = _planet_display_position_mas(record)
    if position is None:
        return
    x_mas, y_mas = position
    ax.plot(
        x_mas,
        y_mas,
        marker="x",
        color="cyan",
        markersize=7.0,
        markeredgewidth=1.4,
    )


def _planet_display_position_mas(record: RunRecord) -> tuple[float, float] | None:
    """Return plot coordinates for the configured scene planet."""

    scene = record.result.state.scene
    if scene.planet_separation_mas is None or scene.planet_position_angle_deg is None:
        return None
    separation = float(scene.planet_separation_mas)
    pa_rad = np.radians(float(scene.planet_position_angle_deg))
    l_mas = separation * np.sin(pa_rad)
    m_mas = separation * np.cos(pa_rad)
    return -l_mas, m_mas
