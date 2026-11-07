"""Headline ``detection_space`` plot (SPEC §8.1, §8.2, §15.4).

The figure overlays detection-limit contrast curves per (instrument
config × J-mag × integration-time) cell on top of the known-exoplanet
population. Both detection-mode and characterization-mode 5σ curves
are drawn — detection-mode as solid lines, characterization-mode as
dashed lines of the matched colour, and ``performance.hdc`` CCF curves
as dash-dot lines in their own colour. The exoplanet scatter comes from
the declared ``[catalog.exoplanets]`` snapshot through the versioned
catalog parser, or else from a bundled NASA Exoplanet Archive snapshot
under KILO paperv3's Lambertian reflected-light assumptions, and is
filtered by observability from the resolved telescope site (default:
Mauna Kea, ``alt ≥ 45°``).
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from numpy.typing import NDArray

import matplotlib  # noqa: F401  # backend choice managed by styles module

from nullsim.outputs.styles import _ensure_headless_backend  # noqa: F401  # side-effect

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from nullsim.catalogs.exoplanets import ExoplanetCatalogConfig, load_catalog
from nullsim.config.catalog import parse_catalog_block
from nullsim.outputs import RunCollection, RunRecord
from nullsim.outputs.plots._kilo_planet_overlay import (
    GEOMETRIC_ALBEDO_DEFAULT,
    LABEL_PLANETS,
    LAT_KECK_DEG,
    METHOD_MARKERS,
    PlotPlanet,
    TYPE_COLORS,
    TYPE_DISPLAY_LABELS,
    load_planets,
    planets_from_catalog,
    trappist1_centroid,
)
from nullsim.outputs.styles import publication_style
from nullsim.physics.ideal_coronagraph import ideal_coronagraph_contrast_curve
from nullsim.physics.photometry import photon_rate_per_m2_per_bin
from nullsim.pipeline import register_plot


_DETECTION_COLOR = "C0"
_CHARACTERIZATION_COLOR = "#27AE60"
_IDEAL_CORONAGRAPH_COLOR = "0.35"
_HDC_COLOR = "#8E44AD"
_HDC_LEGEND_PREFIX = "HDC CCF"
_HDC_ARTIST_GID = "nullsim.detection_space.hdc_ccf"
_IDEAL_CORONAGRAPH_STYLES: dict[int, dict[str, Any]] = {
    2: {"linestyle": (0, (5.0, 2.0)), "linewidth": 1.4},
    4: {"linestyle": (0, (2.0, 1.2)), "linewidth": 1.4},
}
_DEFAULT_LINE_STYLE = {"linestyle": "-", "linewidth": 1.8}
_JMAG_LINE_STYLES: tuple[tuple[float, dict[str, Any]], ...] = (
    (4.0, {"linestyle": "-", "linewidth": 2.0}),
    (6.0, {"linestyle": "--", "linewidth": 1.8}),
    (8.0, {"linestyle": "-.", "linewidth": 1.8}),
    (10.0, {"linestyle": ":", "linewidth": 1.8}),
    (12.0, {"linestyle": (0, (1.0, 1.2)), "linewidth": 1.8}),
)
_LABEL_OFFSETS: dict[str, tuple[float, float]] = {
    "GJ 876 b": (8, -8),
    "GJ 876 c": (8, 8),
    "55 Cnc c": (8, 8),
    "GJ 3512 b": (8, -8),
    "tau Cet e": (8, -8),
    "51 Peg b": (8, 8),
}
_LABEL_DISPLAY: dict[str, str] = {"tau Cet e": r"$\tau$ Cet e"}
_MAX_PHYSICAL_CONTRAST = 1.0
NDArrayFloat = NDArray[np.float64]

_BUNDLED_PLANET_CATALOG = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "exoplanet_catalogs"
    / "nasa_exoplanet_archive_paperv3.csv"
)


@dataclass(frozen=True)
class _ObservabilityConfig:
    """Planet-overlay observability filter for the current run."""

    latitude_deg: float
    min_altitude_deg: float | None
    caption: str


@register_plot
def detection_space(
    runs: RunCollection,
    output_dir: Path,
    formats: Iterable[str],
) -> tuple[Path, ...]:
    """Write the headline detection-space figure (SPEC §8.1)."""

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    observability = _observability_config(runs)
    logging.info("detection_space: planet overlay %s.", observability.caption)
    planets = _load_planet_overlay(_catalog_config(runs), observability=observability)
    stellar_diameter_mas = _stellar_diameter_mas(runs)

    with publication_style():
        figure, axes = plt.subplots(figsize=(10.0, 7.0))
        _style_axes_atmosphere(axes)
        _plot_detection_curves(axes, runs)
        _plot_characterization_curves(axes, runs)
        _plot_hdc_curves(axes, runs)
        _plot_ideal_coronagraph_curves(axes, runs)
        _plot_known_exoplanets(axes, planets)
        _decorate_axes(axes, stellar_diameter_mas=stellar_diameter_mas)
        _draw_diffraction_limit(axes, runs)
        _attach_legends(axes, planets, runs)

        written: list[Path] = []
        for file_format in formats:
            output_path = plot_dir / f"detection_space.{file_format}"
            figure.savefig(output_path, bbox_inches="tight")
            written.append(output_path)
        plt.close(figure)
    return tuple(written)


def _draw_diffraction_limit(axes: plt.Axes, runs: RunCollection) -> None:
    """Vertical dashed line at λ/D, using the longest available baseline.

    The original KILO paper labels this reference as λ/D while using the
    longest available baseline for the Keck-only array geometry.
    """

    if not runs.records:
        return
    record = runs.records[0]
    grid = getattr(record.config, "grid", None)
    if grid is None:
        return
    wavelength_um = getattr(grid, "wavelength_center_um", None)
    if wavelength_um is None or wavelength_um <= 0.0:
        return

    geometry = record.result.state.geometry
    baseline_m = _max_baseline_m(geometry)
    if baseline_m is None or baseline_m <= 0.0:
        return

    lambda_over_d_rad = (wavelength_um * 1.0e-6) / baseline_m
    lambda_over_d_mas = lambda_over_d_rad * (180.0 / np.pi) * 3.6e6
    axes.axvline(
        lambda_over_d_mas,
        color="0.4",
        linestyle=":",
        linewidth=1.2,
        alpha=0.7,
    )
    axes.text(
        lambda_over_d_mas * 1.05,
        5.0e-11,
        r"$\lambda / D$",
        rotation=90,
        ha="left",
        va="bottom",
        fontsize=8,
        color="0.4",
    )


def _max_baseline_m(geometry: Any) -> float | None:
    """Longest pairwise sub-aperture / aperture separation in meters."""

    candidates = (
        getattr(geometry, "baselines_m", None),
        getattr(geometry, "aperture_positions_enu_m", None),
        getattr(geometry, "aperture_positions_pupil_m", None),
    )
    baselines = candidates[0]
    if baselines is not None:
        arr = np.asarray(baselines, dtype=float)
        if arr.size > 0 and np.isfinite(arr).any():
            return float(np.nanmax(np.abs(arr)))
    for positions in candidates[1:]:
        if positions is None:
            continue
        pts = np.asarray(positions, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue
        diffs = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt((diffs**2).sum(axis=-1))
        finite_mask = np.isfinite(dists)
        if not finite_mask.any():
            continue
        return float(dists[finite_mask].max())
    diameter = getattr(geometry, "diameter_m", None)
    if diameter is not None and float(diameter) > 0.0:
        return float(diameter)
    return None


# ----------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------


def _style_axes_atmosphere(axes: plt.Axes) -> None:
    """Match the KILO paper detection-space canvas."""

    axes.grid(True, which="both", alpha=0.3)


# ----------------------------------------------------------------------
# Detection-mode curves
# ----------------------------------------------------------------------


def _plot_detection_curves(axes: plt.Axes, runs: RunCollection) -> None:
    """One paper-style detection-mode contrast curve per sweep cell.

    Cells whose legend label duplicates an already-drawn one are skipped
    (with a logged notice): sweeps over parameters the label does not encode
    (e.g. scene.planet.contrast) otherwise stack visually identical lines
    and repeated legend entries. Only the first such cell is drawn, so
    per-cell Monte Carlo scatter (e.g. the HOSTS exozodi draw) is not
    averaged, just represented by one cell.
    """

    seen: set[str] = set()
    n_duplicates = 0
    for record in _paper_curve_records(runs):
        curve = record.result.state.results.contrast_curve
        if curve is None or curve.size == 0:
            continue
        key = _curve_legend_label(record, mode="det")
        if key in seen:
            n_duplicates += 1
            continue
        seen.add(key)
        separations_mas = np.asarray(curve["separation_mas"], dtype=np.float64)
        contrasts = np.asarray(curve["contrast"], dtype=np.float64)
        order = np.argsort(separations_mas)
        separations_mas = separations_mas[order]
        contrasts = contrasts[order]
        mask = (
            np.isfinite(separations_mas)
            & np.isfinite(contrasts)
            & (separations_mas > 0.0)
            & (contrasts > 0.0)
            & (contrasts <= _MAX_PHYSICAL_CONTRAST)
        )
        if not mask.all():
            warnings.warn(
                f"detection_space: dropped {int((~mask).sum())} non-positive, "
                f"non-finite, or unphysical point(s) from contrast curve for "
                f"{_record_label(record)!r}",
                stacklevel=2,
            )
        separations_mas = separations_mas[mask]
        contrasts = contrasts[mask]
        if separations_mas.size == 0:
            continue
        style = _line_style_for_record(record)
        axes.plot(
            separations_mas,
            contrasts,
            color=_DETECTION_COLOR,
            **style,
            zorder=4,
        )
        if _is_primary_fill_record(record):
            upper_bound = np.full_like(contrasts, fill_value=1.0)
            axes.fill_between(
                separations_mas,
                contrasts,
                upper_bound,
                color=_DETECTION_COLOR,
                alpha=0.15,
                linewidth=0,
                zorder=1,
            )
    if n_duplicates:
        logging.info(
            "detection_space: skipped %d sweep cell(s) whose legend label "
            "duplicates an already-plotted cell.",
            n_duplicates,
        )


# ----------------------------------------------------------------------
# Guyon 2006 theoretically optimal coronagraph bound
# ----------------------------------------------------------------------


def _ideal_coronagraph_orders(runs: RunCollection) -> tuple[int, ...]:
    """Configured [output.detection_space].ideal_coronagraph_orders or ()."""

    output = getattr(runs.config, "output", None)
    config = getattr(output, "detection_space", None)
    if config is None:
        return ()
    return tuple(int(order) for order in config.ideal_coronagraph_orders)


def _plot_ideal_coronagraph_curves(axes: plt.Axes, runs: RunCollection) -> None:
    """Overlay the Guyon 2006 optimal-coronagraph photon bound per order.

    The bound shares the headline record's star, aperture diameter, zodi
    background, and integration time, but routes the starlight through
    perfect optics: the full unobscured primary, the mirror train, and the
    detector QE, with none of the photonic chain's fiber/chip/polarization
    losses. It is the theoretical limit instrument of Guyon et al. (2006,
    ApJS 167, 81), not a model of any real coronagraph.
    """

    orders = _ideal_coronagraph_orders(runs)
    if not orders:
        return
    inputs = _ideal_coronagraph_curve_inputs(runs)
    if inputs is None:
        return
    separations_mas = np.geomspace(0.5, 1100.0, 300)
    for order in orders:
        curve = ideal_coronagraph_contrast_curve(
            separations_mas,
            wavelengths_m=inputs["wavelengths_m"],
            diameter_m=inputs["diameter_m"],
            stellar_radius_mas=inputs["stellar_radius_mas"],
            stellar_rate_hz=inputs["stellar_rate_hz"],
            background_rate_hz=inputs["background_rate_hz"],
            integration_time_s=inputs["integration_time_s"],
            snr_target=inputs["snr_target"],
            order=order,
        )
        mask = np.isfinite(curve) & (curve > 0.0) & (curve <= _MAX_PHYSICAL_CONTRAST)
        if not mask.any():
            continue
        style = _IDEAL_CORONAGRAPH_STYLES.get(
            order, {"linestyle": "--", "linewidth": 1.4}
        )
        axes.plot(
            separations_mas[mask],
            curve[mask],
            color=_IDEAL_CORONAGRAPH_COLOR,
            zorder=3,
            **style,
        )


def _ideal_coronagraph_curve_inputs(runs: RunCollection) -> dict[str, Any] | None:
    """Assemble the bound's inputs from the first headline record.

    Returns None (with a warning) when the run is not a single-aperture
    config or required scene/detector context is missing.
    """

    records = _paper_curve_records(runs)
    if not records:
        return None
    record = records[0]
    state = record.result.state

    telescope = getattr(record.config, "telescope", None)
    aperture = _attr_or_key(telescope, "aperture")
    diameter_m = _attr_or_key(aperture, "diameter_m")
    n_apertures = _attr_or_key(telescope, "n_apertures")
    if diameter_m is None or diameter_m <= 0.0:
        warnings.warn(
            "detection_space: ideal-coronagraph overlay needs "
            "telescope.aperture.diameter_m; skipping.",
            stacklevel=2,
        )
        return None
    if n_apertures not in (None, 1):
        warnings.warn(
            "detection_space: ideal-coronagraph overlay is defined for a "
            f"single clear primary (telescope.n_apertures = {n_apertures}); "
            "skipping.",
            stacklevel=2,
        )
        return None

    wavelength_grid_um = state.throughput.wavelength_grid_um or ()
    if not wavelength_grid_um:
        warnings.warn(
            "detection_space: ideal-coronagraph overlay needs a throughput "
            "wavelength grid; skipping.",
            stacklevel=2,
        )
        return None
    wavelengths_um = np.asarray(wavelength_grid_um, dtype=np.float64)

    detector_payload = _newest_component(state, "detector.")
    if detector_payload is None or "star_magnitude" not in detector_payload:
        warnings.warn(
            "detection_space: ideal-coronagraph overlay needs a detector "
            "component with star_magnitude/photometric_band; skipping.",
            stacklevel=2,
        )
        return None
    star_magnitude = float(detector_payload["star_magnitude"])
    band = str(detector_payload.get("photometric_band", "J"))
    detector_qe = float(detector_payload.get("detector_qe", 1.0))

    obscuration = float(_attr_or_key(aperture, "obscuration_fraction") or 0.0)
    primary_area_m2 = (
        math.pi * (float(diameter_m) / 2.0) ** 2 * (1.0 - obscuration**2)
    )

    mirror_fraction = _mirror_transmission_per_bin(state, wavelengths_um.size)
    bandwidths_um = _bin_widths_um(wavelengths_um)
    rate_per_bin = np.array(
        [
            photon_rate_per_m2_per_bin(
                star_magnitude, float(wl), float(bw), band=band
            )
            for wl, bw in zip(wavelengths_um, bandwidths_um)
        ],
        dtype=np.float64,
    )
    stellar_rate_hz = float(
        np.sum(rate_per_bin * mirror_fraction) * primary_area_m2 * detector_qe
    )
    if stellar_rate_hz <= 0.0:
        warnings.warn(
            "detection_space: ideal-coronagraph stellar rate resolved to "
            f"{stellar_rate_hz}; skipping.",
            stacklevel=2,
        )
        return None

    stellar_diameter_mas = state.scene.star_angular_diameter_mas
    stellar_radius_mas = (
        float(stellar_diameter_mas) / 2.0 if stellar_diameter_mas else 0.0
    )

    background_rate_hz = _ideal_coronagraph_background_rate_hz(
        _newest_component(state, "background.space_zodi"),
        wavelength_m=float(record.config.grid.wavelength_center_um) * 1.0e-6,
        diameter_m=float(diameter_m),
        primary_area_m2=primary_area_m2,
        optics_fraction=float(np.mean(mirror_fraction)) * detector_qe,
    )

    integration_time_s = _contrast_reference_time_s(record)
    if integration_time_s is None:
        observation = getattr(record.config, "observation", None)
        integration_time_s = getattr(observation, "integration_time_s", None)
    if integration_time_s is None or integration_time_s <= 0.0:
        warnings.warn(
            "detection_space: ideal-coronagraph overlay needs a positive "
            "integration time; skipping.",
            stacklevel=2,
        )
        return None

    performance_payload = _newest_component(state, "performance.")
    snr_target = 5.0
    if performance_payload is not None:
        snr_target = float(performance_payload.get("target_snr", 5.0))

    return {
        "wavelengths_m": wavelengths_um * 1.0e-6,
        "diameter_m": float(diameter_m),
        "stellar_radius_mas": stellar_radius_mas,
        "stellar_rate_hz": stellar_rate_hz,
        "background_rate_hz": background_rate_hz,
        "integration_time_s": float(integration_time_s),
        "snr_target": snr_target,
    }


_ARCSEC_PER_RAD = 180.0 / math.pi * 3600.0


def _ideal_coronagraph_background_rate_hz(
    zodi_payload: dict[str, Any] | None,
    *,
    wavelength_m: float,
    diameter_m: float,
    primary_area_m2: float,
    optics_fraction: float,
) -> float:
    """Zodi through a perfect coronagraph on the full primary.

    Local zodi plus exozodi surface brightness at the scene separation
    (photons/s/m^2/arcsec^2 at the telescope entrance) over one
    ``(lambda/D)^2`` resolution element of the primary, times the mirror
    train and detector QE. That is the background the Guyon et al. (2006)
    ideal instrument sees in its planet aperture; the nuller's routed
    per-port zodi is not.
    """

    if zodi_payload is None:
        return 0.0
    local = zodi_payload.get("local_zodi_rate_hz_per_m2_per_arcsec2")
    exozodi = zodi_payload.get(
        "exozodi_rate_hz_per_m2_per_arcsec2_at_scene_separation"
    )
    if local is None or exozodi is None:
        warnings.warn(
            "detection_space: background.space_zodi publishes no zodi surface "
            "brightness; the ideal-coronagraph background falls back to the "
            "nuller's routed dark-port zodi.",
            stacklevel=2,
        )
        per_port = float(zodi_payload.get("zodi_rate_hz_per_dark_port", 0.0))
        n_ports = int(zodi_payload.get("n_physical_dark_ports", 0))
        return per_port * max(n_ports, 0)
    resolution_element_arcsec2 = (wavelength_m / diameter_m * _ARCSEC_PER_RAD) ** 2
    return (
        (float(local) + float(exozodi))
        * primary_area_m2
        * resolution_element_arcsec2
        * optics_fraction
    )


def _attr_or_key(container: Any, name: str) -> Any:
    """Read a config field from a model attribute or a plain-dict key."""

    if container is None:
        return None
    if isinstance(container, dict):
        return container.get(name)
    return getattr(container, name, None)


def _newest_component(state: Any, prefix: str) -> dict[str, Any] | None:
    """Return the newest results component whose name starts with prefix."""

    for name, payload in reversed(tuple(state.results.components.items())):
        if name.startswith(prefix) and isinstance(payload, dict):
            return payload
    return None


def _mirror_transmission_per_bin(state: Any, n_bins: int) -> NDArrayFloat:
    """Per-bin mirror-train transmission, or ones when not recorded."""

    payload = state.throughput.components.get("telescope.mirror_transmission")
    if isinstance(payload, dict):
        fraction = np.asarray(
            payload.get("transmission_fraction", ()), dtype=np.float64
        )
        if fraction.size == n_bins:
            return fraction
    return np.ones(n_bins, dtype=np.float64)


def _bin_widths_um(wavelengths_um: NDArrayFloat) -> NDArrayFloat:
    """Midpoint-edge bin widths matching the detector's rectangular bins."""

    if wavelengths_um.size == 1:
        return np.array([float(wavelengths_um[0])], dtype=np.float64)
    edges = np.empty(wavelengths_um.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (wavelengths_um[:-1] + wavelengths_um[1:])
    edges[0] = wavelengths_um[0] - 0.5 * (wavelengths_um[1] - wavelengths_um[0])
    edges[-1] = wavelengths_um[-1] + 0.5 * (wavelengths_um[-1] - wavelengths_um[-2])
    return np.diff(edges)


# ----------------------------------------------------------------------
# Characterization-mode curves
# ----------------------------------------------------------------------


def _plot_characterization_curves(axes: plt.Axes, runs: RunCollection) -> None:
    """Overlay char-mode 5σ contrast curves as dashed lines matched in colour."""

    seen: set[str] = set()
    for record, color, key, payload in _characterization_curve_entries(runs):
        separations_mas = np.asarray(payload["separations_mas"], dtype=np.float64)
        contrasts = np.asarray(payload["contrast_5sigma"], dtype=np.float64)
        if key in seen:
            continue
        seen.add(key)
        mask = (
            np.isfinite(separations_mas)
            & np.isfinite(contrasts)
            & (separations_mas > 0.0)
            & (contrasts > 0.0)
            & (contrasts <= _MAX_PHYSICAL_CONTRAST)
        )
        success = payload.get("optimization_success_per_anchor")
        if success is not None:
            success_mask = np.asarray(success, dtype=bool)
            if success_mask.shape == contrasts.shape:
                mask &= success_mask
        if bool(payload.get("enforce_null_constraint", False)):
            null_floor = float(payload.get("null_floor", 0.0))
            feasibility_tol = 0.01 * null_floor if null_floor > 0.0 else 0.0
            # Gate only on the design-grid margin the optimizer actually solved
            # against. The full-band static-null margin was a stricter,
            # different-grid constraint the fixed-OPD mesh never targeted (it
            # nulls to null_floor on the broadband design grid, while the
            # full-band check evaluates every detector bin), so it dropped most
            # feasible anchors and collapsed the plotted curve to a stub.
            margins = payload.get("optimization_constraint_margin_per_anchor")
            if margins is not None:
                margin_mask = np.asarray(margins, dtype=np.float64)
                if margin_mask.shape == contrasts.shape:
                    mask &= margin_mask >= -feasibility_tol
        dropped = int(contrasts.size - np.count_nonzero(mask))
        if dropped:
            warnings.warn(
                "detection_space: dropped "
                f"{dropped} failed or unphysical characterization anchor(s).",
                stacklevel=2,
            )
        segments = _valid_curve_segments(separations_mas, contrasts, mask)
        if not segments:
            continue
        style = _line_style_for_record(record)
        for separations_segment, contrast_segment in segments:
            if separations_segment.size >= 2:
                axes.plot(
                    separations_segment,
                    contrast_segment,
                    color=color,
                    **style,
                    zorder=5,
                )
                if _is_primary_fill_record(record):
                    axes.fill_between(
                        separations_segment,
                        contrast_segment,
                        np.full_like(contrast_segment, fill_value=1.0),
                        color=color,
                        alpha=0.10,
                        linewidth=0,
                        zorder=2,
                    )
            else:
                axes.scatter(
                    separations_segment,
                    contrast_segment,
                    color=color,
                    s=18.0,
                    zorder=5,
                )


def _valid_curve_segments(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return sorted contiguous curve segments without bridging invalid points."""

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask_arr = np.asarray(mask, dtype=bool)
    if x_arr.shape != y_arr.shape or x_arr.shape != mask_arr.shape:
        raise ValueError(
            "x, y, and mask must have matching shapes for curve segmentation."
        )
    if x_arr.size == 0:
        return ()

    order = np.argsort(x_arr)
    x_arr = x_arr[order]
    y_arr = y_arr[order]
    mask_arr = mask_arr[order]

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    start: int | None = None
    for index, is_valid in enumerate(mask_arr):
        if is_valid and start is None:
            start = index
        elif not is_valid and start is not None:
            segments.append((x_arr[start:index].copy(), y_arr[start:index].copy()))
            start = None
    if start is not None:
        segments.append((x_arr[start:].copy(), y_arr[start:].copy()))
    return tuple(segments)


_CHARACTERIZATION_PREFIX = "characterization.detection_curve"
_EXTRA_CHARACTERIZATION_COLORS = ("#8E44AD", "#D35400", "#16A085", "#7F8C8D")


def _characterization_curve_entries(
    runs: RunCollection,
) -> tuple[tuple[RunRecord, str, str, dict], ...]:
    """Every characterization curve of the headline records, each labeled.

    Yields ``(record, color, legend_label, payload)``. A record with one
    curve keeps the house green and the plain label; with several, each
    component gets its own color and its component name in the label, so no
    curve is dropped or chosen by insertion order.
    """

    entries = []
    for record in _paper_curve_records(runs):
        curves = [
            (name, payload)
            for name, payload in record.result.state.results.components.items()
            if name.startswith(_CHARACTERIZATION_PREFIX) and isinstance(payload, dict)
        ]
        label = _curve_legend_label(record, mode="char")
        for index, (name, payload) in enumerate(curves):
            if len(curves) == 1:
                entries.append((record, _CHARACTERIZATION_COLOR, label, payload))
                continue
            colors = (_CHARACTERIZATION_COLOR, *_EXTRA_CHARACTERIZATION_COLORS)
            entries.append(
                (record, colors[index % len(colors)], f"{label} [{name}]", payload)
            )
    return tuple(entries)


# ----------------------------------------------------------------------
# HDC cross-correlation curves
# ----------------------------------------------------------------------


def _plot_hdc_curves(axes: plt.Axes, runs: RunCollection) -> None:
    """Overlay every ``performance.hdc`` CCF curve as dash-dot lines.

    Dash-dot rather than the characterization overlay's dashed styling, and its
    own colour, so the three curve families on the figure stay distinguishable.

    Every artist drawn here carries its legend label, and `_attach_legends`
    reads the labels back off the axes. That is the whole ghost-entry defence:
    a payload that yields no curve (all anchors non-detections, malformed
    arrays, no resolving power) leaves no artist and therefore no legend entry,
    and the single identity that dedupes drawing -- the per-record label with
    the resolving power appended -- is by construction the one that dedupes the
    legend.

    No convergence gating here, unlike the ``hdc_ccf`` diagnostic plot: that one
    gates on upstream spectro convergence because it renders a retune that may
    not have converged. A ``performance.hdc`` payload only exists when its
    upstream resolved, and ``performance.standard`` publishes no convergence
    predicate to gate on, so there is nothing to check.
    """

    seen: set[str] = set()
    for record in _paper_curve_records(runs):
        # Every performance.hdc component, not just the first: a config can run
        # the stage more than once (two dark-port resolving powers, say), and
        # rendering only the first silently drops the rest.
        for name, payload in _hdc_payloads(record):
            _draw_one_hdc_curve(axes, record, name, payload, seen)


def _draw_one_hdc_curve(
    axes: plt.Axes,
    record: RunRecord,
    name: str,
    payload: dict,
    seen: set[str],
) -> None:
    """Draw one ``performance.hdc`` payload's CCF curve, if it has one."""

    separations_mas = _finite_1d(payload.get("separations_mas"))
    contrasts = _finite_1d(_hdc_contrast_values(payload))
    if (
        separations_mas is None
        or contrasts is None
        or contrasts.shape != separations_mas.shape
    ):
        logging.info(
            "detection_space: performance.hdc payload for %r carries no "
            "usable separations_mas/ccf_contrast_at_target pair; skipping "
            "the HDC overlay for that cell.",
            _record_label(record),
        )
        return
    # The R-bearing label is the binding contract for this overlay, so a
    # payload that cannot produce one is missing a required key, not a
    # curve to draw with a vaguer label.
    label = _hdc_legend_label(record, payload)
    if label is None:
        logging.info(
            "detection_space: performance.hdc payload for %r has no finite "
            "resolving_power_used; skipping the HDC overlay for that cell.",
            _record_label(record),
        )
        return
    # Dedupe DRAWING by label AND component name, not by label alone. Two
    # components on one record can share a label legitimately -- same J, time,
    # R, and threshold but different spectral templates -- and keying the draw
    # on the label alone would silently drop one of them; component names are
    # unique by construction (unique_component_name), so adding the name
    # separates them. The label still has to be in the key, because two
    # different records can carry the same component name and differ only in
    # what the label says (an R sweep at fixed J). Identical labels collapse
    # to a single legend entry regardless: the legend is harvested off the
    # drawn artists and deduped there, so both curves render under one entry.
    identity = f"{label}::{name}"
    if identity in seen:
        return
    seen.add(identity)
    # NaN anchors are non-detections by design (the CCF never reaches the
    # target SNR there), not failures — drop them and say how many.
    mask = (
        np.isfinite(separations_mas)
        & np.isfinite(contrasts)
        & (separations_mas > 0.0)
        & (contrasts > 0.0)
        & (contrasts <= _MAX_PHYSICAL_CONTRAST)
    )
    dropped = int(contrasts.size - np.count_nonzero(mask))
    if dropped:
        warnings.warn(
            "detection_space: dropped "
            f"{dropped} non-detection or unphysical HDC CCF anchor(s).",
            stacklevel=2,
        )
    segments = _valid_curve_segments(separations_mas, contrasts, mask)
    if not segments:
        return
    style = {**_line_style_for_record(record), "linestyle": "-."}
    # Label the first drawn artist only: the legend wants one entry per
    # curve, not one per segment.
    artist_label: str | None = label
    for separations_segment, contrast_segment in segments:
        if separations_segment.size >= 2:
            artists = axes.plot(
                separations_segment,
                contrast_segment,
                color=_HDC_COLOR,
                **style,
                zorder=5,
                label=artist_label,
            )
        else:
            artists = [
                axes.scatter(
                    separations_segment,
                    contrast_segment,
                    color=_HDC_COLOR,
                    s=18.0,
                    zorder=5,
                    label=artist_label,
                )
            ]
        # Tag ownership on the artist itself. The legend harvest keys on
        # this, not on the label text: a curve from some other family whose
        # label merely mentions HDC must not be picked up.
        for artist in artists:
            artist.set_gid(_HDC_ARTIST_GID)
        artist_label = None


def _drawn_hdc_legend_entries(axes: plt.Axes) -> tuple[tuple[Line2D, str], ...]:
    """Legend handles for the HDC curves actually drawn on ``axes``.

    Read back off the artists rather than recomputed from the payloads, so a
    payload that produced no curve cannot contribute a legend entry.
    """

    entries: list[tuple[Line2D, str]] = []
    seen: set[str] = set()
    for artist in (*axes.lines, *axes.collections):
        if artist.get_gid() != _HDC_ARTIST_GID:
            continue
        label = artist.get_label()
        # Continuation segments of a drawn curve carry matplotlib's underscore
        # placeholder: one entry per curve, not per segment.
        if not isinstance(label, str) or label.startswith("_"):
            continue
        if label in seen:
            continue
        seen.add(label)
        if isinstance(artist, Line2D):
            handle = Line2D(
                [],
                [],
                color=artist.get_color(),
                linestyle=artist.get_linestyle(),
                linewidth=artist.get_linewidth(),
            )
        else:
            handle = Line2D(
                [],
                [],
                color=_HDC_COLOR,
                linestyle="",
                marker="o",
                markersize=4.0,
            )
        entries.append((handle, label))
    return tuple(entries)


def _hdc_payloads(record: RunRecord) -> tuple[tuple[str, dict], ...]:
    """Every ``(name, payload)`` ``performance.hdc`` pair, in pipeline order."""

    payloads = tuple(
        (name, payload)
        for name, payload in record.result.state.results.components.items()
        if name.startswith("performance.hdc") and isinstance(payload, dict)
    )
    if not payloads:
        logging.debug(
            "detection_space: no performance.hdc component for %r; no HDC "
            "overlay.",
            _record_label(record),
        )
    return payloads


def _hdc_contrast_values(payload: dict) -> Any:
    """The CCF contrast curve, threshold-generic key first.

    ``ccf_contrast_5sigma`` is only an alias, and the stage publishes it only
    when the threshold really is 5 sigma, so it is the fallback rather than the
    contract.
    """

    values = payload.get("ccf_contrast_at_target")
    if values is None:
        values = payload.get("ccf_contrast_5sigma")
    return values


def _hdc_threshold_sigma(payload: dict) -> float | None:
    try:
        threshold = float(payload.get("comparison_target_snr"))
    except (TypeError, ValueError):
        return None
    return threshold if math.isfinite(threshold) and threshold > 0.0 else None


def _finite_1d(values: Any) -> NDArrayFloat | None:
    if values is None:
        return None
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.ndim != 1 or array.size == 0:
        return None
    return array


def _hdc_legend_label(record: RunRecord, payload: dict) -> str | None:
    """``"<instrument> HDC CCF (J=4, 4h, R=3000)"``, or None without a finite R.

    Everything inside the parentheses except the record identity comes from
    THIS payload, not from the record's first ``performance.standard``: the
    exposure is the HDC stage's own ``comparison_time_s`` (the basis both its
    curves were computed at), and the threshold is its
    ``comparison_target_snr``. Reading the time off ``performance.standard``
    would mislabel any cell where the two disagree, and two HDC components on
    one record would otherwise be labelled identically.

    The threshold is named only when it is not 5 sigma, so the common case
    stays uncluttered and an unusual one cannot pass for the default.
    """

    try:
        resolving_power = float(payload.get("resolving_power_used"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(resolving_power):
        return None
    extra = f"R={resolving_power:g}"
    threshold = _hdc_threshold_sigma(payload)
    if threshold is not None and not math.isclose(
        threshold, 5.0, rel_tol=1.0e-9, abs_tol=0.0
    ):
        extra = f"{extra}, {threshold:g}\u03c3"
    return _curve_legend_label(
        record,
        mode="hdc",
        extra=extra,
        time_s=_hdc_comparison_time_s(payload),
    )


def _hdc_comparison_time_s(payload: dict) -> float | None:
    try:
        value = float(payload.get("comparison_time_s"))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0.0 else None


# ----------------------------------------------------------------------
# Planet overlay (KILO port)
# ----------------------------------------------------------------------


def _plot_known_exoplanets(axes: plt.Axes, planets: Sequence[PlotPlanet]) -> None:
    """KILO-style scatter: colour by type, marker by discovery method."""

    if not planets:
        return
    for planet in planets:
        color = TYPE_COLORS.get(planet.planet_type, "#7f7f7f")
        marker = METHOD_MARKERS.get(planet.discovery_method, "D")
        axes.scatter(
            [planet.separation_mas],
            [planet.contrast],
            c=[color],
            marker=marker,
            s=30.0,
            alpha=0.6,
            edgecolors="none",
            zorder=7,
        )
        if planet.name in LABEL_PLANETS:
            offset = _LABEL_OFFSETS.get(planet.name, (6, 6))
            axes.annotate(
                _LABEL_DISPLAY.get(planet.name, planet.name),
                xy=(planet.separation_mas, planet.contrast),
                xytext=offset,
                textcoords="offset points",
                fontsize=7,
                color="0.3",
                zorder=8,
            )

    centroid = trappist1_centroid(planets)
    if centroid is not None:
        sep, con = centroid
        axes.annotate(
            "TRAPPIST-1",
            xy=(sep, con),
            xytext=(12, -10),
            textcoords="offset points",
            fontsize=7,
            color="0.3",
            zorder=8,
        )


def _catalog_config(runs: RunCollection) -> ExoplanetCatalogConfig | None:
    """The validated ``[catalog.exoplanets]`` block, or None without one.

    Reads the block from the first record's raw_config and resolves a
    relative ``snapshot_path`` against the source config directory, as the
    loader does when it folds the catalog into ``run_hash``.
    """

    try:
        record = runs.records[0]
    except (IndexError, AttributeError):
        return None
    raw = getattr(record.config, "raw_config", None) or {}
    catalog_block = (
        raw.get("catalog", {}).get("exoplanets") if isinstance(raw, dict) else None
    )
    if not isinstance(catalog_block, dict):
        return None
    source_path = getattr(record.config, "source_path", None)
    return parse_catalog_block(
        catalog_block,
        config_dir=source_path.parent if source_path is not None else None,
    )


def _load_planet_overlay(
    catalog: ExoplanetCatalogConfig | None = None,
    *,
    observability: _ObservabilityConfig | None = None,
) -> tuple[PlotPlanet, ...]:
    """Overlay planets from the declared catalog, else the bundled KILO set.

    A declared ``[catalog.exoplanets]`` snapshot goes through the versioned
    ``load_catalog`` parser (rule version, separation and contrast columns,
    aliases) and must load. Without one, the bundled paperv3 snapshot keeps
    KILO's radius bins and contrast model.
    """

    obs = observability or _ObservabilityConfig(
        latitude_deg=LAT_KECK_DEG,
        min_altitude_deg=45.0,
        caption=r"observable above $45^\circ$ altitude from Mauna Kea",
    )
    if catalog is not None:
        return planets_from_catalog(
            load_catalog(catalog),
            rule_version=catalog.classification_rule_version,
            latitude_deg=obs.latitude_deg,
            min_altitude_deg=obs.min_altitude_deg,
            geometric_albedo=GEOMETRIC_ALBEDO_DEFAULT,
        )
    if not _BUNDLED_PLANET_CATALOG.exists():
        return ()
    try:
        return load_planets(
            _BUNDLED_PLANET_CATALOG,
            latitude_deg=obs.latitude_deg,
            min_altitude_deg=obs.min_altitude_deg,
            geometric_albedo=GEOMETRIC_ALBEDO_DEFAULT,
        )
    except Exception as exc:  # pragma: no cover - defensive against bad CSVs
        warnings.warn(
            f"detection_space: planet overlay disabled ({exc})",
            stacklevel=2,
        )
        return ()


def _observability_config(runs: RunCollection) -> _ObservabilityConfig:
    """Resolve the known-planet observability cut from ``[site]``.

    Latitude and site name come from the resolved (preset-merged) site, so
    a ``cerro_armazones_site`` config cuts at Armazones. Explicit
    ``[site.observability]`` ``latitude_deg``/``site_label`` override them,
    and a site with no latitude falls back to the KILO paperv3 Mauna Kea
    ``alt >= 45 deg`` cut. Space configs can opt out with:

        [site.observability]
        all_sky = true
    """

    site = _site_mapping(runs)
    observability = site.get("observability")
    if not isinstance(observability, dict):
        observability = {}
    if bool(observability.get("all_sky", False)):
        return _ObservabilityConfig(
            latitude_deg=LAT_KECK_DEG,
            min_altitude_deg=None,
            caption="all-sky space observability",
        )

    site_latitude = site.get("latitude_deg")
    latitude_deg = _float_or_default(
        observability.get("latitude_deg", site_latitude),
        LAT_KECK_DEG,
    )
    min_altitude_deg = _float_or_default(
        observability.get("min_altitude_deg"),
        45.0,
    )
    default_label = site.get("name") if site_latitude is not None else None
    site_label = str(observability.get("site_label") or default_label or "Mauna Kea")
    caption = (
        rf"observable above ${min_altitude_deg:g}^\circ$ altitude "
        f"from {site_label}"
    )
    return _ObservabilityConfig(
        latitude_deg=latitude_deg,
        min_altitude_deg=min_altitude_deg,
        caption=caption,
    )


def _site_mapping(runs: RunCollection) -> dict[str, Any]:
    """The first record's resolved ``[site]``: preset fields plus overrides.

    The raw TOML table holds only the preset name in the shipped configs, so
    it cannot supply the latitude.
    """

    try:
        record = runs.records[0]
    except (IndexError, AttributeError):
        return {}
    site_obj = getattr(record.config, "site", None)
    if hasattr(site_obj, "model_dump"):
        return site_obj.model_dump(mode="python")
    return {}


def _float_or_default(value: object, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


# ----------------------------------------------------------------------
# Axes decoration + legends
# ----------------------------------------------------------------------


def _decorate_axes(
    axes: plt.Axes,
    *,
    stellar_diameter_mas: float | None,
) -> None:
    axes.set_xscale("log")
    axes.set_yscale("log")
    x_min = 0.2
    if stellar_diameter_mas is not None and stellar_diameter_mas > 0.0:
        x_min = max(0.05, float(stellar_diameter_mas) / 2.0)
    axes.set_xlim(x_min, 300.0)
    axes.set_ylim(1.0e-10, 1.0e-1)
    axes.set_xlabel("Angular separation (mas)")
    axes.set_ylabel("Planet/star contrast")


def _attach_legends(
    axes: plt.Axes,
    planets: Sequence[PlotPlanet],
    runs: RunCollection,
) -> None:
    """Single KILO-paper-style legend for curves and planet vocabulary."""

    handles: list[Line2D] = []
    labels: list[str] = []
    seen_det: set[str] = set()
    for record in _paper_curve_records(runs):
        curve = record.result.state.results.contrast_curve
        if curve is None:
            continue
        # Mirror the label dedupe in _plot_detection_curves: one legend
        # entry per drawn line.
        key = _curve_legend_label(record, mode="det")
        if key in seen_det:
            continue
        seen_det.add(key)
        handles.append(
            Line2D([], [], color=_DETECTION_COLOR, **_line_style_for_record(record))
        )
        labels.append(_curve_legend_label(record, mode="det"))

    seen_char: set[str] = set()
    for record, color, key, _payload in _characterization_curve_entries(runs):
        if key in seen_char:
            continue
        seen_char.add(key)
        handles.append(Line2D([], [], color=color, **_line_style_for_record(record)))
        labels.append(key)

    # Read the HDC entries back off the drawn artists: payload presence alone
    # must not put a curve in the legend that never made it onto the axes.
    for handle, label in _drawn_hdc_legend_entries(axes):
        handles.append(handle)
        labels.append(label)

    coronagraph_orders = _ideal_coronagraph_orders(runs)
    if coronagraph_orders and _ideal_coronagraph_curve_inputs(runs) is not None:
        for order in coronagraph_orders:
            style = _IDEAL_CORONAGRAPH_STYLES.get(
                order, {"linestyle": "--", "linewidth": 1.4}
            )
            handles.append(
                Line2D([], [], color=_IDEAL_CORONAGRAPH_COLOR, **style)
            )
            labels.append(f"ideal coronagraph, order {order} (Guyon 2006)")

    if planets and handles:
        handles.append(Line2D([], [], linestyle="", color="none"))
        labels.append("")

    seen_types = {p.planet_type for p in planets}
    for planet_type in ("Rocky", "Super-Earth", "Neptune-like", "Gas Giant"):
        if planet_type not in seen_types:
            continue
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                color="w",
                markerfacecolor=TYPE_COLORS[planet_type],
                markeredgecolor="none",
                markersize=7,
                linestyle="",
            )
        )
        labels.append(TYPE_DISPLAY_LABELS.get(planet_type, planet_type))

    if planets and seen_types:
        handles.append(Line2D([], [], linestyle="", color="none"))
        labels.append("")

    seen_methods = {p.discovery_method for p in planets}
    for method in ("Radial Velocity", "Transit", "Imaging", "Other"):
        if method not in seen_methods:
            continue
        handles.append(
            Line2D(
                [],
                [],
                marker=METHOD_MARKERS[method],
                color="none",
                markerfacecolor="0.5",
                markeredgecolor="none",
                markersize=7,
                linestyle="",
            )
        )
        labels.append(method)

    if not handles:
        return
    legend = axes.legend(
        handles,
        labels,
        loc="upper right",
        fontsize=8,
        frameon=True,
        framealpha=0.9,
        handlelength=2.4,
        # State the 5-sigma convention on the figure itself — previously
        # this title helper was dead code and the convention appeared
        # nowhere (review 2026-06-10 Issue 23a).
        title=_contrast_curve_legend_title(runs),
        title_fontsize=8,
    )
    legend.get_frame().set_linewidth(0.8)


def _contrast_curve_legend_title(runs: RunCollection) -> str:
    times = []
    for record in runs.records:
        if record.result.state.results.contrast_curve is None:
            continue
        for name, payload in record.result.state.results.components.items():
            if not name.startswith("performance.standard"):
                continue
            if not isinstance(payload, dict):
                continue
            value = payload.get("contrast_curve_reference_time_s")
            if value is not None:
                times.append(float(value))
            break
    if not times:
        return r"$5\sigma$ contrast limit"
    first = times[0]
    if all(
        math.isclose(value, first, rel_tol=1.0e-12, abs_tol=1.0e-9) for value in times
    ):
        return rf"$5\sigma$ contrast limit" "\n" f"(t = {_format_seconds(first)})"
    return r"$5\sigma$ contrast limit" "\n" "(mixed t)"


def _format_seconds(seconds: float) -> str:
    if math.isclose(seconds, round(seconds), abs_tol=1.0e-9):
        return f"{int(round(seconds))} s"
    return f"{seconds:.3g} s"


# ----------------------------------------------------------------------
# Record-level helpers
# ----------------------------------------------------------------------


def _record_label(record: RunRecord) -> str:
    from nullsim.outputs.plots._axis_labels import compact_record_label

    return compact_record_label(record.coord_tuple, record.config.run.name)


_DEFAULT_HEADLINE_JMAGS = (4.0, 8.0)


def _headline_jmags(runs: RunCollection) -> tuple[float, ...]:
    """Configured headline J magnitudes ([output.detection_space]) or the
    paperv4 default (J=4/J=8)."""

    output = getattr(runs.config, "output", None)
    config = getattr(output, "detection_space", None)
    if config is None:
        return _DEFAULT_HEADLINE_JMAGS
    # An explicitly configured empty tuple means "plot every cell".
    return tuple(float(v) for v in config.headline_jmags)


def _paper_curve_records(runs: RunCollection) -> tuple[RunRecord, ...]:
    """Headline-cell selection for the paper figure.

    Keeps only the configured headline J magnitudes when the sweep includes
    them, and LOGS which cells were withheld — pre-fix 3 of 5 sweep cells
    were silently dropped from the published figure with the selection
    living only in code (review 2026-06-10 Issue 23a).
    """

    records = tuple(runs.records)
    headline = _headline_jmags(runs)
    selected = tuple(
        record
        for record in records
        if (jmag := _jmag_of_record(record)) is not None
        and any(
            math.isclose(jmag, target, rel_tol=0.0, abs_tol=0.05)
            for target in headline
        )
    )
    if selected and len(selected) < len(records):
        dropped = sorted(
            {
                f"J={_jmag_of_record(r):g}"
                for r in records
                if r not in selected and _jmag_of_record(r) is not None
            }
        )
        logging.info(
            "detection_space: plotting headline cells %s; withholding %s "
            "(configure [output.detection_space].headline_jmags to change).",
            [f"J={j:g}" for j in headline],
            dropped,
        )
    return selected if selected else records


def _curve_legend_label(
    record: RunRecord,
    *,
    mode: str,
    extra: str | None = None,
    time_s: float | None = None,
) -> str:
    """Per-record legend label for one curve family.

    ``extra`` appends a family-specific qualifier inside the parentheses (the
    HDC family's resolving power, and its threshold when that is not 5 sigma);
    the record identity ahead of it is shared, so the three families read the
    same way down the legend. ``time_s`` overrides the record-level contrast
    reference time for families that carry their own exposure -- the HDC
    overlay does, and using the record's would mislabel it.
    """

    instrument = _instrument_label(record.config.run.name)
    jmag = _jmag_of_record(record)
    j_label = f"J={jmag:g}" if jmag is not None else _record_label(record)
    time_s = time_s if time_s is not None else _contrast_reference_time_s(record)
    time_label = f", {_format_duration_hours(time_s)}" if time_s is not None else ""
    suffix = f", {extra}" if extra else ""
    if mode == "char":
        return f"{instrument} char-mode ({j_label}{time_label}{suffix})"
    if mode == "hdc":
        return f"{instrument} {_HDC_LEGEND_PREFIX} ({j_label}{time_label}{suffix})"
    return f"{instrument} ({j_label}{time_label}{suffix})"


def _instrument_label(run_name: str) -> str:
    lowered = run_name.lower()
    if "kilo" in lowered:
        return "KILO"
    if "nayra" in lowered:
        return "NAYRA"
    if "hwo" in lowered:
        return "HWO"
    return run_name


def _contrast_reference_time_s(record: RunRecord) -> float | None:
    for name, payload in record.result.state.results.components.items():
        if not name.startswith("performance.standard") or not isinstance(payload, dict):
            continue
        value = payload.get("contrast_curve_reference_time_s")
        if value is None:
            return None
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if math.isfinite(out) and out > 0.0 else None
    return None


def _format_duration_hours(seconds: float) -> str:
    hours = seconds / 3600.0
    if math.isclose(hours, round(hours), abs_tol=1.0e-9):
        return f"{int(round(hours))}h"
    return f"{hours:.3g}h"


def _short_axis_name(axis_name: str) -> str | None:
    """Backwards-compatible re-export of the shared helper."""

    from nullsim.outputs.plots._axis_labels import short_axis_name

    return short_axis_name(axis_name)


def _jmag_of_record(record: RunRecord) -> float | None:
    for axis_name, axis_value in record.coord_tuple:
        if axis_name == "scene.star.jmag":
            try:
                return float(axis_value)
            except (TypeError, ValueError):
                return None
    star = getattr(record.config.scene, "star", None)
    if star is None:
        return None
    return getattr(star, "jmag", None)


def _line_style_for_record(record: RunRecord) -> dict[str, Any]:
    jmag = _jmag_of_record(record)
    if jmag is None:
        return dict(_DEFAULT_LINE_STYLE)
    closest = min(_JMAG_LINE_STYLES, key=lambda entry: abs(entry[0] - jmag))
    return dict(closest[1])


def _is_primary_fill_record(record: RunRecord) -> bool:
    jmag = _jmag_of_record(record)
    return jmag is not None and math.isclose(jmag, 4.0, rel_tol=0.0, abs_tol=0.05)


def _stellar_diameter_mas(runs: RunCollection) -> float | None:
    star = getattr(runs.config.scene, "star", None)
    if star is None:
        return None
    diameter = getattr(star, "angular_diameter_mas", None)
    if diameter is None:
        return None
    return float(diameter)


__all__ = ["detection_space"]
