"""KILO paperv3-style known-exoplanet overlay for ``detection_space``.

Direct port of ``kilo/exoplanets.py`` (HEAD 47607de) limited to
what the detection_space figure needs: load the NASA Exoplanet Archive
``pscomppars`` snapshot, compute reflected-light contrast under
KILO's Lambertian-orbit-averaged assumptions, fill missing radii via
Chen & Kipping (2017), classify by radius, and apply an altitude cut
for the telescope site.

Constants and thresholds match KILO so the markers reproduce
``paperv3/figures/detection_space.pdf`` 1:1.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import cos, pi, sin
from pathlib import Path
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import integrate

from nullsim.catalogs.exoplanets import Exoplanet, classify_planet
from nullsim.physics.reflected_light import (
    lambertian_phase,
    reflected_light_contrast,
)

# KILO constants — see ``kilo/exoplanets.py:18-30``.
GEOMETRIC_ALBEDO_DEFAULT = 0.3
M_EARTH_PER_M_JUP = 317.83
LAT_KECK_DEG = 19.8260  # Mauna Kea, KILO convention.

# Classification radius thresholds in Earth radii (KILO ``exoplanets.py:131``).
ROCKY_MAX = 1.5
SUPER_EARTH_MAX = 2.5
NEPTUNE_MAX = 6.0

# Marker / colour map (KILO ``natural_design.py:2448``).
TYPE_COLORS: dict[str, str] = {
    "Rocky": "#E74C3C",
    "Super-Earth": "#E67E22",
    "Neptune-like": "#2ECC71",
    "Gas Giant": "#3498DB",
}

# Legend display names matching the KILO paper prose; the internal
# classification names above stay KILO-1:1 for the marker reproduction.
TYPE_DISPLAY_LABELS: dict[str, str] = {
    "Super-Earth": "Super Earth",
    "Neptune-like": "Ice Giant",
}
METHOD_MARKERS: dict[str, str] = {
    "Radial Velocity": "o",
    "Transit": "^",
    "Imaging": "*",
    "Other": "D",
}

# KILO's hand-picked label set for paperv3 detection_space.pdf.
LABEL_PLANETS: frozenset[str] = frozenset(
    {"GJ 876 b", "GJ 876 c", "55 Cnc c", "GJ 3512 b", "tau Cet e", "51 Peg b"}
)

# Per-planet label offsets (KILO ``natural_design.py:2477``).
LABEL_OFFSETS: dict[str, tuple[float, float]] = {
    "GJ 876 b": (8, 8),
    "GJ 876 c": (-32, -12),
    "55 Cnc c": (8, -4),
    "GJ 3512 b": (8, 6),
    "tau Cet e": (8, -8),
    "51 Peg b": (8, 8),
}


@dataclass(frozen=True)
class PlotPlanet:
    """One marker on the detection-space figure."""

    name: str
    separation_mas: float
    contrast: float
    planet_type: str
    discovery_method: str
    radius_rearth: float
    radius_is_estimated: bool


@lru_cache(maxsize=1)
def lambertian_orbit_averaged_phase() -> float:
    """``<Φ>`` averaged over orbital phase and inclination — KILO ``≈ 0.3183``.

    KILO ``performance.py:1321``: double integral of ``Φ(α) × sin(i)``
    over ``i ∈ [0, π]`` and ``orbital_phase ∈ [0, 2π]``, divided by 4π.
    Caches because the integral is constant and called per-planet.
    """

    def integrand(inc: float, phase: float) -> float:
        # α between planet and observer for a circular orbit at inclination
        # i: alpha = arccos(sin(i) cos(phase)), clamped against |x| > 1
        # float drift before arccos.
        cos_alpha = sin(inc) * cos(phase)
        cos_alpha = max(-1.0, min(1.0, cos_alpha))
        alpha = float(np.arccos(cos_alpha))
        return lambertian_phase(alpha) * sin(inc)

    value, _err = integrate.dblquad(
        integrand,
        0.0, 2.0 * pi,  # phase
        0.0, pi,        # inclination
        epsabs=1.0e-8,
        epsrel=1.0e-8,
    )
    return float(value / (4.0 * pi))


def chen_kipping_2017_radius(mass_earth: float) -> float:
    """Estimate radius (R_earth) from mass (M_earth) — Chen & Kipping 2017.

    KILO ``exoplanets.py:103``. Three power-law regimes meet at the
    Earth/Neptune/Jupiter mass break points.
    """

    if mass_earth <= 0.0 or not np.isfinite(mass_earth):
        raise ValueError("mass_earth must be positive and finite.")
    if mass_earth < 2.0:
        return 1.008 * (mass_earth ** 0.279)
    if mass_earth < 131.6:
        return 0.808 * (mass_earth ** 0.589)
    return 11.2  # Jovian-saturated branch.


def classify_by_radius(radius_rearth: float) -> str:
    """Return KILO's four-bin classification — ``exoplanets.py:131``."""

    if radius_rearth < ROCKY_MAX:
        return "Rocky"
    if radius_rearth < SUPER_EARTH_MAX:
        return "Super-Earth"
    if radius_rearth < NEPTUNE_MAX:
        return "Neptune-like"
    return "Gas Giant"


def normalize_discovery_method(method: str | None) -> str:
    """Collapse the NASA Exoplanet Archive method field into KILO's 4 bins."""

    if method is None:
        return "Other"
    cleaned = str(method).strip()
    if cleaned == "Radial Velocity":
        return "Radial Velocity"
    if cleaned in ("Transit", "Transit Timing Variations"):
        return "Transit"
    if cleaned in ("Imaging", "Direct Imaging"):
        return "Imaging"
    return "Other"


def max_altitude_deg(declination_deg: float, latitude_deg: float) -> float:
    """Source's max altitude at upper transit from a site at ``latitude_deg``.

    ``alt_max = 90 - |latitude - declination|`` (when in the southern half of
    the local sky; saturates at 90° on the meridian for ``dec == lat``).
    """

    return 90.0 - abs(latitude_deg - declination_deg)


def load_planets(
    csv_path: Path | str,
    *,
    latitude_deg: float = LAT_KECK_DEG,
    min_altitude_deg: float | None = 45.0,
    geometric_albedo: float = GEOMETRIC_ALBEDO_DEFAULT,
    max_distance_pc: float | None = 50.0,
) -> tuple[PlotPlanet, ...]:
    """Return KILO-style plot-ready planets from the NASA Exoplanet Archive CSV.

    Args:
        csv_path: Path to the ``pscomppars`` snapshot CSV.
        latitude_deg: Telescope site latitude. Defaults to Mauna Kea.
        min_altitude_deg: Drop planets whose max altitude at upper
            transit falls below this threshold. ``45°`` puts a
            zenith-distance cap of ``45°`` on the science target —
            consistent with the user's "above 45 degrees altitude"
            requirement. ``None`` disables this cut for all-sky space
            observatories.
        geometric_albedo: ``A_g`` in the Lambertian reflected-light
            contrast model. ``0.3`` matches KILO paperv3.
        max_distance_pc: Drop planets farther than this. KILO uses
            50 pc as the query radius for the NASA TAP lookup; ``None``
            disables.

    Returns:
        Tuple of ``PlotPlanet`` records ready to scatter.
    """

    df = pd.read_csv(csv_path, comment="#")
    phi_orbit = lambertian_orbit_averaged_phase()

    out: list[PlotPlanet] = []
    for _, row in df.iterrows():
        dist_pc = _coerce_float(row.get("sy_dist"))
        sma_au = _coerce_float(row.get("pl_orbsmax"))
        if dist_pc is None or sma_au is None or dist_pc <= 0.0 or sma_au <= 0.0:
            continue
        if max_distance_pc is not None and dist_pc > max_distance_pc:
            continue
        dec_deg = _coerce_float(row.get("dec"))
        if dec_deg is None:
            continue
        if (
            min_altitude_deg is not None
            and max_altitude_deg(dec_deg, latitude_deg) < min_altitude_deg
        ):
            continue

        radius, radius_is_est = _resolve_radius(row)
        if radius is None:
            continue

        ang_sep_mas = 1000.0 * sma_au / dist_pc
        contrast = reflected_light_contrast(
            radius_earth=radius,
            semi_major_axis_au=sma_au,
            geometric_albedo=geometric_albedo,
            phase_angle_rad=0.0,
        ) * phi_orbit
        if not np.isfinite(contrast) or contrast <= 0.0:
            continue

        name = str(row.get("pl_name") or "").strip() or "?"
        out.append(
            PlotPlanet(
                name=name,
                separation_mas=float(ang_sep_mas),
                contrast=float(contrast),
                planet_type=classify_by_radius(radius),
                discovery_method=normalize_discovery_method(
                    row.get("discoverymethod"),
                ),
                radius_rearth=float(radius),
                radius_is_estimated=bool(radius_is_est),
            )
        )
    return tuple(out)


_CATALOG_METHOD_TO_OVERLAY = {"RV": "Radial Velocity"}


def planets_from_catalog(
    catalog: Iterable[Exoplanet],
    *,
    rule_version: str,
    latitude_deg: float = LAT_KECK_DEG,
    min_altitude_deg: float | None = 45.0,
    geometric_albedo: float = GEOMETRIC_ALBEDO_DEFAULT,
    max_distance_pc: float | None = 50.0,
) -> tuple[PlotPlanet, ...]:
    """Plot-ready planets from a versioned ``load_catalog`` snapshot.

    Separation, contrast, and class come from the catalog when it supplies
    them (``separation_mas``, ``contrast_estimate``, and the declared
    classification rule). Missing radii fall back to Chen & Kipping (2017)
    from the mass, and the contrast to the same orbit-averaged Lambertian
    reflected light as ``load_planets`` (needs a semi-major axis). Planets
    without a declination are dropped when the altitude cut is active, and
    the 50 pc cut applies where a distance is given.
    """

    phi_orbit = lambertian_orbit_averaged_phase()
    out: list[PlotPlanet] = []
    for planet in catalog:
        if (
            max_distance_pc is not None
            and planet.distance_pc is not None
            and planet.distance_pc > max_distance_pc
        ):
            continue
        if min_altitude_deg is not None and (
            planet.dec_deg is None
            or max_altitude_deg(planet.dec_deg, latitude_deg) < min_altitude_deg
        ):
            continue
        if planet.separation_mas is None:
            continue

        radius = planet.radius_earth
        radius_is_estimated = False
        if radius is None and planet.mass_earth:
            radius = chen_kipping_2017_radius(planet.mass_earth)
            radius_is_estimated = True

        contrast = planet.contrast_estimate
        if contrast is None:
            if radius is None or planet.semi_major_axis_au is None:
                continue
            contrast = phi_orbit * reflected_light_contrast(
                radius_earth=radius,
                semi_major_axis_au=planet.semi_major_axis_au,
                geometric_albedo=geometric_albedo,
                phase_angle_rad=0.0,
            )

        planet_type = planet.planet_type
        if planet_type is None and radius is not None:
            planet_type = classify_planet(radius, planet.mass_earth, rule_version)
        out.append(
            PlotPlanet(
                name=planet.name,
                separation_mas=float(planet.separation_mas),
                contrast=float(contrast),
                planet_type=planet_type or "Unclassified",
                discovery_method=_CATALOG_METHOD_TO_OVERLAY.get(
                    planet.discovery_method, planet.discovery_method
                ),
                radius_rearth=float("nan") if radius is None else float(radius),
                radius_is_estimated=radius_is_estimated,
            )
        )
    return tuple(out)


def trappist1_centroid(planets: Sequence[PlotPlanet]) -> tuple[float, float] | None:
    """Return the (sep_mas, contrast) centroid for the TRAPPIST-1 system."""

    members = [p for p in planets if p.name.startswith("TRAPPIST-1 ")]
    if not members:
        return None
    sep = float(np.exp(np.mean(np.log([p.separation_mas for p in members]))))
    con = float(np.exp(np.mean(np.log([p.contrast for p in members]))))
    return sep, con


def _coerce_float(value: object) -> float | None:
    """Return ``value`` as ``float`` or ``None`` on NaN/non-numeric."""

    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _resolve_radius(row) -> tuple[float | None, bool]:
    """KILO radius cascade: ``pl_rade`` > ``pl_bmasse`` (CK17) > ``pl_bmassj`` (CK17)."""

    direct = _coerce_float(row.get("pl_rade"))
    if direct is not None and direct > 0.0:
        return direct, False
    mass_earth = _coerce_float(row.get("pl_bmasse"))
    if mass_earth is not None and mass_earth > 0.0:
        return chen_kipping_2017_radius(mass_earth), True
    mass_jup = _coerce_float(row.get("pl_bmassj"))
    if mass_jup is not None and mass_jup > 0.0:
        return chen_kipping_2017_radius(mass_jup * M_EARTH_PER_M_JUP), True
    return None, False


def planet_legend_handles(planets: Iterable[PlotPlanet]) -> tuple[list, list]:
    """Build (handles, labels) pairs grouped by planet type only.

    Avoids the m×n legend explosion of (type × discovery_method) entries —
    the marker shape encoding is documented in the figure caption.
    """

    import matplotlib.lines as mlines

    seen_types = {p.planet_type for p in planets}
    handles, labels = [], []
    for t in ("Rocky", "Super-Earth", "Neptune-like", "Gas Giant"):
        if t not in seen_types:
            continue
        handles.append(
            mlines.Line2D(
                [], [],
                marker="o",
                color="none",
                markerfacecolor=TYPE_COLORS[t],
                markeredgecolor="none",
                markersize=7,
                linestyle="",
            )
        )
        labels.append(t)
    return handles, labels
