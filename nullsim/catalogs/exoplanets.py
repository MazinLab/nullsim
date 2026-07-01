"""Versioned exoplanet catalog snapshot loader (SPEC §8.1, §8.2, §15.4).

The headline ``detection_space`` plot overlays the known exoplanet population on
top of per-cell contrast curves. SPEC §8.1 requires that the snapshot of that
population be *versioned* — source URL, query, retrieval date, file hash,
classification rule version — and that the file hash propagate into ``run_hash``
so a catalog refresh produces a new output directory and a new manifest entry
rather than silently shifting the points underneath an already-published figure.

The CSV schema accepted here is the IPAC / NASA Exoplanet Archive download
format. Only a small subset of columns is required (``pl_name``, ``ra``,
``dec``, ``pl_bmasse``, ``pl_rade``, ``pl_orbsmax``, ``pl_orbeccen``,
``sy_dist``, ``discoverymethod``); extra columns are ignored. A separation in
mas is computed from semi-major axis and distance when not provided directly.
A contrast estimate column may be supplied; when absent the loader leaves it
``None`` and downstream consumers can fall back on a simple flux-ratio model.

Classification is *versioned*: ``classify_planet`` takes a rule-version string
(``"v1"`` today) and dispatches to the matching radius-threshold table. A
later rule revision (e.g. mass-aware Neptune/Sub-Neptune split) increments the
version, which itself enters ``run_hash`` via the catalog config so retroactive
recategorization is traceable.
"""

from __future__ import annotations

import csv
import hashlib
from datetime import date, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PlanetType = Literal["Rocky", "Super-Earth", "Neptune-like", "Gas Giant"]
DiscoveryMethod = Literal["RV", "Transit", "Imaging", "Other"]


_DISCOVERY_METHOD_ALIASES: dict[str, DiscoveryMethod] = {
    "radial velocity": "RV",
    "rv": "RV",
    "transit": "Transit",
    "primary transit": "Transit",
    "transit timing variations": "Transit",
    "imaging": "Imaging",
    "direct imaging": "Imaging",
}


def _normalize_discovery_method(value: str | None) -> DiscoveryMethod:
    """Map free-form discovery-method strings to the four-bucket controlled vocab."""
    if not value:
        return "Other"
    key = value.strip().lower()
    return _DISCOVERY_METHOD_ALIASES.get(key, "Other")


# Classification rule tables. Keyed by version string so historical figures can
# be regenerated under their original rule even after the table is revised.
_CLASSIFY_RULES_V1: tuple[tuple[float, PlanetType], ...] = (
    (1.5, "Rocky"),
    (4.0, "Super-Earth"),
    (10.0, "Neptune-like"),
    (float("inf"), "Gas Giant"),
)
_CLASSIFY_RULES: dict[str, tuple[tuple[float, PlanetType], ...]] = {
    "v1": _CLASSIFY_RULES_V1,
}


def classify_planet(radius_earth: float, mass_earth: float | None, version: str) -> PlanetType:
    """Bucket a planet into the SPEC §8.1 four-class controlled vocabulary.

    Args:
        radius_earth: Planet radius in Earth radii.
        mass_earth: Planet mass in Earth masses (currently unused by ``v1`` but
            kept in the signature so future rules can dispatch on it without an
            API break).
        version: Classification-rule version. ``"v1"`` is the only rule shipped
            today; the version string is also folded into ``run_hash`` via
            ``ExoplanetCatalogConfig``.

    Returns:
        One of ``"Rocky"`` | ``"Super-Earth"`` | ``"Neptune-like"`` | ``"Gas Giant"``.

    Raises:
        ValueError: ``version`` is not a known rule, or ``radius_earth`` is not
            a positive finite number.
    """
    del mass_earth  # reserved for future versions
    if version not in _CLASSIFY_RULES:
        raise ValueError(
            f"Unknown classification rule version {version!r}; known: "
            f"{sorted(_CLASSIFY_RULES)}"
        )
    if not (radius_earth > 0.0 and radius_earth < float("inf")):
        raise ValueError(f"radius_earth must be a positive finite number; got {radius_earth!r}")
    for upper_bound, label in _CLASSIFY_RULES[version]:
        if radius_earth < upper_bound:
            return label
    # Unreachable: the last rule entry uses +inf as its upper bound.
    raise AssertionError("classification rule table missing terminal entry")


def compute_snapshot_hash(snapshot_path: Path) -> str:
    """Return the SHA-256 hex digest of the file contents at ``snapshot_path``.

    Determinism is the contract here: same byte stream → same digest, every
    time, on every machine. The digest enters ``run_hash`` so changing the
    snapshot changes the output directory.
    """
    digest = hashlib.sha256()
    with snapshot_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Exoplanet(BaseModel):
    """One validated planet record from the catalog snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    ra_deg: float | None = None
    dec_deg: float | None = None
    mass_earth: float | None = Field(default=None, ge=0.0)
    radius_earth: float | None = Field(default=None, gt=0.0)
    semi_major_axis_au: float | None = Field(default=None, gt=0.0)
    eccentricity: float | None = Field(default=None, ge=0.0, le=1.0)
    distance_pc: float | None = Field(default=None, gt=0.0)
    separation_mas: float | None = Field(default=None, ge=0.0)
    contrast_estimate: float | None = Field(default=None, gt=0.0)
    planet_type: PlanetType | None = None
    discovery_method: DiscoveryMethod = "Other"

    @model_validator(mode="after")
    def _fill_separation(self) -> Exoplanet:
        """Compute separation_mas from sma + distance when not supplied directly.

        For a circular projected orbit at quadrature, separation in mas is
        ``(a / d_pc) * 1000`` (since 1 AU at 1 pc subtends 1 arcsec). Catalog
        readers may supply either, so this materializes the convenient form
        when only the ingredients are given.
        """
        if self.separation_mas is not None:
            return self
        if self.semi_major_axis_au is None or self.distance_pc is None:
            return self
        derived_mas = (self.semi_major_axis_au / self.distance_pc) * 1000.0
        # frozen=True forbids attribute mutation; rebuild via __dict__ swap
        # is the documented escape hatch for model_validator(mode="after").
        object.__setattr__(self, "separation_mas", derived_mas)
        return self


class ExoplanetCatalogConfig(BaseModel):
    """Validated ``[catalog.exoplanets]`` block from the user TOML.

    The fields mirror SPEC §8.1's catalog-versioning recipe: identify *where*
    the snapshot came from (``source_url``, ``query``), *when* it was pulled
    (``retrieval_date``), *what* it is byte-for-byte (``snapshot_path``,
    ``snapshot_hash``), and *how* the entries were bucketed
    (``classification_rule_version``). All of these except the path are folded
    into the catalog content_hash that propagates into ``run_hash``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_url: str | None = None
    query: str | None = None
    retrieval_date: date
    snapshot_path: Path
    snapshot_hash: str = ""
    classification_rule_version: str = "v1"

    DEFAULT_RULE_VERSION: ClassVar[str] = "v1"

    @field_validator("retrieval_date", mode="before")
    @classmethod
    def _parse_retrieval_date(cls, value: Any) -> date:
        """Accept ISO-8601 date strings; reject anything else loudly."""
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, str):
            try:
                return date.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(
                    f"retrieval_date must be ISO-8601 (YYYY-MM-DD); got {value!r}"
                ) from exc
        raise ValueError(f"retrieval_date must be an ISO-8601 date string; got {value!r}")

    @field_validator("retrieval_date")
    @classmethod
    def _reject_future_dates(cls, value: date) -> date:
        if value > date.today():
            raise ValueError(
                f"retrieval_date {value.isoformat()} is in the future; catalogs cannot be "
                "downloaded from the future"
            )
        return value

    @field_validator("classification_rule_version")
    @classmethod
    def _check_rule_version(cls, value: str) -> str:
        if value not in _CLASSIFY_RULES:
            raise ValueError(
                f"classification_rule_version {value!r} is not registered; known rules: "
                f"{sorted(_CLASSIFY_RULES)}"
            )
        return value

    @model_validator(mode="after")
    def _autocompute_snapshot_hash(self) -> ExoplanetCatalogConfig:
        """Fill in ``snapshot_hash`` when omitted, OR verify it when pinned.

        Three modes:

        * **Auto** (``snapshot_hash`` blank, file exists): hash the file
          bytes and store the digest so it flows into ``run_hash``.
        * **Missing file**: raise immediately. A declared catalog is part of
          run identity and cannot be represented by an empty snapshot hash.
        * **Paranoia** (``snapshot_hash`` pinned): hash the file bytes
          and raise if they don't match the pinned digest. If the file
          is missing entirely under a pinned hash, RAISE: a stale pinned
          digest would otherwise flow into ``run_hash`` and ``content_hash``
          while ``load_catalog`` raises ``FileNotFoundError`` at plot
          time — silent corruption of the run identity. Code Review
          2026-05-16 Issue 17.
        """
        path = self.snapshot_path
        if not path.is_file():
            pin_clause = (
                f" pinned snapshot_hash={self.snapshot_hash!r}" if self.snapshot_hash else ""
            )
            raise ValueError(
                f"ExoplanetCatalogConfig: snapshot file does not exist{pin_clause}: "
                f"{path}. Restore the file or remove the [catalog.exoplanets] block."
            )
        actual = compute_snapshot_hash(path)
        if self.snapshot_hash:
            if self.snapshot_hash != actual:
                raise ValueError(
                    f"ExoplanetCatalogConfig: pinned snapshot_hash "
                    f"{self.snapshot_hash!r} does not match the actual "
                    f"SHA-256 of {path} ({actual!r}). Either refresh the "
                    "pin or restore the original snapshot."
                )
            return self
        object.__setattr__(self, "snapshot_hash", actual)
        return self

    def content_hash(self) -> str:
        """Return the SHA-256 fed into ``run_hash`` for catalog versioning.

        Includes the snapshot bytes hash, the retrieval date, and the
        classification-rule version. Source URL and query are descriptive
        metadata, not identity — two different queries that happen to return
        the same byte stream and use the same rule produce the same figure.
        """
        digest = hashlib.sha256()
        digest.update(b"catalog.exoplanets\x00")
        digest.update(self.snapshot_hash.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(self.retrieval_date.isoformat().encode("utf-8"))
        digest.update(b"\x00")
        digest.update(self.classification_rule_version.encode("utf-8"))
        return digest.hexdigest()


# CSV column aliases. NASA Exoplanet Archive uses the short "pl_*"/"sy_*"
# names; we accept both those and the human-readable spellings the loader
# uses internally.
_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "name": ("pl_name", "name", "planet_name"),
    "ra_deg": ("ra", "ra_deg"),
    "dec_deg": ("dec", "dec_deg"),
    "mass_earth": ("pl_bmasse", "pl_masse", "mass_earth"),
    "radius_earth": ("pl_rade", "radius_earth"),
    "semi_major_axis_au": ("pl_orbsmax", "semi_major_axis_au", "sma_au"),
    "eccentricity": ("pl_orbeccen", "eccentricity"),
    "distance_pc": ("sy_dist", "distance_pc"),
    "separation_mas": ("separation_mas",),
    "contrast_estimate": ("contrast_estimate", "contrast"),
    "planet_type": ("planet_type",),
    "discovery_method": ("discoverymethod", "discovery_method"),
}


def _pick(row: dict[str, str], field_name: str) -> str | None:
    """Return the first non-empty value among the known column aliases."""
    for column in _COLUMN_ALIASES[field_name]:
        if column in row:
            value = row[column]
            if value is None:
                continue
            stripped = value.strip()
            if stripped:
                return stripped
    return None


def _parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_catalog(config: ExoplanetCatalogConfig) -> tuple[Exoplanet, ...]:
    """Read, validate, and classify the exoplanet snapshot CSV.

    Returns a tuple (immutable) so callers cannot accidentally mutate the
    cached catalog between plots. Rows missing both ``radius_earth`` *and* a
    pre-supplied ``planet_type`` are kept but classified as ``None`` — the
    plot just skips them in the color-coded scatter.

    Raises:
        FileNotFoundError: the snapshot file is missing.
        ValueError: the file is not valid CSV or a row fails pydantic validation.
    """
    snapshot_path = config.snapshot_path
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"Exoplanet snapshot not found: {snapshot_path}")

    # Code Review 2026-05-16 Issue 65: cap snapshot size to guard against
    # accidental or malicious oversized inputs. The Exoplanet Archive
    # full-table CSV is ~10 MB; 100 MB gives generous headroom while
    # rejecting clearly-wrong files.
    _MAX_SNAPSHOT_BYTES = 100 * 1024 * 1024  # 100 MB
    file_size = snapshot_path.stat().st_size
    if file_size > _MAX_SNAPSHOT_BYTES:
        raise ValueError(
            f"Exoplanet snapshot at {snapshot_path} is {file_size} bytes "
            f"(> {_MAX_SNAPSHOT_BYTES} bytes cap). Check the file or raise "
            "the threshold if this is a real catalog."
        )

    rule_version = config.classification_rule_version

    planets: list[Exoplanet] = []
    with snapshot_path.open("r", encoding="utf-8", newline="") as handle:
        # Skip leading IPAC `#` comment lines so a raw NASA Exoplanet Archive
        # download works without manual editing.
        rows = (line for line in handle if not line.startswith("#"))
        reader = csv.DictReader(rows)
        if reader.fieldnames is None:
            raise ValueError(f"Catalog file has no header row: {snapshot_path}")
        for row in reader:
            name = _pick(row, "name")
            if not name:
                # Without a name we can't label or deduplicate — drop it.
                continue
            radius_earth = _parse_optional_float(_pick(row, "radius_earth"))
            mass_earth = _parse_optional_float(_pick(row, "mass_earth"))
            planet_type_raw = _pick(row, "planet_type")
            planet_type: PlanetType | None
            if planet_type_raw is not None and planet_type_raw in (
                "Rocky",
                "Super-Earth",
                "Neptune-like",
                "Gas Giant",
            ):
                planet_type = planet_type_raw  # type: ignore[assignment]
            elif radius_earth is not None:
                planet_type = classify_planet(radius_earth, mass_earth, rule_version)
            else:
                planet_type = None

            planet = Exoplanet(
                name=name,
                ra_deg=_parse_optional_float(_pick(row, "ra_deg")),
                dec_deg=_parse_optional_float(_pick(row, "dec_deg")),
                mass_earth=mass_earth,
                radius_earth=radius_earth,
                semi_major_axis_au=_parse_optional_float(_pick(row, "semi_major_axis_au")),
                eccentricity=_parse_optional_float(_pick(row, "eccentricity")),
                distance_pc=_parse_optional_float(_pick(row, "distance_pc")),
                separation_mas=_parse_optional_float(_pick(row, "separation_mas")),
                contrast_estimate=_parse_optional_float(_pick(row, "contrast_estimate")),
                planet_type=planet_type,
                discovery_method=_normalize_discovery_method(_pick(row, "discovery_method")),
            )
            planets.append(planet)
    return tuple(planets)


__all__ = [
    "DiscoveryMethod",
    "Exoplanet",
    "ExoplanetCatalogConfig",
    "PlanetType",
    "classify_planet",
    "compute_snapshot_hash",
    "load_catalog",
]
