"""Tests for the versioned exoplanet catalog loader (SPEC §8.1, §8.2, §15.4)."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from nullsim.catalogs.exoplanets import (
    ExoplanetCatalogConfig,
    classify_planet,
    compute_snapshot_hash,
    load_catalog,
)
from nullsim.config.catalog import parse_catalog_block
from nullsim.config.errors import NullsimConfigError


_FIXTURE = Path(__file__).parent / "fixtures" / "sample_catalog.csv"


# ---------------------------------------------------------------------------
# Classification rule v1: radius-based thresholds.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("radius_earth", "expected_type"),
    [
        (1.0, "Rocky"),
        (1.49, "Rocky"),
        (1.5, "Super-Earth"),
        (2.5, "Super-Earth"),
        (4.0, "Neptune-like"),
        (7.0, "Neptune-like"),
        (10.0, "Gas Giant"),
        (12.0, "Gas Giant"),
        (50.0, "Gas Giant"),
    ],
)
def test_classify_planet_v1_thresholds(radius_earth: float, expected_type: str) -> None:
    """Pin the SPEC §8.1 v1 radius cuts: <1.5, <4, <10, ≥10."""
    assert classify_planet(radius_earth, mass_earth=None, version="v1") == expected_type


def test_classify_planet_unknown_version_raises() -> None:
    with pytest.raises(ValueError, match="Unknown classification rule version"):
        classify_planet(1.0, mass_earth=None, version="vX")


def test_classify_planet_rejects_non_positive_radius() -> None:
    with pytest.raises(ValueError, match="positive finite"):
        classify_planet(0.0, mass_earth=None, version="v1")
    with pytest.raises(ValueError, match="positive finite"):
        classify_planet(float("inf"), mass_earth=None, version="v1")


# ---------------------------------------------------------------------------
# compute_snapshot_hash: deterministic + content-sensitive.
# ---------------------------------------------------------------------------

def test_compute_snapshot_hash_is_deterministic() -> None:
    first = compute_snapshot_hash(_FIXTURE)
    second = compute_snapshot_hash(_FIXTURE)
    assert first == second
    assert len(first) == 64  # SHA-256 hex = 64 chars


def test_compute_snapshot_hash_changes_with_content(tmp_path: Path) -> None:
    file_a = tmp_path / "a.csv"
    file_b = tmp_path / "b.csv"
    file_a.write_text("pl_name,pl_rade\nfoo,1.0\n", encoding="utf-8")
    file_b.write_text("pl_name,pl_rade\nfoo,1.000001\n", encoding="utf-8")
    assert compute_snapshot_hash(file_a) != compute_snapshot_hash(file_b)


# ---------------------------------------------------------------------------
# ExoplanetCatalogConfig validation.
# ---------------------------------------------------------------------------

def test_catalog_config_autocomputes_snapshot_hash() -> None:
    config = ExoplanetCatalogConfig(
        retrieval_date="2026-04-01",
        snapshot_path=_FIXTURE,
    )
    assert config.snapshot_hash == compute_snapshot_hash(_FIXTURE)


def test_catalog_config_accepts_matching_pinned_snapshot_hash() -> None:
    """Finding 5 regression: a pinned snapshot_hash that matches the file
    bytes is accepted and preserved verbatim."""

    pinned = compute_snapshot_hash(_FIXTURE)
    config = ExoplanetCatalogConfig(
        retrieval_date="2026-04-01",
        snapshot_path=_FIXTURE,
        snapshot_hash=pinned,
    )
    assert config.snapshot_hash == pinned


def test_catalog_config_rejects_mismatching_pinned_snapshot_hash() -> None:
    """Finding 5 regression: a pinned snapshot_hash that disagrees with the
    file bytes raises at validation time, instead of silently flowing a
    stale digest into ``run_hash``.
    """

    wrong = "0" * 64
    with pytest.raises(Exception, match="pinned snapshot_hash"):
        ExoplanetCatalogConfig(
            retrieval_date="2026-04-01",
            snapshot_path=_FIXTURE,
            snapshot_hash=wrong,
        )


def test_catalog_config_rejects_future_date() -> None:
    future = (date.today() + timedelta(days=2)).isoformat()
    with pytest.raises(Exception, match="future"):
        ExoplanetCatalogConfig(retrieval_date=future, snapshot_path=_FIXTURE)


def test_catalog_config_requires_iso8601_date() -> None:
    with pytest.raises(Exception, match="ISO-8601"):
        ExoplanetCatalogConfig(retrieval_date="04/01/2026", snapshot_path=_FIXTURE)


def test_catalog_config_rejects_unknown_rule_version() -> None:
    with pytest.raises(Exception, match="not registered"):
        ExoplanetCatalogConfig(
            retrieval_date="2026-04-01",
            snapshot_path=_FIXTURE,
            classification_rule_version="v99",
        )


def test_parse_catalog_block_requires_snapshot_path() -> None:
    with pytest.raises(NullsimConfigError, match="snapshot_path is required"):
        parse_catalog_block({"retrieval_date": "2026-04-01"})


def test_parse_catalog_block_returns_none_when_absent() -> None:
    assert parse_catalog_block(None) is None


def test_parse_catalog_block_resolves_relative_paths(tmp_path: Path) -> None:
    snapshot = tmp_path / "snap.csv"
    snapshot.write_text("pl_name\nfoo\n", encoding="utf-8")
    config = parse_catalog_block(
        {"retrieval_date": "2026-04-01", "snapshot_path": "snap.csv"},
        config_dir=tmp_path,
    )
    assert config is not None
    assert config.snapshot_path == snapshot.resolve()


# ---------------------------------------------------------------------------
# load_catalog: parse the 5-planet fixture.
# ---------------------------------------------------------------------------

def test_load_catalog_returns_five_planets() -> None:
    config = ExoplanetCatalogConfig(
        retrieval_date="2026-04-01",
        snapshot_path=_FIXTURE,
    )
    planets = load_catalog(config)
    assert len(planets) == 5
    names = {p.name for p in planets}
    assert names == {
        "Proxima Cen b",
        "TRAPPIST-1 e",
        "HD 209458 b",
        "51 Peg b",
        "beta Pic b",
    }


def test_load_catalog_classifies_all_four_types() -> None:
    config = ExoplanetCatalogConfig(
        retrieval_date="2026-04-01",
        snapshot_path=_FIXTURE,
    )
    by_name = {p.name: p for p in load_catalog(config)}
    # Proxima Cen b: R=1.08 → Rocky
    assert by_name["Proxima Cen b"].planet_type == "Rocky"
    # TRAPPIST-1 e: R=0.92 → Rocky (sub-1.5)
    assert by_name["TRAPPIST-1 e"].planet_type == "Rocky"
    # HD 209458 b: R=15.46 → Gas Giant
    assert by_name["HD 209458 b"].planet_type == "Gas Giant"
    # 51 Peg b: R=13.5 → Gas Giant
    assert by_name["51 Peg b"].planet_type == "Gas Giant"
    # beta Pic b: R=18.7 → Gas Giant
    assert by_name["beta Pic b"].planet_type == "Gas Giant"


def test_load_catalog_normalizes_discovery_method() -> None:
    config = ExoplanetCatalogConfig(
        retrieval_date="2026-04-01",
        snapshot_path=_FIXTURE,
    )
    by_name = {p.name: p for p in load_catalog(config)}
    assert by_name["Proxima Cen b"].discovery_method == "RV"
    assert by_name["TRAPPIST-1 e"].discovery_method == "Transit"
    assert by_name["beta Pic b"].discovery_method == "Imaging"


def test_load_catalog_includes_super_earth_when_radius_in_range(tmp_path: Path) -> None:
    """Synthetic row covers the radius bucket the headline fixture lacks."""
    snap = tmp_path / "se.csv"
    snap.write_text(
        "pl_name,pl_rade,separation_mas,contrast_estimate,discoverymethod\n"
        "Test SE,2.5,40.0,1e-6,Transit\n"
        "Test Nep,7.0,80.0,1e-5,Imaging\n",
        encoding="utf-8",
    )
    config = ExoplanetCatalogConfig(retrieval_date="2026-04-01", snapshot_path=snap)
    planets = load_catalog(config)
    types = {p.name: p.planet_type for p in planets}
    assert types["Test SE"] == "Super-Earth"
    assert types["Test Nep"] == "Neptune-like"


def test_catalog_content_hash_changes_on_refresh(tmp_path: Path) -> None:
    """Two snapshots with different bytes → different content_hash → different run_hash input."""
    snap_a = tmp_path / "a.csv"
    snap_b = tmp_path / "b.csv"
    snap_a.write_text("pl_name,pl_rade\nfoo,1.0\n", encoding="utf-8")
    snap_b.write_text("pl_name,pl_rade\nfoo,2.0\n", encoding="utf-8")
    config_a = ExoplanetCatalogConfig(retrieval_date="2026-04-01", snapshot_path=snap_a)
    config_b = ExoplanetCatalogConfig(retrieval_date="2026-04-01", snapshot_path=snap_b)
    assert config_a.content_hash() != config_b.content_hash()


def test_catalog_content_hash_changes_on_rule_version_bump(tmp_path: Path) -> None:
    snap = tmp_path / "x.csv"
    snap.write_text("pl_name\nfoo\n", encoding="utf-8")
    config_v1 = ExoplanetCatalogConfig(retrieval_date="2026-04-01", snapshot_path=snap)
    # Force a "v2" entry into the rule registry just for this test so we can
    # prove the version contributes to the hash without needing two real
    # rule tables.
    from nullsim.catalogs import exoplanets as exo_module

    exo_module._CLASSIFY_RULES["v2"] = exo_module._CLASSIFY_RULES["v1"]
    try:
        config_v2 = ExoplanetCatalogConfig(
            retrieval_date="2026-04-01",
            snapshot_path=snap,
            classification_rule_version="v2",
        )
        assert config_v1.content_hash() != config_v2.content_hash()
    finally:
        del exo_module._CLASSIFY_RULES["v2"]
