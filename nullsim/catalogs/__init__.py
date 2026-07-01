"""Versioned external catalogs consumed by registered output products.

The catalog package is the home for any read-only reference data set whose
contents must be pinned into ``run_hash`` so a refresh produces a new output
directory rather than silently shifting the figures (SPEC §8.1, §8.2, §15.4).

Today the only inhabitant is the exoplanet snapshot used by the headline
``detection_space`` plot. Future entries (target lists, instrument throughput
tables) follow the same recipe: a pydantic config object pointing at a
local snapshot file, a SHA-256 hash of that file, and a validated loader.
"""

from __future__ import annotations

from nullsim.catalogs.exoplanets import (
    Exoplanet,
    ExoplanetCatalogConfig,
    classify_planet,
    compute_snapshot_hash,
    load_catalog,
)

__all__ = [
    "Exoplanet",
    "ExoplanetCatalogConfig",
    "classify_planet",
    "compute_snapshot_hash",
    "load_catalog",
]
