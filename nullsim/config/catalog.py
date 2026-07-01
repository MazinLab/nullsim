"""TOML schema bridge for the ``[catalog.exoplanets]`` block (SPEC §15.4).

The user-facing TOML names a snapshot CSV by path; the loader resolves that
path against the config file's directory, validates dates, and (when omitted)
auto-fills the SHA-256. The result is the same ``ExoplanetCatalogConfig`` the
rest of the system consumes.

This module exists so the schema layer doesn't import the loader (and the
loader doesn't have to know about TOML semantics like relative paths).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Mapping

from nullsim.catalogs.exoplanets import ExoplanetCatalogConfig
from nullsim.config.errors import NullsimConfigError


def parse_catalog_block(
    raw: Mapping[str, Any] | None,
    *,
    config_dir: Path | None = None,
) -> ExoplanetCatalogConfig | None:
    """Validate the ``[catalog.exoplanets]`` mapping into ``ExoplanetCatalogConfig``.

    Args:
        raw: The dict pulled out of the parsed TOML under ``catalog.exoplanets``.
            ``None`` (or missing) means no catalog block was supplied.
        config_dir: Directory of the loaded TOML, used to resolve
            ``snapshot_path`` when given as a relative path.

    Returns:
        The validated config, or ``None`` if no block was supplied.

    Raises:
        NullsimConfigError: validation failed (bad date, missing path, etc.).
    """
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise NullsimConfigError(
            f"[catalog.exoplanets] must be a TOML table; got {type(raw).__name__}"
        )

    payload = dict(raw)
    snapshot_value = payload.get("snapshot_path")
    if snapshot_value is None:
        raise NullsimConfigError(
            "[catalog.exoplanets].snapshot_path is required so the catalog "
            "snapshot hash can be folded into run_hash."
        )
    snapshot_path = Path(snapshot_value)
    if not snapshot_path.is_absolute() and config_dir is not None:
        snapshot_path = (config_dir / snapshot_path).resolve()
    payload["snapshot_path"] = snapshot_path

    try:
        return ExoplanetCatalogConfig.model_validate(payload)
    except Exception as exc:  # pydantic ValidationError or our ValueError chain
        raise NullsimConfigError(f"Invalid [catalog.exoplanets]: {exc}") from exc


__all__ = ["parse_catalog_block"]
