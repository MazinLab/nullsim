"""Built-in preset loading for the config layer."""

from __future__ import annotations

import tomllib
from importlib import resources
from pathlib import Path
from typing import Any

from nullsim.config.errors import NullsimConfigError


def preset_path(name: str) -> Any:
    """Return the importlib resource path for a built-in preset name."""

    preset_stem = _preset_stem(name)
    resource = resources.files(__package__).joinpath(f"{preset_stem}.toml")
    if not resource.is_file():
        raise NullsimConfigError(f"Unknown built-in preset {name!r}.")
    return resource


def load_preset(name: str, section: str | None = None) -> dict[str, Any]:
    """Load a built-in TOML preset, optionally unwrapping one section."""

    resource = preset_path(name)
    with resource.open("rb") as handle:
        data = tomllib.load(handle)
    if section is not None and isinstance(data.get(section), dict):
        return dict(data[section])
    return data


def _preset_stem(name: str) -> str:
    if not name:
        raise NullsimConfigError("Preset name must be non-empty.")
    path = Path(name)
    if path.name != name or name in {".", ".."}:
        raise NullsimConfigError("Preset names must be built-in file stems, not paths.")
    return name.removesuffix(".toml")

