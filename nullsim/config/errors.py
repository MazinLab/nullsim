"""Configuration-layer exceptions shared by loaders and pipeline validation."""

from __future__ import annotations


class NullsimConfigError(ValueError):
    """Raised when a resolved nullsim configuration is internally invalid."""

