"""Run context values passed into stages per SPEC §§6.1 and 6.5."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RunContext:
    """Per-stage execution context with deterministic RNG identity.

    Args:
        rng: Stage-local random generator derived from run seed, the stage's
            own spec digest (``rng_identity``), sweep coordinate, and stage
            instance id.
        root_seed: Top-level ``[run].seed`` from the config. Stored so the
            cache key can fold in the full RNG-derivation tuple per
            SPEC §6.5 (Ultra-review High #1 — without this a stochastic
            stage's second run with a different seed could cache-hit
            the first run's entry).
        config_hash: Digest identifying the resolved configuration.
        sweep_coord_tuple: Tuple of ``(axis_name, value)`` pairs for this sweep
            cell. Empty for non-sweep runs.
        instance_id: Stage instance id from the ordered pipeline list.
        run_dir: Directory where run products will be written.
        source_path: Path to the TOML file that produced this run, when known.
            Stages use this to resolve user-specified relative data files
            against the config location rather than the process cwd.
        rng_identity: Canonical digest of this stage's own spec, used (with
            root_seed, sweep coordinate, and instance id) to seed ``rng``.
            Stage-local by design: a config edit that does not touch this
            stage's spec must not change its stochastic draw (review
            2026-06-10 Issue 7). ``None`` only for low-level unit tests
            that build a RunContext by hand; the cache key then falls
            back to ``config_hash``.
    """

    rng: np.random.Generator
    config_hash: str
    sweep_coord_tuple: tuple[tuple[str, Any], ...]
    instance_id: str
    run_dir: Path
    root_seed: int = 0
    source_path: Path | None = None
    rng_identity: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_dir", Path(self.run_dir))
        if self.source_path is not None:
            object.__setattr__(self, "source_path", Path(self.source_path))
