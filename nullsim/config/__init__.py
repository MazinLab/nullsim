"""Configuration-layer package for TOML loading, validation, and sweeps."""

from __future__ import annotations

from nullsim.config.errors import NullsimConfigError
from nullsim.config.factories import (
    SCIENCE_BANDS,
    ScienceBand,
    band_grid_sweep_specs,
    science_band,
)
from nullsim.config.hashes import (
    canonical_toml_dumps,
    compute_config_hash,
    compute_run_hash,
    runtime_dependency_versions,
    runtime_python_version,
    write_resolved_config_toml,
)
from nullsim.config.loader import load_config, merge_presets, resolve_config
from nullsim.config.schema import (
    CacheConfig,
    CatalogConfig,
    ExozodiConfig,
    GridConfig,
    ObservationConfig,
    OutputConfig,
    OutputFormatsConfig,
    PipelineConfig,
    PlanetConfig,
    ResolvedConfig,
    RunConfig,
    SceneConfig,
    SiteConfig,
    StageSpec,
    StarConfig,
    SweepSpec,
    TelescopeConfig,
    TopLevelConfig,
)
from nullsim.config.sweeps import SweepCell, expand_sweeps

__all__ = [
    "CacheConfig",
    "CatalogConfig",
    "ExozodiConfig",
    "GridConfig",
    "NullsimConfigError",
    "ObservationConfig",
    "OutputConfig",
    "OutputFormatsConfig",
    "PipelineConfig",
    "PlanetConfig",
    "ResolvedConfig",
    "RunConfig",
    "SCIENCE_BANDS",
    "SceneConfig",
    "ScienceBand",
    "SiteConfig",
    "StageSpec",
    "StarConfig",
    "SweepCell",
    "SweepSpec",
    "TelescopeConfig",
    "TopLevelConfig",
    "band_grid_sweep_specs",
    "canonical_toml_dumps",
    "compute_config_hash",
    "compute_run_hash",
    "expand_sweeps",
    "load_config",
    "merge_presets",
    "resolve_config",
    "runtime_dependency_versions",
    "runtime_python_version",
    "science_band",
    "write_resolved_config_toml",
]
