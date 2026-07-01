"""Pipeline-layer public interface for nullsim state, stages, and execution."""

from __future__ import annotations

from nullsim.pipeline.cache import ContentCache
from nullsim.pipeline.context import RunContext
from nullsim.pipeline.registry import (
    Registry,
    build_instance,
    list_stages,
    register_plot,
    register_stage,
    register_table,
    registry,
)
from nullsim.pipeline.rng import derive_rng
from nullsim.pipeline.runner import (
    RunResult,
    StageTiming,
    build_stage_instances,
    run,
    validate_dependencies,
)
from nullsim.pipeline.stage import (
    BaseStage,
    ExternalDependency,
    Stage,
    StageParams,
    StateKey,
    external_dependency_digest,
)
from nullsim.pipeline.state import (
    Field,
    Geometry,
    OPDComponent,
    OPDBudget,
    PhotonRates,
    Results,
    Scene,
    SimulationState,
    SpectralGrid,
    Throughput,
    Wavefront,
)

__all__ = [
    "BaseStage",
    "ContentCache",
    "ExternalDependency",
    "Field",
    "Geometry",
    "OPDComponent",
    "OPDBudget",
    "PhotonRates",
    "Results",
    "Registry",
    "RunContext",
    "RunResult",
    "Scene",
    "SimulationState",
    "Stage",
    "StageParams",
    "StageTiming",
    "StateKey",
    "SpectralGrid",
    "Throughput",
    "Wavefront",
    "build_instance",
    "build_stage_instances",
    "derive_rng",
    "external_dependency_digest",
    "list_stages",
    "register_plot",
    "register_stage",
    "register_table",
    "registry",
    "run",
    "validate_dependencies",
]
