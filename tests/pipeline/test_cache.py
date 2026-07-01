from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nullsim.pipeline import (
    BaseStage,
    ContentCache,
    ExternalDependency,
    RunContext,
    SimulationState,
    register_stage,
    run,
)


class CacheKeyStage(BaseStage):
    type_name = "tests.cache.key"
    family = "postprocess"
    consumes = frozenset({"scene"})
    produces = frozenset({"results"})

    def apply(self, state: SimulationState) -> SimulationState:
        return state

    def external_dependencies(self) -> list[ExternalDependency]:
        dependency_path = self.params.get("dependency_path")
        if dependency_path is None:
            return []
        return [ExternalDependency(source="test_dependency", path=Path(dependency_path))]


class DynamicConsumeStage(BaseStage):
    type_name = "tests.cache.dynamic_consume"
    family = "postprocess"
    consumes = frozenset({"scene"})
    produces = frozenset({"results"})

    def required_consumes(self) -> frozenset[str]:
        if self.params.get("use_results"):
            return self.consumes | frozenset({"results"})
        return self.consumes

    def apply(self, state: SimulationState) -> SimulationState:
        return state


@register_stage
class CacheRunStage(BaseStage):
    type_name = "tests.cache.run"
    family = "scene"
    produces = frozenset({"scene"})

    def apply(self, state: SimulationState) -> SimulationState:
        scene = state.scene.add_component("run_value", self.params["value"])
        return replace(state, scene=scene)

    def diagnostics(self) -> dict[str, Any]:
        return {"value": self.params["value"]}


@register_stage
class NoCacheExternalDependencyStage(BaseStage):
    type_name = "tests.cache.no_cache_external_dependency"
    family = "scene"
    produces = frozenset({"scene"})

    def apply(self, state: SimulationState) -> SimulationState:
        scene = state.scene.add_component("no_cache_ran", True)
        return replace(state, scene=scene)

    def external_dependencies(self) -> list[ExternalDependency]:
        raise RuntimeError(
            "external dependency should not be read when cache is disabled",
        )


@register_stage
class SceneIndependentStage(BaseStage):
    """Stage that neither consumes nor produces scene (like atmosphere)."""

    type_name = "tests.cache.scene_independent"
    family = "telescope"
    consumes = frozenset()
    produces = frozenset({"geometry"})

    def apply(self, state: SimulationState) -> SimulationState:
        geometry = state.geometry.add_component("layout", "fixed")
        return replace(state, geometry=geometry)


def _context(tmp_path: Path, instance_id: str = "cache") -> RunContext:
    return RunContext(
        rng=np.random.default_rng(1),
        config_hash="config",
        sweep_coord_tuple=(),
        instance_id=instance_id,
        run_dir=tmp_path,
    )


def test_cache_store_and_load_round_trips_state_and_diagnostics(
    tmp_path: Path,
) -> None:
    cache = ContentCache(tmp_path / "cache", cache_version=1)
    state = SimulationState.empty()
    updated_state = replace(
        state,
        scene=state.scene.add_component("star", {"jmag": 5.0}),
    )
    diagnostics = {"stage": {"loss": 0.1}}
    key = "a" * 64

    cache.store(key, updated_state, diagnostics)
    loaded_state, loaded_diagnostics = cache.load(key)

    assert cache.has(key)
    assert loaded_state.scene.components["star"] == {"jmag": 5.0}
    assert loaded_diagnostics == diagnostics


def test_cache_store_same_key_is_safe_for_concurrent_writers(
    tmp_path: Path,
) -> None:
    """Same-key writers must not race on a shared temp file."""

    cache = ContentCache(tmp_path / "cache", cache_version=1)
    key = "b" * 64
    payload = np.arange(100_000, dtype=np.float64)

    def store(index: int) -> None:
        state = SimulationState.empty()
        updated_state = replace(
            state,
            scene=state.scene.add_component(
                "star",
                {"index": index, "payload": payload},
            ),
        )
        cache.store(key, updated_state, {"stage": {"index": index}})

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(store, range(32)))

    loaded_state, loaded_diagnostics = cache.load(key)
    assert cache.has(key)
    assert loaded_state.scene.components["star"]["index"] in range(32)
    assert loaded_diagnostics["stage"]["index"] in range(32)
    assert not list((tmp_path / "cache").glob("*.tmp"))


def test_cache_key_changes_with_params_consumed_state_and_external_dependency(
    tmp_path: Path,
) -> None:
    dependency_path = tmp_path / "dependency.txt"
    dependency_path.write_text("first", encoding="utf-8")
    cache = ContentCache(tmp_path / "cache", cache_version=1)
    state = SimulationState.empty()
    changed_state = replace(state, scene=state.scene.add_component("star", 5.0))

    stage = CacheKeyStage(
        {"gain": 1, "dependency_path": dependency_path},
        _context(tmp_path),
    )
    changed_params_stage = CacheKeyStage(
        {"gain": 2, "dependency_path": dependency_path},
        _context(tmp_path),
    )

    original_key = cache.key_for(stage, state)
    assert cache.key_for(changed_params_stage, state) != original_key
    assert cache.key_for(stage, changed_state) != original_key

    dependency_path.write_text("second", encoding="utf-8")
    assert cache.key_for(stage, state) != original_key


def test_cache_key_covers_unconsumed_budgets(tmp_path: Path) -> None:
    """The key must change when ANY budget changes, consumed or not.

    The cached value is the full post-stage state, which the runner adopts
    wholesale on a hit. A key blind to unconsumed budgets let a hit replace
    the whole state with another sweep cell's (the kilo_keck_6 J-mag sweep
    regression: a scene-independent stage resurrected scene.star_jmag from a
    different cell, suppressing the characterization curve)."""

    cache = ContentCache(tmp_path / "cache", cache_version=1)
    state = SimulationState.empty()
    state_with_results = replace(
        state,
        results=state.results.add_component("upstream", {"value": 1}),
    )

    static_stage = DynamicConsumeStage({"use_results": False}, _context(tmp_path))
    dynamic_stage = DynamicConsumeStage({"use_results": True}, _context(tmp_path))

    assert cache.key_for(static_stage, state) != cache.key_for(
        static_stage,
        state_with_results,
    )
    assert cache.key_for(dynamic_stage, state) != cache.key_for(
        dynamic_stage,
        state_with_results,
    )


def test_run_reports_cache_miss_then_hit(tmp_path: Path) -> None:
    config = {
        "config_hash": "run-cache-config",
        "run": {"seed": 10},
        "cache": {"enabled": True, "dir": tmp_path / "cache", "version": 1},
        "output": {"dir": tmp_path / "run"},
        "pipeline": {
            "stages": [
                {"id": "scene", "type": "tests.cache.run", "value": "cached"},
            ],
        },
    }

    first_result = run(config)
    second_result = run(config)

    assert first_result.cache_hit_count == 0
    assert not first_result.timings[0].cache_hit
    assert first_result.state.scene.components["run_value"] == "cached"
    assert second_result.cache_hit_count == 1
    assert second_result.timings[0].cache_hit
    assert second_result.diagnostics["scene"] == {"value": "cached"}


def test_run_with_cache_disabled_skips_external_dependency_keying(
    tmp_path: Path,
) -> None:
    config = {
        "config_hash": "no-cache-external-dependency",
        "run": {"seed": 10},
        "cache": {"enabled": False, "dir": tmp_path / "cache", "version": 1},
        "output": {"dir": tmp_path / "run"},
        "pipeline": {
            "stages": [
                {
                    "id": "scene",
                    "type": "tests.cache.no_cache_external_dependency",
                },
            ],
        },
    }

    result = run(config)

    assert result.cache_hit_count == 0
    assert not result.timings[0].cache_hit
    assert result.state.scene.components["no_cache_ran"] is True



def test_corrupted_cache_entry_is_discarded_and_recomputed(tmp_path: Path) -> None:
    """A garbage payload in the cache must NOT crash the run.

    Models the "tampered file in shared scratch" scenario from the cache
    threat model: an attacker (or filesystem corruption, or schema drift
    across versions) writes 64-hex-char files that aren't valid pickles
    of (SimulationState, dict). The cache must discard them and trigger
    recomputation rather than crash or, worse, deserialize arbitrary
    pickle code into a partially-loaded state.
    """
    config = {
        "config_hash": "corrupt-test",
        "run": {"seed": 11},
        "cache": {"enabled": True, "dir": tmp_path / "cache", "version": 1},
        "output": {"dir": tmp_path / "run"},
        "pipeline": {
            "stages": [
                {"id": "scene", "type": "tests.cache.run", "value": "first"},
            ],
        },
    }

    # Cold run populates the cache.
    first = run(config)
    assert first.cache_hit_count == 0

    # Tamper: overwrite every cached payload with garbage that has the
    # right filename shape but isn't a valid pickle.
    cache_dir = Path(tmp_path / "cache")
    pickled = list(cache_dir.glob("*.pickle"))
    assert pickled, "expected at least one cached payload"
    for entry in pickled:
        entry.write_bytes(b"\x00\x01\x02 not a pickle \xff\xfe\xfd")

    # Re-run: must NOT crash, must NOT cache-hit, must recompute.
    second = run(config)
    assert second.cache_hit_count == 0, "tampered cache entries must miss"
    assert all(not t.cache_hit for t in second.timings)
    assert second.state.scene.components["run_value"] == "first"


def test_load_rejects_payload_with_wrong_shape(tmp_path: Path) -> None:
    """A pickle whose unpacked shape isn't (SimulationState, dict) is rejected."""
    import pickle as _pickle

    from nullsim.pipeline.cache import CacheLoadError, ContentCache

    cache = ContentCache(cache_dir=tmp_path / "cache", cache_version=1)
    cache.cache_dir.mkdir(parents=True, exist_ok=True)
    bogus_key = "0" * 64
    cache._path_for(bogus_key).write_bytes(_pickle.dumps({"not": "a tuple"}))

    try:
        cache.load(bogus_key)
    except CacheLoadError:
        pass
    else:
        raise AssertionError("expected CacheLoadError on shape mismatch")

    # File should have been deleted as part of the rejection.
    assert not cache._path_for(bogus_key).exists()



def test_two_instances_same_type_get_distinct_cache_keys(tmp_path: Path) -> None:
    """Issue #1: cache key must include per-instance identity.

    Two instances of the same stage type with identical params used to
    cache-collide because the key only included params + consumed state
    + source file + version + externals + cache_version. For a
    stochastic stage, the second instance would hit the first's entry
    and never run its own RNG path. SPEC §6.1 ("two transport.smf28
    instances") and SPEC §6.5 (per-instance RNG) both rely on the
    instance_id being part of the cache identity.
    """
    from nullsim.pipeline.cache import ContentCache
    from nullsim.pipeline.context import RunContext

    cache = ContentCache(cache_dir=tmp_path / "cache", cache_version=1)

    def _ctx(instance_id: str) -> RunContext:
        return RunContext(
            rng=np.random.default_rng(0),
            config_hash="cfg",
            sweep_coord_tuple=(),
            instance_id=instance_id,
            run_dir=tmp_path,
        )

    a = CacheKeyStage({"value": "same"}, _ctx("first"))
    b = CacheKeyStage({"value": "same"}, _ctx("second"))
    state = SimulationState.empty(grid=None)

    key_a = cache.key_for(a, state)
    key_b = cache.key_for(b, state)
    assert key_a != key_b, (
        f"distinct instance_ids must yield distinct keys "
        f"(both got {key_a})"
    )

    # Re-arming with the *same* instance_id reproduces the original key.
    a2 = CacheKeyStage({"value": "same"}, _ctx("first"))
    assert cache.key_for(a2, state) == key_a


def test_two_classes_same_source_file_get_distinct_cache_keys(tmp_path: Path) -> None:
    """Issue #1 follow-on: class qualname must be in the cache key.

    Two distinct stage classes defined in the same source file share
    the package source digest (which now folds in every ``*.py`` under
    ``nullsim/``); without the qualname guard they could collide when
    params and consumed state happen to match. Class qualname rules
    this out.
    """
    from nullsim.pipeline.cache import ContentCache
    from nullsim.pipeline.context import RunContext

    cache = ContentCache(cache_dir=tmp_path / "cache", cache_version=1)

    def _ctx(instance_id: str) -> RunContext:
        return RunContext(
            rng=np.random.default_rng(0),
            config_hash="cfg",
            sweep_coord_tuple=(),
            instance_id=instance_id,
            run_dir=tmp_path,
        )

    # CacheKeyStage and CacheRunStage are both defined in this test
    # module — same source file. With distinct qualnames the keys
    # must differ even when params and instance_id match.
    a = CacheKeyStage({"value": "v"}, _ctx("inst"))
    b = CacheRunStage({"value": "v"}, _ctx("inst"))
    state = SimulationState.empty(grid=None)

    assert cache.key_for(a, state) != cache.key_for(b, state)



def test_cache_key_raises_on_missing_root_seed(tmp_path: Path) -> None:
    """Phase 5: SPEC §6.5 requires root_seed in the cache identity.

    The pre-Phase-5 code f-stringed ``None`` into the identity bytes
    when components were missing, so two runs whose root_seed both
    happened to be None collided. The function now raises ValueError.
    """
    from nullsim.pipeline.cache import _rng_identity_bytes
    from nullsim.pipeline.context import RunContext

    ctx = RunContext(
        rng=np.random.default_rng(0),
        config_hash="cfg",
        sweep_coord_tuple=(),
        instance_id="inst",
        run_dir=tmp_path,
    )
    # Strip root_seed off the context (None is the bug).
    object.__setattr__(ctx, "root_seed", None)
    stage = CacheKeyStage({"value": "v"}, ctx)
    with pytest.raises(ValueError, match="root_seed"):
        _rng_identity_bytes(stage)


def test_cache_key_raises_on_missing_config_hash(tmp_path: Path) -> None:
    """Same protection for config_hash."""
    from nullsim.pipeline.cache import _rng_identity_bytes
    from nullsim.pipeline.context import RunContext

    ctx = RunContext(
        rng=np.random.default_rng(0),
        config_hash="cfg",
        sweep_coord_tuple=(),
        instance_id="inst",
        run_dir=tmp_path,
    )
    object.__setattr__(ctx, "config_hash", None)
    stage = CacheKeyStage({"value": "v"}, ctx)
    with pytest.raises(ValueError, match="config_hash"):
        _rng_identity_bytes(stage)


def test_affects_stage_state_excludes_outputs_and_cli() -> None:
    """Output writers and CLI entry points are excluded from the stage source
    digest; pipeline/config/physics/stages/catalogs code stays included.

    This is what lets a plot/table/paper or CLI edit reuse the cached (often
    multi-hour) stage optimizations instead of forcing a cold re-run.
    """

    from nullsim.pipeline.cache import _affects_stage_state

    # Included: anything that can run during stage execution.
    assert _affects_stage_state("pipeline/runner.py") is True
    assert _affects_stage_state("config/loader.py") is True
    assert _affects_stage_state("physics/clements.py") is True
    assert _affects_stage_state("stages/chip_optimization/torch.py") is True
    assert _affects_stage_state("stages/characterization/spectro.py") is True
    assert _affects_stage_state("catalogs/exoplanets.py") is True

    # Excluded: pure output writers and CLI entry points (run after stages).
    assert _affects_stage_state("outputs/paper.py") is False
    assert _affects_stage_state("outputs/__init__.py") is False
    assert _affects_stage_state("outputs/plots/fringe_observability.py") is False
    assert _affects_stage_state("outputs/tables/__init__.py") is False
    assert _affects_stage_state("cli.py") is False
    assert _affects_stage_state("__main__.py") is False


def test_stage_family_closures_isolate_char_from_det() -> None:
    """Char-mode code is not a dependency of the det-mode families, so editing
    characterization cannot invalidate the chip_optimization / sensitivity
    cache. Closures are computed from real imports via the AST."""

    from nullsim.pipeline.cache import (
        _compute_stage_family_closures,
        _stage_source_digest,
    )

    closures = _compute_stage_family_closures()
    assert closures is not None

    # Every family depends on itself.
    for family in ("characterization", "chip_optimization", "sensitivity"):
        assert family in closures[family]

    # The det path imports no part of the char path, so editing characterization
    # (which only changes the characterization family digest) leaves the det
    # family digests untouched.
    assert "characterization" not in closures["chip_optimization"]
    assert "characterization" not in closures["sensitivity"]
    assert "performance" not in closures["chip_optimization"]
    assert "performance" not in closures["sensitivity"]

    # Characterization genuinely imports chip_optimization + performance, so
    # editing those still correctly busts the characterization cache.
    assert {"chip_optimization", "performance"} <= closures["characterization"]

    # Distinct closures imply distinct stage source digests.
    assert _stage_source_digest("characterization") != _stage_source_digest(
        "chip_optimization"
    )


def test_family_closure_includes_physics_to_stages_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A physics module importing nullsim.stages.<family> must fold that
    family into the shared-core digest: physics is hashed as core, so any
    family it reaches must invalidate every stage (Review 2026-06-09 Issue 8)."""

    from nullsim.pipeline import cache as cache_mod

    core_families = cache_mod._core_reachable_stage_families()
    assert "chip_optimization" in core_families

    baseline_digest = cache_mod._stage_source_digest("scene")
    original_read_bytes = Path.read_bytes

    def patched_read_bytes(path: Path) -> bytes:
        source = original_read_bytes(path)
        if path.as_posix().endswith("stages/chip_optimization/torch.py"):
            return source + b"\n# synthetic cache test edit\n"
        return source

    monkeypatch.setattr(Path, "read_bytes", patched_read_bytes)
    monkeypatch.setattr(cache_mod, "_SOURCE_DIGESTS", None)
    monkeypatch.setattr(cache_mod, "_SOURCE_DIGESTS_MTIME_NS", None)

    edited_digest = cache_mod._stage_source_digest("scene")
    assert edited_digest != baseline_digest


def test_stage_source_digest_unknown_family_is_conservative() -> None:
    """An unknown / missing family folds in every family digest (the
    conservative whole-package fallback), so an unresolved family can only
    over-invalidate, never serve a stale result."""

    import hashlib

    from nullsim.pipeline.cache import _source_digests, _stage_source_digest

    core_digest, family_digests, _closures = _source_digests()
    expected = hashlib.sha256()
    expected.update(core_digest)
    for name in sorted(family_digests):
        expected.update(family_digests[name])
    assert _stage_source_digest("does-not-exist") == expected.digest()
    assert _stage_source_digest(None) == expected.digest()


def test_cache_key_ignores_config_hash_for_non_rng_stage(tmp_path: Path) -> None:
    """A stage that does not draw from context.rng (uses_rng=False) gets the
    same cache key when only config_hash differs, so an unrelated config edit
    reuses its cached result (Tier 3). Its own params still bust it."""

    cache = ContentCache(tmp_path / "cache", cache_version=1)
    state = SimulationState.empty(grid=None)

    def _ctx(config_hash: str) -> RunContext:
        return RunContext(
            rng=np.random.default_rng(0),
            config_hash=config_hash,
            sweep_coord_tuple=(),
            instance_id="inst",
            run_dir=tmp_path,
        )

    key_a = cache.key_for(CacheKeyStage({"value": "v"}, _ctx("config-A")), state)
    key_b = cache.key_for(CacheKeyStage({"value": "v"}, _ctx("config-B")), state)
    assert key_a == key_b
    key_c = cache.key_for(CacheKeyStage({"value": "w"}, _ctx("config-A")), state)
    assert key_c != key_a


def test_cache_key_depends_on_config_hash_for_rng_stage(tmp_path: Path) -> None:
    """A stage that draws from context.rng (uses_rng=True) keeps config_hash in
    its key, so a config edit correctly invalidates its stochastic result."""

    class _RngStage(CacheKeyStage):
        type_name = "tests.cache.rng"
        uses_rng = True

    cache = ContentCache(tmp_path / "cache", cache_version=1)
    state = SimulationState.empty(grid=None)

    def _ctx(config_hash: str) -> RunContext:
        return RunContext(
            rng=np.random.default_rng(0),
            config_hash=config_hash,
            sweep_coord_tuple=(),
            instance_id="inst",
            run_dir=tmp_path,
        )

    key_a = cache.key_for(_RngStage({"value": "v"}, _ctx("config-A")), state)
    key_b = cache.key_for(_RngStage({"value": "v"}, _ctx("config-B")), state)
    assert key_a != key_b


def test_cache_key_depends_on_wavelength_grid(tmp_path: Path) -> None:
    """Editing the run-global wavelength grid invalidates a stage's key even
    when that stage does not consume a grid-carrying budget."""

    cache = ContentCache(tmp_path / "cache", cache_version=1)
    ctx = RunContext(
        rng=np.random.default_rng(0),
        config_hash="cfg",
        sweep_coord_tuple=(),
        instance_id="inst",
        run_dir=tmp_path,
    )
    stage = CacheKeyStage({"value": "v"}, ctx)  # consumes scene, not a grid budget
    state_a = SimulationState.empty(grid={"wavelength_grid_um": (1.2, 1.25, 1.3)})
    state_b = SimulationState.empty(grid={"wavelength_grid_um": (1.2, 1.25, 1.31)})
    assert cache.key_for(stage, state_a) != cache.key_for(stage, state_b)


def test_tier3_under_declared_stages_declare_their_reads() -> None:
    """The stages the audit found reading undeclared state now declare it in
    cache_key_extras, and the context.rng stages are flagged uses_rng."""

    from nullsim.stages.background.space_zodi import SpaceZodiBackground
    from nullsim.stages.chip_optimization.ideal_kernel import IdealKernelProjector
    from nullsim.stages.chip_optimization.scipy import ChipOptimizationScipy
    from nullsim.stages.chip_optimization.torch import ChipOptimizationTorch
    from nullsim.stages.fringe_tracking.closed_loop import ClosedLoopFringeTracker
    from nullsim.stages.injection.ideal import IdealInjection
    from nullsim.stages.telescope.fixed_aperture import FixedAperture
    from nullsim.stages.transport.smf28 import (
        PM780Transport,
        PM980Transport,
        SMF28Transport,
    )
    from nullsim.stages.wavefront.space_residual import SpaceWavefrontResidual

    assert "throughput" in IdealInjection.cache_key_extras
    assert {"wavefront", "opd_budget"} <= SpaceWavefrontResidual.cache_key_extras
    assert "throughput" in FixedAperture.cache_key_extras
    assert {"scene", "field"} <= ClosedLoopFringeTracker.cache_key_extras
    assert "scene" in ChipOptimizationScipy.cache_key_extras
    assert "scene" in ChipOptimizationTorch.cache_key_extras
    assert "scene" in IdealKernelProjector.cache_key_extras

    assert SMF28Transport.uses_rng is True
    assert PM980Transport.uses_rng is True  # inherited
    assert PM780Transport.uses_rng is True  # inherited
    assert SpaceZodiBackground.uses_rng is True
    assert IdealInjection.uses_rng is False


def test_cache_reuse_after_upstream_edit_matches_fresh(tmp_path: Path) -> None:
    """Tier-3 caveat guard: after an upstream config edit, a cached run must
    equal a fresh no-cache run (no stale result from an under-declared
    dependency), while stages unaffected by the edit still reuse the cache."""

    from nullsim.config import expand_sweeps, load_config

    base = expand_sweeps(load_config("examples/test/kilo_maunakea.toml"))[0].config

    def _with_cache(cfg: Any, enabled: bool, directory: Path) -> Any:
        return cfg.model_copy(
            update={
                "cache": cfg.cache.model_copy(
                    update={"enabled": enabled, "dir": directory}
                )
            }
        )

    def _signature(result: Any) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, component in result.state.results.components.items():
            if not isinstance(component, dict):
                continue
            for field, value in component.items():
                try:
                    arr = np.asarray(value, dtype=np.complex128)
                except (TypeError, ValueError):
                    continue
                if arr.size and np.isfinite(arr).any():
                    out[f"{name}.{field}"] = float(np.nansum(arr.real)) + float(
                        np.nansum(arr.imag)
                    )
        return out

    # Populate the cache with the baseline config.
    run(_with_cache(base, True, tmp_path / "cache"))

    # Edit an upstream stage param; it propagates through wavefront/opd_budget.
    stages = tuple(
        s.model_copy(update={"wind_speed_m_s": 10.5})
        if getattr(s, "id", None) == "atmosphere"
        else s
        for s in base.pipeline.stages
    )
    edited = base.model_copy(
        update={"pipeline": base.pipeline.model_copy(update={"stages": stages})}
    )

    cached = run(_with_cache(edited, True, tmp_path / "cache"))
    fresh = run(_with_cache(edited, False, tmp_path / "nocache"))

    assert cached.cache_hit_count > 0  # stages upstream of / unaffected by the edit
    sig_cached, sig_fresh = _signature(cached), _signature(fresh)
    assert set(sig_cached) == set(sig_fresh)
    for key in sig_cached:
        assert sig_cached[key] == pytest.approx(
            sig_fresh[key], rel=1.0e-9, abs=1.0e-12
        ), key


def test_cache_hit_does_not_resurrect_unconsumed_scene_across_sweep_cells(
    tmp_path: Path,
) -> None:
    """A scene-independent stage's cache hit must not swap in another sweep
    cell's scene budget.

    Regression for the 2026-06-09 kilo_keck_6 J-mag sweep: the runner adopts
    the cached FULL post-stage state on a hit, so a key that ignored
    unconsumed budgets let ``atmosphere.kolmogorov`` (consumes only geometry)
    hit the J=6 cell's entry from the J=4 cell, silently turning every cell
    into star_jmag=6. characterization.detection_curve then skipped via
    skip_jmags=[6, 10, 12] and the detection_space char curve vanished."""

    def _config(value: str, config_hash: str) -> dict[str, Any]:
        return {
            "config_hash": config_hash,
            "run": {"seed": 10},
            "cache": {"enabled": True, "dir": tmp_path / "cache", "version": 1},
            "output": {"dir": tmp_path / "run"},
            "pipeline": {
                "stages": [
                    {"id": "scene", "type": "tests.cache.run", "value": value},
                    {"id": "array", "type": "tests.cache.scene_independent"},
                ],
            },
        }

    first = run(_config("jmag4", "cell-a"))
    second = run(_config("jmag6", "cell-b"))

    assert first.state.scene.components["run_value"] == "jmag4"
    # A cache hit on the scene-independent stage must never bring the other
    # cell's scene along with it.
    assert second.state.scene.components["run_value"] == "jmag6"
    assert second.state.geometry.components["layout"] == "fixed"


def test_stages_reading_context_rng_declare_uses_rng() -> None:
    """Audit every registered stage class: any class whose source touches
    ``context.rng`` (directly or via the ``getattr(self.context, "rng", ...)``
    pattern) must declare ``uses_rng = True`` so the cache key folds in the
    RNG identity. The hand-maintained list in the tier-3 audit missed
    ``polarization.pm_fiber`` (review 2026-06-10 Issue 25)."""

    import inspect

    from nullsim.pipeline.registry import registry

    offenders: list[str] = []
    for type_name, stage_cls in registry.stages.items():
        try:
            source = inspect.getsource(stage_cls)
        except (OSError, TypeError):
            continue
        touches_rng = "context.rng" in source or 'context, "rng"' in source
        if touches_rng and not getattr(stage_cls, "uses_rng", False):
            offenders.append(type_name)
    assert not offenders, (
        f"Stages drawing from context.rng without uses_rng=True: {offenders}"
    )
