from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from nullsim.config import NullsimConfigError
from nullsim.pipeline import (
    BaseStage,
    RunContext,
    SimulationState,
    build_instance,
    list_stages,
    run,
)
from nullsim.stages.chip.identity import IdentityChip
from nullsim.stages.detector.ideal_counter import IdealCounter
from nullsim.physics.ruilier_cassaing import UNOBSTRUCTED_GEOMETRIC_COUPLING_LIMIT
from nullsim.stages.injection.ideal import IdealInjection
from nullsim.stages.scene.point_source import PointSource
from nullsim.stages.telescope.fixed_aperture import FixedAperture

GRID = {
    "wavelength_center_um": 1.65,
    "wavelength_bandwidth_um": 0.30,
    "n_wavelength_bins": 3,
}
EXPECTED_STAGE_TYPES = {
    "scene.point_source",
    "telescope.fixed_aperture",
    "injection.ideal",
    "chip.identity",
    "detector.ideal_counter",
}

# Canonical set of all sub-budget attribute names on SimulationState.
_ALL_BUDGET_NAMES: frozenset[str] = frozenset({
    "scene",
    "geometry",
    "field",
    "opd_budget",
    "throughput",
    "wavefront",
    "photon_rates",
    "results",
})


def assert_only_produces_changed(
    stage: BaseStage,
    before: SimulationState,
    after: SimulationState,
) -> None:
    """Assert apply() did not mutate sub-budgets absent from stage.produces."""
    for name in sorted(_ALL_BUDGET_NAMES - stage.produces):
        before_digest = getattr(before, name).content_digest()
        after_digest = getattr(after, name).content_digest()
        assert before_digest == after_digest, (
            f"Stage {stage.type_name!r} mutated budget {name!r} "
            f"which is not in its produces={stage.produces!r}"
        )


def _context(tmp_path: Path, instance_id: str = "stage") -> RunContext:
    return RunContext(
        rng=np.random.default_rng(1),
        config_hash="config",
        sweep_coord_tuple=(),
        instance_id=instance_id,
        run_dir=tmp_path,
    )


def _stage(
    type_name: str,
    tmp_path: Path,
    **params: object,
) -> BaseStage:
    return build_instance(
        {"id": type_name.rsplit(".", maxsplit=1)[-1], "type": type_name, **params},
        _context(tmp_path, type_name),
    )


def _scene_state(tmp_path: Path, star_jmag: float = 5.0) -> SimulationState:
    state = SimulationState.empty(GRID)
    stage = _stage(
        "scene.point_source",
        tmp_path,
        star_jmag=star_jmag,
        planet_contrast=3.0e-7,
        planet_separation_mas=42.0,
        position_angle_deg=15.0,
    )
    return stage.apply(state)


def _geometry_state(tmp_path: Path, star_jmag: float = 5.0) -> SimulationState:
    state = _scene_state(tmp_path, star_jmag)
    stage = _stage("telescope.fixed_aperture", tmp_path, diameter_m=2.0)
    return stage.apply(state)


def _injected_state(tmp_path: Path, star_jmag: float = 5.0) -> SimulationState:
    state = _geometry_state(tmp_path, star_jmag)
    stage = _stage("injection.ideal", tmp_path)
    return stage.apply(state)


def _chip_state(tmp_path: Path, star_jmag: float = 5.0) -> SimulationState:
    state = _injected_state(tmp_path, star_jmag)
    stage = _stage("chip.identity", tmp_path, insertion_loss_db=0.0)
    return stage.apply(state)


def test_autoload_registers_trivial_stage_types() -> None:
    assert EXPECTED_STAGE_TYPES.issubset(set(list_stages()))


@pytest.mark.parametrize(
    "stage_cls",
    [PointSource, FixedAperture, IdealInjection, IdentityChip, IdealCounter],
)
def test_trivial_stage_declares_non_empty_produces(
    stage_cls: type[BaseStage],
) -> None:
    assert stage_cls.produces


def test_point_source_contract(tmp_path: Path) -> None:
    initial_state = SimulationState.empty(GRID)
    stage = _stage(
        "scene.point_source",
        tmp_path,
        star_jmag=5.0,
        planet_contrast=3.0e-7,
        planet_separation_mas=42.0,
        position_angle_deg=15.0,
    )
    state = stage.apply(initial_state)
    assert_only_produces_changed(stage, initial_state, state)

    assert state.scene.star_jmag == 5.0
    assert state.scene.planet_contrast == 3.0e-7
    assert state.scene.planet_separation_mas == 42.0
    assert state.scene.planet_position_angle_deg == 15.0
    assert state.scene.components["point_source"]["star_jmag"] == 5.0


def test_point_source_carries_planet_radial_velocity_into_scene(
    tmp_path: Path,
) -> None:
    initial_state = SimulationState.empty(GRID)
    stage = _stage(
        "scene.point_source",
        tmp_path,
        planet_radial_velocity_kms=42.0,
    )
    state = stage.apply(initial_state)

    assert state.scene.planet_radial_velocity_kms == pytest.approx(42.0)
    assert state.scene.components["point_source"][
        "planet_radial_velocity_kms"
    ] == pytest.approx(42.0)


def test_point_source_default_radial_velocity_is_30_kms(tmp_path: Path) -> None:
    state = _stage("scene.point_source", tmp_path).apply(SimulationState.empty(GRID))

    assert state.scene.planet_radial_velocity_kms == pytest.approx(30.0)


def test_fixed_aperture_contract(tmp_path: Path) -> None:
    before = _scene_state(tmp_path)
    stage = _stage("telescope.fixed_aperture", tmp_path, diameter_m=2.0)
    state = stage.apply(before)
    assert_only_produces_changed(stage, before, state)

    assert state.geometry.diameter_m == 2.0
    assert state.geometry.area_m2 == pytest.approx(math.pi)
    assert state.geometry.n_apertures == 1
    assert state.geometry.components["fixed_aperture"]["n_apertures"] == 1


def test_fixed_aperture_array_exposes_transverse_pupil_positions(
    tmp_path: Path,
) -> None:
    before = _scene_state(tmp_path)
    stage = _stage(
        "telescope.fixed_aperture",
        tmp_path,
        diameter_m=10.0,
        n_apertures=4,
        array_layout="y_array",
        spacing_m=85.0,
    )

    state = stage.apply(before)

    assert state.geometry.aperture_positions_enu_m is not None
    assert state.geometry.aperture_positions_pupil_m is not None
    assert state.geometry.aperture_positions_enu_m.shape == (4, 3)
    assert state.geometry.aperture_positions_pupil_m.shape == (4, 2)
    np.testing.assert_allclose(
        state.geometry.aperture_positions_pupil_m,
        state.geometry.aperture_positions_enu_m[:, :2],
    )
    np.testing.assert_allclose(
        state.geometry.components["fixed_aperture"]["aperture_positions_pupil_m"],
        state.geometry.aperture_positions_pupil_m,
    )


def test_fixed_aperture_accepts_per_aperture_diameters(tmp_path: Path) -> None:
    before = _scene_state(tmp_path)
    diameters = (8.0, 10.0, 6.0)
    stage = _stage(
        "telescope.fixed_aperture",
        tmp_path,
        diameter_m=10.0,
        aperture_diameters_m=diameters,
        n_apertures=3,
        central_obstruction_ratio=0.2,
    )

    state = stage.apply(before)

    expected_areas = np.pi * (np.asarray(diameters) / 2.0) ** 2 * (1.0 - 0.2**2)
    assert state.geometry.diameter_m == pytest.approx(10.0)
    assert state.geometry.area_m2 == pytest.approx(float(np.mean(expected_areas)))
    assert state.geometry.collecting_area_m2 == pytest.approx(float(np.sum(expected_areas)))
    np.testing.assert_allclose(state.geometry.aperture_diameters_m, diameters)
    np.testing.assert_allclose(state.geometry.aperture_areas_m2, expected_areas)
    payload = state.geometry.components["fixed_aperture"]
    np.testing.assert_allclose(payload["mode_areas_m2"], expected_areas)
    assert payload["mode_area_m2"] is None


def test_ideal_injection_contract(tmp_path: Path) -> None:
    before = _geometry_state(tmp_path)
    stage = _stage("injection.ideal", tmp_path)
    state = stage.apply(before)
    assert_only_produces_changed(stage, before, state)

    # With no upstream wavefront-quality stage, Strehl falls back to 1.0
    # but the η_opt geometric coupling (≈ 0.8145 unobstructed) is still
    # applied — IdealInjection seeds √(η_opt × S × (1 − R_Fresnel)).
    assert state.field.amplitudes is not None
    assert state.field.amplitudes.shape == (1, 3, 1)
    expected_amp = math.sqrt(UNOBSTRUCTED_GEOMETRIC_COUPLING_LIMIT) + 0.0j
    np.testing.assert_allclose(state.field.amplitudes, expected_amp)
    # Cumulative transmission stays at 1.0 — the coupling factor lives
    # in the field amplitudes, with provenance in the component dict.
    assert state.throughput.transmission_fraction == (1.0, 1.0, 1.0)
    assert "injection_ideal" in state.throughput.components
    assert state.wavefront.strehl_ratio == (1.0, 1.0, 1.0)


def test_ideal_injection_opt_out_seeds_unit_amplitudes(tmp_path: Path) -> None:
    """``apply_strehl_coupling=False`` reproduces the pre-Phase-0.3 contract."""

    before = _geometry_state(tmp_path)
    stage = _stage("injection.ideal", tmp_path, apply_strehl_coupling=False)
    state = stage.apply(before)
    assert state.field.amplitudes is not None
    np.testing.assert_allclose(state.field.amplitudes, 1.0 + 0.0j)


def test_identity_chip_contract(tmp_path: Path) -> None:
    """At ``insertion_loss_db=0.0`` the identity chip is a true pass-through."""

    before = _injected_state(tmp_path)
    original_field_amplitudes = np.array(before.field.amplitudes, copy=True)
    stage = _stage("chip.identity", tmp_path, insertion_loss_db=0.0)
    chip_state = stage.apply(before)
    assert_only_produces_changed(stage, before, chip_state)

    assert chip_state.field.amplitudes is not None
    np.testing.assert_allclose(chip_state.field.amplitudes, original_field_amplitudes)
    assert chip_state.throughput.transmission_fraction == (1.0, 1.0, 1.0)
    assert "chip_identity" in chip_state.throughput.components


def test_identity_chip_insertion_loss_default(tmp_path: Path) -> None:
    """Default ``insertion_loss_db=-2.0`` scales field by sqrt(t); cumulative throughput unchanged.

    Chip losses live in field amplitudes per the detector's
    ``|field|² × transmission_fraction`` convention. The chip_identity
    component still records ``t`` for the loss-budget plot, but
    cumulative ``transmission_fraction`` is upstream-only.
    """

    import math

    before = _injected_state(tmp_path)
    original_field_amplitudes = np.array(before.field.amplitudes, copy=True)
    upstream_transmission = before.throughput.transmission_fraction
    stage = _stage("chip.identity", tmp_path)  # use Pydantic default
    chip_state = stage.apply(before)

    expected_t = 10.0 ** (-2.0 / 10.0)
    expected_amp_scale = math.sqrt(expected_t)
    np.testing.assert_allclose(
        chip_state.field.amplitudes,
        original_field_amplitudes * expected_amp_scale,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        chip_state.throughput.transmission_fraction,
        upstream_transmission,
        rtol=1.0e-12,
    )
    payload = chip_state.throughput.components["chip_identity"]
    assert payload["insertion_loss_db"] == -2.0
    np.testing.assert_allclose(payload["transmission_fraction"], (expected_t,) * 3)


def test_identity_chip_rejects_positive_insertion_loss_db() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="insertion_loss_db"):
        IdentityChip.Params(insertion_loss_db=1.0)


def test_ideal_counter_contract(tmp_path: Path) -> None:
    before = _chip_state(tmp_path)
    before = dataclasses.replace(
        before,
        results=before.results.add_component("upstream", {"preserve": True}),
    )
    stage = _stage("detector.ideal_counter", tmp_path)
    detector_state = stage.apply(before)
    assert_only_produces_changed(stage, before, detector_state)

    assert detector_state.photon_rates.rates_hz is not None
    assert detector_state.photon_rates.port_labels == ("bright_0", "dark_0")
    assert detector_state.photon_rates.rates_hz.shape == (2, 3)
    assert before.field.amplitudes is not None
    field_power = np.sum(np.abs(before.field.amplitudes) ** 2, axis=(0, 2))
    area_per_mode_m2 = math.pi / before.field.amplitudes.shape[0]
    expected_bright_rates_hz = (
        1.0e10 * 10.0 ** (-0.4 * 5.0) * area_per_mode_m2 * field_power
    )
    expected_dark_rates_hz = expected_bright_rates_hz * 3.0e-7
    np.testing.assert_allclose(
        detector_state.photon_rates.rates_hz,
        np.vstack(
            [
                expected_bright_rates_hz,
                expected_dark_rates_hz,
            ],
        ),
    )
    assert detector_state.results.null_depth == 3.0e-7
    assert detector_state.results.contrast_curve is not None
    assert detector_state.results.contrast_curve["separation_mas"][0] == 42.0
    assert detector_state.results.contrast_curve["contrast"][0] == 3.0e-7
    assert detector_state.results.components["upstream"] == {"preserve": True}
    component = detector_state.results.components["detector.ideal_counter"]
    assert component["null_depth_raw"] == 3.0e-7
    assert component["port_labels"] == ("bright_0", "dark_0")
    np.testing.assert_allclose(
        component["port_rates"],
        np.asarray(
            [
                np.sum(expected_bright_rates_hz),
                np.sum(expected_dark_rates_hz),
            ],
        ),
    )


def test_ideal_counter_uses_field_power_for_chip_loss(tmp_path: Path) -> None:
    before = _injected_state(tmp_path)
    lossless_chip = _stage("chip.identity", tmp_path, insertion_loss_db=0.0)
    lossy_chip = _stage("chip.identity", tmp_path)
    detector = _stage("detector.ideal_counter", tmp_path)

    lossless_rates = detector.apply(
        lossless_chip.apply(before),
    ).photon_rates.rates_hz
    lossy_rates = detector.apply(lossy_chip.apply(before)).photon_rates.rates_hz

    expected_t = 10.0 ** (-2.0 / 10.0)
    np.testing.assert_allclose(lossy_rates, lossless_rates * expected_t)


def test_registry_validates_trivial_stage_params(tmp_path: Path) -> None:
    with pytest.raises(NullsimConfigError, match="diameter_m"):
        _stage("telescope.fixed_aperture", tmp_path, diameter_m=0.0)


def test_trivial_pipeline_runs_end_to_end_and_jmag_changes_rate(
    tmp_path: Path,
) -> None:
    bright_result = run(_trivial_config(tmp_path, star_jmag=5.0))
    faint_result = run(_trivial_config(tmp_path, star_jmag=6.0))

    assert bright_result.state.results.null_depth == 2.0e-6
    assert faint_result.state.results.null_depth == 2.0e-6
    bright_rate_hz = float(np.sum(bright_result.state.photon_rates.rates_hz))
    faint_rate_hz = float(np.sum(faint_result.state.photon_rates.rates_hz))
    assert faint_rate_hz < bright_rate_hz


def test_ideal_counter_feeds_standard_performance(tmp_path: Path) -> None:
    config = _trivial_config(tmp_path, star_jmag=5.0)
    pipeline = config["pipeline"]
    assert isinstance(pipeline, dict)
    stages = pipeline["stages"]
    assert isinstance(stages, list)
    stages.append({"id": "performance", "type": "performance.standard"})

    result = run(config)

    assert "performance.standard" in result.state.results.components
    assert result.state.results.snr is not None


def test_ideal_injection_params_rejects_nan(tmp_path: Path) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        IdealInjection.Params(nan_value=float("nan"))



def test_basestage_validates_params_once_in_init(tmp_path: Path) -> None:
    """Issue #24: Params.model_validate is called by BaseStage.__init__,
    not by every apply(). The typed model is exposed as self.params_model
    and the dict form (post-default-application) is on self.params.

    A counter-instrumented Params model proves there's exactly one
    validation per stage instance, regardless of how many apply() calls.
    """
    from nullsim.pipeline.context import RunContext
    from pydantic import BaseModel, ConfigDict

    validate_calls = {"n": 0}

    class CountingParams(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)
        x: int = 0

        @classmethod
        def model_validate(cls, *args, **kwargs):  # type: ignore[override]
            validate_calls["n"] += 1
            return super().model_validate(*args, **kwargs)

    from nullsim.pipeline.stage import BaseStage as _BaseStage
    from nullsim.pipeline.state import SimulationState as _SimulationState

    class CountingStage(_BaseStage):
        type_name = "tests.params.counting"
        family = "test"
        Params = CountingParams

        def apply(self, state: _SimulationState) -> _SimulationState:
            assert isinstance(self.params_model, CountingParams)
            assert self.params == {"x": 7}  # default-applied dump
            return state

    context = RunContext(
        rng=np.random.default_rng(0),
        config_hash="cfg",
        sweep_coord_tuple=(),
        instance_id="counting",
        run_dir=tmp_path,
    )
    stage = CountingStage({"x": 7}, context)
    assert validate_calls["n"] == 1, "Params validated once in __init__"

    # Many apply() calls — no further validation.
    state = _SimulationState.empty(grid=None)
    stage.apply(state)
    stage.apply(state)
    stage.apply(state)
    assert validate_calls["n"] == 1, "apply() must not revalidate"


def test_identity_chip_params_rejects_nan(tmp_path: Path) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        IdentityChip.Params(nan_value=float("nan"))


def _trivial_config(tmp_path: Path, star_jmag: float) -> dict[str, object]:
    return {
        "config_hash": f"trivial-{star_jmag}",
        "grid": GRID,
        "run": {"seed": 1},
        "cache": {"enabled": False, "dir": tmp_path / "cache", "version": 1},
        "output": {"dir": tmp_path / "run"},
        "pipeline": {
            "stages": [
                {
                    "id": "scene",
                    "type": "scene.point_source",
                    "star_jmag": star_jmag,
                    "planet_contrast": 2.0e-6,
                    "planet_separation_mas": 50.0,
                },
                {
                    "id": "array",
                    "type": "telescope.fixed_aperture",
                    "diameter_m": 2.0,
                },
                {"id": "injection", "type": "injection.ideal"},
                {"id": "chip", "type": "chip.identity", "insertion_loss_db": 0.0},
                {"id": "detector", "type": "detector.ideal_counter"},
            ],
        },
    }
