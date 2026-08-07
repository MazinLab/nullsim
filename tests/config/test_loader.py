from __future__ import annotations

from pathlib import Path

import pytest

from nullsim.config import NullsimConfigError, load_config
from nullsim.config.loader import _deep_merge


def _write_config(tmp_path: Path, text: str) -> Path:
    config_path = tmp_path / "config.toml"
    config_path.write_text(text, encoding="utf-8")
    return config_path


def _base_config(extra: str = "") -> str:
    # Place ``extra`` BEFORE any [section] so unknown top-level keys land in the
    # top-level table (StageSpec allows extras, but TopLevelConfig forbids them).
    return f"""
{extra}

[run]
name = "loader_test"
seed = 123

[grid]
wavelength_center_um = 1.65
wavelength_bandwidth_um = 0.3
n_wavelength_bins = 3

[site]
preset = "minimal_site"

[site.weather]
seeing_arcsec = 0.8

[telescope]
preset = "minimal_telescope"

[scene.star]
jmag = 4.5
angular_diameter_mas = 2.0
distance_pc = 3.6

[scene.planet]
contrast = 1.0e-7
separation_mas = 100.0
position_angle_deg = 45.0

[observation]
hour_angle_h = 0.0
integration_time_s = 10.0

[[pipeline.stages]]
id = "scene"
type = "scene.point_source"
custom_gain = 0.5

{extra}
"""


def test_load_config_rejects_unknown_top_level_keys(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, _base_config("unexpected = true"))

    with pytest.raises(NullsimConfigError, match="extra_forbidden|unexpected"):
        load_config(config_path)


def test_load_config_rejects_run_extensions(tmp_path: Path) -> None:
    """[run].extensions is reserved (SPEC §9) but not wired up yet.

    Until extension loading lands, accept-but-ignore would be the worst
    failure mode. Loader rejects it loudly with a clear message.
    """
    # _base_config puts extras BEFORE [run]; here we need them INSIDE
    # [run], so build the config inline.
    base = _base_config()
    patched = base.replace(
        '[run]\nname = "loader_test"\nseed = 123',
        '[run]\nname = "loader_test"\nseed = 123\nextensions = ["my_lab.stages"]',
    )
    config_path = _write_config(tmp_path, patched)

    with pytest.raises(NullsimConfigError, match="extensions"):
        load_config(config_path)


def test_load_config_accepts_unknown_stage_params(tmp_path: Path) -> None:
    resolved_config = load_config(_write_config(tmp_path, _base_config()))

    stage = resolved_config.pipeline.stages[0]

    assert stage.id == "scene"
    assert stage.type == "scene.point_source"
    assert stage.custom_gain == 0.5


def test_load_config_extends_parent_and_merges_stages_by_id(tmp_path: Path) -> None:
    parent = _write_config(tmp_path, _base_config())
    child = tmp_path / "child.toml"
    child.write_text(
        f"""
extends = "{parent.name}"

[run]
name = "child"

[[pipeline.stages]]
id = "scene"
custom_gain = 0.9
new_field = "kept"

[[pipeline.stages]]
id = "extra"
type = "chip.identity"
insertion_loss_db = 0.0
""",
        encoding="utf-8",
    )

    resolved_config = load_config(child)

    assert resolved_config.run.name == "child"
    assert [stage.id for stage in resolved_config.pipeline.stages] == [
        "scene",
        "extra",
    ]
    scene = resolved_config.pipeline.stages[0]
    assert scene.type == "scene.point_source"
    assert scene.custom_gain == 0.9
    assert scene.new_field == "kept"


def test_load_config_extends_can_remove_stage(tmp_path: Path) -> None:
    parent = _write_config(tmp_path, _base_config())
    child = tmp_path / "child.toml"
    child.write_text(
        f"""
extends = "{parent.name}"

[[pipeline.stages]]
id = "scene"
remove = true

[[pipeline.stages]]
id = "chip"
type = "chip.identity"
insertion_loss_db = 0.0
""",
        encoding="utf-8",
    )

    resolved_config = load_config(child)

    assert [stage.id for stage in resolved_config.pipeline.stages] == ["chip"]


def test_load_config_extends_rejects_unknown_stage_remove(
    tmp_path: Path,
) -> None:
    parent = _write_config(tmp_path, _base_config())
    child = tmp_path / "child.toml"
    child.write_text(
        f"""
extends = "{parent.name}"

[[pipeline.stages]]
id = "missing"
remove = true
""",
        encoding="utf-8",
    )

    with pytest.raises(NullsimConfigError, match="missing.*remove"):
        load_config(child)


def test_load_config_extends_can_insert_new_stage_before_inherited_id(
    tmp_path: Path,
) -> None:
    parent = _write_config(tmp_path, _base_config())
    child = tmp_path / "child.toml"
    child.write_text(
        f"""
extends = "{parent.name}"

[[pipeline.stages]]
id = "splitter"
type = "injection.mode_splitter"
before = "scene"
copies = 2
""",
        encoding="utf-8",
    )

    resolved_config = load_config(child)

    assert [stage.id for stage in resolved_config.pipeline.stages] == [
        "splitter",
        "scene",
    ]
    splitter = resolved_config.pipeline.stages[0]
    assert splitter.type == "injection.mode_splitter"
    assert splitter.copies == 2
    assert not hasattr(splitter, "before")


def test_load_config_extends_rejects_unknown_stage_insert_target(
    tmp_path: Path,
) -> None:
    parent = _write_config(tmp_path, _base_config())
    child = tmp_path / "child.toml"
    child.write_text(
        f"""
extends = "{parent.name}"

[[pipeline.stages]]
id = "splitter"
type = "injection.mode_splitter"
before = "missing"
copies = 2
""",
        encoding="utf-8",
    )

    with pytest.raises(NullsimConfigError, match="missing"):
        load_config(child)


def test_load_config_extends_rejects_repositioning_existing_stage(
    tmp_path: Path,
) -> None:
    parent = _write_config(tmp_path, _base_config())
    child = tmp_path / "child.toml"
    child.write_text(
        f"""
extends = "{parent.name}"

[[pipeline.stages]]
id = "scene"
before = "scene"
custom_gain = 0.9
""",
        encoding="utf-8",
    )

    with pytest.raises(NullsimConfigError, match="scene.*before/after"):
        load_config(child)


def test_deep_merge_only_special_cases_pipeline_stages() -> None:
    merged = _deep_merge(
        {"nested": {"stages": [{"id": "a", "value": 1}]}},
        {"nested": {"stages": [{"id": "a", "value": 2}]}},
    )

    assert merged["nested"]["stages"] == [{"id": "a", "value": 2}]


def test_preset_merge_replaces_nested_keys(tmp_path: Path) -> None:
    resolved_config = load_config(_write_config(tmp_path, _base_config()))

    site = resolved_config.site.model_dump()
    telescope = resolved_config.telescope.model_dump()

    assert site["weather"]["seeing_arcsec"] == 0.8
    assert site["weather"]["coherence_time_s"] == 0.01
    assert telescope["aperture"]["diameter_m"] == 1.0
    assert telescope["aperture"]["obscuration_fraction"] == 0.0


def test_site_and_telescope_science_presets_load(tmp_path: Path) -> None:
    config_text = (
        _base_config()
        .replace(
            'preset = "minimal_site"',
            'preset = "cerro_armazones_site"',
            1,
        )
        .replace(
            'preset = "minimal_telescope"',
            'preset = "eelt_telescope"',
            1,
        )
    )

    resolved_config = load_config(_write_config(tmp_path, config_text))

    site = resolved_config.site.model_dump()
    telescope = resolved_config.telescope.model_dump()
    assert site["slug"] == "cerro_armazones"
    assert site["atmosphere"]["r0_500nm_m"] == pytest.approx(0.157)
    assert site["sky"]["profile"] == "cerro_armazones"
    assert telescope["slug"] == "eelt"
    assert telescope["aperture"]["diameter_m"] == pytest.approx(39.146)


def test_load_config_materializes_scene_defaults_into_stage_specs(
    tmp_path: Path,
) -> None:
    """Ultra-review High #2 regression: ``load_config`` must materialize
    ``[scene.*]`` defaults into ``scene.point_source`` stage params so the
    public-API path (``run(load_config(path))``) sees the user's values
    instead of each stage's own Params defaults.
    """

    config_text = """
[run]
name = "matz"

[grid]
wavelength_center_um = 1.65
wavelength_bandwidth_um = 0.4
n_wavelength_bins = 1

[scene.star]
jmag = 9.0
angular_diameter_mas = 1.0
distance_pc = 10.0

[scene.planet]
contrast = 1.0e-6
separation_mas = 10.0
position_angle_deg = 0.0

[observation]
integration_time_s = 3600.0

[[pipeline.stages]]
id = "scene"
type = "scene.point_source"
altitude_deg = 90.0
azimuth_deg = 0.0
"""
    cfg_path = tmp_path / "matz.toml"
    cfg_path.write_text(config_text)

    resolved = load_config(cfg_path)
    scene_stage = next(
        s for s in resolved.pipeline.stages if s.type == "scene.point_source"
    )
    extras = scene_stage.model_dump()
    # The scene.point_source stage's own ``star_jmag`` default is 5.0;
    # materialization must have copied the TOML's 9.0 into the spec.
    assert extras.get("star_jmag") == pytest.approx(9.0), (
        f"materialization did not inject scene.star.jmag; spec extras = {extras}"
    )
    assert extras.get("star_distance_pc") == pytest.approx(10.0), (
        f"materialization did not inject scene.star.distance_pc; spec extras = {extras}"
    )


def test_load_config_materializes_planet_radial_velocity(tmp_path: Path) -> None:
    config_text = """
[run]
name = "rv_materialize"

[grid]
wavelength_center_um = 1.65
wavelength_bandwidth_um = 0.4
n_wavelength_bins = 1

[scene.star]
jmag = 9.0
angular_diameter_mas = 1.0
distance_pc = 10.0

[scene.planet]
contrast = 1.0e-6
separation_mas = 10.0
position_angle_deg = 0.0
radial_velocity_kms = -17.5

[observation]
integration_time_s = 3600.0

[[pipeline.stages]]
id = "scene"
type = "scene.point_source"
"""
    cfg_path = tmp_path / "rv.toml"
    cfg_path.write_text(config_text)

    resolved = load_config(cfg_path)
    scene_stage = next(
        s for s in resolved.pipeline.stages if s.type == "scene.point_source"
    )

    assert scene_stage.model_dump().get(
        "planet_radial_velocity_kms"
    ) == pytest.approx(-17.5)


def test_load_config_materializes_site_telescope_defaults_into_stage_specs(
    tmp_path: Path,
) -> None:
    config_text = """
[run]
name = "site_telescope_materialize"

[site]
preset = "cerro_armazones_site"

[telescope]
preset = "eelt_telescope"

[grid]
wavelength_center_um = 1.25
wavelength_bandwidth_um = 0.1
n_wavelength_bins = 1

[scene.star]
jmag = 5.0
angular_diameter_mas = 1.0
distance_pc = 10.0

[scene.planet]
contrast = 1.0e-6
separation_mas = 10.0
position_angle_deg = 0.0

[observation]
integration_time_s = 3600.0

[[pipeline.stages]]
id = "scene"
type = "scene.point_source"

[[pipeline.stages]]
id = "array"
type = "telescope.fixed_aperture"

[[pipeline.stages]]
id = "atmosphere"
type = "atmosphere.kolmogorov"

[[pipeline.stages]]
id = "ao"
type = "ao.pyramid_wfs"

[[pipeline.stages]]
id = "injection"
type = "injection.ideal"

[[pipeline.stages]]
id = "fiber"
type = "injection.single_mode_fiber"

[[pipeline.stages]]
id = "background"
type = "background.sky_surface_brightness"

[[pipeline.stages]]
id = "fringe_tracking"
type = "fringe_tracking.closed_loop"
"""
    resolved = load_config(_write_config(tmp_path, config_text))
    stages = {
        str(stage.id): stage.model_dump(mode="python")
        for stage in resolved.pipeline.stages
    }

    assert stages["array"]["diameter_m"] == pytest.approx(39.146)
    assert stages["array"]["n_apertures"] == 1
    assert stages["array"]["mirror_transmission"] == pytest.approx(0.85)
    assert stages["array"]["ao_bench_transmission"] == pytest.approx(0.85)
    assert stages["array"]["central_obstruction_ratio"] == pytest.approx(0.11)

    assert stages["atmosphere"]["r0_500nm_m"] == pytest.approx(0.157)
    assert stages["atmosphere"]["L0_m"] == pytest.approx(22.0)
    assert stages["atmosphere"]["wind_speed_m_s"] == pytest.approx(9.0)
    assert stages["atmosphere"]["atmospheric_transmission"] == pytest.approx(0.985)

    assert stages["ao"]["central_obstruction"] == pytest.approx(0.11)
    assert stages["ao"]["tau0_500nm_s"] == pytest.approx(5.5e-3)
    assert stages["ao"]["theta0_500nm_arcsec"] == pytest.approx(2.5)
    assert stages["ao"]["v_wind_m_s"] == pytest.approx(9.0)
    assert stages["ao"]["pupil_central_obstruction_diameter_m"] == pytest.approx(4.306)

    assert stages["injection"]["central_obscuration_ratio"] == pytest.approx(0.11)
    assert stages["fiber"]["central_obscuration_ratio"] == pytest.approx(0.11)
    assert stages["background"]["site_profile"] == "cerro_armazones"
    assert stages["fringe_tracking"]["wind_speed_m_s"] == pytest.approx(9.0)


def test_materialized_site_telescope_defaults_do_not_overwrite_stage_overrides(
    tmp_path: Path,
) -> None:
    config_text = """
[run]
name = "site_telescope_override"

[site]
preset = "cerro_armazones_site"

[telescope]
preset = "eelt_telescope"

[grid]
wavelength_center_um = 1.25
wavelength_bandwidth_um = 0.1
n_wavelength_bins = 1

[scene.star]
jmag = 5.0
angular_diameter_mas = 1.0
distance_pc = 10.0

[scene.planet]
contrast = 1.0e-6
separation_mas = 10.0
position_angle_deg = 0.0

[observation]
integration_time_s = 3600.0

[[pipeline.stages]]
id = "scene"
type = "scene.point_source"

[[pipeline.stages]]
id = "array"
type = "telescope.fixed_aperture"
diameter_m = 12.0

[[pipeline.stages]]
id = "atmosphere"
type = "atmosphere.kolmogorov"
r0_500nm_m = 0.2
"""
    resolved = load_config(_write_config(tmp_path, config_text))
    stages = {
        str(stage.id): stage.model_dump(mode="python")
        for stage in resolved.pipeline.stages
    }

    assert stages["array"]["diameter_m"] == pytest.approx(12.0)
    assert stages["array"]["central_obstruction_ratio"] == pytest.approx(0.11)
    assert stages["atmosphere"]["r0_500nm_m"] == pytest.approx(0.2)
    assert stages["atmosphere"]["L0_m"] == pytest.approx(22.0)


def test_instrument_section_materializes_port_counts(tmp_path: Path) -> None:
    config_text = (
        _base_config()
        + """
[instrument]
n_modes = 8
n_input_modes = 4
n_bright_ports = 4
n_dark_ports = 3
n_spectro_ports = 1

[[pipeline.stages]]
id = "injection"
type = "injection.ideal"

[[pipeline.stages]]
id = "chip"
type = "chip.kernel_mzi_mesh"

[[pipeline.stages]]
id = "chip_opt_scipy"
type = "chip_optimization.scipy"

[[pipeline.stages]]
id = "chip_opt_torch"
type = "chip_optimization.torch"

[[pipeline.stages]]
id = "chip_opt_ideal"
type = "chip_optimization.ideal_kernel"

[[pipeline.stages]]
id = "mkid"
type = "detector.mkid"

[[pipeline.stages]]
id = "snspd"
type = "detector.snspd"

[[pipeline.stages]]
id = "fringe_tracking"
type = "fringe_tracking.closed_loop"

[[pipeline.stages]]
id = "sensitivity"
type = "sensitivity.detection_curve"

[[pipeline.stages]]
id = "characterization"
type = "characterization.spectro"

[[pipeline.stages]]
id = "characterization_curve"
type = "characterization.detection_curve"
"""
    )

    resolved = load_config(_write_config(tmp_path, config_text))
    stages = {
        str(stage.id): stage.model_dump(mode="python")
        for stage in resolved.pipeline.stages
    }
    expected_topology = {
        "n_modes": 8,
        "n_bright_ports": 4,
        "n_dark_ports": 3,
        "n_spectro_ports": 1,
    }

    assert stages["injection"]["n_modes"] == 4
    assert stages["fringe_tracking"]["n_modes"] == 4
    for stage_id in (
        "chip",
        "chip_opt_scipy",
        "chip_opt_torch",
        "chip_opt_ideal",
        "mkid",
        "snspd",
        "characterization",
        "characterization_curve",
    ):
        assert {
            key: stages[stage_id][key] for key in expected_topology
        } == expected_topology
    assert {
        key: stages["sensitivity"][key]
        for key in ("n_bright_ports", "n_dark_ports", "n_spectro_ports")
    } == {
        "n_bright_ports": 4,
        "n_dark_ports": 3,
        "n_spectro_ports": 1,
    }
    assert "n_modes" not in stages["sensitivity"]


def test_stage_explicit_port_count_overrides_instrument(tmp_path: Path) -> None:
    config_text = (
        _base_config()
        + """
[instrument]
n_modes = 8
n_input_modes = 4
n_bright_ports = 4
n_dark_ports = 3
n_spectro_ports = 1

[[pipeline.stages]]
id = "chip"
type = "chip.kernel_mzi_mesh"
n_bright_ports = 5
n_dark_ports = 2
"""
    )

    resolved = load_config(_write_config(tmp_path, config_text))
    chip_stage = next(stage for stage in resolved.pipeline.stages if stage.id == "chip")
    chip = chip_stage.model_dump(mode="python")

    assert chip["n_modes"] == 8
    assert chip["n_bright_ports"] == 5
    assert chip["n_dark_ports"] == 2
    assert chip["n_spectro_ports"] == 1


def test_instrument_port_sum_mismatch_raises(tmp_path: Path) -> None:
    config_text = (
        _base_config()
        + """
[instrument]
n_modes = 8
n_input_modes = 4
n_bright_ports = 4
n_dark_ports = 4
n_spectro_ports = 1
"""
    )

    with pytest.raises(NullsimConfigError, match=r"\[instrument\].*sum.*n_modes"):
        load_config(_write_config(tmp_path, config_text))


def test_child_instrument_table_overrides_parent_via_extends(tmp_path: Path) -> None:
    parent = _write_config(
        tmp_path,
        _base_config()
        + """
[instrument]
n_modes = 8
n_input_modes = 4
n_bright_ports = 4
n_dark_ports = 3
n_spectro_ports = 1

[[pipeline.stages]]
id = "chip"
type = "chip.kernel_mzi_mesh"
""",
    )
    child = tmp_path / "child.toml"
    child.write_text(
        f"""
extends = "{parent.name}"

[instrument]
n_bright_ports = 5
n_dark_ports = 2
""",
        encoding="utf-8",
    )

    resolved = load_config(child)
    chip_stage = next(stage for stage in resolved.pipeline.stages if stage.id == "chip")
    chip = chip_stage.model_dump(mode="python")

    assert resolved.instrument.n_modes == 8
    assert resolved.instrument.n_input_modes == 4
    assert resolved.instrument.n_bright_ports == 5
    assert resolved.instrument.n_dark_ports == 2
    assert resolved.instrument.n_spectro_ports == 1
    assert chip["n_modes"] == 8
    assert chip["n_bright_ports"] == 5
    assert chip["n_dark_ports"] == 2
    assert chip["n_spectro_ports"] == 1


def test_load_config_rejects_duplicate_stage_ids(tmp_path: Path) -> None:
    """Ultra-review High #3 regression: two stages with the same ``id``
    must fail at config-load time. Pre-fix the duplicates loaded cleanly,
    shared RNG identity, and silently overwrote each other's diagnostics.
    """

    config_text = """
[run]
name = "dupes"

[grid]
wavelength_center_um = 1.65
wavelength_bandwidth_um = 0.1
n_wavelength_bins = 1

[scene.star]
jmag = 5.0
angular_diameter_mas = 1.0
distance_pc = 10.0

[scene.planet]
contrast = 1.0e-6
separation_mas = 10.0
position_angle_deg = 0.0

[observation]
integration_time_s = 3600.0

[[pipeline.stages]]
id = "stage_x"
type = "scene.point_source"
altitude_deg = 90.0
azimuth_deg = 0.0

[[pipeline.stages]]
id = "stage_x"
type = "scene.point_source"
altitude_deg = 90.0
azimuth_deg = 0.0
"""
    cfg_path = tmp_path / "dupes.toml"
    cfg_path.write_text(config_text)
    with pytest.raises(Exception, match="stage_x"):
        load_config(cfg_path)


def test_runtime_python_version_excludes_build_banner() -> None:
    """Code Review 2026-05-16 Issue 5 regression: runtime_python_version
    must not embed the build date / compiler banner that ``sys.version``
    carries, since that breaks cross-machine ``run_hash`` portability.
    """

    from nullsim.config.hashes import runtime_python_version

    value = runtime_python_version()
    # Must start with the implementation name + version (e.g. "CPython 3.13.0").
    # Reject the sys.version banner shape which contains " (main, " (build info).
    assert "(main, " not in value
    assert "[Clang" not in value
    assert "[GCC" not in value
    assert "[MSC" not in value


def test_load_config_accepts_timedomain_passthrough(tmp_path: Path) -> None:
    config_text = _base_config() + """
[timedomain.engine]
dt_ms = 0.1
duration_s = 5.0

[timedomain.ao]
loop_rate_hz = 1000
"""
    resolved_config = load_config(_write_config(tmp_path, config_text))
    assert resolved_config.timedomain == {
        "engine": {"dt_ms": 0.1, "duration_s": 5.0},
        "ao": {"loop_rate_hz": 1000},
    }


def test_load_config_without_timedomain_leaves_field_none(tmp_path: Path) -> None:
    resolved_config = load_config(_write_config(tmp_path, _base_config()))
    assert resolved_config.timedomain is None
    # exclude_none keeps the canonical text — and therefore config_hash —
    # identical for configs that predate the passthrough section.
    assert "timedomain" not in resolved_config.resolved_toml_text


def test_timedomain_block_folds_into_config_hash(tmp_path: Path) -> None:
    without_block = load_config(_write_config(tmp_path, _base_config()))
    with_block = load_config(
        _write_config(
            tmp_path,
            _base_config() + "\n[timedomain.engine]\ndt_ms = 0.1\n",
        )
    )
    assert without_block.config_hash != with_block.config_hash


def test_load_config_timedomain_merges_through_extends(tmp_path: Path) -> None:
    parent = _write_config(
        tmp_path,
        _base_config() + "\n[timedomain.engine]\ndt_ms = 0.1\nduration_s = 5.0\n",
    )
    child = tmp_path / "child.toml"
    child.write_text(
        f'extends = "{parent.name}"\n\n[timedomain.engine]\nduration_s = 2.0\n',
        encoding="utf-8",
    )
    resolved_config = load_config(child)
    assert resolved_config.timedomain == {
        "engine": {"dt_ms": 0.1, "duration_s": 2.0},
    }
