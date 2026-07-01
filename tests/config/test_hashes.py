from __future__ import annotations

from pathlib import Path

from nullsim.config import compute_run_hash, load_config


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "hash.toml"
    config_path.write_text(
        """
[run]
name = "hash_test"
seed = 123

[grid]
wavelength_center_um = 1.65
wavelength_bandwidth_um = 0.3
n_wavelength_bins = 3

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
""",
        encoding="utf-8",
    )
    return config_path


def test_config_hash_is_stable_across_two_loads(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    first_config = load_config(config_path)
    second_config = load_config(config_path)
    resolved_path = tmp_path / "config.resolved.toml"
    first_config.write_toml(resolved_path)

    assert first_config.config_hash == second_config.config_hash
    assert first_config.resolved_toml_text == second_config.resolved_toml_text
    assert resolved_path.read_text(encoding="utf-8") == first_config.resolved_toml_text


def test_run_hash_changes_with_package_version_or_external_dependency() -> None:
    base_hash = compute_run_hash(
        config_hash="c" * 64,
        package_version="0.1.0",
        external_deps_hashes_iterable=("dep-a",),
        runtime_python_version="python-test",
        key_dep_versions={"numpy": "1", "scipy": "2", "pyarrow": "3"},
        source_digest="source-a",
    )
    version_hash = compute_run_hash(
        config_hash="c" * 64,
        package_version="0.2.0",
        external_deps_hashes_iterable=("dep-a",),
        runtime_python_version="python-test",
        key_dep_versions={"numpy": "1", "scipy": "2", "pyarrow": "3"},
        source_digest="source-a",
    )
    dependency_hash = compute_run_hash(
        config_hash="c" * 64,
        package_version="0.1.0",
        external_deps_hashes_iterable=("dep-b",),
        runtime_python_version="python-test",
        key_dep_versions={"numpy": "1", "scipy": "2", "pyarrow": "3"},
        source_digest="source-a",
    )
    source_hash = compute_run_hash(
        config_hash="c" * 64,
        package_version="0.1.0",
        external_deps_hashes_iterable=("dep-a",),
        runtime_python_version="python-test",
        key_dep_versions={"numpy": "1", "scipy": "2", "pyarrow": "3"},
        source_digest="source-b",
    )

    assert version_hash != base_hash
    assert dependency_hash != base_hash
    assert source_hash != base_hash
