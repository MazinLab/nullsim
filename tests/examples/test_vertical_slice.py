"""End-to-end test for SPEC §15.1 vertical slice via the CLI.

Drives ``nullsim run examples/test/vertical_slice.toml`` through ``subprocess`` so we
exercise the same code path a user runs and verify the output directory has the
expected layout: contrast_curve.{pdf,png}, performance_summary.json,
manifest.json, sweep_results.parquet (with one row per sweep cell × metric).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = PROJECT_ROOT / "examples" / "test" / "vertical_slice.toml"
SWEEP_JMAG_VALUES: tuple[float, ...] = (4.0, 6.0, 8.0, 10.0, 12.0)


def _run_cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "nullsim", *args],
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="module")
def vertical_slice_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the vertical-slice CLI once and reuse the output across assertions."""

    output_dir = tmp_path_factory.mktemp("vertical_slice")
    completed = _run_cli(
        ["run", str(EXAMPLE_CONFIG), "--out", str(output_dir)],
        cwd=PROJECT_ROOT,
    )
    assert completed.returncode == 0, (
        f"nullsim run failed (rc={completed.returncode}):\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    return output_dir


def test_validate_subcommand_accepts_example(tmp_path: Path) -> None:
    """`nullsim validate` must succeed against the shipped example config."""

    completed = _run_cli(["validate", str(EXAMPLE_CONFIG)], cwd=PROJECT_ROOT)
    assert completed.returncode == 0, completed.stderr
    assert "config OK" in completed.stdout


def test_list_stages_subcommand_lists_trivial_stages(tmp_path: Path) -> None:
    """`nullsim list-stages` must show the five vertical-slice trivial stages."""

    completed = _run_cli(["list-stages"], cwd=PROJECT_ROOT)
    assert completed.returncode == 0, completed.stderr
    for expected_type in (
        "scene.point_source",
        "telescope.fixed_aperture",
        "injection.ideal",
        "chip.identity",
        "detector.ideal_counter",
    ):
        assert expected_type in completed.stdout


def test_run_writes_contrast_curve_plot(vertical_slice_run: Path) -> None:
    pdf_path = vertical_slice_run / "plots" / "contrast_curve.pdf"
    png_path = vertical_slice_run / "plots" / "contrast_curve.png"
    assert pdf_path.is_file() and pdf_path.stat().st_size > 0
    assert png_path.is_file() and png_path.stat().st_size > 0


def test_run_writes_performance_summary_json(vertical_slice_run: Path) -> None:
    summary_path = vertical_slice_run / "tables" / "performance_summary.json"
    assert summary_path.is_file()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert len(payload) == len(SWEEP_JMAG_VALUES)
    assert all("null_depth" in row for row in payload)


def test_run_writes_manifest_json(vertical_slice_run: Path) -> None:
    manifest_path = vertical_slice_run / "manifest.json"
    assert manifest_path.is_file()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for required_key in (
        "config_hash",
        "run_hash",
        "package_version",
        "dependency_versions",
        "source_identity",
        "host",
        "rng_root_seed",
        "timings",
        "cache",
        "external_dependencies",
    ):
        assert required_key in payload, f"manifest missing {required_key!r}"
    assert payload["rng_root_seed"] == 1
    assert "source_digest" in payload["source_identity"]
    # Five trivial stages declared in TOML, aggregated across sweep cells.
    assert len(payload["timings"]) == 5
    assert payload["cache"]["stage_count"] == 5 * len(SWEEP_JMAG_VALUES)
    # Sweep block is present because the example sweeps over jmag.
    assert "sweep" in payload
    assert payload["sweep"]["cell_count"] == len(SWEEP_JMAG_VALUES)


def test_manifest_config_hash_matches_resolved_toml_hash(
    vertical_slice_run: Path,
) -> None:
    """Finding 2 regression: the manifest's config_hash must match the
    hash of the materialized resolved TOML that was actually executed
    (and that was also folded into run_hash). Pre-fix, the CLI used the
    pre-materialized resolved config for run_hash while writing the
    materialized cell's hash into the manifest — silently inconsistent.
    """

    from nullsim.config import load_config

    manifest = json.loads((vertical_slice_run / "manifest.json").read_text("utf-8"))
    resolved_toml = vertical_slice_run / "config.resolved.toml"
    assert resolved_toml.is_file()
    executed = load_config(str(resolved_toml))
    assert manifest["config_hash"] == executed.config_hash, (
        "manifest.config_hash should equal the hash of the resolved TOML "
        "that was written next to the manifest (the config that actually ran)."
    )


def test_run_writes_sweep_results_parquet(vertical_slice_run: Path) -> None:
    parquet_path = vertical_slice_run / "sweep_results.parquet"
    assert parquet_path.is_file()
    frame = pd.read_parquet(parquet_path)

    expected_columns = {
        "sweep_coord",
        "cell_run_hash",
        "stage_instance",
        "metric_name",
        "metric_value",
        "wavelength_bin_um",
        "separation_mas",
    }
    assert expected_columns.issubset(set(frame.columns))

    # null_depth is one row per cell (no wavelength binning).
    null_rows = frame[
        (frame["stage_instance"] == "results") & (frame["metric_name"] == "null_depth")
    ]
    assert len(null_rows) == len(SWEEP_JMAG_VALUES)

    # Photon-rate spectra contribute one row per (cell × port × wavelength bin).
    rate_rows = frame[frame["metric_name"] == "rate_hz"]
    assert len(rate_rows) == len(SWEEP_JMAG_VALUES) * 2 * 3

    # cell_run_hash links every row back to a unique cell.
    assert null_rows["cell_run_hash"].nunique() == len(SWEEP_JMAG_VALUES)


def test_run_records_resolved_config_toml(vertical_slice_run: Path) -> None:
    """The resolved TOML must be persisted next to outputs for reproducibility."""

    resolved_path = vertical_slice_run / "config.resolved.toml"
    sweep_resolved_path = vertical_slice_run / "config.sweep.resolved.toml"
    assert resolved_path.is_file()
    assert sweep_resolved_path.is_file()
    text = resolved_path.read_text(encoding="utf-8")
    sweep_text = sweep_resolved_path.read_text(encoding="utf-8")
    assert "[run]" in text
    assert "vertical_slice" in text
    assert "[run]" in sweep_text
    assert "vertical_slice" in sweep_text
