from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

import nullsim  # noqa: F401  # stage registration side effects
from nullsim.config.loader import resolve_config
from nullsim.outputs import RunCollection, RunRecord, _autoload_outputs
from nullsim.outputs.paper import (
    _OutputProduct,
    _characterization_rates_for_planet,
    _copy_registered_product,
    _find_component,
    _latex_characterization_table,
    _latex_exoplanet_table,
    _pdi_snr,
    _stellar_rate_hz,
    write_paper_outputs,
)
from nullsim.pipeline import Field, Geometry, Results, RunResult, SimulationState
from nullsim.pipeline import registry


def test_write_paper_outputs_creates_bundle_manifest_and_core_figures(
    tmp_path: Path,
) -> None:
    _autoload_outputs()
    config = _config()
    collection = _collection(config, _state(), tmp_path)

    written = write_paper_outputs(collection, tmp_path)

    paper_dir = tmp_path / "paper"
    expected = {
        "signal_chain.pdf",
        "array_layout.pdf",
        "baseline_histogram.pdf",
        "cophasing_scheme.pdf",
        "performance_summary.json",
        "performance_summary.csv",
        "paper_manifest.json",
    }
    assert expected.issubset({path.name for path in written})
    for name in expected:
        path = paper_dir / name
        assert path.exists()
        assert path.stat().st_size > 100

    manifest = json.loads((paper_dir / "paper_manifest.json").read_text())
    generated = {entry["file"] for entry in manifest["generated"]}
    out_of_scope = {entry["file"] for entry in manifest["out_of_scope"]}
    assert "null_transmission_map.pdf" in generated
    assert "exoplanet_table.tex" in generated
    assert "characterization_table.tex" in generated
    assert "pdi_exoplanet_table.tex" in generated
    assert "natural_detection_space.pdf" in out_of_scope
    char_tex = (paper_dir / "characterization_table.tex").read_text(
        encoding="utf-8"
    )
    assert r"\label{tab:characterization_survey}" in char_tex
    assert r"$R_{\rm SNR \geq 3}$" in char_tex


def test_latex_tables_use_paper_prose_class_labels() -> None:
    rows = {
        "Neptune-like": [
            {
                "name": "GJ 338 B b",
                "category": "Neptune-like",
                "host_jmag": 4.78,
                "ang_sep_mas": 22.4,
                "reflected_contrast": 2.0e-8,
                "snr_det_4h": 13.5,
                "char_snr_4h": 41.5,
                "char_max_R": 797.0,
            }
        ],
        "Super-Earth": [
            {
                "name": "HD 219134 b",
                "category": "Super-Earth",
                "host_jmag": 4.0,
                "ang_sep_mas": 5.9,
                "reflected_contrast": 3.5e-7,
                "snr_det_4h": 4.6,
                "char_snr_4h": 13.3,
                "char_max_R": 82.0,
            }
        ],
    }

    for table in (_latex_exoplanet_table(rows), _latex_characterization_table(rows)):
        assert "Ice Giant" in table
        assert "Super Earth" in table
        assert "Neptune-like" not in table
        assert "Super-Earth" not in table


def test_find_component_prefers_exact_diagnostic_name(tmp_path: Path) -> None:
    config = _config()
    state = _state()
    full_payload = {"rank": 2}
    state = replace(
        state,
        results=Results(
            components={
                "fringe_observability_pre_sensitivity": {"rank": 0},
                "fringe_observability": full_payload,
            },
        ),
    )
    collection = _collection(config, state, tmp_path)
    record = collection.records[0]

    assert _find_component(record, "fringe_observability") == full_payload


def test_paper_registered_product_regenerates_instead_of_copying_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale_dir = tmp_path / "tables"
    stale_dir.mkdir()
    (stale_dir / "fresh.json").write_text("stale", encoding="utf-8")

    def writer(
        _collection: RunCollection,
        output_dir: Path,
        formats: tuple[str, ...],
    ) -> tuple[Path, ...]:
        table_dir = output_dir / "tables"
        table_dir.mkdir(parents=True, exist_ok=True)
        path = table_dir / f"fresh.{formats[0]}"
        path.write_text("fresh", encoding="utf-8")
        return (path,)

    monkeypatch.setitem(registry._tables, "paper_stale_regression", writer)
    manifest: dict[str, object] = {"generated": [], "skipped": []}
    written: list[Path] = []
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()

    _copy_registered_product(
        product=_OutputProduct(
            "paper_stale_regression",
            "fresh",
            "fresh",
            ("json",),
        ),
        kind="table",
        collection=_collection(_config(), _state(), tmp_path),
        output_dir=tmp_path,
        paper_dir=paper_dir,
        manifest=manifest,
        written=written,
    )

    assert (paper_dir / "fresh.json").read_text(encoding="utf-8") == "fresh"
    assert written == [paper_dir / "fresh.json"]


def test_paper_stellar_rate_prefers_total_performance_rate(tmp_path: Path) -> None:
    state = _state()
    state = replace(
        state,
        results=state.results.add_component(
            "performance.standard",
            {"stellar_rate_hz_total": 123.0},
        ),
    )
    record = _collection(_config(), state, tmp_path).records[0]
    assert _stellar_rate_hz(record) == pytest.approx(123.0)


def test_paper_characterization_rates_are_recomputed_for_planet(
    tmp_path: Path,
) -> None:
    record = _collection(_config(), _state(), tmp_path).records[0]
    char = {
        "detected_stellar_rate_hz": 1.0e6,
        "spectro_throughput": 0.25,
        "spectro_planet_to_star_ratio_per_wavelength": np.array([0.5]),
        "wavelengths_m": (1.0e-6,),
        "band_lambda_m": 1.0e-6,
        "spectro_band_leak_fraction": 1.0e-5,
        "integration_time_s": 10.0,
        "D": 0.0,
        "null_instability": 0.0,
        "servo_bw_hz": 0.0,
        "spectro_indices": np.array([1], dtype=np.intp),
    }

    faint = _characterization_rates_for_planet(
        record=record,
        char=char,
        char_curve=None,
        planet_contrast=1.0e-7,
        planet_separation_mas=10.0,
        fallback_planet_rate_hz=0.0,
        fallback_leakage_rate_hz=0.0,
    )
    bright = _characterization_rates_for_planet(
        record=record,
        char=char,
        char_curve=None,
        planet_contrast=2.0e-7,
        planet_separation_mas=10.0,
        fallback_planet_rate_hz=0.0,
        fallback_leakage_rate_hz=0.0,
    )

    assert faint[0] == pytest.approx(0.0125)
    assert bright[0] == pytest.approx(0.025)
    assert bright[1] == pytest.approx(faint[1])
    assert bright[2] > faint[2]


def test_paper_characterization_rates_use_separation_curve_when_available(
    tmp_path: Path,
) -> None:
    record = _collection(_config(), _state(), tmp_path).records[0]
    char = {
        "detected_stellar_rate_hz": 1.0e6,
        "spectro_throughput": 99.0,
        "spectro_band_leak_fraction": 9.0e-2,
        "integration_time_s": 10.0,
        "D": 0.0,
        "null_instability": 0.0,
        "servo_bw_hz": 0.0,
        "spectro_indices": np.array([1], dtype=np.intp),
    }
    char_curve = {
        "detected_stellar_rate_hz": 1.0e6,
        "separations_mas": np.array([10.0, 20.0]),
        "spectro_throughput_per_anchor": np.array([0.10, 0.40]),
        "spectro_leak_fraction_per_anchor": np.array([1.0e-5, 2.0e-5]),
        "optimization_success_per_anchor": np.array([True, True]),
        "optimization_constraint_margin_per_anchor": np.array([1.0e-6, 1.0e-6]),
        "full_band_constraint_margin_per_anchor": np.array([1.0e-6, 1.0e-6]),
        "enforce_null_constraint": True,
        "null_floor": 1.0e-6,
        "integration_time_s": 10.0,
        "sky_rate_hz_per_spectro_port": 0.0,
        "null_instability": 0.0,
        "servo_bw_hz": 0.0,
    }

    planet, leakage, snr, _max_r = _characterization_rates_for_planet(
        record=record,
        char=char,
        char_curve=char_curve,
        planet_contrast=1.0e-7,
        planet_separation_mas=15.0,
        fallback_planet_rate_hz=0.0,
        fallback_leakage_rate_hz=0.0,
    )

    assert planet == pytest.approx(0.025)
    assert leakage == pytest.approx(15.0)
    assert snr > 0.0


def test_paper_characterization_rates_mask_infeasible_curve_anchor(
    tmp_path: Path,
) -> None:
    record = _collection(_config(), _state(), tmp_path).records[0]
    char_curve = {
        "detected_stellar_rate_hz": 1.0e6,
        "separations_mas": np.array([10.0, 20.0]),
        "spectro_throughput_per_anchor": np.array([0.10, 0.40]),
        "spectro_leak_fraction_per_anchor": np.array([1.0e-5, 2.0e-5]),
        "optimization_success_per_anchor": np.array([True, True]),
        "optimization_constraint_margin_per_anchor": np.array([1.0e-6, 1.0e-6]),
        "full_band_constraint_margin_per_anchor": np.array([-1.0e-4, -1.0e-4]),
        "enforce_null_constraint": True,
        "null_floor": 1.0e-6,
    }

    planet, leakage, snr, _max_r = _characterization_rates_for_planet(
        record=record,
        char=None,
        char_curve=char_curve,
        planet_contrast=1.0e-7,
        planet_separation_mas=15.0,
        fallback_planet_rate_hz=1.0,
        fallback_leakage_rate_hz=2.0,
    )

    assert np.isnan(planet)
    assert np.isnan(leakage)
    assert np.isnan(snr)


def test_pdi_snr_uses_inferred_dark_port_count(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, int] = {}

    def fake_signal_to_noise_pdi(**kwargs: object) -> float:
        captured["n_dark_ports_per_arm"] = int(kwargs["n_dark_ports_per_arm"])
        return 9.0

    import nullsim.physics.pdi as pdi

    monkeypatch.setattr(pdi, "signal_to_noise_pdi", fake_signal_to_noise_pdi)
    snr = _pdi_snr(
        {
            "planet_signal_rate_dark": 1.0,
            "leakage_rate_dark": 2.0,
            "pol_fraction_orbit_avg": 0.3,
        },
        10.0,
        modulation_period_s=60.0,
        n_dark_ports=11,
    )
    assert snr == pytest.approx(9.0)
    assert captured["n_dark_ports_per_arm"] == 11


def test_pdi_displayed_snr_sorts_finite_first_and_sinks_nan() -> None:
    """The PDI survey table sorts by its displayed SNR with NaN/missing last.

    Regression: the table displays ``snr_pdi_continuous_4h`` but was sorted by
    ``snr_char_pdi_continuous_4h``, and NaN sort keys disordered finite rows.
    """

    from nullsim.outputs.paper import _pdi_displayed_snr

    rows = [
        {"name": "a", "snr_pdi_continuous_4h": 0.9, "snr_char_pdi_continuous_4h": 9.0},
        {
            "name": "b",
            "snr_pdi_continuous_4h": float("nan"),
            "snr_char_pdi_continuous_4h": 5.0,
        },
        {"name": "c", "snr_pdi_continuous_4h": 2.1, "snr_char_pdi_continuous_4h": 0.5},
        {"name": "d"},  # missing the displayed metric entirely
    ]

    ordered = sorted(rows, key=_pdi_displayed_snr, reverse=True)
    assert [row["name"] for row in ordered] == ["c", "a", "b", "d"]
    assert _pdi_displayed_snr(rows[1]) == float("-inf")
    assert _pdi_displayed_snr(rows[3]) == float("-inf")


def test_characterization_rates_skipped_curve_cell_yields_nan_snr() -> None:
    """When separation-resolved curves are produced for the run but skipped for
    this cell, the scalar single-scene retune must not supply a per-planet
    characterization SNR (regression for skip_jmags cells)."""

    from nullsim.outputs.paper import _characterization_rates_for_planet

    planet_rate, leak_rate, snr, _max_r = _characterization_rates_for_planet(
        record=None,
        char={"snr": 5.0, "spectro_throughput": 0.3, "optimization_success": True},
        char_curve=None,
        planet_contrast=1.0e-9,
        planet_separation_mas=10.0,
        fallback_planet_rate_hz=2.0,
        fallback_leakage_rate_hz=3.0,
        curve_expected=True,
    )
    assert planet_rate == 2.0
    assert leak_rate == 3.0
    assert np.isnan(snr)

    # No curves anywhere in the run: char=None falls back with SNR 0.0.
    pr, lr, snr0, _max_r0 = _characterization_rates_for_planet(
        record=None,
        char=None,
        char_curve=None,
        planet_contrast=1.0e-9,
        planet_separation_mas=10.0,
        fallback_planet_rate_hz=2.0,
        fallback_leakage_rate_hz=3.0,
    )
    assert (pr, lr, snr0) == (2.0, 3.0, 0.0)


def test_assumed_star_disk_mas_uses_config_star() -> None:
    """The disk column reports the simulated reference star (the catalog has no
    per-host stellar radius), not a per-host angular diameter."""

    from types import SimpleNamespace

    from nullsim.outputs.paper import _assumed_star_disk_mas

    collection = SimpleNamespace(
        config=SimpleNamespace(
            scene=SimpleNamespace(star=SimpleNamespace(angular_diameter_mas=1.5))
        )
    )
    assert _assumed_star_disk_mas(collection) == 1.5
    missing = SimpleNamespace(config=SimpleNamespace(scene=SimpleNamespace(star=None)))
    assert _assumed_star_disk_mas(missing) == 0.0


def _config():
    raw_config = {
        "run": {"name": "paper_test", "seed": 1},
        "grid": {
            "wavelength_center_um": 1.0,
            "wavelength_bandwidth_um": 0.1,
            "n_wavelength_bins": 3,
        },
        "scene": {
            "star": {
                "jmag": 5.0,
                "angular_diameter_mas": 1.0,
                "distance_pc": 5.0,
            },
            "planet": {
                "contrast": 1.0e-6,
                "separation_mas": 20.0,
                "position_angle_deg": 90.0,
            },
        },
        "observation": {"integration_time_s": 10.0},
        "pipeline": {"stages": [{"id": "scene", "type": "scene.point_source"}]},
        "output": {
            "paper": True,
            "plots": ["null_transmission_map"],
            "tables": ["performance_summary"],
            "formats": {"plots": ["pdf"], "tables": ["json"]},
            "null_transmission_map": {
                "n_pix": 17,
                "fov_mas": 40.0,
                "log_scale": True,
                "log_floor": 1.0e-12,
                "save_arrays": True,
            },
        },
    }
    return resolve_config(raw_config)


def _collection(config, state: SimulationState, tmp_path: Path) -> RunCollection:
    result = RunResult(
        state=state,
        diagnostics={},
        timings=(),
        cache_hit_count=0,
        config_hash="cfg",
    )
    return RunCollection(
        config=config,
        records=(
            RunRecord(
                config=config,
                result=result,
                run_dir=tmp_path / "cell",
                coord_tuple=(),
            ),
        ),
        output_dir=tmp_path,
    )


def _state() -> SimulationState:
    wavelength_grid_um = (0.95, 1.0, 1.05)
    state = SimulationState.empty({"wavelength_grid_um": wavelength_grid_um})
    positions_pupil_m = np.array([[-5.0, 0.0], [5.0, 0.0]], dtype=np.float64)
    positions_enu_m = np.array([[-42.0, 0.0, 0.0], [42.0, 0.0, 0.0]], dtype=np.float64)
    field = Field(
        amplitudes=np.ones((2, 3, 1), dtype=np.complex128),
        wavelength_grid_um=wavelength_grid_um,
    )
    unitary = np.array(
        [[1.0, 1.0], [1.0, -1.0]],
        dtype=np.complex128,
    ) / np.sqrt(2.0)
    u_stack = np.stack([unitary, unitary, unitary], axis=2)
    results = Results(null_depth=1.0e-5, snr=7.0, calibration_time_s=42.0)
    results = results.add_component(
        "chip.kernel_mzi_mesh",
        {
            "u_stack": u_stack,
            "wavelength_grid_um": wavelength_grid_um,
            "wl_design_um": 1.0,
            "n_modes": 2,
            "n_input_modes": 2,
            "n_bright_ports": 1,
            "n_dark_ports": 1,
            "n_spectro_ports": 0,
        },
    )
    results = results.add_component(
        "sensitivity.detection_curve",
        {
            "coupling_per_sub": np.ones(2, dtype=np.float64),
            "sub_areas_m2": np.ones(2, dtype=np.float64),
            "sub_diameters_m": np.ones(2, dtype=np.float64),
            "dark_indices": np.array([1], dtype=np.intp),
        },
    )
    results = results.add_component(
        "fringe_observability",
        {
            "n_bright_ports": 1,
            "n_physical_piston_modes": 2,
            "n_differential_piston_modes": 1,
            "rank": 1,
            "condition_number": 10.0,
            "crlb_piston_rms_nm": 1.2,
            "gauss_newton_monte_carlo": {"median_residual_rms_nm": 2.3},
            "cophasing_handoff_monte_carlo": {
                "median_science_residual_rms_nm": 3.4,
                "science_lock_fraction": 0.95,
                "edge_wrap_fraction": 0.01,
            },
        },
    )
    geometry = Geometry(
        diameter_m=1.0,
        area_m2=1.0,
        collecting_area_m2=2.0,
        effective_collecting_area_m2=2.0,
        n_apertures=2,
        aperture_positions_enu_m=positions_enu_m,
        aperture_positions_pupil_m=positions_pupil_m,
    ).add_component(
        "telescope.fixed_aperture",
        {"aperture_positions_enu_m": positions_enu_m},
    )
    return replace(
        state,
        geometry=geometry,
        field=field,
        results=results,
    )
