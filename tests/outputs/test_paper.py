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


def test_find_component_returns_newest_matching_component(tmp_path: Path) -> None:
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

    # Newest first with no exact-name preference: a later suffixed instance
    # supersedes an earlier exact-named one (review 2026-11-07 Codex item 5).
    state = replace(
        state,
        results=Results(
            components={
                "chip.kernel_mzi_mesh": {"rank": 0},
                "chip.kernel_mzi_mesh.deployed": {"rank": 1},
            },
        ),
    )
    record = _collection(config, state, tmp_path).records[0]
    assert _find_component(record, "chip.kernel_mzi_mesh") == {"rank": 1}


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
    captured: dict[str, object] = {}

    def fake_signal_to_noise_pdi(**kwargs: object) -> float:
        captured.update(kwargs)
        return 9.0

    import nullsim.physics.pdi as pdi

    monkeypatch.setattr(pdi, "signal_to_noise_pdi", fake_signal_to_noise_pdi)
    snr = _pdi_snr(
        {
            "planet_signal_rate_dark": 1.0,
            "leakage_rate_dark": 2.0,
            "pol_fraction_orbit_avg": 0.3,
            "sky_rate_per_dark_port": 2.0,
            "dark_rate_per_dark_port": 0.5,
        },
        10.0,
        modulation_period_s=60.0,
        n_dark_ports=11,
        rate_scale=1.0,
    )
    assert snr == pytest.approx(9.0)
    assert captured["n_dark_ports_per_arm"] == 11
    # Review 2026-11-07 row 58: the survey row's sky and dark reach the PDI.
    assert captured["sky_rate_hz_per_dark_port_per_arm"] == pytest.approx(2.0)
    assert captured["dark_rate_hz_per_port_per_arm"] == pytest.approx(0.5)


def test_pdi_dual_pol_factor_only_for_scalar_pipelines_and_only_once(
    tmp_path: Path,
) -> None:
    """Review 2026-11-07 Issue 15: the 0.87 scalar to dual polarization factor
    applies only without polarization stages, and the char fallback (no char
    rates) must not receive it a second time."""

    from nullsim.outputs.paper import _pdi_row

    row = {
        "name": "p",
        "radius_rearth": 10.0,
        "planet_signal_rate_dark": 4.0,
        "leakage_rate_dark": 100.0,
    }
    scalar = _collection(
        _config(), _state(), tmp_path
    )
    dual = _collection(
        _config(extra_stage_types=("polarization.pbs_split",)),
        _state(),
        tmp_path,
    )

    scalar_row = _pdi_row(row, scalar)
    dual_row = _pdi_row(row, dual)

    assert scalar_row["char_planet_signal_rate_pdi"] == pytest.approx(4.0 * 0.87)
    assert scalar_row["char_leakage_rate_pdi"] == pytest.approx(100.0 * 0.87)
    assert dual_row["char_planet_signal_rate_pdi"] == pytest.approx(4.0)
    assert dual_row["char_leakage_rate_pdi"] == pytest.approx(100.0)


def test_interp_gapped_never_bridges_or_clamps() -> None:
    """Review 2026-11-07 row 59: failed anchors are gaps, a lone anchor is
    valid only at its own separation, and outside the grid is NaN."""

    from nullsim.outputs.paper import _interp_gapped

    xs = np.array([30.0, 10.0, 20.0, 40.0])
    ys = np.array([3.0, 1.0, 2.0, 4.0])
    valid = np.array([False, True, True, True])

    assert _interp_gapped(xs, ys, valid, 15.0) == pytest.approx(1.5)
    assert _interp_gapped(xs, ys, valid, 20.0) == pytest.approx(2.0)
    assert np.isnan(_interp_gapped(xs, ys, valid, 25.0))  # touches failed 30
    assert np.isnan(_interp_gapped(xs, ys, valid, 30.0))
    assert np.isnan(_interp_gapped(xs, ys, valid, 5.0))
    assert np.isnan(_interp_gapped(xs, ys, valid, 45.0))

    lone = np.array([False, True, False, False])
    assert _interp_gapped(xs, ys, lone, 10.0) == pytest.approx(1.0)
    assert np.isnan(_interp_gapped(xs, ys, lone, 12.0))


def test_contrast_limit_treats_infinite_contrast_as_gap(tmp_path: Path) -> None:
    from nullsim.outputs.paper import _contrast_limit_at
    from nullsim.stages.detector.ideal_counter import CONTRAST_CURVE_DTYPE

    curve = np.empty(3, dtype=CONTRAST_CURVE_DTYPE)
    curve["separation_mas"] = (10.0, 20.0, 40.0)
    curve["contrast"] = (1.0e-6, np.inf, 1.0e-7)
    state = replace(_state(), results=replace(_state().results, contrast_curve=curve))
    record = _collection(_config(), state, tmp_path).records[0]

    assert np.isnan(_contrast_limit_at(record, 15.0))
    assert np.isnan(_contrast_limit_at(record, 50.0))
    assert _contrast_limit_at(record, 40.0) == pytest.approx(1.0e-7)


def test_char_curve_single_surviving_anchor_is_valid_only_at_its_separation(
    tmp_path: Path,
) -> None:
    record = _collection(_config(), _state(), tmp_path).records[0]
    char_curve = {
        "detected_stellar_rate_hz": 1.0e6,
        "separations_mas": np.array([10.0, 20.0, 40.0]),
        "spectro_throughput_per_anchor": np.array([0.1, 0.2, 0.4]),
        "spectro_leak_fraction_per_anchor": np.array([1.0e-5, 2.0e-5, 4.0e-5]),
        "optimization_success_per_anchor": np.array([False, True, False]),
    }
    kwargs = dict(
        record=record,
        char=None,
        char_curve=char_curve,
        planet_contrast=1.0e-7,
        fallback_planet_rate_hz=1.0,
        fallback_leakage_rate_hz=2.0,
    )

    at_anchor = _characterization_rates_for_planet(planet_separation_mas=20.0, **kwargs)
    off_anchor = _characterization_rates_for_planet(planet_separation_mas=30.0, **kwargs)

    assert at_anchor[0] == pytest.approx(1.0e6 * 1.0e-7 * 0.2)
    assert at_anchor[1] == pytest.approx(1.0e6 * 2.0e-5)
    assert np.isnan(off_anchor[0]) and np.isnan(off_anchor[1])


def test_char_curve_rescaled_to_cell_moves_floor_eps_and_bandwidth() -> None:
    """Review 2026-11-07 Issue 14: a borrowed curve keeps the donor chip's
    throughput and static leak but takes the floor, eps, and bandwidth from
    the host cell's retuned ``characterization.spectro``.

    Hand values: spectro floors 3e-6 (donor) and 1.2e-5 (host) scale the
    per-anchor floor 4x, so leak = static + 4 floor = [8.1e-6, 16.2e-6].
    Bandwidth comes from the host spectro. Without the leak RMS keys there is
    no anchor-wise decomposition, so the host's scalar eps replaces the
    curve's and the donor's per-anchor eps must not survive.
    """

    from nullsim.outputs.paper import _char_curve_rescaled_to_cell

    donor_curve = {
        "spectro_throughput_per_anchor": np.array([0.3, 0.5]),
        "spectro_static_leak_fraction_per_anchor": np.array([1.0e-7, 2.0e-7]),
        "realized_spectro_floor_per_anchor": np.array([2.0e-6, 4.0e-6]),
        "spectro_leak_fraction_per_anchor": np.array([2.1e-6, 4.2e-6]),
        "realized_spectro_floor": 3.0e-6,
        "eps_systematic": 0.05,
        "eps_systematic_per_anchor": np.array([0.04, 0.06]),
        "servo_bw_hz": 3784.0,
    }
    donor_spectro = {"realized_spectro_floor": 3.0e-6, "eps_systematic": 0.05}
    cell_spectro = {
        "realized_spectro_floor": 1.2e-5,
        "eps_systematic": 0.098,
        "servo_bw_hz": 0.0,
        "with_fringe_tracking": False,
    }

    curve = _char_curve_rescaled_to_cell(
        donor_curve, donor_spectro=donor_spectro, cell_spectro=cell_spectro
    )

    np.testing.assert_allclose(curve["spectro_throughput_per_anchor"], [0.3, 0.5])
    np.testing.assert_allclose(
        curve["spectro_leak_fraction_per_anchor"], [8.1e-6, 16.2e-6]
    )
    assert curve["realized_spectro_floor"] == pytest.approx(1.2e-5)
    assert curve["eps_systematic"] == pytest.approx(0.098)
    assert "eps_systematic_per_anchor" not in curve
    assert curve["servo_bw_hz"] == 0.0
    assert curve["with_fringe_tracking"] is False
    # The donor payload is left untouched.
    np.testing.assert_allclose(
        donor_curve["spectro_leak_fraction_per_anchor"], [2.1e-6, 4.2e-6]
    )


def _eps_probe_curve() -> dict:
    """Reviewer's probe: anchor leakages [1e-6, 1e-7], eps [0.01, 0.1].

    nsc = 2, null instability [0.016, 0.16] and speckle CV [0.006, 0.06]:
    hypot(0.008, 0.006) = 0.01 and hypot(0.08, 0.06) = 0.1, so
    eps x leakage = [1e-8, 1e-8].
    """

    return {
        "spectro_static_leak_fraction_per_anchor": np.array([5.0e-7, 5.0e-8]),
        "realized_spectro_floor_per_anchor": np.array([5.0e-7, 5.0e-8]),
        "spectro_leak_fraction_per_anchor": np.array([1.0e-6, 1.0e-7]),
        "null_instability_per_anchor": np.array([0.016, 0.16]),
        "eps_systematic_per_anchor": np.array([0.01, 0.1]),
        "eps_systematic": 0.055,
        "nsc_systematic_reduction": 2.0,
    }


def test_char_curve_rescale_keeps_eps_times_leakage_for_unchanged_budgets() -> None:
    """Review 2026-11-07 Codex item 1: eps is a CV against each anchor's own
    leakage. With donor and host budgets equal, eps x leakage must stay
    [1e-8, 1e-8]; copying one scene eps to every anchor made it
    [1e-7, 1e-8]."""

    from nullsim.outputs.paper import _char_curve_rescaled_to_cell

    spectro = {
        "realized_spectro_floor": 2.0e-6,
        "spectro_leak_rms_fraction": 3.0e-8,
        "eps_systematic": 0.03,
    }
    curve = _char_curve_rescaled_to_cell(
        _eps_probe_curve(), donor_spectro=spectro, cell_spectro=dict(spectro)
    )

    np.testing.assert_allclose(
        curve["eps_systematic_per_anchor"] * curve["spectro_leak_fraction_per_anchor"],
        [1.0e-8, 1.0e-8],
    )


def test_char_curve_rescale_scales_fluctuation_amplitude_not_cv() -> None:
    """Floor ratio 3 and leak RMS ratio 1, by hand: new leakage
    static + 3 floor = [2e-6, 2e-7]; dynamic RMS ni x old leak = [1.6e-8,
    1.6e-8] unchanged, so ni = [8e-3, 0.08] and eps = hypot(ni / 2, speckle)
    = hypot(0.004, 0.006) = 0.0072111 and hypot(0.04, 0.06) = 0.072111.
    A leak RMS ratio of 2 instead doubles the dynamic term at fixed floor:
    hypot(0.016, 0.006) = 0.017088 and hypot(0.16, 0.06) = 0.17088."""

    from nullsim.outputs.paper import _char_curve_rescaled_to_cell

    donor = {"realized_spectro_floor": 1.0e-6, "spectro_leak_rms_fraction": 1.0e-8}
    curve = _char_curve_rescaled_to_cell(
        _eps_probe_curve(),
        donor_spectro=donor,
        cell_spectro={"realized_spectro_floor": 3.0e-6, "spectro_leak_rms_fraction": 1.0e-8},
    )
    np.testing.assert_allclose(curve["spectro_leak_fraction_per_anchor"], [2.0e-6, 2.0e-7])
    np.testing.assert_allclose(
        curve["eps_systematic_per_anchor"], [0.0072111, 0.072111], rtol=1.0e-4
    )

    curve = _char_curve_rescaled_to_cell(
        _eps_probe_curve(),
        donor_spectro=donor,
        cell_spectro={"realized_spectro_floor": 1.0e-6, "spectro_leak_rms_fraction": 2.0e-8},
    )
    np.testing.assert_allclose(
        curve["eps_systematic_per_anchor"], [0.017088, 0.17088], rtol=1.0e-4
    )
    assert curve["eps_systematic"] == pytest.approx(
        float(np.median(curve["eps_systematic_per_anchor"]))
    )


def test_char_curve_rescale_survives_zero_donor_floor() -> None:
    """Review 2026-11-07 Codex item 2: a noiseless donor (floor 0) must not
    divide by zero. Both zero: unchanged. Host 4e-6: every anchor takes the
    host's scene floor and the curve is flagged."""

    from nullsim.outputs.paper import _char_curve_rescaled_to_cell

    zero = {"realized_spectro_floor": 0.0}
    curve = _char_curve_rescaled_to_cell(
        _eps_probe_curve(), donor_spectro=zero, cell_spectro=dict(zero)
    )
    np.testing.assert_allclose(curve["spectro_leak_fraction_per_anchor"], [1.0e-6, 1.0e-7])
    assert "floor_from_host_scene" not in curve

    curve = _char_curve_rescaled_to_cell(
        _eps_probe_curve(),
        donor_spectro=zero,
        cell_spectro={"realized_spectro_floor": 4.0e-6},
    )
    np.testing.assert_allclose(curve["realized_spectro_floor_per_anchor"], [4.0e-6, 4.0e-6])
    np.testing.assert_allclose(curve["spectro_leak_fraction_per_anchor"], [4.5e-6, 4.05e-6])
    assert curve["floor_from_host_scene"] is True


def test_sky_rate_prefers_background_total_over_sky(tmp_path: Path) -> None:
    """Review 2026-11-07 Codex item 4: the PDI background is sky plus thermal
    and exozodi: the rate performance.standard charged, else a background
    stage's background_rate_hz_per_dark_port before its sky."""

    from nullsim.outputs.paper import _sky_rate_per_dark_port

    background = {"sky_rate_hz_per_dark_port": 2.0, "background_rate_hz_per_dark_port": 5.0}
    state = replace(
        _state(),
        results=Results().add_component("background.space_zodi", background),
    )
    assert _sky_rate_per_dark_port(_collection(_config(), state, tmp_path).records[0]) == 5.0

    state = replace(
        state,
        results=state.results.add_component(
            "performance.standard", {"sky_rate_hz_per_port": 7.0}
        ),
    )
    assert _sky_rate_per_dark_port(_collection(_config(), state, tmp_path).records[0]) == 7.0


def test_char_curve_eps_is_interpolated_per_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The curve's per-anchor eps (each anchor's retuned-port leak CV) is used
    at the planet's separation: 0.02 and 0.06 at 10 and 20 mas give 0.04 at
    15 mas, not the 0.5 scalar median."""

    import nullsim.stages.performance.standard as standard

    captured: dict[str, float] = {}
    real = standard._compute_snr

    def spy(**kwargs: object) -> dict:
        captured["eps"] = float(kwargs["eps"])
        return real(**kwargs)

    monkeypatch.setattr(standard, "_compute_snr", spy)
    record = _collection(_config(), _state(), tmp_path).records[0]
    char_curve = {
        "detected_stellar_rate_hz": 1.0e6,
        "separations_mas": np.array([10.0, 20.0]),
        "spectro_throughput_per_anchor": np.array([0.5, 0.5]),
        "spectro_leak_fraction_per_anchor": np.array([1.0e-3, 1.0e-3]),
        "eps_systematic": 0.5,
        "eps_systematic_per_anchor": np.array([0.02, 0.06]),
        "servo_bw_hz": 0.0,
    }
    kwargs = dict(
        record=record,
        char=None,
        planet_contrast=1.0e-4,
        fallback_planet_rate_hz=0.0,
        fallback_leakage_rate_hz=0.0,
    )

    _characterization_rates_for_planet(
        char_curve=char_curve, planet_separation_mas=15.0, **kwargs
    )
    assert captured["eps"] == pytest.approx(0.04)

    scalar_only = {k: v for k, v in char_curve.items() if k != "eps_systematic_per_anchor"}
    _characterization_rates_for_planet(
        char_curve=scalar_only, planet_separation_mas=15.0, **kwargs
    )
    assert captured["eps"] == pytest.approx(0.5)


def test_char_max_r_charges_awg_throughput(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review 2026-11-07 row 31 and Codex item 3: the paper's max R repeats
    characterization.spectro's call. AWG: its throughput, detector pixels per
    port n_detector_pixels_per_bin / n_ports = 6 / 3 = 2, its pixels per
    resolution element, sky 2 Hz per port x 3 ports = 6 Hz, and the stage's
    snr_min and cap. MKID: throughput 1, mkid_n_spectro_pixels, 1 pixel per
    resolution element."""

    import nullsim.physics.spectroscopy as spectroscopy
    from nullsim.outputs.paper import _awg_throughput

    seen: list[dict] = []

    def fake_max_r(**kwargs: object) -> float:
        seen.append(kwargs)
        return 100.0

    monkeypatch.setattr(spectroscopy, "max_spectral_resolution", fake_max_r)
    record = _collection(_config(), _state(), tmp_path).records[0]
    base = {
        "detected_stellar_rate_hz": 1.0e6,
        "spectro_throughput": 0.25,
        "spectro_band_leak_fraction": 1.0e-5,
        "D": 0.0,
        "servo_bw_hz": 0.0,
        "spectro_indices": np.array([1, 2, 3], dtype=np.intp),
        "sky_rate_hz_per_spectro_port": 2.0,
        "snr_min_per_bin": 5.0,
        "resolving_power_cap": 5000.0,
    }
    awg = {"throughput": 0.7, "n_detector_pixels_per_bin": 6, "pixels_per_resolution_element": 3}
    for char in (
        {**base, "spectrograph": "awg", "awg": awg},
        {**base, "spectrograph": "mkid", "mkid_n_spectro_pixels": 4},
    ):
        _characterization_rates_for_planet(
            record=record,
            char=char,
            planet_contrast=1.0e-7,
            planet_separation_mas=10.0,
            fallback_planet_rate_hz=0.0,
            fallback_leakage_rate_hz=0.0,
        )

    awg_call, mkid_call = seen
    assert awg_call["spectro_throughput"] == pytest.approx(0.7)
    assert awg_call["n_spectro_pixels"] == 2
    assert awg_call["pixels_per_resolution_element"] == 3
    assert awg_call["sky_rate_hz_per_spectro_port"] == pytest.approx(6.0)
    assert awg_call["snr_min"] == pytest.approx(5.0)
    assert awg_call["resolving_power_cap"] == pytest.approx(5000.0)
    assert mkid_call["spectro_throughput"] == pytest.approx(1.0)
    assert mkid_call["n_spectro_pixels"] == 4
    assert mkid_call["pixels_per_resolution_element"] == 1
    assert _awg_throughput(None) == 1.0


def test_char_pdi_charges_spectro_port_background(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review 2026-11-07 row 58: the characterization PDI charges the spectro
    ports' sky and dark with the spectro port count. The primitive sums
    per-arm rates over two arms, so the arm-total sky 4 Hz and dark 2 Hz per
    port enter as 2 and 1 Hz per arm on 3 ports (totals 12 and 6 Hz, the
    characterization stage's Sky and D budgets)."""

    import nullsim.physics.pdi as pdi
    from nullsim.outputs.paper import _pdi_row

    calls: list[dict[str, object]] = []

    def fake_signal_to_noise_pdi(**kwargs: object) -> float:
        calls.append(kwargs)
        return 1.0

    monkeypatch.setattr(pdi, "signal_to_noise_pdi", fake_signal_to_noise_pdi)
    row = {
        "name": "p",
        "radius_rearth": 10.0,
        "planet_signal_rate_dark": 4.0,
        "leakage_rate_dark": 100.0,
        "sky_rate_per_dark_port": 9.0,
        "char_planet_signal_rate": 1.0,
        "char_leakage_rate": 10.0,
        "char_n_spectro_ports": 3,
        "char_sky_rate_per_spectro_port": 4.0,
        "char_dark_rate_per_spectro_port": 2.0,
    }
    _pdi_row(row, _collection(_config(), _state(), tmp_path))

    detection, characterization = calls
    assert detection["sky_rate_hz_per_dark_port_per_arm"] == pytest.approx(9.0)
    assert characterization["n_dark_ports_per_arm"] == 3
    assert characterization["sky_rate_hz_per_dark_port_per_arm"] == pytest.approx(2.0)
    assert characterization["dark_rate_hz_per_port_per_arm"] == pytest.approx(1.0)


def test_survey_rows_rescale_borrowed_char_floor_and_flag_rows(
    tmp_path: Path,
) -> None:
    """End to end through ``_survey_rows``: hosts nearest a curveless J=6 cell
    borrow the J=4 chip with the J=6 spectro floor, and are flagged."""

    from nullsim.outputs.paper import _survey_rows

    anchors = np.array([0.1, 1.0e7])

    def cell(jmag: float, *, spectro_floor: float, with_curve: bool) -> RunRecord:
        results = Results().add_component(
            "performance.standard", {"stellar_rate_hz_total": 1.0e6}
        )
        results = results.add_component(
            "characterization.spectro",
            {
                "realized_spectro_floor": spectro_floor,
                "eps_systematic": 0.0,
                "servo_bw_hz": 1000.0,
                "spectro_indices": np.array([1], dtype=np.intp),
                "D": 0.0,
            },
        )
        if with_curve:
            results = results.add_component(
                "characterization.detection_curve",
                {
                    "detected_stellar_rate_hz": 1.0e6,
                    "separations_mas": anchors,
                    "spectro_throughput_per_anchor": np.full(2, 0.5),
                    "spectro_static_leak_fraction_per_anchor": np.full(2, 1.0e-7),
                    "realized_spectro_floor_per_anchor": np.full(2, 2.0e-6),
                    "spectro_leak_fraction_per_anchor": np.full(2, 2.1e-6),
                    "eps_systematic": 0.0,
                },
            )
        return RunRecord(
            config=_config(),
            result=RunResult(
                state=replace(_state(), results=results),
                diagnostics={},
                timings=(),
                cache_hit_count=0,
                config_hash="cfg",
            ),
            run_dir=tmp_path,
            coord_tuple=(("scene.star.jmag", jmag),),
        )

    collection = RunCollection(
        config=_config(),
        records=(
            cell(4.0, spectro_floor=1.0e-5, with_curve=True),
            cell(6.0, spectro_floor=3.0e-5, with_curve=False),
        ),
        output_dir=tmp_path,
    )

    rows = _survey_rows(collection)
    borrowed = [r for r in rows if 5.0 < float(r["host_jmag"] or 0.0) < 7.0]
    native = [r for r in rows if 3.0 < float(r["host_jmag"] or 99.0) < 5.0]
    assert borrowed and native
    for row in borrowed:
        stellar = 1.0e6 * 10.0 ** (-0.4 * (row["host_jmag"] - 4.0))
        # static 1e-7 + floor 2e-6 x (3e-5 / 1e-5)
        assert row["char_leakage_rate"] == pytest.approx(stellar * 6.1e-6)
        assert row["char_floor_scaled_from_jmag"] == 4.0
        assert row["host_jmag_outside_sweep"] == int(row["host_jmag"] > 6.0)
    for row in native:
        assert np.isnan(row["char_floor_scaled_from_jmag"])

    caption = _latex_characterization_table({"Rocky": borrowed[:1]})
    assert "retuned spectro floors" in caption


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


def _config(extra_stage_types: tuple[str, ...] = ()):
    stages = [{"id": "scene", "type": "scene.point_source"}]
    stages += [{"id": f"extra{i}", "type": kind} for i, kind in enumerate(extra_stage_types)]
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
        "pipeline": {"stages": stages},
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
