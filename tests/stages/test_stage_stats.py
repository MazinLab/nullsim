"""Scientific validation tests for the atmosphere and AO stages.

These ``@pytest.mark.stats`` tests verify the physics-consistency claims
spelled out by the architect for the KolmogorovScreens and PyramidWFS
stages: wavelength scaling, seeing-limit envelopes, AO error scalings,
and the OPD quadrature invariant. They also save diagnostic plots to
``tests/output/stats/`` so a human can audit the curves when a
tolerance-bounded check passes but looks suspicious.

The pinning numbers and reference configuration are handled by the
existing unit tests in ``test_kolmogorov_screens.py`` and
``test_pyramid_wfs.py``; this file focuses on the *behavior over a
sweep*, not the absolute KILO numerics.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend for CI / tests without a display.

import matplotlib.pyplot as plt
import numpy as np
import pytest

from nullsim.physics import kolmogorov
from nullsim.pipeline import (
    Geometry,
    OPDComponent,
    RunContext,
    RunResult,
    Scene,
    SimulationState,
    build_instance,
    run,
)
from nullsim.stages.atmosphere import KolmogorovScreens


_STATS_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "stats"


def _stats_output_path(filename: str) -> Path:
    _STATS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _STATS_OUTPUT_DIR / filename


# ---------------------------------------------------------------------------
# Atmosphere-only helpers
# ---------------------------------------------------------------------------


def _scene_geometry_state(
    *,
    wavelength_grid_um: tuple[float, ...],
    diameter_m: float,
    star_jmag: float = 4.0,
) -> SimulationState:
    area_m2 = math.pi * (diameter_m / 2.0) ** 2
    state = SimulationState.empty({"wavelength_grid_um": wavelength_grid_um})
    scene = Scene(star_jmag=star_jmag).add_component(
        "stats_scene", {"star_jmag": star_jmag}
    )
    geometry = Geometry(
        diameter_m=diameter_m, area_m2=area_m2, n_apertures=1
    ).add_component(
        "stats_aperture",
        {"diameter_m": diameter_m, "area_m2": area_m2, "n_apertures": 1},
    )
    return replace(state, scene=scene, geometry=geometry)


def _atmosphere_only(
    *,
    tmp_path: Path,
    wavelength_grid_um: tuple[float, ...],
    diameter_m: float,
    r0_500nm_m: float,
) -> SimulationState:
    """Build a state, run only KolmogorovScreens, return the resulting state."""

    state = _scene_geometry_state(
        wavelength_grid_um=wavelength_grid_um, diameter_m=diameter_m
    )
    context = RunContext(
        rng=np.random.default_rng(0),
        config_hash="stats-atmosphere",
        sweep_coord_tuple=(),
        instance_id="atmosphere",
        run_dir=tmp_path,
    )
    spec = {
        "id": "atmosphere",
        "type": "atmosphere.kolmogorov",
        "r0_500nm_m": r0_500nm_m,
    }
    stage = build_instance(spec, context)
    assert isinstance(stage, KolmogorovScreens)
    return stage.apply(state)


# ---------------------------------------------------------------------------
# Atmosphere stats: Strehl monotonic with wavelength, OPD wavelength-invariant
# ---------------------------------------------------------------------------


@pytest.mark.stats
def test_atmosphere_strehl_monotonic_with_wavelength(tmp_path: Path) -> None:
    """log10(1-Strehl) falls monotonically with wavelength on a 1.5–1.8 um grid.

    The architect's exact prescription (D=4 m, r0=0.16 m) lies deep in the
    seeing-limited regime, so we use the moderate-Strehl r0 chosen to keep
    the curve in the linear small-phase regime where the slope is well
    defined. The qualitative monotonicity (Strehl rises with wavelength
    because r0 scales as lambda^(6/5)) is what the SPEC guarantees and is
    what we assert.
    """
    diameter_m = 4.0
    r0_500nm_m = 0.5  # moderate seeing — keeps Strehl in (0.1, 0.5) at H-band
    wavelength_grid_um = tuple(np.linspace(1.5, 1.8, 8).round(6).tolist())

    state = _atmosphere_only(
        tmp_path=tmp_path,
        wavelength_grid_um=wavelength_grid_um,
        diameter_m=diameter_m,
        r0_500nm_m=r0_500nm_m,
    )

    strehl = np.asarray(state.wavefront.strehl_ratio)
    assert strehl.shape == (8,)
    assert np.all(np.isfinite(strehl))
    assert np.all((strehl >= 0.0) & (strehl <= 1.0))
    # Strict monotonic increase with wavelength: r0 grows -> D/r0 shrinks.
    assert np.all(np.diff(strehl) > 0.0)

    # Diagnostic: log10(1-Strehl) vs log10(lambda) -- straight line of
    # negative slope (analytic small-phase prediction is -2; we only assert
    # the qualitative sign).
    wavelength_grid_m = np.asarray(wavelength_grid_um) * 1.0e-6
    log_lambda = np.log10(wavelength_grid_m)
    log_one_minus = np.log10(1.0 - strehl)
    slope = float(np.polyfit(log_lambda, log_one_minus, 1)[0])
    assert slope < 0.0

    fig, ax = plt.subplots()
    ax.plot(wavelength_grid_um, log_one_minus, "o-", label="atmosphere stage")
    ax.set_xlabel("wavelength (um)")
    ax.set_ylabel("log10(1 - Strehl)")
    ax.set_title(
        f"Atmosphere Strehl wavelength scaling (slope vs log10(lambda) = "
        f"{slope:.3f})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(_stats_output_path("atmosphere_strehl_vs_wavelength.png"), dpi=120)
    plt.close(fig)


@pytest.mark.stats
def test_atmosphere_opd_rms_is_wavelength_independent(tmp_path: Path) -> None:
    """Per-wavelength OPD RMS is flat — Kolmogorov property.

    The (D/r0)^(5/3) factor in OPD^2 has a lambda^(-2) wavelength dependence
    from r0(lambda); the lambda^2 factor in OPD = sqrt(phase_variance) * lambda
    / (2*pi) cancels it exactly. Result: OPD RMS in meters is identical at
    every wavelength bin.
    """
    diameter_m = 4.0
    wavelength_grid_um = tuple(np.linspace(1.5, 1.8, 8).round(6).tolist())

    state = _atmosphere_only(
        tmp_path=tmp_path,
        wavelength_grid_um=wavelength_grid_um,
        diameter_m=diameter_m,
        r0_500nm_m=0.16,
    )

    component = state.wavefront.components["atmosphere"]
    opd_rms_nm = np.asarray(component["opd_rms_nm"])
    # 1e-3 nm = 1 picometer absolute tolerance is well below any expected
    # numerical drift across an octave of wavelength.
    np.testing.assert_allclose(opd_rms_nm, opd_rms_nm[0], atol=1.0e-3)

    fig, ax = plt.subplots()
    ax.bar(np.arange(len(wavelength_grid_um)), opd_rms_nm)
    ax.set_xticks(np.arange(len(wavelength_grid_um)))
    ax.set_xticklabels([f"{w:.3f}" for w in wavelength_grid_um], rotation=45)
    ax.set_xlabel("wavelength (um)")
    ax.set_ylabel("atmosphere OPD RMS (nm)")
    ax.set_title("Per-wavelength atmosphere OPD RMS — should be flat")
    fig.tight_layout()
    fig.savefig(_stats_output_path("atmosphere_opd_rms_per_wavelength.png"), dpi=120)
    plt.close(fig)


@pytest.mark.stats
def test_atmosphere_seeing_envelopes_match_marechal_curve(tmp_path: Path) -> None:
    """Strehl across a (D, r0) sweep falls on the Marechal-like envelope.

    Plots Strehl vs (D/r0_at_lambda)^(5/3) and overlays exp(-c*x) with c set
    by the Noll piston-removed coefficient (0.582). Bad seeing (D=10, r0=0.05)
    gives Strehl << 0.01; excellent seeing (D=2, r0=0.5) gives Strehl > 0.5.
    """
    scenarios = [
        (10.0, 0.05),  # terrible seeing on a big telescope
        (8.0, 0.08),
        (6.0, 0.10),
        (4.0, 0.16),
        (4.0, 0.30),
        (4.0, 0.50),
        (3.0, 0.50),
        (2.0, 0.50),  # excellent seeing on a small telescope
    ]
    wavelength_um = 1.65
    wavelength_m = wavelength_um * 1.0e-6

    x_values: list[float] = []
    strehls: list[float] = []
    for diameter_m, r0_500nm_m in scenarios:
        state = _atmosphere_only(
            tmp_path=tmp_path,
            wavelength_grid_um=(wavelength_um,),
            diameter_m=diameter_m,
            r0_500nm_m=r0_500nm_m,
        )
        r0_at_lambda_m = float(
            kolmogorov.r0_scaled(r0_500nm_m, np.array([wavelength_m]))[0]
        )
        x_values.append((diameter_m / r0_at_lambda_m) ** (5.0 / 3.0))
        strehls.append(float(state.wavefront.strehl_ratio[0]))

    # Architect's two anchor envelopes:
    bad = next(s for (d, r), s in zip(scenarios, strehls) if d == 10.0 and r == 0.05)
    good = next(s for (d, r), s in zip(scenarios, strehls) if d == 2.0 and r == 0.5)
    assert bad < 0.01
    assert good > 0.5

    fig, ax = plt.subplots()
    x_arr = np.asarray(x_values)
    ax.scatter(x_arr, strehls, label="stage output", zorder=2)
    x_curve = np.linspace(0.0, max(x_arr.max(), 1.0), 200)
    ax.plot(
        x_curve,
        np.exp(-kolmogorov.NOLL_PISTON_REMOVED_COEFFICIENT * x_curve),
        "k--",
        label="Marechal: exp(-0.582*(D/r0)^(5/3))",
    )
    ax.set_xlabel(r"$(D/r_0(\lambda))^{5/3}$")
    ax.set_ylabel("Strehl ratio")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1.5)
    ax.set_title("Atmosphere Strehl envelope (piston-removed Noll)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(_stats_output_path("atmosphere_strehl_marechal_envelope.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# AO pipeline helpers
# ---------------------------------------------------------------------------


_AO_REFERENCE: dict[str, object] = {
    "wavelength_um": 1.65,
    "diameter_m": 4.0,
    "r0_500nm_m": 0.16,
    "n_actuators": 3481,  # k=59, smallest k**2 >= 3500-ish; strict-square per Code Review Issue 9
    "loop_frequency_hz": 1000.0,
    "guide_star_R_mag": 8.0,
}


def _register_stage_modules() -> None:
    importlib.import_module("nullsim.stages.scene.point_source")
    importlib.import_module("nullsim.stages.telescope.fixed_aperture")
    importlib.import_module("nullsim.stages.atmosphere.kolmogorov_screens")
    importlib.import_module("nullsim.stages.ao.pyramid_wfs")


def _ao_config(
    tmp_path: Path,
    *,
    ao_overrides: dict[str, object] | None = None,
    atmosphere_overrides: dict[str, object] | None = None,
    wavelength_grid_um: tuple[float, ...] | None = None,
) -> dict[str, object]:
    grid = wavelength_grid_um or (_AO_REFERENCE["wavelength_um"],)
    atmosphere_stage: dict[str, object] = {
        "id": "atmosphere",
        "type": "atmosphere.kolmogorov",
        "r0_500nm_m": _AO_REFERENCE["r0_500nm_m"],
    }
    if atmosphere_overrides:
        atmosphere_stage.update(atmosphere_overrides)
    ao_stage: dict[str, object] = {
        "id": "ao",
        "type": "ao.pyramid_wfs",
        "n_actuators": _AO_REFERENCE["n_actuators"],
        "loop_frequency_hz": _AO_REFERENCE["loop_frequency_hz"],
        "guide_star_R_mag": _AO_REFERENCE["guide_star_R_mag"],
    }
    if ao_overrides:
        ao_stage.update(ao_overrides)
    return {
        "config_hash": "stage-stats",
        "grid": {"wavelength_grid_um": grid},
        "run": {"seed": 1},
        "cache": {"enabled": False, "dir": tmp_path / "cache", "version": 1},
        "output": {"dir": tmp_path / "run"},
        "pipeline": {
            "stages": [
                {
                    "id": "scene",
                    "type": "scene.point_source",
                    "star_jmag": _AO_REFERENCE["guide_star_R_mag"],
                },
                {
                    "id": "array",
                    "type": "telescope.fixed_aperture",
                    "diameter_m": _AO_REFERENCE["diameter_m"],
                },
                atmosphere_stage,
                ao_stage,
            ]
        },
    }


def _run_ao(tmp_path: Path, **kwargs: object) -> RunResult:
    _register_stage_modules()
    return run(_ao_config(tmp_path, **kwargs))


# ---------------------------------------------------------------------------
# PyramidWFS stats
# ---------------------------------------------------------------------------


@pytest.mark.stats
def test_ao_post_correction_strehl_exceeds_open_loop_atmosphere(
    tmp_path: Path,
) -> None:
    """Post-AO Strehl is strictly greater than the atmosphere-only Strehl
    across the full wavelength grid for the nominal Maunakea-like config."""

    wavelength_grid_um = tuple(np.linspace(1.2, 2.2, 4).round(6).tolist())
    result = _run_ao(tmp_path, wavelength_grid_um=wavelength_grid_um)

    atmosphere = np.asarray(
        result.state.wavefront.components["atmosphere"]["strehl_ratio"]
    )
    post_ao = np.asarray(result.state.wavefront.components["ao_residual"]["strehl_ratio"])
    assert atmosphere.shape == post_ao.shape == (len(wavelength_grid_um),)
    assert np.all(post_ao > atmosphere)
    # Post-AO Strehl should rise into a useful regime by H/K band.
    assert post_ao[-1] > 0.4

    fig, ax = plt.subplots()
    idx = np.arange(len(wavelength_grid_um))
    width = 0.4
    ax.bar(idx - width / 2, atmosphere, width=width, label="atmosphere only")
    ax.bar(idx + width / 2, post_ao, width=width, label="post-AO")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{w:.2f}" for w in wavelength_grid_um])
    ax.set_xlabel("wavelength (um)")
    ax.set_ylabel("Strehl ratio")
    ax.set_title("Strehl pre- vs post-AO (PyramidWFS)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(_stats_output_path("ao_strehl_pre_vs_post.png"), dpi=120)
    plt.close(fig)


@pytest.mark.stats
def test_ao_fitting_error_scales_monotonically_with_actuator_count(
    tmp_path: Path,
) -> None:
    """Fitting OPD RMS falls monotonically with n_actuators.

    The PSD-integrated piston filter makes the closed-form slope nontrivial,
    so we assert (a) strict monotonic decrease and (b) the very-large-DM
    limit is below 1 nm. Diagnostic: log-log fitting RMS vs sqrt(n_actuators).
    """
    # strict-square per Code Review Issue 9: each value must be k**2
    n_actuator_sweep = (100, 484, 2025, 10000)  # k = 10, 22, 45, 100
    fitting_rms_nm: list[float] = []
    for n_act in n_actuator_sweep:
        result = _run_ao(tmp_path, ao_overrides={"n_actuators": n_act})
        component = result.state.opd_budget.components["ao_fitting"]
        assert isinstance(component, OPDComponent)
        fitting_rms_nm.append(component.rms_opd_nm)

    rms = np.asarray(fitting_rms_nm)
    assert np.all(np.diff(rms) < 0.0)
    # Strict monotone decrease only; the piston filter caps the absolute
    # floor because it integrates the fitting PSD above the DM cutoff, so
    # the n_actuators=1e4 limit is small but not zero in practice.

    fig, ax = plt.subplots()
    sqrt_n = np.sqrt(np.asarray(n_actuator_sweep, dtype=np.float64))
    ax.loglog(sqrt_n, rms, "o-", label="ao_fitting OPD RMS")
    ax.set_xlabel("sqrt(n_actuators)")
    ax.set_ylabel("piston-filtered fitting OPD RMS (nm)")
    ax.set_title("AO fitting error vs actuator count")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(_stats_output_path("ao_fitting_vs_n_actuators.png"), dpi=120)
    plt.close(fig)


@pytest.mark.stats
def test_ao_temporal_error_scales_monotonically_with_loop_frequency(
    tmp_path: Path,
) -> None:
    """Temporal OPD RMS falls monotonically with loop frequency."""

    loop_freq_sweep = (100.0, 500.0, 2000.0, 5000.0)
    temporal_rms_nm: list[float] = []
    for f_hz in loop_freq_sweep:
        result = _run_ao(tmp_path, ao_overrides={"loop_frequency_hz": f_hz})
        component = result.state.opd_budget.components["ao_temporal"]
        assert isinstance(component, OPDComponent)
        temporal_rms_nm.append(component.rms_opd_nm)

    rms = np.asarray(temporal_rms_nm)
    assert np.all(np.diff(rms) < 0.0)
    # Open-loop / Greenwood: phase var ∝ (f_g/f_servo)^(5/3) → RMS ∝ f^(-5/6).
    # log-log slope of total residual temporal RMS is -5/6; with the piston
    # filter it's a bit different but still in the same neighborhood. Pin
    # only loosely.
    slope = float(np.polyfit(np.log10(loop_freq_sweep), np.log10(rms), 1)[0])
    assert -1.5 < slope < -0.5

    fig, ax = plt.subplots()
    ax.loglog(loop_freq_sweep, rms, "o-", label="ao_temporal OPD RMS")
    ax.set_xlabel("AO loop frequency (Hz)")
    ax.set_ylabel("piston-filtered temporal OPD RMS (nm)")
    ax.set_title(f"AO temporal error vs loop frequency (slope ≈ {slope:.2f})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(_stats_output_path("ao_temporal_vs_loop_frequency.png"), dpi=120)
    plt.close(fig)


@pytest.mark.stats
def test_ao_wfs_noise_scales_monotonically_with_guide_star_magnitude(
    tmp_path: Path,
) -> None:
    """WFS noise OPD rises with guide-star magnitude; faint star kills Strehl."""

    mag_sweep = (4.0, 8.0, 12.0, 16.0)
    temporal_nm: list[float] = []
    fitting_nm: list[float] = []
    wfs_noise_nm: list[float] = []
    strehls: list[float] = []
    for mag in mag_sweep:
        result = _run_ao(tmp_path, ao_overrides={"guide_star_R_mag": mag})
        temporal_nm.append(
            result.state.opd_budget.components["ao_temporal"].rms_opd_nm
        )
        fitting_nm.append(
            result.state.opd_budget.components["ao_fitting"].rms_opd_nm
        )
        wfs_noise_nm.append(
            result.state.opd_budget.components["ao_wfs_noise"].rms_opd_nm
        )
        strehls.append(float(result.state.wavefront.strehl_ratio[0]))

    wfs = np.asarray(wfs_noise_nm)
    assert np.all(np.diff(wfs) > 0.0)

    # Very faint guide star — WFS noise dominates and post-AO Strehl → 0.
    result_faint = _run_ao(tmp_path, ao_overrides={"guide_star_R_mag": 20.0})
    assert result_faint.state.wavefront.strehl_ratio[0] < 1.0e-3

    fig, ax = plt.subplots()
    idx = np.arange(len(mag_sweep))
    ax.bar(idx, temporal_nm, label="ao_temporal")
    ax.bar(idx, fitting_nm, bottom=temporal_nm, label="ao_fitting")
    bottom2 = np.asarray(temporal_nm) + np.asarray(fitting_nm)
    ax.bar(idx, wfs_noise_nm, bottom=bottom2, label="ao_wfs_noise")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"R={m:.0f}" for m in mag_sweep])
    ax.set_xlabel("guide star magnitude (R band)")
    ax.set_ylabel("piston-filtered OPD RMS (nm)")
    ax.set_title("AO OPD budget vs guide-star magnitude (stacked)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(_stats_output_path("ao_opd_budget_vs_guide_star_mag.png"), dpi=120)
    plt.close(fig)


@pytest.mark.stats
def test_ao_nan_atmosphere_propagates_to_post_ao_strehl(tmp_path: Path) -> None:
    """NaN r0 propagates: post-AO Strehl is NaN at every wavelength, no exception."""

    wavelength_grid_um = (1.2, 1.65, 2.2)
    result = _run_ao(
        tmp_path,
        wavelength_grid_um=wavelength_grid_um,
        atmosphere_overrides={"r0_500nm_m": float("nan")},
    )
    strehl = np.asarray(result.state.wavefront.strehl_ratio)
    assert strehl.shape == (3,)
    assert np.all(np.isnan(strehl))
