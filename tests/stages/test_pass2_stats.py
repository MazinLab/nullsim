"""Diagnostic-plot stats tests for SPEC §15.3 Pass-2 stages.

These ``@pytest.mark.stats`` tests sit on top of the per-stage unit tests
for ``injection.single_mode_fiber`` (WP1), ``transport.smf28`` (WP2), and
``delay_lines.geometric`` (WP3). The unit tests already pin the physics
formulas to KILO/NAYRA numerics; this file focuses on the behavior over a
sweep — sqrt-amplitude scaling, dispersion phase zero-crossing at line
center, common-mode RMS → OPD-nm conversion factor, OPD quadrature
stacking, and the cos(azimuth) projection of the source baseline — and
saves diagnostic plots to ``tests/output/stats/`` so a human can audit
the curves when a tolerance-bounded check passes but looks suspicious.

Why a single file rather than one stats file per stage: the architect's
Pass-2 spec couples these stages through a shared ``SimulationState``,
so the diagnostic comparisons are most useful when grouped here.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend for CI / tests without a display.

import matplotlib.pyplot as plt
import numpy as np
import pytest

from nullsim.physics import ruilier_cassaing
from nullsim.pipeline import (
    Field,
    Geometry,
    OPDBudget,
    OPDComponent,
    RunContext,
    Scene,
    SimulationState,
    Throughput,
    Wavefront,
    build_instance,
)
from nullsim.stages.delay_lines import GeometricDelayLines
from nullsim.stages.injection.single_mode_fiber import SingleModeFiber
from nullsim.stages.transport import SMF28Transport
from nullsim.stages.transport.smf28 import chromatic_dispersion_phase_rad


_STATS_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "stats"


def _stats_output_path(filename: str) -> Path:
    _STATS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _STATS_OUTPUT_DIR / filename


def _context(tmp_path: Path, *, instance_id: str = "stats", seed: int = 17) -> RunContext:
    return RunContext(
        rng=np.random.default_rng(seed),
        config_hash="pass2-stats",
        sweep_coord_tuple=(),
        instance_id=instance_id,
        run_dir=tmp_path,
    )


# ---------------------------------------------------------------------------
# WP1: injection.single_mode_fiber
# ---------------------------------------------------------------------------


def _smf_state(
    *,
    wavelength_grid_um: tuple[float, ...],
    strehl_ratio: tuple[float, ...],
    diameter_m: float = 10.0,
) -> SimulationState:
    n_bins = len(wavelength_grid_um)
    amplitudes = np.ones((1, n_bins, 1), dtype=np.complex128)
    state = SimulationState.empty({"wavelength_grid_um": wavelength_grid_um})
    geometry = Geometry(
        diameter_m=diameter_m,
        area_m2=math.pi * (diameter_m / 2.0) ** 2,
        n_apertures=1,
    )
    field = Field(
        amplitudes=amplitudes,
        wavelength_grid_um=wavelength_grid_um,
        mode_labels=("mode_0",),
    )
    throughput = Throughput(
        wavelength_grid_um=wavelength_grid_um,
        transmission_fraction=tuple(1.0 for _ in wavelength_grid_um),
    )
    wavefront = Wavefront(
        wavelength_grid_um=wavelength_grid_um,
        strehl_ratio=strehl_ratio,
        amplitude_jitter_fraction=tuple(0.0 for _ in wavelength_grid_um),
    )
    return replace(
        state,
        geometry=geometry,
        field=field,
        throughput=throughput,
        wavefront=wavefront,
    )


@pytest.mark.stats
def test_smf_field_amplitude_is_sqrt_of_power_coupling_across_grid(
    tmp_path: Path,
) -> None:
    """Field carries sqrt(coupling), throughput carries coupling — the
    KILO ``stellar_input_field`` invariant.

    A violation would show amplitude that scales as coupling (power)
    rather than sqrt(coupling); the diagnostic plot makes that
    immediately visible.
    """
    grid_um = (1.2, 1.4, 1.6, 1.8)
    strehl = (0.5, 0.5, 0.5, 0.5)
    stage = SingleModeFiber(
        {"central_obscuration_ratio": 0.0},
        _context(tmp_path, instance_id="smf_sqrt"),
    )

    result = stage.apply(_smf_state(wavelength_grid_um=grid_um, strehl_ratio=strehl))

    eta_geo = ruilier_cassaing.UNOBSTRUCTED_GEOMETRIC_COUPLING_LIMIT
    expected_power = 0.5 * eta_geo
    expected_amplitude = math.sqrt(expected_power)

    coupling = np.asarray(
        result.throughput.components["single_mode_fiber"]["coupling_efficiency"],
        dtype=np.float64,
    )
    amplitudes = np.abs(result.field.amplitudes[0, :, 0])

    np.testing.assert_allclose(coupling, expected_power, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(amplitudes, expected_amplitude, atol=1e-12, rtol=0.0)

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.arange(len(grid_um))
    ax.plot(bins, np.ones_like(bins, dtype=float), "o-", label="input |E| = 1")
    ax.plot(bins, amplitudes, "s-", label="output |E| = sqrt(coupling)")
    ax.plot(bins, coupling, "^--", label="output throughput = coupling")
    ax.axhline(expected_amplitude, ls=":", color="grey", label=f"sqrt(0.5·η_geo) = {expected_amplitude:.4f}")
    ax.axhline(expected_power, ls=":", color="black", label=f"0.5·η_geo = {expected_power:.4f}")
    ax.set_xlabel("wavelength bin index")
    ax.set_ylabel("amplitude / power fraction")
    ax.set_title("WP1 SMF coupling sqrt-amplitude rule")
    ax.legend()
    fig.tight_layout()
    fig.savefig(_stats_output_path("wp1_smf_sqrt_amplitude_rule.png"), dpi=120)
    plt.close(fig)


@pytest.mark.stats
def test_smf_strehl_zero_yields_zero_field_amplitude_histogram(
    tmp_path: Path,
) -> None:
    """At Strehl=0, coupling is zero on every bin and the field amplitude
    histogram collapses to a delta at zero. No exception should be raised.
    """
    grid_um = (1.2, 1.4, 1.6, 1.8)
    stage = SingleModeFiber({}, _context(tmp_path, instance_id="smf_zero_strehl"))

    result = stage.apply(
        _smf_state(wavelength_grid_um=grid_um, strehl_ratio=(0.0, 0.0, 0.0, 0.0))
    )

    coupling = np.asarray(
        result.throughput.components["single_mode_fiber"]["coupling_efficiency"],
        dtype=np.float64,
    )
    amplitudes = np.abs(result.field.amplitudes[0, :, 0])

    np.testing.assert_allclose(coupling, 0.0, atol=0.0)
    np.testing.assert_allclose(amplitudes, 0.0, atol=0.0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(amplitudes, bins=np.linspace(-0.1, 1.0, 23))
    ax.set_xlabel("output |E|")
    ax.set_ylabel("count")
    ax.set_title("WP1 SMF Strehl=0 amplitude histogram (delta at 0)")
    fig.tight_layout()
    fig.savefig(_stats_output_path("wp1_smf_strehl_zero_histogram.png"), dpi=120)
    plt.close(fig)


@pytest.mark.stats
def test_smf_nan_strehl_mask_matches_output_amplitude_nan_mask(
    tmp_path: Path,
) -> None:
    """A NaN Strehl on one bin propagates to a NaN amplitude on the same
    bin; other bins remain finite and correctly reduced by sqrt(coupling).
    """
    grid_um = (1.2, 1.4, 1.6, 1.8)
    strehl = (0.5, math.nan, 0.5, 0.5)
    stage = SingleModeFiber({}, _context(tmp_path, instance_id="smf_nan"))

    result = stage.apply(_smf_state(wavelength_grid_um=grid_um, strehl_ratio=strehl))

    amplitudes = np.asarray(result.field.amplitudes[0, :, 0])
    nan_mask_strehl = np.array([math.isnan(s) for s in strehl], dtype=bool)
    nan_mask_amp = np.isnan(amplitudes)

    assert np.array_equal(nan_mask_strehl, nan_mask_amp)
    # Finite bins reduce by sqrt(0.5 * eta_geo) — no contamination from the NaN bin.
    expected_amplitude = math.sqrt(
        0.5 * ruilier_cassaing.UNOBSTRUCTED_GEOMETRIC_COUPLING_LIMIT,
    )
    np.testing.assert_allclose(
        np.abs(amplitudes[~nan_mask_amp]),
        expected_amplitude,
        atol=1e-12,
    )

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(
        np.vstack([nan_mask_strehl, nan_mask_amp]).astype(int),
        aspect="auto",
        cmap="Greys",
    )
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["NaN(strehl)", "NaN(|E_out|)"])
    ax.set_xlabel("wavelength bin index")
    ax.set_title("WP1 SMF NaN propagation mask")
    fig.tight_layout()
    fig.savefig(_stats_output_path("wp1_smf_nan_mask.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# WP2: transport.smf28
# ---------------------------------------------------------------------------


def _transport_state(
    *,
    wavelength_grid_um: tuple[float, ...],
    existing_opd: tuple[str, float] | None = None,
    n_modes: int = 2,
) -> SimulationState:
    state = SimulationState.empty({"wavelength_grid_um": wavelength_grid_um})
    amplitudes = np.ones(
        (n_modes, len(wavelength_grid_um), 1),
        dtype=np.complex128,
    )
    field = Field(
        amplitudes=amplitudes,
        wavelength_grid_um=wavelength_grid_um,
        mode_labels=tuple(f"mode_{i}" for i in range(n_modes)),
    )
    throughput = Throughput(
        wavelength_grid_um=wavelength_grid_um,
        transmission_fraction=tuple(1.0 for _ in wavelength_grid_um),
    )
    opd_budget = OPDBudget()
    if existing_opd is not None:
        name, rms_nm = existing_opd
        opd_budget = opd_budget.add_component(
            name, OPDComponent(rms_opd_nm=rms_nm, reduction_stage="pre_smf28"),
        )
        opd_budget = replace(opd_budget, total_rms_opd_nm=rms_nm)
    return replace(state, field=field, throughput=throughput, opd_budget=opd_budget)


@pytest.mark.stats
def test_chromatic_dispersion_phase_passes_through_zero_at_center_wavelength(
    tmp_path: Path,
) -> None:
    """The KILO/NAYRA ``chromatic_dispersion_phase`` formula is built so
    that phi_disp(λ_center) = 0 by construction (the center is the pivot
    point of the linear-in-σ expansion). Off-center bins carry nonzero
    phase. We isolate dispersion by setting ``phi_rms_rad=0``.
    """
    # 1.65 µm H-band centered grid with bandwidth.
    grid_um = (1.45, 1.55, 1.65, 1.75, 1.85)
    stage = SMF28Transport(
        {
            "length_m": 30.0,
            "phi_rms_rad": 0.0,
            "length_matching_error_m": 0.002,
            "attenuation_db_per_km": 0.0,
            "rng_seed": 0,
        },
        _context(tmp_path, instance_id="dispersion_zero_center"),
    )

    result = stage.apply(_transport_state(wavelength_grid_um=grid_um))

    # The center bin is grid[2] = 1.65 µm. Verify exp(i·phi_disp_center) == 1.
    amplitudes = result.field.amplitudes[:, 2, 0]
    # Common-mode phase is zero (phi_rms_rad=0), so any phase here is
    # dispersion alone. The center bin must be exactly the input.
    np.testing.assert_allclose(np.angle(amplitudes), 0.0, atol=1e-14)
    assert float(np.max(np.abs(amplitudes - np.abs(amplitudes)))) < 1e-14

    # Off-center bins must carry nonzero dispersion phase on at least one mode.
    off_center_phases = np.angle(result.field.amplitudes[:, [0, 1, 3, 4], 0])
    assert float(np.max(np.abs(off_center_phases))) > 0.0

    # Plot phi_disp(λ) directly from the physics function for the diagnostic.
    wavelengths_m = np.linspace(1.45e-6, 1.85e-6, 41)
    phase_rad = chromatic_dispersion_phase_rad(
        wavelengths_m,
        1.65e-6,
        n_modes=2,
        length_matching_error_m=0.002,
        rng=np.random.default_rng(0),
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(wavelengths_m * 1e6, phase_rad[:, 0], label="mode 0")
    ax.plot(wavelengths_m * 1e6, phase_rad[:, 1], label="mode 1")
    ax.axvline(1.65, ls=":", color="grey", label="λ_center = 1.65 µm")
    ax.axhline(0.0, ls=":", color="black")
    ax.set_xlabel("wavelength λ [µm]")
    ax.set_ylabel("dispersion phase φ_disp [rad]")
    ax.set_title("WP2 chromatic dispersion phase zero at λ_center")
    ax.legend()
    fig.tight_layout()
    fig.savefig(_stats_output_path("wp2_dispersion_zero_at_center.png"), dpi=120)
    plt.close(fig)


@pytest.mark.stats
def test_fiber_common_mode_phase_rms_to_opd_nm_conversion(tmp_path: Path) -> None:
    """The KILO invariant: σ_OPD [m] = φ_rms [rad] · λ_center [m] / (2π).

    Set φ_rms = 2π · 100e-9 / λ_center to land exactly at 100 nm.
    """
    # SMF28Transport defines λ_center = grid[len//2]; for a 4-bin grid that is
    # grid[2] = 1.65 µm, so we pin the test to that bin to keep the conversion
    # check independent of the grid layout.
    grid_um = (1.45, 1.55, 1.65, 1.75)
    wavelength_center_m = 1.65e-6
    target_opd_nm = 100.0
    phi_rms_rad = 2.0 * math.pi * (target_opd_nm * 1e-9) / wavelength_center_m

    stage = SMF28Transport(
        {
            "length_m": 30.0,
            "phi_rms_rad": phi_rms_rad,
            "length_matching_error_m": 0.0,  # isolate common-mode contribution
            "attenuation_db_per_km": 0.0,
            "rng_seed": 0,
        },
        _context(tmp_path, instance_id="common_mode_opd"),
    )

    result = stage.apply(_transport_state(wavelength_grid_um=grid_um))

    component = result.opd_budget.components["fiber_phase_common_mode"]
    assert component.rms_opd_nm == pytest.approx(target_opd_nm, abs=1e-9)

    conversion_factor_m_per_rad = wavelength_center_m / (2.0 * math.pi)
    fig, ax = plt.subplots(figsize=(6, 4))
    phi_grid = np.linspace(0.0, 2.0 * phi_rms_rad, 50)
    ax.plot(phi_grid, phi_grid * conversion_factor_m_per_rad * 1e9, label="σ_OPD model")
    ax.axhline(target_opd_nm, ls=":", color="grey", label=f"target = {target_opd_nm} nm")
    ax.axvline(phi_rms_rad, ls=":", color="black", label=f"φ_rms = {phi_rms_rad:.4f} rad")
    ax.set_xlabel("φ_rms [rad]")
    ax.set_ylabel("σ_OPD [nm]")
    ax.set_title("WP2 fiber common-mode φ → OPD conversion")
    ax.legend()
    fig.tight_layout()
    fig.savefig(_stats_output_path("wp2_common_mode_opd_conversion.png"), dpi=120)
    plt.close(fig)


@pytest.mark.stats
def test_opd_budget_stacks_in_quadrature_with_atmospheric_piston(
    tmp_path: Path,
) -> None:
    """Pre-existing atmos_piston, fiber_phase_common_mode, and
    fiber_chromatic_dispersion must combine in quadrature; no component
    is marked superseded.
    """
    grid_um = (1.45, 1.55, 1.65, 1.75)
    wavelength_center_m = 1.65e-6
    phi_rms_rad = 2.0 * math.pi * (100.0 * 1e-9) / wavelength_center_m

    stage = SMF28Transport(
        {
            "length_m": 30.0,
            "phi_rms_rad": phi_rms_rad,
            "length_matching_error_m": 0.0005,  # produces nonzero dispersion OPD
            "attenuation_db_per_km": 0.0,
            "rng_seed": 0,
        },
        _context(tmp_path, instance_id="opd_quadrature"),
    )

    result = stage.apply(
        _transport_state(
            wavelength_grid_um=grid_um,
            existing_opd=("atmos_piston", 200.0),
        )
    )

    components = result.opd_budget.components
    assert "atmos_piston" in components
    assert "fiber_phase_common_mode" in components
    assert "fiber_chromatic_dispersion" in components
    # No component is marked superseded.
    for name, comp in components.items():
        assert not getattr(comp, "superseded", False), f"{name} unexpectedly superseded"

    contributions_nm = np.array(
        [
            comp.rms_opd_nm
            for name, comp in components.items()
            if comp.rms_opd_nm is not None
            # The fiber common-mode phase is a global piston recorded for
            # provenance only — physically inert for nulling, so it is
            # deliberately excluded from the quadrature total
            # (review 2026-06-10 Issue 21a).
            and not name.startswith("fiber_phase_common_mode")
        ]
    )
    quadrature_nm = float(np.sqrt(np.sum(contributions_nm**2)))
    assert result.opd_budget.total_rms_opd_nm == pytest.approx(quadrature_nm, rel=1e-9)

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = list(components.keys())
    values = [components[k].rms_opd_nm or 0.0 for k in labels]
    ax.bar(labels, values, label="components")
    ax.axhline(quadrature_nm, ls="--", color="black", label=f"quadrature = {quadrature_nm:.2f} nm")
    ax.axhline(
        result.opd_budget.total_rms_opd_nm,
        ls=":",
        color="red",
        label=f"total = {result.opd_budget.total_rms_opd_nm:.2f} nm",
    )
    ax.set_ylabel("σ_OPD [nm]")
    ax.set_title("WP2 OPD budget quadrature stacking")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(_stats_output_path("wp2_opd_budget_quadrature.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# WP3: delay_lines.geometric
# ---------------------------------------------------------------------------


def _delay_state(
    *,
    aperture_positions_enu_m: np.ndarray,
    altitude_deg: float,
    azimuth_deg: float,
) -> SimulationState:
    grid_um = (1.25, 1.65)
    state = SimulationState.empty({"wavelength_grid_um": grid_um})
    scene = Scene(
        star_jmag=5.0,
        altitude_deg=altitude_deg,
        azimuth_deg=azimuth_deg,
    ).add_component(
        "science_target",
        {
            "star_jmag": 5.0,
            "altitude_deg": altitude_deg,
            "azimuth_deg": azimuth_deg,
        },
    )
    geometry = Geometry(
        n_apertures=aperture_positions_enu_m.shape[0],
        aperture_positions_enu_m=aperture_positions_enu_m,
    ).add_component(
        "stats_array",
        {
            "n_apertures": aperture_positions_enu_m.shape[0],
            "aperture_positions_enu_m": aperture_positions_enu_m,
        },
    )
    return replace(state, scene=scene, geometry=geometry)


def _build_delay_stage(tmp_path: Path, **params: object) -> GeometricDelayLines:
    stage = build_instance(
        {"id": "delay_lines_stats", "type": "delay_lines.geometric", **params},
        _context(tmp_path, instance_id="delay_lines_stats"),
    )
    assert isinstance(stage, GeometricDelayLines)
    return stage


@pytest.mark.stats
def test_delay_lines_at_zenith_have_zero_path_differences(tmp_path: Path) -> None:
    """At zenith (altitude=90°), the projected baseline collapses; every
    aperture sees the same geometric path so Δℓ_i is the zero vector and
    the per-aperture throughput is uniform.
    """
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [40.0, 0.0, 0.0],
            [-30.0, 25.0, 0.0],
        ],
        dtype=np.float64,
    )
    stage = _build_delay_stage(tmp_path, max_delay_m=200.0)

    result = stage.apply(
        _delay_state(
            aperture_positions_enu_m=positions,
            altitude_deg=90.0,
            azimuth_deg=37.0,
        )
    )

    diagnostics = result.throughput.components["delay_line_loss"]["per_aperture"]
    delay_lengths = np.asarray(diagnostics["delay_lengths_m"])
    per_aperture_T = np.asarray(diagnostics["per_aperture_transmission"])

    np.testing.assert_allclose(delay_lengths, 0.0, atol=1e-12)
    np.testing.assert_allclose(per_aperture_T, per_aperture_T[0], atol=1e-12)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(np.arange(positions.shape[0]), delay_lengths, label="Δℓ_i")
    ax.set_xlabel("aperture index")
    ax.set_ylabel("delay length [m]")
    ax.set_title("WP3 zenith degeneracy: Δℓ_i == 0")
    ax.legend()
    fig.tight_layout()
    fig.savefig(_stats_output_path("wp3_zenith_zero_delays.png"), dpi=120)
    plt.close(fig)


@pytest.mark.stats
def test_delay_lines_azimuth_sweep_traces_sin_baseline_projection(
    tmp_path: Path,
) -> None:
    """Two East-separated apertures: Δℓ vs azimuth is a clean sinusoid
    whose amplitude is the baseline projected by cos(altitude). At az=0°
    and az=180° the projected baseline is perpendicular to the source
    direction so Δℓ = 0; az=90° and az=270° are extrema with opposite
    signs (delay assigned to the trailing telescope flips).
    """
    baseline_m = 50.0
    positions = np.array(
        [[0.0, 0.0, 0.0], [baseline_m, 0.0, 0.0]],
        dtype=np.float64,
    )
    altitude_deg = 45.0
    azimuths_deg = np.linspace(0.0, 360.0, 37)
    delay_diffs_m = np.empty_like(azimuths_deg)

    for i, az_deg in enumerate(azimuths_deg):
        stage = _build_delay_stage(
            tmp_path,
            max_delay_m=baseline_m * 2.0,
            attenuation_db_per_m=0.0,
            fiber_lengths_m=(1.0, 1.0),
        )
        result = stage.apply(
            _delay_state(
                aperture_positions_enu_m=positions,
                altitude_deg=altitude_deg,
                azimuth_deg=float(az_deg),
            )
        )
        lengths = np.asarray(
            result.throughput.components["delay_line_loss"]["per_aperture"][
                "delay_lengths_m"
            ]
        )
        # Δℓ = delay assigned to aperture-1 minus delay assigned to aperture-0.
        # Only one of these is nonzero per pointing, by construction
        # (the longer geometric path receives a zero added delay).
        delay_diffs_m[i] = lengths[1] - lengths[0]

    # Check cardinal directions:
    cardinal_indices = {  # az_deg → index
        0.0: 0,
        90.0: int(np.argmin(np.abs(azimuths_deg - 90.0))),
        180.0: int(np.argmin(np.abs(azimuths_deg - 180.0))),
        270.0: int(np.argmin(np.abs(azimuths_deg - 270.0))),
    }
    expected_amplitude = baseline_m * math.cos(math.radians(altitude_deg))
    # Δℓ at az=0 and az=180 (source N/S, baseline E-W): projection vanishes.
    assert abs(delay_diffs_m[cardinal_indices[0.0]]) < 1e-9
    assert abs(delay_diffs_m[cardinal_indices[180.0]]) < 1e-9
    # |Δℓ| at az=90 and az=270 attains the projected baseline magnitude.
    assert abs(abs(delay_diffs_m[cardinal_indices[90.0]]) - expected_amplitude) < 1e-9
    assert abs(abs(delay_diffs_m[cardinal_indices[270.0]]) - expected_amplitude) < 1e-9
    # Sign flips between az=90 and az=270 (different telescope is "trailing").
    assert (
        np.sign(delay_diffs_m[cardinal_indices[90.0]])
        != np.sign(delay_diffs_m[cardinal_indices[270.0]])
    )

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="polar")
    ax.plot(np.radians(azimuths_deg), np.abs(delay_diffs_m), "o-")
    ax.set_title("WP3 |Δℓ| vs azimuth (E-W baseline, alt=45°)")
    fig.tight_layout()
    fig.savefig(_stats_output_path("wp3_azimuth_sinusoid.png"), dpi=120)
    plt.close(fig)
