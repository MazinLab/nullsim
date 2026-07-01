"""Tests for the SMF28Transport stage."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from nullsim.pipeline import (
    Field,
    OPDComponent,
    OPDBudget,
    RunContext,
    SimulationState,
    Throughput,
    build_instance,
    list_stages,
)
from nullsim.stages.transport import PM780Transport, SMF28Transport
from nullsim.stages.transport.smf28 import (
    PM780HP_ATTENUATION_DB_PER_KM,
    PM780HP_WAVELENGTHS_UM,
    PM980_DISPERSION_PS_NM_KM,
    PM980Transport,
    SMF28_DISPERSION_PS_NM_KM,
    chromatic_dispersion_phase_rad,
    pm780hp_attenuation_db_per_km,
    pm980xp_attenuation_db_per_km,
    smf28e_attenuation_db_per_km,
)


def _context(tmp_path: Path, *, seed: int = 1) -> RunContext:
    return RunContext(
        rng=np.random.default_rng(seed),
        config_hash="smf28-transport-pinning",
        sweep_coord_tuple=(),
        instance_id="fiber_to_chip",
        run_dir=tmp_path,
    )


def _stage(context: RunContext, **overrides: object) -> SMF28Transport:
    spec = {
        "id": "fiber_to_chip",
        "type": "transport.smf28",
        "length_m": 30.0,
        **overrides,
    }
    stage = build_instance(spec, context)
    assert isinstance(stage, SMF28Transport)
    return stage


def test_pm980_defaults_use_nayra_dispersion_and_group_index() -> None:
    params = PM980Transport.Params(length_m=5.0)

    assert params.attenuation_curve == "pm980_xp"
    assert params.dispersion_ps_nm_km == pytest.approx(PM980_DISPERSION_PS_NM_KM)
    # n_g was a dead Params field (no computation read it; group index
    # enters only via the delay-line stage's medium model) — removed in
    # review 2026-06-10.


def test_pm780_defaults_to_conservative_800nm_loss() -> None:
    params = PM780Transport.Params(length_m=2.0)

    assert params.attenuation_curve == "pm780_hp"
    assert params.attenuation_db_per_km == pytest.approx(4.0)
    assert pm780hp_attenuation_db_per_km(0.85e-6) == pytest.approx(4.0)
    np.testing.assert_allclose(PM780HP_WAVELENGTHS_UM, [0.77, 0.85, 1.10])
    np.testing.assert_allclose(PM780HP_ATTENUATION_DB_PER_KM, [4.0, 4.0, 4.0])


def _state(
    wavelength_grid_um: tuple[float, ...] = (0.9, 1.0, 1.1),
    *,
    n_modes: int = 4,
    existing_opd_nm: float | None = None,
) -> SimulationState:
    state = SimulationState.empty({"wavelength_grid_um": wavelength_grid_um})
    field = Field(
        amplitudes=np.ones((n_modes, len(wavelength_grid_um), 1), dtype=np.complex128),
        wavelength_grid_um=wavelength_grid_um,
        mode_labels=tuple(f"mode_{mode_index}" for mode_index in range(n_modes)),
    )
    throughput = Throughput(
        wavelength_grid_um=wavelength_grid_um,
        transmission_fraction=tuple(1.0 for _ in wavelength_grid_um),
    )
    opd_budget = OPDBudget(total_rms_opd_nm=existing_opd_nm)
    if existing_opd_nm is not None:
        opd_budget = opd_budget.add_component(
            "existing",
            OPDComponent(rms_opd_nm=existing_opd_nm, reduction_stage="test"),
        )
    return replace(state, field=field, throughput=throughput, opd_budget=opd_budget)


def test_smf28_registers_stage_contract(tmp_path: Path) -> None:
    stage = _stage(_context(tmp_path))

    assert "transport.smf28" in list_stages()
    assert stage.type_name == "transport.smf28"
    assert stage.family == "transport"
    assert stage.consumes == frozenset({"field", "throughput", "opd_budget"})
    assert stage.produces == frozenset({"field", "throughput", "opd_budget"})


def test_pm780_registers_stage_contract(tmp_path: Path) -> None:
    spec = {
        "id": "pm780",
        "type": "transport.pm780",
        "length_m": 2.0,
        "phi_rms_rad": 0.0,
        "length_matching_error_m": 0.0,
    }
    stage = build_instance(spec, _context(tmp_path))

    assert "transport.pm780" in list_stages()
    assert isinstance(stage, PM780Transport)
    assert stage.type_name == "transport.pm780"
    assert stage.consumes == frozenset({"field", "throughput", "opd_budget"})
    assert stage.produces == frozenset({"field", "throughput", "opd_budget"})


def test_smf28e_attenuation_matches_kilo_datasheet_pins() -> None:
    assert smf28e_attenuation_db_per_km(1.31e-6) == pytest.approx(0.33)
    assert smf28e_attenuation_db_per_km(1.55e-6) == pytest.approx(0.19)
    assert smf28e_attenuation_db_per_km(1.0e-6) > smf28e_attenuation_db_per_km(1.65e-6)


def test_attenuation_curves_reject_extrapolation() -> None:
    with pytest.raises(ValueError, match="SMF-28e attenuation requested outside"):
        smf28e_attenuation_db_per_km(np.array([0.8e-6, 1.2e-6]))
    with pytest.raises(ValueError, match="PM980-XP attenuation requested outside"):
        pm980xp_attenuation_db_per_km(1.8e-6)
    with pytest.raises(ValueError, match="PM780-HP attenuation requested outside"):
        pm780hp_attenuation_db_per_km(0.76e-6)


def test_common_mode_phase_matches_kilo_wavelength_scaling_pin(tmp_path: Path) -> None:
    stage = _stage(
        _context(tmp_path),
        attenuation_db_per_km=0.0,
        length_matching_error_m=0.0,
        phi_rms_rad=0.1,
        rng_seed=42,
    )

    result = stage.apply(_state())

    phase_rad = np.angle(result.field.amplitudes[0, :, 0])
    assert phase_rad[1] == pytest.approx(0.03047170797544314)
    np.testing.assert_allclose(phase_rad[0] / phase_rad[1], 1.0 / 0.9, rtol=1.0e-10)
    np.testing.assert_allclose(phase_rad[2] / phase_rad[1], 1.0 / 1.1, rtol=1.0e-10)
    np.testing.assert_allclose(
        np.angle(result.field.amplitudes[:, 1, 0]),
        np.full(4, phase_rad[1]),
    )


def test_chromatic_dispersion_phase_is_bit_zero_at_center_wavelength() -> None:
    wavelengths_m = np.array([0.95e-6, 1.0e-6, 1.05e-6], dtype=np.float64)

    phase_rad = chromatic_dispersion_phase_rad(
        wavelengths_m,
        1.0e-6,
        n_modes=4,
        length_matching_error_m=0.001,
        rng=np.random.default_rng(42),
    )

    assert np.array_equal(phase_rad[1], np.zeros(4, dtype=np.float64))


def test_chromatic_dispersion_matches_kilo_small_for_matched_fibers_pin() -> None:
    wavelengths_m = np.linspace(0.95e-6, 1.05e-6, 10)

    phase_rad = chromatic_dispersion_phase_rad(
        wavelengths_m,
        1.0e-6,
        n_modes=4,
        length_matching_error_m=0.001,
        dispersion_ps_nm_km=SMF28_DISPERSION_PS_NM_KM,
        rng=np.random.default_rng(42),
    )

    assert phase_rad.shape == (10, 4)
    assert float(np.max(np.abs(phase_rad))) == pytest.approx(0.09795084871189812)
    assert float(np.max(np.abs(phase_rad))) < 1.0


def test_chromatic_dispersion_matches_nayra_small_for_matched_fibers_pin() -> None:
    wavelengths_m = np.linspace(0.95e-6, 1.05e-6, 10)

    phase_rad = chromatic_dispersion_phase_rad(
        wavelengths_m,
        1.0e-6,
        n_modes=4,
        length_matching_error_m=0.001,
        dispersion_ps_nm_km=-29.5,
        rng=np.random.default_rng(42),
    )

    assert phase_rad.shape == (10, 4)
    assert float(np.max(np.abs(phase_rad))) == pytest.approx(0.07223875092502485)
    assert float(np.max(np.abs(phase_rad))) < 1.0


def test_stage_applies_loss_to_throughput_only(tmp_path: Path) -> None:
    # Transport loss is recorded only in throughput.transmission_fraction
    # (and its provenance component). The field amplitude is NOT scaled by
    # sqrt(transmission) — the detector multiplies |field|² × throughput,
    # so scaling here as well would double-count (see Finding 1 fix).
    stage = _stage(
        _context(tmp_path),
        length_m=1000.0,
        attenuation_db_per_km=3.0,
        phi_rms_rad=0.0,
        length_matching_error_m=0.0,
    )

    result = stage.apply(_state((1.0, 1.1), n_modes=2))

    expected_power = 10.0 ** (-3.0 / 10.0)
    assert result.throughput.transmission_fraction == pytest.approx(
        (expected_power, expected_power),
    )
    np.testing.assert_allclose(
        np.abs(result.field.amplitudes),
        1.0,
    )
    np.testing.assert_allclose(
        result.throughput.components["transport.smf28_loss"]["transmission_fraction"],
        np.full(2, expected_power),
    )


def test_phi_rms_zero_adds_no_common_mode_phase_or_opd(tmp_path: Path) -> None:
    stage = _stage(
        _context(tmp_path),
        attenuation_db_per_km=0.0,
        phi_rms_rad=0.0,
        length_matching_error_m=0.0,
    )

    result = stage.apply(_state())

    np.testing.assert_allclose(result.field.amplitudes, 1.0 + 0.0j)
    assert "fiber_phase_common_mode" not in result.opd_budget.components


def test_attenuation_zero_adds_unit_throughput_component(tmp_path: Path) -> None:
    stage = _stage(
        _context(tmp_path),
        attenuation_db_per_km=0.0,
        phi_rms_rad=0.0,
        length_matching_error_m=0.0,
    )

    result = stage.apply(_state((1.0, 1.1)))

    assert result.throughput.transmission_fraction == pytest.approx((1.0, 1.0))
    np.testing.assert_allclose(
        result.throughput.components["transport.smf28_loss"]["transmission_fraction"],
        np.ones(2),
    )


def test_length_m_zero_rejected_by_params() -> None:
    with pytest.raises(ValidationError, match="length_m"):
        SMF28Transport.Params(length_m=0.0)


def test_nan_wavelength_grid_propagates_to_outputs(tmp_path: Path) -> None:
    # phi_rms_rad > 0 enables a wavelength-dependent phase modulation so
    # NaN wavelengths can reach the field via exp(1j·NaN). Without any
    # phase or amplitude modulation, NaN wavelength would only show up in
    # throughput (which is the correct behavior — field passes through
    # unchanged when the stage has no field-modifying parameters active).
    stage = _stage(
        _context(tmp_path),
        phi_rms_rad=0.1,
        length_matching_error_m=0.0,
        rng_seed=0,
    )

    result = stage.apply(_state((1.0, math.nan, 1.1), n_modes=2))

    assert math.isnan(result.throughput.transmission_fraction[1])
    assert np.isnan(
        result.throughput.components["transport.smf28_loss"]["transmission_fraction"][1]
    )
    assert np.isnan(result.field.amplitudes[:, 1, :]).all()


def test_opd_components_stack_in_quadrature(tmp_path: Path) -> None:
    wavelengths_m = np.array([0.95e-6, 1.0e-6, 1.05e-6], dtype=np.float64)
    dispersion_phase_rad = chromatic_dispersion_phase_rad(
        wavelengths_m,
        1.0e-6,
        n_modes=4,
        length_matching_error_m=0.001,
        rng=np.random.default_rng(42),
    )
    # Mode 0 is the reference (delta_length_m[0]=0) so its dispersion
    # is identically zero — exclude it from the RMS to match the
    # stage's _dispersion_rms_opd_nm helper (Code Review Issue 11 fix).
    dispersion_opd_nm = float(
        np.sqrt(
            np.mean(
                np.square(
                    (dispersion_phase_rad * wavelengths_m[:, None] / (2.0 * math.pi) * 1.0e9)[:, 1:]
                )
            )
        )
    )
    common_opd_nm = 0.2 * 1.0e-6 / (2.0 * math.pi) * 1.0e9
    stage = _stage(
        _context(tmp_path),
        attenuation_db_per_km=0.0,
        phi_rms_rad=0.2,
        rng_seed=42,
        dispersion_seed=42,
        length_matching_error_m=0.001,
    )

    result = stage.apply(_state((0.95, 1.0, 1.05), existing_opd_nm=3.0))

    assert result.opd_budget.components["fiber_phase_common_mode"].rms_opd_nm == pytest.approx(
        common_opd_nm,
    )
    assert result.opd_budget.components[
        "fiber_chromatic_dispersion"
    ].rms_opd_nm == pytest.approx(dispersion_opd_nm)
    # The common-mode phase is a global piston (physically inert for any
    # interferometric observable): recorded as a provenance component but
    # deliberately EXCLUDED from the quadrature total (review 2026-06-10
    # Issue 21a).
    assert result.opd_budget.total_rms_opd_nm == pytest.approx(
        math.sqrt(3.0**2 + dispersion_opd_nm**2),
    )
