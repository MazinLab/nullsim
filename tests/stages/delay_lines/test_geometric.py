"""Tests for the KILO geometric delay-line stage."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nullsim.pipeline import (
    Field,
    Geometry,
    RunContext,
    Scene,
    SimulationState,
    Throughput,
    build_instance,
)
from nullsim.stages.delay_lines import GeometricDelayLines
from nullsim.stages.delay_lines.geometric import (
    _DelayMedium,
    _chromatic_phase_rad,
    GLOPHOTONICS_HCPCF_1030_1064_60_ATTENUATION_DB_PER_M,
    GLOPHOTONICS_HCPCF_1030_1064_60_DISPERSION_PS_NM_KM,
    GLOPHOTONICS_HCPCF_1550_60_ATTENUATION_DB_PER_M,
    GLOPHOTONICS_HCPCF_1550_60_DISPERSION_PS_NM_KM,
)

GRID = {"wavelength_grid_um": (1.25, 1.65)}
H_BAND_GRID = {"wavelength_grid_um": (1.50, 1.55, 1.60)}
HC_1060_PM_GRID = {"wavelength_grid_um": (1.04, 1.06, 1.08)}
GLO_1064_GRID = {"wavelength_grid_um": (1.03, 1.064, 1.10)}
GLO_1550_GRID = {"wavelength_grid_um": (1.45, 1.55, 1.70)}

KILO_MAUNAKEA_ENU_M = np.array(
    [
        [-127.34722240860809, 54.67151237625118, -15.494478591971955],
        [-73.69510842179734, 122.67565580692155, -15.49458403138594],
        [-263.6262907592356, 5.622078337959604, -21.498418115834934],
        [464.6686215888307, -182.96924652135414, 52.487480739808056],
    ],
    dtype=np.float64,
)
KILO_DELAY_LENGTHS_ALT45_AZ90_M = np.array(
    [100.60927971567116, 138.5469787838142, 0.0, 567.2982020179873],
    dtype=np.float64,
)
KILO_LOSS_ALT45_AZ90_DB = np.array(
    [2.3584635344269307, 3.247791835803698, 0.0, 13.298496186301412],
    dtype=np.float64,
)
KILO_TRANSMISSION_ALT45_AZ90 = np.array(
    [0.580969919187712, 0.4733918932247486, 1.0, 0.04678971300610677],
    dtype=np.float64,
)


def test_geometric_delay_lines_registers_stage_contract(tmp_path: Path) -> None:
    stage = _build_geometric_delay_lines(tmp_path, max_delay_m=1.0)

    assert stage.type_name == "delay_lines.geometric"
    assert stage.family == "delay_lines"
    # INS-9-001: opd_budget is intentionally NOT declared because the stage
    # neither reads nor writes it. False declarations would defeat SPEC §6.1
    # dependency validation.
    # Per Code Review Issue 6 fix: stage now also consumes/produces
    # ``field`` so it can apply per-aperture losses to ``field.amplitudes``
    # per-mode instead of mean-smearing them into a scalar throughput.
    assert stage.consumes == frozenset({"geometry", "throughput", "scene", "field"})
    assert stage.produces == frozenset({"throughput", "field"})


def test_kilo_maunakea_delay_line_pin_matches_reference(tmp_path: Path) -> None:
    stage = _build_geometric_delay_lines(tmp_path, max_delay_m=600.0)
    state = _delay_line_state(
        aperture_positions_enu_m=KILO_MAUNAKEA_ENU_M,
        altitude_deg=45.0,
        azimuth_deg=90.0,
        throughput_fraction=(0.8, 0.6),
    )

    result = stage.apply(state)

    diagnostics = _component_diagnostics(result)
    np.testing.assert_allclose(
        diagnostics["geometric_delay_lengths_m"],
        KILO_DELAY_LENGTHS_ALT45_AZ90_M,
        atol=1.0e-9,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        diagnostics["delay_lengths_m"],
        KILO_DELAY_LENGTHS_ALT45_AZ90_M,
        atol=1.0e-9,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        diagnostics["loss_db"],
        KILO_LOSS_ALT45_AZ90_DB,
        atol=1.0e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        diagnostics["per_aperture_transmission"],
        KILO_TRANSMISSION_ALT45_AZ90,
        atol=1.0e-15,
        rtol=0.0,
    )
    expected_aggregate = float(np.mean(KILO_TRANSMISSION_ALT45_AZ90))
    assert diagnostics["transmission_fraction"] == pytest.approx(expected_aggregate)
    payload = result.throughput.components["delay_line_loss"]
    np.testing.assert_allclose(
        payload["transmission_fraction"],
        [expected_aggregate, expected_aggregate],
    )
    assert result.throughput.transmission_fraction == pytest.approx(
        (0.8 * expected_aggregate, 0.6 * expected_aggregate)
    )
    # INS-9-001: stage must not modify opd_budget — it neither consumes nor
    # produces that sub-budget.
    assert result.opd_budget.content_digest() == state.opd_budget.content_digest()


def test_zenith_over_coplanar_array_has_equal_zero_delays(tmp_path: Path) -> None:
    positions_enu_m = np.array(
        [[0.0, 0.0, 0.0], [85.6, 0.0, 0.0], [0.0, 85.6, 0.0]],
        dtype=np.float64,
    )
    stage = _build_geometric_delay_lines(tmp_path, max_delay_m=1.0)
    state = _delay_line_state(
        aperture_positions_enu_m=positions_enu_m,
        altitude_deg=90.0,
        azimuth_deg=37.0,
        throughput_fraction=(0.7, 0.5),
        fiber_lengths_m=(64.0, 64.0, 64.0),
    )

    result = stage.apply(state)

    diagnostics = _component_diagnostics(result)
    np.testing.assert_allclose(diagnostics["delay_lengths_m"], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        diagnostics["per_aperture_transmission"],
        [1.0, 1.0, 1.0],
    )
    assert result.throughput.transmission_fraction == pytest.approx((0.7, 0.5))


def test_rotating_source_azimuth_permutates_east_west_delays(
    tmp_path: Path,
) -> None:
    positions_enu_m = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    stage = _build_geometric_delay_lines(
        tmp_path,
        max_delay_m=25.0,
        attenuation_db_per_m=0.0,
        fiber_lengths_m=(1.0, 1.0, 1.0),
    )
    east_result = stage.apply(
        _delay_line_state(
            aperture_positions_enu_m=positions_enu_m,
            altitude_deg=0.0,
            azimuth_deg=90.0,
            fiber_lengths_m=(1.0, 1.0, 1.0),
        )
    )
    east_delays_m = _component_diagnostics(east_result)["delay_lengths_m"]

    west_result = stage.apply(
        _delay_line_state(
            aperture_positions_enu_m=positions_enu_m,
            altitude_deg=0.0,
            azimuth_deg=270.0,
            fiber_lengths_m=(1.0, 1.0, 1.0),
        )
    )
    west_delays_m = _component_diagnostics(west_result)["delay_lengths_m"]

    np.testing.assert_allclose(east_delays_m, [10.0, 20.0, 0.0], atol=1.0e-12)
    np.testing.assert_allclose(west_delays_m, [10.0, 0.0, 20.0], atol=1.0e-12)


def test_zero_attenuation_keeps_unity_throughput_component(tmp_path: Path) -> None:
    stage = _build_geometric_delay_lines(
        tmp_path,
        max_delay_m=100.0,
        attenuation_db_per_m=0.0,
        fiber_lengths_m=(64.0, 64.0),
    )
    state = _delay_line_state(
        aperture_positions_enu_m=np.array(
            [[0.0, 0.0, 0.0], [85.6, 0.0, 0.0]],
            dtype=np.float64,
        ),
        altitude_deg=60.0,
        azimuth_deg=90.0,
        throughput_fraction=(0.3, 0.9),
        fiber_lengths_m=(64.0, 64.0),
    )

    result = stage.apply(state)

    payload = result.throughput.components["delay_line_loss"]
    diagnostics = _component_diagnostics(result)
    np.testing.assert_allclose(
        payload["transmission_fraction"],
        [1.0, 1.0],
    )
    assert diagnostics["transmission_fraction"] == pytest.approx(1.0)
    np.testing.assert_allclose(diagnostics["loss_db"], [0.0, 0.0])
    assert result.throughput.transmission_fraction == pytest.approx((0.3, 0.9))


def test_max_delay_too_small_raises_helpful_error(tmp_path: Path) -> None:
    stage = _build_geometric_delay_lines(
        tmp_path,
        max_delay_m=10.0,
        fiber_lengths_m=(64.0, 64.0),
    )
    state = _delay_line_state(
        aperture_positions_enu_m=np.array(
            [[0.0, 0.0, 0.0], [85.6, 0.0, 0.0]],
            dtype=np.float64,
        ),
        altitude_deg=60.0,
        azimuth_deg=90.0,
        fiber_lengths_m=(64.0, 64.0),
    )

    with pytest.raises(ValueError, match="max_delay_m is insufficient"):
        stage.apply(state)


def test_nan_pointing_follows_kilo_nan_propagation(tmp_path: Path) -> None:
    stage = _build_geometric_delay_lines(
        tmp_path,
        max_delay_m=100.0,
        fiber_lengths_m=(64.0, 64.0),
    )
    state = _delay_line_state(
        aperture_positions_enu_m=np.array(
            [[0.0, 0.0, 0.0], [85.6, 0.0, 0.0]],
            dtype=np.float64,
        ),
        altitude_deg=math.nan,
        azimuth_deg=90.0,
        fiber_lengths_m=(64.0, 64.0),
    )

    result = stage.apply(state)

    diagnostics = _component_diagnostics(result)
    assert all(math.isnan(value) for value in diagnostics["delay_lengths_m"])
    assert all(math.isnan(value) for value in diagnostics["loss_db"])
    assert all(math.isnan(value) for value in result.throughput.transmission_fraction)


def test_fixed_delay_line_loss_override_replaces_photonic_model(
    tmp_path: Path,
) -> None:
    stage = _build_geometric_delay_lines(
        tmp_path,
        max_delay_m=1.0,
        fiber_lengths_m=(64.0, 64.0),
        fixed_delay_line_loss_db_per_telescope=(0.0, 5.1),
    )
    state = _delay_line_state(
        aperture_positions_enu_m=np.array(
            [[0.0, 0.0, 0.0], [85.6, 0.0, 0.0]],
            dtype=np.float64,
        ),
        altitude_deg=60.0,
        azimuth_deg=90.0,
        fiber_lengths_m=(64.0, 64.0),
    )

    result = stage.apply(state)

    diagnostics = _component_diagnostics(result)
    np.testing.assert_allclose(diagnostics["loss_db"], [0.0, 5.1])
    np.testing.assert_allclose(diagnostics["delay_lengths_m"], [0.0, 0.0])
    np.testing.assert_allclose(
        diagnostics["per_aperture_transmission"],
        [1.0, 10.0 ** (-0.51)],
        rtol=1.0e-9,
    )


def test_fixed_delay_line_loss_attenuator_preserves_geometric_delay(
    tmp_path: Path,
) -> None:
    stage = _build_geometric_delay_lines(
        tmp_path,
        max_delay_m=20.0,
        fiber_lengths_m=(1.0, 1.0),
        fixed_delay_line_loss_db_per_telescope=(2.0, 2.0),
        fixed_delay_line_loss_mode="attenuator",
    )
    state = _delay_line_state(
        aperture_positions_enu_m=np.array(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        altitude_deg=0.0,
        azimuth_deg=90.0,
        fiber_lengths_m=(1.0, 1.0),
    )

    result = stage.apply(state)

    diagnostics = _component_diagnostics(result)
    np.testing.assert_allclose(diagnostics["geometric_delay_lengths_m"], [0.0, 10.0])
    np.testing.assert_allclose(diagnostics["delay_lengths_m"], [0.0, 10.0])
    np.testing.assert_allclose(diagnostics["loss_db"], [2.0, 2.0])
    assert diagnostics["fixed_delay_line_loss_mode"] == "attenuator"


def test_chromatic_control_requires_delay_medium(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="chromatic_control requires delay_medium"):
        _build_geometric_delay_lines(
            tmp_path,
            max_delay_m=20.0,
            chromatic_control="multi_wavelength_group_delay",
        )


def test_hollow_core_medium_requires_explicit_attenuation(tmp_path: Path) -> None:
    stage = _build_geometric_delay_lines(
        tmp_path,
        max_delay_m=20.0,
        fiber_lengths_m=(1.0, 1.0),
        delay_medium="hollow_core_fiber",
        chromatic_control="multi_wavelength_group_delay",
        reference_wavelength_um=1.55,
    )
    state = _delay_line_state(
        aperture_positions_enu_m=np.array(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        altitude_deg=0.0,
        azimuth_deg=90.0,
        throughput_fraction=(1.0, 1.0, 1.0),
        fiber_lengths_m=(1.0, 1.0),
        grid=H_BAND_GRID,
        field_amplitudes=np.ones((2, 3, 1), dtype=np.complex128),
    )

    with pytest.raises(ValueError, match="attenuation_db_per_m must be set explicitly"):
        stage.apply(state)


@pytest.mark.parametrize(
    (
        "delay_medium",
        "grid",
        "reference_wavelength_um",
        "expected_attenuation_db_per_m",
        "expected_dispersion_ps_nm_km",
    ),
    [
        (
            "glophotonics_hcpcf_1030_1064_60",
            GLO_1064_GRID,
            1.064,
            GLOPHOTONICS_HCPCF_1030_1064_60_ATTENUATION_DB_PER_M,
            GLOPHOTONICS_HCPCF_1030_1064_60_DISPERSION_PS_NM_KM,
        ),
        (
            "glophotonics_hcpcf_1550_60",
            GLO_1550_GRID,
            1.55,
            GLOPHOTONICS_HCPCF_1550_60_ATTENUATION_DB_PER_M,
            GLOPHOTONICS_HCPCF_1550_60_DISPERSION_PS_NM_KM,
        ),
    ],
)
def test_glophotonics_hcpcf_presets_record_catalog_loss_and_caveats(
    tmp_path: Path,
    delay_medium: str,
    grid: dict[str, tuple[float, ...]],
    reference_wavelength_um: float,
    expected_attenuation_db_per_m: float,
    expected_dispersion_ps_nm_km: float,
) -> None:
    stage = _build_geometric_delay_lines(
        tmp_path,
        max_delay_m=20.0,
        fiber_lengths_m=(1.0, 1.0),
        delay_medium=delay_medium,
        chromatic_control="multi_wavelength_group_delay",
        reference_wavelength_um=reference_wavelength_um,
    )
    state = _delay_line_state(
        aperture_positions_enu_m=np.array(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        altitude_deg=0.0,
        azimuth_deg=90.0,
        throughput_fraction=(1.0, 1.0, 1.0),
        fiber_lengths_m=(1.0, 1.0),
        grid=grid,
        field_amplitudes=np.ones((2, 3, 1), dtype=np.complex128),
    )

    result = stage.apply(state)

    diagnostics = _component_diagnostics(result)
    assert diagnostics["delay_medium"] == delay_medium
    assert diagnostics["attenuation_db_per_m"] == pytest.approx(
        expected_attenuation_db_per_m
    )
    assert diagnostics["dispersion_ps_nm_km"] == pytest.approx(
        expected_dispersion_ps_nm_km
    )
    assert diagnostics["polarization_model"] == "assumed_pm_custom_scalar_phase"
    assert "assumed PM-custom" in str(diagnostics["polarization_caveat"])
    assert "measured spool data" in str(diagnostics["dispersion_caveat"])
    np.testing.assert_allclose(diagnostics["delay_lengths_m"], [0.0, 10.0])
    expected_second_transmission = 10.0 ** (
        -(
            diagnostics["physical_delay_lengths_m"][1]
            * expected_attenuation_db_per_m
        )
        / 10.0
    )
    np.testing.assert_allclose(
        diagnostics["per_aperture_transmission"],
        [1.0, expected_second_transmission],
        rtol=1.0e-12,
    )
    assert diagnostics["transmission_fraction"] == pytest.approx(
        float((1.0 + expected_second_transmission) / 2.0)
    )
    assert result.throughput.transmission_fraction == pytest.approx((1.0, 1.0, 1.0))
    payload = result.throughput.components["delay_line_loss"]
    np.testing.assert_allclose(
        payload["transmission_fraction"],
        np.full(3, float((1.0 + expected_second_transmission) / 2.0)),
    )
    assert payload["field_applied_loss"] is True
    assert payload["field_amplitude_applied"] is True
    assert "field amplitudes" in payload["source_model"]
    assert result.field.amplitudes is not None
    expected_field_power = np.full(
        3,
        float((1.0 + expected_second_transmission) / 2.0),
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        np.mean(np.abs(result.field.amplitudes[:, :, 0]) ** 2, axis=0),
        expected_field_power,
    )


@pytest.mark.parametrize(
    ("delay_medium", "bad_grid", "match"),
    [
        (
            "glophotonics_hcpcf_1030_1064_60",
            H_BAND_GRID,
            "supports 0.800-1.150 um",
        ),
        (
            "glophotonics_hcpcf_1550_60",
            HC_1060_PM_GRID,
            "supports 1.400-1.750 um",
        ),
    ],
)
def test_glophotonics_hcpcf_presets_reject_out_of_band_wavelengths(
    tmp_path: Path,
    delay_medium: str,
    bad_grid: dict[str, tuple[float, ...]],
    match: str,
) -> None:
    stage = _build_geometric_delay_lines(
        tmp_path,
        max_delay_m=20.0,
        fiber_lengths_m=(1.0, 1.0),
        delay_medium=delay_medium,
        chromatic_control="multi_wavelength_group_delay",
    )
    state = _delay_line_state(
        aperture_positions_enu_m=np.array(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        altitude_deg=0.0,
        azimuth_deg=90.0,
        throughput_fraction=(1.0, 1.0, 1.0),
        fiber_lengths_m=(1.0, 1.0),
        grid=bad_grid,
        field_amplitudes=np.ones((2, 3, 1), dtype=np.complex128),
    )

    with pytest.raises(ValueError, match=match):
        stage.apply(state)


def test_hollow_core_group_delay_control_applies_h_band_gvd_phase(
    tmp_path: Path,
) -> None:
    stage = _build_geometric_delay_lines(
        tmp_path,
        max_delay_m=20.0,
        attenuation_db_per_m=0.0,
        fiber_lengths_m=(1.0, 1.0),
        delay_medium="hollow_core_fiber",
        chromatic_control="multi_wavelength_group_delay",
        reference_wavelength_um=1.55,
        phase_index_ref=1.0003,
        group_index_ref=1.0004,
        dispersion_ps_nm_km=1.0,
    )
    state = _delay_line_state(
        aperture_positions_enu_m=np.array(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        altitude_deg=0.0,
        azimuth_deg=90.0,
        throughput_fraction=(1.0, 1.0, 1.0),
        fiber_lengths_m=(1.0, 1.0),
        grid=H_BAND_GRID,
        field_amplitudes=np.ones((2, 3, 2), dtype=np.complex128),
    )

    result = stage.apply(state)

    diagnostics = _component_diagnostics(result)
    assert diagnostics["delay_medium"] == "hollow_core_fiber"
    assert diagnostics["chromatic_control"] == "multi_wavelength_group_delay"
    assert diagnostics["group_delay_removed_by_control"] is True
    assert "not modeled" in str(diagnostics["polarization_caveat"])
    np.testing.assert_allclose(diagnostics["delay_lengths_m"], [0.0, 10.0])
    np.testing.assert_allclose(diagnostics["physical_delay_lengths_m"], [0.0, 10.0 / 1.0003])
    phase = np.asarray(diagnostics["chromatic_phase_rad"], dtype=np.float64)
    assert phase.shape == (2, 3)
    np.testing.assert_allclose(phase[0], 0.0)
    assert phase[1, 1] == pytest.approx(0.0, abs=1.0e-14)
    assert np.max(np.abs(phase[1, [0, 2]])) > 0.0

    assert result.field.amplitudes is not None
    np.testing.assert_allclose(result.field.amplitudes[0], 1.0 + 0.0j)
    expected_second = np.repeat(np.exp(1j * phase[1])[:, np.newaxis], 2, axis=1)
    np.testing.assert_allclose(result.field.amplitudes[1], expected_second)
    assert result.throughput.transmission_fraction == pytest.approx((1.0, 1.0, 1.0))


def test_fixed_loss_attenuator_preserves_chromatic_delay_phase(
    tmp_path: Path,
) -> None:
    stage = _build_geometric_delay_lines(
        tmp_path,
        max_delay_m=20.0,
        attenuation_db_per_m=0.0,
        fiber_lengths_m=(1.0, 1.0),
        fixed_delay_line_loss_db_per_telescope=(2.0, 2.0),
        fixed_delay_line_loss_mode="attenuator",
        delay_medium="hollow_core_fiber",
        chromatic_control="multi_wavelength_group_delay",
        reference_wavelength_um=1.55,
        phase_index_ref=1.0003,
        group_index_ref=1.0004,
        dispersion_ps_nm_km=1.0,
    )
    state = _delay_line_state(
        aperture_positions_enu_m=np.array(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        altitude_deg=0.0,
        azimuth_deg=90.0,
        throughput_fraction=(1.0, 1.0, 1.0),
        fiber_lengths_m=(1.0, 1.0),
        grid=H_BAND_GRID,
        field_amplitudes=np.ones((2, 3, 1), dtype=np.complex128),
    )

    result = stage.apply(state)

    diagnostics = _component_diagnostics(result)
    np.testing.assert_allclose(diagnostics["delay_lengths_m"], [0.0, 10.0])
    np.testing.assert_allclose(diagnostics["loss_db"], [2.0, 2.0])
    np.testing.assert_allclose(
        diagnostics["per_aperture_transmission"],
        [10.0 ** (-0.2), 10.0 ** (-0.2)],
        rtol=1.0e-12,
    )
    phase = np.asarray(diagnostics["chromatic_phase_rad"], dtype=np.float64)
    np.testing.assert_allclose(phase[0], 0.0)
    assert np.max(np.abs(phase[1, [0, 2]])) > 0.0
    assert result.field.amplitudes is not None
    expected_amp = math.sqrt(10.0 ** (-0.2))
    np.testing.assert_allclose(np.abs(result.field.amplitudes), expected_amp)


def test_phase_reference_control_keeps_residual_group_phase_slope(
    tmp_path: Path,
) -> None:
    common_params = dict(
        max_delay_m=20.0,
        attenuation_db_per_m=0.0,
        fiber_lengths_m=(1.0, 1.0),
        delay_medium="hollow_core_fiber",
        reference_wavelength_um=1.55,
        phase_index_ref=1.0003,
        group_index_ref=1.0004,
        dispersion_ps_nm_km=0.0,
    )
    state = _delay_line_state(
        aperture_positions_enu_m=np.array(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        altitude_deg=0.0,
        azimuth_deg=90.0,
        throughput_fraction=(1.0, 1.0, 1.0),
        fiber_lengths_m=(1.0, 1.0),
        grid=H_BAND_GRID,
        field_amplitudes=np.ones((2, 3, 1), dtype=np.complex128),
    )

    phase_locked = _build_geometric_delay_lines(
        tmp_path,
        **common_params,
        chromatic_control="phase_reference",
    ).apply(state)
    group_delay_locked = _build_geometric_delay_lines(
        tmp_path,
        **common_params,
        chromatic_control="multi_wavelength_group_delay",
    ).apply(state)

    phase_locked_diag = _component_diagnostics(phase_locked)
    group_locked_diag = _component_diagnostics(group_delay_locked)
    assert phase_locked_diag["group_delay_removed_by_control"] is False
    assert group_locked_diag["group_delay_removed_by_control"] is True
    assert phase_locked_diag["max_abs_chromatic_phase_rad"] == pytest.approx(
        135.0817401484693,
        rel=1.0e-12,
    )
    assert group_locked_diag["max_abs_chromatic_phase_rad"] == pytest.approx(0.0)


def test_phase_reference_residual_slope_is_group_minus_phase_index() -> None:
    # Toy medium: n_phi = 1.45, n_g = 1.47, L_phys = 10 m.
    # delta_omega for 1.20 um vs reference 1.25 um:
    #   d_omega = 2*pi*c*(1/1.20e-6 - 1/1.25e-6)
    # expected residual = L * (n_g - n_phi)/c * d_omega
    #                   = 10 * 0.02 * 2*pi * 33333.33 = 41887.9 rad.
    medium = _DelayMedium(
        name="toy",
        phase_index_ref=1.45,
        group_index_ref=1.47,
        dispersion_ps_nm_km=0.0,
        polarization_model="scalar_common_phase",
    )

    phase = _chromatic_phase_rad(
        wavelength_grid_um=(1.20, 1.25),
        reference_wavelength_um=1.25,
        delay_lengths_m=np.array([10.0 * 1.45], dtype=np.float64),
        medium=medium,
        chromatic_control="phase_reference",
    )

    assert phase[0, 0] == pytest.approx(41887.9, rel=1.0e-4)
    assert phase[0, 1] == pytest.approx(0.0, abs=1.0e-12)


def test_hc_1060_pm_rejects_h_band_grid(tmp_path: Path) -> None:
    stage = _build_geometric_delay_lines(
        tmp_path,
        max_delay_m=20.0,
        attenuation_db_per_m=0.0,
        fiber_lengths_m=(1.0, 1.0),
        delay_medium="hollow_core_1060_pm",
        chromatic_control="multi_wavelength_group_delay",
    )
    state = _delay_line_state(
        aperture_positions_enu_m=np.array(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        altitude_deg=0.0,
        azimuth_deg=90.0,
        throughput_fraction=(1.0, 1.0, 1.0),
        fiber_lengths_m=(1.0, 1.0),
        grid=H_BAND_GRID,
        field_amplitudes=np.ones((2, 3, 1), dtype=np.complex128),
    )

    with pytest.raises(ValueError, match="supports 1.030-1.090 um"):
        stage.apply(state)


def test_hc_1060_pm_records_pm_scalar_caveat(tmp_path: Path) -> None:
    stage = _build_geometric_delay_lines(
        tmp_path,
        max_delay_m=20.0,
        attenuation_db_per_m=0.0,
        fiber_lengths_m=(1.0, 1.0),
        delay_medium="hollow_core_1060_pm",
        chromatic_control="multi_wavelength_group_delay",
        reference_wavelength_um=1.06,
    )
    state = _delay_line_state(
        aperture_positions_enu_m=np.array(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        altitude_deg=0.0,
        azimuth_deg=90.0,
        throughput_fraction=(1.0, 1.0, 1.0),
        fiber_lengths_m=(1.0, 1.0),
        grid=HC_1060_PM_GRID,
        field_amplitudes=np.ones((2, 3, 1), dtype=np.complex128),
    )

    result = stage.apply(state)

    diagnostics = _component_diagnostics(result)
    assert diagnostics["delay_medium"] == "hollow_core_1060_pm"
    assert diagnostics["polarization_model"] == "catalog_pm_scalar_phase"
    assert "PMD" in str(diagnostics["polarization_caveat"])
    assert diagnostics["dispersion_ps_nm_km"] == pytest.approx(80.0)


def test_kilo_test_transport_delay_line_lengths_verbatim_xfail() -> None:
    """Code Review 2026-05-16 Issue 47: gated via ``importorskip`` so the
    ``strict=True`` marker can actually flip when KILO becomes importable.
    Pre-fix the body was unreachable because of the leading
    ``pytest.xfail(...)`` call.
    """

    pytest.importorskip("kilo.array")
    pytest.importorskip("kilo.config")
    pytest.importorskip("kilo.transport")

    from kilo.array import telescope_enu_positions
    from kilo.config import maunakea_telescopes
    from kilo.transport import FiberConfig, compute_delay_line_lengths

    telescope_enu = telescope_enu_positions(maunakea_telescopes())
    fc = FiberConfig()
    for alt in [10.0, 45.0, 90.0]:
        for az in [0.0, 90.0, 180.0, 270.0]:
            lengths = compute_delay_line_lengths(telescope_enu, fc, alt, az)
            assert np.all(lengths >= -1e-10), (
                f"Negative delay at alt={alt}, az={az}: {lengths}"
            )


def _build_geometric_delay_lines(
    tmp_path: Path,
    **params: object,
) -> GeometricDelayLines:
    stage = build_instance(
        {"id": "delay_lines", "type": "delay_lines.geometric", **params},
        _context(tmp_path),
    )
    assert isinstance(stage, GeometricDelayLines)
    return stage


def _context(tmp_path: Path) -> RunContext:
    return RunContext(
        rng=np.random.default_rng(1),
        config_hash="geometric-delay-lines",
        sweep_coord_tuple=(),
        instance_id="delay_lines",
        run_dir=tmp_path,
    )


def _component_diagnostics(result: SimulationState) -> dict[str, object]:
    """Return the per-aperture delay-line breakdown persisted on throughput.

    INS-9-003: diagnostics now live on ``state.throughput.components`` so they
    survive runner cache hits and the stage stays stateless.
    """

    payload = result.throughput.components["delay_line_loss"]
    return payload["per_aperture"]


def _delay_line_state(
    *,
    aperture_positions_enu_m: np.ndarray,
    altitude_deg: float,
    azimuth_deg: float,
    throughput_fraction: tuple[float, ...] = (1.0, 1.0),
    fiber_lengths_m: tuple[float, ...] | None = None,
    grid: dict[str, tuple[float, ...]] = GRID,
    field_amplitudes: np.ndarray | None = None,
) -> SimulationState:
    state = SimulationState.empty(grid)
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
        "test_array",
        {
            "n_apertures": aperture_positions_enu_m.shape[0],
            "aperture_positions_enu_m": aperture_positions_enu_m,
            "fiber_lengths_m": fiber_lengths_m,
        },
    )
    throughput = Throughput(
        wavelength_grid_um=state.throughput.wavelength_grid_um,
        transmission_fraction=throughput_fraction,
    ).add_component(
        "pre_delay",
        np.asarray(throughput_fraction, dtype=np.float64),
    )
    field = state.field
    if field_amplitudes is not None:
        field = Field(
            amplitudes=field_amplitudes,
            wavelength_grid_um=state.field.wavelength_grid_um,
            mode_labels=tuple(
                f"mode_{mode_index}"
                for mode_index in range(field_amplitudes.shape[0])
            ),
        )
    return replace(
        state,
        scene=scene,
        geometry=geometry,
        field=field,
        throughput=throughput,
    )
