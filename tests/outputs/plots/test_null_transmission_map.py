from __future__ import annotations

from dataclasses import replace
import logging
from pathlib import Path

import numpy as np
import pytest

import nullsim  # noqa: F401  # ensures output registration side effects are available
from nullsim.config.loader import resolve_config
from nullsim.outputs import RunCollection, RunRecord
from nullsim.outputs.plots._chip_field_inputs import mode_positions_enu_m
from nullsim.outputs.plots.null_transmission_map import (
    _display_limits_mas,
    _format_contour_label,
    _planet_display_position_mas,
    null_transmission_map,
)
from nullsim.pipeline import (
    Field,
    Geometry,
    Results,
    RunResult,
    Scene,
    SimulationState,
)


def test_null_transmission_map_writes_plot_and_arrays(tmp_path: Path) -> None:
    config = _config()
    collection = _collection(config, _state(), tmp_path)

    written = null_transmission_map(collection, tmp_path, ("png",))

    plot_path = tmp_path / "plots" / "null_transmission_map.png"
    array_path = tmp_path / "plots" / "null_transmission_map.npz"
    assert plot_path in written
    assert plot_path.exists()
    assert plot_path.stat().st_size > 4_000
    assert array_path in written
    arrays = np.load(array_path)
    assert arrays["design_map"].shape == (25, 25)
    assert arrays["band_averaged_map"].shape == (25, 25)
    assert arrays["sub_diameters_m"].shape == (2,)
    assert arrays["positions_pupil_m"].shape == (2, 2)
    assert arrays["positions_projected_m"].shape == (2, 2)
    assert str(arrays["projection_mode"]) == "pupil"
    assert arrays["display_x_mas"][0] < arrays["display_x_mas"][-1]
    assert arrays["design_map"][12, 12] == pytest.approx(0.0, abs=1.0e-14)


def test_null_transmission_map_saves_display_consistent_arrays(tmp_path: Path) -> None:
    """The npz pairs a display-ready (East-left) map with ``display_x_mas``.

    Regression: the saved raw ``design_map`` (West-first) was paired only with
    ``display_x_mas`` (East-first), so a naive re-plot mirrored the PDF. The
    ``*_display`` maps match the plotted figure; the raw maps stay native.
    """

    config = _config()
    collection = _collection(config, _state(), tmp_path)
    null_transmission_map(collection, tmp_path, ("png",))
    arrays = np.load(tmp_path / "plots" / "null_transmission_map.npz")

    np.testing.assert_array_equal(
        arrays["design_map_display"], arrays["design_map"][:, ::-1]
    )
    np.testing.assert_array_equal(
        arrays["band_averaged_map_display"], arrays["band_averaged_map"][:, ::-1]
    )
    np.testing.assert_allclose(
        arrays["display_x_mas"], -arrays["l_coords_mas"][::-1]
    )


def test_null_transmission_map_handles_per_arm_deployment(tmp_path: Path) -> None:
    config = _config(n_pix=17)
    collection = _collection(config, _state(per_pol_arm=True), tmp_path)

    written = null_transmission_map(collection, tmp_path, ("png",))

    assert (tmp_path / "plots" / "null_transmission_map.png") in written
    arrays = np.load(tmp_path / "plots" / "null_transmission_map.npz")
    np.testing.assert_allclose(arrays["unitary_arm_weights"], [0.5, 0.5])
    assert arrays["dark_indices"].shape == (2,)


def test_null_transmission_map_warns_and_skips_missing_chip(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = _config()
    state = _state(include_chip=False)
    collection = _collection(config, state, tmp_path)

    with caplog.at_level(
        logging.WARNING,
        logger="nullsim.outputs.plots.null_transmission_map",
    ):
        written = null_transmission_map(collection, tmp_path, ("png",))

    assert written == ()
    assert "skipping record" in caplog.text
    assert "chip.kernel_mzi_mesh" in caplog.text


def test_null_transmission_map_uses_fallback_without_sensitivity(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = _config(n_pix=17)
    state = _state(include_sensitivity=False)
    collection = _collection(config, state, tmp_path)

    with caplog.at_level(
        logging.WARNING,
        logger="nullsim.outputs.plots.null_transmission_map",
    ):
        written = null_transmission_map(collection, tmp_path, ("png",))

    assert (tmp_path / "plots" / "null_transmission_map.png") in written
    assert "skipping record" not in caplog.text
    arrays = np.load(tmp_path / "plots" / "null_transmission_map.npz")
    assert arrays["design_map"].shape == (17, 17)


def test_null_transmission_map_auto_projects_ground_pointing(
    tmp_path: Path,
) -> None:
    config = _config(n_pix=17, site_latitude_deg=20.0)
    state = _state(
        positions=np.array([[0.0, -5.0], [0.0, 5.0]], dtype=np.float64),
        altitude_deg=60.0,
        azimuth_deg=0.0,
    )
    collection = _collection(config, state, tmp_path)

    null_transmission_map(collection, tmp_path, ("png",))

    arrays = np.load(tmp_path / "plots" / "null_transmission_map.npz")
    np.testing.assert_allclose(
        arrays["positions_pupil_m"],
        [[0.0, -5.0], [0.0, 5.0]],
    )
    np.testing.assert_allclose(
        arrays["positions_projected_m"],
        [[0.0, -5.0 * np.cos(np.radians(30.0))], [0.0, 5.0 * np.cos(np.radians(30.0))]],
        atol=1.0e-12,
    )
    assert str(arrays["projection_mode"]) == "sky"
    assert arrays["projection_declination_rad"] == pytest.approx(np.radians(50.0))


def test_null_transmission_map_keeps_pupil_offsets_unprojected(
    tmp_path: Path,
) -> None:
    """Per-mode telescope origins foreshorten; intra-telescope offsets do not.

    Regression for review 2026-06-10 Issue 20: the writer used to project the
    full mode positions (origin + pupil offset), foreshortening sub-aperture
    offsets that physically track the target. With ``mode_origin_positions_m``
    available the map must match the stage-side projection.
    """

    config = _config(n_pix=17, site_latitude_deg=20.0)
    origins = np.array([[0.0, 100.0], [0.0, 100.0]], dtype=np.float64)
    state = _state(
        positions=np.array([[0.0, 95.0], [0.0, 105.0]], dtype=np.float64),
        altitude_deg=60.0,
        azimuth_deg=0.0,
    )
    state = replace(
        state,
        geometry=state.geometry.add_component(
            "ao.pyramid_wfs",
            {"mode_origin_positions_m": origins},
        ),
    )
    collection = _collection(config, state, tmp_path)

    null_transmission_map(collection, tmp_path, ("png",))

    arrays = np.load(tmp_path / "plots" / "null_transmission_map.npz")
    projected_origin_north = 100.0 * np.sin(np.radians(60.0))
    np.testing.assert_allclose(
        arrays["positions_projected_m"],
        [
            [0.0, projected_origin_north - 5.0],
            [0.0, projected_origin_north + 5.0],
        ],
        atol=1.0e-9,
    )
    assert str(arrays["projection_mode"]) == "sky"


def test_mode_positions_enu_prefers_3d_positions() -> None:
    state = SimulationState.empty({"wavelength_grid_um": (1.25,)})
    state = replace(
        state,
        geometry=Geometry(
            aperture_positions_pupil_m=np.array([[0.0, 1.0], [2.0, 3.0]]),
            aperture_positions_enu_m=np.array(
                [[0.0, 1.0, 4.0], [2.0, 3.0, 5.0]],
            ),
        ),
    )

    positions = mode_positions_enu_m(state, 2)

    np.testing.assert_allclose(positions[:, 2], [4.0, 5.0])


def test_mode_positions_enu_falls_back_to_pupil_modes_when_enu_is_array_level() -> None:
    state = SimulationState.empty({"wavelength_grid_um": (1.25,)})
    state = replace(
        state,
        geometry=Geometry(
            aperture_positions_enu_m=np.array([[0.0, 0.0, 1.0], [10.0, 0.0, 1.0]]),
            aperture_positions_pupil_m=np.array(
                [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
            ),
        ),
    )

    positions = mode_positions_enu_m(state, 3)

    np.testing.assert_allclose(positions[:, :2], [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    np.testing.assert_allclose(positions[:, 2], 0.0)


def test_null_transmission_map_can_force_pupil_projection(
    tmp_path: Path,
) -> None:
    config = _config(
        n_pix=17,
        site_latitude_deg=20.0,
        null_map_overrides={"projection_mode": "pupil"},
    )
    state = _state(
        positions=np.array([[0.0, -5.0], [0.0, 5.0]], dtype=np.float64),
        altitude_deg=60.0,
        azimuth_deg=0.0,
    )
    collection = _collection(config, state, tmp_path)

    null_transmission_map(collection, tmp_path, ("png",))

    arrays = np.load(tmp_path / "plots" / "null_transmission_map.npz")
    np.testing.assert_allclose(
        arrays["positions_projected_m"],
        arrays["positions_pupil_m"],
    )
    assert str(arrays["projection_mode"]) == "pupil"


def test_planet_overlay_uses_east_left_display_coordinates(tmp_path: Path) -> None:
    config = _config()
    record = _collection(config, _state(), tmp_path).records[0]

    x_mas, y_mas = _planet_display_position_mas(record)

    assert x_mas == pytest.approx(-20.0)
    assert y_mas == pytest.approx(0.0, abs=1.0e-12)


def test_display_limits_crop_dark_floor_region() -> None:
    config = _config().output.null_transmission_map
    x_coords = np.linspace(-300.0, 300.0, 7)
    y_coords = np.linspace(-300.0, 300.0, 7)
    transmission = np.full((7, 7), 1.0e-30, dtype=np.float64)
    transmission[2:5, 2:5] = 1.0e-8

    x_limits, y_limits = _display_limits_mas(
        maps=(transmission, transmission),
        x_coords_mas=x_coords,
        y_coords_mas=y_coords,
        config=config,
        planet_position_mas=None,
    )

    assert x_limits[0] > -300.0
    assert x_limits[1] < 300.0
    assert y_limits == x_limits


def test_display_limits_use_relative_floor_for_log_maps() -> None:
    config = _config().output.null_transmission_map
    x_coords = np.linspace(-300.0, 300.0, 7)
    y_coords = np.linspace(-300.0, 300.0, 7)
    transmission = np.full((7, 7), 1.0e-4, dtype=np.float64)
    transmission[2:5, 2:5] = 1.0

    x_limits, y_limits = _display_limits_mas(
        maps=(transmission, transmission),
        x_coords_mas=x_coords,
        y_coords_mas=y_coords,
        config=config,
        planet_position_mas=None,
    )

    assert x_limits == pytest.approx((-110.0, 110.0))
    assert y_limits == x_limits


def test_log_contour_labels_are_readable() -> None:
    assert _format_contour_label(1.0e-6, log_scale=True) == r"$10^{-6}$"
    assert _format_contour_label(0.1, log_scale=True) == r"$10^{-1}$"


def _config(
    *,
    n_pix: int = 25,
    site_latitude_deg: float | None = None,
    null_map_overrides: dict[str, object] | None = None,
):
    null_map_config = {
        "n_pix": n_pix,
        "fov_mas": 40.0,
        "log_scale": True,
        "log_floor": 1.0e-12,
        "save_arrays": True,
    }
    if null_map_overrides:
        null_map_config.update(null_map_overrides)
    raw_config = {
        "run": {"name": "null_map_test", "seed": 1},
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
            "plots": ["null_transmission_map"],
            "formats": {"plots": ["png"], "tables": ["json"]},
            "null_transmission_map": null_map_config,
        },
    }
    if site_latitude_deg is not None:
        raw_config["site"] = {"latitude_deg": site_latitude_deg}
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


def _state(
    *,
    include_chip: bool = True,
    include_sensitivity: bool = True,
    per_pol_arm: bool = False,
    positions: np.ndarray | None = None,
    altitude_deg: float | None = None,
    azimuth_deg: float | None = None,
) -> SimulationState:
    wavelength_grid_um = (0.95, 1.0, 1.05)
    state = SimulationState.empty({"wavelength_grid_um": wavelength_grid_um})
    positions_pupil_m = (
        np.array([[-5.0, 0.0], [5.0, 0.0]], dtype=np.float64)
        if positions is None
        else np.asarray(positions, dtype=np.float64)
    )
    geometry = Geometry(
        diameter_m=1.0,
        area_m2=1.0,
        collecting_area_m2=2.0,
        effective_collecting_area_m2=2.0,
        n_apertures=2,
        aperture_positions_pupil_m=positions_pupil_m,
    )
    n_pol = 2 if per_pol_arm else 1
    field = Field(
        amplitudes=np.ones((2, 3, n_pol), dtype=np.complex128),
        wavelength_grid_um=wavelength_grid_um,
    )
    scene = Scene(
        star_jmag=5.0,
        star_angular_diameter_mas=1.0,
        star_distance_pc=5.0,
        planet_contrast=1.0e-6,
        planet_separation_mas=20.0,
        planet_position_angle_deg=90.0,
        altitude_deg=altitude_deg,
        azimuth_deg=azimuth_deg,
    )
    unitary = np.array(
        [[1.0, 1.0], [1.0, -1.0]],
        dtype=np.complex128,
    ) / np.sqrt(2.0)
    u_stack = np.stack([unitary, unitary, unitary], axis=2)
    results = Results()
    if include_chip:
        chip_payload = {
            "u_stack": u_stack,
            "wavelength_grid_um": wavelength_grid_um,
            "wl_design_um": 1.0,
            "n_modes": 2,
            "n_bright_ports": 1,
            "n_dark_ports": 1,
            "n_spectro_ports": 0,
        }
        if per_pol_arm:
            chip_payload.update(
                {
                    "deployment_mode": "per_pol_arm",
                    "u_stack_per_arm": np.stack([u_stack, u_stack], axis=0),
                    "chip_channel_labels": ("arm_0", "arm_1"),
                }
            )
        results = results.add_component("chip.kernel_mzi_mesh", chip_payload)
    if include_sensitivity:
        results = results.add_component(
            "sensitivity.detection_curve",
            {
                "coupling_per_sub": np.ones(2, dtype=np.float64),
                "sub_areas_m2": np.ones(2, dtype=np.float64),
                "sub_diameters_m": np.ones(2, dtype=np.float64),
                "dark_indices": np.array([1], dtype=np.intp),
            },
        )
    return replace(
        state,
        geometry=geometry,
        field=field,
        scene=scene,
        results=results,
    )
