"""Smoke + regression tests for the chip_diagnostics figure (KILO paperv3 Fig 7)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import nullsim  # noqa: F401  # ensures plot registration runs at import time
from nullsim.config.loader import resolve_config
from nullsim.outputs import RunCollection, RunRecord
from nullsim.outputs.plots.chip_diagnostics import (
    chip_diagnostics_characterization,
    plot_characterization_chip_diagnostics_for_record,
    plot_chip_diagnostics,
)
from nullsim.pipeline import Field, Results, RunResult, SimulationState
from nullsim.stages.chip_optimization import (
    ChipOptimizationParams,
    _build_unitary_stack_numpy,
    _n_params,
)


def _synthetic_chip_state(
    m: int = 4,
    n_dark: int = 2,
    n_wavelengths: int = 5,
    wl_design: float = 1.25,
    bandwidth: float = 0.1,
    dac_bits: int | None = 16,
    seed: int = 0,
):
    """Return (opt_params, chip_params, field, wl_grid) for a small synthetic chip."""

    rng = np.random.default_rng(seed)
    opt_params = rng.uniform(0.0, 2.0 * np.pi, _n_params(m))
    chip_params = ChipOptimizationParams(
        n_modes=m,
        n_bright_ports=m - n_dark,
        n_dark_ports=n_dark,
        wl_design_um=wl_design,
        dac_bits=dac_bits,
    )
    wl_grid = tuple(np.linspace(
        wl_design - bandwidth / 2, wl_design + bandwidth / 2, n_wavelengths
    ).tolist())
    stack = _build_unitary_stack_numpy(
        params_vec=opt_params, n_modes=m, wavelength_grid_um=wl_grid,
        wl_design_um=wl_design, dac_bits=dac_bits,
    )
    star_in = np.ones(m, dtype=np.complex128) / np.sqrt(m)
    field = np.empty((m, n_wavelengths, 1), dtype=np.complex128)
    for k in range(n_wavelengths):
        field[:, k, 0] = stack[:, :, k] @ star_in
    return opt_params, chip_params, field, wl_grid


def test_render_small_synthetic_chip(tmp_path: Path) -> None:
    opt, chip, field, wl_grid = _synthetic_chip_state()
    out = tmp_path / "chip_diagnostics.pdf"
    fig = plot_chip_diagnostics(
        opt_params=opt, chip_params=chip,
        field_amplitudes=field, wavelength_grid_um=wl_grid,
        save_path=out,
    )
    assert out.exists()
    assert out.stat().st_size > 5000
    # 5 panels: port_power, |U|, arg(U), DAC sweep, broadband null
    assert len(fig.axes) >= 5


def test_detection_chip_diagnostics_labels_dark_plus_spectro_set() -> None:
    opt, _, field, wl_grid = _synthetic_chip_state(
        m=4,
        n_dark=1,
        n_wavelengths=3,
        seed=3,
    )
    chip = ChipOptimizationParams(
        n_modes=4,
        n_bright_ports=2,
        n_dark_ports=1,
        n_spectro_ports=1,
        wl_design_um=1.25,
        dac_bits=16,
    )
    fig = plot_chip_diagnostics(
        opt_params=opt,
        chip_params=chip,
        field_amplitudes=field,
        wavelength_grid_um=wl_grid,
    )
    try:
        titles = [ax.get_title() for ax in fig.axes]
        ylabels = [ax.get_ylabel() for ax in fig.axes]
        assert "Dark+spectro depth vs DAC precision" in titles
        assert "Null depth (dark+spectro / total)" in ylabels
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_render_keck_sized_chip(tmp_path: Path) -> None:
    """M=24 (KILO Keck baseline) — slower but exercises the real mesh size."""

    opt, chip, field, wl_grid = _synthetic_chip_state(
        m=24, n_dark=18, n_wavelengths=20, dac_bits=16,
    )
    out = tmp_path / "chip_diagnostics_m24.pdf"
    plot_chip_diagnostics(
        opt_params=opt, chip_params=chip,
        field_amplitudes=field, wavelength_grid_um=wl_grid,
        save_path=out,
    )
    assert out.exists()


def test_broadband_panel_shows_physical_pipeline_input() -> None:
    """Chromatic input phases should be visible beside the clean chip metric."""

    import matplotlib.pyplot as plt

    opt, chip, field, wl_grid = _synthetic_chip_state(
        m=6,
        n_dark=3,
        n_wavelengths=7,
        bandwidth=0.04,
        seed=11,
    )
    stack = _build_unitary_stack_numpy(
        params_vec=opt,
        n_modes=chip.n_modes,
        wavelength_grid_um=wl_grid,
        wl_design_um=chip.wl_design_um,
        dac_bits=chip.dac_bits,
    )
    design_idx = int(np.argmin(np.abs(np.asarray(wl_grid) - chip.wl_design_um)))
    base_input = np.ones(chip.n_modes, dtype=np.complex128) / np.sqrt(chip.n_modes)
    phase_ramp = np.linspace(0.0, 2.5, chip.n_modes, dtype=np.float64)
    for k in range(len(wl_grid)):
        offset = float(k - design_idx)
        physical_input = base_input * np.exp(1j * offset * phase_ramp)
        field[:, k, 0] = stack[:, :, k] @ physical_input

    fig = plot_chip_diagnostics(
        opt_params=opt,
        chip_params=chip,
        field_amplitudes=field,
        wavelength_grid_um=wl_grid,
        deployed_u_stack=stack,
    )
    try:
        broadband_ax = next(
            ax for ax in fig.axes if ax.get_title() == "Broadband null spectrum"
        )
        lines_by_label = {
            line.get_label(): line
            for line in broadband_ax.lines
            if not line.get_label().startswith("_")
        }
        assert "Clean input" in lines_by_label
        assert "Physical pipeline input" in lines_by_label
        clean_y = np.asarray(lines_by_label["Clean input"].get_ydata(), dtype=float)
        physical_y = np.asarray(
            lines_by_label["Physical pipeline input"].get_ydata(),
            dtype=float,
        )
        assert clean_y.shape == physical_y.shape
        assert not np.allclose(clean_y, physical_y)
    finally:
        plt.close(fig)


def test_characterization_chip_diagnostics_uses_spectro_null_set(
    tmp_path: Path,
) -> None:
    """The char diagnostic should plot the retuned spectro port, not the
    pipeline-deployed detection-chip dark set."""

    import matplotlib.pyplot as plt

    record = _synthetic_characterization_record(tmp_path)
    fig = plot_characterization_chip_diagnostics_for_record(record)
    assert fig is not None
    try:
        assert fig._suptitle is not None
        assert "Characterization Chip Diagnostics" in fig._suptitle.get_text()
        broadband_ax = next(
            ax for ax in fig.axes if ax.get_title() == "Broadband null spectrum"
        )
        assert "spectro / total" in broadband_ax.get_ylabel()
    finally:
        plt.close(fig)

    collection = RunCollection(
        config=record.config,
        records=(record,),
        output_dir=tmp_path,
    )
    paths = chip_diagnostics_characterization(collection, tmp_path, ("png",))
    assert len(paths) == 1
    assert paths[0].name == "chip_diagnostics_characterization.png"
    assert paths[0].exists()
    assert paths[0].stat().st_size > 5000


def test_dac_sweep_monotonic_in_bits_on_average() -> None:
    """Higher DAC precision should not catastrophically increase the null."""

    from nullsim.outputs.plots.chip_diagnostics import _plot_null_vs_dac
    import matplotlib.pyplot as plt

    opt, chip, field, wl_grid = _synthetic_chip_state(m=8, n_dark=4, seed=42)
    fig, ax = plt.subplots()
    # Recover the chip-input field at the design wavelength.
    stack = _build_unitary_stack_numpy(
        params_vec=opt, n_modes=chip.n_modes,
        wavelength_grid_um=(chip.wl_design_um,),
        wl_design_um=chip.wl_design_um, dac_bits=chip.dac_bits,
    )
    u_design = stack[:, :, 0]
    design_bin = int(np.argmin(np.abs(np.array(wl_grid) - chip.wl_design_um)))
    star_field_design = u_design.conj().T @ field[:, design_bin, 0]
    null_idx = np.arange(chip.n_modes - chip.n_dark_ports, chip.n_modes, dtype=np.int64)

    result = _plot_null_vs_dac(ax, opt, chip, star_field_design, null_idx, chip.wl_design_um)
    plt.close(fig)

    # End-to-end trend: 20-bit null depth should be much less than 4-bit.
    nds = result["null_depths"]
    assert nds[-1] < nds[0]


def _synthetic_characterization_record(tmp_path: Path) -> RunRecord:
    m = 4
    n_bright = 2
    n_dark = 1
    n_spectro = 1
    wl_design = 1.25
    wl_grid = tuple(np.linspace(1.20, 1.30, 5).tolist())
    rng = np.random.default_rng(123)
    det_params = rng.uniform(0.0, 2.0 * np.pi, _n_params(m))
    char_params = rng.uniform(0.0, 2.0 * np.pi, _n_params(m))
    det_stack = _build_unitary_stack_numpy(
        params_vec=det_params,
        n_modes=m,
        wavelength_grid_um=wl_grid,
        wl_design_um=wl_design,
        dac_bits=16,
    )
    char_stack = _build_unitary_stack_numpy(
        params_vec=char_params,
        n_modes=m,
        wavelength_grid_um=wl_grid,
        wl_design_um=wl_design,
        dac_bits=16,
    )
    input_fields = np.empty((m, len(wl_grid)), dtype=np.complex128)
    base = np.ones(m, dtype=np.complex128) / np.sqrt(m)
    phase = np.linspace(0.0, 0.4, m)
    for k in range(len(wl_grid)):
        input_fields[:, k] = base * np.exp(1j * (k - 2) * phase)
    det_outputs = np.einsum("oik,ik->ok", det_stack, input_fields)[:, :, np.newaxis]

    raw_config = {
        "run": {"name": "char_chip_diag_test", "seed": 1},
        "grid": {
            "wavelength_center_um": wl_design,
            "wavelength_bandwidth_um": 0.1,
            "n_wavelength_bins": len(wl_grid),
        },
        "scene": {
            "star": {"jmag": 5.0, "angular_diameter_mas": 1.0, "distance_pc": 5.0},
            "planet": {
                "contrast": 1.0e-6,
                "separation_mas": 20.0,
                "position_angle_deg": 90.0,
            },
        },
        "observation": {"integration_time_s": 10.0},
        "pipeline": {
            "stages": [
                {
                    "id": "chip_opt",
                    "type": "chip_optimization.torch",
                    "n_modes": m,
                    "n_bright_ports": n_bright,
                    "n_dark_ports": n_dark,
                    "n_spectro_ports": n_spectro,
                    "wl_design_um": wl_design,
                    "dac_bits": 16,
                },
                {
                    "id": "char",
                    "type": "characterization.spectro",
                    "n_modes": m,
                    "n_bright_ports": n_bright,
                    "n_dark_ports": n_dark,
                    "n_spectro_ports": n_spectro,
                    "wl_design_um": wl_design,
                    "dac_bits": 16,
                },
            ],
        },
    }
    config = resolve_config(raw_config)
    state = SimulationState.empty({"wavelength_grid_um": wl_grid})
    state = SimulationState(
        scene=state.scene,
        geometry=state.geometry,
        field=Field(amplitudes=det_outputs, wavelength_grid_um=wl_grid),
        opd_budget=state.opd_budget,
        throughput=state.throughput,
        wavefront=state.wavefront,
        photon_rates=state.photon_rates,
        results=Results(
            components={
                "chip_optimization.torch": {
                    "optimal_chip_params": det_params,
                    "optimal_u_stack": det_stack,
                },
                "characterization.spectro": {
                    "optimal_chip_params": tuple(float(v) for v in char_params),
                    "optimal_u_stack": char_stack,
                    "wavelengths_m": tuple(float(w * 1.0e-6) for w in wl_grid),
                    "bright_indices": np.arange(0, n_bright, dtype=np.intp),
                    "dark_indices": np.arange(
                        n_bright, n_bright + n_dark, dtype=np.intp
                    ),
                    "spectro_indices": np.arange(
                        n_bright + n_dark,
                        n_bright + n_dark + n_spectro,
                        dtype=np.intp,
                    ),
                    "band_lambda_m": wl_design * 1.0e-6,
                },
            },
        ),
    )
    return RunRecord(
        config=config,
        result=RunResult(
            state=state,
            diagnostics={},
            timings=(),
            cache_hit_count=0,
            config_hash="cfg",
        ),
        run_dir=tmp_path / "cell",
        coord_tuple=(),
    )
