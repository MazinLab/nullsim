"""Tests for the Clements MZI-mesh chip kernel stage (SPEC §15.3 Pass 3)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nullsim.config import NullsimConfigError
from nullsim.physics.clements import ClementsDecomposition, reconstruct_clements
from nullsim.pipeline import (
    BaseStage,
    Field,
    RunContext,
    SimulationState,
    Throughput,
    validate_dependencies,
)
from nullsim.stages.chip import KernelMZIMesh


def _context(tmp_path: Path, *, seed: int = 1) -> RunContext:
    return RunContext(
        rng=np.random.default_rng(seed),
        config_hash="kernel-mzi-mesh-pinning",
        sweep_coord_tuple=(),
        instance_id="chip",
        run_dir=tmp_path,
    )


def _stage(
    context: RunContext,
    *,
    n_modes: int,
    params: tuple[float, ...],
    n_bright_ports: int | None = None,
    n_dark_ports: int | None = None,
    n_spectro_ports: int = 0,
    wl_design_um: float = 1.65,
    dac_bits: int | None = None,
    **extra_params: object,
) -> KernelMZIMesh:
    if n_bright_ports is None:
        n_bright_ports = n_modes - 1
    if n_dark_ports is None:
        n_dark_ports = n_modes - n_bright_ports
    # NB: the spec field name "params" collides with stage_params_from_spec,
    # which strips top-level "params". Construct the stage directly with a
    # clean params dict to side-step the registry pre-processing.
    params_dict = {
        "n_modes": n_modes,
        "n_bright_ports": n_bright_ports,
        "n_dark_ports": n_dark_ports,
        "n_spectro_ports": n_spectro_ports,
        "wl_design_um": wl_design_um,
        "params": tuple(float(p) for p in params),
        "dac_bits": dac_bits,
        "insertion_loss_db": 0.0,
    }
    params_dict.update(extra_params)
    return KernelMZIMesh(params_dict, context)


def _state(
    *,
    wavelength_grid_um: tuple[float, ...] = (1.65,),
    n_modes: int = 4,
    amplitudes: np.ndarray | None = None,
    transmission_fraction: tuple[float, ...] | None = None,
) -> SimulationState:
    if amplitudes is None:
        amplitudes = np.ones((n_modes, len(wavelength_grid_um), 1), dtype=np.complex128)
    if transmission_fraction is None:
        transmission_fraction = tuple(1.0 for _ in wavelength_grid_um)
    state = SimulationState.empty({"wavelength_grid_um": wavelength_grid_um})
    field = Field(
        amplitudes=amplitudes,
        wavelength_grid_um=wavelength_grid_um,
        mode_labels=tuple(f"mode_{i}" for i in range(n_modes)),
    )
    throughput = Throughput(
        wavelength_grid_um=wavelength_grid_um,
        transmission_fraction=transmission_fraction,
    )
    return replace(state, field=field, throughput=throughput)


def _n_param_total(n_modes: int) -> int:
    n_mzi = n_modes * (n_modes - 1) // 2
    return 2 * n_mzi + n_modes


def _random_params(n_modes: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 2.0 * np.pi, _n_param_total(n_modes))


@pytest.mark.stats
def test_kernel_mzi_mesh_unitary_preserves_per_bin_power(tmp_path: Path) -> None:
    """Spec (a): per (wl, pol), output total power equals input total power."""
    n_modes = 4
    n_wl = 16
    rng = np.random.default_rng(7)
    wavelength_grid_um = tuple(np.linspace(1.5, 1.8, n_wl).tolist())
    raw_field = rng.standard_normal((n_modes, n_wl, 1)) + 1j * rng.standard_normal(
        (n_modes, n_wl, 1)
    )
    norm_per_bin = np.sqrt(np.sum(np.abs(raw_field) ** 2, axis=0, keepdims=True))
    amplitudes = (raw_field / norm_per_bin).astype(np.complex128)

    params = _random_params(n_modes, seed=42)
    stage = _stage(
        _context(tmp_path),
        n_modes=n_modes,
        params=tuple(params),
        wl_design_um=1.65,
    )
    state = _state(
        wavelength_grid_um=wavelength_grid_um,
        n_modes=n_modes,
        amplitudes=amplitudes,
    )

    result = stage.apply(state)
    out = result.field.amplitudes
    assert out is not None
    p_in = np.sum(np.abs(amplitudes) ** 2, axis=0)
    p_out = np.sum(np.abs(out) ** 2, axis=0)
    np.testing.assert_allclose(p_out, p_in, atol=1e-12, rtol=0.0)


def test_kernel_mzi_mesh_zero_params_is_identity_on_design(tmp_path: Path) -> None:
    """Spec (b): params=zeros at wl_design returns the input field unchanged."""
    n_modes = 4
    n_wl = 1
    rng = np.random.default_rng(11)
    amplitudes = (
        rng.standard_normal((n_modes, n_wl, 1))
        + 1j * rng.standard_normal((n_modes, n_wl, 1))
    ).astype(np.complex128)

    params = (0.0,) * _n_param_total(n_modes)
    stage = _stage(
        _context(tmp_path),
        n_modes=n_modes,
        params=params,
        wl_design_um=1.65,
    )
    state = _state(
        wavelength_grid_um=(1.65,),
        n_modes=n_modes,
        amplitudes=amplitudes,
    )

    result = stage.apply(state)
    out = result.field.amplitudes
    assert out is not None
    assert float(np.max(np.abs(out - amplitudes))) < 1e-12


def test_kernel_mzi_mesh_wavelength_rescaling_matches_clements(
    tmp_path: Path,
) -> None:
    """Spec (e): at wl != wl_design, the realized unitary equals
    reconstruct_clements with phi and alpha scaled by wl_design / wl_target;
    theta is NOT scaled."""
    n_modes = 4
    wavelength_grid_um = (1.5, 1.65, 1.8)
    params = _random_params(n_modes, seed=3)
    stage = _stage(
        _context(tmp_path),
        n_modes=n_modes,
        params=tuple(params),
        wl_design_um=1.65,
    )

    n_mzi = n_modes * (n_modes - 1) // 2
    theta = params[0:n_mzi]
    phi = params[n_mzi : 2 * n_mzi]
    alpha = params[2 * n_mzi : 2 * n_mzi + n_modes]
    order = [
        (lower - 1, lower)
        for col in range(n_modes - 1)
        for lower in range(n_modes - 1, col, -1)
    ]

    # Probe one wavelength bin at a time using delta inputs (canonical basis):
    # column k of U is the stage output for input e_k.
    for wl_target_um in wavelength_grid_um:
        scale = 1.65 / wl_target_um
        expected = reconstruct_clements(
            ClementsDecomposition(
                theta=theta,
                phi=phi * scale,
                alpha=alpha * scale,
                order=order,
                n_modes=n_modes,
            )
        )
        realized = np.empty((n_modes, n_modes), dtype=np.complex128)
        for k in range(n_modes):
            amplitudes = np.zeros((n_modes, 1, 1), dtype=np.complex128)
            amplitudes[k, 0, 0] = 1.0
            state = _state(
                wavelength_grid_um=(wl_target_um,),
                n_modes=n_modes,
                amplitudes=amplitudes,
            )
            out = stage.apply(state).field.amplitudes
            assert out is not None
            realized[:, k] = out[:, 0, 0]
        np.testing.assert_allclose(realized, expected, atol=1e-12, rtol=0.0)


def test_kernel_mzi_mesh_parametric_chromaticity_matches_clements(
    tmp_path: Path,
) -> None:
    """Parametric mode applies theta offsets plus dispersive phase scaling."""

    n_modes = 4
    wavelength_grid_um = (1.5, 1.65, 1.8)
    wl_design_um = 1.65
    theta_slope = 0.4
    theta_quad = -0.2
    n_eff_ref = 2.4
    n_group = 4.0
    params = _random_params(n_modes, seed=12)
    stage = _stage(
        _context(tmp_path),
        n_modes=n_modes,
        params=tuple(params),
        wl_design_um=wl_design_um,
        chip_chromaticity_model="parametric",
        theta_chromaticity_slope_rad_per_frac=theta_slope,
        theta_chromaticity_quadratic_rad_per_frac2=theta_quad,
        phase_chromaticity_model="fixed_opd_neff",
        phase_neff_ref=n_eff_ref,
        phase_group_index=n_group,
    )

    n_mzi = n_modes * (n_modes - 1) // 2
    theta = params[0:n_mzi]
    phi = params[n_mzi : 2 * n_mzi]
    alpha = params[2 * n_mzi : 2 * n_mzi + n_modes]
    order = [
        (lower - 1, lower)
        for col in range(n_modes - 1)
        for lower in range(n_modes - 1, col, -1)
    ]

    state = _state(
        wavelength_grid_um=wavelength_grid_um,
        n_modes=n_modes,
        amplitudes=np.ones(
            (n_modes, len(wavelength_grid_um), 1),
            dtype=np.complex128,
        ),
    )
    result = stage.apply(state)
    u_stack = result.results.components["chip.kernel_mzi_mesh"]["u_stack"]

    for wavelength_index, wavelength_um in enumerate(wavelength_grid_um):
        detuning = wavelength_um / wl_design_um - 1.0
        n_eff = n_eff_ref + (n_eff_ref - n_group) * detuning
        phase_scale = (wl_design_um / wavelength_um) * (n_eff / n_eff_ref)
        theta_offset = theta_slope * detuning + theta_quad * detuning * detuning
        expected = reconstruct_clements(
            ClementsDecomposition(
                theta=theta + theta_offset,
                phi=phi * phase_scale,
                alpha=alpha * phase_scale,
                order=order,
                n_modes=n_modes,
            )
        )
        np.testing.assert_allclose(
            u_stack[:, :, wavelength_index],
            expected,
            atol=1.0e-12,
            rtol=0.0,
        )


def test_kernel_mzi_mesh_rejects_chromatic_knobs_in_legacy_mode(
    tmp_path: Path,
) -> None:
    n_modes = 4
    with pytest.raises(ValueError, match="chip_chromaticity_model='parametric'"):
        _stage(
            _context(tmp_path),
            n_modes=n_modes,
            params=tuple(_random_params(n_modes, seed=15)),
            theta_chromaticity_slope_rad_per_frac=0.1,
        )


def test_kernel_mzi_mesh_dac_quantizes_then_rescales(tmp_path: Path) -> None:
    """Spec (f): DAC quantization is applied at the design wavelength BEFORE
    per-wavelength rescaling. Mid-step params and an off-design wl confirm
    that quantize-then-rescale (correct) differs from rescale-then-quantize."""
    n_modes = 4
    n_mzi = n_modes * (n_modes - 1) // 2
    n_total = _n_param_total(n_modes)
    wl_design_um = 1.65
    wl_target_um = 1.50
    dac_bits = 8
    step = 2.0 * np.pi / (1 << dac_bits)
    # Deliberately off-grid: half a quantization step away from a grid value.
    base = 1.234
    raw_params = np.full(n_total, base + 0.5 * step, dtype=np.float64)

    stage = _stage(
        _context(tmp_path),
        n_modes=n_modes,
        params=tuple(raw_params),
        wl_design_um=wl_design_um,
        dac_bits=dac_bits,
    )

    quantized = step * np.round(raw_params / step)
    theta_q = quantized[0:n_mzi]
    phi_q = quantized[n_mzi : 2 * n_mzi]
    alpha_q = quantized[2 * n_mzi : 2 * n_mzi + n_modes]
    scale = wl_design_um / wl_target_um
    order = [
        (lower - 1, lower)
        for col in range(n_modes - 1)
        for lower in range(n_modes - 1, col, -1)
    ]
    expected_correct = reconstruct_clements(
        ClementsDecomposition(
            theta=theta_q,
            phi=phi_q * scale,
            alpha=alpha_q * scale,
            order=order,
            n_modes=n_modes,
        )
    )
    # Rescale first, then quantize: confirms the orderings are not equivalent
    # for these params (we want the realized unitary to track the FIRST one).
    rescale_first = quantized.copy()
    rescale_first[n_mzi : 2 * n_mzi] = phi_q * scale
    rescale_first[2 * n_mzi : 2 * n_mzi + n_modes] = alpha_q * scale
    rescale_first_quant = step * np.round(rescale_first / step)
    wrong = reconstruct_clements(
        ClementsDecomposition(
            theta=rescale_first_quant[0:n_mzi],
            phi=rescale_first_quant[n_mzi : 2 * n_mzi],
            alpha=rescale_first_quant[2 * n_mzi : 2 * n_mzi + n_modes],
            order=order,
            n_modes=n_modes,
        )
    )
    assert float(np.max(np.abs(expected_correct - wrong))) > 1e-6, (
        "Test is degenerate: quantize-then-rescale matches rescale-then-quantize "
        "for these params; pick params strictly between quantization levels."
    )

    realized = np.empty((n_modes, n_modes), dtype=np.complex128)
    for k in range(n_modes):
        amplitudes = np.zeros((n_modes, 1, 1), dtype=np.complex128)
        amplitudes[k, 0, 0] = 1.0
        state = _state(
            wavelength_grid_um=(wl_target_um,),
            n_modes=n_modes,
            amplitudes=amplitudes,
        )
        out = stage.apply(state).field.amplitudes
        assert out is not None
        realized[:, k] = out[:, 0, 0]
    np.testing.assert_allclose(realized, expected_correct, atol=1e-12, rtol=0.0)


def test_kernel_mzi_mesh_insertion_loss_default_scales_field_amplitudes_only(
    tmp_path: Path,
) -> None:
    """Default ``insertion_loss_db=-2.0`` applies a wavelength-flat post-unitary loss.

    Verifies (a) field amplitudes are scaled by sqrt(t) relative to a
    lossless control with identical unitary, (b) cumulative
    ``transmission_fraction`` passes through UNCHANGED — chip losses
    live in field amplitudes per the detector's ``|field|² ×
    transmission_fraction`` convention, multiplying both would
    double-count — and (c) the throughput component payload still
    records the literal dB value and per-bin ``t`` factor for the
    loss-budget plot.
    """

    import math

    n_modes = 4
    wavelength_grid_um = (1.5, 1.65, 1.8)
    params = _random_params(n_modes, seed=11)

    # Build a lossy stage by directly setting insertion_loss_db.
    lossy_spec = {
        "n_modes": n_modes,
        "n_bright_ports": n_modes - 1,
        "n_dark_ports": 1,
        "wl_design_um": 1.65,
        "params": tuple(float(p) for p in params),
        "insertion_loss_db": -2.0,
    }
    lossy_stage = KernelMZIMesh(lossy_spec, _context(tmp_path, seed=11))
    lossless_stage = _stage(
        _context(tmp_path, seed=11),
        n_modes=n_modes,
        params=tuple(params),
        wl_design_um=1.65,
    )

    state = _state(wavelength_grid_um=wavelength_grid_um, n_modes=n_modes)
    upstream_transmission = state.throughput.transmission_fraction
    lossy_result = lossy_stage.apply(state)
    lossless_result = lossless_stage.apply(state)

    t = 10.0 ** (-2.0 / 10.0)
    sqrt_t = math.sqrt(t)

    np.testing.assert_allclose(
        lossy_result.field.amplitudes,
        lossless_result.field.amplitudes * sqrt_t,
        rtol=1.0e-12,
    )
    # Cumulative transmission must pass through unchanged.
    np.testing.assert_allclose(
        lossy_result.throughput.transmission_fraction,
        upstream_transmission,
        rtol=1.0e-12,
    )
    matching = [
        (name, payload)
        for name, payload in lossy_result.throughput.components.items()
        if name.startswith("chip.kernel_mzi_mesh_loss")
    ]
    assert len(matching) == 1
    _, payload = matching[0]
    assert payload["insertion_loss_db"] == -2.0
    np.testing.assert_allclose(
        payload["transmission_fraction"], (t,) * len(wavelength_grid_um), rtol=1.0e-12,
    )


def test_kernel_mzi_mesh_insertion_loss_does_not_change_null_depth(
    tmp_path: Path,
) -> None:
    """Scalar post-unitary loss commutes with port-power ratios.

    Adding chip insertion loss must NOT change the chip's null depth
    (a ratio of dark/bright port powers) — only their absolute scale.
    This is the formal justification for declaring ``insertion_loss_db``
    after the optimizer in the pipeline order.
    """

    n_modes = 4
    n_bright = 3
    n_dark = 1
    params = _random_params(n_modes, seed=7)
    lossless = KernelMZIMesh(
        {
            "n_modes": n_modes,
            "n_bright_ports": n_bright,
            "n_dark_ports": n_dark,
            "wl_design_um": 1.65,
            "params": tuple(float(p) for p in params),
            "insertion_loss_db": 0.0,
        },
        _context(tmp_path, seed=7),
    )
    lossy = KernelMZIMesh(
        {
            "n_modes": n_modes,
            "n_bright_ports": n_bright,
            "n_dark_ports": n_dark,
            "wl_design_um": 1.65,
            "params": tuple(float(p) for p in params),
            "insertion_loss_db": -2.0,
        },
        _context(tmp_path, seed=7),
    )
    state = _state(n_modes=n_modes)

    def _null_depth(stage: KernelMZIMesh) -> float:
        out = stage.apply(state).field.amplitudes
        power = np.sum(np.abs(out) ** 2, axis=(1, 2))
        return float(power[n_bright:].sum() / power[:n_bright].sum())

    assert _null_depth(lossless) == pytest.approx(_null_depth(lossy), rel=1.0e-12)


def test_kernel_mzi_mesh_rejects_positive_insertion_loss_db() -> None:
    """``insertion_loss_db`` is a signed dB transmission and must be ≤ 0."""

    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="insertion_loss_db"):
        KernelMZIMesh.Params(
            n_modes=4,
            n_bright_ports=3,
            n_dark_ports=1,
            wl_design_um=1.65,
            params=tuple(0.0 for _ in range(2 * 6 + 4)),
            insertion_loss_db=1.0,
        )



# ----------------------------------------------------------------------
# Sub-pass B: chip_optimization → chip wiring tests
# ----------------------------------------------------------------------


def _stage_with_source(
    context: RunContext,
    *,
    n_modes: int,
    optimization_source: str | None,
    n_bright_ports: int | None = None,
    n_dark_ports: int | None = None,
    n_spectro_ports: int = 0,
    wl_design_um: float = 1.65,
    deployment_mode: str = "shared",
) -> KernelMZIMesh:
    if n_bright_ports is None:
        n_bright_ports = n_modes - 1
    if n_dark_ports is None:
        n_dark_ports = n_modes - n_bright_ports
    spec: dict = {
        "n_modes": n_modes,
        "n_bright_ports": n_bright_ports,
        "n_dark_ports": n_dark_ports,
        "n_spectro_ports": n_spectro_ports,
        "wl_design_um": wl_design_um,
        "insertion_loss_db": 0.0,
        "deployment_mode": deployment_mode,
    }
    if optimization_source is not None:
        spec["optimization_source"] = optimization_source
    return KernelMZIMesh(spec, context)


def _state_with_components(
    components: dict,
    *,
    wavelength_grid_um: tuple[float, ...] = (1.65,),
    n_modes: int = 4,
    amplitudes: np.ndarray | None = None,
) -> SimulationState:
    """Build a state whose ``results.components`` is preseeded for the chip lookup."""
    from nullsim.pipeline import Results

    base = _state(
        wavelength_grid_um=wavelength_grid_um,
        n_modes=n_modes,
        amplitudes=amplitudes,
    )
    return replace(base, results=Results(components=components))


def test_kernel_mzi_mesh_optimization_source_auto_finds_upstream(tmp_path: Path) -> None:
    """`optimization_source='auto'` consumes the first chip_optimization.* result."""
    n_modes = 4
    params_tuple = tuple(_random_params(n_modes, seed=2))
    components = {
        "chip_optimization": {
            "optimal_chip_params": params_tuple,
            "optimization_loss_final": 1e-7,
            "optimization_iterations": 42,
            "backend_name": "scipy",
        }
    }
    stage = _stage_with_source(
        _context(tmp_path),
        n_modes=n_modes,
        optimization_source="auto",
        wl_design_um=1.65,
    )
    state = _state_with_components(components, n_modes=n_modes)

    # Compare against an explicit-params control run.
    control = _stage(
        _context(tmp_path),
        n_modes=n_modes,
        params=params_tuple,
        wl_design_um=1.65,
    )
    control_state = _state(n_modes=n_modes)

    actual = stage.apply(state).field.amplitudes
    expected = control.apply(control_state).field.amplitudes
    assert actual is not None and expected is not None
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.0)


def test_kernel_mzi_mesh_shared_mode_keeps_arm_phase_mismatch(
    tmp_path: Path,
) -> None:
    """Default shared deployment preserves the historical one-unitary path."""

    n_modes = 2
    wl_um = 1.25
    phases = np.array(
        [
            [0.0, np.pi / 3.0],
            [np.pi / 2.0, -np.pi / 4.0],
        ],
        dtype=np.float64,
    )
    amplitudes = np.exp(1j * phases)[:, np.newaxis, :] / np.sqrt(n_modes)
    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)
    components = {
        "chip_optimization": {
            "optimal_chip_params": (0.0, 0.0, 0.0, 0.0),
            "optimal_u_stack": hadamard[:, :, np.newaxis],
            "optimization_loss_final": 1e-12,
            "optimization_iterations": 1,
            "backend_name": "test",
        }
    }
    state = _state_with_components(
        components,
        wavelength_grid_um=(wl_um,),
        n_modes=n_modes,
        amplitudes=amplitudes,
    )
    stage = _stage_with_source(
        _context(tmp_path),
        n_modes=n_modes,
        n_bright_ports=1,
        n_dark_ports=1,
        optimization_source="auto",
        wl_design_um=wl_um,
    )

    result = stage.apply(state)
    output = result.field.amplitudes
    assert output is not None

    assert float(np.max(np.abs(output[1, 0, :]))) > 0.2
    payload = result.results.components["chip.kernel_mzi_mesh"]
    assert payload["deployment_mode"] == "shared"
    assert payload["active_chip_count"] == 1
    assert "u_stack_per_arm" not in payload


def test_kernel_mzi_mesh_per_pol_arm_trims_independent_input_phases(
    tmp_path: Path,
) -> None:
    """NAYRA arm chips independently trim static pre-chip arm phases."""

    n_modes = 2
    wl_um = 1.25
    phases = np.array(
        [
            [0.0, np.pi / 3.0],
            [np.pi / 2.0, -np.pi / 4.0],
        ],
        dtype=np.float64,
    )
    amplitudes = np.exp(1j * phases)[:, np.newaxis, :] / np.sqrt(n_modes)
    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)
    components = {
        "instrument.nayra_six_chip": {
            "channels": (
                {"channel_id": "J_A", "is_primary_band": True},
                {"channel_id": "J_B", "is_primary_band": True},
            ),
        },
        "chip_optimization": {
            "optimal_chip_params": (0.0, 0.0, 0.0, 0.0),
            "optimal_u_stack": hadamard[:, :, np.newaxis],
            "optimization_loss_final": 1e-12,
            "optimization_iterations": 1,
            "backend_name": "test",
        },
    }
    state = _state_with_components(
        components,
        wavelength_grid_um=(wl_um,),
        n_modes=n_modes,
        amplitudes=amplitudes,
    )
    stage = _stage_with_source(
        _context(tmp_path),
        n_modes=n_modes,
        n_bright_ports=1,
        n_dark_ports=1,
        optimization_source="auto",
        wl_design_um=wl_um,
        deployment_mode="per_pol_arm",
    )

    result = stage.apply(state)
    output = result.field.amplitudes
    assert output is not None

    np.testing.assert_allclose(output[1, 0, :], 0.0, atol=1.0e-12, rtol=0.0)
    payload = result.results.components["chip.kernel_mzi_mesh"]
    assert payload["deployment_mode"] == "per_pol_arm"
    assert payload["active_chip_count"] == 2
    assert payload["chip_channel_labels"] == ("J_A", "J_B")
    assert np.asarray(payload["u_stack"]).shape == (2, 2, 1)
    assert np.asarray(payload["u_stack_per_arm"]).shape == (2, 2, 2, 1)


def test_kernel_mzi_mesh_per_pol_arm_requires_two_arms(tmp_path: Path) -> None:
    n_modes = 2
    wl_um = 1.25
    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)
    components = {
        "chip_optimization": {
            "optimal_chip_params": (0.0, 0.0, 0.0, 0.0),
            "optimal_u_stack": hadamard[:, :, np.newaxis],
            "optimization_loss_final": 1e-12,
            "optimization_iterations": 1,
            "backend_name": "test",
        }
    }
    state = _state_with_components(
        components,
        wavelength_grid_um=(wl_um,),
        n_modes=n_modes,
        amplitudes=np.ones((n_modes, 1, 1), dtype=np.complex128),
    )
    stage = _stage_with_source(
        _context(tmp_path),
        n_modes=n_modes,
        n_bright_ports=1,
        n_dark_ports=1,
        optimization_source="auto",
        wl_design_um=wl_um,
        deployment_mode="per_pol_arm",
    )

    with pytest.raises(ValueError, match="n_pol=1"):
        stage.apply(state)


def test_kernel_mzi_mesh_optimizer_source_requires_results_dependency(
    tmp_path: Path,
) -> None:
    """Validation must see the optimizer-results handoff when params select it."""

    class FieldThroughputProducer(BaseStage):
        type_name = "tests.kernel_mzi.field_throughput_producer"
        family = "test"
        produces = frozenset({"field", "throughput"})

        def apply(self, state: SimulationState) -> SimulationState:
            return state

    producer = FieldThroughputProducer({}, _context(tmp_path))
    n_modes = 4
    explicit_stage = _stage(
        _context(tmp_path),
        n_modes=n_modes,
        params=tuple(_random_params(n_modes, seed=10)),
    )
    validate_dependencies([producer, explicit_stage])

    optimizer_backed_stage = _stage_with_source(
        _context(tmp_path),
        n_modes=n_modes,
        optimization_source="auto",
    )
    with pytest.raises(NullsimConfigError, match="results"):
        validate_dependencies([producer, optimizer_backed_stage])


def test_kernel_mzi_mesh_optimization_source_exact_key_lookup(tmp_path: Path) -> None:
    """An exact `optimization_source` string resolves to that specific component."""
    n_modes = 4
    params_a = tuple(_random_params(n_modes, seed=3))
    params_b = tuple(_random_params(n_modes, seed=4))
    components = {
        "chip_optimization": {
            "optimal_chip_params": params_a,
            "optimization_loss_final": 1.0,
            "optimization_iterations": 1,
            "backend_name": "scipy",
        },
        "chip_optimization.chip_opt_b": {
            "optimal_chip_params": params_b,
            "optimization_loss_final": 1e-8,
            "optimization_iterations": 100,
            "backend_name": "scipy",
        },
    }
    stage = _stage_with_source(
        _context(tmp_path),
        n_modes=n_modes,
        optimization_source="chip_optimization.chip_opt_b",
        wl_design_um=1.65,
    )
    state = _state_with_components(components, n_modes=n_modes)

    control = _stage(
        _context(tmp_path),
        n_modes=n_modes,
        params=params_b,
        wl_design_um=1.65,
    )
    control_state = _state(n_modes=n_modes)

    actual = stage.apply(state).field.amplitudes
    expected = control.apply(control_state).field.amplitudes
    assert actual is not None and expected is not None
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.0)


def test_kernel_mzi_mesh_no_params_no_source_rejected_at_construction(
    tmp_path: Path,
) -> None:
    """Constructing without params and without optimization_source raises."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="optimization_source"):
        KernelMZIMesh(
            {
                "n_modes": 4,
                "n_bright_ports": 3,
                "n_dark_ports": 1,
                "wl_design_um": 1.65,
            },
            _context(tmp_path),
        )


def test_kernel_mzi_mesh_optimization_source_missing_key_raises(tmp_path: Path) -> None:
    """A literal `optimization_source` that is not present raises KeyError on apply."""
    n_modes = 4
    stage = _stage_with_source(
        _context(tmp_path),
        n_modes=n_modes,
        optimization_source="nonexistent.key",
        wl_design_um=1.65,
    )
    state = _state(n_modes=n_modes)

    with pytest.raises(KeyError, match="nonexistent.key"):
        stage.apply(state)


def test_kernel_mzi_mesh_optimization_source_auto_no_match_raises(tmp_path: Path) -> None:
    """`optimization_source='auto'` with no chip_optimization.* component raises."""
    n_modes = 4
    stage = _stage_with_source(
        _context(tmp_path),
        n_modes=n_modes,
        optimization_source="auto",
        wl_design_um=1.65,
    )
    state = _state(n_modes=n_modes)

    with pytest.raises(KeyError, match="auto"):
        stage.apply(state)


def test_kernel_mzi_mesh_resolved_params_length_mismatch_raises(tmp_path: Path) -> None:
    """An upstream payload with wrong-length params fails apply()."""
    n_modes = 4
    components = {
        "chip_optimization": {
            "optimal_chip_params": (0.1, 0.2, 0.3),  # too short
            "optimization_loss_final": 1e-3,
            "optimization_iterations": 1,
            "backend_name": "scipy",
        }
    }
    stage = _stage_with_source(
        _context(tmp_path),
        n_modes=n_modes,
        optimization_source="auto",
        wl_design_um=1.65,
    )
    state = _state_with_components(components, n_modes=n_modes)

    with pytest.raises(ValueError, match="length"):
        stage.apply(state)


@pytest.mark.stats
def test_chip_optimization_scipy_then_kernel_mzi_mesh_yields_deep_null(
    tmp_path: Path,
) -> None:
    """Cross-stage smoke: optimizer → chip with `auto` lookup → very small dark-port power."""
    from nullsim.stages.chip_optimization.scipy import ChipOptimizationScipy

    n_modes = 4
    n_dark_ports = 1
    n_bright_ports = n_modes - n_dark_ports
    wl_um = 1.65

    # Build a uniform-illumination starlight field: chip_optimization optimizes
    # broadband leakage on the input field exactly, so the chip applied to the
    # same field should produce |dark|^2 ~ scipy's reported loss.
    amplitudes = np.ones((n_modes, 1, 1), dtype=np.complex128) / np.sqrt(n_modes)
    state = _state(
        wavelength_grid_um=(wl_um,),
        n_modes=n_modes,
        amplitudes=amplitudes,
    )

    optimizer = ChipOptimizationScipy(
        {
            "n_modes": n_modes,
            "n_bright_ports": n_bright_ports,
            "n_dark_ports": n_dark_ports,
            "wl_design_um": wl_um,
            "max_iter": 500,
            "seed": 0,
            "target_null_depth": 1e-6,
        },
        _context(tmp_path, seed=0),
    )
    optimized_state = optimizer.apply(state)

    chip = _stage_with_source(
        _context(tmp_path),
        n_modes=n_modes,
        optimization_source="auto",
        n_bright_ports=n_bright_ports,
        n_dark_ports=n_dark_ports,
        wl_design_um=wl_um,
    )
    out_state = chip.apply(optimized_state)
    out = out_state.field.amplitudes
    assert out is not None

    # Dark-port indices: last n_dark_ports modes.
    dark = out[n_bright_ports:, :, :]
    bright = out[:n_bright_ports, :, :]
    dark_power = float(np.sum(np.abs(dark) ** 2))
    bright_power = float(np.sum(np.abs(bright) ** 2))
    total = dark_power + bright_power
    # The optimizer drives broadband leakage well below 1e-3 for m=4 mono input.
    assert dark_power / total < 1e-3, (
        f"dark/total fraction {dark_power / total:g} too high; chip wiring broken?"
    )
