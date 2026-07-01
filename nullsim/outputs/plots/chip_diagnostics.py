"""Chip diagnostics figure (KILO paperv3 Fig 7).

Four-panel chip diagnostic: per-port power distribution, optimized
unitary amplitude+phase, null depth vs DAC bit depth, broadband null
spectrum. Direct port of ``kilo/studies/chip_diagnostics.run_study``
(``chip_diagnostics.pdf`` panel layout), wired to nullsim's
``state.results.components["chip_optimization_*"]`` payload.

Chip-input field is recovered from the post-chip ``state.field`` by
inverting the unitary (``U @ inp = out`` → ``inp = U^H @ out``) so no
extra diagnostic state needs to be captured during the run.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib  # noqa: F401  # backend choice managed by styles module
import numpy as np

from nullsim.outputs.styles import _ensure_headless_backend  # noqa: F401  # side-effect

from matplotlib import pyplot as plt
from numpy.typing import NDArray

from nullsim.outputs import RunCollection, RunRecord
from nullsim.outputs.plots.subaperture_layout import _find_stage
from nullsim.pipeline import register_plot
from nullsim.stages.chip_optimization import (
    ChipOptimizationParams,
    _build_unitary_stack_numpy,
)


# ---------------------------------------------------------------------------
# Per-panel renderers (ports of the KILO ``_plot_*`` helpers).
# ---------------------------------------------------------------------------


def _plot_port_power(
    ax: plt.Axes,
    u_center: NDArray[np.complex128],
    star_field: NDArray[np.complex128],
    bright_idx: NDArray[np.int64],
    dark_idx: NDArray[np.int64],
    spectro_idx: NDArray[np.int64] | None = None,
) -> None:
    """Panel 1: per-port fractional power for the stellar input."""

    m = u_center.shape[0]
    spectro_arr = (
        np.asarray(spectro_idx, dtype=np.int64)
        if spectro_idx is not None
        else np.empty(0, dtype=np.int64)
    )
    star_output = u_center @ star_field
    star_power = np.abs(star_output) ** 2
    star_frac = star_power / np.sum(star_power) if np.sum(star_power) > 0 else star_power

    for idx in bright_idx:
        ax.axvspan(idx - 0.5, idx + 0.5, color="gold", alpha=0.10)
    for idx in dark_idx:
        ax.axvspan(idx - 0.5, idx + 0.5, color="steelblue", alpha=0.10)
    for idx in spectro_arr:
        ax.axvspan(idx - 0.5, idx + 0.5, color="mediumpurple", alpha=0.14)

    ports = np.arange(m)
    ax.bar(ports, star_frac, 0.7, color="C0")
    ax.set_yscale("log")
    ax.set_xlabel("Output port index")
    ax.set_ylabel("Fractional stellar power")
    ax.set_title("Port power distribution", pad=15)
    ax.set_xlim(-0.5, m - 0.5)
    ax.grid(True, alpha=0.2, which="both")

    for idx_arr, label, color in [
        (bright_idx, "Bright", "goldenrod"),
        (dark_idx, "Dark", "steelblue"),
        (spectro_arr, "Spectro", "mediumpurple"),
    ]:
        if idx_arr.size > 0:
            mid = float(np.mean(idx_arr)) / (m - 1) if m > 1 else 0.5
            ax.text(
                mid, 1.02, label, ha="center", va="bottom",
                fontsize=8, color=color, fontweight="bold",
                transform=ax.transAxes,
            )


def _plot_unitary(
    ax_amp: plt.Axes,
    ax_phase: plt.Axes,
    u: NDArray[np.complex128],
) -> None:
    """Panel 2: |U| + arg(U) heatmaps."""

    m = u.shape[0]
    im_amp = ax_amp.imshow(
        np.abs(u), cmap="viridis", vmin=0, vmax=float(np.max(np.abs(u))),
        aspect="equal", origin="upper",
    )
    ax_amp.set_title(r"$|U_{ij}|$")
    ax_amp.set_xlabel("Input mode")
    ax_amp.set_ylabel("Output mode")
    plt.colorbar(im_amp, ax=ax_amp, shrink=0.8)

    im_phase = ax_phase.imshow(
        np.angle(u), cmap="twilight", vmin=-np.pi, vmax=np.pi,
        aspect="equal", origin="upper",
    )
    ax_phase.set_title(r"$\arg(U_{ij})$")
    ax_phase.set_xlabel("Input mode")
    ax_phase.set_ylabel("Output mode")
    cbar = plt.colorbar(im_phase, ax=ax_phase, shrink=0.8)
    cbar.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cbar.set_ticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

    if m > 12:
        for ax in (ax_amp, ax_phase):
            ax.set_xticks(np.arange(0, m, 4))
            ax.set_yticks(np.arange(0, m, 4))


def _plot_null_vs_dac(
    ax: plt.Axes,
    opt_params: NDArray[np.float64],
    chip_params: ChipOptimizationParams,
    star_field: NDArray[np.complex128],
    null_idx: NDArray[np.int64],
    wl_design_um: float,
    *,
    null_label: str = "Null",
) -> dict:
    """Panel 3: null depth vs DAC bit precision (single-wavelength)."""

    bit_values = list(range(4, 23))
    null_depths: list[float] = []

    for bits in bit_values:
        stack = _build_unitary_stack_numpy(
            params_vec=opt_params,
            n_modes=chip_params.n_modes,
            wavelength_grid_um=(wl_design_um,),
            wl_design_um=wl_design_um,
            dac_bits=bits,
            chromaticity_params=chip_params,
        )
        u_q = stack[:, :, 0]
        out = u_q @ star_field
        total = float(np.sum(np.abs(out) ** 2))
        null = float(np.sum(np.abs(out[null_idx]) ** 2))
        null_depths.append(null / total if total > 0 else 0.0)

    # Ideal (no quantization)
    stack_ideal = _build_unitary_stack_numpy(
        params_vec=opt_params,
        n_modes=chip_params.n_modes,
        wavelength_grid_um=(wl_design_um,),
        wl_design_um=wl_design_um,
        dac_bits=None,
        chromaticity_params=chip_params,
    )
    u_ideal = stack_ideal[:, :, 0]
    out_ideal = u_ideal @ star_field
    total_ideal = float(np.sum(np.abs(out_ideal) ** 2))
    ideal_null = (
        float(np.sum(np.abs(out_ideal[null_idx]) ** 2)) / total_ideal
        if total_ideal > 0 else 0.0
    )

    ax.semilogy(bit_values, null_depths, "o-", color="C0", markersize=4, linewidth=1.5)
    ax.axhline(ideal_null, color="gray", linestyle="--", linewidth=1, label="Ideal (float64)")
    for bits, color in ((16, "C2"), (20, "C3")):
        ax.axvline(bits, color=color, linestyle=":", alpha=0.6, linewidth=1)
        ax.text(
            bits - 0.3, 0.95, f"{bits}-bit", fontsize=11, color=color,
            va="top", ha="right", transform=ax.get_xaxis_transform(),
        )
    ax.set_xlabel("DAC bits")
    ax.set_ylabel("Null depth")
    ax.set_title(f"{null_label} depth vs DAC precision")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, which="both")
    ax.set_xlim(3.5, 22.5)
    return {"bits": bit_values, "null_depths": null_depths, "ideal": ideal_null}


def _plot_broadband_null(
    ax: plt.Axes,
    opt_params: NDArray[np.float64],
    chip_params: ChipOptimizationParams,
    star_field_design: NDArray[np.complex128],
    wavelength_grid_um: tuple[float, ...],
    null_idx: NDArray[np.int64],
    deployed_u_stack: NDArray[np.complex128] | None = None,
    physical_output_fields: NDArray[np.complex128] | None = None,
    null_port_label: str = "dark",
) -> dict:
    """Panel 4: chip intrinsic and physical per-wavelength null depths.

    Applies the optimized chip unitary stack ``u_stack[:, :, k]`` to a
    CLEAN nominal stellar field (real-valued, per-mode magnitudes from
    ``star_field_design``, wavelength-flat — matching the optimizer's
    ``_require_field`` convention) and reports dark-port leakage per
    wavelength. This is the chip's intrinsic chromatic response — phase
    noise from AO / transport / delay lines does NOT enter here.

    When ``deployed_u_stack`` is provided, it's used verbatim — that's
    the chromatic (M, M, n_wl) stack the broadband optimizer actually
    evaluated and the chip stage deploys (Tier 3.1 option 2). Without
    it we fall back to rebuilding from ``opt_params`` via
    ``_build_unitary_stack_numpy``, which applies a single canonical
    param vector with per-λ phi/alpha rescaling under the
    lexicographic layout — for torch-broadband chips that yields the
    narrow-band design-λ-only null instead of the deployed broadband
    one (the F001-class chromatic mismatch).

    When ``physical_output_fields`` is provided, this panel also recovers
    the physical chip input at each wavelength via ``U(λ)^H @ E_out(λ)``
    and replots its null. That second curve keeps transport, delay-line,
    and polarization-chain chromatic phases visible instead of hiding
    them behind the clean-input chip metric.

    Pre-fix this panel read ``state.field.amplitudes`` (the post-chip
    output ports of a single realization through the full pipeline), so
    it showed the realized null with phase noise rather than the chip's
    own broadband null. That conflated chip design quality with AO
    realization noise.
    """

    wl_um = np.asarray(wavelength_grid_um, dtype=np.float64)
    if deployed_u_stack is not None:
        u_stack = deployed_u_stack
    else:
        u_stack = _build_unitary_stack_numpy(
            params_vec=opt_params,
            n_modes=chip_params.n_modes,
            wavelength_grid_um=tuple(float(w) for w in wl_um),
            wl_design_um=chip_params.wl_design_um,
            dac_bits=chip_params.dac_bits,
            chromaticity_params=chip_params,
        )

    # Clean nominal field: per-mode magnitude from the recovered
    # design-wavelength input, phases stripped, broadcast across
    # wavelengths. Matches ``_require_field`` in chip_optimization.
    per_mode_mag = np.abs(np.asarray(star_field_design)).astype(np.complex128)

    clean_nd = _null_depth_spectrum(
        u_stack,
        np.broadcast_to(per_mode_mag[:, np.newaxis], (u_stack.shape[0], len(wl_um))),
        null_idx,
    )

    ax.semilogy(
        wl_um, clean_nd, "o-", color="C0", markersize=3, linewidth=1.5,
        label="Clean input",
    )
    clean_mean = float(np.mean(clean_nd))
    ax.axhline(
        clean_mean,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"Clean mean: {clean_mean:.2e}",
    )

    physical_nd: NDArray[np.float64] | None = None
    physical_mean: float | None = None
    if physical_output_fields is not None:
        physical_inputs = _recover_physical_inputs(
            u_stack,
            physical_output_fields,
        )
        physical_nd = _null_depth_spectrum(u_stack, physical_inputs, null_idx)
        physical_mean = float(np.mean(physical_nd))
        ax.semilogy(
            wl_um,
            physical_nd,
            "s-",
            color="C3",
            markersize=3,
            linewidth=1.3,
            label="Physical pipeline input",
        )
        ax.axhline(
            physical_mean,
            color="C3",
            linestyle="--",
            linewidth=1,
            alpha=0.8,
            label=f"Physical pipeline mean: {physical_mean:.2e}",
        )

    ax.axvline(chip_params.wl_design_um, color="0.3", linestyle=":", alpha=0.5, linewidth=1)
    ax.axvspan(wl_um[0], wl_um[-1], color="C0", alpha=0.05)
    ax.set_xlabel(r"Wavelength ($\mu$m)")
    ax.set_ylabel(f"Null depth ({null_port_label} / total)")
    ax.set_title("Broadband null spectrum")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, which="both")
    result = {
        "wavelengths_um": [float(w) for w in wl_um],
        "null_depths": [float(n) for n in clean_nd],
        "mean": clean_mean,
        "clean_null_depths": [float(n) for n in clean_nd],
        "clean_mean": clean_mean,
    }
    if physical_nd is not None and physical_mean is not None:
        result["physical_null_depths"] = [float(n) for n in physical_nd]
        result["physical_mean"] = physical_mean
    return result


def _null_depth_spectrum(
    u_stack: NDArray[np.complex128],
    input_fields: NDArray[np.complex128],
    null_idx: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Return dark/total power for each wavelength in ``u_stack``."""

    fields = np.asarray(input_fields, dtype=np.complex128)
    if fields.shape != (u_stack.shape[0], u_stack.shape[2]):
        raise ValueError(
            "chip_diagnostics input_fields must have shape "
            f"({u_stack.shape[0]}, {u_stack.shape[2]}), got {fields.shape}."
        )
    null_depths = np.zeros(u_stack.shape[2], dtype=np.float64)
    for k in range(u_stack.shape[2]):
        out = u_stack[:, :, k] @ fields[:, k]
        power = np.abs(out) ** 2
        total = float(power.sum())
        if total > 0.0:
            null_depths[k] = float(power[null_idx].sum() / total)
    return null_depths


def _recover_physical_inputs(
    u_stack: NDArray[np.complex128],
    physical_output_fields: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Recover per-wavelength chip inputs from post-chip output fields."""

    outputs = np.asarray(physical_output_fields, dtype=np.complex128)
    if outputs.ndim == 3:
        outputs = outputs[:, :, 0]
    if outputs.shape != (u_stack.shape[0], u_stack.shape[2]):
        raise ValueError(
            "chip_diagnostics physical_output_fields must have shape "
            f"({u_stack.shape[0]}, {u_stack.shape[2]}) or "
            f"({u_stack.shape[0]}, {u_stack.shape[2]}, n_pol), "
            f"got {outputs.shape}."
        )
    recovered = np.empty_like(outputs, dtype=np.complex128)
    for k in range(u_stack.shape[2]):
        recovered[:, k] = u_stack[:, :, k].conj().T @ outputs[:, k]
    return recovered


def _apply_unitary_stack(
    u_stack: NDArray[np.complex128],
    input_fields: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Apply ``u_stack[:, :, k]`` to each wavelength input vector."""

    fields = np.asarray(input_fields, dtype=np.complex128)
    if fields.shape != (u_stack.shape[1], u_stack.shape[2]):
        raise ValueError(
            "chip_diagnostics input_fields must have shape "
            f"({u_stack.shape[1]}, {u_stack.shape[2]}), got {fields.shape}."
        )
    return np.einsum("oik,ik->ok", u_stack, fields, optimize=True)


def _deployed_stack_or_build(
    *,
    opt_params: NDArray[np.float64],
    chip_params: ChipOptimizationParams,
    wavelength_grid_um: tuple[float, ...],
    deployed_u_stack: NDArray[np.complex128] | None,
) -> NDArray[np.complex128]:
    """Return the deployed stack when present, otherwise rebuild it."""

    if deployed_u_stack is not None:
        stack = np.asarray(deployed_u_stack, dtype=np.complex128)
        if stack.shape == (
            chip_params.n_modes,
            chip_params.n_modes,
            len(wavelength_grid_um),
        ):
            return stack
    return _build_unitary_stack_numpy(
        params_vec=opt_params,
        n_modes=chip_params.n_modes,
        wavelength_grid_um=wavelength_grid_um,
        wl_design_um=chip_params.wl_design_um,
        dac_bits=chip_params.dac_bits,
        chromaticity_params=chip_params,
    )


# ---------------------------------------------------------------------------
# Composite figure
# ---------------------------------------------------------------------------


def plot_chip_diagnostics(
    *,
    opt_params: NDArray[np.float64],
    chip_params: ChipOptimizationParams,
    field_amplitudes: NDArray[np.complex128],
    wavelength_grid_um: tuple[float, ...],
    deployed_u_stack: NDArray[np.complex128] | None = None,
    null_idx_override: NDArray[np.int64] | None = None,
    null_port_label: str = "dark",
    title: str | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Render the 4-panel chip diagnostics figure (KILO paperv3 Fig 7).

    When ``deployed_u_stack`` is provided (torch backend / Tier 3.1
    option 2), the broadband-null panel uses it directly — that's the
    chromatic (M, M, n_wl) stack the broadband optimizer actually
    evaluated and the chip stage deploys. Rebuilding from a single
    canonical param vector via ``_build_unitary_stack_numpy`` would
    apply per-λ phi/alpha rescaling under the lexicographic layout and
    yield a different off-design chip than what was deployed (F001).
    """

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1, 1])
    ax_port = fig.add_subplot(gs[0, 0])
    ax_amp = fig.add_subplot(gs[0, 1])
    ax_phase = fig.add_subplot(gs[0, 2])
    ax_dac = fig.add_subplot(gs[1, 0])
    ax_bb = fig.add_subplot(gs[1, 1:])

    # Pick the closest wavelength bin to wl_design.
    wl_arr = np.asarray(wavelength_grid_um, dtype=np.float64)
    design_bin = int(np.argmin(np.abs(wl_arr - chip_params.wl_design_um)))

    # Prefer the deployed chromatic stack when present; otherwise
    # rebuild monochromatically at wl_design from canonical params.
    if deployed_u_stack is not None:
        u_design = deployed_u_stack[:, :, design_bin]
    else:
        stack = _build_unitary_stack_numpy(
            params_vec=opt_params,
            n_modes=chip_params.n_modes,
            wavelength_grid_um=(chip_params.wl_design_um,),
            wl_design_um=chip_params.wl_design_um,
            dac_bits=chip_params.dac_bits,
            chromaticity_params=chip_params,
        )
        u_design = stack[:, :, 0]

    # Recover the chip-input stellar field at the design wavelength from
    # the post-chip ``state.field`` via U^H @ output. Unitary: U^H @ U = I.
    if field_amplitudes.ndim == 3:
        out_design = field_amplitudes[:, design_bin, 0]
    else:
        out_design = field_amplitudes[:, design_bin]
    star_field_design = u_design.conj().T @ out_design

    bright_idx = np.arange(0, chip_params.n_bright_ports, dtype=np.int64)
    dark_start = chip_params.n_bright_ports
    dark_stop = dark_start + chip_params.n_dark_ports
    dark_idx = np.arange(dark_start, dark_stop, dtype=np.int64)
    spectro_idx = np.arange(dark_stop, chip_params.n_modes, dtype=np.int64)
    if null_idx_override is None:
        # Detection-mode null set is dark + spectro.
        null_idx = np.concatenate((dark_idx, spectro_idx))
        active_null_port_label = (
            "dark+spectro" if spectro_idx.size > 0 else null_port_label
        )
    else:
        null_idx = np.asarray(null_idx_override, dtype=np.int64)
        active_null_port_label = null_port_label

    _plot_port_power(
        ax_port,
        u_design,
        star_field_design,
        bright_idx,
        dark_idx,
        spectro_idx,
    )
    _plot_unitary(ax_amp, ax_phase, u_design)
    _plot_null_vs_dac(
        ax_dac, opt_params, chip_params, star_field_design, null_idx,
        wl_design_um=chip_params.wl_design_um,
        null_label=active_null_port_label.capitalize(),
    )
    _plot_broadband_null(
        ax_bb,
        opt_params=opt_params,
        chip_params=chip_params,
        star_field_design=star_field_design,
        wavelength_grid_um=wavelength_grid_um,
        null_idx=null_idx,
        deployed_u_stack=deployed_u_stack,
        physical_output_fields=field_amplitudes,
        null_port_label=active_null_port_label,
    )

    # Title reflects the disjoint port partition. ``n_spectro_ports``
    # only appears in the chip-optimization stage's view of the layout;
    # downstream stages roll spectro into dark for photon routing.
    spectro = chip_params.n_spectro_ports
    spectro_suffix = f" + {spectro} spectro" if spectro > 0 else ""
    if title is None:
        title = (
            f"Chip Diagnostics (M = {chip_params.n_modes}, "
            f"{chip_params.n_bright_ports} bright / "
            f"{chip_params.n_dark_ports} dark{spectro_suffix})"
        )
    fig.suptitle(title, fontsize=14, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if save_path is not None:
        fig.savefig(save_path)
    return fig


# ---------------------------------------------------------------------------
# Registered-plot wrapper
# ---------------------------------------------------------------------------


def _extract_chip_optimization(
    record: RunRecord,
) -> tuple[
    NDArray[np.float64],
    ChipOptimizationParams,
    NDArray[np.complex128] | None,
] | None:
    """Pull the optimal params + ChipOptimizationParams + (optional)
    precomputed chromatic ``optimal_u_stack`` for a given record.

    The torch backend (Tier 3.1 option 2) ships ``optimal_u_stack`` —
    the (M, M, n_wl) chromatic stack the broadband optimizer actually
    evaluated and the chip stage deploys verbatim. Diagnostics that
    need to reproduce the deployed chip MUST use this stack rather
    than rebuilding from ``optimal_chip_params`` via
    ``_build_unitary_stack_numpy``: the latter applies a single param
    vector with per-λ phi/alpha rescaling under the lexicographic
    Clements layout, which is NOT the same chip as the broadband-
    refined chromatic stack (F001 / Tier 3.1).
    """

    components: dict[str, Any] = record.result.state.results.components or {}
    chip_opt_keys = [k for k in components if k.startswith("chip_optimization")]
    if not chip_opt_keys:
        return None
    source_key = _deployed_chip_optimization_source(record, components, chip_opt_keys)
    payload = components[source_key]
    opt_params = np.asarray(payload["optimal_chip_params"], dtype=np.float64)
    u_stack_raw = payload.get("optimal_u_stack")
    u_stack = (
        np.asarray(u_stack_raw, dtype=np.complex128)
        if u_stack_raw is not None
        else None
    )

    # Find the matching ChipOptimizationParams in the resolved config.
    chip_opt_stage = _find_stage(record.config, "chip_optimization.scipy") \
        or _find_stage(record.config, "chip_optimization.torch")
    if chip_opt_stage is not None:
        extra = getattr(chip_opt_stage, "model_extra", None) or {}
        chip_params = ChipOptimizationParams(
            **{k: v for k, v in extra.items() if k != "id" and k != "type"}
        )
        return opt_params, chip_params, u_stack

    # HWO can use chip_optimization.ideal_kernel as a center-wavelength
    # seed and deploy the resulting physical Clements params through the
    # downstream chip stage. In that case the chromaticity knobs live on
    # chip.kernel_mzi_mesh, not on the optimizer stage.
    ideal_stage = _find_stage(record.config, "chip_optimization.ideal_kernel")
    chip_stage = _find_stage(record.config, "chip.kernel_mzi_mesh")
    if ideal_stage is None or chip_stage is None:
        return None
    allowed = set(ChipOptimizationParams.model_fields)
    payload_params: dict[str, Any] = {}
    for stage in (ideal_stage, chip_stage):
        extra = getattr(stage, "model_extra", None) or {}
        payload_params.update(
            {
                str(k): v
                for k, v in extra.items()
                if str(k) in allowed
            }
        )
    chip_params = ChipOptimizationParams(**payload_params)
    return opt_params, chip_params, u_stack


def _deployed_chip_optimization_source(
    record: RunRecord,
    components: dict[str, Any],
    chip_opt_keys: list[str],
) -> str:
    """Return the chip-optimization component deployed by chip.kernel_mzi_mesh."""

    chip_stage = _find_stage(record.config, "chip.kernel_mzi_mesh")
    if chip_stage is not None:
        extra = getattr(chip_stage, "model_extra", None) or {}
        source = extra.get("optimization_source")
        if isinstance(source, str):
            if source == "auto":
                if len(chip_opt_keys) == 1:
                    return chip_opt_keys[0]
            elif source in components:
                return source
    return chip_opt_keys[-1]


def _extract_characterization_spectro(
    record: RunRecord,
) -> tuple[
    NDArray[np.float64],
    ChipOptimizationParams,
    NDArray[np.complex128],
    NDArray[np.int64],
    tuple[float, ...],
] | None:
    """Pull the characterization spectro chip payload for diagnostics."""

    components: dict[str, Any] = record.result.state.results.components or {}
    char_keys = [k for k in components if k.startswith("characterization.spectro")]
    if not char_keys:
        return None
    payload = components[char_keys[-1]]
    raw_stack = payload.get("optimal_u_stack")
    if raw_stack is None:
        return None
    u_stack = np.asarray(raw_stack, dtype=np.complex128)
    if u_stack.ndim != 3 or u_stack.shape[0] != u_stack.shape[1]:
        return None
    n_modes = int(u_stack.shape[0])

    bright_idx = np.asarray(payload.get("bright_indices", ()), dtype=np.int64)
    dark_idx = np.asarray(payload.get("dark_indices", ()), dtype=np.int64)
    spectro_idx = np.asarray(payload.get("spectro_indices", ()), dtype=np.int64)
    if spectro_idx.size == 0:
        return None
    if bright_idx.size + dark_idx.size + spectro_idx.size != n_modes:
        return None

    opt_params = np.asarray(payload.get("optimal_chip_params", ()), dtype=np.float64)
    if opt_params.size == 0:
        return None

    wavelengths_m = np.asarray(payload.get("wavelengths_m", ()), dtype=np.float64)
    if wavelengths_m.size != u_stack.shape[2]:
        raw_broadband = np.asarray(
            payload.get("broadband_wavelengths_m", ()), dtype=np.float64
        )
        if raw_broadband.size == u_stack.shape[2]:
            wavelengths_m = raw_broadband
    if wavelengths_m.size != u_stack.shape[2]:
        return None
    wavelength_grid_um = tuple(float(w * 1.0e6) for w in wavelengths_m)

    wl_design_um = _characterization_wl_design_um(record, payload, wavelength_grid_um)
    dac_bits = _characterization_dac_bits(record)
    chip_params = ChipOptimizationParams(
        n_modes=n_modes,
        n_bright_ports=int(bright_idx.size),
        n_dark_ports=int(dark_idx.size),
        n_spectro_ports=int(spectro_idx.size),
        wl_design_um=wl_design_um,
        dac_bits=dac_bits,
    )
    return opt_params, chip_params, u_stack, spectro_idx, wavelength_grid_um


def _characterization_wl_design_um(
    record: RunRecord,
    payload: dict[str, Any],
    wavelength_grid_um: tuple[float, ...],
) -> float:
    """Resolve the characterization design wavelength in microns."""

    raw_band_lambda = payload.get("band_lambda_m")
    if raw_band_lambda is not None:
        return float(raw_band_lambda) * 1.0e6
    stage = _find_stage(record.config, "characterization.spectro")
    if stage is not None:
        extra = getattr(stage, "model_extra", None) or {}
        raw_design = extra.get("wl_design_um")
        if raw_design is not None:
            return float(raw_design)
    return float(wavelength_grid_um[len(wavelength_grid_um) // 2])


def _characterization_dac_bits(record: RunRecord) -> int | None:
    """Return the characterization-stage DAC setting, if configured."""

    stage = _find_stage(record.config, "characterization.spectro")
    if stage is None:
        return None
    extra = getattr(stage, "model_extra", None) or {}
    raw = extra.get("dac_bits")
    return None if raw is None else int(raw)


def plot_characterization_chip_diagnostics_for_record(
    record: RunRecord,
    *,
    save_path: Path | None = None,
) -> plt.Figure | None:
    """Render the characterization retune unitary diagnostic for one record.

    The live ``state.field`` is the detection-chip output. For the
    characterization diagnostic we first invert the deployed detection
    chip to recover the pre-chip pipeline input, then apply the stored
    characterization unitary stack to that same input. This keeps
    transport/delay-line chromatic phases visible without pretending
    that the characterization chip was the pipeline-deployed chip.
    """

    char_extracted = _extract_characterization_spectro(record)
    if char_extracted is None:
        return None
    char_params, char_chip_params, char_stack, spectro_idx, char_wl_um = (
        char_extracted
    )

    state = record.result.state
    if state.field.amplitudes is None:
        return None
    detection_extracted = _extract_chip_optimization(record)
    if detection_extracted is None:
        return None
    det_params, det_chip_params, det_stack_raw = detection_extracted
    field_wl_um = tuple(float(w) for w in state.field.wavelength_grid_um)
    if len(field_wl_um) != char_stack.shape[2]:
        return None
    if not np.allclose(
        np.asarray(field_wl_um),
        np.asarray(char_wl_um),
        rtol=0.0,
        atol=1.0e-9,
    ):
        return None

    det_stack = _deployed_stack_or_build(
        opt_params=det_params,
        chip_params=det_chip_params,
        wavelength_grid_um=field_wl_um,
        deployed_u_stack=det_stack_raw,
    )
    chip_inputs = _recover_physical_inputs(det_stack, state.field.amplitudes)
    char_outputs = _apply_unitary_stack(char_stack, chip_inputs)[:, :, np.newaxis]

    title = (
        f"Characterization Chip Diagnostics (M = {char_chip_params.n_modes}, "
        f"{char_chip_params.n_bright_ports} bright / "
        f"{char_chip_params.n_dark_ports} dark + "
        f"{char_chip_params.n_spectro_ports} spectro)"
    )
    return plot_chip_diagnostics(
        opt_params=char_params,
        chip_params=char_chip_params,
        field_amplitudes=char_outputs,
        wavelength_grid_um=char_wl_um,
        deployed_u_stack=char_stack,
        null_idx_override=spectro_idx,
        null_port_label="spectro",
        title=title,
        save_path=save_path,
    )


@register_plot
def chip_diagnostics(
    collection: RunCollection,
    output_dir: Path,
    formats: Iterable[str],
) -> tuple[Path, ...]:
    """Registered plot: emits ``chip_diagnostics.<fmt>`` for the first record."""

    if not collection.records:
        return ()
    record = collection.records[0]
    extracted = _extract_chip_optimization(record)
    if extracted is None:
        return ()
    opt_params, chip_params, deployed_u_stack = extracted

    state = record.result.state
    if state.field.amplitudes is None:
        return ()
    wavelength_grid_um = state.field.wavelength_grid_um
    if not wavelength_grid_um:
        return ()

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig = plot_chip_diagnostics(
        opt_params=opt_params,
        chip_params=chip_params,
        field_amplitudes=state.field.amplitudes,
        wavelength_grid_um=tuple(wavelength_grid_um),
        deployed_u_stack=deployed_u_stack,
        save_path=None,
    )
    _append_record_label_to_suptitle(fig, collection, record)

    written: list[Path] = []
    for fmt in formats:
        out = plot_dir / f"chip_diagnostics.{fmt}"
        fig.savefig(out)
        written.append(out)
    plt.close(fig)
    return tuple(written)


def _append_record_label_to_suptitle(
    fig: Any,
    collection: RunCollection,
    record: Any,
) -> None:
    """Identify which sweep cell a single-record figure shows.

    The deployed chip realization is flux dependent, so an unlabeled
    records[0] (the first cell in declaration order) is ambiguous in a
    J-mag sweep (review 2026-06-10 Issue 23d).
    """

    if len(collection.records) <= 1:
        return
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is None:
        return
    from nullsim.outputs.plots._axis_labels import compact_record_label

    label = compact_record_label(record.coord_tuple, record.config.run.name)
    if label:
        suptitle.set_text(f"{suptitle.get_text()}  ·  {label}")


@register_plot
def chip_diagnostics_characterization(
    collection: RunCollection,
    output_dir: Path,
    formats: Iterable[str],
) -> tuple[Path, ...]:
    """Registered plot: emits characterization-chip diagnostics for first record."""

    if not collection.records:
        return ()
    fig = plot_characterization_chip_diagnostics_for_record(collection.records[0])
    if fig is None:
        return ()
    _append_record_label_to_suptitle(fig, collection, collection.records[0])

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        out = plot_dir / f"chip_diagnostics_characterization.{fmt}"
        fig.savefig(out)
        written.append(out)
    plt.close(fig)
    return tuple(written)
