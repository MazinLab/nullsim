"""Clements MZI-mesh chip kernel stage (SPEC §15.3, Pass 3 WP2).

This stage replaces the placeholder ``chip.identity`` for production runs:
it applies a rectangular nearest-neighbour Clements unitary to the
``n_modes`` axis of the field tensor, per wavelength bin. The unitary
itself is lossless; fabrication insertion loss (waveguide propagation,
fiber-to-chip coupling, MMI/MZI internal losses) is modeled by a scalar
``insertion_loss_db`` (default ``-2.0`` dB) applied AFTER the unitary
multiplication. The optimizer (``chip_optimization.*``) runs in an
earlier pipeline stage and only sees pre-chip field amplitudes, so it
always operates on a clean (lossless) input regardless of this knob;
because a scalar post-unitary loss commutes with the optimizer's ratio-
based objectives (broadband null and throughput penalty), the optimal
unitary is mathematically independent of ``insertion_loss_db``.

Reference implementation is KILO ``chip.py`` — see the WP1 survey notes at
``tests/stages/notes/kernel_mzi_mesh.md`` for line-level citations. The
two non-obvious invariants reproduced here:

* **Parameter layout.** ``params`` has length ``2*n_mzi + n_modes`` where
  ``n_mzi = n_modes*(n_modes-1)//2``. The slices are
  ``theta = params[0:n_mzi]`` (MZI internal angles, set by the
  directional-coupler geometry), ``phi = params[n_mzi:2*n_mzi]`` (MZI
  external arm phases, set by physical path-length differences), and
  ``alpha = params[2*n_mzi:2*n_mzi+n_modes]`` (diagonal output phases,
  also from path-length differences).

* **Wavelength rescaling.** ``phi`` and ``alpha`` are proportional to
  ``1/lambda`` because they are accumulated path-length phases; ``theta``
  is wavelength-independent to leading order (achromatic directional
  coupler). DAC quantization is applied at the design wavelength BEFORE
  rescaling, matching the ordering rule from KILO ``chip.py`` lines
  223-231 (see notes file §4). Calling this stage at
  ``wl_target == wl_design`` reproduces the un-rescaled unitary exactly.

The Clements column ordering used here is the one consumed by
``nullsim/physics/clements.py`` (``decompose_clements`` /
``reconstruct_clements``), which iterates the column index outer-loop and
the lower-row inner-loop downward: pairs ``(i, i+1)`` with
``i = n_modes-2, n_modes-3, ..., column_index`` per column. This is the
nearest-neighbour rectangular mesh from Clements et al. (2016), and it is
the convention the physics primitive's reconstruction inverse expects.
The KILO scan order is structurally equivalent (same MZI count, same
total degrees of freedom) but lays out indices column-major with
alternating rows; we re-use the nullsim primitive rather than
re-deriving KILO's exact index sequence inside the stage.

Field-tensor contract: nullsim fields are shaped ``(n_modes, n_wl, n_pol)``
(see ``nullsim/pipeline/state.py``). The unitary acts on the ``n_modes``
axis; the per-wavelength contraction is
``np.einsum("ijw,jwp->iwp", U_stack, field)`` after assembling
``U_stack`` of shape ``(n_modes, n_modes, n_wl)``.

For NAYRA's dual-PBS-arm front end, ``deployment_mode="per_pol_arm"``
keeps the same public field tensor but treats each polarization arm as a
separate physical chip in the active band. The shared optimized band
design is deployed once per arm with an independent diagonal input phase
trim measured at the design wavelength. This models NAYRA's Y/J/H × A/B
six-chip architecture while leaving KILO/HWO's historical shared-unitary
path as the default.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field as PydanticField, model_validator

from nullsim.physics.clements import ClementsDecomposition, reconstruct_clements
from nullsim.pipeline import (
    BaseStage,
    SimulationState,
    Throughput,
    register_stage,
)
from nullsim.stages._utils import unique_component_name
from nullsim.stages.chip_optimization import (
    _chip_chromaticity_metadata,
    _phase_scale_factors_numpy,
    _theta_offsets_numpy,
    _validate_chip_chromaticity_params,
)

_TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class _ChipApplication:
    amplitudes: NDArray[np.complex128]
    metadata: dict[str, object]
    source_model: str


@register_stage
class KernelMZIMesh(BaseStage):
    """Apply a Clements MZI-mesh unitary per wavelength bin.

    Params come from one of two places: an explicit ``params`` tuple (back-
    compat: the historical TOML idiom), or a lookup of an upstream
    ``chip_optimization.*`` results entry via ``optimization_source``. The
    ``"auto"`` sentinel scans ``state.results.components`` for the first key
    starting with ``chip_optimization`` and uses its ``optimal_chip_params``
    payload — this is the path the example TOMLs follow once a
    ``chip_optimization.*`` stage runs upstream.
    """

    type_name = "chip.kernel_mzi_mesh"
    family = "chip"
    consumes = frozenset({"field", "throughput"})
    produces = frozenset({"field", "throughput", "results"})

    class Params(BaseModel):
        """Validated params for a rectangular Clements MZI mesh."""

        model_config = ConfigDict(
            extra="forbid",
            frozen=True,
            allow_inf_nan=False,
        )

        n_modes: int = PydanticField(gt=0)
        n_bright_ports: int = PydanticField(ge=0)
        n_dark_ports: int = PydanticField(ge=0)
        n_spectro_ports: int = PydanticField(default=0, ge=0)
        wl_design_um: float = PydanticField(gt=0.0)
        params: tuple[float, ...] | None = None
        optimization_source: str | None = None
        dac_bits: int | None = None
        insertion_loss_db: float = PydanticField(default=-2.0, le=0.0)
        deployment_mode: Literal["shared", "per_pol_arm"] = "shared"
        chip_chromaticity_model: Literal["legacy", "parametric"] = "legacy"
        theta_chromaticity_slope_rad_per_frac: float = 0.0
        theta_chromaticity_quadratic_rad_per_frac2: float = 0.0
        phase_chromaticity_model: Literal[
            "legacy_inverse_lambda",
            "fixed_opd_neff",
        ] = "legacy_inverse_lambda"
        phase_neff_ref: float = PydanticField(default=2.4, gt=0.0)
        phase_group_index: float = PydanticField(default=2.4, gt=0.0)

        @model_validator(mode="after")
        def _check_port_split(self) -> KernelMZIMesh.Params:
            n_ports = (
                self.n_bright_ports + self.n_dark_ports + self.n_spectro_ports
            )
            if n_ports != self.n_modes:
                raise ValueError(
                    "chip.kernel_mzi_mesh requires "
                    "n_bright_ports + n_dark_ports + n_spectro_ports == n_modes "
                    f"(got {self.n_bright_ports} + {self.n_dark_ports} + "
                    f"{self.n_spectro_ports} != {self.n_modes})."
                )
            _validate_chip_chromaticity_params(self)
            return self

        @model_validator(mode="after")
        def _check_params_or_source(self) -> KernelMZIMesh.Params:
            if self.params is None and self.optimization_source is None:
                raise ValueError(
                    "chip.kernel_mzi_mesh requires either `params` (explicit tuple) "
                    "or `optimization_source` (a chip_optimization results key, or "
                    "'auto' to take the first chip_optimization.* component)."
                )
            return self

        @model_validator(mode="after")
        def _check_params_length(self) -> KernelMZIMesh.Params:
            if self.params is None:
                # Length is validated in apply() once the upstream optimizer
                # result is resolved.
                return self
            n_mzi = self.n_modes * (self.n_modes - 1) // 2
            expected = 2 * n_mzi + self.n_modes
            if len(self.params) != expected:
                raise ValueError(
                    "chip.kernel_mzi_mesh params length must be "
                    f"2*n_mzi + n_modes = {expected} for n_modes={self.n_modes} "
                    f"(got {len(self.params)})."
                )
            return self

    def required_consumes(self) -> frozenset[str]:
        """Include optimizer results only when this instance reads them."""

        if self.params_model.params is None:
            return self.consumes | frozenset({"results"})
        return self.consumes

    def apply(self, state: SimulationState) -> SimulationState:
        params = self.params_model

        field_amplitudes = state.field.amplitudes
        if field_amplitudes is None:
            raise ValueError(
                "chip.kernel_mzi_mesh requires field amplitudes from an earlier stage."
            )
        field_array = np.asarray(field_amplitudes)
        if field_array.ndim != 3:
            raise ValueError(
                "chip.kernel_mzi_mesh requires field amplitudes with shape "
                "[n_modes, n_wavelength_bins, n_pol]."
            )
        n_modes_field = field_array.shape[0]
        if n_modes_field != params.n_modes:
            raise ValueError(
                "chip.kernel_mzi_mesh field n_modes axis must match params.n_modes "
                f"(field has {n_modes_field}, params has {params.n_modes})."
            )

        wavelength_grid_um = _require_wavelength_grid_um(state, field_array)

        resolved_params, rescale_per_wavelength, precomputed_u_stack = (
            _resolve_chip_params(
                explicit_params=params.params,
                optimization_source=params.optimization_source,
                state=state,
                n_modes=params.n_modes,
            )
        )

        if precomputed_u_stack is not None:
            # Tier 3.1 (option 2): the optimizer publishes the chromatic
            # U_stack it actually evaluated. Use it verbatim so the
            # deployed chip matches the optimizer at every wavelength
            # (KILO column-alternating layout + per-λ phase rescaling baked
            # in). DAC quantization was already applied by the optimizer
            # before building the stack; the chip stage's ``dac_bits``
            # is ignored on this path.
            if precomputed_u_stack.shape != (
                params.n_modes, params.n_modes, len(wavelength_grid_um),
            ):
                raise ValueError(
                    "chip.kernel_mzi_mesh precomputed U_stack shape "
                    f"{precomputed_u_stack.shape} does not match "
                    f"(n_modes={params.n_modes}, n_wl={len(wavelength_grid_um)})."
                )
            u_stack = precomputed_u_stack
            source_provenance = (
                "Precomputed U_stack from upstream chip_optimization backend "
                "(KILO column-alternating layout + per-wavelength phase rescaling; "
                "Tier 3.1 option 2)."
            )
        else:
            raw_params = np.asarray(resolved_params, dtype=np.float64)
            quantized_params = _quantize_phases(raw_params, params.dac_bits)
            theta, phi, alpha = _split_params(quantized_params, params.n_modes)
            order = _clements_order(params.n_modes)

            u_stack = _build_unitary_stack(
                theta=theta,
                phi=phi,
                alpha=alpha,
                order=order,
                n_modes=params.n_modes,
                wavelength_grid_um=wavelength_grid_um,
                wl_design_um=params.wl_design_um,
                rescale_per_wavelength=rescale_per_wavelength,
                chromaticity_params=params,
            )
            source_provenance = (
                "Persisted Clements unitary stack for downstream "
                "sensitivity stage; same per-wavelength rescaling as the "
                "field-multiplication path."
            )

        chip_application = _apply_chip_unitary(
            u_stack=u_stack,
            field_array=field_array.astype(np.complex128, copy=False),
            deployment_mode=params.deployment_mode,
            wavelength_grid_um=wavelength_grid_um,
            wl_design_um=float(params.wl_design_um),
            state=state,
        )
        new_amplitudes = chip_application.amplitudes

        # Scalar fabrication insertion loss applied AFTER the unitary.
        # See module docstring for why this is post-optimization-safe.
        transmission_fraction_chip = 10.0 ** (params.insertion_loss_db / 10.0)
        # Compare the dB knob, not the derived float, so a tiny
        # round-trip error cannot flip the branch (review 2026-06-10 nit).
        if float(params.insertion_loss_db) != 0.0:
            new_amplitudes = new_amplitudes * np.sqrt(transmission_fraction_chip)

        field = replace(
            state.field,
            amplitudes=new_amplitudes,
            wavelength_grid_um=wavelength_grid_um,
        )
        throughput = _updated_throughput(
            state=state,
            wavelength_grid_um=wavelength_grid_um,
            instance_id=self.context.instance_id,
            insertion_loss_db=float(params.insertion_loss_db),
            transmission_fraction_chip=float(transmission_fraction_chip),
        )

        components = dict(state.results.components)
        component_name = unique_component_name(
            self.type_name, components, self.context.instance_id,
        )
        components[component_name] = {
            "u_stack": u_stack.astype(np.complex128, copy=True),
            "wavelength_grid_um": wavelength_grid_um,
            "wl_design_um": float(params.wl_design_um),
            "n_modes": int(params.n_modes),
            "n_bright_ports": int(params.n_bright_ports),
            "n_dark_ports": int(params.n_dark_ports),
            "n_spectro_ports": int(params.n_spectro_ports),
            "precomputed_u_stack_used": precomputed_u_stack is not None,
            "deployment_mode": str(params.deployment_mode),
            **_chip_chromaticity_metadata(params),
            **chip_application.metadata,
            "source_model": f"{source_provenance} {chip_application.source_model}",
        }
        results = replace(state.results, components=components)
        return replace(
            state,
            field=field,
            throughput=throughput,
            results=results,
        )


def _require_wavelength_grid_um(
    state: SimulationState,
    field_amplitudes: NDArray[np.complexfloating],
) -> tuple[float, ...]:
    field_grid_um = state.field.wavelength_grid_um
    throughput_grid_um = state.throughput.wavelength_grid_um
    wavelength_grid_um = field_grid_um or throughput_grid_um
    if not wavelength_grid_um:
        raise ValueError(
            "chip.kernel_mzi_mesh requires a non-empty wavelength grid from "
            "the resolved grid config."
        )
    if field_amplitudes.shape[1] != len(wavelength_grid_um):
        raise ValueError(
            "chip.kernel_mzi_mesh field wavelength axis must match the wavelength grid."
        )
    return tuple(float(value) for value in wavelength_grid_um)


def _apply_chip_unitary(
    *,
    u_stack: NDArray[np.complex128],
    field_array: NDArray[np.complex128],
    deployment_mode: str,
    wavelength_grid_um: tuple[float, ...],
    wl_design_um: float,
    state: SimulationState,
) -> _ChipApplication:
    """Apply the chip stack in shared or per-PBS-arm deployment mode."""

    if deployment_mode == "shared":
        return _ChipApplication(
            amplitudes=np.einsum("ijw,jwp->iwp", u_stack, field_array),
            metadata={
                "active_chip_count": 1,
                "chip_channel_labels": ("shared",),
            },
            source_model=(
                "Shared deployment mode: one chip unitary is applied to every "
                "field polarization/channel, preserving the historical KILO/HWO "
                "single-channel behavior."
            ),
        )

    if deployment_mode != "per_pol_arm":
        raise ValueError(
            "chip.kernel_mzi_mesh deployment_mode must be 'shared' or "
            f"'per_pol_arm', got {deployment_mode!r}."
        )

    n_pol = field_array.shape[2]
    if n_pol != 2:
        raise ValueError(
            "chip.kernel_mzi_mesh deployment_mode='per_pol_arm' requires the "
            f"field polarization axis to contain the two PBS arms; got n_pol={n_pol}."
        )

    design_index = _design_wavelength_index(wavelength_grid_um, wl_design_um)
    phase_correction = _input_phase_correction(
        field_array[:, design_index, :],
    )
    corrected_input = field_array * phase_correction[:, np.newaxis, :]
    amplitudes = np.einsum("ijw,jwp->iwp", u_stack, corrected_input)
    u_stack_per_arm = _per_arm_unitary_stack(u_stack, phase_correction)

    return _ChipApplication(
        amplitudes=amplitudes,
        metadata={
            "active_chip_count": int(n_pol),
            "chip_channel_labels": _active_chip_channel_labels(state, n_pol),
            "phase_reference_wavelength_index": int(design_index),
            "phase_reference_wavelength_um": float(wavelength_grid_um[design_index]),
            "input_phase_correction": phase_correction.astype(
                np.complex128, copy=True,
            ),
            "u_stack_per_arm": u_stack_per_arm.astype(np.complex128, copy=True),
        },
        source_model=(
            "Per-PBS-arm deployment mode: the active band is represented by two "
            "independent physical chips, one for each PBS arm. Each arm deploys "
            "the shared optimized band design with an independent diagonal input "
            "phase trim at the design wavelength, calibrating static pre-chip "
            "transport/PM-fiber phases inside that arm's chip. The 3-D 'u_stack' "
            "payload remains the shared nominal band design for downstream "
            "sensitivity calculations; 'u_stack_per_arm' records the deployed "
            "physical arm stacks."
        ),
    )


def _design_wavelength_index(
    wavelength_grid_um: tuple[float, ...],
    wl_design_um: float,
) -> int:
    grid = np.asarray(wavelength_grid_um, dtype=np.float64)
    return int(np.argmin(np.abs(grid - float(wl_design_um))))


def _input_phase_correction(
    design_field: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Return per-mode/per-arm phases that make the design-bin input real."""

    return np.exp(-1j * np.angle(design_field)).astype(np.complex128)


def _per_arm_unitary_stack(
    u_stack: NDArray[np.complex128],
    phase_correction: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Return deployed arm stacks with phase trims folded into U columns."""

    # phase_correction has shape (n_modes, n_pol). Multiplying U's input
    # columns by correction[:, arm] is equivalent to U @ diag(correction).
    return (
        u_stack[np.newaxis, :, :, :]
        * phase_correction.T[:, np.newaxis, :, np.newaxis]
    )


def _active_chip_channel_labels(
    state: SimulationState,
    n_pol: int,
) -> tuple[str, ...]:
    """Return primary-band NAYRA channel labels when the manifest is present."""

    for payload in state.results.components.values():
        if not isinstance(payload, dict):
            continue
        channels = payload.get("channels")
        if not isinstance(channels, (tuple, list)):
            continue
        primary = [
            str(channel.get("channel_id"))
            for channel in channels
            if isinstance(channel, dict) and bool(channel.get("is_primary_band"))
        ]
        if len(primary) == n_pol:
            return tuple(primary)
    if n_pol == 2:
        return ("A", "B")
    return tuple(f"pol_{index}" for index in range(n_pol))


def _quantize_phases(
    raw_params: NDArray[np.float64],
    dac_bits: int | None,
) -> NDArray[np.float64]:
    """Linear DAC quantization over ``[0, 2*pi)`` at the design wavelength.

    Mirrors KILO ``quantize_phases`` (chip.py:96-111): ``step = 2*pi / 2**bits``
    and round-to-nearest, then wrap to ``[0, 2*pi)``. The quantization is
    uniform across theta, phi, and alpha — there is no per-block bit-depth
    control (notes file §5).

    The wrap is unconditional. When ``dac_bits is None`` we still wrap
    to ``[0, 2*pi)`` so the deployed chip matches what the scipy
    optimizer's ``_build_unitary_stack_numpy`` minimized against (which
    wraps unconditionally before applying the per-wavelength ``wl_design
    / wl_target`` rescaling). Wrap and per-wavelength scaling do not
    commute: ``scale * (2pi + x)`` and ``scale * x`` differ for any
    ``scale != 1``, so a missing wrap on deploy reproduces a different
    chromatic chip than the one the loss reported. This was the
    chip_optimization/kernel handoff bug: optimizer reported loss
    ~1e-16, deploy gave null_depth_calibrated ~1.4 in the HWO sanity path.
    """

    if dac_bits is None:
        return np.mod(np.asarray(raw_params, dtype=np.float64), _TWO_PI)
    if dac_bits <= 0:
        raise ValueError(
            f"chip.kernel_mzi_mesh dac_bits must be positive when set (got {dac_bits})."
        )
    step = _TWO_PI / (1 << dac_bits)
    quantized = step * np.round(raw_params / step)
    return np.mod(quantized, _TWO_PI)


def _split_params(
    quantized_params: NDArray[np.float64],
    n_modes: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(theta, phi, alpha)`` slices per notes file §1."""

    n_mzi = n_modes * (n_modes - 1) // 2
    theta = quantized_params[0:n_mzi]
    phi = quantized_params[n_mzi : 2 * n_mzi]
    alpha = quantized_params[2 * n_mzi : 2 * n_mzi + n_modes]
    return theta, phi, alpha


_CHIP_OPT_PREFIX = "chip_optimization"


def _resolve_chip_params(
    *,
    explicit_params: tuple[float, ...] | None,
    optimization_source: str | None,
    state: SimulationState,
    n_modes: int,
) -> tuple[tuple[float, ...], bool, NDArray[np.complex128] | None]:
    """Return ``(params, rescale_per_wavelength, precomputed_u_stack)``.

    Precedence rules:
      * explicit ``params`` always wins (back-compat for the historical TOML idiom);
      * otherwise ``optimization_source == 'auto'`` scans components for the
        first key starting with ``chip_optimization`` (case-insensitive);
      * otherwise the string is treated as an exact component key.

    Length is validated against ``2 * n_mzi + n_modes`` regardless of source.

    The ``rescale_per_wavelength`` flag and the optional
    ``precomputed_u_stack`` both come from the upstream optimizer payload.
    When ``precomputed_u_stack`` is present (Tier 3.1 option 2) the chip
    stage uses it verbatim and ignores ``rescale_per_wavelength`` — the
    optimizer already evaluated the chromatic chip at every detector
    wavelength using the layout + rescaling it actually optimized
    against. Explicit-params consumers default to
    ``rescale_per_wavelength=True`` (back-compat) and never have a
    precomputed stack.
    """

    if explicit_params is not None:
        params = tuple(float(value) for value in explicit_params)
        rescale_per_wavelength = True
        precomputed_u_stack: NDArray[np.complex128] | None = None
    else:
        if optimization_source is None:
            raise ValueError(
                "chip.kernel_mzi_mesh needs either `params` or `optimization_source`."
            )

        components = state.results.components
        if optimization_source.lower() == "auto":
            matching = [
                key for key in components
                if key.lower().startswith(_CHIP_OPT_PREFIX)
            ]
            if not matching:
                raise KeyError(
                    "chip.kernel_mzi_mesh optimization_source='auto' found no "
                    "chip_optimization.* component in state.results.components. "
                    f"Available keys: {sorted(components)}."
                )
            if len(matching) > 1:
                raise ValueError(
                    "chip.kernel_mzi_mesh optimization_source='auto' is ambiguous; "
                    f"matching components: {matching}. Set optimization_source "
                    "to the exact component key."
                )
            source_key = matching[0]
        else:
            if optimization_source not in components:
                raise KeyError(
                    "chip.kernel_mzi_mesh optimization_source="
                    f"{optimization_source!r} not found in state.results.components. "
                    f"Available keys: {sorted(components)}."
                )
            source_key = optimization_source

        payload = components[source_key]
        if "optimal_chip_params" not in payload:
            raise KeyError(
                f"chip.kernel_mzi_mesh upstream component {source_key!r} has no "
                "'optimal_chip_params' field."
            )
        params = tuple(float(value) for value in payload["optimal_chip_params"])
        rescale_per_wavelength = bool(
            payload.get("kernel_rescales_per_wavelength", True)
        )
        precomputed_raw = payload.get("optimal_u_stack")
        if precomputed_raw is not None:
            precomputed_u_stack = np.asarray(precomputed_raw, dtype=np.complex128)
        else:
            precomputed_u_stack = None

    n_mzi = n_modes * (n_modes - 1) // 2
    expected = 2 * n_mzi + n_modes
    if len(params) != expected:
        raise ValueError(
            "chip.kernel_mzi_mesh resolved params length must be "
            f"2*n_mzi + n_modes = {expected} for n_modes={n_modes} "
            f"(got {len(params)})."
        )
    return params, rescale_per_wavelength, precomputed_u_stack


def _clements_order(n_modes: int) -> list[tuple[int, int]]:
    """Return the ``(upper_row, lower_row)`` MZI pair sequence.

    The order matches ``nullsim.physics.clements.decompose_clements``:
    outer loop over ``column_index in range(n_modes - 1)``, inner loop over
    ``lower_row in range(n_modes - 1, column_index, -1)``. Every pair is
    ``(lower_row - 1, lower_row)`` — i.e. nearest-neighbour rectangular
    mesh from Clements et al. (2016).
    """

    return [
        (lower_row - 1, lower_row)
        for column_index in range(n_modes - 1)
        for lower_row in range(n_modes - 1, column_index, -1)
    ]


def _build_unitary_stack(
    *,
    theta: NDArray[np.float64],
    phi: NDArray[np.float64],
    alpha: NDArray[np.float64],
    order: list[tuple[int, int]],
    n_modes: int,
    wavelength_grid_um: tuple[float, ...],
    wl_design_um: float,
    rescale_per_wavelength: bool = True,
    chromaticity_params: object | None = None,
) -> NDArray[np.complex128]:
    """Build the per-wavelength stack of ``(n_modes, n_modes)`` unitaries.

    Returns an array of shape ``(n_modes, n_modes, n_wavelength_bins)``.
    Per notes file §4: ``phi`` and ``alpha`` rescale by
    ``wl_design / wl_target``; ``theta`` is left alone.

    When ``rescale_per_wavelength=False`` (consumer of torch
    backend output, whose canonical params from
    ``_canonicalize_clements_params`` only match the optimized unitary
    at ``wl_design``), the same ``(theta, phi, alpha)`` is reconstructed
    at every wavelength bin — chip is monochromatic at ``wl_design``.
    Ultra-review High #5.
    """

    n_wl = len(wavelength_grid_um)
    phase_scales = _phase_scale_factors_numpy(
        wavelength_grid_um=wavelength_grid_um,
        wl_design_um=wl_design_um,
        chromaticity_params=chromaticity_params,
        rescale_per_wavelength=rescale_per_wavelength,
    )
    theta_offsets = _theta_offsets_numpy(
        wavelength_grid_um=wavelength_grid_um,
        wl_design_um=wl_design_um,
        chromaticity_params=chromaticity_params,
        apply_chromaticity=rescale_per_wavelength,
    )
    stack = np.empty((n_modes, n_modes, n_wl), dtype=np.complex128)
    for k, scale in enumerate(phase_scales):
        decomp = ClementsDecomposition(
            theta=theta + theta_offsets[k],
            phi=phi * scale,
            alpha=alpha * scale,
            order=order,
            n_modes=n_modes,
        )
        stack[:, :, k] = reconstruct_clements(decomp)
    return stack


def _updated_throughput(
    *,
    state: SimulationState,
    wavelength_grid_um: tuple[float, ...],
    instance_id: str,
    insertion_loss_db: float,
    transmission_fraction_chip: float,
) -> Throughput:
    """Record chip insertion loss as a diagnostic component without double-counting.

    Convention (``nullsim/stages/detector/mkid.py:_port_rate_hz_per_wavelength``):
    chip losses are modeled in field amplitudes — the detector forms photon
    rates from ``|U·field|² × transmission_fraction``, so multiplying
    ``transmission_fraction`` by the chip factor on top of the ``sqrt(t)``
    already applied to amplitudes would yield a ``t²`` double-count. Here
    ``transmission_fraction`` passes through unchanged; the chip-stage
    component still records the ``t`` factor so the loss-budget plot and
    debug report attribute the loss correctly.
    """

    n_wavelength_bins = len(wavelength_grid_um)
    current_transmission = _throughput_or_unity(state, n_wavelength_bins)
    components = dict(state.throughput.components)
    component_payload = {
        "wavelength_grid_um": wavelength_grid_um,
        "transmission_fraction": tuple(
            transmission_fraction_chip for _ in range(n_wavelength_bins)
        ),
        "field_amplitude_applied": True,
        "insertion_loss_db": float(insertion_loss_db),
        "source_model": (
            "Clements MZI mesh (unitary; lossless) × wavelength-flat "
            f"fabrication insertion loss ({insertion_loss_db:+.2f} dB → "
            f"t = {transmission_fraction_chip:.6f}). Applied to field "
            "amplitudes (sqrt(t)) AFTER U·field; recorded here for the "
            "loss budget but NOT folded into transmission_fraction to "
            "avoid double-counting at the detector."
        ),
    }
    component_name = unique_component_name(
        "chip.kernel_mzi_mesh_loss",
        components,
        instance_id,
    )
    components[component_name] = component_payload
    return Throughput(
        wavelength_grid_um=wavelength_grid_um,
        transmission_fraction=tuple(float(v) for v in current_transmission),
        components=components,
    )


def _throughput_or_unity(
    state: SimulationState,
    n_wavelength_bins: int,
) -> NDArray[np.float64]:
    if state.throughput.transmission_fraction:
        if len(state.throughput.transmission_fraction) != n_wavelength_bins:
            raise ValueError(
                "chip.kernel_mzi_mesh throughput length must match the wavelength grid."
            )
        return np.asarray(state.throughput.transmission_fraction, dtype=np.float64)
    return np.ones(n_wavelength_bins, dtype=np.float64)


__all__ = ["KernelMZIMesh"]
