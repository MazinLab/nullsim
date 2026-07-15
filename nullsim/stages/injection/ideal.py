"""Architecture placeholder injection stage with perfect coupling.

This module exists to exercise the vertical-slice pipeline from SPEC §15.1.
It is not a physics implementation of Ruilier-Cassaing coupling, AO residuals,
or fiber-mode overlap, but it does honor an upstream multi-aperture
``geometry.n_apertures`` so KILO-style array configs can seed a field of
shape ``(n_apertures, n_wavelength_bins, n_pol)`` ahead of the SMF / chip
stages. Each aperture sees the same unit-amplitude incoming starlight
(point source at infinity); per-aperture differentiation happens later via
atmospheric piston, AO residuals, fiber coupling, and the kernel chip.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field as PydanticField

from nullsim.physics.ruilier_cassaing import (
    FRESNEL_REFLECTION_LOSS,
    OPTIMAL_PUPIL_RADIUS_TO_MODE_FIELD_RADIUS_RATIO,
    coupling_efficiency,
    geometric_coupling_limit,
)
from nullsim.pipeline import (
    BaseStage,
    Field,
    SimulationState,
    Throughput,
    register_stage,
)


@register_stage
class IdealInjection(BaseStage):
    """Ideal Ruilier–Cassaing injection (no fiber mode mismatch beyond η_opt).

    The "ideal" label distinguishes this stage from
    ``injection.single_mode_fiber`` — there is no Gaussian mode-radius
    overlap loss beyond the unobstructed Ruilier–Cassaing optimum η_opt
    = 0.81, no off-axis SMF rolloff, no fiber-length dispersion. The
    stage *does* honor the upstream wavefront Strehl (and, optionally,
    central-obscuration η_opt(ε) and Fresnel reflection loss) when
    seeding the stellar field, matching KILO's ``stellar_input_field``
    convention (``kilo/injection.py:194-223``: ``E = √(η × A)`` with
    ``η = S × η_opt × (1 − R_Fresnel)``).

    Without this Strehl coupling the field carries unit amplitude per
    mode, so the downstream MKID stage (which derives count rates from
    ``|field|²``) sees the full unattenuated stellar flux. That was the
    headline Phase-0 bug: ~3-8× MKID rate overcount in the photon-
    limited contrast regime.
    """

    type_name = "injection.ideal"
    family = "injection"
    # NOTE: wavefront is read opportunistically (see
    # ``_upstream_strehl_per_wavelength``) but NOT in ``consumes`` —
    # the HWO-space pipeline runs this stage with no upstream AO/
    # atmosphere, and the runner enforces a strict produces/consumes
    # graph that would reject a missing ``wavefront`` producer there.
    # ``cache_key_extras`` below folds ``wavefront`` into the cache key
    # without making it
    # a hard dependency, so AO-param-only sweeps invalidate the
    # injected field correctly.
    consumes = frozenset({"scene", "geometry"})
    produces = frozenset({"field", "throughput", "wavefront"})
    # Phase 7: the stage reads upstream Strehl from state.wavefront
    # opportunistically (see the NOTE above). We fold ``wavefront``
    # into the cache key via cache_key_extras so AO-param-only sweeps
    # that change Strehl invalidate the cached injected field. The
    # extras list is NOT enforced by validate_dependencies, so the
    # HWO-space pipeline (no AO/atmosphere upstream) still runs.
    # Reads the upstream Strehl from ``wavefront`` and carries the existing
    # ``throughput`` content forward, neither declared in ``consumes``.
    cache_key_extras = frozenset({"wavefront", "throughput"})

    class Params(BaseModel):
        """Validated params for the ideal injection stage.

        ``n_modes`` is an optional override for single-aperture configs that
        want to seed an N-mode field (e.g. for a kernel-nuller chip whose
        mode axis is larger than ``geometry.n_apertures``). When set, it
        must be a multiple of ``geometry.n_apertures`` (so the broadcast
        replicates each aperture's amplitude across the extra modes).

        ``apply_strehl_coupling`` (default True) scales the seeded field
        amplitudes by ``√(S × η_opt(ε) × (1 − R_Fresnel))`` using
        ``nullsim.physics.ruilier_cassaing.coupling_efficiency``, matching
        the per-port factor that ``sensitivity.detection_curve`` already
        applies to the planet throughput. Defaults match
        ``detection_curve.Params`` so the stellar and planet paths see the
        same coupling factor and the contrast ratio cancels cleanly.
        """

        model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True)

        n_modes: int | None = None
        apply_strehl_coupling: bool = True
        apply_fresnel_injection_loss: bool = False
        central_obscuration_ratio: float = PydanticField(default=0.0, ge=0.0, lt=1.0)
        # Honest-convention knob (2026-07-14 adjudication): the fiber MFD
        # is fixed, so the pupil-plane mode radius scales with wavelength
        # and flat coupling peaks at the design wavelength instead of
        # sitting at the geometric limit everywhere. rho(lambda) =
        # rho_design * (lambda_design / lambda); the design wavelength
        # defaults to the band mean.
        chromatic_coupling_rolloff: bool = False
        coupling_design_wavelength_um: float | None = PydanticField(
            default=None, gt=0.0
        )

    def apply(self, state: SimulationState) -> SimulationState:
        params = self.params_model
        wavelength_grid_um = _require_wavelength_grid_um(state)
        n_wavelength_bins = len(wavelength_grid_um)
        n_apertures = _require_n_apertures(state)
        n_modes = _resolve_n_modes(n_apertures, params.n_modes)

        upstream_strehl = _upstream_strehl_per_wavelength(state, n_wavelength_bins)
        coupling_per_wavelength = _coupling_per_wavelength(
            strehl=upstream_strehl,
            apply_strehl_coupling=params.apply_strehl_coupling,
            apply_fresnel_loss=params.apply_fresnel_injection_loss,
            central_obscuration_ratio=params.central_obscuration_ratio,
        )
        if params.chromatic_coupling_rolloff and params.apply_strehl_coupling:
            coupling_per_wavelength = (
                coupling_per_wavelength
                * _chromatic_rolloff_per_wavelength(
                    wavelength_grid_um=np.asarray(
                        wavelength_grid_um, dtype=np.float64
                    ),
                    design_wavelength_um=params.coupling_design_wavelength_um,
                    central_obscuration_ratio=params.central_obscuration_ratio,
                )
            )

        # KILO ``stellar_input_field`` convention (kilo/injection.py:222):
        # E_k = √(η_k × A_k). The collecting area per sub-aperture is
        # applied downstream (detector/mkid.py: ``area_per_mode_m2 ×
        # |field|²``), so this stage only needs to carry the η factor.
        amplitude_scale = np.sqrt(coupling_per_wavelength).astype(np.float64, copy=False)
        amplitudes = np.broadcast_to(
            amplitude_scale[np.newaxis, :, np.newaxis].astype(np.complex128),
            (n_modes, n_wavelength_bins, 1),
        ).copy()

        field = Field(
            amplitudes=amplitudes,
            wavelength_grid_um=wavelength_grid_um,
            mode_labels=_mode_labels(n_apertures, n_modes),
        )
        # Throughput contributions from upstream stages (telescope mirror
        # transmission, atmosphere, …) flow through unchanged. The
        # coupling factor is carried by the field amplitudes (KILO
        # ``stellar_input_field`` convention; see also
        # ``injection.single_mode_fiber._updated_throughput`` for the
        # same "no double-counting at the detector" note). The
        # ``injection_ideal`` component is flagged ``field_amplitude_applied``
        # so stages that reconstruct an absolute stellar rate from the
        # throughput budget (``characterization.spectro``) fold the coupling
        # back in; the detector reads the top-level ``transmission_fraction``
        # (which excludes it) and gets the coupling from ``|field|^2`` instead.
        unit_transmission = tuple(1.0 for _ in wavelength_grid_um)
        existing_transmission = (
            tuple(state.throughput.transmission_fraction)
            if state.throughput.transmission_fraction
            and len(state.throughput.transmission_fraction) == n_wavelength_bins
            else unit_transmission
        )
        # Of the coupling factor only the Fresnel reflection is a true
        # transmission loss; the Strehl × η_opt overlap term is a
        # point-source wavefront-coupling factor that does NOT apply to a
        # diffuse background (brightness theorem: single-mode coupling of
        # an extended incoherent source is étendue-fixed). Background
        # stages read ``diffuse_transmission_fraction`` instead of
        # ``transmission_fraction`` (review 2026-06-10 Issue 5).
        diffuse_transmission = (
            1.0 - FRESNEL_REFLECTION_LOSS
            if params.apply_fresnel_injection_loss
            else 1.0
        )
        throughput = Throughput(
            wavelength_grid_um=wavelength_grid_um,
            transmission_fraction=existing_transmission,
            components=dict(state.throughput.components),
        ).add_component(
            "injection_ideal",
            {
                "wavelength_grid_um": wavelength_grid_um,
                "transmission_fraction": tuple(
                    float(v) for v in coupling_per_wavelength
                ),
                "strehl_ratio": tuple(float(v) for v in upstream_strehl),
                "apply_strehl_coupling": bool(params.apply_strehl_coupling),
                "apply_fresnel_injection_loss": bool(
                    params.apply_fresnel_injection_loss
                ),
                "central_obscuration_ratio": float(params.central_obscuration_ratio),
                "field_amplitude_applied": True,
                "point_source_coupling_only": True,
                "diffuse_transmission_fraction": tuple(
                    diffuse_transmission for _ in wavelength_grid_um
                ),
            },
        )
        # Preserve the upstream Strehl on the top-level wavefront so
        # downstream stages (notably ``sensitivity.detection_curve``)
        # see the AO/atmosphere-residual Strehl, not a clobbered 1.0.
        # Carry forward upstream wavefront components for the OPD/Strehl
        # provenance chain into chip/detector.
        wavefront = replace(
            state.wavefront,
            wavelength_grid_um=wavelength_grid_um,
            strehl_ratio=tuple(float(v) for v in upstream_strehl),
        ).add_component(
            "injection_ideal",
            {
                "strehl_ratio": tuple(float(v) for v in upstream_strehl),
                "coupling_efficiency": tuple(
                    float(v) for v in coupling_per_wavelength
                ),
                "apply_strehl_coupling": bool(params.apply_strehl_coupling),
                "apply_fresnel_injection_loss": bool(
                    params.apply_fresnel_injection_loss
                ),
                "central_obscuration_ratio": float(params.central_obscuration_ratio),
                "n_apertures": n_apertures,
                "n_modes": n_modes,
            },
        )
        return replace(
            state,
            field=field,
            throughput=throughput,
            wavefront=wavefront,
        )


def _require_wavelength_grid_um(state: SimulationState) -> tuple[float, ...]:
    wavelength_grid_um = state.throughput.wavelength_grid_um or state.field.wavelength_grid_um
    if not wavelength_grid_um:
        raise ValueError(
            "injection.ideal requires a non-empty wavelength grid from the resolved grid config."
        )
    return wavelength_grid_um


def _require_n_apertures(state: SimulationState) -> int:
    """Default to 1 aperture if the upstream telescope stage didn't set it.

    Keeps single-aperture configs (NAYRA, HWO) backward compatible while
    letting multi-telescope configs (KILO) seed an N-mode field directly.
    """

    n_apertures = state.geometry.n_apertures
    if n_apertures is None:
        return 1
    if n_apertures < 1:
        raise ValueError(
            f"injection.ideal requires geometry.n_apertures >= 1 (got {n_apertures})."
        )
    return int(n_apertures)


def _resolve_n_modes(n_apertures: int, n_modes_override: int | None) -> int:
    """Pick the field's mode-axis size, respecting an optional override.

    The override is required to be a positive multiple of ``n_apertures`` so
    each aperture's amplitude can be replicated cleanly across the extra
    modes (no fractional split). The default (``None``) returns
    ``n_apertures``.
    """

    if n_modes_override is None:
        return n_apertures
    if n_modes_override < 1:
        raise ValueError(
            f"injection.ideal requires n_modes >= 1 (got {n_modes_override})."
        )
    if n_modes_override % n_apertures != 0:
        raise ValueError(
            "injection.ideal n_modes override must be a positive multiple of "
            f"geometry.n_apertures (got n_modes={n_modes_override}, "
            f"n_apertures={n_apertures})."
        )
    return int(n_modes_override)


def _mode_labels(n_apertures: int, n_modes: int) -> tuple[str, ...]:
    """Build human-readable mode labels for the seeded field."""

    if n_modes == n_apertures:
        return tuple(f"aperture_{i}" for i in range(n_apertures))
    modes_per_aperture = n_modes // n_apertures
    return tuple(
        f"aperture_{i}_mode_{j}"
        for i in range(n_apertures)
        for j in range(modes_per_aperture)
    )



def _upstream_strehl_per_wavelength(
    state: SimulationState,
    n_wavelength_bins: int,
) -> NDArray[np.float64]:
    """Return the upstream Strehl on the wavelength grid (defaults to 1.0).

    Falls back to a unit Strehl spectrum **only** when no upstream
    wavefront-bearing stage has run at all (no components AND no Strehl
    array) — the ``injection.ideal`` stage is also used in the minimal
    vertical-slice tests and HWO-space pipelines where no atmosphere/AO
    is present.

    Raises when ``state.wavefront`` has any component populated (an
    upstream wavefront-quality stage *did* run) but ``strehl_ratio`` is
    missing — that is an inconsistent state that earlier silently
    became ``Strehl=1``, producing an unphysical field with no warning.
    The cure for an AO-bearing pipeline is for the upstream stage to
    publish ``strehl_ratio``; the cure for a no-AO pipeline is to leave
    ``state.wavefront`` fully empty.
    """

    strehl = state.wavefront.strehl_ratio
    if not strehl:
        if state.wavefront.components:
            raise ValueError(
                "injection.ideal: state.wavefront has components "
                f"{tuple(state.wavefront.components)!r} but strehl_ratio is "
                "missing. An upstream wavefront-quality stage ran without "
                "publishing Strehl — defaulting to 1.0 would silently "
                "produce an unphysical injected field. Either publish "
                "strehl_ratio in the upstream stage, or run a fully "
                "no-AO pipeline (no wavefront components at all)."
            )
        return np.ones(n_wavelength_bins, dtype=np.float64)
    strehl_array = np.asarray(strehl, dtype=np.float64)
    if strehl_array.shape != (n_wavelength_bins,):
        raise ValueError(
            "injection.ideal: state.wavefront.strehl_ratio has shape "
            f"{strehl_array.shape}; expected ({n_wavelength_bins},) to match "
            "field.wavelength_grid_um."
        )
    if np.any(~np.isfinite(strehl_array)):
        raise ValueError(
            "injection.ideal: state.wavefront.strehl_ratio contains non-finite "
            f"values: {strehl_array.tolist()}."
        )
    return np.clip(strehl_array, 0.0, 1.0)


def _coupling_per_wavelength(
    *,
    strehl: NDArray[np.float64],
    apply_strehl_coupling: bool,
    apply_fresnel_loss: bool,
    central_obscuration_ratio: float,
) -> NDArray[np.float64]:
    """Combined per-port coupling factor η(λ) = S × η_opt(ε) × (1 − R_Fresnel).

    Matches the per-port factor that ``sensitivity.detection_curve``
    applies to the planet throughput, so the stellar (via field) and
    planet paths share one coupling convention.

    When ``apply_strehl_coupling`` is False the Strehl×η_opt term is
    dropped (η_opt is bundled with Strehl in
    ``ruilier_cassaing.coupling_efficiency``); the Fresnel-loss
    multiplier is still honored independently because Fresnel is an
    optical reflection loss separate from the wavefront-quality term.
    """

    if apply_strehl_coupling:
        return np.asarray(
            coupling_efficiency(
                strehl,
                central_obscuration_ratio,
                apply_fresnel_loss=apply_fresnel_loss,
            ),
            dtype=np.float64,
        )
    base = np.ones_like(strehl, dtype=np.float64)
    if apply_fresnel_loss:
        base = base * (1.0 - FRESNEL_REFLECTION_LOSS)
    return base


def _chromatic_rolloff_per_wavelength(
    *,
    wavelength_grid_um: NDArray[np.float64],
    design_wavelength_um: float | None,
    central_obscuration_ratio: float,
) -> NDArray[np.float64]:
    """Coupling rolloff factor eta(lambda) / eta(lambda_design).

    A fixed fiber mode-field diameter maps to a pupil-plane Gaussian mode
    whose radius scales with wavelength (w_pupil = lambda f / (pi w_fiber)),
    so the Ruilier overlap ratio is rho(lambda) = rho_design *
    (lambda_design / lambda). The factor is exactly 1 at the design
    wavelength and second-order in the fractional detuning near the
    Ruilier optimum.
    """

    design_um = (
        float(design_wavelength_um)
        if design_wavelength_um is not None
        else float(np.mean(wavelength_grid_um))
    )
    rho_design = OPTIMAL_PUPIL_RADIUS_TO_MODE_FIELD_RADIUS_RATIO
    limit_at_design = float(
        geometric_coupling_limit(
            central_obscuration_ratio,
            pupil_radius_to_mode_field_radius_ratio=rho_design,
        )
    )
    factors = np.array(
        [
            float(
                geometric_coupling_limit(
                    central_obscuration_ratio,
                    pupil_radius_to_mode_field_radius_ratio=(
                        rho_design * design_um / float(wavelength_um)
                    ),
                )
            )
            / limit_at_design
            for wavelength_um in wavelength_grid_um
        ],
        dtype=np.float64,
    )
    return factors


__all__ = ["IdealInjection"]
