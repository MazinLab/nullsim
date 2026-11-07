"""Pyramid WFS AO residual stage backed by KILO/NAYRA provenance.

This SPEC §15.3 stage wraps the KILO adaptive-optics residual model into the
nullsim pipeline state. The implemented terms are exactly the KILO phase
residual terms in ``kilo/atmosphere.py``: ``ao_residual_variance`` and
``ao_piston_phase_components`` combine fitting, temporal-lag, and pyramid-WFS
photon noise only. The pyramid-WFS photon budget comes from KILO ``pwfs.py``
(ported from NAYRA ``wfs.py``): an R-band Vega-zero-point photon rate,
Beer-Lambert atmospheric throughput, central obstruction, pyramid modulation
sensitivity, detector excess noise, and readout floor.

The pure Kolmogorov wavelength scaling and Marechal Strehl conversion use
``nullsim.physics.kolmogorov``. KILO's AO-specific coefficients and piston
filter integrals stay private to this stage for this work package; they are
not promoted to ``nullsim.physics`` until the broader stage layer settles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

# Code Review 2026-05-16 Issue 60: scipy imports were inside the
# lru_cached _piston_filter_fraction call, so the first uncached call
# paid the import cost. Hoist them to module load.
from scipy.integrate import quad as _scipy_integrate_quad
from scipy.special import j1 as _scipy_special_j1

from nullsim.physics import kolmogorov
from nullsim.physics.photometry import estimate_r_mag_from_j
from nullsim.physics.pupil_geometry import (
    PupilGeometry,
    eelt_pupil,
    keck_pupil,
    optimize_subaperture_positions,
)
from nullsim.pipeline import (
    BaseStage,
    OPDBudget,
    OPDComponent,
    SimulationState,
    Wavefront,
    register_stage,
)

_MICRON_TO_M = 1.0e-6
_M_TO_NM = 1.0e9
_FIVE_THIRDS = 5.0 / 3.0

# KILO atmosphere.py constants: continuous facesheet fitting coefficient
# (Hardy 1998 Table 9.2) and Greenwood frequency coefficient.
_FITTING_ERROR_COEFFICIENT = 0.28
_GREENWOOD_FREQUENCY_COEFFICIENT = 0.426
_SERVO_BANDWIDTH_DIVISOR = 10.0

# KILO pwfs.py R-band Vega zero point in photons s^-1 m^-2 um^-1.
_VEGA_R_ZEROPOINT_PH_S_M2_UM = 3.01e10


@dataclass(frozen=True)
class _ResidualArrays:
    fitting_rad2: np.ndarray
    temporal_rad2: np.ndarray
    wfs_noise_rad2: np.ndarray
    total_rad2: np.ndarray
    strehl_ratio: np.ndarray
    residual_opd_m: np.ndarray


@dataclass(frozen=True)
class _PistonArrays:
    temporal_rad2: np.ndarray
    fitting_rad2: np.ndarray
    wfs_noise_rad2: np.ndarray
    total_rad2: np.ndarray
    temporal_opd_m: np.ndarray
    fitting_opd_m: np.ndarray
    wfs_noise_opd_m: np.ndarray
    total_opd_m: np.ndarray
    frac_temporal: float
    frac_fitting: float
    frac_wfs_noise: float


@register_stage
class PyramidWFS(BaseStage):
    """Apply KILO-style pyramid-WFS AO residuals to wavefront and OPD budgets."""

    type_name = "ao.pyramid_wfs"
    family = "ao"
    consumes = frozenset({"geometry", "opd_budget", "wavefront", "scene"})
    produces = frozenset({"opd_budget", "wavefront", "geometry"})

    class Params(BaseModel):
        """Validated params matching KILO/NAYRA AO + PyramidWFS config fields."""

        model_config = ConfigDict(
            extra="forbid",
            frozen=True,
            allow_inf_nan=False,
        )

        n_actuators: int = Field(default=3025, gt=0)
        loop_frequency_hz: float = Field(default=2000.0, gt=0.0)
        guide_star_R_mag: float = 8.0
        # When True, ignore guide_star_R_mag and derive R-band magnitude
        # from scene.star.jmag using a spectral-type R-J color. Used by
        # configs where the science target is also the natural guide star.
        # ``r_minus_j_offset`` remains as an explicit legacy override for
        # configs that intentionally pin a fixed color.
        derive_r_mag_from_scene_jmag: bool = False
        r_minus_j_offset: float | None = None
        guide_star_default_spectral_type: str = "F"
        central_obstruction: float = Field(default=0.0, ge=0.0, lt=1.0)
        tau0_500nm_s: float = Field(default=2.83e-3, gt=0.0)
        derive_wind_from_tau0: bool = True
        theta0_500nm_arcsec: float = Field(default=1.19, gt=0.0)
        v_wind_m_s: float = Field(default=10.0, gt=0.0)
        airmass: float | None = Field(default=None, ge=1.0)
        wfs_wavelength_um: float = Field(default=0.7, gt=0.0)
        wfs_bandwidth_um: float = Field(default=0.3, gt=0.0)
        n_subap_across: int = Field(default=50, gt=0)
        modulation_lambda_d: float = Field(default=0.0, ge=0.0)
        dichroic_throughput: float = Field(default=0.95, gt=0.0, le=1.0)
        atmospheric_throughput: float = Field(default=0.93, gt=0.0, le=1.0)
        mirror_throughput: float = Field(default=0.86, gt=0.0, le=1.0)
        quantum_efficiency: float = Field(default=0.85, gt=0.0, le=1.0)
        excess_noise_sq: float = Field(default=2.0, gt=0.0)
        readout_noise_e_rms: float = Field(default=0.5, ge=0.0)
        piston_subaperture_diameter_m: float | None = Field(default=None, gt=0.0)
        # Derive ``piston_subaperture_diameter_m`` from pupil geometry when
        # the explicit value above is None. ``n_subapertures_per_aperture``
        # is the per-telescope sub-aperture count fed into KILO's
        # ``optimize_subaperture_positions``; ``pupil_template`` chooses
        # the boundary + obstruction + vane recipe (``"keck"`` =
        # hexagonal mosaic with 2.4 m obstruction + 6 vanes;
        # ``"eelt"`` = NAYRA ELT circular pupil with 11% obstruction +
        # 6 broad vanes; ``"circular"`` = simple circular aperture with
        # the obstruction diameter below). When both
        # ``piston_subaperture_diameter_m`` and
        # ``n_subapertures_per_aperture`` are None the stage falls back to
        # the historical fixed-fraction (1.0, 0.03, 1.0) piston filter.
        n_subapertures_per_aperture: int | None = Field(default=None, gt=0)
        pupil_template: Literal["circular", "keck", "eelt", "full_pupil"] | None = None
        pupil_central_obstruction_diameter_m: float = Field(default=0.0, ge=0.0)
        # Honest-convention knobs (2026-07-14 adjudication).
        # Segment-gap fill fraction inside each packed subaperture circle:
        # the packed-circle collecting area otherwise books the full
        # pi*(d_sub/2)^2 although the Keck mosaic's inter-segment gaps
        # fall inside the circles (true-gap fill 0.9964 for the 3.14 m
        # KILO ring).
        subaperture_fill_fraction: float = Field(default=1.0, gt=0.0, le=1.0)
        # Delivered-flux attenuation of the WFS photon budget: the
        # default assumes the full budget always reaches the detector
        # regardless of the aberration being measured.
        wfs_delivered_flux_fraction: float = Field(default=1.0, gt=0.0, le=1.0)

    def apply(self, state: SimulationState) -> SimulationState:
        params = self.params_model
        wavelength_grid_um = _require_wavelength_grid_um(state)
        wavelength_grid_m = np.asarray(wavelength_grid_um, dtype=np.float64) * _MICRON_TO_M
        aperture_diameter_m = _require_aperture_diameter_m(state)
        r0_500nm_m = _require_r0_500nm_m(state)

        actuator_pitch_m = _actuator_pitch_m(aperture_diameter_m, params.n_actuators)
        piston_subaperture_diameter_m = _resolve_piston_subaperture_diameter_m(
            params, aperture_diameter_m
        )
        airmass, airmass_source = _resolve_airmass(state, params)
        r0_airmass_m = _airmass_scaled_r0_m(
            r0_500nm_m,
            wavelength_grid_m,
            airmass,
        )
        effective_r_mag, effective_r_mag_source = _effective_guide_star_r_mag(
            state,
            params,
        )
        effective_wind_speed = _effective_wind_speed_m_s(params, r0_500nm_m)

        residuals = _ao_residual_arrays(
            params=params,
            wavelength_grid_m=wavelength_grid_m,
            aperture_diameter_m=aperture_diameter_m,
            actuator_pitch_m=actuator_pitch_m,
            r0_airmass_m=r0_airmass_m,
            wind_speed_m_s=effective_wind_speed,
            guide_star_R_mag=effective_r_mag,
            airmass=airmass,
        )
        piston = _ao_piston_arrays(
            wavelength_grid_m=wavelength_grid_m,
            actuator_pitch_m=actuator_pitch_m,
            residuals=residuals,
            piston_subaperture_diameter_m=piston_subaperture_diameter_m,
        )

        representative_index = len(wavelength_grid_um) // 2
        wavefront = _updated_wavefront(
            state=state,
            wavelength_grid_um=wavelength_grid_um,
            residuals=residuals,
            component=_wavefront_component(
                params=params,
                aperture_diameter_m=aperture_diameter_m,
                actuator_pitch_m=actuator_pitch_m,
                r0_500nm_m=r0_500nm_m,
                r0_airmass_m=r0_airmass_m,
                effective_wind_speed_m_s=effective_wind_speed,
                wavelength_grid_um=wavelength_grid_um,
                residuals=residuals,
                piston=piston,
                piston_subaperture_diameter_m=piston_subaperture_diameter_m,
                effective_r_mag=effective_r_mag,
                effective_r_mag_source=effective_r_mag_source,
                airmass=airmass,
                airmass_source=airmass_source,
            ),
        )
        opd_budget = _updated_opd_budget(state, piston, representative_index)
        # Populate state.geometry.aperture_positions_pupil_m for downstream
        # consumers (chip_optimization kernel_order > 0 needs per-mode
        # ENU positions to build the kernel-null field stack).
        geometry = _populate_aperture_positions_pupil_m(state, params)
        return replace(
            state,
            opd_budget=opd_budget,
            wavefront=wavefront,
            geometry=geometry,
        )


def _populate_aperture_positions_pupil_m(state, params):
    """Compute global pupil-frame ENU positions for every chip-input mode.

    For an array of N_apertures telescopes each with N_sub sub-apertures,
    mode i = telescope (i // N_sub), sub-aperture (i % N_sub). Global
    position = telescope_enu[i // N_sub][:2] + sub_pupil_xy[i % N_sub].

    Also refines ``state.geometry.effective_collecting_area_m2`` to the
    sum of physical sub-aperture areas (``N_aper × N_sub × π·(d_sub/2)²``).
    Without this refinement the downstream detector and sensitivity
    stages use ``collecting_area_m2 / n_modes`` for the per-mode area,
    which is the *area-equivalent* full-pupil share rather than the
    packed sub-aperture footprint — KILO Keck packs ~44.7% of the
    obstructed primary into 12 d_sub≈1.93 m circles, so the area-
    equivalent overcount is ~2.24× per mode.

    Skipped when ``n_subapertures_per_aperture`` is None / 1 (no sub-
    aperture sampling needed). A single upstream telescope aperture is
    treated as an ENU origin when the telescope stage leaves positions
    unset, which is the NAYRA-in-one-E-ELT-pupil case.
    """

    if params.n_subapertures_per_aperture is None or params.n_subapertures_per_aperture <= 1:
        return state.geometry
    enu = state.geometry.aperture_positions_enu_m
    if enu is None:
        if state.geometry.n_apertures != 1:
            return state.geometry
        enu_arr = np.zeros((1, 3), dtype=np.float64)
    else:
        enu_arr = np.asarray(enu, dtype=np.float64)
    diameter_m = state.geometry.diameter_m
    if diameter_m is None or diameter_m <= 0:
        return state.geometry

    sub_positions_tuple, d_sub = _packed_subaperture_layout(
        int(params.n_subapertures_per_aperture),
        float(diameter_m),
        params.pupil_template,
        float(params.pupil_central_obstruction_diameter_m),
    )
    sub_positions = np.asarray(sub_positions_tuple, dtype=np.float64)  # (N_sub, 2)
    n_apertures = enu_arr.shape[0]
    n_sub = sub_positions.shape[0]
    n_modes = n_apertures * n_sub
    out = np.empty((n_modes, 2), dtype=np.float64)
    for t in range(n_apertures):
        for s in range(n_sub):
            out[t * n_sub + s] = enu_arr[t, :2] + sub_positions[s]

    fill = float(params.subaperture_fill_fraction)
    effective_collecting_area_m2 = (
        float(n_modes) * math.pi * (float(d_sub) / 2.0) ** 2 * fill
    )
    mode_area_m2 = math.pi * (float(d_sub) / 2.0) ** 2 * fill
    full_collecting_area_m2 = state.geometry.collecting_area_m2
    fill_fraction = (
        None
        if full_collecting_area_m2 is None or float(full_collecting_area_m2) <= 0.0
        else effective_collecting_area_m2 / float(full_collecting_area_m2)
    )
    geometry = replace(
        state.geometry,
        aperture_positions_pupil_m=out,
        effective_collecting_area_m2=effective_collecting_area_m2,
    )
    return geometry.add_component(
        "ao_subaperture_packing",
        {
            "n_parent_apertures": int(n_apertures),
            "n_subapertures_per_aperture": int(n_sub),
            "n_modes": int(n_modes),
            "mode_areas_m2": np.full(n_modes, mode_area_m2, dtype=np.float64),
            "mode_diameters_m": np.full(n_modes, float(d_sub), dtype=np.float64),
            # Telescope origin per mode: only this part of a mode position
            # foreshortens off-zenith; the pupil-mounted sub-aperture
            # offset tracks the target (review 2026-06-10 Issue 20).
            "mode_origin_positions_m": np.repeat(
                enu_arr[:, :2], n_sub, axis=0
            ).astype(np.float64),
            "pupil_template": params.pupil_template,
            "piston_subaperture_diameter_m": float(d_sub),
            "effective_collecting_area_m2": float(effective_collecting_area_m2),
            "full_collecting_area_m2": (
                None
                if full_collecting_area_m2 is None
                else float(full_collecting_area_m2)
            ),
            "effective_to_full_collecting_area_fraction": fill_fraction,
            "subaperture_fill_fraction": fill,
            "effective_area_source": "packed_subaperture_circles",
        },
    )


def _require_wavelength_grid_um(state: SimulationState) -> tuple[float, ...]:
    wavelength_grid_um = (
        state.wavefront.wavelength_grid_um
        or state.field.wavelength_grid_um
        or state.throughput.wavelength_grid_um
        or state.photon_rates.wavelength_grid_um
    )
    if not wavelength_grid_um:
        raise ValueError(
            "ao.pyramid_wfs requires a non-empty wavelength grid from the resolved grid config."
        )
    if state.wavefront.strehl_ratio and len(state.wavefront.strehl_ratio) != len(wavelength_grid_um):
        raise ValueError(
            "ao.pyramid_wfs requires wavefront.strehl_ratio to match the wavelength grid length."
        )
    return wavelength_grid_um


def _require_aperture_diameter_m(state: SimulationState) -> float:
    aperture_diameter_m = state.geometry.diameter_m
    if aperture_diameter_m is None:
        raise ValueError(
            "ao.pyramid_wfs requires geometry.diameter_m from an earlier telescope stage."
        )
    if not math.isfinite(aperture_diameter_m) or aperture_diameter_m <= 0.0:
        raise ValueError(
            "ao.pyramid_wfs requires a finite positive geometry.diameter_m, "
            f"got {aperture_diameter_m}."
        )
    return aperture_diameter_m


def _effective_guide_star_r_mag(
    state: SimulationState,
    params: PyramidWFS.Params,
) -> tuple[float | None, str]:
    if not params.derive_r_mag_from_scene_jmag:
        return None, "params.guide_star_R_mag"

    scene_jmag = getattr(state.scene, "star_jmag", None)
    if scene_jmag is None:
        raise ValueError(
            "pyramid_wfs: derive_r_mag_from_scene_jmag=True but "
            "state.scene.star_jmag is None -- run scene.point_source "
            "or supply scene.star.jmag upstream."
        )
    if params.r_minus_j_offset is not None:
        return (
            float(scene_jmag) + float(params.r_minus_j_offset),
            "scene.star.jmag + params.r_minus_j_offset",
        )

    spectral_type = getattr(state.scene, "star_spectral_type", None)
    return (
        estimate_r_mag_from_j(
            float(scene_jmag),
            spectral_type,
            default_spectral_type=params.guide_star_default_spectral_type,
        ),
        "scene.star.jmag + spectral-type R-J color",
    )


def _require_r0_500nm_m(state: SimulationState) -> float:
    r0_500nm_m = state.wavefront.r0_at_500nm_m
    if r0_500nm_m is None:
        raise ValueError(
            "ao.pyramid_wfs requires wavefront.r0_at_500nm_m from "
            "atmosphere.kolmogorov."
        )
    r0_value = float(r0_500nm_m)
    if math.isinf(r0_value) or (math.isfinite(r0_value) and r0_value <= 0.0):
        raise ValueError(
            "ao.pyramid_wfs requires r0_at_500nm_m to be positive where finite."
        )
    return r0_value


def _actuator_pitch_m(aperture_diameter_m: float, n_actuators: int) -> float:
    """Projected DM actuator pitch on the primary, assuming square layout.

    Pre-fix this used ``int(math.sqrt(n_actuators))`` which silently
    rounded down — so ``n_actuators ∈ {4, 5, 6, 7, 8}`` all yielded
    ``n_across = 2`` (identical pitch) and ``n_actuators ∈ {1, 2, 3}``
    yielded ``pitch = aperture_diameter_m`` (entire pupil). The
    Pydantic validator only enforced ``gt=0``, so wrong values silently
    flowed into the AO error budget.

    The user-decided convention is **strict k² square layouts only**:
    if ``n_actuators`` is not a perfect square ≥ 4, raise. Non-square
    DMs are a future feature requiring a Params schema change.
    Code Review 2026-05-16 Issue 9.
    """
    n_across = round(math.sqrt(n_actuators))
    if n_across * n_across != n_actuators or n_across < 2:
        raise ValueError(
            f"PyramidWFS requires n_actuators to be a perfect square k**2 "
            f"with k >= 2 (got n_actuators={n_actuators}). Non-square DMs "
            "would need a Params schema extension."
        )
    return aperture_diameter_m / n_across


@lru_cache(maxsize=64)
def _packed_subaperture_layout(
    n_subapertures_per_aperture: int,
    aperture_diameter_m: float,
    pupil_template: str | None,
    central_obstruction_diameter_m: float,
) -> tuple[tuple[tuple[float, float], ...], float]:
    """Packed sub-aperture positions and diameter for one pupil, cached.

    Single source for BOTH the piston-filter d_sub and the published mode
    positions/areas. Pre-fix the multi-seed placement search ran twice per
    stage application — once cached for d_sub, once uncached for
    positions — wasting ~0.5-2 s per cold cell per worker and leaving the
    two results equal only by determinism (review 2026-06-10 Issue 28b).
    The pupil resolution also unifies the two previously divergent paths:
    a configured central obstruction is honored for circular/untemplated
    pupils in both the filter and the published positions.
    """

    if pupil_template == "keck":
        pupil = keck_pupil()
    elif pupil_template == "eelt":
        pupil = eelt_pupil()
    else:
        pupil = PupilGeometry(
            central_obstruction_diameter_m=central_obstruction_diameter_m
        )
    positions, d_sub = optimize_subaperture_positions(
        n_subapertures_per_aperture, aperture_diameter_m, pupil
    )
    positions_arr = np.asarray(positions, dtype=np.float64)
    return (
        tuple((float(x), float(y)) for x, y in positions_arr),
        float(d_sub),
    )


def _derive_piston_subaperture_diameter_m(
    n_subapertures_per_aperture: int,
    aperture_diameter_m: float,
    pupil_template: str | None,
    central_obstruction_diameter_m: float,
) -> float:
    """Largest packed sub-aperture diameter (shared cached layout)."""

    _, d_sub = _packed_subaperture_layout(
        n_subapertures_per_aperture,
        aperture_diameter_m,
        pupil_template,
        central_obstruction_diameter_m,
    )
    return d_sub


def _resolve_piston_subaperture_diameter_m(
    params: PyramidWFS.Params,
    aperture_diameter_m: float,
) -> float | None:
    """Resolve the piston sub-aperture diameter for this run.

    Priority:
      1. Explicit ``params.piston_subaperture_diameter_m`` if set.
      2. ``pupil_template == "full_pupil"`` → ``aperture_diameter_m``
         directly (the SMF samples the entire pupil; matches NAYRA E-ELT
         and KILO 4-telescope reference conventions).
      3. ``params.n_subapertures_per_aperture`` set → derive via
         ``optimize_subaperture_positions`` against the chosen pupil
         template (``"keck"`` = hexagonal mosaic + 2.4 m obstruction +
         vanes; ``"eelt"`` = NAYRA ELT circular pupil + obstruction +
         vanes; ``"circular"`` = simple circular with the obstruction
         diameter from ``pupil_central_obstruction_diameter_m``).
      4. ``None`` → falls back to the historical fixed-fraction
         (1.0, 0.03, 1.0) filter in ``_piston_filter_fractions``.
    """

    if params.piston_subaperture_diameter_m is not None:
        return float(params.piston_subaperture_diameter_m)
    if params.pupil_template == "full_pupil":
        return float(aperture_diameter_m)
    if params.n_subapertures_per_aperture is None:
        return None
    return _derive_piston_subaperture_diameter_m(
        int(params.n_subapertures_per_aperture),
        float(aperture_diameter_m),
        params.pupil_template,
        float(params.pupil_central_obstruction_diameter_m),
    )


def _airmass_scaled_r0_m(
    r0_500nm_m: float,
    wavelength_grid_m: np.ndarray,
    airmass: float,
) -> np.ndarray:
    r0_zenith_m = np.asarray(kolmogorov.r0_scaled(r0_500nm_m, wavelength_grid_m))
    return r0_zenith_m * (airmass ** (-0.6))


def _resolve_airmass(
    state: SimulationState,
    params: PyramidWFS.Params,
) -> tuple[float, str]:
    """Resolve AO airmass from an explicit override or scene altitude."""

    if params.airmass is not None:
        return float(params.airmass), "params.airmass"

    altitude_deg = state.scene.altitude_deg
    if altitude_deg is None:
        return 1.0, "zenith_default"

    altitude = float(altitude_deg)
    if not math.isfinite(altitude) or altitude <= 0.0 or altitude > 90.0:
        raise ValueError(
            "ao.pyramid_wfs could not derive airmass from scene.altitude_deg="
            f"{altitude_deg!r}; expected 0 < altitude_deg <= 90 or set "
            "ao.pyramid_wfs.airmass explicitly."
        )
    return max(1.0, 1.0 / math.sin(math.radians(altitude))), "scene.altitude_deg"


def _effective_wind_speed_m_s(params: PyramidWFS.Params, r0_500nm_m: float) -> float:
    if params.derive_wind_from_tau0 and params.tau0_500nm_s:
        return kolmogorov.effective_wind_speed_m_s(
            r0_500nm_m=r0_500nm_m,
            tau0_500nm_s=params.tau0_500nm_s,
        )
    return float(params.v_wind_m_s)


def _ao_residual_arrays(
    *,
    params: PyramidWFS.Params,
    wavelength_grid_m: np.ndarray,
    aperture_diameter_m: float,
    actuator_pitch_m: float,
    r0_airmass_m: np.ndarray,
    wind_speed_m_s: float,
    guide_star_R_mag: float | None = None,
    airmass: float = 1.0,
) -> _ResidualArrays:
    fitting_rad2 = _fitting_phase_variance_rad2(actuator_pitch_m, r0_airmass_m)
    temporal_rad2 = _temporal_phase_variance_rad2(
        wind_speed_m_s,
        r0_airmass_m,
        params.loop_frequency_hz,
    )
    wfs_noise_rad2 = _wfs_noise_phase_variance_rad2(
        params=params,
        aperture_diameter_m=aperture_diameter_m,
        wavelength_grid_m=wavelength_grid_m,
        guide_star_R_mag=guide_star_R_mag,
        airmass=airmass,
    )
    total_rad2 = fitting_rad2 + temporal_rad2 + wfs_noise_rad2
    residual_opd_m = _phase_variance_to_opd_m(total_rad2, wavelength_grid_m)
    strehl_ratio = np.asarray(kolmogorov.strehl_marechal(residual_opd_m, wavelength_grid_m))
    return _ResidualArrays(
        fitting_rad2=fitting_rad2,
        temporal_rad2=temporal_rad2,
        wfs_noise_rad2=wfs_noise_rad2,
        total_rad2=total_rad2,
        strehl_ratio=strehl_ratio,
        residual_opd_m=residual_opd_m,
    )


def _fitting_phase_variance_rad2(
    actuator_pitch_m: float,
    r0_m: np.ndarray,
) -> np.ndarray:
    """KILO atmosphere.py ao_residual_variance fitting term."""

    return _FITTING_ERROR_COEFFICIENT * np.power(actuator_pitch_m / r0_m, _FIVE_THIRDS)


def _temporal_phase_variance_rad2(
    wind_speed_m_s: float,
    r0_m: np.ndarray,
    loop_frequency_hz: float,
) -> np.ndarray:
    """KILO atmosphere.py ao_residual_variance Greenwood temporal term."""

    greenwood_frequency_hz = _GREENWOOD_FREQUENCY_COEFFICIENT * wind_speed_m_s / r0_m
    servo_frequency_hz = loop_frequency_hz / _SERVO_BANDWIDTH_DIVISOR
    return np.power(greenwood_frequency_hz / servo_frequency_hz, _FIVE_THIRDS)


def _wfs_noise_phase_variance_rad2(
    *,
    params: PyramidWFS.Params,
    aperture_diameter_m: float,
    wavelength_grid_m: np.ndarray,
    guide_star_R_mag: float | None = None,
    airmass: float = 1.0,
) -> np.ndarray:
    """KILO pwfs.py photon-noise term, scaled from WFS to science wavelength."""

    sigma2_at_wfs_rad2 = _wfs_noise_phase_variance_at_wfs_rad2(
        params=params,
        aperture_diameter_m=aperture_diameter_m,
        guide_star_R_mag=guide_star_R_mag,
        airmass=airmass,
    )
    wfs_wavelength_m = params.wfs_wavelength_um * _MICRON_TO_M
    return sigma2_at_wfs_rad2 * np.square(wfs_wavelength_m / wavelength_grid_m)


def _wfs_noise_phase_variance_at_wfs_rad2(
    *,
    params: PyramidWFS.Params,
    aperture_diameter_m: float,
    guide_star_R_mag: float | None = None,
    airmass: float = 1.0,
) -> float:
    photons_per_frame = _wfs_photons_per_subap_per_frame(
        params=params,
        aperture_diameter_m=aperture_diameter_m,
        guide_star_R_mag=guide_star_R_mag,
        airmass=airmass,
    )
    if photons_per_frame <= 0.0:
        return math.inf
    sensitivity = _pyramid_sensitivity(params.modulation_lambda_d)
    measurement_variance = params.excess_noise_sq * (
        photons_per_frame + params.readout_noise_e_rms**2
    )
    return measurement_variance / (photons_per_frame**2 * sensitivity**2)


def _wfs_photons_per_subap_per_frame(
    *,
    params: PyramidWFS.Params,
    aperture_diameter_m: float,
    guide_star_R_mag: float | None = None,
    airmass: float = 1.0,
) -> float:
    # Allow callers to pass an effective R-mag (e.g. derived from
    # scene.star.jmag); fall back to the Params default otherwise.
    effective_r_mag = (
        float(guide_star_R_mag)
        if guide_star_R_mag is not None
        else params.guide_star_R_mag
    )
    flux_top_atm_ph_s_m2 = (
        _VEGA_R_ZEROPOINT_PH_S_M2_UM
        * 10.0 ** (-0.4 * effective_r_mag)
        * params.wfs_bandwidth_um
    )
    collecting_area_m2 = (
        math.pi
        * (aperture_diameter_m / 2.0) ** 2
        * (1.0 - params.central_obstruction**2)
    )
    atmospheric_throughput = params.atmospheric_throughput**float(airmass)
    rate_at_detector_hz = (
        flux_top_atm_ph_s_m2
        * collecting_area_m2
        * atmospheric_throughput
        * params.mirror_throughput
        * params.dichroic_throughput
        * params.quantum_efficiency
        * params.wfs_delivered_flux_fraction
    )
    n_subap = max(1, _n_subap_illuminated(params))
    return rate_at_detector_hz / (n_subap * params.loop_frequency_hz)


def _n_subap_illuminated(params: PyramidWFS.Params) -> int:
    fill_fraction = math.pi * (1.0 - params.central_obstruction**2) / 4.0
    return int(round(params.n_subap_across * params.n_subap_across * fill_fraction))


def _pyramid_sensitivity(modulation_lambda_d: float) -> float:
    return 1.0 / (1.0 + 2.0 * modulation_lambda_d)


def _ao_piston_arrays(
    *,
    wavelength_grid_m: np.ndarray,
    actuator_pitch_m: float,
    residuals: _ResidualArrays,
    piston_subaperture_diameter_m: float | None,
) -> _PistonArrays:
    frac_temporal, frac_fitting, frac_wfs_noise = _piston_filter_fractions(
        piston_subaperture_diameter_m,
        actuator_pitch_m,
    )
    temporal_rad2 = frac_temporal * residuals.temporal_rad2
    fitting_rad2 = frac_fitting * residuals.fitting_rad2
    wfs_noise_rad2 = frac_wfs_noise * residuals.wfs_noise_rad2
    total_rad2 = temporal_rad2 + fitting_rad2 + wfs_noise_rad2
    return _PistonArrays(
        temporal_rad2=temporal_rad2,
        fitting_rad2=fitting_rad2,
        wfs_noise_rad2=wfs_noise_rad2,
        total_rad2=total_rad2,
        temporal_opd_m=_phase_variance_to_opd_m(temporal_rad2, wavelength_grid_m),
        fitting_opd_m=_phase_variance_to_opd_m(fitting_rad2, wavelength_grid_m),
        wfs_noise_opd_m=_phase_variance_to_opd_m(wfs_noise_rad2, wavelength_grid_m),
        total_opd_m=_phase_variance_to_opd_m(total_rad2, wavelength_grid_m),
        frac_temporal=frac_temporal,
        frac_fitting=frac_fitting,
        frac_wfs_noise=frac_wfs_noise,
    )


def _piston_filter_fractions(
    piston_subaperture_diameter_m: float | None,
    actuator_pitch_m: float,
) -> tuple[float, float, float]:
    """Return ``(frac_temporal, frac_fitting, frac_wfs_noise)`` of the
    spatial PSD that survives the SMF piston-mode filter.

    PSD exponents per KILO ``kilo/atmosphere.py``
    ``ao_piston_phase_components`` lines 845-858 (HEAD ``47607de``):

      * **Fitting** (``-11/3``, integrated over ``k ∈ [k_DM, 200] m⁻¹``):
        Kolmogorov spatial PSD restricted to the uncorrected high-k tail
        that the DM cannot reach. Standard Hardy 1998 result.

      * **Temporal** (``-5/3``, integrated over ``k ∈ [0, k_DM] m⁻¹``):
        servo-lag residual spatial PSD slope. NOT the Kolmogorov
        ``-11/3`` exponent — KILO models the lag residual as
        concentrated at low spatial frequencies with the same exponent
        (``5/3``) that governs the temporal-domain Greenwood-frequency
        scaling. The (Hardy / Roddier / arXiv 1906.03128) literature
        doesn't give a single closed-form spatial exponent for the
        servo-lag residual — it's the Kolmogorov PSD weighted by the
        servo's temporal rejection. KILO's ``-5/3`` is a deliberate
        modeling choice that captures the low-spatial-frequency
        concentration of lag residual; it is the intended convention,
        not a port bug. Code Review 2026-05-16 Issue 10 (REFUTED).

      * **WFS noise** (``0``, integrated over ``k ∈ [0, k_DM] m⁻¹``):
        white spatial PSD. Photon-noise floor of the WFS measurement.

    When ``piston_subaperture_diameter_m`` is ``None``, fall back to
    fixed fractions ``(1.0, 0.03, 1.0)`` (no SMF mode filter applied).
    """

    if piston_subaperture_diameter_m is None:
        return 1.0, 0.03, 1.0

    dm_cutoff_m_inv = 1.0 / (2.0 * actuator_pitch_m)
    frac_fitting = _piston_filter_fraction(
        piston_subaperture_diameter_m,
        psd_exponent=-11.0 / 3.0,
        k_low_m_inv=dm_cutoff_m_inv,
        k_high_m_inv=200.0,
    )
    frac_temporal = _piston_filter_fraction(
        piston_subaperture_diameter_m,
        psd_exponent=-5.0 / 3.0,
        k_low_m_inv=0.0,
        k_high_m_inv=dm_cutoff_m_inv,
    )
    frac_wfs_noise = _piston_filter_fraction(
        piston_subaperture_diameter_m,
        psd_exponent=0.0,
        k_low_m_inv=0.0,
        k_high_m_inv=dm_cutoff_m_inv,
    )
    return frac_temporal, frac_fitting, frac_wfs_noise


@lru_cache(maxsize=256)
def _piston_filter_fraction(
    piston_subaperture_diameter_m: float,
    *,
    psd_exponent: float,
    k_low_m_inv: float,
    k_high_m_inv: float,
) -> float:
    """KILO atmosphere.py _piston_filter_fraction Bessel piston integral.

    Fraction of the radially integrated PSD ``k^p * k`` on
    ``[k_low, k_high]`` that survives the piston filter
    ``(2 J1(pi k d) / (pi k d))^2`` (aperture-average piston filter). The
    unfiltered denominator is the closed form
    ``(k_high^(p+2) - k_low^(p+2)) / (p+2)``; for the temporal term
    (``p = -5/3``, ``k_low = 0``) that is ``3 k_c^(1/3)``. Both integrals
    start at ``k_low`` itself: the ``k^(-2/3)`` endpoint singularity is
    integrable and QUADPACK handles it, whereas the former ``1e-6``
    lower clamp dropped ``3 (1e-6)^(1/3) = 0.03`` from both integrals and
    biased the fraction low by ~1%.

    lru-cached on raw float inputs. The quantized-input scheme documented
    by Code Review 2026-05-16 Issue 61 was never implemented; in practice
    ``d_sub`` comes from one deterministic derivation per run, so
    near-equal-float misses cannot occur on the production path (review
    2026-06-10 Issue 28d). scipy imports hoisted to module load (Issue 60).
    """

    quad = _scipy_integrate_quad
    j1 = _scipy_special_j1

    if k_high_m_inv <= k_low_m_inv or k_low_m_inv < 0.0:
        return 0.0

    lower = k_low_m_inv
    j1_zeros = np.array([3.8317, 7.0156, 10.1735, 13.3237, 16.4706])
    k_zeros = j1_zeros / (math.pi * piston_subaperture_diameter_m)
    breakpoints = k_zeros[(k_zeros > lower) & (k_zeros < k_high_m_inv)]
    edges = np.concatenate([[lower], breakpoints, [k_high_m_inv]])

    numerator = 0.0
    for start, stop in zip(edges[:-1], edges[1:]):
        value, _ = quad(
            lambda k_m_inv: _piston_integrand_filtered(
                k_m_inv,
                piston_subaperture_diameter_m,
                psd_exponent,
                j1,
            ),
            start,
            stop,
            limit=100,
        )
        numerator += value
    power = psd_exponent + 2.0
    denominator = (k_high_m_inv**power - lower**power) / power
    return float(max(0.0, min(1.0, numerator / denominator)))


def _piston_integrand_filtered(
    k_m_inv: float,
    piston_subaperture_diameter_m: float,
    psd_exponent: float,
    bessel_j1: Any,
) -> float:
    argument = math.pi * k_m_inv * piston_subaperture_diameter_m
    if argument < 1.0e-12:
        piston_filter = 1.0
    else:
        piston_filter = (2.0 * bessel_j1(argument) / argument) ** 2
    return k_m_inv**psd_exponent * piston_filter * k_m_inv


def _phase_variance_to_opd_m(
    phase_variance_rad2: np.ndarray,
    wavelength_grid_m: np.ndarray,
) -> np.ndarray:
    return np.sqrt(phase_variance_rad2) * wavelength_grid_m / (2.0 * math.pi)


def _updated_wavefront(
    *,
    state: SimulationState,
    wavelength_grid_um: tuple[float, ...],
    residuals: _ResidualArrays,
    component: dict[str, Any],
) -> Wavefront:
    amplitude_jitter_fraction = (
        state.wavefront.amplitude_jitter_fraction
        if state.wavefront.amplitude_jitter_fraction
        else tuple(0.0 for _ in wavelength_grid_um)
    )
    return replace(
        state.wavefront,
        wavelength_grid_um=wavelength_grid_um,
        strehl_ratio=_float_tuple(residuals.strehl_ratio),
        amplitude_jitter_fraction=amplitude_jitter_fraction,
    ).add_component("ao_residual", component)


def _updated_opd_budget(
    state: SimulationState,
    piston: _PistonArrays,
    representative_index: int,
) -> OPDBudget:
    temporal_opd_nm = float(piston.temporal_opd_m[representative_index] * _M_TO_NM)
    fitting_opd_nm = float(piston.fitting_opd_m[representative_index] * _M_TO_NM)
    wfs_noise_opd_nm = float(piston.wfs_noise_opd_m[representative_index] * _M_TO_NM)

    # Mark the atmospheric piston contribution superseded — AO corrects it —
    # so it stays in components for plot history but stops counting toward the
    # quadrature total. See SPEC §6.2.
    updated_components: dict[str, Any] = dict(state.opd_budget.components)
    atmos = updated_components.get("atmos_piston")
    if isinstance(atmos, OPDComponent) and not atmos.superseded:
        updated_components["atmos_piston"] = replace(atmos, superseded=True)
    for name, rms_opd_nm in (
        ("ao_temporal", temporal_opd_nm),
        ("ao_fitting", fitting_opd_nm),
        ("ao_wfs_noise", wfs_noise_opd_nm),
    ):
        updated_components[name] = OPDComponent(
            rms_opd_nm=rms_opd_nm,
            reduction_stage="pyramid_wfs",
        )

    total_rms_opd_nm = math.sqrt(
        sum(
            component.rms_opd_nm**2
            for component in updated_components.values()
            if isinstance(component, OPDComponent) and not component.superseded
        )
    )
    return replace(
        state.opd_budget,
        components=updated_components,
        total_rms_opd_nm=total_rms_opd_nm,
    )


def _wavefront_component(
    *,
    params: PyramidWFS.Params,
    aperture_diameter_m: float,
    actuator_pitch_m: float,
    r0_500nm_m: float,
    r0_airmass_m: np.ndarray,
    effective_wind_speed_m_s: float,
    wavelength_grid_um: tuple[float, ...],
    residuals: _ResidualArrays,
    piston: _PistonArrays,
    piston_subaperture_diameter_m: float | None,
    effective_r_mag: float | None = None,
    effective_r_mag_source: str = "params.guide_star_R_mag",
    airmass: float = 1.0,
    airmass_source: str = "params.airmass",
) -> dict[str, Any]:
    # A piston uniform over the coupled subaperture shifts the phase of the
    # light it couples into a single-mode fiber, not the coupled power, and
    # the OPD budget already books it for the realized null. SMF coupling
    # therefore sees only the piston-removed residual over the subaperture:
    # S_sub = exp(-(sigma2_total - sigma2_piston,sub)) (Ruilier & Cassaing
    # 2001, coupling with residual phase variance; Marechal form).
    subaperture_rad2 = residuals.total_rad2 - piston.total_rad2
    return {
        "aperture_diameter_m": aperture_diameter_m,
        "actuator_pitch_m": actuator_pitch_m,
        "n_actuators": params.n_actuators,
        "n_subap_across": params.n_subap_across,
        "n_subap_illuminated": _n_subap_illuminated(params),
        "wavelength_grid_um": wavelength_grid_um,
        "r0_500nm_m": r0_500nm_m,
        "r0_airmass_m": _float_tuple(r0_airmass_m),
        "tau0_500nm_s": params.tau0_500nm_s,
        "theta0_500nm_arcsec": params.theta0_500nm_arcsec,
        "v_wind_m_s": params.v_wind_m_s,
        "derive_wind_from_tau0": params.derive_wind_from_tau0,
        "effective_wind_speed_m_s": effective_wind_speed_m_s,
        "airmass": float(airmass),
        "airmass_source": airmass_source,
        "loop_frequency_hz": params.loop_frequency_hz,
        "guide_star_R_mag": (
            float(effective_r_mag)
            if effective_r_mag is not None
            else params.guide_star_R_mag
        ),
        "guide_star_R_mag_source": effective_r_mag_source,
        "guide_star_default_spectral_type": params.guide_star_default_spectral_type,
        "wfs_wavelength_um": params.wfs_wavelength_um,
        "wfs_bandwidth_um": params.wfs_bandwidth_um,
        "wfs_photons_per_subap_per_frame": _wfs_photons_per_subap_per_frame(
            params=params,
            aperture_diameter_m=aperture_diameter_m,
            guide_star_R_mag=effective_r_mag,
            airmass=airmass,
        ),
        "wfs_noise_phase_variance_at_wfs_rad2": _wfs_noise_phase_variance_at_wfs_rad2(
            params=params,
            aperture_diameter_m=aperture_diameter_m,
            guide_star_R_mag=effective_r_mag,
            airmass=airmass,
        ),
        "residual_phase_variance_rad2": {
            "fitting": _float_scalar_or_tuple(residuals.fitting_rad2),
            "temporal": _float_scalar_or_tuple(residuals.temporal_rad2),
            "wfs_noise": _float_scalar_or_tuple(residuals.wfs_noise_rad2),
            "total": _float_scalar_or_tuple(residuals.total_rad2),
        },
        "residual_opd_nm": _float_scalar_or_tuple(residuals.residual_opd_m * _M_TO_NM),
        "piston_phase_variance_rad2": {
            "temporal": _float_scalar_or_tuple(piston.temporal_rad2),
            "fitting": _float_scalar_or_tuple(piston.fitting_rad2),
            "wfs_noise": _float_scalar_or_tuple(piston.wfs_noise_rad2),
            "total": _float_scalar_or_tuple(piston.total_rad2),
        },
        "piston_filter_fraction": {
            "temporal": piston.frac_temporal,
            "fitting": piston.frac_fitting,
            "wfs_noise": piston.frac_wfs_noise,
        },
        "piston_opd_nm": {
            "temporal": _float_scalar_or_tuple(piston.temporal_opd_m * _M_TO_NM),
            "fitting": _float_scalar_or_tuple(piston.fitting_opd_m * _M_TO_NM),
            "wfs_noise": _float_scalar_or_tuple(piston.wfs_noise_opd_m * _M_TO_NM),
            "total": _float_scalar_or_tuple(piston.total_opd_m * _M_TO_NM),
        },
        "piston_subaperture_diameter_m": piston_subaperture_diameter_m,
        "n_subapertures_per_aperture": params.n_subapertures_per_aperture,
        "pupil_template": params.pupil_template,
        "strehl_ratio": _float_tuple(residuals.strehl_ratio),
        "subaperture_residual_phase_variance_rad2": _float_scalar_or_tuple(
            subaperture_rad2
        ),
        "subaperture_strehl_ratio": _float_tuple(np.exp(-subaperture_rad2)),
        # Marechal S = exp(-sigma^2) is only accurate for sigma^2 below
        # ~2 rad^2; at the faint end of an NGS J-mag sweep the residual can
        # exceed that, making the published Strehl pessimistic (review
        # 2026-06-10 Issue 28e). Flagged so downstream audits can see it.
        "strehl_marechal_max_phase_variance_rad2": float(
            np.max(residuals.total_rad2)
        ),
        "strehl_marechal_validity_exceeded": bool(
            np.max(residuals.total_rad2) > 2.0
        ),
        "source_model": "kilo.atmosphere.ao_residual_variance+piston_components",
    }


def _float_tuple(values: np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _float_scalar_or_tuple(values: np.ndarray) -> float | tuple[float, ...]:
    numbers = _float_tuple(values)
    if len(numbers) == 1:
        return numbers[0]
    return numbers


__all__ = ["PyramidWFS"]
