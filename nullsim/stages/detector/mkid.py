"""MKID detector stage (SPEC §15.3 Pass 4a).

Reference implementation: KILO ``detect_mkid`` in
``kilo/detectors.py`` lines 256-287.
Per-port total rate is split evenly across ``n_pixels`` MKIDs; each pixel
saturates independently at ``mkid_max_count_rate_hz`` (50 kHz KILO default
per ``kilo/config.py`` lines 18-32;
NAYRA E-ELT config bumps to 100 kHz per
``nayra/config.py`` lines
320-335). The nullsim stage exposes ``mkid_max_count_rate_hz`` so each
instrument config pins its own value. No MKID dark counts are modeled
(KILO does not model them at the detector layer).

See ``tests/stages/notes/detector_mkid.md`` for the full reference survey
and pinned invariants.

For scalar fields, the output is the usual one row per chip port. For dual-arm
PBS fields, each polarization arm is retained as a physical detector channel
(``bright_0_A``, ``bright_0_B``, ...), and per-pixel saturation is applied to
each physical channel with the bright/dark pixel pools selected by label.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field as PydanticField, model_validator

from nullsim.physics.mkid import (
    apply_energy_response,
    energy_response_matrix,
    energy_response_tail_loss,
    mkid_resolving_power,
)
from nullsim.physics.photometry import (
    photon_rate_per_m2_per_bin,
    stellar_band_magnitude_from_jmag,
)
from nullsim.pipeline import (
    BaseStage,
    PhotonRates,
    SimulationState,
    Throughput,
    register_stage,
)
from nullsim.stages._utils import unique_component_name


@register_stage
class MKIDDetector(BaseStage):
    """Apply MKID per-pixel QE and saturation to produce per-port count rates."""

    type_name = "detector.mkid"
    family = "detector"
    consumes = frozenset({"field", "throughput", "scene", "geometry"})
    produces = frozenset({"photon_rates", "results", "throughput"})

    class Params(BaseModel):
        """Validated params matching KILO/NAYRA MKID detector names."""

        model_config = ConfigDict(
            extra="forbid",
            frozen=True,
            allow_inf_nan=False,
        )

        qe: float = PydanticField(gt=0.0, le=1.0)
        mkid_max_count_rate_hz: float = PydanticField(default=5.0e4, gt=0.0)
        n_pixels_bright: int = PydanticField(ge=1)
        n_pixels_dark: int = PydanticField(ge=1)
        n_bright_ports: int = PydanticField(ge=0)
        n_dark_ports: int = PydanticField(ge=0)
        n_spectro_ports: int = PydanticField(default=0, ge=0)
        n_modes: int = PydanticField(ge=1)
        integration_time_s: float = PydanticField(gt=0.0)
        band: str = "J"
        derive_band_mag_from_scene_jmag: bool = False
        default_spectral_type: str = "F"
        resolving_power_ref: float | None = PydanticField(default=None, gt=0.0)
        lambda_ref_um: float | None = PydanticField(default=None, gt=0.0)

        @model_validator(mode="after")
        def _ports_sum_to_modes(self) -> MKIDDetector.Params:
            n_ports = (
                self.n_bright_ports + self.n_dark_ports + self.n_spectro_ports
            )
            if n_ports != self.n_modes:
                raise ValueError(
                    "n_bright_ports + n_dark_ports + n_spectro_ports must equal n_modes "
                    f"(got {self.n_bright_ports} + {self.n_dark_ports} + "
                    f"{self.n_spectro_ports} != {self.n_modes})."
                )
            return self

        @model_validator(mode="after")
        def _lambda_ref_required_when_resolution_enabled(self) -> MKIDDetector.Params:
            if self.resolving_power_ref is not None and self.lambda_ref_um is None:
                raise ValueError(
                    "lambda_ref_um is required when resolving_power_ref is set."
                )
            return self

    def apply(self, state: SimulationState) -> SimulationState:
        params = self.params_model
        field_amplitudes = _require_field_amplitudes(state, self.type_name)
        wavelength_grid_um = _require_wavelength_grid_um(
            state, field_amplitudes, self.type_name
        )
        if field_amplitudes.shape[0] != params.n_modes:
            raise ValueError(
                f"{self.type_name} expects field with {params.n_modes} modes, "
                f"got {field_amplitudes.shape[0]}."
            )
        n_pol_arms = int(field_amplitudes.shape[2])

        throughput_fraction = _throughput_or_unity(state, len(wavelength_grid_um))
        stellar_magnitude = _stellar_magnitude_for_band(
            state,
            params,
            self.type_name,
        )
        stellar_rate_per_m2_per_bin = _stellar_rate_per_m2_per_bin(
            stellar_magnitude,
            wavelength_grid_um,
            params.band,
        )
        collecting_area_m2 = _resolve_collecting_area_m2(state, self.type_name)

        port_rate_hz_per_wavelength = _port_rate_hz_per_wavelength(
            field_amplitudes=field_amplitudes,
            throughput_fraction=throughput_fraction,
            stellar_rate_per_m2_per_bin=stellar_rate_per_m2_per_bin,
            collecting_area_m2=collecting_area_m2,
            qe=params.qe,
        )

        # Bug #1: null_depth_raw must come from the PRE-saturation rates.
        # When both bright and dark ports saturate at the per-pixel cap,
        # post-cap mean(dark)/mean(bright) collapses to the pixel-count
        # ratio (n_pixels_dark / n_pixels_bright) and silently misreports
        # the chip's actual null depth. KILO's null_depth() (kilo/
        # performance.py L164) is the stellar-power dark/total ratio
        # computed from |U @ field|^2 before any detector quantization.
        port_labels = _port_labels(
            params.n_bright_ports,
            params.n_dark_ports,
            params.n_spectro_ports,
            n_pol=n_pol_arms,
            pol_labels=_polarization_labels(state, n_pol_arms),
        )
        uncapped_port_rates_hz = port_rate_hz_per_wavelength.sum(axis=1)
        null_depth_raw = _null_depth_raw(
            port_rates_hz=uncapped_port_rates_hz,
            port_labels=port_labels,
            n_bright_ports=params.n_bright_ports,
        )

        detector_port_rate_hz_per_wavelength = port_rate_hz_per_wavelength
        response_matrix = None
        tail_loss_per_true_bin = None
        resolving_power_per_bin = None
        if params.resolving_power_ref is not None:
            if len(wavelength_grid_um) < 2:
                raise ValueError(
                    "detector.mkid energy resolution requires at least two "
                    "wavelength bins. Increase grid.n_wavelength_bins or omit "
                    "resolving_power_ref for monochromatic runs."
                )
            lambda_ref_um = float(params.lambda_ref_um)
            response_matrix = energy_response_matrix(
                wavelength_grid_um,
                wavelength_grid_um,
                float(params.resolving_power_ref),
                lambda_ref_um,
            )
            tail_loss_per_true_bin = energy_response_tail_loss(
                wavelength_grid_um,
                wavelength_grid_um,
                float(params.resolving_power_ref),
                lambda_ref_um,
            )
            resolving_power_per_bin = mkid_resolving_power(
                wavelength_grid_um,
                float(params.resolving_power_ref),
                lambda_ref_um,
            )
            detector_port_rate_hz_per_wavelength = apply_energy_response(
                port_rate_hz_per_wavelength,
                response_matrix,
            )

        capped_port_rate_hz_per_wavelength, saturated_per_port = _apply_per_pixel_cap(
            port_rate_hz_per_wavelength=detector_port_rate_hz_per_wavelength,
            port_labels=port_labels,
            n_pixels_bright=params.n_pixels_bright,
            n_pixels_dark=params.n_pixels_dark,
            cap_hz_per_pixel=params.mkid_max_count_rate_hz,
        )

        port_rates_hz = capped_port_rate_hz_per_wavelength.sum(axis=1)
        photon_rates = PhotonRates(
            wavelength_grid_um=wavelength_grid_um,
            port_labels=port_labels,
            rates_hz=capped_port_rate_hz_per_wavelength,
        )

        component_name = unique_component_name(
            self.type_name,
            state.results.components,
            self.context.instance_id,
        )
        component_payload = {
            "null_depth_raw": float(null_depth_raw),
            "port_rates": port_rates_hz.astype(np.float64, copy=True),
            "port_labels": port_labels,
            "saturated_per_port": saturated_per_port,
            "detector_qe": float(params.qe),
            "mkid_max_count_rate_hz": float(params.mkid_max_count_rate_hz),
            "n_pixels_bright": int(params.n_pixels_bright),
            "n_pixels_dark": int(params.n_pixels_dark),
            "n_bright_ports": int(params.n_bright_ports),
            "n_dark_ports": int(params.n_dark_ports),
            "n_spectro_ports": int(params.n_spectro_ports),
            "n_modes": int(params.n_modes),
            "n_pol_arms": n_pol_arms,
            "n_physical_bright_ports": int(
                sum(label.startswith("bright") for label in port_labels)
            ),
            "n_physical_dark_ports": int(
                sum(label.startswith("dark") for label in port_labels)
            ),
            "n_physical_spectro_ports": int(
                sum(label.startswith("spectro") for label in port_labels)
            ),
            "integration_time_s": float(params.integration_time_s),
            "photometric_band": str(params.band).upper(),
            "star_magnitude": float(stellar_magnitude),
            "star_magnitude_source": _stellar_magnitude_source(params),
            "derive_band_mag_from_scene_jmag": bool(
                params.derive_band_mag_from_scene_jmag
            ),
            "integrated_counts_per_port": (
                port_rates_hz * params.integration_time_s
            ).astype(np.float64, copy=True),
            "uncapped_port_rate_hz_per_wavelength": (
                detector_port_rate_hz_per_wavelength.astype(np.float64, copy=True)
            ),
            # Per-port totals BEFORE the per-pixel saturation cap, the exact
            # pre-cap counterpart of ``port_rates``. Saturation is a readout
            # limit, not a statement about how many photons the star sends:
            # anything normalizing a physical quantity against the stellar flux
            # (the detection-mode HDC planet reference, for one) needs the
            # uncapped rate, and silently using the capped one understates the
            # star by the saturation ratio.
            "uncapped_port_rates": (
                detector_port_rate_hz_per_wavelength.sum(axis=1).astype(
                    np.float64, copy=True
                )
            ),
            "detector_saturated": bool(np.any(saturated_per_port)),
            "pre_energy_response_port_rate_hz_per_wavelength": (
                port_rate_hz_per_wavelength.astype(np.float64, copy=True)
            ),
            "collecting_area_m2": float(collecting_area_m2),
            "source_model": (
                "KILO detect_mkid (kilo/detectors.py L256-287) with "
                "per-pixel saturation; nullsim.physics.photometry for "
                "magnitude→photon-rate conversion."
            ),
        }
        if params.resolving_power_ref is not None:
            component_payload.update(
                {
                    "resolving_power_ref": float(params.resolving_power_ref),
                    "lambda_ref_um": lambda_ref_um,
                    "resolving_power_per_bin": np.asarray(
                        resolving_power_per_bin,
                        dtype=np.float64,
                    ).copy(),
                    "true_rates_hz_per_port_wavelength": (
                        port_rate_hz_per_wavelength.astype(np.float64, copy=True)
                    ),
                    "energy_response_matrix": np.asarray(
                        response_matrix,
                        dtype=np.float64,
                    ).copy(),
                    "energy_response_tail_loss_per_true_bin": np.asarray(
                        tail_loss_per_true_bin,
                        dtype=np.float64,
                    ).copy(),
                    "source_model": (
                        "KILO detect_mkid (kilo/detectors.py L256-287) with "
                        "per-pixel saturation plus finite MKID energy-response "
                        "redistribution; nullsim.physics.photometry for "
                        "magnitude→photon-rate conversion."
                    ),
                }
            )
        results = state.results.add_component(component_name, component_payload)
        throughput = _fold_detector_qe(
            state=state,
            qe=float(params.qe),
            wavelength_grid_um=wavelength_grid_um,
            instance_id=self.context.instance_id,
        )

        return replace(
            state,
            photon_rates=photon_rates,
            results=results,
            throughput=throughput,
        )


def _require_field_amplitudes(state: SimulationState, type_name: str) -> NDArray[np.complex128]:
    amplitudes = state.field.amplitudes
    if amplitudes is None:
        raise ValueError(f"{type_name} requires field amplitudes.")
    if amplitudes.ndim != 3:
        raise ValueError(
            f"{type_name} requires field amplitudes with shape "
            "[n_modes, n_wavelength_bins, n_pol]."
        )
    return amplitudes


def _require_wavelength_grid_um(
    state: SimulationState,
    field_amplitudes: NDArray[np.complex128],
    type_name: str,
) -> tuple[float, ...]:
    wavelength_grid_um = state.field.wavelength_grid_um or state.throughput.wavelength_grid_um
    if not wavelength_grid_um:
        raise ValueError(f"{type_name} requires a wavelength grid on field or throughput.")
    if len(wavelength_grid_um) != field_amplitudes.shape[1]:
        raise ValueError(
            f"{type_name} wavelength grid length ({len(wavelength_grid_um)}) "
            f"must match field wavelength axis ({field_amplitudes.shape[1]})."
        )
    return wavelength_grid_um


def _throughput_or_unity(
    state: SimulationState,
    n_wavelength_bins: int,
) -> NDArray[np.float64]:
    if state.throughput.transmission_fraction:
        transmission = np.asarray(state.throughput.transmission_fraction, dtype=np.float64)
        if transmission.size != n_wavelength_bins:
            raise ValueError(
                f"throughput.transmission_fraction length ({transmission.size}) "
                f"must match wavelength axis ({n_wavelength_bins})."
            )
        return transmission
    return np.ones(n_wavelength_bins, dtype=np.float64)


def _fold_detector_qe(
    *,
    state: SimulationState,
    qe: float,
    wavelength_grid_um: tuple[float, ...],
    instance_id: str,
) -> Throughput:
    """Multiply the running throughput by the detector QE.

    The MKID stage already folds QE into the port photon rates, but the
    extracted ``throughput.transmission_fraction`` should also reflect QE so
    that throughput-breakdown outputs see a QE-aware end-to-end transmission.
    """

    n_bins = len(wavelength_grid_um)
    per_channel = np.full(n_bins, float(qe), dtype=np.float64)
    current = (
        np.asarray(state.throughput.transmission_fraction, dtype=np.float64)
        if state.throughput.transmission_fraction
        else np.ones(n_bins, dtype=np.float64)
    )
    if current.size != n_bins:
        raise ValueError(
            "detector.mkid: throughput.transmission_fraction length "
            f"{current.size} does not match wavelength grid length {n_bins}."
        )
    updated = current * per_channel

    component_name = unique_component_name(
        "detector.mkid_qe",
        state.throughput.components,
        instance_id,
    )
    return Throughput(
        wavelength_grid_um=wavelength_grid_um,
        transmission_fraction=tuple(float(v) for v in updated),
        components={
            **state.throughput.components,
            component_name: {
                "wavelength_grid_um": wavelength_grid_um,
                "transmission_fraction": tuple(float(v) for v in per_channel),
                "qe": float(qe),
                "source_model": (
                    "MKID detector quantum efficiency folded into the "
                    "throughput chain (mirrors KILO/NAYRA pre-chip × QE "
                    "convention)."
                ),
            },
        },
    )


def _bandwidth_per_bin_um(wavelength_grid_um: tuple[float, ...]) -> NDArray[np.float64]:
    wl = np.asarray(wavelength_grid_um, dtype=np.float64)
    if wl.size == 1:
        # Single-bin case: use the wavelength value itself as a stand-in for a
        # 100% fractional bandwidth. Single-wavelength grids are a degenerate
        # case used only for unit testing.
        return np.array([wl[0]], dtype=np.float64)
    edges = np.empty(wl.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (wl[:-1] + wl[1:])
    edges[0] = wl[0] - 0.5 * (wl[1] - wl[0])
    edges[-1] = wl[-1] + 0.5 * (wl[-1] - wl[-2])
    return np.diff(edges)


def _stellar_rate_per_m2_per_bin(
    magnitude: float,
    wavelength_grid_um: tuple[float, ...],
    band: str,
) -> NDArray[np.float64]:
    bandwidth_per_bin_um = _bandwidth_per_bin_um(wavelength_grid_um)
    return np.array(
        [
            photon_rate_per_m2_per_bin(
                magnitude,
                float(wl_um),
                float(bw_um),
                band=band,
            )
            for wl_um, bw_um in zip(wavelength_grid_um, bandwidth_per_bin_um)
        ],
        dtype=np.float64,
    )


def _stellar_magnitude_source(params: MKIDDetector.Params) -> str:
    if params.derive_band_mag_from_scene_jmag:
        return (
            "scene.star.jmag converted to detector band with "
            "nullsim.physics.photometry.jmag_to_band_mag."
        )
    return "scene.star.jmag interpreted directly in detector band."


def _stellar_magnitude_for_band(
    state: SimulationState,
    params: MKIDDetector.Params,
    type_name: str,
) -> float:
    if state.scene.star_jmag is None:
        raise ValueError(f"{type_name} requires scene.star_jmag.")
    return stellar_band_magnitude_from_jmag(
        float(state.scene.star_jmag),
        params.band,
        state.scene.star_spectral_type,
        derive_from_jmag=bool(params.derive_band_mag_from_scene_jmag),
        default_spectral_type=params.default_spectral_type,
    )


def _require_area_m2(state: SimulationState, type_name: str) -> float:
    if state.geometry.area_m2 is None:
        raise ValueError(f"{type_name} requires geometry.area_m2.")
    return float(state.geometry.area_m2)


def _resolve_collecting_area_m2(
    state: SimulationState, type_name: str
) -> float:
    """Total per-mode-summed light-coupling area for absolute photon rates.

    Prefers ``geometry.effective_collecting_area_m2`` — the sum of
    physical sub-aperture areas — so per-mode rates ``A_eff / n_modes``
    reflect the packed sub-aperture footprint, not the area-equivalent
    full-pupil share. KILO Keck: 12 d_sub≈1.93 m sub-apertures per
    telescope pack ~44.7% of the 10 m obstructed primary, so the area-
    equivalent overcount is ~2.24× per mode (KILO_PARITY_PLAN §1.1).

    Falls back to ``collecting_area_m2`` (full-pupil obstructed area)
    when no sub-aperture-aware stage has refined the geometry. Used by
    HWO-style single-mode pipelines that don't decompose the pupil.
    Final fallback is ``n_apertures * area_m2`` then ``area_m2`` for
    legacy single-aperture configs.
    """
    if state.geometry.effective_collecting_area_m2 is not None:
        return float(state.geometry.effective_collecting_area_m2)
    if state.geometry.collecting_area_m2 is not None:
        return float(state.geometry.collecting_area_m2)
    area_m2 = _require_area_m2(state, type_name)
    if state.geometry.n_apertures is None:
        return area_m2
    return float(state.geometry.n_apertures) * area_m2


def _port_rate_hz_per_wavelength(
    *,
    field_amplitudes: NDArray[np.complex128],
    throughput_fraction: NDArray[np.float64],
    stellar_rate_per_m2_per_bin: NDArray[np.float64],
    collecting_area_m2: float,
    qe: float,
) -> NDArray[np.float64]:
    # Preserve the polarization axis as physical detector arms. Shape after
    # transpose/reshape is (mode-major physical ports, n_wavelength).
    power_per_mode_pol = (np.abs(field_amplitudes) ** 2).astype(
        np.float64,
        copy=False,
    )
    # The chip's input convention is unit |amplitude|² per mode (one sub-
    # aperture's share of the collecting area). For a unitary chip with
    # M modes in and M out, sum_modes |U @ field|² == sum_modes |field|² == M,
    # so each mode carries collecting_area_m2 / M of the total photon flux.
    # Multiplicative chip losses (modeled in field amplitudes) reduce this
    # in proportion. KILO uses the same area-per-sub × |U field|² convention
    # (kilo/studies/natural_design.py: effective_area = sum(coupling * areas)).
    n_modes = power_per_mode_pol.shape[0]
    area_per_mode_m2 = collecting_area_m2 / float(n_modes)
    photon_rate = (
        stellar_rate_per_m2_per_bin * throughput_fraction * area_per_mode_m2
    )  # shape (n_wavelength,)
    power_per_physical_port = np.transpose(power_per_mode_pol, (0, 2, 1)).reshape(
        n_modes * power_per_mode_pol.shape[2],
        power_per_mode_pol.shape[1],
    )
    return power_per_physical_port * photon_rate[np.newaxis, :] * qe


def _apply_per_pixel_cap(
    *,
    port_rate_hz_per_wavelength: NDArray[np.float64],
    port_labels: tuple[str, ...],
    n_pixels_bright: int,
    n_pixels_dark: int,
    cap_hz_per_pixel: float,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Apply per-pixel saturation cap to each port's TOTAL rate.

    KILO ``detect_mkid`` (``kilo/detectors.py`` L256-287) caps the
    port-total rate divided by ``n_pixels``, not per-(port, wavelength)
    cell. An MKID pixel is energy-resolving but the saturation comes from
    the total photon arrival rate on that pixel, summed over all bins.

    Pre-fix this stage capped each ``(port, bin)`` cell independently
    then summed across bins, so a port that saturated uniformly across
    20 wavelength bins ended up at ``cap × n_pixels × 20`` (Code Review
    2026-05-16 Ultra-review High #4). The fix: aggregate over wavelengths
    to get the per-pixel total rate, cap that, and rescale every bin by
    the same factor so the spectral shape is preserved.
    """

    n_ports = port_rate_hz_per_wavelength.shape[0]
    if len(port_labels) != n_ports:
        raise ValueError(
            "_apply_per_pixel_cap: port_labels length does not match port-rate "
            f"axis ({len(port_labels)} != {n_ports})."
        )
    n_pixels = np.array(
        [
            n_pixels_bright if str(label).startswith("bright") else n_pixels_dark
            for label in port_labels
        ],
        dtype=np.float64,
    )

    port_total_rate = port_rate_hz_per_wavelength.sum(axis=1)
    per_pixel_total = port_total_rate / n_pixels
    saturated_per_port = per_pixel_total > cap_hz_per_pixel
    cap_total_per_port = cap_hz_per_pixel * n_pixels
    # Scale factor < 1 only for saturated ports; otherwise 1.0 (no change).
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(
            saturated_per_port,
            cap_total_per_port / port_total_rate,
            1.0,
        )
    scale = np.where(np.isfinite(scale), scale, 1.0)
    capped = port_rate_hz_per_wavelength * scale[:, np.newaxis]
    return capped, saturated_per_port


def _null_depth_raw(
    *,
    port_rates_hz: NDArray[np.float64],
    port_labels: tuple[str, ...],
    n_bright_ports: int,
) -> float:
    """Raw null depth = mean(dark port rates) / mean(bright port rates).

    Masks on ``port_labels`` rather than slicing positionally. Pre-fix
    used ``port_rates_hz[:n_bright_ports]`` which silently broke when a
    future refactor of ``_port_labels`` emits bright/dark in a different
    order. Code Review 2026-05-16 Issue 15.
    """

    if port_rates_hz.shape != (len(port_labels),):
        raise ValueError(
            "_null_depth_raw: port_rates and port_labels length mismatch "
            f"({port_rates_hz.shape} vs {len(port_labels)})."
        )
    bright_mask = np.array(
        [str(label).startswith("bright") for label in port_labels], dtype=bool
    )
    dark_mask = np.array(
        [str(label).startswith("dark") for label in port_labels], dtype=bool
    )
    if not bright_mask.any() or not dark_mask.any():
        return float("nan")
    bright_mean = float(np.mean(port_rates_hz[bright_mask]))
    if bright_mean <= 0.0:
        return float("nan")
    # n_bright_ports kept on the API for back-compat — used by callers as
    # a cross-check that the label partition matches the schema.
    _ = n_bright_ports
    return float(np.mean(port_rates_hz[dark_mask]) / bright_mean)


def _port_labels(
    n_bright_ports: int,
    n_dark_ports: int,
    n_spectro_ports: int = 0,
    *,
    n_pol: int = 1,
    pol_labels: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    if n_pol <= 1:
        return tuple(
            [f"bright_{i}" for i in range(n_bright_ports)]
            + [f"dark_{i}" for i in range(n_dark_ports)]
            + [f"spectro_{i}" for i in range(n_spectro_ports)]
        )
    labels = pol_labels or tuple(f"pol{i}" for i in range(n_pol))
    if len(labels) != n_pol:
        raise ValueError(
            f"_port_labels: expected {n_pol} polarization labels, got {len(labels)}."
        )

    output: list[str] = []
    for prefix, count in (
        ("bright", n_bright_ports),
        ("dark", n_dark_ports),
        ("spectro", n_spectro_ports),
    ):
        for port_index in range(count):
            output.extend(f"{prefix}_{port_index}_{label}" for label in labels)
    return tuple(output)


def _polarization_labels(state: SimulationState, n_pol: int) -> tuple[str, ...]:
    if n_pol <= 1:
        return ("pol0",)
    for payload in reversed(tuple(state.field.components.values())):
        if not isinstance(payload, Mapping):
            continue
        labels = payload.get("polarization_labels")
        if labels is None:
            continue
        resolved = tuple(str(label) for label in labels)
        if len(resolved) == n_pol:
            return resolved
    if n_pol == 2:
        return ("A", "B")
    return tuple(f"pol{i}" for i in range(n_pol))


__all__ = ["MKIDDetector"]
