"""SNSPD detector stage (SPEC §15.3 Pass 4a).

Reference implementation: KILO ``detect_snspd`` in
``kilo/detectors.py`` lines 289-309.
Single-element SNSPD per port; per-port signal rate is capped at
``snspd_max_count_rate_hz`` (1e8 Hz device default per
``kilo/config.py`` lines 451-459).
``snspd_dark_rate_hz`` is reported separately for the downstream noise
budget (10 Hz KILO and NAYRA default — they agree on all SNSPD fields).
``snspd_timing_resolution_s`` (50 ps default) is documentation only in this
pass; it is emitted into the results component for downstream
coincidence-window stages.

See ``tests/stages/notes/detector_snspd.md`` for the full reference survey
and pinned invariants.

For scalar fields, the output is one SNSPD channel per chip port. For
dual-arm PBS fields, the polarization axis is retained as physical SNSPD
channels (``bright_0_A``, ``bright_0_B``, ...), matching MKID detector
semantics.

Note on dark counts: KILO's ``detect_snspd`` does NOT add
``snspd_dark_rate`` itself — dark counts are tracked separately in the
noise budget. The nullsim stage keeps ``port_rates`` as stellar signal
rates and publishes total signal+dark rates in explicit diagnostic fields.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field as PydanticField, model_validator

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
class SNSPDDetector(BaseStage):
    """Apply SNSPD QE, saturation, and dark counts to produce per-port count rates."""

    type_name = "detector.snspd"
    family = "detector"
    consumes = frozenset({"field", "throughput", "scene", "geometry"})
    produces = frozenset({"photon_rates", "results", "throughput"})

    class Params(BaseModel):
        """Validated params matching KILO/NAYRA SNSPD detector names."""

        model_config = ConfigDict(
            extra="forbid",
            frozen=True,
            allow_inf_nan=False,
        )

        qe: float = PydanticField(gt=0.0, le=1.0)
        snspd_max_count_rate_hz: float = PydanticField(default=1.0e8, gt=0.0)
        snspd_dark_rate_hz: float = PydanticField(default=10.0, ge=0.0)
        snspd_timing_resolution_s: float = PydanticField(default=50.0e-12, gt=0.0)
        n_bright_ports: int = PydanticField(ge=0)
        n_dark_ports: int = PydanticField(ge=0)
        n_spectro_ports: int = PydanticField(default=0, ge=0)
        n_modes: int = PydanticField(ge=1)
        integration_time_s: float = PydanticField(gt=0.0)
        band: str = "J"
        derive_band_mag_from_scene_jmag: bool = False
        default_spectral_type: str = "F"

        @model_validator(mode="after")
        def _ports_sum_to_modes(self) -> SNSPDDetector.Params:
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

        signal_rate_hz_per_wavelength = _port_rate_hz_per_wavelength(
            field_amplitudes=field_amplitudes,
            throughput_fraction=throughput_fraction,
            stellar_rate_per_m2_per_bin=stellar_rate_per_m2_per_bin,
            collecting_area_m2=collecting_area_m2,
            qe=params.qe,
        )

        # Bug #1: null_depth_raw must come from PRE-saturation signal,
        # before the dark-rate background and the fixed cap. Same
        # reasoning as detector/mkid.py — the chip's true null is
        # masked by detector saturation if read from post-cap rates.
        port_labels = _port_labels(
            params.n_bright_ports,
            params.n_dark_ports,
            params.n_spectro_ports,
            n_pol=n_pol_arms,
            pol_labels=_polarization_labels(state, n_pol_arms),
        )
        uncapped_signal_per_port = signal_rate_hz_per_wavelength.sum(axis=1)
        null_depth_raw = _null_depth_raw(
            port_rates_hz=uncapped_signal_per_port,
            port_labels=port_labels,
            n_bright_ports=params.n_bright_ports,
        )

        capped_signal, saturated_per_port = _apply_fixed_cap(
            signal_rate_hz_per_wavelength,
            cap_hz=params.snspd_max_count_rate_hz,
        )

        dark_rate_hz_per_wavelength = _dark_rate_hz_per_wavelength(
            dark_rate_hz=params.snspd_dark_rate_hz,
            n_ports=signal_rate_hz_per_wavelength.shape[0],
            n_wavelength_bins=len(wavelength_grid_um),
        )
        total_rate_hz_per_wavelength = capped_signal + dark_rate_hz_per_wavelength
        signal_port_rates_hz = capped_signal.sum(axis=1)
        total_port_rates_hz = total_rate_hz_per_wavelength.sum(axis=1)
        photon_rates = PhotonRates(
            wavelength_grid_um=wavelength_grid_um,
            port_labels=port_labels,
            rates_hz=capped_signal,
        )

        component_name = unique_component_name(
            self.type_name,
            state.results.components,
            self.context.instance_id,
        )
        results = state.results.add_component(
            component_name,
            {
                "null_depth_raw": float(null_depth_raw),
                "port_rates": signal_port_rates_hz.astype(np.float64, copy=True),
                "port_labels": port_labels,
                "saturated_per_port": saturated_per_port,
                "detector_qe": float(params.qe),
                "snspd_max_count_rate_hz": float(params.snspd_max_count_rate_hz),
                "snspd_dark_rate_hz": float(params.snspd_dark_rate_hz),
                "dark_rate_hz_per_port": float(params.snspd_dark_rate_hz),
                "snspd_timing_resolution_s": float(params.snspd_timing_resolution_s),
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
                    signal_port_rates_hz * params.integration_time_s
                ).astype(np.float64, copy=True),
                "uncapped_port_rate_hz_per_wavelength": (
                    signal_rate_hz_per_wavelength.astype(np.float64, copy=True)
                ),
                # Per-port signal totals BEFORE the fixed cap, the exact
                # pre-cap counterpart of ``port_rates``. Same reasoning as
                # detector.mkid: saturation is a readout limit, so anything
                # normalizing against the stellar flux has to use the uncapped
                # rate rather than the clipped one.
                "uncapped_port_rates": uncapped_signal_per_port.astype(
                    np.float64, copy=True
                ),
                "detector_saturated": bool(np.any(saturated_per_port)),
                "total_port_rates_with_dark": total_port_rates_hz.astype(
                    np.float64, copy=True,
                ),
                "integrated_total_counts_per_port": (
                    total_port_rates_hz * params.integration_time_s
                ).astype(np.float64, copy=True),
                "collecting_area_m2": float(collecting_area_m2),
                "source_model": (
                    "KILO detect_snspd (kilo/detectors.py L289-309) with fixed "
                    "per-port cap and separate uniform dark-count diagnostics; "
                    "nullsim.physics.photometry for magnitude→photon-rate."
                ),
            },
        )
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
    n_bins = len(wavelength_grid_um)
    per_channel = np.full(n_bins, float(qe), dtype=np.float64)
    current = (
        np.asarray(state.throughput.transmission_fraction, dtype=np.float64)
        if state.throughput.transmission_fraction
        else np.ones(n_bins, dtype=np.float64)
    )
    if current.size != n_bins:
        raise ValueError(
            "detector.snspd: throughput.transmission_fraction length "
            f"{current.size} does not match wavelength grid length {n_bins}."
        )
    component_name = unique_component_name(
        "detector.snspd_qe",
        state.throughput.components,
        instance_id,
    )
    return Throughput(
        wavelength_grid_um=wavelength_grid_um,
        transmission_fraction=tuple(float(v) for v in current * per_channel),
        components={
            **state.throughput.components,
            component_name: {
                "wavelength_grid_um": wavelength_grid_um,
                "transmission_fraction": tuple(float(v) for v in per_channel),
                "qe": float(qe),
                "source_model": (
                    "SNSPD detector quantum efficiency folded into the "
                    "throughput chain, matching detector.mkid_qe semantics."
                ),
            },
        },
    )


def _bandwidth_per_bin_um(wavelength_grid_um: tuple[float, ...]) -> NDArray[np.float64]:
    wl = np.asarray(wavelength_grid_um, dtype=np.float64)
    if wl.size == 1:
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


def _stellar_magnitude_source(params: SNSPDDetector.Params) -> str:
    if params.derive_band_mag_from_scene_jmag:
        return (
            "scene.star.jmag converted to detector band with "
            "nullsim.physics.photometry.jmag_to_band_mag."
        )
    return "scene.star.jmag interpreted directly in detector band."


def _stellar_magnitude_for_band(
    state: SimulationState,
    params: SNSPDDetector.Params,
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
    """Per-mode-summed light-coupling area for absolute photon rates.

    Mirrors ``nullsim/stages/detector/mkid.py:_resolve_collecting_area_m2``
    so SNSPD-based sub-aperture pipelines (KILO Keck) get the packed
    sub-aperture footprint rather than the area-equivalent full-pupil
    share. The state contract (state.py:183) mandates
    ``effective_collecting_area_m2`` first; without this prefer-step
    SNSPD overcounts per-mode rates by ~2.24× on KILO Keck (12 d_sub
    sub-apertures packing ~44.7% of the obstructed primary).
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
    n_modes = power_per_mode_pol.shape[0]
    area_per_mode_m2 = collecting_area_m2 / float(n_modes)
    photon_rate = (
        stellar_rate_per_m2_per_bin * throughput_fraction * area_per_mode_m2
    )
    power_per_physical_port = np.transpose(power_per_mode_pol, (0, 2, 1)).reshape(
        n_modes * power_per_mode_pol.shape[2],
        power_per_mode_pol.shape[1],
    )
    return power_per_physical_port * photon_rate[np.newaxis, :] * qe


def _apply_fixed_cap(
    port_rate_hz_per_wavelength: NDArray[np.float64],
    cap_hz: float,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Cap each PORT-TOTAL rate at the SNSPD device-level limit.

    One SNSPD per port; the cap is the detector's saturation limit on
    the total photon arrival rate at that pixel, summed across all
    wavelengths.

    Pre-fix this stage capped each ``(port, bin)`` cell independently
    then summed across bins, so a port that saturated uniformly across
    N wavelength bins ended up at ``cap × N`` total (Code Review
    2026-05-16 Ultra-review High #4). Fix: aggregate over wavelengths
    to get the per-port total rate, compare to ``cap_hz``, and rescale
    every bin by the same factor so the spectral shape is preserved.
    """

    port_total_rate = port_rate_hz_per_wavelength.sum(axis=1)
    saturated_per_port = port_total_rate > cap_hz
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(
            saturated_per_port,
            cap_hz / port_total_rate,
            1.0,
        )
    scale = np.where(np.isfinite(scale), scale, 1.0)
    capped = port_rate_hz_per_wavelength * scale[:, np.newaxis]
    return capped, saturated_per_port


def _dark_rate_hz_per_wavelength(
    *,
    dark_rate_hz: float,
    n_ports: int,
    n_wavelength_bins: int,
) -> NDArray[np.float64]:
    if dark_rate_hz <= 0.0:
        return np.zeros((n_ports, n_wavelength_bins), dtype=np.float64)
    # Distribute the per-port dark rate uniformly across wavelength bins so
    # the per-bin rate × bins sums back to dark_rate_hz per port. SNSPD dark
    # counts are broadband / not energy-resolved.
    per_bin = dark_rate_hz / float(n_wavelength_bins)
    return np.full((n_ports, n_wavelength_bins), per_bin, dtype=np.float64)


def _null_depth_raw(
    *,
    port_rates_hz: NDArray[np.float64],
    port_labels: tuple[str, ...],
    n_bright_ports: int,
) -> float:
    """Raw null depth = mean(dark port rates) / mean(bright port rates).

    Masks on port_labels rather than slicing positionally — see
    mkid._null_depth_raw for the Code Review Issue 15 rationale.
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


__all__ = ["SNSPDDetector"]
