"""Typed SimulationState payload described by SPEC §6.2.

The pipeline passes a frozen :class:`SimulationState` between stateless stages.
The state is split into the eight named sub-budgets from the design document so
stage dependency declarations can consume only the pieces they need. Each
sub-budget tracks named contributions and exposes exact-byte content hashing for
the future cache layer.

Sub-budget digests are computed from a *stable canonical byte stream*, not from
pickle output. This guarantees that pickling a :class:`SubBudget` and reloading
it (as the disk cache does) produces a byte-identical digest, so downstream
stages keep hitting the cache on re-run per SPEC §6.4.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType
from typing import Any, ClassVar, Self
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from nullsim.pipeline._canonical import canonical_bytes as _canonical_bytes

STATE_BUDGET_NAMES: tuple[str, ...] = (
    "scene",
    "geometry",
    "field",
    "opd_budget",
    "throughput",
    "wavefront",
    "photon_rates",
    "results",
)


class _FrozenDict(dict):
    """Dict-compatible immutable mapping used for nested state payloads."""

    def _readonly(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("SimulationState payload mappings are immutable.")

    __setitem__ = _readonly
    __delitem__ = _readonly
    clear = _readonly
    pop = _readonly
    popitem = _readonly
    setdefault = _readonly
    update = _readonly
    __ior__ = _readonly

    def copy(self) -> dict[Any, Any]:
        return dict(self)


@dataclass(frozen=True)
class SpectralGrid:
    """Resolved wavelength grid shared by wavelength-indexed sub-budgets."""

    wavelength_grid_um: tuple[float, ...] = ()

    @classmethod
    def from_value(cls, grid: SpectralGrid | Mapping[str, Any] | Any | None) -> Self:
        """Build a spectral grid from a resolved config-like object.

        Args:
            grid: ``None``, an existing :class:`SpectralGrid`, a mapping with
                either ``wavelength_grid_um`` or center/bandwidth/bin fields,
                or an array-like sequence of wavelengths in microns.

        Returns:
            A normalized spectral grid with wavelengths stored as a tuple.

        Raises:
            ValueError: If a mapping declares a non-positive bin count.
            TypeError: If the value cannot be interpreted as a wavelength grid.
        """

        if grid is None:
            return cls()
        if isinstance(grid, SpectralGrid):
            return grid
        if isinstance(grid, Mapping):
            return cls(_wavelength_grid_from_mapping(grid))
        if hasattr(grid, "model_dump"):
            dumped = grid.model_dump()
            if isinstance(dumped, Mapping):
                return cls(_wavelength_grid_from_mapping(dumped))
        return cls(_coerce_wavelength_tuple(grid))


@dataclass(frozen=True, kw_only=True)
class SubBudget:
    """Base class for immutable named-contribution sub-budgets."""

    components: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for budget_field in fields(self):
            value = getattr(self, budget_field.name)
            if budget_field.name == "components":
                value = MappingProxyType(
                    {
                        component_name: _freeze_payload(component_value)
                        for component_name, component_value in dict(value).items()
                    }
                )
            else:
                value = _freeze_payload(value)
            object.__setattr__(self, budget_field.name, value)

    def __getstate__(self) -> dict[str, Any]:
        """Return pickle state with a plain dict for immutable components."""

        return {
            budget_field.name: (
                _thaw_payload(dict(self.components))
                if budget_field.name == "components"
                else _thaw_payload(getattr(self, budget_field.name))
            )
            for budget_field in fields(self)
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        """Restore pickle state while preserving component immutability."""

        for name, value in state.items():
            if name == "components":
                object.__setattr__(
                    self,
                    name,
                    MappingProxyType(
                        {
                            component_name: _freeze_payload(component_value)
                            for component_name, component_value in dict(value).items()
                        }
                    ),
                )
            else:
                object.__setattr__(self, name, _freeze_payload(value))

    def add_component(self, name: str, value: Any) -> Self:
        """Return a new sub-budget with a named contribution added.

        Args:
            name: Stable contribution name used in diagnostics and output
                breakdowns.
            value: Contribution payload. Plain scalars and numpy arrays are
                hashed exactly by content.

        Returns:
            A new sub-budget instance containing the additional contribution.

        Raises:
            ValueError: If ``name`` is empty or already present.
        """

        if not name:
            raise ValueError("Cannot add a sub-budget component with an empty name.")
        if name in self.components:
            raise ValueError(f"Component {name!r} already exists in {type(self).__name__}.")
        updated_components = dict(self.components)
        updated_components[name] = value
        return replace(self, components=updated_components)

    def content_digest(self) -> bytes:
        """Return a sha256 digest over typed fields and sorted components.

        The digest is computed from a stable, pickle-independent canonical byte
        stream: numpy arrays are normalized to C-contiguous bytes plus dtype
        and shape, mappings are key-sorted, and dataclass payloads are walked
        by field name. This is what makes the disk cache survive a pickle
        round-trip — see SPEC §6.4.
        """

        return hashlib.sha256(self._canonical_bytes()).digest()

    def _canonical_bytes(self) -> bytes:
        digest = hashlib.sha256()
        digest.update(_canonical_bytes(type(self).__name__))
        for budget_field in fields(self):
            if budget_field.name == "components":
                continue
            digest.update(_canonical_bytes(budget_field.name))
            digest.update(_canonical_bytes(getattr(self, budget_field.name)))
        for component_name in sorted(self.components):
            digest.update(_canonical_bytes(component_name))
            digest.update(_canonical_bytes(self.components[component_name]))
        return digest.digest()


def _freeze_payload(value: Any) -> Any:
    """Return an immutable defensive copy for state payload values."""

    if isinstance(value, np.ndarray):
        frozen = np.ascontiguousarray(value).copy()
        frozen.setflags(write=False)
        return frozen
    if isinstance(value, Mapping):
        return _FrozenDict(
            {
                key: _freeze_payload(inner_value)
                for key, inner_value in value.items()
            }
        )
    if isinstance(value, tuple):
        return tuple(_freeze_payload(item) for item in value)
    if isinstance(value, list):
        return tuple(_freeze_payload(item) for item in value)
    return value


def _thaw_payload(value: Any) -> Any:
    """Return a pickle-friendly copy of frozen state payload values."""

    if isinstance(value, Mapping):
        return {key: _thaw_payload(inner_value) for key, inner_value in value.items()}
    if isinstance(value, tuple):
        return tuple(_thaw_payload(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


@dataclass(frozen=True)
class Scene(SubBudget):
    """Astrophysical inputs: stellar, planetary, and background source values.

    ``altitude_deg`` and ``azimuth_deg`` are typed fields rather than entries
    in ``components`` so geometry-consuming stages (e.g. KILO-style geometric
    delay lines) can read them directly without string-keying into a free-form
    diagnostics dict. See SPEC §6.2 — typed sub-budget fields are the load-
    bearing channel for data passed between stages.
    """

    star_jmag: float | None = None
    star_angular_diameter_mas: float | None = None
    star_distance_pc: float | None = None
    star_spectral_type: str | None = None
    planet_contrast: float | None = None
    planet_separation_mas: float | None = None
    planet_radial_velocity_kms: float | None = None
    planet_position_angle_deg: float | None = None
    exozodi_level_zodi: float | None = None
    altitude_deg: float | None = None
    azimuth_deg: float | None = None


@dataclass(frozen=True)
class Geometry(SubBudget):
    """Aperture, baseline, pupil, and observation-angle geometry.

    ``area_m2`` is the representative per-aperture collecting area (a single
    primary's light-bucket, after the linear central-obscuration ratio is
    applied). For heterogeneous arrays, ``aperture_areas_m2`` and
    ``aperture_diameters_m`` carry the per-aperture values and ``area_m2`` is
    their mean for legacy scalar consumers.
    ``collecting_area_m2`` is the total over all apertures
    (``n_apertures * area_m2``) — the light that hits the telescope train.
    ``effective_collecting_area_m2`` is the light actually coupled into
    the chip modes, i.e. the sum of physical sub-aperture areas. For
    pipelines that do not decompose the pupil (HWO single-mode), the
    effective area equals the collecting area; for sub-aperture pipelines
    (KILO Keck) the effective area is smaller because the packed sub-
    apertures leave gaps inside the primary (KILO Keck packs ~44.7% of
    the obstructed primary into 12 d_sub≈1.93 m circles). Detector and
    sensitivity stages MUST scale per-mode photon rates with
    ``effective_collecting_area_m2`` (falling back to
    ``collecting_area_m2`` when the sub-aperture refinement is absent)
    so multi-aperture arrays sum correctly. KILO uses the same convention
    (``effective_area`` in ``kilo/studies/natural_design.py``).
    """

    diameter_m: float | None = None
    area_m2: float | None = None
    collecting_area_m2: float | None = None
    effective_collecting_area_m2: float | None = None
    n_apertures: int | None = None
    aperture_diameters_m: NDArray[Any] | None = None
    aperture_areas_m2: NDArray[Any] | None = None
    aperture_positions_enu_m: NDArray[Any] | None = None
    aperture_positions_pupil_m: NDArray[Any] | None = None
    baselines_m: NDArray[Any] | None = None
    uv_coverage_m: NDArray[Any] | None = None
    parallactic_angle_deg: float | None = None


@dataclass(frozen=True)
class Field(SubBudget):
    """Complex amplitudes with shape ``[n_modes, n_wavelength_bins, n_pol]``."""

    amplitudes: NDArray[Any] | None = None
    wavelength_grid_um: tuple[float, ...] = ()
    mode_labels: tuple[str, ...] = ()


@dataclass(frozen=True)

class OPDComponent:
    """Named optical path difference contribution.

    `superseded=True` marks a contribution that was later corrected by a
    downstream stage (e.g. atmosphere → AO residual). Superseded components
    stay in `OPDBudget.components` for plot/audit history but do NOT count
    toward `OPDBudget.total_rms_opd_nm`. See SPEC §6.2.
    """

    rms_opd_nm: float
    reduction_stage: str | None = None
    superseded: bool = False


@dataclass(frozen=True)
class OPDBudget(SubBudget):
    """Named RMS optical path difference contributions."""

    total_rms_opd_nm: float | None = None


@dataclass(frozen=True)
class Throughput(SubBudget):
    """Multiplicative transmission spectra and component-level losses."""

    wavelength_grid_um: tuple[float, ...] = ()
    transmission_fraction: tuple[float, ...] = ()


@dataclass(frozen=True)
class Wavefront(SubBudget):
    """Strehl and amplitude-jitter quantities on the wavelength grid."""

    wavelength_grid_um: tuple[float, ...] = ()
    strehl_ratio: tuple[float, ...] = ()
    amplitude_jitter_fraction: tuple[float, ...] = ()
    r0_at_500nm_m: float | None = None


@dataclass(frozen=True)
class PhotonRates(SubBudget):
    """Per-port, per-wavelength photon rates with named contributors."""

    wavelength_grid_um: tuple[float, ...] = ()
    port_labels: tuple[str, ...] = ()
    rates_hz: NDArray[Any] | None = None


@dataclass(frozen=True)
class Results(SubBudget):
    """Final scalar and curve products produced by post-processing stages."""

    null_depth: float | None = None
    contrast_curve: NDArray[Any] | None = None
    snr: float | None = None
    calibration_time_s: float | None = None


@dataclass(frozen=True)
class SimulationState:
    """Typed payload flowing through the pipeline's ordered stage sequence."""

    scene: Scene
    geometry: Geometry
    field: Field
    opd_budget: OPDBudget
    throughput: Throughput
    wavefront: Wavefront
    photon_rates: PhotonRates
    results: Results

    budget_names: ClassVar[tuple[str, ...]] = STATE_BUDGET_NAMES

    @classmethod
    def empty(cls, grid: SpectralGrid | Mapping[str, Any] | Any | None = None) -> Self:
        """Return a fully initialized state with empty contributions.

        Args:
            grid: Optional spectral grid information. When present, wavelength
                values are copied into wavelength-indexed sub-budgets.

        Returns:
            A :class:`SimulationState` with all eight SPEC §6.2 sub-budgets.
        """

        spectral_grid = SpectralGrid.from_value(grid)
        wavelength_grid_um = spectral_grid.wavelength_grid_um
        return cls(
            scene=Scene(),
            geometry=Geometry(),
            field=Field(wavelength_grid_um=wavelength_grid_um),
            opd_budget=OPDBudget(),
            throughput=Throughput(wavelength_grid_um=wavelength_grid_um),
            wavefront=Wavefront(wavelength_grid_um=wavelength_grid_um),
            photon_rates=PhotonRates(wavelength_grid_um=wavelength_grid_um),
            results=Results(),
        )

    def content_digest(self) -> bytes:
        """Return a sha256 digest over all state sub-budgets."""

        return self.content_digest_for(self.budget_names)

    def content_digest_for(self, keys: set[str] | frozenset[str] | tuple[str, ...] | list[str]) -> bytes:
        """Return a sha256 digest over selected sub-budgets.

        Args:
            keys: Sub-budget names or dotted state keys such as ``"scene"`` or
                ``"throughput.transmission_fraction"``. Dotted keys digest the
                owning sub-budget.

        Returns:
            A deterministic digest over the selected sub-budgets.

        Raises:
            ValueError: If any key does not name one of the eight sub-budgets.
        """

        digest = hashlib.sha256()
        for budget_name in _normalized_budget_names(keys):
            budget_digest = getattr(self, budget_name).content_digest()
            digest.update(budget_name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(budget_digest)
            digest.update(b"\0")
        return digest.digest()


def _normalized_budget_names(keys: set[str] | frozenset[str] | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    names = {key.split(".", maxsplit=1)[0] for key in keys}
    unknown_names = sorted(names.difference(STATE_BUDGET_NAMES))
    if unknown_names:
        expected_names = ", ".join(STATE_BUDGET_NAMES)
        unknown = ", ".join(unknown_names)
        raise ValueError(f"Unknown state budget key(s): {unknown}. Expected one of: {expected_names}.")
    return tuple(sorted(names))


def _wavelength_grid_from_mapping(grid: Mapping[str, Any]) -> tuple[float, ...]:
    if "wavelength_grid_um" in grid:
        return _coerce_wavelength_tuple(grid["wavelength_grid_um"])

    center_um = grid.get("wavelength_center_um")
    bandwidth_um = grid.get("wavelength_bandwidth_um")
    n_wavelength_bins = grid.get("n_wavelength_bins")
    if center_um is None or bandwidth_um is None or n_wavelength_bins is None:
        return ()

    n_bins = int(n_wavelength_bins)
    if n_bins <= 0:
        raise ValueError("n_wavelength_bins must be positive when building a spectral grid.")
    if n_bins == 1:
        return (float(center_um),)

    # Phase 6: bin-CENTER convention so the total integrated bandwidth
    # matches the configured ``wavelength_bandwidth_um`` exactly. The
    # pre-Phase-6 ``np.linspace(start, stop, n_bins)`` placed samples
    # AT the band edges, then mkid's half-bin edge extrapolation tacked
    # on an extra step at each end -- 20 bins over 0.15 um integrated
    # to ~0.158 um instead of 0.15.
    start_um = float(center_um) - float(bandwidth_um) / 2.0
    stop_um = float(center_um) + float(bandwidth_um) / 2.0
    edges = np.linspace(start_um, stop_um, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return tuple(float(value_um) for value_um in centers)


def _coerce_wavelength_tuple(values: Any) -> tuple[float, ...]:
    try:
        return tuple(float(value_um) for value_um in values)
    except TypeError as exc:
        raise TypeError("Wavelength grid must be an iterable of micron values.") from exc
