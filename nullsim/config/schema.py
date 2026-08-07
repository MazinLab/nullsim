"""Pydantic schemas for the SPEC §7 config surface and SPEC §6.4 hashes.

The models in this module define the user-facing TOML structure for the
vertical slice. They deliberately validate top-level structure early while
leaving per-stage parameters open for the stage registry to validate later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONFIG_METADATA_FIELDS = frozenset(
    {
        "raw_config",
        "source_path",
        "resolved_toml_text",
        "config_hash",
        "run_hash",
        "sweep_coord_tuple",
    }
)


class StrictConfigModel(BaseModel):
    """Base class for frozen config models with unknown keys rejected."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class RunConfig(StrictConfigModel):
    """Top-level run metadata and reproducibility controls."""

    name: str = "nullsim_run"
    description: str | None = None
    seed: int = Field(default=0, ge=0)
    extensions: tuple[str, ...] = Field(default_factory=tuple)
    n_workers: int | None = Field(default=None, ge=1)
    threads_per_worker: int | None = Field(default=None, ge=1)


class GridConfig(StrictConfigModel):
    """Spectral grid declaration in microns."""

    wavelength_center_um: float = Field(gt=0.0)
    wavelength_bandwidth_um: float = Field(ge=0.0)
    n_wavelength_bins: int = Field(gt=0)


class StarConfig(StrictConfigModel):
    """Stellar source parameters."""

    jmag: float
    angular_diameter_mas: float = Field(ge=0.0)
    distance_pc: float = Field(gt=0.0)
    target: str | None = None
    spectral_type: str | None = None


class PlanetConfig(StrictConfigModel):
    """Single-planet source parameters for the vertical slice."""

    contrast: float = Field(ge=0.0)
    separation_mas: float = Field(ge=0.0)
    position_angle_deg: float
    # Star-planet radial velocity in km/s, positive = receding. Unset leaves
    # the scene.point_source Params default in place.
    radial_velocity_kms: float | None = None


class ExozodiConfig(StrictConfigModel):
    """Exozodiacal background level."""

    level_zodi: float = Field(default=0.0, ge=0.0)


class SceneConfig(StrictConfigModel):
    """Astrophysical inputs populated into the scene state budget."""

    star: StarConfig
    planet: PlanetConfig
    exozodi: ExozodiConfig = Field(default_factory=ExozodiConfig)


class SiteConfig(BaseModel):
    """Site preset selection plus free-form site overrides."""

    model_config = ConfigDict(extra="allow", frozen=True)

    preset: str | None = None


class TelescopeConfig(BaseModel):
    """Telescope preset selection plus free-form telescope overrides."""

    model_config = ConfigDict(extra="allow", frozen=True)

    preset: str | None = None


class ObservationConfig(StrictConfigModel):
    """Observation geometry and exposure-time settings."""

    hour_angle_h: float = 0.0
    integration_time_s: float = Field(gt=0.0)


class InstrumentConfig(StrictConfigModel):
    """Shared chip port topology materialized into port-bearing stages."""

    n_modes: int | None = Field(default=None, ge=2)
    n_input_modes: int | None = Field(default=None, ge=1)
    n_bright_ports: int | None = Field(default=None, ge=0)
    n_dark_ports: int | None = Field(default=None, ge=0)
    n_spectro_ports: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _ports_sum_to_modes(self) -> InstrumentConfig:
        ports = (self.n_bright_ports, self.n_dark_ports, self.n_spectro_ports)
        if self.n_modes is not None and all(port is not None for port in ports):
            port_sum = sum(port for port in ports if port is not None)
            if port_sum != self.n_modes:
                raise ValueError(
                    f"[instrument] ports {ports} sum to {port_sum} "
                    f"!= n_modes {self.n_modes}"
                )
        return self


class StageSpec(BaseModel):
    """One ordered pipeline stage instance.

    Unknown keys are allowed here because the selected stage type owns its
    parameter schema. The registry validates those parameters when it builds the
    stage instance.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    id: str = Field(min_length=1)
    type: str = Field(min_length=1)


class PipelineConfig(StrictConfigModel):
    """Explicitly ordered pipeline stage list."""

    stages: tuple[StageSpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _stage_ids_must_be_unique(self) -> PipelineConfig:
        """Reject duplicate stage IDs.

        Pre-fix, two stages with the same ``id`` loaded cleanly. They then
        shared RNG identity (``derive_rng`` keys on instance_id),
        overwrote each other's diagnostics in ``state.results.components``,
        and silently received the first match when ``sweep`` blocks
        targeted a stage by ID. Ultra-review High #3.
        """
        ids = [stage.id for stage in self.stages]
        duplicates = {sid for sid in ids if ids.count(sid) > 1}
        if duplicates:
            raise ValueError(
                "pipeline.stages requires unique stage IDs; duplicates: "
                f"{sorted(duplicates)}"
            )
        return self


class RangeSweepValues(StrictConfigModel):
    """Half-open numeric range sweep values, matching Python range semantics."""

    start: float
    stop: float
    step: float

    @model_validator(mode="after")
    def validate_step_direction(self) -> RangeSweepValues:
        """Reject zero or non-progressing range steps."""

        if self.step == 0.0:
            raise ValueError("range.step must be non-zero.")
        if self.start < self.stop and self.step < 0.0:
            raise ValueError("range.step must be positive when start < stop.")
        if self.start > self.stop and self.step > 0.0:
            raise ValueError("range.step must be negative when start > stop.")
        return self


class LinspaceSweepValues(StrictConfigModel):
    """Linear sweep value generator."""

    start: float
    stop: float
    num: int = Field(gt=0)


class LogspaceSweepValues(StrictConfigModel):
    """Logarithmic sweep value generator using powers of ten."""

    start: float
    stop: float
    num: int = Field(gt=0)


class SweepSpec(StrictConfigModel):
    """One parameter-sweep axis."""

    param: str = Field(min_length=1)
    values: tuple[Any, ...] | None = None
    range: RangeSweepValues | None = None
    linspace: LinspaceSweepValues | None = None
    logspace: LogspaceSweepValues | None = None
    mode: Literal["grid", "zip"] = "grid"

    @model_validator(mode="after")
    def validate_single_value_source(self) -> SweepSpec:
        """Require exactly one sweep value source."""

        sources = [
            self.values is not None,
            self.range is not None,
            self.linspace is not None,
            self.logspace is not None,
        ]
        if sum(sources) != 1:
            raise ValueError(
                "SweepSpec must define exactly one of values, range, linspace, or logspace."
            )
        if self.values is not None and len(self.values) == 0:
            raise ValueError("SweepSpec.values must not be empty.")
        return self


class OutputFormatsConfig(StrictConfigModel):
    """Output file formats for plots and tables."""

    plots: tuple[str, ...] = ("pdf", "png")
    tables: tuple[str, ...] = ("csv", "json")


class ImagingPlanetConfig(StrictConfigModel):
    """Optional planet override for the imaging reconstruction output."""

    separation_mas: float = Field(ge=0.0)
    position_angle_deg: float
    flux_model: Literal["fixed_contrast", "reflected_light"] = "fixed_contrast"
    contrast: float | None = Field(default=None, ge=0.0)
    radius_earth: float | None = Field(default=None, gt=0.0)
    semi_major_axis_au: float | None = Field(default=None, gt=0.0)
    geometric_albedo: float = Field(default=0.3, ge=0.0)
    phase_angle_deg: float = Field(default=90.0, ge=0.0, le=180.0)
    label: str | None = None

    @model_validator(mode="after")
    def validate_flux_model_fields(self) -> ImagingPlanetConfig:
        if self.flux_model == "fixed_contrast":
            if self.contrast is None:
                raise ValueError(
                    "ImagingPlanetConfig with flux_model='fixed_contrast' "
                    "requires contrast."
                )
            return self
        if self.contrast is not None:
            raise ValueError(
                "ImagingPlanetConfig with flux_model='reflected_light' derives "
                "contrast; omit the contrast field."
            )
        if self.radius_earth is None:
            raise ValueError(
                "ImagingPlanetConfig with flux_model='reflected_light' "
                "requires radius_earth."
            )
        if self.semi_major_axis_au is None:
            raise ValueError(
                "ImagingPlanetConfig with flux_model='reflected_light' requires "
                "semi_major_axis_au; projected sky separation is not the orbital "
                "semi-major axis."
            )
        return self


class ImagingReconstructionOutputConfig(StrictConfigModel):
    """Controls for the optional aperture-synthesis imaging output."""

    n_pix: int = Field(default=64, ge=8, le=512)
    fov_mas: float | None = Field(default=None, gt=0.0)
    rotation_mode: Literal["earth_rotation", "spacecraft_roll", "fixed"] = (
        "earth_rotation"
    )
    n_time: int = Field(default=24, ge=1, le=512)
    hour_angle_span_h: float | None = Field(default=None, ge=0.0)
    roll_span_deg: float = Field(default=0.0, ge=0.0)
    n_roll_steps: int = Field(default=1, ge=1, le=512)
    declination_deg: float = Field(default=20.0, ge=-90.0, le=90.0)
    latitude_deg: float | None = Field(default=None, ge=-90.0, le=90.0)
    max_wavelengths: int | None = Field(default=7, ge=1)
    max_records: int = Field(default=1, ge=1)
    seed: int | None = Field(default=None, ge=0)
    poisson: bool = True
    save_arrays: bool = True
    compute_nnls: bool = False
    nnls_wavelength_mode: Literal["band_averaged", "resolved"] = "resolved"
    nnls_candidate_pixels: int = Field(default=1024, ge=1)
    nnls_regularization_fraction: float = Field(default=0.0, ge=0.0)
    chop_mode: Literal["none", "chip_phase"] = "none"
    include_exozodi_glow: bool = True
    exozodi_inner_radius_au: float = Field(default=0.2, gt=0.0)
    exozodi_outer_radius_au: float = Field(default=5.0, gt=0.0)
    exozodi_radial_power: float = Field(default=1.0, ge=0.0)
    exozodi_position_angle_deg: float = 0.0
    planets: tuple[ImagingPlanetConfig, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validate_exozodi_radii(self) -> ImagingReconstructionOutputConfig:
        if self.exozodi_outer_radius_au <= self.exozodi_inner_radius_au:
            raise ValueError("exozodi_outer_radius_au must exceed inner radius.")
        return self


class NullTransmissionMapOutputConfig(StrictConfigModel):
    """Controls for the optional null-transmission map output."""

    fov_mas: float = Field(default=300.0, gt=0.0)
    n_pix: int = Field(default=201, ge=8, le=512)
    include_angular_coupling: bool = True
    projection_mode: Literal["auto", "sky", "pupil"] = "auto"
    declination_deg: float | None = Field(default=None, ge=-90.0, le=90.0)
    latitude_deg: float | None = Field(default=None, ge=-90.0, le=90.0)
    hour_angle_h: float | None = None
    auto_scale_to_owa: bool = True
    display_padding_fraction: float = Field(default=0.10, ge=0.0, le=1.0)
    log_scale: bool = True
    log_floor: float = Field(default=1.0e-12, gt=0.0)
    n_contour_levels: int = Field(default=24, ge=2)
    overlay_planet: bool = True
    # max_records selects the FIRST N records in sweep declaration order
    # (cell 0 = first sweep value, e.g. J=4 in the kilo_keck jmag sweep).
    # Documented per review 2026-06-10 Issue 23d.
    max_records: int = Field(default=1, ge=1)
    save_arrays: bool = True


class DetectionSpaceOutputConfig(StrictConfigModel):
    """Controls for the detection-space headline figure."""

    # Sweep cells (by scene.star.jmag) drawn on the published figure. The
    # paperv4 default is the J=4 / J=8 pair; other cells are withheld with
    # a logged notice (review 2026-06-10 Issue 23a). Empty tuple = plot
    # every cell.
    headline_jmags: tuple[float, ...] = (4.0, 8.0)

    # Overlay the Guyon et al. (2006) theoretically optimal coronagraph
    # photon-noise bound at the listed null orders (2 and/or 4) for the same
    # star, aperture, backgrounds, and integration time. Empty tuple
    # (default) disables the overlay. Single-aperture configs only: the
    # bound is defined for one clear circular primary.
    ideal_coronagraph_orders: tuple[int, ...] = ()


class OutputConfig(StrictConfigModel):
    """Output directory and selected plot/table products."""

    dir: str = "results/{run.name}/{timestamp}"
    plots: tuple[str, ...] = Field(default_factory=tuple)
    tables: tuple[str, ...] = Field(default_factory=tuple)
    paper: bool = False
    formats: OutputFormatsConfig = Field(default_factory=OutputFormatsConfig)
    imaging_reconstruction: ImagingReconstructionOutputConfig | None = None
    null_transmission_map: NullTransmissionMapOutputConfig | None = None
    detection_space: DetectionSpaceOutputConfig | None = None
    sweep_table: Literal["csv", "json", "parquet"] = "parquet"


class CacheConfig(StrictConfigModel):
    """Stage cache controls.

    The cache key folds in package/dependency versions but NOT
    machine-specific context (resolved torch device for
    ``torch_device="auto"``, BLAS/torch thread caps). The cache directory
    is therefore single-machine: sharing or syncing ``.nullsim_cache``
    across hosts can serve entries computed under a different device or
    reduction order (review 2026-06-10 Issue 25).
    """

    enabled: bool = True
    dir: str = ".nullsim_cache"
    version: int = Field(default=1, ge=0)


class CatalogConfig(BaseModel):
    """Container for versioned external reference catalogs (SPEC §8.1, §15.4).

    Only ``exoplanets`` is defined today; new catalogs (target lists, instrument
    throughput tables) are accepted via ``extra="allow"`` and validated in their
    own loader modules. The exoplanet sub-block is kept as a raw mapping here —
    the actual ``ExoplanetCatalogConfig`` validation needs the TOML directory
    to resolve relative ``snapshot_path`` values, so it happens in
    ``nullsim.config.catalog.parse_catalog_block`` once the loader knows where
    the file came from.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    exoplanets: dict[str, Any] | None = None


class TopLevelConfig(StrictConfigModel):
    """Validated top-level nullsim configuration."""

    run: RunConfig
    grid: GridConfig
    site: SiteConfig = Field(default_factory=SiteConfig)
    telescope: TelescopeConfig = Field(default_factory=TelescopeConfig)
    scene: SceneConfig
    observation: ObservationConfig
    instrument: InstrumentConfig = Field(default_factory=InstrumentConfig)
    pipeline: PipelineConfig
    sweep: tuple[SweepSpec, ...] = Field(default_factory=tuple)
    output: OutputConfig = Field(default_factory=OutputConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    catalog: CatalogConfig | None = None
    # Opaque passthrough for the nullsimtd time-domain companion package:
    # nullsim never interprets [timedomain.*]; nullsimtd validates it with
    # its own schema. Present blocks fold into config_hash via the canonical
    # dump; absent blocks are dropped by exclude_none, so pre-existing
    # configs keep their hashes (and cache entries).
    timedomain: dict[str, Any] | None = None

    @property
    def sweeps(self) -> tuple[SweepSpec, ...]:
        """Alias for callers that prefer the plural spelling."""

        return self.sweep


class ResolvedConfig(TopLevelConfig):
    """Preset-merged, validated config plus reproducibility metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    raw_config: dict[str, Any] = Field(default_factory=dict, exclude=True)
    source_path: Path | None = Field(default=None, exclude=True)
    resolved_toml_text: str = Field(default="", exclude=True)
    config_hash: str = Field(default="", exclude=True)
    run_hash: str = Field(default="", exclude=True)
    timestamp: str = Field(default="", exclude=True)
    sweep_coord_tuple: tuple[tuple[str, Any], ...] = Field(
        default_factory=tuple,
        exclude=True,
    )

    def config_mapping(self) -> dict[str, Any]:
        """Return only the TOML configuration fields, excluding metadata."""

        return self.model_dump(
            mode="python",
            exclude=CONFIG_METADATA_FIELDS,
            exclude_none=True,
        )

    def write_toml(self, path: Path | str) -> None:
        """Write the canonical resolved TOML text used for ``config_hash``."""

        output_path = Path(path)
        output_path.write_text(self.resolved_toml_text, encoding="utf-8")
