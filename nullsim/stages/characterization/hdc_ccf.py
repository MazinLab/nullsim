"""High dispersion coronagraphy (HDC) cross-correlation stage.

Sits downstream of ``characterization.spectro``. That stage publishes the
retuned chip's per-wavelength planet throughput, static stellar leakage, and
detected stellar photon rates on the (coarse) detector wavelength grid, plus
the spectrograph description (AWG resolving power and bin count, or the MKID
native resolving power). This stage takes those broadband budgets, dresses
them with real spectral shapes -- a planet albedo, a stellar photosphere, and
an optional telluric transmission -- channelizes them at the spectrograph
resolving power, and evaluates the matched-filter cross-correlation budget of
`nullsim.physics.hdc`.

Nothing about the nuller changes here: the null depth, the throughput, and the
photon rates are exactly the ones ``characterization.spectro`` published. HDC
is an *after-the-fact* analysis gain -- it digs planets out of the noise at
fixed nuller performance by exploiting the spectral asymmetry between a smooth
leaked-starlight continuum and the planet's line forest, shifted away from the
stellar rest frame by the planet's radial velocity. Framing and the
sqrt(N_lines) scaling follow Wang, Mawet, Ruane, Hu & Benneke 2017, AJ 153,
183.

Model choices worth knowing before reading the payload:

- The planet spectrum is reflected starlight: ``contrast x stellar_flux x
  albedo x throughput x telluric``, Doppler shifted as a whole by
  ``scene.planet_radial_velocity_kms``. It therefore carries the stellar line
  forest as well as the planet's own albedo features.
- The leaked stellar spectrum is ``leak_fraction x stellar_flux x telluric``
  at zero radial velocity. Its *mean* never contributes signal (the bright
  port measures it); what it contributes is shot noise and, through the
  null-instability amplitude ``eps_systematic``, the systematic term. The
  template projection of the filtered leak (``systematic_ccf_leakage``) is the
  CCF-space replacement for the broadband leakage rate, and its ratio to the
  raw leak (``ccf_systematic_rejection``) is the HDC systematic gain.
- The radial-velocity scan therefore correlates trial templates against the
  *leak-subtracted* dark-port residual, with the leak's shot noise retained in
  the per-channel variance. Without that subtraction the statistic would peak
  in the stellar rest frame, which is a statement about the star, not the
  planet.

Deterministic: the stage draws no random numbers.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField

from nullsim.physics.hdc import (
    ChannelSpectra,
    build_channel_edges,
    ccf_vs_rv,
    channelize,
    matched_filter_ccf,
    snr_vs_resolving_power,
)
from nullsim.physics.spectra import (
    TemplateSpectrum,
    constant_r_grid,
    convolve_to_resolving_power,
    doppler_shift,
    load_template_csv,
    packaged_spectrum_path,
    resample_flux_conserving,
)
from nullsim.pipeline import (
    BaseStage,
    ExternalDependency,
    SimulationState,
    register_stage,
)
from nullsim.stages._utils import unique_component_name

__all__ = ["HdcCcfStage"]

_SPECTRO_PREFIX = "characterization.spectro"

# Keys the upstream spectro payload must carry for this stage to do anything.
_REQUIRED_SPECTRO_KEYS = frozenset(
    {
        "wavelengths_m",
        "detected_stellar_rate_by_wavelength_hz",
        "spectro_planet_throughput_per_wavelength",
        "spectro_static_leak_fraction_per_wavelength",
        "planet_signal_rate_hz",
        "leakage_rate_hz",
        "eps_systematic",
        "servo_bw_hz",
    }
)

_DEFAULT_PLANET_RV_KMS = 30.0
_R_SCAN_MIN = 100.0
_R_SCAN_MAX = 30_000.0
# Templates are only ever point-sampled onto the analysis grid after being
# smoothed to at most this fraction of the analysis grid's resolving power,
# and the R scan never asks for more than this fraction of the surviving
# content resolution -- a matched filter cannot be evaluated at a resolving
# power the analysis grid does not resolve.
_R_SCAN_CONTENT_FRACTION = 0.5
_STELLAR_TEMPLATE_BY_TYPE = {"G": "G", "K": "K", "M": "M"}
_RESOLVING_POWER_RE = re.compile(r"resolving_power\s*:\s*([0-9.eE+]+)")


@register_stage
class HdcCcfStage(BaseStage):
    """Matched-filter cross-correlation budget for the spectro port."""

    type_name = "characterization.hdc_ccf"
    family = "characterization"
    consumes = frozenset({"results", "scene"})
    produces = frozenset({"results"})

    class Params(BaseModel):
        """Validated parameters for the HDC cross-correlation stage."""

        model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

        # Packaged name ("earth", "venus", "G"/"K"/"M", "maunakea",
        # "armazones") or a path to a two-column CSV.
        planet_template: str = "earth"
        # None resolves from scene.star_spectral_type.
        star_template: str | None = None
        # None (or "none") means no atmosphere: the right setting for space.
        telluric_template: str | None = None
        # None takes the AWG/MKID resolving power from the spectro payload.
        resolving_power: float | None = PydanticField(default=None, gt=0.0)
        telluric_mask_threshold: float = PydanticField(default=0.8, ge=0.0, le=1.0)
        # Analysis grid resolving power = grid_oversampling * instrument R.
        # Must exceed 1 so the instrument line-spread function is resolved.
        grid_oversampling: float = PydanticField(default=3.0, gt=1.0)
        continuum_filter_width_res_elements: int = PydanticField(default=30, ge=0)
        integration_time_s: float = PydanticField(default=3600.0, gt=0.0)
        target_snr: float = PydanticField(default=5.0, gt=0.0)
        snr_vs_r_points: int = PydanticField(default=12, ge=0)
        rv_scan_halfwidth_kms: float = PydanticField(default=150.0, gt=0.0)
        rv_scan_points: int = PydanticField(default=61, ge=0)

    def apply(self, state: SimulationState) -> SimulationState:
        params = self.params_model
        spectro_name, spectro = _find_spectro_component(state)

        band = _CoarseBand.from_spectro(spectro)
        resolving_power, resolving_power_source = _resolve_resolving_power(
            spectro, params.resolving_power
        )
        spectro_integration_time_s = _spectro_integration_time_s(spectro)
        sky_rate_hz, dark_rate_hz = _background_rates_hz(
            spectro, spectro_integration_time_s
        )
        eps_systematic = float(spectro["eps_systematic"])
        servo_bw_hz = float(spectro["servo_bw_hz"])
        leakage_rate_hz = float(spectro["leakage_rate_hz"])
        planet_signal_rate_hz = float(spectro["planet_signal_rate_hz"])
        with_fringe_tracking, systematic_branch_source = _infer_fringe_tracking(
            spectro,
            eps_systematic=eps_systematic,
            leakage_rate_hz=leakage_rate_hz,
            integration_time_s=spectro_integration_time_s,
        )

        planet_rv_kms = _planet_radial_velocity_kms(state)
        r_grid = float(params.grid_oversampling) * resolving_power

        # The coarsest channel grid the run will build sets how far past the
        # band the analysis grid has to reach, because a constant-R channel
        # grid overshoots the band edge by up to one resolution element.
        r_min_used = (
            min(resolving_power, _R_SCAN_MIN)
            if params.snr_vs_r_points > 1
            else resolving_power
        )
        lambda_max_grid_um = band.lambda_max_um * math.exp(1.5 / r_min_used)
        grid_um = constant_r_grid(band.lambda_min_um, lambda_max_grid_um, r_grid)

        max_shift_kms = abs(planet_rv_kms) + float(params.rv_scan_halfwidth_kms)
        templates = _Templates.load(
            params=params,
            state=state,
            config_dir=_config_dir_from_context(self.context),
            lambda_min_um=band.lambda_min_um,
            lambda_max_um=lambda_max_grid_um,
            max_shift_kms=max_shift_kms,
        )
        content_resolving_power = templates.content_resolving_power(r_grid)
        if content_resolving_power <= resolving_power:
            raise ValueError(
                "characterization.hdc_ccf: the spectral templates only resolve "
                f"R={content_resolving_power:.4g}, which does not exceed the "
                f"instrument R={resolving_power:.4g}. Raise grid_oversampling "
                "or use finer template snapshots."
            )
        # Only now is the cap known: the scan cannot ask for a resolving power
        # the templates and the analysis grid do not jointly resolve.
        r_scan = _r_scan_values(params.snr_vs_r_points, content_resolving_power)

        densities = _RateDensities.build(
            band=band,
            grid_um=grid_um,
            templates=templates,
            r_grid=r_grid,
            planet_contrast=_planet_contrast(state),
            planet_signal_rate_hz=planet_signal_rate_hz,
            leakage_rate_hz=leakage_rate_hz,
            planet_rv_kms=planet_rv_kms,
        )

        channel_builder = _ChannelBuilder(
            band=band,
            grid_um=grid_um,
            densities=densities,
            content_resolving_power=content_resolving_power,
            sky_rate_hz=sky_rate_hz,
            dark_rate_hz=dark_rate_hz,
            dark_reference_n_bins=_dark_reference_n_bins(
                spectro, band, resolving_power
            ),
        )
        channels = channel_builder.channels(resolving_power, planet_rv_kms)
        telluric_per_channel = channel_builder.telluric_per_channel(resolving_power)

        mf_kwargs: dict[str, Any] = {
            "integration_time_s": float(params.integration_time_s),
            "eps_systematic": eps_systematic,
            "servo_bw_hz": servo_bw_hz,
            "with_fringe_tracking": with_fringe_tracking,
            "continuum_filter_width_channels": int(
                params.continuum_filter_width_res_elements
            ),
            "telluric_transmission_per_channel": telluric_per_channel,
            "telluric_mask_threshold": float(params.telluric_mask_threshold),
        }
        budget = matched_filter_ccf(channels, **mf_kwargs)

        snr_vs_r = _snr_vs_r(
            r_scan,
            channel_builder=channel_builder,
            planet_rv_kms=planet_rv_kms,
            mf_kwargs=mf_kwargs,
        )
        rv_grid_kms, snr_vs_rv = _ccf_vs_rv(
            params=params,
            planet_rv_kms=planet_rv_kms,
            channels=channels,
            channel_builder=channel_builder,
            resolving_power=resolving_power,
            mf_kwargs=mf_kwargs,
        )

        component_name = unique_component_name(
            self.type_name,
            state.results.components,
            self.context.instance_id,
        )
        payload = {
            "ccf_snr": float(budget.ccf_snr),
            "ccf_snr_photon_limited": float(budget.ccf_snr_photon_limited),
            "ccf_signal_counts": float(budget.signal_counts),
            "ccf_photon_var_counts": float(budget.photon_var_counts),
            "ccf_systematic_var_counts": float(budget.systematic_var_counts),
            "systematic_ccf_leakage": float(budget.systematic_ccf_leakage),
            "ccf_systematic_rejection": (
                float(budget.systematic_ccf_leakage / leakage_rate_hz)
                if leakage_rate_hz > 0.0
                else 0.0
            ),
            "self_subtraction_factor": float(budget.self_subtraction_factor),
            "n_effective_lines": float(budget.n_effective_lines),
            "n_channels_used": int(budget.n_channels_used),
            "n_channels": int(channels.n_channels),
            "telluric_masked_fraction": _telluric_masked_fraction(
                telluric_per_channel, float(params.telluric_mask_threshold)
            ),
            "ccf_channel_centers_um": channels.channel_centers_um.astype(
                np.float64, copy=True
            ),
            "ccf_planet_rate_hz_per_channel": channels.planet_rate_hz.astype(
                np.float64, copy=True
            ),
            "ccf_leak_rate_hz_per_channel": channels.leak_rate_hz.astype(
                np.float64, copy=True
            ),
            "template_weights_per_channel": budget.template_weights.astype(
                np.float64, copy=True
            ),
            "snr_vs_resolving_power": snr_vs_r.astype(np.float64, copy=True),
            "snr_vs_resolving_power_r": r_scan.astype(np.float64, copy=True),
            "snr_vs_rv": snr_vs_rv.astype(np.float64, copy=True),
            "snr_vs_rv_kms": rv_grid_kms.astype(np.float64, copy=True),
            "integration_time_for_target_snr_s": _integration_time_for_target_snr_s(
                budget=budget,
                integration_time_s=float(params.integration_time_s),
                target_snr=float(params.target_snr),
                with_fringe_tracking=with_fringe_tracking,
            ),
            "integration_time_s": float(params.integration_time_s),
            "target_snr": float(params.target_snr),
            "resolving_power_used": float(resolving_power),
            "resolving_power_source": resolving_power_source,
            "analysis_grid_resolving_power": float(r_grid),
            "content_resolving_power": float(content_resolving_power),
            "grid_oversampling": float(params.grid_oversampling),
            "continuum_filter_width_res_elements": int(
                params.continuum_filter_width_res_elements
            ),
            "telluric_mask_threshold": float(params.telluric_mask_threshold),
            "planet_radial_velocity_kms": float(planet_rv_kms),
            "leak_normalization": float(densities.leak_normalization),
            "planet_normalization": float(densities.planet_normalization),
            "eps_systematic_used": eps_systematic,
            "servo_bw_hz_used": servo_bw_hz,
            "with_fringe_tracking": bool(with_fringe_tracking),
            "systematic_branch_source": systematic_branch_source,
            "sky_rate_hz_total": float(sky_rate_hz),
            "dark_rate_hz_total": float(dark_rate_hz),
            "upstream_spectro_component": spectro_name,
            "planet_template_provenance": templates.planet.provenance,
            "star_template_provenance": templates.star.provenance,
            "telluric_template_provenance": (
                templates.telluric.provenance
                if templates.telluric is not None
                else "none (no telluric absorption applied)"
            ),
            "ccf_model": _CCF_MODEL,
            "snr_vs_resolving_power_model": (
                "matched-filter CCF SNR rebuilt at each resolving power: new "
                "constant-R channel grid, re-convolved and re-channelized "
                "planet/leak densities, sky split over the new channel count, "
                "dark held fixed per channel. Capped at "
                f"{_R_SCAN_CONTENT_FRACTION:g} x the template content "
                "resolving power, above which the analysis grid no longer "
                "resolves the instrument line-spread function."
            ),
            "snr_vs_rv_model": (
                "dimensionless normalized detection statistic (a Cauchy-Schwarz "
                "cosine in the inverse-variance metric, not an SNR): trial "
                "templates at each radial velocity correlated against the "
                "leak-subtracted dark-port residual, with the leak's shot "
                "noise retained in the per-channel variance."
            ),
            "source_model": (
                "nullsim characterization.hdc_ccf: nullsim.physics.hdc "
                "matched-filter CCF over packaged nullsim.physics.spectra "
                "template snapshots, driven by the upstream "
                "characterization.spectro per-wavelength budgets."
            ),
        }
        results = state.results.add_component(component_name, payload)
        return replace(state, results=results)

    def external_dependencies(self) -> list[ExternalDependency]:
        """Declare every template CSV whose bytes affect the result.

        Copies the ``calibration.floor_lookup`` pattern: precompute the
        SHA-256 so the manifest carries a stable fingerprint, and fall back to
        a path-only declaration when the file cannot be read (``apply`` will
        surface the same error with a better message).

        ``star_template = None`` resolves from ``scene.star_spectral_type``,
        which is not visible here, so all three packaged stellar snapshots are
        declared as candidates rather than leaving the run hash blind to them.
        """

        params = self.params_model
        config_dir = _config_dir_from_context(self.context)
        entries: list[tuple[str, Path]] = [
            (
                "planet_template",
                _resolve_template_path(params.planet_template, config_dir),
            )
        ]
        if params.star_template is not None:
            entries.append(
                (
                    "star_template",
                    _resolve_template_path(params.star_template, config_dir),
                )
            )
        else:
            for key in sorted(_STELLAR_TEMPLATE_BY_TYPE.values()):
                entries.append(
                    (
                        f"star_template_candidate_{key}",
                        packaged_spectrum_path(key),
                    )
                )
        telluric = _normalized_optional_template(params.telluric_template)
        if telluric is not None:
            entries.append(
                ("telluric_template", _resolve_template_path(telluric, config_dir))
            )

        dependencies: list[ExternalDependency] = []
        for role, path in entries:
            source = f"characterization.hdc_ccf.{role}"
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                dependencies.append(ExternalDependency(source=source, path=path))
                continue
            dependencies.append(
                ExternalDependency(source=source, path=path, content_hash=digest)
            )
        return dependencies


_CCF_MODEL = (
    "Matched-filter cross-correlation after Wang, Mawet, Ruane, Hu & Benneke "
    "2017, AJ 153, 183 ('Observing Exoplanets with High Dispersion "
    "Coronagraphy. I.'). This is an after-the-fact analysis gain: it digs "
    "planets out of the noise at fixed nuller performance, changing no null "
    "depth, throughput, or photon rate published upstream by "
    "characterization.spectro. Planet spectrum = contrast x stellar flux x "
    "albedo x chip throughput x telluric, Doppler shifted by "
    "scene.planet_radial_velocity_kms; leaked starlight = leak fraction x "
    "stellar flux x telluric at zero radial velocity. Both are continuum "
    "filtered by a running median before the template-weighted sum. "
    "ccf_systematic_rejection = systematic_ccf_leakage / leakage_rate_hz is "
    "the template-projected fraction of the raw dark-port leak: the "
    "null-instability amplitude eps_systematic multiplies that projection "
    "rather than the whole band leak, which is where the HDC systematic gain "
    "comes from."
)


class _CoarseBand:
    """The upstream detector wavelength grid and its band edges, in microns."""

    __slots__ = (
        "wavelength_um",
        "bin_width_um",
        "lambda_min_um",
        "lambda_max_um",
        "stellar_rate_hz",
        "planet_throughput",
        "leak_fraction",
    )

    def __init__(
        self,
        *,
        wavelength_um: NDArray[np.float64],
        bin_width_um: NDArray[np.float64],
        stellar_rate_hz: NDArray[np.float64],
        planet_throughput: NDArray[np.float64],
        leak_fraction: NDArray[np.float64],
    ) -> None:
        self.wavelength_um = wavelength_um
        self.bin_width_um = bin_width_um
        self.stellar_rate_hz = stellar_rate_hz
        self.planet_throughput = planet_throughput
        self.leak_fraction = leak_fraction
        self.lambda_min_um = float(wavelength_um[0] - 0.5 * bin_width_um[0])
        self.lambda_max_um = float(wavelength_um[-1] + 0.5 * bin_width_um[-1])

    @classmethod
    def from_spectro(cls, spectro: dict[str, Any]) -> _CoarseBand:
        wavelength_um = (
            np.asarray(spectro["wavelengths_m"], dtype=np.float64) * 1.0e6
        )
        if wavelength_um.ndim != 1 or wavelength_um.size < 2:
            raise ValueError(
                "characterization.hdc_ccf requires an upstream "
                "characterization.spectro wavelength grid with at least two "
                "bins; a single-bin band carries no spectral information."
            )
        order = np.argsort(wavelength_um)
        wavelength_um = wavelength_um[order]

        def column(key: str) -> NDArray[np.float64]:
            values = np.asarray(spectro[key], dtype=np.float64)
            if values.shape != wavelength_um.shape:
                raise ValueError(
                    f"characterization.hdc_ccf: spectro payload {key!r} has "
                    f"shape {values.shape}; expected {wavelength_um.shape} to "
                    "match the wavelength grid."
                )
            return values[order]

        edges = np.empty(wavelength_um.size + 1, dtype=np.float64)
        edges[1:-1] = 0.5 * (wavelength_um[:-1] + wavelength_um[1:])
        edges[0] = wavelength_um[0] - 0.5 * (wavelength_um[1] - wavelength_um[0])
        edges[-1] = wavelength_um[-1] + 0.5 * (
            wavelength_um[-1] - wavelength_um[-2]
        )

        # The retuned-chip static leak already folds in the finite stellar disk
        # (see spectro._static_spectro_leak_fraction); the realized dynamic
        # floor is the scalar added on top, exactly as spectro forms
        # spectro_leak_fraction.
        leak_fraction = column("spectro_static_leak_fraction_per_wavelength") + float(
            spectro.get("realized_spectro_floor", 0.0)
        )
        planet_throughput = column("spectro_planet_throughput_per_wavelength")
        ratio_key = "spectro_planet_to_star_ratio_per_wavelength"
        if ratio_key in spectro:
            planet_throughput = planet_throughput * column(ratio_key)

        return cls(
            wavelength_um=wavelength_um,
            bin_width_um=np.diff(edges),
            stellar_rate_hz=column("detected_stellar_rate_by_wavelength_hz"),
            planet_throughput=planet_throughput,
            leak_fraction=np.clip(leak_fraction, 0.0, None),
        )


class _Template:
    """A band-cropped template resampled onto an exact constant-R grid.

    The packaged CSVs store their constant-R grids to ten significant digits,
    which is not log-uniform enough for `convolve_to_resolving_power`'s 1e-6
    check. Re-laying the samples on an exact `constant_r_grid` of the same
    spacing fixes that without changing the content.
    """

    __slots__ = ("spectrum", "spectral_resolving_power", "grid_resolving_power")

    def __init__(
        self,
        *,
        spectrum: TemplateSpectrum,
        spectral_resolving_power: float,
        grid_resolving_power: float,
    ) -> None:
        self.spectrum = spectrum
        self.spectral_resolving_power = spectral_resolving_power
        self.grid_resolving_power = grid_resolving_power

    @property
    def provenance(self) -> str:
        return self.spectrum.provenance

    @property
    def content_resolving_power(self) -> float:
        """Resolution the samples actually carry (content or sampling limited)."""

        return min(self.spectral_resolving_power, self.grid_resolving_power)


class _Templates:
    """The loaded planet / star / telluric templates, cropped to the band."""

    __slots__ = ("planet", "star", "telluric", "resolving_powers")

    def __init__(
        self,
        *,
        planet: _Template,
        star: _Template,
        telluric: _Template | None,
    ) -> None:
        self.planet = planet
        self.star = star
        self.telluric = telluric
        self.resolving_powers = {
            "planet": planet.content_resolving_power,
            "star": star.content_resolving_power,
            "telluric": (
                telluric.content_resolving_power
                if telluric is not None
                else math.inf
            ),
        }

    @classmethod
    def load(
        cls,
        *,
        params: HdcCcfStage.Params,
        state: SimulationState,
        config_dir: Path | None,
        lambda_min_um: float,
        lambda_max_um: float,
        max_shift_kms: float,
    ) -> _Templates:
        # Doppler shifting moves the grid by rv/c; crop with that margin plus
        # a little slack so no interpolation ever runs off the template.
        margin = 1.0e-3 + abs(max_shift_kms) / 299792.458
        lo = lambda_min_um * (1.0 - margin)
        hi = lambda_max_um * (1.0 + margin)

        star_name = params.star_template
        if star_name is None:
            star_name = _star_template_name(state.scene.star_spectral_type)
        telluric_name = _normalized_optional_template(params.telluric_template)

        planet = _load_cropped(
            params.planet_template, "albedo", config_dir, lo, hi, "planet"
        )
        star = _load_cropped(star_name, "stellar_flux", config_dir, lo, hi, "star")
        telluric = (
            _load_cropped(
                telluric_name, "transmission", config_dir, lo, hi, "telluric"
            )
            if telluric_name is not None
            else None
        )
        return cls(planet=planet, star=star, telluric=telluric)

    def content_resolving_power(self, r_grid: float) -> float:
        """Resolving power the analysis grid actually carries after sampling."""

        return min(min(self.resolving_powers.values()), float(r_grid))


class _RateDensities:
    """Planet and leak photon-rate densities (Hz/um) on the analysis grid."""

    __slots__ = (
        "grid_um",
        "telluric",
        "has_telluric",
        "leak_density",
        "leak_normalization",
        "planet_normalization",
        "_planet_density_at_rv",
    )

    def __init__(
        self,
        *,
        grid_um: NDArray[np.float64],
        telluric: NDArray[np.float64],
        has_telluric: bool,
        leak_density: NDArray[np.float64],
        leak_normalization: float,
        planet_normalization: float,
        planet_density_at_rv: Any,
    ) -> None:
        self.grid_um = grid_um
        self.telluric = telluric
        self.has_telluric = has_telluric
        self.leak_density = leak_density
        self.leak_normalization = leak_normalization
        self.planet_normalization = planet_normalization
        self._planet_density_at_rv = planet_density_at_rv

    def planet_density(self, rv_kms: float) -> NDArray[np.float64]:
        """Planet rate density (Hz/um) with the whole spectrum shifted by rv."""

        return self._planet_density_at_rv(rv_kms)

    @classmethod
    def build(
        cls,
        *,
        band: _CoarseBand,
        grid_um: NDArray[np.float64],
        templates: _Templates,
        r_grid: float,
        planet_contrast: float,
        planet_signal_rate_hz: float,
        leakage_rate_hz: float,
        planet_rv_kms: float,
    ) -> _RateDensities:
        # Coarse detector budgets, interpolated and then windowed to the band:
        # outside the detector band there is no flux, and a hard window keeps
        # the band integrals equal to the published totals.
        in_band = (grid_um >= band.lambda_min_um) & (grid_um <= band.lambda_max_um)
        stellar_density = np.where(
            in_band,
            np.interp(
                grid_um,
                band.wavelength_um,
                band.stellar_rate_hz / band.bin_width_um,
            ),
            0.0,
        )
        throughput = np.interp(grid_um, band.wavelength_um, band.planet_throughput)
        leak_fraction = np.interp(grid_um, band.wavelength_um, band.leak_fraction)

        star_shape = _shape_on_grid(templates.star, grid_um, r_grid, 0.0)
        star_norm = _weighted_mean(star_shape, stellar_density)
        star_shape = star_shape / star_norm
        stellar_density = stellar_density * star_shape

        telluric = (
            _shape_on_grid(templates.telluric, grid_um, r_grid, 0.0)
            if templates.telluric is not None
            else np.ones_like(grid_um)
        )
        telluric = np.clip(telluric, 0.0, 1.0)

        leak_density_raw = stellar_density * leak_fraction
        leak_normalization = _normalization(
            leakage_rate_hz, leak_density_raw, grid_um
        )
        leak_density = leak_density_raw * leak_normalization * telluric

        albedo_rest = _shape_on_grid(templates.planet, grid_um, r_grid, 0.0)
        albedo_norm = _weighted_mean(albedo_rest, stellar_density * throughput)

        coarse_density = np.where(
            in_band,
            np.interp(
                grid_um,
                band.wavelength_um,
                band.stellar_rate_hz / band.bin_width_um,
            ),
            0.0,
        )

        def planet_density_raw(rv_kms: float) -> NDArray[np.float64]:
            albedo = _shape_on_grid(templates.planet, grid_um, r_grid, rv_kms)
            star_rv = (
                star_shape
                if rv_kms == 0.0
                else _shape_on_grid(templates.star, grid_um, r_grid, rv_kms)
                / star_norm
            )
            return (
                planet_contrast
                * coarse_density
                * star_rv
                * throughput
                * albedo
                / albedo_norm
            )

        planet_normalization = _normalization(
            planet_signal_rate_hz, planet_density_raw(planet_rv_kms), grid_um
        )

        def planet_density(rv_kms: float) -> NDArray[np.float64]:
            return planet_density_raw(rv_kms) * planet_normalization * telluric

        return cls(
            grid_um=grid_um,
            telluric=telluric,
            has_telluric=templates.telluric is not None,
            leak_density=leak_density,
            leak_normalization=leak_normalization,
            planet_normalization=planet_normalization,
            planet_density_at_rv=planet_density,
        )


class _ChannelBuilder:
    """Turns the analysis-grid densities into `ChannelSpectra` at any R."""

    __slots__ = (
        "band",
        "grid_um",
        "densities",
        "content_resolving_power",
        "sky_rate_hz",
        "dark_rate_hz",
        "dark_reference_n_bins",
    )

    def __init__(
        self,
        *,
        band: _CoarseBand,
        grid_um: NDArray[np.float64],
        densities: _RateDensities,
        content_resolving_power: float,
        sky_rate_hz: float,
        dark_rate_hz: float,
        dark_reference_n_bins: int,
    ) -> None:
        self.band = band
        self.grid_um = grid_um
        self.densities = densities
        self.content_resolving_power = content_resolving_power
        self.sky_rate_hz = sky_rate_hz
        self.dark_rate_hz = dark_rate_hz
        self.dark_reference_n_bins = dark_reference_n_bins

    def edges(self, resolving_power: float) -> NDArray[np.float64]:
        return build_channel_edges(
            self.band.lambda_min_um, self.band.lambda_max_um, resolving_power
        )

    def telluric_per_channel(
        self, resolving_power: float
    ) -> NDArray[np.float64] | None:
        # ``None`` means "no telluric mask at all", which is what
        # matched_filter_ccf wants for a space config or a unit-transmission
        # atmosphere; anything else would mask on an all-ones array.
        if not self.densities.has_telluric:
            return None
        return np.clip(
            resample_flux_conserving(
                self.grid_um, self.densities.telluric, self.edges(resolving_power)
            ),
            0.0,
            1.0,
        )

    def channels(
        self,
        resolving_power: float,
        rv_kms: float,
        *,
        planet_only: bool = False,
    ) -> ChannelSpectra:
        edges = self.edges(resolving_power)
        n_channels = edges.size - 1
        planet = self._channelize(
            self.densities.planet_density(rv_kms), edges, resolving_power
        )
        if planet_only:
            zeros = np.zeros(n_channels, dtype=np.float64)
            return ChannelSpectra(
                channel_centers_um=_channel_centers(edges),
                channel_edges_um=edges,
                planet_rate_hz=planet,
                leak_rate_hz=zeros,
                sky_rate_hz=zeros,
                dark_rate_hz=zeros,
            )
        leak = self._channelize(
            self.densities.leak_density, edges, resolving_power
        )
        return ChannelSpectra(
            channel_centers_um=_channel_centers(edges),
            channel_edges_um=edges,
            planet_rate_hz=planet,
            leak_rate_hz=leak,
            sky_rate_hz=np.full(
                n_channels, self.sky_rate_hz / n_channels, dtype=np.float64
            ),
            dark_rate_hz=np.full(
                n_channels,
                self.dark_rate_hz / self.dark_reference_n_bins,
                dtype=np.float64,
            ),
        )

    def _channelize(
        self,
        density: NDArray[np.float64],
        edges: NDArray[np.float64],
        resolving_power: float,
    ) -> NDArray[np.float64]:
        return _clip_tiny_negatives(
            channelize(
                self.grid_um,
                density,
                edges,
                resolving_power,
                r_in=self.content_resolving_power,
            )
        )


def _snr_vs_r(
    r_values: NDArray[np.float64],
    *,
    channel_builder: _ChannelBuilder,
    planet_rv_kms: float,
    mf_kwargs: dict[str, Any],
) -> NDArray[np.float64]:
    """CCF SNR across the resolving-power scan, remasking tellurics at each R.

    ``snr_vs_resolving_power`` forwards a single fixed ``mf_kwargs`` mapping,
    but the telluric mask is per channel and therefore R dependent, so the
    scan is driven one point at a time with the mask rebuilt for that grid.
    """

    if r_values.size == 0:
        return np.empty(0, dtype=np.float64)

    def build(resolving_power: float) -> ChannelSpectra:
        return channel_builder.channels(resolving_power, planet_rv_kms)

    values = np.empty(r_values.size, dtype=np.float64)
    for index, resolving_power in enumerate(r_values):
        kwargs = dict(mf_kwargs)
        kwargs["telluric_transmission_per_channel"] = (
            channel_builder.telluric_per_channel(float(resolving_power))
        )
        values[index] = snr_vs_resolving_power(
            build, (float(resolving_power),), **kwargs
        )[0]
    return values


def _ccf_vs_rv(
    *,
    params: HdcCcfStage.Params,
    planet_rv_kms: float,
    channels: ChannelSpectra,
    channel_builder: _ChannelBuilder,
    resolving_power: float,
    mf_kwargs: dict[str, Any],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if params.rv_scan_points <= 1:
        return (np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64))

    rv_grid = np.linspace(
        planet_rv_kms - float(params.rv_scan_halfwidth_kms),
        planet_rv_kms + float(params.rv_scan_halfwidth_kms),
        int(params.rv_scan_points),
    )
    # The leaked starlight mean is measured on the bright port and removed
    # before cross-correlation (standard HDC practice, and already implicit in
    # matched_filter_ccf, whose signal term uses the planet alone). Its shot
    # noise is not removed, so it is folded into the variance budget here.
    data_channels = ChannelSpectra(
        channel_centers_um=channels.channel_centers_um,
        channel_edges_um=channels.channel_edges_um,
        planet_rate_hz=channels.planet_rate_hz,
        leak_rate_hz=np.zeros(channels.n_channels, dtype=np.float64),
        sky_rate_hz=channels.sky_rate_hz + channels.leak_rate_hz,
        dark_rate_hz=channels.dark_rate_hz,
    )

    def trial(rv_kms: float) -> ChannelSpectra:
        return channel_builder.channels(resolving_power, rv_kms, planet_only=True)

    return rv_grid, ccf_vs_rv(
        trial, rv_grid, data_channels=data_channels, **mf_kwargs
    )


def _find_spectro_component(state: SimulationState) -> tuple[str, dict[str, Any]]:
    for name, payload in state.results.components.items():
        if not name.startswith(_SPECTRO_PREFIX) or not isinstance(payload, dict):
            continue
        if _REQUIRED_SPECTRO_KEYS.issubset(payload):
            return name, payload
    raise ValueError(
        "characterization.hdc_ccf requires an upstream characterization.spectro "
        "component carrying the per-wavelength instrument budgets "
        f"({sorted(_REQUIRED_SPECTRO_KEYS)}). Add the characterization.spectro "
        "stage before characterization.hdc_ccf so the cross-correlation reuses "
        "the same retuned chip throughput, leakage, and photon rates."
    )


def _resolve_resolving_power(
    spectro: dict[str, Any], override: float | None
) -> tuple[float, str]:
    if override is not None:
        return float(override), "characterization.hdc_ccf resolving_power param"
    spectrograph = str(spectro.get("spectrograph", "awg")).lower()
    if spectrograph == "mkid":
        value = spectro.get("mkid_resolving_power")
        key = "mkid_resolving_power"
    else:
        value = spectro.get("awg_resolving_power")
        key = "awg_resolving_power"
    if value is None or not (float(value) > 0.0):
        raise ValueError(
            "characterization.hdc_ccf could not resolve a spectrograph "
            f"resolving power: the upstream spectro payload has no usable "
            f"{key!r}. Set the stage's resolving_power param explicitly."
        )
    return float(value), f"characterization.spectro {key}"


def _spectro_integration_time_s(spectro: dict[str, Any]) -> float:
    """Recover the upstream integration time from published counts and rates.

    ``characterization.spectro`` publishes ``S``/``B`` as counts and
    ``planet_signal_rate_hz``/``leakage_rate_hz`` as rates, but not the
    integration time itself; the ratio recovers it exactly.
    """

    for count_key, rate_key in (
        ("S", "planet_signal_rate_hz"),
        ("B", "leakage_rate_hz"),
    ):
        count = spectro.get(count_key)
        rate = spectro.get(rate_key)
        if count is None or rate is None:
            continue
        if float(rate) > 0.0 and float(count) > 0.0:
            return float(count) / float(rate)
    raise ValueError(
        "characterization.hdc_ccf could not recover the upstream integration "
        "time from the characterization.spectro payload: none of the "
        "(S, planet_signal_rate_hz) or (B, leakage_rate_hz) pairs are both "
        "positive."
    )


def _background_rates_hz(
    spectro: dict[str, Any], integration_time_s: float
) -> tuple[float, float]:
    """Band-total sky and dark rates in Hz from the published count budgets."""

    sky_counts = float(spectro.get("Sky", 0.0) or 0.0)
    dark_counts = float(spectro.get("D", 0.0) or 0.0)
    return (
        max(sky_counts, 0.0) / integration_time_s,
        max(dark_counts, 0.0) / integration_time_s,
    )


def _dark_reference_n_bins(
    spectro: dict[str, Any], band: _CoarseBand, resolving_power: float
) -> int:
    """Channel count the published dark budget was spread over.

    Dark counts are per detector pixel, so the *per channel* dark rate is what
    is physically fixed: the band total grows with the channel count. The AWG
    payload states its bin count directly; otherwise fall back to the channel
    count at the working resolving power.
    """

    n_bins = spectro.get("awg_n_bins")
    if n_bins is not None and int(n_bins) > 0:
        return int(n_bins)
    return max(
        build_channel_edges(
            band.lambda_min_um, band.lambda_max_um, resolving_power
        ).size
        - 1,
        1,
    )


def _infer_fringe_tracking(
    spectro: dict[str, Any],
    *,
    eps_systematic: float,
    leakage_rate_hz: float,
    integration_time_s: float,
) -> tuple[bool, str]:
    """Recover which systematic branch ``characterization.spectro`` used.

    The spectro stage does not publish ``with_fringe_tracking``, but it does
    publish ``systematic_var``, and the two branches of
    ``performance.standard._systematic_variance`` differ by a factor of
    ``2 f_servo t``. Match the published variance against both closed forms.
    """

    published = spectro.get("systematic_var")
    servo_bw_hz = float(spectro.get("servo_bw_hz", 0.0) or 0.0)
    if (
        published is None
        or eps_systematic <= 0.0
        or leakage_rate_hz <= 0.0
        or servo_bw_hz <= 0.0
        or float(published) <= 0.0
    ):
        return True, (
            "assumed fringe-tracked: the upstream characterization.spectro "
            "payload carries no non-zero systematic_var to discriminate the "
            "branch"
        )

    tracked = (
        eps_systematic**2
        * leakage_rate_hz**2
        * integration_time_s
        / (2.0 * servo_bw_hz)
    )
    untracked = (eps_systematic * leakage_rate_hz * integration_time_s) ** 2
    value = float(published)
    if abs(math.log(value / tracked)) <= abs(math.log(value / untracked)):
        return True, (
            "inferred fringe-tracked from characterization.spectro "
            "systematic_var = eps^2 B^2 t / (2 f_servo)"
        )
    return False, (
        "inferred not fringe-tracked from characterization.spectro "
        "systematic_var = (eps B t)^2"
    )


def _planet_contrast(state: SimulationState) -> float:
    contrast = state.scene.planet_contrast
    if contrast is None:
        raise ValueError(
            "characterization.hdc_ccf requires scene.planet_contrast so the "
            "planet spectrum keeps its band-integrated contrast meaning."
        )
    return float(contrast)


def _planet_radial_velocity_kms(state: SimulationState) -> float:
    value = state.scene.planet_radial_velocity_kms
    if value is None:
        return _DEFAULT_PLANET_RV_KMS
    return float(value)


def _star_template_name(spectral_type: str | None) -> str:
    if spectral_type:
        for character in str(spectral_type):
            key = character.upper()
            if key in _STELLAR_TEMPLATE_BY_TYPE:
                return _STELLAR_TEMPLATE_BY_TYPE[key]
    raise ValueError(
        "characterization.hdc_ccf could not pick a stellar template from "
        f"scene.star_spectral_type={spectral_type!r}. Packaged snapshots cover "
        f"{sorted(_STELLAR_TEMPLATE_BY_TYPE)} spectral types; set the stage's "
        "star_template param to a packaged name or a CSV path."
    )


def _normalized_optional_template(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "none":
        return None
    return text


def _looks_like_path(spec: str) -> bool:
    return "/" in spec or spec.lower().endswith(".csv")


def _resolve_template_path(spec: str, config_dir: Path | None) -> Path:
    """Resolve a packaged template name or a (possibly relative) CSV path.

    Bare names go through ``packaged_spectrum_path`` so an unknown name raises
    a ``KeyError`` listing the valid ones instead of a confusing missing-file
    error. Paths resolve against the config directory first, then the working
    directory, then the package root -- the ``calibration.floor_lookup``
    convention.
    """

    text = str(spec).strip()
    if not _looks_like_path(text):
        return packaged_spectrum_path(text)
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return candidate
    if config_dir is not None:
        config_relative = (config_dir / candidate).resolve()
        if config_relative.exists():
            return config_relative
    if candidate.exists():
        return candidate
    import nullsim

    package_root = Path(nullsim.__file__).resolve().parent.parent
    bundled = package_root / candidate
    if bundled.exists():
        return bundled
    return candidate


def _config_dir_from_context(context: Any) -> Path | None:
    source_path = getattr(context, "source_path", None)
    if source_path is None:
        return None
    return Path(source_path).expanduser().resolve().parent


def _load_cropped(
    spec: str,
    kind: str,
    config_dir: Path | None,
    lambda_min_um: float,
    lambda_max_um: float,
    role: str,
) -> _Template:
    path = _resolve_template_path(spec, config_dir)
    spectrum = load_template_csv(path, kind=kind)
    if (
        spectrum.wavelength_um[0] > lambda_min_um
        or spectrum.wavelength_um[-1] < lambda_max_um
    ):
        raise ValueError(
            f"characterization.hdc_ccf {role} template {path} covers "
            f"[{spectrum.wavelength_um[0]:.4g}, {spectrum.wavelength_um[-1]:.4g}] "
            f"um but the analysis band needs [{lambda_min_um:.4g}, "
            f"{lambda_max_um:.4g}] um including the Doppler-shift margin."
        )
    start = max(
        int(np.searchsorted(spectrum.wavelength_um, lambda_min_um, "left")) - 1, 0
    )
    stop = min(
        int(np.searchsorted(spectrum.wavelength_um, lambda_max_um, "right")) + 1,
        spectrum.wavelength_um.size,
    )
    if stop - start < 3:
        raise ValueError(
            f"characterization.hdc_ccf {role} template {path} has fewer than "
            "three samples inside the analysis band."
        )
    wavelength = spectrum.wavelength_um[start:stop]
    values = spectrum.values[start:stop]

    grid_resolving_power = 1.0 / float(np.median(np.diff(np.log(wavelength))))
    exact_grid = constant_r_grid(
        float(wavelength[0]), float(wavelength[-1]), grid_resolving_power
    )
    return _Template(
        spectrum=TemplateSpectrum(
            wavelength_um=exact_grid,
            values=np.interp(exact_grid, wavelength, values),
            kind=spectrum.kind,
            provenance=spectrum.provenance,
        ),
        spectral_resolving_power=_template_resolving_power(spectrum),
        grid_resolving_power=grid_resolving_power,
    )


def _template_resolving_power(spectrum: TemplateSpectrum) -> float:
    """Read the snapshot's own resolving power out of its provenance header."""

    match = _RESOLVING_POWER_RE.search(spectrum.provenance)
    if match is None:
        return math.inf
    try:
        value = float(match.group(1))
    except ValueError:
        return math.inf
    return value if value > 0.0 else math.inf


def _shape_on_grid(
    template: _Template,
    grid_um: NDArray[np.float64],
    r_grid: float,
    rv_kms: float,
) -> NDArray[np.float64]:
    """Doppler shift, degrade to the analysis grid, and sample onto it.

    Smoothing to the analysis grid's own resolving power before point sampling
    keeps template structure the grid cannot represent from aliasing into it.
    The shift happens first, on the template's native (finer) grid, so a
    sub-resolution-element radial velocity survives.
    """

    spectrum = template.spectrum
    shifted = spectrum if rv_kms == 0.0 else doppler_shift(spectrum, rv_kms)
    if template.content_resolving_power > r_grid:
        values = convolve_to_resolving_power(
            shifted.wavelength_um,
            shifted.values,
            r_grid,
            r_in=template.spectral_resolving_power,
        )
    else:
        values = shifted.values
    return np.interp(grid_um, shifted.wavelength_um, values)


def _weighted_mean(
    values: NDArray[np.float64], weights: NDArray[np.float64]
) -> float:
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError(
            "characterization.hdc_ccf cannot normalize a spectral shape: the "
            "weighting photon-rate density sums to zero over the band."
        )
    mean = float(np.sum(weights * values)) / total
    if not (mean > 0.0) or not math.isfinite(mean):
        raise ValueError(
            "characterization.hdc_ccf spectral shape has a non-positive "
            f"band-weighted mean ({mean!r}); check the template values."
        )
    return mean


def _normalization(
    target_rate_hz: float, density: NDArray[np.float64], grid_um: NDArray[np.float64]
) -> float:
    """Factor that makes the density's band integral match a published rate."""

    integral = float(np.trapezoid(density, grid_um))
    if integral <= 0.0:
        if target_rate_hz <= 0.0:
            return 1.0
        raise ValueError(
            "characterization.hdc_ccf built a non-positive band integral for a "
            f"positive published rate ({target_rate_hz!r} Hz); the spectral "
            "shapes or the coarse budgets are inconsistent."
        )
    if target_rate_hz <= 0.0:
        return 0.0
    return target_rate_hz / integral


def _channel_centers(edges: NDArray[np.float64]) -> NDArray[np.float64]:
    # Geometric mean: the channel grid is log-uniform, so this is the centre in
    # the coordinate the channels are actually uniform in.
    return np.sqrt(edges[:-1] * edges[1:])


def _clip_tiny_negatives(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Zero rounding-level negatives; anything larger is a real bug."""

    array = np.asarray(values, dtype=np.float64)
    if not np.any(array < 0.0):
        return array
    scale = float(np.max(np.abs(array)))
    tolerance = 1.0e-9 * scale if scale > 0.0 else 0.0
    worst = float(np.min(array))
    if worst < -tolerance:
        raise ValueError(
            "characterization.hdc_ccf produced a physically negative channel "
            f"rate ({worst!r} Hz); this is not an interpolation rounding "
            "artifact."
        )
    return np.clip(array, 0.0, None)


def _r_scan_values(
    n_points: int, content_resolving_power: float
) -> NDArray[np.float64]:
    if n_points <= 1:
        return np.empty(0, dtype=np.float64)
    r_max = min(_R_SCAN_MAX, _R_SCAN_CONTENT_FRACTION * content_resolving_power)
    if r_max <= _R_SCAN_MIN:
        return np.empty(0, dtype=np.float64)
    return np.logspace(
        math.log10(_R_SCAN_MIN), math.log10(r_max), int(n_points)
    ).astype(np.float64)


def _telluric_masked_fraction(
    telluric_per_channel: NDArray[np.float64] | None, threshold: float
) -> float:
    if telluric_per_channel is None or telluric_per_channel.size == 0:
        return 0.0
    return float(np.mean(telluric_per_channel < threshold))


def _integration_time_for_target_snr_s(
    *,
    budget: Any,
    integration_time_s: float,
    target_snr: float,
    with_fringe_tracking: bool,
) -> float:
    """Solve the CCF SNR for the time that reaches ``target_snr``.

    With fringe tracking both variances grow like ``t`` while the signal grows
    like ``t``, so ``SNR ~ sqrt(t)`` and the scaling is exact. Without it the
    systematic variance grows like ``t**2`` and the SNR saturates, so the
    quadratic is solved the same way as
    ``performance.standard._integration_time_for_snr``.
    """

    snr = float(budget.ccf_snr)
    if snr <= 0.0 or integration_time_s <= 0.0:
        return math.inf
    if with_fringe_tracking:
        return integration_time_s * (target_snr / snr) ** 2

    signal_rate = float(budget.signal_counts) / integration_time_s
    photon_var_rate = float(budget.photon_var_counts) / integration_time_s
    systematic_var_rate_sq = float(budget.systematic_var_counts) / (
        integration_time_s**2
    )
    denominator = signal_rate**2 - target_snr**2 * systematic_var_rate_sq
    if denominator <= 0.0:
        return math.inf
    return target_snr**2 * photon_var_rate / denominator
