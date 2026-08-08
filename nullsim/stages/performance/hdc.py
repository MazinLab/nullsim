"""Detection-mode high dispersion coronagraphy contrast curves.

Where ``characterization.hdc_ccf`` scores a *known* planet on the spectro port,
this stage inverts the same matched filter on the *dark* ports: at every
separation anchor the sensitivity stage published, it asks how faint a planet
the observation would still detect at ``target_snr``. That inversion is
`nullsim.physics.hdc.ccf_contrast_at_target_snr`, and stringing its answers
across the anchors is a CCF contrast curve — the HDC counterpart of the
broadband curve ``performance.standard`` already publishes.

As in characterization mode, nothing about the nuller changes: the null depth,
the throughputs, the leakage, and the photon budgets are exactly the ones
``performance.standard`` and ``sensitivity.detection_curve`` published. HDC is
an after-the-fact analysis gain, bought by the spectral asymmetry between a
smooth leaked-starlight continuum and the planet's line forest shifted off the
stellar rest frame. Framing follows Wang, Mawet, Ruane, Hu & Benneke 2017,
AJ 153, 183.

Reuse: the spectral machinery is imported wholesale from
``nullsim.stages.characterization.hdc_ccf`` rather than restated — template
loading and cropping, the stellar photon-flux-to-energy-flux lambda division,
the telluric unit-band-mean shape plus raw absolute mask, the high-resolution
analysis grid, the density normalizations, and the channel builder. Only the
*inputs* to that machinery differ, and they differ in two ways worth knowing:

* The upstream stellar budget here is a single band total
  (``stellar_rate_hz_total``), not a per-wavelength array. It is spread across
  the band as a rate density proportional to lambda, which is the photon-rate
  tilt of the flat-``F_lambda`` zero-point model that produced it
  (``physics/photometry.py``). The stellar template then contributes only its
  ``F_lambda`` shape, exactly as in characterization mode.
* Both the planet throughput and the null are grey across the band here: the
  sensitivity stage publishes one throughput per separation, and
  ``performance.standard`` one leakage rate. So the anchors differ only in
  amplitude, and the channelization is done once and scaled — exact, because
  convolution and binning are linear.

Deterministic: the stage draws no random numbers.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField

from nullsim.physics.hdc import (
    ChannelSpectra,
    ccf_contrast_at_target_snr,
    matched_filter_ccf,
)
from nullsim.pipeline import (
    BaseStage,
    ExternalDependency,
    SimulationState,
    register_stage,
)
from nullsim.stages._utils import unique_component_name

# Phase 1 spectral machinery. Cross-stage private imports are the established
# convention in this package (characterization.detection_curve imports from
# characterization.spectro, which imports from performance.standard); sharing
# beats restating, because a drifting copy of the lambda-division or the
# telluric shape rule would silently disagree with characterization mode.
from nullsim.stages.characterization.hdc_ccf import (
    _CoarseBand,
    _ChannelBuilder,
    _DetectionBudget,
    _RateDensities,
    _Templates,
    _config_dir_from_context,
    _normalized_optional_template,
    _planet_radial_velocity_kms,
    _resolve_fringe_tracking,
    _resolve_template_path,
    _template_sha256,
    _R_SCAN_MAX,
)
from nullsim.physics.spectra import constant_r_grid, packaged_spectrum_path

__all__ = ["HdcPerformanceStage"]

_PERFORMANCE_PREFIX = "performance.standard"
_SENSITIVITY_PREFIX = "sensitivity.detection_curve"
_STELLAR_TEMPLATE_KEYS = ("G", "K", "M")

_REQUIRED_PERFORMANCE_KEYS = frozenset(
    {
        "leakage_rate_hz",
        "stellar_rate_hz_total",
        "sky_rate_hz_per_port",
        "dark_rate_hz_per_port",
        "n_dark_ports",
        "eps_systematic_used",
        "servo_bw_hz",
    }
)
_REQUIRED_SENSITIVITY_KEYS = frozenset(
    {"separations_mas", "planet_throughputs", "wavelengths_m"}
)

# One matched-filter refinement is enough when the correction is this small.
_REFINEMENT_TOLERANCE = 0.02


@register_stage
class HdcPerformanceStage(BaseStage):
    """CCF contrast curve across the sensitivity stage's separation anchors."""

    type_name = "performance.hdc"
    family = "performance"
    consumes = frozenset({"results", "scene"})
    produces = frozenset({"results"})

    class Params(BaseModel):
        """Validated parameters for the detection-mode HDC contrast curve."""

        model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

        planet_template: str = "earth"
        star_template: str | None = None
        telluric_template: str | None = None
        # Dispersal of the dark ports. Unlike characterization mode there is no
        # upstream AWG payload to inherit, so the disperser is configured here.
        awg_resolving_power: float = PydanticField(default=3000.0, gt=0.0)
        awg_throughput: float = PydanticField(default=0.8, ge=0.0, le=1.0)
        dark_count_hz_per_channel: float | None = PydanticField(
            default=None, ge=0.0
        )
        telluric_mask_threshold: float = PydanticField(default=0.8, ge=0.0, le=1.0)
        grid_oversampling: float = PydanticField(default=3.0, gt=1.0)
        continuum_filter_width_res_elements: int = PydanticField(default=30, ge=0)
        integration_time_s: float | None = PydanticField(default=None, gt=0.0)
        target_snr: float = PydanticField(default=5.0, gt=0.0)
        # Explicit upstream selection; None takes the most recently added
        # compatible component, which is the one immediately upstream.
        upstream_performance: str | None = None
        upstream_sensitivity: str | None = None

    def apply(self, state: SimulationState) -> SimulationState:
        params = self.params_model

        performance_name, performance = _find_component(
            state,
            prefix=_PERFORMANCE_PREFIX,
            required=_REQUIRED_PERFORMANCE_KEYS,
            requested=params.upstream_performance,
            param_name="upstream_performance",
        )
        sensitivity_name, sensitivity = _find_component(
            state,
            prefix=_SENSITIVITY_PREFIX,
            required=_REQUIRED_SENSITIVITY_KEYS,
            requested=params.upstream_sensitivity,
            param_name="upstream_sensitivity",
        )

        separations_mas = np.asarray(
            sensitivity["separations_mas"], dtype=np.float64
        )
        throughputs = np.asarray(
            sensitivity["planet_throughputs"], dtype=np.float64
        )
        if separations_mas.ndim != 1 or separations_mas.size == 0:
            raise ValueError(
                "performance.hdc requires a non-empty separations_mas grid from "
                f"{sensitivity_name!r}."
            )
        if throughputs.shape != separations_mas.shape:
            raise ValueError(
                "performance.hdc: sensitivity planet_throughputs has shape "
                f"{throughputs.shape}; expected {separations_mas.shape} to match "
                "separations_mas."
            )

        integration_time_s, integration_time_source = _resolve_integration_time_s(
            performance, params.integration_time_s
        )
        eps_systematic = float(performance["eps_systematic_used"])
        servo_bw_hz = float(performance["servo_bw_hz"])
        with_fringe_tracking, systematic_branch_source = _resolve_fringe_tracking(
            performance,
            eps_systematic=eps_systematic,
            leakage_rate_hz=float(performance["leakage_rate_hz"]),
            integration_time_s=integration_time_s,
        )

        stellar_rate_hz_total = float(performance["stellar_rate_hz_total"])
        if not (stellar_rate_hz_total > 0.0):
            raise ValueError(
                "performance.hdc requires a positive stellar_rate_hz_total from "
                f"{performance_name!r}; got {stellar_rate_hz_total!r}."
            )
        n_dark_ports = max(int(performance["n_dark_ports"]), 1)

        band = _band_from_upstream(sensitivity, stellar_rate_hz_total)
        resolving_power = float(params.awg_resolving_power)
        r_grid = float(params.grid_oversampling) * max(
            resolving_power, _R_SCAN_MAX
        )
        grid_um = constant_r_grid(
            band.lambda_min_um, band.lambda_max_um, r_grid
        )

        planet_rv_kms = _planet_radial_velocity_kms(state)
        templates = _Templates.load(
            params=params,
            state=state,
            config_dir=_config_dir_from_context(self.context),
            lambda_min_um=band.lambda_min_um,
            lambda_max_um=float(grid_um[-1]),
            max_shift_kms=abs(planet_rv_kms),
        )
        content_resolving_power = templates.content_resolving_power(r_grid)
        if content_resolving_power < resolving_power:
            raise ValueError(
                "performance.hdc: the spectral templates only resolve "
                f"R={content_resolving_power:.4g}, below the requested dark-port "
                f"R={resolving_power:.4g}. Raise grid_oversampling or use finer "
                "template snapshots."
            )

        # Reference amplitudes. The planet reference is the unit-contrast band
        # rate at unit planet throughput; the leak reference is unity. Both the
        # throughput and the null are grey, so per anchor these scale by a pure
        # number and the channelization below is done exactly once.
        reference_budget = _DetectionBudget(
            planet_rate_hz=stellar_rate_hz_total * float(params.awg_throughput),
            leak_rate_hz=1.0,
            sky_rate_hz=(
                float(performance["sky_rate_hz_per_port"]) * n_dark_ports
            ),
            dark_rate_hz_per_channel=_dark_rate_hz_per_channel(
                performance, params.dark_count_hz_per_channel, n_dark_ports
            ),
            observed_band_um=None,
            source="performance.hdc reference amplitudes",
        )
        densities = _RateDensities.build(
            band=band,
            grid_um=grid_um,
            templates=templates,
            r_grid=r_grid,
            planet_contrast=1.0,
            planet_signal_rate_hz=reference_budget.planet_rate_hz,
            leakage_rate_hz=reference_budget.leak_rate_hz,
            planet_rv_kms=planet_rv_kms,
        )
        channel_builder = _ChannelBuilder(
            band=band,
            grid_um=grid_um,
            densities=densities,
            content_resolving_power=content_resolving_power,
            budget=reference_budget,
        )
        reference_channels = channel_builder.channels(
            resolving_power, planet_rv_kms
        )
        telluric_per_channel = channel_builder.telluric_per_channel(
            resolving_power
        )
        n_channels = reference_channels.n_channels

        leak_per_anchor = _leak_rate_hz_per_anchor(
            performance, separations_mas, float(params.awg_throughput)
        )
        sky_per_anchor = _sky_rate_hz_per_anchor(
            performance, separations_mas, n_dark_ports
        )

        solver_kwargs: dict[str, Any] = {
            "integration_time_s": integration_time_s,
            "target_snr": float(params.target_snr),
            "eps_systematic": eps_systematic,
            "servo_bw_hz": servo_bw_hz,
            "with_fringe_tracking": with_fringe_tracking,
            "continuum_filter_width_channels": int(
                params.continuum_filter_width_res_elements
            ),
            "telluric_transmission_per_channel": telluric_per_channel,
            "telluric_mask_threshold": float(params.telluric_mask_threshold),
        }

        n_anchors = separations_mas.size
        contrast = np.full(n_anchors, np.nan, dtype=np.float64)
        refined_snr = np.full(n_anchors, np.nan, dtype=np.float64)
        optimism = np.full(n_anchors, np.nan, dtype=np.float64)
        n_effective_lines = np.full(n_anchors, np.nan, dtype=np.float64)
        systematic_var = np.full(n_anchors, np.nan, dtype=np.float64)
        photon_var_zero = np.full(n_anchors, np.nan, dtype=np.float64)
        reasons: list[str | None] = [None] * n_anchors
        corrected = 0

        for index in range(n_anchors):
            planet_unit = reference_channels.planet_rate_hz * float(
                throughputs[index]
            )
            leak = reference_channels.leak_rate_hz * float(leak_per_anchor[index])
            sky = reference_channels.sky_rate_hz * (
                float(sky_per_anchor[index]) / reference_budget.sky_rate_hz
                if reference_budget.sky_rate_hz > 0.0
                else 0.0
            )
            dark = reference_channels.dark_rate_hz
            try:
                solution = ccf_contrast_at_target_snr(
                    planet_unit, leak, sky, dark, **solver_kwargs
                )
            except ValueError as error:
                # Reachable in ordinary sweeps: a low-throughput anchor whose
                # lines all land in telluric-masked channels has no contrast
                # that reaches the target. Record the anchor as a
                # non-detection instead of aborting the whole curve.
                reasons[index] = str(error)
                continue

            value, snr, did_correct = _refine_contrast(
                solution.contrast_at_target_snr,
                planet_unit_rate_hz=planet_unit,
                channels=reference_channels,
                leak=leak,
                sky=sky,
                dark=dark,
                target_snr=float(params.target_snr),
                solver_kwargs=solver_kwargs,
            )
            contrast[index] = value
            refined_snr[index] = snr
            optimism[index] = (
                float(params.target_snr) / snr - 1.0 if snr > 0.0 else np.nan
            )
            corrected += int(did_correct)
            systematic_var[index] = float(solution.systematic_var_counts)
            photon_var_zero[index] = float(solution.photon_var_zero_contrast)
            n_effective_lines[index] = _participation_ratio(
                solution.template_weights
            )

        broadband = _broadband_contrast(performance, separations_mas)
        with np.errstate(divide="ignore", invalid="ignore"):
            gain = broadband / contrast

        component_name = unique_component_name(
            self.type_name,
            state.results.components,
            self.context.instance_id,
        )
        payload = {
            "separations_mas": separations_mas.astype(np.float64, copy=True),
            "ccf_contrast_5sigma": contrast.astype(np.float64, copy=True),
            "broadband_contrast_5sigma": broadband.astype(np.float64, copy=True),
            "ccf_contrast_gain": np.asarray(gain, dtype=np.float64),
            "target_snr": float(params.target_snr),
            "integration_time_s": float(integration_time_s),
            "integration_time_source": integration_time_source,
            "resolving_power_used": float(resolving_power),
            "awg_throughput_used": float(params.awg_throughput),
            "n_channels": int(n_channels),
            "ccf_channel_centers_um": (
                reference_channels.channel_centers_um.astype(np.float64, copy=True)
            ),
            "observed_band_um": (
                float(band.lambda_min_um),
                float(band.lambda_max_um),
            ),
            "n_effective_lines_per_anchor": n_effective_lines.astype(
                np.float64, copy=True
            ),
            "systematic_var_counts_per_anchor": systematic_var.astype(
                np.float64, copy=True
            ),
            "photon_var_zero_contrast_per_anchor": photon_var_zero.astype(
                np.float64, copy=True
            ),
            "refined_snr_per_anchor": refined_snr.astype(np.float64, copy=True),
            "solver_optimism_per_anchor": optimism.astype(np.float64, copy=True),
            "corrected_anchor_count": int(corrected),
            "non_detection_reasons": tuple(reasons),
            "n_non_detections": int(sum(1 for r in reasons if r is not None)),
            "leak_rate_hz_per_anchor": leak_per_anchor.astype(
                np.float64, copy=True
            ),
            "planet_throughputs": throughputs.astype(np.float64, copy=True),
            "stellar_rate_hz_total": stellar_rate_hz_total,
            "eps_systematic_used": eps_systematic,
            "servo_bw_hz_used": servo_bw_hz,
            "with_fringe_tracking_used": bool(with_fringe_tracking),
            "systematic_branch_source": systematic_branch_source,
            "planet_template_provenance": templates.planet.provenance,
            "star_template_provenance": templates.star.provenance,
            "telluric_template_provenance": (
                templates.telluric.provenance
                if templates.telluric is not None
                else "none (no telluric absorption applied)"
            ),
            "planet_template_sha256": _template_sha256(templates.planet.path),
            "star_template_sha256": _template_sha256(templates.star.path),
            "telluric_template_sha256": (
                _template_sha256(templates.telluric.path)
                if templates.telluric is not None
                else None
            ),
            "upstream_performance_component": performance_name,
            "upstream_sensitivity_component": sensitivity_name,
            "ccf_model": _CCF_MODEL,
            "detection_budget_model": _detection_budget_model(
                params, performance, n_dark_ports
            ),
            "leak_shape_model": _LEAK_SHAPE_MODEL,
            "source_model": (
                "nullsim performance.hdc: "
                "nullsim.physics.hdc.ccf_contrast_at_target_snr inverted at "
                "each sensitivity.detection_curve separation anchor, over the "
                "spectral machinery shared with characterization.hdc_ccf."
            ),
        }
        results = state.results.add_component(component_name, payload)
        return replace(state, results=results)

    def external_dependencies(self) -> list[ExternalDependency]:
        """Declare every template CSV whose bytes affect the curve.

        Same hashed pattern as ``characterization.hdc_ccf``, including the
        three-candidate stellar declaration: ``star_template = None`` resolves
        from ``scene.star_spectral_type``, which is not visible here.
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
            for key in _STELLAR_TEMPLATE_KEYS:
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
            source = f"performance.hdc.{role}"
            digest = _template_sha256(path)
            if digest is None:
                dependencies.append(ExternalDependency(source=source, path=path))
                continue
            dependencies.append(
                ExternalDependency(source=source, path=path, content_hash=digest)
            )
        return dependencies


_CCF_MODEL = (
    "Detection-mode matched-filter cross-correlation after Wang, Mawet, Ruane, "
    "Hu & Benneke 2017, AJ 153, 183. At each separation anchor the matched "
    "filter is inverted -- nullsim.physics.hdc.ccf_contrast_at_target_snr "
    "solves SNR(c) = target_snr in closed form -- to give the faintest planet "
    "the same observation still detects. This is an after-the-fact analysis "
    "gain at FIXED nuller performance: the null depth, planet throughputs, "
    "leakage, and photon budgets are exactly the ones performance.standard and "
    "sensitivity.detection_curve published, and nothing here changes them. "
    "The closed form freezes the matched-filter weights at zero contrast, "
    "which makes it generically FIRST-ORDER OPTIMISTIC once the systematic "
    "share of the variance is non-negligible (see the solver docstring's "
    "eta_sys expansion). Every anchor is therefore refined: one "
    "matched_filter_ccf evaluation at the returned contrast gives the achieved "
    "SNR (refined_snr_per_anchor) and the optimism "
    "(solver_optimism_per_anchor = target/refined - 1, positive when the "
    "solver was optimistic); if the achieved SNR is off target by more than "
    "2%, the contrast is corrected by a log-log secant on SNR(c) and "
    "re-evaluated, and the published contrast and refined SNR are the "
    "post-correction values. The correction is not a fixed power of "
    "target/refined: SNR(c) is linear in c while the leak dominates the "
    "variance -- where a contrast curve lives -- and only becomes sqrt(c) once "
    "the planet's own shot noise takes over, so a fixed square exponent "
    "overshoots in the linear regime. corrected_anchor_count records how many "
    "anchors needed the step."
)

_LEAK_SHAPE_MODEL = (
    "Leaked starlight is distributed with the stellar x telluric line shape "
    "and a GREY null: performance.standard publishes one band-integrated "
    "leakage rate, not a per-wavelength null depth, so the leak spectrum here "
    "is the stellar spectrum scaled to that total (per anchor, using "
    "contrast_curve_leakage_rate_hz when the upstream stage published a "
    "per-separation curve, otherwise the scalar leakage_rate_hz). A "
    "chromatic null would change the leak's spectral structure and hence the "
    "template projection; this model cannot represent that."
)


def _find_component(
    state: SimulationState,
    *,
    prefix: str,
    required: frozenset[str],
    requested: str | None,
    param_name: str,
) -> tuple[str, dict[str, Any]]:
    """Locate an upstream component by explicit name or by pipeline recency.

    Mirrors the phase 1 ``characterization.hdc_ccf`` fix: with a name given it
    must exist and be compatible; otherwise the MOST RECENTLY ADDED compatible
    component wins, which in insertion order is the one immediately upstream.
    """

    components = state.results.components
    if requested is not None:
        payload = components.get(requested)
        if payload is None:
            available = sorted(
                name for name in components if name.startswith(prefix)
            )
            raise ValueError(
                f"performance.hdc {param_name} {requested!r} is not present on "
                f"state.results. Available {prefix} components: {available}."
            )
        if not isinstance(payload, dict) or not required.issubset(payload):
            missing = sorted(
                required - set(payload)
                if isinstance(payload, dict)
                else required
            )
            raise ValueError(
                f"performance.hdc {param_name} {requested!r} is not a usable "
                f"{prefix} payload; missing keys {missing}."
            )
        return requested, payload

    for name, payload in reversed(list(components.items())):
        if not name.startswith(prefix) or not isinstance(payload, dict):
            continue
        if required.issubset(payload):
            return name, payload
    raise ValueError(
        f"performance.hdc requires an upstream {prefix} component carrying "
        f"{sorted(required)}. Add the {prefix} stage before performance.hdc so "
        "the cross-correlation reuses the same throughput, leakage, and photon "
        "budgets."
    )


def _band_from_upstream(
    sensitivity: dict[str, Any], stellar_rate_hz_total: float
) -> _CoarseBand:
    """Build the coarse band from the sensitivity wavelength grid.

    Detection mode has no per-wavelength stellar array to inherit -- only the
    band total -- so the total is spread as a rate density proportional to
    lambda. That is the photon-rate tilt of the flat-``F_lambda`` zero-point
    model that produced it: ``photon_rate_per_m2_per_bin`` divides a fixed band
    flux density by the photon energy, giving a rate density of ``lambda *
    F_flat``. The stellar template then supplies only the ``F_lambda`` shape
    (it is lambda-divided at load), so the product is ``lambda * F_lambda`` --
    the same convention characterization mode arrives at from its per-bin
    array.

    Planet throughput and leak fraction are grey and get renormalized away, so
    they are carried as ones; the anchors differ only in amplitude.
    """

    wavelength_um = np.asarray(sensitivity["wavelengths_m"], dtype=np.float64) * 1.0e6
    if wavelength_um.ndim != 1 or wavelength_um.size < 2:
        raise ValueError(
            "performance.hdc requires a sensitivity.detection_curve wavelength "
            "grid with at least two bins; a single-bin band carries no spectral "
            "information."
        )
    wavelength_um = np.sort(wavelength_um)

    edges = np.empty(wavelength_um.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (wavelength_um[:-1] + wavelength_um[1:])
    edges[0] = wavelength_um[0] - 0.5 * (wavelength_um[1] - wavelength_um[0])
    edges[-1] = wavelength_um[-1] + 0.5 * (wavelength_um[-1] - wavelength_um[-2])
    bin_width_um = np.diff(edges)

    tilt = wavelength_um * bin_width_um
    stellar_rate_hz = stellar_rate_hz_total * tilt / float(np.sum(tilt))
    return _CoarseBand(
        wavelength_um=wavelength_um,
        bin_width_um=bin_width_um,
        stellar_rate_hz=stellar_rate_hz,
        planet_throughput=np.ones_like(wavelength_um),
        leak_fraction=np.ones_like(wavelength_um),
    )


def _resolve_integration_time_s(
    performance: dict[str, Any], override: float | None
) -> tuple[float, str]:
    if override is not None:
        return float(override), "performance.hdc integration_time_s param"
    for key in ("integration_time_for_target_snr_s", "integration_time_s"):
        value = performance.get(key)
        if value is None:
            continue
        candidate = float(value)
        if math.isfinite(candidate) and candidate > 0.0:
            source = f"performance.standard {key}"
            if key != "integration_time_for_target_snr_s":
                source += (
                    " (integration_time_for_target_snr_s was not finite and "
                    "positive: the broadband target is unreachable there)"
                )
            return candidate, source
    raise ValueError(
        "performance.hdc could not resolve an integration time: the upstream "
        "performance.standard payload has neither a finite positive "
        "integration_time_for_target_snr_s nor integration_time_s. Set the "
        "stage's integration_time_s param explicitly."
    )


def _dark_rate_hz_per_channel(
    performance: dict[str, Any],
    override: float | None,
    n_dark_ports: int,
) -> float:
    if override is not None:
        return float(override)
    return max(float(performance["dark_rate_hz_per_port"]), 0.0) * n_dark_ports


def _detection_budget_model(
    params: HdcPerformanceStage.Params,
    performance: dict[str, Any],
    n_dark_ports: int,
) -> str:
    if params.dark_count_hz_per_channel is not None:
        dark = (
            "dark = the configured dark_count_hz_per_channel, charged per "
            "channel"
        )
    else:
        dark = (
            f"dark = dark_rate_hz_per_port x {n_dark_ports} dark ports, charged "
            "PER CHANNEL. The assumption is one detector pixel set per "
            "spectrograph channel, so dark does not disperse and the band "
            "total grows with the channel count -- the same convention "
            "characterization.hdc_ccf takes from the AWG per-bin metadata. Set "
            "dark_count_hz_per_channel to override it"
        )
    return (
        "Dark-port AWG budget: every optical rate (planet, leak, sky) is "
        f"multiplied by awg_throughput={float(params.awg_throughput):.4g} for "
        "the extra dispersal loss the broadband budget does not carry; "
        "sky = sky_rate_hz_per_port x "
        f"{n_dark_ports} dark ports, spread as a flat density and channelized "
        "like the sources so it forfeits the same uncovered-band share; "
        f"{dark}. Stellar band total "
        f"{float(performance['stellar_rate_hz_total']):.6g} Hz."
    )


def _leak_rate_hz_per_anchor(
    performance: dict[str, Any],
    separations_mas: NDArray[np.float64],
    awg_throughput: float,
) -> NDArray[np.float64]:
    """Detected dark-port leak per anchor, preferring the per-separation curve.

    ``performance.standard`` publishes ``contrast_curve_leakage_rate_hz`` when
    it had a per-separation null curve to work from; that is strictly better
    than the single scalar, so use it when the grid lines up.
    """

    scalar = max(float(performance["leakage_rate_hz"]), 0.0)
    per_sep = performance.get("contrast_curve_leakage_rate_hz")
    if per_sep is not None:
        values = np.asarray(per_sep, dtype=np.float64)
        if values.shape == separations_mas.shape:
            return np.clip(values, 0.0, None) * awg_throughput
    return np.full(separations_mas.size, scalar * awg_throughput, dtype=np.float64)


def _sky_rate_hz_per_anchor(
    performance: dict[str, Any],
    separations_mas: NDArray[np.float64],
    n_dark_ports: int,
) -> NDArray[np.float64]:
    scalar = max(float(performance["sky_rate_hz_per_port"]), 0.0) * n_dark_ports
    per_sep = performance.get("contrast_curve_sky_rate_hz_per_port")
    if per_sep is not None:
        values = np.asarray(per_sep, dtype=np.float64)
        if values.shape == separations_mas.shape:
            return np.clip(values, 0.0, None) * n_dark_ports
    return np.full(separations_mas.size, scalar, dtype=np.float64)


def _broadband_contrast(
    performance: dict[str, Any], separations_mas: NDArray[np.float64]
) -> NDArray[np.float64]:
    """The broadband 5-sigma curve resampled onto the anchor grid."""

    curve_mas = np.asarray(
        performance.get("contrast_curve_mas", ()), dtype=np.float64
    )
    curve_value = np.asarray(
        performance.get("contrast_curve_value", ()), dtype=np.float64
    )
    if curve_mas.size == 0 or curve_mas.shape != curve_value.shape:
        return np.full(separations_mas.size, np.nan, dtype=np.float64)
    order = np.argsort(curve_mas)
    return np.interp(
        separations_mas, curve_mas[order], curve_value[order]
    ).astype(np.float64, copy=False)


def _participation_ratio(weights: NDArray[np.float64]) -> float:
    """``(sum |w|)^2 / sum w^2`` of the frozen matched-filter weights."""

    values = np.asarray(weights, dtype=np.float64)
    square_sum = float(np.sum(values * values))
    if square_sum <= 0.0:
        return 0.0
    absolute_sum = float(np.sum(np.abs(values)))
    return absolute_sum * absolute_sum / square_sum


def _refine_contrast(
    contrast: float,
    *,
    planet_unit_rate_hz: NDArray[np.float64],
    channels: ChannelSpectra,
    leak: NDArray[np.float64],
    sky: NDArray[np.float64],
    dark: NDArray[np.float64],
    target_snr: float,
    solver_kwargs: dict[str, Any],
) -> tuple[float, float, bool]:
    """Evaluate the achieved SNR at ``contrast`` and correct it onto target.

    The closed-form solver freezes its weights at zero contrast, which is
    generically first-order optimistic once systematics are active. One
    `matched_filter_ccf` evaluation at the returned contrast says how far off
    it landed; if the shortfall exceeds `_REFINEMENT_TOLERANCE` the contrast is
    corrected and re-evaluated.

    The correction is a log-log secant rather than a fixed power of
    ``target/refined``. ``SNR(c) = c A t / sqrt(t B2 + c t C2 + V_sys)`` has a
    local log-log slope that runs between 1 and 1/2: it is **linear in c**
    while the leak dominates the variance -- which is where a contrast curve
    actually lives -- and only becomes ``sqrt(c)`` once the planet's own shot
    noise takes over. Assuming the ``sqrt`` regime unconditionally, i.e.
    scaling by ``(target/refined)**2``, overshoots badly in the linear one:
    starting from half the solved contrast on a leak-dominated probe it lands
    at SNR 9.97 against a target of 5. So the first step uses the linear guess,
    and if that misses, the measured slope from the two evaluations gives the
    third. At most three evaluations, and it converges in either regime.

    Returns ``(contrast, refined_snr, corrected)``.
    """

    def achieved(value: float) -> float:
        budget = matched_filter_ccf(
            ChannelSpectra(
                channel_centers_um=channels.channel_centers_um,
                channel_edges_um=channels.channel_edges_um,
                planet_rate_hz=planet_unit_rate_hz * value,
                leak_rate_hz=leak,
                sky_rate_hz=sky,
                dark_rate_hz=dark,
            ),
            integration_time_s=solver_kwargs["integration_time_s"],
            eps_systematic=solver_kwargs["eps_systematic"],
            servo_bw_hz=solver_kwargs["servo_bw_hz"],
            with_fringe_tracking=solver_kwargs["with_fringe_tracking"],
            continuum_filter_width_channels=solver_kwargs[
                "continuum_filter_width_channels"
            ],
            telluric_transmission_per_channel=solver_kwargs[
                "telluric_transmission_per_channel"
            ],
            telluric_mask_threshold=solver_kwargs["telluric_mask_threshold"],
        )
        return float(budget.ccf_snr)

    def usable(value: float) -> bool:
        return math.isfinite(value) and value > 0.0

    snr = achieved(contrast)
    if not usable(snr):
        return contrast, snr, False
    if abs(snr / target_snr - 1.0) <= _REFINEMENT_TOLERANCE:
        return contrast, snr, False

    # Step 1: the leak-dominated (linear) guess.
    next_contrast = contrast * (target_snr / snr)
    next_snr = achieved(next_contrast)
    if not usable(next_snr):
        return contrast, snr, False
    if abs(next_snr / target_snr - 1.0) <= _REFINEMENT_TOLERANCE:
        return next_contrast, next_snr, True

    # Step 2: the measured log-log slope closes the remaining gap.
    log_contrast_step = math.log(next_contrast / contrast)
    log_snr_step = math.log(next_snr / snr)
    if abs(log_contrast_step) > 0.0 and abs(log_snr_step) > 0.0:
        slope = log_snr_step / log_contrast_step
        if math.isfinite(slope) and slope > 0.0:
            final_contrast = next_contrast * (target_snr / next_snr) ** (
                1.0 / slope
            )
            final_snr = achieved(final_contrast)
            if usable(final_snr) and abs(final_snr - target_snr) < abs(
                next_snr - target_snr
            ):
                return final_contrast, final_snr, True
    return next_contrast, next_snr, True
