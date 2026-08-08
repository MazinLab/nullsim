"""Detection-mode high dispersion coronagraphy contrast curves.

Where ``characterization.hdc_ccf`` scores a *known* planet on the spectro port,
this stage inverts the same matched filter on the *dark* ports: at every
separation anchor the sensitivity stage published, it asks how faint a planet
the observation would still detect at ``target_snr``. That inversion is
`nullsim.physics.hdc.ccf_contrast_at_target_snr`, and stringing its answers
across the anchors is a CCF contrast curve — the HDC counterpart of the
broadband curve ``performance.standard`` already publishes.

As in characterization mode, nothing about the nuller changes: the null depth,
the throughputs, and the leakage are the ones ``performance.standard`` and
``sensitivity.detection_curve`` published. The detected photon budget is not
quite theirs -- dispersing the dark ports costs an extra ``awg_throughput`` on
every optical rate, and detector dark is recharged per spectrograph channel
rather than per port -- but nothing in the nuller itself moves. HDC is
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
* The planet throughput is grey across the band -- the sensitivity stage
  publishes one value per separation -- so the planet spectrum is channelized
  once and scaled per anchor, which is exact because convolution and binning
  are linear. The NULL need not be grey: when the sensitivity stage published a
  chromatic residual surface, the per-wavelength null profile is reconstructed
  per anchor and those anchors are rechannelized individually. Grey is the
  fallback, not the model.

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
    _weighted_mean,
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
_DEFAULT_TARGET_SNR = 5.0


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
        # Both default to None so the stage can INHERIT the broadband curve's
        # own settings and keep the gain comparison apples to apples. An
        # explicit value that disagrees with the broadband metadata is an
        # error, not something to silently rescale around.
        integration_time_s: float | None = PydanticField(default=None, gt=0.0)
        target_snr: float | None = PydanticField(default=None, gt=0.0)
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
        # performance.standard records which sensitivity component it used.
        # Following that link is what keeps the throughputs, the null curve,
        # and the leakage rate describing the same solution; falling back to
        # recency could pair this stage's throughputs with a different run's
        # leak.
        linked = performance.get("upstream_sensitivity_component")
        linked = linked if isinstance(linked, str) and linked else None
        sensitivity_request = params.upstream_sensitivity
        if sensitivity_request is not None and linked is not None:
            if sensitivity_request != linked:
                # Overriding the link would take the broadband curve, the leak,
                # and the sky from the performance stage's solution while
                # taking the throughputs and the chromatic null from a
                # different one -- and comparison_model would still claim both
                # curves describe the same observation. There is no honest
                # reading of that, so refuse it.
                raise ValueError(
                    "performance.hdc upstream_sensitivity "
                    f"{sensitivity_request!r} conflicts with "
                    f"{performance_name!r}, which was computed against "
                    f"{linked!r}. Mixing them would pair one solution's "
                    "broadband curve, leakage, and sky with another's planet "
                    "throughputs and chromatic null while ccf_contrast_gain "
                    "claims the two curves describe the same observation. "
                    "Either drop upstream_sensitivity so the link is followed, "
                    "or select a performance.standard component that used "
                    f"{sensitivity_request!r}."
                )
            linked_source = (
                "performance.hdc upstream_sensitivity param (agrees with the "
                f"{performance_name} link)"
            )
        elif sensitivity_request is not None:
            linked_source = (
                "performance.hdc upstream_sensitivity param "
                f"({performance_name} publishes no sensitivity link)"
            )
        elif linked is not None:
            sensitivity_request = linked
            linked_source = (
                f"{performance_name} upstream_sensitivity_component link"
            )
        else:
            linked_source = "most recently added compatible component"
        sensitivity_name, sensitivity = _find_component(
            state,
            prefix=_SENSITIVITY_PREFIX,
            required=_REQUIRED_SENSITIVITY_KEYS,
            requested=sensitivity_request,
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

        broadband = _broadband_contrast(performance, separations_mas)
        basis = _resolve_comparison_basis(
            performance,
            integration_time_override=params.integration_time_s,
            target_snr_override=params.target_snr,
            has_broadband_curve=bool(np.any(np.isfinite(broadband))),
        )
        integration_time_s = basis.integration_time_s
        target_snr = basis.target_snr
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
        # Reference amplitudes: planet at unit contrast and unit throughput,
        # leak and sky at unit band total. Everything downstream scales these
        # by a pure number per anchor, so the sky reference is deliberately
        # UNITY rather than the actual rate -- scaling a zero scalar sky by a
        # per-anchor ratio would zero every anchor even when the per-anchor
        # curve is positive.
        reference_budget = _DetectionBudget(
            planet_rate_hz=stellar_rate_hz_total * float(params.awg_throughput),
            leak_rate_hz=1.0,
            sky_rate_hz=1.0,
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
            performance,
            separations_mas,
            n_dark_ports,
            float(params.awg_throughput),
        )
        (
            leak_shape_factor,
            leak_shape_model,
            leak_shape_notes,
        ) = _chromatic_leak_factors(
            sensitivity=sensitivity,
            performance=performance,
            separations_mas=separations_mas,
            grid_um=grid_um,
            leak_density=densities.leak_density,
        )

        solver_kwargs: dict[str, Any] = {
            "integration_time_s": integration_time_s,
            "target_snr": float(target_snr),
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
        residual = np.full(n_anchors, np.nan, dtype=np.float64)
        n_effective_lines = np.full(n_anchors, np.nan, dtype=np.float64)
        systematic_var = np.full(n_anchors, np.nan, dtype=np.float64)
        photon_var_zero = np.full(n_anchors, np.nan, dtype=np.float64)
        reasons: list[str | None] = [None] * n_anchors
        leak_channels = np.zeros((n_anchors, n_channels), dtype=np.float64)
        corrected = 0

        for index in range(n_anchors):
            planet_unit = reference_channels.planet_rate_hz * float(
                throughputs[index]
            )
            if leak_shape_factor is None:
                leak = reference_channels.leak_rate_hz * float(
                    leak_per_anchor[index]
                )
            else:
                # A chromatic residual makes the leak SHAPE anchor dependent,
                # so this anchor's leak has to be rechannelized rather than
                # scaled off the grey reference.
                leak = channel_builder.channelize_density(
                    densities.leak_density * leak_shape_factor[index],
                    resolving_power,
                ) * float(leak_per_anchor[index])
            leak_channels[index] = leak
            sky = reference_channels.sky_rate_hz * float(sky_per_anchor[index])
            dark = reference_channels.dark_rate_hz
            # The whole solve-and-refine sits inside the catch: the refinement
            # calls matched_filter_ccf, which validates its own inputs and can
            # raise ValueError too, and one bad anchor must not abort the curve.
            try:
                solution = ccf_contrast_at_target_snr(
                    planet_unit, leak, sky, dark, **solver_kwargs
                )
                value, snr, first_snr, did_correct = _refine_contrast(
                    solution,
                    planet_unit_rate_hz=planet_unit,
                    channels=reference_channels,
                    leak=leak,
                    sky=sky,
                    dark=dark,
                    target_snr=float(target_snr),
                    solver_kwargs=solver_kwargs,
                )
            except ValueError as error:
                # Reachable in ordinary sweeps: a low-throughput anchor whose
                # lines all land in telluric-masked channels has no contrast
                # that reaches the target. Record the anchor as a
                # non-detection instead of aborting the whole curve.
                reasons[index] = str(error)
                continue

            contrast[index] = value
            refined_snr[index] = snr
            # The ORIGINAL solver bias, measured at the first evaluation and
            # before any correction: that is what diagnoses the closed form.
            optimism[index] = (
                float(target_snr) / first_snr - 1.0
                if first_snr > 0.0
                else np.nan
            )
            residual[index] = (
                snr / float(target_snr) - 1.0 if snr > 0.0 else np.nan
            )
            corrected += int(did_correct)
            systematic_var[index] = float(solution.systematic_var_counts)
            photon_var_zero[index] = float(solution.photon_var_zero_contrast)
            n_effective_lines[index] = _participation_ratio(
                solution.template_weights
            )

        with np.errstate(divide="ignore", invalid="ignore"):
            gain = broadband / contrast

        component_name = unique_component_name(
            self.type_name,
            state.results.components,
            self.context.instance_id,
        )
        payload = {
            "separations_mas": separations_mas.astype(np.float64, copy=True),
            # Threshold-generic names. comparison_target_snr is the
            # authoritative threshold; the *_5sigma aliases below exist only
            # when it really is 5, so a consumer cannot read "5 sigma" off a
            # curve computed at some other target.
            "ccf_contrast_at_target": contrast.astype(np.float64, copy=True),
            "broadband_contrast_at_target": broadband.astype(
                np.float64, copy=True
            ),
            "ccf_contrast_gain": np.asarray(gain, dtype=np.float64),
            "target_snr": float(target_snr),
            "integration_time_s": float(integration_time_s),
            "integration_time_source": basis.source,
            "comparison_time_s": float(basis.integration_time_s),
            "comparison_target_snr": float(basis.target_snr),
            "comparison_model": basis.model,
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
            "n_effective_channels_per_anchor": n_effective_lines.astype(
                np.float64, copy=True
            ),
            # Deprecated alias, see CcfBudget.n_effective_lines.
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
            "post_correction_residual_per_anchor": residual.astype(
                np.float64, copy=True
            ),
            "corrected_anchor_count": int(corrected),
            "non_detection_reasons": tuple(reasons),
            "n_non_detections": int(sum(1 for r in reasons if r is not None)),
            "leak_rate_hz_per_anchor": leak_per_anchor.astype(
                np.float64, copy=True
            ),
            "sky_rate_hz_per_anchor": sky_per_anchor.astype(
                np.float64, copy=True
            ),
            "ccf_planet_rate_hz_per_channel_unit_contrast": (
                reference_channels.planet_rate_hz.astype(np.float64, copy=True)
            ),
            "ccf_leak_rate_hz_per_channel": leak_channels.astype(
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
            "leak_shape_model": leak_shape_model,
            "leak_shape_fallback_reasons": leak_shape_notes,
            "upstream_sensitivity_link_source": linked_source,
            "source_model": (
                "nullsim performance.hdc: "
                "nullsim.physics.hdc.ccf_contrast_at_target_snr inverted at "
                "each sensitivity.detection_curve separation anchor, over the "
                "spectral machinery shared with characterization.hdc_ccf."
            ),
        }
        if math.isclose(float(target_snr), 5.0, rel_tol=1.0e-9, abs_tol=0.0):
            # Back-compatible aliases, present ONLY at a genuine 5-sigma
            # threshold. Absent otherwise, so a consumer that keys on them
            # fails loudly instead of mislabelling the curve.
            payload["ccf_contrast_5sigma"] = contrast.astype(
                np.float64, copy=True
            )
            payload["broadband_contrast_5sigma"] = broadband.astype(
                np.float64, copy=True
            )
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
    "Detection-mode photon-weighted cross-correlation after Wang, Mawet, Ruane, "
    "Hu & Benneke 2017, AJ 153, 183. At each separation anchor the matched "
    "filter is inverted -- nullsim.physics.hdc.ccf_contrast_at_target_snr "
    "solves SNR(c) = target_snr in closed form -- to give the faintest planet "
    "the same observation still detects. This is an after-the-fact analysis "
    "gain at FIXED nuller performance: the null depth, planet throughputs, and "
    "leakage are the ones performance.standard and sensitivity.detection_curve "
    "published, and nothing here changes them. The detected photon budget does "
    "differ from theirs, because dispersing the dark ports is not free: every "
    "optical rate carries an extra awg_throughput, and detector dark is "
    "recharged per spectrograph channel rather than per port (see "
    "detection_budget_model). The closed form freezes the matched-filter "
    "weights at zero contrast, which makes it generically FIRST-ORDER BIASED "
    "once the systematic share of the variance is non-negligible; the SIGN of "
    "that bias is spectrum dependent -- the structural coefficient in the "
    "solver's expansion can go either way, and probes have come out both "
    "optimistic and pessimistic -- so the correction is two-sided. Every "
    "anchor is therefore refined: one matched_filter_ccf evaluation at the "
    "returned contrast gives the achieved SNR and the solver's own bias "
    "(solver_optimism_per_anchor = target/first_evaluation - 1, positive when "
    "the solver was optimistic and negative when it was pessimistic, always "
    "measured BEFORE any correction). The sequence is hard bounded: solve, "
    "evaluate once, and if the achieved SNR is off target by more than 2%, "
    "apply exactly ONE correction and evaluate once more -- never further. The "
    "correction is a Newton step in log-log space using the analytic slope "
    "d ln SNR / d ln c = 1 - (1/2)(c t C2)/(t B2 + c t C2 + V_sys), whose "
    "coefficients the solver already returns, so it costs no extra evaluation. "
    "It is not a fixed power of target/refined: SNR(c) is linear in c while "
    "the leak dominates the variance -- where a contrast curve lives -- and "
    "only becomes sqrt(c) once the planet's own shot noise takes over, so a "
    "fixed square exponent overshoots in the linear regime. If the final "
    "evaluation is STILL off target by more than 2%, the anchor is published "
    "as a NON-DETECTION rather than as a contrast: a curve point that does not "
    "actually reach the stated threshold is worse than a missing one. "
    "refined_snr_per_anchor and post_correction_residual_per_anchor report "
    "where the converged anchors landed; corrected_anchor_count records how "
    "many needed the step. In the fixtures exercised so far that count is "
    "zero -- the leak projection HDC suppresses leaves the solver almost "
    "unbiased -- but that is fixture-supported observation, not a general "
    "guarantee."
)

_GREY_LEAK_SHAPE_MODEL = (
    "GREY null. Leaked starlight is distributed with the stellar x telluric "
    "line shape only: the upstream sensitivity component published no "
    "residual_starlight_contrast_by_wavelength surface, so there is no "
    "chromatic null structure to fold in, and performance.standard publishes "
    "only a band-integrated leakage rate. Per-anchor amplitude comes from "
    "contrast_curve_leakage_rate_hz when the upstream stage published a "
    "per-separation curve, otherwise the scalar leakage_rate_hz. A chromatic "
    "null would change the leak's spectral structure and hence its projection "
    "onto the planet template; this fallback cannot represent that."
)

_CHROMATIC_LEAK_SHAPE_MODEL = (
    "CHROMATIC null. Leaked starlight is distributed with the stellar x "
    "telluric line shape TIMES the per-anchor row of the upstream "
    "sensitivity.detection_curve residual_starlight_contrast_by_wavelength "
    "surface, interpolated onto the analysis grid and normalized to unit band "
    "mean under the grey leak density, so the wavelength structure enters "
    "while the published band total is preserved. Per-anchor amplitude comes "
    "from contrast_curve_leakage_rate_hz when available, otherwise the scalar "
    "leakage_rate_hz. Because the leak shape is anchor dependent here, each "
    "anchor is rechannelized rather than scaled off a grey reference."
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


class _ComparisonBasis:
    """The integration time and target SNR the gain comparison is made at.

    ``ccf_contrast_gain`` divides two contrast curves, so they have to describe
    the same observation. They easily do not: ``performance.standard`` stores
    its broadband curve at ``contrast_curve_reference_time_s`` -- which is its
    own param, defaulting to the detector integration time, NOT the
    ``integration_time_for_target_snr_s`` it also publishes -- and at its own
    ``target_snr``. Taking the target-SNR time here (the obvious reading) and
    an independent target would divide curves computed at different exposures
    against different thresholds, and the ratio would look like a physical gain.

    So both are INHERITED from the broadband curve's published metadata by
    default. An explicit override that disagrees is refused rather than
    reconciled: rescaling the stored broadband curve would mean recomputing it,
    which is the upstream stage's job, not this one's.
    """

    __slots__ = ("integration_time_s", "target_snr", "source", "model")

    def __init__(
        self,
        *,
        integration_time_s: float,
        target_snr: float,
        source: str,
        model: str,
    ) -> None:
        self.integration_time_s = integration_time_s
        self.target_snr = target_snr
        self.source = source
        self.model = model


def _resolve_comparison_basis(
    performance: dict[str, Any],
    *,
    integration_time_override: float | None,
    target_snr_override: float | None,
    has_broadband_curve: bool,
) -> _ComparisonBasis:
    broadband_time = _positive_finite(
        performance.get("contrast_curve_reference_time_s")
    )
    broadband_target = _positive_finite(performance.get("target_snr"))

    if has_broadband_curve and broadband_time is not None:
        _refuse_mismatch(
            "integration_time_s",
            integration_time_override,
            broadband_time,
            "contrast_curve_reference_time_s",
        )
        integration_time_s = broadband_time
        time_source = "performance.standard contrast_curve_reference_time_s"
    else:
        integration_time_s, time_source = _fallback_integration_time_s(
            performance, integration_time_override
        )

    if has_broadband_curve and broadband_target is not None:
        _refuse_mismatch(
            "target_snr", target_snr_override, broadband_target, "target_snr"
        )
        target_snr = broadband_target
        target_source = "performance.standard target_snr"
    elif target_snr_override is not None:
        target_snr = float(target_snr_override)
        target_source = "performance.hdc target_snr param"
    elif broadband_target is not None:
        target_snr = broadband_target
        target_source = "performance.standard target_snr"
    else:
        target_snr = _DEFAULT_TARGET_SNR
        target_source = f"performance.hdc default ({_DEFAULT_TARGET_SNR})"

    if has_broadband_curve:
        model = (
            "ccf_contrast_gain = broadband_contrast_at_target / "
            "ccf_contrast_at_target (the *_contrast_5sigma aliases carry the "
            "same arrays, but only when comparison_target_snr is exactly 5), "
            "and both curves are guaranteed to describe "
            f"the SAME observation: integration time {integration_time_s:.6g} s "
            f"and target SNR {target_snr:.6g}, inherited from the broadband "
            f"curve's own metadata ({time_source}; {target_source}). The "
            "broadband curve is stored at contrast_curve_reference_time_s, "
            "which is NOT the same as the integration_time_for_target_snr_s "
            "the same payload publishes, so inheriting is what makes the ratio "
            "meaningful. An explicit integration_time_s or target_snr that "
            "disagrees with the broadband metadata is refused rather than "
            "reconciled."
        )
    else:
        model = (
            "No broadband curve was published upstream (empty "
            "contrast_curve_mas), so broadband_contrast_at_target and "
            "ccf_contrast_gain are NaN and there is nothing to match. The CCF "
            f"curve stands alone at integration time {integration_time_s:.6g} s "
            f"and target SNR {target_snr:.6g} ({time_source}; {target_source})."
        )
    return _ComparisonBasis(
        integration_time_s=integration_time_s,
        target_snr=target_snr,
        source=f"{time_source}; {target_source}",
        model=model,
    )


def _refuse_mismatch(
    param_name: str,
    override: float | None,
    broadband_value: float,
    broadband_key: str,
) -> None:
    if override is None:
        return
    if math.isclose(float(override), broadband_value, rel_tol=1.0e-9, abs_tol=0.0):
        return
    raise ValueError(
        f"performance.hdc {param_name}={float(override)!r} disagrees with the "
        f"broadband curve's {broadband_key}={broadband_value!r}. "
        "ccf_contrast_gain divides the two curves, so they must describe the "
        "same observation. Either drop the override so this stage inherits the "
        f"broadband setting, or set performance.standard's {broadband_key} to "
        "match. This stage will not rescale or recompute the stored broadband "
        "curve."
    )


def _finite_non_negative(value: Any) -> float | None:
    """Coerce to a finite, non-negative float, or None if it is neither."""

    if value is None:
        return None
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    return candidate if math.isfinite(candidate) and candidate >= 0.0 else None


def _positive_finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    return candidate if math.isfinite(candidate) and candidate > 0.0 else None


def _fallback_integration_time_s(
    performance: dict[str, Any], override: float | None
) -> tuple[float, str]:
    """Integration time when there is no broadband curve to match."""

    if override is not None:
        return float(override), "performance.hdc integration_time_s param"
    for key in (
        "contrast_curve_reference_time_s",
        "integration_time_for_target_snr_s",
        "integration_time_s",
    ):
        candidate = _positive_finite(performance.get(key))
        if candidate is not None:
            return candidate, f"performance.standard {key}"
    raise ValueError(
        "performance.hdc could not resolve an integration time: the upstream "
        "performance.standard payload has no finite positive "
        "contrast_curve_reference_time_s, integration_time_for_target_snr_s, "
        "or integration_time_s. Set the stage's integration_time_s param."
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


def _performance_curve_on_anchors(
    performance: dict[str, Any],
    key: str,
    separations_mas: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    """Resample a performance per-separation array onto the anchor grid.

    ``performance.standard`` publishes its per-separation curves on its OWN
    ``contrast_curve_mas`` grid. Accepting them by array length instead would
    misassign by index whenever the two grids have the same number of points
    but different coordinates -- which is exactly what happens when the
    performance stage and this stage read different sensitivity components.
    """

    values = performance.get(key)
    if values is None:
        return None
    curve = np.asarray(values, dtype=np.float64)
    coordinates = np.asarray(
        performance.get("contrast_curve_mas", ()), dtype=np.float64
    )
    if curve.ndim != 1 or curve.size == 0:
        return None
    if coordinates.shape != curve.shape:
        return None
    order = np.argsort(coordinates)
    return np.interp(
        separations_mas, coordinates[order], curve[order]
    ).astype(np.float64, copy=False)


def _leak_rate_hz_per_anchor(
    performance: dict[str, Any],
    separations_mas: NDArray[np.float64],
    awg_throughput: float,
) -> NDArray[np.float64]:
    """Detected dark-port leak per anchor, preferring the per-separation curve.

    ``performance.standard`` publishes ``contrast_curve_leakage_rate_hz`` when
    it had a per-separation null curve to work from; that is strictly better
    than the single scalar, so use it, resampled onto the anchor grid by its
    own separation coordinates.
    """

    per_sep = _performance_curve_on_anchors(
        performance, "contrast_curve_leakage_rate_hz", separations_mas
    )
    if per_sep is not None:
        return np.clip(per_sep, 0.0, None) * awg_throughput
    scalar = max(float(performance["leakage_rate_hz"]), 0.0)
    return np.full(separations_mas.size, scalar * awg_throughput, dtype=np.float64)


def _sky_rate_hz_per_anchor(
    performance: dict[str, Any],
    separations_mas: NDArray[np.float64],
    n_dark_ports: int,
    awg_throughput: float,
) -> NDArray[np.float64]:
    """Detected dark-port sky per anchor.

    ``awg_throughput`` applies to sky exactly as it does to the planet and the
    leak: ``awg_spectrum`` multiplies every optical rate by the disperser
    throughput before binning, and sky is an optical rate.
    """

    per_sep = _performance_curve_on_anchors(
        performance, "contrast_curve_sky_rate_hz_per_port", separations_mas
    )
    if per_sep is not None:
        return np.clip(per_sep, 0.0, None) * n_dark_ports * awg_throughput
    scalar = max(float(performance["sky_rate_hz_per_port"]), 0.0) * n_dark_ports
    return np.full(
        separations_mas.size, scalar * awg_throughput, dtype=np.float64
    )


def _chromatic_leak_factors(
    *,
    sensitivity: dict[str, Any],
    performance: dict[str, Any],
    separations_mas: NDArray[np.float64],
    grid_um: NDArray[np.float64],
    leak_density: NDArray[np.float64],
) -> tuple[NDArray[np.float64] | None, str, tuple[str | None, ...]]:
    """Per-anchor multiplicative leak shape from the chromatic null profile.

    The residual-starlight surface is **not** the shape of the whole leak, and
    treating it as one is badly wrong in the common case. Sensitivity combines
    it with a grey base -- ``_apply_residual_starlight_to_effective_null``,
    modes ``max`` / ``add`` / ``replace``, published as
    ``residual_starlight_combine_mode`` -- and ``performance.standard`` then
    raises further grey floors over the result
    (``_contrast_curve_leakage_rates_from_null_curve`` takes elementwise maxima
    with ``null_depth_residual`` and ``calibration_floor``). With a grey 1e-6
    base and a 1e-10 residual under ``max``, the true null is grey everywhere;
    normalizing the raw residual instead would make the entire leak chromatic
    and invent structure that is not in the budget.

    So the per-wavelength null is reconstructed the way the upstream stages
    built it: ``combine(grey_base, residual(lambda))`` with the published mode,
    then the performance grey floors, and only THEN normalized to unit band
    mean (weighted by the grey leak density) to become the leak shape. The
    grey base is ``effective_null_before_residual_starlight``, the scalar the
    sensitivity stage actually passed to the combine -- not
    ``effective_null_with_disk_by_separation``, which is the POST-combine curve
    and would double count the residual.

    Returns ``(factors_or_None, leak_shape_model, per_anchor_fallback_notes)``.
    ``None`` means every anchor is grey.
    """

    notes: list[str | None] = [None] * separations_mas.size

    surface = sensitivity.get("residual_starlight_contrast_by_wavelength")
    wavelengths_um = sensitivity.get("residual_starlight_wavelengths_um")
    combine_mode = sensitivity.get("residual_starlight_combine_mode")
    # Non-negative, not positive: a ZERO grey base is legal upstream for the
    # max and add modes, and it is the case where the chromatic residual
    # matters most -- max(0, r) = r and 0 + r = r both leave the residual
    # standing. Requiring positivity here silently greyed exactly that.
    grey_base = _finite_non_negative(
        sensitivity.get("effective_null_before_residual_starlight")
    )

    if surface is None or wavelengths_um is None:
        return None, _GREY_LEAK_SHAPE_MODEL, tuple(notes)
    if combine_mode not in ("max", "add", "replace"):
        return (
            None,
            _GREY_LEAK_SHAPE_MODEL
            + " A residual surface was published but "
            f"residual_starlight_combine_mode={combine_mode!r} is not one of "
            "max/add/replace, so the profile cannot be reconstructed and grey "
            "is used.",
            tuple(notes),
        )
    if grey_base is None and combine_mode != "replace":
        return (
            None,
            _GREY_LEAK_SHAPE_MODEL
            + " A residual surface was published but "
            "effective_null_before_residual_starlight is unavailable, so the "
            f"{combine_mode!r} combine has no grey base and grey is used.",
            tuple(notes),
        )

    values = np.asarray(surface, dtype=np.float64)
    axis_um = np.asarray(wavelengths_um, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] != separations_mas.size
        or values.shape[1] != axis_um.size
        or axis_um.size < 2
        or not np.all(np.isfinite(values))
    ):
        return None, _GREY_LEAK_SHAPE_MODEL, tuple(notes)

    order = np.argsort(axis_um)
    axis_um = axis_um[order]
    values = values[:, order]
    floor = _grey_null_floor(performance)
    base = 0.0 if grey_base is None else float(grey_base)

    factors = np.ones((separations_mas.size, grid_um.size), dtype=np.float64)
    any_chromatic = False
    for index in range(separations_mas.size):
        residual = np.clip(values[index], 0.0, None)
        profile = np.interp(grid_um, axis_um, residual)
        if combine_mode == "max":
            profile = np.maximum(profile, base)
        elif combine_mode == "add":
            profile = profile + base
        if floor > 0.0:
            profile = np.maximum(profile, floor)
        mean = _safe_weighted_mean(profile, leak_density, grid_um)
        if mean is None:
            # A valid all-zero residual row under ``replace`` leaves no
            # profile to normalize. Grey for this anchor; never abort.
            notes[index] = (
                "chromatic leak shape degenerate at this anchor (combined "
                "null profile has no positive band mean); fell back to grey"
            )
            continue
        factors[index] = profile / mean
        any_chromatic = True

    if not any_chromatic:
        return (
            None,
            _GREY_LEAK_SHAPE_MODEL
            + " A residual surface was published but every anchor's combined "
            "null profile was degenerate, so grey is used throughout.",
            tuple(notes),
        )
    return factors, _chromatic_leak_shape_model(combine_mode, base, floor), tuple(
        notes
    )


def _grey_null_floor(performance: dict[str, Any]) -> float:
    """The grey floors performance.standard raises over the null curve.

    Mirrors ``_contrast_curve_leakage_rates_from_null_curve``: the null curve
    is maxed against ``null_depth_residual`` and ``calibration_floor`` before
    becoming a leakage rate. Both are grey, so they flatten the profile.
    """

    floor = 0.0
    for key in ("null_depth_residual", "calibration_floor"):
        value = _positive_finite(performance.get(key))
        if value is not None:
            floor = max(floor, value)
    return floor


def _safe_weighted_mean(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
    grid_um: NDArray[np.float64],
) -> float | None:
    """`_weighted_mean` that reports degeneracy instead of raising."""

    try:
        return _weighted_mean(values, weights, grid_um)
    except ValueError:
        return None


def _chromatic_leak_shape_model(
    combine_mode: str, grey_base: float, floor: float
) -> str:
    return (
        "CHROMATIC null. The per-wavelength null profile is reconstructed the "
        "way the upstream stages built it, not taken raw from the residual "
        f"surface: combine_mode={combine_mode!r} applied to a grey base of "
        f"{grey_base:.6g} (sensitivity effective_null_before_residual_starlight, "
        "the scalar that stage actually passed to the combine) and the "
        "per-anchor row of residual_starlight_contrast_by_wavelength, then "
        f"raised over a grey floor of {floor:.6g} "
        "(max of performance.standard's null_depth_residual and "
        "calibration_floor, mirroring "
        "_contrast_curve_leakage_rates_from_null_curve). That combined "
        "profile is normalized to unit band mean under the grey leak density, "
        "so the wavelength structure enters while the published band total is "
        "preserved. When the base dominates the residual the profile comes out "
        "flat and the leak is effectively grey, which is the correct answer. "
        "Anchors whose combined profile is degenerate fall back to grey "
        "individually; see leak_shape_fallback_reasons. Because the leak shape "
        "is anchor dependent, each anchor is rechannelized rather than scaled "
        "off a grey reference."
    )


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
    """``(sum |w|)^2 / sum w^2`` of the frozen filter weights.

    An effective *channel* count, not a line count: at any oversampling one
    spectral line occupies several channels.
    """

    values = np.asarray(weights, dtype=np.float64)
    square_sum = float(np.sum(values * values))
    if square_sum <= 0.0:
        return 0.0
    absolute_sum = float(np.sum(np.abs(values)))
    return absolute_sum * absolute_sum / square_sum


def _refine_contrast(
    solution: Any,
    *,
    planet_unit_rate_hz: NDArray[np.float64],
    channels: ChannelSpectra,
    leak: NDArray[np.float64],
    sky: NDArray[np.float64],
    dark: NDArray[np.float64],
    target_snr: float,
    solver_kwargs: dict[str, Any],
) -> tuple[float, float, float, bool]:
    """Check the solved contrast against the true statistic and correct once.

    The closed-form solver freezes its matched-filter weights at zero contrast,
    which makes it first-order biased once systematics are active. The sign is
    spectrum dependent -- the structural coefficient in the solver's expansion
    can go either way, and probes have come out both optimistic and
    pessimistic -- so the correction is two-sided, not a one-way top-up.
    The contract here is a hard-bounded three-step sequence, never more:

    1. one `matched_filter_ccf` evaluation at the solved contrast;
    2. if the achieved SNR is off target by more than
       `_REFINEMENT_TOLERANCE`, ONE correction;
    3. one final evaluation.

    The correction is a Newton step in log-log space. ``SNR(c) = c A t /
    sqrt(t B2 + c t C2 + V_sys)`` has the analytic local slope

    ``d ln SNR / d ln c = 1 - (1/2) (c t C2) / (t B2 + c t C2 + V_sys)``,

    which runs from 1 while the leak dominates the variance -- where a contrast
    curve actually lives -- to 1/2 once the planet's own shot noise takes over.
    All four coefficients come back from the solver, so the slope costs no
    extra evaluation. Correcting with a *fixed* exponent instead is what fails:
    assuming the ``sqrt`` regime unconditionally, i.e. scaling by
    ``(target/refined)**2``, overshoots badly in the linear one -- starting
    from half the solved contrast on a leak-dominated probe it lands at SNR
    9.97 against a target of 5.

    Returns ``(contrast, refined_snr, first_snr, corrected)``. ``first_snr`` is
    the pre-correction evaluation, which is what diagnoses the solver's own
    bias.
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

    contrast = float(solution.contrast_at_target_snr)
    first_snr = achieved(contrast)
    if not (math.isfinite(first_snr) and first_snr > 0.0):
        raise ValueError(
            "refinement did not converge (the matched filter returned SNR "
            f"{first_snr!r} at the solved contrast {contrast!r})"
        )
    if abs(first_snr / target_snr - 1.0) <= _REFINEMENT_TOLERANCE:
        return contrast, first_snr, first_snr, False

    # Off target: exactly one correction step is allowed.
    best_contrast, best_snr, corrected = contrast, first_snr, False
    slope = _log_log_slope(solution, solver_kwargs["integration_time_s"], contrast)
    corrected_contrast = contrast * (target_snr / first_snr) ** (1.0 / slope)
    if math.isfinite(corrected_contrast) and corrected_contrast > 0.0:
        corrected_snr = achieved(corrected_contrast)
        if (
            math.isfinite(corrected_snr)
            and corrected_snr > 0.0
            and abs(corrected_snr - target_snr) < abs(first_snr - target_snr)
        ):
            best_contrast, best_snr, corrected = (
                corrected_contrast,
                corrected_snr,
                True,
            )

    # Whatever the step achieved, the published point has to actually reach the
    # threshold. Merely improving on a bad start is not enough.
    if abs(best_snr - target_snr) / target_snr > _REFINEMENT_TOLERANCE:
        stage = (
            "after one correction step"
            if corrected
            else "and the single correction step did not help"
        )
        raise ValueError(
            f"refinement did not converge (SNR {best_snr:.6g} vs target "
            f"{target_snr:.6g}, {abs(best_snr / target_snr - 1.0):.1%} off, "
            f"{stage})"
        )
    return best_contrast, best_snr, first_snr, corrected


def _log_log_slope(solution: Any, integration_time_s: float, contrast: float) -> float:
    """``d ln SNR / d ln c`` from the solver's own coefficients.

    Clamped to ``[0.5, 1]``, the analytic range of the expression: outside it
    the inputs are degenerate and a wilder exponent would only make the single
    correction step worse.
    """

    photon = float(solution.photon_var_zero_contrast) * integration_time_s
    planet = (
        float(solution.planet_selfnoise_coefficient) * integration_time_s * contrast
    )
    systematic = float(solution.systematic_var_counts)
    total = photon + planet + systematic
    if not math.isfinite(total) or total <= 0.0:
        return 1.0
    slope = 1.0 - 0.5 * (planet / total)
    if not math.isfinite(slope):
        return 1.0
    return min(max(slope, 0.5), 1.0)
