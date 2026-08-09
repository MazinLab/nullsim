"""The single acceptance predicate for a ``characterization.spectro`` solution.

Three consumers have to agree on whether a retuned spectro solution is usable:
``characterization.hdc_ccf`` (which must not publish plausible-looking CCF
numbers for a solution the pipeline rejected), the ``characterization_summary``
table (which masks its SNR and rate columns), and the ``hdc_ccf`` plot (which
suppresses its figure). Three copies of the rule would drift, and a drift here
is invisible: every copy still produces a number, just a different one. So the
rule lives here once and every consumer calls it.

Acceptance is *optimizer* convergence, not measurement feasibility: the
optimizer reported success and, when a null constraint was enforced, the
solution sits inside the constraint tolerance. A converged solution can still
be an undetectable one -- that is what the SNR columns are for.
"""

from __future__ import annotations

from typing import Any, NamedTuple

__all__ = [
    "REASON_NULL_CONSTRAINT_VIOLATED",
    "REASON_OPTIMIZER_FAILED",
    "SpectroAcceptance",
    "spectro_solution_acceptance",
    "spectro_solution_accepted",
]

# Machine-readable rejection codes. Published verbatim in the invalidity key of
# a rejected ``characterization.hdc_ccf`` payload, so consumers can branch on
# the code instead of parsing prose.
REASON_OPTIMIZER_FAILED = "spectro_optimizer_did_not_converge"
REASON_NULL_CONSTRAINT_VIOLATED = "spectro_null_constraint_violated"

# The constraint margin is allowed to sit this far below zero, as a fraction of
# the null floor, before the solution counts as constraint violating.
_CONSTRAINT_TOLERANCE_FRACTION = 0.01


class SpectroAcceptance(NamedTuple):
    """Whether a spectro payload is usable, and why not when it is not."""

    accepted: bool
    reason: str | None
    detail: str | None


def spectro_solution_acceptance(payload: dict[str, Any]) -> SpectroAcceptance:
    """Decide whether a ``characterization.spectro`` payload may be consumed.

    Missing keys are read as "no evidence of failure": payloads written before
    a given diagnostic existed must not be retro-rejected, because refusing to
    consume a solution we cannot prove bad is just as wrong as publishing one
    we know is bad.
    """

    success = payload.get("optimization_success")
    if success is not None and not bool(success):
        return SpectroAcceptance(
            accepted=False,
            reason=REASON_OPTIMIZER_FAILED,
            detail=(
                "the characterization.spectro optimizer reported "
                f"optimization_success={success!r}"
            ),
        )
    if not bool(payload.get("enforce_null_constraint", False)):
        return SpectroAcceptance(accepted=True, reason=None, detail=None)
    margin = payload.get("optimization_constraint_margin")
    if margin is None:
        return SpectroAcceptance(accepted=True, reason=None, detail=None)
    try:
        margin_value = float(margin)
    except (TypeError, ValueError):
        return SpectroAcceptance(accepted=True, reason=None, detail=None)
    null_floor = payload.get("null_floor")
    try:
        tolerance = (
            _CONSTRAINT_TOLERANCE_FRACTION * float(null_floor)
            if null_floor is not None
            else 0.0
        )
    except (TypeError, ValueError):
        tolerance = 0.0
    if margin_value >= -tolerance:
        return SpectroAcceptance(accepted=True, reason=None, detail=None)
    return SpectroAcceptance(
        accepted=False,
        reason=REASON_NULL_CONSTRAINT_VIOLATED,
        detail=(
            "the characterization.spectro retune violates its enforced null "
            f"constraint (optimization_constraint_margin={margin_value:.6g}, "
            f"tolerance {-tolerance:.6g})"
        ),
    )


def spectro_solution_accepted(payload: dict[str, Any]) -> bool:
    """``spectro_solution_acceptance(payload).accepted``, for boolean callers."""

    return spectro_solution_acceptance(payload).accepted
