"""Per-stage pipeline debug report (multi-page portrait PDF).

Walks the executed pipeline stage by stage, rendering a portrait letter
page per stage that surfaces every load-bearing quantity the pipeline
produced. Closes with three rollup pages: spectral loss budget, OPD
budget, detector & performance.

Design: editorial scientific report — serif body (matching nullsim's
publication style), sienna accent, warm grayscale, header strip with
section eyebrow and page number on every page. Each panel carries an
annotated headline number so silent regressions are visible at a glance.
"""

from __future__ import annotations

import json
import math
import textwrap
from collections.abc import Callable, Iterable
from contextlib import suppress
from pathlib import Path
from typing import Any

import matplotlib  # noqa: F401  # backend choice managed by styles module
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, LogLocator

from nullsim.outputs import RunCollection, RunRecord
from nullsim.outputs.styles import _ensure_headless_backend  # noqa: F401  # side-effect
from nullsim.physics.fringe_tracking import piston_psd_knee_hz
from nullsim.pipeline import register_plot

# ---------------------------------------------------------------------------
# Page geometry & palette
# ---------------------------------------------------------------------------

_PAGE_W, _PAGE_H = 8.5, 11.0          # portrait letter, inches
_MARGIN_L = 0.55
_MARGIN_R = 0.55
_HEADER_Y = 10.55                      # eyebrow / page-number row, inches from bottom
_RULE_Y = 10.40                        # horizontal rule under header
_TITLE_Y = 10.10                       # big title baseline
_CONTENT_TOP = 9.55                    # top of usable content area
_CONTENT_BOTTOM = 0.55                 # bottom of usable content area
_FOOTER_Y = 0.32                       # footer text baseline

_INK = "#1a1a24"
_PAPER = "#ffffff"
_ACCENT = "#a23b2e"      # burnt sienna
_COOL = "#1e5060"        # deep teal
_MUTE = "#7a6e60"        # warm gray
_RULE = "#cfc8b8"        # warm tan
_STRIPE = "#f5f1e8"      # cream
_BRIGHT_PORT = "#c08229" # warm gold
_DARK_PORT = "#244a6b"   # deep blue
_GOOD = "#2a7d52"
_WARN = "#b87333"
_BAD = "#a23b2e"

_RC_DEBUG: dict[str, object] = {
    "axes.edgecolor": _INK,
    "axes.linewidth": 0.7,
    "axes.labelcolor": _INK,
    "axes.titlecolor": _INK,
    "text.color": _INK,
    "text.parse_math": False,          # underscores render literally everywhere
    "axes.unicode_minus": False,       # cmr10 lacks U+2212; use ASCII hyphen
    "axes.formatter.use_mathtext": False,  # otherwise ticks become $\mathdefault{...}$
    "xtick.color": _INK,
    "ytick.color": _INK,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 10.5,
    "axes.titleweight": "regular",
    "axes.titlepad": 6.0,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "grid.color": _RULE,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.7,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "standard",
    "savefig.pad_inches": 0.0,
    # The debug report uses DejaVu Serif as its body font rather than cmr10,
    # because cmr10's _ character is a dot-above (U+02D9), which mangles
    # identifiers like "ao_pyramid". DejaVu Serif renders ASCII underscores
    # correctly and carries the Greek µ and minus glyphs cmr10 lacks.
    # cmr10 is kept as the fallback for math glyphs we might still want.
    "font.serif": ["DejaVu Serif", "cmr10", "STIXGeneral"],
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.monospace": ["DejaVu Sans Mono", "Courier New"],
}

# "µm" — uses U+00B5 (micro sign) which falls back cleanly to DejaVu Serif
# when cmr10 lacks the glyph. Used in axis labels in place of $\mu$m, which
# would require math parsing that we disable globally above.
_UM = "µm"

_IMPORTANT_RESULT_TOKENS = (
    "null",
    "snr",
    "throughput",
    "rate_hz",
    "background",
    "sky",
    "saturated",
    "resolving_power",
    "integration_time",
    "target",
    "jitter",
    "strehl",
    "opd",
    "coupling_source",
    "band",
    "site_profile",
    "source",
)

_COMPONENT_LABEL_ALIASES = {
    "fringe_tracking.closed_loop_residual": "fringe_tracking_residual",
}


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _fx(x_in: float) -> float:
    """Inches → figure x-coord."""
    return x_in / _PAGE_W


def _fy(y_in: float) -> float:
    """Inches → figure y-coord."""
    return y_in / _PAGE_H


def _new_page() -> plt.Figure:
    fig = plt.figure(figsize=(_PAGE_W, _PAGE_H), dpi=150)
    fig.patch.set_facecolor(_PAPER)
    return fig


def _add_axes_in(fig: plt.Figure, x0: float, y0: float, x1: float, y1: float) -> plt.Axes:
    """Add axes positioned in inches (x0,y0)→(x1,y1)."""
    return fig.add_axes([
        _fx(x0), _fy(y0), _fx(x1 - x0), _fy(y1 - y0),
    ])


def _inset_for_ylabels(ax: plt.Axes, inches: float) -> None:
    """Shift an axes' left edge right by `inches` to make room for long
    y-tick labels (used by horizontal bar charts whose category names would
    otherwise spill past the page margin)."""
    pos = ax.get_position()
    dx = inches / _PAGE_W
    new_x0 = pos.x0 + dx
    new_w = max(0.1, pos.width - dx)
    ax.set_position([new_x0, pos.y0, new_w, pos.height])


def _log_formatter() -> FuncFormatter:
    """Plain-text log-axis labels (e.g. "10⁹"), since the default mathtext
    formatter would render literally as ``$10^{9}$`` under
    ``text.parse_math=False``."""
    superscript = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")

    def _fmt(x: float, _pos: int) -> str:
        if x <= 0:
            return ""
        exp = int(round(np.log10(x)))
        if abs(10.0 ** exp - x) / x > 0.01:        # not a pure power of 10
            return ""
        return f"10{str(exp).translate(superscript)}"

    return FuncFormatter(_fmt)


def _ellipsize(text: Any, max_chars: int) -> str:
    """Return a single-line string clipped to fit a known report column."""

    value = str(text)
    if len(value) <= max_chars:
        return value
    if max_chars <= 1:
        return value[:max_chars]
    return value[: max_chars - 1] + "…"


def _component_label(name: str, *, max_chars: int = 32) -> str:
    """Readable chart label for long stage/component identifiers."""

    if name in _COMPONENT_LABEL_ALIASES:
        return _COMPONENT_LABEL_ALIASES[name]
    short = name
    for prefix in (
        "pipeline.stages.",
        "fringe_tracking.closed_loop_",
        "background.sky_surface_brightness_",
    ):
        if short.startswith(prefix):
            short = short[len(prefix):]
    if len(short) <= max_chars:
        return short
    return "…" + short[-(max_chars - 1):]


def _port_kind_masks(labels: Iterable[Any]) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    """Return bright, dark, spectro, and unclassified masks for port labels."""

    label_strings = [str(label) for label in labels]
    is_bright = np.array(
        [label.startswith("bright") for label in label_strings], dtype=bool
    )
    is_dark = np.array(
        [label.startswith("dark") for label in label_strings], dtype=bool
    )
    is_spectro = np.array(
        [label.startswith("spectro") for label in label_strings], dtype=bool
    )
    other = ~(is_bright | is_dark | is_spectro)
    return is_bright, is_dark, is_spectro, other


def _detector_layout_payload(state: Any) -> dict[str, Any]:
    """Best-effort detector payload carrying logical/physical port counts."""

    for name, payload in state.results.components.items():
        if not isinstance(payload, dict):
            continue
        if name.startswith("detector") and (
            "port_labels" in payload or "n_modes" in payload
        ):
            return payload
    return {}


def _optional_int(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    with suppress(TypeError, ValueError):
        return int(value)
    return None


def _detector_layout_caption(
    state: Any,
    labels: Iterable[Any],
    n_ports: int,
) -> str:
    """Human-readable physical-port layout, avoiding N/mode ambiguity."""

    is_bright, is_dark, is_spectro, other = _port_kind_masks(labels)
    payload = _detector_layout_payload(state)
    n_modes = _optional_int(payload, "n_modes")
    n_arms = _optional_int(payload, "n_pol_arms")
    if n_modes is not None and n_arms is not None and n_arms > 1:
        prefix = (
            f"{n_ports} physical detector ports "
            f"({n_modes} logical chip modes x {n_arms} arms)"
        )
    elif n_modes is not None:
        prefix = f"{n_ports} detector ports ({n_modes} logical chip modes)"
    else:
        prefix = f"{n_ports} detector ports"

    parts = [
        f"bright {int(is_bright.sum())}",
        f"spectro {int(is_spectro.sum())}",
        f"dark {int(is_dark.sum())}",
    ]
    if other.any():
        parts.append(f"other {int(other.sum())}")
    return f"{prefix}  ·  " + "  ·  ".join(parts)


def _add_rule(fig: plt.Figure, x0: float, x1: float, y: float,
              color: str = _INK, lw: float = 1.0) -> None:
    fig.add_artist(Line2D(
        [_fx(x0), _fx(x1)], [_fy(y), _fy(y)],
        color=color, linewidth=lw, transform=fig.transFigure,
        solid_capstyle="butt",
    ))


def _add_filled_rect(fig: plt.Figure, x0: float, y0: float, x1: float, y1: float,
                     color: str, alpha: float = 1.0) -> None:
    fig.add_artist(Rectangle(
        (_fx(x0), _fy(y0)), _fx(x1 - x0), _fy(y1 - y0),
        facecolor=color, edgecolor="none", alpha=alpha,
        transform=fig.transFigure, zorder=0,
    ))


def _add_filled_rect_ax(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float,
                        color: str, alpha: float = 1.0) -> None:
    ax.add_patch(Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        facecolor=color, edgecolor="none", alpha=alpha,
        transform=ax.transAxes, zorder=0, clip_on=False,
    ))


def _ftext(fig: plt.Figure, x_in: float, y_in: float, s: str, **kw: Any) -> None:
    kw.setdefault("ha", "left")
    kw.setdefault("va", "baseline")
    fig.text(_fx(x_in), _fy(y_in), s, **kw)


def _header(fig: plt.Figure, eyebrow: str, title: str,
            page_num: int, total: int) -> None:
    """Draw header strip: eyebrow + title left, page count right, rule below."""
    _ftext(fig, _MARGIN_L, _HEADER_Y, eyebrow.upper(),
           fontsize=7.5, color=_MUTE, family="sans-serif",
           weight="bold", parse_math=False,
           ha="left", va="baseline")
    _ftext(fig, _PAGE_W - _MARGIN_R, _HEADER_Y,
           f"{page_num:02d} / {total:02d}",
           fontsize=7.5, color=_MUTE, family="sans-serif",
           weight="bold", parse_math=False,
           ha="right", va="baseline")
    _add_rule(fig, _MARGIN_L, _PAGE_W - _MARGIN_R, _RULE_Y,
              color=_INK, lw=1.0)
    _ftext(fig, _MARGIN_L, _TITLE_Y, title,
           fontsize=18, color=_INK, family="serif",
           parse_math=False, ha="left", va="baseline")


def _footer(fig: plt.Figure, left: str, right: str = "") -> None:
    _add_rule(fig, _MARGIN_L, _PAGE_W - _MARGIN_R, _FOOTER_Y + 0.12,
              color=_RULE, lw=0.5)
    _ftext(fig, _MARGIN_L, _FOOTER_Y, left,
           fontsize=7, color=_MUTE, family="sans-serif",
           parse_math=False, ha="left", va="baseline")
    if right:
        _ftext(fig, _PAGE_W - _MARGIN_R, _FOOTER_Y, right,
               fontsize=7, color=_MUTE, family="sans-serif",
               parse_math=False, ha="right", va="baseline")


def _block_label(fig: plt.Figure, y_top: float, label: str,
                 kicker: str | None = None) -> None:
    """Render a block label band: thin accent rule + label + optional kicker."""
    x0 = _MARGIN_L
    x1 = _PAGE_W - _MARGIN_R
    # accent rule
    _add_rule(fig, x0, x0 + 0.35, y_top + 0.04, color=_ACCENT, lw=2.0)
    # label
    _ftext(fig, x0 + 0.45, y_top, label,
           fontsize=10.5, color=_INK, family="serif",
           parse_math=False, ha="left", va="baseline")
    if kicker:
        _ftext(fig, x1, y_top, kicker,
               fontsize=8.5, color=_MUTE, family="sans-serif",
               parse_math=False, ha="right", va="baseline")


def _short_repr(v: Any) -> str:
    if isinstance(v, np.ndarray):
        return _array_repr(v)
    if isinstance(v, (list, tuple)):
        n = len(v)
        if n == 0:
            return "()"
        if n <= 3:
            return repr(v)
        if all(isinstance(item, (bool, np.bool_)) for item in v):
            return f"[{n} bools] true={sum(bool(item) for item in v)}/{n}"
        try:
            arr = np.asarray(v, dtype=float)
            return (f"[{n} vals] min={arr.min():.3e}  med={np.median(arr):.3e}  "
                    f"max={arr.max():.3e}")
        except (TypeError, ValueError):
            return f"[{n} entries] first={v[0]!r}"
    if isinstance(v, dict):
        keys = list(v)
        if not keys:
            return "{}"
        preview = ", ".join(str(k) for k in keys[:4])
        suffix = "" if len(keys) <= 4 else ", …"
        return f"dict[{len(keys)}]: {preview}{suffix}"
    if isinstance(v, str):
        return repr(_ellipsize(v, 68))
    if isinstance(v, float):
        return f"{v:.6e}" if abs(v) < 1e-3 or abs(v) >= 1e4 else f"{v:.6g}"
    return repr(v)


def _array_repr(value: np.ndarray) -> str:
    """Compact representation for report tables.

    The old raw ``repr(ndarray)`` routinely crossed the page boundary and hid
    the useful scalar diagnostics below it. This keeps shape/range information
    visible while making arrays cheap to scan.
    """

    arr = np.asarray(value)
    if arr.size == 0:
        return f"array shape={arr.shape} empty"
    finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.number) else None
    if np.issubdtype(arr.dtype, np.bool_):
        true_count = int(np.count_nonzero(arr))
        return f"bool array shape={arr.shape} true={true_count}/{arr.size}"
    if np.issubdtype(arr.dtype, np.complexfloating):
        mag = np.abs(arr)
        finite_mag = mag[np.isfinite(mag)]
        if finite_mag.size:
            return (
                f"complex array shape={arr.shape} |x| "
                f"min={finite_mag.min():.3e} med={np.median(finite_mag):.3e} "
                f"max={finite_mag.max():.3e}"
            )
        return f"complex array shape={arr.shape} no finite magnitudes"
    if finite is not None and finite.size:
        return (
            f"array shape={arr.shape} min={finite.min():.3e} "
            f"med={np.median(finite):.3e} max={finite.max():.3e}"
        )
    if finite is not None:
        return f"array shape={arr.shape} no finite values"
    flat = arr.reshape(-1)
    return f"array shape={arr.shape} first={flat[0]!r}"


def _port_labels_repr(labels: list[Any] | tuple[Any, ...]) -> str:
    is_bright, is_dark, is_spectro, other = _port_kind_masks(labels)
    parts = [
        f"bright={int(is_bright.sum())}",
        f"spectro={int(is_spectro.sum())}",
        f"dark={int(is_dark.sum())}",
    ]
    if other.any():
        parts.append(f"other={int(other.sum())}")
    return (
        f"[{len(labels)} physical entries] "
        + " ".join(parts)
        + f" first={labels[0]!r}"
    )


def _result_value_repr(key: str, value: Any) -> str:
    if key == "port_labels" and isinstance(value, (list, tuple)) and value:
        return _port_labels_repr(value)
    return _short_repr(value)


def _is_array_like_result(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return value.ndim > 0 and value.size > 1
    return isinstance(value, (list, tuple)) and len(value) > 3


def _result_item_rank(item: tuple[str, Any]) -> tuple[int, str]:
    """Sort report rows so headline scalars appear before bulky payloads."""

    key, value = item
    key_lower = key.lower()
    important = any(token in key_lower for token in _IMPORTANT_RESULT_TOKENS)
    if important and not _is_array_like_result(value):
        return (0, key_lower)
    if not _is_array_like_result(value):
        return (1, key_lower)
    if important:
        return (2, key_lower)
    return (3, key_lower)


def _human_time(s: float) -> str:
    if not np.isfinite(float(s)):
        return "n/a"
    if s < 1.0:
        return f"{s * 1000:.0f} ms"
    if s < 60.0:
        return f"{s:.2f} s"
    if s < 3600.0:
        m = int(s // 60)
        return f"{m} m {s - 60 * m:.0f} s"
    h = int(s // 3600)
    m = int((s - 3600 * h) // 60)
    return f"{h} h {m} m"


def _human_count(n: float) -> str:
    if n < 1e3:
        return f"{n:.0f}"
    if n < 1e6:
        return f"{n / 1e3:.1f}k"
    if n < 1e9:
        return f"{n / 1e6:.1f}M"
    if n < 1e12:
        return f"{n / 1e9:.2f}G"
    return f"{n:.2e}"


def _is_stage_component(
    name: str,
    stage_id: str,
    stage_type: str | None = None,
) -> bool:
    # Components are usually named by the stage TYPE (type_name), not the
    # config stage id; matching on the id alone silently dropped every
    # stage whose id differs from its type prefix (characterization_curve,
    # chip_opt, delay_lines, amplitude_pickoff — review 2026-06-10
    # Issue 8).
    if name == stage_id or name.startswith((f"{stage_id}_", f"{stage_id}.")):
        return True
    if stage_type:
        return name == stage_type or name.startswith(
            (f"{stage_type}_", f"{stage_type}.")
        )
    return False


# ---------------------------------------------------------------------------
# Cover / overview page
# ---------------------------------------------------------------------------

_ALL_SUB_BUDGETS = (
    "scene",
    "geometry",
    "field",
    "opd_budget",
    "throughput",
    "wavefront",
    "photon_rates",
    "results",
)


def _stage_contributions(
    record: RunRecord,
    sid: str,
    stype: str | None = None,
) -> list[str]:
    state = record.result.state
    wrote: list[str] = []
    for sb_name in _ALL_SUB_BUDGETS:
        sb = getattr(state, sb_name)
        components = getattr(sb, "components", None)
        if not components:
            continue
        for name in components:
            if _is_stage_component(name, sid, stype):
                wrote.append(sb_name)
                break
    return wrote


def _stage_has_content(record: RunRecord, stage_spec: Any) -> bool:
    return bool(
        _stage_contributions(
            record,
            stage_spec.id,
            getattr(stage_spec, "type", None),
        )
    )


def _overview_page(record: RunRecord, page_num: int, total: int) -> plt.Figure:
    state = record.result.state
    cfg = record.config
    timings = {t.instance_id: t for t in record.result.timings}
    stages = list(cfg.pipeline.stages)
    wallclock = sum(t.elapsed_s for t in record.result.timings)

    fig = _new_page()
    _header(fig, "nullsim · debug report", cfg.run.name, page_num, total)

    subtitle = (
        "Per-cell pipeline diagnostics"
        if record.coord_tuple
        else "Per-stage pipeline diagnostics"
    )
    _ftext(fig, _MARGIN_L, _TITLE_Y - 0.28,
           subtitle,
           fontsize=10.5, color=_MUTE, family="sans-serif",
           ha="left", va="baseline")

    # ---------- METADATA CARD ----------
    card_top = 9.30
    card_bot = 8.40
    _add_filled_rect(fig, _MARGIN_L, card_bot,
                     _PAGE_W - _MARGIN_R, card_top, _STRIPE)
    _block_label(fig, card_top + 0.13, "Run metadata")

    from nullsim.outputs.plots._axis_labels import compact_record_label
    sweep_str = (
        compact_record_label(record.coord_tuple, "(single run)")
        if record.coord_tuple
        else "(single run)"
    )
    run_hash_label = "cell run hash" if record.coord_tuple else "run hash"
    config_hash_label = "cell config hash" if record.coord_tuple else "config hash"
    meta_rows = [
        (run_hash_label,    record.config.run_hash),
        (config_hash_label, record.config.config_hash),
        ("sweep coord",     sweep_str),
        ("stages",          f"{len(stages)}   ·   cache hits  {record.result.cache_hit_count}"),
        ("wallclock",       _human_time(wallclock)),
    ]
    row_h = (card_top - 0.10 - card_bot - 0.10) / len(meta_rows)
    for i, (k, v) in enumerate(meta_rows):
        y = card_top - 0.18 - (i + 0.7) * row_h
        _ftext(fig, _MARGIN_L + 0.20, y, k.upper(),
               fontsize=7.5, color=_MUTE, family="sans-serif",
               weight="bold")
        _ftext(fig, _MARGIN_L + 1.55, y, str(v),
               fontsize=10, color=_INK, family="monospace")

    # ---------- STAGE TABLE ----------
    table_top = 8.15
    _block_label(fig, table_top, "Pipeline stages",
                 kicker=f"{len(stages)} stages")

    table_x0 = _MARGIN_L
    table_x1 = _PAGE_W - _MARGIN_R
    # Column layout (in inches from table_x0):
    #  idx 0.05 | id 0.45 | type 2.20 | time-bar 4.05 | cache 5.65 | wrote 6.30
    col_x = {
        "idx":   table_x0 + 0.05,
        "id":    table_x0 + 0.45,
        "type":  table_x0 + 2.20,
        "time":  table_x0 + 4.05,
        "cache": table_x0 + 5.55,
        "wrote": table_x0 + 6.20,
    }
    bar_max_w = 0.65
    chip_w = 0.50

    hdr_y = table_top - 0.30
    _add_rule(fig, table_x0, table_x1, hdr_y + 0.16, color=_INK, lw=0.7)
    for key, label in (
        ("idx", "#"), ("id", "ID"), ("type", "TYPE"),
        ("time", "TIME"), ("cache", "CACHE"), ("wrote", "CONTRIBUTIONS"),
    ):
        _ftext(fig, col_x[key], hdr_y, label,
               fontsize=7.5, color=_MUTE, family="sans-serif",
               weight="bold", va="baseline")
    _add_rule(fig, table_x0, table_x1, hdr_y - 0.06, color=_INK, lw=0.5)

    row_h = 0.22
    max_t = max((t.elapsed_s for t in record.result.timings if t.elapsed_s > 0),
                default=1.0)

    for i, spec in enumerate(stages):
        y = hdr_y - 0.25 - i * row_h
        if y - row_h * 0.5 < 1.80:
            break
        if i % 2 == 0:
            _add_filled_rect(fig, table_x0 + 0.01, y - row_h + 0.18,
                             table_x1 - 0.01, y + 0.16, _STRIPE, alpha=0.6)
        t = timings.get(spec.id)
        elapsed = t.elapsed_s if t else None
        cache_hit = bool(t and t.cache_hit)
        contribs = _stage_contributions(record, spec.id, getattr(spec, "type", None))

        _ftext(fig, col_x["idx"], y, f"{i}",
               fontsize=8.5, color=_MUTE, family="monospace")
        _ftext(fig, col_x["id"], y, _ellipsize(spec.id, 20),
               fontsize=9, color=_INK, family="monospace")
        _ftext(fig, col_x["type"], y, _ellipsize(spec.type, 24),
               fontsize=8.0, color=_MUTE, family="monospace")

        # Time bar + numeric
        if elapsed is not None:
            bar_w = bar_max_w * min(1.0, elapsed / max_t) if max_t > 0 else 0
            bar_color = _ACCENT if elapsed > 0.5 * max_t else _COOL
            _add_filled_rect(fig, col_x["time"], y - 0.03,
                             col_x["time"] + bar_w, y + 0.06,
                             bar_color, alpha=0.85)
            _ftext(fig, col_x["time"] + bar_max_w + 0.05, y,
                   _human_time(elapsed),
                   fontsize=8, color=_INK, family="monospace")
        else:
            _ftext(fig, col_x["time"], y, "-",
                   fontsize=8, color=_MUTE, family="monospace")

        # Cache chip
        chip_color = _GOOD if cache_hit else _MUTE
        chip_text = "HIT" if cache_hit else "MISS"
        cx_chip = col_x["cache"]
        _add_filled_rect(fig, cx_chip, y - 0.04, cx_chip + chip_w, y + 0.11,
                         chip_color, alpha=0.18)
        _ftext(fig, cx_chip + chip_w / 2, y, chip_text,
               fontsize=7, color=chip_color, family="sans-serif",
               weight="bold", ha="center", va="baseline")

        # Contributions — abbreviate to fit the narrow column without
        # truncating mid-word at the right margin.
        _ABBR = {
            "throughput": "thpt",
            "wavefront": "wave",
            "opd_budget": "opd",
            "results": "res",
            "field": "fld",
            "geometry": "geom",
            "photon_rates": "phot",
            "scene": "scn",
        }
        contrib_str = "  ".join(_ABBR.get(c, c) for c in contribs) if contribs else "-"
        _ftext(fig, col_x["wrote"], y, contrib_str,
               fontsize=8, color=_INK, family="monospace")

    if record.result.cache_hit_count:
        _ftext(
            fig, table_x0 + 0.05, 1.95,
            "Cache note: HIT row timings are cache lookup/materialization time, "
            "not the original stage compute time.",
            fontsize=7.5, color=_MUTE, family="sans-serif",
            ha="left", va="baseline",
        )

    # ---------- HEADLINE RESULTS ----------
    results = dict(state.results.components)
    sens = results.get("sensitivity") or results.get("sensitivity.detection_curve") or {}
    perf = results.get("performance") or results.get("performance.standard") or {}
    calib = results.get("calibration") or results.get("calibration.floor_lookup") or {}
    detector = next((v for k, v in results.items() if k.startswith("detector")), {})

    null_raw = _first_finite_float(
        calib.get("null_depth_raw") if isinstance(calib, dict) else None,
        detector.get("null_depth_raw") if isinstance(detector, dict) else None,
    )
    null_cal = _first_finite_float(
        calib.get("null_depth_residual", calib.get("null_depth_calibrated"))
        if isinstance(calib, dict) else None,
        perf.get("null_depth_residual", perf.get("null_depth_calibrated"))
        if isinstance(perf, dict) else None,
    )
    final_null = _first_finite_float(
        perf.get("null_depth_effective") if isinstance(perf, dict) else None,
        state.results.null_depth,
        sens.get("effective_null_with_disk") if isinstance(sens, dict) else None,
        null_cal,
    )
    snr = perf.get("snr") if isinstance(perf, dict) else None
    int_time = perf.get("integration_time_for_target_snr_s") if isinstance(perf, dict) else None
    target_snr = perf.get("target_snr") if isinstance(perf, dict) else None

    card_top = 1.70
    card_bot = 0.78
    _add_filled_rect(fig, _MARGIN_L, card_bot,
                     _PAGE_W - _MARGIN_R, card_top, _STRIPE)
    _add_rule(fig, _MARGIN_L, _PAGE_W - _MARGIN_R, card_top + 0.005,
              color=_ACCENT, lw=2.0)
    _ftext(fig, _MARGIN_L + 0.20, card_top - 0.20,
           "HEADLINE RESULTS",
           fontsize=7.5, color=_ACCENT, family="sans-serif",
           weight="bold")

    metrics = [
        ("Detector raw null", f"{null_raw:.3e}" if null_raw is not None else "n/a",
         "diagnostic"),
        ("Final science null", f"{final_null:.3e}" if final_null is not None else "n/a",
         "used for SNR"),
        ("SNR", f"{snr:.2f}" if snr is not None else "n/a",
         f"target = {target_snr:g}" if target_snr is not None else ""),
        ("t for SNR target",
         _human_time(int_time) if int_time is not None else "n/a",
         "integration"),
    ]
    cell_w = (_PAGE_W - _MARGIN_L - _MARGIN_R) / len(metrics)
    for i, (label, value, sub) in enumerate(metrics):
        cx = _MARGIN_L + (i + 0.5) * cell_w
        _ftext(fig, cx, card_top - 0.45, label.upper(),
               fontsize=7.5, color=_MUTE, family="sans-serif",
               weight="bold", ha="center", va="baseline")
        _ftext(fig, cx, card_top - 0.75, value,
               fontsize=15, color=_INK, family="serif",
               ha="center", va="baseline")
        if sub:
            _ftext(fig, cx, card_top - 0.95, sub,
                   fontsize=7.5, color=_MUTE, family="sans-serif",
                   ha="center", va="baseline")

    _footer(fig, f"run · {record.config.run.name}",
            right=f"hash {record.config.run_hash[:12]}")
    return fig


# ---------------------------------------------------------------------------
# Stage page (one per pipeline stage that contributed anything)
# ---------------------------------------------------------------------------

def _stage_panels(record: RunRecord, stage_spec: Any) -> list[tuple[str, str, Callable]]:
    sid = stage_spec.id
    state = record.result.state
    stype = stage_spec.type

    def _match(comp_dict: dict) -> dict:
        return {
            k: v for k, v in comp_dict.items()
            if _is_stage_component(k, sid, stype)
        }

    throughput = _match(dict(state.throughput.components))
    opd = _match(dict(state.opd_budget.components))
    wavefront = _match(dict(state.wavefront.components))
    results = _match(dict(state.results.components))

    wl_full = state.throughput.wavelength_grid_um or state.wavefront.wavelength_grid_um
    wl_full = np.asarray(wl_full, dtype=np.float64) if wl_full else None

    panels: list[tuple[str, str, Callable]] = []

    # Stage-specific dispatch. The chip/FT pages are otherwise nearly
    # empty (a flat-zero throughput chart and a single bar) because the
    # generic auto-panels don't know which results component carries the
    # interesting payload. Hand-routed panels surface the chip-optimizer
    # convergence + per-port leakage and the FT σ² breakdown.
    if stype.startswith(("chip.", "chip_optimization.")):
        chip_results = dict(state.results.components)
        chip_opt = chip_results.get("chip_optimization") or {}
        if isinstance(chip_opt, dict) and chip_opt:
            panels.append((
                "Chip optimizer", "chart",
                lambda ax, r=chip_opt, full_results=chip_results:
                _panel_chip_optimizer(ax, r, full_results),
            ))
        if state.photon_rates.rates_hz is not None and state.photon_rates.port_labels:
            panels.append((
                "Per-port leakage (post-detector)", "chart",
                lambda ax, rates=state.photon_rates: _panel_chip_leakage(ax, rates),
            ))
    elif stype == "fringe_tracking.closed_loop":
        ft_break = results.get("fringe_tracking.closed_loop")
        if isinstance(ft_break, dict) and ft_break:
            wavelength_center_m = float(ft_break.get("wavelength_center_m", 1.25e-6))
            panels.append((
                "FT residual σ² breakdown", "chart",
                lambda ax, b=ft_break, c=wavelength_center_m: _panel_ft_breakdown(
                    ax, b, c
                ),
            ))
            panels.append((
                "Disturbance & control model", "text",
                lambda ax, b=ft_break, r=record, s=stage_spec:
                _panel_ft_disturbances(ax, b, r, s),
            ))
    elif stype == "diagnostics.fringe_observability":
        payload = next((v for v in results.values() if isinstance(v, dict)), None)
        if isinstance(payload, dict) and payload:
            panels.append((
                "Piston observability", "chart",
                lambda ax, p=payload: _panel_fringe_observability(ax, p),
            ))

    if throughput:
        # Suppress chip throughput panels only when every chip-loss component
        # is a flat unit transmission. Non-unit chip insertion loss is
        # diagnostic and should be shown.
        suppress = stype.startswith("chip.") and all(
            name.endswith(("_loss", ".loss")) and (
                not isinstance(v, dict)
                or all(abs(x - 1.0) < 1e-12 for x in v.get("transmission_fraction", ()))
            )
            for name, v in throughput.items()
        )
        if not suppress:
            panels.append(("Spectral throughput", "chart",
                           lambda ax, t=throughput: _panel_throughput(ax, t)))
    if wavefront:
        panels.append(("Wavefront quality", "chart",
                       lambda ax, w=wavefront, g=wl_full: _panel_wavefront(ax, w, g)))
    # FT page already shows the breakdown above; skip the single-bar
    # OPD panel that only adds noise.
    if opd and stype != "fringe_tracking.closed_loop":
        panels.append(("OPD contribution", "chart",
                       lambda ax, o=opd: _panel_opd(ax, o)))
    if results:
        # The chip + FT pages already extracted their headline results.
        # The generic dump duplicates them and steals a lot of vertical
        # space; suppress on those stages.
        suppress_results = stype.startswith(
            ("chip.", "chip_optimization.")
        ) or stype in {
            "fringe_tracking.closed_loop",
            "diagnostics.fringe_observability",
        }
        if not suppress_results:
            panels.append(("Stage results", "text",
                           lambda ax, r=results: _panel_results(ax, r)))
    return panels


def _stage_page(record: RunRecord, stage_spec: Any,
                page_num: int, total: int) -> plt.Figure:
    panels = _stage_panels(record, stage_spec)
    fig = _new_page()
    _header(fig, f"stage · {stage_spec.type}",
            stage_spec.id, page_num, total)
    _ftext(fig, _MARGIN_L, _TITLE_Y - 0.28,
           f"Stage {_pipeline_index(record, stage_spec)} of "
           f"{len(record.config.pipeline.stages)}    "
           f"·    {_panel_summary(panels)}",
           fontsize=10, color=_MUTE, family="sans-serif",
           ha="left", va="baseline")

    if not panels:
        _ftext(fig, _PAGE_W / 2, 5.5,
               "Stage produced no payload",
               fontsize=12, color=_MUTE, family="serif",
               ha="center", va="center")
        _footer(fig, f"stage · {stage_spec.id}")
        return fig

    n_chart = sum(1 for _, t, _ in panels if t == "chart")
    n_text = sum(1 for _, t, _ in panels if t == "text")

    gap = 0.65                                  # vertical space between panels
    avail = _CONTENT_TOP - _CONTENT_BOTTOM
    label_h = 0.30                              # space reserved above the axes for our block label

    # Per-chart height heuristic — leaves visual room for xlabel + tick labels
    if n_chart >= 3:
        chart_h = 2.10
    elif n_chart == 2:
        chart_h = 2.85
    elif n_chart == 1 and n_text == 0:
        # Single chart, no text — fill the page (less label + footer breathing room)
        chart_h = avail - label_h - 0.30
    else:
        chart_h = 3.10

    total_chart = n_chart * (chart_h + label_h)
    total_gap = max(0, len(panels) - 1) * gap
    text_total = avail - total_chart - total_gap
    text_h = max(1.5, text_total / max(n_text, 1)) if n_text else 0

    y = _CONTENT_TOP
    for label, ptype, render in panels:
        h = chart_h if ptype == "chart" else text_h
        block_top = y
        _block_label(fig, block_top, label,
                     kicker=_panel_kicker(record, stage_spec, label))
        ax_top = block_top - label_h
        ax_bot = ax_top - h
        ax = _add_axes_in(fig, _MARGIN_L + 0.05, ax_bot,
                          _PAGE_W - _MARGIN_R - 0.05, ax_top)
        render(ax)
        y = ax_bot - gap

    _footer(fig, f"stage · {stage_spec.id}",
            right=f"type · {stage_spec.type}")
    return fig


def _pipeline_index(record: RunRecord, stage_spec: Any) -> int:
    for i, s in enumerate(record.config.pipeline.stages):
        if s.id == stage_spec.id:
            return i
    return -1


def _panel_summary(panels: list[tuple[str, str, Callable]]) -> str:
    return "  ·  ".join(label.lower() for label, _, _ in panels)


def _panel_kicker(record: RunRecord, stage_spec: Any, panel_label: str) -> str:
    """Right-aligned annotation on a panel label — carries the headline stat."""
    state = record.result.state
    sid = stage_spec.id
    stype = getattr(stage_spec, "type", None)

    def _matched(d: dict) -> dict:
        return {k: v for k, v in d.items() if _is_stage_component(k, sid, stype)}

    if panel_label.startswith("Spectral"):
        comps = _matched(dict(state.throughput.components))
        losses_db: list[float] = []
        for payload in comps.values():
            if not isinstance(payload, dict):
                continue
            t = payload.get("transmission_fraction")
            if not t:
                continue
            t_arr = np.asarray(t, dtype=np.float64)
            loss = -10.0 * np.log10(np.where(t_arr > 0, t_arr, np.nan))
            losses_db.append(float(np.nanmedian(loss)))
        if losses_db:
            return f"median loss   {np.mean(losses_db):.3f} dB  ·  {len(comps)} comp"
        return f"{len(comps)} component" + ("s" if len(comps) != 1 else "")
    if panel_label.startswith("Wavefront"):
        comps = _matched(dict(state.wavefront.components))
        strehls: list[float] = []
        for payload in comps.values():
            if not isinstance(payload, dict):
                continue
            s = payload.get("strehl_ratio")
            if not s:
                continue
            strehls.append(float(np.nanmean(np.asarray(s, dtype=np.float64))))
        if strehls:
            mean_s = float(np.mean(strehls))
            # 3-digit fixed is unreadable below 0.01 (open-loop atmosphere
            # at typical Maunakea σ_OPD ~ 2 µm rounds to "0.000"). Use
            # 3-sig fig instead, with a scientific fallback below 1e-4.
            if mean_s >= 0.01:
                txt = f"{mean_s:.3f}"
            elif mean_s >= 1.0e-4:
                txt = f"{mean_s:.3g}"
            else:
                txt = f"{mean_s:.2e}"
            return f"mean Strehl   {txt}  ·  {len(comps)} comp"
        return f"{len(comps)} component" + ("s" if len(comps) != 1 else "")
    if panel_label.startswith("OPD"):
        comps = _matched(dict(state.opd_budget.components))
        rms_active: list[float] = []
        for payload in comps.values():
            rms = getattr(payload, "rms_opd_nm", None)
            sup = getattr(payload, "superseded", False)
            if rms is None and isinstance(payload, dict):
                rms = payload.get("rms_opd_nm")
                sup = payload.get("superseded", False)
            if rms is not None and not sup:
                rms_active.append(float(rms))
        if rms_active:
            quad = float(np.sqrt(sum(r * r for r in rms_active)))
            return f"Σ (quadrature)   {quad:.2f} nm  ·  {len(comps)} comp"
        return f"{len(comps)} component" + ("s" if len(comps) != 1 else "")
    if panel_label.startswith("Stage results"):
        comps = _matched(dict(state.results.components))
        n_keys = sum(len(p) for p in comps.values() if isinstance(p, dict))
        return f"{len(comps)} section · {n_keys} key" + ("s" if n_keys != 1 else "")
    return ""


# ---------------------------------------------------------------------------
# Stage panels
# ---------------------------------------------------------------------------

def _panel_throughput(ax: plt.Axes, components: dict[str, Any]) -> None:
    """Per-wavelength loss in dB for each component."""
    ax.set_facecolor(_PAPER)
    plotted = False
    palette = [_ACCENT, _COOL, _MUTE, _BRIGHT_PORT, _GOOD, _DARK_PORT]
    for i, (name, payload) in enumerate(components.items()):
        if not isinstance(payload, dict):
            continue
        wl = payload.get("wavelength_grid_um")
        t = payload.get("transmission_fraction")
        if not wl or not t:
            continue
        wl_arr = np.asarray(wl, dtype=np.float64)
        t_arr = np.asarray(t, dtype=np.float64)
        loss_db = -10.0 * np.log10(np.where(t_arr > 0, t_arr, np.nan))
        color = palette[i % len(palette)]
        ax.plot(wl_arr, loss_db, "-", color=color, linewidth=1.4,
                label=_component_label(name))
        ax.plot(wl_arr, loss_db, "o", color=color, markersize=2.5)
        plotted = True
    if not plotted:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no spectral payload",
                ha="center", va="center", color=_MUTE,
                fontsize=10, transform=ax.transAxes,
                family="serif")
        return
    ax.set_xlabel(f"Wavelength ({_UM})")
    ax.set_ylabel("Loss (dB)")
    ax.grid(True, axis="y", color=_RULE, linewidth=0.4, alpha=0.7)
    ax.legend(loc="best", fontsize=7.5, frameon=False)


def _panel_wavefront(ax: plt.Axes, components: dict[str, Any],
                     wl_grid: np.ndarray | None) -> None:
    """Strehl ratio vs wavelength (µm if grid known, else bin index).

    Auto-switches to log scale when every series stays below ~5% so an
    open-loop atmospheric component (Strehl ≈ exp(−σ²) ≈ 0 at typical
    Maunakea σ_OPD ~ 2 µm) is still readable. Annotates each curve with
    its in-band mean so the kicker number isn't the user's only handle.
    """
    ax.set_facecolor(_PAPER)
    series: list[tuple[str, np.ndarray]] = []
    for name, payload in components.items():
        if not isinstance(payload, dict):
            continue
        strehl = payload.get("strehl_ratio")
        if not strehl:
            continue
        s_arr = np.asarray(strehl, dtype=np.float64)
        if s_arr.size == 0:
            continue
        series.append((name, s_arr))
    if not series:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no strehl payload",
                ha="center", va="center", color=_MUTE,
                fontsize=10, transform=ax.transAxes,
                family="serif")
        return

    palette = [_COOL, _ACCENT, _GOOD, _MUTE, _DARK_PORT]
    max_strehl = max(float(np.max(s)) for _, s in series)

    # Every series is identically zero (or numerically negligible) — log
    # scale would hide all the data below the y-floor, leaving a blank
    # panel. Render a linear [0, 1] frame with a single annotation
    # instead, so the panel is honest about what it has to show.
    if max_strehl <= 1.0e-12:
        for name, s_arr in series:
            ax.plot(
                wl_grid if (wl_grid is not None and len(wl_grid) == len(s_arr))
                else np.arange(len(s_arr)),
                s_arr,
                "-o",
                color=palette[0],
                linewidth=1.4,
                markersize=2.5,
                label=f"{_component_label(name)}  ⟨S⟩={float(np.mean(s_arr)):.3g}",
            )
        ax.set_xlabel(
            f"Wavelength ({_UM})"
            if wl_grid is not None and len(wl_grid) == len(series[0][1])
            else "Wavelength bin index"
        )
        ax.set_ylabel("Strehl ratio")
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, color=_RULE, linewidth=0.5, linestyle=":")
        ax.text(
            0.5, 0.5,
            "Strehl ≡ 0 (open-loop)",
            transform=ax.transAxes, ha="center", va="center",
            color=_MUTE, fontsize=10, family="serif",
        )
        ax.grid(True, axis="y", color=_RULE, linewidth=0.4, alpha=0.7)
        ax.legend(loc="lower right", fontsize=7.5, frameon=False)
        return

    use_log = max_strehl < 0.05
    for i, (name, s_arr) in enumerate(series):
        color = palette[i % len(palette)]
        # Floor for log axis: a strict zero would blow up.
        plot_s = np.where(s_arr > 1e-12, s_arr, 1e-12) if use_log else s_arr
        if wl_grid is not None and len(wl_grid) == len(s_arr):
            ax.plot(wl_grid, plot_s, "-", color=color, linewidth=1.4,
                    label=f"{_component_label(name)}  ⟨S⟩={float(np.mean(s_arr)):.3g}")
            ax.plot(wl_grid, plot_s, "o", color=color, markersize=2.5)
            ax.set_xlabel(f"Wavelength ({_UM})")
        else:
            x = np.arange(len(s_arr))
            ax.plot(x, plot_s, "-o", color=color, linewidth=1.4,
                    markersize=2.5,
                    label=f"{_component_label(name)}  ⟨S⟩={float(np.mean(s_arr)):.3g}")
            ax.set_xlabel("Wavelength bin index")
    ax.set_ylabel("Strehl ratio")
    if use_log:
        ax.set_yscale("log")
        # Plain-text log labels — matplotlib's default LogFormatter
        # renders raw $10^{-1}$ under text.parse_math=False.
        ax.yaxis.set_major_formatter(_log_formatter())
        # Pin a reasonable lower limit so a Strehl of 1e-12 doesn't squash
        # everything else.
        ax.set_ylim(max(1e-9, max_strehl * 1e-4), 1.05)
        ax.text(0.99, 0.97, "log scale (Strehl < 5%)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, color=_MUTE, family="serif")
    else:
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, color=_RULE, linewidth=0.5, linestyle=":")
    ax.grid(True, axis="y", color=_RULE, linewidth=0.4, alpha=0.7)
    ax.legend(loc="lower right", fontsize=7.5, frameon=False)


def _panel_opd(ax: plt.Axes, components: dict[str, Any]) -> None:
    """Ranked horizontal bars of RMS OPD per component."""
    _inset_for_ylabels(ax, inches=1.30)
    ax.set_facecolor(_PAPER)
    rows: list[tuple[str, float, bool]] = []
    for name, payload in components.items():
        rms = getattr(payload, "rms_opd_nm", None)
        sup = getattr(payload, "superseded", False)
        if rms is None and isinstance(payload, dict):
            rms = payload.get("rms_opd_nm")
            sup = payload.get("superseded", False)
        if rms is None:
            continue
        rows.append((name, float(rms), bool(sup)))
    if not rows:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no rms_opd_nm payload",
                ha="center", va="center", color=_MUTE,
                fontsize=10, transform=ax.transAxes,
                family="serif")
        return
    rows.sort(key=lambda r: r[1])
    names = [_component_label(r[0]) for r in rows]
    vals = np.array([r[1] for r in rows])
    colors = [_MUTE if r[2] else _ACCENT for r in rows]
    y = np.arange(len(names))
    bars = ax.barh(y, vals, color=colors, edgecolor=_INK, linewidth=0.4,
                   height=0.55)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("RMS OPD (nm)")
    ax.grid(True, axis="x", color=_RULE, linewidth=0.4, alpha=0.7)
    ax.set_axisbelow(True)
    xmax = vals.max() if vals.max() > 0 else 1.0
    pad = xmax * 0.012
    for b, v, (_, _, sup) in zip(bars, vals, rows):
        ax.text(v + pad, b.get_y() + b.get_height() / 2,
                f"{v:.2f} nm" + ("  (superseded)" if sup else ""),
                va="center", ha="left", fontsize=7.5,
                color=_MUTE if sup else _INK,
                family="sans-serif")
    ax.set_xlim(0, xmax * 1.30)
    # Pad ylim so few bars don't look enormous. Minimum 5 slot heights.
    slots_min = 5
    pad_below = max(0.5, (slots_min - len(names)) / 2 + 0.5)
    pad_above = max(0.5, (slots_min - len(names)) / 2 + 0.5)
    ax.set_ylim(-pad_below, (len(names) - 1) + pad_above)


def _panel_results(ax: plt.Axes, components: dict[str, Any]) -> None:
    """Compact key/value table grouped by component.

    Scalars and headline health metrics are shown first; bulky arrays are
    summarized by shape/range and pushed lower so they do not crowd out the
    values a human checks first.
    """
    ax.set_axis_off()
    if not components:
        ax.text(0.5, 0.5, "(empty)", ha="center", va="center",
                color=_MUTE, fontsize=10, transform=ax.transAxes,
                family="serif")
        return

    bbox = ax.get_position()
    ax_h_in = bbox.height * _PAGE_H
    if ax_h_in <= 0:
        ax_h_in = 4.0
    row_h_in = 0.26
    section_lead_in = 0.36          # space reserved for the [section] header bar
    section_gap_in = 0.18           # gap after a section block

    y_in = ax_h_in
    for name, payload in components.items():
        # Section header bar
        y_in -= 0.10
        _add_filled_rect_ax(ax, 0.0, (y_in - 0.04) / ax_h_in,
                            1.0, (y_in + 0.20) / ax_h_in,
                            _STRIPE)
        ax.text(0.01, (y_in + 0.04) / ax_h_in, f"[{name}]",
                transform=ax.transAxes, fontsize=10, color=_ACCENT,
                family="serif", va="bottom", weight="bold")
        y_in -= section_lead_in
        if not isinstance(payload, dict):
            ax.text(0.02, y_in / ax_h_in, repr(payload),
                    transform=ax.transAxes, fontsize=9, color=_INK,
                    family="monospace", va="top")
            y_in -= row_h_in
            y_in -= section_gap_in
            continue
        items = sorted(payload.items(), key=_result_item_rank)
        for k, v in items:
            if y_in < row_h_in:
                ax.text(0.02, 0.01, "(more entries truncated)",
                        transform=ax.transAxes, fontsize=8,
                        color=_MUTE, family="sans-serif", va="bottom",
                        style="italic")
                return
            key_str = _ellipsize(k, 34)
            ax.text(0.02, y_in / ax_h_in, key_str,
                    transform=ax.transAxes, fontsize=9, color=_MUTE,
                    family="sans-serif", weight="bold", va="top",
                    clip_on=True)
            v_str = _result_value_repr(k, v)
            v_str = _ellipsize(v_str, 50)
            ax.text(0.43, y_in / ax_h_in, v_str,
                    transform=ax.transAxes, fontsize=8.2, color=_INK,
                    family="monospace", va="top", clip_on=True)
            y_in -= row_h_in
        y_in -= section_gap_in


# ---------------------------------------------------------------------------
# KILO reference snapshot
# ---------------------------------------------------------------------------

_KILO_REFERENCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "data" / "kilo_reference" / "keck_only_detection_space_cache.json"
)


def _kilo_reference_cache() -> dict[str, Any] | None:
    """Return the bundled KILO paperv3 detection_space snapshot, or None."""
    if not _KILO_REFERENCE_PATH.exists():
        return None
    try:
        with _KILO_REFERENCE_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _kilo_null_val_dark() -> float | None:
    """KILO Keck-only ``null_val_dark`` (~7.28e-5 at the M24 design)."""
    cache = _kilo_reference_cache()
    if not isinstance(cache, dict):
        return None
    primary = cache.get("primary")
    if not isinstance(primary, dict):
        return None
    val = primary.get("null_val_dark")
    return float(val) if val is not None else None


def _kilo_contrast_at_jmag(jmag: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (separations_mas, contrast) for the KILO J=jmag reference."""
    cache = _kilo_reference_cache()
    if not isinstance(cache, dict):
        return None
    seps = cache.get("separations_mas")
    if not seps:
        return None
    primary = cache.get("primary", {})
    curves = primary.get("curves", [])
    for row in curves:
        if isinstance(row, dict) and abs(float(row.get("jmag", -1)) - jmag) < 1e-6:
            return (
                np.asarray(seps, dtype=np.float64),
                np.asarray(row.get("contrast", []), dtype=np.float64),
            )
    return None


# ---------------------------------------------------------------------------
# Chip-stage panels
# ---------------------------------------------------------------------------


def _panel_chip_optimizer(
    ax: plt.Axes,
    chip_opt: dict[str, Any],
    full_results: dict[str, Any],
) -> None:
    """Optimizer convergence + headline loss + KILO reference context.

    Plots ``optimization_convergence_history`` when present (loss vs
    iter per restart). When the history isn't captured, falls back to a
    text card showing the final loss, iteration budget, backend, and
    a comparison against the bundled KILO ``null_val_dark`` from
    ``keck_only_detection_space_cache.json``.
    """
    ax.set_facecolor(_PAPER)
    # Producers (scipy and, since review 2026-06-10 Issue 9, torch) publish
    # ``convergence_history`` as flat (nfev, loss) pairs for the BEST
    # restart. The old reader looked up ``optimization_convergence_history``,
    # a key nothing ever published, so this panel was dead code that always
    # claimed "history not captured".
    history = chip_opt.get("convergence_history")
    final_loss = chip_opt.get("optimization_loss_final")
    iters = chip_opt.get("optimization_iterations")
    backend = chip_opt.get("backend_name", "?")

    sens = full_results.get("sensitivity.detection_curve") or full_results.get("sensitivity")
    chip_ideal = None
    chip_ideal_kilo = None
    if isinstance(sens, dict):
        chip_ideal = sens.get("ideal_chip_null_depth")
        # KILO-convention chip null (mean(dark)/mean(bright)), directly
        # comparable to multiband.J.null_depth_dark.
        chip_ideal_kilo = sens.get("ideal_chip_null_depth_kilo")

    kilo_ref = _kilo_null_val_dark()

    if isinstance(history, (list, tuple)) and history:
        arr = np.asarray(history, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 1:
            nfev = arr[:, 0]
            loss = arr[:, 1]
            mask = np.isfinite(loss) & (loss > 0)
            if mask.any():
                ax.plot(
                    nfev[mask], loss[mask],
                    "-", color=_ACCENT, linewidth=1.2, alpha=0.9,
                    label="best restart",
                )
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(_log_formatter())
                ax.yaxis.set_minor_locator(
                    LogLocator(base=10.0, subs="auto", numticks=12)
                )
                ax.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
                ax.set_xlabel("objective evaluations (nfev)")
                ax.set_ylabel("Loss (sum |U @ field|^2 at null indices)")
                ax.grid(True, color=_RULE, linewidth=0.4, alpha=0.7)
                if isinstance(final_loss, (int, float)):
                    ax.axhline(
                        float(final_loss), color=_INK, linewidth=0.6,
                        linestyle=":", alpha=0.5,
                    )
                ax.legend(loc="upper right", fontsize=7.5, frameon=False)
                ax.text(
                    0.01, 0.02,
                    _chip_optimizer_caption(
                        final_loss,
                        iters,
                        backend,
                        chip_ideal_kilo
                        if isinstance(chip_ideal_kilo, (int, float))
                        else chip_ideal,
                        kilo_ref,
                    ),
                    transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=8, color=_INK, family="monospace",
                )
                return

    # Fallback text card.
    ax.set_axis_off()
    lines = []
    if isinstance(final_loss, (int, float)):
        lines.append(f"optimization_loss_final  =  {float(final_loss):.4e}")
    if isinstance(iters, (int, float)):
        lines.append(f"optimization_iterations  =  {int(iters)}")
    lines.append(f"backend_name             =  {backend!r}")
    if isinstance(chip_ideal, (int, float)):
        lines.append("")
        lines.append(
            f"ideal_chip_null_depth    =  {float(chip_ideal):.3e}  "
            "(dark-power fraction; pipeline input)"
        )
    if isinstance(chip_ideal_kilo, (int, float)):
        lines.append(
            f"ideal_chip_null_depth_kilo =  {float(chip_ideal_kilo):.3e}  "
            "(mean(dark)/mean(bright); KILO null_depth_dark convention)"
        )
    # Compare the KILO-convention scalar, not the dark-power-fraction scalar
    # (they differ by ~n_dark/n_bright).
    reference_value = chip_ideal_kilo if isinstance(chip_ideal_kilo, (int, float)) and chip_ideal_kilo > 0 else None
    if kilo_ref is not None and reference_value is not None:
        ratio = float(reference_value) / kilo_ref
        lines.append(
            f"KILO null_val_dark reference =  {kilo_ref:.3e}    "
            f"[ratio {ratio:.2f}]"
        )
    lines.append("")
    lines.append(
        "convergence history not captured — set "
        "`capture_convergence_history = true` on the chip_optimization "
        "stage to plot loss-vs-iter per restart here."
    )
    wrapped_lines: list[str] = []
    for line in lines:
        if not line:
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(
            textwrap.wrap(
                line,
                width=86,
                subsequent_indent="    ",
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
    ax.text(
        0.02, 0.98, "\n".join(wrapped_lines),
        transform=ax.transAxes, ha="left", va="top",
        fontsize=8.6, color=_INK, family="monospace",
        clip_on=True,
    )


def _chip_optimizer_caption(
    final_loss: Any, iters: Any, backend: Any,
    chip_ideal: Any, kilo_ref: float | None,
) -> str:
    bits = []
    if isinstance(final_loss, (int, float)):
        bits.append(f"loss_final = {float(final_loss):.3e}")
    if isinstance(iters, (int, float)):
        bits.append(f"iters = {int(iters)}")
    bits.append(f"backend = {backend}")
    if kilo_ref is not None and isinstance(chip_ideal, (int, float)) and chip_ideal > 0:
        # ``chip_ideal`` here is the KILO-convention scalar
        # (mean(dark)/mean(bright)) when available — the only value
        # directly comparable to paperv3 null_val_dark (review 2026-06-10
        # Issue 24).
        ratio = float(chip_ideal) / kilo_ref
        bits.append(f"chip_ideal/KILO = ×{ratio:.1f}")
    return "  ·  ".join(bits)


def _panel_chip_leakage(ax: plt.Axes, photon_rates: Any) -> None:
    """Bar of mean per-port stellar leakage (post-detector, per wavelength)."""
    rates = np.asarray(photon_rates.rates_hz, dtype=np.float64)
    labels = list(photon_rates.port_labels or [])
    if rates.ndim != 2 or rates.size == 0 or not labels:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no per-port photon rates",
                ha="center", va="center", color=_MUTE,
                fontsize=10, transform=ax.transAxes, family="serif")
        return
    per_port = rates.sum(axis=1)  # summed over wavelength

    bright_mask, dark_mask, spec_mask, _ = _port_kind_masks(labels)

    ax.set_facecolor(_PAPER)
    x = np.arange(len(labels))
    colors = [
        _ACCENT if b else (_DARK_PORT if d else (_GOOD if s else _MUTE))
        for b, d, s in zip(bright_mask, dark_mask, spec_mask)
    ]
    ax.bar(x, per_port, color=colors, edgecolor="none", alpha=0.85)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(_log_formatter())
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs="auto", numticks=12))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
    ax.set_ylabel("sum-over-wl port rate (Hz)")
    ax.set_xlabel("Physical detector-port index")
    ax.grid(True, axis="y", which="both", color=_RULE, linewidth=0.4, alpha=0.6)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    # Mark the bright/dark partition boundary if the layout has both.
    n_bright = int(bright_mask.sum())
    n_dark = int(dark_mask.sum())
    if n_bright and n_dark:
        ax.axvline(n_bright - 0.5, color=_INK, linewidth=0.6,
                   linestyle="--", alpha=0.5)
    # Compact legend with sum-per-partition.
    parts = []
    if bright_mask.any():
        bright_sum = per_port[bright_mask].sum()
        parts.append(
            f"physical bright ({n_bright}): Σ = {bright_sum:.2e} Hz"
        )
    if spec_mask.any():
        spec_sum = per_port[spec_mask].sum()
        parts.append(
            f"physical spectro ({int(spec_mask.sum())}): "
            f"Σ = {spec_sum:.2e} Hz"
        )
    if dark_mask.any():
        dark_sum = per_port[dark_mask].sum()
        parts.append(
            f"physical dark ({n_dark}): Σ = {dark_sum:.2e} Hz"
        )
    # Reserve ~half a decade of headroom above the tallest bar so the
    # partition-sum annotation does not collide with the bar tops.
    top_rate = float(per_port.max())
    if top_rate > 0.0:
        ax.set_ylim(top=top_rate * 10.0)
    ax.text(0.99, 0.98, "\n".join(parts),
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7.6, color=_INK, family="monospace",
            bbox={
                "facecolor": _PAPER,
                "edgecolor": "none",
                "alpha": 0.9,
                "pad": 1.2,
            })


def _panel_ft_breakdown(
    ax: plt.Axes, breakdown: dict[str, Any], wavelength_center_m: float,
) -> None:
    """Bar chart of FT residual σ² split (KILO_PARITY_PLAN §2.3).

    Converts each rad² term to nm RMS via λ/(2π) so the chart shares
    units with the OPD budget on the wavefront-error-budget page.
    Shows the per-mode IID terms, the common differential disturbance
    terms (inter-telescope piston + vibration, or the photon-Kalman
    estimation / prediction / line-floor split), and the totals.
    """
    ax.set_facecolor(_PAPER)
    ax.tick_params(axis="y", labelsize=8)
    _inset_for_ylabels(ax, inches=2.0)
    nm_per_rad = float(wavelength_center_m) * 1e9 / (2.0 * math.pi)
    is_photon_kalman = breakdown.get("control_model") == "photon_kalman"

    def _nm(key: str) -> float:
        val = breakdown.get(key)
        if val is None or float(val) <= 0:
            return 0.0
        return math.sqrt(float(val)) * nm_per_rad

    rows = [
        ("iid/mode · irreducible (fit+WFS)", _nm("sigma2_irreducible_rad2"), _ACCENT),
        ("iid/mode · uncorr. temporal", _nm("sigma2_uncorrectable_temporal_rad2"), _ACCENT),
        ("iid/mode · servo lag (reduced)", _nm("sigma2_lag_reduced_rad2"), _ACCENT),
    ]
    if not is_photon_kalman:
        rows.append(("iid/mode · photon servo", _nm("sigma2_photon_rad2"), _ACCENT))
    rows.append(("IID per mode (total)", _nm("sigma2_iid_per_mode_rad2"), _GOOD))
    if is_photon_kalman:
        rows.extend([
            ("common · Kalman estimation", _nm("sigma2_photon_kalman_estimation_rad2"), _COOL),
            ("common · prediction (atm piston)", _nm("sigma2_photon_kalman_prediction_rad2"), _COOL),
            ("common · tracked-line floor", _nm("sigma2_photon_kalman_line_floor_rad2"), _COOL),
        ])
    else:
        rows.extend([
            ("common · inter-telescope piston", _nm("sigma2_inter_telescope_rad2"), _COOL),
            ("common · vibration residual", _nm("sigma2_vibration_rad2"), _COOL),
        ])
    rows.extend([
        ("Common differential (total)", _nm("sigma2_common_differential_rad2"), _GOOD),
        ("Residual total (iid + common)", _nm("sigma2_residual_rad2"), _GOOD),
        ("sigma_uncorrected (cap)", _nm("sigma2_uncorrected_rad2"), _MUTE),
    ])
    # Drop zero-magnitude rows for cleanliness.
    rows = [r for r in rows if r[1] > 0]
    if not rows:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no FT σ² breakdown payload",
                ha="center", va="center", color=_MUTE,
                fontsize=10, transform=ax.transAxes, family="serif")
        return
    y = np.arange(len(rows))[::-1]
    vals = [r[1] for r in rows]
    colors = [r[2] for r in rows]
    bars = ax.barh(y, vals, color=colors, edgecolor="none", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8, family="serif")
    ax.set_xlabel("RMS OPD-equivalent (nm)")
    ax.grid(True, axis="x", color=_RULE, linewidth=0.4, alpha=0.7)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() * 1.02, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f} nm", va="center", ha="left",
                fontsize=8, color=_INK, family="monospace")

    eta = breakdown.get("piston_sensing_efficiency")
    caption_bits = [f"control = {breakdown.get('control_model', 'classical')}"]
    if is_photon_kalman:
        f_3db = breakdown.get("photon_kalman_f_3db_equiv_hz")
        if isinstance(f_3db, (int, float)):
            caption_bits.append(f"f_3dB,equiv = {float(f_3db):.0f} Hz")
    else:
        f_sample = breakdown.get("f_sample_hz")
        f_3db = breakdown.get("f_3db_hz", breakdown.get("servo_bandwidth_hz"))
        if isinstance(f_sample, (int, float)):
            caption_bits.append(f"f_sample = {float(f_sample):.0f} Hz")
        if isinstance(f_3db, (int, float)):
            caption_bits.append(f"f_3dB = {float(f_3db):.0f} Hz")
    if isinstance(eta, (int, float)):
        caption_bits.append(f"eta = {float(eta):.3f}")
    if breakdown.get("shot_exceeds_open_loop"):
        caption_bits.append("[shot noise exceeds open-loop budget]")
    ax.text(0.01, -0.12, "   ·   ".join(caption_bits),
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7.5, color=_MUTE, family="monospace")


def _panel_ft_disturbances(
    ax: plt.Axes, breakdown: dict[str, Any], record: RunRecord, stage_spec: Any,
) -> None:
    """Key/value audit of the FT disturbance model and control loop.

    Surfaces the inputs a reader needs to judge the closed-loop piston
    floor: control model and bandwidth, the inter-telescope von Karman
    piston (uncorrected σ and PSD knee), the vibration line spectrum and
    broadband floor (with accelerometer-fusion suppression), the
    Greenwood effective wind vs the configured wind, the residual
    decomposition in nm, and which OPD components the FT residual
    supersedes.
    """
    ax.set_axis_off()
    state = record.result.state
    spec_params = {
        k: v for k, v in stage_spec.model_dump().items() if k not in ("id", "type")
    }
    ao_payload = state.wavefront.components.get("ao_residual")
    ao_payload = ao_payload if isinstance(ao_payload, dict) else {}

    def _f(value: Any, fmt: str = "{:.3g}") -> str:
        return fmt.format(float(value)) if isinstance(value, (int, float)) else "n/a"

    rows: list[tuple[str, str, str]] = []  # (kind, key, value)

    # -- Control loop -----------------------------------------------------
    rows.append(("section", "control loop", ""))
    is_pk = breakdown.get("control_model") == "photon_kalman"
    rows.append(("kv", "control_model", str(breakdown.get("control_model", "classical"))))
    if is_pk:
        rows.append(("kv", "boxcar window / f_3dB equiv",
                     f"{_f(breakdown.get('photon_kalman_window_s'), '{:.4g}')} s"
                     f"  /  {_f(breakdown.get('photon_kalman_f_3db_equiv_hz'), '{:.0f}')} Hz"))
        crlb = breakdown.get("crlb_m_sqrt_s")
        crlb_str = f"{float(crlb) * 1e9:.3g} nm·sqrt(s)" if isinstance(crlb, (int, float)) else "n/a"
        rows.append(("kv", "loop latency / shot CRLB",
                     f"{_f(breakdown.get('loop_latency_s'), '{:.3g}')} s  /  {crlb_str}"))
    else:
        rows.append(("kv", "f_sample / f_3dB",
                     f"{_f(breakdown.get('f_sample_hz'), '{:.0f}')} Hz"
                     f"  /  {_f(breakdown.get('f_3db_hz'), '{:.0f}')} Hz"
                     f"  (fraction {_f(breakdown.get('servo_bandwidth_fraction'))},"
                     f" source {breakdown.get('servo_bandwidth_source', '?')})"))
    rows.append(("kv", "tracker rate (usable, per port)",
                 f"{_f(breakdown.get('tracker_usable_rate_per_port_hz'), '{:.3g}')} Hz"
                 + ("  [RATE-CAPPED]" if breakdown.get("tracker_rate_clipped") else "")))

    # -- Disturbance inputs -----------------------------------------------
    rows.append(("section", "disturbance model", ""))
    eff_wind = ao_payload.get("effective_wind_speed_m_s")
    cfg_wind = spec_params.get("wind_speed_m_s")
    rows.append(("kv", "wind: Greenwood / configured",
                 f"{_f(eff_wind, '{:.2f}')} m/s (from tau0)  /  {_f(cfg_wind, '{:.2f}')} m/s"))
    inter_on = bool(spec_params.get("inter_telescope_piston", False))
    if inter_on:
        outer_scale = spec_params.get("outer_scale_m")
        diameter_m = state.geometry.diameter_m
        vk_str = f"ON · L0 {_f(outer_scale, '{:.1f}')} m"
        # The stage's own pair-averaged, aperture-filtered value.
        sigma_vk_m = breakdown.get("inter_telescope_uncorrected_piston_rms_m")
        if isinstance(sigma_vk_m, (int, float)):
            vk_str += f" · uncorr sigma {sigma_vk_m * 1e9:.0f} nm"
            if (
                isinstance(eff_wind, (int, float)) and eff_wind > 0
                and isinstance(outer_scale, (int, float)) and outer_scale > 0
                and diameter_m is not None and float(diameter_m) > 0
            ):
                knee_hz = piston_psd_knee_hz(
                    wind_speed_m_s=float(eff_wind),
                    outer_scale_m=float(outer_scale),
                    aperture_diameter_m=float(diameter_m),
                )
                vk_str += f" · PSD knee {knee_hz:.2f} Hz"
        rows.append(("kv", "inter-telescope VK piston", vk_str))
    else:
        rows.append(("kv", "inter-telescope VK piston", "OFF"))
    line_freqs = spec_params.get("vibration_line_frequencies_hz") or ()
    line_rms = spec_params.get("vibration_line_rms_nm") or ()
    if line_freqs:
        pairs = ", ".join(
            f"{float(f):g}/{float(a):g}" for f, a in zip(line_freqs, line_rms)
        )
        rows.append(("kv", "vibration lines (Hz/nm RMS)",
                     f"{pairs}  (post-feedforward)"))
    else:
        rows.append(("kv", "vibration lines", "none configured"))
    bb_cfg = spec_params.get("vibration_broadband_rms_nm")
    bb_eff = breakdown.get("vibration_broadband_rms_effective_nm")
    bb_str = f"{_f(bb_cfg, '{:.1f}')} nm"
    if breakdown.get("accelerometer_fusion"):
        bb_str += (
            f"  ->  {_f(bb_eff, '{:.1f}')} nm effective"
            f"  (accel fusion: coh {_f(breakdown.get('fusion_coherent_fraction'), '{:.2f}')},"
            f" supp x{_f(breakdown.get('fusion_broadband_suppression'), '{:.1f}')})"
        )
    rows.append(("kv", "vibration broadband floor", bb_str))
    if is_pk:
        rows.append(("kv", "tracked lines / residual floor",
                     f"{breakdown.get('tracked_line_count', 0)} lines"
                     f"  ·  {_f(breakdown.get('line_tracking_residual_floor_nm'), '{:.1f}')} nm/line"))

    # -- Residual decomposition --------------------------------------------
    rows.append(("section", "residual decomposition (nm RMS, OPD)", ""))
    rows.append(("kv", "total / iid per mode / common diff.",
                 f"{_f(breakdown.get('residual_rms_opd_nm'), '{:.2f}')}"
                 f"  /  {_f(breakdown.get('residual_iid_per_mode_nm'), '{:.2f}')}"
                 f"  /  {_f(breakdown.get('residual_common_differential_nm'), '{:.2f}')}"))
    rows.append(("kv", "inter-tel / vibration residual",
                 f"{_f(breakdown.get('sigma_inter_telescope_residual_nm'), '{:.2f}')}"
                 f"  /  {_f(breakdown.get('sigma_vibration_residual_nm'), '{:.2f}')}"))
    if is_pk:
        rows.append(("kv", "PK est / predict / line floor",
                     f"{_f(breakdown.get('photon_kalman_estimation_nm'), '{:.2f}')}"
                     f"  /  {_f(breakdown.get('photon_kalman_prediction_nm'), '{:.2f}')}"
                     f"  /  {_f(breakdown.get('photon_kalman_line_floor_nm'), '{:.2f}')}"))

    # -- OPD bookkeeping -----------------------------------------------------
    superseded = [
        name for name, payload in state.opd_budget.components.items()
        if getattr(payload, "superseded", False)
        or (isinstance(payload, dict) and payload.get("superseded", False))
    ]
    rows.append(("section", "opd bookkeeping", ""))
    rows.append(("kv", "supersedes (replaced in budget)",
                 ", ".join(superseded) if superseded else "none"))

    # -- Render -------------------------------------------------------------
    bbox = ax.get_position()
    ax_h_in = bbox.height * _PAGE_H
    if ax_h_in <= 0:
        ax_h_in = 4.0
    row_h_in = 0.225
    y_in = ax_h_in
    for kind, key, value in rows:
        if y_in < row_h_in:
            ax.text(0.02, 0.01, "(more entries truncated)",
                    transform=ax.transAxes, fontsize=8, color=_MUTE,
                    family="sans-serif", va="bottom", style="italic")
            return
        if kind == "section":
            y_in -= 0.14
            _add_filled_rect_ax(ax, 0.0, (y_in - 0.04) / ax_h_in,
                                1.0, (y_in + 0.18) / ax_h_in, _STRIPE)
            ax.text(0.01, (y_in + 0.02) / ax_h_in, f"[{key}]",
                    transform=ax.transAxes, fontsize=9.5, color=_ACCENT,
                    family="serif", va="bottom", weight="bold")
            y_in -= 0.26
            continue
        ax.text(0.02, y_in / ax_h_in, key,
                transform=ax.transAxes, fontsize=8.5, color=_MUTE,
                family="sans-serif", weight="bold", va="top", clip_on=True)
        ax.text(0.34, y_in / ax_h_in, _ellipsize(value, 68),
                transform=ax.transAxes, fontsize=8.0, color=_INK,
                family="monospace", va="top", clip_on=True)
        y_in -= row_h_in


def _panel_fringe_observability(ax: plt.Axes, payload: dict[str, Any]) -> None:
    """Compact singular-spectrum panel for the fringe-observability stage."""

    singular = np.asarray(payload.get("singular_values_per_m", ()), dtype=np.float64)
    ax.set_facecolor(_PAPER)
    if singular.size == 0:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no singular-value payload",
                ha="center", va="center", color=_MUTE,
                fontsize=10, transform=ax.transAxes, family="serif")
        return

    normalized = singular / max(float(singular[0]), np.finfo(float).tiny)
    x = np.arange(1, singular.size + 1)
    ax.semilogy(x, normalized, "o-", color=_COOL, linewidth=1.4, markersize=3.6)
    ax.yaxis.set_major_formatter(_log_formatter())
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
    tolerance = float(payload.get("singular_value_tolerance_per_m", 0.0))
    if tolerance > 0.0:
        ax.axhline(
            tolerance / max(float(singular[0]), np.finfo(float).tiny),
            color=_WARN,
            linestyle="--",
            linewidth=1.0,
        )
    n_dof = int(payload.get("n_differential_piston_modes", singular.size))
    ax.axvline(n_dof, color=_MUTE, linestyle=":", linewidth=1.0)
    ax.set_xlabel("Singular mode index")
    ax.set_ylabel("Singular value / max")
    ax.grid(True, which="both", color=_RULE, linewidth=0.4, alpha=0.65)
    ax.tick_params(axis="both", labelsize=8)

    cond = _as_float_or_none(payload.get("condition_number"))
    cond_text = "inf" if cond is None or not np.isfinite(cond) else f"{cond:.2e}"
    scalar_eta = _as_float_or_none(payload.get("scalar_piston_sensing_efficiency"))
    mc = payload.get("monte_carlo") if isinstance(payload.get("monte_carlo"), dict) else {}
    mc_med = _as_float_or_none(mc.get("median_residual_rms_nm"))
    nonlinear = (
        payload.get("nonlinear_one_step_monte_carlo")
        if isinstance(payload.get("nonlinear_one_step_monte_carlo"), dict)
        else {}
    )
    nonlinear_med = _as_float_or_none(nonlinear.get("median_residual_rms_nm"))
    cophasing = (
        payload.get("cophasing_handoff_monte_carlo")
        if isinstance(payload.get("cophasing_handoff_monte_carlo"), dict)
        else {}
    )
    cophasing_med = _as_float_or_none(
        cophasing.get("median_acquisition_residual_rms_nm")
    )
    maintenance_med = _as_float_or_none(
        cophasing.get("median_science_residual_rms_nm")
    )
    crlb = _as_float_or_none(payload.get("crlb_piston_rms_nm"))
    sweep = payload.get("channel_sweep")
    single_rank = None
    if isinstance(sweep, dict):
        with suppress(Exception):
            single_rank = int(sweep.get("single_channel_rank"))
    bits = [
        f"rank = {int(payload.get('rank', 0))}/{n_dof}",
        f"cond = {cond_text}",
        f"channels = {int(payload.get('n_spectral_channels_used', 0))}",
    ]
    if single_rank is not None:
        bits.append(f"1-ch rank = {single_rank}")
    if crlb is not None and np.isfinite(crlb):
        bits.append(f"CRLB = {crlb:.2g} nm")
    if scalar_eta is not None:
        bits.append(f"FT eta = {scalar_eta:.3f}")
    if mc_med is not None and np.isfinite(mc_med):
        bits.append(f"linear MC = {mc_med:.2g} nm")
    if nonlinear_med is not None and np.isfinite(nonlinear_med):
        bits.append(f"1-step nonlinear = {nonlinear_med:.2g} nm")
    if cophasing_med is not None and np.isfinite(cophasing_med):
        bits.append(f"cophasing acq = {cophasing_med:.2g} nm")
    if maintenance_med is not None and np.isfinite(maintenance_med):
        bits.append(f"science maint = {maintenance_med:.2g} nm")
    ax.text(0.01, -0.20, "   ·   ".join(bits),
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color=_MUTE, family="monospace")


# ---------------------------------------------------------------------------
# Spectral loss budget rollup
# ---------------------------------------------------------------------------

def _spectral_loss_page(record: RunRecord, page_num: int, total: int) -> plt.Figure:
    state = record.result.state
    components = dict(state.throughput.components)

    fig = _new_page()
    _header(fig, "spectral budget", "Per-component loss",
            page_num, total)
    _ftext(fig, _MARGIN_L, _TITLE_Y - 0.28,
           "Where the photons go before the detector",
           fontsize=10, color=_MUTE, family="sans-serif",
           ha="left", va="baseline")

    # Pre-compute median loss per component for ranking
    comp_loss: list[tuple[str, np.ndarray, np.ndarray]] = []  # name, wl, loss_db
    for name, payload in components.items():
        if not isinstance(payload, dict):
            continue
        wl = payload.get("wavelength_grid_um")
        t = payload.get("transmission_fraction")
        if not wl or not t:
            continue
        wl_arr = np.asarray(wl, dtype=np.float64)
        t_arr = np.asarray(t, dtype=np.float64)
        loss = -10.0 * np.log10(np.where(t_arr > 0, t_arr, np.nan))
        comp_loss.append((name, wl_arr, loss))

    # ---------- Block 1: spectral loss curves ----------
    _block_label(fig, _CONTENT_TOP, "Spectral loss per component",
                 kicker=f"{len(comp_loss)} component" + ("s" if len(comp_loss) != 1 else ""))
    ax1 = _add_axes_in(fig, _MARGIN_L + 0.10, 6.95,
                       _PAGE_W - _MARGIN_R - 0.05, 9.15)
    ax1.set_facecolor(_PAPER)
    # Use matplotlib's tab10 qualitative palette (10 visually distinct
    # hues). The prior 7-colour brand palette had two amber-orange
    # entries that collided when this page renders 7 components.
    tab10 = plt.get_cmap("tab10")
    for i, (name, wl, loss) in enumerate(comp_loss):
        color = tab10(i % tab10.N)
        ax1.plot(wl, loss, "-o", color=color, linewidth=1.3,
                 markersize=2.5, label=_component_label(name, max_chars=28))
    ax1.set_xlabel(f"Wavelength ({_UM})")
    ax1.set_ylabel("Loss (dB)")
    ax1.grid(True, axis="y", color=_RULE, linewidth=0.4, alpha=0.7)
    ax1.legend(fontsize=7.5, loc="best", frameon=False, ncol=2)

    # ---------- Block 2: cumulative transmission ----------
    _block_label(fig, 6.40, "Cumulative pre-detector transmission")
    ax2 = _add_axes_in(fig, _MARGIN_L + 0.10, 4.30,
                       _PAGE_W - _MARGIN_R - 0.05, 6.10)
    ax2.set_facecolor(_PAPER)
    if state.throughput.wavelength_grid_um and state.throughput.transmission_fraction:
        wl = np.asarray(state.throughput.wavelength_grid_um, dtype=np.float64)
        t = np.asarray(state.throughput.transmission_fraction, dtype=np.float64)
        # Fold in components that are applied to FIELD AMPLITUDES rather
        # than the top-level transmission chain (chip insertion loss, the
        # injection coupling) — otherwise this curve overstated delivered
        # pre-detector transmission by the chip's 2 dB relative to the
        # per-component panel directly above it (review 2026-06-10
        # Issue 24). Detector QE components (detector.*_qe) are divided
        # back out: this curve stops at the detector (review 2026-11-07
        # row 64).
        field_applied = np.ones_like(t)
        detector_qe = np.ones_like(t)
        any_field_applied = False
        for name, payload in state.throughput.components.items():
            if not isinstance(payload, dict):
                continue
            if name.startswith("detector.") and "qe" in payload:
                qe = np.asarray(payload["transmission_fraction"], dtype=np.float64)
                if qe.shape == t.shape:
                    detector_qe *= qe
                continue
            if payload.get("field_amplitude_applied") is not True:
                continue
            fraction = payload.get("transmission_fraction")
            if fraction is None:
                continue
            fraction_arr = np.asarray(fraction, dtype=np.float64)
            if fraction_arr.shape == t.shape:
                field_applied *= fraction_arr
                any_field_applied = True
        t_total = t * field_applied / detector_qe
        ax2.fill_between(wl, 0, t_total * 100, color=_ACCENT, alpha=0.15)
        ax2.plot(wl, t_total * 100, "-o", color=_ACCENT, linewidth=1.6,
                 markersize=3.0)
        ax2.set_xlabel(f"Wavelength ({_UM})")
        ax2.set_ylabel("Transmission (%)")
        ax2.set_ylim(0, max(100.0, float(t_total.max()) * 100 * 1.15))
        ax2.grid(True, axis="y", color=_RULE, linewidth=0.4, alpha=0.7)
        mid_i = len(wl) // 2
        ax2.annotate(f"{t_total[mid_i] * 100:.2f}% @ {wl[mid_i]:.3f} {_UM}",
                     xy=(wl[mid_i], t_total[mid_i] * 100),
                     xytext=(8, -16), textcoords="offset points",
                     fontsize=8, color=_INK, family="sans-serif",
                     arrowprops={"arrowstyle": "-", "color": _MUTE, "lw": 0.5})
        if any_field_applied:
            ax2.text(
                0.99, 0.95,
                "includes field-amplitude-applied losses "
                "(chip insertion, injection coupling)",
                transform=ax2.transAxes, ha="right", va="top",
                fontsize=7, color=_MUTE, family="sans-serif",
            )
    else:
        ax2.set_axis_off()

    # ---------- Block 3: weakest-link callout ----------
    if comp_loss:
        med = [(name, float(np.nanmedian(loss))) for name, _, loss in comp_loss]
        med.sort(key=lambda x: -x[1])
        _block_label(fig, 3.85, "Loss ranking",
                     kicker=f"largest: {med[0][0]}   {med[0][1]:.3f} dB")
        ax3 = _add_axes_in(fig, _MARGIN_L + 0.10, 1.15,
                           _PAGE_W - _MARGIN_R - 0.05, 3.55)
        _inset_for_ylabels(ax3, inches=1.25)
        ax3.set_facecolor(_PAPER)
        names = [_component_label(n, max_chars=30) for n, _ in med]
        vals = np.array([v for _, v in med])
        y = np.arange(len(names))[::-1]
        colors = [_ACCENT if i == 0 else _COOL for i in range(len(names))]
        bars = ax3.barh(y, vals, color=colors, edgecolor=_INK,
                        linewidth=0.4, height=0.65)
        ax3.set_yticks(y)
        ax3.set_yticklabels(names, fontsize=8.5)
        ax3.set_xlabel("Median loss (dB)")
        ax3.grid(True, axis="x", color=_RULE, linewidth=0.4, alpha=0.7)
        ax3.set_axisbelow(True)
        xmax = vals.max() if vals.max() > 0 else 1.0
        for b, v in zip(bars, vals):
            ax3.text(v + xmax * 0.015, b.get_y() + b.get_height() / 2,
                     f"{v:.3f} dB", va="center", ha="left",
                     fontsize=7.5, color=_INK,
                     family="sans-serif")
        ax3.set_xlim(0, xmax * 1.20)

    _footer(fig, "spectral loss budget")
    return fig


# ---------------------------------------------------------------------------
# OPD budget rollup
# ---------------------------------------------------------------------------

def _opd_summary_page(record: RunRecord, page_num: int, total: int) -> plt.Figure:
    state = record.result.state
    components = dict(state.opd_budget.components)

    fig = _new_page()
    _header(fig, "opd budget", "Wavefront error budget",
            page_num, total)
    _ftext(fig, _MARGIN_L, _TITLE_Y - 0.28,
           "RMS OPD per source · active vs superseded",
           fontsize=10, color=_MUTE, family="sans-serif",
           ha="left", va="baseline")

    rows: list[tuple[str, float, bool]] = []
    for name, payload in components.items():
        rms = getattr(payload, "rms_opd_nm", None)
        sup = getattr(payload, "superseded", False)
        if rms is None and isinstance(payload, dict):
            rms = payload.get("rms_opd_nm")
            sup = payload.get("superseded", False)
        if rms is None:
            continue
        rows.append((name, float(rms), bool(sup)))

    active_rows = [r for r in rows if not r[2]]
    superseded_rows = [r for r in rows if r[2]]

    # ---------- Block 1: active components ----------
    _block_label(fig, _CONTENT_TOP, "Active OPD components",
                 kicker=f"{len(active_rows)} active   ·   {len(superseded_rows)} superseded")
    ax1 = _add_axes_in(fig, _MARGIN_L + 0.10, 5.40,
                       _PAGE_W - _MARGIN_R - 0.05, 9.10)
    _inset_for_ylabels(ax1, inches=1.45)
    ax1.set_facecolor(_PAPER)
    if active_rows:
        active_sorted = sorted(active_rows, key=lambda r: r[1])
        names = [_component_label(r[0]) for r in active_sorted]
        vals = np.array([r[1] for r in active_sorted])
        y = np.arange(len(names))
        max_v = vals.max() if vals.max() > 0 else 1.0
        colors = [_ACCENT if v >= 0.7 * max_v else _COOL for v in vals]
        bars = ax1.barh(y, vals, color=colors, edgecolor=_INK,
                        linewidth=0.4, height=0.7)
        ax1.set_yticks(y)
        ax1.set_yticklabels(names, fontsize=9)
        ax1.set_xlabel("RMS OPD (nm)")
        ax1.grid(True, axis="x", color=_RULE, linewidth=0.4, alpha=0.7)
        ax1.set_axisbelow(True)
        for b, v in zip(bars, vals):
            ax1.text(v + max_v * 0.012, b.get_y() + b.get_height() / 2,
                     f"{v:.2f} nm", va="center", ha="left",
                     fontsize=8, color=_INK, family="sans-serif")
        ax1.set_xlim(0, max_v * 1.18)
    else:
        ax1.set_axis_off()
        ax1.text(0.5, 0.5, "no active OPD components",
                 ha="center", va="center", color=_MUTE,
                 fontsize=11, transform=ax1.transAxes,
                 family="serif")

    # ---------- Block 2: superseded reference (separate scale) ----------
    _block_label(fig, 4.95, "Superseded (reference only)")
    ax2 = _add_axes_in(fig, _MARGIN_L + 0.10, 3.00,
                       _PAGE_W - _MARGIN_R - 0.05, 4.65)
    _inset_for_ylabels(ax2, inches=1.45)
    ax2.set_facecolor(_PAPER)
    if superseded_rows:
        sup_sorted = sorted(superseded_rows, key=lambda r: r[1])
        names = [_component_label(r[0]) for r in sup_sorted]
        vals = np.array([r[1] for r in sup_sorted])
        y = np.arange(len(names))
        ax2.barh(y, vals, color=_MUTE, edgecolor=_INK,
                 linewidth=0.4, height=0.6, alpha=0.55,
                 hatch="///")
        ax2.set_yticks(y)
        ax2.set_yticklabels(names, fontsize=9)
        ax2.set_xlabel("RMS OPD (nm)")
        ax2.grid(True, axis="x", color=_RULE, linewidth=0.4, alpha=0.7)
        ax2.set_axisbelow(True)
        max_v = vals.max() if vals.max() > 0 else 1.0
        for v, yi in zip(vals, y):
            ax2.text(v + max_v * 0.012, yi, f"{v:.1f} nm",
                     va="center", ha="left", fontsize=8,
                     color=_MUTE, family="sans-serif")
        ax2.set_xlim(0, max_v * 1.18)
    else:
        ax2.set_axis_off()
        ax2.text(0.5, 0.5, "no superseded components",
                 ha="center", va="center", color=_MUTE,
                 fontsize=10, transform=ax2.transAxes,
                 family="serif")

    # ---------- Block 3: summary metrics card ----------
    quad_active = float(np.sqrt(sum(r[1] ** 2 for r in active_rows)))
    reported = state.opd_budget.total_rms_opd_nm or 0.0
    largest = max(active_rows, key=lambda r: r[1]) if active_rows else None

    card_top = 2.60
    card_bot = 0.78
    _add_filled_rect(fig, _MARGIN_L, card_bot,
                     _PAGE_W - _MARGIN_R, card_top, _STRIPE)
    _add_rule(fig, _MARGIN_L, _PAGE_W - _MARGIN_R, card_top + 0.005,
              color=_ACCENT, lw=2.0)
    _ftext(fig, _MARGIN_L + 0.20, card_top - 0.22,
           "OPD BUDGET SUMMARY",
           fontsize=7.5, color=_ACCENT, family="sans-serif",
           weight="bold")

    metrics = [
        ("Reported total", f"{reported:.2f} nm",
         "OPDBudget.total_rms_opd_nm"),
        ("Quadrature (active)", f"{quad_active:.2f} nm",
         f"{len(active_rows)} components"),
        ("Largest active",
         f"{largest[1]:.2f} nm" if largest else "n/a",
         largest[0] if largest else ""),
        ("Dominant fraction",
         f"{(largest[1] / quad_active * 100):.1f} %" if largest and quad_active > 0 else "n/a",
         "of quadrature sum"),
    ]
    cell_w = (_PAGE_W - _MARGIN_L - _MARGIN_R) / len(metrics)
    for i, (label, value, sub) in enumerate(metrics):
        cx = _MARGIN_L + (i + 0.5) * cell_w
        _ftext(fig, cx, card_top - 0.50, label.upper(),
               fontsize=7.5, color=_MUTE, family="sans-serif",
               weight="bold", ha="center", va="baseline")
        _ftext(fig, cx, card_top - 0.85, value,
               fontsize=14, color=_INK, family="serif",
               ha="center", va="baseline")
        if sub:
            _ftext(fig, cx, card_top - 1.10, sub,
                   fontsize=7.5, color=_MUTE, family="sans-serif",
                   ha="center", va="baseline")

    _footer(fig, "opd budget rollup")
    return fig


# ---------------------------------------------------------------------------
# Detector & performance rollup
# ---------------------------------------------------------------------------

def _detector_page(record: RunRecord, page_num: int, total: int) -> plt.Figure:
    state = record.result.state
    rates = np.asarray(state.photon_rates.rates_hz, dtype=np.float64)
    labels = state.photon_rates.port_labels

    if rates.ndim == 3:
        port_total = rates.sum(axis=(1, 2))
    elif rates.ndim == 2:
        port_total = rates.sum(axis=1)
    else:
        port_total = rates
    n_ports = port_total.size

    is_bright, is_dark, is_spectro, other = _port_kind_masks(labels)

    fig = _new_page()
    _header(fig, "detector · performance",
            "Photon flux & null-depth performance",
            page_num, total)
    _ftext(fig, _MARGIN_L, _TITLE_Y - 0.28,
           _detector_layout_caption(state, labels, n_ports),
           fontsize=10, color=_MUTE, family="sans-serif",
           ha="left", va="baseline")

    # ---------- Block 1: per-port photon rate ----------
    _block_label(fig, _CONTENT_TOP, "Detected per-port photon rate",
                 kicker="post-saturation  ·  summed over wavelength")
    ax = _add_axes_in(fig, _MARGIN_L + 0.10, 5.95,
                      _PAGE_W - _MARGIN_R - 0.05, 9.15)
    ax.set_facecolor(_PAPER)
    x = np.arange(n_ports)
    ax.bar(x[is_bright], port_total[is_bright], color=_BRIGHT_PORT,
           label=f"bright ({int(is_bright.sum())})",
           edgecolor=_INK, linewidth=0.3, width=0.85)
    ax.bar(x[is_dark], port_total[is_dark], color=_DARK_PORT,
           label=f"dark ({int(is_dark.sum())})",
           edgecolor=_INK, linewidth=0.3, width=0.85)
    if is_spectro.any():
        ax.bar(x[is_spectro], port_total[is_spectro], color=_GOOD,
               label=f"spectro ({int(is_spectro.sum())})",
               edgecolor=_INK, linewidth=0.3, width=0.85)
    if other.any():
        ax.bar(x[other], port_total[other], color=_MUTE,
               label=f"other ({int(other.sum())})",
               edgecolor=_INK, linewidth=0.3, width=0.85)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(_log_formatter())
    ax.set_xlabel("Physical detector-port index")
    ax.set_ylabel("Photon rate (Hz)")
    ax.grid(True, axis="y", color=_RULE, linewidth=0.4, alpha=0.7, which="both")
    ax.set_axisbelow(True)

    # Saturation overlay if any
    saturated = None
    for payload in state.results.components.values():
        if not isinstance(payload, dict):
            continue
        sat = payload.get("saturated_per_port")
        if sat is None:
            sat = payload.get("saturated_ports")
        if sat is not None:
            saturated = np.asarray(sat, dtype=bool)
            break
    if saturated is not None and saturated.size == n_ports and saturated.any():
        sat_idx = np.where(saturated)[0]
        y_min = port_total[port_total > 0].min() if (port_total > 0).any() else 1.0
        ax.scatter(sat_idx, np.full_like(sat_idx, y_min, dtype=float) * 0.4,
                   marker="^", color=_BAD, s=24, zorder=5,
                   label=f"saturated ({int(saturated.sum())})")
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    # ---------- Block 2: bright vs dark flux split ----------
    bright_total = float(port_total[is_bright].sum())
    dark_total = float(port_total[is_dark].sum())
    mean_bright = (float(np.mean(port_total[is_bright]))
                   if is_bright.any() else 0.0)
    mean_dark = (float(np.mean(port_total[is_dark]))
                 if is_dark.any() else 0.0)
    observed_null = (mean_dark / mean_bright) if mean_bright > 0 else float("nan")

    _block_label(fig, 5.45, "Bright vs dark integrated flux")
    ax2 = _add_axes_in(fig, _MARGIN_L + 0.10, 4.05,
                       _PAGE_W - _MARGIN_R - 0.05, 5.15)
    _inset_for_ylabels(ax2, inches=1.10)
    ax2.set_facecolor(_PAPER)
    ax2.barh([0], [bright_total], color=_BRIGHT_PORT, height=0.55,
             edgecolor=_INK, linewidth=0.4)
    ax2.barh([1], [dark_total], color=_DARK_PORT, height=0.55,
             edgecolor=_INK, linewidth=0.4)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Bright (sum)", "Dark (sum)"], fontsize=9)
    ax2.set_xscale("log")
    ax2.xaxis.set_major_formatter(_log_formatter())
    ax2.set_xlabel("Photon rate (Hz)")
    ax2.grid(True, axis="x", color=_RULE, linewidth=0.4, alpha=0.7, which="both")
    ax2.set_axisbelow(True)
    if bright_total > 0:
        ax2.text(bright_total, 0, f"  {bright_total:.3e} Hz",
                 va="center", ha="left", fontsize=8, color=_INK,
                 family="sans-serif")
    if dark_total > 0:
        ax2.text(dark_total, 1, f"  {dark_total:.3e} Hz",
                 va="center", ha="left", fontsize=8, color=_INK,
                 family="sans-serif")

    # ---------- Block 3: headline performance card ----------
    results = dict(state.results.components)
    sens = results.get("sensitivity") or results.get("sensitivity.detection_curve") or {}
    perf = results.get("performance") or results.get("performance.standard") or {}
    calib = results.get("calibration") or results.get("calibration.floor_lookup") or {}
    detector = next((v for k, v in results.items() if k.startswith("detector")), {})

    null_raw = _first_finite_float(
        calib.get("null_depth_raw") if isinstance(calib, dict) else None,
        detector.get("null_depth_raw") if isinstance(detector, dict) else None,
    )
    null_cal = _first_finite_float(
        calib.get("null_depth_residual", calib.get("null_depth_calibrated"))
        if isinstance(calib, dict) else None,
    )
    final_null = _first_finite_float(
        perf.get("null_depth_effective") if isinstance(perf, dict) else None,
        state.results.null_depth,
        sens.get("effective_null_with_disk") if isinstance(sens, dict) else None,
        null_cal,
    )
    snr = perf.get("snr") if isinstance(perf, dict) else None
    int_time = perf.get("integration_time_s") if isinstance(perf, dict) else None

    card_top = 3.65
    card_bot = 0.78
    _add_filled_rect(fig, _MARGIN_L, card_bot,
                     _PAGE_W - _MARGIN_R, card_top, _STRIPE)
    _add_rule(fig, _MARGIN_L, _PAGE_W - _MARGIN_R, card_top + 0.005,
              color=_ACCENT, lw=2.0)
    _ftext(fig, _MARGIN_L + 0.20, card_top - 0.22,
           "PERFORMANCE",
           fontsize=7.5, color=_ACCENT, family="sans-serif",
           weight="bold")

    metrics_row1 = [
        ("Detector raw null",
         f"{null_raw:.3e}" if null_raw is not None else "n/a",
         "diagnostic, pre-cap"),
        ("Final science null",
         f"{final_null:.3e}" if final_null is not None else "n/a",
         "used for SNR"),
        ("Observed mean(d)/mean(b)",
         f"{observed_null:.3e}" if not np.isnan(observed_null) else "n/a",
         "post-detector ratio"),
    ]
    cell_w1 = (_PAGE_W - _MARGIN_L - _MARGIN_R) / len(metrics_row1)
    for i, (lab, val, sub) in enumerate(metrics_row1):
        cx = _MARGIN_L + (i + 0.5) * cell_w1
        _ftext(fig, cx, card_top - 0.55, lab.upper(),
               fontsize=7.5, color=_MUTE, family="sans-serif",
               weight="bold", ha="center", va="baseline")
        _ftext(fig, cx, card_top - 0.95, val,
               fontsize=14, color=_INK, family="serif",
               ha="center", va="baseline")
        if sub:
            _ftext(fig, cx, card_top - 1.20, sub,
                   fontsize=7.5, color=_MUTE, family="sans-serif",
                   ha="center", va="baseline")

    _add_rule(fig, _MARGIN_L + 0.30, _PAGE_W - _MARGIN_R - 0.30,
              card_top - 1.55, color=_RULE, lw=0.5)

    metrics_row2 = [
        ("SNR", f"{snr:.2f}" if snr is not None else "n/a", ""),
        ("Integration", _human_time(int_time) if int_time is not None else "n/a", ""),
        ("Bright total", f"{bright_total:.2e} Hz", _human_count(bright_total) + " Hz"),
        ("Dark total",   f"{dark_total:.2e} Hz",   _human_count(dark_total) + " Hz"),
    ]
    cell_w2 = (_PAGE_W - _MARGIN_L - _MARGIN_R) / len(metrics_row2)
    for i, (lab, val, sub) in enumerate(metrics_row2):
        cx = _MARGIN_L + (i + 0.5) * cell_w2
        _ftext(fig, cx, card_top - 1.85, lab.upper(),
               fontsize=7.5, color=_MUTE, family="sans-serif",
               weight="bold", ha="center", va="baseline")
        _ftext(fig, cx, card_top - 2.20, val,
               fontsize=12, color=_INK, family="serif",
               ha="center", va="baseline")
        if sub:
            _ftext(fig, cx, card_top - 2.42, sub,
                   fontsize=7, color=_MUTE, family="sans-serif",
                   ha="center", va="baseline")

    # Two leakage truths: the static-chip dark-port total the detector stage
    # reported vs the realized leakage the sensitivity MC publishes. SNR uses
    # the realized value; the static-chip number is a chip-only diagnostic.
    leak_static = (
        sens.get("leakage_rate_hz_static_chip") if isinstance(sens, dict) else None
    )
    leak_effective = (
        sens.get("leakage_rate_hz_effective") if isinstance(sens, dict) else None
    )
    leak_ratio = (
        sens.get("effective_to_static_leakage_ratio") if isinstance(sens, dict) else None
    )
    if leak_static is not None or leak_effective is not None:
        static_str = (
            f"{float(leak_static):.2e} Hz" if leak_static is not None else "n/a"
        )
        effective_str = (
            f"{float(leak_effective):.2e} Hz"
            if leak_effective is not None
            else "n/a"
        )
        ratio_str = (
            f" (×{float(leak_ratio):.1f} above static)"
            if isinstance(leak_ratio, (int, float))
            else ""
        )
        _ftext(
            fig,
            _MARGIN_L + 0.20,
            card_top - 2.72,
            f"static-chip leakage (detector stage) = {static_str}   ·   "
            f"realized leakage (sensitivity MC) = {effective_str}{ratio_str}   "
            "— SNR uses the realized value",
            fontsize=7, color=_MUTE, family="sans-serif",
            ha="left", va="baseline",
        )

    _footer(fig, "detector · performance")
    return fig


def _assumptions_page(record: RunRecord, page_num: int, total: int) -> plt.Figure:
    """Final text page: key modeling assumptions and limits, with live values.

    States the model boundaries a reader needs before trusting the numbers:
    the frequency-domain (no time-domain servo) fringe-tracking budget, the
    residual-after-feedforward vibration inputs, the IID vs rank-1 common
    piston treatment, what the effective null is assembled from, which
    systematic terms the SNR uses, the disk-floor caveat, and the
    [instrument] port topology actually simulated.
    """
    state = record.result.state
    results = dict(state.results.components)
    ft = next(
        (v for k, v in results.items()
         if k.startswith("fringe_tracking") and isinstance(v, dict)),
        {},
    )
    sens = (
        results.get("sensitivity")
        or results.get("sensitivity.detection_curve")
        or {}
    )
    perf = (
        results.get("performance")
        or results.get("performance.standard")
        or {}
    )
    chip_opt = results.get("chip_optimization") or {}
    ao = state.wavefront.components.get("ao_residual")
    ao = ao if isinstance(ao, dict) else {}

    def _num(payload: Any, key: str, fmt: str = "{:.3g}") -> str:
        value = payload.get(key) if isinstance(payload, dict) else None
        return fmt.format(float(value)) if isinstance(value, (int, float)) else "n/a"

    sections: list[tuple[str, list[str]]] = []

    # ---- Fringe tracking & piston control ----
    ft_bullets: list[str] = []
    if ft:
        control_model = str(ft.get("control_model", "classical"))
        ft_bullets.append(
            "Frequency-domain residual budget only — there is no time-domain "
            f"servo simulation anywhere in the pipeline. The '{control_model}' "
            "loop trades photon-estimation noise against disturbance rejection "
            "analytically."
        )
        if control_model == "photon_kalman":
            ft_bullets.append(
                "The photon-Kalman model is a boxcar estimator plus "
                "structure-function prediction — conservative relative to a true "
                f"Kalman smoother (window = {_num(ft, 'photon_kalman_window_s', '{:.3g}')} s, "
                f"f_3dB,equiv = {_num(ft, 'photon_kalman_f_3db_equiv_hz', '{:.0f}')} Hz)."
            )
        n_lines = ft.get("tracked_line_count")
        if isinstance(n_lines, (int, float)) and n_lines:
            ft_bullets.append(
                f"The {int(n_lines)}-line vibration spectrum is the RESIDUAL after "
                "the observatory's accelerometer feedforward (x5-6 suppression "
                "assumed: Bohm+ 2014, Ertel+ 2020); broadband floor "
                f"{_num(ft, 'vibration_broadband_rms_effective_nm', '{:.1f}')} nm "
                "effective after fusion. Lines are stationary inputs — no slews or "
                "wind-shake transients."
            )
        ft_bullets.append(
            f"Per-mode FT residual ({_num(ft, 'residual_iid_per_mode_nm', '{:.2f}')} nm) "
            f"is IID across the {ft.get('n_modes', '?')} modes and averages down; the "
            f"common differential piston ({_num(ft, 'residual_common_differential_nm', '{:.2f}')} nm) "
            "is a rank-1 ±telescope-sign mode that does NOT average down and is "
            "propagated through the realized-null MC as a covariance."
        )
        eff_wind = ao.get("effective_wind_speed_m_s")
        if isinstance(eff_wind, (int, float)):
            ft_bullets.append(
                f"Temporal-lag wind is the Greenwood effective wind from tau0: "
                f"{float(eff_wind):.1f} m/s "
                f"(tau0 = {_num(ao, 'tau0_500nm_s', '{:.3g}')} s, "
                f"r0 = {_num(ao, 'r0_500nm_m', '{:.2f}')} m at 500 nm) — not the "
                "configured site wind."
            )
        superseded = [
            name for name, payload in state.opd_budget.components.items()
            if getattr(payload, "superseded", False)
            or (isinstance(payload, dict) and payload.get("superseded", False))
        ]
        if superseded:
            ft_bullets.append(
                "In the OPD budget the FT residual supersedes (replaces, not adds "
                "to): " + ", ".join(superseded) + ". The reported total uses only "
                "active components."
            )
    else:
        ft_bullets.append("No fringe-tracking stage in this pipeline.")
    sections.append(("Fringe tracking & piston control", ft_bullets))

    # ---- Null depth & systematics ----
    null_bullets: list[str] = []
    if sens:
        null_bullets.append(
            "Effective null is assembled additively: realized "
            f"({_num(sens, 'realized_null_depth', '{:.2e}')}) + stellar-disk floor "
            f"({_num(sens, 'disk_null_floor', '{:.2e}')}) + setpoint trim "
            f"({_num(sens, 'null_setpoint_leakage', '{:.2e}')}) + setpoint dither "
            f"({_num(sens, 'setpoint_dither_leakage', '{:.2e}')}) = "
            f"{_num(sens, 'effective_null_with_disk', '{:.2e}')}."
        )
        null_bullets.append(
            "Two leakage truths: the detector-stage static-chip rate "
            f"({_num(sens, 'leakage_rate_hz_static_chip', '{:.2e}')} Hz) is a "
            "chip-only diagnostic; SNR uses the realized effective rate "
            f"({_num(sens, 'leakage_rate_hz_effective', '{:.2e}')} Hz, "
            f"×{_num(sens, 'effective_to_static_leakage_ratio', '{:.1f}')})."
        )
    if perf:
        null_bullets.append(
            f"eps_systematic = hypot(null_instability ({_num(sens, 'null_instability', '{:.3f}')}) "
            f"/ NSC ({_num(perf, 'nsc_systematic_reduction', '{:g}')}), residual speckle "
            f"stability ({_num(perf, 'residual_speckle_stability_factor', '{:g}')})) = "
            f"{_num(perf, 'eps_systematic_used', '{:.4f}')}. NSC reduces ONLY the "
            "null-instability systematic; mean-leakage photon noise is unchanged."
        )
    disk_note = chip_opt.get("disk_null_floor_note") if isinstance(chip_opt, dict) else None
    if disk_note:
        null_bullets.append(
            f"Stellar-disk floor caveat (chip optimizer): {disk_note}. The floor "
            f"is evaluated at a stellar diameter of "
            f"{_num(sens, 'stellar_diameter_mas', '{:.2f}')} mas and moves if the "
            "chip is re-optimized."
        )
    cal_applied = perf.get("calibration_floor_applied") if isinstance(perf, dict) else None
    if cal_applied is not None:
        null_bullets.append(
            "Calibration floor "
            + ("applied" if cal_applied else "NOT applied")
            + f" on this run (source: {perf.get('calibration_floor_source') or 'none'})."
        )
    if not null_bullets:
        null_bullets.append("No sensitivity/performance payloads on this run.")
    sections.append(("Null depth & systematics", null_bullets))

    # ---- Instrument topology ----
    inst = getattr(record.config, "instrument", None)
    inst_bullets: list[str] = []
    if inst is not None and any(
        getattr(inst, f, None) is not None
        for f in ("n_modes", "n_input_modes", "n_bright_ports",
                  "n_dark_ports", "n_spectro_ports")
    ):
        def _inst(field: str) -> str:
            value = getattr(inst, field, None)
            return str(value) if value is not None else "unset"
        inst_bullets.append(
            f"[instrument] port topology: n_modes = {_inst('n_modes')} (chip), "
            f"n_input_modes = {_inst('n_input_modes')}, ports = "
            f"{_inst('n_bright_ports')} bright / {_inst('n_dark_ports')} dark / "
            f"{_inst('n_spectro_ports')} spectro — materialized into all "
            "port-bearing stages."
        )
    else:
        inst_bullets.append(
            "No [instrument] section — each stage declares its own port counts."
        )
    if isinstance(sens, dict) and sens.get("chip_component"):
        inst_bullets.append(
            f"Chip: {sens.get('chip_component')} "
            f"(deployment {sens.get('chip_deployment_mode', '?')}, "
            f"{_num(sens, 'chip_n_input_modes', '{:g}')} inputs -> "
            f"{_num(sens, 'chip_n_output_ports', '{:g}')} output ports); detector: "
            f"{perf.get('upstream_detector_component', 'n/a') if isinstance(perf, dict) else 'n/a'}."
        )
    sections.append(("Instrument topology", inst_bullets))

    # ---- Render ----
    fig = _new_page()
    _header(fig, "assumptions & limits", "Model assumptions and limits",
            page_num, total)
    _ftext(fig, _MARGIN_L, _TITLE_Y - 0.28,
           "What this simulation does and does not model · values pulled live from this run",
           fontsize=10, color=_MUTE, family="sans-serif",
           ha="left", va="baseline")

    line_h = 0.165
    y = _CONTENT_TOP
    truncated = False
    for title, bullets in sections:
        if y < _CONTENT_BOTTOM + 0.8:
            truncated = True
            break
        _block_label(fig, y, title)
        y -= 0.42
        for bullet in bullets:
            wrapped = textwrap.wrap(bullet, width=104) or [""]
            if y - line_h * len(wrapped) < _CONTENT_BOTTOM:
                truncated = True
                break
            for j, line in enumerate(wrapped):
                prefix = "—  " if j == 0 else "    "
                _ftext(fig, _MARGIN_L + 0.10, y, prefix + line,
                       fontsize=8.5, color=_INK, family="sans-serif",
                       ha="left", va="top")
                y -= line_h
            y -= 0.10
        y -= 0.28
    if truncated:
        _ftext(fig, _MARGIN_L + 0.10, _CONTENT_BOTTOM + 0.05,
               "(further entries truncated)", fontsize=7.5, color=_MUTE,
               family="sans-serif", style="italic", ha="left", va="bottom")

    _footer(fig, "assumptions & limits")
    return fig


def _null_waterfall_page(record: RunRecord, page_num: int, total: int) -> plt.Figure:
    """Where the contrast budget goes, in one chart.

    Walks the per-stage null-depth contributions surfaced by the
    sensitivity / detector / calibration stages and renders them as a
    horizontal log-scale bar chart with a KILO reference marker for the
    ideal-chip term. Pre-Phase-A this story was scattered across the
    sensitivity / performance text dumps; the waterfall keeps it on a
    single page.
    """
    state = record.result.state
    results = dict(state.results.components)

    sens = (
        results.get("sensitivity")
        or results.get("sensitivity.detection_curve")
        or {}
    )
    calib = (
        results.get("calibration")
        or results.get("calibration.floor_lookup")
        or {}
    )
    detector = next(
        (v for k, v in results.items() if k.startswith("detector") and isinstance(v, dict)),
        {},
    )
    perf = (
        results.get("performance")
        or results.get("performance.standard")
        or {}
    )

    rows: list[tuple[str, float | None, str]] = []
    chip_ideal = sens.get("ideal_chip_null_depth") if isinstance(sens, dict) else None
    chip_ideal_kilo = (
        sens.get("ideal_chip_null_depth_kilo") if isinstance(sens, dict) else None
    )
    realized = sens.get("realized_null_depth") if isinstance(sens, dict) else None
    disk = sens.get("disk_null_floor") if isinstance(sens, dict) else None
    effective = sens.get("effective_null_with_disk") if isinstance(sens, dict) else None
    raw = detector.get("null_depth_raw") if isinstance(detector, dict) else None
    cal_floor = calib.get("calibration_floor") if isinstance(calib, dict) else None
    cal_null = (
        calib.get("null_depth_residual", calib.get("null_depth_calibrated"))
        if isinstance(calib, dict) else None
    )
    perf_effective = (
        perf.get("null_depth_effective") if isinstance(perf, dict) else None
    )
    final_null = _first_finite_float(
        perf_effective,
        state.results.null_depth,
        effective,
        cal_null,
    )

    setpoint_leak = (
        sens.get("null_setpoint_leakage") if isinstance(sens, dict) else None
    )
    dither_leak = (
        sens.get("setpoint_dither_leakage") if isinstance(sens, dict) else None
    )
    rows.append(("Chip ideal (no AO/scintillation)",
                 _as_float_or_none(chip_ideal), "chip"))
    rows.append(("Realized null  (+AO piston + scintillation)",
                 _as_float_or_none(realized), "ao"))
    rows.append(("Stellar-disk floor  (partial-disk leakage)",
                 _as_float_or_none(disk), "disk"))
    if setpoint_leak is not None and float(setpoint_leak) > 0:
        rows.append(("Setpoint-trim leakage  (dark-port bias)",
                     _as_float_or_none(setpoint_leak), "setpoint"))
    if dither_leak is not None and float(dither_leak) > 0:
        rows.append(("Setpoint dither leakage  (trim modulation)",
                     _as_float_or_none(dither_leak), "setpoint"))
    rows.append(("Effective null  (realized+disk+setpoint)",
                 _as_float_or_none(effective), "effective"))
    if raw is not None:
        rows.append(("Detector null_depth_raw (diagnostic, pre-cap)",
                     _as_float_or_none(raw), "detector"))
    if cal_floor is not None:
        rows.append(("Calibration floor (subtracted)",
                     _as_float_or_none(cal_floor), "cal_floor"))
    if cal_null is not None:
        rows.append(("Detector raw minus calibration floor (diagnostic)",
                     _as_float_or_none(cal_null), "calibrated"))
    if final_null is not None:
        rows.append(("Perf. null_depth_effective (used for SNR)",
                     final_null, "final"))

    fig = _new_page()
    _header(fig, "NULL-DEPTH WATERFALL",
            "Null-depth budget", page_num, total)
    _ftext(fig, _MARGIN_L, _TITLE_Y - 0.28,
           "Where the contrast budget goes  ·  one bar per stage, log scale",
           fontsize=10, color=_MUTE, family="sans-serif",
           ha="left", va="baseline")

    finite_rows = [(label, val, tag) for label, val, tag in rows if val is not None and val > 0]
    if not finite_rows:
        _ftext(fig, _PAGE_W / 2, 5.5,
               "No null-depth components present on this run",
               fontsize=12, color=_MUTE, family="serif",
               ha="center", va="center")
        _footer(fig, "null waterfall")
        return fig

    # Main bar chart: y-axis labels carry the stage name + numeric value;
    # bars run on a log-x.
    ax = _add_axes_in(
        fig, _MARGIN_L + 2.35, 2.95, _PAGE_W - _MARGIN_R - 0.10, 7.6,
    )
    ax.set_facecolor(_PAPER)
    y_pos = np.arange(len(finite_rows))[::-1]
    vals = [v for _, v, _ in finite_rows]

    tag_to_color = {
        "chip": _ACCENT,
        "ao": _DARK_PORT,
        "disk": _COOL,
        "setpoint": _WARN,
        "effective": _MUTE,
        "detector": _MUTE,
        "cal_floor": _GOOD,
        "calibrated": _MUTE,
        "final": _ACCENT,
    }
    colors = [tag_to_color.get(r[2], _MUTE) for r in finite_rows]
    bars = ax.barh(y_pos, vals, color=colors, edgecolor="none", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r[0] for r in finite_rows], fontsize=8.5, family="serif")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(_log_formatter())
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs="auto", numticks=12))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
    ax.set_xlabel("Null depth (fractional, log)")
    ax.grid(True, axis="x", which="both", color=_RULE, linewidth=0.4, alpha=0.6)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() * 1.05, bar.get_y() + bar.get_height() / 2,
                f"{v:.2e}", va="center", ha="left",
                fontsize=8, color=_INK, family="monospace")

    # KILO reference overlay: pin null_val_dark as a dashed vertical line
    # against the chip-ideal bar.
    kilo_ref = _kilo_null_val_dark()
    if kilo_ref is not None:
        ax.axvline(kilo_ref, color=_GOOD, linewidth=1.0, linestyle="--", alpha=0.7)
        ax.text(kilo_ref, len(finite_rows) - 0.4,
                f"KILO null_val_dark = {kilo_ref:.2e}",
                rotation=90, va="bottom", ha="right",
                fontsize=7.5, color=_GOOD, family="monospace")
        badge_value = (
            chip_ideal_kilo
            if isinstance(chip_ideal_kilo, (int, float)) and chip_ideal_kilo > 0
            else chip_ideal
        )
        if isinstance(badge_value, (int, float)) and badge_value > 0:
            # KILO-convention scalar when available (review 2026-06-10).
            ratio = float(badge_value) / kilo_ref
            verdict = "OK" if ratio <= 2.0 else f"×{ratio:.1f} off"
            _ftext(fig, _PAGE_W - _MARGIN_R, _TITLE_Y - 0.28,
                   f"chip_ideal_kilo / KILO = {verdict}",
                   fontsize=9, color=(_GOOD if ratio <= 2.0 else _ACCENT),
                   family="monospace", ha="right", va="baseline")

    # Headline card at the bottom.
    if perf:
        snr = perf.get("snr") if isinstance(perf, dict) else None
        int_time = perf.get("integration_time_for_target_snr_s") if isinstance(perf, dict) else None
        target_snr = perf.get("target_snr") if isinstance(perf, dict) else None
        instab = sens.get("null_instability") if isinstance(sens, dict) else None
        eps_used = perf.get("eps_systematic_used") if isinstance(perf, dict) else None
        nsc = perf.get("nsc_systematic_reduction") if isinstance(perf, dict) else None
        speckle = (
            perf.get("residual_speckle_stability_factor")
            if isinstance(perf, dict) else None
        )
        card_top = 2.45
        card_bot = 0.78
        _add_filled_rect(fig, _MARGIN_L, card_bot,
                         _PAGE_W - _MARGIN_R, card_top, _STRIPE)
        _add_rule(fig, _MARGIN_L, _PAGE_W - _MARGIN_R, card_top + 0.005,
                  color=_ACCENT, lw=2.0)
        _ftext(fig, _MARGIN_L + 0.20, card_top - 0.20,
               "BUDGET SUMMARY", fontsize=7.5, color=_ACCENT,
               family="sans-serif", weight="bold")
        metrics = [
            ("final null",
             f"{final_null:.3e}" if final_null is not None else "n/a", ""),
            ("SNR", f"{snr:.2f}" if isinstance(snr, (int, float)) else "n/a",
             f"target = {target_snr:g}" if target_snr else ""),
            ("t for target",
             _human_time(int_time) if isinstance(int_time, (int, float)) else "n/a",
             "integration"),
        ]
        cell_w = (_PAGE_W - _MARGIN_L - _MARGIN_R) / len(metrics)
        for i, (label, value, sub) in enumerate(metrics):
            cx = _MARGIN_L + (i + 0.5) * cell_w
            _ftext(fig, cx, card_top - 0.45, label.upper(),
                   fontsize=7.5, color=_MUTE, family="sans-serif",
                   weight="bold", ha="center", va="baseline")
            _ftext(fig, cx, card_top - 0.72, value,
                   fontsize=13, color=_INK, family="serif",
                   ha="center", va="baseline")
            if sub:
                _ftext(fig, cx, card_top - 0.92, sub,
                       fontsize=7.5, color=_MUTE, family="sans-serif",
                       ha="center", va="baseline")
        # Systematic-eps row: which terms the systematic SNR variance uses.
        metrics2 = [
            ("null instability (MC eps)",
             f"{instab:.3f}" if isinstance(instab, (int, float)) else "n/a"),
            ("eps_systematic used",
             f"{eps_used:.4f}" if isinstance(eps_used, (int, float)) else "n/a"),
            ("NSC reduction",
             f"÷{nsc:g}" if isinstance(nsc, (int, float)) else "n/a"),
        ]
        cell_w2 = (_PAGE_W - _MARGIN_L - _MARGIN_R) / len(metrics2)
        for i, (label, value) in enumerate(metrics2):
            cx = _MARGIN_L + (i + 0.5) * cell_w2
            _ftext(fig, cx, card_top - 1.15, label.upper(),
                   fontsize=7, color=_MUTE, family="sans-serif",
                   weight="bold", ha="center", va="baseline")
            _ftext(fig, cx, card_top - 1.38, value,
                   fontsize=10.5, color=_INK, family="serif",
                   ha="center", va="baseline")
        speckle_str = (
            f"{speckle:g}" if isinstance(speckle, (int, float)) else "n/a"
        )
        _ftext(fig, _MARGIN_L + 0.20, card_top - 1.58,
               "eps_systematic = hypot(null_instability / NSC, residual_speckle_stability "
               f"= {speckle_str}) — NSC reduces only the instability term; "
               "mean-leakage photon noise is unchanged",
               fontsize=6.8, color=_MUTE, family="sans-serif",
               ha="left", va="baseline")
    _footer(fig, "null waterfall")
    return fig


def _as_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _first_finite_float(*values: Any) -> float | None:
    """Return the first finite value from a priority-ordered list."""

    for value in values:
        finite = _as_float_or_none(value)
        if finite is not None:
            return finite
    return None


_COORD_SHORT_NAMES: dict[str, str] = {
    "scene.star.jmag": "J",
}


def _canon_coord_value(v: Any) -> tuple:
    """Hashable canonical form of a sweep coordinate value.

    Floats are rounded to 12 decimal places so that binary-roundoff noise
    does not split otherwise-identical coordinate series across groups.
    """

    if isinstance(v, float):
        if not np.isfinite(v):
            return ("f_nan",)
        return ("f", round(float(v), 12))
    if isinstance(v, (list, tuple)):
        return ("seq", tuple(_canon_coord_value(x) for x in v))
    return ("o", v)


def _partition_signature(series_values: list[Any]) -> tuple[int, ...]:
    """Canonical signature of how a series partitions cell indices.

    Two axes share a signature iff they group cells identically. Used to
    fold zipped sweep axes (e.g. ``N_sub=[18,36]`` and
    ``n_bright_ports=[4,9]``) onto a single representative — the values
    differ but the cell partition is the same.
    """

    seen: dict[tuple, int] = {}
    sig: list[int] = []
    for v in series_values:
        cv = _canon_coord_value(v)
        if cv not in seen:
            seen[cv] = len(seen)
        sig.append(seen[cv])
    return tuple(sig)


def _shorten_coord_key(key: str) -> str:
    if key in _COORD_SHORT_NAMES:
        return _COORD_SHORT_NAMES[key]
    if key.startswith("pipeline.stages."):
        rest = key[len("pipeline.stages."):]
        # ``<stage_id>.<field>`` — almost always the field is what carries
        # information. NAYRA-style configs use long descriptive stage ids.
        if "." in rest:
            field = rest.split(".", 1)[1]
            if field == "n_subapertures_per_aperture":
                return "N_sub"
            return field
        return rest
    return key


def _short_coord_value(v: Any) -> str:
    if isinstance(v, float):
        if np.isfinite(v) and float(v).is_integer():
            return str(int(round(v)))
        return f"{v:g}"
    return str(v)


def _coord_dedupe(summary: dict[str, list[Any]]) -> tuple[list[str], list[str]]:
    """Build compact per-cell coord labels and a short axis title.

    Three reductions run in order:

    1. drop axes that are constant across the sweep,
    2. fold axes that partition cells identically (zipped sweep axes
       where the *partition* matches even though the *values* differ —
       e.g. ``n_subapertures_per_aperture=[18,36]`` and
       ``n_bright_ports=[4,9]`` zipped together),
    3. strip the verbose ``pipeline.stages.<id>.`` prefix in the label.

    The summary also has ``coord_axis_reps`` populated so other plots
    (group labels, etc.) can reuse the dedupe result.
    """

    coord_dicts = summary.get("coord_value") or []
    if not coord_dicts:
        summary["coord_axis_reps"] = []
        return ["primary"] * len(coord_dicts), []
    all_keys = sorted({
        k for c in coord_dicts if isinstance(c, dict) for k in c.keys()
    })
    if not all_keys:
        summary["coord_axis_reps"] = []
        return ["primary"] * len(coord_dicts), []
    series = {
        k: [c.get(k) if isinstance(c, dict) else None for c in coord_dicts]
        for k in all_keys
    }
    varying = [
        k for k in all_keys
        if len({_canon_coord_value(v) for v in series[k]}) > 1
    ]
    # Partition-signature dedupe: two axes with the same partition of cell
    # indices (i.e. perfectly correlated value series) collapse together.
    seen: dict[tuple, list[str]] = {}
    group_order: list[tuple] = []
    for k in varying:
        sig = _partition_signature(series[k])
        if sig in seen:
            seen[sig].append(k)
        else:
            seen[sig] = [k]
            group_order.append(sig)
    representatives: list[str] = []
    for sig in group_order:
        keys = seen[sig]
        # Stable preference: J first, then any subaperture-count axis,
        # then shortest readable name.
        keys_sorted = sorted(keys, key=lambda k: (
            0 if k == "scene.star.jmag"
            else 1 if "n_subapertures_per_aperture" in k
            else 2,
            len(k),
            k,
        ))
        representatives.append(keys_sorted[0])
    short_names = [_shorten_coord_key(r) for r in representatives]
    labels: list[str] = []
    for i in range(len(coord_dicts)):
        bits = [
            f"{short_names[j]}={_short_coord_value(series[representatives[j]][i])}"
            for j in range(len(representatives))
        ]
        labels.append(", ".join(bits) if bits else "primary")
    summary["coord_axis_reps"] = list(representatives)
    return labels, short_names


def _group_label_from_coord(rep_coord: dict[str, Any]) -> str:
    """Compress a non-primary group coordinate dict into a short label.

    Keys whose values match (under :func:`_canon_coord_value`) are folded
    onto one representative, so an N=18 group with 15 zipped axes shows
    as ``N_sub=18`` rather than the full list.
    """

    if not rep_coord:
        return ""
    by_val: dict[tuple, list[str]] = {}
    order: list[tuple] = []
    for k, v in rep_coord.items():
        cv = _canon_coord_value(v)
        if cv not in by_val:
            by_val[cv] = []
            order.append(cv)
        by_val[cv].append(k)
    bits: list[str] = []
    for cv in order:
        keys = by_val[cv]
        kr = sorted(keys, key=lambda k: (
            1 if "n_subapertures_per_aperture" in k else 2,
            len(k),
            k,
        ))[0]
        bits.append(f"{_shorten_coord_key(kr)}={_short_coord_value(rep_coord[kr])}")
    return ", ".join(bits)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

@register_plot
def debug_report(
    collection: RunCollection,
    output_dir: Path,
    formats: Iterable[str],
) -> tuple[Path, ...]:
    """Write per-cell debug PDFs and (for sweeps) a sweep-overview PDF.

    Each ``RunRecord`` in ``collection.records`` produces its own
    ``debug_report.pdf`` under ``record.run_dir / "plots"``. The
    primary cell (``records[0]``) is also mirrored to
    ``output_dir / "plots" / "debug_report.pdf"`` for back-compat with
    callers that always expected one PDF at the sweep root.

    When the run is a sweep (``len(records) > 1``), an additional
    ``debug_report_sweep.pdf`` is written at the sweep root with a
    cross-cell summary table, headline-metric trends vs the sweep
    coordinate, and contrast curves overlaid with the KILO paperv3
    reference (when the sweep coordinate is ``scene.star.jmag``).
    """
    del formats  # Debug reports are always written as PDFs.
    if not collection.records:
        return ()

    seen: set[Path] = set()
    written: list[Path] = []

    def _emit(path: Path, renderer: Callable[[Path], None]) -> None:
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        path.parent.mkdir(parents=True, exist_ok=True)
        renderer(path)
        written.append(path)

    for i, record in enumerate(collection.records):
        # Skip cells whose run failed without a real state payload.
        if record.result is None or record.result.state is None:
            continue
        per_cell_path = record.run_dir / "plots" / "debug_report.pdf"
        _emit(per_cell_path, lambda p, r=record: _write_single_cell_pdf(r, p))
        if i == 0:
            primary_path = output_dir / "plots" / "debug_report.pdf"
            _emit(primary_path, lambda p, r=record: _write_single_cell_pdf(r, p))

    if len(collection.records) > 1:
        sweep_path = output_dir / "plots" / "debug_report_sweep.pdf"
        sweep_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            _write_sweep_pdf(collection, sweep_path)
            written.append(sweep_path)
        except Exception as exc:  # noqa: BLE001 — surface, never block
            # Sweep PDF is value-add; falling back to per-cell PDFs
            # alone is acceptable. Print a hint and continue.
            import sys
            print(
                f"nullsim debug_report: sweep PDF skipped ({exc!r})",
                file=sys.stderr,
            )

    return tuple(written)


def _write_single_cell_pdf(record: RunRecord, out: Path) -> None:
    """Render one record's per-cell debug PDF to ``out``."""

    state = record.result.state
    page_callables: list[Callable[[int, int], plt.Figure]] = []
    page_callables.append(lambda pn, tot: _overview_page(record, pn, tot))
    for spec in record.config.pipeline.stages:
        if _stage_has_content(record, spec):
            page_callables.append(
                lambda pn, tot, s=spec: _stage_page(record, s, pn, tot)
            )
    if dict(state.throughput.components):
        page_callables.append(
            lambda pn, tot: _spectral_loss_page(record, pn, tot)
        )
    if dict(state.opd_budget.components):
        page_callables.append(
            lambda pn, tot: _opd_summary_page(record, pn, tot)
        )
    # The waterfall page reads sensitivity + calibration + performance.
    # Show it whenever any of those produced a payload; the page falls
    # back to a polite "no payload" message otherwise.
    sens_keys = {"sensitivity", "sensitivity.detection_curve"}
    cal_keys = {"calibration", "calibration.floor_lookup"}
    has_waterfall_input = (
        any(k in state.results.components for k in sens_keys | cal_keys)
        or any(
            k.startswith("performance") for k in state.results.components
        )
    )
    if has_waterfall_input:
        page_callables.append(
            lambda pn, tot: _null_waterfall_page(record, pn, tot)
        )
    if state.photon_rates.rates_hz is not None and state.photon_rates.port_labels:
        page_callables.append(
            lambda pn, tot: _detector_page(record, pn, tot)
        )
    # Always close with the assumptions-and-limits text page so a reader
    # leaves knowing the model boundaries, not just the numbers.
    page_callables.append(
        lambda pn, tot: _assumptions_page(record, pn, tot)
    )

    total = len(page_callables)
    with plt.rc_context(_RC_DEBUG), PdfPages(out) as pdf:
        for i, render in enumerate(page_callables, start=1):
            fig = render(i, total)
            pdf.savefig(fig, facecolor=fig.get_facecolor())
            plt.close(fig)


# ---------------------------------------------------------------------------
# Sweep-overview PDF (cross-cell summary, headline trends, KILO overlay)
# ---------------------------------------------------------------------------


def _write_sweep_pdf(collection: RunCollection, out: Path) -> None:
    """Build ``debug_report_sweep.pdf`` for a multi-cell run."""

    records = [
        r for r in collection.records
        if r.result is not None and r.result.state is not None
    ]
    if not records:
        return
    summary = _sweep_summary(records)
    pages: list[Callable[[int, int], plt.Figure]] = [
        lambda pn, tot: _sweep_cover_page(collection, records, summary, pn, tot),
        lambda pn, tot: _sweep_table_page(records, summary, pn, tot),
        lambda pn, tot: _sweep_trends_page(records, summary, pn, tot),
    ]
    if _sweep_has_jmag_axis(summary):
        pages.append(
            lambda pn, tot: _sweep_contrast_page(records, summary, pn, tot)
        )

    total = len(pages)
    with plt.rc_context(_RC_DEBUG), PdfPages(out) as pdf:
        for i, render in enumerate(pages, start=1):
            fig = render(i, total)
            pdf.savefig(fig, facecolor=fig.get_facecolor())
            plt.close(fig)


def _sweep_summary(records: list[RunRecord]) -> dict[str, list[Any]]:
    """Extract per-cell headline metrics in record order.

    Pulls ``chip_ideal``, ``realized_null``, ``disk_floor``,
    ``effective_null``, final science null, ``σ_OPD per sub-ap``,
    ``mean Strehl``, ``SNR``, ``t for SNR target`` and the canonical-
    separation contrasts (3/10/30/100/300 mas) from each record's
    results bundle. Missing values come out as ``None`` rather than
    raising so the rest of the sweep page still renders.

    After the per-cell pass, ``coord_label`` is rewritten to a compact
    form via :func:`_coord_dedupe`: zipped axes are folded together,
    constant axes are dropped, and the verbose
    ``pipeline.stages.<id>.`` prefix is stripped.
    """

    out: dict[str, list[Any]] = {
        "coord_label": [],
        "coord_value": [],
        "jmag": [],
        "chip_ideal": [],
        "chip_ideal_kilo": [],
        "realized_null": [],
        "disk_floor": [],
        "effective_null": [],
        "final_null": [],
        "sigma_opd_nm": [],
        "mean_strehl": [],
        "snr": [],
        "t_for_target_s": [],
        "contrast_curve_seps_mas": [],
        "contrast_curve_values": [],
    }
    for record in records:
        state = record.result.state
        sens = (
            state.results.components.get("sensitivity")
            or state.results.components.get("sensitivity.detection_curve")
            or {}
        )
        calib = (
            state.results.components.get("calibration")
            or state.results.components.get("calibration.floor_lookup")
            or {}
        )
        perf = (
            state.results.components.get("performance")
            or state.results.components.get("performance.standard")
            or {}
        )
        coords = record.coord_mapping
        label_bits = [f"{k}={v}" for k, v in coords.items()]
        out["coord_label"].append(", ".join(label_bits) if label_bits else "primary")
        out["coord_value"].append(coords)
        out["jmag"].append(_as_float_or_none(coords.get("scene.star.jmag")))
        out["chip_ideal"].append(
            _as_float_or_none(sens.get("ideal_chip_null_depth")) if isinstance(sens, dict) else None
        )
        out["chip_ideal_kilo"].append(
            _as_float_or_none(sens.get("ideal_chip_null_depth_kilo"))
            if isinstance(sens, dict)
            else None
        )
        out["realized_null"].append(
            _as_float_or_none(sens.get("realized_null_depth")) if isinstance(sens, dict) else None
        )
        out["disk_floor"].append(
            _as_float_or_none(sens.get("disk_null_floor")) if isinstance(sens, dict) else None
        )
        out["effective_null"].append(
            _as_float_or_none(sens.get("effective_null_with_disk")) if isinstance(sens, dict) else None
        )
        detector_residual = (
            calib.get("null_depth_residual", calib.get("null_depth_calibrated"))
            if isinstance(calib, dict) else None
        )
        out["final_null"].append(
            _first_finite_float(
                perf.get("null_depth_effective") if isinstance(perf, dict) else None,
                state.results.null_depth,
                sens.get("effective_null_with_disk") if isinstance(sens, dict) else None,
                detector_residual,
            )
        )
        # The source key is suffixed ``_m`` — convert to nm here, once,
        # instead of unit-sniffing magnitudes at render time (review
        # 2026-06-10 nit).
        sigma_opd_m = (
            _as_float_or_none(sens.get("ao_piston_opd_rms_per_sub_m"))
            if isinstance(sens, dict)
            else None
        )
        out["sigma_opd_nm"].append(
            None if sigma_opd_m is None else float(sigma_opd_m) * 1.0e9
        )
        # Mean Strehl from any wavefront component the AO stage published.
        strehl = None
        for name, payload in state.wavefront.components.items():
            if not isinstance(payload, dict):
                continue
            if name.startswith("ao") or name == "ao_residual":
                s = payload.get("strehl_ratio")
                if s:
                    strehl = float(np.mean(np.asarray(s, dtype=np.float64)))
                    break
        out["mean_strehl"].append(strehl)
        out["snr"].append(
            _as_float_or_none(perf.get("snr")) if isinstance(perf, dict) else None
        )
        out["t_for_target_s"].append(
            _as_float_or_none(perf.get("integration_time_for_target_snr_s"))
            if isinstance(perf, dict) else None
        )
        # Contrast curve — only available when sensitivity has the per-sep grid.
        seps_arr = None
        c_arr = None
        if isinstance(sens, dict):
            seps = sens.get("separations_mas")
            if seps is not None:
                try:
                    seps_arr = np.asarray(seps, dtype=np.float64)
                except (TypeError, ValueError):
                    seps_arr = None
        if isinstance(perf, dict):
            cvs = perf.get("contrast_curve_value")
            if cvs is not None:
                try:
                    c_arr = np.asarray(cvs, dtype=np.float64)
                except (TypeError, ValueError):
                    c_arr = None
        # The performance stage's contrast curve uses its own separations
        # grid; prefer that one so the shapes match.
        if isinstance(perf, dict):
            perf_seps = perf.get("contrast_curve_mas")
            if perf_seps is not None:
                with suppress(TypeError, ValueError):
                    seps_arr = np.asarray(perf_seps, dtype=np.float64)
        out["contrast_curve_seps_mas"].append(seps_arr)
        out["contrast_curve_values"].append(c_arr)

    short_labels, short_axis_keys = _coord_dedupe(out)
    out["coord_label"] = short_labels
    out["coord_short_axis"] = short_axis_keys
    return out


def _sweep_has_jmag_axis(summary: dict[str, list[Any]]) -> bool:
    """True iff every cell records a finite ``scene.star.jmag``."""
    jmags = summary.get("jmag", [])
    return bool(jmags) and all(isinstance(j, (int, float)) and j is not None for j in jmags)


def _sweep_cover_page(
    collection: RunCollection,
    records: list[RunRecord],
    summary: dict[str, list[Any]],
    page_num: int,
    total: int,
) -> plt.Figure:
    fig = _new_page()
    primary = records[0]
    _header(fig, "NULLSIM · SWEEP DEBUG REPORT",
            primary.config.run.name, page_num, total)
    _ftext(fig, _MARGIN_L, _TITLE_Y - 0.28,
           f"{len(records)} cells  ·  cross-cell summary  ·  KILO reference overlay",
           fontsize=10, color=_MUTE, family="sans-serif",
           ha="left", va="baseline")

    # Run metadata card mirrors the per-cell overview (high on the page,
    # right under the title — was 8.40-7.05 which left an awkward gap).
    meta_top = 9.30
    meta_bot = 7.95
    _add_filled_rect(fig, _MARGIN_L, meta_bot,
                     _PAGE_W - _MARGIN_R, meta_top, _STRIPE)
    _add_rule(fig, _MARGIN_L, _PAGE_W - _MARGIN_R, meta_top + 0.005,
              color=_ACCENT, lw=2.0)
    _ftext(fig, _MARGIN_L + 0.20, meta_top - 0.20,
           "RUN METADATA", fontsize=7.5, color=_ACCENT,
           family="sans-serif", weight="bold")
    rows = [
        ("RUN HASH", collection.config.run_hash),
        ("CONFIG HASH", collection.config.config_hash),
        ("FIRST CELL", primary.config.run_hash[:12]),
        ("CELLS", str(len(records))),
        ("COORD", _coord_axis_label(summary)),
    ]
    row_h = 0.22
    for i, (k, v) in enumerate(rows):
        y = meta_top - 0.55 - i * row_h
        _ftext(fig, _MARGIN_L + 0.20, y, k,
               fontsize=8, color=_MUTE, family="sans-serif", weight="bold")
        _ftext(fig, _MARGIN_L + 1.65, y, str(v),
               fontsize=9.5, color=_INK, family="monospace")

    # KILO reference context.
    kilo_ref = _kilo_null_val_dark()
    chip_ideals = [
        v for v in summary["chip_ideal_kilo"] if v is not None
    ] or [v for v in summary["chip_ideal"] if v is not None]
    median_chip = float(np.median(chip_ideals)) if chip_ideals else None
    panel_top = 7.70
    panel_bot = 6.10
    _add_filled_rect(fig, _MARGIN_L, panel_bot,
                     _PAGE_W - _MARGIN_R, panel_top, _STRIPE)
    _add_rule(fig, _MARGIN_L, _PAGE_W - _MARGIN_R, panel_top + 0.005,
              color=_ACCENT, lw=2.0)
    _ftext(fig, _MARGIN_L + 0.20, panel_top - 0.20,
           "KILO REFERENCE (chip-ideal vs paperv3 null_val_dark)",
           fontsize=7.5, color=_ACCENT,
           family="sans-serif", weight="bold")
    if kilo_ref is not None and median_chip is not None:
        ratio = median_chip / kilo_ref
        _ftext(fig, _MARGIN_L + 0.20, panel_top - 0.65,
               f"median chip_ideal_kilo = {median_chip:.3e}",
               fontsize=11, color=_INK, family="monospace")
        _ftext(fig, _MARGIN_L + 0.20, panel_top - 0.95,
               f"KILO null_val_dark = {kilo_ref:.3e}",
               fontsize=11, color=_INK, family="monospace")
        _ftext(fig, _MARGIN_L + 0.20, panel_top - 1.30,
               f"ratio to reference = x {ratio:.2f}", fontsize=13, color=_MUTE,
               family="serif", weight="bold")
    else:
        _ftext(fig, _MARGIN_L + 0.20, panel_top - 0.65,
               "(KILO reference or chip_ideal unavailable)",
               fontsize=11, color=_MUTE, family="serif")

    # Cell list as a small table.
    list_top = 5.85
    _block_label(fig, list_top, "Cells",
                 kicker=f"{len(records)} cell" + ("s" if len(records) != 1 else ""))
    col_x = {
        "idx": _MARGIN_L + 0.05,
        "coord": _MARGIN_L + 0.55,
        "chip": _MARGIN_L + 2.90,
        "real": _MARGIN_L + 4.00,
        "final": _MARGIN_L + 5.10,
        "snr": _MARGIN_L + 6.20,
    }
    hdr_y = list_top - 0.30
    _add_rule(fig, _MARGIN_L, _PAGE_W - _MARGIN_R, hdr_y + 0.16, color=_INK, lw=0.7)
    for key, label in (
        ("idx", "#"), ("coord", "COORD"), ("chip", "CHIP_IDEAL"),
        ("real", "REALIZED"), ("final", "FINAL NULL"), ("snr", "SNR"),
    ):
        _ftext(fig, col_x[key], hdr_y, label,
               fontsize=7.5, color=_MUTE, family="sans-serif",
               weight="bold", va="baseline")
    _add_rule(fig, _MARGIN_L, _PAGE_W - _MARGIN_R, hdr_y - 0.06, color=_INK, lw=0.5)
    row_h = 0.22
    for i in range(len(records)):
        y = hdr_y - 0.25 - i * row_h
        if y < 1.40:
            break
        if i % 2 == 0:
            _add_filled_rect(fig, _MARGIN_L + 0.01, y - row_h + 0.18,
                             _PAGE_W - _MARGIN_R - 0.01, y + 0.16,
                             _STRIPE, alpha=0.6)
        _ftext(fig, col_x["idx"], y, str(i),
               fontsize=8.5, color=_MUTE, family="monospace")
        _ftext(fig, col_x["coord"], y, summary["coord_label"][i],
               fontsize=8.5, color=_INK, family="monospace")
        for key, val in (
            ("chip", summary["chip_ideal"][i]),
            ("real", summary["realized_null"][i]),
            ("final", summary["final_null"][i]),
        ):
            text = f"{val:.2e}" if isinstance(val, (int, float)) else "-"
            _ftext(fig, col_x[key], y, text,
                   fontsize=8.5, color=_INK, family="monospace")
        snr = summary["snr"][i]
        _ftext(fig, col_x["snr"], y,
               f"{snr:.2f}" if isinstance(snr, (int, float)) else "-",
               fontsize=8.5, color=_INK, family="monospace")

    _footer(fig, f"sweep · {primary.config.run.name}",
            right=f"hash {collection.config.run_hash[:12]}")
    return fig


def _coord_axis_label(summary: dict[str, list[Any]]) -> str:
    short = summary.get("coord_short_axis")
    if short:
        return ", ".join(short)
    coords = summary.get("coord_value", [])
    if not coords:
        return "(no sweep coordinate)"
    keys: set[str] = set()
    for c in coords:
        if isinstance(c, dict):
            keys.update(c.keys())
    if not keys:
        return "(primary cell)"
    return ", ".join(sorted(keys))


def _sweep_table_page(
    records: list[RunRecord],
    summary: dict[str, list[Any]],
    page_num: int,
    total: int,
) -> plt.Figure:
    fig = _new_page()
    primary = records[0]
    _header(fig, "SWEEP TABLE",
            "Per-cell headline metrics", page_num, total)
    _ftext(fig, _MARGIN_L, _TITLE_Y - 0.28,
           "Each row is one sweep cell. KILO null_val_dark badge per cell.",
           fontsize=10, color=_MUTE, family="sans-serif",
           ha="left", va="baseline")

    kilo_ref = _kilo_null_val_dark()
    # Layout: 8 columns + leading index. Use a vertical span across most
    # of the page; smaller fonts so 15+ cells fit if needed.
    # Dropped DISK and EFFECTIVE — redundant given FINAL NULL and the
    # per-cell waterfall page. 8 columns at fontsize 8 fit cleanly
    # across the 7.5-inch usable width (margin 0.50 each side).
    col_keys = [
        ("idx", "#", 0.25),
        ("coord", "COORD", 1.95),       # fits "scene.star.jmag=12.0"
        ("chip", "CHIP_IDEAL", 0.95),
        ("kilo_x", "× KILO", 0.65),
        ("real", "REALIZED", 0.90),
        ("final", "FINAL NULL", 0.90),
        ("opd", "OPD_RMS (nm)", 1.05),  # ASCII; sigma was stripped by font fallback
        ("snr", "SNR", 0.55),
    ]
    x = _MARGIN_L
    col_x: dict[str, float] = {}
    for key, _label, width in col_keys:
        col_x[key] = x
        x += width

    # Anchor the table just below the subtitle (subtitle baseline 9.82").
    # Was 8.40 which left ~1.4" of dead space.
    table_top = 9.30
    hdr_y = table_top - 0.30
    _add_rule(fig, _MARGIN_L, _PAGE_W - _MARGIN_R, hdr_y + 0.16,
              color=_INK, lw=0.7)
    for key, label, _ in col_keys:
        _ftext(fig, col_x[key], hdr_y, label,
               fontsize=7.5, color=_MUTE, family="sans-serif",
               weight="bold", va="baseline")
    _add_rule(fig, _MARGIN_L, _PAGE_W - _MARGIN_R, hdr_y - 0.06,
              color=_INK, lw=0.5)

    row_h = 0.22
    for i in range(len(records)):
        y = hdr_y - 0.25 - i * row_h
        if y < 1.40:
            break
        if i % 2 == 0:
            _add_filled_rect(fig, _MARGIN_L + 0.01, y - row_h + 0.18,
                             _PAGE_W - _MARGIN_R - 0.01, y + 0.16,
                             _STRIPE, alpha=0.6)
        _ftext(fig, col_x["idx"], y, str(i),
               fontsize=8, color=_MUTE, family="monospace")
        _ftext(fig, col_x["coord"], y, summary["coord_label"][i],
               fontsize=8, color=_INK, family="monospace")
        chip = summary["chip_ideal"][i]
        _ftext(fig, col_x["chip"], y,
               f"{chip:.2e}" if isinstance(chip, (int, float)) else "-",
               fontsize=8, color=_INK, family="monospace")
        chip_kilo = summary["chip_ideal_kilo"][i]
        ratio_value = chip_kilo if isinstance(chip_kilo, (int, float)) else chip
        if kilo_ref and isinstance(ratio_value, (int, float)) and ratio_value > 0:
            ratio = ratio_value / kilo_ref
            color = _GOOD if ratio <= 2.0 else _ACCENT
            _ftext(fig, col_x["kilo_x"], y, f"× {ratio:.1f}",
                   fontsize=8, color=color, family="monospace")
        else:
            _ftext(fig, col_x["kilo_x"], y, "-",
                   fontsize=8, color=_MUTE, family="monospace")
        for key, val in (
            ("real", summary["realized_null"][i]),
            ("final", summary["final_null"][i]),
        ):
            text = f"{val:.2e}" if isinstance(val, (int, float)) else "-"
            _ftext(fig, col_x[key], y, text,
                   fontsize=8, color=_INK, family="monospace")
        opd = summary["sigma_opd_nm"][i]
        if isinstance(opd, (int, float)):
            _ftext(fig, col_x["opd"], y, f"{float(opd):.2f}",
                   fontsize=8, color=_INK, family="monospace")
        else:
            _ftext(fig, col_x["opd"], y, "-",
                   fontsize=8, color=_MUTE, family="monospace")
        snr = summary["snr"][i]
        _ftext(fig, col_x["snr"], y,
               f"{snr:.2f}" if isinstance(snr, (int, float)) else "-",
               fontsize=8, color=_INK, family="monospace")

    _footer(fig, f"sweep · {primary.config.run.name}",
            right=f"{len(records)} cells")
    return fig


def _sweep_trends_page(
    records: list[RunRecord],
    summary: dict[str, list[Any]],
    page_num: int,
    total: int,
) -> plt.Figure:
    fig = _new_page()
    primary = records[0]
    coord_label = _coord_axis_label(summary)
    # Wrap onto the subtitle line if the axis description is wide enough
    # to overflow the title block — main title shows the coord menu, the
    # subtitle the chart legend.
    title = f"Headline metrics vs {coord_label}"
    if len(title) > 78:
        title = "Headline metrics vs sweep coords"
    _header(fig, "SWEEP TRENDS", title, page_num, total)
    _ftext(fig, _MARGIN_L, _TITLE_Y - 0.28,
           f"Coords: {coord_label}  ·  log-scale null curves left, SNR/sigma_OPD/Strehl right.",
           fontsize=9, color=_MUTE, family="sans-serif",
           ha="left", va="baseline")

    # Use jmag as the x-axis if scene.star.jmag is a sweep coordinate.
    # Cells that share all other coords form one curve per metric, so a
    # J × N grid sweep renders as two clean curves rather than a zigzag.
    coord_vals = summary.get("coord_value", []) or []
    # Only keep dedupe representatives in the group label so constant
    # axes (e.g. kernel_order=0) and zipped duplicates do not bloat it.
    axis_reps = list(summary.get("coord_axis_reps") or [])
    group_axis_keys = [k for k in axis_reps if k != "scene.star.jmag"]
    if _sweep_has_jmag_axis(summary):
        xs = np.asarray(summary["jmag"], dtype=np.float64)
        x_label = "J magnitude"
        groups: dict[tuple, dict[str, Any]] = {}
        group_order: list[tuple] = []
        for i, c in enumerate(coord_vals):
            cdict = c if isinstance(c, dict) else {}
            non_jmag = {k: cdict.get(k) for k in group_axis_keys}
            sig = tuple(sorted(
                (k, _canon_coord_value(v)) for k, v in non_jmag.items()
            ))
            if sig not in groups:
                groups[sig] = {"idxs": [], "rep": non_jmag}
                group_order.append(sig)
            groups[sig]["idxs"].append(i)
        cell_groups: list[tuple[list[int], str]] = []
        for sig in group_order:
            entry = groups[sig]
            idxs = sorted(entry["idxs"], key=lambda j: xs[j])
            cell_groups.append((idxs, _group_label_from_coord(entry["rep"])))
    else:
        xs = np.arange(len(records), dtype=np.float64)
        x_label = "cell index"
        cell_groups = [(list(range(len(records))), "")]

    def _row(values: list[Any]) -> np.ndarray:
        return np.asarray(
            [v if isinstance(v, (int, float)) else np.nan for v in values],
            dtype=np.float64,
        )

    chip = _row(summary["chip_ideal"])
    realized = _row(summary["realized_null"])
    disk = _row(summary["disk_floor"])
    effective = _row(summary["effective_null"])
    final = _row(summary["final_null"])
    snr = _row(summary["snr"])
    opd = _row(summary["sigma_opd_nm"])
    strehl = _row(summary["mean_strehl"])

    # σ_OPD recorded in meters by the sensitivity stage; convert to nm for plotting.
    opd_nm = opd  # collector already converted the _m source key to nm

    # Left column: null-depth components (log y). Inset enough to fit
    # the "10⁻ⁿ" y-tick labels — these render outside the axes box.
    ax_null = _add_axes_in(fig, _MARGIN_L + 0.85, 1.40, 4.45, 9.10)
    ax_null.set_facecolor(_PAPER)
    ax_null.set_title("Null depth vs sweep coord",
                      fontsize=9, color=_INK, loc="left", pad=6.0)

    metric_series = [
        ("chip_ideal", chip, _ACCENT),
        ("realized", realized, _DARK_PORT),
        ("disk_floor", disk, _COOL),
        ("effective", effective, _MUTE),
        ("final", final, _GOOD),
    ]
    # Linestyles encode the cell-group (e.g. N_sub=18 vs N_sub=36). With
    # only one group, just use solid lines and skip the group legend.
    linestyles = ["-", "--", ":", "-."]

    def _plot_per_group(ax: plt.Axes, vals: np.ndarray, color: str,
                        metric_label: str | None = None,
                        log_positive: bool = False) -> None:
        for g_idx, (idxs, _g_label) in enumerate(cell_groups):
            arr = np.asarray(idxs, dtype=int)
            if arr.size == 0:
                continue
            x_arr = xs[arr]
            v_arr = vals[arr]
            mask = np.isfinite(v_arr)
            if log_positive:
                mask &= v_arr > 0
            if not mask.any():
                continue
            order = np.argsort(x_arr[mask])
            ls = linestyles[g_idx % len(linestyles)]
            ax.plot(
                x_arr[mask][order], v_arr[mask][order],
                marker="o", linestyle=ls, color=color,
                linewidth=1.4, markersize=3.5,
                label=metric_label if (g_idx == 0 and metric_label) else None,
            )

    for name, vals, color in metric_series:
        _plot_per_group(ax_null, vals, color, metric_label=name,
                        log_positive=True)

    kilo_ref = _kilo_null_val_dark()
    if kilo_ref:
        ax_null.axhline(kilo_ref, color=_GOOD, linewidth=1.0,
                        linestyle="--", alpha=0.7)
        ax_null.text(xs[-1] if xs.size else 0.0, kilo_ref * 1.08,
                     f"KILO = {kilo_ref:.2e}",
                     fontsize=7.5, color=_GOOD, family="monospace",
                     ha="right", va="bottom")
    ax_null.set_yscale("log")
    ax_null.yaxis.set_major_formatter(_log_formatter())
    ax_null.yaxis.set_minor_locator(LogLocator(base=10.0, subs="auto", numticks=12))
    ax_null.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
    ax_null.set_xlabel(x_label)
    ax_null.set_ylabel("Null depth (fractional)")
    ax_null.grid(True, which="both", color=_RULE, linewidth=0.4, alpha=0.6)
    ax_null.legend(loc="lower right", fontsize=7.5, frameon=True,
                   facecolor=_PAPER, edgecolor=_RULE, ncol=1,
                   title="metric", title_fontsize=7.5)

    # Right column: SNR, sigma_OPD, Strehl — same grouping.
    plot_w = 3.10
    x0 = 5.20
    x1 = x0 + plot_w
    top_y = 9.10
    bot_y = 1.40
    gap = 0.30
    panel_h = (top_y - bot_y - 2 * gap) / 3.0

    snr_top = top_y
    snr_bot = snr_top - panel_h
    opd_top = snr_bot - gap
    opd_bot = opd_top - panel_h
    strehl_top = opd_bot - gap
    strehl_bot = bot_y

    ax_snr = _add_axes_in(fig, x0, snr_bot, x1, snr_top)
    ax_snr.set_facecolor(_PAPER)
    _plot_per_group(ax_snr, snr, _ACCENT)
    ax_snr.set_ylabel("SNR")
    ax_snr.grid(True, color=_RULE, linewidth=0.4, alpha=0.6)
    ax_snr.tick_params(axis="x", labelbottom=False)
    ax_snr.set_title("SNR", fontsize=8.5, color=_INK,
                     loc="left", pad=4.0)

    ax_opd = _add_axes_in(fig, x0, opd_bot, x1, opd_top)
    ax_opd.set_facecolor(_PAPER)
    _plot_per_group(ax_opd, opd_nm, _DARK_PORT)
    ax_opd.set_ylabel("sigma_OPD per sub (nm)")
    ax_opd.grid(True, color=_RULE, linewidth=0.4, alpha=0.6)
    ax_opd.tick_params(axis="x", labelbottom=False)
    ax_opd.set_title("sigma_OPD per sub  (KILO J=4 ~ 1.35 nm)",
                     fontsize=8.5, color=_INK, loc="left", pad=4.0)

    ax_strehl = _add_axes_in(fig, x0, strehl_bot, x1, strehl_top)
    ax_strehl.set_facecolor(_PAPER)
    _plot_per_group(ax_strehl, strehl, _COOL)
    ax_strehl.set_ylim(0, 1.05)
    ax_strehl.set_xlabel(x_label)
    ax_strehl.set_ylabel("Mean Strehl (AO)")
    ax_strehl.grid(True, color=_RULE, linewidth=0.4, alpha=0.6)
    ax_strehl.set_title("AO Strehl", fontsize=8.5, color=_INK,
                        loc="left", pad=4.0)

    # If more than one cell-group, surface the group → linestyle mapping
    # in a compact legend below the SNR panel so the user can read which
    # line belongs to which (e.g.) N_sub value.
    if len(cell_groups) > 1:
        from matplotlib.lines import Line2D

        handles = [
            Line2D([0], [0], color=_INK,
                   linestyle=linestyles[g_idx % len(linestyles)],
                   linewidth=1.4,
                   label=g_label or f"group {g_idx}")
            for g_idx, (_idxs, g_label) in enumerate(cell_groups)
        ]
        ax_snr.legend(handles=handles, loc="upper right",
                      fontsize=7.0, frameon=True,
                      facecolor=_PAPER, edgecolor=_RULE,
                      title="cell group", title_fontsize=7.0,
                      ncol=1)

    _footer(fig, f"sweep · {primary.config.run.name}",
            right=f"{len(records)} cells")
    return fig


def _sweep_contrast_page(
    records: list[RunRecord],
    summary: dict[str, list[Any]],
    page_num: int,
    total: int,
) -> plt.Figure:
    fig = _new_page()
    primary = records[0]
    _header(fig, "SWEEP CONTRAST",
            "Per-cell contrast curves vs KILO paperv3",
            page_num, total)
    _ftext(fig, _MARGIN_L, _TITLE_Y - 0.28,
           "nullsim solid · KILO dashed where matching J is in the bundled reference cache.",
           fontsize=10, color=_MUTE, family="sans-serif",
           ha="left", va="baseline")

    ax = _add_axes_in(fig, _MARGIN_L + 0.95, 1.10,
                      _PAGE_W - _MARGIN_R - 0.10, 8.20)
    ax.set_facecolor(_PAPER)

    palette = [_ACCENT, _COOL, _DARK_PORT, _GOOD, _MUTE]
    plotted_any = False
    # Track positive finite values actually plotted so we can clamp the
    # y range — contrast metrics diverge at large separation where the
    # planet-throughput denominator approaches zero.
    finite_min, finite_max = np.inf, -np.inf
    # Drop unphysical "contrast > 1" tails (planet would need more flux
    # than the star) so the divergent edge doesn't blow up the axis.
    physical_ceiling = 1.0
    for i in range(len(records)):
        seps = summary["contrast_curve_seps_mas"][i]
        vals = summary["contrast_curve_values"][i]
        if seps is None or vals is None or seps.size == 0:
            continue
        mask = (
            np.isfinite(seps) & (seps > 0)
            & np.isfinite(vals) & (vals > 0) & (vals < physical_ceiling)
        )
        if not mask.any():
            continue
        color = palette[i % len(palette)]
        label = summary["coord_label"][i]
        ax.plot(seps[mask], vals[mask], "-", color=color, linewidth=1.4,
                label=f"nullsim {label}")
        plotted_any = True
        finite_min = min(finite_min, float(np.nanmin(vals[mask])))
        finite_max = max(finite_max, float(np.nanmax(vals[mask])))

        jmag = summary["jmag"][i]
        if isinstance(jmag, (int, float)):
            ref = _kilo_contrast_at_jmag(float(jmag))
            if ref is not None:
                ref_seps, ref_vals = ref
                ref_mask = (
                    np.isfinite(ref_seps) & (ref_seps > 0)
                    & np.isfinite(ref_vals) & (ref_vals > 0)
                    & (ref_vals < physical_ceiling)
                )
                if ref_mask.any():
                    ax.plot(ref_seps[ref_mask], ref_vals[ref_mask],
                            "--", color=color, linewidth=1.0, alpha=0.6,
                            label=f"KILO J={jmag:g}")
                    finite_min = min(finite_min, float(np.nanmin(ref_vals[ref_mask])))
                    finite_max = max(finite_max, float(np.nanmax(ref_vals[ref_mask])))

    if not plotted_any:
        ax.set_axis_off()
        ax.text(0.5, 0.5,
                "No contrast curves recorded (performance stage didn't run?)",
                ha="center", va="center", color=_MUTE,
                fontsize=11, transform=ax.transAxes, family="serif")
    else:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(_log_formatter())
        ax.yaxis.set_major_formatter(_log_formatter())
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs="auto", numticks=12))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs="auto", numticks=12))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
        if np.isfinite(finite_min) and np.isfinite(finite_max) and finite_max > finite_min:
            # Pad ~half a decade each side so the lines aren't flush
            # against the axis edges.
            ax.set_ylim(finite_min * 10 ** -0.4, finite_max * 10 ** 0.4)
        ax.set_xlabel("Separation (mas)")
        ax.set_ylabel("Required contrast (5-sigma)")
        ax.grid(True, which="both", color=_RULE, linewidth=0.4, alpha=0.6)
        # Legend lives inside the axes so long coord labels can't push
        # past the figure edge. ncol scales with how many curves there are.
        n_curves = int(plotted_any) and sum(
            1 for v in summary["contrast_curve_values"] if v is not None
        )
        ax.legend(loc="upper right", fontsize=7.0, frameon=True,
                  facecolor=_PAPER, edgecolor=_RULE,
                  ncol=1 if n_curves <= 6 else 2)

    _footer(fig, f"sweep · {primary.config.run.name}",
            right="contrast curves")
    return fig
