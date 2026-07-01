"""Shared matplotlib styling for nullsim output plots.

Centralizes the rcParams every plot module would otherwise re-declare.
Keep this lightweight: presentation knobs only, no plotting logic.
Use the ``Agg`` backend so headless runs (CI, sweep workers) don't try
to open a window.
"""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator

import os

import matplotlib

# Code Review 2026-05-16 Issue 38: don't force the Agg backend at module
# import time — that silently switches a Jupyter user's ``%matplotlib
# inline`` backend when nullsim is imported. Switch only when:
#   * NULLSIM_HEADLESS is set in the environment, OR
#   * matplotlib hasn't picked a backend yet (e.g. headless CI).
def _ensure_headless_backend() -> None:
    if os.environ.get("NULLSIM_HEADLESS"):
        matplotlib.use("Agg", force=True)
        return
    current = matplotlib.get_backend()
    if not current or current.lower() in {"agg"}:
        matplotlib.use("Agg", force=True)


_ensure_headless_backend()

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib import rcParams  # noqa: E402


# Ported verbatim from KILO's kilo/__init__.py (_configure_matplotlib)
# and kilo/plotting.py (_setup_style). Kept literal so figures generated
# by nullsim look identical to the KILO paperv3 figures. Update both
# halves together if KILO's style ever changes.
PUBLICATION_RC: dict[str, object] = {
    # KILO _configure_matplotlib (kilo/__init__.py)
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    # KILO _setup_style (kilo/plotting.py)
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "lines.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.fontsize": 11,
    "figure.figsize": (8.0, 6.0),
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
}


def apply_publication_style() -> None:
    """Globally set the publication rcParams. Idempotent."""
    rcParams.update(PUBLICATION_RC)


@contextmanager
def publication_style() -> Iterator[None]:
    """Apply the publication style for the duration of a `with` block."""
    with plt.rc_context(PUBLICATION_RC):
        yield


# Auto-apply at import time, matching KILO's kilo/__init__.py behavior.
# Notebook users who want different fonts can override rcParams after
# `import nullsim`.
apply_publication_style()


__all__ = ["PUBLICATION_RC", "apply_publication_style", "publication_style"]
