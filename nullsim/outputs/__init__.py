"""Output catalog package for plots and tables produced from run results."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import os
from pathlib import Path
import tempfile
from typing import Any

from nullsim.config import ResolvedConfig
from nullsim.pipeline import RunResult


@dataclass(frozen=True)
class RunRecord:
    """One executed sweep cell or single-run result."""

    config: ResolvedConfig
    result: RunResult
    run_dir: Path
    coord_tuple: tuple[tuple[str, Any], ...]

    @property
    def coord_mapping(self) -> dict[str, Any]:
        """Return sweep coordinates as a mutable row mapping."""

        return dict(self.coord_tuple)


@dataclass(frozen=True)
class RunCollection:
    """Collection-level payload passed to registered outputs."""

    config: ResolvedConfig
    records: tuple[RunRecord, ...]
    output_dir: Path


def _autoload_outputs() -> None:
    """Trigger registration of plot/table writers.

    Earlier this ran at module-import time, which pulled matplotlib (~6
    MB of imports and rcParams setup) into ``nullsim validate``. The
    validate path never produces outputs; importing the heavy stack on
    every config check is wasteful and clutters the terminal with
    matplotlib cache warnings before "config OK" prints.

    The CLI ``run`` path explicitly calls this once before writing
    outputs; downstream code constructing ``RunCollection`` for the
    library API can call it too.
    """

    _ensure_writable_matplotlib_config_dir()
    import_module("nullsim.outputs.plots")
    import_module("nullsim.outputs.tables")


def _ensure_writable_matplotlib_config_dir() -> None:
    """Avoid matplotlib cache warnings in read-only home directories."""

    if os.environ.get("MPLCONFIGDIR"):
        return
    default_dir = Path.home() / ".matplotlib"
    if _is_writable_dir(default_dir):
        return
    fallback = Path(tempfile.gettempdir()) / "nullsim-matplotlib"
    fallback.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(fallback)


def _is_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".nullsim-write-test"
        probe.write_text("", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError:
        return False
    return True


__all__ = ["RunCollection", "RunRecord", "_autoload_outputs"]
