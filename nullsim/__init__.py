"""Top-level package for the nullsim photonic nuller simulator."""

from __future__ import annotations

from importlib import import_module

__version__ = "0.1.0"

import_module("nullsim.stages._autoload")
# nullsim.outputs is imported lazily: ``RunCollection``/``RunRecord``
# are dataclasses that don't need matplotlib, and the plot/table
# autoload is deferred to ``nullsim.outputs._autoload_outputs()``
# (called by the CLI's run path). ``validate`` no longer pulls
# matplotlib into the process.
import_module("nullsim.outputs")
