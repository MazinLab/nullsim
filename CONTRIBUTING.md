# Contributing to nullsim

Thanks for your interest in `nullsim`. This file covers the development
workflow, coding standards, and test conventions. The design document is
[`nullsim.md`](nullsim.md), and the navigation guide for the package is
[`code-map.md`](code-map.md).

## Development setup

```bash
git clone https://github.com/MazinLab/nullsim.git
cd nullsim
pip install -e .[dev]        # base install plus pytest and the parallel test runner
pip install -e .[torch]      # optional: Torch chip-optimization backend
```

`nullsim` targets Python 3.12 and 3.13.

## Workflow

1. Branch off `main`.
2. Make the change, run the relevant tests, and commit with a focused
   message.
3. Open a pull request against `main`. CI runs the full pytest suite on
   Python 3.12 and 3.13 (see `.github/workflows/ci.yml`).

Keep changes surgical: a reviewer reading the diff should understand why
each chunk is there. If a cleanup is incidental to your change, call it out
in the PR description rather than folding it in silently.

## Tests

The pytest configuration lives in `pyproject.toml` under
`[tool.pytest.ini_options]`.

```bash
pytest tests/        # full suite
pytest -m stats      # only the stochastic / statistical-validation tests
```

| Marker | Purpose |
|---|---|
| `@pytest.mark.stats` | Stochastic / statistical-validation tests; may write diagnostic plots under `tests/output/`. |

Pinning tests use `@pytest.mark.xfail(strict=True)` for invariants that are
load bearing — they fail loudly if the behavior shifts, including if they
ever *start* passing unexpectedly. Cover the happy path, the edge cases the
code can actually hit, and the error paths that exist.

## Adding a stage or an output

The extension mechanism is registration, not editing the core. See the
"Adding a Stage" and "Adding an Output" sections of
[`code-map.md`](code-map.md) for the step by step contracts. In short:
subclass `BaseStage`, declare `type_name` / `family` / `consumes` /
`produces`, implement `apply`, decorate with `@register_stage`, list the
module in `nullsim/stages/_autoload.py`, and add focused tests under
`tests/`.

Important stage contracts:

- Do not mutate existing state budgets in place; return replaced budgets.
- Preserve existing components when adding one to a budget.
- Declare every required state read in `consumes`.

## Style

- Type hints on all library-code signatures and class attributes. Use
  `from __future__ import annotations` for forward references.
- Public functions get docstrings: summary, params, returns, raises.
- Keep physical units explicit in names or docstrings (`separation_mas`,
  `wavelength_um`), and validate array shapes at function entry for
  non-trivial operations.
- Put pure math in `nullsim/physics/`, test it under `tests/physics/`, and
  wrap it in a stage only once the primitive is pinned.
- Explicit over clever, fail fast and loud, no silently swallowed errors.

## Changelog

Add a dated fragment under `changelog.d/` describing your change rather than
editing `CHANGELOG.md` directly; maintainers roll the fragments into
`CHANGELOG.md` on a controlled cadence.
