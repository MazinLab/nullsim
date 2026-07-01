---
date: 2026-05-16
---
## CI / packaging: drop the unused `[validation]` HCIPy extra

The `[validation]` extra and its companion CI job were aspirational placeholders — no code ever imported HCIPy and the job's final command was `pytest -m validation … || echo "No validation tests collected; skipping cleanly."`, so it installed ~120 MB of dependency and proved nothing.

Removed:
- `[validation]` optional-dependency entry and the `validation` pytest marker from `pyproject.toml`.
- The `validation` job from `.github/workflows/ci.yml`.
- Install/usage references from `README.md` and `CONTRIBUTING.md`.

`nullsim.md` design-doc references softened from "ships an opt-in `nullsim/validation/` package" to "HCIPy would be the natural fit if ground-truth cross-checks are added later." The extra is trivial to re-add the day someone writes the first `@pytest.mark.validation` test.
