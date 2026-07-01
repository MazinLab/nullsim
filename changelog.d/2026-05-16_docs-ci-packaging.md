---
date: 2026-05-16
---

### §15.8 — docs, CI, and packaging (closes SPEC §15 implementation roadmap)

- `README.md` ships at the repo root: project overview, install instructions,
  quick-start TOML snippet, links to examples and SPEC.
- `CONTRIBUTING.md` ships: development setup, test commands, the parity-reference
  refresh workflow, and the fragment-based doc maintenance model.
- `.github/workflows/ci.yml` ships with three matrix jobs: `test` (pytest, Python 3.12
  and 3.13), `parity` (parity test suite, frozen reference JSONs), and `validation`
  (hcipy-backed integration tests, optional extra).
- `pyproject.toml` updated: `readme = "README.md"`, MIT license, authors list,
  `[validation]` optional-dependency group (`hcipy>=0.7`), and `[project.urls]`
  placeholders for homepage and issue tracker.

This commit closes the SPEC §15 implementation roadmap. All eight steps are complete:
vertical slice → physics primitives → four stage scoped passes → detection_space output
catalog → KILO parity → NAYRA parity → HWO → docs/CI/packaging.
