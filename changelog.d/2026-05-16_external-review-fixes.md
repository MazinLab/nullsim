---
date: 2026-05-16
---
### 5 external-agent code-review fixes (commit 979b284)

Addressed findings from the outside-inspector (Codex) code-review pass:

- `nullsim/cli.py`: additional CLI robustness fixes.
- `nullsim/pipeline/_canonical.py`: canonical-bytes hashing correctness.
- `nullsim/pipeline/cache.py`, `runner.py`: cache invalidation and runner validation edge cases.
- `tests/pipeline/test_cache.py`, `test_runner_validation.py`: tests for the fixed paths.
