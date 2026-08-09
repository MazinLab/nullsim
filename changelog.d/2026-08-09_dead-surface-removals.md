---
date: 2026-08-09
---
### Removed: two dead public surfaces

A test-suite cull left two never-exercised entry points with nothing behind
them. Both are gone. Neither was reachable from any shipped config or from
prod code, so nothing in the repository changes behavior — but both were
user-visible, so a config or script written against them will now fail rather
than silently do nothing.

- **`sensitivity.detection_curve` param `amp_wfs_n_modes_override`**. The
  override branch in `_resolve_amp_servo_n_modes` had never executed: no
  config, preset, or example ever set the field to a non-`None` value. A
  config that sets it now fails stage param validation instead of being
  accepted and ignored. The pickoff, chip, and port-count fallbacks that
  actually resolve the amplitude-servo mode count are untouched.
- **`nullsim.physics` aliases `jones_identity` and
  `jones_random_birefringence`**. KILO-era names re-exported alongside the
  primitives they aliased. Use `identity_jones` and `random_birefringence`,
  which are unchanged. `jones_hwp` and `jones_pbs_arm` have prod callers and
  stay.
