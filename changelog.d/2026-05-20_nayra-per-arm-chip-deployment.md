---
date: 2026-05-20
---
### NAYRA per-arm chip deployment (`chip.kernel_mzi_mesh`)

`KernelMZIMesh` gains a `deployment_mode` parameter (`"shared"` | `"per_pol_arm"`,
default `"shared"`).

In `"per_pol_arm"` mode the chip design is deployed once per PBS polarization
arm with an independent diagonal input-phase correction per arm, matching the
NAYRA dual-PBS-arm physical layout where two separate photonic chips process the
two polarization channels independently. The result component records both the
shared design `u_stack` and the per-arm deployed `u_stack_per_arm`.

The `"shared"` default is unchanged; existing configs require no modification.

`examples/nayra_eelt.toml` updated to set `deployment_mode = "per_pol_arm"`.
Tests added in `tests/stages/chip/test_kernel_mzi_mesh.py`.
