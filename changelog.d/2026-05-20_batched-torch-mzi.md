---
date: 2026-05-20
---
### Batched torch MZI solver

Added an explicit `torch_device` selector for the torch chip optimizer and a
batched real/imag Clements unitary builder. The torch backend now evaluates
broadband wavelength stacks through the batched builder and uses the same path
to publish the deployed chromatic `optimal_u_stack`, giving CUDA runs a larger
per-step workload while preserving the existing BFGS optimizer contract.
