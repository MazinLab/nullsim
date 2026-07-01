---
date: 2026-05-17
---
### Phase 7 — injection cache: fold upstream wavefront into cache key

Per the KILO parity audit (2026-05-17), `injection.ideal` read the
upstream Strehl from `state.wavefront` opportunistically but never
declared `wavefront` in `consumes`. Source-code edits invalidated the
cache (via the package source digest) but AO-parameter-only changes
that altered the wavefront's Strehl without touching code did NOT
invalidate the cached injected field — silent stale-state reuse on
AO sweeps.

The reason `wavefront` was kept out of `consumes`: the HWO-space
pipeline runs `injection.ideal` with no upstream AO/atmosphere stage
producing `wavefront`, and `validate_dependencies` would reject the
missing producer.

- New `Stage.cache_key_extras: ClassVar[frozenset[StateKey]]` slot
  (default empty). State keys listed here participate in the cache
  key when present but are NOT enforced by the validator. Lets a
  stage opt in to opportunistic-read invalidation without making the
  read mandatory.
- `ContentCache.key_for` folds
  `state.content_digest_for(instance.cache_key_extras)` into the key
  alongside the regular consumed-state digest.
- `IdealInjection.cache_key_extras = frozenset({"wavefront"})` closes
  the gap. The HWO pipeline still runs (no validator change), and AO
  sweeps now invalidate the injected field correctly.
- Regression test:
  `test_injection_ideal_cache_key_changes_with_upstream_strehl`
  builds two minimal states with different Strehl values and asserts
  the cache keys differ.
