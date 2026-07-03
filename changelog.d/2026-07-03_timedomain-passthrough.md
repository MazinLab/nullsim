---
date: 2026-07-03
---

### Opaque `[timedomain]` passthrough section

- `TopLevelConfig` gains an optional `timedomain: dict | None` field so the
  nullsimtd time-domain companion package can carry its `[timedomain.*]`
  blocks in shared TOML configs. nullsim never interprets the section;
  nullsimtd validates it with its own schema.
- Present blocks fold into `config_hash` via the canonical dump. Absent
  blocks are dropped by `exclude_none`, so pre-existing configs keep their
  hashes and cache entries.
- The section deep-merges through `extends` chains like any other top-level
  table (parent values inherited, child keys win).
