"""Deterministic per-stage RNG derivation (SPEC §6.5, stage-local identity).

SPEC §6.5 originally described the RNG identity as
``SeedSequence(entropy=(root_seed, config_hash, sweep_coord_tuple, instance_id))``.
The ``config_hash`` component was replaced by the stage's OWN spec digest
(``rng_identity``, computed by the runner): seeding from the full resolved
config made every stochastic draw — and, through the input-state digests,
every downstream cache entry — change on cosmetic config edits such as
adding a plot name (review 2026-06-10 Issue 7). A stage's draw now changes
only when its own spec, the sweep coordinate, the root seed, or its
instance id changes.

The recipe cannot be implemented literally: :class:`numpy.random.SeedSequence`
requires ``int`` or sequence-of-int entropy, while ``rng_identity`` is a hex
string, ``sweep_coord_tuple`` is a tuple of ``(str, Any)`` pairs, and
``instance_id`` is a string. To stay parallelism-invariant and
content-addressed, this module canonicalizes
``(rng_identity, sweep_coord_tuple, instance_id)`` via a stable type-tagged
byte encoding, SHA-256-hashes the result, splits the digest into eight
little-endian ``uint32`` entropy words, and passes
``(root_seed, *entropy_words)`` to :class:`numpy.random.SeedSequence`. The
type-tagged canonicalization prevents collisions between (e.g.) the string
``"1"`` and the integer ``1``.

The implementation here is the source of truth.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from nullsim.pipeline._canonical import canonical_bytes as _canonical_bytes


def derive_rng(
    root_seed: int,
    rng_identity: str,
    sweep_coord_tuple: tuple[tuple[str, Any], ...],
    instance_id: str,
) -> np.random.Generator:
    """Return a parallelism-invariant RNG stream for one stage instance.

    See the module docstring for the derivation recipe and the
    canonicalization step required because ``SeedSequence`` only accepts
    integer entropy.

    Args:
        root_seed: Integer root seed from the resolved run configuration.
        rng_identity: Digest of the stage's own spec (the runner passes
            ``canonical_digest(stage_spec).hex()``); any stable string
            identifying the stochastic stage's inputs is acceptable.
        sweep_coord_tuple: Tuple of ``(axis_name, value)`` pairs for this sweep
            cell. Empty for non-sweep runs.
        instance_id: Stage instance id from the ordered pipeline list.

    Returns:
        A NumPy default generator seeded from the full stage identity.

    Raises:
        ValueError: If ``root_seed`` is negative or ``instance_id`` is empty.
    """

    if root_seed < 0:
        raise ValueError("root_seed must be non-negative for deterministic RNG derivation.")
    if not instance_id:
        raise ValueError("instance_id must be non-empty for deterministic RNG derivation.")

    identity_bytes = _canonical_bytes((rng_identity, sweep_coord_tuple, instance_id))
    identity_digest = hashlib.sha256(identity_bytes).digest()
    entropy_words = tuple(
        int.from_bytes(identity_digest[index : index + 4], "little")
        for index in range(0, len(identity_digest), 4)
    )
    seed_sequence = np.random.SeedSequence((int(root_seed), *entropy_words))
    return np.random.default_rng(seed_sequence)

