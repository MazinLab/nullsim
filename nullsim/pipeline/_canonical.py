"""Deterministic content-addressed byte encoding for hashing.

Single source of truth for all `_canonical_bytes`/`_canonicalize` callers in the
nullsim pipeline (state, rng, cache). Encoding properties:

- Type-tagged: a string ``"1"`` and the integer ``1`` digest differently.
- Stable across processes: no `repr(obj)` of arbitrary objects (would expose
  memory addresses); unknown types raise instead of silently producing an
  unstable digest.
- Numpy-array layout-invariant: arrays are normalized to C-contiguous bytes
  plus dtype and shape, so cache entries survive pickle round-trips and
  view/copy differences.
- Mappings are key-sorted; dataclasses are walked field-by-field in
  declaration order; objects implementing ``_canonical_bytes()`` are
  delegated to (lets ``SubBudget`` participate without an import cycle).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import PurePath
from typing import Any

import numpy as np


def canonical_bytes(value: Any) -> bytes:
    """Return a deterministic byte encoding for content hashing.

    Raises ``TypeError`` for unsupported types so callers cannot accidentally
    produce an unstable digest via Python's default ``repr``. Stage authors
    needing to hash a new type should add a branch here, not bypass.
    """

    if value is None:
        return b"n:"
    if isinstance(value, bool):
        return f"b:{int(value)}".encode()
    if isinstance(value, int):
        return f"i:{value}".encode()
    if isinstance(value, float):
        return f"f:{value!r}".encode()
    if isinstance(value, complex):
        return f"c:{value.real!r}:{value.imag!r}".encode()
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        return b"s:" + len(encoded).to_bytes(8, "little") + encoded
    if isinstance(value, bytes):
        return b"y:" + len(value).to_bytes(8, "little") + value
    if isinstance(value, PurePath):
        encoded = str(value).encode("utf-8")
        return b"p:" + len(encoded).to_bytes(8, "little") + encoded
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        array_bytes = contiguous.tobytes(order="C")
        dtype_bytes = str(contiguous.dtype).encode("utf-8")
        shape_bytes = repr(tuple(int(dim) for dim in contiguous.shape)).encode("utf-8")
        return (
            b"a:"
            + len(dtype_bytes).to_bytes(8, "little")
            + dtype_bytes
            + len(shape_bytes).to_bytes(8, "little")
            + shape_bytes
            + len(array_bytes).to_bytes(8, "little")
            + array_bytes
        )
    if isinstance(value, np.generic):
        return canonical_bytes(value.item())
    # Duck-type SubBudget without importing it (would be circular).
    canonical_method = getattr(value, "_canonical_bytes", None)
    if callable(canonical_method) and not isinstance(value, type):
        try:
            result = canonical_method()
        except TypeError:
            result = None
        if isinstance(result, (bytes, bytearray)):
            return b"u:" + bytes(result)
    if isinstance(value, Mapping):
        # Type-tagged keys: sort by canonical_bytes(key) and emit the
        # original key value (not str(key)) so {1: "a"} and {"1": "a"}
        # produce different digests, matching the module's stated
        # type-tagged guarantee. Pre-compute key bytes once for each
        # item to avoid re-encoding during sort + emit.
        keyed = [(canonical_bytes(k), k, v) for k, v in value.items()]
        keyed.sort(key=lambda triple: triple[0])
        return _sequence_bytes(
            b"m",
            tuple((k, v) for _, k, v in keyed),
        )
    if isinstance(value, tuple):
        return _sequence_bytes(b"t", value)
    if isinstance(value, list):
        return _sequence_bytes(b"l", value)
    if isinstance(value, (set, frozenset)):
        items_sorted = sorted(value, key=canonical_bytes)
        return _sequence_bytes(b"x", tuple(items_sorted))
    if hasattr(value, "model_dump"):
        return b"d:" + canonical_bytes(value.model_dump())
    if is_dataclass(value) and not isinstance(value, type):
        payload = tuple(
            (field_info.name, getattr(value, field_info.name))
            for field_info in fields(value)
        )
        type_name = type(value).__qualname__
        return (
            b"o:"
            + canonical_bytes(type_name)
            + _sequence_bytes(b"of", payload)
        )
    raise TypeError(
        f"canonical_bytes: no stable encoding for {type(value).__name__!r}. "
        "Add a branch in nullsim.pipeline._canonical rather than silently "
        "falling through to repr() — repr is not stable across processes."
    )


def canonical_digest(value: Any) -> bytes:
    """SHA-256 of the canonical byte encoding."""
    return hashlib.sha256(canonical_bytes(value)).digest()


def _sequence_bytes(prefix: bytes, values: Any) -> bytes:
    payload = bytearray(prefix + b":")
    payload.extend(len(values).to_bytes(8, "little"))
    for item in values:
        encoded = canonical_bytes(item)
        payload.extend(len(encoded).to_bytes(8, "little"))
        payload.extend(encoded)
    return bytes(payload)


__all__ = ["canonical_bytes", "canonical_digest"]
