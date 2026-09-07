"""Deterministic serialization for evidence records before SHA-256 (Step 6)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CanonicalEvidence:
    canonical_json: str
    canonical_bytes: bytes


def canonicalize_evidence(evidence: Any) -> CanonicalEvidence:
    """Serialize evidence as sorted, compact, UTF-8 JSON without hashing."""
    if hasattr(evidence, "as_dict"):
        evidence = evidence.as_dict()
    canonical_json = json.dumps(
        evidence,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return CanonicalEvidence(
        canonical_json=canonical_json,
        canonical_bytes=canonical_json.encode("utf-8"),
    )
