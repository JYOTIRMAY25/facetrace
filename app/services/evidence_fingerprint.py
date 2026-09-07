"""SHA-256 fingerprints for canonical evidence bytes."""

from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class EvidenceFingerprint:
    algorithm: str
    fingerprint: str
    input_encoding: str = "UTF-8"
    status: str = "READY_FOR_BLOCKCHAIN"

    def as_dict(self) -> dict[str, str]:
        return {
            "algorithm": self.algorithm,
            "fingerprint": self.fingerprint,
            "input_encoding": self.input_encoding,
            "status": self.status,
        }


def fingerprint_evidence(canonical_evidence_bytes: bytes) -> EvidenceFingerprint:
    """Hash exactly the canonical UTF-8 bytes produced by Step 5."""
    if not isinstance(canonical_evidence_bytes, bytes):
        raise TypeError("canonical_evidence_bytes must be bytes")
    digest = hashlib.sha256(canonical_evidence_bytes).hexdigest()
    if not _SHA256_HEX.fullmatch(digest):
        raise RuntimeError("SHA-256 returned an invalid hexadecimal digest")
    return EvidenceFingerprint("SHA-256", digest)


def verify_evidence_fingerprint(
    canonical_evidence_bytes: bytes,
    expected_fingerprint: str,
) -> bool:
    """Recompute and compare a fingerprint using constant-time comparison."""
    if not isinstance(expected_fingerprint, str) or not _SHA256_HEX.fullmatch(
        expected_fingerprint
    ):
        return False
    actual = fingerprint_evidence(canonical_evidence_bytes).fingerprint
    return hmac.compare_digest(actual, expected_fingerprint)


__all__ = [
    "EvidenceFingerprint",
    "fingerprint_evidence",
    "verify_evidence_fingerprint",
]
