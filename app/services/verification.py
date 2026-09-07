"""Verification stage interface: recompute locally, compare with on-chain.

TODO(STEP 2): implement ``HashVerificationService`` behind
:class:`VerificationService`:
  * recompute the fingerprint from the *original discovered post* using the
    same :class:`~app.services.fingerprint.FingerprintService`
  * read the anchored hash via
    :meth:`~app.services.blockchain.BlockchainService.read_fingerprint`
  * compare case-insensitively after stripping any ``0x`` prefix
  * return VERIFIED on match, NOT VERIFIED on mismatch
  * add a tamper test: alter one canonical field, expect NOT VERIFIED

What a VERIFIED result means, and what it does not:
  * It means the discovered content is byte-for-byte unchanged since its
    fingerprint was anchored, and that the anchor is on-chain.
  * It does NOT mean the pipeline has identified a person. FaceTrace reports a
    content match, never legal identity.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..models.pipeline import (
    BlockchainRecord,
    Fingerprint,
    MatchingPost,
    VerificationResult,
)
from . import StageNotImplementedError

__all__ = ["VerificationService", "PendingVerificationService"]


@runtime_checkable
class VerificationService(Protocol):
    """Compares a locally recomputed fingerprint against the on-chain record."""

    def recompute(self, post: MatchingPost) -> Fingerprint:
        """Rebuild the fingerprint from the original post data."""
        ...

    def verify(self, post: MatchingPost, record: BlockchainRecord) -> VerificationResult:
        """Recompute, read the chain, and compare.

        Always returns a :class:`VerificationResult`; a mismatch is a valid
        NOT VERIFIED outcome, not an exception.
        """
        ...


class PendingVerificationService:
    """STEP 1 placeholder for :class:`VerificationService`."""

    name = "pending-verification-service"

    def recompute(self, post: MatchingPost) -> Fingerprint:
        raise StageNotImplementedError("Fingerprint recomputation")

    def verify(self, post: MatchingPost, record: BlockchainRecord) -> VerificationResult:
        raise StageNotImplementedError("Fingerprint verification")
