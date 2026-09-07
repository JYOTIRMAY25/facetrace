"""Verification stage implementation."""

from __future__ import annotations

from typing import Optional

from ..models.pipeline import (
    BlockchainRecord,
    Fingerprint,
    MatchingPost,
    VerificationResult,
    VerificationStatus,
)
from .blockchain import BlockchainService
from .blockchain_sim import BlockchainReadError
from .fingerprint import FingerprintService
from . import ServiceError

class HashVerificationService:
    """Verifies content by recomputing fingerprints and comparing against on-chain records."""

    def __init__(self, fingerprint_service: FingerprintService, blockchain_service: BlockchainService):
        self._fingerprint_service = fingerprint_service
        self._blockchain_service = blockchain_service

    def recompute(self, post: MatchingPost) -> Fingerprint:
        """Rebuild the fingerprint from the original post data."""
        if not isinstance(post, MatchingPost):
            raise ServiceError("INVALID_EVIDENCE: Expected a valid MatchingPost instance")
        return self._fingerprint_service.fingerprint(post)

    def verify(self, post: Optional[MatchingPost], record: Optional[BlockchainRecord]) -> VerificationResult:
        """Recompute, read the chain, and compare."""
        if post is None or not isinstance(post, MatchingPost):
            return VerificationResult(
                computed_hash="",
                on_chain_hash="",
                match=False,
                status=VerificationStatus.INVALID_EVIDENCE
            )

        if record is None or not getattr(record, "record_id", None):
            return VerificationResult(
                computed_hash="",
                on_chain_hash="",
                match=False,
                status=VerificationStatus.FINGERPRINT_UNAVAILABLE
            )

        try:
            recomputed = self.recompute(post)
        except Exception:
            return VerificationResult(
                computed_hash="",
                on_chain_hash="",
                match=False,
                status=VerificationStatus.INVALID_EVIDENCE
            )

        try:
            stored_hash = self._blockchain_service.read_fingerprint(record.record_id)
        except BlockchainReadError:
            return VerificationResult(
                computed_hash=recomputed.hash,
                on_chain_hash="",
                match=False,
                status=VerificationStatus.FINGERPRINT_UNAVAILABLE
            )
        except Exception:
            return VerificationResult(
                computed_hash=recomputed.hash,
                on_chain_hash="",
                match=False,
                status=VerificationStatus.BLOCKCHAIN_UNAVAILABLE
            )

        if not stored_hash:
            return VerificationResult(
                computed_hash=recomputed.hash,
                on_chain_hash="",
                match=False,
                status=VerificationStatus.FINGERPRINT_UNAVAILABLE
            )

        # Compare case-insensitively after stripping optional '0x' prefix
        clean_computed = recomputed.hash.lower().replace("0x", "")
        clean_stored = stored_hash.lower().replace("0x", "")

        match = clean_computed == clean_stored

        return VerificationResult(
            computed_hash=recomputed.hash,
            on_chain_hash=stored_hash,
            match=match,
            status=VerificationStatus.VERIFIED if match else VerificationStatus.NOT_VERIFIED
        )

