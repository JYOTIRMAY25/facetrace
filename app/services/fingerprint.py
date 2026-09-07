"""Fingerprint stage implementation: canonicalize a post, then SHA-256 it."""

from __future__ import annotations

import hashlib
import unicodedata
import re
from typing import Protocol, runtime_checkable
from urllib.parse import urlsplit, urlunsplit

from ..models.pipeline import (
    CANONICAL_FIELD_ORDER,
    CANONICALIZATION_VERSION,
    CanonicalPayload,
    Fingerprint,
    MatchingPost,
)
from . import StageNotImplementedError

__all__ = [
    "CANONICAL_FIELD_ORDER",
    "CANONICALIZATION_VERSION",
    "FingerprintService",
    "Sha256FingerprintService",
    "PendingFingerprintService",
]

@runtime_checkable
class FingerprintService(Protocol):
    """Turns a matching post into a deterministic cryptographic fingerprint."""

    #: Hash algorithm name, recorded on every fingerprint.
    algorithm: str
    #: Canonicalization version this implementation produces.
    canonicalization_version: str

    def canonicalize(self, post: MatchingPost) -> CanonicalPayload:
        """Build the deterministic pre-hash representation of ``post``."""
        ...

    def fingerprint(self, post: MatchingPost) -> Fingerprint:
        """Canonicalize then hash. Must be a pure function of ``post``."""
        ...


class Sha256FingerprintService:
    """Production implementation: SHA-256 over NFC-normalized canonical fields."""

    algorithm: str = "SHA-256"
    canonicalization_version: str = CANONICALIZATION_VERSION

    def _normalize_text(self, text: str) -> str:
        """NFC normalize, collapse whitespace, strip."""
        if not text:
            return ""
        # NFC normalization
        text = unicodedata.normalize("NFC", text)
        # Collapse whitespace runs
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _normalize_url(self, url: str) -> str:
        """Lowercase scheme/host, strip default port/trailing slash."""
        if not url:
            return ""
        parts = urlsplit(url)
        # Lowercase scheme and host
        scheme = parts.scheme.lower()
        netloc = parts.netloc.lower()
        
        # Strip default port (basic check, can be improved if needed)
        if ":" in netloc:
            host, port = netloc.split(":", 1)
            if (scheme == "http" and port == "80") or (scheme == "https" and port == "443"):
                netloc = host

        # Strip trailing slash on empty path
        path = parts.path
        if not path or path == "/":
            path = ""

        return urlunsplit((scheme, netloc, path, parts.query, parts.fragment))

    def canonicalize(self, post: MatchingPost) -> CanonicalPayload:
        """Build the deterministic pre-hash representation of ``post``."""
        
        # Prepare fields
        field_values = {
            "url": self._normalize_url(post.url),
            "source": self._normalize_text(post.source),
            "title": self._normalize_text(post.title),
            "text": self._normalize_text(post.text),
            "image_url": self._normalize_url(post.image_url or ""),
        }
        
        ordered_values = [field_values[f] for f in CANONICAL_FIELD_ORDER]
        
        # Join with \n, prefix with version
        payload_str = "\n".join(ordered_values)
        full_payload = f"{self.canonicalization_version}\n{payload_str}"
        
        return CanonicalPayload(
            canonicalization_version=self.canonicalization_version,
            field_order=CANONICAL_FIELD_ORDER,
            payload=full_payload,
        )

    def fingerprint(self, post: MatchingPost) -> Fingerprint:
        """Canonicalize then hash. Must be a pure function of ``post``."""
        payload = self.canonicalize(post)
        
        # SHA-256 over UTF-8 encoded payload
        hash_digest = hashlib.sha256(payload.payload.encode("utf-8")).hexdigest()
        
        return Fingerprint(
            algorithm=self.algorithm,
            canonicalization_version=self.canonicalization_version,
            hash=hash_digest,
        )


class PendingFingerprintService:
    """STEP 1 placeholder for :class:`FingerprintService`."""

    name = "pending-fingerprint-service"
    algorithm = "SHA-256"
    canonicalization_version = CANONICALIZATION_VERSION

    def canonicalize(self, post: MatchingPost) -> CanonicalPayload:
        raise StageNotImplementedError("Post canonicalization")

    def fingerprint(self, post: MatchingPost) -> Fingerprint:
        raise StageNotImplementedError("SHA-256 fingerprinting")
