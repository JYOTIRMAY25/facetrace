"""Matching stage: compare faces and select a reliable post.

STEP 4 provides :class:`InsightFaceMatcher` (candidate image -> detect ->
encode -> cosine similarity) and :class:`DefaultMatchSelector` (rank,
threshold, decide).

Hard rules honoured:
  * Never force a match.  Below threshold, return
    ``MatchDecision(matched=False, reason="NO_MATCH")``.
  * The selected post retains its original URL and source.
  * The threshold comes from ``FACE_MATCH_THRESHOLD`` (default 0.65).
  * A match is a *content* similarity claim.  It does not establish or
    assert legal identity.
  * Embedding vectors are never logged, serialised, or written to disk.
"""

from __future__ import annotations

import logging
from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from ..config import Settings, load_settings
from ..models.pipeline import (
    ErrorCode,
    FaceEncoding,
    MatchDecision,
    MatchingPost,
    SearchResult,
)
from . import ServiceError, StageNotImplementedError

__all__ = [
    "FaceMatcher",
    "InsightFaceMatcher",
    "DefaultMatchSelector",
    "MatchSelector",
    "PendingFaceMatcher",
    "PendingMatchSelector",
]

logger = logging.getLogger(__name__)

#: Maximum number of candidates to encode per selection round.
_MAX_CANDIDATES = 30

#: HTTP timeout (seconds) when downloading a candidate image.
_CANDIDATE_TIMEOUT = 15.0


# --------------------------------------------------------------------------
# Protocols
# --------------------------------------------------------------------------


@runtime_checkable
class FaceMatcher(Protocol):
    """Scores how similar two face encodings are."""

    metric: str

    def similarity(self, left: FaceEncoding, right: FaceEncoding) -> float:
        """Return a similarity in ``[0.0, 1.0]`` (higher is more similar)."""
        ...

    def encode_candidate(self, image_url: str) -> FaceEncoding | None:
        """Encode the face in a candidate image, or ``None`` if unusable."""
        ...


@runtime_checkable
class MatchSelector(Protocol):
    """Picks the single most reliable matching post, or reports no match."""

    threshold: float

    def select(
        self, encoding: FaceEncoding, results: Sequence[SearchResult]
    ) -> MatchDecision:
        """Rank candidates and apply the threshold."""
        ...


# --------------------------------------------------------------------------
# InsightFace matcher
# --------------------------------------------------------------------------


class InsightFaceMatcher:
    """InsightFace-backed face matcher using cosine similarity on L2-normalised embeddings.

    The ``encoding_reference`` field on :class:`FaceEncoding` is a SHA-256
    digest of the raw vector.  This matcher maintains a small in-memory
    cache keyed by that digest so :meth:`similarity` can look up both sides
    without re-running inference.
    """

    name: str = "insightface-cosine-matcher"
    metric: str = "cosine"

    def __init__(self, face_identifier: object) -> None:
        self._face_id = face_identifier
        #: digest -> EmbeddingVector
        self._embedding_cache: dict[str, object] = {}

    def register_embedding(self, encoding: FaceEncoding, vector: object) -> None:
        """Cache an embedding vector for later similarity comparison."""
        self._embedding_cache[encoding.encoding_reference] = vector

    def similarity(self, left: FaceEncoding, right: FaceEncoding) -> float:
        """Cosine similarity via dot product on L2-normalised vectors.

        Both embeddings must have been registered or produced by
        :meth:`encode_candidate`.  Raises :class:`ServiceError` if a
        vector is missing from the cache.
        """
        left_vec = self._embedding_cache.get(left.encoding_reference)
        right_vec = self._embedding_cache.get(right.encoding_reference)
        if left_vec is None or right_vec is None:
            raise ServiceError(
                f"Embedding not cached for comparison "
                f"(left={'found' if left_vec is not None else 'missing'}, "
                f"right={'found' if right_vec is not None else 'missing'})",
                code=ErrorCode.FACE_ENCODING_FAILED.value,
                retryable=False,
            )
        # InsightFace w600k_r50 returns L2-normalised embeddings, so
        # cosine similarity equals the dot product.
        score = float(np.dot(left_vec.values, right_vec.values))  # type: ignore[union-attr]
        return max(0.0, min(1.0, score))

    def encode_candidate(self, image_url: str) -> Optional[FaceEncoding]:
        """Download a candidate image and encode its face.

        Returns a :class:`FaceEncoding` with the embedding cached for
        later similarity lookup, or ``None`` when the image cannot be
        processed (no face, multiple faces, decode failure, network error).
        """
        try:
            image_bytes = _fetch_image_bytes(image_url)
        except ServiceError:
            logger.debug("candidate media unavailable: %s", image_url)
            return None

        try:
            result, embedding = self._face_id.analyze_with_embedding(  # type: ignore[union-attr]
                image_bytes
            )
        except ServiceError:
            logger.debug("candidate face processing failed for %s", image_url)
            return None

        if not result.success or embedding is None or result.encoding is None:
            logger.debug(
                "candidate %s: face_count=%d success=%s",
                image_url,
                result.face_count,
                result.success,
            )
            return None

        self._embedding_cache[result.encoding.encoding_reference] = embedding
        return result.encoding

    def clear_cache(self) -> None:
        """Drop all cached embedding vectors (memory hygiene)."""
        self._embedding_cache.clear()


# --------------------------------------------------------------------------
# Default match selector
# --------------------------------------------------------------------------


class DefaultMatchSelector:
    """Ranks candidates by cosine similarity and applies the configured threshold.

    The caller must register the query embedding via
    :meth:`set_query_embedding` before calling :meth:`select`.
    """

    name: str = "default-match-selector"

    def __init__(
        self,
        settings: Optional[Settings] = None,
        matcher: Optional[InsightFaceMatcher] = None,
    ) -> None:
        _settings = settings or load_settings()
        self.threshold: float = _settings.face_match_threshold
        if matcher is None:
            # The face_identifier will be wired by the orchestrator; here we
            # just need the matcher instance with the protocol.
            self._matcher = InsightFaceMatcher.__new__(InsightFaceMatcher)
            self._matcher._embedding_cache = {}
        else:
            self._matcher = matcher

    def set_query_embedding(
        self, encoding: FaceEncoding, vector: object
    ) -> None:
        """Register the query embedding for this selection round."""
        self._matcher.register_embedding(encoding, vector)
        self._query_encoding = encoding

    def select(
        self, encoding: FaceEncoding, results: Sequence[SearchResult]
    ) -> MatchDecision:
        """Rank candidates and apply the threshold.

        Always returns a :class:`MatchDecision`.  ``matched=False`` is a
        valid expected state, not an error.
        """
        if not results:
            return MatchDecision(
                matched=False,
                threshold=self.threshold,
                reason="NO_CANDIDATES",
            )

        # Ensure the query embedding is registered.
        if encoding.encoding_reference not in self._matcher._embedding_cache:
            raise ServiceError(
                "Query embedding not registered; call set_query_embedding() first",
                code=ErrorCode.FACE_ENCODING_FAILED.value,
                retryable=False,
            )

        best_score = 0.0
        best_result: Optional[SearchResult] = None
        candidates_considered = 0

        for result in results[:_MAX_CANDIDATES]:
            if not result.image_url:
                logger.debug(
                    "skipping result without image_url: %s", result.url
                )
                continue

            candidates_considered += 1
            candidate_encoding = self._matcher.encode_candidate(
                result.image_url
            )

            if candidate_encoding is None:
                logger.debug(
                    "candidate rejected (no face or invalid media): %s",
                    result.url,
                )
                continue

            try:
                score = self._matcher.similarity(encoding, candidate_encoding)
            except ServiceError:
                logger.debug(
                    "similarity computation failed for %s", result.url
                )
                continue

            logger.debug("candidate %s: similarity=%.4f", result.url, score)

            if score > best_score:
                best_score = score
                best_result = result

        if best_result is None:
            if candidates_considered > 0:
                reason = "NO_MATCH"
            else:
                reason = "NO_VALID_CANDIDATES"
            return MatchDecision(
                matched=False,
                threshold=self.threshold,
                candidates_considered=candidates_considered,
                reason=reason,
            )

        if best_score >= self.threshold:
            post = MatchingPost(
                result_id=best_result.result_id,
                source=best_result.source,
                url=best_result.url,
                title=best_result.title,
                text=best_result.text,
                image_url=best_result.image_url,
                match_score=best_score,
                metadata=dict(best_result.metadata),
            )
            return MatchDecision(
                matched=True,
                post=post,
                threshold=self.threshold,
                best_score=best_score,
                candidates_considered=candidates_considered,
                reason=(
                    f"Candidate similarity {best_score:.4f} exceeded "
                    f"the configured matching threshold ({self.threshold})."
                ),
            )

        # Best score found but below threshold.
        return MatchDecision(
            matched=False,
            threshold=self.threshold,
            best_score=best_score,
            candidates_considered=candidates_considered,
            reason="NO_MATCH",
        )



# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _fetch_image_bytes(image_url: str) -> bytes:
    """Helper to download or read image for matching."""
    if not image_url:
        raise ServiceError("Image URL is empty", retryable=False)

    if image_url.startswith("file://"):
        from urllib.parse import unquote, urlparse
        import os
        from pathlib import Path
        parsed = urlparse(image_url)
        raw_path = unquote(parsed.path)
        if os.name == "nt" and raw_path.startswith("/") and len(raw_path) > 2 and raw_path[2] == ":":
            raw_path = raw_path.lstrip("/")
        p = Path(raw_path)
        if not p.exists():
            raise ServiceError(f"Local candidate image missing: {p}", retryable=False)
        return p.read_bytes()

    if os.path.exists(image_url):
        from pathlib import Path
        return Path(image_url).read_bytes()

    try:
        import httpx
    except ImportError:
        raise ServiceError("httpx not installed", retryable=False)

    try:
        response = httpx.get(image_url, timeout=_CANDIDATE_TIMEOUT, follow_redirects=True)
        response.raise_for_status()
        return response.content
    except Exception as e:
        raise ServiceError(f"Failed to fetch image: {e}", retryable=True)


# --------------------------------------------------------------------------
# STEP 1 placeholders (kept for backward compatibility)
# --------------------------------------------------------------------------


class PendingFaceMatcher:
    """STEP 1 placeholder for :class:`FaceMatcher`."""

    name = "pending-face-matcher"
    metric = "unset"

    def similarity(self, left: FaceEncoding, right: FaceEncoding) -> float:
        raise StageNotImplementedError("Face similarity scoring")

    def encode_candidate(self, image_url: str) -> FaceEncoding | None:
        raise StageNotImplementedError("Candidate face encoding")


class PendingMatchSelector:
    """STEP 1 placeholder for :class:`MatchSelector`."""

    name = "pending-match-selector"
    threshold = 0.0

    def select(
        self, encoding: FaceEncoding, results: Sequence[SearchResult]
    ) -> MatchDecision:
        raise StageNotImplementedError("Match selection")
        raise ServiceError(f"Failed to fetch image: {e}", retryable=True)

@runtime_checkable
class FaceMatcher(Protocol):
    """Scores how similar two face encodings are."""

    #: Similarity metric name, e.g. ``"cosine"`` — recorded for auditability.
    metric: str

    def similarity(self, left: FaceEncoding, right: FaceEncoding) -> float:
        """Return a similarity in ``[0.0, 1.0]`` (higher is more similar)."""
        ...

    def encode_candidate(self, image_url: str) -> FaceEncoding | None:
        """Encode the face in a candidate image, or ``None`` if unusable."""
        ...


@runtime_checkable
class MatchSelector(Protocol):
    """Picks the single most reliable matching post, or reports no match."""

    #: Minimum similarity required to call something a match.
    threshold: float

    def select(
        self, encoding: FaceEncoding, results: Sequence[SearchResult]
    ) -> MatchDecision:
        """Rank candidates and apply the threshold.

        Always returns a :class:`MatchDecision`; "no reliable match" is an
        expected outcome, not an exception.
        """
        ...


class PendingFaceMatcher:
    """STEP 1 placeholder for :class:`FaceMatcher`."""

    name = "pending-face-matcher"
    metric = "unset"

    def similarity(self, left: FaceEncoding, right: FaceEncoding) -> float:
        raise StageNotImplementedError("Face similarity scoring")

    def encode_candidate(self, image_url: str) -> FaceEncoding | None:
        raise StageNotImplementedError("Candidate face encoding")


class PendingMatchSelector:
    """STEP 1 placeholder for :class:`MatchSelector`."""

    name = "pending-match-selector"
    threshold = 0.0

    def select(
        self, encoding: FaceEncoding, results: Sequence[SearchResult]
    ) -> MatchDecision:
        raise StageNotImplementedError("Match selection")
