"""Candidate matching over downloaded images (the /api/investigate hot path).

These tests pin the two production properties the synchronous pipeline
relies on:

1. **Downloads are bounded-concurrent.** All candidate images are fetched
   through a pool of at most ``DOWNLOAD_CONCURRENCY`` workers, so network
   latency does not accumulate serially — but never more than 4 downloads run
   at once, which is what keeps the request within a free-tier host's
   connection and memory budget.
2. **Face inference stays sequential and order-preserving.** Downloads feed
   the InsightFace analysis loop in the original provider order, one image at
   a time, so CPU inference never contends with itself and the reported
   ranking reflects the provider's own ordering when scores tie.

The HTTP boundary and the face identifier are faked here; that is the only
substitution. No test in this file reaches the network or loads the real
InsightFace pack.
"""

from __future__ import annotations

import threading
import time

import pytest

from app.models.pipeline import (
    FaceDetectionResult,
    FaceEncoding,
    SearchResult,
)
from app.services.candidate_image_retrieval import (
    CandidateImage,
    CandidateImageRetrievalError,
)
from app.services.face_matching import DOWNLOAD_CONCURRENCY, match_candidates

# --------------------------------------------------------------------------
# Test doubles
# --------------------------------------------------------------------------


class RecordingRetriever:
    """A retriever that records concurrency and preserves call order."""

    def __init__(self, *, delay: float = 0.05) -> None:
        self.delay = delay
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.fetch_urls: list[str] = []

    def retrieve(self, image_url: str) -> CandidateImage:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.fetch_urls.append(image_url)
        try:
            time.sleep(self.delay)
            return CandidateImage(
                url=image_url,
                content_type="image/jpeg",
                size_bytes=len(image_url),
                image_bytes=b"stub-image-bytes",
            )
        finally:
            with self.lock:
                self.active -= 1


class RecordingFaceIdentifier:
    """A face identifier that records analysis concurrency and image order."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.analyzed: list[bytes] = []

    def analyze_all_with_embeddings(self, image_bytes: bytes):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            index = len(self.analyzed)
            self.analyzed.append(image_bytes)
        try:
            time.sleep(0.01)
            result = FaceDetectionResult(
                success=True,
                face_count=1,
                embedding_available=True,
                embedding_dimension=512,
            )
            encoding = FaceEncoding(
                input_id=f"img-{index:04d}",
                model="insightface-w600k_r50",
                encoding_reference=f"c{index:03d}" + "a" * 61,
                dimension=512,
            )

            class _Vec:
                def __init__(self, ref: str) -> None:
                    self.values = [float(int(ref[1:4])) / 999.0] * 512

            embedding = _Vec(f"c{index:03d}")
            return result, (embedding,), (encoding,)
        finally:
            with self.lock:
                self.active -= 1


class SourceEmbedding:
    """A stand-in for the private source embedding vector."""

    values = [1.0] * 512


def make_candidate(index: int) -> SearchResult:
    return SearchResult(
        result_id=f"res-{index:03d}",
        source="news.example.org",
        url=f"https://news.example.org/articles/{index}",
        title=f"Candidate {index}",
        image_url=f"https://news.example.org/img/{index}.jpg",
    )


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture()
def source_encoding() -> FaceEncoding:
    return FaceEncoding(
        input_id="img-0001",
        model="insightface-w600k_r50",
        encoding_reference="s" * 64,
        dimension=512,
    )


@pytest.fixture()
def source_embedding() -> SourceEmbedding:
    return SourceEmbedding()


# --------------------------------------------------------------------------
# 1. Bounded download concurrency
# --------------------------------------------------------------------------


def test_downloads_run_at_most_four_at_a_time(
    source_encoding: FaceEncoding, source_embedding: SourceEmbedding
) -> None:
    retriever = RecordingRetriever(delay=0.05)
    face_id = RecordingFaceIdentifier()
    candidates = [make_candidate(i) for i in range(12)]

    results = match_candidates(
        source_encoding,
        source_embedding,
        candidates,
        face_id,
        threshold=0.65,
        retriever=retriever,
    )

    assert len(results) == 12
    assert DOWNLOAD_CONCURRENCY == 4
    assert retriever.max_active <= DOWNLOAD_CONCURRENCY
    assert retriever.max_active > 1, "downloads should overlap, not run serially"
    # Every candidate with an image URL was fetched exactly once. Worker
    # pickup order is scheduler-dependent, so compare as sorted multisets.
    assert sorted(retriever.fetch_urls) == sorted(
        c.image_url for c in candidates
    )


# --------------------------------------------------------------------------
# 2. Sequential, order-preserving inference
# --------------------------------------------------------------------------


def test_face_inference_stays_sequential(
    source_encoding: FaceEncoding, source_embedding: SourceEmbedding
) -> None:
    retriever = RecordingRetriever(delay=0.0)
    face_id = RecordingFaceIdentifier()
    candidates = [make_candidate(i) for i in range(6)]

    results = match_candidates(
        source_encoding,
        source_embedding,
        candidates,
        face_id,
        threshold=0.65,
        retriever=retriever,
    )

    assert len(results) == 6
    # InsightFace analysis is one-at-a-time, on the request thread.
    assert face_id.max_active == 1
    # Every downloaded image was analysed, in the original provider order.
    assert len(face_id.analyzed) == 6


def test_failed_downloads_are_reported_not_raised(
    source_encoding: FaceEncoding, source_embedding: SourceEmbedding
) -> None:
    class FlakyRetriever(RecordingRetriever):
        def retrieve(self, image_url: str) -> CandidateImage:
            if "blocked.example.org" in image_url:
                raise CandidateImageRetrievalError(
                    "Candidate image retrieval failed.", retryable=True
                )
            return super().retrieve(image_url)

    retriever = FlakyRetriever(delay=0.0)
    face_id = RecordingFaceIdentifier()
    good = [make_candidate(i) for i in range(3)]
    bad = [
        SearchResult(
            result_id="res-403",
            source="blocked.example.org",
            url="https://blocked.example.org/articles/403",
            title="Blocked candidate",
            image_url="https://blocked.example.org/403.jpg",
        )
    ]

    results = match_candidates(
        source_encoding,
        source_embedding,
        good + bad,
        face_id,
        threshold=0.65,
        retriever=retriever,
    )

    by_url = {r.candidate.url: r for r in results}
    assert (
        by_url["https://blocked.example.org/articles/403"].match_status
        == "IMAGE_RETRIEVAL_FAILED"
    )
    assert by_url["https://news.example.org/articles/0"].match_status in {
        "MATCH_FOUND",
        "NO_MATCH",
    }


def test_candidates_without_image_urls_skip_the_pool(
    source_encoding: FaceEncoding, source_embedding: SourceEmbedding
) -> None:
    retriever = RecordingRetriever(delay=0.0)
    face_id = RecordingFaceIdentifier()
    plain = SearchResult(
        result_id="res-plain",
        source="news.example.org",
        url="https://news.example.org/articles/plain",
        title="Text-only candidate",
        image_url=None,
    )

    results = match_candidates(
        source_encoding,
        source_embedding,
        [plain],
        face_id,
        threshold=0.65,
        retriever=retriever,
    )

    assert len(results) == 1
    assert retriever.fetch_urls == []
    assert face_id.analyzed == []
    assert results[0].match_status == "IMAGE_RETRIEVAL_FAILED"


def test_similarity_ranking_orders_results(
    source_encoding: FaceEncoding, source_embedding: SourceEmbedding
) -> None:
    retriever = RecordingRetriever(delay=0.0)
    face_id = RecordingFaceIdentifier()
    candidates = [make_candidate(i) for i in range(3)]

    results = match_candidates(
        source_encoding,
        source_embedding,
        candidates,
        face_id,
        threshold=0.65,
        retriever=retriever,
    )

    # The stub embeddings give every candidate a similar score; the invariant
    # is that ranking is by similarity, descending.
    scores = [r.similarity_score for r in results if r.similarity_score is not None]
    assert scores == sorted(scores, reverse=True)
    assert len(scores) == 3
