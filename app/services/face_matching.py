"""Candidate retrieval, face encoding, similarity, and deterministic ranking."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Sequence

from ..models.pipeline import FaceEncoding, SearchResult
from .candidate_image_retrieval import (
    CandidateImage,
    CandidateImageRetrievalError,
    CandidateImageRetriever,
)
from .face_insightface import EmbeddingVector, InsightFaceIdentifier
from .matching import InsightFaceMatcher

#: Candidate image downloads run through a small bounded pool so network
#: latency does not accumulate serially. Face inference stays sequential on
#: the request thread: concurrent CPU inference would contend on one core.
DOWNLOAD_CONCURRENCY = 4


@dataclass(frozen=True)
class CandidateMatchResult:
    candidate: SearchResult
    candidate_face_index: int | None
    distance: float | None
    similarity_score: float | None
    match_status: str
    error: str | None = None


def match_candidates(
    source_encoding: FaceEncoding,
    source_embedding: EmbeddingVector,
    candidates: Sequence[SearchResult],
    face_identifier: InsightFaceIdentifier,
    *,
    threshold: float,
    retriever: CandidateImageRetriever | None = None,
) -> list[CandidateMatchResult]:
    retriever = retriever or CandidateImageRetriever()
    matcher = InsightFaceMatcher(face_identifier)
    matcher.register_embedding(source_encoding, source_embedding)

    # Download phase: bounded-concurrent, results keyed by candidate index so
    # the analysis loop below consumes them in the original provider order.
    downloads: dict[int, CandidateImage | CandidateImageRetrievalError] = {}
    with ThreadPoolExecutor(max_workers=DOWNLOAD_CONCURRENCY) as pool:
        futures = {
            index: pool.submit(retriever.retrieve, candidate.image_url)
            for index, candidate in enumerate(candidates)
            if candidate.image_url
        }
        for index, future in futures.items():
            try:
                downloads[index] = future.result()
            except CandidateImageRetrievalError as exc:
                downloads[index] = exc

    results: list[CandidateMatchResult] = []
    for index, candidate in enumerate(candidates):
        if not candidate.image_url:
            results.append(CandidateMatchResult(candidate, None, None, None, "IMAGE_RETRIEVAL_FAILED", "No candidate image URL."))
            continue
        downloaded = downloads.get(index)
        if isinstance(downloaded, CandidateImageRetrievalError):
            results.append(
                CandidateMatchResult(
                    candidate, None, None, None, downloaded.code, str(downloaded)
                )
            )
            continue
        image = downloaded
        assert image is not None  # every indexed candidate has a completed download
        result, embeddings, encodings = face_identifier.analyze_all_with_embeddings(
            image.image_bytes
        )
        if result.error_code is not None:
            results.append(
                CandidateMatchResult(
                    candidate,
                    None,
                    None,
                    None,
                    "IMAGE_INVALID" if result.error_code.value == "INVALID_IMAGE" else "MATCH_ERROR",
                    result.error or None,
                )
            )
            continue
        if result.face_count == 0:
            results.append(CandidateMatchResult(candidate, None, None, None, "NO_FACE_IN_CANDIDATE"))
            continue
        if not embeddings or not encodings:
            results.append(CandidateMatchResult(candidate, None, None, None, "MATCH_ERROR"))
            continue
        best_index = 0
        best_score = -1.0
        for index, (candidate_embedding, candidate_encoding) in enumerate(zip(embeddings, encodings)):
            matcher.register_embedding(candidate_encoding, candidate_embedding)
            score = matcher.similarity(source_encoding, candidate_encoding)
            if score > best_score:
                best_score = score
                best_index = index
        score = best_score
        distance = 1.0 - score
        status = "MATCH_FOUND" if score >= threshold else "NO_MATCH"
        results.append(CandidateMatchResult(candidate, best_index, distance, score, status))
    return sorted(
        results,
        key=lambda item: item.similarity_score if item.similarity_score is not None else -1.0,
        reverse=True,
    )
