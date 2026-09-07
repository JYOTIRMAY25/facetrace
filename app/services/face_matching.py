"""Candidate retrieval, face encoding, similarity, and deterministic ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..models.pipeline import FaceEncoding, SearchResult
from .candidate_image_retrieval import (
    CandidateImageRetrievalError,
    CandidateImageRetriever,
)
from .face_insightface import EmbeddingVector, InsightFaceIdentifier
from .matching import InsightFaceMatcher


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
    results: list[CandidateMatchResult] = []
    for candidate in candidates:
        if not candidate.image_url:
            results.append(CandidateMatchResult(candidate, None, None, None, "IMAGE_RETRIEVAL_FAILED", "No candidate image URL."))
            continue
        try:
            image = retriever.retrieve(candidate.image_url)
        except CandidateImageRetrievalError as exc:
            results.append(
                CandidateMatchResult(
                    candidate, None, None, None, exc.code, str(exc)
                )
            )
            continue
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
