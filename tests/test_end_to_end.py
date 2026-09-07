"""End-to-end integration tests for FaceTrace backend pipeline (Steps 1–7 integration)."""

from __future__ import annotations

import pytest
from pathlib import Path
from dataclasses import replace

from app.config import load_settings
from app.models.pipeline import (
    ErrorCode,
    MatchingPost,
    SearchOutcome,
    SearchResult,
    SearchStatus,
    VerificationStatus,
)
from app.orchestrator import Orchestrator, build_real_orchestrator
from app.services.blockchain_sim import InMemoryBlockchainService
from app.services.face_insightface import build_face_identifier
from app.services.fingerprint import Sha256FingerprintService
from app.services.matching import DefaultMatchSelector, InsightFaceMatcher
from app.services.verification_impl import HashVerificationService


class MockSearchProvider:
    name = "mock-search-provider"

    def __init__(self, results=()):
        self.results = results
        self.error = None

    def build_query(self, encoding):
        from app.models.pipeline import SearchQuery
        return SearchQuery(query_terms="mock query")

    def search_for_encoding(self, encoding):
        if self.error:
            return SearchOutcome(status=SearchStatus.ERROR, error=self.error, error_code=ErrorCode.SEARCH_FAILED)
        return SearchOutcome(status=SearchStatus.SUCCESS if self.results else SearchStatus.NO_RESULTS, results=tuple(self.results))

    def search(self, query):
        return self.results


class FailingBlockchainService:
    network = "local"

    def connect(self):
        return False

    def write_fingerprint(self, record_id, fingerprint):
        raise RuntimeError("Blockchain RPC connection refused")

    def read_fingerprint(self, record_id):
        raise RuntimeError("Blockchain read timeout")


def test_e2e_success(single_face_image: Path, face_identifier) -> None:
    """TEST 1 — SUCCESS: Full pipeline execution leading to VERIFIED."""
    settings = load_settings(env={})
    face_id = face_identifier
    
    candidate_result = SearchResult(
        result_id="res_001",
        source="Wikimedia Commons",
        url="https://commons.wikimedia.org/wiki/File:Eileen_Collins.jpg",
        title="Public domain photograph of Eileen Collins",
        text="Eileen Marie Collins is a retired NASA astronaut and United States Air Force colonel.",
        image_url=single_face_image.as_uri(),
        metadata={"license": "Public Domain"}
    )
    search_svc = MockSearchProvider(results=[candidate_result])
    matcher = InsightFaceMatcher(face_id)
    selector = DefaultMatchSelector(settings=settings, matcher=matcher)
    fingerprints = Sha256FingerprintService()
    blockchain = InMemoryBlockchainService()
    verification = HashVerificationService(fingerprints, blockchain)

    orchestrator = Orchestrator(
        settings=settings,
        face=face_id,
        search=search_svc,
        selector=selector,
        fingerprints=fingerprints,
        blockchain=blockchain,
        verification=verification,
    )

    result = orchestrator.run(str(single_face_image))

    assert result.ok is True
    assert result.status is VerificationStatus.VERIFIED
    assert result.encoding is not None
    assert result.match is not None
    assert result.match.matched is True
    assert result.match.post.title == "Public domain photograph of Eileen Collins"
    assert result.fingerprint is not None
    assert result.record is not None
    assert result.verification is not None
    assert result.verification.match is True


def test_e2e_no_face(flat_no_face_image: Path, face_identifier) -> None:
    """TEST 2 — NO FACE: Input image contains no face; stops before search."""
    settings = load_settings(env={})
    orchestrator = Orchestrator(settings=settings, face=face_identifier)
    result = orchestrator.run(str(flat_no_face_image))

    assert result.status is VerificationStatus.INCONCLUSIVE
    assert result.error is not None
    assert result.error.code is ErrorCode.NO_FACE_DETECTED
    assert result.match is None


def test_e2e_multiple_faces(multi_face_image: Path, face_identifier) -> None:
    """TEST 3 — MULTIPLE FACES: Input contains multiple faces; rejected."""
    settings = load_settings(env={})
    orchestrator = Orchestrator(settings=settings, face=face_identifier)
    result = orchestrator.run(str(multi_face_image))

    assert result.status is VerificationStatus.INCONCLUSIVE
    assert result.error is not None
    assert result.error.code is ErrorCode.MULTIPLE_FACES
    assert result.match is None


def test_e2e_no_search_results(single_face_image: Path, face_identifier) -> None:
    """TEST 4 — NO SEARCH RESULTS: Search returns 0 candidates."""
    settings = load_settings(env={})
    face_id = face_identifier
    search_svc = MockSearchProvider(results=[])
    matcher = InsightFaceMatcher(face_id)
    selector = DefaultMatchSelector(settings=settings, matcher=matcher)

    orchestrator = Orchestrator(
        settings=settings,
        face=face_id,
        search=search_svc,
        selector=selector,
    )

    result = orchestrator.run(str(single_face_image))

    assert result.status is VerificationStatus.NO_MATCH
    assert result.match is not None
    assert result.match.matched is False
    assert result.match.reason == "NO_CANDIDATES"


def test_e2e_no_match(single_face_image: Path, flat_no_face_image: Path, face_identifier) -> None:
    """TEST 5 — NO MATCH: Candidates evaluated but none pass similarity threshold."""
    settings = load_settings(env={})
    face_id = face_identifier
    
    candidate_result = SearchResult(
        result_id="res_002",
        source="Test Source",
        url="https://example.org/photo.jpg",
        title="Unrelated Image",
        text="No face here.",
        image_url=flat_no_face_image.as_uri(),
    )
    search_svc = MockSearchProvider(results=[candidate_result])
    matcher = InsightFaceMatcher(face_id)
    selector = DefaultMatchSelector(settings=settings, matcher=matcher)

    orchestrator = Orchestrator(
        settings=settings,
        face=face_id,
        search=search_svc,
        selector=selector,
    )

    result = orchestrator.run(str(single_face_image))

    assert result.status is VerificationStatus.NO_MATCH
    assert result.match is not None
    assert result.match.matched is False


def test_e2e_blockchain_failure(single_face_image: Path, face_identifier) -> None:
    """TEST 6 — BLOCKCHAIN FAILURE: Chain write/read fails."""
    settings = load_settings(env={})
    face_id = face_identifier
    candidate_result = SearchResult(
        result_id="res_003",
        source="Wikimedia Commons",
        url="https://commons.wikimedia.org/wiki/File:Eileen_Collins.jpg",
        title="Eileen Collins",
        text="Astronaut photo",
        image_url=single_face_image.as_uri(),
    )
    search_svc = MockSearchProvider(results=[candidate_result])
    matcher = InsightFaceMatcher(face_id)
    selector = DefaultMatchSelector(settings=settings, matcher=matcher)
    fingerprints = Sha256FingerprintService()
    failing_blockchain = FailingBlockchainService()

    orchestrator = Orchestrator(
        settings=settings,
        face=face_id,
        search=search_svc,
        selector=selector,
        fingerprints=fingerprints,
        blockchain=failing_blockchain,
    )

    result = orchestrator.run(str(single_face_image))

    assert result.status is VerificationStatus.BLOCKCHAIN_UNAVAILABLE
    assert result.status is not VerificationStatus.VERIFIED
    assert result.error is not None
    assert result.error.code is ErrorCode.BLOCKCHAIN_WRITE_FAILED


def test_e2e_tampering(single_face_image: Path, face_identifier) -> None:
    """TEST 7 — TAMPERING: Modify an evidence field -> NOT_VERIFIED."""
    settings = load_settings(env={})
    face_id = face_identifier
    candidate_result = SearchResult(
        result_id="res_004",
        source="Wikimedia Commons",
        url="https://commons.wikimedia.org/wiki/File:Eileen_Collins.jpg",
        title="Original Title",
        text="Original Text",
        image_url=single_face_image.as_uri(),
    )
    search_svc = MockSearchProvider(results=[candidate_result])
    matcher = InsightFaceMatcher(face_id)
    selector = DefaultMatchSelector(settings=settings, matcher=matcher)
    fingerprints = Sha256FingerprintService()
    blockchain = InMemoryBlockchainService()
    verification = HashVerificationService(fingerprints, blockchain)

    orchestrator = Orchestrator(
        settings=settings,
        face=face_id,
        search=search_svc,
        selector=selector,
        fingerprints=fingerprints,
        blockchain=blockchain,
        verification=verification,
    )

    result = orchestrator.run(str(single_face_image))
    assert result.status is VerificationStatus.VERIFIED

    # Tamper with title in the matching post
    tampered_post = replace(result.match.post, title="Modified Tampered Title")
    verify_res = verification.verify(tampered_post, result.record)

    assert verify_res.match is False
    assert verify_res.status is VerificationStatus.NOT_VERIFIED


def test_e2e_verification_success(single_face_image: Path, face_identifier) -> None:
    """TEST 8 — VERIFICATION SUCCESS: Untampered evidence remains VERIFIED."""
    settings = load_settings(env={})
    face_id = face_identifier
    candidate_result = SearchResult(
        result_id="res_005",
        source="Wikimedia Commons",
        url="https://commons.wikimedia.org/wiki/File:Eileen_Collins.jpg",
        title="Authentic Evidence Title",
        text="Authentic Text",
        image_url=single_face_image.as_uri(),
    )
    search_svc = MockSearchProvider(results=[candidate_result])
    matcher = InsightFaceMatcher(face_id)
    selector = DefaultMatchSelector(settings=settings, matcher=matcher)
    fingerprints = Sha256FingerprintService()
    blockchain = InMemoryBlockchainService()
    verification = HashVerificationService(fingerprints, blockchain)

    orchestrator = Orchestrator(
        settings=settings,
        face=face_id,
        search=search_svc,
        selector=selector,
        fingerprints=fingerprints,
        blockchain=blockchain,
        verification=verification,
    )

    result = orchestrator.run(str(single_face_image))
    assert result.status is VerificationStatus.VERIFIED

    # Verify original unchanged post
    verify_res = verification.verify(result.match.post, result.record)
    assert verify_res.match is True
    assert verify_res.status is VerificationStatus.VERIFIED