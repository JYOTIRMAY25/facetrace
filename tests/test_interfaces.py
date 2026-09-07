"""Service interfaces are replaceable, and STEP 1 placeholders invent nothing.

The pipeline contract under test:

    FaceInput -> FaceIdentifier -> SearchProvider -> MatchSelector
    -> FingerprintService -> BlockchainService -> VerificationService
"""

from __future__ import annotations

from typing import Sequence

import pytest

from app.models.pipeline import (
    BlockchainRecord,
    BoundingBox,
    CanonicalPayload,
    DetectedFace,
    FaceEncoding,
    FaceInput,
    Fingerprint,
    MatchDecision,
    MatchingPost,
    SearchQuery,
    SearchResult,
    VerificationResult,
    VerificationStatus,
)
from app.services import ServiceError, StageNotImplementedError
from app.services.blockchain import BlockchainService, PendingBlockchainService
from app.services.face import FaceIdentifier, PendingFaceIdentifier
from app.services.fingerprint import FingerprintService, PendingFingerprintService
from app.services.matching import (
    FaceMatcher,
    MatchSelector,
    PendingFaceMatcher,
    PendingMatchSelector,
)
from app.services.search import (
    PendingSearchProvider,
    PendingSearchResultParser,
    SearchProvider,
    SearchResultParser,
)
from app.services.verification import PendingVerificationService, VerificationService

PLACEHOLDERS_AND_PROTOCOLS = [
    (PendingFaceIdentifier(), FaceIdentifier),
    (PendingSearchProvider(), SearchProvider),
    (PendingSearchResultParser(), SearchResultParser),
    (PendingFaceMatcher(), FaceMatcher),
    (PendingMatchSelector(), MatchSelector),
    (PendingFingerprintService(), FingerprintService),
    (PendingBlockchainService(), BlockchainService),
    (PendingVerificationService(), VerificationService),
]


@pytest.mark.parametrize(
    ("placeholder", "protocol"),
    PLACEHOLDERS_AND_PROTOCOLS,
    ids=lambda value: getattr(value, "__name__", type(value).__name__),
)
def test_placeholder_satisfies_its_protocol(placeholder: object, protocol: type) -> None:
    assert isinstance(placeholder, protocol)


def test_step_not_implemented_is_a_service_error_and_not_retryable() -> None:
    error = StageNotImplementedError("Face detection")
    assert isinstance(error, ServiceError)
    assert error.code == "NOT_IMPLEMENTED"
    assert error.retryable is False
    assert "STEP 2" in str(error)


def test_service_error_carries_code_and_retry_flag() -> None:
    error = ServiceError("provider timed out", code="SEARCH_FAILED", retryable=True)
    assert error.code == "SEARCH_FAILED"
    assert error.retryable is True


def test_face_placeholder_refuses_to_produce_data() -> None:
    face = PendingFaceIdentifier()
    detected = DetectedFace(BoundingBox(0, 0, 1, 1))
    face_input = FaceInput("i1", "img.jpg")

    with pytest.raises(StageNotImplementedError):
        face.load("img.jpg")
    with pytest.raises(StageNotImplementedError):
        face.detect(face_input)
    with pytest.raises(StageNotImplementedError):
        face.encode(face_input, detected)


def test_search_placeholder_refuses_to_fabricate_results() -> None:
    provider = PendingSearchProvider()
    encoding = FaceEncoding("i1", "model-x", "ref-1")

    with pytest.raises(StageNotImplementedError):
        provider.build_query(encoding)
    with pytest.raises(StageNotImplementedError):
        provider.search(SearchQuery("q1"))
    with pytest.raises(StageNotImplementedError):
        PendingSearchResultParser().parse({}, source="example")


def test_matching_placeholder_refuses_to_force_a_match() -> None:
    encoding = FaceEncoding("i1", "model-x", "ref-1")
    results: Sequence[SearchResult] = [SearchResult("r1", "example", "https://example.test")]

    with pytest.raises(StageNotImplementedError):
        PendingMatchSelector().select(encoding, results)
    with pytest.raises(StageNotImplementedError):
        PendingFaceMatcher().similarity(encoding, encoding)
    with pytest.raises(StageNotImplementedError):
        PendingFaceMatcher().encode_candidate("https://example.test/img.jpg")


def test_fingerprint_placeholder_declares_algorithm_but_computes_nothing() -> None:
    service = PendingFingerprintService()
    post = MatchingPost("r1", "example", "https://example.test/p/1")

    assert service.algorithm == "SHA-256"
    assert service.canonicalization_version == "1"
    with pytest.raises(StageNotImplementedError):
        service.canonicalize(post)
    with pytest.raises(StageNotImplementedError):
        service.fingerprint(post)


def test_blockchain_placeholder_refuses_to_write_or_read() -> None:
    service = PendingBlockchainService()

    with pytest.raises(StageNotImplementedError):
        service.connect()
    with pytest.raises(StageNotImplementedError):
        service.write_fingerprint("rec1", Fingerprint(hash="0" * 64))
    with pytest.raises(StageNotImplementedError):
        service.read_fingerprint("rec1")


def test_verification_placeholder_refuses_to_declare_a_verdict() -> None:
    service = PendingVerificationService()
    post = MatchingPost("r1", "example", "https://example.test/p/1")
    record = BlockchainRecord("local", "rec1", "0" * 64)

    with pytest.raises(StageNotImplementedError):
        service.recompute(post)
    with pytest.raises(StageNotImplementedError):
        service.verify(post, record)


def test_a_custom_implementation_can_replace_a_placeholder() -> None:
    """Proves the protocols are structural: no inheritance required."""

    class StubFingerprintService:
        algorithm = "SHA-256"
        canonicalization_version = "1"

        def canonicalize(self, post: MatchingPost) -> CanonicalPayload:
            return CanonicalPayload("1", ("url",), post.url)

        def fingerprint(self, post: MatchingPost) -> Fingerprint:
            return Fingerprint(hash="f" * 64)

    stub = StubFingerprintService()
    assert isinstance(stub, FingerprintService)
    assert stub.fingerprint(MatchingPost("r1", "example", "https://example.test")).hash == "f" * 64


def test_verification_result_can_express_both_verdicts() -> None:
    verified = VerificationResult("a" * 64, "a" * 64, True, VerificationStatus.VERIFIED)
    tampered = VerificationResult("a" * 64, "b" * 64, False, VerificationStatus.NOT_VERIFIED)

    assert verified.status.value == "VERIFIED"
    assert tampered.status.value == "NOT VERIFIED"


def test_match_decision_shape_supports_a_clear_no_match() -> None:
    decision = MatchDecision(matched=False, reason="NO_RELIABLE_MATCH", threshold=0.65)
    assert decision.post is None
    assert decision.reason == "NO_RELIABLE_MATCH"
