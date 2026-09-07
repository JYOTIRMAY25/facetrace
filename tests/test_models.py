"""Structured pipeline result models."""

from __future__ import annotations

import dataclasses
import json

import pytest

from app.models.pipeline import (
    CANONICAL_FIELD_ORDER,
    CANONICALIZATION_VERSION,
    PIPELINE_STAGES,
    BlockchainRecord,
    BoundingBox,
    CanonicalPayload,
    DetectedFace,
    ErrorCode,
    FaceEncoding,
    FaceInput,
    Fingerprint,
    MatchDecision,
    MatchingPost,
    PipelineError,
    PipelineResult,
    SearchQuery,
    SearchResult,
    Stage,
    StageResult,
    StageStatus,
    VerificationResult,
    VerificationStatus,
)


def test_pipeline_stage_order_matches_architecture() -> None:
    """Order mirrors SYSTEM_ARCHITECTURE.md; the CLI numbering depends on it."""
    assert PIPELINE_STAGES == (
        Stage.VALIDATE_INPUT,
        Stage.DETECT_FACE,
        Stage.ENCODE_FACE,
        Stage.SEARCH,
        Stage.SELECT_MATCH,
        Stage.FINGERPRINT,
        Stage.BLOCKCHAIN_WRITE,
        Stage.BLOCKCHAIN_READ,
        Stage.RECOMPUTE_FINGERPRINT,
        Stage.VERIFY,
    )
    assert len(set(PIPELINE_STAGES)) == len(PIPELINE_STAGES)


def test_verification_status_values_are_the_strings_the_cli_prints() -> None:
    assert VerificationStatus.VERIFIED.value == "VERIFIED"
    assert VerificationStatus.NOT_VERIFIED.value == "NOT VERIFIED"


def test_error_codes_cover_the_documented_failure_states() -> None:
    """IMPLEMENTATION_PLAN Phase 17 error list must exist as stable codes."""
    documented = {
        "INVALID_IMAGE",
        "NO_FACE_DETECTED",
        "MULTIPLE_FACES",
        "FACE_ENCODING_FAILED",
        "SEARCH_FAILED",
        "NO_SEARCH_RESULTS",
        "NO_RELIABLE_MATCH",
        "FINGERPRINT_FAILED",
        "BLOCKCHAIN_CONNECTION_FAILED",
        "BLOCKCHAIN_WRITE_FAILED",
        "BLOCKCHAIN_READ_FAILED",
        "VERIFICATION_FAILED",
    }
    assert documented <= {code.value for code in ErrorCode}


def test_canonicalization_contract_is_declared() -> None:
    assert CANONICALIZATION_VERSION == "1"
    assert CANONICAL_FIELD_ORDER == ("url", "source", "title", "text", "image_url")
    # A tuple, so the hashed field order cannot be mutated at runtime.
    assert isinstance(CANONICAL_FIELD_ORDER, tuple)


def test_stage_data_models_are_immutable() -> None:
    """Stage payloads are frozen so a later stage cannot rewrite history."""
    for model in (
        FaceInput("i1", "img.jpg"),
        BoundingBox(0, 0, 10, 10),
        DetectedFace(BoundingBox(0, 0, 10, 10)),
        FaceEncoding("i1", "model-x", "ref-1"),
        SearchQuery("q1"),
        SearchResult("r1", "example", "https://example.test/p/1"),
        MatchingPost("r1", "example", "https://example.test/p/1"),
        MatchDecision(matched=False),
        CanonicalPayload("1", CANONICAL_FIELD_ORDER, "payload"),
        Fingerprint(),
        BlockchainRecord("local", "rec1", "abc"),
        VerificationResult("a", "a", True, VerificationStatus.VERIFIED),
        PipelineError(ErrorCode.SEARCH_FAILED, Stage.SEARCH, "boom"),
    ):
        assert dataclasses.is_dataclass(model)
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(model, dataclasses.fields(model)[0].name, "mutated")


def test_face_encoding_holds_a_reference_not_a_vector() -> None:
    """Privacy: no raw embedding field exists on the model."""
    field_names = {f.name for f in dataclasses.fields(FaceEncoding)}
    assert "encoding_reference" in field_names
    assert not field_names & {"embedding", "vector", "descriptor", "raw"}


def test_no_match_is_a_first_class_state() -> None:
    decision = MatchDecision(
        matched=False, candidates_considered=7, threshold=0.65, reason="NO_RELIABLE_MATCH"
    )
    assert decision.matched is False
    assert decision.post is None
    assert decision.candidates_considered == 7


def test_pipeline_result_defaults_are_inconclusive() -> None:
    result = PipelineResult()
    assert result.status is VerificationStatus.INCONCLUSIVE
    assert result.stages == []
    assert result.error is None
    assert result.ok is False


def test_pipeline_result_lookup_and_ok_flag() -> None:
    result = PipelineResult(
        input_id="i1",
        stages=[
            StageResult(Stage.VALIDATE_INPUT, StageStatus.SUCCESS, duration_ms=5),
            StageResult(Stage.SEARCH, StageStatus.PENDING),
        ],
        status=VerificationStatus.VERIFIED,
    )

    found = result.stage_result(Stage.VALIDATE_INPUT)
    assert found is not None and found.status is StageStatus.SUCCESS
    assert result.stage_result(Stage.VERIFY) is None
    assert result.ok is True

    result.error = PipelineError(ErrorCode.SEARCH_FAILED, Stage.SEARCH, "boom")
    assert result.ok is False


def test_pipeline_result_is_json_serialisable() -> None:
    result = PipelineResult(
        input_id="i1",
        image_path="data/input.jpg",
        stages=[StageResult(Stage.VALIDATE_INPUT, StageStatus.SUCCESS)],
        status=VerificationStatus.NOT_VERIFIED,
        fingerprint=Fingerprint(hash="0" * 64),
        verification=VerificationResult("a" * 64, "b" * 64, False, VerificationStatus.NOT_VERIFIED),
    )

    payload = json.loads(json.dumps(result.to_dict(), default=str))
    assert payload["status"] == "NOT VERIFIED"
    assert payload["stages"][0]["stage"] == "validate_input"
    assert payload["fingerprint"]["algorithm"] == "SHA-256"
