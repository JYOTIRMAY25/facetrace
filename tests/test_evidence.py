from types import SimpleNamespace

from app.services.canonical_evidence import canonicalize_evidence
from app.services.evidence_extraction import (
    classify_social_url,
    extract_match_evidence,
)


def _match(url="https://www.instagram.com/p/abc", score=0.91):
    candidate = SimpleNamespace(
        title="Observed result",
        url=url,
        source="instagram.com",
        image_url="https://cdn.example/image.jpg",
    )
    return SimpleNamespace(
        candidate=candidate,
        similarity_score=score,
        distance=1.0 - score,
        candidate_face_index=1,
        match_status="MATCH_FOUND",
    )


def _evidence(**kwargs):
    return extract_match_evidence(
        investigation_id="inv-1",
        source={"filename": "face.png", "mime_type": "image/png", "file_size": 10, "sha256": None},
        provider="test-provider",
        searched_at="2026-01-01T00:00:00Z",
        candidate_count=3,
        match=_match(**kwargs),
    )


def test_match_creates_complete_evidence():
    evidence = _evidence()
    assert evidence is not None
    assert evidence.match["similarity_score"] == 0.91
    assert evidence.match["distance"] == 0.08999999999999997
    assert evidence.match["candidate_face_index"] == 1
    assert evidence.social_media["platform"] == "instagram"


def test_non_match_does_not_create_evidence():
    match = _match()
    match.match_status = "NO_MATCH"
    assert _evidence() is not None
    assert extract_match_evidence(
        investigation_id="inv-1", source={}, provider="provider",
        searched_at="2026-01-01T00:00:00Z", candidate_count=1, match=match
    ) is None


def test_social_and_non_social_classification():
    assert classify_social_url("https://x.com/example")["is_social"] is True
    assert classify_social_url("https://example.org/item")["is_social"] is False


def test_canonicalization_is_deterministic_and_sensitive_to_changes():
    first = canonicalize_evidence(_evidence())
    second = canonicalize_evidence(_evidence())
    assert first.canonical_bytes == second.canonical_bytes
    changed = canonicalize_evidence(_evidence(score=0.92))
    assert changed.canonical_bytes != first.canonical_bytes


def test_evidence_excludes_embedding_and_path():
    evidence = _evidence().as_dict()
    serialized = canonicalize_evidence(evidence).canonical_json
    assert "embedding" not in serialized
    assert "filesystem" not in serialized
    assert "\\tmp\\" not in serialized
