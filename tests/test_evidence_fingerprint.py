import hashlib

from app.services.canonical_evidence import canonicalize_evidence
from app.services.evidence_extraction import extract_match_evidence
from app.services.evidence_fingerprint import (
    fingerprint_evidence,
    verify_evidence_fingerprint,
)
from tests.test_evidence import _evidence


def test_known_sha256_vector_is_lowercase_64_hex():
    result = fingerprint_evidence(b"abc")
    assert result.fingerprint == hashlib.sha256(b"abc").hexdigest()
    assert len(result.fingerprint) == 64
    assert result.fingerprint == result.fingerprint.lower()
    assert all(character in "0123456789abcdef" for character in result.fingerprint)


def test_same_canonical_bytes_have_same_fingerprint():
    evidence = _evidence()
    first = fingerprint_evidence(canonicalize_evidence(evidence).canonical_bytes)
    second = fingerprint_evidence(canonicalize_evidence(evidence).canonical_bytes)
    assert first == second


def test_meaningful_evidence_changes_tamper_fingerprint():
    original = _evidence()
    original_bytes = canonicalize_evidence(original).canonical_bytes
    assert verify_evidence_fingerprint(
        original_bytes, fingerprint_evidence(original_bytes).fingerprint
    )

    changed_evidence = _evidence(url="https://example.org/changed")
    changed_evidence_score = _evidence(score=0.92)
    for changed in (changed_evidence, changed_evidence_score):
        changed_bytes = canonicalize_evidence(changed).canonical_bytes
        assert changed_bytes != original_bytes
        assert not verify_evidence_fingerprint(
            changed_bytes, fingerprint_evidence(original_bytes).fingerprint
        )


def test_provider_and_timestamp_changes_tamper_fingerprint():
    evidence = _evidence()
    original_bytes = canonicalize_evidence(evidence).canonical_bytes
    provider_changed = type(evidence)(
        investigation_id=evidence.investigation_id,
        evidence_version=evidence.evidence_version,
        match_status=evidence.match_status,
        source=evidence.source,
        search={**evidence.search, "provider": "other-provider"},
        match=evidence.match,
        social_media=evidence.social_media,
    )
    timestamp_changed = type(evidence)(
        investigation_id=evidence.investigation_id,
        evidence_version=evidence.evidence_version,
        match_status=evidence.match_status,
        source=evidence.source,
        search={**evidence.search, "searched_at": "2026-01-02T00:00:00Z"},
        match=evidence.match,
        social_media=evidence.social_media,
    )
    assert canonicalize_evidence(provider_changed).canonical_bytes != original_bytes
    assert canonicalize_evidence(timestamp_changed).canonical_bytes != original_bytes


def test_invalid_expected_fingerprint_is_rejected():
    assert not verify_evidence_fingerprint(b"abc", "not-a-hash")


def test_incomplete_match_does_not_produce_evidence():
    match = _evidence().match
    assert match is not None
    incomplete = type("Match", (), {
        "match_status": "NO_MATCH",
        "candidate": type("Candidate", (), {"url": "https://example.org"})(),
        "similarity_score": None,
        "distance": None,
    })()
    assert extract_match_evidence(
        investigation_id="inv",
        source={},
        provider="provider",
        searched_at="2026-01-01T00:00:00Z",
        candidate_count=1,
        match=incomplete,
    ) is None
