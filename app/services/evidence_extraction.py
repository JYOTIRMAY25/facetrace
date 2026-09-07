"""Build safe evidence records from observed pipeline results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlsplit

SOCIAL_PLATFORMS = {
    "instagram.com": "instagram",
    "x.com": "x",
    "twitter.com": "twitter",
    "facebook.com": "facebook",
    "threads.net": "threads",
    "tiktok.com": "tiktok",
}


@dataclass(frozen=True)
class EvidenceRecord:
    investigation_id: str
    evidence_version: str
    match_status: str
    source: Mapping[str, Any]
    search: Mapping[str, Any]
    match: Mapping[str, Any] | None
    social_media: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "investigation_id": self.investigation_id,
            "evidence_version": self.evidence_version,
            "match_status": self.match_status,
            "source": dict(self.source),
            "search": dict(self.search),
            "match": dict(self.match) if self.match is not None else None,
            "social_media": dict(self.social_media),
        }


def classify_social_url(url: str | None) -> dict[str, Any]:
    """Classify only domains that are unambiguously social platforms."""
    if not url:
        return {"is_social": False, "platform": None, "url": None}
    hostname = (urlsplit(url).hostname or "").lower().removeprefix("www.")
    platform = SOCIAL_PLATFORMS.get(hostname)
    if platform is None:
        return {"is_social": False, "platform": None, "url": url}
    return {"is_social": True, "platform": platform, "url": url}


def extract_match_evidence(
    *,
    investigation_id: str,
    source: Mapping[str, Any],
    provider: str,
    searched_at: str,
    candidate_count: int,
    match: Any,
) -> EvidenceRecord | None:
    """Convert one actual MATCH_FOUND result into an evidence record."""
    if not provider or not searched_at or match.match_status != "MATCH_FOUND":
        return None
    candidate = match.candidate
    if not candidate.url or match.similarity_score is None or match.distance is None:
        return None
    match_data = {
        "title": candidate.title or None,
        "url": candidate.url,
        "source_domain": candidate.source,
        "image_url": candidate.image_url,
        "similarity_score": match.similarity_score,
        "distance": match.distance,
        "candidate_face_index": match.candidate_face_index,
        "match_status": match.match_status,
    }
    return EvidenceRecord(
        investigation_id=investigation_id,
        evidence_version="1",
        match_status="MATCH_FOUND",
        source=dict(source),
        search={
            "provider": provider,
            "searched_at": searched_at,
            "candidate_count": candidate_count,
            "provider_status": "success",
        },
        match=match_data,
        social_media=classify_social_url(candidate.url),
    )
