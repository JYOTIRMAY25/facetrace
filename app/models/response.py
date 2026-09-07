"""Stable response models for the backend investigation API."""

from pydantic import BaseModel, Field


class BoundingBoxResponse(BaseModel):
    x: int
    y: int
    width: int
    height: int


class DetectedFaceResponse(BaseModel):
    face_index: int
    bbox: BoundingBoxResponse
    confidence: float


class CandidateResponse(BaseModel):
    title: str | None = None
    url: str
    source_domain: str
    image_url: str | None = None
    thumbnail_url: str | None = None
    provider: str
    candidate_face_index: int | None = None
    distance: float | None = None
    similarity_score: float | None = None
    match_status: str


class SocialMediaEvidenceResponse(BaseModel):
    is_social: bool
    platform: str | None = None
    url: str | None = None


class EvidenceResponse(BaseModel):
    evidence_version: str
    search_provider: str
    candidate_count: int
    social_media: SocialMediaEvidenceResponse


class CanonicalizationResponse(BaseModel):
    status: str


class FingerprintResponse(BaseModel):
    algorithm: str | None = None
    value: str | None = None
    input_encoding: str | None = None
    status: str


class BlockchainResponse(BaseModel):
    status: str
    network: str | None = None
    chain_id: int | None = None
    transaction_hash: str | None = None
    block_number: int | None = None
    contract_address: str | None = None
    evidence_hash: str | None = None
    explorer_url: str | None = None


class InvestigationResponse(BaseModel):
    investigation_id: str | None = None
    status: str
    message: str
    faces_detected: int = 0
    faces: list[DetectedFaceResponse] = Field(default_factory=list)
    encoding_generated: bool = False
    embedding_dimensions: list[int] = Field(default_factory=list)
    detection_time_ms: int = 0
    encoding_time_ms: int = 0
    total_time_ms: int = 0
    provider: str | None = None
    candidates: list[CandidateResponse] = Field(default_factory=list)
    search_time_ms: int = 0
    matches: list[CandidateResponse] = Field(default_factory=list)
    evidence: EvidenceResponse | None = None
    canonicalization: CanonicalizationResponse | None = None
    fingerprint: FingerprintResponse | None = None
    blockchain: BlockchainResponse | None = None


class ErrorResponse(BaseModel):
    code: str
    message: str
