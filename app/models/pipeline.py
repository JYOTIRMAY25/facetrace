"""Typed models for every FaceTrace pipeline stage.

These are data carriers only — no business logic lives here. Stages are
expected to consume and return these types instead of untyped dictionaries so
failures stay debuggable and the CLI can render structured output.

Field sets follow ``../../DATA_MODEL.md``.

Privacy note: ``FaceEncoding`` deliberately stores an *encoding reference*
(an opaque local handle or digest), not a raw embedding vector. Raw biometric
data must never be written to ``data/``, to logs, or on-chain.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional

__all__ = [
    "CANONICAL_FIELD_ORDER",
    "CANONICALIZATION_VERSION",
    "PIPELINE_STAGES",
    "BlockchainRecord",
    "BoundingBox",
    "CanonicalPayload",
    "DetectedFace",
    "ErrorCode",
    "FaceDetectionResult",
    "FaceEncoding",
    "FaceInput",
    "Fingerprint",
    "MatchDecision",
    "MatchingPost",
    "PipelineError",
    "PipelineResult",
    "SearchOutcome",
    "SearchQuery",
    "SearchResult",
    "SearchStatus",
    "Stage",
    "StageResult",
    "StageStatus",
    "VerificationResult",
    "VerificationStatus",
]


# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------


class Stage(str, Enum):
    """Ordered pipeline stages. Values are stable identifiers for logs/JSON."""

    VALIDATE_INPUT = "validate_input"
    DETECT_FACE = "detect_face"
    ENCODE_FACE = "encode_face"
    SEARCH = "search"
    SELECT_MATCH = "select_match"
    FINGERPRINT = "fingerprint"
    BLOCKCHAIN_WRITE = "blockchain_write"
    BLOCKCHAIN_READ = "blockchain_read"
    RECOMPUTE_FINGERPRINT = "recompute_fingerprint"
    VERIFY = "verify"


#: Canonical execution order used by the orchestrator and the CLI display.
PIPELINE_STAGES: tuple[Stage, ...] = (
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


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    NOT_IMPLEMENTED = "not_implemented"


class ErrorCode(str, Enum):
    """Stable error codes surfaced by the CLI (see IMPLEMENTATION_PLAN Phase 17)."""

    INVALID_IMAGE = "INVALID_IMAGE"
    NO_FACE_DETECTED = "NO_FACE_DETECTED"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    FACE_ENCODING_FAILED = "FACE_ENCODING_FAILED"
    #: The local model pack could not be loaded (missing files / no network on
    #: first use). Distinct from FACE_ENCODING_FAILED so it is diagnosable.
    FACE_MODEL_UNAVAILABLE = "FACE_MODEL_UNAVAILABLE"
    SEARCH_FAILED = "SEARCH_FAILED"
    NO_SEARCH_RESULTS = "NO_SEARCH_RESULTS"
    #: The selected provider is missing credentials or settings it needs. Kept
    #: separate from SEARCH_FAILED so "you have not configured this" is never
    #: mistaken for "the provider is broken".
    SEARCH_NOT_CONFIGURED = "SEARCH_NOT_CONFIGURED"
    #: No usable query could be built from the input. The search stage refuses
    #: to invent query terms, so this is a failure rather than a blank search.
    SEARCH_QUERY_INVALID = "SEARCH_QUERY_INVALID"
    #: The candidate carried no publicly accessible media reference, so there was
    #: nothing legitimate to compare. Never treated as a weak match.
    CANDIDATE_MEDIA_MISSING = "CANDIDATE_MEDIA_MISSING"
    #: The candidate's media reference existed but could not be retrieved
    #: (HTTP failure, blocked by the host, oversized, non-image response).
    CANDIDATE_MEDIA_UNAVAILABLE = "CANDIDATE_MEDIA_UNAVAILABLE"
    NO_RELIABLE_MATCH = "NO_RELIABLE_MATCH"
    FINGERPRINT_FAILED = "FINGERPRINT_FAILED"
    BLOCKCHAIN_CONNECTION_FAILED = "BLOCKCHAIN_CONNECTION_FAILED"
    BLOCKCHAIN_WRITE_FAILED = "BLOCKCHAIN_WRITE_FAILED"
    BLOCKCHAIN_READ_FAILED = "BLOCKCHAIN_READ_FAILED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    CONFIG_INVALID = "CONFIG_INVALID"
    #: STEP 1 sentinel — a stage exists but its implementation lands in STEP 2.
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"


class VerificationStatus(str, Enum):
    """Terminal pipeline verdict. Values are the exact strings the CLI prints."""

    VERIFIED = "VERIFIED"
    NOT_VERIFIED = "NOT VERIFIED"
    #: Pipeline stopped before a comparison could be made.
    INCONCLUSIVE = "INCONCLUSIVE"
    NO_MATCH = "NO_MATCH"
    CANDIDATE_MEDIA_UNAVAILABLE = "CANDIDATE_MEDIA_UNAVAILABLE"
    BLOCKCHAIN_UNAVAILABLE = "BLOCKCHAIN_UNAVAILABLE"
    FINGERPRINT_UNAVAILABLE = "FINGERPRINT_UNAVAILABLE"
    INVALID_EVIDENCE = "INVALID_EVIDENCE"
    VERIFICATION_ERROR = "VERIFICATION_ERROR"


class SearchStatus(str, Enum):
    """Outcome of one search-stage execution.

    ``NO_RESULTS`` is a legitimate answer, not a failure: a provider that
    genuinely found nothing must say so. It must never be upgraded into a
    match, and the pipeline must not substitute invented content for it.
    """

    SUCCESS = "SUCCESS"
    NO_RESULTS = "NO_RESULTS"
    ERROR = "ERROR"


# --------------------------------------------------------------------------
# Canonicalization contract
# --------------------------------------------------------------------------

#: Bump when the canonical form changes; recorded alongside every fingerprint
#: so an old on-chain record stays interpretable.
CANONICALIZATION_VERSION: str = "1"

#: Exact field order hashed by the fingerprint service. Verification must reuse
#: this tuple so the recomputed hash is byte-identical.
CANONICAL_FIELD_ORDER: tuple[str, ...] = (
    "url",
    "source",
    "title",
    "text",
    "image_url",
)


# --------------------------------------------------------------------------
# Face stage
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FaceInput:
    """A validated image submitted to the pipeline."""

    input_id: str
    image_path: str
    face_count: int = 0
    width: int = 0
    height: int = 0
    #: Digest of the image bytes — lets a run be reproduced without the file.
    image_sha256: str = ""


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class DetectedFace:
    bounding_box: BoundingBox
    detector_confidence: float = 0.0
    detector_model: str = ""


@dataclass(frozen=True)
class FaceEncoding:
    """A face representation. ``encoding_reference`` is opaque, not a vector."""

    input_id: str
    model: str
    encoding_reference: str
    dimension: int = 0


@dataclass(frozen=True)
class FaceDetectionResult:
    """Structured outcome of the validate -> detect -> encode stage.

    Deliberately carries no embedding values — only ``embedding_available`` and
    ``embedding_dimension``. The vector itself stays in memory, behind
    ``app.services.face_insightface.EmbeddingVector``, and is never persisted,
    serialised, or logged.

    ``success`` is true only for exactly one detected face with an embedding.
    Zero faces and more than one face are both structured failures, not
    exceptions.
    """

    success: bool
    face_count: int = 0
    bbox: Optional[BoundingBox] = None
    confidence: float = 0.0
    embedding_available: bool = False
    embedding_dimension: int = 0
    error: str = ""
    error_code: Optional[ErrorCode] = None
    detector_model: str = ""
    encoder_model: str = ""
    #: Reference-only encoding record; ``None`` unless ``success`` is true.
    encoding: Optional[FaceEncoding] = None
    #: Set when the input was a file; empty for in-memory bytes.
    image_path: str = ""
    image_sha256: str = ""
    #: Every detected face, with metadata only. Embeddings remain in memory.
    faces: tuple[DetectedFace, ...] = ()
    detection_duration_ms: int = 0
    encoding_duration_ms: int = 0
    total_duration_ms: int = 0


# --------------------------------------------------------------------------
# Search stage
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchQuery:
    """A query derived from the input. Must depend on the submitted image.

    ``image_reference`` carries the face encoding's opaque reference so a run
    can be audited back to the exact image that triggered it. It is *not* sent
    to a provider — an embedding is meaningless to a web search engine.

    ``image_url`` is only for reverse-image-search providers, and only ever
    holds a URL the operator supplied. FaceTrace never publishes or uploads the
    local image file itself.
    """

    query_id: str
    terms: tuple[str, ...] = ()
    image_reference: Optional[str] = None
    provider: str = ""
    max_results: int = 10
    image_url: Optional[str] = None


@dataclass(frozen=True)
class SearchResult:
    """One untrusted candidate returned by a search provider.

    Every field is either copied from the provider's response or recorded by
    this process (``result_id``, ``retrieved_at``). Nothing here may be
    invented, and a result is *not* a claim that the content depicts the person
    whose face was submitted — only the matching stage may make that judgement.

    ``match_score`` therefore stays at its default through the whole search
    stage; it is the matching stage's field to populate.

    ``text`` is the provider's snippet/caption. It keeps that name because
    :data:`CANONICAL_FIELD_ORDER` hashes it. :attr:`snippet` is a read-only
    alias for readability.

    ``retrieved_at`` is deliberately absent from the canonical field order: it
    changes on every run, so hashing it would make verification impossible.
    """

    result_id: str
    source: str
    url: str
    title: str = ""
    text: str = ""
    image_url: Optional[str] = None
    match_score: float = 0.0
    metadata: Mapping[str, str] = field(default_factory=dict)
    #: Provider-reported publication timestamp, when it supplies one.
    published_at: str = ""
    #: UTC ISO-8601 instant this process received the result.
    retrieved_at: str = ""

    @property
    def snippet(self) -> str:
        """Alias for :attr:`text`, the provider-supplied snippet/caption."""
        return self.text


@dataclass(frozen=True)
class SearchOutcome:
    """Structured result of one search execution.

    Exactly three shapes are possible, and the status says which:

    * ``SUCCESS``   — ``results`` holds genuine candidates from the provider.
    * ``NO_RESULTS`` — the search ran and legitimately found nothing.
    * ``ERROR``     — ``error``/``error_code`` explain the structured failure.

    ``results`` is empty for both ``NO_RESULTS`` and ``ERROR``, so no caller can
    mistake a failure for a candidate.
    """

    status: SearchStatus
    provider: str = ""
    query: Optional[SearchQuery] = None
    results: tuple[SearchResult, ...] = ()
    error: str = ""
    error_code: Optional[ErrorCode] = None
    retryable: bool = False
    #: UTC ISO-8601 instant the search completed.
    retrieved_at: str = ""
    #: HTTP attempts made, including retries. 0 when nothing was sent.
    attempts: int = 0

    @property
    def succeeded(self) -> bool:
        return self.status is SearchStatus.SUCCESS

    @property
    def result_count(self) -> int:
        return len(self.results)


# --------------------------------------------------------------------------
# Matching stage
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchingPost:
    """A search result that cleared the match threshold. Retains its real URL."""

    result_id: str
    source: str
    url: str
    title: str = ""
    text: str = ""
    image_url: Optional[str] = None
    match_score: float = 0.0
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MatchDecision:
    """Outcome of match selection. ``matched=False`` is a valid, expected state."""

    matched: bool
    post: Optional[MatchingPost] = None
    threshold: float = 0.0
    best_score: float = 0.0
    candidates_considered: int = 0
    reason: str = ""



@dataclass(frozen=True)
class CandidateMatch:
    candidate_url: str
    source: str
    similarity_score: float
    face_detected: bool
    validation_status: str
    rejection_reason: Optional[str] = None
    metadata: Mapping[str, str] = field(default_factory=dict)

@dataclass(frozen=True)
class MatchResult:
    status: str  # MATCH_FOUND, NO_MATCH, NO_CANDIDATES, CANDIDATE_MEDIA_UNAVAILABLE, ERROR
    candidates_evaluated: tuple[CandidateMatch, ...] = ()
    best_candidate: Optional[MatchingPost] = None
    threshold_used: float = 0.0
    error: Optional[str] = None

# --------------------------------------------------------------------------
# Fingerprint stage
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalPayload:
    """Deterministic pre-hash representation of a matching post."""

    canonicalization_version: str
    field_order: tuple[str, ...]
    payload: str


@dataclass(frozen=True)
class Fingerprint:
    algorithm: str = "SHA-256"
    canonicalization_version: str = CANONICALIZATION_VERSION
    hash: str = ""


# --------------------------------------------------------------------------
# Blockchain stage
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockchainRecord:
    """An anchored fingerprint. Post content and biometrics stay off-chain."""

    network: str
    record_id: str
    record_hash: str
    transaction_hash: str = ""
    contract_address: str = ""
    block_number: int = 0


# --------------------------------------------------------------------------
# Verification stage
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VerificationResult:
    computed_hash: str
    on_chain_hash: str
    match: bool
    status: VerificationStatus


# --------------------------------------------------------------------------
# Orchestration envelope
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineError:
    code: ErrorCode
    stage: Stage
    message: str
    #: Only transient failures (network, RPC, provider timeout) may be retried.
    retryable: bool = False


@dataclass
class StageResult:
    stage: Stage
    status: StageStatus = StageStatus.PENDING
    duration_ms: int = 0
    data: dict[str, Any] = field(default_factory=dict)
    error: Optional[PipelineError] = None


@dataclass
class PipelineResult:
    """Everything a caller needs to render or audit one pipeline run."""

    input_id: str = ""
    image_path: str = ""
    stages: list[StageResult] = field(default_factory=list)
    status: VerificationStatus = VerificationStatus.INCONCLUSIVE
    face_input: Optional[FaceInput] = None
    encoding: Optional[FaceEncoding] = None
    match: Optional[MatchDecision] = None
    fingerprint: Optional[Fingerprint] = None
    record: Optional[BlockchainRecord] = None
    verification: Optional[VerificationResult] = None
    error: Optional[PipelineError] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.status is VerificationStatus.VERIFIED

    def stage_result(self, stage: Stage) -> Optional[StageResult]:
        """Return the recorded result for ``stage``, or ``None``."""
        for item in self.stages:
            if item.stage is stage:
                return item
        return None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable snapshot (all enums are ``str`` subclasses)."""
        return asdict(self)
