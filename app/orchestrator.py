"""Pipeline orchestration.

The orchestrator is the only module that knows the full stage order. It depends
solely on the service protocols, so any capability is replaceable:

    FaceInput -> FaceIdentifier -> SearchProvider -> MatchSelector
    -> FingerprintService -> BlockchainService -> VerificationService
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from .config import Settings, load_settings
from .models.pipeline import (
    PIPELINE_STAGES,
    ErrorCode,
    FaceDetectionResult,
    PipelineError,
    PipelineResult,
    SearchOutcome,
    SearchStatus,
    Stage,
    StageResult,
    StageStatus,
    VerificationStatus,
)
from .services import ServiceError, StageNotImplementedError
from .services.blockchain import BlockchainService, PendingBlockchainService
from .services.face import FaceIdentifier, PendingFaceIdentifier
from .services.fingerprint import FingerprintService, PendingFingerprintService
from .services.matching import MatchSelector, PendingMatchSelector
from .services.search import PendingSearchProvider, SearchProvider
from .services.verification import PendingVerificationService, VerificationService

__all__ = ["Orchestrator", "build_default_orchestrator", "build_real_orchestrator"]


@dataclass
class Orchestrator:
    """Coordinates one end-to-end FaceTrace run.

    Every collaborator is injected, so implementations and test doubles
    drop in without changing this class.
    """

    settings: Settings = field(default_factory=load_settings)
    face: FaceIdentifier = field(default_factory=PendingFaceIdentifier)
    search: SearchProvider = field(default_factory=PendingSearchProvider)
    selector: MatchSelector = field(default_factory=PendingMatchSelector)
    fingerprints: FingerprintService = field(default_factory=PendingFingerprintService)
    blockchain: BlockchainService = field(default_factory=PendingBlockchainService)
    verification: VerificationService = field(
        default_factory=PendingVerificationService
    )

    #: Stage order this orchestrator executes.
    stages: tuple[Stage, ...] = PIPELINE_STAGES

    def describe(self) -> list[StageResult]:
        """Return the stage plan as pending results — no execution."""
        return [StageResult(stage=stage, status=StageStatus.PENDING) for stage in self.stages]

    def service_map(self) -> dict[str, str]:
        """Name of the implementation bound to each replaceable capability."""
        return {
            "FaceIdentifier": _impl_name(self.face),
            "SearchProvider": _impl_name(self.search),
            "MatchSelector": _impl_name(self.selector),
            "FingerprintService": _impl_name(self.fingerprints),
            "BlockchainService": _impl_name(self.blockchain),
            "VerificationService": _impl_name(self.verification),
        }

    def run(self, image_path: str, *, input_id: Optional[str] = None) -> PipelineResult:
        """Run the end-to-end pipeline for ``image_path``."""
        req_id = input_id or f"req_{uuid.uuid4().hex[:12]}"
        stage_map = {
            stage: StageResult(stage=stage, status=StageStatus.PENDING)
            for stage in self.stages
        }

        # Shortcut if placeholder face identifier is used
        if isinstance(self.face, PendingFaceIdentifier):
            for s in self.stages:
                stage_map[s].status = StageStatus.NOT_IMPLEMENTED
            return PipelineResult(
                input_id=req_id,
                image_path=image_path,
                stages=list(stage_map.values()),
                status=VerificationStatus.INCONCLUSIVE,
                error=PipelineError(
                    code=ErrorCode.NOT_IMPLEMENTED,
                    stage=Stage.VALIDATE_INPUT,
                    message="FaceTrace STEP 1 placeholder face service in use.",
                    retryable=False,
                ),
            )

        return self._execute_pipeline(req_id, image_path, stage_map)

    def _mark_remaining_skipped(self, stage_map: dict[Stage, StageResult], failed_stage: Stage) -> None:
        if failed_stage in self.stages:
            idx = self.stages.index(failed_stage)
            for s in self.stages[idx + 1 :]:
                if stage_map[s].status == StageStatus.PENDING:
                    stage_map[s].status = StageStatus.SKIPPED
    def _execute_pipeline(
        self, req_id: str, image_path: str, stage_map: dict[Stage, StageResult]
    ) -> PipelineResult:
        # Stage 1: VALIDATE_INPUT
        t0 = time.perf_counter()
        stage_map[Stage.VALIDATE_INPUT].status = StageStatus.RUNNING

        try:
            if hasattr(self.face, "analyze_with_embedding"):
                det_res, raw_embedding = self.face.analyze_with_embedding(image_path)
            else:
                face_input = self.face.load(image_path)
                faces = self.face.detect(face_input)
                if not faces:
                    det_res = FaceDetectionResult(success=False, face_count=0, error_code=ErrorCode.NO_FACE_DETECTED)
                    raw_embedding = None
                elif len(faces) > 1:
                    det_res = FaceDetectionResult(success=False, face_count=len(faces), error_code=ErrorCode.MULTIPLE_FACES)
                    raw_embedding = None
                else:
                    encoding = self.face.encode(face_input, faces[0])
                    det_res = FaceDetectionResult(success=True, face_count=1, encoding=encoding)
                    raw_embedding = getattr(encoding, "embedding", None)
        except ServiceError as exc:
            code = ErrorCode(exc.code) if exc.code in ErrorCode.__members__ else ErrorCode.INVALID_IMAGE
            stage_map[Stage.VALIDATE_INPUT].status = StageStatus.FAILED
            err = PipelineError(code=code, stage=Stage.VALIDATE_INPUT, message=str(exc), retryable=exc.retryable)
            stage_map[Stage.VALIDATE_INPUT].error = err
            self._mark_remaining_skipped(stage_map, Stage.VALIDATE_INPUT)
            return PipelineResult(input_id=req_id, image_path=image_path, stages=list(stage_map.values()), status=VerificationStatus.INCONCLUSIVE, error=err)
        except Exception as exc:  # noqa: BLE001
            stage_map[Stage.VALIDATE_INPUT].status = StageStatus.FAILED
            err = PipelineError(code=ErrorCode.INVALID_IMAGE, stage=Stage.VALIDATE_INPUT, message=str(exc), retryable=False)
            stage_map[Stage.VALIDATE_INPUT].error = err
            self._mark_remaining_skipped(stage_map, Stage.VALIDATE_INPUT)
            return PipelineResult(input_id=req_id, image_path=image_path, stages=list(stage_map.values()), status=VerificationStatus.INCONCLUSIVE, error=err)

        if not det_res.success and det_res.error_code == ErrorCode.INVALID_IMAGE:
            stage_map[Stage.VALIDATE_INPUT].status = StageStatus.FAILED
            err = PipelineError(code=ErrorCode.INVALID_IMAGE, stage=Stage.VALIDATE_INPUT, message=det_res.error or "Invalid image", retryable=False)
            stage_map[Stage.VALIDATE_INPUT].error = err
            self._mark_remaining_skipped(stage_map, Stage.VALIDATE_INPUT)
            return PipelineResult(input_id=req_id, image_path=image_path, stages=list(stage_map.values()), status=VerificationStatus.INCONCLUSIVE, error=err)

        stage_map[Stage.VALIDATE_INPUT].status = StageStatus.SUCCESS
        stage_map[Stage.VALIDATE_INPUT].duration_ms = int((time.perf_counter() - t0) * 1000)

        return self._execute_detection(req_id, image_path, stage_map, det_res, raw_embedding)

    def _execute_detection(
        self, req_id: str, image_path: str, stage_map: dict[Stage, StageResult], det_res: FaceDetectionResult, raw_embedding: object
    ) -> PipelineResult:
        # Stage 2: DETECT_FACE
        t0 = time.perf_counter()
        stage_map[Stage.DETECT_FACE].status = StageStatus.RUNNING
        if not det_res.success and det_res.error_code in (ErrorCode.NO_FACE_DETECTED, ErrorCode.MULTIPLE_FACES):
            stage_map[Stage.DETECT_FACE].status = StageStatus.FAILED
            err = PipelineError(code=det_res.error_code, stage=Stage.DETECT_FACE, message=det_res.error or "Face detection failed", retryable=False)
            stage_map[Stage.DETECT_FACE].error = err
            self._mark_remaining_skipped(stage_map, Stage.DETECT_FACE)
            return PipelineResult(input_id=req_id, image_path=image_path, stages=list(stage_map.values()), status=VerificationStatus.INCONCLUSIVE, error=err)

        stage_map[Stage.DETECT_FACE].status = StageStatus.SUCCESS
        stage_map[Stage.DETECT_FACE].duration_ms = int((time.perf_counter() - t0) * 1000)
        stage_map[Stage.DETECT_FACE].data = {"face_count": det_res.face_count, "bbox": det_res.bbox}

        # Stage 3: ENCODE_FACE
        t0 = time.perf_counter()
        stage_map[Stage.ENCODE_FACE].status = StageStatus.RUNNING
        if not det_res.success or det_res.encoding is None:
            stage_map[Stage.ENCODE_FACE].status = StageStatus.FAILED
            err = PipelineError(code=det_res.error_code or ErrorCode.FACE_ENCODING_FAILED, stage=Stage.ENCODE_FACE, message=det_res.error or "Face encoding failed", retryable=False)
            stage_map[Stage.ENCODE_FACE].error = err
            self._mark_remaining_skipped(stage_map, Stage.ENCODE_FACE)
            return PipelineResult(input_id=req_id, image_path=image_path, stages=list(stage_map.values()), status=VerificationStatus.INCONCLUSIVE, error=err)

        encoding = det_res.encoding
        stage_map[Stage.ENCODE_FACE].status = StageStatus.SUCCESS
        stage_map[Stage.ENCODE_FACE].duration_ms = int((time.perf_counter() - t0) * 1000)
        stage_map[Stage.ENCODE_FACE].data = {"encoding_reference": encoding.encoding_reference, "model": encoding.model}

        return self._execute_search_and_matching(req_id, image_path, stage_map, encoding, raw_embedding)

    def _execute_search_and_matching(
        self, req_id: str, image_path: str, stage_map: dict[Stage, StageResult], encoding: object, raw_embedding: object
    ) -> PipelineResult:
        # Stage 4: SEARCH
        t0 = time.perf_counter()
        stage_map[Stage.SEARCH].status = StageStatus.RUNNING
        try:
            if hasattr(self.search, "search_for_encoding"):
                search_outcome = self.search.search_for_encoding(encoding)
            else:
                query = self.search.build_query(encoding)
                results = self.search.search(query)
                search_outcome = SearchOutcome(status=SearchStatus.SUCCESS if results else SearchStatus.NO_RESULTS, results=tuple(results))
        except ServiceError as exc:
            code = ErrorCode(exc.code) if exc.code in ErrorCode.__members__ else ErrorCode.SEARCH_FAILED
            stage_map[Stage.SEARCH].status = StageStatus.FAILED
            err = PipelineError(code=code, stage=Stage.SEARCH, message=str(exc), retryable=exc.retryable)
            stage_map[Stage.SEARCH].error = err
            self._mark_remaining_skipped(stage_map, Stage.SEARCH)
            return PipelineResult(input_id=req_id, image_path=image_path, stages=list(stage_map.values()), status=VerificationStatus.INCONCLUSIVE, encoding=encoding, error=err)

        dt = int((time.perf_counter() - t0) * 1000)
        if search_outcome.status == SearchStatus.ERROR or search_outcome.error_code:
            stage_map[Stage.SEARCH].status = StageStatus.FAILED
            err = PipelineError(code=search_outcome.error_code or ErrorCode.SEARCH_FAILED, stage=Stage.SEARCH, message=search_outcome.error or "Search error", retryable=False)
            stage_map[Stage.SEARCH].error = err
            self._mark_remaining_skipped(stage_map, Stage.SEARCH)
            return PipelineResult(input_id=req_id, image_path=image_path, stages=list(stage_map.values()), status=VerificationStatus.INCONCLUSIVE, encoding=encoding, error=err)

        stage_map[Stage.SEARCH].status = StageStatus.SUCCESS
        stage_map[Stage.SEARCH].duration_ms = dt
        stage_map[Stage.SEARCH].data = {"results_count": len(search_outcome.results)}

        # Stage 5: SELECT_MATCH
        t0 = time.perf_counter()
        stage_map[Stage.SELECT_MATCH].status = StageStatus.RUNNING
        if hasattr(self.selector, "set_query_embedding") and raw_embedding is not None:
            self.selector.set_query_embedding(encoding, raw_embedding)

        match_decision = self.selector.select(encoding, search_outcome.results)
        stage_map[Stage.SELECT_MATCH].status = StageStatus.SUCCESS
        stage_map[Stage.SELECT_MATCH].duration_ms = int((time.perf_counter() - t0) * 1000)
        stage_map[Stage.SELECT_MATCH].data = {
            "matched": match_decision.matched,
            "threshold": match_decision.threshold,
            "best_score": match_decision.best_score,
            "reason": match_decision.reason,
        }

        if not match_decision.matched or match_decision.post is None:
            self._mark_remaining_skipped(stage_map, Stage.SELECT_MATCH)
            terminal_status = (
                VerificationStatus.CANDIDATE_MEDIA_UNAVAILABLE
                if match_decision.reason == "NO_VALID_CANDIDATES"
                else VerificationStatus.NO_MATCH
            )
            return PipelineResult(
                input_id=req_id,
                image_path=image_path,
                stages=list(stage_map.values()),
                status=terminal_status,
                encoding=encoding,
                match=match_decision,
            )

        return self._execute_verification_flow(req_id, image_path, stage_map, encoding, match_decision)

    def _execute_verification_flow(
        self, req_id: str, image_path: str, stage_map: dict[Stage, StageResult], encoding: object, match_decision: object
    ) -> PipelineResult:
        post = match_decision.post

        # Stage 6: FINGERPRINT
        t0 = time.perf_counter()
        stage_map[Stage.FINGERPRINT].status = StageStatus.RUNNING
        try:
            fingerprint = self.fingerprints.fingerprint(post)
            stage_map[Stage.FINGERPRINT].status = StageStatus.SUCCESS
            stage_map[Stage.FINGERPRINT].duration_ms = int((time.perf_counter() - t0) * 1000)
            stage_map[Stage.FINGERPRINT].data = {"hash": fingerprint.hash, "algorithm": fingerprint.algorithm}
        except Exception as exc:  # noqa: BLE001
            stage_map[Stage.FINGERPRINT].status = StageStatus.FAILED
            err = PipelineError(code=ErrorCode.FINGERPRINT_FAILED, stage=Stage.FINGERPRINT, message=str(exc), retryable=False)
            stage_map[Stage.FINGERPRINT].error = err
            self._mark_remaining_skipped(stage_map, Stage.FINGERPRINT)
            return PipelineResult(input_id=req_id, image_path=image_path, stages=list(stage_map.values()), status=VerificationStatus.INCONCLUSIVE, encoding=encoding, match=match_decision, error=err)

        # Stage 7: BLOCKCHAIN_WRITE
        t0 = time.perf_counter()
        stage_map[Stage.BLOCKCHAIN_WRITE].status = StageStatus.RUNNING
        record_id = post.result_id or f"rec_{req_id}"
        try:
            record = self.blockchain.write_fingerprint(record_id, fingerprint)
            stage_map[Stage.BLOCKCHAIN_WRITE].status = StageStatus.SUCCESS
            stage_map[Stage.BLOCKCHAIN_WRITE].duration_ms = int((time.perf_counter() - t0) * 1000)
            stage_map[Stage.BLOCKCHAIN_WRITE].data = {"record_id": record.record_id, "tx_hash": record.transaction_hash}
        except Exception as exc:  # noqa: BLE001
            stage_map[Stage.BLOCKCHAIN_WRITE].status = StageStatus.FAILED
            err = PipelineError(code=ErrorCode.BLOCKCHAIN_WRITE_FAILED, stage=Stage.BLOCKCHAIN_WRITE, message=str(exc), retryable=True)
            stage_map[Stage.BLOCKCHAIN_WRITE].error = err
            self._mark_remaining_skipped(stage_map, Stage.BLOCKCHAIN_WRITE)
            return PipelineResult(input_id=req_id, image_path=image_path, stages=list(stage_map.values()), status=VerificationStatus.BLOCKCHAIN_UNAVAILABLE, encoding=encoding, match=match_decision, fingerprint=fingerprint, error=err)

        # Stage 8: BLOCKCHAIN_READ
        t0 = time.perf_counter()
        stage_map[Stage.BLOCKCHAIN_READ].status = StageStatus.RUNNING
        try:
            stored_hash = self.blockchain.read_fingerprint(record.record_id)
            stage_map[Stage.BLOCKCHAIN_READ].status = StageStatus.SUCCESS
            stage_map[Stage.BLOCKCHAIN_READ].duration_ms = int((time.perf_counter() - t0) * 1000)
            stage_map[Stage.BLOCKCHAIN_READ].data = {"stored_hash": stored_hash}
        except Exception as exc:  # noqa: BLE001
            stage_map[Stage.BLOCKCHAIN_READ].status = StageStatus.FAILED
            err = PipelineError(code=ErrorCode.BLOCKCHAIN_READ_FAILED, stage=Stage.BLOCKCHAIN_READ, message=str(exc), retryable=True)
            stage_map[Stage.BLOCKCHAIN_READ].error = err
            self._mark_remaining_skipped(stage_map, Stage.BLOCKCHAIN_READ)
            return PipelineResult(input_id=req_id, image_path=image_path, stages=list(stage_map.values()), status=VerificationStatus.BLOCKCHAIN_UNAVAILABLE, encoding=encoding, match=match_decision, fingerprint=fingerprint, record=record, error=err)

        # Stage 9: RECOMPUTE_FINGERPRINT
        t0 = time.perf_counter()
        stage_map[Stage.RECOMPUTE_FINGERPRINT].status = StageStatus.RUNNING
        try:
            if hasattr(self.verification, "recompute"):
                recomputed_fp = self.verification.recompute(post)
            else:
                recomputed_fp = self.fingerprints.fingerprint(post)
            stage_map[Stage.RECOMPUTE_FINGERPRINT].status = StageStatus.SUCCESS
            stage_map[Stage.RECOMPUTE_FINGERPRINT].duration_ms = int((time.perf_counter() - t0) * 1000)
            stage_map[Stage.RECOMPUTE_FINGERPRINT].data = {"recomputed_hash": recomputed_fp.hash}
        except Exception as exc:  # noqa: BLE001
            stage_map[Stage.RECOMPUTE_FINGERPRINT].status = StageStatus.FAILED
            err = PipelineError(code=ErrorCode.FINGERPRINT_FAILED, stage=Stage.RECOMPUTE_FINGERPRINT, message=str(exc), retryable=False)
            stage_map[Stage.RECOMPUTE_FINGERPRINT].error = err
            self._mark_remaining_skipped(stage_map, Stage.RECOMPUTE_FINGERPRINT)
            return PipelineResult(input_id=req_id, image_path=image_path, stages=list(stage_map.values()), status=VerificationStatus.INCONCLUSIVE, encoding=encoding, match=match_decision, fingerprint=fingerprint, record=record, error=err)

        # Stage 10: VERIFY
        t0 = time.perf_counter()
        stage_map[Stage.VERIFY].status = StageStatus.RUNNING
        verification_result = self.verification.verify(post, record)
        stage_map[Stage.VERIFY].status = StageStatus.SUCCESS
        stage_map[Stage.VERIFY].duration_ms = int((time.perf_counter() - t0) * 1000)
        stage_map[Stage.VERIFY].data = {
            "verified": verification_result.match,
            "status": verification_result.status,
            "computed_hash": verification_result.computed_hash,
            "on_chain_hash": verification_result.on_chain_hash,
        }

        return PipelineResult(
            input_id=req_id,
            image_path=image_path,
            stages=list(stage_map.values()),
            status=verification_result.status,
            encoding=encoding,
            match=match_decision,
            fingerprint=fingerprint,
            record=record,
            verification=verification_result,
        )


def _impl_name(service: object) -> str:
    return str(getattr(service, "name", type(service).__name__))


def build_default_orchestrator(settings: Optional[Settings] = None) -> Orchestrator:
    """Build an orchestrator wired with the STEP 1 placeholder services."""
    return Orchestrator(settings=settings or load_settings())


def build_real_orchestrator(settings: Optional[Settings] = None) -> Orchestrator:
    """Build an orchestrator wired with production services."""
    s = settings or load_settings()

    from .services.blockchain_sim import InMemoryBlockchainService
    from .services.face_insightface import build_face_identifier
    from .services.fingerprint import Sha256FingerprintService
    from .services.matching import DefaultMatchSelector, InsightFaceMatcher
    from .services.search_web import build_search_service
    from .services.verification_impl import HashVerificationService

    face_id = build_face_identifier(s)
    search_svc = build_search_service(s)
    matcher = InsightFaceMatcher(face_id)
    selector = DefaultMatchSelector(settings=s, matcher=matcher)
    fingerprints = Sha256FingerprintService()

    if s.blockchain_rpc_url and s.blockchain_private_key and s.contract_address:
        try:
            from .services.blockchain_web3 import Web3BlockchainService

            blockchain: BlockchainService = Web3BlockchainService(s)
        except Exception:
            blockchain = InMemoryBlockchainService()
    else:
        blockchain = InMemoryBlockchainService()

    verification = HashVerificationService(fingerprints, blockchain)

    return Orchestrator(
        settings=s,
        face=face_id,
        search=search_svc,
        selector=selector,
        fingerprints=fingerprints,
        blockchain=blockchain,
        verification=verification,
    )
