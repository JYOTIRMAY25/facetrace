import logging
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from ...core.config import Settings
from ...models.pipeline import ErrorCode, SearchStatus, Fingerprint
from ...models.response import (
    BoundingBoxResponse,
    CandidateResponse,
    DetectedFaceResponse,
    InvestigationResponse,
)
from ...config import load_settings
from ...services import ServiceError
from ...services.face_insightface import build_face_identifier
from ...services.search_web import build_search_service
from ...services.face_matching import match_candidates
from ...services.candidate_image_retrieval import CandidateImageRetriever
from ...services.evidence_extraction import extract_match_evidence
from ...services.canonical_evidence import canonicalize_evidence
from ...services.evidence_fingerprint import fingerprint_evidence
from ...services.blockchain import BlockchainError, Web3BlockchainService
from ...utils.files import store_validated_upload

logger = logging.getLogger("facetrace.investigation")
router = APIRouter()
face_identifier = build_face_identifier()
search_service = build_search_service(load_settings())


@router.post("/investigate", response_model=InvestigationResponse)
async def investigate(
    request: Request,
    file: UploadFile | None = File(default=None),
) -> InvestigationResponse:
    settings: Settings = request.app.state.settings
    if file is None:
        raise HTTPException(status_code=400, detail={"code": "INVALID_IMAGE", "message": "An image file is required."})
    logger.info("investigation started")
    investigation_id = str(uuid.uuid4())
    path: Path | None = None
    try:
        path = await store_validated_upload(file, settings.data_dir / "tmp", settings.max_upload_size_bytes)
        started = time.perf_counter()
        result, source_embedding = face_identifier.analyze_with_embedding(path)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        if (
            result.error_code is not None
            and result.error_code not in (ErrorCode.NO_FACE_DETECTED, ErrorCode.MULTIPLE_FACES)
        ):
            raise HTTPException(
                status_code=422,
                detail={"code": result.error_code.value, "message": result.error or "Face analysis failed."},
            )
        if result.error_code is ErrorCode.NO_FACE_DETECTED:
            raise HTTPException(
                status_code=422,
                detail={"code": "NO_FACE_DETECTED", "message": "No usable face was detected in the provided image."},
            )
        faces = [
            DetectedFaceResponse(
                face_index=index,
                bbox=BoundingBoxResponse(
                    x=face.bounding_box.x,
                    y=face.bounding_box.y,
                    width=face.bounding_box.width,
                    height=face.bounding_box.height,
                ),
                confidence=face.detector_confidence,
            )
            for index, face in enumerate(result.faces)
        ]
        if not result.success or result.encoding is None:
            logger.info(
                "multiple faces detected faces=%d; search not started",
                result.face_count,
            )
            return InvestigationResponse(
                status="multiple_faces_detected",
                message="Multiple faces detected; search was not started.",
                faces_detected=result.face_count,
                faces=faces,
                encoding_generated=False,
                detection_time_ms=result.detection_duration_ms,
                encoding_time_ms=result.encoding_duration_ms,
                total_time_ms=result.total_duration_ms,
            )
        missing = search_service.provider.missing_configuration()
        if missing:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "SEARCH_PROVIDER_NOT_CONFIGURED",
                    "message": "The configured search provider is not ready.",
                },
            )
        search_started = time.perf_counter()
        if hasattr(search_service.provider, "search_uploaded_image"):
            outcome = search_service.search_uploaded_image(path, result.encoding)
        else:
            outcome = search_service.search_for_encoding(result.encoding)
        search_time_ms = int((time.perf_counter() - search_started) * 1000)
        if outcome.status is SearchStatus.NO_RESULTS:
            candidates = []
        elif outcome.error_code is not None:
            code = outcome.error_code.value
            if code == "SEARCH_FAILED" and "timed out" in outcome.error.lower():
                code = "SEARCH_TIMEOUT"
            raise HTTPException(
                status_code=503 if code in {
                    "SEARCH_PROVIDER_NOT_CONFIGURED",
                    "SEARCH_NOT_CONFIGURED",
                    "SEARCH_TIMEOUT",
                    "SEARCH_RATE_LIMITED",
                    "SEARCH_FAILED",
                } else 502,
                detail={"code": code, "message": outcome.error or "Search failed."},
            )
        candidates = [
            CandidateResponse(
                title=item.title or None,
                url=item.url,
                source_domain=item.source,
                image_url=item.image_url,
                thumbnail_url=item.metadata.get("thumbnail_url"),
                provider=outcome.provider,
                match_status="NOT_EVALUATED",
            )
            for item in outcome.results
        ]
        if source_embedding is None:
            raise HTTPException(
                status_code=422,
                detail={"code": "FACE_ENCODING_FAILED", "message": "Source face encoding was unavailable."},
            )
        ranked = match_candidates(
            result.encoding,
            source_embedding,
            outcome.results,
            face_identifier,
            threshold=load_settings().face_match_threshold,
            retriever=CandidateImageRetriever(
                timeout=float(load_settings().request_timeout_s),
                max_bytes=load_settings().candidate_image_max_bytes,
            ),
        )
        matches = [
            CandidateResponse(
                title=item.candidate.title or None,
                url=item.candidate.url,
                source_domain=item.candidate.source,
                image_url=item.candidate.image_url,
                thumbnail_url=item.candidate.metadata.get("thumbnail_url"),
                provider=outcome.provider,
                candidate_face_index=item.candidate_face_index,
                distance=item.distance,
                similarity_score=item.similarity_score,
                match_status=item.match_status,
            )
            for item in ranked
        ]
        matches_found = [item for item in matches if item.match_status == "MATCH_FOUND"]
        ranked_matches_found = [item for item in ranked if item.match_status == "MATCH_FOUND"]
        if matches_found:
            status = "MATCH_FOUND"
        elif any(item.match_status == "NO_MATCH" for item in matches):
            status = "NO_MATCH"
        elif candidates:
            status = "NO_VALID_CANDIDATE_MATCH"
        else:
            status = "no_candidates"
        evidence = None
        canonicalization = None
        fingerprint = None
        blockchain = None
        if ranked_matches_found:
            evidence_record = extract_match_evidence(
                investigation_id=investigation_id,
                source={
                    "filename": file.filename or "",
                    "mime_type": file.content_type or "",
                    "file_size": path.stat().st_size,
                    "sha256": None,
                },
                provider=outcome.provider,
                searched_at=outcome.retrieved_at,
                candidate_count=len(candidates),
                match=ranked_matches_found[0],
            )
            if evidence_record is not None:
                canonical = canonicalize_evidence(evidence_record)
                evidence_fingerprint = fingerprint_evidence(canonical.canonical_bytes)
                evidence = {
                    "evidence_version": evidence_record.evidence_version,
                    "search_provider": evidence_record.search["provider"],
                    "candidate_count": evidence_record.search["candidate_count"],
                    "social_media": evidence_record.social_media,
                }
                canonicalization = {"status": "READY"}
                fingerprint = {
                    "algorithm": evidence_fingerprint.algorithm,
                    "value": evidence_fingerprint.fingerprint,
                    "input_encoding": evidence_fingerprint.input_encoding,
                    "status": evidence_fingerprint.status,
                }
                blockchain_settings = load_settings()
                blockchain_service = Web3BlockchainService(blockchain_settings)
                try:
                    status = "BLOCKCHAIN_PENDING"
                    record = blockchain_service.write_fingerprint(
                        investigation_id,
                        Fingerprint(
                            algorithm=evidence_fingerprint.algorithm,
                            hash=evidence_fingerprint.fingerprint,
                        ),
                    )
                    status = "BLOCKCHAIN_CONFIRMED"
                    blockchain_service.read_fingerprint(
                        Fingerprint(
                            algorithm=evidence_fingerprint.algorithm,
                            hash=evidence_fingerprint.fingerprint,
                        )
                    )
                    explorer_url = None
                    if blockchain_settings.blockchain_explorer_url:
                        explorer_url = blockchain_settings.blockchain_explorer_url.rstrip("/") + "/" + record.transaction_hash
                    status = "BLOCKCHAIN_VERIFIED"
                    blockchain = {
                        "status": "VERIFIED",
                        "network": record.network,
                        "chain_id": blockchain_settings.blockchain_chain_id or None,
                        "transaction_hash": record.transaction_hash,
                        "block_number": record.block_number,
                        "contract_address": record.contract_address,
                        "evidence_hash": record.record_hash,
                        "explorer_url": explorer_url,
                    }
                except BlockchainError as exc:
                    status = "BLOCKCHAIN_FAILED"
                    blockchain = {"status": "BLOCKCHAIN_FAILED", "error_code": exc.code}
            else:
                status = "EVIDENCE_INCOMPLETE"
                fingerprint = {"status": "NOT_GENERATED"}
        elif status != "MATCH_FOUND":
            fingerprint = {"status": "NOT_GENERATED"}
        logger.info(
            "face analysis complete faces=%d encoding_generated=%s duration_ms=%d",
            result.face_count,
            result.embedding_available or result.face_count > 0,
            elapsed_ms,
        )
        return InvestigationResponse(
            investigation_id=investigation_id,
            status=status,
            message=(
                "Candidate retrieval and face comparison completed."
                if candidates
                else "Real search completed with no candidate sources."
            ),
            faces_detected=result.face_count,
            faces=faces,
            encoding_generated=result.face_count > 0,
            embedding_dimensions=(
                [result.embedding_dimension] if result.embedding_dimension else []
            ),
            detection_time_ms=result.detection_duration_ms,
            encoding_time_ms=result.encoding_duration_ms,
            total_time_ms=result.total_duration_ms,
            provider=outcome.provider,
            candidates=candidates,
            matches=matches,
            search_time_ms=search_time_ms,
            evidence=evidence,
            canonicalization=canonicalization,
            fingerprint=fingerprint,
            blockchain=blockchain,
        )
    except ServiceError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    finally:
        await file.close()
        if path is not None:
            path.unlink(missing_ok=True)
