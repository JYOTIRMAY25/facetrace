from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
from app.orchestrator import build_real_orchestrator
from app.models.pipeline import PipelineResult, MatchingPost, BlockchainRecord, VerificationStatus
from app.services.fingerprint import Sha256FingerprintService
from app.services.blockchain_sim import InMemoryBlockchainService
from app.services.verification_impl import HashVerificationService
import os
import tempfile
from pathlib import Path

app = FastAPI()
orchestrator = build_real_orchestrator()

# Default Verification Engine dependencies
_fp_service = Sha256FingerprintService()
_bc_service = InMemoryBlockchainService()
verification_engine = HashVerificationService(_fp_service, _bc_service)

class VerifyRequest(BaseModel):
    post: dict
    record: dict

@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "FaceTrace Backend Pipeline"}

@app.post("/investigate")
@app.post("/api/investigate")
async def investigate(
    request: Request,
    file: UploadFile | None = File(default=None),
):
    """Run an investigation without retaining the uploaded image."""
    upload = file
    if upload is None:
        form = await request.form()
        candidate = form.get("image")
        if hasattr(candidate, "read") and hasattr(candidate, "filename"):
            upload = candidate
    if upload is None:
        raise HTTPException(status_code=400, detail="An image file is required.")

    suffix = Path(upload.filename or "").suffix[:10]
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=suffix, prefix="facetrace-", delete=False
        ) as buffer:
            temporary_path = buffer.name
            while chunk := await upload.read(1024 * 1024):
                buffer.write(chunk)

        result = orchestrator.run(temporary_path)
        payload = result.to_dict()
        # The local temporary path is an internal implementation detail.
        payload.pop("image_path", None)
        return payload
    finally:
        await upload.close()
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass

@app.post("/api/verify")
async def verify(request: VerifyRequest):
    """Expose independent fingerprint recomputation and on-chain verification."""
    try:
        post_data = request.post
        matching_post = MatchingPost(
            result_id=post_data.get("result_id", ""),
            source=post_data.get("source", ""),
            url=post_data.get("url", ""),
            title=post_data.get("title", ""),
            text=post_data.get("text", ""),
            image_url=post_data.get("image_url")
        )

        record_data = request.record
        blockchain_record = BlockchainRecord(
            network=record_data.get("network", "local"),
            record_id=record_data.get("record_id", ""),
            record_hash=record_data.get("record_hash", ""),
            transaction_hash=record_data.get("transaction_hash", ""),
            contract_address=record_data.get("contract_address", ""),
            block_number=record_data.get("block_number", 0)
        )

        result = verification_engine.verify(matching_post, blockchain_record)
        return {
            "computed_hash": result.computed_hash,
            "on_chain_hash": result.on_chain_hash,
            "match": result.match,
            "status": result.status.value if isinstance(result.status, VerificationStatus) else str(result.status)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
