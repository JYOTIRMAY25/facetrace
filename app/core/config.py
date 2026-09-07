"""Environment-backed settings for the HTTP backend."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from ..config import PROJECT_ROOT


class Settings(BaseModel):
    """Settings required by the backend foundation.

    Future service credentials remain optional until their pipeline stages exist.
    """

    app_name: str = "FaceTrace Backend Pipeline"
    environment: str = "development"
    data_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data")
    max_upload_size_bytes: int = 10 * 1024 * 1024
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )
    search_api_key: str = ""
    search_engine_id: str = ""
    blockchain_rpc_url: str = ""
    blockchain_private_key: str = ""
    blockchain_contract_address: str = ""


def get_settings() -> Settings:
    """Load settings without requiring any future-stage credentials."""
    raw_origins = os.getenv("CORS_ORIGINS", "")
    origins = [item.strip() for item in raw_origins.split(",") if item.strip()]
    return Settings(
        environment=os.getenv("ENVIRONMENT", "development"),
        data_dir=Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data"))),
        max_upload_size_bytes=int(
            os.getenv("MAX_UPLOAD_SIZE_BYTES", str(10 * 1024 * 1024))
        ),
        cors_origins=origins or [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        search_api_key=os.getenv("SEARCH_API_KEY", ""),
        search_engine_id=os.getenv("SEARCH_ENGINE_ID", ""),
        blockchain_rpc_url=os.getenv("BLOCKCHAIN_RPC_URL", ""),
        blockchain_private_key=os.getenv("BLOCKCHAIN_PRIVATE_KEY", ""),
        blockchain_contract_address=os.getenv("BLOCKCHAIN_CONTRACT_ADDRESS", ""),
    )
