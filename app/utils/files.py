"""Safe temporary upload handling."""

from __future__ import annotations

import secrets
from pathlib import Path

from fastapi import UploadFile

from ..core.security import ALLOWED_EXTENSIONS, ALLOWED_IMAGE_TYPES


class FileValidationError(Exception):
    def __init__(self, code: str, message: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def _signature_matches(content_type: str, prefix: bytes) -> bool:
    signatures = ALLOWED_IMAGE_TYPES.get(content_type)
    if not signatures:
        return False
    if content_type == "image/webp":
        return len(prefix) >= 12 and prefix[:4] == b"RIFF" and prefix[8:12] == b"WEBP"
    return any(prefix.startswith(signature) for signature in signatures)


async def store_validated_upload(
    upload: UploadFile, destination: Path, max_size: int
) -> Path:
    """Validate and write an upload using a random server-side filename."""
    suffix = Path(upload.filename or "").suffix.lower()
    if upload.content_type not in ALLOWED_IMAGE_TYPES or suffix not in ALLOWED_EXTENSIONS:
        raise FileValidationError(
            "UNSUPPORTED_MEDIA_TYPE", "Only JPEG, PNG, and WEBP images are supported.", 415
        )

    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{secrets.token_urlsafe(24)}{suffix}"
    total = 0
    first_chunk = b""
    try:
        with target.open("xb") as output:
            while chunk := await upload.read(1024 * 1024):
                if not first_chunk:
                    first_chunk = chunk[:32]
                total += len(chunk)
                if total > max_size:
                    raise FileValidationError(
                        "FILE_TOO_LARGE", "The uploaded image exceeds the size limit.", 413
                    )
                output.write(chunk)
        if not _signature_matches(upload.content_type, first_chunk):
            raise FileValidationError(
                "INVALID_IMAGE", "The uploaded file is not a valid supported image.", 400
            )
        return target
    except Exception:
        target.unlink(missing_ok=True)
        raise
