"""Safe retrieval and validation of public candidate images."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit

import cv2
import httpx
import numpy as np

from . import ServiceError


@dataclass(frozen=True)
class CandidateImage:
    url: str
    content_type: str
    size_bytes: int
    image_bytes: bytes


class CandidateImageRetrievalError(ServiceError):
    def __init__(
        self,
        message: str,
        *,
        invalid_image: bool = False,
        retryable: bool = False,
    ) -> None:
        super().__init__(
            message,
            code="IMAGE_INVALID" if invalid_image else "IMAGE_RETRIEVAL_FAILED",
            retryable=retryable,
        )


class CandidateImageRetriever:
    def __init__(self, *, timeout: float = 15.0, max_bytes: int = 10 * 1024 * 1024) -> None:
        self.timeout = timeout
        self.max_bytes = max_bytes

    def retrieve(self, image_url: str) -> CandidateImage:
        parts = urlsplit(image_url)
        if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
            raise CandidateImageRetrievalError("Candidate image URL must use HTTP or HTTPS.")
        try:
            with httpx.stream(
                "GET", image_url, timeout=self.timeout, follow_redirects=True
            ) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                if content_type not in {"image/jpeg", "image/png", "image/webp"}:
                    raise CandidateImageRetrievalError("Candidate response is not a supported image.")
                declared = response.headers.get("content-length")
                if declared and int(declared) > self.max_bytes:
                    raise CandidateImageRetrievalError("Candidate image exceeds the size limit.")
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > self.max_bytes:
                        raise CandidateImageRetrievalError("Candidate image exceeds the size limit.")
                    chunks.append(chunk)
        except CandidateImageRetrievalError:
            raise
        except httpx.TimeoutException as exc:
            raise CandidateImageRetrievalError("Candidate image retrieval timed out.", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise CandidateImageRetrievalError("Candidate image retrieval failed.", retryable=True) from exc
        payload = b"".join(chunks)
        if cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR) is None:
            raise CandidateImageRetrievalError(
                "Candidate image bytes are invalid.", invalid_image=True
            )
        return CandidateImage(
            url=image_url,
            content_type=content_type,
            size_bytes=len(payload),
            image_bytes=payload,
        )
