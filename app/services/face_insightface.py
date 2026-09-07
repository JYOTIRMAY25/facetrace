"""InsightFace-backed implementation of the :class:`FaceIdentifier` protocol.

Pipeline slice implemented here (STEP 2):

    image path or bytes -> validate/decode (OpenCV) -> detect (InsightFace
    det_10g) -> exactly-one-face gate -> embed (InsightFace w600k_r50)

Why this lives in its own module: :mod:`app.services.face` holds the protocol
and must stay importable without OpenCV / NumPy / InsightFace installed. This
adapter carries those dependencies, so importing it is opt-in.

Privacy and safety properties, enforced below rather than merely documented:

* **Local only.** Detection and embedding run through ONNX Runtime on the CPU.
  The image is never sent to any external service. The only network access
  InsightFace performs is a one-time *download of model weights* on first use;
  set ``FACE_MODEL_ROOT`` to a pre-provisioned directory to run fully offline.
* **No gender/age inference.** Only the ``detection`` and ``recognition``
  modules are loaded; the pack's ``genderage`` and landmark models are skipped.
* **No permanent storage.** Embeddings exist only in memory. A single-entry
  cache holds the most recent image's analysis so the protocol's
  ``detect``/``encode`` pair does not re-run inference; :meth:`clear_cache`
  drops it. Nothing is written to disk.
* **Never logged.** :class:`EmbeddingVector` redacts itself under ``repr``,
  ``str``, and f-string formatting, and no log statement takes the vector.
  Logs carry only counts, a rounded score, a bounding box, the dimension, and a
  truncated digest.
* **Not identity.** An embedding supports a *content* similarity comparison. It
  does not establish, prove, or assert anyone's legal identity.

Only process images you are authorised to process.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import cv2
import numpy as np

from ..config import Settings, load_settings
from ..models.pipeline import (
    BoundingBox,
    DetectedFace,
    ErrorCode,
    FaceDetectionResult,
    FaceEncoding,
    FaceInput,
)
from . import ServiceError

__all__ = [
    "EmbeddingVector",
    "InsightFaceIdentifier",
    "ImageSource",
    "build_face_identifier",
]

logger = logging.getLogger(__name__)

#: Accepted input kinds — a filesystem path or raw encoded image bytes.
ImageSource = Union[str, "os.PathLike[str]", bytes, bytearray, memoryview]

#: Upper bound on decoded pixels, so a malicious/huge file cannot exhaust RAM.
MAX_PIXELS = 64_000_000


class EmbeddingVector:
    """An immutable face embedding that refuses to print its own values.

    The vector is the most sensitive artefact in the pipeline, so the obvious
    accidents are closed off: ``repr``, ``str``, and ``format`` all redact, and
    the backing array is read-only. Callers that genuinely need the numbers ask
    explicitly via :attr:`values` or :meth:`tolist`.
    """

    __slots__ = ("_values", "_digest")

    def __init__(self, values: Any) -> None:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 1:
            raise ValueError(f"embedding must be 1-D, got shape {array.shape}")
        if array.size == 0:
            raise ValueError("embedding must not be empty")
        if not np.all(np.isfinite(array)):
            raise ValueError("embedding contains non-finite values")
        array = array.copy()
        array.setflags(write=False)
        self._values = array
        self._digest = hashlib.sha256(array.tobytes()).hexdigest()

    @property
    def values(self) -> np.ndarray:
        """The read-only float32 vector. Explicit opt-in to the raw numbers."""
        return self._values

    @property
    def dimension(self) -> int:
        return int(self._values.size)

    @property
    def digest(self) -> str:
        """SHA-256 of the vector bytes — a stable, non-invertible reference."""
        return self._digest

    def tolist(self) -> list[float]:
        """Explicit opt-in conversion. Do not pass the result to a logger."""
        return [float(value) for value in self._values]

    def __len__(self) -> int:
        return self.dimension

    def __repr__(self) -> str:
        return (
            f"<EmbeddingVector dim={self.dimension} "
            f"sha256={self._digest[:12]}... values=REDACTED>"
        )

    __str__ = __repr__

    def __format__(self, format_spec: str) -> str:
        # Keeps f"{embedding}" and f"{embedding:.3f}" from leaking values.
        return repr(self)


@dataclass(frozen=True)
class _AnalysedFace:
    """One detected face plus its embedding, held in memory only."""

    bbox: BoundingBox
    score: float
    embedding: EmbeddingVector


@dataclass(frozen=True)
class _Analysis:
    """Decoded-image metadata and every face found in it."""

    image_sha256: str
    width: int
    height: int
    faces: tuple[_AnalysedFace, ...]
    detection_duration_ms: int
    encoding_duration_ms: int
    total_duration_ms: int


def _invalid_image(message: str) -> ServiceError:
    return ServiceError(message, code=ErrorCode.INVALID_IMAGE.value, retryable=False)


class InsightFaceIdentifier:
    """Detects and encodes exactly one face, using local InsightFace models.

    Satisfies :class:`app.services.face.FaceIdentifier`. Model loading is lazy,
    so constructing this object is cheap and never touches the network.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        analyzer: Any = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.model_pack = self.settings.face_model_pack
        self.det_size = int(self.settings.face_det_size)
        self.min_det_score = float(self.settings.face_min_det_score)
        self.name = f"insightface-{self.model_pack}"
        self.detector_model = self.settings.face_detector_model
        self.encoder_model = self.settings.face_encoder_model
        #: Injected analyzer (tests / alternative packs). Must expose ``get``.
        self._analyzer = analyzer
        self._cache: Optional[_Analysis] = None

    # -- model -----------------------------------------------------------

    def _ensure_analyzer(self) -> Any:
        """Load the model pack once. Raises FACE_MODEL_UNAVAILABLE on failure."""
        if self._analyzer is not None:
            return self._analyzer

        try:
            from insightface.app import FaceAnalysis
        except Exception as exc:  # pragma: no cover - install-time problem
            raise ServiceError(
                f"InsightFace is not importable: {exc}",
                code=ErrorCode.FACE_MODEL_UNAVAILABLE.value,
                retryable=False,
            ) from exc

        kwargs: dict[str, Any] = {
            "name": self.model_pack,
            "providers": ["CPUExecutionProvider"],
            # Load detection + recognition only: no gender/age or landmark
            # inference, which this stage does not need.
            "allowed_modules": ["detection", "recognition"],
        }
        if self.settings.face_model_root:
            kwargs["root"] = self.settings.face_model_root

        # InsightFace narrates model loading on stdout; keep it out of CLI
        # output and demote it to a debug log line.
        chatter = io.StringIO()
        try:
            with contextlib.redirect_stdout(chatter):
                analyzer = FaceAnalysis(**kwargs)
                analyzer.prepare(ctx_id=-1, det_size=(self.det_size, self.det_size))
        except Exception as exc:
            raise ServiceError(
                f"Could not load InsightFace pack {self.model_pack!r}: {exc}",
                code=ErrorCode.FACE_MODEL_UNAVAILABLE.value,
                retryable=False,
            ) from exc
        finally:
            noise = chatter.getvalue().strip()
            if noise:
                logger.debug("insightface model load: %s", noise.replace("\n", " | "))

        logger.info(
            "loaded local face models pack=%s det_size=%d modules=detection,recognition",
            self.model_pack,
            self.det_size,
        )
        self._analyzer = analyzer
        return analyzer

    def warm_up(self) -> "InsightFaceIdentifier":
        """Load the local model pack now rather than on first analysis.

        Raises ``FACE_MODEL_UNAVAILABLE`` if the pack cannot be loaded, which
        lets a caller (or ``--health``) surface that before processing an image.
        """
        self._ensure_analyzer()
        return self

    @property
    def analyzer(self) -> Any:
        """The loaded analyzer, loading it on first access.

        Exposed so a second identifier (e.g. one with a different score
        threshold) can reuse an already-loaded pack instead of reloading it.
        """
        return self._ensure_analyzer()

    def clear_cache(self) -> None:
        """Drop the in-memory analysis, including its embeddings."""
        self._cache = None

    # -- decoding --------------------------------------------------------

    @staticmethod
    def _read_bytes(source: ImageSource) -> tuple[bytes, str]:
        """Return ``(encoded_bytes, image_path)`` for a path or bytes input."""
        if isinstance(source, (bytes, bytearray, memoryview)):
            payload = bytes(source)
            if not payload:
                raise _invalid_image("Image bytes are empty.")
            return payload, ""

        path = Path(os.fspath(source))
        if not path.exists():
            raise _invalid_image(f"Image file does not exist: {path}")
        if not path.is_file():
            raise _invalid_image(f"Image path is not a file: {path}")
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise _invalid_image(f"Image file could not be read: {exc}") from exc
        if not payload:
            raise _invalid_image(f"Image file is empty: {path}")
        return payload, str(path)

    def _decode(self, payload: bytes) -> np.ndarray:
        """Decode encoded bytes to a BGR array, or raise INVALID_IMAGE.

        Reading the bytes ourselves and using ``cv2.imdecode`` (rather than
        ``cv2.imread``) keeps non-ASCII paths working on Windows.
        """
        buffer = np.frombuffer(payload, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise _invalid_image(
                "Image could not be decoded; the file is corrupt or not a "
                "supported image format."
            )
        if image.ndim != 3 or image.shape[2] != 3:
            raise _invalid_image(f"Expected a 3-channel image, got shape {image.shape}")
        height, width = image.shape[:2]
        if height < 1 or width < 1:
            raise _invalid_image("Decoded image has no pixels.")
        if height * width > MAX_PIXELS:
            raise _invalid_image(
                f"Image is too large to process: {width}x{height} exceeds "
                f"{MAX_PIXELS} pixels."
            )
        return image

    # -- analysis --------------------------------------------------------

    def _analyse(self, source: ImageSource) -> tuple[_Analysis, str]:
        """Decode and run detection + embedding. Cached per image digest."""
        payload, image_path = self._read_bytes(source)
        digest = hashlib.sha256(payload).hexdigest()

        cached = self._cache
        if cached is not None and cached.image_sha256 == digest:
            return cached, image_path

        image = self._decode(payload)
        height, width = image.shape[:2]
        analyzer = self._ensure_analyzer()

        detection_started = time.perf_counter()
        try:
            detections = analyzer.get(image)
        except Exception as exc:
            raise ServiceError(
                f"Face analysis failed: {exc}",
                code=ErrorCode.FACE_ENCODING_FAILED.value,
                retryable=False,
            ) from exc
        detection_duration_ms = int((time.perf_counter() - detection_started) * 1000)

        faces: list[_AnalysedFace] = []
        encoding_started = time.perf_counter()
        for detection in detections or ():
            score = float(getattr(detection, "det_score", 0.0) or 0.0)
            if score < self.min_det_score:
                continue
            try:
                embedding = EmbeddingVector(_pick_embedding(detection))
            except ValueError as exc:
                # A malformed vector is a stage failure, not a crash.
                raise ServiceError(
                    f"The recognition model returned an unusable embedding: {exc}",
                    code=ErrorCode.FACE_ENCODING_FAILED.value,
                    retryable=False,
                ) from exc
            faces.append(
                _AnalysedFace(
                    bbox=_to_bounding_box(detection.bbox, width, height),
                    score=score,
                    embedding=embedding,
                )
            )

        faces.sort(key=lambda face: face.score, reverse=True)
        analysis = _Analysis(
            image_sha256=digest,
            width=width,
            height=height,
            faces=tuple(faces),
            detection_duration_ms=detection_duration_ms,
            encoding_duration_ms=int((time.perf_counter() - encoding_started) * 1000),
            total_duration_ms=int((time.perf_counter() - detection_started) * 1000),
        )
        self._cache = analysis

        logger.info(
            "face analysis complete faces=%d image=%dx%d image_sha256=%s",
            len(faces),
            width,
            height,
            digest[:12],
        )
        return analysis, image_path

    # -- structured entry point -----------------------------------------

    def analyze(self, source: ImageSource) -> FaceDetectionResult:
        """Validate, detect, and encode; report the outcome structurally.

        Expected rejections (undecodable image, zero faces, several faces) come
        back as ``success=False`` with an ``error_code`` rather than as
        exceptions, so a caller can branch without ``try``.
        """
        result, _ = self.analyze_with_embedding(source)
        return result

    def analyze_with_embedding(
        self, source: ImageSource
    ) -> tuple[FaceDetectionResult, Optional[EmbeddingVector]]:
        """Same as :meth:`analyze`, plus the in-memory embedding on success.

        The vector is returned separately and deliberately kept off
        :class:`FaceDetectionResult`, so a result object can be serialised or
        logged without leaking biometric data.
        """
        try:
            analysis, image_path = self._analyse(source)
        except ServiceError as exc:
            code = _code_or(exc.code, ErrorCode.INVALID_IMAGE)
            logger.warning("face stage rejected input: %s", exc)
            return (
                FaceDetectionResult(
                    success=False,
                    error=str(exc),
                    error_code=code,
                    detector_model=self.detector_model,
                    encoder_model=self.encoder_model,
                ),
                None,
            )

        common = {
            "detector_model": self.detector_model,
            "encoder_model": self.encoder_model,
            "image_path": image_path,
            "image_sha256": analysis.image_sha256,
            "faces": tuple(
                DetectedFace(
                    bounding_box=face.bbox,
                    detector_confidence=face.score,
                    detector_model=self.detector_model,
                )
                for face in analysis.faces
            ),
            "detection_duration_ms": analysis.detection_duration_ms,
            "encoding_duration_ms": analysis.encoding_duration_ms,
            "total_duration_ms": analysis.total_duration_ms,
        }
        count = len(analysis.faces)

        if count == 0:
            logger.info("face stage: no face detected")
            return (
                FaceDetectionResult(
                    success=False,
                    face_count=0,
                    error=(
                        "No face could be detected in the image. Provide a "
                        "clearer, front-facing image of a single face."
                    ),
                    error_code=ErrorCode.NO_FACE_DETECTED,
                    **common,
                ),
                None,
            )

        if count > 1:
            logger.info("face stage: %d faces detected, expected exactly one", count)
            return (
                FaceDetectionResult(
                    success=False,
                    face_count=count,
                    error=(
                        f"Detected {count} faces. Provide an image containing "
                        f"exactly one face."
                    ),
                    error_code=ErrorCode.MULTIPLE_FACES,
                    **common,
                ),
                None,
            )

        face = analysis.faces[0]
        embedding = face.embedding
        encoding = FaceEncoding(
            input_id=_input_id(analysis.image_sha256),
            model=self.encoder_model,
            encoding_reference=embedding.digest,
            dimension=embedding.dimension,
        )
        logger.info(
            "face stage: 1 face accepted score=%.4f bbox=%s embedding_dim=%d "
            "embedding_sha256=%s",
            face.score,
            (face.bbox.x, face.bbox.y, face.bbox.width, face.bbox.height),
            embedding.dimension,
            embedding.digest[:12],
        )
        return (
            FaceDetectionResult(
                success=True,
                face_count=1,
                bbox=face.bbox,
                confidence=face.score,
                embedding_available=True,
                embedding_dimension=embedding.dimension,
                encoding=encoding,
                **common,
            ),
            embedding,
        )

    def analyze_all_with_embeddings(
        self, source: ImageSource
    ) -> tuple[FaceDetectionResult, tuple[EmbeddingVector, ...], tuple[FaceEncoding, ...]]:
        """Return every detected face and its in-memory encoding for matching."""
        try:
            analysis, image_path = self._analyse(source)
        except ServiceError as exc:
            code = _code_or(exc.code, ErrorCode.INVALID_IMAGE)
            return FaceDetectionResult(success=False, error=str(exc), error_code=code), (), ()
        faces = tuple(
            DetectedFace(
                bounding_box=item.bbox,
                detector_confidence=item.score,
                detector_model=self.detector_model,
            )
            for item in analysis.faces
        )
        encodings = tuple(
            FaceEncoding(
                input_id=_input_id(analysis.image_sha256),
                model=self.encoder_model,
                encoding_reference=item.embedding.digest,
                dimension=item.embedding.dimension,
            )
            for item in analysis.faces
        )
        result = FaceDetectionResult(
            success=len(faces) == 1,
            face_count=len(faces),
            faces=faces,
            detector_model=self.detector_model,
            encoder_model=self.encoder_model,
            image_path=image_path,
            image_sha256=analysis.image_sha256,
            embedding_available=bool(encodings),
            embedding_dimension=encodings[0].dimension if encodings else 0,
            bbox=faces[0].bounding_box if len(faces) == 1 else None,
            confidence=faces[0].detector_confidence if len(faces) == 1 else 0.0,
            encoding=encodings[0] if len(encodings) == 1 else None,
            error_code=ErrorCode.NO_FACE_DETECTED if not faces else (
                ErrorCode.MULTIPLE_FACES if len(faces) > 1 else None
            ),
        )
        return result, tuple(item.embedding for item in analysis.faces), encodings

    # -- FaceIdentifier protocol ----------------------------------------

    def load(self, image_path: str) -> FaceInput:
        """Validate the image and return its metadata.

        Detection has to run to populate ``face_count``; the result is cached,
        so a following :meth:`detect` costs nothing extra.
        """
        analysis, resolved = self._analyse(image_path)
        return FaceInput(
            input_id=_input_id(analysis.image_sha256),
            image_path=resolved,
            face_count=len(analysis.faces),
            width=analysis.width,
            height=analysis.height,
            image_sha256=analysis.image_sha256,
        )

    def detect(self, face_input: FaceInput) -> Sequence[DetectedFace]:
        """Return every face found, enforcing the single-face contract.

        Raises ``NO_FACE_DETECTED`` / ``MULTIPLE_FACES``; use :meth:`analyze`
        for the non-raising form.
        """
        analysis = self._require_analysis(face_input)
        count = len(analysis.faces)
        if count == 0:
            raise ServiceError(
                "No face could be detected in the image.",
                code=ErrorCode.NO_FACE_DETECTED.value,
                retryable=False,
            )
        if count > 1:
            raise ServiceError(
                f"Detected {count} faces. Provide an image containing exactly one face.",
                code=ErrorCode.MULTIPLE_FACES.value,
                retryable=False,
            )
        return [
            DetectedFace(
                bounding_box=face.bbox,
                detector_confidence=face.score,
                detector_model=self.detector_model,
            )
            for face in analysis.faces
        ]

    def encode(self, face_input: FaceInput, face: DetectedFace) -> FaceEncoding:
        """Return the reference-only encoding for a detected face.

        Use :meth:`analyze_with_embedding` when the vector itself is needed.
        """
        analysis = self._require_analysis(face_input)
        match = _match_by_bbox(analysis.faces, face.bounding_box)
        if match is None:
            raise ServiceError(
                "The supplied face does not belong to the analysed image.",
                code=ErrorCode.FACE_ENCODING_FAILED.value,
                retryable=False,
            )
        return FaceEncoding(
            input_id=face_input.input_id or _input_id(analysis.image_sha256),
            model=self.encoder_model,
            encoding_reference=match.embedding.digest,
            dimension=match.embedding.dimension,
        )

    def embedding_for(self, face_input: FaceInput) -> EmbeddingVector:
        """The in-memory embedding of the single face in ``face_input``."""
        self.detect(face_input)  # re-applies the exactly-one-face gate
        analysis = self._require_analysis(face_input)
        return analysis.faces[0].embedding

    def _require_analysis(self, face_input: FaceInput) -> _Analysis:
        cached = self._cache
        if cached is not None and (
            not face_input.image_sha256 or cached.image_sha256 == face_input.image_sha256
        ):
            return cached
        if face_input.image_path:
            analysis, _ = self._analyse(face_input.image_path)
            return analysis
        raise ServiceError(
            "No analysis is cached for this input; call load() first.",
            code=ErrorCode.FACE_ENCODING_FAILED.value,
            retryable=False,
        )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _input_id(image_sha256: str) -> str:
    return f"img_{image_sha256[:16]}"


def _code_or(raw: str, fallback: ErrorCode) -> ErrorCode:
    try:
        return ErrorCode(raw)
    except ValueError:
        return fallback


def _to_bounding_box(raw: Any, width: int, height: int) -> BoundingBox:
    """Convert an InsightFace ``[x1, y1, x2, y2]`` box, clamped to the image."""
    x1, y1, x2, y2 = (float(value) for value in np.asarray(raw, dtype=np.float32)[:4])
    left = max(0, min(int(round(min(x1, x2))), width))
    top = max(0, min(int(round(min(y1, y2))), height))
    right = max(0, min(int(round(max(x1, x2))), width))
    bottom = max(0, min(int(round(max(y1, y2))), height))
    return BoundingBox(x=left, y=top, width=max(0, right - left), height=max(0, bottom - top))


def _pick_embedding(detection: Any) -> Any:
    """Prefer the L2-normalised embedding, falling back to the raw vector."""
    for attribute in ("normed_embedding", "embedding"):
        vector = getattr(detection, attribute, None)
        if vector is not None:
            return vector
    raise ServiceError(
        "The recognition model returned no embedding for the detected face.",
        code=ErrorCode.FACE_ENCODING_FAILED.value,
        retryable=False,
    )


def _match_by_bbox(
    faces: Sequence[_AnalysedFace], bbox: BoundingBox, tolerance: int = 2
) -> Optional[_AnalysedFace]:
    for face in faces:
        if (
            abs(face.bbox.x - bbox.x) <= tolerance
            and abs(face.bbox.y - bbox.y) <= tolerance
            and abs(face.bbox.width - bbox.width) <= tolerance
            and abs(face.bbox.height - bbox.height) <= tolerance
        ):
            return face
    return None


def build_face_identifier(settings: Optional[Settings] = None) -> InsightFaceIdentifier:
    """Construct the real face identifier. Models load lazily on first use."""
    return InsightFaceIdentifier(settings or load_settings())
