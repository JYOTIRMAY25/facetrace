"""Face stage interface: load an image, detect a face, encode it.

TODO(STEP 2): implement ``OpenCVFaceIdentifier`` (or an InsightFace-backed
equivalent) behind :class:`FaceIdentifier`:
  * load and validate the image (reject corrupt/unsupported files)
  * detect faces and return bounding boxes
  * raise NO_FACE_DETECTED for zero faces and MULTIPLE_FACES for >1
  * produce an encoding and record the model name/version
  * store only an ``encoding_reference``, never a raw embedding vector

Only process images the user is authorised to process.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ..models.pipeline import DetectedFace, FaceEncoding, FaceInput
from . import StageNotImplementedError

__all__ = ["FaceIdentifier", "PendingFaceIdentifier"]


@runtime_checkable
class FaceIdentifier(Protocol):
    """Detects and encodes the single face in a submitted image."""

    #: Human-readable implementation name, shown by ``--health``.
    name: str

    def load(self, image_path: str) -> FaceInput:
        """Validate the image and return its metadata.

        Raises a :class:`~app.services.ServiceError` with code
        ``INVALID_IMAGE`` when the file is missing, corrupt, or unsupported.
        """
        ...

    def detect(self, face_input: FaceInput) -> Sequence[DetectedFace]:
        """Return every face found in the image (possibly empty)."""
        ...

    def encode(self, face_input: FaceInput, face: DetectedFace) -> FaceEncoding:
        """Produce an encoding for one detected face."""
        ...


class PendingFaceIdentifier:
    """STEP 1 placeholder. Satisfies :class:`FaceIdentifier`, implements nothing."""

    name = "pending-face-identifier"

    def load(self, image_path: str) -> FaceInput:
        raise StageNotImplementedError("Face image loading/validation")

    def detect(self, face_input: FaceInput) -> Sequence[DetectedFace]:
        raise StageNotImplementedError("Face detection")

    def encode(self, face_input: FaceInput, face: DetectedFace) -> FaceEncoding:
        raise StageNotImplementedError("Face encoding")
