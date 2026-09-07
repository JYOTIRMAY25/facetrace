"""The real face stage: validate image -> detect face -> encode face.

Every assertion here runs against the local InsightFace models on genuinely
generated images. Nothing about a successful detection is stubbed, faked, or
hardcoded: if the detector stops finding the face in the fixture photograph,
these tests fail rather than pass.

See ``conftest.py`` for the fixture images and their public-domain source.

An embedding here supports a content-similarity comparison. It does not
establish anyone's legal identity.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="the face stage requires NumPy")
pytest.importorskip("cv2", reason="the face stage requires OpenCV")

from app.models.pipeline import (  # noqa: E402
    BoundingBox,
    DetectedFace,
    ErrorCode,
    FaceDetectionResult,
    FaceEncoding,
    FaceInput,
)
from app.services import ServiceError  # noqa: E402
from app.services.face import FaceIdentifier  # noqa: E402
from app.services.face_insightface import (  # noqa: E402
    EmbeddingVector,
    InsightFaceIdentifier,
    build_face_identifier,
)


# --------------------------------------------------------------------------
# 1. invalid image input
# --------------------------------------------------------------------------


def test_missing_file_is_rejected(face_identifier: InsightFaceIdentifier) -> None:
    result = face_identifier.analyze("does-not-exist-anywhere.png")

    assert result.success is False
    assert result.error_code is ErrorCode.INVALID_IMAGE
    assert "does not exist" in result.error
    assert result.embedding_available is False
    assert result.encoding is None


def test_non_image_bytes_are_rejected(
    face_identifier: InsightFaceIdentifier, corrupt_image: Path
) -> None:
    """A file named .jpg whose contents are text must not decode."""
    result = face_identifier.analyze(corrupt_image)

    assert result.success is False
    assert result.error_code is ErrorCode.INVALID_IMAGE
    assert "could not be decoded" in result.error
    assert result.face_count == 0
    assert result.bbox is None


def test_truncated_image_is_rejected(
    face_identifier: InsightFaceIdentifier, truncated_image: Path
) -> None:
    result = face_identifier.analyze(truncated_image)

    assert result.success is False
    assert result.error_code is ErrorCode.INVALID_IMAGE


def test_empty_file_is_rejected(
    face_identifier: InsightFaceIdentifier, empty_image_file: Path
) -> None:
    result = face_identifier.analyze(empty_image_file)

    assert result.success is False
    assert result.error_code is ErrorCode.INVALID_IMAGE
    assert "empty" in result.error


def test_empty_bytes_are_rejected(face_identifier: InsightFaceIdentifier) -> None:
    result = face_identifier.analyze(b"")

    assert result.success is False
    assert result.error_code is ErrorCode.INVALID_IMAGE


def test_directory_path_is_rejected(
    face_identifier: InsightFaceIdentifier, face_fixture_dir: Path
) -> None:
    result = face_identifier.analyze(face_fixture_dir)

    assert result.success is False
    assert result.error_code is ErrorCode.INVALID_IMAGE
    assert "not a file" in result.error


def test_invalid_image_raises_through_the_protocol_path(
    face_identifier: InsightFaceIdentifier, corrupt_image: Path
) -> None:
    """``analyze`` reports; ``load`` raises. Both must agree it is invalid."""
    with pytest.raises(ServiceError) as excinfo:
        face_identifier.load(str(corrupt_image))

    assert excinfo.value.code == ErrorCode.INVALID_IMAGE.value
    assert excinfo.value.retryable is False


# --------------------------------------------------------------------------
# 2. image with no face
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_name", ["flat_no_face_image", "noise_no_face_image"])
def test_image_with_no_face_is_rejected(
    face_identifier: InsightFaceIdentifier,
    request: pytest.FixtureRequest,
    fixture_name: str,
) -> None:
    image: Path = request.getfixturevalue(fixture_name)
    result = face_identifier.analyze(image)

    assert result.success is False
    assert result.face_count == 0
    assert result.error_code is ErrorCode.NO_FACE_DETECTED
    assert "no face could be detected" in result.error.lower()
    assert result.bbox is None
    assert result.confidence == 0.0
    assert result.embedding_available is False
    assert result.embedding_dimension == 0
    assert result.encoding is None
    # The image was still decoded successfully, so its metadata is populated.
    assert result.image_sha256


def test_no_face_raises_through_the_protocol_path(
    face_identifier: InsightFaceIdentifier, flat_no_face_image: Path
) -> None:
    face_input = face_identifier.load(str(flat_no_face_image))
    assert face_input.face_count == 0

    with pytest.raises(ServiceError) as excinfo:
        face_identifier.detect(face_input)

    assert excinfo.value.code == ErrorCode.NO_FACE_DETECTED.value


# --------------------------------------------------------------------------
# 3. image with exactly one face
# --------------------------------------------------------------------------


def test_single_face_image_is_accepted(
    face_identifier: InsightFaceIdentifier, single_face_image: Path
) -> None:
    result = face_identifier.analyze(single_face_image)

    assert result.success is True
    assert result.face_count == 1
    assert result.error == ""
    assert result.error_code is None

    assert isinstance(result.bbox, BoundingBox)
    assert result.bbox.width > 0 and result.bbox.height > 0
    assert 0.0 < result.confidence <= 1.0
    assert result.confidence >= face_identifier.min_det_score

    assert result.embedding_available is True
    assert result.embedding_dimension > 0
    assert result.detector_model and result.encoder_model
    assert result.image_path == str(single_face_image)
    assert len(result.image_sha256) == 64


def test_bounding_box_lies_inside_the_image(
    face_identifier: InsightFaceIdentifier, single_face_image: Path
) -> None:
    result = face_identifier.analyze(single_face_image)
    face_input = face_identifier.load(str(single_face_image))

    assert result.bbox is not None
    assert 0 <= result.bbox.x < face_input.width
    assert 0 <= result.bbox.y < face_input.height
    assert result.bbox.x + result.bbox.width <= face_input.width
    assert result.bbox.y + result.bbox.height <= face_input.height


def test_image_bytes_are_accepted_like_a_path(
    face_identifier: InsightFaceIdentifier, single_face_image: Path
) -> None:
    from_path = face_identifier.analyze(single_face_image)
    face_identifier.clear_cache()
    from_bytes = face_identifier.analyze(single_face_image.read_bytes())

    assert from_bytes.success is True
    assert from_bytes.face_count == 1
    assert from_bytes.image_sha256 == from_path.image_sha256
    assert from_bytes.bbox == from_path.bbox
    # Bytes have no filesystem location to report.
    assert from_bytes.image_path == ""


def test_protocol_pipeline_produces_an_encoding(
    face_identifier: InsightFaceIdentifier, single_face_image: Path
) -> None:
    """The STEP 1 contract, end to end: load -> detect -> encode."""
    face_input = face_identifier.load(str(single_face_image))
    assert isinstance(face_input, FaceInput)
    assert face_input.face_count == 1
    assert face_input.width > 0 and face_input.height > 0

    detected = face_identifier.detect(face_input)
    assert len(detected) == 1
    assert isinstance(detected[0], DetectedFace)
    assert detected[0].detector_confidence > 0
    assert detected[0].detector_model == face_identifier.detector_model

    encoding = face_identifier.encode(face_input, detected[0])
    assert isinstance(encoding, FaceEncoding)
    assert encoding.input_id == face_input.input_id
    assert encoding.model == face_identifier.encoder_model
    assert encoding.dimension > 0


def test_identifier_satisfies_the_face_identifier_protocol(
    face_identifier: InsightFaceIdentifier,
) -> None:
    """The real implementation drops into the STEP 1 interface unchanged."""
    assert isinstance(face_identifier, FaceIdentifier)
    assert face_identifier.name.startswith("insightface-")


def test_a_stricter_score_threshold_changes_the_outcome(
    face_identifier: InsightFaceIdentifier, single_face_image: Path
) -> None:
    """Proves the detector score is real and gated, not a constant.

    Reuses the already-loaded model pack, so this costs no extra load time.
    """
    from app.config import load_settings

    strict = InsightFaceIdentifier(
        load_settings(env={"FACE_MIN_DET_SCORE": "0.999"}),
        analyzer=face_identifier.analyzer,
    )
    result = strict.analyze(single_face_image)

    assert result.success is False
    assert result.face_count == 0
    assert result.error_code is ErrorCode.NO_FACE_DETECTED


# --------------------------------------------------------------------------
# 4. image with multiple faces
# --------------------------------------------------------------------------


def test_multiple_faces_are_rejected(
    face_identifier: InsightFaceIdentifier, multi_face_image: Path
) -> None:
    result = face_identifier.analyze(multi_face_image)

    assert result.success is False
    assert result.face_count > 1
    assert result.error_code is ErrorCode.MULTIPLE_FACES
    assert "exactly one face" in result.error.lower()
    assert str(result.face_count) in result.error
    assert result.embedding_available is False
    assert result.encoding is None


def test_multiple_faces_raise_through_the_protocol_path(
    face_identifier: InsightFaceIdentifier, multi_face_image: Path
) -> None:
    face_input = face_identifier.load(str(multi_face_image))
    assert face_input.face_count > 1

    with pytest.raises(ServiceError) as excinfo:
        face_identifier.detect(face_input)

    assert excinfo.value.code == ErrorCode.MULTIPLE_FACES.value
    assert excinfo.value.retryable is False


# --------------------------------------------------------------------------
# 5 & 6. a real embedding, with the expected numeric structure
# --------------------------------------------------------------------------


def test_embedding_is_generated_for_the_detected_face(
    face_identifier: InsightFaceIdentifier, single_face_image: Path
) -> None:
    result, embedding = face_identifier.analyze_with_embedding(single_face_image)

    assert result.success is True
    assert embedding is not None
    assert result.embedding_available is True
    assert result.embedding_dimension == embedding.dimension


def test_embedding_is_a_non_empty_numeric_vector(
    face_identifier: InsightFaceIdentifier, single_face_image: Path
) -> None:
    _, embedding = face_identifier.analyze_with_embedding(single_face_image)
    assert embedding is not None

    values = embedding.values
    assert isinstance(values, np.ndarray)
    assert values.dtype == np.float32
    assert values.ndim == 1
    assert values.size >= 128, "a real face embedding should be a high-dimension vector"
    assert values.size == embedding.dimension == len(embedding)
    assert np.all(np.isfinite(values))

    # Not a placeholder vector: it has real magnitude and real spread.
    norm = float(np.linalg.norm(values))
    assert norm > 0.0
    assert not np.allclose(values, 0.0)
    assert float(values.std()) > 0.0
    assert len(np.unique(values)) > values.size // 2

    # buffalo_l returns an L2-normalised embedding.
    assert math.isclose(norm, 1.0, rel_tol=1e-3)


def test_embedding_values_are_read_only(
    face_identifier: InsightFaceIdentifier, single_face_image: Path
) -> None:
    _, embedding = face_identifier.analyze_with_embedding(single_face_image)
    assert embedding is not None

    with pytest.raises(ValueError):
        embedding.values[0] = 1.0


def test_embedding_is_reproducible_for_the_same_image(
    face_identifier: InsightFaceIdentifier, single_face_image: Path
) -> None:
    """The same pixels must yield the same vector — no randomness."""
    _, first = face_identifier.analyze_with_embedding(single_face_image)
    face_identifier.clear_cache()
    _, second = face_identifier.analyze_with_embedding(single_face_image)

    assert first is not None and second is not None
    assert first.digest == second.digest
    assert np.array_equal(first.values, second.values)


def test_embedding_depends_on_the_image_content(
    face_identifier: InsightFaceIdentifier,
    single_face_image: Path,
    flipped_face_image: Path,
) -> None:
    """Different pixels must give a different vector — not a constant."""
    _, original = face_identifier.analyze_with_embedding(single_face_image)
    face_identifier.clear_cache()
    _, mirrored = face_identifier.analyze_with_embedding(flipped_face_image)

    assert original is not None and mirrored is not None
    assert original.dimension == mirrored.dimension
    assert original.digest != mirrored.digest
    assert not np.array_equal(original.values, mirrored.values)


def test_encoding_reference_is_a_digest_not_a_vector(
    face_identifier: InsightFaceIdentifier, single_face_image: Path
) -> None:
    result, embedding = face_identifier.analyze_with_embedding(single_face_image)

    assert embedding is not None
    assert result.encoding is not None
    reference = result.encoding.encoding_reference
    assert len(reference) == 64
    assert all(character in "0123456789abcdef" for character in reference)
    assert reference == embedding.digest
    assert result.encoding.dimension == embedding.dimension


def test_embedding_vector_rejects_malformed_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        EmbeddingVector(np.array([], dtype=np.float32))
    with pytest.raises(ValueError, match="1-D"):
        EmbeddingVector(np.zeros((2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        EmbeddingVector(np.array([1.0, np.nan, 3.0], dtype=np.float32))


def test_embeddings_are_not_retained_after_the_cache_is_cleared(
    face_identifier: InsightFaceIdentifier, single_face_image: Path
) -> None:
    """Nothing biometric survives ``clear_cache`` — no permanent storage."""
    face_input = face_identifier.load(str(single_face_image))
    assert face_identifier.embedding_for(face_input) is not None

    face_identifier.clear_cache()
    detached = FaceInput(
        input_id=face_input.input_id,
        image_path="",  # no file to re-read
        face_count=face_input.face_count,
        image_sha256=face_input.image_sha256,
    )
    with pytest.raises(ServiceError, match="No analysis is cached"):
        face_identifier.embedding_for(detached)


# --------------------------------------------------------------------------
# 7. the embedding never reaches the logs
# --------------------------------------------------------------------------


def _renderings(value: float) -> list[str]:
    """Plausible ways a careless log statement could print one component."""
    if abs(value) < 1e-6:  # '0.000000' would collide with unrelated text
        return [repr(value)]
    return [repr(value), f"{value:.6f}", f"{value:.8f}"]


def test_embedding_is_never_written_to_the_logs(
    face_identifier: InsightFaceIdentifier,
    single_face_image: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG):
        result, embedding = face_identifier.analyze_with_embedding(single_face_image)

    assert result.success is True
    assert embedding is not None

    logged = caplog.text
    assert logged, "the face stage should still log its non-sensitive metadata"

    # The whole vector, in the forms an f-string or %s would produce.
    assert str(embedding.tolist()) not in logged
    assert repr(embedding.values) not in logged
    assert str(embedding.values) not in logged

    # And no individual component either.
    for value in embedding.tolist():
        for rendered in _renderings(value):
            assert rendered not in logged, f"embedding value {rendered} leaked to logs"

    # What *is* logged: counts, a reference, and the dimension.
    assert f"embedding_dim={embedding.dimension}" in logged
    assert embedding.digest[:12] in logged


def test_embedding_wrapper_redacts_itself(
    face_identifier: InsightFaceIdentifier, single_face_image: Path
) -> None:
    """repr, str and format must all refuse to print the vector."""
    _, embedding = face_identifier.analyze_with_embedding(single_face_image)
    assert embedding is not None

    first = repr(float(embedding.values[0]))
    for rendered in (repr(embedding), str(embedding), f"{embedding}", format(embedding, ".6f")):
        assert "REDACTED" in rendered
        assert first not in rendered
        assert str(embedding.dimension) in rendered


def test_detection_result_carries_no_biometric_values(
    face_identifier: InsightFaceIdentifier, single_face_image: Path
) -> None:
    """A result may be serialised or printed freely; the vector is not in it."""
    result, embedding = face_identifier.analyze_with_embedding(single_face_image)
    assert embedding is not None

    serialised = json.dumps(dataclasses.asdict(result))
    rendered = f"{serialised}{result!r}"
    for value in embedding.tolist():
        for candidate in _renderings(value):
            assert candidate not in rendered

    assert '"embedding_available": true' in serialised
    assert embedding.digest in serialised  # the reference, not the vector


# --------------------------------------------------------------------------
# result model shape
# --------------------------------------------------------------------------


def test_face_detection_result_is_immutable() -> None:
    result = FaceDetectionResult(success=False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.success = True  # type: ignore[misc]


def test_face_detection_result_defaults_to_a_clean_failure() -> None:
    result = FaceDetectionResult(success=False)

    assert result.face_count == 0
    assert result.bbox is None
    assert result.embedding_available is False
    assert result.encoding is None
    assert result.error_code is None


def test_builder_returns_a_lazily_loaded_identifier() -> None:
    """Constructing the service must not load models or touch the network."""
    identifier = build_face_identifier()

    assert isinstance(identifier, InsightFaceIdentifier)
    assert isinstance(identifier, FaceIdentifier)
    assert identifier.model_pack
    assert identifier.det_size > 0
