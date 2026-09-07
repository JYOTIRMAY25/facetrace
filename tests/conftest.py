"""Shared fixtures for the face-processing tests.

FIXTURE SOURCE — read this before changing anything below.

No image file is committed to this repository. Every fixture image is generated
at test time inside pytest's temporary directory and removed with it.

The one photograph of a person used here is ``skimage.data.astronaut()``: a
NASA publicity photograph of astronaut Eileen Collins, released into the public
domain, and shipped as a data file inside the installed ``scikit-image``
package (``skimage/data/astronaut.png``). It is read from that local file, so
there is no network fetch, no personal or private photo, and no image of a
person who has not consented to publication. ``scikit-image`` is already
installed as a transitive dependency of ``insightface``, so relying on it adds
no new dependency.

Derived fixtures:

* **single face** — the photograph unmodified.
* **multiple faces** — the photograph next to a copy of itself, so the frame
  genuinely contains two faces for the detector to find.
* **no face** — a flat grey frame and a seeded-noise frame. Neither contains a
  person, so "zero faces" is the honest result rather than a rigged one.
* **invalid** — non-image bytes, a truncated PNG, an empty file, a directory.

The detector and encoder are never stubbed: every detection count, bounding
box, score, and embedding asserted in these tests comes from the real local
InsightFace models running on these generated images.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.services.face_insightface import InsightFaceIdentifier

#: Runtime dependencies of the face stage. Missing any of them skips these
#: tests rather than failing them, so the STEP 1 suite still runs on a machine
#: without the heavy wheels installed.
FACE_STAGE_MODULES = ("cv2", "numpy", "insightface", "onnxruntime")


def _require_face_stage() -> None:
    for module in FACE_STAGE_MODULES:
        pytest.importorskip(module, reason=f"{module} is required by the face stage")


def _astronaut_bgr() -> Any:
    """The public-domain NASA photograph, as an OpenCV BGR array."""
    _require_face_stage()
    data = pytest.importorskip(
        "skimage.data",
        reason="scikit-image supplies the public-domain fixture photograph",
    )
    import cv2

    return cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)


def _write_png(path: Path, image: Any) -> Path:
    import cv2

    assert cv2.imwrite(str(path), image), f"could not write fixture image {path}"
    return path


# --------------------------------------------------------------------------
# image fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def face_fixture_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Throwaway directory for generated images. Never inside the repository."""
    return tmp_path_factory.mktemp("face-fixtures")


@pytest.fixture(scope="session")
def single_face_image(face_fixture_dir: Path) -> Path:
    """An image containing exactly one real face."""
    return _write_png(face_fixture_dir / "single_face.png", _astronaut_bgr())


@pytest.fixture(scope="session")
def multi_face_image(face_fixture_dir: Path) -> Path:
    """An image containing two real faces, side by side."""
    import numpy as np

    face = _astronaut_bgr()
    return _write_png(face_fixture_dir / "two_faces.png", np.hstack([face, face]))


@pytest.fixture(scope="session")
def flat_no_face_image(face_fixture_dir: Path) -> Path:
    """A uniform grey frame — no person in it at all."""
    import numpy as np

    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    return _write_png(face_fixture_dir / "flat_grey.png", frame)


@pytest.fixture(scope="session")
def noise_no_face_image(face_fixture_dir: Path) -> Path:
    """Deterministic random noise — no person in it at all."""
    import numpy as np

    rng = np.random.default_rng(20260901)
    frame = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    return _write_png(face_fixture_dir / "noise.png", frame)


@pytest.fixture(scope="session")
def flipped_face_image(face_fixture_dir: Path) -> Path:
    """The same photograph mirrored — different pixels, still one face.

    Used to show the embedding is derived from image content rather than being
    a constant.
    """
    import cv2

    return _write_png(
        face_fixture_dir / "flipped_face.png", cv2.flip(_astronaut_bgr(), 1)
    )


@pytest.fixture(scope="session")
def corrupt_image(face_fixture_dir: Path) -> Path:
    """A ``.jpg`` whose bytes are not an image."""
    path = face_fixture_dir / "not_really_an_image.jpg"
    path.write_bytes(b"This file has a .jpg name and is plain text.\n" * 4)
    return path


@pytest.fixture(scope="session")
def truncated_image(face_fixture_dir: Path) -> Path:
    """A real PNG header followed by nothing usable."""
    path = face_fixture_dir / "truncated.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 24)
    return path


@pytest.fixture(scope="session")
def empty_image_file(face_fixture_dir: Path) -> Path:
    path = face_fixture_dir / "empty.png"
    path.write_bytes(b"")
    return path


# --------------------------------------------------------------------------
# service fixture
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _loaded_face_identifier() -> "InsightFaceIdentifier":
    """Load the local model pack once for the whole session."""
    _require_face_stage()
    from app.services import ServiceError
    from app.services.face_insightface import build_face_identifier

    identifier = build_face_identifier()
    try:
        identifier.warm_up()
    except ServiceError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"local InsightFace model pack is unavailable: {exc}")
    return identifier


@pytest.fixture
def face_identifier(
    _loaded_face_identifier: "InsightFaceIdentifier",
) -> "InsightFaceIdentifier":
    """The real identifier, with no analysis carried over from another test."""
    _loaded_face_identifier.clear_cache()
    return _loaded_face_identifier
