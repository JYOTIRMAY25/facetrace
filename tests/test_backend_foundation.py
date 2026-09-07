from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "FaceTrace Backend Pipeline",
    }


def test_valid_no_face_image_is_rejected_by_face_stage(flat_no_face_image: Path):
    response = client.post(
        "/api/investigate",
        files={"file": ("source.png", flat_no_face_image.read_bytes(), "image/png")},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "NO_FACE_DETECTED"


def test_invalid_file_type():
    response = client.post(
        "/api/investigate",
        files={"file": ("source.txt", BytesIO(b"not an image"), "text/plain")},
    )
    assert response.status_code == 415
    assert response.json()["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_invalid_image_signature():
    response = client.post(
        "/api/investigate",
        files={"file": ("source.png", BytesIO(b"not a png"), "image/png")},
    )
    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_IMAGE"


def test_face_image_reaches_search_configuration(single_face_image: Path, monkeypatch):
    from app.api.routes import investigation
    from app.config import load_settings
    from app.services.search_web import build_search_service

    monkeypatch.setattr(
        investigation,
        "search_service",
        build_search_service(
            load_settings(
                env={
                    "SEARCH_PROVIDER": "serpapi-google-lens",
                    "SERPAPI_API_KEY": "",
                },
                use_dotenv=False,
            )
        ),
    )
    response = client.post(
        "/api/investigate",
        files={"file": ("face.png", single_face_image.read_bytes(), "image/png")},
    )
    assert response.status_code == 503
    assert response.json()["code"] == "SEARCH_PROVIDER_NOT_CONFIGURED"


def test_multiple_face_image_returns_all_faces(multi_face_image: Path):
    response = client.post(
        "/api/investigate",
        files={"file": ("faces.png", multi_face_image.read_bytes(), "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "multiple_faces_detected"
    assert payload["faces_detected"] == len(payload["faces"]) > 1
    assert [face["face_index"] for face in payload["faces"]] == list(
        range(payload["faces_detected"])
    )


def test_oversized_file():
    old_limit = app.state.settings.max_upload_size_bytes
    app.state.settings.max_upload_size_bytes = 8
    try:
        response = client.post(
            "/api/investigate",
            files={"file": ("source.png", BytesIO(PNG), "image/png")},
        )
    finally:
        app.state.settings.max_upload_size_bytes = old_limit
    assert response.status_code == 413
    assert response.json()["code"] == "FILE_TOO_LARGE"


def test_missing_file():
    response = client.post("/api/investigate")
    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_IMAGE"


def test_malformed_request():
    response = client.post(
        "/api/investigate",
        data={"file": "not multipart file data"},
    )
    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_REQUEST"


def test_startup_does_not_load_face_models(monkeypatch):
    """Render gives the service 512 Mi: loading the InsightFace pack during
    application startup caused an out-of-memory crash.

    Starting the app must perform zero model loads. Every path that could
    load the pack is intercepted at the class level, so a reintroduced
    warm-up hook - however it captures the identifier - fails this test.
    """
    import app.main as main_module
    from app.services.face_insightface import InsightFaceIdentifier

    loads: list[str] = []

    def warm_up_spy(self):
        loads.append("warm_up")
        return self

    def ensure_analyzer_spy(self):
        loads.append("_ensure_analyzer")
        return object()  # never used: startup must not get this far

    monkeypatch.setattr(InsightFaceIdentifier, "warm_up", warm_up_spy)
    monkeypatch.setattr(
        InsightFaceIdentifier, "_ensure_analyzer", ensure_analyzer_spy
    )

    with TestClient(main_module.app):
        pass

    assert loads == []


def test_face_identifier_is_constructed_lazy():
    """The production wiring must stay lazy: building the identifier loads no
    models, so a fresh deployment only pays the model-load cost on the first
    investigation request (which the 512 Mi plan can absorb)."""
    from app.config import load_settings
    from app.services.face_insightface import (
        InsightFaceIdentifier,
        build_face_identifier,
    )

    identifier = build_face_identifier(load_settings(use_dotenv=False))

    assert isinstance(identifier, InsightFaceIdentifier)
    assert identifier._analyzer is None, "pack must not load at construction"
    assert identifier._cache is None
