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


def test_startup_warms_the_face_models(monkeypatch):
    """The startup hook loads the InsightFace pack before the first request.

    The warmed analyzer is the same instance the investigation route already
    holds, so detection and embedding behaviour are unchanged: only the lazy
    model load moves out of the request path.
    """
    import app.main as main_module

    calls: list[str] = []

    class WarmDouble:
        def warm_up(self):
            calls.append("warm")
            return self

    monkeypatch.setattr(main_module, "investigation_face_identifier", WarmDouble())

    with TestClient(main_module.app):
        pass

    assert calls == ["warm"]


def test_startup_survives_a_failed_warmup(monkeypatch, caplog):
    """A warm-up failure is logged and must never prevent the app serving."""
    import app.main as main_module

    class Boom:
        def warm_up(self):
            raise RuntimeError("model pack unavailable")

    monkeypatch.setattr(main_module, "investigation_face_identifier", Boom())

    with TestClient(main_module.app) as warm_client:
        response = warm_client.get("/api/health")

    assert response.status_code == 200
    assert "warm-up skipped" in caplog.text
