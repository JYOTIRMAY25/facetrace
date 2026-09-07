from fastapi.testclient import TestClient
from app.api import app, verification_engine

client = TestClient(app)

def test_verify_endpoint_success():
    post_dict = {
        "result_id": "rec-123",
        "source": "Reddit",
        "url": "https://reddit.com/r/test",
        "title": "Post Title",
        "text": "Post text content"
    }

    # Pre-register in memory store for test
    from app.models.pipeline import MatchingPost
    m_post = MatchingPost(
        result_id="rec-123",
        source="Reddit",
        url="https://reddit.com/r/test",
        title="Post Title",
        text="Post text content"
    )
    fp = verification_engine._fingerprint_service.fingerprint(m_post)
    verification_engine._blockchain_service.write_fingerprint("rec-123", fp)

    record_dict = {
        "network": "in-memory-simulator",
        "record_id": "rec-123",
        "record_hash": fp.hash
    }

    response = client.post("/api/verify", json={"post": post_dict, "record": record_dict})
    assert response.status_code == 200
    data = response.json()
    assert data["match"] is True
    assert data["status"] == "VERIFIED"
    assert data["computed_hash"] == fp.hash

def test_verify_endpoint_tampered():
    post_dict = {
        "result_id": "rec-123",
        "source": "Reddit",
        "url": "https://reddit.com/r/test",
        "title": "TAMPERED Title",
        "text": "Post text content"
    }

    record_dict = {
        "network": "in-memory-simulator",
        "record_id": "rec-123",
        "record_hash": "somehash"
    }

    response = client.post("/api/verify", json={"post": post_dict, "record": record_dict})
    assert response.status_code == 200
    data = response.json()
    assert data["match"] is False
    assert data["status"] == "NOT VERIFIED"
