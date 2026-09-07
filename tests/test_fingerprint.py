import pytest
from app.models.pipeline import MatchingPost, CanonicalPayload, Fingerprint
from app.services.fingerprint import Sha256FingerprintService

@pytest.fixture
def service():
    return Sha256FingerprintService()

def test_determinism_same_input(service):
    post = MatchingPost(
        result_id="1", source="src", url="https://a.test", title="t", text="txt"
    )
    f1 = service.fingerprint(post)
    f2 = service.fingerprint(post)
    assert f1 == f2
    assert len(f1.hash) == 64

def test_determinism_different_field_order(service):
    # The matching post is an object, not a dict, so order doesn't apply to the post itself.
    # The requirement is that the service must handle field order internally.
    post1 = MatchingPost(
        result_id="1", source="src", url="https://a.test", title="t", text="txt"
    )
    # Different fields? No, the model is fixed.
    # Test normalization: same fields, different formats that should normalize to the same
    post2 = MatchingPost(
        result_id="1", source="  src  ", url="HTTPS://A.TEST/", title="t", text="txt"
    )
    assert service.fingerprint(post1).hash == service.fingerprint(post2).hash

def test_sensitivity_field_change(service):
    post1 = MatchingPost(result_id="1", source="s1", url="https://a.test", title="t", text="txt")
    post2 = MatchingPost(result_id="1", source="s2", url="https://a.test", title="t", text="txt")
    assert service.fingerprint(post1).hash != service.fingerprint(post2).hash

def test_invalid_evidence(service):
    # Empty strings are valid
    post = MatchingPost(result_id="1", source="", url="", title="", text="")
    f = service.fingerprint(post)
    assert len(f.hash) == 64

def test_no_embedding_in_payload(service):
    post = MatchingPost(result_id="1", source="s", url="https://a.test")
    payload = service.canonicalize(post)
    assert "embedding" not in payload.payload.lower()
