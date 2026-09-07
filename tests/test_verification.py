import pytest
from app.models.pipeline import MatchingPost, BlockchainRecord, VerificationStatus
from app.services.fingerprint import Sha256FingerprintService
from app.services.blockchain_sim import InMemoryBlockchainService, BlockchainConnectionError
from app.services.verification_impl import HashVerificationService

@pytest.fixture
def fp_service():
    return Sha256FingerprintService()

@pytest.fixture
def bc_service():
    return InMemoryBlockchainService()

@pytest.fixture
def verification_service(fp_service, bc_service):
    return HashVerificationService(fp_service, bc_service)

def test_verify_match(verification_service, fp_service, bc_service):
    post = MatchingPost(result_id="rec-1", source="Reddit", url="https://reddit.com/r/test", title="Original Title", text="Original Text")
    fp = fp_service.fingerprint(post)
    record = bc_service.write_fingerprint("rec-1", fp)
    
    result = verification_service.verify(post, record)
    assert result.status == VerificationStatus.VERIFIED
    assert result.match is True
    assert result.computed_hash == fp.hash
    assert result.on_chain_hash == fp.hash

def test_verify_tampered_title(verification_service, fp_service, bc_service):
    post = MatchingPost(result_id="rec-1", source="Reddit", url="https://reddit.com/r/test", title="Original Title", text="Original Text")
    fp = fp_service.fingerprint(post)
    record = bc_service.write_fingerprint("rec-1", fp)

    tampered_post = MatchingPost(result_id="rec-1", source="Reddit", url="https://reddit.com/r/test", title="TAMPERED Title", text="Original Text")
    result = verification_service.verify(tampered_post, record)
    assert result.status == VerificationStatus.NOT_VERIFIED
    assert result.match is False

def test_verify_tampered_url(verification_service, fp_service, bc_service):
    post = MatchingPost(result_id="rec-1", source="Reddit", url="https://reddit.com/r/test", title="Original Title")
    fp = fp_service.fingerprint(post)
    record = bc_service.write_fingerprint("rec-1", fp)

    tampered_post = MatchingPost(result_id="rec-1", source="Reddit", url="https://reddit.com/r/tampered", title="Original Title")
    result = verification_service.verify(tampered_post, record)
    assert result.status == VerificationStatus.NOT_VERIFIED
    assert result.match is False

def test_verify_tampered_source(verification_service, fp_service, bc_service):
    post = MatchingPost(result_id="rec-1", source="Reddit", url="https://reddit.com/r/test", title="Original Title")
    fp = fp_service.fingerprint(post)
    record = bc_service.write_fingerprint("rec-1", fp)

    tampered_post = MatchingPost(result_id="rec-1", source="Twitter", url="https://reddit.com/r/test", title="Original Title")
    result = verification_service.verify(tampered_post, record)
    assert result.status == VerificationStatus.NOT_VERIFIED
    assert result.match is False

def test_blockchain_unavailable(fp_service):
    class ErrorBlockchainService:
        def read_fingerprint(self, record_id: str) -> str:
            raise RuntimeError("RPC Connection Refused")

    v_service = HashVerificationService(fp_service, ErrorBlockchainService())
    post = MatchingPost(result_id="rec-1", source="Reddit", url="https://reddit.com/r/test")
    record = BlockchainRecord(network="local", record_id="rec-1", record_hash="hash")
    
    result = v_service.verify(post, record)
    assert result.status == VerificationStatus.BLOCKCHAIN_UNAVAILABLE
    assert result.match is False

def test_fingerprint_unavailable(verification_service, fp_service):
    post = MatchingPost(result_id="rec-1", source="Reddit", url="https://reddit.com/r/test")
    # Record ID not stored in in-memory blockchain
    record = BlockchainRecord(network="local", record_id="unregistered-id", record_hash="hash")
    
    result = verification_service.verify(post, record)
    assert result.status == VerificationStatus.FINGERPRINT_UNAVAILABLE
    assert result.match is False

def test_invalid_evidence(verification_service):
    record = BlockchainRecord(network="local", record_id="rec-1", record_hash="hash")
    result = verification_service.verify(None, record)
    assert result.status == VerificationStatus.INVALID_EVIDENCE
    assert result.match is False

