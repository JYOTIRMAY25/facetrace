import pytest
from types import SimpleNamespace

from app.config import load_settings
from app.models.pipeline import Fingerprint
from app.services.blockchain import BlockchainError, Web3BlockchainService


def test_missing_configuration_is_controlled():
    service = Web3BlockchainService(load_settings(env={}))
    with pytest.raises(BlockchainError) as error:
        service.connect()
    assert error.value.code == "BLOCKCHAIN_NOT_CONFIGURED"


def test_malformed_fingerprint_is_rejected_before_rpc():
    settings = load_settings(
        env={
            "BLOCKCHAIN_RPC_URL": "http://127.0.0.1:8545",
            "BLOCKCHAIN_PRIVATE_KEY": "0x" + "1" * 64,
            "CONTRACT_ADDRESS": "0x" + "2" * 40,
        }
    )
    service = Web3BlockchainService(settings)
    with pytest.raises(BlockchainError) as error:
        service.write_fingerprint("inv-1", Fingerprint(hash="not-a-hash"))
    assert error.value.code == "BLOCKCHAIN_TRANSACTION_FAILED"


class _ReadFunction:
    def __init__(self, value):
        self.value = value

    def call(self):
        return self.value


class _ReadContract:
    def __init__(self, value):
        self.functions = SimpleNamespace(
            records=lambda _: _ReadFunction(value)
        )


class _ReadWeb3:
    def __init__(self, value):
        self.eth = SimpleNamespace(
            chain_id=11155111,
            contract=lambda **kwargs: _ReadContract(value),
        )
        self._value = value

    def is_connected(self):
        return True

    def to_checksum_address(self, address):
        return address

def _configured_service(monkeypatch, record):
    settings = load_settings(
        env={
            "BLOCKCHAIN_RPC_URL": "http://rpc.invalid",
            "BLOCKCHAIN_PRIVATE_KEY": "0x" + "1" * 64,
            "BLOCKCHAIN_CHAIN_ID": "11155111",
            "BLOCKCHAIN_CONTRACT_ADDRESS": "0x" + "2" * 40,
        }
    )
    service = Web3BlockchainService(settings)
    web3 = _ReadWeb3(record)
    monkeypatch.setattr(service, "_build_web3", lambda: web3)
    return service


def test_read_back_matching_hash_is_verified(monkeypatch):
    fingerprint = "ab" * 32
    service = _configured_service(
        monkeypatch,
        (bytes.fromhex(fingerprint), "inv-1", 1, "0x" + "3" * 40),
    )

    assert service.read_fingerprint(Fingerprint(hash=fingerprint)) == fingerprint


def test_read_back_mismatch_fails_verification(monkeypatch):
    service = _configured_service(
        monkeypatch,
        (bytes.fromhex("cd" * 32), "inv-1", 1, "0x" + "3" * 40),
    )

    with pytest.raises(BlockchainError) as error:
        service.read_fingerprint(Fingerprint(hash="ab" * 32))

    assert error.value.code == "BLOCKCHAIN_VERIFICATION_FAILED"
