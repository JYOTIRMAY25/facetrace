"""Blockchain stage implementations."""

from typing import Dict
from ..models.pipeline import BlockchainRecord, Fingerprint
from .blockchain import BlockchainService
from . import ServiceError

class BlockchainConnectionError(ServiceError):
    code = "BLOCKCHAIN_CONNECTION_FAILED"
    retryable = True

class BlockchainWriteError(ServiceError):
    code = "BLOCKCHAIN_WRITE_FAILED"
    retryable = True

class BlockchainReadError(ServiceError):
    code = "BLOCKCHAIN_READ_FAILED"
    retryable = True

class InMemoryBlockchainService:
    """Simulator in-memory blockchain service for offline test and demo scenarios."""

    network: str = "in-memory-simulator"

    def __init__(self) -> None:
        self._store: Dict[str, str] = {}

    def connect(self) -> bool:
        return True

    def write_fingerprint(self, record_id: str, fingerprint: Fingerprint) -> BlockchainRecord:
        if not record_id or not fingerprint or not fingerprint.hash:
            raise BlockchainWriteError("Invalid record_id or fingerprint")
        self._store[record_id] = fingerprint.hash
        return BlockchainRecord(
            network=self.network,
            record_id=record_id,
            record_hash=fingerprint.hash,
            transaction_hash="0xsimulatedtxhash",
            contract_address="0xsimulatedcontractaddress"
        )

    def read_fingerprint(self, record_id: str) -> str:
        if record_id not in self._store:
            raise BlockchainReadError(f"Record '{record_id}' not found in simulator storage.")
        return self._store[record_id]
