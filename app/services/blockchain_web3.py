"""Blockchain service implementation using Web3.py."""

import os
from typing import Optional
from web3 import Web3
from ..models.pipeline import BlockchainRecord, Fingerprint
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

class Web3BlockchainService:
    """Writes and reads fingerprint records on an EVM-compatible chain."""

    def __init__(self, rpc_url: str, private_key: str, contract_address: str, network: str = "local"):
        self.network = network
        self._w3 = Web3(Web3.HTTPProvider(rpc_url))
        self._private_key = private_key
        self._contract_address = contract_address
        
        # Simplified ABI for demo
        self._contract_abi = [
            {"inputs": [{"internalType": "string", "name": "recordId", "type": "string"}, {"internalType": "string", "name": "fingerprint", "type": "string"}], "name": "recordFingerprint", "outputs": [], "stateMutability": "nonpayable", "type": "function"},
            {"inputs": [{"internalType": "string", "name": "recordId", "type": "string"}], "name": "getFingerprint", "outputs": [{"internalType": "string", "name": "", "type": "string"}], "stateMutability": "view", "type": "function"}
        ]
        self._contract = self._w3.eth.contract(address=contract_address, abi=self._contract_abi)
        self._account = self._w3.eth.account.from_key(private_key)

    def connect(self) -> bool:
        """Verify RPC connectivity."""
        if not self._w3.is_connected():
            raise BlockchainConnectionError("Failed to connect to blockchain RPC.")
        return True

    def write_fingerprint(self, record_id: str, fingerprint: Fingerprint) -> BlockchainRecord:
        """Submit the fingerprint to the smart contract."""
        try:
            nonce = self._w3.eth.get_transaction_count(self._account.address)
            tx = self._contract.functions.recordFingerprint(
                record_id, fingerprint.hash
            ).build_transaction({
                "from": self._account.address,
                "nonce": nonce,
                "gas": 200000,
                "gasPrice": self._w3.eth.gas_price
            })
            
            signed_tx = self._w3.eth.account.sign_transaction(tx, self._private_key)
            tx_hash = self._w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            
            return BlockchainRecord(
                network=self.network,
                record_id=record_id,
                record_hash=fingerprint.hash,
                transaction_hash=self._w3.to_hex(tx_hash),
                contract_address=self._contract_address
            )
        except Exception as e:
            raise BlockchainWriteError(f"Transaction failed: {e}")

    def read_fingerprint(self, record_id: str) -> str:
        """Read the stored fingerprint hash back from the contract."""
        try:
            return self._contract.functions.getFingerprint(record_id).call()
        except Exception as e:
            raise BlockchainReadError(f"Read failed: {e}")
