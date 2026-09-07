"""EVM blockchain writer for confirmed evidence fingerprints."""

from __future__ import annotations

import re
from typing import Any, Protocol, runtime_checkable

from ..models.pipeline import BlockchainRecord, Fingerprint
from . import ServiceError, StageNotImplementedError

HASH_RE = re.compile(r"^[0-9a-f]{64}$")
REGISTRY_ABI = [
    {
        "inputs": [
            {"internalType": "bytes32", "name": "evidenceHash", "type": "bytes32"},
            {"internalType": "string", "name": "investigationId", "type": "string"},
        ],
        "name": "recordEvidence",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
        "name": "records",
        "outputs": [
            {"internalType": "bytes32", "name": "evidenceHash", "type": "bytes32"},
            {"internalType": "string", "name": "investigationId", "type": "string"},
            {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
            {"internalType": "address", "name": "recorder", "type": "address"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
]


class BlockchainError(ServiceError):
    """A blockchain operation failed without being converted to success."""

    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@runtime_checkable
class BlockchainService(Protocol):
    network: str

    def connect(self) -> bool:
        ...

    def write_fingerprint(self, record_id: str, fingerprint: Fingerprint) -> BlockchainRecord:
        ...

    def read_fingerprint(self, fingerprint: Fingerprint) -> str:
        ...


class Web3BlockchainService:
    """Submit a fingerprint to a deployed FingerprintRegistry contract."""

    def __init__(self, settings: Any, *, web3_factory: Any | None = None) -> None:
        self.settings = settings
        self.network = settings.blockchain_network
        self._web3_factory = web3_factory

    def _build_web3(self) -> Any:
        if not self.settings.blockchain_rpc_url or not self.settings.blockchain_private_key:
            raise BlockchainError("BLOCKCHAIN_NOT_CONFIGURED", "Blockchain configuration is incomplete.")
        try:
            from web3 import Web3
            factory = self._web3_factory or Web3
            return factory(Web3.HTTPProvider(self.settings.blockchain_rpc_url))
        except Exception as exc:
            raise BlockchainError("BLOCKCHAIN_CONNECTION_FAILED", "Unable to create the RPC client.", retryable=True) from exc

    def connect(self) -> bool:
        web3 = self._build_web3()
        try:
            if not web3.is_connected():
                raise BlockchainError("BLOCKCHAIN_CONNECTION_FAILED", "The configured RPC endpoint is unavailable.", retryable=True)
            chain_id = int(web3.eth.chain_id)
            expected = int(self.settings.blockchain_chain_id or 0)
            if expected and chain_id != expected:
                raise BlockchainError("BLOCKCHAIN_CONNECTION_FAILED", "The connected chain ID does not match configuration.")
            from eth_account import Account
            Account.from_key(self.settings.blockchain_private_key)
            if not self.settings.contract_address:
                raise BlockchainError("BLOCKCHAIN_NOT_CONFIGURED", "A deployed contract address is required.")
            return True
        except BlockchainError:
            raise
        except Exception as exc:
            raise BlockchainError("BLOCKCHAIN_CONNECTION_FAILED", "Blockchain wallet or network validation failed.") from exc

    def write_fingerprint(self, record_id: str, fingerprint: Fingerprint) -> BlockchainRecord:
        if fingerprint.algorithm != "SHA-256" or not HASH_RE.fullmatch(fingerprint.hash):
            raise BlockchainError("BLOCKCHAIN_TRANSACTION_FAILED", "The evidence fingerprint is not a valid SHA-256 value.")
        web3 = self._build_web3()
        self.connect()
        try:
            from eth_account import Account
            account = Account.from_key(self.settings.blockchain_private_key)
            contract = web3.eth.contract(
                address=web3.to_checksum_address(self.settings.contract_address),
                abi=REGISTRY_ABI,
            )
            evidence_hash = bytes.fromhex(fingerprint.hash)
            nonce = web3.eth.get_transaction_count(account.address)
            transaction = contract.functions.recordEvidence(
                evidence_hash, record_id
            ).build_transaction({
                "from": account.address,
                "nonce": nonce,
                "chainId": int(web3.eth.chain_id),
                "gas": int(contract.functions.recordEvidence(evidence_hash, record_id).estimate_gas({"from": account.address}) * 2),
            })
            signed = account.sign_transaction(transaction)
            tx_hash = web3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = web3.eth.wait_for_transaction_receipt(
                tx_hash, timeout=self.settings.blockchain_tx_timeout_s
            )
            if int(receipt["status"]) != 1:
                raise BlockchainError("BLOCKCHAIN_TRANSACTION_REVERTED", "The blockchain transaction reverted.")
            transaction_hash = web3.to_hex(tx_hash)
            block_number = int(receipt["blockNumber"])
            return BlockchainRecord(
                network=self.network,
                record_id=record_id,
                record_hash=fingerprint.hash,
                transaction_hash=transaction_hash,
                contract_address=self.settings.contract_address,
                block_number=block_number,
            )
        except BlockchainError:
            raise
        except TimeoutError as exc:
            raise BlockchainError("BLOCKCHAIN_CONFIRMATION_TIMEOUT", "Transaction confirmation timed out.", retryable=True) from exc
        except Exception as exc:
            raise BlockchainError("BLOCKCHAIN_TRANSACTION_FAILED", "Blockchain transaction submission failed.", retryable=True) from exc

    def read_fingerprint(self, fingerprint: Fingerprint) -> str:
        """Read the registry entry and return its stored evidence hash."""
        if fingerprint.algorithm != "SHA-256" or not HASH_RE.fullmatch(fingerprint.hash):
            raise BlockchainError("BLOCKCHAIN_VERIFICATION_FAILED", "The evidence fingerprint is not valid.")
        web3 = self._build_web3()
        self.connect()
        try:
            contract = web3.eth.contract(
                address=web3.to_checksum_address(self.settings.contract_address),
                abi=REGISTRY_ABI,
            )
            record = contract.functions.records(bytes.fromhex(fingerprint.hash)).call()
            on_chain_hash = record[0].hex()
            if on_chain_hash != fingerprint.hash:
                raise BlockchainError(
                    "BLOCKCHAIN_VERIFICATION_FAILED",
                    "The on-chain evidence hash does not match the local fingerprint.",
                )
            return on_chain_hash
        except BlockchainError:
            raise
        except Exception as exc:
            raise BlockchainError(
                "BLOCKCHAIN_VERIFICATION_FAILED",
                "Unable to read back the blockchain evidence record.",
                retryable=True,
            ) from exc


class PendingBlockchainService:
    name = "pending-blockchain-service"
    network = "unset"

    def connect(self) -> bool:
        raise StageNotImplementedError("Blockchain connection")

    def write_fingerprint(self, record_id: str, fingerprint: Fingerprint) -> BlockchainRecord:
        raise StageNotImplementedError("Blockchain write")

    def read_fingerprint(self, fingerprint: Fingerprint) -> str:
        raise StageNotImplementedError("Blockchain read-back")


__all__ = [
    "BlockchainError",
    "BlockchainService",
    "PendingBlockchainService",
    "REGISTRY_ABI",
    "Web3BlockchainService",
]
