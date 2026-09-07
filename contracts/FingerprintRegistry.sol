// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract FingerprintRegistry {
    struct EvidenceRecord {
        bytes32 evidenceHash;
        string investigationId;
        uint256 timestamp;
        address recorder;
    }

    mapping(bytes32 => EvidenceRecord) public records;

    function recordEvidence(bytes32 evidenceHash, string calldata investigationId) external {
        require(records[evidenceHash].timestamp == 0, "already recorded");
        records[evidenceHash] = EvidenceRecord(
            evidenceHash,
            investigationId,
            block.timestamp,
            msg.sender
        );
    }
}
