// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {FingerprintRegistry} from "../src/FingerprintRegistry.sol";

interface Vm {
    function envUint(string calldata name) external returns (uint256);
    function envString(string calldata name) external returns (string memory);
    function startBroadcast(uint256 privateKey) external;
    function stopBroadcast() external;
}

contract DeployFingerprintRegistry {
    Vm private constant vm =
        Vm(address(uint160(uint256(keccak256("hevm cheat code")))));

    function run() external returns (FingerprintRegistry registry) {
        // Read deployment configuration at execution time; no credentials are
        // stored in the repository or embedded in the deployment artifact.
        vm.envString("BLOCKCHAIN_NETWORK");
        uint256 privateKey = vm.envUint("BLOCKCHAIN_PRIVATE_KEY");
        vm.startBroadcast(privateKey);
        registry = new FingerprintRegistry();
        vm.stopBroadcast();
    }
}
