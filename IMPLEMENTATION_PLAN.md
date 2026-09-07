# FaceTrace — Implementation Plan (STEP 1 record)

This file records what STEP 1 delivered and where STEP 2 plugs in. The overall
programme plan lives in `../IMPLEMENTATION_PLAN(2).md`, which remains the source
of truth for phase scope.

---

## 1. STEP 1 scope

Delivered:

* Project structure under `facetrace/`.
* Environment-driven configuration (`app/config.py`) that works with an empty
  environment and never leaks secrets.
* Structured models for every pipeline stage (`app/models/pipeline.py`),
  including stage enums, the documented error codes, and the run envelope.
* Replaceable service protocols with `Pending*` placeholders that raise
  `StageNotImplementedError` (`app/services/`).
* Orchestrator wiring and the stage order (`app/orchestrator.py`).
* CLI health entry point (`app/main.py`).
* Tests covering imports, configuration, models, interface substitutability, and
  the CLI.

Explicitly **not** delivered (STEP 2 onwards): face detection, face encoding,
face matching, genuine search, match selection, canonicalization/hashing code,
the smart contract, the blockchain client, and verification logic.

## 2. Deviation from `../IMPLEMENTATION_PLAN(2).md`

That document proposes a `src/` package with per-capability subpackages
(`src/face/detector.py`, `src/search/provider.py`, ...). STEP 1 was directed to
build the flatter `app/` layout instead:

| Programme plan | Built in STEP 1 |
|---|---|
| `src/config.py` | `app/config.py` |
| `src/models/*.py` (five modules) | `app/models/pipeline.py` (one module) |
| `src/face/{detector,encoder,matcher}.py` | `app/services/face.py` + `app/services/matching.py` |
| `src/search/{provider,parser,selector}.py` | `app/services/search.py` + `app/services/matching.py` |
| `src/fingerprint/hasher.py` | `app/services/fingerprint.py` |
| `src/blockchain/{client,contract,verifier}.py` | `app/services/blockchain.py` + `app/services/verification.py` |
| `src/pipeline/orchestrator.py` | `app/orchestrator.py` |
| `src/cli.py` + root `main.py` | `app/main.py` (`python -m app.main`) |

Same capabilities and the same interface boundary, fewer files. Two consequences
the team should decide on before STEP 2:

1. **Run command.** `../DEMO_PLAN.md` records
   `python main.py --image samples/input.jpg`. The current command is
   `python -m app.main --image data/input.jpg`. Either update the demo script or
   add a thin root `main.py` shim in STEP 2.
2. **Sample images.** The programme plan uses `samples/`; STEP 1 created `data/`
   (fully git-ignored) because face images should not be committed.

## 3. STEP 2 entry points

Each item below replaces one `Pending*` placeholder. The protocol is the
contract — no orchestrator change should be needed.

### 3.1 Face (`app/services/face.py`)

Implement `FaceIdentifier`:

* `load` — validate the file; raise `ServiceError(code="INVALID_IMAGE")` on a
  missing, corrupt, or unsupported file. Populate `FaceInput.image_sha256`.
* `detect` — return bounding boxes. Zero faces → `NO_FACE_DETECTED`;
  more than one → `MULTIPLE_FACES`.
* `encode` — produce a `FaceEncoding` carrying an opaque
  `encoding_reference` and the model name/version. Never persist a raw vector.

Tests: valid single face, no face, multiple faces, corrupt file, unsupported
format.

### 3.2 Search (`app/services/search.py`)

Implement `SearchProvider` against a real provider:

* `build_query` must derive terms from the encoding, so the query changes with
  the input. A constant query fails the "genuine search" requirement.
* `search` must actually execute and return real source URLs. An empty result
  set is `NO_SEARCH_RESULTS`, never a synthesised match.
* Transient network/provider failures → `SEARCH_FAILED` with `retryable=True`.
* Respect the provider's terms, rate limits, robots rules, and authentication.
  Do not bypass access controls.

Tests: results returned, provider failure, empty results.

### 3.3 Matching (`app/services/matching.py`)

Implement `FaceMatcher` and `MatchSelector`:

* Encode candidate images, score similarity, rank, then apply
  `settings.face_match_threshold`.
* Below threshold → `MatchDecision(matched=False, reason="NO_RELIABLE_MATCH")`.
* The chosen `MatchingPost` keeps its original URL and source.
* Record the final threshold in `README.md` once tuned experimentally.

### 3.4 Fingerprint (`app/services/fingerprint.py`)

Implement `Sha256FingerprintService` following the version-1 canonicalization
rules already documented in the module docstring and `README.md`.

Tests: determinism (same post → same hash), sensitivity (one changed character →
different hash), and explicit canonicalization cases (missing field, odd
whitespace, mixed-case host).

### 3.5 Contract and blockchain (`app/services/blockchain.py`)

* Add `contracts/FingerprintRegistry.sol` with
  `recordFingerprint(identifier, fingerprint)` and `getFingerprint(identifier)`.
* Store only `record_id`, `fingerprint`, and a timestamp on-chain. Post content,
  images, and encodings stay off-chain.
* Implement a Web3.py service against a local chain or a public testnet.
* Map failures to `BLOCKCHAIN_CONNECTION_FAILED`, `BLOCKCHAIN_WRITE_FAILED`,
  `BLOCKCHAIN_READ_FAILED`; only transient ones are `retryable`.
* Never log the private key.
* `read_fingerprint` must perform a real read, not echo the written value.

### 3.6 Verification (`app/services/verification.py`)

* Recompute from the original post through the same `FingerprintService`.
* Compare against the on-chain hash, case-insensitively, `0x` prefix stripped.
* Return `VERIFIED` / `NOT VERIFIED`; a mismatch is a result, not an exception.
* Add the tamper test: change one canonical field → `NOT VERIFIED`.

### 3.7 Orchestrator (`app/orchestrator.py`)

Replace the `NOT_IMPLEMENTED` body of `run()` with real execution:

* Iterate `PIPELINE_STAGES`, timing each stage into
  `StageResult.duration_ms` and recording a summary in `StageResult.data`.
* Convert any `ServiceError` into a `PipelineError` and stop the run.
* Retry only `retryable` failures, at most `settings.max_retries` times. Never
  retry an invalid image, a missing face, invalid credentials, or a hash
  mismatch.
* Populate `PipelineResult.status` from the verification result.

### 3.8 CLI (`app/main.py`)

Render the numbered stage report from `../DESIGN_BRIEF.md` (`[01] Face scan ✓`
… `STATUS: VERIFIED`) from `PipelineResult.stages`, and add `--tamper` for the
demo's mismatch case.

## 4. Constraints carried into STEP 2

* No hardcoded or pre-selected matching post; the search must genuinely run.
* No API keys, private keys, or `.env` in version control.
* No raw biometric data in `data/`, in logs, or on-chain.
* `VERIFIED` reports content integrity, not legal identity — keep that wording
  in the CLI, the README, and the demo narration.
* Only process content the user is authorised to process.

## 5. Suggested STEP order

```text
STEP 2  face detection + encoding
STEP 3  genuine search provider + parser
STEP 4  face matching + match selection
STEP 5  canonicalization + SHA-256
STEP 6  FingerprintRegistry.sol + blockchain client
STEP 7  verification + tamper test
STEP 8  orchestrator execution + CLI stage report
STEP 9  end-to-end run, README finalisation, demo recording
```
