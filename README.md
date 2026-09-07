# FaceTrace

**HH Goa 2026 — Task 3: Face Identification & Blockchain Verification**

FaceTrace takes a face image, searches the web/social sources for content that
matches it, fingerprints the discovered content with SHA-256, anchors that
fingerprint on a blockchain, then reads the record back and recomputes the
fingerprint to report **VERIFIED** or **NOT VERIFIED**.

> ### Scope and safety
>
> * This is a hackathon prototype.
> * **Only process images and content you have permission to process.**
> * A `VERIFIED` result means *the discovered content is unchanged since its
>   fingerprint was anchored on-chain*. It reports a **content match** — it does
>   **not** establish, prove, or assert any person's legal identity.
> * External search results are treated as untrusted data.
> * No API key or private key is ever committed; secrets live in `.env` only.

---

## Current status: Step 5 — evidence extraction and canonicalization complete

| Capability | STEP 1 | STEP 2 |
|---|---|---|
| Project structure, config loading, result models | ✅ done | — |
| Replaceable service interfaces + placeholders | ✅ done | — |
| CLI health check | ✅ done | — |
| Face detection / encoding | interface only | to implement |
| Genuine web/social search | interface only | ✅ implemented (STEP 3) |
| Match selection | interface only | to implement |
| SHA-256 fingerprint | interface + rules documented | to implement |
| Blockchain write/read | interface only | to implement |
| Verification / tamper test | interface only | to implement |

The backend now performs local face detection and face encoding after upload.
It still implements **no** reverse-image search, candidate matching, SHA-256,
blockchain, or verification behavior. Uploaded images are validated, written
to a random temporary filename, analyzed locally, searched through the
configured provider, compared against retrieved candidate images, and removed
after the request.
Every placeholder service raises `StageNotImplementedError` rather than
returning invented data, and `--image` exits non-zero with `NOT_IMPLEMENTED`.

---

## Pipeline

```text
Face image
   ↓  validate_input
Face detection            → NO_FACE_DETECTED / MULTIPLE_FACES
   ↓  detect_face
Face encoding
   ↓  encode_face
Genuine web/social search → SEARCH_FAILED / NO_SEARCH_RESULTS
   ↓  search
Real matching post        → NO_RELIABLE_MATCH
   ↓  select_match
Canonicalize (version 1)
   ↓  fingerprint
SHA-256 fingerprint
   ↓  blockchain_write
Blockchain record
   ↓  blockchain_read
On-chain hash
   ↓  recompute_fingerprint
Local hash rebuild
   ↓  verify
Compare
  ↙        ↘
MATCH      MISMATCH
  ↓            ↓
VERIFIED   NOT VERIFIED
```

## Replaceable architecture

The orchestrator depends only on protocols, so any capability can be swapped
without touching the pipeline:

```text
FaceInput → FaceIdentifier → SearchProvider → MatchSelector
→ FingerprintService → BlockchainService → VerificationService
```

| Interface | Module | STEP 1 binding |
|---|---|---|
| `FaceIdentifier` | `app/services/face.py` | `PendingFaceIdentifier` |
| `SearchProvider`, `SearchResultParser` | `app/services/search.py` | `PendingSearchProvider` |
| `FaceMatcher`, `MatchSelector` | `app/services/matching.py` | `PendingMatchSelector` |
| `FingerprintService` | `app/services/fingerprint.py` | `PendingFingerprintService` |
| `BlockchainService` | `app/services/blockchain.py` | `PendingBlockchainService` |
| `VerificationService` | `app/services/verification.py` | `PendingVerificationService` |

## Layout

```text
facetrace/
├── app/
│   ├── __init__.py          version + content-match disclaimer
│   ├── config.py            environment-driven Settings
│   ├── main.py              CLI entry point (--health / --image)
│   ├── orchestrator.py      stage order + dependency injection
│   ├── models/
│   │   ├── __init__.py
│   │   └── pipeline.py      all stage models, enums, error codes
│   └── services/
│       ├── __init__.py      ServiceError / StageNotImplementedError
│       ├── face.py
│       ├── face_insightface.py   local detection + encoding (STEP 2)
│       ├── search.py
│       ├── search_http.py        HTTP boundary + credential scrubbing (STEP 3)
│       ├── search_web.py         live search providers (STEP 3)
│       ├── matching.py
│       ├── fingerprint.py
│       ├── blockchain.py
│       └── verification.py
├── tests/                   import, config, model, interface, CLI tests
├── data/                    local artefacts (git-ignored except .gitkeep)
├── .env.example
├── .gitignore
├── pytest.ini
├── requirements.txt
├── IMPLEMENTATION_PLAN.md   STEP 1 record + STEP 2 entry points
└── README.md
```

Project documents (PRD, architecture, data model, demo plan) live in the parent
directory of this project.

## Setup

```bash
cd facetrace
python -m venv .venv
```

Activate it — Windows:

```bash
.venv/Scripts/activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

```bash
cp .env.example .env
```

Every variable is optional in STEP 1 — the package imports and the health check
runs with all of them blank. `.env.example` documents each one. `--health`
reports exactly which variables a live search or a live blockchain write would
still need.

`.gitignore` excludes `.env`, `*.key`, `*.pem`, and everything under `data/`.

## Run

Start the FastAPI backend:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Foundation endpoints:

* `GET /api/health` — service health.
* `POST /api/investigate` — validates a JPEG, PNG, or WEBP upload, detects all
  faces, and generates in-memory embeddings without exposing them in the
  response.

For one face the endpoint runs the configured genuine search provider and
returns normalized real candidate URLs. Multiple faces are not searched and
return `422 MULTIPLE_FACES`; no face returns `422 NO_FACE_DETECTED`.

### Evidence Model

For a real `MATCH_FOUND`, FaceTrace records the investigation identifier, safe
source-image metadata, configured search provider and timestamp, candidate count,
the observed candidate URLs, face index, similarity, distance, and match status.
Raw images, embeddings, credentials, headers, cookies, and temporary paths are
deliberately excluded. Social-media classification is based only on the returned
candidate domain.

## Evidence Fingerprinting

FaceTrace extracts evidence, canonicalizes it, encodes the canonical JSON as
UTF-8, and calculates a SHA-256 digest over those exact bytes. The resulting
lowercase 64-character hexadecimal value represents that precise evidence
state. Even a small change to the canonical evidence produces a different
fingerprint. The uploaded image itself is not used as the evidence fingerprint.

## Verification

Verification recalculates SHA-256 from canonical evidence bytes and compares it
with the expected digest using a constant-time comparison. This proves that the
fingerprint corresponds to the exact evidence representation that was hashed; it
does not prove that social-media content is authentic.

## Blockchain

Step 7 supports EVM-compatible networks through Web3.py. A deployed
`FingerprintRegistry` stores only the SHA-256 evidence hash, investigation ID,
recording timestamp, and recorder address. Images, embeddings, canonical JSON,
credentials, and other private data remain off-chain.

The RPC endpoint, chain ID, signing key, contract address, network label, and
optional explorer base URL are supplied through environment variables:
`BLOCKCHAIN_RPC_URL`, `BLOCKCHAIN_CHAIN_ID`, `BLOCKCHAIN_PRIVATE_KEY`,
`BLOCKCHAIN_CONTRACT_ADDRESS` (or the existing `CONTRACT_ADDRESS`), 
`BLOCKCHAIN_NETWORK`, and `BLOCKCHAIN_EXPLORER_URL`.

Deploy [FingerprintRegistry.sol](D:/task%2003%20goa/facetrace/contracts/FingerprintRegistry.sol)
to a local chain or testnet, then set its address before enabling writes.
The service validates RPC connectivity, configured chain ID, wallet key, and
receipt status. A transaction is reported as confirmed only after a successful
receipt; no transaction hash or block number is fabricated. Blockchain
read-back and independent verification are intentionally deferred to Step 8.

### Foundry deployment setup

The reproducible deployment source is under `src/`, with configuration in
`foundry.toml` and the deployment script in `script/DeployFingerprintRegistry.s.sol`.
The contract uses Solidity `0.8.20`, matching its pragma. Install Foundry using
the official installer, then from the `facetrace` directory run:

```powershell
forge build
forge script script/DeployFingerprintRegistry.s.sol:DeployFingerprintRegistry `
  --rpc-url $env:BLOCKCHAIN_RPC_URL `
  --broadcast
```

The deployment command requires `BLOCKCHAIN_RPC_URL`, `BLOCKCHAIN_PRIVATE_KEY`,
`BLOCKCHAIN_CHAIN_ID=11155111`, and `BLOCKCHAIN_NETWORK=sepolia` in the local
environment. Never commit `.env` or print the private key. After a confirmed
deployment, copy the returned contract address into both
`BLOCKCHAIN_CONTRACT_ADDRESS` and the compatible `CONTRACT_ADDRESS` setting in
your local `.env`. The generated `out/`, `cache/`, and `broadcast/` directories
are ignored by Git.

### Canonical Evidence

Evidence is serialized as deterministic compact JSON with sorted keys, explicit
nulls, stable booleans and numbers, no insignificant whitespace, and UTF-8
encoding. This produces byte-for-byte stable input for the later SHA-256 step;
this step does not calculate a hash or call blockchain services.

### Genuine reverse-image search

The configured provider adapters are:

* `serpapi-google-lens` — genuine image-based Google Lens search via SerpAPI.
  It requires `SEARCH_API_KEY` and an operator-supplied public
  `SEARCH_IMAGE_URL`; the local upload is never published automatically.
* `google-custom-search` — Google Programmable Search image API, requiring
  `SEARCH_API_KEY`, `SEARCH_ENGINE_ID`, and `SEARCH_TERMS`.
* `wikimedia-commons` — real Commons API search, requiring `SEARCH_TERMS` and
  a contactable `SEARCH_USER_AGENT`.

Provider responses are sanitized, deduplicated, and returned without invented
URLs. Missing configuration, timeouts, rate limits, and empty results are
reported explicitly. ### Candidate image retrieval and face matching

Candidate media is retrieved only from URLs returned by the genuine search
provider. Retrieval follows HTTP(S) redirects, enforces supported image
content types, a 10 MiB response limit, request timeouts, and decoded-image
validation. Each candidate is processed independently, so inaccessible or
invalid media does not terminate the investigation.

Candidate faces use the same local InsightFace model as the source. Similarity
is cosine similarity over the model's L2-normalized embeddings; distance is
`1 - similarity`. `FACE_MATCH_THRESHOLD` controls `MATCH_FOUND` classification
(default `0.65`, meaning the cosine score must be at least 0.65). Results are
ranked by the actual computed similarity. The API never returns embeddings,
image bytes, credentials, or filesystem paths.

This is visual similarity matching only. It does not establish legal identity,
ownership, authorship, account ownership, or authenticity of a person.

InsightFace runs locally through CPU ONNX Runtime. Model weights may be
downloaded on first use unless `FACE_MODEL_ROOT` points to a pre-provisioned
model directory. Embeddings are never persisted, logged, or returned.

The upload limit defaults to 10 MiB and can be changed with
`MAX_UPLOAD_SIZE_BYTES`. Error responses use stable codes such as
`INVALID_IMAGE`, `FILE_TOO_LARGE`, and `UNSUPPORTED_MEDIA_TYPE`; they never
include local paths, secrets, or stack traces.

Health check (config + wiring + stage plan, secrets redacted):

```bash
python -m app.main --health
```

Machine-readable:

```bash
python -m app.main --health --json
```

Pipeline run — STEP 1 reports `NOT_IMPLEMENTED` and exits 3:

```bash
python -m app.main --image data/input.jpg
```

Exit codes: `0` success · `1` bad usage · `2` config problem · `3` pipeline
incomplete · `4` completed as `NOT VERIFIED`.

## Genuine web/social search (STEP 3)

The search really executes against an external provider. There is no canned
payload, no pre-selected post, and no hardcoded result anywhere in `app/`; a
faked HTTP boundary exists only inside `tests/`.

A face embedding cannot be handed to a search engine — search APIs take text or
an image URL, not a 512-dimensional vector. So this stage runs the search each
provider genuinely supports, collects **candidates**, and keeps the biometric
comparison local for a later stage. **No result is claimed to depict the person
in the submitted image.** The image itself never leaves the machine.

| `SEARCH_PROVIDER` | What it searches | Required variables |
|---|---|---|
| `wikimedia-commons` *(default)* | Commons file pages + real media URLs, keyless | `SEARCH_TERMS`, `SEARCH_USER_AGENT` |
| `google-custom-search` | Google Programmable Search JSON API, image mode | `SEARCH_API_KEY`, `SEARCH_ENGINE_ID`, `SEARCH_TERMS` |
| `serpapi-google-lens` | SerpAPI Google Lens reverse **image** search | `SEARCH_API_KEY`, `SEARCH_IMAGE_URL` |

Credentials come from the environment only. They are never hardcoded, never
printed, and scrubbed out of every log line and error message — including URLs
embedded in third-party exceptions. `--health` reports which variables are still
missing without echoing any value.

Wikimedia's [robot policy](https://w.wiki/4wJS) requires a `User-Agent` naming
the tool **and a contact**, and its API answers HTTP 403 without one. FaceTrace
will not fake a contact to get past that filter, so the keyless provider reports
`SEARCH_USER_AGENT` as missing configuration until you supply your own:

```bash
export SEARCH_USER_AGENT="FaceTrace/0.1 (HH Goa 2026 Task 3; you@example.org)"
```

Run one real search — local face detection and encoding, then a live query:

```bash
python -m app.main --search --image data/input.jpg --terms "public figure name"
```

Reverse image search needs a public URL you are permitted to use; the local file
is still never uploaded:

```bash
python -m app.main --search --image data/input.jpg --provider serpapi-google-lens --image-url https://example.org/photo.jpg
```

Outcomes: `SUCCESS` with genuine candidates · `NO_RESULTS` when the provider
honestly returned nothing usable (never a match) · `ERROR` with a structured
code (`SEARCH_NOT_CONFIGURED`, `SEARCH_QUERY_INVALID`, `SEARCH_FAILED`).

The stage refuses to invent query terms: a text provider with no `SEARCH_TERMS`
fails with `SEARCH_QUERY_INVALID` rather than running a blank search. Provider
content is untrusted — only `http(s)` URLs are accepted, HTML and control
characters are stripped, strings are length-capped, and a candidate with no
genuine URL is dropped rather than patched up.

## Tests

```bash
python -m pytest
```

STEP 1 tests cover: every module imports without STEP 2 dependencies;
configuration loads from an explicit mapping, rejects bad numbers, and never
leaks secrets through `repr()` or the health report; `.env.example` stays in
sync with `config.py` and holds no values; the stage order matches the
architecture document; stage models are frozen and JSON-serialisable; every
placeholder satisfies its protocol and refuses to produce data; the CLI health
check succeeds and an `--image` run reports `NOT_IMPLEMENTED`.

`tests/test_search_live.py` performs a **real** search and is skipped unless you
opt in, so a plain `pytest` never touches the network:

```bash
FACETRACE_LIVE_SEARCH=1 SEARCH_TERMS="public figure name" python -m pytest tests/test_search_live.py -v
```

It skips itself when the selected provider is not configured, and asserts that a
configured `SEARCH_API_KEY` appears nowhere in the outcome.

## Canonicalization (version 1)

Fixed before any hashing code exists, because verification must reproduce it
byte-for-byte:

* Fields, in order: `url`, `source`, `title`, `text`, `image_url`.
* Unicode NFC normalisation on every text field.
* Collapse whitespace runs to a single space, then strip.
* URLs: lowercase scheme and host, drop the default port and a trailing slash
  on an empty path; leave path, query, and fragment untouched.
* A missing field emits an empty string — never skipped, so field boundaries
  stay stable.
* Join with `\n`, prefix the version, encode UTF-8, then SHA-256.
* Any change to these rules requires a version bump.

## Limitations

* **STEP 1 verifies nothing.** No stage is implemented; the pipeline entry point
  exists to be replaced.
* Face matching is probabilistic: false positives and false negatives are both
  expected, and the threshold is unvalidated until tuned experimentally.
* A match is a content match, never legal identity.
* Search coverage is bounded by the chosen provider's index and terms of use.
* Blockchain results on a local chain or testnet carry no mainnet guarantees.

## Next step

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the STEP 2 entry points.
