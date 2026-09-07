"""FaceTrace CLI entry point.

Usage:

    python -m app.main --health          # config + wiring check
    python -m app.main --health --json    # machine-readable
    python -m app.main --version
    python -m app.main --image data/input.jpg   # reports NOT_IMPLEMENTED (exit 3)

    # STEP 3: local face encoding, then one genuine web/social search
    python -m app.main --search --image data/input.jpg --terms "public figure name"
    python -m app.main --search --image data/input.jpg \\
        --provider serpapi-google-lens --image-url https://example.org/photo.jpg

Exit codes:
    0  success
    1  bad usage / invalid arguments
    2  configuration problem (including a search provider that is not configured)
    3  pipeline could not complete (NOT_IMPLEMENTED, face failure, search error)
    4  completed without a usable outcome (NOT VERIFIED, or search NO_RESULTS)

TODO(STEP 4+): render the numbered stage report from DESIGN_BRIEF.md
(``[01] Face scan ✓`` ... ``STATUS: VERIFIED``) driven by
``PipelineResult.stages``, and add ``--tamper`` to demonstrate that a modified
canonical field produces NOT VERIFIED.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Any, Optional, Sequence

from . import CONTENT_MATCH_DISCLAIMER, __version__
from .config import Settings, load_settings
from .models.pipeline import ErrorCode, SearchStatus, VerificationStatus
from .orchestrator import Orchestrator, build_default_orchestrator

__all__ = ["build_parser", "health_report", "main", "run_search"]

BANNER = (
    "+----------------------------------------+\n"
    "|              FACETRACE                 |\n"
    "|  Face-derived content verification     |\n"
    "+----------------------------------------+"
)

#: Printed with every search result set. The search stage finds candidates; it
#: does not decide that any of them depicts the submitted face.
SEARCH_CANDIDATE_NOTE = (
    "Search results are unverified external content. No result is claimed to "
    "depict the person in the submitted image; that comparison happens in a "
    "later stage."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="facetrace",
        description=(
            "Find web/social content matching a face image and verify it against "
            "a blockchain-anchored fingerprint. Reports a CONTENT match, not "
            "legal identity."
        ),
        epilog=CONTENT_MATCH_DISCLAIMER,
    )
    parser.add_argument("--image", help="path to the face image to process")
    parser.add_argument(
        "--search",
        action="store_true",
        help=(
            "run the face stage locally and then one genuine search; "
            "requires --image"
        ),
    )
    parser.add_argument(
        "--provider",
        help="override SEARCH_PROVIDER for this run",
    )
    parser.add_argument(
        "--terms",
        action="append",
        metavar="TERMS",
        help=(
            "operator-supplied query terms (comma-separated, repeatable). "
            "Required by the text-driven providers: the search stage will not "
            "invent query terms."
        ),
    )
    parser.add_argument(
        "--image-url",
        dest="image_url",
        help=(
            "public http(s) image URL for a reverse-image-search provider. "
            "The local image file is never uploaded."
        ),
    )
    parser.add_argument(
        "--max-results",
        dest="max_results",
        type=int,
        help="override SEARCH_MAX_RESULTS for this run",
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="print configuration and service wiring, then exit",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument("--version", action="version", version=f"facetrace {__version__}")
    return parser


def _search_health(settings: Settings) -> dict[str, object]:
    """Search-provider readiness. Reports presence of credentials, never values."""
    from .services.search_web import (
        PROVIDER_NAMES,
        SearchNotConfiguredError,
        SearchQueryError,
        build_search_provider,
    )

    section: dict[str, object] = {"available_providers": list(PROVIDER_NAMES)}
    try:
        section.update(build_search_provider(settings).describe())
    except (SearchNotConfiguredError, SearchQueryError) as exc:
        # Caught by the classes the provider module itself raises, so this stays
        # correct even when ``app.services`` has been re-imported (test_imports).
        section.update(
            {
                "provider": settings.search_provider,
                "ready": False,
                "error": str(exc),
            }
        )
    return section


def health_report(orchestrator: Optional[Orchestrator] = None) -> dict[str, object]:
    """Collect a secret-free snapshot of configuration and wiring."""
    orch = orchestrator or build_default_orchestrator()
    settings = orch.settings
    return {
        "version": __version__,
        "step": 1,
        "step_1_scope": "interfaces, configuration, and result models only",
        "config": settings.redacted(),
        "services": orch.service_map(),
        "search": _search_health(settings),
        "pipeline_stages": [stage.value for stage in orch.stages],
        "missing_for_live_search": list(settings.missing_for_search()),
        "missing_for_live_blockchain": list(settings.missing_for_blockchain()),
        "disclaimer": CONTENT_MATCH_DISCLAIMER,
    }


def _print_health(report: dict[str, object], as_json: bool) -> int:
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    print(BANNER)
    print(f"\nversion : {report['version']}")
    print(f"step    : {report['step']}  ({report['step_1_scope']})")

    print("\nConfiguration (secrets redacted)")
    config = report["config"]
    assert isinstance(config, dict)
    for key, value in config.items():
        print(f"  {key:<34} {value}")

    print("\nService wiring (all replaceable)")
    services = report["services"]
    assert isinstance(services, dict)
    for interface, impl in services.items():
        print(f"  {interface:<22} -> {impl}")

    search = report.get("search")
    if isinstance(search, dict):
        print("\nSearch provider (STEP 3, live)")
        for key, value in sorted(search.items()):
            print(f"  {key:<22} {value}")

    print("\nPipeline stages")
    stages = report["pipeline_stages"]
    assert isinstance(stages, list)
    for index, stage in enumerate(stages, start=1):
        print(f"  [{index:02d}] {stage}")

    for label, key in (
        ("live search", "missing_for_live_search"),
        ("live blockchain", "missing_for_live_blockchain"),
    ):
        missing = report[key]
        assert isinstance(missing, list)
        if missing:
            print(f"\nNot configured for {label}; missing: {', '.join(missing)}")
        else:
            print(f"\nConfigured for {label}.")

    print(f"\nNOTE: {report['disclaimer']}")
    return 0


def _search_settings(settings: Settings, args: argparse.Namespace) -> Settings:
    """Apply per-run CLI overrides on top of the environment configuration.

    Overrides flow through :class:`Settings` rather than through extra provider
    arguments, so the provider protocol stays exactly as STEP 1 defined it.
    """
    overrides: dict[str, Any] = {}
    if args.provider:
        overrides["search_provider"] = args.provider
    if args.terms:
        overrides["search_terms"] = ",".join(args.terms)
    if args.image_url:
        overrides["search_image_url"] = args.image_url
    if args.max_results:
        overrides["search_max_results"] = int(args.max_results)
    return settings.with_overrides(**overrides) if overrides else settings


def _print_search(outcome: Any, *, image_path: str) -> None:
    print(BANNER)
    print(f"\nimage    : {image_path}")
    print(f"provider : {outcome.provider}")

    query = outcome.query
    if query is not None:
        print(f"query id : {query.query_id}")
        if query.terms:
            print(f"terms    : {', '.join(query.terms)}")
        if query.image_url:
            print(f"image url: {query.image_url}")

    print(f"status   : {outcome.status.value}")
    print(f"attempts : {outcome.attempts}")
    print(f"results  : {outcome.result_count}")

    for index, result in enumerate(outcome.results, start=1):
        print(f"\n[{index:02d}] {result.title or '(no title)'}")
        print(f"     url      : {result.url}")
        print(f"     source   : {result.source}")
        if result.image_url:
            print(f"     image    : {result.image_url}")
        if result.published_at:
            print(f"     published: {result.published_at}")
        if result.text:
            print(f"     text     : {result.text}")

    if outcome.error:
        code = outcome.error_code.value if outcome.error_code else "SEARCH"
        print(f"\n[{code}] {outcome.error}")

    print(f"\nNOTE: {SEARCH_CANDIDATE_NOTE}")
    print(f"NOTE: {CONTENT_MATCH_DISCLAIMER}")


def run_search(args: argparse.Namespace, settings: Settings) -> int:
    """Encode the face locally, then run one genuine search for candidates.

    The image stays on this machine: only the operator-supplied query terms (or
    an operator-supplied public image URL) are sent to the provider. Nothing here
    asserts that a candidate depicts the person in the image.
    """
    from .services.face_insightface import build_face_identifier
    from .services.search_web import (
        SearchNotConfiguredError,
        SearchQueryError,
        build_search_service,
    )

    settings = _search_settings(settings, args)

    detection = build_face_identifier(settings).analyze(args.image)
    if not detection.success or detection.encoding is None:
        code = (
            detection.error_code.value
            if detection.error_code
            else ErrorCode.FACE_ENCODING_FAILED.value
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "stage": "encode_face",
                        "error_code": code,
                        "error": detection.error,
                        "face_count": detection.face_count,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(BANNER)
            print(f"\nimage    : {args.image}")
            print(f"\n[encode_face] {code}")
            print(f"  {detection.error}")
        return 3

    try:
        service = build_search_service(settings)
    except (SearchNotConfiguredError, SearchQueryError) as exc:
        print(f"{exc.code}: {exc}", file=sys.stderr)
        return 2

    outcome = service.search_for_encoding(detection.encoding)

    if args.json:
        print(json.dumps(asdict(outcome), indent=2, sort_keys=True, default=str))
    else:
        _print_search(outcome, image_path=args.image)

    if outcome.status is SearchStatus.SUCCESS:
        return 0
    if outcome.error_code in (
        ErrorCode.SEARCH_NOT_CONFIGURED,
        ErrorCode.SEARCH_QUERY_INVALID,
    ):
        return 2
    if outcome.status is SearchStatus.NO_RESULTS:
        return 4
    return 3


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.search and not args.image:
        print("--search requires --image", file=sys.stderr)
        return 1

    if not args.health and not args.image:
        parser.print_help()
        return 1

    try:
        settings = load_settings()
    except ValueError as exc:
        print(f"CONFIG_INVALID: {exc}", file=sys.stderr)
        return 2

    orchestrator = build_default_orchestrator(settings)

    if args.health:
        return _print_health(health_report(orchestrator), args.json)

    if args.search:
        return run_search(args, settings)

    result = orchestrator.run(args.image)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True, default=str))
    else:
        print(BANNER)
        print(f"\nimage  : {result.image_path}")
        print(f"status : {result.status.value}")
        if result.error is not None:
            print(f"\n[{result.error.stage.value}] {result.error.code.value}")
            print(f"  {result.error.message}")
        print(f"\nNOTE: {CONTENT_MATCH_DISCLAIMER}")

    if result.error is not None:
        return 3
    if result.status is not VerificationStatus.VERIFIED:
        return 4
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI shim
    raise SystemExit(main())


# HTTP application. Kept in this module so the documented command
# ``uvicorn app.main:app`` works without removing the existing CLI.
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging

from .api.routes.health import router as health_router
from .api.routes.investigation import router as investigation_router
from .core.config import get_settings
from .core.logging import configure_logging
from .utils.files import FileValidationError
from .models.pipeline import (
    BlockchainRecord,
    MatchingPost,
    VerificationStatus,
)
from .services.blockchain_sim import InMemoryBlockchainService
from .services.fingerprint import Sha256FingerprintService
from .services.verification_impl import HashVerificationService

configure_logging()
app = FastAPI(
    title="FaceTrace Backend",
    version="0.1.0",
    responses={400: {"description": "Invalid request"}, 413: {"description": "File too large"}},
)
app.state.settings = get_settings()
_fingerprint_service = Sha256FingerprintService()
_blockchain_service = InMemoryBlockchainService()
verification_engine = HashVerificationService(
    _fingerprint_service, _blockchain_service
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=app.state.settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.include_router(health_router, prefix="/api")
app.include_router(investigation_router, prefix="/api")


class VerifyRequest(BaseModel):
    post: dict
    record: dict


@app.post("/api/verify")
async def verify(request: VerifyRequest) -> dict[str, object]:
    """Preserve the existing verification endpoint for the current client."""
    post_data = request.post
    record_data = request.record
    matching_post = MatchingPost(
        result_id=post_data.get("result_id", ""),
        source=post_data.get("source", ""),
        url=post_data.get("url", ""),
        title=post_data.get("title", ""),
        text=post_data.get("text", ""),
        image_url=post_data.get("image_url"),
    )
    blockchain_record = BlockchainRecord(
        network=record_data.get("network", "local"),
        record_id=record_data.get("record_id", ""),
        record_hash=record_data.get("record_hash", ""),
        transaction_hash=record_data.get("transaction_hash", ""),
        contract_address=record_data.get("contract_address", ""),
        block_number=record_data.get("block_number", 0),
    )
    result = verification_engine.verify(matching_post, blockchain_record)
    return {
        "computed_hash": result.computed_hash,
        "on_chain_hash": result.on_chain_hash,
        "match": result.match,
        "status": result.status.value
        if isinstance(result.status, VerificationStatus)
        else str(result.status),
    }


@app.exception_handler(FileValidationError)
async def file_validation_error_handler(
    request: Request, exc: FileValidationError
) -> JSONResponse:
    logging.getLogger("facetrace.http").warning("image rejected: %s", exc.code)
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": exc.code, "message": exc.message},
    )


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"code": "INVALID_REQUEST", "message": "The request body is invalid."},
    )


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict) and "code" in detail and "message" in detail:
        content = detail
    else:
        content = {"code": "HTTP_ERROR", "message": str(detail)}
    return JSONResponse(status_code=exc.status_code, content=content)


@app.exception_handler(Exception)
async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logging.getLogger("facetrace.http").exception("request failed")
    return JSONResponse(
        status_code=500,
        content={"code": "INTERNAL_SERVER_ERROR", "message": "An internal server error occurred."},
    )
