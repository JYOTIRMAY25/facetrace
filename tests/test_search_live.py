"""Live search integration test — opt-in, real network.

Skipped unless ``FACETRACE_LIVE_SEARCH=1`` is set, so ``pytest`` on a plain
checkout never reaches out to the internet. Run it to prove the search stage
performs a genuine query rather than replaying a fixture::

    FACETRACE_LIVE_SEARCH=1 SEARCH_TERMS="Jimmy Wales" pytest tests/test_search_live.py -v

The default provider (``wikimedia-commons``) needs **no credentials**. To exercise
a credentialed provider, export its variables as well::

    SEARCH_PROVIDER=google-custom-search SEARCH_API_KEY=... SEARCH_ENGINE_ID=...
    SEARCH_PROVIDER=serpapi-google-lens  SEARCH_API_KEY=... SEARCH_IMAGE_URL=...

Nothing here asserts that a result depicts any particular person: the search
stage only collects public candidates. Only query terms (or an
operator-supplied public image URL) leave this machine — never the face image.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict

import pytest

from app.config import load_settings
from app.models.pipeline import FaceEncoding, SearchStatus
from app.services.search_web import build_search_service

LIVE_FLAG = "FACETRACE_LIVE_SEARCH"

pytestmark = pytest.mark.skipif(
    os.environ.get(LIVE_FLAG, "").strip() not in {"1", "true", "TRUE", "yes"},
    reason=f"live search disabled; set {LIVE_FLAG}=1 to run real queries",
)


@pytest.fixture(scope="module")
def service():  # type: ignore[no-untyped-def]
    settings = load_settings()
    described = build_search_service(settings).describe()
    if described["missing_configuration"]:
        pytest.skip(
            f"provider {described['provider']} needs: "
            f"{described['missing_configuration']}"
        )
    return build_search_service(settings)


@pytest.fixture(scope="module")
def encoding() -> FaceEncoding:
    """A reference-only encoding, standing in for the local face stage.

    The live test deliberately avoids loading InsightFace: the search stage only
    ever receives this opaque reference, never a biometric vector.
    """
    return FaceEncoding(
        input_id="live-integration",
        model="insightface-w600k_r50",
        encoding_reference="live" * 16,
        dimension=512,
    )


def test_live_search_returns_real_candidates(service, encoding: FaceEncoding) -> None:  # type: ignore[no-untyped-def]
    outcome = service.search_for_encoding(encoding)

    print("\n" + json.dumps(asdict(outcome), indent=2, sort_keys=True, default=str))

    if outcome.status is SearchStatus.ERROR:
        pytest.fail(f"live search failed: {outcome.error_code} {outcome.error}")

    assert outcome.attempts >= 1, "no HTTP request was made"
    assert outcome.provider == service.name
    assert outcome.retrieved_at.endswith("Z")

    if outcome.status is SearchStatus.NO_RESULTS:
        pytest.skip(
            "the provider genuinely returned no candidates for these terms; "
            "this is a valid NO_RESULTS, not a failure"
        )

    assert outcome.status is SearchStatus.SUCCESS
    assert outcome.results

    for result in outcome.results:
        assert result.url.startswith(("http://", "https://"))
        assert result.source
        assert result.retrieved_at.endswith("Z")
        # The search stage never scores a candidate; matching does that.
        assert result.match_score == 0.0


def test_live_search_does_not_leak_the_api_key(service, encoding: FaceEncoding) -> None:  # type: ignore[no-untyped-def]
    key = service.settings.search_api_key
    if not key:
        pytest.skip("no SEARCH_API_KEY configured for this provider")

    outcome = service.search_for_encoding(encoding)
    serialised = json.dumps(asdict(outcome), default=str)

    assert key not in serialised
    assert key not in outcome.error
    assert key not in repr(outcome)
