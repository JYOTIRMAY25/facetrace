"""Genuine web/social search (STEP 3).

The external HTTP boundary is faked here — that is the only thing these tests
substitute. :class:`FakeTransport` lives in this test module and is never
importable as an application default: :func:`test_application_default_transport_is_httpx`
pins the shipped transport to the real ``httpx`` one.

Every payload below is shaped like the provider's documented JSON response. The
tests assert that each returned field traces back to that payload, so an
implementation that invented, defaulted, or "helpfully" repaired a result would
fail rather than pass quietly.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional, Sequence

import pytest

from app.config import Settings
from app.models.pipeline import (
    ErrorCode,
    FaceEncoding,
    SearchQuery,
    SearchResult,
    SearchStatus,
)
from app.services import ServiceError
from app.services.search import SearchProvider
from app.services.search_http import (
    REDACTED,
    HttpClient,
    HttpResponse,
    HttpxTransport,
    SearchTransportError,
    merge_url,
    redact_secrets,
    redact_url,
)
from app.services.search_web import (
    PROVIDER_NAMES,
    PROVIDERS,
    GoogleCustomSearchProvider,
    SearchNotConfiguredError,
    SerpApiLensProvider,
    WikimediaCommonsProvider,
    build_search_provider,
    build_search_service,
    resolve_provider_name,
    sanitize_text,
    sanitize_url,
)

# --------------------------------------------------------------------------
# Test-only HTTP boundary
# --------------------------------------------------------------------------


@dataclass
class RecordedRequest:
    url: str
    params: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 0.0


class FakeTransport:
    """Replays queued responses and records what was sent.

    Test scaffolding only. Queue entries are either an :class:`HttpResponse` or
    an exception instance to raise. Extra requests are a test failure, so a
    retry loop cannot silently spin.
    """

    def __init__(self, *responses: Any) -> None:
        self._queue = list(responses)
        self.requests: list[RecordedRequest] = []
        self.served: list[HttpResponse] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout: float,
    ) -> HttpResponse:
        self.requests.append(
            RecordedRequest(url, dict(params), dict(headers), float(timeout))
        )
        if not self._queue:
            raise AssertionError(f"unexpected extra request #{len(self.requests)}")
        item = self._queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        # Mirror the real transport: the stored URL is always redacted.
        response = HttpResponse(
            status_code=item.status_code,
            text=item.text,
            url=redact_url(merge_url(url, params)),
            headers=item.headers,
        )
        self.served.append(response)
        return response

    def post(
        self,
        url: str,
        *,
        data: Mapping[str, Any],
        files: Mapping[str, tuple[str, bytes, str]],
        headers: Mapping[str, str],
        timeout: float,
    ) -> HttpResponse:
        self.requests.append(RecordedRequest(url, dict(data), dict(headers), float(timeout)))
        if not self._queue:
            raise AssertionError(f"unexpected extra request #{len(self.requests)}")
        item = self._queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        response = HttpResponse(
            status_code=item.status_code,
            text=item.text,
            url=redact_url(url),
            headers=item.headers,
        )
        self.served.append(response)
        return response

    @property
    def sent_params(self) -> dict[str, Any]:
        assert self.requests, "no request was sent"
        return self.requests[-1].params


def ok(payload: Any) -> HttpResponse:
    return HttpResponse(status_code=200, text=json.dumps(payload))


def status(code: int, payload: Any = None, *, text: str = "") -> HttpResponse:
    return HttpResponse(
        status_code=code, text=text if payload is None else json.dumps(payload)
    )


class RecordingSleeper:
    """Replaces ``time.sleep`` so retry tests do not actually wait."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


# --------------------------------------------------------------------------
# Provider-shaped payloads
# --------------------------------------------------------------------------

COMMONS_PAGE_URL = "https://commons.wikimedia.org/wiki/File:Sample_portrait.jpg"
COMMONS_FILE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/4/47/Sample_portrait.jpg"
)
COMMONS_SECOND_PAGE_URL = (
    "https://commons.wikimedia.org/wiki/File:Sample_portrait_second.jpg"
)
COMMONS_SECOND_FILE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/9/91/Sample_portrait_second.jpg"
)

WIKIMEDIA_PAYLOAD: dict[str, Any] = {
    "batchcomplete": True,
    "query": {
        "pages": [
            {
                "pageid": 4210123,
                "ns": 6,
                "title": "File:Sample portrait.jpg",
                "imageinfo": [
                    {
                        "timestamp": "2011-03-15T09:12:44Z",
                        "url": COMMONS_FILE_URL,
                        "descriptionurl": COMMONS_PAGE_URL,
                        "mime": "image/jpeg",
                        "extmetadata": {
                            "ImageDescription": {
                                "value": (
                                    "<p>Portrait recorded at a "
                                    '<a href="/wiki/Conference">conference</a>.</p>'
                                ),
                                "source": "commons-desc-page",
                            },
                            "DateTimeOriginal": {
                                "value": "2011-03-14",
                                "source": "commons-desc-page",
                            },
                            "LicenseShortName": {
                                "value": "CC BY-SA 3.0",
                                "source": "commons-desc-page",
                            },
                            "Artist": {
                                "value": '<a href="/wiki/User:Someone">Someone</a>',
                                "source": "commons-desc-page",
                            },
                        },
                    }
                ],
            },
            {
                "pageid": 4210124,
                "ns": 6,
                "title": "File:Sample portrait second.jpg",
                "imageinfo": [
                    {
                        "timestamp": "2014-07-02T18:00:01Z",
                        "url": COMMONS_SECOND_FILE_URL,
                        "descriptionurl": COMMONS_SECOND_PAGE_URL,
                        "mime": "image/jpeg",
                        "extmetadata": {
                            "LicenseShortName": {
                                "value": "CC0",
                                "source": "commons-desc-page",
                            }
                        },
                    }
                ],
            },
        ]
    },
}

WIKIMEDIA_EMPTY_PAYLOAD: dict[str, Any] = {
    "batchcomplete": True,
    "query": {"searchinfo": {"totalhits": 0}},
}

GOOGLE_ARTICLE_URL = "https://news.example.org/articles/interview"
GOOGLE_IMAGE_URL = "https://cdn.example.org/img/portrait-1.jpg"

GOOGLE_PAYLOAD: dict[str, Any] = {
    "kind": "customsearch#search",
    "searchInformation": {"totalResults": "1"},
    "items": [
        {
            "kind": "customsearch#result",
            "title": "Conference interview coverage",
            "link": GOOGLE_IMAGE_URL,
            "displayLink": "news.example.org",
            "snippet": "Portrait published alongside the interview.",
            "mime": "image/jpeg",
            "fileFormat": "image/jpeg",
            "image": {
                "contextLink": GOOGLE_ARTICLE_URL,
                "thumbnailLink": "https://cdn.example.org/img/portrait-1-thumb.jpg",
                "height": 1200,
                "width": 800,
            },
        }
    ],
}

SERPAPI_PAYLOAD: dict[str, Any] = {
    "search_metadata": {"id": "6600aa", "status": "Success"},
    "visual_matches": [
        {
            "position": 1,
            "title": "Conference interview coverage",
            "link": GOOGLE_ARTICLE_URL,
            "source": "news.example.org",
            "thumbnail": "https://serpapi.example/thumb-1.jpg",
        }
    ],
}

PUBLIC_IMAGE_URL = "https://cdn.example.org/operator-supplied/photo.jpg"
SERPAPI_UPLOAD_PAYLOAD = {"image_id": "img-upload-123"}


# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------


@pytest.fixture()
def encoding() -> FaceEncoding:
    """A reference-only encoding, as the face stage produces it.

    No image or biometric vector is needed here: the search stage only ever sees
    the opaque reference.
    """
    return FaceEncoding(
        input_id="img-0001",
        model="insightface-w600k_r50",
        encoding_reference="a" * 64,
        dimension=512,
    )


#: A ``User-Agent`` carrying a contact, as Wikimedia's robot policy requires.
#: Not a real address — it only has to satisfy the local readiness check, since
#: no test in this file reaches the network.
TEST_USER_AGENT = "FaceTrace/0.1 (tests; facetrace-tests@example.test)"


def make_service(
    transport: FakeTransport,
    *,
    sleeper: Optional[RecordingSleeper] = None,
    **overrides: Any,
) -> Any:
    """Build the real search service against a faked HTTP boundary."""
    base: dict[str, Any] = {
        "search_terms": "sample portrait",
        "search_user_agent": TEST_USER_AGENT,
        "max_retries": 2,
    }
    base.update(overrides)
    return build_search_service(
        Settings(**base), transport=transport, sleeper=sleeper or RecordingSleeper()
    )


# --------------------------------------------------------------------------
# 1. Successful search-provider response
# --------------------------------------------------------------------------


def test_successful_search_returns_genuine_candidates(encoding: FaceEncoding) -> None:
    transport = FakeTransport(ok(WIKIMEDIA_PAYLOAD))
    outcome = make_service(transport).search_for_encoding(encoding)

    assert outcome.status is SearchStatus.SUCCESS
    assert outcome.succeeded is True
    assert outcome.error == ""
    assert outcome.error_code is None
    assert outcome.provider == "wikimedia-commons"
    assert outcome.result_count == 2
    assert outcome.attempts == 1
    assert outcome.retrieved_at.endswith("Z")

    first = outcome.results[0]
    assert first.url == COMMONS_PAGE_URL
    assert first.image_url == COMMONS_FILE_URL
    assert first.title == "File:Sample portrait.jpg"
    assert first.source == "commons.wikimedia.org"
    assert first.published_at == "2011-03-14"
    assert first.retrieved_at.endswith("Z")
    # HTML from the provider's description field is stripped, text preserved.
    assert first.text == "Portrait recorded at a conference ."
    assert "<" not in first.text
    assert first.metadata["license"] == "CC BY-SA 3.0"
    assert first.metadata["mime"] == "image/jpeg"


def test_the_search_actually_hits_the_provider_endpoint(encoding: FaceEncoding) -> None:
    transport = FakeTransport(ok(WIKIMEDIA_PAYLOAD))
    make_service(transport, search_terms="sample portrait").search_for_encoding(encoding)

    assert len(transport.requests) == 1
    request = transport.requests[0]
    assert request.url == "https://commons.wikimedia.org/w/api.php"
    assert request.params["gsrsearch"] == "sample portrait"
    assert request.params["gsrnamespace"] == "6"
    # Wikimedia's policy: identify the client on every request.
    assert request.headers["User-Agent"].startswith("FaceTrace/")


@pytest.mark.parametrize(
    ("provider_name", "payload", "settings_overrides", "expected_url"),
    [
        (
            "wikimedia-commons",
            WIKIMEDIA_PAYLOAD,
            {"search_terms": "sample portrait"},
            COMMONS_PAGE_URL,
        ),
        (
            "google-custom-search",
            GOOGLE_PAYLOAD,
            {
                "search_terms": "sample portrait",
                "search_api_key": "test-key-value",
                "search_engine_id": "cx-000",
            },
            GOOGLE_ARTICLE_URL,
        ),
        (
            "serpapi-google-lens",
            SERPAPI_PAYLOAD,
            {"search_api_key": "test-key-value", "search_image_url": PUBLIC_IMAGE_URL},
            GOOGLE_ARTICLE_URL,
        ),
    ],
)
def test_every_registered_provider_performs_a_real_request(
    encoding: FaceEncoding,
    provider_name: str,
    payload: dict[str, Any],
    settings_overrides: dict[str, Any],
    expected_url: str,
) -> None:
    """No adapter may answer from a canned result: each must issue one request."""
    transport = FakeTransport(ok(payload))
    service = make_service(
        transport, search_provider=provider_name, **settings_overrides
    )
    outcome = service.search_for_encoding(encoding)

    assert len(transport.requests) == 1
    assert transport.requests[0].url.startswith("https://")
    assert outcome.status is SearchStatus.SUCCESS
    assert outcome.results[0].url == expected_url
    assert outcome.provider == provider_name


def test_reverse_image_search_sends_only_the_operator_supplied_url(
    encoding: FaceEncoding,
) -> None:
    """The local image file is never uploaded; only the given URL is sent."""
    transport = FakeTransport(ok(SERPAPI_PAYLOAD))
    service = make_service(
        transport,
        search_provider="serpapi-google-lens",
        search_api_key="test-key-value",
        search_image_url=PUBLIC_IMAGE_URL,
    )
    outcome = service.search_for_encoding(encoding)

    assert transport.sent_params["url"] == PUBLIC_IMAGE_URL
    assert transport.sent_params["engine"] == "google_lens"
    # The encoding reference is audit metadata; it is never transmitted.
    assert encoding.encoding_reference not in json.dumps(transport.sent_params)
    assert outcome.query is not None
    assert outcome.query.image_reference == encoding.encoding_reference


def test_serpapi_upload_then_lens_search_uses_real_image_id(
    encoding: FaceEncoding, tmp_path: Any
) -> None:
    image = tmp_path / "face.png"
    image.write_bytes(b"real-image-bytes")
    transport = FakeTransport(ok(SERPAPI_UPLOAD_PAYLOAD), ok(SERPAPI_PAYLOAD))
    service = make_service(
        transport,
        search_provider="serpapi-google-lens",
        search_api_key="test-key-value",
    )

    outcome = service.search_uploaded_image(image, encoding)

    assert outcome.status is SearchStatus.SUCCESS
    assert len(transport.requests) == 2
    assert transport.requests[0].url == "https://serpapi.com/image"
    assert transport.requests[1].params["image_id"] == "img-upload-123"
    assert transport.requests[1].params["type"] == "all"
    assert outcome.results[0].url == GOOGLE_ARTICLE_URL


def test_serpapi_upload_missing_key_is_controlled(tmp_path: Any) -> None:
    provider = SerpApiLensProvider(settings=Settings(search_provider="serpapi-google-lens"))
    assert provider.missing_configuration() == ("SERPAPI_API_KEY",)


def test_providers_satisfy_the_step_1_search_protocol() -> None:
    for provider_cls in PROVIDERS.values():
        provider = provider_cls(settings=Settings())
        assert isinstance(provider, SearchProvider)


def test_registry_and_aliases_resolve_to_documented_providers() -> None:
    assert PROVIDER_NAMES == (
        "google-custom-search",
        "serpapi-google-lens",
        "wikimedia-commons",
    )
    assert resolve_provider_name("") == "wikimedia-commons"
    assert resolve_provider_name("Google") == "google-custom-search"
    assert resolve_provider_name("lens") == "serpapi-google-lens"

    with pytest.raises(SearchNotConfiguredError, match="unknown SEARCH_PROVIDER"):
        resolve_provider_name("definitely-not-a-provider")


def test_application_default_transport_is_httpx() -> None:
    """The shipped code path uses real HTTP; the fake exists only in tests."""
    provider = build_search_provider(Settings())
    assert isinstance(provider, WikimediaCommonsProvider)
    assert isinstance(provider.client.transport, HttpxTransport)
    assert not isinstance(provider.client.transport, FakeTransport)


# --------------------------------------------------------------------------
# 2. Empty search result
# --------------------------------------------------------------------------


def test_empty_provider_response_is_no_results_not_a_match(
    encoding: FaceEncoding,
) -> None:
    transport = FakeTransport(ok(WIKIMEDIA_EMPTY_PAYLOAD))
    outcome = make_service(transport).search_for_encoding(encoding)

    assert outcome.status is SearchStatus.NO_RESULTS
    assert outcome.succeeded is False
    assert outcome.results == ()
    assert outcome.result_count == 0
    assert outcome.error_code is ErrorCode.NO_SEARCH_RESULTS
    assert outcome.attempts == 1
    # An empty search must never be dressed up as a candidate.
    assert "match" not in outcome.error or "not a match" in outcome.error


def test_no_results_when_the_payload_has_no_items(encoding: FaceEncoding) -> None:
    for payload in ({}, {"query": {}}, {"query": {"pages": []}}):
        transport = FakeTransport(ok(payload))
        outcome = make_service(transport).search_for_encoding(encoding)
        assert outcome.status is SearchStatus.NO_RESULTS, payload
        assert outcome.results == ()


def test_google_response_without_items_is_no_results(encoding: FaceEncoding) -> None:
    transport = FakeTransport(ok({"searchInformation": {"totalResults": "0"}}))
    outcome = make_service(
        transport,
        search_provider="google-custom-search",
        search_api_key="test-key-value",
        search_engine_id="cx-000",
    ).search_for_encoding(encoding)

    assert outcome.status is SearchStatus.NO_RESULTS
    assert outcome.results == ()


# --------------------------------------------------------------------------
# 3. External / search failure
# --------------------------------------------------------------------------


def test_transport_failure_is_a_structured_error(encoding: FaceEncoding) -> None:
    sleeper = RecordingSleeper()
    transport = FakeTransport(
        SearchTransportError("connection reset", retryable=True),
        SearchTransportError("connection reset", retryable=True),
        SearchTransportError("connection reset", retryable=True),
    )
    outcome = make_service(transport, sleeper=sleeper).search_for_encoding(encoding)

    assert outcome.status is SearchStatus.ERROR
    assert outcome.results == ()
    assert outcome.error_code is ErrorCode.SEARCH_FAILED
    assert outcome.retryable is True
    assert "connection reset" in outcome.error
    # max_retries=2 means three attempts, with a wait between each.
    assert len(transport.requests) == 3
    assert outcome.attempts == 3
    assert sleeper.delays == [0.5, 1.0]


def test_non_retryable_transport_failure_is_not_retried(encoding: FaceEncoding) -> None:
    transport = FakeTransport(SearchTransportError("bad request", retryable=False))
    outcome = make_service(transport).search_for_encoding(encoding)

    assert outcome.status is SearchStatus.ERROR
    assert outcome.retryable is False
    assert len(transport.requests) == 1
    assert outcome.attempts == 1


def test_server_error_is_retried_then_reported(encoding: FaceEncoding) -> None:
    transport = FakeTransport(
        status(503, {"error": {"info": "service unavailable"}}),
        status(503, {"error": {"info": "service unavailable"}}),
        status(503, {"error": {"info": "service unavailable"}}),
    )
    outcome = make_service(transport).search_for_encoding(encoding)

    assert outcome.status is SearchStatus.ERROR
    assert outcome.error_code is ErrorCode.SEARCH_FAILED
    assert outcome.retryable is True
    assert "HTTP 503" in outcome.error
    assert "service unavailable" in outcome.error
    assert len(transport.requests) == 3


def test_transient_error_recovers_on_retry(encoding: FaceEncoding) -> None:
    transport = FakeTransport(status(429, {}), ok(WIKIMEDIA_PAYLOAD))
    outcome = make_service(transport).search_for_encoding(encoding)

    assert outcome.status is SearchStatus.SUCCESS
    assert outcome.attempts == 2
    assert len(transport.requests) == 2


def test_rejected_credentials_report_not_configured(encoding: FaceEncoding) -> None:
    transport = FakeTransport(
        status(403, {"error": {"message": "API key not valid"}}),
    )
    outcome = make_service(
        transport,
        search_provider="google-custom-search",
        search_api_key="test-key-value",
        search_engine_id="cx-000",
    ).search_for_encoding(encoding)

    assert outcome.status is SearchStatus.ERROR
    assert outcome.error_code is ErrorCode.SEARCH_NOT_CONFIGURED
    assert outcome.retryable is False
    assert "HTTP 403" in outcome.error
    assert len(transport.requests) == 1, "an auth failure must not be retried"


def test_missing_credentials_fail_before_any_request(encoding: FaceEncoding) -> None:
    transport = FakeTransport()
    outcome = make_service(
        transport, search_provider="google-custom-search"
    ).search_for_encoding(encoding)

    assert outcome.status is SearchStatus.ERROR
    assert outcome.error_code is ErrorCode.SEARCH_NOT_CONFIGURED
    assert "SEARCH_API_KEY" in outcome.error
    assert "SEARCH_ENGINE_ID" in outcome.error
    assert transport.requests == [], "no request may be sent without credentials"


def test_wikimedia_requires_a_contactable_user_agent(encoding: FaceEncoding) -> None:
    """Wikimedia's robot policy is enforced up front, not discovered as a 403.

    The keyless provider needs no credentials, but the API rejects an agent with
    no contact. Reporting that as missing configuration keeps the operator's
    fix obvious and avoids sending a request the policy forbids.
    """
    transport = FakeTransport()
    outcome = make_service(
        transport, search_user_agent="FaceTrace/0.1 (no contact here)"
    ).search_for_encoding(encoding)

    assert outcome.status is SearchStatus.ERROR
    assert outcome.error_code is ErrorCode.SEARCH_NOT_CONFIGURED
    assert "SEARCH_USER_AGENT" in outcome.error
    assert "robot policy" in outcome.error
    assert transport.requests == [], "no request may be sent against the policy"


def test_provider_rejection_with_a_plain_text_body_is_still_explained(
    encoding: FaceEncoding,
) -> None:
    """A non-JSON rejection body is surfaced, sanitised — Wikimedia sends text."""
    policy_text = (
        "Please respect our robot policy https://w.wiki/4wJS when crawling us."
    )
    transport = FakeTransport(status(403, text=policy_text))
    outcome = make_service(transport).search_for_encoding(encoding)

    assert outcome.status is SearchStatus.ERROR
    assert outcome.error_code is ErrorCode.SEARCH_NOT_CONFIGURED
    assert "HTTP 403" in outcome.error
    assert "https://w.wiki/4wJS" in outcome.error
    # The keyless provider must not be blamed for an API key it never uses.
    assert "API key" not in outcome.error


def test_non_json_body_is_a_structured_error(encoding: FaceEncoding) -> None:
    transport = FakeTransport(HttpResponse(status_code=200, text="<html>nope</html>"))
    outcome = make_service(transport).search_for_encoding(encoding)

    assert outcome.status is SearchStatus.ERROR
    assert outcome.error_code is ErrorCode.SEARCH_FAILED
    assert outcome.results == ()
    assert "non-JSON" in outcome.error
    # The untrusted body itself is not echoed back.
    assert "<html>" not in outcome.error


def test_provider_error_payload_is_reported(encoding: FaceEncoding) -> None:
    transport = FakeTransport(
        ok({"error": {"code": "badvalue", "info": "Unrecognized value for parameter."}})
    )
    outcome = make_service(transport).search_for_encoding(encoding)

    assert outcome.status is SearchStatus.ERROR
    assert outcome.error_code is ErrorCode.SEARCH_FAILED
    assert "badvalue" in outcome.error
    assert outcome.results == ()


def test_text_provider_refuses_to_invent_query_terms(encoding: FaceEncoding) -> None:
    transport = FakeTransport()
    outcome = make_service(transport, search_terms="").search_for_encoding(encoding)

    assert outcome.status is SearchStatus.ERROR
    assert outcome.error_code is ErrorCode.SEARCH_QUERY_INVALID
    assert outcome.results == ()
    assert transport.requests == [], "a blank search must not be sent"


def test_reverse_image_provider_requires_a_public_url(encoding: FaceEncoding) -> None:
    transport = FakeTransport()
    outcome = make_service(
        transport,
        search_provider="serpapi-google-lens",
        search_api_key="test-key-value",
        search_image_url="",
    ).search_for_encoding(encoding)

    assert outcome.error_code is ErrorCode.SEARCH_QUERY_INVALID
    assert transport.requests == []


def test_unexpected_provider_shape_does_not_crash(encoding: FaceEncoding) -> None:
    transport = FakeTransport(ok(["not", "an", "object"]))
    outcome = make_service(transport).search_for_encoding(encoding)

    assert outcome.status is SearchStatus.ERROR
    assert outcome.error_code is ErrorCode.SEARCH_FAILED
    assert outcome.results == ()


# --------------------------------------------------------------------------
# 4. Result URL preservation
# --------------------------------------------------------------------------


def test_result_urls_are_preserved_exactly(encoding: FaceEncoding) -> None:
    transport = FakeTransport(ok(WIKIMEDIA_PAYLOAD))
    outcome = make_service(transport).search_for_encoding(encoding)

    pages = WIKIMEDIA_PAYLOAD["query"]["pages"]
    expected_pages = [page["imageinfo"][0]["descriptionurl"] for page in pages]
    expected_files = [page["imageinfo"][0]["url"] for page in pages]

    assert [result.url for result in outcome.results] == expected_pages
    assert [result.image_url for result in outcome.results] == expected_files


def test_google_result_keeps_both_the_page_and_the_image_url(
    encoding: FaceEncoding,
) -> None:
    transport = FakeTransport(ok(GOOGLE_PAYLOAD))
    outcome = make_service(
        transport,
        search_provider="google-custom-search",
        search_api_key="test-key-value",
        search_engine_id="cx-000",
    ).search_for_encoding(encoding)

    result = outcome.results[0]
    assert result.url == GOOGLE_ARTICLE_URL, "the real source page must survive"
    assert result.image_url == GOOGLE_IMAGE_URL
    assert result.source == "news.example.org"


def test_result_ids_are_derived_from_the_real_url(encoding: FaceEncoding) -> None:
    transport = FakeTransport(ok(WIKIMEDIA_PAYLOAD))
    outcome = make_service(transport).search_for_encoding(encoding)

    ids = [result.result_id for result in outcome.results]
    assert len(set(ids)) == len(ids)
    assert all(rid.startswith("wikimedia-commons:") for rid in ids)


def test_duplicate_urls_are_collapsed(encoding: FaceEncoding) -> None:
    page = WIKIMEDIA_PAYLOAD["query"]["pages"][0]
    transport = FakeTransport(ok({"query": {"pages": [page, dict(page)]}}))
    outcome = make_service(transport).search_for_encoding(encoding)

    assert outcome.result_count == 1


# --------------------------------------------------------------------------
# 5. Result data is not fabricated
# --------------------------------------------------------------------------


def test_every_result_field_traces_back_to_the_payload(encoding: FaceEncoding) -> None:
    raw = json.dumps(WIKIMEDIA_PAYLOAD)
    transport = FakeTransport(ok(WIKIMEDIA_PAYLOAD))
    outcome = make_service(transport).search_for_encoding(encoding)

    for result in outcome.results:
        assert result.url in raw
        assert result.image_url is not None and result.image_url in raw
        assert result.published_at in raw
        for word in result.title.split():
            assert word in raw
        for value in result.metadata.values():
            if value not in ("wikimedia-commons",):  # provider label, recorded locally
                assert value in raw or value.replace(" ", "") in raw


def test_search_stage_never_assigns_a_match_score(encoding: FaceEncoding) -> None:
    """Deciding a candidate depicts the person is the matching stage's job."""
    transport = FakeTransport(ok(WIKIMEDIA_PAYLOAD))
    outcome = make_service(transport).search_for_encoding(encoding)

    assert all(result.match_score == 0.0 for result in outcome.results)


def test_candidate_without_a_usable_url_is_dropped_not_invented(
    encoding: FaceEncoding,
) -> None:
    payload = {
        "query": {
            "pages": [
                {"pageid": 1, "title": "File:No imageinfo.jpg"},
                {
                    "pageid": 2,
                    "title": "File:No description url.jpg",
                    "imageinfo": [{"url": COMMONS_FILE_URL, "mime": "image/jpeg"}],
                },
                {
                    "pageid": 3,
                    "title": "File:Hostile scheme.jpg",
                    "imageinfo": [
                        {
                            "descriptionurl": "javascript:alert(1)",
                            "url": "data:image/png;base64,AAAA",
                        }
                    ],
                },
                WIKIMEDIA_PAYLOAD["query"]["pages"][0],
            ]
        }
    }
    transport = FakeTransport(ok(payload))
    outcome = make_service(transport).search_for_encoding(encoding)

    assert outcome.result_count == 1
    assert outcome.results[0].url == COMMONS_PAGE_URL


def test_all_candidates_unusable_yields_no_results(encoding: FaceEncoding) -> None:
    payload = {
        "query": {
            "pages": [
                {
                    "pageid": 3,
                    "title": "File:Hostile scheme.jpg",
                    "imageinfo": [{"descriptionurl": "javascript:alert(1)"}],
                }
            ]
        }
    }
    transport = FakeTransport(ok(payload))
    outcome = make_service(transport).search_for_encoding(encoding)

    assert outcome.status is SearchStatus.NO_RESULTS
    assert outcome.results == ()


def test_max_results_caps_what_is_returned(encoding: FaceEncoding) -> None:
    transport = FakeTransport(ok(WIKIMEDIA_PAYLOAD))
    outcome = make_service(transport, search_max_results=1).search_for_encoding(encoding)

    assert outcome.result_count == 1
    assert transport.sent_params["gsrlimit"] == "1"


def test_untrusted_strings_are_sanitised() -> None:
    assert sanitize_text("<b>hi</b>  there\n") == "hi there"
    assert sanitize_text("bad\x00control\x07chars") == "badcontrolchars"
    assert sanitize_text("&amp;amp") == "&amp"
    assert sanitize_text("x" * 50, 10).endswith("…")
    assert len(sanitize_text("x" * 50, 10)) == 10
    assert sanitize_text({"nested": "object"}) == ""
    assert sanitize_text(None) == ""


def test_only_http_urls_are_accepted() -> None:
    assert sanitize_url("https://example.test/a?b=c") == "https://example.test/a?b=c"
    assert sanitize_url("http://example.test/a") == "http://example.test/a"
    assert sanitize_url("javascript:alert(1)") == ""
    assert sanitize_url("data:text/html,<script>") == ""
    assert sanitize_url("file:///etc/passwd") == ""
    assert sanitize_url("/relative/path") == ""
    assert sanitize_url("https://example.test/" + "x" * 3000) == ""
    assert sanitize_url(None) == ""


def test_query_depends_on_the_submitted_image(encoding: FaceEncoding) -> None:
    provider = WikimediaCommonsProvider(
        settings=Settings(search_terms="sample portrait")
    )
    other = FaceEncoding(
        input_id="img-0002",
        model=encoding.model,
        encoding_reference="b" * 64,
        dimension=512,
    )

    first = provider.build_query(encoding)
    again = provider.build_query(encoding)
    second = provider.build_query(other)

    assert first.query_id == again.query_id, "the same input must be reproducible"
    assert first.query_id != second.query_id, "a different face must query differently"
    assert first.image_reference == encoding.encoding_reference
    assert first.terms == ("sample portrait",)
    assert first.provider == "wikimedia-commons"


def test_operator_terms_are_split_trimmed_and_deduplicated() -> None:
    provider = WikimediaCommonsProvider(
        settings=Settings(search_terms="  alpha , beta ,, alpha ,gamma ")
    )
    query = provider.build_query(
        FaceEncoding(input_id="i", model="m", encoding_reference="r", dimension=512)
    )

    assert query.terms == ("alpha", "beta", "gamma")


def test_search_outcome_is_json_serialisable(encoding: FaceEncoding) -> None:
    """The CLI's --json mode must not need a custom encoder."""
    transport = FakeTransport(ok(WIKIMEDIA_PAYLOAD))
    outcome = make_service(transport).search_for_encoding(encoding)

    payload = json.loads(json.dumps(asdict(outcome)))
    assert payload["status"] == "SUCCESS"
    assert payload["results"][0]["url"] == COMMONS_PAGE_URL


def test_search_result_snippet_aliases_the_hashed_text_field() -> None:
    result = SearchResult(
        result_id="r1", source="example", url="https://example.test/p/1", text="caption"
    )
    assert result.snippet == result.text == "caption"


# --------------------------------------------------------------------------
# 6. API credentials are never logged
# --------------------------------------------------------------------------

SECRET_KEY = "AIzaSyTESTONLY-not-a-real-key-0123456789"


def _google_service(transport: FakeTransport, **overrides: Any) -> Any:
    return make_service(
        transport,
        search_provider="google-custom-search",
        search_api_key=SECRET_KEY,
        search_engine_id="cx-000",
        **overrides,
    )


def test_api_key_is_sent_but_never_logged(
    encoding: FaceEncoding, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    transport = FakeTransport(ok(GOOGLE_PAYLOAD))
    outcome = _google_service(transport).search_for_encoding(encoding)

    # The key really is used — the search is authenticated, not faked.
    assert transport.sent_params["key"] == SECRET_KEY
    assert outcome.status is SearchStatus.SUCCESS

    assert SECRET_KEY not in caplog.text
    assert SECRET_KEY not in repr(outcome)
    assert SECRET_KEY not in json.dumps(asdict(outcome), default=str)
    # The response object stores a redacted URL, so logging it cannot leak either.
    assert SECRET_KEY not in transport.served[0].url
    assert f"key={REDACTED}" in transport.served[0].url


def test_retry_log_lines_redact_the_api_key(
    encoding: FaceEncoding, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    transport = FakeTransport(status(500, {}), status(500, {}), status(500, {}))
    outcome = _google_service(transport).search_for_encoding(encoding)

    assert outcome.status is SearchStatus.ERROR
    assert len(transport.requests) == 3
    assert caplog.text, "the retry path must actually log"
    assert SECRET_KEY not in caplog.text
    assert f"key={REDACTED}" in caplog.text


def test_transport_error_messages_are_scrubbed(
    encoding: FaceEncoding, caplog: pytest.LogCaptureFixture
) -> None:
    """A third-party exception embedding the request URL must not leak the key."""
    caplog.set_level(logging.DEBUG)
    leaky = SearchTransportError(
        f"ConnectError for https://www.googleapis.com/customsearch/v1?key={SECRET_KEY}"
        f"&cx=cx-000 (token={SECRET_KEY})",
        retryable=False,
    )
    transport = FakeTransport(leaky)
    outcome = _google_service(transport).search_for_encoding(encoding)

    assert outcome.status is SearchStatus.ERROR
    assert SECRET_KEY not in outcome.error
    assert SECRET_KEY not in caplog.text
    assert REDACTED in outcome.error


def test_provider_error_text_is_scrubbed(encoding: FaceEncoding) -> None:
    """Even a provider that echoes the key back is not allowed to leak it."""
    transport = FakeTransport(
        status(400, {"error": {"message": f"invalid key {SECRET_KEY}"}})
    )
    outcome = _google_service(transport).search_for_encoding(encoding)

    assert outcome.status is SearchStatus.ERROR
    assert SECRET_KEY not in outcome.error
    assert REDACTED in outcome.error


def test_serpapi_key_is_scrubbed_from_errors(encoding: FaceEncoding) -> None:
    transport = FakeTransport(ok({"error": f"Invalid API key: {SECRET_KEY}"}))
    outcome = make_service(
        transport,
        search_provider="serpapi-google-lens",
        search_api_key=SECRET_KEY,
        search_image_url=PUBLIC_IMAGE_URL,
    ).search_for_encoding(encoding)

    assert outcome.error_code is ErrorCode.SEARCH_NOT_CONFIGURED
    assert SECRET_KEY not in outcome.error


def test_redaction_helpers() -> None:
    url = f"https://api.example.test/v1?q=cat&key={SECRET_KEY}&cx=abc"
    assert redact_url(url) == f"https://api.example.test/v1?q=cat&key={REDACTED}&cx=abc"
    assert SECRET_KEY not in redact_url(url)
    # A URL without a query is returned untouched.
    assert redact_url("https://api.example.test/v1") == "https://api.example.test/v1"

    assert redact_secrets(f"api_key={SECRET_KEY}") == f"api_key={REDACTED}"
    assert redact_secrets(f"bare {SECRET_KEY}", (SECRET_KEY,)) == f"bare {REDACTED}"
    # Very short values are ignored rather than mangling unrelated text.
    assert redact_secrets("abc def", ("ab",)) == "abc def"


def test_credential_headers_are_redacted_before_storage() -> None:
    from app.services.search_http import _safe_headers

    stored = _safe_headers(
        {"Authorization": f"Bearer {SECRET_KEY}", "Content-Type": "application/json"}
    )
    assert stored["Authorization"] == REDACTED
    assert stored["Content-Type"] == "application/json"

    client = HttpClient(transport=FakeTransport(), secrets=(SECRET_KEY,))
    sent = client.request_headers({"Authorization": f"Bearer {SECRET_KEY}"})
    assert sent["Authorization"] == f"Bearer {SECRET_KEY}"  # really sent
    assert SECRET_KEY not in client.scrub(str(sent))  # never logged


def test_health_section_reports_readiness_without_the_key() -> None:
    from app.main import _search_health

    section = _search_health(
        Settings(search_provider="google-custom-search", search_api_key=SECRET_KEY)
    )

    assert section["provider"] == "google-custom-search"
    assert section["ready"] is False
    assert section["missing_configuration"] == ["SEARCH_ENGINE_ID"]
    assert SECRET_KEY not in json.dumps(section)


def test_health_section_reports_an_unknown_provider() -> None:
    from app.main import _search_health

    section = _search_health(Settings(search_provider="nope"))
    assert section["ready"] is False
    assert "unknown SEARCH_PROVIDER" in str(section["error"])


# --------------------------------------------------------------------------
# Service façade contract
# --------------------------------------------------------------------------


def test_failure_outcomes_never_carry_results(encoding: FaceEncoding) -> None:
    """No caller can mistake a failure or an empty search for a candidate."""
    cases: Sequence[FakeTransport] = (
        FakeTransport(status(500, {}), status(500, {}), status(500, {})),
        FakeTransport(ok(WIKIMEDIA_EMPTY_PAYLOAD)),
        FakeTransport(SearchTransportError("boom", retryable=False)),
    )
    for transport in cases:
        outcome = make_service(transport).search_for_encoding(encoding)
        assert outcome.status is not SearchStatus.SUCCESS
        assert outcome.results == ()
        assert outcome.succeeded is False


def test_service_search_accepts_a_prebuilt_query() -> None:
    transport = FakeTransport(ok(WIKIMEDIA_PAYLOAD))
    service = make_service(transport)
    query = SearchQuery(
        query_id="deadbeefdeadbeef",
        terms=("sample portrait",),
        provider="wikimedia-commons",
        max_results=5,
    )

    outcome = service.search(query)
    assert outcome.status is SearchStatus.SUCCESS
    assert outcome.query is query


def test_service_describes_itself_for_health() -> None:
    service = make_service(FakeTransport())
    described = service.describe()

    assert described["provider"] == "wikimedia-commons"
    assert described["ready"] is True
    assert described["needs_terms"] is True
    assert described["needs_image_url"] is False


def test_unexpected_exception_is_reported_not_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)

    class ExplodingProvider(WikimediaCommonsProvider):
        def search(self, query: SearchQuery) -> tuple[SearchResult, ...]:
            raise RuntimeError(f"kaboom {SECRET_KEY}")

    from app.services.search_web import SearchService

    settings = Settings(search_terms="sample portrait", search_api_key=SECRET_KEY)
    service = SearchService(
        settings=settings, provider=ExplodingProvider(settings=settings)
    )
    outcome = service.search_for_encoding(
        FaceEncoding(input_id="i", model="m", encoding_reference="r", dimension=512)
    )

    assert outcome.status is SearchStatus.ERROR
    assert outcome.error_code is ErrorCode.SEARCH_FAILED
    assert "RuntimeError" in outcome.error
    assert SECRET_KEY not in outcome.error
    assert SECRET_KEY not in caplog.text


def test_service_errors_map_to_pipeline_error_codes() -> None:
    assert SearchNotConfiguredError("x").code == ErrorCode.SEARCH_NOT_CONFIGURED.value
    assert SearchTransportError("x").code == ErrorCode.SEARCH_FAILED.value
    assert isinstance(SearchNotConfiguredError("x"), ServiceError)


def test_changing_the_setting_swaps_the_adapter() -> None:
    """The interface is replaceable: one setting selects the whole adapter."""
    for name, expected in PROVIDERS.items():
        provider = build_search_provider(Settings(search_provider=name))
        assert type(provider) is expected
        assert provider.name == name

    assert SerpApiLensProvider(settings=Settings()).needs_image_url is True
    assert GoogleCustomSearchProvider(settings=Settings()).needs_terms is True
