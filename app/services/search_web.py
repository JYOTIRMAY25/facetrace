"""Genuine web/social search providers (STEP 3).

Strategy
--------
A face embedding cannot be handed to a web search engine: search APIs accept
text or an image URL, not a 512-dimensional vector. FaceTrace therefore runs the
search that each provider genuinely supports and keeps the biometric comparison
local, exactly as ``IMPLEMENTATION_PLAN.md`` Phase 7 prescribes::

    Search results -> drop invalid results -> candidate images/posts
    -> face matching (STEP 4) -> similarity ranking

So this stage answers only "what real, public content is out there that could be
worth comparing?" Deciding whether any of it depicts the submitted face is the
matching stage's job, and nothing here claims a result belongs to that person.

Three adapters ship, all behind :class:`app.services.search.SearchProvider`:

``wikimedia-commons``
    Keyless. Wikimedia's public Action API returns real Commons file pages plus
    their real image URLs, so the pipeline is demonstrable end-to-end with no
    credentials at all. Text-driven: it needs operator-supplied terms, and a
    ``SEARCH_USER_AGENT`` naming a contact, as Wikimedia's robot policy requires.
``google-custom-search``
    Google Programmable Search JSON API — the ``SEARCH_API_KEY`` /
    ``SEARCH_ENGINE_ID`` pair named in the project specification. Text-driven,
    image search mode, so every candidate carries an image URL.
``serpapi-google-lens``
    Reverse *image* search: the genuinely image-driven path. It needs a public
    image URL that the operator supplies (``SEARCH_IMAGE_URL``). FaceTrace never
    uploads the local face image anywhere — that promise from STEP 2 stands.

Rules honoured here
-------------------
* The search really executes; there are no canned payloads in this module.
* Every :class:`SearchResult` field is copied from the provider response or
  recorded locally. Nothing is invented, and a result with no usable URL is
  dropped rather than patched up.
* The query varies with the image: ``query_id`` is derived from the encoding
  reference, so a run is auditable back to the exact face that triggered it.
* Zero candidates is a legitimate ``NO_RESULTS`` answer, never a match.
* Provider content is untrusted: URLs must be ``http(s)``, HTML is stripped,
  control characters are removed, and every string is length-capped.
* Credentials live only in the environment and are scrubbed from every log line
  and error message by :mod:`app.services.search_http`.
"""

from __future__ import annotations

import hashlib
import html
import logging
import re
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional, Sequence
from urllib.parse import urlsplit

from ..config import Settings
from ..models.pipeline import (
    ErrorCode,
    FaceEncoding,
    SearchOutcome,
    SearchQuery,
    SearchResult,
    SearchStatus,
)
from . import ServiceError
from .search_http import (
    HttpClient,
    HttpResponse,
    HttpTransport,
    HttpxTransport,
    SearchTransportError,
    clean_control_chars,
    redact_secrets,
)

__all__ = [
    "ALLOWED_URL_SCHEMES",
    "PROVIDERS",
    "PROVIDER_NAMES",
    "GoogleCustomSearchProvider",
    "SearchNotConfiguredError",
    "SearchQueryError",
    "SearchService",
    "SerpApiLensProvider",
    "WikimediaCommonsProvider",
    "build_http_client",
    "build_search_provider",
    "build_search_service",
    "sanitize_text",
    "sanitize_url",
]

LOGGER = logging.getLogger(__name__)

#: Only these schemes are accepted from a provider. Blocks javascript:, data:,
#: file: and friends before an untrusted URL reaches the CLI or a browser.
ALLOWED_URL_SCHEMES: frozenset[str] = frozenset({"http", "https"})

MAX_TITLE_CHARS = 300
MAX_TEXT_CHARS = 1000
MAX_URL_CHARS = 2048
MAX_METADATA_VALUE_CHARS = 200
MAX_TERM_CHARS = 200
MAX_TERMS = 12

_TAG_RE = re.compile(r"<[^>]*>")
_WHITESPACE_RE = re.compile(r"\s+")


def _image_media_type(path: Path) -> str:
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "application/octet-stream")

#: A contact — an address or a URL — inside the ``User-Agent``. Wikimedia's robot
#: policy rejects an anonymous agent with HTTP 403, so this is checked up front
#: rather than discovered as an opaque rejection mid-run.
_CONTACT_RE = re.compile(r"(?i)(@|https?://)")


class SearchNotConfiguredError(ServiceError):
    """The chosen provider is missing settings it needs. Not a provider fault."""

    code = ErrorCode.SEARCH_NOT_CONFIGURED.value
    retryable = False


class SearchQueryError(ServiceError):
    """No usable query could be built. The stage refuses to invent terms."""

    code = ErrorCode.SEARCH_QUERY_INVALID.value
    retryable = False


# --------------------------------------------------------------------------
# Sanitisation of untrusted provider content
# --------------------------------------------------------------------------


def _now_iso() -> str:
    """Current UTC instant, second precision, ISO-8601 with a ``Z`` suffix."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def sanitize_text(value: Any, limit: int = MAX_TEXT_CHARS) -> str:
    """Normalise one untrusted provider string.

    Unescapes entities, strips HTML tags (Wikimedia's ``extmetadata`` fields are
    HTML fragments), removes control characters, collapses whitespace, and caps
    the length. Content is preserved, not rewritten — truncation is marked with
    an ellipsis so a shortened snippet is never mistaken for the full text.
    """
    if value is None or isinstance(value, (dict, list, tuple)):
        return ""
    text = value if isinstance(value, str) else str(value)
    text = html.unescape(_TAG_RE.sub(" ", text))
    text = _WHITESPACE_RE.sub(" ", clean_control_chars(text)).strip()
    if limit > 0 and len(text) > limit:
        return text[: max(0, limit - 1)].rstrip() + "…"
    return text


def sanitize_url(value: Any) -> str:
    """Return ``value`` if it is a plain absolute http(s) URL, else ``""``.

    Rejecting instead of repairing matters: a fabricated or coerced URL would
    break the "preserve the actual source URL" requirement.
    """
    if not isinstance(value, str):
        return ""
    candidate = clean_control_chars(value).strip()
    if not candidate or len(candidate) > MAX_URL_CHARS:
        return ""
    parts = urlsplit(candidate)
    if parts.scheme.lower() not in ALLOWED_URL_SCHEMES or not parts.netloc:
        return ""
    return candidate


def _host_of(url: str) -> str:
    return urlsplit(url).netloc.lower()


def _sanitize_metadata(pairs: Iterable[tuple[str, Any]]) -> dict[str, str]:
    """Keep only non-empty, sanitised metadata values."""
    metadata: dict[str, str] = {}
    for name, value in pairs:
        cleaned = sanitize_text(value, MAX_METADATA_VALUE_CHARS)
        if cleaned:
            metadata[name] = cleaned
    return metadata


def _extmetadata(info: Mapping[str, Any], name: str) -> str:
    """Read one Wikimedia ``extmetadata`` entry, which nests under ``value``."""
    block = info.get("extmetadata")
    if not isinstance(block, Mapping):
        return ""
    entry = block.get(name)
    if isinstance(entry, Mapping):
        return sanitize_text(entry.get("value"), MAX_TEXT_CHARS)
    return sanitize_text(entry, MAX_TEXT_CHARS)


def _result_id(provider: str, url: str, index: int) -> str:
    """Stable id derived from the *real* URL, so it cannot be a stand-in for one."""
    digest = hashlib.sha256(f"{provider}|{index}|{url}".encode("utf-8")).hexdigest()
    return f"{provider}:{digest[:16]}"


def _has_contact(user_agent: str) -> bool:
    """Whether ``user_agent`` names a contact, as Wikimedia's robot policy requires."""
    return bool(_CONTACT_RE.search(user_agent or ""))


def _split_terms(raw: str) -> tuple[str, ...]:
    """Split operator-supplied terms on commas, then trim, cap, and de-duplicate."""
    terms: list[str] = []
    for chunk in str(raw or "").split(","):
        term = _WHITESPACE_RE.sub(" ", clean_control_chars(chunk)).strip()
        if term and term not in terms:
            terms.append(term[:MAX_TERM_CHARS])
        if len(terms) >= MAX_TERMS:
            break
    return tuple(terms)


# --------------------------------------------------------------------------
# Provider base
# --------------------------------------------------------------------------


@dataclass
class _HttpSearchProvider:
    """Shared plumbing for the HTTP-backed providers.

    Subclasses supply :attr:`name`, :attr:`endpoint`, request parameters, and a
    response parser. This base owns configuration checks, query construction,
    status handling, and the no-fabrication guarantees.
    """

    #: Provider identifier, recorded on every result.
    name: str = "http-search-provider"
    #: Absolute API endpoint. Never contains a credential.
    endpoint: str = ""
    #: Settings a live run needs, reported by :meth:`missing_configuration`.
    required_settings: tuple[str, ...] = ()
    #: Whether the provider is text-driven and therefore needs operator terms.
    needs_terms: bool = True
    #: Whether the provider is a reverse-image search needing a public image URL.
    needs_image_url: bool = False

    #: Plain-class attributes, not dataclass fields: what the operator should do
    #: when the provider is unconfigured, and what to check when it says no.
    configuration_hint = ""
    auth_hint = "check the API key, its quota, and the engine id"

    settings: Settings = field(default_factory=Settings)
    client: HttpClient = field(default_factory=lambda: HttpClient(HttpxTransport()))
    #: HTTP attempts made by the most recent :meth:`search`, retries included.
    last_attempts: int = 0

    # -- configuration ----------------------------------------------------

    def missing_configuration(self) -> tuple[str, ...]:
        """Environment variables this provider still needs. Empty when ready."""
        return ()

    def describe(self) -> dict[str, object]:
        """Secret-free provider summary for ``--health``."""
        return {
            "provider": self.name,
            "endpoint": self.endpoint,
            "needs_terms": self.needs_terms,
            "needs_image_url": self.needs_image_url,
            "required_settings": list(self.required_settings),
            "missing_configuration": list(self.missing_configuration()),
            "configuration_hint": self.configuration_hint,
            "ready": not self.missing_configuration(),
        }

    # -- query ------------------------------------------------------------

    def build_query(self, encoding: FaceEncoding) -> SearchQuery:
        """Derive the query for ``encoding`` from settings the operator supplied.

        ``query_id`` mixes the encoding reference in, so it changes with the
        image; the reference itself is an opaque local handle and is never sent
        to the provider. Terms are never invented: a text provider with no terms
        raises :class:`SearchQueryError` instead of running a blank search.
        """
        terms = _split_terms(self.settings.search_terms)
        image_url = sanitize_url(self.settings.search_image_url)
        max_results = max(1, int(self.settings.search_max_results or 10))

        if self.needs_terms and not terms:
            raise SearchQueryError(
                "no search terms supplied. This provider searches text, and the "
                "search stage will not invent query terms: set SEARCH_TERMS "
                "(or pass --terms) with the public identifier you are checking."
            )
        if self.needs_image_url and not image_url:
            raise SearchQueryError(
                "no public image URL supplied. This provider performs reverse "
                "image search and FaceTrace never uploads the local image: set "
                "SEARCH_IMAGE_URL (or pass --image-url) to an http(s) URL you "
                "are permitted to use."
            )

        reference = encoding.encoding_reference or ""
        seed = "|".join(
            (self.name, encoding.input_id or "", reference, "".join(terms), image_url)
        )
        return SearchQuery(
            query_id=hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16],
            terms=terms,
            image_reference=reference or None,
            provider=self.name,
            max_results=max_results,
            image_url=image_url or None,
        )

    # -- execution --------------------------------------------------------

    def search(self, query: SearchQuery) -> tuple[SearchResult, ...]:
        """Execute the search and return sanitised candidates.

        An empty tuple means the provider genuinely returned nothing usable.
        Failures raise a :class:`~app.services.ServiceError` subclass.
        """
        self.last_attempts = 0
        self.client.last_attempts = 0

        missing = self.missing_configuration()
        if missing:
            hint = f" {self.configuration_hint}" if self.configuration_hint else ""
            raise SearchNotConfiguredError(
                f"provider {self.name!r} is not configured; set: "
                f"{', '.join(missing)}.{hint}"
            )
        self._validate_query(query)

        try:
            response, attempts = self.client.get(self.endpoint, params=self._params(query))
        finally:
            # Report the real attempt count even when the request failed.
            self.last_attempts = self.client.last_attempts

        if not response.ok:
            raise self._status_error(response)

        payload = response.json()
        if not isinstance(payload, Mapping):
            raise SearchTransportError(
                f"provider {self.name!r} returned {type(payload).__name__} where an "
                "object was expected",
                retryable=False,
            )
        self._raise_payload_error(payload)

        retrieved_at = _now_iso()
        results: list[SearchResult] = []
        seen: set[str] = set()
        for index, raw in enumerate(self._iter_items(payload)):
            if len(results) >= query.max_results:
                break
            if not isinstance(raw, Mapping):
                continue
            result = self._parse_item(raw, index=index, retrieved_at=retrieved_at)
            if result is None or result.url in seen:
                continue
            seen.add(result.url)
            results.append(result)

        LOGGER.info(
            "search provider=%s query_id=%s attempts=%d candidates=%d",
            self.name,
            query.query_id,
            attempts,
            len(results),
        )
        return tuple(results)

    def _validate_query(self, query: SearchQuery) -> None:
        if self.needs_terms and not query.terms:
            raise SearchQueryError(
                f"provider {self.name!r} needs search terms, but the query has none"
            )
        if self.needs_image_url and not sanitize_url(query.image_url or ""):
            raise SearchQueryError(
                f"provider {self.name!r} needs a public http(s) image URL"
            )

    # -- subclass hooks ---------------------------------------------------

    def _params(self, query: SearchQuery) -> dict[str, Any]:
        raise NotImplementedError

    def _iter_items(self, payload: Mapping[str, Any]) -> Sequence[Any]:
        raise NotImplementedError

    def _parse_item(
        self, raw: Mapping[str, Any], *, index: int, retrieved_at: str
    ) -> Optional[SearchResult]:
        raise NotImplementedError

    def _raise_payload_error(self, payload: Mapping[str, Any]) -> None:
        """Turn an HTTP-200 provider error body into a structured failure."""
        return None

    # -- error mapping ----------------------------------------------------

    def _provider_message(self, response: HttpResponse) -> str:
        """Best-effort provider explanation, sanitised and credential-scrubbed."""
        try:
            payload = response.json()
        except ServiceError:
            # Not JSON. Wikimedia's 403 body is plain text naming the policy that
            # was breached, which is exactly what the operator needs to see.
            return self.client.scrub(sanitize_text(response.text, 200))
        if not isinstance(payload, Mapping):
            return ""
        error = payload.get("error")
        if isinstance(error, Mapping):
            message = error.get("message") or error.get("info") or ""
        else:
            message = error or payload.get("message") or ""
        return self.client.scrub(sanitize_text(message, MAX_TEXT_CHARS))

    def _status_error(self, response: HttpResponse) -> ServiceError:
        status = response.status_code
        detail = self._provider_message(response)
        suffix = f": {detail}" if detail else ""

        if status in (401, 403):
            return SearchNotConfiguredError(
                f"provider {self.name!r} rejected the request (HTTP {status}) — "
                f"{self.auth_hint}{suffix}"
            )
        retryable = response.retryable_status
        return SearchTransportError(
            f"provider {self.name!r} returned HTTP {status}{suffix}",
            retryable=retryable,
        )


# --------------------------------------------------------------------------
# Wikimedia Commons (keyless)
# --------------------------------------------------------------------------


@dataclass
class WikimediaCommonsProvider(_HttpSearchProvider):
    """Wikimedia Commons Action API — real file pages, no credentials.

    Searches namespace 6 (File:) and asks for ``imageinfo``, so every candidate
    carries both its real description-page URL and its real media URL.

    Wikimedia's robot policy requires a ``User-Agent`` that names a contact, and
    the API answers HTTP 403 without one (observed live with the default agent).
    That is a configuration problem, not a provider fault, so it is reported by
    :meth:`missing_configuration` before any request is made rather than being
    discovered as an opaque rejection. FaceTrace will not fake a contact to get
    past the filter — the operator supplies a real one in ``SEARCH_USER_AGENT``.
    """

    name: str = "wikimedia-commons"
    endpoint: str = "https://commons.wikimedia.org/w/api.php"
    required_settings: tuple[str, ...] = ("SEARCH_USER_AGENT",)
    needs_terms: bool = True
    needs_image_url: bool = False
    source_host: str = "commons.wikimedia.org"

    configuration_hint = (
        "Wikimedia's robot policy (https://w.wiki/4wJS) requires a User-Agent "
        "naming the tool and a contact, e.g. "
        "'FaceTrace/0.1 (HH Goa 2026 Task 3; you@example.org)'."
    )
    auth_hint = (
        "check SEARCH_USER_AGENT — Wikimedia requires a contactable agent — and "
        "the robot policy at https://w.wiki/4wJS"
    )

    def missing_configuration(self) -> tuple[str, ...]:
        """No credentials needed, but the agent must carry a contact."""
        if not _has_contact(self.settings.search_user_agent):
            return ("SEARCH_USER_AGENT",)
        return ()

    def _params(self, query: SearchQuery) -> dict[str, Any]:
        return {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "generator": "search",
            "gsrsearch": " ".join(query.terms),
            "gsrnamespace": "6",
            "gsrlimit": str(max(1, min(50, query.max_results))),
            "prop": "imageinfo",
            "iiprop": "url|timestamp|mime|extmetadata",
            "iiextmetadatafilter": (
                "ImageDescription|DateTimeOriginal|Artist|LicenseShortName|Credit"
            ),
        }

    def _raise_payload_error(self, payload: Mapping[str, Any]) -> None:
        error = payload.get("error")
        if isinstance(error, Mapping):
            code = sanitize_text(error.get("code"), 80)
            info = sanitize_text(error.get("info"), MAX_TEXT_CHARS)
            raise SearchTransportError(
                f"Wikimedia API error {code or 'unknown'}: {info}", retryable=False
            )

    def _iter_items(self, payload: Mapping[str, Any]) -> Sequence[Any]:
        block = payload.get("query")
        if not isinstance(block, Mapping):
            return ()
        pages = block.get("pages")
        if isinstance(pages, Mapping):  # formatversion=1 shape, keyed by page id
            return tuple(pages.values())
        if isinstance(pages, list):
            return tuple(pages)
        return ()

    def _parse_item(
        self, raw: Mapping[str, Any], *, index: int, retrieved_at: str
    ) -> Optional[SearchResult]:
        infos = raw.get("imageinfo")
        info = infos[0] if isinstance(infos, list) and infos else {}
        if not isinstance(info, Mapping):
            return None

        page_url = sanitize_url(info.get("descriptionurl"))
        image_url = sanitize_url(info.get("url"))
        if not page_url:
            # No genuine source URL means no usable candidate. Do not substitute one.
            return None

        published_at = _extmetadata(info, "DateTimeOriginal") or sanitize_text(
            info.get("timestamp"), 64
        )
        return SearchResult(
            result_id=_result_id(self.name, page_url, index),
            source=_host_of(page_url) or self.source_host,
            url=page_url,
            title=sanitize_text(raw.get("title"), MAX_TITLE_CHARS),
            text=_extmetadata(info, "ImageDescription"),
            image_url=image_url or None,
            metadata=_sanitize_metadata(
                (
                    ("provider", self.name),
                    ("mime", info.get("mime")),
                    ("license", _extmetadata(info, "LicenseShortName")),
                    ("credit", _extmetadata(info, "Credit")),
                    ("artist", _extmetadata(info, "Artist")),
                    ("page_id", raw.get("pageid")),
                )
            ),
            published_at=published_at,
            retrieved_at=retrieved_at,
        )


# --------------------------------------------------------------------------
# Google Programmable Search (SEARCH_API_KEY + SEARCH_ENGINE_ID)
# --------------------------------------------------------------------------


@dataclass
class GoogleCustomSearchProvider(_HttpSearchProvider):
    """Google Programmable Search JSON API, image mode.

    ``SEARCH_API_KEY`` is the API key and ``SEARCH_ENGINE_ID`` is the engine
    (``cx``) id. Both are read from the environment; neither is ever logged.
    Image mode is used so each candidate has an image URL for STEP 4 matching,
    with ``image.contextLink`` preserved as the real page the image came from.
    """

    name: str = "google-custom-search"
    endpoint: str = "https://www.googleapis.com/customsearch/v1"
    required_settings: tuple[str, ...] = ("SEARCH_API_KEY", "SEARCH_ENGINE_ID")
    needs_terms: bool = True
    needs_image_url: bool = False

    configuration_hint = (
        "SEARCH_API_KEY is a Google API key with the Custom Search API enabled; "
        "SEARCH_ENGINE_ID is the Programmable Search engine (cx) id."
    )

    def missing_configuration(self) -> tuple[str, ...]:
        missing: list[str] = []
        if not self.settings.search_api_key:
            missing.append("SEARCH_API_KEY")
        if not self.settings.search_engine_id:
            missing.append("SEARCH_ENGINE_ID")
        return tuple(missing)

    def _params(self, query: SearchQuery) -> dict[str, Any]:
        return {
            "key": self.settings.search_api_key,
            "cx": self.settings.search_engine_id,
            "q": " ".join(query.terms),
            "num": str(max(1, min(10, query.max_results))),
            "searchType": "image",
        }

    def _raise_payload_error(self, payload: Mapping[str, Any]) -> None:
        error = payload.get("error")
        if isinstance(error, Mapping):
            message = self.client.scrub(
                sanitize_text(error.get("message"), MAX_TEXT_CHARS)
            )
            raise SearchTransportError(
                f"Google Custom Search error: {message}", retryable=False
            )

    def _iter_items(self, payload: Mapping[str, Any]) -> Sequence[Any]:
        items = payload.get("items")
        return tuple(items) if isinstance(items, list) else ()

    def _parse_item(
        self, raw: Mapping[str, Any], *, index: int, retrieved_at: str
    ) -> Optional[SearchResult]:
        image = raw.get("image") if isinstance(raw.get("image"), Mapping) else {}
        image_url = sanitize_url(raw.get("link"))
        page_url = sanitize_url(image.get("contextLink")) or image_url
        if not page_url:
            return None

        return SearchResult(
            result_id=_result_id(self.name, page_url, index),
            source=sanitize_text(raw.get("displayLink"), 200) or _host_of(page_url),
            url=page_url,
            title=sanitize_text(raw.get("title"), MAX_TITLE_CHARS),
            text=sanitize_text(raw.get("snippet"), MAX_TEXT_CHARS),
            image_url=image_url or None,
            metadata=_sanitize_metadata(
                (
                    ("provider", self.name),
                    ("mime", raw.get("mime")),
                    ("file_format", raw.get("fileFormat")),
                    ("thumbnail_url", sanitize_url(image.get("thumbnailLink"))),
                )
            ),
            retrieved_at=retrieved_at,
        )


# --------------------------------------------------------------------------
# SerpAPI Google Lens (reverse image search)
# --------------------------------------------------------------------------


@dataclass
class SerpApiLensProvider(_HttpSearchProvider):
    """Reverse image search via SerpAPI's Google Lens engine.

    The only genuinely image-driven adapter. It takes a **public image URL the
    operator supplies** — FaceTrace does not upload the local face image, so this
    path is opt-in and requires the operator to have the right to publish that
    URL. ``SEARCH_API_KEY`` carries the SerpAPI key.
    """

    name: str = "serpapi-google-lens"
    endpoint: str = "https://serpapi.com/search"
    upload_endpoint: str = "https://serpapi.com/image"
    required_settings: tuple[str, ...] = ("SERPAPI_API_KEY",)
    needs_terms: bool = False
    needs_image_url: bool = True

    configuration_hint = (
        "SERPAPI_API_KEY is your SerpAPI private key."
    )
    auth_hint = "check the SerpAPI key and its remaining search quota"

    def missing_configuration(self) -> tuple[str, ...]:
        return () if self.settings.search_api_key else ("SERPAPI_API_KEY",)

    def search_uploaded_image(
        self, image_path: Path, encoding: FaceEncoding
    ) -> Sequence[SearchResult]:
        try:
            content = image_path.read_bytes()
            if not content:
                raise SearchTransportError("uploaded image is empty", retryable=False)
            upload_payload, _, _ = self.client.post_json(
                self.upload_endpoint,
                data={"api_key": self.settings.search_api_key},
                files={"image": (image_path.name, content, _image_media_type(image_path))},
            )
            self._raise_payload_error(upload_payload)
            image_id = upload_payload.get("image_id")
            if not isinstance(image_id, str) or not image_id:
                raise SearchTransportError("SerpAPI image upload returned no image_id.", retryable=False)
            payload, _, attempts = self.client.get_json(
                self.endpoint,
                params={
                    "engine": "google_lens",
                    "image_id": image_id,
                    "type": "all",
                    "api_key": self.settings.search_api_key,
                    "hl": "en",
                },
            )
            self.last_attempts = attempts + 1
            self._raise_payload_error(payload)
            retrieved_at = _now_iso()
            return tuple(
                parsed
                for index, raw in enumerate(self._iter_items(payload))
                if isinstance(raw, Mapping)
                for parsed in (self._parse_item(raw, index=index, retrieved_at=retrieved_at),)
                if parsed is not None
            )
        except SearchTransportError:
            raise
        except OSError as exc:
            raise SearchTransportError("unable to read uploaded image", retryable=False) from exc

    def _params(self, query: SearchQuery) -> dict[str, Any]:
        return {
            "engine": "google_lens",
            "url": query.image_url or "",
            "api_key": self.settings.search_api_key,
            "hl": "en",
        }

    def _raise_payload_error(self, payload: Mapping[str, Any]) -> None:
        error = payload.get("error")
        if not error:
            return
        message = self.client.scrub(sanitize_text(error, MAX_TEXT_CHARS))
        if "api key" in message.lower():
            raise SearchNotConfiguredError(f"SerpAPI rejected the key: {message}")
        raise SearchTransportError(f"SerpAPI error: {message}", retryable=False)

    def _iter_items(self, payload: Mapping[str, Any]) -> Sequence[Any]:
        matches = payload.get("visual_matches")
        return tuple(matches) if isinstance(matches, list) else ()

    def _parse_item(
        self, raw: Mapping[str, Any], *, index: int, retrieved_at: str
    ) -> Optional[SearchResult]:
        page_url = sanitize_url(raw.get("link"))
        if not page_url:
            return None
        image_url = sanitize_url(raw.get("image")) or sanitize_url(raw.get("thumbnail"))

        return SearchResult(
            result_id=_result_id(self.name, page_url, index),
            source=sanitize_text(raw.get("source"), 200) or _host_of(page_url),
            url=page_url,
            title=sanitize_text(raw.get("title"), MAX_TITLE_CHARS),
            text=sanitize_text(raw.get("snippet"), MAX_TEXT_CHARS),
            image_url=image_url or None,
            metadata=_sanitize_metadata(
                (
                    ("provider", self.name),
                    ("engine", "google_lens"),
                    ("position", raw.get("position")),
                )
            ),
            published_at=sanitize_text(raw.get("date"), 64),
            retrieved_at=retrieved_at,
        )


# --------------------------------------------------------------------------
# Registry and construction
# --------------------------------------------------------------------------

#: Registered provider adapters, keyed by ``SEARCH_PROVIDER`` value.
PROVIDERS: Mapping[str, type[_HttpSearchProvider]] = {
    "wikimedia-commons": WikimediaCommonsProvider,
    "google-custom-search": GoogleCustomSearchProvider,
    "serpapi-google-lens": SerpApiLensProvider,
}

PROVIDER_NAMES: tuple[str, ...] = tuple(sorted(PROVIDERS))

#: Convenience spellings accepted for ``SEARCH_PROVIDER``.
_ALIASES: Mapping[str, str] = {
    "": "wikimedia-commons",
    "commons": "wikimedia-commons",
    "wikimedia": "wikimedia-commons",
    "google": "google-custom-search",
    "google-cse": "google-custom-search",
    "serpapi": "serpapi-google-lens",
    "google-lens": "serpapi-google-lens",
    "lens": "serpapi-google-lens",
}


def resolve_provider_name(raw: str) -> str:
    """Map a configured ``SEARCH_PROVIDER`` value to a registered provider name."""
    key = str(raw or "").strip().lower()
    resolved = _ALIASES.get(key, key) or "wikimedia-commons"
    if resolved not in PROVIDERS:
        raise SearchNotConfiguredError(
            f"unknown SEARCH_PROVIDER {raw!r}; available: {', '.join(PROVIDER_NAMES)}"
        )
    return resolved


def build_http_client(
    settings: Settings,
    *,
    transport: Optional[HttpTransport] = None,
    sleeper: Optional[Any] = None,
) -> HttpClient:
    """Build the HTTP client for a search run.

    ``secrets`` is populated from the configured key so every log line and error
    message this client produces has that value scrubbed out.
    """
    client = HttpClient(
        transport=transport if transport is not None else HttpxTransport(),
        timeout=float(settings.request_timeout_s or 30),
        max_retries=int(settings.max_retries or 0),
        user_agent=settings.search_user_agent,
        secrets=tuple(v for v in (settings.search_api_key,) if v),
    )
    if sleeper is not None:
        client.sleeper = sleeper
    return client


def build_search_provider(
    settings: Optional[Settings] = None,
    *,
    transport: Optional[HttpTransport] = None,
    client: Optional[HttpClient] = None,
    sleeper: Optional[Any] = None,
) -> _HttpSearchProvider:
    """Instantiate the configured provider adapter.

    ``transport`` / ``client`` exist so tests can drive the provider against a
    fake HTTP boundary. Production passes neither and gets ``httpx``.
    """
    resolved = settings or Settings()
    provider_cls = PROVIDERS[resolve_provider_name(resolved.search_provider)]
    return provider_cls(
        settings=resolved,
        client=client
        or build_http_client(resolved, transport=transport, sleeper=sleeper),
    )


# --------------------------------------------------------------------------
# Service façade
# --------------------------------------------------------------------------

_ERROR_CODES: Mapping[str, ErrorCode] = {code.value: code for code in ErrorCode}


@dataclass
class SearchService:
    """Runs a provider and reports SUCCESS / NO_RESULTS / ERROR.

    The façade never raises for an expected condition: a missing key, a broken
    provider, and an empty result set all come back as a :class:`SearchOutcome`
    the caller can render. ``results`` is empty for anything but ``SUCCESS``, so
    a failure can never be mistaken for a candidate.
    """

    settings: Settings
    provider: _HttpSearchProvider

    @property
    def name(self) -> str:
        return self.provider.name

    def describe(self) -> dict[str, object]:
        return self.provider.describe()

    def build_query(self, encoding: FaceEncoding) -> SearchQuery:
        return self.provider.build_query(encoding)

    def search_for_encoding(self, encoding: FaceEncoding) -> SearchOutcome:
        """Build the query for ``encoding`` and execute it."""
        try:
            query = self.provider.build_query(encoding)
        except ServiceError as exc:
            return self._error_outcome(exc, query=None)
        return self.search(query)

    def search_uploaded_image(self, image_path: Path, encoding: FaceEncoding) -> SearchOutcome:
        provider = self.provider
        if not isinstance(provider, SerpApiLensProvider):
            return self.search_for_encoding(encoding)
        try:
            results = tuple(provider.search_uploaded_image(image_path, encoding))
        except ServiceError as exc:
            return self._error_outcome(exc, query=None)
        if not results:
            return SearchOutcome(
                status=SearchStatus.NO_RESULTS,
                provider=self.name,
                error="the provider returned no usable candidates.",
                error_code=ErrorCode.NO_SEARCH_RESULTS,
                retrieved_at=_now_iso(),
                attempts=provider.last_attempts,
            )
        return SearchOutcome(
            status=SearchStatus.SUCCESS,
            provider=self.name,
            results=results,
            retrieved_at=_now_iso(),
            attempts=provider.last_attempts,
        )

    def search(self, query: SearchQuery) -> SearchOutcome:
        """Execute ``query`` and wrap the result in a :class:`SearchOutcome`."""
        try:
            results = tuple(self.provider.search(query))
        except ServiceError as exc:
            return self._error_outcome(exc, query=query)
        except Exception as exc:  # noqa: BLE001 - never leak a raw traceback upward
            detail = redact_secrets(
                f"unexpected {type(exc).__name__}: {exc}",
                (self.settings.search_api_key,),
            )
            # Deliberately not LOGGER.exception: a traceback from third-party code
            # can echo a request URL that still carries the API key.
            LOGGER.error(
                "unexpected search failure in provider=%s: %s", self.name, detail
            )
            return SearchOutcome(
                status=SearchStatus.ERROR,
                provider=self.name,
                query=query,
                error=detail,
                error_code=ErrorCode.SEARCH_FAILED,
                retryable=False,
                retrieved_at=_now_iso(),
                attempts=self.provider.last_attempts,
            )

        if not results:
            return SearchOutcome(
                status=SearchStatus.NO_RESULTS,
                provider=self.name,
                query=query,
                error=(
                    "the provider returned no usable candidates. This is a genuine "
                    "empty result, not a match."
                ),
                error_code=ErrorCode.NO_SEARCH_RESULTS,
                retryable=False,
                retrieved_at=_now_iso(),
                attempts=self.provider.last_attempts,
            )

        return SearchOutcome(
            status=SearchStatus.SUCCESS,
            provider=self.name,
            query=query,
            results=results,
            retrieved_at=_now_iso(),
            attempts=self.provider.last_attempts,
        )

    def _error_outcome(
        self, exc: ServiceError, *, query: Optional[SearchQuery]
    ) -> SearchOutcome:
        code = _ERROR_CODES.get(getattr(exc, "code", ""), ErrorCode.SEARCH_FAILED)
        return SearchOutcome(
            status=SearchStatus.ERROR,
            provider=self.name,
            query=query,
            error=redact_secrets(str(exc), (self.settings.search_api_key,)),
            error_code=code,
            retryable=bool(getattr(exc, "retryable", False)),
            retrieved_at=_now_iso(),
            attempts=self.provider.last_attempts,
        )


def build_search_service(
    settings: Optional[Settings] = None,
    *,
    transport: Optional[HttpTransport] = None,
    client: Optional[HttpClient] = None,
    sleeper: Optional[Any] = None,
) -> SearchService:
    """Build the search façade around the configured provider."""
    resolved = settings or Settings()
    return SearchService(
        settings=resolved,
        provider=build_search_provider(
            resolved, transport=transport, client=client, sleeper=sleeper
        ),
    )
