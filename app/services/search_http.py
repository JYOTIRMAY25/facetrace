"""HTTP boundary for the search stage.

Every outbound search request in FaceTrace goes through :class:`HttpTransport`.
Keeping the boundary behind a protocol means unit tests can substitute a fake
transport that replays a recorded provider payload, while the application always
runs the real :class:`HttpxTransport`. A fake transport is a *test* artefact and
never a fallback inside the application.

Credential safety is enforced here rather than in each provider:

* :class:`HttpResponse` only ever holds a redacted URL, so a response object
  cannot leak an API key even if it is logged or pretty-printed.
* :class:`HttpClient` scrubs every error message it raises or logs, using both a
  pattern for ``key=``-style query parameters and the literal secret values it
  was given.
* Response bodies are never logged — they are untrusted third-party content.

``httpx`` is imported lazily inside :meth:`HttpxTransport.get` so importing
``app.*`` stays dependency-free (see ``tests/test_imports.py``).
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, runtime_checkable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from ..models.pipeline import ErrorCode
from . import ServiceError

__all__ = [
    "DEFAULT_USER_AGENT",
    "REDACTED",
    "RETRYABLE_STATUS",
    "SECRET_PARAM_NAMES",
    "HttpClient",
    "HttpResponse",
    "HttpTransport",
    "HttpxTransport",
    "SearchTransportError",
    "merge_url",
    "redact_secrets",
    "redact_url",
]

LOGGER = logging.getLogger(__name__)

#: Replacement token written in place of any credential.
REDACTED = "[REDACTED]"

#: Query-parameter names whose values are treated as credentials.
SECRET_PARAM_NAMES: frozenset[str] = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "auth",
        "authorization",
        "key",
        "password",
        "secret",
        "serpapi_key",
        "signature",
        "token",
    }
)

#: HTTP statuses worth retrying: transient server and rate-limit conditions.
RETRYABLE_STATUS: frozenset[int] = frozenset({408, 425, 429, 500, 502, 503, 504})

DEFAULT_USER_AGENT = "FaceTrace/0.1 (HH Goa 2026 Task 3 prototype)"

#: Matches ``key=value`` pairs for credential-bearing parameter names anywhere in
#: free text, so a URL embedded in a third-party exception message is scrubbed too.
_SECRET_PAIR_RE = re.compile(
    r"(?i)\b(" + "|".join(sorted(SECRET_PARAM_NAMES)) + r")=([^&\s\"'>\]]+)"
)

#: Header names never echoed into a log line.
_SECRET_HEADER_NAMES: frozenset[str] = frozenset(
    {"authorization", "proxy-authorization", "x-api-key", "cookie"}
)

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class SearchTransportError(ServiceError):
    """A search request could not be completed.

    Messages reaching this exception are already scrubbed by
    :class:`HttpClient`; raisers must still avoid interpolating raw URLs.
    """

    code = ErrorCode.SEARCH_FAILED.value
    retryable = False


def redact_secrets(text: str, secrets: Iterable[str] = ()) -> str:
    """Remove credentials from ``text``.

    Two passes: credential-shaped query parameters, then the literal secret
    values supplied by the caller. Very short secrets (< 4 characters) are
    ignored because substituting them would mangle unrelated text without
    protecting anything meaningful.
    """
    scrubbed = _SECRET_PAIR_RE.sub(lambda m: f"{m.group(1)}={REDACTED}", str(text))
    for secret in secrets:
        value = (secret or "").strip()
        if len(value) >= 4:
            scrubbed = scrubbed.replace(value, REDACTED)
    return scrubbed


def redact_url(url: str) -> str:
    """Return ``url`` with credential-bearing query parameters replaced."""
    parts = urlsplit(str(url))
    if not parts.query:
        return str(url)
    pairs = [
        (name, REDACTED if name.lower() in SECRET_PARAM_NAMES else value)
        for name, value in parse_qsl(parts.query, keep_blank_values=True)
    ]
    # safe="[]" keeps the REDACTED marker readable instead of percent-encoding it.
    return urlunsplit(parts._replace(query=urlencode(pairs, safe="[]")))


def merge_url(url: str, params: Optional[Mapping[str, Any]] = None) -> str:
    """Combine ``url`` with ``params`` into the full request target.

    Used for log lines and for fake transports; the real transport passes
    ``params`` to ``httpx`` so it does the encoding.
    """
    parts = urlsplit(str(url))
    pairs = parse_qsl(parts.query, keep_blank_values=True)
    for name, value in (params or {}).items():
        pairs.append((str(name), "" if value is None else str(value)))
    return urlunsplit(parts._replace(query=urlencode(pairs)))


def _safe_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Header snapshot with credential-bearing values removed."""
    return {
        name: (REDACTED if name.lower() in _SECRET_HEADER_NAMES else value)
        for name, value in headers.items()
    }


@dataclass(frozen=True)
class HttpResponse:
    """One provider response.

    ``url`` is stored already redacted (see :func:`redact_url`) so this object is
    safe to log or repr. ``text`` is untrusted third-party content and must be
    sanitised before it is used or displayed.
    """

    status_code: int
    text: str = ""
    url: str = ""
    headers: Mapping[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    @property
    def retryable_status(self) -> bool:
        return self.status_code in RETRYABLE_STATUS

    def json(self) -> Any:
        """Decode the body as JSON.

        Raises :class:`SearchTransportError` rather than letting a
        ``JSONDecodeError`` escape, and never includes the body in the message —
        a provider error page can contain arbitrary content.
        """
        try:
            return json.loads(self.text)
        except ValueError as exc:
            raise SearchTransportError(
                f"provider returned a non-JSON body (HTTP {self.status_code}, "
                f"{len(self.text)} bytes)",
                retryable=False,
            ) from exc


@runtime_checkable
class HttpTransport(Protocol):
    """Minimal outbound-HTTP surface the search providers depend on."""

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout: float,
    ) -> HttpResponse:
        """Perform one GET request.

        Must raise :class:`SearchTransportError` for connection-level failures
        (``retryable=True`` when transient) and return an :class:`HttpResponse`
        for anything the server actually answered, including 4xx and 5xx.
        """
        ...

    def post(
        self,
        url: str,
        *,
        data: Mapping[str, Any],
        files: Mapping[str, tuple[str, bytes, str]],
        headers: Mapping[str, str],
        timeout: float,
    ) -> HttpResponse:
        ...


@dataclass
class HttpxTransport:
    """The real transport, backed by ``httpx``.

    ``httpx`` is imported inside :meth:`get` on purpose: importing the ``app``
    package must not pull in heavy third-party dependencies.
    """

    name: str = "httpx"
    follow_redirects: bool = True

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout: float,
    ) -> HttpResponse:
        try:
            import httpx  # noqa: PLC0415 - lazy on purpose
        except ModuleNotFoundError as exc:  # pragma: no cover - install issue
            raise SearchTransportError(
                "httpx is not installed; run: pip install -r requirements.txt",
                retryable=False,
            ) from exc

        try:
            response = httpx.get(
                url,
                params=dict(params),
                headers=dict(headers),
                timeout=timeout,
                follow_redirects=self.follow_redirects,
            )
        except httpx.TimeoutException as exc:
            raise SearchTransportError(
                f"search request timed out after {timeout}s ({type(exc).__name__})",
                retryable=True,
            ) from exc
        except httpx.HTTPError as exc:
            # str(exc) can embed the request URL, so scrub before it is stored.
            raise SearchTransportError(
                f"search request failed: {type(exc).__name__}: "
                f"{redact_secrets(str(exc))}",
                retryable=True,
            ) from exc

        return HttpResponse(
            status_code=response.status_code,
            text=response.text,
            url=redact_url(str(response.url)),
            headers=_safe_headers(dict(response.headers)),
        )

    def post(
        self,
        url: str,
        *,
        data: Mapping[str, Any],
        files: Mapping[str, tuple[str, bytes, str]],
        headers: Mapping[str, str],
        timeout: float,
    ) -> HttpResponse:
        try:
            import httpx
            response = httpx.post(
                url,
                data=dict(data),
                files=dict(files),
                headers=dict(headers),
                timeout=timeout,
                follow_redirects=self.follow_redirects,
            )
        except httpx.TimeoutException as exc:
            raise SearchTransportError(
                f"search upload timed out after {timeout}s", retryable=True
            ) from exc
        except httpx.HTTPError as exc:
            raise SearchTransportError(
                f"search upload failed: {type(exc).__name__}: {redact_secrets(str(exc))}",
                retryable=True,
            ) from exc
        return HttpResponse(
            status_code=response.status_code,
            text=response.text,
            url=redact_url(str(response.url)),
            headers=_safe_headers(dict(response.headers)),
        )


@dataclass
class HttpClient:
    """Retrying, credential-scrubbing wrapper around an :class:`HttpTransport`.

    ``secrets`` holds the literal credential values in play for this run. They
    are used only to scrub outgoing log lines and error messages; nothing here
    ever prints or returns them.

    ``sleeper`` is injectable so tests exercise the retry path without waiting.
    """

    transport: HttpTransport
    timeout: float = 30.0
    max_retries: int = 2
    user_agent: str = DEFAULT_USER_AGENT
    secrets: tuple[str, ...] = ()
    accept: str = "application/json"
    backoff_s: float = 0.5
    sleeper: Callable[[float], None] = time.sleep
    #: Attempts made by the most recent :meth:`get`, retries included. Set even
    #: when the request ultimately raises, so a caller can report it.
    last_attempts: int = 0

    def scrub(self, text: str) -> str:
        """Redact this run's credentials from arbitrary text."""
        return redact_secrets(text, self.secrets)

    def request_headers(self, extra: Optional[Mapping[str, str]] = None) -> dict[str, str]:
        headers = {
            "User-Agent": self.user_agent or DEFAULT_USER_AGENT,
            "Accept": self.accept,
        }
        headers.update(dict(extra or {}))
        return headers

    def get(
        self,
        url: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> tuple[HttpResponse, int]:
        """GET ``url``, retrying transient failures.

        Returns ``(response, attempts)``. A non-transient 4xx is returned rather
        than raised so the calling provider can distinguish "your key is
        rejected" from "the provider is down".
        """
        request_headers = self.request_headers(headers)
        safe_target = redact_url(merge_url(url, params))
        total = max(1, int(self.max_retries) + 1)

        attempts = 0
        last_response: Optional[HttpResponse] = None
        last_error: Optional[SearchTransportError] = None

        while attempts < total:
            attempts += 1
            self.last_attempts = attempts
            try:
                response = self.transport.get(
                    url,
                    params=dict(params or {}),
                    headers=request_headers,
                    timeout=self.timeout,
                )
            except SearchTransportError as exc:
                last_error = self._scrubbed(exc)
                if not exc.retryable or attempts >= total:
                    raise last_error from exc
                LOGGER.warning(
                    "search transport error (attempt %d/%d) for %s: %s",
                    attempts,
                    total,
                    safe_target,
                    last_error,
                )
                self.sleeper(self.backoff_s * attempts)
                continue

            last_response = response
            if response.retryable_status and attempts < total:
                LOGGER.warning(
                    "search provider returned HTTP %d (attempt %d/%d) for %s",
                    response.status_code,
                    attempts,
                    total,
                    safe_target,
                )
                self.sleeper(self.backoff_s * attempts)
                continue
            return response, attempts

        if last_response is not None:  # pragma: no cover - loop always returns
            return last_response, attempts
        raise last_error or SearchTransportError(
            "search request failed with no response", retryable=True
        )

    def get_json(
        self,
        url: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> tuple[Any, HttpResponse, int]:
        """GET ``url`` and decode a JSON body. Returns ``(payload, response, attempts)``."""
        response, attempts = self.get(url, params=params, headers=headers)
        return response.json(), response, attempts

    def post_json(
        self,
        url: str,
        *,
        data: Mapping[str, Any],
        files: Mapping[str, tuple[str, bytes, str]],
    ) -> tuple[Any, HttpResponse, int]:
        response = self.transport.post(
            url,
            data=data,
            files=files,
            headers=self.request_headers(),
            timeout=self.timeout,
        )
        self.last_attempts = 1
        return response.json(), response, 1

    def _scrubbed(self, exc: SearchTransportError) -> SearchTransportError:
        """Copy ``exc`` with this run's credentials removed from its message."""
        return SearchTransportError(
            self.scrub(str(exc)), code=exc.code, retryable=exc.retryable
        )


def clean_control_chars(value: str) -> str:
    """Strip control characters from untrusted provider text."""
    return _CONTROL_CHARS_RE.sub("", value)
