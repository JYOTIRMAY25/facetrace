"""Search stage interface: genuine web/social search.

TODO(STEP 2): implement a real provider behind :class:`SearchProvider`.

Hard rules for any implementation:
  * The search must actually execute against the configured provider.
  * The query must depend on the submitted image — no fixed query strings.
  * Never return a hardcoded, pre-selected, or synthesised result.
  * An empty result set is a valid outcome (``NO_SEARCH_RESULTS``); it must not
    be converted into a match.
  * Every result must carry its real source URL.
  * Respect the provider's terms, API policy, rate limits, and robots rules.
    Do not bypass access controls or authentication.
  * Treat all returned content as untrusted input.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from ..models.pipeline import FaceEncoding, SearchQuery, SearchResult
from . import StageNotImplementedError

__all__ = [
    "SearchProvider",
    "SearchResultParser",
    "PendingSearchProvider",
    "PendingSearchResultParser",
]


@runtime_checkable
class SearchProvider(Protocol):
    """Executes a genuine search and returns raw candidate results."""

    #: Provider identifier recorded on every result and shown by ``--health``.
    name: str

    def build_query(self, encoding: FaceEncoding) -> SearchQuery:
        """Derive a provider query from the face encoding.

        The returned query must vary with the input; a constant query would
        make the search decorative rather than genuine.
        """
        ...

    def search(self, query: SearchQuery) -> Sequence[SearchResult]:
        """Execute the search.

        Returns an empty sequence when the provider legitimately found nothing.
        Raises a :class:`~app.services.ServiceError` with code ``SEARCH_FAILED``
        (``retryable=True`` for transient network/provider errors).
        """
        ...


@runtime_checkable
class SearchResultParser(Protocol):
    """Normalises one provider-specific payload into a :class:`SearchResult`."""

    def parse(self, raw: Any, *, source: str) -> SearchResult:
        """Extract only the fields needed for verification.

        Keep this to url / title / text / image_url / source / selected
        metadata. Normalisation here must not overlap with canonicalization —
        see :mod:`app.services.fingerprint`.
        """
        ...


class PendingSearchProvider:
    """STEP 1 placeholder. Satisfies :class:`SearchProvider`, implements nothing."""

    name = "pending-search-provider"

    def build_query(self, encoding: FaceEncoding) -> SearchQuery:
        raise StageNotImplementedError("Search query construction")

    def search(self, query: SearchQuery) -> Sequence[SearchResult]:
        raise StageNotImplementedError("Genuine web/social search")


class PendingSearchResultParser:
    """STEP 1 placeholder for :class:`SearchResultParser`."""

    def parse(self, raw: Any, *, source: str) -> SearchResult:
        raise StageNotImplementedError("Search result parsing")
