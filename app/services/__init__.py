"""Replaceable service interfaces for FaceTrace.

The pipeline is defined entirely in terms of the protocols below, so any
capability can be swapped without touching the orchestrator:

    FaceInput -> FaceIdentifier -> SearchProvider -> MatchSelector
    -> FingerprintService -> BlockchainService -> VerificationService

Each module also ships a ``Pending*`` placeholder implementation that satisfies
its protocol and raises :class:`StageNotImplementedError`. These exist so the
wiring, the CLI, and the tests are exercisable in STEP 1 without faking any
result. Replace them in STEP 2 — never make one return invented data.
"""

from __future__ import annotations

__all__ = ["StageNotImplementedError", "ServiceError"]


class ServiceError(RuntimeError):
    """Base class for recoverable service-layer failures.

    Carries the stable :class:`~app.models.pipeline.ErrorCode` string so the
    orchestrator can turn any raised failure into a structured stage error.
    """

    #: Overridden by subclasses / raisers with an ``ErrorCode`` value.
    code: str = "SERVICE_ERROR"
    #: Whether the orchestrator may retry the operation.
    retryable: bool = False

    def __init__(
        self, message: str, *, code: str | None = None, retryable: bool | None = None
    ) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code
        if retryable is not None:
            self.retryable = retryable


class StageNotImplementedError(ServiceError):
    """Raised by STEP 1 placeholders. Never retryable — it is not transient."""

    code = "NOT_IMPLEMENTED"
    retryable = False

    def __init__(self, what: str) -> None:
        super().__init__(f"{what} is not implemented yet (scheduled for STEP 2).")
