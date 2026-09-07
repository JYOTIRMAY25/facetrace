"""Minimal request models reserved for the investigation API."""

from pydantic import BaseModel


class InvestigationRequest(BaseModel):
    """Metadata accepted alongside an uploaded image."""

    source_name: str | None = None
