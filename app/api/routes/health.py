from fastapi import APIRouter, Request

from ...core.config import Settings

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict[str, str]:
    settings: Settings = request.app.state.settings
    return {"status": "ok", "service": settings.app_name}
