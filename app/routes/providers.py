from fastapi import APIRouter
from app.core.config import settings

router = APIRouter(prefix="/providers", tags=["providers"])


@router.get("")
async def list_providers():
    """List all configured providers and their models."""
    providers = settings.get_providers()
    return {
        pid: {
            "name": p.name,
            "models": p.models,
            "default_model": p.default_model,
        }
        for pid, p in providers.items()
    }
