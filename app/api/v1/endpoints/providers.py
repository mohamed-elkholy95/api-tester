from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.dependencies import get_db_session
from app.repositories.provider_repo import ProviderRepository
from app.services.provider_service import ProviderService

router = APIRouter(prefix="/providers", tags=["providers"])

async def get_provider_service(session: AsyncSession = Depends(get_db_session)) -> ProviderService:
    repo = ProviderRepository(session)
    return ProviderService(repo)

@router.get("")
async def list_providers(provider_service: ProviderService = Depends(get_provider_service)):
    """List all configured providers (env + DB) and their models.

    DB-stored providers override .env providers with the same ID so users can
    update a provider's settings from the UI without restarting the server.
    """
    merged = await provider_service.get_merged_providers()

    # Need to know which ones came from DB. The repository can be queried, or we can just 
    # check if the provider came from DB.
    # To keep it simple, since get_merged_providers() yields all, let's just 
    # query the DB again to see which ones are DB-originated.
    # Or, the ProviderService could expose `get_db_providers()` as well.
    # Since ProviderService encapsulates this, let's access the repo.
    db_providers = await provider_service.provider_repo.get_all_providers()

    return {
        pid: {
            "name": p.name,
            "models": p.models,
            "default_model": p.default_model,
            "source": "db" if pid in db_providers else "env",
        }
        for pid, p in merged.items()
    }
