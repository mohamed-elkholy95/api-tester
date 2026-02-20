from app.core.config import settings, ProviderConfig
from app.repositories.provider_repo import ProviderRepository


class ProviderService:
    def __init__(self, provider_repo: ProviderRepository):
        self.provider_repo = provider_repo

    async def get_merged_providers(self) -> dict[str, ProviderConfig]:
        """Merge .env providers with DB-stored providers (DB wins on ID clash)."""
        env_providers = settings.get_providers()
        db_providers = await self.provider_repo.get_all_providers()
        return {**env_providers, **db_providers}
