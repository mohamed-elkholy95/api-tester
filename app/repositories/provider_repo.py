import json
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.api_settings import ProviderSetting
from app.core.config import ProviderConfig


class ProviderRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_all_providers(self) -> dict[str, ProviderConfig]:
        """Load providers persisted via the Settings UI from SQLite."""
        result: dict[str, ProviderConfig] = {}
        rows = (await self.session.execute(select(ProviderSetting))).scalars().all()

        for row in rows:
            try:
                models = json.loads(row.models_json or "[]")
            except Exception:
                models = []
            try:
                extra_headers = json.loads(row.extra_headers_json or "{}")
            except Exception:
                extra_headers = {}

            result[row.id] = ProviderConfig(
                name=row.name,
                base_url=row.base_url,
                api_key=row.api_key or "",
                models=models,
                default_model=row.default_model or "",
                supports_stream_usage=row.supports_stream_usage,
                min_temperature=row.min_temperature,
                extra_headers=extra_headers,
            )

        return result
