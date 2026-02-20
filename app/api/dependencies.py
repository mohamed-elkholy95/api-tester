from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.database import async_session_factory

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for providing a SQLAlchemy async session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
