from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text
from contextlib import asynccontextmanager
from app.core.config import settings
import os

# Ensure data directory exists
db_path = settings.database_url.replace("sqlite+aiosqlite:///", "")
db_dir = os.path.dirname(db_path)
if db_dir:
    os.makedirs(db_dir, exist_ok=True)

engine = create_async_engine(settings.database_url, echo=False)
async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    from app.models.base import Base
    import app.models.tables  # noqa: F401 — registers all ORM models with Base.metadata
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await migrate_db()


async def migrate_db():
    """Add new columns to existing tables if they don't exist (idempotent)."""
    new_columns = [
        ("benchmark_runs", "inter_chunk_ms_avg", "FLOAT"),
        ("benchmark_runs", "inter_chunk_ms_p95", "FLOAT"),
        ("benchmark_runs", "total_chars", "INTEGER"),
        ("benchmark_runs", "total_words", "INTEGER"),
        ("benchmark_suites", "avg_inter_chunk_ms", "FLOAT"),
        ("benchmark_suites", "avg_total_chars", "FLOAT"),
    ]
    async with engine.begin() as conn:
        for table, col, col_type in new_columns:
            try:
                await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}"))
            except Exception:
                pass  # Column already exists — safe to ignore


async def close_db():
    await engine.dispose()


@asynccontextmanager
async def get_db():
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
