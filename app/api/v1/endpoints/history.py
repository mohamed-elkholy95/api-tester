from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.dependencies import get_db_session
from app.models.tables import BenchmarkSuite, BenchmarkRun
from app.schemas.benchmark import SuiteSummary, SuiteDetail, RunDetail

router = APIRouter(prefix="/history", tags=["history"])


@router.get("", response_model=list[SuiteSummary])
async def list_suites(
    provider_id: str | None = None,
    model: str | None = None,
    mode: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_db_session)
):
    """List benchmark suites with optional filters."""
    stmt = select(BenchmarkSuite).order_by(desc(BenchmarkSuite.created_at))
    if provider_id:
        stmt = stmt.where(BenchmarkSuite.provider_id == provider_id)
    if model:
        stmt = stmt.where(BenchmarkSuite.model == model)
    if mode:
        stmt = stmt.where(BenchmarkSuite.mode == mode)
    stmt = stmt.offset(offset).limit(limit)

    result = await session.execute(stmt)
    suites = result.scalars().all()
    return [SuiteSummary.model_validate(s) for s in suites]


@router.get("/{suite_id}", response_model=SuiteDetail)
async def get_suite(suite_id: str, session: AsyncSession = Depends(get_db_session)):
    """Get full suite detail with all individual run metrics."""
    result = await session.execute(
        select(BenchmarkSuite).where(BenchmarkSuite.id == suite_id)
    )
    suite = result.scalar_one_or_none()
    if not suite:
        raise HTTPException(404, "Suite not found")

    runs_result = await session.execute(
        select(BenchmarkRun)
        .where(BenchmarkRun.suite_id == suite_id)
        .order_by(BenchmarkRun.run_number)
    )
    runs = runs_result.scalars().all()

    # Build SuiteDetail manually to avoid lazy-loading the runs relationship
    suite_dict = SuiteSummary.model_validate(suite).model_dump()
    detail = SuiteDetail(
        **suite_dict,
        runs=[RunDetail.model_validate(r) for r in runs],
    )
    return detail


@router.delete("/{suite_id}")
async def delete_suite(suite_id: str, session: AsyncSession = Depends(get_db_session)):
    """Delete a benchmark suite and all its runs."""
    result = await session.execute(
        select(BenchmarkSuite).where(BenchmarkSuite.id == suite_id)
    )
    suite = result.scalar_one_or_none()
    if not suite:
        raise HTTPException(404, "Suite not found")
    await session.delete(suite)
    return {"status": "deleted", "suite_id": suite_id}


@router.delete("")
async def delete_all_suites(session: AsyncSession = Depends(get_db_session)):
    """Delete all benchmark suites and their runs."""
    result = await session.execute(select(BenchmarkSuite))
    suites = result.scalars().all()
    count = len(suites)
    for suite in suites:
        await session.delete(suite)
    return {"status": "deleted", "count": count}
