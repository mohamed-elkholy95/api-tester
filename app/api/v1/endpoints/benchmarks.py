from fastapi import APIRouter, HTTPException, Depends
from sse_starlette.sse import EventSourceResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db_session
from app.repositories.benchmark_repo import BenchmarkRepository
from app.repositories.provider_repo import ProviderRepository
from app.services.benchmark_runner import BenchmarkRunner
from app.services.provider_service import ProviderService
from app.services.speed_tests import run_ping, run_stress, run_cold_start
from app.schemas.benchmark import (
    SingleBenchmarkRequest, MultiBenchmarkRequest,
    ConcurrentBenchmarkRequest, ComparisonBenchmarkRequest,
    PingRequest, StressTestRequest, ColdStartRequest,
)

router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])

async def get_provider_service(session: AsyncSession = Depends(get_db_session)) -> ProviderService:
    repo = ProviderRepository(session)
    return ProviderService(repo)

async def get_benchmark_runner(
    session: AsyncSession = Depends(get_db_session),
    provider_service: ProviderService = Depends(get_provider_service)
) -> BenchmarkRunner:
    repo = BenchmarkRepository(session)
    return BenchmarkRunner(benchmark_repo=repo, provider_service=provider_service)

async def _validate_provider(provider_id: str, provider_service: ProviderService):
    available = await provider_service.get_merged_providers()
    if provider_id not in available:
        raise HTTPException(400, f"Unknown provider '{provider_id}'. Available: {list(available.keys())}")


@router.post("/single")
async def benchmark_single(
    req: SingleBenchmarkRequest,
    provider_service: ProviderService = Depends(get_provider_service),
    runner: BenchmarkRunner = Depends(get_benchmark_runner)
):
    await _validate_provider(req.provider_id, provider_service)
    return EventSourceResponse(
        runner.run_single(req.provider_id, req.model, req.prompt,
                          req.max_tokens, req.temperature)
    )


@router.post("/multi")
async def benchmark_multi(
    req: MultiBenchmarkRequest,
    provider_service: ProviderService = Depends(get_provider_service),
    runner: BenchmarkRunner = Depends(get_benchmark_runner)
):
    await _validate_provider(req.provider_id, provider_service)
    return EventSourceResponse(
        runner.run_multi(req.provider_id, req.model, req.prompt,
                         req.num_runs, req.max_tokens, req.temperature)
    )


@router.post("/concurrent")
async def benchmark_concurrent(
    req: ConcurrentBenchmarkRequest,
    provider_service: ProviderService = Depends(get_provider_service),
    runner: BenchmarkRunner = Depends(get_benchmark_runner)
):
    await _validate_provider(req.provider_id, provider_service)
    return EventSourceResponse(
        runner.run_concurrent(req.provider_id, req.model, req.prompt,
                              req.num_runs, req.concurrency,
                              req.max_tokens, req.temperature)
    )


@router.post("/comparison")
async def benchmark_comparison(
    req: ComparisonBenchmarkRequest,
    provider_service: ProviderService = Depends(get_provider_service),
    runner: BenchmarkRunner = Depends(get_benchmark_runner)
):
    for pm in req.providers:
        await _validate_provider(pm["provider_id"], provider_service)
    return EventSourceResponse(
        runner.run_comparison(req.providers, req.prompt,
                              req.num_runs, req.max_tokens, req.temperature)
    )


# --- Speed Test Endpoints ---

@router.post("/ping")
async def benchmark_ping(
    req: PingRequest,
    provider_service: ProviderService = Depends(get_provider_service)
):
    """Health check: minimal 1-token request to measure TTFB and classify API health."""
    await _validate_provider(req.provider_id, provider_service)
    result = await run_ping(req)
    return result


@router.post("/stress")
async def benchmark_stress(
    req: StressTestRequest,
    provider_service: ProviderService = Depends(get_provider_service)
):
    """Throughput stress test: ramp concurrency levels and measure TPS at each level."""
    await _validate_provider(req.provider_id, provider_service)
    return EventSourceResponse(run_stress(req))


@router.post("/cold-start")
async def benchmark_cold_start(
    req: ColdStartRequest,
    provider_service: ProviderService = Depends(get_provider_service)
):
    """Cold-start probe: measure TTFB after idle gaps vs warm follow-up requests."""
    await _validate_provider(req.provider_id, provider_service)
    return EventSourceResponse(run_cold_start(req))
