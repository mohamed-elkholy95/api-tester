from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse
from app.services.benchmark_runner import BenchmarkRunner
from app.services.speed_tests import run_ping, run_stress, run_cold_start
from app.schemas.benchmark import (
    SingleBenchmarkRequest, MultiBenchmarkRequest,
    ConcurrentBenchmarkRequest, ComparisonBenchmarkRequest,
    PingRequest, StressTestRequest, ColdStartRequest,
)

router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])
runner = BenchmarkRunner()


@router.post("/single")
async def benchmark_single(req: SingleBenchmarkRequest):
    _validate_provider(req.provider_id)
    return EventSourceResponse(
        runner.run_single(req.provider_id, req.model, req.prompt,
                          req.max_tokens, req.temperature)
    )


@router.post("/multi")
async def benchmark_multi(req: MultiBenchmarkRequest):
    _validate_provider(req.provider_id)
    return EventSourceResponse(
        runner.run_multi(req.provider_id, req.model, req.prompt,
                         req.num_runs, req.max_tokens, req.temperature)
    )


@router.post("/concurrent")
async def benchmark_concurrent(req: ConcurrentBenchmarkRequest):
    _validate_provider(req.provider_id)
    return EventSourceResponse(
        runner.run_concurrent(req.provider_id, req.model, req.prompt,
                              req.num_runs, req.concurrency,
                              req.max_tokens, req.temperature)
    )


@router.post("/comparison")
async def benchmark_comparison(req: ComparisonBenchmarkRequest):
    for pm in req.providers:
        _validate_provider(pm["provider_id"])
    return EventSourceResponse(
        runner.run_comparison(req.providers, req.prompt,
                              req.num_runs, req.max_tokens, req.temperature)
    )


# --- Speed Test Endpoints ---

@router.post("/ping")
async def benchmark_ping(req: PingRequest):
    """Health check: minimal 1-token request to measure TTFB and classify API health."""
    _validate_provider(req.provider_id)
    result = await run_ping(req)
    return result


@router.post("/stress")
async def benchmark_stress(req: StressTestRequest):
    """Throughput stress test: ramp concurrency levels and measure TPS at each level."""
    _validate_provider(req.provider_id)
    return EventSourceResponse(run_stress(req))


@router.post("/cold-start")
async def benchmark_cold_start(req: ColdStartRequest):
    """Cold-start probe: measure TTFB after idle gaps vs warm follow-up requests."""
    _validate_provider(req.provider_id)
    return EventSourceResponse(run_cold_start(req))


def _validate_provider(provider_id: str):
    if provider_id not in runner.providers:
        available = list(runner.providers.keys())
        raise HTTPException(400, f"Unknown provider '{provider_id}'. Available: {available}")
