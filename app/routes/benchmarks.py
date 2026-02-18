from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse
from app.services.benchmark_runner import BenchmarkRunner
from app.schemas.benchmark import (
    SingleBenchmarkRequest, MultiBenchmarkRequest,
    ConcurrentBenchmarkRequest, ComparisonBenchmarkRequest,
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


def _validate_provider(provider_id: str):
    if provider_id not in runner.providers:
        available = list(runner.providers.keys())
        raise HTTPException(400, f"Unknown provider '{provider_id}'. Available: {available}")
