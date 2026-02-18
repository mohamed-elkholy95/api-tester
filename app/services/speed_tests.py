"""
speed_tests.py — Ping, Stress Test, and Cold-Start probe orchestration.
"""
import asyncio
import time
import uuid
from typing import AsyncGenerator

from app.core.config import settings
from app.services.llm_client import LLMClient
from app.services.statistics import compute_suite_stats
from app.schemas.benchmark import (
    PingRequest, PingResult,
    StressTestRequest, StressLevelEvent, StressCompleteEvent,
    ColdStartRequest, ColdStartProbeEvent, ColdStartCompleteEvent,
)


def _get_client(provider_id: str) -> LLMClient:
    providers = settings.get_providers()
    return LLMClient(provider_id, providers[provider_id])


# ---------------------------------------------------------------------------
# 1. Ping / Health Check
# ---------------------------------------------------------------------------

async def run_ping(req: PingRequest) -> PingResult:
    """Send a minimal 1-token request and classify API health."""
    client = _get_client(req.provider_id)
    try:
        result = await client.benchmark_streaming(
            model=req.model,
            prompt="Hi",
            max_tokens=1,
            temperature=0.7,
        )
        if result.status == "error":
            return PingResult(
                provider_id=req.provider_id,
                model=req.model,
                status="error",
                ttfb_ms=result.ttfb_ms or None,
                round_trip_ms=result.total_latency_ms or None,
                health="error",
            )
        # Classify health by TTFB
        ttfb = result.ttfb_ms
        if ttfb < 500:
            health = "ok"
        elif ttfb < 2000:
            health = "slow"
        else:
            health = "degraded"
        return PingResult(
            provider_id=req.provider_id,
            model=req.model,
            status="success",
            ttfb_ms=round(ttfb, 2),
            round_trip_ms=round(result.total_latency_ms, 2),
            health=health,
        )
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# 2. Throughput Stress Test (SSE)
# ---------------------------------------------------------------------------

async def run_stress(req: StressTestRequest) -> AsyncGenerator[dict, None]:
    """Ramp concurrency levels and measure TPS at each level."""
    levels_data: list[StressLevelEvent] = []

    for concurrency in req.concurrency_levels:
        client = _get_client(req.provider_id)
        semaphore = asyncio.Semaphore(concurrency)
        level_results = []

        async def _worker():
            async with semaphore:
                r = await client.benchmark_streaming(
                    model=req.model,
                    prompt=req.prompt,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                )
                level_results.append(r)

        tasks = [asyncio.create_task(_worker()) for _ in range(req.runs_per_level)]
        await asyncio.gather(*tasks, return_exceptions=True)
        await client.close()

        successful = [r for r in level_results if r.status == "success"]
        error_count = len(level_results) - len(successful)
        tps_vals = [r.tokens_per_second for r in successful if r.tokens_per_second]
        latencies = [r.total_latency_ms for r in successful if r.total_latency_ms]

        level_event = StressLevelEvent(
            concurrency=concurrency,
            avg_tps=round(sum(tps_vals) / len(tps_vals), 2) if tps_vals else None,
            avg_latency_ms=round(sum(latencies) / len(latencies), 2) if latencies else None,
            error_rate=round(error_count / req.runs_per_level, 3),
            successful_runs=len(successful),
        )
        levels_data.append(level_event)
        yield {"event": "stress_level_complete", "data": level_event.model_dump_json()}

    # Find peak TPS level
    peak_level = None
    peak_tps = None
    for lv in levels_data:
        if lv.avg_tps is not None:
            if peak_tps is None or lv.avg_tps > peak_tps:
                peak_tps = lv.avg_tps
                peak_level = lv.concurrency

    complete = StressCompleteEvent(
        levels=levels_data,
        peak_tps_concurrency=peak_level,
        peak_tps=peak_tps,
    )
    yield {"event": "stress_complete", "data": complete.model_dump_json()}


# ---------------------------------------------------------------------------
# 3. Cold-Start Probe (SSE)
# ---------------------------------------------------------------------------

async def run_cold_start(req: ColdStartRequest) -> AsyncGenerator[dict, None]:
    """Measure cold-start TTFB vs warm TTFB with intentional idle gaps."""
    cold_ttfbs: list[float] = []
    warm_ttfbs: list[float] = []
    probe_num = 0

    for probe_idx in range(req.num_cold_probes):
        # --- Cold probe (after gap) ---
        if probe_idx > 0:
            # Wait gap_seconds before each cold probe (skip before first)
            yield {"event": "tick", "data": f'{{"waiting_gap_seconds": {req.gap_seconds}}}'}
            await asyncio.sleep(req.gap_seconds)

        probe_num += 1
        client = _get_client(req.provider_id)
        try:
            cold_result = await client.benchmark_streaming(
                model=req.model,
                prompt=req.prompt,
                max_tokens=req.max_tokens,
                temperature=0.7,
            )
        finally:
            await client.close()

        cold_ttfb = cold_result.ttfb_ms if cold_result.status == "success" else None
        if cold_ttfb:
            cold_ttfbs.append(cold_ttfb)

        cold_event = ColdStartProbeEvent(
            probe_number=probe_num,
            probe_type="cold",
            ttfb_ms=round(cold_ttfb, 2) if cold_ttfb else None,
            total_latency_ms=round(cold_result.total_latency_ms, 2),
        )
        yield {"event": "cold_start_probe", "data": cold_event.model_dump_json()}

        # --- Warm probe (immediate follow-up) ---
        probe_num += 1
        client = _get_client(req.provider_id)
        try:
            warm_result = await client.benchmark_streaming(
                model=req.model,
                prompt=req.prompt,
                max_tokens=req.max_tokens,
                temperature=0.7,
            )
        finally:
            await client.close()

        warm_ttfb = warm_result.ttfb_ms if warm_result.status == "success" else None
        if warm_ttfb:
            warm_ttfbs.append(warm_ttfb)

        warm_event = ColdStartProbeEvent(
            probe_number=probe_num,
            probe_type="warm",
            ttfb_ms=round(warm_ttfb, 2) if warm_ttfb else None,
            total_latency_ms=round(warm_result.total_latency_ms, 2),
        )
        yield {"event": "cold_start_probe", "data": warm_event.model_dump_json()}

    # Summary
    avg_cold = round(sum(cold_ttfbs) / len(cold_ttfbs), 2) if cold_ttfbs else None
    avg_warm = round(sum(warm_ttfbs) / len(warm_ttfbs), 2) if warm_ttfbs else None
    ratio = round(avg_cold / avg_warm, 2) if (avg_cold and avg_warm and avg_warm > 0) else None

    complete = ColdStartCompleteEvent(
        avg_cold_ttfb_ms=avg_cold,
        avg_warm_ttfb_ms=avg_warm,
        cold_vs_warm_ratio=ratio,
    )
    yield {"event": "cold_start_complete", "data": complete.model_dump_json()}
