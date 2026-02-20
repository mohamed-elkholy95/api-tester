import asyncio
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator

from app.core.config import settings
from app.services.llm_client import LLMClient, RunResult
from app.services.statistics import compute_suite_stats
from app.services.provider_service import ProviderService
from app.repositories.benchmark_repo import BenchmarkRepository
from app.schemas.benchmark import (
    RunMetricEvent, SuiteCompleteEvent, TickEvent, TokenChunkEvent,
)


class BenchmarkRunner:

    def __init__(self, benchmark_repo: BenchmarkRepository, provider_service: ProviderService):
        self.benchmark_repo = benchmark_repo
        self.provider_service = provider_service

    async def _get_client(self, provider_id: str) -> LLMClient:
        providers = await self.provider_service.get_merged_providers()
        provider = providers[provider_id]
        return LLMClient(provider_id, provider)

    # --- Single Run ---

    async def run_single(
        self, provider_id: str, model: str, prompt: str,
        max_tokens: int, temperature: float,
    ) -> AsyncGenerator[dict, None]:
        """Single run benchmark, yielding SSE events."""
        suite_id = str(uuid.uuid4())
        run_number = 1
        client = await self._get_client(provider_id)
        sse_queue: asyncio.Queue[dict | None] = asyncio.Queue()

        async def tick_cb(elapsed: float, phase: str):
            ev = TickEvent(suite_id=suite_id, run_number=run_number,
                           elapsed_seconds=elapsed, phase=phase)
            await sse_queue.put({"event": "tick", "data": ev.model_dump_json()})

        cumulative_chars = 0
        cumulative_words = 0

        async def chunk_cb(idx: int, text: str, elapsed_ms: float, inter_ms: float):
            nonlocal cumulative_chars, cumulative_words
            cumulative_chars += len(text)
            cumulative_words += len(text.split())
            ev = TokenChunkEvent(
                suite_id=suite_id, run_number=run_number,
                chunk_index=idx, text=text,
                elapsed_ms=elapsed_ms, inter_chunk_ms=inter_ms,
                cumulative_chars=cumulative_chars,
                cumulative_words=cumulative_words,
            )
            await sse_queue.put({"event": "token_chunk", "data": ev.model_dump_json()})

        result_holder: list[RunResult] = []

        async def _run():
            r = await client.benchmark_streaming(
                model, prompt, max_tokens, temperature,
                tick_callback=tick_cb, chunk_callback=chunk_cb,
            )
            result_holder.append(r)
            await sse_queue.put(None)  # sentinel

        runner_task = asyncio.create_task(_run())

        try:
            while True:
                item = await sse_queue.get()
                if item is None:
                    break
                yield item

            await runner_task
            result = result_holder[0]

            event = RunMetricEvent(
                suite_id=suite_id, run_number=1,
                provider_id=provider_id, model=model,
                status=result.status, ttfb_ms=result.ttfb_ms,
                total_latency_ms=result.total_latency_ms,
                tokens_per_second=result.tokens_per_second,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                response_preview=result.response_text[:500],
                error_message=result.error_message,
                inter_chunk_ms_avg=result.inter_chunk_ms_avg or None,
                inter_chunk_ms_p95=result.inter_chunk_ms_p95 or None,
                total_chars=result.total_chars or None,
                total_words=result.total_words or None,
            )
            yield {"event": "run_metric", "data": event.model_dump_json()}

            await self.benchmark_repo.save_suite(suite_id, "single", provider_id, model,
                                       prompt, 1, 1, max_tokens, temperature, [result])
            stats = compute_suite_stats([self._result_to_dict(result)])
            complete = SuiteCompleteEvent(
                suite_id=suite_id, status="completed",
                total_runs=1, successful_runs=1 if result.status == "success" else 0,
                **stats,
            )
            yield {"event": "suite_complete", "data": complete.model_dump_json()}

        finally:
            await client.close()

    # --- Multi-Run (Sequential N runs, same config) ---

    async def run_multi(
        self, provider_id: str, model: str, prompt: str,
        num_runs: int, max_tokens: int, temperature: float,
    ) -> AsyncGenerator[dict, None]:
        suite_id = str(uuid.uuid4())
        client = await self._get_client(provider_id)
        results: list[RunResult] = []

        try:
            for i in range(num_runs):
                run_number = i + 1
                sse_queue: asyncio.Queue[dict | None] = asyncio.Queue()
                cumulative_chars = 0
                cumulative_words = 0

                async def tick_cb(elapsed: float, phase: str, _rn=run_number, _q=sse_queue):
                    ev = TickEvent(suite_id=suite_id, run_number=_rn,
                                   elapsed_seconds=elapsed, phase=phase)
                    await _q.put({"event": "tick", "data": ev.model_dump_json()})

                async def chunk_cb(idx: int, text: str, elapsed_ms: float, inter_ms: float,
                                   _rn=run_number, _q=sse_queue):
                    nonlocal cumulative_chars, cumulative_words
                    cumulative_chars += len(text)
                    cumulative_words += len(text.split())
                    ev = TokenChunkEvent(
                        suite_id=suite_id, run_number=_rn,
                        chunk_index=idx, text=text,
                        elapsed_ms=elapsed_ms, inter_chunk_ms=inter_ms,
                        cumulative_chars=cumulative_chars,
                        cumulative_words=cumulative_words,
                    )
                    await _q.put({"event": "token_chunk", "data": ev.model_dump_json()})

                result_holder: list[RunResult] = []

                async def _run(_q=sse_queue):
                    r = await client.benchmark_streaming(
                        model, prompt, max_tokens, temperature,
                        tick_callback=tick_cb, chunk_callback=chunk_cb,
                    )
                    result_holder.append(r)
                    await _q.put(None)

                runner_task = asyncio.create_task(_run())

                while True:
                    item = await sse_queue.get()
                    if item is None:
                        break
                    yield item

                await runner_task
                result = result_holder[0]
                results.append(result)

                event = RunMetricEvent(
                    suite_id=suite_id, run_number=run_number,
                    provider_id=provider_id, model=model,
                    status=result.status, ttfb_ms=result.ttfb_ms,
                    total_latency_ms=result.total_latency_ms,
                    tokens_per_second=result.tokens_per_second,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    response_preview=result.response_text[:500],
                    error_message=result.error_message,
                    inter_chunk_ms_avg=result.inter_chunk_ms_avg or None,
                    inter_chunk_ms_p95=result.inter_chunk_ms_p95 or None,
                    total_chars=result.total_chars or None,
                    total_words=result.total_words or None,
                )
                yield {"event": "run_metric", "data": event.model_dump_json()}

            await self.benchmark_repo.save_suite(suite_id, "multi", provider_id, model,
                                       prompt, num_runs, 1, max_tokens, temperature, results)
            stats = compute_suite_stats([self._result_to_dict(r) for r in results])
            complete = SuiteCompleteEvent(
                suite_id=suite_id, status="completed",
                total_runs=num_runs,
                successful_runs=sum(1 for r in results if r.status == "success"),
                **stats,
            )
            yield {"event": "suite_complete", "data": complete.model_dump_json()}

        finally:
            await client.close()

    # --- Concurrent (Async parallel with semaphore) ---

    async def run_concurrent(
        self, provider_id: str, model: str, prompt: str,
        num_runs: int, concurrency: int, max_tokens: int, temperature: float,
    ) -> AsyncGenerator[dict, None]:
        """Fire num_runs requests with max `concurrency` in flight at once."""
        suite_id = str(uuid.uuid4())
        client = await self._get_client(provider_id)
        semaphore = asyncio.Semaphore(concurrency)
        queue: asyncio.Queue[dict | None] = asyncio.Queue()
        results: list[RunResult] = []

        async def _worker(run_num: int):
            async with semaphore:
                cumulative_chars = 0
                cumulative_words = 0

                async def tick_cb(elapsed: float, phase: str):
                    ev = TickEvent(suite_id=suite_id, run_number=run_num,
                                   elapsed_seconds=elapsed, phase=phase)
                    await queue.put({"event": "tick", "data": ev.model_dump_json()})

                async def chunk_cb(idx: int, text: str, elapsed_ms: float, inter_ms: float):
                    nonlocal cumulative_chars, cumulative_words
                    cumulative_chars += len(text)
                    cumulative_words += len(text.split())
                    ev = TokenChunkEvent(
                        suite_id=suite_id, run_number=run_num,
                        chunk_index=idx, text=text,
                        elapsed_ms=elapsed_ms, inter_chunk_ms=inter_ms,
                        cumulative_chars=cumulative_chars,
                        cumulative_words=cumulative_words,
                    )
                    await queue.put({"event": "token_chunk", "data": ev.model_dump_json()})

                result = await client.benchmark_streaming(
                    model, prompt, max_tokens, temperature,
                    tick_callback=tick_cb, chunk_callback=chunk_cb,
                )
                results.append(result)
                event = RunMetricEvent(
                    suite_id=suite_id, run_number=run_num,
                    provider_id=provider_id, model=model,
                    status=result.status, ttfb_ms=result.ttfb_ms,
                    total_latency_ms=result.total_latency_ms,
                    tokens_per_second=result.tokens_per_second,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    response_preview=result.response_text[:500],
                    error_message=result.error_message,
                    inter_chunk_ms_avg=result.inter_chunk_ms_avg or None,
                    inter_chunk_ms_p95=result.inter_chunk_ms_p95 or None,
                    total_chars=result.total_chars or None,
                    total_words=result.total_words or None,
                )
                await queue.put({"event": "run_metric", "data": event.model_dump_json()})

        async def _run_all():
            tasks = [asyncio.create_task(_worker(i + 1)) for i in range(num_runs)]
            await asyncio.gather(*tasks, return_exceptions=True)
            await queue.put(None)  # Sentinel

        runner_task = asyncio.create_task(_run_all())

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item

            await runner_task

            await self.benchmark_repo.save_suite(suite_id, "concurrent", provider_id, model,
                                       prompt, num_runs, concurrency, max_tokens, temperature, results)
            stats = compute_suite_stats([self._result_to_dict(r) for r in results])
            complete = SuiteCompleteEvent(
                suite_id=suite_id, status="completed",
                total_runs=num_runs,
                successful_runs=sum(1 for r in results if r.status == "success"),
                **stats,
            )
            yield {"event": "suite_complete", "data": complete.model_dump_json()}

        finally:
            await client.close()

    # --- Cross-Provider Comparison ---

    async def run_comparison(
        self, providers_models: list[dict], prompt: str,
        num_runs: int, max_tokens: int, temperature: float,
    ) -> AsyncGenerator[dict, None]:
        """Run same prompt across multiple providers for head-to-head comparison."""
        suite_id = str(uuid.uuid4())
        all_results: list[RunResult] = []
        clients: list[LLMClient] = []

        try:
            for pm in providers_models:
                pid, model = pm["provider_id"], pm["model"]
                client = await self._get_client(pid)
                clients.append(client)

                for i in range(num_runs):
                    run_number = i + 1
                    result = await client.benchmark_streaming(
                        model, prompt, max_tokens, temperature,
                    )
                    all_results.append(result)

                    event = RunMetricEvent(
                        suite_id=suite_id, run_number=run_number,
                        provider_id=pid, model=model,
                        status=result.status, ttfb_ms=result.ttfb_ms,
                        total_latency_ms=result.total_latency_ms,
                        tokens_per_second=result.tokens_per_second,
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                        response_preview=result.response_text[:500],
                        error_message=result.error_message,
                        inter_chunk_ms_avg=result.inter_chunk_ms_avg or None,
                        inter_chunk_ms_p95=result.inter_chunk_ms_p95 or None,
                        total_chars=result.total_chars or None,
                        total_words=result.total_words or None,
                    )
                    yield {"event": "run_metric", "data": event.model_dump_json()}

            await self.benchmark_repo.save_suite(suite_id, "comparison", "multi", "multi",
                                       prompt, num_runs * len(providers_models),
                                       1, max_tokens, temperature, all_results)
            stats = compute_suite_stats([self._result_to_dict(r) for r in all_results])
            complete = SuiteCompleteEvent(
                suite_id=suite_id, status="completed",
                total_runs=len(all_results),
                successful_runs=sum(1 for r in all_results if r.status == "success"),
                **stats,
            )
            yield {"event": "suite_complete", "data": complete.model_dump_json()}

        finally:
            for c in clients:
                await c.close()



    @staticmethod
    def _result_to_dict(r: RunResult) -> dict:
        return {
            "status": r.status, "ttfb_ms": r.ttfb_ms,
            "total_latency_ms": r.total_latency_ms,
            "tokens_per_second": r.tokens_per_second,
            "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
            "inter_chunk_ms_avg": r.inter_chunk_ms_avg,
            "total_chars": r.total_chars,
        }
