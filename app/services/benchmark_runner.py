import asyncio
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator

from app.core.config import settings, ProviderConfig
from app.services.llm_client import LLMClient, RunResult
from app.services.statistics import compute_suite_stats
from app.models.database import get_db
from app.models.tables import BenchmarkSuite, BenchmarkRun, SuiteStatus
from app.schemas.benchmark import RunMetricEvent, SuiteCompleteEvent


class BenchmarkRunner:

    def __init__(self):
        self.providers = settings.get_providers()

    def _get_client(self, provider_id: str) -> LLMClient:
        provider = self.providers[provider_id]
        return LLMClient(provider_id, provider)

    # --- Single Run ---

    async def run_single(
        self, provider_id: str, model: str, prompt: str,
        max_tokens: int, temperature: float,
    ) -> AsyncGenerator[str, None]:
        """Single run benchmark, yielding SSE events."""
        suite_id = str(uuid.uuid4())
        client = self._get_client(provider_id)

        try:
            result = await client.benchmark_streaming(model, prompt, max_tokens, temperature)

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
            )
            yield {"event": "run_metric", "data": event.model_dump_json()}

            await self._persist_suite(suite_id, "single", provider_id, model,
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
    ) -> AsyncGenerator[str, None]:
        suite_id = str(uuid.uuid4())
        client = self._get_client(provider_id)
        results: list[RunResult] = []

        try:
            for i in range(num_runs):
                result = await client.benchmark_streaming(model, prompt, max_tokens, temperature)
                results.append(result)

                event = RunMetricEvent(
                    suite_id=suite_id, run_number=i + 1,
                    provider_id=provider_id, model=model,
                    status=result.status, ttfb_ms=result.ttfb_ms,
                    total_latency_ms=result.total_latency_ms,
                    tokens_per_second=result.tokens_per_second,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    response_preview=result.response_text[:500],
                    error_message=result.error_message,
                )
                yield {"event": "run_metric", "data": event.model_dump_json()}

            await self._persist_suite(suite_id, "multi", provider_id, model,
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
    ) -> AsyncGenerator[str, None]:
        """Fire num_runs requests with max `concurrency` in flight at once."""
        suite_id = str(uuid.uuid4())
        client = self._get_client(provider_id)
        semaphore = asyncio.Semaphore(concurrency)
        queue: asyncio.Queue[RunMetricEvent | None] = asyncio.Queue()
        results: list[RunResult] = []

        async def _worker(run_num: int):
            async with semaphore:
                result = await client.benchmark_streaming(model, prompt, max_tokens, temperature)
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
                )
                await queue.put(event)

        async def _run_all():
            tasks = [asyncio.create_task(_worker(i + 1)) for i in range(num_runs)]
            await asyncio.gather(*tasks, return_exceptions=True)
            await queue.put(None)  # Sentinel to signal completion

        # Launch workers in background task
        runner_task = asyncio.create_task(_run_all())

        try:
            # Yield SSE events as workers complete (real-time)
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield {"event": "run_metric", "data": event.model_dump_json()}

            await runner_task

            await self._persist_suite(suite_id, "concurrent", provider_id, model,
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
    ) -> AsyncGenerator[str, None]:
        """Run same prompt across multiple providers for head-to-head comparison."""
        suite_id = str(uuid.uuid4())
        all_results: list[RunResult] = []
        clients: list[LLMClient] = []

        try:
            for pm in providers_models:
                pid, model = pm["provider_id"], pm["model"]
                client = self._get_client(pid)
                clients.append(client)

                for i in range(num_runs):
                    result = await client.benchmark_streaming(model, prompt, max_tokens, temperature)
                    all_results.append(result)

                    event = RunMetricEvent(
                        suite_id=suite_id, run_number=i + 1,
                        provider_id=pid, model=model,
                        status=result.status, ttfb_ms=result.ttfb_ms,
                        total_latency_ms=result.total_latency_ms,
                        tokens_per_second=result.tokens_per_second,
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                        response_preview=result.response_text[:500],
                        error_message=result.error_message,
                    )
                    yield {"event": "run_metric", "data": event.model_dump_json()}

            await self._persist_suite(suite_id, "comparison", "multi", "multi",
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

    # --- Persistence (SQLAlchemy) ---

    async def _persist_suite(
        self, suite_id: str, mode: str, provider_id: str, model: str,
        prompt: str, num_runs: int, concurrency: int,
        max_tokens: int, temperature: float, results: list[RunResult],
    ):
        """Save suite + all run results to SQLite via SQLAlchemy."""
        stats = compute_suite_stats([self._result_to_dict(r) for r in results])

        suite = BenchmarkSuite(
            id=suite_id, mode=mode, provider_id=provider_id, model=model,
            prompt=prompt, num_runs=num_runs, concurrency=concurrency,
            max_tokens=max_tokens, temperature=temperature,
            status=SuiteStatus.COMPLETED,
            completed_at=datetime.now(timezone.utc),
            avg_ttfb_ms=stats.get("avg_ttfb_ms"),
            avg_tps=stats.get("avg_tps"),
            avg_latency_ms=stats.get("avg_latency_ms"),
            p50_ttfb_ms=stats.get("p50_ttfb_ms"),
            p95_ttfb_ms=stats.get("p95_ttfb_ms"),
            p99_ttfb_ms=stats.get("p99_ttfb_ms"),
            total_input_tokens=stats.get("total_input_tokens"),
            total_output_tokens=stats.get("total_output_tokens"),
            error_count=stats.get("error_count", 0),
        )

        runs = []
        for i, r in enumerate(results):
            runs.append(BenchmarkRun(
                suite_id=suite_id, run_number=i + 1,
                provider_id=r.provider_id, model=r.model,
                status=r.status, error_message=r.error_message,
                ttfb_ms=r.ttfb_ms, total_latency_ms=r.total_latency_ms,
                tokens_per_second=r.tokens_per_second,
                input_tokens=r.input_tokens, output_tokens=r.output_tokens,
                total_tokens=r.total_tokens,
                response_preview=r.response_text[:500] if r.response_text else None,
            ))

        async with get_db() as session:
            session.add(suite)
            session.add_all(runs)

    @staticmethod
    def _result_to_dict(r: RunResult) -> dict:
        return {
            "status": r.status, "ttfb_ms": r.ttfb_ms,
            "total_latency_ms": r.total_latency_ms,
            "tokens_per_second": r.tokens_per_second,
            "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
        }
