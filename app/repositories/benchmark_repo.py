from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.tables import BenchmarkSuite, BenchmarkRun, SuiteStatus
from app.services.llm_client import RunResult
from app.services.statistics import compute_suite_stats


class BenchmarkRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def save_suite(
        self, suite_id: str, mode: str, provider_id: str, model: str,
        prompt: str, num_runs: int, concurrency: int,
        max_tokens: int, temperature: float, results: list[RunResult],
    ) -> BenchmarkSuite:
        """Save a suite and its constituent runs to the database."""
        # Convert results to dictionaries for stats computation
        result_dicts = [self._result_to_dict(r) for r in results]
        stats = compute_suite_stats(result_dicts)

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
            avg_inter_chunk_ms=stats.get("avg_inter_chunk_ms"),
            avg_total_chars=stats.get("avg_total_chars"),
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
                inter_chunk_ms_avg=r.inter_chunk_ms_avg or None,
                inter_chunk_ms_p95=r.inter_chunk_ms_p95 or None,
                total_chars=r.total_chars or None,
                total_words=r.total_words or None,
            ))

        self.session.add(suite)
        self.session.add_all(runs)
        
        # Note: Do not commit here, so the session dependency handles the transaction lifecycle
        return suite

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
