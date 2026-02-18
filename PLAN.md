# LLM API Benchmark Tester — Implementation Plan

## Context

Build a professional FastAPI application to benchmark LLM APIs that support OpenAI-compatible endpoints. Supported providers: **GLM (ZhipuAI), Kimi (Moonshot), MiniMax, DeepSeek, OpenAI, OpenRouter**. The app measures latency, TTFB, tokens/second, and token counts in real-time with streaming SSE. Supports async concurrent testing and persists all results/history with SQLAlchemy + SQLite. Uses **international API endpoints** and a **vanilla HTML/JS dashboard** (no build step).

**Project directory:** `/home/mac/Desktop/projects/ai-ml/api-tester`

---

## 1. Project Structure

```
api-tester/
├── pyproject.toml
├── .env                          # API keys (gitignored)
├── .env.example                  # Template
├── .gitignore
├── app/
│   ├── __init__.py
│   ├── main.py                   # FastAPI app + lifespan
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py             # Settings + provider configs
│   ├── models/
│   │   ├── __init__.py
│   │   ├── database.py           # Async engine + session
│   │   └── tables.py             # SQLAlchemy ORM models
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── benchmark.py          # Pydantic request/response schemas
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm_client.py         # OpenAI-compatible streaming client
│   │   ├── benchmark_runner.py   # Benchmark orchestration (single/multi/concurrent/compare)
│   │   └── statistics.py         # Aggregation: avg, P50, P95, P99
│   ├── routes/
│   │   ├── __init__.py           # Router aggregation
│   │   ├── benchmarks.py         # Benchmark endpoints (SSE streaming)
│   │   ├── providers.py          # Provider listing endpoint
│   │   └── history.py            # History/results endpoints
│   └── static/
│       ├── index.html            # Dashboard UI
│       ├── css/
│       │   └── style.css
│       └── js/
│           ├── app.js            # Main dashboard logic
│           ├── api.js            # SSE + fetch helpers
│           └── charts.js         # Chart.js rendering
└── tests/
    ├── __init__.py
    ├── conftest.py
    └── test_benchmarks.py
```

---

## 2. Dependencies

**File: `pyproject.toml`**

```toml
[project]
name = "api-tester"
version = "0.1.0"
description = "LLM API Benchmark Tester"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "pydantic>=2.10.0",
    "pydantic-settings>=2.6.0",
    "sse-starlette>=2.1.0",
    "httpx>=0.28.0",
    "openai>=1.60.0",
    "sqlalchemy>=2.0.0",
    "aiosqlite>=0.20.0",
    "python-dotenv>=1.0.0",
    "python-multipart>=0.0.20",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24.0",
    "ruff",
]
```

---

## 3. File-by-File Implementation

### 3.1 `app/core/config.py` — Settings + Provider Registry

All providers configured as data — adding a new provider = adding a dict entry, zero code changes.

```python
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import Optional


class ProviderConfig(BaseModel):
    """Configuration for an OpenAI-compatible LLM provider."""
    name: str
    base_url: str
    api_key: str = ""
    models: list[str] = []
    default_model: str = ""
    supports_stream_usage: bool = True  # Some providers don't support stream_options
    min_temperature: float = 0.0  # MiniMax requires > 0.0


class Settings(BaseSettings):
    app_title: str = "LLM API Benchmark Tester"
    app_version: str = "0.1.0"
    database_url: str = "sqlite+aiosqlite:///./data/benchmarks.db"

    # Provider API keys (from .env)
    glm_api_key: str = ""
    kimi_api_key: str = ""
    minimax_api_key: str = ""
    deepseek_api_key: str = ""
    openai_api_key: str = ""
    openrouter_api_key: str = ""

    # Benchmark defaults
    default_max_tokens: int = 512
    default_temperature: float = 0.7
    default_runs: int = 3
    max_concurrency: int = 10
    request_timeout: float = 120.0

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    def get_providers(self) -> dict[str, ProviderConfig]:
        """Build provider registry from settings. Only providers with API keys are active."""
        providers = {}
        if self.glm_api_key:
            providers["glm"] = ProviderConfig(
                name="GLM (ZhipuAI)",
                base_url="https://api.z.ai/api/paas/v4/",
                api_key=self.glm_api_key,
                models=["glm-4-flash", "glm-4-plus", "glm-4"],
                default_model="glm-4-flash",
            )
        if self.kimi_api_key:
            providers["kimi"] = ProviderConfig(
                name="Kimi (Moonshot)",
                base_url="https://api.moonshot.ai/v1",
                api_key=self.kimi_api_key,
                models=["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
                default_model="moonshot-v1-8k",
            )
        if self.minimax_api_key:
            providers["minimax"] = ProviderConfig(
                name="MiniMax",
                base_url="https://api.minimax.io/v1",
                api_key=self.minimax_api_key,
                models=["MiniMax-M2.5", "MiniMax-M2.5-highspeed", "MiniMax-M2"],
                default_model="MiniMax-M2.5",
                min_temperature=0.01,  # MiniMax rejects 0.0
            )
        if self.deepseek_api_key:
            providers["deepseek"] = ProviderConfig(
                name="DeepSeek",
                base_url="https://api.deepseek.com/v1",
                api_key=self.deepseek_api_key,
                models=["deepseek-chat", "deepseek-reasoner"],
                default_model="deepseek-chat",
            )
        if self.openai_api_key:
            providers["openai"] = ProviderConfig(
                name="OpenAI",
                base_url="https://api.openai.com/v1",
                api_key=self.openai_api_key,
                models=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
                default_model="gpt-4o-mini",
            )
        if self.openrouter_api_key:
            providers["openrouter"] = ProviderConfig(
                name="OpenRouter",
                base_url="https://openrouter.ai/api/v1",
                api_key=self.openrouter_api_key,
                models=["openai/gpt-4o", "anthropic/claude-3.5-sonnet", "meta-llama/llama-3.1-70b-instruct"],
                default_model="openai/gpt-4o",
                supports_stream_usage=False,
            )
        return providers


settings = Settings()
```

---

### 3.2 `app/models/database.py` — Async SQLAlchemy Engine

```python
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from contextlib import asynccontextmanager
from app.core.config import settings
import os

# Ensure data directory exists
os.makedirs(os.path.dirname(settings.database_url.replace("sqlite+aiosqlite:///", "")), exist_ok=True)

engine = create_async_engine(settings.database_url, echo=False)
async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    from app.models.tables import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


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
```

---

### 3.3 `app/models/tables.py` — SQLAlchemy ORM Models

Two tables: `benchmark_suites` (a test session) and `benchmark_runs` (individual measurements).

```python
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, ForeignKey, Enum as SAEnum
from sqlalchemy.orm import DeclarativeBase, relationship
from datetime import datetime, timezone
import enum


class Base(DeclarativeBase):
    pass


class SuiteStatus(str, enum.Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class BenchmarkSuite(Base):
    """A benchmark session — groups multiple runs together."""
    __tablename__ = "benchmark_suites"

    id = Column(String(36), primary_key=True)             # UUID
    mode = Column(String(20), nullable=False)              # single|multi|concurrent|comparison
    provider_id = Column(String(50), nullable=False)       # "glm", "kimi", etc. or "multi"
    model = Column(String(100), nullable=False)
    prompt = Column(Text, nullable=False)
    num_runs = Column(Integer, default=1)
    concurrency = Column(Integer, default=1)
    max_tokens = Column(Integer, default=512)
    temperature = Column(Float, default=0.7)
    status = Column(SAEnum(SuiteStatus), default=SuiteStatus.RUNNING)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)

    # Aggregated stats (filled on completion)
    avg_ttfb_ms = Column(Float, nullable=True)
    avg_tps = Column(Float, nullable=True)
    avg_latency_ms = Column(Float, nullable=True)
    p50_ttfb_ms = Column(Float, nullable=True)
    p95_ttfb_ms = Column(Float, nullable=True)
    p99_ttfb_ms = Column(Float, nullable=True)
    total_input_tokens = Column(Integer, nullable=True)
    total_output_tokens = Column(Integer, nullable=True)
    error_count = Column(Integer, default=0)

    runs = relationship("BenchmarkRun", back_populates="suite", cascade="all, delete-orphan")


class BenchmarkRun(Base):
    """A single API call measurement."""
    __tablename__ = "benchmark_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    suite_id = Column(String(36), ForeignKey("benchmark_suites.id"), nullable=False)
    run_number = Column(Integer, nullable=False)
    provider_id = Column(String(50), nullable=False)
    model = Column(String(100), nullable=False)
    status = Column(String(20), default="success")        # success|error
    error_message = Column(Text, nullable=True)

    # Timing metrics
    ttfb_ms = Column(Float, nullable=True)                 # Time to First Token
    total_latency_ms = Column(Float, nullable=True)        # End-to-end time
    tokens_per_second = Column(Float, nullable=True)       # Output TPS

    # Token counts
    input_tokens = Column(Integer, nullable=True)
    output_tokens = Column(Integer, nullable=True)
    total_tokens = Column(Integer, nullable=True)

    # Response
    response_preview = Column(String(500), nullable=True)  # First 500 chars
    recorded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    suite = relationship("BenchmarkSuite", back_populates="runs")
```

---

### 3.4 `app/schemas/benchmark.py` — Pydantic Schemas

```python
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# --- Requests ---

class SingleBenchmarkRequest(BaseModel):
    provider_id: str
    model: str
    prompt: str
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class MultiBenchmarkRequest(SingleBenchmarkRequest):
    num_runs: int = Field(default=3, ge=1, le=50)


class ConcurrentBenchmarkRequest(SingleBenchmarkRequest):
    num_runs: int = Field(default=10, ge=1, le=100)
    concurrency: int = Field(default=5, ge=1, le=20)


class ComparisonBenchmarkRequest(BaseModel):
    """Run the same prompt across multiple provider+model combos."""
    providers: list[dict]  # [{"provider_id": "glm", "model": "glm-4-flash"}, ...]
    prompt: str
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    num_runs: int = Field(default=3, ge=1, le=20)


# --- SSE Event Data ---

class RunMetricEvent(BaseModel):
    """Sent via SSE for each completed run."""
    suite_id: str
    run_number: int
    provider_id: str
    model: str
    status: str
    ttfb_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    response_preview: Optional[str] = None
    error_message: Optional[str] = None


class SuiteCompleteEvent(BaseModel):
    """Sent via SSE when entire suite finishes."""
    suite_id: str
    status: str
    total_runs: int
    successful_runs: int
    error_count: int
    avg_ttfb_ms: Optional[float] = None
    avg_tps: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    p50_ttfb_ms: Optional[float] = None
    p95_ttfb_ms: Optional[float] = None
    p99_ttfb_ms: Optional[float] = None
    total_input_tokens: Optional[int] = None
    total_output_tokens: Optional[int] = None


# --- History Responses ---

class SuiteSummary(BaseModel):
    id: str
    mode: str
    provider_id: str
    model: str
    prompt: str
    num_runs: int
    concurrency: int
    status: str
    created_at: datetime
    avg_ttfb_ms: Optional[float] = None
    avg_tps: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    error_count: int = 0

    model_config = {"from_attributes": True}


class RunDetail(BaseModel):
    id: int
    run_number: int
    provider_id: str
    model: str
    status: str
    ttfb_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    response_preview: Optional[str] = None
    error_message: Optional[str] = None
    recorded_at: datetime

    model_config = {"from_attributes": True}


class SuiteDetail(SuiteSummary):
    runs: list[RunDetail] = []
    p50_ttfb_ms: Optional[float] = None
    p95_ttfb_ms: Optional[float] = None
    p99_ttfb_ms: Optional[float] = None
    total_input_tokens: Optional[int] = None
    total_output_tokens: Optional[int] = None
```

---

### 3.5 `app/services/llm_client.py` — Streaming LLM Client

The core measurement engine. Uses `openai.AsyncOpenAI` pointed at each provider's base URL.

```python
import time
from dataclasses import dataclass, field
from openai import AsyncOpenAI
from app.core.config import ProviderConfig


@dataclass
class RunResult:
    """Raw measurement from a single API call."""
    provider_id: str
    model: str
    status: str = "success"
    error_message: str | None = None
    ttfb_ms: float = 0.0
    total_latency_ms: float = 0.0
    tokens_per_second: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    response_text: str = ""
    chunks_received: int = 0


class LLMClient:
    """Async OpenAI-compatible client with benchmark instrumentation."""

    def __init__(self, provider_id: str, provider: ProviderConfig):
        self.provider_id = provider_id
        self.provider = provider
        self.client = AsyncOpenAI(
            api_key=provider.api_key,
            base_url=provider.base_url,
            timeout=120.0,
        )

    async def benchmark_streaming(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> RunResult:
        """Execute one streaming API call and measure everything."""
        result = RunResult(provider_id=self.provider_id, model=model)

        # Respect provider min temperature
        effective_temp = max(temperature, self.provider.min_temperature)

        try:
            create_kwargs: dict = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": effective_temp,
                "stream": True,
            }
            if self.provider.supports_stream_usage:
                create_kwargs["stream_options"] = {"include_usage": True}

            t_start = time.perf_counter()
            first_token_time: float | None = None
            chunks: list[str] = []

            stream = await self.client.chat.completions.create(**create_kwargs)

            async for chunk in stream:
                # Usage chunk (final, no content)
                if chunk.usage:
                    result.input_tokens = chunk.usage.prompt_tokens or 0
                    result.output_tokens = chunk.usage.completion_tokens or 0
                    result.total_tokens = chunk.usage.total_tokens or 0
                    continue

                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    chunks.append(delta.content)
                    result.chunks_received += 1

            t_end = time.perf_counter()

            # Compute metrics
            result.response_text = "".join(chunks)
            result.total_latency_ms = (t_end - t_start) * 1000

            if first_token_time is not None:
                result.ttfb_ms = (first_token_time - t_start) * 1000
                decode_duration = t_end - first_token_time
                if decode_duration > 0 and result.chunks_received > 1:
                    # Use API-reported output tokens if available, else chunk count
                    token_count = result.output_tokens or result.chunks_received
                    result.tokens_per_second = token_count / decode_duration

        except Exception as e:
            result.status = "error"
            result.error_message = f"{type(e).__name__}: {str(e)[:300]}"
            result.total_latency_ms = (time.perf_counter() - t_start) * 1000

        return result

    async def close(self):
        await self.client.close()
```

---

### 3.6 `app/services/statistics.py` — Aggregation Math

```python
import statistics


def compute_percentile(values: list[float], pct: float) -> float | None:
    """Compute percentile (0-100) from sorted values."""
    if not values:
        return None
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * (pct / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_v):
        return sorted_v[f]
    return sorted_v[f] + (k - f) * (sorted_v[c] - sorted_v[f])


def compute_suite_stats(runs: list[dict]) -> dict:
    """Aggregate stats from a list of run dicts (successful only)."""
    successful = [r for r in runs if r.get("status") == "success"]
    if not successful:
        return {"error_count": len(runs)}

    ttfbs = [r["ttfb_ms"] for r in successful if r.get("ttfb_ms")]
    tps_vals = [r["tokens_per_second"] for r in successful if r.get("tokens_per_second")]
    latencies = [r["total_latency_ms"] for r in successful if r.get("total_latency_ms")]
    in_tokens = sum(r.get("input_tokens", 0) for r in successful)
    out_tokens = sum(r.get("output_tokens", 0) for r in successful)

    return {
        "avg_ttfb_ms": round(statistics.mean(ttfbs), 2) if ttfbs else None,
        "avg_tps": round(statistics.mean(tps_vals), 2) if tps_vals else None,
        "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else None,
        "p50_ttfb_ms": round(compute_percentile(ttfbs, 50), 2) if ttfbs else None,
        "p95_ttfb_ms": round(compute_percentile(ttfbs, 95), 2) if ttfbs else None,
        "p99_ttfb_ms": round(compute_percentile(ttfbs, 99), 2) if ttfbs else None,
        "total_input_tokens": in_tokens,
        "total_output_tokens": out_tokens,
        "error_count": len(runs) - len(successful),
    }
```

---

### 3.7 `app/services/benchmark_runner.py` — Orchestration (Core File)

Handles all 4 modes: single, multi-run, concurrent, cross-provider comparison. Yields SSE events in real-time via `asyncio.Queue`.

```python
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
            yield f"event: run_metric\ndata: {event.model_dump_json()}\n\n"

            await self._persist_suite(suite_id, "single", provider_id, model,
                                       prompt, 1, 1, max_tokens, temperature, [result])
            stats = compute_suite_stats([self._result_to_dict(result)])
            complete = SuiteCompleteEvent(
                suite_id=suite_id, status="completed",
                total_runs=1, successful_runs=1 if result.status == "success" else 0,
                **stats,
            )
            yield f"event: suite_complete\ndata: {complete.model_dump_json()}\n\n"

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
                yield f"event: run_metric\ndata: {event.model_dump_json()}\n\n"

            await self._persist_suite(suite_id, "multi", provider_id, model,
                                       prompt, num_runs, 1, max_tokens, temperature, results)
            stats = compute_suite_stats([self._result_to_dict(r) for r in results])
            complete = SuiteCompleteEvent(
                suite_id=suite_id, status="completed",
                total_runs=num_runs,
                successful_runs=sum(1 for r in results if r.status == "success"),
                **stats,
            )
            yield f"event: suite_complete\ndata: {complete.model_dump_json()}\n\n"

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
                yield f"event: run_metric\ndata: {event.model_dump_json()}\n\n"

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
            yield f"event: suite_complete\ndata: {complete.model_dump_json()}\n\n"

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
                    yield f"event: run_metric\ndata: {event.model_dump_json()}\n\n"

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
            yield f"event: suite_complete\ndata: {complete.model_dump_json()}\n\n"

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
```

---

### 3.8 `app/routes/benchmarks.py` — SSE Streaming Endpoints

```python
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
```

---

### 3.9 `app/routes/providers.py` — Provider Listing

```python
from fastapi import APIRouter
from app.core.config import settings

router = APIRouter(prefix="/providers", tags=["providers"])


@router.get("")
async def list_providers():
    """List all configured providers and their models."""
    providers = settings.get_providers()
    return {
        pid: {
            "name": p.name,
            "models": p.models,
            "default_model": p.default_model,
        }
        for pid, p in providers.items()
    }
```

---

### 3.10 `app/routes/history.py` — Results History (SQLAlchemy Queries)

```python
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select, desc
from app.models.database import get_db
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
):
    """List benchmark suites with optional filters."""
    async with get_db() as session:
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
async def get_suite(suite_id: str):
    """Get full suite detail with all individual run metrics."""
    async with get_db() as session:
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

        detail = SuiteDetail.model_validate(suite)
        detail.runs = [RunDetail.model_validate(r) for r in runs]
        return detail


@router.delete("/{suite_id}")
async def delete_suite(suite_id: str):
    """Delete a benchmark suite and all its runs."""
    async with get_db() as session:
        result = await session.execute(
            select(BenchmarkSuite).where(BenchmarkSuite.id == suite_id)
        )
        suite = result.scalar_one_or_none()
        if not suite:
            raise HTTPException(404, "Suite not found")
        await session.delete(suite)
        return {"status": "deleted", "suite_id": suite_id}
```

---

### 3.11 `app/routes/__init__.py` — Router Aggregation

```python
from fastapi import APIRouter
from app.routes.benchmarks import router as benchmarks_router
from app.routes.providers import router as providers_router
from app.routes.history import router as history_router


def create_api_router() -> APIRouter:
    api = APIRouter(prefix="/api/v1")
    api.include_router(benchmarks_router)
    api.include_router(providers_router)
    api.include_router(history_router)
    return api
```

---

### 3.12 `app/main.py` — Application Entry Point

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from app.core.config import settings
from app.models.database import init_db, close_db
from app.routes import create_api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    await close_db()


app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes first (take priority over static mount)
app.include_router(create_api_router())

# Static files (dashboard) — mounted last as catch-all
static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
```

---

### 3.13 `app/static/index.html` — Dashboard UI

Professional single-page dashboard with tabs for each benchmark mode, real-time results table, charts, and history view. Uses:
- **Vanilla JS** (no framework overhead)
- **Chart.js** via CDN for charts
- **CSS Grid** layout, clean dark theme

Key sections:
- Provider/model selector (populated from `/api/v1/providers`)
- Benchmark mode tabs: Single | Multi-Run | Concurrent | Comparison
- Real-time results table (populated via SSE)
- Summary stats cards (TTFB, TPS, Latency, Tokens)
- Line chart for TTFB trend, bar chart for TPS comparison
- History tab with sortable/filterable table

---

### 3.14 `app/static/js/api.js` — SSE Client Helper

```javascript
export async function streamBenchmark(endpoint, body, callbacks) {
  const response = await fetch(`/api/v1/benchmarks/${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    let eventType = "";
    for (const line of lines) {
      if (line.startsWith("event: ")) {
        eventType = line.slice(7).trim();
      } else if (line.startsWith("data: ")) {
        const data = JSON.parse(line.slice(6));
        if (eventType === "run_metric") callbacks.onRunMetric?.(data);
        if (eventType === "suite_complete") callbacks.onComplete?.(data);
      }
    }
  }
}

export async function fetchProviders() {
  const resp = await fetch("/api/v1/providers");
  return resp.json();
}

export async function fetchHistory(params = {}) {
  const qs = new URLSearchParams(params).toString();
  const resp = await fetch(`/api/v1/history?${qs}`);
  return resp.json();
}

export async function fetchSuiteDetail(suiteId) {
  const resp = await fetch(`/api/v1/history/${suiteId}`);
  return resp.json();
}
```

---

### 3.15 `app/static/js/charts.js` — Chart.js Rendering

```javascript
import Chart from "https://cdn.jsdelivr.net/npm/chart.js@4/+esm";

export function createTTFBChart(canvasId) {
  return new Chart(document.getElementById(canvasId), {
    type: "line",
    data: { labels: [], datasets: [{ label: "TTFB (ms)", data: [], borderColor: "#3b82f6", tension: 0.3 }] },
    options: { responsive: true, scales: { y: { beginAtZero: true, title: { display: true, text: "ms" } } } },
  });
}

export function createTPSChart(canvasId) {
  return new Chart(document.getElementById(canvasId), {
    type: "bar",
    data: { labels: [], datasets: [{ label: "Tokens/sec", data: [], backgroundColor: "#10b981" }] },
    options: { responsive: true, scales: { y: { beginAtZero: true, title: { display: true, text: "tok/s" } } } },
  });
}

export function addDataPoint(chart, label, value) {
  chart.data.labels.push(label);
  chart.data.datasets[0].data.push(value);
  chart.update("none"); // no animation for real-time perf
}
```

---

## 4. Environment Setup

**File: `.env.example`**
```env
# Fill in the API keys for providers you want to benchmark (leave blank to skip)
GLM_API_KEY=your-zhipuai-api-key
KIMI_API_KEY=your-moonshot-api-key
MINIMAX_API_KEY=your-minimax-api-key
DEEPSEEK_API_KEY=your-deepseek-api-key
OPENAI_API_KEY=your-openai-api-key
OPENROUTER_API_KEY=your-openrouter-api-key
```

**File: `.gitignore`**
```
.env
__pycache__/
*.pyc
data/
.venv/
*.db
```

---

## 5. Implementation Order

| Step | Files | Description |
|------|-------|-------------|
| 1 | `pyproject.toml`, `.env.example`, `.gitignore` | Project scaffolding |
| 2 | `app/core/config.py` | Settings + provider registry |
| 3 | `app/models/database.py`, `app/models/tables.py` | SQLAlchemy models + async engine |
| 4 | `app/schemas/benchmark.py` | All Pydantic schemas |
| 5 | `app/services/llm_client.py` | Streaming benchmark client |
| 6 | `app/services/statistics.py` | Aggregation math |
| 7 | `app/services/benchmark_runner.py` | Orchestration (core) |
| 8 | `app/routes/providers.py`, `app/routes/history.py`, `app/routes/benchmarks.py`, `app/routes/__init__.py` | All API routes |
| 9 | `app/main.py` | App entry point |
| 10 | `app/static/` (HTML + CSS + JS) | Dashboard frontend |
| 11 | `tests/` | Tests |
| 12 | Git init + first commit | Repository setup |

---

## 6. Verification Plan

1. **Install & run:**
   ```bash
   cd /home/mac/Desktop/projects/ai-ml/api-tester
   uv venv && source .venv/bin/activate
   uv pip install -e ".[dev]"
   cp .env.example .env  # fill in at least one API key
   uvicorn app.main:app --reload --port 8000
   ```

2. **Test providers endpoint:** `GET http://localhost:8000/api/v1/providers` — should list configured providers

3. **Test single benchmark:** POST to `/api/v1/benchmarks/single` with a valid provider — verify SSE events stream back with TTFB, TPS, tokens

4. **Test concurrent benchmark:** POST to `/api/v1/benchmarks/concurrent` with `concurrency: 5, num_runs: 10` — verify parallel execution and all metrics

5. **Test history:** `GET /api/v1/history` — verify suites are persisted after benchmarks; `GET /api/v1/history/{id}` — verify individual run details

6. **Dashboard:** Open `http://localhost:8000/` — verify provider dropdown populates, benchmark runs show real-time results, charts render, history loads

7. **Run tests:** `pytest tests/ -v`

---

## 7. API Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/providers` | List configured providers + models |
| `POST` | `/api/v1/benchmarks/single` | Single benchmark run (SSE stream) |
| `POST` | `/api/v1/benchmarks/multi` | Multi-run benchmark (SSE stream) |
| `POST` | `/api/v1/benchmarks/concurrent` | Concurrent benchmark (SSE stream) |
| `POST` | `/api/v1/benchmarks/comparison` | Cross-provider comparison (SSE stream) |
| `GET` | `/api/v1/history` | List past benchmark suites (filterable) |
| `GET` | `/api/v1/history/{suite_id}` | Get suite detail + all run metrics |
| `DELETE` | `/api/v1/history/{suite_id}` | Delete a benchmark suite |
| `GET` | `/` | Dashboard UI |
| `GET` | `/docs` | FastAPI Swagger docs |
