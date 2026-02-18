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
