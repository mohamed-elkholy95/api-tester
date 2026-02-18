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


# --- Speed Test Requests ---

class PingRequest(BaseModel):
    provider_id: str
    model: str


class StressTestRequest(BaseModel):
    provider_id: str
    model: str
    prompt: str
    max_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    concurrency_levels: list[int] = Field(default=[1, 2, 5, 10])
    runs_per_level: int = Field(default=3, ge=1, le=10)


class ColdStartRequest(BaseModel):
    provider_id: str
    model: str
    prompt: str = "Hello"
    max_tokens: int = Field(default=10, ge=1, le=100)
    num_cold_probes: int = Field(default=2, ge=1, le=5)
    gap_seconds: int = Field(default=30, ge=5, le=120)


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
    # New: output monitor fields
    inter_chunk_ms_avg: Optional[float] = None
    inter_chunk_ms_p95: Optional[float] = None
    total_chars: Optional[int] = None
    total_words: Optional[int] = None


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
    # New: output monitor aggregates
    avg_inter_chunk_ms: Optional[float] = None
    avg_total_chars: Optional[float] = None


class TickEvent(BaseModel):
    """Sent via SSE every 0.5s while a run is in-flight."""
    suite_id: str
    run_number: int
    elapsed_seconds: float
    phase: str  # "waiting_first_token" | "streaming"


class TokenChunkEvent(BaseModel):
    """Sent via SSE for every content chunk received from the API."""
    suite_id: str
    run_number: int
    chunk_index: int
    text: str
    elapsed_ms: float
    inter_chunk_ms: float
    cumulative_chars: int
    cumulative_words: int


# --- Speed Test SSE Events ---

class PingResult(BaseModel):
    provider_id: str
    model: str
    status: str
    ttfb_ms: Optional[float] = None
    round_trip_ms: Optional[float] = None
    health: str  # "ok" | "slow" | "error"


class StressLevelEvent(BaseModel):
    concurrency: int
    avg_tps: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    error_rate: float = 0.0
    successful_runs: int = 0


class StressCompleteEvent(BaseModel):
    levels: list[StressLevelEvent]
    peak_tps_concurrency: Optional[int] = None
    peak_tps: Optional[float] = None


class ColdStartProbeEvent(BaseModel):
    probe_number: int
    probe_type: str  # "cold" | "warm"
    ttfb_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None


class ColdStartCompleteEvent(BaseModel):
    avg_cold_ttfb_ms: Optional[float] = None
    avg_warm_ttfb_ms: Optional[float] = None
    cold_vs_warm_ratio: Optional[float] = None


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
    # New fields
    inter_chunk_ms_avg: Optional[float] = None
    inter_chunk_ms_p95: Optional[float] = None
    total_chars: Optional[int] = None
    total_words: Optional[int] = None

    model_config = {"from_attributes": True}


class SuiteDetail(SuiteSummary):
    runs: list[RunDetail] = []
    p50_ttfb_ms: Optional[float] = None
    p95_ttfb_ms: Optional[float] = None
    p99_ttfb_ms: Optional[float] = None
    total_input_tokens: Optional[int] = None
    total_output_tokens: Optional[int] = None
    avg_inter_chunk_ms: Optional[float] = None
    avg_total_chars: Optional[float] = None
