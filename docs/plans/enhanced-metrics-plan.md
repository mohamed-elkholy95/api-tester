# Enhanced Metrics Plan — Seconds Monitor, Output Monitor & Speed/Responsiveness Tests

**Date:** 2026-02-17  
**Project:** LLM API Benchmark Tester (`/home/mac/Desktop/projects/ai-ml/api-tester`)  
**Status:** Draft

---

## 1. Overview

The current benchmark engine captures TTFB, total latency, tokens/second, and token counts per run.
This plan extends the system with three complementary enhancement areas:

| Area | What it adds |
|---|---|
| **Seconds Monitor** | Real-time elapsed-time ticker streamed via SSE while a request is in-flight |
| **Output Monitor** | Live token-by-token output streaming with per-chunk timing and cumulative word/char counts |
| **Speed & Responsiveness Tests** | New dedicated test modes: throughput stress test, cold-start latency probe, and API health/ping check |

---

## 2. Current State

### Existing Metrics (per `BenchmarkRun`)

| Metric | Field | Notes |
|---|---|---|
| Time to First Token | `ttfb_ms` | ms from request send → first content chunk |
| Total Latency | `total_latency_ms` | ms from request send → stream end |
| Tokens per Second | `tokens_per_second` | output tokens / decode duration |
| Input Tokens | `input_tokens` | from API usage chunk |
| Output Tokens | `output_tokens` | from API usage chunk |
| Total Tokens | `total_tokens` | input + output |
| Response Preview | `response_preview` | first 500 chars |

### Existing Statistics (`statistics.py`)

- `avg_ttfb_ms`, `avg_tps`, `avg_latency_ms`
- `p50_ttfb_ms`, `p95_ttfb_ms`, `p99_ttfb_ms`
- `total_input_tokens`, `total_output_tokens`

### Existing SSE Event Types

- `run_metric` — emitted after each run completes
- `suite_complete` — emitted when all runs finish

---

## 3. Enhancement Area 1 — Seconds Monitor

### Goal
Stream a live elapsed-time counter (in seconds) to the UI while a benchmark request is in-flight, so users can see real-time progress rather than waiting for the run to complete.

### Design

#### 3.1 New SSE Event: `tick`

Emitted every **0.5 seconds** while a run is active. Carries:

```json
{
  "event": "tick",
  "data": {
    "suite_id": "uuid",
    "run_number": 1,
    "elapsed_seconds": 3.5,
    "phase": "waiting_first_token"   // or "streaming"
  }
}
```

**Phases:**
- `waiting_first_token` — request sent, no content yet (TTFB window)
- `streaming` — first token received, content flowing

#### 3.2 Implementation in `llm_client.py`

Add a `benchmark_streaming_with_ticks()` method (or extend the existing one with a `tick_callback` parameter):

```python
async def benchmark_streaming(
    self, model, prompt, max_tokens, temperature,
    tick_callback: Callable[[float, str], Awaitable[None]] | None = None
) -> RunResult:
    ...
    t_start = time.perf_counter()
    phase = "waiting_first_token"

    async def _ticker():
        while not done_event.is_set():
            elapsed = time.perf_counter() - t_start
            if tick_callback:
                await tick_callback(elapsed, phase)
            await asyncio.sleep(0.5)

    ticker_task = asyncio.create_task(_ticker())
    ...
    # When first token arrives: phase = "streaming"
    ...
    done_event.set()
    await ticker_task
```

#### 3.3 New SSE Event Schema (`schemas/benchmark.py`)

```python
class TickEvent(BaseModel):
    suite_id: str
    run_number: int
    elapsed_seconds: float
    phase: str  # "waiting_first_token" | "streaming"
```

#### 3.4 UI Integration (`static/js/app.js`)

- Listen for `tick` SSE events
- Display a live `⏱ 3.5s` counter in the run card
- Color-code phase: orange = waiting, green = streaming

---

## 4. Enhancement Area 2 — Output Monitor

### Goal
Stream each token chunk as it arrives with per-chunk timing, enabling a live "typewriter" output view and fine-grained inter-token latency analysis.

### Design

#### 4.1 New SSE Event: `token_chunk`

Emitted for **every content chunk** received from the API:

```json
{
  "event": "token_chunk",
  "data": {
    "suite_id": "uuid",
    "run_number": 1,
    "chunk_index": 42,
    "text": " world",
    "elapsed_ms": 1234.5,
    "inter_chunk_ms": 18.2,
    "cumulative_chars": 312,
    "cumulative_words": 54
  }
}
```

#### 4.2 New Metrics Captured in `RunResult`

Extend `RunResult` dataclass in `llm_client.py`:

```python
@dataclass
class RunResult:
    ...
    # New fields
    chunk_timings_ms: list[float] = field(default_factory=list)  # elapsed_ms per chunk
    inter_chunk_ms_avg: float = 0.0   # avg inter-chunk gap
    inter_chunk_ms_p95: float = 0.0   # P95 inter-chunk gap (jitter indicator)
    total_chars: int = 0
    total_words: int = 0
```

#### 4.3 New DB Columns (`models/tables.py`)

Add to `BenchmarkRun`:

```python
inter_chunk_ms_avg = Column(Float, nullable=True)   # avg gap between chunks
inter_chunk_ms_p95 = Column(Float, nullable=True)   # P95 chunk gap (jitter)
total_chars        = Column(Integer, nullable=True)
total_words        = Column(Integer, nullable=True)
```

Add to `BenchmarkSuite`:

```python
avg_inter_chunk_ms = Column(Float, nullable=True)
avg_total_chars    = Column(Float, nullable=True)
```

#### 4.4 New Statistics (`statistics.py`)

```python
def compute_inter_chunk_stats(chunk_timings: list[float]) -> dict:
    """Compute avg and P95 inter-chunk gap from a list of per-chunk elapsed timestamps."""
    if len(chunk_timings) < 2:
        return {"inter_chunk_ms_avg": None, "inter_chunk_ms_p95": None}
    gaps = [chunk_timings[i] - chunk_timings[i-1] for i in range(1, len(chunk_timings))]
    return {
        "inter_chunk_ms_avg": round(statistics.mean(gaps), 3),
        "inter_chunk_ms_p95": round(compute_percentile(gaps, 95), 3),
    }
```

#### 4.5 UI Integration

- Add a collapsible **"Live Output"** panel per run
- Render tokens as they arrive (typewriter effect)
- Show a mini sparkline of inter-chunk gaps (using Chart.js)
- Display `total_chars`, `total_words`, `inter_chunk_ms_avg` in the run summary card

---

## 5. Enhancement Area 3 — Speed & Responsiveness Tests

Three new dedicated test modes, each accessible via a new route and UI tab.

### 5.1 API Health / Ping Check

**Purpose:** Verify an API endpoint is reachable and measure cold-start overhead before any benchmark.

**New Route:** `POST /api/benchmarks/ping`

**Request:**
```json
{ "provider_id": "deepseek", "model": "deepseek-chat" }
```

**What it does:**
1. Sends a minimal prompt (`"Hi"`, `max_tokens=1`) with streaming
2. Measures: connection time, TTFB, total round-trip
3. Returns a simple health status: `ok | slow | error`

**Response:**
```json
{
  "provider_id": "deepseek",
  "model": "deepseek-chat",
  "status": "ok",
  "ttfb_ms": 312.4,
  "round_trip_ms": 489.1,
  "health": "ok"   // ok (<500ms TTFB), slow (500-2000ms), error
}
```

**New Schema:**
```python
class PingRequest(BaseModel):
    provider_id: str
    model: str

class PingResult(BaseModel):
    provider_id: str
    model: str
    status: str
    ttfb_ms: float | None
    round_trip_ms: float | None
    health: str  # "ok" | "slow" | "error"
```

---

### 5.2 Throughput Stress Test

**Purpose:** Ramp up concurrent requests to find the provider's throughput ceiling and measure how TPS degrades under load.

**New Route:** `GET /api/benchmarks/stress` (SSE stream)

**Request:**
```json
{
  "provider_id": "glm",
  "model": "glm-4-flash",
  "prompt": "...",
  "max_tokens": 256,
  "temperature": 0.7,
  "concurrency_levels": [1, 2, 5, 10],
  "runs_per_level": 3
}
```

**What it does:**
- For each concurrency level, runs `runs_per_level` concurrent requests
- Emits a `stress_level_complete` SSE event after each level with aggregated TPS, latency, error rate
- Emits a `stress_complete` event with the full ramp-up table

**New SSE Events:**

```json
// After each concurrency level
{
  "event": "stress_level_complete",
  "data": {
    "concurrency": 5,
    "avg_tps": 42.1,
    "avg_latency_ms": 1823.4,
    "error_rate": 0.0,
    "successful_runs": 3
  }
}

// After all levels
{
  "event": "stress_complete",
  "data": {
    "levels": [
      {"concurrency": 1, "avg_tps": 18.2, "avg_latency_ms": 890.1, "error_rate": 0.0},
      {"concurrency": 2, "avg_tps": 34.5, "avg_latency_ms": 1102.3, "error_rate": 0.0},
      {"concurrency": 5, "avg_tps": 42.1, "avg_latency_ms": 1823.4, "error_rate": 0.0},
      {"concurrency": 10, "avg_tps": 38.7, "avg_latency_ms": 3241.8, "error_rate": 0.1}
    ],
    "peak_tps_concurrency": 5,
    "peak_tps": 42.1
  }
}
```

**New Schema:**
```python
class StressTestRequest(BaseModel):
    provider_id: str
    model: str
    prompt: str
    max_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    concurrency_levels: list[int] = Field(default=[1, 2, 5, 10])
    runs_per_level: int = Field(default=3, ge=1, le=10)
```

---

### 5.3 Responsiveness / Cold-Start Probe

**Purpose:** Measure how quickly a provider responds to the *very first* request after a period of inactivity (cold-start latency), versus warm subsequent requests.

**New Route:** `POST /api/benchmarks/cold-start`

**What it does:**
1. Sends N requests sequentially with a configurable **gap** between them (e.g. 30s)
2. Labels each as `cold` (first after gap) or `warm` (immediate follow-up)
3. Compares TTFB between cold and warm runs

**Request:**
```json
{
  "provider_id": "kimi",
  "model": "moonshot-v1-8k",
  "prompt": "Hello",
  "max_tokens": 10,
  "num_cold_probes": 2,
  "gap_seconds": 30
}
```

**New SSE Events:**

```json
// After each probe
{
  "event": "cold_start_probe",
  "data": {
    "probe_number": 1,
    "type": "cold",
    "ttfb_ms": 1842.3,
    "total_latency_ms": 2100.1
  }
}

// Summary
{
  "event": "cold_start_complete",
  "data": {
    "avg_cold_ttfb_ms": 1842.3,
    "avg_warm_ttfb_ms": 312.4,
    "cold_vs_warm_ratio": 5.9
  }
}
```

---

## 6. File Change Summary

### Modified Files

| File | Changes |
|---|---|
| `app/services/llm_client.py` | Add `tick_callback` param, `chunk_timings_ms`, `inter_chunk_ms_avg/p95`, `total_chars/words` to `RunResult` |
| `app/services/statistics.py` | Add `compute_inter_chunk_stats()` |
| `app/services/benchmark_runner.py` | Wire tick events + chunk events into existing run methods; add `run_stress()`, `run_cold_start()` |
| `app/schemas/benchmark.py` | Add `TickEvent`, `TokenChunkEvent`, `PingRequest/Result`, `StressTestRequest`, `ColdStartRequest` |
| `app/models/tables.py` | Add `inter_chunk_ms_avg`, `inter_chunk_ms_p95`, `total_chars`, `total_words` to `BenchmarkRun`; add `avg_inter_chunk_ms` to `BenchmarkSuite` |
| `app/routes/benchmarks.py` | Add `/ping`, `/stress`, `/cold-start` endpoints |
| `app/static/js/app.js` | Handle `tick`, `token_chunk`, `stress_level_complete`, `cold_start_probe` SSE events |
| `app/static/index.html` | Add "Speed Tests" tab; live output panel; seconds counter in run cards |
| `app/static/css/style.css` | Styles for ticker, live output panel, stress ramp chart |

### New Files

| File | Purpose |
|---|---|
| `app/services/speed_tests.py` | Isolated logic for ping, stress, cold-start orchestration |
| `tests/test_speed_tests.py` | Unit + integration tests for new test modes |
| `tests/test_metrics.py` | Tests for new metrics: inter-chunk stats, tick events, output monitor |

---

## 7. Database Migration

Since new columns are added to existing tables, a lightweight migration is needed.

**Approach:** Use SQLAlchemy's `text()` to run `ALTER TABLE` statements in the `init_db()` lifespan function, guarded by a `try/except` (SQLite ignores duplicate column errors with `OperationalError`).

```python
async def migrate_db():
    """Add new columns if they don't exist (idempotent)."""
    new_columns = [
        ("benchmark_runs", "inter_chunk_ms_avg", "FLOAT"),
        ("benchmark_runs", "inter_chunk_ms_p95", "FLOAT"),
        ("benchmark_runs", "total_chars", "INTEGER"),
        ("benchmark_runs", "total_words", "INTEGER"),
        ("benchmark_suites", "avg_inter_chunk_ms", "FLOAT"),
        ("benchmark_suites", "avg_total_chars", "FLOAT"),
    ]
    async with engine.begin() as conn:
        for table, col, col_type in new_columns:
            try:
                await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}"))
            except Exception:
                pass  # Column already exists
```

---

## 8. Implementation Order

```
Phase 1 — Foundation (no UI changes yet)
  1. Extend RunResult dataclass (llm_client.py)
  2. Add tick_callback + chunk event emission (llm_client.py)
  3. Add inter-chunk stats (statistics.py)
  4. Add new DB columns + migrate_db() (tables.py, database.py)
  5. Add new Pydantic schemas (schemas/benchmark.py)

Phase 2 — New Test Modes
  6. Implement speed_tests.py (ping, stress, cold-start)
  7. Add new routes (benchmarks.py)
  8. Wire tick + chunk events in benchmark_runner.py

Phase 3 — UI
  9. Handle new SSE events in app.js
  10. Add live output panel + seconds counter (index.html, app.js)
  11. Add Speed Tests tab (index.html, app.js)
  12. Style new components (style.css)

Phase 4 — Tests
  13. Write test_metrics.py
  14. Write test_speed_tests.py
```

---

## 9. Verification Plan

### Automated Tests

```bash
# From project root with venv active
cd /home/mac/Desktop/projects/ai-ml/api-tester
source .venv/bin/activate

# Run all tests
pytest tests/ -v

# Run only new metric tests
pytest tests/test_metrics.py -v

# Run only speed test tests
pytest tests/test_speed_tests.py -v
```

### Manual Verification (with server running)

```bash
# Server already running on port 8000
# Open browser to http://localhost:8000

# 1. Seconds Monitor
#    - Start any benchmark run
#    - Confirm ⏱ elapsed counter updates every ~0.5s in the run card
#    - Confirm phase changes from orange "waiting" to green "streaming" after first token

# 2. Output Monitor
#    - Start a single-run benchmark
#    - Expand the "Live Output" panel
#    - Confirm tokens appear in real-time (typewriter effect)
#    - Confirm inter-chunk latency sparkline appears after run completes

# 3. Ping / Health Check
#    - Navigate to "Speed Tests" tab
#    - Click "Ping" for a configured provider
#    - Confirm result shows ttfb_ms, round_trip_ms, and health badge (ok/slow/error)

# 4. Stress Test
#    - Set concurrency levels [1, 2, 5]
#    - Run stress test
#    - Confirm SSE events arrive per level
#    - Confirm final ramp-up chart renders

# 5. Cold-Start Probe
#    - Set gap_seconds=5 (for quick testing)
#    - Run cold-start probe
#    - Confirm cold vs warm TTFB comparison is displayed
```

### API Endpoint Smoke Tests (curl)

```bash
# Ping
curl -X POST http://localhost:8000/api/benchmarks/ping \
  -H "Content-Type: application/json" \
  -d '{"provider_id": "deepseek", "model": "deepseek-chat"}'

# Stress test (SSE)
curl -N -X POST http://localhost:8000/api/benchmarks/stress \
  -H "Content-Type: application/json" \
  -d '{"provider_id":"glm","model":"glm-4-flash","prompt":"Hi","concurrency_levels":[1,2],"runs_per_level":2}'

# Cold-start probe (SSE)
curl -N -X POST http://localhost:8000/api/benchmarks/cold-start \
  -H "Content-Type: application/json" \
  -d '{"provider_id":"kimi","model":"moonshot-v1-8k","prompt":"Hello","max_tokens":10,"num_cold_probes":1,"gap_seconds":5}'
```

---

## 10. Open Questions / Decisions Needed

1. **Tick frequency** — 0.5s proposed. Should this be configurable per-request?
2. **Token chunk SSE volume** — For long responses, `token_chunk` events could be hundreds per run. Should there be a `chunk_batch_size` to group N chunks per event to reduce SSE overhead?
3. **Stress test rate limiting** — Should the stress test respect a per-provider rate limit config to avoid getting banned?
4. **Cold-start gap** — Max `gap_seconds` should be capped (e.g. 120s) to prevent runaway requests. Agreed?
5. **DB storage of chunk timings** — Storing the full `chunk_timings_ms` list per run would require a JSON column or a new `chunk_events` table. Proposed: store only aggregated stats (`inter_chunk_ms_avg`, `p95`) in `BenchmarkRun`, not the raw list.
