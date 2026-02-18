import asyncio
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable
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
    # --- New enhanced metrics ---
    chunk_timings_ms: list[float] = field(default_factory=list)  # elapsed_ms at each chunk
    inter_chunk_ms_avg: float = 0.0   # avg gap between consecutive chunks
    inter_chunk_ms_p95: float = 0.0   # P95 inter-chunk gap (jitter indicator)
    total_chars: int = 0
    total_words: int = 0


class LLMClient:
    """Async OpenAI-compatible client with benchmark instrumentation."""

    def __init__(self, provider_id: str, provider: ProviderConfig):
        self.provider_id = provider_id
        self.provider = provider
        self.client = AsyncOpenAI(
            api_key=provider.api_key,
            base_url=provider.base_url,
            timeout=120.0,
            default_headers=provider.extra_headers or None,
        )

    async def benchmark_streaming(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        tick_callback: Callable[[float, str], Awaitable[None]] | None = None,
        chunk_callback: Callable[[int, str, float, float], Awaitable[None]] | None = None,
    ) -> RunResult:
        """Execute one streaming API call and measure everything.

        Args:
            tick_callback: async fn(elapsed_seconds, phase) called every 0.5s while in-flight.
            chunk_callback: async fn(chunk_index, text, elapsed_ms, inter_chunk_ms) per chunk.
        """
        result = RunResult(provider_id=self.provider_id, model=model)

        # Respect provider min temperature
        effective_temp = max(temperature, self.provider.min_temperature)

        t_start = time.perf_counter()
        done_event = asyncio.Event()
        phase = "waiting_first_token"

        # --- Ticker task ---
        async def _ticker():
            while not done_event.is_set():
                elapsed = time.perf_counter() - t_start
                if tick_callback:
                    try:
                        await tick_callback(round(elapsed, 2), phase)
                    except Exception:
                        pass
                try:
                    await asyncio.wait_for(asyncio.shield(done_event.wait()), timeout=0.5)
                except asyncio.TimeoutError:
                    pass

        ticker_task = asyncio.create_task(_ticker()) if tick_callback else None

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

            first_token_time: float | None = None
            last_chunk_time: float | None = None
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
                if delta:
                    # Some providers (e.g. GLM reasoning models) put text in
                    # reasoning_content when delta.content is None/empty.
                    text = delta.content or getattr(delta, 'reasoning_content', None) or ''
                    if text:
                        now = time.perf_counter()
                        elapsed_ms = (now - t_start) * 1000

                        if first_token_time is None:
                            first_token_time = now
                            phase = "streaming"

                        inter_chunk_ms = (now - last_chunk_time) * 1000 if last_chunk_time else 0.0
                        last_chunk_time = now

                        chunks.append(text)
                        result.chunks_received += 1
                        result.chunk_timings_ms.append(elapsed_ms)

                        if chunk_callback:
                            try:
                                await chunk_callback(
                                    result.chunks_received,
                                    text,
                                    elapsed_ms,
                                    inter_chunk_ms,
                                )
                            except Exception:
                                pass


            t_end = time.perf_counter()

            # Compute metrics
            result.response_text = "".join(chunks)
            result.total_latency_ms = (t_end - t_start) * 1000
            result.total_chars = len(result.response_text)
            result.total_words = len(result.response_text.split()) if result.response_text else 0

            if first_token_time is not None:
                result.ttfb_ms = (first_token_time - t_start) * 1000
                decode_duration = t_end - first_token_time
                if decode_duration > 0 and result.chunks_received > 1:
                    token_count = result.output_tokens or result.chunks_received
                    result.tokens_per_second = token_count / decode_duration

            # Compute inter-chunk stats
            if len(result.chunk_timings_ms) >= 2:
                gaps = [
                    result.chunk_timings_ms[i] - result.chunk_timings_ms[i - 1]
                    for i in range(1, len(result.chunk_timings_ms))
                ]
                result.inter_chunk_ms_avg = round(sum(gaps) / len(gaps), 3)
                sorted_gaps = sorted(gaps)
                k = (len(sorted_gaps) - 1) * 0.95
                f = int(k)
                c = min(f + 1, len(sorted_gaps) - 1)
                result.inter_chunk_ms_p95 = round(
                    sorted_gaps[f] + (k - f) * (sorted_gaps[c] - sorted_gaps[f]), 3
                )

        except Exception as e:
            result.status = "error"
            result.error_message = f"{type(e).__name__}: {str(e)[:300]}"
            result.total_latency_ms = (time.perf_counter() - t_start) * 1000
        finally:
            done_event.set()
            if ticker_task:
                await ticker_task

        return result

    async def close(self):
        await self.client.close()
