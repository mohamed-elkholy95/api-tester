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
            default_headers=provider.extra_headers or None,
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

        t_start = time.perf_counter()

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
