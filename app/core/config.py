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
    extra_headers: dict[str, str] = {}  # Custom HTTP headers for specific providers


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
                name="GLM (ZhipuAI / Z.ai)",
                base_url="https://api.z.ai/api/coding/paas/v4/",
                api_key=self.glm_api_key,
                models=[
                    "glm-5",
                    "glm-4.7",
                    "glm-4.7-flash",
                    "glm-4.6",
                ],
                default_model="glm-5",
            )
        if self.kimi_api_key:
            providers["kimi"] = ProviderConfig(
                name="Kimi Code",
                base_url="https://api.kimi.com/coding/v1",
                api_key=self.kimi_api_key,
                models=[
                    "kimi-k2.5",
                    "kimi-k2-turbo-preview",
                    "kimi-k2-0905-preview",
                    "kimi-k2-thinking",
                    "kimi-k2-thinking-turbo",
                    "kimi-for-coding",
                ],
                default_model="kimi-k2.5",
                extra_headers={"User-Agent": "claude-code/1.0"},
            )
        if self.minimax_api_key:
            providers["minimax"] = ProviderConfig(
                name="MiniMax",
                base_url="https://api.minimax.io/v1",
                api_key=self.minimax_api_key,
                models=[
                    "MiniMax-M2.5",
                    "MiniMax-M2.5-Lightning",
                    "MiniMax-M2.1",
                    "MiniMax-M2",
                    "MiniMax-Text-01",
                ],
                default_model="MiniMax-M2.5",
                min_temperature=0.01,  # MiniMax rejects 0.0
            )
        if self.deepseek_api_key:
            providers["deepseek"] = ProviderConfig(
                name="DeepSeek",
                base_url="https://api.deepseek.com/v1",
                api_key=self.deepseek_api_key,
                models=[
                    "deepseek-chat",
                    "deepseek-reasoner",
                ],
                default_model="deepseek-chat",
            )
        if self.openai_api_key:
            providers["openai"] = ProviderConfig(
                name="OpenAI",
                base_url="https://api.openai.com/v1",
                api_key=self.openai_api_key,
                models=[
                    "gpt-5.2",
                    "gpt-5.1",
                    "gpt-4o",
                    "gpt-4o-mini",
                ],
                default_model="gpt-5.2",
            )
        if self.openrouter_api_key:
            providers["openrouter"] = ProviderConfig(
                name="OpenRouter",
                base_url="https://openrouter.ai/api/v1",
                api_key=self.openrouter_api_key,
                models=[
                    "anthropic/claude-opus-4.6",
                    "anthropic/claude-sonnet-4.5",
                    "openai/gpt-5.2",
                    "google/gemini-3-flash",
                    "deepseek/deepseek-v3.2",
                    "moonshot/kimi-k2.5",
                ],
                default_model="anthropic/claude-opus-4.6",
                supports_stream_usage=False,
            )
        return providers


settings = Settings()
