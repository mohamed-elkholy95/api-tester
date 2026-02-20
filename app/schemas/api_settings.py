"""Pydantic schemas for provider settings API endpoints."""
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class ProviderSettingCreate(BaseModel):
    """Body for POST /settings/providers — create or update a provider."""

    id: str = Field(..., min_length=1, max_length=100, description="Stable provider ID (e.g. 'openrouter')")
    name: str = Field(..., min_length=1, max_length=200, description="Display name")
    base_url: str = Field(..., min_length=1, max_length=500, description="OpenAI-compatible base URL")
    api_key: str = Field(default="", max_length=500, description="API key (stored in plaintext for MVP)")
    models: list[str] = Field(default_factory=list, description="List of model IDs")
    default_model: str = Field(default="", max_length=200)
    supports_stream_usage: bool = Field(default=True, description="Whether to pass stream_options.include_usage")
    min_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    extra_headers: dict[str, str] = Field(default_factory=dict, description="Extra HTTP headers")

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        # Only allow URL-safe characters
        if not all(c.isalnum() or c in "-_." for c in v):
            raise ValueError("id must contain only alphanumeric characters, hyphens, underscores, or dots")
        return v.lower()


class ProviderSettingUpdate(BaseModel):
    """Body for PATCH /settings/providers/{id} — partial update."""

    name: Optional[str] = Field(None, min_length=1, max_length=200)
    base_url: Optional[str] = Field(None, min_length=1, max_length=500)
    api_key: Optional[str] = Field(None, max_length=500)
    models: Optional[list[str]] = None
    default_model: Optional[str] = Field(None, max_length=200)
    supports_stream_usage: Optional[bool] = None
    min_temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    extra_headers: Optional[dict[str, str]] = None


class ProviderSettingResponse(BaseModel):
    """Response shape for a single provider (api_key partially masked)."""

    id: str
    name: str
    base_url: str
    api_key_hint: str = Field(description="Last 4 chars of api_key, or empty string")
    models: list[str]
    default_model: str
    supports_stream_usage: bool
    min_temperature: float
    extra_headers: dict[str, str]
    source: str = "db"
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
