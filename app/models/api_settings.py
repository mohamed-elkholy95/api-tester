"""SQLAlchemy model for user-saved LLM provider settings."""
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, String, Text

from app.models.base import Base


class ProviderSetting(Base):
    """Persisted provider configuration added/edited through the Settings UI."""

    __tablename__ = "provider_settings"

    # User-chosen stable identifier, e.g. "openrouter", "my-ollama"
    id = Column(String(100), primary_key=True)
    name = Column(String(200), nullable=False)
    base_url = Column(String(500), nullable=False)

    # Stored in plaintext for MVP (acceptable per plan review)
    api_key = Column(String(500), nullable=False, default="")

    # JSON array of model ID strings, e.g. '["gpt-4o", "gpt-4o-mini"]'
    models_json = Column(Text, nullable=False, default="[]")

    default_model = Column(String(200), nullable=False, default="")
    supports_stream_usage = Column(Boolean, nullable=False, default=True)
    min_temperature = Column(Float, nullable=False, default=0.0)

    # Optional JSON object of extra HTTP headers, e.g. '{"X-Title": "My App"}'
    extra_headers_json = Column(Text, nullable=False, default="{}")

    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    updated_at = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
