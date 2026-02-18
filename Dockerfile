# ── Stage 1: dependency builder ──────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency manifests first (layer-cache friendly)
COPY pyproject.toml .

# Install into an isolated prefix so we can copy just the site-packages
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir ".[dev]" 2>/dev/null || \
    pip install --prefix=/install --no-cache-dir .

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="LLM API Benchmark Tester" \
      org.opencontainers.image.description="Professional benchmarking tool for LLM APIs" \
      org.opencontainers.image.source="https://github.com/mohamed-elkholy95/api-tester" \
      org.opencontainers.image.licenses="MIT"

# Non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY app/ ./app/
COPY pyproject.toml .

# Persistent data directory (SQLite DB lives here)
RUN mkdir -p /app/data && chown -R appuser:appuser /app

USER appuser

# Expose the application port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/providers')" || exit 1

# Run with production settings (no --reload)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
