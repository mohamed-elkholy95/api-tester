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
