import statistics


def compute_percentile(values: list[float], pct: float) -> float | None:
    """Compute percentile (0-100) from a list of values."""
    if not values:
        return None
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * (pct / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_v):
        return sorted_v[f]
    return sorted_v[f] + (k - f) * (sorted_v[c] - sorted_v[f])


def compute_suite_stats(runs: list[dict]) -> dict:
    """Aggregate stats from a list of run dicts (successful only)."""
    successful = [r for r in runs if r.get("status") == "success"]
    if not successful:
        return {"error_count": len(runs)}

    ttfbs = [r["ttfb_ms"] for r in successful if r.get("ttfb_ms")]
    tps_vals = [r["tokens_per_second"] for r in successful if r.get("tokens_per_second")]
    latencies = [r["total_latency_ms"] for r in successful if r.get("total_latency_ms")]
    in_tokens = sum(r.get("input_tokens", 0) for r in successful)
    out_tokens = sum(r.get("output_tokens", 0) for r in successful)

    # New: inter-chunk and output stats
    inter_chunk_avgs = [r["inter_chunk_ms_avg"] for r in successful if r.get("inter_chunk_ms_avg")]
    total_chars_vals = [r["total_chars"] for r in successful if r.get("total_chars")]

    result = {
        "avg_ttfb_ms": round(statistics.mean(ttfbs), 2) if ttfbs else None,
        "avg_tps": round(statistics.mean(tps_vals), 2) if tps_vals else None,
        "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else None,
        "p50_ttfb_ms": round(compute_percentile(ttfbs, 50), 2) if ttfbs else None,
        "p95_ttfb_ms": round(compute_percentile(ttfbs, 95), 2) if ttfbs else None,
        "p99_ttfb_ms": round(compute_percentile(ttfbs, 99), 2) if ttfbs else None,
        "total_input_tokens": in_tokens,
        "total_output_tokens": out_tokens,
        "error_count": len(runs) - len(successful),
        # New fields
        "avg_inter_chunk_ms": round(statistics.mean(inter_chunk_avgs), 3) if inter_chunk_avgs else None,
        "avg_total_chars": round(statistics.mean(total_chars_vals), 1) if total_chars_vals else None,
    }
    return result


def compute_inter_chunk_stats(chunk_timings_ms: list[float]) -> dict:
    """Compute avg and P95 inter-chunk gap from per-chunk elapsed timestamps (ms)."""
    if len(chunk_timings_ms) < 2:
        return {"inter_chunk_ms_avg": None, "inter_chunk_ms_p95": None}
    gaps = [
        chunk_timings_ms[i] - chunk_timings_ms[i - 1]
        for i in range(1, len(chunk_timings_ms))
    ]
    return {
        "inter_chunk_ms_avg": round(statistics.mean(gaps), 3),
        "inter_chunk_ms_p95": round(compute_percentile(gaps, 95), 3),
    }
