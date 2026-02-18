import statistics


def compute_percentile(values: list[float], pct: float) -> float | None:
    """Compute percentile (0-100) from sorted values."""
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

    return {
        "avg_ttfb_ms": round(statistics.mean(ttfbs), 2) if ttfbs else None,
        "avg_tps": round(statistics.mean(tps_vals), 2) if tps_vals else None,
        "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else None,
        "p50_ttfb_ms": round(compute_percentile(ttfbs, 50), 2) if ttfbs else None,
        "p95_ttfb_ms": round(compute_percentile(ttfbs, 95), 2) if ttfbs else None,
        "p99_ttfb_ms": round(compute_percentile(ttfbs, 99), 2) if ttfbs else None,
        "total_input_tokens": in_tokens,
        "total_output_tokens": out_tokens,
        "error_count": len(runs) - len(successful),
    }
