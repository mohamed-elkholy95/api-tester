// api.js — SSE + fetch helpers

async function streamBenchmark(endpoint, body, callbacks) {
    const response = await fetch(`/api/v1/benchmarks/${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(error.detail || `HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        let eventType = "";
        for (const line of lines) {
            if (line.startsWith("event: ")) {
                eventType = line.slice(7).trim();
            } else if (line.startsWith("data: ")) {
                try {
                    const data = JSON.parse(line.slice(6));
                    if (eventType === "run_metric") callbacks.onRunMetric?.(data);
                    if (eventType === "suite_complete") callbacks.onComplete?.(data);
                    if (eventType === "tick") callbacks.onTick?.(data);
                    if (eventType === "token_chunk") callbacks.onTokenChunk?.(data);
                    if (eventType === "stress_level_complete") callbacks.onStressLevel?.(data);
                    if (eventType === "stress_complete") callbacks.onStressComplete?.(data);
                    if (eventType === "cold_start_probe") callbacks.onColdProbe?.(data);
                    if (eventType === "cold_start_complete") callbacks.onColdComplete?.(data);
                } catch (e) {
                    console.warn("Failed to parse SSE data:", e);
                }
            }
        }
    }
}

async function pingProvider(body) {
    const resp = await fetch("/api/v1/benchmarks/ping", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    return resp.json();
}

async function fetchProviders() {
    const resp = await fetch("/api/v1/providers");
    if (!resp.ok) throw new Error("Failed to fetch providers");
    return resp.json();
}

async function fetchHistory(params = {}) {
    const qs = new URLSearchParams(params).toString();
    const resp = await fetch(`/api/v1/history?${qs}`);
    if (!resp.ok) throw new Error("Failed to fetch history");
    return resp.json();
}

async function fetchSuiteDetail(suiteId) {
    const resp = await fetch(`/api/v1/history/${suiteId}`);
    if (!resp.ok) throw new Error("Failed to fetch suite detail");
    return resp.json();
}

async function deleteSuite(suiteId) {
    const resp = await fetch(`/api/v1/history/${suiteId}`, { method: "DELETE" });
    if (!resp.ok) throw new Error("Failed to delete suite");
    return resp.json();
}

async function deleteAllHistory() {
    const resp = await fetch("/api/v1/history", { method: "DELETE" });
    if (!resp.ok) throw new Error("Failed to delete all history");
    return resp.json();
}

// ─── Settings API ──────────────────────────────────────────────────────────

const settingsApi = {
    async listProviders() {
        const resp = await fetch("/api/v1/settings/providers");
        if (!resp.ok) throw new Error("Failed to fetch DB providers");
        return resp.json();
    },

    async upsertProvider(data) {
        const resp = await fetch("/api/v1/settings/providers", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data),
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(err.detail || `HTTP ${resp.status}`);
        }
        return resp.json();
    },

    async deleteProvider(id) {
        const resp = await fetch(`/api/v1/settings/providers/${encodeURIComponent(id)}`, {
            method: "DELETE",
        });
        if (!resp.ok && resp.status !== 204) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(err.detail || `HTTP ${resp.status}`);
        }
    },

    async fetchRemoteModels(id) {
        const resp = await fetch(`/api/v1/settings/providers/${encodeURIComponent(id)}/models`);
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(err.detail || `HTTP ${resp.status}`);
        }
        return resp.json(); // { provider_id, models: [...] }
    },
};

