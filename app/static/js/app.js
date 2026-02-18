// app.js — Main dashboard logic

(function () {
    'use strict';

    // ===== State =====
    let providers = {};
    let currentMode = 'single';
    let isRunning = false;
    let runCount = 0;
    let totalExpectedRuns = 0;
    let ttfbChart, tpsChart;

    // ===== DOM Elements =====
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    // ===== Init =====
    document.addEventListener('DOMContentLoaded', async () => {
        ttfbChart = createTTFBChart('ttfb-chart');
        tpsChart = createTPSChart('tps-chart');
        setupNavigation();
        setupModeTabs();
        setupForm();
        setupHistory();
        setupSpeedTests();
        await loadProviders();
        await loadPastResults();
    });

    // ===== Load Past Results from DB =====
    async function loadPastResults() {
        try {
            const suites = await fetchHistory({});
            if (!suites || suites.length === 0) return;

            const tbody = $('#results-tbody');

            // Load the most recent 20 suites (they come newest-first), reverse so oldest is at top
            const toLoad = suites.slice(0, 20).reverse();

            for (const suite of toLoad) {
                try {
                    const detail = await fetchSuiteDetail(suite.id);
                    if (!detail.runs || detail.runs.length === 0) continue;

                    // Add suite header separator
                    const sep = document.createElement('tr');
                    sep.className = 'run-separator';
                    const date = new Date(suite.created_at).toLocaleString();
                    sep.innerHTML = `<td colspan="12" style="text-align:center; padding:0.4rem; font-size:0.7rem; color:var(--text-muted); border-bottom:1px solid var(--border-subtle); letter-spacing:0.03em;">── ${date} · ${suite.mode} · ${suite.provider_id}/${suite.model} ──</td>`;
                    tbody.appendChild(sep);

                    // Add each run row
                    for (const r of detail.runs) {
                        addResultRow({
                            run_number: r.run_number,
                            provider_id: r.provider_id,
                            model: r.model,
                            status: r.status,
                            ttfb_ms: r.ttfb_ms,
                            total_latency_ms: r.total_latency_ms,
                            tokens_per_second: r.tokens_per_second,
                            input_tokens: r.input_tokens,
                            output_tokens: r.output_tokens,
                            inter_chunk_ms_avg: r.inter_chunk_ms_avg,
                            total_words: r.total_words,
                        });
                    }
                } catch (e) {
                    console.warn('Failed to load suite detail:', suite.id, e);
                }
            }
        } catch (e) {
            console.warn('Failed to load past results:', e);
        }
    }

    // ===== Navigation =====
    function setupNavigation() {
        $$('.nav-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                $$('.nav-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                const view = btn.dataset.view;
                $$('.view').forEach(v => v.classList.remove('active'));
                $(`#${view}-view`).classList.add('active');
                if (view === 'history') loadHistory();
            });
        });
    }

    // ===== Mode Tabs =====
    function setupModeTabs() {
        $$('.mode-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                $$('.mode-tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                currentMode = tab.dataset.mode;
                updateFormVisibility();
            });
        });
    }

    function updateFormVisibility() {
        const showRuns = ['multi', 'concurrent', 'comparison'].includes(currentMode);
        const showConcurrency = currentMode === 'concurrent';
        const showComparison = currentMode === 'comparison';
        const showSingleProvider = !showComparison;

        $('#num-runs-group').classList.toggle('hidden', !showRuns);
        $('#concurrency-group').classList.toggle('hidden', !showConcurrency);
        $('#comparison-providers-group').classList.toggle('hidden', !showComparison);
        $('#single-provider-row').classList.toggle('hidden', showComparison);

        if (showComparison && $('#comparison-providers-list').children.length === 0) {
            addComparisonRow();
            addComparisonRow();
        }
    }

    // ===== Providers =====
    async function loadProviders() {
        try {
            providers = await fetchProviders();
            const providerSelect = $('#provider-select');
            providerSelect.innerHTML = '';

            if (Object.keys(providers).length === 0) {
                providerSelect.innerHTML = '<option value="">No providers configured</option>';
                return;
            }

            for (const [pid, p] of Object.entries(providers)) {
                const opt = document.createElement('option');
                opt.value = pid;
                opt.textContent = p.name;
                providerSelect.appendChild(opt);
            }

            // Restore saved provider or use first
            const savedProvider = localStorage.getItem('bench_provider');
            if (savedProvider && providers[savedProvider]) {
                providerSelect.value = savedProvider;
            }

            providerSelect.addEventListener('change', () => {
                localStorage.setItem('bench_provider', providerSelect.value);
                updateModelSelect(providerSelect.value);
            });
            updateModelSelect(providerSelect.value);

            // Populate speed test provider selects
            populateSpeedProviderSelects();
        } catch (e) {
            console.error('Failed to load providers:', e);
            $('#provider-select').innerHTML = '<option value="">Error loading providers</option>';
        }
    }

    function updateModelSelect(providerId) {
        const modelSelect = $('#model-select');
        modelSelect.innerHTML = '';
        if (!providerId || !providers[providerId]) {
            modelSelect.innerHTML = '<option value="">Select provider first</option>';
            return;
        }
        const p = providers[providerId];
        for (const m of p.models) {
            const opt = document.createElement('option');
            opt.value = m;
            opt.textContent = m;
            if (m === p.default_model) opt.selected = true;
            modelSelect.appendChild(opt);
        }

        // Restore saved model for this provider
        const savedModel = localStorage.getItem('bench_model_' + providerId);
        if (savedModel) {
            const exists = Array.from(modelSelect.options).some(o => o.value === savedModel);
            if (exists) modelSelect.value = savedModel;
        }

        // Save model on change
        modelSelect.addEventListener('change', () => {
            localStorage.setItem('bench_model_' + providerId, modelSelect.value);
        });
    }

    // ===== Comparison Rows =====
    function addComparisonRow() {
        const list = $('#comparison-providers-list');
        const item = document.createElement('div');
        item.className = 'comparison-item fade-in';

        const provSel = document.createElement('select');
        provSel.className = 'comp-provider';
        for (const [pid, p] of Object.entries(providers)) {
            const opt = document.createElement('option');
            opt.value = pid;
            opt.textContent = p.name;
            provSel.appendChild(opt);
        }

        const modSel = document.createElement('select');
        modSel.className = 'comp-model';

        function updateModels() {
            modSel.innerHTML = '';
            const p = providers[provSel.value];
            if (!p) return;
            for (const m of p.models) {
                const opt = document.createElement('option');
                opt.value = m;
                opt.textContent = m;
                if (m === p.default_model) opt.selected = true;
                modSel.appendChild(opt);
            }
        }
        provSel.addEventListener('change', updateModels);
        updateModels();

        const removeBtn = document.createElement('button');
        removeBtn.className = 'remove-btn';
        removeBtn.textContent = '✕';
        removeBtn.type = 'button';
        removeBtn.addEventListener('click', () => item.remove());

        item.appendChild(provSel);
        item.appendChild(modSel);
        item.appendChild(removeBtn);
        list.appendChild(item);
    }

    // ===== Form Submission =====
    function setupForm() {
        $('#add-comparison-provider')?.addEventListener('click', addComparisonRow);

        $('#benchmark-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            if (isRunning) return;
            await runBenchmark();
        });

        // Clear All results
        $('#clear-results-btn')?.addEventListener('click', async () => {
            if (!confirm('Delete ALL benchmark history? This cannot be undone.')) return;
            try {
                await deleteAllHistory();
                $('#results-tbody').innerHTML = '';
                resetStats();
                resetChart(ttfbChart);
                resetChart(tpsChart);
            } catch (e) {
                console.error('Failed to clear history:', e);
            }
        });
    }

    async function runBenchmark() {
        isRunning = true;
        const btn = $('#run-benchmark-btn');
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> Running...';

        // Reset
        runCount = 0;
        resetChart(ttfbChart);
        resetChart(tpsChart);

        // Show live output panel and reset it
        const liveCard = $('#live-output-card');
        liveCard.classList.remove('hidden');
        $('#live-output-body').textContent = '';
        $('#live-output-meta').textContent = '';
        $('#ticker-badge').textContent = '⏱ 0.0s';
        $('#ticker-badge').className = 'ticker-badge';
        $('#phase-badge').textContent = 'waiting';
        $('#phase-badge').className = 'phase-badge phase-waiting';

        // Add separator if there are existing results
        const tbody = $('#results-tbody');
        if (tbody.children.length > 0) {
            const sep = document.createElement('tr');
            sep.className = 'run-separator';
            sep.innerHTML = `<td colspan="12" style="text-align:center; padding:0.3rem; font-size:0.7rem; color:var(--text-muted); border-bottom:1px solid var(--border-subtle); letter-spacing:0.05em;">── New Run ──</td>`;
            tbody.appendChild(sep);
        }
        resetStats();

        const prompt = $('#prompt-input').value;
        const maxTokens = parseInt($('#max-tokens').value);
        const temperature = parseFloat($('#temperature').value);
        const numRuns = parseInt($('#num-runs').value) || 1;
        const concurrency = parseInt($('#concurrency').value) || 5;

        let endpoint, body;

        if (currentMode === 'comparison') {
            const items = $$('.comparison-item');
            const providersList = [];
            items.forEach(item => {
                providersList.push({
                    provider_id: item.querySelector('.comp-provider').value,
                    model: item.querySelector('.comp-model').value,
                });
            });
            endpoint = 'comparison';
            body = { providers: providersList, prompt, max_tokens: maxTokens, temperature, num_runs: numRuns };
            totalExpectedRuns = providersList.length * numRuns;
        } else {
            const providerId = $('#provider-select').value;
            const model = $('#model-select').value;

            if (!providerId || !model) {
                alert('Please select a provider and model.');
                resetRunButton();
                return;
            }

            body = { provider_id: providerId, model, prompt, max_tokens: maxTokens, temperature };

            if (currentMode === 'single') {
                endpoint = 'single';
                totalExpectedRuns = 1;
            } else if (currentMode === 'multi') {
                endpoint = 'multi';
                body.num_runs = numRuns;
                totalExpectedRuns = numRuns;
            } else {
                endpoint = 'concurrent';
                body.num_runs = numRuns;
                body.concurrency = concurrency;
                totalExpectedRuns = numRuns;
            }
        }

        // Show progress
        if (totalExpectedRuns > 1) {
            $('#progress-container').classList.remove('hidden');
            updateProgress(0, totalExpectedRuns);
        }

        try {
            await streamBenchmark(endpoint, body, {
                onRunMetric: handleRunMetric,
                onComplete: handleSuiteComplete,
                onTick: handleTick,
                onTokenChunk: handleTokenChunk,
            });
        } catch (e) {
            console.error('Benchmark error:', e);
            showError(e.message);
        } finally {
            resetRunButton();
        }
    }

    function resetRunButton() {
        isRunning = false;
        const btn = $('#run-benchmark-btn');
        btn.disabled = false;
        btn.innerHTML = '<span class="btn-icon">▶</span> Run Benchmark';
    }

    // ===== SSE Event Handlers =====

    function handleTick(data) {
        const badge = $('#ticker-badge');
        badge.textContent = `⏱ ${data.elapsed_seconds.toFixed(1)}s`;

        const phaseBadge = $('#phase-badge');
        if (data.phase === 'streaming') {
            phaseBadge.textContent = 'streaming';
            phaseBadge.className = 'phase-badge phase-streaming';
            badge.className = 'ticker-badge ticker-streaming';
        } else {
            phaseBadge.textContent = 'waiting';
            phaseBadge.className = 'phase-badge phase-waiting';
            badge.className = 'ticker-badge';
        }
    }

    function handleTokenChunk(data) {
        const body = $('#live-output-body');
        // Append text as a span for typewriter effect
        const span = document.createElement('span');
        span.textContent = data.text;
        span.className = 'chunk-token';
        body.appendChild(span);
        // Auto-scroll
        body.scrollTop = body.scrollHeight;

        // Update meta
        $('#live-output-meta').textContent =
            `${data.cumulative_chars} chars · ${data.cumulative_words} words · chunk #${data.chunk_index}`;
    }

    function handleRunMetric(data) {
        runCount++;
        addResultRow(data);

        const label = data.provider_id !== data.model
            ? `${data.provider_id}/${data.model} #${data.run_number}`
            : `Run ${data.run_number}`;

        if (data.ttfb_ms) addDataPoint(ttfbChart, label, Math.round(data.ttfb_ms));
        if (data.tokens_per_second) addDataPoint(tpsChart, label, Math.round(data.tokens_per_second * 10) / 10);

        if (totalExpectedRuns > 1) {
            updateProgress(runCount, totalExpectedRuns);
        }

        // Update live output meta with final stats
        if (data.inter_chunk_ms_avg) {
            $('#live-output-meta').textContent +=
                ` · chunk avg ${data.inter_chunk_ms_avg.toFixed(1)}ms`;
        }
    }

    function handleSuiteComplete(data) {
        $('#progress-container').classList.add('hidden');

        // Update stats
        $('#stat-ttfb').textContent = data.avg_ttfb_ms != null ? Math.round(data.avg_ttfb_ms) : '—';
        $('#stat-tps').textContent = data.avg_tps != null ? data.avg_tps.toFixed(1) : '—';
        $('#stat-latency').textContent = data.avg_latency_ms != null ? Math.round(data.avg_latency_ms) : '—';
        $('#stat-tokens').textContent = data.total_input_tokens != null
            ? `${data.total_input_tokens} / ${data.total_output_tokens}`
            : '—';

        const successRate = data.total_runs > 0
            ? Math.round((data.successful_runs / data.total_runs) * 100)
            : 0;
        $('#stat-success').textContent = successRate;

        // Animate stats
        $$('.stat-card').forEach((card) => {
            card.classList.remove('fade-in');
            void card.offsetWidth; // Reflow
            card.classList.add('fade-in');
        });

        // Update ticker to done
        $('#ticker-badge').textContent = '✓ done';
        $('#ticker-badge').className = 'ticker-badge ticker-done';
        $('#phase-badge').textContent = 'complete';
        $('#phase-badge').className = 'phase-badge phase-done';
    }

    // ===== Results Table =====
    function addResultRow(data) {
        const tbody = $('#results-tbody');
        const tr = document.createElement('tr');
        tr.className = 'fade-in';

        const statusClass = data.status === 'success' ? 'status-success' : 'status-error';
        const statusIcon = data.status === 'success' ? '✓' : '✗';

        const chunkAvg = data.inter_chunk_ms_avg != null ? data.inter_chunk_ms_avg.toFixed(1) : '—';
        const words = data.total_words != null ? data.total_words : '—';

        tr.innerHTML = `
            <td>${data.run_number}</td>
            <td>${data.provider_id}</td>
            <td>${data.model}</td>
            <td><span class="status-badge ${statusClass}">${statusIcon} ${data.status}</span></td>
            <td>${data.ttfb_ms != null ? Math.round(data.ttfb_ms) : '—'}</td>
            <td>${data.total_latency_ms != null ? Math.round(data.total_latency_ms) : '—'}</td>
            <td>${data.tokens_per_second != null ? data.tokens_per_second.toFixed(1) : '—'}</td>
            <td>${data.input_tokens ?? '—'}</td>
            <td>${data.output_tokens ?? '—'}</td>
            <td>${chunkAvg}</td>
            <td>${words}</td>
            <td><button class="btn btn-sm btn-danger row-del" title="Remove" style="padding:0.1rem 0.35rem; font-size:0.65rem; line-height:1;">✕</button></td>
        `;
        tr.querySelector('.row-del').addEventListener('click', () => tr.remove());
        tbody.appendChild(tr);
    }

    // ===== Progress =====
    function updateProgress(current, total) {
        const pct = total > 0 ? (current / total) * 100 : 0;
        $('#progress-fill').style.width = `${pct}%`;
        $('#progress-text').textContent = `${current} / ${total} runs`;
    }

    // ===== Stats Reset =====
    function resetStats() {
        ['stat-ttfb', 'stat-tps', 'stat-latency', 'stat-tokens', 'stat-success'].forEach(id => {
            $(`#${id}`).textContent = '—';
        });
    }

    // ===== Error Display =====
    function showError(message) {
        const tbody = $('#results-tbody');
        const tr = document.createElement('tr');
        tr.innerHTML = `<td colspan="12" style="color: var(--accent-danger); text-align: center; font-family: var(--font-sans);">⚠ Error: ${message}</td>`;
        tbody.appendChild(tr);
    }

    // ===== History =====
    function setupHistory() {
        $('#refresh-history-btn')?.addEventListener('click', loadHistory);
        $('#history-mode-filter')?.addEventListener('change', loadHistory);
        $('#close-modal-btn')?.addEventListener('click', () => {
            $('#suite-detail-modal').classList.add('hidden');
        });
        $('#suite-detail-modal')?.addEventListener('click', (e) => {
            if (e.target === $('#suite-detail-modal')) {
                $('#suite-detail-modal').classList.add('hidden');
            }
        });
        // Clear All in History tab
        $('#clear-all-history-btn')?.addEventListener('click', async () => {
            if (!confirm('Delete ALL benchmark history? This cannot be undone.')) return;
            try {
                await deleteAllHistory();
                $('#results-tbody').innerHTML = '';
                resetStats();
                resetChart(ttfbChart);
                resetChart(tpsChart);
                loadHistory();
            } catch (e) {
                console.error('Failed to clear all history:', e);
            }
        });
    }

    async function loadHistory() {
        const tbody = $('#history-tbody');
        tbody.innerHTML = '<tr><td colspan="10" class="empty-state">Loading...</td></tr>';

        try {
            const mode = $('#history-mode-filter')?.value || '';
            const params = {};
            if (mode) params.mode = mode;

            const suites = await fetchHistory(params);

            if (suites.length === 0) {
                tbody.innerHTML = '<tr><td colspan="10" class="empty-state">No benchmark history yet</td></tr>';
                return;
            }

            tbody.innerHTML = '';
            for (const s of suites) {
                const tr = document.createElement('tr');
                tr.className = 'fade-in';

                const date = new Date(s.created_at).toLocaleString();

                tr.innerHTML = `
                    <td style="font-family: var(--font-sans);">${date}</td>
                    <td><span class="status-badge status-running">${s.mode}</span></td>
                    <td>${s.provider_id}</td>
                    <td>${s.model}</td>
                    <td>${s.num_runs}</td>
                    <td>${s.avg_ttfb_ms != null ? Math.round(s.avg_ttfb_ms) : '—'}</td>
                    <td>${s.avg_tps != null ? s.avg_tps.toFixed(1) : '—'}</td>
                    <td>${s.avg_latency_ms != null ? Math.round(s.avg_latency_ms) : '—'}</td>
                    <td>${s.error_count}</td>
                    <td>
                        <button class="btn btn-sm btn-outline view-detail-btn" data-suite-id="${s.id}">View</button>
                        <button class="btn btn-sm btn-danger delete-suite-btn" data-suite-id="${s.id}">✕</button>
                    </td>
                `;
                tbody.appendChild(tr);
            }

            // Attach event listeners
            $$('.view-detail-btn').forEach(btn => {
                btn.addEventListener('click', () => showSuiteDetail(btn.dataset.suiteId));
            });
            $$('.delete-suite-btn').forEach(btn => {
                btn.addEventListener('click', async () => {
                    if (confirm('Delete this benchmark suite?')) {
                        await deleteSuite(btn.dataset.suiteId);
                        loadHistory();
                    }
                });
            });
        } catch (e) {
            console.error('Failed to load history:', e);
            tbody.innerHTML = '<tr><td colspan="10" class="empty-state">Error loading history</td></tr>';
        }
    }

    async function showSuiteDetail(suiteId) {
        const modal = $('#suite-detail-modal');
        const body = $('#suite-detail-body');
        body.innerHTML = '<p class="pulse" style="text-align:center;">Loading details...</p>';
        modal.classList.remove('hidden');

        try {
            const detail = await fetchSuiteDetail(suiteId);

            let html = `
                <div class="stats-grid" style="margin-bottom: 1rem;">
                    <div class="stat-card glass-card">
                        <div class="stat-label">Mode</div>
                        <div class="stat-value" style="font-size: 1rem;">${detail.mode}</div>
                    </div>
                    <div class="stat-card glass-card">
                        <div class="stat-label">Avg TTFB</div>
                        <div class="stat-value">${detail.avg_ttfb_ms != null ? Math.round(detail.avg_ttfb_ms) : '—'}</div>
                        <div class="stat-unit">ms</div>
                    </div>
                    <div class="stat-card glass-card">
                        <div class="stat-label">Avg TPS</div>
                        <div class="stat-value">${detail.avg_tps != null ? detail.avg_tps.toFixed(1) : '—'}</div>
                        <div class="stat-unit">tok/s</div>
                    </div>
                    <div class="stat-card glass-card">
                        <div class="stat-label">P95 TTFB</div>
                        <div class="stat-value">${detail.p95_ttfb_ms != null ? Math.round(detail.p95_ttfb_ms) : '—'}</div>
                        <div class="stat-unit">ms</div>
                    </div>
                    <div class="stat-card glass-card">
                        <div class="stat-label">Chunk Avg</div>
                        <div class="stat-value">${detail.avg_inter_chunk_ms != null ? detail.avg_inter_chunk_ms.toFixed(1) : '—'}</div>
                        <div class="stat-unit">ms</div>
                    </div>
                </div>
                <p style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.5rem;">
                    <strong>Prompt:</strong> ${escapeHtml(detail.prompt.substring(0, 200))}${detail.prompt.length > 200 ? '...' : ''}
                </p>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th>#</th><th>Provider</th><th>Model</th><th>Status</th>
                                <th>TTFB</th><th>Latency</th><th>TPS</th><th>In</th><th>Out</th>
                                <th>Chunk Avg</th><th>Words</th>
                            </tr>
                        </thead>
                        <tbody>
            `;

            for (const r of detail.runs) {
                const sc = r.status === 'success' ? 'status-success' : 'status-error';
                html += `
                    <tr>
                        <td>${r.run_number}</td>
                        <td>${r.provider_id}</td>
                        <td>${r.model}</td>
                        <td><span class="status-badge ${sc}">${r.status}</span></td>
                        <td>${r.ttfb_ms != null ? Math.round(r.ttfb_ms) : '—'}</td>
                        <td>${r.total_latency_ms != null ? Math.round(r.total_latency_ms) : '—'}</td>
                        <td>${r.tokens_per_second != null ? r.tokens_per_second.toFixed(1) : '—'}</td>
                        <td>${r.input_tokens ?? '—'}</td>
                        <td>${r.output_tokens ?? '—'}</td>
                        <td>${r.inter_chunk_ms_avg != null ? r.inter_chunk_ms_avg.toFixed(1) : '—'}</td>
                        <td>${r.total_words ?? '—'}</td>
                    </tr>
                `;
            }

            html += '</tbody></table></div>';
            body.innerHTML = html;
        } catch (e) {
            body.innerHTML = `<p style="color: var(--accent-danger);">Error loading details: ${e.message}</p>`;
        }
    }

    // ===== Speed Tests =====
    function populateSpeedProviderSelects() {
        const selects = [
            { prov: '#ping-provider-select', mod: '#ping-model-select' },
            { prov: '#stress-provider-select', mod: '#stress-model-select' },
            { prov: '#cold-provider-select', mod: '#cold-model-select' },
        ];

        for (const { prov, mod } of selects) {
            const provSel = $(prov);
            const modSel = $(mod);
            if (!provSel) continue;

            provSel.innerHTML = '';
            for (const [pid, p] of Object.entries(providers)) {
                const opt = document.createElement('option');
                opt.value = pid;
                opt.textContent = p.name;
                provSel.appendChild(opt);
            }

            function updateMod() {
                modSel.innerHTML = '';
                const p = providers[provSel.value];
                if (!p) return;
                for (const m of p.models) {
                    const opt = document.createElement('option');
                    opt.value = m;
                    opt.textContent = m;
                    if (m === p.default_model) opt.selected = true;
                    modSel.appendChild(opt);
                }
            }
            provSel.addEventListener('change', updateMod);
            updateMod();
        }
    }

    function setupSpeedTests() {
        // --- Ping ---
        $('#run-ping-btn')?.addEventListener('click', async () => {
            const btn = $('#run-ping-btn');
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span> Pinging...';

            const providerId = $('#ping-provider-select').value;
            const model = $('#ping-model-select').value;

            try {
                const result = await pingProvider({ provider_id: providerId, model });
                const resultEl = $('#ping-result');
                resultEl.classList.remove('hidden');

                $('#ping-ttfb').textContent = result.ttfb_ms != null ? result.ttfb_ms.toFixed(1) : '—';
                $('#ping-rtt').textContent = result.round_trip_ms != null ? result.round_trip_ms.toFixed(1) : '—';

                const healthEl = $('#ping-health');
                healthEl.textContent = result.health;
                healthEl.className = 'ping-value health-' + result.health;
            } catch (e) {
                alert('Ping failed: ' + e.message);
            } finally {
                btn.disabled = false;
                btn.innerHTML = '<i data-lucide="send" class="icon-btn"></i> Run Ping';
                lucide.createIcons();
            }
        });

        // --- Stress Test ---
        $('#run-stress-btn')?.addEventListener('click', async () => {
            const btn = $('#run-stress-btn');
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span> Running Stress Test...';

            const providerId = $('#stress-provider-select').value;
            const model = $('#stress-model-select').value;
            const prompt = $('#stress-prompt').value;
            const maxTokens = parseInt($('#stress-max-tokens').value);
            const levelsRaw = $('#stress-levels').value;
            const concurrencyLevels = levelsRaw.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
            const runsPerLevel = parseInt($('#stress-runs').value);

            const tbody = $('#stress-tbody');
            tbody.innerHTML = '';
            $('#stress-results').classList.remove('hidden');
            $('#stress-summary').textContent = '';

            try {
                await streamBenchmark('stress', {
                    provider_id: providerId, model, prompt,
                    max_tokens: maxTokens,
                    concurrency_levels: concurrencyLevels,
                    runs_per_level: runsPerLevel,
                }, {
                    onStressLevel: (data) => {
                        const tr = document.createElement('tr');
                        tr.className = 'fade-in';
                        tr.innerHTML = `
                            <td>${data.concurrency}</td>
                            <td>${data.avg_tps != null ? data.avg_tps.toFixed(2) : '—'}</td>
                            <td>${data.avg_latency_ms != null ? Math.round(data.avg_latency_ms) : '—'}</td>
                            <td>${(data.error_rate * 100).toFixed(1)}%</td>
                            <td>${data.successful_runs}</td>
                        `;
                        tbody.appendChild(tr);
                    },
                    onStressComplete: (data) => {
                        if (data.peak_tps != null) {
                            $('#stress-summary').innerHTML = `
                                <div class="speed-summary-box">
                                    🏆 Peak TPS: <strong>${data.peak_tps.toFixed(2)}</strong>
                                    at concurrency <strong>${data.peak_tps_concurrency}</strong>
                                </div>
                            `;
                        }
                    },
                });
            } catch (e) {
                alert('Stress test failed: ' + e.message);
            } finally {
                btn.disabled = false;
                btn.innerHTML = '<i data-lucide="zap" class="icon-btn"></i> Run Stress Test';
                lucide.createIcons();
            }
        });

        // --- Cold-Start Probe ---
        $('#run-cold-btn')?.addEventListener('click', async () => {
            const btn = $('#run-cold-btn');
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span> Running Cold-Start Probe...';

            const providerId = $('#cold-provider-select').value;
            const model = $('#cold-model-select').value;
            const numProbes = parseInt($('#cold-probes').value);
            const gapSeconds = parseInt($('#cold-gap').value);

            const tbody = $('#cold-tbody');
            tbody.innerHTML = '';
            $('#cold-results').classList.remove('hidden');
            $('#cold-summary').textContent = '';

            try {
                await streamBenchmark('cold-start', {
                    provider_id: providerId, model,
                    num_cold_probes: numProbes,
                    gap_seconds: gapSeconds,
                }, {
                    onColdProbe: (data) => {
                        const tr = document.createElement('tr');
                        tr.className = 'fade-in';
                        const typeClass = data.probe_type === 'cold' ? 'status-error' : 'status-success';
                        tr.innerHTML = `
                            <td>${data.probe_number}</td>
                            <td><span class="status-badge ${typeClass}">${data.probe_type}</span></td>
                            <td>${data.ttfb_ms != null ? data.ttfb_ms.toFixed(1) : '—'}</td>
                            <td>${data.total_latency_ms != null ? data.total_latency_ms.toFixed(1) : '—'}</td>
                        `;
                        tbody.appendChild(tr);
                    },
                    onColdComplete: (data) => {
                        if (data.avg_cold_ttfb_ms != null) {
                            $('#cold-summary').innerHTML = `
                                <div class="speed-summary-box">
                                    ❄️ Cold TTFB: <strong>${data.avg_cold_ttfb_ms.toFixed(1)}ms</strong> ·
                                    🔥 Warm TTFB: <strong>${data.avg_warm_ttfb_ms?.toFixed(1) ?? '—'}ms</strong> ·
                                    Ratio: <strong>${data.cold_vs_warm_ratio?.toFixed(2) ?? '—'}×</strong>
                                </div>
                            `;
                        }
                    },
                });
            } catch (e) {
                alert('Cold-start probe failed: ' + e.message);
            } finally {
                btn.disabled = false;
                btn.innerHTML = '<i data-lucide="snowflake" class="icon-btn"></i> Run Cold-Start Probe';
                lucide.createIcons();
            }
        });
    }

    function escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

})();
