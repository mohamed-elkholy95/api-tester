// charts.js — Chart.js TTFB + TPS rendering

const chartDefaults = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: {
        legend: {
            labels: { color: '#94a3b8', font: { family: "'Inter', sans-serif", size: 11 } }
        }
    },
    scales: {
        x: {
            ticks: { color: '#64748b', font: { size: 10 } },
            grid: { color: 'rgba(99, 102, 241, 0.06)' }
        },
        y: {
            beginAtZero: true,
            ticks: { color: '#64748b', font: { size: 10 } },
            grid: { color: 'rgba(99, 102, 241, 0.06)' }
        }
    }
};

function createTTFBChart(canvasId) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'TTFB (ms)',
                data: [],
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                borderWidth: 2,
                tension: 0.3,
                fill: true,
                pointBackgroundColor: '#6366f1',
                pointRadius: 4,
                pointHoverRadius: 6,
            }]
        },
        options: {
            ...chartDefaults,
            scales: {
                ...chartDefaults.scales,
                y: { ...chartDefaults.scales.y, title: { display: true, text: 'ms', color: '#64748b' } }
            }
        }
    });
}

function createTPSChart(canvasId) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Tokens/sec',
                data: [],
                backgroundColor: 'rgba(16, 185, 129, 0.6)',
                borderColor: '#10b981',
                borderWidth: 1,
                borderRadius: 4,
            }]
        },
        options: {
            ...chartDefaults,
            scales: {
                ...chartDefaults.scales,
                y: { ...chartDefaults.scales.y, title: { display: true, text: 'tok/s', color: '#64748b' } }
            }
        }
    });
}

function addDataPoint(chart, label, value) {
    chart.data.labels.push(label);
    chart.data.datasets[0].data.push(value);
    chart.update('none'); // no animation for real-time perf
}

function addComparisonDataPoint(chart, label, value, datasetIndex) {
    if (datasetIndex === 0) {
        chart.data.labels.push(label);
    }
    while (chart.data.datasets.length <= datasetIndex) {
        const colors = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#3b82f6', '#8b5cf6'];
        const color = colors[chart.data.datasets.length % colors.length];
        chart.data.datasets.push({
            label: `Provider ${chart.data.datasets.length + 1}`,
            data: [],
            borderColor: color,
            backgroundColor: color + '20',
            borderWidth: 2,
            tension: 0.3,
            pointRadius: 4,
        });
    }
    chart.data.datasets[datasetIndex].data.push(value);
    chart.update('none');
}

function resetChart(chart) {
    chart.data.labels = [];
    chart.data.datasets.forEach(ds => { ds.data = []; });
    // Remove extra datasets
    while (chart.data.datasets.length > 1) {
        chart.data.datasets.pop();
    }
    chart.update('none');
}
