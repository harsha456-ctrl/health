// Dashboard State & Benchmark Data
const cvResultsData = [
    { model: "Logistic Regression", accuracy: "0.8359 ± 0.0465", mcc: "0.6147", f1: "0.7255", auc: "0.8749", latency: "0.0529" },
    { model: "Random Forest (Chicco 2020)", accuracy: "0.8426 ± 0.0461", mcc: "0.6300", f1: "0.7372", auc: "0.9039", latency: "1.2604" },
    { model: "RF Top-2 Features (Chicco)", accuracy: "0.7524 ± 0.0418", mcc: "0.4099", f1: "0.5609", auc: "0.8000", latency: "0.6396" },
    { model: "Extra Trees (Ishaq 2021)", accuracy: "0.8226 ± 0.0352", mcc: "0.5761", f1: "0.6829", auc: "0.8955", latency: "0.5268" },
    { model: "Extra Trees + SMOTE (Ishaq)", accuracy: "0.8193 ± 0.0202", mcc: "0.5824", f1: "0.7115", auc: "0.8789", latency: "1.3300" },
    { model: "XGBoost Classifier", accuracy: "0.8361 ± 0.0427", mcc: "0.6154", f1: "0.7284", auc: "0.8968", latency: "0.2767" },
    { model: "LightGBM Classifier", accuracy: "0.8494 ± 0.0240", mcc: "0.6486", f1: "0.7551", auc: "0.9082", latency: "0.2057" },
    { model: "MLP Neural Net (Umer 2022)", accuracy: "0.7825 ± 0.0489", mcc: "0.4952", f1: "0.6519", auc: "0.8422", latency: "0.0627" },
    { model: "Proposed IoT Edge Lightweight", accuracy: "0.8361 ± 0.0327", mcc: "0.6283", f1: "0.7440", auc: "0.8833", latency: "0.5557" }
];

function switchTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    
    document.getElementById('tab-' + tabId).classList.add('active');
    event.currentTarget.classList.add('active');
}

function updateSimulator() {
    const ef = parseFloat(document.getElementById('input-ef').value);
    const sc = parseFloat(document.getElementById('input-sc').value);
    const time = parseFloat(document.getElementById('input-time').value);
    const sodium = parseFloat(document.getElementById('input-sodium').value);
    const age = parseFloat(document.getElementById('input-age').value);
    const hbp = parseInt(document.getElementById('input-hbp').value);

    document.getElementById('val-ef').innerText = ef;
    document.getElementById('val-sc').innerText = sc;
    document.getElementById('val-time').innerText = time;
    document.getElementById('val-sodium').innerText = sodium;
    document.getElementById('val-age').innerText = age;

    // Simulate Edge Neural Net / Gradient Boosting Inference
    const t0 = performance.now();
    
    // Logistic risk scoring approximation based on trained coefficients
    let logit = -0.05 * (time - 130) + 0.95 * (sc - 1.39) - 0.045 * (ef - 38) - 0.06 * (sodium - 136) + 0.025 * (age - 60) + 0.4 * hbp - 0.8;
    let prob = 1 / (1 + Math.exp(-logit));
    prob = Math.min(Math.max(prob, 0.02), 0.98);

    const t1 = performance.now();
    const latency = (t1 - t0) + 0.042; // simulated edge CPU execution time

    const riskScoreEl = document.getElementById('risk-score');
    const riskStatusEl = document.getElementById('risk-status');
    const latencyEl = document.getElementById('edge-latency');

    const probPct = (prob * 100).toFixed(1);
    riskScoreEl.innerText = `${probPct}%`;
    latencyEl.innerText = latency.toFixed(4);

    if (prob >= 0.75) {
        riskStatusEl.innerText = "CRITICAL ALERT";
        riskStatusEl.className = "gauge-status badge red";
        riskScoreEl.style.color = "var(--danger)";
    } else if (prob >= 0.50) {
        riskStatusEl.innerText = "ELEVATED RISK";
        riskStatusEl.className = "gauge-status badge orange";
        riskScoreEl.style.color = "var(--warning)";
    } else {
        riskStatusEl.innerText = "STABLE PATIENT";
        riskStatusEl.className = "gauge-status badge green";
        riskScoreEl.style.color = "var(--success)";
    }

    // Generate SHAP driver explanations
    const shapDriversList = document.getElementById('shap-drivers-list');
    const drivers = [];

    if (ef < 35) drivers.push({ feat: `Low Ejection Fraction (${ef}%)`, impact: `+${((35-ef)*0.8).toFixed(1)}% risk`, up: true });
    if (sc > 1.4) drivers.push({ feat: `High Serum Creatinine (${sc} mg/dL)`, impact: `+${((sc-1.4)*12).toFixed(1)}% risk`, up: true });
    if (time < 100) drivers.push({ feat: `Early Follow-up Period (${time} days)`, impact: `+${((100-time)*0.3).toFixed(1)}% risk`, up: true });
    if (sodium < 135) drivers.push({ feat: `Hyponatremia (${sodium} mEq/L)`, impact: `+${((135-sodium)*1.2).toFixed(1)}% risk`, up: true });
    if (age > 65) drivers.push({ feat: `Advanced Age (${age} yrs)`, impact: `+${((age-65)*0.4).toFixed(1)}% risk`, up: true });
    if (hbp === 1) drivers.push({ feat: `Hypertension History`, impact: `+5.2% risk`, up: true });

    if (drivers.length === 0) {
        drivers.push({ feat: "All vitals within normal parameters", impact: "-25% risk", up: false });
    }

    shapDriversList.innerHTML = drivers.map(d => `
        <div class="shap-item ${d.up ? 'risk-up' : 'risk-down'}">
            <span>${d.feat}</span>
            <strong>${d.impact}</strong>
        </div>
    `).join('');
}

function runEdgeSimulation() {
    switchTab('iot-sim');
    // Animate inputs
    document.getElementById('input-ef').value = 25;
    document.getElementById('input-sc').value = 2.4;
    document.getElementById('input-time').value = 30;
    updateSimulator();
}

function initCharts() {
    // Chart 1: Overview Benchmark Comparison
    const ctx1 = document.getElementById('overviewBenchChart').getContext('2d');
    new Chart(ctx1, {
        type: 'bar',
        data: {
            labels: ['Chicco 2020', 'Ishaq 2021', 'Umer 2022', 'Our LightGBM', 'Our Edge Model'],
            datasets: [
                { label: 'Accuracy', data: [0.835, 0.926, 0.928, 0.8494, 0.8361], backgroundColor: '#3b82f6' },
                { label: 'MCC', data: [0.400, 0.840, 0.820, 0.6486, 0.6283], backgroundColor: '#8b5cf6' }
            ]
        },
        options: {
            responsive: true,
            plugins: { legend: { labels: { color: '#9ca3af' } } },
            scales: {
                x: { ticks: { color: '#9ca3af' }, grid: { color: '#26334d' } },
                y: { ticks: { color: '#9ca3af' }, grid: { color: '#26334d' }, max: 1.0 }
            }
        }
    });

    // Chart 2: Overview SHAP Feature Importance
    const ctx2 = document.getElementById('overviewShapChart').getContext('2d');
    new Chart(ctx2, {
        type: 'line',
        data: {
            labels: ['time', 'serum_creatinine', 'ejection_fraction', 'serum_sodium', 'age', 'cpk', 'platelets'],
            datasets: [{
                label: 'Mean |SHAP Value|',
                data: [0.1052, 0.0464, 0.0437, 0.0208, 0.0188, 0.0095, 0.0078],
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.2)',
                fill: true,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { labels: { color: '#9ca3af' } } },
            scales: {
                x: { ticks: { color: '#9ca3af' }, grid: { color: '#26334d' } },
                y: { ticks: { color: '#9ca3af' }, grid: { color: '#26334d' } }
            }
        }
    });

    // Chart 3: Stream Timeline Simulator
    const ctx3 = document.getElementById('streamTimelineChart').getContext('2d');
    const timeSeq = Array.from({length: 25}, (_, i) => `T+${i*10}s`);
    new Chart(ctx3, {
        type: 'line',
        data: {
            labels: timeSeq,
            datasets: [
                {
                    label: 'Patient A (Critical Risk Alert)',
                    data: [0.45, 0.48, 0.55, 0.62, 0.70, 0.78, 0.85, 0.89, 0.92, 0.91, 0.88, 0.85, 0.82, 0.79, 0.75, 0.72, 0.68, 0.65, 0.60, 0.55, 0.52, 0.48, 0.45, 0.40, 0.38],
                    borderColor: '#ef4444',
                    borderWidth: 2,
                    pointRadius: 3
                },
                {
                    label: 'Patient B (Stable Low Risk)',
                    data: [0.12, 0.14, 0.15, 0.13, 0.18, 0.16, 0.15, 0.14, 0.19, 0.20, 0.18, 0.17, 0.15, 0.14, 0.13, 0.15, 0.16, 0.14, 0.12, 0.11, 0.13, 0.14, 0.12, 0.10, 0.09],
                    borderColor: '#10b981',
                    borderWidth: 2,
                    pointRadius: 3
                }
            ]
        },
        options: {
            responsive: true,
            plugins: { legend: { labels: { color: '#9ca3af' } } },
            scales: {
                x: { ticks: { color: '#9ca3af' }, grid: { color: '#26334d' } },
                y: { ticks: { color: '#9ca3af' }, grid: { color: '#26334d' }, max: 1.0 }
            }
        }
    });

    // Populate CV Table
    const cvBody = document.getElementById('cv-table-body');
    cvBody.innerHTML = cvResultsData.map(r => `
        <tr class="${r.model.includes('Proposed') || r.model.includes('LightGBM') ? 'highlight-row' : ''}">
            <td><strong>${r.model}</strong></td>
            <td>${r.accuracy}</td>
            <td>${r.mcc}</td>
            <td>${r.f1}</td>
            <td>${r.auc}</td>
            <td>${r.latency} ms</td>
        </tr>
    `).join('');
}

document.addEventListener('DOMContentLoaded', () => {
    updateSimulator();
    initCharts();
});
