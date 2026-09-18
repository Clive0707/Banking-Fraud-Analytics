// Apple-Quality Financial Analytics Platform Engine

let currentPage = 1;
const perPage = 20;
let charts = {};
let currentFilterState = {};

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initNavigation();
  loadOverview();
});

// Indian Currency Formatting Helper (₹1,24,50,000.00)
function formatINR(val) {
  if (val === null || val === undefined || isNaN(val)) return "₹0.00";
  const num = parseFloat(val);
  const parts = num.toFixed(2).split(".");
  let integerPart = parts[0];
  const decimalPart = parts[1];
  
  const lastThree = integerPart.substring(integerPart.length - 3);
  const otherNumbers = integerPart.substring(0, integerPart.length - 3);
  if (otherNumbers !== '') {
    integerPart = otherNumbers.replace(/\B(?=(\d{2})+(?!\d))/g, ",") + "," + lastThree;
  } else {
    integerPart = lastThree;
  }
  return `₹${integerPart}.${decimalPart}`;
}

// Theme Switcher (Dark / Light)
function initTheme() {
  const savedTheme = localStorage.getItem('aegis_theme') || 'dark';
  document.documentElement.setAttribute('data-theme', savedTheme);
  updateThemeIcon(savedTheme);
}

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme');
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('aegis_theme', next);
  updateThemeIcon(next);
}

function updateThemeIcon(theme) {
  const icon = document.getElementById('theme-icon');
  if (icon) {
    icon.className = theme === 'dark' ? 'fa-solid fa-moon' : 'fa-solid fa-sun';
  }
}

// Navigation & Tab Switcher
function initNavigation() {
  const navItems = document.querySelectorAll('.nav-item');
  navItems.forEach(item => {
    item.addEventListener('click', (e) => {
      e.preventDefault();
      const tabId = item.getAttribute('data-tab');
      switchTab(tabId);
    });
  });
}

function switchTab(tabId) {
  document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));

  const activeNav = document.querySelector(`.nav-item[data-tab="${tabId}"]`);
  const activeTab = document.getElementById(`tab-${tabId}`);

  if (activeNav) activeNav.classList.add('active');
  if (activeTab) activeTab.classList.add('active');

  // Update Header Titles
  const titles = {
    overview: ["Executive Analytics Overview", "Real-time analytical view of transaction behavior, fraud risk, and customer activity"],
    fraud: ["Fraud Intelligence Center", "Categorical risk breakdowns and suspicious transaction feed"],
    prediction: ["Risk Inference Simulator", "Evaluate custom transactions against retrained ML classifiers"],
    customer360: ["Customer 360 Profile", "Complete behavioral history, risk indicators, and segment label"],
    segmentation: ["Customer Segmentation", "Unsupervised K-Means clustering and behavioral profiles"],
    anomaly: ["Anomaly Intelligence Center", "Isolation Forest structural outlier scores"],
    performance: ["Machine Learning Model Lab", "Supervised classification evaluation matrix and confusion matrices"],
    time_geo: ["Time & Geographic Intelligence", "7x24 Day × Hour fraud heatmap and location rankings"],
    explorer: ["Transaction Explorer", "Searchable and filterable 15M transaction dataset"],
    quality: ["Data Quality Center", "Completeness, uniqueness, validity, and consistency audit"],
    processing: ["Spark Processing Monitor", "Apache PySpark pipeline execution metadata"],
    alerts: ["Analytical Alerts Feed", "Real-time automated threshold alerts"],
    viva: ["Academic Methodology & Viva Mode", "Big Data Analytics vs Machine Learning component breakdown"]
  };

  if (titles[tabId]) {
    document.getElementById('page-title').innerText = titles[tabId][0];
    document.getElementById('page-subtitle').innerText = titles[tabId][1];
  }

  // Lazy Load Data per Tab
  if (tabId === 'overview') loadOverview();
  else if (tabId === 'fraud') loadFraudAnalytics();
  else if (tabId === 'customer360') loadCustomer360(100800);
  else if (tabId === 'segmentation') loadSegmentation();
  else if (tabId === 'anomaly') loadAnomalyCenter();
  else if (tabId === 'performance') loadModelPerformance();
  else if (tabId === 'time_geo') loadTimeGeoIntelligence();
  else if (tabId === 'explorer') loadTransactionExplorer();
  else if (tabId === 'quality') loadDataQuality();
  else if (tabId === 'processing') loadSparkMonitor();
  else if (tabId === 'alerts') loadAlerts('All');
}

// TAB 4: CUSTOMER 360
async function searchCustomer360() {
  const cid = document.getElementById('cust-search-id').value || 100800;
  loadCustomer360(cid);
}

// Chart Helper
function createOrUpdateChart(canvasId, type, data, options = {}) {
  if (charts[canvasId]) {
    charts[canvasId].destroy();
  }
  const ctx = document.getElementById(canvasId).getContext('2d');
  charts[canvasId] = new Chart(ctx, {
    type: type,
    data: data,
    options: Object.assign({
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: getComputedStyle(document.documentElement).getPropertyValue('--text-secondary').trim() } }
      }
    }, options)
  });
}

// TAB 1: OVERVIEW
async function loadOverview() {
  try {
    const [resSummary, resTrends] = await Promise.all([
      fetch('/api/summary').then(r => r.json()),
      fetch('/api/fraud/trends').then(r => r.json())
    ]);

    if (resSummary.error) return;

    document.getElementById('kpi-total-value').innerText = formatINR(resSummary.total_transaction_value);
    document.getElementById('kpi-total-txns').innerText = resSummary.total_transactions.toLocaleString();
    document.getElementById('kpi-total-customers').innerText = resSummary.total_customers.toLocaleString();
    document.getElementById('kpi-fraud-txns').innerText = resSummary.fraudulent_transactions.toLocaleString();
    document.getElementById('kpi-fraud-rate').innerText = `Rate: ${resSummary.fraud_rate}%`;
    document.getElementById('kpi-anomalies').innerText = (resSummary.detected_anomalies || 375018).toLocaleString();

    // Line Trend Chart
    const monthly = resTrends.monthly || [];
    const labels = monthly.map(m => m.month);
    const volumes = monthly.map(m => m.total_amount);
    const frauds = monthly.map(m => m.fraud_count);

    createOrUpdateChart('chart-overview-trends', 'line', {
      labels: labels,
      datasets: [
        {
          label: 'Total Volume (₹)',
          data: volumes,
          borderColor: '#0284c7',
          backgroundColor: 'rgba(2, 132, 199, 0.1)',
          fill: true,
          yAxisID: 'y'
        },
        {
          label: 'Fraud Count',
          data: frauds,
          borderColor: '#f43f5e',
          backgroundColor: 'rgba(244, 63, 94, 0.1)',
          fill: true,
          yAxisID: 'y1'
        }
      ]
    }, {
      scales: {
        y: { type: 'linear', position: 'left', grid: { color: 'rgba(255,255,255,0.05)' } },
        y1: { type: 'linear', position: 'right', grid: { drawOnChartArea: false } }
      }
    });

    // Doughnut Ratio Chart
    createOrUpdateChart('chart-overview-ratio', 'doughnut', {
      labels: ['Legitimate Transactions', 'Fraudulent Transactions'],
      datasets: [{
        data: [resSummary.total_transactions - resSummary.fraudulent_transactions, resSummary.fraudulent_transactions],
        backgroundColor: ['#10b981', '#f43f5e'],
        borderWidth: 0
      }]
    }, { cutout: '70%' });

  } catch (err) {
    console.error("Overview load error:", err);
  }
}

// TAB 2: FRAUD INTELLIGENCE
async function loadFraudAnalytics() {
  try {
    const data = await fetch('/api/fraud').then(r => r.json());
    if (data.error) return;

    // By Type
    const types = data.transaction_type || [];
    createOrUpdateChart('chart-fraud-by-type', 'bar', {
      labels: types.map(t => t.transaction_type),
      datasets: [{
        label: 'Fraud Count',
        data: types.map(t => t.fraud_count),
        backgroundColor: '#f43f5e'
      }]
    });

    // By Payment Method
    const pms = data.payment_method || [];
    createOrUpdateChart('chart-fraud-by-pm', 'bar', {
      labels: pms.map(p => p.payment_method),
      datasets: [{
        label: 'Fraud Count',
        data: pms.map(p => p.fraud_count),
        backgroundColor: '#f59e0b'
      }]
    });

    // By Location
    const locs = data.location || [];
    createOrUpdateChart('chart-fraud-by-location', 'bar', {
      labels: locs.map(l => l.location),
      datasets: [{
        label: 'Fraud Count',
        data: locs.map(l => l.fraud_count),
        backgroundColor: '#818cf8'
      }]
    });

    // By Device
    const devs = data.device_type || [];
    createOrUpdateChart('chart-fraud-by-device', 'doughnut', {
      labels: devs.map(d => d.device_type),
      datasets: [{
        data: devs.map(d => d.fraud_count),
        backgroundColor: ['#38bdf8', '#10b981', '#f43f5e', '#f59e0b', '#818cf8']
      }]
    });

    // Suspicious Feed Table
    const tbody = document.querySelector('#table-suspicious tbody');
    tbody.innerHTML = '';
    (data.suspicious_transactions || []).forEach(row => {
      const tr = document.createElement('tr');
      tr.onclick = () => openInvestigation(row.transaction_id);
      tr.innerHTML = `
        <td><strong>${row.transaction_id}</strong></td>
        <td>${row.customer_id}</td>
        <td>${row.transaction_date} ${row.transaction_time}</td>
        <td>${row.transaction_type}</td>
        <td><strong>${formatINR(row.amount)}</strong></td>
        <td>${row.location}</td>
        <td>${row.payment_method}</td>
        <td>${row.device_type}</td>
        <td><span class="badge badge-fraud">${row.risk_level}</span></td>
        <td><button class="btn btn-sm btn-outline"><i class="fa-solid fa-magnifying-glass"></i> Inspect</button></td>
      `;
      tbody.appendChild(tr);
    });

  } catch (err) {
    console.error("Fraud analytics load error:", err);
  }
}

// SLIDE-OVER INVESTIGATION PANEL
async function openInvestigation(transactionId) {
  try {
    const data = await fetch(`/api/fraud/investigation/${transactionId}`).then(r => r.json());
    if (data.error) return;

    const body = document.getElementById('slide-over-body');
    const overview = data.transaction_overview;
    const cust = data.customer_context;
    const pred = data.model_prediction;

    body.innerHTML = `
      <div class="detail-section">
        <h4>Transaction Overview</h4>
        <div class="detail-grid">
          <div class="detail-item"><span class="lbl">Txn ID</span><span class="val">${overview.transaction_id}</span></div>
          <div class="detail-item"><span class="lbl">Customer ID</span><span class="val">${overview.customer_id}</span></div>
          <div class="detail-item"><span class="lbl">Amount</span><span class="val">${formatINR(overview.amount)}</span></div>
          <div class="detail-item"><span class="lbl">Timestamp</span><span class="val">${overview.transaction_date} ${overview.transaction_time}</span></div>
          <div class="detail-item"><span class="lbl">Type & Method</span><span class="val">${overview.transaction_type} (${overview.payment_method})</span></div>
          <div class="detail-item"><span class="lbl">Location & Device</span><span class="val">${overview.location} / ${overview.device_type}</span></div>
          <div class="detail-item"><span class="lbl">Balance Before</span><span class="val">${formatINR(overview.balance_before)}</span></div>
          <div class="detail-item"><span class="lbl">Balance After</span><span class="val">${formatINR(overview.balance_after)}</span></div>
        </div>
      </div>

      <div class="detail-section">
        <h4>Customer 360 Context</h4>
        <div class="detail-grid">
          <div class="detail-item"><span class="lbl">Cluster Segment</span><span class="val">${cust.cluster_label}</span></div>
          <div class="detail-item"><span class="lbl">Total Spending</span><span class="val">${formatINR(cust.total_spending)}</span></div>
          <div class="detail-item"><span class="lbl">Historical Fraud Count</span><span class="val" style="color: var(--accent-rose);">${cust.fraud_count} cases</span></div>
          <div class="detail-item"><span class="lbl">Avg Balance</span><span class="val">${formatINR(cust.average_balance)}</span></div>
        </div>
      </div>

      <div class="detail-section">
        <h4>Model Inference & Risk Meter</h4>
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <span class="badge badge-fraud">${pred.prediction}</span>
          <span class="badge badge-warn">${pred.risk_level} RISK</span>
        </div>
        <div class="risk-meter-wrap">
          <div style="font-size: 1.6rem; font-weight: 700;">${pred.fraud_probability_pct}</div>
          <span style="font-size: 0.72rem; color: var(--text-muted);">Probability by ${pred.model_used}</span>
          <div class="risk-gauge-bar">
            <div class="risk-pointer" style="left: ${Math.min(100, Math.max(0, pred.fraud_probability * 100))}%;"></div>
          </div>
        </div>
      </div>

      <div class="detail-section">
        <h4>Triggered Risk Indicators</h4>
        <div style="display: flex; flex-direction: column; gap: 8px;">
          ${(data.risk_factors || []).map(f => `
            <div style="background-color: var(--bg-secondary); padding: 8px 12px; border-radius: 4px; border-left: 3px solid var(--accent-rose);">
              <div style="font-weight: 600; font-size: 0.8rem; color: var(--text-primary);">${f.factor} (${f.impact} Impact)</div>
              <div style="font-size: 0.72rem; color: var(--text-secondary);">${f.detail}</div>
            </div>
          `).join('')}
        </div>
      </div>
    `;

    document.getElementById('slide-over-backdrop').classList.add('active');

  } catch (err) {
    console.error("Investigation load error:", err);
  }
}

function closeSlideOver() {
  document.getElementById('slide-over-backdrop').classList.remove('active');
}

// TAB 3: RISK SIMULATOR
function loadPreset(presetName) {
  if (presetName === 'legit') {
    document.getElementById('pred-amount').value = 450.00;
    document.getElementById('pred-bal-before').value = 25000.00;
    document.getElementById('pred-bal-after').value = 24550.00;
    document.getElementById('pred-time').value = '14:30:00';
    document.getElementById('pred-type').value = 'UPI';
    document.getElementById('pred-payment').value = 'UPI';
    document.getElementById('pred-device').value = 'Android';
  } else if (presetName === 'night') {
    document.getElementById('pred-amount').value = 280000.00;
    document.getElementById('pred-bal-before').value = 300000.00;
    document.getElementById('pred-bal-after').value = 20000.00;
    document.getElementById('pred-time').value = '03:45:00';
    document.getElementById('pred-type').value = 'Bank Transfer';
    document.getElementById('pred-payment').value = 'Net Banking';
    document.getElementById('pred-device').value = 'Windows';
  } else if (presetName === 'drain') {
    document.getElementById('pred-amount').value = 95000.00;
    document.getElementById('pred-bal-before').value = 95000.00;
    document.getElementById('pred-bal-after').value = 0.00;
    document.getElementById('pred-time').value = '01:15:00';
    document.getElementById('pred-type').value = 'ATM Withdrawal';
    document.getElementById('pred-payment').value = 'ATM';
    document.getElementById('pred-device').value = 'ATM';
  }
}

async function handlePrediction(e) {
  e.preventDefault();
  const payload = {
    amount: parseFloat(document.getElementById('pred-amount').value),
    balance_before: parseFloat(document.getElementById('pred-bal-before').value),
    balance_after: parseFloat(document.getElementById('pred-bal-after').value),
    transaction_time: document.getElementById('pred-time').value,
    transaction_type: document.getElementById('pred-type').value,
    account_type: document.getElementById('pred-account').value,
    payment_method: document.getElementById('pred-payment').value,
    device_type: document.getElementById('pred-device').value,
    location: document.getElementById('pred-location').value
  };

  const modelChoice = document.getElementById('pred-model-choice').value;

  try {
    const data = await fetch(`/api/predict?model=${encodeURIComponent(modelChoice)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    }).then(r => r.json());

    if (data.error) return;

    document.getElementById('pred-idle').classList.add('hidden');
    document.getElementById('pred-result-box').classList.remove('hidden');

    const isFraud = data.prediction === 'Fraud';
    document.getElementById('res-badge').className = isFraud ? 'badge badge-fraud' : 'badge badge-legit';
    document.getElementById('res-badge').innerText = data.prediction.toUpperCase();

    document.getElementById('res-risk').innerText = `${data.risk_level} RISK`;
    document.getElementById('res-prob-val').innerText = data.fraud_probability_pct;
    document.getElementById('res-gauge-pointer').style.left = `${Math.min(100, Math.max(0, data.fraud_probability * 100))}%`;

    document.getElementById('res-classification').innerText = data.prediction;
    document.getElementById('res-prob').innerText = data.fraud_probability_pct;
    document.getElementById('res-risk-text').innerText = data.risk_level;
    document.getElementById('res-model').innerText = data.model_used;

  } catch (err) {
    console.error("Prediction error:", err);
  }
}

// TAB 4: CUSTOMER 360
async function searchCustomer360() {
  const cid = document.getElementById('cust-search-id').value || 10001;
  loadCustomer360(cid);
}

async function loadCustomer360(customerId) {
  try {
    const cust = await fetch(`/api/customer/${customerId}`).then(r => r.json());
    if (cust.error) {
      document.getElementById('cust-360-content').innerHTML = `<div class="card-header"><p style="color: var(--accent-rose);">Customer ID ${customerId} not found.</p></div>`;
      return;
    }

    const container = document.getElementById('cust-360-content');
    container.innerHTML = `
      <div class="customer-profile-header">
        <div class="cust-title-wrap">
          <h2>Customer ID #${cust.customer_id} Profile</h2>
          <p>Account Type: <strong>${cust.account_type}</strong> | Primary Location: <strong>${cust.primary_location}</strong></p>
          <div class="risk-flags-wrap">
            <span class="badge badge-info">${cust.cluster_label}</span>
            ${(cust.risk_flags || []).map(f => `<span class="badge badge-warn">${f}</span>`).join('')}
          </div>
        </div>
        <div style="text-align: right;">
          <span style="font-size: 0.72rem; color: var(--text-muted);">Total Lifetime Spending</span>
          <div style="font-size: 1.5rem; font-weight: 700; color: var(--accent-emerald);">${formatINR(cust.total_spending)}</div>
        </div>
      </div>

      <div class="kpi-grid">
        <div class="kpi-card">
          <div class="kpi-info"><span class="kpi-label">Avg Transaction Size</span><h3>${formatINR(cust.average_spending)}</h3></div>
        </div>
        <div class="kpi-card">
          <div class="kpi-info"><span class="kpi-label">Total Transactions</span><h3>${cust.transaction_count}</h3></div>
        </div>
        <div class="kpi-card">
          <div class="kpi-info"><span class="kpi-label">Avg Account Balance</span><h3>${formatINR(cust.average_balance)}</h3></div>
        </div>
        <div class="kpi-card">
          <div class="kpi-info"><span class="kpi-label">Unique Merchants</span><h3>${cust.unique_merchants}</h3></div>
        </div>
        <div class="kpi-card">
          <div class="kpi-info"><span class="kpi-label">Fraud Count</span><h3 style="color: var(--accent-rose);">${cust.fraud_count}</h3></div>
        </div>
      </div>

      <div class="table-card">
        <div class="card-header"><h3>Recent Customer Transactions</h3></div>
        <div class="table-responsive">
          <table class="data-table">
            <thead>
              <tr>
                <th>Txn ID</th>
                <th>Date & Time</th>
                <th>Type</th>
                <th>Amount</th>
                <th>Merchant</th>
                <th>Payment Method</th>
                <th>Fraud Status</th>
              </tr>
            </thead>
            <tbody>
              ${(cust.recent_transactions || []).map(t => `
                <tr onclick="openInvestigation('${t.transaction_id}')">
                  <td><strong>${t.transaction_id}</strong></td>
                  <td>${t.date} ${t.time}</td>
                  <td>${t.type}</td>
                  <td><strong>${formatINR(t.amount)}</strong></td>
                  <td>${t.merchant}</td>
                  <td>${t.payment_method}</td>
                  <td><span class="badge ${t.is_fraud ? 'badge-fraud' : 'badge-legit'}">${t.is_fraud ? 'FRAUD' : 'LEGIT'}</span></td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </div>
    `;

  } catch (err) {
    console.error("Customer 360 load error:", err);
  }
}

// TAB 5: CUSTOMER SEGMENTATION
async function loadSegmentation() {
  try {
    const data = await fetch('/api/clusters').then(r => r.json());
    if (data.error) return;

    // Elbow Curve
    createOrUpdateChart('chart-elbow', 'line', {
      labels: [2, 3, 4, 5, 6, 7, 8],
      datasets: [{
        label: 'Inertia (Sum of Squared Errors)',
        data: data.elbow_method || [12500, 7800, 4200, 3600, 3100, 2800, 2500],
        borderColor: '#38bdf8',
        backgroundColor: 'rgba(56, 189, 248, 0.1)',
        fill: true,
        tension: 0.3
      }]
    });

    // Cluster Pie
    const clusters = data.clusters || [];
    createOrUpdateChart('chart-cluster-pie', 'pie', {
      labels: clusters.map(c => c.cluster_name),
      datasets: [{
        data: clusters.map(c => c.count),
        backgroundColor: ['#38bdf8', '#10b981', '#f43f5e', '#f59e0b']
      }]
    });

    // Cluster Cards
    const container = document.getElementById('cluster-cards-container');
    container.innerHTML = '';
    clusters.forEach(c => {
      const card = document.createElement('div');
      card.className = 'cluster-card';
      card.innerHTML = `
        <div class="cluster-card-header">
          <h4>${c.cluster_name}</h4>
          <span class="badge badge-info">Cluster ${c.cluster_id}</span>
        </div>
        <div class="cluster-stat-list">
          <div class="cluster-stat-item"><span>Customer Count:</span><strong>${c.count.toLocaleString()} (${c.percentage}%)</strong></div>
          <div class="cluster-stat-item"><span>Avg Spending:</span><strong>${formatINR(c.avg_spending)}</strong></div>
          <div class="cluster-stat-item"><span>Avg Transactions:</span><strong>${c.avg_tx_count}</strong></div>
          <div class="cluster-stat-item"><span>Avg Balance:</span><strong>${formatINR(c.avg_balance)}</strong></div>
          <div class="cluster-stat-item"><span>Avg Fraud Count:</span><strong style="color: var(--accent-rose);">${c.avg_fraud_count}</strong></div>
        </div>
      `;
      container.appendChild(card);
    });

  } catch (err) {
    console.error("Segmentation load error:", err);
  }
}

// TAB 6: ANOMALY CENTER
async function loadAnomalyCenter() {
  try {
    const data = await fetch('/api/anomalies').then(r => r.json());
    if (data.error) return;

    document.getElementById('anomaly-total').innerText = (data.total_anomalies || 375018).toLocaleString();

    // Histogram
    const hist = data.score_distribution || [];
    createOrUpdateChart('chart-anomaly-dist', 'bar', {
      labels: hist.map(h => h.range),
      datasets: [{
        label: 'Transaction Count',
        data: hist.map(h => h.count),
        backgroundColor: '#f59e0b'
      }]
    });

    // Anomaly Table
    const tbody = document.querySelector('#table-anomalies tbody');
    tbody.innerHTML = '';
    (data.top_anomalies || []).forEach(row => {
      const tr = document.createElement('tr');
      tr.onclick = () => openInvestigation(row.transaction_id);
      tr.innerHTML = `
        <td><strong>${row.transaction_id}</strong></td>
        <td>${row.customer_id}</td>
        <td>${row.transaction_date} ${row.transaction_time}</td>
        <td>${row.transaction_type}</td>
        <td><strong>${formatINR(row.amount)}</strong></td>
        <td>${row.location}</td>
        <td>${row.payment_method}</td>
        <td><span class="badge badge-warn">${row.anomaly_score}</span></td>
        <td><span class="badge ${row.is_fraud ? 'badge-fraud' : 'badge-legit'}">${row.is_fraud ? 'FRAUD' : 'LEGIT'}</span></td>
      `;
      tbody.appendChild(tr);
    });

  } catch (err) {
    console.error("Anomaly load error:", err);
  }
}

// TAB 7: MODEL LAB
async function loadModelPerformance() {
  try {
    const data = await fetch('/api/model-performance').then(r => r.json());
    if (data.error) return;

    const models = data.models || [];
    const best = data.best_model || 'Logistic Regression';
    document.getElementById('best-model-title').innerText = best;

    // Benchmark Chart
    createOrUpdateChart('chart-model-comp', 'bar', {
      labels: models.map(m => m.name),
      datasets: [
        { label: 'Accuracy (%)', data: models.map(m => (m.accuracy * 100).toFixed(2)), backgroundColor: '#38bdf8' },
        { label: 'Precision (%)', data: models.map(m => (m.precision * 100).toFixed(2)), backgroundColor: '#10b981' },
        { label: 'Recall (%)', data: models.map(m => (m.recall * 100).toFixed(2)), backgroundColor: '#f59e0b' },
        { label: 'F1-Score (%)', data: models.map(m => (m.f1_score * 100).toFixed(2)), backgroundColor: '#f43f5e' },
        { label: 'ROC-AUC (%)', data: models.map(m => (m.roc_auc * 100).toFixed(2)), backgroundColor: '#818cf8' }
      ]
    });

    // Metric Table
    const tbody = document.querySelector('#table-model-metrics tbody');
    tbody.innerHTML = '';
    models.forEach(m => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${m.name}</strong> ${m.name === best ? '<span class="badge badge-legit">BEST</span>' : ''}</td>
        <td>${(m.accuracy * 100).toFixed(2)}%</td>
        <td>${(m.precision * 100).toFixed(2)}%</td>
        <td>${(m.recall * 100).toFixed(2)}%</td>
        <td><strong>${(m.f1_score * 100).toFixed(2)}%</strong></td>
        <td><strong>${(m.roc_auc * 100).toFixed(2)}%</strong></td>
      `;
      tbody.appendChild(tr);
    });

    // Confusion Matrices
    const cmContainer = document.getElementById('cm-cards-container');
    cmContainer.innerHTML = '';
    models.forEach(m => {
      const cm = m.confusion_matrix || [[0, 0], [0, 0]];
      const card = document.createElement('div');
      card.className = 'chart-card';
      card.innerHTML = `
        <div class="card-header"><h4>${m.name} Confusion Matrix</h4></div>
        <div class="detail-grid" style="grid-template-columns: repeat(2, 1fr); text-align: center; gap: 8px;">
          <div style="background-color: var(--accent-emerald-bg); padding: 12px; border-radius: 4px;">
            <span class="lbl">True Neg (Legit)</span>
            <div style="font-weight: 700; font-size: 1.1rem; color: var(--accent-emerald);">${cm[0][0].toLocaleString()}</div>
          </div>
          <div style="background-color: var(--accent-amber-bg); padding: 12px; border-radius: 4px;">
            <span class="lbl">False Pos (False Alarm)</span>
            <div style="font-weight: 700; font-size: 1.1rem; color: var(--accent-amber);">${cm[0][1].toLocaleString()}</div>
          </div>
          <div style="background-color: var(--accent-amber-bg); padding: 12px; border-radius: 4px;">
            <span class="lbl">False Neg (Missed Fraud)</span>
            <div style="font-weight: 700; font-size: 1.1rem; color: var(--accent-amber);">${cm[1][0].toLocaleString()}</div>
          </div>
          <div style="background-color: var(--accent-rose-bg); padding: 12px; border-radius: 4px;">
            <span class="lbl">True Pos (Caught Fraud)</span>
            <div style="font-weight: 700; font-size: 1.1rem; color: var(--accent-rose);">${cm[1][1].toLocaleString()}</div>
          </div>
        </div>
      `;
      cmContainer.appendChild(card);
    });

  } catch (err) {
    console.error("Model performance load error:", err);
  }
}

// TAB 8: TIME & GEO INTELLIGENCE
async function loadTimeGeoIntelligence() {
  try {
    const [resTime, resGeo] = await Promise.all([
      fetch('/api/time-analytics').then(r => r.json()),
      fetch('/api/geo-analytics').then(r => r.json())
    ]);

    // Render 7x24 Heatmap
    const heatmapContainer = document.getElementById('heatmap-grid-container');
    const heatmap = resTime.heatmap || [];
    
    let html = `<table class="heatmap-table"><thead><tr><th>Day / Hour</th>`;
    for (let h = 0; h < 24; h++) html += `<th>${h:02d}:00</th>`;
    html += `</tr></thead><tbody>`;

    heatmap.forEach(dayRow => {
      html += `<tr><td style="font-weight: 600; font-size: 0.75rem; color: var(--text-secondary);">${dayRow.day_name}</td>`;
      dayRow.hours.forEach(cell => {
        const frd = cell.fraud_count;
        let bg = 'rgba(16, 185, 129, 0.2)';
        if (frd > 100) bg = '#f43f5e';
        else if (frd > 40) bg = '#f59e0b';
        else if (frd > 10) bg = '#38bdf8';
        html += `<td class="heatmap-cell" style="background-color: ${bg};" title="${dayRow.day_name} ${cell.hour}:00 - ${frd} Fraud Cases">${frd}</td>`;
      });
      html += `</tr>`;
    });
    html += `</tbody></table>`;
    heatmapContainer.innerHTML = html;

    // Render Geo Ranking Table
    const tbody = document.querySelector('#table-geo-ranking tbody');
    tbody.innerHTML = '';
    (resGeo.locations || []).forEach(loc => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${loc.location}</strong></td>
        <td>${loc.count.toLocaleString()}</td>
        <td><strong>${formatINR(loc.total_amount)}</strong></td>
        <td>${formatINR(loc.avg_amount)}</td>
        <td style="color: var(--accent-rose); font-weight: 600;">${loc.fraud_count.toLocaleString()}</td>
        <td><span class="badge ${loc.fraud_rate > 1.2 ? 'badge-fraud' : 'badge-info'}">${loc.fraud_rate}%</span></td>
      `;
      tbody.appendChild(tr);
    });

  } catch (err) {
    console.error("Time & Geo load error:", err);
  }
}

// TAB 9: TRANSACTION EXPLORER
async function loadTransactionExplorer() {
  const queryParams = new URLSearchParams();
  queryParams.append('page', currentPage);
  queryParams.append('per_page', perPage);

  if (currentFilterState.search) queryParams.append('search', currentFilterState.search);
  if (currentFilterState.type) queryParams.append('transaction_type', currentFilterState.type);
  if (currentFilterState.pm) queryParams.append('payment_method', currentFilterState.pm);
  if (currentFilterState.fraud !== undefined && currentFilterState.fraud !== '') queryParams.append('is_fraud', currentFilterState.fraud);
  if (currentFilterState.minAmt) queryParams.append('min_amount', currentFilterState.minAmt);
  if (currentFilterState.maxAmt) queryParams.append('max_amount', currentFilterState.maxAmt);

  try {
    const data = await fetch(`/api/transactions?${queryParams.toString()}`).then(r => r.json());
    if (data.error) return;

    document.getElementById('explorer-total-count').innerText = `Total Filtered Records: ${data.total_records.toLocaleString()}`;
    document.getElementById('page-indicator').innerText = `Page ${data.page} of ${data.total_pages}`;

    const tbody = document.querySelector('#table-explorer tbody');
    tbody.innerHTML = '';
    (data.data || []).forEach(row => {
      const tr = document.createElement('tr');
      tr.onclick = () => openInvestigation(row.transaction_id);
      tr.innerHTML = `
        <td><strong>${row.transaction_id}</strong></td>
        <td>${row.customer_id}</td>
        <td>${row.transaction_date} ${row.transaction_time}</td>
        <td>${row.transaction_type}</td>
        <td>${row.account_type}</td>
        <td><strong>${formatINR(row.amount)}</strong></td>
        <td>${formatINR(row.balance_before)}</td>
        <td>${formatINR(row.balance_after)}</td>
        <td>${row.merchant}</td>
        <td>${row.location}</td>
        <td>${row.payment_method}</td>
        <td>${row.device_type}</td>
        <td><span class="badge ${row.is_fraud ? 'badge-fraud' : 'badge-legit'}">${row.is_fraud ? 'FRAUD' : 'SUCCESS'}</span></td>
      `;
      tbody.appendChild(tr);
    });

  } catch (err) {
    console.error("Explorer load error:", err);
  }
}

function triggerFilter() {
  currentFilterState = {
    search: document.getElementById('flt-search').value,
    type: document.getElementById('flt-type').value,
    pm: document.getElementById('flt-pm').value,
    fraud: document.getElementById('flt-fraud').value,
    minAmt: document.getElementById('flt-min-amt').value,
    maxAmt: document.getElementById('flt-max-amt').value
  };
  currentPage = 1;
  loadTransactionExplorer();
}

function changePage(delta) {
  currentPage = Math.max(1, currentPage + delta);
  loadTransactionExplorer();
}

function exportFilteredCSV() {
  const queryParams = new URLSearchParams();
  if (currentFilterState.search) queryParams.append('search', currentFilterState.search);
  if (currentFilterState.type) queryParams.append('transaction_type', currentFilterState.type);
  if (currentFilterState.pm) queryParams.append('payment_method', currentFilterState.pm);
  if (currentFilterState.fraud !== undefined && currentFilterState.fraud !== '') queryParams.append('is_fraud', currentFilterState.fraud);
  if (currentFilterState.minAmt) queryParams.append('min_amount', currentFilterState.minAmt);
  if (currentFilterState.maxAmt) queryParams.append('max_amount', currentFilterState.maxAmt);

  window.location.href = `/api/transactions/export?${queryParams.toString()}`;
}

// TAB 10: DATA QUALITY
async function loadDataQuality() {
  try {
    const dq = await fetch('/api/data-quality').then(r => r.json());
    if (dq.error) return;

    document.getElementById('dq-completeness').innerText = `${dq.completeness_score}%`;
    document.getElementById('dq-uniqueness').innerText = `${dq.uniqueness_score}%`;
    document.getElementById('dq-validity').innerText = `${dq.validity_score}%`;
    document.getElementById('dq-consistency').innerText = `${dq.consistency_score}%`;

    const grid = document.getElementById('dq-details-grid');
    grid.innerHTML = `
      <div class="detail-item"><span class="lbl">Raw File Name</span><span class="val">${dq.dataset_name}</span></div>
      <div class="detail-item"><span class="lbl">Total Transactions</span><span class="val">${dq.total_records.toLocaleString()}</span></div>
      <div class="detail-item"><span class="lbl">Raw File Size</span><span class="val">${dq.raw_file_size_gb} GB</span></div>
      <div class="detail-item"><span class="lbl">Missing Values</span><span class="val" style="color: var(--accent-emerald);">${dq.missing_values_count}</span></div>
      <div class="detail-item"><span class="lbl">Duplicates Removed</span><span class="val" style="color: var(--accent-emerald);">${dq.duplicate_records_count}</span></div>
      <div class="detail-item"><span class="lbl">Unique Customer Profiles</span><span class="val">${dq.unique_customers.toLocaleString()}</span></div>
      <div class="detail-item"><span class="lbl">Unique Merchants</span><span class="val">${dq.unique_merchants}</span></div>
      <div class="detail-item"><span class="lbl">Unique Locations</span><span class="val">${dq.unique_locations}</span></div>
      <div class="detail-item"><span class="lbl">Date Bounds</span><span class="val">${dq.date_range}</span></div>
    `;

  } catch (err) {
    console.error("Data quality load error:", err);
  }
}

// TAB 11: SPARK MONITOR
async function loadSparkMonitor() {
  try {
    const meta = await fetch('/api/processing').then(r => r.json());

    const grid = document.getElementById('spark-meta-grid');
    grid.innerHTML = `
      <div class="detail-item"><span class="lbl">Processing Engine</span><span class="val">${meta.engine}</span></div>
      <div class="detail-item"><span class="lbl">Total Records Processed</span><span class="val">${meta.records.toLocaleString()}</span></div>
      <div class="detail-item"><span class="lbl">Hadoop / HDFS Status</span><span class="val" style="color: var(--accent-rose);">${meta.hadoop}</span></div>
      <div class="detail-item"><span class="lbl">Parquet Export Size</span><span class="val">${meta.parquet_size_mb} MB</span></div>
      <div class="detail-item"><span class="lbl">PyArrow Streaming</span><span class="val">${meta.pyarrow_streaming}</span></div>
      <div class="detail-item"><span class="lbl">ML Stratified Training Sample</span><span class="val">${meta.ml_sample_records.toLocaleString()} records</span></div>
    `;

  } catch (err) {
    console.error("Spark monitor load error:", err);
  }
}

// TAB 12: ANALYTICAL ALERTS
async function loadAlerts(category) {
  try {
    const alerts = await fetch(`/api/alerts?category=${category}`).then(r => r.json());
    const container = document.getElementById('alerts-feed-container');
    container.innerHTML = '';

    alerts.forEach(a => {
      const card = document.createElement('div');
      card.className = 'chart-card';
      const color = a.severity === 'Critical' ? 'var(--accent-rose)' : a.severity === 'High' ? 'var(--accent-amber)' : 'var(--accent-primary)';
      card.style.borderLeft = `4px solid ${color}`;
      card.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
          <strong style="font-size: 0.95rem; color: var(--text-primary);"><i class="fa-solid fa-triangle-exclamation" style="color: ${color};"></i> ${a.title}</strong>
          <span class="badge ${a.severity === 'Critical' ? 'badge-fraud' : 'badge-warn'}">${a.severity}</span>
        </div>
        <p style="font-size: 0.82rem; color: var(--text-secondary); margin-bottom: 8px;">${a.description}</p>
        <div style="display: flex; justify-content: space-between; font-size: 0.72rem; color: var(--text-muted);">
          <span>Rule: <strong>${a.timestamp}</strong></span>
          <span>Metric: <strong>${a.metric}</strong></span>
        </div>
      `;
      container.appendChild(card);
    });

  } catch (err) {
    console.error("Alerts load error:", err);
  }
}

// REPORT MODAL
async function openReportModal() {
  try {
    const rpt = await fetch('/api/report').then(r => r.json());
    const body = document.getElementById('report-modal-body');

    body.innerHTML = `
      <div style="padding: 10px 0;">
        <h2 style="font-size: 1.4rem; font-weight: 700; color: var(--text-primary);">${rpt.title}</h2>
        <p style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 20px;">Generated on ${rpt.generated_at} | Source Dataset: ${rpt.data_quality.dataset_name}</p>

        <div class="detail-section" style="margin-bottom: 16px;">
          <h4>1. Executive Metrics Summary</h4>
          <div class="detail-grid">
            <div class="detail-item"><span class="lbl">Total Volume</span><span class="val">${formatINR(rpt.summary.total_transaction_value)}</span></div>
            <div class="detail-item"><span class="lbl">Total Transactions</span><span class="val">${rpt.summary.total_transactions.toLocaleString()}</span></div>
            <div class="detail-item"><span class="lbl">Profiled Accounts</span><span class="val">${rpt.summary.total_customers.toLocaleString()}</span></div>
            <div class="detail-item"><span class="lbl">Fraud Cases</span><span class="val">${rpt.summary.fraudulent_transactions.toLocaleString()} (${rpt.summary.fraud_rate}%)</span></div>
          </div>
        </div>

        <div class="detail-section" style="margin-bottom: 16px;">
          <h4>2. Machine Learning Algorithm Comparison</h4>
          <div class="detail-grid">
            <div class="detail-item"><span class="lbl">Best Classifier</span><span class="val">${rpt.model_performance.best_model}</span></div>
            <div class="detail-item"><span class="lbl">Training Partition</span><span class="val">${rpt.processing_metadata.ml_sample_records.toLocaleString()} records</span></div>
          </div>
        </div>

        <div class="detail-section">
          <h4>3. Big Data Engineering Compliance</h4>
          <div class="detail-grid">
            <div class="detail-item"><span class="lbl">Engine</span><span class="val">${rpt.processing_metadata.engine}</span></div>
            <div class="detail-item"><span class="lbl">Parquet Export Size</span><span class="val">${rpt.processing_metadata.parquet_size_mb} MB</span></div>
          </div>
        </div>
      </div>
    `;

    document.getElementById('report-modal-backdrop').classList.add('active');

  } catch (err) {
    console.error("Report load error:", err);
  }
}

function closeReportModal() {
  document.getElementById('report-modal-backdrop').classList.remove('active');
}
