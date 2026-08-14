// Banking Fraud Analytics Dashboard App Logic

let charts = {};
let currentExplorerPage = 1;
let explorerFilterTimeout = null;

document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  loadOverviewTab();
});

// NAVIGATION
function initNavigation() {
  const navItems = document.querySelectorAll('.nav-item');
  navItems.forEach(item => {
    item.addEventListener('click', (e) => {
      e.preventDefault();
      const tabId = item.getAttribute('data-tab');

      // Update Nav active
      navItems.forEach(n => n.classList.remove('active'));
      item.classList.add('active');

      // Update Tab content
      document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
      const activeTab = document.getElementById(`tab-${tabId}`);
      if (activeTab) activeTab.classList.add('active');

      // Update Header title
      updateHeaderTitle(tabId);

      // Load section data
      loadTabContent(tabId);
    });
  });
}

function updateHeaderTitle(tabId) {
  const titleMap = {
    overview: ["Analytics Overview", "Real-time enterprise banking transactions & ML insights"],
    fraud: ["Fraud Pattern Analytics", "Categorical & spatial distribution of fraudulent behavior"],
    prediction: ["Real-Time Fraud Inference", "Predict fraud probability with trained classification models"],
    segmentation: ["Customer Segmentation (K-Means)", "Behavioral profiling and cluster analysis"],
    anomaly: ["Unsupervised Anomaly Detection", "Isolation Forest statistical outlier identification"],
    performance: ["Model Performance Benchmark", "Comparative evaluation of 3 Machine Learning algorithms"],
    explorer: ["Transaction Explorer", "Searchable multi-filtered enterprise transaction table"]
  };

  if (titleMap[tabId]) {
    document.getElementById('page-title').innerText = titleMap[tabId][0];
    document.getElementById('page-subtitle').innerText = titleMap[tabId][1];
  }
}

function loadTabContent(tabId) {
  switch(tabId) {
    case 'overview': loadOverviewTab(); break;
    case 'fraud': loadFraudTab(); break;
    case 'prediction': break; // Form based
    case 'segmentation': loadSegmentationTab(); break;
    case 'anomaly': loadAnomalyTab(); break;
    case 'performance': loadPerformanceTab(); break;
    case 'explorer': loadExplorerTab(); break;
  }
}

function refreshDashboard() {
  const activeNav = document.querySelector('.nav-item.active');
  const tabId = activeNav ? activeNav.getAttribute('data-tab') : 'overview';
  loadTabContent(tabId);
}

// 1. OVERVIEW TAB
async function loadOverviewTab() {
  try {
    const [summaryRes, trendsRes] = await Promise.all([
      fetch('/api/summary').then(r => r.json()),
      fetch('/api/fraud/trends').then(r => r.json())
    ]);

    if (summaryRes.error) return;

    document.getElementById('kpi-total-txns').innerText = summaryRes.total_transactions ? summaryRes.total_transactions.toLocaleString() : '--';
    document.getElementById('kpi-total-value').innerText = summaryRes.total_transaction_value ? '₹' + summaryRes.total_transaction_value.toLocaleString() : '--';
    document.getElementById('kpi-total-customers').innerText = summaryRes.total_customers ? summaryRes.total_customers.toLocaleString() : '--';
    document.getElementById('kpi-fraud-txns').innerText = summaryRes.fraudulent_transactions ? summaryRes.fraudulent_transactions.toLocaleString() : '--';
    document.getElementById('kpi-fraud-rate').innerText = `Fraud Rate: ${summaryRes.fraud_rate || 0}%`;
    document.getElementById('kpi-anomalies').innerText = summaryRes.detected_anomalies ? summaryRes.detected_anomalies.toLocaleString() : '--';

    // Trends Chart
    if (trendsRes.monthly) {
      const labels = trendsRes.monthly.map(m => m.month);
      const txCounts = trendsRes.monthly.map(m => m.total_transactions);
      const fraudCounts = trendsRes.monthly.map(m => m.fraud_count);

      renderChart('chart-trends', 'line', {
        labels: labels,
        datasets: [
          {
            label: 'Total Volume',
            data: txCounts,
            borderColor: '#06b6d4',
            backgroundColor: 'rgba(6, 182, 212, 0.1)',
            fill: true,
            tension: 0.3
          },
          {
            label: 'Fraud Count',
            data: fraudCounts,
            borderColor: '#f43f5e',
            backgroundColor: 'rgba(244, 63, 94, 0.2)',
            fill: true,
            tension: 0.3
          }
        ]
      });
    }

    // Ratio Doughnut Chart
    if (summaryRes.total_transactions) {
      const legit = summaryRes.total_transactions - summaryRes.fraudulent_transactions;
      renderChart('chart-fraud-ratio', 'doughnut', {
        labels: ['Legitimate', 'Fraudulent'],
        datasets: [{
          data: [legit, summaryRes.fraudulent_transactions],
          backgroundColor: ['#10b981', '#f43f5e'],
          borderWidth: 0
        }]
      });
    }

  } catch (err) {
    console.error('Error loading overview:', err);
  }
}

// 2. FRAUD ANALYTICS TAB
async function loadFraudTab() {
  try {
    const res = await fetch('/api/fraud').then(r => r.json());
    if (res.error) return;

    // Fraud by Type
    if (res.transaction_type) {
      renderChart('chart-fraud-by-type', 'bar', {
        labels: res.transaction_type.map(d => d.transaction_type),
        datasets: [{
          label: 'Fraud Count',
          data: res.transaction_type.map(d => d.fraud_count),
          backgroundColor: '#6366f1'
        }]
      });
    }

    // Fraud by Payment Method
    if (res.payment_method) {
      renderChart('chart-fraud-by-pm', 'bar', {
        labels: res.payment_method.map(d => d.payment_method),
        datasets: [{
          label: 'Fraud Count',
          data: res.payment_method.map(d => d.fraud_count),
          backgroundColor: '#06b6d4'
        }]
      });
    }

    // Fraud by Location
    if (res.location) {
      renderChart('chart-fraud-by-location', 'bar', {
        labels: res.location.map(d => d.location),
        datasets: [{
          label: 'Fraud Count',
          data: res.location.map(d => d.fraud_count),
          backgroundColor: '#f59e0b'
        }]
      });
    }

    // Fraud by Device Type
    if (res.device_type) {
      renderChart('chart-fraud-by-device', 'doughnut', {
        labels: res.device_type.map(d => d.device_type),
        datasets: [{
          data: res.device_type.map(d => d.fraud_count),
          backgroundColor: ['#0284c7', '#10b981', '#6366f1', '#f59e0b', '#f43f5e']
        }]
      });
    }

    // Suspicious Table
    if (res.suspicious_transactions) {
      const tbody = document.querySelector('#table-suspicious tbody');
      tbody.innerHTML = '';
      res.suspicious_transactions.forEach(row => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><code>${row.transaction_id}</code></td>
          <td>${row.customer_id}</td>
          <td>${row.transaction_date} ${row.transaction_time}</td>
          <td>${row.transaction_type}</td>
          <td><strong>₹${row.amount.toLocaleString()}</strong></td>
          <td>${row.merchant}</td>
          <td>${row.location}</td>
          <td>${row.payment_method}</td>
          <td>${row.device_type}</td>
          <td><span class="badge badge-fraud">${row.status}</span></td>
        `;
        tbody.appendChild(tr);
      });
    }

  } catch (err) {
    console.error('Error loading fraud tab:', err);
  }
}

// 3. FRAUD PREDICTION PRESETS & FORM
function loadPreset(type) {
  if (type === 'legit') {
    document.getElementById('pred-amount').value = '450.00';
    document.getElementById('pred-bal-before').value = '50000.00';
    document.getElementById('pred-bal-after').value = '49550.00';
    document.getElementById('pred-time').value = '14:30:00';
    document.getElementById('pred-type').value = 'Card Payment';
    document.getElementById('pred-account').value = 'Savings';
    document.getElementById('pred-payment').value = 'Debit Card';
    document.getElementById('pred-device').value = 'Android';
    document.getElementById('pred-location').value = 'Bengaluru';
  } else if (type === 'night') {
    document.getElementById('pred-amount').value = '125000.00';
    document.getElementById('pred-bal-before').value = '130000.00';
    document.getElementById('pred-bal-after').value = '5000.00';
    document.getElementById('pred-time').value = '03:45:00';
    document.getElementById('pred-type').value = 'Bank Transfer';
    document.getElementById('pred-account').value = 'Salary';
    document.getElementById('pred-payment').value = 'Net Banking';
    document.getElementById('pred-device').value = 'Windows';
    document.getElementById('pred-location').value = 'Mumbai';
  } else if (type === 'drain') {
    document.getElementById('pred-amount').value = '295000.00';
    document.getElementById('pred-bal-before').value = '295000.00';
    document.getElementById('pred-bal-after').value = '0.00';
    document.getElementById('pred-time').value = '02:10:00';
    document.getElementById('pred-type').value = 'ATM Withdrawal';
    document.getElementById('pred-account').value = 'Savings';
    document.getElementById('pred-payment').value = 'ATM';
    document.getElementById('pred-device').value = 'ATM';
    document.getElementById('pred-location').value = 'Delhi';
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
    location: document.getElementById('pred-location').value,
    transaction_date: new Date().toISOString().split('T')[0]
  };

  const modelChoice = document.getElementById('pred-model-choice').value;

  try {
    const res = await fetch(`/api/predict?model=${encodeURIComponent(modelChoice)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    }).then(r => r.json());

    if (res.error) {
      alert(res.error);
      return;
    }

    document.getElementById('pred-idle').classList.add('hidden');
    const resultBox = document.getElementById('pred-result-box');
    resultBox.classList.remove('hidden');

    const badge = document.getElementById('res-badge');
    const riskBadge = document.getElementById('res-risk');
    const probVal = document.getElementById('res-prob-val');
    const circle = document.getElementById('res-circle');

    badge.innerText = res.prediction.toUpperCase();
    if (res.prediction === 'Fraud') {
      badge.className = 'prediction-badge badge-fraud';
      circle.style.borderColor = '#f43f5e';
    } else {
      badge.className = 'prediction-badge badge-legit';
      circle.style.borderColor = '#10b981';
    }

    riskBadge.innerText = `${res.risk_level.toUpperCase()} RISK`;
    if (res.risk_level === 'High') riskBadge.className = 'risk-badge badge-fraud';
    else if (res.risk_level === 'Medium') riskBadge.className = 'risk-badge badge-warn';
    else riskBadge.className = 'risk-badge badge-legit';

    probVal.innerText = res.fraud_probability_pct;
    document.getElementById('res-classification').innerText = res.prediction;
    document.getElementById('res-prob').innerText = res.fraud_probability_pct;
    document.getElementById('res-risk-text').innerText = res.risk_level;
    document.getElementById('res-model').innerText = res.model_used;

  } catch (err) {
    console.error('Prediction error:', err);
  }
}

// 4. CUSTOMER SEGMENTATION TAB
async function loadSegmentationTab() {
  try {
    const res = await fetch('/api/clusters').then(r => r.json());
    if (res.error) return;

    // Elbow Curve
    if (res.elbow_curve) {
      renderChart('chart-elbow', 'line', {
        labels: res.elbow_curve.map(d => `K=${d.k}`),
        datasets: [{
          label: 'Inertia (Sum of Squared Errors)',
          data: res.elbow_curve.map(d => d.inertia),
          borderColor: '#6366f1',
          backgroundColor: 'rgba(99, 102, 241, 0.1)',
          fill: true,
          tension: 0.2
        }]
      });
    }

    // Segment Distribution
    if (res.cluster_profiles) {
      renderChart('chart-cluster-pie', 'pie', {
        labels: res.cluster_profiles.map(c => c.label),
        datasets: [{
          data: res.cluster_profiles.map(c => c.count),
          backgroundColor: ['#0284c7', '#10b981', '#f59e0b', '#f43f5e']
        }]
      });

      // Render Cards
      const container = document.getElementById('cluster-cards-container');
      container.innerHTML = '';
      res.cluster_profiles.forEach(c => {
        const card = document.createElement('div');
        card.className = 'cluster-card';
        card.innerHTML = `
          <div class="cluster-card-header">
            <h4>${c.label}</h4>
            <span class="badge badge-legit">Cluster ${c.cluster_id}</span>
          </div>
          <div class="cluster-stat-list">
            <div class="cluster-stat-item"><span>Customer Count:</span> <strong>${c.count.toLocaleString()} (${c.percentage}%)</strong></div>
            <div class="cluster-stat-item"><span>Avg Txn Amount:</span> <strong>₹${c.stats.average_transaction_amount.toLocaleString()}</strong></div>
            <div class="cluster-stat-item"><span>Total Spend:</span> <strong>₹${c.stats.total_transaction_amount.toLocaleString()}</strong></div>
            <div class="cluster-stat-item"><span>Txn Count:</span> <strong>${c.stats.transaction_count}</strong></div>
            <div class="cluster-stat-item"><span>Fraud Count:</span> <strong>${c.stats.fraud_count}</strong></div>
          </div>
        `;
        container.appendChild(card);
      });
    }

  } catch (err) {
    console.error('Error loading segmentation tab:', err);
  }
}

// 5. ANOMALY DETECTION TAB
async function loadAnomalyTab() {
  try {
    const res = await fetch('/api/anomalies').then(r => r.json());
    if (res.error) return;

    document.getElementById('anomaly-total').innerText = res.total_anomalies ? res.total_anomalies.toLocaleString() : '--';
    document.getElementById('anomaly-pct').innerText = `${res.anomaly_percentage || 0}%`;

    // Histogram chart
    if (res.score_distribution) {
      renderChart('chart-anomaly-dist', 'bar', {
        labels: res.score_distribution.map(d => `${d.bin_start} to ${d.bin_end}`),
        datasets: [{
          label: 'Transaction Count',
          data: res.score_distribution.map(d => d.count),
          backgroundColor: '#f59e0b'
        }]
      });
    }

    // Top Anomalies Table
    if (res.top_suspicious) {
      const tbody = document.querySelector('#table-anomalies tbody');
      tbody.innerHTML = '';
      res.top_suspicious.forEach(row => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><code>${row.transaction_id}</code></td>
          <td>${row.customer_id}</td>
          <td>${row.transaction_date} ${row.transaction_time}</td>
          <td>${row.transaction_type}</td>
          <td><strong>₹${row.amount.toLocaleString()}</strong></td>
          <td>${row.location}</td>
          <td>${row.payment_method}</td>
          <td><span class="badge badge-warn">${row.anomaly_score}</span></td>
          <td>${row.is_fraud ? '<span class="badge badge-fraud">Fraud</span>' : '<span class="badge badge-legit">Legit</span>'}</td>
        `;
        tbody.appendChild(tr);
      });
    }

  } catch (err) {
    console.error('Error loading anomaly tab:', err);
  }
}

// 6. MODEL PERFORMANCE TAB
async function loadPerformanceTab() {
  try {
    const res = await fetch('/api/model-performance').then(r => r.json());
    if (res.error) return;

    const modelsData = res.models;
    if (modelsData) {
      const modelNames = Object.keys(modelsData);
      const metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'];
      
      const datasets = metrics.map((m, idx) => {
        const colors = ['#0284c7', '#10b981', '#f59e0b', '#6366f1', '#f43f5e'];
        return {
          label: m.toUpperCase().replace('_', '-'),
          data: modelNames.map(n => modelsData[n][m]),
          backgroundColor: colors[idx]
        };
      });

      renderChart('chart-model-comp', 'bar', {
        labels: modelNames,
        datasets: datasets
      });

      // Render Table
      const tbody = document.querySelector('#table-model-metrics tbody');
      tbody.innerHTML = '';
      modelNames.forEach(name => {
        const m = modelsData[name];
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><strong>${name}</strong></td>
          <td>${(m.accuracy * 100).toFixed(2)}%</td>
          <td>${(m.precision * 100).toFixed(2)}%</td>
          <td>${(m.recall * 100).toFixed(2)}%</td>
          <td><strong>${(m.f1_score * 100).toFixed(2)}%</strong></td>
          <td>${(m.roc_auc * 100).toFixed(2)}%</td>
        `;
        tbody.appendChild(tr);
      });

      // Render Confusion Matrices Cards
      const cmContainer = document.getElementById('cm-cards-container');
      cmContainer.innerHTML = '';
      modelNames.forEach(name => {
        const cm = modelsData[name].confusion_matrix; // [[TN, FP], [FN, TP]]
        const card = document.createElement('div');
        card.className = 'cm-card';
        card.innerHTML = `
          <h4>${name} Confusion Matrix</h4>
          <div class="cm-box-grid">
            <div class="cm-box cm-tn"><span class="val">${cm[0][0].toLocaleString()}</span><span class="lbl">True Negative</span></div>
            <div class="cm-box cm-fp"><span class="val">${cm[0][1].toLocaleString()}</span><span class="lbl">False Positive</span></div>
            <div class="cm-box cm-fn"><span class="val">${cm[1][0].toLocaleString()}</span><span class="lbl">False Negative</span></div>
            <div class="cm-box cm-tp"><span class="val">${cm[1][1].toLocaleString()}</span><span class="lbl">True Positive</span></div>
          </div>
        `;
        cmContainer.appendChild(card);
      });

      if (res.best_model) {
        document.getElementById('best-model-title').innerText = res.best_model;
      }
    }

  } catch (err) {
    console.error('Error loading performance tab:', err);
  }
}

// 7. TRANSACTION EXPLORER TAB
async function loadExplorerTab(page = 1) {
  currentExplorerPage = page;
  const search = document.getElementById('flt-search').value;
  const type = document.getElementById('flt-type').value;
  const pm = document.getElementById('flt-pm').value;
  const fraud = document.getElementById('flt-fraud').value;
  const minAmt = document.getElementById('flt-min-amt').value;
  const maxAmt = document.getElementById('flt-max-amt').value;

  let url = `/api/transactions?page=${page}&per_page=20`;
  if (search) url += `&search=${encodeURIComponent(search)}`;
  if (type) url += `&transaction_type=${encodeURIComponent(type)}`;
  if (pm) url += `&payment_method=${encodeURIComponent(pm)}`;
  if (fraud) url += `&is_fraud=${fraud}`;
  if (minAmt) url += `&min_amount=${minAmt}`;
  if (maxAmt) url += `&max_amount=${maxAmt}`;

  try {
    const res = await fetch(url).then(r => r.json());
    if (res.error) return;

    document.getElementById('explorer-total-count').innerText = `Total Found: ${res.total_records.toLocaleString()}`;
    document.getElementById('page-indicator').innerText = `Page ${res.page} of ${res.total_pages}`;

    const tbody = document.querySelector('#table-explorer tbody');
    tbody.innerHTML = '';
    res.data.forEach(row => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><code>${row.transaction_id}</code></td>
        <td>${row.customer_id}</td>
        <td>${row.transaction_date} ${row.transaction_time}</td>
        <td>${row.transaction_type}</td>
        <td>${row.account_type}</td>
        <td><strong>₹${row.amount.toLocaleString()}</strong></td>
        <td>₹${row.balance_before.toLocaleString()}</td>
        <td>₹${row.balance_after.toLocaleString()}</td>
        <td>${row.merchant}</td>
        <td>${row.location}</td>
        <td>${row.payment_method}</td>
        <td>${row.device_type}</td>
        <td>${row.is_fraud ? '<span class="badge badge-fraud">Fraud</span>' : '<span class="badge badge-legit">Completed</span>'}</td>
      `;
      tbody.appendChild(tr);
    });

  } catch (err) {
    console.error('Error loading explorer tab:', err);
  }
}

function triggerFilter() {
  clearTimeout(explorerFilterTimeout);
  explorerFilterTimeout = setTimeout(() => {
    loadExplorerTab(1);
  }, 300);
}

function changePage(delta) {
  loadExplorerTab(currentExplorerPage + delta);
}

// HELPER: CHART.JS WRAPPER
function renderChart(canvasId, type, data, options = {}) {
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
        legend: { labels: { color: '#94a3b8' } }
      },
      scales: (type === 'bar' || type === 'line') ? {
        x: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } },
        y: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } }
      } : {}
    }, options)
  });
}
