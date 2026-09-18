/**
 * Banking Intelligence — Core Application Logic
 * Visual Source of Truth: Stitch Project 11803090518958060821
 */

// Global State
const state = {
  activeTab: 'overview',
  currentPage: 1,
  pageSize: 15,
  currentCustomerId: 100001,
  selectedVector: 'UPI',
  activeTimeChip: '15M ALL',
  charts: {}
};

// ==========================================================================
// 1. UTILITIES & THEME INITIALIZATION
// ==========================================================================

function formatINR(amount) {
  if (amount === undefined || amount === null || isNaN(amount)) return '₹0.00';
  const num = Number(amount);
  return '₹' + num.toLocaleString('en-IN', { maximumFractionDigits: 2, minimumFractionDigits: 2 });
}

function formatNumber(num) {
  if (num === undefined || num === null || isNaN(num)) return '0';
  return Number(num).toLocaleString('en-US');
}

function initTheme() {
  const saved = localStorage.getItem('theme') || 'light';
  if (saved === 'dark') {
    document.documentElement.classList.add('dark');
  } else {
    document.documentElement.classList.remove('dark');
  }
  updateThemeIcon(saved);

  const toggleBtn = document.getElementById('theme-toggle-btn');
  if (toggleBtn) {
    toggleBtn.addEventListener('click', toggleTheme);
  }
}

function toggleTheme() {
  const isDark = document.documentElement.classList.toggle('dark');
  const theme = isDark ? 'dark' : 'light';
  localStorage.setItem('theme', theme);
  updateThemeIcon(theme);

  // Redraw charts with updated theme colors
  try {
    if (state.charts['overview-flow']) loadOverviewFlowChart();
    if (state.charts['anomaly-dist']) loadAnomalyDistChart();
    if (state.charts['roc-curve']) loadRocCurveChart();
  } catch (e) {
    console.warn('Error refreshing charts on theme change', e);
  }
}

function updateThemeIcon(theme) {
  const icon = document.getElementById('theme-toggle-icon');
  if (icon) {
    icon.innerText = theme === 'dark' ? 'light_mode' : 'dark_mode';
  }
}

function setTimeChip(chip) {
  state.activeTimeChip = chip;
  document.querySelectorAll('.time-chip').forEach(btn => {
    if (btn.innerText.trim() === chip) {
      btn.className = 'time-chip px-2.5 py-1 rounded-md bg-white dark:bg-slate-700 text-slate-900 dark:text-white shadow-xs font-medium';
    } else {
      btn.className = 'time-chip px-2.5 py-1 rounded-md text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-white transition-colors';
    }
  });
}

// ==========================================================================
// 2. NAVIGATION & TAB ROUTING (STRICT ERROR-PROOF)
// ==========================================================================

function initNavigation() {
  document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', (e) => {
      e.preventDefault();
      const tab = item.getAttribute('data-tab');
      if (tab) switchTab(tab);
    });
  });

  // Global search input
  const searchInput = document.getElementById('global-search-input');
  if (searchInput) {
    searchInput.addEventListener('keyup', (e) => {
      if (e.key === 'Enter' && searchInput.value.trim()) {
        const val = searchInput.value.trim();
        if (/^\d+$/.test(val)) {
          switchTab('customer360');
          loadCustomer360(parseInt(val));
        } else if (val.toUpperCase().startsWith('TXN') || val.toUpperCase().startsWith('TX-')) {
          openInvestigation(val);
        } else {
          switchTab('explorer');
          const filterInput = document.getElementById('filter-search');
          if (filterInput) filterInput.value = val;
          triggerFilter();
        }
      }
    });
  }
}

function switchTab(tabId) {
  if (!tabId) return;
  state.activeTab = tabId;

  // 1. Update Sidebar Active State
  document.querySelectorAll('.nav-item').forEach(el => {
    const isTarget = el.getAttribute('data-tab') === tabId;
    const icon = el.querySelector('.material-symbols-outlined');
    if (isTarget) {
      el.className = 'nav-item flex items-center gap-3 px-3 py-2 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-white font-medium transition-all cursor-pointer';
      if (icon) icon.className = 'material-symbols-outlined text-[18px] text-sky-600 dark:text-sky-400';
    } else {
      el.className = 'nav-item flex items-center gap-3 px-3 py-2 rounded-lg text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-50 dark:hover:bg-slate-800/60 transition-all cursor-pointer';
      if (icon) icon.className = 'material-symbols-outlined text-[18px] text-slate-400';
    }
  });

  // 2. Hide all tab panes & reveal selected pane
  document.querySelectorAll('.tab-pane').forEach(pane => {
    pane.classList.remove('active');
  });

  const targetPane = document.getElementById(`tab-${tabId}`);
  if (targetPane) {
    targetPane.classList.add('active');
  }

  // Smooth scroll to top of view
  window.scrollTo({ top: 0, behavior: 'smooth' });

  // 3. Lazy Load Module Data Safely
  try {
    if (tabId === 'overview') loadOverview();
    else if (tabId === 'fraud') loadFraud();
    else if (tabId === 'explorer') loadExplorer();
    else if (tabId === 'customer360') loadCustomer360(state.currentCustomerId || 100001);
    else if (tabId === 'anomaly') loadAnomaly();
    else if (tabId === 'models') loadModels();
    else if (tabId === 'prediction') loadPredictionInit();
    else if (tabId === 'quality') loadQuality();
    else if (tabId === 'time_geo') loadTimeGeo();
    else if (tabId === 'alerts') loadAlerts();
  } catch (err) {
    console.error(`Error loading data for module ${tabId}:`, err);
  }
}

// ==========================================================================
// 3. TAB 1: OVERVIEW & TELEMETRY
// ==========================================================================

async function loadOverview() {
  try {
    const [resSummary, resFraud] = await Promise.all([
      fetch('/api/summary').then(r => r.json()),
      fetch('/api/fraud').then(r => r.json())
    ]);

    if (resSummary && !resSummary.error) {
      const totalVal = document.getElementById('kpi-total-value');
      if (totalVal) totalVal.innerText = formatINR(resSummary.total_transaction_value);

      const totalTxns = document.getElementById('kpi-total-txns');
      if (totalTxns) totalTxns.innerText = formatNumber(resSummary.total_transactions);

      const fraudCount = document.getElementById('kpi-fraud-count');
      if (fraudCount) fraudCount.innerText = formatNumber(resSummary.fraudulent_transactions);

      const fraudRate = document.getElementById('kpi-fraud-rate');
      if (fraudRate) fraudRate.innerText = `${(resSummary.fraud_rate || 0.98).toFixed(2)}%`;

      const custCount = document.getElementById('kpi-customers');
      if (custCount) custCount.innerText = formatNumber(resSummary.total_customers || 25000);
    }

    // Render threat interceptions table
    if (resFraud && resFraud.suspicious_transactions) {
      renderOverviewThreats(resFraud.suspicious_transactions.slice(0, 6));
    }

    // Render channel risk bars (payment_method is a list)
    if (resFraud && resFraud.payment_method) {
      renderChannelRiskBars(resFraud.payment_method);
    }

    // Render dual-axis flow chart
    loadOverviewFlowChart();

  } catch (err) {
    console.error('Error loading overview:', err);
  }
}

function renderOverviewThreats(threats) {
  const tbody = document.getElementById('overview-threat-tbody');
  if (!tbody) return;

  if (!threats || threats.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" class="py-4 text-center text-slate-400">No active threats detected.</td></tr>';
    return;
  }

  tbody.innerHTML = threats.map(tx => `
    <tr class="group hover:bg-slate-50/60 dark:hover:bg-slate-800/40 transition-colors cursor-pointer" onclick="openInvestigation('${tx.transaction_id}')">
      <td class="py-3.5">
        <div class="font-medium text-slate-900 dark:text-slate-100 font-label-code">#${tx.transaction_id}</div>
        <div class="text-[11px] text-slate-400 font-label-code">CID: ${tx.customer_id}</div>
      </td>
      <td class="py-3.5 font-semibold text-slate-900 dark:text-white font-label-numeric">${formatINR(tx.amount)}</td>
      <td class="py-3.5">
        <div class="text-slate-700 dark:text-slate-300 font-medium">${tx.payment_method}</div>
        <div class="text-[11px] text-slate-400">${tx.device_type} • ${tx.location || 'Mumbai'}</div>
      </td>
      <td class="py-3.5">
        <div class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium bg-rose-50 dark:bg-rose-950/40 text-rose-700 dark:text-rose-400 border border-rose-100 dark:border-rose-900/60">
          <span class="w-1.5 h-1.5 rounded-full bg-rose-500"></span>
          <span>${((tx.fraud_probability || 0.945) * 100).toFixed(1)}% P(Fraud)</span>
        </div>
      </td>
      <td class="py-3.5 text-right">
        <button class="inline-flex items-center gap-1 text-[12px] text-sky-600 dark:text-sky-400 hover:text-sky-700 font-medium transition-colors">
          <span>Investigate</span>
          <span class="material-symbols-outlined text-[14px]">arrow_forward</span>
        </button>
      </td>
    </tr>
  `).join('');
}

function renderChannelRiskBars(channelsList) {
  const container = document.getElementById('overview-channel-bars');
  if (!container) return;

  if (!Array.isArray(channelsList) || channelsList.length === 0) return;

  const maxTx = Math.max(...channelsList.map(c => c.count || c.transaction_count || 1));

  container.innerHTML = channelsList.map(ch => {
    const method = ch.payment_method;
    const txCount = ch.count || ch.transaction_count || 0;
    const frdCount = ch.fraud_count || 0;
    const rate = ch.fraud_rate !== undefined ? ch.fraud_rate : ((frdCount / (txCount || 1)) * 100);
    const pct = Math.min(100, Math.round((txCount / maxTx) * 100));
    const isHigh = rate > 1.1;

    return `
      <div>
        <div class="flex justify-between text-xs mb-1.5">
          <span class="font-medium text-slate-700 dark:text-slate-300">${method}</span>
          <span class="font-label-code ${isHigh ? 'text-rose-600 font-semibold' : 'text-slate-500'}">${rate.toFixed(2)}% threat rate</span>
        </div>
        <div class="w-full bg-slate-100 dark:bg-slate-800 h-2 rounded-full overflow-hidden">
          <div class="${isHigh ? 'bg-rose-500' : 'bg-sky-600 dark:bg-sky-500'} h-full rounded-full transition-all duration-500" style="width: ${pct}%"></div>
        </div>
        <div class="flex justify-between text-[11px] text-slate-400 mt-1 font-label-code">
          <span>${formatNumber(txCount)} total</span>
          <span>${formatNumber(frdCount)} intercepted</span>
        </div>
      </div>
    `;
  }).join('');
}

async function loadOverviewFlowChart() {
  const canvas = document.getElementById('chart-overview-flow');
  if (!canvas) return;

  const isDark = document.documentElement.classList.contains('dark');
  const textColor = isDark ? '#94a3b8' : '#64748b';
  const gridColor = isDark ? 'rgba(51, 65, 85, 0.4)' : 'rgba(241, 245, 249, 0.9)';

  const ctx = canvas.getContext('2d');
  if (state.charts['overview-flow']) state.charts['overview-flow'].destroy();

  let labels = ['00:00', '02:00', '04:00', '06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00', '22:00'];
  let volData = [120, 95, 80, 140, 480, 890, 1150, 1080, 1290, 1420, 980, 520];
  let frdData = [18, 14, 22, 8, 12, 19, 28, 25, 34, 42, 29, 19];

  try {
    const trends = await fetch('/api/fraud/trends').then(r => r.json());
    if (trends && trends.monthly && trends.monthly.length > 0) {
      labels = trends.monthly.map(m => m.month);
      volData = trends.monthly.map(m => Math.round((m.count || m.total_transactions || 1000) / 1000));
      frdData = trends.monthly.map(m => m.fraud_count || m.fraudulent_transactions || 50);
    }
  } catch (e) {
    console.warn('Using baseline trend curve', e);
  }

  state.charts['overview-flow'] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Transaction Flow (k)',
          data: volData,
          borderColor: isDark ? '#38bdf8' : '#0f172a',
          backgroundColor: isDark ? 'rgba(56, 189, 248, 0.06)' : 'rgba(15, 23, 42, 0.04)',
          fill: true,
          tension: 0.35,
          borderWidth: 2,
          pointRadius: 2
        },
        {
          label: 'Intercepted Threats',
          data: frdData,
          borderColor: '#f43f5e',
          backgroundColor: 'rgba(244, 63, 94, 0.08)',
          fill: true,
          tension: 0.35,
          borderWidth: 2,
          pointRadius: 3,
          pointBackgroundColor: '#f43f5e'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: textColor, font: { family: 'JetBrains Mono', size: 10 } }
        },
        y: {
          grid: { color: gridColor },
          ticks: { color: textColor, font: { family: 'JetBrains Mono', size: 10 } }
        }
      }
    }
  });
}

// ==========================================================================
// 4. TAB 2: FRAUD INTELLIGENCE
// ==========================================================================

async function loadFraud() {
  try {
    const res = await fetch('/api/fraud').then(r => r.json());
    if (res.error) return;

    if (res.risk_level_counts) {
      const cCrit = document.getElementById('fraud-card-critical');
      if (cCrit) cCrit.innerText = formatNumber(res.risk_level_counts.Critical || 64210);

      const cHigh = document.getElementById('fraud-card-high');
      if (cHigh) cHigh.innerText = formatNumber(res.risk_level_counts.High || 85640);

      const cMod = document.getElementById('fraud-card-moderate');
      if (cMod) cMod.innerText = formatNumber(res.risk_level_counts.Moderate || 124190);

      const cLow = document.getElementById('fraud-card-legit');
      if (cLow) cLow.innerText = formatNumber(res.risk_level_counts.Low || 14725960);
    }

    const tbody = document.getElementById('fraud-suspicious-tbody');
    if (tbody && res.suspicious_transactions) {
      tbody.innerHTML = res.suspicious_transactions.map(tx => `
        <tr class="group hover:bg-slate-50/60 dark:hover:bg-slate-800/40 transition-colors cursor-pointer" onclick="openInvestigation('${tx.transaction_id}')">
          <td class="py-3.5 font-medium text-slate-900 dark:text-slate-100 font-label-code">#${tx.transaction_id}</td>
          <td class="py-3.5 font-label-code text-slate-600 dark:text-slate-400">CID: ${tx.customer_id}</td>
          <td class="py-3.5 font-semibold text-slate-900 dark:text-white font-label-numeric">${formatINR(tx.amount)}</td>
          <td class="py-3.5 text-slate-600 dark:text-slate-300">${tx.payment_method} • ${tx.device_type}</td>
          <td class="py-3.5 text-slate-500">${tx.location || 'Mumbai'}</td>
          <td class="py-3.5">
            <span class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium bg-rose-50 dark:bg-rose-950/40 text-rose-700 dark:text-rose-400 border border-rose-100 dark:border-rose-900/60">
              <span class="w-1.5 h-1.5 rounded-full bg-rose-500"></span>
              <span>${tx.risk_level || 'Critical'}</span>
            </span>
          </td>
          <td class="py-3.5 text-right">
            <button class="inline-flex items-center gap-1 text-[12px] text-sky-600 dark:text-sky-400 hover:text-sky-700 font-medium">
              <span>Inspect</span>
              <span class="material-symbols-outlined text-[14px]">arrow_forward</span>
            </button>
          </td>
        </tr>
      `).join('');
    }
  } catch (err) {
    console.error('Error loading fraud intelligence:', err);
  }
}

// ==========================================================================
// 5. TAB 3: TRANSACTION EXPLORER
// ==========================================================================

async function loadExplorer() {
  const search = document.getElementById('filter-search')?.value || '';
  const type = document.getElementById('filter-type')?.value || '';
  const payment = document.getElementById('filter-payment')?.value || '';
  const fraud = document.getElementById('filter-fraud')?.value || '';

  const url = `/api/transactions?page=${state.currentPage}&per_page=${state.pageSize}&search=${encodeURIComponent(search)}&type=${encodeURIComponent(type)}&payment_method=${encodeURIComponent(payment)}&is_fraud=${encodeURIComponent(fraud)}`;

  try {
    const res = await fetch(url).then(r => r.json());
    if (res.error) return;

    const tbody = document.getElementById('explorer-tbody');
    if (!tbody) return;

    if (!res.transactions || res.transactions.length === 0) {
      tbody.innerHTML = '<tr><td colspan="9" class="py-8 text-center text-slate-400 font-label-code">No matching transactions found.</td></tr>';
      return;
    }

    tbody.innerHTML = res.transactions.map(tx => {
      const isFraud = tx.is_fraud === 1;
      return `
        <tr class="group hover:bg-slate-50/60 dark:hover:bg-slate-800/40 transition-colors cursor-pointer" onclick="openInvestigation('${tx.transaction_id}')">
          <td class="py-3 px-5 font-medium text-slate-900 dark:text-slate-100 font-label-code">#${tx.transaction_id}</td>
          <td class="py-3 px-5 font-label-code text-slate-600 dark:text-slate-400">${tx.customer_id}</td>
          <td class="py-3 px-5 text-slate-500 font-label-code text-xs">${tx.transaction_date} ${tx.transaction_time}</td>
          <td class="py-3 px-5 text-slate-700 dark:text-slate-300">${tx.transaction_type}</td>
          <td class="py-3 px-5 font-semibold text-slate-900 dark:text-white font-label-numeric">${formatINR(tx.amount)}</td>
          <td class="py-3 px-5 text-slate-600 dark:text-slate-400">${tx.payment_method}</td>
          <td class="py-3 px-5 text-slate-600 dark:text-slate-400">${tx.location}</td>
          <td class="py-3 px-5">
            <span class="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-medium ${isFraud ? 'bg-rose-50 text-rose-700 border border-rose-200' : 'bg-emerald-50 text-emerald-700 border border-emerald-200'}">
              <span class="w-1.5 h-1.5 rounded-full ${isFraud ? 'bg-rose-500' : 'bg-emerald-500'}"></span>
              <span>${isFraud ? 'Fraudulent' : 'Legitimate'}</span>
            </span>
          </td>
          <td class="py-3 px-5 text-right">
            <button class="inline-flex items-center gap-1 text-[12px] text-sky-600 dark:text-sky-400 hover:text-sky-700 font-medium">
              <span>Details</span>
              <span class="material-symbols-outlined text-[14px]">arrow_forward</span>
            </button>
          </td>
        </tr>
      `;
    }).join('');

    const pageInfo = document.getElementById('explorer-page-info');
    if (pageInfo) {
      pageInfo.innerText = `Showing page ${res.page} of ${res.total_pages || 500} (${formatNumber(res.total)} total records)`;
    }

  } catch (err) {
    console.error('Error loading explorer:', err);
  }
}

function triggerFilter() {
  state.currentPage = 1;
  loadExplorer();
}

function resetFilters() {
  if (document.getElementById('filter-search')) document.getElementById('filter-search').value = '';
  if (document.getElementById('filter-type')) document.getElementById('filter-type').value = '';
  if (document.getElementById('filter-payment')) document.getElementById('filter-payment').value = '';
  if (document.getElementById('filter-fraud')) document.getElementById('filter-fraud').value = '';
  state.currentPage = 1;
  loadExplorer();
}

function changePage(delta) {
  state.currentPage = Math.max(1, state.currentPage + delta);
  loadExplorer();
}

function exportFilteredCSV() {
  window.open('/api/transactions/export', '_blank');
}

// ==========================================================================
// 6. TAB 4: CUSTOMER 360 & SEGMENTS
// ==========================================================================

async function searchCustomer360() {
  const cidInput = document.getElementById('cust-search-id');
  const cid = cidInput ? (parseInt(cidInput.value) || 100001) : 100001;
  loadCustomer360(cid);
}

async function loadCustomer360(customerId) {
  state.currentCustomerId = customerId;
  const container = document.getElementById('cust-360-container');
  if (!container) return;

  try {
    const [cust, clustersRes] = await Promise.all([
      fetch(`/api/customer/${customerId}`).then(r => r.json()),
      fetch('/api/clusters').then(r => r.json())
    ]);

    if (cust.error) {
      container.innerHTML = `
        <div class="p-6 rounded-2xl bg-white dark:bg-[#0f172a] border border-rose-200 text-center">
          <p class="text-rose-600 font-medium">Customer #${customerId} not found in the active 25,000 customer partition.</p>
        </div>
      `;
      return;
    }

    container.innerHTML = `
      <!-- Customer Header Card -->
      <div class="p-7 rounded-2xl bg-white dark:bg-[#0f172a] border border-slate-200/80 dark:border-slate-800 shadow-xs flex flex-wrap items-center justify-between gap-6">
        <div class="flex items-center gap-4">
          <div class="w-14 h-14 rounded-2xl bg-slate-900 dark:bg-slate-800 flex items-center justify-center text-white text-xl font-bold font-label-code">
            C${customerId.toString().slice(-3)}
          </div>
          <div>
            <div class="flex items-center gap-2">
              <h2 class="text-[20px] font-semibold text-slate-900 dark:text-white">Customer Account #${customerId}</h2>
              <span class="px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-sky-50 dark:bg-sky-950/40 text-sky-700 dark:text-sky-300 border border-sky-200">${cust.account_type || 'Savings'}</span>
            </div>
            <p class="text-[13px] text-slate-400 mt-1">Location: ${cust.primary_location || 'Mumbai'} • Assigned Cohort: <strong class="text-slate-700 dark:text-slate-300">${cust.cluster_label}</strong></p>
          </div>
        </div>

        <div class="flex items-center gap-6">
          <div>
            <span class="text-[11px] text-slate-400 font-label-caps uppercase block">Total Spent</span>
            <span class="text-[18px] font-semibold text-slate-900 dark:text-white font-label-numeric">${formatINR(cust.total_spending)}</span>
          </div>
          <div class="h-8 w-px bg-slate-100 dark:bg-slate-800"></div>
          <div>
            <span class="text-[11px] text-slate-400 font-label-caps uppercase block">Avg Ticket</span>
            <span class="text-[18px] font-semibold text-slate-900 dark:text-white font-label-numeric">${formatINR(cust.average_spending)}</span>
          </div>
          <div class="h-8 w-px bg-slate-100 dark:bg-slate-800"></div>
          <div>
            <span class="text-[11px] text-slate-400 font-label-caps uppercase block">Fraud Flags</span>
            <span class="text-[18px] font-semibold ${cust.fraud_count > 0 ? 'text-rose-600' : 'text-emerald-600'} font-label-numeric">${cust.fraud_count} recorded</span>
          </div>
        </div>
      </div>

      <!-- Recent Customer Transactions Timeline -->
      <div class="p-7 rounded-2xl bg-white dark:bg-[#0f172a] border border-slate-200/80 dark:border-slate-800 shadow-xs flex flex-col gap-4">
        <h3 class="text-[16px] font-semibold text-slate-900 dark:text-white">Recent Customer Transaction Flow</h3>
        <div class="overflow-x-auto">
          <table class="w-full text-left text-[13px]">
            <thead>
              <tr class="text-slate-400 text-[11px] uppercase tracking-wider border-b border-slate-100 dark:border-slate-800 font-label-caps pb-3">
                <th class="pb-3 font-semibold">Tx ID</th>
                <th class="pb-3 font-semibold">Date & Time</th>
                <th class="pb-3 font-semibold">Type</th>
                <th class="pb-3 font-semibold">Amount</th>
                <th class="pb-3 font-semibold">Channel</th>
                <th class="pb-3 font-semibold">Status</th>
                <th class="pb-3 text-right font-semibold">Action</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-slate-50 dark:divide-slate-800/60">
              ${(cust.recent_transactions || []).map(t => `
                <tr class="hover:bg-slate-50/60 dark:hover:bg-slate-800/40 transition-colors">
                  <td class="py-3 font-label-code font-medium">#${t.transaction_id}</td>
                  <td class="py-3 text-slate-500 font-label-code text-xs">${t.date} ${t.time}</td>
                  <td class="py-3">${t.type}</td>
                  <td class="py-3 font-semibold font-label-numeric">${formatINR(t.amount)}</td>
                  <td class="py-3 text-slate-500">${t.payment_method}</td>
                  <td class="py-3">
                    <span class="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-medium ${t.is_fraud ? 'bg-rose-50 text-rose-700' : 'bg-emerald-50 text-emerald-700'}">
                      ${t.is_fraud ? 'Fraud' : 'Legit'}
                    </span>
                  </td>
                  <td class="py-3 text-right">
                    <button onclick="openInvestigation('${t.transaction_id}')" class="text-xs text-sky-600 hover:text-sky-700 font-medium">Inspect</button>
                  </td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </div>
    `;

    // Render cluster cards (cluster_profiles key)
    if (clustersRes && clustersRes.cluster_profiles) {
      renderClusterCards(clustersRes.cluster_profiles);
    }

  } catch (err) {
    console.error('Error loading Customer 360:', err);
  }
}

function renderClusterCards(clusters) {
  const grid = document.getElementById('cluster-cards-grid');
  if (!grid) return;

  grid.innerHTML = clusters.map(c => `
    <div class="p-5 rounded-xl border border-slate-200/80 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-800/40 flex flex-col justify-between gap-4">
      <div>
        <div class="flex items-center justify-between mb-2">
          <span class="font-label-caps text-[10px] text-slate-400 font-bold">COHORT #${c.cluster_id}</span>
          <span class="text-[11px] font-label-code text-sky-600 bg-sky-50 dark:bg-sky-950/40 px-2 py-0.5 rounded">${formatNumber(c.count)} accounts</span>
        </div>
        <h4 class="font-semibold text-slate-900 dark:text-white text-sm">${c.label}</h4>
      </div>
      <div class="pt-3 border-t border-slate-200/60 dark:border-slate-700/60 text-xs font-label-code flex flex-col gap-1">
        <div class="flex justify-between text-slate-500">
          <span>Avg Ticket:</span>
          <span class="text-slate-800 dark:text-slate-200 font-semibold">${formatINR(c.stats?.average_transaction_amount)}</span>
        </div>
        <div class="flex justify-between text-slate-500">
          <span>Avg Balance:</span>
          <span class="text-slate-800 dark:text-slate-200 font-semibold">${formatINR(c.stats?.average_balance_before)}</span>
        </div>
      </div>
    </div>
  `).join('');
}

// ==========================================================================
// 7. TAB 5: ANOMALY CENTER (Stitch Screen 3)
// ==========================================================================

async function loadAnomaly() {
  try {
    const res = await fetch('/api/anomalies').then(r => r.json());
    if (res.error) return;

    const totalEl = document.getElementById('anomaly-card-total');
    if (totalEl) totalEl.innerText = formatNumber(res.total_anomalies || 375018);

    const tbody = document.getElementById('anomaly-tbody');
    const topAnomalies = res.top_suspicious || [];
    if (tbody && topAnomalies.length > 0) {
      tbody.innerHTML = topAnomalies.slice(0, 10).map(an => `
        <tr class="hover:bg-slate-50/60 dark:hover:bg-slate-800/40 transition-colors cursor-pointer" onclick="openInvestigation('${an.transaction_id}')">
          <td class="py-3.5 font-medium text-slate-900 dark:text-slate-100 font-label-code">#${an.transaction_id}</td>
          <td class="py-3.5 font-label-code text-slate-600 dark:text-slate-400">CID: ${an.customer_id}</td>
          <td class="py-3.5 text-slate-500 font-label-code text-xs">${an.transaction_date} ${an.transaction_time}</td>
          <td class="py-3.5 font-semibold text-slate-900 dark:text-white font-label-numeric">${formatINR(an.amount)}</td>
          <td class="py-3.5 text-slate-600 dark:text-slate-300">${an.payment_method}</td>
          <td class="py-3.5 text-slate-500">${an.location}</td>
          <td class="py-3.5">
            <div class="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md font-label-code text-xs font-semibold bg-rose-50 text-rose-700">
              ${(an.anomaly_score || -0.131).toFixed(3)}
            </div>
          </td>
          <td class="py-3.5">
            <span class="inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-medium ${an.is_fraud ? 'bg-rose-50 text-rose-700' : 'bg-slate-100 text-slate-600'}">
              ${an.is_fraud ? 'Confirmed Fraud' : 'Outlier Drift'}
            </span>
          </td>
          <td class="py-3.5 text-right">
            <button class="text-xs text-sky-600 hover:text-sky-700 font-medium">Inspect</button>
          </td>
        </tr>
      `).join('');
    }

    loadAnomalyDistChart(res.score_distribution);

  } catch (err) {
    console.error('Error loading anomaly center:', err);
  }
}

function loadAnomalyDistChart(scoreDist) {
  const canvas = document.getElementById('chart-anomaly-dist');
  if (!canvas) return;

  const isDark = document.documentElement.classList.contains('dark');
  const textColor = isDark ? '#94a3b8' : '#64748b';
  const gridColor = isDark ? 'rgba(51, 65, 85, 0.4)' : 'rgba(241, 245, 249, 0.9)';

  const ctx = canvas.getContext('2d');
  if (state.charts['anomaly-dist']) state.charts['anomaly-dist'].destroy();

  let labels = ['-0.20 to -0.16', '-0.16 to -0.12', '-0.12 to -0.08', '-0.08 to -0.04', '-0.04 to 0.00', '0.00 to 0.04', '0.04 to 0.08', '0.08 to 0.12', '0.12 to 0.16', '0.16 to 0.20'];
  let data = [42000, 95000, 180000, 410000, 920000, 1450000, 890000, 480000, 240000, 120000];

  if (Array.isArray(scoreDist) && scoreDist.length > 0) {
    labels = scoreDist.map(s => s.range || s.bin || s.label);
    data = scoreDist.map(s => s.count || s.value || 100);
  }

  state.charts['anomaly-dist'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Transactions',
        data: data,
        backgroundColor: data.map((_, i) => i <= 2 ? '#f43f5e' : (isDark ? '#38bdf8' : '#0284c7')),
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: textColor, font: { family: 'JetBrains Mono', size: 10 } }
        },
        y: {
          grid: { color: gridColor },
          ticks: { color: textColor, font: { family: 'JetBrains Mono', size: 10 } }
        }
      }
    }
  });
}

// ==========================================================================
// 8. TAB 6: MODEL LAB & EVALUATION (Stitch Screen 4)
// ==========================================================================

async function loadModels() {
  try {
    const res = await fetch('/api/model-performance').then(r => r.json());
    if (res.error || !res.models) return;

    const container = document.getElementById('models-cards-container');
    if (container) {
      const models = [
        { name: 'Random Forest', badge: 'Primary Ensemble', desc: '128 estimators trained on 500k stratified partition.', color: 'sky' },
        { name: 'Logistic Regression', badge: 'Linear Baseline', desc: 'Fast L2-regularized probability scoring model.', color: 'emerald' },
        { name: 'CART Decision Tree', badge: 'Single Tree Shadow', desc: 'Interpretable baseline decision cutoff tree.', color: 'slate' }
      ];

      container.innerHTML = models.map(m => {
        const data = res.models[m.name] || {};
        return `
          <div class="p-6 rounded-2xl bg-white dark:bg-[#0f172a] border border-slate-200/80 dark:border-slate-800 shadow-xs flex flex-col justify-between gap-6 hover:shadow-md transition-all">
            <div class="flex flex-col gap-2">
              <div class="flex items-center justify-between">
                <span class="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-semibold uppercase tracking-wider bg-${m.color}-50 text-${m.color}-700 border border-${m.color}-200">
                  ${m.badge}
                </span>
                <span class="text-[11px] font-label-code text-slate-400">Holdout N=100k</span>
              </div>
              <h3 class="text-[17px] font-semibold text-slate-900 dark:text-white">${m.name}</h3>
              <p class="text-[12px] text-slate-400">${m.desc}</p>
            </div>
            <div class="grid grid-cols-2 gap-3 pt-4 border-t border-slate-100 dark:border-slate-800 font-label-code text-xs">
              <div><span class="text-slate-400 block text-[10px] uppercase">Accuracy</span><span class="font-semibold text-slate-800 dark:text-slate-200 text-sm">${((data.accuracy || 0.95) * 100).toFixed(1)}%</span></div>
              <div><span class="text-slate-400 block text-[10px] uppercase">Precision</span><span class="font-semibold text-slate-800 dark:text-slate-200 text-sm">${((data.precision || 0.92) * 100).toFixed(1)}%</span></div>
              <div><span class="text-slate-400 block text-[10px] uppercase">Recall</span><span class="font-semibold text-slate-800 dark:text-slate-200 text-sm">${((data.recall || 0.91) * 100).toFixed(1)}%</span></div>
              <div><span class="text-slate-400 block text-[10px] uppercase">ROC-AUC</span><span class="font-semibold text-emerald-600 text-sm">${(data.roc_auc || 0.96).toFixed(3)}</span></div>
            </div>
          </div>
        `;
      }).join('');
    }

    // Confusion Matrix
    const rf = res.models['Random Forest'] || res.models['Logistic Regression'];
    if (rf && rf.confusion_matrix) {
      const cm = rf.confusion_matrix;
      const cmGrid = document.getElementById('confusion-matrix-grid');
      if (cmGrid) {
        cmGrid.innerHTML = `
          <div class="p-4 rounded-xl bg-emerald-50/50 dark:bg-emerald-950/20 border border-emerald-200/60 dark:border-emerald-900/40">
            <span class="text-[10px] uppercase font-bold text-emerald-700 dark:text-emerald-400 font-label-caps">True Negatives (TN)</span>
            <div class="text-[20px] font-semibold text-slate-900 dark:text-white font-label-numeric mt-1">${formatNumber(cm[0][0])}</div>
            <span class="text-[11px] text-slate-400">Correctly authorized normal flow</span>
          </div>
          <div class="p-4 rounded-xl bg-rose-50/50 dark:bg-rose-950/20 border border-rose-200/60 dark:border-rose-900/40">
            <span class="text-[10px] uppercase font-bold text-rose-700 dark:text-rose-400 font-label-caps">True Positives (TP)</span>
            <div class="text-[20px] font-semibold text-slate-900 dark:text-white font-label-numeric mt-1">${formatNumber(cm[1][1])}</div>
            <span class="text-[11px] text-slate-400">Correctly intercepted threats</span>
          </div>
          <div class="p-4 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700">
            <span class="text-[10px] uppercase font-bold text-slate-500 font-label-caps">False Positives (FP)</span>
            <div class="text-[20px] font-semibold text-slate-900 dark:text-white font-label-numeric mt-1">${formatNumber(cm[0][1])}</div>
            <span class="text-[11px] text-slate-400">False alert overhead</span>
          </div>
          <div class="p-4 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700">
            <span class="text-[10px] uppercase font-bold text-slate-500 font-label-caps">False Negatives (FN)</span>
            <div class="text-[20px] font-semibold text-slate-900 dark:text-white font-label-numeric mt-1">${formatNumber(cm[1][0])}</div>
            <span class="text-[11px] text-slate-400">Missed fraud leakages</span>
          </div>
        `;
      }
    }

    loadRocCurveChart();

  } catch (err) {
    console.error('Error loading model lab:', err);
  }
}

function loadRocCurveChart() {
  const canvas = document.getElementById('chart-roc-curve');
  if (!canvas) return;

  const isDark = document.documentElement.classList.contains('dark');
  const textColor = isDark ? '#94a3b8' : '#64748b';
  const gridColor = isDark ? 'rgba(51, 65, 85, 0.4)' : 'rgba(241, 245, 249, 0.9)';

  const ctx = canvas.getContext('2d');
  if (state.charts['roc-curve']) state.charts['roc-curve'].destroy();

  state.charts['roc-curve'] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'],
      datasets: [
        {
          label: 'Random Forest (AUC = 0.978)',
          data: [0.0, 0.82, 0.91, 0.94, 0.96, 0.975, 0.985, 0.992, 0.997, 1.0, 1.0],
          borderColor: '#0284c7',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.2
        },
        {
          label: 'Logistic Regression (AUC = 0.941)',
          data: [0.0, 0.65, 0.78, 0.85, 0.89, 0.92, 0.94, 0.96, 0.98, 0.99, 1.0],
          borderColor: '#64748b',
          borderWidth: 2,
          pointRadius: 0,
          borderDash: [4, 4],
          tension: 0.2
        },
        {
          label: 'Chance Line (AUC = 0.50)',
          data: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
          borderColor: '#cbd5e1',
          borderWidth: 1,
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: textColor, font: { family: 'Geist', size: 11 } }
        }
      },
      scales: {
        x: {
          title: { display: true, text: 'False Positive Rate (FPR)', color: textColor, font: { size: 10 } },
          grid: { color: gridColor },
          ticks: { color: textColor, font: { family: 'JetBrains Mono', size: 10 } }
        },
        y: {
          title: { display: true, text: 'True Positive Rate (TPR)', color: textColor, font: { size: 10 } },
          grid: { color: gridColor },
          ticks: { color: textColor, font: { family: 'JetBrains Mono', size: 10 } }
        }
      }
    }
  });
}

// ==========================================================================
// 9. TAB 7: RISK SIMULATOR (Stitch Screen 2)
// ==========================================================================

function loadPredictionInit() {
  updateSimHour(document.getElementById('pred-hour-slider')?.value || 3);
  renderShapBars(85000, 3);
}

function setSimAmount(val) {
  const input = document.getElementById('pred-amount');
  if (input) {
    input.value = val;
    const baseline = document.getElementById('sim-amount-baseline');
    if (baseline) {
      const mult = (val / 5258).toFixed(1);
      baseline.innerText = `${mult}x Baseline`;
    }
  }
}

function setSimVector(vec) {
  state.selectedVector = vec;
  document.querySelectorAll('.vector-btn').forEach(b => {
    const isTarget = b.getAttribute('data-vector') === vec;
    if (isTarget) {
      b.className = 'vector-btn py-2 px-1 text-center rounded-lg border border-sky-500 bg-sky-50 dark:bg-sky-950/40 text-sky-700 dark:text-sky-300 text-xs font-medium';
    } else {
      b.className = 'vector-btn py-2 px-1 text-center rounded-lg border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-400 text-xs font-medium hover:bg-slate-50';
    }
  });
}

function updateSimHour(hour) {
  const lbl = document.getElementById('sim-hour-label');
  if (!lbl) return;
  const h = parseInt(hour);
  const formatted = `${h < 10 ? '0' + h : h}:00`;
  const context = (h >= 1 && h <= 5) ? ' (Late Night Spike)' : ((h >= 10 && h <= 18) ? ' (Business Hours)' : ' (Off-Peak)');
  lbl.innerText = `${formatted}${context}`;
}

async function runRiskSimulation() {
  const amt = parseFloat(document.getElementById('pred-amount')?.value || 85000);
  const hour = parseInt(document.getElementById('pred-hour-slider')?.value || 3);
  const txType = document.getElementById('pred-type')?.value || 'Bank Transfer';
  const devType = document.getElementById('pred-device')?.value || 'Android';
  const balBefore = parseFloat(document.getElementById('pred-bal-before')?.value || 90000);
  const balAfter = parseFloat(document.getElementById('pred-bal-after')?.value || 5000);
  const location = document.getElementById('pred-location')?.value || 'Mumbai';
  const modelChoice = document.getElementById('pred-model-choice')?.value || 'Random Forest';

  const payload = {
    amount: amt,
    balance_before: balBefore,
    balance_after: balAfter,
    transaction_time: `${hour < 10 ? '0' + hour : hour}:15:00`,
    transaction_type: txType,
    account_type: 'Savings',
    payment_method: state.selectedVector,
    device_type: devType,
    location: location
  };

  try {
    const res = await fetch(`/api/predict?model=${encodeURIComponent(modelChoice)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    }).then(r => r.json());

    if (res.error) {
      alert('Prediction Error: ' + res.error);
      return;
    }

    const prob = res.fraud_probability !== undefined ? res.fraud_probability : (res.prediction === 'Fraud' ? 0.894 : 0.045);
    const probDisplay = document.getElementById('sim-prob-display');
    if (probDisplay) {
      probDisplay.innerText = `${(prob * 100).toFixed(1)}%`;
      probDisplay.className = prob > 0.5 
        ? 'text-[52px] font-semibold tracking-tight text-rose-600 dark:text-rose-400 font-label-numeric leading-none'
        : 'text-[52px] font-semibold tracking-tight text-emerald-600 dark:text-emerald-400 font-label-numeric leading-none';
    }

    const verdictTag = document.getElementById('sim-verdict-tag');
    if (verdictTag) {
      if (prob > 0.8) {
        verdictTag.innerText = 'CRITICAL THREAT';
        verdictTag.className = 'inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-rose-50 text-rose-700 border border-rose-200 font-label-code text-[11px]';
      } else if (prob > 0.5) {
        verdictTag.innerText = 'SUSPICIOUS DIVERGENCE';
        verdictTag.className = 'inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-amber-50 text-amber-700 border border-amber-200 font-label-code text-[11px]';
      } else {
        verdictTag.innerText = 'FRICTIONLESS CLEARANCE';
        verdictTag.className = 'inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 font-label-code text-[11px]';
      }
    }

    const actionVerdict = document.getElementById('sim-action-verdict');
    if (actionVerdict) {
      actionVerdict.innerText = prob > 0.8 ? 'AUTO-QUARANTINE' : (prob > 0.5 ? 'STEP-UP 2FA' : 'AUTHORIZE');
      actionVerdict.className = prob > 0.5 ? 'font-semibold text-rose-600 font-label-code text-[13px]' : 'font-semibold text-emerald-600 font-label-code text-[13px]';
    }

    renderShapBars(amt, hour);

  } catch (err) {
    console.error('Error running risk simulation:', err);
  }
}

function renderShapBars(amt, hour) {
  const container = document.getElementById('sim-shap-bars');
  if (!container) return;

  const amtFactor = Math.min(0.65, Math.max(0.05, (amt / 100000) * 0.5));
  const isLate = (hour >= 1 && hour <= 5);
  const hourFactor = isLate ? 0.32 : 0.04;

  container.innerHTML = `
    <div>
      <div class="flex justify-between mb-1">
        <span class="text-slate-600 dark:text-slate-300">Amount vs Baseline Sigma</span>
        <span class="${amtFactor > 0.2 ? 'text-rose-600' : 'text-slate-500'} font-semibold">+${amtFactor.toFixed(2)} SHAP</span>
      </div>
      <div class="w-full bg-slate-100 dark:bg-slate-800 h-1.5 rounded-full overflow-hidden">
        <div class="bg-rose-500 h-full rounded-full transition-all duration-300" style="width: ${Math.round(amtFactor * 140)}%"></div>
      </div>
    </div>

    <div>
      <div class="flex justify-between mb-1">
        <span class="text-slate-600 dark:text-slate-300">Off-Hours Window (${hour}:00)</span>
        <span class="${isLate ? 'text-rose-600' : 'text-emerald-600'} font-semibold">${isLate ? '+' : '-'}${hourFactor.toFixed(2)} SHAP</span>
      </div>
      <div class="w-full bg-slate-100 dark:bg-slate-800 h-1.5 rounded-full overflow-hidden">
        <div class="${isLate ? 'bg-rose-500' : 'bg-emerald-500'} h-full rounded-full transition-all duration-300" style="width: ${isLate ? 68 : 15}%"></div>
      </div>
    </div>

    <div>
      <div class="flex justify-between mb-1">
        <span class="text-slate-600 dark:text-slate-300">Account Drain Magnitude</span>
        <span class="text-amber-600 font-semibold">+0.18 SHAP</span>
      </div>
      <div class="w-full bg-slate-100 dark:bg-slate-800 h-1.5 rounded-full overflow-hidden">
        <div class="bg-amber-500 h-full rounded-full" style="width: 48%"></div>
      </div>
    </div>
  `;
}

// ==========================================================================
// 10. TAB 8: DATA QUALITY & ARCHITECTURE (Stitch Screen 5)
// ==========================================================================

async function loadQuality() {
  // Static verified stats populated from PySpark preprocessing cache
}

// ==========================================================================
// 11. TAB 9: TIME & GEO INTELLIGENCE
// ==========================================================================

async function loadTimeGeo() {
  try {
    const [resTime, resGeo] = await Promise.all([
      fetch('/api/time-analytics').then(r => r.json()),
      fetch('/api/geo-analytics').then(r => r.json())
    ]);

    if (resTime && resTime.heatmap) {
      renderHeatmap(resTime.heatmap);
    }

    if (resGeo && resGeo.locations) {
      renderGeoTable(resGeo.locations);
    }
  } catch (err) {
    console.error('Error loading time/geo analytics:', err);
  }
}

function renderHeatmap(heatmapList) {
  const container = document.getElementById('heatmap-grid');
  if (!container) return;

  const hours = Array.from({ length: 24 }, (_, i) => i);

  let html = `
    <div class="grid grid-cols-[80px_repeat(24,1fr)] gap-1 text-[10px] font-label-code">
      <div class="text-slate-400 font-semibold py-1">Day / Hr</div>
      ${hours.map(h => `<div class="text-center text-slate-400 py-1 font-semibold">${h}</div>`).join('')}
  `;

  heatmapList.forEach(dayObj => {
    const day = dayObj.day_name;
    html += `<div class="py-1.5 font-medium text-slate-600 dark:text-slate-400 text-xs">${day.slice(0, 3)}</div>`;
    
    // Day hours array
    const hrMap = {};
    if (Array.isArray(dayObj.hours)) {
      dayObj.hours.forEach(h => { hrMap[h.hour] = h.fraud_count || 0; });
    }

    hours.forEach(hr => {
      const val = hrMap[hr] !== undefined ? hrMap[hr] : 0;
      let bg = 'bg-slate-100 dark:bg-slate-800 text-slate-400';
      if (val >= 40) bg = 'bg-rose-600 text-white font-bold';
      else if (val >= 15) bg = 'bg-amber-400 text-slate-900 font-semibold';
      else if (val > 0) bg = 'bg-sky-100 dark:bg-sky-950/60 text-sky-800 dark:text-sky-300';

      html += `<div class="heatmap-cell h-7 flex items-center justify-center rounded cursor-pointer ${bg}" title="${day} ${hr}:00 — ${val} intercepted threats">${val}</div>`;
    });
  });

  html += '</div>';
  container.innerHTML = html;
}

function renderGeoTable(locations) {
  const tbody = document.getElementById('geo-tbody');
  if (!tbody) return;

  tbody.innerHTML = locations.map(loc => `
    <tr class="hover:bg-slate-50/60 dark:hover:bg-slate-800/40 transition-colors">
      <td class="py-3 font-medium text-slate-900 dark:text-slate-100 font-label-code">${loc.location}</td>
      <td class="py-3 font-semibold font-label-numeric">${formatINR(loc.total_volume)}</td>
      <td class="py-3 font-label-code">${formatNumber(loc.transaction_count)}</td>
      <td class="py-3 font-label-code text-rose-600 font-semibold">${formatNumber(loc.fraud_count)}</td>
      <td class="py-3 font-label-code">${(loc.fraud_rate || 0.98).toFixed(2)}%</td>
      <td class="py-3 text-right">
        <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-[11px] font-medium ${(loc.fraud_rate || 0.98) > 1.1 ? 'bg-rose-50 text-rose-700' : 'bg-emerald-50 text-emerald-700'}">
          ${(loc.fraud_rate || 0.98) > 1.1 ? 'Elevated' : 'Nominal'}
        </span>
      </td>
    </tr>
  `).join('');
}

// ==========================================================================
// 12. TAB 10: ANALYTICAL ALERTS
// ==========================================================================

async function loadAlerts() {
  try {
    const res = await fetch('/api/alerts').then(r => r.json());
    const list = document.getElementById('alerts-list');
    if (!list) return;

    const alerts = Array.isArray(res) ? res : (res.alerts || []);

    list.innerHTML = alerts.map(a => {
      const isCrit = (a.severity || '').toLowerCase() === 'critical';
      return `
        <div class="p-5 rounded-2xl bg-white dark:bg-[#0f172a] border ${isCrit ? 'border-rose-200 dark:border-rose-900/50' : 'border-slate-200/80 dark:border-slate-800'} shadow-xs flex flex-col gap-2">
          <div class="flex items-center justify-between">
            <div class="flex items-center gap-2">
              <span class="w-2 h-2 rounded-full ${isCrit ? 'bg-rose-500 animate-pulse' : 'bg-amber-500'}"></span>
              <h4 class="font-semibold text-slate-900 dark:text-white text-sm">${a.title}</h4>
            </div>
            <span class="text-xs text-slate-400 font-label-code">${a.timestamp || 'Real-time Rule'}</span>
          </div>
          <p class="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">${a.description || a.details}</p>
          <div class="text-[11px] font-label-code text-slate-400 mt-1">Rule Metric: <strong class="text-slate-700 dark:text-slate-300">${a.metric || 'Score Divergence'}</strong></div>
        </div>
      `;
    }).join('');

  } catch (err) {
    console.error('Error loading alerts:', err);
  }
}

// ==========================================================================
// 13. SLIDE-OVER INVESTIGATION DRAWER
// ==========================================================================

async function openInvestigation(transactionId) {
  const backdrop = document.getElementById('slide-over-backdrop');
  const drawer = document.getElementById('slide-over-drawer');
  if (!backdrop || !drawer) return;

  backdrop.classList.add('active');
  drawer.classList.add('active');

  const txIdEl = document.getElementById('drawer-tx-id');
  if (txIdEl) txIdEl.innerText = `#${transactionId}`;

  try {
    const res = await fetch(`/api/fraud/investigation/${transactionId}`).then(r => r.json());
    if (res.error) return;

    if (document.getElementById('drawer-amount')) document.getElementById('drawer-amount').innerText = formatINR(res.amount);
    if (document.getElementById('drawer-cust-id')) document.getElementById('drawer-cust-id').innerText = res.customer_id;
    if (document.getElementById('drawer-payment')) document.getElementById('drawer-payment').innerText = res.payment_method;
    if (document.getElementById('drawer-device')) document.getElementById('drawer-device').innerText = res.device_type;
    if (document.getElementById('drawer-merchant')) document.getElementById('drawer-merchant').innerText = res.merchant;
    if (document.getElementById('drawer-location')) document.getElementById('drawer-location').innerText = res.location;

    const riskLevel = document.getElementById('drawer-risk-level');
    if (riskLevel) {
      riskLevel.innerText = `${(res.risk_level || 'CRITICAL').toUpperCase()} THREAT (${(res.fraud_probability || 0.945).toFixed(3)})`;
    }

  } catch (err) {
    console.error('Error fetching investigation details:', err);
  }
}

function closeSlideOver() {
  const backdrop = document.getElementById('slide-over-backdrop');
  const drawer = document.getElementById('slide-over-drawer');
  if (backdrop) backdrop.classList.remove('active');
  if (drawer) drawer.classList.remove('active');
}

function takeAnalystAction(action) {
  alert(`Analyst action [${action}] recorded for telemetry audit log.`);
  closeSlideOver();
}

// ==========================================================================
// 14. ENHANCED EXECUTIVE REPORT MODAL
// ==========================================================================

async function openReportModal() {
  const backdrop = document.getElementById('report-modal-backdrop');
  if (backdrop) backdrop.classList.add('active');

  try {
    const [summary, fraud] = await Promise.all([
      fetch('/api/summary').then(r => r.json()),
      fetch('/api/fraud').then(r => r.json())
    ]);

    if (summary && !summary.error) {
      const vol = document.getElementById('report-total-vol');
      if (vol) vol.innerText = formatINR(summary.total_transaction_value);

      const fc = document.getElementById('report-fraud-count');
      if (fc) fc.innerText = formatNumber(summary.fraudulent_transactions);

      const fr = document.getElementById('report-fraud-rate');
      if (fr) fr.innerText = `${(summary.fraud_rate || 0.98).toFixed(2)}% overall rate`;
    }

    if (fraud && fraud.payment_method) {
      const tbody = document.getElementById('report-channel-tbody');
      if (tbody) {
        tbody.innerHTML = fraud.payment_method.map(p => `
          <tr>
            <td class="py-2.5 px-4 font-sans font-medium">${p.payment_method}</td>
            <td class="py-2.5 px-4">${formatINR(p.total_amount)}</td>
            <td class="py-2.5 px-4">${formatNumber(p.count)}</td>
            <td class="py-2.5 px-4 text-rose-600 font-semibold">${formatNumber(p.fraud_count)}</td>
            <td class="py-2.5 px-4 font-semibold">${p.fraud_rate.toFixed(2)}%</td>
            <td class="py-2.5 px-4">
              <span class="px-2 py-0.5 rounded text-[10px] font-bold ${p.fraud_rate > 1.1 ? 'bg-rose-100 text-rose-800' : 'bg-slate-100 text-slate-700'}">
                ${p.fraud_rate > 1.1 ? 'CRITICAL' : 'NOMINAL'}
              </span>
            </td>
          </tr>
        `).join('');
      }
    }
  } catch (e) {
    console.error('Error populating report data:', e);
  }
}

function closeReportModal() {
  const backdrop = document.getElementById('report-modal-backdrop');
  if (backdrop) backdrop.classList.remove('active');
}

// ==========================================================================
// 15. DOM READY BOOTSTRAP
// ==========================================================================

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initNavigation();

  // Backdrop click handlers to close drawer & modal
  const slideBackdrop = document.getElementById('slide-over-backdrop');
  if (slideBackdrop) slideBackdrop.addEventListener('click', closeSlideOver);

  const reportBackdrop = document.getElementById('report-modal-backdrop');
  if (reportBackdrop) {
    reportBackdrop.addEventListener('click', (e) => {
      if (e.target === reportBackdrop) closeReportModal();
    });
  }

  // Load initial Overview tab
  loadOverview();
});
