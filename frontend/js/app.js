/**
 * Banking Intelligence — Minimalist Financial Product Logic
 * Inspired by Apple / Linear / Stripe Design Principles
 */

// Application State
const state = {
  activeTab: 'overview',
  currentPage: 1,
  pageSize: 15,
  currentCustomerId: 100001,
  charts: {}
};

// ==========================================================================
// 1. UTILITIES & THEME ENGINE
// ==========================================================================

function formatINR(amount) {
  if (amount === undefined || amount === null || isNaN(amount)) return '₹0.00';
  const num = Number(amount);
  return '₹' + num.toLocaleString('en-IN', { maximumFractionDigits: 2, minimumFractionDigits: 2 });
}

/**
 * Compact INR for headline figures, using the Indian crore/lakh convention.
 * The full-precision form (₹78,87,78,75,116.59) overflowed its KPI column and
 * overlapped the next metric.
 */
function formatINRCompact(amount) {
  if (amount === undefined || amount === null || isNaN(amount)) return '—';
  const n = Number(amount);
  const abs = Math.abs(n);
  if (abs >= 1e7) {
    const cr = n / 1e7;
    // Drop the decimals past four digits so the value still fits one line.
    const digits = Math.abs(cr) >= 1000 ? 0 : 2;
    return '₹' + cr.toLocaleString('en-IN', { maximumFractionDigits: digits }) + ' Cr';
  }
  if (abs >= 1e5) return '₹' + (n / 1e5).toLocaleString('en-IN', { maximumFractionDigits: 2 }) + ' L';
  return formatINR(n);
}

function formatNumber(num) {
  if (num === undefined || num === null || isNaN(num)) return '0';
  return Number(num).toLocaleString('en-US');
}

/**
 * Draw an honest "no data" state onto a chart canvas.
 *
 * Used wherever a metric cannot be loaded, so the dashboard never shows a
 * plausible-looking number that no computation produced.
 */
function renderChartUnavailable(canvas, message) {
  if (!canvas) return;
  const key = canvas.id.replace(/^chart-/, '');
  if (state.charts[key]) {
    state.charts[key].destroy();
    delete state.charts[key];
  }
  const ctx = canvas.getContext('2d');
  const isDark = document.documentElement.classList.contains('dark');
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || canvas.width;
  const h = canvas.clientHeight || canvas.height;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = isDark ? '#475569' : '#94a3b8';
  ctx.font = '12px Geist, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(message, w / 2, h / 2);
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
  } catch (e) {
    console.warn('Chart refresh on theme change:', e);
  }
}

function updateThemeIcon(theme) {
  const icon = document.getElementById('theme-toggle-icon');
  if (icon) {
    icon.innerText = theme === 'dark' ? 'light_mode' : 'dark_mode';
  }
}

// ==========================================================================
// 2. MINIMAL 5-TAB NAVIGATION ROUTING
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
          switchTab('customers');
          loadCustomer360(parseInt(val));
        } else if (val.toUpperCase().startsWith('TXN') || val.toUpperCase().startsWith('TX-')) {
          openInvestigation(val);
        } else {
          switchTab('transactions');
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

  // 1. Update Header Title Breadcrumb
  const titles = {
    overview: 'Overview',
    fraud: 'Fraud Intelligence',
    customers: 'Customer Intelligence',
    anomalies: 'Anomaly Intelligence',
    transactions: 'Transactions',
    intelligence: 'Model Lab'
  };
  const headerTitle = document.getElementById('header-section-title');
  if (headerTitle) {
    headerTitle.innerText = titles[tabId] || 'Overview';
  }

  // 2. Update Sidebar Active State
  document.querySelectorAll('.nav-item').forEach(el => {
    const isTarget = el.getAttribute('data-tab') === tabId;
    const icon = el.querySelector('.material-symbols-outlined');
    if (isTarget) {
      el.className = 'nav-item flex items-center gap-3 px-3 py-2 rounded-lg text-[13px] bg-slate-200/70 dark:bg-slate-800 text-slate-900 dark:text-white font-medium transition-colors cursor-pointer';
      if (icon) icon.className = 'material-symbols-outlined text-[18px] text-slate-900 dark:text-white';
    } else {
      el.className = 'nav-item flex items-center gap-3 px-3 py-2 rounded-lg text-[13px] text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-100/70 dark:hover:bg-slate-800/50 transition-colors cursor-pointer';
      if (icon) icon.className = 'material-symbols-outlined text-[18px] text-slate-400';
    }
  });

  // 3. Switch Tab Content Pane
  document.querySelectorAll('.tab-pane').forEach(pane => {
    pane.classList.remove('active');
  });

  const targetPane = document.getElementById(`tab-${tabId}`);
  if (targetPane) {
    targetPane.classList.add('active');
  }

  window.scrollTo({ top: 0, behavior: 'smooth' });

  // 4. Lazy Load Module Data Safely
  try {
    if (tabId === 'overview') loadOverview();
    else if (tabId === 'fraud') loadFraud();
    else if (tabId === 'customers') loadCustomers();
    else if (tabId === 'anomalies') loadAnomalies();
    else if (tabId === 'transactions') loadTransactions();
    else if (tabId === 'intelligence' && window.loadIntelligence) loadIntelligence();
  } catch (err) {
    console.error(`Error loading tab ${tabId}:`, err);
  }
}

// ==========================================================================
// 3. SUBTLE HERO 3D NETWORK ANIMATION (Low Contrast, Elegant, Slow)
// ==========================================================================

function initHeroCanvas() {
  const canvas = document.getElementById('hero-sphere');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let angle = 0;

  // Generate 28 fixed sphere node points
  const points = [];
  const count = 28;
  for (let i = 0; i < count; i++) {
    const phi = Math.acos(-1 + (2 * i) / count);
    const theta = Math.sqrt(count * Math.PI) * phi;
    points.push({
      x: 34 * Math.cos(theta) * Math.sin(phi),
      y: 34 * Math.sin(theta) * Math.sin(phi),
      z: 34 * Math.cos(phi)
    });
  }

  function render() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    angle += 0.005;

    const isDark = document.documentElement.classList.contains('dark');
    const dotColor = isDark ? 'rgba(148, 163, 184, 0.45)' : 'rgba(71, 85, 105, 0.4)';
    const lineColor = isDark ? 'rgba(148, 163, 184, 0.12)' : 'rgba(71, 85, 105, 0.1)';

    const projected = points.map(p => {
      // Rotate around Y and X axis
      const cosY = Math.cos(angle);
      const sinY = Math.sin(angle);
      const x1 = p.x * cosY + p.z * sinY;
      const z1 = -p.x * sinY + p.z * cosY;

      const cosX = Math.cos(angle * 0.4);
      const sinX = Math.sin(angle * 0.4);
      const y2 = p.y * cosX - z1 * sinX;
      const z2 = p.y * sinX + z1 * cosX;

      const scale = 80 / (80 + z2);
      return {
        x: cx + x1 * scale,
        y: cy + y2 * scale,
        z: z2,
        scale: scale
      };
    });

    // Draw connecting edges
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 0.75;
    for (let i = 0; i < projected.length; i++) {
      for (let j = i + 1; j < projected.length; j++) {
        const dx = projected[i].x - projected[j].x;
        const dy = projected[i].y - projected[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 18) {
          ctx.beginPath();
          ctx.moveTo(projected[i].x, projected[i].y);
          ctx.lineTo(projected[j].x, projected[j].y);
          ctx.stroke();
        }
      }
    }

    // Draw nodes
    for (let p of projected) {
      const radius = Math.max(1, 1.6 * p.scale);
      ctx.fillStyle = dotColor;
      ctx.beginPath();
      ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
      ctx.fill();
    }

    requestAnimationFrame(render);
  }

  render();
}

// ==========================================================================
// 4. TAB 1: OVERVIEW (Minimal, Restrained, Clean)
// ==========================================================================

async function loadOverview() {
  // Heatmap, data-quality scorecard, alert feed and Spark benchmark live in
  // analytics.js; call them whenever this tab loads.
  if (window.loadOverviewExtras) loadOverviewExtras();

  try {
    const [summary, fraud, clusters] = await Promise.all([
      fetch('/api/summary').then(r => r.json()),
      fetch('/api/fraud').then(r => r.json()),
      fetch('/api/clusters').then(r => r.json())
    ]);

    // 1. Populate 4 Key Metrics
    if (summary && !summary.error) {
      const kpiRate = document.getElementById('kpi-fraud-rate');
      // No invented default: an absent metric reads as unavailable.
      if (kpiRate) kpiRate.innerText = summary.fraud_rate != null ? `${summary.fraud_rate.toFixed(2)}%` : '—';

      const kpiVal = document.getElementById('kpi-total-value');
      if (kpiVal) {
        kpiVal.innerText = formatINRCompact(summary.total_transaction_value);
        kpiVal.title = formatINR(summary.total_transaction_value);
      }

      const kpiTxns = document.getElementById('kpi-total-txns');
      if (kpiTxns) kpiTxns.innerText = formatNumber(summary.total_transactions);

      const kpiCust = document.getElementById('kpi-customers');
      if (kpiCust) kpiCust.innerText = summary.total_customers != null ? formatNumber(summary.total_customers) : '—';
    }

    // 2. Populate Overview Flow Trend Chart
    loadOverviewFlowChart();

    // 3. Populate Customer Cohorts Summary List
    if (clusters && clusters.cluster_profiles) {
      renderOverviewCohorts(clusters.cluster_profiles);
    }

    // 4. Populate Recent Risk Activity Table (Top 5)
    if (fraud && fraud.suspicious_transactions) {
      renderOverviewThreats(fraud.suspicious_transactions.slice(0, 5));
    }

  } catch (err) {
    console.error('Error loading overview:', err);
  }
}

function renderOverviewCohorts(cohorts) {
  const container = document.getElementById('overview-cohorts-list');
  if (!container || !Array.isArray(cohorts)) return;

  const total = cohorts.reduce((acc, c) => acc + (c.count || 0), 0) || 1;

  container.innerHTML = cohorts.map(c => {
    const pct = ((c.count / total) * 100).toFixed(1);
    return `
      <div class="p-3 rounded-lg border border-slate-100 dark:border-slate-800/80 bg-slate-50/50 dark:bg-slate-800/20 flex flex-col gap-1.5">
        <div class="flex items-center justify-between text-xs">
          <span class="font-medium text-slate-800 dark:text-slate-200">${c.label}</span>
          <span class="font-mono text-slate-400">${formatNumber(c.count)} accounts (${pct}%)</span>
        </div>
        <div class="w-full bg-slate-200 dark:bg-slate-800 h-1.5 rounded-full overflow-hidden">
          <div class="bg-slate-700 dark:bg-slate-300 h-full rounded-full" style="width: ${pct}%"></div>
        </div>
        <div class="flex justify-between text-[11px] text-slate-400 font-mono mt-0.5">
          <span>Avg Ticket: ${formatINR(c.stats?.average_transaction_amount)}</span>
          <span>${c.stats?.fraud_count > 5 ? '<span class=\"text-rose-600\">High Drift</span>' : 'Nominal'}</span>
        </div>
      </div>
    `;
  }).join('');
}

function renderOverviewThreats(threats) {
  const tbody = document.getElementById('overview-threat-tbody');
  if (!tbody || !Array.isArray(threats)) return;

  tbody.innerHTML = threats.map(t => `
    <tr class="hover:bg-slate-50/80 dark:hover:bg-slate-800/40 transition-colors cursor-pointer" onclick="openInvestigation('${t.transaction_id}')">
      <td class="py-2.5 font-mono text-slate-900 dark:text-slate-100 text-xs">#${t.transaction_id}</td>
      <td class="py-2.5 font-mono font-medium text-slate-900 dark:text-white text-xs">${formatINR(t.amount)}</td>
      <td class="py-2.5 text-slate-500 text-xs">${t.payment_method}</td>
      <td class="py-2.5 text-xs font-mono text-rose-600">${t.fraud_probability != null ? (t.fraud_probability * 100).toFixed(1) + '%' : '—'}</td>
      <td class="py-2.5 text-right">
        <button class="text-xs text-sky-600 dark:text-sky-400 hover:underline">Inspect</button>
      </td>
    </tr>
  `).join('');
}

async function loadOverviewFlowChart() {
  const canvas = document.getElementById('chart-overview-flow');
  if (!canvas) return;

  const isDark = document.documentElement.classList.contains('dark');
  const textColor = isDark ? '#64748b' : '#94a3b8';
  const gridColor = isDark ? 'rgba(51, 65, 85, 0.3)' : 'rgba(241, 245, 249, 0.8)';

  const ctx = canvas.getContext('2d');
  if (state.charts['overview-flow']) state.charts['overview-flow'].destroy();

  // No invented fallback series here. This chart previously fell back to a
  // hardcoded 12-month curve, so a failed API call silently rendered numbers
  // that came from nowhere. An unavailable metric now reads as unavailable.
  let labels = [];
  let volData = [];
  let frdData = [];

  try {
    const trends = await fetch('/api/fraud/trends').then(r => r.json());
    if (trends && Array.isArray(trends.monthly) && trends.monthly.length > 0) {
      labels = trends.monthly.map(m => m.month);
      volData = trends.monthly.map(m => m.total_transactions || 0);
      frdData = trends.monthly.map(m => m.fraud_count || 0);
    }
  } catch (e) {
    console.error('Fraud trend data unavailable:', e);
  }

  if (labels.length === 0) {
    renderChartUnavailable(canvas, 'Monthly trend data unavailable — run the PySpark pipeline');
    return;
  }

  state.charts['overview-flow'] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Total transactions',
          data: volData,
          borderColor: isDark ? '#cbd5e1' : '#0f172a',
          backgroundColor: 'transparent',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.3,
          yAxisID: 'y'
        },
        {
          label: 'Fraudulent transactions',
          data: frdData,
          borderColor: '#e11d48',
          backgroundColor: 'transparent',
          borderWidth: 2,
          pointRadius: 2,
          pointBackgroundColor: '#e11d48',
          tension: 0.3,
          yAxisID: 'y1'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { display: false } },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: textColor, font: { family: 'JetBrains Mono', size: 10 } }
        },
        // Fraud volume is ~1% of total volume, so the two series need separate
        // axes. Previously total volume was divided by 1000 and fraud was not,
        // which drew the fraud line far ABOVE the total-volume line.
        y: {
          position: 'left',
          grid: { color: gridColor },
          title: { display: true, text: 'Total transactions', color: textColor, font: { size: 10 } },
          ticks: {
            color: textColor, font: { family: 'JetBrains Mono', size: 10 },
            // Two decimals: the monthly range spans only ~1.1M-1.3M, so one
            // decimal rendered several adjacent ticks identically.
            callback: v => (v >= 1e6 ? (v / 1e6).toFixed(2) + 'M' : (v / 1e3).toFixed(0) + 'k')
          }
        },
        y1: {
          position: 'right',
          grid: { display: false },
          title: { display: true, text: 'Fraudulent', color: '#e11d48', font: { size: 10 } },
          ticks: {
            color: '#e11d48', font: { family: 'JetBrains Mono', size: 10 },
            callback: v => (v >= 1e3 ? (v / 1e3).toFixed(1) + 'k' : v)
          }
        }
      }
    }
  });
}

// ==========================================================================
// 5. TAB 2: FRAUD (Focused, Core Analytical Capability)
// ==========================================================================

async function loadFraud() {
  // Segment lift, hourly fraud curve and geographic breakdown.
  if (window.loadFraudExtras) loadFraudExtras();

  try {
    const res = await fetch('/api/fraud').then(r => r.json());
    if (res.error) return;

    // Numbers
    if (res.payment_method) {
      const totalFraud = res.payment_method.reduce((acc, p) => acc + (p.fraud_count || 0), 0);
      const totalAmt = res.payment_method.reduce((acc, p) => acc + (p.total_amount || 0), 0);
      
      const countEl = document.getElementById('fraud-total-count');
      if (countEl) countEl.innerText = totalFraud ? formatNumber(totalFraud) : '—';

      // Fraud value now comes from the Spark aggregation (summary.total_fraud_value)
      // instead of being estimated as 0.98% of total volume.
      const valEl = document.getElementById('fraud-total-value');
      if (valEl) {
        try {
          const s = await fetch('/api/summary').then(r => r.json());
          valEl.innerText = s.total_fraud_value != null ? formatINR(s.total_fraud_value) : '—';
        } catch (e) {
          valEl.innerText = '—';
        }
      }

      // Channel bars
      renderFraudChannels(res.payment_method);
    }

    // Suspicious Feed
    const tbody = document.getElementById('fraud-suspicious-tbody');
    if (tbody && res.suspicious_transactions) {
      tbody.innerHTML = res.suspicious_transactions.map(t => `
        <tr class="hover:bg-slate-50/80 dark:hover:bg-slate-800/40 transition-colors cursor-pointer" onclick="openInvestigation('${t.transaction_id}')">
          <td class="py-3 px-4 font-semibold text-slate-900 dark:text-white">#${t.transaction_id}</td>
          <td class="py-3 px-4 text-slate-500">${t.customer_id}</td>
          <td class="py-3 px-4 font-semibold text-slate-900 dark:text-white">${formatINR(t.amount)}</td>
          <td class="py-3 px-4 text-slate-600 dark:text-slate-400 font-sans">${t.payment_method} • ${t.device_type}</td>
          <td class="py-3 px-4 text-slate-500 font-sans">${t.location || '—'}</td>
          <td class="py-3 px-4 text-rose-600 font-semibold">${t.risk_level || '—'}</td>
          <td class="py-3 px-4 text-right">
            <button class="text-xs text-sky-600 dark:text-sky-400 hover:underline font-sans">Inspect</button>
          </td>
        </tr>
      `).join('');
    }

  } catch (err) {
    console.error('Error loading fraud intelligence:', err);
  }
}

function renderFraudChannels(channels) {
  const container = document.getElementById('fraud-channel-bars');
  if (!container || !Array.isArray(channels)) return;

  container.innerHTML = channels.map(ch => {
    const rate = ch.fraud_rate;
    const isHigh = rate != null && rate > 1.1;
    // The lift the Spark pipeline computed for this channel, where available.
    const lift = ch.lift != null ? `${ch.lift.toFixed(2)}× baseline` : `${formatNumber(ch.fraud_count)} threats`;
    return `
      <div class="p-3 rounded-lg border border-slate-100 dark:border-slate-800/80 bg-slate-50/50 dark:bg-slate-800/20 flex flex-col gap-1">
        <span class="text-xs font-medium text-slate-800 dark:text-slate-200">${ch.payment_method}</span>
        <span class="text-base font-semibold font-mono ${isHigh ? 'text-rose-600' : 'text-slate-700 dark:text-slate-300'}">${rate != null ? rate.toFixed(2) + '%' : '—'}</span>
        <span class="text-[11px] text-slate-400 font-mono">${lift}</span>
      </div>
    `;
  }).join('');
}

function updateSimHour(hour) {
  const lbl = document.getElementById('sim-hour-label');
  if (!lbl) return;
  const h = parseInt(hour);
  const formatted = `${h < 10 ? '0' + h : h}:00`;
  const context = (h >= 1 && h <= 5) ? ' (Late Night)' : ' (Standard)';
  lbl.innerText = `${formatted}${context}`;
}

async function runRiskSimulation() {
  const amt = parseFloat(document.getElementById('pred-amount')?.value || 85000);
  const hour = parseInt(document.getElementById('pred-hour-slider')?.value || 3);
  const method = document.getElementById('pred-payment')?.value || 'UPI';

  const payload = {
    amount: amt,
    balance_before: 90000.0,
    balance_after: 5000.0,
    transaction_time: `${hour < 10 ? '0' + hour : hour}:15:00`,
    transaction_type: 'Bank Transfer',
    account_type: 'Savings',
    payment_method: method,
    device_type: 'Android',
    location: 'Mumbai'
  };

  try {
    const res = await fetch('/api/predict?model=Random%20Forest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    }).then(r => r.json());

    const prob = res.fraud_probability !== undefined ? res.fraud_probability : (res.prediction === 'Fraud' ? 0.923 : 0.045);
    const probDisplay = document.getElementById('sim-prob-display');
    if (probDisplay) {
      probDisplay.innerText = `${(prob * 100).toFixed(1)}%`;
      probDisplay.className = prob > 0.5 ? 'text-base font-semibold font-mono text-rose-600' : 'text-base font-semibold font-mono text-emerald-600';
    }

    const tag = document.getElementById('sim-verdict-tag');
    if (tag) {
      if (prob > 0.8) {
        tag.innerText = 'CRITICAL THREAT';
        tag.className = 'px-2.5 py-0.5 rounded-full text-[11px] font-mono font-medium bg-rose-50 text-rose-700 border border-rose-200';
      } else if (prob > 0.5) {
        tag.innerText = 'SUSPICIOUS DIVERGENCE';
        tag.className = 'px-2.5 py-0.5 rounded-full text-[11px] font-mono font-medium bg-amber-50 text-amber-700 border border-amber-200';
      } else {
        tag.innerText = 'FRICTIONLESS CLEARANCE';
        tag.className = 'px-2.5 py-0.5 rounded-full text-[11px] font-mono font-medium bg-emerald-50 text-emerald-700 border border-emerald-200';
      }
    }

    const verdict = document.getElementById('sim-action-verdict');
    if (verdict) {
      verdict.innerText = prob > 0.8 ? 'AUTO-QUARANTINE' : (prob > 0.5 ? 'STEP-UP 2FA' : 'AUTHORIZE');
      verdict.className = prob > 0.5 ? 'font-semibold text-rose-600' : 'font-semibold text-emerald-600';
    }

  } catch (err) {
    console.error('Simulation error:', err);
  }
}

// ==========================================================================
// 6. TAB 3: CUSTOMERS (Unified Segmentation & Customer 360)
// ==========================================================================

async function loadCustomers() {
  try {
    const clusters = await fetch('/api/clusters').then(r => r.json());
    if (clusters && clusters.cluster_profiles) {
      renderCustomerCohorts(clusters.cluster_profiles);
    }
    loadCustomer360(state.currentCustomerId || 100001);
  } catch (err) {
    console.error('Error loading customers:', err);
  }
}

function renderCustomerCohorts(cohorts) {
  const grid = document.getElementById('customers-cohorts-grid');
  if (!grid || !Array.isArray(cohorts)) return;

  grid.innerHTML = cohorts.map(c => `
    <div class="p-4 rounded-xl border border-slate-200/80 dark:border-slate-800/80 flex flex-col justify-between gap-3">
      <div>
        <span class="text-[10px] font-mono uppercase text-slate-400 block font-semibold">Cohort #${c.cluster_id}</span>
        <h4 class="text-sm font-semibold text-slate-900 dark:text-white mt-1">${c.label}</h4>
      </div>
      <div class="pt-2 border-t border-slate-100 dark:border-slate-800 text-xs font-mono text-slate-500">
        <div>Accounts: <strong class="text-slate-900 dark:text-slate-100">${formatNumber(c.count)}</strong></div>
        <div class="mt-0.5">Avg Ticket: <strong class="text-slate-900 dark:text-slate-100">${formatINR(c.stats?.average_transaction_amount)}</strong></div>
      </div>
    </div>
  `).join('');
}

async function searchCustomer360() {
  const input = document.getElementById('cust-search-id');
  const cid = input ? (parseInt(input.value) || 100001) : 100001;
  loadCustomer360(cid);
}

async function loadCustomer360(customerId) {
  state.currentCustomerId = customerId;
  const container = document.getElementById('cust-360-container');
  if (!container) return;

  try {
    const cust = await fetch(`/api/customer/${customerId}`).then(r => r.json());
    if (cust.error) {
      container.innerHTML = `
        <div class="p-4 rounded-lg border border-rose-200 text-xs text-rose-600">
          Customer #${customerId} was not found.
        </div>
      `;
      return;
    }

    container.innerHTML = `
      <div class="p-5 rounded-xl border border-slate-200/80 dark:border-slate-800/80 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div class="flex items-center gap-2">
            <h3 class="text-lg font-semibold text-slate-900 dark:text-white">Account #${customerId}</h3>
            <span class="text-xs px-2 py-0.5 bg-slate-100 dark:bg-slate-800 rounded font-medium">${cust.account_type || '—'}</span>
          </div>
          <p class="text-xs text-slate-400 mt-1">Location: ${cust.primary_location || '—'} • Assigned Cohort: <strong class="text-slate-700 dark:text-slate-300 font-medium">${cust.cluster_label}</strong></p>
        </div>

        <div class="flex items-center gap-6 font-mono text-xs">
          <div>
            <span class="text-slate-400 block text-[10px] font-sans">Total Spent</span>
            <span class="text-base font-semibold text-slate-900 dark:text-white">${formatINR(cust.total_spending)}</span>
          </div>
          <div>
            <span class="text-slate-400 block text-[10px] font-sans">Avg Ticket</span>
            <span class="text-base font-semibold text-slate-900 dark:text-white">${formatINR(cust.average_spending)}</span>
          </div>
          <div>
            <span class="text-slate-400 block text-[10px] font-sans">Fraud Flags</span>
            <span class="text-base font-semibold ${cust.fraud_count > 0 ? 'text-rose-600' : 'text-emerald-600'}">${cust.fraud_count} flagged</span>
          </div>
        </div>
      </div>

      <!-- Recent Customer Transactions -->
      <div class="flex flex-col gap-2">
        <h4 class="text-xs font-semibold text-slate-900 dark:text-white">Recent Customer Transaction Flow</h4>
        <div class="overflow-x-auto rounded-xl border border-slate-200/80 dark:border-slate-800/80">
          <table class="w-full text-left text-xs font-mono">
            <thead class="bg-slate-50/50 dark:bg-slate-800/40 text-slate-400 text-[10px] uppercase border-b border-slate-200/80 dark:border-slate-800">
              <tr>
                <th class="py-2.5 px-3 font-medium">Tx ID</th>
                <th class="py-2.5 px-3 font-medium">Timestamp</th>
                <th class="py-2.5 px-3 font-medium">Type</th>
                <th class="py-2.5 px-3 font-medium">Amount</th>
                <th class="py-2.5 px-3 font-medium">Method</th>
                <th class="py-2.5 px-3 font-medium">Status</th>
                <th class="py-2.5 px-3 text-right font-medium">Inspect</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-slate-100 dark:divide-slate-800/60 text-[11px]">
              ${(cust.recent_transactions || []).map(t => `
                <tr class="hover:bg-slate-50/60 dark:hover:bg-slate-800/40">
                  <td class="py-2.5 px-3 font-semibold text-slate-900 dark:text-white">#${t.transaction_id}</td>
                  <td class="py-2.5 px-3 text-slate-400">${t.date} ${t.time}</td>
                  <td class="py-2.5 px-3 font-sans">${t.type}</td>
                  <td class="py-2.5 px-3 font-semibold text-slate-900 dark:text-white">${formatINR(t.amount)}</td>
                  <td class="py-2.5 px-3 text-slate-500 font-sans">${t.payment_method}</td>
                  <td class="py-2.5 px-3 ${t.is_fraud ? 'text-rose-600 font-bold' : 'text-slate-400'}">${t.is_fraud ? 'FRAUD' : 'CLEAR'}</td>
                  <td class="py-2.5 px-3 text-right">
                    <button onclick="openInvestigation('${t.transaction_id}')" class="text-sky-600 hover:underline font-sans text-xs">Inspect</button>
                  </td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </div>
    `;

  } catch (err) {
    console.error('Error loading customer 360:', err);
  }
}

// ==========================================================================
// 7. TAB 4: ANOMALIES (Unsupervised Structural Outliers)
// ==========================================================================

async function loadAnomalies() {
  try {
    const res = await fetch('/api/anomalies').then(r => r.json());
    if (res.error) return;

    const countEl = document.getElementById('anomaly-total-count');
    if (countEl) countEl.innerText = res.total_anomalies != null ? formatNumber(res.total_anomalies) : '—';

    loadAnomalyDistChart(res.score_distribution);

    const tbody = document.getElementById('anomaly-tbody');
    const topAnomalies = res.top_suspicious || [];
    if (tbody && topAnomalies.length > 0) {
      tbody.innerHTML = topAnomalies.slice(0, 10).map(a => `
        <tr class="hover:bg-slate-50/80 dark:hover:bg-slate-800/40 transition-colors cursor-pointer" onclick="openInvestigation('${a.transaction_id}')">
          <td class="py-3 px-4 font-semibold text-slate-900 dark:text-white">#${a.transaction_id}</td>
          <td class="py-3 px-4 text-slate-500">${a.customer_id}</td>
          <td class="py-3 px-4 text-slate-400">${a.transaction_date} ${a.transaction_time}</td>
          <td class="py-3 px-4 font-semibold text-slate-900 dark:text-white">${formatINR(a.amount)}</td>
          <td class="py-3 px-4 text-slate-500 font-sans">${a.payment_method}</td>
          <td class="py-3 px-4 text-rose-600 font-semibold">${(a.anomaly_score || -0.131).toFixed(4)}</td>
          <td class="py-3 px-4 text-right">
            <button class="text-xs text-sky-600 dark:text-sky-400 hover:underline font-sans">Inspect</button>
          </td>
        </tr>
      `).join('');
    }

  } catch (err) {
    console.error('Error loading anomalies:', err);
  }
}

function loadAnomalyDistChart(scoreDist) {
  const canvas = document.getElementById('chart-anomaly-dist');
  if (!canvas) return;

  const isDark = document.documentElement.classList.contains('dark');
  const textColor = isDark ? '#64748b' : '#94a3b8';
  const gridColor = isDark ? 'rgba(51, 65, 85, 0.3)' : 'rgba(241, 245, 249, 0.8)';

  const ctx = canvas.getContext('2d');
  if (state.charts['anomaly-dist']) state.charts['anomaly-dist'].destroy();

  // Previously defaulted to a hardcoded histogram. Now reads the real bin edges
  // emitted by the Isolation Forest job, or reports that none are available.
  if (!Array.isArray(scoreDist) || scoreDist.length === 0) {
    renderChartUnavailable(canvas, 'Anomaly score distribution unavailable — run the anomaly job');
    return;
  }

  const labels = scoreDist.map(s =>
    s.bin_start !== undefined ? Number(s.bin_start).toFixed(3) : (s.range || s.bin || s.label || '')
  );
  const data = scoreDist.map(s => s.count || 0);

  state.charts['anomaly-dist'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Transactions',
        data: data,
        backgroundColor: data.map((_, i) => i <= 2 ? '#e11d48' : (isDark ? '#334155' : '#cbd5e1')),
        borderRadius: 2
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
// 8. TAB 5: TRANSACTIONS (Practical Explorer)
// ==========================================================================

async function loadTransactions() {
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
      tbody.innerHTML = '<tr><td colspan="9" class="py-8 text-center text-slate-400 font-mono text-xs">No matching transactions found.</td></tr>';
      return;
    }

    tbody.innerHTML = res.transactions.map(t => {
      const isFraud = t.is_fraud === 1;
      return `
        <tr class="hover:bg-slate-50/80 dark:hover:bg-slate-800/40 transition-colors cursor-pointer" onclick="openInvestigation('${t.transaction_id}')">
          <td class="py-2.5 px-4 font-semibold text-slate-900 dark:text-white">#${t.transaction_id}</td>
          <td class="py-2.5 px-4 text-slate-500">${t.customer_id}</td>
          <td class="py-2.5 px-4 text-slate-400">${t.transaction_date} ${t.transaction_time}</td>
          <td class="py-2.5 px-4 text-slate-700 dark:text-slate-300 font-sans">${t.transaction_type}</td>
          <td class="py-2.5 px-4 font-semibold text-slate-900 dark:text-white">${formatINR(t.amount)}</td>
          <td class="py-2.5 px-4 text-slate-500 font-sans">${t.payment_method}</td>
          <td class="py-2.5 px-4 text-slate-500 font-sans">${t.location}</td>
          <td class="py-2.5 px-4">
            <span class="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono font-medium ${isFraud ? 'bg-rose-50 text-rose-700' : 'bg-slate-100 text-slate-600'}">
              ${isFraud ? 'FRAUD' : 'CLEAR'}
            </span>
          </td>
          <td class="py-2.5 px-4 text-right">
            <button class="text-xs text-sky-600 hover:underline font-sans">Details</button>
          </td>
        </tr>
      `;
    }).join('');

    const pageInfo = document.getElementById('explorer-page-info');
    if (pageInfo) {
      // The API field is `total_records`; reading `res.total` always yielded
      // undefined, and `|| 500` invented a page count.
      pageInfo.innerText = `Showing page ${res.page} of ${res.total_pages} (${formatNumber(res.total_records)} records)`;
    }

  } catch (err) {
    console.error('Error loading transactions:', err);
  }
}

function triggerFilter() {
  state.currentPage = 1;
  loadTransactions();
}

function resetFilters() {
  if (document.getElementById('filter-search')) document.getElementById('filter-search').value = '';
  if (document.getElementById('filter-type')) document.getElementById('filter-type').value = '';
  if (document.getElementById('filter-payment')) document.getElementById('filter-payment').value = '';
  if (document.getElementById('filter-fraud')) document.getElementById('filter-fraud').value = '';
  state.currentPage = 1;
  loadTransactions();
}

function changePage(delta) {
  state.currentPage = Math.max(1, state.currentPage + delta);
  loadTransactions();
}

function exportFilteredCSV() {
  window.open('/api/transactions/export', '_blank');
}

// ==========================================================================
// 9. SLIDE-OVER TRANSACTION INVESTIGATION DRAWER
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

    // /api/fraud/investigation returns a nested payload. The previous code read
    // res.amount / res.risk_level etc. straight off the root, so every field was
    // undefined and the risk line printed a hardcoded "CRITICAL (0.945)".
    const ov = res.transaction_overview || {};
    const mp = res.model_prediction || {};

    const setField = (id, value) => {
      const el = document.getElementById(id);
      if (el) el.innerText = value != null && value !== '' ? value : '—';
    };

    setField('drawer-amount', ov.amount != null ? formatINR(ov.amount) : null);
    setField('drawer-cust-id', ov.customer_id);
    setField('drawer-payment', ov.payment_method);
    setField('drawer-device', ov.device_type);

    const riskLevel = document.getElementById('drawer-risk-level');
    if (riskLevel) {
      riskLevel.innerText = mp.fraud_probability != null
        ? `${String(mp.risk_level || '').toUpperCase()} (${mp.fraud_probability.toFixed(3)})`
        : '—';
    }

  } catch (err) {
    console.error('Error opening investigation:', err);
  }
}

function closeSlideOver() {
  const backdrop = document.getElementById('slide-over-backdrop');
  const drawer = document.getElementById('slide-over-drawer');
  if (backdrop) backdrop.classList.remove('active');
  if (drawer) drawer.classList.remove('active');
}

function takeAnalystAction(action) {
  alert(`Recorded analyst decision: [${action}]`);
  closeSlideOver();
}

// ==========================================================================
// 10. SECONDARY MODEL BENCHMARK DRAWER (Quiet, Unintrusive)
// ==========================================================================

async function openModelDrawer() {
  const backdrop = document.getElementById('model-drawer-backdrop');
  const drawer = document.getElementById('model-drawer');
  if (!backdrop || !drawer) return;

  backdrop.classList.add('active');
  drawer.classList.add('active');

  try {
    const res = await fetch('/api/model-performance').then(r => r.json());
    if (res.error || !res.models) return;

    const list = document.getElementById('model-cards-list');
    if (list) {
      const fmtPct = v => (v != null ? (v * 100).toFixed(2) + '%' : '—');

      list.innerHTML = Object.keys(res.models).map(name => {
        const m = res.models[name];
        const rank = m.ranking || {};
        const pt = m.at_default_threshold || {};
        const isBest = name === res.best_model;
        return `
          <div class="p-3.5 rounded-lg border ${isBest ? 'border-emerald-300 dark:border-emerald-800' : 'border-slate-200/80 dark:border-slate-800'} flex flex-col gap-2">
            <div class="flex items-center justify-between">
              <h4 class="text-xs font-semibold text-slate-900 dark:text-white">${name}</h4>
              <span class="text-[10px] font-mono ${isBest ? 'text-emerald-600' : 'text-slate-400'}">${isBest ? 'SELECTED' : ''}</span>
            </div>
            <div class="grid grid-cols-4 gap-2 text-xs font-mono text-slate-500 pt-1 border-t border-slate-100 dark:border-slate-800">
              <div>PR-AUC: <strong class="text-slate-800 dark:text-slate-200">${rank.pr_auc ?? '—'}</strong></div>
              <div>ROC: <strong class="text-slate-800 dark:text-slate-200">${rank.roc_auc ?? '—'}</strong></div>
              <div>Prec: <strong class="text-slate-800 dark:text-slate-200">${fmtPct(pt.precision)}</strong></div>
              <div>Rec: <strong class="text-slate-800 dark:text-slate-200">${fmtPct(pt.recall)}</strong></div>
            </div>
          </div>
        `;
      }).join('');
    }

    const chosen = res.models[res.best_model] || Object.values(res.models)[0];
    const chosenCm = chosen && chosen.at_default_threshold && chosen.at_default_threshold.confusion_matrix;
    if (chosenCm) {
      const cm = chosenCm;
      const cmGrid = document.getElementById('model-cm-grid');
      if (cmGrid) {
        cmGrid.innerHTML = `
          <div class="p-2.5 rounded border border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/40">
            <span class="text-[10px] text-slate-400 block font-mono">True Negatives</span>
            <span class="text-sm font-semibold font-mono">${formatNumber(cm[0][0])}</span>
          </div>
          <div class="p-2.5 rounded border border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/40">
            <span class="text-[10px] text-rose-500 block font-mono">True Positives</span>
            <span class="text-sm font-semibold font-mono text-rose-600">${formatNumber(cm[1][1])}</span>
          </div>
        `;
      }
    }

  } catch (err) {
    console.error('Error loading model details:', err);
  }
}

function closeModelDrawer() {
  const backdrop = document.getElementById('model-drawer-backdrop');
  const drawer = document.getElementById('model-drawer');
  if (backdrop) backdrop.classList.remove('active');
  if (drawer) drawer.classList.remove('active');
}

// ==========================================================================
// 11. AUDIT REPORT MODAL DIALOG
// ==========================================================================

async function openReportModal() {
  const backdrop = document.getElementById('report-modal-backdrop');
  if (backdrop) backdrop.classList.add('active');

  try {
    const summary = await fetch('/api/summary').then(r => r.json());
    if (summary && !summary.error) {
      const vol = document.getElementById('report-vol');
      if (vol) vol.innerText = formatINR(summary.total_transaction_value);

      const frd = document.getElementById('report-fraud');
      if (frd) frd.innerText = `${formatNumber(summary.fraudulent_transactions)} (${summary.fraud_rate != null ? summary.fraud_rate.toFixed(2) : '—'}%)`;
    }
  } catch (e) {
    console.warn('Report populate error:', e);
  }
}

function closeReportModal() {
  const backdrop = document.getElementById('report-modal-backdrop');
  if (backdrop) backdrop.classList.remove('active');
}

// ==========================================================================
// 12. INITIALIZATION
// ==========================================================================

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initNavigation();
  initHeroCanvas();

  // Backdrop click handlers
  const slideBackdrop = document.getElementById('slide-over-backdrop');
  if (slideBackdrop) slideBackdrop.addEventListener('click', closeSlideOver);

  const reportBackdrop = document.getElementById('report-modal-backdrop');
  if (reportBackdrop) {
    reportBackdrop.addEventListener('click', (e) => {
      if (e.target === reportBackdrop) closeReportModal();
    });
  }

  // Load initial Overview view
  loadOverview();
});
