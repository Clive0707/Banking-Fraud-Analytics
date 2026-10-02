/**
 * Banking Intelligence — Model Lab & extended analytics panels.
 *
 * This module renders the analytics the backend already computed but the
 * dashboard never displayed. Twelve of the twenty-three API endpoints had no
 * consumer at all, including the 7x24 fraud heatmap, hourly fraud curves, geo
 * analytics, the measured data-quality scores and the alert feed.
 *
 * It also adds the evaluation views that matter for an imbalanced fraud problem
 * and that no amount of accuracy reporting can substitute for: precision-recall
 * curves against the no-skill baseline, cumulative gain, and the cost curve that
 * turns a threshold into rupees.
 *
 * Kept separate from app.js so the existing dashboard logic is untouched.
 */

'use strict';

const lab = {
  charts: {},
  cache: {}
};

// Series colours chosen to stay distinguishable in both themes and for the
// common forms of colour vision deficiency.
const SERIES_COLORS = [
  '#2563eb', // blue
  '#e11d48', // rose
  '#059669', // emerald
  '#d97706', // amber
  '#7c3aed'  // violet
];

const MUTED = '#94a3b8';

// ===========================================================================
// Helpers
// ===========================================================================

function labTheme() {
  const isDark = document.documentElement.classList.contains('dark');
  return {
    isDark,
    text: isDark ? '#64748b' : '#94a3b8',
    grid: isDark ? 'rgba(51, 65, 85, 0.3)' : 'rgba(241, 245, 249, 0.9)',
    surface: isDark ? '#0e131f' : '#ffffff'
  };
}

async function getJSON(url, cacheKey) {
  if (cacheKey && lab.cache[cacheKey]) return lab.cache[cacheKey];
  const res = await fetch(url);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `${url} returned ${res.status}`);
  }
  const data = await res.json();
  if (cacheKey) lab.cache[cacheKey] = data;
  return data;
}

function destroyChart(key) {
  if (lab.charts[key]) {
    lab.charts[key].destroy();
    delete lab.charts[key];
  }
}

function mountChart(key, canvasId, config) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return null;
  destroyChart(key);
  lab.charts[key] = new Chart(canvas.getContext('2d'), config);
  return lab.charts[key];
}

function setHTML(id, html) {
  const el = document.getElementById(id);
  if (el) el.innerHTML = html;
}

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.innerText = text;
}

function panelError(id, message) {
  setHTML(id, `
    <div class="flex items-center gap-2 text-[12px] text-slate-400 dark:text-slate-500 py-6 justify-center">
      <span class="material-symbols-outlined text-[16px]">info</span>
      <span>${message}</span>
    </div>
  `);
}

function pct(v, digits = 2) {
  if (v === null || v === undefined || isNaN(v)) return '—';
  return `${(Number(v) * 100).toFixed(digits)}%`;
}

function num(v) {
  if (v === null || v === undefined || isNaN(v)) return '—';
  return Number(v).toLocaleString('en-US');
}

function inr(v) {
  if (v === null || v === undefined || isNaN(v)) return '—';
  const n = Number(v);
  if (Math.abs(n) >= 1e7) return `₹${(n / 1e7).toFixed(2)} Cr`;
  if (Math.abs(n) >= 1e5) return `₹${(n / 1e5).toFixed(2)} L`;
  return `₹${n.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
}

// ===========================================================================
// Intelligence tab (Model Lab)
// ===========================================================================

async function loadIntelligence() {
  await Promise.allSettled([
    renderBaselineCallout(),
    renderModelTable(),
    renderCurves(),
    renderGainCurve(),
    renderCostCurve(),
    renderPrecisionAtK(),
    renderFeatureImportance(),
    renderLeakagePanel()
  ]);
}

/**
 * The single most important number on the page: a constant "never fraud"
 * predictor's accuracy. Every model in the original benchmark scored below it.
 */
async function renderBaselineCallout() {
  try {
    const perf = await getJSON('/api/model-performance', 'perf');
    const base = perf.no_skill_baseline;
    if (!base) return panelError('baseline-callout', 'Baseline not available');

    const best = perf.models?.[perf.best_model];
    const prAuc = best?.ranking?.pr_auc;
    const randomPr = base.random_scorer?.pr_auc;

    setHTML('baseline-callout', `
      <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div>
          <div class="text-[11px] uppercase tracking-wide text-slate-400 mb-1">Fraud prevalence</div>
          <div class="text-[22px] font-semibold text-slate-900 dark:text-white">${base.prevalence_pct}%</div>
          <div class="text-[11px] text-slate-400 mt-0.5">${num(base.positives)} of ${num(base.positives + base.negatives)} test rows</div>
        </div>
        <div>
          <div class="text-[11px] uppercase tracking-wide text-slate-400 mb-1">"Never fraud" accuracy</div>
          <div class="text-[22px] font-semibold text-amber-600 dark:text-amber-500">${pct(base.always_negative.accuracy)}</div>
          <div class="text-[11px] text-slate-400 mt-0.5">A constant predictor. Accuracy is meaningless here.</div>
        </div>
        <div>
          <div class="text-[11px] uppercase tracking-wide text-slate-400 mb-1">Random scorer PR-AUC</div>
          <div class="text-[22px] font-semibold text-slate-500">${randomPr ?? '—'}</div>
          <div class="text-[11px] text-slate-400 mt-0.5">The floor any model must beat.</div>
        </div>
        <div>
          <div class="text-[11px] uppercase tracking-wide text-slate-400 mb-1">Best model PR-AUC</div>
          <div class="text-[22px] font-semibold text-emerald-600 dark:text-emerald-500">${prAuc ?? '—'}</div>
          <div class="text-[11px] text-slate-400 mt-0.5">${perf.best_model || '—'} · ${prAuc && randomPr ? (prAuc / randomPr).toFixed(2) + '× random' : ''}</div>
        </div>
      </div>
    `);
  } catch (err) {
    panelError('baseline-callout', err.message);
  }
}

async function renderModelTable() {
  try {
    const perf = await getJSON('/api/model-performance', 'perf');
    const models = perf.models || {};
    // Rank by the selection metric so the table reads as a leaderboard.
    const names = Object.keys(models).sort(
      (a, b) => (models[b].ranking?.pr_auc ?? 0) - (models[a].ranking?.pr_auc ?? 0)
    );
    if (!names.length) return panelError('model-table', 'No models trained yet');

    const rows = names.map(name => {
      const m = models[name];
      const r = m.ranking || {};
      const d = m.at_default_threshold || {};
      const cv = m.cross_validation || {};
      const top1 = (m.precision_at_k || []).find(k => k.capacity_fraction === 0.01) || {};
      const isBest = name === perf.best_model;
      return `
        <tr class="border-b border-slate-100 dark:border-slate-800/60 ${isBest ? 'bg-emerald-50/50 dark:bg-emerald-900/10' : ''}">
          <td class="py-2.5 px-3 text-[12px] font-medium text-slate-900 dark:text-white whitespace-nowrap">
            ${name}${isBest ? ' <span class="text-[10px] text-emerald-600 dark:text-emerald-500 font-semibold">BEST</span>' : ''}
          </td>
          <td class="py-2.5 px-3 text-[12px] font-mono text-slate-900 dark:text-white">${r.pr_auc ?? '—'}</td>
          <td class="py-2.5 px-3 text-[12px] font-mono text-slate-500">${cv.pr_auc_mean ? `${cv.pr_auc_mean} ± ${cv.pr_auc_std}` : '—'}</td>
          <td class="py-2.5 px-3 text-[12px] font-mono text-slate-500">${r.roc_auc ?? '—'}</td>
          <td class="py-2.5 px-3 text-[12px] font-mono text-slate-500">${top1.lift ? top1.lift.toFixed(2) + '×' : '—'}</td>
          <td class="py-2.5 px-3 text-[12px] font-mono text-slate-500">${d.f1_score ?? '—'}</td>
          <td class="py-2.5 px-3 text-[12px] font-mono text-slate-400">${m.timing?.train_seconds ? m.timing.train_seconds + 's' : '—'}</td>
        </tr>
      `;
    }).join('');

    setHTML('model-table', `
      <table class="w-full text-left">
        <thead>
          <tr class="border-b border-slate-200 dark:border-slate-800">
            <th class="py-2 px-3 text-[10px] uppercase tracking-wide text-slate-400 font-medium">Model</th>
            <th class="py-2 px-3 text-[10px] uppercase tracking-wide text-slate-400 font-medium">PR-AUC</th>
            <th class="py-2 px-3 text-[10px] uppercase tracking-wide text-slate-400 font-medium">CV PR-AUC</th>
            <th class="py-2 px-3 text-[10px] uppercase tracking-wide text-slate-400 font-medium">ROC-AUC</th>
            <th class="py-2 px-3 text-[10px] uppercase tracking-wide text-slate-400 font-medium">Top-1% lift</th>
            <th class="py-2 px-3 text-[10px] uppercase tracking-wide text-slate-400 font-medium">F1 @0.5</th>
            <th class="py-2 px-3 text-[10px] uppercase tracking-wide text-slate-400 font-medium">Train</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
      <p class="text-[11px] text-slate-400 mt-3 px-3">
        Selection metric is <span class="font-medium">${perf.selection_metric || 'pr_auc'}</span>, not accuracy.
        ${perf.n_features ? `${perf.n_features} features, ` : ''}${num(perf.train_rows)} train / ${num(perf.test_rows)} test rows.
      </p>
    `);
  } catch (err) {
    panelError('model-table', err.message);
  }
}

/** Precision-recall and ROC curves for every model, with their baselines. */
async function renderCurves() {
  try {
    const perf = await getJSON('/api/model-performance', 'perf');
    const theme = labTheme();
    const models = perf.models || {};
    const names = Object.keys(models);
    if (!names.length) return;

    const prevalence = perf.no_skill_baseline?.prevalence ?? 0;

    // --- PR curves ---
    const prDatasets = names.map((name, i) => ({
      label: name,
      data: (models[name].pr_curve || []).map(p => ({ x: p.recall, y: p.precision })),
      borderColor: SERIES_COLORS[i % SERIES_COLORS.length],
      backgroundColor: 'transparent',
      borderWidth: 1.8,
      pointRadius: 0,
      tension: 0
    }));
    prDatasets.push({
      label: `No-skill (${prevalence.toFixed(4)})`,
      data: [{ x: 0, y: prevalence }, { x: 1, y: prevalence }],
      borderColor: MUTED,
      borderDash: [4, 4],
      borderWidth: 1.2,
      pointRadius: 0
    });

    mountChart('pr', 'chart-pr-curves', {
      type: 'line',
      data: { datasets: prDatasets },
      options: curveOptions(theme, 'Recall', 'Precision', { yMax: Math.max(0.1, prevalence * 8) })
    });

    // --- ROC curves ---
    const rocDatasets = names.map((name, i) => ({
      label: name,
      data: (models[name].roc_curve || []).map(p => ({ x: p.fpr, y: p.tpr })),
      borderColor: SERIES_COLORS[i % SERIES_COLORS.length],
      backgroundColor: 'transparent',
      borderWidth: 1.8,
      pointRadius: 0,
      tension: 0
    }));
    rocDatasets.push({
      label: 'Random',
      data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
      borderColor: MUTED,
      borderDash: [4, 4],
      borderWidth: 1.2,
      pointRadius: 0
    });

    mountChart('roc', 'chart-roc-curves', {
      type: 'line',
      data: { datasets: rocDatasets },
      options: curveOptions(theme, 'False positive rate', 'True positive rate', { yMax: 1 })
    });
  } catch (err) {
    console.error('Curve rendering failed:', err);
  }
}

function curveOptions(theme, xLabel, yLabel, opts = {}) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'nearest', intersect: false },
    plugins: {
      legend: {
        position: 'bottom',
        labels: { boxWidth: 10, boxHeight: 10, font: { size: 10 }, color: theme.text, usePointStyle: true }
      },
      tooltip: {
        callbacks: {
          label: c => `${c.dataset.label}: (${c.parsed.x.toFixed(3)}, ${c.parsed.y.toFixed(4)})`
        }
      }
    },
    scales: {
      x: {
        type: 'linear', min: 0, max: 1,
        title: { display: true, text: xLabel, color: theme.text, font: { size: 10 } },
        grid: { color: theme.grid }, ticks: { color: theme.text, font: { size: 10 } }
      },
      y: {
        type: 'linear', min: 0, max: opts.yMax ?? 1,
        title: { display: true, text: yLabel, color: theme.text, font: { size: 10 } },
        grid: { color: theme.grid }, ticks: { color: theme.text, font: { size: 10 } }
      }
    }
  };
}

/**
 * Cumulative gain: the chart that shows why a low-F1 model is still worth
 * deploying. Reviewing the riskiest x% of transactions captures far more than
 * x% of the fraud.
 */
async function renderGainCurve() {
  try {
    const perf = await getJSON('/api/model-performance', 'perf');
    const theme = labTheme();
    const best = perf.best_model;
    const gain = perf.models?.[best]?.gain_curve || [];
    if (!gain.length) return;

    mountChart('gain', 'chart-gain-curve', {
      type: 'line',
      data: {
        datasets: [
          {
            label: `${best} (model ranking)`,
            data: gain.map(g => ({ x: g.reviewed_pct, y: g.fraud_captured_pct })),
            borderColor: SERIES_COLORS[0],
            backgroundColor: 'rgba(37, 99, 235, 0.08)',
            fill: true,
            borderWidth: 2,
            pointRadius: 0
          },
          {
            label: 'Random review order',
            data: [{ x: 0, y: 0 }, { x: 100, y: 100 }],
            borderColor: MUTED,
            borderDash: [4, 4],
            borderWidth: 1.2,
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
            labels: { boxWidth: 10, boxHeight: 10, font: { size: 10 }, color: theme.text, usePointStyle: true }
          },
          tooltip: {
            callbacks: {
              label: c => `Review ${c.parsed.x}% → catch ${c.parsed.y.toFixed(1)}% of fraud`
            }
          }
        },
        scales: {
          x: {
            type: 'linear', min: 0, max: 100,
            title: { display: true, text: '% of portfolio reviewed', color: theme.text, font: { size: 10 } },
            grid: { color: theme.grid }, ticks: { color: theme.text, font: { size: 10 } }
          },
          y: {
            min: 0, max: 100,
            title: { display: true, text: '% of fraud captured', color: theme.text, font: { size: 10 } },
            grid: { color: theme.grid }, ticks: { color: theme.text, font: { size: 10 } }
          }
        }
      }
    });
  } catch (err) {
    console.error('Gain curve failed:', err);
  }
}

/**
 * Expected net saving against the decision threshold, with the optimum marked.
 * This is what converts a risk score into an operating policy.
 */
async function renderCostCurve() {
  try {
    const th = await getJSON('/api/thresholds', 'thresholds');
    const perf = await getJSON('/api/model-performance', 'perf');
    const theme = labTheme();
    const best = perf.best_model;
    const block = th.models?.[best];
    if (!block || !block.cost_curve?.length) {
      return panelError('cost-summary', 'Threshold analysis not available');
    }

    const curve = block.cost_curve;
    const optimal = block.cost_optimal;

    mountChart('cost', 'chart-cost-curve', {
      type: 'line',
      data: {
        datasets: [
          {
            label: 'Net saving',
            data: curve.map(c => ({ x: c.threshold, y: c.net_saving })),
            borderColor: SERIES_COLORS[2],
            backgroundColor: 'rgba(5, 150, 105, 0.08)',
            fill: true,
            borderWidth: 2,
            pointRadius: 0
          },
          {
            label: 'Cost-optimal threshold',
            data: optimal ? [{ x: optimal.threshold, y: optimal.net_saving }] : [],
            borderColor: SERIES_COLORS[1],
            backgroundColor: SERIES_COLORS[1],
            pointRadius: 5,
            pointStyle: 'circle',
            showLine: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'bottom',
            labels: { boxWidth: 10, boxHeight: 10, font: { size: 10 }, color: theme.text, usePointStyle: true }
          },
          tooltip: {
            callbacks: {
              label: c => `threshold ${c.parsed.x.toFixed(3)} → ${inr(c.parsed.y)}`
            }
          }
        },
        scales: {
          x: {
            type: 'linear',
            title: { display: true, text: 'Decision threshold', color: theme.text, font: { size: 10 } },
            grid: { color: theme.grid }, ticks: { color: theme.text, font: { size: 10 } }
          },
          y: {
            title: { display: true, text: 'Expected net saving', color: theme.text, font: { size: 10 } },
            grid: { color: theme.grid },
            ticks: { color: theme.text, font: { size: 10 }, callback: v => inr(v) }
          }
        }
      }
    });

    const assume = th.cost_assumptions || {};
    if (optimal) {
      setHTML('cost-summary', `
        <div class="grid grid-cols-2 md:grid-cols-5 gap-4">
          <div>
            <div class="text-[11px] text-slate-400 mb-1">Optimal threshold</div>
            <div class="text-[18px] font-semibold font-mono text-slate-900 dark:text-white">${optimal.threshold.toFixed(3)}</div>
          </div>
          <div>
            <div class="text-[11px] text-slate-400 mb-1">Alerts raised</div>
            <div class="text-[18px] font-semibold text-slate-900 dark:text-white">${num(optimal.alerts_raised)}</div>
          </div>
          <div>
            <div class="text-[11px] text-slate-400 mb-1">Frauds caught</div>
            <div class="text-[18px] font-semibold text-slate-900 dark:text-white">${num(optimal.frauds_caught)}</div>
          </div>
          <div>
            <div class="text-[11px] text-slate-400 mb-1">Fraud value recovered</div>
            <div class="text-[18px] font-semibold text-slate-900 dark:text-white">${inr(optimal.fraud_value_caught)}</div>
          </div>
          <div>
            <div class="text-[11px] text-slate-400 mb-1">Net saving</div>
            <div class="text-[18px] font-semibold text-emerald-600 dark:text-emerald-500">${inr(optimal.net_saving)}</div>
          </div>
        </div>
        <p class="text-[11px] text-slate-400 mt-3">
          Assumes ₹${num(assume.cost_per_review_inr)} per manual review and
          ${pct(assume.recovery_rate, 0)} recovery on a caught fraud, measured over the
          ${num(perf.test_rows)}-row held-out test set.
        </p>
      `);
    }
  } catch (err) {
    panelError('cost-summary', err.message);
  }
}

/** precision@k: what an analyst team actually experiences. */
async function renderPrecisionAtK() {
  try {
    const perf = await getJSON('/api/model-performance', 'perf');
    const best = perf.best_model;
    const rows = perf.models?.[best]?.precision_at_k || [];
    if (!rows.length) return panelError('patk-table', 'Not available');

    setHTML('patk-table', `
      <table class="w-full text-left">
        <thead>
          <tr class="border-b border-slate-200 dark:border-slate-800">
            <th class="py-2 px-3 text-[10px] uppercase tracking-wide text-slate-400 font-medium">Review capacity</th>
            <th class="py-2 px-3 text-[10px] uppercase tracking-wide text-slate-400 font-medium">Reviewed</th>
            <th class="py-2 px-3 text-[10px] uppercase tracking-wide text-slate-400 font-medium">Frauds caught</th>
            <th class="py-2 px-3 text-[10px] uppercase tracking-wide text-slate-400 font-medium">Precision</th>
            <th class="py-2 px-3 text-[10px] uppercase tracking-wide text-slate-400 font-medium">Recall</th>
            <th class="py-2 px-3 text-[10px] uppercase tracking-wide text-slate-400 font-medium">Lift</th>
          </tr>
        </thead>
        <tbody>
          ${rows.map(r => `
            <tr class="border-b border-slate-100 dark:border-slate-800/60">
              <td class="py-2 px-3 text-[12px] font-medium text-slate-900 dark:text-white">Top ${r.capacity_pct}%</td>
              <td class="py-2 px-3 text-[12px] font-mono text-slate-500">${num(r.reviewed)}</td>
              <td class="py-2 px-3 text-[12px] font-mono text-slate-500">${num(r.frauds_caught)}</td>
              <td class="py-2 px-3 text-[12px] font-mono text-slate-900 dark:text-white">${pct(r.precision_at_k)}</td>
              <td class="py-2 px-3 text-[12px] font-mono text-slate-500">${pct(r.recall_at_k)}</td>
              <td class="py-2 px-3 text-[12px] font-mono font-semibold text-emerald-600 dark:text-emerald-500">${r.lift.toFixed(2)}×</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    `);
  } catch (err) {
    panelError('patk-table', err.message);
  }
}

async function renderFeatureImportance() {
  try {
    const fi = await getJSON('/api/feature-importance', 'fi');
    const perf = await getJSON('/api/model-performance', 'perf');
    const theme = labTheme();

    const entry = (fi.models || []).find(m => m.model === perf.best_model) || (fi.models || [])[0];
    if (!entry) return;

    const top = entry.features.slice(0, 12).reverse();
    mountChart('fi', 'chart-feature-importance', {
      type: 'bar',
      data: {
        labels: top.map(f => f.feature),
        datasets: [{
          label: entry.kind,
          data: top.map(f => f.importance),
          backgroundColor: SERIES_COLORS[0],
          borderRadius: 2,
          barThickness: 12
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { color: theme.grid }, ticks: { color: theme.text, font: { size: 10 } } },
          y: { grid: { display: false }, ticks: { color: theme.text, font: { size: 10 } } }
        }
      }
    });
    setText('fi-caption', `${entry.model} · ${entry.kind.replace(/_/g, ' ')}`);
  } catch (err) {
    console.error('Feature importance failed:', err);
  }
}

/**
 * The target-leakage audit. This is the finding that most distinguishes the
 * analysis: transaction_status predicts fraud perfectly and had to be excluded.
 */
async function renderLeakagePanel() {
  try {
    const leak = await getJSON('/api/leakage', 'leak');
    const s = leak.status_leakage;
    if (!s || !s.available) return panelError('leakage-panel', 'Leakage audit not available');

    const rule = s.zero_model_rule || {};
    const cmp = leak.leaky_vs_clean || {};
    const clean = cmp.clean_no_status || {};
    const leaky = cmp.leaky_with_status || {};
    const infl = cmp.inflation || {};

    const breakdownRows = (s.breakdown || []).map(b => `
      <tr class="border-b border-slate-100 dark:border-slate-800/60">
        <td class="py-2 px-3 text-[12px] font-medium text-slate-900 dark:text-white">${b.status}</td>
        <td class="py-2 px-3 text-[12px] font-mono text-slate-500">${num(b.legitimate)}</td>
        <td class="py-2 px-3 text-[12px] font-mono text-slate-500">${num(b.fraudulent)}</td>
        <td class="py-2 px-3 text-[12px] font-mono font-semibold ${b.fraud_rate_pct >= 99 ? 'text-rose-600 dark:text-rose-500' : 'text-slate-900 dark:text-white'}">${b.fraud_rate_pct}%</td>
      </tr>
    `).join('');

    setHTML('leakage-panel', `
      <div class="flex items-start gap-3 p-4 rounded-lg bg-rose-50 dark:bg-rose-900/15 border border-rose-200/70 dark:border-rose-800/40 mb-5">
        <span class="material-symbols-outlined text-[20px] text-rose-600 dark:text-rose-500 mt-0.5">warning</span>
        <div>
          <div class="text-[13px] font-semibold text-rose-900 dark:text-rose-300">${s.verdict}</div>
          <p class="text-[12px] text-rose-800/80 dark:text-rose-300/70 mt-1 leading-relaxed">${s.explanation}</p>
        </div>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <div class="text-[11px] uppercase tracking-wide text-slate-400 mb-2 px-3">transaction_status × is_fraud</div>
          <table class="w-full text-left">
            <thead>
              <tr class="border-b border-slate-200 dark:border-slate-800">
                <th class="py-2 px-3 text-[10px] uppercase tracking-wide text-slate-400 font-medium">Status</th>
                <th class="py-2 px-3 text-[10px] uppercase tracking-wide text-slate-400 font-medium">Legit</th>
                <th class="py-2 px-3 text-[10px] uppercase tracking-wide text-slate-400 font-medium">Fraud</th>
                <th class="py-2 px-3 text-[10px] uppercase tracking-wide text-slate-400 font-medium">Fraud rate</th>
              </tr>
            </thead>
            <tbody>${breakdownRows}</tbody>
          </table>
          <p class="text-[11px] text-slate-400 mt-3 px-3 leading-relaxed">
            The rule <code class="font-mono text-slate-600 dark:text-slate-300">${rule.rule || ''}</code>
            reaches <span class="font-semibold text-slate-700 dark:text-slate-200">precision ${rule.precision}</span>
            and <span class="font-semibold text-slate-700 dark:text-slate-200">recall ${rule.recall}</span>
            with no model at all — ${num(rule.frauds_matched)} frauds from
            ${num(rule.transactions_matched)} matched transactions and
            ${num(rule.false_positives)} false positives.
          </p>
        </div>

        <div>
          <div class="text-[11px] uppercase tracking-wide text-slate-400 mb-2 px-3">Same model, with and without the leak</div>
          <table class="w-full text-left">
            <thead>
              <tr class="border-b border-slate-200 dark:border-slate-800">
                <th class="py-2 px-3 text-[10px] uppercase tracking-wide text-slate-400 font-medium">Metric</th>
                <th class="py-2 px-3 text-[10px] uppercase tracking-wide text-slate-400 font-medium">Clean</th>
                <th class="py-2 px-3 text-[10px] uppercase tracking-wide text-slate-400 font-medium">Leaky</th>
              </tr>
            </thead>
            <tbody>
              ${[['PR-AUC', 'pr_auc'], ['ROC-AUC', 'roc_auc'], ['Precision', 'precision'], ['Recall', 'recall'], ['F1', 'f1_score']].map(([label, key]) => `
                <tr class="border-b border-slate-100 dark:border-slate-800/60">
                  <td class="py-2 px-3 text-[12px] text-slate-600 dark:text-slate-300">${label}</td>
                  <td class="py-2 px-3 text-[12px] font-mono text-slate-900 dark:text-white">${clean[key] ?? '—'}</td>
                  <td class="py-2 px-3 text-[12px] font-mono text-rose-600 dark:text-rose-500">${leaky[key] ?? '—'}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
          <div class="mt-3 px-3 p-3 rounded-lg bg-slate-50 dark:bg-slate-800/40">
            <div class="text-[12px] text-slate-700 dark:text-slate-200">
              Including the leaked column inflates PR-AUC
              <span class="font-semibold text-rose-600 dark:text-rose-500">${infl.pr_auc_multiple ?? '—'}×</span>
              and F1 <span class="font-semibold text-rose-600 dark:text-rose-500">${infl.f1_multiple ?? '—'}×</span>.
            </div>
            <p class="text-[11px] text-slate-400 mt-1.5 leading-relaxed">${infl.interpretation || ''}</p>
          </div>
        </div>
      </div>
    `);
  } catch (err) {
    panelError('leakage-panel', err.message);
  }
}

// ===========================================================================
// Overview extras — heatmap, data quality, alerts, pipeline
// ===========================================================================

async function loadOverviewExtras() {
  await Promise.allSettled([
    renderHeatmap(),
    renderDataQuality(),
    renderAlerts(),
    renderPipelineBenchmark()
  ]);
}

/**
 * 7x24 day-by-hour fraud heatmap. The backend has always computed this grid;
 * nothing ever rendered it.
 */
async function renderHeatmap() {
  try {
    const ta = await getJSON('/api/time-analytics', 'time');
    const grid = ta.heatmap || [];
    if (!grid.length) return panelError('heatmap-grid', 'Heatmap data unavailable');

    const allRates = grid.flatMap(d => d.hours.map(h => h.fraud_rate || 0));
    const maxRate = Math.max(...allRates, 0.0001);

    const cellFor = (cell) => {
      const intensity = (cell.fraud_rate || 0) / maxRate;
      // Single-hue ramp: lightness carries the value, so it stays readable in
      // both themes and when printed in greyscale.
      const alpha = 0.06 + intensity * 0.94;
      return `
        <div class="h-5 rounded-[2px] cursor-default transition-transform hover:scale-125 hover:z-10 relative"
             style="background-color: rgba(225, 29, 72, ${alpha.toFixed(3)})"
             title="${cell.day} ${String(cell.hour).padStart(2, '0')}:00 — ${cell.fraud_rate}% fraud (${num(cell.fraud_count)} of ${num(cell.count)})">
        </div>
      `;
    };

    setHTML('heatmap-grid', `
      <div class="overflow-x-auto">
        <div class="min-w-[640px]">
          <div class="grid gap-1 mb-1" style="grid-template-columns: 38px repeat(24, 1fr)">
            <div></div>
            ${Array.from({ length: 24 }, (_, h) =>
              `<div class="text-[9px] text-slate-400 text-center">${h % 3 === 0 ? String(h).padStart(2, '0') : ''}</div>`
            ).join('')}
          </div>
          ${grid.map(day => `
            <div class="grid gap-1 mb-1" style="grid-template-columns: 38px repeat(24, 1fr)">
              <div class="text-[10px] text-slate-400 flex items-center">${day.day_name.slice(0, 3)}</div>
              ${day.hours.map(cellFor).join('')}
            </div>
          `).join('')}
        </div>
      </div>
      <div class="flex items-center justify-between mt-3">
        <p class="text-[11px] text-slate-400">Fraud rate by weekday and hour, across all 15M transactions.</p>
        <div class="flex items-center gap-2">
          <span class="text-[10px] text-slate-400">0%</span>
          <div class="flex gap-0.5">
            ${[0.06, 0.25, 0.45, 0.65, 0.85, 1].map(a =>
              `<div class="w-4 h-3 rounded-[2px]" style="background-color: rgba(225, 29, 72, ${a})"></div>`
            ).join('')}
          </div>
          <span class="text-[10px] text-slate-400">${maxRate.toFixed(2)}%</span>
        </div>
      </div>
    `);
  } catch (err) {
    panelError('heatmap-grid', err.message);
  }
}

/** The measured data-quality scores, including the consistency defect. */
async function renderDataQuality() {
  try {
    const dq = await getJSON('/api/data-quality', 'dq');
    const scores = [
      ['Completeness', dq.completeness_score],
      ['Uniqueness', dq.uniqueness_score],
      ['Validity', dq.validity_score],
      ['Consistency', dq.consistency_score]
    ];
    const cc = dq.consistency_check || {};

    setHTML('data-quality-panel', `
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        ${scores.map(([label, v]) => {
          const ok = v >= 99.5;
          return `
            <div>
              <div class="flex items-baseline justify-between mb-1.5">
                <span class="text-[11px] text-slate-400">${label}</span>
                <span class="text-[13px] font-semibold font-mono ${ok ? 'text-emerald-600 dark:text-emerald-500' : 'text-amber-600 dark:text-amber-500'}">${v !== undefined ? v.toFixed(2) + '%' : '—'}</span>
              </div>
              <div class="h-1.5 rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
                <div class="h-full rounded-full ${ok ? 'bg-emerald-500' : 'bg-amber-500'}" style="width: ${Math.max(0, Math.min(100, v || 0))}%"></div>
              </div>
            </div>
          `;
        }).join('')}
      </div>
      ${cc.violations ? `
        <div class="flex items-start gap-2.5 p-3 rounded-lg bg-amber-50 dark:bg-amber-900/15 border border-amber-200/70 dark:border-amber-800/40">
          <span class="material-symbols-outlined text-[18px] text-amber-600 dark:text-amber-500 mt-0.5">rule</span>
          <div>
            <div class="text-[12px] font-medium text-amber-900 dark:text-amber-300">
              ${num(cc.violations)} rows (${cc.violation_pct}%) fail <code class="font-mono">${cc.rule}</code>
            </div>
            <p class="text-[11px] text-amber-800/80 dark:text-amber-300/70 mt-1 leading-relaxed">${cc.finding}</p>
          </div>
        </div>
      ` : ''}
      <p class="text-[11px] text-slate-400 mt-3">
        Measured across ${num(dq.total_records_raw || dq.total_records)} rows and
        ${dq.total_columns} columns. ${num(dq.missing_values_count)} missing values,
        ${num(dq.duplicate_records_count)} duplicate transaction ids.
      </p>
    `);
  } catch (err) {
    panelError('data-quality-panel', err.message);
  }
}

/** Computed alert feed with the evidence pointer for each alert. */
async function renderAlerts() {
  try {
    const alerts = await getJSON('/api/alerts', 'alerts');
    if (!Array.isArray(alerts) || !alerts.length) {
      return panelError('alerts-feed', 'No alerts generated');
    }

    const sevStyle = {
      Critical: 'bg-rose-50 dark:bg-rose-900/15 text-rose-700 dark:text-rose-400 border-rose-200/70 dark:border-rose-800/40',
      High: 'bg-orange-50 dark:bg-orange-900/15 text-orange-700 dark:text-orange-400 border-orange-200/70 dark:border-orange-800/40',
      Medium: 'bg-amber-50 dark:bg-amber-900/15 text-amber-700 dark:text-amber-400 border-amber-200/70 dark:border-amber-800/40',
      Info: 'bg-slate-50 dark:bg-slate-800/40 text-slate-600 dark:text-slate-400 border-slate-200/70 dark:border-slate-700/40'
    };

    setHTML('alerts-feed', alerts.map(a => `
      <div class="flex items-start gap-3 py-3 border-b border-slate-100 dark:border-slate-800/60 last:border-0">
        <span class="px-2 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wide border ${sevStyle[a.severity] || sevStyle.Info} whitespace-nowrap">${a.severity}</span>
        <div class="min-w-0 flex-1">
          <div class="text-[12.5px] font-medium text-slate-900 dark:text-white">${a.title}</div>
          <p class="text-[11.5px] text-slate-500 dark:text-slate-400 mt-0.5 leading-relaxed">${a.description}</p>
          ${a.evidence ? `<div class="text-[10px] text-slate-400 mt-1 font-mono">↳ ${a.evidence}</div>` : ''}
        </div>
        <span class="text-[11px] font-mono text-slate-400 whitespace-nowrap">${a.metric || ''}</span>
      </div>
    `).join(''));
  } catch (err) {
    panelError('alerts-feed', err.message);
  }
}

/** Per-stage Spark timings, so the "big data" claim is demonstrable. */
async function renderPipelineBenchmark() {
  try {
    const b = await getJSON('/api/processing', 'bench');
    if (b.error || !b.stages) return panelError('pipeline-panel', b.error || 'Benchmark unavailable');

    const maxSecs = Math.max(...b.stages.map(s => s.seconds), 0.01);

    setHTML('pipeline-panel', `
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-5">
        <div>
          <div class="text-[11px] text-slate-400 mb-1">Rows processed</div>
          <div class="text-[18px] font-semibold text-slate-900 dark:text-white">${num(b.input_rows)}</div>
        </div>
        <div>
          <div class="text-[11px] text-slate-400 mb-1">Total runtime</div>
          <div class="text-[18px] font-semibold text-slate-900 dark:text-white">${b.total_seconds}s</div>
        </div>
        <div>
          <div class="text-[11px] text-slate-400 mb-1">Engine</div>
          <div class="text-[13px] font-semibold text-slate-900 dark:text-white mt-1">Spark ${b.spark_version || ''} <span class="font-mono text-[11px] text-slate-400">${b.spark_master || ''}</span></div>
        </div>
        <div>
          <div class="text-[11px] text-slate-400 mb-1">Hadoop / HDFS</div>
          <div class="text-[13px] font-semibold text-emerald-600 dark:text-emerald-500 mt-1">Not used</div>
        </div>
      </div>
      <div class="space-y-2">
        ${b.stages.map(s => `
          <div class="flex items-center gap-3">
            <div class="w-44 text-[11px] text-slate-500 dark:text-slate-400 font-mono truncate">${s.stage}</div>
            <div class="flex-1 h-4 rounded bg-slate-100 dark:bg-slate-800 overflow-hidden">
              <div class="h-full rounded bg-slate-400 dark:bg-slate-600" style="width: ${(s.seconds / maxSecs * 100).toFixed(1)}%"></div>
            </div>
            <div class="w-16 text-right text-[11px] font-mono text-slate-500">${s.seconds.toFixed(2)}s</div>
            <div class="w-28 text-right text-[10px] font-mono text-slate-400">${s.rows_per_second ? num(s.rows_per_second) + ' r/s' : ''}</div>
          </div>
        `).join('')}
      </div>
    `);
  } catch (err) {
    panelError('pipeline-panel', err.message);
  }
}

// ===========================================================================
// Fraud extras — segment lift, hourly curve, geo
// ===========================================================================

async function loadFraudExtras() {
  await Promise.allSettled([
    renderSegmentLift(),
    renderHourlyFraud(),
    renderGeoLift()
  ]);
}

/** Measured fraud lift per risk segment: the evidence behind the model features. */
async function renderSegmentLift() {
  try {
    const sl = await getJSON('/api/segment-lift', 'lift');
    const theme = labTheme();
    const segs = (sl.segments || []).slice().sort((a, b) => a.lift_vs_baseline - b.lift_vs_baseline);
    if (!segs.length) return;

    mountChart('lift', 'chart-segment-lift', {
      type: 'bar',
      data: {
        labels: segs.map(s => s.segment),
        datasets: [{
          label: 'Lift vs baseline',
          data: segs.map(s => s.lift_vs_baseline),
          backgroundColor: segs.map(s =>
            s.lift_vs_baseline >= 5 ? '#e11d48'
              : s.lift_vs_baseline >= 2 ? '#d97706'
                : s.lift_vs_baseline >= 1 ? '#64748b' : '#059669'
          ),
          borderRadius: 2,
          barThickness: 14
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: c => {
                const s = segs[c.dataIndex];
                return `${s.lift_vs_baseline}× — ${s.fraud_rate_pct}% of ${num(s.transactions)} txns`;
              }
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: `Lift vs ${sl.baseline_fraud_rate_pct}% baseline`, color: theme.text, font: { size: 10 } },
            grid: { color: theme.grid }, ticks: { color: theme.text, font: { size: 10 }, callback: v => v + '×' }
          },
          y: { grid: { display: false }, ticks: { color: theme.text, font: { size: 10 } } }
        }
      }
    });
  } catch (err) {
    console.error('Segment lift failed:', err);
  }
}

/** Fraud rate by hour of day — the 6x overnight spike, measured on 15M rows. */
async function renderHourlyFraud() {
  try {
    const ta = await getJSON('/api/time-analytics', 'time');
    const theme = labTheme();
    const hourly = ta.hourly || [];
    if (!hourly.length) return;

    mountChart('hourly', 'chart-hourly-fraud', {
      type: 'bar',
      data: {
        labels: hourly.map(h => h.hour_label),
        datasets: [
          {
            type: 'line',
            label: 'Fraud rate (%)',
            data: hourly.map(h => h.fraud_rate),
            borderColor: SERIES_COLORS[1],
            backgroundColor: 'rgba(225, 29, 72, 0.08)',
            fill: true,
            borderWidth: 2,
            pointRadius: 0,
            yAxisID: 'y',
            tension: 0.25
          },
          {
            type: 'bar',
            label: 'Transactions',
            data: hourly.map(h => h.count),
            backgroundColor: theme.isDark ? 'rgba(51,65,85,0.6)' : 'rgba(203,213,225,0.7)',
            borderRadius: 2,
            yAxisID: 'y1'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: {
            position: 'bottom',
            labels: { boxWidth: 10, boxHeight: 10, font: { size: 10 }, color: theme.text, usePointStyle: true }
          }
        },
        scales: {
          x: { grid: { display: false }, ticks: { color: theme.text, font: { size: 9 }, maxRotation: 0, autoSkipPadding: 8 } },
          y: {
            position: 'left', beginAtZero: true,
            title: { display: true, text: 'Fraud rate (%)', color: theme.text, font: { size: 10 } },
            grid: { color: theme.grid }, ticks: { color: theme.text, font: { size: 10 } }
          },
          y1: {
            position: 'right', beginAtZero: true,
            title: { display: true, text: 'Volume', color: theme.text, font: { size: 10 } },
            grid: { display: false },
            ticks: { color: theme.text, font: { size: 10 }, callback: v => (v / 1000) + 'k' }
          }
        }
      }
    });
  } catch (err) {
    console.error('Hourly fraud failed:', err);
  }
}

/** Fraud rate by location, separating the international corridors. */
async function renderGeoLift() {
  try {
    const geo = await getJSON('/api/geo-analytics', 'geo');
    const theme = labTheme();
    const intl = new Set(geo.international_locations || []);
    const locs = (geo.locations || []).slice().sort((a, b) => b.fraud_rate - a.fraud_rate);
    if (!locs.length) return;

    mountChart('geo', 'chart-geo-lift', {
      type: 'bar',
      data: {
        labels: locs.map(l => l.location),
        datasets: [{
          label: 'Fraud rate (%)',
          data: locs.map(l => l.fraud_rate),
          backgroundColor: locs.map(l => intl.has(l.location) ? '#e11d48' : (theme.isDark ? '#475569' : '#cbd5e1')),
          borderRadius: 2
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: c => {
                const l = locs[c.dataIndex];
                return `${l.fraud_rate}% · ${num(l.fraud_count)} frauds of ${num(l.count)}${intl.has(l.location) ? ' · international' : ''}`;
              }
            }
          }
        },
        scales: {
          x: { grid: { display: false }, ticks: { color: theme.text, font: { size: 9 }, maxRotation: 45, minRotation: 45 } },
          y: {
            beginAtZero: true,
            title: { display: true, text: 'Fraud rate (%)', color: theme.text, font: { size: 10 } },
            grid: { color: theme.grid }, ticks: { color: theme.text, font: { size: 10 } }
          }
        }
      }
    });
    setText('geo-caption', `Rose bars are international corridors (${[...intl].join(', ')}).`);
  } catch (err) {
    console.error('Geo lift failed:', err);
  }
}

// ===========================================================================
// Theme handling — redraw every lab chart when the theme flips
// ===========================================================================

function refreshLabCharts() {
  const active = document.querySelector('.tab-pane.active')?.id;
  if (active === 'tab-intelligence') loadIntelligence();
  else if (active === 'tab-overview') loadOverviewExtras();
  else if (active === 'tab-fraud') loadFraudExtras();
}

document.addEventListener('DOMContentLoaded', () => {
  const btn = document.getElementById('theme-toggle-btn');
  if (btn) btn.addEventListener('click', () => setTimeout(refreshLabCharts, 60));
});

// Exposed for app.js, which calls these from switchTab and the tab loaders.
window.loadIntelligence = loadIntelligence;
window.loadOverviewExtras = loadOverviewExtras;
window.loadFraudExtras = loadFraudExtras;
