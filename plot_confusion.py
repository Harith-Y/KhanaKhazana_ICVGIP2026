"""
Generates an interactive HTML confusion-matrix report.

Usage:
    python plot_confusion.py --data runs/<run>/eval_test_confusion.json
    python plot_confusion.py --data runs/<run>/eval_test_confusion.json --out confusion.html
"""

import argparse
import json
from pathlib import Path


HTML_TEMPLATE = r"""<!doctype html>
<title>Confusion Matrix - Food Classifier</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  :root {
    --bg: #ffffff; --surface: #f8fafc; --border: #e2e8f0;
    --text: #0f172a; --muted: #64748b; --accent: #0369a1;
    --good: #166534; --good-bg: #dcfce7;
    --bad:  #991b1b; --bad-bg:  #fee2e2;
    --cell-0: #f0f9ff; --cell-1: #bae6fd; --cell-2: #38bdf8;
    --cell-3: #0284c7; --cell-4: #0369a1; --cell-5: #1e3a8a;
    --diag: #1e3a8a;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0f172a; --surface: #1e293b; --border: #334155;
      --text: #f1f5f9; --muted: #94a3b8; --accent: #38bdf8;
      --good: #bbf7d0; --good-bg: #14532d;
      --bad:  #fecaca; --bad-bg:  #7f1d1d;
      --cell-0: #1e293b; --cell-1: #0c4a6e; --cell-2: #075985;
      --cell-3: #0369a1; --cell-4: #0284c7; --cell-5: #7dd3fc;
      --diag: #7dd3fc;
    }
  }
  :root[data-theme="light"] {
    --bg: #ffffff; --surface: #f8fafc; --border: #e2e8f0;
    --text: #0f172a; --muted: #64748b; --accent: #0369a1;
    --good: #166534; --good-bg: #dcfce7;
    --bad:  #991b1b; --bad-bg:  #fee2e2;
    --cell-0: #f0f9ff; --cell-1: #bae6fd; --cell-2: #38bdf8;
    --cell-3: #0284c7; --cell-4: #0369a1; --cell-5: #1e3a8a;
    --diag: #1e3a8a;
  }
  :root[data-theme="dark"] {
    --bg: #0f172a; --surface: #1e293b; --border: #334155;
    --text: #f1f5f9; --muted: #94a3b8; --accent: #38bdf8;
    --good: #bbf7d0; --good-bg: #14532d;
    --bad:  #fecaca; --bad-bg:  #7f1d1d;
    --cell-0: #1e293b; --cell-1: #0c4a6e; --cell-2: #075985;
    --cell-3: #0369a1; --cell-4: #0284c7; --cell-5: #7dd3fc;
    --diag: #7dd3fc;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: var(--bg); color: var(--text);
         font-size: 14px; line-height: 1.5; padding: 24px; max-width: 1400px; margin: 0 auto; }
  h1 { font-size: 20px; font-weight: 700; margin-bottom: 4px; }
  .subtitle { color: var(--muted); font-size: 13px; margin-bottom: 24px; }
  .stats { display: flex; gap: 16px; margin-bottom: 32px; flex-wrap: wrap; }
  .stat { background: var(--surface); border: 1px solid var(--border);
          border-radius: 8px; padding: 16px 24px; }
  .stat-val { font-size: 28px; font-weight: 700; color: var(--accent); }
  .stat-lbl { font-size: 12px; color: var(--muted); text-transform: uppercase;
              letter-spacing: .05em; margin-top: 2px; }
  .layout { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 32px; }
  @media (max-width: 900px) { .layout { grid-template-columns: 1fr; } }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
          padding: 16px; overflow: hidden; }
  .card h2 { font-size: 14px; font-weight: 600; margin-bottom: 12px; color: var(--muted);
             text-transform: uppercase; letter-spacing: .05em; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; padding: 6px 8px; border-bottom: 2px solid var(--border);
       color: var(--muted); font-weight: 600; cursor: pointer; user-select: none; }
  th:hover { color: var(--text); }
  td { padding: 5px 8px; border-bottom: 1px solid var(--border); }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: var(--border); }
  .acc-pill { display: inline-block; padding: 2px 8px; border-radius: 99px;
              font-size: 12px; font-weight: 600; }
  .acc-good { background: var(--good-bg); color: var(--good); }
  .acc-bad  { background: var(--bad-bg);  color: var(--bad); }
  .acc-mid  { background: var(--surface); color: var(--muted); border: 1px solid var(--border); }
  .bar-wrap { background: var(--border); border-radius: 4px; height: 8px; width: 100%; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 4px; background: var(--accent); }
  .heatmap-wrap { overflow: auto; }
  canvas { display: block; cursor: crosshair; }
  #tooltip { position: fixed; background: var(--surface); border: 1px solid var(--border);
             border-radius: 6px; padding: 8px 12px; font-size: 12px; pointer-events: none;
             display: none; z-index: 100; max-width: 240px; box-shadow: 0 4px 12px rgba(0,0,0,.15); }
  #tooltip strong { display: block; margin-bottom: 2px; }
  .theme-btn { position: fixed; top: 16px; right: 16px; background: var(--surface);
               border: 1px solid var(--border); border-radius: 6px; padding: 6px 12px;
               cursor: pointer; font-size: 13px; color: var(--text); }
  .search { width: 100%; padding: 7px 10px; border: 1px solid var(--border);
            border-radius: 6px; background: var(--bg); color: var(--text);
            font-size: 13px; margin-bottom: 10px; }
  .search:focus { outline: 2px solid var(--accent); }
  .detail-bar { display: flex; align-items: center; gap: 8px; padding: 4px 0; font-size: 12px; }
  .detail-bar-label { width: 140px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
                      flex-shrink: 0; }
  .detail-bar-count { width: 36px; text-align: right; color: var(--muted); flex-shrink: 0; }
  .full-matrix-section { margin-top: 8px; }
  .full-matrix-section summary { cursor: pointer; font-weight: 600; font-size: 13px;
                                  color: var(--muted); padding: 8px 0; }
  .full-matrix-section summary:hover { color: var(--text); }
</style>
<button class="theme-btn" onclick="toggleTheme()">Toggle theme</button>
<div id="tooltip"></div>

<h1>Confusion Matrix - Food Classifier</h1>
<p class="subtitle" id="subtitle"></p>

<div class="stats" id="stats"></div>

<div class="layout">
  <div class="card">
    <h2>Worst classes</h2>
    <input class="search" id="cls-search" placeholder="Search class…" oninput="filterTable()">
    <div style="overflow-y:auto;max-height:480px">
      <table id="cls-table">
        <thead><tr>
          <th onclick="sortTable(0)">Class ↕</th>
          <th onclick="sortTable(1)">Accuracy ↕</th>
          <th onclick="sortTable(2)">Top confusion ↕</th>
        </tr></thead>
        <tbody id="cls-tbody"></tbody>
      </table>
    </div>
  </div>
  <div class="card">
    <h2 id="detail-title">Click a class to inspect</h2>
    <div id="detail-content" style="color:var(--muted);font-size:13px">
      Select a class from the table on the left to see its full confusion breakdown.
    </div>
  </div>
</div>

<div class="card" style="margin-bottom:32px">
  <h2>Top confusion pairs (off-diagonal)</h2>
  <div style="overflow-y:auto;max-height:400px">
    <table id="pairs-table">
      <thead><tr>
        <th onclick="sortPairs(0)"># ↕</th>
        <th onclick="sortPairs(1)">True class ↕</th>
        <th onclick="sortPairs(2)">Predicted as ↕</th>
        <th onclick="sortPairs(3)">Count ↕</th>
        <th onclick="sortPairs(4)">% of true class ↕</th>
      </tr></thead>
      <tbody id="pairs-tbody"></tbody>
    </table>
  </div>
</div>

<details class="full-matrix-section card">
  <summary>Full 100×100 heatmap (row-normalised)</summary>
  <p style="font-size:12px;color:var(--muted);margin:8px 0">
    Each row shows how images of one true class were classified. Diagonal = correct.
    Hover for details. Classes sorted by accuracy (worst first).
  </p>
  <div class="heatmap-wrap" style="margin-top:12px">
    <canvas id="heatmap"></canvas>
  </div>
</details>

<script>
const DATA = __DATA__;

const classes = DATA.classes;
const matrix  = DATA.matrix;
const perAcc  = DATA.per_class_acc;
const N = classes.length;

// Sort order: worst accuracy first
const sortedIdx = [...Array(N).keys()].sort((a,b) => perAcc[classes[a]] - perAcc[classes[b]]);
const posOf = {}; // class name → sorted position
sortedIdx.forEach((ci, pos) => posOf[classes[ci]] = pos);

// ── Stats ────────────────────────────────────────────────────────────────────
document.getElementById('subtitle').textContent =
  `${DATA.split} split · ${N} classes · evaluated ${matrix.reduce((s,r)=>s+r.reduce((a,b)=>a+b,0),0).toLocaleString()} images`;

const statEl = document.getElementById('stats');
[
  [DATA.top1_acc.toFixed(2) + '%', 'Top-1 accuracy'],
  [DATA.top5_acc.toFixed(2) + '%', 'Top-5 accuracy'],
  [worstClass(), 'Worst class'],
  [topPair(),    'Top confusion pair'],
].forEach(([val, lbl]) => {
  statEl.innerHTML += `<div class="stat"><div class="stat-val">${val}</div><div class="stat-lbl">${lbl}</div></div>`;
});

function worstClass() {
  return classes.reduce((w, c) => perAcc[c] < perAcc[w] ? c : w);
}
function topPair() {
  let best = {gt: -1, pred: -1, count: 0};
  for (let g = 0; g < N; g++) for (let p = 0; p < N; p++) {
    if (g !== p && matrix[g][p] > best.count) best = {gt: g, pred: p, count: matrix[g][p]};
  }
  return best.gt >= 0 ? `${classes[best.gt]} → ${classes[best.pred]}` : '-';
}

// ── Worst-classes table ──────────────────────────────────────────────────────
let clsRows = sortedIdx.map(ci => {
  const name = classes[ci];
  const acc  = perAcc[name];
  const row  = matrix[ci];
  const total = row.reduce((a,b)=>a+b,0);
  // top off-diagonal prediction
  let topConfPred = -1, topConfCount = 0;
  for (let p = 0; p < N; p++) {
    if (p !== ci && row[p] > topConfCount) { topConfCount = row[p]; topConfPred = p; }
  }
  const topConfName  = topConfPred >= 0 ? classes[topConfPred] : '-';
  const topConfPct   = total > 0 ? (topConfCount / total * 100).toFixed(0) : 0;
  return { ci, name, acc, total, topConfName, topConfPct, topConfCount };
});

function accPill(acc) {
  const cls = acc >= 80 ? 'acc-good' : acc < 50 ? 'acc-bad' : 'acc-mid';
  return `<span class="acc-pill ${cls}">${acc.toFixed(1)}%</span>`;
}

function renderClsTable(rows) {
  document.getElementById('cls-tbody').innerHTML = rows.map(r => `
    <tr onclick="showDetail(${r.ci})" style="cursor:pointer">
      <td>${r.name}</td>
      <td>${accPill(r.acc)}</td>
      <td>${r.topConfName} <span style="color:var(--muted);font-size:11px">(${r.topConfPct}%)</span></td>
    </tr>`).join('');
}
renderClsTable(clsRows);

function filterTable() {
  const q = document.getElementById('cls-search').value.toLowerCase();
  renderClsTable(clsRows.filter(r => r.name.includes(q)));
}

let clsSortDir = [1, -1, 1];
function sortTable(col) {
  clsSortDir[col] *= -1;
  const d = clsSortDir[col];
  clsRows.sort((a,b) => {
    if (col === 0) return d * a.name.localeCompare(b.name);
    if (col === 1) return d * (a.acc - b.acc);
    if (col === 2) return d * (a.topConfCount - b.topConfCount);
  });
  renderClsTable(clsRows);
}

// ── Detail panel ─────────────────────────────────────────────────────────────
function showDetail(ci) {
  const name  = classes[ci];
  const row   = matrix[ci];
  const total = row.reduce((a,b)=>a+b,0);
  const acc   = perAcc[name];

  // top 15 predictions
  const preds = row.map((c,p) => ({p, c})).sort((a,b)=>b.c-a.c).slice(0,15);
  const maxC  = preds[0]?.c || 1;

  document.getElementById('detail-title').textContent = name;
  document.getElementById('detail-content').innerHTML = `
    <div style="margin-bottom:12px">
      ${accPill(acc)} &nbsp;
      <span style="font-size:12px;color:var(--muted)">${total} test images</span>
    </div>
    ${preds.map(({p, c}) => {
      const pct = total > 0 ? (c / total * 100).toFixed(1) : 0;
      const isDiag = (p === ci);
      const barColor = isDiag ? 'var(--accent)' : 'var(--bad)';
      const label = isDiag ? `✓ ${classes[p]}` : classes[p];
      return `<div class="detail-bar">
        <div class="detail-bar-label" title="${classes[p]}">${label}</div>
        <div class="bar-wrap" style="flex:1">
          <div class="bar-fill" style="width:${c/maxC*100}%;background:${barColor}"></div>
        </div>
        <div class="detail-bar-count">${c}</div>
        <div style="width:36px;text-align:right;font-size:11px;color:var(--muted)">${pct}%</div>
      </div>`;
    }).join('')}`;
}

// ── Top confusion pairs ──────────────────────────────────────────────────────
let pairs = [];
for (let g = 0; g < N; g++) {
  const total = matrix[g].reduce((a,b)=>a+b,0);
  for (let p = 0; p < N; p++) {
    if (g !== p && matrix[g][p] > 0)
      pairs.push({ gt: classes[g], pred: classes[p], count: matrix[g][p],
                   pct: total > 0 ? (matrix[g][p]/total*100).toFixed(1) : 0 });
  }
}
pairs.sort((a,b) => b.count - a.count);

let pairsDir = [1,1,1,-1,1];
function renderPairs(rows) {
  document.getElementById('pairs-tbody').innerHTML =
    rows.slice(0, 60).map((r, i) => `
      <tr>
        <td style="color:var(--muted)">${i+1}</td>
        <td>${r.gt}</td>
        <td>${r.pred}</td>
        <td><strong>${r.count}</strong></td>
        <td>${r.pct}%</td>
      </tr>`).join('');
}
renderPairs(pairs);

function sortPairs(col) {
  pairsDir[col] *= -1;
  const d = pairsDir[col];
  pairs.sort((a,b) => {
    if (col === 1) return d * a.gt.localeCompare(b.gt);
    if (col === 2) return d * a.pred.localeCompare(b.pred);
    if (col === 3) return d * (a.count - b.count);
    if (col === 4) return d * (parseFloat(a.pct) - parseFloat(b.pct));
    return 0;
  });
  renderPairs(pairs);
}

// ── Heatmap ───────────────────────────────────────────────────────────────────
const CELL = 6;
const LABEL_W = 110;
const LABEL_H = 110;
const canvas = document.getElementById('heatmap');
const W = LABEL_W + N * CELL;
const H = LABEL_H + N * CELL;
canvas.width  = W;
canvas.height = H;
const ctx = canvas.getContext('2d');

// sequential blue ramp (6 stops, light→dark)
const RAMP_L = ['#f0f9ff','#bae6fd','#38bdf8','#0284c7','#0369a1','#1e40af'];
const RAMP_D = ['#1e293b','#0c4a6e','#075985','#0369a1','#0284c7','#38bdf8'];
function getRamp() {
  const theme = document.documentElement.getAttribute('data-theme') ||
    (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  return theme === 'dark' ? RAMP_D : RAMP_L;
}

function interpolateColor(ramp, t) {
  const n = ramp.length - 1;
  const i = Math.min(Math.floor(t * n), n - 1);
  const f = t * n - i;
  const hex = c => parseInt(c, 16);
  const parse = h => [hex(h.slice(1,3)), hex(h.slice(3,5)), hex(h.slice(5,7))];
  const a = parse(ramp[i]), b = parse(ramp[i+1]);
  const r = Math.round(a[0] + (b[0]-a[0])*f);
  const g = Math.round(a[1] + (b[1]-a[1])*f);
  const bv= Math.round(a[2] + (b[2]-a[2])*f);
  return `rgb(${r},${g},${bv})`;
}

// row totals for normalisation
const rowTotals = matrix.map(r => r.reduce((a,b)=>a+b,0));

function drawHeatmap() {
  const ramp = getRamp();
  const isDark = (ramp === RAMP_D);
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = isDark ? '#0f172a' : '#ffffff';
  ctx.fillRect(0, 0, W, H);

  // draw cells (sorted order)
  for (let si = 0; si < N; si++) {
    const gi = sortedIdx[si];
    const total = rowTotals[gi];
    for (let sj = 0; sj < N; sj++) {
      const pj = sortedIdx[sj];
      const count = matrix[gi][pj];
      const t = total > 0 ? count / total : 0;
      ctx.fillStyle = interpolateColor(ramp, t);
      ctx.fillRect(LABEL_W + sj * CELL, LABEL_H + si * CELL, CELL, CELL);
    }
  }

  // y-axis labels (class names, abbreviated, every 5)
  ctx.font = '8px system-ui, sans-serif';
  ctx.textAlign = 'right';
  ctx.fillStyle = isDark ? '#94a3b8' : '#64748b';
  for (let si = 0; si < N; si += 5) {
    const gi = sortedIdx[si];
    const name = classes[gi].slice(0, 14);
    const y = LABEL_H + si * CELL + CELL * 2.5;
    ctx.fillText(name, LABEL_W - 4, y);
  }

  // x-axis labels (rotated, every 5)
  ctx.save();
  for (let sj = 0; sj < N; sj += 5) {
    const pj = sortedIdx[sj];
    const name = classes[pj].slice(0, 14);
    const x = LABEL_W + sj * CELL + CELL / 2;
    ctx.save();
    ctx.translate(x, LABEL_H - 4);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'left';
    ctx.fillText(name, 0, 0);
    ctx.restore();
  }
  ctx.restore();
}

drawHeatmap();

// Tooltip on canvas hover
const tooltip = document.getElementById('tooltip');
canvas.addEventListener('mousemove', e => {
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  const si = Math.floor((my - LABEL_H) / CELL);
  const sj = Math.floor((mx - LABEL_W) / CELL);
  if (si < 0 || si >= N || sj < 0 || sj >= N) {
    tooltip.style.display = 'none'; return;
  }
  const gi = sortedIdx[si];
  const pj = sortedIdx[sj];
  const count = matrix[gi][pj];
  const total = rowTotals[gi];
  const pct = total > 0 ? (count/total*100).toFixed(1) : 0;
  const label = gi === pj ? '✓ Correct' : '✗ Confused';
  tooltip.innerHTML = `<strong>${label}</strong>
    True: <b>${classes[gi]}</b><br>
    Predicted: <b>${classes[pj]}</b><br>
    Count: ${count} &nbsp; (${pct}% of row)`;
  tooltip.style.display = 'block';
  tooltip.style.left = (e.clientX + 14) + 'px';
  tooltip.style.top  = (e.clientY - 10) + 'px';
});
canvas.addEventListener('mouseleave', () => tooltip.style.display = 'none');

// click cell → open detail
canvas.addEventListener('click', e => {
  const rect = canvas.getBoundingClientRect();
  const si = Math.floor((e.clientY - rect.top  - LABEL_H) / CELL);
  if (si >= 0 && si < N) showDetail(sortedIdx[si]);
});

// ── Theme toggle ──────────────────────────────────────────────────────────────
function toggleTheme() {
  const cur = document.documentElement.getAttribute('data-theme') ||
    (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  document.documentElement.setAttribute('data-theme', cur === 'dark' ? 'light' : 'dark');
  drawHeatmap();
}
</script>
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=Path, help="eval_*_confusion.json")
    p.add_argument("--out",  type=Path, default=None)
    args = p.parse_args()

    data = json.loads(args.data.read_text())
    out  = args.out or args.data.parent / "confusion.html"

    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(data))
    out.write_text(html)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
