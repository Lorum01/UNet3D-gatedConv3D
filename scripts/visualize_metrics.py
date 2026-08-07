"""Genera un report HTML autosufficiente (nessuna dipendenza esterna/internet) che
raccoglie tutti i metrics.csv / seeds_summary_*.csv sotto una cartella (default
Model_Results) e li rende esplorabili con grafici e una tabella filtrabile,
invece di dover aprire decine di CSV a mano.

Uso:
    python scripts/visualize_metrics.py
    python scripts/visualize_metrics.py --root Model_Results --out Model_Results/metrics_report.html

Rilancialo ogni volta che finiscono nuove run: rilegge tutto da zero, non
modifica alcun CSV originale.

Per ogni cartella diretta sotto --root (eccetto 'deprecated'):
- se contiene seeds_summary_{test,val,train}.csv (run multiseed), usa quelli:
  mean/std gia' aggregati tra i seed (vedi aggregate_metrics_across_seeds).
- altrimenti, se contiene <split>/metrics.csv (run singola, non multiseed),
  la tratta come un punto unico (std=0, n_seeds=1).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SPLITS_ORDER = ["train", "val", "test"]
METRICS_ORDER = ["combined_loss", "mse", "psnr", "ssim"]


def _safe_float(x: str) -> Optional[float]:
    x = (x or "").strip()
    return float(x) if x else None


def _extract_n_seeds(first_line: str) -> Optional[int]:
    m = re.search(r"n=(\d+)", first_line)
    return int(m.group(1)) if m else None


def _parse_seeds_summary(path: Path) -> Tuple[Dict[Tuple[str, str], Dict[str, Dict[str, float]]], List[str], Optional[int]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    n_seeds = _extract_n_seeds(lines[0]) if lines else None
    reader = csv.DictReader(lines[1:])
    t_cols = [c for c in (reader.fieldnames or []) if c not in ("branch", "metric", "stat")]
    result: Dict[Tuple[str, str], Dict[str, Dict[str, float]]] = {}
    for row in reader:
        key = (row["branch"], row["metric"])
        entry = result.setdefault(key, {})
        vals = {c: float(row[c]) for c in t_cols}
        if row["stat"] == "seed_mean":
            entry["mean_vals"] = vals
        elif row["stat"] == "seed_std":
            entry["std_vals"] = vals
    return result, t_cols, n_seeds


def _collect_rows(root: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    configs: List[str] = []

    for config_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "deprecated"):
        found_any = False
        for split in SPLITS_ORDER:
            summary_path = config_dir / f"seeds_summary_{split}.csv"
            if summary_path.exists():
                parsed, t_cols, n_seeds = _parse_seeds_summary(summary_path)
                t_labels = [c for c in t_cols if c != "mean"]
                for (branch, metric), entry in parsed.items():
                    mean_vals = entry.get("mean_vals", {})
                    std_vals = entry.get("std_vals", {})
                    rows.append({
                        "config": config_dir.name,
                        "split": split,
                        "branch": branch,
                        "metric": metric,
                        "t_labels": t_labels,
                        "t": [mean_vals.get(c) for c in t_labels],
                        "std_t": [std_vals.get(c, 0.0) for c in t_labels],
                        "mean": mean_vals.get("mean"),
                        "std_mean": std_vals.get("mean", 0.0),
                        "n_seeds": n_seeds,
                    })
                    found_any = True
                continue

            plain_path = config_dir / split / "metrics.csv"
            if plain_path.exists():
                with open(plain_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    t_labels = [c for c in (reader.fieldnames or []) if c not in ("branch", "metric", "mean")]
                    for row in reader:
                        mean_val = _safe_float(row.get("mean", ""))
                        if mean_val is None and not any(row.get(c) for c in t_labels):
                            continue  # riga vuota/malformata
                        rows.append({
                            "config": config_dir.name,
                            "split": split,
                            "branch": row["branch"],
                            "metric": row["metric"],
                            "t_labels": t_labels,
                            "t": [_safe_float(row[c]) for c in t_labels],
                            "std_t": [0.0 for _ in t_labels],
                            "mean": mean_val,
                            "std_mean": 0.0,
                            "n_seeds": 1,
                        })
                        found_any = True
        if found_any:
            configs.append(config_dir.name)

    return rows, configs


HTML_TEMPLATE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Metrics report</title>
<style>
:root {
  color-scheme: light;
  --page:            #f9f9f7;
  --surface-1:       #fcfcfb;
  --text-primary:    #0b0b0b;
  --text-secondary:  #52514e;
  --text-muted:      #898781;
  --gridline:        #e1e0d9;
  --axis:            #c3c2b7;
  --border:          rgba(11,11,11,0.10);
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; --series-4: #eda100;
  --series-5: #e87ba4; --series-6: #008300; --series-7: #4a3aa7; --series-8: #e34948;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page:            #0d0d0d;
    --surface-1:       #1a1a19;
    --text-primary:    #ffffff;
    --text-secondary:  #c3c2b7;
    --text-muted:      #898781;
    --gridline:        #2c2c2a;
    --axis:            #383835;
    --border:          rgba(255,255,255,0.10);
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
    --series-5: #d55181; --series-6: #008300; --series-7: #9085e9; --series-8: #e66767;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page:            #0d0d0d;
  --surface-1:       #1a1a19;
  --text-primary:    #ffffff;
  --text-secondary:  #c3c2b7;
  --text-muted:      #898781;
  --gridline:        #2c2c2a;
  --axis:            #383835;
  --border:          rgba(255,255,255,0.10);
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
  --series-5: #d55181; --series-6: #008300; --series-7: #9085e9; --series-8: #e66767;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 24px; background: var(--page); color: var(--text-primary);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
}
h1 { font-size: 18px; margin: 0 0 4px; }
.subtitle { color: var(--text-secondary); margin: 0 0 20px; font-size: 13px; }
.card {
  background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
  padding: 16px; margin-bottom: 16px;
}
.filters { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; margin-bottom: 16px; }
.filters label { color: var(--text-secondary); font-size: 12px; margin-right: 6px; }
select, input[type=text], button {
  font: inherit; background: var(--surface-1); color: var(--text-primary);
  border: 1px solid var(--axis); border-radius: 6px; padding: 5px 8px;
}
button { cursor: pointer; }
button:hover { background: var(--gridline); }
button:disabled { opacity: 0.45; cursor: default; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; margin: 4px 0 16px; }
.legend-item { display: flex; align-items: center; gap: 6px; cursor: pointer; user-select: none; font-size: 12px; color: var(--text-secondary); }
.legend-item.off { opacity: 0.35; }
.legend-item svg { flex: none; }
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }
.chart-title { font-size: 12px; font-weight: 600; color: var(--text-primary); margin: 0 0 4px; text-transform: uppercase; letter-spacing: 0.02em; }
.chart-wrap { position: relative; }
svg.chart { width: 100%; height: 220px; display: block; overflow: visible; }
.tick text { fill: var(--text-muted); font-size: 10px; }
.gridline { stroke: var(--gridline); stroke-width: 1; }
.axis-line { stroke: var(--axis); stroke-width: 1; }
.crosshair { stroke: var(--axis); stroke-width: 1; pointer-events: none; opacity: 0; }
.tooltip {
  position: absolute; pointer-events: none; background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 8px; padding: 8px 10px; font-size: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.15);
  opacity: 0; transform: translate(-50%, -100%); white-space: nowrap; z-index: 10;
}
.tooltip .row { display: flex; align-items: center; gap: 6px; }
.tooltip .row + .row { margin-top: 2px; }
.tooltip .val { font-weight: 600; font-variant-numeric: tabular-nums; }
.tooltip .name { color: var(--text-secondary); }
.tooltip .hdr { color: var(--text-muted); font-size: 11px; margin-bottom: 4px; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: 12px; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--gridline); white-space: nowrap; }
th { color: var(--text-muted); font-weight: 600; cursor: pointer; position: sticky; top: 0; background: var(--surface-1); }
th:hover { color: var(--text-primary); }
tbody tr:hover { background: var(--gridline); }
.table-wrap { max-width: 100%; overflow-x: auto; max-height: 480px; overflow-y: auto; }
.empty { color: var(--text-muted); padding: 24px; text-align: center; }
</style>
</head>
<body>
<h1>Metrics report</h1>
<p class="subtitle" id="subtitle"></p>

<div class="card filters">
  <div><label for="split-sel">Split</label><select id="split-sel"></select></div>
  <div><label for="branch-sel">Branch</label><select id="branch-sel"></select></div>
  <div style="flex:1"></div>
  <div><input type="text" id="search" placeholder="Filtra config…"></div>
  <div><button id="export-btn" type="button">Esporta CSV</button></div>
</div>

<div class="card">
  <div class="legend" id="legend"></div>
  <div class="charts" id="charts"></div>
</div>

<div class="card">
  <div class="table-wrap" id="table-wrap"></div>
</div>

<div class="tooltip" id="tooltip"></div>

<script>
const DATA = __DATA_JSON__;
const METRICS_ORDER = __METRICS_JSON__;
const METRIC_LABEL = {combined_loss: "combined_loss (norm. space)", mse: "MSE (0-1 space)", psnr: "PSNR (dB)", ssim: "SSIM"};

function seriesColorAndDash(index) {
  const color = `var(--series-${(index % 8) + 1})`;
  const dashes = [null, "6,3", "2,2", "8,2,2,2"];
  const dash = dashes[Math.floor(index / 8) % dashes.length];
  return {color, dash};
}

const allConfigs = [...new Set(DATA.map(r => r.config))].sort();
const configIndex = new Map(allConfigs.map((c, i) => [c, i]));
const hiddenConfigs = new Set();
let searchTerm = "";

const splitOrder = ["train", "val", "test"];
const availSplits = splitOrder.filter(s => DATA.some(r => r.split === s));
const splitSel = document.getElementById("split-sel");
availSplits.forEach(s => { const o = document.createElement("option"); o.value = s; o.textContent = s; splitSel.appendChild(o); });
if (availSplits.includes("test")) splitSel.value = "test";

const branchOrder = ["pred", "predm"];
const availBranches = branchOrder.filter(b => DATA.some(r => r.branch === b));
const branchSel = document.getElementById("branch-sel");
availBranches.forEach(b => { const o = document.createElement("option"); o.value = b; o.textContent = b; branchSel.appendChild(o); });

document.getElementById("subtitle").textContent =
  `${allConfigs.length} config, ${DATA.length} righe (branch x metric x split) — ` + REPORT_META;

function currentFilter() {
  return { split: splitSel.value, branch: branchSel.value };
}

function fmtNum(x, digits) {
  if (x === null || x === undefined || Number.isNaN(x)) return "–";
  return x.toFixed(digits);
}

function digitsForMetric(metric) {
  return metric === "psnr" ? 2 : (metric === "mse" ? 4 : 3);
}

function renderLegend() {
  const el = document.getElementById("legend");
  el.textContent = "";
  const visibleConfigs = allConfigs.filter(c => !searchTerm || c.toLowerCase().includes(searchTerm));
  visibleConfigs.forEach(c => {
    const idx = configIndex.get(c);
    const {color, dash} = seriesColorAndDash(idx);
    const item = document.createElement("div");
    item.className = "legend-item" + (hiddenConfigs.has(c) ? " off" : "");
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("width", "20"); svg.setAttribute("height", "10");
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", "0"); line.setAttribute("y1", "5"); line.setAttribute("x2", "20"); line.setAttribute("y2", "5");
    line.setAttribute("stroke", color); line.setAttribute("stroke-width", "2");
    if (dash) line.setAttribute("stroke-dasharray", dash);
    svg.appendChild(line);
    const label = document.createElement("span");
    label.textContent = c.replace(/_best_model$/, "");
    item.appendChild(svg);
    item.appendChild(label);
    item.addEventListener("click", () => {
      if (hiddenConfigs.has(c)) hiddenConfigs.delete(c); else hiddenConfigs.add(c);
      renderAll();
    });
    el.appendChild(item);
  });
}

function niceTicks(min, max, count) {
  if (min === max) { min -= 1; max += 1; }
  const span = max - min;
  const pad = span * 0.08;
  min -= pad; max += pad;
  const ticks = [];
  for (let i = 0; i <= count; i++) ticks.push(min + (span + 2 * pad) * i / count);
  return { min, max, ticks };
}

function buildChart(metric, rows) {
  const width = 520, height = 220, padL = 46, padR = 12, padT = 10, padB = 22;
  const innerW = width - padL - padR, innerH = height - padT - padB;

  const wrap = document.createElement("div");
  wrap.className = "chart-wrap";
  const title = document.createElement("p");
  title.className = "chart-title";
  title.textContent = METRIC_LABEL[metric] || metric;
  wrap.appendChild(title);

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "chart");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("preserveAspectRatio", "none");

  if (rows.length === 0) {
    wrap.appendChild(svg);
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "Nessun dato per questa combinazione.";
    wrap.appendChild(empty);
    return wrap;
  }

  const tLabels = rows[0].t_labels;
  let vmin = Infinity, vmax = -Infinity;
  rows.forEach(r => r.t.forEach((v, i) => {
    if (v === null) return;
    const sd = r.std_t[i] || 0;
    vmin = Math.min(vmin, v - sd);
    vmax = Math.max(vmax, v + sd);
  }));
  if (!Number.isFinite(vmin) || !Number.isFinite(vmax)) { vmin = 0; vmax = 1; }
  const { min, max, ticks } = niceTicks(vmin, vmax, 4);

  const x = i => padL + innerW * (i / Math.max(1, tLabels.length - 1));
  const y = v => padT + innerH * (1 - (v - min) / (max - min));

  const gGrid = document.createElementNS("http://www.w3.org/2000/svg", "g");
  ticks.forEach(t => {
    const ln = document.createElementNS("http://www.w3.org/2000/svg", "line");
    ln.setAttribute("class", "gridline");
    ln.setAttribute("x1", padL); ln.setAttribute("x2", width - padR);
    ln.setAttribute("y1", y(t)); ln.setAttribute("y2", y(t));
    gGrid.appendChild(ln);
    const txt = document.createElementNS("http://www.w3.org/2000/svg", "text");
    txt.setAttribute("class", "tick"); txt.setAttribute("x", padL - 6); txt.setAttribute("y", y(t) + 3);
    txt.setAttribute("text-anchor", "end");
    txt.textContent = fmtNum(t, digitsForMetric(metric));
    gGrid.appendChild(txt);
  });
  svg.appendChild(gGrid);

  const axisX = document.createElementNS("http://www.w3.org/2000/svg", "line");
  axisX.setAttribute("class", "axis-line");
  axisX.setAttribute("x1", padL); axisX.setAttribute("x2", width - padR);
  axisX.setAttribute("y1", height - padB); axisX.setAttribute("y2", height - padB);
  svg.appendChild(axisX);

  tLabels.forEach((lbl, i) => {
    const txt = document.createElementNS("http://www.w3.org/2000/svg", "text");
    txt.setAttribute("class", "tick"); txt.setAttribute("x", x(i)); txt.setAttribute("y", height - padB + 14);
    txt.setAttribute("text-anchor", "middle");
    txt.textContent = lbl;
    svg.appendChild(txt);
  });

  rows.forEach(r => {
    const idx = configIndex.get(r.config);
    const { color, dash } = seriesColorAndDash(idx);
    const validIdx = r.t.map((v, i) => i).filter(i => r.t[i] !== null && r.t[i] !== undefined);
    if (validIdx.length === 0) return;

    if (validIdx.some(i => (r.std_t[i] || 0) > 0)) {
      const upper = validIdx.map(i => [x(i), y(r.t[i] + (r.std_t[i] || 0))]);
      const lower = [...validIdx].reverse().map(i => [x(i), y(r.t[i] - (r.std_t[i] || 0))]);
      const pts = [...upper, ...lower].map(p => p.join(",")).join(" ");
      const area = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
      area.setAttribute("points", pts);
      area.setAttribute("fill", color);
      area.setAttribute("opacity", "0.10");
      svg.appendChild(area);
    }

    const path = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    path.setAttribute("points", validIdx.map(i => `${x(i)},${y(r.t[i])}`).join(" "));
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", color);
    path.setAttribute("stroke-width", "2");
    path.setAttribute("stroke-linejoin", "round");
    path.setAttribute("stroke-linecap", "round");
    if (dash) path.setAttribute("stroke-dasharray", dash);
    svg.appendChild(path);

    validIdx.forEach(i => {
      const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      dot.setAttribute("cx", x(i)); dot.setAttribute("cy", y(r.t[i])); dot.setAttribute("r", "4");
      dot.setAttribute("fill", color);
      dot.setAttribute("stroke", "var(--surface-1)"); dot.setAttribute("stroke-width", "2");
      svg.appendChild(dot);
    });
  });

  const crosshair = document.createElementNS("http://www.w3.org/2000/svg", "line");
  crosshair.setAttribute("class", "crosshair");
  crosshair.setAttribute("y1", padT); crosshair.setAttribute("y2", height - padB);
  svg.appendChild(crosshair);

  const hitRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  hitRect.setAttribute("x", padL); hitRect.setAttribute("y", padT);
  hitRect.setAttribute("width", innerW); hitRect.setAttribute("height", innerH);
  hitRect.setAttribute("fill", "transparent");
  svg.appendChild(hitRect);

  const tooltip = document.getElementById("tooltip");
  hitRect.addEventListener("mousemove", ev => {
    const rect = svg.getBoundingClientRect();
    const relX = (ev.clientX - rect.left) / rect.width * width;
    let i = Math.round((relX - padL) / innerW * (tLabels.length - 1));
    i = Math.max(0, Math.min(tLabels.length - 1, i));
    crosshair.setAttribute("x1", x(i)); crosshair.setAttribute("x2", x(i));
    crosshair.style.opacity = "1";

    tooltip.textContent = "";
    const hdr = document.createElement("div");
    hdr.className = "hdr";
    hdr.textContent = `${METRIC_LABEL[metric] || metric} · ${tLabels[i]}`;
    tooltip.appendChild(hdr);
    rows.forEach(r => {
      const row = document.createElement("div");
      row.className = "row";
      const key = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      key.setAttribute("width", "14"); key.setAttribute("height", "8");
      const ln = document.createElementNS("http://www.w3.org/2000/svg", "line");
      const { color, dash } = seriesColorAndDash(configIndex.get(r.config));
      ln.setAttribute("x1", "0"); ln.setAttribute("y1", "4"); ln.setAttribute("x2", "14"); ln.setAttribute("y2", "4");
      ln.setAttribute("stroke", color); ln.setAttribute("stroke-width", "2");
      if (dash) ln.setAttribute("stroke-dasharray", dash);
      key.appendChild(ln);
      const val = document.createElement("span");
      val.className = "val";
      const sd = r.std_t[i] || 0;
      val.textContent = fmtNum(r.t[i], digitsForMetric(metric)) + (sd > 0 ? ` ± ${fmtNum(sd, digitsForMetric(metric))}` : "");
      const name = document.createElement("span");
      name.className = "name";
      name.textContent = r.config.replace(/_best_model$/, "");
      row.appendChild(key); row.appendChild(val); row.appendChild(name);
      tooltip.appendChild(row);
    });
    tooltip.style.left = (rect.left + x(i) * rect.width / width + window.scrollX) + "px";
    tooltip.style.top = (rect.top + window.scrollY - 8) + "px";
    tooltip.style.opacity = "1";
  });
  hitRect.addEventListener("mouseleave", () => {
    crosshair.style.opacity = "0";
    tooltip.style.opacity = "0";
  });

  wrap.appendChild(svg);
  return wrap;
}

function renderCharts() {
  const { split, branch } = currentFilter();
  const container = document.getElementById("charts");
  container.textContent = "";
  METRICS_ORDER.forEach(metric => {
    const rows = DATA.filter(r => r.split === split && r.branch === branch && r.metric === metric
      && !hiddenConfigs.has(r.config) && (!searchTerm || r.config.toLowerCase().includes(searchTerm)))
      .sort((a, b) => configIndex.get(a.config) - configIndex.get(b.config));
    container.appendChild(buildChart(metric, rows));
  });
}

let lastTableRows = [];

function renderTable() {
  const { split, branch } = currentFilter();
  const rows = DATA.filter(r => r.split === split && r.branch === branch
    && (!searchTerm || r.config.toLowerCase().includes(searchTerm)));
  const wrap = document.getElementById("table-wrap");
  wrap.textContent = "";
  document.getElementById("export-btn").disabled = rows.length === 0;
  if (rows.length === 0) {
    lastTableRows = [];
    const p = document.createElement("p"); p.className = "empty"; p.textContent = "Nessuna riga.";
    wrap.appendChild(p); return;
  }
  const tLabels = rows[0].t_labels;
  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const trh = document.createElement("tr");
  const cols = ["config", "metric", ...tLabels, "mean", "n"];
  let sortCol = 0, sortDir = 1;
  cols.forEach((c, ci) => {
    const th = document.createElement("th");
    th.textContent = c;
    th.addEventListener("click", () => {
      sortDir = (sortCol === ci) ? -sortDir : 1;
      sortCol = ci;
      draw();
    });
    trh.appendChild(th);
  });
  thead.appendChild(trh);
  table.appendChild(thead);
  const tbody = document.createElement("tbody");
  table.appendChild(tbody);
  wrap.appendChild(table);

  function draw() {
    const sorted = [...rows].sort((a, b) => {
      const av = sortCol === 0 ? a.config : sortCol === 1 ? a.metric : sortCol === cols.length - 1 ? a.n_seeds : (sortCol === cols.length - 2 ? a.mean : a.t[sortCol - 2]);
      const bv = sortCol === 0 ? b.config : sortCol === 1 ? b.metric : sortCol === cols.length - 1 ? b.n_seeds : (sortCol === cols.length - 2 ? b.mean : b.t[sortCol - 2]);
      if (av === bv) return 0;
      if (typeof av === "string") return av.localeCompare(bv) * sortDir;
      return (av - bv) * sortDir;
    });
    lastTableRows = sorted;
    tbody.textContent = "";
    sorted.forEach(r => {
      const tr = document.createElement("tr");
      const digits = digitsForMetric(r.metric);
      const cells = [
        r.config.replace(/_best_model$/, ""),
        r.metric,
        ...r.t.map((v, i) => fmtNum(v, digits) + ((r.std_t[i] || 0) > 0 ? ` ± ${fmtNum(r.std_t[i], digits)}` : "")),
        fmtNum(r.mean, digits) + (r.std_mean > 0 ? ` ± ${fmtNum(r.std_mean, digits)}` : ""),
        r.n_seeds ?? "–",
      ];
      cells.forEach(v => { const td = document.createElement("td"); td.textContent = v; tr.appendChild(td); });
      tbody.appendChild(tr);
    });
  }
  draw();
}

function renderAll() {
  renderLegend();
  renderCharts();
  renderTable();
}

function csvEscape(v) {
  const s = v === null || v === undefined ? "" : String(v);
  return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}

function exportCsv() {
  if (lastTableRows.length === 0) return;
  const tLabels = lastTableRows[0].t_labels;
  const header = ["config", "split", "branch", "metric",
    ...tLabels, ...tLabels.map(l => "std_" + l), "mean", "std_mean", "n_seeds"];
  const lines = [header.map(csvEscape).join(",")];
  lastTableRows.forEach(r => {
    const line = [r.config, r.split, r.branch, r.metric, ...r.t, ...r.std_t, r.mean, r.std_mean, r.n_seeds];
    lines.push(line.map(csvEscape).join(","));
  });
  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const { split, branch } = currentFilter();
  const a = document.createElement("a");
  a.href = url;
  a.download = `metrics_${split}_${branch}${searchTerm ? "_" + searchTerm : ""}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

splitSel.addEventListener("change", renderAll);
branchSel.addEventListener("change", renderAll);
document.getElementById("search").addEventListener("input", ev => {
  searchTerm = ev.target.value.trim().toLowerCase();
  renderAll();
});
document.getElementById("export-btn").addEventListener("click", exportCsv);

renderAll();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=str, default="Model_Results", help="Cartella radice con i risultati (default: Model_Results).")
    parser.add_argument("--out", type=str, default=None, help="Path del report HTML (default: <root>/metrics_report.html).")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Cartella non trovata: {root}")
    out_path = Path(args.out) if args.out else root / "metrics_report.html"

    rows, configs = _collect_rows(root)
    if not rows:
        raise SystemExit(f"Nessun metrics.csv / seeds_summary_*.csv trovato sotto {root}")

    html = HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(rows))
    html = html.replace("__METRICS_JSON__", json.dumps(METRICS_ORDER))
    meta = f"generato il {datetime.now().strftime('%Y-%m-%d %H:%M')} da scripts/visualize_metrics.py --root {root}"
    html = html.replace("REPORT_META;", f"{meta!r};")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"{len(configs)} config, {len(rows)} righe -> {out_path}")


if __name__ == "__main__":
    main()
