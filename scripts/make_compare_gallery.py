"""Gallery HTML per confrontare piu' run di inferenza sugli STESSI sample.

Uso:
    python scripts/make_compare_gallery.py --base Model_Results/test_2025 \
        --runs flip07_remap flip07_clamp01 legacy07_remap legacy07_clamp01

Ogni run deve avere la struttura prodotta da test_model_create_gifs_3ch:
    <base>/<run>/by_class/<class>/<event>/<sample>/{compare.gif, input_t*, target_t*, pred_*, predm_*}

I sample vengono appaiati fra run per chiave (event, sample_name). Scrive
`<base>/compare.html`, self-contained, con path RELATIVI a `<base>` (non copia
le immagini): aprire con doppio click tenendo intatta la cartella dei risultati.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def _t_index(name: str) -> int:
    m = re.match(r"\w+_t(\d+)_", name)
    return int(m.group(1)) if m else 0


def _frames(sample_dir: Path, prefix_re: str) -> List[str]:
    pat = re.compile(prefix_re)
    names = [p.name for p in sample_dir.iterdir()
             if p.is_file() and p.suffix.lower() == ".jpg" and pat.match(p.name)]
    names.sort(key=lambda n: (_t_index(n), n))
    return names


def _pred_frames(sample_dir: Path, prefix: str, exclude: Optional[str] = None) -> List[str]:
    names = [p.name for p in sample_dir.iterdir()
             if p.is_file() and p.suffix.lower() == ".jpg" and p.name.startswith(prefix)
             and (exclude is None or not p.name.startswith(exclude))]
    return sorted(names)


def _load_metrics(run_dir: Path) -> Dict[str, Dict[str, float]]:
    """{'pred': {...mean row...}, 'predm': {...}} dai metrics.csv sotto by_class."""
    out: Dict[str, Dict[str, float]] = {}
    for csv_path in run_dir.rglob("metrics.csv"):
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("lag") == "mean":
                        pt = row.get("prediction_type", "")
                        out[pt] = {
                            k: float(row[k]) for k in ("mse", "mse_lpips", "ssim", "f1")
                            if row.get(k) not in (None, "")
                        }
        except Exception:
            pass
        break  # un solo metrics.csv per run (by_class/class_*/)
    return out


def collect(base: Path, runs: List[str]) -> Dict[str, Any]:
    # events -> sample_name -> { run -> payload }
    tree: Dict[str, Dict[str, Dict[str, Any]]] = {}
    run_meta: Dict[str, Any] = {}

    for run in runs:
        run_dir = base / run
        run_meta[run] = {"metrics": _load_metrics(run_dir)}
        for gif_path in sorted(run_dir.rglob("compare.gif")):
            sd = gif_path.parent
            parts = sd.relative_to(run_dir).parts  # (by_class, class_x, event, sample)
            event = parts[-2] if len(parts) >= 2 else "unknown"
            sample = parts[-1]
            rel = sd.relative_to(base).as_posix()

            def paths(names: List[str]) -> List[str]:
                return [f"{rel}/{n}" for n in names]

            tree.setdefault(event, {}).setdefault(sample, {})[run] = {
                "gif": f"{rel}/compare.gif",
                "inputs": paths(_frames(sd, r"input_t(\d+)_")),
                "targets": paths(_frames(sd, r"target_t(\d+)_")),
                "preds": paths(_pred_frames(sd, "pred_", exclude="predm_")),
                "predms": paths(_pred_frames(sd, "predm_")),
            }

    def _sample_key(name: str):
        m = re.findall(r"\d+", name)
        return tuple(int(x) for x in m) if m else (name,)

    events = []
    for ev in sorted(tree):
        samples = [{"name": s, "runs": tree[ev][s]} for s in sorted(tree[ev], key=_sample_key)]
        events.append({"name": ev, "samples": samples})

    return {"runs": runs, "run_meta": run_meta, "events": events}


HTML = r"""<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<title>Model Compare — __BASE_NAME__</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root { --bg:#0b0d12; --panel:#12151c; --panel2:#171b24; --panel3:#1d2230; --border:#262b36;
  --text:#e6e9ef; --muted:#8b93a3; --accent:#6ea8ff; --accent-fg:#0b0d12; }
@media (prefers-color-scheme: light) {
  :root { --bg:#f5f6f8; --panel:#fff; --panel2:#eef0f4; --panel3:#e4e8ef; --border:#dde1e8;
    --text:#1a1d24; --muted:#5b6272; --accent:#2563eb; --accent-fg:#fff; } }
* { box-sizing:border-box; }
html,body { height:100%; margin:0; }
body { background:var(--bg); color:var(--text); font-family:-apple-system,"Segoe UI",Roboto,sans-serif; display:flex; flex-direction:column; }
header { background:var(--panel); border-bottom:1px solid var(--border); padding:10px 16px; display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
header h1 { font-size:15px; margin:0; font-weight:600; white-space:nowrap; }
header .meta { color:var(--muted); font-size:12px; }
#search { flex:1; min-width:160px; background:var(--panel2); border:1px solid var(--border); color:var(--text); border-radius:8px; padding:7px 12px; font-size:13px; }
.toggle { border:1px solid var(--border); background:var(--panel2); color:var(--muted); border-radius:999px; padding:5px 12px; font-size:12px; cursor:pointer; user-select:none; }
.toggle.active { background:var(--accent); color:var(--accent-fg); border-color:var(--accent); }
#app { flex:1; display:flex; min-height:0; }
aside { width:250px; flex:none; border-right:1px solid var(--border); background:var(--panel); overflow-y:auto; padding:8px; }
.event-group summary { cursor:pointer; list-style:none; padding:6px 8px; border-radius:6px; font-size:12.5px; font-weight:600; display:flex; justify-content:space-between; }
.event-group summary::-webkit-details-marker { display:none; }
.event-group summary:hover { background:var(--panel2); }
.event-group summary .n { color:var(--muted); font-weight:400; }
.event-group[open] summary { color:var(--accent); }
.sample-list { list-style:none; margin:0; padding:0 0 4px 10px; }
.sample-item { padding:5px 8px; border-radius:6px; font-size:11.5px; color:var(--muted); cursor:pointer; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.sample-item:hover { background:var(--panel2); color:var(--text); }
.sample-item.selected { background:var(--accent); color:var(--accent-fg); font-weight:600; }
.hidden { display:none !important; }
main { flex:1; overflow-y:auto; padding:16px 22px 60px; }
.breadcrumb { color:var(--muted); font-size:12px; margin-bottom:10px; }
.breadcrumb b { color:var(--text); }
.nav-bar { display:flex; gap:8px; align-items:center; margin-bottom:16px; padding:8px 12px; background:var(--panel); border:1px solid var(--border); border-radius:10px; }
.nav-btn { border:1px solid var(--border); background:var(--panel2); color:var(--text); border-radius:8px; padding:6px 12px; font-size:13px; cursor:pointer; }
.nav-btn:hover:not(:disabled) { border-color:var(--accent); color:var(--accent); }
.nav-btn:disabled { opacity:.35; cursor:default; }
.nav-pos { font-size:12px; color:var(--muted); min-width:64px; text-align:center; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:14px; margin-bottom:22px; }
.card { background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:12px; }
.card h3 { margin:0 0 8px; font-size:13px; display:flex; justify-content:space-between; align-items:baseline; gap:8px; }
.card h3 .tag { font-size:10px; color:var(--muted); font-weight:400; }
.card img.gif { width:100%; border-radius:8px; border:1px solid var(--border); background:var(--panel3); }
.mrow { font-size:11px; color:var(--muted); margin-top:8px; display:flex; flex-wrap:wrap; gap:10px; }
.mrow b { color:var(--text); font-weight:600; }
.section-title { font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); margin:20px 0 10px; }
table.frames { border-collapse:collapse; width:100%; }
table.frames td, table.frames th { border:1px solid var(--border); padding:4px; text-align:center; vertical-align:middle; }
table.frames th { font-size:11px; color:var(--muted); font-weight:600; background:var(--panel); position:sticky; left:0; }
table.frames img { width:104px; height:104px; object-fit:cover; border-radius:6px; display:block; margin:0 auto; background:var(--panel3); }
table.frames .rowlab { font-size:11px; color:var(--muted); white-space:nowrap; }
table.frames tr:has(.is-target) { background:var(--panel2); }
table.frames .is-target { color:var(--text); font-weight:700; }
.wrap-x { overflow-x:auto; }
.diffbadge { font-size:10px; }
</style>
</head>
<body>
<header>
  <h1>🌋 Model Compare</h1>
  <span class="meta">__BASE_NAME__ · __EVENT_COUNT__ eventi · __RUN_COUNT__ run · __GENERATED_AT__</span>
  <input id="search" type="text" placeholder="Filtra sample...">
  <span class="toggle active" data-kind="pred">pred</span>
  <span class="toggle" data-kind="predm">pred mod</span>
</header>
<div id="app">
  <aside id="sidebar"></aside>
  <main id="viewer"></main>
</div>
<script>
const DATA = __DATA_JSON__;
const RUNS = DATA.runs;
const RUN_META = DATA.run_meta;
let sel = { event:0, sample:0 };
let kind = "pred"; // pred | predm

const sidebar = document.getElementById("sidebar");
const viewer = document.getElementById("viewer");
const searchInput = document.getElementById("search");

function renderSidebar() {
  sidebar.innerHTML = "";
  DATA.events.forEach((ev, ei) => {
    const det = document.createElement("details");
    det.className = "event-group";
    det.open = (ei === sel.event);
    det.dataset.event = ev.name.toLowerCase();
    const sm = document.createElement("summary");
    sm.innerHTML = `<span>${ev.name}</span><span class="n">${ev.samples.length}</span>`;
    det.appendChild(sm);
    const ul = document.createElement("ul");
    ul.className = "sample-list";
    ev.samples.forEach((s, si) => {
      const li = document.createElement("li");
      li.className = "sample-item";
      li.dataset.name = s.name.toLowerCase();
      li.textContent = s.name.replace(/^sample_/, "");
      if (ei === sel.event && si === sel.sample) li.classList.add("selected");
      li.addEventListener("click", () => { sel = {event:ei, sample:si}; renderSidebar(); renderViewer(); });
      ul.appendChild(li);
    });
    det.appendChild(ul);
    sidebar.appendChild(det);
  });
  applyFilter();
}

function applyFilter() {
  const q = searchInput.value.trim().toLowerCase();
  document.querySelectorAll(".event-group").forEach(evEl => {
    let any = false;
    evEl.querySelectorAll(".sample-item").forEach(li => {
      const show = !q || li.dataset.name.includes(q) || evEl.dataset.event.includes(q);
      li.classList.toggle("hidden", !show);
      if (show) any = true;
    });
    evEl.classList.toggle("hidden", !any);
    if (any && q) evEl.open = true;
  });
}
searchInput.addEventListener("input", applyFilter);

document.querySelectorAll(".toggle").forEach(t => t.addEventListener("click", () => {
  document.querySelectorAll(".toggle").forEach(x => x.classList.remove("active"));
  t.classList.add("active");
  kind = t.dataset.kind;
  renderViewer();
}));

function fmt(x) { return (x === undefined || x === null) ? "–" : x.toFixed(4); }

function metricsRow(run) {
  const m = (RUN_META[run] && RUN_META[run].metrics && RUN_META[run].metrics[kind]) || null;
  if (!m) return "";
  return `<div class="mrow">
    <span>MSE <b>${fmt(m.mse)}</b></span>
    <span>MSE+LPIPS <b>${fmt(m.mse_lpips)}</b></span>
    <span>SSIM <b>${fmt(m.ssim)}</b></span>
    <span>F1 <b>${fmt(m.f1)}</b></span></div>`;
}

function renderViewer() {
  const ev = DATA.events[sel.event];
  const s = ev.samples[sel.sample];
  const atFirst = sel.sample === 0, atLast = sel.sample === ev.samples.length - 1;
  const anyRun = RUNS.find(r => s.runs[r]);
  const inputs = anyRun ? s.runs[anyRun].inputs : [];

  // una riga target per ogni modalità di denorm presente (suffisso del nome run: _remap / _clamp01)
  const modeOf = r => r.split("_").pop();
  const modes = [...new Set(RUNS.map(modeOf))];
  const targetsByMode = {};
  modes.forEach(m => {
    const rr = RUNS.find(r => modeOf(r) === m && s.runs[r]);
    targetsByMode[m] = rr ? s.runs[rr].targets : [];
  });
  const T = (targetsByMode[modes[0]] || []).length;

  const cards = RUNS.map(r => {
    const d = s.runs[r];
    if (!d) return `<div class="card"><h3>${r}</h3><div class="mrow">— nessun sample —</div></div>`;
    return `<div class="card">
      <h3><span>${r}</span><span class="tag">${kind}</span></h3>
      <img class="gif" src="${d.gif}" alt="${r}">
      ${metricsRow(r)}
    </div>`;
  }).join("");

  // tabella frame-by-frame: righe = [input(solo t finali)/target, poi un rigo per run], colonne = t0..t3
  const headCols = Array.from({length:T}, (_,t)=>`<th>t${t}</th>`).join("");
  function imgRow(label, paths, cls) {
    const tds = Array.from({length:T}, (_,t) => {
      const p = paths && paths[t];
      return `<td>${p ? `<a href="${p}" target="_blank"><img src="${p}" loading="lazy"></a>` : "–"}</td>`;
    }).join("");
    return `<tr><td class="rowlab ${cls||""}">${label}</td>${tds}</tr>`;
  }
  let rows = "";
  modes.forEach(m => { rows += imgRow("target · " + m, targetsByMode[m], "is-target"); });
  RUNS.forEach(r => {
    const d = s.runs[r];
    rows += imgRow(r + " · " + modeOf(r), d ? (kind === "pred" ? d.preds : d.predms) : null, "is-" + modeOf(r));
  });

  const inStrip = inputs.map((p,i)=>`<td><a href="${p}" target="_blank"><img src="${p}" loading="lazy"></a><div class="rowlab">in t${i}</div></td>`).join("");

  viewer.innerHTML = `
    <div class="breadcrumb"><b>${ev.name}</b> / ${s.name}</div>
    <div class="nav-bar">
      <button class="nav-btn" id="prev" ${atFirst?"disabled":""}>← precedente</button>
      <span class="nav-pos">${sel.sample+1} / ${ev.samples.length}</span>
      <button class="nav-btn" id="next" ${atLast?"disabled":""}>successivo →</button>
      <span class="meta" style="margin-left:8px">frecce ← → per navigare</span>
    </div>
    <div class="section-title">Input (4 frame condivisi)</div>
    <div class="wrap-x"><table class="frames"><tr>${inStrip}</tr></table></div>
    <div class="section-title">Confronto animato — ${RUNS.length} run affiancati</div>
    <div class="grid">${cards}</div>
    <div class="section-title">Frame-by-frame · ${kind} vs target</div>
    <div class="wrap-x"><table class="frames"><thead><tr><th>&nbsp;</th>${headCols}</tr></thead><tbody>${rows}</tbody></table></div>
  `;
  document.getElementById("prev")?.addEventListener("click", () => { if(!atFirst){ sel.sample--; renderSidebar(); renderViewer(); }});
  document.getElementById("next")?.addEventListener("click", () => { if(!atLast){ sel.sample++; renderSidebar(); renderViewer(); }});
}

document.addEventListener("keydown", e => {
  if (document.activeElement === searchInput) return;
  if (e.key === "ArrowLeft") document.getElementById("prev")?.click();
  if (e.key === "ArrowRight") document.getElementById("next")?.click();
});

renderSidebar();
renderViewer();
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Gallery HTML di confronto fra piu' run di inferenza.")
    ap.add_argument("--base", "-b", required=True, help="Cartella che contiene le sottocartelle run.")
    ap.add_argument("--runs", "-r", nargs="+", required=True, help="Nomi delle sottocartelle run da confrontare.")
    ap.add_argument("--out", default=None, help="File output (default: <base>/compare.html).")
    args = ap.parse_args()

    base = Path(args.base).expanduser().resolve()
    if not base.is_dir():
        raise SystemExit(f"Cartella non trovata: {base}")
    missing = [r for r in args.runs if not (base / r).is_dir()]
    if missing:
        raise SystemExit(f"Run mancanti sotto {base}: {missing}")

    data = collect(base, args.runs)
    n_ev = len(data["events"])
    n_samples = sum(len(e["samples"]) for e in data["events"])

    from datetime import datetime
    html = (HTML
            .replace("__BASE_NAME__", base.name)
            .replace("__EVENT_COUNT__", str(n_ev))
            .replace("__RUN_COUNT__", str(len(args.runs)))
            .replace("__GENERATED_AT__", datetime.now().strftime("%Y-%m-%d %H:%M"))
            .replace("__DATA_JSON__", json.dumps(data, ensure_ascii=False)))

    out_path = Path(args.out).expanduser().resolve() if args.out else (base / "compare.html")
    out_path.write_text(html, encoding="utf-8")
    print(f"[compare] {len(args.runs)} run, {n_ev} eventi, {n_samples} sample-slot")
    print(f"[compare] Scritto: {out_path}")


if __name__ == "__main__":
    main()
