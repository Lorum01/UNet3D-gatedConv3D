"""Genera una gallery HTML statica e autonoma per esplorare i risultati di un'inferenza.

Uso:
    python scripts/make_gallery.py --root Model_Results/Model_Results_flip

Scansiona ricorsivamente `--root` cercando le cartelle sample (identificate dalla
presenza di `compare.gif`, prodotto da test_model_create_gifs_3ch), le raggruppa
per split (test/val/train) ed evento (cartella data, es. "2011_04_10"), e scrive
`gallery.html` dentro `--root`. Il file usa solo path RELATIVI alle immagini gia'
su disco (non le copia): va aperto con un doppio click nel browser tenendo la
cartella dei risultati intatta.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


_SPLIT_ORDER = {"test": 0, "val": 1, "train": 2}


def _t_index(prefix_re: str, name: str) -> int:
    m = re.match(prefix_re, name)
    return int(m.group(1)) if m else 0


def _frames(sample_dir: Path, prefix_re: str) -> List[str]:
    pat = re.compile(prefix_re)
    names = [p.name for p in sample_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jpg" and pat.match(p.name)]
    names.sort(key=lambda n: (_t_index(prefix_re, n), n))
    return names


def _pred_frames(sample_dir: Path, prefix: str, exclude_prefix: str | None = None) -> List[str]:
    names = [
        p.name for p in sample_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".jpg" and p.name.startswith(prefix)
        and (exclude_prefix is None or not p.name.startswith(exclude_prefix))
    ]
    return sorted(names)


def find_samples(root: Path) -> List[Dict[str, Any]]:
    samples = []
    for gif_path in sorted(root.rglob("compare.gif")):
        sample_dir = gif_path.parent
        rel_parts = sample_dir.relative_to(root).parts

        if len(rel_parts) >= 3:
            split_label, event_name = rel_parts[0], rel_parts[1]
        elif len(rel_parts) == 2:
            split_label, event_name = rel_parts[0], "unknown_event"
        else:
            split_label, event_name = "root", "unknown_event"

        rel_dir = sample_dir.relative_to(root).as_posix()

        def _paths(names: List[str]) -> List[str]:
            return [f"{rel_dir}/{n}" for n in names]

        samples.append({
            "split": split_label,
            "event": event_name,
            "name": sample_dir.name,
            "gif": f"{rel_dir}/compare.gif",
            "inputs": _paths(_frames(sample_dir, r"input_t(\d+)_")),
            "targets": _paths(_frames(sample_dir, r"target_t(\d+)_")),
            "preds": _paths(_pred_frames(sample_dir, "pred_", exclude_prefix="predm_")),
            "predms": _paths(_pred_frames(sample_dir, "predm_")),
        })
    return samples


def _split_sort_key(name: str) -> tuple:
    return (_SPLIT_ORDER.get(name, 99), name)


def build_tree(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """[{name, samples:[...]}] per evento, dentro [{name, events:[...]}] per split, ordinato."""
    by_split: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for s in samples:
        by_split.setdefault(s["split"], {}).setdefault(s["event"], []).append(s)

    tree = []
    for split in sorted(by_split, key=_split_sort_key):
        events = []
        for event in sorted(by_split[split]):
            events.append({"name": event, "samples": by_split[split][event]})
        tree.append({"name": split, "events": events})
    return tree


HTML_TEMPLATE = r"""<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<title>Model Results Viewer — __ROOT_NAME__</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {
  --bg: #0b0d12; --panel: #12151c; --panel2: #171b24; --panel3: #1d2230; --border: #262b36;
  --text: #e6e9ef; --muted: #8b93a3; --accent: #6ea8ff; --accent-fg:#0b0d12;
}
@media (prefers-color-scheme: light) {
  :root { --bg:#f5f6f8; --panel:#ffffff; --panel2:#eef0f4; --panel3:#e4e8ef; --border:#dde1e8; --text:#1a1d24; --muted:#5b6272; --accent:#2563eb; --accent-fg:#ffffff; }
}
* { box-sizing: border-box; }
html, body { height:100%; margin:0; }
body { background:var(--bg); color:var(--text); font-family: -apple-system, "Segoe UI", Roboto, sans-serif; display:flex; flex-direction:column; }

header { background:var(--panel); border-bottom:1px solid var(--border); padding:10px 16px; display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
header h1 { font-size:15px; margin:0; font-weight:600; white-space:nowrap; }
header .meta { color:var(--muted); font-size:12px; white-space:nowrap; }
#search { flex:1; min-width:160px; background:var(--panel2); border:1px solid var(--border); color:var(--text); border-radius:8px; padding:7px 12px; font-size:13px; }
.chip { border:1px solid var(--border); background:var(--panel2); color:var(--muted); border-radius:999px; padding:5px 12px; font-size:12px; cursor:pointer; user-select:none; }
.chip.active { background:var(--accent); color:var(--accent-fg); border-color:var(--accent); }

#app { flex:1; display:flex; min-height:0; }

aside { width:290px; flex:none; border-right:1px solid var(--border); background:var(--panel); overflow-y:auto; padding:8px; }
.split-group { margin-bottom:6px; }
.split-title { font-size:11px; font-weight:700; letter-spacing:.04em; color:var(--muted); padding:8px 8px 4px; text-transform:uppercase; }
.event-group summary { cursor:pointer; list-style:none; padding:6px 8px; border-radius:6px; font-size:12.5px; font-weight:600; display:flex; justify-content:space-between; gap:6px; }
.event-group summary::-webkit-details-marker { display:none; }
.event-group summary:hover { background:var(--panel2); }
.event-group summary .n { color:var(--muted); font-weight:400; }
.event-group[open] summary { color:var(--accent); }
.sample-list { list-style:none; margin:0; padding:0 0 4px 14px; }
.sample-item { padding:5px 8px; border-radius:6px; font-size:12px; color:var(--muted); cursor:pointer; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.sample-item:hover { background:var(--panel2); color:var(--text); }
.sample-item.selected { background:var(--accent); color:var(--accent-fg); font-weight:600; }
.hidden { display:none !important; }
.no-match { padding:16px; color:var(--muted); font-size:12px; text-align:center; }

main { flex:1; overflow-y:auto; padding: 18px 26px 60px; }
.breadcrumb { color:var(--muted); font-size:12px; margin-bottom:10px; }
.breadcrumb b { color:var(--text); }

.nav-bar { display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin-bottom:18px; padding:10px 14px; background:var(--panel); border:1px solid var(--border); border-radius:12px; }
.nav-group { display:flex; align-items:center; gap:6px; padding-right:14px; border-right:1px solid var(--border); }
.nav-group:last-child { border-right:none; }
.nav-group .lbl { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; margin-right:4px; }
.nav-btn2 { border:1px solid var(--border); background:var(--panel2); color:var(--text); border-radius:8px; padding:6px 12px; font-size:13px; cursor:pointer; }
.nav-btn2:hover:not(:disabled) { border-color:var(--accent); color:var(--accent); }
.nav-btn2:disabled { opacity:.35; cursor:default; }
.nav-pos { font-size:12px; color:var(--muted); min-width:60px; text-align:center; }

.gif-panel { background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:16px; margin-bottom:20px; text-align:center; }
.gif-panel img { max-width:100%; border-radius:8px; }
.gif-panel .cap { color:var(--muted); font-size:11px; margin-top:8px; }

.frames-panel { background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:16px; }
.frame-row { margin-bottom:18px; }
.frame-row:last-child { margin-bottom:0; }
.frame-row h3 { font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); margin:0 0 10px; display:flex; align-items:center; gap:8px; }
.frame-row h3 .dot { width:8px; height:8px; border-radius:50%; display:inline-block; }
.dot.input { background:#6ea8ff; } .dot.target { background:#8bd17c; } .dot.pred { background:#ffb86e; } .dot.predm { background:#ff8ea3; }
.strip { display:grid; grid-auto-flow:column; grid-auto-columns:minmax(130px,1fr); gap:10px; overflow-x:auto; padding-bottom:4px; }
.frame-cell a { display:block; }
.frame-cell img { width:100%; aspect-ratio:1/1; object-fit:cover; border-radius:8px; border:1px solid var(--border); background:var(--panel3); }
.frame-cell .t { font-size:10px; color:var(--muted); text-align:center; margin-top:4px; }

.empty-state { padding:60px 20px; text-align:center; color:var(--muted); }
</style>
</head>
<body>
<header>
  <h1>🌋 Model Results Viewer</h1>
  <span class="meta">__ROOT_NAME__ · __SAMPLE_COUNT__ sample · generata __GENERATED_AT__</span>
  <input id="search" type="text" placeholder="Filtra per evento o nome sample...">
  <div id="splitChips"></div>
</header>

<div id="app">
  <aside id="sidebar"></aside>
  <main id="viewer"></main>
</div>

<script>
const TREE = __DATA_JSON__;
const splitOrder = { test: 0, val: 1, train: 2 };

let sel = { split: 0, event: 0, sample: 0 };
const activeSplits = new Set(TREE.map(s => s.name));

// ---------------------------- Sidebar ----------------------------
const sidebar = document.getElementById("sidebar");

function renderSidebar() {
  sidebar.innerHTML = "";
  TREE.forEach((split, si) => {
    const grp = document.createElement("div");
    grp.className = "split-group";
    grp.dataset.split = split.name;

    const title = document.createElement("div");
    title.className = "split-title";
    title.textContent = split.name;
    grp.appendChild(title);

    split.events.forEach((ev, ei) => {
      const det = document.createElement("details");
      det.className = "event-group";
      det.open = (si === sel.split && ei === sel.event);
      det.dataset.event = ev.name.toLowerCase();

      const summary = document.createElement("summary");
      summary.innerHTML = `<span>${ev.name}</span><span class="n">${ev.samples.length}</span>`;
      det.appendChild(summary);

      const ul = document.createElement("ul");
      ul.className = "sample-list";
      ev.samples.forEach((s, smi) => {
        const li = document.createElement("li");
        li.className = "sample-item";
        li.dataset.name = s.name.toLowerCase();
        li.textContent = s.name;
        if (si === sel.split && ei === sel.event && smi === sel.sample) li.classList.add("selected");
        li.addEventListener("click", () => selectSample(si, ei, smi));
        ul.appendChild(li);
      });
      det.appendChild(ul);
      grp.appendChild(det);
    });
    sidebar.appendChild(grp);
  });
  applyFilter();
}

// ---------------------------- Split chips ----------------------------
const chipsBox = document.getElementById("splitChips");
TREE.forEach(split => {
  const chip = document.createElement("span");
  chip.className = "chip active";
  chip.textContent = split.name;
  chip.addEventListener("click", () => {
    if (activeSplits.has(split.name)) { activeSplits.delete(split.name); chip.classList.remove("active"); }
    else { activeSplits.add(split.name); chip.classList.add("active"); }
    applyFilter();
  });
  chipsBox.appendChild(chip);
});

// ---------------------------- Filtro ----------------------------
const searchInput = document.getElementById("search");
searchInput.addEventListener("input", applyFilter);

function applyFilter() {
  const q = searchInput.value.trim().toLowerCase();
  let anyVisible = false;
  document.querySelectorAll(".split-group").forEach(grp => {
    const splitOn = activeSplits.has(grp.dataset.split);
    let anyEvent = false;
    grp.querySelectorAll(".event-group").forEach(evEl => {
      const evMatches = !q || evEl.dataset.event.includes(q);
      let anySample = false;
      evEl.querySelectorAll(".sample-item").forEach(li => {
        const show = splitOn && (evMatches || li.dataset.name.includes(q));
        li.classList.toggle("hidden", !show);
        if (show) anySample = true;
      });
      evEl.classList.toggle("hidden", !anySample);
      if (anySample) { anyEvent = true; if (q) evEl.open = true; }
    });
    grp.classList.toggle("hidden", !anyEvent);
    if (anyEvent) anyVisible = true;
  });
}

// ---------------------------- Viewer ----------------------------
const viewer = document.getElementById("viewer");

function currentSample() {
  return TREE[sel.split].events[sel.event].samples[sel.sample];
}

function selectSample(si, ei, smi) {
  sel = { split: si, event: ei, sample: smi };
  renderSidebar();
  renderViewer();
}

function rowHtml(title, dotClass, paths) {
  if (!paths.length) return "";
  const cells = paths.map((p, i) => {
    const base = p.split("/").pop().replace(/\.jpg$/i, "");
    return `<div class="frame-cell"><a href="${p}" target="_blank"><img src="${p}" loading="lazy"></a><div class="t">t${i} · ${base}</div></div>`;
  }).join("");
  return `<div class="frame-row"><h3><span class="dot ${dotClass}"></span>${title}</h3><div class="strip">${cells}</div></div>`;
}

function renderViewer() {
  if (!TREE.length) {
    viewer.innerHTML = `<div class="empty-state">Nessun sample trovato sotto la cartella dei risultati.</div>`;
    return;
  }
  const split = TREE[sel.split];
  const event = split.events[sel.event];
  const samples = event.samples;
  const s = samples[sel.sample];

  const atFirstSample = sel.sample === 0;
  const atLastSample = sel.sample === samples.length - 1;
  const atFirstEvent = sel.split === 0 && sel.event === 0;
  const atLastEvent = sel.split === TREE.length - 1 && sel.event === split.events.length - 1;

  viewer.innerHTML = `
    <div class="breadcrumb"><b>${split.name.toUpperCase()}</b> / <b>${event.name}</b> / ${s.name}</div>

    <div class="nav-bar">
      <div class="nav-group">
        <span class="lbl">Sample</span>
        <button class="nav-btn2" id="prevSample" ${atFirstSample ? "disabled" : ""}>← precedente</button>
        <span class="nav-pos">${sel.sample + 1} / ${samples.length}</span>
        <button class="nav-btn2" id="nextSample" ${atLastSample ? "disabled" : ""}>successivo →</button>
      </div>
      <div class="nav-group">
        <span class="lbl">Evento</span>
        <button class="nav-btn2" id="prevEvent" ${atFirstEvent ? "disabled" : ""}>← precedente</button>
        <span class="nav-pos">${event.name}</span>
        <button class="nav-btn2" id="nextEvent" ${atLastEvent ? "disabled" : ""}>successivo →</button>
      </div>
    </div>

    <div class="gif-panel">
      <img src="${s.gif}" alt="compare gif">
      <div class="cap">Confronto animato: input → target / pred / pred mod</div>
    </div>

    <div class="frames-panel">
      ${rowHtml("Input", "input", s.inputs)}
      ${rowHtml("Target", "target", s.targets)}
      ${rowHtml("Pred", "pred", s.preds)}
      ${rowHtml("Pred mod", "predm", s.predms)}
    </div>
  `;

  document.getElementById("prevSample")?.addEventListener("click", () => !atFirstSample && selectSample(sel.split, sel.event, sel.sample - 1));
  document.getElementById("nextSample")?.addEventListener("click", () => !atLastSample && selectSample(sel.split, sel.event, sel.sample + 1));
  document.getElementById("prevEvent")?.addEventListener("click", () => !atFirstEvent && jumpEvent(-1));
  document.getElementById("nextEvent")?.addEventListener("click", () => !atLastEvent && jumpEvent(1));
}

function jumpEvent(dir) {
  let si = sel.split, ei = sel.event + dir;
  if (ei < 0) { si -= 1; ei = si >= 0 ? TREE[si].events.length - 1 : 0; }
  if (si >= 0 && ei >= TREE[si].events.length) { si += 1; ei = 0; }
  if (si < 0 || si >= TREE.length) return;
  selectSample(si, ei, 0);
}

document.addEventListener("keydown", (e) => {
  if (document.activeElement === searchInput) return;
  if (e.key === "ArrowLeft") document.getElementById("prevSample")?.click();
  if (e.key === "ArrowRight") document.getElementById("nextSample")?.click();
  if (e.key === "ArrowUp") document.getElementById("prevEvent")?.click();
  if (e.key === "ArrowDown") document.getElementById("nextEvent")?.click();
});

renderSidebar();
renderViewer();
</script>
</body>
</html>
"""


def build_gallery(root: Path) -> str:
    samples = find_samples(root)
    tree = build_tree(samples)

    from datetime import datetime
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    html_out = HTML_TEMPLATE
    html_out = html_out.replace("__ROOT_NAME__", root.name or str(root))
    html_out = html_out.replace("__SAMPLE_COUNT__", str(len(samples)))
    html_out = html_out.replace("__GENERATED_AT__", generated_at)
    html_out = html_out.replace("__DATA_JSON__", json.dumps(tree, ensure_ascii=False))
    return html_out, len(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera gallery.html per esplorare i risultati di un'inferenza.")
    parser.add_argument("--root", "-r", required=True, type=str, help="Cartella risultati (es. Model_Results/Model_Results_flip).")
    parser.add_argument("--out", type=str, default=None, help="Nome file di output (default: gallery.html dentro --root).")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Cartella non trovata: {root}")

    out_path = Path(args.out).expanduser().resolve() if args.out else (root / "gallery.html")
    html_out, n_samples = build_gallery(root)
    out_path.write_text(html_out, encoding="utf-8")

    print(f"[gallery] {n_samples} sample trovati sotto '{root}'")
    print(f"[gallery] Scritto: {out_path}")


if __name__ == "__main__":
    main()
