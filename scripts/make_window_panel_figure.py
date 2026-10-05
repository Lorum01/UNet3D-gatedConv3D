"""Figura a 12 pannelli (4 input / 4 target / 4 prediction) per singole finestre.

Struttura identica alle figure di riferimento:
  riga 1  Input Frame 1..4      (t_-3 .. t_0)
  riga 2  Target Frame 5..8     (t_0 + Δt .. t_0 + 4Δt)
  riga 3  Prediction Frame 5..8 (t_0 + Δt .. t_0 + 4Δt)
Ogni pannello ha la lettera (a..l) in alto a sinistra e un titolo su due righe
(etichetta + data/ora presa dal nome file, quindi corretta anche a cavallo di
mezzanotte).

Sorgente: Model_Results_windows/<run>/sample_batch1_sample<idx>_label<cls>/
(prodotta da scripts/infer_single_windows.py — modello seed 424, alpha 0.7,
remap (x+1)/2).

Uso:
    python scripts/make_window_panel_figure.py
"""
from __future__ import annotations

import os
import re
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_DIR = os.path.join(ROOT, "Model_Results_windows", "seed424_remap")
OUT_DIR = os.path.join(ROOT, "Model_Results_windows", "figures")

DT_MIN = 15  # minuti fra un frame e il successivo

# (cartella sample, primo frame di input, etichetta file di output)
SAMPLES = [
    ("sample_batch1_sample2_label3", "2021_06_20_2327", "panel_20210620_2327_seed424_a07"),
    ("sample_batch1_sample3_label4", "2024_07_07_0442", "panel_20240707_0442_seed424_a07"),
]

_TS_RE = re.compile(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})(\d{2})")


def parse_ts(fname: str) -> datetime:
    y, mo, d, h, mi = _TS_RE.search(fname).groups()
    return datetime(int(y), int(mo), int(d), int(h), int(mi))


def load_series(sample_dir: str, prefix: str, n: int = 4):
    """Ritorna [(array HxWx3, datetime), ...] ordinati per timestamp."""
    files = [f for f in os.listdir(sample_dir)
             if f.startswith(prefix) and f.endswith(".npy")]
    files = sorted(files, key=parse_ts)[:n]
    out = []
    for f in files:
        arr = np.clip(np.load(os.path.join(sample_dir, f)), 0.0, 1.0)
        out.append((arr, parse_ts(f)))
    return out


IN_TAGS = [r"$t_{-3}$", r"$t_{-2}$", r"$t_{-1}$", r"$t_{0}$"]
OUT_TAGS = [r"$t_0 + \Delta t$", r"$t_0 + 2\Delta t$", r"$t_0 + 3\Delta t$", r"$t_0 + 4\Delta t$"]

# contrasto di visualizzazione: stessa trasformazione (stretch lineare globale +
# gamma) su tutti e 3 i canali e su tutti i 12 frame -> le tinte restano coerenti
# e input/target/prediction sono confrontabili a colpo d'occhio.
ENH_PCT = (1.0, 99.0)
ENH_GAMMA = 1.6


def enhance(frames: list[np.ndarray]) -> list[np.ndarray]:
    stack = np.stack(frames)
    lo, hi = np.percentile(stack, ENH_PCT)
    out = np.clip((stack - lo) / (hi - lo), 0.0, 1.0) ** ENH_GAMMA
    return [out[i] for i in range(len(frames))]


def make_figure(sample_dir: str, out_path: str, enhanced: bool = False) -> None:
    inp = load_series(sample_dir, "input_t")
    tgt = load_series(sample_dir, "target_t")
    prd = load_series(sample_dir, "pred_SEVIRI")

    if enhanced:
        arrs = enhance([a for a, _ in inp + tgt + prd])
        inp = [(arrs[i], inp[i][1]) for i in range(4)]
        tgt = [(arrs[4 + i], tgt[i][1]) for i in range(4)]
        prd = [(arrs[8 + i], prd[i][1]) for i in range(4)]

    rows = [
        ("Input Frame",      inp, IN_TAGS,  1),
        ("Target Frame",     tgt, OUT_TAGS, 5),
        ("Prediction Frame", prd, OUT_TAGS, 5),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(11.0, 8.7))
    letters = list("abcdefghijkl")
    li = 0
    for r, (label, series, tags, start_n) in enumerate(rows):
        for c in range(4):
            ax = axes[r, c]
            arr, ts = series[c]
            ax.imshow(arr, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            title = (f"{label} {start_n + c} ({tags[c]})\n"
                     f"{ts.strftime('%d/%m/%Y %H:%M')}")
            ax.set_title(title, fontsize=10, fontweight="bold", linespacing=1.35, pad=6)
            ax.text(0.028, 0.972, letters[li], transform=ax.transAxes,
                    fontsize=13, fontweight="bold", va="top", ha="left",
                    color="white",
                    bbox=dict(boxstyle="square,pad=0.18", fc="black", ec="none"))
            li += 1

    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01,
                        wspace=0.06, hspace=0.28)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.replace(".png", ".pdf"))
    plt.close(fig)
    print("scritto", out_path)


def main():
    for sample_dir, _first, name in SAMPLES:
        d = os.path.join(RUN_DIR, sample_dir)
        make_figure(d, os.path.join(OUT_DIR, name + ".png"), enhanced=False)
        make_figure(d, os.path.join(OUT_DIR, name + "_enhanced.png"), enhanced=True)


if __name__ == "__main__":
    main()
