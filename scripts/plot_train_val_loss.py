#!/usr/bin/env python3
"""
Ridisegna la curva train/val loss di una run (legge `train_*.log` nella cartella
del checkpoint) nello stile "best-epoch annotato": linea train blu, val arancione,
riga verticale tratteggiata + pallino verde sulla best epoch, annotazione.

Uso:
    python scripts/plot_train_val_loss.py <checkpoint_dir> [--out NOME.png]
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_log(log_path: str):
    txt = open(log_path, encoding="utf-8").read()
    parts = re.split(r"=== EPOCH (\d+)/\d+ ===", txt)
    epochs, train, val = [], [], []
    for i in range(1, len(parts), 2):
        n = int(parts[i])
        body = parts[i + 1]
        tr = re.search(r"\[TRAIN\] MSE-Lpips Loss: ([\d.]+)", body)
        va = re.search(r"\[VAL\s*\] MSE-Lpips Loss: ([\d.]+)", body)
        if tr and va:
            epochs.append(n)
            train.append(float(tr.group(1)))
            val.append(float(va.group(1)))
    return epochs, train, val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint_dir")
    ap.add_argument("--out", default="train_val_loss_annotated.png")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    logs = sorted(glob.glob(os.path.join(args.checkpoint_dir, "train_*.log")))
    if not logs:
        raise SystemExit(f"nessun train_*.log in {args.checkpoint_dir}")
    epochs, train, val = parse_log(logs[-1])
    if not epochs:
        raise SystemExit(f"nessuna epoca trovata in {logs[-1]}")

    best_i = min(range(len(val)), key=lambda k: val[k])
    best_ep, best_val = epochs[best_i], val[best_i]

    name = os.path.basename(os.path.normpath(args.checkpoint_dir))
    title = args.title or f"Train vs. validation loss — {name}"

    plt.rcParams.update({
        "figure.facecolor": "#fdfdfb",
        "axes.facecolor": "#fdfdfb",
        "font.size": 12,
    })
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)

    ax.plot(epochs, train, color="#1f77b4", lw=2, label="Train loss")
    ax.plot(epochs, val, color="#ff7f0e", lw=2, label="Val loss")

    ax.axvline(best_ep, color="#b0b0b0", ls="--", lw=1)
    ax.plot([best_ep], [best_val], "o", color="#2ca02c", ms=8, zorder=5)
    span = max(val) - min(val)
    left_side = best_ep > 0.6 * max(epochs)
    dx = -max(epochs) * 0.04 if left_side else max(epochs) * 0.03
    ax.annotate(
        f"best epoch {best_ep}\nval {best_val:.4f}",
        xy=(best_ep, best_val),
        xytext=(best_ep + dx, best_val + span * 0.28),
        ha="right" if left_side else "left",
        fontsize=11, color="#404040",
        arrowprops=dict(arrowstyle="-", color="#b0b0b0", lw=1),
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE-LPIPS loss")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)
    ax.grid(True, color="#e2e2e2", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="upper right", fontsize=12)
    ax.margins(x=0.02)

    out_path = os.path.join(args.checkpoint_dir, args.out)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"saved {out_path}  (best epoch {best_ep}, val {best_val:.4f}, {len(epochs)} epochs)")


if __name__ == "__main__":
    main()
