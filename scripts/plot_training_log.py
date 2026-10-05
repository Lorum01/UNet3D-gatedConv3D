"""Parse a training.log produced by train_loop.py and render a professional
train/val loss (+ learning rate) figure.

Usage:
    python scripts/plot_training_log.py <checkpoint_dir> [--out OUT.png]
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt

TRAIN_RE = re.compile(r"\[TRAIN\] MSE-Lpips Loss: ([\d.]+)")
VAL_RE = re.compile(r"\[VAL\s*\] MSE-Lpips Loss: ([\d.]+)")
LR_RE = re.compile(r"\[LR\s*\] ([\d.]+)")

# Palette (dataviz skill reference palette, light mode)
COLOR_TRAIN = "#2a78d6"   # categorical slot 1 - blue
COLOR_VAL = "#eb6834"     # categorical slot 2 - orange
COLOR_BEST = "#0ca30c"    # status "good"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e3e1dc"
SURFACE = "#fcfcfb"


def parse_log(log_path: Path):
    train_losses, val_losses, lrs = [], [], []
    train_v = val_v = lr_v = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        m = TRAIN_RE.search(line)
        if m:
            train_v = float(m.group(1))
            continue
        m = VAL_RE.search(line)
        if m:
            val_v = float(m.group(1))
            continue
        m = LR_RE.search(line)
        if m:
            lr_v = float(m.group(1))
            if train_v is not None and val_v is not None:
                train_losses.append(train_v)
                val_losses.append(val_v)
                lrs.append(lr_v)
                train_v = val_v = lr_v = None
    return train_losses, val_losses, lrs


def plot(train_losses, val_losses, out_path: Path, title: str):
    epochs = range(1, len(train_losses) + 1)
    best_epoch = min(range(len(val_losses)), key=lambda i: val_losses[i]) + 1
    best_val = min(val_losses)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": TEXT_PRIMARY,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_SECONDARY,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
    })

    fig, ax_loss = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    ax_loss.set_facecolor(SURFACE)

    ax_loss.plot(epochs, train_losses, color=COLOR_TRAIN, linewidth=2, label="Train loss")
    ax_loss.plot(epochs, val_losses, color=COLOR_VAL, linewidth=2, label="Val loss")
    ax_loss.scatter([best_epoch], [best_val], s=45, zorder=5,
                     facecolor=COLOR_BEST, edgecolor=SURFACE, linewidth=1.2)
    ax_loss.annotate(
        f"best epoch {best_epoch}\nval {best_val:.4f}",
        xy=(best_epoch, best_val), xytext=(10, 18), textcoords="offset points",
        fontsize=9.5, color=TEXT_SECONDARY,
        arrowprops=dict(arrowstyle="-", color=GRID_COLOR, lw=1),
    )
    ax_loss.axvline(best_epoch, color=GRID_COLOR, linewidth=1, linestyle="--", zorder=0)

    ax_loss.set_ylabel("MSE-LPIPS loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_title(title, fontsize=13, fontweight="bold", color=TEXT_PRIMARY, pad=12, loc="left")
    ax_loss.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax_loss.spines[["top", "right"]].set_visible(False)
    legend = ax_loss.legend(frameon=False, loc="upper right")
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    log_path = args.checkpoint_dir / "training.log"
    out_path = args.out or args.checkpoint_dir / "train_val_loss_pro.png"

    train_losses, val_losses, _lrs = parse_log(log_path)
    if not train_losses:
        raise SystemExit(f"No epoch data found in {log_path}")

    title = f"Train vs. validation loss — {args.checkpoint_dir.name}"
    plot(train_losses, val_losses, out_path, title)
    print(f"Saved {out_path} ({len(train_losses)} epochs)")


if __name__ == "__main__":
    main()
