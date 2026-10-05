"""Inferenza su singole finestre scelte a mano (evento + primo frame di input).

Riusa esattamente la stessa pipeline di `test_model_create_gifs_3ch`
(input 4 -> pred 4, poi step autoregressivo per pred_mod) e il remap
(x+1)/2 -> [0,1] per salvataggi/visualizzazioni.

Uso:
    python scripts/infer_single_windows.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from unet3d_gatedconv3d.models.model import build_model
from unet3d_gatedconv3d.data.dataloader import CustomDataset, custom_collate_fn
from unet3d_gatedconv3d.inference.test_utility import test_model_create_gifs_3ch

DATASET = "/home/basile/workdir/vona/Dataset"
INPUT_LEN = 4
PRED_LEN = 4

MODEL_CFG = dict(
    unet_in_channels=3,
    unet_base_channels=32,
    unet_num_levels=5,
    unet_out_channels=64,
    stackedconv_hidden_dims=[64, 128, 256, 512],
    stackedconv_kernel_size=3,
    stackedconv_padding=1,
    final_out_channels=3,
)

# (nome cartella evento, primo frame di input, classe evento — la classe serve solo al naming)
WINDOWS = [
    ("2012_01_05", "SEVIRI_AshRGB_2012_01_05_0627.npy", 1),
    ("2021_07_08", "SEVIRI_AshRGB_2021_07_08_2057.npy", 1),
    ("2021_06_20", "SEVIRI_AshRGB_2021_06_20_2327.npy", 3),
    ("2024_07_07", "SEVIRI_AshRGB_2024_07_07_0442.npy", 4),
    ("2021_06_23", "SEVIRI_AshRGB_2021_06_23_0357.npy", 1),
]

RUNS = {
    "seed42_remap":  os.path.join(ROOT, "Checkpoints_legacy_window_split/best_model.pth"),
    "seed424_remap": os.path.join(ROOT, "Checkpoints_seed424/best_model.pth"),
}

OUT_ROOT = os.path.join(ROOT, "Model_Results_windows")
DEVICE = "cpu"  # le GPU sono occupate dai training; qui bastano pochi forward


def load_window(event: str, first_frame: str):
    ev_dir = os.path.join(DATASET, event)
    files = sorted(os.listdir(ev_dir))
    start = files.index(first_frame)
    win = files[start:start + INPUT_LEN + PRED_LEN]
    if len(win) < INPUT_LEN + PRED_LEN:
        raise ValueError(f"{event}: finestra troppo corta a partire da {first_frame}")
    frames = [np.load(os.path.join(ev_dir, f)).astype(np.float32) for f in win]  # (H,W,C) in [0,1]
    frames = np.stack(frames, axis=0)  # (T,H,W,C)
    inp = frames[:INPUT_LEN]
    tgt = frames[INPUT_LEN:INPUT_LEN + PRED_LEN]
    in_fn = [os.path.join(ev_dir, f) for f in win[:INPUT_LEN]]
    tg_fn = [os.path.join(ev_dir, f) for f in win[INPUT_LEN:INPUT_LEN + PRED_LEN]]
    return inp, tgt, in_fn, tg_fn


def main():
    inputs, targets, in_fns, tg_fns, labels = [], [], [], [], []
    for event, ff, cls in WINDOWS:
        inp, tgt, ifn, tfn = load_window(event, ff)
        inputs.append(inp)
        targets.append(tgt)
        in_fns.append(ifn)
        tg_fns.append(tfn)
        labels.append(int(cls))  # usato solo per il naming della cartella sample
        print(f"[win] {event}: input {[os.path.basename(p) for p in ifn]}")
        print(f"[win] {event}: target {[os.path.basename(p) for p in tfn]}")

    ds = CustomDataset(
        inputs, targets, np.array(labels),
        scale_to_neg1_pos1=True,
        input_filenames=in_fns, target_filenames=tg_fns,
    )
    loader = DataLoader(ds, batch_size=len(inputs), shuffle=False, collate_fn=custom_collate_fn)

    for run_name, ckpt in RUNS.items():
        print(f"\n===== {run_name} :: {ckpt} =====")
        model = build_model(MODEL_CFG)
        save_dir = os.path.join(OUT_ROOT, run_name)
        test_model_create_gifs_3ch(
            model=model,
            test_loader=loader,
            device=DEVICE,
            save_dir=save_dir,
            max_batches=10,
            gif_fps=2,
            mean=None, std=None,
            scale_to_neg1_pos1=True,   # remap (x+1)/2 -> [0,1]
            checkpoint_path=ckpt,
            save_images=True,
            save_gifs=True,
            show_plots=False,
        )
    print("\nFatto. Risultati in", OUT_ROOT)


if __name__ == "__main__":
    main()
