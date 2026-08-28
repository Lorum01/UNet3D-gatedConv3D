"""
Dry-run: costruisce SOLO i dataloader (quindi calcola lo split a livello finestra
e scrive i CSV split_seed<N>_*.csv), senza avviare il training.

    python scripts/dump_split.py -c configs/train_legacy_window_split.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from unet3d_gatedconv3d.config import load_config  # noqa: E402
from unet3d_gatedconv3d.pipelines.prep_data import build_dataloaders  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--out-dir", default=None, help="Dove scrivere i CSV dello split (default: cartella corrente).")
    args = ap.parse_args()

    cfg = load_config(Path(args.config).resolve(), project_root=PROJECT_ROOT)
    cfg["mode"] = "train"
    if args.out_dir:
        cfg.setdefault("data", {})["split_out_dir"] = args.out_dir

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    print(
        f"\nFinestre: train={len(train_loader.dataset)} "
        f"val={len(val_loader.dataset)} test={len(test_loader.dataset)}"
    )


if __name__ == "__main__":
    main()
