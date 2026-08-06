"""Unisce tutti i metrics.csv di inferenza sparsi nelle cartelle split in un unico
CSV centralizzato, senza toccare i file originali.

Uso:
    python scripts/merge_inference_metrics.py --root Model_Results

Il merge avviene gia' automaticamente a fine pipeline di inferenza (vedi
pipeline.run): questo script serve per rigenerare il CSV a mano, ad es. dopo aver
spostato/cancellato dei run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from unet3d_gatedconv3d.inference.metrics import merge_metrics_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("Model_Results"),
                         help="Cartella radice in cui cercare i metrics.csv (default: Model_Results)")
    parser.add_argument("--out", type=Path, default=None,
                         help="Path del CSV unito (default: <root>/all_metrics.csv)")
    args = parser.parse_args()

    out_path = args.out or (args.root / "all_metrics.csv")
    n_rows = merge_metrics_csv(str(args.root), str(out_path))
    if n_rows == 0:
        raise SystemExit(f"Nessun metrics.csv trovato sotto {args.root}")
    print(f"Unite {n_rows} righe in {out_path}")


if __name__ == "__main__":
    main()
