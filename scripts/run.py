from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from unet3d_gatedconv3d.config import load_config  # noqa: E402
from unet3d_gatedconv3d.pipelines.pipeline import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="UNet3D + GatedConv3D runner (train / inference).")
    parser.add_argument("--config", "-c", required=True, type=str, help="Path al file YAML di configurazione.")
    parser.add_argument("--mode", choices=["train", "infer"], default=None, help="Override della modalità.")
    args = parser.parse_args()

    cfg = load_config(Path(args.config), project_root=PROJECT_ROOT)
    if args.mode is not None:
        cfg["mode"] = args.mode

    run(cfg, project_root=PROJECT_ROOT)


if __name__ == "__main__":
    main()

