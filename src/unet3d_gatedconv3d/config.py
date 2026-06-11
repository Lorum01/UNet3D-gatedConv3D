from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _defaults() -> Dict[str, Any]:
    return {
        "mode": "train",
        "paths": {
            "dataset_folder": "../Dataset",
            "excel_path": "../EventsEtnaCLASS_Final.xlsx",
        },
        "data": {
            "split_strategy": "train_val_test",  # "train_val_test" | "by_class"
            "input_length": 4,
            "prediction_length": 4,
            "stride": 1,
            "expected_input_shape": [4, 100, 100, 3],
            # Normalization strategy for dataset values:
            # - "none": no transform (assumes inputs already in desired range)
            # - "neg1pos1": map [0,1] -> [-1,1]
            # - "standardize": (x - mean) / std with mean/std computed on train split
            # If omitted, legacy behavior follows `scale_to_neg1_pos1`.
            "normalization": None,
            "scale_to_neg1_pos1": True,
            "use_percent_distribution": True,
            "class_pct": {
                1: {"train": 0.7, "val": 0.15, "test": 0.15},
                2: {"train": 0.7, "val": 0.15, "test": 0.15},
                3: {"train": 0.7, "val": 0.15, "test": 0.15},
                4: {"train": 0.7, "val": 0.15, "test": 0.15},
            },
            "split_seed": 424,
        },
        "dataloader": {
            "batch_size_train": 4,
            "batch_size_val": 4,
            "batch_size_test": 4,
        },
        "model": {
            "unet_in_channels": 3,
            "unet_base_channels": 32,
            "unet_num_levels": 5,
            "unet_out_channels": 64,
            "stackedconv_hidden_dims": [64, 128, 256, 512],
            "stackedconv_kernel_size": 3,
            "stackedconv_padding": 1,
            "final_out_channels": 3,
        },
        "train": {
            "device": "cuda",
            "num_epochs": 250,
            "lr": 1e-3,
            "checkpoint_dir": "Checkpoints",
            "checkpoint_interval": 10,
            "patience_early_stopping": 80,
            "patience_lr_scheduler": 12,
            "lr_factor": 0.5,
            "lr_threshold": 1e-4,
            "alpha": 0.7,
            "show_plots": False,
        },
        "infer": {
            "device": "cuda",
            "checkpoint_path": "../convUnet_Best_Test_mod_1/best_model.pth",
            "save_dir_test": "Model_Results_1/test",
            "save_dir_val": "Model_Results_1/val",
            "save_dir_train": "Model_Results_1/train",
            "save_dir_by_class_root": "Model_Results_1/by_class",
            "max_batches": 10,
            "gif_fps": 2,
            "mean": None,
            "std": None,
            # If set, overrides data.scale_to_neg1_pos1 for visualization denorm.
            # None keeps backward-compatible behavior (follow data.scale_to_neg1_pos1).
            "denorm_from_neg1_pos1": None,
            "save_images": True,
            "save_gifs": True,
            "show_plots": False,
            "run_test": True,
            "run_val": False,
            "run_train": False,
        },
    }


def _resolve_path(p: str, project_root: Path) -> str:
    path = Path(p)
    if path.is_absolute():
        return str(path)
    return str((project_root / path).resolve())


def load_config(path: Path, project_root: Path) -> Dict[str, Any]:
    cfg = _defaults()

    with open(path, "r", encoding="utf-8") as f:
        ycfg = yaml.safe_load(f) or {}
    _deep_update(cfg, ycfg)

    cfg["paths"]["dataset_folder"] = _resolve_path(cfg["paths"]["dataset_folder"], project_root)
    cfg["paths"]["excel_path"] = _resolve_path(cfg["paths"]["excel_path"], project_root)
    # checkpoint dir should remain relative to project root (created there)
    return cfg
