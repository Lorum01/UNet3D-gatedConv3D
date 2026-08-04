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
            "dataset_dir": "../Dataset",
            "excel_path": "../EventsEtnaCLASS_Final.xlsx",
        },
        "dataset": {
            "input_length": 4,
            "prediction_length": 4,
            "stride": 1,
            "image_size": [100, 100],  # [H, W]
            "channels": 3,
            "normalization": {
                # "none": no transform (assumes inputs already in desired range)
                # "neg1pos1": map [0,1] -> [-1,1]
                # "standardize": (x - mean) / std
                #   - with split.strategy: train_val_test, mean/std are computed on the
                #     train split automatically (the values below are ignored).
                #   - with split.strategy: by_class, mean/std below are REQUIRED.
                "mode": "neg1pos1",
                "mean": None,
                "std": None,
            },
            "split": {
                "strategy": "train_val_test",  # "train_val_test" | "by_class"
                "seed": 424,
                # Only used when strategy: train_val_test. Percentages per event class
                # (as found in the `Class` column of the Excel file).
                "class_percentages": {
                    1: {"train": 0.7, "val": 0.15, "test": 0.15},
                    2: {"train": 0.7, "val": 0.15, "test": 0.15},
                    3: {"train": 0.7, "val": 0.15, "test": 0.15},
                    4: {"train": 0.7, "val": 0.15, "test": 0.15},
                },
            },
        },
        "dataloader": {
            "batch_size": {
                "train": 4,
                "val": 4,
                "test": 4,
            },
        },
        "model": {
            "unet": {
                "in_channels": 3,
                "base_channels": 32,
                "num_levels": 5,
                "out_channels": 64,
            },
            "stacked_conv": {
                "hidden_dims": [64, 128, 256, 512],
                "kernel_size": 3,
                "padding": 1,
            },
            "final_out_channels": 3,
        },
        "train": {
            "device": "cuda",
            "num_epochs": 250,
            "lr": 1e-3,
            # If True, wrap the model in nn.DataParallel *when the machine actually has
            # more than one visible CUDA GPU*. On a single-GPU/CPU machine this is a no-op.
            "use_data_parallel": False,
            "checkpoint": {
                "dir": "Checkpoints",
                "interval": 10,
            },
            "early_stopping": {
                "patience": 80,
            },
            "lr_scheduler": {
                "patience": 12,
                "factor": 0.5,
                "threshold": 1e-4,
            },
            "loss": {
                "alpha": 0.7,
                # Come preparare l'output del modello prima di passarlo a LPIPS
                # (non influisce sulla MSE, che usa sempre gli outputs originali):
                # "none":  nessuna trasformazione (default). Gli outputs possono
                #          uscire da [-1,1] se il modello non ha un'attivazione
                #          limitata in uscita; la LPIPS riceve input fuori dal
                #          suo dominio senza errori, ma in modo meno affidabile.
                # "clamp": torch.clamp(outputs, -1, 1). Taglio netto: gradiente
                #          del ramo LPIPS nullo sugli elementi fuori range.
                # "tanh":  torch.tanh(outputs). Compressione morbida in (-1,1),
                #          senza zone a gradiente esattamente zero.
                "lpips_input_mode": "none",
            },
            "show_plots": False,
        },
        "infer": {
            "device": "cuda",
            "checkpoint_path": "../Checkpoints/convUnet_Best_Test_mod_1/best_model.pth",
            "max_batches": 10,
            "gif_fps": 2,
            # Overrides for visualization denormalization only (mapping model output back
            # to [0,1] for saving/plotting). Leave null to derive automatically from
            # dataset.normalization (recommended).
            "normalization_override": {
                "mean": None,
                "std": None,
                "denorm_from_neg1_pos1": None,
            },
            "save": {
                "images": True,
                "gifs": True,
                "show_plots": False,
                "dirs": {
                    "test": "Model_Results_1/test",
                    "val": "Model_Results_1/val",
                    "train": "Model_Results_1/train",
                    "by_class_root": "Model_Results_1/by_class",
                },
            },
            "run": {
                "test": True,
                "val": False,
                "train": False,
            },
            "metrics": {
                # Se True, oltre a immagini/GIF calcola e salva (metrics.csv per split)
                # la loss combinata MSE+LPIPS (stesso alpha/lpips_input_mode di train.loss)
                # e SSIM/PSNR/MSE, sui rami "pred" (diretto) e "predm" (autoregressivo).
                "enabled": True,
                # None = valuta l'intero split; altrimenti limita il numero di batch
                # (indipendente da infer.max_batches, che riguarda solo immagini/GIF).
                "max_batches": None,
            },
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

    cfg["paths"]["dataset_dir"] = _resolve_path(cfg["paths"]["dataset_dir"], project_root)
    cfg["paths"]["excel_path"] = _resolve_path(cfg["paths"]["excel_path"], project_root)
    # checkpoint dir/checkpoint_path should remain relative to project root (resolved
    # where they are used, so a fresh run can still create them under project_root).
    return cfg
