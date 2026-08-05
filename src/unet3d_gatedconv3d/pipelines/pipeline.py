from __future__ import annotations

import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

from ..models.model import build_model
from ..training.train_loop import training_loop_with_validation_3d
from ..inference.test_utility import test_model_create_gifs_3ch
from ..inference.metrics import evaluate_metrics_3d, save_metrics_csv

from .prep_data import build_dataloaders, save_split_assignments


def _run_metrics_for_split(
    *,
    model,
    loader,
    device: str,
    out_dir: Path,
    alpha: float,
    lpips_input_mode: str,
    mean,
    std,
    scale_to_neg1_pos1: bool,
    checkpoint_path: str,
    max_batches,
    split_name: str,
) -> None:
    """Calcola loss combinata MSE+LPIPS + SSIM/PSNR/MSE (rami pred/predm) e salva metrics.csv."""
    metrics = evaluate_metrics_3d(
        model=model,
        dataloader=loader,
        device=device,
        alpha=alpha,
        lpips_input_mode=lpips_input_mode,
        mean=mean,
        std=std,
        scale_to_neg1_pos1=scale_to_neg1_pos1,
        checkpoint_path=checkpoint_path,
        max_batches=max_batches,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    save_metrics_csv(metrics, str(out_dir / "metrics.csv"))
    for branch in ("pred", "predm"):
        if branch not in metrics:
            continue
        b = metrics[branch]
        print(
            f"[metrics][{split_name}][{branch}] combined_loss(alpha={alpha})={b['combined_loss']['mean']:.4f} "
            f"ssim={b['ssim']['mean']:.4f} psnr={b['psnr']['mean']:.2f}dB mse={b['mse']['mean']:.6f} "
            f"(n_samples={metrics['n_samples']})"
        )


def _seed_training_run(seed: Optional[int]) -> int:
    """Fissa i RNG di Python/NumPy/Torch per l'inizializzazione dei pesi e lo shuffle
    dei batch di training. Va chiamata DOPO build_dataloaders() (che ha gia' consumato
    dataset.split.seed per calcolare lo split) e PRIMA di build_model(), cosi' che
    dataset.split.seed resti l'unico responsabile dell'appartenenza degli eventi agli
    split: variare train.seed tra piu' run non cambia lo split, solo pesi iniziali e
    ordine dei batch. Se seed e' None ne viene generato uno nuovo (non deterministico)
    e restituito al chiamante, cosi' da poterlo registrare (es. in config_resolved.yaml)
    anche quando non specificato esplicitamente in config.
    """
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big")
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def _save_run_config(config_path: Optional[Path], cfg: Dict[str, Any], out_dir: Path) -> None:
    """Salva il config usato per la run (raw + risolto) dentro out_dir."""
    if config_path is not None and Path(config_path).exists():
        shutil.copy2(str(config_path), str(out_dir / Path(config_path).name))
    with open(out_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def run(cfg: Dict[str, Any], project_root: Path, config_path: Optional[Path] = None) -> None:
    mode = (cfg.get("mode") or "train").lower().strip()
    if mode not in {"train", "infer"}:
        raise ValueError(f"Unsupported mode={mode!r}. Expected 'train' or 'infer'.")

    loaders, split_info = build_dataloaders(cfg)

    if mode == "train":
        used_seed = _seed_training_run((cfg.get("train") or {}).get("seed"))
        cfg["train"]["seed"] = used_seed
        print(f"[train] train.seed={used_seed} (dataset.split.seed invariato: split identico ad altre run)")

    model = build_model(cfg["model"])

    dcfg = cfg.get("dataset") or {}
    normalization_mode = str(dcfg.get("normalization", {}).get("mode", "neg1pos1")).strip().lower()
    data_scale_to_neg1_pos1 = normalization_mode == "neg1pos1"

    if mode == "train":
        if isinstance(loaders, dict):
            raise ValueError("mode=train richiede dataset.split.strategy='train_val_test' (non 'by_class').")
        train_loader, val_loader, _test_loader = loaders
        tcfg = cfg["train"]
        checkpoint_dir = project_root / tcfg["checkpoint"]["dir"]
        if checkpoint_dir.exists():
            # Evita di sovrascrivere una run precedente: stessa logica che prima viveva
            # dentro training_loop_with_validation_3d, spostata qui perche' ora la cartella
            # finale deve essere decisa PRIMA di scrivere split_assignments.csv (cosi' resta
            # nella stessa cartella di training.log/metrics.csv/checkpoint, non in due diverse).
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            orig_dir = checkpoint_dir
            checkpoint_dir = checkpoint_dir.parent / f"{checkpoint_dir.name}_{ts}"
            print(f"Checkpoint directory '{orig_dir}' already exists. Using new directory: '{checkpoint_dir}'")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _save_run_config(config_path, cfg, checkpoint_dir)
        save_split_assignments(split_info, checkpoint_dir / "split_assignments.csv")
        training_loop_with_validation_3d(
            model,
            train_loader,
            val_loader,
            num_epochs=int(tcfg["num_epochs"]),
            lr=float(tcfg["lr"]),
            device=str(tcfg["device"]),
            patience_early_stopping=int(tcfg["early_stopping"]["patience"]),
            patience_lr_scheduler=int(tcfg["lr_scheduler"]["patience"]),
            factor=float(tcfg["lr_scheduler"]["factor"]),
            threshold=float(tcfg["lr_scheduler"]["threshold"]),
            checkpoint_interval=int(tcfg["checkpoint"]["interval"]),
            checkpoint_dir=str(checkpoint_dir),
            alpha=float(tcfg["loss"]["alpha"]),
            lpips_input_mode=str(tcfg["loss"].get("lpips_input_mode", "none")),
            show_plots=bool(tcfg.get("show_plots", False)),
            use_data_parallel=bool(tcfg.get("use_data_parallel", False)),
        )
        return

    # infer
    icfg = cfg["infer"]
    device = icfg["device"]
    checkpoint_path = icfg["checkpoint_path"]
    save_cfg = icfg["save"]
    save_dirs = save_cfg["dirs"]

    resolved_ckpt = str((project_root / checkpoint_path).resolve()) if not Path(checkpoint_path).is_absolute() else checkpoint_path

    # Visualization denorm choice:
    # - if infer.normalization_override.denorm_from_neg1_pos1 is set, it overrides
    #   dataset.normalization.mode
    # - else it follows dataset.normalization.mode (recommended default)
    norm_override = icfg.get("normalization_override", {}) or {}
    denorm_from_neg1_pos1 = norm_override.get("denorm_from_neg1_pos1", None)
    if denorm_from_neg1_pos1 is None:
        denorm_from_neg1_pos1 = data_scale_to_neg1_pos1
    else:
        denorm_from_neg1_pos1 = bool(denorm_from_neg1_pos1)

    override_mean = norm_override.get("mean", None)
    override_std = norm_override.get("std", None)

    loss_cfg = cfg.get("train", {}).get("loss", {})
    metrics_cfg = icfg.get("metrics", {})
    metrics_enabled = bool(metrics_cfg.get("enabled", True))
    metrics_alpha = float(loss_cfg.get("alpha", 0.7))
    metrics_lpips_input_mode = str(loss_cfg.get("lpips_input_mode", "none"))
    metrics_max_batches = metrics_cfg.get("max_batches")
    metrics_max_batches = int(metrics_max_batches) if metrics_max_batches is not None else None

    # Caso richiesto: inferenza raggruppata per classe (da Excel), senza split train/val/test
    if isinstance(loaders, dict):
        by_class_root = str(project_root / save_dirs.get("by_class_root", "Model_Results_1/by_class"))
        for class_id, loader in loaders.items():
            out_dir = str(Path(by_class_root) / f"class_{class_id}")
            test_model_create_gifs_3ch(
                model=model,
                test_loader=loader,
                device=device,
                save_dir=out_dir,
                max_batches=int(icfg["max_batches"]),
                gif_fps=int(icfg["gif_fps"]),
                mean=override_mean,
                std=override_std,
                scale_to_neg1_pos1=denorm_from_neg1_pos1,
                checkpoint_path=resolved_ckpt,
                save_images=bool(save_cfg["images"]),
                save_gifs=bool(save_cfg["gifs"]),
                show_plots=bool(save_cfg["show_plots"]),
            )
            if metrics_enabled:
                _run_metrics_for_split(
                    model=model,
                    loader=loader,
                    device=device,
                    out_dir=Path(out_dir),
                    alpha=metrics_alpha,
                    lpips_input_mode=metrics_lpips_input_mode,
                    mean=override_mean,
                    std=override_std,
                    scale_to_neg1_pos1=denorm_from_neg1_pos1,
                    checkpoint_path=resolved_ckpt,
                    max_batches=metrics_max_batches,
                    split_name=f"class_{class_id}",
                )
        return

    train_loader, val_loader, test_loader = loaders
    run_cfg = icfg.get("run", {})

    # split_assignments.csv scritto UNA sola volta nella cartella comune a
    # test/val/train (non duplicato in ognuna delle sottocartelle di split).
    results_root = Path(os.path.commonpath([
        str((project_root / save_dirs.get("test", "Model_Results_1/test")).resolve()),
        str((project_root / save_dirs.get("val", "Model_Results_1/val")).resolve()),
        str((project_root / save_dirs.get("train", "Model_Results_1/train")).resolve()),
    ]))
    save_split_assignments(split_info, results_root / "split_assignments.csv")

    if bool(run_cfg.get("test", True)):
        test_dir = project_root / save_dirs.get("test", "Model_Results_1/test")
        test_model_create_gifs_3ch(
            model=model,
            test_loader=test_loader,
            device=device,
            save_dir=str(test_dir),
            max_batches=int(icfg["max_batches"]),
            gif_fps=int(icfg["gif_fps"]),
            mean=override_mean,
            std=override_std,
            scale_to_neg1_pos1=denorm_from_neg1_pos1,
            checkpoint_path=resolved_ckpt,
            save_images=bool(save_cfg["images"]),
            save_gifs=bool(save_cfg["gifs"]),
            show_plots=bool(save_cfg["show_plots"]),
        )
        if metrics_enabled:
            _run_metrics_for_split(
                model=model,
                loader=test_loader,
                device=device,
                out_dir=test_dir,
                alpha=metrics_alpha,
                lpips_input_mode=metrics_lpips_input_mode,
                mean=override_mean,
                std=override_std,
                scale_to_neg1_pos1=denorm_from_neg1_pos1,
                checkpoint_path=resolved_ckpt,
                max_batches=metrics_max_batches,
                split_name="test",
            )

    if bool(run_cfg.get("val", False)):
        val_dir = project_root / save_dirs.get("val", "Model_Results_1/val")
        test_model_create_gifs_3ch(
            model=model,
            test_loader=val_loader,
            device=device,
            save_dir=str(val_dir),
            max_batches=int(icfg["max_batches"]),
            gif_fps=int(icfg["gif_fps"]),
            mean=override_mean,
            std=override_std,
            scale_to_neg1_pos1=denorm_from_neg1_pos1,
            checkpoint_path=resolved_ckpt,
            save_images=bool(save_cfg["images"]),
            save_gifs=bool(save_cfg["gifs"]),
            show_plots=bool(save_cfg["show_plots"]),
        )
        if metrics_enabled:
            _run_metrics_for_split(
                model=model,
                loader=val_loader,
                device=device,
                out_dir=val_dir,
                alpha=metrics_alpha,
                lpips_input_mode=metrics_lpips_input_mode,
                mean=override_mean,
                std=override_std,
                scale_to_neg1_pos1=denorm_from_neg1_pos1,
                checkpoint_path=resolved_ckpt,
                max_batches=metrics_max_batches,
                split_name="val",
            )

    if bool(run_cfg.get("train", False)):
        train_dir = project_root / save_dirs.get("train", "Model_Results_1/train")
        test_model_create_gifs_3ch(
            model=model,
            test_loader=train_loader,
            device=device,
            save_dir=str(train_dir),
            max_batches=int(icfg["max_batches"]),
            gif_fps=int(icfg["gif_fps"]),
            mean=override_mean,
            std=override_std,
            scale_to_neg1_pos1=denorm_from_neg1_pos1,
            checkpoint_path=resolved_ckpt,
            save_images=bool(save_cfg["images"]),
            save_gifs=bool(save_cfg["gifs"]),
            show_plots=bool(save_cfg["show_plots"]),
        )
        if metrics_enabled:
            _run_metrics_for_split(
                model=model,
                loader=train_loader,
                device=device,
                out_dir=train_dir,
                alpha=metrics_alpha,
                lpips_input_mode=metrics_lpips_input_mode,
                mean=override_mean,
                std=override_std,
                scale_to_neg1_pos1=denorm_from_neg1_pos1,
                checkpoint_path=resolved_ckpt,
                max_batches=metrics_max_batches,
                split_name="train",
            )
