from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ..models.model import build_model
from ..training.train_loop import training_loop_with_validation_3d
from ..inference.test_utility import test_model_create_gifs_3ch

from .prep_data import build_dataloaders, save_split_assignments


def run(cfg: Dict[str, Any], project_root: Path) -> None:
    mode = (cfg.get("mode") or "train").lower().strip()
    if mode not in {"train", "infer"}:
        raise ValueError(f"Unsupported mode={mode!r}. Expected 'train' or 'infer'.")

    loaders, split_info = build_dataloaders(cfg)
    model = build_model(cfg["model"])

    dcfg = cfg.get("dataset") or {}
    normalization_mode = str(dcfg.get("normalization", {}).get("mode", "neg1pos1")).strip().lower()
    data_scale_to_neg1_pos1 = normalization_mode == "neg1pos1"

    if mode == "train":
        if isinstance(loaders, dict):
            raise ValueError("mode=train richiede dataset.split.strategy='train_val_test' (non 'by_class').")
        train_loader, val_loader, _test_loader = loaders
        tcfg = cfg["train"]
        result = training_loop_with_validation_3d(
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
            checkpoint_dir=str(project_root / tcfg["checkpoint"]["dir"]),
            alpha=float(tcfg["loss"]["alpha"]),
            lpips_input_mode=str(tcfg["loss"].get("lpips_input_mode", "none")),
            show_plots=bool(tcfg.get("show_plots", False)),
            use_data_parallel=bool(tcfg.get("use_data_parallel", False)),
        )
        save_split_assignments(split_info, Path(result["checkpoint_dir"]) / "split_assignments.csv")
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
        return

    train_loader, val_loader, test_loader = loaders
    run_cfg = icfg.get("run", {})

    if bool(run_cfg.get("test", True)):
        test_dir = project_root / save_dirs.get("test", "Model_Results_1/test")
        save_split_assignments(split_info, test_dir / "split_assignments.csv")
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

    if bool(run_cfg.get("val", False)):
        val_dir = project_root / save_dirs.get("val", "Model_Results_1/val")
        save_split_assignments(split_info, val_dir / "split_assignments.csv")
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

    if bool(run_cfg.get("train", False)):
        train_dir = project_root / save_dirs.get("train", "Model_Results_1/train")
        save_split_assignments(split_info, train_dir / "split_assignments.csv")
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
