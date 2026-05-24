from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from ..models.model import build_model
from ..training.train_loop import training_loop_with_validation_3d
from ..inference.test_utility import test_model_create_gifs_3ch

from .prep_data import build_dataloaders


def run(cfg: Dict[str, Any], project_root: Path) -> None:
    mode = (cfg.get("mode") or "train").lower().strip()
    if mode not in {"train", "infer"}:
        raise ValueError(f"Unsupported mode={mode!r}. Expected 'train' or 'infer'.")

    loaders = build_dataloaders(cfg)
    model = build_model(cfg["model"])
    data_scale_to_neg1_pos1 = bool((cfg.get("data") or {}).get("scale_to_neg1_pos1", True))

    if mode == "train":
        if isinstance(loaders, dict):
            raise ValueError("mode=train richiede data.split_strategy='train_val_test' (non 'by_class').")
        train_loader, val_loader, _test_loader = loaders
        tcfg = cfg["train"]
        training_loop_with_validation_3d(
            model,
            train_loader,
            val_loader,
            num_epochs=int(tcfg["num_epochs"]),
            lr=float(tcfg["lr"]),
            device=str(tcfg["device"]),
            patience_early_stopping=int(tcfg["patience_early_stopping"]),
            patience_lr_scheduler=int(tcfg["patience_lr_scheduler"]),
            factor=float(tcfg["lr_factor"]),
            threshold=float(tcfg["lr_threshold"]),
            checkpoint_interval=int(tcfg["checkpoint_interval"]),
            checkpoint_dir=str(project_root / tcfg["checkpoint_dir"]),
            alpha=float(tcfg["alpha"]),
            show_plots=bool(tcfg.get("show_plots", False)),
        )
        return

    # infer
    icfg = cfg["infer"]
    device = icfg["device"]
    checkpoint_path = icfg["checkpoint_path"]
    save_dir_test = icfg.get("save_dir_test", "Model_Results_1/test")
    save_dir_val = icfg.get("save_dir_val", "Model_Results_1/val")
    save_dir_train = icfg.get("save_dir_train", "Model_Results_1/train")

    resolved_ckpt = str((project_root / checkpoint_path).resolve()) if not Path(checkpoint_path).is_absolute() else checkpoint_path

    # Visualization denorm choice:
    # - if infer.denorm_from_neg1_pos1 is set, it overrides data.scale_to_neg1_pos1
    # - else it follows data.scale_to_neg1_pos1 (backwards-compatible behavior)
    denorm_from_neg1_pos1 = icfg.get("denorm_from_neg1_pos1", None)
    if denorm_from_neg1_pos1 is None:
        denorm_from_neg1_pos1 = data_scale_to_neg1_pos1
    else:
        denorm_from_neg1_pos1 = bool(denorm_from_neg1_pos1)

    # Caso richiesto: inferenza raggruppata per classe (da Excel), senza split train/val/test
    if isinstance(loaders, dict):
        by_class_root = str(project_root / icfg.get("save_dir_by_class_root", "Model_Results_1/by_class"))
        for class_id, loader in loaders.items():
            out_dir = str(Path(by_class_root) / f"class_{class_id}")
            test_model_create_gifs_3ch(
                model=model,
                test_loader=loader,
                device=device,
                save_dir=out_dir,
                max_batches=int(icfg["max_batches"]),
                gif_fps=int(icfg["gif_fps"]),
                mean=icfg.get("mean", None),
                std=icfg.get("std", None),
                scale_to_neg1_pos1=denorm_from_neg1_pos1,
                checkpoint_path=resolved_ckpt,
                save_images=bool(icfg["save_images"]),
                save_gifs=bool(icfg["save_gifs"]),
                show_plots=bool(icfg["show_plots"]),
            )
        return

    train_loader, val_loader, test_loader = loaders

    if bool(icfg.get("run_test", True)):
        test_model_create_gifs_3ch(
            model=model,
            test_loader=test_loader,
            device=device,
            save_dir=str(project_root / save_dir_test),
            max_batches=int(icfg["max_batches"]),
            gif_fps=int(icfg["gif_fps"]),
            mean=icfg.get("mean", None),
            std=icfg.get("std", None),
            scale_to_neg1_pos1=denorm_from_neg1_pos1,
            checkpoint_path=resolved_ckpt,
            save_images=bool(icfg["save_images"]),
            save_gifs=bool(icfg["save_gifs"]),
            show_plots=bool(icfg["show_plots"]),
        )

    if bool(icfg.get("run_val", False)):
        test_model_create_gifs_3ch(
            model=model,
            test_loader=val_loader,
            device=device,
            save_dir=str(project_root / save_dir_val),
            max_batches=int(icfg["max_batches"]),
            gif_fps=int(icfg["gif_fps"]),
            mean=icfg.get("mean", None),
            std=icfg.get("std", None),
            scale_to_neg1_pos1=denorm_from_neg1_pos1,
            checkpoint_path=resolved_ckpt,
            save_images=bool(icfg["save_images"]),
            save_gifs=bool(icfg["save_gifs"]),
            show_plots=bool(icfg["show_plots"]),
        )

    if bool(icfg.get("run_train", False)):
        test_model_create_gifs_3ch(
            model=model,
            test_loader=train_loader,
            device=device,
            save_dir=str(project_root / save_dir_train),
            max_batches=int(icfg["max_batches"]),
            gif_fps=int(icfg["gif_fps"]),
            mean=icfg.get("mean", None),
            std=icfg.get("std", None),
            scale_to_neg1_pos1=denorm_from_neg1_pos1,
            checkpoint_path=resolved_ckpt,
            save_images=bool(icfg["save_images"]),
            save_gifs=bool(icfg["save_gifs"]),
            show_plots=bool(icfg["show_plots"]),
        )
