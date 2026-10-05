"""
Train / val loss (combined MSE+LPIPS, same criterion as training) for the six
seed-424 models, computed BOTH for the direct 4->4 prediction and for the
2-step AUTOREGRESSIVE prediction (`pred_mod` from test_utility:
 pred1 = model(x); pred2 = model([x[-2:], pred1[:2]]); predm = [pred1[:2], pred2[:2]]).

No images are written to disk: everything is computed in-memory, so this is safe
to run with almost no free disk space.

Loss is evaluated in the SAME normalized [-1,1] space as train_loop, with the
model in eval() mode and batch_size 4 (per-batch losses averaged), matching
`evaluate_model_3d`. Training's *train* loss was measured in train() mode, so the
"direct / train" number here can differ slightly from the training curve
(BatchNorm running stats vs batch stats, dropout off).

Output: Model_Results_autoregressive_loss/autoregressive_loss.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from unet3d_gatedconv3d.config import load_config  # noqa: E402
from unet3d_gatedconv3d.pipelines.prep_data import build_dataloaders  # noqa: E402
from unet3d_gatedconv3d.models.model import build_model  # noqa: E402
from unet3d_gatedconv3d.training.loss_function import weighted_mse_lpips_loss  # noqa: E402
from unet3d_gatedconv3d.inference.test_utility import permute_to_bcthw  # noqa: E402

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# (label, train-config, checkpoint) - alpha + model cfg are read from the config.
MODELS = [
    ("seed424_alpha0.0", "configs/train_seed424_alpha0.yaml", "Checkpoints_seed424_alpha0/best_model.pth"),
    ("seed424_alpha0.5", "configs/train_seed424_alpha05.yaml", "Checkpoints_seed424_alpha05/best_model.pth"),
    ("seed424_alpha0.7", "configs/train_seed424.yaml", "Checkpoints_seed424/best_model.pth"),
    ("seed424_alpha1.0", "configs/train_seed424_alpha1.yaml", "Checkpoints_seed424_alpha1/best_model.pth"),
    ("encdec_gatedconv3d_seed424", "configs/train_legacy_encdec_gatedconv3d_seed424.yaml",
     "Checkpoints_legacy_encdec_gatedconv3d_seed424/best_model.pth"),
    ("resnet3d_seed424", "configs/train_legacy_resnet3d_seed424.yaml",
     "Checkpoints_legacy_resnet3d_seed424/best_model.pth"),
]


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> None:
    state = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(state, dict) and not any(hasattr(v, "shape") for v in state.values()):
        for k in ("state_dict", "model_state_dict", "model"):
            if k in state and isinstance(state[k], dict):
                state = state[k]
                break
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    inc = model.load_state_dict(state, strict=False)
    n_total = len(list(model.state_dict().keys()))
    if len(inc.missing_keys) > n_total // 2:
        raise RuntimeError(f"{ckpt_path}: only {n_total - len(inc.missing_keys)}/{n_total} params loaded")
    print(f"  checkpoint {ckpt_path.name}: {n_total - len(inc.missing_keys)}/{n_total} params OK")


def autoregressive_pred(model: torch.nn.Module, data: torch.Tensor) -> torch.Tensor:
    """2-step autoregressive 4-frame prediction (predm from test_utility)."""
    pred1 = model(data)
    last2_inputs = data[:, :, 2:, :, :]
    first2_pred1 = pred1[:, :, :2, :, :]
    new_input = torch.cat([last2_inputs, first2_pred1], dim=2)
    pred2 = model(new_input)
    first2_pred2 = pred2[:, :, :2, :, :]
    return torch.cat([first2_pred1, first2_pred2], dim=2)


@torch.no_grad()
def eval_split(model, loader, alpha):
    """Mean per-batch combined loss for direct and autoregressive predictions."""
    model.eval()
    tot_direct = tot_auto = 0.0
    n = 0
    for batch in loader:
        data, targets = batch[:2]
        data = permute_to_bcthw(data).to(DEVICE)
        targets = permute_to_bcthw(targets).to(DEVICE)
        direct = model(data)
        auto = autoregressive_pred(model, data)
        tot_direct += weighted_mse_lpips_loss(direct, targets, alpha=alpha).item()
        tot_auto += weighted_mse_lpips_loss(auto, targets, alpha=alpha).item()
        n += 1
    return tot_direct / n, tot_auto / n


def main() -> None:
    out_dir = PROJECT_ROOT / "Model_Results_autoregressive_loss"
    out_dir.mkdir(exist_ok=True)
    rows = []

    loaders_cache = {}
    for label, cfg_path, ckpt in MODELS:
        print(f"\n=== {label} ===")
        cfg = load_config(PROJECT_ROOT / cfg_path, project_root=PROJECT_ROOT)
        cfg["mode"] = "infer"
        alpha = float(cfg["train"]["alpha"])
        # deterministic loaders, batch_size 4 as in training
        cfg.setdefault("dataloader", {})
        cfg["dataloader"].update(batch_size_train=4, batch_size_val=4, batch_size_test=4)
        cfg.setdefault("data", {})["split_out_dir"] = str(out_dir)

        split_seed = int(cfg["data"].get("split_seed", 424))
        if split_seed not in loaders_cache:
            train_loader, val_loader, _ = build_dataloaders(cfg)
            loaders_cache[split_seed] = (train_loader, val_loader)
        train_loader, val_loader = loaders_cache[split_seed]

        model = build_model(cfg["model"]).to(DEVICE)
        load_checkpoint(model, PROJECT_ROOT / ckpt)

        tr_direct, tr_auto = eval_split(model, train_loader, alpha)
        va_direct, va_auto = eval_split(model, val_loader, alpha)
        print(f"  TRAIN  direct={tr_direct:.5f}  autoregressive={tr_auto:.5f}")
        print(f"  VAL    direct={va_direct:.5f}  autoregressive={va_auto:.5f}")
        rows.append(dict(
            model=label, alpha=alpha,
            train_loss_direct=round(tr_direct, 6), train_loss_autoregressive=round(tr_auto, 6),
            val_loss_direct=round(va_direct, 6), val_loss_autoregressive=round(va_auto, 6),
            autoreg_gap_val=round(va_auto - va_direct, 6),
        ))
        del model
        torch.cuda.empty_cache()

    csv_path = out_dir / "autoregressive_loss.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWritten {csv_path}")


if __name__ == "__main__":
    main()
