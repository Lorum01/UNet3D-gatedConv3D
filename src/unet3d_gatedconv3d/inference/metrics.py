from __future__ import annotations

import csv
import math
import os
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torchmetrics.functional.image import structural_similarity_index_measure

from ..training.loss_function import weighted_mse_lpips_loss
from .test_utility import build_denorm, ensure_bcthw, load_checkpoint, to_device

# Rami di predizione valutati, stessi nomi/logica di test_model_create_gifs_3ch:
# "pred"  = predizione diretta (4 frame in -> 4 frame out, una sola inferenza)
# "predm" = predizione modificata: primi 2 frame da pred1, ultimi 2 frame da una
#           seconda inferenza autoregressiva (i primi 2 output vengono rimessi in input).
_BRANCHES = ("pred", "predm")

# Le immagini sono confrontate dopo denormalizzazione in [0,1] (vedi denorm() sotto).
_DATA_RANGE = 1.0


@torch.no_grad()
def evaluate_metrics_3d(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device | str,
    alpha: float = 0.7,
    lpips_input_mode: str = "none",
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
    scale_to_neg1_pos1: bool = False,
    checkpoint_path: Optional[str] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Valuta il modello su `dataloader` con lo stesso schema di inferenza 2-step
    autoregressivo di test_model_create_gifs_3ch (rami "pred" e "predm"),
    calcolando per ciascun ramo, come vera media pesata per campione sull'intero
    split valutato (non solo sull'ultimo batch), sia sull'intero orizzonte ("mean")
    sia per singolo timestep futuro ("t0".."t{T-1}") — utile per vedere quanto
    l'errore cresce con l'orizzonte di predizione / il rollout autoregressivo:
      - combined_loss: alpha*MSE + (1-alpha)*LPIPS per singolo frame, sui tensori
        normalizzati esattamente come in training/validation (stessa loss, stesso
        alpha, via weighted_mse_lpips_loss chiamata un frame alla volta cosi' da
        poterla scomporre per t senza toccare loss_function.py, condivisa col
        training). Sommata sui T frame con peso uniforme da' lo stesso identico
        risultato di chiamarla una volta sull'intero tensore (B,C,T,H,W).
      - mse / psnr / ssim: calcolate sui frame denormalizzati in [0,1] (stesso spazio
        usato per salvare le immagini).
        Il psnr e' derivato analiticamente dalla mse gia' aggregata (10*log10(1/mse)),
        non da una media di valori psnr calcolati per singolo batch: essendo il psnr
        una funzione non lineare (logaritmica) della mse, mediare valori gia' calcolati
        per batch NON darebbe la media sull'intero split, mentre la mse stessa e'
        un'operazione lineare quindi la sua media pesata per campione e' sempre esatta.
        Stesso discorso vale per combined_loss (MSE e LPIPS sono entrambe medie
        aritmetiche, quindi pesarle per B e dividere per n_samples e' esatto).

    Ritorna:
        {
          "n_batches": int, "n_samples": int,
          "pred":  {"combined_loss": {...}, "mse": {...}, "psnr": {...}, "ssim": {...}},
          "predm": {...},
        }
        dove ogni sotto-dict ha chiavi "t0".."t{T-1}" e "mean".
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device
    model = model.to(device)
    model = load_checkpoint(model, checkpoint_path, device)
    model.eval()

    denorm = build_denorm(mean, std, device, scale_to_neg1_pos1=scale_to_neg1_pos1)

    per_t_sums = {branch: None for branch in _BRANCHES}  # branch -> {"combined_loss"/"mse"/"ssim": [sum_per_t]}
    n_batches = 0
    n_samples = 0

    for batch in dataloader:
        if max_batches is not None and n_batches >= max_batches:
            break

        data, targets = batch[:2]
        data = ensure_bcthw(data, expected_channels=3, name="data")
        targets = ensure_bcthw(targets, expected_channels=3, name="targets")
        data = to_device(data, device)
        targets = to_device(targets, device)

        pred1 = model(data)  # (B,3,4,H,W)

        last2_inputs = data[:, :, 2:, :, :]
        first2_pred1 = pred1[:, :, :2, :, :]
        new_input = torch.cat([last2_inputs, first2_pred1], dim=2)
        pred2 = model(new_input)
        first2_pred2 = pred2[:, :, :2, :, :]
        pred_mod = torch.cat([first2_pred1, first2_pred2], dim=2)

        outputs = {"pred": pred1, "predm": pred_mod}

        B, _, T, _, _ = targets.shape
        n_batches += 1
        n_samples += B

        targets_d = denorm(targets)

        for branch, out in outputs.items():
            if per_t_sums[branch] is None:
                per_t_sums[branch] = {"combined_loss": [0.0] * T, "mse": [0.0] * T, "ssim": [0.0] * T}

            out_d = denorm(out)
            for t in range(T):
                out_t = out_d[:, :, t, :, :]
                tgt_t = targets_d[:, :, t, :, :]
                per_t_sums[branch]["mse"][t] += float(F.mse_loss(out_t, tgt_t).item()) * B
                # SSIM di torchmetrics fa una media lineare (pixel + batch), quindi
                # accumulare "valore_batch * B" e dividere per n_samples a fine loop
                # equivale esattamente a una media sull'intero split (verificato).
                per_t_sums[branch]["ssim"][t] += float(
                    structural_similarity_index_measure(out_t, tgt_t, data_range=_DATA_RANGE).item()
                ) * B

                # weighted_mse_lpips_loss chiamata su un solo frame (slice T=1) invece che
                # sull'intero tensore: stessa loss di training/validation, ma scomponibile
                # per timestep. targets NON denormalizzati qui: la combined_loss lavora
                # sempre nello spazio normalizzato, come in training.
                combined_t = weighted_mse_lpips_loss(
                    out[:, :, t:t + 1, :, :], targets[:, :, t:t + 1, :, :],
                    alpha=alpha, lpips_input_mode=lpips_input_mode,
                )
                per_t_sums[branch]["combined_loss"][t] += float(combined_t.item()) * B

    result: Dict[str, Any] = {"n_batches": n_batches, "n_samples": n_samples}
    for branch in _BRANCHES:
        if per_t_sums[branch] is None:
            continue
        branch_result: Dict[str, Any] = {}
        for metric_name, sums_per_t in per_t_sums[branch].items():
            T = len(sums_per_t)
            per_t_avg = [s / max(n_samples, 1) for s in sums_per_t]
            metric_dict = {f"t{t}": per_t_avg[t] for t in range(T)}
            metric_dict["mean"] = sum(per_t_avg) / T
            branch_result[metric_name] = metric_dict

        # PSNR derivato analiticamente dalla MSE gia' mediata sull'intero split
        # (non da una media di PSNR-per-batch: essendo PSNR = 10*log10(1/MSE) una
        # funzione non lineare della MSE, mediare valori di PSNR gia' calcolati per
        # batch NON equivale alla media sugli elementi dello split; farlo cosi'
        # invece e' esatto, perche' la MSE stessa e' gia' una media lineare corretta).
        mse_dict = branch_result["mse"]
        psnr_dict = {
            key: 10.0 * math.log10((_DATA_RANGE ** 2) / max(mse_val, 1e-12))
            for key, mse_val in mse_dict.items()
        }
        branch_result["psnr"] = psnr_dict

        result[branch] = branch_result

    return result


def save_metrics_csv(metrics: Dict[str, Any], csv_path: str) -> None:
    """Salva il dict di evaluate_metrics_3d in un CSV: una riga per (branch, metric)."""
    out_dir = os.path.dirname(csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    t_keys: list[str] = []
    for branch in _BRANCHES:
        if branch in metrics:
            t_keys = sorted(
                (k for k in metrics[branch]["mse"].keys() if k != "mean"),
                key=lambda k: int(k[1:]),
            )
            break

    def _fmt(x: float) -> str:
        return f"{x:.3f}"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["branch", "metric"] + t_keys + ["mean"])
        for branch in _BRANCHES:
            if branch not in metrics:
                continue
            for metric_name in ("combined_loss", "mse", "psnr", "ssim"):
                row = metrics[branch][metric_name]
                writer.writerow([branch, metric_name] + [_fmt(row[k]) for k in t_keys] + [_fmt(row["mean"])])
