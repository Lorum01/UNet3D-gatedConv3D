from __future__ import annotations

import csv
import math
import os
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


_SPLIT_MERGE_ORDER = {"train": 0, "val": 1, "test": 2}
_METRIC_MERGE_ORDER = {"combined_loss": 0, "mse": 1, "psnr": 2, "ssim": 3}


def merge_metrics_csv(root: str, out_csv_path: Optional[str] = None) -> int:
    """Unisce tutti i metrics.csv trovati ricorsivamente sotto `root` (uno per
    ogni cartella split, formato scritto da save_metrics_csv) in un unico CSV
    <root>/all_metrics.csv (o out_csv_path se specificato), aggiungendo le colonne
    `run` (cartella del run, genitore della cartella split) e `split`. Non tocca
    i metrics.csv originali. Le righe sono ordinate per run, split, branch, metric.

    Richiamata sia da scripts/merge_inference_metrics.py (CLI) sia automaticamente
    a fine pipeline di inferenza, cosi' che il CSV aggregato resti sempre aggiornato
    senza doverlo rigenerare a mano.

    Ritorna il numero di righe scritte (0 se nessun metrics.csv trovato).
    """
    root_path = Path(root)
    out_path = Path(out_csv_path) if out_csv_path else root_path / "all_metrics.csv"
    files = sorted(p for p in root_path.rglob("metrics.csv") if p.resolve() != out_path.resolve())
    if not files:
        return 0

    t_cols: List[str] = []
    rows: List[Dict[str, str]] = []
    for f in files:
        run_name = f.parent.parent.relative_to(root_path).as_posix()
        split = f.parent.name
        with open(f, "r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if not t_cols:
                t_cols = [c for c in reader.fieldnames or [] if c not in ("branch", "metric")]
            for row in reader:
                rows.append({"run": run_name, "split": split, **row})

    rows.sort(key=lambda r: (
        r["run"],
        _SPLIT_MERGE_ORDER.get(r["split"], 99),
        _BRANCHES.index(r["branch"]) if r["branch"] in _BRANCHES else 99,
        _METRIC_MERGE_ORDER.get(r["metric"], 99),
    ))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["run", "split", "branch", "metric"] + t_cols)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def aggregate_metrics_across_seeds(
    seed_csv_paths: Dict[int, str],
    out_csv_path: str,
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Aggrega piu' metrics.csv (uno per ogni run di training con train.seed diverso ma
    stesso identico dataset.split.seed, quindi stesso split) calcolando media e
    deviazione standard CAMPIONARIA (ddof=1; 0.0 se un solo seed disponibile) tra i
    seed, per ciascuna combinazione (branch, metric, colonna t0..t{T-1}/mean).

    seed_csv_paths: {seed: path/to/metrics.csv}, ognuno nel formato scritto da
    save_metrics_csv (una riga per branch/metric, colonne t0..t{T-1},mean).

    Scrive out_csv_path con colonne branch,metric,stat,t0..t{T-1},mean: due righe per
    (branch,metric), "seed_mean" e "seed_std". Ritorna lo stesso aggregato in memoria:
    {branch: {metric: {column: {"seed_mean", "seed_std", "n_seeds"}}}}.
    """
    seeds = sorted(seed_csv_paths.keys())
    if not seeds:
        raise ValueError("seed_csv_paths e' vuoto: nessun seed da aggregare.")

    per_seed_rows: Dict[int, Dict[Tuple[str, str], Dict[str, float]]] = {}
    columns: list[str] = []
    branch_metric_keys: list[Tuple[str, str]] = []

    for seed in seeds:
        with open(seed_csv_paths[seed], "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not columns:
                columns = [c for c in (reader.fieldnames or []) if c not in ("branch", "metric")]
            rows: Dict[Tuple[str, str], Dict[str, float]] = {}
            for row in reader:
                key = (row["branch"], row["metric"])
                rows[key] = {c: float(row[c]) for c in columns}
                if key not in branch_metric_keys:
                    branch_metric_keys.append(key)
        per_seed_rows[seed] = rows

    out_dir = os.path.dirname(out_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    aggregated: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["# seeds", ",".join(str(s) for s in seeds), f"n={len(seeds)}"])
        writer.writerow(["branch", "metric", "stat"] + columns)
        for branch, metric in branch_metric_keys:
            mean_row, std_row = [], []
            for col in columns:
                values = [
                    per_seed_rows[s][(branch, metric)][col]
                    for s in seeds
                    if (branch, metric) in per_seed_rows[s]
                ]
                m = statistics.mean(values)
                sd = statistics.stdev(values) if len(values) > 1 else 0.0
                aggregated.setdefault(branch, {}).setdefault(metric, {})[col] = {
                    "seed_mean": m,
                    "seed_std": sd,
                    "n_seeds": float(len(values)),
                }
                mean_row.append(f"{m:.4f}")
                std_row.append(f"{sd:.4f}")
            writer.writerow([branch, metric, "seed_mean"] + mean_row)
            writer.writerow([branch, metric, "seed_std"] + std_row)

    return aggregated
