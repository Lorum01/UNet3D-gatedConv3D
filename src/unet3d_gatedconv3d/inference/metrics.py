"""Metriche quantitative di inferenza: MSE, MSE+LPIPS bilanciata (alpha) e SSIM.

Calcolate frame per frame (dimensione T) su tensori gia' denormalizzati in [0,1]
(la stessa rappresentazione salvata come jpg/gif in test_utility.py), cosi' i
valori sono confrontabili tra run con normalizzazioni diverse (none/neg1pos1/
standardize).
"""

import torch
import torch.nn.functional as F
from collections import defaultdict

from ..training.loss_function import _get_lpips_model, _prepare_lpips_input

# Metriche accumulate qui, nell'ordine in cui vengono scritte nel CSV finale.
_METRIC_KEYS = ("mse", "mse_lpips", "ssim", "precision", "recall", "f1")


def _gaussian_window(window_size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(1)
    window_2d = g @ g.t()
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def _ssim_frame(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0,
                 window_size: int = 11, sigma: float = 1.5) -> float:
    """SSIM (formulazione standard, finestra gaussiana) mediata su batch/canali/spazio."""
    B, C, H, W = pred.shape
    window = _gaussian_window(window_size, sigma, C, pred.device, pred.dtype)
    pad = window_size // 2

    mu_pred = F.conv2d(pred, window, padding=pad, groups=C)
    mu_target = F.conv2d(target, window, padding=pad, groups=C)
    mu_pred_sq, mu_target_sq = mu_pred.pow(2), mu_target.pow(2)
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred * pred, window, padding=pad, groups=C) - mu_pred_sq
    sigma_target_sq = F.conv2d(target * target, window, padding=pad, groups=C) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, window, padding=pad, groups=C) - mu_pred_target

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)) / (
        (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
    )
    return ssim_map.mean().item()


def compute_frame_metrics(pred01: torch.Tensor, target01: torch.Tensor,
                           alpha: float = 0.7, lpips_input_mode: str = "none",
                           threshold: float = 0.5) -> dict:
    """
    Calcola MSE, LPIPS, MSE+LPIPS, SSIM e metriche binarie per ciascun timestep.

    Args:
        pred01, target01: tensori (B, C, T, H, W) gia' denormalizzati in [0,1].
        alpha: peso della MSE nella combinazione (1-alpha per LPIPS), coerente con
            l'alpha usato in training per il modello valutato.
        lpips_input_mode: come preparare `pred01` (portato a [-1,1]) prima di LPIPS,
            vedi training.loss_function._prepare_lpips_input ("none"/"clamp"/"tanh").
        threshold: soglia per binarizzare predizione e target nelle metriche pixel-level.

    Returns:
        dict con, per ciascuna chiave in ("mse", "lpips", "mse_lpips", "ssim",
        "precision", "recall", "f1"),
        una lista di T valori (uno per timestep) e la relativa media su T
        (chiave f"{nome}_mean").
    """
    _, _, T, _, _ = pred01.shape
    device_str = str(pred01.device)

    try:
        lpips_model = _get_lpips_model(device_str)
    except ModuleNotFoundError:
        lpips_model = None

    mse_t, ssim_t, lpips_t = [], [], []
    precision_t, recall_t, f1_t = [], [], []
    with torch.no_grad():
        for t in range(T):
            p, g = pred01[:, :, t, :, :], target01[:, :, t, :, :]

            mse_t.append(F.mse_loss(p, g).item())
            ssim_t.append(_ssim_frame(p, g, data_range=1.0))

            pred_binary = p >= threshold
            target_binary = g >= threshold
            true_positive = (pred_binary & target_binary).sum().item()
            false_positive = (pred_binary & ~target_binary).sum().item()
            false_negative = (~pred_binary & target_binary).sum().item()
            precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
            recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
            precision_t.append(precision)
            recall_t.append(recall)
            f1_t.append(f1)

            if lpips_model is not None:
                p_lpips = _prepare_lpips_input(p * 2.0 - 1.0, lpips_input_mode)
                g_lpips = g * 2.0 - 1.0
                lpips_t.append(lpips_model(p_lpips, g_lpips).mean().item())
            else:
                lpips_t.append(float("nan"))

    mse_lpips_t = [alpha * m + (1 - alpha) * l for m, l in zip(mse_t, lpips_t)]

    def _mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    return {
        "mse": mse_t, "mse_mean": _mean(mse_t),
        "lpips": lpips_t, "lpips_mean": _mean(lpips_t),
        "mse_lpips": mse_lpips_t, "mse_lpips_mean": _mean(mse_lpips_t),
        "ssim": ssim_t, "ssim_mean": _mean(ssim_t),
        "precision": precision_t, "precision_mean": _mean(precision_t),
        "recall": recall_t, "recall_mean": _mean(recall_t),
        "f1": f1_t, "f1_mean": _mean(f1_t),
    }


class MetricsAccumulator:
    """Accumula metriche per lag, dando lo stesso peso a ogni evento."""

    def __init__(self, num_timesteps: int, alpha: float = 0.7, lpips_input_mode: str = "none",
                 threshold: float = 0.5):
        self.num_timesteps = num_timesteps
        self.alpha = alpha
        self.lpips_input_mode = lpips_input_mode
        self.threshold = threshold
        self._event_values = {
            k: defaultdict(lambda: [[] for _ in range(num_timesteps)])
            for k in _METRIC_KEYS
        }
        self.n_samples = 0

    def update(self, pred01: torch.Tensor, target01: torch.Tensor, event_names=None) -> None:
        """Aggiunge un batch associando ogni sample al relativo evento."""
        B = pred01.shape[0]
        if event_names is None:
            event_names = [f"sample_{self.n_samples + i}" for i in range(B)]
        if len(event_names) != B:
            raise ValueError("event_names deve contenere un nome per ogni sample del batch")

        for sample_idx, event_name in enumerate(event_names):
            sample = compute_frame_metrics(
                pred01[sample_idx:sample_idx + 1], target01[sample_idx:sample_idx + 1],
                alpha=self.alpha, lpips_input_mode=self.lpips_input_mode, threshold=self.threshold,
            )
            event_name = str(event_name)
            for key in _METRIC_KEYS:
                for t in range(self.num_timesteps):
                    self._event_values[key][event_name][t].append(sample[key][t])
        self.n_samples += B

    def per_timestep(self) -> dict:
        event_names = self.event_names()
        if not event_names:
            return {k: [float("nan")] * self.num_timesteps for k in _METRIC_KEYS}
        result = {}
        for key in _METRIC_KEYS:
            values_by_timestep = []
            for t in range(self.num_timesteps):
                event_means = [
                    sum(values) / len(values)
                    for event_name in event_names
                    if (values := self._event_values[key][event_name][t])
                ]
                values_by_timestep.append(
                    sum(event_means) / len(event_means) if event_means else float("nan")
                )
            result[key] = values_by_timestep
        return result

    def event_names(self):
        return sorted({
            event_name
            for key in _METRIC_KEYS
            for event_name in self._event_values[key]
        })

    def std_per_timestep(self) -> dict:
        """Restituisce la deviazione standard campionaria tra le medie degli eventi."""
        event_names = self.event_names()
        result = {}
        for key in _METRIC_KEYS:
            values_by_timestep = []
            for t in range(self.num_timesteps):
                event_means = [
                    sum(values) / len(values)
                    for event_name in event_names
                    if (values := self._event_values[key][event_name][t])
                ]
                if len(event_means) < 2:
                    values_by_timestep.append(0.0 if event_means else float("nan"))
                    continue
                mean = sum(event_means) / len(event_means)
                variance = sum((value - mean) ** 2 for value in event_means) / (len(event_means) - 1)
                values_by_timestep.append(variance ** 0.5)
            result[key] = values_by_timestep
        return result

    def summary_per_timestep(self) -> dict:
        """Restituisce media, deviazione standard, minimo e massimo tra gli eventi."""
        event_names = self.event_names()
        result = {stat: {} for stat in ("mean", "std", "min", "max")}
        for key in _METRIC_KEYS:
            result["mean"][key] = []
            result["std"][key] = []
            result["min"][key] = []
            result["max"][key] = []
            for t in range(self.num_timesteps):
                event_means = [
                    sum(values) / len(values)
                    for event_name in event_names
                    if (values := self._event_values[key][event_name][t])
                ]
                if not event_means:
                    for stat in result:
                        result[stat][key].append(float("nan"))
                    continue
                mean = sum(event_means) / len(event_means)
                variance = (
                    sum((value - mean) ** 2 for value in event_means) / (len(event_means) - 1)
                    if len(event_means) > 1 else 0.0
                )
                result["mean"][key].append(mean)
                result["std"][key].append(variance ** 0.5)
                result["min"][key].append(min(event_means))
                result["max"][key].append(max(event_means))
        return result

    def overall_mean(self) -> dict:
        per_t = self.per_timestep()
        return {k: (sum(vals) / len(vals) if vals else float("nan")) for k, vals in per_t.items()}


def write_metrics_csv(csv_path: str, accumulators: dict) -> None:
    """Scrive metriche per lag e deviazioni standard tra eventi in un CSV.

    Include una riga
    "mean" finale per ciascun prediction_type in `accumulators` (es. {"pred": acc, "predm": acc})."""
    import csv

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        metric_columns = []
        for key in _METRIC_KEYS:
            metric_columns.extend((key, f"{key}_std", f"{key}_min", f"{key}_max"))
        writer.writerow(["prediction_type", "lag", *metric_columns, "event_count"])
        for pred_type, acc in accumulators.items():
            summary = acc.summary_per_timestep()
            event_count = len(acc.event_names())
            for t in range(acc.num_timesteps):
                values = []
                for key in _METRIC_KEYS:
                    values.extend(f"{summary[stat][key][t]:.6f}" for stat in ("mean", "std", "min", "max"))
                writer.writerow([
                    pred_type, t + 1, *values, event_count,
                ])
            mean = {
                key: sum(summary["mean"][key]) / len(summary["mean"][key])
                for key in _METRIC_KEYS
            }
            mean_values = []
            for key in _METRIC_KEYS:
                mean_values.extend(f"{mean[key]:.6f}" for _ in range(4))
            writer.writerow([
                pred_type, "mean", *mean_values, event_count,
            ])
