"""Metriche quantitative su una cartella di risultati di inferenza.

Legge i .npy gia' denormalizzati in [0,1] salvati da `test_model_create_gifs_3ch`
(target_t{t}_*.npy, pred_*.npy, predm_*.npy) e calcola, frame per frame (lag 1..T):

    MSE, MSE+LPIPS (alpha), SSIM, precision/recall/F1        (+ LPIPS, PSNR a video)

Aggregazione identica a quella usata per VONA_all_metrics.xlsx:
  - per ogni evento si fa la media sui suoi campioni (per lag);
  - media / std(ddof=1) / min / max sono calcolate SULLE MEDIE PER EVENTO;
  - riga `lag=mean`: media sui lag 1..T (le 4 colonne stat riportano lo stesso valore).

Scrive `<results_dir>/metrics.csv` nello schema:
  prediction_type,lag,<key>,<key>_std,<key>_min,<key>_max,...,event_count
con key in (mse, mse_lpips, ssim, precision, recall, f1) e prediction_type in
(pred, predm).

Uso:
    python scripts/compute_metrics.py Model_Results_test2025/remap01/by_class --alpha 0.7
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

# Ordine con cui le metriche finiscono nel CSV (schema VONA_all_metrics).
_METRIC_KEYS = ("mse", "mse_lpips", "ssim", "precision", "recall", "f1")
_EVENT_RE = re.compile(r"(\d{4}_\d{2}_\d{2})")


# --------------------------- SSIM (finestra gaussiana) ---------------------------
def _gaussian_window(window_size, sigma, channels, device, dtype):
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(1)
    window_2d = g @ g.t()
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def _ssim_frame(pred, target, data_range=1.0, window_size=11, sigma=1.5):
    B, C, H, W = pred.shape
    window = _gaussian_window(window_size, sigma, C, pred.device, pred.dtype)
    pad = window_size // 2
    mu_p = F.conv2d(pred, window, padding=pad, groups=C)
    mu_t = F.conv2d(target, window, padding=pad, groups=C)
    mu_p2, mu_t2, mu_pt = mu_p.pow(2), mu_t.pow(2), mu_p * mu_t
    sig_p2 = F.conv2d(pred * pred, window, padding=pad, groups=C) - mu_p2
    sig_t2 = F.conv2d(target * target, window, padding=pad, groups=C) - mu_t2
    sig_pt = F.conv2d(pred * target, window, padding=pad, groups=C) - mu_pt
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_pt + c1) * (2 * sig_pt + c2)) / (
        (mu_p2 + mu_t2 + c1) * (sig_p2 + sig_t2 + c2))
    return ssim_map.mean().item()


# --------------------------- LPIPS (lazy) ---------------------------
_LPIPS_MODEL = None


def _get_lpips(device):
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        try:
            import contextlib
            import io
            import lpips
        except ModuleNotFoundError:
            return None
        with contextlib.redirect_stdout(io.StringIO()):
            _LPIPS_MODEL = lpips.LPIPS(net="alex", version="0.1").to(device)
        _LPIPS_MODEL.eval()
    return _LPIPS_MODEL


# --------------------------- IO helper ---------------------------
def _load_hwc_as_chw(path):
    a = np.load(path).astype(np.float32)      # (H, W, C)
    return np.transpose(a, (2, 0, 1))         # (C, H, W)


def collect_samples(results_dir):
    out = []
    for root, _dirs, files in os.walk(results_dir):
        if os.path.basename(root).startswith("sample_") and any(f.startswith("target_t") for f in files):
            out.append(root)
    return sorted(out)


def build_frame_to_event(dataset_folder):
    """{stem_del_frame -> cartella-evento} dal dataset originale.

    Serve perche' alcune cartelle-evento contengono frame di due date di calendario
    (eruzioni a cavallo di mezzanotte): l'evento e' la CARTELLA, non la data nel
    nome file. `_flipped` viene rimosso dal nome evento.
    """
    mapping = {}
    if not dataset_folder or not os.path.isdir(dataset_folder):
        return mapping
    for ev in os.listdir(dataset_folder):
        p = os.path.join(dataset_folder, ev)
        if not os.path.isdir(p):
            continue
        ev_name = ev[:-len("_flipped")] if ev.endswith("_flipped") else ev
        for f in os.listdir(p):
            mapping[os.path.splitext(f)[0]] = ev_name
    return mapping


def _seq_from_sample(sample_dir, pred_prefix, frame_to_event=None):
    """(pred (1,C,T,H,W), target (1,C,T,H,W), event_key) oppure None."""
    files = os.listdir(sample_dir)
    targets = {}
    for f in files:
        m = re.match(r"target_t(\d+)_(.+)\.npy$", f)
        if m:
            targets[int(m.group(1))] = (m.group(2), f)
    if not targets:
        return None
    preds = {os.path.splitext(f)[0][len(pred_prefix):]: f
             for f in files if f.startswith(pred_prefix) and f.endswith(".npy")}

    T = max(targets) + 1
    tgt_seq, pred_seq, event_key = [], [], None
    for t in range(T):
        suffix, tfile = targets[t]
        if suffix not in preds:
            return None
        tgt_seq.append(_load_hwc_as_chw(os.path.join(sample_dir, tfile)))
        pred_seq.append(_load_hwc_as_chw(os.path.join(sample_dir, preds[suffix])))
        if event_key is None:
            if frame_to_event and suffix in frame_to_event:
                event_key = frame_to_event[suffix]
            else:
                mm = _EVENT_RE.search(suffix)
                event_key = mm.group(1) if mm else "unknown"

    tgt = torch.from_numpy(np.stack(tgt_seq, axis=1)).unsqueeze(0)
    pred = torch.from_numpy(np.stack(pred_seq, axis=1)).unsqueeze(0)
    return pred, tgt, event_key


def frame_metrics(pred01, target01, alpha, threshold, device):
    """(1,C,T,H,W) in [0,1] -> dict metric -> lista per lag. Include lpips/psnr."""
    pred01 = pred01.to(device).clamp(0, 1)
    target01 = target01.to(device).clamp(0, 1)
    _, _, T, _, _ = pred01.shape
    lpips_model = _get_lpips(device)

    res = defaultdict(list)
    with torch.no_grad():
        for t in range(T):
            p, g = pred01[:, :, t], target01[:, :, t]
            mse = F.mse_loss(p, g).item()
            res["mse"].append(mse)
            res["psnr"].append(float("inf") if mse == 0 else 10.0 * float(np.log10(1.0 / mse)))
            res["ssim"].append(_ssim_frame(p, g, data_range=1.0))

            pb, gb = p >= threshold, g >= threshold
            tp = (pb & gb).sum().item()
            fp = (pb & ~gb).sum().item()
            fn = (~pb & gb).sum().item()
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            res["precision"].append(prec)
            res["recall"].append(rec)
            res["f1"].append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)

            lp = lpips_model(p * 2 - 1, g * 2 - 1).mean().item() if lpips_model is not None else float("nan")
            res["lpips"].append(lp)
            res["mse_lpips"].append(alpha * mse + (1 - alpha) * lp)
    return res


# --------------------------- aggregazione stile VONA ---------------------------
def _std(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5


def summarize(event_lag_values, T):
    """event_lag_values[event][t] = lista di valori (uno per campione).
    Ritorna (per_lag_stats, mean_row) dove per_lag_stats[t] = (mean,std,min,max)
    calcolati sulle medie per evento; mean_row = (v,v,v,v) con v = media sui lag."""
    events = sorted(event_lag_values)
    per_lag = []
    for t in range(T):
        ev_means = [sum(v) / len(v) for e in events if (v := event_lag_values[e][t])]
        if not ev_means:
            per_lag.append((float("nan"),) * 4)
            continue
        m = sum(ev_means) / len(ev_means)
        per_lag.append((m, _std(ev_means), min(ev_means), max(ev_means)))
    valid_means = [s[0] for s in per_lag if np.isfinite(s[0])]
    mv = sum(valid_means) / len(valid_means) if valid_means else float("nan")
    return per_lag, (mv, mv, mv, mv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--alpha", type=float, default=0.7,
                    help="peso MSE nella loss combinata usato per mse_lpips (= alpha di training)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="default: <results_dir>/metrics.csv")
    ap.add_argument("--dataset-folder", default=None,
                    help="dataset originale: risolve l'evento dalla CARTELLA (non dalla "
                         "data nel nome file, che sdoppia le eruzioni a cavallo di mezzanotte)")
    args = ap.parse_args()

    device = torch.device(args.device)
    sample_dirs = collect_samples(args.results_dir)
    if not sample_dirs:
        raise SystemExit(f"Nessuna cartella 'sample_*' sotto {args.results_dir}")

    frame_to_event = build_frame_to_event(args.dataset_folder)
    if args.dataset_folder:
        print(f"  evento risolto dalla cartella dataset ({len(frame_to_event)} frame mappati)")

    # acc[pt][key][event][t] = lista valori (uno per campione)
    def _acc():
        return defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    acc = {"pred": _acc(), "predm": _acc()}
    events_seen = set()
    T_global = 0
    n_ok = 0

    for sd in sample_dirs:
        got = {pt: _seq_from_sample(sd, pt + "_", frame_to_event) for pt in ("pred", "predm")}
        if got["pred"] is None:
            print(f"[skip] {sd}")
            continue
        n_ok += 1
        for pt, g in got.items():
            if g is None:
                continue
            pred, tgt, ev = g
            events_seen.add(ev)
            fm = frame_metrics(pred, tgt, args.alpha, args.threshold, device)
            T_global = max(T_global, len(fm["mse"]))
            for key, vals in fm.items():
                for t, v in enumerate(vals):
                    acc[pt][key][ev][t].append(v)

    print(f"\n{args.results_dir}")
    print(f"  campioni: {n_ok}/{len(sample_dirs)}  |  eventi: {sorted(events_seen)}  |  "
          f"lag: {T_global}  |  alpha: {args.alpha}  |  soglia: {args.threshold}")

    out_path = args.out or os.path.join(args.results_dir, "metrics.csv")
    header = ["prediction_type", "lag"]
    for k in _METRIC_KEYS:
        header += [k, f"{k}_std", f"{k}_min", f"{k}_max"]
    header.append("event_count")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for pt in ("pred", "predm"):
            summ = {k: summarize(acc[pt][k], T_global) for k in _METRIC_KEYS}
            for t in range(T_global):
                row = [pt, t + 1]
                for k in _METRIC_KEYS:
                    row += [f"{x:.6f}" for x in summ[k][0][t]]
                row.append(len(events_seen))
                w.writerow(row)
            row = [pt, "mean"]
            for k in _METRIC_KEYS:
                row += [f"{x:.6f}" for x in summ[k][1]]
            row.append(len(events_seen))
            w.writerow(row)

    # riepilogo a video (pred, riga mean) + lpips/psnr informativi
    lp_summ = summarize(acc["pred"]["lpips"], T_global)[1][0]
    ps_summ = summarize(acc["pred"]["psnr"], T_global)[1][0]
    print("  pred  (media eventi, media lag):  " + "  ".join(
        f"{k}={summarize(acc['pred'][k], T_global)[1][0]:.5f}" for k in _METRIC_KEYS)
        + f"  | lpips={lp_summ:.5f}  psnr={ps_summ:.3f}")
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
