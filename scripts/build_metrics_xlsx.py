"""Raccoglie i metrics.csv delle run di inferenza di QUESTO workspace in un unico
xlsx, con la stessa struttura di vona_backup_branch/VONA_all_metrics.xlsx:

    Summary      metriche principali (mse, ssim, precision, recall, f1) per
                 modello / prediction_type / lag 1-4
    All metrics  tutte le righe dei CSV (lag 1-4 + mean), inclusa mse_lpips
    Sources      CSV di origine e righe importate
    Read me      definizioni

Uso:
    python scripts/build_metrics_xlsx.py \
        --base Model_Results_test2025 --out Model_Results_test2025/legacy_test2025_metrics.xlsx
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

GROUP = "vona_legacy_window_split"

# dir della run -> etichetta leggibile nel foglio (ordine = ordine nei fogli)
MODEL_LABELS = {
    # --- test 2025 (subdir by_class) ---
    "seed42_alpha0_remap": "seed42_alpha0.0",
    "seed42_alpha05_remap": "seed42_alpha0.5",
    "seed42_remap": "seed42_alpha0.7",
    "seed42_alpha07_remap": "seed42_alpha0.7",
    "seed42_alpha1_remap": "seed42_alpha1.0",
    "seed424_alpha0_remap": "seed424_alpha0.0",
    "seed424_alpha05_remap": "seed424_alpha0.5",
    "seed424_remap": "seed424_alpha0.7",
    "seed424_alpha1_remap": "seed424_alpha1.0",
    "legacy07_remap": "legacy_07 (checkpoint esterno)",
    "legacy_encdec_remap": "encdec_gatedconv3d (seed42, a0.7)",
    "legacy_resnet_remap": "resnet3d (seed42, a0.7)",
    "legacy_encdec_seed424_remap": "encdec_gatedconv3d (seed424, a0.7)",
    "legacy_resnet_seed424_remap": "resnet3d (seed424, a0.7)",
    # --- test split proprio del modello (subdir test) ---
    "seed42_alpha0": "seed42_alpha0.0 (seed42 test split)",
    "seed42_alpha05": "seed42_alpha0.5 (seed42 test split)",
    "seed42_alpha1": "seed42_alpha1.0 (seed42 test split)",
    "seed424_alpha0": "seed424_alpha0.0 (seed424 test split)",
    "seed424_alpha05": "seed424_alpha0.5 (seed424 test split)",
    "seed424_alpha1": "seed424_alpha1.0 (seed424 test split)",
    "legacy07_on_seed42split": "legacy_07 (seed42 test split)",
    "legacy07_on_seed424split": "legacy_07 (seed424 test split)",
    "legacy_encdec": "encdec_gatedconv3d a0.7 (seed42 test split)",
    "legacy_resnet": "resnet3d a0.7 (seed42 test split)",
    "legacy_encdec_seed424": "encdec_gatedconv3d a0.7 (seed424 test split)",
    "legacy_resnet_seed424": "resnet3d a0.7 (seed424 test split)",
}
# le due run "seed42_remap"/"seed424_remap" nel testsplit sono alpha 0.7:
MODEL_LABELS_TESTSPLIT = {
    "seed42_remap": "seed42_alpha0.7 (seed42 test split)",
    "seed424_remap": "seed424_alpha0.7 (seed424 test split)",
}

# split di validation seed 424 (subdir val)
MODEL_LABELS_VALSPLIT = {
    "seed424_alpha0": "seed424_alpha0.0",
    "seed424_alpha05": "seed424_alpha0.5",
    "seed424_alpha07": "seed424_alpha0.7",
    "seed424_alpha1": "seed424_alpha1.0",
    "legacy_encdec_seed424": "encdec_gatedconv3d (seed424, a0.7)",
    "legacy_resnet_seed424": "resnet3d (seed424, a0.7)",
}

_STAT_KEYS_ALL = ("mse", "mse_lpips", "ssim", "precision", "recall", "f1")
_STAT_KEYS_SUMMARY = ("mse", "ssim", "precision", "recall", "f1")


def _cols_for(keys):
    out = []
    for k in keys:
        out += [k, f"{k}_std", f"{k}_min", f"{k}_max"]
    return out


def _lag_sort_key(v):
    return (1, 0) if str(v) == "mean" else (0, int(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Model_Results_test2025")
    ap.add_argument("--out", default=None)
    ap.add_argument("--subdir", default="by_class",
                    help="sottocartella con metrics.csv: 'by_class' (test 2025), 'test' (split di test), 'val' (split di validation)")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="sottocartelle run da includere (default: tutte quelle con metrics.csv)")
    args = ap.parse_args()

    base = os.path.abspath(args.base)
    out_path = args.out or os.path.join(base, "legacy_test2025_metrics.xlsx")
    labels = dict(MODEL_LABELS)
    if args.subdir == "test":
        labels.update(MODEL_LABELS_TESTSPLIT)
    elif args.subdir == "val":
        labels = dict(MODEL_LABELS_VALSPLIT)

    csvs = sorted(glob.glob(os.path.join(base, "*", args.subdir, "metrics.csv")))
    if args.runs:
        wanted = set(args.runs)
        csvs = [c for c in csvs if c.split(os.sep)[-3] in wanted]
    else:
        # default: escludi le run 'clamp' (interessa solo il remap)
        csvs = [c for c in csvs if "clamp" not in c.split(os.sep)[-3].lower()]
    if not csvs:
        raise SystemExit(f"Nessun metrics.csv sotto {base}/*/by_class/")

    def _order(csv_path):
        run = csv_path.split(os.sep)[-3]
        keys = list(labels)
        return (keys.index(run) if run in keys else len(keys), run)
    csvs.sort(key=_order)

    all_rows, summary_rows, sources = [], [], []
    for csv_path in csvs:
        run = csv_path.split(os.sep)[-3]
        model = labels.get(run, run)
        df = pd.read_csv(csv_path)
        df.insert(0, "model", model)
        df.insert(0, "group", GROUP)
        all_rows.append(df)

        summ = df[df["lag"] != "mean"].copy()
        summ["lag"] = summ["lag"].astype(int)
        summary_rows.append(summ[["group", "model", "prediction_type", "lag",
                                  *_cols_for(_STAT_KEYS_SUMMARY), "event_count"]])

        sources.append({
            "group": GROUP, "model": model,
            "source_csv": os.path.relpath(csv_path, base),
            "rows_imported": len(df),
            "original_lag_column": 1,
        })

    all_df = pd.concat(all_rows, ignore_index=True)
    all_df = all_df[["group", "model", "prediction_type", "lag",
                     *_cols_for(_STAT_KEYS_ALL), "event_count"]]
    all_df["_k"] = all_df["lag"].map(_lag_sort_key)
    all_df = all_df.sort_values(["group", "model", "prediction_type", "_k"]).drop(columns="_k")

    summary_df = pd.concat(summary_rows, ignore_index=True).sort_values(
        ["group", "model", "prediction_type", "lag"])

    sources_df = pd.DataFrame(sources)

    if args.subdir == "test":
        scope_line = ("Ogni modello e' valutato sul PROPRIO split di test a livello di finestra "
                      "(class_pct 70/15/15, split_seed del modello; 143 finestre, "
                      "59 eventi per seed42 / 65 per seed424; evento = cartella del dataset). "
                      "legacy_07 non ha uno split proprio: e' valutato su entrambi (seed42 e seed424). "
                      "Output rimappato (x+1)/2 -> [0,1].")
        title = "VONA legacy — metriche sullo split di test proprio"
    elif args.subdir == "val":
        scope_line = ("Ogni modello e' valutato sul suo split di VALIDATION a livello di finestra "
                      "(class_pct 70/15/15, split_seed 424; 144 finestre; evento = cartella del dataset). "
                      "Output rimappato (x+1)/2 -> [0,1].")
        title = "VONA legacy — metriche sullo split di validation (seed 424)"
    else:
        scope_line = ("Tutte le run: inferenza sulle 2 serie 2025 (2025_06_02, 2025_12_27), "
                      "output rimappato (x+1)/2 -> [0,1].")
        title = "VONA legacy — metriche test 2025"

    readme = pd.DataFrame({title: [
        "",
        "Contenuto",
        "Summary: mse, ssim, precision, recall, f1 per modello / prediction_type / lag 1-4.",
        "All metrics: tutte le righe dei CSV test (lag 1-4 + mean), inclusa mse_lpips.",
        "Sources: percorsi dei CSV utilizzati e righe importate.",
        "",
        "Definizioni",
        scope_line,
        "std, min, max sono calcolate sulle medie per evento (std campionaria ddof=1).",
        "mse_lpips = alpha*mse + (1-alpha)*lpips, con alpha = quello di training del modello.",
        "Precision, recall, F1 sono pixel-level, soglia 0.5, canali RGB aggregati.",
        "pred = predizione diretta 4-step; predm = predizione modificata (2+2 autoregressiva): lag 1-2 coincidono.",
        f"group = {GROUP} (workspace /home/basile/workdir/vona_legacy_window_split).",
    ]})

    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        summary_df.to_excel(xw, sheet_name="Summary", index=False)
        all_df.to_excel(xw, sheet_name="All metrics", index=False)
        sources_df.to_excel(xw, sheet_name="Sources", index=False)
        readme.to_excel(xw, sheet_name="Read me", index=False)

    print(f"[xlsx] {len(csvs)} run -> {out_path}")
    for s in sources:
        print(f"   {s['model']:32} <- {s['source_csv']}")


if __name__ == "__main__":
    main()
