from __future__ import annotations

from collections import Counter
from typing import Any, Dict

import numpy as np
from torch.utils.data import DataLoader, Subset

from ..data.input_utility import (
    load_series_from_folders,
    load_event_classes_from_excel,
    create_sequences_multiple_series_fixed_input,
    check_range_of_images,
)
from ..data.dataloader import (
    CustomDataset,
    pct_to_counts,
    split_by_class_distribution,
    compute_mean_std,
    report_split_coverage,
    check_dataset_range,
    custom_collate_fn,
)


def _load_events(cfg: Dict[str, Any]):
    paths = cfg["paths"]

    dataset_folder = paths["dataset_folder"]
    excel_path = paths["excel_path"]

    # 1) Carica serie + filenames
    all_series, all_series_filenames = load_series_from_folders(dataset_folder)
    print(f"Numero totale di serie caricate: {len(all_series)}")
    if len(all_series) > 0:
        print(f"Dimensione della prima serie: {all_series[0].shape}")

    # 2) Classi da Excel
    event_class_dict = load_event_classes_from_excel(excel_path)
    event_labels = [event_class_dict[i] for i, _ in enumerate(all_series)]

    class_counts = Counter(event_labels)
    print("Numero totale serie per ogni classe:")
    for classe, count in class_counts.items():
        print(f"{classe}: {count}")

    return all_series, all_series_filenames, event_labels


def _build_sequences_from_events(
    *,
    all_series,
    all_series_filenames,
    event_labels,
    dcfg: Dict[str, Any],
):
    # 3) Sequenze (con filenames) - labels a livello sequenza = classe dell'evento sorgente
    result = create_sequences_multiple_series_fixed_input(
        all_series=all_series,
        all_classes=event_labels,
        input_length=int(dcfg["input_length"]),
        prediction_length=int(dcfg["prediction_length"]),
        stride=int(dcfg["stride"]),
        all_series_filenames=all_series_filenames,
    )
    if len(result) == 5:
        all_inputs, all_targets, labels, input_fnames, target_fnames = result
    else:
        all_inputs, all_targets, labels = result
        input_fnames, target_fnames = None, None

    print("Dimensione totale input:", getattr(all_inputs, "shape", None))
    print("Dimensione totale target:", getattr(all_targets, "shape", None))
    print("Dimensione totale labels:", getattr(labels, "shape", None))

    check_range_of_images(all_inputs, all_targets)

    expected = tuple(dcfg["expected_input_shape"])
    invalid_inputs = [idx for idx, inp in enumerate(all_inputs) if getattr(inp, "shape", None) != expected]
    invalid_targets = [idx for idx, tgt in enumerate(all_targets) if getattr(tgt, "shape", None) != expected]
    if invalid_inputs:
        print(f"[ATTENZIONE] {len(invalid_inputs)} input con shape non attesa. Esempio: {all_inputs[invalid_inputs[0]].shape}")
    if invalid_targets:
        print(f"[ATTENZIONE] {len(invalid_targets)} target con shape non attesa. Esempio: {all_targets[invalid_targets[0]].shape}")

    # Cast float32
    all_inputs_list = [x.astype(np.float32) for x in all_inputs]
    all_targets_list = [x.astype(np.float32) for x in all_targets]

    return all_inputs_list, all_targets_list, labels, input_fnames, target_fnames


def build_dataloaders(cfg: Dict[str, Any]):
    dcfg = cfg["data"]
    lcfg = cfg.get("dataloader", {})

    split_strategy = str(dcfg.get("split_strategy", "train_val_test")).strip().lower()
    normalization = dcfg.get("normalization", None)
    if normalization is None:
        # Legacy behavior: follow `scale_to_neg1_pos1`
        normalization = "neg1pos1" if bool(dcfg.get("scale_to_neg1_pos1", True)) else "none"
    normalization = str(normalization).strip().lower()
    if normalization not in {"none", "neg1pos1", "standardize"}:
        raise ValueError(f"Unsupported data.normalization={normalization!r}. Use 'none'|'neg1pos1'|'standardize'.")

    scale_to_neg1_pos1 = normalization == "neg1pos1"
    all_series, all_series_filenames, event_labels = _load_events(cfg)

    if split_strategy == "by_class":
        all_inputs_list, all_targets_list, labels, input_fnames, target_fnames = _build_sequences_from_events(
            all_series=all_series,
            all_series_filenames=all_series_filenames,
            event_labels=event_labels,
            dcfg=dcfg,
        )

        mean_c = std_c = None
        if normalization == "standardize":
            mean_c = dcfg.get("mean", None)
            std_c = dcfg.get("std", None)
            if mean_c is None or std_c is None:
                raise ValueError("data.normalization='standardize' with split_strategy='by_class' requires data.mean and data.std in config.")

        # Dataset completo (come notebook: scaling opzionale + filenames)
        full_dataset = CustomDataset(
            all_inputs_list,
            all_targets_list,
            labels,
            mean=mean_c,
            std=std_c,
            scale_to_neg1_pos1=scale_to_neg1_pos1,
            input_filenames=input_fnames,
            target_filenames=target_fnames,
        )

        # Nessuno split train/val/test: crea subset per classe dall'Excel
        unique_classes = sorted(set(int(x) for x in labels.tolist()))
        loaders_by_class = {}
        bs = int(lcfg.get("batch_size_test", 4))

        for c in unique_classes:
            idxs = [i for i, lab in enumerate(labels.tolist()) if int(lab) == c]
            subset = Subset(full_dataset, idxs)
            loaders_by_class[c] = DataLoader(subset, batch_size=bs, shuffle=False, collate_fn=custom_collate_fn)

        # Per compatibilità: restituisce un dict quando split_strategy=by_class
        check_dataset_range(full_dataset, standardized=True)
        return loaders_by_class

    if split_strategy != "train_val_test":
        raise ValueError(f"Unsupported data.split_strategy={split_strategy!r}. Use 'train_val_test' or 'by_class'.")

    # ---- split train/val/test (default) ----
    # Split a livello EVENTO per evitare leakage (finestre dello stesso evento in split diversi).
    if bool(dcfg.get("use_percent_distribution", True)):
        class_distribution = pct_to_counts(dcfg["class_pct"], event_labels, round_method="round")
    else:
        raise ValueError("use_percent_distribution=false non è supportato in questa versione (manca CLASS_DISTRIBUTION).")

    seed = int(dcfg.get("split_seed", 424))
    train_event_idx, val_event_idx, test_event_idx = split_by_class_distribution(
        event_labels, class_distribution, shuffle=True, seed=seed
    )
    print("Class distribution used for EVENT split:", class_distribution)
    report_split_coverage(train_event_idx, val_event_idx, test_event_idx, total_len=len(event_labels))
    # Sanity: nessuna sovrapposizione fra eventi dei vari split
    overlap_tv = set(train_event_idx).intersection(val_event_idx)
    overlap_tt = set(train_event_idx).intersection(test_event_idx)
    overlap_vt = set(val_event_idx).intersection(test_event_idx)
    if overlap_tv or overlap_tt or overlap_vt:
        print(f"[ATTENZIONE] Overlap eventi fra split: train∩val={len(overlap_tv)}, train∩test={len(overlap_tt)}, val∩test={len(overlap_vt)}")
    print(f"Eventi per split: train={len(train_event_idx)} val={len(val_event_idx)} test={len(test_event_idx)}")

    def _select_events(indices):
        sel_series = [all_series[i] for i in indices]
        sel_fnames = [all_series_filenames[i] for i in indices]
        sel_labels = [event_labels[i] for i in indices]
        return sel_series, sel_fnames, sel_labels

    tr_series, tr_fnames, tr_labels = _select_events(train_event_idx)
    va_series, va_fnames, va_labels = _select_events(val_event_idx)
    te_series, te_fnames, te_labels = _select_events(test_event_idx)

    tr_inputs, tr_targets, tr_y, tr_in_fn, tr_tg_fn = _build_sequences_from_events(
        all_series=tr_series,
        all_series_filenames=tr_fnames,
        event_labels=tr_labels,
        dcfg=dcfg,
    )
    va_inputs, va_targets, va_y, va_in_fn, va_tg_fn = _build_sequences_from_events(
        all_series=va_series,
        all_series_filenames=va_fnames,
        event_labels=va_labels,
        dcfg=dcfg,
    )
    te_inputs, te_targets, te_y, te_in_fn, te_tg_fn = _build_sequences_from_events(
        all_series=te_series,
        all_series_filenames=te_fnames,
        event_labels=te_labels,
        dcfg=dcfg,
    )

    mean_c = std_c = None
    if normalization == "standardize":
        # Calcola mean/std SOLO sul TRAIN (senza trasformazioni) e riusa su val/test.
        temp_train = CustomDataset(tr_inputs, tr_targets, tr_y, scale_to_neg1_pos1=False)
        mean_c, std_c = compute_mean_std(temp_train, range(len(temp_train)))
        print("Mean canali (train):", mean_c)
        print("Std  canali (train):", std_c)

    train_dataset = CustomDataset(
        tr_inputs,
        tr_targets,
        tr_y,
        mean=mean_c,
        std=std_c,
        scale_to_neg1_pos1=scale_to_neg1_pos1,
        input_filenames=tr_in_fn,
        target_filenames=tr_tg_fn,
    )
    val_dataset = CustomDataset(
        va_inputs,
        va_targets,
        va_y,
        mean=mean_c,
        std=std_c,
        scale_to_neg1_pos1=scale_to_neg1_pos1,
        input_filenames=va_in_fn,
        target_filenames=va_tg_fn,
    )
    test_dataset = CustomDataset(
        te_inputs,
        te_targets,
        te_y,
        mean=mean_c,
        std=std_c,
        scale_to_neg1_pos1=scale_to_neg1_pos1,
        input_filenames=te_in_fn,
        target_filenames=te_tg_fn,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(lcfg["batch_size_train"]),
        shuffle=True,
        collate_fn=custom_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(lcfg["batch_size_val"]),
        shuffle=False,
        collate_fn=custom_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(lcfg["batch_size_test"]),
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    check_dataset_range(train_dataset, standardized=True)
    check_dataset_range(val_dataset, standardized=True)
    check_dataset_range(test_dataset, standardized=True)
    return train_loader, val_loader, test_loader
