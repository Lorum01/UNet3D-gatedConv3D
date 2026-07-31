# =============================
#       IMPORT & CONFIG
# =============================
import os
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

try:
    import cv2  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None


def _resize_mask(mask: np.ndarray, image_size: tuple) -> np.ndarray:
    target_h, target_w = image_size
    if cv2 is not None:
        # cv2.resize takes (width, height)
        return cv2.resize(mask, (target_w, target_h))

    # Fallback senza OpenCV: usa SciPy (preserva i valori float).
    from scipy.ndimage import zoom

    h, w = mask.shape[:2]
    if h == 0 or w == 0:
        raise ValueError(f"Invalid mask shape for resize: {mask.shape}")

    if mask.ndim == 2:
        return zoom(mask, (target_h / h, target_w / w), order=1)
    if mask.ndim == 3:
        return zoom(mask, (target_h / h, target_w / w, 1), order=1)
    raise ValueError(f"Unsupported mask ndim={mask.ndim} for resize")


def load_event_classes_from_excel(excel_path):
    """
    Legge il file Excel e restituisce due dizionari (chiave = indice riga):
       - event_class_dict: indice_evento -> classe
       - event_split_override_dict: indice_evento -> "train"/"val"/"test" o None
    Assumendo che l'ordine delle righe corrisponda all'ordine in cui verranno
    caricate le cartelle "base" (senza suffisso "_flipped") con sorted().

    La colonna "Split" e' opzionale: se assente o vuota per una riga, quell'evento
    segue lo split calcolato dalle percentuali per classe (dataset.split.class_percentages).
    """
    df = pd.read_excel(excel_path)
    event_class_dict = {}
    event_split_override_dict = {}
    has_split_col = "Split" in df.columns
    for i, row in df.iterrows():
        event_class_dict[i] = row["Class"]

        split_value = row["Split"] if has_split_col else None
        if split_value is None or pd.isna(split_value):
            split_value = None
        else:
            split_value = str(split_value).strip().lower()
            if split_value not in {"train", "val", "test"}:
                raise ValueError(
                    f"Excel '{excel_path}': colonna 'Split' non valida alla riga {i}: "
                    f"{split_value!r} (atteso 'train'/'val'/'test' o vuoto)."
                )
        event_split_override_dict[i] = split_value
    return event_class_dict, event_split_override_dict


def load_series_from_folders(mask_folder, image_size=(100, 100)):
    """
    Carica serie temporali da una struttura di cartelle.

    Args:
        mask_folder: cartella contenente una sottocartella per evento.
        image_size: (H, W) target a cui ridimensionare ogni frame.

    Ritorna:
      - all_series: lista di array NumPy (T, H, W, C) o (T, H, W) per ogni evento
      - all_series_filenames: lista di liste di stringhe, una per ogni evento,
        contenente i nomi dei file/percorsi corrispondenti a ciascun frame.
      - all_event_names: lista dei nomi delle sottocartelle evento, nello stesso
        ordine (e con lo stesso indice) di `all_series`/`all_series_filenames`.
    """
    all_series = []
    all_series_filenames = []
    all_event_names = []

    for event_folder in sorted(os.listdir(mask_folder)):
        mask_event_path = os.path.join(mask_folder, event_folder)
        if not os.path.isdir(mask_event_path):
            continue

        mask_files = sorted(os.listdir(mask_event_path))
        mask_event_data = []
        mask_event_filelist = []

        for mask_file in mask_files:
            full_path = os.path.join(mask_event_path, mask_file)
            if not os.path.isfile(full_path):
                continue
            mask = np.load(full_path)
            mask_resized = _resize_mask(mask, image_size)
            mask_event_data.append(mask_resized)
            mask_event_filelist.append(full_path)

        mask_event_data = np.array(mask_event_data)
        all_series.append(mask_event_data)
        all_series_filenames.append(mask_event_filelist)
        all_event_names.append(event_folder)

    return all_series, all_series_filenames, all_event_names


def create_sequences_multiple_series_fixed_input(
    all_series,
    all_classes,
    input_length,
    prediction_length,
    stride,
    all_series_filenames=None
):
    """
    Genera sequenze di input e target da più serie temporali con una lunghezza di input fissa.
    RItorna:
      all_inputs, all_targets, labels
      e se all_series_filenames non è None:
      all_input_filenames, all_target_filenames
    """
    all_inputs = []
    all_targets = []
    labels = []

    all_input_filenames = []
    all_target_filenames = []

    for series_idx, (series, class_label) in enumerate(zip(all_series, all_classes)):
        print(f"Processing series {series_idx+1}/{len(all_series)} - shape: {series.shape} - classe: {class_label}")
        T = len(series)

        series_filenames = None
        if all_series_filenames is not None:
            series_filenames = all_series_filenames[series_idx]

        valid_count = 0
        for i in range(0, T - input_length - prediction_length + 1, stride):
            input_seq = series[i : i + input_length]
            target_seq = series[i + input_length : i + input_length + prediction_length]

            all_inputs.append(input_seq)
            all_targets.append(target_seq)
            labels.append(class_label)

            if series_filenames is not None:
                input_seq_fnames = series_filenames[i : i + input_length]
                target_seq_fnames = series_filenames[i + input_length : i + input_length + prediction_length]
                all_input_filenames.append(input_seq_fnames)
                all_target_filenames.append(target_seq_fnames)

            valid_count += 1

        print(f"Series {series_idx} - Valid sequences: {valid_count}")

    if len(all_inputs) > 0:
        print("Concatenating sequences from all series...")
        all_inputs = np.array(all_inputs, dtype=object)
        all_targets = np.array(all_targets, dtype=object)
        labels = np.array(labels, dtype=int)
        print(f"Total sequences: {len(all_inputs)}")
    else:
        print("No valid sequences found in all series.")
        all_inputs = np.array([], dtype=object)
        all_targets = np.array([], dtype=object)
        labels = np.array([])

    if all_series_filenames is not None:
        all_input_filenames = np.array(all_input_filenames, dtype=object)
        all_target_filenames = np.array(all_target_filenames, dtype=object)
        return all_inputs, all_targets, labels, all_input_filenames, all_target_filenames
    else:
        return all_inputs, all_targets, labels


def check_range_of_images(all_inputs, all_targets):
    """
    Verifica il valore minimo e massimo in all_inputs e all_targets,
    prima di effettuare qualsiasi normalizzazione/standardizzazione.
    """
    min_val = float('inf')
    max_val = float('-inf')

    for i in range(len(all_inputs)):
        data_np = all_inputs[i]
        target_np = all_targets[i]
        d_min, d_max = data_np.min(), data_np.max()
        t_min, t_max = target_np.min(), target_np.max()
        if d_min < min_val: min_val = d_min
        if d_max > max_val: max_val = d_max
        if t_min < min_val: min_val = t_min
        if t_max > max_val: max_val = t_max

    print("=== Check range (prima della normalizzazione) ===")
    print(f"Valore minimo trovato: {min_val}")
    print(f"Valore massimo trovato: {max_val}")
    print("===============================================\n")


def plot_sequence(sequence, title, fnames=None):
    """
    Visualizza una sequenza di immagini (o una singola immagine) usando Matplotlib.
    sequence: array di shape (T, H, W, C)  (oppure (T,H,W) se in scala di grigi).
    title: titolo della figura.
    fnames: lista/array di filenames corrispondenti a 'sequence', da mostrare in ogni subplot (senza estensione).
    """
    if isinstance(sequence, list):
        sequence = np.array(sequence)

    if sequence.dtype not in (np.float32, np.float64):
        sequence = sequence.astype(float)

    num_frames = sequence.shape[0]
    fig, axes = plt.subplots(1, num_frames, figsize=(15, 5))
    fig.suptitle(title, fontsize=16)

    if num_frames == 1:
        if isinstance(axes, np.ndarray):
            axes = axes[0]
        axes.imshow(sequence[0])
        axes.axis('off')
        if fnames is not None:
            short_name = os.path.splitext(os.path.basename(fnames[0]))[0]
            axes.set_title(short_name, fontsize=10)
    else:
        for i in range(num_frames):
            ax = axes[i]
            ax.imshow(sequence[i])
            ax.axis('off')
            if fnames is not None and i < len(fnames):
                short_name = os.path.splitext(os.path.basename(fnames[i]))[0]
                ax.set_title(short_name, fontsize=10)

    plt.show()
