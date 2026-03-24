import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import random
from collections import defaultdict
import matplotlib.pyplot as plt


# -------------------------------------
# CustomDataset (RESTITUISCE 4 Elementi)
# -------------------------------------
class CustomDataset(Dataset):
    """
    Dataset personalizzato per gestire (input, target, label, filenames).
    - Se mean/std sono forniti, esegue la standardizzazione (x - mean) / std;
    - Se scale_to_neg1_pos1=True, i dati vengono mappati da [0,1]->[-1,1].
    - Ritorna: (data, target, label, (input_fnames, target_fnames))
    """
    def __init__(
        self,
        data,
        targets,
        labels,
        mean=None,
        std=None,
        scale_to_neg1_pos1=False,
        input_filenames=None,
        target_filenames=None
    ):
        print("Inizializzazione del dataset personalizzato...")
        self.data    = [torch.tensor(sample, dtype=torch.float32) for sample in data]
        self.targets = [torch.tensor(sample, dtype=torch.float32) for sample in targets]
        self.labels  = torch.tensor(labels, dtype=torch.long)

        self.input_fnames  = input_filenames
        self.target_fnames = target_filenames

        self.mean = None
        self.std  = None
        self.scale_to_neg1_pos1 = scale_to_neg1_pos1

        if mean is not None and std is not None:
            self.mean = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
            self.std  = torch.tensor(std,  dtype=torch.float32).view(1, -1, 1, 1)

        print(f"Dataset creato con {len(self.labels)} campioni totali.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data   = self.data[idx]      # (T, H, W, C)
        target = self.targets[idx]   # (T_pred, H, W, C)
        label  = self.labels[idx]

        input_fnames  = self.input_fnames[idx]  if self.input_fnames  is not None else None
        target_fnames = self.target_fnames[idx] if self.target_fnames is not None else None

        # (T, C, H, W)
        data   = data.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)

        # 1) scaling [-1,1] se richiesto
        if self.scale_to_neg1_pos1:
            data   = data * 2.0 - 1.0
            target = target * 2.0 - 1.0

        # 2) standardizzazione se mean/std definiti (usata solo se non fai scaling)
        elif self.mean is not None and self.std is not None:
            data   = (data - self.mean) / self.std
            target = (target - self.mean) / self.std

        return data, target, label, (input_fnames, target_fnames)


def split_by_class_distribution(labels, class_distribution, shuffle=True, seed=None):
    """
    Suddivide gli indici in train/val/test in base a un dict con numeri FISSI per classe.
    Ritorna: (train_idx, val_idx, test_idx).
    """
    if seed is not None:
        random.seed(seed)

    class_to_indices = defaultdict(list)
    for idx, c in enumerate(labels):
        class_to_indices[c].append(idx)

    train_indices, val_indices, test_indices = [], [], []

    for c, dist in class_distribution.items():
        c_train = dist["train"]
        c_val   = dist["val"]
        c_test  = dist["test"]

        idx_list = class_to_indices[c]
        if shuffle:
            random.shuffle(idx_list)

        total_needed = c_train + c_val + c_test
        if len(idx_list) < total_needed:
            print(f"[ATTENZIONE] Classe {c}: richiesti {total_needed}, disponibili {len(idx_list)}.")

        used = 0
        train_indices += idx_list[used : used + c_train]; used += c_train
        val_indices   += idx_list[used : used + c_val  ]; used += c_val
        test_indices  += idx_list[used : used + c_test ]; used += c_test

    return train_indices, val_indices, test_indices


def compute_mean_std(dataset, indices):
    """
    Calcola media e std dei canali (3) sui soli campioni 'indices'.
    """
    sum_c    = np.zeros(3, dtype=np.float64)
    sum_sq_c = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    for i in indices:
        data, _, _, _ = dataset[i]  # data = (T, C, H, W)
        data_np = data.numpy()
        C   = data_np.shape[1]
        THW = data_np.shape[0] * data_np.shape[2] * data_np.shape[3]
        data_reshaped = data_np.reshape(C, THW)
        sum_c    += data_reshaped.sum(axis=1)
        sum_sq_c += (data_reshaped**2).sum(axis=1)
        total_pixels += THW

    mean = sum_c / total_pixels
    var  = (sum_sq_c / total_pixels) - (mean**2)
    std  = np.sqrt(var)
    return mean, std


def display_batch_details(dataloader):
    """
    Stampa per ogni batch: lunghezze input/target e labels.
    """
    for batch_idx, (inputs, targets, lbls) in enumerate(dataloader):
        input_lengths  = [inp.shape[0] for inp in inputs]
        target_lengths = [tar.shape[0] for tar in targets]
        labels_list    = lbls.tolist()
        print(f"Batch {batch_idx + 1}:")
        print(f"  Input lengths:  {input_lengths}")
        print(f"  Target lengths: {target_lengths}")
        print(f"  Labels:         {labels_list}\n")


def show_images_from_batch(
    data, 
    targets, 
    labels, 
    max_samples=1, 
    mean=None, 
    std=None,
    input_fnames_batch=None,
    target_fnames_batch=None
):
    """
    Mostra i frame di input (prima riga) e target (seconda riga) per al più 'max_samples'.
    data.shape = (B, T, C, H, W)
    """
    batch_size = data.size(0)
    time_steps = data.size(1)
    pred_steps = targets.size(1)
    num_samples = min(max_samples, batch_size)
    
    for i in range(num_samples):
        lbl = labels[i].item() if hasattr(labels[i], "item") else labels[i]
        this_input_fnames  = input_fnames_batch[i]  if (input_fnames_batch  is not None) else None
        this_target_fnames = target_fnames_batch[i] if (target_fnames_batch is not None) else None

        n_cols = max(time_steps, pred_steps)
        fig, axes = plt.subplots(2, n_cols, figsize=(3*n_cols, 6))
        if n_cols == 1:
            axes = axes.reshape(2, 1)
        fig.suptitle(f"Sample index: {i} - Label: {lbl}", fontsize=14)

        # INPUT
        for t in range(n_cols):
            if t < time_steps:
                frame_in = data[i, t].permute(1, 2, 0).cpu().numpy()
                if mean is not None and std is not None:
                    frame_in = frame_in * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
                frame_in = np.clip(frame_in, 0, 1)
                title_str = f"Input Frame {t}"
                if this_input_fnames is not None and t < len(this_input_fnames):
                    title_str += f"\n{os.path.basename(this_input_fnames[t])}"
                axes[0, t].imshow(frame_in, vmin=0, vmax=1)
                axes[0, t].set_title(title_str, fontsize=10)
                axes[0, t].axis("off")
            else:
                axes[0, t].axis("off")

        # TARGET
        for t in range(n_cols):
            if t < pred_steps:
                frame_out = targets[i, t].permute(1, 2, 0).cpu().numpy()
                if mean is not None and std is not None:
                    frame_out = frame_out * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
                frame_out = np.clip(frame_out, 0, 1)
                title_str = f"Target Frame {t}"
                if this_target_fnames is not None and t < len(this_target_fnames):
                    title_str += f"\n{os.path.basename(this_target_fnames[t])}"
                axes[1, t].imshow(frame_out, vmin=0, vmax=1)
                axes[1, t].set_title(title_str, fontsize=10)
                axes[1, t].axis("off")
            else:
                axes[1, t].axis("off")

        plt.tight_layout()
        plt.show()


def check_dataset_range(dataset, standardized=False):
    """
    Controlla min/max di data e target in tutto il dataset.
    """
    min_val = float('inf')
    max_val = float('-inf')
    for i in range(len(dataset)):
        data, target, _, _ = dataset[i]
        d_min, d_max = data.min().item(),   data.max().item()
        t_min, t_max = target.min().item(), target.max().item()
        min_val = min(min_val, d_min, t_min)
        max_val = max(max_val, d_max, t_max)

    print(f"\nValore minimo trovato: {min_val:.4f}")
    print(f"Valore massimo trovato: {max_val:.4f}")

    if min_val < -1.0 or max_val > 1.0:
        print("ATTENZIONE: i dati NON sono tutti in [-1,1]!")
    else:
        print("OK: tutti i dati sono in [-1,1].")



def custom_collate_fn(batch):
    """
    batch: lista di elementi
      (data, targets, labels, (input_fnames, target_fnames))
    -> ritorna: data_tensor, target_tensor, labels_tensor, (input_fnames_list, target_fnames_list)
    """
    data_list, target_list, labels_list = [], [], []
    input_fnames_list, target_fnames_list = [], []

    for (d, t, l, fnames_tuple) in batch:
        inf, tf = fnames_tuple
        data_list.append(d)
        target_list.append(t)
        labels_list.append(l)
        input_fnames_list.append(inf)
        target_fnames_list.append(tf)

    data_tensor   = torch.stack(data_list, dim=0)    # (B, T, C, H, W)
    target_tensor = torch.stack(target_list, dim=0)  # (B, T, C, H, W)
    labels_tensor = torch.stack(labels_list, dim=0)  # (B,)
    return data_tensor, target_tensor, labels_tensor, (input_fnames_list, target_fnames_list)


def report_split_coverage(train_idx, val_idx, test_idx, total_len, verbose=True):
    """
    Riassume lo split e segnala duplicati e non assegnati.
    """
    n_train, n_val, n_test = len(train_idx), len(val_idx), len(test_idx)
    dup_train_val  = set(train_idx).intersection(val_idx)
    dup_train_test = set(train_idx).intersection(test_idx)
    dup_val_test   = set(val_idx).intersection(test_idx)
    duplicated     = dup_train_val | dup_train_test | dup_val_test
    covered        = set(train_idx) | set(val_idx) | set(test_idx)
    uncovered      = set(range(total_len)) - covered

    summary = {
        "train": n_train,
        "val":   n_val,
        "test":  n_test,
        "total_assigned": n_train + n_val + n_test,
        "duplicates": sorted(duplicated),
        "n_duplicates": len(duplicated),
        "uncovered": len(uncovered),
    }

    if verbose:
        print("\n=== Distribuzione attuale ===")
        print(f" train: {n_train}")
        print(f" val  : {n_val}")
        print(f" test : {n_test}")
        print(f"  TOT assegnati : {n_train + n_val + n_test} / {total_len}")
        if summary["n_duplicates"] > 0:
            print(f"\n[ATTENZIONE] Ci sono {summary['n_duplicates']} duplicati:")
            print(summary["duplicates"])
        if summary["uncovered"] > 0:
            print(f"\nCampioni NON assegnati a nessun set: {summary['uncovered']}")
        else:
            print("\nTutti i campioni sono assegnati (senza duplicati).")
    return summary

def pct_to_counts(class_pct, labels, round_method='round'):
    """
    class_pct: {class: {"train":0.6,"val":0.2,"test":0.2}, ...}
    labels: array-like di label per ogni sample (event-level)
    Ritorna: class_distribution con conteggi interi per classe adatti a split_by_class_distribution.
    This function converts per-class split percentages into integer sample counts per class so you can use them with a splitter that expects counts
    """
    from collections import defaultdict
    import numpy as np

    class_to_indices = defaultdict(list)
    for idx, c in enumerate(labels):
        class_to_indices[c].append(idx)

    class_counts = {}
    for c, pct in class_pct.items():
        n_available = len(class_to_indices.get(c, []))
        if n_available == 0:
            class_counts[c] = {"train":0, "val":0, "test":0}
            continue

        t = pct.get("train", 0)
        v = pct.get("val", 0)
        te = pct.get("test", 0)
        # calcola raw counts
        n_train = t * n_available
        n_val   = v * n_available
        n_test  = te * n_available

        if round_method == 'floor':
            n_train, n_val, n_test = int(np.floor(n_train)), int(np.floor(n_val)), int(np.floor(n_test))
        else:
            n_train, n_val, n_test = int(round(n_train)), int(round(n_val)), int(round(n_test))

        total = n_train + n_val + n_test
        if total > n_available:
            # ridistribuisci eccedenza dal test -> val -> train
            diff = total - n_available
            for key in ('test','val','train'):
                if diff<=0: break
                if key=='test' and n_test>0:
                    dec = min(diff, n_test)
                    n_test -= dec; diff -= dec
                if key=='val' and diff>0 and n_val>0:
                    dec = min(diff, n_val)
                    n_val -= dec; diff -= dec
                if key=='train' and diff>0 and n_train>0:
                    dec = min(diff, n_train)
                    n_train -= dec; diff -= dec
        elif total < n_available:
            # assegna resto al train
            n_train += (n_available - total)

        class_counts[c] = {"train": n_train, "val": n_val, "test": n_test}
    return class_counts
