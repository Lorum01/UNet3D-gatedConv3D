import os
from typing import Sequence, Optional, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import imageio.v2 as imageio



def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def to_device(x, device: torch.device):
    """Safely move a tensor to a device if it's a torch Tensor."""
    return x.to(device) if isinstance(x, torch.Tensor) else x


def build_denorm(mean: Optional[Sequence[float]],
                 std: Optional[Sequence[float]],
                 device: torch.device):
    """Return a function that denormalizes a (B,C,T,H,W) tensor to [0,1]."""
    if mean is None or std is None:
        def _iden(x: torch.Tensor) -> torch.Tensor:
            return x.clamp(0, 1)
        return _iden

    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device).view(1, -1, 1, 1, 1)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=device).view(1, -1, 1, 1, 1)

    def _denorm(x: torch.Tensor) -> torch.Tensor:
        return (x * std_t + mean_t).clamp(0, 1)

    return _denorm


def as_rgb_uint8(img_chw: np.ndarray) -> np.ndarray:
    """(C,H,W) in [0,1] -> (H,W,C) uint8 [0,255]."""
    img_rgb = np.transpose(img_chw, (1, 2, 0))  # (H,W,C)
    img_rgb_255 = (np.clip(img_rgb, 0, 1) * 255).astype(np.uint8)
    return img_rgb_255


def save_np_and_jpg(out_dir: str, base: str, img_chw: np.ndarray) -> None:
    """Save both NPY and JPG versions of a single (C,H,W) array in [0,1]."""
    ensure_dir(out_dir)
    np.save(os.path.join(out_dir, f"{base}.npy"), np.transpose(img_chw, (1, 2, 0)))  # save as (H,W,C)
    Image.fromarray(as_rgb_uint8(img_chw)).save(os.path.join(out_dir, f"{base}.jpg"), quality=95)


def frames_to_gif(frames_chw: List[np.ndarray], gif_path: str, fps: int) -> None:
    """Save a list of (C,H,W) frames in [0,1] to an animated GIF with the given fps."""
    ensure_dir(os.path.dirname(gif_path))
    frames_rgb = [as_rgb_uint8(frame) for frame in frames_chw]
    duration = 1.0 / max(fps, 1)
    imageio.mimsave(gif_path, frames_rgb, duration=duration)


def permute_to_bcthw(x: torch.Tensor) -> torch.Tensor:
    """Ensure shape is (B, C, T, H, W). If already so, return as-is."""
    if x.dim() != 5:
        raise ValueError(f"Expected 5D tensor (B,*,*,*,*), got shape {tuple(x.shape)}")
    # if channels are not on dim=1, assume (B, T, C, H, W) and permute
    return x if x.shape[1] in (1, 3) else x.permute(0, 2, 1, 3, 4)

#Test autoregressivo 2-step (4 in -> 4 out, poi 2-step autoregressivo) con salvataggio immagini e GIF.

def test_model_create_gifs_3ch(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device | str,
    save_dir: str,
    max_batches: int = 1,
    gif_fps: int = 2,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
    checkpoint_path: Optional[str] = "convUnet_Best_Test_mod_1/best_model.pth",
    save_images: bool = True,
    save_gifs: bool = True,
    show_plots: bool = True,
) -> None:
    """
    Test autoregressivo 2-step (4 in -> 4 out, poi 2-step autoregressivo) con salvataggio immagini e GIF.

    Struttura cartelle per ogni sample `sample_batch{b}_sample{i}_label{L}`:
        input_t{t}_{basename-input}.npy|jpg
        target_t{t}_{basename-target}.npy|jpg
        pred_{basename-target}.npy|jpg
        predm_{basename-target}.npy|jpg
        (opz.) input.gif | target.gif | pred.gif | predm.gif
        {sample_idx}_input.png
        {sample_idx}_target_pred.png

    Parametri
    ---------
    model : nn.Module
        Modello PyTorch che mappa (B,3,4,H,W) -> (B,3,4,H,W).
    test_loader : DataLoader
        Loader che restituisce (data, targets, labels, filenames) oppure (data, targets, ...).
    device : torch.device | str
        Dispositivo ("cuda", "cpu" o torch.device).
    save_dir : str
        Directory radice per i risultati.
    max_batches : int
        Numero massimo di batch da elaborare.
    gif_fps : int
        FPS per le GIF animate.
    mean, std : Optional[Sequence[float]]
        Statistiche per la denormalizzazione (per canali). Se None, clamp in [0,1].
    checkpoint_path : str | None
        Percorso al checkpoint del modello; se None, non carica pesi.
    save_images : bool
        Se True, salva NPY/JPG per ogni frame.
    save_gifs : bool
        Se True, salva le GIF per input/target/pred/predm.
    show_plots : bool
        Se True, mostra le figure matplotlib.
    """

    # ------------------------- Preparazione -------------------------
    ensure_dir(save_dir)
    device = torch.device(device) if not isinstance(device, torch.device) else device
    model = model.to(device)

    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location=device)
        model = torch.nn.DataParallel(model) # Per caricare modelli salvati con DataParallel

        # Compatibility mapping: some checkpoints use attribute name `convlstm`
        # while current model uses `stackedConv`. Detect and remap keys.
        ck_keys = list(state.keys()) if isinstance(state, dict) else []
        model_keys = list(model.state_dict().keys())

        if any('convlstm' in k for k in ck_keys) and any('stackedConv' in k for k in model_keys):
            new_state = {}
            for k, v in state.items():
                new_k = k.replace('convlstm', 'stackedConv')
                new_state[new_k] = v
            state = new_state

        try:
            model.load_state_dict(state)
        except RuntimeError:
            # Fallback: try non-strict loading to tolerate minor naming/size diffs
            model.load_state_dict(state, strict=False)
    model.eval()

    denorm = build_denorm(mean, std, device)

    # --------------------------- Main Loop ---------------------------
    batch_count = 0
    with torch.no_grad():
        for batch in test_loader:
            batch_count += 1
            if batch_count > max_batches:
                break

            # Supporta sia (data, targets, labels, filenames) sia (data, targets, ...)
            if len(batch) >= 4:
                data, targets, labels, filenames = batch[:4]
                input_fnames_batch, target_fnames_batch = filenames
            else:
                data, targets = batch[:2]
                labels = [None] * data.shape[0]
                input_fnames_batch = target_fnames_batch = None

            # Porta a (B,3,T,H,W)
            data = permute_to_bcthw(data)
            targets = permute_to_bcthw(targets)

            data = to_device(data, device)
            targets = to_device(targets, device)

            # ------------------- 1ª inferenza -------------------
            pred1 = model(data)  # (B,3,4,H,W)

            # Prepara input per la 2ª inferenza (ancora normalizzato)
            last2_inputs = data[:, :, 2:, :, :]      # (B,3,2,H,W)
            first2_pred1 = pred1[:, :, :2, :, :]     # (B,3,2,H,W)
            new_input = torch.cat([last2_inputs, first2_pred1], dim=2)  # (B,3,4,H,W)

            # ------------------- 2ª inferenza -------------------
            pred2 = model(new_input)                 # (B,3,4,H,W)
            first2_pred2 = pred2[:, :, :2, :, :]     # (B,3,2,H,W)

            # Predizione modificata (4 frame): concat dei primi 2 di pred1 e dei primi 2 di pred2
            pred_mod = torch.cat([first2_pred1, first2_pred2], dim=2)  # (B,3,4,H,W)

            # ---------------- Denormalizzazione per I/O ----------------
            data_d, targets_d = denorm(data), denorm(targets)
            pred1_d, predm_d  = denorm(pred1), denorm(pred_mod)

            # In numpy (B,C,T,H,W)
            data_np = data_d.cpu().numpy()
            tgt_np  = targets_d.cpu().numpy()
            p1_np   = pred1_d.cpu().numpy()
            pm_np   = predm_d.cpu().numpy()

            B = data_np.shape[0]
            T_in, T_out = data_np.shape[2], tgt_np.shape[2]  # attesi = 4, 4

            for i in range(B):
                # Etichetta (se presente)
                label_i = (
                    labels[i].item() if (labels is not None and hasattr(labels[i], "item"))
                    else (labels[i] if labels is not None else "NA")
                )

                # Recupero nomi file del sample i (tuple di liste) se disponibili
                input_fnames_i  = input_fnames_batch[i]  if input_fnames_batch  is not None else None  # len=T_in
                target_fnames_i = target_fnames_batch[i] if target_fnames_batch is not None else None   # len=T_out

                sample_idx = f"batch{batch_count}_sample{i}_label{label_i}"
                sample_dir = os.path.join(save_dir, f"sample_{sample_idx}")
                ensure_dir(sample_dir)

                # Trasponi a (T, C, H, W)
                # data_np[i] è (C, T, H, W): per ottenere (T, C, H, W) servono gli assi (1, 0, 2, 3)
                in_seq  = np.transpose(data_np[i], (1, 0, 2, 3))
                tgt_seq = np.transpose(tgt_np[i],  (1, 0, 2, 3))
                p1_seq  = np.transpose(p1_np[i],   (1, 0, 2, 3))
                pm_seq  = np.transpose(pm_np[i],   (1, 0, 2, 3))

                # --------------------------
                # 1) SALVATAGGIO INPUT
                # --------------------------
                if save_images:
                    for t in range(T_in):
                        in_fname_base = (
                            os.path.splitext(os.path.basename(input_fnames_i[t]))[0]
                            if input_fnames_i is not None else f"input_t{t}"
                        )
                        save_np_and_jpg(sample_dir, f"input_t{t}_{in_fname_base}", in_seq[t])

                # --------------------------
                # 2) SALVATAGGIO TARGET
                # --------------------------
                if save_images:
                    for t in range(T_out):
                        tg_fname_base = (
                            os.path.splitext(os.path.basename(target_fnames_i[t]))[0]
                            if target_fnames_i is not None else f"target_t{t}"
                        )
                        save_np_and_jpg(sample_dir, f"target_t{t}_{tg_fname_base}", tgt_seq[t])

                # --------------------------
                # 3) SALVATAGGIO PRED ORIG
                #    (usa basename del target corrispondente)
                # --------------------------
                if save_images:
                    for t in range(T_out):
                        if target_fnames_i is not None:
                            tg_fname_base = os.path.splitext(os.path.basename(target_fnames_i[t]))[0]
                            pred_base = f"pred_{tg_fname_base}"
                        else:
                            pred_base = f"pred_t{t}"
                        save_np_and_jpg(sample_dir, pred_base, p1_seq[t])

                # --------------------------
                # 4) SALVATAGGIO PRED MOD
                #    (stessa logica, prefisso 'predm_')
                # --------------------------
                if save_images:
                    for t in range(T_out):
                        if target_fnames_i is not None:
                            tg_fname_base = os.path.splitext(os.path.basename(target_fnames_i[t]))[0]
                            predm_base = f"predm_{tg_fname_base}"
                        else:
                            predm_base = f"predm_t{t}"
                        save_np_and_jpg(sample_dir, predm_base, pm_seq[t])

                # --------------------------
                # 5) SALVATAGGIO GIF COMPOSITA (sequenze concatenate nel tempo)
                #   - Pannello 1: 4 input + 4 target  (tot 8 frame)
                #   - Pannello 2: 4 input + 4 pred    (tot 8 frame)
                #   - Pannello 3: 4 input + 4 predm   (tot 8 frame)
                #   Il file risultante è un'unica GIF con i tre pannelli affiancati per ogni frame.
                # --------------------------
                if save_gifs:
                    def _frame_from_concat(input_seq_tc_hw: np.ndarray,
                                           other_seq_tc_hw: np.ndarray,
                                           t: int,
                                           tin: int,
                                           tout: int) -> np.ndarray:
                        """Restituisce l'immagine (H,W,3) per il frame t della sequenza concatenata input+other."""
                        if t < tin:
                            img_chw = input_seq_tc_hw[t]
                        else:
                            idx = min(t - tin, tout - 1)
                            img_chw = other_seq_tc_hw[idx]
                        return as_rgb_uint8(img_chw)

                    tin, tout = T_in, T_out
                    T_all = tin + tout

                    frames_combined = []
                    for t in range(T_all):
                        A = _frame_from_concat(in_seq, tgt_seq, t, tin, tout)   # input+target
                        B = _frame_from_concat(in_seq, p1_seq,  t, tin, tout)   # input+pred
                        C = _frame_from_concat(in_seq, pm_seq,  t, tin, tout)   # input+predm
                        frames_combined.append(np.hstack([A, B, C]))

                    gif_path = os.path.join(sample_dir, "compare.gif")
                    duration = 1.0 / max(gif_fps, 1)
                    imageio.mimsave(gif_path, frames_combined, duration=duration)

                # ====== Figura INPUT ======
                fig_in, axes_in = plt.subplots(1, T_in, figsize=(3 * T_in, 3))
                fig_in.suptitle(f"Sample {sample_idx} - INPUT frames", fontsize=12)
                for t in range(T_in):
                    ax = axes_in[t] if T_in > 1 else axes_in
                    ax.imshow(as_rgb_uint8(in_seq[t]))
                    title_str = f"Input t={t}"
                    if input_fnames_i is not None:
                        title_str += f"\n{os.path.basename(input_fnames_i[t])}"
                    ax.set_title(title_str, fontsize=9)
                    ax.axis("off")
                plt.tight_layout()
                in_fig_path = os.path.join(sample_dir, f"{sample_idx}_input.png")
                plt.savefig(in_fig_path, dpi=100)
                if show_plots:
                    plt.show()
                else:
                    plt.close(fig_in)

                # ====== Figura TARGET / PRED ORIG / PRED MOD ======
                fig_out, axes_out = plt.subplots(3, T_out, figsize=(3 * T_out, 9))
                fig_out.suptitle(f"Sample {sample_idx} - TARGET / PRED ORIG / PRED MOD", fontsize=12)

                rows = (tgt_seq, p1_seq, pm_seq)
                row_names = ("TARGET", "PRED ORIG", "PRED MOD")

                for r, (seq, rname) in enumerate(zip(rows, row_names)):
                    for t in range(T_out):
                        ax = axes_out[r, t] if T_out > 1 else axes_out[r]
                        ax.imshow(as_rgb_uint8(seq[t]))
                        if r == 0:
                            title_str = f"t={t}"
                            if target_fnames_i is not None:
                                title_str += f"\n{os.path.basename(target_fnames_i[t])}"
                            ax.set_title(title_str, fontsize=9)
                        if t == 0:
                            ax.set_ylabel(rname, rotation=90, labelpad=10, fontsize=10)
                        ax.axis("off")

                plt.tight_layout()
                out_fig_path = os.path.join(sample_dir, f"{sample_idx}_target_pred.png")
                plt.savefig(out_fig_path, dpi=100)
                if show_plots:
                    plt.show()
                else:
                    plt.close(fig_out)

    print("\n== Test completato (IMMAGINI + GIF + NPY/JPG, naming basato sui filename originali) ==\n")


