import csv
import logging
import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from .loss_function import (
    get_lpips_range_stats,
    reset_lpips_range_stats,
    weighted_mse_lpips_loss,
)
from .lr_scheduler import create_scheduler


def _build_logger(checkpoint_dir: str) -> logging.Logger:
    """
    Logger dedicato a una singola run di training: scrive sia su console che su
    `training.log` dentro checkpoint_dir. Un nuovo handler per ogni run (nome del
    logger legato a checkpoint_dir) evita che run consecutive nello stesso processo
    si mescolino nello stesso file.
    """
    logger = logging.getLogger(f"unet3d_gatedconv3d.train.{os.path.basename(checkpoint_dir)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(os.path.join(checkpoint_dir, "training.log"), encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    return logger

def _maybe_wrap_data_parallel(model, device: str, use_data_parallel: bool):
    """
    Wrap `model` in nn.DataParallel only if requested AND the machine actually
    has more than one visible CUDA GPU. Otherwise return the model unchanged.

    Returns (model, is_wrapped).
    """
    if not use_data_parallel:
        return model, False

    if not str(device).lower().startswith("cuda"):
        print(f"[DataParallel] Richiesto ma device={device!r} non è cuda: ignorato.")
        return model, False

    if not torch.cuda.is_available():
        print("[DataParallel] Richiesto ma torch.cuda.is_available()=False: ignorato.")
        return model, False

    n_gpus = torch.cuda.device_count()
    if n_gpus <= 1:
        print(f"[DataParallel] Richiesto ma solo {n_gpus} GPU visibile/i: ignorato (serve >1).")
        return model, False

    print(f"[DataParallel] Attivo su {n_gpus} GPU.")
    return torch.nn.DataParallel(model), True


def _unwrapped_state_dict(model):
    """Return a plain (non-DataParallel) state_dict, regardless of wrapping."""
    if isinstance(model, torch.nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def split_batch(batch):
    """
    Scompone il batch in (data, targets, labels, filenames).
    """
    data, targets = batch[:2]
    labels = batch[2] if len(batch) > 2 else None
    filenames = batch[3] if len(batch) > 3 else None
    return data, targets, labels, filenames


# Training e Validation

def train_one_epoch_3d(model, dataloader, optimizer, device="cuda", alpha=0.5, lpips_input_mode="none"):
    """
    Esegue una epoca di training su modello 3D.
    Input batch:  (B, C, T, H, W)
    Input modello: (B, C, T, H, W).
    Loss: weighted_mse_lpips_loss(outputs, targets, alpha, lpips_input_mode).
    """
    model.train()
    running_loss = 0.0
    total_batches = 0

    for batch in dataloader:
        data, targets, _, _ = split_batch(batch)

        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(data)  # (B, C, T, H, W)

        # Loss composita MSE+LPIPS pesata da alpha
        loss = weighted_mse_lpips_loss(outputs, targets, alpha=alpha, lpips_input_mode=lpips_input_mode)
        # Per sola MSE:  loss = F.mse_loss(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_batches += 1

    epoch_loss = running_loss / total_batches if total_batches > 0 else 0.0
    return epoch_loss


@torch.no_grad()
def evaluate_model_3d(model, dataloader, device="cuda", alpha=0.5, lpips_input_mode="none"):
    """
    Valutazione su validation/test set con la stessa loss del training.
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for batch in dataloader:
        data, targets, _, _ = split_batch(batch)

        data = data.to(device)
        targets = targets.to(device)

        outputs = model(data)
        loss = weighted_mse_lpips_loss(outputs, targets, alpha=alpha, lpips_input_mode=lpips_input_mode)
        # Per sola MSE: loss = F.mse_loss(outputs, targets)

        total_loss += loss.item()
        total_batches += 1

    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    return avg_loss



# Training Loop:
# - ReduceLROnPlateau sul val_loss
# - Checkpointing periodico + best/last
# - Early Stopping manuale (epochs_no_improve)

def training_loop_with_validation_3d(
    model,
    train_loader,
    val_loader,
    num_epochs=10,
    lr=1e-3,
    device="cuda",
    patience_early_stopping=5,  # pazienza per early stop
    patience_lr_scheduler=6,    # pazienza per Riduzione LR
    factor=0.5,                 # fattore riduzione LR
    threshold=1e-4,             # soglia "miglioramento"
    checkpoint_interval=1,      # salva ogni N epoche
    checkpoint_dir="checkpoints",
    alpha=0.5,
    lpips_input_mode="none",
    show_plots: bool = False,
    use_data_parallel: bool = False,
):
    """
    Loop completo:
    - Ottimizzatore Adam
    - Scheduler ReduceLROnPlateau su val_loss
    - Early stopping se nessun miglioramento per 'patience_early_stopping' epoche
    - Checkpoint: best, last, e periodici
    - Plot finale di train/val loss

    `use_data_parallel`: se True, avvolge il modello in nn.DataParallel solo se la
    macchina espone effettivamente più di una GPU CUDA visibile; altrimenti viene
    ignorato silenziosamente (con un messaggio informativo) e si allena su singolo
    device. I checkpoint salvati contengono sempre uno state_dict "bare" (senza
    prefisso `module.`), indipendentemente dal wrapping usato in training.
    """
    model.to(device)
    model, _is_parallel = _maybe_wrap_data_parallel(model, device, use_data_parallel)

    # Ottimizzatore
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Scheduler: riduce il LR se la val_loss non migliora
    scheduler = create_scheduler(
        optimizer, factor=factor,
        patience=patience_lr_scheduler,
        threshold=threshold
    )

    # La risoluzione "evita sovrascritture" (suffisso data/ora se la cartella esiste
    # gia') e' decisa a monte da pipeline.run(), PRIMA di scrivere split_assignments.csv,
    # cosi' che tutti gli artefatti di una run (split_assignments, log, metrics, checkpoint)
    # finiscano nella stessa cartella. Qui la cartella esiste gia': non va rinominata.
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger = _build_logger(checkpoint_dir)
    logger.info(f"Checkpoint dir: {checkpoint_dir}")
    logger.info(
        f"num_epochs={num_epochs} lr={lr} device={device} alpha={alpha} "
        f"lpips_input_mode={lpips_input_mode} "
        f"patience_early_stopping={patience_early_stopping} patience_lr_scheduler={patience_lr_scheduler}"
    )

    metrics_path = os.path.join(checkpoint_dir, "metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "lr", "is_best"])

    best_val_loss = float('inf')  # traccia del best
    epochs_no_improve = 0         # contatore per early stopping
    train_losses = []
    val_losses = []

    try:
        for epoch in range(num_epochs):
            logger.info(f"=== EPOCH {epoch+1}/{num_epochs} ===")

            # Azzera i contatori di range LPIPS: verranno riempiti da train+val
            # di questa epoca e riassunti in un unico log qui sotto (invece di
            # un warning ad ogni batch).
            reset_lpips_range_stats()

            # Training
            train_loss = train_one_epoch_3d(
                model, train_loader, optimizer, device=device, alpha=alpha,
                lpips_input_mode=lpips_input_mode,
            )

            # Validazione
            val_loss = evaluate_model_3d(
                model, val_loader, device=device, alpha=alpha,
                lpips_input_mode=lpips_input_mode,
            )

            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"[TRAIN] MSE-Lpips Loss: {train_loss:.4f}")
            logger.info(f"[VAL  ] MSE-Lpips Loss: {val_loss:.4f}")
            logger.info(f"[LR   ] {current_lr:.6f}")

            # Report aggregato (una volta per epoca) di quanti output erano
            # fuori dal range [-1,1] atteso da LPIPS, indipendentemente da
            # lpips_input_mode (che corregge solo l'input a LPIPS, non lo
            # segnala).
            range_stats = get_lpips_range_stats()
            if range_stats["outputs_total"] > 0:
                out_of_range = range_stats["outputs_out_of_range"]
                total = range_stats["outputs_total"]
                pct = 100.0 * out_of_range / total
                logger.info(
                    f"[LPIPS] outputs fuori da [-1,1]: {out_of_range}/{total} "
                    f"elementi ({pct:.2f}%) — lpips_input_mode={lpips_input_mode}"
                )

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Step dello scheduler su metrica di validazione
            scheduler.step(val_loss)

            # Salva best se migliora
            is_best = val_loss < best_val_loss - threshold
            if is_best:
                best_val_loss = val_loss
                epochs_no_improve = 0  # reset per early stopping
                best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save(_unwrapped_state_dict(model), best_model_path)
                logger.info(f"  -> Val loss migliorata. Best model salvato in '{best_model_path}'")
            else:
                # Non migliora abbastanza: incrementa contatore
                epochs_no_improve += 1

            with open(metrics_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([epoch + 1, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{current_lr:.8f}", int(is_best)])

            # Checkpoint periodico
            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
                torch.save(_unwrapped_state_dict(model), checkpoint_path)
                logger.info(f"  -> Checkpoint salvato in '{checkpoint_path}'")

            # Early Stopping manuale
            if epochs_no_improve >= patience_early_stopping:
                logger.info("Early stopping attivato (nessun miglioramento sufficiente).")
                break

    except KeyboardInterrupt:
        logger.warning("Interruzione da tastiera. Procedo con salvataggi e plot.")

    # Salva il modello dell'ultima epoca
    last_model_path = os.path.join(checkpoint_dir, "last_model.pth")
    torch.save(_unwrapped_state_dict(model), last_model_path)
    logger.info(f"Modello dell'ultima epoca salvato in '{last_model_path}'")

    # Plot finale delle loss per epoca
    plt.figure()
    epochs_range = range(1, len(train_losses) + 1)
    plt.plot(epochs_range, train_losses, 'b-o', label="Train Loss")
    plt.plot(epochs_range, val_losses,   'r-o', label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE-Lpips Loss")
    plt.title("Train vs. Validation Loss")
    plt.legend()
    plt.grid(True)
    # Save figure to checkpoint directory and optionally show
    out_plot_path = os.path.join(checkpoint_dir, "train_val_loss.png")
    try:
        plt.savefig(out_plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"Train/Val loss plot saved to {out_plot_path}")
    except Exception as e:
        logger.warning(f"Could not save plot to {out_plot_path}: {e}")

    if show_plots:
        try:
            plt.show()
        except Exception:
            # In headless environments plt.show() may fail; ignore
            pass
    else:
        plt.close()

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "checkpoint_dir": checkpoint_dir,
    }
