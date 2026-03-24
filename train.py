import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from utility.loss_function import weighted_mse_lpips_loss
from utility.LR_scheduler import create_scheduler

def split_batch(batch):
    """
    Scompone il batch in (data, targets, labels, filenames).
    """
    data, targets = batch[:2]
    labels = batch[2] if len(batch) > 2 else None
    filenames = batch[3] if len(batch) > 3 else None
    return data, targets, labels, filenames


# Training e Validation

def train_one_epoch_3d(model, dataloader, optimizer, device="cuda", alpha=0.5): 
    """
    Esegue una epoca di training su modello 3D.
    Input batch:  (B, T, C, H, W)
    Input modello: (B, C, T, H, W) -> quindi permuta.
    Loss: weighted_mse_lpips_loss(outputs, targets, alpha).
    """
    model.train()
    running_loss = 0.0
    total_batches = 0

    for batch in dataloader:
        data, targets, _, _ = split_batch(batch)

        # Adatta al layout atteso dal modello: (B, T, C, H, W) -> (B, C, T, H, W)
        data = data.permute(0, 2, 1, 3, 4).to(device)
        targets = targets.permute(0, 2, 1, 3, 4).to(device)

        optimizer.zero_grad()
        outputs = model(data)  # (B, C, T, H, W)

        # Loss composita MSE+LPIPS pesata da alpha
        loss = weighted_mse_lpips_loss(outputs, targets, alpha=alpha)
        # Per sola MSE:  loss = F.mse_loss(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_batches += 1

    epoch_loss = running_loss / total_batches if total_batches > 0 else 0.0
    return epoch_loss


@torch.no_grad()
def evaluate_model_3d(model, dataloader, device="cuda", alpha=0.5):
    """
    Valutazione su validation/test set con la stessa loss del training.
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for batch in dataloader:
        data, targets, _, _ = split_batch(batch)
        
        # Stessa permuta del training
        data = data.permute(0, 2, 1, 3, 4).to(device)
        targets = targets.permute(0, 2, 1, 3, 4).to(device)

        outputs = model(data)
        loss = weighted_mse_lpips_loss(outputs, targets, alpha=alpha)
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
    show_plots: bool = False
):
    """
    Loop completo:
    - Ottimizzatore Adam
    - Scheduler ReduceLROnPlateau su val_loss
    - Early stopping se nessun miglioramento per 'patience_early_stopping' epoche
    - Checkpoint: best, last, e periodici
    - Plot finale di train/val loss
    """
    model.to(device)

    # Ottimizzatore
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Scheduler: riduce il LR se la val_loss non migliora
    scheduler = create_scheduler(
        optimizer, factor=factor,
        patience=patience_lr_scheduler,
        threshold=threshold
    )

    # Prepara cartella checkpoint
    # Se la cartella esiste già, per evitare sovrascritture aggiungi un suffisso
    # con data/ora al nome (es. Checkpoints_20251230_153045)
    if os.path.exists(checkpoint_dir):
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        orig_dir = checkpoint_dir
        checkpoint_dir = f"{checkpoint_dir}_{ts}"
        print(f"Checkpoint directory '{orig_dir}' already exists. Using new directory: '{checkpoint_dir}'")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_val_loss = float('inf')  # traccia del best
    epochs_no_improve = 0         # contatore per early stopping
    train_losses = []
    val_losses = []
    
    try:
        for epoch in range(num_epochs):
            print(f"\n=== EPOCH {epoch+1}/{num_epochs} ===")
        
            # Training
            train_loss = train_one_epoch_3d(model, train_loader, optimizer, device=device, alpha=alpha)
        
            # Validazione
            val_loss = evaluate_model_3d(model, val_loader, device=device, alpha=alpha)
        
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[TRAIN] MSE-Lpips Loss: {train_loss:.4f}")
            print(f"[VAL  ] MSE-Lpips Loss: {val_loss:.4f}")
            print(f"[LR   ] {current_lr:.6f}")
        
            train_losses.append(train_loss)
            val_losses.append(val_loss)
        
            # Step dello scheduler su metrica di validazione
            scheduler.step(val_loss)
        
            # Salva best se migliora
            if val_loss < best_val_loss - threshold:
                best_val_loss = val_loss
                epochs_no_improve = 0  # reset per early stopping
                best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save(model.state_dict(), best_model_path)
                print(f"  -> Val loss migliorata. Best model salvato in '{best_model_path}'")
            else:
                # Non migliora abbastanza: incrementa contatore
                epochs_no_improve += 1
        
            # Checkpoint periodico
            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  -> Checkpoint salvato in '{checkpoint_path}'")
        
            # Early Stopping manuale
            if epochs_no_improve >= patience_early_stopping:
                print("Early stopping attivato (nessun miglioramento sufficiente).")
                break

    except KeyboardInterrupt:
        print("\nInterruzione da tastiera. Procedo con salvataggi e plot.")
    
    # Salva il modello dell'ultima epoca
    last_model_path = os.path.join(checkpoint_dir, "last_model.pth")
    torch.save(model.state_dict(), last_model_path)
    print(f"Modello dell'ultima epoca salvato in '{last_model_path}'")
    
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
        print(f"Train/Val loss plot saved to {out_plot_path}")
    except Exception as e:
        print(f"Warning: could not save plot to {out_plot_path}: {e}")

    if show_plots:
        try:
            plt.show()
        except Exception:
            # In headless environments plt.show() may fail; ignore
            pass
    else:
        plt.close()
    
    return {
        "train_losses": train_losses,
        "val_losses": val_losses
    }
