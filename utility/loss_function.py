import torch
import torch.nn.functional as F
import lpips
import contextlib
import io

def weighted_mse_lpips_loss(outputs, targets, alpha=0.5):
    """
    Calcola una loss combinata come media pesata tra MSE e LPIPS.

    Args:
        outputs: Predizioni del modello con forma (B, C, T, H, W).
        targets: Target con forma (B, C, T, H, W).
        alpha (float): Peso assegnato alla MSE. Il peso della LPIPS sarà (1 - alpha).

    Note:
        - LPIPS valuta la differenza percettiva tra immagini RGB (C=3) nel range [-1,1].
        - Se le immagini non sono normalizzate in questo range, la metrica perde significato.
        - La LPIPS viene calcolata frame per frame lungo la dimensione temporale T.
    """
    # Calcolo standard della MSE Loss
    mse_loss = F.mse_loss(outputs, targets)
    
    # Crea il modello LPIPS
    with contextlib.redirect_stdout(io.StringIO()):
        loss_fn = lpips.LPIPS(net='alex', version='0.1')
    loss_fn.to(outputs.device)

    # Dimensioni attese: (B, C, T, H, W)
    B, C, T, H, W = outputs.shape
    lpips_loss_total = 0.0

    # Ciclo sui frame temporali per calcolare la LPIPS media
    for t in range(T):
        # Frame corrente del batch (B, C, H, W)
        out_frame = outputs[:, :, t, :, :]
        tgt_frame = targets[:, :, t, :, :]

        # LPIPS restituisce un tensor di forma (B, 1, 1, 1)
        lpips_frame = loss_fn(out_frame, tgt_frame)
        lpips_loss_total += lpips_frame.mean()

    # Media temporale della LPIPS
    lpips_loss = lpips_loss_total / T
    
    # Loss combinata pesata
    combined_loss = alpha * mse_loss + (1 - alpha) * lpips_loss
    return combined_loss
