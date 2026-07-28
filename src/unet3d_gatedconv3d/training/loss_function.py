import contextlib
import functools
import io
import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Evita di ripetere il warning "lpips non installato" ad ogni batch.
_lpips_missing_warned = False

# Tolleranza numerica per il controllo del range atteso da LPIPS ([-1, 1]).
_LPIPS_RANGE_ATOL = 1e-3

# Modalità supportate per preparare `outputs` prima di passarlo a LPIPS.
_VALID_LPIPS_INPUT_MODES = ("none", "clamp", "tanh")

# Contatori accumulati silenziosamente (nessun log ad ogni batch): il chiamante
# li legge/azzera a intervalli propri, es. una volta per epoca in train_loop.py.
_lpips_range_stats = {
    "outputs_out_of_range": 0,
    "outputs_total": 0,
    "targets_out_of_range": 0,
    "targets_total": 0,
}


@functools.lru_cache(maxsize=None)
def _get_lpips_model(device_str):
    """Crea (una sola volta per device) e mette in cache il modello LPIPS."""
    import lpips  # type: ignore

    with contextlib.redirect_stdout(io.StringIO()):
        model = lpips.LPIPS(net='alex', version='0.1')
    return model.to(device_str).eval()


def _record_lpips_range(tensor, key_prefix):
    """Accumula (senza loggare) quanti elementi escono dal range [-1, 1] atteso da LPIPS."""
    out_of_range = (tensor < -1.0 - _LPIPS_RANGE_ATOL) | (tensor > 1.0 + _LPIPS_RANGE_ATOL)
    _lpips_range_stats[f"{key_prefix}_out_of_range"] += int(out_of_range.sum().item())
    _lpips_range_stats[f"{key_prefix}_total"] += tensor.numel()


def get_lpips_range_stats():
    """Ritorna una copia dei contatori di range accumulati dall'ultimo reset."""
    return dict(_lpips_range_stats)


def reset_lpips_range_stats():
    """Azzera i contatori di range (da chiamare, es., una volta per epoca)."""
    for key in _lpips_range_stats:
        _lpips_range_stats[key] = 0


def _prepare_lpips_input(tensor, mode):
    """Porta `tensor` nel range [-1, 1] atteso da LPIPS, secondo `mode` ('none'/'clamp'/'tanh')."""
    if mode == "clamp":
        return tensor.clamp(-1.0, 1.0)
    if mode == "tanh":
        return torch.tanh(tensor)
    return tensor  # "none": nessuna trasformazione (comportamento originale)


def weighted_mse_lpips_loss(outputs, targets, alpha=0.5, lpips_input_mode="none"):
    """
    Calcola una loss combinata come media pesata tra MSE e LPIPS.

    Args:
        outputs: Predizioni del modello con forma (B, C, T, H, W).
        targets: Target con forma (B, C, T, H, W).
        alpha (float): Peso assegnato alla MSE. Il peso della LPIPS sarà (1 - alpha).
        lpips_input_mode (str): come preparare `outputs` solo per il ramo LPIPS
            (la MSE è sempre calcolata sugli `outputs` originali, non trasformati):
            - "none":  nessuna trasformazione (default; comportamento originale).
            - "clamp": torch.clamp(outputs, -1, 1).
            - "tanh":  torch.tanh(outputs).

    Note:
        - LPIPS valuta la differenza percettiva tra immagini RGB (C=3) nel range [-1,1].
        - Se le immagini non sono normalizzate in questo range, la metrica perde significato.
        - La LPIPS viene calcolata frame per frame lungo la dimensione temporale T.
        - Il modello LPIPS viene istanziato una sola volta per device e riutilizzato
          (cache a livello di modulo via functools.lru_cache).
        - Gli elementi fuori dal range [-1,1] vengono solo conteggiati qui (vedi
          get_lpips_range_stats/reset_lpips_range_stats); il logging aggregato è
          responsabilità del chiamante.
    """
    global _lpips_missing_warned

    if lpips_input_mode not in _VALID_LPIPS_INPUT_MODES:
        raise ValueError(
            f"lpips_input_mode deve essere uno tra {_VALID_LPIPS_INPUT_MODES}, "
            f"ricevuto {lpips_input_mode!r}"
        )

    # Calcolo standard della MSE Loss (sempre sugli outputs originali, non trasformati)
    mse_loss = F.mse_loss(outputs, targets)

    # Import lazy: in alcuni ambienti `lpips` potrebbe non essere installato.
    # In quel caso, ripiega su MSE pura (mantiene l'esecuzione del progetto).
    try:
        loss_fn = _get_lpips_model(str(outputs.device))
    except ModuleNotFoundError:
        if not _lpips_missing_warned:
            logger.warning(
                "Modulo 'lpips' non installato: weighted_mse_lpips_loss ripiega su MSE pura "
                "(il termine percettivo pesato (1 - alpha) non viene calcolato)."
            )
            _lpips_missing_warned = True
        return mse_loss

    # Conteggio (silenzioso) di quanti elementi escono dal range atteso da LPIPS.
    _record_lpips_range(outputs, "outputs")
    _record_lpips_range(targets, "targets")

    # Dimensioni attese: (B, C, T, H, W)
    B, C, T, H, W = outputs.shape
    lpips_loss_total = 0.0

    lpips_outputs = _prepare_lpips_input(outputs, lpips_input_mode)

    # Ciclo sui frame temporali per calcolare la LPIPS media
    for t in range(T):
        # Frame corrente del batch (B, C, H, W)
        out_frame = lpips_outputs[:, :, t, :, :]
        tgt_frame = targets[:, :, t, :, :]

        # LPIPS restituisce un tensor di forma (B, 1, 1, 1)
        lpips_frame = loss_fn(out_frame, tgt_frame)
        lpips_loss_total += lpips_frame.mean()

    # Media temporale della LPIPS
    lpips_loss = lpips_loss_total / T

    # Loss combinata pesata
    combined_loss = alpha * mse_loss + (1 - alpha) * lpips_loss
    return combined_loss
