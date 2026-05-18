
import torch.optim as optim

def create_scheduler(optimizer, factor=0.5, patience=3, threshold=1e-4):
    """
    Crea un ReduceLROnPlateau Scheduler che riduce il learning rate 
    quando la val_loss non migliora per 'patience' epoche.

    Args:
        optimizer (torch.optim.Optimizer): Ottimizzatore su cui agire.
        factor (float): Fattore di riduzione del learning rate.
                        Es. 0.5 dimezza il lr.
        patience (int): Numero di epoche senza miglioramento prima di ridurre il lr.
        threshold (float): Miglioramento minimo per essere considerato significativo.

    Returns:
        scheduler (ReduceLROnPlateau): Oggetto scheduler configurato.
    """
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',      # minimizziamo la val_loss
        factor=factor,   # di quanto ridurre il lr
        patience=patience, 
        threshold=threshold,
    )
    return scheduler
