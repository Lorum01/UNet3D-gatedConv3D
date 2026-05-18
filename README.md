# Unet3D + GatedConv3D (Packet Project) — 05/2026

Conversione del notebook `Unet3D_ConvLSTM3D_Model_1.ipynb` in un progetto organizzato in pacchetti (cartella `src/`) mantenendo la stessa logica di caricamento/manipolazione dati, split e uso del modello.

## Avvio rapido

Da questa cartella:

- Train:
  - `python scripts/run.py --config configs/train.yaml`
- Solo inferenza:
  - `python scripts/run.py --config configs/infer.yaml`

## Note

- I path del dataset e dell’Excel sono configurabili nei file YAML sotto `configs/`.
- In inferenza puoi scegliere su quali split eseguire (test/val/train) tramite `infer.run_*`.

