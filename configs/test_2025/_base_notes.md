# Test eventi 2025 — 4 run

Matrice: 2 checkpoint x 2 modalita' di denorm in output.

| file config                     | checkpoint                                      | denorm_from_neg1_pos1 | output root                                        |
|---------------------------------|------------------------------------------------|-----------------------|---------------------------------------------------|
| inference_flip07_remap.yaml     | Checkpoints/Checkpoints_flip_07/best_model.pth  | true  (rimappa (x+1)/2) | Model_Results/test_2025/flip07_remap/by_class     |
| inference_flip07_clamp01.yaml   | Checkpoints/Checkpoints_flip_07/best_model.pth  | false (clamp cieco 0,1) | Model_Results/test_2025/flip07_clamp01/by_class   |
| inference_legacy07_remap.yaml   | Checkpoints/Checkpoint_legacy_07/best_model.pth | true                  | Model_Results/test_2025/legacy07_remap/by_class   |
| inference_legacy07_clamp01.yaml | Checkpoints/Checkpoint_legacy_07/best_model.pth | false                 | Model_Results/test_2025/legacy07_clamp01/by_class |

## Stato: ESEGUITO (2026-08-27)
- `paths.dataset_dir`  -> ./Dataset_test_2025  (2 serie .npy: 2025_06_02 [50 frame], 2025_12_27 [43 frame])
- `paths.excel_path`   -> ./configs/test_2025/EventsTest2025.xlsx (2 righe, Class=2, Split=test; estratte da EventsEtnaCLASS_Final.xlsx)
- 79 finestre totali (43 + 36). Output in Model_Results/test_2025/<run>/by_class/class_2/
- legacy_07: alpha=0.7, lpips_input_mode=none (CONFERMATO dall'utente)

## Note
- `dataset.normalization.mode: neg1pos1` resta invariato in tutte e 4: e' lo scaling
  dell'INPUT al modello e deve combaciare col training. Cambia solo l'override di
  denorm in OUTPUT (`infer.normalization_override.denorm_from_neg1_pos1`).
- `denorm_from_neg1_pos1` influenza sia le immagini/GIF salvate sia le metriche
  (MSE / MSE+LPIPS / SSIM / precision-recall-f1): con `false` l'output del modello
  viene troncato a [0,1] prima del calcolo. => confronti sensati SOLO fra checkpoint
  a parita' di modalita', non fra `remap` e `clamp01`.
- `infer.loss.alpha` / `lpips_input_mode`: flip_07 = 0.7 / none (da training.log).
  legacy_07 = DA CONFERMARE (assunto 0.7 / none dal nome).
- Architettura identica per entrambi i checkpoint (verificata: 267 chiavi, match totale
  con base=32, num_levels=5, out=64, hidden=[64,128,256,512], final=3).
- `split.strategy: by_class` => niente split train/val/test, manda in inferenza TUTTE
  le finestre di TUTTI gli eventi presenti in `dataset_dir`, raggruppate per classe.
