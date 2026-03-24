import numpy as np
from collections import Counter

import torch.nn as nn
import torch
import argparse
import yaml
import difflib

from Unet3D_StackedConv3D import StackedConv3D, UNet3D
from train import training_loop_with_validation_3d
from test_utility import test_model_create_gifs_3ch

from utility.input_utility import (
    load_series_from_folders,    
    load_event_classes_from_excel,
    create_sequences_multiple_series_fixed_input,
    check_range_of_images,
    plot_sequence
)

from torch.utils.data import DataLoader, Subset
from utility.dataloader import (
    CustomDataset,
    pct_to_counts,
    split_by_class_distribution,
    compute_mean_std,
    report_split_coverage,
    check_dataset_range,
    custom_collate_fn,
    show_images_from_batch
) 

#  ---------------------------------------------------------------------
#       INPUT PARAMETERS
#  ---------------------------------------------------------------------

# ----- Parametri di utilizzo rapido -----
DATASET_FOLDER = "./Dataset"              # Cartella con sottocartelle degli eventi
EXCEL_PATH  = "EventsEtnaCLASS_Final.xlsx"   # File Excel con colonna "Class"

# Parametri per la generazione delle sequenze
INPUT_LENGTH       = 4
PREDICTION_LENGTH  = 4
STRIDE             = 1

# Shape attesa
EXPECTED_INPUT_SHAPE = (4, 100, 100, 3)  # (T, H, W, C) se lavori con 3 canali

#  ---------------------------------------------------------------------
#       DATALOADER SETUP
#  ---------------------------------------------------------------------

# Oggetti attesi già creati dalla fase precedente del notebook:
#   all_inputs, all_targets, labels, input_fnames, target_fnames

# Scelta standardizzazione/scaling per il dataset finale
SCALE_TO_NEG1_POS1 = True     # True: mappa [0,1] -> [-1,1]; False: nessuna mappatura
#USE_MEAN_STD       = False    # Se vuoi usare mean/std (invece di scaling), setta True e SCALE_TO_NEG1_POS1=False (sconsigliato)

# Distribuzione per classe (fissa)
CLASS_DISTRIBUTION = {
    1: {"train":26, "val":5,  "test":3},
    2: {"train":7,  "val":2,  "test":1},
    3: {"train":28, "val":6,  "test":4},
    4: {"train":6,  "val":2,  "test":1},
}

# Opzione: usare percentuali invece di conteggi fissi
USE_PERCENT_DISTRIBUTION = True   # True per usare CLASS_PCT -> conteggi via pct_to_counts (evitare False -> da correggere)
# Percentuali di default per tutte le classi
CLASS_PCT = {
    1: {"train":0.7, "val":0.15, "test":0.15},
    2: {"train":0.7, "val":0.15, "test":0.15},
    3: {"train":0.7, "val":0.15, "test":0.15},
    4: {"train":0.7, "val":0.15, "test":0.15},
}

# DataLoader
BATCH_SIZE_TRAIN = 4
BATCH_SIZE_VAL   = 4
BATCH_SIZE_TEST  = 4

#  ---------------------------------------------------------------------
#       MODEL PARAMETERS
#  ---------------------------------------------------------------------

# Parametri modello
UNET_IN_CHANNELS   = 3
UNET_BASE_CHANNELS = 32
UNET_NUM_LEVELS    = 5
UNET_OUT_CHANNELS  = 64

STACKEDCONV_IN_DIM   = UNET_OUT_CHANNELS # deve corrispondere all'output dell'UNet
STACKEDCONV_HIDDENS  = [64, 128, 256, 512]
STACKEDCONV_K        = 3
STACKEDCONV_PADDING  = 1
STACKEDCONV_OUT_DIM   = 512

FINAL_OUT_CHANNELS = 3  # canali output del modello finale

# #  ---------------------------------------------------------------------
# #       TRAINING PARAMETERS
# #  ---------------------------------------------------------------------
# # Flag per abilitare il training (se False salta la fase di training)
# TRAIN = False

# # Iperparametri per la funzione `training_loop_with_validation_3d`
# NUM_EPOCHS = 50
# LR = 1e-3
# DEVICE = "cuda"  # usa "cpu" se non è disponibile CUDA
# PATIENCE_EARLY_STOPPING = 10
# PATIENCE_LR_SCHEDULER = 6
# LR_FACTOR = 0.5
# LR_THRESHOLD = 1e-4
# CHECKPOINT_INTERVAL = 1
# CHECKPOINT_DIR = "Checkpoints"
# ALPHA = 0.5

# #  ---------------------------------------------------------------------
# #       TEST SETUP
# #  ---------------------------------------------------------------------

# # Parametri di test per la funzione `test_model_create_gifs_3ch`
# # Modifica questi valori per controllare comportamento e destinazioni dei risultati di test
# TEST_DEVICE = "cuda"                 # "cuda" o "cpu" o torch.device
# TEST_SAVE_DIR_TEST  = "Unet3D_Results_1/test"
# TEST_SAVE_DIR_VAL   = "Unet3D_Results_1/val"
# TEST_SAVE_DIR_TRAIN = "Unet3D_Results_1/train"
# TEST_MAX_BATCHES_DEFAULT = 150
# TEST_GIF_FPS = 2
# TEST_MEAN = None   # mean per canale per denormalizzazione (None -> clamp [0,1])
# TEST_STD  = None   # std per canale per denormalizzazione
# TEST_CHECKPOINT_PATH = "../convUnet_Best_Test_mod_1/best_model.pth"
# TEST_SAVE_IMAGES = True
# TEST_SAVE_GIFS  = True
# TEST_SHOW_PLOTS = False




#  ---------------------------------------------------------------------
#   MAIN FUNCTION
#  ---------------------------------------------------------------------

def main():
    # --- Argparse + YAML config support ---
    parser = argparse.ArgumentParser(description="Run training and testing for Unet3D model")
    parser.add_argument("--config", "-c", type=str, help="YAML config file to override defaults")
    parser.add_argument("--no-train", action="store_true", help="Skip training phase")
    parser.add_argument("--device", type=str, help="Device to use, e.g. cuda or cpu")
    parser.add_argument("--checkpoint-dir", type=str, help="Checkpoint directory to override")
    parser.add_argument("--test-checkpoint", type=str, help="Checkpoint path for tests")
    args = parser.parse_args()

    # Build config from module-level defaults. Use globals().get(...) so
    # references don't trigger UnboundLocalError when local names are assigned
    # later in this function.
    g = globals()
    cfg = {
        "TRAIN": g.get("TRAIN", False),
        "NUM_EPOCHS": g.get("NUM_EPOCHS", 50),
        "LR": g.get("LR", 1e-3),
        "DEVICE": g.get("DEVICE", "cuda"),
        "PATIENCE_EARLY_STOPPING": g.get("PATIENCE_EARLY_STOPPING", 10),
        "PATIENCE_LR_SCHEDULER": g.get("PATIENCE_LR_SCHEDULER", 6),
        "LR_FACTOR": g.get("LR_FACTOR", 0.5),
        "LR_THRESHOLD": g.get("LR_THRESHOLD", 1e-4),
        "CHECKPOINT_INTERVAL": g.get("CHECKPOINT_INTERVAL", 1),
        "CHECKPOINT_DIR": g.get("CHECKPOINT_DIR", "Checkpoints"),
        "ALPHA": g.get("ALPHA", 0.5),
        "TEST_DEVICE": g.get("TEST_DEVICE", "cuda"),
        "TEST_SAVE_DIR_TEST": g.get("TEST_SAVE_DIR_TEST", "Model_Results_1/test"),
        "TEST_SAVE_DIR_VAL": g.get("TEST_SAVE_DIR_VAL", "Model_Results_1/val"),
        "TEST_SAVE_DIR_TRAIN": g.get("TEST_SAVE_DIR_TRAIN", "Model_Results_1/train"),
        "TEST_MAX_BATCHES_DEFAULT": g.get("TEST_MAX_BATCHES_DEFAULT", 150),
        "TEST_GIF_FPS": g.get("TEST_GIF_FPS", 2),
        "TEST_MEAN": g.get("TEST_MEAN", None),
        "TEST_STD": g.get("TEST_STD", None),
        "TEST_CHECKPOINT_PATH": g.get("TEST_CHECKPOINT_PATH", "../convUnet_Best_Test_mod_1/best_model.pth"),
        "TEST_SAVE_IMAGES": g.get("TEST_SAVE_IMAGES", True),
        "TEST_SAVE_GIFS": g.get("TEST_SAVE_GIFS", True),
        "TEST_SHOW_PLOTS": g.get("TEST_SHOW_PLOTS", False),
        "TEST_RUN_TEST_SET": g.get("TEST_RUN_TEST_SET", True),
        "TEST_RUN_VAL_SET": g.get("TEST_RUN_VAL_SET", False),
        "TEST_RUN_TRAIN_SET": g.get("TEST_RUN_TRAIN_SET", False),
    }

    # Load YAML config if provided and update
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            ycfg = yaml.safe_load(f) or {}

        # Validate YAML keys against allowed config keys and suggest close matches
        allowed_keys = set(cfg.keys())
        unknown_keys = [k for k in ycfg.keys() if k not in allowed_keys]
        if unknown_keys:
            suggestions = {}
            for uk in unknown_keys:
                close = difflib.get_close_matches(uk, allowed_keys, n=3, cutoff=0.6)
                suggestions[uk] = close
            msg_lines = ["Unknown config keys found in YAML:"]
            for uk in unknown_keys:
                s = suggestions.get(uk) or []
                if s:
                    msg_lines.append(f"  - {uk}  (did you mean: {', '.join(s)})")
                else:
                    msg_lines.append(f"  - {uk}")
            msg_lines.append("\nAllowed keys are: " + ", ".join(sorted(allowed_keys)))
            raise ValueError("\n".join(msg_lines))

        # flat update: keys in YAML should match cfg keys
        cfg.update(ycfg)

    # Apply CLI overrides
    if args.no_train:
        cfg["TRAIN"] = False
    if args.device:
        cfg["DEVICE"] = args.device
    if args.checkpoint_dir:
        cfg["CHECKPOINT_DIR"] = args.checkpoint_dir
    if args.test_checkpoint:
        cfg["TEST_CHECKPOINT_PATH"] = args.test_checkpoint

    # Expose config as local variables used below
    TRAIN = cfg["TRAIN"]
    NUM_EPOCHS = cfg["NUM_EPOCHS"]
    LR = cfg["LR"]
    DEVICE = cfg["DEVICE"]
    PATIENCE_EARLY_STOPPING = cfg["PATIENCE_EARLY_STOPPING"]
    PATIENCE_LR_SCHEDULER = cfg["PATIENCE_LR_SCHEDULER"]
    LR_FACTOR = cfg["LR_FACTOR"]
    LR_THRESHOLD = cfg["LR_THRESHOLD"]
    CHECKPOINT_INTERVAL = cfg["CHECKPOINT_INTERVAL"]
    CHECKPOINT_DIR = cfg["CHECKPOINT_DIR"]
    ALPHA = cfg["ALPHA"]

    TEST_DEVICE = cfg["TEST_DEVICE"]
    TEST_SAVE_DIR_TEST = cfg["TEST_SAVE_DIR_TEST"]
    TEST_SAVE_DIR_VAL = cfg["TEST_SAVE_DIR_VAL"]
    TEST_SAVE_DIR_TRAIN = cfg["TEST_SAVE_DIR_TRAIN"]
    TEST_MAX_BATCHES_DEFAULT = cfg["TEST_MAX_BATCHES_DEFAULT"]
    TEST_GIF_FPS = cfg["TEST_GIF_FPS"]
    TEST_MEAN = cfg["TEST_MEAN"]
    TEST_STD = cfg["TEST_STD"]
    TEST_CHECKPOINT_PATH = cfg["TEST_CHECKPOINT_PATH"]
    TEST_SAVE_IMAGES = cfg["TEST_SAVE_IMAGES"]
    TEST_SAVE_GIFS = cfg["TEST_SAVE_GIFS"]
    TEST_SHOW_PLOTS = cfg["TEST_SHOW_PLOTS"]

    #  ---------------------------------------------------------------------
    #   EXECUTION: INPUT CREATION
    #  ---------------------------------------------------------------------

    # 1) Carica le serie e i filenames
    all_series, all_series_filenames = load_series_from_folders(DATASET_FOLDER)
    print(f"Numero totale di serie caricate: {len(all_series)}")
    if len(all_series) > 0:
        print(f"Dimensione della prima serie: {all_series[0].shape}")

    # 2) Carica la classe di ogni evento dal file Excel
    event_class_dict = load_event_classes_from_excel(EXCEL_PATH)

    # Crea array classi allineato alle serie
    all_classes = [event_class_dict[i] for i, _ in enumerate(all_series)]
    print("Esempio classi assegnate:", all_classes[:5])

    # 3) Distribuzione classi
    class_counts = Counter(all_classes)
    print("Numero totale serie per ogni classe:")
    for classe, count in class_counts.items():
        print(f"{classe}: {count}")

    # 4) Crea sequenze
    result = create_sequences_multiple_series_fixed_input(
        all_series=all_series,
        all_classes=all_classes,
        input_length=INPUT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        stride=STRIDE,
        all_series_filenames=all_series_filenames
    )

    # Unpack (con o senza filenames)
    if len(result) == 5:
        all_inputs, all_targets, labels, input_fnames, target_fnames = result
    else:
        all_inputs, all_targets, labels = result
        input_fnames, target_fnames = None, None

    print("Dimensione totale input:", all_inputs.shape)
    print("Dimensione totale target:", all_targets.shape)
    print("Dimensione totale labels:", labels.shape)
    if input_fnames is not None:
        print("Dimensione input_fnames:", input_fnames.shape)
    if target_fnames is not None:
        print("Dimensione target_fnames:", target_fnames.shape)

    # 5) Check range valori (prima di normalizzare)
    check_range_of_images(all_inputs, all_targets)

    # 6) Controllo shape attesa (opzionale)
    invalid_inputs  = [idx for idx, inp in enumerate(all_inputs)  if inp.shape != EXPECTED_INPUT_SHAPE]
    invalid_targets = [idx for idx, tgt in enumerate(all_targets) if tgt.shape != EXPECTED_INPUT_SHAPE]

    if not invalid_inputs:
        print(f"Tutte le sequenze di input hanno la forma attesa: {EXPECTED_INPUT_SHAPE}")
    else:
        print(f"Ci sono {len(invalid_inputs)} sequenze di input con forme non attese.")
        for idx in invalid_inputs[:5]:
            print(f" - Input sequence at index {idx} ha forma {all_inputs[idx].shape}")

    if not invalid_targets:
        print(f"Tutte le sequenze di target hanno la forma attesa: {EXPECTED_INPUT_SHAPE}")
    else:
        print(f"Ci sono {len(invalid_targets)} sequenze di target con forme non attese.")
        for idx in invalid_targets[:5]:
            print(f" - Target sequence at index {idx} ha forma {all_targets[idx].shape}")

    # Visualizzazione campioni casuali disabilitata su richiesta.
    # Se vuoi riattivarla, decommenta il blocco sottostante.
    #
    # # 7) Visualizza campioni casuali (solo se tutte le shape hanno la forma attesa)
    # if not invalid_inputs and not invalid_targets:
    #     num_samples_to_plot = min(5, len(all_inputs))
    #     sample_indices = np.random.choice(len(all_inputs), num_samples_to_plot, replace=False)
    #
    #     print(f"\nVisualizzazione di {num_samples_to_plot} campioni casuali:")
    #     for idx in sample_indices:
    #         print(f"\nCampione {idx}:")
    #         print(f" - Classe: {labels[idx]}")
    #         print(f" - Input shape: {all_inputs[idx].shape}")
    #         print(f" - Target shape: {all_targets[idx].shape}")
    #         if input_fnames is not None:
    #             print(f" - Input filenames: {input_fnames[idx]}")
    #         if target_fnames is not None:
    #             print(f" - Target filenames: {target_fnames[idx]}")
    #
    #         if input_fnames is not None:
    #             plot_sequence(all_inputs[idx], title=f"Input Sequence (Class: {labels[idx]})", fnames=input_fnames[idx])
    #         else:
    #             plot_sequence(all_inputs[idx], title=f"Input Sequence (Class: {labels[idx]})")
    #
    #         if target_fnames is not None:
    #             plot_sequence(all_targets[idx], title=f"Target Sequence (Class: {labels[idx]})", fnames=target_fnames[idx])
    #         else:
    #             plot_sequence(all_targets[idx], title=f"Target Sequence (Class: {labels[idx]})")
    # else:
    #     print("\nVerifica interrotta a causa di forme non valide nelle sequenze.")


    #  ---------------------------------------------------------------------
    #   EXECUTION: DATALOADER SETUP
    #  ---------------------------------------------------------------------


    # 1) Cast a float32
    all_inputs_list  = [x.astype(np.float32) for x in all_inputs]
    all_targets_list = [x.astype(np.float32) for x in all_targets]

    # 2) Dataset provvisorio per calcolo mean/std
    temp_dataset = CustomDataset(all_inputs_list, all_targets_list, labels)

    # 3) Split fisso per classe
    if USE_PERCENT_DISTRIBUTION:
        class_distribution = pct_to_counts(CLASS_PCT, labels, round_method='round')
    else:
        class_distribution = CLASS_DISTRIBUTION

    train_idx, val_idx, test_idx = split_by_class_distribution(labels, class_distribution, shuffle=True, seed=424)
    print("Class distribution used for split:", class_distribution)


    # 4) Mean/Std (calcolati sui soli train_idx)
    mean_c, std_c = compute_mean_std(temp_dataset, train_idx)
    print("Mean canali:", mean_c)
    print("Std  canali:", std_c)
    #final_mean = mean_c if (USE_MEAN_STD and not SCALE_TO_NEG1_POS1) else None
    #final_std  = std_c  if (USE_MEAN_STD and not SCALE_TO_NEG1_POS1) else None

    full_dataset = CustomDataset(
        all_inputs_list,
        all_targets_list,
        labels,
        #mean=final_mean,
        #std=final_std,
        scale_to_neg1_pos1=SCALE_TO_NEG1_POS1,
        input_filenames=input_fnames,
        target_filenames=target_fnames
    )

    # 6) Report split
    report_split_coverage(train_idx, val_idx, test_idx, total_len=len(full_dataset))

    # 7) Subset e DataLoader
    train_subset = Subset(full_dataset, train_idx)
    val_subset   = Subset(full_dataset, val_idx)
    test_subset  = Subset(full_dataset, test_idx)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE_TRAIN, shuffle=True,  collate_fn=custom_collate_fn)
    val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE_VAL,   shuffle=False, collate_fn=custom_collate_fn)
    test_loader  = DataLoader(test_subset, batch_size=BATCH_SIZE_TEST,  shuffle=False, collate_fn=custom_collate_fn)

    # 8) Check range (standardized=True perché potresti aver scalato o standardizzato)
    check_dataset_range(full_dataset, standardized=True)

    # # 9) (Opzionale) esempio di un batch dal train_loader con visualizzazione
    # for batch_idx, (data, targets, lbls, filenames) in enumerate(test_loader):
    #     input_fnames_batch, target_fnames_batch = filenames
    #     print(f"[TRAIN] Batch index: {batch_idx}")
    #     print("Data shape:   ", data.shape)     # (B, T, C, H, W)
    #     print("Target shape: ", targets.shape)  # (B, T, C, H, W)
    #     print("Labels shape: ", lbls.shape)     # (B,)
    #     if input_fnames_batch[0] is not None:
    #         print("  Primo campione, input filenames:", input_fnames_batch[0])
    #     if target_fnames_batch[0] is not None:
    #         print("  Primo campione, target filenames:", target_fnames_batch[0])
    #     show_images_from_batch(
    #         data, targets, lbls,
    #         max_samples=4,
    #         input_fnames_batch=input_fnames_batch,
    #         target_fnames_batch=target_fnames_batch
    #     )
    #     break

    # ---------------------------------------------------------------------
    # MODEL CREATION: UNet + StackedConv3D
    # ---------------------------------------------------------------------
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            # Estrazione di feature UNet 3D
            self.unet = UNet3D(in_channels=UNET_IN_CHANNELS, base_channels=UNET_BASE_CHANNELS, num_levels=UNET_NUM_LEVELS, out_channels=UNET_OUT_CHANNELS)
            # Elaborazione temporale con StackedConv3D
            self.stackedConv = StackedConv3D(input_dim=STACKEDCONV_IN_DIM, hidden_dims=STACKEDCONV_HIDDENS, kernel_size=STACKEDCONV_K, padding=STACKEDCONV_PADDING)
            # Convoluzione finale per riprodurre il numero di canali di output (es. 3)
            self.final_conv = nn.Conv3d(STACKEDCONV_OUT_DIM, FINAL_OUT_CHANNELS, kernel_size=1)
            
        def forward(self, x):
            """
            x: tensor di dimensione (B, 3, T, H, W)
            """
            features = self.unet(x)       # -> (B, 64, T, H', W')
            stackedConv_out = self.stackedConv(features)  # -> (B, 256, T, H', W')
            output   = self.final_conv(stackedConv_out)  # -> (B, 3, T, H', W')
            return output

    # ---------------------------------------------------------------------
    # TRAINING SETUP AND EXECUTION
    # ---------------------------------------------------------------------

    model = Model()
    if TRAIN == True:
        train_losses, val_losses = training_loop_with_validation_3d(
            model,
            train_loader,
            val_loader,
            num_epochs=NUM_EPOCHS,
            lr=LR,
            device=DEVICE,
            patience_early_stopping=PATIENCE_EARLY_STOPPING,
            patience_lr_scheduler=PATIENCE_LR_SCHEDULER,
            lr_factor=LR_FACTOR,
            lr_threshold=LR_THRESHOLD,
            checkpoint_interval=CHECKPOINT_INTERVAL,
            checkpoint_dir=CHECKPOINT_DIR,
            alpha=ALPHA,
            show_plots=TEST_SHOW_PLOTS
        )


    # ---------------------------------------------------------------------
    # TEST SETUP AND EXECUTION
    # ---------------------------------------------------------------------

    # Test set
    if cfg["TEST_RUN_TEST_SET"]:
        test_model_create_gifs_3ch(
            model=model,
            test_loader=test_loader,
            device=TEST_DEVICE,
            save_dir=TEST_SAVE_DIR_TEST,
            max_batches=TEST_MAX_BATCHES_DEFAULT,
            gif_fps=TEST_GIF_FPS,
            checkpoint_path=TEST_CHECKPOINT_PATH,
            save_images=TEST_SAVE_IMAGES,
            save_gifs=TEST_SAVE_GIFS,
            show_plots=TEST_SHOW_PLOTS,
        )

    # Validation set
    if cfg["TEST_RUN_VAL_SET"]:
        test_model_create_gifs_3ch(
            model=model,
            test_loader=val_loader,
            device=TEST_DEVICE,
            save_dir=TEST_SAVE_DIR_VAL,
            max_batches=TEST_MAX_BATCHES_DEFAULT,
            gif_fps=TEST_GIF_FPS,
            checkpoint_path=TEST_CHECKPOINT_PATH,
            save_images=TEST_SAVE_IMAGES,
            save_gifs=TEST_SAVE_GIFS,
            show_plots=TEST_SHOW_PLOTS,
        )

    # Training set
    if cfg["TEST_RUN_TRAIN_SET"]:
        test_model_create_gifs_3ch(
            model=model,
            test_loader=train_loader,
            device=TEST_DEVICE,
            save_dir=TEST_SAVE_DIR_TRAIN,
            max_batches=TEST_MAX_BATCHES_DEFAULT,
            gif_fps=TEST_GIF_FPS,
            checkpoint_path=TEST_CHECKPOINT_PATH,
            save_images=TEST_SAVE_IMAGES,
            save_gifs=TEST_SAVE_GIFS,
            show_plots=TEST_SHOW_PLOTS,
        )


if __name__ == "__main__":
    main()

