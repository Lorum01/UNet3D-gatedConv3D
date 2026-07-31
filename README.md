# Unet3D + GatedConv3D (Packet Project) - 05/2026

Conversion of the notebook `Unet3D_ConvLSTM3D_Model_1.ipynb` into a packaged project (`src/`), keeping the same data loading/manipulation logic, splitting, and model usage.

## Quick start

From this folder (`cleaned-repo`):

- Train:
  - `python scripts/run.py --config configs/train.yaml`
- Inference (with train/val/test split):
  - `python scripts/run.py --config configs/inference.yaml`
- Inference "by class" (group by class, no split):
  - `python scripts/run.py --config configs/inference_by_class.yaml`

When running inference, the runner also saves in the results folder:
- the console output log (`run_YYYYMMDD-HHMMSS.log`)
- the config file passed via `--config` (copied as-is)
- the resolved/merged config actually used (`config_resolved.yaml`)

## Config file structure

Config YAML files are organized in nested sections that mirror where each setting is used:

```
mode                 # "train" | "infer"
paths:                dataset_dir, excel_path
dataset:               input_length, prediction_length, stride, image_size, channels
  normalization:        mode ("none"|"neg1pos1"|"standardize"), mean, std
  split:                strategy ("train_val_test"|"by_class"), seed, class_percentages
dataloader:            batch_size: {train, val, test}
model:
  unet:                 in_channels, base_channels, num_levels, out_channels
  stacked_conv:         hidden_dims, kernel_size, padding
  final_out_channels
train:                 device, num_epochs, lr, use_data_parallel
  checkpoint:           dir, interval
  early_stopping:       patience
  lr_scheduler:         patience, factor, threshold
  loss:                 alpha
  show_plots
infer:                 device, checkpoint_path, max_batches, gif_fps
  normalization_override: mean, std, denorm_from_neg1_pos1  (all null = derive from dataset.normalization)
  save:                 images, gifs, show_plots, dirs: {test, val, train, by_class_root}
  run:                  test, val, train
```

Any field you omit falls back to the defaults in `src/unet3d_gatedconv3d/config.py`. `dataset.mean`/`dataset.std` (nested under `dataset.normalization`) and `infer.normalization_override.*` are the only fields whose requirement depends on other fields — see "Normalization" below.

## Dataset layout (`Dataset/`)

The loader expects a dataset folder containing **one subfolder per event**; inside each event folder there are the temporal frames stored as `.npy` files.

Example (default in configs: `./Dataset`):

```
Dataset/
  event_0001/
    000.npy
    001.npy
    ...
  event_0002/
    000.npy
    001.npy
    ...
  ...
```

### Loading rules (what the code does)

These rules come from `src/unet3d_gatedconv3d/data/input_utility.py`.

- **Event ordering:** event subfolders are read with `sorted(os.listdir(dataset_dir))`.
  - Therefore the event order depends on the **folder name** (lexicographic order).
- **Frame ordering:** files inside each event folder are read with `sorted(os.listdir(event_folder))`.
  - Therefore the temporal order depends on the **file name**.
  - Recommendation: use zero-padded names (`000.npy`, `001.npy`, ...) to avoid orders like `1.npy, 10.npy, 2.npy`.
- **Frame format (`.npy`):**
  - each `.npy` must contain a NumPy array that is 3D: `(H, W, C)`;
  - each frame is resized to `dataset.image_size` (default **100x100**) during loading;
  - the pipeline is configured to work with `dataset.channels` channels (default **3**, must match `model.unet.in_channels`), so in practice you should store frames with shape **`(H, W, 3)`**.
- **Value range:** values are expected to be in **[0, 1]** before preprocessing.
  - Use `dataset.normalization.mode` to select preprocessing:
    - `none`: no transform
    - `neg1pos1`: map **[0,1] -> [-1,1]**
    - `standardize`: apply **(x - mean) / std** (mean/std computed on train split)
- **Minimum sequence length per event:** to generate at least one valid sequence, each event must have at least:
  - `T >= input_length + prediction_length`
  - and sequences are extracted with a sliding window with step `stride`.

## Excel file layout (event classes + optional split override)

The Excel file (default in configs: `./EventsEtnaCLASS_Final.xlsx`) is used to assign a **class** to each event, and optionally to force a specific **train/val/test split**.

These rules come from `src/unet3d_gatedconv3d/data/input_utility.py` and `src/unet3d_gatedconv3d/pipelines/prep_data.py`.

- The sheet must contain a column named **`Class`** (case-sensitive).
- Excel row **0** is assigned to the **first "base" event folder** loaded from the dataset, row **1** to the second, etc.
  - "Base" means: event folders **not** ending in `_flipped` (see below). Row order must match the order of base event folders after `sorted()`.
  - The number of Excel rows must be **exactly equal** to the number of base event folders (a mismatch raises an error at load time).
- Values in `Class` must be numeric (typically integers: `1, 2, 3, 4`, consistent with `dataset.split.class_percentages` in the configs).

### `_flipped` event folders (augmented mirror variants)

An event folder whose name ends in `_flipped` (e.g. `2021_03_12_flipped`) is treated as an augmented, mirrored copy of the base event with the same name minus the suffix (e.g. `2021_03_12`). These folders:

- do **not** need their own Excel row — they automatically inherit the **class** of their base event;
- always end up in the **same train/val/test split as their base event**, whatever that split turns out to be (explicit override or randomly assigned) — this prevents a series and its mirrored duplicate from leaking across splits.
- if a `*_flipped` folder exists with no matching base folder (same name without the suffix), loading raises an error.

### Optional `Split` column (force an event into a specific split)

Add a column named **`Split`** to force a base event into a specific split instead of letting it participate in the normal `dataset.split.class_percentages` random split:

- Accepted values (case-insensitive): `train`, `val`, `test`, or empty/blank.
- Empty (the default for a row) means: no override, the event is split according to `dataset.split.class_percentages` as usual.
- Only base events can carry an override; `_flipped` folders always mirror their base event's split (see above) regardless of this column.
- If the overrides for a class/split consume more events than the target count computed from `class_percentages`, the remaining target is **clamped to 0** and a warning is printed (no error) — the random split just has fewer events left to place for that class/split.

Minimal example:

| (row) | Class | Split |
|------:|------:|------:|
| 0     | 3     |       |
| 1     | 1     | train |
| 2     | 1     |       |
| ...   | ...   | ...   |

Tip: you can add extra columns (e.g. `EventName`) to document the mapping, but at the moment the code uses only `Class`, `Split`, and the **row order** (matched against base event folders only).

## Notes

- `paths.dataset_dir` and `paths.excel_path` are configurable in the YAML files under `configs/`.
- In inference you can choose which splits to run (test/val/train) via `infer.run.*` (config `inference.yaml`).
- `train.use_data_parallel` wraps the model in `nn.DataParallel` only if the machine actually exposes more than one visible CUDA GPU; otherwise it's ignored (with a log message). Checkpoints are always saved as a plain (non-`DataParallel`) state_dict regardless of this setting, so a checkpoint trained on a multi-GPU machine loads the same way on a single-GPU/CPU one.
- When `dataset.split.strategy: train_val_test`, `split_assignments.csv` (columns: `event, class, split`) is written to the checkpoint dir **before training starts** (right after the split is computed), so it's preserved even if training crashes or is interrupted; for inference it's written to each active `test`/`val`/`train` results dir. Not written for `split.strategy: by_class` (there's no train/val/test distinction there).
- See "Excel file layout" above for the `_flipped` event convention and the optional `Split` column to force specific events into a given split.

## Normalization & visualization (`dataset.normalization` vs `infer.normalization_override`)

There are two separate concerns:

- `dataset.normalization` controls what the **model receives** (done inside the dataset).
- `infer.normalization_override` (mean/std/denorm_from_neg1_pos1) controls how outputs are **mapped back to [0,1] for saving/plotting**, and only needs to be set if you want to *override* the value derived automatically from `dataset.normalization`.

### Recommended combinations

| `dataset.normalization.mode` | Dataset output range | What to set in `infer.normalization_override` |
|---|---|---|
| `none` | whatever your `.npy` contains (typically `[0,1]`) | leave `denorm_from_neg1_pos1: false` (or omit) and `mean`/`std: null` |
| `neg1pos1` | `[-1,1]` | leave `denorm_from_neg1_pos1: true` (and `mean`/`std: null`) |
| `standardize` | standardized (unbounded) | set `mean`/`std` (train stats) and `denorm_from_neg1_pos1: false` |

### Notes / caveats

- For `dataset.normalization.mode: standardize`:
  - in `dataset.split.strategy: train_val_test`, mean/std are computed on the **train split** automatically and applied to train/val/test (the `dataset.normalization.mean`/`std` values in the config are ignored in this case);
  - in `dataset.split.strategy: by_class`, there is no train split, so `dataset.normalization.mean` and `dataset.normalization.std` are **required** in the config.
- `infer.normalization_override.denorm_from_neg1_pos1` is meaningful only when the dataset uses `neg1pos1`; for `standardize`, visualization should use `infer.normalization_override.mean`/`std` instead.
