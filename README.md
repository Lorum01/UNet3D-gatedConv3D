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

- **Event ordering:** event subfolders are read with `sorted(os.listdir(dataset_folder))`.
  - Therefore the event order depends on the **folder name** (lexicographic order).
- **Frame ordering:** files inside each event folder are read with `sorted(os.listdir(event_folder))`.
  - Therefore the temporal order depends on the **file name**.
  - Recommendation: use zero-padded names (`000.npy`, `001.npy`, ...) to avoid orders like `1.npy, 10.npy, 2.npy`.
- **Frame format (`.npy`):**
  - each `.npy` must contain a NumPy array that is 3D: `(H, W, C)`;
  - each frame is resized to **100x100** during loading;
  - the pipeline is configured to work with **3 channels** (config `expected_input_shape: [..., 3]` and model `unet_in_channels: 3`), so in practice you should store frames with shape **`(H, W, 3)`**.
- **Value range:** values are expected to be in **[0, 1]** before preprocessing.
  - Recommended: use `data.normalization` to select preprocessing:
    - `none`: no transform
    - `neg1pos1`: map **[0,1] -> [-1,1]**
    - `standardize`: apply **(x - mean) / std** (mean/std computed on train split)
  - Backward-compatible behavior: if `data.normalization` is omitted/null, the code follows `data.scale_to_neg1_pos1` (legacy).
- **Minimum sequence length per event:** to generate at least one valid sequence, each event must have at least:
  - `T >= input_length + prediction_length`
  - and sequences are extracted with a sliding window with step `stride`.

## Excel file layout (event classes)

The Excel file (default in configs: `./EventsEtnaCLASS_Final.xlsx`) is used to assign a **class** to each event.

These rules come from `src/unet3d_gatedconv3d/data/input_utility.py` and `src/unet3d_gatedconv3d/pipelines/prep_data.py`.

- The sheet must contain a column named **`Class`** (case-sensitive).
- Excel row **0** is assigned to the **first event** loaded from the dataset, row **1** to the second, etc.
  - In practice: the row order must match the order of event folders after `sorted()`.
- The number of rows must be **>=** the number of events (subfolders) in the dataset.
- Values in `Class` must be numeric (typically integers: `1, 2, 3, 4`, consistent with `data.class_pct` in the configs).

Minimal example:

| (row) | Class |
|------:|------:|
| 0     | 3     |
| 1     | 1     |
| 2     | 1     |
| ...   | ...   |

Tip: you can add extra columns (e.g. `EventName`) to document the mapping, but at the moment the code uses only `Class` and the **row order**.

## Notes

- Dataset and Excel paths are configurable in the YAML files under `configs/`.
- In inference you can choose which splits to run (test/val/train) via `infer.run_*` (config `infer.yaml`).

## Normalization & visualization (`data.normalization` vs `infer.denorm_from_neg1_pos1`)

There are two separate concerns:

- `data.normalization` controls what the **model receives** (done inside the dataset).
- `infer.denorm_from_neg1_pos1` (and `infer.mean/std`) control how outputs are **mapped back to [0,1] for saving/plotting**.

### Recommended combinations

| `data.normalization` | Dataset output range | What to set in inference for correct visualization |
|---|---|---|
| `none` | whatever your `.npy` contains (typically `[0,1]`) | `infer.denorm_from_neg1_pos1: false` (or omit) and keep `infer.mean/std: null` |
| `neg1pos1` | `[-1,1]` | `infer.denorm_from_neg1_pos1: true` (and keep `infer.mean/std: null`) |
| `standardize` | standardized (unbounded) | set `infer.mean` and `infer.std` (train stats) and set `infer.denorm_from_neg1_pos1: false` |

### Notes / caveats

- For `data.normalization: standardize`:
  - in `split_strategy: train_val_test`, mean/std are computed on the **train split** and applied to train/val/test;
  - in `split_strategy: by_class`, there is no train split, so you must provide the statistics in the config (currently `data.mean` and `data.std` are required by the loader logic).
- `infer.denorm_from_neg1_pos1` is meaningful only when the dataset uses `neg1pos1`; for `standardize`, visualization should use `infer.mean/std` instead.
