from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from unet3d_gatedconv3d.config import load_config  # noqa: E402
from unet3d_gatedconv3d.pipelines.pipeline import run  # noqa: E402
from unet3d_gatedconv3d.inference.metrics import aggregate_metrics_across_seeds  # noqa: E402


def _resolve_cli_path(p: str) -> Path:
    path = Path(p).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _seed_suffixed_dir(path_str: str, seed: int) -> str:
    """'./Checkpoints_x' -> 'Checkpoints_x_seed{seed}' (cartella sorella, stesso livello)."""
    p = Path(path_str)
    return str(p.parent / f"{p.name}_seed{seed}")


def _seed_suffixed_leaf(path_str: str, seed: int) -> str:
    """'Model_Results/run/test' -> 'Model_Results/run/seed_{seed}/test' (nidificata sotto seed_N)."""
    p = Path(path_str)
    return str(p.parent / f"seed_{seed}" / p.name)


def _parse_seeds(args: argparse.Namespace) -> List[int]:
    if args.seeds:
        return [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if args.n_runs:
        return [args.base_seed + i for i in range(args.n_runs)]
    raise SystemExit("Specificare --seeds (es. '1,2,3,4,5') oppure --n-runs (con --base-seed opzionale).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Esegue training + inferenza per piu' seed di training (train.seed diverso "
            "per ogni run), tenendo dataset.split.seed INVARIATO cosi' che l'appartenenza "
            "degli eventi a train/val/test resti identica in tutte le run. Al termine "
            "aggrega le metriche di inferenza calcolando media e deviazione standard tra "
            "i seed, per ogni split/branch/metrica/timestep."
        )
    )
    parser.add_argument("--train-config", required=True, type=str, help="Path al config YAML di training (mode: train).")
    parser.add_argument("--infer-config", required=True, type=str, help="Path al config YAML di inferenza (mode: infer).")
    parser.add_argument("--seeds", type=str, default=None, help="Lista di seed separati da virgola, es. '1,2,3,4,5'.")
    parser.add_argument("--n-runs", type=int, default=None, help="Alternativa a --seeds: genera N seed consecutivi da --base-seed.")
    parser.add_argument("--base-seed", type=int, default=0, help="Primo seed usato con --n-runs (default 0).")
    parser.add_argument("--skip-train", action="store_true", help="Salta il training: richiede checkpoint gia' presenti per ogni seed, esegue solo inferenza+aggregazione.")
    parser.add_argument("--skip-infer", action="store_true", help="Esegue solo il training per ogni seed, salta inferenza e aggregazione.")
    args = parser.parse_args()

    seeds = _parse_seeds(args)
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"Seed duplicati in {seeds!r}.")
    if len(seeds) < 2:
        print("[warn] Con un solo seed la deviazione standard tra seed sara' sempre 0.0.")

    train_config_path = _resolve_cli_path(args.train_config)
    infer_config_path = _resolve_cli_path(args.infer_config)

    base_train_cfg = load_config(train_config_path, project_root=PROJECT_ROOT)
    base_infer_cfg = load_config(infer_config_path, project_root=PROJECT_ROOT)

    split_seed_train = ((base_train_cfg.get("dataset") or {}).get("split") or {}).get("seed")
    split_seed_infer = ((base_infer_cfg.get("dataset") or {}).get("split") or {}).get("seed")
    if split_seed_train != split_seed_infer:
        raise SystemExit(
            f"dataset.split.seed diverso tra train ({split_seed_train!r}) e infer "
            f"({split_seed_infer!r}): gli split non sarebbero identici tra loro. "
            "Allinea i due config prima di continuare."
        )

    strategy = str(((base_train_cfg.get("dataset") or {}).get("split") or {}).get("strategy", "train_val_test")).strip().lower()
    if strategy != "train_val_test":
        raise SystemExit(
            f"run_multi_seed.py supporta solo dataset.split.strategy='train_val_test' (trovato {strategy!r})."
        )

    orig_checkpoint_dir = base_train_cfg["train"]["checkpoint"]["dir"]
    orig_save_dirs = dict(base_infer_cfg["infer"]["save"]["dirs"])
    run_cfg = base_infer_cfg["infer"].get("run", {}) or {}
    active_splits = [s for s in ("test", "val", "train") if bool(run_cfg.get(s, False))]
    if not active_splits:
        raise SystemExit("Nessuno split attivo in infer.run (test/val/train tutti False): niente da aggregare.")

    seed_checkpoint_dirs: Dict[int, str] = {seed: _seed_suffixed_dir(orig_checkpoint_dir, seed) for seed in seeds}
    seed_save_dirs: Dict[int, Dict[str, str]] = {
        seed: {split: _seed_suffixed_leaf(orig_save_dirs[split], seed) for split in active_splits}
        for seed in seeds
    }

    # --- training ---
    if not args.skip_train:
        for seed in seeds:
            ckpt_dir = PROJECT_ROOT / seed_checkpoint_dirs[seed]
            if ckpt_dir.exists():
                raise SystemExit(
                    f"La cartella checkpoint per seed={seed} esiste gia': {ckpt_dir}\n"
                    "Rimuovila/rinominala, oppure usa --skip-train se il training per questi "
                    "seed e' gia' stato completato in precedenza."
                )
        for seed in seeds:
            print(f"\n=== [train] seed={seed} ===")
            cfg = load_config(train_config_path, project_root=PROJECT_ROOT)
            cfg["mode"] = "train"
            cfg["train"]["seed"] = seed
            cfg["train"]["checkpoint"]["dir"] = seed_checkpoint_dirs[seed]
            run(cfg, project_root=PROJECT_ROOT, config_path=train_config_path)
    else:
        print("[skip-train] Salto il training: uso i checkpoint gia' presenti su disco.")

    if args.skip_infer:
        print("[skip-infer] Salto inferenza e aggregazione.")
        return

    # --- inference ---
    for seed in seeds:
        ckpt_rel = f"{seed_checkpoint_dirs[seed]}/best_model.pth"
        resolved_ckpt = PROJECT_ROOT / ckpt_rel
        if not resolved_ckpt.exists():
            raise SystemExit(f"Checkpoint mancante per seed={seed}: {resolved_ckpt}")

        print(f"\n=== [infer] seed={seed} ===")
        cfg = load_config(infer_config_path, project_root=PROJECT_ROOT)
        cfg["mode"] = "infer"
        cfg["infer"]["checkpoint_path"] = ckpt_rel
        for split in active_splits:
            cfg["infer"]["save"]["dirs"][split] = seed_save_dirs[seed][split]
        run(cfg, project_root=PROJECT_ROOT, config_path=infer_config_path)

    # --- aggregazione media/deviazione standard tra seed ---
    print("\n=== [aggregate] media/deviazione standard tra seed ===")
    metrics_enabled = bool((base_infer_cfg.get("infer") or {}).get("metrics", {}).get("enabled", True))
    if not metrics_enabled:
        print("[warn] infer.metrics.enabled=False: nessun metrics.csv e' stato prodotto, aggregazione saltata.")
        return

    summary_root = Path(os.path.commonpath(
        [str((PROJECT_ROOT / orig_save_dirs[s]).resolve()) for s in active_splits]
    ))
    for split in active_splits:
        seed_csv_paths: Dict[int, str] = {}
        for seed in seeds:
            csv_path = PROJECT_ROOT / seed_save_dirs[seed][split] / "metrics.csv"
            if not csv_path.exists():
                print(f"[warn] metrics.csv mancante per split={split} seed={seed}: {csv_path}")
                continue
            seed_csv_paths[seed] = str(csv_path)
        if not seed_csv_paths:
            print(f"[warn] Nessun metrics.csv trovato per split={split}: aggregazione saltata.")
            continue
        out_csv = summary_root / f"seeds_summary_{split}.csv"
        aggregate_metrics_across_seeds(seed_csv_paths, str(out_csv))
        print(f"[aggregate][{split}] {len(seed_csv_paths)}/{len(seeds)} seed -> {out_csv}")


if __name__ == "__main__":
    main()
