from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, Iterable, TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from unet3d_gatedconv3d.config import load_config  # noqa: E402
from unet3d_gatedconv3d.pipelines.pipeline import run  # noqa: E402


class _TeeTextIO:
    def __init__(self, primary: TextIO, secondary: TextIO) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, s: str) -> int:
        n = self._primary.write(s)
        self._secondary.write(s)
        return n

    def flush(self) -> None:
        self._primary.flush()
        self._secondary.flush()

    def isatty(self) -> bool:  # pragma: no cover
        return bool(getattr(self._primary, "isatty", lambda: False)())

    @property
    def encoding(self) -> str:  # pragma: no cover
        return getattr(self._primary, "encoding", "utf-8")


def _infer_artifact_dirs(cfg: Dict[str, Any], project_root: Path) -> list[Path]:
    icfg = cfg.get("infer") or {}
    dcfg = cfg.get("dataset") or {}
    save_cfg = icfg.get("save") or {}
    save_dirs = save_cfg.get("dirs") or {}
    run_cfg = icfg.get("run") or {}

    split_strategy = str((dcfg.get("split") or {}).get("strategy", "train_val_test")).strip().lower()
    if split_strategy == "by_class":
        out_root = save_dirs.get("by_class_root", "Model_Results_1/by_class")
        return [project_root / out_root]

    dirs: list[Path] = []
    if bool(run_cfg.get("test", True)):
        dirs.append(project_root / save_dirs.get("test", "Model_Results_1/test"))
    if bool(run_cfg.get("val", False)):
        dirs.append(project_root / save_dirs.get("val", "Model_Results_1/val"))
    if bool(run_cfg.get("train", False)):
        dirs.append(project_root / save_dirs.get("train", "Model_Results_1/train"))

    if not dirs:
        dirs.append(project_root / save_dirs.get("test", "Model_Results_1/test"))
    return dirs


def _copy_run_artifacts(*, config_path: Path, cfg: Dict[str, Any], artifact_dirs: Iterable[Path]) -> None:
    try:
        import yaml  # dipendenza già usata dal progetto
    except Exception:  # pragma: no cover
        yaml = None  # type: ignore[assignment]

    for out_dir in artifact_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)

        if config_path.exists():
            shutil.copy2(str(config_path), str(out_dir / config_path.name))

        if yaml is not None:
            with open(out_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


@contextlib.contextmanager
def _tee_output_to_file(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8", buffering=1) as f:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = _TeeTextIO(old_out, f)  # type: ignore[assignment]
        sys.stderr = _TeeTextIO(old_err, f)  # type: ignore[assignment]
        try:
            yield f
        finally:
            sys.stdout = old_out  # type: ignore[assignment]
            sys.stderr = old_err  # type: ignore[assignment]


def main() -> None:
    parser = argparse.ArgumentParser(description="UNet3D + GatedConv3D runner (train / inference).")
    parser.add_argument("--config", "-c", required=True, type=str, help="Path al file YAML di configurazione.")
    parser.add_argument("--mode", choices=["train", "infer"], default=None, help="Override della modalità.")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()

    cfg = load_config(config_path, project_root=PROJECT_ROOT)
    if args.mode is not None:
        cfg["mode"] = args.mode

    mode = str(cfg.get("mode") or "train").strip().lower()
    if mode == "infer":
        artifact_dirs = _infer_artifact_dirs(cfg, project_root=PROJECT_ROOT)
        ts = _dt.datetime.now(_dt.timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")
        log_name = f"run_{ts}.log"

        # Salva config (raw + resolved) prima dell'esecuzione, così resta anche in caso di crash.
        _copy_run_artifacts(config_path=config_path, cfg=cfg, artifact_dirs=artifact_dirs)

        primary_dir = artifact_dirs[0]
        primary_log_path = primary_dir / log_name
        with _tee_output_to_file(primary_log_path):
            print(f"[run] Started at: {ts}")
            print(f"[run] CWD: {os.getcwd()}")
            print(f"[run] Config: {config_path}")
            print(f"[run] Output dir(s): {', '.join(str(p) for p in artifact_dirs)}")
            print("")
            run(cfg, project_root=PROJECT_ROOT)
            print("")
            print(f"[run] Finished. Log saved to: {primary_log_path}")

        # Copia il log anche nelle altre cartelle risultati (se presenti), senza rilanciare l'inferenza.
        for out_dir in artifact_dirs[1:]:
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(primary_log_path), str(out_dir / log_name))
        return

    run(cfg, project_root=PROJECT_ROOT)


if __name__ == "__main__":
    main()
