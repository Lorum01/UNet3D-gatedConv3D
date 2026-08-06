from __future__ import annotations

import argparse
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_MULTI_SEED = PROJECT_ROOT / "scripts" / "run_multi_seed.py"


def _load_jobs(batch_config_path: Path) -> List[Dict[str, Any]]:
    with open(batch_config_path, "r") as f:
        spec = yaml.safe_load(f)
    jobs = spec.get("jobs") if isinstance(spec, dict) else spec
    if not jobs:
        raise SystemExit(f"Nessun job trovato in {batch_config_path}")
    return jobs


def _job_name(job: Dict[str, Any], index: int) -> str:
    return str(job.get("name") or f"{Path(job['train_config']).stem}_{index}")


def _job_cmd(job: Dict[str, Any], gpu: int) -> List[str]:
    cmd = [
        sys.executable, str(RUN_MULTI_SEED),
        "--train-config", str(job["train_config"]),
        "--infer-config", str(job["infer_config"]),
        "--device", f"cuda:{gpu}",
    ]
    if job.get("seeds") is not None:
        cmd += ["--seeds", ",".join(str(s) for s in job["seeds"])]
    else:
        if job.get("n_runs") is None:
            raise SystemExit(f"Job {job!r} deve specificare 'seeds' oppure 'n_runs'.")
        cmd += ["--n-runs", str(job["n_runs"])]
        if job.get("base_seed") is not None:
            cmd += ["--base-seed", str(job["base_seed"])]
    if job.get("skip_train"):
        cmd.append("--skip-train")
    if job.get("skip_infer"):
        cmd.append("--skip-infer")
    return cmd


def _run_queue(gpu: int, jobs: List[Dict[str, Any]], log_dir: Path) -> None:
    for i, job in enumerate(jobs):
        name = _job_name(job, i)
        log_path = log_dir / f"gpu{gpu}_{i:02d}_{name}.log"
        cmd = _job_cmd(job, gpu)
        print(f"[gpu{gpu}] ({i + 1}/{len(jobs)}) avvio '{name}' -> log: {log_path}", flush=True)
        print(f"[gpu{gpu}]   {' '.join(cmd)}", flush=True)
        with open(log_path, "w") as log_file:
            result = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=log_file, stderr=subprocess.STDOUT)
        if result.returncode == 0:
            print(f"[gpu{gpu}] ({i + 1}/{len(jobs)}) '{name}' OK", flush=True)
        else:
            print(
                f"[gpu{gpu}] ({i + 1}/{len(jobs)}) '{name}' FALLITO (exit {result.returncode}) "
                f"- interrompo la coda su gpu{gpu}, vedi {log_path}",
                flush=True,
            )
            return


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Esegue piu' run multiseed (scripts/run_multi_seed.py) distribuendole su piu' GPU: "
            "i job assegnati alla stessa GPU girano in sequenza (uno alla volta), ma le GPU "
            "lavorano in parallelo tra loro. Ogni job scrive il proprio log in un file separato."
        )
    )
    parser.add_argument("--batch-config", required=True, type=str, help="YAML con la lista dei job (vedi configs/multi_seed_batch_example.yaml).")
    parser.add_argument("--gpus", type=str, default="0,1", help="Elenco indici GPU disponibili, separati da virgola (default '0,1').")
    parser.add_argument("--log-dir", type=str, default=None, help="Cartella per i log per-job (default logs/multiseed_batch/<timestamp>).")
    args = parser.parse_args()

    available_gpus = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
    if not available_gpus:
        raise SystemExit("--gpus non puo' essere vuoto.")

    jobs = _load_jobs(Path(args.batch_config))

    queues: Dict[int, List[Dict[str, Any]]] = {gpu: [] for gpu in available_gpus}
    for i, job in enumerate(jobs):
        gpu = job.get("gpu")
        if gpu is None:
            gpu = available_gpus[i % len(available_gpus)]
        elif gpu not in available_gpus:
            raise SystemExit(f"Job '{_job_name(job, i)}' richiede gpu={gpu}, non presente in --gpus={available_gpus}.")
        queues[gpu].append(job)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path(args.log_dir) if args.log_dir else PROJECT_ROOT / "logs" / "multiseed_batch" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Log per-job in: {log_dir}")

    for gpu, queue in queues.items():
        names = [_job_name(j, i) for i, j in enumerate(queue)]
        print(f"[gpu{gpu}] {len(queue)} job in coda: {names}")

    threads = [
        threading.Thread(target=_run_queue, args=(gpu, queue, log_dir), daemon=False)
        for gpu, queue in queues.items() if queue
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("\nTutte le code sono terminate.")


if __name__ == "__main__":
    main()
