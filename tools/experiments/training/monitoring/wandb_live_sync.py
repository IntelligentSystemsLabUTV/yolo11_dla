#!/usr/bin/env python3
"""Continuously mirror an active Ultralytics results.csv file to a W&B run."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from _paths import WORKSPACE


import argparse
import csv
import json
import os
import signal
import socket
import time
from pathlib import Path

import wandb


RUNS_DIR = WORKSPACE / "tools" / "ultralytics" / "runs" / "detect"

DEFAULT_ENTITY = "alexandru-cretu-university-of-rome-tor-vergata"
DEFAULT_PROJECT = "yolo_dla"
DEFAULT_RUN_ID = "yolo11n_dla_coco2017"

STOP_REQUESTED = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Ultralytics run directory; auto-detected when omitted.")
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--poll", type=float, default=15.0, help="Seconds between results.csv checks.")
    parser.add_argument("--once", action="store_true", help="Backfill once and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print epochs without contacting W&B.")
    return parser.parse_args()


def process_is_training(pid: int) -> bool:
    """Return whether pid is a live train_coco_dla.py process."""
    proc = Path("/proc") / str(pid)
    if not proc.exists():
        return False
    try:
        command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return False
    return "train_coco_dla.py" in command


def training_is_active(run_dir: Path) -> bool:
    """Return whether the run directory's recorded trainer is active."""
    pid_file = run_dir / "train.pid"
    try:
        return process_is_training(int(pid_file.read_text().strip()))
    except (FileNotFoundError, OSError, ValueError):
        return False


def find_run_dir(explicit: Path | None) -> Path:
    """Find the active Ultralytics run, falling back to the newest results.csv."""
    if explicit is not None:
        run_dir = explicit.expanduser().resolve()
        if not (run_dir / "results.csv").exists():
            raise FileNotFoundError(f"results.csv not found in {run_dir}")
        return run_dir

    result_files = list(RUNS_DIR.glob("*/results.csv"))
    active = [path.parent for path in result_files if training_is_active(path.parent)]
    if len(active) == 1:
        return active[0]
    if len(active) > 1:
        names = ", ".join(str(path) for path in active)
        raise RuntimeError(f"Multiple active runs found ({names}); pass --run-dir.")
    if not result_files:
        raise FileNotFoundError(f"No results.csv files found under {RUNS_DIR}")
    return max(result_files, key=lambda path: path.stat().st_mtime).parent


def read_results(results_file: Path) -> list[tuple[int, dict[str, float]]]:
    """Read all complete metric rows from an Ultralytics results.csv."""
    rows: list[tuple[int, dict[str, float]]] = []
    with results_file.open(newline="") as stream:
        for raw in csv.DictReader(stream):
            row = {(key or "").strip(): value for key, value in raw.items()}
            epoch_value = row.pop("epoch", "")
            if not epoch_value:
                continue
            try:
                epoch = int(float(epoch_value))
                metrics = {key: float(value) for key, value in row.items() if key and value not in {"", None}}
            except (TypeError, ValueError):
                # The trainer may have been appending this row while it was read. Retry it on the next poll.
                continue
            rows.append((epoch, metrics))
    return sorted(rows, key=lambda item: item[0])


def state_file_for(run_dir: Path) -> Path:
    return run_dir / ".wandb-live-sync.json"


def read_local_step(run_dir: Path) -> int:
    try:
        return int(json.loads(state_file_for(run_dir).read_text())["last_synced_epoch"])
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return -1


def write_local_step(run_dir: Path, step: int) -> None:
    state_file = state_file_for(run_dir)
    temporary = state_file.with_suffix(".tmp")
    temporary.write_text(json.dumps({"last_synced_epoch": step}, indent=2) + "\n")
    temporary.replace(state_file)


def get_remote_step(entity: str, project: str, run_id: str) -> int:
    """Return the greatest W&B history step already stored remotely."""
    remote = wandb.Api().run(f"{entity}/{project}/{run_id}")
    # `lastHistoryStep` works even when a run has no `_step` column in its
    # exported schema. Querying scan_history(keys=["_step"]) raises on such runs.
    last_step = remote.lastHistoryStep
    return int(last_step) if last_step is not None else -1


def request_stop(_signum: int, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def main() -> None:
    args = parse_args()
    if args.poll <= 0:
        raise ValueError("--poll must be greater than zero")

    run_dir = find_run_dir(args.run_dir)
    results_file = run_dir / "results.csv"
    print(f"Run directory: {run_dir}", flush=True)
    print(f"Results file:  {results_file}", flush=True)
    print(f"W&B target:   {args.entity}/{args.project}/{args.run_id}", flush=True)

    if args.dry_run:
        for epoch, metrics in read_results(results_file):
            print(f"Would sync epoch {epoch}: {len(metrics)} metrics")
        return

    local_step = read_local_step(run_dir)
    remote_step = get_remote_step(args.entity, args.project, args.run_id)
    last_step = max(local_step, remote_step)
    print(f"Last local/remote epoch: {local_step}/{remote_step}", flush=True)

    settings = wandb.Settings(
        mode="shared",
        x_label=f"results-csv-{socket.gethostname()}-{os.getpid()}",
        x_primary=False,
        x_update_finish_state=False,
    )
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        id=args.run_id,
        settings=settings,
        force=True,
    )

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, request_stop)

    try:
        while True:
            for epoch, metrics in read_results(results_file):
                if epoch <= last_step:
                    continue
                run.log(metrics, step=epoch, commit=True)
                run.summary["csv_sync_last_epoch"] = epoch
                last_step = epoch
                write_local_step(run_dir, epoch)
                print(f"Synced epoch {epoch} ({len(metrics)} metrics)", flush=True)

            active = training_is_active(run_dir)
            if args.once or STOP_REQUESTED or not active:
                reason = "--once" if args.once else "stop requested" if STOP_REQUESTED else "training stopped"
                print(f"Exiting: {reason}. Last synced epoch: {last_step}", flush=True)
                break
            time.sleep(args.poll)
    finally:
        # x_update_finish_state=False prevents this worker from marking the training run finished or crashed.
        run.finish(exit_code=0)


if __name__ == "__main__":
    main()
