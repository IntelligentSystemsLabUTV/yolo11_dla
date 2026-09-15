#!/usr/bin/env python3
"""Synchronize completed Ultralytics epochs to W&B once, then exit."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from _paths import WORKSPACE


import argparse
import csv
from pathlib import Path

import wandb


RUNS_DIR = WORKSPACE / "tools" / "ultralytics" / "runs" / "detect"

DEFAULT_ENTITY = "alexandru-cretu-university-of-rome-tor-vergata"
DEFAULT_PROJECT = "yolo_dla"
# Keep the repaired history separate from the old run, whose steps used mixed
# zero-based and one-based epoch conventions.
DEFAULT_RUN_ID = "yolo11n_dla_coco2017_clean"
DEFAULT_RUN_NAME = "coco-yolo11n-dla-clean"
SYNC_SCHEMA = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Ultralytics run directory; auto-detected when omitted.")
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def process_is_training(pid: int) -> bool:
    proc = Path("/proc") / str(pid)
    if not proc.exists():
        return False
    try:
        command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return False
    return "train_coco_dla.py" in command


def training_is_active(run_dir: Path) -> bool:
    try:
        pid = int((run_dir / "train.pid").read_text().strip())
    except (FileNotFoundError, OSError, ValueError):
        return False
    return process_is_training(pid)


def find_run_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        run_dir = explicit.expanduser().resolve()
        if not (run_dir / "results.csv").is_file():
            raise FileNotFoundError(f"results.csv not found in {run_dir}")
        return run_dir

    result_files = list(RUNS_DIR.glob("*/results.csv"))
    active = [path.parent for path in result_files if training_is_active(path.parent)]
    if len(active) == 1:
        return active[0]
    if len(active) > 1:
        raise RuntimeError("Multiple active runs found; pass --run-dir explicitly.")
    if not result_files:
        raise FileNotFoundError(f"No results.csv found under {RUNS_DIR}")
    return max(result_files, key=lambda path: path.stat().st_mtime).parent


def read_results(results_file: Path) -> list[tuple[int, dict[str, float]]]:
    """Return one-based Ultralytics CSV epochs and their metrics."""
    epochs: dict[int, dict[str, float]] = {}
    with results_file.open(newline="") as stream:
        for raw in csv.DictReader(stream):
            row = {(key or "").strip(): value for key, value in raw.items()}
            epoch_value = row.pop("epoch", "")
            if not epoch_value:
                continue
            try:
                epoch = int(float(epoch_value))
                metrics = {
                    key: float(value)
                    for key, value in row.items()
                    if key and value not in {"", None}
                }
            except (TypeError, ValueError):
                # Ignore a partially written final row; it will be read next time.
                continue
            epochs[epoch] = metrics
    return sorted(epochs.items())


def remote_epochs(entity: str, project: str, run_id: str) -> set[int]:
    """Read epochs written by this script, or report an incompatible target."""
    try:
        remote = wandb.Api().run(f"{entity}/{project}/{run_id}")
    except wandb.errors.CommError:
        return set()

    schema = remote.config.get("csv_sync_schema")
    if schema != SYNC_SCHEMA:
        raise RuntimeError(
            f"W&B run {run_id!r} was not created by this clean synchronizer. "
            "Use a new --run-id; writing into the old mixed-step run is unsafe."
        )

    return {
        int(row["epoch"])
        for row in remote.scan_history(keys=["epoch"], page_size=500)
        if row.get("epoch") is not None
    }


def main() -> None:
    args = parse_args()
    run_dir = find_run_dir(args.run_dir)
    results_file = run_dir / "results.csv"
    rows = read_results(results_file)
    if not rows:
        raise RuntimeError(f"No complete epoch rows found in {results_file}")
    local_epochs = {epoch for epoch, _metrics in rows}
    missing_local = sorted(set(range(1, rows[-1][0] + 1)) - local_epochs)
    if missing_local:
        raise RuntimeError(
            "results.csv itself has missing epochs: "
            + ", ".join(map(str, missing_local))
            + ". Repair the CSV before synchronizing."
        )

    print(f"Results: {results_file}")
    print(f"Local epochs: 1..{rows[-1][0]} ({len(rows)} rows)")
    print(f"W&B: {args.entity}/{args.project}/{args.run_id}")

    if args.dry_run:
        return

    existing = remote_epochs(args.entity, args.project, args.run_id)
    missing = [(epoch, metrics) for epoch, metrics in rows if epoch not in existing]
    if not missing:
        print("Already synchronized; nothing to do.")
        return

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        id=args.run_id,
        name=args.name,
        resume="allow",
        force=True,
        config={
            "csv_sync_schema": SYNC_SCHEMA,
            "source_results": str(results_file),
            "epoch_numbering": "one-based",
        },
        tags=("yolo11n", "dla", "coco2017", "bbox", "csv-sync"),
    )
    try:
        run.define_metric("epoch")
        run.define_metric("*", step_metric="epoch")
        for epoch, metrics in missing:
            run.log({"epoch": epoch, **metrics}, step=epoch, commit=True)
            print(f"Synced epoch {epoch}")
        run.summary["last_completed_epoch"] = rows[-1][0]
        run.summary["source_rows"] = len(rows)
    finally:
        run.finish(exit_code=0)

    print(f"Done: {run.url}")


if __name__ == "__main__":
    main()
