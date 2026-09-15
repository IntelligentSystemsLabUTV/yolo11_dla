#!/usr/bin/env python3
"""Train or resume a supported YOLO model on COCO 2017."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import WORKSPACE


import argparse
import os
import re
import signal
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

ULTRALYTICS = WORKSPACE / "tools" / "ultralytics"
# Running this file from tools/ would otherwise make the checkout directory
# itself look like an empty "ultralytics" namespace package.
sys.path.insert(0, str(ULTRALYTICS))

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules import Conv  # noqa: E402


def build_model(source) -> YOLO:
    """Build a model without leaking its YAML `activation:` into Conv.default_act."""
    default_act = Conv.default_act
    try:
        return YOLO(source)
    finally:
        Conv.default_act = default_act


@dataclass(frozen=True)
class ModelSpec:
    """Files and output location associated with a trainable model."""

    source: Path
    data: Path
    project: Path
    run_name: str
    wandb_run_id: str
    pretrained: Path | None = None


MODEL_CONFIGS = ULTRALYTICS / "ultralytics" / "cfg" / "models"
RUNS = ULTRALYTICS / "runs"
DEFAULT_EPOCHS = 100
MODELS = {
    "yolo-dla-n": ModelSpec(
        # New architecture: no shape-compatible checkpoint exists, so this trains from scratch.
        source=MODEL_CONFIGS / "dla" / "yolo-dla.yaml",
        data=WORKSPACE / "tools" / "datasets" / "coco-bbox.yaml",
        project=RUNS / "detect",
        run_name="coco-yolo-dla-n",
        wandb_run_id="yolo_dla_n_coco2017",
    ),
    "yolo11n-dla": ModelSpec(
        source=MODEL_CONFIGS / "11" / "yolo11-dla.yaml",
        data=WORKSPACE / "tools" / "datasets" / "coco-bbox.yaml",
        project=RUNS / "detect",
        run_name="coco-yolo11n-dla",
        wandb_run_id="yolo11n_dla_coco2017",
        pretrained=WORKSPACE / "yolo11n.pt",
    ),
    "yolo11s-dla": ModelSpec(
        # Ultralytics strips the scale letter and loads yolo11-dla.yaml with scale="s".
        source=MODEL_CONFIGS / "11" / "yolo11s-dla.yaml",
        data=WORKSPACE / "tools" / "datasets" / "coco-bbox.yaml",
        project=RUNS / "detect",
        run_name="coco-yolo11s-dla",
        wandb_run_id="yolo11s_dla_coco2017",
        pretrained=WORKSPACE / "yolo11s.pt",
    ),
    "yolo11n-dla-seg": ModelSpec(
        source=MODEL_CONFIGS / "11" / "yolo11-dla-seg.yaml",
        data=WORKSPACE / "tools" / "datasets" / "coco-seg.yaml",
        project=RUNS / "segment",
        run_name="coco-yolo11n-dla-seg",
        wandb_run_id="yolo11n_dla_seg_coco2017",
        pretrained=WORKSPACE / "tools" / "yolo11n-seg.pt",
    ),
    "yolo11n": ModelSpec(
        source=WORKSPACE / "yolo11n.pt",
        data=WORKSPACE / "tools" / "datasets" / "coco-bbox.yaml",
        project=RUNS / "detect",
        run_name="coco-yolo11n",
        wandb_run_id="yolo11n_coco2017",
    ),
    "yolo26n": ModelSpec(
        source=WORKSPACE / "yolo26n.pt",
        data=WORKSPACE / "tools" / "datasets" / "coco-bbox.yaml",
        project=RUNS / "detect",
        run_name="coco-yolo26n",
        wandb_run_id="yolo26n_coco2017",
    ),
}

WANDB_ENTITY = "<wandb-entity>"
WANDB_PROJECT = "yolo_dla"


def model_source_exists(source: Path) -> bool:
    """Return whether a model source or its Ultralytics unified-scale YAML exists."""
    if source.exists():
        return True
    if source.suffix not in {".yaml", ".yml"}:
        return False
    unified_source = Path(re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(source)))
    return unified_source.exists()


def newest_resumable_checkpoint(spec: ModelSpec) -> Path | None:
    """Return the newest checkpoint that still contains training state."""
    weights_dir = spec.project / spec.run_name / "weights"
    candidates = [weights_dir / "last.pt", *weights_dir.glob("epoch*.pt")]
    candidates = sorted(
        (path for path in candidates if path.exists()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("epoch", -1) >= 0 and checkpoint.get("optimizer") is not None:
            return path
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh", action="store_true", help="Start a new run instead of resuming last.pt.")
    parser.add_argument(
        "--model",
        choices=tuple(MODELS),
        default="yolo11n-dla",
        help="Model architecture/weights to train (default: %(default)s).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            f"Total target epoch, including completed epochs when resuming. "
            f"Fresh-run default: {DEFAULT_EPOCHS}; resume default: checkpoint target."
        ),
    )
    parser.add_argument("--batch", type=int, default=32, help="Images per batch; use -1 for automatic sizing.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--wandb-run-id",
        help="Override the model-specific W&B run ID. Resume uses the same ID.",
    )
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be greater than zero")
    return args


def install_graceful_stop(model: YOLO) -> None:
    """Turn SIGINT/SIGTERM into a clean stop after the current batch."""
    previous: dict[int, signal.Handlers] = {}
    requested = False

    def on_train_start(trainer) -> None:

        def request_stop(signum: int, _frame) -> None:
            nonlocal requested
            if requested:
                print("\nA graceful stop is already pending; waiting for checkpoint save.", flush=True)
                return
            requested = True
            trainer.stop = True
            print(
                f"\nReceived {signal.Signals(signum).name}. "
                "Finishing the current batch, validating, and saving last.pt...",
                flush=True,
            )

        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, request_stop)

    def restore_signals(_trainer) -> None:
        for signum, handler in previous.items():
            signal.signal(signum, handler)

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("teardown", restore_signals)


def main() -> None:
    args = parse_args()
    spec = MODELS[args.model]
    if not spec.data.exists():
        raise FileNotFoundError(f"Dataset config is missing: {spec.data}")
    if not model_source_exists(spec.source):
        raise FileNotFoundError(f"Model is missing: {spec.source}")
    if spec.pretrained is not None and not spec.pretrained.exists():
        print(f"Pretrained weights will be downloaded automatically: {spec.pretrained.name}", flush=True)

    wandb_run_id = args.wandb_run_id or spec.wandb_run_id
    os.environ["WANDB_ENTITY"] = WANDB_ENTITY
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    os.environ["WANDB_RUN_ID"] = wandb_run_id
    os.environ["WANDB_NAME"] = spec.run_name

    resume_checkpoint = None if args.fresh else newest_resumable_checkpoint(spec)
    if resume_checkpoint is not None:
        target = f"epoch {args.epochs}" if args.epochs is not None else "the checkpoint target epoch"
        print(
            f"Resuming {args.model} from {resume_checkpoint} up to {target}.",
            flush=True,
        )
        model = build_model(resume_checkpoint)
        install_graceful_stop(model)
        train_args = {
            "resume": True,
            "device": args.device,
            "workers": args.workers,
            "batch": args.batch,
            "save": True,
            "save_period": 1,
        }
        if args.epochs is not None:
            train_args["epochs"] = args.epochs
        model.train(**train_args)
        return

    epochs = args.epochs if args.epochs is not None else DEFAULT_EPOCHS
    print(f"Starting fresh {args.model} run from {spec.source} for {epochs} epochs.", flush=True)
    model = build_model(spec.source)
    if spec.pretrained is not None:
        # DLA layers remain randomly initialized; every compatible layer is transferred.
        model.load(str(spec.pretrained))
    install_graceful_stop(model)
    model.train(
        data=str(spec.data),
        epochs=epochs,
        imgsz=640,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(spec.project),
        name=spec.run_name,
        exist_ok=True,
        save=True,
        save_period=1,
        pretrained=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
