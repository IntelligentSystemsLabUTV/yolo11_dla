# Experiment and analysis tooling

Every script used to export, build, calibrate, verify, measure, summarize and audit YOLO11-DLA-n, with stock YOLO11n as the baseline. The model code itself stays in the Ultralytics fork at `../ultralytics/` and the trained weights in `../../weights/`; datasets, engines and measurements are not duplicated here.

Read [`../../docs/EXPERIMENTS.md`](../../docs/EXPERIMENTS.md) for the protocol and the exact commands of both campaigns, and [`../../docs/BUILD.md`](../../docs/BUILD.md) for export and TensorRT builds.

## Entry point

```bash
python tools/experiments/run.py --help
python tools/experiments/run.py COMMAND [arguments]
```

`run.py` forwards arguments to the script in the matching directory without a shell. The scripts can also be invoked directly from any working directory: repository paths are resolved from the script location, while paths you pass stay relative to your own cwd.

| Command | Script | Purpose |
|---|---|---|
| `prepare` | `export/prepare.py` | Copy both checkpoints, assert model identity, export ONNX with checked shapes, hash everything |
| `reuse-existing` | `orchestration/reuse_existing.py` | Register an existing FP16 build directory: links, manifests, hashes, no rebuild |
| `int8-split` | `int8/split_coco.py` | Reproducible disjoint 500/4,500 COCO split, seed 2027 |
| `int8-calibrate` | `int8/calibrate.py` | EntropyCalibrator2 cache, pre-fusion, batch one |
| `int8-build` | `int8/build.py` | The four INT8 `trtexec` builds, including native INT8 I/O for the strict engine |
| `int8-reuse` | `int8/reuse.py` | Register an existing INT8 matrix, verify caches/ONNX/split/inspector, extract the exact INT8 scales |
| `int8-rescore-fp16` | `int8/rescore.py` | Recompute COCOeval from stored FP16 predictions on the INT8 held-out IDs, no inference |
| `build` | `orchestration/suite.py` | Optional rebuild of the engine matrix |
| `parity` | `orchestration/suite.py` | Layout oracle, raw GPU/DLA comparison, decoder and dense/sparse checks |
| `accuracy` | `orchestration/suite.py` | Matched COCO evaluations over the engine set |
| `micro` | `orchestration/suite.py` | `trtexec` engine-plus-transfer timing, optional separate profiles |
| `pipeline` | `orchestration/suite.py` | Resident multi-image application pipeline with per-frame records |
| `energy` | `orchestration/suite.py` | Paced replay with `tegrastats` telemetry and a matched idle interval |
| `summarize` | `analysis/summarize.py` | Stage aggregates: `results.csv`, `report.json`, `energy_rails.csv` |
| `evaluate` | `evaluation/evaluate_coco.py` | The COCO evaluator itself, for PyTorch checkpoints or native-format engines |
| `check-model` | `evaluation/verify_model.py` | Deterministic CPU ONNX and decoder verification |
| `benchmark` | `benchmarks/benchmark.py` | The pipeline runner, callable directly (e.g. `--dense-decode`) |
| `e2e` | `benchmarks/yolo_e2e_tester.py` | Single-sequence end-to-end tester |
| `audit` | `analysis/analyze_fp16.py` | Audit the raw FP16 `trtexec` traces and regenerate the audit CSV/JSON and figures |
| `extract-checkpoint` | `analysis/extract_checkpoint.py` | Re-extract the checkpoint's training history and metadata |
| `precision-results` | `paper/update_precision_results.py` | Manuscript tables, deployment figures and the audited result record |
| `check-paper` | `paper/check_paper.py` | PDF and TeX preflight |
| `train` | `training/train_coco_dla.py` | The trainer used for the released checkpoint |
| `recover-commands` | `history/recover_commands.py` | Rebuild the command archive from preserved logs and attachments |

## Training

The trainer builds YOLO11-DLA from the shared `yolo11-dla.yaml` configuration.
The selected scale determines the network depth and width and the stock
checkpoint used for initialization:

| Scale | `--model` | Initial weights | Run directory under `tools/ultralytics/runs/detect/` |
|---|---|---|---|
| Nano | `yolo11n-dla` | `yolo11n.pt` | `coco-yolo11n-dla` |
| Small | `yolo11s-dla` | `yolo11s.pt` | `coco-yolo11s-dla` |
| Medium | `yolo11m-dla` | `yolo11m.pt` | `coco-yolo11m-dla` |

Tensors with matching names and shapes are transferred from the stock
checkpoint; unmatched tensors keep their initialization. Missing stock
checkpoints are downloaded automatically. Each model has a separate W&B run
ID in the `yolo_dla` project.

Run commands from the repository root with PyTorch and the local Ultralytics
fork installed. Use `--data` to supply a COCO detection YAML pointing to
train2017 and val2017. Without this option, the trainer looks for
`tools/datasets/coco-bbox.yaml`, which must be provided locally.

```bash
python tools/experiments/run.py train --model yolo11n-dla --data /path/to/coco.yaml --epochs 500 --batch 32
python tools/experiments/run.py train --model yolo11s-dla --data /path/to/coco.yaml --epochs 500 --batch 32
python tools/experiments/run.py train --model yolo11m-dla --data /path/to/coco.yaml --epochs 500 --batch 32
```

Run these commands sequentially when sharing one GPU. Training uses 640-pixel
images, device `0` and eight data-loader workers by default. Select another
GPU with `--device` and adjust workers with `--workers`. Small and medium
require more memory; `--batch -1` enables automatic batch sizing on one GPU.
A fresh run defaults to 100 epochs. The examples specify 500 to match the
nano's recorded epoch budget.

The trainer automatically resumes the latest checkpoint containing optimizer
state in the selected run directory. On resume, omit `--data`: the dataset
comes from the checkpoint. `--epochs` specifies the total target, including
completed epochs; omitting it preserves the checkpoint's target.

```bash
python tools/experiments/run.py train --model yolo11s-dla --epochs 500 --batch 32
```

Use `--fresh` to restart in the same run directory, potentially replacing its
artifacts. `--wandb-run-id` overrides the model's default tracking ID; it does
not change the local output directory.

The shell wrapper runs training in the background and provides status and
stop commands. Select the same model for each operation:

```bash
bash tools/experiments/training/coco_dla_train.sh start --model yolo11s-dla --data /path/to/coco.yaml --epochs 500
bash tools/experiments/training/coco_dla_train.sh status --model yolo11s-dla
bash tools/experiments/training/coco_dla_train.sh stop --model yolo11s-dla
bash tools/experiments/training/coco_dla_train.sh resume --model yolo11s-dla --epochs 500
```

Logs are written to `train.log` in the model's run directory. A stop request
allows the current batch to finish, then validates and saves a checkpoint.

The trainer also supports other model families and segmentation; use
`python tools/experiments/run.py train --help` to list the available choices.
The reported deployment measurements cover stock YOLO11-n and YOLO11-DLA-n.
Small and medium require their own evaluation, TensorRT build and engine
inspection on the target Jetson to establish accuracy, performance and strict
DLA placement.

## Layout

```text
tools/experiments/
├── run.py                 single entry point
├── _paths.py, common.py   repository paths, hashing, provenance capture, shared protocol helpers
├── export/                checkpoint copy and ONNX export
├── datasets/              COCO val2017 download and presence check
├── int8/                  split, calibration, build, registration, native scales, FP16 rescoring
├── evaluation/            COCO AP, ONNX verification, layout/decoder validation
├── benchmarks/            runtime, sequences, pacing, telemetry
├── orchestration/         build/measure matrix and randomized repetition blocks
├── analysis/              log audit, checkpoint history, CSV/figure/summary generation
├── paper/                 manuscript tables, figures and preflight
├── training/              trainer and background start/resume/stop control
│   └── monitoring/        Weights & Biases sync and results server
├── configs/               hardware notes template
├── tests/                 software regression suite
└── history/               command archive, provenance, recovered terminal sources
```

## Conventions

The deployment measurement stages write to a **new** output directory and do not resume interrupted runs. `--dry-run` prints the planned command list without creating output or importing CUDA/TensorRT. Commands are built as argument lists, never through shell interpolation. Artifact hashes are verified before each stage, and a stage stops on an unexpected failure rather than continuing with a partial matrix. No script changes clocks, power mode or cooling.

```bash
python -m pytest tools/experiments/tests -q
```

The test suite is host-side only: it covers NMS edge cases, packing and padding for every layout, power integration and coverage, pacing and drops, inspector acceptance, reuse without export, rejection of modified engines and mismatched caches, and reproducible splits. It does **not** certify TensorRT or DLA execution.

`history/` is a provenance archive, not a list of commands to execute blindly. It preserves original paths faithfully — including ones that no longer resolve — and the current runtime includes fixes made after some of those measurements were taken.
