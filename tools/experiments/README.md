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
| `collect-results` | `analysis/collect_results.py` | Rebuild the audited result record from the campaign summaries |
| `train` | `training/train_coco_dla.py` | The trainer used for the released checkpoint |
| `recover-commands` | `history/recover_commands.py` | Rebuild the command archive from preserved logs and attachments |

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
├── training/              trainer and historical start/resume/stop control
│   └── monitoring/        Weights & Biases sync and results server
├── configs/               hardware notes template
├── tests/                 software regression suite
└── history/               command archive, provenance, recovered terminal sources
```

## Conventions

Every stage writes to a **new** output directory; there is no resume. `--dry-run` prints the planned command list without creating output or importing CUDA/TensorRT. Commands are built as argument lists, never through shell interpolation. Artifact hashes are verified before each stage, and a stage stops on an unexpected failure rather than continuing with a partial matrix. No script changes clocks, power mode or cooling.

```bash
python -m pytest tools/experiments/tests -q
```

The test suite is host-side only: it covers NMS edge cases, packing and padding for every layout, power integration and coverage, pacing and drops, inspector acceptance, reuse without export, rejection of modified engines and mismatched caches, and reproducible splits. It does **not** certify TensorRT or DLA execution.

## Two cautions

`training/` also retains options for other scales, tasks and model families, because the trainer is shared. **They are not part of the measured matrix**, which is exactly stock YOLO11-n and YOLO11-DLA-n. The trainer keeps its original run paths, resume behaviour and Weights & Biases settings: do not start a new training run with it before separating the output directory and the run ID.

`history/` is a provenance archive, not a list of commands to execute blindly. It preserves original paths faithfully — including ones that no longer resolve — and the current runtime includes fixes made after some of those measurements were taken.
