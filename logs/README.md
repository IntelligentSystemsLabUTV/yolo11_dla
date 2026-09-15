# `logs/` — models and measurement artifacts

**This directory is not tracked.** Following the DUA convention, `logs/.gitignore` ignores everything here except the control files and this page. It is the working directory where the models, the engines and every measurement campaign live, on the machine that runs them.

The repository ships the code, the protocol and the **derived** record the manuscript is built from — `tools/paper/analysis/` holds the audited CSV/JSON with the SHA-256 of each raw source. The raw artifacts described below are produced by the commands in [`../docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md) and are not carried in git.

## What the scripts expect to find here

| Path | What it is | How to obtain it |
| --- | --- | --- |
| `yolo11n-dla-500ep.pt` | the trained YOLO11-DLA-n checkpoint, SHA-256 `417967ac…` | **cannot be regenerated from this repository**: 500 epochs on COCO train2017. Obtain it separately, or retrain with `run.py train` |
| `yolo11n.pt` | the stock YOLO11n checkpoint | upstream Ultralytics |
| `yolo11n-dla-500ep.onnx`, `yolo11n.onnx` | the two exports every engine is built from | `run.py prepare`, see [`../docs/BUILD.md`](../docs/BUILD.md) |
| `datasets/coco/` | val2017 images and `instances_val2017.json` | `bash tools/experiments/datasets/download_coco_val2017.sh` |

## What each stage writes here

| Path | Produced by | Contents |
| --- | --- | --- |
| `fp16_matrix/` | `trtexec` | FP16 engines, verbose build logs, inspector `*.layers.json`, benchmark traces `*.times.json`, separate layer profiles, and the **failed** stock strict build that documents the rejection at the attention `MatMul` |
| `int8_calibration/{stock,adapted}/` | `run.py int8-calibrate` | entropy calibration caches with manifests recording the ONNX hash, split hash, preprocessing hash and per-image list |
| `int8_matrix/` | `run.py int8-build` | INT8 engines, build logs, inspector output |
| `int8_ready/` | `run.py int8-reuse` | the registered INT8 engine set: uniform names, manifest with the full build commands and inspector verdicts, the exact float32 quantization scales bound to the engine hash, the 500/4,500 split and the held-out annotation file |
| `icra2027_final/existing/` | `run.py reuse-existing` | the FP16 build and microbenchmark evidence registered by relative link and hash, with its build and micro summaries |
| `icra2027_final/run_01/` | `parity`, `accuracy`, `pipeline`, `energy` | the FP16 campaign, plus the `summary-*` aggregates the paper cites |
| `icra2027_int8/run_01/` | the same stages, plus `int8-rescore-fp16` | the INT8 campaign and the paired FP16 rescoring on the held-out cohort |

Each run directory carries a `manifest.json` with the command, the artifact hashes, the platform provenance queries and the protocol settings; timing stages also write per-frame `frames.jsonl`, energy stages `telemetry.jsonl`, and `summarize` writes `results.csv`, `report.json` and `energy_rails.csv`.

## Checking a campaign

The summaries are re-derivable from the raw records, which is the cheapest integrity check on a set of artifacts:

```bash
python tools/experiments/run.py summarize --stage logs/icra2027_int8/run_01/pipeline --output /tmp/check
diff <(sort /tmp/check/results.csv) <(sort logs/icra2027_int8/run_01/pipeline-summary/results.csv)
```

It reproduces byte-identically for the pipeline, energy and microbenchmark stages. The accuracy stages are the exception: `summarize` reads their bulk `frames.jsonl`, but each job's own `manifest.json` is self-sufficient — it carries the full COCOeval metrics, the engine SHA-256, the annotation SHA-256 and the list of evaluated image IDs — so every accuracy row stays checkable without it.

## Two traps in this directory

`yolo11n-dla-gpu-fallback-fp16.*` is **stock YOLO11n**, not the adapted model: the `dla` in the prefix refers to the DLA fallback build, and the build command loads `yolo11n.onnx`. The only end-to-end JSON records in `fp16_matrix/` belong to the independent YOLO-DLA family, which is outside the paper's scope and cannot fill YOLO11-DLA rows. Both are documented in [`../docs/PROVENANCE.md`](../docs/PROVENANCE.md).

Absolute paths recorded inside manifests and logs begin with `/home/neo/workspace`, the container mount point on the Jetson, or with `logs/paper/…` in the oldest records. They are provenance strings, not paths that resolve on a fresh checkout.
