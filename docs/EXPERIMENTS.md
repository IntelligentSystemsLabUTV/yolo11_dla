# Measurement protocol

Two campaigns produced the paper: an FP16 campaign in `logs/icra2027_final/` and an INT8 campaign in `logs/icra2027_int8/`. Both evaluate the same four configurations — stock YOLO11n on GPU, stock YOLO11n on DLA with GPU fallback, YOLO11-DLA-n on GPU, YOLO11-DLA-n on strict DLA — at batch one and 640×640, from already-built engines. No stage in either campaign rebuilt an engine.

Everything runs through one entry point, which forwards arguments to the script in the matching directory without a shell:

```bash
python tools/experiments/run.py --help
```

Common rules for every stage: the output directory must be **new** (there is no resume), `--dry-run` prints the command list without creating output or importing CUDA/TensorRT, engine hashes are verified before any measurement, and the stage stops on an unexpected failure rather than continuing with a partial matrix. No script changes clocks or power mode. Do not run two stages concurrently.

## Environment for the commands below

```bash
export PAPER_TOOLS="$PWD/tools/experiments"
export COCO_VAL="$PWD/logs/datasets/coco/images/val2017"
export COCO_ANN="$PWD/logs/datasets/coco/annotations/instances_val2017.json"
export PYTHONPATH="$PWD/tools/ultralytics${PYTHONPATH:+:$PYTHONPATH}"
export TRTEXEC=/usr/src/tensorrt/bin/trtexec
export TEGRASTATS=/tmp/tegrastats-paper

export FP16_ENGINES="$PWD/logs/icra2027_final/existing/engines"
export FP16_RUN="$PWD/logs/icra2027_final/run_01"
export INT8_ENGINES="$PWD/logs/int8_ready"
export INT8_RUN="$PWD/logs/icra2027_int8/run_01"
```

`logs/int8_ready/evaluation.json` is the 4,500-image held-out COCO annotation file the INT8 accuracy stage reads. It is written by `run.py int8-reuse`, and can also be rebuilt from the image IDs that the same registration stores in `logs/int8_ready/manifest.json` under `evaluation_protocol.image_ids`.

## 1. Parity: layouts and decoder, before anything else

```bash
python "$PAPER_TOOLS/run.py" parity --engines "$FP16_ENGINES" \
  --images "$COCO_VAL" --annotations "$COCO_ANN" --output "$FP16_RUN/parity" --limit 50

python "$PAPER_TOOLS/run.py" parity --engines "$INT8_ENGINES" \
  --images "$COCO_VAL" --annotations "$INT8_ENGINES/evaluation.json" \
  --output "$INT8_RUN/parity" --limit 50 --raw-diagnostic-only
```

This stage verifies pack and unpack separately against a scalar-offset oracle, records formats, strides, capacities and physical byte counts, then compares the two adapted engines on 50 identically preprocessed real images: raw absolute and relative error per scale, decoded coordinate and score error, post-NMS lists, and dense-versus-sparse agreement at both confidence 0.001 multi-label and confidence 0.25 single-label.

The raw GPU-versus-DLA gate is **exploratory**, at `atol=0.1, rtol=0.05`. It is not a scientific equivalence threshold and not an AP tolerance. It **fails**, and the failure is reported rather than tuned away: 50/49/49 of 50 images fail for FP16 P3/P4/P5, and all 50 fail at every level for INT8. `--raw-diagnostic-only` records the deviations for INT8 without treating the FP16-era tolerances as an INT8 acceptance criterion. Wrong layouts, non-finite values and dense/sparse divergence remain blocking errors in both cases. Decoder parity uses `atol=1e-3, rtol=1e-5` and passes: 8 layout checks and 200 dense/sparse comparisons per precision campaign.

If a different tolerance is ever used, declare it before collection with `--raw-atol/--raw-rtol`.

## 2. Accuracy

Common protocol for every evaluation: the same image IDs in the same order with verified JPEG hashes, centred square 640 letterbox with padding 114 and RGB normalization by 255, confidence 0.001, class-aware NMS at IoU 0.7, multi-label candidates, at most 300 boxes per image, single-threaded CPU post-processing, and COCOeval's standard detection limits. These are validation thresholds and are deliberately different from the application thresholds used for timing.

```bash
# FP16, six evaluations on the 5,000 val2017 images: four dense plus the two adapted engines sparse
python "$PAPER_TOOLS/run.py" accuracy --engines "$FP16_ENGINES" \
  --images "$COCO_VAL" --annotations "$COCO_ANN" --output "$FP16_RUN/accuracy"

# INT8, on the 4,500 held-out IDs; the stock-GPU evaluation is linked, not re-run
python "$PAPER_TOOLS/run.py" accuracy --engines "$INT8_ENGINES" \
  --images "$COCO_VAL" --annotations "$INT8_ENGINES/evaluation.json" \
  --output "$INT8_RUN/accuracy" \
  --reuse-evaluation "stock-gpu=$PWD/logs/int8-evaluation.8ZbovD/accuracy-stock-gpu"

# FP16 AP on exactly the same 4,500 held-out IDs, by rescoring stored predictions -- no inference
python "$PAPER_TOOLS/run.py" int8-rescore-fp16 \
  --source "$FP16_RUN/accuracy" --engines "$INT8_ENGINES" --output "$INT8_RUN/fp16-heldout"

python "$PAPER_TOOLS/run.py" summarize --stage "$INT8_RUN/accuracy"      --output "$INT8_RUN/accuracy-summary"
python "$PAPER_TOOLS/run.py" summarize --stage "$INT8_RUN/fp16-heldout"  --output "$INT8_RUN/fp16-heldout-summary"
python "$PAPER_TOOLS/run.py" summarize --stage "$FP16_RUN/accuracy"      --output "$FP16_RUN/summary-accuracy"
```

`--reuse-evaluation` verifies the engine hash, the image IDs and hashes, and the thresholds before linking an existing complete evaluation; a smoke run cannot be passed off as a complete one. `int8-rescore-fp16` recomputes COCOeval on CPU from the stored FP16 predictions, selecting exactly the 4,500 INT8 IDs: this is what makes the FP16 and INT8 AP columns of the paper a paired comparison. The original full-5,000-image FP16 values (39.314 / 39.245 / 37.995 / 37.991) must not be compared against held-out INT8 AP.

Optional PyTorch checkpoint evaluations can be added to the FP16 stage with `--stock-checkpoint` and `--adapted-checkpoint`, which turns six evaluations into eight without any export. They were not used to attribute an export effect, because the paired stock checkpoint identity was never recovered.

## 3. Engine-plus-transfer latency

The `trtexec` microbenchmark measures `T_H2D + T_device + T_D2H`, excluding preprocessing, host decoding and NMS. Protocol: 2 s warm-up, 30 s timing window, one inference stream, spin waiting, generated input data, transfers enabled, no profiling in the base run.

```bash
python "$PAPER_TOOLS/run.py" micro --engines "$INT8_ENGINES" --trtexec "$TRTEXEC" \
  --output "$INT8_RUN/micro" --duration 30 --repeats 1
python "$PAPER_TOOLS/run.py" summarize --stage "$INT8_RUN/micro" --output "$INT8_RUN/micro-summary"
```

The FP16 microbenchmarks were **not re-run**: the four historical traces, one launch per configuration and 20,167 timing samples in total, are registered from `logs/fp16_matrix/` into `logs/icra2027_final/existing/` and summarized there. Their protocol was verified from the original logs (`--warmUp=2000 --duration=30 --useSpinWait --exportTimes=...`, with `--infStreams=1` implicit and reported in the logs' Inference Options section).

Two reading rules. `trtexec`'s "GPU Compute Time" field measures device execution even for a DLA engine and does not imply GPU computation. Query throughput uses host wall time and can differ from the reciprocal of median latency, because transfers and execution overlap. P95 and P99 describe frame variation inside one test; they are not confidence intervals across independent trials, and one launch per configuration does not estimate between-launch variability.

`--profile` requests separate profiling runs. It was used only for the historical FP16 profiles; the new suite's profiling default is a 10 s timed profile, which differs from the historical `--warmUp=1000 --duration=0 --iterations=100 --separateProfileRun` runs, so the two must not be pooled silently.

## 4. Application pipeline

```bash
python "$PAPER_TOOLS/run.py" pipeline --engines "$INT8_ENGINES" \
  --images "$COCO_VAL" --annotations "$COCO_ANN" \
  --output "$INT8_RUN/pipeline" --limit 100 --duration 60 --repeats 1
python "$PAPER_TOOLS/run.py" summarize --stage "$INT8_RUN/pipeline" --output "$INT8_RUN/pipeline-summary"
```

The first 100 validation images in sorted ID order — the same list for both precisions, note `COCO_ANN` and not the held-out file — are preloaded into RAM, then cycled for 60 s after 30 warm-up frames and 30 s of stabilization, with the engine resident. Post-processing is single-threaded CPU at confidence 0.25, single-label candidates and NMS IoU 0.45; the adapted engines use the sparse decoder.

The measured boundary includes preprocessing, packing and quantization, H2D, engine execution, D2H, unpacking and dequantization, decoding and NMS. It excludes disk access, visualization, camera acquisition and ROS transport. It is a service loop over images already in memory, not a camera-to-result or rosbag replay measurement.

Each frame record holds preprocess, pack+H2D, engine, D2H+unpack, postprocess, total, candidate count and detections. The total is a real interval, not a sum of stage medians, so stage medians do not add up to the median total. The rate is completed frames divided by real duration, not the reciprocal of the median. Because the test has a fixed duration, faster configurations complete more cycles: 28–29 passes per image on GPU against 10–12 on the DLA paths.

The `engine_ms` field here measures the host-side runtime call and synchronization, which is not the pure kernel time and not `trtexec`'s engine-plus-transfer boundary. The two latency tables answer different questions and must not be merged.

## 5. Module energy at a matched delivered rate

```bash
python "$PAPER_TOOLS/run.py" energy --engines "$INT8_ENGINES" \
  --images "$COCO_VAL" --annotations "$COCO_ANN" \
  --output "$INT8_RUN/energy-10hz" --limit 100 --fps 10 --duration 60 --repeats 1 \
  --tegrastats "$TEGRASTATS" --power-profile agx-orin
python "$PAPER_TOOLS/run.py" summarize --stage "$INT8_RUN/energy-10hz" --output "$INT8_RUN/energy-summary"
```

Four active configurations plus one idle interval, 60 s each, in deterministic randomized order, with the same 100-image cycle released at 10 Hz and the application thresholds and warm-up protocol of stage 4. Every active configuration completed 600 frames, six per image, with no observed drop and no miss of the 100 ms release-to-result deadline. The simulated queue has capacity one and keeps the most recent frame; each skipped slot is counted as a drop.

`tegrastats` samples are requested every 100 ms and timestamped with the host monotonic clock on arrival. The collector stores **all** rails and the original lines in `telemetry.jsonl`, checks before stabilization that the process is alive and reporting the three required rails, and aborts before measuring if any is missing. Integration uses the instantaneous value before the `/`, converts mW to W, takes samples common to the three rails, and interpolates at the boundaries of the frames' own interval. It rejects missing rails, non-finite values, insufficient coverage and gaps above 0.5 s; observed gaps stayed below 0.113 s.

```
E_frame = ∫ P_module(t) dt / N_completed
```

Each `summary.json` stores the module total and the three per-rail integrals, with profile, scope, interval, mean watts, joules and joules per frame. Idle has energy and power but no joules per frame. The summarizer pairs the idle of the same block and subtracts its mean power times the active run's duration to produce the incremental figures; negative values after subtraction would be preserved for analysis, not clipped. `results.csv` carries the totals and the achieved rate, `energy_rails.csv` the per-rail breakdown, and `report.json` both plus between-launch statistics — which, with a single repetition, are not estimated.

Two caveats that belong with any citation of these numbers: sensor-to-host delay is not calibrated, and idle-subtracted values are only as stable as the idle baseline. Measured idle is 7.002 W for the FP16 campaign and 7.001 W for INT8; the INT8 campaign used its own new idle interval, never the FP16 one.

A campaign already collected with a single `--rail` can be recomputed as module power from the stored telemetry, without re-running inference, by passing `--power-profile agx-orin` to `summarize` with a new output directory. Without that flag, older results stay labelled as single-rail measurements.

## Aggregation rules

`summarize` reads a stage directory and writes `results.csv`, `report.json` and, for energy, `energy_rails.csv`, keeping individual launches distinct. It verifies the hashes of archived evidence, refuses to aggregate AP across runs with different images, hashes or thresholds, does not treat thousands of frames as independent replicates, and does not merge separate campaigns. The reported values are collected from these summaries by `run.py collect-results`, see [`RESULTS.md`](RESULTS.md).

## What was deliberately not measured

A concurrent GPU workload and a robotics replay were planned as stage 6 and are **absent**. Without them no claim of resource isolation, GPU availability for another task, or improved concurrent-task performance is supportable, and the paper makes none. The remaining limitations are collected in [`PROVENANCE.md`](PROVENANCE.md).
