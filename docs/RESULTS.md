# Results and their sources

Every published number is traced here to the file it came from. `11n` is stock YOLO11-n, `11-DLA` is YOLO11-DLA-n; `DLA` is strict placement with fallback disabled, `DLA+GPU` is the stock model built with `--allowGPUFallback`. All measurements: Jetson AGX Orin, L4T 36.4.4, TensorRT 10.3.0, batch one, 640×640, 50 W, `jetson_clocks` disabled.

The published values are **collected** from the measured summaries, not typed:

```bash
python tools/experiments/run.py collect-results    # writes analysis/precision_results.json
```

That command reads the eight CSVs listed below and records their SHA-256 in [`../analysis/precision_results.json`](../analysis/precision_results.json), which is tracked and is the machine-readable version of this page. The CSVs themselves are campaign output under the untracked `logs/`: the recorded hashes are what ties a value to the run that produced it.

## Engine-plus-transfer latency

`trtexec`, median and P95 in ms, rate in queries/s. Boundary: H2D + device + D2H; no preprocessing, decoding or NMS.

| Model / device | Mode | Median | P95 | Rate |
|---|---|---:|---:|---:|
| 11n / GPU | FP16 | 3.740 | 3.753 | 288.3 |
| 11n / DLA+GPU | FP16 | 34.678 | 34.942 | 29.2 |
| 11-DLA / GPU | FP16 | 3.500 | 3.512 | 317.2 |
| 11-DLA / DLA | FP16 | 27.515 | 27.718 | 37.2 |
| 11n / GPU | INT8 | 3.021 | 3.030 | 357.0 |
| 11n / DLA+GPU | INT8 | 13.068 | 13.094 | 79.5 |
| 11-DLA / GPU | INT8 | 2.828 | 2.836 | 393.4 |
| 11-DLA / DLA | INT8 | 4.167 | 4.173 | 263.7 |

- FP16 rows: [`../analysis/engine_summary.csv`](../analysis/engine_summary.csv), derived by `run.py audit` from the four raw traces in `logs/fp16_matrix/*.times.json` (8,651 + 9,519 + 878 + 1,119 = **20,167 timing samples**, one launch per configuration). Mirrored with uniform names under `logs/icra2027_final/existing/micro/` and summarized in `logs/icra2027_final/existing/summary-micro/results.csv`.
- INT8 rows: `logs/icra2027_int8/run_01/micro-summary/results.csv`, from `logs/icra2027_int8/run_01/micro/repeat01-*.times.json` (2,387 + 7,915 + 11,805 + 10,713 samples).

Derived statements in the text: INT8 cuts strict DLA median from 27.515 to 4.167 ms, a 6.60× speedup, and raises throughput from 37.2 to 263.7 queries/s. Strict DLA beats stock fallback at this boundary in both modes — 20.7% lower median in FP16, 3.14× lower in INT8 — while the same adapted model on GPU stays faster, at 3.500 ms FP16 and 2.828 ms INT8. Strict INT8 device execution alone is 3.789 ms (`computeMs_median` in the INT8 micro summary).

## Accuracy on the 4,500 held-out COCO val2017 images

Disjoint from the 500 calibration images. AP in percentage points. Confidence 0.001, class-aware NMS IoU 0.7, multi-label, max 300 detections, COCOeval standard limits.

| Model / device | Mode | AP | AP₅₀ | AP₇₅ |
|---|---|---:|---:|---:|
| 11n / GPU | FP16 | 39.378 | 55.190 | 42.845 |
| 11n / DLA+GPU | FP16 | 39.308 | 55.160 | 42.823 |
| 11-DLA / GPU | FP16 | 37.879 | 53.633 | 41.008 |
| 11-DLA / DLA | FP16 | 37.876 | 53.614 | 41.016 |
| 11n / GPU | INT8 | 33.175 | 47.324 | 36.069 |
| 11n / DLA+GPU | INT8 | 19.827 | 35.241 | 20.867 |
| 11-DLA / GPU | INT8 | 35.951 | 51.063 | 39.323 |
| 11-DLA / DLA | INT8 | 35.408 | 50.790 | 38.614 |

- FP16 rows: `logs/icra2027_int8/run_01/fp16-heldout-summary/results.csv` — the stored FP16 predictions rescored on exactly the INT8 held-out IDs, with no new inference.
- INT8 rows: `logs/icra2027_int8/run_01/accuracy-summary/results.csv`. The `stock-gpu` row is the linked complete evaluation whose manifest is preserved at `logs/int8-evaluation.8ZbovD/accuracy-stock-gpu/manifest.json`.
- Both summaries also carry AP_small / AP_medium / AP_large and the engine SHA-256 of every row.

Derived statements: strict INT8 loses 2.468 AP against strict FP16 while adapted GPU loses 1.928. Within a precision mode, strict DLA minus adapted GPU is **−0.003** AP in FP16 and **−0.543** in INT8 — the negligible FP16 placement difference does not carry over to INT8. Stock GPU loses 6.203 points; stock fallback loses 19.48 and reaches only 19.827 AP. That last result is **not** an accuracy-equivalent baseline and has no isolated causal explanation from these tests; architecture, training budget and calibration all differ, so it supports no general claim about quantization robustness.

The full-5,000-image FP16 values, from `logs/icra2027_final/run_01/summary-accuracy/results.csv`, are 39.314 (stock GPU), 39.245 (stock fallback), 37.995 (adapted GPU) and 37.991 (strict DLA). They are a different cohort and must not be compared with held-out INT8 AP. The checkpoint's own 500-epoch validation history ends at AP 37.581 / AP₅₀ 53.151 under yet another protocol and establishes nothing about export.

## Application pipeline

Same 100 images, 60 s per test, latency in ms, rate in frames/s. Includes preprocessing, native I/O, inference, decoding and NMS.

| Model / device | FP16 median | P95 | P99 | Rate | INT8 median | P95 | P99 | Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 11n / GPU | 20.030 | 24.658 | 27.702 | 47.99 | 17.556 | 22.104 | 23.681 | 54.70 |
| 11n / DLA+GPU | 51.241 | 59.542 | 64.478 | 18.95 | 29.020 | 37.510 | 40.293 | 32.71 |
| 11-DLA / GPU | 20.307 | 24.584 | 28.591 | 47.63 | 18.212 | 22.467 | 23.780 | 53.00 |
| 11-DLA / DLA | 56.820 | 65.042 | 76.886 | 17.19 | 30.675 | 34.812 | 36.322 | 31.93 |

- FP16: `logs/icra2027_final/run_01/summary-pipeline/results.csv`, per-frame records in `logs/icra2027_final/run_01/pipeline/repeat01-*/frames.jsonl`.
- INT8: `logs/icra2027_int8/run_01/pipeline-summary/results.csv`, records in `logs/icra2027_int8/run_01/pipeline/repeat01-*/frames.jsonl`.

**The ordering reverses here.** Strict DLA is 10.9% slower than stock fallback in FP16 and 5.7% slower in INT8, despite winning at the engine boundary. The INT8 stage medians explain why: strict DLA spends 7.463 ms on input packing, quantization and H2D, 4.522 ms in the engine call, 10.729 ms on output transfer, unpacking and dequantization, and 4.628 ms in post-processing, against 2.318 / 13.467 / 1.500 / 5.830 ms for stock fallback. Native I/O processing on the CPU is a first-order cost in this implementation. Stage medians do not sum to the median total, and the `engine_ms` stage includes runtime-call and synchronization overhead beyond device execution.

Host effects are not uniform either: INT8 median preprocessing is 2.954 ms for strict DLA against 5.568 ms for fallback **on the same images and the same code**. Frequency governor and load are plausible explanations; the logs do not prove a cause.

## Module energy at 10 Hz

600 frames per test, `VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0`. Power in W, energy in J/frame.

| Model / device | Mode | Power | Total | Incr. |
|---|---|---:|---:|---:|
| 11n / GPU | FP16 | 7.605 | 0.761 | 0.060 |
| 11n / DLA+GPU | FP16 | 9.166 | 0.917 | 0.216 |
| 11-DLA / GPU | FP16 | 7.606 | 0.761 | 0.060 |
| 11-DLA / DLA | FP16 | 8.738 | 0.874 | 0.174 |
| 11n / GPU | INT8 | 7.599 | 0.760 | 0.060 |
| 11n / DLA+GPU | INT8 | 7.961 | 0.796 | 0.096 |
| 11-DLA / GPU | INT8 | 7.599 | 0.760 | 0.060 |
| 11-DLA / DLA | INT8 | 8.062 | 0.806 | 0.106 |

- FP16: `logs/icra2027_final/run_01/summary-energy-10hz-module/{results,energy_rails}.csv`, telemetry in `.../energy-10hz-module/repeat01-*/telemetry.jsonl`, idle 7.002 W.
- INT8: `logs/icra2027_int8/run_01/energy-summary/{results,energy_rails}.csv`, telemetry in `.../energy-10hz/repeat01-*/telemetry.jsonl`, idle 7.001 W.

Strict INT8 uses 0.806 J/frame, 7.7% below strict FP16 but 6.1% above adapted INT8 GPU. Strict FP16 uses 4.7% less than stock FP16 fallback, but strict INT8 is slightly **above** its fallback counterpart (0.796 J/frame) — whose accuracy, at 19.827 AP, is far lower. GPU module energy sits at 0.760–0.761 J/frame in both modes. These are single-launch values at one operating point, and the idle-subtracted column inherits the baseline's stability.

`logs/icra2027_final/run_01/energy-10hz/` is a **failed** attempt, aborted because `tegrastats` was missing; it is kept as a record and excluded from results. The usable FP16 campaign is `energy-10hz-module`.

## Placement and the FP16 layer profile

The placement claim and its inspector evidence are tabulated in [`BUILD.md`](BUILD.md). The separate FP16 layer profile — 93.97% DLA, 4.73% reformats, 1.30% GPU compute, 34.312 ms summed device profile with 32.244 ms in DLA segments — comes from `logs/fp16_matrix/*.profile.json` via `run.py audit`, tabulated in [`../analysis/layer_profile.csv`](../analysis/layer_profile.csv) and recorded in [`../analysis/fp16_audit.json`](../analysis/fp16_audit.json). It characterizes FP16 only.

## Export, layout and decoder checks

| Claim | Value | Source |
|---|---|---|
| CPU external-decoder error, one random input (seed 7) | max abs 1.22×10⁻⁴ | `analysis/model_verification.json` |
| ONNX vs checkpoint raw maps, same input | max abs 3.07×10⁻⁴ | same |
| Layout checks per precision campaign | 8 passed | `*/parity/comparison/report.json` |
| Dense/sparse decoder comparisons per campaign | 200 passed | same |
| Raw GPU/DLA gate, FP16 P3/P4/P5 | fails on 50/49/49 of 50 | same |
| Raw GPU/DLA gate, INT8 | fails on 50/50 at every level | same |
| Dense vs sparse AP, 5,000 FP16 images | differ by < 10⁻⁷ points | `logs/icra2027_final/run_01/summary-accuracy/results.csv` |
| Dense vs sparse detections, FP16 GPU image 74092 | 252 sparse vs 251 dense | `logs/icra2027_final/run_01/accuracy/*/manifest.json` |
| Dense vs sparse AP, 4,500 INT8 images | identical at reported precision | `logs/icra2027_int8/run_01/accuracy-summary/results.csv` |
| Dense vs sparse detections, INT8 GPU image 408774 | 295 sparse vs 294 dense | `logs/icra2027_int8/run_01/accuracy/*/manifest.json` |

Aggregate AP agreement is not bitwise identity, and it is not equality of every post-NMS detection. The raw gate is reported diagnostically; it was not passed.

## Numbers that are not results

`dla_yolo11n*.md` in the development workspace documents **segmentation at 512**, not the detection-at-640 work in the paper. The `yolo-dla-n-*` artifacts belong to an independent model family outside the paper's scope, and the only end-to-end JSON records kept in `fp16_matrix/` are theirs: they cannot fill YOLO11-DLA rows. `best-mAP50-95_0.37395.pt` is named after a value that its own contents do not confirm. See [`PROVENANCE.md`](PROVENANCE.md).
