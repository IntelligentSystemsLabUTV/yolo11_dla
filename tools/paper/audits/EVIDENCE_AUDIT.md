# Reproducible evidence audit

From the workspace root, run:

```sh
python tools/YOLO11-DLA_ICRA/analysis/analyze_fp16.py
```

Dependencies: Python 3, NumPy, and Matplotlib. The script reads the original
`logs/fp16_matrix/` files without changing them. It checks 20,167 timing records
against the benchmark summaries, maps profiler entries to engine-inspector
layer types, and records SHA-256 hashes of input artifacts. This executes no
Jetson benchmark and no new accuracy evaluation. Alternate input/output paths
are available through `--logs`, `--output`, and `--figures`.

Generated outputs:

- `fp16_audit.json`: full measurements, commands, source hashes, hardware
  evidence, limitations, and explicitly separate secondary E2E summaries.
- `engine_summary.csv`: the four YOLO11 configurations.
- `layer_profile.csv`: every separate-profile entry with its inspector type.
- `../figures/latency_comparison.pdf` and `.svg`: paired device and
  transfer-inclusive medians. Whiskers extend to the within-run 95th percentile.
  The GPU and DLA panels have different, explicitly labeled horizontal scales.
- `../figures/fallback_profile.pdf` and `.svg`: summed profiler means and
  optimized-engine node counts. Profile data are from a separate run, with 129
  profile samples for fallback and 137 for strict DLA. The count panel omits
  no-op and constant metadata entries.
- `../figures/training_history.pdf` and `.svg`: validation metrics stored in
  `checkpoint_training.json`. This plot is generated when that file exists.
  It contains one training history with no inferred seed uncertainty.

`checkpoint_training.json` and `.csv` were separately extracted from the
supplied `logs/yolo11n-dla-500ep.pt` checkpoint; their source hash and training
metadata are retained in the JSON. `verify_model.py` and
`model_verification.json` separately document model/export parity checks.

## Quantitative findings

| Configuration | Device median (ms) | Transfer-inclusive median (ms) | P95 (ms) | Throughput (q/s) | Timed queries |
|---|---:|---:|---:|---:|---:|
| YOLO11n, GPU | 3.46680 | 3.74023 | 3.75293 | 288.2730 | 8,651 |
| YOLO11-DLA-n, GPU | 3.15039 | 3.50000 | 3.51172 | 317.2070 | 9,519 |
| YOLO11n, DLA+GPU fallback | 34.19850 | 34.67770 | 34.94190 | 29.1688 | 878 |
| YOLO11-DLA-n, strict DLA | 26.81640 | 27.51460 | 27.71780 | 37.2062 | 1,119 |

The strict retrofit reduces the measured transfer-inclusive median by 20.656%
and increases reported saturated throughput by 27.555% relative to baseline
fallback. The GPU retrofit reduces the transfer-inclusive median by 6.423%
relative to the GPU baseline. These compare different learned architectures and
output interfaces; they do not isolate the causal effect of a single graph edit.

The baseline strict build fails first at
`/model.10/m/m.0/attn/MatMul`. Its fallback inspector has four DLA loadables,
22 reformat nodes, eleven other GPU compute nodes, and ten no-op/constant
entries. The strict retrofit inspector has exactly one DLA loadable and no GPU
or reformat nodes. The separate fallback profile assigns 32.24401 ms (93.97318%)
to DLA, 1.6215468 ms (4.7258921%) to reformat nodes, and 0.4463994 ms
(1.3010006%) to GPU compute nodes. Thus, 94.0% denotes profiled time share,
not a percentage of layers, operators, FLOPs, or electrical power. The strict
profile reports 26.8716 ms in its sole DLA entry. These profile means should not
be substituted for the benchmark-run device or transfer-inclusive statistics.

## Evidence boundaries

1. **A filename is not model identity.** The file prefix
   `yolo11n-dla-gpu-fallback-fp16` uses the original `yolo11n.onnx`, as its
   build command establishes. It is the baseline fallback configuration.
2. **Warmup is already excluded.** Each timing JSON has exactly the number of
   entries in the reported timed trace. Some first enqueue timestamps precede
   the nominal 2,000 ms warmup boundary, but their device computations begin
   afterward. Do not discard entries using enqueue timestamps or remove an
   additional 2 s.
3. **Latency is a defined sum.** `trtexec` reports H2D + device interval + D2H.
   The JSON independently rounds these values, so their sums differ by tiny
   rounding errors. It is not necessarily the elapsed time between the first
   transfer event and the last event when queued work overlaps. The label
   `GPU Compute Time` also describes a DLA engine's device interval. It does not
   establish GPU execution for that engine.
4. **Throughput is distinct.** The benchmark pipelines work and allows
   transfer/execution overlap, so queries per second are not
   `1000 / median latency`. The sequential image rate in the E2E summary is a
   different metric.
5. **Percentiles describe one run.** Every configuration has one process/run.
   Its many query samples do not create independent experimental replicates.
   The figure whiskers are within-run P95 endpoints, not confidence intervals.
   For these traces, the observed TensorRT percentile convention is the sorted
   sample at index `floor(n*p/100)`; NumPy's default interpolation gives a
   slightly different P95/P99 for the 878-sample fallback trace. Both forms are
   retained in JSON and the manuscript uses the reported TensorRT values.
6. **The E2E rows are secondary evidence.** The four relevant rows exist in
   `FP16_RESULTS.md`, but their raw YOLO11 E2E JSON files are absent. The two
   E2E JSON files actually present concern the excluded custom YOLO-DLA model.
   Consequently no new E2E distribution or confidence interval is generated.
   The reported 56.497 ms strict versus 52.782 ms fallback medians indicate
   7.038% higher sequential E2E latency in that summary, despite the better
   engine-boundary result. The stages were independently summarized and their
   medians must not be added and labeled the total median.
7. **Device identity is incomplete.** The logs establish Orin, compute
   capability 8.7, 16 SMs, 62,840 MiB, 256-bit memory bus, TensorRT 10.3.0, and
   DLA core 0. They do not establish an exact carrier/module SKU, JetPack/L4T,
   CUDA/cuDNN versions, power mode, actual clocks, or thermal/power measurements.
   The application clock fields (1.3 GHz compute and 0.816 GHz memory) carry an
   explicit warning that they do not reflect actual running clock rates.
8. **FP16 is a build mode.** The builds use `--fp16`; TensorRT reports
   `FP32+FP16` as the permitted precision set. Inspector output formats include
   FP32 intermediates in the GPU retrofit and fallback builds. The strict-DLA
   interface is FP16. Avoid asserting that every operation of every engine is
   FP16.
9. **Layout evidence has limits.** The strict builder requests
   `--inputIOFormats=fp16:dla_hwc4` and `--outputIOFormats=fp16:chw16`.
   The inspector describes the input as channel major FP16 with channel
   divisibility by four. The raw YOLO11 E2E runtime queries needed to resolve
   format naming and byte strides are missing; do not infer their values from
   another model's JSON.
10. **Compatibility does not establish all system benefits.** One DLA
    loadable is strong placement evidence for this learned engine. It does not
    establish lower energy, no CUDA API/runtime involvement, absence of shared
    memory interference, an application-wide GPU-free implementation, or
    superior concurrent-workload behavior. Those require final experiments.

## Suggested figure captions

**Latency:** TensorRT engine-boundary latency for FP16-enabled builds at batch
one and 640 × 640. Bars show medians; whiskers extend to within-run P95. The
device interval is the quantity TensorRT labels “GPU Compute Time,” including
when computation runs on DLA. Transfer-inclusive latency adds H2D and D2H;
image preprocessing and detector postprocessing are excluded. Panel scales
differ. Each configuration has one 30 s run after 2 s warmup.

**Profile:** Engine placement and separately measured profile composition.
The fallback baseline spends 94.0% of profiled time in DLA loadables while
retaining GPU computation and reformatting. The adapted engine is one DLA
loadable. Upper bars sum means from separate profile runs and exclude external
H2D/D2H; lower counts are optimized inspector entries, excluding no-op and
constant metadata.

**Training:** Validation AP stored in the supplied 500-epoch YOLO11-DLA-n
checkpoint, ending at AP50 = 53.151 and AP50:95 = 37.581. The curve describes
one stored training run; it is not a fresh evaluation or validation of the
serialized TensorRT engine.

## Recovery update — 2026-09-09

The August 25 conversation attachments were recovered and archived under
[../../yolo11_dla_paper/history/sources/](../../experiments/history/sources/index.json).
The final E2E terminal output directly confirms all four YOLO11-family summary
rows, including strict median 56.497 ms and fallback median 52.782 ms, and records
the strict input CHW4/output CHW16 formats with context strides. This updates
points 6 and 9 of the original audit: terminal provenance and runtime binding
records are now available. The per-frame JSON arrays remain missing; no repeated
experiment, new confidence interval, engine hash chain or dataset AP was recovered.
The earlier 20/200 default-stream/dense runs are archived separately from the
later 10/100 runs with sparse decoding for the adapted model. See
[HISTORICAL_COMMANDS.md](../../../docs/PROVENANCE.md).
