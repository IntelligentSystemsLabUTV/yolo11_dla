# Provenance and limitations

This page records where each artifact came from, what was reused rather than re-measured, which traps exist in the historical material, and what the evidence does **not** support. It is deliberately written to be usable against the paper: if a claim is not listed as measured here, it was not measured.

## What was measured when

| Stage | FP16 | INT8 |
|---|---|---|
| Engine builds and inspection | Collected earlier, registered into `logs/icra2027_final/existing/` without rebuilding | Collected earlier in `logs/int8_matrix/`, registered into `logs/int8_ready/` without rebuilding |
| Engine-plus-transfer microbenchmark | **Reused**: four historical traces, one launch each, 20,167 samples | Newly collected, one launch each |
| Layer profiles | **Reused**: historical `--separateProfileRun` output for fallback and strict | Not collected |
| Parity (layout + decoder) | `run_01/parity` | `run_01/parity`, with `--raw-diagnostic-only` |
| COCO accuracy | `run_01/accuracy`, six evaluations over 5,000 images | `run_01/accuracy`, five evaluations over 4,500 held-out images, plus one linked stock-GPU evaluation |
| Paired FP16 AP on the held-out cohort | Rescored from stored FP16 predictions, no inference | — |
| Application pipeline | `run_01/pipeline`, four configurations, 60 s, one repeat | `run_01/pipeline`, same |
| Module energy at 10 Hz | `run_01/energy-10hz-module`, four configurations plus idle | `run_01/energy-10hz`, same, with its own idle |

Nothing was rebuilt to populate a tidier directory structure. Registration creates relative links, manifests and hashes; every later stage verifies the engine hash before it measures.

## Verified hash chain

The chain from trained weights to measured engine is stated as hashes across the tracked audit files and reproduced by each campaign's own manifests:

```
checkpoint                  417967ac…  ← checkpoint_training.json, model_verification.json
        ↓ prepare (opset 20, static 1×3×640×640)
adapted ONNX export         2712fe11…  ← int8_calibration/adapted/manifest.json, model_verification.json
        ↓ int8-calibrate (EntropyCalibrator2, 500 images, seed 2027, pre-fusion)
logs/int8_calibration/adapted/calibration.cache   f9082ab7…  ← int8_ready/adapted-strict.engine.quantization.json
        ↓ trtexec --useDLACore=0 --int8 --fp16 --inputIOFormats=int8:dla_hwc4 --outputIOFormats=int8:chw32
logs/int8_matrix/debug-strict-native-int8.gKgeWc/adapted-strict-int8.engine   fd530775…
        ↓ registered as logs/int8_ready/adapted-strict.engine (symlink, same bytes)
        ↓ measured by parity / accuracy / micro / pipeline / energy, each verifying the hash
logs/icra2027_int8/run_01/*/manifest.json + accuracy-summary/results.csv   artifact_sha256 fd530775…
```

The eight CSVs behind the manuscript tables are hash-recorded in `tools/paper/analysis/precision_results.json`, which is tracked; the CSVs themselves are campaign output under the untracked `logs/`.

The **stock** side of the chain is weaker by construction. `logs/yolo11n.onnx` (`a771561a…`) is the export that every stock engine was built from, and it is consistent across the calibration manifests and build commands, but the identity of the upstream `yolo11n.pt` checkpoint it came from was never recovered. Any stock-versus-adapted AP difference therefore mixes architecture, training budget and checkpoint provenance; the adapted GPU engine, not the stock model, is the controlled comparison for placement.

## Traps in the historical material

Three of these were found during the evidence audit and are worth repeating, because filenames lie:

1. **`yolo11n-dla-gpu-fallback-fp16.*` is stock YOLO11n**, not the adapted model. The `dla` in the prefix refers to the DLA fallback build, and the build command loads `yolo11n.onnx`. It is the stock DLA+GPU fallback baseline.
2. **The only end-to-end JSON records kept in `fp16_matrix/` belong to the independent YOLO-DLA family**, which is outside the paper's scope. They cannot be used to fill YOLO11-DLA rows.
3. **`dla_yolo11n*.md` in the development workspace documents segmentation at 512×512**, an earlier phase of the work, not the 640×640 detection results of the paper. `best-mAP50-95_0.37395.pt` is named after a value its own contents do not confirm. Neither belongs to this study.

Historical absolute paths are preserved faithfully in the archived records rather than rewritten: the original TensorRT logs refer to `logs/paper/fp16_matrix/…`, which does not correspond to any current directory, and the remaining absolute paths are the container's own mount point on the Jetson. Those strings are provenance, not usable paths.

## Command archive

`tools/experiments/history/` is an archive, not a runbook. `commands.json` holds 64 commands with original text, source, line, SHA-256 and evidence type: 11 recovered from preserved TensorRT logs (5 builds, 4 benchmarks, 2 profiles, each with its `RUNNING` line and `PASSED`/`FAILED` outcome), 10 recovered from terminal attachments, and 43 that were only *proposed* in conversation and prove nothing about execution. `OBSERVED_COMMANDS.md` collects only the commands with execution evidence. `runtime/` keeps the version of the end-to-end tester that predates the September fixes, so that older numbers are not silently attributed to the corrected code. `migration.json` records the file hashes from the earlier reorganization of these scripts.

Two details recovered this way matter for reading Table I: `--infStreams=1` was not explicit in the original benchmark commands but is reported in the logs' Inference Options section, and the historical profiling runs used `--warmUp=1000 --duration=0 --iterations=100 --separateProfileRun`, whose exported profiles actually contain 129 and 137 samples. The current suite profiles for 10 s instead, so the two protocols must not be pooled.

## Software checks that were passed

These are host-side regression tests of the harness, run without an accelerator. They do not certify TensorRT or DLA execution, and no result below is a scientific measurement.

- 33 tests in `tools/experiments/tests/`: NMS candidate edge cases and threshold ties, end-to-end NMS, sparse/dense multi-label and empty cases, CHW16/CHW4/DLA_HWC4/DLA_LINEAR packing with padding, power rail integration and coverage/gap handling, pacing and drops, strict flags and inspector acceptance, randomized repetition blocks, reuse without export, rejection of modified engines and of caches bound to another ONNX, and reproducible disjoint INT8 splits.
- The inspector was checked against the real JSON files: it accepts the single-node strict DLA engine and rejects both the four-node fallback and the pure GPU engines as strict.
- The micro summarizer was checked against the existing stock GPU log: all 8,651 samples retained, median 3.74023 ms, warm-up not discarded twice.
- The energy collector was exercised against a simulated `tegrastats` process, including the correct rejection of a source exposing only one rail.
- `compileall` and `pyflakes` pass. Flake8 in the development environment has a plugin conflict on `--application-import-names`, so static checking was done with pyflakes directly.
- The COCO evaluator's CPU branch was smoke-tested against an explicitly synthetic single-image annotation; that AP is a fixture, not a result.
- PDF preflight (`make -C tools/paper check`): eight pages, US Letter, 14 embedded fonts, no Type 3 fonts, no links, no bookmarks, no overfull boxes, no unresolved references, empty author metadata.

## Limitations

**Scope.** One board, one L4T/TensorRT stack, the nano scale, detection only, 640×640, batch one, and the tested FP16 and INT8 build modes. Other DLA generations, input sizes, scales, tasks and precision assignments are untested. INT8 uses one 500-image calibration subset; alternative calibration sets, quantization-aware training and other precision assignments were not evaluated.

**Statistics.** One launch per configuration per stage. Percentiles describe frame variation *inside* a test; there are no confidence intervals across independent runs. With `jetson_clocks` disabled, frequency and scheduling variation affect host processing, and the 4.7% FP16 energy difference in particular deserves repetitions before it is leaned on. The 100-image cycle is not a full-dataset timing distribution.

**Placement is not isolation.** Strict DLA energy traces still reach 7% GR3D utilization in FP16 and 10% in INT8, and the runtime still needs a CUDA stream and device-visible buffers. CPU, memory, power, transfers and runtime remain shared. No concurrent GPU workload was measured, so nothing here supports a claim of resource isolation, recovered GPU capacity, or improved concurrent-task performance.

**Numerical equivalence was not established.** The exploratory raw GPU/DLA gate fails in both precisions and is reported as a diagnostic. Aggregate AP agreement between dense and sparse decoding is not bitwise identity. TensorRT itself warns that the DLA softmax approximation can introduce numerical error.

**Causal attribution.** The stock INT8 fallback's 19.827 AP cannot be attributed to a specific operator from these tests. The preprocessing difference between GPU and DLA paths on identical data has plausible explanations but no proof in the logs. Zero observed deadline misses at 10 Hz do not establish hard real-time guarantees, and junction temperatures of 45.9–47.4 °C do not constitute a throttling audit.

**Energy scope.** Module power, three rails, not wall-plug and not the full carrier. Sensor-to-host delay is uncalibrated. Idle-subtracted values depend on the stability of a single idle interval. Values are specific to the 10 Hz operating point.

**Environment.** ROS 2 Jazzy is present on the platform but camera acquisition, transport, mapping, control and any robot-level integration are outside the measured boundary. The benchmark is a standalone runner, not a ROS 2 node, and a COCO image sequence is not a rosbag.

## Anonymization

This branch is prepared for anonymous review. Three kinds of value were redacted from otherwise unmodified records, and nothing else in them was altered; no measured value was touched.

Absolute filesystem paths and the source repository URL appear as `<redacted>` in `tools/paper/analysis/checkpoint_training.json` and inside the two files under `weights/` — the checkpoint recorded them in `train_args`, in its `git` block and in the serialized model's own `yaml` and `args` attributes, and the ONNX recorded one in its `description` metadata. The Weights & Biases entity in the training scripts appears as `<wandb-entity>`.

Redacting strings inside a binary changes its file hash, so `weights/` does not match the as-built SHA-256 that the manifests record. What is unchanged is checkable and is the right thing to compare: the checkpoint's parameters are bit-identical (`sha256` over the sorted `state_dict` tensor bytes, `94e3df7af25b7d803c59255c174b1c4d…`) and the ONNX graph is bit-identical (`sha256(graph.SerializeToString())`, `c15763f5fa8e56fc0d4cc71424dc3c50…`), with every ONNX metadata key preserved. Both hash pairs are tabulated in [`BUILD.md`](BUILD.md). The container definitions, the workspace scaffolding and the editor configuration of the full repository are omitted here because they carry organizational identifiers; they are not needed to build, run or audit anything documented in these pages.

## Detailed audits

The long-form audits are preserved verbatim under [`../tools/paper/audits/`](../tools/paper/audits/):

| File | Contents |
|---|---|
| `EVIDENCE_AUDIT.md` | Source and provenance audit of the historical FP16 evidence, measurement definitions, quantitative findings and statistical limits |
| `MODEL_AUDIT.md` | Implementation provenance, training discrepancies and runtime caveats |
| `PRECISION_UPDATE.md` | Sources, interpretation and reproduction commands for the combined FP16/INT8 results |
| `REFERENCE_AUDIT.md` | Primary-source verification of every citation |
| `GUIDELINES.md` | Submission requirements the manuscript was prepared against |
| `NOTES.md` | Revision record of the manuscript, including corrections made to earlier claims |
