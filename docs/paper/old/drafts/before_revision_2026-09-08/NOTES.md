# NOTES — YOLO11-DLA ICRA 2027 (descoped, single-model)

Layout choice: `main.tex` (documentclass, title, abstract, keywords, bibliography) plus `sections/*.tex` and `figures/*.tex`, one file per section. Build with `make` (runs `tectonic` inside the DUA container, since the host has no TeX) or `make local`. `make todo` lists every placeholder. `make pages` reprints the page count from `main.log`.

Class: `\documentclass[conference]{IEEEtran}` as requested. `mathptmx` is loaded because IEEEtran selects the `ptm` text family but no math family, which left all math in Computer Modern; this is a font-correctness fix, not a size change, and it did not change the page count (9 pages before and after).

`structure.md` was **not** found anywhere in the repository or attached to the task. I followed the structure summary given in the task text verbatim (section order, section budgets, four figures, RQ-organized Sec. V). Flagging it in case the file exists elsewhere and contains constraints I have not seen.

## 1. Provenance of every number in the manuscript

Two sources only, as instructed: **(a)** `logs/fp16_matrix/` (device measurements), **(b)** `tools/YOLO-DLA_ICRA/` (model-level accuracy). Paths below are relative to the repository root. "Recomputed" means I derived the value from raw fields rather than copying the old paper's arithmetic; every such value reproduced the old paper's figure unless the discrepancy list in §3 says otherwise.

### 1.1 Platform (Sec. V-A)

| Value | Source | Field |
|---|---|---|
| device name `Orin`, compute capability 8.7, 16 SMs, 62 840 MiB device memory, 256-bit bus, 1.3 GHz application compute clock | (a) `logs/fp16_matrix/yolo11n-gpu-fp16.build.log` | `=== Device Information ===` block |
| TensorRT 10.3.0 | (a) every `*.build.log` / `*.benchmark.log` | banner `TensorRT v100300`, `TensorRT version: 10.3.0` |
| DLA core 0 | (a) all DLA `*.build.log` / `*.benchmark.log` | `--useDLACore=0`, `DLACore: 0` |
| DLA pools: 1 MiB managed SRAM, 1024 MiB local DRAM, 512 MiB global DRAM | (a) `yolo11n-dla-gpu-fallback-fp16.build.log` | `DLA Memory Pool Sizes:` |
| trtexec protocol: 2 s warm-up, 30 s duration, 1 stream, spin-wait, transfers enabled | (a) `*.benchmark.log` | `=== Inference Options ===` block |
| tester protocol: 10 warm-up / 100 timed iterations, CPU post-processing, conf 0.25, IoU 0.45, 960x1280 source letterboxed to 640x640, disk I/O and drawing excluded | (a) `logs/fp16_matrix/FP16_RESULTS.md` | "Common E2E conditions" |

### 1.2 Accuracy (Table II, Sec. V-B)

| Value | Source |
|---|---|
| YOLO11n 65.64 / 50.25 / 55.14 / 39.40 | (b) `tools/YOLO-DLA_ICRA/sections/experimental_protocol.tex`, `tab:model_results` |
| YOLO11-DLA-n 64.70 / 49.03 / 53.17 / 37.68 | (b) same table |
| retention 95.6 % | recomputed: 37.68 / 39.40 = 0.95634 |
| deltas 0.94 / 1.22 / 1.97 / 1.72 points | recomputed from the two rows |

### 1.3 Engine boundary (Table III, Sec. V-C)

All from (a) `logs/fp16_matrix/*.benchmark.log`, fields `Latency: median`, `percentile(95%)`, `Throughput`, `H2D Latency`, `GPU Compute Time`, `D2H Latency`.

| Configuration | log file | median / P95 / throughput |
|---|---|---|
| YOLO11n / GPU | `yolo11n-gpu-fp16.benchmark.log` | 3.74023 / 3.75293 / 288.273 |
| YOLO11n / DLA+GPU | `yolo11n-dla-gpu-fallback-fp16.benchmark.log` | 34.6777 / 34.9419 / 29.1688 |
| YOLO11-DLA-n / GPU | `yolo11n-dla-gpu-fp16.benchmark.log` | 3.5 / 3.51172 / 317.207 |
| YOLO11-DLA-n / strict DLA | `yolo11n-dla-strict-dla-fp16.benchmark.log` | 27.5146 / 27.7178 / 37.2062 |

Naming note: in `logs/fp16_matrix/` the model tag `yolo11n-dla` is the retrofit (paper name YOLO11-DLA-n) and the mode tags are `gpu`, `dla-gpu-fallback`, `strict-dla`; the ONNX inputs named on the trtexec command lines (`yolo11n.onnx` vs `yolo11n-dla.onnx`) confirm the mapping.

| Derived value in text | Recomputation |
|---|---|
| 7.36x engine-boundary penalty | 27.5146 / 3.74023 = 7.357 |
| 7.86x vs the retrofit's own GPU engine | 27.5146 / 3.5 = 7.861 |
| 12.9 % of baseline throughput | 37.2062 / 288.273 = 0.1291 |
| 26.0 % (fallback above strict) | 34.6777 / 27.5146 = 1.2603 |
| 97.5 % of engine time is device execution | 26.8164 / 27.5146 = 0.9746 |
| H2D 0.393 / exec 26.816 / D2H 0.305 ms | `yolo11n-dla-strict-dla-fp16.benchmark.log` medians (sum 27.514, consistent with the 27.5146 median) |
| retrofit GPU vs baseline GPU: -6.4 % median, +10.0 % throughput | 3.5/3.74023 = 0.9358; 317.207/288.273 = 1.1004 |

| Engine artifact value | Source |
|---|---|
| 6.52 MiB strict engine, 8.15 MiB baseline GPU engine | `ls -l logs/fp16_matrix/*.engine`: 6 840 572 B / 1048576 = 6.523; 8 550 228 B = 8.154. (Also 7 190 740 B = 6.858 MiB fallback, 8 282 452 B = 7.899 MiB retrofit GPU — not used in the text.) |
| execution-context memory 0 / 9.47 / 22.80 MiB | `Total Activation Memory` in `*-strict-dla-*.build.log` (0), `yolo11n-gpu-fp16.build.log` (9 932 800 B = 9.473 MiB), `yolo11n-dla-gpu-fallback-fp16.build.log` (23 910 400 B = 22.804 MiB) |
| build time 18.7 s / 471.7 s | `Engine built in` in `yolo11n-dla-strict-dla-fp16.build.log` (18.7046 s) and `yolo11n-gpu-fp16.build.log` (471.742 s) |

| Placement/partition value | Source |
|---|---|
| strict build: 299 DLA layers, "No layer is running on GPU" | `yolo11n-dla-strict-dla-fp16.build.log`, `[DlaLayer]` count = 299, `---------- Layers Running on GPU ----------` followed by `No layer is running on GPU` |
| strict engine = one `DLA` layer with bindings `images`, `pred_p3/4/5` | `yolo11n-dla-strict-dla-fp16.layers.json` (single layer, `LayerType: DLA`) |
| rejection at `/model.10/m/m.0/attn/MatMul` | `yolo11n-strict-dla-fp16.build.log` lines 3338–3341 (`MatMul is unsupported on DLA`, `Assertion allowGPUFallback failed`, `DLA validation failed`) |
| fallback build: 317 DLA layers, 17 GPU layers = 2 MATRIX_MULTIPLY + 8 SHUFFLE + 2 SLICE + 1 ELEMENTWISE + 4 CONSTANT | `yolo11n-dla-gpu-fallback-fp16.build.log`, `Layers Running on DLA` / `Layers Running on GPU` lists (counted) |
| fallback reasons: "Input tensor has less than 4 dimensions", "Dimension: 3 (8400) exceeds maximum allowed size for DLA: 8192" | same file, ~line 3746 ff. |
| fallback engine after fusion: 47 nodes = 4 DLA + 11 GPU compute + 22 reformat + 7 no-op + 3 constant | `yolo11n-dla-gpu-fallback-fp16.layers.json`, census by `LayerType` (GPU compute = 6 PointWiseV2 + 2 gemm + 1 Shuffle + 1 CaskSoftMaxV2 + 1 CaskConvolution) |
| profiler shares 93.97 % DLA, 4.73 % reformat, 1.31 % GPU compute | `yolo11n-dla-gpu-fallback-fp16.profile.json`, `averageMs` per node aggregated by the `LayerType` of the matching node in `*.layers.json`; total 34.312 ms |
| largest DLA segment 42.54 %, 14.581 ms, ends at the attention transpose | same profile.json, node `{ForeignNode[/model.0/conv/Conv.../model.10/m/m.0/attn/Transpose]}`, `percentage` and `medianMs` |
| DLA softmax caveat | `yolo11n-dla-strict-dla-fp16.build.log` line 3316: "DLA Softmax optimization is enabled because the volume of non-axis dimensions is greater than 1. This may cause a small error in the output." |

### 1.4 Application-level accounting (Table IV, Fig. 3, Sec. V-D)

Stage medians, totals, P95/P99 and sequential rates: (a) `logs/fp16_matrix/FP16_RESULTS.md`, "Repeated-image E2E comparison" table and the four per-configuration distribution tables. I/O column = `Pack + H2D` + `D2H + unpack` (2.843 / 4.002 / 3.848 / 13.992 — recomputed). Detections, candidate counts and the 0.3 px agreement: same file, "Detection check" section. Sparse-vs-dense post-processing (17.011 -> 3.717, 26.744 -> 5.711): same file, "Interpretation" item 4.

| Derived value in text | Recomputation |
|---|---|
| 2.94x application penalty | 56.497 / 19.203 = 2.9421 |
| stage decomposition of the 37.294 ms gap | engine 27.788-8.153 = 19.635; I/O 13.992-2.843 = 11.149; pre 8.946-4.900 = 4.046; post 5.711-3.250 = 2.461 (sum 37.291, vs 37.294 for the difference of the two median totals) |
| 6.4 ms cross-process CPU inflation | (8.946-4.572) + (5.711-3.717) = 4.374 + 1.994 = 6.368 |
| 3.28 MB input buffer | 4 x 640 x 640 x 2 B = 3 276 800 B; 3 channels padded to 4, row alignment 64 B is already satisfied (640 x 4 x 2 = 5120 B) |
| 2.42 MB output buffers | 144 x (80^2+40^2+20^2) x 2 B = 144 x 8400 x 2 = 2 419 200 B; 144 = 9 x 16 so CHW16 adds no channel padding |
| 0.42 / 0.39 GB/s conversion rates | 3.2768 MB / 7.755 ms; 2.4192 MB / 6.237 ms |
| inverse median rates 52.08 / 17.70 im/s | 1000/19.203; 1000/56.497 |

Binding formats (Sec. IV-C): `--inputIOFormats=fp16:dla_hwc4 --outputIOFormats=fp16:chw16` from the trtexec command line in `yolo11n-dla-strict-dla-fp16.build.log`; the format descriptions ("Channel major FP16 format where channel % 4 == 0", "Sixteen wide channel vectorized row major FP16 format") from `yolo11n-dla-strict-dla-fp16.layers.json`. The claim that packing/unpacking is single-threaded CPU PyTorch is read from the implementation: `tools/yolo_e2e_tester.py`, `_pack_input` / `_unpack_output` (zero-filled padded tensor, `permute`, `contiguous`, `view`, synchronous `copy_`) and `torch.set_num_threads(args.cpu_threads)` with `cpu_threads: 1` recorded in the tester JSON.

### 1.5 Method section

Non-numeric, but checked against code rather than against the old paper: `C2DLA` / `DLABlock` / `DLAAttention` in `tools/ultralytics/ultralytics/nn/modules/block.py:1517-1639` (Eqs. 5–9, including `d = max(C/4, 8)`, the channel softmax, the 3x3 depthwise on V, the separate `proj_k`/`proj_v`, and the `1x1 -> SiLU -> 1x1` FFN with `act=False` on the second convolution); `DetectDLA` in `ultralytics/nn/modules/head.py:417-456` (packed per-scale export, training path untouched); weight transfer in `tools/ultralytics/examples/train_dla.py` (`hot_start_c2dla`, `freeze=10` phase then full fine-tune at a lower `lr0`); sparse decoder in `tools/yolo_e2e_tester.py:422` (`decode_raw_nms_sparse`, inverse-sigmoid threshold, dense mode retained as the equivalence gate).

Equation renumbering from the old paper: old (1) -> new (1); old (2) **deleted**; old (3) -> new (2); old (4) -> new (3); old (5) -> new (4); old (6)-(10) -> new (5)-(9); old (11)-(17) **deleted** with Sec. III-D.

## 2. Placeholders (`\TODO{}`) and what closes each

| # | Location | What is missing | What would fill it |
|---|---|---|---|
| 1 | `main.tex`, abstract | the RQ4 headline result | energy per frame and isolation numbers from Sec. V-E |
| 2 | Sec. IV-C | runtime-reported binding format string for the strict retrofit engine | one tester run against `yolo11n-dla-strict-dla-fp16.engine` with its `--timings-json` kept (the current `logs/fp16_matrix/` has no tester JSON for this engine) |
| 3 | Sec. V-A | module SKU/carrier, JetPack/L4T, CUDA, cuDNN, `nvpmodel` mode, `jetson_clocks` state, measured GPU/DLA/EMC clocks, thermal state | record them on the device at run time; the build logs contain none of it |
| 4 | Sec. V-A | training schedule of the reported retrofit checkpoint | the actual run configuration (see discrepancy §3.2) |
| 5 | Table II, last column + following paragraph | mAP50-95 measured from the **serialized strict DLA engine** | validator driven through the custom tester path; the stock Ultralytics TensorRT backend cannot bind CHW16 outputs or a DLA image input |
| 6 | Sec. V-D + Table IV extra row | optimized-I/O bound | CUDA or NEON packer, pinned staging buffer, conversion overlapped with execution, decoder consuming CHW16 directly |
| 7 | Table V (energy) — 18 cells | mean board power, energy/frame, idle-subtracted energy/frame, across two or more power modes | board-power logging after thermal stabilization at a fixed delivered frame rate |
| 8 | Fig. 4 + Sec. V-E | co-execution / isolation results | concurrent CUDA workload experiment and the ROS 2 replay |
| 9 | Sec. V-F | three ablations (phase-2 effect, transferred vs random init of the substituted block, CPU vs GPU host decode) | training runs and one extra tester sweep |

The abstract intentionally carries a `\TODO`: the paper's central claim (energy and GPU availability repay the latency price) is unmeasured, and the abstract must not read as if it were measured.

## 3. Discrepancies and things flagged for you

**3.1 Accuracy cannot be cross-checked against the logs.** `logs/fp16_matrix/` contains no accuracy artifact of any kind, and I found no `results.csv`, COCO eval JSON or validation log anywhere in the repository. Table II therefore rests on source (b) alone. Please confirm 65.64 / 50.25 / 55.14 / 39.40 and 64.70 / 49.03 / 53.17 / 37.68 against the actual validation runs.

**3.2 A checkpoint file name disagrees with the reported retrofit accuracy and epoch budget.** `logs/best-mAP50-95_0.37395.pt` implies mAP50-95 = 37.395, while the old paper (and now Table II) reports 37.68. Separately, `logs/yolo11n-dla-500ep.pt` implies a 500-epoch retrofit run, while the old paper states "the same 600-epoch training budget" for both models. I did not put an epoch count into the new paper; Sec. V-A carries `\TODO` #4 instead. Both files are in `logs/`, i.e. neither in source (a) nor in source (b), so I did not treat them as authoritative — but the 0.37395 vs 37.68 gap needs a decision from you.

**3.3 Input format naming: `CHW4` vs `DLA_HWC4`.** The engines were built with `--inputIOFormats=fp16:dla_hwc4`, and the build-time layer dump describes the input as the DLA image format ("channel major FP16 where channel % 4 == 0"). The old paper and `FP16_RESULTS.md` call this "CHW4" in the end-to-end discussion. In the one tester JSON that records a runtime query (a run of the model that has been removed from this paper), the TensorRT Python API reports `TensorFormat.CHW4` with 4 components on the vectorized channel dimension, i.e. the same buffer geometry under a different name. The new paper says "DLA image format" and carries `\TODO` #2; the buffer size (3.28 MB) is unaffected either way.

**3.4 The 13.992 ms figure covers both directions, not just the CHW16 outputs.** Your task text describes it as "13.992 ms for ~2.42 MB of FP16". Recomputed from `FP16_RESULTS.md`: 13.992 = 7.755 (input packing + H2D, 3.28 MB) + 6.237 (D2H + output unpacking, 2.42 MB). The paper reports the split and both buffer sizes.

**3.5 The cross-process anomaly is larger than preprocessing alone.** Preprocessing is 4.900/4.572 ms in the two GPU rows and 8.617/8.946 ms in the two DLA rows (+4.374 ms for the retrofit pair), as you noted. Post-processing shows the same pattern: 3.717 ms (retrofit on GPU) vs 5.711 ms (retrofit on strict DLA), +1.994 ms, for identical single-threaded CPU code on the identical 13 surviving anchors. Total unexplained CPU-side inflation in the DLA-row processes: 6.368 ms, i.e. about 17 % of the 37.3 ms application-level gap. Reported in Sec. V-D as an unexplained cross-process measurement difference, with no thermal/clock/scheduling attribution.

**3.6 "Execution-context device memory" is really the build-reported activation memory.** The 0 / 9.47 / 22.80 MiB values are `Total Activation Memory` lines in the build logs, not a runtime allocator query. Wording in Sec. V-C reflects that.

**3.7 Old-paper arithmetic I could reproduce exactly** (so nothing to flag, listed for completeness): 20.7 % and 27.6 % strict-vs-fallback, 94.0 % DLA share (93.97 % recomputed), 21.1 % engine-stage reduction (21.05 %), 7.3 %/3.3 % GPU-control comparison, 6.86/6.52 MiB engine sizes, 9.47/8.52/22.80 MiB memory, 52.08/18.95/50.43/17.70 im/s, 4 loadables / 11 GPU kernels / 22 reformat nodes, 13 of 8400 anchors. **Old-paper figures I could not verify:** the accuracy row values (§3.1) and the 600-epoch budget (§3.2). Nothing else in the old paper's numeric claims failed to reproduce.

**3.8 AI-use acknowledgment.** Rewritten, not copied: the old wording names a specific commercial tool, which is an identifying detail in a double-anonymous submission and also dates the manuscript. The new version says "a large-language-model coding assistant", states what it was used for (implementation inspection, drafting and editing prose in all sections) and that the authors verified everything and are responsible. **Please confirm against the current IEEE/RAS and ICRA 2027 policy**: whether such a disclosure is required, permitted in the acknowledgment of an anonymous submission (acknowledgments are sometimes required to be omitted until camera-ready), or must go in a separate statement. I could not verify the 2027 policy text from the repository, and I have not checked it online.

Naming note: the bibliography file is `references.bib`, not `refs.bib`, so that `\bibliography{references}` matches the file already tracked in this directory; it holds only cited entries. One orphan was dropped from the carried-over file: `ding2021repvgg` (RepVGG), cited only by the removed constraint-native path. All 17 remaining entries are cited.

## 4. Page count and what I would cut next

**Current: 9 pages (limit is 8, references included).** `Output written on main.xdv (9 pages, 127424 bytes)`. Compiles clean: no undefined references, no undefined citations, no overfull hbox of any size. The only warnings are 14 underfull hboxes (all inside the `p{}` columns of Table I and the two-line table headers) and six font-shape warnings emitted by IEEEtran itself at class-load time (`TU/ptm` before `fontenc` takes effect, and one small-caps-italic shape in the bibliography); body text is `T1/ptm`, i.e. Times, as required.

Section starts, from `main.aux`: I p1, II p2, III p2, IV p4, V p5, VI p8, references spill to p9. Rendered page 9 holds exactly: three lines of the conclusion, the seven-line acknowledgment, and the 17-entry bibliography (15 entries in column 1, 2 in column 2), i.e. about **1.15 columns of content that pages 1-8 have to absorb**.

I have **not** cut anything. Ranked list of what I would cut, with rough savings (lines of a single column):

1. Sec. V-C, "Two secondary observations" paragraph: keep the GPU-control sentence, drop the build-time/engine-size/memory sentence (it is already in this file and adds nothing to the argument). **~7 lines.**
2. Intro, "Two intermediate positions deserve to be closed off early": compress to three sentences; the detail is repeated in V-C and in the conclusion. **~10 lines.**
3. Table I: merge the four low-information inherited rows (positional encoding, activations, box representation, assignment/losses) into one "everything else, including the $r=16$ head and its assignment" row. **~6 lines**, and it removes most of the underfull-hbox warnings.
4. Sec. IV-D, the paragraph after the gate list: compress the gate-2/gate-3 discussion to two sentences. **~8 lines.**
5. Sec. V-A, "Deployment protocol": drop what Sec. IV already states (build modes, FP16, batch, resolution) and keep only the measurement protocol. **~7 lines.**
6. Conclusion: it currently restates every headline number; keep the price and the open question, drop the restated decomposition. **~8 lines.**
7. Sec. III-A: drop the transferability sentence about NPUs/Edge TPUs/FPGAs (the point is made in the contributions and in Sec. IV-D). **~5 lines.**
8. Sec. II, paragraph 1: drop the closing "search is the natural follow-up" sentence. **~4 lines.**
9. Fig. 4 placeholder box: 3.1 cm -> 2.0 cm. It is a placeholder, so this costs nothing now, but it will have to come back when the real figure lands. **~8 lines.** I list it last deliberately: it borrows space from a future result rather than removing text.

10. Drop Fig. 2 (the C2DLA block) and keep only Eqs. (5)-(9), which specify the block completely, moving the caption's content into two sentences of III-C2. **~20 lines**, the largest single saving available, at the cost of the only picture of the substituted block. It is last because you asked for exactly four figures, so this is a scope decision rather than an editing one.

Items 1-8 total roughly 55 lines, i.e. about one column, which is close to but probably just short of the 1.15 columns needed; items 1-8 plus either 9 or 10 gives margin. What I would **not** do without being told: touch `\setlength`/margins/font sizes, shorten the bibliography, drop Fig. 3 (the signature figure), or remove any of the honesty paragraphs in Sec. V-D and V-F.
