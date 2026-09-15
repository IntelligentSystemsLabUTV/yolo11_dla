# Export, TensorRT build and placement verification

## ONNX export

Both models are exported with the same settings: opset 20, static `1×3×640×640` input, batch one, no dynamic axes, no ONNX simplification, no embedded NMS, on CPU.

```bash
# fill in a hardware-notes file first, starting from configs/hardware-notes.example.md
python tools/experiments/run.py prepare \
  --stock /path/to/yolo11n.pt \
  --adapted weights/yolo11-dla-n.pt \
  --hardware-notes tools/experiments/configs/hardware-notes.md \
  --output NEW_DIRECTORY
```

`prepare` copies each checkpoint, asserts that layer 10 and the head are the expected types (`C2PSA`/`Detect` for stock, `C2DLA`/`DetectDLA` for adapted), asserts COCO detection with `reg_max = 16` and the nano width at layer 10, exports, runs the ONNX checker, and asserts the output shapes — `[[1,144,80,80],[1,144,40,40],[1,144,20,20]]` for the adapted model and `[[1,84,8400]]` for stock. It records SHA-256 for every checkpoint and every export.

The artifacts every engine in the paper was built from are identified by the **as-built** hashes below, which are the values recorded in the calibration manifests, the build commands and the verification record.

| Artifact | As built, SHA-256 | Distributed copy |
|---|---|---|
| stock export `logs/yolo11n.onnx` | `a771561a8a897283e0454caa7951a23de3494169b65ec7842094e021aec68f17` | not distributed |
| adapted export | `2712fe1182428d45a3dbf4c00de596507a79333dcd1f4c9d0791c61d7c29fcca` | `weights/yolo11-dla-n.onnx`, `77b9106ec556277354e953a01a7b43b50b2435c4a5b314dec9d9b1783b8ba1d4` |
| adapted checkpoint | `417967ac4a0af77f4a4e1f1f31ba4e31b38775853fa7393f0f8824f2a3511535` | `weights/yolo11-dla-n.pt`, `440c27347dbf6c84ca8ae9609588769ab8809270bd4a0e36af70760aeb50d683` |

The two files under `weights/` are the as-built artifacts with the local filesystem paths and repository URL recorded inside them replaced by `<redacted>`, which is why their file hashes differ. Nothing else was changed, and that is checkable: the ONNX graph is bit-identical (`sha256(graph.SerializeToString())` = `c15763f5fa8e56fc0d4cc71424dc3c50…` before and after), every ONNX metadata key is preserved, and the checkpoint's parameters are bit-identical (`sha256` over the sorted `state_dict` tensor bytes = `94e3df7af25b7d803c59255c174b1c4d…` before and after). Compare those two invariants, not the file hashes, when checking a copy against the record.

The adapted ONNX graph contains 89 `Conv`, 76 `Sigmoid`, 76 `Mul`, 9 `Split`, 14 `Add`, 20 `Concat`, 3 `MaxPool`, 2 `Resize` and one `Softmax`. It contains **no** `MatMul`, `Gemm`, `Reshape`, `Transpose`, `Div` or `Slice`. The Sigmoid/Mul pairs are the retained SiLU activations; these are pre-fusion node counts, not execution kernels. Source: [`../analysis/model_verification.json`](../analysis/model_verification.json), regenerable with `python tools/experiments/run.py check-model`.

The same record holds the deterministic CPU checks on one random input (seed 7): the external host decoder reproduces the inherited `Detect` output to a maximum absolute difference of 1.22×10⁻⁴, and ONNX Runtime reproduces the checkpoint's raw maps to 3.07×10⁻⁴. These are CPU export checks; they say nothing about TensorRT or DLA numerics.

## TensorRT builds

Eight engines back the paper: four configurations × two precision modes. Commands are reproduced from the build manifests, with the recorded absolute paths shortened to repository-relative ones. All builds use `--skipInference` because timing is collected separately.

### FP16

```bash
# stock YOLO11n, GPU
trtexec --onnx=logs/yolo11n.onnx --saveEngine=ENG/yolo11n-gpu-fp16.engine \
        --fp16 --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw \
        --profilingVerbosity=detailed --dumpLayerInfo --exportLayerInfo=ENG/yolo11n-gpu-fp16.layers.json --skipInference

# stock YOLO11n, DLA with GPU fallback
trtexec --onnx=logs/yolo11n.onnx --saveEngine=ENG/yolo11n-dla-gpu-fallback-fp16.engine \
        --useDLACore=0 --allowGPUFallback --fp16 \
        --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw \
        --profilingVerbosity=detailed --dumpLayerInfo --exportLayerInfo=... --skipInference --verbose

# YOLO11-DLA-n, GPU  (same-model placement control)
trtexec --onnx=weights/yolo11-dla-n.onnx --saveEngine=ENG/yolo11n-dla-gpu-fp16.engine \
        --fp16 --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw \
        --profilingVerbosity=detailed --dumpLayerInfo --exportLayerInfo=... --skipInference

# YOLO11-DLA-n, strict DLA  (no --allowGPUFallback, native I/O formats)
trtexec --onnx=weights/yolo11-dla-n.onnx --saveEngine=ENG/yolo11n-dla-strict-dla-fp16.engine \
        --useDLACore=0 --fp16 \
        --inputIOFormats=fp16:dla_hwc4 --outputIOFormats=fp16:chw16 \
        --profilingVerbosity=detailed --dumpLayerInfo --exportLayerInfo=... --skipInference --verbose

# stock YOLO11n, strict DLA  --  FAILS, and the failure is the point
trtexec --onnx=logs/yolo11n.onnx --saveEngine=ENG/yolo11n-strict-dla-fp16.engine \
        --useDLACore=0 --fp16 \
        --inputIOFormats=fp16:dla_hwc4 --outputIOFormats=fp16:chw16 ... --verbose
```

### INT8

INT8 builds enable both `--int8` and `--fp16` and supply a per-model calibration cache. The strict engine is the only one that also changes its external binding formats to INT8.

```bash
# stock YOLO11n, GPU
trtexec --onnx=logs/yolo11n.onnx --saveEngine=... --int8 --fp16 \
        --calib=logs/int8_calibration/stock/calibration.cache \
        --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw ... --skipInference --verbose

# stock YOLO11n, DLA with GPU fallback
trtexec --onnx=logs/yolo11n.onnx --saveEngine=... --useDLACore=0 --allowGPUFallback --int8 --fp16 \
        --calib=logs/int8_calibration/stock/calibration.cache \
        --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw ... --skipInference --verbose

# YOLO11-DLA-n, GPU
trtexec --onnx=weights/yolo11-dla-n.onnx --saveEngine=... --int8 --fp16 \
        --calib=logs/int8_calibration/adapted/calibration.cache \
        --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw ... --skipInference --verbose

# YOLO11-DLA-n, strict DLA, native INT8 bindings
trtexec --onnx=weights/yolo11-dla-n.onnx --saveEngine=... --useDLACore=0 --int8 --fp16 \
        --calib=logs/int8_calibration/adapted/calibration.cache \
        --inputIOFormats=int8:dla_hwc4 --outputIOFormats=int8:chw32 ... --skipInference --verbose
```

For a new campaign the wrapper `python tools/experiments/run.py int8-build` issues these commands with the same flags; it has no resume and must not be re-run over an existing matrix.

## Placement verification

Compilation success is necessary but not sufficient, and the verbose placement report is not the final word: the optimized-engine inspector is. Acceptance requires, for each strict engine, **exactly one DLA node, zero GPU nodes and zero reformat nodes** — one DLA loadable.

| Configuration | Mode | DLA nodes | Other optimized nodes | Verdict |
|---|---|---|---|---|
| 11n / GPU | FP16 | 0 | — | GPU control |
| 11n / DLA+GPU | FP16 | 4 | 11 GPU compute, 22 reformat, 10 no-op/constant | partitioned |
| 11-DLA / GPU | FP16 | 0 | — | GPU control |
| **11-DLA / DLA** | FP16 | **1** | **0** | **fallback-free** |
| 11n / strict DLA | FP16 | — | — | **build rejected at attention MatMul** |
| 11n / GPU | INT8 | 0 | 160 | GPU control |
| 11n / DLA+GPU | INT8 | 4 | 34, of which 23 reformat | partitioned |
| 11-DLA / GPU | INT8 | 0 | 144 | GPU control |
| **11-DLA / DLA** | INT8 | **1** | **0** | **fallback-free** |

The FP16 strict build assigns 299 layers to DLA and none to GPU. The INT8 strict build assigns 298 layers to INT8 and one softmax to FP16 — all of them on DLA. Fallback-free is therefore not the same as integer-only.

Evidence: `logs/fp16_matrix/*.layers.json` and `logs/int8_matrix/**/*.layers.json`, mirrored under uniform names in `logs/icra2027_final/existing/engines/` and `logs/int8_ready/`. The `one_dla_no_gpu_or_reformat` flag in each manifest is the machine-readable version of the table above, and `logs/icra2027_final/existing/summary-build/results.csv` is its FP16 summary.

Two cautions carried into the paper. First, this is configuration-specific: TensorRT's DLA softmax support depends on target and shape, and the compiler warns that the selected approximation can introduce numerical error. A source rewrite that compiles on this Orin/TensorRT pair is not evidence for another DLA generation, input size or precision. Second, a successful build does not validate the decoding of the raw outputs — that is checked separately, on the board, by `parity`.

The FP16 separate layer profile (a distinct `trtexec` run, not the timing run) attributes 93.97% of stock fallback time to DLA, 4.73% to reformats and 1.30% to GPU compute, summing to 34.312 ms of device profile with 32.244 ms inside DLA segments. It is FP16-only and is plotted in the FP16 placement profile; the per-layer table is [`../analysis/layer_profile.csv`](../analysis/layer_profile.csv).

## INT8 calibration

Post-training entropy calibration, no labels, no additional training.

- 500 COCO val2017 images sampled without replacement with seed 2027, leaving **4,500 disjoint images** for accuracy. The split is `logs/int8_ready/split.json`; the 4,500 held-out IDs are also embedded in `logs/int8_ready/manifest.json` under `evaluation_protocol.image_ids`.
- The two models get **separate** caches, generated before fusion with TensorRT's `IInt8EntropyCalibrator2`, batch one, on FP32 normalized RGB letterboxed inputs, from the same image IDs.
- Each model's cache is then shared between its GPU and its DLA placement.
- `logs/int8_calibration/{stock,adapted}/manifest.json` record the ONNX hash, the split hash, the preprocessing hash, the per-image list with hashes, the algorithm, the batch and the resulting cache hash.

```bash
python tools/experiments/run.py int8-split    --annotations "$COCO_ANN" --output "$INT8_SPLIT"   # 500 / 4500, seed 2027
python tools/experiments/run.py int8-calibrate --onnx weights/yolo11-dla-n.onnx --model adapted \
       --split "$INT8_SPLIT" --images "$COCO_VAL" --output logs/int8_calibration/adapted
```

The `calibration-gpu.engine` produced while generating a cache is a preparation artifact with FP32 bindings, not one of the paper's configurations.

## Native I/O layouts and the host boundary

The strict engines are the only ones with non-linear external bindings. GPU and stock fallback engines keep linear FP16 bindings in both precision modes.

| | FP16 strict | INT8 strict |
|---|---|---|
| Input format | `fp16:dla_hwc4` | `int8:dla_hwc4` |
| Input bytes | 4 × 640 × 640 × 2 = 3,276,800 | 1,638,400 |
| Output format | `fp16:chw16` | `int8:chw32` |
| Output bytes | 2,419,200 (144 divisible by 16, no channel padding) | 1,344,000 (144 padded to 160) |

`dla_hwc4` pads the three input channels to four in interleaved storage; both input layouts satisfy the documented row alignment. For the FP16 engine TensorRT reports CHW4 input and CHW16 outputs, and the padded-input interpretation is validated for this shape only, not for arbitrary CHW4 bindings.

For native INT8 bindings the host quantizes normalized pixels as `q = clip(round(x/s), -128, 127)` with ties-to-even, and dequantizes raw outputs as `y = s·q`. The per-tensor scales are exact float32 values read from the calibration cache and cross-checked against the explicit build-time I/O ranges in the successful build log; they are stored in `logs/int8_ready/adapted-strict.engine.quantization.json`, bound to the engine hash:

| Binding | Scale | Logged range |
|---|---|---|
| `images` | 0.007568359375 | ±0.961182 |
| `pred_p3` | 0.34104010462760925 | ±43.3121 |
| `pred_p4` | 0.38649243116378784 | ±49.0845 |
| `pred_p5` | 0.4072129428386688 | ±51.7160 |

The runtime refuses to load an INT8 engine without valid scales. The scales are not tuned against AP: `trtexec` 10.3 assigns the cache ranges to the INT8 bindings explicitly, and the printed ranges are rounded — they are for checking, not for computing the scales.

Changing precision therefore changes both the arithmetic and the host I/O path, not just the serialized engine. The reference runner does packing, unpacking and INT8 conversion on the CPU; the timed input stage includes packing and H2D, and the timed output stage includes D2H, unpacking and any dequantization. These are not pure layout-conversion timings. A CUDA stream and device-visible buffers are still needed even with no GPU-assigned learned layers, which is one reason strict placement does not mean the GPU is idle.

Because logical shape alone is insufficient, the runtime records format, vectorized dimension, component count and context strides for every binding, and `parity` verifies pack and unpack separately against a scalar-offset oracle with patterns that vary by channel, row and column, including the padding.

## Registering an existing engine set

Neither campaign rebuilds engines. `reuse-existing` (FP16) and `int8-reuse` (INT8) register an existing build directory into a uniform layout with relative links, manifests and hashes, without copying engines, deserializing them or invoking TensorRT.

```bash
python tools/experiments/run.py reuse-existing --source logs/fp16_matrix --output logs/icra2027_final/existing

python tools/experiments/run.py int8-reuse \
  --matrix logs/int8_matrix --calibration logs/int8_calibration --split "$INT8_SPLIT" \
  --stock-onnx logs/yolo11n.onnx --adapted-onnx weights/yolo11-dla-n.onnx \
  --output logs/int8_ready
```

`int8-reuse` additionally finds exactly one successful build per configuration, verifies caches, ONNX files, the disjoint split, the recorded commands and the inspector output, extracts the exact INT8 scales, and writes the quantization sidecar. It does not modify the original `logs/int8_matrix/manifest.json`, which still documents the first failed build attempt.

Every later stage verifies engine hashes before it measures anything, so the registration is what ties the tables back to these files.
