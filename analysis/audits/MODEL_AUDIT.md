# YOLO11-DLA implementation and artifact audit

Audited 2026-09-08. This is a private preparation document: checkpoint metadata contains identifying paths and repository details. It is not part of the anonymous submission.

Update 2026-09-10: the author confirms compatible-weight initialization from
YOLO11n. The repository and checkpoint revision establish the partial loading
path and reconstruct automatic optimizer selection as MuSGD. See
[TRAINING_RECONSTRUCTION.md](../../docs/ARCHITECTURE.md).
The historical open points below are superseded for initialization source and
the described optimizer recipe, not for missing historical file hashes.
The six deployed-engine COCO evaluations are now complete; see
[RUN_01_AUDIT.md](../../docs/PROVENANCE.md).

## What the implemented detector actually changes

- `../ultralytics/ultralytics/cfg/models/11/yolo11-dla.yaml`: YOLO11 structure and nano scaling retained; YAML layer 10 substitutes C2DLA, and layer 23 substitutes DetectDLA. This study evaluates object detection, not the separate segmentation configuration or independent YOLO-DLA detector.
- `../ultralytics/ultralytics/nn/modules/block.py:1517`: DLAAttention computes `proj_out(proj_attn(softmax_channel(proj_k(x))) + pe(proj_v(x)))`. `pe` is **3×3 depthwise**, not 5×5. The two input projections are separate. Channel softmax is input-dependent, with no global token interaction. It is inspired by projection-based attention but is not algebraically equivalent to C2PSA or the complete external-attention paper.
- `block.py:1570` and `block.py:1598`: DLABlock preserves two residuals and the pointwise FFN; C2DLA preserves split/process/concatenate/project. Nano uses input256, processed128, bottleneck32, one inner block.
- `../ultralytics/ultralytics/nn/modules/head.py:417`: DetectDLA invokes ordinary Detect except during export. Export retains regression and class towers, emitting three raw rank-four maps. No replacement with direct boxes, no one-to-one prediction head, and no loss redesign.
- `../ultralytics/ultralytics/engine/exporter.py:621`: ONNX exporter handles raw output names and metadata. The current ONNX artifact is opset20, input `(1,3,640,640)`, outputs `pred_p3/p4/p5` of `(1,144,80,80)`, `(1,144,40,40)`, `(1,144,20,20)`.
- `../ultralytics/ultralytics/utils/dla.py`: external DFL decoding is implemented and accepts CPU execution. Existing comments saying post-processing must use GPU are stale.
- `../yolo_e2e_tester.py:166`: the existing TensorRTEngine runner handles native formats, records context strides, and packs/unpacks logical tensors. Its detection postprocessor supports dense and confidence-first sparse DFL. Full native-layout correctness still needs target validation.

## Checkpoint-backed facts

`../../logs/yolo11n-dla-500ep.pt`:

- SHA256 `417967ac4a0af77f4a4e1f1f31ba4e31b38775853fa7393f0f8824f2a3511535`.
- 2,615,696 unfused parameters.
- Embedded history: **500 epochs**, highest stored AP at epoch500, AP=0.37581, AP50=0.53151, precision=0.65034, recall=0.48557.
- The stripped checkpoint's scalar `epoch` is -1. Epoch500 is established by the **history**, not by that scalar or filename.
- Stored arguments: image640, batch32, seed0, epochs500, optimizer `auto`, pretrained requested, AMP enabled, freeze=None, rect=False, dataset `coco-bbox.yaml`, validation IoU0.7, max_det300.
- Stored package version8.4.48 and source commit `3f2711b10e21eedf4e2efaa907c1322d844ea2d7`; these do not imply the current working tree is identical. Exact source and training log still need recovery.
- `analysis/extract_checkpoint.py` regenerates the private JSON/CSV. The metrics are **training validation**, not independent validation or DLA AP.

The other checkpoint `../../logs/best-mAP50-95_0.37395.pt` has metadata inconsistent with its filename: the recovered record corresponds to epoch21 and AP0.31981. Do not infer metrics from filenames.

The previous paper's 37.68/39.40 AP pair and 95.6% retention are not established by the supplied primary artifacts. They were removed from the revised paper. A clean paired validation can restore a measured retention statement with the actual values.

## Training recipe distinction

`../train_coco_dla.py:194` performs one full fine-tuning run after partial compatible-weight loading. `../ultralytics/examples/train_dla.py` additionally implements an optional block-aware hot start and freeze10/fine-tune two-phase recipe. The 500-epoch checkpoint has no evidence that the latter recipe was used. Its initial source checkpoint, realized optimizer, and training logs remain P2; do not attribute accuracy to optional code paths merely because they exist.

## Checks completed locally

`analysis/verify_model.py` passed on the trained checkpoint with deterministic uniform-random input, seed7, shape `(1,3,640,640)`:

- External dense PyTorch decoder versus the same adapted model's ordinary inference: max absolute error0.0001220703125, mean5.44855396e-8; elementwise `rtol=1e-5, atol=1e-4` passed.
- ONNX structural validation and static outputs passed; 89Conv,76Sigmoid,76Mul,9Split,14Add,20Concat,3MaxPool,1Softmax,2Resize; no MatMul/Gemm/Div/Reshape/Transpose/Slice.
- ONNX Runtime CPU raw outputs versus checkpoint: maximum errors P3=0.000207901, P4=0.000307083, P5=0.0000686646; `rtol=1e-4, atol=1e-3` passed.
- ONNX SHA256 `2712fe1182428d45a3dbf4c00de596507a79333dcd1f4c9d0791c61d7c29fcca`.

The machine-readable record is `analysis/model_verification.json`. One deterministic CPU input supports interface checks, not dataset AP, sparse-NMS parity, or TensorRT engine equivalence. Existing build logs refer to another ONNX pathname and do not carry its hash, so fresh hash-linked exports/builds remain necessary.

## Deployment paths to avoid confusing

1. The generic Ultralytics TensorRT exporter enables `GPU_FALLBACK` in `utils/export/engine.py:180`. An ordinary successful `yolo export ... device=dla:0` is not strict certification. Use explicit trtexec without `--allowGPUFallback`.
2. The generic TensorRT backend allocates from logical shapes and cannot be assumed to support the strict engine's native buffers. Do not use a one-line `yolo val model=strict.engine` as the final validation without fixing and verifying that backend.
3. DetectionValidator's raw-output metadata lookup uses `getattr(self, 'model', None)` while BaseValidator constructs its backend in a local `model`; the raw ONNX/generic backend path needs attention too. The new `analysis/evaluate_coco.py` bypasses this path and uses the existing format-aware runner plus explicit dense decoding and multi-label NMS.
4. `examples/test_dla_engine.py` is a segmentation-oriented example whose main path requires a proto tensor; `--nm=0` alone does not turn it into this detector's validator.
5. Strict TensorRT DLA execution is not DLA standalone runtime operation and is not proof of zero CUDA activity, zero shared DRAM use, or system-wide GPU isolation.

`analysis/evaluate_coco.py` was smoke-tested end to end on CPU using the real checkpoint and a deliberately synthetic one-image annotation fixture. That test exercised predictions, category mapping, JSON output, and COCOeval plumbing. Its artificial AP is not a research result. Its TensorRT branch has **not** been executed here; P3 remains open.
