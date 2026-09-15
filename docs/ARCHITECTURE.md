# Model: YOLO11-DLA-n

## Where the code lives

YOLO11-DLA is a patch to Ultralytics, not a separate codebase. The applied result is the companion repository at <https://anonymous.4open.science/r/ultralytics-dla>, which the tooling expects at `tools/ultralytics/`. [`MODEL_PATCH.md`](MODEL_PATCH.md) contains every change required to reproduce it from upstream revision `a462bb65` — the three new blocks, the head, the model YAML, the host decoder, and the export, backend and post-processing hooks — with the parts belonging to other experiments explicitly excluded.

| Upstream file | Change |
|---|---|
| `ultralytics/nn/modules/block.py` | add `DLAAttention`, `DLABlock`, `C2DLA` |
| `ultralytics/nn/modules/head.py` | add `DetectDLA` |
| `ultralytics/cfg/models/11/yolo11-dla.yaml` | new: stock YOLO11 with layer 10 and the head substituted |
| `ultralytics/utils/dla.py` | new: host-side decoder for the raw maps |
| `ultralytics/nn/tasks.py`, `nn/modules/__init__.py` | register and export the new modules |
| `ultralytics/engine/exporter.py`, `nn/backends/base.py` | binding names and typed export metadata |
| `ultralytics/models/yolo/detect/{predict,val}.py`, `utils/nms.py` | route raw outputs through the decoder; fix a six-candidate NMS ambiguity |

Cloning that repository to `tools/ultralytics/`, or applying the patch to a plain upstream checkout there, and putting the path on `PYTHONPATH` is all the experiment tooling needs.

## Why stock YOLO11n cannot be placed on DLA

With `--useDLACore=0` and fallback disabled, TensorRT 10.3.0 rejects the stock graph at the `C2PSA` attention block: `MATRIX_MULTIPLY` is unsupported on DLA, and the spatial-affinity path also needs head transposes and a softmax over image positions. The preserved failure log is `logs/fp16_matrix/yolo11n-strict-dla-fp16.build.log`.

The decoded prediction interface is the second obstacle. The stock `Detect` head flattens and concatenates across scales, takes the DFL expectation, decodes against an anchor grid and applies sigmoid, producing a `1×84×8400` tensor. Those operations live on tensors with fewer than four dimensions and use slicing, division and non-NCHW broadcasts, none of which DLA accepts.

Allowing fallback does not make the problem disappear: it partitions the graph. The stock FP16 fallback engine contains four DLA loadables with eleven GPU compute nodes, twenty-two reformats and ten no-op/constant nodes between them (`logs/fp16_matrix/yolo11n-dla-gpu-fallback-fp16.layers.json`).

## C2DLA

`C2DLA` keeps `C2PSA`'s split–process–merge wrapper: a 1×1 projection produces two channel groups, one bypasses the inner block, the other is processed, and the two are concatenated and projected. Only the inner block changes.

For the processed group `X` with `C` channels and bottleneck `d = max(⌊C/4⌋, 8)`:

```
K = φ_k(X)                       1×1 projection, C → d
V = φ_v(X)                       1×1 projection, C → C
A = softmax_over_channels(K)     normalization along the d channel axis, per position
Z = φ_o( φ_a(A) + ψ_3×3(V) )     φ_a: d → C bias-free 1×1, ψ: 3×3 depthwise, φ_o: C → C
X'  = X  + Z
X'' = X' + FFN(X')               FFN: C → 2C, SiLU, 2C → C
```

Batch normalization accompanies the relevant convolutions and folds into them at inference; `φ_a` is bias-free and carries no batch normalization. In the nano configuration the enclosing block sees 256 channels, so `C = 128` and `d = 32`, and one inner block runs at the 20×20 scale.

Every operation stays four-dimensional `(N, C, H, W)`. There is no spatial affinity matrix, no `MatMul`, no head transpose and no normalization over image positions. The softmax that remains normalizes across a channel axis at each spatial position, which TensorRT does place on DLA for this target and shape — with a documented approximation warning, quoted in [`BUILD.md`](BUILD.md).

Key and value projections are deliberately separate convolutions rather than one fused QKV projection, to avoid introducing a fused-extraction subgraph in the replacement branch. Static channel splits elsewhere in the retained graph compile without trouble, so the design does not rest on a claim that splits or SiLU are unsupported.

This is a hardware-aware substitute, not an equivalence. It mixes channels and adds local 3×3 context; it does not exchange information between distant positions, it is not algebraically equivalent to `C2PSA`, and no greater representational power is claimed. The accuracy consequence is measured, not argued.

## DetectDLA

The backbone, `C3k2` stages, SPPF, feature-pyramid connections and the regression/classification towers of YOLO11n are all retained. `DetectDLA` changes only what crosses the accelerator boundary. For each pyramid level ℓ ∈ {3,4,5} it concatenates the tower outputs along the channel axis and exports them raw:

```
P_ℓ = Concat_C(R_ℓ, S_ℓ) ∈ R^(B × (4r + n_c) × H_ℓ × W_ℓ),   r = 16, n_c = 80
```

At batch one and 640×640 the engine bindings are `pred_p3` = `1×144×80×80`, `pred_p4` = `1×144×40×40` and `pred_p5` = `1×144×20×20`, at strides 8, 16 and 32. Classification values are logits; the box channels are unnormalized DFL distributions, not decoded coordinates.

Export therefore stops before flattening, cross-scale concatenation, the DFL expectation, grid decoding and sigmoid. Confidence filtering and variable-cardinality NMS stay on the host by construction.

During training and ordinary PyTorch inference `DetectDLA` calls the inherited `Detect` path, so losses, assignment and validation behaviour are unchanged. This is a deployment-partitioning change, not a new prediction formulation, and it is not a direct-regression or one-to-one (NMS-free) head.

The boundary has a cost that must be stated: at FP16 the three raw maps carry 2,419,200 bytes against 1,411,200 bytes for the stock decoded output, a factor of 1.714. Lower engine-plus-transfer latency therefore does not by itself imply a faster detector pipeline.

## Host-side decoding

The host splits each `P_ℓ` into 64 regression and 80 class channels. An anchor survives when `σ(max_c S_ℓ,c(h,w)) > τ`; sigmoid is applied **before** the comparison so that the sparse path reproduces the dense path's floating-point threshold behaviour exactly. Regression decoding is independent of class confidence and is deferred until after rejection:

```
d̂_q = Σ_{j=0}^{r-1} j · softmax(R_q)_j ,   q ∈ {l, t, r, b}
```

Cell centres sit at `(w + 0.5, h + 0.5)`; the four distances form the box, the box is scaled by the level stride, class sigmoid is applied, class-aware NMS runs, and coordinates are mapped back through the letterbox transform.

If `m` of `N = 8400` anchors survive, dense class scanning stays `O(N·n_c)` while the DFL softmax/expectation work drops from `O(4rN)` to `O(4rm)`. It does not reduce transfer volume. In the FP16 strict 10 Hz test the number of surviving anchors ranges from 0 to 254, median 28.5 over the 100-image cycle.

Dense and sparse decoding are verified to agree: on all 5,000 FP16 images the AP difference is below 10⁻⁷ points, and on the 4,500 held-out INT8 images the AP is identical at reported precision. They are **not** bitwise identical — one GPU image (74092) yields 252 sparse against 251 dense detections in FP16, and one (408774) yields 295 against 294 in INT8.

Two bugs found while preparing the harness were reproduced and fixed, and they affect how older numbers must be read:

1. In `ultralytics/utils/nms.py`, an ordinary `(1, 84, 6)` tensor with exactly six candidates was misread as end-to-end detections `(B, N, 6)`. The explicit class count now disambiguates the YOLO11 case.
2. In the end-to-end tester, the sparse filter on raw logits could diverge from the dense filter near the threshold through rounding. Sigmoid is now applied to each anchor's maximum logit before the comparison.

Measurements taken before these fixes were not reused for the sparse path; the records predating them are archived under `tools/experiments/history/runtime/`.

## Initialization and training

The nano model is initialized from pretrained `yolo11n.pt`. The loader intersects source and destination state dictionaries by tensor name **and** shape, copies the matching entries and loads them non-strictly (`BaseModel.load` → `intersect_dicts` in the fork). Everything compatible — backbone, neck, detection towers, and the parts of the replacement block that happen to match — is transferred; the rest keeps its model initialization. No QKV remapping and no frozen-backbone stage are involved, and no specialized two-phase recipe was used.

Recorded training configuration, extracted from the checkpoint itself into [`../tools/paper/analysis/checkpoint_training.json`](../tools/paper/analysis/checkpoint_training.json):

- COCO train2017, 500 epochs, batch 32, 640×640, seed 0, 8 workers, AMP, `freeze=None`, `nbs=64`, `optimizer=auto`.
- Base LR 0.01, `lrf=0.01`, momentum 0.9, weight decay 5×10⁻⁴, three warm-up epochs then linear decay; mosaic closed for the last ten epochs.
- HSV h/s/v 0.015/0.7/0.4, translate 0.1, scale 0.5, fliplr 0.5, mosaic 1.0; MixUp and copy-paste disabled.
- 2,615,696 unfused parameters. Checkpoint-recorded validation history ends at AP 37.581 and AP₅₀ 53.151.

The optimizer identity is a **reconstruction**, not a stored optimizer state. Inspecting the archived Ultralytics revision recorded in the checkpoint (`3f2711b10e21eedf4e2efaa907c1322d844ea2d7`): the trainer computes expected iterations as `ceil(len(dataset) / max(batch_size, nbs)) · epochs`, and above 10,000 iterations `optimizer=auto` selects MuSGD with LR 0.01 and momentum 0.9, overriding the nominal 0.937 in the arguments; selected classification-head parameter groups get a 3× LR multiplier. The eight recorded `lr/pg*` series are consistent with that reading (epoch 5: 0.0099208 and 0.0297624; final epoch: 0.0001198 and 0.0003594).

Two consequences for interpretation. First, the checkpoint's 37.581 AP comes from the training-time validation protocol and is not comparable with the deployed-engine AP in the paper. Second, the stock and adapted models were not trained on equal budgets, so any stock-vs-adapted difference mixes architecture, training and — for INT8 — calibration; the adapted GPU engine, not the stock model, is the controlled comparison for placement.
