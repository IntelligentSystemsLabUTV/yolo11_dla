# DLA-YOLO

Design record for the `yolo-dla` model family: a one-stage detector whose entire deployable graph compiles to a single NVIDIA DLA loadable on Jetson Orin, with no GPU fallback layers.

This document explains how the network is constructed, why each component has the shape it has, why the result is still a YOLO rather than something else wearing the name, and what the architecture sweeps measured. Every number here was produced from the checked-in configuration; the provenance of each is marked as **measured** (run on hardware or on the exported graph), **derived** (arithmetic from a vendor spec sheet), or **pending**.

Related documents: [DLA-YOLO11N.md](../docs/DLA-YOLO11N.md) covers the earlier `yolo11-dla` conversion (C2DLA/DLAAttention on the YOLO11 skeleton), which remains the primary experimental baseline. [tools/ultralytics/YOLO-DLA.md](ultralytics/YOLO-DLA.md) is the short user-facing README for this family. [tools/YOLO-DLA_ICRA/] (the YOLO-DLA manuscript, kept in its own repository) is the paper draft, which currently describes the `yolo11-dla` design under the name "YOLO-DLA" and needs reconciling with the naming used here.

---

## 1. Status

The architecture below is **v0**, the configuration currently in training on COCO 2017 from scratch (no pretrained initialization exists for this family). Compliance and latency are confirmed on hardware; accuracy is pending.

| | |
|---|---|
| Config | [ultralytics/cfg/models/dla/yolo-dla.yaml](ultralytics/ultralytics/cfg/models/dla/yolo-dla.yaml), scale `n` |
| Parameters | 3,019,336 training (dual-assignment) / **2,392,716 deployed** (fused) |
| Compute | **7.72 GFLOPs** at 640×640 = 3.86 GMAC |
| Engine | 5.4456 MiB, **one DLA loadable, zero GPU layers** |
| Accuracy | pending — training run in progress |

---

## 2. The operator contract

DLA is a fixed-function convolution accelerator, not a general tensor engine. The design is a direct consequence of its restrictions, so they are worth stating precisely. The following is from the TensorRT 10.x "Working with DLA" layer-support tables, cross-checked against the build logs in [logs/](../logs/). Anything marked version-dependent should be re-verified when the JetPack or TensorRT version changes, because these tables have moved between releases.

| Aspect | Restriction |
|---|---|
| Tensors | 4-D only, static shapes; non-batch dimensions in [1, 8192]; min/opt/max profile values must be identical |
| Convolution | 2-D only; kernel [1, 32]; stride [1, 8]; padding [0, 31] and strictly less than the kernel; groups [1, 8192] for linear/CHW16/CHW32, but only **[1, 4]** for `kDLA_HWC4` |
| Activation | **ReLU, Sigmoid, TanH, Clipped ReLU, Leaky ReLU only** |
| Pooling | MAX and AVERAGE; window [1, 8]; stride [1, 16]; padding [0, 7] |
| Resize | exactly 4 scales, first two must equal 1; nearest [1, 32], bilinear [1, 4] |
| ElementWise | Sum, Sub, Product, Max, Min, Div, Pow; broadcasting only for NCHW-equal, NC11, N111 shapes |
| Concatenation | channel axis only; ≥ 2 inputs; identical spatial dimensions |
| Softmax | Orin only; axis dimension ≤ 1024 in the optimized mode, which permits non-axis volume > 1 at the cost of a small approximation error |
| Slice | 4-D input, CHW dimensions, **static slicing only**; Orin only |
| Shuffle / Reshape | 4-D input only; batch dimension cannot participate |
| Partitioning | at most 16 loadables resident per core; Orin defaults to 0.5 MiB managed SRAM per subnetwork |

Two consequences drive everything downstream. First, there is no MatMul and no GEMM, so spatial self-attention cannot be expressed. Second — and this is the one that shapes the head — any single non-compliant operator does not merely fall back on its own: TensorRT pulls the whole fused subgraph (a `ForeignNode`) onto the GPU, including compliant convolutions that happened to fuse with it. One 3-D tensor anywhere in the compute path can therefore drag an otherwise perfect backbone back onto the GPU. The design target is not "mostly DLA" but "no non-compliant operator anywhere between the input and the engine outputs".

Note also what is *not* forbidden, because two of these are commonly assumed to be and are not: static 4-D channel `Slice` is supported on Orin, and SiLU is expressible as Sigmoid followed by an elementwise Product, both of which are DLA operations. Where the design avoids these anyway, it is a performance or toolchain pragmatism, and it is flagged as such below rather than presented as a hardware requirement.

---

## 3. Design derivation

The family starts from the YOLO11n topology and replaces exactly those components that violate the contract. Each subsection states the constraint, then the decision.

### 3.1 `DLARepCSP` — the CSP stage

**Constraint.** None, strictly. A conventional CSP block using `chunk`/`split` traces to ONNX `Split` or `Slice`, and in the round-2 experiments (`logs/dla_yolo11n-v3.md`, lines 121–143) the slice parameters arrived as a chain of roughly ten `Constant`/`Cast`/`ShapeTensorFromDims` nodes per slice, which TensorRT placed on the GPU and which forced a DLA→GPU→DLA reformat hop on every forward pass.

**Decision.** Two *independent* 1×1 projections from the block input instead of one projection followed by a channel split:

```
x ──┬── short: Conv1x1(c1 → h) ─────────────────────────┐
    └── main:  Conv1x1(c1 → h) → RepConv3x3 × n ────────┤
                                                        └── Concat(axis=1) → Conv1x1(2h → c2)
```

with `h = max(int(c2 · e), 16)` and `e = 0.5`. At scale `n` the depth multiplier 0.50 against `repeats: 2` gives `n = 1`, so each stage carries a single 3×3 unit in the transformed branch.

The `RepConv` units are re-parameterizable: at training time each is a 3×3 branch plus a 1×1 branch plus a BatchNorm identity branch, and `model.fuse()` collapses all three into one 3×3 convolution with bias. This buys training-time capacity at zero deployment cost, and it is why the training and deployment parameter counts differ (3.019 M versus 2.393 M). Numerical equivalence of the fusion is asserted in `tests/test_yolo_dla.py::test_yolo_dla_repconv_fuses_without_changing_raw_outputs`.

**Honest caveat.** The two-projection form costs an extra 1×1 convolution over a genuine CSP split, and the toolchain problem that motivated it appears to be solved: exporting a static `chunk(2, 1)` and running `onnxslim` (which the current export path does, via `simplify=True`) folds the shape-helper cascade completely, leaving exactly two `Slice` nodes whose start/end/axes arrive as initializers — **measured** on a synthetic graph. Combined with Orin's support for static 4-D CHW slice, restoring a real CSP split is a live candidate for reducing the projection cost. It is listed as an ablation in §9 rather than changed silently, because it needs an on-device strict build to confirm.

### 3.2 `DLALocalContext` — replacing self-attention

**Constraint.** YOLO11's terminal `C2PSA` block computes `softmax(qᵀk · scale)` and `v @ attnᵀ`. Both map to TensorRT `MATRIX_MULTIPLY`, which DLA does not implement, so the entire block falls back to the GPU. There is no way to express spatial self-attention inside the contract: the affinity matrix requires a MatMul, and forming it requires a transpose across non-batch dimensions.

**Decision.** Give up global, content-based mixing and keep only what the contract allows: a *local, input-dependent* gate. Each mixer stage computes

```
mixer(x) = Conv1x1(ReLU(BN(Conv1x1(x))) ⊙ σ(BN(DWConv5x5(x))))
x ← x + mixer(x)
```

The value path is a pointwise projection; the gate is a 5×5 depthwise convolution followed by Sigmoid, producing a per-channel, per-position multiplier in (0, 1); the product is an elementwise Product between two tensors of identical shape, so no broadcasting rule is engaged. Every operator is in the table above.

What this preserves from attention: the modulation is **data-dependent**, which distinguishes it from a plain residual convolution — the gate is computed from the input, not learned as a static weight. What it gives up: the interaction is local (5×5 at stride 32, so a 160-pixel input footprint) rather than global, it is single-headed, and there is no content-based affinity between distant positions.

Placement is layer 10, immediately after SPPF, at P5/32 where the spatial resolution is 20×20 and the channel count 256 — the cheapest place in the network to spend on context. At scale `n` the depth multiplier gives `n = 1`, so the deployed graph contains exactly one mixer, one Sigmoid, one Mul and one Add, accounting for 1.4% of total MACs. This is a deliberate cost choice but it does limit how much the component can contribute; see §9.

### 3.3 `YOLODLADetect` — the head

**Constraint.** This is where the baseline broke worst. The standard `Detect` path flattens each level to `(B, C, H·W)`, concatenates across levels, applies DFL integral decoding (which reshapes to `(B, 4, reg_max, A)` to softmax over bins), converts distances to boxes, and multiplies by per-level strides. That path produces rank-3 tensors, a `Div`, and per-level stride broadcasts that trace as `(1, 1, H, W)` — N1HW, which is not a permitted broadcast shape. In round 1 the resulting fallback cascade was roughly 200 lines of warnings and dragged the fully-compliant `Proto` branch onto the GPU with it (`logs/dla_yolo11n-v2.md`).

**Decision, part one — the towers.** Each pyramid level gets a decoupled pair of towers, `Conv1x1 → RepConv3x3 → RepConv3x3 → Conv1x1`, with box width `max(32, ch[0]//2, 4·reg_max) = 32` and class width `max(64, min(nc, 128)) = 80`. Dense 3×3 rather than YOLO11's depthwise-separable towers, because depthwise convolution is legal on DLA but wastes its 16-wide atomic channel dimension: a `groups = C` convolution presents one channel per group to a datapath built for sixteen.

**Decision, part two — `reg_max = 1`.** Removing DFL removes the per-bin softmax and the reshape it needs. The box branch regresses the four distances `(l, t, r, b)` directly. This follows upstream YOLO26, which ships `reg_max: 1` for exactly this NMS-free deployment reason, so it is a validated design point rather than a compromise. A DLA-legal DFL is nonetheless constructible — four separate 16-channel branches, a channel-axis Softmax on each (axis dimension 16, well inside the ≤ 1024 limit), then a fixed 1×1 convolution with weights `[0 … 15]` to take the expectation, then a Concat — and it is listed as an ablation.

**Decision, part three — dual assignment, one-to-one at deploy.** Training runs the standard Ultralytics one-to-many branch (`cv2`/`cv3`) alongside a one-to-one branch (`one2one_cv2`/`one2one_cv3`), as in YOLOv10 and YOLO26. `model.fuse()` discards the one-to-many branch. The deployed head is the one-to-one branch, so no NMS is required at all — which matters here because TopK and NMS are not DLA operations and would each fragment the partition. Gradient flow to both branches is asserted in `tests/test_yolo_dla.py::test_yolo_dla_dual_assignment_loss_backpropagates`.

**Decision, part four — headless export.** This is the structural move. In export mode the head emits **one packed 4-D tensor per pyramid level** and nothing else:

```python
out.append(torch.cat([box_head[i](feats[i]), cls_head[i](feats[i])], dim=1))
```

giving `(1, 4·reg_max + nc, H_i, W_i)` = `(1, 84, H_i, W_i)` for COCO. Channel-axis Concat on equal-spatial 4-D inputs is a DLA primitive. Decode — anchor generation, `dist2bbox`, sigmoid, ranking — happens outside the engine.

The alternative of deferring the flatten to the very last operation was tried and is structurally unable to work: the final 3-D concatenation is itself non-compliant, and the ForeignNode fusion propagates that backwards over everything that merged with it. Once the decode leaves the engine, no 3-D tensor exists anywhere in the graph and the partition stops fragmenting.

### 3.4 Everything else, and why it needed no change

The stem and downsampling convolutions are 3×3 stride-2 with padding 1 — inside kernel [1, 32], stride [1, 8], padding [0, 31] and padding < kernel. SPPF is three MAX pools with window 5, stride 1, padding 2 — inside window [1, 8], stride [1, 16], padding [0, 7]. The neck's two upsamples are nearest-neighbour with scales `[1, 1, 2, 2]` — exactly four scales, first two equal to 1, integer factor inside [1, 32]. The PAN concatenations are all channel-axis on equal spatial dimensions. Activation is ReLU, the one activation DLA fuses into the convolution's SDP stage at no extra pass.

---

## 4. Why this is still a YOLO

The name matters because the constraints are severe enough that one could satisfy them with something that is no longer recognisably a YOLO. The defining properties are intact:

**One forward pass, no proposals.** A single fully-convolutional pass produces all predictions. There is no region proposal stage, no learned object queries, no Transformer decoder, no iterative refinement.

**Convolutional backbone plus PAN neck at P3/P4/P5.** Features are produced at strides 8, 16 and 32 and fused top-down and then bottom-up, which is the YOLOv8/YOLO11 neck topology unchanged. The channel plan mirrors YOLO11n stage for stage.

**Dense per-position prediction, anchor-free.** Every spatial position on every level predicts a box and a class distribution. Boxes are direct distance regressions from an anchor point, converted by the same `dist2bbox` used by upstream Ultralytics, with the same `make_anchors` grid at offset 0.5.

**Decoupled heads with task-aligned assignment.** Separate box and classification towers, trained with the standard Ultralytics TAL assigner through the existing `E2EDetectLoss`. No loss code was modified for this family.

**Direct regression, no NMS.** The one-to-one branch is the upstream NMS-free formulation, not an invention here.

The one genuine departure is the *engine boundary*, not the model: the decode that upstream performs inside `Detect._inference` is performed by the caller instead. It is the same arithmetic on the same tensors — `tests/test_yolo_dla.py::test_yolo_dla_external_decode_matches_python_inference` asserts bitwise-close agreement with the standard path via `torch.testing.assert_close`. The network's learned function is identical to a YOLO's; only the partition of who computes the final coordinate transform has changed.

---

## 5. Architecture reference

Scale `n`: depth 0.50, width 0.25, max_channels 1024. Shapes **measured** by forward hooks at 640×640.

| # | Module | from | Output |
|---|---|---|---|
| 0 | Conv 3×3 s2 | -1 | 1×16×320×320 (P1/2) |
| 1 | Conv 3×3 s2 | -1 | 1×32×160×160 (P2/4) |
| 2 | DLARepCSP | -1 | 1×64×160×160 |
| 3 | Conv 3×3 s2 | -1 | 1×64×80×80 (P3/8) |
| 4 | DLARepCSP | -1 | 1×128×80×80 |
| 5 | Conv 3×3 s2 | -1 | 1×128×40×40 (P4/16) |
| 6 | DLARepCSP | -1 | 1×128×40×40 |
| 7 | Conv 3×3 s2 | -1 | 1×256×20×20 (P5/32) |
| 8 | DLARepCSP | -1 | 1×256×20×20 |
| 9 | SPPF k5 | -1 | 1×256×20×20 |
| 10 | DLALocalContext | -1 | 1×256×20×20 |
| 11 | Upsample ×2 | -1 | 1×256×40×40 |
| 12 | Concat | -1, 6 | 1×384×40×40 |
| 13 | DLARepCSP | -1 | 1×128×40×40 |
| 14 | Upsample ×2 | -1 | 1×128×80×80 |
| 15 | Concat | -1, 4 | 1×256×80×80 |
| 16 | DLARepCSP | -1 | 1×64×80×80 → **P3** |
| 17 | Conv 3×3 s2 | -1 | 1×64×40×40 |
| 18 | Concat | -1, 13 | 1×192×40×40 |
| 19 | DLARepCSP | -1 | 1×128×40×40 → **P4** |
| 20 | Conv 3×3 s2 | -1 | 1×128×20×20 |
| 21 | Concat | -1, 10 | 1×384×20×20 |
| 22 | DLARepCSP | -1 | 1×256×20×20 → **P5** |
| 23 | YOLODLADetect | 16, 19, 22 | `pred_p3/p4/p5` |

Head: `ch = (64, 128, 256)`, box tower width 32, class tower width 80, `no = 4·1 + 80 = 84`.

MAC distribution (**measured** from the exported graph): the head is **32.6%** of all MACs, and the P3 class tower alone — two dense 3×3 80→80 convolutions at 80×80 — is 737 MMAC, or 19% of the entire network. That imbalance is the subject of the sweeps in §9.

---

## 6. The engine boundary

**Inside the engine.** Every convolution: backbone, neck, both head towers at all three levels. Input 1×3×640×640, outputs `pred_p3` 1×84×80×80, `pred_p4` 1×84×40×40, `pred_p5` 1×84×20×20, channel-packed as `[box | class_logits]`.

**Outside the engine**, in [ultralytics/utils/dla.py](ultralytics/ultralytics/utils/dla.py): channel unpacking, anchor grid construction, `dist2bbox`, class sigmoid, and the one-to-one top-k ranking. `decode_dla_outputs` reproduces the upstream contract exactly; `decode_dla_backend_outputs` activates only when the loaded model carries `dla_raw` metadata, so PyTorch inference and every non-DLA model pass through untouched.

The export writes `dla_raw`, `dla_outputs`, `dla_strides`, `reg_max` and (for segmentation) `nm` into the model metadata, and the predictor and validator hooks read it back through `AutoBackend`. This means an exported DLA engine or ONNX drops into the normal `yolo predict` / `yolo val` flow with no special-casing at the call site — **measured**: an exported ONNX runs end-to-end through `DetectionPredictor` and recovers `dla_raw=True`, `dla_strides=[8.0, 16.0, 32.0]`, `reg_max=1`, `end2end=True`.

Decode cost, **measured** on x86 with 4 threads at 8400 anchors: 0.67 ms for the default per-class top-k path, 0.53 ms agnostic. The decode is deliberately cheap; the expensive part of the postprocess is the surrounding `Results` construction, not the arithmetic moved out of the engine. Orin's CPU is slower and this needs measuring there.

---

## 7. Verification

### Exported graph, at 640×640 — **measured**

```
ops: Conv 68, Relu 59, Concat 16, MaxPool 3, Resize 2, Sigmoid 1, Mul 1, Add 1   (151 nodes)
non-rank-4 activations:      NONE
forbidden ops:               NONE   (MatMul, Gemm, Div, Slice, Reshape, Transpose, Softmax,
                                     TopK, Gather, Split, Squeeze, Unsqueeze, Shape, Cast, …)
broadcasting elementwise:    NONE   (every Add/Mul has equal-shaped inputs)
Concat axes:                 {1}
Conv constraint violations:  none   (1 grouped conv: the 5×5 depthwise gate)
MaxPool:                     k=5 s=1 p=2
Resize:                      nearest, scales [1, 1, 2, 2]
outputs:                     pred_p3 [1,84,80,80]  pred_p4 [1,84,40,40]  pred_p5 [1,84,20,20]
```

The audit checks operator *constraints*, not just operator names — kernel/stride/padding/group ranges per convolution, pooling window and padding, resize mode and scale integrality, elementwise input-shape equality, and concatenation axis.

### Strict build on hardware — **measured**

AGX Orin 64 GB devkit, TensorRT 10.3.0, DLA core 0, FP16, `--inputIOFormats=fp16:dla_hwc4 --outputIOFormats=fp16:chw16`, GPU fallback **disabled**:

```
Name: {ForeignNode[/model.0/conv/Conv.../model.23/Concat_2]}, LayerType: DLA
  Inputs:  images  [1,3,640,640]  Channel major FP16 where channel % 4 == 0
  Outputs: pred_p3/p4/p5          Sixteen wide channel vectorized row major FP16
Profile: 1 layer, 21.5371 ms, 100.0%
```

One `ForeignNode`, one loadable, **zero GPU layers and zero reformat layers**, engine 5.4456 MiB. This is the compliance result the design was for: the whole network, not a fragment of it.

---

## 8. Performance characterization

Latency, **measured**, same board and settings, batch 1:

| workload | MACs | device time | throughput | achieved rate | enqueue |
|---|---|---|---|---|---|
| yolo-dla-n @640 on **DLA0** | 3.86 G | 21.505 ms | 46.43 qps | 0.179 TMAC/s | 0.128 ms |
| yolo-dla-n @640 on **GPU** | 3.86 G | 1.932 ms | 517 qps | 2.00 TMAC/s | 1.068 ms |
| dense-conv probe on **DLA0** | 30.25 G | 101.31 ms | 9.82 qps | 0.299 TMAC/s | 0.125 ms |
| dense-conv probe on **GPU** | 30.25 G | 3.221 ms | 310 qps | 9.39 TMAC/s | 0.157 ms |

The probe is a deliberately trivial reference graph — a stride-8 stem followed by eight dense 3×3 256→256 convolutions at 80×80 — used as a roofline: the friendliest workload either device will ever see.

Two conclusions follow, and they point in opposite directions.

**The architecture is efficient for the target.** Normalising each device against *its own* dense-conv ceiling, yolo-dla-n achieves **60%** of the DLA's rate but only **21%** of the GPU's. The network is roughly three times better matched to DLA than to an Ampere SM array, which is what a hardware-first design should look like: 68 mostly-small convolutions with 16–256 channels do not fill tensor cores but do suit DLA's fixed pipeline. The enqueue figures say the same thing from the host side — one loadable costs 0.128 ms to submit, while the 71-node GPU engine costs 1.068 ms, over half its own compute time, on Orin's CPU. DLA offload frees host cycles as well as SM cycles.

**The board is not delivering.** In absolute terms DLA is 11.1× slower than the GPU on the model and 31.5× on the probe, against a **derived** spec-sheet expectation of roughly 3.2× (AGX Orin 64 GB: 275 sparse INT8 TOPS total, 170 GPU + 105 for two DLAs; halve for dense, halve again for FP16). The DLA is therefore running about an order of magnitude below its own achievable ceiling. The GPU reaching 44% of its derived FP16 peak on the same graph, in the same process, on the same memory system, rules out a board-wide power, thermal or memory-clock cap; and the DLA's timing is flat from the very first window (21.47–21.67 ms, no ramp) where the GPU visibly ramps 2.09 → 1.93 ms under DVFS. A pinned-low DLA clock domain is the leading hypothesis: `jetson_clocks` locks CPU, GPU and EMC but is documented not to lock DLA, and `nvpmodel` caps DLA frequency per power mode independently.

This is **pending** and tracked separately. It does not affect training, and it does not affect any ratio quoted above — the relative figures are clock-invariant, which is why the architectural conclusion stands regardless of how the clock question resolves. If the DLA reached the same fraction of peak the GPU does, yolo-dla-n would land near 2.2 ms, roughly level with the GPU while occupying none of it.

---

## 9. Architecture sweeps

The head's 32.6% MAC share against a backbone with a single 3×3 unit per stage suggested the compute budget was misallocated. Two sweeps were run on the built model — **measured** parameter counts and `get_flops` at 640×640, fused, so these are static-complexity numbers and not accuracy claims.

### Sweep A — head cost

| head configuration | train params | deploy params | GFLOPs |
|---|---|---|---|
| **v0: 2 RepConv/tower, cls width 80** | 3,019,336 | 2,392,716 | **7.72** |
| 1 RepConv/tower, cls width 80 | 2,569,864 | 2,191,932 | 6.60 |
| 1 RepConv/tower, cls width 64 | 2,408,840 | 2,118,620 | 6.21 |
| 2 RepConv/tower, cls width 64 | 2,719,496 | 2,257,148 | 6.98 |

Trimming the head to one RepConv per tower and a 64-wide class tower frees 1.51 GFLOPs — 19.6% of the whole network — from a component whose upstream equivalents (YOLO11n, YOLO26n) use two *depthwise* convolutions where this uses two dense ones.

### Sweep B — reallocation ladder

Each row is cumulative on the row above.

| variant | change | train params | deploy params | GFLOPs |
|---|---|---|---|---|
| V0 | v0 as built (ReLU) | 3,019,336 | 2,392,716 | 7.72 |
| V1 | head trim (1 rep, cls 64) | 2,408,840 | 2,118,620 | 6.21 |
| V2 | + SiLU | 2,408,840 | 2,118,620 | 6.21 |
| V3 | + backbone 2 blocks/stage | 2,666,568 | 2,349,308 | 7.39 |
| V4 | + neck 2 blocks/stage | 2,924,296 | 2,579,996 | 7.86 |
| V5 | V3 + LocalContext n=2 | 2,805,576 | 2,487,548 | 7.50 |

The informative result is V5 against V0: **the backbone depth can be doubled, the activation upgraded to SiLU, and the context block made non-trivial, all for slightly less compute than v0 spends today** (7.50 versus 7.72 GFLOPs), by paying for it out of the head. V5 was built, exported and audited — `Conv 69, Sigmoid 60, Mul 60, Concat 16, MaxPool 3, Add 2, Resize 2`, all rank-4, no forbidden operators, no broadcasting — so the SiLU switch does not break the operator contract at the ONNX level. Whether it still forms a single loadable is **pending** an on-device strict build.

Single-knob deltas measured directly off v0, for constructing an ablation table:

| knob | how | train params | deploy params | GFLOPs |
|---|---|---|---|---|
| — | v0 reference | 3,019,336 | 2,392,716 | 7.72 |
| SiLU | `activation: torch.nn.SiLU()` in the YAML | 3,019,336 | 2,392,716 | 7.72 |
| head trim | `YOLODLADetect.tower_reps = 1`, `.cls_width = 64` | 2,408,840 | 2,118,620 | 6.21 |
| deep backbone | backbone `DLARepCSP` repeats `2 → 4` | 3,277,064 | 2,623,404 | 8.90 |
| context n=2 | `DLALocalContext` repeats `1 → 4` | 3,158,344 | 2,530,956 | 7.83 |

SiLU leaves both columns unchanged because it adds no parameters and `thop` does not attribute FLOPs to activations — its cost is entirely in extra tensor passes on the accelerator, which is exactly why it has to be measured on-device rather than inferred from this table.

References at the same measurement settings: YOLO11n 2,616,248 params / 6.48 GFLOPs; YOLO26n 2,408,932 / 5.36; yolo11-dla 2,607,960 / 6.48.

### Decision

**v0 is what is being trained.** The sweeps establish static complexity, not accuracy, and no variant has been trained — so preferring one on reasoning alone would make the headline model a guess. v0 is the configuration the design documents describe, it is the one already verified on hardware, and it is the reference every ablation is measured against. The variants above are the ablation ladder, and each is a small diff reachable through the knobs in the table rather than a fork of the code.

Two of the four knobs deserve a stated prior for when the ablations are run. SiLU is the best-supported change in the literature for detectors of this size and is DLA-legal, but it adds 60 Sigmoid and 60 Mul nodes — two extra full-tensor passes per convolution, where ReLU fuses into the convolution's own SDP stage at no cost — so it trades latency for accuracy and the exchange rate must be measured on-device. The `DLALocalContext` repeat count is the weakest: at `n = 1` the block is 1.4% of MACs, so an ablation against removing it will probably be indistinguishable from noise, which is an argument for raising it if the component is to carry weight in the paper.

---

## 10. Known gaps

**Non-linear I/O formats cannot be consumed through `AutoBackend`.** The strict build that yields zero GPU layers requires `fp16:dla_hwc4` in and `fp16:chw16` out, but the TensorRT backend allocates every binding as a linear `np.empty(shape, dtype)` and never calls `get_tensor_format()`. CHW16 stores 84 channels as six 16-channel blocks, so reading that buffer as `(1, 84, 80, 80)` yields misaligned data. `trtexec` benchmarks without checking output correctness, so this does not show up in the compliance run. Either the bindings become format-aware, or the deployment accepts the in-engine reformat layer and reports it.

**The Ultralytics export path cannot produce the strict engine.** `ultralytics/utils/export/engine.py` sets `trt.BuilderFlag.GPU_FALLBACK` unconditionally whenever DLA is requested, and sets no I/O formats. So `yolo export format=engine device=dla:0` produces a fallback-enabled engine, and the benchmarked artifact is not the deployable artifact. A strict option is needed before the paper's coverage claim and its deployment story refer to the same file.

**Scales `s` and `m` are unreachable.** `guess_model_scale` matches `yolo(e-)?v?\d+([nslmx])`, which requires digits immediately after `yolo`; no filename of the form `yolo-dla*` can ever match, so the YAML's hardcoded `scale: n` always wins. Irrelevant for the current run, blocking for `yolo-dla-s`.

**Latent: end-to-end plus mask coefficients.** `decode_dla_backend_outputs` with `end2end=True` and `nm > 0` returns `(B, K, 6+nm)`, and `non_max_suppression` recognises the end-to-end case only via `shape[-1] == 6`, so NMS would run on already-ranked output. Not reachable today — `yolo11-dla-seg` is `end2end=False` and this family has no segmentation variant — but it will matter when one is added.

**Fixed input resolution.** Required by DLA; not a defect, but it constrains multi-scale training and any variable-resolution deployment.

---

## 11. Reproduction

```bash
cd tools/ultralytics

# structure and complexity
yolo detect model=ultralytics/cfg/models/dla/yolo-dla.yaml info

# unit tests: export contract, dual-assignment gradients, RepConv fusion, external-decode parity
python3 -m pytest tests/test_yolo_dla.py -q

# export (3 named rank-4 outputs)
yolo export model=ultralytics/cfg/models/dla/yolo-dla.yaml format=onnx imgsz=640 simplify=True dynamic=False
```

Strict DLA build on the Jetson. The absence of `--allowGPUFallback` is the point: a successful build is itself the compliance result.

```bash
/usr/src/tensorrt/bin/trtexec --onnx=yolo-dla.onnx --saveEngine=yolo-dla-fp16.engine \
  --useDLACore=0 --fp16 \
  --inputIOFormats=fp16:dla_hwc4 --outputIOFormats=fp16:chw16 \
  --profilingVerbosity=detailed --dumpLayerInfo --dumpProfile --separateProfileRun \
  --useSpinWait --warmUp=2000 --duration=15 --avgRuns=50

# pass criteria
grep -cE "LayerType: DLA" run.log          # expect 1
grep -iE "falling back|not supported" run.log   # expect no matches
```

Training, from scratch — no shape-compatible checkpoint exists for this family, so `--epochs 100` would say very little:

```bash
./tools/coco_dla_train.sh start --fresh --model yolo-dla-n --epochs 500
./tools/coco_dla_train.sh status --model yolo-dla-n
```

Note that every subcommand needs `--model`; the launcher defaults to `yolo11n-dla`. Model construction goes through a wrapper that saves and restores `Conv.default_act`, because a YAML `activation:` key mutates that class attribute process-wide and would otherwise rebuild a later YOLO11 or YOLO26 baseline with the wrong activation — a silent contamination of exactly the comparison the paper depends on.
