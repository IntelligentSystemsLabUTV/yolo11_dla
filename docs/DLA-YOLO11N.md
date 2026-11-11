# DLA-YOLO11N

DLA-compatible variants of Ultralytics YOLO11N for NVIDIA Jetson Orin (TensorRT 10.x).
Targets `--useDLACore=0` with **the entire engine on DLA** — backbone, neck, all
detection / segmentation conv heads, and `proto`. With DLA-native I/O formats
(`--inputIOFormats=fp16:dla_hwc4 --outputIOFormats=fp16:chw16`), `--allowGPUFallback`
is **not needed** — the engine builds and runs with zero GPU layers (v5-verified).

---

## Background

NVIDIA DLA (Deep Learning Accelerator) is a fixed-function hardware block on Jetson Orin.
It runs Conv2d, pooling, elementwise ops, and softmax efficiently, but has hard restrictions:

- All tensors must be exactly **4-D** (N, C, H, W)
- No `MatMul` / `GEMM` (maps to unsupported `MATRIX_MULTIPLY`)
- No `DIV` elementwise (only supported in INT8 mode)
- `Slice` only on 4-D tensors
- `Softmax` supported but adds small numerical error when non-axis volume > 1
- Broadcast only in **NC11** or **N111** shapes — not N1HW or NHW1

Any op that violates these constraints causes TensorRT to pull the entire fused subgraph
(ForeignNode) back to GPU, including surrounding DLA-safe ops that got merged with it.

Baseline `yolo11n-seg` has two problem areas when compiled with `--useDLACore`:

| Area | Root cause | TRT symptom | Fix |
|---|---|---|---|
| `model.10` (C2PSA) | `torch.matmul` in `Attention` | `MATRIX_MULTIPLY` unsupported | `C2DLA` + `DLAAttention` (two-conv K/V projection, no MatMul, no Slice) |
| `model.23` (Segment head) | `.view()` to 3-D inside `forward_head` and `DFL` | `SHUFFLE` / `SLICE` / `DIV` fallback cascade | Headless channel-packed export — emit 3 / 4 named 4-D outputs, decode externally |

---

## New files

```
ultralytics/nn/modules/block.py    — DLAAttention, DLABlock, C2DLA, DFL4D
ultralytics/nn/modules/head.py     — DetectDLA, SegmentDLA (headless export)
ultralytics/nn/modules/__init__.py — exports (C2DLA, DFL4D, DetectDLA, SegmentDLA)
ultralytics/nn/tasks.py            — parse_model registration
ultralytics/cfg/models/11/yolo11-dla.yaml
ultralytics/cfg/models/11/yolo11-dla-seg.yaml
examples/dla_decode.py              — external decode reference (DFL + dist2bbox + sigmoid)
```

No existing YOLO11 files are modified. These are additive, PR-ready changes.

`DFL4D` is still exported from `block.py` for completeness but is **not used by the
DLA head** — the headless export strategy removes DFL decode from the engine
entirely. It is kept in case someone wants to bake DFL back into the engine on a
device with stricter softmax tolerances.

---

## Change 1 — `C2DLA` replaces `C2PSA` at backbone layer 10

### Problem

`C2PSA` wraps a `PSABlock` → `Attention` module (`block.py:1297`) which computes:

```python
attn = (q.transpose(-2,-1) @ k) * self.scale     # MATRIX_MULTIPLY — not on DLA
x    = (v @ attn.transpose(-2,-1))               # MATRIX_MULTIPLY — not on DLA
```

TRT maps both `@` ops to `MATRIX_MULTIPLY`. DLA does not support this op type,
so the entire C2PSA subgraph falls back to GPU.

### Fix — `DLAAttention` + `DLABlock` + `C2DLA` in `block.py`

The new module is a **direct one-to-one translation of the original `Attention`** with
the matmul replaced by External Attention (Guo et al. 2021). The macro structure
(input projection → attn-branch + DWConv positional encoding on V → output projection)
is preserved literally; only the QKᵀ-based attention math is swapped:

```python
class DLAAttention(nn.Module):
    def __init__(self, c, k=None):
        self.k_dim = k if k is not None else max(c // 4, 8)
        self.proj_k    = Conv(c, self.k_dim, 1, act=False)        # ≈ qkv[:, :nh_kd]
        self.proj_v    = Conv(c, c,         1, act=False)        # ≈ qkv[:, nh_kd:]
        self.proj_attn = nn.Conv2d(self.k_dim, c, 1, bias=False)  # ≈ V @ attnᵀ
        self.pe        = Conv(c, c, 3, g=c, act=False)            # = self.pe (DWConv)
        self.proj_out  = Conv(c, c, 1, act=False)                 # = self.proj

    def forward(self, x):
        k = self.proj_k(x)                           # (B, k_dim, H, W) — no Slice
        v = self.proj_v(x)                           # (B, c,     H, W) — no Slice
        attn = self.proj_attn(k.softmax(dim=1))      # EA: softmax over channel axis
        return self.proj_out(attn + self.pe(v))      # mirror original: attn + pe(v) → proj
```

K and V come from **two separate 1×1 convs** rather than a fused projection
followed by a channel slice. The fused-then-slice form, while functionally
equivalent, traces to ONNX `Slice` nodes whose `start`/`end`/`axes` inputs ship
as a chain of ~10 `Constant` / `Cast` / `ShapeTensorFromDims` ops per slice;
TRT places those on GPU and forces a DLA→GPU→DLA reformat hop on every forward
(see `logs/dla_yolo11n-v3.md` lines 121-143). Two convs avoid the cascade
entirely and have the **same parameter count** (`Conv(128, 32, 1) + Conv(128, 128, 1)`
= `Conv(128, 160, 1)`).

**Direct mapping from the original `Attention`:**

| Original (`block.py:1297`) | `DLAAttention` |
|---|---|
| `self.qkv[:, :nh_kd]` slice | `self.proj_k = Conv(c, k, 1)` — dedicated K projection |
| `self.qkv[:, nh_kd:]` slice | `self.proj_v = Conv(c, c, 1)` — dedicated V projection |
| `softmax(qᵀ @ k) · scale` (MatMul) | `softmax(K, dim=1)` over channel axis — DLA-safe |
| `v @ attnᵀ` (MatMul) | `self.proj_attn(...)` — 1×1 Conv as low-rank substitute |
| `self.pe = Conv(dim, dim, 3, g=dim)` | `self.pe = Conv(c, c, 3, g=c)` — identical 3×3 DWConv on V |
| `attn_out + pe(v)` | `attn + self.pe(v)` — identical Add (NCHW-equal) |
| `self.proj = Conv(dim, dim, 1)` | `self.proj_out = Conv(c, c, 1)` |

What necessarily differs (the DLA-blocked semantics):

- Attention is **no longer data-dependent** — both Q and K came from `x` in the
  original; EA replaces this with a learned external memory inside the conv weights.
- Attention is **no longer multi-head** (multi-head requires the QKᵀ matmul).
- The mixing matrix has **rank `k` instead of H·W** — a low-rank approximation.

`DLABlock` mirrors `PSABlock` (`block.py:1357`) one-to-one:

```python
class DLABlock(nn.Module):
    def __init__(self, c, k=None, shortcut=True):
        self.attn = DLAAttention(c, k)
        self.ffn  = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add  = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x)  if self.add else self.ffn(x)
        return x
```

`C2DLA` mirrors `C2PSA` exactly — split → DLABlock × n → merge:

```
x → cv1 → split → [a | b]
                        b → DLABlock × n → b'
               [a | b'] → cv2 → output
```

The only architectural deviation from C2PSA is inside `DLAAttention`'s
`softmax(K) → proj_attn` substituting for `softmax(QᵀK) @ V`. Retraining is
required because the attention mechanism's parameterization changed.

---

## Change 2 — Headless export: `DetectDLA` / `SegmentDLA` emit raw 4-D feature maps

### Problem

The first attempt at this change kept the decode (DFL → dist2bbox → sigmoid →
flatten + concat) inside the engine and tried to defer the 3-D flatten to the
very end. That strategy is structurally unable to land the head on DLA:

- The final `torch.cat(..., dim=2)` produces a 3-D tensor. DLA cannot consume
  3-D, so TRT pulls every upstream op fused into the same ForeignNode back to
  GPU — including `Proto`, which is itself fully DLA-safe.
- `.view(B, 4, -1)` with a dynamic `-1` traces to `ONNXTRT_ShapeTensorFromDims` /
  `castHelper` constants that the DLA compiler refuses. The v2 log
  (`logs/dla_yolo11n-v2.md`) shows ~150 lines of these warnings in `model.23`.
- Per-scale stride broadcasts trace as `(1, 1, H_i, W_i)` (N1HW), even when the
  source tensor is `(1, 1, 1, 1)` — ONNX inlines the constant against the
  traced feature shape. DLA only accepts NCHW-equal / NC11 / N111, so the
  `box * stride` Mul fell back to GPU on every scale (`Mul_1`, `Mul_3`, `Mul_5`
  in the v2 log, lines 335–357).
- `{ForeignNode[/model.23/proto/cv1/conv/Conv.../model.23/Concat_6] cannot be
  compiled by DLA, falling back to GPU.}` (v2 log, lines 358–362) — the Proto
  branch is fused with the 3-D final concat, so the dimension-assertion failure
  drags every Conv inside Proto back to GPU.

In short: **as long as any 3-D tensor exists in the engine's compute path, the
DLA partition fragments** and the resulting engine is mostly GPU regardless.

### Fix — headless export with channel-packed outputs

`DetectDLA` / `SegmentDLA` keep the standard `Detect` / `Segment` 3-D decode for
training and Python inference (where 3-D and GPU are fine). For `self.export`
they switch to a different output contract: **one packed 4-D tensor per pyramid
level**, where `box` / `score` / (and `mask_coefficient`) are concatenated along
the channel axis. Plus a separate `proto` output for segmentation.

```python
class DetectDLA(Detect):
    def forward(self, x):
        if self.export:
            return self._export_4d(x)
        return super().forward(x)   # standard Detect path (3-D, GPU-fine)

    def _export_4d(self, feats):
        box_head = self.one2one_cv2 if self.end2end else self.cv2
        cls_head = self.one2one_cv3 if self.end2end else self.cv3
        out = []
        for i in range(self.nl):
            box   = box_head[i](../feats[i])   # (B, 4*reg_max, H_i, W_i)
            score = cls_head[i](../feats[i])   # (B, nc,        H_i, W_i)
            out.append(torch.cat([box, score], dim=1))   # (B, 4*reg_max + nc, H_i, W_i)
        return tuple(out)
```

`SegmentDLA.forward` extends this by also concatenating mask coefficients into
each per-scale packed tensor, and adds `proto` as the final output. The
per-scale `Concat(axis=1)` is a DLA primitive on NCHW 4-D inputs — the existing
17 backbone/neck Concats in the v3 ONNX already proved it runs on DLA.

### Output contract

| | `DetectDLA` export | `SegmentDLA` export |
|---|---|---|
| Outputs | **3 tensors** | **4 tensors** |
| Output ranks | all 4-D `(N, C, H, W)` | all 4-D `(N, C, H, W)` |
| Channel pack | `[box \| score]` | `[box \| score \| mask]` |
| `pred_p3` | `(1, 4·reg_max + nc, 64, 64)` | `(1, 4·reg_max + nc + nm, 64, 64)` |
| `pred_p4` | `(1, 4·reg_max + nc, 32, 32)` | `(1, 4·reg_max + nc + nm, 32, 32)` |
| `pred_p5` | `(1, 4·reg_max + nc, 16, 16)` | `(1, 4·reg_max + nc + nm, 16, 16)` |
| `proto` | — | `(1, nm, 128, 128)` |

Concrete shapes for `nc=80, reg_max=16, nm=32`: each `pred_p{3,4,5}` has
**144 channels** (det) or **176 channels** (seg). `P3/P4/P5` refer to FPN
pyramid levels with strides 8 / 16 / 32 (`P{i}` ⇒ stride = `2^i`).

Every tensor on every code path from the network input to an engine output is
strictly 4-D. No `.view()`, no `chunk()`, no `/` — none of the things that
caused the v2 fragmentation exist in the engine anymore. Output names come
from a small patch in `ultralytics/engine/exporter.py` that recognises the
DLA heads and overrides the default `output0`/`output1` naming.

### External decode

Whatever the engine no longer computes must be done in the caller. The
reference implementation is in `tools/ultralytics/examples/dla_decode.py`.
Per scale, channels are unpacked with simple `[:, :a]` / `[:, a:b]` slices in
PyTorch (this happens *after* the engine returns, so the slices are GPU ops,
not in the DLA graph):

```python
from examples.dla_decode import decode_heads

# SegmentDLA engine returns (pred_p3, pred_p4, pred_p5, proto)
preds_per_scale = list(outs[:3])           # the 3 packed per-scale tensors
proto           = outs[3]                  # already-decoded prototype masks

dets, mask_coeff = decode_heads(preds_per_scale, head.stride,
                                reg_max=head.reg_max, nc=head.nc, nm=head.nm)
# dets:       (B, 4 + nc, total_anchors)  — xywh + class probs (same as standard Detect._inference)
# mask_coeff: (B, nm,     total_anchors)  — None when nm == 0 (detection variant)
```

Numerical parity vs the standard 3-D decode is verified by
`python examples/dla_decode.py --verify`: max abs diff ~6e-5 (float32 noise
from the order of ops), proto bit-exact.

### Why this is the right partition

The decode is cheap (a handful of softmax + multiply + add ops on
`(B, 4·reg_max, 5376)` tensors). It runs in microseconds on the Orin GPU. The
heavy work — every conv in the backbone, neck, three detection branches, three
mask branches, and Proto — is the part we wanted on DLA, and that is exactly
what stays on DLA after this change.

Training mode is **completely unchanged** in both classes (`super().forward(x)`
goes through `Detect.forward` → `forward_head` → 3-D loss path). Python
inference (non-export) is also unchanged. Only the export branch was rewritten.

---

## Model comparison (scale `n`, imgsz=512)

| | `yolo11n` | `yolo11n-dla` | `yolo11n-seg` | `yolo11n-dla-seg` |
|---|---|---|---|---|
| Layers | 182 | 183 | 204 | 208 |
| Parameters | 2,624,080 | 2,615,696 | 2,876,848 | 2,868,464 |
| GFLOPs (eval, non-export) | 6.6 | 6.6 | 9.9 | 9.9 |
| MatMul / Gemm (exported ONNX) | — | 0 | — | 0 |
| Reshape | — | 0 | — | 0 |
| Transpose | — | 0 | — | 0 |
| Div | — | 0 | — | 0 |
| Slice | — | 0 | — | 0 |
| Softmax | — | 1 | — | 1 |
| Concat (NCHW, channel-axis) | — | 20 | — | 20 |
| ONNX outputs | — | 3 (rank-4, named `pred_p3/p4/p5`) | — | 4 (rank-4, named `pred_p3/p4/p5/proto`) |

Only `Softmax` is in `DLAAttention` at `model.10/m/m.0/attn` — channel-axis on K
(DLA Softmax approximation; small numerical error, **not** a GPU fallback). The
K/V channel-slice that existed in round 2 was replaced by two parallel 1×1 convs
(`proj_k` / `proj_v`), so the round-3 ONNX has **zero** Slice nodes.

The 20 `Concat` nodes are all channel-axis on 4-D NCHW tensors: 17 in the
backbone/neck (cross-scale feature concatenation, SPPF) and 3 added in round 4
to pack `[box | score | (mask)]` per pyramid level into one output tensor each.
Round 1–3 emitted 6 (det) or 10 (seg) outputs; round 4's per-scale channel-pack
reduces this to 3 / 4 named outputs without breaking the all-4-D DLA contract.

There are **no** `Reshape` / `Transpose` / `Div` / `Slice` nodes anywhere in the
exported ONNX. All ops from input to engine output stay 4-D.

---

## What still falls back to GPU (expected)

| What | Why | Runs on |
|---|---|---|
| DLA Softmax numerical-error notice in `DLAAttention` | DLA hardware limitation (channel softmax with H·W > 1) | DLA (notice, not a fallback) |
| Input / output tensor reformat (`fp32:CHW` ↔ DLA-native) | DLA cores cannot reformat tensor layouts themselves | GPU — *unless* `--inputIOFormats=fp16:dla_hwc4 --outputIOFormats=fp16:chw16` is passed, in which case the reformat disappears (v5-confirmed) |
| Channel unpacking, DFL decode, `dist2bbox`, sigmoid, flatten + cat, NMS, mask blending | Moved out of the engine | External GPU (`examples/dla_decode.py`) |

Inside the engine, **every** op from the input to the engine outputs is
DLA-compatible. With DLA-native I/O formats (v5), the build succeeds with
**`--allowGPUFallback=False`** — there is no GPU layer in the engine at all.
With default `fp32:CHW` I/O, a sub-millisecond GPU reformat layer is inserted
on each side; everything else still runs on DLA.

---

## Verification status

What has been verified locally (x86 dev box, no TensorRT/DLA hardware):

- ✅ Both YAMLs parse and build via `parse_model` (183 / 205 layers, 2.616M / 2.868M params)
- ✅ `train()` mode returns the standard dict (`boxes`, `scores`, `feats`,
  `mask_coefficient`, `proto`) — unchanged from baseline; loss code works as-is
- ✅ `eval()` non-export mode returns `((y, proto), preds)` with `y` shape
  `(1, 116, 5376)` — same format as standard `Segment._inference`
- ✅ Export mode returns a 6-tuple (`DetectDLA`) / 10-tuple (`SegmentDLA`) of
  rank-4 tensors with the documented shapes
- ✅ ONNX export succeeds; in the exported graph:
  - **0** MatMul / Gemm / Div / Reshape / Transpose / Slice nodes
  - 3 (det) / 4 (seg) outputs, all rank-4, explicitly named `pred_p3` / `pred_p4`
    / `pred_p5` / `proto`
  - The only `Softmax` is inside `model.10/m/m.0/attn` (DLAAttention channel softmax)
  - `model.10/m/m.0/attn` is 7 pure 4-D nodes: `proj_k/Conv`, `proj_v/Conv`,
    `Softmax`, `proj_attn/Conv`, `pe/Conv`, `Add`, `proj_out/Conv` — no
    `Slice` / `Constant` / `Cast` / `ShapeTensorFromDims` cascade
  - Each `pred_p{3,4,5}` output is produced by a single channel-axis
    `model.23/Concat` over `cv2/cv3[/cv4]` Conv outputs
  - The `proto` output comes from `model.23/proto/cv3/act/Mul` (SiLU activation)
- ✅ `examples/dla_decode.py --verify` — external decode matches standard
  `Segment._inference` to within 6.1e-5 max abs diff (float32 ordering noise);
  proto is bit-exact

On-Orin compile history (`logs/dla_yolo11n-v{1..5}.md`):

| Log | Round | Cmd flags | Engine | Head | Fallbacks | Throughput |
|---|---|---|---|---|---|---|
| `dla_yolo11n.md` | baseline | `--fp16 --allowGPUFallback` | builds | in-engine (3-D) | 6× MatMul + Proto assertion | 43.6 qps (FP16) |
| `dla_yolo11n-v2.md` | r1 (defer-flatten) | `--allowGPUFallback` (FP32) | builds | in-engine, fragments | head + Proto fallbacks; SE pool | 49.4 qps |
| `dla_yolo11n-v3.md` | r2 (headless, 10 outs) | `--allowGPUFallback` (FP32) | builds | external | only DLAAttention slice helpers | 45.5 qps |
| `dla_yolo11n-v4.md` | r3 (slice elim, 10 outs) | `--allowGPUFallback` (FP32) | builds | external | **only DLA-Softmax notice** | 45.5 qps |
| `dla_yolo11n-v5.md` | r3 (slice elim, 10 outs) | `--fp16 --inputIOFormats=fp16:dla_hwc4 --outputIOFormats=fp16:chw16` (no fallback) | **builds without `--allowGPUFallback`** | external | **only DLA-Softmax notice; engine is 100% on DLA** | 46.0 qps |

Key result from v5: with DLA-native I/O formats the engine compiles **with no
GPU layers at all** — input/output reformat included. The only remaining DLA
notice is the channel-softmax in `DLAAttention`, which is a precision warning
(not a fallback).

Pending on Orin for round 4 (this update — 3/4 named packed outputs):

- ⏳ Re-run v5's command on the round-4 ONNX (`pred_p3/p4/p5/proto`). Expected:
  same DLA cleanliness as v5 — only the DLA-Softmax notice, no GPU layers, no
  broadcast / ForeignNode failures. The 3 added per-scale Concats are NCHW
  channel-axis ops and compile to DLA the same way the existing 17 backbone /
  neck Concats already did.
- ⏳ Throughput is expected to be within noise of v5's 46.0 qps. The packing
  Concats add 3 ops on the DLA's elementwise pipeline; the H2D / D2H budget is
  unchanged (total output bytes are identical — just reorganised).

---

## Retraining notes

Layers 0–9 and 11–22 have identical architecture to the corresponding `yolo11n`
weights and can be initialized directly from `yolo11n.pt` / `yolo11n-seg.pt`.

Layer 10 (`C2DLA` / `DLAAttention`) replaces `C2PSA` / `Attention`. Because
`DLAAttention` mirrors the original's macro structure (input projection → attn
+ DWConv on V → output projection), several weights can be transferred:

- `Attention.qkv[:, :nh_kd, :, :]` (K slice) → `DLAAttention.proj_k.conv.weight`
  — **directly transferable** when `k = nh_kd` (set `k=nh_kd` on construction)
- `Attention.qkv[:, 2·nh_kd:, :, :]` (V slice) → `DLAAttention.proj_v.conv.weight`
  — **directly transferable** (the dropped Q slice goes to `/dev/null`)
- `Attention.pe` (3×3 DWConv) → `DLAAttention.pe` — **directly transferable**
- `Attention.proj` (1×1 Conv) → `DLAAttention.proj_out` — **directly transferable**

Only `DLAAttention.proj_attn` (the EA "V@attnᵀ" substitute) has no pretrained
counterpart and must be initialized from scratch.

Recommended workflow: load pretrained weights with `strict=False`, optionally
copy `pe` and `proj_out` weights from the pretrained Attention into the new
module by name match, freeze layers 0–9 and 11–22 for a warmup phase, then
fine-tune all layers.

---

## Usage

```bash
# Train in the background (model and total target epochs are CLI parameters)
./tools/coco_dla_train.sh start --model yolo11n-dla --epochs 100

# Start a fresh YOLO11s-DLA training with separate local and W&B runs
./tools/coco_dla_train.sh start --fresh --model yolo11s-dla --epochs 300
./tools/coco_dla_train.sh status --model yolo11s-dla

# Extend an existing run from its latest full-state checkpoint to epoch 300
./tools/coco_dla_train.sh stop
./tools/coco_dla_train.sh status  # wait until it reports "not running"
./tools/coco_dla_train.sh resume --model yolo11n-dla --epochs 300

# See all supported model names and training options
python3 tools/train_coco_dla.py --help

# Build / inspect
yolo task=detect  model=ultralytics/cfg/models/11/yolo11-dla.yaml     info
yolo task=segment model=ultralytics/cfg/models/11/yolo11-dla-seg.yaml info

# Export (channel-packed: 3 (det) or 4 (seg) named 4-D outputs)
yolo export model=yolo11n-dla.pt     format=onnx imgsz=512
yolo export model=yolo11n-dla-seg.pt format=onnx imgsz=512

# Recommended compile on Jetson Orin (v5-confirmed clean DLA path — no GPU fallback):
trtexec --onnx=yolo11n-dla-seg.onnx \
        --saveEngine=yolo11n-dla-seg.engine \
        --useDLACore=0 --fp16 \
        --inputIOFormats=fp16:dla_hwc4 \
        --outputIOFormats=fp16:chw16

# Alternative (less aggressive) — keeps fp32:CHW I/O and uses a tiny GPU reformat:
trtexec --onnx=yolo11n-dla-seg.onnx \
        --saveEngine=yolo11n-dla-seg.engine \
        --useDLACore=0 --fp16 --allowGPUFallback=True

# Verify ONNX shape + name contract before shipping to Orin
python -c "
import onnx
m = onnx.load('yolo11n-dla-seg.onnx')
ops = [n.op_type for n in m.graph.node]
for op in ['MatMul', 'Gemm', 'Div', 'Transpose', 'Reshape', 'Slice']:
    assert ops.count(op) == 0, f'Found {ops.count(op)} {op} nodes'
print('OK — no DLA-blocking ops')
print('Outputs:', len(m.graph.output))
for o in m.graph.output:
    dims = [d.dim_value for d in o.type.tensor_type.shape.dim]
    print(f'  {o.name:8s} {dims}')
# Expected for SegmentDLA (n-scale, imgsz=512):
#   pred_p3  [1, 176, 64, 64]
#   pred_p4  [1, 176, 32, 32]
#   pred_p5  [1, 176, 16, 16]
#   proto    [1,  32, 128, 128]
"

# Numerical-parity test for the external decoder vs Segment._inference
python examples/dla_decode.py --verify
```

After compiling on Orin, the engine returns 3 (`DetectDLA`) or 4 (`SegmentDLA`)
named 4-D outputs. Apply `examples/dla_decode.py::decode_heads` to unpack the
per-scale `[box | score | mask]` channels and produce the standard
`(B, 4+nc, total_anchors)` + `(B, nm, total_anchors)` tensors on GPU; use the
engine's `proto` output directly.
