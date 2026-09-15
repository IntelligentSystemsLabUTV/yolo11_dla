# The YOLO11-DLA patch to Ultralytics

Everything YOLO11-DLA-n adds to stock Ultralytics, and nothing else. The fork branch also carries an independent from-scratch detector family (YOLO-DLA) and a segmentation variant; neither is part of this paper, and both are listed under [What is deliberately excluded](#what-is-deliberately-excluded) so that the boundary is explicit rather than implied.

Base revision: upstream Ultralytics `a462bb65` ("Professionalize YOLOv5 tutorials", #24468), the merge base of the fork. Applying the ten changes below to that revision reproduces the model, the export and the host decoder used for every result in the paper. The already-applied result is published at <https://anonymous.4open.science/r/ultralytics-dla>; this page is the change-by-change account of what is in it and why, and lets the patch be reapplied to any other checkout.

| File | Change | Required for |
| --- | --- | --- |
| `ultralytics/nn/modules/block.py` | add `DLAAttention`, `DLABlock`, `C2DLA` | the architecture |
| `ultralytics/nn/modules/head.py` | add `DetectDLA` | the export boundary |
| `ultralytics/cfg/models/11/yolo11-dla.yaml` | new file | the model definition |
| `ultralytics/nn/modules/__init__.py` | export the two new public names | module resolution |
| `ultralytics/nn/tasks.py` | register both in `parse_model` | building from YAML |
| `ultralytics/utils/dla.py` | new file | host-side decoding of the raw maps |
| `ultralytics/engine/exporter.py` | output names, metadata, INT8 TopK bypass | ONNX/engine export |
| `ultralytics/nn/backends/base.py` | typed metadata round-trip | reading an exported engine |
| `ultralytics/models/yolo/detect/{predict,val}.py` | route raw outputs through the decoder | `predict()` / `val()` on an engine |
| `ultralytics/utils/nms.py` | six-candidate disambiguation | correctness of the sparse path |

## 1. `ultralytics/nn/modules/block.py`

Three new classes, appended after `C2PSA`. `DLAAttention` is the substitution that makes strict placement possible: the stock `Attention` computes `softmax(QᵀK) @ V`, and DLA has no `MATRIX_MULTIPLY`. The replacement keeps the macro structure — input projection, attention branch plus depthwise positional encoding on V, output projection — and replaces only the attention arithmetic with a channel-axis softmax through a bottleneck, which is a 1×1 convolution and a softmax over an axis DLA does support.

```python
class DLAAttention(nn.Module):
    """DLA-compatible analog of the original PSA Attention module.

    Direct one-to-one translation of `Attention`. Macro structure is preserved —
    input projection → (attention branch + DWConv positional encoding on V) →
    output projection. The only deviation from the original is the attention math
    itself: `softmax(QᵀK) @ V` is replaced by `proj_attn(softmax_channel(proj_k(x)))`,
    which uses only 1×1 convs and channel-axis softmax, all DLA-safe.

    The K and V projections are two separate 1×1 convs (not a fused projection
    followed by a channel slice). The fused-then-slice form, while functionally
    equivalent, traces to an ONNX `Slice` chain that ships as ~10 helper
    `Constant` / `Cast` / `ShapeTensorFromDims` ops per slice — TRT places those
    on GPU and forces a DLA→GPU→DLA register-copy hop per scale. Two convs avoid
    that entirely and keep the same parameter count.

    Mapping from original `Attention`:
        self.qkv[:, :nh_kd] → self.proj_k    (1×1 conv producing the K-bottleneck)
        self.qkv[:, nh_kd:] → self.proj_v    (1×1 conv producing V)
        QᵀK @ V            → self.proj_attn  (1×1 conv as low-rank substitute for V@attnᵀ)
        self.pe            → self.pe         (3×3 DWConv on V, unchanged)
        self.proj          → self.proj_out   (final 1×1 projection)

    Attention is no longer data-dependent or multi-head (those require MatMul);
    instead it's a fixed low-rank channel mixer with rank `k`.
    """

    def __init__(self, c: int, k: int | None = None) -> None:
        super().__init__()
        self.c = c
        self.k_dim = k if k is not None else max(c // 4, 8)
        self.proj_k = Conv(c, self.k_dim, 1, act=False)
        self.proj_v = Conv(c, c, 1, act=False)
        self.proj_attn = nn.Conv2d(self.k_dim, c, 1, bias=False)
        self.pe = Conv(c, c, 3, g=c, act=False)
        self.proj_out = Conv(c, c, 1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: parallel proj_k / proj_v → (EA + pe(V)) → proj_out."""
        k = self.proj_k(x)
        v = self.proj_v(x)
        attn = self.proj_attn(k.softmax(dim=1))
        return self.proj_out(attn + self.pe(v))


class DLABlock(nn.Module):
    """DLA-compatible analog of PSABlock: DLAAttention + FFN with residual connections.

    Drop-in replacement for `PSABlock`. Structurally identical — `attn + ffn`, both
    wrapped in residuals — with `DLAAttention` substituted for `Attention` so the
    entire block compiles to NVIDIA DLA (no MATRIX_MULTIPLY).
    """

    def __init__(self, c: int, k: int | None = None, shortcut: bool = True) -> None:
        super().__init__()
        self.attn = DLAAttention(c, k)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """attn + ffn with residuals — identical to PSABlock's forward."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2DLA(nn.Module):
    """C2PSA with DLA-compatible spatial mixing (no MatMul/Attention).

    Structurally identical to C2PSA — split input channels, process one half
    through a stack of DLABlocks, then merge. Same init signature as C2PSA for
    substitution in YAML configs.

    Examples:
        >>> c2dla = C2DLA(c1=256, c2=256, n=2, e=0.5)
        >>> x = torch.randn(1, 256, 16, 16)
        >>> y = c2dla(x)  # shape (1, 256, 16, 16), fully DLA-compatible
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5) -> None:
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(DLABlock(self.c) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through split-mix-merge: one half through DLABlocks, then concat."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))
```

For the nano scale the enclosing block sees 256 channels, so `C2DLA` splits into two groups of `C = 128`, the bottleneck is `k_dim = max(128 // 4, 8) = 32`, and one `DLABlock` runs at the 20×20 level. This matches the paper's Section III-B: `φ_k: C→d`, `φ_v: C→C`, `φ_a: d→C`, `φ_o: C→C`, `ψ` a 3×3 depthwise convolution, and an FFN that expands `C → 2C → C`.

> **Two docstrings in the fork contradict the implementation and the paper.** `C2DLA` says it "uses depthwise conv instead of QKV attention", and the model YAML (Section 3 below) says it uses "a 5×5 depthwise conv". The implemented block uses a **3×3** depthwise convolution *and* a channel-axis softmax, which is what the paper describes and what the exported graph contains. Fix the comments before publication; a reviewer reading the code against the paper will find this.

## 2. `ultralytics/nn/modules/head.py`

One new class, appended after `Segment26`. It changes nothing about training: the export path is the only branch that differs, so losses, label assignment and ordinary PyTorch inference keep running through the inherited `Detect`.

```python
class DetectDLA(Detect):
    """Detect head for DLA-only engines: emits one packed 4-D tensor per scale.

    Training and Python-inference paths are unchanged (standard 3-D `Detect` path).
    Export mode emits **3 outputs** — one per scale — with `box` and `score`
    concatenated along the channel axis: `(B, 4·reg_max + nc, H_i, W_i)`. Every
    op from input to engine output stays 4-D and DLA-compatible (channel-axis
    `Concat` on NCHW tensors is supported by DLA).

    DFL decode, dist2bbox, anchor decode, sigmoid, and NMS must be done by
    external code after the engine returns; see `ultralytics/utils/dla.py`.

    The earlier "defer-flatten-until-the-end" approach was structurally unable to
    land the head on DLA: the final `torch.cat(..., dim=2)` produced a 3-D tensor,
    which dragged all upstream 4-D ops back to GPU via ForeignNode fusion.
    """

    def forward(self, x: list[torch.Tensor]):
        """Export: 3 packed 4-D heads (one per scale). Otherwise: standard Detect path."""
        if self.export:
            return self._export_4d(x)
        return super().forward(x)

    def _export_4d(self, feats: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        """Emit one `(B, 4·reg_max + nc, H_i, W_i)` tensor per scale (packed box+score).

        Output order: `(pred_p3, pred_p4, pred_p5)` for `feats` at strides 8, 16, 32.
        """
        box_head = self.one2one_cv2 if self.end2end else self.cv2
        cls_head = self.one2one_cv3 if self.end2end else self.cv3
        out: list[torch.Tensor] = []
        for i in range(self.nl):
            box = box_head[i](feats[i])    # (B, 4*reg_max, H_i, W_i)
            score = cls_head[i](feats[i])  # (B, nc,        H_i, W_i)
            out.append(torch.cat([box, score], dim=1))  # (B, 4*reg_max + nc, H_i, W_i)
        return tuple(out)
```

At batch one, 640×640, `reg_max=16` and `nc=80` this produces the bindings the paper reports: `pred_p3` `1×144×80×80`, `pred_p4` `1×144×40×40`, `pred_p5` `1×144×20×20`.

## 3. `ultralytics/cfg/models/11/yolo11-dla.yaml`

New file. The backbone, neck and detection towers are byte-for-byte the stock YOLO11 ones; exactly two lines differ from `yolo11.yaml` — layer 10 and the head.

```yaml
# Parameters
nc: 80 # number of classes
scales: # model compound scaling constants
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]

backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]] # 9
  - [-1, 2, C2DLA, [1024]] # 10  <-- was C2PSA

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 2, C3k2, [512, False]] # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, C3k2, [256, False]] # 16 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 2, C3k2, [512, False]] # 19 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]] # cat head P5
  - [-1, 2, C3k2, [1024, True]] # 22 (P5/32-large)

  - [[16, 19, 22], 1, DetectDLA, [nc]] # <-- was Detect
```

The filename carries no scale letter, so `guess_model_scale` returns an empty string and `parse_model` falls back to the first entry of `scales`, emitting `no model scale passed. Assuming scale='n'.` That is how the paper's nano model was built, and it yields the 2,615,696 unfused parameters recorded in the checkpoint. If you prefer to pin the scale explicitly, see the optional `tasks.py` change in Section 5.

## 4. `ultralytics/nn/modules/__init__.py`

Export the two public names. Only the entries below are needed; the fork also exports `DFL4D`, `SegmentDLA` and three YOLO-DLA symbols, which are not.

```diff
 from .block import (
     C1,
     C2,
+    C2DLA,
     C2PSA,
@@
 from .head import (
     OBB26,
     Classify,
     Detect,
+    DetectDLA,
     LRPCHead,
@@
 __all__ = (
     "AIFI",
     "C1",
     "C2",
+    "C2DLA",
     "C2PSA",
@@
     "Detect",
+    "DetectDLA",
     "Focus",
```

## 5. `ultralytics/nn/tasks.py`

`parse_model` needs to know the two new modules. `C2DLA` goes in both frozensets that handle repeat-count scaling and channel arguments, exactly where `C2PSA` already appears; `DetectDLA` goes in the head frozensets alongside `Detect`.

```diff
 from ultralytics.nn.modules import (
     AIFI,
     C1,
     C2,
+    C2DLA,
     C2PSA,
@@
     Detect,
+    DetectDLA,
     DWConv,
@@
 def parse_model(d, ch, verbose=True):
@@  # first frozenset: modules taking (c1, c2, *args)
             GhostBottleneck,
             SPP,
             SPPF,
+            C2DLA,
             C2fPSA,
             C2PSA,
@@  # second frozenset: modules whose repeat count scales with depth
             C3x,
             RepC3,
+            C2DLA,
             C2fPSA,
@@  # head modules
         elif m in frozenset(
             {
                 Detect,
+                DetectDLA,
                 WorldDetect,
@@
             args.extend([reg_max, end2end, [ch[x] for x in f]])
@@
             if m in {
                 Detect,
+                DetectDLA,
                 YOLOEDetect,
```

**Optional, and not used for the paper's run.** The fork also relaxes scale resolution so that a `scale:` key inside a YAML survives a filename without a scale letter:

```diff
-    d["scale"] = guess_model_scale(path)
+    d["scale"] = guess_model_scale(path) or d.get("scale")
```

Without it the behaviour is the fallback described in Section 3, which is what actually produced the released checkpoint. Apply it only if you intend to pin the scale in the YAML; it changes which weights you get if you do.

## 6. `ultralytics/utils/dla.py`

New file: the host-side decoder. This is the computation that `DetectDLA` removed from the engine — DFL expectation, anchor decode, `dist2bbox`, sigmoid — plus the glue that runs it automatically when a loaded engine advertises raw DLA outputs.

```python
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""CPU post-processing helpers for raw four-dimensional DLA head outputs."""

from __future__ import annotations

from typing import Any

import torch


def decode_dla_outputs(
    outputs: list[torch.Tensor] | tuple[torch.Tensor, ...],
    strides: list[float] | tuple[float, ...] | torch.Tensor,
    reg_max: int,
    nc: int,
    *,
    nm: int = 0,
    end2end: bool,
    max_det: int = 300,
    agnostic: bool = False,
    device: str | torch.device | None = "cpu",
) -> torch.Tensor:
    """Decode packed DLA P3/P4/P5 maps into the standard Ultralytics prediction format.

    Returns, for one-to-one heads, a ``(B, K, 6 + nm)`` xyxy tensor ready for the
    end-to-end predictor path; otherwise a ``(B, 4 + nc + nm, A)`` xywh tensor
    ready for the standard NMS path.
    """
    from ultralytics.utils.tal import dist2bbox, make_anchors

    if len(outputs) != len(strides):
        raise ValueError(f"Expected {len(strides)} DLA prediction maps, received {len(outputs)}")
    if reg_max < 1:
        raise ValueError(f"reg_max must be positive, received {reg_max}")

    maps = [x.detach().to(device=device, dtype=torch.float32) for x in outputs]
    box_channels = 4 * reg_max
    expected_channels = box_channels + nc + nm
    for i, prediction in enumerate(maps):
        if prediction.ndim != 4 or prediction.shape[1] < expected_channels:
            raise ValueError(
                f"Invalid DLA output {i}: expected (B, >= {expected_channels}, H, W), got {tuple(prediction.shape)}"
            )

    boxes_4d = [x[:, :box_channels] for x in maps]
    scores_4d = [x[:, box_channels : box_channels + nc] for x in maps]
    masks_4d = [x[:, box_channels + nc : expected_channels] for x in maps] if nm else []
    stride_tensor = torch.as_tensor(strides, dtype=torch.float32, device=maps[0].device)
    anchors, stride_per_anchor = (x.transpose(0, 1) for x in make_anchors(boxes_4d, stride_tensor, 0.5))

    batch = maps[0].shape[0]
    boxes = torch.cat([x.reshape(batch, box_channels, -1) for x in boxes_4d], dim=-1)
    scores = torch.cat([x.reshape(batch, nc, -1) for x in scores_4d], dim=-1).sigmoid()
    masks = torch.cat([x.reshape(batch, nm, -1) for x in masks_4d], dim=-1) if nm else None
    if reg_max > 1:
        anchors_count = boxes.shape[-1]
        distribution = boxes.reshape(batch, 4, reg_max, anchors_count).softmax(dim=2)
        projection = torch.arange(reg_max, dtype=distribution.dtype, device=distribution.device).view(1, 1, -1, 1)
        boxes = (distribution * projection).sum(dim=2)

    decoded = dist2bbox(boxes, anchors.unsqueeze(0), xywh=not end2end, dim=1) * stride_per_anchor
    predictions = (
        torch.cat((decoded, scores, masks), dim=1) if masks is not None else torch.cat((decoded, scores), dim=1)
    )
    return _rank_one2one(predictions, nc, max_det, agnostic) if end2end else predictions


def decode_dla_backend_outputs(
    outputs: Any,
    model: Any,
    *,
    max_det: int,
    agnostic: bool = False,
) -> Any:
    """Decode backend outputs only when export metadata identifies a raw DLA head."""
    if not getattr(model, "dla_raw", False):
        return outputs
    if isinstance(outputs, torch.Tensor):
        return outputs  # Already decoded by a task-specific predictor.
    if not isinstance(outputs, (list, tuple)):
        raise TypeError(f"Raw DLA metadata requires a list of output maps, got {type(outputs).__name__}")
    count = int(getattr(model, "dla_outputs", 3))
    nm = int(getattr(model, "nm", 0))
    predictions = decode_dla_outputs(
        outputs[:count],
        getattr(model, "dla_strides", (8.0, 16.0, 32.0)),
        int(getattr(model, "reg_max", 1)),
        len(model.names),
        nm=nm,
        end2end=bool(getattr(model, "end2end", False)),
        max_det=max_det,
        agnostic=agnostic,
    )
    if nm:
        if len(outputs) <= count:
            raise ValueError("Raw DLA segmentation output is missing its prototype tensor")
        return predictions, outputs[count].detach().to(device="cpu", dtype=torch.float32)
    return predictions


def _rank_one2one(predictions: torch.Tensor, nc: int, max_det: int, agnostic: bool) -> torch.Tensor:
    """Apply the Ultralytics one-to-one ranking contract outside the DLA engine."""
    predictions = predictions.transpose(1, 2)
    boxes, scores, extra = predictions.split((4, nc, predictions.shape[-1] - 4 - nc), dim=-1)
    batch, anchors, classes = scores.shape
    k = min(max_det, anchors)

    if agnostic:
        confidence, labels = scores.max(dim=-1, keepdim=True)
        confidence, indices = confidence.topk(k, dim=1)
        labels = labels.gather(dim=1, index=indices)
        boxes = boxes.gather(dim=1, index=indices.expand(-1, -1, 4))
        extra = extra.gather(dim=1, index=indices.expand(-1, -1, extra.shape[-1]))
        return torch.cat((boxes, confidence, labels.float(), extra), dim=-1)

    anchor_indices = scores.amax(dim=-1).topk(k, dim=1).indices.unsqueeze(-1)
    selected_scores = scores.gather(dim=1, index=anchor_indices.expand(-1, -1, classes))
    confidence, class_indices = selected_scores.flatten(1).topk(k, dim=1)
    box_indices = anchor_indices[torch.arange(batch, device=scores.device)[:, None], class_indices // classes]
    boxes = boxes.gather(dim=1, index=box_indices.expand(-1, -1, 4))
    extra = extra.gather(dim=1, index=box_indices.expand(-1, -1, extra.shape[-1]))
    return torch.cat(
        (boxes, confidence.unsqueeze(-1), (class_indices % classes).float().unsqueeze(-1), extra), dim=-1
    )
```

The `nm` branches are only exercised by the excluded segmentation head; they are kept here because removing them would mean rewriting the function rather than reproducing it.

This is the *dense* decoder, and it is the one the paper's CPU verification checks against the inherited `Detect` output (maximum absolute difference 1.22×10⁻⁴). The *confidence-first sparse* decoder used for the deployed pipeline and energy measurements is a separate implementation in the benchmark harness, `tools/experiments/benchmarks/`, not in the fork; the two are compared image by image in the `parity` stage. See [`EXPERIMENTS.md`](EXPERIMENTS.md).

## 7. `ultralytics/engine/exporter.py`

Three changes. Explicit binding names, so the deployment table reads `pred_p3/p4/p5` instead of `output0`/`498`/`511`; export metadata, so a loaded engine can be decoded without out-of-band knowledge; and a bypass of the JetPack 6 INT8 end-to-end TopK workaround, which does not apply to a head that has no in-engine TopK.

```diff
-from ultralytics.nn.modules import C2f, Classify, Detect, RTDETRDecoder, Segment26
+from ultralytics.nn.modules import C2f, Classify, Detect, DetectDLA, RTDETRDecoder, Segment26
@@
-            if fmt == "engine" and self.args.int8:
+            # Raw DLA heads bypass in-engine TopK, so the JetPack 6 TopK workaround does not apply to them.
+            if fmt == "engine" and self.args.int8 and not isinstance(model.model[-1], DetectDLA):
                 # TensorRT 10.3.0 on JetPack 6 with int8 has known end2end build issues
@@
             "end2end": getattr(model, "end2end", False),
         }  # model metadata
+        head = model.model[-1] if hasattr(model, "model") else None
+        if isinstance(head, DetectDLA):
+            self.metadata.update(
+                {
+                    "dla_raw": True,
+                    "dla_outputs": head.nl,
+                    "dla_strides": head.stride.tolist(),
+                    "reg_max": head.reg_max,
+                }
+            )
@@
-        output_names = ["output0", "output1"] if self.model.task == "segment" else ["output0"]
+        # DLA-headless heads emit one packed 4-D tensor per scale. Override the default
+        # 1-or-2-output naming with explicit per-scale names.
+        head = self.model.model[-1] if hasattr(self.model, "model") else None
+        if isinstance(head, DetectDLA):
+            output_names = [f"pred_p{i + 3}" for i in range(head.nl)]
+        else:
+            output_names = ["output0", "output1"] if self.model.task == "segment" else ["output0"]
```

The fork's version of these hunks also branches on `SegmentDLA` and writes an `nm` metadata key; drop both if you exclude segmentation, as this document does.

## 8. `ultralytics/nn/backends/base.py`

Engine metadata is stored as strings. The new keys need the same type coercion the existing ones get, or `dla_strides` comes back as the string `"[8.0, 16.0, 32.0]"` and `reg_max` as `"16"`.

```diff
         for k, v in metadata.items():
-            if k in {"stride", "batch", "channels"}:
+            if k in {"stride", "batch", "channels", "dla_outputs", "reg_max", "nm"}:
                 metadata[k] = int(v)
-            elif k in {"imgsz", "names", "kpt_shape", "kpt_names", "args", "end2end"} and isinstance(v, str):
+            elif k in {
+                "imgsz",
+                "names",
+                "kpt_shape",
+                "kpt_names",
+                "args",
+                "end2end",
+                "dla_raw",
+                "dla_strides",
+            } and isinstance(v, str):
                 metadata[k] = ast.literal_eval(v)
```

## 9. `ultralytics/models/yolo/detect/{predict,val}.py`

Both post-processors call the decoder before NMS. `decode_dla_backend_outputs` is a no-op unless the loaded backend advertises `dla_raw`, so stock models are unaffected. Without this, `YOLO(engine).val()` and `.predict()` receive three raw 4-D maps and fail.

```diff
 # ultralytics/models/yolo/detect/predict.py, in DetectionPredictor.postprocess
+        from ultralytics.utils.dla import decode_dla_backend_outputs
+
+        preds = decode_dla_backend_outputs(
+            preds,
+            self.model,
+            max_det=self.args.max_det,
+            agnostic=self.args.agnostic_nms,
+        )
         save_feats = getattr(self, "_feats", None) is not None
         preds = nms.non_max_suppression(
```

```diff
 # ultralytics/models/yolo/detect/val.py, in DetectionValidator.postprocess
+        from ultralytics.utils.dla import decode_dla_backend_outputs
+
+        preds = decode_dla_backend_outputs(
+            preds,
+            getattr(self, "model", None),
+            max_det=self.args.max_det,
+            agnostic=self.args.single_cls or self.args.agnostic_nms,
+        )
         outputs = nms.non_max_suppression(
```

The fork applies the identical change to `segment/predict.py` and `segment/val.py`; excluded here.

## 10. `ultralytics/utils/nms.py`

A correctness fix, not a feature. `non_max_suppression` detected end-to-end predictions by testing `prediction.shape[-1] == 6`. A raw decoder that leaves exactly six surviving anchors produces an ordinary `(1, 84, 6)` tensor in BCN layout, which that test misreads as six six-column detections in BNC layout. The explicit class count disambiguates.

```diff
-    if prediction.shape[-1] == 6 or end2end:  # end-to-end model (BNC, i.e. 1,300,6)
+    # A raw decoder can leave exactly six anchors in BCN layout. With explicit
+    # nc, (B, 4 + nc, 6) is an ordinary detector output, not six-column BNC.
+    if end2end or (prediction.shape[-1] == 6 and (not nc or prediction.shape[1] != 4 + nc)):
```

This bug and the companion sparse-threshold bug in the benchmark harness were found and fixed while preparing the measurement campaign. Numbers collected before the fixes were not reused for the sparse path; see [`PROVENANCE.md`](PROVENANCE.md).

## Optional: `ultralytics/engine/trainer.py`

Not part of the model. One line adds `"epochs"` to the arguments restored when a run resumes from `last.pt`, so a resumed run keeps its original epoch budget instead of inheriting the value on the command line. It affects how the 500-epoch run was managed, not what the architecture computes.

```diff
                 for k in (
+                    "epochs",
                     "imgsz",
                     "batch",
                     "device",
```

## What is deliberately excluded

Present on the fork branch, not needed for this paper, and listed so the boundary is explicit.

| Excluded | Why |
| --- | --- |
| `ultralytics/nn/modules/yolo_dla.py`, `cfg/models/dla/yolo-dla.yaml`, `YOLO-DLA.md` | the independent from-scratch YOLO-DLA detector family. A different model, out of scope, and none of its results appear in the paper |
| `YOLODLADetect`, `DLARepCSP`, `DLALocalContext` in `__init__.py` and `tasks.py` | the same family's head and blocks |
| `SegmentDLA` in `head.py`, `cfg/models/11/yolo11-dla-seg.yaml`, the segmentation branches of `exporter.py`, and `models/yolo/segment/{predict,val}.py` | the segmentation variant. The paper covers detection only |
| `DFL4D` in `block.py` | **dead code**. It is defined and exported but never referenced: `DetectDLA` removes the DFL from the engine entirely instead of making it 4-D, which is what made strict placement work |
| `examples/train_dla.py` | an optional two-phase warm-start/hot-start recipe with QKV remapping and a frozen-backbone stage. **It was not used for the released checkpoint**, which was a plain full fine-tune from `yolo11n.pt` via `intersect_dicts`. Shipping it next to the model invites the wrong conclusion about how the weights were obtained |
| `examples/test_dla_engine.py` | an older segmentation engine example, not the detection validator used for any paper result |
| `examples/dla_decode.py` | a standalone reference decoder that duplicates `utils/dla.py`. Useful reading, redundant as code |
| `tests/test_yolo_dla.py` | six of its seven tests target the excluded family or the segmentation config; only `test_yolo11_dla_configuration_remains_independent` applies here. Port that one if you want a regression test |
| `ultralytics/utils/callbacks/wb.py` | Weights & Biases run-ID recovery for resumed logging. Training infrastructure, unrelated to the model or the measurements |

## Verifying a port

After applying Sections 1–10 to upstream `a462bb65`:

```bash
python -c "
from ultralytics import YOLO
m = YOLO('ultralytics/cfg/models/11/yolo11-dla.yaml').model
assert type(m.model[10]).__name__ == 'C2DLA'
assert type(m.model[-1]).__name__ == 'DetectDLA'
print(sum(p.numel() for p in m.parameters()), 'parameters')  # expect 2615696
"
```

Then export and check the operator contract: the graph must contain **no** `MatMul`, `Gemm`, `Reshape`, `Transpose`, `Div` or `Slice`, and exactly one `Softmax`. The reference counts for the released model are 89 `Conv`, 76 `Sigmoid`, 76 `Mul`, 9 `Split`, 14 `Add`, 20 `Concat`, 3 `MaxPool`, 2 `Resize`, 1 `Softmax`, with outputs `pred_p3/p4/p5` at `1×144×{80,40,20}²`. `python tools/experiments/run.py check-model` performs exactly this check and writes [`../analysis/model_verification.json`](../analysis/model_verification.json).
