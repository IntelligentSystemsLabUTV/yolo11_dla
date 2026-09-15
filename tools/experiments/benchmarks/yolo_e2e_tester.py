#!/usr/bin/env python3
"""End-to-end TensorRT tester for YOLO11, YOLO11-DLA, and YOLO-DLA.

The tester measures an in-memory image pipeline:

    BGR image -> letterbox/normalize -> TensorRT -> decode/NMS -> scaled boxes

Disk image loading and drawing/saving are intentionally outside the timed region.
The TensorRT stage supports ordinary linear bindings and DLA-native FP16
``DLA_HWC4``/``CHW16`` bindings. By default, raw DLA outputs are copied back and
post-processed on the CPU so a strict-DLA run does not use CUDA for decode/NMS.

Examples (run on the Jetson that built the engines):

    python tools/experiments/run.py e2e \\
        --engine logs/fp16_matrix/yolo11n-gpu-fp16.engine

    python tools/experiments/run.py e2e \\
        --engine logs/fp16_matrix/yolo11n-dla-gpu-fallback-fp16.engine \\
        --dla-core 0

    python tools/experiments/run.py e2e \\
        --engine logs/fp16_matrix/yolo11n-dla-strict-dla-fp16.engine \\
        --dla-core 0 --post-device cpu

For the future constraint-native YOLO-DLA engine, ``--head auto`` recognizes
three 84-channel maps as the one-to-one, ``reg_max=1`` contract. It can also be
selected explicitly with ``--head raw-one2one --reg-max 1``.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import WORKSPACE


import argparse
import datetime as dt
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

try:
    import tensorrt as trt
except ImportError:  # Allows --help and syntax checks away from the Jetson.
    trt = None


TOOLS_DIR = WORKSPACE / "tools"
WORKSPACE_DIR = TOOLS_DIR.parent
LOCAL_ULTRALYTICS = TOOLS_DIR / "ultralytics"
DEFAULT_IMAGE = WORKSPACE_DIR / "logs" / "biascica_simone.jpg"

if LOCAL_ULTRALYTICS.is_dir():
    sys.path.insert(0, str(LOCAL_ULTRALYTICS))


COCO_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)


@dataclass
class TensorMeta:
    """Logical and physical metadata for one TensorRT I/O tensor."""

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    format_name: str
    format_desc: str
    components: int
    vectorized_dim: int
    strides: tuple[int, ...]
    physical_elements: int


@dataclass
class Transform:
    """Letterbox transform needed to project detections to the original image."""

    input_hw: tuple[int, int]
    original_hw: tuple[int, int]
    ratio: float
    pad_xy: tuple[int, int]


def _require_runtime() -> None:
    missing = []
    if trt is None:
        missing.append("TensorRT Python bindings")
    if not torch.cuda.is_available():
        missing.append("CUDA-enabled PyTorch")
    if missing:
        raise RuntimeError("Missing runtime requirement(s): " + ", ".join(missing))


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _format_is(fmt: Any, name: str) -> bool:
    enum_value = getattr(trt.TensorFormat, name, None)
    return enum_value is not None and fmt == enum_value


def _torch_dtype(dtype: Any) -> torch.dtype:
    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT8: torch.int8,
        trt.DataType.INT32: torch.int32,
    }
    if hasattr(trt.DataType, "BOOL"):
        mapping[trt.DataType.BOOL] = torch.bool
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise TypeError(f"Unsupported TensorRT binding dtype: {dtype}") from exc


def _physical_elements(
    shape: tuple[int, ...], fmt: Any, dtype: torch.dtype, components: int, dla_hwc4_line_bytes: int
) -> int:
    """Return required scalar elements for a physical TensorRT binding."""
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"Only static positive binding shapes are supported, received {shape}")
    if _format_is(fmt, "LINEAR"):
        return math.prod(shape)
    if len(shape) != 4:
        raise NotImplementedError(f"Non-linear format {fmt} with non-NCHW shape {shape} is unsupported")

    batch, channels, height, width = shape
    item_size = torch.empty((), dtype=dtype).element_size()
    if any(_format_is(fmt, name) for name in ("CHW2", "CHW4", "CHW16", "CHW32")):
        return batch * math.ceil(channels / components) * height * width * components
    if _format_is(fmt, "DLA_HWC4"):
        padded_channels = 1 if channels == 1 else 4
        row_bytes = _align_up(width * padded_channels * item_size, dla_hwc4_line_bytes)
        return batch * height * (row_bytes // item_size)
    if _format_is(fmt, "DLA_LINEAR"):
        row_bytes = _align_up(width * item_size, 64)
        return batch * channels * height * (row_bytes // item_size)
    if any(_format_is(fmt, name) for name in ("HWC8", "HWC16")):
        padded_channels = math.ceil(channels / components) * components
        return batch * height * width * padded_channels
    raise NotImplementedError(f"Tensor format {fmt} is not implemented")


class TensorRTEngine:
    """TensorRT 10.x runner with explicit packing for DLA-native I/O formats."""

    def __init__(
        self,
        engine_path: Path,
        *,
        device: str = "cuda:0",
        dla_core: int | None = None,
        dla_hwc4_line_bytes: int = 64,
    ) -> None:
        _require_runtime()
        if not engine_path.is_file():
            raise FileNotFoundError(engine_path)
        if dla_hwc4_line_bytes <= 0:
            raise ValueError("--dla-hwc4-line-bytes must be positive")

        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        # TensorRT warns that enqueueV3 on CUDA's default stream inserts extra
        # synchronization. Keep a dedicated stream for engine execution so the
        # measured engine stage reflects the same asynchronous contract used by
        # a production runtime.
        self.stream = torch.cuda.Stream(device=self.device)
        self.dla_hwc4_line_bytes = dla_hwc4_line_bytes
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")
        self.runtime = trt.Runtime(self.logger)
        if dla_core is not None:
            if not hasattr(self.runtime, "DLA_core"):
                raise RuntimeError("This TensorRT Python runtime does not expose Runtime.DLA_core")
            self.runtime.DLA_core = dla_core

        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"TensorRT could not deserialize {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT could not create an execution context")

        self.input_names: list[str] = []
        self.output_names: list[str] = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            (self.input_names if mode == trt.TensorIOMode.INPUT else self.output_names).append(name)
        if len(self.input_names) != 1:
            raise ValueError(f"Expected one image input, found {self.input_names}")

        self.metadata = {name: self._tensor_meta(name) for name in self.input_names + self.output_names}
        self.quantization = None
        int8_names = [name for name, meta in self.metadata.items() if meta.dtype == torch.int8]
        if int8_names:
            from int8.quantization import load_sidecar
            self.quantization = load_sidecar(engine_path, int8_names)
        self.device_buffers: dict[str, torch.Tensor] = {}
        for name, meta in self.metadata.items():
            buffer = torch.empty(meta.physical_elements, dtype=meta.dtype, device=self.device)
            self.device_buffers[name] = buffer
            if not self.context.set_tensor_address(name, buffer.data_ptr()):
                raise RuntimeError(f"Could not bind TensorRT tensor {name}")

    def _tensor_meta(self, name: str) -> TensorMeta:
        shape = tuple(int(x) for x in self.context.get_tensor_shape(name))
        fmt = self.engine.get_tensor_format(name)
        dtype = _torch_dtype(self.engine.get_tensor_dtype(name))
        components = int(self.engine.get_tensor_components_per_element(name))
        vectorized_dim = int(self.engine.get_tensor_vectorized_dim(name))
        try:
            strides = tuple(int(x) for x in self.context.get_tensor_strides(name))
        except Exception:
            strides = ()
        return TensorMeta(
            name=name,
            shape=shape,
            dtype=dtype,
            format_name=str(fmt),
            format_desc=self.engine.get_tensor_format_desc(name),
            components=components,
            vectorized_dim=vectorized_dim,
            strides=strides,
            physical_elements=_physical_elements(
                shape, fmt, dtype, components, self.dla_hwc4_line_bytes
            ),
        )

    @property
    def input_meta(self) -> TensorMeta:
        return self.metadata[self.input_names[0]]

    def describe(self) -> None:
        print("\nEngine bindings:")
        for name in self.input_names + self.output_names:
            meta = self.metadata[name]
            direction = "IN " if name in self.input_names else "OUT"
            print(
                f"  {direction} {name:16s} logical={meta.shape} dtype={meta.dtype} "
                f"format={meta.format_name} physical_elements={meta.physical_elements}"
            )
            print(f"      {meta.format_desc}; strides={meta.strides}")

    def _pack_input(self, tensor: torch.Tensor, meta: TensorMeta) -> torch.Tensor:
        if tensor.device.type != "cpu" or tensor.ndim != 4:
            raise ValueError(f"Input must be a CPU NCHW tensor, received {tensor.device} {tuple(tensor.shape)}")
        if tuple(tensor.shape) != meta.shape:
            raise ValueError(f"Input shape {tuple(tensor.shape)} does not match engine shape {meta.shape}")

        fmt = self.engine.get_tensor_format(meta.name)
        if meta.dtype == torch.int8:
            scale = self.quantization['bindings'][meta.name]['scale']
            source = tensor.float().div(scale).round().clamp(-128, 127).to(torch.int8)
        else:
            source = tensor.to(dtype=meta.dtype)
        if _format_is(fmt, "LINEAR"):
            return source.contiguous().view(-1)

        batch, channels, height, width = meta.shape
        if any(_format_is(fmt, name) for name in ("CHW2", "CHW4", "CHW16", "CHW32")):
            components = meta.components
            tiles = math.ceil(channels / components)
            padded = torch.zeros((batch, tiles * components, height, width), dtype=meta.dtype)
            padded[:, :channels].copy_(source)
            return padded.view(batch, tiles, components, height, width).permute(0, 1, 3, 4, 2).contiguous().view(-1)
        if _format_is(fmt, "DLA_HWC4"):
            padded_channels = 1 if channels == 1 else 4
            item_size = torch.empty((), dtype=meta.dtype).element_size()
            row_bytes = _align_up(width * padded_channels * item_size, self.dla_hwc4_line_bytes)
            row_pixels = row_bytes // item_size // padded_channels
            packed = torch.zeros((batch, height, row_pixels, padded_channels), dtype=meta.dtype)
            packed[:, :, :width, :channels].copy_(source.permute(0, 2, 3, 1))
            return packed.view(-1)
        if _format_is(fmt, "DLA_LINEAR"):
            item_size = torch.empty((), dtype=meta.dtype).element_size()
            row_width = _align_up(width * item_size, 64) // item_size
            packed = torch.zeros((batch, channels, height, row_width), dtype=meta.dtype)
            packed[..., :width].copy_(source)
            return packed.view(-1)
        if any(_format_is(fmt, name) for name in ("HWC8", "HWC16")):
            padded_channels = math.ceil(channels / meta.components) * meta.components
            packed = torch.zeros((batch, height, width, padded_channels), dtype=meta.dtype)
            packed[..., :channels].copy_(source.permute(0, 2, 3, 1))
            return packed.view(-1)
        raise NotImplementedError(f"Input format {meta.format_name} is unsupported")

    def _unpack_output(self, name: str, target_device: torch.device) -> torch.Tensor:
        meta = self.metadata[name]
        fmt = self.engine.get_tensor_format(name)
        flat = self.device_buffers[name] if target_device.type == "cuda" else self.device_buffers[name].cpu()
        if meta.dtype == torch.int8:
            # Scale after D2H, within output_unpack_ms; padding remains zero.
            flat = flat.float() * self.quantization['bindings'][name]['scale']
        if _format_is(fmt, "LINEAR"):
            return flat.view(meta.shape).float()

        if len(meta.shape) != 4:
            raise NotImplementedError(f"Cannot unpack {meta.format_name} with shape {meta.shape}")
        batch, channels, height, width = meta.shape
        if any(_format_is(fmt, item) for item in ("CHW2", "CHW4", "CHW16", "CHW32")):
            components = meta.components
            tiles = math.ceil(channels / components)
            tensor = flat.view(batch, tiles, height, width, components).permute(0, 1, 4, 2, 3).contiguous()
            return tensor.view(batch, tiles * components, height, width)[:, :channels].float()
        if _format_is(fmt, "DLA_LINEAR"):
            item_size = torch.empty((), dtype=meta.dtype).element_size()
            row_width = _align_up(width * item_size, 64) // item_size
            return flat.view(batch, channels, height, row_width)[..., :width].float()
        if any(_format_is(fmt, item) for item in ("HWC8", "HWC16", "DLA_HWC4")):
            padded_channels = 1 if channels == 1 else math.ceil(channels / meta.components) * meta.components
            if _format_is(fmt, "DLA_HWC4"):
                padded_channels = 1 if channels == 1 else 4
                item_size = torch.empty((), dtype=meta.dtype).element_size()
                row_bytes = _align_up(width * padded_channels * item_size, self.dla_hwc4_line_bytes)
                row_width = row_bytes // item_size // padded_channels
            else:
                row_width = width
            tensor = flat.view(batch, height, row_width, padded_channels)[:, :, :width, :channels]
            return tensor.permute(0, 3, 1, 2).contiguous().float()
        raise NotImplementedError(f"Output format {meta.format_name} is unsupported")

    def infer(self, tensor: torch.Tensor, post_device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        """Run one inference and return logical outputs plus stage times in milliseconds."""
        input_name = self.input_names[0]
        input_meta = self.metadata[input_name]

        start = time.perf_counter()
        packed = self._pack_input(tensor, input_meta)
        if packed.numel() != input_meta.physical_elements:
            raise RuntimeError(
                f"Packed input has {packed.numel()} elements; expected {input_meta.physical_elements}"
            )
        self.device_buffers[input_name].copy_(packed, non_blocking=False)
        torch.cuda.synchronize(self.device)
        input_done = time.perf_counter()

        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 returned false")
        self.stream.synchronize()
        engine_done = time.perf_counter()

        outputs = {name: self._unpack_output(name, post_device) for name in self.output_names}
        if post_device.type == "cuda":
            torch.cuda.synchronize(self.device)
        output_done = time.perf_counter()
        return outputs, {
            "input_pack_h2d_ms": (input_done - start) * 1e3,
            "engine_ms": (engine_done - input_done) * 1e3,
            "output_unpack_ms": (output_done - engine_done) * 1e3,
        }


def letterbox(image: np.ndarray, input_hw: tuple[int, int]) -> tuple[np.ndarray, Transform]:
    """Resize and pad using the standard YOLO centered letterbox transform."""
    original_h, original_w = image.shape[:2]
    input_h, input_w = input_hw
    ratio = min(input_h / original_h, input_w / original_w)
    resized_w, resized_h = round(original_w * ratio), round(original_h * ratio)
    pad_w, pad_h = input_w - resized_w, input_h - resized_h
    left, right = round(pad_w / 2 - 0.1), round(pad_w / 2 + 0.1)
    top, bottom = round(pad_h / 2 - 0.1), round(pad_h / 2 + 0.1)
    if (resized_w, resized_h) != (original_w, original_h):
        image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    transform = Transform(
        input_hw=input_hw,
        original_hw=(original_h, original_w),
        ratio=ratio,
        pad_xy=(left, top),
    )
    return image, transform


def preprocess(image: np.ndarray, input_hw: tuple[int, int]) -> tuple[torch.Tensor, Transform]:
    """Letterbox BGR input, convert to normalized RGB NCHW float32 on CPU."""
    padded, transform = letterbox(image, input_hw)
    array = padded[:, :, ::-1].transpose(2, 0, 1)
    array = np.ascontiguousarray(array, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0), transform


def identify_head(outputs: dict[str, torch.Tensor], head: str, nc: int, reg_max: int | None) -> tuple[str, int | None]:
    """Resolve standard, raw-NMS, or raw-one-to-one post-processing."""
    tensors = list(outputs.values())
    if head == "standard":
        return head, None
    if head in {"raw-nms", "raw-one2one"}:
        if reg_max is None:
            channels = tensors[0].shape[1]
            if (channels - nc) < 4 or (channels - nc) % 4:
                raise ValueError(f"Cannot infer reg_max from {channels} output channels and nc={nc}")
            reg_max = (channels - nc) // 4
        return head, reg_max

    if len(tensors) == 1 and tensors[0].ndim == 3:
        return "standard", None
    if len(tensors) == 3 and all(tensor.ndim == 4 for tensor in tensors):
        channels = tensors[0].shape[1]
        if any(tensor.shape[1] != channels for tensor in tensors):
            raise ValueError("Raw pyramid outputs have inconsistent channel counts")
        inferred = reg_max
        if inferred is None:
            if (channels - nc) < 4 or (channels - nc) % 4:
                raise ValueError(f"Cannot infer reg_max from {channels} output channels and nc={nc}")
            inferred = (channels - nc) // 4
        return ("raw-one2one" if inferred == 1 else "raw-nms"), inferred
    raise ValueError(
        "Could not infer head contract. Use --head standard, --head raw-nms, or --head raw-one2one explicitly."
    )


def decode_raw_nms_sparse(
    maps: list[torch.Tensor],
    strides: list[float],
    reg_max: int,
    nc: int,
    conf: float,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Confidence-filter raw anchors before the comparatively expensive DFL decode.

    Class confidence is independent of box regression, so removing anchors whose
    best class is below ``conf`` before DFL is equivalent to dense decode followed
    by the same confidence filter. The returned tensor retains the standard
    ``(B, 4 + nc, A)`` xywh contract expected by Ultralytics NMS.
    """
    decoded_boxes: list[torch.Tensor] = []
    decoded_scores: list[torch.Tensor] = []
    candidate_anchors = 0
    projection = torch.arange(reg_max, dtype=torch.float32, device=device)

    for prediction, stride in zip(maps, strides):
        prediction = prediction.detach().to(device=device, dtype=torch.float32)
        if prediction.shape[0] != 1:
            raise ValueError(f"Sparse raw decode currently requires batch size 1, got {prediction.shape[0]}")

        _, _, _, width = prediction.shape
        box_logits = prediction[0, : 4 * reg_max].reshape(4 * reg_max, -1)
        score_logits = prediction[0, 4 * reg_max : 4 * reg_max + nc].reshape(nc, -1)
        # Sigmoid of the maximum costs one sigmoid per anchor and preserves the
        # dense decoder's floating-point threshold behavior, including ties.
        # Comparing against logit(conf) is equivalent only in exact arithmetic.
        keep = score_logits.amax(dim=0).sigmoid() > conf
        indices = keep.nonzero(as_tuple=False).flatten()
        if not indices.numel():
            continue

        candidate_anchors += int(indices.numel())
        distribution = box_logits[:, indices].transpose(0, 1).reshape(-1, 4, reg_max).softmax(dim=2)
        distances = (distribution * projection.view(1, 1, -1)).sum(dim=2)

        anchor_x = (indices % width).to(torch.float32) + 0.5
        anchor_y = torch.div(indices, width, rounding_mode="floor").to(torch.float32) + 0.5
        left, top, right, bottom = distances.unbind(dim=1)
        x1, y1 = anchor_x - left, anchor_y - top
        x2, y2 = anchor_x + right, anchor_y + bottom
        boxes = torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), dim=0)
        decoded_boxes.append(boxes * stride)
        decoded_scores.append(score_logits[:, indices].sigmoid())

    if not decoded_boxes:
        return torch.empty((1, 4 + nc, 0), dtype=torch.float32, device=device), 0
    prediction = torch.cat((torch.cat(decoded_boxes, dim=1), torch.cat(decoded_scores, dim=1)), dim=0)
    return prediction.unsqueeze(0), candidate_anchors


def postprocess(
    outputs: dict[str, torch.Tensor],
    *,
    head: str,
    reg_max: int | None,
    nc: int,
    conf: float,
    iou: float,
    max_det: int,
    transform: Transform,
    agnostic: bool,
    post_device: torch.device,
    dense_decode: bool = False,
) -> tuple[torch.Tensor, dict[str, int | str]]:
    """Decode/select detections and return CPU results plus candidate counts."""
    from ultralytics.utils.dla import decode_dla_outputs
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import scale_boxes

    tensors = list(outputs.values())
    if head == "standard":
        decode_mode = "standard"
        prediction = tensors[0].to(post_device, dtype=torch.float32)
        expected = 4 + nc
        if prediction.ndim != 3:
            raise ValueError(f"Standard YOLO output must be rank 3, received {tuple(prediction.shape)}")
        if prediction.shape[1] != expected and prediction.shape[2] == expected:
            prediction = prediction.transpose(1, 2).contiguous()
        if prediction.shape[1] != expected:
            raise ValueError(f"Expected standard output with {expected} channels, got {tuple(prediction.shape)}")
        candidate_anchors = int((prediction[:, 4 : 4 + nc].amax(dim=1) > conf).sum().item())
        detections = non_max_suppression(
            prediction, conf_thres=conf, iou_thres=iou, agnostic=agnostic, max_det=max_det, nc=nc
        )[0]
    else:
        if reg_max is None:
            raise RuntimeError("Raw head requires reg_max")
        maps = sorted(tensors, key=lambda tensor: tensor.shape[-2] * tensor.shape[-1], reverse=True)
        channels = 4 * reg_max + nc
        if any(tensor.shape[1] != channels for tensor in maps):
            raise ValueError(f"Expected raw maps with {channels} channels, got {[tuple(x.shape) for x in maps]}")
        strides = [transform.input_hw[1] / tensor.shape[-1] for tensor in maps]
        one2one = head == "raw-one2one"
        if one2one or dense_decode:
            decode_mode = "dense"
            prediction = decode_dla_outputs(
                maps,
                strides,
                reg_max,
                nc,
                end2end=one2one,
                max_det=max_det,
                agnostic=agnostic,
                device=post_device,
            )
            candidate_anchors = int(
                (prediction[..., 4] > conf).sum().item()
                if one2one
                else (prediction[:, 4 : 4 + nc].amax(dim=1) > conf).sum().item()
            )
        else:
            decode_mode = "sparse"
            prediction, candidate_anchors = decode_raw_nms_sparse(
                maps, strides, reg_max, nc, conf, post_device
            )
        detections = non_max_suppression(
            prediction,
            conf_thres=conf,
            iou_thres=iou,
            agnostic=agnostic,
            max_det=max_det,
            nc=nc,
            end2end=one2one,
        )[0]

    if detections.numel():
        detections = detections.clone()
        ratio_pad = ((transform.ratio, transform.ratio), transform.pad_xy)
        detections[:, :4] = scale_boxes(
            transform.input_hw, detections[:, :4], transform.original_hw, ratio_pad=ratio_pad
        )
    return detections.detach().cpu(), {"candidate_anchors": candidate_anchors, "decode_mode": decode_mode}


def draw_detections(image: np.ndarray, detections: torch.Tensor, names: tuple[str, ...]) -> np.ndarray:
    """Draw detections on a copy of the original BGR image."""
    output = image.copy()
    rng = np.random.default_rng(42)
    colors = rng.integers(40, 240, size=(max(len(names), 1), 3), dtype=np.uint8)
    for x1, y1, x2, y2, confidence, class_id, *_ in detections.tolist():
        class_index = int(class_id)
        color = tuple(int(x) for x in colors[class_index % len(colors)])
        label_name = names[class_index] if 0 <= class_index < len(names) else str(class_index)
        label = f"{label_name} {confidence:.2f}"
        p1, p2 = (round(x1), round(y1)), (round(x2), round(y2))
        cv2.rectangle(output, p1, p2, color, 2, cv2.LINE_AA)
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_y = max(p1[1], text_h + baseline + 2)
        cv2.rectangle(output, (p1[0], label_y - text_h - baseline - 2), (p1[0] + text_w, label_y), color, -1)
        cv2.putText(output, label, (p1[0], label_y - baseline - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return output


def summarize(samples: list[float]) -> dict[str, float]:
    array = np.asarray(samples, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "min": float(array.min()),
        "max": float(array.max()),
        "stdev": float(statistics.pstdev(samples)),
    }


def print_timings(summaries: dict[str, dict[str, float]], runs: int) -> None:
    print(f"\nTimed in-memory pipeline over {runs} runs (image file read and drawing excluded):")
    print(f"  {'stage':24s} {'mean':>9s} {'median':>9s} {'p95':>9s} {'p99':>9s} {'min':>9s} {'max':>9s}")
    for name, stats in summaries.items():
        print(
            f"  {name:24s} {stats['mean']:9.3f} {stats['median']:9.3f} {stats['p95']:9.3f} "
            f"{stats['p99']:9.3f} {stats['min']:9.3f} {stats['max']:9.3f}"
        )
    print(f"\n  Sequential E2E rate from median total: {1000.0 / summaries['total_ms']['median']:.2f} images/s")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", type=Path, required=True, help="TensorRT .engine built on this Jetson")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help=f"Input image (default: {DEFAULT_IMAGE})")
    parser.add_argument("--output", type=Path, help="Annotated output image; defaults beside the input image")
    parser.add_argument("--timings-json", type=Path, help="Timing JSON; defaults beside the annotated output")
    parser.add_argument("--device", default="cuda:0", help="CUDA device used for TensorRT buffers")
    parser.add_argument("--cpu-threads", type=int, default=1, help="PyTorch/OpenCV CPU threads used by the pipeline")
    parser.add_argument("--dla-core", type=int, help="Set TensorRT runtime DLA core, normally 0")
    parser.add_argument(
        "--dla-hwc4-line-bytes",
        type=int,
        default=64,
        help="DLA_HWC4 row alignment in bytes; NVIDIA Orin uses 64",
    )
    parser.add_argument("--post-device", choices=("cpu", "cuda"), default="cpu", help="Decode/NMS device")
    parser.add_argument(
        "--head",
        choices=("auto", "standard", "raw-nms", "raw-one2one"),
        default="auto",
        help="Output contract; auto handles the paper's three detector variants",
    )
    parser.add_argument("--reg-max", type=int, help="DFL bins for raw heads; inferred from channels by default")
    parser.add_argument("--nc", type=int, default=80, help="Number of detection classes")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections")
    parser.add_argument("--agnostic-nms", action="store_true", help="Use class-agnostic NMS/ranking")
    parser.add_argument(
        "--dense-decode",
        action="store_true",
        help="Decode DFL for every raw anchor (reference/audit mode); default raw-NMS decode is confidence-first",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Untimed full-pipeline warm-up runs")
    parser.add_argument("--runs", type=int, default=100, help="Timed full-pipeline runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.runs < 1:
        raise ValueError("--warmup must be non-negative and --runs must be positive")
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be positive")
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        raise ValueError("--conf and --iou must lie in [0, 1]")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(args.cpu_threads)
    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    runner = TensorRTEngine(
        args.engine,
        device=args.device,
        dla_core=args.dla_core,
        dla_hwc4_line_bytes=args.dla_hwc4_line_bytes,
    )
    runner.describe()
    input_shape = runner.input_meta.shape
    if len(input_shape) != 4 or input_shape[0] != 1 or input_shape[1] != 3:
        raise ValueError(f"Expected static 1x3xHxW image input, received {input_shape}")
    input_hw = (input_shape[2], input_shape[3])
    post_device = torch.device(args.device if args.post_device == "cuda" else "cpu")

    first_tensor, first_transform = preprocess(image, input_hw)
    first_outputs, _ = runner.infer(first_tensor, post_device)
    resolved_head, resolved_reg_max = identify_head(first_outputs, args.head, args.nc, args.reg_max)
    decode_mode = "dense" if args.dense_decode or resolved_head == "raw-one2one" else "sparse"
    if resolved_head == "standard":
        decode_mode = "standard"
    print(
        f"\nPost-processing contract: head={resolved_head}, reg_max={resolved_reg_max}, "
        f"device={post_device}, decode={decode_mode}"
    )
    print(f"Input image: {args.image} ({image.shape[1]}x{image.shape[0]}); engine input: {input_hw[1]}x{input_hw[0]}")

    stage_samples: dict[str, list[float]] = {
        "preprocess_ms": [],
        "input_pack_h2d_ms": [],
        "engine_ms": [],
        "output_unpack_ms": [],
        "postprocess_ms": [],
        "total_ms": [],
    }
    last_detections = torch.empty((0, 6), dtype=torch.float32)
    last_postprocess_stats: dict[str, int | str] = {}
    total_iterations = args.warmup + args.runs
    with torch.inference_mode():
        for iteration in range(total_iterations):
            total_start = time.perf_counter()
            preprocess_start = total_start
            input_tensor, transform = preprocess(image, input_hw)
            preprocess_end = time.perf_counter()

            outputs, inference_times = runner.infer(input_tensor, post_device)
            postprocess_start = time.perf_counter()
            detections, postprocess_stats = postprocess(
                outputs,
                head=resolved_head,
                reg_max=resolved_reg_max,
                nc=args.nc,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                transform=transform,
                agnostic=args.agnostic_nms,
                post_device=post_device,
                dense_decode=args.dense_decode,
            )
            total_end = time.perf_counter()
            last_detections = detections
            last_postprocess_stats = postprocess_stats

            if iteration >= args.warmup:
                stage_samples["preprocess_ms"].append((preprocess_end - preprocess_start) * 1e3)
                for name, value in inference_times.items():
                    stage_samples[name].append(value)
                stage_samples["postprocess_ms"].append((total_end - postprocess_start) * 1e3)
                stage_samples["total_ms"].append((total_end - total_start) * 1e3)

    summaries = {name: summarize(values) for name, values in stage_samples.items()}
    print_timings(summaries, args.runs)
    print(f"\nDetections at conf={args.conf:.2f}: {len(last_detections)}")
    print(f"Pre-NMS/ranking candidate anchors: {last_postprocess_stats['candidate_anchors']}")
    for detection in last_detections[:20].tolist():
        x1, y1, x2, y2, confidence, class_id = detection[:6]
        class_index = int(class_id)
        name = COCO_NAMES[class_index] if class_index < len(COCO_NAMES) else str(class_index)
        print(f"  {name:16s} conf={confidence:.3f} box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")

    output_path = args.output or args.image.with_name(f"{args.image.stem}_{args.engine.stem}_e2e.jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated = draw_detections(image, last_detections, COCO_NAMES[: args.nc])
    if not cv2.imwrite(str(output_path), annotated):
        raise RuntimeError(f"Could not write annotated image: {output_path}")

    timings_path = args.timings_json or output_path.with_suffix(".json")
    timings_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": sys.argv,
        "engine": str(args.engine.resolve()),
        "engine_sha256": sha256(args.engine),
        "image": str(args.image.resolve()),
        "image_sha256": sha256(args.image),
        "annotated_output": str(output_path.resolve()),
        "tensor_rt_version": trt.__version__,
        "torch_version": torch.__version__,
        "opencv_version": cv2.__version__,
        "platform": platform.platform(),
        "cuda_device_name": torch.cuda.get_device_name(torch.device(args.device)),
        "cuda_compute_capability": torch.cuda.get_device_capability(torch.device(args.device)),
        "device": args.device,
        "dla_core": args.dla_core,
        "post_device": args.post_device,
        "dense_decode": args.dense_decode,
        "cpu_threads": args.cpu_threads,
        "head": resolved_head,
        "reg_max": resolved_reg_max,
        "nc": args.nc,
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "warmup_runs": args.warmup,
        "timed_runs": args.runs,
        "disk_read_and_drawing_timed": False,
        "transform": asdict(first_transform),
        "bindings": {name: asdict(meta) | {"dtype": str(meta.dtype)} for name, meta in runner.metadata.items()},
        "detections": int(len(last_detections)),
        "postprocess_stats": last_postprocess_stats,
        "detection_records": [
            {
                "xyxy": [float(value) for value in detection[:4]],
                "confidence": float(detection[4]),
                "class_id": int(detection[5]),
                "class_name": COCO_NAMES[int(detection[5])] if int(detection[5]) < len(COCO_NAMES) else None,
            }
            for detection in last_detections.tolist()
        ],
        "timings_ms": summaries,
        "timing_samples_ms": stage_samples,
    }
    timings_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved annotated image: {output_path}")
    print(f"Saved timing report:  {timings_path}")


if __name__ == "__main__":
    main()
