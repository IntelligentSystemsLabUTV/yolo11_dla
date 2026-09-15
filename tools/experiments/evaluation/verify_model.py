#!/usr/bin/env python3
"""Reproduce local CPU structure and decoder checks; does not execute a TensorRT engine."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import WORKSPACE, ARTIFACTS


import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(WORKSPACE / "tools/ultralytics"))

import numpy as np
import onnx
import onnxruntime as ort
import torch
from ultralytics import YOLO
from ultralytics.utils.dla import decode_dla_outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=WORKSPACE / "logs/yolo11n-dla-500ep.pt")
    parser.add_argument("--onnx", type=Path, default=WORKSPACE / "logs/yolo11n-dla-500ep.onnx")
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "model_verification.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = YOLO(str(args.checkpoint)).model.float().eval()
    image = torch.rand(1, 3, 640, 640)
    head = model.model[-1]
    with torch.inference_mode():
        reference = model(image)[0]
        head.export = True
        raw = model(image)
        head.export = False
        decoded = decode_dla_outputs(raw, head.stride, head.reg_max, head.nc, end2end=False)
    torch.testing.assert_close(decoded, reference, rtol=1e-5, atol=1e-4)
    difference = (decoded - reference).abs()

    graph = onnx.load(args.onnx)
    onnx.checker.check_model(graph)
    counts = Counter(node.op_type for node in graph.graph.node)
    assert not any(counts[op] for op in ("MatMul", "Gemm", "Div", "Reshape", "Transpose", "Slice"))
    def dimensions(value):
        return [d.dim_value for d in value.type.tensor_type.shape.dim]
    assert [(v.name, dimensions(v)) for v in graph.graph.output] == [
        ("pred_p3", [1, 144, 80, 80]),
        ("pred_p4", [1, 144, 40, 40]),
        ("pred_p5", [1, 144, 20, 20]),
    ]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.onnx), sess_options=options, providers=["CPUExecutionProvider"])
    ort_outputs = session.run(None, {"images": image.numpy()})
    output_checks = []
    for output, actual, expected in zip(session.get_outputs(), ort_outputs, raw):
        delta = np.abs(actual - expected.numpy())
        # Export fusion and checkpoint FP16 storage introduce small FP32 differences.
        np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-4, atol=1e-3)
        output_checks.append({"name": output.name, "max_abs_difference": float(delta.max()),
                              "mean_abs_difference": float(delta.mean())})
    report = {
        "scope": "CPU verification of trained YOLO11-DLA export and decoder; neither stock-YOLO11 equivalence nor TensorRT/DLA numerical validation",
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "onnx_sha256": hashlib.sha256(args.onnx.read_bytes()).hexdigest(),
        "versions": {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__},
        "input": {"shape": list(image.shape), "generator": "torch.rand", "seed": 7, "threads": 2,
                  "sha256": hashlib.sha256(image.numpy().tobytes()).hexdigest()},
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "decoder": {"passed": True, "rtol": 1e-5, "atol": 1e-4,
                    "shape": list(decoded.shape), "max_abs_difference": float(difference.max()),
                    "mean_abs_difference": float(difference.mean())},
        "onnx": {"passed": True, "opsets": [{"domain": o.domain, "version": o.version} for o in graph.opset_import],
                 "operators": dict(counts), "outputs": [{"name": o.name, "shape": dimensions(o)} for o in graph.graph.output],
                 "raw_output_rtol": 1e-4, "raw_output_atol": 1e-3, "raw_output_comparison": output_checks},
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
