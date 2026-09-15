#!/usr/bin/env python3
"""Export the stock YOLO11n and proposed YOLO11-DLA checkpoints."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import configure_imports
configure_imports()

import argparse
from pathlib import Path
import shutil

from common import new_directory, provenance, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stock', type=Path, required=True)
    parser.add_argument('--adapted', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hardware-notes', type=Path, required=True,
                        help='Module/carrier, power mode, clocks, cooling and dataset/training provenance')
    args = parser.parse_args()
    for path in (args.stock, args.adapted, args.hardware_notes):
        if not path.is_file():
            raise FileNotFoundError(path)
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.modules import C2DLA, C2PSA, Detect, DetectDLA
    import onnx

    torch.set_num_threads(2)
    out = new_directory(args.output)
    manifest = {'status': 'running', 'provenance': provenance(), 'artifacts': {}}
    shutil.copy2(args.hardware_notes, out/'hardware-notes.md')
    write_json(out/'manifest.json', manifest)
    for label, source, mixer, head in [('stock', args.stock, C2PSA, Detect),
                                       ('adapted', args.adapted, C2DLA, DetectDLA)]:
        target = out/f'{label}.pt'
        shutil.copy2(source, target)
        original = YOLO(str(target))
        if type(original.model.model[10]) is not mixer or type(original.model.model[-1]) is not head:
            raise ValueError(f'{label}: expected YOLO11 nano {mixer.__name__}/{head.__name__}')
        if original.model.model[-1].nc != 80 or original.model.model[-1].reg_max != 16:
            raise ValueError('Expected COCO detection with reg_max=16')
        if original.model.model[10].cv1.conv.in_channels != 256:
            raise ValueError('This protocol is for the nano scale only')
        parameters = sum(p.numel() for p in original.model.parameters())
        exported = Path(original.export(format='onnx', imgsz=640, batch=1, dynamic=False,
                                        simplify=False, opset=20, half=False, nms=False, device='cpu'))
        graph = onnx.load(exported)
        onnx.checker.check_model(graph)
        shapes = [[d.dim_value for d in o.type.tensor_type.shape.dim] for o in graph.graph.output]
        expected = [[1,144,s,s] for s in (80,40,20)] if head is DetectDLA else [[1,84,8400]]
        if shapes != expected:
            raise ValueError(f'{label}: unexpected ONNX outputs {shapes}')
        manifest['artifacts'][label] = {
            'checkpoint_sha256': sha256(target), 'source_checkpoint': str(source.resolve()),
            'source_checkpoint_sha256': sha256(source), 'onnx_sha256': sha256(exported),
            'mixer': mixer.__name__, 'head': head.__name__, 'outputs': shapes,
            'parameters': parameters, 'interpretation': 'main model'}
        write_json(out/'manifest.json', manifest)
    manifest['status'] = 'complete'
    write_json(out/'manifest.json', manifest)


if __name__ == '__main__':
    main()
