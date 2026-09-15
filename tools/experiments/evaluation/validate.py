#!/usr/bin/env python3
"""Target-side native-layout audit, same-model GPU/DLA raw errors and dense/sparse parity."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import configure_imports
configure_imports()

import argparse
from dataclasses import asdict
from pathlib import Path
import math

from common import image_sequence, new_directory, provenance, sha256, write_json


def error_stats(reference, actual, atol, rtol):
    import torch
    if reference.shape != actual.shape:
        return {'passed': False, 'reference_shape': list(reference.shape), 'actual_shape': list(actual.shape)}
    if not torch.isfinite(reference).all() or not torch.isfinite(actual).all():
        return {'passed': False, 'error': 'non-finite values'}
    diff = (reference-actual).abs()
    if not diff.numel():
        return {'passed': True, 'elements': 0}
    relative = diff/reference.abs().clamp_min(1e-6)
    return {'passed': bool(torch.all(diff <= atol+rtol*reference.abs())),
            'elements': diff.numel(), 'max_abs': diff.max().item(), 'mean_abs': diff.mean().item(),
            'max_relative_floor_1e6': relative.max().item(), 'mean_relative_floor_1e6': relative.mean().item(),
            'outside_tolerance': int((diff > atol+rtol*reference.abs()).sum())}


def reference_storage(meta, logical):
    """Independent scalar-offset oracle; includes physical padding, not pack/unpack roundtrip alone."""
    import numpy as np
    shape = meta.shape
    fmt = meta.format_name.split('.')[-1]
    array = logical.numpy()
    if fmt == 'LINEAR':
        return array.reshape(-1).copy()
    n, c, h, w = np.indices(shape, sparse=True)
    _, channels, height, width = shape
    if fmt in ('CHW2','CHW4','CHW16','CHW32'):
        v = meta.components
        offsets = ((((n*math.ceil(channels/v)+c//v)*height+h)*width+w)*v+c%v)
    elif fmt == 'DLA_HWC4':
        cp = 1 if channels == 1 else 4
        row = math.ceil(width*cp*array.itemsize/64)*64//array.itemsize
        offsets = (n*height+h)*row+w*cp+c
    elif fmt == 'DLA_LINEAR':
        row = math.ceil(width*array.itemsize/64)*64//array.itemsize
        offsets = ((n*channels+c)*height+h)*row+w
    else:
        raise ValueError(f'Format outside this audit protocol: {fmt}')
    expected = np.zeros(meta.physical_elements, dtype=array.dtype)
    if int(offsets.max()) >= len(expected):
        raise ValueError('Reported buffer capacity is insufficient')
    expected[offsets.reshape(-1)] = array.reshape(-1)
    return expected


def check_layout(runner):
    import numpy as np
    import torch
    records = {}
    for name, meta in runner.metadata.items():
        if len(meta.shape) != 4:
            raise ValueError('Raw-engine layout audit requires rank-four bindings')
        n,c,h,w = np.indices(meta.shape, sparse=True)
        # Integers representable exactly in FP16, varying with channel and both coordinates.
        codes = ((n*977+c*37+h*11+w*3) % 1021).astype(np.float32)
        logical = torch.from_numpy(codes).to(meta.dtype)
        if meta.dtype == torch.int8:
            logical = torch.from_numpy(((codes % 255)-128).astype(np.int8))
        floating = logical.float()
        if meta.dtype == torch.int8:
            floating = floating * runner.quantization['bindings'][name]['scale']
        expected = torch.from_numpy(reference_storage(meta, logical))
        packed = runner._pack_input(floating, meta)
        torch.testing.assert_close(packed, expected, rtol=0, atol=0)
        # Test the actual unpacker using independently arranged physical storage.
        saved = runner.device_buffers[name]
        try:
            runner.device_buffers[name] = expected
            unpacked = runner._unpack_output(name, torch.device('cpu'))
            torch.testing.assert_close(unpacked, floating, rtol=0, atol=0)
        finally:
            runner.device_buffers[name] = saved
        records[name] = {'passed': True, 'metadata': asdict(meta),
                         'physical_bytes': expected.numel()*expected.element_size(),
                         'scope': 'CPU format oracle; GPU/DLA raw comparison separately tests engine interpretation'}
    return records


def decoder_parity(maps, conf, multi_label):
    import torch
    from ultralytics.utils.dla import decode_dla_outputs
    from ultralytics.utils.nms import non_max_suppression
    from yolo_e2e_tester import decode_raw_nms_sparse
    strides = [640/t.shape[-1] for t in maps]
    dense = decode_dla_outputs(maps, strides, 16, 80, end2end=False, device='cpu')
    keep = dense[:,4:].amax(1)[0] > conf
    selected = dense[:,:,keep]
    sparse, count = decode_raw_nms_sparse(maps, strides, 16, 80, conf, torch.device('cpu'))
    before = error_stats(selected, sparse, 1e-3, 1e-5)
    result = {'conf': conf, 'multi_label': multi_label, 'dense_candidates': int(keep.sum()),
              'sparse_candidates': count, 'before_nms': before,
              'scores_within_1e6_of_threshold': int(((dense[:,4:]-conf).abs() <= 1e-6).sum())}
    detections = []
    for prediction in (dense, sparse):
        detections.append(non_max_suppression(prediction.clone(), conf_thres=conf, iou_thres=0.7 if multi_label else 0.45,
                                             nc=80, multi_label=multi_label, max_det=300, max_time_img=60)[0])
    result['after_nms'] = error_stats(detections[0], detections[1], 1e-3, 1e-5)
    result['passed'] = before['passed'] and result['after_nms']['passed']
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu-engine', type=Path, required=True)
    p.add_argument('--dla-engine', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, default=50)
    p.add_argument('--cpu-threads', type=int, default=1)
    p.add_argument('--raw-atol', type=float, default=0.1)
    p.add_argument('--raw-rtol', type=float, default=0.05)
    p.add_argument('--raw-diagnostic-only', action='store_true',
                   help='Record cross-engine raw tolerance failures without using them as a gate; layout/decoder/finite checks still gate')
    args = p.parse_args()
    if args.limit < 1 or args.cpu_threads < 1 or args.raw_atol < 0 or args.raw_rtol < 0:
        p.error('Invalid limit, threads or tolerance')
    import cv2
    import torch
    from ultralytics.utils.dla import decode_dla_outputs
    from ultralytics.utils.nms import non_max_suppression
    from yolo_e2e_tester import TensorRTEngine, preprocess
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(args.cpu_threads)
    out = new_directory(args.output)
    report = {'status': 'running', 'arguments': vars(args), 'provenance': provenance(),
              'gpu_sha256': sha256(args.gpu_engine), 'dla_sha256': sha256(args.dla_engine),
              'raw_tolerance_note': 'Exploratory gate fixed before run; not an AP acceptance criterion',
              'frames': []}
    write_json(out/'report.json', report)
    try:
        sequence = image_sequence(args.images, args.annotations, args.limit)
        write_json(out/'sequence.json', sequence)
        runners = {'gpu': TensorRTEngine(args.gpu_engine), 'dla': TensorRTEngine(args.dla_engine, dla_core=0)}
        report['quantization'] = {k: r.quantization for k,r in runners.items()}
        report['layouts'] = {k: check_layout(r) for k,r in runners.items()}
        passed = True
        with torch.inference_mode():
            for entry in sequence:
                img = cv2.imread(entry['path'])
                if img is None:
                    raise ValueError(f"Cannot decode {entry['path']}")
                x, _ = preprocess(img, (640,640))
                outputs = {}
                for name, runner in runners.items():
                    raw, _ = runner.infer(x, torch.device('cpu'))
                    outputs[name] = sorted(raw.values(), key=lambda t: t.shape[-1], reverse=True)
                    if [tuple(t.shape) for t in outputs[name]] != [(1,144,s,s) for s in (80,40,20)]:
                        raise ValueError('Expected YOLO11-DLA raw output shapes')
                raw_errors = [error_stats(g,d,args.raw_atol,args.raw_rtol)
                              for g,d in zip(outputs['gpu'],outputs['dla'])]
                decoded = {k: decode_dla_outputs(v,[8,16,32],16,80,end2end=False,device='cpu')
                           for k,v in outputs.items()}
                parity = {k: [decoder_parity(v,conf,multi) for conf,multi in [(0.001,True),(0.25,False)]]
                          for k,v in outputs.items()}
                detections = {k: non_max_suppression(v.clone(), conf_thres=0.25, iou_thres=0.45, nc=80,
                                                     max_det=300, max_time_img=60)[0].tolist()
                              for k,v in decoded.items()}
                record = {'image_id': entry['image_id'], 'raw_scale_errors': raw_errors,
                          'decoded_box_error_px': error_stats(decoded['gpu'][:,:4],decoded['dla'][:,:4],1,0),
                          'class_score_error': error_stats(decoded['gpu'][:,4:],decoded['dla'][:,4:],0.01,0),
                          'dense_sparse': parity, 'post_nms_detections': detections,
                          'note': 'Cross-engine post-NMS lists retained for inspection; no identical-box assumption'}
                report['frames'].append(record)
                finite = all(torch.isfinite(t).all().item() for maps in outputs.values() for t in maps)
                passed &= finite and (args.raw_diagnostic_only or all(e['passed'] for e in raw_errors)) and all(e['passed'] for v in parity.values() for e in v)
                write_json(out/'report.json', report)
        report.update(status='complete', passed=passed)
        write_json(out/'report.json', report)
        if not passed:
            raise RuntimeError('Numerical or decoder gate failed; investigate saved per-image results before AP/timing')
    except BaseException as exc:
        report.update(status='failed', error=str(exc))
        write_json(out/'report.json', report)
        raise


if __name__ == '__main__':
    main()
