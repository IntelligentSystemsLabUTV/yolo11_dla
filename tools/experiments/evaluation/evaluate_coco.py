#!/usr/bin/env python3
"""Matched batch-one COCO evaluation via PyTorch or the existing native-layout TRT runner.

The TensorRT branch must be validated on the target Jetson before reporting AP.
This script defaults to dense decoding and multi-label NMS for accuracy, not the
confidence-0.25 sparse visual-demo protocol. File reads are outside service timing.
The optional sparse engine path retains the same low-confidence AP settings.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import ROOT, WORKSPACE

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path[:0] = [str(ROOT / 'benchmarks'), str(WORKSPACE / 'tools/ultralytics')]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', type=Path)
    source.add_argument('--engine', type=Path)
    p.add_argument('--images', type=Path, required=True, help='COCO val2017 image directory')
    p.add_argument('--annotations', type=Path, required=True, help='instances_val2017.json')
    p.add_argument('--output', type=Path, required=True, help='New output directory; refuses overwrites')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dla-core', type=int)
    p.add_argument('--limit', type=int, help='Smoke-test subset only; omit for final 5000-image AP')
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--cpu-threads', type=int, default=1)
    p.add_argument('--conf', type=float, default=0.001)
    p.add_argument('--iou', type=float, default=0.7)
    p.add_argument('--max-det', type=int, default=300)
    p.add_argument('--sparse-decode', action='store_true',
                   help='Raw engine only: validate sparse decoder AP with matched low-confidence settings')
    return p.parse_args()


def main():
    args = parse_args()
    if args.warmup < 0 or args.cpu_threads < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError('warmup >= 0, threads >= 1, limit >= 1 required')
    if not 0 < args.conf < 1 or not 0 < args.iou < 1 or args.max_det < 1:
        raise ValueError('Require confidence/IoU in (0,1), max-det >= 1')
    if args.dla_core is not None and args.engine is None:
        raise ValueError('--dla-core requires --engine')
    if args.sparse_decode and args.engine is None:
        raise ValueError('--sparse-decode requires a raw YOLO11-DLA engine')
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    import cv2
    import numpy as np
    import torch
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from ultralytics import YOLO
    from ultralytics.data.converter import coco80_to_coco91_class
    from ultralytics.utils.dla import decode_dla_outputs
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import scale_boxes
    from yolo_e2e_tester import TensorRTEngine, decode_raw_nms_sparse, identify_head, preprocess

    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(args.cpu_threads)
    coco = COCO(str(args.annotations))
    image_ids = sorted(coco.getImgIds())
    if args.limit is not None:
        image_ids = image_ids[:args.limit]
    if not image_ids:
        raise ValueError('Annotation file contains no selected images')
    category_map = coco80_to_coco91_class()
    if sorted(coco.getCatIds()) != sorted(category_map):
        raise ValueError('Expected the official 80 COCO category IDs')
    for image_id in image_ids:
        path = args.images / coco.imgs[image_id]['file_name']
        if not path.is_file():
            raise FileNotFoundError(path)
    runner = None
    model = None
    if args.engine:
        runner = TensorRTEngine(args.engine, device=args.device, dla_core=args.dla_core)
        runner.describe()
        if runner.input_meta.shape != (1, 3, 640, 640):
            raise ValueError('This protocol certifies only static 1x3x640x640 engines')
    else:
        model = YOLO(str(args.checkpoint)).model.float().to(args.device).eval()
        if model.model[-1].nc != 80:
            raise ValueError('Expected an 80-class detection checkpoint')
    args.output.mkdir(parents=True)
    manifest = {
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'artifact_sha256': sha256(args.engine or args.checkpoint),
        'annotation_sha256': sha256(args.annotations),
        'image_ids': image_ids,
        'full_coco_val2017_candidate': args.limit is None and len(image_ids) == 5000,
        'preprocess': 'BGR->RGB, square centered letterbox640, /255; batch1',
        'postprocess': f'CPU {"sparse" if args.sparse_decode else "dense"} decode, multi_label=True, class-aware NMS, max_nms=30000',
        'script_sha256': sha256(Path(__file__)),
        'runtime_script_sha256': sha256(ROOT/'benchmarks/yolo_e2e_tester.py'),
        'timing': 'In-memory sequential service only; excludes file I/O, JSON writing, COCO evaluation; accuracy settings, not live camera latency',
        'versions': {'torch': torch.__version__, 'opencv': cv2.__version__},
        'bindings': {name: asdict(meta) for name, meta in runner.metadata.items()} if runner else None,
        'quantization': runner.quantization if runner else None,
        'status': 'running',
    }
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2, default=str)+'\n')

    def evaluate_image(image):
        start = time.perf_counter()
        x, transform = preprocess(image, (640, 640))
        prep_end = time.perf_counter()
        if runner:
            outputs, stages = runner.infer(x, torch.device('cpu'))
            infer_end = time.perf_counter()
            head, reg_max = identify_head(outputs, 'auto', 80, None)
            if head == 'standard':
                if args.sparse_decode:
                    raise ValueError('--sparse-decode requires raw outputs')
                prediction = next(iter(outputs.values())).float()
                if prediction.shape[1] != 84 and prediction.shape[2] == 84:
                    prediction = prediction.transpose(1, 2).contiguous()
            elif head == 'raw-nms':
                maps = sorted(outputs.values(), key=lambda t: t.shape[-1]*t.shape[-2], reverse=True)
                strides = [640/t.shape[-1] for t in maps]
                if args.sparse_decode:
                    prediction, _ = decode_raw_nms_sparse(maps, strides, reg_max, 80,
                                                         args.conf, torch.device('cpu'))
                else:
                    prediction = decode_dla_outputs(maps, strides, reg_max, 80,
                                                    end2end=False, device='cpu')
            else:
                raise ValueError('This evaluation covers standard/raw-nms YOLO11 only')
            # Dense decode is included in postprocess time below.
        else:
            x = x.to(args.device)
            result = model(x)
            prediction = result[0] if isinstance(result, tuple) else result
            prediction = prediction.detach().float().cpu()
            infer_end = time.perf_counter()
            stages = {'model_and_transfer_ms': (infer_end - prep_end)*1000}
        if prediction.ndim != 3 or prediction.shape[1] != 84:
            raise ValueError(f'Expected decoded (1,84,N), got {tuple(prediction.shape)}')
        detections = non_max_suppression(prediction, conf_thres=args.conf, iou_thres=args.iou,
                                        multi_label=True, max_det=args.max_det, nc=80,
                                        max_time_img=60.0, max_nms=30000)[0]
        if len(detections):
            detections = detections.clone()
            detections[:, :4] = scale_boxes((640,640), detections[:, :4], transform.original_hw,
                                            ratio_pad=((transform.ratio,transform.ratio),transform.pad_xy))
        end = time.perf_counter()
        return detections.cpu(), {
            'preprocess_ms': (prep_end-start)*1000, **stages,
            'postprocess_ms': (end-infer_end)*1000, 'total_ms': (end-start)*1000,
            'monotonic_start_s': start, 'monotonic_end_s': end,
        }

    predictions = []
    durations = []
    with torch.inference_mode(), (args.output/'frames.jsonl').open('w') as records:
        for index in range(-args.warmup, len(image_ids)):
            image_id = image_ids[0] if index < 0 else image_ids[index]
            image_path = args.images/coco.imgs[image_id]['file_name']
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f'Image decode failed: {image_path}')
            detections, timing = evaluate_image(image)
            if index < 0:
                continue
            image_predictions = []
            for row in detections.tolist():
                x1,y1,x2,y2,score,cls = row
                item = {'image_id': image_id, 'category_id': category_map[int(cls)],
                        'bbox': [x1,y1,x2-x1,y2-y1], 'score': score}
                image_predictions.append(item)
            predictions.extend(image_predictions)
            records.write(json.dumps({'image_id':image_id, 'image_sha256':sha256(image_path),
                                      'detections':image_predictions, **timing})+'\n')
            records.flush()
            durations.append(timing['total_ms'])
            if (index+1)%100 == 0:
                print(f'{index+1}/{len(image_ids)} images', flush=True)
    (args.output/'predictions.json').write_text(json.dumps(predictions)+'\n')
    if predictions:
        detected = coco.loadRes(predictions)
    else:
        detected = COCO()
        detected.dataset = {'images': coco.dataset['images'], 'categories': coco.dataset['categories'],
                            'annotations': []}
        detected.createIndex()
    evaluator = COCOeval(coco, detected, 'bbox')
    evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = ['AP','AP50','AP75','AP_small','AP_medium','AP_large',
             'AR1','AR10','AR100','AR_small','AR_medium','AR_large']
    manifest.update(status='complete', metrics_fraction=dict(zip(names,map(float,evaluator.stats))),
                    service_latency_ms={'median':float(np.median(durations)),
                                        'p95':float(np.percentile(durations,95)),
                                        'p99':float(np.percentile(durations,99))})
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2, default=str)+'\n')


if __name__ == '__main__':
    main()
