#!/usr/bin/env python3
"""Resident-engine multi-image service benchmark, paced replay, or matched idle energy interval."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import configure_imports
configure_imports()

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from common import image_sequence, new_directory, provenance, sha256, stats, write_json
from telemetry import Telemetry, add_power_arguments, energy_parts, integrate_power, power_selection, subtract_idle


def release_slot(now, origin, fps, next_slot):
    """Queue capacity one, keep latest due frame; count skipped release slots explicitly."""
    latest = max(next_slot, int((now-origin)*fps))
    return latest, latest-next_slot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--engine', type=Path)
    source.add_argument('--idle', action='store_true')
    p.add_argument('--images', type=Path)
    p.add_argument('--annotations', type=Path)
    p.add_argument('--limit', type=int, default=100, help='Preloaded sequence length, sorted COCO image IDs')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dla-core', type=int)
    p.add_argument('--duration', type=float, default=60)
    p.add_argument('--fps', type=float, default=0, help='0: sequential saturation; otherwise fixed release rate')
    p.add_argument('--deadline-ms', type=float, default=100)
    p.add_argument('--warmup', type=int, default=30)
    p.add_argument('--stabilize-seconds', type=float, default=30)
    p.add_argument('--cpu-threads', type=int, default=1)
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--iou', type=float, default=0.45)
    p.add_argument('--dense-decode', action='store_true')
    p.add_argument('--tegrastats', help='Executable path; no sudo or clock/power mode changes')
    add_power_arguments(p)
    p.add_argument('--idle-summary', type=Path, help='Same-condition idle summary for subtraction')
    args = p.parse_args()
    if args.duration <= 0 or args.fps < 0 or args.warmup < 0 or args.cpu_threads < 1 or args.stabilize_seconds < 0:
        p.error('Invalid duration, rate, warmup, threads or stabilization')
    if not 0 < args.conf < 1 or not 0 < args.iou < 1 or args.deadline_ms <= 0:
        p.error('Invalid confidence, IoU or deadline')
    measure_power = bool(args.rail or args.power_profile)
    if measure_power and not args.tegrastats:
        p.error('--rail/--power-profile requires --tegrastats')
    if args.idle_summary and not measure_power:
        p.error('--idle-summary requires --rail or --power-profile')
    if args.engine and (args.images is None or args.annotations is None):
        p.error('--engine requires --images and --annotations')
    out = new_directory(args.output)
    report = {'status': 'running', 'arguments': vars(args), 'provenance': provenance(),
              'boundary': 'preloaded BGR images; sequential CPU preprocessing, native TRT, CPU NMS; no disk I/O',
              'queue_policy': 'latest-only, capacity 1; skipped scheduled releases counted as drops',
              'rate_definition': 'completed frames / measured wall interval, including pacing and loop overhead'}
    write_json(out/'summary.json', report)
    records = []
    monitor = None
    try:
        if args.engine:
            import cv2
            import torch
            import tensorrt
            from yolo_e2e_tester import TensorRTEngine, identify_head, postprocess, preprocess
            torch.set_num_threads(args.cpu_threads)
            torch.set_num_interop_threads(1)
            cv2.setNumThreads(args.cpu_threads)
            sequence = image_sequence(args.images, args.annotations, args.limit)
            images = [cv2.imread(e['path']) for e in sequence]
            if any(img is None for img in images):
                raise ValueError('Image decode failure')
            write_json(out/'sequence.json', sequence)
            runner = TensorRTEngine(args.engine, dla_core=args.dla_core)
            runner.describe()
            if runner.input_meta.shape != (1,3,640,640):
                raise ValueError('Expected static 1x3x640x640')
            cpu = torch.device('cpu')
            x, _ = preprocess(images[0], (640,640))
            outputs, _ = runner.infer(x, cpu)
            head, reg_max = identify_head(outputs, 'auto', 80, None)
            if head not in ('standard', 'raw-nms') or (head == 'raw-nms' and reg_max != 16):
                raise ValueError('Only YOLO11 and YOLO11-DLA detection are in scope')
            report.update(engine_sha256=sha256(args.engine), annotation_sha256=sha256(args.annotations),
                          runtime_versions={'torch': torch.__version__, 'cuda': torch.version.cuda,
                                            'cudnn': torch.backends.cudnn.version(), 'tensorrt': tensorrt.__version__,
                                            'opencv': cv2.__version__},
                          bindings={k: asdict(v) for k,v in runner.metadata.items()},
                          quantization=runner.quantization)

            @torch.inference_mode()
            def frame(index):
                start = time.perf_counter()
                tensor, transform = preprocess(images[index % len(images)], (640,640))
                prep_end = time.perf_counter()
                output, stages = runner.infer(tensor, cpu)
                infer_end = time.perf_counter()
                detections, post_stats = postprocess(
                    output, head=head, reg_max=reg_max, nc=80, conf=args.conf, iou=args.iou,
                    max_det=300, transform=transform, agnostic=False, post_device=cpu,
                    dense_decode=args.dense_decode)
                end = time.perf_counter()
                return {'start_s': start, 'end_s': end, 'preprocess_ms': (prep_end-start)*1000,
                        **stages, 'postprocess_ms': (end-infer_end)*1000, 'total_ms': (end-start)*1000,
                        **post_stats, 'detections': len(detections),
                        'image_id': sequence[index % len(images)]['image_id']}
            for i in range(args.warmup):
                frame(i)
        if args.tegrastats:
            monitor = Telemetry(args.tegrastats)
            if measure_power:
                monitor.wait_for_rails(power_selection(args.rail, args.power_profile)['rails'])
        stable_end = time.perf_counter() + args.stabilize_seconds
        while time.perf_counter() < stable_end:
            if args.engine:
                tick = time.perf_counter()
                frame(0)
                if args.fps:
                    time.sleep(max(0, 1/args.fps - (time.perf_counter()-tick)))
            else:
                time.sleep(min(0.1, max(0, stable_end-time.perf_counter())))
        # Collect a power sample before the timed window even with stabilization=0.
        if monitor:
            time.sleep(0.3)
        start = time.perf_counter()
        stop = start + args.duration
        next_slot = dropped = deadline_misses = 0
        if args.idle:
            time.sleep(args.duration)
        else:
            while time.perf_counter() < stop:
                slot = next_slot
                release = time.perf_counter()
                if args.fps:
                    release = start + slot/args.fps
                    if release >= stop:
                        time.sleep(max(0, stop-time.perf_counter()))
                        break
                    time.sleep(max(0, release-time.perf_counter()))
                    now = time.perf_counter()
                    if now >= stop:
                        break
                    slot, skipped = release_slot(now, start, args.fps, next_slot)
                    dropped += skipped
                    release = start + slot/args.fps
                record = frame(slot)
                record.update(slot=slot, release_s=release,
                              queue_ms=(record['start_s']-release)*1000,
                              release_to_result_ms=(record['end_s']-release)*1000)
                record['deadline_missed'] = record['release_to_result_ms'] > args.deadline_ms
                deadline_misses += int(record['deadline_missed'])
                records.append(record)
                next_slot = slot+1
        end = time.perf_counter()
        if args.fps and not args.idle:
            import math
            scheduled = math.ceil(args.duration*args.fps)
            dropped += max(0, scheduled-next_slot)
        else:
            scheduled = len(records)
        report.update(start_s=start, end_s=end, scheduled_stop_s=stop, completed=len(records),
                      interval_s=end-start, achieved_fps=len(records)/(end-start), dropped=dropped,
                      scheduled=scheduled, deadline_misses_completed=deadline_misses,
                      missed_or_dropped=deadline_misses+dropped,
                      latency_ms={key: stats([r[key] for r in records]) for key in
                                  ('preprocess_ms','input_pack_h2d_ms','engine_ms','output_unpack_ms',
                                   'postprocess_ms','total_ms','queue_ms','release_to_result_ms')},
                      candidates=stats([r['candidate_anchors'] for r in records]))
        if monitor:
            time.sleep(0.3)  # Bracket the final frame, which may finish after the nominal stop.
            monitor.close(out/'telemetry.jsonl')
            if measure_power:
                report['energy'] = integrate_power(monitor.rows, start, end, rail=args.rail,
                                                   power_profile=args.power_profile, completed=len(records))
                if args.idle_summary:
                    idle = json.loads(args.idle_summary.read_text())
                    if idle['status'] != 'complete' or not idle['arguments']['idle']:
                        raise ValueError('Require completed idle run')
                    report['energy']['idle_summary_sha256'] = sha256(args.idle_summary)
                    report['energy'].update(subtract_idle(report['energy'], idle['energy'], len(records)))
                    for name, part in report['energy']['per_rail'].items():
                        part.update(subtract_idle(part, energy_parts(idle['energy'])[name], len(records)))
        report['status'] = 'complete'
    except BaseException as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        if monitor:
            monitor.close(out/'telemetry.jsonl')
        (out/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        write_json(out/'summary.json', report)


if __name__ == '__main__':
    main()
