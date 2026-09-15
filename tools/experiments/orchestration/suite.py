#!/usr/bin/env python3
"""Plan or run the YOLO11-DLA Jetson matrix. Each stage uses a fresh directory."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import ROOT

import argparse
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import time

from common import new_directory, provenance, sha256, write_json
from telemetry import Telemetry, add_power_arguments, power_selection

CONFIGS = {'stock-gpu': ('stock', None), 'adapted-gpu': ('adapted', None),
           'stock-fallback': ('stock', 0), 'adapted-strict': ('adapted', 0)}


def build_command(trtexec, artifacts, output, name, model, core=None, fallback=False, native=False):
    command = [trtexec, f'--onnx={artifacts/model}.onnx', f'--saveEngine={output/name}.engine',
               '--fp16', '--profilingVerbosity=detailed', '--dumpLayerInfo',
               f'--exportLayerInfo={output/name}.layers.json', '--skipInference', '--verbose',
               '--inputIOFormats=fp16:dla_hwc4' if native else '--inputIOFormats=fp16:chw',
               '--outputIOFormats=fp16:chw16' if native else '--outputIOFormats=fp16:chw']
    if core is not None:
        command += [f'--useDLACore={core}']
    if fallback:
        command += ['--allowGPUFallback']
    return command


def inspect_strict(path):
    data = json.loads(Path(path).read_text())
    layers = data.get('Layers', data) if isinstance(data, dict) else data
    if not isinstance(layers, list) or not layers:
        raise ValueError('Missing inspector layer list')
    kinds = [layer.get('LayerType', layer.get('ParameterType', 'UNKNOWN')) for layer in layers]
    dla = sum(k in ('DlaLayer', 'DLA') for k in kinds)
    other = [k for k in kinds if k not in ('DlaLayer', 'DLA', 'NoOp', 'Constant', 'shape_call')]
    return {'one_dla_no_gpu_or_reformat': dla == 1 and not other,
            'dla_nodes': dla, 'other_types': other, 'layer_types': kinds}


def plan(args):
    output = args.output.resolve()
    artifacts = args.artifacts.resolve() if args.artifacts else None
    engines = args.engines.resolve() if args.engines else None
    jobs = []
    def add(name, command, **extra):
        jobs.append({'name': name, 'command': list(map(str, command)), **extra})
    if args.stage == 'build':
        for name, (model, core) in CONFIGS.items():
            add(name, build_command(args.trtexec, artifacts, output, name, model, core,
                                    name == 'stock-fallback', name == 'adapted-strict'),
                model=model, strict_required=name == 'adapted-strict')
        # Retain the stock strict-compilation outcome as the compatibility baseline.
        # Its decoded rank-three output requires linear I/O.
        add('stock-strict', build_command(args.trtexec, artifacts, output,
                                         'stock-strict', 'stock', 0),
            model='stock', exploratory=True)
    elif args.stage == 'accuracy':
        base = [sys.executable, ROOT/'evaluation/evaluate_coco.py', '--images', args.images,
                '--annotations', args.annotations, '--cpu-threads', args.cpu_threads]
        if args.limit:
            base += ['--limit', args.limit]
        for model, checkpoint in checkpoint_paths(args).items():
            add(model+'-pt', base+['--checkpoint', checkpoint, '--output', output/(model+'-pt')])
        for name, (_, core) in CONFIGS.items():
            command = base+['--engine', engines/f'{name}.engine', '--output', output/name]
            if core is not None:
                command += ['--dla-core', core]
            add(name, command)
        # Low-confidence sparse validation is separate from dense engine AP.
        for name in ('adapted-gpu', 'adapted-strict'):
            command = base+['--engine', engines/f'{name}.engine', '--sparse-decode',
                            '--output', output/(name+'-sparse')]
            if name == 'adapted-strict':
                command += ['--dla-core', 0]
            add(name+'-sparse', command)
    elif args.stage == 'parity':
        add('parity', [sys.executable, ROOT/'evaluation/validate.py', '--gpu-engine', engines/'adapted-gpu.engine',
                       '--dla-engine', engines/'adapted-strict.engine', '--images', args.images,
                       '--annotations', args.annotations, '--output', output/'comparison',
                       '--limit', args.limit or 50, '--cpu-threads', args.cpu_threads,
                       '--raw-atol', args.raw_atol, '--raw-rtol', args.raw_rtol])
        if getattr(args, 'raw_diagnostic_only', False):
            jobs[-1]['command'].append('--raw-diagnostic-only')
    else:
        rng = random.Random(args.seed)
        for repeat in range(1, args.repeats+1):
            names = list(CONFIGS)
            if args.stage == 'energy':
                names += ['idle']
            rng.shuffle(names)
            for name in names:
                tag = f'repeat{repeat:02d}-{name}'
                core = CONFIGS.get(name, (None, None))[1]
                if args.stage == 'micro':
                    command = [args.trtexec, f'--loadEngine={engines/name}.engine', '--warmUp=2000',
                               f'--duration={args.duration}', '--infStreams=1', '--useSpinWait',
                               f'--exportTimes={output/tag}.times.json']
                    if core is not None:
                        command += [f'--useDLACore={core}']
                else:
                    command = [sys.executable, ROOT/'benchmarks/benchmark.py', '--output', output/tag,
                               '--duration', args.duration, '--fps', args.fps,
                               '--stabilize-seconds', args.stabilize_seconds,
                               '--cpu-threads', args.cpu_threads, '--deadline-ms', args.deadline_ms]
                    if name == 'idle':
                        command += ['--idle']
                    else:
                        command += ['--engine', engines/f'{name}.engine', '--images', args.images,
                                    '--annotations', args.annotations, '--limit', args.limit or 100]
                    if core is not None:
                        command += ['--dla-core', core]
                    if args.tegrastats:
                        command += ['--tegrastats', args.tegrastats]
                    if args.stage == 'energy':
                        if getattr(args, 'power_profile', None):
                            command += ['--power-profile', args.power_profile]
                        else:
                            command += ['--rail', args.rail]
                add(tag, command, repeat=repeat, configuration=name)
        if args.stage == 'micro' and getattr(args, 'profile', False):
            # Separate profiling runs, never mixed into unprofiled latency totals.
            for name, (_, core) in CONFIGS.items():
                command = [args.trtexec, f'--loadEngine={engines/name}.engine', '--dumpProfile',
                           f'--exportProfile={output/name}.profile.json', '--separateProfileRun',
                           '--duration=10', '--warmUp=2000']
                if core is not None:
                    command += [f'--useDLACore={core}']
                add(name+'-profile', command, profile_only=True)
    return jobs


def checkpoint_paths(args):
    paths = {}
    for model in ('stock', 'adapted'):
        explicit = getattr(args, model + '_checkpoint', None)
        if explicit:
            paths[model] = explicit.resolve()
        elif args.artifacts:
            paths[model] = args.artifacts.resolve()/f'{model}.pt'
    return paths


def validate_inputs(args):
    record = {}
    if args.artifacts:
        artifacts = args.artifacts.resolve()
        prepared = json.loads((artifacts/'manifest.json').read_text())
        if prepared['status'] != 'complete':
            raise ValueError('prepare.py did not complete')
        for name, item in prepared['artifacts'].items():
            for ext, key in [('onnx', 'onnx_sha256'), ('pt', 'checkpoint_sha256')]:
                if sha256(artifacts/f'{name}.{ext}') != item[key]:
                    raise ValueError(f'Artifact changed after export: {name}.{ext}')
        record['artifacts_manifest_sha256'] = sha256(artifacts/'manifest.json')
    if args.stage != 'build':
        engines = args.engines.resolve()
        built = json.loads((engines/'manifest.json').read_text())
        if built['status'] != 'complete':
            raise ValueError('Engine manifest incomplete')
        archived = built.get('source_kind') in ('archived_fp16_matrix', 'archived_int8_matrix')
        if not archived and (not args.artifacts or built['artifacts_manifest_sha256'] !=
                             record['artifacts_manifest_sha256']):
            raise ValueError('Build manifest belongs to different exports; provide its --artifacts')
        jobs = {j['name']: j for j in built['jobs']}
        for name in CONFIGS:
            if name not in jobs or sha256(engines/f'{name}.engine') != jobs[name]['engine_sha256']:
                raise ValueError(f'Engine missing or changed: {name}')
        # Check the archived logs/inspectors as well as the serialized engines.
        for name, digest in built.get('evidence_sha256', {}).items():
            if sha256(engines/name) != digest:
                raise ValueError(f'Archived evidence changed: {name}')
        record['engines_manifest_sha256'] = sha256(engines/'manifest.json')
        record['engine_source_kind'] = built.get('source_kind', 'new_build')
        if archived:
            record['historical_provenance_limits'] = built['historical_provenance_limits']
        if built.get('source_kind') == 'archived_int8_matrix':
            record['evaluation_protocol'] = built['evaluation_protocol']
            if args.stage in ('accuracy', 'parity') and sha256(args.annotations) != built['evaluation_protocol']['annotation_sha256']:
                raise ValueError('INT8 accuracy/parity require registered held-out annotations')
            if args.stage in ('pipeline', 'energy') and sha256(args.annotations) != built['workload_annotation_sha256']:
                raise ValueError('INT8 pipeline/energy require original full COCO annotations to preserve FP16 workload IDs')
    if args.stage == 'accuracy':
        record['checkpoint_sha256'] = {name: sha256(path) for name, path in checkpoint_paths(args).items()}
    return record


def reusable_evaluations(args, jobs):
    """Validate complete past evaluations before reusing them; never rerun stock AP."""
    requested = getattr(args, 'reuse_evaluation', [])
    if requested and args.stage != 'accuracy':
        raise ValueError('--reuse-evaluation applies only to accuracy')
    result = {}
    job_names = {j['name'] for j in jobs}
    for item in requested:
        name, directory = item.split('=', 1)
        if name in result or name not in job_names or name.endswith('-pt'):
            raise ValueError(f'Invalid/duplicate reused evaluation: {name}')
        directory = Path(directory).resolve()
        manifest = json.loads((directory/'manifest.json').read_text())
        old = manifest['arguments']
        engine_name = name.removesuffix('-sparse')
        if (manifest['status'] != 'complete' or
                manifest['artifact_sha256'] != sha256(args.engines/(engine_name+'.engine')) or
                manifest['annotation_sha256'] != sha256(args.annotations) or
                old['limit'] != args.limit or old['conf'] != 0.001 or old['iou'] != 0.7 or
                old['max_det'] != 300 or old['cpu_threads'] != args.cpu_threads or
                bool(old['sparse_decode']) != name.endswith('-sparse')):
            raise ValueError(f'Reused evaluation protocol/engine mismatch: {name}')
        # Image identity is checked against the current dataset, not just filenames.
        from common import image_sequence
        sequence = image_sequence(args.images, args.annotations, args.limit)
        rows = [json.loads(line) for line in (directory/'frames.jsonl').read_text().splitlines()]
        if manifest['image_ids'] != [r['image_id'] for r in sequence] or [
                (r['image_id'], r['image_sha256']) for r in rows] != [
                (r['image_id'], r['sha256']) for r in sequence]:
            raise ValueError(f'Reused evaluation image sequence mismatch: {name}')
        sidecar = args.engines/(engine_name+'.engine.quantization.json')
        if sidecar.exists() and manifest.get('quantization') != json.loads(sidecar.read_text()):
            raise ValueError(f'Reused quantization metadata mismatch: {name}')
        if not (directory/'predictions.json').is_file() or not manifest.get('metrics_fraction'):
            raise ValueError(f'Missing reusable predictions/metrics: {name}')
        result[name] = directory
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['build','parity','accuracy','micro','pipeline','energy'])
    p.add_argument('--artifacts', type=Path, help='prepare.py directory; required for new builds only')
    p.add_argument('--stock-checkpoint', type=Path, help='Optional stock checkpoint for accuracy, without exporting')
    p.add_argument('--adapted-checkpoint', type=Path, help='Optional adapted checkpoint for accuracy, without exporting')
    p.add_argument('--profile', action='store_true', help='Add fresh layer profiles to micro; existing profiles need no rerun')
    p.add_argument('--engines', type=Path, help='Completed build or reuse-existing engines directory')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--images', type=Path)
    p.add_argument('--annotations', type=Path)
    p.add_argument('--trtexec', default='/usr/src/tensorrt/bin/trtexec')
    p.add_argument('--dry-run', action='store_true', help='Print JSON plan only, no files or runtime imports')
    p.add_argument('--limit', type=int, help='Subset for smoke tests or preloaded timing sequence')
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--seed', type=int, default=2027)
    p.add_argument('--duration', type=int, default=60)
    p.add_argument('--fps', type=float, default=0)
    p.add_argument('--deadline-ms', type=float, default=100)
    p.add_argument('--stabilize-seconds', type=float, default=30)
    p.add_argument('--cpu-threads', type=int, default=1)
    p.add_argument('--tegrastats', help='Executable path')
    add_power_arguments(p)
    p.add_argument('--raw-atol', type=float, default=0.1, help='Exploratory numerical gate, declare before target run')
    p.add_argument('--raw-rtol', type=float, default=0.05)
    p.add_argument('--raw-diagnostic-only', action='store_true', help='Parity: raw differences diagnostic; layout, finite values and decoder still gate')
    p.add_argument('--reuse-evaluation', action='append', default=[], metavar='NAME=DIRECTORY',
                   help='Accuracy: reuse an already completed matching evaluation, preserving its manifest')
    args = p.parse_args()
    if args.stage == 'build' and not args.artifacts:
        p.error('new builds require --artifacts; use reuse-existing for archived engines')
    if args.stage != 'accuracy' and (args.stock_checkpoint or args.adapted_checkpoint):
        p.error('checkpoint arguments apply only to accuracy')
    if args.stage != 'micro' and args.profile:
        p.error('--profile applies only to micro')
    if args.raw_diagnostic_only and args.stage != 'parity':
        p.error('--raw-diagnostic-only applies only to parity')
    if args.stage != 'build' and not args.engines:
        p.error('--engines required')
    if args.stage in ('parity','accuracy','pipeline','energy') and (not args.images or not args.annotations):
        p.error('--images and --annotations required')
    if args.stage == 'energy' and (not args.tegrastats or not (args.rail or args.power_profile)):
        p.error('energy requires --tegrastats and either --rail or --power-profile agx-orin')
    if args.stage != 'energy' and (args.rail or args.power_profile):
        p.error('--rail/--power-profile applies only to energy')
    if args.duration < 1 or args.repeats < 1 or args.cpu_threads < 1 or args.fps < 0 or args.stabilize_seconds < 0:
        p.error('Invalid duration/repeats/threads/fps/stabilization')
    if args.limit is not None and args.limit < 1:
        p.error('--limit must be positive')
    jobs = plan(args)
    if args.dry_run:
        print(json.dumps(jobs, indent=2))
        return
    inputs = validate_inputs(args)
    reuse = reusable_evaluations(args, jobs)
    out = new_directory(args.output)
    manifest = {'status': 'running', 'arguments': vars(args), 'provenance': provenance(), 'jobs': jobs,
                **inputs}
    if args.stage == 'energy':
        manifest['power_definition'] = power_selection(args.rail, args.power_profile)
    write_json(out/'manifest.json', manifest)
    try:
        for job in jobs:
            print(f"[{args.stage}] {job['name']}", flush=True)
            if job['name'] in reuse:
                previous = reuse[job['name']]
                (out/job['name']).symlink_to(previous, target_is_directory=True)
                job.update(returncode=0, reused_from=str(previous))
                for filename in ('manifest.json', 'frames.jsonl', 'predictions.json'):
                    manifest.setdefault('evidence_sha256', {})[job['name']+'/'+filename] = sha256(previous/filename)
                write_json(out/'manifest.json', manifest)
                continue
            job['start_s'] = time.perf_counter()
            monitor = Telemetry(args.tegrastats) if args.stage == 'micro' and args.tegrastats else None
            try:
                with (out/(job['name']+'.log')).open('w') as log:
                    result = subprocess.run(job['command'], stdout=log, stderr=subprocess.STDOUT)
            finally:
                if monitor:
                    monitor.close(out/(job['name']+'.telemetry.jsonl'))
            job.update(returncode=result.returncode, end_s=time.perf_counter())
            if args.stage == 'build':
                logtext = (out/(job['name']+'.log')).read_text()
                engine = out/(job['name']+'.engine')
                if result.returncode == 0:
                    if not engine.is_file() or not engine.stat().st_size:
                        raise RuntimeError(f"Missing serialized engine: {job['name']}")
                    job['engine_sha256'] = sha256(engine)
                    job['inspection'] = inspect_strict(out/(job['name']+'.layers.json'))
                    if job.get('strict_required') and not job['inspection']['one_dla_no_gpu_or_reformat']:
                        raise RuntimeError('Strict adapted build did not produce exactly one DLA node')
                elif job.get('exploratory'):
                    # Only compiler rejection is an expected negative result; missing libraries,
                    # malformed commands or parser errors must stop the suite.
                    rejected = re.search(r'(not supported on DLA|unsupported.*DLA|could not be assigned to DLA|cannot.*DLA|DLA.*does not support)', logtext, re.I)
                    if not rejected or 'Finished parsing network model' not in logtext:
                        raise RuntimeError(f"Unclassified failure in {job['name']}; inspect its log")
                    job['outcome'] = 'compiler_rejected_strict_dla'
                else:
                    raise RuntimeError(f"Required build failed: {job['name']}")
            elif result.returncode:
                raise RuntimeError(f"{job['name']} failed with status {result.returncode}; inspect log")
            write_json(out/'manifest.json', manifest)
        manifest['status'] = 'complete'
    except BaseException as exc:
        manifest.update(status='failed', error=str(exc))
        raise
    finally:
        write_json(out/'manifest.json', manifest)


if __name__ == '__main__':
    main()
