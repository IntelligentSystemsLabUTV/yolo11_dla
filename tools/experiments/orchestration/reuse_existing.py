#!/usr/bin/env python3
"""Register existing YOLO11 FP16 engines, build evidence and timing traces; never build or run them."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import configure_imports
configure_imports()

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import time

from common import new_directory, sha256, write_json
from suite import CONFIGS, inspect_strict

PREFIXES = {
    'stock-gpu': 'yolo11n-gpu-fp16',
    'adapted-gpu': 'yolo11n-dla-gpu-fp16',
    'stock-fallback': 'yolo11n-dla-gpu-fallback-fp16',
    'adapted-strict': 'yolo11n-dla-strict-dla-fp16',
    'stock-strict': 'yolo11n-strict-dla-fp16',
}
LIMITS = [
    'Hashes identify files at registration time, not at their historical creation time.',
    'Build logs record ONNX paths; no historical checkpoint-to-ONNX-to-engine hash chain is inferred.',
    'Each archived microbenchmark is one launch. Hardware mode/clocks must be recovered from original records.',
    'Registration does not deserialize engines or certify compatibility with the current target runtime.',
]


def recorded_command(path, success=True):
    text = path.read_text()
    marker = 'PASSED' if success else 'FAILED'
    match = re.search(r'^&&&& ' + marker + r' TensorRT\.trtexec .* # (.+)$', text, re.M)
    if not match:
        raise ValueError(f'Missing recorded {marker} command: {path}')
    return shlex.split(match.group(1))


def register(source, output):
    source = Path(source).resolve()
    # Validate all required evidence before creating the registration directory.
    builds, timings = [], []
    for name, prefix in PREFIXES.items():
        negative = name == 'stock-strict'
        command = recorded_command(source/f'{prefix}.build.log', success=not negative)
        model = CONFIGS[name][0] if not negative else 'stock'
        onnx = next((arg.split('=', 1)[1] for arg in command if arg.startswith('--onnx=')), '')
        expected = 'yolo11n.onnx' if model == 'stock' else 'yolo11n-dla.onnx'
        if Path(onnx).name != expected or '--fp16' not in command:
            raise ValueError(f'Unexpected model or precision in {name} build log')
        core = '--useDLACore=0' in command
        fallback = '--allowGPUFallback' in command
        if core != (name in ('stock-fallback', 'adapted-strict', 'stock-strict')) or fallback != (name == 'stock-fallback'):
            raise ValueError(f'Unexpected placement flags for {name}')
        job = {'name': name, 'model': model, 'command': command,
               'returncode': None, 'recorded_result': 'FAILED' if negative else 'PASSED',
               'outcome': 'archived_compiler_rejected_strict_dla' if negative else 'reused_existing_engine',
               'recorded_onnx_path': onnx}
        if negative:
            text = (source/f'{prefix}.build.log').read_text()
            if 'MatMul' not in text or 'Finished parsing network model' not in text:
                raise ValueError('Stock strict rejection evidence does not match the retained baseline')
        else:
            engine = source/f'{prefix}.engine'
            if engine.stat().st_size == 0:
                raise ValueError(f'Empty engine: {engine}')
            save = next((arg.split('=', 1)[1] for arg in command if arg.startswith('--saveEngine=')), '')
            if Path(save).name != engine.name:
                raise ValueError(f'Engine name disagrees with build log: {name}')
            job['engine_sha256'] = sha256(engine)
            job['inspection'] = inspect_strict(source/f'{prefix}.layers.json')
            if name == 'adapted-strict' and not job['inspection']['one_dla_no_gpu_or_reformat']:
                raise ValueError('Archived strict engine inspector contains GPU/reformat or multiple DLA nodes')
            benchmark = recorded_command(source/f'{prefix}.benchmark.log')
            load = next((arg.split('=', 1)[1] for arg in benchmark if arg.startswith('--loadEngine=')), '')
            if Path(load).name != engine.name:
                raise ValueError(f'Benchmark names a different engine: {name}')
            samples = json.loads((source/f'{prefix}.times.json').read_text())
            if not samples:
                raise ValueError(f'Empty timing trace: {name}')
            timings.append({'name': f'repeat01-{name}', 'configuration': name, 'repeat': 1,
                            'command': benchmark, 'recorded_result': 'PASSED', 'samples': len(samples),
                            'engine_sha256': job['engine_sha256']})
        builds.append(job)

    out = new_directory(output)
    engines = new_directory(out/'engines')
    micro = new_directory(out/'micro')
    evidence, micro_evidence = {}, {}

    def link(original, target, hashes):
        target.symlink_to(os.path.relpath(original, target.parent))
        hashes[target.name] = sha256(original)

    for name, prefix in PREFIXES.items():
        for suffix in ('build.log', 'engine', 'layers.json', 'profile.log', 'profile.json'):
            original = source/f'{prefix}.{suffix}'
            if original.is_file():
                link(original, engines/f'{name}.{suffix}', evidence)
        if name in CONFIGS:
            link(source/f'{prefix}.benchmark.log', micro/f'repeat01-{name}.log', micro_evidence)
            link(source/f'{prefix}.times.json', micro/f'repeat01-{name}.times.json', micro_evidence)
    shared = {'status': 'complete', 'source_kind': 'archived_fp16_matrix',
              'source_directory': str(source), 'registration_time_ns': time.time_ns(),
              'historical_provenance_limits': LIMITS}
    write_json(engines/'manifest.json', {**shared, 'arguments': {'stage': 'build'},
                                       'jobs': builds, 'evidence_sha256': evidence})
    write_json(micro/'manifest.json', {**shared, 'arguments': {'stage': 'micro'}, 'jobs': timings,
                                     'evidence_sha256': micro_evidence,
                                     'engines_manifest_sha256': sha256(engines/'manifest.json')})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Existing logs/fp16_matrix directory')
    parser.add_argument('--output', type=Path, required=True, help='New directory for relative links and manifests')
    args = parser.parse_args()
    out = register(args.source, args.output)
    print(f'Registered four existing engines and their evidence: {out / "engines"}')
    print(f'Archived microbenchmarks ready for summarize: {out / "micro"}')
    print('No export, build, benchmark or source-file modification performed.')


if __name__ == '__main__':
    main()
