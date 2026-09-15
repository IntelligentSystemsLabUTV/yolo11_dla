#!/usr/bin/env python3
"""Register existing INT8 engines and exact binding scales; never build or calibrate."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import configure_imports
configure_imports()

import argparse
import json
import re
import shlex
import shutil

from common import new_directory, sha256, write_json
from int8.build import check_cache
from int8.quantization import native_scales
from suite import inspect_strict

EXPECTED = {
    'stock-gpu': ('stock', False, None),
    'adapted-gpu': ('adapted', False, None),
    'stock-fallback': ('stock', True, 0),
    'adapted-strict': ('adapted', False, 0),
}


def discover(matrix, onnx, caches):
    found = {name: [] for name in EXPECTED}
    for log in matrix.rglob('*.log'):
        text = log.read_text(errors='replace')
        endings = [line for line in text.splitlines() if line.startswith('&&&& PASSED TensorRT.trtexec')]
        if len(endings) != 1 or '[E]' in text:
            continue
        if '[TensorRT v100300]' not in endings[0]:
            raise ValueError(f'Expected TensorRT 10.3 build: {log}')
        command = shlex.split(endings[0].split(' # ', 1)[1])
        flags = {arg.split('=', 1)[0]: arg.split('=', 1)[1] if '=' in arg else True for arg in command[1:]}
        if not all(k in flags for k in ('--onnx', '--calib', '--saveEngine', '--exportLayerInfo', '--int8', '--fp16')):
            continue
        for name, (model, fallback, core) in EXPECTED.items():
            if Path(flags['--onnx']).resolve() != onnx[model].resolve():
                continue
            if ('--allowGPUFallback' in flags) != fallback or flags.get('--useDLACore') != (str(core) if core is not None else None):
                continue
            if Path(flags['--calib']).resolve() != (caches[model]/'calibration.cache').resolve():
                raise ValueError(f'Unexpected calibration path: {log}')
            native = name == 'adapted-strict'
            if (flags.get('--inputIOFormats'), flags.get('--outputIOFormats')) != (
                    'int8:dla_hwc4' if native else 'fp16:chw', 'int8:chw32' if native else 'fp16:chw'):
                continue
            # Archive may have moved since the build: artifacts must be adjacent to its log.
            engine = log.parent/Path(flags['--saveEngine']).name
            layers = log.parent/Path(flags['--exportLayerInfo']).name
            if not engine.is_file() or engine.stat().st_size == 0 or not layers.is_file():
                raise ValueError(f'Missing successful-build artifact: {log}')
            inspection = inspect_strict(layers)
            if native and not inspection['one_dla_no_gpu_or_reformat']:
                raise ValueError('Strict engine is not a single DLA node')
            if core is None and inspection['dla_nodes']:
                raise ValueError(f'GPU engine contains DLA nodes: {name}')
            if fallback and not inspection['dla_nodes']:
                raise ValueError('Fallback baseline contains no DLA nodes')
            found[name].append(dict(name=name, engine=engine, log=log, layers=layers,
                                    command=command, inspection=inspection))
    for name, matches in found.items():
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one successful build for {name}, found {len(matches)}; use a matrix directory containing only the intended artifacts')
    return [matches[0] for matches in found.values()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--split', type=Path, required=True)
    p.add_argument('--stock-onnx', type=Path, required=True)
    p.add_argument('--adapted-onnx', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    onnx = {'stock': args.stock_onnx, 'adapted': args.adapted_onnx}
    caches = {m: args.calibration/m for m in onnx}
    calibration = {m: check_cache(caches[m], onnx[m], m) for m in onnx}
    split = json.loads((args.split/'manifest.json').read_text())
    if split['status'] != 'complete' or split['calibration_count'] != 500 or split['evaluation_count'] != 4500:
        raise ValueError('Expected completed 500/4500 split')
    for data in calibration.values():
        if data['split_manifest_sha256'] != sha256(args.split/'manifest.json') or data.get('images_consumed') != 500:
            raise ValueError('Calibration differs from split or is incomplete')
        if data['tensorrt_version'] != '10.3.0':
            raise ValueError('Native I/O scale extraction verified for TensorRT 10.3.0 only')
    if calibration['stock']['preprocess_sha256'] != calibration['adapted']['preprocess_sha256']:
        raise ValueError('Calibration preprocessing differs')
    if calibration['stock']['calibration_images'] != calibration['adapted']['calibration_images'] or calibration['stock']['calibration_images'] != split['calibration_images']:
        raise ValueError('Calibration image IDs/hashes differ from split')
    evaluation = args.split/'evaluation.json'
    if sha256(evaluation) != split['evaluation_annotations_sha256']:
        raise ValueError('Held-out annotations changed')
    evaluation_ids = sorted(i['id'] for i in json.loads(evaluation.read_text())['images'])
    calibration_ids = {i['image_id'] for i in split['calibration_images']}
    if len(set(evaluation_ids)) != 4500 or calibration_ids.intersection(evaluation_ids) or evaluation_ids != split['evaluation_image_ids']:
        raise ValueError('Held-out IDs invalid or overlap calibration')
    jobs = discover(args.matrix.resolve(), onnx, caches)
    strict = next(j for j in jobs if j['name'] == 'adapted-strict')
    scales = native_scales(caches['adapted']/'calibration.cache', strict['log'],
                          ('images', 'pred_p3', 'pred_p4', 'pred_p5'))
    out = new_directory(args.output)
    evidence = {}
    def copy(source, name):
        shutil.copy2(source, out/name)
        evidence[name] = sha256(out/name)
    for model in onnx:
        copy(caches[model]/'manifest.json', model+'.calibration.json')
        copy(caches[model]/'calibration.cache', model+'.cache')
    copy(args.split/'manifest.json', 'split.json')
    copy(evaluation, 'evaluation.json')
    if (args.matrix/'manifest.json').is_file():
        copy(args.matrix/'manifest.json', 'original-matrix.json')
    for job in jobs:
        name = job['name']
        (out/(name+'.engine')).symlink_to(job['engine'].resolve())
        job['engine_sha256'] = sha256(job['engine'])
        job['returncode'] = 0
        copy(job['log'], name+'.build.log')
        copy(job['layers'], name+'.layers.json')
    quantization = dict(schema=1, engine_sha256=strict['engine_sha256'], bindings=scales,
                        cache_sha256=calibration['adapted']['cache_sha256'],
                        build_log_sha256=sha256(strict['log']),
                        source='Exact calibration float32 scales; explicit TRT 10.3 I/O ranges cross-checked in successful build log',
                        rounding='nearest ties to even; clamp [-128,127]; symmetric zero point 0')
    sidecar = 'adapted-strict.engine.quantization.json'
    write_json(out/sidecar, quantization)
    evidence[sidecar] = sha256(out/sidecar)
    write_json(out/'manifest.json', dict(status='complete', source_kind='archived_int8_matrix',
        jobs=jobs, evidence_sha256=evidence, quantization=quantization,
        workload_annotation_sha256=split['annotations_sha256'],
        evaluation_protocol=dict(name='held-out COCO-val2017', image_count=4500,
            image_ids=evaluation_ids, annotation_sha256=sha256(evaluation)),
        historical_provenance_limits='Registration verifies current engines/evidence/cache/ONNX hashes and logged commands; no retroactive proof of unchanged build-time ONNX/cache files. No rebuild or calibration performed.'))
    print(f'Registered four existing engines in {out}; no builds/calibration. Native scales: {scales}')


if __name__ == '__main__':
    main()
