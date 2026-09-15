#!/usr/bin/env python3
"""Build the INT8-enabled matrix using the historical FP16 I/O and placement flags."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import configure_imports
configure_imports()

import argparse
import json
import re
import shlex
import subprocess
from common import new_directory, provenance, sha256, write_json
from suite import inspect_strict


def commands(trtexec, stock, adapted, stock_cache, adapted_cache, output, stock_strict=False):
    rows = [('yolo11n-gpu-int8', stock, stock_cache, None, False, False),
            ('yolo11n-dla-gpu-int8', adapted, adapted_cache, None, False, False),
            ('yolo11n-dla-gpu-fallback-int8', stock, stock_cache, 0, True, False),
            ('yolo11n-dla-strict-dla-int8', adapted, adapted_cache, 0, False, True)]
    if stock_strict:
        rows.append(('yolo11n-strict-dla-int8', stock, stock_cache, 0, False, True))
    jobs = []
    for name, onnx, cache, core, fallback, native in rows:
        cmd = [trtexec, f'--onnx={onnx}', f'--saveEngine={output/name}.engine']
        if core is not None:
            cmd += [f'--useDLACore={core}']
        if fallback:
            cmd += ['--allowGPUFallback']
        cmd += ['--int8', '--fp16', f'--calib={cache}',
                '--inputIOFormats=int8:dla_hwc4' if native else '--inputIOFormats=fp16:chw',
                '--outputIOFormats=int8:chw32' if native else '--outputIOFormats=fp16:chw',
                '--profilingVerbosity=detailed', '--dumpLayerInfo',
                f'--exportLayerInfo={output/name}.layers.json', '--skipInference']
        if core is not None:
            cmd += ['--verbose']
        jobs.append({'name': name, 'command': cmd, 'optional_negative_control': name == 'yolo11n-strict-dla-int8'})
    return jobs


def check_cache(directory, onnx, model):
    data = json.loads((directory/'manifest.json').read_text())
    if data['status'] != 'complete' or data['model'] != model or data['onnx_sha256'] != sha256(onnx):
        raise ValueError(f'Calibration does not match this {model} ONNX: {directory}')
    if data['cache_sha256'] != sha256(directory/'calibration.cache'):
        raise ValueError(f'Calibration cache changed: {directory}')
    if data['algorithm'] != 'IInt8EntropyCalibrator2' or not data['calibrate_before_fusion']:
        raise ValueError('Require before-fusion EntropyCalibrator2 for shared GPU/DLA calibration')
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stock-onnx', type=Path, required=True)
    p.add_argument('--adapted-onnx', type=Path, required=True)
    p.add_argument('--stock-calibration', type=Path, required=True)
    p.add_argument('--adapted-calibration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--trtexec', default='/usr/src/tensorrt/bin/trtexec')
    p.add_argument('--stock-strict', action='store_true', help='Also attempt the optional stock strict compilation control')
    p.add_argument('--dry-run', action='store_true', help='Print shell commands only; no cache checks, directories or builds')
    args = p.parse_args()
    out = args.output.resolve()
    jobs = commands(args.trtexec, args.stock_onnx.resolve(), args.adapted_onnx.resolve(),
                    args.stock_calibration.resolve()/'calibration.cache',
                    args.adapted_calibration.resolve()/'calibration.cache', out, args.stock_strict)
    if args.dry_run:
        for job in jobs:
            print(shlex.join(job['command'])+' > '+shlex.quote(str(out/(job['name']+'.build.log')))+' 2>&1')
        return
    stock = check_cache(args.stock_calibration, args.stock_onnx, 'stock')
    adapted = check_cache(args.adapted_calibration, args.adapted_onnx, 'adapted')
    if stock['split_manifest_sha256'] != adapted['split_manifest_sha256']:
        raise ValueError('Use the same fixed calibration images for both models')
    if stock['preprocess_sha256'] != adapted['preprocess_sha256']:
        raise ValueError('Calibration preprocessing code differs between the two models')
    help_text = subprocess.check_output([args.trtexec, '--help'], stderr=subprocess.STDOUT, text=True)
    version = re.search(r'TensorRT v(\d+)', help_text)
    for cache in (stock, adapted):
        major, minor, patch = map(int, cache['tensorrt_version'].split('.')[:3])
        if not version or int(version[1]) != major*10000+minor*100+patch:
            raise ValueError('trtexec version does not match calibration TensorRT; calibrate on the target stack')
    new_directory(out)
    record = {'status': 'running', 'jobs': jobs, 'provenance': provenance(),
              'stock_calibration': stock, 'adapted_calibration': adapted,
              'precision': 'INT8 enabled, FP16 allowed; native INT8 I/O for strict, FP16 I/O otherwise; inspect actual layer precisions',
              'scope': 'INT8 build results, not accuracy or benchmark results'}
    write_json(out/'manifest.json', record)
    try:
        for job in jobs:
            print(f'[INT8 build] {job["name"]}', flush=True)
            with (out/(job['name']+'.build.log')).open('w') as log:
                result = subprocess.run(job['command'], stdout=log, stderr=subprocess.STDOUT)
            job['returncode'] = result.returncode
            if result.returncode:
                text = (out/(job['name']+'.build.log')).read_text()
                rejection = re.search(r'not supported on DLA|unsupported.*DLA|could not be assigned to DLA|cannot.*DLA|DLA.*does not support', text, re.I)
                if not job['optional_negative_control'] or not rejection or 'Finished parsing network model' not in text:
                    raise RuntimeError(f'Build failed: {job["name"]}; inspect the build log')
                job['outcome'] = 'compiler_rejected_strict_dla'
            else:
                engine = out/(job['name']+'.engine')
                if not engine.is_file() or not engine.stat().st_size:
                    raise RuntimeError(f'No engine produced: {job["name"]}')
                job['engine_sha256'] = sha256(engine)
                job['inspection'] = inspect_strict(out/(job['name']+'.layers.json'))
                if job['name'] == 'yolo11n-dla-strict-dla-int8' and not job['inspection']['one_dla_no_gpu_or_reformat']:
                    raise RuntimeError('Adapted strict INT8 build is not a single DLA loadable without GPU/reformat')
            write_json(out/'manifest.json', record)
        record['status'] = 'complete'
    except BaseException as exc:
        record.update(status='failed', error=str(exc))
        raise
    finally:
        write_json(out/'manifest.json', record)


if __name__ == '__main__':
    main()
