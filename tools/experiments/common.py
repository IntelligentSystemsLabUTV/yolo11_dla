"""Shared provenance and protocol helpers; importing this module needs no GPU."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from _paths import ROOT, WORKSPACE, PAPER
HERE = ROOT
# A measurement run must not silently change the Python environment during export.
os.environ['YOLO_AUTOINSTALL'] = 'false'
sys.path[:0] = [str(ROOT / 'benchmarks'), str(WORKSPACE / 'tools/ultralytics')]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n')
    temporary.replace(path)


def new_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def capture(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        return {'command': command, 'returncode': result.returncode,
                'stdout': result.stdout, 'stderr': result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'command': command, 'error': str(exc)}


def provenance():
    files = sorted(p for p in ROOT.rglob('*.py') if 'history' not in p.relative_to(ROOT).parts)
    record = {'command': sys.argv, 'python': sys.version, 'platform': platform.platform(),
              'wall_time_ns': time.time_ns(), 'monotonic_ns': time.perf_counter_ns(),
              'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
              'cpu_affinity': sorted(os.sched_getaffinity(0)),
              'scripts_sha256': {str(p.relative_to(WORKSPACE)): sha256(p) for p in files}}
    record['package_versions'] = {}
    for package in ('torch', 'torchvision', 'numpy', 'opencv-python', 'onnx', 'tensorrt', 'pycocotools'):
        try:
            record['package_versions'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            record['package_versions'][package] = 'not in Python distribution metadata; see system queries'
    for name, repo in [('workspace', WORKSPACE), ('ultralytics', WORKSPACE/'tools/ultralytics'),
                       ('paper', PAPER)]:
        record[name] = {key: capture(['git', '-C', str(repo), *args]) for key, args in
                        [('revision', ['rev-parse', 'HEAD']), ('status', ['status', '--short']),
                         ('diff', ['diff', 'HEAD'])]}
    record['system'] = [capture(cmd) for cmd in [
        ['uname', '-a'], ['cat', '/etc/nv_tegra_release'],
        ['cat', '/proc/device-tree/model'], ['nvpmodel', '-q', '--verbose'],
        ['jetson_clocks', '--show'], ['nvcc', '--version'],
        ['dpkg-query', '-W', 'nvidia-jetpack', 'nvidia-l4t-core', 'libnvinfer*', 'libcudnn*']]]
    return record


def image_sequence(images, annotations, limit):
    data = json.loads(Path(annotations).read_text())
    entries = sorted(data['images'], key=lambda entry: entry['id'])
    if limit is not None:
        if limit < 1:
            raise ValueError('limit must be positive')
        entries = entries[:limit]
    if not entries:
        raise ValueError('Empty image sequence')
    result = []
    for entry in entries:
        path = Path(images) / entry['file_name']
        if not path.is_file():
            raise FileNotFoundError(path)
        result.append({'image_id': entry['id'], 'path': str(path.resolve()), 'sha256': sha256(path)})
    return result


def stats(values):
    import numpy as np
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError('Non-finite measurements')
    return dict(zip(['mean', 'median', 'p95', 'p99', 'min', 'max'], map(float,
                    [array.mean(), *np.percentile(array, [50, 95, 99]), array.min(), array.max()])))
