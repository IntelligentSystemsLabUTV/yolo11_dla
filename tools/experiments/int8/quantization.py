"""TRT 10.3 native INT8 I/O metadata; no TensorRT or torch dependency."""
import json
import math
import re
import struct
from pathlib import Path

from common import sha256


def read_cache(path):
    lines = Path(path).read_text().splitlines()
    if not lines or not lines[0].startswith('TRT-'):
        raise ValueError('Not a TensorRT calibration cache')
    result = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        name, value = line.rsplit(':', 1)
        name, value = name.strip(), value.strip()
        if name in result or not re.fullmatch(r'[0-9a-fA-F]{8}', value):
            raise ValueError(f'Duplicate/malformed cache entry: {name}')
        scale = struct.unpack('!f', bytes.fromhex(value))[0]
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f'Invalid calibration scale: {name}')
        result[name] = scale
    return result


def native_scales(cache, log, names):
    """Use exact cache floats, cross-check explicitly assigned I/O ranges in build log.

    TRT release/10.3 samples/common/sampleEngines.cpp sets native I/O ranges
    from calibration scales * 127 (setTensorScalesFromCalibration).
    The comparison tolerance is for six-significant-digit log formatting only.
    """
    scales = read_cache(cache)
    text = Path(log).read_text()
    if '&&&& PASSED TensorRT.trtexec' not in text or '[E]' in text:
        raise ValueError('Require successful verbose build log')
    ranges = {}
    for name, low, high in re.findall(
            r'Setting dynamic range for (.+?) to \[([^,]+),([^\]]+)\]', text):
        if name in ranges:
            raise ValueError(f'Ambiguous binding range: {name}')
        ranges[name] = (float(low), float(high))
    result = {}
    for name in names:
        if name not in scales or name not in ranges:
            raise ValueError(f'Missing exact cache scale / explicit build range for {name}')
        low, high = ranges[name]
        if not (math.isclose(-low, scales[name]*127, rel_tol=2e-5) and
                math.isclose(high, scales[name]*127, rel_tol=2e-5)):
            raise ValueError(f'Cache/build binding scale mismatch: {name}')
        result[name] = {'scale': scales[name], 'zero_point': 0,
                        'logged_range': [low, high]}
    return result


def load_sidecar(engine, names):
    path = Path(str(engine)+'.quantization.json')
    if not path.is_file():
        raise ValueError(f'INT8 bindings require verified scales: missing {path}; run int8-reuse')
    data = json.loads(path.read_text())
    if data.get('schema') != 1 or data.get('engine_sha256') != sha256(engine):
        raise ValueError('Quantization metadata does not match engine')
    if set(data.get('bindings', {})) != set(names):
        raise ValueError('Quantization metadata binding names differ')
    for name, entry in data['bindings'].items():
        scale = entry['scale']
        if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0 or entry.get('zero_point') != 0:
            raise ValueError(f'Invalid symmetric INT8 scale: {name}')
    return data
