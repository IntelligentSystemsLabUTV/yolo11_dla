"""INT8 preparation must preserve held-out IDs and pair each graph with its own real cache."""
import json
from pathlib import Path

import pytest

from common import sha256
from int8.build import check_cache, commands
from int8.split_coco import partition


def test_calibration_split_is_deterministic_disjoint_and_filters_annotations():
    data = {'images': [{'id': i, 'file_name': f'{i}.jpg'} for i in reversed(range(5000))],
            'annotations': [{'id': i, 'image_id': i, 'category_id': 1} for i in range(5000)],
            'categories': [{'id': 1}]}
    calibration, heldout = partition(data, 500, 2027)
    a = {image['id'] for image in calibration['images']}
    b = {image['id'] for image in heldout['images']}
    assert len(a) == 500 and len(b) == 4500 and not a & b and a | b == set(range(5000))
    assert {ann['image_id'] for ann in heldout['annotations']} == b
    assert partition(data, 500, 2027) == (calibration, heldout)
    assert partition(data, 500, 2028) != (calibration, heldout)
    with pytest.raises(ValueError):
        partition(data, 5000, 2027)


def test_int8_matrix_native_strict_bindings_and_correct_fallback_source():
    jobs = commands('trtexec', Path('/stock.onnx'), Path('/adapted.onnx'),
                    Path('/stock.cache'), Path('/adapted.cache'), Path('/out'))
    assert len(jobs) == 4
    for job in jobs:
        command = job['command']
        assert '--int8' in command and '--fp16' in command and '--skipInference' in command
        if 'strict' not in job['name']:
            assert not any('IOFormats=int8' in arg for arg in command)
        if 'fallback' in job['name']:
            assert '--onnx=/stock.onnx' in command and '--calib=/stock.cache' in command
            assert '--allowGPUFallback' in command
        if 'strict' in job['name']:
            assert '--calib=/adapted.cache' in command and '--allowGPUFallback' not in command
            assert '--inputIOFormats=int8:dla_hwc4' in command and '--outputIOFormats=int8:chw32' in command


def test_cache_rejects_different_model_and_modified_data(tmp_path):
    onnx = tmp_path/'model.onnx'
    onnx.write_bytes(b'ONNX fixture')
    (tmp_path/'calibration.cache').write_bytes(b'TRT-fixture')
    data = {'status': 'complete', 'model': 'stock', 'onnx_sha256': sha256(onnx),
            'cache_sha256': sha256(tmp_path/'calibration.cache'),
            'algorithm': 'IInt8EntropyCalibrator2', 'calibrate_before_fusion': True}
    (tmp_path/'manifest.json').write_text(json.dumps(data))
    assert check_cache(tmp_path, onnx, 'stock') == data
    with pytest.raises(ValueError, match='does not match'):
        check_cache(tmp_path, onnx, 'adapted')
    (tmp_path/'calibration.cache').write_bytes(b'changed')
    with pytest.raises(ValueError, match='cache changed'):
        check_cache(tmp_path, onnx, 'stock')
