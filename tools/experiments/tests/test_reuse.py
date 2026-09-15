"""Archived engines must stay usable without exports, with integrity checks intact."""
import argparse
import json
from pathlib import Path

import pytest

from reuse_existing import PREFIXES, register
from suite import plan, validate_inputs


@pytest.fixture
def archive(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    for name, prefix in PREFIXES.items():
        negative = name == 'stock-strict'
        onnx = 'yolo11n.onnx' if name.startswith('stock') else 'yolo11n-dla.onnx'
        flags = ' --useDLACore=0' if name in ('stock-strict', 'stock-fallback', 'adapted-strict') else ''
        if name == 'stock-fallback':
            flags += ' --allowGPUFallback'
        result = 'FAILED' if negative else 'PASSED'
        command = f'trtexec --onnx={onnx} --saveEngine={prefix}.engine --fp16{flags}'
        (source/f'{prefix}.build.log').write_text(
            f'Finished parsing network model\nMatMul\n&&&& {result} TensorRT.trtexec [TensorRT v100300] # {command}\n')
        if not negative:
            # Synthetic bytes for file-integrity testing; never deserialized.
            (source/f'{prefix}.engine').write_bytes(name.encode())
            (source/f'{prefix}.layers.json').write_text(json.dumps({'Layers': [
                {'LayerType': 'DlaLayer' if name == 'adapted-strict' else 'Convolution'}]}))
            (source/f'{prefix}.benchmark.log').write_text(
                f'&&&& PASSED TensorRT.trtexec [TensorRT v100300] # trtexec --loadEngine={prefix}.engine\n')
            (source/f'{prefix}.times.json').write_text('[{"latencyMs": 1}]')
    return source


def test_registration_preserves_sources_and_checks_engine_integrity(archive, tmp_path):
    before = {p.name: p.read_bytes() for p in archive.iterdir()}
    out = register(archive, tmp_path/'registered')
    assert before == {p.name: p.read_bytes() for p in archive.iterdir()}
    args = argparse.Namespace(artifacts=None, engines=out/'engines', stage='parity')
    assert validate_inputs(args)['engine_source_kind'] == 'archived_fp16_matrix'
    assert (out/'engines/stock-fallback.engine').resolve().name == 'yolo11n-dla-gpu-fallback-fp16.engine'
    (archive/'yolo11n-gpu-fp16.engine').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Engine missing or changed'):
        validate_inputs(args)


def test_registration_rejects_wrong_fallback_model_before_writing(archive, tmp_path):
    log = archive/'yolo11n-dla-gpu-fallback-fp16.build.log'
    log.write_text(log.read_text().replace('yolo11n.onnx', 'yolo11n-dla.onnx'))
    out = tmp_path/'registered'
    with pytest.raises(ValueError, match='Unexpected model'):
        register(archive, out)
    assert not out.exists()


def test_accuracy_without_exports_can_add_a_checkpoint(archive, tmp_path):
    out = register(archive, tmp_path/'registered')
    args = argparse.Namespace(stage='accuracy', artifacts=None, engines=out/'engines',
                              output=tmp_path/'accuracy', images=Path('/images'),
                              annotations=Path('/annotations'), cpu_threads=1, limit=None,
                              adapted_checkpoint=None, stock_checkpoint=None)
    assert len(plan(args)) == 6
    assert validate_inputs(args)['checkpoint_sha256'] == {}
    args.adapted_checkpoint = tmp_path/'adapted.pt'
    args.adapted_checkpoint.write_bytes(b'checkpoint fixture')
    assert len(plan(args)) == 7
    assert set(validate_inputs(args)['checkpoint_sha256']) == {'adapted'}
    assert not any('--onnx' in token for job in plan(args) for token in job['command'])
