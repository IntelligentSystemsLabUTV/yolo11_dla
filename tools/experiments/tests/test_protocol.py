"""CPU regression tests for measurement validity and native-layout/decoder edge cases."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import configure_imports
configure_imports()

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from benchmark import release_slot
from suite import build_command, inspect_strict, plan
from telemetry import integrate, parse_rails
from validate import decoder_parity, reference_storage
from yolo_e2e_tester import TensorMeta, TensorRTEngine, decode_raw_nms_sparse
from ultralytics.utils.nms import non_max_suppression


def test_six_sparse_anchors_are_not_end2end_detections():
    p = torch.zeros(1,84,6)
    p[0,0] = torch.arange(6)*50+20
    p[0,1:4] = 10
    p[0,4] = 0.9
    result = non_max_suppression(p, nc=80, conf_thres=0.25)[0]
    assert result.shape == (6,6)
    assert torch.all(result[:,5] == 0)
    assert torch.allclose(result[:,4], torch.full((6,),0.9))


@pytest.mark.parametrize('conf', [0.001,0.25,0.5])
def test_sparse_threshold_matches_float_sigmoid(conf):
    center = torch.logit(torch.tensor(conf, dtype=torch.float32))
    below = torch.nextafter(center, torch.tensor(-float('inf')))
    above = torch.nextafter(center, torch.tensor(float('inf')))
    logits = torch.stack([below,center,above])
    maps = [torch.zeros(1,144,1,3)]
    maps[0][:,64:] = -100
    maps[0][0,64,0,:] = logits
    _, count = decode_raw_nms_sparse(maps, [8],16,80,conf,torch.device('cpu'))
    assert count == int((logits.sigmoid() > conf).sum())


def test_dense_sparse_multilabel_and_empty():
    torch.manual_seed(17)
    maps = [torch.randn(1,144,s,s) for s in (8,4,2)]
    for m in maps:
        m[:,64:] -= 9
    for conf,multi in [(0.001,True),(0.25,False)]:
        assert decoder_parity(maps,conf,multi)['passed']


@pytest.mark.parametrize('fmt,shape,components,physical', [
    ('CHW16',(1,19,2,3),16,192),
    ('DLA_HWC4',(1,3,2,9),4,128),
    ('CHW4',(1,3,2,9),4,72),
    ('DLA_LINEAR',(1,3,2,9),1,192),
])
def test_native_layout_padding_against_offset_oracle(fmt,shape,components,physical):
    import numpy as np
    import yolo_e2e_tester as tester
    # No TensorRT/CUDA initialization; expose only enum equality to the real packer.
    original = tester.trt
    tester.trt = SimpleNamespace(TensorFormat=SimpleNamespace(**{fmt: fmt}))
    try:
        runner = object.__new__(TensorRTEngine)
        runner.engine = SimpleNamespace(get_tensor_format=lambda name: fmt)
        runner.dla_hwc4_line_bytes = 64
        meta = TensorMeta('x',shape,torch.float16,fmt,'fixture',components,1,(),physical)
        runner.metadata = {'x':meta}
        logical = torch.arange(np.prod(shape)).reshape(shape).half()
        expected = torch.from_numpy(reference_storage(meta,logical))
        assert expected.numel() == physical
        torch.testing.assert_close(runner._pack_input(logical,meta),expected,rtol=0,atol=0)
        runner.device_buffers = {'x':expected}
        torch.testing.assert_close(runner._unpack_output('x',torch.device('cpu')),logical.float(),rtol=0,atol=0)
    finally:
        tester.trt = original


def test_power_units_integration_and_missing_coverage():
    assert parse_rails('VDD_IN 12000mW/11000mW VDD_CPU_GPU_CV 3500/3000') == {
        'VDD_IN':12, 'VDD_CPU_GPU_CV':3.5}
    rows = [{'monotonic_s':i/10,'rails_w':{'VDD_IN':10+i}} for i in range(11)]
    result = integrate(rows,'VDD_IN',0.05,0.95)
    assert result['joules'] == pytest.approx(13.5)
    with pytest.raises(ValueError,match='bracket'):
        integrate(rows,'VDD_IN',-0.1,1)
    with pytest.raises(ValueError,match='gap'):
        integrate(rows[::5],'VDD_IN',0,1,max_gap_s=0.2)


def test_drop_schedule_and_strict_flags():
    assert release_slot(1.36,1,10,1) == (3,2)
    command = build_command('trtexec',Path('/models'),Path('/out'),'adapted-strict','adapted',0,native=True)
    assert '--allowGPUFallback' not in command
    assert '--inputIOFormats=fp16:dla_hwc4' in command
    assert '--onnx=/models/adapted.onnx' in command


def test_inspector_rejects_gpu_and_unknown_nodes(tmp_path):
    import json
    path = tmp_path/'layers.json'
    for kinds, expected in [(['DlaLayer'],True),(['DlaLayer','Reformat'],False),(['DlaLayer','Mystery'],False)]:
        path.write_text(json.dumps({'Layers':[{'LayerType':k} for k in kinds]}))
        assert inspect_strict(path)['one_dla_no_gpu_or_reformat'] is expected


def test_explicit_end2end_still_supported_at_ambiguous_shape():
    p = torch.zeros(1,84,6)
    p[:,:,4] = 0.9
    assert non_max_suppression(p,nc=80,end2end=True)[0].shape == (84,6)


def test_randomized_plan_preserves_replicates_and_pairing():
    args=argparse.Namespace(output=Path('/out'),artifacts=Path('/artifacts'),engines=Path('/engines'),
                            stage='energy',seed=7,repeats=5,duration=60,fps=10,stabilize_seconds=30,
                            cpu_threads=1,deadline_ms=100,images=Path('/images'),annotations=Path('/ann'),
                            limit=100,tegrastats='tegrastats',rail='VDD_IN')
    jobs=plan(args)
    assert len(jobs) == 25
    for repeat in range(1,6):
        block=[j for j in jobs if j['repeat']==repeat]
        assert {j['configuration'] for j in block} == {
            'idle','stock-gpu','adapted-gpu','stock-fallback','adapted-strict'}
    assert len({j['name'] for j in jobs}) == 25
    assert jobs == plan(args)
    assert all('yolo-dla' not in ' '.join(j['command']) for j in jobs)
