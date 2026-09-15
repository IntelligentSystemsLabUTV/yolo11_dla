import json
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from common import sha256
from int8.quantization import read_cache, native_scales, load_sidecar
from validate import reference_storage, check_layout
from yolo_e2e_tester import TensorRTEngine, TensorMeta, _physical_elements


def test_exact_cache_scale_and_build_range_gate(tmp_path):
    cache = tmp_path/'cache'
    cache.write_text('TRT-100300-EntropyCalibration2\nimages: '+struct.pack('!f', 0.125).hex()+'\n')
    log = tmp_path/'build.log'
    log.write_text('Setting dynamic range for images to [-15.875,15.875]\n&&&& PASSED TensorRT.trtexec\n')
    assert native_scales(cache, log, ['images'])['images']['scale'] == 0.125
    log.write_text('Setting dynamic range for images to [-127,127]\n&&&& PASSED TensorRT.trtexec\n')
    with pytest.raises(ValueError, match='mismatch'):
        native_scales(cache, log, ['images'])
    cache.write_text('TRT-100300-EntropyCalibration2\nimages: 7fc00000\n')
    with pytest.raises(ValueError, match='Invalid'):
        read_cache(cache)


def test_sidecar_requires_matching_engine_and_all_binding_scales(tmp_path):
    engine = tmp_path/'model.engine'
    engine.write_bytes(b'engine')
    with pytest.raises(ValueError, match='missing'):
        load_sidecar(engine, ['images'])
    data = {'schema': 1, 'engine_sha256': sha256(engine), 'bindings': {'images': {'scale': .25, 'zero_point': 0}}}
    Path(str(engine)+'.quantization.json').write_text(json.dumps(data))
    assert load_sidecar(engine, ['images']) == data
    with pytest.raises(ValueError, match='names'):
        load_sidecar(engine, ['images','pred_p3'])
    engine.write_bytes(b'different')
    with pytest.raises(ValueError, match='match engine'):
        load_sidecar(engine, ['images'])


@pytest.mark.parametrize('fmt,shape,components', [('DLA_HWC4',(1,3,2,9),4), ('CHW32',(1,35,2,3),32)])
def test_int8_native_quantized_layout_against_independent_oracle(monkeypatch, fmt, shape, components):
    import yolo_e2e_tester as tester
    monkeypatch.setattr(tester, 'trt', SimpleNamespace(TensorFormat=SimpleNamespace(**{fmt:fmt})))
    runner = object.__new__(TensorRTEngine)
    runner.engine = SimpleNamespace(get_tensor_format=lambda name: fmt)
    runner.dla_hwc4_line_bytes = 64
    physical = _physical_elements(shape, fmt, torch.int8, components, 64)
    meta = TensorMeta('x',shape,torch.int8,fmt,'fixture',components,1,(),physical)
    runner.metadata = {'x':meta}
    runner.quantization = {'bindings': {'x': {'scale':0.25,'zero_point':0}}}
    values = np.resize(np.array([-100,-32.125,-.625,-.375,.375,.625,31.875,100],np.float32), shape)
    # np.rint is an independent ties-to-even oracle, including saturation endpoints.
    quantized = torch.from_numpy(np.clip(np.rint(values/.25),-128,127).astype(np.int8))
    expected = torch.from_numpy(reference_storage(meta,quantized))
    torch.testing.assert_close(runner._pack_input(torch.from_numpy(values),meta),expected,rtol=0,atol=0)
    runner.device_buffers = {'x':expected}
    torch.testing.assert_close(runner._unpack_output('x',torch.device('cpu')),quantized.float()*.25,rtol=0,atol=0)
    assert check_layout(runner)['x']['passed']
