import argparse
import json
from pathlib import Path
import struct

import pytest

from common import sha256, write_json
from int8.reuse import main as register
from suite import reusable_evaluations, validate_inputs


def test_register_existing_matrix_and_verify_scales_without_runtime(tmp_path, monkeypatch):
    matrix = tmp_path/'matrix'
    matrix.mkdir()
    split = tmp_path/'split'
    split.mkdir()
    calibration = tmp_path/'calibration'
    calibration.mkdir()
    entries = [{'image_id':i,'sha256':'fixture','file_name':str(i)+'.jpg'} for i in range(500)]
    evaluation_ids = list(range(500,5000))
    write_json(split/'evaluation.json', {'images':[{'id':i} for i in evaluation_ids]})
    write_json(split/'manifest.json', dict(status='complete',calibration_count=500,evaluation_count=4500,
        calibration_images=entries,evaluation_image_ids=evaluation_ids,annotations_sha256='original',
        evaluation_annotations_sha256=sha256(split/'evaluation.json')))
    names = ('images','pred_p3','pred_p4','pred_p5')
    for model in ('stock','adapted'):
        (tmp_path/(model+'.onnx')).write_bytes(model.encode())
        directory = calibration/model
        directory.mkdir()
        cache = 'TRT-100300-EntropyCalibration2\n'+''.join(n+': '+struct.pack('!f',.125).hex()+'\n' for n in names)
        (directory/'calibration.cache').write_text(cache)
        write_json(directory/'manifest.json', dict(status='complete',model=model,
            onnx_sha256=sha256(tmp_path/(model+'.onnx')),cache_sha256=sha256(directory/'calibration.cache'),
            algorithm='IInt8EntropyCalibrator2',calibrate_before_fusion=True, images_consumed=500,
            tensorrt_version='10.3.0',calibration_images=entries,preprocess_sha256='same',
            split_manifest_sha256=sha256(split/'manifest.json')))
    for name,model,core in [('stock-gpu','stock',None),('adapted-gpu','adapted',None),
                            ('stock-fallback','stock',0),('adapted-strict','adapted',0)]:
        directory = matrix/name
        directory.mkdir()
        (directory/'model.engine').write_bytes(name.encode())
        layers = [{'LayerType':'DLA' if core == 0 else 'CaskConvolution'}]
        write_json(directory/'layers.json',{'Layers':layers})
        native = name == 'adapted-strict'
        cmd = ['trtexec',f'--onnx={tmp_path/(model+".onnx")}',f'--calib={calibration/model/"calibration.cache"}',
               f'--saveEngine={directory/"model.engine"}', f'--exportLayerInfo={directory/"layers.json"}',
               '--int8','--fp16','--inputIOFormats='+('int8:dla_hwc4' if native else 'fp16:chw'),
               '--outputIOFormats='+('int8:chw32' if native else 'fp16:chw')]
        if core is not None:cmd.append('--useDLACore=0')
        if name == 'stock-fallback':cmd.append('--allowGPUFallback')
        ranges = ''.join(f'Setting dynamic range for {n} to [-15.875,15.875]\n' for n in names) if native else ''
        import shlex
        (directory/'build.log').write_text(ranges+'&&&& PASSED TensorRT.trtexec [TensorRT v100300] # '+shlex.join(cmd)+'\n')
    output = tmp_path/'ready'
    monkeypatch.setattr('sys.argv',['int8-reuse','--matrix',str(matrix),'--calibration',str(calibration),
        '--split',str(split),'--stock-onnx',str(tmp_path/'stock.onnx'),
        '--adapted-onnx',str(tmp_path/'adapted.onnx'),'--output',str(output)])
    register()
    data = json.loads((output/'manifest.json').read_text())
    assert data['status'] == 'complete' and len(data['jobs']) == 4
    assert (output/'adapted-strict.engine').is_symlink()
    assert data['quantization']['bindings']['images']['scale'] == .125
    args = argparse.Namespace(artifacts=None, engines=output, stage='accuracy',
                              annotations=output/'evaluation.json', stock_checkpoint=None, adapted_checkpoint=None)
    assert validate_inputs(args)['engine_source_kind'] == 'archived_int8_matrix'
    (output/'adapted-strict.engine.quantization.json').write_text('{}')
    with pytest.raises(ValueError,match='evidence changed'):
        validate_inputs(args)


def test_reused_ap_checks_images_and_does_not_require_new_runtime_hash(tmp_path):
    images = tmp_path/'images'
    images.mkdir()
    (images/'one.jpg').write_bytes(b'fixture image')
    annotations = tmp_path/'annotations.json'
    write_json(annotations,{'images':[{'id':1,'file_name':'one.jpg'}]})
    engines = tmp_path/'engines'
    engines.mkdir()
    (engines/'stock-gpu.engine').write_bytes(b'engine')
    old = tmp_path/'old'
    old.mkdir()
    write_json(old/'manifest.json',dict(status='complete',artifact_sha256=sha256(engines/'stock-gpu.engine'),
        annotation_sha256=sha256(annotations),image_ids=[1],metrics_fraction={'AP':.33},
        runtime_script_sha256='previous runtime; unchanged FP16 math',
        arguments=dict(limit=None,conf=.001,iou=.7,max_det=300,cpu_threads=1,sparse_decode=False)))
    (old/'frames.jsonl').write_text(json.dumps({'image_id':1,'image_sha256':sha256(images/'one.jpg')})+'\n')
    write_json(old/'predictions.json',[])
    args = argparse.Namespace(reuse_evaluation=[f'stock-gpu={old}'],stage='accuracy',engines=engines,
        images=images,annotations=annotations,limit=None,cpu_threads=1)
    assert reusable_evaluations(args,[{'name':'stock-gpu'}]) == {'stock-gpu':old.resolve()}
    (images/'one.jpg').write_bytes(b'changed')
    with pytest.raises(ValueError,match='image sequence mismatch'):
        reusable_evaluations(args,[{'name':'stock-gpu'}])
