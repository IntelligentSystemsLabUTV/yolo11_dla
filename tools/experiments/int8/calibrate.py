#!/usr/bin/env python3
"""Calibrate an existing static ONNX on the target TensorRT 10.x stack, using EntropyCalibrator2."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import ROOT, configure_imports
configure_imports()

import argparse
import json
from common import new_directory, provenance, sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--onnx', type=Path, required=True)
    p.add_argument('--model', choices=('stock', 'adapted'), required=True)
    p.add_argument('--split', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import cv2
    import torch
    import tensorrt as trt
    from yolo_e2e_tester import preprocess

    if not trt.__version__.startswith('10.') or not hasattr(trt, 'IInt8EntropyCalibrator2'):
        raise RuntimeError('This workflow requires TensorRT 10.x implicit calibration; use the same target stack as trtexec')
    split = json.loads((args.split/'manifest.json').read_text())
    if split['status'] != 'complete':
        raise ValueError('Incomplete calibration split')
    entries = split['calibration_images']
    if not entries or len(entries) != split['calibration_count']:
        raise ValueError('Invalid calibration image list')
    for entry in entries:
        if sha256(args.images/entry['file_name']) != entry['sha256']:
            raise ValueError(f'Calibration image changed: {entry["file_name"]}')
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.cuda.set_device(0)
    logger = trt.Logger(trt.Logger.VERBOSE)
    trt.init_libnvinfer_plugins(logger, '')
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # TensorRT 10 uses explicit batch by default.
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(args.onnx.resolve())):
        raise ValueError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    if network.num_inputs != 1 or tuple(network.get_input(0).shape) != (1,3,640,640):
        raise ValueError('Expected one static input (1,3,640,640)')
    inp = network.get_input(0)
    if inp.dtype != trt.float32:
        raise ValueError('Calibration workflow expects the original FP32 ONNX input')
    expected = [(1,84,8400)] if args.model == 'stock' else [(1,144,s,s) for s in (80,40,20)]
    outputs = [tuple(network.get_output(i).shape) for i in range(network.num_outputs)]
    if outputs != expected:
        raise ValueError(f'Unexpected {args.model} outputs: {outputs}')
    if any(network.get_layer(i).type in (trt.LayerType.QUANTIZE, trt.LayerType.DEQUANTIZE)
           for i in range(network.num_layers)):
        raise ValueError('Q/DQ ONNX is not supported by this implicit-calibration workflow')
    out = new_directory(args.output)
    record = {'status': 'running', 'model': args.model, 'onnx_sha256': sha256(args.onnx),
              'tensorrt_version': trt.__version__, 'split_manifest_sha256': sha256(args.split/'manifest.json'),
              'preprocess_sha256': sha256(ROOT/'benchmarks/yolo_e2e_tester.py'),
              'calibration_images': entries, 'input_name': inp.name, 'input_shape': list(inp.shape),
              'algorithm': 'IInt8EntropyCalibrator2', 'calibrate_before_fusion': True,
              'calibration_batch': 1, 'calibration_input': 'FP32 normalized RGB, centered letterbox640, pad114',
              'provenance': provenance()}
    write_json(out/'manifest.json', record)

    class Calibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self.index = 0
            self.error = None
            self.buffer = torch.empty((1,3,640,640), dtype=torch.float32, device='cuda:0')

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            try:
                if list(names) != [inp.name]:
                    raise ValueError(f'Unexpected calibration bindings: {names}')
                if self.index == len(entries):
                    return None
                entry = entries[self.index]
                image = cv2.imread(str(args.images/entry['file_name']))
                if image is None:
                    raise ValueError(f'Cannot decode {entry["file_name"]}')
                tensor, _ = preprocess(image, (640,640))
                self.buffer.copy_(tensor)
                torch.cuda.synchronize(0)
                self.index += 1
                if self.index % 50 == 0 or self.index == len(entries):
                    print(f'Calibration {self.index}/{len(entries)} images', flush=True)
                return [int(self.buffer.data_ptr())]
            except Exception as exc:
                self.error = str(exc)
                raise

        def read_calibration_cache(self):
            return None  # A new output directory always means fresh real-image calibration.

        def write_calibration_cache(self, cache):
            (out/'calibration.cache').write_bytes(bytes(cache))

    try:
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        calibrator = Calibrator()
        config.int8_calibrator = calibrator
        serialized = builder.build_serialized_network(network, config)
        if serialized is None or calibrator.error or calibrator.index != len(entries):
            raise RuntimeError(f'Calibration/build incomplete: {calibrator.index}/{len(entries)} images; {calibrator.error}')
        cache = out/'calibration.cache'
        if not cache.is_file() or not cache.read_bytes().startswith(b'TRT-'):
            raise RuntimeError('TensorRT did not write a valid calibration cache')
        # TensorRT produces a GPU engine while generating the cache. Preserve it as evidence.
        # Its FP32 bindings differ from the FP16-boundary paper matrix built with trtexec.
        (out/'calibration-gpu.engine').write_bytes(bytes(serialized))
        record.update(status='complete', images_consumed=calibrator.index,
                      cache_sha256=sha256(cache), calibration_engine_sha256=sha256(out/'calibration-gpu.engine'),
                      calibration_engine_note='GPU cache-generation artifact with original FP32 bindings; not a paper-matrix engine')
    except BaseException as exc:
        record.update(status='failed', error=str(exc))
        raise
    finally:
        write_json(out/'manifest.json', record)
    print(f'Calibration cache ready: {out / "calibration.cache"}')


if __name__ == '__main__':
    main()
