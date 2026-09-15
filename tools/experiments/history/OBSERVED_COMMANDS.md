# Observed commands, recovered from logs and attachments

An archive, not a runbook: the outputs and paths are the original ones and no longer resolve.

For new measurements follow [`docs/EXPERIMENTS.md`](../../../docs/EXPERIMENTS.md), always with new output directories.


## CMD-001 — PASSED

Source: [logs/fp16_matrix/yolo11n-dla-gpu-fallback-fp16.benchmark.log](../../../logs/fp16_matrix/yolo11n-dla-gpu-fallback-fp16.benchmark.log), line 1.


```bash
trtexec --loadEngine=logs/paper/fp16_matrix/yolo11n-dla-gpu-fallback-fp16.engine --useDLACore=0 --warmUp=2000 --duration=30 --useSpinWait --exportTimes=logs/paper/fp16_matrix/yolo11n-dla-gpu-fallback-fp16.times.json
```


## CMD-002 — PASSED

Source: [logs/fp16_matrix/yolo11n-dla-gpu-fallback-fp16.build.log](../../../logs/fp16_matrix/yolo11n-dla-gpu-fallback-fp16.build.log), line 1.


```bash
trtexec --onnx=logs/paper/yolo11n.onnx --saveEngine=logs/paper/fp16_matrix/yolo11n-dla-gpu-fallback-fp16.engine --useDLACore=0 --allowGPUFallback --fp16 --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --profilingVerbosity=detailed --dumpLayerInfo --exportLayerInfo=logs/paper/fp16_matrix/yolo11n-dla-gpu-fallback-fp16.layers.json --skipInference --verbose
```


## CMD-003 — PASSED

Source: [logs/fp16_matrix/yolo11n-dla-gpu-fallback-fp16.profile.log](../../../logs/fp16_matrix/yolo11n-dla-gpu-fallback-fp16.profile.log), line 1.


```bash
trtexec --loadEngine=logs/paper/fp16_matrix/yolo11n-dla-gpu-fallback-fp16.engine --useDLACore=0 --warmUp=1000 --duration=0 --iterations=100 --separateProfileRun --dumpProfile --exportProfile=logs/paper/fp16_matrix/yolo11n-dla-gpu-fallback-fp16.profile.json
```


## CMD-004 — PASSED

Source: [logs/fp16_matrix/yolo11n-dla-gpu-fp16.benchmark.log](../../../logs/fp16_matrix/yolo11n-dla-gpu-fp16.benchmark.log), line 1.


```bash
trtexec --loadEngine=logs/paper/fp16_matrix/yolo11n-dla-gpu-fp16.engine --warmUp=2000 --duration=30 --useSpinWait --exportTimes=logs/paper/fp16_matrix/yolo11n-dla-gpu-fp16.times.json
```


## CMD-005 — PASSED

Source: [logs/fp16_matrix/yolo11n-dla-gpu-fp16.build.log](../../../logs/fp16_matrix/yolo11n-dla-gpu-fp16.build.log), line 1.


```bash
trtexec --onnx=logs/paper/yolo11n-dla.onnx --saveEngine=logs/paper/fp16_matrix/yolo11n-dla-gpu-fp16.engine --fp16 --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --profilingVerbosity=detailed --dumpLayerInfo --exportLayerInfo=logs/paper/fp16_matrix/yolo11n-dla-gpu-fp16.layers.json --skipInference
```


## CMD-006 — PASSED

Source: [logs/fp16_matrix/yolo11n-dla-strict-dla-fp16.benchmark.log](../../../logs/fp16_matrix/yolo11n-dla-strict-dla-fp16.benchmark.log), line 1.


```bash
trtexec --loadEngine=logs/paper/fp16_matrix/yolo11n-dla-strict-dla-fp16.engine --useDLACore=0 --warmUp=2000 --duration=30 --useSpinWait --exportTimes=logs/paper/fp16_matrix/yolo11n-dla-strict-dla-fp16.times.json
```


## CMD-007 — PASSED

Source: [logs/fp16_matrix/yolo11n-dla-strict-dla-fp16.build.log](../../../logs/fp16_matrix/yolo11n-dla-strict-dla-fp16.build.log), line 1.


```bash
trtexec --onnx=logs/paper/yolo11n-dla.onnx --saveEngine=logs/paper/fp16_matrix/yolo11n-dla-strict-dla-fp16.engine --useDLACore=0 --fp16 --inputIOFormats=fp16:dla_hwc4 --outputIOFormats=fp16:chw16 --profilingVerbosity=detailed --dumpLayerInfo --exportLayerInfo=logs/paper/fp16_matrix/yolo11n-dla-strict-dla-fp16.layers.json --skipInference --verbose
```


## CMD-008 — PASSED

Source: [logs/fp16_matrix/yolo11n-dla-strict-dla-fp16.profile.log](../../../logs/fp16_matrix/yolo11n-dla-strict-dla-fp16.profile.log), line 1.


```bash
trtexec --loadEngine=logs/paper/fp16_matrix/yolo11n-dla-strict-dla-fp16.engine --useDLACore=0 --warmUp=1000 --duration=0 --iterations=100 --separateProfileRun --dumpProfile --exportProfile=logs/paper/fp16_matrix/yolo11n-dla-strict-dla-fp16.profile.json
```


## CMD-009 — PASSED

Source: [logs/fp16_matrix/yolo11n-gpu-fp16.benchmark.log](../../../logs/fp16_matrix/yolo11n-gpu-fp16.benchmark.log), line 1.


```bash
trtexec --loadEngine=logs/paper/fp16_matrix/yolo11n-gpu-fp16.engine --warmUp=2000 --duration=30 --useSpinWait --exportTimes=logs/paper/fp16_matrix/yolo11n-gpu-fp16.times.json
```


## CMD-010 — PASSED

Source: [logs/fp16_matrix/yolo11n-gpu-fp16.build.log](../../../logs/fp16_matrix/yolo11n-gpu-fp16.build.log), line 1.


```bash
trtexec --onnx=logs/paper/yolo11n.onnx --saveEngine=logs/paper/fp16_matrix/yolo11n-gpu-fp16.engine --fp16 --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --profilingVerbosity=detailed --dumpLayerInfo --exportLayerInfo=logs/paper/fp16_matrix/yolo11n-gpu-fp16.layers.json --skipInference
```


## CMD-011 — FAILED

Source: [logs/fp16_matrix/yolo11n-strict-dla-fp16.build.log](../../../logs/fp16_matrix/yolo11n-strict-dla-fp16.build.log), line 1.


```bash
trtexec --onnx=logs/paper/yolo11n.onnx --saveEngine=logs/paper/fp16_matrix/yolo11n-strict-dla-fp16.engine --useDLACore=0 --fp16 --inputIOFormats=fp16:dla_hwc4 --outputIOFormats=fp16:chw16 --profilingVerbosity=detailed --dumpLayerInfo --exportLayerInfo=logs/paper/fp16_matrix/yolo11n-strict-dla-fp16.layers.json --skipInference --verbose
```


## CMD-012 — terminal_summary

Source: [history/sources/2026-08-25-e2e-final-sparse.txt](sources/2026-08-25-e2e-final-sparse.txt), line 1.


```bash
python tools/yolo_e2e_tester.py \
  --engine logs/paper/fp16_matrix/yolo11n-dla-strict-dla-fp16.engine \
  --image logs/biascica_simone.jpg \
  --dla-core 0 \
  --post-device cpu \
  --warmup 10 \
  --runs 100
```


## CMD-013 — terminal_summary

Source: [history/sources/2026-08-25-e2e-final-sparse.txt](sources/2026-08-25-e2e-final-sparse.txt), line 40.


```bash
python tools/yolo_e2e_tester.py \
  --engine logs/paper/fp16_matrix/yolo11n-dla-gpu-fp16.engine \
  --image logs/biascica_simone.jpg \
  --warmup 10 \
  --runs 100
```


## CMD-014 — terminal_summary

Source: [history/sources/2026-08-25-e2e-final-sparse.txt](sources/2026-08-25-e2e-final-sparse.txt), line 82.


```bash
python tools/yolo_e2e_tester.py \
  --engine logs/paper/fp16_matrix/yolo11n-gpu-fp16.engine \
  --image logs/biascica_simone.jpg \
  --post-device cpu \
  --warmup 10 \
  --runs 100
```


## CMD-015 — terminal_summary

Source: [history/sources/2026-08-25-e2e-final-sparse.txt](sources/2026-08-25-e2e-final-sparse.txt), line 115.


```bash
python tools/yolo_e2e_tester.py \
  --engine logs/paper/fp16_matrix/yolo11n-dla-gpu-fallback-fp16.engine \
  --image logs/biascica_simone.jpg \
  --dla-core 0 \
  --post-device cpu \
  --warmup 10 \
  --runs 100
```


## CMD-016 — terminal_summary

Source: [history/sources/2026-08-25-e2e-initial-default-stream.txt](sources/2026-08-25-e2e-initial-default-stream.txt), line 1.


```bash
python tools/yolo_e2e_tester.py \
  --engine logs/paper/fp16_matrix/yolo11n-gpu-fp16.engine \
  --image logs/biascica_simone.jpg \
  --post-device cpu \
  --cpu-threads 1 \
  --warmup 20 \
  --runs 200
```


## CMD-017 — terminal_summary

Source: [history/sources/2026-08-25-e2e-initial-default-stream.txt](sources/2026-08-25-e2e-initial-default-stream.txt), line 35.


```bash
python tools/yolo_e2e_tester.py \
  --engine logs/paper/fp16_matrix/yolo11n-dla-gpu-fallback-fp16.engine \
  --image logs/biascica_simone.jpg \
  --dla-core 0 \
  --post-device cpu \
  --cpu-threads 1 \
  --warmup 20 \
  --runs 200
```


## CMD-018 — terminal_summary

Source: [history/sources/2026-08-25-e2e-initial-default-stream.txt](sources/2026-08-25-e2e-initial-default-stream.txt), line 70.


```bash
python tools/yolo_e2e_tester.py \
  --engine logs/paper/fp16_matrix/yolo11n-dla-gpu-fp16.engine \
  --image logs/biascica_simone.jpg \
  --post-device cpu \
  --cpu-threads 1 \
  --warmup 20 \
  --runs 200
```


## CMD-019 — terminal_summary

Source: [history/sources/2026-08-25-e2e-initial-default-stream.txt](sources/2026-08-25-e2e-initial-default-stream.txt), line 109.


```bash
python tools/yolo_e2e_tester.py \
  --engine logs/paper/fp16_matrix/yolo11n-dla-strict-dla-fp16.engine \
  --image logs/biascica_simone.jpg \
  --dla-core 0 \
  --post-device cpu \
  --cpu-threads 1 \
  --warmup 20 \
  --runs 200
```


## CMD-020 — PASSED

Source: [history/sources/2026-08-25-initial-adapted-strict.txt](sources/2026-08-25-initial-adapted-strict.txt), line 1.


```bash
trtexec --onnx=logs/yolo11n-dla-500ep.onnx --saveEngine=logs/yolo11n-dla-500ep.engine --useDLACore=0 --inputIOFormats=fp16:dla_hwc4 --outputIOFormats=fp16:chw16 --fp16
```


## CMD-021 — PASSED

Source: [history/sources/2026-08-25-initial-stock-gpu.txt](sources/2026-08-25-initial-stock-gpu.txt), line 1.


```bash
trtexec --onnx=logs/yolo11n.onnx --saveEngine=logs/yolo11n.engine --fp16
```
