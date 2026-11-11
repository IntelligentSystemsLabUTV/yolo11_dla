# Inventario degli script

Una sola implementazione per file; i vecchi percorsi rimandano a quelli canonici.

[orchestration/reuse_existing.py](../orchestration/reuse_existing.py), invocabile
con `run.py reuse-existing`, registra engine e risultati storici senza export o
build. È un nuovo script, senza un precedente percorso da mantenere.

La preparazione INT8 aggiunge [int8/split_coco.py](../int8/split_coco.py),
[int8/calibrate.py](../int8/calibrate.py) e [int8/build.py](../int8/build.py),
invocabili con `run.py int8-split`, `int8-calibrate` e `int8-build`.
Il protocollo e i comandi trtexec sono in [INT8_RUNBOOK.md](INT8_RUNBOOK.md).

| Percorso attuale | Percorso precedente |
|---|---|
| [benchmarks/yolo_e2e_tester.py](../benchmarks/yolo_e2e_tester.py) | `tools/yolo_e2e_tester.py` |
| [training/train_coco_dla.py](../training/train_coco_dla.py) | `tools/train_coco_dla.py` |
| [training/coco_dla_train.sh](../training/coco_dla_train.sh) | `tools/coco_dla_train.sh` |
| [training/monitoring/coco_results_server.sh](../training/monitoring/coco_results_server.sh) | `tools/coco_results_server.sh` |
| [training/monitoring/wandb_live_sync.py](../training/monitoring/wandb_live_sync.py) | `tools/wandb_live_sync.py` |
| [training/monitoring/wandb_sync_once.py](../training/monitoring/wandb_sync_once.py) | `tools/wandb_sync_once.py` |
| [training/monitoring/wandb_live_sync.sh](../training/monitoring/wandb_live_sync.sh) | `tools/wandb_live_sync.sh` |
| [analysis/analyze_fp16.py](../analysis/analyze_fp16.py) | `tools/YOLO11-DLA_ICRA/analysis/analyze_fp16.py` |
| [analysis/extract_checkpoint.py](../analysis/extract_checkpoint.py) | `tools/YOLO11-DLA_ICRA/analysis/extract_checkpoint.py` |
| [evaluation/evaluate_coco.py](../evaluation/evaluate_coco.py) | `tools/YOLO11-DLA_ICRA/analysis/evaluate_coco.py` |
| [evaluation/verify_model.py](../evaluation/verify_model.py) | `tools/YOLO11-DLA_ICRA/analysis/verify_model.py` |
| [paper/check_paper.py](../paper/check_paper.py) | `tools/YOLO11-DLA_ICRA/analysis/check_paper.py` |
| [common.py](../common.py) | `tools/YOLO11-DLA_ICRA/analysis/jetson/common.py` |
| [export/prepare.py](../export/prepare.py) | `tools/YOLO11-DLA_ICRA/analysis/jetson/prepare.py` |
| [evaluation/validate.py](../evaluation/validate.py) | `tools/YOLO11-DLA_ICRA/analysis/jetson/validate.py` |
| [benchmarks/benchmark.py](../benchmarks/benchmark.py) | `tools/YOLO11-DLA_ICRA/analysis/jetson/benchmark.py` |
| [benchmarks/telemetry.py](../benchmarks/telemetry.py) | `tools/YOLO11-DLA_ICRA/analysis/jetson/telemetry.py` |
| [orchestration/suite.py](../orchestration/suite.py) | `tools/YOLO11-DLA_ICRA/analysis/jetson/suite.py` |
| [analysis/summarize.py](../analysis/summarize.py) | `tools/YOLO11-DLA_ICRA/analysis/jetson/summarize.py` |
| [tests/test_protocol.py](../tests/test_protocol.py) | `tools/YOLO11-DLA_ICRA/analysis/jetson/test_protocol.py` |
| [docs/JETSON_RUNBOOK.md](../docs/JETSON_RUNBOOK.md) | `tools/YOLO11-DLA_ICRA/analysis/jetson/README.md` |
| [docs/LOCAL_VALIDATION.md](../docs/LOCAL_VALIDATION.md) | `tools/YOLO11-DLA_ICRA/analysis/jetson/LOCAL_VALIDATION.md` |
| [configs/hardware-notes.example.md](../configs/hardware-notes.example.md) | `tools/YOLO11-DLA_ICRA/analysis/jetson/hardware-notes.example.md` |
| [docs/FINAL_TESTS.md](../docs/FINAL_TESTS.md) | `tools/YOLO11-DLA_ICRA/FINAL_TESTS.md` |

## Dipendenze nel fork Ultralytics

Il codice di architettura e decoding condiviso resta nel repository del modello:

- [yolo11-dla.yaml](../../ultralytics/ultralytics/cfg/models/11/yolo11-dla.yaml): configurazione nano/scale.
- [block.py](../../ultralytics/ultralytics/nn/modules/block.py): C2DLA e DLAAttention.
- [head.py](../../ultralytics/ultralytics/nn/modules/head.py): DetectDLA.
- [dla.py](../../ultralytics/ultralytics/utils/dla.py), [nms.py](../../ultralytics/ultralytics/utils/nms.py): decoder e NMS.
- [examples/train_dla.py](../../ultralytics/examples/train_dla.py): ricetta opzionale a due fasi; uso nel training del checkpoint non dimostrato.
- [examples/dla_decode.py](../../ultralytics/examples/dla_decode.py): verifica decoder originale.
- [examples/test_dla_engine.py](../../ultralytics/examples/test_dla_engine.py): storico esempio segmentation; non è il validatore detection del paper.

Questi file sono referenziati, non copiati fuori dal fork. Il trainer storico
conserva le sue opzioni anche per modelli esclusi; la suite finale seleziona solo
stock YOLO11n e YOLO11-DLA-n.

Hash dei file prima dello spostamento: [migration.json](../history/migration.json).
