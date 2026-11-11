# Analysis artifacts

For the current YOLO11-DLA-only Jetson test matrix, start with
[raccolta script](../../../tools/experiments/README.md) and [FINAL_TESTS.md](FINAL_TESTS.md).
The scripts cover checkpoint export and engine builds, native-layout and decoder checks,
matched dense/sparse COCO AP, repeated multi-image timing, and board telemetry.
Concurrent GPU workload integration is deferred at the author's request.

The manuscript includes FP16 and INT8 results. The main accuracy comparison
uses the same 4,500 held-out COCO IDs for both precisions, excluding the 500
calibration images. Read [PRECISION_UPDATE.md](../../analysis/audits/PRECISION_UPDATE.md) for the
current sources, interpretation, and reproduction commands.

The completed FP16 `run_01` results remain available. Read
[RUN_01_AUDIT.md](../../../tools/experiments/docs/RUN_01_AUDIT.md) for the deployed
COCO, pipeline, and module-energy audit; `EVIDENCE_AUDIT.md` below describes
the earlier historical artifacts.

Run `python tools/yolo11_dla_paper/run.py audit` from the workspace
root to reproduce the FP16 audit, CSV tables, and three vector figures.
NumPy and Matplotlib are required; no accelerator is needed.

Read [EVIDENCE_AUDIT.md](../../analysis/audits/EVIDENCE_AUDIT.md) for the full source/provenance audit,
measurement definitions, quantitative findings, limitations, and figure
captions. The machine-readable audit is [fp16_audit.json](../../analysis/fp16_audit.json).

The separately extracted checkpoint history is in
[checkpoint_training.json](../../analysis/checkpoint_training.json) and
[checkpoint_training.csv](../../analysis/checkpoint_training.csv). Model/export parity is
checked by [verify_model.py](../../../tools/experiments/evaluation/verify_model.py); see its command-line options and
the generated `model_verification.json`.

Additional tools added during the manuscript revision:

- `extract_checkpoint.py`: regenerate checkpoint history/metadata (private identifying source paths).
- `verify_model.py`: deterministic CPU ONNX and dense-decoder comparison; results in `model_verification.json`.
- `evaluate_coco.py`: matched COCO evaluator for PyTorch and native-format TensorRT; CPU branch smoke-tested; six TensorRT evaluations completed in `run_01`. See `../FINAL_TESTS.md`.
- `check_paper.py`: exactly-eight-page PDF preflight, embedded fonts, links, metadata, and TeX checks.

Gli script risiedono ora in `tools/yolo11_dla_paper/`. I file Python qui sono
collegamenti di compatibilità; CSV, JSON e figure restano artifact del manoscritto.
