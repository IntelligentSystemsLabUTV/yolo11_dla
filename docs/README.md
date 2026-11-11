# Documentation

The documentation, manuscript and derived results are organized here:

| Path | Contents |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Model architecture, initialization and training recipe |
| [ENVIRONMENT.md](ENVIRONMENT.md) | Hardware and software environment |
| [BUILD.md](BUILD.md) | Export, TensorRT builds and placement verification |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Measurement protocol and commands |
| [RESULTS.md](RESULTS.md) | Accuracy, latency and energy results with their sources |
| [PROVENANCE.md](PROVENANCE.md) | Artifact provenance and experimental limitations |
| [paper/](paper/README.md) | LaTeX manuscript, PDF, tables, publication figures, editorial audits and archived drafts |
| [analysis/](analysis/README.md) | Derived CSV/JSON results, validation records and evidence audits |
| [img/](img/) | Images used by the repository documentation |
| [DLA-YOLO11N.md](DLA-YOLO11N.md) | Detailed YOLO11-DLA technical notes |

Training and measurement scripts live in [tools/experiments/](../tools/experiments/README.md).
Raw campaign artifacts live under `logs/` at the repository root.

Build the manuscript from the repository root with `make -C docs/paper`.
Regenerate its tables and figures with
`python tools/experiments/run.py precision-results`.
