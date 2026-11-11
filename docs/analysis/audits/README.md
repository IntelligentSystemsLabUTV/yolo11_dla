# Preserved audits

Long-form records written while the measurements were being collected and checked, kept verbatim except for cross-reference paths, which were repointed to this repository's layout. Prose inside them still names the development workspace (`tools/YOLO11-DLA_ICRA/`, `tools/yolo11_dla_paper/`, `logs/paper/...`); those are provenance strings, not paths that resolve here, and commands such as `make -C tools/YOLO11-DLA_ICRA precision-results` describe how the record was produced at the time, not how to produce it now — see [`../README.md`](../README.md) for the current entry points.

| File | Contents |
|---|---|
| `EVIDENCE_AUDIT.md` | Source and provenance audit of the historical FP16 evidence: measurement definitions, quantitative findings, statistical limits |
| `MODEL_AUDIT.md` | Implementation provenance, training discrepancies, runtime caveats |
| `PRECISION_UPDATE.md` | Sources, interpretation and reproduction commands for the combined FP16/INT8 results |

The editorial audits that accompanied these — citation verification, submission requirements, and the manuscript revision record — belong to the manuscript and are not distributed here.
