# Preserved audits

Long-form records written while preparing the manuscript, kept verbatim except for cross-reference paths, which were repointed to this repository's layout. Prose inside them still names the development workspace (`tools/YOLO11-DLA_ICRA/`, `tools/yolo11_dla_paper/`, `logs/paper/...`); those are provenance strings, not paths that resolve here.

| File | Contents |
|---|---|
| `EVIDENCE_AUDIT.md` | Source and provenance audit of the historical FP16 evidence: measurement definitions, quantitative findings, statistical limits, figure captions |
| `MODEL_AUDIT.md` | Implementation provenance, training discrepancies, runtime caveats |
| `PRECISION_UPDATE.md` | Sources, interpretation and reproduction commands for the combined FP16/INT8 results |
| `REFERENCE_AUDIT.md` | Primary-source verification of every citation |
| `GUIDELINES.md` | Submission requirements the manuscript was prepared against, verified against the call and the publisher's policies |
| `NOTES.md` | Revision record, including claims that earlier drafts made and that were subsequently corrected or removed |

`NOTES.md` is written chronologically: statements that engine AP or energy were "missing" describe an earlier draft and are superseded by the measured results in [`../../../docs/RESULTS.md`](../../../docs/RESULTS.md).
