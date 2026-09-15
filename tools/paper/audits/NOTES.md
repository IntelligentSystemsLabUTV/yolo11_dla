# Eight-page YOLO11-DLA paper — revision record

Updated 2026-09-10 with the audited `run_01` campaign. Start with [main.pdf](../main.pdf) and [FINAL_TESTS.md](../../../docs/EXPERIMENTS.md).

## Current manuscript update — run_01

The abstract, introduction, deployment description, evaluation, discussion, and
conclusion now include the six completed COCO engine evaluations, the four
multi-image pipeline runs, and module energy at 10 Hz. See
[RUN_01_AUDIT.md](../../../docs/PROVENANCE.md) for checks and sources.

- Adapted GPU/strict DLA: 37.995/37.991 AP; stock GPU: 39.314 AP.
- Pipeline median: strict 56.820 ms versus fallback 51.241 ms.
- Module energy: strict 0.874 J/frame versus fallback 0.917 and adapted GPU 0.761.
- The raw numerical gate remains failed; dense/sparse AP is practically unchanged.
- The new tables replace the old provisional single-image pipeline transcription.
  The historical training figure remains archived but is omitted from the current
  manuscript to prioritize deployed-engine results within eight pages.
- The author confirms NVIDIA 50 W mode, `jetson_clocks` disabled, ROS 2 Jazzy,
  and YOLO11n initialization through compatible-weight transfer. The archived
  trainer revision reconstructs MuSGD automatic selection and the training recipe;
  see [TRAINING_RECONSTRUCTION.md](../../../docs/ARCHITECTURE.md).
- The manuscript describes measured tests and frame percentiles, retaining that
  uncertainty across independent repetitions was not estimated. Original manifests
  still record one launch per configuration. No extra repetitions are required
  for the current agreed scope.
- INT8 is temporarily omitted. Concurrency is outside the measured claims, and
  paired checkpoint AP is not used to infer an export effect. The only remaining
  manuscript placeholder is author verification/disclosure (P8).

The following sections are the historical revision record; statements that engine
AP or energy were missing describe the earlier draft and are superseded above.

Jetson test preparation updated 2026-09-09: [analysis/jetson/README.md](../../../docs/EXPERIMENTS.md)
contains executable scripts for the remaining YOLO11-DLA tests. GPU concurrency
is deferred pending the author's workload choice. The sparse host decoder and
six-candidate NMS edge case were corrected; historical measurements have not
been replaced with results from the revised runtime.

The paper now focuses on the DLA-compatible YOLO11 nano detector. The independent YOLO-DLA model is excluded from its experiments. The previous source, figures, notes, and nine-page PDF are preserved under `drafts/before_revision_2026-09-08/`; the reference project `../YOLO-DLA_ICRA/`, model code, and original logs were not edited.

## Main corrections and supported result

- Switched from IEEEtran to the official `ieeeconf` Letter/10pt conference template, retaining its margins. The local class matches the official template after line-ending normalization.
- The draft is exactly **eight pages including references and acknowledgment**, anonymous, with no identifying metadata or embedded PDF links. Experimental placeholders remain visible, so it is not ready for submission.
- Reframed the contribution around implemented compatibility, explicit graph/I/O boundaries, and measured execution. Removed unmeasured energy/isolation benefits, claims that fallback is never useful, and claims that two changes are a proved minimum.
- Corrected C2DLA to input-dependent **channel softmax plus 3×3 local depthwise convolution**. It does not preserve global self-attention or exactly implement the cited external-attention formulation.
- Removed unsupported AP 37.68/reference 39.40/95.6% retention. The retained checkpoint's **500-epoch training record reports AP 37.581 and AP50=53.151**. A fresh paired baseline evaluation and serialized-engine AP remain missing.
- Distinguished an available full-fine-tune record from an optional, unverified two-phase/block-transfer recipe in the source.
- Raw build/inspection evidence confirms one strict DLA loadable and no GPU-assigned learned layer. The fallback filename is misleading: it builds **stock YOLO11n**, not the adapted model.
- Strict adapted median engine-plus-transfer latency is 27.515 ms versus 34.678 ms for stock fallback: **20.7% lower**, with **27.6% greater saturated throughput**. The adapted GPU engine is 3.500 ms, so DLA remains 7.86× slower in that same-model engine comparison.
- YOLO11-family end-to-end JSON records are missing. Retained summary values are labeled provisional; combined I/O values are sums of stage medians and include transfers. No raw measurements from the independent YOLO-DLA model were substituted.

## Deliverables

| File | Purpose |
|---|---|
| `main.tex`, `sections/`, `references.bib` | Editable eight-page manuscript |
| `figures/boundary.tex`, `figures/c2dla.tex` | Architecture and block schematics |
| `figures/latency_comparison.pdf`, `fallback_profile.pdf`, `training_history.pdf` | Reproducible vector figures; SVG copies included |
| `analysis/analyze_fp16.py` | Raw-log audit, CSV/JSON summaries, figure regeneration |
| `analysis/extract_checkpoint.py` | Re-extracts the retained training history and metadata |
| `analysis/verify_model.py`, `model_verification.json` | Local deterministic CPU ONNX/decoder verification |
| `analysis/evaluate_coco.py` | Matched PyTorch/native-layout TensorRT COCO evaluator for final tests |
| `analysis/check_paper.py`, `paper_preflight.json` | Local PDF/font/link/page/reference checks |
| `analysis/EVIDENCE_AUDIT.md` | Raw evidence and statistical limitations |
| `MODEL_AUDIT.md` | Implementation provenance, training discrepancies, runtime caveats |
| `FINAL_TESTS.md` | P1–P8 instructions, real commands, and remaining harness integration |
| `GUIDELINES.md`, `REFERENCE_AUDIT.md` | Official ICRA 2027 rules and primary-source reference verification |

## Build and verified checks

```bash
make -C tools/YOLO11-DLA_ICRA
make -C tools/YOLO11-DLA_ICRA audit
make -C tools/YOLO11-DLA_ICRA check-model
```

`make` uses the local Tectonic installation. `audit` needs Python, NumPy, Matplotlib, PyTorch, and the trusted supplied checkpoint. `check-model` additionally needs ONNX and ONNX Runtime. `check_paper.py` uses PyMuPDF.

Completed: raw-log assertions over 20,167 timing samples; trained-model CPU dense decoder parity; ONNX validation and CPU raw-output agreement; CPU integration smoke test of the COCO evaluation runner using an explicitly synthetic annotation fixture; exact eight-page build with 12 embedded fonts, no Type 3 fonts, links, bookmarks, overfull boxes, or undefined references. The synthetic smoke AP is not a research result and is not in the manuscript. The TensorRT branch of the new evaluator has not been tested on this host.

Final experiment priorities: P1 hardware/artifact identity; P2 matched independent accuracy; P3 serialized-engine accuracy/numerical parity; P4 repeated multi-image timing; P6 GPU contention and robotics replay; P7 measured board energy. P8 is author verification and final disclosure/submission preparation.

The new PDF names OpenAI Codex (GPT-6) and its actual assistance, as required for substantive AI-generated content. It leaves author verification explicitly pending. Do not distribute private checkpoint metadata, source-path audits, or identifying JSON as an anonymous supplement.

## Figure/reference revision

Figures 1 and 2 are now full-width vector schematics with a shared palette and larger Helvetica labels. Figure 1 shows the actual three-scale top-down/bottom-up neck, the C2DLA replacement, raw DetectDLA outputs, and CPU boundary. Figure 2 shows both the split/process/merge wrapper and the channel-softmax/local-convolution block, with both residuals. The former comparison table repeated these diagrams and was removed; the introduction was tightened to retain exactly eight pages without changing margins or body font size.

The bibliography increased from 17 to 26 cited references, adding YOLOv3/v4, YOLOX, YOLOv6/v9, CSPNet, FPN, PANet, and RepVGG. Primary-source verification is appended to `REFERENCE_AUDIT.md`.

- `figures/boundary.tikz`, `figures/c2dla.tikz`: editable diagram content.
- `figures/architecture_styles.tex`: shared colors and typography.
- `figures/architecture_overview.pdf`, `figures/c2dla_detail.pdf`: standalone vector exports; PNG previews are beside them.
- `make diagrams`: rebuild the standalone PDFs; `make` rebuilds the manuscript directly from the TikZ sources.

The prior eight-page version is preserved in `drafts/before_figure_revision/`. The revised PDF passes the local eight-page check with 14 embedded fonts and no overfull boxes, undefined references, bitmap fonts, or embedded links. Existing experimental placeholders remain unchanged.

All paper scripts are now centralized in [../yolo11_dla_paper/](../../experiments/README.md).
The recovered August 25 terminal records and command provenance are documented in
[HISTORICAL_COMMANDS.md](../../../docs/PROVENANCE.md).
