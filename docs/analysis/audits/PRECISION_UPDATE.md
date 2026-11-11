# FP16 and INT8 manuscript integration — 2026-09-14

The manuscript reports eight configurations: stock YOLO11n on GPU and with
DLA+GPU fallback, and YOLO11-DLA-n on GPU and strict DLA, each in FP16 and INT8.
The INT8 build mode allows FP16 operations; the strict INT8 build has 298 INT8
layers and one FP16 softmax, all assigned to one DLA loadable.

## Evidence used

All paths below are relative to the workspace root.

| Manuscript content | Source |
| --- | --- |
| FP16 held-out accuracy | `logs/icra2027_int8/run_01/fp16-heldout-summary/results.csv` |
| INT8 held-out accuracy | `logs/icra2027_int8/run_01/accuracy-summary/results.csv` |
| FP16 engine timing | `tools/YOLO11-DLA_ICRA/analysis/engine_summary.csv` (historical FP16 timing audit) |
| INT8 engine timing | `logs/icra2027_int8/run_01/micro-summary/results.csv` |
| FP16 pipeline | `logs/icra2027_final/run_01/summary-pipeline/results.csv` |
| INT8 pipeline | `logs/icra2027_int8/run_01/pipeline-summary/results.csv` |
| FP16 module energy | `logs/icra2027_final/run_01/summary-energy-10hz-module/results.csv` |
| INT8 module energy | `logs/icra2027_int8/run_01/energy-summary/results.csv` |
| INT8 placement and I/O | `logs/int8_matrix/` successful build logs and layer inspectors; `logs/int8_ready/` registered engine metadata |
| INT8 layout and decoder checks | `logs/icra2027_int8/run_01/parity/comparison/report.json` |

`precision_results.json` stores the source hashes and imported result rows,
including AP by object size and timing counts not displayed in the compact
manuscript tables. The original FP16 analysis and its figures remain available;
the FP16 placement-profile figure used in the manuscript is explicitly labeled
FP16. No INT8 layer-time profile is inferred from it.

## Comparability and interpretation

- Main AP comparisons use the same 4,500 held-out images, disjoint from the
  500 calibration images selected with seed 2027. FP16 was rescored from saved
  detections; the original full-5,000 FP16 scores are retained separately in
  the text. Calibration uses separate stock/adapted caches and the same images.
- Pipeline and energy compare the same 100-image workloads across precisions.
  They have their own boundaries and thresholds, distinct from accuracy and
  the engine-plus-transfer microbenchmark. INT8 strict I/O conversion is timed.
- Strict INT8 versus strict FP16: AP 37.876 → 35.408; median engine-plus-transfer
  latency 27.515 → 4.167 ms; application latency 56.820 → 30.675 ms; module
  energy at 10 Hz 0.874 → 0.806 J/frame.
- Adapted INT8 GPU remains faster and lower-energy than strict INT8 DLA, with
  AP 35.951 versus 35.408. Thus INT8 improves the strict deployment operating
  point but does not establish universal superiority over GPU.
- Stock INT8 fallback achieves only 19.827 AP under this protocol. Its timing
  and energy are reported, but it is not an accuracy-equivalent baseline.
  The cause of its accuracy loss has not been isolated.
- Raw GPU/DLA tolerance checks fail in both modes; layout and dense/sparse
  decoder checks pass. Aggregate AP agreement is not bitwise equivalence.
- Energy sums the three recorded module rails. Separate FP16 and INT8 idle
  baselines are used; each active test delivered 600 frames at 10 Hz.
- The 50 W mode and disabled `jetson_clocks` are author-reported platform
  settings; the container did not automatically verify those controls.
  Frame percentiles do not estimate uncertainty across independent tests.
- Concurrent robotic-task performance remains unmeasured. No concurrency,
  ablation, or additional YOLO-DLA-family experiment is introduced.

## Reproduction and validation

From the workspace root:

```bash
make -C tools/YOLO11-DLA_ICRA precision-results
make -C tools/YOLO11-DLA_ICRA local
```

The first command regenerates four LaTeX tables, the FP16/INT8 latency plot
(PDF/SVG), and the source-hashed result artifact. It needs Python, NumPy,
Matplotlib, and the archived CSV files. It performs no calibration, engine
build, or inference. The second compiles the paper and checks the eight-page
PDF, embedded fonts, references, links, and overflow. The submission portal
remains the authority for upload-format acceptance.

See `precision_update_validation.json` for numeric and visual checks of this
revision and `paper_preflight.json` for the current PDF format check.
Earlier editorial validation JSON files describe their respective historical
revisions, not the current precision comparison.

Table layout uses the complete single-column width for accuracy, engine timing,
and energy. The application-pipeline table spans both columns and pairs FP16
and INT8 horizontally. This layout is preserved by `precision-results`; see
`table_layout_validation.json` for value and PDF checks.
