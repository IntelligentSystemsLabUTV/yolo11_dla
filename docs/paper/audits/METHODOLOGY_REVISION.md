# Methodology revision

The baseline is the current manuscript at the latest rewrite request. This
revision follows the methodological framing request and preserves the author's
intervening source edits, including the COCO train2017 statement, the absence
of a stock strict INT8 attempt, and the existing rounding of 19.48 AP points.
No training, build, inference, or numerical re-evaluation was performed.

## Section-by-section changelog

- **Abstract:** the placement problem and workflow precede the YOLO11 changes;
  all current AP, latency, and energy values remain. The closing passage now
  gives a direct practical lesson. The stock INT8 fallback score stays in the
  evaluation and Table II, not the Abstract.
- **I — Introduction:** states the practical question, introduces YOLO11 as a
  case study, and follows compatibility analysis, targeted changes, strict
  compilation, engine inspection, native I/O, decoder/accuracy checks, and
  measurements. Contributions now follow workflow → implementation → evaluation.
  Concurrent-task benefits remain motivation, not a measured result.
- **II — Related Work:** shortens the final comparison and repeated caveats;
  the existing cited work and citation order remain unchanged.
- **III — Adaptation Methodology and Design:** adds a compatibility-analysis
  subsection before the YOLO11-specific C2DLA implementation. Equations retain
  their original contents and numbers. DetectDLA illustrates a static boundary
  for this detector. The channel-mixing limitation is stated once; tensor
  shapes, output volumes, architecture, and training settings remain.
  DFL is expanded on first use. The redundant learning-rate consistency
  sentence is removed.
- **IV — Export and Runtime:** presents strict compilation, engine inspection,
  native layouts, precision-specific I/O, and decoder validation as workflow
  steps. Buffer sizes, scales, equations, and runtime details remain.
- **V — Evaluation:** frames existing tests as questions about placement and
  deployment costs, and describes saved artifacts as traceability records.
  Diagnostic raw-map checks and aggregate AP agreement are stated directly.
  All protocols, thresholds, values, tables, and figures remain.
- **VI — Discussion:** draws practical lessons about placement checks, static
  boundaries, native I/O, the application boundary, and quantization. Important
  numerical comparisons remain, together with the GPU advantage and test limits.
- **VII — Conclusion:** summarizes the YOLO11 result and the workflow lesson;
  other models/DLA configurations and concurrent tasks still require validation.

## Contribution paragraph: before

> This work develops a hardware/software co-design approach for verified learned-graph placement. First, an architectural adaptation removes accelerator-incompatible learned computation, and an explicit static prediction boundary separates the remaining learned graph from host-side decoding and variable-cardinality post-processing, retaining YOLO11's surrounding architecture and training formulation. Second, fallback-disabled compilation and optimized-engine inspection verify placement, while native I/O handling connects the DLA loadable to the host runtime. Third, deployed-engine evaluation measures accuracy, engine-plus-transfer latency, application-pipeline latency, host-side overhead, and module energy at a matched delivered rate in FP16 and calibrated INT8, with GPU controls for both models.

## Contribution paragraph: after

> This work makes three contributions. First, it describes a practical workflow for adapting and verifying a detector for fallback-free DLA execution, from compatibility analysis and a static accelerator boundary to fallback-disabled compilation, optimized-engine inspection, and native I/O handling. Second, the workflow is demonstrated on YOLO11 through C2DLA and DetectDLA, retaining the surrounding backbone, neck, detection towers, and training formulation. Third, the resulting deployment is evaluated in FP16 and INT8 for placement, accuracy, engine-plus-transfer latency, application-pipeline latency, host-side cost, and module energy, with GPU controls for both models.

## Abstract closing passage: before

The Abstract is one LaTeX paragraph; the comparison below covers its final
passage following the reported GPU energy value.

> The contribution combines verified learned-graph placement with application-level characterization of precision-dependent accuracy, latency, host-side processing, and energy trade-offs for embedded robotic perception, rather than universal accelerator superiority.

## Abstract closing passage: after

> The case study shows that DLA compatibility must be evaluated together with accuracy, host-side processing, and module energy. Faster accelerator execution alone is not enough to select a deployment.

## Removed or consolidated sentences

The substantive information in the consolidated passages remains in the
indicated section. Ordinary wording replacements are visible in
`methodology_revision.diff`.

- **III — removed:** “The recorded group learning rates are consistent with
  this configuration.” The optimizer, learning rate, multiplier, and schedule
  already provide the reproducibility details.
- **I — removed from the closing paragraph:** “The engine-plus-transfer latency
  advantage over stock fallback reverses in application-pipeline latency,
  demonstrating why placement must be assessed jointly with accuracy, host-side
  overhead, and the operating point.” The measured reversal remains in V-F and
  informs the practical discussion.
- **II — shortened:** “These findings motivate explicit workload placement,
  without establishing concurrent-task benefits for the adapted detector.”
  Replaced by a direct motivation sentence; the concurrency limit remains in
  I, VI, and VII as requested.
- **IV — replaced:** “Build success also does not validate raw-output decoding.”
  Replaced by a direct pointer to the decoder validation step in V-E.
- **V-C — replaced:** “Neither result establishes comparable accuracy or
  application performance.” Replaced by the transition to those measurements.
- **V-E — simplified:** “Raw agreement is therefore reported diagnostically,
  not claimed as a passed numerical gate.” The raw-map results are now described
  as diagnostic, with all failed-image counts retained.
- **V-E — simplified:** “Aggregate accuracy agreement does not establish bitwise
  identity or equality of every post-NMS detection.” Replaced by the direct
  statement that matching AP does not mean every post-NMS detection is identical.
- **III-C — consolidated:** repeated statements about four-dimensional tensors,
  missing spatial affinities, and representational limits now form one concise
  limitation paragraph. Non-equivalence, changed hypothesis class, and absence
  of a representational-superiority claim remain explicit.

## Evidence boundaries

The workflow is demonstrated on one YOLO11 nano variant under the recorded
Jetson/TensorRT conditions. There is no evidence of validated transfer to other
architectures or of improved concurrent-task performance; neither is claimed.
The existing statement that no fallback-disabled stock INT8 build was attempted
is preserved. Available build records do not document such a test. Calibration
and strict-native INT8 failures of other configurations are not reinterpreted
as stock strict INT8 evidence.

The current manuscript specifies COCO train2017, which is preserved as supplied.
The checkpoint records a historical `coco-bbox.yaml` path whose file is absent;
the standard `coco.yaml` alone does not independently confirm that historical
configuration. No new dataset reconstruction is claimed.

## Final verification

The PDF has eight pages including references, with embedded fonts, no overflow,
no unresolved references/citations, and preserved anonymity. All five figures
and four tables are referenced. Equations (1)–(12), table contents, figure assets,
and reference numbering are unchanged. Numeric tokens in each edited source
match the baseline; the highlighted AP/latency/energy values remain exact.
No authorial “we”, “our”, or “ours” remains. All pages were inspected visually.
See `methodology_revision_validation.json` and the exact source diff.
