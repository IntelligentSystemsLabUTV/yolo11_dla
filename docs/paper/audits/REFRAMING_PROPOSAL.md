# ICRA reframing proposal — 11 September 2026

Status: applied and refined on 11 September 2026. The revised LaTeX sources and
`main.pdf` are the authoritative deliverables. The original proposal below is
retained as design history; its replacement passages are not a second copy of
the final manuscript. The applied revision uses “Embedded Robotic Perception”
in the title, adds exactly four verified academic references, replaces
“independently placeable” with explicit DLA assignment without an isolation
claim, and compresses repeated prose to retain eight pages. No new experiments
or measurements were introduced. See `REFERENCE_AUDIT.md` and
`docs/analysis/reframing_validation.json` for the applied revision's checks.

## A. Diagnosis

The paper already contains the evidence and qualifications needed for an embedded
robotic deployment study. Its hierarchy still gives the detector adaptation and
TensorRT compatibility greater prominence than the system-design question.
The title foregrounds adaptation to an accelerator; the Introduction moves from
robots to compiler details before fully establishing why explicit placement
matters. Related Work devotes substantial space to detector evolution, and the
contribution paragraph understates the static accelerator/host boundary as an
independent contribution.

The strongest reframing is to make verified placement the problem, YOLO11 the
representative case, and the accuracy/engine/pipeline/energy comparisons the
answer. The negative pipeline result becomes a central design lesson. The
concurrency limitation remains explicit in the scope discussion, while the
Conclusion ends on demonstrated deployment capability. Use “two localized
adaptations,” not a proved minimum, a new attention optimum, or a general result
across accelerators.

## B. Proposed narrative — seven sentences

Embedded robots combine perception, localization, mapping, planning, and control
on shared heterogeneous compute resources. A dedicated neural accelerator is an
additional placement option only when the workload assigned to it is explicitly
identified. Nominal DLA deployment with GPU fallback does not establish that the
learned graph resides entirely on DLA. YOLO11 provides a representative case in
which learned operations and the prediction interface obstruct strict placement.
YOLO11-DLA makes two localized changes: a compatible learned substitute and a
static interface separating the accelerator graph from host post-processing.
The existing tests establish fallback-free placement and nearly unchanged
adapted-model AP, while showing that host costs reverse the engine-latency
advantage and that the energy comparison depends on the selected baseline.
The contribution is a verified placement option and a deployment methodology
that considers accuracy, engine execution, host overhead, and module energy
jointly, without asserting benefits to concurrent robotic tasks.

## C. Exact change map

Line anchors refer to the current source, before applying this proposal.
Sections E–I give the complete replacements for the title, Abstract,
Introduction, contributions, Discussion, and Conclusion. Section D gives the
remaining localized replacements. Do not apply a replacement twice.

| ID | Location / exact paragraph anchor | Current purpose | ICRA framing problem | Replacement |
|---|---|---|---|---|
| T | `main.tex`, `\\title` | Names YOLO11 adaptation and DLA compatibility | Robotics deployment is absent from the title | E |
| A | `main.tex`, complete `abstract` environment | Summarizes adaptation and measurements | Opens with GPU sharing but immediately becomes an export problem | F |
| I1 | `sections/introduction.tex:3`, “Embedded robots share compute…” through paragraph beginning “YOLO11…” | Motivation, platform, example detector | System need for explicit placement is underdeveloped | G, paragraphs 1–3 |
| I2 | `sections/introduction.tex:22`, “We present…” and “Our scope…” | Describes modifications and scope | Boundary and co-design rationale are less prominent than module names | G, paragraphs 4–5 |
| I3 | `sections/introduction.tex:38`, “The contribution is threefold…” | Lists implementation and evaluation | Static accelerator boundary is not a distinct contribution | H, already included in G |
| I4 | `sections/introduction.tex:46`, “The adapted graph compiles…” | Summarizes quantitative tradeoffs | Useful evidence; retain and connect to placement choice | G, final paragraph |
| R | `sections/related_work.tex:3–64`, four paragraph groups | Detector history, structure, edge design, mixing/compilers | Detector chronology dominates; robotic deployment is implicit | D1, three groups |
| M1 | `sections/method.tex`, `Design Objective and Scope`, text before Eq. (1) | Defines a useful detector under a compilation constraint | Does not explicitly identify placement as the design variable | D2 |
| M2 | `sections/method.tex`, paragraph after Eq. (2), “The first changes…” | Distinguishes architecture and interface changes | “Architecture comparison” can suggest an isolated architectural effect | D3 |
| M3 | `sections/method.tex`, `C2DLA`, paragraph beginning “The softmax output remains…” | Explains representational limits | Compatibility purpose can be stated more directly | D4 |
| M4 | `sections/method.tex`, `DetectDLA`, paragraph beginning “In export mode…” | Describes exported operations and preserved training | Reads as exporter behavior rather than accelerator boundary design | D5 |
| V | `sections/evaluation.tex:3–9`, opening paragraph | States three broad evaluation questions and provenance | Five deployment boundaries are not explicit as research questions | D6 |
| P | `sections/evaluation.tex:222–229`, “Table…reverses…” | Quantifies pipeline reversal and timing definitions | Add the deployment lesson without weakening the negative result | D7 |
| D | `sections/discussion.tex`, complete section | Summarizes tradeoffs and limitations | Can foreground reusable deployment lessons more strongly | I1 |
| C | `sections/conclusion.tex`, complete section | Recaps results and ends on concurrency | Final emphasis falls on an unmeasured consequence | I2 |

Keep all equations, labels, tables, figures, training details, protocols, and
other experimental prose unchanged. Keep Section IV (Export and Runtime)
unchanged: its placement audit, native layouts, CUDA boundary, and decoder
qualification already provide the required technical evidence. Keep the
failed raw numerical gate and dense/sparse qualification in Section V-E.

## D. Localized replacement text

### D1. Related Work — replace the four groups with these three

**Efficient object detectors.**

YOLO evolved from single-network detection `\cite{redmon2016yolo}` through
multi-scale prediction and feature/training refinements
`\cite{redmon2018yolov3,bochkovskiy2020yolov4}`, anchor-free heads and assignment
`\cite{ge2021yolox}`, deployment-oriented structures `\cite{li2022yolov6}`,
training and aggregation improvements `\cite{wang2023yolov7,wang2024yolov9}`,
and NMS-free prediction `\cite{wang2024yolov10}`. These developments motivate
YOLO11 as a representative detector; GPU benchmarking does not establish
complete placement on DLA. We retain YOLO11's detection towers and
distributional regression `\cite{jocher2024yolo11,li2020gfl}`. Its surrounding
structure draws on feature partitioning and pyramid aggregation
`\cite{wang2020cspnet,lin2017fpn,liu2018panet}`. Unlike RepVGG-style deployment
re-parameterization `\cite{ding2021repvgg,li2022yolov6}`, our adaptation changes
the incompatible attention computation while preserving this surrounding
structure. The objective is a deployable learned graph, not another AP/FLOPs
optimum.

**Hardware-aware and edge network design.**

YOLO-ReT and YOLOBench address edge-GPU detection and controlled evaluation
across embedded hardware `\cite{ganesh2022yoloret,lazarevich2023yolobench}`.
ActNAS examines hardware-sensitive activation choices, including compiler
effects `\cite{sah2025actnas}`. MnasNet and ProxylessNAS incorporate target
latency into architecture selection `\cite{tan2019mnasnet,cai2019proxyless}`;
Once-for-All supports deployment specialization `\cite{cai2020ofa}`; MCUNet
co-designs networks and runtime `\cite{lin2020mcunet}`. Here strict compilation
is a feasibility constraint for a manually specified adaptation. C2DLA uses
projection and normalization related to external attention
`\cite{guo2023external}`, but only mixes channels at each position and adds
local depthwise context. It does not reproduce full external attention or
claim representational superiority.

**Heterogeneous deployment for embedded robotic perception.**

A robotic perception workload needs an explicit accelerator boundary before
its placement can be reasoned about alongside other computation. Compiler
frameworks such as TVM lower graphs for target hardware `\cite{chen2018tvm}`;
TensorRT provides DLA compilation, native I/O formats, and optional GPU fallback
under layer restrictions `\cite{nvidia_dla_working}`. Compiler transformations
make source-level operator inspection insufficient to establish the executed
placement. We therefore combine fallback-disabled compilation with optimized
engine inspection, and assess the resulting detector at engine, host-pipeline,
and module-energy boundaries. The retained SiLU operations illustrate this
need to inspect compiled behavior rather than reject a source-level activation
name. This deployment objective is distinct from optimizing isolated GPU
inference or demonstrating improved concurrent robotic execution.

### D2. Method — opening before the acceptance equation

We treat learned-graph placement as a design constraint for embedded robotic
perception. Let $G_0$ be the reference learned graph and $G$ the adapted graph.
We seek a useful detector whose complete learned computation can be assigned
to the tested DLA configuration:

*Then retain the acceptance equation and its definitions unchanged.*

### D3. Method — paragraph immediately after the two-substitution equation

The first substitution changes an accelerator-incompatible learned computation;
the second defines a static accelerator boundary while preserving the ordinary
training path. Together they retain the surrounding YOLO11 backbone, neck,
and detection towers. The adapted GPU engine provides a same-model placement
control; comparisons with stock YOLO11 additionally include architectural and
training differences and use different prediction interfaces.

### D4. C2DLA — replace “The softmax output remains…” paragraph

The substitute preserves input-dependent feature modulation through channel
softmax and local spatial context through depthwise convolution, while retaining
static four-dimensional tensors and avoiding spatial affinity matrices and
MatMul. Its purpose is compatibility with the tested DLA constraints. Channel
mixing does not exchange information between distant positions, so this is an
architectural tradeoff rather than an algebraic replacement for self-attention
or a claim of greater representational power.

### D5. DetectDLA — replace “In export mode…” paragraph

DetectDLA defines the boundary between a static learned accelerator graph and
host-side detection processing. Export stops before flattening, cross-scale
concatenation, DFL expectation, grid decoding, and sigmoid; confidence filtering
and variable-cardinality NMS remain outside the engine. During training and
ordinary PyTorch inference, DetectDLA invokes the inherited Detect path,
retaining its losses and assignment behavior. The change concerns deployment
partitioning, not a direct-regression or one-to-one prediction formulation.

### D6. Evaluation — replace the opening paragraph

We evaluate five aspects of the deployment problem: (i) whether the complete
learned graph compiles for DLA with fallback disabled; (ii) whether placement
preserves useful detection accuracy; (iii) the latency cost at the engine
boundary; (iv) the complete-pipeline cost including preprocessing, native-format
I/O, decoding, and NMS; and (v) module energy at a matched delivered rate.
Retained build/timing artifacts and later tests of the same serialized engines
support these comparisons. Matching engine hashes, per-image predictions,
timing records, and power traces connect placement to measured behavior.

### D7. Pipeline — replace the interpretation paragraph after Table III

The application boundary reverses the engine-boundary ordering: strict DLA's
median total is 10.9% higher than stock fallback's (Table `\ref{tab:e2e}`).
Their runtime inference-stage medians are 27.781 and 35.143 ms, respectively,
but strict packing/H2D costs 7.747 versus 2.338 ms, D2H/unpacking 6.506 versus
1.518 ms, and post-processing 8.400 versus 6.062 ms. Thus, accelerator placement
must be assessed beyond engine timing: native-format I/O and host processing
can reverse the apparent advantage. Stage medians cannot be summed into an
exact decomposition of median total latency. The inference-stage timer includes
the host runtime call and synchronization, so it differs from Eq.
`\eqref{eq:latency}` and pure kernel timing.

*Retain the following preprocessing/clock paragraph unchanged. It prevents
attributing the observed pipeline differences solely to accelerator hardware.*

## E. Revised title

**YOLO11-DLA: Fallback-Free Object Detection on NVIDIA DLA for Embedded Robotic Systems**

This names the deployment capability and intended application domain without
implying evaluated robotic behavior or GPU isolation. The present title is
technically correct, but this replacement makes the ICRA positioning visible
before the Abstract. “Hardware-aware” is established in the Abstract and
contributions rather than adding another long phrase to the title.

## F. Fully rewritten Abstract

Embedded robots execute perception, localization, mapping, planning, and control
on shared heterogeneous compute resources. Dedicated accelerators offer
additional placement options for these workloads, but their use requires
verifying which parts of a perception model actually execute on them. Modern
detectors can contain learned operations and prediction interfaces that prevent
complete placement on NVIDIA's Deep Learning Accelerator (DLA). We present
YOLO11-DLA, a hardware-aware adaptation with two localized changes: channel
softmax and depthwise convolution replace spatial self-attention, and a static
multi-scale prediction interface separates the learned graph from host decoding
and non-maximum suppression. On Jetson Orin with TensorRT 10.3, the nano model
compiles into one DLA loadable with no GPU-assigned learned layer, whereas
unmodified YOLO11n fails strict compilation. On 5,000 COCO validation images,
adapted GPU and strict DLA achieve 37.995 and 37.991 AP, versus 39.314 for stock
GPU. At FP16, batch one, and $640\times640$, median engine-plus-transfer latency
is 27.515 ms versus 34.678 ms for stock fallback, a 20.7% reduction. However,
complete-pipeline latency is higher: 56.820 ms versus 51.241 ms. In 60 s tests
at 10 Hz, strict DLA uses 0.874 J/frame of module energy versus 0.917 for stock
fallback and 0.761 for adapted GPU. These results establish an additional
learned-workload placement option for embedded robotic perception, with a
measured accuracy, latency, host-processing, and energy tradeoff rather than
universal inference superiority.

## G. Fully rewritten Introduction

Embedded robots combine visual perception, localization, mapping, planning,
and manipulation or control on shared compute resources. Heterogeneous embedded
platforms provide CPUs, GPUs, and dedicated neural accelerators, but assigning
a task to an auxiliary accelerator requires an explicit account of which
computation resides there. A detector intended for DLA execution may still
use GPU kernels, complicating the assumed division of work across the platform.

We distinguish nominal DLA deployment from verified fallback-free placement of
the learned graph. TensorRT can request DLA execution while permitting
unsupported regions to fall back to GPU `\cite{nvidia_dla_working}`. Operator
support, tensor shapes, normalization axes, precision, and I/O layouts therefore
affect placement. GPU efficiency and successful export do not establish that
a complete detector graph can execute on a restricted auxiliary accelerator.

YOLO11 `\cite{jocher2024yolo11}` provides a representative case. Its
convolutional backbone and neck coexist with spatial self-attention and a
decoded prediction interface. Attention matrix products and the flattened
prediction path obstruct strict DLA compilation in the tested configuration.
The question is therefore how localized architectural and interface changes
can enable complete learned-graph placement, and what accuracy, latency, host
processing, and energy costs follow.

YOLO11-DLA preserves the surrounding backbone, neck, detection towers, and
training formulation. A channel-softmax/local-convolution substitute replaces
the incompatible attention computation, while three raw static prediction maps
define the accelerator interface. Decoding, confidence filtering, and NMS
execute on the host. The evaluated nano instance is denoted YOLO11-DLA-n;
DLA throughout denotes NVIDIA's accelerator.

This boundary makes learned-graph placement verifiable without assuming that
CUDA services, transfers, preprocessing, or shared memory disappear. Recovering
GPU capacity for another robotic workload remains a system-level hypothesis;
here we establish the placement capability and quantify its standalone costs
on the tested Orin/TensorRT configuration.

The contributions are threefold. First, a hardware-aware adaptation removes
incompatible learned operations while preserving the surrounding YOLO11
architecture and training formulation. Second, an explicit static prediction
boundary separates the accelerator graph from host decoding and
variable-cardinality post-processing, enabling fallback-disabled compilation
and optimized-engine placement verification. Third, a deployment-level
evaluation distinguishes compiler placement, deployed accuracy, engine latency,
host-pipeline overhead, and module energy at a matched delivered rate. Together,
these contributions characterize a dedicated placement option for embedded
robotic perception on the tested platform.

The adapted graph compiles as one DLA loadable, with engine-plus-transfer
latency 20.7% below stock fallback but 7.86 times its own GPU latency. Strict
placement changes adapted-model AP by 0.00468 points, while complete-pipeline
latency is 10.9% higher than stock fallback. The 10 Hz tests measure 4.7% lower
module energy per frame than fallback and 14.9% higher energy than adapted GPU.
These results show why accelerator placement must be judged jointly with
accuracy, host costs, and the operating point rather than engine timing alone.

## H. Rewritten contribution paragraph

This is paragraph 6 of G, repeated here for review. Insert it once only.

The contributions are threefold. First, a hardware-aware adaptation removes
incompatible learned operations while preserving the surrounding YOLO11
architecture and training formulation. Second, an explicit static prediction
boundary separates the accelerator graph from host decoding and
variable-cardinality post-processing, enabling fallback-disabled compilation
and optimized-engine placement verification. Third, a deployment-level
evaluation distinguishes compiler placement, deployed accuracy, engine latency,
host-pipeline overhead, and module energy at a matched delivered rate. Together,
these contributions characterize a dedicated placement option for embedded
robotic perception on the tested platform.

## I. Revised Discussion and Conclusion

### I1. Discussion — complete replacement, preserving the scope qualifications

**Placement and application cost are distinct design questions.**

Accelerator compatibility depends on architecture and interfaces, not only an
export flag. On the tested Orin/TensorRT stack, two localized adaptations make
the complete learned graph compilable as one DLA loadable. Fallback-disabled
compilation and optimized-engine inspection verify that placement; paired COCO
evaluation separately establishes an adapted-model AP change below 0.005 points.
Neither the source architecture nor build success alone establishes both facts.

The engine-boundary advantage over stock fallback does not survive inclusion
of the host pipeline. Engine-plus-transfer latency decreases by 20.7%, but
complete-pipeline median latency increases by 10.9%. Native-format conversion,
transfers, and external decoding move work to the CPU and memory subsystem.
This reversal demonstrates why embedded perception deployment must be evaluated
at multiple boundaries. At 10 Hz, measured module energy is 4.7% below stock
fallback, while adapted GPU remains both faster and lower-energy than strict
DLA at the tested rate.

Strict placement consequently supplies an independently placeable learned
workload on the tested heterogeneous SoC, and thus an additional scheduling
option for an embedded robot. It does not establish improved scheduling
outcomes. Both models sustain the tested 10 Hz rate without observed deadline
misses; the contribution is verified dedicated placement with quantified costs,
not a claim that the dedicated accelerator is always preferable.

**Scope of the evidence.**

The evidence covers the nano detector at $640\times640$, batch one, FP16, and
the reported Orin/TensorRT stack. Matched image IDs, thresholds, and runtime
conventions support the deployment comparisons, but stock and adapted training
budgets differ. Their AP difference therefore does not isolate the effect of
the attention replacement.

Frame percentiles describe the measured tests; uncertainty across independent
repetitions was not estimated. The 50 W mode with `jetson_clocks` disabled
permits frequency variation, and pipeline timings include the observed host
scheduling and frequency behavior. The 100-image cycle is not a full-dataset
timing distribution, and zero observed misses at 10 Hz do not establish hard
real-time guarantees.

CPU, memory, power, transfers, and runtime remain shared. Strict-pipeline GR3D
utilization reaches 7% in the energy trace, so learned-graph placement does not
imply GPU inactivity or task isolation. The environment includes ROS 2
`\cite{macenski2022ros2}`, but camera acquisition, ROS transport, mapping,
control, and concurrent-task behavior are outside the measured boundary.
Recovery of GPU capacity for another workload remains a system-level hypothesis.

### I2. Conclusion — complete replacement

YOLO11-DLA demonstrates hardware-aware deployment of a modern detector on a
restricted accelerator for embedded robotic perception. Its channel-softmax/
local-convolution substitute and static prediction boundary enable one
fallback-free DLA loadable on the measured Orin/TensorRT stack, where stock
YOLO11n fails strict compilation. Engine-plus-transfer latency is 27.515 ms,
20.7% below stock fallback but 7.86 times the adapted GPU latency. COCO AP is
37.995 on adapted GPU and 37.991 on strict DLA, versus 39.314 on stock GPU.
The complete pipeline reverses the latency ordering: 56.820 ms for strict DLA
versus 51.241 ms for fallback. At 10 Hz, module energy is 0.874 J/frame for
strict DLA, versus 0.917 for fallback and 0.761 for adapted GPU.

Effects on concurrent robotic workloads remain outside these measurements
and require system-level evaluation to establish. The demonstrated result is
an additional learned-workload placement option with quantified tradeoffs.
Modern robotic perception networks may require architecture-level adaptation
before dedicated accelerators can serve as explicit compute resources;
YOLO11-DLA shows why placement, accuracy, engine latency, host overhead, and
energy must be considered jointly.

## Page budget and application boundary

The proposal replaces prose rather than adding experimental sections. Shorten
the detector-history discussion first; preserve all equations, table/figure
contents, experimental details, and qualifications. The repeated paragraph H
is not additional manuscript text. The proposal itself is not part of the
submission. Final page count can only be established after these replacements
are applied and typeset; no PDF revision or new experiment is performed here.

Verification of this proposal: the 34 recorded manuscript, PDF, bibliography,
and figure files remain byte-for-byte unchanged. All 24 citation keys in the
current Related Work are retained in D1; every proposed citation resolves to
an existing bibliography entry. All numerical values in the current Abstract
are retained in F. A consistent approximate prose count across Abstract,
Introduction, Related Work, Discussion, and Conclusion is 97 words lower than
in the current manuscript; this excludes the small localized method/evaluation
replacements and is not a typeset page-count guarantee.
