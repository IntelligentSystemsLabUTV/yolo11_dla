# Reference audit and focused literature recommendations

Current status (2026-09-11): the applied ICRA reframing cites **30 entries**.
The four additions and their verification are recorded at the end of this file.
Earlier recommendations below are historical; they are not instructions to add
further documentation or references to the current manuscript.

Audit date: 2026-09-08. Scope: the 17 entries in the original `references.bib`. Existing TeX and bibliography were not changed by this audit. The table distinguishes bibliographic correctness from whether a source is needed in the final argument.

## Existing entries

| BibTeX key | Verification and recommended treatment |
|---|---|
| `redmon2016yolo` | Authors, title, CVPR 2016, and pp. 779–788 match the [CVF proceedings](https://openaccess.thecvf.com/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html). Appropriate historical anchor; no need for an extended history of every YOLO release. |
| `wang2023yolov7` | Correct authors, title, venue, year, and pp. 7464–7475 in the [CVF proceedings](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_YOLOv7_Trainable_Bag-of-Freebies_Sets_New_State-of-the-Art_for_Real-Time_Object_Detectors_CVPR_2023_paper.html). Keep if discussing training/design evolution. |
| `wang2024yolov10` | Authors, title, NeurIPS 37, and 2024 match the [official proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c34ddd05eb089991f06f3c5dc36836e0-Abstract-Conference.html). The proceedings also provide DOI `10.52202/079017-3429`. Useful for distinguishing NMS-free detection from this paper's external decode/NMS boundary. |
| `ganesh2022yoloret` | Correct authors, WACV 2022, pp. 3267–3277 in [CVF](https://openaccess.thecvf.com/content/WACV2022/html/Ganesh_YOLO-ReT_Towards_High_Accuracy_Real-Time_Object_Detection_on_Edge_GPUs_WACV_2022_paper.html). Directly relevant edge-GPU detector design; it does not establish NVIDIA DLA compilation. |
| `tan2019mnasnet` | Authors, title, and pp. 2820–2828 match the [CVF version](https://openaccess.thecvf.com/content_CVPR_2019/html/Tan_MnasNet_Platform-Aware_Neural_Architecture_Search_for_Mobile_CVPR_2019_paper.html). The registered IEEE DOI uses alternate pagination 2815–2823; keep a consistent proceedings pagination convention. Useful for actual device latency instead of FLOPs as a target. |
| `cai2019proxyless` | Correct ICLR 2019 authors/title, verified against the [original OpenReview paper](https://openreview.net/pdf?id=HylVB3AqYm). Optional context for target-hardware specialization; this project should not imply that it performs NAS unless it does. |
| `cai2020ofa` | Correct title, authors, ICLR 2020 per the [authors' project and citation](https://hanlab.mit.edu/projects/ofa). The forum URL currently presents an access challenge. Optional; remove if not needed for a specific comparison. |
| `lin2020mcunet` | Authors/title, NeurIPS 33, 2020 match the [official proceedings](https://proceedings.neurips.cc/paper_files/paper/2020/hash/86c51678350f656dcc7f490a43946ee5-Abstract.html). Existing pagination is consistent with proceedings citations. Valid architecture/runtime co-design context, but MCU inference is less direct than the YOLO accelerator work below. |
| `chen2018tvm` | Every substantive field, including pp. 578–594, matches the [USENIX BibTeX](https://www.usenix.org/conference/osdi18/presentation/chen). This sometimes looks unusual but is the publisher's own pagination. Optional if only TensorRT is actually used. |
| `jocher2024yolo11` | Authors, 2024, version 11.0.0, and repository match the [Ultralytics recommended software citation](https://docs.ultralytics.com/models/yolo11). `@misc` is a reasonable classic-BibTeX substitute for `@software`. This is a software citation, not a peer-reviewed YOLO11 paper. Record the actual local fork commit and package version separately; 11.0.0 is the model citation version. |
| `li2020gfl` | Correct authors/title and NeurIPS 33, 2020 per the [official proceedings](https://proceedings.neurips.cc/paper_files/paper/2020/hash/f0bda020d2470f2e74990a07a607ebd9-Abstract.html). Appropriate attribution for distributional box regression; a changed deployment decoder is not a new DFL training loss. |
| `guo2023external` | Correct authors/title, TPAMI 45(5), pp. 5436–5447, DOI; [IEEE Xplore](https://ieeexplore.ieee.org/document/9912362/) distinguishes online publication in October 2022 from the May 2023 journal issue. Keep 2023. Cite as inspiration; do not imply the local EA implementation reproduces all original normalization operations. |
| `nvidia_dla_working` | [Current URL](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-dla.html) is real, but mutable and now discusses support through older releases. It is not a frozen TensorRT 10.3 specification. Prefer the archive/release-note pair below. |
| `nvidia_dla_restrictions` | [Current restrictions page](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dla-layer-restrictions.html) is real. Overlaps heavily with the archived DLA guide. Merge to avoid three vendor citations for one topic. |
| `nvidia_dla_build` | [Current build page](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dla-build-and-run.html) is real. It covers loadables embedded in TensorRT and standalone loadables; these are distinct deployment modes. Merge into the archived guide unless directly needed. |
| `lin2014coco` | Authors, title, ECCV 2014, pp. 740–755 and DOI match [Springer](https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48). Essential when reporting COCO metrics; specify the actual 2017 train/validation split and evaluation protocol in the paper. |
| `macenski2022ros2` | Bibliographic fields agree with the [author manuscript](https://arxiv.org/abs/2211.07752) and DOI registry: Science Robotics 7(66), eabm6074, 2022. Publisher lists Steven Macenski and Chris Lalancette; current entry already does so. Keep only if the paper discusses an actual ROS 2 system or motivates middleware-specific constraints. |

## TensorRT versioning and technical claim boundaries

The purported `archives/tensorrt-1030/...` guide and PDF URLs redirected to the latest documentation during this audit; they must not be represented as a successfully checked immutable archive. The accessible [TensorRT 10.x DLA guide](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-with-dla.html) is a legacy family archive, whereas [10.3.0 release notes](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.3.0.html) explicitly identify that release. Cite both as appropriate, and use the project's actual TensorRT 10.3 compiler logs as the decisive evidence of model placement.

The archived guide supports these bounds:

- DLA uses FP16/INT8 and static build/runtime dimensions.
- ReLU, sigmoid, and elementwise product are supported. SiLU is not a listed fused activation, but its mathematical decomposition is not automatically unsupported.
- Orin supports softmax with shape-dependent limits; blanket claims that DLA cannot execute softmax are incorrect.
- Shuffle/slice rank restrictions and convolution CBUF capacity can obstruct compilation even when an operator family is supported.
- GPU fallback can introduce graph partitions and reformatting. Successful fallback-enabled execution does not prove all inference ran on DLA.
- A TensorRT engine built without GPU fallback differs from a standalone DLA loadable with direct compatible I/O.

The versioned release notes additionally record DLA transpose-merging limitations, lack of exclusive-padding average pooling, and a DLA broadcasting accuracy issue involving `kDLA_LINEAR`. These are reasons to validate the actual built engine numerically and not infer equivalence from ONNX legality alone.

Recommended consolidated vendor entries:

```bibtex
@misc{nvidia_dla_working,
  author       = {{NVIDIA Corporation}},
  title        = {Working with {DLA}},
  howpublished = {{TensorRT} 10.x Archived Documentation},
  url          = {https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-with-dla.html},
  note         = {Accessed: 2026-09-08}
}

@misc{nvidia_trt103,
  author       = {{NVIDIA Corporation}},
  title        = {{TensorRT} 10.3.0 Release Notes},
  howpublished = {{TensorRT} Documentation},
  year         = {2024},
  url          = {https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.3.0.html},
  note         = {Accessed: 2026-09-08}
}
```

No publication year is invented for the consolidated mutable archive. Do not change an existing access date merely for appearance; the dates above reflect this actual audit.

## Two focused additions

[ActNAS](https://openaccess.thecvf.com/content/CVPR2025W/MAI/html/Sah_ActNAS__Generating_Efficient_YOLO_Models_using_Activation_NAS_CVPRW_2025_paper.html) is particularly relevant because it studies activation selection in YOLO on CPU, NPU, and GPU and accounts for compiler optimizations. Attribute the existing hardware-aware ReLU/SiLU tradeoff; distinguish the present work through its particular YOLO11 DLA operator choices, graph boundary, and measured compilation behavior.

```bibtex
@inproceedings{sah2025actnas,
  author    = {Sudhakar Sah and Ravish Kumar and Darshan C. Ganji and Ehsan Saboori},
  title     = {{ActNAS}: Generating Efficient {YOLO} Models using Activation {NAS}},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages     = {1845--1853},
  year      = {2025}
}
```

[YOLOBench](https://openaccess.thecvf.com/content/ICCV2023W/RCV/html/Lazarevich_YOLOBench_Benchmarking_Efficient_Object_Detectors_on_Embedded_Systems_ICCVW_2023_paper.html) compares YOLO variants under a controlled training environment across four embedded hardware categories. It supports insisting on matched training and hardware-local accuracy/latency tradeoffs, rather than comparing unrelated published FPS values.

```bibtex
@inproceedings{lazarevich2023yolobench,
  author    = {Ivan Lazarevich and Matteo Grimaldi and Ravish Kumar and Saptarshi Mitra and Shahrukh Khan and Sudhakar Sah},
  title     = {{YOLOBench}: Benchmarking Efficient Object Detectors on Embedded Systems},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops},
  pages     = {1169--1178},
  year      = {2023}
}
```

Both are **workshop papers**; preserve that venue status. No need to add all available hardware-aware NAS papers merely to lengthen the bibliography.

## Further comparison to check before making a priority claim

A recent [author preprint by Vaishnav Raju](https://arxiv.org/abs/2608.11770), dated 12 August 2026, reports zero-GPU-fallback deployment of classification backbones on Jetson DLA alongside a GPU detector. It is adjacent rather than a DLA-resident YOLO11 detector. It reinforces avoiding claims such as “first DLA-compatible vision pipeline.” This audit checked its abstract; read its methods and validate publication status before giving it a larger role. No “first” claim is needed for the current paper.

The strongest short related-work structure is: edge-aware detection (YOLO-ReT, YOLOBench, ActNAS); target-hardware design (one or two of MnasNet/ProxylessNAS); actual YOLO11 implementation and distributional regression/EA attribution; then NVIDIA's documented restrictions. COCO supports evaluation. Retain broader runtime or robotics references only when the manuscript uses them for a concrete technical point.

## Figure and reference revision

The revised manuscript now cites **26 references**, adding nine to the prior 17. All added entries are cited in the related-work argument; none were added with `nocite` merely to expand the bibliography.

| Added reference | Primary source checked | Role in this manuscript |
|---|---|---|
| YOLOv3 (2018 technical report) | [Author preprint](https://arxiv.org/abs/1804.02767) | Multi-scale YOLO prediction history |
| YOLOv4 (2020 technical report) | [Author preprint](https://arxiv.org/abs/2004.10934) | Backbone/neck and training-design context |
| YOLOX (2021 technical report) | [Author preprint](https://arxiv.org/abs/2107.08430) | Decoupled anchor-free head and assignment |
| YOLOv6 (2022 technical report) | [Author preprint](https://arxiv.org/abs/2209.02976) | Deployment, re-parameterization, and quantization |
| YOLOv9 (ECCV 2024) | [ECVA proceedings](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/4462_ECCV_2024_paper.php) | PGI and efficient layer aggregation |
| CSPNet (CVPR Workshops 2020) | [CVF proceedings](https://openaccess.thecvf.com/content_CVPRW_2020/html/w28/Wang_CSPNet_A_New_Backbone_That_Can_Enhance_Learning_Capability_of_CVPRW_2020_paper.html) | Split/aggregate feature paths; workshop author order and CVF pagination retained |
| FPN (CVPR 2017) | [CVF proceedings](https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html) | Top-down feature pyramid and lateral connections |
| PANet (CVPR 2018) | [CVF proceedings](https://openaccess.thecvf.com/content_cvpr_2018/html/Liu_Path_Aggregation_Network_CVPR_2018_paper.html) | Bottom-up feature aggregation |
| RepVGG (CVPR 2021) | [CVF proceedings](https://openaccess.thecvf.com/content/CVPR 2021/html/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.html) | Restored from the other local paper as a deployment-design comparison; not claimed as part of C2DLA |

The YOLOv6 framework figure (Fig. 2, PDF page 3) and PANet's feature-map/path diagrams were inspected for visual inspiration. The new schematics are original TikZ drawings based on this repository's YOLO11-DLA YAML and module implementation, not reproductions of those figures. Figure 1 uses labeled backbone/neck/head/CPU panels and multi-scale feature plates; Figure 2 separates the C2DLA wrapper from the DLABlock operations. No example detections or empirical imagery were synthesized.

The reference draft `../YOLO-DLA_ICRA/` was re-read. RepVGG was its relevant missing reference; the other shared papers were already in the current bibliography. Three overlapping vendor documentation citations were not reintroduced solely to increase the count.

## Applied ICRA reframing: four verified additions (2026-09-11)

The manuscript retains all 26 existing entries and adds exactly the four works
requested by the authors. All 30 entries are cited, citation keys resolve, and
there are no duplicate keys or titles. No additional vendor documentation,
deployment examples, blogs, forums, or technical issue references were added.

| Key | Verified publication | Verification source and supported use |
|---|---|---|
| `hernandez2024optimizing` | Nicolás Hernández, Francisco Almeida, and Vicente Blanco. “Optimizing convolutional neural networks for IoT devices: performance and energy efficiency of quantization techniques.” *The Journal of Supercomputing*, vol. 80, pp. 12686–12705, 2024. DOI: `10.1007/s11227-024-05929-w`. | [Springer publication](https://link.springer.com/article/10.1007/s11227-024-05929-w) confirms all fields. Section 4.2 discusses incompatible layers requiring GPU execution, increased latency, and energy per inference. The manuscript distinguishes these metrics from performance per watt; it does not generalize the result to every DLA-compatible network. |
| `archet2023energy` | Agathe Archet, Nicolas Ventroux, Nicolas Gac, and François Orieux. “Energy-Efficient Use of an Embedded Heterogeneous SoC for the Inference of CNNs.” *2023 26th Euromicro Conference on Digital System Design (DSD)*, pp. 30–38, 2023. DOI: `10.1109/DSD60849.2023.00015`. | [IEEE Xplore](https://ieeexplore.ieee.org/document/10456862/) supplies the abstract and conference identity; [IEEE-deposited Crossref metadata](https://api.crossref.org/works/10.1109/DSD60849.2023.00015) confirms authors, pages, and DOI. The conference year is 2023; the later Xplore addition date is not the publication year. Used for CNN design and inference choices across Orin CPU/GPU/DLA and their latency/energy tradeoffs. |
| `tayal2024multiinstance` | Mumuksh Tayal and Yogesh Simmhan. “Evaluating Multi-Instance DNN Inferencing on Multiple Accelerators of an Edge Device.” *2024 IEEE 31st International Conference on High Performance Computing, Data and Analytics Workshop (HiPCW)*, pp. 181–182, 2024. DOI: `10.1109/HIPCW63042.2024.00068`. | [IEEE publication](https://ieeexplore.ieee.org/document/10898439/) and [IEEE-deposited Crossref metadata](https://api.crossref.org/works/10.1109/HIPCW63042.2024.00068) identify the original two-page work. The [author's publication page](https://mumukshtayal.github.io/#publications) links that IEEE work and describes investigation of concurrent CUDA/Tensor Core/DLA resource congestion. The manuscript uses this bounded description, not numerical results from the later expanded arXiv paper. |
| `chakraborty2025profiling` | Abhinaba Chakraborty, Wouter Tavernier, Akis Kourtis, Mario Pickavet, Andreas Oikonomakis, and Didier Colle. “Profiling Concurrent Vision Inference Workloads on NVIDIA Jetson.” *2025 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)*, pp. 359–361, 2025. DOI: `10.1109/ISPASS64960.2025.00043`. | [IEEE publication](https://ieeexplore.ieee.org/document/11096359/) and [IEEE-deposited Crossref metadata](https://api.crossref.org/works/10.1109/ISPASS64960.2025.00043) confirm bibliographic fields. The [three-page conference manuscript hosted by IMEC](https://imec-publications.be/bitstreams/99035855-96ef-47c7-832e-3fc6890a7a47/download) was read, rather than substituting the later expanded preprint. Used for concurrency-related contention, CPU–GPU scheduling, and the limits of aggregate utilization metrics. |

These concurrent-inference studies motivate explicit placement; they do not
validate concurrent-task performance, GPU inactivity, or isolation for YOLO11-DLA.
The four new DOI fields are retained in `references.bib`; the existing IEEEtran
style determines their printed representation.
