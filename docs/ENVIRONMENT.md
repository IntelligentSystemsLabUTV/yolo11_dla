# Environment

Every number in the paper comes from one board. The values below are not a recommended configuration, they are what the manifests recorded at measurement time; each measurement directory carries its own `provenance` block with the same queries, so a run can always be checked against this page rather than trusted from it.

## Target platform

| | |
|---|---|
| Board | NVIDIA Jetson AGX Orin Developer Kit (`/proc/device-tree/model`) |
| Host name in traces | `orinagx1` |
| L4T | R36.4.4, kernel 5.15.148-tegra, aarch64 |
| CUDA | runtime 12.6, `nvcc` 12.6.68 |
| TensorRT | 10.3.0.30-1+cuda12.5 (`libnvinfer*`) |
| cuDNN | 9.3.0.75 |
| PyTorch | 2.5.0a0+872d972e41.nv24.8, torchvision 0.20.0 |
| Python | 3.10.12 |
| numpy / onnx / pycocotools | 1.23.5 / 1.21.0 / 2.0.11 |
| ROS 2 | Jazzy, present in the image but outside the measured boundary |
| Power mode | NVIDIA 50 W |
| `jetson_clocks` | disabled — CPU frequency varies during measurement |
| DLA | core 0 |

`nvpmodel -q` and `jetson_clocks --show` were **not available inside the measurement container**, so the manifests record failed queries for both. The 50 W mode and the disabled clocks are author-declared platform conditions, not values recovered from the logs. Telemetry shows CPU frequencies moving mostly between 729 and 1497 MHz, and junction temperature around 45.9–47.4 °C during the energy tests; that is not a complete throttling audit.

## Containers

Both containers are DUA environment images and are checked in under [`../docker/`](../docker/).

`container-jetson6` is the JetPack 6 image in which every Jetson measurement ran — builds, calibration, parity, accuracy, `trtexec` microbenchmarks, the application pipeline and the energy campaigns. It is privileged, uses the NVIDIA runtime and host networking, and mounts the workspace at `/home/neo/workspace`, which is why recorded absolute paths in the manifests begin with that prefix.

`container-x86-cudev` is the x86 CUDA development image used for training the checkpoint and for host-side analysis.

`tegrastats` is not present inside the Jetson container. For the energy campaigns the host binary was copied in once:

```bash
# on the Jetson host
docker cp -L /usr/bin/tegrastats devcontainer-yolo_dla-jetson6-1:/tmp/tegrastats-paper
```

and then passed explicitly with `--tegrastats /tmp/tegrastats-paper`. If the container is recreated, repeat the copy and check that it reports the three rails before collecting anything. The energy collector starts and stops the binary itself for each test, including idle, and never kills a `tegrastats` process it did not start.

## Power rails

The AGX Orin exposes `VDD_GPU_SOC`, `VDD_CPU_CV` and `VIN_SYS_5V0`. The `agx-orin` power profile integrates their sum as **module power**. `VDD_CPU_CV` already includes DLA and PVA, and `VIN_SYS_5V0` already includes the DDR rail, so DDR is not added a second time. This is not a wall-plug measurement and does not cover the carrier board, peripherals or the power supply. The rail topology is documented by NVIDIA in the [Platform Power and Performance guide](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html#software-based-power-consumption-modeling) and in the [forum clarification on the INA3221 layout](https://forums.developer.nvidia.com/t/jetson-agx-orin-ina3221-power-monitor-layout/223111).

## Analysis host requirements

Reading the evidence requires nothing. Regenerating derived artifacts requires:

| Task | Needs |
|---|---|
| `run.py audit`, `make -C tools/paper precision-results` | Python, NumPy, Matplotlib |
| `run.py extract-checkpoint` | PyTorch, and the checkpoint in `logs/` |
| `run.py check-model` | ONNX, ONNX Runtime, PyTorch |
| `make -C tools/paper check` | PyMuPDF |
| `make -C tools/paper` | Tectonic |
| `pytest tools/experiments/tests` | pytest, NumPy |
| Anything that touches an engine | The target board, its TensorRT, and the Ultralytics fork on `PYTHONPATH` |

Serialized TensorRT engines are tied to this GPU, TensorRT version and driver. An engine is the exact measured artifact of one build, not a portable model: on any other stack it must be rebuilt from the ONNX files with the commands in [`BUILD.md`](BUILD.md).

## Dataset

COCO val2017 images and `instances_val2017.json`. The evaluator reads the COCO JSON directly and does not convert labels to YOLO format. To install it on the target:

```bash
bash tools/experiments/datasets/download_coco_val2017.sh
export COCO_VAL="$PWD/logs/datasets/coco/images/val2017"
export COCO_ANN="$PWD/logs/datasets/coco/annotations/instances_val2017.json"
```

The script resumes interrupted downloads, extracts only the detection validation annotations, and verifies that the 5,000 images named by the JSON and the 80 categories are present. It does not decode the JPEGs or certify official hashes. About 1.07 GB is downloaded; allow 3 GB with the extraction. `logs/datasets/` is untracked.
