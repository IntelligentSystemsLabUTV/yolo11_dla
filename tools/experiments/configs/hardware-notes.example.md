# Hardware notes — fill in before the final measurements

Copy this file to `hardware-notes.md` and complete it. `prepare` copies it into the artifact directory, so it travels with the run it describes. Declare unrecovered fields explicitly rather than leaving them blank.

- Jetson module / SKU / RAM:
- Carrier board:
- JetPack / L4T:
- Container image or environment, and its digest:
- TensorRT / CUDA / cuDNN / PyTorch:
- `nvpmodel` mode, name and ID as verified on this board:
- Clock policy, and the commands actually applied:
- Fan / heatsink / ambient conditions:
- CPU affinity and concurrent processes:
- Power rail selected, its units, and its scope (whole board or module only):
- COCO: origin of train2017/val2017 and the annotations, and their path on the target:
- Stock checkpoint: origin, version, hash:
- Adapted checkpoint: origin, hash:
- Adapted training: original command, initialization, recovered logs:
- Fields that could not be recovered, stated explicitly:

The scripts query the platform state but never change `nvpmodel`, clocks, fan or privileges. If the automatic queries fail, run `nvpmodel -q --verbose` and `jetson_clocks --show` with the necessary privileges and archive their output alongside this file.
