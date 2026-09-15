#!/usr/bin/env python3
"""Audit supplied YOLO11 FP16 logs and regenerate the manuscript figures.

The original logs are read only. All summary data are parsed from source files,
not copied from the manuscript. This is an audit, not a benchmark execution.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import WORKSPACE, PAPER


import argparse
import collections
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONFIGS = [
    ("yolo11n-gpu-fp16", "YOLO11n", "GPU", "yolo11n.onnx"),
    ("yolo11n-dla-gpu-fp16", "YOLO11-DLA-n", "GPU", "yolo11n-dla.onnx"),
    ("yolo11n-dla-gpu-fallback-fp16", "YOLO11n", "DLA+GPU fallback", "yolo11n.onnx"),
    ("yolo11n-dla-strict-dla-fp16", "YOLO11-DLA-n", "strict DLA", "yolo11n-dla.onnx"),
]
NONCOMPUTE = {"NoOp", "Constant", "shape_call"}
METRICS = {"Latency": "latencyMs", "GPU Compute Time": "computeMs", "H2D Latency": "h2dMs", "D2H Latency": "d2hMs"}
COLORS = {"device": "#7E8D9F", "total": "#007F75", "DLA": "#007F75", "Reformat": "#E2A33D", "GPU compute": "#725598"}


def source_record(path: Path, logs: Path) -> dict:
    return {"path": str(path.relative_to(logs)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def percentile_trtexec(values, percent):
    """Observed trtexec percentile convention: sorted[floor(n*p/100)].

    The reported log is authoritative. This matches every supplied p95/p99;
    numpy's default interpolated quantile differs for the 878-sample run.
    """
    a = np.sort(np.asarray(values, dtype=float))
    return float(a[min(int(len(a) * percent / 100), len(a) - 1)])


def stats(values):
    a = np.asarray(values, dtype=float)
    return {"n": len(a), "min": float(a.min()), "mean": float(a.mean()), "median": float(np.median(a)), "p95": percentile_trtexec(a, 95), "p99": percentile_trtexec(a, 99), "max": float(a.max()), "p95_numpy_linear": float(np.percentile(a, 95)), "p99_numpy_linear": float(np.percentile(a, 99))}


def required(pattern, text):
    m = re.search(pattern, text)
    if m is None:
        raise ValueError(f"Missing expected source field: {pattern}")
    return m


def log_metric(log, label):
    line = required(r"\[I\] " + re.escape(label) + r": min = ([^\n]+)", log).group(1)
    labels = {"min": r"^([\d.eE+-]+) ms", "max": r"max = ([\d.eE+-]+)", "mean": r"mean = ([\d.eE+-]+)", "median": r"median = ([\d.eE+-]+)", "p95": r"percentile\(95%\) = ([\d.eE+-]+)", "p99": r"percentile\(99%\) = ([\d.eE+-]+)"}
    return {key: float(required(pattern, line).group(1)) for key, pattern in labels.items()}


def category(layer_type):
    if layer_type == "DLA":
        return "DLA"
    if layer_type == "Reformat":
        return "Reformat"
    if layer_type in NONCOMPUTE:
        return "metadata/no-op"
    return "GPU compute"


def analyze_config(logs, stem, model, execution, expected_onnx):
    paths = {kind: logs / f"{stem}.{kind}" for kind in ["build.log", "benchmark.log", "times.json", "layers.json"]}
    build = paths["build.log"].read_text()
    log = paths["benchmark.log"].read_text()
    times = json.loads(paths["times.json"].read_text())
    inspection = json.loads(paths["layers.json"].read_text())
    command = build.splitlines()[0].split(" # ", 1)[1]
    onnx = Path(required(r"--onnx=(\S+)", command).group(1)).name
    assert onnx == expected_onnx, (stem, onnx, expected_onnx)
    assert "&&&& PASSED" in build and "&&&& PASSED" in log
    count, duration = required(r"Timing trace has (\d+) queries over ([\d.]+) s", log).groups()
    assert len(times) == int(count), "The supplied JSON must be the timed trace, not warmup + timed samples"
    all_stats = {key: stats([t[key] for t in times]) for key in METRICS.values()}
    reported = {key: log_metric(log, label) for label, key in METRICS.items()}
    for key in METRICS.values():
        for stat in ["median", "p95", "p99"]:
            assert abs(all_stats[key][stat] - reported[key][stat]) < .00011, (stem, key, stat)
    # JSON fields are independently rounded; tolerate <0.2 us in the sum.
    additive_error = max(abs(t["latencyMs"] - t["h2dMs"] - t["computeMs"] - t["d2hMs"]) for t in times)
    assert additive_error < .0002
    layer_types = collections.Counter(x["LayerType"] for x in inspection["Layers"])
    counts = collections.Counter(category(x["LayerType"]) for x in inspection["Layers"])
    result = {
        "id": stem, "model": model, "execution": execution, "source_onnx_basename": onnx,
        "build_command": command, "benchmark_command": log.splitlines()[0].split(" # ", 1)[1],
        "build_passed": True, "gpu_fallback_permitted": "--allowGPUFallback" in command,
        "engine_bytes": (logs / f"{stem}.engine").stat().st_size,
        "timed_queries": int(count), "timed_wall_seconds": float(duration),
        "throughput_qps_reported": float(required(r"\[I\] Throughput: ([\d.]+) qps", log).group(1)),
        "warmup_queries": int(required(r"Warmup completed (\d+) queries", log).group(1)),
        "warmup_ms": int(required(r"Warmup completed \d+ queries over (\d+) ms", log).group(1)),
        "trace_stats_ms": all_stats, "reported_stats_ms": reported,
        "trace_latency_component_sum_max_error_ms": additive_error,
        "first_start_enqueue_ms": times[0]["startEnqMs"], "first_start_compute_ms": times[0]["startComputeMs"],
        "layer_type_counts": dict(layer_types), "layer_category_counts": dict(counts),
        "bindings": inspection["Bindings"],
        "strict_single_DLA_loadable": len(inspection["Layers"]) == 1 and inspection["Layers"][0]["LayerType"] == "DLA" and "--allowGPUFallback" not in command,
        "sources": [source_record(p, logs) for p in paths.values()],
    }
    result["sources"].append(source_record(logs / f"{stem}.engine", logs))
    profile_path = logs / f"{stem}.profile.json"
    if profile_path.exists():
        profile = json.loads(profile_path.read_text())
        type_by_name = {x["Name"]: x["LayerType"] for x in inspection["Layers"]}
        aggregates = {k: {"entries": 0, "average_ms": 0., "percentage": 0.} for k in ["DLA", "Reformat", "GPU compute", "metadata/no-op"]}
        details = []
        for entry in profile:
            if "name" not in entry:
                continue
            assert entry["name"] in type_by_name, "All profiler names must map to inspector layers"
            kind = category(type_by_name[entry["name"]])
            aggregates[kind]["entries"] += 1
            aggregates[kind]["average_ms"] += entry["averageMs"]
            aggregates[kind]["percentage"] += entry["percentage"]
            details.append({**entry, "category": kind, "inspector_type": type_by_name[entry["name"]]})
        result["profile"] = {"separate_run": True, "profile_count": profile[0]["count"], "aggregates": aggregates, "entries": details, "average_ms_sum": sum(x["average_ms"] for x in aggregates.values())}
        result["sources"].append(source_record(profile_path, logs))
        result["sources"].append(source_record(logs / f"{stem}.profile.log", logs))
    return result


def summary_e2e(logs):
    """Keep secondary summaries explicitly separate from raw benchmark evidence."""
    path = logs / "FP16_RESULTS.md"
    text = path.read_text().split("## Repeated-image E2E comparison", 1)[1].split("### Full timing distributions", 1)[0]
    rows = []
    for line in text.splitlines():
        cells = [x.strip().replace("**", "") for x in line.strip("|").split("|")]
        if not cells[0].startswith("YOLO11"):
            continue
        rows.append({"configuration": cells[0], **dict(zip(["preprocess", "pack_h2d", "engine", "d2h_unpack", "postprocess", "total", "total_p95", "total_p99", "sequential_rate_im_s"], [float(x.replace(" im/s", "")) for x in cells[1:]]))})
    assert len(rows) == 4
    raw = sorted(p.name for p in logs.glob("*e2e.json"))
    return {"evidence_level": "secondary consolidated summary; corresponding YOLO11 raw E2E JSON files absent", "source": source_record(path, logs), "raw_E2E_JSON_files_present_in_input_directory": raw, "stage_statistic": "independently computed medians; do not add to infer total median", "rows": rows}


def setup_plotting():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False, "axes.edgecolor": "#9DA4AC", "axes.linewidth": .6, "xtick.color": "#45505D", "ytick.color": "#25313C", "savefig.facecolor": "white"})


def clean_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E4E8EB", linewidth=.5)
    ax.tick_params(axis="y", length=0)


def save_figure(fig, directory, name):
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight", pad_inches=.025, metadata={"Creator": "analysis/analyze_fp16.py", "CreationDate": None, "ModDate": None})
    fig.savefig(directory / f"{name}.svg", bbox_inches="tight", pad_inches=.025)
    plt.close(fig)


def latency_figure(runs, directory):
    fig, axes = plt.subplots(2, 1, figsize=(3.46, 3.08))
    fig.subplots_adjust(left=.315, right=.96, top=.845, bottom=.12, hspace=.88)
    labels = ["Device interval", "With H2D + D2H"]
    handles = []
    for panel, (ax, group, xmax, ticks) in enumerate(zip(axes, [runs[:2], runs[2:]], [4.6, 42], [[0, 1, 2, 3, 4], [0, 10, 20, 30, 40]])):
        for j, metric in enumerate(["computeMs", "latencyMs"]):
            vals = np.array([r["reported_stats_ms"][metric]["median"] for r in group])
            p95 = np.array([r["reported_stats_ms"][metric]["p95"] for r in group])
            ys = np.arange(2) + (j - .5) * .29
            bars = ax.barh(ys, vals, height=.25, color=COLORS["device" if j == 0 else "total"], label=labels[j])
            ax.errorbar(vals, ys, xerr=[np.zeros(2), p95 - vals], fmt="none", color="#202D36", linewidth=.7, capsize=1.5)
            for y, v in zip(ys, vals):
                ax.text(v + xmax * .02, y, f"{v:.2f}", va="center", fontsize=6.8)
            if panel == 0:
                handles.append(bars)
        ax.set_yticks([0, 1], ["YOLO11n" + ("\nfallback" if panel else ""), "YOLO11-DLA-n" + ("\nstrict" if panel else "")])
        ax.set_ylim(1.48, -.48)
        ax.set_xlim(0, xmax)
        ax.set_xticks(ticks)
        ax.set_xlabel("Latency (ms)", labelpad=2)
        ax.set_title("(a) GPU execution" if panel == 0 else "(b) DLA deployment", loc="left", pad=6)
        clean_axis(ax)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.54, 1), ncol=1, frameon=False, handlelength=1.1, labelspacing=.3)
    save_figure(fig, directory, "latency_comparison")


def profile_figure(runs, directory):
    fallback, strict = runs[2:]
    fig, axes = plt.subplots(2, 1, figsize=(3.46, 3.0), gridspec_kw={"height_ratios": [1, 1.1]})
    fig.subplots_adjust(left=.265, right=.97, top=.84, bottom=.15, hspace=.95)
    ax = axes[0]
    left = np.zeros(2)
    for kind in ["DLA", "Reformat", "GPU compute"]:
        vals = np.array([r["profile"]["aggregates"][kind]["average_ms"] for r in [fallback, strict]])
        ax.barh([0, 1], vals, left=left, color=COLORS[kind], height=.48, label=kind)
        left += vals
    for y, r in enumerate([fallback, strict]):
        ax.text(r["profile"]["average_ms_sum"] + .65, y, f'{r["profile"]["average_ms_sum"]:.2f}', va="center", fontsize=7)
        ax.text(r["profile"]["aggregates"]["DLA"]["average_ms"] / 2, y, f'{r["profile"]["aggregates"]["DLA"]["percentage"]:.1f}% DLA', color="white", ha="center", va="center", fontsize=7)
    ax.set_yticks([0, 1], ["YOLO11n\nfallback", "YOLO11-DLA-n\nstrict"])
    ax.set_ylim(1.55, -.55)
    ax.set_xlim(0, 40)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_xlabel("Summed profiler means (ms)", labelpad=2)
    ax.set_title("(a) Separately profiled engine", loc="left", pad=7)
    clean_axis(ax)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.54, 1), frameon=False, ncol=3, columnspacing=.7, handlelength=1.05, handletextpad=.4)
    ax = axes[1]
    kinds = ["DLA", "Reformat", "GPU compute"]
    x = np.arange(3)
    for offset, r, color, hatch in [(-.16, fallback, "#7E8D9F", ""), (.16, strict, "#007F75", "")]:
        vals = [r["layer_category_counts"].get(k, 0) for k in kinds]
        ax.bar(x + offset, vals, width=.28, color=color, hatch=hatch)
        for xx, v in zip(x + offset, vals):
            ax.text(xx, v + .6, str(v), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, ["DLA\nloadables", "Reformat\nnodes", "GPU compute\nnodes"])
    ax.set_ylim(0, 27)
    ax.set_yticks([0, 10, 20])
    ax.set_ylabel("Count", labelpad=2)
    ax.set_title("(b) Optimized engine inspection", loc="left", pad=6)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E4E8EB", linewidth=.5)
    ax.text(.99, .91, "gray: fallback   teal: strict", transform=ax.transAxes, fontsize=6.3, ha="right")
    save_figure(fig, directory, "fallback_profile")


def training_figure(path, directory):
    """Visualize the stored history only; no new validation or seed band."""
    checkpoint = json.loads(path.read_text())
    history = checkpoint["train_results"]
    epochs = np.asarray(history["epoch"])
    assert len(epochs) == checkpoint["history_epochs"]
    fig, ax = plt.subplots(figsize=(3.46, 2.13))
    fig.subplots_adjust(left=.15, right=.985, top=.96, bottom=.205)
    for key, label, color in [("metrics/mAP50(B)", r"AP$_{50}$", "#7E8D9F"), ("metrics/mAP50-95(B)", r"AP$_{50:95}$", "#007F75")]:
        values = np.asarray(history[key]) * 100
        ax.plot(epochs, values, color=color, linewidth=1.1, label=label)
        ax.plot(epochs[-1], values[-1], "o", color=color, markersize=2.7)
        ax.text(epochs[-1] + 8, values[-1], f"{values[-1]:.2f}", color=color, va="center", fontsize=7)
    ax.set_xlim(0, 585)
    ax.set_ylim(0, 60)
    ax.set_xticks([0, 100, 200, 300, 400, 500])
    ax.set_yticks([0, 20, 40, 60])
    ax.set_xlabel("Training epoch", labelpad=2)
    ax.set_ylabel("Stored validation AP (%)", labelpad=3)
    ax.set_axisbelow(True)
    ax.grid(color="#E4E8EB", linewidth=.5)
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    save_figure(fig, directory, "training_history")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, default=WORKSPACE / "logs/fp16_matrix")
    parser.add_argument("--output", type=Path, default=PAPER / "analysis")
    parser.add_argument("--figures", type=Path, default=PAPER / "figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    runs = [analyze_config(args.logs, *config) for config in CONFIGS]
    fail = args.logs / "yolo11n-strict-dla-fp16.build.log"
    fail_text = fail.read_text()
    assert "&&&& FAILED" in fail_text and "MatMul is unsupported on DLA" in fail_text
    baseline, retrofit, fallback, strict = runs
    reference_log = (args.logs / "yolo11n-gpu-fp16.benchmark.log").read_text()
    hardware = {
        "device": required(r"Selected Device: ([^\n]+)", reference_log).group(1),
        "compute_capability": required(r"Compute Capability: ([^\n]+)", reference_log).group(1),
        "SMs": int(required(r"SMs: (\d+)", reference_log).group(1)),
        "global_memory_MiB": int(required(r"Device Global Memory: (\d+) MiB", reference_log).group(1)),
        "memory_bus_bits": int(required(r"Memory Bus Width: (\d+) bits", reference_log).group(1)),
        "TensorRT": required(r"TensorRT version: ([\d.]+)", reference_log).group(1),
        "DLA_core": int(required(r"--useDLACore=(\d+)", strict["benchmark_command"]).group(1)),
        "reported_application_compute_clock_GHz": float(required(r"Application Compute Clock Rate: ([\d.]+) GHz", reference_log).group(1)),
        "reported_application_memory_clock_GHz": float(required(r"Application Memory Clock Rate: ([\d.]+) GHz", reference_log).group(1)),
        "clock_caveat": "The logs explicitly state application clock rates do not reflect actual running rates.",
    }
    audit = {
        "scope": "YOLO11n baseline and YOLO11-DLA-n only; custom YOLO-DLA excluded",
        "benchmark_date": datetime.strptime(required(r"\[(\d{2}/\d{2}/\d{4})-", reference_log).group(1), "%m/%d/%Y").date().isoformat(),
        "hardware_recorded": hardware,
        "hardware_missing": ["exact module SKU and carrier", "JetPack/L4T", "CUDA/cuDNN versions", "nvpmodel power mode", "jetson_clocks policy", "measured CPU/GPU/DLA/EMC clocks", "thermal and ambient state", "power/energy measurements"],
        "measurement_boundary": "trtexec Latency = H2D + device interval + D2H; not synchronized image-to-detection wall latency; device interval is called GPU Compute Time even when DLA executes it",
        "trace_policy": "JSON length equals reported post-warmup trace length. No second warmup cut; enqueue may precede 2000ms for a query whose compute starts after warmup.",
        "quantile_policy": "Report trtexec log percentiles; reconstruct with sorted[floor(n*p/100)]. JSON also contains NumPy interpolated percentiles for comparison.",
        "replication_limit": "One benchmark process/run per configuration; query percentiles describe within-run variation, not uncertainty across independent runs.",
        "throughput_limit": "Reported saturated/pipelined queries per second is not 1000/median latency and is not sequential image-to-detection rate.",
        "runs": runs,
        "baseline_strict_failure": {"source": source_record(fail, args.logs), "failure": "First reported unsupported operator /model.10/m/m.0/attn/MatMul; fallback disabled", "build_command": fail_text.splitlines()[0].split(" # ", 1)[1]},
        "derived_comparisons": {
            "strict_vs_fallback_transfer_inclusive_median_reduction_percent": 100 * (1 - strict["reported_stats_ms"]["latencyMs"]["median"] / fallback["reported_stats_ms"]["latencyMs"]["median"]),
            "strict_vs_fallback_throughput_increase_percent": 100 * (strict["throughput_qps_reported"] / fallback["throughput_qps_reported"] - 1),
            "retrofit_vs_baseline_GPU_transfer_inclusive_median_reduction_percent": 100 * (1 - retrofit["reported_stats_ms"]["latencyMs"]["median"] / baseline["reported_stats_ms"]["latencyMs"]["median"]),
        },
        "e2e_summary_only": summary_e2e(args.logs),
    }
    (args.output / "fp16_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    fields = ["id", "model", "execution", "source_onnx_basename", "timed_queries", "warmup_queries", "engine_bytes", "throughput_qps_reported", "device_median_ms", "inclusive_median_ms", "inclusive_p95_ms", "inclusive_p99_ms", "dla_loadables", "gpu_compute_nodes", "reformat_nodes"]
    with (args.output / "engine_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for r in runs:
            writer.writerow({**{k: r[k] for k in fields if k in r}, "device_median_ms": r["reported_stats_ms"]["computeMs"]["median"], "inclusive_median_ms": r["reported_stats_ms"]["latencyMs"]["median"], "inclusive_p95_ms": r["reported_stats_ms"]["latencyMs"]["p95"], "inclusive_p99_ms": r["reported_stats_ms"]["latencyMs"]["p99"], "dla_loadables": r["layer_category_counts"].get("DLA", 0), "gpu_compute_nodes": r["layer_category_counts"].get("GPU compute", 0), "reformat_nodes": r["layer_category_counts"].get("Reformat", 0)})
    with (args.output / "layer_profile.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, ["configuration", "name", "inspector_type", "category", "timeMs", "averageMs", "medianMs", "percentage"])
        writer.writeheader()
        for r in runs:
            for entry in r.get("profile", {}).get("entries", []):
                writer.writerow({"configuration": r["id"], **entry})
    setup_plotting()
    latency_figure(runs, args.figures)
    profile_figure(runs, args.figures)
    training_path = args.output / "checkpoint_training.json"
    if training_path.exists():
        training_figure(training_path, args.figures)
    print(json.dumps({"runs_audited": len(runs), "source_trace_samples": sum(r["timed_queries"] for r in runs), "derived_comparisons": audit["derived_comparisons"], "output": str(args.output), "figures": str(args.figures)}, indent=2))


if __name__ == "__main__":
    main()
