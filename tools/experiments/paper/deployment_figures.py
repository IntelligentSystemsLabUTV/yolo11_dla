"""Vector figures from paired FP16/INT8 experimental summaries."""
from collections import Counter
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ORDER = ('stock-gpu', 'stock-fallback', 'adapted-gpu', 'adapted-strict')
# Pastel palette from the manuscript's architecture figures.
COLORS = {
    'stock-gpu': ('#BBDCFE', '#74A2D7'),
    'stock-fallback': ('#E9DCFD', '#BEA9DD'),
    'adapted-gpu': ('#FFDFBC', '#CA9255'),
    'adapted-strict': ('#CBEBC7', '#7EB07B'),
}
LABELS = {
    'stock-gpu': 'YOLO11-n / GPU',
    'stock-fallback': 'YOLO11-n / DLA+GPU',
    'adapted-gpu': 'YOLO11-DLA-n / GPU',
    'adapted-strict': 'YOLO11-DLA-n / strict DLA',
}
STAGES = (
    ('Preprocess', 'preprocess_ms_median'),
    ('Input boundary', 'input_pack_h2d_ms_median'),
    ('Inference', 'engine_ms_median'),
    ('Output boundary', 'output_unpack_ms_median'),
    ('Postprocess', 'postprocess_ms_median'),
)


def setup():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8,
        'axes.titlesize': 9, 'axes.labelsize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 8, 'legend.fontsize': 7,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#9DA4AC', 'axes.linewidth': .6,
        'savefig.facecolor': 'white',
    })


def save(fig, directory, name):
    for ext in ('pdf', 'svg'):
        metadata = {'Creator': 'paper/deployment_figures.py'}
        if ext == 'pdf':
            metadata.update(CreationDate=None, ModDate=None)
        fig.savefig(directory / f'{name}.{ext}', bbox_inches='tight',
                    pad_inches=.03, metadata=metadata)
    plt.close(fig)


def inspector_composition(workspace):
    """Read optimized nodes only; never use source assignments or profiler rows."""
    sources = {
        'FP16': {
            'stock-fallback': 'fp16_matrix/yolo11n-dla-gpu-fallback-fp16.layers.json',
            'adapted-strict': 'fp16_matrix/yolo11n-dla-strict-dla-fp16.layers.json',
        },
        'INT8': {
            'stock-fallback': 'int8_ready/stock-fallback.layers.json',
            'adapted-strict': 'int8_ready/adapted-strict.layers.json',
        },
    }
    categories = {'DLA': 'DLA loadables', 'Reformat': 'Reformat',
                  'NoOp': 'Metadata / no-op', 'Constant': 'Metadata / no-op'}
    for kind in ('PointWiseV2', 'gemm', 'CaskGemmMatrixMultiply', 'Shuffle',
                 'CaskSoftMaxV2', 'CaskConvolution'):
        categories[kind] = 'GPU compute'
    result = {}
    for mode, deployments in sources.items():
        result[mode] = {}
        for name, relative in deployments.items():
            data = json.loads((workspace / 'logs' / relative).read_text())
            layers = data['Layers']
            counts = Counter(categories[layer['LayerType']] for layer in layers)
            result[mode][name] = {**counts, 'total': len(layers)}
    return result


def engine_composition(composition, directory):
    setup()
    fig, axes = plt.subplots(2, 1, figsize=(3.46, 2.7), sharex=True)
    fig.subplots_adjust(left=.33, right=.97, bottom=.16, top=.78, hspace=.75)
    styles = [
        ('DLA loadables', '#CBEBC7', '#527A50', ''),
        ('Reformat', '#E9DCFD', '#8C76AD', '///'),
        ('GPU compute', '#BBDCFE', '#527AA5', '\\\\'),
        ('Metadata / no-op', '#F3F5F7', '#7D848B', '...'),
    ]
    for ax, mode in zip(axes, ('FP16', 'INT8')):
        names = ('stock-fallback', 'adapted-strict')
        left = np.zeros(2)
        for label, fill, edge, hatch in styles:
            values = [composition[mode][name].get(label, 0) for name in names]
            ax.barh([0, 1], values, left=left, height=.48, color=fill,
                    edgecolor=edge, linewidth=.6, hatch=hatch, label=label)
            left += values
        for y, name in enumerate(names):
            total = composition[mode][name]['total']
            ax.text(total + 1, y, str(total), va='center', fontsize=7)
        ax.set_yticks([0, 1], ['YOLO11-n /\nDLA+GPU', 'YOLO11-DLA-n /\nstrict DLA'])
        ax.tick_params(axis='y', length=0, labelsize=7)
        ax.set_ylim(1.6, -.6)
        ax.set_xlim(0, 54)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.set_title(f'({"a" if mode == "FP16" else "b"}) {mode}', loc='left', fontsize=8, pad=5)
        ax.spines['left'].set_visible(False)
        ax.grid(axis='x', color='#E4E8EB', linewidth=.5)
        ax.set_axisbelow(True)
    axes[1].set_xlabel('Optimized-engine nodes')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.52, 1.01), ncol=2, frameon=False,
               handlelength=1.5, columnspacing=1.0, fontsize=6.8)
    save(fig, directory, 'engine_composition')


def stage_breakdown(pipeline, directory):
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.45), sharey=True)
    fig.subplots_adjust(left=.16, right=.985, top=.85, bottom=.11, wspace=.18)
    y = np.arange(len(STAGES)) * 1.4
    for ax, mode, xmax in zip(axes, ('FP16', 'INT8'), (41, 17)):
        for i, name in enumerate(ORDER):
            values = [float(pipeline[mode][name][key]) for _, key in STAGES]
            offset = (i - 1.5) * .26
            fill, edge = COLORS[name]
            ax.barh(y + offset, values, height=.19, color=fill,
                    edgecolor=edge, linewidth=.6, label=LABELS[name])
            for yy, value in zip(y + offset, values):
                ax.text(value + xmax * .012, yy, f'{value:.3f}',
                        va='center', fontsize=7)
        ax.set_yticks(y, [label for label, _ in STAGES])
        ax.set_ylim(y[-1] + .65, -.65)
        ax.set_xlim(0, xmax)
        ax.set_xticks([0, 10, 20, 30, 40] if mode == 'FP16' else [0, 4, 8, 12, 16])
        ax.set_title(f'({"a" if mode == "FP16" else "b"}) {mode}', loc='left', pad=7)
        ax.set_xlabel('Median stage time (ms)')
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.grid(axis='x', color='#E4E8EB', linewidth=.5)
        ax.set_axisbelow(True)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.53, 1.01), ncol=2, frameon=False,
               handlelength=1.6, columnspacing=2.0)
    save(fig, directory, 'application_stages')


def deployment_tradeoffs(accuracy, pipeline, energy, directory):
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.15), sharey=True)
    fig.subplots_adjust(left=.085, right=.985, bottom=.18, top=.72, wspace=.18)
    markers = {'stock-gpu': 'o', 'adapted-gpu': 'o',
               'stock-fallback': 's', 'adapted-strict': '^'}
    # Reuse the blue/green palette of the earlier figures.
    precision = {'FP16': COLORS['stock-gpu'], 'INT8': COLORS['adapted-strict']}
    for ax, data, key, xlabel, title in zip(
            axes, (pipeline, energy), ('total_ms_median', 'joules_per_frame'),
            ('Median application latency (ms)', 'Module energy (J/frame at 10 Hz)'),
            ('(a) Accuracy vs application latency', '(b) Accuracy vs module energy')):
        for mode in ('FP16', 'INT8'):
            fill, edge = precision[mode]
            for name in ORDER:
                x = float(data[mode][name][key])
                ap = float(accuracy[mode][name]['AP'])
                ax.scatter(x, ap, marker=markers[name], s=42, linewidths=1.1,
                           facecolors='white' if mode == 'FP16' else fill,
                           edgecolors=edge, zorder=3)
                # Shapes identify fallback/strict; only GPU points need model labels.
                if name.endswith('gpu'):
                    stock = name == 'stock-gpu'
                    # Keep full names away from the tightly grouped GPU points.
                    if mode == 'FP16':
                        label_y = 41.0 if stock else 38.2
                        label_x = 25 if key == 'total_ms_median' else .775
                    else:
                        label_y = 31.0 if stock else 32.5
                        label_x = (21 if stock else 35) if key == 'total_ms_median' else (.775 if stock else .82)
                    ax.annotate('YOLO11-n' if stock else 'YOLO11-DLA-n', (x, ap),
                                xytext=(label_x, label_y), textcoords='data',
                                ha='left', va='center', fontsize=6.5, color='#333333',
                                arrowprops={'arrowstyle': '-', 'color': '#888888',
                                            'lw': .45, 'shrinkA': 2, 'shrinkB': 4})
        ax.set_title(title, loc='left', pad=7, fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylim(18, 42)
        ax.set_yticks([20, 25, 30, 35, 40])
        ax.grid(color='#E4E8EB', linewidth=.5)
        ax.set_axisbelow(True)
    axes[0].set_xlim(13, 61)
    axes[0].set_xticks([20, 30, 40, 50, 60])
    axes[1].set_xlim(.748, .934)
    axes[1].set_xticks([.76, .80, .84, .88, .92])
    axes[0].set_ylabel(r'COCO AP$_{50:95}$')
    shape_handles = [Line2D([], [], marker=m, linestyle='', color='#555555',
                           markerfacecolor='white', markersize=5, label=label)
                     for m, label in [('o', 'GPU'), ('s', 'DLA+GPU'), ('^', 'strict DLA')]]
    mode_handles = [Line2D([], [], marker='o', linestyle='', markersize=5,
                          markeredgecolor=edge, markerfacecolor='white' if mode == 'FP16' else fill,
                          label=mode)
                    for mode, (fill, edge) in precision.items()]
    fig.legend(handles=shape_handles, title='Device', title_fontsize=8,
               loc='upper center', bbox_to_anchor=(.32, 1.01), ncol=3,
               frameon=False, columnspacing=1.3)
    fig.legend(handles=mode_handles, title='Precision', title_fontsize=8,
               loc='upper center', bbox_to_anchor=(.80, 1.01), ncol=2,
               frameon=False, columnspacing=1.3)
    save(fig, directory, 'deployment_tradeoffs')
