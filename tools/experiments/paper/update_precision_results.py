#!/usr/bin/env python3
"""Generate manuscript FP16/INT8 tables and deployment figures from audited results."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import ARTIFACTS, PAPER, WORKSPACE
from common import sha256, write_json
import csv
import matplotlib
matplotlib.use('Agg')
from paper.deployment_figures import (
    stage_breakdown, deployment_tradeoffs, engine_composition, inspector_composition,
)

SOURCES = {}
def rows(path):
    SOURCES[str(path.relative_to(WORKSPACE))] = sha256(path)
    with path.open() as stream:
        return {r['configuration']:r for r in csv.DictReader(stream)}

def main():
    SOURCES.clear()
    fp = WORKSPACE/'logs/icra2027_final/run_01'
    iq = WORKSPACE/'logs/icra2027_int8/run_01'
    accuracy = {'FP16':rows(iq/'fp16-heldout-summary/results.csv'),
                'INT8':rows(iq/'accuracy-summary/results.csv')}
    pipeline = {'FP16':rows(fp/'summary-pipeline/results.csv'),
                'INT8':rows(iq/'pipeline-summary/results.csv')}
    energy = {'FP16':rows(fp/'summary-energy-10hz-module/results.csv'),
              'INT8':rows(iq/'energy-summary/results.csv')}
    micro = {'INT8':rows(iq/'micro-summary/results.csv')}
    ids = {'stock-gpu':'yolo11n-gpu-fp16','adapted-gpu':'yolo11n-dla-gpu-fp16',
           'stock-fallback':'yolo11n-dla-gpu-fallback-fp16','adapted-strict':'yolo11n-dla-strict-dla-fp16'}
    path = ARTIFACTS/'engine_summary.csv'
    SOURCES[str(path.relative_to(WORKSPACE))] = sha256(path)
    with path.open() as stream:
        source = {r['id']:r for r in csv.DictReader(stream)}
    micro['FP16'] = {name:{'latencyMs_median':source[key]['inclusive_median_ms'],
                          'latencyMs_p95':source[key]['inclusive_p95_ms'],
                          'achieved_fps':source[key]['throughput_qps_reported']} for name,key in ids.items()}
    order = ('stock-gpu','stock-fallback','adapted-gpu','adapted-strict')
    names = {'stock-gpu':('YOLO11-n', 'GPU'),'stock-fallback':('YOLO11-n', 'DLA+GPU'),
             'adapted-gpu':('YOLO11-DLA-n', 'GPU'),'adapted-strict':('YOLO11-DLA-n', 'DLA')}
    tables = PAPER/'tables'
    tables.mkdir(exist_ok=True)
    specs = {
        'precision_accuracy':(accuracy, [('AP',3),('AP50',3),('AP75',3)],
            'Accuracy on 4,500 held-out COCO val2017 images, disjoint from INT8 calibration. AP values are percentage points; INT8 allows mixed precision; stock and adapted training budgets differ.',
            'tab:accuracy',r'Model & Device & Mode & AP & AP$_{50}$ & AP$_{75}$'),
        'precision_energy':(energy,[('joules_per_frame',3),('idle_subtracted_joules_per_frame',3)],
            'Module energy at 10\\,Hz, 600 frames per test. Total and idle-subtracted energy: J/frame.',
            'tab:energy',r'Model & Device & Mode & \shortstack{Total\\J/frame} & \shortstack{Idle-subtracted\\J/frame}')}
    for filename,(data,columns,caption,label,header) in specs.items():
        lines=[r'\begin{table}[t]',r'\caption{'+caption+'}',r'\label{'+label+r'}\centering\footnotesize',
               r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lll'+'r'*len(columns)+'@{}}',r'\toprule',header+r' \\',r'\midrule']
        for i,mode in enumerate(('FP16','INT8')):
            if i:lines.append(r'\midrule')
            for name in order:
                vals=[f'{float(data[mode][name][key]):.{decimals}f}' for key,decimals in columns]
                lines.append(' & '.join([*names[name],mode,*vals])+r' \\')
        lines += [r'\bottomrule',r'\end{tabular*}',r'\end{table}']
        (tables/(filename+'.tex')).write_text('\n'.join(lines)+'\n')
    lines = [r'\begin{table*}[t]',
             r'\caption{Deployment latency at the engine and application boundaries, batch one, $640\times640$. All values are milliseconds. Engine measurements include transfers (30\,s); application measurements include preprocessing, native I/O, inference, decode, and NMS on the same 100 images (60\,s).}',
             r'\label{tab:latency}\centering\footnotesize',
             r'\setlength{\tabcolsep}{4pt}',
             r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lllrrrr@{}}',
             r'\toprule',
             r'& & & \multicolumn{2}{c}{Engine + transfer} & \multicolumn{2}{c}{Application} \\',
             r'\cmidrule(lr){4-5}\cmidrule(l){6-7}',
             r'Model & Device & Mode & Median & P95 & Median & P95 \\',
             r'\midrule']
    for i, mode in enumerate(('FP16', 'INT8')):
        if i:
            lines.append(r'\midrule')
        for name in order:
            values = [micro[mode][name][key] for key in ('latencyMs_median', 'latencyMs_p95')]
            values += [pipeline[mode][name][key] for key in ('total_ms_median', 'total_ms_p95')]
            lines.append(' & '.join([*names[name], mode, *[f'{float(v):.3f}' for v in values]]) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular*}', r'\end{table*}']
    (tables/'precision_latency.tex').write_text('\n'.join(lines)+'\n')
    engine_composition(inspector_composition(WORKSPACE), PAPER/'figures')
    stage_breakdown(pipeline, PAPER/'figures')
    deployment_tradeoffs(accuracy, pipeline, energy, PAPER/'figures')
    write_json(ARTIFACTS/'precision_results.json',dict(sources_sha256=SOURCES,
        accuracy=accuracy,micro=micro,pipeline=pipeline,energy=energy,
        scope='Imported experimental results; no new engine build or inference. AP cohorts are paired 4500-image held-out.'))

if __name__=='__main__':main()
