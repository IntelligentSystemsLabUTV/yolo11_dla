#!/usr/bin/env python3
"""Generate manuscript FP16/INT8 tables and a vector plot from audited results."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import PAPER, WORKSPACE
from common import sha256, write_json
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

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
    path = PAPER/'analysis/engine_summary.csv'
    SOURCES[str(path.relative_to(WORKSPACE))] = sha256(path)
    with path.open() as stream:
        source = {r['id']:r for r in csv.DictReader(stream)}
    micro['FP16'] = {name:{'latencyMs_median':source[key]['inclusive_median_ms'],
                          'latencyMs_p95':source[key]['inclusive_p95_ms'],
                          'achieved_fps':source[key]['throughput_qps_reported']} for name,key in ids.items()}
    order = ('stock-gpu','stock-fallback','adapted-gpu','adapted-strict')
    names = {'stock-gpu':'11n / GPU','stock-fallback':'11n / DLA+GPU',
             'adapted-gpu':'11-DLA / GPU','adapted-strict':'11-DLA / DLA'}
    tables = PAPER/'tables'
    tables.mkdir(exist_ok=True)
    specs = {
        'precision_accuracy':(accuracy, [('AP',3),('AP50',3),('AP75',3)],
            'Accuracy on 4,500 held-out COCO val2017 images, disjoint from INT8 calibration. AP values are percentage points; INT8 allows mixed precision.',
            'tab:accuracy',r'Model / device & Mode & AP & AP$_{50}$ & AP$_{75}$'),
        'precision_engine':(micro,[('latencyMs_median',3),('latencyMs_p95',3),('achieved_fps',1)],
            'Engine-plus-transfer latency, batch one, $640\\times640$. Median and P95 are milliseconds; rate is queries/s.',
            'tab:engine',r'Model / device & Mode & Median & P95 & Rate'),
        'precision_pipeline':(pipeline,[('total_ms_median',3),('total_ms_p95',3),('total_ms_p99',3),('achieved_fps',2)],
            'Application pipeline on the same 100 images, 60\\,s per test. Latency: ms; rate: frames/s. Includes preprocessing, native I/O, inference, decode and NMS.',
            'tab:e2e',r'Model / device & Mode & Median & P95 & P99 & Rate'),
        'precision_energy':(energy,[('average_w',3),('joules_per_frame',3),('idle_subtracted_joules_per_frame',3)],
            'Module energy at 10\\,Hz, 600 frames per test. Power: W; total and idle-subtracted energy: J/frame.',
            'tab:energy',r'Model / device & Mode & Power & Total & Incr.')}
    for filename,(data,columns,caption,label,header) in specs.items():
        if filename == 'precision_pipeline':
            # Pair precisions horizontally in the densest table. Keep the other
            # tables at one full column without scaling their text or numbers.
            lines = [r'\begin{table*}[t]', r'\caption{'+caption+'}',
                     r'\label{'+label+r'}\centering\footnotesize',
                     r'\setlength{\tabcolsep}{4pt}',
                     r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrrrr@{}}',
                     r'\toprule',
                     r'& \multicolumn{4}{c}{FP16} & \multicolumn{4}{c}{INT8} \\',
                     r'\cmidrule(lr){2-5}\cmidrule(l){6-9}',
                     r'Model / device & Median & P95 & P99 & Rate & Median & P95 & P99 & Rate \\',
                     r'\midrule']
            for name in order:
                vals = [f'{float(data[mode][name][key]):.{decimals}f}'
                        for mode in ('FP16', 'INT8') for key,decimals in columns]
                lines.append(' & '.join([names[name], *vals])+r' \\')
            lines += [r'\bottomrule', r'\end{tabular*}', r'\end{table*}']
            (tables/(filename+'.tex')).write_text('\n'.join(lines)+'\n')
            continue
        lines=[r'\begin{table}[t]',r'\caption{'+caption+'}',r'\label{'+label+r'}\centering\footnotesize',
               r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}ll'+'r'*len(columns)+'@{}}',r'\toprule',header+r' \\',r'\midrule']
        for i,mode in enumerate(('FP16','INT8')):
            if i:lines.append(r'\midrule')
            for name in order:
                vals=[f'{float(data[mode][name][key]):.{decimals}f}' for key,decimals in columns]
                lines.append(' & '.join([names[name],mode,*vals])+r' \\')
        lines += [r'\bottomrule',r'\end{tabular*}',r'\end{table}']
        (tables/(filename+'.tex')).write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'pdf.fonttype':42,'ps.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,1,figsize=(3.46,2.85))
    fig.subplots_adjust(left=.31,right=.98,bottom=.13,top=.85,hspace=.68)
    for ax,group,title,xmax in zip(axes,[('stock-gpu','adapted-gpu'),('stock-fallback','adapted-strict')],
                                  ['(a) GPU execution','(b) DLA deployment'],[4.5,40]):
        for i,(mode,color) in enumerate([('FP16','#52788A'),('INT8','#C97B34')]):
            vals=np.array([float(micro[mode][n]['latencyMs_median']) for n in group])
            p95=np.array([float(micro[mode][n]['latencyMs_p95']) for n in group])
            y=np.arange(2)+(i-.5)*.30
            ax.barh(y,vals,height=.27,color=color,label=mode)
            ax.errorbar(vals,y,xerr=[np.zeros(2),p95-vals],fmt='none',color='#222222',capsize=1.5,linewidth=.6)
            for yy,v in zip(y,vals):ax.text(v+xmax*.015,yy,f'{v:.2f}',va='center',fontsize=6.5)
        ax.set_yticks([0,1],['YOLO11n'+('\nfallback' if 'fallback' in group[0] else ''),
                              'YOLO11-DLA-n'+('\nstrict' if 'strict' in group[1] else '')])
        ax.set_ylim(1.5,-.5);ax.set_xlim(0,xmax)
        ax.set_xlabel('Engine + transfer latency (ms)',labelpad=2)
        ax.set_title(title,loc='left',fontsize=8,pad=4)
        ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False)
    for ext in ('pdf','svg'):
        fig.savefig(PAPER/'figures'/('precision_latency.'+ext),bbox_inches='tight',pad_inches=.025,
                    metadata={'Creator':'update_precision_results.py'} if ext=='pdf' else None)
    plt.close(fig)
    write_json(PAPER/'analysis/precision_results.json',dict(sources_sha256=SOURCES,
        accuracy=accuracy,micro=micro,pipeline=pipeline,energy=energy,
        scope='Imported experimental results; no new engine build or inference. AP cohorts are paired 4500-image held-out.'))

if __name__=='__main__':main()
