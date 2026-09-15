#!/usr/bin/env python3
"""Summarize one completed stage without pooling independent launches into fake replicates."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import configure_imports
configure_imports()

import argparse
import csv
import json
from pathlib import Path
import re
import statistics

from common import new_directory, sha256, write_json
from telemetry import add_power_arguments, energy_parts, energy_signature, integrate_power, subtract_idle


def aggregate_runs(rows, metric):
    groups = {}
    for row in rows:
        if metric in row and row[metric] is not None:
            groups.setdefault(row['configuration'], []).append(row[metric])
    return {key: {'independent_launches': len(values), 'values': values,
                  'mean': statistics.mean(values), 'sample_stdev': statistics.stdev(values) if len(values)>1 else None,
                  'min': min(values), 'max': max(values)} for key,values in groups.items()}


def accuracy(source, jobs):
    results = {}
    signature = None
    image_hashes = None
    for job in jobs:
        directory = source/job['name']
        manifest = json.loads((directory/'manifest.json').read_text())
        if manifest['status'] != 'complete':
            raise ValueError(f'Incomplete evaluation {directory}')
        args = manifest['arguments']
        this = (manifest['annotation_sha256'], manifest['image_ids'], args['conf'],args['iou'],args['max_det'])
        hashes = [(r['image_id'],r['image_sha256']) for r in
                  (json.loads(line) for line in (directory/'frames.jsonl').read_text().splitlines())]
        if signature is not None and (signature != this or image_hashes != hashes):
            raise ValueError('Accuracy runs have different inputs or evaluation thresholds')
        signature, image_hashes = this, hashes
        results[job['name']] = {'full_coco_val2017_candidate': manifest['full_coco_val2017_candidate'],
                               'artifact_sha256':manifest['artifact_sha256'],
                               **{k:100*v for k,v in manifest['metrics_fraction'].items() if k.startswith('AP')}}
    pairs = [('adapted-pt','stock-pt'), ('adapted-gpu','adapted-pt'), ('adapted-strict','adapted-gpu'),
             ('stock-gpu','stock-pt'), ('stock-fallback','stock-gpu'),
             ('adapted-strict-sparse','adapted-strict'), ('adapted-gpu-sparse','adapted-gpu')]
    comparisons = {a+' minus '+b: results[a]['AP']-results[b]['AP'] for a,b in pairs if a in results and b in results}
    baseline = results.get('stock-pt',{}).get('AP',0)
    return {'AP_points':results, 'paired_delta_AP_points':comparisons,
            'adapted_pt_retained_AP_percent':100*results['adapted-pt']['AP']/baseline if baseline>0 and 'adapted-pt' in results else None,
            'publication_gate': 'All full_coco flags must be true; verify official val2017 identity and inspect engine AP delta. No acceptance threshold is silently inferred.'}


def summarize_energy(rows, summaries):
    signature = None
    definition = None
    components = []
    for row in rows:
        summary = summaries[f"repeat{row['repeat']:02d}-{row['configuration']}"]
        energy = summary['energy']
        current = energy_signature(energy)
        if signature is not None and current != signature:
            raise ValueError('Cannot compare different power definitions (rails or scope)')
        signature = current
        definition = {'rails': list(current[0]), 'power_scope': current[1],
                      'aggregation': energy.get('aggregation', 'single'),
                      'power_profile': energy.get('power_profile')}
        row.update(power_scope=current[1], power_rails='+'.join(current[0]))
        parts = energy_parts(energy)
        idle_energy = None
        if row['configuration'] != 'idle':
            idle = summaries[f"repeat{row['repeat']:02d}-idle"]
            if idle['status'] != 'complete' or not idle['arguments']['idle']:
                raise ValueError('Require completed matched idle run')
            idle_energy = idle['energy']
            row.update(subtract_idle(energy, idle_energy, row['completed']))
        for name, part in parts.items():
            component = {'configuration': row['configuration'], 'repeat': row['repeat'], 'rail': name,
                         'average_w': part['average_w'], 'joules': part['joules'],
                         'interval_s': energy['interval_s'], 'completed': row['completed'],
                         'joules_per_frame': part['joules']/row['completed'] if row['completed'] else None}
            if idle_energy is not None:
                component.update(subtract_idle(part, energy_parts(idle_energy)[name], row['completed']))
            components.append(component)
    return {'power_definition': definition, 'energy_by_rail': components}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    add_power_arguments(p)
    args = p.parse_args()
    source = args.stage.resolve()
    manifest = json.loads((source/'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('Stage incomplete; inspect its manifest and failure logs')
    for name, digest in manifest.get('evidence_sha256', {}).items():
        if sha256(source/name) != digest:
            raise ValueError(f'Archived evidence changed: {name}')
    stage = manifest['arguments']['stage']
    recompute_energy = bool(args.rail or args.power_profile)
    if recompute_energy and stage != 'energy':
        p.error('--rail/--power-profile recomputes telemetry only for an energy stage')
    out = new_directory(args.output)
    report = {'stage':stage, 'source_manifest_sha256':sha256(source/'manifest.json')}
    if recompute_energy:
        report['reintegrated_telemetry_sha256'] = {}
    if manifest.get('source_kind') == 'archived_fp16_matrix':
        report['source_kind'] = manifest['source_kind']
        report['historical_provenance_limits'] = manifest['historical_provenance_limits']
    rows = []
    if stage == 'accuracy':
        report.update(accuracy(source,manifest['jobs']))
        protocol = manifest.get('evaluation_protocol')
        if protocol:
            for job in manifest['jobs']:
                evaluation = json.loads((source/job['name']/'manifest.json').read_text())
                if (evaluation['annotation_sha256'] != protocol['annotation_sha256'] or
                        evaluation['image_ids'] != protocol['image_ids']):
                    raise ValueError('Evaluation is not the complete registered held-out set')
            report['evaluation_protocol'] = protocol
            report['publication_gate'] = ('Complete registered held-out COCO-val2017, disjoint from calibration; '
                                          'not full-val2017 AP. Inspect quantization accuracy deltas before claims.')
        rows = [{'configuration':name,**record} for name,record in report['AP_points'].items()]
    elif stage in ('pipeline','energy','micro'):
        sequence_hash = None
        summaries = {}
        for job in manifest['jobs']:
            if job.get('profile_only'):
                continue
            name = job['name']
            row = {'configuration':job['configuration'], 'repeat':job['repeat']}
            if stage == 'micro':
                times = json.loads((source/f'{name}.times.json').read_text())
                if not times:
                    raise ValueError('Empty trtexec timing trace')
                text = (source/f'{name}.log').read_text()
                qps = re.search(r'Throughput:\s*([\d.eE+-]+)\s*qps',text)
                if not qps:
                    raise ValueError('Missing trtexec throughput summary')
                row.update(samples=len(times), achieved_fps=float(qps.group(1)))
                for key in ('latencyMs','computeMs','h2dMs','d2hMs'):
                    values = sorted(t[key] for t in times)
                    row[key+'_median'] = statistics.median(values)
                    # Match observed trtexec percentile convention, not numpy interpolation.
                    row[key+'_p95'] = values[min(len(values)-1,int(len(values)*0.95))]
                    row[key+'_p99'] = values[min(len(values)-1,int(len(values)*0.99))]
            else:
                s = json.loads((source/name/'summary.json').read_text())
                summaries[name] = s
                if s['status'] != 'complete':
                    raise ValueError(f'Incomplete run {name}')
                if recompute_energy:
                    telemetry = source/name/'telemetry.jsonl'
                    samples = [json.loads(line) for line in telemetry.read_text().splitlines()]
                    s['energy'] = integrate_power(samples, s['start_s'], s['end_s'], rail=args.rail,
                                                  power_profile=args.power_profile, completed=s['completed'])
                    report['reintegrated_telemetry_sha256'][name] = sha256(telemetry)
                if not s['arguments']['idle']:
                    sequence = json.loads((source/name/'sequence.json').read_text())
                    identity = [(e['image_id'],e['sha256']) for e in sequence]
                    if sequence_hash is not None and identity != sequence_hash:
                        raise ValueError('Timing runs used different image sequences')
                    sequence_hash = identity
                row.update(achieved_fps=s['achieved_fps'], completed=s['completed'], dropped=s['dropped'],
                           missed_or_dropped=s['missed_or_dropped'], interval_s=s['interval_s'])
                for key, value in s['latency_ms'].items():
                    if value:
                        row.update({key+'_'+q:value[q] for q in ('median','p95','p99')})
                if 'energy' in s:
                    row.update(average_w=s['energy']['average_w'],joules=s['energy']['joules'],
                               joules_per_frame=s['energy'].get('joules_per_frame'))
            rows.append(row)
        if stage == 'energy':
            report.update(summarize_energy(rows, summaries))
            report['energy_caveats'] = ['Verify hardware mode/clocks/rail scope match idle.',
                'agx-orin is total module power, not wall-plug or complete carrier-system power.',
                'Host receipt timestamps include uncalibrated telemetry delay.',
                'Compare measured delivered rates, drops and deadlines before claiming an energy gain.']
        report['runs'] = rows
        keys = sorted({key for row in rows for key,value in row.items()
                       if key not in ('configuration','repeat') and isinstance(value, (int,float))})
        report['between_launches'] = {key:aggregate_runs(rows,key) for key in keys}
        report['replication_warning'] = 'Per-frame quantiles are summarized per launch. Frames are not independent experimental replicates.'
    elif stage == 'build':
        rows = [{'configuration':j['name'], 'returncode':j['returncode'], 'outcome':j.get('outcome','built'),
                 'model':j['model'], 'one_dla_no_gpu_or_reformat':j.get('inspection',{}).get('one_dla_no_gpu_or_reformat')}
                for j in manifest['jobs']]
        report['builds'] = rows
    else:
        report['parity'] = json.loads((source/'comparison/report.json').read_text())
    if rows:
        keys = sorted({key for row in rows for key in row})
        with (out/'results.csv').open('w',newline='') as handle:
            writer = csv.DictWriter(handle,fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    if report.get('energy_by_rail'):
        components = report['energy_by_rail']
        with (out/'energy_rails.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in components for key in row}))
            writer.writeheader()
            writer.writerows(components)
    write_json(out/'report.json',report)
    print(f"Saved {out/'report.json'}")


if __name__ == '__main__':
    main()
