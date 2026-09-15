#!/usr/bin/env python3
"""Record the FP16/INT8 results from the stage summaries; never runs a measurement."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import ARTIFACTS, WORKSPACE
from common import sha256, write_json
import csv

SOURCES = {}


def rows(path):
    """Index one summary CSV by configuration, recording its hash."""
    SOURCES[str(path.relative_to(WORKSPACE))] = sha256(path)
    with path.open() as stream:
        return {r['configuration']: r for r in csv.DictReader(stream)}


def main():
    SOURCES.clear()
    fp = WORKSPACE / 'logs/icra2027_final/run_01'
    iq = WORKSPACE / 'logs/icra2027_int8/run_01'
    accuracy = {'FP16': rows(iq / 'fp16-heldout-summary/results.csv'),
                'INT8': rows(iq / 'accuracy-summary/results.csv')}
    pipeline = {'FP16': rows(fp / 'summary-pipeline/results.csv'),
                'INT8': rows(iq / 'pipeline-summary/results.csv')}
    energy = {'FP16': rows(fp / 'summary-energy-10hz-module/results.csv'),
              'INT8': rows(iq / 'energy-summary/results.csv')}
    micro = {'INT8': rows(iq / 'micro-summary/results.csv')}

    # FP16 timing comes from the audited historical traces, keyed by engine identifier.
    ids = {'stock-gpu': 'yolo11n-gpu-fp16', 'adapted-gpu': 'yolo11n-dla-gpu-fp16',
           'stock-fallback': 'yolo11n-dla-gpu-fallback-fp16',
           'adapted-strict': 'yolo11n-dla-strict-dla-fp16'}
    path = ARTIFACTS / 'engine_summary.csv'
    SOURCES[str(path.relative_to(WORKSPACE))] = sha256(path)
    with path.open() as stream:
        source = {r['id']: r for r in csv.DictReader(stream)}
    micro['FP16'] = {name: {'latencyMs_median': source[key]['inclusive_median_ms'],
                            'latencyMs_p95': source[key]['inclusive_p95_ms'],
                            'achieved_fps': source[key]['throughput_qps_reported']}
                     for name, key in ids.items()}

    write_json(ARTIFACTS / 'precision_results.json', dict(
        sources_sha256=SOURCES, accuracy=accuracy, micro=micro,
        pipeline=pipeline, energy=energy,
        scope='Imported experimental results; no new engine build or inference. '
              'AP cohorts are paired 4500-image held-out.'))
    print(f'Recorded {len(SOURCES)} sources in {ARTIFACTS / "precision_results.json"}')


if __name__ == '__main__':
    main()
