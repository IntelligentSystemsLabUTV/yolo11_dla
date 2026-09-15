"""Module power must use simultaneous, non-overlapping rails and matched idle scope."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from _paths import ROOT
from analysis.summarize import summarize_energy
from suite import plan
from telemetry import integrate_power, parse_rails, subtract_idle


def samples():
    # Different slopes test summation and interpolation at non-sample boundaries.
    return [{'monotonic_s': i/10, 'rails_w': {
        'VDD_GPU_SOC': 2+i/10, 'VDD_CPU_CV': 0.4+2*i/10,
        'VIN_SYS_5V0': 3.7+3*i/10, 'VDDQ_VDD2_1V8AO': 100}}
        for i in range(11)]


def test_module_sum_interpolates_all_rails_without_counting_ddr_twice():
    result = integrate_power(samples(), 0.1, 0.9, power_profile='agx-orin', completed=8)
    assert result['power_scope'] == 'module'
    assert result['average_w'] == pytest.approx(9.1)
    assert result['joules'] == pytest.approx(7.28)
    assert result['joules_per_frame'] == pytest.approx(0.91)
    assert len(result['per_rail']) == 3
    assert sum(p['joules'] for p in result['per_rail'].values()) == pytest.approx(result['joules'])
    assert result['per_rail']['VDD_CPU_CV']['average_w'] == pytest.approx(1.4)


def test_actual_tegrastats_format_uses_instantaneous_values():
    values = parse_rails('VDD_GPU_SOC 2398mW/9999mW VDD_CPU_CV 399mW/8888mW VIN_SYS_5V0 3705mW/3724mW')
    rows = [{'monotonic_s': i/10, 'rails_w': values} for i in range(11)]
    result = integrate_power(rows, 0.05, 0.95, power_profile='agx-orin')
    assert result['average_w'] == pytest.approx(6.502)
    assert result['joules_per_frame'] is None


@pytest.mark.parametrize('fault', ['missing', 'nan', 'gap', 'coverage'])
def test_invalid_module_samples_cannot_silently_reduce_total(fault):
    rows = samples()
    if fault == 'missing':
        del rows[5]['rails_w']['VDD_CPU_CV']
    elif fault == 'nan':
        rows[5]['rails_w']['VDD_CPU_CV'] = float('nan')
    elif fault == 'gap':
        rows = rows[:2] + rows[9:]
    else:
        rows = rows[3:]
    with pytest.raises(ValueError):
        integrate_power(rows, 0.1, 0.9, power_profile='agx-orin')


def test_single_rail_stays_single_and_idle_must_match_scope():
    single = integrate_power(samples(), 0.1, 0.9, rail='VDD_GPU_SOC')
    module = integrate_power(samples(), 0.1, 0.9, power_profile='agx-orin')
    assert single['power_scope'] == 'selected_rail'
    assert single['average_w'] == pytest.approx(2.5)
    with pytest.raises(ValueError, match='definitions differ'):
        subtract_idle(module, single, 8)
    higher_idle = {**module, 'average_w': 10}
    assert subtract_idle(module, higher_idle, 8)['idle_subtracted_joules_per_frame'] == pytest.approx(-0.09)


def test_energy_plan_applies_module_profile_to_idle_and_all_engines():
    args = argparse.Namespace(output=Path('/out'), artifacts=None, engines=Path('/engines'),
                              stage='energy', seed=7, repeats=1, duration=60, fps=10,
                              stabilize_seconds=30, cpu_threads=1, deadline_ms=100,
                              images=Path('/images'), annotations=Path('/annotations'), limit=100,
                              tegrastats='/tmp/tegrastats-paper', rail=None, power_profile='agx-orin')
    jobs = plan(args)
    assert len(jobs) == 5
    assert any(j['configuration'] == 'idle' for j in jobs)
    for job in jobs:
        command = job['command']
        assert command[command.index('--power-profile')+1] == 'agx-orin'
        assert '--rail' not in command


def energy_fixture():
    active = integrate_power(samples(), 0.1, 0.9, power_profile='agx-orin', completed=8)
    idle_samples = copy.deepcopy(samples())
    for sample in idle_samples:
        sample['rails_w'] = {k: v/2 for k,v in sample['rails_w'].items()}
    idle = integrate_power(idle_samples, 0.1, 0.9, power_profile='agx-orin')
    return active, idle, idle_samples


def test_summary_total_and_per_rail_idle_subtraction_agree():
    active, idle, _ = energy_fixture()
    rows = [{'configuration': 'adapted-gpu', 'repeat': 1, 'completed': 8},
            {'configuration': 'idle', 'repeat': 1, 'completed': 0}]
    summaries = {'repeat01-adapted-gpu': {'energy': active},
                 'repeat01-idle': {'status': 'complete', 'arguments': {'idle': True}, 'energy': idle}}
    result = summarize_energy(rows, summaries)
    assert rows[0]['idle_subtracted_joules_per_frame'] == pytest.approx(0.455)
    parts = [p for p in result['energy_by_rail'] if p['configuration'] == 'adapted-gpu']
    assert sum(p['idle_subtracted_joules_per_frame'] for p in parts) == pytest.approx(0.455)
    summaries['repeat01-idle']['energy']['power_scope'] = 'selected_rail'
    with pytest.raises(ValueError, match='definitions differ'):
        summarize_energy(rows, summaries)


def test_cli_summary_and_reintegration_preserve_original_results(tmp_path):
    active, idle, idle_samples = energy_fixture()
    source = tmp_path/'stage'
    source.mkdir()
    jobs = []
    for name, energy, readings in [('adapted-gpu', active, samples()), ('idle', idle, idle_samples)]:
        tag = 'repeat01-'+name
        folder = source/tag
        folder.mkdir()
        is_idle = name == 'idle'
        summary = {'status': 'complete', 'arguments': {'idle': is_idle}, 'energy': energy,
                   'start_s': 0.1, 'end_s': 0.9, 'interval_s': 0.8, 'completed': 0 if is_idle else 8,
                   'achieved_fps': 0 if is_idle else 10, 'dropped': 0, 'missed_or_dropped': 0,
                   'latency_ms': {}}
        (folder/'summary.json').write_text(json.dumps(summary))
        (folder/'telemetry.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in readings))
        (folder/'sequence.json').write_text('[{"image_id": 1, "sha256": "fixture"}]')
        jobs.append({'name': tag, 'configuration': name, 'repeat': 1})
    (source/'manifest.json').write_text(json.dumps({'status': 'complete', 'arguments': {'stage': 'energy'}, 'jobs': jobs}))
    output = tmp_path/'summary'
    command = [sys.executable, str(ROOT/'run.py'), 'summarize', '--stage', str(source), '--output']
    subprocess.run(command+[str(output)], check=True, capture_output=True)
    report = json.loads((output/'report.json').read_text())
    assert report['runs'][0]['joules_per_frame'] == pytest.approx(0.91)
    assert report['runs'][0]['idle_subtracted_joules_per_frame'] == pytest.approx(0.455)
    assert len((output/'energy_rails.csv').read_text().splitlines()) == 7
    # Simulate previously completed single-rail summaries with all raw rails retained.
    for name, readings in [('adapted-gpu', samples()), ('idle', idle_samples)]:
        path = source/('repeat01-'+name)/'summary.json'
        summary = json.loads(path.read_text())
        summary['energy'] = integrate_power(readings, 0.1, 0.9, rail='VDD_GPU_SOC', completed=summary['completed'])
        path.write_text(json.dumps(summary))
    before = {str(p): p.read_bytes() for p in source.rglob('*') if p.is_file()}
    recovered = tmp_path/'recovered'
    subprocess.run(command+[str(recovered), '--power-profile', 'agx-orin'], check=True, capture_output=True)
    report = json.loads((recovered/'report.json').read_text())
    assert report['runs'][0]['joules_per_frame'] == pytest.approx(0.91)
    assert report['power_definition']['power_scope'] == 'module'
    assert len(report['reintegrated_telemetry_sha256']) == 2
    assert before == {str(p): p.read_bytes() for p in source.rglob('*') if p.is_file()}
