"""Timestamp tegrastats samples and integrate an explicit rail or module power profile."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import configure_imports
configure_imports()

import json
from pathlib import Path
import re
import subprocess
import threading
import time

# Current/average in milliwatts, with optional explicit mW suffix (JetPack variants).
POWER = re.compile(r'\b([A-Za-z][A-Za-z0-9_]*)\s+(\d+(?:\.\d+)?)(?:mW)?/(\d+(?:\.\d+)?)(?:mW)?\b')

# NVIDIA AGX Orin INA3221 rail topology: VIN_SYS_5V0 already includes VDDQ.
# https://forums.developer.nvidia.com/t/jetson-agx-orin-ina3221-power-monitor-layout/223111
POWER_PROFILES = {'agx-orin': ('VDD_GPU_SOC', 'VDD_CPU_CV', 'VIN_SYS_5V0')}


def add_power_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--rail', help='One explicitly selected physical rail; reports this rail only')
    group.add_argument('--power-profile', choices=POWER_PROFILES,
                       help='agx-orin: total module power = VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0')


def power_selection(rail=None, power_profile=None):
    if bool(rail) == bool(power_profile):
        raise ValueError('Select exactly one --rail or --power-profile')
    rails = list(POWER_PROFILES[power_profile]) if power_profile else [rail]
    return {'rail': '+'.join(rails), 'rails': rails, 'power_profile': power_profile,
            'power_scope': 'module' if power_profile else 'selected_rail',
            'aggregation': 'sum' if len(rails) > 1 else 'single',
            'power_value': 'instantaneous; first value in each tegrastats current/average pair'}


def energy_signature(energy):
    """Legacy single-rail reports remain single-rail, never silently upgraded to module power."""
    return (tuple(energy.get('rails', [energy['rail']])),
            energy.get('power_scope', 'selected_rail'))


def energy_parts(energy):
    parts = energy.get('per_rail')
    if parts is None and len(energy_signature(energy)[0]) == 1:
        return {energy['rail']: energy}
    if parts is None or set(parts) != set(energy_signature(energy)[0]):
        raise ValueError('Missing component energy for selected power rails')
    return parts


def subtract_idle(active, idle, completed):
    if energy_signature(active) != energy_signature(idle):
        raise ValueError('Active and idle power definitions differ (rails or scope)')
    joules = active['joules'] - idle['average_w'] * active['interval_s']
    return {'idle_subtracted_joules': joules,
            'idle_subtracted_joules_per_frame': joules/completed if completed else None}


def parse_rails(line):
    return {name: float(current)/1000 for name, current, _ in POWER.findall(line)
            if name not in ('RAM', 'SWAP', 'IRAM')}


class Telemetry:
    def __init__(self, command='tegrastats', interval_ms=100):
        self.rows = []
        self.closed = False
        self.process = subprocess.Popen([command, '--interval', str(interval_ms)],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for line in self.process.stdout:
            self.rows.append({'monotonic_s': time.perf_counter(), 'wall_time_ns': time.time_ns(),
                              'rails_w': parse_rails(line), 'raw': line.rstrip()})

    def wait_for_rails(self, rails, timeout=5):
        """Fail before the measured interval if the tool exits or required sensors are absent."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                break
            if any(all(rail in row['rails_w'] for rail in rails) for row in list(self.rows)):
                return
            time.sleep(0.05)
        observed = sorted({rail for row in list(self.rows) for rail in row['rails_w']})
        raise RuntimeError(f'tegrastats did not provide required rails {rails}; observed {observed}; '
                           f'process status {self.process.poll()}; inspect telemetry.jsonl')

    def close(self, output):
        if self.closed:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.thread.join(timeout=5)
        self.process.stdout.close()
        Path(output).write_text(''.join(json.dumps(row)+'\n' for row in self.rows))
        self.closed = True


def integrate(rows, rail, start, end, max_gap_s=0.5):
    import numpy as np
    samples = [(r['monotonic_s'], r['rails_w'][rail]) for r in rows if rail in r['rails_w']]
    if end <= start or len(samples) < 2:
        raise ValueError('Invalid integration interval or insufficient power samples')
    times, power = map(np.asarray, zip(*samples))
    if not np.isfinite([start, end]).all() or not np.isfinite(times).all() or not np.isfinite(power).all():
        raise ValueError('Non-finite power values or timestamps')
    if np.any(np.diff(times) <= 0) or times[0] > start or times[-1] < end:
        raise ValueError('Power samples must be ordered and bracket the whole measured interval')
    left = max(0, int(np.searchsorted(times, start, side='right'))-1)
    right = min(len(times)-1, int(np.searchsorted(times, end)))
    observed_gap = float(np.diff(times[left:right+1]).max())
    if observed_gap > max_gap_s:
        raise ValueError(f'Power sample gap {observed_gap:.3f}s exceeds {max_gap_s}s')
    inner = times[(times > start) & (times < end)]
    grid = np.concatenate(([start], inner, [end]))
    watts = np.interp(grid, times, power)
    joules = float(np.sum((watts[:-1]+watts[1:])*0.5*np.diff(grid)))
    return {'rail': rail, 'joules': joules, 'average_w': joules/(end-start),
            'interval_s': end-start, 'max_sample_gap_s': observed_gap,
            'timestamp_boundary': 'host receipt of tegrastats stdout; sensor/pipe delay not calibrated'}


def integrate_power(rows, start, end, *, rail=None, power_profile=None, max_gap_s=0.5, completed=0):
    selection = power_selection(rail, power_profile)
    required = selection['rails']
    complete = []
    for row in rows:
        values = row['rails_w']
        if not values:  # Preserve diagnostic text in raw telemetry, not in the power curve.
            continue
        missing = [name for name in required if name not in values]
        if missing:
            if start <= row['monotonic_s'] <= end:
                raise ValueError(f'Missing required power rails {missing} inside the measured interval')
            continue
        complete.append(row)
    # Every component uses identical sample times and identical interpolated boundaries.
    parts = {name: integrate(complete, name, start, end, max_gap_s) for name in required}
    for part in parts.values():
        part['joules_per_frame'] = part['joules']/completed if completed else None
    first = next(iter(parts.values()))
    joules = sum(part['joules'] for part in parts.values())
    return {**selection, 'schema_version': 2, 'joules': joules, 'average_w': joules/(end-start),
            'joules_per_frame': joules/completed if completed else None,
            'interval_s': end-start, 'start_s': start, 'end_s': end,
            'per_rail': parts, 'max_sample_gap_s': first['max_sample_gap_s'],
            'timestamp_boundary': first['timestamp_boundary']}
