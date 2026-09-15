#!/usr/bin/env python3
"""One entry point for the paper scripts; forwards arguments without a shell."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
COMMANDS = {
    'reuse-existing': ('orchestration/reuse_existing.py',),
    'int8-split': ('int8/split_coco.py',),
    'int8-calibrate': ('int8/calibrate.py',),
    'int8-build': ('int8/build.py',),
    'int8-reuse': ('int8/reuse.py',),
    'int8-rescore-fp16': ('int8/rescore.py',),
    'prepare': ('export/prepare.py',),
    **{stage: ('orchestration/suite.py', stage) for stage in
       ('build', 'parity', 'accuracy', 'micro', 'pipeline', 'energy')},
    'evaluate': ('evaluation/evaluate_coco.py',),
    'check-model': ('evaluation/verify_model.py',),
    'benchmark': ('benchmarks/benchmark.py',),
    'e2e': ('benchmarks/yolo_e2e_tester.py',),
    'summarize': ('analysis/summarize.py',),
    'audit': ('analysis/analyze_fp16.py',),
    'extract-checkpoint': ('analysis/extract_checkpoint.py',),
    'check-paper': ('paper/check_paper.py',),
    'train': ('training/train_coco_dla.py',),
    'recover-commands': ('history/recover_commands.py',),
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print(__doc__)
        print('Usage: python tools/experiments/run.py COMMAND [arguments]')
        print('Commands: ' + ', '.join(COMMANDS))
        print('AGX Orin module energy: energy --power-profile agx-orin [other arguments]')
        print('Summaries: summarize --stage RUN --output NEW_DIR (totals, per-rail energy and matched idle)')
        return
    if sys.argv[1] not in COMMANDS:
        raise SystemExit(f'Unknown command: {sys.argv[1]} (use --help)')
    script, *prefix = COMMANDS[sys.argv[1]]
    os.execv(sys.executable, [sys.executable, str(ROOT/script), *prefix, *sys.argv[2:]])


if __name__ == '__main__':
    main()
