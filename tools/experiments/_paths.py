"""Canonical repository paths, independent of the current working directory."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
TOOLS = WORKSPACE / 'tools'
ULTRALYTICS = TOOLS / 'ultralytics'
ARTIFACTS = WORKSPACE / 'analysis'
def configure_imports():
    for path in (ROOT, ROOT/'benchmarks', ROOT/'evaluation', ROOT/'orchestration', ULTRALYTICS):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

configure_imports()
