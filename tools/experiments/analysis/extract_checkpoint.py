#!/usr/bin/env python3
"""Extract supplied trusted checkpoint metadata/history; does not train or validate."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import WORKSPACE, ARTIFACTS

import csv, hashlib, json, sys
import torch
sys.path.insert(0,str(WORKSPACE/'tools/ultralytics'))
source=WORKSPACE/'logs/yolo11n-dla-500ep.pt'
checkpoint=torch.load(source,map_location='cpu',weights_only=False)
history=checkpoint['train_results']
model=checkpoint.get('ema') or checkpoint['model']
best=max(range(len(history['epoch'])),key=lambda i:history['metrics/mAP50-95(B)'][i])
report={'source':str(source.relative_to(WORKSPACE)), 'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'evidence_label':'Checkpoint-recorded training validation; not an independent re-evaluation or deployed-engine AP',
        'parameter_count':sum(p.numel() for p in model.parameters()),
        'checkpoint_epoch':checkpoint.get('epoch'), 'history_epochs':len(history['epoch']),
        'best_history_epoch':history['epoch'][best], 'best_history_ap5095':history['metrics/mAP50-95(B)'][best],
        **{k:checkpoint.get(k) for k in ['date','version','git','train_args','train_metrics','train_results']}}
out=ARTIFACTS
(out/'checkpoint_training.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
with (out/'checkpoint_training.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(history)
    w.writerows(zip(*(history[k] for k in history)))
print('Extracted',report['history_epochs'],'epochs; final AP',checkpoint['train_metrics']['metrics/mAP50-95(B)'])
