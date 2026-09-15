#!/usr/bin/env python3
"""Re-evaluate saved FP16 predictions on INT8 held-out IDs, without inference."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import configure_imports
configure_imports()

import argparse
import json
from common import new_directory, sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='Completed FP16 accuracy stage')
    p.add_argument('--engines', type=Path, required=True, help='int8-reuse output with held-out protocol')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    registered = json.loads((args.engines/'manifest.json').read_text())
    if registered.get('status') != 'complete' or registered.get('source_kind') != 'archived_int8_matrix':
        raise ValueError('Require registered INT8 matrix')
    protocol = registered['evaluation_protocol']
    annotations = args.engines/'evaluation.json'
    if sha256(annotations) != protocol['annotation_sha256']:
        raise ValueError('Held-out annotation hash mismatch')
    original = json.loads((args.source/'manifest.json').read_text())
    if original['status'] != 'complete' or original['arguments']['stage'] != 'accuracy':
        raise ValueError('Require completed FP16 accuracy stage')
    ids = protocol['image_ids']
    selected = set(ids)
    out = new_directory(args.output)
    record = dict(status='running', arguments={'stage':'accuracy'}, jobs=[],
                  evaluation_protocol=protocol, source_manifest_sha256=sha256(args.source/'manifest.json'),
                  source_kind='rescored_fp16_predictions', evidence_sha256={})
    write_json(out/'manifest.json', record)
    try:
        for name in ('stock-gpu','adapted-gpu','stock-fallback','adapted-strict',
                     'adapted-gpu-sparse','adapted-strict-sparse'):
            source = args.source/name
            old = json.loads((source/'manifest.json').read_text())
            if old['status'] != 'complete' or not selected.issubset(old['image_ids']):
                raise ValueError(f'Missing completed coverage: {source}')
            frames = [json.loads(line) for line in (source/'frames.jsonl').read_text().splitlines()]
            frames = [r for r in frames if r['image_id'] in selected]
            if [r['image_id'] for r in frames] != ids:
                raise ValueError('FP16 frames must cover held-out IDs exactly once in order')
            predictions = [r for r in json.loads((source/'predictions.json').read_text()) if r['image_id'] in selected]
            coco = COCO(str(annotations))
            if predictions:
                detected = coco.loadRes(predictions)
            else:
                detected = COCO()
                detected.dataset = {'images':coco.dataset['images'], 'categories':coco.dataset['categories'], 'annotations':[]}
                detected.createIndex()
            evaluator = COCOeval(coco, detected, 'bbox')
            evaluator.params.imgIds = ids
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
            destination = new_directory(out/name)
            write_json(destination/'predictions.json', predictions)
            # These are archived timing records, not new performance measurements.
            (destination/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in frames))
            metrics = dict(zip(('AP','AP50','AP75','AP_small','AP_medium','AP_large',
                                'AR1','AR10','AR100','AR_small','AR_medium','AR_large'),map(float,evaluator.stats)))
            updated = {**old, 'image_ids':ids, 'annotation_sha256':protocol['annotation_sha256'],
                       'full_coco_val2017_candidate':False, 'metrics_fraction':metrics,
                       'service_latency_ms':None, 'timing':'No new inference; frames retain archived FP16 timing only',
                       'rescore_source':str(source.resolve()),
                       'rescore_source_hashes':{f:sha256(source/f) for f in ('manifest.json','frames.jsonl','predictions.json')},
                       'arguments':{**old['arguments'],'annotations':str(annotations.resolve()),
                                    'output':str(destination.resolve()), 'limit':None}}
            write_json(destination/'manifest.json', updated)
            record['jobs'].append({'name':name,'returncode':0})
            for f in ('manifest.json','frames.jsonl','predictions.json'):
                record['evidence_sha256'][name+'/'+f] = sha256(destination/f)
            write_json(out/'manifest.json', record)
        record['status'] = 'complete'
    except BaseException as exc:
        record.update(status='failed',error=str(exc))
        raise
    finally:
        write_json(out/'manifest.json', record)


if __name__ == '__main__':
    main()
