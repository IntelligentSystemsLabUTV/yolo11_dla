#!/usr/bin/env python3
"""Freeze disjoint COCO calibration and evaluation IDs without copying images."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import configure_imports
configure_imports()

import argparse
import json
import random
from common import new_directory, sha256, write_json


def partition(data, count, seed):
    entries = sorted(data['images'], key=lambda image: image['id'])
    ids = [image['id'] for image in entries]
    if len(set(ids)) != len(ids) or not 0 < count < len(ids):
        raise ValueError('Require unique image IDs and a non-empty held-out set')
    selected = set(random.Random(seed).sample(ids, count))
    def subset(keep):
        return {**data, 'images': [image for image in entries if image['id'] in keep],
                'annotations': [ann for ann in data['annotations'] if ann['image_id'] in keep]}
    return subset(selected), subset(set(ids)-selected)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, default=500)
    p.add_argument('--seed', type=int, default=2027)
    args = p.parse_args()
    data = json.loads(args.annotations.read_text())
    if len(data['images']) != 5000 or len(data['categories']) != 80:
        raise ValueError('Expected full COCO val2017, 5,000 images and 80 categories')
    calibration, heldout = partition(data, args.count, args.seed)
    for image in data['images']:
        if not (args.images/image['file_name']).is_file():
            raise FileNotFoundError(args.images/image['file_name'])
    records = [{'image_id': image['id'], 'file_name': image['file_name'],
                'sha256': sha256(args.images/image['file_name'])} for image in calibration['images']]
    out = new_directory(args.output)
    write_json(out/'calibration.json', calibration)
    write_json(out/'evaluation.json', heldout)
    write_json(out/'manifest.json', {
        'status': 'complete', 'seed': args.seed, 'calibration_count': len(records),
        'evaluation_count': len(heldout['images']), 'calibration_images': records,
        'annotations_sha256': sha256(args.annotations),
        'calibration_annotations_sha256': sha256(out/'calibration.json'),
        'evaluation_annotations_sha256': sha256(out/'evaluation.json'),
        'evaluation_image_ids': [image['id'] for image in heldout['images']],
        'protocol': 'Calibration uses images only. Report FP16/INT8 AP on the same held-out IDs; not full-val2017 AP.'})
    print(f'Saved {len(records)} calibration and {len(heldout["images"])} held-out images in {out}; no image copies')


if __name__ == '__main__':
    main()
