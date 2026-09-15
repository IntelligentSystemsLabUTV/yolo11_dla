#!/usr/bin/env bash
# COCO source: https://cocodataset.org/#download
# Use the HTTPS S3 endpoint of the official images.cocodataset.org bucket.
set -euo pipefail
coco_workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: bash download_coco_val2017.sh [dataset-root]"
    echo "Default: $coco_workspace/logs/datasets/coco; requires wget, unzip and python3."
    exit 0
fi
if (( $# > 1 )); then
    echo 'Expected at most one dataset-root argument.' >&2
    exit 2
fi
coco_root="${1:-$coco_workspace/logs/datasets/coco}"

validate_coco() {
    python3 - "$coco_root" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
data = json.loads((root/'annotations/instances_val2017.json').read_text())
images = data['images']
if len(images) != 5000 or len({im['id'] for im in images}) != 5000 or len(data['categories']) != 80:
    raise SystemExit('Expected 5,000 distinct images and 80 COCO categories.')
missing = [im['file_name'] for im in images
           if not (root/'images/val2017'/im['file_name']).is_file()
           or (root/'images/val2017'/im['file_name']).stat().st_size == 0]
if missing:
    raise SystemExit(f'Missing or empty images: {len(missing)}; first: {missing[:3]}')
print(f'OK: 5,000 images and detection annotations in {root.resolve()}')
PY
}

if validate_coco >/dev/null 2>&1; then
    validate_coco
    echo 'Dataset already present; no download needed.'
    exit 0
fi

for executable in wget unzip python3; do
    command -v "$executable" >/dev/null || { echo "Missing command: $executable" >&2; exit 1; }
done
mkdir -p "$coco_root/.downloads" "$coco_root/images" "$coco_root/annotations"
wget -c -O "$coco_root/.downloads/val2017.zip" \
    https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip
wget -c -O "$coco_root/.downloads/annotations_trainval2017.zip" \
    https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q -n "$coco_root/.downloads/val2017.zip" -d "$coco_root/images"
unzip -q -n "$coco_root/.downloads/annotations_trainval2017.zip" \
    annotations/instances_val2017.json -d "$coco_root"
validate_coco
