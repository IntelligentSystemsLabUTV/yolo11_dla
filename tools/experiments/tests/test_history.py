"""Provenance regressions: never mix runs or mistake suggestions for observations."""
import sys

from _paths import ROOT
sys.path.insert(0, str(ROOT/'history'))
from recover_commands import extract_commands, scope, terminal_records


def test_multiline_command_and_terminal_boundaries(tmp_path):
    path=tmp_path/'terminal.txt'
    path.write_text('❯ python tools/yolo_e2e_tester.py \\\n'
                    '  --engine first-yolo11.engine --runs 100\n'
                    'total_ms 20 19 21 22 18 23\n'
                    '❯ python tools/yolo_e2e_tester.py --engine second-yolo11.engine\n'
                    'total_ms 50 49 51 52 48 53\n')
    rows=terminal_records(path,'fixture','recovered_user_terminal_attachment')
    assert len(rows)==2
    assert rows[0]['stages_ms']['total_ms']['median']==19
    assert rows[1]['stages_ms']['total_ms']['median']==49
    assert all(r['per_frame_samples_recovered'] is False for r in rows)


def test_extraction_does_not_promote_finish_markers_or_collect_secrets():
    text='&&&& RUNNING TensorRT.trtexec # trtexec --onnx=yolo11n.onnx\n'
    text+='&&&& PASSED TensorRT.trtexec # trtexec --onnx=yolo11n.onnx\n'
    text+='python tools/train_coco_dla.py --api_key hidden\n'
    assert len(extract_commands(text))==1


def test_other_model_families_and_segmentation_are_excluded():
    assert scope('trtexec --onnx=yolo-dla-n-600ep.onnx')=='excluded_model_or_task'
    assert scope('trtexec --onnx=yolo11n-dla-seg.onnx')=='excluded_model_or_task'
    assert scope('trtexec --onnx=yolo11n-dla.onnx')=='yolo11_or_training_tool'
