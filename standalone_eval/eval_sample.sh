#!/usr/bin/env bash
# Usage: bash standalone_eval/eval_sample.sh
# submission_path=standalone_eval/sample_val_preds.jsonl
submission_path=standalone_eval/hl_val_submission.jsonl
gt_path=data/annotations/QVH/highlight_val_release.jsonl
save_path=standalone_eval/val_preds_metrics.json

PYTHONPATH=$PYTHONPATH:. python standalone_eval/eval.py \
--submission_path ${submission_path} \
--gt_path ${gt_path} \
--save_path ${save_path}
