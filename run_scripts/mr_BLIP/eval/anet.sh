# Should return:
# {"agg_metrics": 32.647, "r1": {"0.5": 53.79, "0.55": 49.43, "0.6": 44.78, "0.65": 40.21, "0.7": 35.47, "0.75": 30.73, "0.8": 25.94, "0.85": 20.9, "0.9": 15.57, "0.95": 9.65}, "mAP": {"0.5": 53.79, "0.55": 49.43, "0.6": 44.78, "0.65": 40.21, "0.7": 35.47, "0.75": 30.73, "0.8": 25.95, "0.85": 20.9, "0.9": 15.57, "0.95": 9.65, "average": 32.65}, "mIoU": 0.515230127209404, "invalid_predictions": 0.0, "total": 17032}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node=8 evaluate.py --cfg-path lavis/projects/mr_BLIP/eval/anet.yaml