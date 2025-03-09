# Should return:
# {'agg_metrics': 41.40999999999999, 'r1': {'0.5': 69.31, '0.55': 65.13, '0.6': 59.48, '0.65': 55.0, '0.7': 49.29, '0.75': 41.68, '0.8': 32.9, '0.85': 23.51, '0.9': 12.46, '0.95': 5.34}, 'mAP': {'0.5': 66.96, '0.55': 62.53, '0.6': 57.18, '0.65': 52.04, '0.7': 46.43, '0.75': 39.46, '0.8': 30.58, '0.85': 20.9, '0.9': 10.31, '0.95': 4.2, 'average': 39.06}, 'mIoU': 0.5863397571818805, 'invalid_predictions': 0.0, 'total': 3720}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node=8 evaluate.py --cfg-path lavis/projects/mr_BLIP/eval/charades.yaml