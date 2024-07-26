import logging
import json
import os
import numpy as np
import ast
import re
from collections import OrderedDict, defaultdict
import multiprocessing as mp

import torch
import torch.distributed as dist
import wandb

from lavis.common.registry import registry
from lavis.common.logger import MetricLogger, SmoothedValue
from lavis.tasks.base_task import BaseTask
from lavis.common.dist_utils import (
    main_process,
    is_dist_avail_and_initialized,
    is_main_process,
)
from lavis.datasets.data_utils import prepare_sample

from lavis.tasks.tal_eval import ANETdetection


@registry.register_task("temporal_action_localization")
class TALTask(BaseTask):
    def __init__(self):
        super().__init__()
        # read txt file with class names and remove "\n"
        with open("lavis/tasks/ANet_classes.txt", "r") as f:
            self.classes = f.read().splitlines()

    def valid_step(self, model, samples):
        results = []

        outputs = model.generate(samples)
        answer = outputs["answer"]
        qid = outputs["qid"]
        pred = outputs["prediction"]
        raw_pred = outputs["raw_prediction"]
        duration = outputs["duration"]
        assert len(qid) == len(answer)
        assert len(qid) == len(pred)

        i = 0
        for a, q, p, rp, d in zip(answer, qid, pred, raw_pred, duration):
            results.append(
                {
                    "qid": q + "_" + str(i),
                    "raw_prediction": rp,
                    "prediction": p,
                    "target": a,
                    "duration": d,
                }
            )
            i += 1

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
        )

        metrics = self._report_metrics(
            eval_result_file=eval_result_file, split_name=split_name
        )

        if is_main_process() and wandb.run is not None:
            # print(metrics["mAP"])
            wandb.log(
                {
                    "eval/agg_metrics": metrics["agg_metrics"] * 100,
                    # "eval/R1@0.3": metrics["r1"]["0.3"],
                    "eval/R1@0.5": metrics["r1"]["0.5"] * 100,
                    "eval/R1@0.7": metrics["r1"]["0.7"] * 100,
                    # "eval/mAP@0.3": metrics["mAP"]["0.3"],
                    "eval/mAP@0.5": metrics["mAP"]["0.5"] * 100,
                    "eval/mAP@0.75": metrics["mAP"]["0.75"] * 100,
                    # "eval/mIoU": metrics["mIoU"],
                    "eval/invalid_predictions": metrics["invalid_predictions"],
                }
            )

        return metrics

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        results = []

        i = 0
        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            samples.update({"iters": i})

            eval_output = self.valid_step(model=model, samples=samples)
            results.extend(eval_output)
            i += 1

            # print("breaking in evaluation in moment_retrieval.py")
            # break

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

    @main_process
    def _report_metrics(self, eval_result_file, split_name):
        """Report metrics to the logger and save to the output directory.
        Compute r1 for the moment retrieval task.
        Args:
            eval_result_file (str): path to the evaluation result file.
            split_name (str): split name, e.g. val, test
        """
        results = json.load(open(eval_result_file))
        total_num = len(results)
        invalid_pred_num = 0
        class_label_mismatch = 0

        preds = {
            "video-id": [],
            "t-start": [],
            "t-end": [],
            "label": [],
            "score": [],
        }
        targets = {
            "video-id": [],
            "t-start": [],
            "t-end": [],
            "label": [],
        }

        for r in results:
            targets_interpreted = self.tal_str_to_list(r["target"])

            for target in targets_interpreted:
                targets["video-id"].append(r["qid"])
                targets["t-start"].append(target[0])
                targets["t-end"].append(target[1])
                targets["label"].append(target[2])

            preds_interpreted = self.tal_str_to_list(r["prediction"])

            for pred in preds_interpreted:

                # count invalid predictions
                if preds_interpreted == [[-1, -1, -1]]:
                    invalid_pred_num += 1
                    break

                if len(pred) != 3:
                    print(f"Got a sublist with more or less than 3 elements!{pred}")
                    invalid_pred_num += 1
                    continue

                # TODO: make sure predicted label matches a class from ANet
                if pred[2] in self.classes:
                    label_tmp = pred[2]
                else:
                    label_tmp = "Error: class label mismatch!"
                    class_label_mismatch += 1

                # for debugging purposes, set pred labels to gt labels
                # get the gt label from the first respective target based on the video-id
                # idx_target = targets["video-id"].index(r["qid"])
                # label_tmp = targets["label"][idx_target]

                preds["video-id"].append(r["qid"])
                preds["t-start"].append(pred[0])
                preds["t-end"].append(pred[1])
                preds["label"].append(label_tmp)
                preds["score"].append(1)

        thresholds = np.linspace(0.5, 0.95, 10)

        self.anet_detection = ANETdetection(targets, tiou_thresholds=thresholds)

        mAP, average_mAP, mRecall, _, _ = self.anet_detection.evaluate(preds)
        # only  take recall@1 which is at mRecall[:][0]
        mRecall = mRecall[:, 0]
        # make recall and mAP into a dictionary with the thresholds as keys
        mRecall = {str(round(t, 2)): r for t, r in zip(thresholds, mRecall)}
        mAP = {str(round(t, 2)): a for t, a in zip(thresholds, mAP)}

        # log metrics
        metrics = {
            "agg_metrics": average_mAP,
            "r1": mRecall,
            "mAP": mAP,
            "mIoU": 0,
            # "mIoU": mIoU,
            "invalid_predictions": invalid_pred_num / total_num,
            "class_label_mismatch": class_label_mismatch,
            "total": total_num,
        }
        log_stats = {split_name: {k: v for k, v in metrics.items()}}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        logging.info(metrics)
        return metrics

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = self.train_step(model=model, samples=samples)

            # after_train_step()
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # update gradients every accum_grad_iters iterations
            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            # log to wandb
            if is_main_process() and wandb.run is not None:
                wandb.log(
                    {
                        "train/lr": optimizer.param_groups[0]["lr"],
                    }
                )

            # print("breaking in train_inner_loop in moment_retrieval.py")
            # break

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))

        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    def tal_str_to_list(self, m):
        """Convert a string of moments and a class label to a list of moments and labels.
        If predicted string is not a list, it means that the model has not yet learned to predict the right format.
        In that case, we return [[-1, -1]] to represent an error.
        This will then lead to an IoU of 0.
        Args:
            m (str): a string of moments, e.g. "[[0, 1, "label"], [4, 7, "label"]]"
        Returns:
            list: a list of moments, e.g. [[0, 1, "label"], [4, 7, "label"]]
        """
        if m == "[[-1, -1, -1]]":
            return [[-1, -1, -1]]

        # check if the string has the right format of a nested list using regex
        # the list should look like this: [[0, 1, "label"], [4, 7, "label"], ...]
        # if not, return [[-1, -1]]
        if not re.match(r"\[\[.*\]\]", m):
            return [[-1, -1, -1]]

        try:
            _m = ast.literal_eval(m)
        except:
            return [[-1, -1, -1]]

        # if _m is not a list, it means that the model has not predicted any relevant windows
        # return error
        if not isinstance(_m, list):
            # raise ValueError()
            return [[-1, -1, -1]]

        # if not nested list, make it nested

        # if a sublist of _m has more than 3 elements, it means that the model has not learned to predict the right format
        # substitute that sublist with [-1, -1]
        for i in range(len(_m)):
            # if isinstance(i, int):
            #     _m[i] = [-1, -1]
            if len(_m[i]) != 3:
                # print(f"Got a sublist with more or less than 3 elements!{_m[i]}")
                _m[i] = [-1, -1, -1]

        return _m

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

        datasets = dict()

        datasets_config = cfg.datasets_cfg
        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]
            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets()

            datasets[name] = dataset

        return datasets
