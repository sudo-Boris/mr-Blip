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
from tqdm import tqdm

from lavis.common.registry import registry
from lavis.common.logger import MetricLogger, SmoothedValue
from lavis.tasks.base_task import BaseTask
from lavis.common.dist_utils import (
    main_process,
    is_dist_avail_and_initialized,
    is_main_process,
)
from lavis.datasets.data_utils import prepare_sample

from lavis.tasks.mr_eval import eval_submission
from lavis.tasks.metrics_own import r1_and_mIoU


@registry.register_task("moment_retrieval")
class MomentRetrievalTask(BaseTask):
    def __init__(self):
        super().__init__()

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

    def after_evaluation(self, val_result, split_name, epoch, from_file=False, **kwargs):
        if not from_file:
            eval_result_file = self.save_result(
                result=val_result,
                result_dir=registry.get_path("result_dir"),
                filename="{}_epoch{}".format(split_name, epoch),
            )
        else:
            eval_result_file = kwargs.get("results_path")

        metrics = self._report_metrics(
            eval_result_file=eval_result_file, split_name=split_name, **kwargs
        )

        print("R1@0.5 | R1@0.7 | mIoU")
        print(f'{metrics["r1"]["0.5"]} & {metrics["r1"]["0.7"]} & {metrics["mIoU"]*100:.2f}')
        print("R1@0.5 | R1@0.7 | mAP@0.5 | mAP@0.75 | mIoU")
        print(f'{metrics["r1"]["0.5"]} & {metrics["r1"]["0.7"]} & {metrics["mAP"]["0.5"]} & {metrics["mAP"]["0.75"]} & {metrics["mIoU"]*100:.2f}')

        if is_main_process() and wandb.run is not None:
            # print(metrics["mAP"])
            wandb.log(
                {
                    "eval/agg_metrics": metrics["agg_metrics"],
                    # "eval/R1@0.3": metrics["r1"]["0.3"],
                    "eval/R1@0.5": metrics["r1"]["0.5"],
                    "eval/R1@0.7": metrics["r1"]["0.7"],
                    # "eval/mAP@0.3": metrics["mAP"]["0.3"],
                    "eval/mAP@0.5": metrics["mAP"]["0.5"],
                    "eval/mAP@0.75": metrics["mAP"]["0.75"],
                    "eval/mIoU": metrics["mIoU"],
                    "eval/invalid_predictions": metrics["invalid_predictions"],
                }
            )

        return metrics

    def evaluation(self, model, data_loader, cuda_enabled=True, save_every_sample=False):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        results = []

        i = 0
        for samples in tqdm(metric_logger.log_every(data_loader, print_freq, header)):
            sample_file = os.path.join(registry.get_path("result_dir"), 'samples', f"{samples['query_id'][0]}.json")
            try:
                with open(sample_file, 'r') as f:
                    eval_output = json.load(f)
                print(f"Skipping sample {samples['query_id'][0]} because it already exists")
            except Exception as e:
                print(f"Running {samples['query_id'][0]} because could not find or load result file: {e}")
                samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

                samples.update({"iters": i})

                eval_output = self.valid_step(model=model, samples=samples)
                if save_every_sample:
                    assert len(samples['query_id']) == 1, f"Expected a single query_id, but got {samples['query_id']}"
                    assert 'question_id' not in samples
                    os.makedirs(os.path.dirname(sample_file), exist_ok=True)
                    print(f"Saving sample {samples['query_id'][0]} to {sample_file}")
                    with open(sample_file, 'w') as f:
                        json.dump(eval_output, f)
            results.extend(eval_output)
            i += 1

            # print("breaking in evaluation in moment_retrieval.py")
            # break

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

    @main_process
    def _report_metrics(self, eval_result_file, split_name, **kwargs):
        """Report metrics to the logger and save to the output directory.
        Compute r1 for the moment retrieval task.
        Args:
            eval_result_file (str): path to the evaluation result file.
            split_name (str): split name, e.g. val, test
        """
        try:
            results = json.load(open(eval_result_file))
        except:
            raise ValueError("eval_result_file is not a valid json file")
            import pdb; pdb.set_trace()
        total_num = len(results)

        results_interpreted = [
            {
                "qid": r["qid"],
                "pred_relevant_windows": self.moment_str_to_list(
                    r["prediction"], kwargs.get("enforce_single_window", False),
                    kwargs.get("post_process_gpt4o", False)
                ),
                "relevant_windows": self.moment_str_to_list(
                    r["target"], kwargs.get("enforce_single_window", False),
                    kwargs.get("post_process_gpt4o", False)
                ),
            }
            for r in results
        ]

        all_metrics = eval_submission(results_interpreted, results_interpreted)

        r1_avg = all_metrics["brief"]["MR-full-R1-avg"]
        mIoU = all_metrics["brief"]["MR-full-mIoU"]
        invalid_pred_num = all_metrics["brief"]["MR-full-invalid_pred_num"]

        # log metrics
        metrics = {
            "agg_metrics": r1_avg,
            "r1": all_metrics["full"]["MR-R1"],
            "mAP": all_metrics["full"]["MR-mAP"],
            "mIoU": mIoU,
            "invalid_predictions": invalid_pred_num / total_num,
            "total": total_num,
        }

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

    def moment_str_to_list(self, m, enforce_single_window=False, post_process_gpt4o=False):
        return moment_str_to_list_gpt4o(m, enforce_single_window=enforce_single_window,
                                        post_process_gpt4o=post_process_gpt4o)
    

def nest_level(obj):
    # Not a list? So the nest level will always be 0:
    if type(obj) != list:
        return 0

    # Now we're dealing only with list objects:

    max_level = 0
    for item in obj:
        # Getting recursively the level for each item in the list,
        # then updating the max found level:
        max_level = max(max_level, nest_level(item))

    # Adding 1, because 'obj' is a list (here is the recursion magic):
    return max_level + 1

def filter_post_process_gpt4o(m):
    if not isinstance(m, str):
        # import pdb; pdb.set_trace()
        raise ValueError("m is not a valid moment string")
    m = re.sub(r'[a-zA-Z]', '', m)
    m = remove_special_chars(m)
    m = re.sub(r'\],(?!\[)', ']', m)  # remove comma after ] if not followed by [
    m = re.sub(r'(?<!\]),]', ']', m)  # remove comma before ] if not preceded by ]
    m = m.strip('.').strip(',')
    while '].' in m or '.]' in m: # or '],' in m or ',]' in m:
        m = m.replace('].', ']').replace('.]', '').replace('],', ']').replace(',]', '')
        m = re.sub(r'\],(?!\[)', ']', m)  # remove comma after ] if not followed by [
        m = re.sub(r'(?<!\]),]', ']', m)  # remove comma before ] if not preceded by ]
        m = m.strip('.').strip(',')
    n_opening_brackets = m.count('[')
    n_closing_brackets = m.count(']')
    if n_opening_brackets > n_closing_brackets:
        m = m.replace('[', '', n_opening_brackets - n_closing_brackets)
    elif n_closing_brackets > n_opening_brackets:
        m = m[::-1].replace(']', '', n_closing_brackets - n_opening_brackets)[::-1]
    if len(m) == 0:
        raise ValueError("m is not a valid moment string")
        return "[[-1, -1]]"
    while '.' in [m[0], m[-1]] or ',' in [m[0], m[-1]]:
        m = m.strip('.')
        m = m.strip(',')


    try:
        nl = nest_level(ast.literal_eval(m))
    except:
        # import pdb; pdb.set_trace()
        raise ValueError("m is not a valid moment string")
        return "[[-1, -1]]"
    
    if nl > 2:
        m = m[(nl-2):(2-nl)]
        

    if not re.match(r"\[\[.*\]\]", m):
        print("m:", m)
        if len(m) > 2:
            if m.startswith('[['):
                m = m[1:-1]
                if not re.match(r"\[\[.*\]\]", m):
                    # import pdb; pdb.set_trace()
                    raise ValueError("m is not a valid moment string")
                    return "[[-1, -1]]"
            else:
                m = "[" + m + "]"
                if not re.match(r"\[\[.*\]\]", m):
                    # import pdb; pdb.set_trace()
                    raise ValueError("m is not a valid moment string")
                    return "[[-1, -1]]"
        else:
            # import pdb; pdb.set_trace()
            raise ValueError("m is not a valid moment string")
            return "[[-1, -1]]"
    return m

def remove_special_chars(m):
    special_chars = ['\\', ':', '}', '*', '`', ' ', '\n', '(', ')', "'"]
    for c in special_chars:
        m = m.replace(c, '')
    return m


def moment_str_to_list_gpt4o(m, enforce_single_window=False, post_process_gpt4o=False):
    """Convert a string of moments to a list of moments.
    If predicted string is not a list, it means that the model has not yet learned to predict the right format.
    In that case, we return [[-1, -1]] to represent an error.
    This will then lead to an IoU of 0.
    Args:
        m (str): a string of moments, e.g. "[[0, 1], [4, 7]]"
    Returns:
        list: a list of moments, e.g. [[0, 1], [4, 7]]
    """
    if m == "[[-1, -1]]":
        return [[-1, -1]]

    if post_process_gpt4o:
        m = filter_post_process_gpt4o(m)
    else:
        if not re.match(r"\[\[.*\]\]", m):
            raise ValueError("m is not a valid moment string")
            import pdb; pdb.set_trace()
            return [[-1, -1]]

    try:
        _m = ast.literal_eval(m)
    except:
        raise ValueError("m is not a valid moment string")
        import pdb; pdb.set_trace()
        return [[-1, -1]]

    # if _m is not a list, it means that the model has not predicted any relevant windows
    # return error
    if not isinstance(_m, list):
        # raise ValueError()
        raise ValueError("m is not a valid moment string")
        import pdb; pdb.set_trace()
        return [[-1, -1]]
    
    try:
        if enforce_single_window and isinstance(_m[0], list):
            if len(_m[0]) == 0:
                raise ValueError("m is not a valid moment string")
                import pdb; pdb.set_trace()
                return [[-1, -1]]
            elif isinstance(_m[0][0], list) and isinstance(_m[0][0][0], int):
                _m = [_m[0][-1]]
    except:
        raise ValueError("m is not a valid moment string")
        import pdb; pdb.set_trace()
        return [[-1, -1]]

    # if not nested list, make it nested

    # if a sublist of _m has more than 2 elements, it means that the model has not learned to predict the right format
    # substitute that sublist with [-1, -1]
    for i in range(len(_m)):
        # if isinstance(i, int):
        #     _m[i] = [-1, -1]
        if len(_m[i]) != 2:
            raise ValueError("m is not a valid moment string")
            # print(f"Got a sublist with more or less than 2 elements!{_m[i]}")
            import pdb; pdb.set_trace()
            _m[i] = [-1, -1]

    return _m
