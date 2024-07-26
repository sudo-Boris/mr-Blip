"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import random
from glob import glob

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners.runner_base import RunnerBase
from lavis.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluating")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    cfg = Config(parse_args())

    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    # setup wandb
    if cfg.run_cfg.wandb and get_rank() == 0:
        wandb.init(
            project=cfg.run_cfg.wandb_project,
            name=cfg.run_cfg.wandb_name,
            config=cfg.to_dict(),
            job_type="train",
        )

    # add number of occurance in log dir for same name cfg.run_cfg.wandb_name
    # e.g. cfg.run_cfg.wandb_name_1, cfg.run_cfg.wandb_name_2, cfg.run_cfg.wandb_name_3
    # get number of dirs in cfg.run_cfg.output_dir that are cfg.run_cfg.wandb_name_*

    def count_directories(parent_directory, pattern):
        return len(
            [
                d
                for d in glob(
                    os.path.join(os.getcwd(), "lavis", parent_directory, pattern)
                )
                if os.path.isdir(d)
            ]
        )

    num_dirs = count_directories(cfg.run_cfg.output_dir, f"{cfg.run_cfg.wandb_name}*")
    run_name = f"{cfg.run_cfg.wandb_name}-{num_dirs + 1}"

    cfg.run_cfg.output_dir = os.path.join(cfg.run_cfg.output_dir, run_name)

    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    runner = RunnerBase(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.evaluate(skip_reload=True)


if __name__ == "__main__":
    main()
