"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.common.registry import registry
from lavis.common.utils import get_cache_path
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.moment_retrieval_dataset import MomentRetrievalDataset


class MomentRetrievalBuilder(BaseDatasetBuilder):
    train_dataset_cls = MomentRetrievalDataset
    eval_dataset_cls = MomentRetrievalDataset

    def build(self):
        datasets = super().build()

        return datasets


@registry.register_builder("qvh")
class QVHBuilder(MomentRetrievalBuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/qvh/defaults.yaml",
    }


@registry.register_builder("charades_sta")
class Charades_STABuilder(MomentRetrievalBuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/charades_sta/defaults.yaml",
    }


@registry.register_builder("charades_sta-seconds_decimal")
class Charades_STA_seconds_decimal_Builder(MomentRetrievalBuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/charades_sta/seconds_decimal.yaml",
    }


@registry.register_builder("charades_sta-relative_decimal")
class Charades_STA_relative_decimal_Builder(MomentRetrievalBuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/charades_sta/relative_decimal.yaml",
    }


@registry.register_builder("charades_sta-relative_integer")
class Charades_STA_relative_integer_Builder(MomentRetrievalBuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/charades_sta/relative_integer.yaml",
    }


@registry.register_builder("anet")
class ANetBuilder(MomentRetrievalBuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/anet/defaults.yaml",
    }


@registry.register_builder("tacos")
class TACoSBuilder(MomentRetrievalBuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/tacos/defaults.yaml",
    }


@registry.register_builder("tacos-relative_integer")
class TACoSBuilder(MomentRetrievalBuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/tacos/relative_integer.yaml",
    }


@registry.register_builder("mixed")
class MixedBuilder(MomentRetrievalBuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mixed/defaults.yaml",
    }


# open-ended QA
