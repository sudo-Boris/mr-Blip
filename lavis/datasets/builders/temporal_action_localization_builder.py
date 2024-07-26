"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.common.registry import registry
from lavis.common.utils import get_cache_path
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.temporal_action_localization_dataset import (
    TemporalActionLocalizationDataset,
)


class TemporalActionLocalizationBuilder(BaseDatasetBuilder):
    train_dataset_cls = TemporalActionLocalizationDataset
    eval_dataset_cls = TemporalActionLocalizationDataset

    def build(self):
        datasets = super().build()

        return datasets


@registry.register_builder("anet_TAL")
class ANetTALBuilder(TemporalActionLocalizationBuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/anet_TAL/defaults.yaml",
    }
