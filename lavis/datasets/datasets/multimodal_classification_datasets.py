"""
 Adapted from Copyright (c)  2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from abc import abstractmethod
from lavis.datasets.datasets.base_dataset import BaseDataset


class MultimodalClassificationDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths,
                 start_idx=None, end_idx=None):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, start_idx, end_idx)

        self.class_labels = None

    @abstractmethod
    def _build_class_labels(self):
        pass

    @abstractmethod
    def _load_auxiliary_mappings(self):
        pass

