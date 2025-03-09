"""
 Adapted from Copyright (c)  2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import re
import torch
from lavis.processors import transforms_video
from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor
from lavis.datasets.data_utils import load_video
from lavis.processors.randaugment import RandomAugment
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import cv2
import base64
import os

MAX_INT = registry.get("MAX_INT")


class ToUint8(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.to(torch.uint8)

    def __repr__(self):
        return self.__class__.__name__


class ToTHWC(object):
    """
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (C, T, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, H, W, C)
    """

    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.permute(1, 2, 3, 0)

    def __repr__(self):
        return self.__class__.__name__


class Gpt4oVideoBaseProcessor(BaseProcessor):
    def __init__(self, n_frms=MAX_INT, fps=None):
        self.n_frms = n_frms
        self.fps = fps


@registry.register_processor("gpt4o_caption")
class Gpt4oCaptionProcessor(BaseProcessor):
    """
    Same as BlipCaptionProcessor
    """
    def __init__(self, prompt="", max_words=50):
        self.prompt = prompt
        self.max_words = max_words

    def __call__(self, caption):
        caption = self.prompt + self.pre_caption(caption)

        return caption

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        prompt = cfg.get("prompt", "")
        max_words = cfg.get("max_words", 50)

        return cls(prompt=prompt, max_words=max_words)

    def pre_caption(self, caption):
        caption = re.sub(
            r"([.!\"()*#:;~])",
            " ",
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > self.max_words:
            caption = " ".join(caption_words[: self.max_words])

        return caption



@registry.register_processor("gpt4o_video_eval")
class Gpt4oVideoEvalProcessor(Gpt4oVideoBaseProcessor):
    def __init__(self, n_frms=MAX_INT, fps=None):
        super().__init__(n_frms=n_frms, fps=fps)
        self.n_frms = n_frms
        self.fps = fps
        max_image_pixels = int(os.getenv("CV_IO_MAX_IMAGE_PIXELS", 8847360))
        assert max_image_pixels >= 8847360, \
            f"CV_IO_MAX_IMAGE_PIXELS must be set to at least 8847360, but got {max_image_pixels}"

    def __call__(self, vpath, clip_proposal=None):
        clip, indices, fps = load_video(
            video_path=vpath,
            n_frms=self.n_frms,
            fps=self.fps,
            sampling="uniform",
            clip_proposal=clip_proposal,
            type="numpy",
        )
        # cv2.imencode expects height, width, channels
        clip = clip.transpose(1, 2, 3, 0) # so [C, T, H, W] -> [T, H, W, C]
        base64Frames = self.clip_to_base64frames(clip) # list of strings (each string one base64 str frame)
        return base64Frames, indices, fps

    @classmethod
    def clip_to_base64frames(cls, clip):
        base64Frames = []
        for t in range(clip.shape[0]):
            frame = clip[t, :, :, :]
            _, buffer = cv2.imencode(".jpg", frame) 
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        return base64Frames

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        n_frms = cfg.get("n_frms", MAX_INT)
        fps = cfg.get("fps", None)

        return cls(n_frms=n_frms, fps=fps)
    
