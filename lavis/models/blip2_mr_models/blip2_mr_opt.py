"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging

import os
import sys
import re
import ast

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.peft_model import PeftModel
import wandb

sys.path.append(sys.path[0] + "/..")

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_opt import OPTForCausalLM, OPTConfig
from transformers import AutoTokenizer
from lavis.common.dist_utils import is_main_process, download_cached_file
from lavis.common.utils import get_abs_path, is_url


@registry.register_model("blip2_opt_mr")
class Blip2_OPT_MR(Blip2Base):
    """
    BLIP2 OPT model.
    Supported model types:
        - pretrained_opt2.7b: pretrained model with OPT2.7b
        - pretrained_opt6.7b: pretrained model with OPT6.7b
        - caption_coco_opt2.7b: fintuned image captioning model with OPT2.7b
        - caption_coco_opt6.7b: fintuned image captioning model with OPT6.7b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_opt", "caption_coco_opt2.7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_opt2.7b": "configs/models/blip2/blip2_pretrain_opt2.7b.yaml",
        "pretrain_opt6.7b": "configs/models/blip2/blip2_pretrain_opt6.7b.yaml",
        "caption_coco_opt2.7b": "configs/models/blip2/blip2_caption_opt2.7b.yaml",
        "caption_coco_opt6.7b": "configs/models/blip2/blip2_caption_opt6.7b.yaml",
    }

    def __init__(
        self,
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        opt_model="facebook/opt-2.7b",
        prompt="",
        max_txt_len=200,
        num_beams=5,
        input_time_format="seconds_integers",
        interleave_data=False,
        task="lora",
    ):
        super().__init__()

        self.task = task
        self.use_lora = True if "lora" in task else False
        self.use_wandb = True if wandb.run is not None else False
        self.log_samples_every_n = 500
        self.log_samples_every_n_eval = 100
        self.num_beams = num_beams

        self.input_time_format = input_time_format
        self.interleave_data = interleave_data

        if self.use_wandb and is_main_process():
            self.wandb_table_data = []
            self.wandb_table_data_eval = []

        self.tokenizer = self.init_tokenizer()

        ### Vision backbone ######################################################
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        ##########################################################################

        ### Text backbone ########################################################
        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
        opt_config = OPTConfig.from_pretrained(opt_model)
        self.opt_model = OPTForCausalLM.from_pretrained(
            opt_model, config=opt_config  # torch_dtype=torch.float16
        )
        for name, param in self.opt_model.named_parameters():
            param.requires_grad = False
        # self.eos_token_id = self.opt_tokenizer(
        #     "\n", add_special_tokens=False
        # ).input_ids[0]

        self.annoying_numbers, _ = self.find_annoying_numbers(self.opt_tokenizer, 300)
        self.annoying_numbers_replacement_dict = (
            self.find_annoying_numbers_replacement_dict(self.annoying_numbers)
        )

        logging.info(
            "Annoying numbers and their replacement: {}".format(
                self.annoying_numbers_replacement_dict
            )
        )

        ### LORA ##########

        if self.use_lora:
            # If targeting all linear layers
            model_modules = str(self.opt_model.modules)
            pattern = r"\((\w+)\): Linear"
            linear_layer_names = re.findall(pattern, model_modules)

            names = []
            # Print the names of the Linear layers
            for name in linear_layer_names:
                names.append(name)
            target_modules = list(set(names))

            lora_config = LoraConfig(
                r=8,
                target_modules=target_modules,
                lora_alpha=8,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )

            self.opt_model = get_peft_model(self.opt_model, lora_config)
            self.opt_model.print_trainable_parameters()
        else:
            # freeze Opt
            for name, param in self.opt_model.named_parameters():
                param.requires_grad = False
                param.data = param.data.bfloat16()
        ##########################################################################

        ### Q-Former for Image Embeddings ########################################
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        )
        ##########################################################################

        self.max_txt_len = max_txt_len
        self.num_query_token = num_query_token
        self.prompt = prompt
        prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

    def forward(self, samples):
        image, qid = samples["video"], samples["query_id"]
        image_log = image
        video_prompt, video_prompt_end = (
            samples["timestamps"],
            samples["video_prompt_end"],
        )
        query_prompt, task_prompt = samples["query_prompt"], samples["task_prompt"]
        answer = samples["relevant_windows"]

        # uniform sampling
        b, t, c, w, h = image.shape
        image = image.reshape(-1, c, w, h)
        with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu"))):
            image_embeds = self.ln_vision(self.visual_encoder(image))  # bt, n, c
        _, n, _ = image_embeds.shape
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )  # bt n c

        ### Apply Q-Former for Image Embeddings ####################################
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        inputs_opt = self.opt_proj(query_output.last_hidden_state)

        # reshape the frames for opt from (bt, n, c) to (b, t * n, c)
        inputs_opt = inputs_opt.reshape(b, t, inputs_opt.shape[-2], -1)  # b, t, n, c
        inputs_opt = inputs_opt.reshape(b, -1, inputs_opt.shape[-1])  # b, t * n, c

        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
            image.device
        )  # b, t, n
        atts_opt = atts_opt.reshape(b, -1)  # b, t * n

        self.opt_tokenizer.padding_side = "right"

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            inputs_embs_mr, inputs_atts_mr, video_prompt = self.prompt_concatenation(
                video_prompt,
                inputs_opt,
                atts_opt,
                video_prompt_end,
                query_prompt,
                task_prompt,
            )

            ### Encode answer ################################################
            # add </s> to each answer in batch
            answer_w_eos = [a + "</s>" for a in answer]

            output_tokens_mr = self.opt_tokenizer(
                answer_w_eos,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

            targets_mr = output_tokens_mr.input_ids.masked_fill(
                output_tokens_mr.input_ids == self.opt_tokenizer.pad_token_id, -100
            )
            output_tokens_mask_mr = output_tokens_mr.attention_mask

            ### Apply moment retrieval prompt ######################################
            outputs = self.opt_model(
                inputs_embeds=inputs_embs_mr,
                attention_mask=inputs_atts_mr,
                # decoder_attention_mask=output_tokens_mask_mr,
                return_dict=True,
                labels=targets_mr,
            )
            loss = outputs.loss

            log = {}

            if self.use_wandb:
                log["train/log_likelihood_loss"] = loss.item()

            # write the following to a wandb table
            if self.use_wandb and is_main_process():
                # Log images and predictions
                if samples["iters"] % self.log_samples_every_n == 0:
                    # get samples
                    idx = torch.randint(0, b, (1,)).item()
                    frames = []
                    for frame in image_log[idx]:
                        frame = frame.cpu().numpy().transpose(1, 2, 0)
                        frame = wandb.Image(frame)
                        frames.append(frame)
                    pred = self.opt_tokenizer.decode(
                        torch.argmax(outputs.logits[idx], dim=1),
                        # skip_special_tokens=True,
                    )
                    _qid = qid[idx]
                    processed_pred = self.post_process(pred)
                    if self.interleave_data:
                        query = video_prompt[idx] + "</vid>" + query_prompt[idx]
                    else:
                        query = (
                            video_prompt[idx] + "<frames> </vid>" + query_prompt[idx]
                        )
                    answer = answer[idx]
                    duration = samples["duration"][idx]

                    # Annoying wandb workaround ...
                    # add samples to wandb table data log
                    self.wandb_table_data.append(
                        [
                            _qid,
                            frames,
                            query,
                            pred,
                            processed_pred,
                            answer,
                            duration,
                        ]
                    )

                    # create new table objects
                    wandb_table = wandb.Table(
                        columns=[
                            "qid",
                            "frames",
                            "query",
                            "pred",
                            "processed_pred",
                            "answer",
                            "duration",
                        ]
                    )

                    # add samples to table
                    for row in self.wandb_table_data:
                        wandb_table.add_data(*row)

                    log["Samples_during_training"] = wandb_table

                # Log iteration
                wandb.log(log)

        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=8,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        out = {}
        image, qid = samples["video"], samples["query_id"]
        image_log = image
        video_prompt, video_prompt_end = (
            samples["timestamps"],
            samples["video_prompt_end"],
        )
        query_prompt, task_prompt = samples["query_prompt"], samples["task_prompt"]
        answer = samples["relevant_windows"]

        # uniform sampling
        b, t, c, w, h = image.shape
        image = image.reshape(-1, c, w, h)
        with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu"))):
            image_embeds = self.ln_vision(self.visual_encoder(image))  # bt, n, c
        _, n, _ = image_embeds.shape
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )  # bt n c

        ### Apply Q-Former for Image Embeddings ####################################
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        inputs_opt = self.opt_proj(query_output.last_hidden_state)

        # reshape the frames for opt from (bt, n, c) to (b, t * n, c)
        inputs_opt = inputs_opt.reshape(b, t, inputs_opt.shape[-2], -1)  # b, t, n, c
        inputs_opt = inputs_opt.reshape(b, -1, inputs_opt.shape[-1])  # b, t * n, c

        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
            image.device
        )  # b, t, n
        atts_opt = atts_opt.reshape(b, -1)  # b, t * n

        self.opt_tokenizer.padding_side = "right"

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            inputs_embs_mr, inputs_atts_mr, video_prompt = self.prompt_concatenation(
                video_prompt,
                inputs_opt,
                atts_opt,
                video_prompt_end,
                query_prompt,
                task_prompt,
            )

            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = self.prompt

            prompt = [prompt] * b

            opt_tokens = self.opt_tokenizer(prompt, return_tensors="pt").to(
                image.device
            )
            input_ids = opt_tokens.input_ids
            attention_mask = torch.cat(
                [inputs_atts_mr, opt_tokens.attention_mask], dim=1
            )

            outputs = self.opt_model.generate(
                inputs_embeds=inputs_embs_mr,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                # eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_scores=True,
            )
            output_text = self.opt_tokenizer.batch_decode(outputs.sequences)

        # if duration is Tensor, convert to list
        if isinstance(samples["duration"], torch.Tensor):
            out["duration"] = samples["duration"].tolist()
        else:
            out["duration"] = samples["duration"]

        if (
            self.input_time_format == "relative_integers"
            or self.input_time_format == "relative_floats"
        ):
            # print("relative")
            prediction = [self.post_process(pred) for pred in output_text]
            out["prediction"] = self.convert_to_absolute_time(
                prediction, out["duration"]
            )
        else:
            out["prediction"] = [self.post_process(pred) for pred in output_text]

        out["raw_prediction"] = output_text
        out["answer"] = answer
        out["qid"] = qid

        # write the following to a wandb table
        if self.use_wandb and is_main_process():
            # Log images and predictions
            if samples["iters"] % self.log_samples_every_n_eval == 0:
                # get samples
                idx = torch.randint(0, b, (1,)).item()
                frames = []
                for frame in image_log[idx]:
                    frame = frame.cpu().numpy().transpose(1, 2, 0)
                    frame = wandb.Image(frame)
                    frames.append(frame)
                pred = output_text[idx]
                _qid = qid[idx]

                if self.interleave_data:
                    query = video_prompt[idx] + "</vid>" + query_prompt[idx]
                else:
                    query = video_prompt[idx] + "<frames> </vid> " + query_prompt[idx]
                answer = answer[idx]
                duration = out["duration"][idx]
                processed_pred = self.post_process(pred)
                if (
                    self.input_time_format == "relative_integers"
                    or self.input_time_format == "relative_floats"
                ):
                    processed_pred = self.convert_to_absolute_time(
                        [processed_pred], [duration]
                    )

                # Annoying wandb workaround ...
                # add samples to wandb table data log
                self.wandb_table_data_eval.append(
                    [
                        _qid,
                        frames,
                        query,
                        pred,
                        processed_pred,
                        answer,
                        duration,
                    ]
                )

                # create new table objects
                wandb_table = wandb.Table(
                    columns=[
                        "qid",
                        "frames",
                        "query",
                        "pred",
                        "processed_pred",
                        "answer",
                        "duration",
                    ]
                )

                # add samples to table
                for row in self.wandb_table_data_eval:
                    wandb_table.add_data(*row)

                wandb.log({"Samples_during_eval": wandb_table})

        return out

    @classmethod
    def from_config(cls, cfg):
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        opt_model = cfg.get("opt_model")
        num_beams = cfg.get("num_beams", 5)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        input_time_format = cfg.get("input_time_format", "seconds_integers")
        interleave_data = cfg.get("interleave_data", False)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_len", 200)
        task = cfg.get("task", "qformer_freeze_lora")

        model = cls(
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            opt_model=opt_model,
            num_beams=num_beams,
            prompt=prompt,
            max_txt_len=max_txt_len,
            input_time_format=input_time_format,
            interleave_data=interleave_data,
            task=task,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def load_checkpoint_from_config(self, cfg, **kwargs):
        """
        Load checkpoint as specified in the config file.

        If load_finetuned is True, load the finetuned model; otherwise, load the pretrained model.
        When loading the pretrained model, each task-specific architecture may define their
        own load_from_pretrained() method.
        """
        load_finetuned = cfg.get("load_finetuned", True)
        if load_finetuned:
            # load pre-trained weights
            pretrain_path = cfg.get("pretrained", None)
            assert pretrain_path is not None, "Pretrain_path is None."
            self.load_from_pretrained(url_or_filename=pretrain_path, **kwargs)
            logging.info("load pretrained weights from %s" % pretrain_path)

            # get finetuned lora adapter
            finetune_path = cfg.get("finetuned", None)
            assert (
                finetune_path is not None
            ), "Found load_finetuned is True, but finetune_path is None."
            self.load_checkpoint(url_or_filename=finetune_path)
            logging.info("load finetuned weights from %s" % finetune_path)
        else:
            # load pre-trained weights
            pretrain_path = cfg.get("pretrained", None)
            assert pretrain_path is not None, "Pretrain_path is None."
            self.load_from_pretrained(url_or_filename=pretrain_path, **kwargs)
            logging.info("load pretrained weights from %s" % pretrain_path)

    def prompt_concatenation(
        self,
        video_prompt,
        frames_for_opt,
        frames_atts_for_opt,
        video_prompt_end,
        query_prompt,
        task_prompt,
    ):

        ### video prompt
        # [t1, t2, t3, ..., duration]
        # </vid> = <extra_id_0>\n
        if "only_frames" in self.task:
            video_prompt = ["<vid>" for _ in range(len(video_prompt))]
            video_prompt_end = ["<extra_id_0>\n" for _ in range(len(video_prompt_end))]
        elif "add_duration" in self.task:
            video_prompt_end = [
                "{}<extra_id_0>\n".format(">" + prompt[-1].item())
                for prompt in video_prompt
            ]
            video_prompt = ["<vid>" for _ in range(len(video_prompt))]

        if self.input_time_format == "framenumbers":
            # add frame numbers to the video prompt
            new_video_prompts = []
            for prompt in video_prompt:
                _video_prompt = ">".join(str(i) for i in range(len(prompt)))
                _video_prompt += ">" + prompt[-1].item()
                new_video_prompts.append(_video_prompt)
            video_prompt = new_video_prompts

        elif self.input_time_format == "relative_floats":
            # add frame positions relative to video duration (in decimals -> 0 - 1)
            new_video_prompts = []
            # iterate over the batch
            for prompt in video_prompt:
                duration = prompt[-1].item()
                # convert to relative timestamps
                _video_prompt = ">".join(
                    str(round((timestamp.item() / duration), 2))
                    for timestamp in prompt[:-1]
                )
                # add the video duration
                _video_prompt += ">" + str(round(duration))
                new_video_prompts.append(_video_prompt)
            video_prompt = new_video_prompts

        elif self.input_time_format == "relative_integers":
            # add frame positions relative to video duration (in integers -> 0 - 100)
            new_video_prompts = []
            # iterate over the batch
            for prompt in video_prompt:

                duration = prompt[-1].item()
                # convert to relative timestamps
                _video_prompt = ">".join(
                    str(int(round((timestamp.item() / duration), 2) * 100))
                    for timestamp in prompt[:-1]
                )
                # add the video duration
                _video_prompt += ">" + str(round(duration))
                new_video_prompts.append(_video_prompt)
            video_prompt = new_video_prompts

        elif self.input_time_format == "seconds_integers":
            # add frame positions in seconds
            new_video_prompts = []
            # iterate over the batch
            for prompt in video_prompt:

                duration = prompt[-1].item()
                # convert to relative timestamps
                _video_prompt = ">".join(
                    (
                        str(int(round(timestamp.item())))
                        if round(timestamp.item()) not in self.annoying_numbers
                        else str(
                            self.annoying_numbers_replacement_dict[
                                round(timestamp.item())
                            ]
                        )
                    )
                    for timestamp in prompt[:-1]
                )
                # add the video duration
                duration = (
                    round(duration)
                    if round(duration) not in self.annoying_numbers
                    else self.annoying_numbers_replacement_dict[round(duration)]
                )
                # _video_prompt += ">" + str(duration)
                _video_prompt = ">" + _video_prompt + ">" + str(duration)
                new_video_prompts.append(_video_prompt)

                # for debugging
                # if a number in timestamp is annoying, breakpoint
                # for timestamp in prompt[:-1]:
                #     if round(timestamp.item()) in self.annoying_numbers:
                #         print("hello")

            video_prompt = new_video_prompts

        elif self.input_time_format == "seconds_floats":
            # add frame positions in seconds
            new_video_prompts = []
            # iterate over the batch
            for prompt in video_prompt:

                duration = prompt[-1].item()
                # convert to relative timestamps
                _video_prompt = ">".join(
                    str(round(timestamp.item(), 2)) for timestamp in prompt[:-1]
                )
                # add the video duration
                _video_prompt += ">" + str(round(duration))
                new_video_prompts.append(_video_prompt)
            video_prompt = new_video_prompts

        video_prompt_tokens = self.opt_tokenizer(
            video_prompt,
            padding="longest",
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(frames_for_opt.device)
        video_prompt_embs = self.opt_model.model.model.decoder.embed_tokens(
            video_prompt_tokens.input_ids
        )

        # </vid> = <extra_id_0>\n
        video_prompt_end_tokens = self.opt_tokenizer(
            video_prompt_end,
            padding="longest",
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(frames_for_opt.device)
        video_prompt_end_embs = self.opt_model.model.model.decoder.embed_tokens(
            video_prompt_end_tokens.input_ids
        )

        ### query_prompt + task_prompt
        # Question: q
        # Given the video and the query, find the relevant windows.
        # Relevant windows: [start_time, end_time]

        # concatenate query_prompt and task_prompt (list[str])
        if "no_task_prompt" in self.task:
            text_prompt = [q for q in query_prompt]
        else:
            text_prompt = [q + t for q, t in zip(query_prompt, task_prompt)]

        text_prompt_tokens = self.opt_tokenizer(
            text_prompt,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(frames_for_opt.device)
        text_prompt_embs = self.opt_model.model.model.decoder.embed_tokens(
            text_prompt_tokens.input_ids
        )
        start_token_embeds = self.opt_model.model.model.decoder.embed_tokens(
            torch.tensor([self.opt_tokenizer.bos_token_id] * 1).to(
                frames_for_opt.device
            )
        )

        if self.interleave_data:
            assert (
                "integer" in self.input_time_format
            ), "Interleaving only works with integer time formats where one number is one token."

            # add frame positions relative to video duration (in seconds)
            interleaved_video_prompt_embs = []
            video_prompt = []

            b, t_n, c = frames_for_opt.shape
            n = self.num_query_token
            t = t_n // n
            # iterate over the batch
            for j, (timestamp_embs, frame_embs) in enumerate(
                zip(video_prompt_embs, frames_for_opt)
            ):
                interleaved_prompt = torch.tensor([]).to(frames_for_opt.device)
                _video_prompt = ["</s>"]
                # iternate over the number of frames -> t
                for i in range(t):
                    # each frame has n tokens -> shape (t*n, c)
                    frame_emb = frame_embs[i * n : i * n + n]

                    # get every 2nd token to get actual timestamps [>, t1, >, t2, >, t3, >, ..., >, duration]
                    timestamp_emb = timestamp_embs[2 * i + 1].unsqueeze(0)

                    # for logging of input design
                    _video_prompt.append(
                        "f{i}-".format(i=i)
                        + self.opt_tokenizer.decode(
                            video_prompt_tokens.input_ids[j][2 * i + 1]
                        )
                        + ">"
                    )

                    # frame i and corresponding timestamp
                    frame_and_time = torch.cat(
                        [
                            frame_emb,
                            timestamp_emb,
                        ]
                    )
                    # add frame and timestamp "pair" to the interleaved prompt
                    interleaved_prompt = torch.cat([interleaved_prompt, frame_and_time])

                # add one more seperator and the duration tokens to the interleaved prompt
                # duration_embs_seperator = text_embs[2 * i + 1].unsqueeze(0)
                # duration_embs = text_embs[2 * i + 2].unsqueeze(0)
                duration_embs_seperator = timestamp_embs[-2].unsqueeze(0)
                duration_embs = timestamp_embs[-1].unsqueeze(0)
                interleaved_prompt = torch.cat(
                    [
                        start_token_embeds,
                        interleaved_prompt,
                        duration_embs_seperator,
                        duration_embs,
                    ]
                )

                # batch level list of interleaved video prompt
                interleaved_video_prompt_embs.append(interleaved_prompt)

                # for logging of input design
                # append the decoded video duration
                video_prompt.append(
                    "".join(_video_prompt)
                    + self.opt_tokenizer.decode(video_prompt_tokens.input_ids[j][-1])
                )

            interleaved_video_prompt_embs = torch.stack(
                interleaved_video_prompt_embs
            ).to(frames_for_opt.device)

            ### Concatenate interleaved_video_prompt, video_prompt_end, text_prompt
            inputs_embs_mr = torch.cat(
                [
                    interleaved_video_prompt_embs,
                    video_prompt_end_embs,
                    text_prompt_embs,
                ],
                dim=1,
            )

            interleaved_video_prompt_attn_embs = torch.ones(
                interleaved_video_prompt_embs.size()[:-1], dtype=torch.long
            ).to(frames_for_opt.device)
            interleaved_video_prompt_attn_embs = (
                interleaved_video_prompt_attn_embs.reshape(b, -1)
            )

            inputs_atts_mr = torch.cat(
                [
                    interleaved_video_prompt_attn_embs,
                    video_prompt_end_tokens.attention_mask,
                    text_prompt_tokens.attention_mask,
                ],
                dim=1,
            )
        else:
            ### Concatenate video_prompt, frames_for_opt, video_prompt_end, text_prompt
            inputs_embs_mr = torch.cat(
                [
                    video_prompt_embs,
                    frames_for_opt,
                    video_prompt_end_embs,
                    text_prompt_embs,
                ],
                dim=1,
            )

            inputs_atts_mr = torch.cat(
                [
                    video_prompt_tokens.attention_mask,
                    frames_atts_for_opt,
                    video_prompt_end_tokens.attention_mask,
                    text_prompt_tokens.attention_mask,
                ],
                dim=1,
            )

        # text = [t + "\n" for t in samples["text_input"]]

        # opt_tokens = self.opt_tokenizer(
        #     text,
        #     return_tensors="pt",
        #     padding="longest",
        #     truncation=True,
        #     max_length=self.max_txt_len,
        # ).to(image.device)

        # targets = opt_tokens.input_ids.masked_fill(
        #     opt_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        # )
        # if self.prompt:
        #     targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt

        # empty_targets = (
        #     torch.ones(atts_opt.size(), dtype=torch.long).to(image.device).fill_(-100)
        # )
        # targets = torch.cat([empty_targets, targets], dim=1)

        # inputs_embeds = self.opt_model.model.model.decoder.embed_tokens(opt_tokens.input_ids)
        # inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
        # inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
        # attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

        return inputs_embs_mr, inputs_atts_mr, video_prompt

    def convert_to_absolute_time(self, prediction, duration):
        """Convert relative timestamps to absolute timestamps.
        Args:
            prediction (list(str)): A list of predicted moments. Each moment is a list of start and end time of the moment, e.g. "[[0.0, 0.5], [1.0, 2.0]]"
            duration (list): A list of video durations.
        Returns:
            list(str): A list strings with the predicted moments as absolute timestamps.
        """

        assert (
            self.input_time_format == "relative_integers"
            or self.input_time_format == "relative_floats"
        ), "This function is only used for relative timestamps"

        # print("prediction before conversion:", prediction)

        # convert moments from string to list of floats
        prediction = [self.moment_str_to_list(m) for m in prediction]

        # TODO: copy each duration element i len(prediction[i]) times to handle cases where a prediction has multiple windows

        absolute_prediction = []

        for pred, dur in zip(prediction, duration):

            if self.input_time_format == "relative_integers":
                absolute_prediction.append(
                    [
                        (
                            [
                                round((float(start) / 100) * dur, 2),
                                round((float(end) / 100) * dur, 2),
                            ]
                            if start != -1 and end != -1
                            else [-1, -1]
                        )
                        for start, end in pred
                    ]
                )
            else:
                absolute_prediction.append(
                    [
                        (
                            [round(float(start) * dur, 2), round(float(end) * dur, 2)]
                            if start != -1 and end != -1
                            else [-1, -1]
                        )
                        for start, end in pred
                    ]
                )
            # print("absolute_prediction:", absolute_prediction)

        # convert moments from list of floats to string
        absolute_prediction = [str(m) for m in absolute_prediction]

        return absolute_prediction

    def logits_to_moments(self, logits):
        """Convert logits to moments.
        Args:
            logits (torch.Tensor): A tensor of shape (batch_size, num_tokens, vocab_size)
        Returns:
            list(list(int)): A list of moments. Each moment is a list of start and end time of the moment, e.g. [[0.0, 0.5], [1.0, 2.0]]
        """
        pred_string = self.opt_tokenizer.batch_decode(torch.argmax(logits, dim=2))
        # TODO: handle the case where the </s> token is not part of the string
        # then return an empty string # TODO: handle the case where \ is the last character in the string, add another \ to get \\
        pred_string = [pred.split("</s>")[1] for pred in pred_string]
        post_processed_predictions = [self.post_process(pred) for pred in pred_string]
        pred_moments = [self.moment_str_to_list(m) for m in post_processed_predictions]

        return pred_moments

    def post_process(self, pred):
        """Post process predicted output to be in the format of moments, i.e. [[0, 1], [4, 7]].
            - if no comma, i.e. " " → add comma, i.e. ", "
            - if t_start > t_end → swap them
            - if two comma: ",," → ","
        Args:
            pred (str): predicted output with potential errors, e.g. "[[0, 1], [4, 7]]"
        Returns:
            str: post processed predicted output, e.g. "[[0, 1], [4, 7]]"
        """

        pred = pred.split("</s>")[0]
        # pred = pred.split("</s>")[1]

        # check if the string has the right format of a nested list
        # the list should look like this: [[0, 1], [4, 7], ...]
        # if not, return "[[-1, -1]]"
        if not re.match(r"\[\[.*\]\]", pred):
            return "[[-1, -1]]"

        # remove the first and last bracket
        # e.g.
        #   [[0, 1] [4, 7]] -> [0, 1] [4, 7]
        #   [[0, 1], [4, 7]] -> [0, 1], [4, 7]
        pred = pred[1:-1]

        # split at any white space that is followed by a "]" to get a list of windows
        # e.g.
        #   "[0, 1] [4, 7]" → ["[0, 1]", "[4, 7]"]
        #   "[0, 1], [4, 7]" → ["[0, 1],", "[4, 7]"]
        windows = re.split(r"\s+(?=\])", pred)

        output = []

        for window in windows:
            # if there is one or more comma at the end of the window, remove it
            # e.g.
            #   "[0, 1]," → "[0, 1]"
            #   "[0, 1],," → "[0, 1]"
            window = re.sub(r",+$", "", window)

            # if there is no comma in the window, add one
            # e.g.
            #   "[0 1]" → "[0, 1]"
            window = re.sub(r"(\d) (\d)", r"\1, \2", window)

            # if there are two or more commas in the window, remove all but one
            # e.g.
            #   "[0,, 1]" → "[0, 1]"
            window = re.sub(r",+", ",", window)

            # if the two numbers are not in the right order, swap them
            # e.g.
            #   "[1, 0]" → "[0, 1]"
            # find all numbers in the window
            numbers = re.findall(r"\d+", window)
            # get the two numbers
            if len(numbers) == 2:
                t_start, t_end = numbers
                if int(t_start) > int(t_end):
                    window = "[" + t_end + ", " + t_start + "]"
            # if the window does not have 2 numbers return [-1, -1]
            else:
                window = "[-1, -1]"

            output.append(window)

        output = "[" + ", ".join(output) + "]"

        return output

    def moment_str_to_list(self, m):
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

        # check if the string has the right format of a nested list using regex
        # the list should look like this: [[0, 1], [4, 7], ...]
        # if not, return [[-1, -1]]
        if not re.match(r"\[\[.*\]\]", m):
            return [[-1, -1]]

        try:
            _m = ast.literal_eval(m)
        except:
            return [[-1, -1]]

        # if _m is not a list, it means that the model has not predicted any relevant windows
        # return error
        if not isinstance(_m, list):
            # raise ValueError()
            return [[-1, -1]]

        # if not nested list, make it nested

        # if a sublist of _m has more than 2 elements, it means that the model has not learned to predict the right format
        # substitute that sublist with [-1, -1]
        for i in range(len(_m)):
            if isinstance(i, int):
                _m[i] = [-1, -1]
            if len(_m[i]) != 2:
                # print(f"Got a sublist with more or less than 2 elements!{_m[i]}")
                _m[i] = [-1, -1]

        return _m

    def compute_IoU(self, pred, target):
        """Compute IoU between two windows.
        Args:
            pred (list): a list of start and end time of a window, e.g. [0.0, 0.5]
            target (list): a list of start and end time of a window, e.g. [1.0, 2.0]
        Returns:
            float: IoU between pred and target
        """

        def compute_overlap(pred, target):
            assert len(pred) == 2 and len(target) == 2, f"{pred}, {target}"
            if pred[0] > target[1] or pred[1] < target[0]:
                return 0
            else:
                # print(f"Overlap! ---- {pred} <==> {target}")
                return min(pred[1], target[1]) - max(pred[0], target[0])

        def compute_union(pred, target):
            assert len(pred) == 2 and len(target) == 2, f"{pred}, {target}"
            if pred[0] > target[1] or pred[1] < target[0]:
                return 0
            else:
                return max(pred[1], target[1]) - min(pred[0], target[0])

        try:
            union = compute_union(pred, target)
        except:
            print(f"Union error: {pred}, {target}")
            union = 0
        if union == 0:
            return 0

        try:
            overlap = compute_overlap(pred, target)
        except:
            print(f"Overlap error: {pred}, {target}")
            overlap = 0
        return overlap / union

    def find_annoying_numbers(
        self,
        tokenizer=AutoTokenizer.from_pretrained("facebook/opt-2.7b", use_fast=False),
        range_end=300,
    ):
        """
        Find numbers that are tokenized in more than one token by the Opt tokenizer.

        Args:
            tokenizer: A tokenizer object from the transformers library.
            range_end: The range of numbers to check.

        Returns:
            annoying_numbers: A list of numbers that are tokenized in more than one token.
            annoying_numbers_spance: A list of numbers that are tokenized in more than one token, but the first token is a space.
        """

        annoying_numbers = []
        annoying_numbers_spance = []
        for i in range(0, range_end):
            tokens = tokenizer(
                str(i),
                padding="longest",
                add_special_tokens=False,
                truncation=True,
                max_length=300,
                return_tensors="pt",
            )
            # print("len tokens:", len(tokens[:-1]))

            n_tokens = len(tokens["input_ids"].tolist()[0])

            if n_tokens > 1:
                if tokens["input_ids"].tolist()[0][0] == 3:
                    # for t in tokens:
                    #     print(tokenizer.decode(t))
                    # print("===")
                    annoying_numbers_spance.append(i)
                else:
                    # for t in tokens["input_ids"].tolist()[0]:
                    #     print(tokenizer.decode(t))
                    # print("===")
                    annoying_numbers.append(i)

        return annoying_numbers, annoying_numbers_spance

    def find_annoying_numbers_replacement_dict(self, annoying_numbers):
        """
        Find a the closes integer replacement for numbers that are tokenized in more than one token by the Opt tokenizer.

        Args:
            annoying_numbers: A list of numbers that are tokenized in more than one token.

        Returns:
            annoying_numbers_replacement_dict: A dictionary with the number as key and the replacement as value.
        """

        annoying_numbers_replacement_dict = {}
        for i in annoying_numbers:
            for j in range(100):
                if (i + j) not in annoying_numbers:
                    new_i = i + j
                    break
                if (i - j) not in annoying_numbers:
                    new_i = i - j
                    break

            annoying_numbers_replacement_dict[i] = new_i

        return annoying_numbers_replacement_dict
