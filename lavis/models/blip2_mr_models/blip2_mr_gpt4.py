import copy
import os
import re
import time
import warnings
from tenacity import retry, wait_exponential, stop_after_attempt

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base
import torch
from lavis.models.blip2_mr_models.utils import (
    post_process_gpt4o, convert_to_absolute_time, moment_str_to_list,
    get_frames_gpt4o, post_process_gpt4o_qa,
)
# import cv2
# from moviepy.editor import VideoFileClip
# import base64
from openai import OpenAI, InternalServerError


@registry.register_model("mr_gpt4o")
class gpt4o_MR(Blip2Base):
    """
    BLIP2 model that uses GPT-4 for moment retrieval evaluation.
    Inherits from BLIP2 base model but adds GPT-4 evaluation capabilities.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "gpt4o": "configs/models/mr_gpt4o.yaml",
    }

    def __init__(
        self,
        # img_size=224,
        # drop_path_rate=0,
        # use_grad_checkpoint=False,
        # vit_precision="fp16",
        # freeze_vit=True,
        # num_query_token=32,
        # t5_model="google/flan-t5-xl",
        prompt="",
        max_tokens=200,
        # apply_lemmatizer=False,
        input_time_format="seconds_integers",
        interleave_data=False,
        use_short_interleave=False,
        frame_token_aggregation=None,
        task=None,
        model="gpt4o",
        temperature=0,
        use_task_prompt=False,
        only_frames=True,
        add_duration=False,
        use_timestamps=False,
        significant_digits_input_format=2,
        num_frames_for_answer=60,
        fps_for_answer=None,
        resample_frames=True,
        use_localizer=True,
        in_context_localizer=False,
        oracle_localizer=False,
        use_multiple_moments=False,
        use_clipstamps=False,
        use_cot=True,
        image_detail="low",
        grounded_qa=True,
        **kwargs
    ):
        super().__init__()
        
        # GPT-4 specific configurations
        self.model = model
        self.temperature = temperature
        self.use_cot = use_cot
        # Initialize other components from BLIP2_MR
        assert task is not None, "Task is required for GPT-4"
        self.task = task
        if "TAL" in task:
            raise NotImplementedError("TAL is not supported for GPT-4")
        else:
            self.post_process = post_process_gpt4o
        self.input_time_format = input_time_format
        self.interleave_data = interleave_data
        self.use_short_interleave = use_short_interleave
        self.use_task_prompt = use_task_prompt
        self.only_frames = only_frames
        self.add_duration = add_duration
        self.use_timestamps = use_timestamps
        self.frame_token_aggregation = frame_token_aggregation
        self.significant_digits_input_format = significant_digits_input_format
        self.image_detail = image_detail
        # qa args
        self.num_frames_for_answer = num_frames_for_answer
        self.fps_for_answer = fps_for_answer
        self.resample_frames = resample_frames
        self.use_localizer = use_localizer
        self.in_context_localizer = in_context_localizer
        self.oracle_localizer = oracle_localizer
        self.use_multiple_moments = use_multiple_moments
        self.use_clipstamps = use_clipstamps
        assert isinstance(grounded_qa, bool), f"grounded_qa must be a boolean, got {type(grounded_qa)=}"
        self.grounded_qa = grounded_qa
        self.max_total_frames = 250 # gpt4o api does not allow more
        print("#########################")
        print((
            f"Model configuration: {self.input_time_format=}, "
            f"{self.interleave_data=}, {self.use_task_prompt=}, "
            f"{self.only_frames=}, {self.add_duration=}"
        ))
        print("#########################")

        supported_openai_models = ['gpt-4o', 'gpt-4o-mini', 'gpt-4o-2024-11-20', 'gpt-4o-2024-08-06']
        assert model in supported_openai_models, f"Only {supported_openai_models} are supported for now, not {model}"
        self.sampling_kwargs = {
            "temperature": temperature,
            # **sampling_kwargs,
            "model": model,
            "max_tokens": max_tokens,
        }
        self.client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'), max_retries=10)
        self.convert_to_absolute_time = convert_to_absolute_time

    def get_answer_format_prompt(self):
        if self.input_time_format == "frames_integers":
            example = "[0, 30]"  # Frame numbers
            multiple_example = "[[0, 30], [45, 75]]"  # Multiple frame ranges
        elif self.input_time_format == "seconds_integers":
            # example = "[[0, 1], [4, 7]]"
            # we use single windows for now
            example = "[0, 1]"
            multiple_example = "[[0, 1], [4, 7]]"
        elif self.input_time_format == "seconds_floats":
            # example = "[[0, 1], [4, 7]]"
            # we use single windows for now
            assert self.significant_digits_input_format <= 5, \
                "Only up to 5 significant digits are supported for now"
            example = (f"[{0.35794:.{self.significant_digits_input_format}f}, "
                       f"{1.48399:.{self.significant_digits_input_format}f}]")
            multiple_example = (f"[[{0.35794:.{self.significant_digits_input_format}f}, "
                    f"{1.48399:.{self.significant_digits_input_format}f}], "
                    f"[{4.13814:.{self.significant_digits_input_format}f}, "
                    f"{7.03892:.{self.significant_digits_input_format}f}]]")
        elif self.input_time_format == "relative_integers":
            example = "[5, 26]"
            multiple_example = "[[5, 26], [45, 67]]"
        elif self.input_time_format == "relative_floats":
            assert self.significant_digits_input_format <= 5, \
                "Only up to 5 significant digits are supported for now"
            example = (f"[{0.35794:.{self.significant_digits_input_format}f}, "
                       f"{0.48399:.{self.significant_digits_input_format}f}]")
            multiple_example = (f"[[{0.35794:.{self.significant_digits_input_format}f}, "
                    f"{0.48399:.{self.significant_digits_input_format}f}], "
                    f"[{0.72414:.{self.significant_digits_input_format}f}, "
                    f"{0.93910:.{self.significant_digits_input_format}f}]]")
        else:
            raise NotImplementedError((
                f"Only frames_integers, seconds_integers, seconds_floats, relative_integers, relative_floats "
                f"are supported for now, not {self.input_time_format}"
            ))
        
        if self.use_multiple_moments:
            plural = "s"
            prompt = (
                "The answer should be in the format of a list of lists, each indicating the start and end of a window of moment. "
                f"For a each window, use the format [start_window, end_window], for instance {example}. "
                "The relevant actions might happen more than once in the video. In that case, "
                "you should provide multiple windows as a nested list "
                f"[[start_window_1, end_window_1], [start_window_2, end_window_2], ...], for instance {multiple_example}. "
                "It's important that your answer is in this format, otherwise the evaluation will fail. "
                "Before providing the windows, explain your reasoning step by step about why you chose these windows. "
                "After your reasoning, output `ANSWER: <your answer>` in the format specified above. "
                "If you detect multiple instances of the same action, you should return all of them. "
                "It's better to return more windows than to miss relevant moments. "
                "If you're not completely sure about a window but think it might be relevant, include it. "
                "The only constraint is that each window must be reasonably likely to contain the queried action. "
            )
        else:
            plural = ""
            prompt = (
                "The answer should be in the format of a list indicating the start and end of a window of moment, "
                f"[start_window, end_window], for instance {example}. "
                "If you detect multiple windows for the same moment, choose the most relevant one. "
                "It's important your final answer only contains one window. "
            )
        prompt += (
            "It's important that your answer is in this format, otherwise the evaluation will fail. "
            f"Before providing the window{plural}, explain your reasoning step by step about why you chose this window{plural}. "
            "After your reasoning, output `ANSWER: <your answer>` in the format specified above. "
        )
        if self.use_multiple_moments:
            prompt += (
                "If you detect multiple instances of the relevant action or event, you should return all of them. "
                "If you are not completely certain about a window, you can express your degree of certainty "
                "and the reason for your uncertainty in your thought process, "
                "but still return it if you think it is most likely to contain the queried action or event. "
                "The only constraint is that each window must be reasonably likely to contain the queried action or event. "
            )
        else:
            prompt += (
                "For instance, for cutting onion this could be between the time we see that the scene "
                "takes place in the kitchen and the time we see the onions being boiled in the pan. "
            )
        prompt += "It is very important that the answer is in this format, otherwise the evaluation will fail. "
        return prompt

    def get_cot_prompt(self, use_cot=True):
        if use_cot:
            cot_prompt = (
                "Think step by step. Reason about the events in the video and how they relate to the query. "
                "After your reasoning, output `ANSWER: <your answer>` in the format specified in the task prompt. "
                "Always provide a non-empty answer after your thoughts. "
            )
        else:
            cot_prompt = ""
        base_prompt_cot = (
            "The frames were sampled uniformly from the video. "
            f"{cot_prompt}"
            "If you think the event does not take place in the video, give your best guess, "
            "as otherwise the evaluation will be marked as incorrect. "
            "Never provide an empty list for <your answer>. "
            "The descriptions of moments are sometimes imprecise, so retrieve the closest moment. "
            "If you don't see an event remotely similar to the description, "
            "guess what is the most likely moment given the context. "
            "For instance, for cutting onion this could be between the time we see that the scene "
            "takes place in the kitchen and the time we see the onions being boiled in the pan. "
        )
        if self.use_multiple_moments:
            base_prompt_cot += (
                "Remember that the same action or event might happen multiple times in the video. "
                "If you see multiple instances of the same action, you should return all of them. "
                "If you have uncertainty about the window, you can express it in your thought process, "
                "but still return it if you think it is most likely to contain the queried action or event. "
            )
        if not self.add_duration:
            time_unit, numerical_precision = self.input_time_format.split("_")
            assert time_unit == "seconds"
            precision_prompt = {
                "integers": "integer seconds",
                "floats": f"seconds, optionally using up to {self.significant_digits_input_format} decimal places"
            }[numerical_precision]
            base_prompt_cot += f"Give the answer in {precision_prompt}."
        base_prompt_cot += "\n"
        return base_prompt_cot

    def get_qa_cot_prompt(self):
        if self.in_context_localizer:
            if self.use_multiple_moments:
                prompt = "The previous images were frames of the clips "
            else:
                prompt = "The previous images were frames of the clip "
            prompt += (
                "obtained from the windows you provided, "
                "now with a higher number of frames for this window. "
                "Pay attention to the previous frames from the entire video, not just the recent clip. "
                "It might be the case that clip is not that relevant to the question, "
                "or that important information is not in the shorter clip but in the previous frames. "
            )
        else:
            prompt = (
                "The previous images were frames of a video clip. "
                )
        prompt += (
            "You will be asked a question and given multiple options. "
            "Think step by step. Reason about the events in the video and how they relate to the query. "
            "After your reasoning, output `ANSWER: <your answer>` in the format specified in the task prompt. "
            "Always provide a non-empty answer after your thoughts. "
            "If you think the event does not take place in the video, give your best guess, "
            "as otherwise the evaluation will be marked as incorrect. "
            "Never provide an empty answer in the place of <your answer>. "
            "The descriptions of questions and answers are sometimes imprecise, "
            "so retrieve the most related answer. "
            "If you don't see an event or answer remotely similar to the description, "
            "guess what is the most likely answer given the context. "
            "For instance, for a question asking what was the action in the video, "
            "if you see a scene that takes place in the kitchen and then you see "
            "onions being boiled in the pan, you could assume that onions were cut as an action, "
            "even if you don't see the actual cutting. "
        )
        if self.in_context_localizer:
            prompt += "You shouldn't provide any windows in your answer now, as you already did before."
        return prompt
    
    def get_answer_qa_format_prompt(self, option_letters):
        
        p1 = (
            f"The answer options are associated with a letter out of {', '.join(option_letters)}. "
            "It's important your final answer only contains one letter, otherwise the evaluation will fail. "
            f"After `ANSWER:`, write only a letter out of {', '.join(option_letters)}, nothing else. "
            "Do not use any special characters or punctuation around or after `ANSWER:`. "
            "Do not highlight, make bold or use any markdown formatting or any other kind of formatting "
            "around `ANSWER:`. "
        )
        if self.use_cot:
            p1 += "After providing you reasoning, "
        else:
            p1 += "Directly " 
        p1 += (
            f"output simply `ANSWER: <letter {'/'.join(option_letters)}>`. Nothing else. "
            "Don't write Option or the actual text associated with the option."
        )
        return p1
    
    def get_mr_messages(self, samples):
        # assert isinstance(samples['duration'], list) and isinstance(samples['duration'], list) and isinstance(samples['duration'][0], (int, float)), f"Expected a plain list of ints or floats for duration, but got type {type(samples['duration'])=} {len(samples['duration'])=} {type(samples['duration'][0])=}"
        assert isinstance(samples['duration'], list) and isinstance(samples['duration'][0], (int, float)), f"Expected a plain list of ints or floats for duration, but got type {type(samples['duration'])=} {len(samples['duration'])=} {type(samples['duration'][0])=}"
        
        implied_batch_size = len(samples['duration'])
        # TODO: not sure how to do multi-window now ...
        if len(samples['video']) == implied_batch_size and isinstance(samples['video'][0][0], str):
            assert len(samples['timestamps']) == implied_batch_size, "Timestamps and video must have the same batch size"
            assert len(samples['timestamps'][0]) == len(samples['video'][0]), "Timestamps and video must have the same sequence length"
        else:
            raise ValueError("Cannot get implied batch size. Video is in wrong format.")
        
        contents = self.prompt_concatenation(
            samples['timestamps'],
            samples['duration'],
            samples.get('fps', [None]), # the fps samples for mr gpt, not the original fps of the video
            samples['video'],
            samples['query_prompt'],
            samples['task_prompt'],
        )
        assert len(contents) == 1, "Only batch size 1 is supported for now"
        content = contents[0]
        # (Pdb) samples['task_prompt']
        # ['Given the video and the query, find the relevant windows.\nRelevant windows: ']
        # (Pdb) samples["video_prompt_end"]
        # ['<extra_id_0>']
        # (Pdb) samples["query_prompt"]
        # ['Query: A girl and her mother cooked while talking with each other on facetime.\n']

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
        return messages
    
    def generate(
            self, 
            samples,
            **kwargs
        ):
        """
        input:
        - samples: dict of 
            - 'video', 'duration', 'query_id', 'timestamps', 'video_prompt_end', 
            - 'query_prompt', 'task_prompt', 'relevant_windows', 'iters'
        output:
        - dict of qid, raw_prediction, prediction, target, duration
        """
        if isinstance(samples["duration"], torch.Tensor):
            samples["duration"] = samples["duration"].tolist()
        messages = self.get_mr_messages(samples)
        if self.interleave_data:
            print(f"{messages[-1]['content'][-5]=}")
            print(f"{messages[-1]['content'][-3]=}")
        print(f"{messages[-1]['content'][-1]=}")
        outputs = self._call_gpt_with_retry(messages)
        pred_ans = [outputs.choices[0].message.content] # list for consistency with batching

        # process response
        out = {}
        out["duration"] = samples["duration"]
        prediction = [self.post_process(pred) for pred in pred_ans]
        if (
            self.input_time_format == "relative_integers"
            or self.input_time_format == "relative_floats"
        ):
            out["prediction"] = self.convert_to_absolute_time(
                prediction, out["duration"], self.input_time_format, self.significant_digits_input_format
            )
        else:
            out["prediction"] = prediction

        out["input_messages"] = messages
        out["raw_prediction"] = pred_ans
        out["answer"] = samples["relevant_windows"]
        out["qid"] = samples["query_id"]      


        print(f"{out['raw_prediction']=} {out['prediction']=} {out['answer']=} {out['qid']=}")
        return out
    
    @classmethod
    def from_config(cls, cfg):
        """
        Class must implement this method
        """
        # Similar to BLIP2_MR's from_config but with GPT-4 specific parameters
        # img_size = cfg.get("image_size")
        # num_query_token = cfg.get("num_query_token")
        # t5_model = cfg.get("t5_model")
        prompt = cfg.get("prompt", "")
        
        # GPT-4 specific configs
        gpt_temperature = cfg.get("temperature", 0.7)
        
        model = cls(
            task=cfg.get("task"),
            # img_size=img_size,
            # num_query_token=num_query_token,
            # t5_model=t5_model,
            prompt=prompt,
            model=cfg.get("gpt_model", "gpt-4o"),
            temperature=gpt_temperature,
            max_tokens=cfg.get("max_tokens", 150),
            interleave_data=cfg.get("interleave_data", False),
            use_short_interleave=cfg.get("use_short_interleave", False),
            use_timestamps=cfg.get("use_timestamps", False),
            use_task_prompt=cfg.get("use_task_prompt", False),
            input_time_format=cfg.get("input_time_format", "seconds_integers"),
            only_frames=cfg.get("only_frames", True),
            add_duration=cfg.get("add_duration", False),
            significant_digits_input_format=cfg.get("significant_digits_input_format", 2),
            image_detail=cfg.get("image_detail", "low"),
            # qa specific
            num_frames_for_answer=cfg.get("num_frames_for_answer", None), # 60),
            fps_for_answer=cfg.get("fps_for_answer", None),
            resample_frames=cfg.get("resample_frames", True),
            use_localizer=cfg.get("use_localizer", True),
            in_context_localizer=cfg.get("in_context_localizer", False),
            oracle_localizer=cfg.get("oracle_localizer", False),
            use_multiple_moments=cfg.get("use_multiple_moments", False),
            use_clipstamps=cfg.get("use_clipstamps", False),
            grounded_qa=cfg.get("grounded_qa", True),
        )
        
        return model
    
    def cast_times(self, times, duration, fps_ratio=None):
        times = [t.item() if isinstance(t, torch.Tensor) else t for t in times]
        duration = duration.item() if isinstance(duration, torch.Tensor) else duration
        time_unit, number_precision = self.input_time_format.split("_")
        if time_unit == "frames" and number_precision == "integers":
            assert fps_ratio is not None, "fps must be provided for casting time to frame ids"
            if len(times) > 1:
                return list(range(len(times)))
            # print(f"{times=} {fps_ratio=}")
            return [int(round(t.item() * fps_ratio.item())) for t in times]
        elif number_precision == "integers" and time_unit == "seconds":
            return [int(round(t)) for t in times]
        elif number_precision == "floats" and time_unit == "seconds":
            return [round(t, self.significant_digits_input_format) for t in times]
        elif number_precision == "integers" and time_unit == "relative":
            # [0 to 10^significant_digits_input_format]
            sd = self.significant_digits_input_format
            return [int(round((t / duration), sd) * 10**sd)
                     for t in times]
        elif number_precision == "floats" and time_unit == "relative":
            sd = self.significant_digits_input_format
            # 0 to 1 
            return [round((t / duration), sd)
                     for t in times]
        else:
            raise NotImplementedError((
                f"Only frames_integers, seconds_integers, seconds_floats, relative_integers, relative_floats "
                f"are supported for now, not {self.input_time_format}"
            ))

    def get_duration_prompt(self, durations):
        # assert len(durations) == 1, "Only one duration is supported for now"
        implemented_formats = ["frames_integers", "seconds_integers", "seconds_floats", "relative_integers", "relative_floats"]
        assert self.input_time_format in implemented_formats, \
            f"Only {implemented_formats} are supported for now, not {self.input_time_format}"
        time_unit = self.input_time_format.split("_")[0]
        if time_unit == "frames":
            return [f"The video has {d} frames.\n " for d in durations]
        elif time_unit == "seconds":
            return [f"The video lasts {d} seconds.\n " for d in durations]
        elif time_unit == "relative":
            return [f"The video lasts {d} timesteps.\n " for d in durations]
        else:
            raise NotImplementedError(f"Only frames, seconds and relative are supported for now, not {time_unit}")
        
    def apply_template_to_frames(self, base64Frames):
        if isinstance(base64Frames, list) \
            and all((isinstance(x, tuple) and len(x) == 1 and isinstance(x[0], str)) for x in base64Frames):
            base64Frames = [x[0] for x in base64Frames]
        if not (isinstance(base64Frames, list) and isinstance(base64Frames[0], str)):
            raise ValueError(f"base64Frames should be a list of strings, but got {type(base64Frames)=}")
        assert isinstance(base64Frames, list) and isinstance(base64Frames[0], str), f"base64Frames should be a list of strings, but got {type(base64Frames[0])=}"
        return [
            *map(lambda x: {"type": "image_url", 
                            "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": self.image_detail}}, base64Frames) # it was low for Charades, QVH
        ]
    
    def get_timestamps_prompt(self, timestamps):
        """
        Processes one batch of timestamps.
        If interleave_data is True, returns a list of timestamps prompts
        to be interleaved, of the same length as the original timestamps list.
        Otherwise, returns a list with a single prompt to be appended to the prompt.
        """
        time_unit, _ = self.input_time_format.split("_")
        if time_unit == "frames":
            if self.interleave_data:
                f = lambda t: f"frame {t}: "
            else:
                f = "frame numbers"
        elif time_unit == "seconds":
            if self.interleave_data:
                f = lambda t: f"{t} seconds: "
            else:
                f = "times in seconds"
        elif time_unit == "relative":
            if self.interleave_data:
                f = lambda t: f"timestep {t}: "
            else:
                f = "timesteps"
        else:
            raise NotImplementedError(f"Only frames, seconds and relative are supported for now, not {time_unit}")
        if self.interleave_data:
            if self.use_short_interleave:
                return [f"{t}" for t in timestamps]
            else:
                return [f"Frame at {f(t)}" for t in timestamps]
        else:
            return [f"The frames presented take place at the following {f}: {', '.join(map(str, timestamps))}. \n"]

    def prompt_concatenation(
        self,
        timestamps,
        durations,
        fps,
        base64Frames,
        query_prompt,
        task_prompt,
    ):

        ### video prompt
        # </vid> = <extra_id_0>\n
        video_prompt = [self.apply_template_to_frames(f) for f in base64Frames] # batched
        # video_prompt = self.apply_template_to_frames(base64Frames)
        # if "only_frames" in self.task:
        assert isinstance(fps, list) and len(fps) == len(durations), "fps and durations must have the same length"
        timestamps = [self.cast_times(t, d, f) for t, d, f in zip(timestamps, durations, fps)]
        durations = [self.cast_times([d], d, f)[0] for d, f in zip(durations, fps)]

        if self.only_frames:
            assert not self.add_duration, "Cannot have both only_frames and add_duration"
            assert (
                not self.input_time_format
            ), "Set input_time_format to False in the config to use only frames without timestamps."
            # prompt will at the end look as follows:
            # <vid> f1 > f2 > ... > fT > </vid>\n
            # video_prompt = ["<vid>" for _ in range(len(timestamps))]
        # elif "add_duration" in self.task:
        elif self.add_duration:
            # prompt will at the end look as follows:
            # <vid> f1 > f2 > ... > fT > duration </vid>\n
            query_prompt = [dp + qp for dp, qp in zip(self.get_duration_prompt(durations), query_prompt)]


        ### query_prompt + task_prompt
        # Question: q
        # Given the video and the query, find the relevant windows.
        # Relevant windows: [start_time, end_time]
        # concatenate query_prompt and task_prompt (list[str])
        # if "no_task_prompt" in self.task:
        if not self.use_task_prompt:
            text_prompt = [q + self.get_cot_prompt() + self.get_answer_format_prompt() for q in query_prompt]
        else:
            text_prompt = [q + t.replace('Relevant windows:', '') + self.get_cot_prompt() + self.get_answer_format_prompt()
                           for q, t in zip(query_prompt, task_prompt)]

        video_prompt, text_prompt = self.get_timestamped(timestamps, video_prompt, text_prompt)
        
        for i, _ in enumerate(video_prompt):
            video_prompt[i].append(text_prompt[i])
        
        return video_prompt
    
    def interleave_timestamps(self, timestamps, video_prompt):
        # Not necessarily true for GPT-4
        # assert (
        #     "integer" in self.input_time_format
        # ), "Interleaving only works with integer time formats where one number is one token."
        # iterate over the batch
        _batched_video_prompt = []
        for j, (timestamps_frames, frames) in enumerate(
            zip(timestamps, video_prompt) # these are batched
        ):
            _video_prompt = []
            if self.use_short_interleave:
                time_unit, numerical_precision = self.input_time_format.split("_")
                if time_unit == "seconds":
                    time_ref = "time in seconds"
                elif time_unit == "relative":
                    time_ref = "timesteps"
                else:
                    raise NotImplementedError
                # if numerical_precision == "floats":
                #     time_ref += " with decimals"
                # elif numerical_precision == "integers":
                #     time_ref += " as integers"
                # else:
                #     raise NotImplementedError
                _video_prompt.append({
                    "type": "text", 
                    "text": f"You will be presented frames of a video. Each frame is preceded with a number indicating the {time_ref}. "
                })
            timestamps_prompt = self.get_timestamps_prompt(timestamps_frames)
            if len(frames) != len(timestamps_frames):
                raise ValueError(f"We expect the number of frames ({len(frames)}) to be the same as the number of timestamps ({len(timestamps_frames)})")
            assert len(frames) == len(timestamps_frames), f"We expect the number of frames ({len(frames)}) to be the same as the number of timestamps ({len(timestamps_frames)})"
            
            # iternate over the number of frames -> t
            for i in range(len(timestamps_frames)):
                assert isinstance(timestamps_frames[i], (int, float)), f"timestamp expected to be an integer or float, but got {timestamps_frames[i]} of type {type(timestamps_frames[i])}" # check chat template ahs not been aplied yet
                _video_prompt.append({
                    "type": "text", "text": timestamps_prompt[i]
                })
                _video_prompt.append(frames[i])
            _batched_video_prompt.append(_video_prompt)

        return _batched_video_prompt

    def suffix_timestamps(self, timestamps, text_prompt):
        new_text_prompt = []
        for j, (timestamps_frames, text_prompt_element) in enumerate(
            zip(timestamps, text_prompt) # these are batched
        ):
            timestamps_prompt = self.get_timestamps_prompt(timestamps_frames)
            assert len(timestamps_prompt) == 1, "timestamp_prompt should be a single string"
            new_text_prompt.append(timestamps_prompt[0] + text_prompt_element)
        return new_text_prompt

    def get_timestamped(self, timestamps, video_prompt, text_prompt):
        if self.interleave_data:
            assert self.use_timestamps, "Interleaving only works with timestamps"
            video_prompt = self.interleave_timestamps(timestamps, video_prompt)
        elif self.use_timestamps:
            text_prompt = self.suffix_timestamps(timestamps, text_prompt)
        return video_prompt, text_prompt

    def get_relevant_moments(self, samples, relevant_moments_out):
        if self.resample_frames:
            relevant_moments, relevant_frames = self.get_relevant_frames_resampled(
                samples, relevant_moments_out
            )  # b, num_frames_for_answer, h, w, c
        else:
            assert not self.use_multiple_moments, "Multiple moments not supported for now"
            relevant_moments, relevant_frames = self.get_relevant_frames(
                samples, relevant_moments_out
            )  # b, num_frames_for_answer, c, w, h
            
        return relevant_moments, relevant_frames
    
    @torch.no_grad()
    def videoQA_generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=50,
        min_length=8,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        output_attentions=False,
    ):
        ### Stage 1: Moment Retrieval/ Localizer

        if "relevant_windows" not in samples:
            # gt windows not provided
            samples["relevant_windows"] = [[0, 0]]  # dummy answer
        else:
            samples["relevant_windows"] = samples["relevant_windows"]
        samples["query_id"] = samples["question_id"]
        # call generate() to get the moment retrieval output
        messages_history = []
        out_ans = {}
        out_ans["nframes"] = 0
        if self.use_localizer or self.oracle_localizer:
            assert self.grounded_qa, "Grounded QA must be set to use the localizer"
            if self.use_localizer:
                assert not self.oracle_localizer, "Cannot have both use_localizer and oracle_localizer"
                out_mr = self.generate(
                    samples,
                    use_nucleus_sampling=use_nucleus_sampling,
                    num_beams=num_beams,
                    max_length=max_length,
                    min_length=min_length,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_captions=num_captions,
                    temperature=temperature,
                    output_attentions=output_attentions,
                )
                if self.in_context_localizer:
                    content_assistant = out_mr["raw_prediction"]
                    assert not self.use_multiple_moments, "Multiple moments not supported for in-context mr gpt4o localizer"
                    assert len(content_assistant) == 1, "Only batch size 1 supported for now"
                    output_messages = [{
                        "role": "assistant",
                        "content": content_assistant[0]
                    }]
                    messages_history = out_mr["input_messages"] + output_messages
                    

                ### Stage 2: VideoQA/ Answerer
                # 1. uniform sampling of num_frames_for_answer from the relevant moment retrieved (out_mr['prediction'])
                relevant_moments_out = out_mr["prediction"]
                out_ans["nframes"] += len(samples["video"])
            elif self.oracle_localizer:
                assert not self.use_localizer, "Cannot have both use_localizer and oracle_localizer"
                relevant_moments_out = samples["relevant_windows"].view(-1, 2).tolist()
                if self.in_context_localizer:
                    messages_history = self.get_mr_messages(samples)
                    output_messages = [{
                        "role": "assistant",
                        "content": f"I observe {len(relevant_moments_out)} relevant moment{'' if len(relevant_moments_out) == 1 else 's'} "
                                  f"to the {samples['query_prompt'][0].lower().split('options: ')[0]}. ANSWER: {relevant_moments_out}"
                    }]
                    messages_history += output_messages
                    out_ans["nframes"] += len(samples["video"])
                relevant_moments_out = [relevant_moments_out]

            relevant_moments, relevant_frames = self.get_relevant_moments(samples, relevant_moments_out)
            # 2. get answer based on those frames using videoQA_answer()
            base64frames = relevant_frames
        else:
            base64frames = samples['video']
        if self.use_multiple_moments:
            samples["relevant_frames"] = []
            assert len(base64frames) == 1, "Only batch size 1 supported for now"
            for i, f in enumerate(base64frames[0]):
                ibase64frames = self.apply_template_to_frames(f)
                samples["relevant_frames"].extend(ibase64frames)
                out_ans["nframes"] += len(ibase64frames)
                if self.use_clipstamps:
                    samples["relevant_frames"].append({"type": "text", "text": f"Clip of relevant moment number {i+1}: "})
        else:
            assert len(base64frames) == 1, "Only batch size 1 supported for now"
            # extract single batch element and single window
            if self.oracle_localizer or self.use_localizer:
                base64frames = [f[0] for f in base64frames] # get first window
            samples["relevant_frames"] = self.apply_template_to_frames(base64frames[0])
            out_ans["nframes"] += len(samples["relevant_frames"])
        answerer_out = self.videoQA_answer(samples, messages_history=messages_history)
        out_ans = {**out_ans, **answerer_out}
        if self.use_localizer or self.oracle_localizer:
            out_ans["relevant_moments"] = [relevant_moments]
        elif self.grounded_qa:
            assert "relevant_moments" in out_ans
        else:
            assert not self.grounded_qa, "If doing grounded QA, relevant moments must be returned"
            out_ans["relevant_moments"] = [[-1, -1]]

        # # write the following to a wandb table
        # if self.use_wandb and is_main_process():
        #     log = {}
        #     predictions = out_ans["output_text"]
        #     # map the predictions {0, 1, 2, 3, 4} to the actual answers letters
        #     predictions = [self.ANS_MAPPING_I_TO_C[pred] for pred in predictions]
        #     # Log images and predictions
        #     if samples["iters"] % self.log_samples_every_n == 0:
        #         out, self.wandb_table_data = format_wandb_log_images_and_predictions_QA(
        #             samples=samples,
        #             wandb_table_data=self.wandb_table_data_eval,
        #             pred_mr=relevant_moments_out,
        #             pred=predictions,
        #             train_data=False,
        #         )
        #         log.update(out)
        #     # Log iteration
        #     wandb.log(log)
        print(f"videoQA_generate {out_ans=}")

        return out_ans

    def get_relevant_frames(self, samples, relevant_moments_out):
        assert self.fps_for_answer is None, "fps_for_answer not supported"
        relevant_moments = []  # for the batch

        for i, sample in enumerate(relevant_moments_out):
            moments_formatted = moment_str_to_list(sample)

            if moments_formatted == [[-1, -1]]:
                moments_formatted = [0, samples["duration"][i].item()]
            elif len(moments_formatted) > 1:
                # if there are multiple moments, take the first one (FOR NOW)
                moments_formatted = moments_formatted[0]
            else:
                moments_formatted = moments_formatted[0]

            if moments_formatted[1] > samples["duration"][i].item():
                moments_formatted[1] = round(samples["duration"][i].item())

            relevant_moments.append(moments_formatted)

        assert len(relevant_moments) == samples["video"].shape[0]

        relevant_frames = []  # b, num_frames_for_answer, c, w, h

        for i, (start, end) in enumerate(relevant_moments):

            if start >= end:
                end = samples["duration"][i].item()

            # start_idx is the index in the timestamps tensor that is closest to the start time
            start_idx = torch.argmin(torch.abs(samples["timestamps"][i] - start)).item()
            end_idx = torch.argmin(torch.abs(samples["timestamps"][i] - end)).item()

            # get the frames from the start_idx to end_idx
            frames = samples["video"][i, start_idx : end_idx + 1]

            assert frames.shape[0] > 0, "No frames found for the relevant moment."

            # if the number of frames is less than num_frames_for_answer, pad with the last frame
            if frames.shape[0] < self.num_frames_for_answer:
                pad_frames = torch.stack(
                    [frames[-1] for _ in range(self.num_frames_for_answer - frames.shape[0])]
                )
                frames = torch.cat([frames, pad_frames])

            # if the number of frames is more than self.num_frames_for_answer, sample uniformly self.num_frames_for_answer frames
            elif frames.shape[0] > self.num_frames_for_answer:
                idxs = torch.linspace(
                    0, frames.shape[0] - 1, self.num_frames_for_answer
                ).long()
                frames = frames[idxs]

            relevant_frames.append(frames)

        relevant_frames = torch.stack(
            relevant_frames
        )  # b, num_frames_for_answer, c, w, h

        return relevant_moments, relevant_frames

    def get_relevant_frames_resampled(
        self, samples, relevant_moments
    ):
        """Resample the frames for the relevant moments given the new start and end times.

        Args:
            samples (dict): A dictionary containing the following keys:
                - video (torch.Tensor): A tensor of shape (batch_size, num_frames, 3, H, W)
                - timestamps (torch.Tensor): A tensor of shape (batch_size, num_frames)
                - duration (torch.Tensor): A tensor of shape (batch_size)
            relevant_moments (list): A list of strings of length batch_size. Th
            num_frames_for_answer (int): The number of frames to sample for each relevant moment.

        Returns:
            relevant_frames (list): A list of tensors of shape (num_frames_for_answer, 3, H, W) for each relevant moment.

        """
        batch_moments_formatted, batch_relevant_frames = [], []  # for the batch

        for i, sample in enumerate(relevant_moments): # looping over batch elements
            if isinstance(sample, str):
                moments_formatted = moment_str_to_list(sample)
            elif isinstance(sample, list) and isinstance(sample[0], list) \
                and len(sample[0]) == 2 and isinstance(sample[0][0], (int, float)) \
                and not self.use_multiple_moments:
                moments_formatted = [sample[0]]
            else:
                moments_formatted = sample

            if moments_formatted == [[-1, -1]]:
                moments_formatted = [[0, round(samples["duration"][i].item())]]
            elif len(moments_formatted) == 1 and isinstance(moments_formatted[0], list) \
                and isinstance(moments_formatted[0][0], (int, float)) \
                and not self.use_multiple_moments:
                if moments_formatted[0][1] > samples["duration"][i].item():
                    moments_formatted[0][1] = round(samples["duration"][i].item())
            else:
                assert self.use_multiple_moments, "Multiple moments received, but use_multiple_moments is not set"
                for j, m in enumerate(copy.deepcopy(moments_formatted)):
                    if m[1] > samples["duration"][i].item():
                        moments_formatted[j][1] = round(samples["duration"][i].item())
            batch_moments_formatted.append(moments_formatted)

        
            total_frames, total_duration = 0, 0
            fps = self.fps_for_answer if self.fps_for_answer is not None else 1
            for start, end in moments_formatted:
                total_duration += end - start
                total_frames += (end - start) * fps
            unrestricted_total_frames = copy.deepcopy(total_frames)
            if self.fps_for_answer is None:
                total_frames = self.num_frames_for_answer
            elif self.num_frames_for_answer is not None:
                print(f"{self.num_frames_for_answer=}")
                total_frames = min(total_frames, self.num_frames_for_answer)
            if self.in_context_localizer:
                total_frames = min(total_frames, self.max_total_frames - len(samples['video']) - len(moments_formatted))
            print(f"total_frames={total_frames} total_duration={total_duration} {self.fps_for_answer=} {type(self.fps_for_answer)=}")

            relevant_frames = []  # b, num_frames_for_answer, c, w, h
            assert len(samples["video_path"]) == 1, "Only batch size 1 supported for now"
            total_nframes = 0
            for i, (start, end) in enumerate(moments_formatted):
                # import pdb; pdb.set_trace()
                # assert self.input_time_format in ["seconds_integers", "seconds_floats"], \
                #     "Only seconds supported for now" # othewise not sure if relevant moments is in absolute time or relative here

                if start >= end:
                    end = samples["duration"][i].item()
                frames_ratio = total_frames / unrestricted_total_frames
                if i + 1 == len(moments_formatted) and not self.fps_for_answer:
                    nframes = self.num_frames_for_answer - total_nframes
                else:
                    nframes = max(1, int(round(frames_ratio * (end - start) * fps)))

                total_nframes += nframes
                print(f"using {nframes=} frames for {start=} to {end=}")
                frames = get_frames_gpt4o(
                    samples["video_path"][0], start, end, nframes
                )

                relevant_frames.append(frames)
            batch_relevant_frames.append(relevant_frames)

        return batch_moments_formatted, batch_relevant_frames  # b, num_frames_for_answer, h, w, c

    def videoQA_answer(
        self,
        samples,
        messages_history=None,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=50,
        min_length=8,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        output_attentions=False,
    ):
        out = {}
        if messages_history is None:
            messages_history = []

        ### Q&A input and answer
        qa_input = samples["qa_input"]
        option_letters = self.extract_option_letters(qa_input[0])
        relevant_frames = samples["relevant_frames"]
        if self.use_localizer or self.oracle_localizer:
            assert self.grounded_qa, "Grounded QA must be set to use the localizer"
            qa_prompt = [cot + ans + qa for cot, ans, qa in 
                        zip([self.get_qa_cot_prompt()]*len(qa_input), 
                            [self.get_answer_qa_format_prompt(option_letters)]*len(qa_input),
                            qa_input)]
        else:
            # get_cot_prompt(self, use_cot=True)
            assert self.use_cot
            if self.grounded_qa:
                durations = samples["duration"]
                fps = samples.get("fps", [None for _ in durations])
                durations = [self.cast_times([d], d, f)[0] for d, f in zip(durations, fps)]
                qa_prompt = [
                    cot_qa + (
                        "Additionally, you must return a window with the relevant moment "
                        "in order to justify your answer with the relevant video frames. "
                    ) + cot_mr + (
                        "After your reasoning, first output the window with the relevant moment. "
                        "To do this write WINDOW: [start_window, end_window]. "
                    ) + ans_mr + (
                        "After the window, write you final answer to the multiple-choice question after ANSWER: <your answer>. "
                    ) + ans_qa + (
                    "Ultimately, your answer after reasoning should look like this, for example: WINDOW: [0, 10] ANSWER: B \n" 
                    ) +dur_prompt + qa 
                    for cot_qa, cot_mr, ans_mr, ans_qa, dur_prompt, qa in zip(
                        [self.get_qa_cot_prompt()]*len(qa_input), 
                        [self.get_cot_prompt(use_cot=False)]*len(qa_input), # use_cot False here as get_qa_cot_prompt already adds cot
                        [self.get_answer_format_prompt()]*len(qa_input),
                        [self.get_answer_qa_format_prompt(option_letters)]*len(qa_input),
                        self.get_duration_prompt(durations),
                        qa_input
                    )
                ]
            else:
                assert not self.use_timestamps, "Technically you can use timestamps without grounded QA, " \
                    "but it doesn't make much sense"
                assert not self.in_context_localizer, \
                    "In-context localizer not supported without grounded QA, " \
                    "as prompt is different"
                qa_prompt = [
                    cot_qa + ans_qa + (
                    "Ultimately, your answer after reasoning should look like this, for example: ANSWER: B \n" 
                    ) + qa 
                    for cot_qa, ans_qa, qa in zip(
                        [self.get_qa_cot_prompt()]*len(qa_input), 
                        [self.get_answer_qa_format_prompt(option_letters)]*len(qa_input),
                        qa_input
                    )
                ]
            if self.use_timestamps:
                assert self.interleave_data, "Suffix'ing timestamps not supported"
                assert len(samples["timestamps"]) == 1, "Only batch size 1 supported for now"
                timestamps = [self.cast_times(t, d, f) for t, d, f in 
                              zip(samples["timestamps"], samples["duration"], 
                                  samples.get("fps", [None for _ in samples["duration"]]))]
                relevant_frames = self.interleave_timestamps(timestamps, [relevant_frames])[0]


        assert len(qa_prompt) == 1, "Only batch size 1 supported for now"
        content = [
            *relevant_frames,
            {"type": "text", "text": qa_prompt[0]},
        ]
        messages = messages_history + [
            {
                "role": "user",
                "content": content,
            }
        ]
        print(f"VideoQA_answer input: {content[-1]=}")
        outputs = self._call_gpt_with_retry(messages)
        raw_ans = outputs.choices[0].message.content

        # 5. properly format the output dict
        if not (self.use_localizer or self.oracle_localizer) and self.grounded_qa:
            # parse both for qa answer and relevant moment.
            raw_ans_w = raw_ans.split("WINDOW")[-1].split("Window")[-1].strip().replace(":", "")
            print(f"{raw_ans=} -> {raw_ans_w=}")
            raw_ans_w = raw_ans_w.split("ANSWER")[0].split("Answer")[0].split(':')[0].strip()
            print(f"{raw_ans_w=} -> {raw_ans_w=}")
            pred_ans_w = post_process_gpt4o(raw_ans_w)
            print(f"{raw_ans_w=} -> {pred_ans_w=}")
            out["relevant_moments"] = [pred_ans_w]
            
        pred_ans = post_process_gpt4o_qa(raw_ans, options=",".join(option_letters))
        # TODO: maybe we need to output index?
        pred_ans_idx = [option_letters.index(pred_ans)]
        # out["output_text"] = pred_ans # "{letter}" only, already parsed
        out["output_text"] = pred_ans_idx # index, already parsed
        out["answer"] = samples["qa_output"] # "Option {letter}", ground truth
        out["qid"] = samples["question_id"]
        out["relevant_moments_gt"] = samples["relevant_windows"]

        return out

    @staticmethod
    def extract_option_letters(text):
        # Find all Option matches to count them
        option_matches = re.findall(r'Option [A-Z]:', text)
        
        # Extract option letters
        options = re.findall(r'Option ([A-Z]):', text)
        
        # Verify that the number of extracted letters matches the number of Option matches
        assert len(options) == len(option_matches), f"Mismatch in options: found {len(options)} letters, {len(option_matches)} matches"
        
        return options

    def _call_gpt_with_retry(self, messages):
        base_delay = 1  # Start with 1 second delay
        attempt = 0
        
        while True:
            try:
                return self.client.chat.completions.create(
                    messages=messages,
                    **self.sampling_kwargs,
                )
            except InternalServerError as e:
                attempt += 1
                
                # Lower image detail on retry
                for message in messages:
                    if "content" in message and isinstance(message["content"], list):
                        for item in message["content"]:
                            if isinstance(item, dict) and item.get("type") == "image_url":
                                item["image_url"]["detail"] = "low"
                
                delay = base_delay * (2 ** (attempt - 1))  # Exponential backoff
                warnings.warn(
                    f"Attempt {attempt} failed. Retrying with low detail images in {delay} seconds "
                    f"after {attempt} attempts. Error: {str(e)}")
                time.sleep(delay)

