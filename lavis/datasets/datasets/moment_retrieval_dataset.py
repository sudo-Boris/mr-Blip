import os

import torch

from lavis.datasets.datasets.base_dataset import BaseDataset


class MomentRetrievalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        # set video clip if 'start'&'end' timestamp in data
        if "start" in ann:
            start, end = float(ann["start"]), float(ann["end"])
            # start, end = int(float(ann["start"]) * 100), int(float(ann["end"]) * 100)
            clip = [start, end]
        else:
            clip = None

        vname = ann["video"]
        video_path = os.path.join(self.vis_root, vname + ".mp4")

        frms, indices, fps = self.vis_processor(video_path, clip_proposal=clip)
        query = ann["query"]
        relevant_windows = str(ann["relevant_windows"])

        query_prompt = "Query: " + query + "\n"
        task_prompt = "Given the video and the query, find the relevant windows.\nRelevant windows: "

        # generate video prompt in the following format:
        # <vid><t><t+1><t+2>…<duration>[frame embeddings]</vid>
        # where <vid> is the video id, and <t> are the timestamps of each frame
        frms = frms.permute(1, 0, 2, 3)
        time_stamps = [float(idx / fps) for idx in indices]
        duration = ann["duration"]

        timestamps = [round(t, 2) for t in time_stamps]
        # timestamps.append(duration)
        timestamps = torch.tensor(timestamps)

        duration = torch.tensor(duration)

        video_prompt_end = "<extra_id_0>"

        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "video": frms,
            "duration": duration,
            "query_id": ann["qid"],
            "timestamps": timestamps,
            "video_prompt_end": video_prompt_end,
            "query_prompt": query_prompt,
            "task_prompt": task_prompt,
            "relevant_windows": relevant_windows,
        }
