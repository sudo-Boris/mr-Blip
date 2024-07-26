import os
import re
import ast

import torch
from torch.cuda.amp import autocast as autocast
import wandb

# set the environment variable TOKENIZERS_PARALLELISM = false
# to disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def post_process(pred):
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

    # split at any white space that is followed by a "[" to get a list of windows
    # e.g.
    #   "[0, 1] [4, 7]" → ["[0, 1]", "[4, 7]"]
    #   "[0, 1], [4, 7]" → ["[0, 1],", "[4, 7]"]
    windows = re.split(r"\s+(?=\[)", pred)

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

        output.append(window)

    output = "[" + ", ".join(output) + "]"

    return output


def post_process_TAL(pred):
    """Post process predicted output to be in the format of moments with labels, i.e. [[0, 1, "label"], [4, 7, "label"]].
        - if no comma, i.e. " " → add comma, i.e. ", "
        - if t_start > t_end → swap them
        - if two comma: ",," → ","
    Args:
        pred (str): predicted output with potential errors, e.g. "[[0, 1, "label"], [4, 7, "label"]]"
    Returns:
        str: post processed predicted output, e.g. "[[0, 1, "label"], [4, 7, "label"]]"
    """

    pred = pred.split("</s>")[0]

    # if the string ends with a ",]", remove the comma
    # e.g.
    #   "[[0, 1, "label"], [4, 7, "label"],]" -> "[[0, 1, "label"], [4, 7, "label"]]"
    pred = re.sub(r",+\]", "]", pred)

    # check if the string has the right format of a nested list
    # the list should look like this: [[0, 1, "label"], [4, 7, "label"], ...]
    # if not, return "[[-1, -1]]"
    if not re.match(r"\[\[.*\]\]", pred):
        return "[[-1, -1, -1]]"

    # remove the first and last bracket
    # e.g.
    #   [[0, 1, "label"] [4, 7, "label"]] -> [0, 1, "label"] [4, 7, "label"]
    #   [[0, 1, "label"], [4, 7, "label"]] -> [0, 1, "label"], [4, 7, "label"]
    pred = pred[1:-1]

    # split at any white space that is followed by a "[" to get a list of windows
    # e.g.
    #   "[0, 1, "label"] [4, 7, "label"]" → ["[0, 1, "label"]", "[4, 7, "label"]"]
    #   "[0, 1, "label"], [4, 7, "label"]" → ["[0, 1, "label"],", "[4, 7, "label"]"]
    windows = re.split(r"\s+(?=\[)", pred)

    output = []

    for window in windows:
        # if there is one or more comma at the end of the window, remove it
        # e.g.
        #   "[0, 1, "label"]," → "[0, 1, "label"]"
        #   "[0, 1, "label"],," → "[0, 1, "label"]"
        window = re.sub(r",+$", "", window)

        # if there is no comma in the window, add one
        # e.g.
        #   "[0 1, "label"]" → "[0, 1, "label"]"
        window = re.sub(r"(\d) (\d)", r"\1, \2", window)
        # e.g.
        #  "[0, 1 "label"]" → "[0, 1, "label"]"
        window = re.sub(r"(\d), (\d) (\w+)", r"\1, \2, \3", window)
        # e.g.
        #  "[0 1 label]" → "[0, 1, "label"]"
        window = re.sub(r"(\d) (\d) (\w+)", r"\1, \2, \3", window)

        # if there are two or more commas in the window, remove all but one
        # e.g.
        #   "[0,, 1, "label"]" → "[0, 1, "label"]"
        window = re.sub(r",+", ",", window)

        # if the two numbers are not in the right order, swap them
        # e.g.
        #   "[1, 0, "label"]" → "[0, 1, "label"]"
        # find all numbers in the window
        numbers = re.findall(r"\d+", window)
        # find the labels in the window
        # first remove the numbers
        text = re.sub(r"\d+", "", window)
        # then find the labels
        label = re.findall(r"\w+", text)
        if label == []:
            label = ['"No label"']

        # get the two numbers
        if len(numbers) == 2:
            t_start, t_end = numbers
            if int(t_start) > int(t_end):
                window = "[" + t_end + ", " + t_start + ", '" + " ".join(label) + "']"
            # else:
            #     window = "[" + t_start + ", " + t_end + ", '" + " ".join(label) + "']"
        else:
            return "[[-1, -1, -1]]"

        output.append(window)

    output = "[" + ", ".join(output) + "]"

    return output


def format_wandb_log_images_and_predictions(
    samples,
    wandb_table_data,
    pred,
    video_prompt,
    post_process_fn=post_process,
    input_time_format=None,
    interleave_data=True,
    train_data=True,
):
    out = {}

    image, qid = samples["video"], samples["query_id"]
    b, t, c, h, w = image.size()
    query_prompt = samples["query_prompt"]
    answer = samples["relevant_windows"]

    # get samples
    idx = torch.randint(0, b, (1,)).item()
    frames = []
    for frame in image[idx]:
        frame = frame.cpu().numpy().transpose(1, 2, 0)
        frame = wandb.Image(frame)
        frames.append(frame)
    if interleave_data:
        query = video_prompt[idx] + "</vid>" + query_prompt[idx]
    else:
        query = video_prompt[idx] + "<frames> </vid>" + query_prompt[idx]

    pred = pred[idx]
    processed_pred = post_process_fn(pred)
    qid = qid[idx]
    answer = answer[idx]
    duration = samples["duration"][idx]

    if (
        input_time_format == "relative_integers"
        or input_time_format == "relative_floats"
    ):
        processed_pred = convert_to_absolute_time([processed_pred], [duration])

    # Annoying wandb workaround ...
    # add samples to wandb table data log
    wandb_table_data.append(
        [
            qid,
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
    for row in wandb_table_data:
        wandb_table.add_data(*row)

    if train_data:
        out["Samples_during_training"] = wandb_table
    else:
        out["Samples_during_eval"] = wandb_table

    return out, wandb_table_data


def convert_to_absolute_time(prediction, duration, input_time_format):
    """Convert relative timestamps to absolute timestamps.
    Args:
        prediction (list(str)): A list of predicted moments. Each moment is a list of start and end time of the moment, e.g. "[[0.0, 0.5], [1.0, 2.0]]"
        duration (list): A list of video durations.
    Returns:
        list(str): A list strings with the predicted moments as absolute timestamps.
    """

    assert (
        input_time_format == "relative_integers"
        or input_time_format == "relative_floats"
    ), "This function is only used for relative timestamps"

    # print("prediction before conversion:", prediction)

    # convert moments from string to list of floats
    prediction = [moment_str_to_list(m) for m in prediction]

    # TODO: copy each duration element i len(prediction[i]) times to handle cases where a prediction has multiple windows

    absolute_prediction = []

    for pred, dur in zip(prediction, duration):

        if input_time_format == "relative_integers":
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


def moment_str_to_list(m):
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


def tal_str_to_list(m):
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


def get_timestamps_as_seconds_integers(
    timestamps, durations, annoying_numbers_replacement_dict=None
):
    # add frame positions in seconds
    new_video_prompts = []
    new_timestamps = []
    new_durations = []
    # iterate over the batch
    for t, d in zip(timestamps, durations):

        duration = d.item()
        # convert to absolute timestamps
        _video_prompt = ">".join(
            (
                str(int(round(timestamp.item())))
                if round(timestamp.item())
                not in annoying_numbers_replacement_dict.keys()
                else str(annoying_numbers_replacement_dict[round(timestamp.item())])
            )
            for timestamp in t
        )
        # add the video duration
        duration = (
            round(duration)
            if round(duration) not in annoying_numbers_replacement_dict.keys()
            else annoying_numbers_replacement_dict[round(duration)]
        )
        # _video_prompt += ">" + str(duration)
        _video_prompt = ">" + _video_prompt + ">" + str(duration)
        new_video_prompts.append(_video_prompt)

        new_timestamps.append(
            torch.tensor(
                [
                    int(
                        round(timestamp.item())
                        if round(timestamp.item())
                        not in annoying_numbers_replacement_dict.keys()
                        else annoying_numbers_replacement_dict[round(timestamp.item())]
                    )
                    for timestamp in t
                ]
            )
        )
        new_durations.append(duration)

    return new_timestamps, new_durations, new_video_prompts


def get_timestamps_as_relative_integers(
    timestamps, durations, annoying_numbers_replacement_dict=None
):
    # add frame positions relative to video duration (in integers -> 0 - 100)
    new_video_prompts = []
    new_timestamps = []
    # iterate over the batch
    for t, d in zip(timestamps, durations):

        duration = d.item()
        # convert to relative timestamps
        _video_prompt = ">".join(
            str(int(round((timestamp.item() / duration), 2) * 100)) for timestamp in t
        )
        # add the video duration
        _video_prompt += ">" + str(round(duration))
        new_video_prompts.append(_video_prompt)

        new_timestamps.append(
            torch.tensor(
                [int(round((timestamp.item() / duration), 2) * 100) for timestamp in t]
            )
        )

    return new_timestamps, durations, new_video_prompts


def get_timestamps_as_seconds_floats(
    timestamps, durations, annoying_numbers_replacement_dict=None
):
    # add frame positions in seconds
    new_video_prompts = []
    new_timestamps = []
    # iterate over the batch
    for t, d in zip(timestamps, durations):

        duration = d.item()
        # convert to relative timestamps
        _video_prompt = ">".join(str(round(timestamp.item(), 2)) for timestamp in t)
        # add the video duration
        _video_prompt += ">" + str(round(duration))
        new_video_prompts.append(_video_prompt)

        new_timestamps.append(
            torch.tensor([round(timestamp.item(), 2) for timestamp in t])
        )

    return new_timestamps, durations, new_video_prompts


def get_timestamps_as_relative_floats(
    timestamps, durations, annoying_numbers_replacement_dict=None
):
    # add frame positions relative to video duration (in decimals -> 0 - 1)
    new_video_prompts = []
    new_timestamps = []
    # iterate over the batch
    for t, d in zip(timestamps, durations):
        duration = d.item()

        _video_prompt = ">".join(
            str(round((timestamp.item() / duration), 2)) for timestamp in t[:-1]
        )
        # add the video duration
        _video_prompt += ">" + str(round(duration))
        new_video_prompts.append(_video_prompt)

        # convert to relative timestamps
        new_timestamps.append(
            torch.tensor(
                [round((timestamp.item() / duration), 2) for timestamp in t]
                + [round(duration)]
            )
        )

    return new_timestamps, durations, new_video_prompts


def get_timestamps_as_framenumbers(
    timestamps, durations, annoying_numbers_replacement_dict=None
):
    # add frame numbers to the video prompt
    new_video_prompts = []
    new_timestamps = []
    for t, d in zip(timestamps, durations):
        _video_prompt = ">".join(str(i) for i in range(len(t)))
        _video_prompt += ">" + d.item()
        new_video_prompts.append(_video_prompt)

        # add frame numbers to the timestamps
        new_timestamps.append(torch.tensor([i for i in range(len(t))]))

    return new_timestamps, durations, new_video_prompts
