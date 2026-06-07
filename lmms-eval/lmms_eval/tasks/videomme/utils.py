import os
import re
import sys
from functools import partial
from pathlib import Path

import cv2
import datasets
import numpy as np
import yaml
from loguru import logger as eval_logger

VIDEO_TYPE = ["short", "medium", "long"]
CATEGORIES = ["Knowledge", "Film & Television", "Sports Competition", "Artistic Performance", "Life Record", "Multilingual"]

SUB_CATEGORIES = [
    "Humanity & History",
    "Literature & Art",
    "Biology & Medicine",
    "Finance & Commerce",
    "Astronomy",
    "Geography",
    "Law",
    "Life Tip",
    "Technology",
    "Animation",
    "Movie & TV Show",
    "Documentary",
    "News Report",
    "Esports",
    "Basketball",
    "Football",
    "Athletics",
    "Other Sports",
    "Stage Play",
    "Magic Show",
    "Variety Show",
    "Acrobatics",
    "Handicraft",
    "Food",
    "Fashion",
    "Daily Life",
    "Travel",
    "Pet & Animal",
    "Exercise",
    "Multilingual",
]

TASK_CATEGORIES = [
    "Temporal Perception",
    "Spatial Perception",
    "Attribute Perception",
    "Action Recognition",
    "Object Recognition",
    "OCR Problems",
    "Counting Problem",
    "Temporal Reasoning",
    "Spatial Reasoning",
    "Action Reasoning",
    "Object Reasoning",
    "Information Synopsis",
]

replace_prompt = " Please answer yes or no."

# with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
#     raw_data = f.readlines()
#     safe_data = []
#     for i, line in enumerate(raw_data):
#         # remove function definition since yaml load cannot handle it
#         if "!function" not in line:
#             safe_data.append(line)

#     config = yaml.safe_load("".join(safe_data))

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
# cache_dir = os.path.join(hf_home, cache_dir)
# base_cache_dir = config["dataset_kwargs"]["cache_dir"]
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "videomme.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


def parse_subtitle_time(time_str):
    h, m, s_ms = time_str.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def load_subtitles(subtitle_path):
    subtitles = {}
    with open(subtitle_path, "r", encoding="utf-8") as file:
        content = file.read().split("\n\n")
        for section in content:
            if section.strip():
                lines = section.split("\n")
                if len(lines) >= 3:
                    time_range = lines[1].split(" --> ")
                    start_time = parse_subtitle_time(time_range[0])
                    end_time = parse_subtitle_time(time_range[1])
                    text = " ".join(line for line in lines[2:])
                    subtitles[(start_time, end_time)] = text
    return subtitles


def convert_time_to_frame(time_in_seconds, fps):
    return int(time_in_seconds * fps)


def extract_subtitles(video_path, subtitle_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    subtitles = load_subtitles(subtitle_path)

    subtitle_frames = []
    for (start_time, end_time), text in subtitles.items():
        start_frame = convert_time_to_frame(start_time, fps)
        end_frame = convert_time_to_frame(end_time, fps)
        subtitle_frames.append((start_frame, end_frame, text))

    return subtitle_frames, total_frame


def videmme_process_docs_base(dataset: datasets.Dataset, type: str) -> datasets.Dataset:
    return dataset.filter(lambda x: x["duration"] == type)


videomme_process_docs_long = partial(videmme_process_docs_base, type="long")


def videomme_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["videoID"] + ".mp4"
    video_path = os.path.join(cache_dir, "data", video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.exists(video_path.replace("mp4", "MP4")):
        video_path = video_path.replace("mp4", "MP4")
    elif os.path.exists(video_path.replace("mp4", "mkv")):
        video_path = video_path.replace("mp4", "mkv")
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


def _videomme_doc_to_text_default(doc, post_prompt=None):
    option_prompt = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option."
    question = doc["question"]
    option = "\n".join([f"{opt}" for i, opt in enumerate(doc["options"])])
    question = question + "\n" + option
    post_prompt = post_prompt if post_prompt is not None else "The best answer is:"
    return option_prompt + "\n" + question + "\n" + post_prompt


def videomme_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}

    if "format" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["format"] == "qwen3_vl":
        return videomme_doc_to_text_qwen3vl(doc, lmms_eval_specific_kwargs)

    post_prompt = lmms_eval_specific_kwargs["post_prompt"] if "post_prompt" in lmms_eval_specific_kwargs else "The best answer is:"
    return _videomme_doc_to_text_default(doc, post_prompt)


def videomme_doc_to_text_qwen3vl(doc, lmms_eval_specific_kwargs=None):
    # Keep Qwen3-VL on the same VideoMME prompt as assets/videomme.jsonl.
    return _videomme_doc_to_text_default(doc, "\nAnswer with the option's letter from the given choices directly.")


# Frames + Subs
# This video's subtitles are listed below:
# 【subtitles】

# Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.
# 【question】
# The best answer is:
# Frames / Frames + Audio
# Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
# 【question】
# The best answer is:


def videomme_doc_to_text_subtitle(doc, lmms_eval_specific_kwargs=None):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["videoID"] + ".mp4"
    video_path = os.path.join(cache_dir, "data", video_path)
    subtitle_path = os.path.join(cache_dir, "subtitle", doc["videoID"] + ".srt")
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(subtitle_path):  # Denote have subtitle
        subtitle = open(subtitle_path).readlines()
    else:
        subtitle = ""
    subtitles_prompt = "This video's subtitles are listed below: \n"
    if subtitle == "":
        subtitle = "No subtitles available"
    else:
        if "gemini_api_flag" in lmms_eval_specific_kwargs:  # specific for gemini_api
            if lmms_eval_specific_kwargs["gemini_api_flag"] == "full subtitle":
                textlist = []
                for ele in subtitle:
                    pattern = r'<font color="white" size=".72c">(.*?)</font>'
                    matches = re.findall(pattern, ele)
                    if matches:
                        textlist.append(matches[0])
                subtitle_text = "\n".join(textlist)
        else:
            if "frame_num" in lmms_eval_specific_kwargs:
                frame_num = lmms_eval_specific_kwargs["frame_num"]
                subtitle_by_frame, total_frame = extract_subtitles(video_path, subtitle_path)
                if frame_num == -1:
                    frame_num = total_frame
                uniform_sampled_frames = np.linspace(0, total_frame - 1, frame_num, dtype=int).tolist()

                subtitle_by_frame_idx = []
                for frame_idx in uniform_sampled_frames:
                    for idx, title in enumerate(subtitle_by_frame):
                        if frame_idx < title[1] and frame_idx >= title[0]:
                            subtitle_by_frame_idx.append(idx)
                subtitle_by_frame_idx = list(set(subtitle_by_frame_idx))

                textlist = []
                for idx in subtitle_by_frame_idx:
                    pattern = r'<font color="white" size=".72c">(.*?)</font>'
                    raw_text = re.findall(pattern, subtitle_by_frame[idx][2])
                    try:
                        textlist.append(raw_text[0])
                    except:
                        continue
                subtitle_text = "\n".join(textlist)
        subtitle = subtitle_text

    if "format" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["format"] == "qwen3_vl":
        prompt = videomme_doc_to_text_qwen3vl(doc, lmms_eval_specific_kwargs)
        full_prompt = subtitles_prompt + subtitle + "\n" + prompt
        return full_prompt

    option_prompt = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option."
    question = doc["question"]
    option = "\n".join([f"{opt}" for i, opt in enumerate(doc["options"])])
    question = question + "\n" + option
    full_prompt = subtitles_prompt + subtitle + "\n" + option_prompt + "\n" + question + "\n" + "The best answer is:"
    return full_prompt


def _strip_choice_prefix(text):
    return re.sub(r"^\s*[A-Z]\s*[\.\)]\s*", "", str(text or "").strip(), flags=re.IGNORECASE).strip()


def _normalize_choice_letters(valid_choices=None):
    if not valid_choices:
        return "ABCD"
    letters = []
    iterable = valid_choices if not isinstance(valid_choices, str) else list(valid_choices)
    for item in iterable:
        letter = str(item).strip().upper()[:1]
        if re.fullmatch(r"[A-Z]", letter) and letter not in letters:
            letters.append(letter)
    return "".join(letters) or "ABCD"


def _extract_labeled_option(option):
    if option is None:
        return None
    text = str(option or "").strip()
    if not text:
        return None
    match = re.match(r"^\s*([A-Z])\s*[\.\)]\s*(.+?)\s*$", text, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).upper(), match.group(2).strip()


def _infer_choice_info(sample):
    option_text_by_letter = {}

    def add_option(label, option_text=""):
        letter = str(label or "").strip().upper()[:1]
        if not re.fullmatch(r"[A-Z]", letter):
            return
        option_text = _strip_choice_prefix(option_text)
        option_text_by_letter.setdefault(letter, option_text)

    for key in ("choice_labels", "choices_labels", "valid_choices"):
        labels = sample.get(key)
        if isinstance(labels, (list, tuple)):
            for label in labels:
                add_option(str(label))

    for key in ("option", "options", "choices"):
        options = sample.get(key)
        if isinstance(options, dict):
            for label, option_text in options.items():
                add_option(str(label), str(option_text))
        elif isinstance(options, (list, tuple)):
            for idx, option in enumerate(options):
                labeled = _extract_labeled_option(option)
                if labeled:
                    add_option(*labeled)
                elif idx < 26:
                    add_option(chr(ord("A") + idx), str(option))

    prompt = str(sample.get("input") or "")
    for match in re.finditer(r"(?m)^\s*([A-Z])\s*[\.\)]\s*(.+?)\s*$", prompt):
        add_option(match.group(1), match.group(2))

    answer = str(sample.get("answer") or "").strip().upper()
    if re.fullmatch(r"[A-Z]", answer):
        add_option(answer)

    return _normalize_choice_letters(list(option_text_by_letter.keys())), option_text_by_letter


def _extract_choice_letter(text, valid_choices=None, option_text_by_letter=None):
    if not text:
        return ""
    raw = (text or "").strip()
    t = raw.upper()
    if not t:
        return ""
    valid_choice_letters = _normalize_choice_letters(valid_choices)
    choice_class = re.escape(valid_choice_letters)

    # 1) Strict single-token answer forms: "A", "(B)", "C.", "[D]".
    m = re.match(rf"^\s*[\(\[]?\s*([{choice_class}])\s*[\)\].,:;!?\u3002\uff0c\uff1a\uff1b]?[\s]*$", t)
    if m:
        return m.group(1)

    # 2) Common prefixed forms: "Answer: B", "Option C", "Choice is D".
    prefixed_patterns = [
        rf"\b(?:ANSWER|OPTION|CHOICE)\b\s*(?:IS)?\s*[:=\-]?\s*[\(\[]?\s*([{choice_class}])\b",
        rf"\b(?:THE\s+ANSWER\s+IS|I\s+CHOOSE|I\s+PICK)\b\s*[:=\-]?\s*[\(\[]?\s*([{choice_class}])\b",
    ]
    for pat in prefixed_patterns:
        m = re.search(pat, t)
        if m:
            return m.group(1)

    # 3) Standalone option tokens, including LMMS-Eval-style "(A)", "A ", and "A.".
    m = re.search(rf"[\(\[]\s*([{choice_class}])\s*[\)\]]", t)
    if m:
        return m.group(1)
    m = re.search(rf"\b([{choice_class}])\s*\.", t)
    if m:
        return m.group(1)
    m = re.search(rf"\b([{choice_class}])\b", t)
    if m:
        return m.group(1)

    # 4) LMMS-Eval also falls back to option text matching for verbose generations.
    if option_text_by_letter:
        raw_lower = raw.lower()
        matches = []
        for letter in valid_choice_letters:
            option_text = _strip_choice_prefix(option_text_by_letter.get(letter, ""))
            if len(option_text.split()) < 2:
                continue
            idx = raw_lower.find(option_text.lower())
            if idx >= 0:
                matches.append((idx, letter))
        if matches:
            matches.sort(key=lambda item: item[0])
            return matches[0][1]
    return ""


def _extract_choice_letter_lmms_eval(text):
    if not text:
        return ""
    try:
        from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer

        return str(extract_mcq_answer(text, choices=["A", "B", "C", "D"]) or "").strip().upper()[:1]
    except Exception:
        raw = str(text or "").strip()
        if not raw:
            return ""
        patterns = [
            r"^\s*[\(\[]?\s*([A-Da-d])\s*[\)\].,:;!?\u3002\uff0c\uff1a\uff1b]?\s*$",
            r"\b(?:ANSWER|OPTION|CHOICE)\b\s*(?:IS)?\s*[:=\-]?\s*[\(\[]?\s*([A-Da-d])\b",
            r"\b(?:THE\s+ANSWER\s+IS|I\s+CHOOSE|I\s+PICK)\b\s*[:=\-]?\s*[\(\[]?\s*([A-Da-d])\b",
            r"[\(\[]\s*([A-Da-d])\s*[\)\]]",
            r"\b([A-Da-d])\s*\.",
            r"\b([A-Da-d])\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, raw, flags=re.IGNORECASE)
            if match:
                return match.group(1).upper()
        return ""


def extract_characters_regex(s, valid_choices=None, option_text_by_letter=None):
    return _extract_choice_letter_lmms_eval(s)


matrices = []

for i in VIDEO_TYPE:
    for j in CATEGORIES:
        for k in SUB_CATEGORIES:
            for l in TASK_CATEGORIES:
                matrices.append(f"{i}_{j}_{k}_{l}")


def videomme_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videomme score), value: metric value
    """
    pred = results[0]
    valid_choices, option_text_by_letter = _infer_choice_info(doc)
    pred_ans = extract_characters_regex(pred, valid_choices, option_text_by_letter)
    gt_ans = doc["answer"]

    category = doc["domain"]
    sub_category = doc["sub_category"]
    task_category = doc["task_type"]
    # score: 0/1 correctness for stderr calculation
    # videoID: for clustered stderr (questions from same video are correlated)
    score = 1.0 if pred_ans.lower() == gt_ans.lower() else 0.0
    data_dict = {
        "question_id": doc["question_id"],
        "duration": doc["duration"],
        "category": category,
        "sub_category": sub_category,
        "task_category": task_category,
        "pred_answer": pred_ans,
        "answer": gt_ans,
        "score": score,
        "videoID": doc["videoID"],
    }

    return {"videomme_perception_score": data_dict}


def videomme_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    category2score = {}

    for video_type in VIDEO_TYPE:
        for category in CATEGORIES:
            for sub_category in SUB_CATEGORIES:
                for task_category in TASK_CATEGORIES:
                    key = f"{video_type}_{category}_{sub_category}_{task_category}"
                    category2score[key] = {"correct": 0, "answered": 0}

    for result in results:
        video_type = result["duration"]
        category = result["category"]
        sub_category = result["sub_category"]
        task_category = result["task_category"]
        key = f"{video_type}_{category}_{sub_category}_{task_category}"
        category2score[key]["answered"] += 1
        category2score[key]["correct"] += result["pred_answer"] == result["answer"]

    for video_type in VIDEO_TYPE:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if video_type in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        eval_logger.info(f"Evaluation on video Type: {video_type}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    for category in CATEGORIES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if category in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        eval_logger.info(f"Evaluation on Categories: {category}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    for sub_cate in SUB_CATEGORIES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if sub_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        eval_logger.info(f"Evaluation on Video Sub Categories: {sub_cate}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    for task_cate in TASK_CATEGORIES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if task_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        eval_logger.info(f"Evaluation on Task Categories: {task_cate}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    total_correct = 0
    total_answered = 0
    for k, v in category2score.items():
        total_correct += v["correct"]
        total_answered += v["answered"]
    eval_logger.info(f"Overall Performance: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")
    return 100 * total_correct / total_answered if total_answered > 0 else 0
