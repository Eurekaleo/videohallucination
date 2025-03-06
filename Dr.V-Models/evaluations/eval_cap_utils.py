import os
import json
import random        
import numpy as np
import torch
from tqdm import tqdm

def setup_seed(seed=428):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def evaluate(
    model,
    model_name,
    qa_path,
    qa_type,
    video_dir_path,
    output_dir_path,
    seed=42,
):
    setup_seed(seed)

    output_path = os.path.join(output_dir_path, f"{qa_type}_{model_name}.json")
    if os.path.exists(output_path):
        paired_qas = json.load(open(output_path))
        scores = cal_score(paired_qas)
        return scores

    paired_qas = json.load(open(qa_path))
    print(f"start eval | model: {model_name} | qa_type: {qa_type}")

    for qa_dct in tqdm(paired_qas):
        question = qa_dct["question"]
        video_path = os.path.join(video_dir_path, qa_dct["video"])
        options = qa_dct["option"]

        question_with_options = f"{question}\nOptions:\n"
        for option_key, option_value in options.items():
            question_with_options += f"{option_key}: {option_value}\n"  # 直接使用 option_value
        question_with_options += "Select the best option and generate a video caption based on the selected option."

        predict = model.generate(
            instruction=question_with_options,
            video_path=video_path
        )

        qa_dct["predict"] = predict.strip()

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    with open(output_path, "w") as jp:
        json.dump(paired_qas, jp, indent=4)

    scores = {
        "message": "Model outputs saved. Please evaluate the results manually or with a separate script."
    }
    return scores


def cal_score(results):
    return {
        "message": "Model outputs saved. Please evaluate the results manually or with a separate script."
    }