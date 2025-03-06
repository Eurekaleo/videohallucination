import os
import re
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

        question_with_instruction = f"{question}\nAnswer with 'yes' or 'no'."

        predict = model.generate(
            instruction=question_with_instruction,
            video_path=video_path
        )

        predict = predict.strip().lower()
        if "yes" in predict:
            predict = "yes"
        elif "no" in predict:
            predict = "no"
        else:
            predict = "unknown"  

        qa_dct["predict"] = predict

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    with open(output_path, "w") as jp:
        json.dump(paired_qas, jp, indent=4)

    scores = cal_score(paired_qas)
    return scores


def cal_score(results):
    acc = 0
    unmatched_cases = []

    for result in results:
        hit = 0

        answer = result["answer"].lower() 
        predict = result["predict"].lower() 

        if predict == answer:
            hit = 1
        else:
            unmatched_cases.append({
                "predicted": predict,
                "expected": answer,
            })

        acc += hit

    scores = {
        "accuracy": acc / len(results),
        "unmatched_cases": unmatched_cases 
    }
    return scores