import os, sys
import json
import argparse
sys.path.append(os.getcwd())

video_path = "/home/mluo/video.hallucination/run/VideoHallucer/videohallucer_datasets/object/videos/star/0BZAD.mp4"

instruction = "Is there a dog in the video? Response with 'yes' or 'no'."
# instruction = "Describe the Video: "

configs = json.load(open("./config.json"))

DATA_DIR = configs['DATA_DIR']
CKPT_DIR = configs['CKPT_DIR']

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str,
                    default="", 
                    choices=["VideoChatGPT", "VideoChat2", "VideoLLaVA", "LLaMA-VID", "VideoLaVIT", "PLLaVA", "LLaVA-NeXT-Video",
                             "Gemini-1.5-pro", "GPT4O", "GPT4V"])
args = parser.parse_args()
TESTING_MODEL=args.model_name


def load_model(TESTING_MODEL):
    if TESTING_MODEL == 'VideoChatGPT':
        from videochatgpt_modeling import VideoChatGPT
        ckpt_path = f"{CKPT_DIR}/Video-ChatGPT-7B"
        model = VideoChatGPT({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "VideoChat2":
        from videochat_modeling import VideoChat
        ckpt_path = f"{CKPT_DIR}/VideoChat2"
        model = VideoChat({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "VideoLLaVA":
        from videollava_modeling import VideoLLaVA
        ckpt_path = f"{CKPT_DIR}/VideoLLaVA/Video-LLaVA-7B"
        model = VideoLLaVA({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "LLaMA-VID":
        from llamavid_modeling import LLaMAVID
        ckpt_path = f"{CKPT_DIR}/LLaMA-VID-7B"
        model = LLaMAVID({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "VideoLaVIT":
        from videolavit_modeling import VideoLaVIT
        ckpt_path = f"{CKPT_DIR}/Video-LaVIT-v1"
        model = VideoLaVIT({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "PLLaVA":
        from pllava_modeling import PLLaVA
        ckpt_path = f"{CKPT_DIR}/PLLaVA/pllava-7b"
        model = PLLaVA({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "LLaVA-NeXT-Video":
        from llavanext_modeling import LLaVANeXT
        ckpt_path = f"{CKPT_DIR}/LLaVA-NeXT-Video/LLaVA-NeXT-Video-7B-DPO"
        model = LLaVANeXT({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "Gemini-1.5-pro":
        from gemini_modeling import Gemini
        model = Gemini({"model_path": None, "device": 0})
    elif TESTING_MODEL == "GPT4O":
        from gpt4o_modeling import GPT4O
        model = GPT4O({"model_path": None, "device": 0})

    return model

model = load_model(TESTING_MODEL)


pred = model.generate(
    instruction=instruction,
    video_path=video_path,
)
print(f'Instruction:\t{instruction}')
print(f'Answer:\t{pred}')
print('-'*20)
