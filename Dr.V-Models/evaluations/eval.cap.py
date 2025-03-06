import argparse
import json
import sys
import os
import numpy as np
import random
import uuid
from collections import defaultdict
from typing import Callable
from tqdm import tqdm

from eval_cap_utils import evaluate

sys.path.append(os.getcwd())

configs = json.load(open("./config.json"))

DATA_DIR = configs['DATA_DIR']
CKPT_DIR = configs['CKPT_DIR']

os.environ['DECORD_EOF_RETRY_MAX'] = '20480'


def load_model(TESTING_MODEL):
    if TESTING_MODEL == 'VideoChatGPT':
        from videochatgpt_modeling import VideoChatGPT
        ckpt_path = f"{CKPT_DIR}/Video-ChatGPT-7B"
        model = VideoChatGPT({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "Video-LLaMA-2":
        from videollama_modeling import VideoLLaMA
        ckpt_path = f"{CKPT_DIR}/Video-LLaMA-2-7B-Finetuned"
        model = VideoLLaMA({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "Video-LLaMA-2-13B":
        from videollama_modeling import VideoLLaMA
        ckpt_path = f"{CKPT_DIR}/Video-LLaMA-2-13B-Finetuned"
        model = VideoLLaMA({"model_path": ckpt_path, "device": 0})
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
    elif TESTING_MODEL == "MiniGPT4-Video":
        from minigpt4video_modeling import MiniGPT4Video
        ckpt_path = f"{CKPT_DIR}/MiniGPT4-Video/checkpoints"
        model = MiniGPT4Video({"model_path": ckpt_path, "device": 0})
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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="", 
                        choices=["VideoChatGPT", "Valley2", "Video-LLaMA-2", "VideoChat2", "VideoLLaVA", "LLaMA-VID", "VideoLaVIT", "MiniGPT4-Video", "PLLaVA", "LLaVA-NeXT-Video", "ShareGPT4Video",
                                 "Gemini-1.5-pro", "GPT4O",
                                 "LLaVA", "GPT4V", 
                                 "Video-LLaMA-2-13B", "LLaMA-VID-13B", "PLLaVA-13B", 
                                 "PLLaVA-34B", "LLaVA-NeXT-Video-34B"])

    parser.add_argument(
        "--output_dir_path", type=str, default="results",
    )
    parser.add_argument("--device", type=int, default=0)


    # new.1
    parser.add_argument(
        "--eval_object",
        action="store_true",
        default=False,
        help="Whether to evaluate on object hallucination",
    )
    
    # new.2
    parser.add_argument(
        "--eval_color",
        action="store_true",
        default=False,
        help="Whether to evaluate on color hallucination",
    )
    
    # new.3
    parser.add_argument(
        "--eval_number",
        action="store_true",
        default=False,
        help="Whether to evaluate on number hallucination",
    )
    
    # new.4
    parser.add_argument(
        "--eval_location",
        action="store_true",
        default=False,
        help="Whether to evaluate on location hallucination",
    )
    
    # new.5
    parser.add_argument(
        "--eval_static_relation",
        action="store_true",
        default=False,
        help="Whether to evaluate on static relation hallucination",
    )
    
    # new.6
    parser.add_argument(
        "--eval_ocr",
        action="store_true",
        default=False,
        help="Whether to evaluate on ocr hallucination",
    )
    
    
    # new.7
    parser.add_argument(
        "--eval_action",
        action="store_true",
        default=False,
        help="Whether to evaluate on action hallucination",
    )
    
    # new.8
    parser.add_argument(
        "--eval_dynamic_attribute",
        action="store_true",
        default=False,
        help="Whether to evaluate on dynamic attribute hallucination",
    )
    
    # new.9
    parser.add_argument(
        "--eval_dynamic_relation",
        action="store_true",
        default=False,
        help="Whether to evaluate on dynamic relation hallucination",
    )
    
    # new.10
    parser.add_argument(
        "--eval_sequence",
        action="store_true",
        default=False,
        help="Whether to evaluate on sequence hallucination",
    )
    
    # new.11
    parser.add_argument(
        "--eval_factual_prediction",
        action="store_true",
        default=False,
        help="Whether to evaluate on fatctual prediction hallucination",
    )
    
    # new.12
    parser.add_argument(
        "--eval_counterfactual_prediction",
        action="store_true",
        default=False,
        help="Whether to evaluate on counter factual prediction hallucination",
    )
    
    # new.13
    parser.add_argument(
        "--eval_context_based_explanation",
        action="store_true",
        default=False,
        help="Whether to evaluate on context based explanation hallucination",
    )
    
    
    # new.14
    parser.add_argument(
        "--eval_knowledge_based_explanation",
        action="store_true",
        default=False,
        help="Whether to evaluate on knowledge based explanation hallucination",
    )


    ## new.1
    parser.add_argument(
        "--object_path",
        type=str,
        default="object/object_c1.json",
    )
    parser.add_argument(
        "--object_video_dir_path",
        type=str,
        default="object/videos",
    )
    
    # new.2
    parser.add_argument(
        "--color_path",
        type=str,
        default="color/color_c1.json",
    )
    parser.add_argument(
        "--color_video_dir_path",
        type=str,
        default="color/videos",
    )
    
    # new.3
    parser.add_argument(
        "--number_path",
        type=str,
        default="number/number_c1.json",
    )
    parser.add_argument(
        "--number_video_dir_path",
        type=str,
        default="number/videos",
    )
 
    # new.4
    parser.add_argument(
        "--location_path",
        type=str,
        default="location/location_c1.json",
    )
    parser.add_argument(
        "--location_video_dir_path",
        type=str,
        default="location/videos",
    )  
        
    # new.5
    parser.add_argument(
        "--static_relation_path",
        type=str,
        default="static_relation/static_relation_c1.json",
    )
    parser.add_argument(
        "--static_relation_video_dir_path",
        type=str,
        default="static_relation/videos",
    )

    # new.6
    parser.add_argument(
        "--ocr_path",
        type=str,
        default="ocr/ocr_c1.json",
    )
    parser.add_argument(
        "--ocr_video_dir_path",
        type=str,
        default="ocr/videos",
    )  


    # new.7
    parser.add_argument(
        "--action_path",
        type=str,
        default="action/action_c1.json",
    )
    parser.add_argument(
        "--action_video_dir_path",
        type=str,
        default="action/videos",
    )  
 
    # new.8
    parser.add_argument(
        "--dynamic_attribute_path",
        type=str,
        default="dynamic_attribute/dynamic_attribute_c1.json",
    )
    parser.add_argument(
        "--dynamic_attribute_video_dir_path",
        type=str,
        default="dynamic_attribute/videos",
    )   

    # new.9
    parser.add_argument(
        "--dynamic_relation_path",
        type=str,
        default="dynamic_relation/dynamic_relation_c1.json",
    )
    parser.add_argument(
        "--dynamic_relation_video_dir_path",
        type=str,
        default="dynamic_relation/videos",
    )  

    # new.10
    parser.add_argument(
        "--sequence_path",
        type=str,
        default="sequence/sequence_c1.json",
    )
    parser.add_argument(
        "--sequence_video_dir_path",
        type=str,
        default="sequence/videos",
    )  

    # new.11
    parser.add_argument(
        "--factual_prediction_path",
        type=str,
        default="factual_prediction/factual_prediction_c1.json",
    )
    parser.add_argument(
        "--factual_prediction_video_dir_path",
        type=str,
        default="factual_prediction/videos",
    )  
    
    # new.12
    parser.add_argument(
        "--counterfactual_prediction_path",
        type=str,
        default="counterfactual_prediction/counterfactual_prediction_c1.json",
    )
    parser.add_argument(
        "--counterfactual_prediction_video_dir_path",
        type=str,
        default="counterfactual_prediction/videos",
    )  
    
    
    # new.13
    parser.add_argument(
        "--context_based_explanation_path",
        type=str,
        default="context_based_explanation/context_based_explanation_c1.json",
    )
    parser.add_argument(
        "--context_based_explanation_video_dir_path",
        type=str,
        default="context_based_explanation/videos",
    )      
     
    
    # new.14
    parser.add_argument(
        "--knowledge_based_explanation_path",
        type=str,
        default="knowledge_based_explanation/knowledge_based_explanation_c1.json",
    )
    parser.add_argument(
        "--knowledge_based_explanation_video_dir_path",
        type=str,
        default="knowledge_based_explanation/videos",
    )    
    
    args = parser.parse_args()
    
    model = load_model(args.model_name)
    final_result = {}
    
    if args.eval_object:
        object_scores = evaluate(
            model=model,
            model_name=args.model_name,
            qa_path=os.path.join(DATA_DIR, args.object_path),
            qa_type='object',
            video_dir_path=os.path.join(DATA_DIR, args.object_video_dir_path),
            output_dir_path=args.output_dir_path   
        )
        final_result["object"] = object_scores

    if args.eval_color:
        color_scores = evaluate(
            model=model,
            model_name=args.model_name,
            qa_path=os.path.join(DATA_DIR, args.color_path),
            qa_type='color',
            video_dir_path=os.path.join(DATA_DIR, args.color_video_dir_path),
            output_dir_path=args.output_dir_path
        )
        final_result["color"] = color_scores

    if args.eval_number:
        number_scores = evaluate(
            model=model,
            model_name=args.model_name,
            qa_path=os.path.join(DATA_DIR, args.number_path),
            qa_type='number',
            video_dir_path=os.path.join(DATA_DIR, args.number_video_dir_path),
            output_dir_path=args.output_dir_path
        )
        final_result["number"] = number_scores

    if args.eval_location:
        location_scores = evaluate(
            model=model,
            model_name=args.model_name,
            qa_path=os.path.join(DATA_DIR, args.location_path),
            qa_type='location',
            video_dir_path=os.path.join(DATA_DIR, args.location_video_dir_path),
            output_dir_path=args.output_dir_path
        )
        final_result["location"] = location_scores

    if args.eval_static_relation:
        static_relation_scores = evaluate(
            model=model,
            model_name=args.model_name,
            qa_path=os.path.join(DATA_DIR, args.static_relation_path),
            qa_type='static_relation',
            video_dir_path=os.path.join(DATA_DIR, args.static_relation_video_dir_path),
            output_dir_path=args.output_dir_path
        )
        final_result["static_relation"] = static_relation_scores

    if args.eval_ocr:
        ocr_scores = evaluate(
            model=model,
            model_name=args.model_name,
            qa_path=os.path.join(DATA_DIR, args.ocr_path),
            qa_type='ocr',
            video_dir_path=os.path.join(DATA_DIR, args.ocr_video_dir_path),
            output_dir_path=args.output_dir_path
        )
        final_result["ocr"] = ocr_scores

    if args.eval_action:
        action_scores = evaluate(
            model=model,
            model_name=args.model_name,
            qa_path=os.path.join(DATA_DIR, args.action_path),
            qa_type='action',
            video_dir_path=os.path.join(DATA_DIR, args.action_video_dir_path),
            output_dir_path=args.output_dir_path
        )
        final_result["action"] = action_scores

    if args.eval_dynamic_attribute:
        dynamic_attribute_scores = evaluate(
            model=model,
            model_name=args.model_name,
            qa_path=os.path.join(DATA_DIR, args.dynamic_attribute_path),
            qa_type='dynamic_attribute',
            video_dir_path=os.path.join(DATA_DIR, args.dynamic_attribute_video_dir_path),
            output_dir_path=args.output_dir_path
        )
        final_result["dynamic_attribute"] = dynamic_attribute_scores

    if args.eval_dynamic_relation:
        dynamic_relation_scores = evaluate(
            model=model,
            model_name=args.model_name,
            qa_path=os.path.join(DATA_DIR, args.dynamic_relation_path),
            qa_type='dynamic_relation',
            video_dir_path=os.path.join(DATA_DIR, args.dynamic_relation_video_dir_path),
            output_dir_path=args.output_dir_path
        )
        final_result["dynamic_relation"] = dynamic_relation_scores

    if args.eval_sequence:
        sequence_scores = evaluate(
            model=model,
            model_name=args.model_name,
            qa_path=os.path.join(DATA_DIR, args.sequence_path),
            qa_type='sequence',
            video_dir_path=os.path.join(DATA_DIR, args.sequence_video_dir_path),
            output_dir_path=args.output_dir_path
        )
        final_result["sequence"] = sequence_scores

    if args.eval_factual_prediction:
        factual_prediction_scores = evaluate(
            model=model,
            model_name=args.model_name,
            qa_path=os.path.join(DATA_DIR, args.factual_prediction_path),
            qa_type='factual_prediction',
            video_dir_path=os.path.join(DATA_DIR, args.factual_prediction_video_dir_path),
            output_dir_path=args.output_dir_path
        )
        final_result["factual_prediction"] = factual_prediction_scores

    if args.eval_counterfactual_prediction:
        counterfactual_prediction_scores = evaluate(
            model=model,
            model_name=args.model_name,
            qa_path=os.path.join(DATA_DIR, args.counterfactual_prediction_path),
            qa_type='counterfactual_prediction',
            video_dir_path=os.path.join(DATA_DIR, args.counterfactual_prediction_video_dir_path),
            output_dir_path=args.output_dir_path
        )
        final_result["counterfactual_prediction"] = counterfactual_prediction_scores

    if args.eval_context_based_explanation:
        context_based_explanation_scores = evaluate(
            model=model,
            model_name=args.model_name,
            qa_path=os.path.join(DATA_DIR, args.context_based_explanation_path),
            qa_type='context_based_explanation',
            video_dir_path=os.path.join(DATA_DIR, args.context_based_explanation_video_dir_path),
            output_dir_path=args.output_dir_path
        )
        final_result["context_based_explanation"] = context_based_explanation_scores

    if args.eval_knowledge_based_explanation:
        knowledge_based_explanation_scores = evaluate(
            model=model,
            model_name=args.model_name,
            qa_path=os.path.join(DATA_DIR, args.knowledge_based_explanation_path),
            qa_type='knowledge_based_explanation',
            video_dir_path=os.path.join(DATA_DIR, args.knowledge_based_explanation_video_dir_path),
            output_dir_path=args.output_dir_path
        )
        final_result["knowledge_based_explanation"] = knowledge_based_explanation_scores
    
    
    final_acc = 0
    eval_type = ""
    for halluc_type, result in final_result.items():
        eval_type += halluc_type + "_"
        if "accuracy" in result: 
            final_acc += result["accuracy"]
    if len(final_result.keys()) != 0:
        final_acc = final_acc / len(final_result.keys())
        final_result["all"] = {
            "accuracy": final_acc,
        }

        eval_result_path = os.path.join(args.output_dir_path, f"{eval_type}{args.model_name}_caption_evaluation_results.json")
        with open(eval_result_path, "w") as jp:
            json.dump(final_result, jp, indent=4)
        print("="*20)
        print("Final Accuracy: ", final_acc)
        print("="*20)

if __name__ == "__main__":
    main()