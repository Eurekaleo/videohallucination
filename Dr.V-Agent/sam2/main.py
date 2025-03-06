import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import cv2

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

from sam2.sam2_video_predictor import SAM2VideoPredictor
from ultralytics import YOLO

class ObjectTracker:
    def __init__(self, model_name):
        if model_name == "Florence2+SAM2":
            # Initialize Florence2 model
            self.flo2 = AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-large-ft', trust_remote_code=True).eval()
            self.flo2_processor = AutoProcessor.from_pretrained('microsoft/Florence-2-large-ft', trust_remote_code=True)
            
            # Initialize SAM2 model
            self.sam_predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-large")
        elif model_name == "YOLO-World":
            # Initialize YOLO-World model
            self.yolo_world = YOLO("yolov8x-world.pt") 
        else:
            raise ValueError(f"Unsupported model name: {model_name}. Please choose either 'Florence2+SAM2' or 'YOLO-World'.")

    def show_mask(self, mask, ax, fig_save_path, obj_id=None, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
        plt.savefig(fig_save_path)

    def process_video_dir(self, video_dir):
        # scan all the JPEG frame names in this directory
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        return frame_names
    
    def florence2(self, video_dir, text_input=None, need_all_frame_bboxes=False, task_prompt='<CAPTION_TO_PHRASE_GROUNDING>'):
        """
        Calling the Microsoft Florence2 model
        """
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input
        
        frame_names = self.process_video_dir(video_dir)
        answer_list = []
        first_answer = None
        if need_all_frame_bboxes: # slow
            for i, frame in enumerate(frame_names):
                image = Image.open(os.path.join(video_dir, frame))
                inputs = self.flo2_processor(text=prompt, images=image, return_tensors="pt")

                generated_ids = self.flo2.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    early_stopping=False,
                    do_sample=False,
                    num_beams=3,
                )
                generated_text = self.flo2_processor.batch_decode(generated_ids,
                                                        skip_special_tokens=False)[0]
                parsed_answer = self.flo2_processor.post_process_generation(
                    generated_text,
                    task=task_prompt,
                    image_size=(image.width, image.height))
                
                if i == 0:
                    first_answer = parsed_answer['<CAPTION_TO_PHRASE_GROUNDING>']
                
                answer_list.append(parsed_answer['<CAPTION_TO_PHRASE_GROUNDING>'])
        else:
            image = Image.open(os.path.join(video_dir, frame_names[0]))
            inputs = self.flo2_processor(text=prompt, images=image, return_tensors="pt")

            generated_ids = self.flo2.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
            )
            generated_text = self.flo2_processor.batch_decode(generated_ids,
                                                    skip_special_tokens=False)[0]
            parsed_answer = self.flo2_processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=(image.width, image.height))
            
            first_answer = parsed_answer['<CAPTION_TO_PHRASE_GROUNDING>']

        return first_answer, answer_list

    def sam2(self, video_dir, bboxes, vis_frame_stride):

        frame_names = self.process_video_dir(video_dir)
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = self.sam_predictor.init_state(video_path=video_dir)
            ann_frame_idx = 0  # the frame index we interact with

            for i, box in enumerate(bboxes):
                ann_obj_id = i  # give a unique id to each object we interact with (it can be any integers)

                box = np.array(box, dtype=np.float32)
                _, out_obj_ids, out_mask_logits = self.sam_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    box=box,
                )

            # run propagation throughout the video and collect the results in a dict
            video_segments = {}  # video_segments contains the per-frame segmentation results
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam_predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

            # render the segmentation results every few frames
            plt.close("all")
            for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
                plt.figure(figsize=(6, 4))
                plt.title(f"frame {out_frame_idx}")
                plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
                fig_save_folder = 'fig_results_sam2'
                if not os.path.exists(fig_save_folder):
                    os.makedirs(fig_save_folder)
                for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                    self.show_mask(out_mask, plt.gca(), f'./{fig_save_folder}/frame{out_frame_idx}.jpg', obj_id=out_obj_id)

    def ObjectTrack_Florence2AndSam2(self, video_dir, object_names: list[str], vis_frame_stride=30, need_all_frame_bboxes=False):
        object_names = ", ".join(object_names)
        first_frame_res, res_list = self.florence2(video_dir=video_dir, text_input=object_names, need_all_frame_bboxes=need_all_frame_bboxes)
        self.sam2(video_dir=video_dir, bboxes=first_frame_res['bboxes'], vis_frame_stride=vis_frame_stride)

        box_list = []
        label_list = []
        for item in res_list:
            box_list.append(item['bboxes'])
            label_list.append(item['labels'])
        
        return {'bboxes': box_list, 'labels': label_list}


    def ObjectTrack_YOLOWorld(self, video_path, object_names: list[str], vis_frame_stride=30):
        self.yolo_world.set_classes(object_names)
        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        fig_save_folder = 'fig_results_yoloworld'
        if not os.path.exists(fig_save_folder):
            os.makedirs(fig_save_folder)
        box_list = []
        label_list = []
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()
            if success:
                # Run YOLO tracking on the frame, persisting tracks between frames
                results = self.yolo_world.track(frame, persist=True)
                
                # Visualize the results on the frame
                annotated_frame = results[0].plot()
                boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
                cls = results[0].boxes.cls.cpu().numpy().tolist()
                box_list.append(boxes)
                labels = []
                if len(cls) != 0:
                    for item in cls:
                        labels.append(results[0].names[int(item)])
                label_list.append(labels)

                # Save the annotated frame as an image
                if frame_count % vis_frame_stride == 0:
                    cv2.imwrite(f"./{fig_save_folder}/frame_{frame_count}.jpg", annotated_frame)
                
                frame_count += 1

                # Display the annotated frame
                # cv2.imshow("YOLO11 Tracking", annotated_frame)

                # # Break the loop if 'q' is pressed
                # if cv2.waitKey(1) & 0xFF == ord("q"):
                #     break
            else:
                # Break the loop if the end of the video is reached
                break

        # Release the video capture object and close the display window
        cap.release()
        cv2.destroyAllWindows()

        return {'bboxes': box_list, 'labels': label_list}


# Usage
model = "Florence2+SAM2"
tracker = ObjectTracker(model)
video_dir = "./notebooks/videos/bedroom"
object_names = ["boy", "girl"]
need_all_frame_bboxes = False # if True, slow
results = tracker.ObjectTrack_Florence2AndSam2(video_dir, object_names, need_all_frame_bboxes)
print(results['bboxes'])
print(results['labels'])

# --------------------------------------------------------------------------------------------

model = "YOLO-World"
tracker = ObjectTracker(model)
video_path = "./notebooks/videos/bedroom.mp4"
object_names = ["boy", "girl"]
results = tracker.ObjectTrack_YOLOWorld(video_path, object_names)
print(results['bboxes'])
print(results['labels'])
