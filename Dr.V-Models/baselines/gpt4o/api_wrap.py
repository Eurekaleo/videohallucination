import openai
import cv2
import numpy as np
import time
import base64


class OpenAIAPIWrapper:
    def __init__(self, model="gpt-4o", temperature=0.7, time_out=5):
        self.model = model
        self.temperature = temperature
        self.time_out = time_out

        openai.api_key = ""

    def request(self, usr_question, system_content=None, video_path=None):
        base64Frames, _ = self.process_video(video_path, seconds_per_frame=2)

        total_frames = len(base64Frames)
        if total_frames == 0:
            raise ValueError("No frames were extracted from the video.")

        indices = np.linspace(0, total_frames - 1, 4, dtype=int)
        sampled_base64Frames = [base64Frames[i] for i in indices]

        messages = [
            {"role": "system", "content": system_content or "Use the video to answer the provided question."},
            {"role": "user", "content": "These are the frames from the video:"},
        ]

        for frame in sampled_base64Frames:
            messages.append({
                "role": "user", 
                "content": f"data:image/jpeg;base64,{frame}"
            })

        messages.append({"role": "user", "content": usr_question})

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )

        resp = response.choices[0].message["content"]
        total_tokens = response.usage["total_tokens"]

        return resp, total_tokens

    def process_video(self, video_path, seconds_per_frame=2):
        base64Frames = []
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise FileNotFoundError(f"Unable to open video: {video_path}")

        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        frames_to_skip = max(1, int(fps * seconds_per_frame))

        for frame_idx in range(0, total_frames, frames_to_skip):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = video.read()
            if not success:
                continue
            _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 10])  # 设置 JPEG 质量为 30
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        
        video.release()

        if not base64Frames:
            raise ValueError("No frames were extracted from the video.")
        
        return base64Frames, None

    def get_completion(self, user_prompt=None, system_prompt=None, video_path=None, max_try=10):
        gpt_cv_nlp = ""
        total_tokens = 0

        while max_try > 0:
            try:
                gpt_cv_nlp, total_tokens = self.request(user_prompt, system_prompt, video_path)
                break
            except Exception as e:
                print(f"Encountered error: {e}")
                time.sleep(self.time_out)
                max_try -= 1

        if not gpt_cv_nlp:
            raise RuntimeError("Failed to get a valid response from OpenAI API after multiple retries.")
        
        return gpt_cv_nlp, total_tokens