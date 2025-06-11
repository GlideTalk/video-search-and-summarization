import os
import requests
from models.openai_compat.openai_compat_model import tensor_to_base64_jpeg
from via_logger import TimeMeasure, logger


class GeminiCompatModel:
    """Minimal wrapper for a Gemini-compatible vision language model."""

    def __init__(self):
        self._endpoint = os.getenv("GEMINI_ENDPOINT", "")
        self._api_key = os.getenv("GEMINI_API_KEY", "")

    @staticmethod
    def get_model_info():
        return "gemini-compat", "external", "gemini"

    def generate(self, prompt, video_embeds, video_frames_times, generation_config=None, chunk=None):
        responses = []
        generation_config = generation_config or {}
        for idx, times in enumerate(video_frames_times):
            images = [tensor_to_base64_jpeg(video_embeds[idx], j) for j in range(len(times))]
            payload = {"prompt": prompt, "images": images, **generation_config}
            headers = {"Authorization": f"Bearer {self._api_key}"}
            with TimeMeasure("Gemini model inference"):
                resp = requests.post(self._endpoint, json=payload, headers=headers, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                text = data.get("text", data.get("output", ""))
            responses.append(text)
        return responses, [{"input_tokens": 0, "output_tokens": 0}] * len(responses)
