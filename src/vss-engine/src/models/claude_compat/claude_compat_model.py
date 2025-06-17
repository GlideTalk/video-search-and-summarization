import os
import json
from langchain_aws.chat_models import ChatBedrock
from models.openai_compat.openai_compat_model import tensor_to_base64_jpeg
from via_logger import TimeMeasure, logger


class ClaudeCompatModel:
    """AWS Bedrock Claude model accessed via langchain-aws."""

    def __init__(self):
        self._client = ChatBedrock(
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        self._model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"

    @staticmethod
    def get_model_info():
        return "claude-compat", "external", "claude"

    def generate(self, prompt, video_embeds, video_frames_times, generation_config=None, chunk=None):
        responses = []
        generation_config = generation_config or {}
        
        for idx, times in enumerate(video_frames_times):
            # Build content with text and images for Bedrock Claude
            content = [{"type": "text", "text": prompt}]
            for j in range(len(times)):
                image_b64 = tensor_to_base64_jpeg(video_embeds[idx], j)
                content.append({
                    "type": "image", 
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg", 
                        "data": image_b64
                    }
                })
            
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": generation_config.get("max_tokens", 1000),
                "messages": [{"role": "user", "content": content}],
                "temperature": generation_config.get("temperature", 0.7)
            })
            
            with TimeMeasure("Claude model inference"):
                try:
                    response = self._client.client.invoke_model(
                        modelId=self._model_id,
                        body=body,
                        contentType="application/json"
                    )
                    
                    response_body = json.loads(response['body'].read().decode('utf-8'))
                    text = response_body['content'][0]['text']
                    
                    logger.info(f"Claude request: {prompt[:100]}...")
                    logger.info(f"Claude response: {text[:100]}...")
                except Exception as e:
                    logger.error(f"Claude Bedrock API error: {e}")
                    text = f"Error: {str(e)}"
                    
            responses.append(text)
            
        return responses, [{"input_tokens": 0, "output_tokens": 0}] * len(responses)
