import os
import json
import time
from typing import List, Dict, Any, Optional
from langchain_aws.chat_models import ChatBedrock
from langchain_core.messages import HumanMessage
from models.openai_compat.openai_compat_model import tensor_to_base64_jpeg
from via_logger import TimeMeasure, logger


class ClaudeCompatModel:
    """Optimized AWS Bedrock Claude model with enterprise-grade error handling and performance optimizations."""

    def __init__(self):
        # Validate required environment variables
        self._validate_aws_credentials()
        
        # Configuration
        self._region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self._model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"
        
        # Initialize optimized client with retry configuration
        self._client = self._create_bedrock_client()
        
        # Performance optimization settings
        self._max_retries = int(os.getenv("BEDROCK_MAX_RETRIES", "3"))
        self._base_delay = float(os.getenv("BEDROCK_BASE_DELAY", "1.0"))
        self._max_delay = float(os.getenv("BEDROCK_MAX_DELAY", "60.0"))
        
        logger.info(f"Claude Bedrock client initialized for region: {self._region}")
    
    def _validate_aws_credentials(self) -> None:
        """Validate required AWS credentials are present."""
        required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required AWS credentials: {', '.join(missing_vars)}")
    
    def _create_bedrock_client(self) -> ChatBedrock:
        """Create optimized Bedrock client with proper configuration."""
        return ChatBedrock(
            model_id=self._model_id,
            region_name=self._region,
            credentials_profile_name=None,  # Use environment variables
            model_kwargs={
                "max_tokens": 4096,
                "temperature": 0.1,  # Lower temperature for more consistent results
                "top_p": 0.9,
                "anthropic_version": "bedrock-2023-05-31"
            }
        )

    @staticmethod
    def get_model_info():
        return "claude-compat", "external", "claude"
    
    def _exponential_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter."""
        delay = min(self._base_delay * (2 ** attempt), self._max_delay)
        # Add jitter to prevent thundering herd
        jitter = delay * 0.1 * (0.5 - os.urandom(1)[0] / 255.0)
        return delay + jitter
    
    def _execute_with_retry(self, func, *args, **kwargs) -> Any:
        """Execute function with exponential backoff retry logic."""
        last_exception = None
        
        for attempt in range(self._max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()
                
                # Don't retry on authentication or validation errors
                if any(keyword in error_msg for keyword in ['unauthorized', 'forbidden', 'invalid', 'bad request']):
                    logger.error(f"Non-retryable error: {e}")
                    raise e
                
                if attempt < self._max_retries:
                    delay = self._exponential_backoff(attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {self._max_retries + 1} attempts failed. Last error: {e}")
        
        raise last_exception
    
    def _create_message_content(self, prompt: str, video_embeds, frame_indices: List[int]) -> List[Dict[str, Any]]:
        """Create optimized message content with batch image processing."""
        content = [{"type": "text", "text": prompt}]
        
        # Batch process images for better performance
        for frame_idx in frame_indices:
            try:
                image_b64 = tensor_to_base64_jpeg(video_embeds, frame_idx)
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_b64
                    }
                })
            except Exception as e:
                logger.warning(f"Failed to process frame {frame_idx}: {e}")
                continue
        
        return content

    def generate(self, prompt: str, video_embeds, video_frames_times: List[List], generation_config: Optional[Dict] = None, chunk=None) -> tuple[List[str], List[Dict]]:
        """Generate responses for video frames with optimized batch processing and error handling."""
        responses = []
        token_usage = []
        generation_config = generation_config or {}
        
        # Extract generation parameters with cost-performance balance
        max_tokens = generation_config.get("max_tokens", 2048)  # Optimized for cost vs capability
        temperature = generation_config.get("temperature", 0.1)  # Lower for more consistent results
        top_p = generation_config.get("top_p", 0.9)
        
        logger.info(f"Processing {len(video_frames_times)} video segments with Claude Sonnet")
        
        for idx, times in enumerate(video_frames_times):
            try:
                with TimeMeasure(f"Claude inference for segment {idx + 1}/{len(video_frames_times)}"):
                    # Create optimized message content
                    frame_indices = list(range(len(times)))
                    content = self._create_message_content(prompt, video_embeds[idx], frame_indices)
                    
                    # Create human message with multimodal content
                    message = HumanMessage(content=content)
                    
                    # Configure model for this request
                    configured_client = self._client.with_config(
                        configurable={
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                            "top_p": top_p
                        }
                    )
                    
                    # Execute with retry logic
                    def _invoke_model():
                        response = configured_client.invoke([message])
                        return response.content
                    
                    text = self._execute_with_retry(_invoke_model)
                    
                    # Extract token usage if available (for cost tracking)
                    usage_info = {
                        "input_tokens": getattr(configured_client, 'input_tokens', 0),
                        "output_tokens": getattr(configured_client, 'output_tokens', 0),
                        "model_id": self._model_id
                    }
                    
                    responses.append(text)
                    token_usage.append(usage_info)
                    
                    # Log for monitoring and debugging
                    logger.info(f"Segment {idx + 1}: Generated {len(text)} chars with {len(frame_indices)} frames")
                    
            except Exception as e:
                logger.error(f"Failed to process video segment {idx}: {e}")
                # Provide graceful degradation
                error_response = f"Unable to process video segment {idx + 1}: {type(e).__name__}"
                responses.append(error_response)
                token_usage.append({"input_tokens": 0, "output_tokens": 0, "model_id": self._model_id})
        
        logger.info(f"Claude processing completed: {len(responses)} responses generated")
        return responses, token_usage
