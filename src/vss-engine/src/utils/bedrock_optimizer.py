# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AWS Bedrock Optimizer for Video Search and Summarization.

Specialized optimizer for video processing workloads with Claude models.
Includes video-specific optimizations like frame batching and temporal caching.
"""

import os
import time
import hashlib
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict

from via_logger import logger


@dataclass
class VideoProcessingConfig:
    """Optimized configuration for video processing with Claude."""
    model_id: str = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    region: str = "us-east-1"
    max_tokens: int = 4096
    temperature: float = 0.1  # Lower for consistent video analysis
    top_p: float = 0.9
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    
    # Video-specific optimizations
    max_frames_per_request: int = 10  # Optimal frame batch size
    frame_sampling_strategy: str = "uniform"  # uniform, keyframe, adaptive
    enable_temporal_caching: bool = True
    temporal_cache_window: int = 300  # 5 minutes
    
    # Cost optimization
    prefer_batch_processing: bool = True
    dynamic_token_limits: bool = True


class VideoBedrockOptimizer:
    """Specialized optimizer for video processing with AWS Bedrock Claude."""
    
    def __init__(self):
        self.config = VideoProcessingConfig()
        self._temporal_cache: Dict[str, Tuple[Any, float]] = {}
        self._frame_processing_stats = defaultdict(list)
        self._cost_tracking = []
        
        # Performance optimization settings from environment
        self.config.max_frames_per_request = int(os.getenv("BEDROCK_MAX_FRAMES", "10"))
        self.config.max_retries = int(os.getenv("BEDROCK_MAX_RETRIES", "3"))
        self.config.base_delay = float(os.getenv("BEDROCK_BASE_DELAY", "1.0"))
        
        logger.info("Video Bedrock Optimizer initialized with optimized settings")
    
    def optimize_frame_batch(self, total_frames: int, video_duration: float) -> Tuple[List[int], str]:
        """
        Optimize frame selection for video processing.
        
        Returns:
            Tuple of (frame_indices, strategy_used)
        """
        max_frames = self.config.max_frames_per_request
        
        if total_frames <= max_frames:
            return list(range(total_frames)), "all_frames"
        
        # Intelligent frame sampling based on video characteristics
        if video_duration > 300:  # Long videos (5+ minutes)
            # Use keyframe-like sampling with emphasis on transitions
            indices = self._adaptive_sampling(total_frames, max_frames)
            return indices, "adaptive"
        elif video_duration > 60:  # Medium videos (1-5 minutes)
            # Uniform sampling with slight bias toward beginning and end
            indices = self._weighted_uniform_sampling(total_frames, max_frames)
            return indices, "weighted_uniform"
        else:  # Short videos (<1 minute)
            # Simple uniform sampling
            step = total_frames // max_frames
            indices = list(range(0, total_frames, step))[:max_frames]
            return indices, "uniform"
    
    def _adaptive_sampling(self, total_frames: int, target_count: int) -> List[int]:
        """Adaptive sampling that emphasizes scene changes and key moments."""
        # Allocate frames with bias toward beginning (30%), middle (40%), end (30%)
        begin_count = int(target_count * 0.3)
        middle_count = int(target_count * 0.4)
        end_count = target_count - begin_count - middle_count
        
        begin_frames = list(range(0, total_frames // 4, max(1, (total_frames // 4) // max(1, begin_count))))[:begin_count]
        
        middle_start = total_frames // 4
        middle_end = 3 * total_frames // 4
        middle_step = max(1, (middle_end - middle_start) // max(1, middle_count))
        middle_frames = list(range(middle_start, middle_end, middle_step))[:middle_count]
        
        end_start = 3 * total_frames // 4
        end_step = max(1, (total_frames - end_start) // max(1, end_count))
        end_frames = list(range(end_start, total_frames, end_step))[:end_count]
        
        return sorted(begin_frames + middle_frames + end_frames)
    
    def _weighted_uniform_sampling(self, total_frames: int, target_count: int) -> List[int]:
        """Uniform sampling with slight emphasis on beginning and end."""
        step = total_frames / target_count
        indices = []
        
        for i in range(target_count):
            # Add slight randomization to avoid always hitting the same frame types
            base_idx = int(i * step)
            # Add small offset for better coverage
            offset = int((i % 3 - 1) * step * 0.1)  # -10% to +10% offset
            idx = max(0, min(total_frames - 1, base_idx + offset))
            indices.append(idx)
        
        return sorted(list(set(indices)))  # Remove duplicates and sort
    
    def generate_temporal_cache_key(self, video_id: str, start_time: float, end_time: float, prompt: str) -> str:
        """Generate cache key for temporal video segments."""
        cache_data = {
            "video_id": video_id,
            "start_time": round(start_time, 2),
            "end_time": round(end_time, 2),
            "prompt": prompt,
            "model": self.config.model_id
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_string.encode()).hexdigest()
    
    def get_cached_video_analysis(self, cache_key: str) -> Optional[Any]:
        """Get cached video analysis if within temporal window."""
        if cache_key in self._temporal_cache:
            result, timestamp = self._temporal_cache[cache_key]
            if time.time() - timestamp < self.config.temporal_cache_window:
                logger.debug(f"Temporal cache hit for video segment: {cache_key[:16]}...")
                return result
            else:
                del self._temporal_cache[cache_key]
        return None
    
    def cache_video_analysis(self, cache_key: str, result: Any) -> None:
        """Cache video analysis result."""
        self._temporal_cache[cache_key] = (result, time.time())
        logger.debug(f"Cached video analysis: {cache_key[:16]}...")
    
    def calculate_optimal_token_limit(self, frame_count: int, prompt_length: int) -> int:
        """Calculate optimal token limit based on input complexity."""
        if not self.config.dynamic_token_limits:
            return self.config.max_tokens
        
        # Base tokens for response
        base_tokens = 1000
        
        # Additional tokens per frame (empirically determined)
        frame_tokens = frame_count * 150
        
        # Additional tokens for complex prompts
        prompt_tokens = min(prompt_length // 4, 500)
        
        # Total with safety margin
        total_tokens = base_tokens + frame_tokens + prompt_tokens
        
        # Clamp to reasonable limits
        return min(max(total_tokens, 1000), 4096)
    
    def track_video_processing_performance(self, 
                                         frame_count: int, 
                                         processing_time: float, 
                                         token_usage: Dict[str, int],
                                         success: bool) -> None:
        """Track performance metrics for video processing optimization."""
        
        stats = {
            "timestamp": time.time(),
            "frame_count": frame_count,
            "processing_time": processing_time,
            "tokens_per_second": token_usage.get("output_tokens", 0) / max(processing_time, 0.1),
            "frames_per_second": frame_count / max(processing_time, 0.1),
            "success": success,
            "input_tokens": token_usage.get("input_tokens", 0),
            "output_tokens": token_usage.get("output_tokens", 0)
        }
        
        self._frame_processing_stats["recent"].append(stats)
        
        # Keep only recent stats (last 100 entries)
        if len(self._frame_processing_stats["recent"]) > 100:
            self._frame_processing_stats["recent"] = self._frame_processing_stats["recent"][-100:]
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """Get performance insights for optimization."""
        stats = self._frame_processing_stats["recent"]
        
        if not stats:
            return {"status": "no_data"}
        
        successful_stats = [s for s in stats if s["success"]]
        
        if not successful_stats:
            return {"status": "no_successful_processing"}
        
        # Calculate performance metrics
        avg_processing_time = sum(s["processing_time"] for s in successful_stats) / len(successful_stats)
        avg_frames_per_sec = sum(s["frames_per_second"] for s in successful_stats) / len(successful_stats)
        avg_tokens_per_sec = sum(s["tokens_per_second"] for s in successful_stats) / len(successful_stats)
        
        # Find optimal frame count based on performance
        frame_performance = defaultdict(list)
        for stat in successful_stats:
            frame_performance[stat["frame_count"]].append(stat["processing_time"])
        
        optimal_frame_count = min(frame_performance.keys(), 
                                key=lambda x: sum(frame_performance[x]) / len(frame_performance[x]))
        
        return {
            "status": "healthy",
            "avg_processing_time": avg_processing_time,
            "avg_frames_per_second": avg_frames_per_sec,
            "avg_tokens_per_second": avg_tokens_per_sec,
            "success_rate": len(successful_stats) / len(stats),
            "optimal_frame_count": optimal_frame_count,
            "total_processed": len(stats),
            "cache_hit_ratio": self._calculate_cache_hit_ratio()
        }
    
    def _calculate_cache_hit_ratio(self) -> float:
        """Calculate cache hit ratio from recent activity."""
        # This would require tracking cache hits/misses in practice
        # For now, estimate based on cache size
        cache_size = len(self._temporal_cache)
        recent_requests = len(self._frame_processing_stats["recent"])
        
        if recent_requests == 0:
            return 0.0
        
        # Simple estimation - in practice, this would be tracked more precisely
        estimated_hit_ratio = min(cache_size / max(recent_requests, 1), 0.8)
        return estimated_hit_ratio
    
    def get_cost_optimization_recommendations(self) -> List[str]:
        """Get cost optimization recommendations for video processing."""
        recommendations = []
        insights = self.get_performance_insights()
        
        if insights.get("status") != "healthy":
            return ["Insufficient data for recommendations"]
        
        # Frame count optimization
        if insights["optimal_frame_count"] < self.config.max_frames_per_request:
            recommendations.append(
                f"Consider reducing max frames per request to {insights['optimal_frame_count']} "
                f"for better cost/performance ratio"
            )
        
        # Cache optimization
        if insights["cache_hit_ratio"] < 0.3:
            recommendations.append(
                "Low cache hit ratio detected - consider increasing temporal cache window "
                "or improving video segment identification"
            )
        
        # Performance optimization
        if insights["avg_processing_time"] > 30:  # 30 seconds threshold
            recommendations.append(
                "High processing times detected - consider frame sampling optimization "
                "or prompt simplification"
            )
        
        if insights["success_rate"] < 0.95:
            recommendations.append(
                "Low success rate detected - review retry logic and error handling"
            )
        
        return recommendations
    
    def cleanup_temporal_cache(self, max_age_seconds: int = None) -> int:
        """Clean up expired temporal cache entries."""
        max_age = max_age_seconds or self.config.temporal_cache_window
        cutoff_time = time.time() - max_age
        
        expired_keys = [
            key for key, (_, timestamp) in self._temporal_cache.items()
            if timestamp < cutoff_time
        ]
        
        for key in expired_keys:
            del self._temporal_cache[key]
        
        logger.info(f"Cleaned up {len(expired_keys)} expired temporal cache entries")
        return len(expired_keys)
    
    def update_config_from_performance(self) -> bool:
        """Auto-tune configuration based on performance data."""
        insights = self.get_performance_insights()
        
        if insights.get("status") != "healthy":
            return False
        
        updated = False
        
        # Auto-adjust max frames per request
        if insights["optimal_frame_count"] != self.config.max_frames_per_request:
            old_value = self.config.max_frames_per_request
            self.config.max_frames_per_request = insights["optimal_frame_count"]
            logger.info(f"Auto-tuned max_frames_per_request: {old_value} -> {insights['optimal_frame_count']}")
            updated = True
        
        # Auto-adjust cache window based on hit ratio
        if insights["cache_hit_ratio"] > 0.7:
            # High hit ratio - can extend cache window
            old_window = self.config.temporal_cache_window
            self.config.temporal_cache_window = min(old_window * 1.2, 600)  # Max 10 minutes
            if old_window != self.config.temporal_cache_window:
                logger.info(f"Auto-tuned temporal_cache_window: {old_window} -> {self.config.temporal_cache_window}")
                updated = True
        
        return updated


# Global video optimizer instance
video_bedrock_optimizer = VideoBedrockOptimizer()