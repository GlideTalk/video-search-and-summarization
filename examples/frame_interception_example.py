#!/usr/bin/env python3
######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
######################################################################################################
"""
Frame Interception Example

This example demonstrates how to use the frame interception feature to capture
and save all decoded video frames before the frame selector processes them.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the parent directory to Python path to import VSS modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "vss-engine" / "src" / "vlm_pipeline"))

from video_file_frame_getter import VideoFileFrameGetter, DefaultFrameSelector
from chunk_info import ChunkInfo


def demonstrate_frame_interception(video_file, num_selected_frames=4):
    """
    Demonstrate frame interception functionality
    
    Args:
        video_file: Path to video file to process
        num_selected_frames: Number of frames to select (for comparison)
    """
    print("=" * 60)
    print("Frame Interception Demonstration")
    print("=" * 60)
    
    # Create temporary directory for intercepted frames
    with tempfile.TemporaryDirectory(prefix="frame_interception_") as temp_dir:
        print(f"Intercepted frames will be saved to: {temp_dir}")
        
        # Initialize VideoFileFrameGetter with frame interception enabled
        frame_getter = VideoFileFrameGetter(
            frame_selector=DefaultFrameSelector(num_selected_frames),
            enable_jpeg_output=False,
            audio_support=False,
            enable_frame_interception=True,
            frame_interception_dir=temp_dir,
            frame_interception_format="jpg"
        )
        
        # Create chunk info for the entire video
        chunk = ChunkInfo()
        chunk.file = video_file
        chunk.start_pts = 0
        chunk.end_pts = -1  # Process entire file
        
        print(f"Processing video: {video_file}")
        print(f"Frame selector will choose: {num_selected_frames} frames")
        print("Frame interceptor will save: ALL frames")
        print()
        
        try:
            # Process the video
            selected_frames, frame_timestamps, audio_frames = frame_getter.get_frames(chunk)
            
            # Get interception statistics
            stats = frame_getter.get_frame_interceptor_stats()
            
            # Print results
            print("Processing Results:")
            print(f"  Selected frames: {len(selected_frames)}")
            print(f"  Selected timestamps: {frame_timestamps}")
            print()
            print("Frame Interception Results:")
            print(f"  Intercepted frames: {stats['frames_intercepted']}")
            print(f"  Save directory: {stats['save_directory']}")
            print(f"  Save format: {stats['save_format']}")
            print()
            
            # List saved files
            saved_files = list(Path(temp_dir).glob("*.jpg"))
            print(f"Saved frame files ({len(saved_files)} total):")
            for i, file_path in enumerate(sorted(saved_files)[:10]):  # Show first 10
                print(f"  {file_path.name}")
            if len(saved_files) > 10:
                print(f"  ... and {len(saved_files) - 10} more files")
            
            print()
            print("Summary:")
            print(f"  - Frame selector chose {len(selected_frames)} frames for processing")
            print(f"  - Frame interceptor saved {stats['frames_intercepted']} total frames")
            print(f"  - Interception captured {stats['frames_intercepted'] - len(selected_frames)} additional frames")
            
        except Exception as e:
            print(f"Error processing video: {e}")
            return False
    
    return True


def demonstrate_comparison():
    """Demonstrate the difference between normal processing and frame interception"""
    print("\n" + "=" * 60)
    print("Comparison: Normal vs Frame Interception")
    print("=" * 60)
    
    # This would typically use a real video file
    # For demonstration purposes, we'll show the concept
    
    print("Normal Processing:")
    print("  Video → Decoder → Frame Selector (chooses N frames) → Processing")
    print("  Result: Only selected frames are available")
    print()
    print("With Frame Interception:")
    print("  Video → Decoder → Frame Interceptor (saves ALL) → Frame Selector → Processing")
    print("                          ↓")
    print("                    All frames saved")
    print("  Result: Selected frames for processing + ALL frames saved to disk")
    print()


def main():
    """Main demonstration function"""
    print("Frame Interception Feature Demonstration")
    print("This example shows how to capture all decoded video frames.")
    print()
    
    # Check if video file is provided
    if len(sys.argv) < 2:
        print("Usage: python frame_interception_example.py <video_file>")
        print()
        print("Example:")
        print("  python frame_interception_example.py /path/to/video.mp4")
        print()
        demonstrate_comparison()
        return
    
    video_file = sys.argv[1]
    
    # Check if video file exists
    if not os.path.exists(video_file):
        print(f"Error: Video file not found: {video_file}")
        return
    
    # Run demonstration
    success = demonstrate_frame_interception(video_file)
    
    if success:
        demonstrate_comparison()
        print("\n" + "=" * 60)
        print("Demonstration completed successfully!")
        print("The frame interception feature allows you to capture every")
        print("decoded frame while still using the frame selector for processing.")
    else:
        print("Demonstration failed. Please check the video file and try again.")


if __name__ == "__main__":
    main()