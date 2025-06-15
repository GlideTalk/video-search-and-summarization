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
"""Frame Interceptor

This module provides functionality to intercept and save all decoded video frames
before the frame selector processes them. This allows capturing every frame from
the video stream regardless of frame selection criteria.
"""

import os
import ctypes
import uuid
import numpy as np
import torch
import cupy as cp
import pyds
from PIL import Image
from via_logger import logger


class FrameInterceptor:
    """Intercepts and saves all decoded video frames before frame selection"""
    
    def __init__(self, save_directory=None, save_format="jpg", enabled=False):
        """
        Initialize frame interceptor
        
        Args:
            save_directory: Directory to save intercepted frames (auto-generated if None)
            save_format: Format to save frames ('jpg', 'png', 'raw')
            enabled: Whether frame interception is enabled
        """
        self.enabled = enabled
        if not self.enabled:
            return
            
        self.save_format = save_format
        self.frame_count = 0
        
        # Generate unique save directory if not provided
        if save_directory is None:
            save_directory = f"/tmp/intercepted_frames_{uuid.uuid4()}"
        
        self.save_directory = save_directory
        
        # Create save directory
        try:
            os.makedirs(self.save_directory, exist_ok=True)
            logger.info(f"Frame interceptor initialized: {self.save_directory}")
        except Exception as e:
            logger.error(f"Failed to create interceptor directory: {e}")
            self.enabled = False
        
    def intercept_frame_gpu(self, buffer, width, height, pts=None):
        """
        Intercept and save a decoded frame from GPU buffer
        
        Args:
            buffer: GStreamer buffer containing frame data
            width: Frame width
            height: Frame height
            pts: Presentation timestamp (optional)
        """
        if not self.enabled:
            return
            
        try:
            # Extract GPU memory pointer and create tensor from it using
            # DeepStream Python Bindings and cupy
            _, shape, strides, dataptr, size = pyds.get_nvds_buf_surface_gpu(hash(buffer), 0)
            
            ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
            ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
            owner = None
            c_data_ptr = ctypes.pythonapi.PyCapsule_GetPointer(dataptr, None)
            unownedmem = cp.cuda.UnownedMemory(c_data_ptr, size, owner)
            memptr = cp.cuda.MemoryPointer(unownedmem, 0)
            n_frame_gpu = cp.ndarray(
                shape=shape, dtype=np.uint8, memptr=memptr, strides=strides, order="C"
            )
            
            # Convert to PyTorch tensor and move to CPU
            image_tensor = torch.tensor(
                n_frame_gpu, dtype=torch.uint8, requires_grad=False, device="cuda"
            )
            frame_data = image_tensor.cpu().detach().numpy()
            
            # Save frame
            self._save_frame(frame_data, width, height, pts)
            
        except Exception as e:
            logger.error(f"Error intercepting GPU frame: {e}")
    
    def intercept_frame_cpu(self, buffer, width, height, pts=None):
        """
        Intercept and save a decoded frame from CPU buffer
        
        Args:
            buffer: GStreamer buffer containing frame data
            width: Frame width  
            height: Frame height
            pts: Presentation timestamp (optional)
        """
        if not self.enabled:
            return
            
        try:
            # Raw buffer mapping for CPU frames
            _, mapinfo = buffer.map(Gst.MapFlags.READ)
            frame_data = np.frombuffer(mapinfo.data, dtype=np.uint8)
            buffer.unmap(mapinfo)
            
            # Reshape to image dimensions
            if len(frame_data) >= width * height * 3:
                frame_data = frame_data[:width * height * 3].reshape(height, width, 3)
                self._save_frame(frame_data, width, height, pts)
            else:
                logger.warning(f"Insufficient frame data: {len(frame_data)} bytes for {width}x{height}")
            
        except Exception as e:
            logger.error(f"Error intercepting CPU frame: {e}")
    
    def _save_frame(self, frame_data, width, height, pts=None):
        """Save frame data to file"""
        if not self.enabled:
            return
            
        try:
            timestamp_str = f"{pts/1e9:.3f}" if pts else f"{self.frame_count:06d}"
            
            if self.save_format == "raw":
                # Save raw numpy array
                filename = f"frame_{timestamp_str}.npy"
                filepath = os.path.join(self.save_directory, filename)
                np.save(filepath, frame_data)
                
            else:
                # Convert to image format
                if len(frame_data.shape) == 3:
                    if frame_data.shape[0] == 3:  # CHW format
                        frame_data = frame_data.transpose(1, 2, 0)  # Convert to HWC
                    elif frame_data.shape[2] == 3:  # Already HWC
                        pass
                    else:
                        # Reshape if needed
                        frame_data = frame_data.reshape(height, width, 3)
                else:
                    frame_data = frame_data.reshape(height, width, 3)
                
                # Ensure uint8 format
                frame_data = frame_data.astype(np.uint8)
                
                # Create PIL image and save
                pil_image = Image.fromarray(frame_data)
                filename = f"frame_{timestamp_str}.{self.save_format}"
                filepath = os.path.join(self.save_directory, filename)
                pil_image.save(filepath)
            
            self.frame_count += 1
            if self.frame_count % 100 == 0:  # Log every 100th frame to avoid spam
                logger.info(f"Intercepted {self.frame_count} frames, latest: {filepath}")
                
        except Exception as e:
            logger.error(f"Error saving intercepted frame: {e}")
    
    def get_stats(self):
        """Get interception statistics"""
        return {
            "enabled": self.enabled,
            "frames_intercepted": self.frame_count,
            "save_directory": self.save_directory if self.enabled else None,
            "save_format": self.save_format if self.enabled else None
        }