# Frame Interception Feature

This document describes the frame interception functionality added to the video processing pipeline.

## Overview

The frame interception feature allows capturing and saving all decoded video frames before the frame selector processes them. This ensures that every frame from the video stream is saved, regardless of the frame selection criteria.

## Key Features

- **Complete Frame Capture**: Intercepts all decoded frames before frame selection
- **Multiple Save Formats**: Supports JPEG, PNG, and raw NumPy array formats
- **GPU Memory Support**: Handles both CPU and GPU buffers using NVIDIA DeepStream
- **Configurable Output**: Customizable save directory and format
- **Performance Optimized**: Minimal impact on video processing pipeline
- **Statistics Tracking**: Provides frame count and save location information

## Architecture

### Integration Points

The frame interceptor is integrated at two key locations in the video processing pipeline:

1. **Standard Pipeline** (`_create_pipeline`):
   ```
   uridecodebin → nvvideoconvert → [FRAME INTERCEPTION] → frame_selector → appsink
   ```

2. **OSD Pipeline** (`_create_osd_pipeline`):
   ```
   uridecodebin → nvstreammux → nvinferserver → nvtracker → [FRAME INTERCEPTION] → frame_selector → nvdsosd
   ```

### Frame Flow

```
Video Input → Hardware Decoder → GPU Buffer → Frame Interceptor → Frame Selector → Cache
                                                    ↓
                                              Save to Disk
```

## Usage

### Command Line Interface

```bash
# Enable frame interception with default settings
python video_file_frame_getter.py video.mp4 --enable-frame-interception=True

# Specify custom directory and format
python video_file_frame_getter.py video.mp4 \
    --enable-frame-interception=True \
    --frame-interception-dir=/path/to/save/frames \
    --frame-interception-format=png

# Use with live streams
python video_file_frame_getter.py rtsp://stream-url \
    --enable-frame-interception=True \
    --chunk-duration=10
```

### Programmatic Usage

```python
from vlm_pipeline.video_file_frame_getter import VideoFileFrameGetter, DefaultFrameSelector
from chunk_info import ChunkInfo

# Initialize with frame interception enabled
frame_getter = VideoFileFrameGetter(
    frame_selector=DefaultFrameSelector(8),
    enable_frame_interception=True,
    frame_interception_dir="/tmp/my_frames",
    frame_interception_format="jpg"
)

# Process video file
chunk = ChunkInfo()
chunk.file = "video.mp4"
chunk.start_pts = 0
chunk.end_pts = -1  # Process entire file

frames, timestamps, audio = frame_getter.get_frames(chunk)

# Get interception statistics
stats = frame_getter.get_frame_interceptor_stats()
print(f"Intercepted {stats['frames_intercepted']} frames")
print(f"Saved to: {stats['save_directory']}")
```

## Configuration Options

### Constructor Parameters

- `enable_frame_interception` (bool): Enable/disable frame interception (default: False)
- `frame_interception_dir` (str): Directory to save frames (auto-generated if None)
- `frame_interception_format` (str): Save format - "jpg", "png", or "raw" (default: "jpg")

### Command Line Arguments

- `--enable-frame-interception`: Enable frame interception
- `--frame-interception-dir`: Custom save directory
- `--frame-interception-format`: Save format (jpg, png, raw)

## File Naming Convention

Intercepted frames are saved with the following naming pattern:
- `frame_{timestamp}.{format}` - where timestamp is in seconds (e.g., `frame_12.345.jpg`)
- `frame_{counter}.{format}` - if no timestamp available (e.g., `frame_000123.jpg`)

## Performance Considerations

- Frame interception adds minimal overhead to the video processing pipeline
- GPU memory access is optimized using NVIDIA DeepStream bindings and CuPy
- Disk I/O is the primary performance bottleneck when saving many frames
- Consider using raw format for maximum processing speed if post-processing is needed

## Memory Management

- Frames are processed and saved immediately to minimize memory usage
- GPU tensors are properly moved to CPU before saving to avoid memory leaks
- Buffer mapping is handled safely with proper cleanup

## Error Handling

- Graceful handling of GPU/CPU buffer access errors
- Directory creation with proper error reporting
- Invalid frame data detection and logging
- Continues processing even if individual frame saves fail

## Limitations

- Requires NVIDIA GPU and DeepStream for GPU buffer access
- Large video files may generate many frame files
- Disk space requirements scale with video length and resolution
- Frame timestamps depend on video stream metadata availability

## Troubleshooting

### Common Issues

1. **Permission Errors**: Ensure write permissions for the save directory
2. **GPU Memory Errors**: Check CUDA/DeepStream installation
3. **Large File Count**: Consider using raw format and post-processing for efficiency
4. **Missing Frames**: Verify GPU buffer access and DeepStream configuration

### Debug Information

Enable verbose logging to see frame interception details:
```python
import logging
logging.getLogger('via_logger').setLevel(logging.DEBUG)
```

## Examples

### Save All Frames from Video File
```bash
python video_file_frame_getter.py my_video.mp4 \
    --enable-frame-interception=True \
    --frame-interception-dir=/tmp/all_frames \
    --num-frames=8
```

### Intercept Frames from RTSP Stream
```bash
python video_file_frame_getter.py rtsp://camera-ip/stream \
    --enable-frame-interception=True \
    --chunk-duration=5 \
    --frame-interception-format=png
```

### Process with Computer Vision Pipeline
```bash
python video_file_frame_getter.py video.mp4 \
    --enable-frame-interception=True \
    --enable-cv-pipeline=True \
    --frame-interception-dir=/tmp/cv_frames
```

## Future Enhancements

- Frame filtering based on quality metrics
- Compressed video output from intercepted frames
- Real-time frame analysis and selective saving
- Integration with cloud storage for large-scale processing