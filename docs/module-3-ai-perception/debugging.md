# Debugging AI Perception Systems

## Common AI Perception Issues and Solutions

### 1. Model Loading and Execution Issues

**Problem**: AI models fail to load or execute properly.

**Solutions**:
- Check GPU availability and CUDA compatibility:
  ```bash
  nvidia-smi
  nvcc --version
  python3 -c "import torch; print(torch.cuda.is_available())"
  ```
- Verify model dependencies:
  ```bash
  python3 -c "import torch; import torchvision; import ultralytics"
  ```
- Check model file integrity:
  ```bash
  ls -la ~/.cache/torch/hub/
  ls -la ~/.ultralytics/weights/
  ```

**Example of proper model loading with error handling**:
```python
def load_model_safely(self):
    try:
        # Check if CUDA is available
        if torch.cuda.is_available():
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.model = self.model.to('cuda')
            self.get_logger().info('Model loaded on GPU')
        else:
            self.get_logger().warn('CUDA not available, loading model on CPU')
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

        self.model.eval()
        return True
    except Exception as e:
        self.get_logger().error(f'Failed to load model: {e}')
        return False
```

### 2. Performance and Latency Issues

**Problem**: Perception system runs too slowly or causes frame drops.

**Solutions**:
- Monitor GPU utilization:
  ```bash
  watch -n 1 nvidia-smi
  ```
- Check CPU and memory usage:
  ```bash
  htop
  ```
- Reduce input resolution:
  ```python
  # Preprocess image to smaller size
  height, width = image.shape[:2]
  new_height, new_width = height//2, width//2
  resized_image = cv2.resize(image, (new_width, new_height))
  ```

### 3. Memory Management Issues

**Problem**: Out of memory errors during model inference.

**Solutions**:
- Clear GPU cache regularly:
  ```python
  torch.cuda.empty_cache()
  ```
- Use smaller batch sizes:
  ```python
  # Process images one at a time
  for single_image in batch:
      result = model(single_image)
  ```
- Monitor memory usage:
  ```python
  if torch.cuda.is_available():
      memory_allocated = torch.cuda.memory_allocated()
      memory_reserved = torch.cuda.memory_reserved()
      self.get_logger().info(f'GPU Memory - Allocated: {memory_allocated}, Reserved: {memory_reserved}')
  ```

### 4. Sensor Data Issues

**Problem**: Camera images not received or incorrect format.

**Solutions**:
- Check camera topic:
  ```bash
  ros2 topic list | grep camera
  ros2 topic echo /camera/image_raw --field header
  ```
- Verify image format:
  ```bash
  ros2 topic info /camera/image_raw
  ```
- Test with image view:
  ```bash
  ros2 run image_view image_view_raw image:=/camera/image_raw
  ```

**Example of proper image handling**:
```python
def image_callback(self, msg):
    try:
        # Convert ROS Image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Verify image dimensions
        if cv_image.size == 0:
            self.get_logger().error('Received empty image')
            return

        # Process image
        self.process_image(cv_image)

    except CvBridgeError as e:
        self.get_logger().error(f'CvBridge error: {e}')
    except Exception as e:
        self.get_logger().error(f'Error processing image: {e}')
```

### 5. Object Detection Accuracy Issues

**Problem**: Poor detection accuracy or false positives.

**Solutions**:
- Adjust confidence thresholds:
  ```python
  # Filter detections by confidence
  high_conf_detections = detections[detections['confidence'] > 0.5]
  ```
- Use Non-Maximum Suppression (NMS):
  ```python
  # Apply NMS to remove duplicate detections
  indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
  ```
- Validate detection results:
  ```python
  def validate_detections(self, detections):
      valid_detections = []
      for detection in detections:
          # Check if bounding box is valid
          if (detection['xmax'] > detection['xmin'] and
              detection['ymax'] > detection['ymin'] and
              0 <= detection['confidence'] <= 1.0):
              valid_detections.append(detection)
      return valid_detections
  ```

## Debugging Tools for AI Perception

### 1. Profiling Tools

**PyTorch Profiler**:
```python
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    # Run your model inference here
    results = model(image)

# Print profiling results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

**ROS 2 Performance Tools**:
```bash
# Monitor topic frequency
ros2 topic hz /camera/image_raw

# Monitor node resource usage
ros2 run top top

# Check message delays
ros2 topic delay /object_detections
```

### 2. Visualization Tools

**OpenCV Debug Visualization**:
```python
def visualize_detections(self, image, detections):
    debug_image = image.copy()

    for detection in detections:
        # Draw bounding box
        pt1 = (int(detection['xmin']), int(detection['ymin']))
        pt2 = (int(detection['xmax']), int(detection['ymax']))
        cv2.rectangle(debug_image, pt1, pt2, (0, 255, 0), 2)

        # Draw label
        label = f"{detection['name']}: {detection['confidence']:.2f}"
        cv2.putText(debug_image, label, (pt1[0], pt1[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image
    cv2.imshow('Debug Visualization', debug_image)
    cv2.waitKey(1)
```

**RViz2 Visualization**:
```bash
# Launch RViz2 for visualization
rviz2

# Add displays for:
# - Image display for camera feeds
# - MarkerArray for detection results
# - PointCloud2 for 3D perception
# - TF for coordinate transforms
```

### 3. Logging and Monitoring

**Detailed Logging**:
```python
def debug_log_detection_info(self, detections):
    self.get_logger().debug(f'Number of detections: {len(detections)}')

    for i, detection in enumerate(detections):
        self.get_logger().debug(f'Detection {i}: {detection["name"]} '
                               f'Confidence: {detection["confidence"]:.2f} '
                               f'BBox: ({detection["xmin"]}, {detection["ymin"]}, '
                               f'{detection["xmax"]}, {detection["ymax"]})')
```

**Performance Monitoring**:
```python
import time

class PerformanceMonitor:
    def __init__(self):
        self.frame_times = []
        self.start_time = None

    def start_frame(self):
        self.start_time = time.time()

    def end_frame(self):
        if self.start_time:
            frame_time = time.time() - self.start_time
            self.frame_times.append(frame_time)

            # Keep only last 100 frames for averaging
            if len(self.frame_times) > 100:
                self.frame_times.pop(0)

    def get_fps(self):
        if self.frame_times:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            return 1.0 / avg_time if avg_time > 0 else 0.0
        return 0.0
```

## AI-Specific Debugging Techniques

### 1. Model Input Validation

**Input Preprocessing Debug**:
```python
def debug_preprocessing(self, original_image, processed_image):
    # Log input statistics
    self.get_logger().debug(f'Original image shape: {original_image.shape}, '
                           f'dtype: {original_image.dtype}')
    self.get_logger().debug(f'Processed image shape: {processed_image.shape}, '
                           f'dtype: {processed_image.dtype}')

    # Check for valid input range
    if processed_image.min() < 0 or processed_image.max() > 1:
        self.get_logger().warn('Processed image values are outside [0, 1] range')

    # Save debug images if needed
    if self.debug_save_images:
        cv2.imwrite(f'debug_input_{int(time.time())}.png', original_image)
```

### 2. Confidence and Threshold Tuning

**Adaptive Thresholding**:
```python
class AdaptiveThreshold:
    def __init__(self, initial_threshold=0.5, adjustment_rate=0.01):
        self.threshold = initial_threshold
        self.adjustment_rate = adjustment_rate
        self.good_detections = 0
        self.total_detections = 0

    def adjust_threshold(self, is_good_detection):
        self.total_detections += 1
        if is_good_detection:
            self.good_detections += 1

        # Adjust threshold based on success rate
        success_rate = self.good_detections / max(1, self.total_detections)

        if success_rate < 0.7:  # Too few good detections, lower threshold
            self.threshold = max(0.1, self.threshold - self.adjustment_rate)
        elif success_rate > 0.9:  # Too many detections, raise threshold
            self.threshold = min(0.9, self.threshold + self.adjustment_rate)
```

### 3. Model Output Validation

**Output Validation**:
```python
def validate_model_output(self, results):
    """Validate model output format and values"""
    try:
        # Check if results have expected structure
        if not hasattr(results, 'pandas'):
            self.get_logger().error('Results do not have pandas method')
            return False

        # Get detections
        detections = results.pandas().xyxy[0]

        # Validate detection format
        required_columns = ['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'name']
        for col in required_columns:
            if col not in detections.columns:
                self.get_logger().error(f'Missing column {col} in detections')
                return False

        # Validate numerical values
        for _, detection in detections.iterrows():
            if (detection['xmin'] >= detection['xmax'] or
                detection['ymin'] >= detection['ymax']):
                self.get_logger().warn(f'Invalid bounding box: {detection}')

            if not 0 <= detection['confidence'] <= 1:
                self.get_logger().warn(f'Invalid confidence value: {detection["confidence"]}')

        return True
    except Exception as e:
        self.get_logger().error(f'Error validating model output: {e}')
        return False
```

## Common Error Messages and Solutions

### "CUDA out of memory"
- Clear GPU cache: `torch.cuda.empty_cache()`
- Reduce batch size or input resolution
- Use mixed precision: `model.half()`

### "Segmentation fault" during inference
- Check CUDA driver compatibility
- Verify GPU memory is sufficient
- Update PyTorch/CUDA versions

### "Input type isn't supported"
- Verify input tensor format and dimensions
- Check data type compatibility
- Ensure proper preprocessing

### "No module named 'ultralytics'"
- Install required packages: `pip3 install ultralytics`
- Check Python environment
- Verify package installation path

## Debugging Best Practices

### 1. Isolate Components

Test perception components independently:
```bash
# Test model separately
python3 -c "import torch; model = torch.hub.load('ultralytics/yolov5', 'yolov5s'); print('Model loaded successfully')"

# Test image pipeline separately
ros2 run image_view image_view_raw image:=/camera/image_raw
```

### 2. Use Debug Modes

Enable detailed logging:
```python
# In your node
import rclpy.logging
logger = self.get_logger()
logger.set_level(rclpy.logging.LoggingSeverity.DEBUG)
```

### 3. Performance Baselines

Establish performance baselines:
```bash
# Monitor baseline performance
ros2 topic hz /camera/image_raw
ros2 topic hz /object_detections
```

### 4. Validation Checks

Implement comprehensive validation:
```python
def comprehensive_validation(self, image, detections):
    # Validate input image
    if image is None or image.size == 0:
        return False, "Empty image"

    # Validate detection format
    if detections is None:
        return False, "No detections"

    # Validate detection content
    for detection in detections:
        if not self.is_valid_detection(detection):
            return False, f"Invalid detection: {detection}"

    return True, "Valid"
```

## Debugging Checklist

Before deploying AI perception systems, verify:

- [ ] Model loads successfully on target hardware
- [ ] Input images are received in correct format
- [ ] GPU memory usage is within limits
- [ ] Detection accuracy meets requirements
- [ ] System performance is stable
- [ ] Error handling is implemented
- [ ] Logging provides sufficient debug information
- [ ] Safety mechanisms are in place