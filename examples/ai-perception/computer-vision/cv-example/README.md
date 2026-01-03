# Computer Vision Example

## Overview

This example demonstrates computer vision techniques using OpenCV and ROS 2 for educational robotics. The system processes camera images to detect objects, track them, and provide perception data for navigation and interaction.

## Components

### 1. Object Detection Node
The main component that performs object detection:
- Image processing and conversion
- Feature detection and classification
- Detection result publishing
- Visualization capabilities

### 2. OpenCV Integration
- Image preprocessing and enhancement
- Feature extraction and matching
- Object detection algorithms
- Visualization and debugging tools

## Prerequisites

- ROS 2 Humble Hawksbill
- OpenCV (cv2)
- vision_msgs package
- cv_bridge for image conversion
- Camera sensor for input

## Installation

1. Install OpenCV:
```bash
pip install opencv-python numpy
```

2. Install ROS 2 vision packages:
```bash
sudo apt update
sudo apt install ros-humble-vision-msgs ros-humble-vision-opencv
```

## Usage

### Running the Basic Node

```bash
# Source your ROS workspace
source install/setup.bash

# Run the basic computer vision node
ros2 run cv_example object_detection_node
```

### Running the YOLO-Based Node

```bash
# Run with YOLO implementation (mock in this example)
ros2 run cv_example object_detection_node yolo
```

### Launching with Camera

```bash
# If using a camera driver
ros2 launch your_camera_package camera.launch.py

# Then run the computer vision node
ros2 run cv_example object_detection_node
```

## Configuration

### Parameters

The computer vision node uses several parameters:

- `confidence_threshold`: Minimum confidence for detection
- `nms_threshold`: Non-maximum suppression threshold
- `class_labels`: List of object classes to detect

### Launch File

Create a launch file to bring up the complete system:

```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time')
    node_type = LaunchConfiguration('node_type')

    # Declare launch arguments
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )

    declare_node_type_cmd = DeclareLaunchArgument(
        'node_type',
        default_value='basic',
        description='Type of CV node to run: basic or yolo'
    )

    # Computer vision node
    cv_node = Node(
        package='cv_example',
        executable='object_detection_node',
        name='computer_vision_node',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'node_type': node_type}
        ],
        output='screen'
    )

    # Create the launch description and populate
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_node_type_cmd)

    # Add nodes
    ld.add_action(cv_node)

    return ld
```

## Topics

### Subscribed Topics
- `/camera/image_raw` - Camera image for processing

### Published Topics
- `/cv_detections` - Detection results in vision_msgs format
- `/cv_visualization` - Processed image with detections drawn
- `/cv_status` - Status information about detection results

## Educational Applications

### Learning Objectives
- Computer vision fundamentals
- Object detection algorithms
- Image processing techniques
- ROS 2 integration with vision systems
- Real-time processing concepts

### Classroom Activities
- Compare different detection algorithms
- Study the effect of lighting conditions
- Explore feature extraction techniques
- Implement custom object detection

## Performance Considerations

### Hardware Requirements
- CPU with sufficient processing power for real-time image processing
- Adequate memory for image buffers
- Camera with appropriate resolution and frame rate

### Optimization Tips
- Reduce image resolution for faster processing
- Use efficient algorithms for real-time performance
- Consider hardware acceleration options
- Optimize detection parameters for your use case

## Integration with Other Systems

### Navigation Integration
```python
# Example of using CV detections for navigation
def detection_callback(self, msg):
    """Process detections for navigation safety"""
    for detection in msg.detections:
        if detection.results[0].hypothesis.class_id == 'person':
            # Slow down or stop if person detected
            self.safety_stop()
```

### Perception Pipeline
```python
# Example of integrating with perception pipeline
def integrate_with_perception(self, detections):
    """Integrate CV results with other perception components"""
    # Combine with LiDAR data
    # Update object tracking
    # Publish fused perception results
```

## Troubleshooting

### Common Issues

1. **No Detections**
   - Check camera calibration and image quality
   - Verify lighting conditions
   - Adjust detection thresholds

2. **Performance Issues**
   - Reduce image resolution
   - Lower frame rate
   - Optimize detection parameters

3. **False Positives**
   - Adjust confidence thresholds
   - Improve lighting conditions
   - Use more sophisticated detection algorithms

### Debugging Commands

```bash
# Monitor detection results
ros2 topic echo /cv_detections

# Check image feed
ros2 run image_view image_view image:=/camera/image_raw

# Visualize results
ros2 run image_view image_view image:=/cv_visualization

# Monitor performance
ros2 topic hz /cv_detections
```

## Extensions

### Advanced Features
- Face recognition capabilities
- Gesture detection
- 3D object detection
- Multi-camera processing

### Deep Learning Integration
- Custom trained models
- TensorFlow/PyTorch integration
- Transfer learning
- Online learning capabilities

## References

- [OpenCV Documentation](https://docs.opencv.org/)
- [ROS 2 Vision Tutorials](http://wiki.ros.org/cv_bridge)
- [Vision Messages](https://github.com/ros-perception/vision_msgs)

## Next Steps

- Integrate with Isaac Sim for simulation
- Add neural network-based detection
- Implement tracking capabilities
- Create educational curriculum around computer vision