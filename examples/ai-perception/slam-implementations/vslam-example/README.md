# Isaac ROS Visual SLAM Example

## Overview

This example demonstrates Visual SLAM (Simultaneous Localization and Mapping) using Isaac ROS components. The system processes stereo or monocular camera images to estimate the robot's pose and build a map of the environment.

## Components

### 1. VSLAM Node
The main component that performs visual SLAM using:
- Feature detection and matching
- Stereo depth estimation
- Pose estimation and tracking
- Trajectory recording

### 2. Isaac ROS Integration
- Hardware-accelerated image processing
- Stereo camera support
- TF tree management
- Odometry publishing

## Prerequisites

- ROS 2 Humble Hawksbill
- Isaac ROS packages
- OpenCV
- Camera sensor (stereo or monocular)
- Compatible GPU for hardware acceleration

## Installation

1. Install Isaac ROS packages:
```bash
sudo apt update
sudo apt install ros-humble-isaac-ros-*
```

2. Install Python dependencies:
```bash
pip install opencv-python numpy
```

## Usage

### Running with Stereo Camera

```bash
# Launch the VSLAM node
ros2 run vslam_example vslam_node

# Or using a launch file (see below)
ros2 launch vslam_example vslam_stereo.launch.py
```

### Running with Monocular Camera

```bash
# The node automatically falls back to monocular processing if stereo is not available
ros2 run vslam_example vslam_node
```

## Launch File

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

    # Declare launch arguments
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )

    # VSLAM node
    vslam_node = Node(
        package='vslam_example',
        executable='vslam_node',
        name='vslam_node',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Create the launch description and populate
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)

    # Add nodes
    ld.add_action(vslam_node)

    return ld
```

## Topics

### Subscribed Topics
- `/camera/left/image_rect_color` - Left camera image (stereo)
- `/camera/right/image_rect_color` - Right camera image (stereo)
- `/camera/left/camera_info` - Left camera info (stereo)
- `/camera/right/camera_info` - Right camera info (stereo)
- `/camera/image_raw` - Monocular camera image (fallback)

### Published Topics
- `/vslam/odometry` - Estimated odometry
- `/vslam/pose` - Estimated pose
- `/tf` - Transform tree

## Parameters

The node uses several parameters for tuning:
- Feature detection parameters (number of features, matching thresholds)
- Camera calibration parameters
- Motion estimation parameters

## Educational Applications

### Learning Objectives
- Visual SLAM algorithms and implementation
- Feature detection and matching
- Stereo vision and depth estimation
- Robot localization and mapping
- ROS integration with computer vision

### Classroom Activities
- Compare stereo vs. monocular SLAM
- Analyze the effect of different feature detectors
- Study the impact of lighting conditions on SLAM performance
- Explore trajectory optimization techniques

## Performance Considerations

### Hardware Requirements
- GPU for hardware acceleration (recommended)
- Sufficient CPU for image processing
- Adequate memory for map storage

### Optimization Tips
- Reduce image resolution for faster processing
- Adjust feature detection parameters for performance
- Use appropriate matching algorithms for your use case

## Troubleshooting

### Common Issues

1. **No Features Detected**
   - Ensure adequate lighting in the environment
   - Check camera calibration
   - Verify image quality

2. **Drift in Trajectory**
   - Increase number of tracked features
   - Improve camera calibration
   - Use loop closure techniques

3. **High CPU/GPU Usage**
   - Reduce image resolution
   - Lower feature detection parameters
   - Optimize matching algorithms

### Debugging

Monitor the following topics to debug VSLAM performance:
```bash
# Monitor odometry
ros2 topic echo /vslam/odometry

# Check feature matches
ros2 topic hz /vslam/odometry

# Visualize results in RViz
ros2 run rviz2 rviz2
```

## Extensions

### Advanced Features
- Loop closure detection
- Map optimization
- Multi-session mapping
- Semantic SLAM

### Integration
- Combine with LiDAR SLAM
- Integrate with navigation stack
- Add relocalization capabilities

## References

- [Isaac ROS Documentation](https://nvidia-isaac-ros.github.io/)
- [Visual SLAM Survey](https://arxiv.org/abs/1606.05830)
- [OpenCV Feature Detection](https://docs.opencv.org/)

## Next Steps

- Integrate with Isaac Sim for simulation
- Add neural network-based features
- Implement dynamic object tracking
- Create educational curriculum around VSLAM