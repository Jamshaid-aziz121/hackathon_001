# AI Perception Tooling and SDKs

## NVIDIA Isaac ROS Setup

### Installation Prerequisites

Before installing Isaac ROS packages, ensure you have:
- ROS 2 Humble Hawksbill
- NVIDIA GPU with CUDA support (Compute Capability 6.0+)
- CUDA 11.8 or later
- cuDNN 8.6 or later
- Isaac ROS Dev Kit

### Installing Isaac ROS Packages

Install the core Isaac ROS packages:

```bash
# Add the Isaac ROS repository
sudo apt update && sudo apt install curl gnupg lsb-release
curl -sSL https://nvidia.github.io/isaac_ros/setup/installer.sh | sudo bash
sudo apt update

# Install Isaac ROS common packages
sudo apt install ros-humble-isaac-ros-common

# Install specific perception packages
sudo apt install ros-humble-isaac-ros-visual-slam
sudo apt install ros-humble-isaac-ros-pointcloud-utils
sudo apt install ros-humble-isaac-ros-stereo-image-pipeline
sudo apt install ros-humble-isaac-ros-rosbag-utils
```

### Isaac ROS Common Components

#### VSLAM (Visual Simultaneous Localization and Mapping)

The VSLAM package provides GPU-accelerated visual SLAM capabilities:

```bash
# Install VSLAM package
sudo apt install ros-humble-isaac-ros-visual-slam

# Example launch file for stereo VSLAM
# stereo_vslam.launch.py
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    container = ComposableNodeContainer(
        name='vslam_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            ComposableNode(
                package='isaac_ros_visual_slam',
                plugin='isaac_ros::visual_slam::VisualSLAMNode',
                name='visual_slam',
                parameters=[{
                    'enable_rectified_pose': True,
                    'map_frame': 'map',
                    'odometry_frame': 'odom',
                    'base_frame': 'base_link',
                    'enable_observations_view': True,
                    'enable_slam_visualization': True,
                    'enable_landmarks_view': True,
                    'enable_imu': False,
                }],
                remappings=[
                    ('/visual_slam/image0', '/camera/left/image_raw'),
                    ('/visual_slam/image1', '/camera/right/image_raw'),
                    ('/visual_slam/camera_info0', '/camera/left/camera_info'),
                    ('/visual_slam/camera_info1', '/camera/right/camera_info'),
                ],
            )
        ],
        output='screen'
    )

    return LaunchDescription([container])
```

#### Point Cloud Processing

The point cloud utilities package provides tools for processing 3D point cloud data:

```bash
# Install point cloud utilities
sudo apt install ros-humble-isaac-ros-pointcloud-utils

# Example usage in a node
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from isaac_ros_pointcloud_utils.msg import PointCloud2Packet

class PointCloudProcessor(Node):
    def __init__(self):
        super().__init__('pointcloud_processor')
        self.subscription = self.create_subscription(
            PointCloud2,
            'input_pointcloud',
            self.pointcloud_callback,
            10)
        self.publisher = self.create_publisher(
            PointCloud2,
            'output_pointcloud',
            10)

    def pointcloud_callback(self, msg):
        # Process point cloud data
        self.get_logger().info(f'Processing point cloud with {len(msg.data)} bytes')
        # Add your processing logic here
        self.publisher.publish(msg)
```

## Isaac Navigation (Nav2) Setup

### Installing Navigation Packages

```bash
# Install Nav2 packages
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup

# Install Isaac-specific navigation packages
sudo apt install ros-humble-isaac-ros-navigation
```

### Navigation Configuration

Create a basic navigation configuration file `nav2_config.yaml`:

```yaml
amcl:
  ros__parameters:
    use_sim_time: True
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_footprint"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: false
    global_frame_id: "map"
    lambda_short: 0.1
    laser_likelihood_max_dist: 2.0
    laser_max_range: 100.0
    laser_min_range: -1.0
    laser_model_type: "likelihood_field"
    max_beams: 60
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.99
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    save_pose_rate: 0.5
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.25
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05

bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: "map"
    robot_base_frame: "base_link"
    odom_topic: "/odom"
    bt_loop_duration: 10
    default_server_timeout: 20
    enable_groot_monitoring: True
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    # Specify the path to the Behavior Tree XML file
    default_bt_xml_filename: "navigate_w_replanning_and_recovery.xml"
    plugin_lib_names:
    - nav2_compute_path_to_pose_action_bt_node
    - nav2_compute_path_through_poses_action_bt_node
    - nav2_follow_path_action_bt_node
    - nav2_back_up_action_bt_node
    - nav2_spin_action_bt_node
    - nav2_wait_action_bt_node
    - nav2_clear_costmap_service_bt_node
    - nav2_is_stuck_condition_bt_node
    - nav2_goal_reached_condition_bt_node
    - nav2_goal_updated_condition_bt_node
    - nav2_initial_pose_received_condition_bt_node
    - nav2_reinitialize_global_localization_service_bt_node
    - nav2_rate_controller_bt_node
    - nav2_distance_controller_bt_node
    - nav2_speed_controller_bt_node
    - nav2_truncate_path_action_bt_node
    - nav2_goal_updater_node_bt_node
    - nav2_recovery_node_bt_node
    - nav2_pipeline_sequence_bt_node
    - nav2_round_robin_node_bt_node
    - nav2_transform_available_condition_bt_node
    - nav2_time_expired_condition_bt_node
    - nav2_path_expiring_timer_condition
    - nav2_distance_traveled_condition_bt_node
    - nav2_single_trigger_bt_node
    - nav2_is_battery_low_condition_bt_node
    - nav2_navigate_through_poses_action_bt_node
    - nav2_navigate_to_pose_action_bt_node
    - nav2_remove_passed_goals_action_bt_node
    - nav2_planner_selector_bt_node
    - nav2_controller_selector_bt_node
    - nav2_goal_checker_selector_bt_node
```

## Isaac Sim Integration

### Setting up Isaac Sim for Perception Training

Isaac Sim provides a photorealistic simulation environment for training perception systems:

1. **Install Isaac Sim** from the NVIDIA Developer Zone
2. **Launch Isaac Sim**:
   ```bash
   cd /path/to/isaac-sim
   ./python.sh
   ```

3. **Create perception training scenarios**:
   - Configure lighting conditions
   - Set up object randomization
   - Define sensor configurations
   - Generate synthetic datasets

### Synthetic Data Generation

Isaac Sim can generate large datasets for training AI perception models:

```python
# Example of setting up synthetic data generation in Isaac Sim
import omni
from omni.isaac.synthetic_utils import SyntheticDataHelper

# Configure synthetic data capture
synthetic_data_helper = SyntheticDataHelper()
synthetic_data_helper.set_camera_params(
    width=640,
    height=480,
    fov=60.0
)

# Set up randomization
synthetic_data_helper.set_randomization_params(
    texture_randomization=True,
    lighting_randomization=True,
    object_placement_randomization=True
)

# Capture synthetic data
for i in range(10000):  # Generate 10,000 images
    synthetic_data_helper.capture_frame(f"dataset/frame_{i:05d}.png")
```

## OpenCV and Computer Vision Tools

### Installing OpenCV for ROS

```bash
sudo apt install ros-humble-vision-opencv ros-humble-cv-bridge python3-opencv
```

### Basic Computer Vision Node

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ComputerVisionNode(Node):
    def __init__(self):
        super().__init__('computer_vision_node')

        # Create subscriber for camera images
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)

        # Create publisher for processed images
        self.publisher = self.create_publisher(
            Image,
            '/camera/image_processed',
            10)

        # Initialize OpenCV bridge
        self.bridge = CvBridge()

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Apply computer vision processing
            processed_image = self.process_image(cv_image)

            # Convert back to ROS Image message
            processed_msg = self.bridge.cv2_to_imgmsg(processed_image, "bgr8")
            processed_msg.header = msg.header

            # Publish processed image
            self.publisher.publish(processed_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def process_image(self, image):
        # Example: Apply Canny edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        # Convert back to 3-channel for display
        result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return result

def main(args=None):
    rclpy.init(args=args)
    node = ComputerVisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Deep Learning Integration

### Installing Deep Learning Frameworks

```bash
# Install PyTorch with CUDA support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install TensorFlow with GPU support
pip3 install tensorflow[and-cuda]
```

### Object Detection Node

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
import torch
from torchvision import transforms

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')

        # Load pre-trained model (e.g., YOLOv5)
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.eval()

        # Create subscriber and publisher
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)

        self.publisher = self.create_publisher(
            Detection2DArray,
            '/detections',
            10)

        self.bridge = CvBridge()

    def image_callback(self, msg):
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Preprocess image for model
            input_tensor = transforms.ToTensor()(cv_image).unsqueeze(0)

            # Run inference
            with torch.no_grad():
                results = self.model(input_tensor)

            # Process results
            detections_msg = self.process_detections(results, msg.header)

            # Publish detections
            self.publisher.publish(detections_msg)

        except Exception as e:
            self.get_logger().error(f'Error in object detection: {e}')

    def process_detections(self, results, header):
        detections_msg = Detection2DArray()
        detections_msg.header = header

        # Convert YOLO results to vision_msgs format
        for *xyxy, conf, cls in results.xyxy[0].tolist():
            detection = Detection2D()
            detection.header = header

            # Set bounding box
            bbox = detection.bbox
            bbox.center.x = (xyxy[0] + xyxy[2]) / 2
            bbox.center.y = (xyxy[1] + xyxy[3]) / 2
            bbox.size_x = xyxy[2] - xyxy[0]
            bbox.size_y = xyxy[3] - xyxy[1]

            # Set detection result
            result = ObjectHypothesisWithPose()
            result.hypothesis.class_id = str(int(cls))
            result.hypothesis.score = float(conf)
            detection.results.append(result)

            detections_msg.detections.append(detection)

        return detections_msg

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Performance Optimization Tools

### GPU Utilization Monitoring

```bash
# Monitor GPU usage
nvidia-smi

# Monitor in real-time
watch -n 1 nvidia-smi
```

### Isaac ROS Performance Tools

```bash
# Install Isaac ROS performance tools
sudo apt install ros-humble-isaac-ros-performance

# Use performance analyzer
ros2 run isaac_ros_performance performance_analyzer
```

## Best Practices for AI Perception Development

### 1. Modularity
- Design perception components as modular nodes
- Use composition to combine multiple algorithms
- Implement proper interfaces for easy replacement

### 2. Performance
- Use GPU acceleration where possible
- Optimize data transfer between nodes
- Implement efficient memory management

### 3. Robustness
- Handle sensor failures gracefully
- Implement fallback mechanisms
- Validate sensor data before processing

### 4. Testing
- Test with various lighting conditions
- Validate performance with different object types
- Use simulation for comprehensive testing