# Mini Project: Smart Object Sorting Robot with AI Perception

## Project Overview

In this mini project, you'll build a smart object sorting robot that uses AI perception to identify, classify, and sort objects into different categories. This project combines computer vision, machine learning, and robotic manipulation to create an intelligent sorting system.

## Learning Objectives

- Implement real-time object detection and classification
- Integrate perception with robotic manipulation
- Create a complete perception-action pipeline
- Handle multiple object types with different sorting rules
- Implement safety and error handling in perception systems

## Prerequisites

- ROS 2 Humble Hawksbill installed
- Basic knowledge of computer vision
- Understanding of robotic manipulation concepts
- Experience with camera and sensor integration
- NVIDIA Isaac ROS packages (optional for GPU acceleration)

## Project Requirements

- Detect and classify objects in real-time
- Sort objects into at least 3 categories (e.g., colors, shapes, materials)
- Handle multiple objects simultaneously
- Implement safety mechanisms to prevent errors
- Provide feedback on sorting decisions

## Step-by-Step Implementation

### Step 1: Create the Project Package

Create a new ROS 2 package for the sorting robot:

```bash
mkdir -p ~/ros2_ws/src/smart_sorting_robot
cd ~/ros2_ws/src/smart_sorting_robot
ros2 pkg create --build-type ament_python smart_sorting_robot
```

### Step 2: Install Dependencies

Install the required Python dependencies:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install ultralytics  # For YOLO models
pip3 install opencv-python
pip3 install numpy
```

### Step 3: Create the Object Detection Node

Create the main object detection node at `smart_sorting_robot/smart_sorting_robot/object_detector.py`:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
from collections import defaultdict

class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector')

        # Initialize YOLO model
        self.get_logger().info('Loading object detection model...')
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            if torch.cuda.is_available():
                self.model = self.model.to('cuda')
                self.get_logger().info('Model loaded on GPU')
            else:
                self.get_logger().info('Model loaded on CPU')
            self.model.eval()
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            raise

        # Object categories for sorting
        self.sorting_categories = {
            'bottle': 'recycle',
            'cup': 'recycle',
            'person': 'ignore',
            'chair': 'ignore',
            'book': 'library',
            'cell phone': 'electronics',
            'laptop': 'electronics',
            'backpack': 'storage',
            'umbrella': 'storage',
            'orange': 'food',
            'apple': 'food',
            'banana': 'food',
            'pizza': 'food',
            'donut': 'food'
        }

        # Create subscriber and publishers
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)

        self.detection_publisher = self.create_publisher(
            Detection2DArray,
            '/object_detections',
            10)

        self.sorting_publisher = self.create_publisher(
            String,
            '/object_sorting_commands',
            10)

        self.annotated_publisher = self.create_publisher(
            Image,
            '/camera/image_annotated',
            10)

        # Initialize CV bridge
        self.bridge = CvBridge()

        self.get_logger().info('Object Detector initialized')

    def image_callback(self, msg):
        """Process incoming camera image"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Perform object detection
            detections = self.detect_objects(cv_image)

            # Create Detection2DArray message
            detections_msg = self.create_detections_message(detections, msg.header)

            # Publish detections
            self.detection_publisher.publish(detections_msg)

            # Process detections for sorting
            self.process_detections_for_sorting(detections, msg.header)

            # Annotate image with detections and sorting info
            annotated_image = self.annotate_image(cv_image, detections)

            # Publish annotated image
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
            annotated_msg.header = msg.header
            self.annotated_publisher.publish(annotated_msg)

        except Exception as e:
            self.get_logger().error(f'Error in image callback: {e}')

    def detect_objects(self, image):
        """Run object detection on image"""
        try:
            # Run inference
            results = self.model(image)
            # Convert to pandas dataframe
            detections = results.pandas().xyxy[0]
            return detections
        except Exception as e:
            self.get_logger().error(f'Error in detection: {e}')
            return None

    def create_detections_message(self, detections, header):
        """Convert detection results to ROS message"""
        detections_msg = Detection2DArray()
        detections_msg.header = header

        if detections is not None:
            for _, detection in detections.iterrows():
                detection_msg = Detection2D()
                detection_msg.header = header

                # Set bounding box
                bbox = detection_msg.bbox
                bbox.center.x = (detection['xmin'] + detection['xmax']) / 2
                bbox.center.y = (detection['ymin'] + detection['ymax']) / 2
                bbox.size_x = detection['xmax'] - detection['xmin']
                bbox.size_y = detection['ymax'] - detection['ymin']

                # Set detection result
                result = ObjectHypothesisWithPose()
                result.hypothesis.class_id = str(detection['name'])
                result.hypothesis.score = float(detection['confidence'])
                detection_msg.results.append(result)

                detections_msg.detections.append(detection_msg)

        return detections_msg

    def process_detections_for_sorting(self, detections, header):
        """Process detections and determine sorting actions"""
        if detections is not None:
            for _, detection in detections.iterrows():
                class_name = detection['name']
                confidence = detection['confidence']

                # Only process high-confidence detections
                if confidence > 0.5:
                    # Determine sorting category
                    sort_category = self.sorting_categories.get(class_name, 'unknown')

                    # Only publish commands for valid categories
                    if sort_category != 'ignore':
                        command_msg = String()
                        command_msg.data = f"SORT {class_name} TO {sort_category} AT ({detection['xmin']:.1f},{detection['ymin']:.1f})"
                        self.sorting_publisher.publish(command_msg)

                        self.get_logger().info(f'Sorting command: {command_msg.data}')

    def annotate_image(self, image, detections):
        """Draw bounding boxes and sorting info on image"""
        annotated_image = image.copy()

        if detections is not None:
            for _, detection in detections.iterrows():
                class_name = detection['name']
                confidence = detection['confidence']
                sort_category = self.sorting_categories.get(class_name, 'unknown')

                # Extract bounding box coordinates
                xmin, ymin, xmax, ymax = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])

                # Choose color based on sort category
                if sort_category == 'recycle':
                    color = (0, 255, 0)  # Green
                elif sort_category == 'electronics':
                    color = (255, 0, 0)  # Blue
                elif sort_category == 'food':
                    color = (0, 0, 255)  # Red
                elif sort_category == 'library':
                    color = (255, 255, 0)  # Cyan
                elif sort_category == 'storage':
                    color = (255, 0, 255)  # Magenta
                else:
                    color = (128, 128, 128)  # Gray for unknown

                # Draw bounding box
                cv2.rectangle(annotated_image, (xmin, ymin), (xmax, ymax), color, 2)

                # Draw label and confidence
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(annotated_image, label, (xmin, ymin - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Draw sort category
                if sort_category != 'ignore':
                    cv2.putText(annotated_image, f"-> {sort_category}", (xmin, ymin - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return annotated_image

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down object detector')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 4: Create the Sorting Controller Node

Create the sorting controller at `smart_sorting_robot/smart_sorting_robot/sorting_controller.py`:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Point, Pose
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import time
import math

class SortingController(Node):
    def __init__(self):
        super().__init__('sorting_controller')

        # Sorting station coordinates (simulated)
        self.sorting_stations = {
            'recycle': Point(x=1.0, y=0.0, z=0.0),
            'electronics': Point(x=0.0, y=1.0, z=0.0),
            'food': Point(x=-1.0, y=0.0, z=0.0),
            'library': Point(x=0.0, y=-1.0, z=0.0),
            'storage': Point(x=0.7, y=0.7, z=0.0)
        }

        # Current robot state
        self.robot_position = Point(x=0.0, y=0.0, z=0.0)
        self.object_detected = False
        self.current_object = None
        self.current_object_position = None

        # Create subscriber for sorting commands
        self.command_subscriber = self.create_subscription(
            String,
            '/object_sorting_commands',
            self.command_callback,
            10)

        # Publishers for robot control
        self.joint_command_publisher = self.create_publisher(
            Float64MultiArray,
            '/joint_commands',
            10)

        self.gripper_command_publisher = self.create_publisher(
            Float64MultiArray,
            '/gripper_commands',
            10)

        # Timer for robot state updates
        self.state_timer = self.create_timer(0.1, self.update_robot_state)

        self.get_logger().info('Sorting Controller initialized')

    def command_callback(self, msg):
        """Process sorting commands"""
        try:
            command = msg.data
            self.get_logger().info(f'Received command: {command}')

            # Parse command
            parts = command.split()
            if len(parts) >= 4 and parts[0] == 'SORT':
                object_name = parts[1]
                sort_category = parts[3]

                # Extract position from command
                pos_str = ' '.join(parts[4:])
                pos_str = pos_str.replace('AT (', '').replace(')', '')
                pos_parts = pos_str.split(',')
                if len(pos_parts) >= 2:
                    x, y = float(pos_parts[0]), float(pos_parts[1])
                    object_position = Point(x=x, y=y, z=0.0)

                    # Execute sorting task
                    self.execute_sorting_task(object_name, sort_category, object_position)

        except Exception as e:
            self.get_logger().error(f'Error processing command: {e}')

    def execute_sorting_task(self, object_name, sort_category, object_position):
        """Execute a complete sorting task"""
        try:
            self.get_logger().info(f'Starting sorting task: {object_name} to {sort_category}')

            # Check if sort category is valid
            if sort_category not in self.sorting_stations:
                self.get_logger().warn(f'Unknown sort category: {sort_category}')
                return

            # Move to object position
            self.move_to_position(object_position)

            # Pick up object
            self.pick_up_object()

            # Move to sorting station
            station_position = self.sorting_stations[sort_category]
            self.move_to_position(station_position)

            # Place object
            self.place_object()

            # Return to home position
            home_position = Point(x=0.0, y=0.0, z=0.0)
            self.move_to_position(home_position)

            self.get_logger().info(f'Completed sorting task: {object_name} to {sort_category}')

        except Exception as e:
            self.get_logger().error(f'Error in sorting task: {e}')

    def move_to_position(self, target_position):
        """Move robot to target position (simulated)"""
        self.get_logger().info(f'Moving to position: ({target_position.x:.2f}, {target_position.y:.2f})')

        # Simulate movement
        # In a real robot, this would involve path planning and navigation
        time.sleep(1.0)  # Simulate movement time

        # Update robot position
        self.robot_position = target_position

    def pick_up_object(self):
        """Pick up object (simulated)"""
        self.get_logger().info('Picking up object')

        # Simulate gripper control
        gripper_cmd = Float64MultiArray()
        gripper_cmd.data = [0.0]  # Close gripper
        self.gripper_command_publisher.publish(gripper_cmd)

        time.sleep(0.5)  # Simulate pick time

    def place_object(self):
        """Place object (simulated)"""
        self.get_logger().info('Placing object')

        # Simulate gripper control
        gripper_cmd = Float64MultiArray()
        gripper_cmd.data = [1.0]  # Open gripper
        self.gripper_command_publisher.publish(gripper_cmd)

        time.sleep(0.5)  # Simulate place time

    def update_robot_state(self):
        """Update robot state (simulated)"""
        # In a real robot, this would update from actual sensors
        pass

def main(args=None):
    rclpy.init(args=args)
    node = SortingController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down sorting controller')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 5: Create the Perception Quality Monitor

Create a quality monitor for the perception system at `smart_sorting_robot/smart_sorting_robot/perception_quality_monitor.py`:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String, Float32
from sensor_msgs.msg import Image
import time
from collections import deque

class PerceptionQualityMonitor(Node):
    def __init__(self):
        super().__init__('perception_quality_monitor')

        # Quality metrics
        self.detection_rate = 0.0
        self.confidence_threshold = 0.5
        self.frame_count = 0
        self.detection_count = 0
        self.quality_score = 1.0

        # Time tracking
        self.last_frame_time = time.time()
        self.frame_times = deque(maxlen=10)  # Track last 10 frame times

        # Create subscribers
        self.detection_subscriber = self.create_subscription(
            Detection2DArray,
            '/object_detections',
            self.detection_callback,
            10)

        self.image_subscriber = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)

        # Create publishers for quality metrics
        self.quality_publisher = self.create_publisher(
            String,
            '/perception_quality_status',
            10)

        self.fps_publisher = self.create_publisher(
            Float32,
            '/perception_fps',
            10)

        self.confidence_publisher = self.create_publisher(
            Float32,
            '/perception_avg_confidence',
            10)

        # Timer for quality assessment
        self.quality_timer = self.create_timer(1.0, self.assess_quality)

        self.get_logger().info('Perception Quality Monitor initialized')

    def image_callback(self, msg):
        """Track frame rate"""
        current_time = time.time()
        if self.last_frame_time != 0:
            frame_time = current_time - self.last_frame_time
            self.frame_times.append(frame_time)

        self.last_frame_time = current_time
        self.frame_count += 1

    def detection_callback(self, msg):
        """Process detections and assess quality"""
        try:
            high_conf_detections = 0
            total_confidence = 0.0

            for detection in msg.detections:
                if detection.results:
                    confidence = detection.results[0].hypothesis.score
                    total_confidence += confidence

                    if confidence > self.confidence_threshold:
                        high_conf_detections += 1

            self.detection_count += len(msg.detections)

            # Calculate average confidence
            if msg.detections:
                avg_confidence = total_confidence / len(msg.detections)
                conf_msg = Float32()
                conf_msg.data = avg_confidence
                self.confidence_publisher.publish(conf_msg)

        except Exception as e:
            self.get_logger().error(f'Error in detection callback: {e}')

    def assess_quality(self):
        """Assess and publish perception quality metrics"""
        try:
            # Calculate FPS
            if self.frame_times:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

                fps_msg = Float32()
                fps_msg.data = float(fps)
                self.fps_publisher.publish(fps_msg)

                # Calculate detection rate
                if self.frame_count > 0:
                    detection_rate = self.detection_count / self.frame_count
                else:
                    detection_rate = 0.0

                # Calculate quality score based on multiple factors
                quality_score = self.calculate_quality_score(fps, detection_rate)

                # Publish quality status
                status_msg = String()
                status_msg.data = f"FPS: {fps:.1f}, Detection Rate: {detection_rate:.2f}, Quality: {quality_score:.2f}"
                self.quality_publisher.publish(status_msg)

                self.get_logger().info(f'Perception Quality: {status_msg.data}')

                # Reset counters
                self.frame_count = 0
                self.detection_count = 0

        except Exception as e:
            self.get_logger().error(f'Error in quality assessment: {e}')

    def calculate_quality_score(self, fps, detection_rate):
        """Calculate overall perception quality score"""
        # Weight different factors
        fps_weight = 0.4
        detection_rate_weight = 0.3
        stability_weight = 0.3

        # FPS score (good performance is > 10 FPS)
        fps_score = min(fps / 10.0, 1.0)

        # Detection rate score (good rate is > 0.5)
        detection_score = min(detection_rate / 0.5, 1.0)

        # Stability score based on frame time variance
        if len(self.frame_times) > 1:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            variance = sum((t - avg_time) ** 2 for t in self.frame_times) / len(self.frame_times)
            stability_score = max(0.0, 1.0 - variance * 100)  # Lower variance = higher score
        else:
            stability_score = 1.0

        # Calculate weighted average
        quality_score = (
            fps_score * fps_weight +
            detection_score * detection_rate_weight +
            stability_score * stability_weight
        )

        return quality_score

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionQualityMonitor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down perception quality monitor')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 6: Create the Main Control Node

Create the main control node that orchestrates the entire system at `smart_sorting_robot/smart_sorting_robot/sorting_robot_control.py`:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image
import threading
import time
from smart_sorting_robot.object_detector import ObjectDetector
from smart_sorting_robot.sorting_controller import SortingController
from smart_sorting_robot.perception_quality_monitor import PerceptionQualityMonitor

class SortingRobotControl(Node):
    def __init__(self):
        super().__init__('sorting_robot_control')

        # Initialize component nodes
        self.object_detector = ObjectDetector()
        self.sorting_controller = SortingController()
        self.quality_monitor = PerceptionQualityMonitor()

        # System state
        self.system_active = True
        self.error_count = 0
        self.max_errors = 5

        # Create subscribers
        self.status_subscriber = self.create_subscription(
            String,
            '/system_status',
            self.status_callback,
            10)

        self.error_subscriber = self.create_subscription(
            String,
            '/system_error',
            self.error_callback,
            10)

        # Create publisher for system status
        self.status_publisher = self.create_publisher(
            String,
            '/system_status',
            10)

        # Create publisher for system active state
        self.active_publisher = self.create_publisher(
            Bool,
            '/system_active',
            10)

        # Start component nodes in separate threads
        self.detector_thread = threading.Thread(target=self.run_detector)
        self.controller_thread = threading.Thread(target=self.run_controller)
        self.monitor_thread = threading.Thread(target=self.run_monitor)

        # Start threads
        self.detector_thread.start()
        self.controller_thread.start()
        self.monitor_thread.start()

        # Timer for system health checks
        self.health_timer = self.create_timer(2.0, self.system_health_check)

        # Publish initial status
        status_msg = String()
        status_msg.data = "Sorting robot system initialized and running"
        self.status_publisher.publish(status_msg)

        self.get_logger().info('Sorting Robot Control initialized')

    def run_detector(self):
        """Run the object detector in a separate thread"""
        rclpy.spin(self.object_detector)

    def run_controller(self):
        """Run the sorting controller in a separate thread"""
        rclpy.spin(self.sorting_controller)

    def run_monitor(self):
        """Run the quality monitor in a separate thread"""
        rclpy.spin(self.quality_monitor)

    def status_callback(self, msg):
        """Handle status messages from components"""
        self.get_logger().info(f'Component status: {msg.data}')

    def error_callback(self, msg):
        """Handle error messages"""
        self.error_count += 1
        self.get_logger().error(f'System error: {msg.data}')

        if self.error_count >= self.max_errors:
            self.get_logger().error('Too many errors, shutting down system')
            self.system_active = False

    def system_health_check(self):
        """Perform periodic system health checks"""
        try:
            # Check if system should remain active
            active_msg = Bool()
            active_msg.data = self.system_active
            self.active_publisher.publish(active_msg)

            if not self.system_active:
                self.get_logger().warn('System is not active')

        except Exception as e:
            self.get_logger().error(f'Error in health check: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = SortingRobotControl()

    try:
        # Keep the main thread alive
        while rclpy.ok() and node.system_active:
            time.sleep(0.1)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down sorting robot control')
    finally:
        # Shutdown component nodes
        node.object_detector.destroy_node()
        node.sorting_controller.destroy_node()
        node.quality_monitor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 7: Create Launch File

Create a launch file to start the entire sorting system at `smart_sorting_robot/launch/sorting_robot.launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'
        ),

        # Object Detector Node
        Node(
            package='smart_sorting_robot',
            executable='object_detector',
            name='object_detector',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),

        # Sorting Controller Node
        Node(
            package='smart_sorting_robot',
            executable='sorting_controller',
            name='sorting_controller',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),

        # Perception Quality Monitor Node
        Node(
            package='smart_sorting_robot',
            executable='perception_quality_monitor',
            name='perception_quality_monitor',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),

        # Main Control Node
        Node(
            package='smart_sorting_robot',
            executable='sorting_robot_control',
            name='sorting_robot_control',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        )
    ])
```

### Step 8: Update setup.py

Update the `setup.py` file to make the nodes executable:

```python
from setuptools import setup
from glob import glob
import os

package_name = 'smart_sorting_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Smart Object Sorting Robot with AI Perception',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'object_detector = smart_sorting_robot.object_detector:main',
            'sorting_controller = smart_sorting_robot.sorting_controller:main',
            'perception_quality_monitor = smart_sorting_robot.perception_quality_monitor:main',
            'sorting_robot_control = smart_sorting_robot.sorting_robot_control:main',
        ],
    },
)
```

### Step 9: Build the Package

Build the package:

```bash
cd ~/ros2_ws
colcon build --packages-select smart_sorting_robot
source install/setup.bash
```

### Step 10: Run the Sorting Robot System

Run the complete sorting robot system:

```bash
# Terminal 1: Launch the sorting robot system
ros2 launch smart_sorting_robot sorting_robot.launch.py
```

To test the system, you'll need to provide camera images. If using a simulated environment:

```bash
# Terminal 2: Publish test images (if available)
# This would depend on your specific camera setup or simulation
```

### Step 11: Monitor System Performance

Monitor the system performance using various topics:

```bash
# Terminal 3: Monitor sorting commands
ros2 topic echo /object_sorting_commands

# Terminal 4: Monitor perception quality
ros2 topic echo /perception_quality_status

# Terminal 5: Monitor FPS
ros2 topic echo /perception_fps

# Terminal 6: Visualize annotated images
# Use RViz2 to view /camera/image_annotated
```

## Expected Results

When running the complete system, you should see:

1. Object detection working in real-time
2. Objects being classified and assigned to sorting categories
3. Sorting commands being generated and published
4. Quality metrics being monitored and reported
5. The system maintaining stable performance

## Extensions

Try these extensions to enhance your sorting robot:

1. **Improved Classification**: Train a custom model for specific objects in your environment
2. **3D Object Sorting**: Add depth perception for more accurate sorting
3. **Multiple Robot Coordination**: Implement multiple robots working together
4. **Learning System**: Add reinforcement learning to improve sorting decisions
5. **Quality Control**: Implement verification of sorting accuracy
6. **Error Recovery**: Add robust error handling and recovery mechanisms

## Evaluation Criteria

Your implementation should:
- Accurately detect and classify objects in real-time
- Generate appropriate sorting commands
- Maintain stable performance metrics
- Handle multiple objects simultaneously
- Provide feedback on system quality
- Include proper error handling and safety measures