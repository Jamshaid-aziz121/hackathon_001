# AI Perception Implementation Walkthrough

## Creating an Object Detection and Tracking System

In this walkthrough, we'll implement a complete AI perception system that performs object detection and tracking using ROS 2 and NVIDIA Isaac tools. This system will detect objects in a camera feed and track them across multiple frames.

## Prerequisites

- ROS 2 Humble Hawksbill installed
- NVIDIA GPU with CUDA support
- Isaac ROS packages installed
- OpenCV and PyTorch installed
- Basic understanding of ROS 2 concepts

## Step 1: Create the Perception Package

First, create a new ROS 2 package for our perception system:

```bash
mkdir -p ~/ros2_ws/src/ai_perception_system
cd ~/ros2_ws/src/ai_perception_system
ros2 pkg create --build-type ament_python ai_perception_system
```

## Step 2: Install Required Dependencies

Install the required Python dependencies:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install opencv-python
pip3 install ultralytics  # For YOLO models
```

## Step 3: Create the Object Detection Node

Create the main object detection node at `ai_perception_system/ai_perception_system/object_detection_node.py`:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
from typing import List, Tuple

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')

        # Initialize model
        self.get_logger().info('Loading YOLOv5 model...')
        try:
            # Load YOLOv5 model (this will download if not present)
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            # Move model to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.to('cuda')
                self.get_logger().info('Model loaded on GPU')
            else:
                self.get_logger().info('Model loaded on CPU')
            self.model.eval()
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            raise

        # Create subscriber and publisher
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)

        self.publisher = self.create_publisher(
            Detection2DArray,
            '/object_detections',
            10)

        # Create publisher for annotated image (for visualization)
        self.annotated_publisher = self.create_publisher(
            Image,
            '/camera/image_annotated',
            10)

        # Initialize CV bridge
        self.bridge = CvBridge()

        self.get_logger().info('Object Detection Node initialized')

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
            self.publisher.publish(detections_msg)

            # Annotate image with detections
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
            # Convert image to tensor
            if torch.cuda.is_available():
                # Run inference
                results = self.model(image)
                # Convert to CPU for processing
                detections = results.pandas().xyxy[0]  # Pandas dataframe
                return detections
            else:
                # Run inference on CPU
                results = self.model(image)
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
                # Create Detection2D message
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

    def annotate_image(self, image, detections):
        """Draw bounding boxes on image"""
        annotated_image = image.copy()

        if detections is not None:
            for _, detection in detections.iterrows():
                # Extract bounding box coordinates
                xmin, ymin, xmax, ymax = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])

                # Draw bounding box
                cv2.rectangle(annotated_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                # Draw label
                label = f"{detection['name']}: {detection['confidence']:.2f}"
                cv2.putText(annotated_image, label, (xmin, ymin - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return annotated_image

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down object detection node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Step 4: Create the Object Tracking Node

Create an object tracking node at `ai_perception_system/ai_perception_system/object_tracking_node.py`:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from typing import Dict, List
import time
import math

class TrackedObject:
    """Represents a tracked object"""
    def __init__(self, detection, track_id):
        self.id = track_id
        self.last_detection = detection
        self.position = Point()
        self.position.x = detection.bbox.center.x
        self.position.y = detection.bbox.center.y
        self.position.z = 0.0
        self.class_id = detection.results[0].hypothesis.class_id if detection.results else "unknown"
        self.confidence = detection.results[0].hypothesis.score if detection.results else 0.0
        self.last_seen = time.time()
        self.velocity = Point()
        self.velocity.x = 0.0
        self.velocity.y = 0.0
        self.velocity.z = 0.0

class ObjectTrackingNode(Node):
    def __init__(self):
        super().__init__('object_tracking_node')

        # Tracking parameters
        self.max_displacement = 100  # Maximum pixel displacement for matching
        self.max_track_age = 5.0     # Maximum time (seconds) to keep track without update
        self.track_id_counter = 0    # Counter for assigning unique track IDs

        # Dictionary to store active tracks
        self.tracks: Dict[int, TrackedObject] = {}

        # Create subscriber for detections
        self.subscription = self.create_subscription(
            Detection2DArray,
            '/object_detections',
            self.detections_callback,
            10)

        # Create publisher for tracked objects
        self.publisher = self.create_publisher(
            Detection2DArray,
            '/tracked_objects',
            10)

        # Timer for cleaning up old tracks
        self.cleanup_timer = self.create_timer(1.0, self.cleanup_old_tracks)

        self.get_logger().info('Object Tracking Node initialized')

    def detections_callback(self, msg):
        """Process incoming detections and update tracks"""
        try:
            # Get current time
            current_time = time.time()

            # Convert detections to list for processing
            detections = list(msg.detections)

            # Update existing tracks or create new ones
            for detection in detections:
                matched = False
                detection_x = detection.bbox.center.x
                detection_y = detection.bbox.center.y

                # Try to match with existing tracks
                for track_id, track in self.tracks.items():
                    # Calculate distance to existing track
                    distance = math.sqrt(
                        (detection_x - track.position.x) ** 2 +
                        (detection_y - track.position.y) ** 2
                    )

                    # If close enough, update the track
                    if distance < self.max_displacement:
                        # Update track position and velocity
                        dx = detection_x - track.position.x
                        dy = detection_y - track.position.y
                        dt = current_time - track.last_seen

                        if dt > 0:
                            track.velocity.x = dx / dt
                            track.velocity.y = dy / dt

                        track.position.x = detection_x
                        track.position.y = detection_y
                        track.last_detection = detection
                        track.last_seen = current_time
                        track.confidence = detection.results[0].hypothesis.score if detection.results else 0.0

                        matched = True
                        break

                # If no match found, create a new track
                if not matched:
                    new_track = TrackedObject(detection, self.track_id_counter)
                    self.tracks[self.track_id_counter] = new_track
                    self.track_id_counter += 1

            # Create and publish tracked objects message
            tracked_msg = self.create_tracked_message(msg.header)
            self.publisher.publish(tracked_msg)

        except Exception as e:
            self.get_logger().error(f'Error in detections callback: {e}')

    def create_tracked_message(self, header):
        """Create a Detection2DArray with tracking information"""
        tracked_msg = Detection2DArray()
        tracked_msg.header = header

        for track_id, track in self.tracks.items():
            # Create a detection message for this track
            detection = Detection2D()
            detection.header = header

            # Set bounding box from last detection
            detection.bbox = track.last_detection.bbox

            # Add tracking information to results
            result = track.last_detection.results[0] if track.last_detection.results else None
            if result:
                # Add track ID to class_id
                original_class = result.hypothesis.class_id
                result.hypothesis.class_id = f"{original_class}_track_{track_id}"
                detection.results.append(result)

            tracked_msg.detections.append(detection)

        return tracked_msg

    def cleanup_old_tracks(self):
        """Remove tracks that haven't been updated for a while"""
        current_time = time.time()
        tracks_to_remove = []

        for track_id, track in self.tracks.items():
            if current_time - track.last_seen > self.max_track_age:
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.tracks[track_id]
            self.get_logger().info(f'Removed old track: {track_id}')

def main(args=None):
    rclpy.init(args=args)
    node = ObjectTrackingNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down object tracking node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Step 5: Create the Perception Pipeline Node

Create a node that orchestrates the entire perception pipeline at `ai_perception_system/ai_perception_system/perception_pipeline_node.py`:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String
from ai_perception_system.object_detection_node import ObjectDetectionNode
from ai_perception_system.object_tracking_node import ObjectTrackingNode
import threading
import time

class PerceptionPipelineNode(Node):
    def __init__(self):
        super().__init__('perception_pipeline_node')

        # Initialize component nodes
        self.detection_node = ObjectDetectionNode()
        self.tracking_node = ObjectTrackingNode()

        # Create subscriber for system status
        self.status_subscriber = self.create_subscription(
            String,
            '/perception_status',
            self.status_callback,
            10)

        # Create publisher for system status
        self.status_publisher = self.create_publisher(
            String,
            '/perception_status',
            10)

        # Start component nodes in separate threads
        self.detection_thread = threading.Thread(target=self.run_detection_node)
        self.tracking_thread = threading.Thread(target=self.run_tracking_node)

        # Start threads
        self.detection_thread.start()
        self.tracking_thread.start()

        # Publish initial status
        status_msg = String()
        status_msg.data = "Perception pipeline initialized"
        self.status_publisher.publish(status_msg)

        self.get_logger().info('Perception Pipeline Node initialized')

    def run_detection_node(self):
        """Run the detection node in a separate thread"""
        rclpy.spin(self.detection_node)

    def run_tracking_node(self):
        """Run the tracking node in a separate thread"""
        rclpy.spin(self.tracking_node)

    def status_callback(self, msg):
        """Handle status messages"""
        self.get_logger().info(f'System status: {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionPipelineNode()

    try:
        # Keep the main thread alive
        while rclpy.ok():
            time.sleep(0.1)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down perception pipeline')
    finally:
        # Shutdown component nodes
        node.detection_node.destroy_node()
        node.tracking_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Step 6: Create Launch File

Create a launch file to start the entire perception system at `ai_perception_system/launch/perception_system.launch.py`:

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

        # Object Detection Node
        Node(
            package='ai_perception_system',
            executable='object_detection_node',
            name='object_detection_node',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),

        # Object Tracking Node
        Node(
            package='ai_perception_system',
            executable='object_tracking_node',
            name='object_tracking_node',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),

        # Perception Pipeline Node (optional - orchestrates the system)
        Node(
            package='ai_perception_system',
            executable='perception_pipeline_node',
            name='perception_pipeline_node',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        )
    ])
```

## Step 7: Update setup.py

Update the `setup.py` file to make the nodes executable:

```python
from setuptools import setup
from glob import glob
import os

package_name = 'ai_perception_system'

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
    description='AI Perception System for Robotics',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'object_detection_node = ai_perception_system.object_detection_node:main',
            'object_tracking_node = ai_perception_system.object_tracking_node:main',
            'perception_pipeline_node = ai_perception_system.perception_pipeline_node:main',
        ],
    },
)
```

## Step 8: Create Configuration Files

Create a configuration file for the perception system at `ai_perception_system/config/perception_config.yaml`:

```yaml
object_detection_node:
  ros__parameters:
    # Detection parameters
    confidence_threshold: 0.5
    nms_threshold: 0.4
    max_objects: 100
    # Enable/disable GPU acceleration
    use_gpu: true
    # Model parameters
    model_name: "yolov5s"
    input_width: 640
    input_height: 480

object_tracking_node:
  ros__parameters:
    # Tracking parameters
    max_displacement: 100.0
    max_track_age: 5.0
    min_track_confidence: 0.3
    # Velocity smoothing
    velocity_smoothing_factor: 0.7
```

## Step 9: Build and Test the System

Build the package:

```bash
cd ~/ros2_ws
colcon build --packages-select ai_perception_system
source install/setup.bash
```

## Step 10: Run the Perception System

First, you'll need a camera source. If you're using a simulated robot, you can run:

```bash
# Terminal 1: Launch the perception system
ros2 launch ai_perception_system perception_system.launch.py
```

To test with a simulated camera, you can use a sample image publisher or run with a Gazebo simulation that provides camera data.

## Step 11: Visualize Results

Visualize the results using RViz2:

```bash
# Terminal 2: Launch RViz2
rviz2
```

In RViz2, add displays for:
- Image display for `/camera/image_annotated` to see detections
- MarkerArray display for tracked objects (if you create a visualization node)

## Step 12: Create a Visualization Node

Create a visualization node to show tracked objects as markers at `ai_perception_system/ai_perception_system/tracking_visualizer.py`:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

class TrackingVisualizer(Node):
    def __init__(self):
        super().__init__('tracking_visualizer')

        # Create subscriber for tracked objects
        self.subscription = self.create_subscription(
            Detection2DArray,
            '/tracked_objects',
            self.tracked_objects_callback,
            10)

        # Create publisher for visualization markers
        self.marker_publisher = self.create_publisher(
            MarkerArray,
            '/tracking_markers',
            10)

        self.get_logger().info('Tracking Visualizer initialized')

    def tracked_objects_callback(self, msg):
        """Process tracked objects and create visualization markers"""
        try:
            marker_array = MarkerArray()

            for i, detection in enumerate(msg.detections):
                # Create marker for bounding box
                bbox_marker = Marker()
                bbox_marker.header = msg.header
                bbox_marker.ns = "tracking"
                bbox_marker.id = i * 2  # Use even IDs for bounding boxes
                bbox_marker.type = Marker.LINE_STRIP
                bbox_marker.action = Marker.ADD

                # Set position and scale
                center_x = detection.bbox.center.x
                center_y = detection.bbox.center.y
                size_x = detection.bbox.size_x
                size_y = detection.bbox.size_y

                # Define rectangle points
                half_x = size_x / 2
                half_y = size_y / 2

                p1 = Point(x=center_x - half_x, y=center_y - half_y, z=0.0)
                p2 = Point(x=center_x + half_x, y=center_y - half_y, z=0.0)
                p3 = Point(x=center_x + half_x, y=center_y + half_y, z=0.0)
                p4 = Point(x=center_x - half_x, y=center_y + half_y, z=0.0)

                bbox_marker.points = [p1, p2, p3, p4, p1]  # Close the rectangle

                # Set color (green for tracked objects)
                bbox_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
                bbox_marker.scale.x = 0.02  # Line width

                # Create text marker for track ID
                text_marker = Marker()
                text_marker.header = msg.header
                text_marker.ns = "tracking_labels"
                text_marker.id = i * 2 + 1  # Use odd IDs for labels
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD

                # Position text above bounding box
                text_marker.pose.position.x = center_x
                text_marker.pose.position.y = center_y - half_y - 10  # Above the box
                text_marker.pose.position.z = 0.0
                text_marker.pose.orientation.w = 1.0

                # Extract track ID from class_id if available
                if detection.results:
                    class_id = detection.results[0].hypothesis.class_id
                    if "_track_" in class_id:
                        track_id = class_id.split("_track_")[-1]
                        text_marker.text = f"ID: {track_id}"
                    else:
                        text_marker.text = class_id
                else:
                    text_marker.text = f"Track {i}"

                # Set text properties
                text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
                text_marker.scale.z = 10.0  # Text height

                # Add markers to array
                marker_array.markers.append(bbox_marker)
                marker_array.markers.append(text_marker)

            # Publish marker array
            self.marker_publisher.publish(marker_array)

        except Exception as e:
            self.get_logger().error(f'Error in tracked objects callback: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = TrackingVisualizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down tracking visualizer')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Add the visualizer to the launch file and setup.py:

Update setup.py to include the visualizer:

```python
entry_points={
    'console_scripts': [
        'object_detection_node = ai_perception_system.object_detection_node:main',
        'object_tracking_node = ai_perception_system.object_tracking_node:main',
        'perception_pipeline_node = ai_perception_system.perception_pipeline_node:main',
        'tracking_visualizer = ai_perception_system.tracking_visualizer:main',
    ],
},
```

Update the launch file to include the visualizer:

```python
# Add this to the launch file
Node(
    package='ai_perception_system',
    executable='tracking_visualizer',
    name='tracking_visualizer',
    parameters=[{'use_sim_time': use_sim_time}],
    output='screen'
),
```

## Verification

Your perception system should now:
1. Detect objects in camera images using deep learning
2. Track objects across multiple frames
3. Publish detection and tracking information
4. Visualize tracked objects with bounding boxes and IDs
5. Handle multiple objects simultaneously