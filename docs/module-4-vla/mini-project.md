# Mini Project: Voice-Controlled Educational Robot Assistant

## Project Overview

In this mini project, you'll build a voice-controlled educational robot assistant that can understand natural language commands, perceive its environment, and execute educational tasks. The robot will serve as an interactive teaching aid for basic STEM concepts.

## Learning Objectives

- Integrate speech recognition with robotic action
- Combine vision processing with language understanding
- Implement a complete VLA pipeline
- Create an interactive educational interface
- Handle multiple input modalities simultaneously

## Prerequisites

- ROS 2 Humble Hawksbill installed
- Microphone and camera for the robot
- Basic understanding of speech recognition
- Experience with computer vision
- Knowledge of robotic manipulation (if available)

## Project Requirements

- Accept voice commands from users
- Process visual information from camera
- Execute appropriate educational actions
- Provide feedback through speech or text
- Maintain safety and ethical standards
- Demonstrate basic STEM concepts

## Step-by-Step Implementation

### Step 1: Create the Project Package

Create a new ROS 2 package for the educational robot:

```bash
mkdir -p ~/ros2_ws/src/voice_controlled_robot
cd ~/ros2_ws/src/voice_controlled_robot
ros2 pkg create --build-type ament_python voice_controlled_robot
```

### Step 2: Install Dependencies

Create a requirements file:

**voice_controlled_robot/requirements.txt**:
```
torch>=1.12.0
torchvision>=0.13.0
torchaudio>=0.12.0
openai-whisper
transformers>=4.21.0
opencv-python>=4.5.0
numpy>=1.21.0
SpeechRecognition
pyaudio
```

Install dependencies:
```bash
pip3 install -r requirements.txt
```

### Step 3: Create the Voice Command Parser

Create a node to parse voice commands:

**voice_controlled_robot/voice_controlled_robot/voice_command_parser.py**:
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import re

class VoiceCommandParser(Node):
    def __init__(self):
        super().__init__('voice_command_parser')

        # Define command patterns
        self.command_patterns = {
            # Educational commands
            r'.*show me.*gravity.*|.*demonstrate.*gravity.*': 'DEMONSTRATE_GRAVITY',
            r'.*show me.*friction.*|.*demonstrate.*friction.*': 'DEMONSTRATE_FRICTION',
            r'.*show me.*geometry.*|.*draw.*shape.*': 'DEMONSTRATE_GEOMETRY',
            r'.*what.*color.*': 'QUERY_COLOR',
            r'.*how many.*': 'QUERY_COUNT',
            r'.*hello.*|.*hi.*robot.*': 'GREETING',
            r'.*bye.*|.*goodbye.*': 'FAREWELL',

            # Movement commands
            r'.*move forward.*|.*go forward.*': 'MOVE_FORWARD',
            r'.*move backward.*|.*go backward.*': 'MOVE_BACKWARD',
            r'.*turn left.*': 'TURN_LEFT',
            r'.*turn right.*': 'TURN_RIGHT',
            r'.*stop.*': 'STOP',

            # Object interaction
            r'.*pick up.*|.*grasp.*': 'GRASP_OBJECT',
            r'.*put down.*|.*release.*': 'RELEASE_OBJECT',
            r'.*find.*|.*look for.*': 'SEARCH_OBJECT'
        }

        # Create subscriber for transcribed text
        self.text_subscriber = self.create_subscription(
            String,
            '/transcribed_text',
            self.text_callback,
            10)

        # Create publisher for parsed commands
        self.command_publisher = self.create_publisher(
            String,
            '/parsed_voice_commands',
            10)

        self.get_logger().info('Voice Command Parser initialized')

    def text_callback(self, msg):
        """Process incoming text and parse commands"""
        try:
            text = msg.data.lower().strip()
            self.get_logger().info(f'Received voice command: {text}')

            # Parse command using patterns
            parsed_command = self.parse_command(text)

            if parsed_command != 'UNKNOWN':
                # Publish parsed command
                command_msg = String()
                command_msg.data = parsed_command
                self.command_publisher.publish(command_msg)

                self.get_logger().info(f'Parsed command: {parsed_command}')
            else:
                self.get_logger().info('Unknown command received')

        except Exception as e:
            self.get_logger().error(f'Error in text callback: {e}')

    def parse_command(self, text):
        """Parse text command using regex patterns"""
        for pattern, command in self.command_patterns.items():
            if re.search(pattern, text):
                return command

        return 'UNKNOWN'

def main(args=None):
    rclpy.init(args=args)
    node = VoiceCommandParser()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down voice command parser')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 4: Create the Vision Processing Node

Create a node to process visual information:

**voice_controlled_robot/voice_controlled_robot/vision_processor.py**:
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import json

class VisionProcessor(Node):
    def __init__(self):
        super().__init__('vision_processor')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Object detection parameters
        self.object_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        # Create subscriber for camera images
        self.image_subscriber = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)

        # Create publisher for detected objects
        self.object_publisher = self.create_publisher(
            String,
            '/detected_objects',
            10)

        # Create publisher for scene analysis
        self.scene_publisher = self.create_publisher(
            String,
            '/scene_analysis',
            10)

        self.get_logger().info('Vision Processor initialized')

    def image_callback(self, msg):
        """Process incoming camera image"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Perform basic object detection (using color detection as a simple example)
            detected_objects = self.detect_objects(cv_image)

            # Publish detected objects
            objects_msg = String()
            objects_msg.data = json.dumps(detected_objects)
            self.object_publisher.publish(objects_msg)

            # Perform scene analysis
            scene_analysis = self.analyze_scene(cv_image, detected_objects)

            # Publish scene analysis
            scene_msg = String()
            scene_msg.data = scene_analysis
            self.scene_publisher.publish(scene_msg)

            self.get_logger().info(f'Detected {len(detected_objects)} objects')

        except Exception as e:
            self.get_logger().error(f'Error in image callback: {e}')

    def detect_objects(self, image):
        """Detect objects in the image (simplified color-based detection)"""
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Define color ranges for common objects
            color_ranges = {
                'red': ([0, 50, 50], [10, 255, 255]),
                'green': ([40, 50, 50], [80, 255, 255]),
                'blue': ([100, 50, 50], [130, 255, 255]),
                'yellow': ([20, 50, 50], [40, 255, 255]),
                'purple': ([130, 50, 50], [160, 255, 255]),
                'orange': ([10, 50, 50], [20, 255, 255])
            }

            detected_objects = []

            for color_name, (lower, upper) in color_ranges.items():
                # Create mask for color
                lower = np.array(lower)
                upper = np.array(upper)
                mask = cv2.inRange(hsv, lower, upper)

                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Filter contours by size
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 500:  # Minimum area threshold
                        # Get bounding box
                        x, y, w, h = cv2.boundingRect(contour)

                        obj_info = {
                            'label': f'{color_name}_object',
                            'confidence': 0.8,  # Simplified confidence
                            'bbox': [x, y, x+w, y+h],
                            'area': area
                        }
                        detected_objects.append(obj_info)

            return detected_objects

        except Exception as e:
            self.get_logger().error(f'Error in object detection: {e}')
            return []

    def analyze_scene(self, image, objects):
        """Analyze the scene and provide description"""
        try:
            height, width = image.shape[:2]

            # Count objects
            object_count = len(objects)

            # Identify main colors
            color_count = {}
            for obj in objects:
                color = obj['label'].split('_')[0]  # Extract color from label
                color_count[color] = color_count.get(color, 0) + 1

            # Create scene description
            main_colors = [color for color, count in sorted(color_count.items(), key=lambda x: x[1], reverse=True)[:3]]

            description = f"Scene contains {object_count} color objects: {', '.join(main_colors)}"

            return description

        except Exception as e:
            self.get_logger().error(f'Error in scene analysis: {e}')
            return "Scene analysis failed"

def main(args=None):
    rclpy.init(args=args)
    node = VisionProcessor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down vision processor')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 5: Create the VLA Integration Node

Create the main node that integrates voice, vision, and action:

**voice_controlled_robot/voice_controlled_robot/vla_integrator.py**:
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import json
import time

class VLAIntegrator(Node):
    def __init__(self):
        super().__init__('vla_integrator')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # System state
        self.last_voice_command = None
        self.last_scene_analysis = None
        self.last_detected_objects = []
        self.command_received_time = None

        # Create subscribers
        self.voice_command_subscriber = self.create_subscription(
            String,
            '/parsed_voice_commands',
            self.voice_command_callback,
            10)

        self.scene_subscriber = self.create_subscription(
            String,
            '/scene_analysis',
            self.scene_callback,
            10)

        self.object_subscriber = self.create_subscription(
            String,
            '/detected_objects',
            self.object_callback,
            10)

        # Create publishers
        self.action_publisher = self.create_publisher(
            String,
            '/robot_actions',
            10)

        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10)

        self.feedback_publisher = self.create_publisher(
            String,
            '/robot_feedback',
            10)

        # Timer for action execution
        self.action_timer = self.create_timer(0.5, self.execute_action)

        self.get_logger().info('VLA Integrator initialized')

    def voice_command_callback(self, msg):
        """Process voice commands"""
        try:
            command = msg.data
            self.last_voice_command = command
            self.command_received_time = time.time()

            self.get_logger().info(f'Received voice command: {command}')

            # Provide immediate feedback
            feedback_msg = String()
            feedback_msg.data = f"Received command: {command}"
            self.feedback_publisher.publish(feedback_msg)

        except Exception as e:
            self.get_logger().error(f'Error in voice command callback: {e}')

    def scene_callback(self, msg):
        """Process scene analysis"""
        try:
            self.last_scene_analysis = msg.data
        except Exception as e:
            self.get_logger().error(f'Error in scene callback: {e}')

    def object_callback(self, msg):
        """Process detected objects"""
        try:
            self.last_detected_objects = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f'Error in object callback: {e}')
            self.last_detected_objects = []

    def execute_action(self):
        """Execute integrated VLA action"""
        try:
            if self.last_voice_command and self.last_scene_analysis:
                # Integrate voice command with visual information
                action = self.integrate_vla()

                if action:
                    # Publish action
                    action_msg = String()
                    action_msg.data = action
                    self.action_publisher.publish(action_msg)

                    # Execute if it's a movement command
                    if "MOVE" in action or "TURN" in action:
                        twist_cmd = self.create_movement_command(action)
                        if twist_cmd:
                            self.cmd_vel_publisher.publish(twist_cmd)

                    # Provide feedback
                    feedback_msg = String()
                    feedback_msg.data = f"Executing: {action}"
                    self.feedback_publisher.publish(feedback_msg)

                    self.get_logger().info(f'Executing action: {action}')

                    # Clear command after execution
                    self.last_voice_command = None

        except Exception as e:
            self.get_logger().error(f'Error in action execution: {e}')

    def integrate_vla(self):
        """Integrate voice command with visual information"""
        command = self.last_voice_command
        scene = self.last_scene_analysis
        objects = self.last_detected_objects

        # Educational demonstrations
        if command == 'DEMONSTRATE_GRAVITY':
            return "DEMONSTRATE_GRAVITY: Moving to clear area and preparing demonstration"

        elif command == 'DEMONSTRATE_FRICTION':
            return "DEMONSTRATE_FRICTION: Looking for surfaces to demonstrate friction"

        elif command == 'DEMONSTRATE_GEOMETRY':
            return "DEMONSTRATE_GEOMETRY: Drawing geometric shapes"

        elif command == 'QUERY_COLOR':
            if objects:
                # Find most common color
                color_count = {}
                for obj in objects:
                    color = obj['label'].split('_')[0]
                    color_count[color] = color_count.get(color, 0) + 1

                if color_count:
                    most_common_color = max(color_count, key=color_count.get)
                    return f"QUERY_COLOR_RESPONSE: I see mostly {most_common_color} objects in the scene"
            return "QUERY_COLOR_RESPONSE: I don't see any colored objects right now"

        elif command == 'QUERY_COUNT':
            return f"QUERY_COUNT_RESPONSE: I see {len(objects)} objects in the scene"

        elif command == 'GREETING':
            return "GREETING_RESPONSE: Hello! I'm your educational robot assistant. How can I help you learn today?"

        elif command == 'FAREWELL':
            return "FAREWELL_RESPONSE: Goodbye! It was great learning with you!"

        # Movement commands
        elif command == 'MOVE_FORWARD':
            return "MOVE_FORWARD: Moving forward slowly"

        elif command == 'MOVE_BACKWARD':
            return "MOVE_BACKWARD: Moving backward slowly"

        elif command == 'TURN_LEFT':
            return "TURN_LEFT: Turning left"

        elif command == 'TURN_RIGHT':
            return "TURN_RIGHT: Turning right"

        elif command == 'STOP':
            return "STOP: Stopping all movement"

        # Default response
        return f"UNKNOWN_COMMAND: {command}"

    def create_movement_command(self, action):
        """Create Twist command for movement actions"""
        twist = Twist()

        if "MOVE_FORWARD" in action:
            twist.linear.x = 0.2
        elif "MOVE_BACKWARD" in action:
            twist.linear.x = -0.2
        elif "TURN_LEFT" in action:
            twist.angular.z = 0.5
        elif "TURN_RIGHT" in action:
            twist.angular.z = -0.5
        elif "STOP" in action:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        else:
            return None

        return twist

def main(args=None):
    rclpy.init(args=args)
    node = VLAIntegrator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down VLA integrator')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 6: Create the Educational Content Node

Create a node to provide educational content:

**voice_controlled_robot/voice_controlled_robot/educational_content_provider.py**:
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time

class EducationalContentProvider(Node):
    def __init__(self):
        super().__init__('educational_content_provider')

        # Educational content database
        self.educational_content = {
            'DEMONSTRATE_GRAVITY': {
                'explanation': "Gravity is a natural force that pulls objects toward each other. On Earth, gravity pulls everything toward the center of the planet.",
                'demonstration': "I can demonstrate gravity by showing how objects fall down instead of floating away.",
                'fun_fact': "Did you know that all objects fall at the same rate regardless of their weight? This was discovered by Galileo!"
            },
            'DEMONSTRATE_FRICTION': {
                'explanation': "Friction is the force that resists motion when two surfaces touch each other.",
                'demonstration': "I can demonstrate friction by moving across different surfaces to show how it affects motion.",
                'fun_fact': "Friction is why we can walk without slipping and why cars can stop when braking!"
            },
            'DEMONSTRATE_GEOMETRY': {
                'explanation': "Geometry is the study of shapes, sizes, and positions of objects.",
                'demonstration': "I can form different geometric shapes with my movements.",
                'fun_fact': "Geometry helps us understand the world around us, from the shapes of buildings to the orbits of planets!"
            }
        }

        # Create subscribers
        self.action_subscriber = self.create_subscription(
            String,
            '/robot_actions',
            self.action_callback,
            10)

        # Create publishers
        self.content_publisher = self.create_publisher(
            String,
            '/educational_content',
            10)

        self.feedback_publisher = self.create_publisher(
            String,
            '/robot_feedback',
            10)

        self.get_logger().info('Educational Content Provider initialized')

    def action_callback(self, msg):
        """Process actions and provide educational content"""
        try:
            action = msg.data

            # Check if this is an educational demonstration
            for demo_type in self.educational_content.keys():
                if demo_type in action:
                    content = self.educational_content[demo_type]

                    # Create educational content message
                    content_msg = String()
                    content_msg.data = f"EDUCATIONAL_CONTENT: {content['explanation']}\n\n{content['demonstration']}\n\n{content['fun_fact']}"
                    self.content_publisher.publish(content_msg)

                    # Provide verbal feedback
                    feedback_msg = String()
                    feedback_msg.data = content['explanation']
                    self.feedback_publisher.publish(feedback_msg)

                    self.get_logger().info(f'Provided educational content for {demo_type}')

                    return

            # Handle other types of actions
            if 'GREETING_RESPONSE' in action:
                feedback_msg = String()
                feedback_msg.data = "Hello! I'm your educational robot assistant. I can demonstrate science concepts and answer questions!"
                self.feedback_publisher.publish(feedback_msg)

            elif 'QUERY_COLOR_RESPONSE' in action:
                # Extract color information from action
                if 'I see mostly' in action:
                    color_info = action.split('I see mostly')[1].split('objects')[0].strip()
                    feedback_msg = String()
                    feedback_msg.data = f"I see mostly {color_info} objects. Colors help us identify and categorize things in our environment!"
                    self.feedback_publisher.publish(feedback_msg)

            elif 'QUERY_COUNT_RESPONSE' in action:
                # Extract count information from action
                if 'I see' in action:
                    count_info = action.split('I see')[1].split('objects')[0].strip()
                    feedback_msg = String()
                    feedback_msg.data = f"I see {count_info} objects. Counting helps us understand quantities in our environment!"
                    self.feedback_publisher.publish(feedback_msg)

        except Exception as e:
            self.get_logger().error(f'Error in action callback: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = EducationalContentProvider()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down educational content provider')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 7: Create the Main Controller Node

Create the main controller that orchestrates the system:

**voice_controlled_robot/voice_controlled_robot/main_controller.py**:
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist
from voice_controlled_robot.voice_command_parser import VoiceCommandParser
from voice_controlled_robot.vision_processor import VisionProcessor
from voice_controlled_robot.vla_integrator import VLAIntegrator
from voice_controlled_robot.educational_content_provider import EducationalContentProvider
import threading
import time

class MainController(Node):
    def __init__(self):
        super().__init__('main_controller')

        # Initialize all system components
        self.voice_parser = VoiceCommandParser()
        self.vision_processor = VisionProcessor()
        self.vla_integrator = VLAIntegrator()
        self.content_provider = EducationalContentProvider()

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

        # Create publishers
        self.status_publisher = self.create_publisher(
            String,
            '/system_status',
            10)

        self.active_publisher = self.create_publisher(
            Bool,
            '/system_active',
            10)

        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10)

        # Start all component nodes in separate threads
        self.threads = [
            threading.Thread(target=self.run_voice_parser),
            threading.Thread(target=self.run_vision_processor),
            threading.Thread(target=self.run_vla_integrator),
            threading.Thread(target=self.run_content_provider)
        ]

        # Start all threads
        for thread in self.threads:
            thread.start()

        # Timer for system health checks
        self.health_timer = self.create_timer(2.0, self.system_health_check)

        # Publish initial status
        status_msg = String()
        status_msg.data = "Voice-controlled educational robot system initialized and ready"
        self.status_publisher.publish(status_msg)

        self.get_logger().info('Main Controller initialized')

    def run_voice_parser(self):
        """Run voice command parser"""
        rclpy.spin(self.voice_parser)

    def run_vision_processor(self):
        """Run vision processor"""
        rclpy.spin(self.vision_processor)

    def run_vla_integrator(self):
        """Run VLA integrator"""
        rclpy.spin(self.vla_integrator)

    def run_content_provider(self):
        """Run educational content provider"""
        rclpy.spin(self.content_provider)

    def status_callback(self, msg):
        """Handle status messages from components"""
        self.get_logger().info(f'System component status: {msg.data}')

    def error_callback(self, msg):
        """Handle error messages"""
        self.error_count += 1
        self.get_logger().error(f'System error: {msg.data}')

        if self.error_count >= self.max_errors:
            self.get_logger().error('Too many errors, shutting down system')
            self.system_active = False

    def system_health_check(self):
        """Perform system health checks"""
        try:
            # Check if system should remain active
            active_msg = Bool()
            active_msg.data = self.system_active
            self.active_publisher.publish(active_msg)

            if not self.system_active:
                self.get_logger().warn('System is not active')

            # Log system status
            self.get_logger().info(f'System Health - Active: {self.system_active}, Errors: {self.error_count}')

        except Exception as e:
            self.get_logger().error(f'Error in health check: {e}')

    def destroy_node(self):
        """Clean up all system components"""
        self.system_active = False

        # Wait for threads to finish
        for thread in self.threads:
            if thread.is_alive():
                thread.join()

        # Destroy component nodes
        self.voice_parser.destroy_node()
        self.vision_processor.destroy_node()
        self.vla_integrator.destroy_node()
        self.content_provider.destroy_node()

        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = MainController()

    try:
        # Keep the main thread alive
        while rclpy.ok() and node.system_active:
            time.sleep(0.1)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down main controller')
    finally:
        # Ensure robot stops
        stop_cmd = Twist()
        node.cmd_vel_publisher.publish(stop_cmd)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 8: Create Launch File

Create a launch file to start the entire system:

**voice_controlled_robot/launch/voice_controlled_robot.launch.py**:
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

        # Voice Command Parser Node
        Node(
            package='voice_controlled_robot',
            executable='voice_command_parser',
            name='voice_command_parser',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),

        # Vision Processor Node
        Node(
            package='voice_controlled_robot',
            executable='vision_processor',
            name='vision_processor',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),

        # VLA Integrator Node
        Node(
            package='voice_controlled_robot',
            executable='vla_integrator',
            name='vla_integrator',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),

        # Educational Content Provider Node
        Node(
            package='voice_controlled_robot',
            executable='educational_content_provider',
            name='educational_content_provider',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),

        # Main Controller Node
        Node(
            package='voice_controlled_robot',
            executable='main_controller',
            name='main_controller',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        )
    ])
```

### Step 9: Update setup.py

Update the setup.py file:

**voice_controlled_robot/setup.py**:
```python
from setuptools import setup
from glob import glob
import os

package_name = 'voice_controlled_robot'

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
    description='Voice-Controlled Educational Robot Assistant',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'voice_command_parser = voice_controlled_robot.voice_command_parser:main',
            'vision_processor = voice_controlled_robot.vision_processor:main',
            'vla_integrator = voice_controlled_robot.vla_integrator:main',
            'educational_content_provider = voice_controlled_robot.educational_content_provider:main',
            'main_controller = voice_controlled_robot.main_controller:main',
        ],
    },
)
```

### Step 10: Build and Run the System

Build the package:

```bash
cd ~/ros2_ws
colcon build --packages-select voice_controlled_robot
source install/setup.bash
```

Run the voice-controlled robot system:

```bash
# Terminal 1: Launch the system
ros2 launch voice_controlled_robot voice_controlled_robot.launch.py
```

### Step 11: Test the System

To test the system, you can publish commands or use speech recognition if connected to a microphone:

```bash
# Terminal 2: Publish test commands
ros2 topic pub /transcribed_text std_msgs/String "data: 'show me gravity'"
```

Monitor the system's responses:

```bash
# Terminal 3: Monitor robot feedback
ros2 topic echo /robot_feedback

# Terminal 4: Monitor robot actions
ros2 topic echo /robot_actions

# Terminal 5: Monitor detected objects
ros2 topic echo /detected_objects
```

## Expected Results

When running the complete system, you should see:

1. Voice commands being received and parsed
2. Camera feed being processed for object detection
3. Educational content being provided based on commands
4. Robot executing appropriate movements and demonstrations
5. System providing feedback on actions and educational content

## Extensions

Try these extensions to enhance your voice-controlled robot:

1. **Improved Speech Recognition**: Integrate Whisper for better accuracy
2. **Advanced Object Detection**: Use deep learning models for better object recognition
3. **Natural Language Understanding**: Implement more sophisticated NLU
4. **Manipulation Skills**: Add robotic arm control for physical demonstrations
5. **Multi-Modal Interaction**: Add touch or gesture recognition
6. **Learning Adaptation**: Track student interactions and adapt responses
7. **Safety Features**: Implement comprehensive safety checks and emergency stops

## Evaluation Criteria

Your implementation should:
- Accurately parse voice commands
- Process visual information effectively
- Integrate voice and vision for intelligent responses
- Provide educational content appropriately
- Execute actions safely and reliably
- Handle errors gracefully
- Maintain system stability