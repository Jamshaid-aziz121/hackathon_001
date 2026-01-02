# Vision-Language-Action Implementation Walkthrough

## Creating a Complete VLA System for Educational Robotics

In this walkthrough, we'll implement a complete Vision-Language-Action (VLA) system that allows users to control a robot using natural language commands. The system will perceive its environment through vision, understand human instructions through language, and execute appropriate actions.

## Prerequisites

- ROS 2 Humble Hawksbill installed
- NVIDIA GPU with CUDA support (for deep learning acceleration)
- Basic knowledge of ROS 2 concepts
- Python 3.8+ with pip

## Step 1: Create the VLA Project Package

First, create a new ROS 2 package for our VLA system:

```bash
mkdir -p ~/ros2_ws/src/vla_educational_robot
cd ~/ros2_ws/src/vla_educational_robot
ros2 pkg create --build-type ament_python vla_educational_robot
```

## Step 2: Install Required Dependencies

Create a requirements file for our project:

**vla_educational_robot/requirements.txt**:
```
torch>=1.12.0
torchvision>=0.13.0
torchaudio>=0.12.0
openai-whisper
transformers>=4.21.0
accelerate
sentencepiece
opencv-python>=4.5.0
numpy>=1.21.0
py_trees>=2.2.0
SpeechRecognition
pyaudio
```

Install the dependencies:

```bash
pip3 install -r requirements.txt
```

## Step 3: Create the Audio Input Node

Create a node to handle audio input from a microphone:

**vla_educational_robot/vla_educational_robot/audio_input_node.py**:
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import AudioData
import pyaudio
import numpy as np
import threading
import queue

class AudioInputNode(Node):
    def __init__(self):
        super().__init__('audio_input_node')

        # Audio parameters
        self.rate = 16000  # Sample rate
        self.chunk = 1024  # Buffer size
        self.format = pyaudio.paInt16
        self.channels = 1

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()

        # Create publisher for audio data
        self.audio_publisher = self.create_publisher(
            AudioData,
            '/audio_input',
            10)

        # Start audio recording in a separate thread
        self.audio_queue = queue.Queue()
        self.recording = True
        self.audio_thread = threading.Thread(target=self.record_audio)
        self.audio_thread.start()

        self.get_logger().info('Audio Input Node initialized')

    def record_audio(self):
        """Record audio in a separate thread"""
        try:
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )

            while self.recording:
                data = stream.read(self.chunk)
                # Convert to AudioData message
                audio_msg = AudioData()
                audio_msg.data = data
                self.audio_publisher.publish(audio_msg)

            stream.stop_stream()
            stream.close()
        except Exception as e:
            self.get_logger().error(f'Error in audio recording: {e}')

    def destroy_node(self):
        """Clean up audio resources"""
        self.recording = False
        if self.audio_thread.is_alive():
            self.audio_thread.join()
        self.audio.terminate()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = AudioInputNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down audio input node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Step 4: Create the Speech-to-Text Node

Create a node to convert speech to text using Whisper:

**vla_educational_robot/vla_educational_robot/speech_to_text_node.py**:
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import AudioData
import whisper
import numpy as np
import io
import wave
import tempfile
import threading
import queue

class SpeechToTextNode(Node):
    def __init__(self):
        super().__init__('speech_to_text_node')

        # Load Whisper model
        self.get_logger().info('Loading Whisper model...')
        self.model = whisper.load_model("base")  # Use "small" or "medium" for better accuracy

        # Audio processing parameters
        self.audio_buffer = []
        self.buffer_size = 16000 * 2  # 2 seconds of audio at 16kHz
        self.silence_threshold = 0.01  # Threshold for voice activity detection

        # Create subscribers
        self.audio_subscriber = self.create_subscription(
            AudioData,
            '/audio_input',
            self.audio_callback,
            10)

        # Create publishers
        self.text_publisher = self.create_publisher(
            String,
            '/transcribed_text',
            10)

        self.listening_publisher = self.create_publisher(
            Bool,
            '/is_listening',
            10)

        # Threading for non-blocking transcription
        self.transcription_queue = queue.Queue()
        self.transcription_thread = threading.Thread(target=self.process_transcriptions)
        self.transcription_thread.start()

        self.get_logger().info('Speech-to-Text Node initialized')

    def audio_callback(self, msg):
        """Process incoming audio data"""
        try:
            # Convert audio data to numpy array
            audio_array = np.frombuffer(msg.data, dtype=np.int16).astype(np.float32) / 32768.0

            # Add to buffer
            self.audio_buffer.extend(audio_array)

            # Check if buffer is full
            if len(self.audio_buffer) >= self.buffer_size:
                # Check if there's significant audio (not just silence)
                audio_segment = np.array(self.audio_buffer)
                if np.mean(np.abs(audio_segment)) > self.silence_threshold:
                    # Queue for transcription
                    self.transcription_queue.put(self.audio_buffer.copy())
                    self.get_logger().info('Audio segment queued for transcription')

                # Keep only the last portion to maintain continuity
                self.audio_buffer = self.audio_buffer[-int(self.buffer_size/4):]

        except Exception as e:
            self.get_logger().error(f'Error in audio callback: {e}')

    def process_transcriptions(self):
        """Process transcriptions in a separate thread"""
        while rclpy.ok():
            try:
                # Wait for audio to be available
                if not self.transcription_queue.empty():
                    audio_data = self.transcription_queue.get()

                    # Transcribe audio using Whisper
                    result = self.model.transcribe(np.array(audio_data))
                    recognized_text = result["text"].strip()

                    if recognized_text:
                        # Publish recognized text
                        text_msg = String()
                        text_msg.data = recognized_text
                        self.text_publisher.publish(text_msg)

                        self.get_logger().info(f'Transcribed: {recognized_text}')

                        # Publish listening status
                        listening_msg = Bool()
                        listening_msg.data = True
                        self.listening_publisher.publish(listening_msg)

            except Exception as e:
                self.get_logger().error(f'Error in transcription processing: {e}')

    def destroy_node(self):
        """Clean up transcription thread"""
        if self.transcription_thread.is_alive():
            self.transcription_thread.join()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SpeechToTextNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down speech-to-text node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Step 5: Create the Language Understanding Node

Create a node to interpret natural language commands:

**vla_educational_robot/vla_educational_robot/language_understanding_node.py**:
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Pose
from transformers import pipeline
import re
import json

class LanguageUnderstandingNode(Node):
    def __init__(self):
        super().__init__('language_understanding_node')

        # Initialize NLP pipeline
        self.get_logger().info('Initializing language understanding...')

        # For educational purposes, we'll use a rule-based approach
        # In practice, you might use a more sophisticated model
        self.command_patterns = {
            # Movement commands
            r'.*move forward.*|.*go forward.*|.*forward.*': self.create_forward_command,
            r'.*move backward.*|.*go backward.*|.*backward.*|.*back.*': self.create_backward_command,
            r'.*turn left.*|.*go left.*|.*left.*': self.create_left_command,
            r'.*turn right.*|.*go right.*|.*right.*': self.create_right_command,
            r'.*stop.*|.*halt.*': self.create_stop_command,
            r'.*spin.*|.*rotate.*': self.create_spin_command,

            # Object interaction commands
            r'.*pick up.*|.*grasp.*|.*take.*': self.create_grasp_command,
            r'.*put down.*|.*release.*|.*drop.*': self.create_release_command,
            r'.*find.*|.*locate.*|.*look for.*': self.create_search_command,

            # Educational commands
            r'.*hello.*|.*hi.*': self.create_greeting_command,
            r'.*what.*color.*': self.create_color_query_command,
            r'.*how many.*': self.create_count_query_command,
        }

        # Create subscribers
        self.text_subscriber = self.create_subscription(
            String,
            '/transcribed_text',
            self.text_callback,
            10)

        # Create publishers
        self.command_publisher = self.create_publisher(
            String,
            '/parsed_commands',
            10)

        self.action_publisher = self.create_publisher(
            String,
            '/robot_actions',
            10)

        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10)

        self.get_logger().info('Language Understanding Node initialized')

    def text_callback(self, msg):
        """Process incoming text commands"""
        try:
            text = msg.data.lower().strip()
            self.get_logger().info(f'Processing command: {text}')

            # Parse command using patterns
            action = self.parse_command(text)

            if action:
                # Publish parsed command
                command_msg = String()
                command_msg.data = action
                self.command_publisher.publish(command_msg)

                # Publish action
                action_msg = String()
                action_msg.data = action
                self.action_publisher.publish(action_msg)

                # If it's a movement command, publish to cmd_vel
                if action.startswith('MOVE_'):
                    twist_cmd = self.create_twist_command(action)
                    if twist_cmd:
                        self.cmd_vel_publisher.publish(twist_cmd)

                self.get_logger().info(f'Parsed action: {action}')

        except Exception as e:
            self.get_logger().error(f'Error in text callback: {e}')

    def parse_command(self, text):
        """Parse natural language command and return action"""
        for pattern, handler in self.command_patterns.items():
            if re.search(pattern, text):
                return handler(text)

        # If no pattern matches, return unknown command
        return f"UNKNOWN_COMMAND: {text}"

    def create_forward_command(self, text):
        """Create forward movement command"""
        # Extract distance if specified
        distance_match = re.search(r'(\d+(?:\.\d+)?)\s*(meters?|m)', text)
        if distance_match:
            distance = float(distance_match.group(1))
            return f"MOVE_FORWARD_DISTANCE: {distance}"
        else:
            return "MOVE_FORWARD_CONTINUOUS"

    def create_backward_command(self, text):
        """Create backward movement command"""
        return "MOVE_BACKWARD"

    def create_left_command(self, text):
        """Create left turn command"""
        return "TURN_LEFT"

    def create_right_command(self, text):
        """Create right turn command"""
        return "TURN_RIGHT"

    def create_stop_command(self, text):
        """Create stop command"""
        return "STOP"

    def create_spin_command(self, text):
        """Create spin command"""
        return "SPIN"

    def create_grasp_command(self, text):
        """Create grasp command"""
        return "GRASP_OBJECT"

    def create_release_command(self, text):
        """Create release command"""
        return "RELEASE_OBJECT"

    def create_search_command(self, text):
        """Create search command"""
        return "SEARCH_OBJECT"

    def create_greeting_command(self, text):
        """Create greeting command"""
        return "GREETING"

    def create_color_query_command(self, text):
        """Create color query command"""
        return "QUERY_COLOR"

    def create_count_query_command(self, text):
        """Create count query command"""
        return "QUERY_COUNT"

    def create_twist_command(self, action):
        """Create Twist message for movement commands"""
        twist = Twist()

        if action == "MOVE_FORWARD_CONTINUOUS":
            twist.linear.x = 0.2
        elif action == "MOVE_BACKWARD":
            twist.linear.x = -0.2
        elif action == "TURN_LEFT":
            twist.angular.z = 0.5
        elif action == "TURN_RIGHT":
            twist.angular.z = -0.5
        elif action == "SPIN":
            twist.angular.z = 1.0
        elif action == "STOP":
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        else:
            return None

        return twist

def main(args=None):
    rclpy.init(args=args)
    node = LanguageUnderstandingNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down language understanding node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Step 6: Create the Vision Processing Node

Create a node to process visual information:

**vla_educational_robot/vla_educational_robot/vision_processing_node.py**:
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import threading
import queue

class VisionProcessingNode(Node):
    def __init__(self):
        super().__init__('vision_processing_node')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Load pre-trained object detection model
        self.get_logger().info('Loading vision model...')
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()

        # Create subscribers
        self.image_subscriber = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)

        # Create publishers
        self.object_publisher = self.create_publisher(
            String,
            '/detected_objects',
            10)

        self.vision_publisher = self.create_publisher(
            String,
            '/vision_analysis',
            10)

        # Threading for non-blocking processing
        self.processing_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self.process_images)
        self.processing_thread.start()

        self.get_logger().info('Vision Processing Node initialized')

    def image_callback(self, msg):
        """Process incoming image"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Queue for processing
            self.processing_queue.put((msg.header, cv_image))

        except Exception as e:
            self.get_logger().error(f'Error in image callback: {e}')

    def process_images(self):
        """Process images in a separate thread"""
        while rclpy.ok():
            try:
                if not self.processing_queue.empty():
                    header, cv_image = self.processing_queue.get()

                    # Perform object detection
                    objects = self.detect_objects(cv_image)

                    # Publish detected objects
                    objects_msg = String()
                    objects_msg.data = json.dumps(objects)
                    self.object_publisher.publish(objects_msg)

                    # Perform scene analysis
                    analysis = self.analyze_scene(cv_image, objects)

                    # Publish vision analysis
                    analysis_msg = String()
                    analysis_msg.data = analysis
                    self.vision_publisher.publish(analysis_msg)

                    self.get_logger().info(f'Detected {len(objects)} objects')

            except Exception as e:
                self.get_logger().error(f'Error in image processing: {e}')

    def detect_objects(self, image):
        """Detect objects in the image"""
        try:
            # Preprocess image
            transform = T.Compose([
                T.ToTensor(),
            ])
            img_tensor = transform(image).unsqueeze(0)

            # Run inference
            with torch.no_grad():
                predictions = self.model(img_tensor)

            # Extract relevant information
            objects = []
            for i, score in enumerate(predictions[0]['scores']):
                if score > 0.5:  # Confidence threshold
                    bbox = predictions[0]['boxes'][i].numpy()
                    label = predictions[0]['labels'][i].item()

                    # Convert COCO label ID to name (simplified)
                    label_name = self.coco_id_to_name(label)

                    obj_info = {
                        'label': label_name,
                        'confidence': float(score),
                        'bbox': bbox.tolist()
                    }
                    objects.append(obj_info)

            return objects

        except Exception as e:
            self.get_logger().error(f'Error in object detection: {e}')
            return []

    def analyze_scene(self, image, objects):
        """Analyze the scene and provide context"""
        try:
            # Count objects
            object_count = len(objects)

            # Identify main objects
            main_objects = [obj['label'] for obj in objects if obj['confidence'] > 0.7]

            # Analyze spatial relationships
            scene_description = f"Scene contains {object_count} objects: {', '.join(main_objects[:5])}"  # Limit to first 5

            return scene_description

        except Exception as e:
            self.get_logger().error(f'Error in scene analysis: {e}')
            return "Scene analysis failed"

    def coco_id_to_name(self, id):
        """Convert COCO dataset ID to object name"""
        # Simplified mapping (in practice, you'd use the full COCO dataset mapping)
        coco_names = {
            1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
            6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
            11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
            16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
            21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
            27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
            34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
            39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
            43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup',
            48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana',
            53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot',
            58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair',
            63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet',
            72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
            77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster',
            81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase',
            87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
        }
        return coco_names.get(id, f'object_{id}')

    def destroy_node(self):
        """Clean up processing thread"""
        if self.processing_thread.is_alive():
            self.processing_thread.join()
        super().destroy_node()

def main(args=None):
    import json
    rclpy.init(args=args)
    node = VisionProcessingNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down vision processing node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Step 7: Create the VLA Integration Node

Create the main node that integrates vision, language, and action:

**vla_educational_robot/vla_educational_robot/vla_integration_node.py**:
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import json
import threading
import time

class VLAIntegrationNode(Node):
    def __init__(self):
        super().__init__('vla_integration_node')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # State variables
        self.last_vision_analysis = ""
        self.last_command = ""
        self.last_objects = []

        # Create subscribers
        self.command_subscriber = self.create_subscription(
            String,
            '/parsed_commands',
            self.command_callback,
            10)

        self.vision_subscriber = self.create_subscription(
            String,
            '/vision_analysis',
            self.vision_callback,
            10)

        self.object_subscriber = self.create_subscription(
            String,
            '/detected_objects',
            self.object_callback,
            10)

        # Create publishers
        self.action_publisher = self.create_publisher(
            String,
            '/integrated_actions',
            10)

        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10)

        # Timer for action execution
        self.action_timer = self.create_timer(0.1, self.execute_integrated_action)

        self.get_logger().info('VLA Integration Node initialized')

    def command_callback(self, msg):
        """Process incoming commands"""
        self.last_command = msg.data
        self.get_logger().info(f'Received command: {msg.data}')

    def vision_callback(self, msg):
        """Process vision analysis"""
        self.last_vision_analysis = msg.data

    def object_callback(self, msg):
        """Process detected objects"""
        try:
            self.last_objects = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f'Error parsing objects: {e}')
            self.last_objects = []

    def execute_integrated_action(self):
        """Execute integrated vision-language-action"""
        try:
            if self.last_command and self.last_vision_analysis:
                # Integrate vision and language to make decisions
                action = self.integrate_vla()

                if action:
                    # Publish integrated action
                    action_msg = String()
                    action_msg.data = action
                    self.action_publisher.publish(action_msg)

                    # Execute if it's a movement command
                    if "MOVE" in action:
                        twist_cmd = self.create_movement_command(action)
                        if twist_cmd:
                            self.cmd_vel_publisher.publish(twist_cmd)

                    self.get_logger().info(f'Integrated action: {action}')

        except Exception as e:
            self.get_logger().error(f'Error in integrated action: {e}')

    def integrate_vla(self):
        """Integrate vision and language for action planning"""
        command = self.last_command
        vision = self.last_vision_analysis
        objects = self.last_objects

        # Simple integration logic
        if "SEARCH_OBJECT" in command:
            # Look for objects mentioned in the command
            search_term = command.lower()
            for obj in objects:
                if obj['label'] in search_term and obj['confidence'] > 0.7:
                    return f"MOVE_TO_OBJECT: {obj['label']} at position {obj['bbox']}"

        elif "QUERY_COLOR" in command:
            # If asking about colors, describe the scene
            return f"SCENE_DESCRIPTION: {vision}"

        elif "QUERY_COUNT" in command:
            # If asking for counts, provide object count
            return f"OBJECT_COUNT: {len(objects)} objects detected in scene"

        elif "GREETING" in command:
            # If greeting, acknowledge and describe environment
            return f"GREETING_RESPONSE: Hello! I see {len(objects)} objects in the environment."

        # Default to original command if no special integration needed
        return command

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
    node = VLAIntegrationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down VLA integration node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Step 8: Create the Action Execution Node

Create a node to execute robot actions:

**vla_educational_robot/vla_educational_robot/action_execution_node.py**:
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import time

class ActionExecutionNode(Node):
    def __init__(self):
        super().__init__('action_execution_node')

        # Robot state
        self.current_action = None
        self.action_start_time = None
        self.action_duration = 0.0

        # Safety parameters
        self.safety_distance = 0.3  # meters
        self.emergency_stop = False

        # Create subscribers
        self.action_subscriber = self.create_subscription(
            String,
            '/integrated_actions',
            self.action_callback,
            10)

        self.laser_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10)

        # Create publishers
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10)

        self.status_publisher = self.create_publisher(
            String,
            '/action_status',
            10)

        # Timer for action execution
        self.action_timer = self.create_timer(0.1, self.execute_action)

        self.get_logger().info('Action Execution Node initialized')

    def action_callback(self, msg):
        """Process incoming actions"""
        action = msg.data
        self.get_logger().info(f'New action received: {action}')

        # Update current action
        self.current_action = action
        self.action_start_time = time.time()

        # Determine action duration based on type
        if "MOVE_FORWARD_DISTANCE" in action:
            # Extract distance and calculate time (assuming 0.2 m/s)
            try:
                distance = float(action.split(':')[1])
                self.action_duration = distance / 0.2
            except:
                self.action_duration = 1.0  # Default to 1 second
        else:
            self.action_duration = 1.0  # Default duration

    def laser_callback(self, msg):
        """Process laser scan for safety"""
        try:
            # Check for obstacles in front
            front_scan = msg.ranges[len(msg.ranges)//2-30:len(msg.ranges)//2+30]
            min_distance = min([r for r in front_scan if 0 < r < float('inf')], default=float('inf'))

            # Set emergency stop if obstacle is too close
            self.emergency_stop = min_distance < self.safety_distance

            if self.emergency_stop:
                self.get_logger().warn(f'Obstacle detected at {min_distance:.2f}m, emergency stop activated')

        except Exception as e:
            self.get_logger().error(f'Error in laser callback: {e}')

    def execute_action(self):
        """Execute the current action"""
        try:
            if self.emergency_stop:
                # Emergency stop - send stop command
                stop_cmd = Twist()
                self.cmd_vel_publisher.publish(stop_cmd)
                status_msg = String()
                status_msg.data = "EMERGENCY_STOP"
                self.status_publisher.publish(status_msg)
                return

            if self.current_action is None:
                return

            # Check if action should continue
            elapsed_time = time.time() - self.action_start_time if self.action_start_time else 0

            if elapsed_time > self.action_duration:
                # Action completed, stop the robot
                stop_cmd = Twist()
                self.cmd_vel_publisher.publish(stop_cmd)
                self.current_action = None
                self.get_logger().info('Action completed')
                return

            # Execute the action
            twist_cmd = self.create_action_command(self.current_action)
            if twist_cmd:
                self.cmd_vel_publisher.publish(twist_cmd)

                status_msg = String()
                status_msg.data = f"EXECUTING: {self.current_action} ({elapsed_time:.1f}s/{self.action_duration:.1f}s)"
                self.status_publisher.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f'Error in action execution: {e}')

    def create_action_command(self, action):
        """Create Twist command for the given action"""
        twist = Twist()

        if "MOVE_FORWARD_CONTINUOUS" in action:
            twist.linear.x = 0.2
        elif "MOVE_BACKWARD" in action:
            twist.linear.x = -0.2
        elif "TURN_LEFT" in action:
            twist.angular.z = 0.5
        elif "TURN_RIGHT" in action:
            twist.angular.z = -0.5
        elif "SPIN" in action:
            twist.angular.z = 1.0
        elif "MOVE_TO_OBJECT" in action:
            # For moving to an object, we'll move forward for now
            # In a real system, this would involve more complex navigation
            twist.linear.x = 0.1
        else:
            # Default to stop if unknown action
            return None

        return twist

def main(args=None):
    rclpy.init(args=args)
    node = ActionExecutionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down action execution node')
    finally:
        # Ensure robot stops
        stop_cmd = Twist()
        node.cmd_vel_publisher.publish(stop_cmd)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Step 9: Create the Main VLA System Node

Create the main node that orchestrates the entire system:

**vla_educational_robot/vla_educational_robot/vla_system_manager.py**:
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import threading
import time
from vla_educational_robot.audio_input_node import AudioInputNode
from vla_educational_robot.speech_to_text_node import SpeechToTextNode
from vla_educational_robot.language_understanding_node import LanguageUnderstandingNode
from vla_educational_robot.vision_processing_node import VisionProcessingNode
from vla_educational_robot.vla_integration_node import VLAIntegrationNode
from vla_educational_robot.action_execution_node import ActionExecutionNode

class VLA_SystemManager(Node):
    def __init__(self):
        super().__init__('vla_system_manager')

        # Initialize all component nodes
        self.audio_input = AudioInputNode()
        self.speech_to_text = SpeechToTextNode()
        self.language_understanding = LanguageUnderstandingNode()
        self.vision_processing = VisionProcessingNode()
        self.vla_integration = VLAIntegrationNode()
        self.action_execution = ActionExecutionNode()

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

        # Start all component nodes in separate threads
        self.threads = [
            threading.Thread(target=self.run_audio_input),
            threading.Thread(target=self.run_speech_to_text),
            threading.Thread(target=self.run_language_understanding),
            threading.Thread(target=self.run_vision_processing),
            threading.Thread(target=self.run_vla_integration),
            threading.Thread(target=self.run_action_execution)
        ]

        # Start all threads
        for thread in self.threads:
            thread.start()

        # Timer for system health checks
        self.health_timer = self.create_timer(2.0, self.system_health_check)

        # Publish initial status
        status_msg = String()
        status_msg.data = "VLA Educational Robot System initialized and running"
        self.status_publisher.publish(status_msg)

        self.get_logger().info('VLA System Manager initialized')

    def run_audio_input(self):
        """Run audio input node"""
        rclpy.spin(self.audio_input)

    def run_speech_to_text(self):
        """Run speech-to-text node"""
        rclpy.spin(self.speech_to_text)

    def run_language_understanding(self):
        """Run language understanding node"""
        rclpy.spin(self.language_understanding)

    def run_vision_processing(self):
        """Run vision processing node"""
        rclpy.spin(self.vision_processing)

    def run_vla_integration(self):
        """Run VLA integration node"""
        rclpy.spin(self.vla_integration)

    def run_action_execution(self):
        """Run action execution node"""
        rclpy.spin(self.action_execution)

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

            # Log system status
            self.get_logger().info(f'VLA System Health - Active: {self.system_active}, Errors: {self.error_count}')

        except Exception as e:
            self.get_logger().error(f'Error in health check: {e}')

    def destroy_node(self):
        """Clean up all component nodes"""
        self.system_active = False

        # Wait for threads to finish
        for thread in self.threads:
            if thread.is_alive():
                thread.join()

        # Destroy component nodes
        self.audio_input.destroy_node()
        self.speech_to_text.destroy_node()
        self.language_understanding.destroy_node()
        self.vision_processing.destroy_node()
        self.vla_integration.destroy_node()
        self.action_execution.destroy_node()

        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = VLA_SystemManager()

    try:
        # Keep the main thread alive
        while rclpy.ok() and node.system_active:
            time.sleep(0.1)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down VLA system manager')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Step 10: Create Launch File

Create a launch file to start the entire VLA system:

**vla_educational_robot/launch/vla_educational_robot.launch.py**:
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

        # Audio Input Node
        Node(
            package='vla_educational_robot',
            executable='audio_input_node',
            name='audio_input_node',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),

        # Speech-to-Text Node
        Node(
            package='vla_educational_robot',
            executable='speech_to_text_node',
            name='speech_to_text_node',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),

        # Language Understanding Node
        Node(
            package='vla_educational_robot',
            executable='language_understanding_node',
            name='language_understanding_node',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),

        # Vision Processing Node
        Node(
            package='vla_educational_robot',
            executable='vision_processing_node',
            name='vision_processing_node',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),

        # VLA Integration Node
        Node(
            package='vla_educational_robot',
            executable='vla_integration_node',
            name='vla_integration_node',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),

        # Action Execution Node
        Node(
            package='vla_educational_robot',
            executable='action_execution_node',
            name='action_execution_node',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),

        # VLA System Manager (orchestrates the whole system)
        Node(
            package='vla_educational_robot',
            executable='vla_system_manager',
            name='vla_system_manager',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        )
    ])
```

## Step 11: Update setup.py

Update the setup.py file to make the nodes executable:

**vla_educational_robot/setup.py**:
```python
from setuptools import setup
from glob import glob
import os

package_name = 'vla_educational_robot'

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
    description='Vision-Language-Action System for Educational Robotics',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'audio_input_node = vla_educational_robot.audio_input_node:main',
            'speech_to_text_node = vla_educational_robot.speech_to_text_node:main',
            'language_understanding_node = vla_educational_robot.language_understanding_node:main',
            'vision_processing_node = vla_educational_robot.vision_processing_node:main',
            'vla_integration_node = vla_educational_robot.vla_integration_node:main',
            'action_execution_node = vla_educational_robot.action_execution_node:main',
            'vla_system_manager = vla_educational_robot.vla_system_manager:main',
        ],
    },
)
```

## Step 12: Build and Run the System

Build the package:

```bash
cd ~/ros2_ws
colcon build --packages-select vla_educational_robot
source install/setup.bash
```

Run the complete VLA system:

```bash
# Terminal 1: Launch the VLA system
ros2 launch vla_educational_robot vla_educational_robot.launch.py
```

## Step 13: Test the System

To test the system, you can speak commands to the robot (if using a real robot with microphone) or publish test commands:

```bash
# Terminal 2: Publish test commands
ros2 topic pub /transcribed_text std_msgs/String "data: 'move forward'"
```

Monitor the system's responses:

```bash
# Terminal 3: Monitor actions
ros2 topic echo /robot_actions

# Terminal 4: Monitor vision analysis
ros2 topic echo /vision_analysis

# Terminal 5: Monitor system status
ros2 topic echo /system_status
```

## Verification

Your complete VLA system should now:
1. Capture audio input from a microphone
2. Convert speech to text using Whisper
3. Interpret natural language commands
4. Process visual information from a camera
5. Integrate vision and language for decision making
6. Execute appropriate actions on the robot
7. Handle safety considerations (obstacle avoidance)
8. Provide feedback on system status

The system demonstrates the complete Vision-Language-Action pipeline for educational robotics applications.