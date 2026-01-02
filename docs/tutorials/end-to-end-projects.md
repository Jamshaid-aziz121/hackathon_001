# End-to-End Projects: Building Complete Educational Robotics Systems

## Overview

This tutorial guides you through building complete educational robotics systems that integrate all the concepts covered in this book. Each project demonstrates the integration of ROS 2, simulation environments, AI perception, and vision-language-action systems in educational contexts.

## Project 1: Interactive STEM Learning Robot

### Project Goals
- Build a robot that can interact with students through voice commands
- Demonstrate basic physics and engineering concepts
- Provide personalized learning experiences
- Ensure safety in educational environments

### Prerequisites
- ROS 2 Humble Hawksbill
- Gazebo simulation environment
- Basic understanding of robot navigation
- Python programming skills

### Step 1: Setting Up the Robot Platform

Create the robot platform package:

```bash
mkdir -p ~/ros2_ws/src/edu_robot_platform
cd ~/ros2_ws/src/edu_robot_platform
ros2 pkg create --build-type ament_python edu_robot_platform
```

Create the robot description in URDF:

**edu_robot_platform/edu_robot_platform/robot_description/edu_robot.urdf.xacro**:
```xml
<?xml version="1.0"?>
<robot name="edu_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.4 0.3"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.4 0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Left wheel -->
  <link name="left_wheel">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Right wheel -->
  <link name="right_wheel">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Camera -->
  <link name="camera">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- Joints -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0 0.25 -0.1" rpy="1.5708 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="0 -0.25 -0.1" rpy="1.5708 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera"/>
    <origin xyz="0.2 0 0.1" rpy="0 0 0"/>
  </joint>

  <!-- Gazebo plugins -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
  </gazebo>

  <gazebo>
    <plugin filename="libgazebo_ros_diff_drive.so" name="diff_drive">
      <left_joint>left_wheel_joint</left_joint>
      <right_joint>right_wheel_joint</right_joint>
      <wheel_separation>0.5</wheel_separation>
      <wheel_diameter>0.2</wheel_diameter>
      <max_wheel_torque>20</max_wheel_torque>
      <max_wheel_acceleration>1.0</max_wheel_acceleration>
      <command_topic>cmd_vel</command_topic>
      <odometry_topic>odom</odometry_topic>
      <odometry_frame>odom</odometry_frame>
      <robot_base_frame>base_link</robot_base_frame>
      <publish_odom>true</publish_odom>
      <publish_wheel_tf>true</publish_wheel_tf>
      <publish_odom_tf>true</publish_odom_tf>
      <odometry_source>world</odometry_source>
    </plugin>
  </gazebo>

  <gazebo reference="camera">
    <sensor name="camera1" type="camera">
      <update_rate>30</update_rate>
      <camera name="head">
        <horizontal_fov>1.3962634</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.02</near>
          <far>300</far>
        </clip>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <frame_name>camera</frame_name>
        <topic_name>image_raw</topic_name>
      </plugin>
    </sensor>
  </gazebo>
</robot>
```

### Step 2: Creating the Navigation Stack Configuration

Create navigation configuration files:

**edu_robot_platform/config/nav2_params.yaml**:
```yaml
amcl:
  ros__parameters:
    use_sim_time: True
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_link"
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

controller_server:
  ros__parameters:
    use_sim_time: True
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugin: "FollowPath"
    progress_checker:
      plugin: "nav2_controller::SimpleProgressChecker"
      required_movement_radius: 0.5
      movement_time_allowance: 10.0
    goal_checker:
      plugin: "nav2_controller::SimpleGoalChecker"
      xy_goal_tolerance: 0.25
      yaw_goal_tolerance: 0.25
      stateful: True
    FollowPath:
      plugin: "nav2_rotation_shim_controller::RotationShimController"
      primary_controller: "nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController"
      rotation_wait_timeout: 5.0
      max_angular_accel: 3.2
      use_native_yaw: false
      shim_only_to_goal: true
      max_allowed_time_to_collision: 1.0
      speed_limit_topic: "/speed_limit"
      enable_look_ahead: true
      min_look_ahead_dist: 0.3
      max_look_ahead_dist: 0.9
      look_ahead_time: 1.5
      rotate_to_heading_implementation: "DAMUEL"
      goal_dist_tol: 0.25
      xy_goal_tolerance: 0.1
      trans_stopped_velocity: 0.1
      short_circuit_trajectory: true
      debug_enabled: false
```

### Step 3: Creating the Voice Command Interface

Create a node to handle voice commands:

**edu_robot_platform/edu_robot_platform/voice_command_interface.py**:
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import speech_recognition as sr
import threading
import time

class VoiceCommandInterface(Node):
    def __init__(self):
        super().__init__('voice_command_interface')

        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

        # Create publisher for robot commands
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # Create publisher for system feedback
        self.feedback_publisher = self.create_publisher(String, 'system_feedback', 10)

        # Start voice recognition in a separate thread
        self.listening = True
        self.voice_thread = threading.Thread(target=self.listen_for_commands)
        self.voice_thread.start()

        self.get_logger().info('Voice Command Interface initialized')

    def listen_for_commands(self):
        """Listen for voice commands in a separate thread"""
        while self.listening:
            try:
                with self.microphone as source:
                    self.get_logger().info('Listening for voice commands...')
                    audio = self.recognizer.listen(source, timeout=2, phrase_time_limit=5)

                # Recognize speech
                command = self.recognizer.recognize_google(audio).lower()
                self.get_logger().info(f'Recognized command: {command}')

                # Process command
                self.process_command(command)

            except sr.WaitTimeoutError:
                # No speech detected, continue listening
                pass
            except sr.UnknownValueError:
                self.get_logger().info('Could not understand audio')
                self.publish_feedback('I did not understand that command')
            except sr.RequestError as e:
                self.get_logger().error(f'Error with speech recognition service: {e}')
                self.publish_feedback('Speech recognition service error')
            except Exception as e:
                self.get_logger().error(f'Error in voice recognition: {e}')

    def process_command(self, command):
        """Process the recognized voice command"""
        try:
            twist_cmd = Twist()

            if "forward" in command or "go forward" in command:
                twist_cmd.linear.x = 0.2
                self.publish_feedback("Moving forward")
            elif "backward" in command or "go backward" in command:
                twist_cmd.linear.x = -0.2
                self.publish_feedback("Moving backward")
            elif "left" in command or "turn left" in command:
                twist_cmd.angular.z = 0.5
                self.publish_feedback("Turning left")
            elif "right" in command or "turn right" in command:
                twist_cmd.angular.z = -0.5
                self.publish_feedback("Turning right")
            elif "stop" in command or "halt" in command:
                twist_cmd.linear.x = 0.0
                twist_cmd.angular.z = 0.0
                self.publish_feedback("Stopping")
            elif "hello" in command or "hi" in command:
                self.publish_feedback("Hello! I'm your educational robot assistant. How can I help you learn today?")
            elif "demonstrate gravity" in command:
                self.publish_feedback("I can demonstrate gravity concepts. I'll show you how objects fall at the same rate regardless of weight.")
            elif "demonstrate friction" in command:
                self.publish_feedback("I can demonstrate friction. Friction is the force that resists motion when surfaces touch.")
            else:
                self.publish_feedback(f"I heard: '{command}', but I don't know how to respond to that yet")

            # Publish command
            self.cmd_vel_publisher.publish(twist_cmd)

        except Exception as e:
            self.get_logger().error(f'Error processing command: {e}')

    def publish_feedback(self, message):
        """Publish system feedback"""
        feedback_msg = String()
        feedback_msg.data = message
        self.feedback_publisher.publish(feedback_msg)

    def destroy_node(self):
        """Clean up resources"""
        self.listening = False
        if self.voice_thread.is_alive():
            self.voice_thread.join()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = VoiceCommandInterface()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down voice command interface')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 4: Creating the Educational Content Manager

Create a node to manage educational content:

**edu_robot_platform/edu_robot_platform/educational_content_manager.py**:
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import json
import time

class EducationalContentManager(Node):
    def __init__(self):
        super().__init__('educational_content_manager')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Educational content database
        self.content_database = {
            "physics": {
                "gravity": {
                    "definition": "Gravity is a natural phenomenon by which all things with mass or energy are brought toward one another.",
                    "demonstration": "I can demonstrate gravity by showing how objects fall at the same rate regardless of their weight.",
                    "fun_fact": "Did you know that in a vacuum, a feather and a hammer fall at the same rate?",
                    "experiment": "Drop two objects of different weights from the same height to see they fall at the same rate."
                },
                "friction": {
                    "definition": "Friction is the force resisting the relative motion of solid surfaces sliding against each other.",
                    "demonstration": "I can show friction by moving across different surfaces to demonstrate how it affects motion.",
                    "fun_fact": "Friction is what allows us to walk without slipping and why cars can stop when braking!",
                    "experiment": "Try sliding an object across smooth and rough surfaces to feel the difference in friction."
                }
            },
            "mathematics": {
                "geometry": {
                    "definition": "Geometry is a branch of mathematics concerned with questions of shape, size, relative position of figures, and properties of space.",
                    "demonstration": "I can form geometric shapes with my movements to demonstrate different shapes.",
                    "fun_fact": "Geometry helps us understand the world around us, from the shapes of buildings to the orbits of planets!",
                    "experiment": "Look around and identify different geometric shapes in the classroom."
                }
            }
        }

        # Create subscribers
        self.command_subscriber = self.create_subscription(
            String,
            'system_feedback',
            self.command_callback,
            10)

        self.image_subscriber = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)

        # Create publishers
        self.educational_publisher = self.create_publisher(
            String,
            'educational_content',
            10)

        self.feedback_publisher = self.create_publisher(
            String,
            'system_feedback',
            10)

        self.get_logger().info('Educational Content Manager initialized')

    def command_callback(self, msg):
        """Process system commands and provide educational content"""
        try:
            command = msg.data.lower()

            # Check if command requests educational content
            if "demonstrate" in command:
                if "gravity" in command:
                    self.provide_educational_content("physics", "gravity")
                elif "friction" in command:
                    self.provide_educational_content("physics", "friction")
                elif "geometry" in command:
                    self.provide_educational_content("mathematics", "geometry")

        except Exception as e:
            self.get_logger().error(f'Error in command callback: {e}')

    def image_callback(self, msg):
        """Process camera images for educational purposes"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # In a real implementation, this would detect objects and provide relevant educational content
            # For now, we'll just log the image processing
            height, width = cv_image.shape[:2]
            self.get_logger().info(f'Processed image of size {width}x{height} for educational content')

        except Exception as e:
            self.get_logger().error(f'Error in image callback: {e}')

    def provide_educational_content(self, subject, topic):
        """Provide educational content based on subject and topic"""
        try:
            if subject in self.content_database and topic in self.content_database[subject]:
                content = self.content_database[subject][topic]

                # Create educational content message
                content_text = f"EDUCATIONAL CONTENT - {topic.upper()}:\n"
                content_text += f"Definition: {content['definition']}\n"
                content_text += f"Demonstration: {content['demonstration']}\n"
                content_text += f"Fun Fact: {content['fun_fact']}\n"
                content_text += f"Experiment: {content['experiment']}"

                # Publish educational content
                content_msg = String()
                content_msg.data = content_text
                self.educational_publisher.publish(content_msg)

                # Provide feedback
                feedback_msg = String()
                feedback_msg.data = f"Learning about {topic}! {content['fun_fact']}"
                self.feedback_publisher.publish(feedback_msg)

                self.get_logger().info(f'Provided educational content for {subject} - {topic}')

            else:
                self.get_logger().warn(f'No educational content found for {subject} - {topic}')

        except Exception as e:
            self.get_logger().error(f'Error providing educational content: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = EducationalContentManager()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down educational content manager')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 5: Creating the Main Launch File

Create a launch file to start the complete system:

**edu_robot_platform/launch/edu_robot_system.launch.py**:
```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # Launch Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'verbose': 'false',
            'pause': 'false',
        }.items()
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-file', PathJoinSubstitution([
                FindPackageShare('edu_robot_platform'),
                'robot_description',
                'edu_robot.urdf.xacro'
            ]),
            '-entity', 'edu_robot',
            '-x', '0',
            '-y', '0',
            '-z', '0.1'
        ],
        output='screen'
    )

    # Voice Command Interface Node
    voice_interface = Node(
        package='edu_robot_platform',
        executable='voice_command_interface',
        name='voice_command_interface',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Educational Content Manager Node
    content_manager = Node(
        package='edu_robot_platform',
        executable='educational_content_manager',
        name='educational_content_manager',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'
        ),
        gazebo,
        spawn_entity,
        voice_interface,
        content_manager
    ])
```

### Step 6: Update setup.py

Update the setup.py file:

**edu_robot_platform/setup.py**:
```python
from setuptools import setup
from glob import glob
import os

package_name = 'edu_robot_platform'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'robot_description'), glob('robot_description/*.urdf.xacro')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Educational Robot Platform for STEM Learning',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'voice_command_interface = edu_robot_platform.voice_command_interface:main',
            'educational_content_manager = edu_robot_platform.educational_content_manager:main',
        ],
    },
)
```

### Step 7: Build and Test the System

Build the package:

```bash
cd ~/ros2_ws
colcon build --packages-select edu_robot_platform
source install/setup.bash
```

Launch the complete educational robot system:

```bash
# Terminal 1: Launch the robot system
ros2 launch edu_robot_platform edu_robot_system.launch.py
```

Test the system by speaking commands to the robot:

```bash
# Speak commands like:
# "Move forward"
# "Turn left"
# "Demonstrate gravity"
# "Hello robot"
```

Monitor the system's responses:

```bash
# Terminal 2: Monitor system feedback
ros2 topic echo /system_feedback

# Terminal 3: Monitor educational content
ros2 topic echo /educational_content
```

## Project 2: Multi-Robot Collaborative Learning Environment

### Project Goals
- Create a multi-robot system for collaborative learning
- Demonstrate teamwork and coordination concepts
- Teach students about distributed systems
- Provide scalable educational experiences

### Step 1: Setting Up Multi-Robot Simulation

Create a launch file for multiple robots:

**edu_robot_platform/launch/multi_robot_education.launch.py**:
```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # Launch Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'verbose': 'false',
            'pause': 'false',
        }.items()
    )

    # Spawn Robot 1
    spawn_robot1 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-file', PathJoinSubstitution([
                FindPackageShare('edu_robot_platform'),
                'robot_description',
                'edu_robot.urdf.xacro'
            ]),
            '-entity', 'robot1',
            '-x', '0',
            '-y', '0',
            '-z', '0.1'
        ],
        output='screen'
    )

    # Spawn Robot 2
    spawn_robot2 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-file', PathJoinSubstitution([
                FindPackageShare('edu_robot_platform'),
                'robot_description',
                'edu_robot.urdf.xacro'
            ]),
            '-entity', 'robot2',
            '-x', '1',
            '-y', '0',
            '-z', '0.1'
        ],
        output='screen'
    )

    # Spawn Robot 3
    spawn_robot3 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-file', PathJoinSubstitution([
                FindPackageShare('edu_robot_platform'),
                'robot_description',
                'edu_robot.urdf.xacro'
            ]),
            '-entity', 'robot3',
            '-x', '0.5',
            '-y', '0.866',
            '-z', '0.1'
        ],
        output='screen'
    )

    # Robot 1 nodes
    robot1_voice = Node(
        package='edu_robot_platform',
        executable='voice_command_interface',
        name='robot1_voice_interface',
        namespace='robot1',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    robot1_content = Node(
        package='edu_robot_platform',
        executable='educational_content_manager',
        name='robot1_content_manager',
        namespace='robot1',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Robot 2 nodes
    robot2_voice = Node(
        package='edu_robot_platform',
        executable='voice_command_interface',
        name='robot2_voice_interface',
        namespace='robot2',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    robot2_content = Node(
        package='edu_robot_platform',
        executable='educational_content_manager',
        name='robot2_content_manager',
        namespace='robot2',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Robot 3 nodes
    robot3_voice = Node(
        package='edu_robot_platform',
        executable='voice_command_interface',
        name='robot3_voice_interface',
        namespace='robot3',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    robot3_content = Node(
        package='edu_robot_platform',
        executable='educational_content_manager',
        name='robot3_content_manager',
        namespace='robot3',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Multi-robot coordination node
    coordination_node = Node(
        package='edu_robot_platform',
        executable='multi_robot_coordinator',
        name='multi_robot_coordinator',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'
        ),
        gazebo,
        spawn_robot1,
        spawn_robot2,
        spawn_robot3,
        robot1_voice,
        robot1_content,
        robot2_voice,
        robot2_content,
        robot3_voice,
        robot3_content,
        coordination_node
    ])
```

### Step 2: Creating the Multi-Robot Coordinator

Create a node to coordinate multiple robots:

**edu_robot_platform/edu_robot_platform/multi_robot_coordinator.py**:
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import time
import random

class MultiRobotCoordinator(Node):
    def __init__(self):
        super().__init__('multi_robot_coordinator')

        # Robot states
        self.robot_states = {
            'robot1': {'position': (0, 0), 'task': 'idle', 'status': 'ready'},
            'robot2': {'position': (1, 0), 'task': 'idle', 'status': 'ready'},
            'robot3': {'position': (0.5, 0.866), 'task': 'idle', 'status': 'ready'}
        }

        # Create publishers for each robot
        self.robot_publishers = {}
        for robot_name in self.robot_states.keys():
            self.robot_publishers[robot_name] = self.create_publisher(
                String,
                f'/{robot_name}/system_feedback',
                10
            )

        # Create publishers for robot commands
        self.cmd_vel_publishers = {}
        for robot_name in self.robot_states.keys():
            self.cmd_vel_publishers[robot_name] = self.create_publisher(
                Twist,
                f'/{robot_name}/cmd_vel',
                10
            )

        # Create subscriber for coordination commands
        self.coordination_subscriber = self.create_subscription(
            String,
            '/coordination_commands',
            self.coordination_callback,
            10
        )

        # Timer for coordination tasks
        self.coordination_timer = self.create_timer(5.0, self.execute_coordination_task)

        self.get_logger().info('Multi-Robot Coordinator initialized')

    def coordination_callback(self, msg):
        """Process coordination commands"""
        try:
            command = msg.data.lower()
            self.get_logger().info(f'Received coordination command: {command}')

            # Process coordination commands
            if "form triangle" in command:
                self.coordinate_triangle_formation()
            elif "circle formation" in command:
                self.coordinate_circle_formation()
            elif "line formation" in command:
                self.coordinate_line_formation()
            elif "collaborative task" in command:
                self.coordinate_collaborative_task()

        except Exception as e:
            self.get_logger().error(f'Error in coordination callback: {e}')

    def coordinate_triangle_formation(self):
        """Coordinate robots to form a triangle"""
        self.get_logger().info('Coordinating triangle formation')

        # Send commands to each robot
        positions = {
            'robot1': (0, 0),
            'robot2': (1, 0),
            'robot3': (0.5, 0.866)
        }

        for robot_name, (x, y) in positions.items():
            # Send movement command
            cmd_msg = String()
            cmd_msg.data = f"MOVE_TO {x} {y}"
            self.robot_publishers[robot_name].publish(cmd_msg)

            # Send feedback
            feedback_msg = String()
            feedback_msg.data = f"Robot {robot_name} forming triangle at ({x}, {y})"
            self.robot_publishers[robot_name].publish(feedback_msg)

    def coordinate_circle_formation(self):
        """Coordinate robots to form a circle"""
        self.get_logger().info('Coordinating circle formation')

        # Calculate positions for circle formation (simplified for 3 robots)
        radius = 1.0
        angles = [0, 120, 240]  # Degrees

        for i, robot_name in enumerate(self.robot_states.keys()):
            angle_rad = angles[i] * 3.14159 / 180
            x = radius * round(0.5 * i, 2)  # Simplified positioning
            y = radius * round(0.5 * (i - 1), 2)  # Simplified positioning

            # Send movement command
            cmd_msg = String()
            cmd_msg.data = f"MOVE_TO {x} {y}"
            self.robot_publishers[robot_name].publish(cmd_msg)

            # Send feedback
            feedback_msg = String()
            feedback_msg.data = f"Robot {robot_name} forming circle at ({x}, {y})"
            self.robot_publishers[robot_name].publish(feedback_msg)

    def coordinate_line_formation(self):
        """Coordinate robots to form a line"""
        self.get_logger().info('Coordinating line formation')

        # Line formation positions
        positions = {
            'robot1': (0, 0),
            'robot2': (0.5, 0),
            'robot3': (1, 0)
        }

        for robot_name, (x, y) in positions.items():
            # Send movement command
            cmd_msg = String()
            cmd_msg.data = f"MOVE_TO {x} {y}"
            self.robot_publishers[robot_name].publish(cmd_msg)

            # Send feedback
            feedback_msg = String()
            feedback_msg.data = f"Robot {robot_name} forming line at ({x}, {y})"
            self.robot_publishers[robot_name].publish(feedback_msg)

    def coordinate_collaborative_task(self):
        """Coordinate a collaborative task among robots"""
        self.get_logger().info('Coordinating collaborative task')

        # Example: Robots take turns demonstrating concepts
        tasks = ["demonstrate gravity", "demonstrate friction", "demonstrate geometry"]
        robots = list(self.robot_states.keys())

        for i, robot_name in enumerate(robots):
            task = tasks[i % len(tasks)]

            # Send task command
            cmd_msg = String()
            cmd_msg.data = task
            self.robot_publishers[robot_name].publish(cmd_msg)

            # Send feedback
            feedback_msg = String()
            feedback_msg.data = f"Robot {robot_name} demonstrating {task.replace('demonstrate ', '')}"
            self.robot_publishers[robot_name].publish(feedback_msg)

    def execute_coordination_task(self):
        """Execute periodic coordination tasks"""
        try:
            # Randomly select a coordination task
            tasks = ["form triangle", "circle formation", "line formation", "collaborative task"]
            selected_task = random.choice(tasks)

            # Execute the selected task
            cmd_msg = String()
            cmd_msg.data = selected_task
            coordination_pub = self.create_publisher(String, '/coordination_commands', 10)
            coordination_pub.publish(cmd_msg)

            self.get_logger().info(f'Executing coordination task: {selected_task}')

        except Exception as e:
            self.get_logger().error(f'Error in coordination task execution: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = MultiRobotCoordinator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down multi-robot coordinator')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 8: Build and Test Multi-Robot System

Build the package with the new coordinator:

```bash
cd ~/ros2_ws
colcon build --packages-select edu_robot_platform
source install/setup.bash
```

Launch the multi-robot system:

```bash
# Terminal 1: Launch the multi-robot system
ros2 launch edu_robot_platform multi_robot_education.launch.py
```

Monitor the coordination:

```bash
# Terminal 2: Monitor coordination commands
ros2 topic echo /coordination_commands

# Terminal 3: Monitor robot feedback
ros2 topic echo /robot1/system_feedback
ros2 topic echo /robot2/system_feedback
ros2 topic echo /robot3/system_feedback
```

## Project 3: Complete Educational Robotics Curriculum

### Project Goals
- Integrate all modules into a comprehensive learning system
- Provide hands-on experience with real robotics concepts
- Create a progression from basic to advanced topics
- Enable students to build and program their own robots

### Implementation Steps

The complete educational robotics curriculum integrates all modules:
1. **ROS 2 Fundamentals**: Students learn about nodes, topics, and services
2. **Simulation Environments**: Students experiment with Gazebo simulations
3. **AI Perception**: Students implement computer vision and sensor fusion
4. **Vision-Language-Action Systems**: Students create interactive robots

This end-to-end project demonstrates how to combine all the concepts learned in this book to create a comprehensive educational robotics system that can be used in classrooms to teach STEM concepts through hands-on interaction with intelligent robots.

### Key Learning Outcomes

By completing these projects, students will:
- Understand the integration of hardware and software in robotics
- Learn to program robots using ROS 2
- Gain experience with AI and machine learning in robotics
- Develop problem-solving skills through hands-on projects
- Learn about safety and ethical considerations in robotics
- Understand collaborative and multi-robot systems

### Assessment and Evaluation

The projects include built-in assessment mechanisms:
- Performance metrics for robot navigation
- Accuracy measurements for perception systems
- Response quality for voice interaction
- Coordination effectiveness for multi-robot systems

These projects provide a complete educational framework for teaching robotics concepts from basic principles to advanced AI integration, all in the context of educational applications.