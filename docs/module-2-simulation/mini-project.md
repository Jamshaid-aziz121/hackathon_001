# Mini Project: Multi-Robot Formation Control in Gazebo

## Project Overview

In this mini project, you'll implement a multi-robot formation control system in Gazebo simulation. This project will demonstrate coordination between multiple robots while maintaining a specific geometric formation pattern.

## Learning Objectives

- Implement multi-robot communication in ROS 2
- Design formation control algorithms
- Use simulation for testing multi-robot systems
- Handle robot-to-robot communication and coordination

## Prerequisites

- ROS 2 Humble Hawksbill installed
- Gazebo Garden installed
- Basic understanding of robot kinematics
- Experience with ROS 2 topics and message passing

## Project Requirements

- Control 3 robots to maintain a triangular formation
- Robots should follow a leader robot
- Formation should be maintained even when leader changes direction
- System should be robust to communication delays

## Step-by-Step Implementation

### Step 1: Create Robot Model Package

First, create a package for your robot model:

```bash
mkdir -p ~/ros2_ws/src/multi_robot_formation
cd ~/ros2_ws/src/multi_robot_formation
ros2 pkg create --build-type ament_python multi_robot_formation
```

### Step 2: Create Multi-Robot URDF Model

Create a URDF file that defines the robot with namespaces: `multi_robot_formation/urdf/formation_robot.urdf.xacro`:

```xml
<?xml version="1.0"?>
<robot name="formation_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:property name="robot_namespace" value="$(arg robot_namespace)" />

  <xacro:macro name="formation_robot" params="robot_namespace init_x init_y init_yaw">
    <gazebo>
      <plugin name="differential_drive_${robot_namespace}" filename="libgazebo_ros_diff_drive.so">
        <left_joint>${robot_namespace}/left_wheel_joint</left_joint>
        <right_joint>${robot_namespace}/right_wheel_joint</right_joint>
        <wheel_separation>0.4</wheel_separation>
        <wheel_diameter>0.2</wheel_diameter>
        <max_wheel_torque>20</max_wheel_torque>
        <max_wheel_acceleration>1.0</max_wheel_acceleration>
        <command_topic>cmd_vel</command_topic>
        <odometry_topic>odom</odometry_topic>
        <odometry_frame>odom</odometry_frame>
        <robot_base_frame>${robot_namespace}/base_link</robot_base_frame>
        <publish_odom>true</publish_odom>
        <publish_wheel_tf>true</publish_wheel_tf>
        <publish_odom_tf>true</publish_odom_tf>
        <odometry_source>world</odometry_source>
      </plugin>
    </gazebo>

    <link name="${robot_namespace}/base_link">
      <visual>
        <geometry>
          <box size="0.3 0.2 0.1"/>
        </geometry>
        <material name="${robot_namespace}_blue">
          <color rgba="0 0 1 0.8"/>
        </material>
      </visual>
      <collision>
        <geometry>
          <box size="0.3 0.2 0.1"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="1.0"/>
        <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
      </inertial>
    </link>

    <link name="${robot_namespace}/left_wheel">
      <visual>
        <geometry>
          <cylinder radius="0.1" length="0.05"/>
        </geometry>
        <material name="${robot_namespace}_black">
          <color rgba="0 0 0 0.8"/>
        </material>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="0.1" length="0.05"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.2"/>
        <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
      </inertial>
    </link>

    <link name="${robot_namespace}/right_wheel">
      <visual>
        <geometry>
          <cylinder radius="0.1" length="0.05"/>
        </geometry>
        <material name="${robot_namespace}_black">
          <color rgba="0 0 0 0.8"/>
        </material>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="0.1" length="0.05"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.2"/>
        <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
      </inertial>
    </link>

    <joint name="${robot_namespace}/left_wheel_joint" type="continuous">
      <parent link="${robot_namespace}/base_link"/>
      <child link="${robot_namespace}/left_wheel"/>
      <origin xyz="0 0.15 -0.05" rpy="1.5708 0 0"/>
      <axis xyz="0 0 1"/>
    </joint>

    <joint name="${robot_namespace}/right_wheel_joint" type="continuous">
      <parent link="${robot_namespace}/base_link"/>
      <child link="${robot_namespace}/right_wheel"/>
      <origin xyz="0 -0.15 -0.05" rpy="1.5708 0 0"/>
      <axis xyz="0 0 1"/>
    </joint>
  </xacro:macro>
</robot>
```

### Step 3: Create Launch File for Multiple Robots

Create a launch file to spawn multiple robots: `multi_robot_formation/launch/multi_robot_formation.launch.py`:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch Arguments
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

    # Spawn Robot 1 (Leader)
    spawn_robot1 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'robot1',
            '-topic', 'robot1/robot_description',
            '-x', '0',
            '-y', '0',
            '-z', '0.1'
        ],
        output='screen'
    )

    # Spawn Robot 2 (Follower)
    spawn_robot2 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'robot2',
            '-topic', 'robot2/robot_description',
            '-x', '1',
            '-y', '0',
            '-z', '0.1'
        ],
        output='screen'
    )

    # Spawn Robot 3 (Follower)
    spawn_robot3 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'robot3',
            '-topic', 'robot3/robot_description',
            '-x', '0.5',
            '-y', '0.866',  # Approximately 1 meter away at 60 degrees
            '-z', '0.1'
        ],
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
        spawn_robot3
    ])
```

### Step 4: Create Formation Control Node

Create the main formation control node: `multi_robot_formation/multi_robot_formation/formation_controller.py`:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
import math
import numpy as np

class FormationController(Node):
    def __init__(self):
        super().__init__('formation_controller')

        # Robot names
        self.robots = ['robot1', 'robot2', 'robot3']  # robot1 is leader
        self.formation_positions = {
            'robot1': (0.0, 0.0),      # Leader position
            'robot2': (-1.0, 0.0),     # Left follower
            'robot3': (-0.5, -0.866)   # Right follower (equilateral triangle)
        }

        # Current robot positions
        self.robot_positions = {robot: (0.0, 0.0, 0.0) for robot in self.robots}  # x, y, theta

        # Create publishers for robot velocities
        self.cmd_vel_publishers = {}
        for robot in self.robots:
            self.cmd_vel_publishers[robot] = self.create_publisher(
                Twist, f'/{robot}/cmd_vel', 10)

        # Create subscribers for robot odometry
        for robot in self.robots:
            self.create_subscription(
                Odometry,
                f'/{robot}/odom',
                lambda msg, r=robot: self.odom_callback(msg, r),
                10)

        # Timer for control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Formation controller initialized')

    def odom_callback(self, msg, robot_name):
        """Update robot position from odometry message"""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        self.robot_positions[robot_name] = (x, y, yaw)

    def calculate_desired_position(self, robot_name, leader_pos):
        """Calculate desired position for robot in formation"""
        if robot_name == 'robot1':  # Leader
            return leader_pos  # Leader follows its own path

        # Get desired offset from formation
        offset_x, offset_y = self.formation_positions[robot_name]

        # Rotate offset by leader's orientation
        leader_theta = leader_pos[2]
        rotated_offset_x = offset_x * math.cos(leader_theta) - offset_y * math.sin(leader_theta)
        rotated_offset_y = offset_x * math.sin(leader_theta) + offset_y * math.cos(leader_theta)

        # Calculate desired position
        desired_x = leader_pos[0] + rotated_offset_x
        desired_y = leader_pos[1] + rotated_offset_y

        return (desired_x, desired_y, leader_theta)

    def control_loop(self):
        """Main control loop for formation maintenance"""
        leader_pos = self.robot_positions['robot1']

        for robot in self.robots:
            if robot == 'robot1':  # Leader control - simple forward movement
                cmd_vel = Twist()
                cmd_vel.linear.x = 0.3  # Leader moves forward at 0.3 m/s
                cmd_vel.angular.z = 0.0  # Leader doesn't rotate by itself
            else:  # Follower control
                # Get desired position in formation
                desired_pos = self.calculate_desired_position(robot, leader_pos)
                current_pos = self.robot_positions[robot]

                # Calculate error
                error_x = desired_pos[0] - current_pos[0]
                error_y = desired_pos[1] - current_pos[1]

                # Simple proportional control
                k_linear = 0.5
                k_angular = 2.0

                distance_error = math.sqrt(error_x**2 + error_y**2)

                # Calculate angle to desired position
                desired_angle = math.atan2(error_y, error_x)
                current_angle = current_pos[2]

                # Calculate angular error (with wraparound)
                angle_error = desired_angle - current_angle
                while angle_error > math.pi:
                    angle_error -= 2 * math.pi
                while angle_error < -math.pi:
                    angle_error += 2 * math.pi

                # Create velocity command
                cmd_vel = Twist()
                cmd_vel.linear.x = min(k_linear * distance_error, 0.5)  # Limit speed
                cmd_vel.angular.z = k_angular * angle_error

            # Publish command
            self.cmd_vel_publishers[robot].publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    controller = FormationController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down')
    finally:
        # Stop all robots
        for robot in controller.robots:
            stop_msg = Twist()
            controller.cmd_vel_publishers[robot].publish(stop_msg)
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 5: Create Visualization Node

Create a node to visualize the formation: `multi_robot_formation/multi_robot_formation/formation_visualizer.py`:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
import math

class FormationVisualizer(Node):
    def __init__(self):
        super().__init__('formation_visualizer')

        # Robot names
        self.robots = ['robot1', 'robot2', 'robot3']
        self.robot_positions = {robot: (0.0, 0.0, 0.0) for robot in self.robots}

        # Publisher for visualization markers
        self.marker_publisher = self.create_publisher(MarkerArray, 'formation_markers', 10)

        # Subscribers for robot positions
        for robot in self.robots:
            self.create_subscription(
                Odometry,
                f'/{robot}/odom',
                lambda msg, r=robot: self.odom_callback(msg, r),
                10)

        # Timer for publishing visualization
        self.vis_timer = self.create_timer(0.1, self.publish_markers)

    def odom_callback(self, msg, robot_name):
        """Update robot position"""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.robot_positions[robot_name] = (x, y, 0)  # Only need x,y for visualization

    def publish_markers(self):
        """Publish visualization markers for formation"""
        marker_array = MarkerArray()

        # Create markers for each robot
        for i, robot in enumerate(self.robots):
            x, y, _ = self.robot_positions[robot]

            # Robot marker
            robot_marker = Marker()
            robot_marker.header.frame_id = "odom"
            robot_marker.header.stamp = self.get_clock().now().to_msg()
            robot_marker.ns = "formation"
            robot_marker.id = i
            robot_marker.type = Marker.SPHERE
            robot_marker.action = Marker.ADD
            robot_marker.pose.position.x = x
            robot_marker.pose.position.y = y
            robot_marker.pose.position.z = 0.1
            robot_marker.pose.orientation.w = 1.0
            robot_marker.scale.x = 0.2
            robot_marker.scale.y = 0.2
            robot_marker.scale.z = 0.2

            # Color based on robot role (leader is red, followers are blue)
            if robot == 'robot1':
                robot_marker.color.r = 1.0  # Red for leader
                robot_marker.color.g = 0.0
                robot_marker.color.b = 0.0
            else:
                robot_marker.color.r = 0.0  # Blue for followers
                robot_marker.color.g = 0.0
                robot_marker.color.b = 1.0
            robot_marker.color.a = 1.0

            marker_array.markers.append(robot_marker)

        # Create lines connecting robots to show formation
        line_marker = Marker()
        line_marker.header.frame_id = "odom"
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.ns = "formation"
        line_marker.id = len(self.robots)
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.pose.orientation.w = 1.0
        line_marker.scale.x = 0.05  # Line width

        # Add points for the formation lines
        for robot in self.robots:
            x, y, _ = self.robot_positions[robot]
            point = Point()
            point.x = x
            point.y = y
            point.z = 0.1
            line_marker.points.append(point)

        # Close the triangle
        x, y, _ = self.robot_positions['robot1']
        point = Point()
        point.x = x
        point.y = y
        point.z = 0.1
        line_marker.points.append(point)

        # White line color
        line_marker.color.r = 1.0
        line_marker.color.g = 1.0
        line_marker.color.b = 1.0
        line_marker.color.a = 0.8

        marker_array.markers.append(line_marker)

        # Publish the marker array
        self.marker_publisher.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    visualizer = FormationVisualizer()
    rclpy.spin(visualizer)
    visualizer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 6: Update setup.py

Update the `setup.py` file to make the nodes executable:

```python
from setuptools import setup

package_name = 'multi_robot_formation'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/urdf', ['urdf/formation_robot.urdf.xacro']),
        ('share/' + package_name + '/launch', ['launch/multi_robot_formation.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Multi-robot formation control in Gazebo',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'formation_controller = multi_robot_formation.formation_controller:main',
            'formation_visualizer = multi_robot_formation.formation_visualizer:main',
        ],
    },
)
```

### Step 7: Build and Run the Simulation

Build the package:

```bash
cd ~/ros2_ws
colcon build --packages-select multi_robot_formation
source install/setup.bash
```

Launch the simulation with multiple robots:

```bash
# Terminal 1: Launch Gazebo with multiple robots
ros2 launch multi_robot_formation multi_robot_formation.launch.py
```

In another terminal, run the formation controller:

```bash
# Terminal 2: Run the formation controller
ros2 run multi_robot_formation formation_controller
```

In a third terminal, run the visualizer:

```bash
# Terminal 3: Run the formation visualizer
ros2 run multi_robot_formation formation_visualizer
```

Visualize in RViz2:

```bash
# Terminal 4: Launch RViz2
rviz2
```

In RViz2, add:
- A "By Topic" display for `/formation_markers`
- Set the fixed frame to "odom"

## Expected Results

When running the simulation, you should see:
- Three robots in Gazebo
- The leader (red) moving forward
- The followers (blue) maintaining a triangular formation
- Lines connecting the robots showing the formation structure
- The formation maintained even as the leader changes direction

## Extensions

Try these extensions to enhance your formation control system:

1. **Dynamic Formation**: Allow changing formation patterns (line, diamond, etc.)
2. **Obstacle Avoidance**: Add obstacle detection to formation control
3. **Communication Failure**: Handle cases where robots lose communication
4. **Formation Reconfiguration**: Allow robots to join or leave the formation
5. **Path Following**: Make the leader follow a specific path while maintaining formation

## Evaluation Criteria

Your implementation should:
- Maintain the triangular formation as the leader moves
- Have followers follow the leader at appropriate positions
- Show stable control without oscillation
- Work correctly in Gazebo simulation
- Be visualized properly in RViz2