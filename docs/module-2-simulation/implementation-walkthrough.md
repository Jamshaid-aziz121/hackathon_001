# Simulation Implementation Walkthrough

## Creating a Gazebo Simulation for Educational Robotics

In this walkthrough, we'll create a complete Gazebo simulation environment with a simple robot model that students can use to learn basic robotics concepts.

## Prerequisites

- ROS 2 Humble Hawksbill installed
- Gazebo Garden installed
- Basic understanding of URDF robot modeling

## Step 1: Create a Robot Model (URDF)

First, let's create a simple differential drive robot model. Create a new ROS 2 package:

```bash
mkdir -p ~/ros2_ws/src/educational_robot_description
cd ~/ros2_ws/src/educational_robot_description
ros2 pkg create --build-type ament_cmake educational_robot_description
```

Create the URDF file at `educational_robot_description/urdf/educational_robot.urdf`:

```xml
<?xml version="1.0"?>
<robot name="educational_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.15"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.15"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Left Wheel -->
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
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Right Wheel -->
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
      <mass value="0.2"/>
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
      <mass value="0.05"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- Joints -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0 0.2 -0.05" rpy="1.5708 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="0 -0.2 -0.05" rpy="1.5708 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera"/>
    <origin xyz="0.2 0 0.1" rpy="0 0 0"/>
  </joint>

  <!-- Gazebo Plugins -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
  </gazebo>

  <gazebo reference="left_wheel">
    <material>Gazebo/Black</material>
  </gazebo>

  <gazebo reference="right_wheel">
    <material>Gazebo/Black</material>
  </gazebo>

  <gazebo reference="camera">
    <material>Gazebo/Red</material>
  </gazebo>
</robot>
```

## Step 2: Add Gazebo Plugins for Simulation

Create a Gazebo plugin configuration file at `educational_robot_description/urdf/educational_robot.gazebo.xacro`:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Differential Drive Plugin -->
  <gazebo>
    <plugin filename="libgazebo_ros_diff_drive.so" name="diff_drive">
      <left_joint>left_wheel_joint</left_joint>
      <right_joint>right_wheel_joint</right_joint>
      <wheel_separation>0.4</wheel_separation>
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

  <!-- Camera Plugin -->
  <gazebo reference="camera">
    <sensor name="camera1" type="camera">
      <update_rate>30</update_rate>
      <camera name="head">
        <horizontal_fov>1.3962634</horizontal_fov>
        <image>
          <width>800</width>
          <height>600</height>
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

## Step 3: Create a Launch File

Create a launch file at `educational_robot_description/launch/educational_robot.launch.py`:

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

    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': '''
                <?xml version="1.0"?>
                <robot name="educational_robot">
                    <link name="base_link"/>
                </robot>
            '''
        }],
        arguments=[PathJoinSubstitution([
            FindPackageShare('educational_robot_description'),
            'urdf',
            'educational_robot.urdf'
        ])]
    )

    # Spawn Robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'educational_robot',
            '-x', '0',
            '-y', '0',
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
        robot_state_publisher,
        spawn_entity
    ])
```

## Step 4: Build and Run the Simulation

Build the package:

```bash
cd ~/ros2_ws
colcon build --packages-select educational_robot_description
source install/setup.bash
```

Launch the simulation:

```bash
ros2 launch educational_robot_description educational_robot.launch.py
```

## Step 5: Control the Robot

In a separate terminal, control the robot using teleop:

```bash
# Install teleop package if not already installed
sudo apt install ros-humble-teleop-twist-keyboard

# Control the robot
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

## Step 6: Visualize with RViz2

Launch RViz2 to visualize the robot and its sensors:

```bash
# Terminal 1: Launch RViz2
rviz2

# In RViz2:
# 1. Set Fixed Frame to "base_link" or "odom"
# 2. Add displays for:
#    - RobotModel (to see the robot)
#    - LaserScan (if you add a laser)
#    - Image (to see camera feed)
```

## Step 7: Create a Simple Controller

Create a basic controller node to programmatically control the robot. Create `educational_robot_description/educational_robot_description/simple_controller.py`:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class SimpleController(Node):
    def __init__(self):
        super().__init__('simple_controller')
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.control_loop)
        self.step = 0
        self.start_time = self.get_clock().now()

    def control_loop(self):
        msg = Twist()

        # Square movement pattern
        current_time = self.get_clock().now()
        elapsed = (current_time - self.start_time).nanoseconds / 1e9

        # Move in a square pattern (4 seconds per side)
        if elapsed % 8 < 4:  # Moving forward
            msg.linear.x = 0.5
            msg.angular.z = 0.0
        else:  # Turning
            msg.linear.x = 0.0
            msg.angular.z = 0.5

        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    controller = SimpleController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Update the setup.py to make the controller executable:

```python
entry_points={
    'console_scripts': [
        'simple_controller = educational_robot_description.simple_controller:main',
    ],
},
```

Build again and run the controller:

```bash
cd ~/ros2_ws
colcon build --packages-select educational_robot_description
source install/setup.bash

# Run the controller
ros2 run educational_robot_description simple_controller
```

## Verification

Your simulation should now:
1. Launch Gazebo with your robot model
2. Allow teleoperation with keyboard controls
3. Show the robot moving in a square pattern when using the controller
4. Publish sensor data that can be visualized in RViz2