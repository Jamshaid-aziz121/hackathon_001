# Mini Project: ROS 2 TurtleBot3 Obstacle Avoidance

## Project Overview

In this mini project, you'll implement an obstacle avoidance algorithm for a simulated TurtleBot3 robot using ROS 2. This project will demonstrate how to process sensor data and control a robot to navigate safely around obstacles.

## Learning Objectives

- Process LiDAR sensor data in ROS 2
- Implement a reactive control algorithm
- Use ROS 2 topics for robot control
- Test your implementation in simulation

## Prerequisites

- ROS 2 Humble Hawksbill installed
- TurtleBot3 packages installed
- Gazebo simulation environment

## Step-by-Step Implementation

### Step 1: Launch the Simulation

First, set up your environment and launch the TurtleBot3 simulation:

```bash
# Terminal 1
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo empty_world.launch.py

# Terminal 2
ros2 run turtlebot3_teleop teleop_keyboard
```

### Step 2: Create the Obstacle Avoidance Node

Create a new package for the project:

```bash
mkdir -p ~/ros2_ws/src/obstacle_avoidance
cd ~/ros2_ws/src/obstacle_avoidance
ros2 pkg create --build-type ament_python obstacle_avoidance
```

Create the main node file `obstacle_avoidance/obstacle_avoidance/obstacle_avoidance_node.py`:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math

class ObstacleAvoidanceNode(Node):
    def __init__(self):
        super().__init__('obstacle_avoidance_node')

        # Create subscriber for laser scan data
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Create publisher for velocity commands
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)

        # Robot state
        self.laser_data = None
        self.safe_distance = 0.5  # meters
        self.linear_speed = 0.2   # m/s
        self.angular_speed = 0.5  # rad/s

    def scan_callback(self, msg):
        self.laser_data = msg

    def control_loop(self):
        if self.laser_data is None:
            return

        # Get minimum distance in front of robot (front 30 degrees)
        front_ranges = []
        # Front is typically at index around len(ranges)/2
        center_idx = len(self.laser_data.ranges) // 2
        front_range = 15  # Look at 15 indices on each side of center

        for i in range(center_idx - front_range, center_idx + front_range):
            i = i % len(self.laser_data.ranges)  # Handle wrap-around
            if not math.isnan(self.laser_data.ranges[i]) and self.laser_data.ranges[i] > 0:
                front_ranges.append(self.laser_data.ranges[i])

        if not front_ranges:
            return

        min_front_distance = min(front_ranges)

        # Create velocity command
        cmd_vel = Twist()

        if min_front_distance < self.safe_distance:
            # Obstacle detected, turn
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = self.angular_speed
        else:
            # Path clear, move forward
            cmd_vel.linear.x = self.linear_speed
            cmd_vel.angular.z = 0.0

        # Publish command
        self.cmd_vel_publisher.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoidanceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 3: Update setup.py

Update the `setup.py` file to make the node executable:

```python
from setuptools import setup

package_name = 'obstacle_avoidance'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Obstacle avoidance for TurtleBot3',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'obstacle_avoidance = obstacle_avoidance.obstacle_avoidance_node:main',
        ],
    },
)
```

### Step 4: Build and Run

Build the package:

```bash
cd ~/ros2_ws
colcon build --packages-select obstacle_avoidance
source install/setup.bash
```

Run the obstacle avoidance node:

```bash
ros2 run obstacle_avoidance obstacle_avoidance
```

### Step 5: Test in Simulation

1. Launch the TurtleBot3 simulation in a separate terminal
2. Run your obstacle avoidance node
3. Watch as the robot moves forward and turns to avoid obstacles
4. Use RViz2 to visualize the robot's sensor data and movement

## Enhancements

Try these enhancements to improve your obstacle avoidance algorithm:

1. **Wall Following**: Implement a wall following behavior
2. **Smarter Turning**: Use sensor data from multiple directions for better turning decisions
3. **Path Planning**: Implement a more sophisticated path planning algorithm
4. **Dynamic Obstacles**: Handle moving obstacles in the environment

## Evaluation

Your implementation should:
- Successfully avoid obstacles in the simulation
- Move forward when the path is clear
- Stop or turn appropriately when obstacles are detected
- Not collide with obstacles