# Case Study: Educational Robotics Lab with Simulation

## Scenario: University Robotics Education Program

This case study explores how a university robotics department can implement a comprehensive simulation-based learning environment using Gazebo and ROS 2. The goal is to provide students with hands-on experience in robotics programming without requiring expensive physical robots for every student.

## Background

The university has 100+ students enrolled in robotics courses but only 5 physical TurtleBot3 robots. The department wants to:
- Provide individual access to robot simulation for all students
- Enable collaborative projects between students
- Offer a progression from simulation to real hardware
- Support various robotics topics (navigation, perception, manipulation)

## Requirements

### Educational Requirements
- Students can run their own simulation instances
- Multiple simulation environments (indoor, outdoor, obstacle courses)
- Integration with existing ROS 2 curriculum
- Support for different robot models (differential drive, manipulator arms)
- Performance monitoring and logging for grading

### Technical Requirements
- Scalable simulation environment
- Integration with ROS 2 and common robotics tools
- Support for sensor simulation (LiDAR, cameras, IMU)
- Version control for student projects
- Easy deployment and management

## Solution Architecture

### Simulation Infrastructure

The solution uses a distributed architecture with:
- Centralized Gazebo simulation server
- Student workstations running ROS 2 nodes
- Cloud-based option for remote access
- Git-based project management

### Robot Models

Multiple robot models are available:
- Educational differential drive robot (similar to TurtleBot3)
- Simple manipulator arm
- Multi-robot scenarios (swarm robotics)

## Implementation

### 1. Simulation Environment Setup

Create standardized world files for different scenarios:

**Simple Room Environment (`worlds/simple_room.sdf`)**:
```xml
<sdf version="1.7">
  <world name="simple_room">
    <!-- Lighting -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.1 -0.1 -1</direction>
    </light>

    <!-- Ground plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>10 10</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
            <specular>0.7 0.7 0.7 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Walls -->
    <model name="wall_1">
      <pose>-5 0 1 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.1 10 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.1 10 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.4 0.4 0.4 1</ambient>
            <diffuse>0.4 0.4 0.4 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Add more walls to complete the room -->
    <model name="wall_2">
      <pose>5 0 1 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.1 10 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.1 10 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.4 0.4 0.4 1</ambient>
            <diffuse>0.4 0.4 0.4 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <model name="wall_3">
      <pose>0 -5 1 0 0 -1.57</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.1 10 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.1 10 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.4 0.4 0.4 1</ambient>
            <diffuse>0.4 0.4 0.4 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <model name="wall_4">
      <pose>0 5 1 0 0 -1.57</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.1 10 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.1 10 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.4 0.4 0.4 1</ambient>
            <diffuse>0.4 0.4 0.4 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Add obstacles for navigation challenges -->
    <model name="obstacle_1">
      <pose>2 2 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>1 0 0 1</ambient>
            <diffuse>1 0 0 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

### 2. Student Project Template

Create a standardized project template that students can use:

**Project Structure**:
```
student_robotics_project/
├── src/
│   └── robot_controller/
│       ├── CMakeLists.txt
│       ├── package.xml
│       └── robot_controller/
│           ├── __init__.py
│           └── robot_controller_node.py
├── worlds/
│   └── simple_room.sdf
├── launch/
│   └── student_robot.launch.py
├── config/
│   └── robot_params.yaml
└── README.md
```

**Robot Controller Node (`robot_controller/robot_controller/robot_controller_node.py`)**:
```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import math

class StudentRobotController(Node):
    def __init__(self):
        super().__init__('student_robot_controller')

        # Create publishers and subscribers
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.scan_subscriber = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10)

        # Create timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)

        # Robot state
        self.laser_data = None
        self.obstacle_detected = False
        self.course_stage = 0  # For multi-stage assignments

        self.get_logger().info('Student Robot Controller initialized')

    def scan_callback(self, msg):
        self.laser_data = msg
        # Check for obstacles in front of robot
        if self.laser_data:
            # Check front 30 degrees
            front_ranges = self.laser_data.ranges[:15] + self.laser_data.ranges[-15:]
            min_range = min([r for r in front_ranges if not math.isnan(r) and r > 0], default=float('inf'))
            self.obstacle_detected = min_range < 1.0  # 1 meter threshold

    def control_loop(self):
        cmd_vel = Twist()

        if self.course_stage == 0:
            # Stage 0: Simple forward movement
            cmd_vel.linear.x = 0.2 if not self.obstacle_detected else 0.0
            cmd_vel.angular.z = 0.0

        elif self.course_stage == 1:
            # Stage 1: Obstacle avoidance
            if not self.obstacle_detected:
                cmd_vel.linear.x = 0.2
                cmd_vel.angular.z = 0.0
            else:
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = 0.5  # Turn right to avoid obstacle

        # Publish command
        self.cmd_vel_publisher.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    controller = StudentRobotController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down')
    finally:
        # Stop the robot before shutting down
        stop_msg = Twist()
        controller.cmd_vel_publisher.publish(stop_msg)
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 3. Assessment and Grading System

Create a system for evaluating student performance:

**Performance Metrics Node (`robot_controller/robot_controller/assessment_node.py`)**:
```python
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
import math

class AssessmentNode(Node):
    def __init__(self):
        super().__init__('assessment_node')

        # Subscriptions
        self.odom_subscriber = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)
        self.scan_subscriber = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10)

        # Publishers for assessment metrics
        self.distance_traveled_publisher = self.create_publisher(Float32, 'assessment/distance_traveled', 10)
        self.collision_count_publisher = self.create_publisher(Float32, 'assessment/collision_count', 10)

        # Initialize variables
        self.start_position = None
        self.current_position = None
        self.total_distance = 0.0
        self.collision_count = 0
        self.last_position = None

    def odom_callback(self, msg):
        current_pos = msg.pose.pose.position

        if self.start_position is None:
            self.start_position = (current_pos.x, current_pos.y)
            self.last_position = (current_pos.x, current_pos.y)
        else:
            # Calculate distance traveled since last update
            dx = current_pos.x - self.last_position[0]
            dy = current_pos.y - self.last_position[1]
            distance = math.sqrt(dx*dx + dy*dy)
            self.total_distance += distance
            self.last_position = (current_pos.x, current_pos.y)

            # Publish metrics
            dist_msg = Float32()
            dist_msg.data = self.total_distance
            self.distance_traveled_publisher.publish(dist_msg)

    def scan_callback(self, msg):
        # Detect potential collisions based on laser scan
        min_distance = min([r for r in msg.ranges if not math.isnan(r) and r > 0], default=float('inf'))

        if min_distance < 0.2:  # Collision threshold
            self.collision_count += 1

        collision_msg = Float32()
        collision_msg.data = float(self.collision_count)
        self.collision_count_publisher.publish(collision_msg)

def main(args=None):
    rclpy.init(args=args)
    assessment = AssessmentNode()
    rclpy.spin(assessment)
    assessment.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Educational Benefits

This simulation-based approach provides:

### Scalability
- All students can work simultaneously without hardware limitations
- No scheduling conflicts for robot access
- Cost-effective for large classes

### Safety
- No risk of damaging expensive hardware
- Safe environment for experimenting with complex behaviors
- No physical safety concerns

### Flexibility
- Easy to reset simulation scenarios
- Multiple environments available
- Can simulate dangerous or difficult real-world scenarios

### Assessment
- Objective metrics for grading
- Detailed performance tracking
- Replay capabilities for reviewing student work

## Challenges and Solutions

### Challenge: Simulation vs. Reality Gap
**Solution**: Include exercises that highlight sim-to-real differences and teach students to account for them.

### Challenge: Network and Performance Issues
**Solution**: Provide cloud-based simulation options and optimize simulation models for performance.

### Challenge: Student Engagement
**Solution**: Create engaging scenarios and competitions to maintain interest.

## Extensions

Students can extend this system by:
- Adding more complex navigation algorithms
- Implementing computer vision applications
- Creating multi-robot coordination systems
- Developing custom world environments
- Integrating with machine learning models