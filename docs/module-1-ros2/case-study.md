# Case Study: Educational Robot Control System

## Scenario: University Robotics Lab

In this case study, we'll explore how ROS 2 can be used to create an educational robot control system for a university robotics lab. The system will allow students to remotely control a robot and perform basic experiments.

## Requirements

- Remote control of a mobile robot
- Real-time sensor data visualization
- Educational interface for students
- Safety features to prevent damage
- Multi-user support for classroom environments

## System Architecture

### Robot Node
The robot node runs on the physical robot and handles:
- Motor control
- Sensor data collection
- Safety monitoring
- Communication with the control station

### Control Station Node
The control station node runs on the lab computers and provides:
- User interface for controlling the robot
- Sensor data visualization
- Experiment logging
- Safety overrides

### Communication Pattern

The system uses a combination of topics and services:
- `/cmd_vel` topic for sending velocity commands
- `/sensor_data` topic for receiving sensor information
- `/emergency_stop` service for immediate stopping
- `/experiment_control` action for complex experiment sequences

## Implementation

### Robot Controller Node

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Imu
from std_srvs.srv import Trigger

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        # Publishers for sensor data
        self.scan_publisher = self.create_publisher(LaserScan, '/sensor_data/scan', 10)
        self.imu_publisher = self.create_publisher(Imu, '/sensor_data/imu', 10)

        # Subscriber for velocity commands
        self.cmd_vel_subscriber = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)

        # Service server for emergency stop
        self.emergency_stop_service = self.create_service(
            Trigger, '/emergency_stop', self.emergency_stop_callback)

        # Timer for publishing sensor data
        self.timer = self.create_timer(0.1, self.publish_sensor_data)

        self.is_emergency_stopped = False

    def cmd_vel_callback(self, msg):
        if not self.is_emergency_stopped:
            # Send command to hardware
            self.send_command_to_hardware(msg)
        else:
            # Stop the robot if emergency stopped
            self.send_command_to_hardware(Twist())

    def emergency_stop_callback(self, request, response):
        self.is_emergency_stopped = True
        response.success = True
        response.message = "Emergency stop activated"
        return response

    def publish_sensor_data(self):
        # Publish simulated sensor data
        scan_msg = LaserScan()
        # ... populate scan_msg with data
        self.scan_publisher.publish(scan_msg)

        imu_msg = Imu()
        # ... populate imu_msg with data
        self.imu_publisher.publish(imu_msg)

    def send_command_to_hardware(self, cmd_vel):
        # Interface with actual robot hardware
        pass

def main(args=None):
    rclpy.init(args=args)
    controller = RobotController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()
```

## Educational Benefits

This case study demonstrates several key educational concepts:
- Distributed system design
- Real-time communication
- Safety in robotic systems
- Sensor integration
- Remote operation

## Extensions

Students can extend this system by:
- Adding computer vision capabilities
- Implementing autonomous navigation
- Creating a web-based interface
- Adding multiple robots to the system