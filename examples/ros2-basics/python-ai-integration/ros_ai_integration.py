#!/usr/bin/env python3

"""
Python-AI ROS Integration Example

This example demonstrates how to integrate AI algorithms with ROS 2 for educational robotics.
It includes a simple neural network for decision making and interfaces with ROS topics.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from std_msgs.msg import Float32
import numpy as np
import math
import time


class SimpleNeuralNetwork:
    """
    A simple neural network for educational purposes.
    This network has one hidden layer and can be used for basic decision making.
    """
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights randomly
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))

        # Activation function (sigmoid)
        self.sigmoid = lambda x: 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    def forward(self, X):
        """Forward pass through the network"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        output = self.sigmoid(self.z2)
        return output

    def predict(self, X):
        """Make a prediction with the network"""
        return self.forward(X)


class AIBasedRobotController(Node):
    """
    A ROS 2 node that uses AI for robot control decisions.
    This node subscribes to sensor data, processes it through an AI model,
    and publishes velocity commands.
    """
    def __init__(self):
        super().__init__('ai_robot_controller')

        # Initialize neural network
        # Input: 5 laser readings + 2 for goal direction = 7
        # Output: linear velocity, angular velocity = 2
        self.nn = SimpleNeuralNetwork(input_size=7, hidden_size=10, output_size=2)

        # Robot state
        self.laser_data = None
        self.goal_x = 5.0  # Goal position (x)
        self.goal_y = 5.0  # Goal position (y)
        self.robot_x = 0.0  # Current robot position (x)
        self.robot_y = 0.0  # Current robot position (y)

        # ROS interfaces
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.laser_sub = self.create_subscription(LaserScan, 'scan', self.laser_callback, 10)
        self.goal_pub = self.create_publisher(String, 'goal_status', 10)

        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('AI Robot Controller initialized')

    def laser_callback(self, msg):
        """Callback for laser scan data"""
        # Take a few key readings from the laser scan for obstacle detection
        # Front, front-left, front-right, left, right
        num_readings = len(msg.ranges)
        front_idx = num_readings // 2
        front_left_idx = num_readings // 2 - num_readings // 8
        front_right_idx = num_readings // 2 + num_readings // 8
        left_idx = num_readings // 4
        right_idx = 3 * num_readings // 4

        # Get the readings (with safety check for invalid values)
        def safe_range(val):
            if math.isinf(val) or math.isnan(val):
                return 10.0  # Return a safe default value
            return val

        self.laser_data = [
            safe_range(msg.ranges[front_idx]),      # Front distance
            safe_range(msg.ranges[front_left_idx]), # Front-left distance
            safe_range(msg.ranges[front_right_idx]),# Front-right distance
            safe_range(msg.ranges[left_idx]),       # Left distance
            safe_range(msg.ranges[right_idx])       # Right distance
        ]

    def calculate_goal_direction(self):
        """Calculate direction to goal relative to robot position"""
        dx = self.goal_x - self.robot_x
        dy = self.goal_y - self.robot_y
        distance = math.sqrt(dx*dx + dy*dy)

        # Normalize direction
        if distance > 0:
            goal_dir_x = dx / distance
            goal_dir_y = dy / distance
        else:
            goal_dir_x = 0.0
            goal_dir_y = 0.0

        return goal_dir_x, goal_dir_y, distance

    def control_loop(self):
        """Main control loop using AI decision making"""
        if self.laser_data is None:
            return

        # Calculate goal direction
        goal_dir_x, goal_dir_y, distance_to_goal = self.calculate_goal_direction()

        # Prepare input for neural network
        # Input: [front_dist, front_left_dist, front_right_dist, left_dist, right_dist, goal_dir_x, goal_dir_y]
        nn_input = np.array([self.laser_data + [goal_dir_x, goal_dir_y]])

        # Get AI decision (normalized velocities)
        nn_output = self.nn.predict(nn_input)

        # Convert normalized output to actual velocities
        linear_vel = float(nn_output[0, 0] * 0.5)  # Scale to reasonable linear velocity
        angular_vel = float(nn_output[0, 1] * 1.0)  # Scale to reasonable angular velocity

        # Safety checks to avoid collisions
        min_obstacle_dist = min(self.laser_data)
        if min_obstacle_dist < 0.5:  # If obstacle is too close
            # Emergency stop or avoidance behavior
            linear_vel = 0.0
            if min_obstacle_dist < 0.3:
                angular_vel = 0.5  # Turn away from obstacle

        # Create and publish velocity command
        twist = Twist()
        twist.linear.x = linear_vel
        twist.angular.z = angular_vel

        self.cmd_vel_pub.publish(twist)

        # Publish goal status for monitoring
        status_msg = String()
        status_msg.data = f"Distance to goal: {distance_to_goal:.2f}m, Vel: ({linear_vel:.2f}, {angular_vel:.2f})"
        self.goal_pub.publish(status_msg)

        self.get_logger().debug(f'AI Decision - Linear: {linear_vel:.2f}, Angular: {angular_vel:.2f}')


class TrainingSimulator(Node):
    """
    A training simulator that generates training data for the AI model.
    This simulates different scenarios for training the neural network.
    """
    def __init__(self):
        super().__init__('training_simulator')

        # Training data storage
        self.training_inputs = []
        self.training_outputs = []

        # Timer for generating training data
        self.timer = self.create_timer(0.5, self.generate_training_data)

        self.get_logger().info('Training Simulator initialized')

    def generate_training_data(self):
        """
        Generate training data based on simple rules.
        This simulates an expert controller providing training examples.
        """
        # Simulate sensor inputs
        front_dist = np.random.uniform(0.2, 5.0)
        front_left_dist = np.random.uniform(0.2, 5.0)
        front_right_dist = np.random.uniform(0.2, 5.0)
        left_dist = np.random.uniform(0.2, 5.0)
        right_dist = np.random.uniform(0.2, 5.0)

        # Simulate goal direction (normalized)
        goal_dir_x = np.random.uniform(-1.0, 1.0)
        goal_dir_y = np.random.uniform(-1.0, 1.0)

        # Normalize goal direction
        goal_norm = np.sqrt(goal_dir_x**2 + goal_dir_y**2)
        if goal_norm > 0:
            goal_dir_x /= goal_norm
            goal_dir_y /= goal_norm

        inputs = [front_dist, front_left_dist, front_right_dist, left_dist, right_dist, goal_dir_x, goal_dir_y]

        # Generate "expert" output based on simple rules
        linear_vel, angular_vel = self.expert_controller(inputs)

        # Store training data
        self.training_inputs.append(inputs)
        self.training_outputs.append([linear_vel, angular_vel])

        self.get_logger().debug(f'Generated training data: {inputs} -> ({linear_vel:.2f}, {angular_vel:.2f})')

        # Periodically save training data
        if len(self.training_inputs) % 100 == 0:
            self.save_training_data()

    def expert_controller(self, inputs):
        """
        Simple expert controller that provides training examples.
        This implements basic obstacle avoidance and goal seeking.
        """
        front, front_left, front_right, left, right, goal_x, goal_y = inputs

        linear_vel = 0.0
        angular_vel = 0.0

        # Obstacle avoidance
        min_front_dist = min(front, front_left, front_right)
        if min_front_dist < 0.8:
            # Too close to obstacle, slow down and turn
            linear_vel = 0.1
            if front_left < front_right:
                angular_vel = -0.3  # Turn right
            else:
                angular_vel = 0.3   # Turn left
        else:
            # Safe to move forward, also consider goal direction
            linear_vel = 0.3

            # Adjust angular velocity based on goal direction
            if goal_y > 0.5:  # Goal is more forward
                angular_vel = goal_x * 0.2  # Turn toward goal
            else:
                angular_vel = goal_x * 0.3

        # Limit velocities
        linear_vel = max(0.0, min(0.5, linear_vel))
        angular_vel = max(-0.5, min(0.5, angular_vel))

        return linear_vel, angular_vel

    def save_training_data(self):
        """Save training data to file"""
        if len(self.training_inputs) > 0:
            inputs_array = np.array(self.training_inputs)
            outputs_array = np.array(self.training_outputs)

            np.save('training_inputs.npy', inputs_array)
            np.save('training_outputs.npy', outputs_array)

            self.get_logger().info(f'Saved {len(self.training_inputs)} training examples')


def main(args=None):
    """Main function to run the AI-ROS integration example"""
    rclpy.init(args=args)

    # Choose which node to run
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        node = TrainingSimulator()
        print("Running in training mode - generating data for AI model")
    else:
        node = AIBasedRobotController()
        print("Running AI-based robot controller")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()