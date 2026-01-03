#!/usr/bin/env python3

"""
Simple ROS 2 Publisher Example

This example demonstrates the basic publisher-subscriber pattern in ROS 2.
The publisher sends "Hello World" messages to a topic at regular intervals.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SimplePublisher(Node):
    """A simple publisher node that sends messages to a topic."""

    def __init__(self):
        """Initialize the publisher node."""
        super().__init__('simple_publisher')

        # Create a publisher for the 'chatter' topic with String messages
        self.publisher = self.create_publisher(String, 'chatter', 10)

        # Create a timer to publish messages every 0.5 seconds
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Counter to track message sequence
        self.i = 0

        # Log initialization
        self.get_logger().info('Simple Publisher node initialized')

    def timer_callback(self):
        """Callback function that publishes messages at regular intervals."""
        # Create a new String message
        msg = String()
        msg.data = f'Hello World: {self.i}'

        # Publish the message
        self.publisher.publish(msg)

        # Log the published message
        self.get_logger().info(f'Publishing: "{msg.data}"')

        # Increment the counter
        self.i += 1


def main(args=None):
    """Main function to run the publisher node."""
    # Initialize the ROS 2 client library
    rclpy.init(args=args)

    # Create the publisher node
    simple_publisher = SimplePublisher()

    try:
        # Keep the node running until interrupted
        rclpy.spin(simple_publisher)
    except KeyboardInterrupt:
        # Handle graceful shutdown when interrupted
        simple_publisher.get_logger().info('Shutting down publisher node')
    finally:
        # Clean up resources
        simple_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()