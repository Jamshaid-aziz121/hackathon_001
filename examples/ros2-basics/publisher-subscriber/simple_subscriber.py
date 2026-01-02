#!/usr/bin/env python3

"""
Simple ROS 2 Subscriber Example

This example demonstrates the basic publisher-subscriber pattern in ROS 2.
The subscriber receives messages from the 'chatter' topic and logs them.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SimpleSubscriber(Node):
    """A simple subscriber node that receives messages from a topic."""

    def __init__(self):
        """Initialize the subscriber node."""
        super().__init__('simple_subscriber')

        # Create a subscription to the 'chatter' topic with String messages
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10  # QoS depth
        )

        # Prevent unused variable warning
        self.subscription  # pylint: disable=pointless-statement

        # Log initialization
        self.get_logger().info('Simple Subscriber node initialized')

    def listener_callback(self, msg):
        """
        Callback function that is called when a message is received.

        Args:
            msg (String): The received message containing the data
        """
        # Log the received message
        self.get_logger().info(f'I heard: "{msg.data}"')


def main(args=None):
    """Main function to run the subscriber node."""
    # Initialize the ROS 2 client library
    rclpy.init(args=args)

    # Create the subscriber node
    simple_subscriber = SimpleSubscriber()

    try:
        # Keep the node running until interrupted
        rclpy.spin(simple_subscriber)
    except KeyboardInterrupt:
        # Handle graceful shutdown when interrupted
        simple_subscriber.get_logger().info('Shutting down subscriber node')
    finally:
        # Clean up resources
        simple_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()