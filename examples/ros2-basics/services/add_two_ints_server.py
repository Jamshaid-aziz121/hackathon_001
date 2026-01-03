#!/usr/bin/env python3

"""
ROS 2 Service Server Example

This example demonstrates how to create a service server in ROS 2.
The server provides an 'add_two_ints' service that adds two integers.
"""

import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts


class AddTwoIntsServer(Node):
    """A service server that adds two integers."""

    def __init__(self):
        """Initialize the service server node."""
        super().__init__('add_two_ints_server')

        # Create a service server for the AddTwoInts service
        self.srv = self.create_service(
            AddTwoInts,
            'add_two_ints',
            self.add_two_ints_callback
        )

        # Log initialization
        self.get_logger().info('Add Two Ints Server node initialized')

    def add_two_ints_callback(self, request, response):
        """
        Callback function that handles service requests.

        Args:
            request (AddTwoInts.Request): The service request containing two integers
            response (AddTwoInts.Response): The service response containing the sum

        Returns:
            AddTwoInts.Response: The response with the sum of the two integers
        """
        # Perform the addition
        response.sum = request.a + request.b

        # Log the operation
        self.get_logger().info(
            f'Incoming request: a={request.a}, b={request.b}, sum={response.sum}'
        )

        # Return the response
        return response


def main(args=None):
    """Main function to run the service server node."""
    # Initialize the ROS 2 client library
    rclpy.init(args=args)

    # Create the service server node
    add_two_ints_server = AddTwoIntsServer()

    try:
        # Keep the node running until interrupted
        rclpy.spin(add_two_ints_server)
    except KeyboardInterrupt:
        # Handle graceful shutdown when interrupted
        add_two_ints_server.get_logger().info('Shutting down service server')
    finally:
        # Clean up resources
        add_two_ints_server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()