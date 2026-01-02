#!/usr/bin/env python3

"""
ROS 2 Service Client Example

This example demonstrates how to create a service client in ROS 2.
The client calls the 'add_two_ints' service to add two integers.
"""

import sys
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts


class AddTwoIntsClient(Node):
    """A service client that calls the add_two_ints service."""

    def __init__(self):
        """Initialize the service client node."""
        super().__init__('add_two_ints_client')

        # Create a service client for the AddTwoInts service
        self.client = self.create_client(AddTwoInts, 'add_two_ints')

        # Wait for the service to be available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        # Create a request object
        self.request = AddTwoInts.Request()

        # Log initialization
        self.get_logger().info('Add Two Ints Client node initialized')

    def send_request(self, a, b):
        """
        Send a request to the service server.

        Args:
            a (int): First integer to add
            b (int): Second integer to add

        Returns:
            AddTwoInts.Response: The response from the service server
        """
        # Set the request parameters
        self.request.a = a
        self.request.b = b

        # Make an asynchronous service call
        self.future = self.client.call_async(self.request)

        # Wait for the response
        rclpy.spin_until_future_complete(self, self.future)

        # Return the response
        return self.future.result()


def main(args=None):
    """Main function to run the service client node."""
    # Initialize the ROS 2 client library
    rclpy.init(args=args)

    # Create the service client node
    add_two_ints_client = AddTwoIntsClient()

    # Check if command line arguments are provided
    if len(sys.argv) != 3:
        print('Usage: ros2 run ros2_basics add_two_ints_client <int1> <int2>')
        return 1

    try:
        # Parse command line arguments
        a = int(sys.argv[1])
        b = int(sys.argv[2])

        # Send the request and get the response
        response = add_two_ints_client.send_request(a, b)

        # Log the result
        add_two_ints_client.get_logger().info(
            f'Result of add_two_ints: {a} + {b} = {response.sum}'
        )

    except ValueError:
        # Handle invalid input
        add_two_ints_client.get_logger().error(
            'Please provide two valid integers as arguments'
        )
        return 1
    except Exception as e:
        # Handle other exceptions
        add_two_ints_client.get_logger().error(f'Error: {e}')
        return 1
    finally:
        # Clean up resources
        add_two_ints_client.destroy_node()
        rclpy.shutdown()

    return 0


if __name__ == '__main__':
    main()