#!/usr/bin/env python3

"""
Nav2 Navigation Example Node
This node demonstrates autonomous navigation using Nav2 and perception integration
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import String
import tf2_ros
from tf2_ros import TransformException
from tf2_geometry_msgs import do_transform_pose
import numpy as np
import math


class Nav2NavigationNode(Node):
    def __init__(self):
        super().__init__('nav2_navigation_node')

        # Initialize action client for navigation
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Initialize TF buffer for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Create subscribers
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # Create publishers
        self.status_pub = self.create_publisher(
            String,
            '/navigation_status',
            10
        )

        # Navigation state
        self.current_pose = None
        self.map_data = None
        self.navigation_goal = None
        self.is_navigating = False

        # Navigation parameters
        self.safety_distance = 0.5  # meters
        self.goal_tolerance = 0.5   # meters

        # Timer for navigation safety checks
        self.safety_timer = self.create_timer(0.5, self.safety_check)

        self.get_logger().info('Nav2 navigation node initialized')

    def odom_callback(self, msg):
        """Update current robot pose from odometry"""
        self.current_pose = msg.pose.pose

    def map_callback(self, msg):
        """Store map data"""
        self.map_data = msg

    def scan_callback(self, msg):
        """Process laser scan for obstacle detection"""
        # Check for obstacles in front of robot
        front_scan = msg.ranges[len(msg.ranges)//2 - 30:len(msg.ranges)//2 + 30]
        min_distance = min([r for r in front_scan if not (math.isinf(r) or math.isnan(r))], default=float('inf'))

        if min_distance < self.safety_distance and self.is_navigating:
            self.get_logger().warn(f'Obstacle detected at {min_distance:.2f}m, pausing navigation')
            # In a real implementation, you would send a cancel goal request
            # self.nav_to_pose_client.cancel_goal()

    def safety_check(self):
        """Periodic safety checks during navigation"""
        if not self.is_navigating:
            return

        # Check if current pose is valid
        if self.current_pose is None:
            self.get_logger().warn('No current pose available')
            return

        # Additional safety checks can be added here
        # For example: check if robot is stuck, verify goal is still valid, etc.

    def send_navigation_goal(self, x, y, theta=0.0):
        """Send navigation goal to Nav2"""
        # Wait for action server
        if not self.nav_to_pose_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Navigation action server not available')
            return False

        # Create navigation goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        # Set goal position
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0

        # Convert theta to quaternion
        goal_msg.pose.pose.orientation.z = math.sin(theta / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(theta / 2.0)

        # Send goal
        self.get_logger().info(f'Sending navigation goal to ({x:.2f}, {y:.2f})')
        self.navigation_goal = (x, y)
        self.is_navigating = True

        # Send goal asynchronously
        future = self.nav_to_pose_client.send_goal_async(
            goal_msg,
            feedback_callback=self.navigation_feedback_callback
        )

        future.add_done_callback(self.navigation_result_callback)
        return True

    def navigation_feedback_callback(self, feedback_msg):
        """Handle navigation feedback"""
        self.get_logger().debug(f'Navigation feedback: {feedback_msg.feedback.distance_remaining:.2f}m remaining')

    def navigation_result_callback(self, future):
        """Handle navigation result"""
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().info('Navigation goal rejected')
                self.is_navigating = False
                return

            self.get_logger().info('Navigation goal accepted, waiting for result...')
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self.navigation_complete_callback)

        except Exception as e:
            self.get_logger().error(f'Exception in navigation result: {e}')
            self.is_navigating = False

    def navigation_complete_callback(self, future):
        """Handle completion of navigation"""
        try:
            result = future.result().result
            self.get_logger().info(f'Navigation completed with result: {result}')
        except Exception as e:
            self.get_logger().error(f'Exception in navigation complete: {e}')
        finally:
            self.is_navigating = False

            # Publish navigation status
            status_msg = String()
            status_msg.data = 'navigation_completed'
            self.status_pub.publish(status_msg)

    def get_safe_navigation_goals(self):
        """Identify safe navigation goals based on map data"""
        if self.map_data is None:
            return []

        safe_goals = []
        map_width = self.map_data.info.width
        map_height = self.map_data.info.height
        resolution = self.map_data.info.resolution

        # Sample potential goals in a grid pattern
        for i in range(0, map_width, 20):  # Every 20 cells
            for j in range(0, map_height, 20):
                # Convert map coordinates to world coordinates
                world_x = (i * resolution) + self.map_data.info.origin.position.x
                world_y = (j * resolution) + self.map_data.info.origin.position.y

                # Check if this cell is free (value 0) and not unknown (-1)
                map_index = j * map_width + i
                if 0 <= map_index < len(self.map_data.data):
                    cell_value = self.map_data.data[map_index]
                    if cell_value >= 0 and cell_value < 50:  # Free space (less than 50% occupied)
                        safe_goals.append((world_x, world_y))

        return safe_goals

    def navigate_to_safe_goals(self):
        """Navigate to a series of safe goals"""
        safe_goals = self.get_safe_navigation_goals()
        if not safe_goals:
            self.get_logger().warn('No safe navigation goals found')
            return

        self.get_logger().info(f'Found {len(safe_goals)} safe navigation goals')

        # For this example, navigate to the first safe goal
        if safe_goals:
            x, y = safe_goals[0]
            self.send_navigation_goal(x, y)


def main(args=None):
    rclpy.init(args=args)
    node = Nav2NavigationNode()

    try:
        # Example: Navigate to a specific goal
        # In a real implementation, you might call this based on perception results
        node.send_navigation_goal(2.0, 2.0, 0.0)  # Navigate to (2, 2) with 0 rotation

        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()