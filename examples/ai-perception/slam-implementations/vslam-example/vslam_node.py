#!/usr/bin/env python3

"""
Isaac ROS Visual SLAM Example
This node demonstrates Visual SLAM using Isaac ROS components
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster
import numpy as np
import cv2
from cv_bridge import CvBridge
import message_filters
from message_filters import ApproximateTimeSynchronizer


class IsaacVSLAMNode(Node):
    def __init__(self):
        super().__init__('isaac_vslam_node')

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Initialize VSLAM components
        self.prev_image = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.trajectory = []
        self.current_pose = np.eye(4)  # 4x4 identity matrix for pose

        # Create subscribers for stereo camera images
        self.left_image_sub = message_filters.Subscriber(self, Image, '/camera/left/image_rect_color')
        self.right_image_sub = message_filters.Subscriber(self, Image, '/camera/right/image_rect_color')
        self.left_info_sub = message_filters.Subscriber(self, CameraInfo, '/camera/left/camera_info')
        self.right_info_sub = message_filters.Subscriber(self, CameraInfo, '/camera/right/camera_info')

        # Synchronize stereo images and camera info
        self.ts = ApproximateTimeSynchronizer(
            [self.left_image_sub, self.right_image_sub, self.left_info_sub, self.right_info_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.stereo_callback)

        # Alternative: Use monocular camera if stereo not available
        self.mono_image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.mono_callback,
            10
        )

        # Create publishers
        self.odom_pub = self.create_publisher(Odometry, '/vslam/odometry', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/vslam/pose', 10)

        # Create TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Initialize feature detector (using ORB as example)
        self.feature_detector = cv2.ORB_create(nfeatures=1000)

        # Camera parameters (will be updated from camera info)
        self.camera_matrix = None
        self.dist_coeffs = None

        self.get_logger().info('Isaac VSLAM node initialized')

    def stereo_callback(self, left_msg, right_msg, left_info_msg, right_info_msg):
        """Process synchronized stereo images for VSLAM"""
        try:
            # Convert ROS images to OpenCV
            left_cv = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='bgr8')
            right_cv = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')

            # Update camera parameters from camera info
            self.camera_matrix = np.array(left_info_msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(left_info_msg.d)

            # Process stereo pair for depth estimation
            depth_map = self.compute_depth_map(left_cv, right_cv)

            # Perform visual SLAM using the depth information
            self.process_vslam(left_cv, depth_map)

        except Exception as e:
            self.get_logger().error(f'Error processing stereo images: {e}')

    def mono_callback(self, image_msg):
        """Process monocular image for VSLAM (fallback option)"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

            # Perform monocular VSLAM
            self.process_monocular_vslam(cv_image)

        except Exception as e:
            self.get_logger().error(f'Error processing monocular image: {e}')

    def compute_depth_map(self, left_image, right_image):
        """Compute depth map from stereo images"""
        # Convert to grayscale
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

        # Create stereo matcher
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,
            blockSize=11,
            P1=8 * 3 * 11**2,
            P2=32 * 3 * 11**2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        # Compute disparity
        disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

        # Convert disparity to depth
        # This is a simplified calculation - in practice, you'd use proper stereo calibration
        baseline = 0.1  # Baseline in meters (example value)
        focal_length = self.camera_matrix[0, 0] if self.camera_matrix is not None else 640  # Example focal length
        depth_map = (baseline * focal_length) / (disparity + 1e-6)  # Add small value to avoid division by zero

        return depth_map

    def process_vslam(self, image, depth_map):
        """Process stereo VSLAM"""
        # Convert to grayscale for feature detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect features
        keypoints = self.feature_detector.detect(gray)
        keypoints, descriptors = self.feature_detector.compute(gray, keypoints)

        if self.prev_keypoints is not None and len(self.prev_keypoints) > 10:
            # Match features between current and previous frames
            matcher = cv2.BFMatcher()
            matches = matcher.knnMatch(self.prev_descriptors, descriptors, k=2)

            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            if len(good_matches) >= 10:
                # Extract matched points
                prev_pts = np.float32([self.prev_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                curr_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Estimate motion using essential matrix
                E, mask = cv2.findEssentialMat(
                    curr_pts, prev_pts,
                    cameraMatrix=self.camera_matrix,
                    method=cv2.RANSAC,
                    prob=0.999,
                    threshold=1.0
                )

                if E is not None:
                    # Recover pose
                    _, R, t, mask_pose = cv2.recoverPose(E, curr_pts, prev_pts, self.camera_matrix)

                    # Update current pose
                    translation = t.flatten()
                    rotation_matrix = np.eye(4)
                    rotation_matrix[:3, :3] = R
                    rotation_matrix[:3, 3] = translation

                    self.current_pose = self.current_pose @ rotation_matrix

                    # Publish odometry
                    self.publish_odometry(self.current_pose, image_msg.header.stamp)

                    # Store trajectory
                    self.trajectory.append(self.current_pose.copy())

        # Update previous frame data
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors

    def process_monocular_vslam(self, image):
        """Process monocular VSLAM (simplified)"""
        # Convert to grayscale for feature detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect features
        keypoints = self.feature_detector.detect(gray)
        keypoints, descriptors = self.feature_detector.compute(gray, keypoints)

        if self.prev_keypoints is not None and len(self.prev_keypoints) > 10:
            # Match features between current and previous frames
            matcher = cv2.BFMatcher()
            matches = matcher.knnMatch(self.prev_descriptors, descriptors, k=2)

            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            if len(good_matches) >= 10:
                # Extract matched points
                prev_pts = np.float32([self.prev_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                curr_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Calculate optical flow to estimate motion
                # This is a simplified approach - full monocular SLAM requires scale recovery
                flow = curr_pts - prev_pts
                avg_flow = np.mean(flow, axis=0).flatten()

                # Update pose based on optical flow (simplified)
                self.current_pose[0, 3] += avg_flow[0] * 0.01  # Scale factor
                self.current_pose[1, 3] += avg_flow[1] * 0.01  # Scale factor

                # Publish odometry
                self.publish_odometry(self.current_pose, self.get_clock().now().to_msg())

                # Store trajectory
                self.trajectory.append(self.current_pose.copy())

        # Update previous frame data
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors

    def publish_odometry(self, pose_matrix, timestamp):
        """Publish odometry and pose messages"""
        # Create odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = timestamp
        odom_msg.header.frame_id = 'map'
        odom_msg.child_frame_id = 'vslam_camera'

        # Convert pose matrix to position and orientation
        position = pose_matrix[:3, 3]
        rotation_matrix = pose_matrix[:3, :3]

        # Convert rotation matrix to quaternion
        qw = np.sqrt(1 + rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2]) / 2
        qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / (4 * qw)
        qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / (4 * qw)
        qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / (4 * qw)

        odom_msg.pose.pose.position.x = position[0]
        odom_msg.pose.pose.position.y = position[1]
        odom_msg.pose.pose.position.z = position[2]
        odom_msg.pose.pose.orientation.w = qw
        odom_msg.pose.pose.orientation.x = qx
        odom_msg.pose.pose.orientation.y = qy
        odom_msg.pose.pose.orientation.z = qz

        # Publish odometry
        self.odom_pub.publish(odom_msg)

        # Create and publish pose message
        pose_msg = PoseStamped()
        pose_msg.header.stamp = timestamp
        pose_msg.header.frame_id = 'map'
        pose_msg.pose = odom_msg.pose.pose
        self.pose_pub.publish(pose_msg)

        # Broadcast transform
        t = TransformStamped()
        t.header.stamp = timestamp
        t.header.frame_id = 'map'
        t.child_frame_id = 'vslam_camera'
        t.transform.translation.x = position[0]
        t.transform.translation.y = position[1]
        t.transform.translation.z = position[2]
        t.transform.rotation.w = qw
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz

        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = IsaacVSLAMNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()