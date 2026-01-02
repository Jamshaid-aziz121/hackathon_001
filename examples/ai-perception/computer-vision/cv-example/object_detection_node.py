#!/usr/bin/env python3

"""
Computer Vision Object Detection Example
This node demonstrates object detection using OpenCV and ROS 2
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import threading
from collections import defaultdict


class ComputerVisionNode(Node):
    def __init__(self):
        super().__init__('computer_vision_node')

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Initialize detection parameters
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4

        # Class labels for common objects
        self.class_labels = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        # Create subscriber for camera images
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Create publisher for detections
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/cv_detections',
            10
        )

        # Create publisher for visualization
        self.vis_pub = self.create_publisher(
            Image,
            '/cv_visualization',
            10
        )

        # Create publisher for status
        self.status_pub = self.create_publisher(
            String,
            '/cv_status',
            10
        )

        # Detection history for tracking
        self.detection_history = defaultdict(list)

        # Thread lock for processing
        self.processing_lock = threading.Lock()

        self.get_logger().info('Computer vision node initialized')

    def image_callback(self, msg):
        """Process incoming camera images"""
        with self.processing_lock:
            try:
                # Convert ROS Image to OpenCV image
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except Exception as e:
                self.get_logger().error(f'Error converting image: {e}')
                return

            # Perform object detection
            detections = self.perform_object_detection(cv_image)

            # Create Detection2DArray message
            detection_array = Detection2DArray()
            detection_array.header = msg.header

            # Process detections
            for detection in detections:
                detection_msg = Detection2D()
                detection_msg.header = msg.header

                # Set bounding box
                bbox = detection['bbox']
                detection_msg.bbox.size_x = bbox[2]  # width
                detection_msg.bbox.size_y = bbox[3]  # height
                detection_msg.bbox.center.x = bbox[0] + bbox[2] / 2  # center x
                detection_msg.bbox.center.y = bbox[1] + bbox[3] / 2  # center y

                # Set hypothesis
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = str(detection['class_id'])
                hypothesis.hypothesis.score = detection['confidence']
                detection_msg.results.append(hypothesis)

                detection_array.detections.append(detection_msg)

            # Publish detections
            self.detection_pub.publish(detection_array)

            # Create visualization image
            vis_image = self.draw_detections(cv_image, detections)
            vis_msg = self.bridge.cv2_to_imgmsg(vis_image, encoding='bgr8')
            vis_msg.header = msg.header
            self.vis_pub.publish(vis_msg)

            # Publish status
            status_msg = String()
            status_msg.data = f'Detected {len(detections)} objects'
            self.status_pub.publish(status_msg)

            self.get_logger().info(f'Detected {len(detections)} objects')

    def perform_object_detection(self, image):
        """
        Perform object detection using OpenCV DNN module
        This is a simplified example using a mock model
        In a real implementation, you would load a pre-trained model
        """
        # This is a mock implementation - in reality, you would use:
        # - A pre-trained model (YOLO, SSD, etc.)
        # - OpenCV DNN module with a model file
        # - Or Isaac ROS hardware-accelerated detection

        # For demonstration, create some mock detections based on image analysis
        height, width = image.shape[:2]

        # Use color-based detection as a simple example
        detections = []

        # Detect red objects (potential stop signs, apples, etc.)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_red = (0, 50, 50)
        upper_red = (10, 255, 255)
        mask_red = cv2.inRange(hsv, lower_red, upper_red)

        # Find contours of red regions
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours_red:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter small areas
                x, y, w, h = cv2.boundingRect(contour)
                if w > 20 and h > 20:  # Minimum size filter
                    detections.append({
                        'bbox': [x, y, w, h],
                        'class_id': 9,  # stop sign
                        'confidence': 0.7
                    })

        # Detect blue objects (potential sky, water, etc.)
        lower_blue = (100, 50, 50)
        upper_blue = (130, 255, 255)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours_blue:
            area = cv2.contourArea(contour)
            if area > 1000:  # Larger area for blue objects
                x, y, w, h = cv2.boundingRect(contour)
                if w > 30 and h > 30:
                    detections.append({
                        'bbox': [x, y, w, h],
                        'class_id': 0,  # person (or another class)
                        'confidence': 0.6
                    })

        # Detect rectangular objects (potential books, laptops, etc.)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 800:  # Filter for medium-sized objects
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Check if it's rectangular (4 sides)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    if 0.5 < aspect_ratio < 2.0:  # Reasonable aspect ratio
                        detections.append({
                            'bbox': [x, y, w, h],
                            'class_id': 62,  # laptop (or another class)
                            'confidence': 0.65
                        })

        return detections

    def draw_detections(self, image, detections):
        """Draw detection results on the image"""
        output_image = image.copy()

        for detection in detections:
            bbox = detection['bbox']
            class_id = detection['class_id']
            confidence = detection['confidence']

            # Draw bounding box
            x, y, w, h = bbox
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw label and confidence
            label = f"{self.class_labels[class_id] if class_id < len(self.class_labels) else f'Class {class_id}'}: {confidence:.2f}"
            cv2.putText(output_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return output_image


class YOLOObjectDetectionNode(ComputerVisionNode):
    """
    A more advanced version using YOLO for object detection
    This would be used in a real implementation with actual YOLO weights
    """
    def __init__(self):
        super().__init__()

        # YOLO parameters
        self.yolo_confidence_threshold = 0.5
        self.yolo_nms_threshold = 0.4

        # Initialize YOLO model (mock initialization)
        # In a real implementation, you would load YOLO weights here
        self.yolo_model = None
        self.yolo_layers = []
        self.yolo_output_layers = []

        self.get_logger().info('YOLO-based computer vision node initialized')

    def perform_object_detection(self, image):
        """
        Perform object detection using YOLO
        This is a mock implementation - in reality, you would use actual YOLO
        """
        # This is a placeholder for YOLO implementation
        # Real implementation would:
        # 1. Preprocess image for YOLO input
        # 2. Run inference through YOLO model
        # 3. Post-process outputs to get detections

        # For demonstration, return the same as base class
        return super().perform_object_detection(image)


def main(args=None):
    rclpy.init(args=args)

    # Choose which node to run
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'yolo':
        node = YOLOObjectDetectionNode()
        print("Running YOLO-based object detection")
    else:
        node = ComputerVisionNode()
        print("Running basic computer vision detection")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()