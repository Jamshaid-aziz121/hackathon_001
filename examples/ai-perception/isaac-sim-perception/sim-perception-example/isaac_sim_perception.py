#!/usr/bin/env python3

"""
Isaac Sim Perception Example
This script demonstrates perception capabilities within Isaac Sim environment
"""

import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.robots import Robot
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.prims import RigidPrim
from omni.isaac.sensor import Camera, LidarRtx
from omni.isaac.core.materials import OmniPBR
from omni.isaac.core.utils.carb import set_carb_setting
from pxr import Gf, UsdGeom
import carb
import numpy as np
import argparse
import sys
import os


class IsaacSimPerceptionExample:
    def __init__(self):
        # Initialize Isaac Sim
        self._simulation_app = omni.simulator.SimulationApp({"headless": False})

        # Set up the world
        self.world = World(stage_units_in_meters=1.0)

        # Initialize components
        self.robot = None
        self.camera = None
        self.lidar = None
        self.objects = []

        # Set up the environment
        self.setup_environment()

        # Add perception sensors to robot
        self.add_perception_sensors()

        # Add objects for perception
        self.add_perception_objects()

    def setup_environment(self):
        """Set up the basic environment"""
        # Add ground plane
        self.world.scene.add_default_ground_plane()

        # Add lighting
        carb.log_info("Setting up lighting...")
        from omni.isaac.core.utils.prims import create_prim
        create_prim(
            prim_path="/World/Light",
            prim_type="DistantLight",
            position=np.array([0, 0, 5]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0])
        )

        # Set up default physics parameters
        self.world.get_physics_context().set_gravity(9.81)

    def add_perception_sensors(self):
        """Add perception sensors to the robot"""
        # Create a simple robot with sensors
        from omni.isaac.core.utils.prims import create_prim

        # Create robot prim
        robot_prim_path = "/World/Robot"

        # Create robot body
        create_prim(
            prim_path=robot_prim_path + "/chassis",
            prim_type="Cylinder",
            position=np.array([0.0, 0.0, 0.2]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),
            scale=np.array([0.3, 0.3, 0.15]),
            semantic_label="chassis"
        )

        # Add wheels
        create_prim(
            prim_path=robot_prim_path + "/left_wheel",
            prim_type="Cylinder",
            position=np.array([0.0, 0.15, 0.1]),
            orientation=np.array([0.707, 0.0, 0.0, 0.707]),
            scale=np.array([0.1, 0.1, 0.05]),
            semantic_label="wheel"
        )

        create_prim(
            prim_path=robot_prim_path + "/right_wheel",
            prim_type="Cylinder",
            position=np.array([0.0, -0.15, 0.1]),
            orientation=np.array([0.707, 0.0, 0.0, 0.707]),
            scale=np.array([0.1, 0.1, 0.05]),
            semantic_label="wheel"
        )

        # Add camera sensor
        self.camera = self.world.scene.add(
            Camera(
                prim_path=robot_prim_path + "/camera",
                name="robot_camera",
                position=np.array([0.2, 0.0, 0.2]),
                frequency=30,
                resolution=(640, 480)
            )
        )

        # Add LIDAR sensor
        self.lidar = self.world.scene.add(
            LidarRtx(
                prim_path=robot_prim_path + "/lidar",
                name="robot_lidar",
                translation=np.array([0.0, 0.0, 0.3]),
                config="Example_Rotary",
                min_range=0.1,
                max_range=10.0
            )
        )

    def add_perception_objects(self):
        """Add objects for the robot to perceive"""
        # Add colored cubes for object detection
        colors = [
            np.array([1.0, 0.0, 0.0]),  # Red
            np.array([0.0, 1.0, 0.0]),  # Green
            np.array([0.0, 0.0, 1.0]),  # Blue
            np.array([1.0, 1.0, 0.0]),  # Yellow
        ]

        positions = [
            np.array([1.0, 0.5, 0.1]),
            np.array([1.0, -0.5, 0.1]),
            np.array([2.0, 0.0, 0.1]),
            np.array([2.0, 1.0, 0.1]),
        ]

        for i, (color, pos) in enumerate(zip(colors, positions)):
            obj = self.world.scene.add(
                DynamicCuboid(
                    prim_path=f"/World/Object_{i}",
                    name=f"object_{i}",
                    position=pos,
                    size=0.2,
                    color=color
                )
            )
            self.objects.append(obj)

    def run_perception_demo(self):
        """Run the perception demonstration"""
        # Reset the world
        self.world.reset()

        # Main simulation loop
        frame_count = 0
        max_frames = 300  # Run for about 10 seconds at 30 FPS

        while frame_count < max_frames:
            # Step the world
            self.world.step(render=True)

            if self.world.is_playing():
                # Process perception data every few frames
                if frame_count % 10 == 0:
                    self.process_perception_data()

            frame_count += 1

            # Check for exit condition
            if carb.input.get_keyboard().is_pressed(carb.input.KeyboardInput.ESCAPE):
                print("Simulation stopped by user")
                break

    def process_perception_data(self):
        """Process perception data from sensors"""
        # Get camera data
        try:
            camera_data = self.camera.get_rgb()
            if camera_data is not None:
                print(f"Camera data shape: {camera_data.shape}")

                # Simple color-based object detection simulation
                self.simulate_object_detection(camera_data)
        except Exception as e:
            carb.log_warn(f"Camera data error: {e}")

        # Get LIDAR data
        try:
            lidar_data = self.lidar.get_linear_depth_data()
            if lidar_data is not None:
                print(f"LIDAR data points: {len(lidar_data)}")

                # Process LIDAR data for obstacle detection
                self.process_lidar_data(lidar_data)
        except Exception as e:
            carb.log_warn(f"LIDAR data error: {e}")

    def simulate_object_detection(self, image_data):
        """Simulate object detection from camera data"""
        # This is a simplified simulation of object detection
        # In a real implementation, you would use a trained model

        # For demonstration, we'll just log that we're processing the image
        height, width, channels = image_data.shape
        print(f"Processing {width}x{height} image for object detection")

        # In a real implementation, you would:
        # 1. Preprocess the image
        # 2. Run it through a detection model
        # 3. Process the results
        # 4. Publish detection messages via ROS bridge if needed

    def process_lidar_data(self, lidar_data):
        """Process LIDAR data for obstacle detection"""
        # Find minimum distance to detect closest obstacles
        if len(lidar_data) > 0:
            min_distance = np.min(lidar_data[np.isfinite(lidar_data)])
            if min_distance < 1.0:  # If something is within 1 meter
                print(f"Obstacle detected at {min_distance:.2f}m")

    def cleanup(self):
        """Clean up the simulation"""
        self.world.clear()
        self._simulation_app.close()


def main():
    """Main function to run the Isaac Sim perception example"""
    perception_example = IsaacSimPerceptionExample()

    try:
        perception_example.run_perception_demo()
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
    finally:
        perception_example.cleanup()


if __name__ == "__main__":
    main()