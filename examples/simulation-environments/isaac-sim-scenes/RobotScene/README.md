# Isaac Sim Scene Example for Educational Robotics

## Overview

This example demonstrates how to create a basic Isaac Sim scene that can be used for educational robotics applications. Isaac Sim provides high-fidelity physics simulation, photorealistic rendering, and AI training capabilities that are valuable for educational purposes.

## Prerequisites

- NVIDIA Isaac Sim (available through NVIDIA Omniverse)
- Compatible GPU with CUDA support (RTX series recommended)
- Isaac Sim Python API
- ROS 2 (Humble Hawksbill) for ROS bridge functionality

## Setup Instructions

### 1. Installing Isaac Sim

1. Download Isaac Sim from the NVIDIA Developer website
2. Install and configure Isaac Sim following the official documentation
3. Ensure your GPU drivers and CUDA installation are properly configured

### 2. Basic Isaac Sim Python Script

Here's a basic Python script to create a simple robot scene in Isaac Sim:

```python
import omni
from pxr import Gf, UsdGeom
import carb
import numpy as np

# Import Isaac Sim modules
from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.prims import RigidPrim, Articulation, RigidBody


class EducationalRobotWorld:
    def __init__(self):
        # Create a world instance
        self.world = World(stage_units_in_meters=1.0)

        # Set up the environment
        self.setup_environment()

        # Add a simple robot
        self.add_robot()

        # Add some objects for the robot to interact with
        self.add_objects()

    def setup_environment(self):
        """Set up the basic environment"""
        # Add ground plane
        self.world.scene.add_default_ground_plane()

        # Add lighting
        carb.log_info("Setting up lighting...")
        create_prim(
            prim_path="/World/Light",
            prim_type="DistantLight",
            position=np.array([0, 0, 5]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0])
        )

    def add_robot(self):
        """Add a simple educational robot to the scene"""
        # For this example, we'll create a simple differential drive robot
        # In practice, you might load a URDF or USD robot model

        # Create the robot prim
        robot_prim_path = "/World/Robot"

        # Create a simple robot with a chassis and wheels
        # Chassis
        create_prim(
            prim_path=robot_prim_path + "/chassis",
            prim_type="Cylinder",
            position=np.array([0.0, 0.0, 0.2]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),
            scale=np.array([0.3, 0.3, 0.15]),
            semantic_label="chassis"
        )

        # Left wheel
        create_prim(
            prim_path=robot_prim_path + "/left_wheel",
            prim_type="Cylinder",
            position=np.array([0.0, 0.2, 0.1]),
            orientation=np.array([0.707, 0.0, 0.0, 0.707]),  # Rotate 90 degrees around X
            scale=np.array([0.1, 0.1, 0.05]),
            semantic_label="wheel"
        )

        # Right wheel
        create_prim(
            prim_path=robot_prim_path + "/right_wheel",
            prim_type="Cylinder",
            position=np.array([0.0, -0.2, 0.1]),
            orientation=np.array([0.707, 0.0, 0.0, 0.707]),  # Rotate 90 degrees around X
            scale=np.array([0.1, 0.1, 0.05]),
            semantic_label="wheel"
        )

    def add_objects(self):
        """Add objects for the robot to interact with"""
        # Add a cube for the robot to navigate around
        self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/obstacle",
                name="obstacle",
                position=np.array([1.0, 0.0, 0.1]),
                size=0.2,
                color=np.array([0.8, 0.2, 0.2])
            )
        )

        # Add a target object
        self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/target",
                name="target",
                position=np.array([-1.0, 1.0, 0.1]),
                size=0.15,
                color=np.array([0.2, 0.8, 0.2])
            )
        )

    def run_simulation(self):
        """Run the simulation loop"""
        # Reset the world
        self.world.reset()

        # Simulation loop
        while True:
            # Step the world
            self.world.step(render=True)

            # Add your robot control logic here
            # For example, simple movement commands
            if self.world.is_playing():
                # Example: Move robot forward
                pass

            # Break condition (add your own condition)
            if carb.input.get_keyboard().is_pressed(carb.input.KeyboardInput.ESCAPE):
                print("Simulation stopped by user")
                break

    def cleanup(self):
        """Clean up the simulation"""
        self.world.clear()


# Example usage
def main():
    # Initialize Isaac Sim
    simulation = EducationalRobotWorld()

    try:
        # Run the simulation
        simulation.run_simulation()
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
    finally:
        # Clean up
        simulation.cleanup()


if __name__ == "__main__":
    main()
```

### 3. ROS Bridge Setup

To connect Isaac Sim with ROS 2, you can use the Isaac ROS bridge:

```python
# Additional imports for ROS bridge
from omni.isaac.ros_bridge.scripts import ros_bridge_node

class EducationalRobotWorldWithROS(EducationalRobotWorld):
    def __init__(self):
        super().__init__()
        self.setup_ros_bridge()

    def setup_ros_bridge(self):
        """Set up ROS bridge for communication"""
        # Create ROS bridge node
        # This allows communication between Isaac Sim and ROS 2
        pass

    def run_simulation_with_ros(self):
        """Run simulation with ROS communication"""
        self.world.reset()

        while True:
            # Step the world
            self.world.step(render=True)

            # ROS communication happens automatically through the bridge
            # You can publish/subscribe to ROS topics from your robot control code

            # Example: Publish robot position to ROS
            if self.world.is_playing():
                # Get robot position
                robot_position = self.get_robot_position()

                # Publish to ROS topic (this is conceptual - actual implementation varies)
                # ros_publisher.publish(robot_position)
                pass

            # Break condition
            if carb.input.get_keyboard().is_pressed(carb.input.KeyboardInput.ESCAPE):
                break
```

### 4. Advanced Features for Education

#### Physics Simulation
```python
# Configure physics properties for realistic simulation
from omni.isaac.core.physics_context import PhysicsContext

def configure_physics(self):
    """Configure physics for educational purposes"""
    physics_context = PhysicsContext()
    physics_context.set_gravity(9.81)  # Earth gravity
    physics_context.set_subspace_count(1)  # Single subspace
    physics_context.set_stage_units_in_meters(1.0)  # Scale
```

#### Sensor Simulation
```python
from omni.isaac.sensor import Camera, LidarRtx

def add_sensors(self):
    """Add educational sensors to the robot"""
    # Add a camera to the robot
    camera = self.world.scene.add(
        Camera(
            prim_path="/World/Robot/camera",
            name="robot_camera",
            position=np.array([0.2, 0.0, 0.2]),
            frequency=30,
            resolution=(640, 480)
        )
    )

    # Add a LIDAR sensor
    lidar = self.world.scene.add(
        LidarRtx(
            prim_path="/World/Robot/lidar",
            name="robot_lidar",
            translation=np.array([0.0, 0.0, 0.3]),
            config="Example_Rotary",
            min_range=0.1,
            max_range=10.0
        )
    )
```

## Educational Applications

### 1. Navigation and Path Planning
- Demonstrate A* and Dijkstra algorithms
- Visualize path planning in 3D
- Test obstacle avoidance strategies

### 2. Sensor Fusion
- Combine data from multiple sensors
- Implement Kalman filters
- Demonstrate SLAM concepts

### 3. AI and Machine Learning
- Reinforcement learning for robot control
- Computer vision applications
- Behavior learning from demonstrations

### 4. Multi-Robot Systems
- Coordination and communication
- Swarm robotics concepts
- Distributed algorithms

## Isaac Sim Features for Education

### 1. High-Fidelity Physics
- Accurate collision detection
- Realistic material properties
- Complex joint dynamics

### 2. Photorealistic Rendering
- High-quality visual output
- Realistic lighting conditions
- Support for domain randomization

### 3. AI Training Environment
- Built-in support for reinforcement learning
- Synthetic data generation
- Curriculum learning capabilities

## Troubleshooting

### Common Issues:

1. **GPU Requirements**: Ensure CUDA-compatible GPU with sufficient VRAM
2. **Performance**: Adjust rendering quality for real-time simulation
3. **Physics Stability**: Tune physics parameters for stable simulation
4. **ROS Bridge**: Verify network configuration and topic names

## Next Steps

- Integrate with real robot hardware
- Create curriculum-specific scenarios
- Implement advanced AI algorithms
- Connect with cloud computing resources for large-scale training