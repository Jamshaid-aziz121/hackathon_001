# Isaac Sim Technical Setup for Educational Robotics

## System Requirements

### Hardware Requirements
- **GPU**: NVIDIA RTX series (RTX 3070 or higher recommended)
- **VRAM**: 8GB or more
- **RAM**: 16GB or more
- **CPU**: Multi-core processor (Intel i7 or AMD Ryzen 7)
- **Storage**: SSD with 20GB+ free space

### Software Requirements
- **OS**: Ubuntu 20.04 LTS or Windows 10/11
- **GPU Drivers**: NVIDIA drivers supporting CUDA 11.8+
- **CUDA**: CUDA Toolkit 11.8 or later
- **Isaac Sim**: Latest version from NVIDIA Developer Zone
- **Python**: 3.8 or later (usually included with Isaac Sim)

## Installation Process

### 1. Install NVIDIA GPU Drivers
```bash
# For Ubuntu
sudo apt update
sudo apt install nvidia-driver-535  # Or latest available version
sudo reboot
```

### 2. Install Isaac Sim
1. Download Isaac Sim from NVIDIA Developer website
2. Run the installer:
```bash
chmod +x isaac-sim-*.run
sudo ./isaac-sim-*.run
```
3. Follow the installation wizard

### 3. Verify Installation
```bash
# Launch Isaac Sim
./isaac-sim/python.sh -c "import omni; print('Isaac Sim installed successfully')"
```

## Isaac Sim Python API

### Core Modules
```python
# World and simulation control
from omni.isaac.core import World
from omni.isaac.core.simulation_app import SimulationApp

# Robot and object creation
from omni.isaac.core.robots import Robot
from omni.isaac.core.objects import DynamicCuboid, DynamicSphere
from omni.isaac.core.prims import RigidPrim, Articulation

# Utils for scene setup
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import create_prim

# Physics and rendering
from omni.isaac.core.physics_context import PhysicsContext
from omni.isaac.core.render_context import RenderProduct
```

### Basic Scene Creation
```python
import omni
from omni.isaac.core import World
import numpy as np

# Initialize Isaac Sim
config = {"headless": False}  # Set to True for headless mode
simulation_app = SimulationApp(config)

# Create world
world = World(stage_units_in_meters=1.0)

# Add ground plane
world.scene.add_default_ground_plane()

# Add objects
from omni.isaac.core.objects import DynamicCuboid
cube = world.scene.add(
    DynamicCuboid(
        prim_path="/World/cube",
        name="my_cube",
        position=np.array([0.5, 0.5, 0.5]),
        size=0.2,
        color=np.array([0.8, 0.1, 0.1])
    )
)

# Reset and step the world
world.reset()
for i in range(100):
    world.step(render=True)

# Shutdown
simulation_app.close()
```

## Robot Integration

### Loading Robot Models
```python
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot

def load_robot(world, robot_path, prim_path, position):
    """Load a robot from USD file"""
    # Add robot to stage
    add_reference_to_stage(usd_path=robot_path, prim_path=prim_path)

    # Add robot to world
    robot = world.scene.add(
        Robot(
            prim_path=prim_path,
            name="my_robot",
            position=position
        )
    )

    return robot
```

### Robot Control
```python
def control_robot(robot, joint_positions):
    """Control robot joints"""
    # Set joint positions
    robot.set_joints_default_state(positions=joint_positions)

    # Apply actions
    robot.apply_action(robot.get_articulation_controller().forward(
        joint_positions=joint_positions
    ))
```

## ROS Integration

### Setting up ROS Bridge
```python
# Install ROS bridge extension
# In Isaac Sim, go to Window > Extensions > Isaac ROS Bridge

# Example ROS publisher in Isaac Sim
from geometry_msgs.msg import Twist
import rclpy

def setup_ros_publisher():
    """Setup ROS publisher for robot control"""
    rclpy.init()
    node = rclpy.create_node('isaac_sim_controller')
    publisher = node.create_publisher(Twist, '/cmd_vel', 10)
    return node, publisher
```

### ROS Subscriptions
```python
def setup_ros_subscriber():
    """Setup ROS subscriber for sensor data"""
    rclpy.init()
    node = rclpy.create_node('isaac_sim_sensor')

    def sensor_callback(msg):
        # Process sensor data
        pass

    subscriber = node.create_subscription(
        LaserScan,
        '/scan',
        sensor_callback,
        10
    )

    return node, subscriber
```

## Educational Scenarios

### 1. Basic Navigation Scenario
```python
class NavigationScenario:
    def __init__(self, world):
        self.world = world
        self.setup_scenario()

    def setup_scenario(self):
        """Setup a basic navigation scenario"""
        # Add robot
        self.robot = self.world.scene.add(
            Robot(
                prim_path="/World/Robot",
                name="nav_robot",
                position=np.array([0, 0, 0.5])
            )
        )

        # Add obstacles
        for i in range(5):
            self.world.scene.add(
                DynamicCuboid(
                    prim_path=f"/World/Obstacle_{i}",
                    name=f"obstacle_{i}",
                    position=np.array([i, 0, 0.2]),
                    size=0.3
                )
            )

        # Add goal
        self.world.scene.add(
            DynamicSphere(
                prim_path="/World/Goal",
                name="goal",
                position=np.array([5, 5, 0.2]),
                radius=0.2,
                color=np.array([0, 1, 0])
            )
        )

    def run_navigation(self):
        """Run navigation algorithm"""
        # Implement navigation logic
        pass
```

### 2. Sensor Fusion Scenario
```python
from omni.isaac.sensor import Camera, LidarRtx

class SensorFusionScenario:
    def __init__(self, world):
        self.world = world
        self.setup_sensors()

    def setup_sensors(self):
        """Add multiple sensors to robot"""
        # Add camera
        self.camera = self.world.scene.add(
            Camera(
                prim_path="/World/Robot/Camera",
                name="robot_camera",
                position=np.array([0.2, 0, 0.2]),
                frequency=30,
                resolution=(640, 480)
            )
        )

        # Add LIDAR
        self.lidar = self.world.scene.add(
            LidarRtx(
                prim_path="/World/Robot/Lidar",
                name="robot_lidar",
                translation=np.array([0.0, 0.0, 0.3]),
                config="Example_Rotary",
                min_range=0.1,
                max_range=10.0
            )
        )

    def process_sensor_data(self):
        """Process and fuse sensor data"""
        # Get camera data
        camera_data = self.camera.get_rgb()

        # Get LIDAR data
        lidar_data = self.lidar.get_linear_depth_data()

        # Implement sensor fusion algorithm
        pass
```

## Performance Optimization

### Physics Optimization
```python
def optimize_physics(world):
    """Optimize physics settings for better performance"""
    physics_context = world.physics_sim_view
    physics_context.set_subspace_count(1)
    physics_context.set_enabled_gpu_dynamics(True)
    physics_context.set_broadphase_type("GPU")
```

### Rendering Optimization
```python
def optimize_rendering(simulation_app):
    """Optimize rendering settings"""
    # Reduce rendering quality for better performance
    carb.settings.get_settings().set("/app/window/syncToRender", False)
    carb.settings.get_settings().set("/rtx/aa/op", "none")
```

## Debugging and Visualization

### Debugging Tools
```python
def enable_debugging():
    """Enable Isaac Sim debugging features"""
    # Enable physics debug visualization
    carb.settings.get_settings().set("/app/renderer/debugDraw", True)

    # Enable collision visualization
    carb.settings.get_settings().set("/persistent/isaac/DebugDraw/CollisionGroups", True)
```

### Visualization Markers
```python
from omni.isaac.debug_draw import DebugDraw
import omni.kit.commands

def draw_path(path_points):
    """Draw a path for visualization"""
    debug_draw = DebugDraw()

    for i in range(len(path_points) - 1):
        start = path_points[i]
        end = path_points[i + 1]

        # Draw line between points
        debug_draw.draw_line(
            start_point=start,
            end_point=end,
            color=(1.0, 0.0, 0.0),  # Red color
            thickness=0.01
        )
```

## Common Issues and Solutions

### 1. Performance Issues
- **Solution**: Reduce physics update rate or simplify scene geometry
- **Settings**: Adjust `/physics/timeStepsPerSecond` in settings

### 2. GPU Memory Issues
- **Solution**: Reduce texture resolution or use lower quality settings
- **Settings**: Adjust `/renderer/quality/level` in settings

### 3. Physics Instability
- **Solution**: Increase solver iterations or reduce timestep
- **Settings**: Adjust `/physics/solverType` and `/physics/timeStep`

### 4. ROS Connection Issues
- **Solution**: Verify network configuration and firewall settings
- **Settings**: Check ROS_IP and ROS_MASTER_URI environment variables

## Best Practices for Education

### 1. Modularity
- Create reusable components for different scenarios
- Use configuration files for easy parameter adjustment
- Implement clear interfaces between components

### 2. Documentation
- Comment code thoroughly for student understanding
- Provide usage examples for each component
- Create step-by-step tutorials

### 3. Safety
- Implement safety checks and limits
- Provide error handling for common issues
- Create recovery procedures for simulation failures

## Resources for Further Learning

- Isaac Sim Documentation: https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html
- NVIDIA Isaac ROS: https://github.com/NVIDIA-ISAAC-ROS
- Isaac Sim Examples: Included with installation
- Omniverse Forum: https://forums.developer.nvidia.com/c/omniverse/simulation/