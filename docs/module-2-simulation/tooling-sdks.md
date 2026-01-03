# Simulation Tooling and SDKs

## Gazebo Garden Setup

### Installation

For Ubuntu 22.04 with ROS 2 Humble:
```bash
sudo apt install gz-harmonic
```

### Basic Commands

Launch Gazebo:
```bash
gz sim
```

Launch with a specific world:
```bash
gz sim -r empty.sdf
```

### Gazebo Tools

- `gz sim`: Main simulation engine
- `gz topic`: Command-line interface for topics
- `gz service`: Command-line interface for services
- `gz model`: Model manipulation tools

## ROS 2 Integration

### Gazebo ROS Packages

Install the ROS 2 Gazebo packages:
```bash
sudo apt install ros-humble-gazebo-ros-pkgs
```

### Launch Files

Create launch files to start Gazebo with your robot:
```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    return LaunchDescription([
        # Launch Gazebo
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                get_package_share_directory('gazebo_ros'),
                '/launch/gazebo.launch.py'
            ]),
        )
    ])
```

## Unity Robotics Setup

### Prerequisites

- Unity Hub
- Unity 2021.3 LTS or later
- Unity Robotics Package (com.unity.robotics.urdf-importer)
- ROS TCP Connector

### Installation Process

1. Install Unity Hub
2. Install Unity 2021.3 LTS
3. Create a new 3D project
4. Add the ROS TCP Connector package via Package Manager
5. Import the URDF Importer package

### Basic Unity Robotics Workflow

1. Import URDF models using the URDF Importer
2. Set up ROS communication using the TCP Connector
3. Create simulation environments
4. Implement robot behaviors

## Isaac Sim Setup

### Prerequisites

- NVIDIA GPU with CUDA support
- Isaac Sim installation from NVIDIA Developer Zone
- Isaac ROS packages

### Basic Usage

Launch Isaac Sim:
```bash
isaac-sim python.sh
```

## Creating Simulation Environments

### World Files (Gazebo)

Create SDF world files to define environments:
```xml
<sdf version="1.7">
  <world name="small_room">
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.1 -0.1 -1</direction>
    </light>

    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

### Unity Scenes

In Unity, create scenes with:
- Terrain for outdoor environments
- 3D models for obstacles
- Lighting settings
- Physics materials

## Simulation Best Practices

### Performance Optimization

- Use simplified collision meshes
- Limit physics update rates where possible
- Use Level of Detail (LOD) for complex models
- Optimize textures and materials

### Accuracy Considerations

- Calibrate physical properties to match real hardware
- Account for sim-to-real differences
- Validate simulation results with physical tests
- Document simulation assumptions and limitations