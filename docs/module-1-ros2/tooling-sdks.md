# ROS 2 Tooling and SDKs

## Development Tools

### ROS 2 Command Line Tools

ROS 2 provides a comprehensive set of command-line tools for managing nodes, topics, services, and other ROS graph elements:

```bash
# List all available topics
ros2 topic list

# Echo messages from a topic
ros2 topic echo <topic_name> <msg_type>

# List all active nodes
ros2 node list

# Run a node
ros2 run <package_name> <executable_name>

# Launch multiple nodes
ros2 launch <package_name> <launch_file>.py
```

### Creating Packages

Use the `ros2 pkg create` command to create new packages:

```bash
ros2 pkg create --build-type ament_python <package_name>
```

For C++ packages:
```bash
ros2 pkg create --build-type ament_cmake <package_name>
```

## Development Environment

### Workspaces

ROS 2 uses workspaces to organize your development:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
colcon build
source install/setup.bash
```

### Build Systems

- `ament_cmake`: For C++ packages using CMake
- `ament_python`: For Python packages
- `ament_gradle`: For Java packages

## IDE Support

### Visual Studio Code
- ROS2 Extension Pack
- Syntax highlighting for ROS 2 packages
- Debugging support

### Eclipse
- ROS Development Tools
- Project management for ROS packages