# Cross-Module Reference Guide

## Overview

This reference guide provides a comprehensive overview of how different modules in the educational robotics system interconnect and reference each other. It serves as a quick reference for understanding the relationships between ROS 2 fundamentals, simulation environments, AI perception, and vision-language-action systems.

## Module Interconnections

### ROS 2 (Module 1) ↔ Simulation (Module 2)

The ROS 2 system communicates with simulation environments through standard interfaces:

#### Communication Interfaces
- **Topics**: `/scan`, `/camera/image_raw`, `/tf`, `/odom`
- **Services**: `/spawn_entity`, `/reset_simulation`
- **Actions**: Navigation goals, trajectory execution

#### Example Integration
```python
# Publishing to simulation from ROS 2
publisher = node.create_publisher(LaserScan, '/scan', 10)

# Subscribing to simulation data
subscriber = node.create_subscription(Odometry, '/odom', callback, 10)

# Using Gazebo plugins
<plugin filename="libgazebo_ros_diff_drive.so" name="diff_drive">
  <command_topic>cmd_vel</command_topic>
  <odometry_topic>odom</odometry_topic>
</plugin>
```

### ROS 2 (Module 1) ↔ AI Perception (Module 3)

ROS 2 nodes communicate with AI perception systems:

#### Perception Topics
- `/camera/image_raw` → Image processing
- `/object_detections` → Detection results
- `/tracked_objects` → Tracking results
- `/semantic_map` → Semantic understanding

#### Example Integration
```python
# Publishing camera images for perception
image_publisher = node.create_publisher(Image, '/camera/image_raw', 10)

# Subscribing to perception results
detection_subscriber = node.create_subscription(
    Detection2DArray, '/object_detections', detection_callback, 10)
```

### Simulation (Module 2) ↔ AI Perception (Module 3)

Simulation environments provide data for AI perception:

#### Sensor Simulation
- **Camera Simulation**: RGB, depth, thermal sensors
- **LiDAR Simulation**: 2D/3D LiDAR with realistic noise
- **IMU Simulation**: Inertial measurement units
- **GPS Simulation**: Position and orientation data

#### Example Configuration
```xml
<!-- Gazebo camera sensor -->
<sensor name="camera1" type="camera">
  <camera name="head">
    <horizontal_fov>1.3962634</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
    </image>
  </camera>
  <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
    <topic_name>image_raw</topic_name>
  </plugin>
</sensor>
```

### AI Perception (Module 3) ↔ VLA Systems (Module 4)

AI perception feeds into vision-language-action systems:

#### Perception-to-Action Pipeline
1. **Object Detection** → Identify objects for interaction
2. **Scene Understanding** → Context for language interpretation
3. **Tracking** → Continuous monitoring for action planning
4. **Classification** → Object properties for manipulation

#### Example Integration
```python
# Perception results feed into VLA system
def detection_callback(self, msg):
    # Process detections
    for detection in msg.detections:
        # Integrate with language understanding
        self.process_vla_integration(detection)

def process_vla_integration(self, detection):
    # Combine vision with language and action
    object_label = detection.results[0].hypothesis.class_id
    action = self.plan_action_for_object(object_label)
    self.execute_action(action)
```

## Cross-Module Architecture Patterns

### 1. Publisher-Subscriber Pattern

Used across all modules for loose coupling:

```python
# Module 1 (ROS 2) publishes sensor data
sensor_publisher = node.create_publisher(SensorMsg, '/sensor_data', 10)

# Module 2 (Simulation) subscribes to control commands
cmd_subscriber = node.create_subscription(CmdMsg, '/cmd_vel', cmd_callback, 10)

# Module 3 (AI Perception) processes sensor data
sensor_subscriber = node.create_subscription(SensorMsg, '/sensor_data', process_sensor_data, 10)

# Module 4 (VLA) acts on perception results
perception_subscriber = node.create_subscription(PerceptionMsg, '/perception_results', vla_callback, 10)
```

### 2. Client-Server Pattern

Used for service-oriented communication:

```python
# Service server (any module)
service = node.create_service(RequestType, '/service_name', service_callback)

# Service client (any module)
client = node.create_client(RequestType, '/service_name')
```

### 3. Action Pattern

Used for goal-oriented tasks:

```python
# Action server (navigation, manipulation)
action_server = ActionServer(node, ActionName, '/action_name', execute_callback)

# Action client (requesting navigation, manipulation)
action_client = ActionClient(node, ActionName, '/action_name')
```

## Common Message Types Across Modules

### Sensor Messages
- `sensor_msgs/Image` - Camera images (all modules)
- `sensor_msgs/LaserScan` - LiDAR data (simulation → perception → navigation)
- `sensor_msgs/PointCloud2` - 3D point clouds (perception → navigation)
- `sensor_msgs/Imu` - Inertial data (simulation → navigation)

### Navigation Messages
- `nav_msgs/Odometry` - Robot position (simulation → navigation)
- `nav_msgs/Path` - Planned paths (navigation → action)
- `geometry_msgs/PoseStamped` - Goal positions (VLA → navigation)

### Perception Messages
- `vision_msgs/Detection2DArray` - Object detections (perception → VLA)
- `vision_msgs/TrackedObjectArray` - Tracked objects (perception → VLA)
- `std_msgs/String` - Classification results (perception → VLA)

## Cross-Module Configuration

### Parameter Sharing

Parameters can be shared across modules:

```yaml
# Shared parameters across modules
robot_description: &robot_description
  model: "turtlebot3_waffle"
  namespace: "robot1"

simulation_params: &simulation_params
  use_sim_time: true
  physics_engine: "ode"

perception_params: &perception_params
  confidence_threshold: 0.5
  max_objects: 10

vla_params: &vla_params
  command_timeout: 5.0
  safety_distance: 0.5
```

### Launch File Integration

Launch files can coordinate multiple modules:

```python
def generate_launch_description():
    # Launch nodes from different modules
    ros_nodes = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([FindPackageShare('ros_module'), '/launch/nodes.launch.py'])
    )

    simulation_nodes = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([FindPackageShare('simulation_module'), '/launch/sim.launch.py'])
    )

    perception_nodes = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([FindPackageShare('perception_module'), '/launch/perception.launch.py'])
    )

    vla_nodes = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([FindPackageShare('vla_module'), '/launch/vla.launch.py'])
    )

    return LaunchDescription([
        ros_nodes,
        simulation_nodes,
        perception_nodes,
        vla_nodes
    ])
```

## Cross-Module Design Patterns

### 1. Observer Pattern

Monitor changes across modules:

```python
class CrossModuleObserver:
    def __init__(self):
        # Subscribe to relevant topics from all modules
        self.ros_subscriber = self.create_subscription(
            ROSStatus, '/ros_status', self.ros_status_callback, 10)

        self.sim_subscriber = self.create_subscription(
            SimStatus, '/sim_status', self.sim_status_callback, 10)

        self.perception_subscriber = self.create_subscription(
            PerceptionStatus, '/perception_status', self.perception_status_callback, 10)

        self.vla_subscriber = self.create_subscription(
            VLAStatus, '/vla_status', self.vla_status_callback, 10)

    def ros_status_callback(self, msg):
        self.update_system_state('ros', msg)

    def sim_status_callback(self, msg):
        self.update_system_state('simulation', msg)

    def perception_status_callback(self, msg):
        self.update_system_state('perception', msg)

    def vla_status_callback(self, msg):
        self.update_system_state('vla', msg)

    def update_system_state(self, module, status):
        # Update cross-module state
        self.system_state[module] = status
        self.check_cross_module_conditions()
```

### 2. Mediator Pattern

Coordinate interactions between modules:

```python
class SystemMediator:
    def __init__(self):
        # Initialize connections to all modules
        self.ros_connector = ROSConnector()
        self.sim_connector = SimConnector()
        self.perception_connector = PerceptionConnector()
        self.vla_connector = VLAConnector()

    def coordinate_action(self, request):
        # Coordinate action across modules
        sim_ok = self.sim_connector.check_environment(request)
        perception_ok = self.perception_connector.analyze_scene(request)
        ros_ok = self.ros_connector.verify_capabilities(request)

        if all([sim_ok, perception_ok, ros_ok]):
            # Execute coordinated action
            self.vla_connector.execute_request(request)
            return True
        else:
            return False
```

## Performance Considerations

### Message Throughput
- **High-frequency topics**: `/camera/image_raw`, `/scan` - 10-30 Hz
- **Medium-frequency topics**: `/odom`, `/tf` - 10-50 Hz
- **Low-frequency topics**: `/map`, `/goal_status` - 1-5 Hz

### Memory Usage
- **Simulation**: High memory for physics calculations
- **Perception**: High GPU memory for neural networks
- **VLA**: Moderate memory for action planning
- **ROS**: Low memory for message passing

### Processing Latency
- **Sensor processing**: < 50ms
- **Perception**: < 100ms (depending on model size)
- **Action planning**: < 200ms
- **System integration**: < 500ms

## Debugging Cross-Module Issues

### Common Issues
1. **Timing mismatches**: Different modules operating at different frequencies
2. **Data format incompatibilities**: Messages not matching expected formats
3. **Resource conflicts**: Multiple modules competing for resources
4. **Synchronization problems**: Events not properly coordinated

### Debugging Commands
```bash
# Monitor all topics across modules
ros2 topic list

# Check message rates
ros2 topic hz /topic_name

# Echo messages to verify content
ros2 topic echo /topic_name

# Check node connections
ros2 run rqt_graph rqt_graph

# Monitor system resources
ros2 run top top
```

## Best Practices for Cross-Module Development

### 1. Standardized Interfaces
- Use common message types
- Follow naming conventions
- Maintain API compatibility

### 2. Loose Coupling
- Minimize direct dependencies
- Use message passing instead of direct calls
- Implement proper abstraction layers

### 3. Error Handling
- Handle module failures gracefully
- Implement fallback mechanisms
- Provide meaningful error messages

### 4. Testing
- Test modules in isolation
- Test cross-module interactions
- Verify system integration

This cross-module reference guide provides a comprehensive overview of how the different educational robotics modules work together, enabling developers to understand the interconnections and build integrated systems effectively.