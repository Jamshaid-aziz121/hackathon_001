# Debugging Simulation Environments

## Common Simulation Issues and Solutions

### 1. Robot Not Moving in Simulation

**Problem**: Robot appears in Gazebo but doesn't respond to velocity commands.

**Solutions**:
- Check that the Gazebo plugin is correctly configured in your URDF:
  ```xml
  <gazebo>
    <plugin filename="libgazebo_ros_diff_drive.so" name="diff_drive">
      <!-- Make sure joint names match your URDF -->
      <left_joint>left_wheel_joint</left_joint>
      <right_joint>right_wheel_joint</right_joint>
      <!-- Check topic names match -->
      <command_topic>cmd_vel</command_topic>
    </plugin>
  </gazebo>
  ```
- Verify the robot is properly spawned:
  ```bash
  ros2 topic list | grep cmd_vel
  ros2 service call /spawn_entity gazebo_msgs/srv/SpawnEntity "{name: 'robot', xml: '<robot>...'}"
  ```
- Check for physics simulation issues:
  ```bash
  gz topic -t /world/default/model/robot_name/joint_state -e
  ```

### 2. TF Transform Issues

**Problem**: Robot frames are not properly connected or transforms are missing.

**Solutions**:
- Check the robot state publisher is running:
  ```bash
  ros2 run robot_state_publisher robot_state_publisher --ros-args --log-level debug
  ```
- Verify joint states are being published:
  ```bash
  ros2 topic echo /joint_states
  ```
- Use tf2 tools to check transforms:
  ```bash
  ros2 run tf2_tools view_frames
  ros2 run tf2_ros tf2_echo base_link odom
  ```

### 3. Sensor Data Problems

**Problem**: Sensors are not publishing data or data is incorrect.

**Solutions**:
- Verify sensor plugins are correctly configured in URDF:
  ```xml
  <gazebo reference="camera">
    <sensor name="camera1" type="camera">
      <camera name="head">
        <horizontal_fov>1.3962634</horizontal_fov>
        <image>
          <width>800</width>
          <height>600</height>
        </image>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <frame_name>camera</frame_name>
        <topic_name>image_raw</topic_name>
      </plugin>
    </sensor>
  </gazebo>
  ```
- Check sensor topics:
  ```bash
  ros2 topic list | grep -E "(camera|scan|imu)"
  ros2 topic echo /camera/image_raw --field data
  ```

### 4. Timing and Synchronization Issues

**Problem**: Simulation time doesn't match ROS time or controllers run at wrong frequency.

**Solutions**:
- Check if using simulation time:
  ```python
  # In your node
  use_sim_time = self.get_parameter('use_sim_time').value
  ```
- Verify the clock publisher:
  ```bash
  ros2 topic echo /clock
  ```
- Set simulation to real-time mode for debugging:
  ```bash
  gz world -i  # Interactive mode
  ```

## Debugging Tools for Simulation

### 1. Gazebo Tools

**Gazebo GUI**:
```bash
gz sim  # Launch Gazebo GUI
```

**Gazebo Command Line Tools**:
```bash
# List all topics
gz topic -l

# Echo a topic
gz topic -t /world/default/model/robot_name/pose -e

# Check model properties
gz model -m robot_name -i
```

### 2. ROS 2 Simulation Tools

**Check Simulation State**:
```bash
# List all nodes
ros2 node list

# Check topics
ros2 topic list

# Monitor message rates
ros2 topic hz /cmd_vel
```

**RViz2 for Visualization**:
```bash
rviz2

# In RViz2, add displays for:
# - RobotModel (to see URDF)
# - TF (to see transforms)
# - LaserScan (for LiDAR data)
# - Image (for camera data)
# - Pose (for robot position)
```

### 3. Logging and Monitoring

**Enable Debug Logging**:
```bash
# Set logging level
export RCUTILS_LOGGING_SEVERITY_THRESHOLD=DEBUG

# Or for a specific node
ros2 run package_name node_name --ros-args --log-level debug
```

**Monitor Resource Usage**:
```bash
# Monitor nodes
ros2 run top top

# Check CPU/memory of specific processes
htop  # Look for gzserver and gzclient processes
```

## Simulation-Specific Debugging Techniques

### 1. Physics Debugging

**Check Physics Properties**:
- Verify inertial properties in URDF are realistic
- Check collision meshes match visual meshes
- Ensure mass and inertia values are appropriate

**Example of proper inertial definition**:
```xml
<inertial>
  <mass value="1.0"/>
  <inertia ixx="0.01" ixy="0.0" ixz="0.0"
           iyy="0.01" iyz="0.0" izz="0.02"/>
</inertial>
```

### 2. Sensor Debugging

**Validate Sensor Data**:
```python
# In your sensor processing node
def sensor_callback(self, msg):
    # Add debugging
    self.get_logger().debug(f"Sensor data: {len(msg.ranges) if hasattr(msg, 'ranges') else 'N/A'} values")
    if hasattr(msg, 'ranges'):
        valid_ranges = [r for r in msg.ranges if not math.isnan(r) and r > 0]
        self.get_logger().debug(f"Valid sensor readings: {len(valid_ranges)}")
```

### 3. Multi-Robot Debugging

**Namespace Issues**:
- Ensure each robot has a unique namespace
- Verify topic names are properly namespaced
- Check TF frames include robot names

**Example launch file for multiple robots**:
```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # Robot 1
        Node(
            package='package_name',
            executable='node_name',
            name='controller_robot1',
            namespace='robot1',
            parameters=[{'use_sim_time': True}]
        ),
        # Robot 2
        Node(
            package='package_name',
            executable='node_name',
            name='controller_robot2',
            namespace='robot2',
            parameters=[{'use_sim_time': True}]
        )
    ])
```

## Performance Debugging

### 1. Simulation Speed Issues

**Problem**: Simulation runs too slow or too fast.

**Solutions**:
- Adjust real-time update rate in world file:
  ```xml
  <physics type="ode">
    <real_time_update_rate>1000</real_time_update_rate>
    <max_step_size>0.001</max_step_size>
  </physics>
  ```
- Reduce visual complexity
- Simplify collision meshes
- Limit sensor update rates

### 2. Memory Usage

Monitor memory usage of simulation processes:
```bash
# Check memory usage
ps aux | grep gz
# Monitor over time
watch -n 1 'ps aux | grep gz'
```

## Best Practices for Simulation Debugging

### 1. Incremental Testing

- Start with a simple robot model
- Add sensors one by one
- Test each component separately
- Use simple worlds before complex ones

### 2. Logging Strategy

- Use appropriate log levels
- Log sensor data ranges during startup
- Record robot states during operation
- Log error conditions with context

### 3. Visualization

- Use RViz2 to visualize sensor data
- Display TF trees to verify transforms
- Use Gazebo's built-in visualization tools
- Create custom markers for debugging

### 4. Configuration Management

- Keep separate config files for different debugging scenarios
- Use launch file parameters for easy configuration changes
- Version control your world and robot files

## Debugging Checklist

Before running complex simulations, verify:

- [ ] Robot model loads without errors
- [ ] All joints are properly defined
- [ ] Sensors are publishing data
- [ ] TF tree is complete and correct
- [ ] Control topics are connected
- [ ] Physics properties are realistic
- [ ] Simulation time is synchronized
- [ ] No namespace conflicts in multi-robot scenarios

## Common Error Messages and Solutions

### "Joint state has NaN values"
- Check URDF joint limits and initial positions
- Verify joint state publisher configuration

### "Controller failed to update"
- Check control loop frequency
- Verify command message format
- Check for timing issues

### "Transform timeout"
- Verify tf2 broadcaster is running
- Check for circular dependencies in TF tree
- Ensure transforms are being published at sufficient rate