# Debugging ROS 2 Applications

## Common Issues and Solutions

### 1. Nodes Not Communicating

**Problem**: Nodes are not receiving messages from topics.

**Solutions**:
- Check that nodes are on the same ROS domain ID:
  ```bash
  echo $ROS_DOMAIN_ID
  ```
- Verify topic names match exactly:
  ```bash
  ros2 topic list
  ros2 topic info <topic_name>
  ```
- Ensure message types match between publisher and subscriber
- Check that nodes are properly initialized and spinning

### 2. Parameter Issues

**Problem**: Parameters are not being set or retrieved correctly.

**Solutions**:
- List all parameters for a node:
  ```bash
  ros2 param list <node_name>
  ```
- Get a specific parameter:
  ```bash
  ros2 param get <node_name> <param_name>
  ```
- Set a parameter:
  ```bash
  ros2 param set <node_name> <param_name> <value>
  ```

### 3. Build Issues

**Problem**: Package fails to build with colcon.

**Solutions**:
- Clean the build directory:
  ```bash
  rm -rf build/ install/ log/
  ```
- Check dependencies in `package.xml`:
  ```xml
  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  ```
- Verify `setup.py` has correct entry points
- Check that all imports are available

### 4. Lifecycle Issues

**Problem**: Nodes are not starting properly or shutting down unexpectedly.

**Solutions**:
- Ensure `rclpy.init()` is called before creating nodes
- Verify the node is being spun with `rclpy.spin()`
- Check for exceptions in callbacks that might cause crashes
- Implement proper cleanup in destructor methods

## Debugging Tools

### 1. ROS 2 Command Line Tools

**ros2 doctor**: Check ROS setup and diagnose issues
```bash
ros2 doctor
```

**ros2 run**: Run a specific node for debugging
```bash
ros2 run <package_name> <executable_name>
```

**ros2 launch**: Launch multiple nodes with debugging
```bash
ros2 launch <package_name> <launch_file> --show-args
```

### 2. Logging

Use different log levels for debugging:

```python
import rclpy
from rclpy.node import Node

class DebugNode(Node):
    def __init__(self):
        super().__init__('debug_node')

        # Different log levels
        self.get_logger().debug('Debug message')
        self.get_logger().info('Info message')
        self.get_logger().warn('Warning message')
        self.get_logger().error('Error message')
        self.get_logger().fatal('Fatal message')
```

Set log level from command line:
```bash
export RCUTILS_LOGGING_SEVERITY_THRESHOLD=DEBUG
```

### 3. Visualization with RViz2

For robotics applications, RViz2 is invaluable:
```bash
ros2 run rviz2 rviz2
```

### 4. Message Inspection

Inspect messages on topics:
```bash
ros2 topic echo <topic_name>
```

With specific message type:
```bash
ros2 topic echo <topic_name> <msg_type>
```

## Best Practices for Debugging

### 1. Use Descriptive Node and Topic Names
```python
# Good
self.node = Node('robot_controller')
self.cmd_vel_pub = self.create_publisher(Twist, '/robot/cmd_vel', 10)

# Avoid generic names
self.node = Node('node1')
self.pub = self.create_publisher(Twist, '/topic', 10)
```

### 2. Implement Health Checks
```python
def check_system_health(self):
    # Check if all required services are available
    if not self.client.service_is_ready():
        self.get_logger().error('Required service not available')
        return False
    return True
```

### 3. Use Configuration Files
Store parameters in YAML files rather than hardcoding:
```yaml
robot_controller:
  ros__parameters:
    linear_speed: 0.2
    angular_speed: 0.5
    safe_distance: 0.5
```

### 4. Add Diagnostic Messages
```python
def publish_diagnostics(self):
    # Publish diagnostic messages for monitoring
    diag_msg = DiagnosticArray()
    # Add diagnostic status
    self.diag_publisher.publish(diag_msg)
```

## Performance Debugging

### 1. Check Message Rates
```bash
# Monitor message rate
ros2 topic hz <topic_name>
```

### 2. Monitor CPU Usage
```bash
# Monitor node resource usage
ros2 run top top
```

### 3. Memory Issues
- Watch for memory leaks in long-running nodes
- Use weak references where appropriate
- Monitor node lifecycle

## Testing Your Debugging Skills

Try these debugging exercises:

1. Create a publisher-subscriber pair where the message types don't match. Observe the error.
2. Set up a node with a parameter that has the wrong type. Debug the issue.
3. Create a node that crashes in a callback. Use logging to identify the issue.
4. Introduce a timing issue by publishing messages too quickly. Use `ros2 topic hz` to diagnose.