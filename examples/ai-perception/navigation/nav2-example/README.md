# Nav2 Navigation Example

## Overview

This example demonstrates autonomous navigation using the Navigation2 (Nav2) stack integrated with AI perception systems. The system combines perception data with navigation capabilities to enable intelligent robot movement in educational environments.

## Components

### 1. Navigation Node
The main component that interfaces with Nav2 and manages navigation tasks:
- Action client for `navigate_to_pose`
- Laser scan processing for obstacle detection
- Safety monitoring during navigation
- Integration with perception systems

### 2. Nav2 Integration
- Path planning and execution
- Local and global costmap management
- Recovery behaviors
- Lifecycle management

## Prerequisites

- ROS 2 Humble Hawksbill
- Navigation2 packages
- Navigation2_msgs
- TF2 libraries
- A robot with odometry and laser scanner

## Installation

1. Install Navigation2:
```bash
sudo apt update
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup
```

2. Install additional dependencies:
```bash
sudo apt install ros-humble-nav2-msgs ros-humble-tf2-geometry-msgs
```

## Usage

### Running the Navigation Node

```bash
# Source your ROS workspace
source install/setup.bash

# Run the navigation node
ros2 run nav2_example navigation_node
```

### Launching with Nav2 Stack

```bash
# Launch Nav2 with your robot configuration
ros2 launch nav2_bringup navigation_launch.py use_sim_time:=false

# Then run the navigation node
ros2 run nav2_example navigation_node
```

### Setting Navigation Goals

The navigation node provides several ways to set goals:

```bash
# Using the node's method
ros2 run nav2_example navigation_node  # Sets a default goal

# Using RViz2
ros2 run rviz2 rviz2  # Set 2D Pose Estimate and 2D Nav Goal

# Using command line
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose "{pose: {header: {frame_id: 'map'}, pose: {position: {x: 1.0, y: 1.0, z: 0.0}, orientation: {z: 0.0, w: 1.0}}}}"
```

## Configuration

### Parameters

The navigation node uses several parameters:

- `safety_distance`: Minimum distance to obstacles before pausing navigation
- `goal_tolerance`: Distance tolerance for reaching goals
- `use_sim_time`: Whether to use simulation time

### Costmap Configuration

For proper integration with perception, configure your costmaps:

```yaml
# local_costmap_params.yaml
local_costmap:
  global_frame: odom
  robot_base_frame: base_link
  update_frequency: 5.0
  publish_frequency: 2.0
  static_map: false
  rolling_window: true
  width: 3
  height: 3
  resolution: 0.05
  observation_sources: laser_scan_sensor
  laser_scan_sensor: {sensor_frame: laser_frame, topic: scan, observation_persistence: 0.0, max_obstacle_height: 2.0, min_obstacle_height: 0.0, obstacle_range: 2.5, raytrace_range: 3.0, clearing: true, marking: true}
```

## Topics

### Subscribed Topics
- `/scan` - Laser scan data for obstacle detection
- `/map` - Static map for global planning
- `/odom` - Robot odometry for localization
- `/tf` - Transform tree for coordinate frames

### Published Topics
- `/navigation_status` - Current navigation status
- `/cmd_vel` - Velocity commands (published by Nav2)

### Actions
- `/navigate_to_pose` - Navigation goal action (used by action client)

## Educational Applications

### Learning Objectives
- Navigation2 architecture and components
- Path planning algorithms (A*, Dijkstra, etc.)
- Costmap management and obstacle avoidance
- Robot safety and emergency procedures
- Integration of perception and navigation

### Classroom Activities
- Compare different path planners
- Study costmap inflation parameters
- Explore recovery behaviors
- Implement perception-based navigation goals

## Integration with Perception

The navigation system can be integrated with perception systems:

1. **Dynamic Obstacle Avoidance**: Perception detects dynamic obstacles and updates costmaps
2. **Goal Selection**: Perception identifies interesting locations and sets navigation goals
3. **Safe Navigation**: Perception ensures safe passage for educational environments

### Example Integration

```python
# In a perception-aware navigation system
def perception_callback(self, detection_msg):
    """Process perception results to update navigation"""
    for detection in detection_msg.detections:
        if detection.class_id == 'student':
            # Update costmap to be more cautious around students
            self.update_costmap_for_pedestrian(detection)
        elif detection.class_id == 'obstacle':
            # Add obstacle to local costmap
            self.add_obstacle_to_costmap(detection)
```

## Performance Considerations

### Hardware Requirements
- Adequate CPU for path planning algorithms
- Sufficient memory for costmap representation
- Real-time capable system for safety

### Tuning Parameters
- Path planner frequency
- Costmap resolution
- Robot footprint
- Velocity limits

## Troubleshooting

### Common Issues

1. **Navigation Fails to Start**
   - Check that Nav2 stack is running
   - Verify all required transforms are available
   - Ensure proper costmap configuration

2. **Robot Gets Stuck**
   - Adjust inflation radius in costmap
   - Check robot footprint configuration
   - Verify sensor data quality

3. **Poor Path Quality**
   - Tune path planner parameters
   - Adjust costmap resolution
   - Check for TF timing issues

### Debugging Commands

```bash
# Check if navigation action server is available
ros2 action list

# Monitor navigation status
ros2 topic echo /navigation_status

# Visualize in RViz2
ros2 run rviz2 rviz2

# Check transforms
ros2 run tf2_tools view_frames
```

## Extensions

### Advanced Features
- Multi-floor navigation
- Socially-aware navigation
- Formation navigation
- Learning-based navigation

### Perception Integration
- Semantic costmaps
- Human-aware navigation
- Object-aware path planning
- Vision-based localization

## References

- [Navigation2 Documentation](https://navigation.ros.org/)
- [Nav2 Tutorials](https://navigation.ros.org/tutorials/)
- [Costmap 2D Documentation](http://wiki.ros.org/costmap_2d)

## Next Steps

- Integrate with Isaac Sim for simulation
- Add learning capabilities to navigation
- Implement educational-specific navigation behaviors
- Create curriculum around autonomous navigation