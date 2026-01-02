# Unity Scene Setup for ROS Integration

## Scene File Structure

This document outlines the structure and components of the Unity scene for educational robotics simulation.

## Scene Components

### 1. ROS Connection Manager

GameObject: `ROSConnectionManager`
- Component: `ROSConnection`
- Configuration:
  - IP Address: 127.0.0.1 (localhost)
  - Port: 10000
  - Reconnect Interval: 2 seconds

### 2. Robot Model

GameObject: `EducationalRobot`
- Components:
  - `RobotController` (custom script)
  - `Rigidbody` (for physics simulation)
  - `MeshRenderer` (for visualization)

#### Robot Sub-components:
- `Chassis`: Main body of the robot
- `Wheel_FL`: Front left wheel
- `Wheel_FR`: Front right wheel
- `Wheel_BL`: Back left wheel
- `Wheel_BR`: Back right wheel

### 3. Environment Objects

- `GroundPlane`: Flat surface for the robot to move on
- `Main Camera`: Scene camera with follow script
- `Directional Light`: Basic lighting

## ROS Topic Integration

### Publishers:
- `/cmd_vel` - Twist messages for robot movement
- `/odom` - Odometry data
- `/tf` - Transform data

### Subscribers:
- `/joint_states` - Joint position data
- `/scan` - LIDAR simulation data (if added)
- `/image_raw` - Camera simulation data (if added)

## Scene Settings

### Physics Settings:
- Gravity: (0, -9.81, 0)
- Fixed Timestep: 0.02
- Maximum Allowed Timestep: 0.3333

### Time Settings:
- Time Scale: 1.0
- Maximum Particle Timestep: 0.03

## Robot Controller Script

The RobotController script handles ROS communication:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry_msgs;

public class RobotController : MonoBehaviour
{
    [Header("Robot Configuration")]
    [SerializeField] private float wheelRadius = 0.1f;
    [SerializeField] private float wheelSeparation = 0.4f;

    [Header("Movement Settings")]
    [SerializeField] private float maxLinearVelocity = 1.0f;
    [SerializeField] private float maxAngularVelocity = 1.0f;

    private ROSConnection ros;
    private Rigidbody robotRigidbody;

    void Start()
    {
        ros = ROSConnection.instance;
        robotRigidbody = GetComponent<Rigidbody>();

        // Start publishing to ROS
        ros.RegisterPublisher<Twist>("/cmd_vel");

        // Subscribe to ROS topics if needed
        ros.Subscribe<JointState>("/joint_states", OnJointStateReceived);
    }

    void Update()
    {
        // Example: Publish robot velocity
        PublishVelocity();
    }

    private void PublishVelocity()
    {
        // Create twist message
        var twist = new Twist();
        twist.linear.x = robotRigidbody.velocity.x;
        twist.linear.y = robotRigidbody.velocity.y;
        twist.angular.z = transform.angularVelocity.z;  // Simplified

        ros.Publish("/cmd_vel", twist);
    }

    private void OnJointStateReceived(JointState jointState)
    {
        // Process joint state data
        // Update wheel positions based on joint angles
    }
}
```

## Camera Setup

GameObject: `Main Camera`
- Component: `Camera`
- Component: `RobotCameraFollower` (custom script)

### Camera Follower Script:
```csharp
using UnityEngine;

public class RobotCameraFollower : MonoBehaviour
{
    [SerializeField] private Transform target;
    [SerializeField] private Vector3 offset = new Vector3(-3, 3, -3);
    [SerializeField] private float smoothSpeed = 0.125f;

    void LateUpdate()
    {
        if (target != null)
        {
            Vector3 desiredPosition = target.position + offset;
            Vector3 smoothedPosition = Vector3.Lerp(transform.position, desiredPosition, smoothSpeed);
            transform.position = smoothedPosition;

            transform.LookAt(target);
        }
    }
}
```

## Environment Configuration

### Ground Plane:
- Size: 20x20 units
- Material: Basic ground material
- Physics: Static collider

### Lighting:
- Type: Directional
- Intensity: 1.0
- Color: White (1, 1, 1)
- Rotation: (50, -30, 0)

## Scene Usage

1. **Start ROS Bridge**: Ensure ROS bridge is running
2. **Load Scene**: Open the Unity scene in the editor
3. **Configure Connection**: Verify ROS connection settings
4. **Play Scene**: Press Play button to start simulation
5. **Control Robot**: Use external ROS nodes to send commands

## Educational Scenarios

### Basic Navigation:
- Create waypoints for the robot to navigate to
- Implement simple path planning algorithms
- Visualize planned vs. actual paths

### Sensor Simulation:
- Add simulated LIDAR sensor
- Implement camera feed simulation
- Create sensor fusion examples

### Multi-Robot:
- Add multiple robot instances
- Implement coordination algorithms
- Demonstrate swarm behavior

## Performance Considerations

- Keep scene complexity moderate for real-time performance
- Use efficient collision meshes
- Optimize rendering settings for simulation
- Monitor frame rate for consistent physics simulation

## Extension Points

- Add articulated robot arms
- Implement more complex sensors
- Create interactive environments
- Add AI/ML components for learning