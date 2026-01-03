# Unity Scene Example for Educational Robotics

## Overview

This example demonstrates how to create a basic Unity scene that can be integrated with ROS for educational robotics applications. The scene includes a simple robot model and setup instructions for ROS communication.

## Prerequisites

- Unity Hub and Unity 2021.3 LTS or later
- Unity Robotics Hub package
- ROS 2 (Humble Hawksbill recommended)
- ROS-TCP-Connector package

## Setup Instructions

### 1. Installing Unity Robotics Hub

1. Open Unity Hub and create a new 3D project
2. In the Unity Package Manager (Window > Package Manager), click the "+" button and select "Add package from git URL..."
3. Add the following packages:
   - `com.unity.robotics.ros-tcp-connector` - For ROS communication
   - `com.unity.robotics.urdf-importer` - For importing robot models

### 2. Creating the Scene

Create a basic scene with the following components:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using ROS2;

public class RobotController : MonoBehaviour
{
    [SerializeField]
    private float linearVelocity = 1.0f;
    [SerializeField]
    private float angularVelocity = 1.0f;

    private ROSConnection ros;
    private float cmdVelX = 0.0f;
    private float cmdVelZ = 0.0f;

    void Start()
    {
        // Get the ROS connection static instance
        ros = ROSConnection.instance;

        // Start the ROS publisher
        ros.RegisterPublisher<Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry_msgs.Twist>("/cmd_vel");
    }

    void Update()
    {
        // Create and publish the message
        var twist = new Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry_msgs.Twist();
        twist.linear = new Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry_msgs.Vector3(cmdVelX, 0, 0);
        twist.angular = new Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry_msgs.Vector3(0, 0, cmdVelZ);

        ros.Publish("/cmd_vel", twist);
    }

    void OnGUI()
    {
        // Simple UI for controlling the robot
        GUIStyle style = new GUIStyle();
        style.fontSize = 24;
        style.normal.textColor = Color.white;

        GUILayout.BeginArea(new Rect(10, 10, 300, 300));

        if (GUILayout.Button("Move Forward", GUILayout.Height(50)))
        {
            cmdVelX = linearVelocity;
            cmdVelZ = 0;
        }

        GUILayout.BeginHorizontal();
        if (GUILayout.Button("Turn Left", GUILayout.Height(50)))
        {
            cmdVelX = 0;
            cmdVelZ = angularVelocity;
        }

        if (GUILayout.Button("Turn Right", GUILayout.Height(50)))
        {
            cmdVelX = 0;
            cmdVelZ = -angularVelocity;
        }
        GUILayout.EndHorizontal();

        if (GUILayout.Button("Move Backward", GUILayout.Height(50)))
        {
            cmdVelX = -linearVelocity;
            cmdVelZ = 0;
        }

        if (GUILayout.Button("Stop", GUILayout.Height(50)))
        {
            cmdVelX = 0;
            cmdVelZ = 0;
        }

        GUILayout.EndArea();
    }
}
```

### 3. Creating a Basic Robot Model

You can create a simple robot model in Unity using basic primitives:

1. Create an empty GameObject and name it "Robot"
2. Add a Capsule as the main body (rename to "Chassis")
3. Add 4 Cylinders as wheels (rename to "Wheel_FL", "Wheel_FR", "Wheel_BL", "Wheel_BR")
4. Position the wheels at the corners of the chassis
5. Add the RobotController script to the Robot GameObject

### 4. Setting up ROS Communication

1. In the Unity scene, create an empty GameObject named "ROSConnection"
2. Add the "ROS TCP Connection" component to this object
3. Configure the IP address and port to match your ROS setup
4. The default is usually localhost:10000

### 5. Scene Structure

A basic scene structure would look like:

```
Scene
├── ROSConnection (with ROS TCP Connection component)
├── Robot
│   ├── Chassis (Capsule collider and mesh)
│   ├── Wheel_FL (Cylinder collider and mesh)
│   ├── Wheel_FR (Cylinder collider and mesh)
│   ├── Wheel_BL (Cylinder collider and mesh)
│   ├── Wheel_BR (Cylinder collider and mesh)
│   └── RobotController (script component)
├── Main Camera
├── Directional Light
└── Plane (as ground)
```

### 6. Testing the Setup

1. Start your ROS environment
2. Launch the scene in Unity
3. Use the on-screen controls to move the robot
4. Verify that ROS topics are being published by running:
   ```bash
   ros2 topic echo /cmd_vel
   ```

## Educational Applications

This Unity scene can be extended for various educational purposes:

- Path planning visualization
- Sensor simulation (camera, LIDAR)
- Multi-robot coordination
- Human-robot interaction studies
- Robotics algorithm testing

## Troubleshooting

### Common Issues:

1. **Connection Issues**: Ensure ROS and Unity are on the same network and ports are open
2. **Performance**: Reduce scene complexity if Unity runs slowly
3. **Coordinate Systems**: Be aware of differences between Unity (left-handed) and ROS (right-handed) coordinate systems

## Next Steps

- Add more complex robot models
- Implement sensor simulation
- Create educational scenarios and challenges
- Integrate with real robot hardware