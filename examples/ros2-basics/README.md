# ROS 2 Basics Examples

This directory contains basic ROS 2 examples to help you learn fundamental concepts.

## Publisher-Subscriber Example

The publisher-subscriber example demonstrates the basic communication pattern in ROS 2.

### Running the Example

1. **Terminal 1 - Start the publisher**:
   ```bash
   ros2 run ros2_basics simple_publisher
   ```

2. **Terminal 2 - Start the subscriber**:
   ```bash
   ros2 run ros2_basics simple_subscriber
   ```

The publisher will send "Hello World" messages every 0.5 seconds, and the subscriber will log them.

## Services Example

The services example demonstrates request-response communication in ROS 2.

### Running the Example

1. **Terminal 1 - Start the service server**:
   ```bash
   ros2 run ros2_basics add_two_ints_server
   ```

2. **Terminal 2 - Call the service**:
   ```bash
   ros2 run ros2_basics add_two_ints_client 1 2
   ```

The server will add the two numbers (1 + 2 = 3) and return the result.

### Testing with Command Line Tools

You can also test the service using ROS 2 command line tools:

```bash
# List available services
ros2 service list

# Call the service directly
ros2 service call /add_two_ints example_interfaces/srv/AddTwoInts "{a: 5, b: 3}"

# Check service type
ros2 service type /add_two_ints
```

## Understanding the Code

### Publisher-Subscriber Pattern

- **Publisher**: Creates messages and sends them to a topic
- **Subscriber**: Listens to a topic and processes incoming messages
- **Topic**: Named bus for message exchange
- **Message**: Data structure passed between nodes

### Service Pattern

- **Service Server**: Provides functionality and responds to requests
- **Service Client**: Makes requests and receives responses
- **Service**: Named endpoint for request-response communication
- **Request/Response**: Structured data for the service interaction

## Educational Value

These examples help students understand:

1. **Node Creation**: How to create ROS 2 nodes
2. **Communication Patterns**: Publisher-subscriber and service patterns
3. **Message Types**: How to use standard message types
4. **Logging**: How to log information for debugging
5. **Resource Management**: Proper cleanup of resources

## Troubleshooting

### Common Issues

1. **Node Not Found**:
   - Ensure you've sourced your workspace: `source install/setup.bash`
   - Check that the package is built: `colcon build`

2. **Topic Not Connecting**:
   - Verify topic names match exactly
   - Check that both publisher and subscriber are running

3. **Service Not Available**:
   - Ensure the service server is running before calling the service
   - Check service name matches exactly

### Useful Commands

```bash
# List all nodes
ros2 node list

# List all topics
ros2 topic list

# Echo messages on a topic
ros2 topic echo /chatter std_msgs/msg/String

# Get information about a node
ros2 node info <node_name>
```

## Next Steps

After mastering these basic examples, try:

1. Creating your own message types
2. Adding parameters to your nodes
3. Using actions for long-running tasks
4. Implementing more complex communication patterns