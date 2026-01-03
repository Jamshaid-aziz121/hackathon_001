# Quickstart Guide: Educational Robotics Book

## Overview

This guide provides a rapid introduction to the Educational Robotics Book project, designed for educators, students, and developers interested in robotics and AI applications in educational environments.

## Prerequisites

Before starting with the book content, ensure you have:

- Basic programming knowledge (Python preferred)
- Understanding of fundamental computer science concepts
- Familiarity with Linux command line (helpful but not required)
- Access to a computer capable of running simulation environments

## Getting Started with ROS 2 (Module 1)

### Install ROS 2 Humble Hawksbill

1. Set up your Ubuntu 22.04 environment:
   ```bash
   sudo apt update
   sudo apt install software-properties-common
   sudo add-apt-repository universe
   ```

2. Add the ROS 2 GPG key and repository:
   ```bash
   sudo apt update && sudo apt install curl gnupg lsb-release
   curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
   ```

3. Install ROS 2 packages:
   ```bash
   sudo apt update
   sudo apt install ros-humble-desktop
   sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
   ```

4. Initialize rosdep:
   ```bash
   sudo rosdep init
   rosdep update
   ```

5. Source the ROS 2 environment:
   ```bash
   source /opt/ros/humble/setup.bash
   ```

### Your First ROS 2 Publisher-Subscriber

1. Create a new workspace:
   ```bash
   mkdir -p ~/ros2_ws/src
   cd ~/ros2_ws
   colcon build
   source install/setup.bash
   ```

2. Create a simple publisher node (publisher_member_function.py):
   ```python
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String

   class MinimalPublisher(Node):
       def __init__(self):
           super().__init__('minimal_publisher')
           self.publisher_ = self.create_publisher(String, 'topic', 10)
           timer_period = 0.5  # seconds
           self.timer = self.create_timer(timer_period, self.timer_callback)
           self.i = 0

       def timer_callback(self):
           msg = String()
           msg.data = f'Hello World: {self.i}'
           self.publisher_.publish(msg)
           self.get_logger().info(f'Publishing: "{msg.data}"')
           self.i += 1

   def main(args=None):
       rclpy.init(args=args)
       minimal_publisher = MinimalPublisher()
       rclpy.spin(minimal_publisher)
       minimal_publisher.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

3. Run the publisher:
   ```bash
   python3 publisher_member_function.py
   ```

4. In another terminal, run the subscriber:
   ```bash
   ros2 run demo_nodes_py listener
   ```

## Getting Started with Simulation (Module 2)

### Install Gazebo Garden

1. Add the Gazebo repository:
   ```bash
   sudo apt install wget
   wget https://packages.osrfoundation.org/gazebo.gpg -O /tmp/gazebo.gpg
   sudo cp /tmp/gazebo.gpg /usr/share/keyrings/
   echo "deb [arch=amd64 signed-by=/usr/share/keyrings/gazebo.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
   sudo apt update
   ```

2. Install Gazebo Garden:
   ```bash
   sudo apt install gz-harmonic
   ```

3. Test the installation:
   ```bash
   gz sim
   ```

## Getting Started with AI Perception (Module 3)

### NVIDIA Isaac Setup

For this module, you'll need access to NVIDIA Isaac tools. You can use NVIDIA Isaac Sim for simulation or the Isaac ROS packages for real hardware integration.

1. Install Isaac ROS Dev Kit (for development):
   - Follow the installation guide at: https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_dev.html

2. Test with a simple perception pipeline:
   ```bash
   # This will be covered in detail in Module 3
   ```

## Getting Started with Vision-Language-Action (Module 4)

### Voice-to-Action Setup

1. Install required Python packages:
   ```bash
   pip3 install openai-whisper
   pip3 install speechrecognition
   pip3 install transformers
   ```

2. Basic voice command recognition:
   ```python
   import speech_recognition as sr

   def recognize_speech():
       r = sr.Recognizer()
       with sr.Microphone() as source:
           print("Say something!")
           audio = r.listen(source)

       try:
           command = r.recognize_google(audio)
           print(f"You said: {command}")
           return command
       except sr.UnknownValueError:
           print("Could not understand audio")
       except sr.RequestError as e:
           print(f"Error: {e}")
   ```

## Running Examples

### From the Examples Directory

1. Navigate to the examples directory:
   ```bash
   cd ~/educational-robotics-book/examples
   ```

2. Each module has its own example directory:
   - `ros2-basics/` - Basic ROS 2 examples
   - `simulation-environments/` - Gazebo and Unity examples
   - `ai-perception/` - Isaac and perception examples
   - `vla-systems/` - Vision-Language-Action examples

3. Follow the README in each directory for specific instructions.

## Validation Checklist

Before proceeding with advanced topics, ensure you have:

- [ ] Successfully installed ROS 2 Humble Hawksbill
- [ ] Run your first publisher-subscriber example
- [ ] Installed and launched Gazebo simulation
- [ ] Verified your Python environment for AI tools
- [ ] Reviewed the project structure and content organization

## Next Steps

1. Start with Module 1: The Robotic Nervous System (ROS 2) for foundational concepts
2. Progress through each module sequentially for optimal learning
3. Complete the mini-projects at the end of each module
4. Refer to the debugging section when encountering issues

## Getting Help

- Check the troubleshooting section of each module
- Review the common failures and solutions
- Consult the citations for additional resources
- Join the educational robotics community forums